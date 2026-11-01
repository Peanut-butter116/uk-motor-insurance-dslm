"""Colab CPU adapter v1 for the exact supported D1000 evaluation source.

No research code is changed on disk. A narrowly checked restore adapter removes
Kaggle Secrets loading and the read-only input path; credentials use Colab userdata.
"""
import ast
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import uuid

STORE = 'scarletthe0116/uk-motor-d1000-models-results'
KERNEL = 'scarletthe0116/uk-motor-d1000-train-eval'
SOURCE_SHA = '35e54432a06d06285699d8e9096cabd1bf2811693515934a092b1ea2c09d7133'
ARCHIVE_SHA = 'aa9e5eba75d908c13da03c0c64ce47c55794e9aaeb74d16a25312fd90f86a604'
INPUT_ROOT = Path('/content/d1000-private-input')
SCRATCH = 'scarletthe0116/uk-motor-d1000-recovery-scratch-v1'
LEASE = 'scarletthe0116/uk-motor-d1000-evaluation-lease-v1'
TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}
CREDENTIAL_PROLOGUE = '''    from kaggle import api
    from kaggle_secrets import UserSecretsClient
    import os
    secrets=UserSecretsClient()
    for key in ('KAGGLE_API_TOKEN','GROQ_API_KEY','GEMINI_API_KEY'):
        os.environ[key]=secrets.get_secret(key)
    api.authenticate()
'''


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def supported_bundle(notebook):
    """Read exact private code snapshot, not arbitrary downloaded executable code."""
    source = ''.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError('Supported source hash mismatch')
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'files' for t in n.targets)]
    if len(assignments) != 1: raise ValueError('Expected one frozen code bundle')
    files = ast.literal_eval(assignments[0].value)
    for name, content in files.items():
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or not isinstance(content, str):
            raise ValueError('Unsafe source bundle path')
        if p.parts[0] not in ('experiments', 'src', 'eval', 'cloud', 'configs', 'scripts'):
            raise ValueError('Unexpected source bundle path')
    return files


def stage_bundle(code, files):
    """Reuse verified code after a preflight failure, never overwrite changed code."""
    code = Path(code); code.mkdir(parents=True, exist_ok=True)
    expected_py = {str(PurePosixPath(n)) for n in files if n.endswith('.py')}
    actual_py = {str(p.relative_to(code)) for p in code.rglob('*.py')}
    if actual_py - expected_py:
        raise RuntimeError('Unexpected executable source in recovery directory')
    for name, text in files.items():
        p = code/name
        if p.exists() and (not p.is_file() or p.is_symlink() or p.read_text() != text):
            raise RuntimeError('Existing recovery source differs from pinned bundle')
    for name, text in files.items():
        p = code/name; p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists(): p.write_text(text)
        os.chmod(p,0o600)


def restore_adapter(source):
    """Change credential acquisition only; all original restore checks retained."""
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'restore')
    original = ast.get_source_segment(source, node) + '\n'
    if original.count(CREDENTIAL_PROLOGUE) != 1:
        raise ValueError('Credential prologue differs from reviewed source')
    adapted = original.replace(CREDENTIAL_PROLOGUE,
                               '    from kaggle import api\n    api.authenticate()\n', 1)
    # Colab exposes /kaggle/input as read-only; use a private cloud directory.
    if adapted.count("Path('/kaggle/input')") != 1:
        raise ValueError('Mounted-input expression differs from reviewed source')
    adapted = adapted.replace("Path('/kaggle/input')", "Path('/content/d1000-private-input')", 1)
    # Only this function is compiled, under the unchanged module globals.
    return compile(adapted, '<reviewed-colab-restore-v1>', 'exec')


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def verify_journals(root):
    """Validate saved counts and retain exact unknown-outcome protection."""
    root = Path(root); votes = {}
    for provider, count in (('groq', 36), ('gemini', 345)):
        entries = rows(root/'judging'/f'{provider}.jsonl')
        ids = [r['evaluation_id'] for r in entries]
        if len(entries) != count or len(set(ids)) != count:
            raise ValueError('Saved judgement population mismatch')
        votes[provider] = set(ids)
    if len(votes['groq'] & votes['gemini']) != 32:
        raise ValueError('Paired population mismatch')
    uncertain = {'groq': 0, 'gemini': 0}
    for provider in uncertain:
        folder = root/'judge_runtime/evaluation_cache_v2'/provider
        for path in folder.glob('*.attempts.json'):
            history = json.loads(path.read_text())
            if not history or history[-1]['status'] not in ('in_flight','transport_uncertain','response_saved'):
                continue
            last = history[-1]; uncertain[provider] += 1
            if provider == 'gemini':
                if last['status'] != 'transport_uncertain' or last.get('error_category') != 'gemini HTTP 503':
                    raise ValueError('Unknown-outcome population changed')
                if (folder/(last['request_sha256']+'.raw.json')).exists():
                    raise ValueError('New durable reply requires separate reconciliation')
    if uncertain != {'groq': 0, 'gemini': 1}:
        raise ValueError('Expected protected Gemini uncertain request')
    # Hash all journals, caches and quota ledgers; this function never edits them.
    retained = {str(p.relative_to(root)): sha(p) for directory in ('judging', 'judge_runtime')
                for p in (root/directory).rglob('*') if p.is_file()}
    return {'groq':36, 'gemini':345, 'paired':32, 'uncertain':uncertain,
            'journals_manifest_sha256':hashlib.sha256(json.dumps(retained,sort_keys=True).encode()).hexdigest()}


class ColabRecovery:
    def __init__(self):
        self.ready = False; self.started = False
        self.session = uuid.uuid4().hex
        self.receipt = {'adapter':'colab_cpu_v1_1','status':'BLOCKED','new_judge_calls':0}

    def prepare(self, *, private_colab_transfer_approved=False):
        # This explicit gate comes before credential retrieval or private downloads.
        if not private_colab_transfer_approved:
            raise PermissionError('Confirm project permission for private checkpoint processing in your Colab VM')
        if not Path('/content').is_dir(): raise RuntimeError('Managed Colab runtime required')
        from google.colab import userdata
        if os.environ.get('COLAB_GPU') not in (None, '', '0'):
            raise RuntimeError('Select CPU runtime first')
        Path('/kaggle/working').mkdir(parents=True, exist_ok=True)
        os.chmod('/kaggle/working', 0o700)
        self.log = Path('/kaggle/working/recovery-private.log')
        self.log.touch(mode=0o600, exist_ok=True); os.chmod(self.log,0o600)
        # Colab Secrets are local to this notebook/account; no Kaggle key transfer.
        for key in ('KAGGLE_API_TOKEN', 'GROQ_API_KEY', 'GEMINI_API_KEY'):
            value = userdata.get(key)
            if not isinstance(value,str) or not value: raise RuntimeError('Required Colab secret unavailable')
            os.environ[key] = value
            del value
        with self.log.open('a') as log, redirect_stdout(log), redirect_stderr(log):
            self._prepare()
        self.ready = True
        self.receipt.update(status='READY_NOT_RUNNING', judge_credentials_available=True,
                            checkpoint=239, production_dataset_writes=0)
        return self.receipt.copy()

    def _prepare(self):
        from kaggle import api
        self.api = api; api.authenticate()
        # Pull only code/metadata in this private cloud VM; includes private audit sidebands.
        if api.kernels_status(KERNEL).status.name not in TERMINAL:
            raise RuntimeError('Kaggle evaluation writer is active')
        with tempfile.TemporaryDirectory() as tmp:
            api.kernels_pull(KERNEL+'/18', tmp, metadata=True)
            notebooks = list(Path(tmp).glob('*.ipynb'))
            if len(notebooks) != 1: raise RuntimeError('Ambiguous supported notebook')
            files = supported_bundle(json.loads(notebooks[0].read_text()))
        code = Path('/kaggle/working/code'); code.mkdir(exist_ok=True)
        stage_bundle(code, files)
        sys.path.insert(0,str(code)); os.chdir(code)
        from cloud.d1000 import evaluation_only as evaluation
        from cloud.d1000.storage import info, publish
        self.evaluation = evaluation
        self.info = info; self.publish = publish
        audit = json.loads((code/'cloud/d1000/evaluation_repair_v1.json').read_text())
        for p,h in {**audit['destination_hashes'], **audit['immutable_code_hashes']}.items():
            if sha(code/p) != h: raise RuntimeError('Audited source mismatch')
        # Private sideband hashes must match the existing repair audit.
        for name, field in (('groq_probe_quota_deltas.json','quota_deltas_sha256'),
                            ('groq_calibration14_receipt.json','groq_calibration_receipt_sha256')):
            if sha(code/'cloud/d1000'/name) != audit[field]: raise RuntimeError('Private audit sideband mismatch')
        d = info(api, STORE)
        if not d.is_private or d.current_version_number != 239: raise RuntimeError('Reconcile latest checkpoint first')
        INPUT_ROOT.mkdir(parents=True, exist_ok=True); os.chmod(INPUT_ROOT,0o700)
        mount = INPUT_ROOT/'recovery239'; mount.mkdir(exist_ok=True); os.chmod(mount,0o700)
        for name in ('progress.json','artifacts.bin'):
            api.dataset_download_file(STORE+'/239',name,path=str(mount),quiet=True,force=True,unzip=True)
        state = json.loads((mount/'progress.json').read_text())
        if state['artifacts_sha256'] != ARCHIVE_SHA or sha(mount/'artifacts.bin') != ARCHIVE_SHA:
            raise RuntimeError('Checkpoint239 archive mismatch')
        # Original restore verifies latest private metadata, mounted exact progress,
        # 912 responses, 76 questions, 3 adapters and unchanged config.
        scope = evaluation.__dict__.copy()
        exec(restore_adapter(files['cloud/d1000/evaluation_only.py']),scope)
        state, preservation = scope['restore']()
        if preservation['immutable_manifest_sha256'] != audit['immutable_manifest_sha256']:
            raise RuntimeError('Frozen benchmark/model inventory mismatch')
        self.receipt['preservation'] = preservation
        self.receipt['saved_judgements'] = verify_journals(evaluation.ROOT)
        # Private scratch dataset carries only nonce, never research data.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/'nonce.txt').write_text(self.session)
            try:
                previous = info(api,SCRATCH)
            except Exception as exc:
                status = getattr(getattr(exc,'response',None),'status_code',None)
                if status != 404: raise
                previous = None
            if previous is not None and not previous.is_private: raise RuntimeError('Scratch must be private')
            receipt = publish(folder,SCRATCH,create=previous is None)
            if not receipt['readback_verified'] or not receipt['private']: raise RuntimeError('Scratch readback failed')
            self.receipt['scratch_private_persistence'] = receipt
        if api.kernels_status(KERNEL).status.name not in TERMINAL or info(api,STORE).current_version_number != 239:
            raise RuntimeError('Writer or checkpoint changed during preflight')
        self.state = state; self.preservation = preservation
        # Bind only a reviewed restore adapter; original source files remain byte-identical.
        evaluation.restore = lambda: (self.state,self.preservation)

    def resume(self):
        if not self.ready or self.started: raise RuntimeError('Successful preflight and one start required')
        api = self.api; evaluation = self.evaluation
        if api.kernels_status(KERNEL).status.name not in TERMINAL:
            raise RuntimeError('Another evaluation writer is active')
        if self.info(api,STORE).current_version_number != 239:
            raise RuntimeError('Checkpoint changed; do not resume stale state')
        # A fixed private lease is created once. Existing lease => fail closed.
        # On interrupted sessions, user must reconcile/release the lease before restart.
        with self.log.open('a') as log, redirect_stdout(log), redirect_stderr(log):
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp); (folder/'owner.json').write_text(json.dumps({'session':self.session,'source_checkpoint':239}))
                lease=self.publish(folder,LEASE,create=True)
                self.lease_version=lease['version']
            fd=os.open('/kaggle/working/d1000-authoritative-writer.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            os.close(fd); self.started=True
            self.receipt.pop('new_judge_calls', None)
            original_checkpoint=evaluation.checkpoint
            def checked_checkpoint(state):
                d=self.info(api,LEASE)
                if not d.is_private or d.current_version_number != self.lease_version:
                    raise RuntimeError('Authoritative writer lease changed')
                if api.kernels_status(KERNEL).status.name not in TERMINAL:
                    raise RuntimeError('Unexpected Kaggle writer')
                return original_checkpoint(state)
            evaluation.checkpoint=checked_checkpoint
            evaluation.main()  # Original checks, calibration, replay guards and 120s persistence.
        # Report only aggregate data after original private write/readback.
        receipt=json.loads(Path('/kaggle/working/evaluation_receipt.json').read_text())
        self.receipt.update(status=receipt['status'],checkpoint=receipt['last_persistence']['version'],
            judge_counts=receipt['judge_counts'],paired_judgements=receipt.get('paired_judgements'),
            readback_verified=receipt['last_persistence']['readback_verified'],
            new_persisted_judgements={p:receipt['judge_counts'][p]-n for p,n in (('groq',36),('gemini',345))})
        return self.receipt.copy()
