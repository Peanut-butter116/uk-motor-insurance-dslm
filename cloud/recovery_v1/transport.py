"""Verify archived transport against pinned production metadata; never reset ledgers."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout,redirect_stderr

PRODUCTION='scarletthe0116/uk-motor-qa-production-v1'
ARCHIVE='a88e3259cde5357f273f97a9ff2afa0821359815a7238c78332056d16511fe0e'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_verified(judging, root, manifest):
    if manifest.get('archive_sha256')!=ARCHIVE:
        raise RuntimeError('Original production archive identity differs')
    root=Path(root);code,runtime=judging.transport_paths(root)
    expected={code/name:manifest['files']['uk-insurance-QA-data/src/'+name]
              for name in ('providers.py','common.py','kelly.py')}
    expected[runtime/'provider_verification.json']=manifest['files']['uk-insurance-QA-data-local/production-v1/provider_verification.json']
    def provider_runtime(actual_root):
        if Path(actual_root).resolve()!=root.resolve():raise RuntimeError('Unexpected transport root')
        for path,value in expected.items():
            if not path.is_file() or digest(path)!=value:raise RuntimeError('Archived transport hash mismatch')
        ledgers=[runtime/'provider_usage'/f'{p}.json' for p in ('groq','gemini')]
        if not all(p.is_file() for p in ledgers):raise RuntimeError('Existing quota ledgers required')
        before={p:digest(p) for p in ledgers}
        sys.path.insert(0,str(code.parent))
        providers=importlib.import_module('production_transport.providers')
        if Path(providers.__file__).resolve()!=(code/'providers.py').resolve():
            raise RuntimeError('Unexpected transport import location')
        if before!={p:digest(p) for p in ledgers}:raise RuntimeError('Quota ledger changed during transport loading')
        return providers,runtime
    # Verify before installing; no code, receipts or ledgers are written.
    provider_runtime(root)
    judging.provider_runtime=provider_runtime
    return {'transport_hashes_verified':len(expected),'quota_ledgers_preserved':True,
            'source_production_version':319,'source_archive_sha256':ARCHIVE}


def prepare(recovery):
    from cloud.d1000 import judging
    from cloud.d1000.storage import info
    folder=Path('/content/d1000-private-input/transport319');folder.mkdir(parents=True,exist_ok=True)
    folder.chmod(0o700)
    with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        if not info(recovery.api,PRODUCTION).is_private:raise RuntimeError('Production source must remain private')
        recovery.api.dataset_download_file(PRODUCTION+'/319','migration_manifest.json',
            path=str(folder),quiet=True,force=True,unzip=True)
    path=folder/'migration_manifest.json';path.chmod(0o600)
    result=install_verified(judging,recovery.evaluation.ROOT,json.loads(path.read_text()))
    providers,runtime=judging.provider_runtime(recovery.evaluation.ROOT)
    for provider,model in (('groq','openai/gpt-oss-120b'),('gemini','gemini-3.5-flash-lite')):
        providers.verify_receipt(runtime,provider,model)
    recovery.receipt['archived_transport']=result
    return result
