"""Continue only the known pre-checkpoint mount failure under the SAME lease."""
import ast
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
from pathlib import Path
import threading

LEASE='scarletthe0116/uk-motor-d1000-evaluation-lease-v1'
STORE='scarletthe0116/uk-motor-d1000-models-results'
KERNEL='scarletthe0116/uk-motor-d1000-train-eval'


def continue_verified(recovery, failure, monitor, sink, verify_journals):
    if not recovery.ready or not recovery.started or not hasattr(recovery,'lease_version'):
        raise RuntimeError('Existing verified lease and preflight required')
    frames=[];tb=failure.__traceback__
    while tb is not None:
        frames.append((Path(tb.tb_frame.f_code.co_filename).name,tb.tb_frame.f_code.co_name));tb=tb.tb_next
    if not isinstance(failure,StopIteration) or frames[-1]!=('judging.py','provider_runtime'):
        raise RuntimeError('Only the diagnosed missing transport mount may continue here')
    if any(t.is_alive() and getattr(getattr(t,'_target',None),'__name__','')=='worker' for t in threading.enumerate()):
        raise RuntimeError('Evaluation worker already active')
    if not Path('/kaggle/working/d1000-authoritative-writer.lock').is_file():
        raise RuntimeError('Existing local exclusive writer lock required')
    if recovery.receipt.get('archived_transport',{}).get('transport_hashes_verified')!=4:
        raise RuntimeError('Pinned archived transport preflight required')
    if Path('/kaggle/working/evaluation_receipt.json').exists():
        raise RuntimeError('A production checkpoint exists; reconcile it instead')
    if recovery.state['last_persistence']['version']!=239:
        raise RuntimeError('Unexpected source checkpoint')
    from cloud.d1000.storage import verify_file
    if verify_journals(recovery.evaluation.ROOT)!=recovery.receipt['saved_judgements']:
        raise RuntimeError('Journals changed since verified preflight')
    api=recovery.api
    d=recovery.info(api,LEASE)
    if not d.is_private or d.current_version_number!=recovery.lease_version:
        raise RuntimeError('Writer lease changed')
    owner=json.dumps({'session':recovery.session,'source_checkpoint':239}).encode()
    verify_file(LEASE,recovery.lease_version,'owner.json',hashlib.sha256(owner).hexdigest())
    recovery.evaluation.verify_unchanged()
    # Compile the original on-disk checkpoint function, whose file remains pinned.
    evaluation=recovery.evaluation;path=Path(evaluation.__file__)
    audit=json.loads(Path('cloud/d1000/evaluation_repair_v1.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=audit['destination_hashes']['cloud/d1000/evaluation_only.py']:
        raise RuntimeError('Original checkpoint source changed')
    source=path.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='checkpoint')
    scope=evaluation.__dict__.copy();exec(compile(ast.get_source_segment(source,node),'original-audited-checkpoint','exec'),scope)
    original=scope['checkpoint']
    def checked_checkpoint(state):
        lease=recovery.info(api,LEASE)
        if not lease.is_private or lease.current_version_number!=recovery.lease_version:
            raise RuntimeError('Authoritative writer lease changed')
        if api.kernels_status(KERNEL).status.name not in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'):
            raise RuntimeError('Unexpected Kaggle writer')
        result=original(state)
        report=monitor['safe_progress'](state)
        report.update(monitor['saved_vote_counts']('/kaggle/working/publish/artifacts.bin'))
        sink(report)
        return result
    evaluation.checkpoint=checked_checkpoint
    if recovery.info(api,STORE).current_version_number!=239:
        raise RuntimeError('Results changed since verified restoration')
    # No new lease, flag reset, cache deletion, quota reset or request replay.
    with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        evaluation.main()
    return {'status':'SESSION_ENDED','lease_reused':True,'lease_version':recovery.lease_version}
