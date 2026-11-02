"""Return to the already-calibrated original judges after a failed alternative.

This is a same-VM, same-lease transition. Failed alternative calibration never
authorises that model for research or resets its bounded attempt journal.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import time


def validate_transition(control, suspension, alternative, persistence, lease):
    from cloud.single_judge_fast.lifecycle import clock
    if control.get('phase')!='suspended' or suspension.get('phase')!='suspended':
        raise RuntimeError('Controlled suspension required')
    if suspension.get('active_provider_threads')!=0 or suspension.get('active_provider_tasks')!=0:
        raise RuntimeError('Historical requests were not drained')
    if alternative.get('phase')!='calibration_failed' or alternative.get('calibration')!='FAIL':
        raise RuntimeError('Explicit failed alternative receipt required')
    if alternative.get('new_votes')!=0 or alternative.get('source_checkpoint')!=suspension.get('expected_checkpoint'):
        raise RuntimeError('Alternative performed research or changed suspension identity')
    if not persistence.get('private') or not persistence.get('readback_verified'):
        raise RuntimeError('Latest private readback required')
    if persistence['version']<=suspension['expected_checkpoint']:
        raise RuntimeError('Alternative evidence checkpoint required')
    if suspension['lease_version']!=lease or alternative['lease_version']!=lease:
        raise RuntimeError('Authoritative lease changed')
    if alternative['original_deadline_epoch']!=control['deadline_epoch']:
        raise RuntimeError('Deadline changed')
    clock(control,time.time(),suspension['unique_judged'])


async def resume(recovery,sink):
    from cloud.single_judge_fast import engine,lifecycle
    from cloud.third_judge_fast.engine import preserved,audit_archive
    root=Path(recovery.evaluation.ROOT);folder=root/'single_judge_fast_v1';ctx=recovery.third
    if not recovery.ready or not recovery.started or not recovery.fast_started or getattr(recovery,'third_started',False):
        raise RuntimeError('Existing stopped original writer required')
    if ctx.get('active',0) or lifecycle.provider_threads() or any(
        t.get_name() in ('fast-groq','fast-gemini') and not t.done() for t in asyncio.all_tasks()):
        raise RuntimeError('Provider work remains active')
    prior=json.loads((folder/'protocol.json').read_text())
    proof=json.loads((folder/'suspension.json').read_text())
    validate_transition(prior,proof,ctx['state'],recovery.state['last_persistence'],recovery.lease_version)
    if ctx.get('calibration_verified') or ctx['votes']['gemini31']:
        raise RuntimeError('Unexpected alternative research state')
    preserved(root,ctx['manifest'])
    ledger=json.loads((ctx['shared']/'provider_usage/gemini.json').read_text())
    if ledger['entries'][:len(ctx['ledger_prefix'])]!=ctx['ledger_prefix']:
        raise RuntimeError('Historical quota reservations changed')
    audit_archive('/kaggle/working/publish/artifacts.bin',ctx['folder'],0)
    guarded=[p for d in ('judging','judge_runtime','third_judge_fast_v1') for p in (root/d).rglob('*') if p.is_file()]
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in guarded}
    with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        engine.writer_guard(recovery)
        report=engine.prepare(recovery) # Revalidates original 9/9 receipts, votes, models and all unknown IDs.
    if any(not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in before.items()):
        raise RuntimeError('Preflight changed protected state')
    if report['protected_task_counts']!={'groq':0,'gemini':4}:
        raise RuntimeError('Unknown population changed; reconcile before dispatch')
    stop=folder/'STOP_REQUESTED'
    if not stop.is_file():raise RuntimeError('Suspension marker missing')
    audit={'transition':'failed_alternative_to_original_judges_v1','source_checkpoint':recovery.state['last_persistence']['version'],
           'lease_version':recovery.lease_version,'protected_requests':4,'alternative_research_disabled':True,
           'original_deadline_epoch':prior['deadline_epoch'],'original_calibration_revalidated':True}
    (root/'recovery_audits/after_calibration_v1.json').write_text(json.dumps(audit,indent=2))
    stop.rename(folder/('STOP_ACKNOWLEDGED-after-calibration-'+str(time.time_ns())))
    recovery.fast_resume_control=prior
    sink({**report,'event':'original_judges_resume_preflight','alternative_judge_enabled':False})
    with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        return await engine.coordinate(recovery,sink)
