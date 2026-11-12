"""Same-VM continuation requires a deliberately drained, verified suspension.

No flags, leases, journals, reservations or original deadline are reset.
Provider-wide failures still stop that worker; no automatic outage retry loop.
"""
import asyncio
import hashlib
import json
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import time


async def resume_same_runtime(recovery,sink):
    from cloud.single_judge_fast import engine,lifecycle
    from cloud.single_judge_fast.resume import validate_suspension
    root=Path(recovery.evaluation.ROOT);folder=root/'single_judge_fast_v1'
    if not recovery.fast_started or not recovery.started or not recovery.ready:
        raise RuntimeError('Existing authoritative writer required')
    if lifecycle.provider_threads() or any(t.get_name() in ('fast-groq','fast-gemini') and not t.done() for t in asyncio.all_tasks()):
        raise RuntimeError('Prior provider work has not drained')
    proof=json.loads((folder/'suspension.json').read_text())
    prior=json.loads((folder/'protocol.json').read_text())
    validate_suspension(recovery.state,prior,proof,recovery.state['last_persistence']['version'],time.time())
    if proof['lease_version']!=recovery.lease_version:raise RuntimeError('Lease version changed')
    # Both preflight and continuation retain every attempt and quota byte.
    guarded=[p for d in ('judging','judge_runtime') for p in (root/d).rglob('*') if p.is_file()]
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in guarded}
    recovery.fast_resume_control=prior
    with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        engine.writer_guard(recovery)
        report=engine.prepare(recovery)
    if any(not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in before.items()):
        raise RuntimeError('Preflight changed protected research state')
    stop=folder/'STOP_REQUESTED'
    if not stop.is_file():raise RuntimeError('Controlled suspension marker missing')
    stop.rename(folder/('STOP_ACKNOWLEDGED-'+str(time.time_ns())))
    sink({**report,'event':'same_lease_quarantine_continuation','original_deadline_epoch':prior['deadline_epoch']})
    with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        return await engine.coordinate(recovery,sink)
