"""Drain the old background worker after its control cell is interrupted at a checkpoint.

Never interrupt provider threads, restart the VM, reset flags, or release the lease.
"""
import json
from pathlib import Path
import threading
import time

PROTOCOL='d1000-single-judge-fast-v1'


def old_workers():
    return [t for t in threading.enumerate() if t.is_alive() and getattr(getattr(t,'_target',None),'__name__','')=='worker']


def drain(recovery,sink,timeout=660):
    from cloud.d1000 import judge_recovery
    if not recovery.started or not recovery.ready or recovery.lease_version!=1:
        raise RuntimeError('Existing verified writer required')
    previous=judge_recovery.complete
    def no_new_calls(*args,**kwargs):
        raise RuntimeError('Safe protocol handoff: no new old-protocol requests')
    judge_recovery.complete=no_new_calls
    until=time.monotonic()+timeout
    try:
        while old_workers():
            if time.monotonic()>until:raise RuntimeError('Old worker still active; no new writer allowed')
            for thread in old_workers():thread.join(timeout=2)
        # Reached only after any submitted complete() returned and its worker
        # journalled a result or protected uncertain outcome. No request is killed.
        root=Path(recovery.evaluation.ROOT)
        audit={'protocol_transition_requested':PROTOCOL,'session_lease_reused':True,
               'old_workers_drained':True,'transition_epoch':time.time(),
               'prior_checkpoint':recovery.state['last_persistence']['version']}
        folder=root/'recovery_audits';folder.mkdir(exist_ok=True)
        (folder/'single_judge_handoff.json').write_text(json.dumps(audit,sort_keys=True))
        recovery.state['status']='dual_judge_stopped_for_single_judge_handoff'
        recovery.evaluation.checkpoint(recovery.state)
        audit['final_dual_checkpoint']=recovery.state['last_persistence']['version']
        sink({'event':'old_coordinator_drained','old_worker_count':len(old_workers()),
              'same_lease_version':recovery.lease_version,'checkpoint':audit['final_dual_checkpoint'],
              'private_readback_verified':recovery.state['last_persistence']['readback_verified']})
        recovery.fast_handoff=audit
        return audit
    finally:
        # If draining times out keep the stop gate installed; never revive calls.
        if not old_workers():judge_recovery.complete=previous
