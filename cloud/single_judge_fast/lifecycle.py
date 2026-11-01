"""A controlled, fully drained checkpoint is required before runtime renewal.

An arbitrary VM termination is NOT treated as evidence that unsaved requests were
unprocessed. Such recovery stays blocked pending separate outcome reconciliation.
"""
import asyncio
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from cloud.single_judge_fast.protocol import PROTOCOL,MODELS,select,load_validated,rows


def clock(prior,now,coverage):
    if prior is None:return now,now+20*3600,coverage
    if prior.get('protocol')!=PROTOCOL or prior.get('phase')!='suspended':raise RuntimeError('Verified suspended protocol required')
    start=prior['launch_epoch'];end=prior['deadline_epoch']
    if abs(end-start-72000)>.001:raise RuntimeError('Original deadline changed')
    if now>=end-900:raise RuntimeError('Insufficient time before original deadline')
    return start,end,prior['baseline_unique']


def provider_threads():
    matches=[]
    for thread_id,frame in sys._current_frames().items():
        while frame is not None:
            name=frame.f_code.co_filename;function=frame.f_code.co_name
            request=Path(name).name=='judge_recovery.py' and function in ('complete','observed_transport')
            persistence=any(part in name for part in ('evaluation_only','storage.py','continue_startup','original-audited-checkpoint')) and function in ('checkpoint','checked_checkpoint','publish','verify_file')
            if request or persistence:
                matches.append(thread_id);break
            frame=frame.f_back
    return matches


async def suspend(recovery,sink,timeout=660):
    from cloud.d1000 import judging,judge_recovery as jr
    from cloud.single_judge_fast import engine,metrics
    from experiments import core
    root=Path(recovery.evaluation.ROOT);folder=root/'single_judge_fast_v1'
    if not getattr(recovery,'fast_started',False) or not hasattr(recovery,'fast_control'):raise RuntimeError('Existing fast coordinator required')
    (folder/'STOP_REQUESTED').touch(exist_ok=True)
    until=time.monotonic()+timeout
    while True:
        workers=[t for t in asyncio.all_tasks() if t is not asyncio.current_task() and t.get_name() in ('fast-groq','fast-gemini') and not t.done()]
        if not workers and not provider_threads():break
        if time.monotonic()>until:raise RuntimeError('Provider work still active; cannot suspend or renew runtime')
        await asyncio.sleep(1)
    engine.writer_guard(recovery)
    candidates=judging.tasks(root);providers,_=judging.provider_runtime(root)
    # Preserve any fully validated cache success returned before an interrupted
    # control cell could append its provider journal. Never issue an API call.
    known={p:{v['evaluation_id'] for v in rows(root/'judging'/f'{p}.jsonl')} for p in MODELS}
    for task,_ in candidates:
        for p,model in MODELS.items():
            if task['evaluation_id'] in known[p]:continue
            key=core.digest({'task':task,'provider':p,'model':model,'protocol':jr.PROTOCOL})
            cache=root/'judge_runtime/evaluation_cache_v2'/p/(key+'.json')
            if not cache.exists():continue
            record=json.loads(cache.read_text())
            if record['protocol']!=jr.PROTOCOL or record['input_sha256']!=core.digest(task) or record['generation_config']['model']!=model:raise RuntimeError('Cached success identity differs')
            result={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'evaluations':[record['indexed_answer']]})}}]}
            if jr.validate(result,task,providers)!=record['answer']:raise RuntimeError('Cached result validation differs')
            value={**record['answer'],'provider':p,'model':model,'request_sha256':record['request_sha256'],
                   'usage':record['usage'],'input_sha256':record['input_sha256'],'evaluation_protocol':PROTOCOL}
            with (root/'judging'/f'{p}.jsonl').open('a') as f:
                f.write(json.dumps(value)+'\n');f.flush();__import__('os').fsync(f.fileno())
            known[p].add(task['evaluation_id'])
    votes=load_validated(root,candidates);summary=metrics.summarise(root,votes)
    owner=json.dumps({'session':recovery.session,'source_checkpoint':239}).encode()
    proof={'protocol':PROTOCOL,'phase':'suspended','active_provider_threads':0,'active_provider_tasks':0,
           'expected_checkpoint':recovery.state['last_persistence']['version']+1,
           'lease_version':recovery.lease_version,'lease_owner_sha256':hashlib.sha256(owner).hexdigest(),
           'suspended_epoch':time.time(),'unique_judged':summary['unique_judged'],
           'launch_epoch':recovery.fast_control['launch_epoch'],'deadline_epoch':recovery.fast_control['deadline_epoch']}
    recovery.fast_control.update(phase='suspended',metrics=summary)
    (folder/'protocol.json').write_text(json.dumps(recovery.fast_control,sort_keys=True,indent=2))
    (folder/'suspension.json').write_text(json.dumps(proof,sort_keys=True,indent=2))
    recovery.state['single_judge_fast']=recovery.fast_control
    recovery.state['status']='single_judge_suspended_for_supported_runtime_renewal'
    await asyncio.to_thread(recovery.evaluation.checkpoint,recovery.state)
    if recovery.state['last_persistence']['version']!=proof['expected_checkpoint']:raise RuntimeError('Suspension checkpoint version changed')
    actual=recovery.fast_archive_audit('/kaggle/working/publish/artifacts.bin')
    if actual['persisted_judge_counts']!={p:len(votes[p]) for p in MODELS} or any(actual['persisted_duplicate_votes'].values()):raise RuntimeError('Suspension journal readback differs')
    recovery.fast_suspension=proof
    sink({'event':'single_judge_suspended_readback','checkpoint':proof['expected_checkpoint'],
          'private_readback_verified':True,'unique_judged':summary['unique_judged'],'remaining':summary['remaining'],
          'active_provider_threads':0,'active_provider_tasks':0,'original_deadline_epoch':proof['deadline_epoch']})
    return proof
