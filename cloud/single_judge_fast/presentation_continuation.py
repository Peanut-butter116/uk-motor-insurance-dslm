"""Owner-authorised 2 November completion window, unchanged judging protocol.

The missed 20-hour deadline remains historical evidence. This separate execution
window never rewrites that clock, repeats Gemini's exhausted tasks, truncates a
prompt, resets quotas or starts before the recorded Groq availability boundary.
"""
import asyncio
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import time
from cloud.single_judge_fast.protocol import PROTOCOL,MODELS,load_validated,select,unknown_ids,quota_audit
from cloud.single_judge_fast.engine import writer_guard
from cloud.single_judge_fast.lifecycle import provider_threads

NOT_BEFORE=1793644833.557822
ORIGINAL_DEADLINE=1793632557.1127477


def validate_entry(result,state,not_before,window_end,now):
    if result.get('status')!='PARTIAL_STOPPED' or result.get('checkpoint')>state['last_persistence']['version']:
        raise RuntimeError('Verified normal original coordinator finish required')
    if not state['last_persistence'].get('private') or not state['last_persistence'].get('readback_verified'):
        raise RuntimeError('Private readback required')
    if state['single_judge_fast']['deadline_epoch']!=ORIGINAL_DEADLINE:
        raise RuntimeError('Historical deadline changed')
    if not_before<NOT_BEFORE or window_end<=not_before or now>=window_end:
        raise RuntimeError('Invalid authorised execution window')


async def run_when_available(recovery,original_result,sink,*,window_end):
    from cloud.d1000 import judging,judge_recovery as jr
    from cloud.single_judge_fast import metrics
    validate_entry(original_result,recovery.state,NOT_BEFORE,window_end,time.time())
    if getattr(recovery,'presentation_continuation_entered',False):raise RuntimeError('Continuation already entered')
    if provider_threads() or any(t.get_name() in ('fast-groq','fast-gemini') and not t.done() for t in asyncio.all_tasks()):
        raise RuntimeError('Original workers remain active')
    recovery.presentation_continuation_entered=True
    sink({'event':'groq_presentation_window_queued','not_before_epoch':NOT_BEFORE,'window_end_epoch':window_end,
          'original_deadline_epoch':ORIGINAL_DEADLINE,'judge_calls':0,'same_lease_version':recovery.lease_version})
    # Legitimate deferred evaluation; no API calls or synthetic keepalive traffic.
    while time.time()<NOT_BEFORE:await asyncio.sleep(min(30,NOT_BEFORE-time.time()))
    if time.time()>=window_end:return {'status':'WINDOW_MISSED','judge_calls':0}
    writer_guard(recovery)
    root=Path(recovery.evaluation.ROOT);folder=root/'presentation_continuation';folder.mkdir(exist_ok=True)
    (folder/'source.py').write_bytes(Path(__file__).read_bytes())
    providers,runtime=judging.provider_runtime(root);candidates=judging.tasks(root)
    votes=load_validated(root,candidates);covered=set(select(votes));baseline=len(covered)
    if baseline!=original_result['unique_judged']:raise RuntimeError('Research state changed since normal stop')
    protected=set().union(*(unknown_ids(root,candidates,p) for p in MODELS))
    if len(protected)!=4:raise RuntimeError('Protected request population changed')
    jr.verify_calibration(json.loads((root/'judging/groq-calibration-v2.json').read_text()))
    values={k:os.environ[k] for k in ('GROQ_API_KEY','GEMINI_API_KEY')}
    catalogue,_=providers.transport(values,'groq')
    if not providers.catalogue_contains('groq',MODELS['groq'],catalogue):raise RuntimeError('Original calibrated Groq model unavailable')
    limits=providers.verify_receipt(runtime,'groq',MODELS['groq']);costs={}
    for task,_ in candidates:
        if task['evaluation_id'] in covered|protected:continue
        body=jr.body('groq',MODELS['groq'],task)
        costs[task['evaluation_id']]=providers.token_reservation('groq',MODELS['groq'],body['messages']+[{'role':'user','content':json.dumps(body['response_format'])}],2048)
    failed=set();new=0;active=False
    control={'protocol':PROTOCOL,'operational_window':'owner-authorised-presentation-completion-2026-11-02',
             'not_before_epoch':NOT_BEFORE,'window_end_epoch':window_end,'original_deadline_epoch':ORIGINAL_DEADLINE,
             'source_checkpoint':recovery.state['last_persistence']['version'],'same_lease_version':recovery.lease_version,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_rubric_unchanged':True,
             'provider':'groq','model':MODELS['groq'],'concurrency':1,'protected_requests':4}
    async def checkpoint(event):
        if active:raise RuntimeError('Active request cannot race checkpoint')
        writer_guard(recovery);summary=metrics.summarise(root,votes)
        report={**control,'unique_judged':len(covered),'remaining':912-len(covered),
                'non_quarantined_missing':912-len(covered)-len(protected-covered),
                'quarantined_unjudged':len(protected-covered),'protected_requests':len(protected),'new_votes':new,
                'provider_votes':{p:len(votes[p]) for p in MODELS},'phase':phase,
                'quota_audit':quota_audit(runtime,providers),'active_provider_calls':0}
        recovery.state['presentation_continuation']=report
        (folder/'progress.json').write_text(json.dumps(report,indent=2))
        await asyncio.to_thread(recovery.evaluation.checkpoint,recovery.state)
        audit=recovery.fast_archive_audit('/kaggle/working/publish/artifacts.bin')
        if audit['persisted_judge_counts']!=report['provider_votes'] or any(audit['persisted_duplicate_votes'].values()):
            raise RuntimeError('Continuation readback differs')
        result={**report,'event':event,'checkpoint':recovery.state['last_persistence']['version'],
                'archive_sha256':recovery.state['artifacts_sha256'],'readback_verified':True}
        sink(result);return result
    phase='preflight_verified';await checkpoint('groq_presentation_startup_readback')
    try:
        with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
            while time.time()<window_end-900:
                tasks=[t for t,_ in candidates if t['evaluation_id'] not in covered|protected|failed
                       and costs.get(t['evaluation_id'],10**20)<=limits['tpm']]
                if not tasks:phase='no_more_fitting_safe_tasks';break
                task=min(tasks,key=lambda t:(costs[t['evaluation_id']],t['evaluation_id']))
                ident=task['evaluation_id']
                try:
                    phase='judging';active=True
                    record=await asyncio.to_thread(jr.complete,values,'groq',MODELS['groq'],task,runtime,providers,
                        transport=lambda v,p,b:jr.observed_transport(v,p,b,runtime,providers))
                    row={**record['answer'],'provider':'groq','model':MODELS['groq'],'request_sha256':record['request_sha256'],
                         'usage':record['usage'],'input_sha256':record['input_sha256'],'evaluation_protocol':PROTOCOL}
                    with (root/'judging/groq.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
                    votes['groq'][ident]=row;covered.add(ident);new+=1
                except providers.QuotaPause as exc:
                    phase='quota_wait';active=False
                    await checkpoint('groq_presentation_quota_wait_readback')
                    if exc.until>=window_end-900:break
                    while time.time()<exc.until:await asyncio.sleep(min(30,max(.2,exc.until-time.time())))
                except Exception as exc:
                    failed.add(ident);protected.update(unknown_ids(root,candidates,'groq'))
                    if 'Known-invalid judgement exhausted' not in str(exc):phase='provider_stopped';break
                finally:active=False
                if new and new%3==0:await checkpoint('groq_presentation_votes_readback')
            else:phase='window_end_drain'
            return await checkpoint('groq_presentation_terminal_readback')
    finally:
        # No request is cancelled to meet the operational window. Original
        # complete() bounds transport timeout and saves outcomes durably.
        if not active:recovery.presentation_continuation_finished=True
