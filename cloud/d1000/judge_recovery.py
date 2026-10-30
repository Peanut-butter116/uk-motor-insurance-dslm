"""Schema-aware, quota-reserved judge transport with durable bounded retries."""
import json
from pathlib import Path
import time
import asyncio
from cloud.d1000 import judging

PROTOCOL='d1000-evaluation-only-schema-v2'
INVARIANTS='''
Transport clarification of the unchanged rubric (not extra grading labels):
Return exactly the evaluations wrapper and all required item fields, including
citation_assessments (empty when no supplied student evidence/citations).
fact_scores are INTEGER 0/1/2, separate from correctness 0/.5/1.
A deterministic refusal makes no substantive policy claim: evidence_support
must be JSON null, even when the refusal is correct. Refusal correctness is a
separate decision under the existing rubric. A refusal needs no invented
positive policy-support span. Missing reference evidence must remain uncertain.
Use only supplied indexed IDs; do not invent or silently coerce values.
'''


def body(provider,model,task,feedback=None):
    messages=[{'role':'user','content':judging.indexed.prompt([task])+judging.EXTENSION+INVARIANTS}]
    if feedback:messages.append({'role':'user','content':'The previous response was invalid: '+feedback+'. Return a fresh complete JSON judgement under the same rubric; do not change facts to satisfy a score.'})
    result={'model':model,'messages':messages,'temperature':0,'stream':False,
        'response_format':{'type':'json_schema','json_schema':{'name':'motor_semantic_judgement','strict':True,'schema':judging.schema(task)}}}
    if provider=='groq':result.update(max_completion_tokens=2048,reasoning_effort='low')
    else:result.update(max_tokens=2048,reasoning_effort='low')
    return result


def validate(result,task,providers):
    choice=result.get('choices',[{}])[0];message=choice.get('message',{})
    if choice.get('finish_reason')!='stop':raise ValueError('incomplete_or_refused_output')
    if message.get('refusal'):raise ValueError('model_refusal')
    value=providers.parse_json_content(message.get('content'))
    providers.validate_answer(value,judging.schema(task))
    if len(value['evaluations'])!=1:raise ValueError('Expected exactly one evaluation')
    return judging.normalise(value['evaluations'][0],task)


def complete(values,provider,model,task,runtime,providers):
    """Cache identity binds the opaque task, model and exact transport protocol.

    Unknown requests are recovered from durable raw replies or blocked; never
    replayed. Only known invalid replies permit another bounded attempt.
    """
    digest=judging.core.digest
    limits=providers.verify_receipt(runtime,provider,model)
    folder=Path(runtime)/'evaluation_cache_v2'/provider;folder.mkdir(parents=True,exist_ok=True)
    ident=digest({'task':task,'provider':provider,'model':model,'protocol':PROTOCOL})
    path=folder/(ident+'.json');history_path=folder/(ident+'.attempts.json')
    if path.exists():
        record=json.loads(path.read_text());judging.normalise(record['indexed_answer'],task)
        return record

    history=json.loads(history_path.read_text()) if history_path.exists() else []
    quota=providers.Quotas(Path(runtime)/'provider_usage'/(provider+'.json'),provider,limits)
    while True:
        unknown=bool(history and history[-1]['status'] in ('in_flight','transport_uncertain','response_saved'))
        if unknown:
            entry=history[-1];raw=folder/(entry['request_sha256']+'.raw.json')
            if not raw.exists():raise RuntimeError('Unknown request outcome; no automatic replay')
            result=json.loads(raw.read_text());actual=entry['generation_config']
        else:
            if len(history)>=3:raise RuntimeError('Known-invalid judgement exhausted three bounded attempts')
            feedback=history[-1].get('validation_error') if history else None
            actual=body(provider,model,task,feedback)
            # Include on-wire schema tokens in the conservative account reservation.
            counted=actual['messages']+[{'role':'user','content':json.dumps(actual['response_format'])}]
            request_hash=digest({'provider':provider,'body':actual,'attempt':len(history),'protocol':PROTOCOL})
            reservation=quota.reserve(request_hash,providers.token_reservation(provider,model,counted,2048))
            entry={'request_sha256':request_hash,'status':'in_flight','reservation':reservation,
                   'generation_config':{k:v for k,v in actual.items() if k!='messages'},'task_sha256':digest(task)}
            history.append(entry);providers.atomic_private(history_path,history)
            raw=folder/(request_hash+'.raw.json')
            try:result,_=providers.transport(values,provider,actual)
            except providers.QuotaPause as e:
                history.pop();providers.atomic_private(history_path,history);quota.pause(e.until);raise
            except Exception:
                entry['status']='transport_uncertain';providers.atomic_private(history_path,history);raise
            providers.atomic_private(raw,result);entry['status']='response_saved';providers.atomic_private(history_path,history)
        usage=result.get('usage',{});quota.reconcile(entry['reservation'],usage)
        try:
            canonical=validate(result,task,providers)
        except Exception as e:
            # Fixed validators emit structural categories, never protected labels.
            error=getattr(e,'kind',str(e)) or type(e).__name__
            entry.update(status='known_invalid',validation_error=error[:160],usage=usage,
                         finish_reason=result.get('choices',[{}])[0].get('finish_reason'))
            providers.atomic_private(history_path,history)
            continue
        indexed_answer=providers.parse_json_content(result['choices'][0]['message']['content'])['evaluations'][0]
        record={'answer':canonical,'indexed_answer':indexed_answer,'usage':usage,'request_sha256':entry['request_sha256'],
                'input_sha256':digest(task),'generation_config':entry['generation_config'],'protocol':PROTOCOL}
        providers.atomic_private(path,record);entry.update(status='success',usage=usage);providers.atomic_private(history_path,history)
        return record


async def run(root,state,calibration_ready):
    from eval.semantic_fixtures import fixtures
    from experiments import core
    import os
    root=Path(root);providers,runtime=judging.provider_runtime(root)
    values={k:os.environ[k] for k in ('GROQ_API_KEY','GEMINI_API_KEY')}
    out=root/'judging';out.mkdir(exist_ok=True)
    candidates=judging.tasks(root)
    assert len(candidates)==912,'All 912 opaque tasks must be available before calls'
    deadline=time.monotonic()+9*3600
    async def worker(provider,model):
        status=state.setdefault('judges',{})[provider]={'model':model,'status':'preflight','completed':0,'calibration':'PENDING'}
        async def call(task):
            while True:
                try:return await asyncio.to_thread(complete,values,provider,model,task,runtime,providers)
                except providers.QuotaPause as e:
                    status.update(status='quota_wait',resume_after_epoch=e.until)
                    if time.monotonic()+max(0,e.until-time.time())>deadline:raise RuntimeError('Verified free quota exceeds this session; resume from cache')
                    while time.time()<=e.until:await asyncio.sleep(min(30,max(1,e.until-time.time()+1)))
        try:
            providers.verify_receipt(runtime,provider,model)
            catalogue,_=await asyncio.to_thread(providers.transport,values,provider)
            assert providers.catalogue_contains(provider,model,catalogue),'Exact judge model absent'
            status['catalogue_verified']=True
            receipt=out/(provider+'-calibration-v2.json')
            anchors=fixtures();fixture_hash=core.digest(anchors)
            if receipt.exists():
                calibration=json.loads(receipt.read_text())
                assert calibration['protocol']==PROTOCOL and calibration['fixtures_sha256']==fixture_hash
            else:
                results=[]
                for task,expected in anchors:
                    record=await call(dict(task,citations=[],student_evidence_available=False))
                    v=record['answer']
                    # Labels are only compared after validation, never sent as feedback.
                    passed=v['correctness']==expected
                    if task['deterministic_abstention']:
                        passed=passed and v['evidence_support'] is None and v['abstention_category']==('incorrect_abstention' if task['gold_answerable'] else 'correct_abstention')
                    results.append({'id':task['evaluation_id'],'passed':passed,'correctness':v['correctness'],'expected':expected,'evidence_support':v['evidence_support'],'abstention_category':v['abstention_category'],'request_sha256':record['request_sha256']})
                    status['calibration_cases_completed']=len(results)
                calibration={'protocol':PROTOCOL,'fixtures_sha256':fixture_hash,'passed':len(results)==9 and all(v['passed'] for v in results),'results':results}
                providers.atomic_private(receipt,calibration)
            status['calibration']='PASS' if calibration['passed'] else 'FAIL'
            if not calibration['passed']:status['status']='calibration_failed';return
            status['status']='calibration_passed_waiting_persistence'
            # Coordinator verifies a private checkpoint before permitting research calls.
            while not calibration_ready.is_set():await asyncio.sleep(1)
            journal=out/(provider+'.jsonl')
            done={r['evaluation_id'] for r in core.rows(journal)} if journal.exists() else set()
            assert len(done)<=912
            for task,meta in candidates:
                if task['evaluation_id'] in done:continue
                try:record=await call(task)
                except Exception as e:
                    reason=str(e)
                    if 'exceeds verified free token capacity' in reason or 'Known-invalid judgement exhausted' in reason:
                        with (out/(provider+'-evaluation-failures-v2.jsonl')).open('a') as f:
                            f.write(json.dumps({'evaluation_id':task['evaluation_id'],'type':type(e).__name__,'reason':reason[:300]})+'\n')
                        status['missing_failures']=status.get('missing_failures',0)+1
                        continue
                    status.update(status='blocked',reason=reason[:300],completed=len(done));return
                with journal.open('a') as f:
                    f.write(json.dumps({**record['answer'],'provider':provider,'model':model,'request_sha256':record['request_sha256'],'usage':record['usage'],'input_sha256':record['input_sha256']})+'\n');f.flush();os.fsync(f.fileno())
                done.add(task['evaluation_id']);status.update(status='running',completed=len(done))
            status.update(status='complete',completed=len(done))
        except Exception as e:
            status.update(status='calibration_blocked' if status['calibration']!='PASS' else 'blocked',reason=str(e)[:300])
    await asyncio.gather(*(worker(p,m) for p,m in {'groq':'openai/gpt-oss-120b','gemini':'gemini-3.5-flash-lite'}.items()))
