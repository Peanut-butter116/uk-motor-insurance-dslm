"""Calibrated, pointwise batched transport; full historical population first."""
import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core
from eval import semantic_judge as sj
from eval import judge_span_ids as indexed
from eval.gemini_judge import GeminiJudge, QuotaError, InvalidOutput, ProviderError, batch_bound, batch_schema, batch_output_budget, BATCH_WRAPPER
from eval.semantic_fixtures import fixtures
from scripts.run_semantic_judge import secure


def pack(tasks, mapping, seed, size=10):
    """Never place two answers to the same question/group in one request."""
    rng=random.Random(seed); by_group={}
    for t in tasks:by_group.setdefault(mapping[t['evaluation_id']]['question_group'],[]).append(t)
    keys=sorted(by_group);rng.shuffle(keys);queue=deque(keys)
    for values in by_group.values():rng.shuffle(values)
    while queue:
        batch=[];seen=set()
        while queue and len(batch)<size:
            group=queue.popleft()
            if group in seen:
                queue.appendleft(group);break
            task=by_group[group][-1]
            if batch_bound(batch+[task])>175000:
                queue.appendleft(group)
                if not batch:raise ValueError('Single task exceeds safe batch bound')
                break
            by_group[group].pop();batch.append(task);seen.add(group)
            if by_group[group]:queue.append(group)
        yield batch


def _execute(directory, free_receipt, shared_usage, catalogue, env_file,
            reserve_requests=50, request_limit=300, include_new=False, vote_indices=None, batch_size=10):
    out=secure(directory);manifest=json.loads((out/'judge_manifest.json').read_text())
    calibration=json.loads((out/'calibration_indexed_v3.json').read_text())
    if not calibration['passed'] or calibration['rubric_version']!=sj.VERSION or calibration['fixture_sha256']!=core.digest(fixtures()) or calibration.get('thinking_level') not in ('medium','high'):
        raise ValueError('Fictional batch calibration gate has not passed')
    tasks=core.rows(out/'judge_inputs.jsonl');mapping=core.unique(core.rows(out/'private_mapping.jsonl'),'evaluation_id')
    if core.sha(out/'judge_inputs.jsonl')!=manifest['inputs_sha256']:raise ValueError('Changed frozen inputs')
    if manifest.get('mapping_sha256') and core.sha(out/'private_mapping.jsonl')!=manifest['mapping_sha256']:raise ValueError('Changed private mapping')
    keys={}
    for line in Path(env_file).read_text().splitlines():
        if line.startswith('GEMINI_API_KEY='):keys['key']=line.split('=',1)[1].strip().strip('"').strip("'")
    free=json.loads(Path(free_receipt).read_text());free=free.get('gemini',free)
    if calibration['model']!=free['model']:raise ValueError('Calibration belongs to a different judge')
    vote_indices=vote_indices or [1,2,3]
    if type(batch_size) is not int or not 1<=batch_size<=10:raise ValueError('Batch size must be 1–10')
    if len(set(vote_indices))!=len(vote_indices) or set(vote_indices)-{1,2,3}:raise ValueError('Invalid vote indices')
    if manifest.get('judge_plan') and any(manifest['judge_plan'].get(str(v))!=free['model'] for v in vote_indices):
        raise ValueError('Worker model does not match planned vote assignment')
    client=GeminiJudge(keys['key'],free['model'],json.loads((out/'gemini_semantic_permission.json').read_text()),free,
                       out/'gemini_usage.jsonl',Path(shared_usage),json.loads(Path(catalogue).read_text()),reserve_requests=reserve_requests,thinking_level=calibration['thinking_level'])
    existing=core.rows(out/'judge_votes.jsonl');done={(v['evaluation_id'],v['vote']) for v in existing}
    if len(done)!=len(existing):raise ValueError('Duplicate votes')
    if any(v.get('transport_version')!=indexed.TRANSPORT for v in existing):raise ValueError('Do not mix transport protocols')
    cast=core.rows(out/'judge_attempts.jsonl');attempt_counts={}
    for a in cast:
        if a.get('error','') and 'Gemini HTTP' in a['error']:continue
        unit=(a['evaluation_id'],a['vote']);attempt_counts[unit]=attempt_counts.get(unit,0)+1
    manifest.update(provider='gemini',model=free['model'],transport_version=indexed.TRANSPORT,
         batch_wrapper_sha256=core.digest(BATCH_WRAPPER),rubric_version=sj.VERSION,prompt_sha256=core.digest(sj.PROMPT),
         thinking_level=calibration['thinking_level'],worker_vote_indices=vote_indices,fictional_calibration_sha256=core.sha(out/'calibration_indexed_v3.json'),
         mapping_sha256=core.sha(out/'private_mapping.jsonl'),
         paid_api_enabled=False,permission_status='owner_authorised_full_semantic_payload',
         production_reserve={'requests_per_day':reserve_requests,'requests_per_minute':1,'tokens_per_minute':50000},
         batch_max_tasks=batch_size,batch_max_bound=175000,batch_seed=20261020,maximum_attempts_per_vote=2,
         primary_schedule='historical all three vote rounds and one invalid-output retry pass, then new models if enabled')
    core.write_json(out/'judge_manifest.json',manifest);calls=0
    populations=['historical_sft21']+(['six_model_base_v1'] if include_new else [])
    expected={(t['evaluation_id'],v) for t in tasks if mapping[t['evaluation_id']]['experiment'] in populations for v in vote_indices}
    def finish(reason):
        state={'reason':reason,'expected_vote_units':len(expected),'valid_vote_units':len(done&expected),
               'pending_vote_units':len(expected-done),'vote_dispatch_complete':not(expected-done),
               'research_validation_complete':False,'requests_this_invocation':calls}
        core.write_json(out/'dispatch_status.json',state)
        print(json.dumps(state),flush=True)
        return state
    for population in populations:
        for retry_pass in range(2):
            for vote in vote_indices:
                todo=[t for t in tasks if mapping[t['evaluation_id']]['experiment']==population
                      and (t['evaluation_id'],vote) not in done and attempt_counts.get((t['evaluation_id'],vote),0)<2]
                for batch in pack(todo,mapping,20261020+vote,size=batch_size):
                    if calls>=request_limit:return finish('request_limit')
                    request_id=f'b{len(core.rows(out/"judge_requests.jsonl")):05d}'
                    receipt={'request_id':request_id,'vote':vote,'seed':20261020+vote,'population':population,
                             'task_ids':[t['evaluation_id'] for t in batch],'input_sha256':core.digest(batch),
                             'response_schema_sha256':core.digest(batch_schema(batch)),'transport_version':indexed.TRANSPORT,
                             'indexed_view_sha256':core.digest([indexed.view(t) for t in batch])}
                    receipt['max_output_tokens']=batch_output_budget(batch)
                    usage=None;raw_batch=None
                    try:
                        while True:
                            try:results,usage=client.call_many(batch,20261020+vote);break
                            except QuotaError as e:
                                if e.retry_after is None:
                                    print(str(e),flush=True);return finish('free_quota_guard')
                                print('Waiting for shared free minute quota',round(e.retry_after,1),flush=True)
                                time.sleep(min(60,e.retry_after))
                    except InvalidOutput as e:
                        usage=e.usage;raw_batch=e.raw;results=[{'valid':False,'error':str(e),'raw':''} for t in batch]
                    except (ValueError,RuntimeError) as e:
                        if isinstance(e,ProviderError):receipt['provider_error']=e.details
                        results=[{'valid':False,'error':str(e),'raw':''} for t in batch]
                    calls+=1;receipt.update(usage=usage,raw_invalid_batch=raw_batch,valid_items=sum(r['valid'] for r in results))
                    with (out/'judge_requests.jsonl').open('a') as f:f.write(json.dumps(receipt,ensure_ascii=False)+'\n')
                    for i,(task,r) in enumerate(zip(batch,results)):
                        unit=(task['evaluation_id'],vote);attempt_counts[unit]=attempt_counts.get(unit,0)+1
                        record={'evaluation_id':task['evaluation_id'],'vote':vote,'seed':20261020+vote,
                             'provider':'gemini','model':usage['model'] if usage else client.model,
                             'judge_thinking_level':client.thinking_level,
                             'transport_version':indexed.TRANSPORT,'request_id':request_id,
                             'rubric_version':sj.VERSION,'prompt_sha256':core.digest(sj.PROMPT),
                             'valid':r['valid'],'raw':r['raw'],'error':r.get('error'),
                             'input_tokens':usage['usage'].get('promptTokenCount') if i==0 and usage else None,
                             'output_tokens':usage['usage'].get('candidatesTokenCount') if i==0 and usage else None,
                             'thinking_tokens':usage['usage'].get('thoughtsTokenCount',0) if i==0 and usage else None,
                             'total_tokens':usage['usage'].get('totalTokenCount') if i==0 and usage else None,
                             'token_usage_scope':'entire request, attributed to first item only',
                             'estimated_cost_usd':0}
                        with (out/'judge_attempts.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                        if r['valid']:
                            with (out/'judge_votes.jsonl').open('a') as f:f.write(json.dumps({**r['judgement'],**{k:v for k,v in record.items() if k not in ('raw','error')}},ensure_ascii=False)+'\n')
                            done.add(unit)
                    print(request_id,population,'vote',vote,'items',len(batch),'valid',receipt['valid_items'],'total valid votes',len(done),flush=True)
                    if receipt.get('provider_error'):
                        print('Provider rejected request; stopped for engineering repair',flush=True);return finish('provider_rejected_request')
    return finish('all_scheduled_units_attempted')


def execute(**kwargs):
    import fcntl
    out=secure(kwargs['directory'])
    with (out/'.judge_worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another judge worker already owns this run') from None
        return _execute(**kwargs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','free-receipt','shared-usage','catalogue','env-file'):p.add_argument('--'+name,required=True)
    p.add_argument('--reserve-requests',type=int,default=50);p.add_argument('--request-limit',type=int,default=300)
    p.add_argument('--vote-indices',nargs='+',type=int,choices=[1,2,3])
    p.add_argument('--batch-size',type=int,default=10)
    p.add_argument('--include-new',action='store_true');execute(**vars(p.parse_args()))
