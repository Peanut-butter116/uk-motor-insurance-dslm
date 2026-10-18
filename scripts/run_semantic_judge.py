"""Private, resumable semantic judging. No remote research-data transfer by default."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import core
from eval import semantic_judge as sj
from eval.semantic_fixtures import fixtures

MODEL = 'mlx-community/Phi-4-mini-instruct-4bit'
REVISION = 'ac1c269cb4222a4e136a3d09edad301056c1f36a'


def secure(path):
    path = Path(path).resolve()
    if path == core.ROOT or core.ROOT in path.parents:
        raise ValueError('Restricted judge materials must be outside public repository')
    return path


def evidence_groups(home, directory):
    """Private source-dependence map; no benchmark or reference edits."""
    home=Path(home);directory=secure(directory)
    gold=[g for g in core.benchmark(home) if g['split']=='test' and g['line']=='motor']
    parent={g['id']:g['id'] for g in gold};first={}
    def root(q):
        while parent[q]!=q:q=parent[q]
        return q
    actual={}
    if (directory/'judge_inputs.jsonl').exists() and (directory/'private_mapping.jsonl').exists():
        mapping=core.unique(core.rows(directory/'private_mapping.jsonl'),'evaluation_id')
        for t in core.rows(directory/'judge_inputs.jsonl'):
            q=mapping[t['evaluation_id']]['qid']
            actual.setdefault(q,set()).update(c.get('source',{}).get('filename') for c in t['reference_evidence'] if c.get('source',{}).get('filename'))
    for g in gold:
        sources=actual.get(g['id']) or {c['doc_file'] for c in g['gold_citations']} or {'__missing_reference__'}
        for source in sources:
            q=first.setdefault(source,g['id']);parent[root(g['id'])]=root(q)
    members={}
    for q in parent:members.setdefault(root(q),[]).append(q)
    group_ids={r:core.digest(sorted(qs)) for r,qs in members.items()}
    groups={q:group_ids[root(q)] for q in parent}
    value={'gold_sha256':core.sha(home/'data/gold/gold_v2.jsonl'),'method':'connected questions sharing any actual judging reference source document (including frozen context for unanswerable tasks); unresolved references share one conservative group',
           'groups':groups,'n_groups':len(set(groups.values()))}
    target=directory/'evidence_groups.json'
    if target.exists():
        if json.loads(target.read_text())!=value:raise ValueError('Existing evidence grouping differs')
    else:
        with target.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
        os.chmod(target,0o600)
    print('Evidence dependence groups:',value['n_groups'])


def reference_package(g,chunks,context_chunks):
    reference=[];seen=set();complete=bool(g['gold_citations'])
    for cite in g['gold_citations']:
        matches=[c for c in chunks if c['meta']['filename']==cite['doc_file']
                 and str(c['meta']['section_id'])==str(cite['section'])]
        if not matches:complete=False
        for c in matches:
            if c['id'] not in seen:
                seen.add(c['id']);reference.append({'text':c['text'],'source':c['meta']})
    if not g['answerable'] and not reference:
        reference=[{'text':c['text'],'source':c.get('meta',{}),
                    'scope':'frozen policy context for a verified unanswerable task; not a positive gold citation'} for c in context_chunks]
        complete=bool(reference) and g.get('review_status') in ('council_verified','council_adjudicated')
    return reference,complete


def csv_write(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(dict.fromkeys(k for row in rows for k in row)) if rows else ['evaluation_id'])
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore',lineterminator='\n'); w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()})


def full_correct_change(pairs):
    scored=[(g,None if a is None else int(a==1),None if b is None else int(b==1)) for g,a,b in pairs]
    result=sj.paired_bootstrap(scored)
    return {'fully_correct_difference_pp':100*result['difference'] if result['difference'] is not None else None,
            'fully_correct_ci_low_pp':100*result['ci_low'] if result['ci_low'] is not None else None,
            'fully_correct_ci_high_pp':100*result['ci_high'] if result['ci_high'] is not None else None}


def prepare(home, output, new_predictions=None, exclude_history=False):
    home = Path(home).resolve(); out = core.private_output(output); os.chmod(out, 0o700)
    gold = {g['id']: g for g in core.benchmark(home) if g['split']=='test' and g['line']=='motor'}
    if len(gold) != 76 or sum(g['answerable'] for g in gold.values()) != 67:
        raise ValueError('Motor population mismatch')
    contexts_path = home/'experiments/20261007/test-eval/contexts.jsonl'
    contexts = core.unique(core.rows(contexts_path), 'qid')
    chunks_path = home/'data/chunks.jsonl'; chunks = core.rows(chunks_path)
    source = home/'experiments/2026-10-20/audit/paired_results.csv'
    with source.open() as f:
        history = list(csv.DictReader(f))
    historical = [r for r in history if r['question_id'] in gold]
    if len(historical) != 608 or len({(r['question_id'],r['condition']) for r in historical}) != 608:
        raise ValueError('Incomplete historical motor responses')
    rows = []
    cfg = core.config()['models']
    if exclude_history and not new_predictions:
        raise ValueError('A new-prediction source is required when excluding history')
    for r in ([] if exclude_history else historical):
        rows.append({'qid': r['question_id'], 'model': r['model'], 'condition': r['condition'],
                     'revision': cfg[r['model']]['revision'], 'model_id': cfg[r['model']]['id'],
                     'experiment': 'historical_sft21', 'answer': r['prediction'],
                     'evidence_available': r['condition'].endswith('openbook'), 'output_complete': True,
                     'citation_resolution': r['citation_resolution'] or None,
                     'parse_success': r['parse_success']=='True', 'latency_seconds': r['latency_seconds']})
    if new_predictions:
        for r in core.rows(secure(new_predictions)):
            rows.append({'qid': r['qid'], 'model': r['model']['id'], 'model_id': r['model']['id'],
                         'revision': r['model']['revision'], 'condition': r['mode'], 'experiment': 'six_model_base_v1',
                         'answer': r['answer'], 'evidence_available': r['mode']=='openbook',
                         'output_complete': r['final_answer_present'] and not r['hit_output_cap'],
                         'citation_resolution': r['resolved_citation_count']/r['citation_count'] if r['citation_count'] else None,
                         'parse_success': r['parse_success'], 'latency_seconds': r['latency_seconds']})
    ab = core.load_module('_semantic_abstention', 'eval/abstention.py')
    packages = []
    for r in rows:
        g=gold[r['qid']]
        reference,complete=reference_package(g,chunks,contexts[r['qid']]['chunks'])
        abst = ab.is_abstention_assertion(r['answer'], abstain_sentence=core.common.ABSTAIN_SENTENCE,
                                         citation_re=core.common.CITATION_RE)
        citations = core.rag.verify_citations(r['answer'],contexts[r['qid']]['chunks']) if r['evidence_available'] else {'n_total':0,'n_supported':0}
        packages.append({**{k:v for k,v in r.items() if k not in ('answer','evidence_available')},
            'citation_count':citations['n_total'] if r['evidence_available'] else None,
            'resolved_citation_count':citations['n_supported'] if r['evidence_available'] else None,
            'question_type': g['question_type'], 'question_group': core.digest(g['question'].strip().casefold()),
            'gold_answerable':g['answerable'], 'question':g['question'], 'reference_answer':g['gold_answer'],
            'key_facts':g['key_facts'], 'must_not_assert':g.get('must_not_assert',''),
            'reference_evidence':reference, 'reference_complete':complete,
            'student_evidence':contexts[r['qid']]['chunks'] if r['evidence_available'] else [],
            'student_evidence_available':r['evidence_available'], 'model_answer':r['answer'],
            'citations': [m.group(0) for m in core.common.CITATION_RE.finditer(r['answer'])],
            'deterministic_abstention':abst})
    tasks, mapping = sj.blind(packages, 20261020)
    core.write_rows(out/'judge_inputs.jsonl', tasks); core.write_rows(out/'private_mapping.jsonl', mapping)
    for filename in ['judge_votes.jsonl', 'judge_attempts.jsonl']:
        (out/filename).touch(mode=0o600)
    core.write_json(out/'judge_manifest.json', {'rubric_version':sj.VERSION,'prompt_sha256':core.digest(sj.PROMPT),
        'shuffle_seed':20261020,'population':len(tasks),'historical_responses':0 if exclude_history else 608,
        'historical_nonabstaining_answerable':sum(t['gold_answerable'] and not t['deterministic_abstention'] for t,m in zip(tasks,mapping) if m['experiment']=='historical_sft21'),
        'required_votes':3*len(tasks),'gold_sha256':core.sha(home/'data/gold/gold_v2.jsonl'),
        'corpus_sha256':core.sha(chunks_path),'contexts_sha256':core.sha(contexts_path),
        'historical_source_sha256':core.sha(source),'inputs_sha256':core.sha(out/'judge_inputs.jsonl'),
        'mapping_sha256':core.sha(out/'private_mapping.jsonl'),'reference_package_version':'frozen-policy-scope-v2.2.1',
        'new_predictions_sha256':core.sha(new_predictions) if new_predictions else None,
        'historical_output_completeness':'unknown; historical inference did not record output-cap status',
        'human_review_completed':0,'provider':'not_executed','paid_api_enabled':False})
    # Stratify independently of outcomes: both answerability classes, all arms.
    sample=[]; strata={}
    for t,m in zip(tasks,mapping):
        key=(m['model'],m['condition'],t['gold_answerable'])
        strata.setdefault(key,[]).append(t)
    while len(sample)<28 and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(sample)<28: sample.append(strata[key].pop(0))
    csv_write(out/'human_review_sample.csv', [{**t, 'human_correctness':'','human_evidence_support':'','human_reason':''} for t in sample])
    for p in out.iterdir(): os.chmod(p,0o600)
    print('Prepared blinded responses:',len(tasks),'required votes:',3*len(tasks))


def local_generator(model_dir):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    p=Path(model_dir)
    # HF download metadata starts with the pinned commit, not JSON.
    pin=(p/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]
    if pin!=REVISION: raise ValueError('Local judge revision differs from pinned checkpoint')
    if any((p/'.cache/huggingface/download'/str(w.name+'.metadata')).read_text().splitlines()[0]!=REVISION for w in p.glob('*.safetensors')):
        raise ValueError('Judge weights revision differs from pinned checkpoint')
    model,tokenizer=load(str(p),tokenizer_config={'trust_remote_code':False})
    def call(task,seed):
        mx.random.seed(seed)
        prompt=tokenizer.apply_chat_template([{'role':'user','content':sj.PROMPT+'\nTASK:\n'+json.dumps(task,ensure_ascii=False)}], tokenize=False,add_generation_prompt=True)
        n=len(tokenizer.encode(prompt))
        if n+1200>16384: raise ValueError('Judge context budget exceeded; no truncation')
        start=time.monotonic()
        answer=generate(model,tokenizer,prompt=prompt,max_tokens=1200,sampler=make_sampler(temp=.2),verbose=False)
        return answer,{'input_tokens':n,'output_tokens':len(tokenizer.encode(answer)),
                       'latency_seconds':time.monotonic()-start,'estimated_cost_usd':0}
    return call


def calibrate(model_dir, output):
    out=core.private_output(output); call=local_generator(model_dir); results=[]
    for task,expected in fixtures():
        raw,usage=call(task,20261020); error=None; vote=None
        try: vote=sj.validate(raw,task)
        except (ValueError,TypeError,KeyError) as e: error=str(e)
        results.append({'evaluation_id':task['evaluation_id'],'expected_correctness':expected,
                        'passed':vote is not None and vote['correctness']==expected,
                        'raw':raw,'vote':vote,'error':error,**usage})
        core.write_json(out/'calibration.json',{'model':MODEL,'revision':REVISION,'rubric_version':sj.VERSION,
             'prompt_sha256':core.digest(sj.PROMPT),'passed':len(results)==len(fixtures()) and all(r['passed'] for r in results),'examples':results})
        print(task['evaluation_id'],'pass' if results[-1]['passed'] else 'fail',flush=True)


def run(directory, model_dir, calibration, limit=24):
    directory=secure(directory); cal=json.loads(secure(calibration).read_text())
    if not cal['passed'] or cal['revision']!=REVISION or cal['prompt_sha256']!=core.digest(sj.PROMPT):
        raise ValueError('Fictional calibration gate has not passed')
    tasks=core.rows(directory/'judge_inputs.jsonl')
    if core.sha(directory/'judge_inputs.jsonl')!=json.loads((directory/'judge_manifest.json').read_text())['inputs_sha256']:
        raise ValueError('Judging inputs changed')
    previous=core.rows(directory/'judge_votes.jsonl'); done={(r['evaluation_id'],r['vote']) for r in previous}
    if len(done)!=len(previous): raise ValueError('Duplicate persisted vote')
    call=local_generator(model_dir); attempts=0
    for task in tasks:
        for vote_index in range(1,4):
            if (task['evaluation_id'],vote_index) in done: continue
            if attempts>=limit:return
            attempts+=1; seed=20261020+vote_index
            raw,usage=call(task,seed); record={'evaluation_id':task['evaluation_id'],'vote':vote_index,
                'seed':seed,'provider':'local_mlx','model':MODEL,'revision':REVISION,**usage,'raw':raw}
            try:
                v=sj.validate(raw,task);record['valid']=True
                with (directory/'judge_votes.jsonl').open('a') as f:f.write(json.dumps({**v,**{k:v for k,v in record.items() if k!='raw'}})+'\n')
            except (ValueError,KeyError,TypeError) as e:record.update(valid=False,error=str(e))
            with (directory/'judge_attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(task['evaluation_id'],vote_index,record['valid'],flush=True)


def run_gemini(directory, permission, free_receipt, shared_usage, catalogue,
               env_file, model='gemini-3.5-flash-lite', limit=24, question_limit=76,
               reserve_requests=50, wait_for_minute_quota=False):
    """Requires a separately recorded full-payload permission; never infer one."""
    from eval.gemini_judge import GeminiJudge, QuotaError, InvalidOutput, REQUIRED_FIELDS, PURPOSE
    if not 1<=question_limit<=76:raise ValueError('Question limit must be 1–76')
    permission=json.loads(Path(permission).read_text())
    if not sj.permission_allows(permission,'gemini',REQUIRED_FIELDS,PURPOSE):
        raise PermissionError('Full-payload semantic permission required before key loading')
    directory=secure(directory);free=json.loads(Path(free_receipt).read_text())
    free=free.get('gemini',free); key=None
    for line in Path(env_file).read_text().splitlines():
        if line.startswith('GEMINI_API_KEY='):key=line.split('=',1)[1].strip().strip('"').strip("'")
    if not key:raise ValueError('Missing GEMINI_API_KEY')
    client=GeminiJudge(key,model,permission,free,directory/'gemini_usage.jsonl',Path(shared_usage),json.loads(Path(catalogue).read_text()),reserve_requests=reserve_requests)
    tasks=core.rows(directory/'judge_inputs.jsonl')
    manifest=json.loads((directory/'judge_manifest.json').read_text())
    if core.sha(directory/'judge_inputs.jsonl')!=manifest['inputs_sha256']:raise ValueError('Changed judge inputs')
    # API/schema smoke validation is distinct from reliability calibration. All
    # resulting scores remain unvalidated until independent review is supplied.
    manifest.update(provider='gemini',model=model,permission_sha256=core.digest(permission),
                    free_verification_sha256=core.digest(free),paid_api_enabled=False)
    core.write_json(directory/'judge_manifest.json',manifest)
    mapping=core.unique(core.rows(directory/'private_mapping.jsonl'),'evaluation_id')
    # Choose the matched pilot BEFORE any scores. Include unanswerable questions,
    # then deterministic shuffled answerable questions; every selected qid has
    # all eight historical arms. New models stay in the same frozen input batch
    # but are deferred until historical coverage is complete.
    import random
    rng=random.Random(20261020);answerable={};unanswerable={}
    for t in tasks:
        m=mapping[t['evaluation_id']]
        if m['experiment']=='historical_sft21':
            (answerable if t['gold_answerable'] else unanswerable)[m['qid']]=True
    a=sorted(answerable);u=sorted(unanswerable);rng.shuffle(a);rng.shuffle(u)
    chosen=u[:min(2,question_limit)]+a[:max(0,question_limit-2)]
    if question_limit==76:chosen=u+a
    rank={q:i for i,q in enumerate(chosen)}
    tasks=[t for t in tasks if mapping[t['evaluation_id']]['experiment']=='historical_sft21' and mapping[t['evaluation_id']]['qid'] in rank]
    tasks.sort(key=lambda t:(rank[mapping[t['evaluation_id']]['qid']],mapping[t['evaluation_id']]['condition']))
    manifest.update(gemini_question_selection_seed=20261020,gemini_matched_question_limit=question_limit,
                    gemini_production_request_reserve=reserve_requests,gemini_selected_qids=chosen)
    core.write_json(directory/'judge_manifest.json',manifest)
    previous=core.rows(directory/'judge_votes.jsonl');done={(v['evaluation_id'],v['vote']) for v in previous}
    if len(done)!=len(previous):raise ValueError('Duplicate persisted vote')
    attempts=0
    for task in tasks:
        # Historical primary population precedes new-model extension without
        # exposing this scheduling metadata to the provider.
        for vote_index in range(1,4):
            if (task['evaluation_id'],vote_index) in done:continue
            if attempts>=limit:return
            seed=20261020+vote_index;record={'evaluation_id':task['evaluation_id'],'vote':vote_index,'seed':seed,'provider':'gemini','model':model}
            record['response_schema_sha256']=core.digest(sj.response_schema(task))
            record['rubric_version']=sj.VERSION
            record['prompt_sha256']=core.digest(sj.PROMPT)
            try:
                while True:
                    try:
                        raw,usage=client.call(task,seed);break
                    except QuotaError as e:
                        if not wait_for_minute_quota or e.retry_after is None:raise
                        print('Waiting for free minute quota',round(e.retry_after,1),flush=True)
                        time.sleep(min(e.retry_after,60))
                v=sj.validate(raw,task)
                record.update(valid=True,raw=raw,input_tokens=usage['usage'].get('promptTokenCount',0),
                    output_tokens=usage['usage'].get('candidatesTokenCount',0),latency_seconds=usage['latency_seconds'],estimated_cost_usd=0)
                with (directory/'judge_votes.jsonl').open('a') as f:f.write(json.dumps({**v,**{k:v for k,v in record.items() if k!='raw'}})+'\n')
            except QuotaError as e:
                print(str(e));return
            except InvalidOutput as e:
                record.update(valid=False,error=str(e),raw=e.raw,
                    input_tokens=e.usage['usage'].get('promptTokenCount',0),output_tokens=e.usage['usage'].get('candidatesTokenCount',0))
            except (ValueError,RuntimeError) as e:
                record.update(valid=False,error=str(e),input_tokens=0,output_tokens=0)
            attempts+=1
            with (directory/'judge_attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(task['evaluation_id'],vote_index,record['valid'],flush=True)


def aggregate(directory, public):
    directory=secure(directory);tasks=core.rows(directory/'judge_inputs.jsonl');mapping=core.unique(core.rows(directory/'private_mapping.jsonl'),'evaluation_id')
    initial_manifest=json.loads((directory/'judge_manifest.json').read_text())
    if initial_manifest.get('mapping_sha256') and core.sha(directory/'private_mapping.jsonl')!=initial_manifest['mapping_sha256']:
        raise ValueError('Private identity mapping changed')
    votes=core.rows(directory/'judge_votes.jsonl');groups={}; seen=set()
    by_task=core.unique(tasks,'evaluation_id')
    source_groups={}
    if (directory/'evidence_groups.json').exists():
        source_receipt=json.loads((directory/'evidence_groups.json').read_text())
        source_groups=source_receipt['groups']
        if source_receipt['gold_sha256']!=json.loads((directory/'judge_manifest.json').read_text())['gold_sha256']:
            raise ValueError('Source groups use a different benchmark')
    for v in votes:
        unit=(v['evaluation_id'],v['vote'])
        if unit in seen or v['evaluation_id'] not in by_task: raise ValueError('Duplicate or unknown persisted vote')
        seen.add(unit);sj.validate(json.dumps({k:v[k] for k in sj.FIELDS}),by_task[v['evaluation_id']])
        groups.setdefault(v['evaluation_id'],[]).append(v)
    primary=[]; preliminary=[];per=[]
    for t in tasks:
        vs=groups.get(t['evaluation_id'],[]); merged=sj.merge(vs,t,judge_plan=initial_manifest.get('judge_plan')) if vs else None
        first=[v for v in vs if v['vote']==1]
        one=sj.merge(first,t,required=1,judge_plan=initial_manifest.get('judge_plan')) if first else None
        base={**mapping[t['evaluation_id']],'answerable':t['gold_answerable'],
              'abstention':t['deterministic_abstention'],'student_evidence_available':t['student_evidence_available']}
        per.append({**base,**(merged or {'correctness':None,'evidence_support':None,'n_votes':len(vs)}),
                    'status':'three_vote_complete' if merged else 'pending'})
        if merged:primary.append({**base,**merged})
        if one:preliminary.append({**base,**one})
    csv_write(directory/'per_question_results.csv',per);csv_write(directory/'judge_results.csv',primary)
    csv_write(directory/'preliminary_single_vote_results.csv',preliminary)
    csv_write(directory/'judge_disagreements.csv',[r for r in primary if r['disagreement'] or r['needs_human_review']])
    summaries=[];grounding=[]
    for key in sorted({(r['experiment'],r['model'],r['condition']) for r in per}):
        rs=[r for r in per if (r['experiment'],r['model'],r['condition'])==key]
        cs=[r['correctness'] for r in rs if r['correctness'] is not None]; es=[r['evidence_support'] for r in rs if r['evidence_support'] is not None]
        a=[r for r in rs if r['answerable']];u=[r for r in rs if not r['answerable']]
        summaries.append({'experiment':key[0],'model':key[1],'model_id':rs[0]['model_id'],'revision':rs[0]['revision'],'condition':key[2],'population_n':len(rs),
            'judged_n':len(cs),'pending_or_uncertain_n':len(rs)-len(cs),'mean_correctness':statistics.mean(cs) if cs else None,
            'fully_correct_pct':100*cs.count(1)/len(cs) if cs else None,'partial_pct':100*cs.count(.5)/len(cs) if cs else None,
            'incorrect_pct':100*cs.count(0)/len(cs) if cs else None,'unnecessary_refusal_n':sum(r['abstention'] for r in a),
            'answerable_n':len(a),'unnecessary_refusal_rate':sum(r['abstention'] for r in a)/len(a),
            'correct_refusal_n':sum(r['abstention'] for r in u),'unanswerable_n':len(u),
            'correct_refusal_rate':sum(r['abstention'] for r in u)/len(u),
            'balanced_abstention':.5*(sum(not r['abstention'] for r in a)/len(a)+sum(r['abstention'] for r in u)/len(u)),
            'citation_n':sum(r.get('citation_count') or 0 for r in rs),
            'citation_resolution':sum(r.get('resolved_citation_count') or 0 for r in rs)/sum(r.get('citation_count') or 0 for r in rs) if sum(r.get('citation_count') or 0 for r in rs) else None,
            'mean_latency_seconds':statistics.mean(float(r['latency_seconds']) for r in rs),
            'parse_rate':statistics.mean(r['parse_success'] for r in rs),'status':'unjudged' if not cs else 'preliminary_unvalidated'})
        scored=[r for r in rs if r.get('protocol')=='three_vote' and r['correctness'] is not None
                and (r['evidence_support'] is not None or r['abstention'])]
        grounding.append({'experiment':key[0],'model':key[1],'model_id':rs[0]['model_id'],'revision':rs[0]['revision'],'condition':key[2],'population_n':len(rs),
                          'reference_support_judged_n':len(es),'mean_reference_support':statistics.mean(es) if es else None,
                          'unsupported_claim_judged_n':len(scored),
                          'unsupported_material_claim_rate':statistics.mean(r['unsupported_material_claim'] for r in scored) if scored else None,
                          'student_citation_grounding':None,'student_grounding_status':'separate_claim_citation_review_pending' if rs[0]['student_evidence_available'] else 'not_applicable_no_student_evidence'})
    comparisons=[];paired_private=[];decomposition=[]
    for model in ('qwen3b','ministral3b'):
        for before,after in [('base-closedbook','sft-closedbook'),('base-openbook','sft-openbook'),('base-closedbook','base-openbook'),('sft-closedbook','sft-openbook')]:
            aa={r['qid']:r for r in per if r['model']==model and r['condition']==model+'-'+before}
            bb={r['qid']:r for r in per if r['model']==model and r['condition']==model+'-'+after}
            pairs=[(aa[q]['question_group'],aa[q]['correctness'],bb[q]['correctness']) for q in sorted(aa.keys()&bb.keys())]
            comparisons.append({'model':model,'before':before,'after':after,'eligible_pairs':len(pairs),**sj.paired_bootstrap(pairs),
                                **full_correct_change(pairs),
                                'status':'pending' if not any(a is not None and b is not None for _,a,b in pairs) else 'partial_unvalidated'})
            paired_private.extend({'model':model,'qid':q,'before':before,'after':after,'before_correctness':aa[q]['correctness'],
                 'after_correctness':bb[q]['correctness'],'difference':bb[q]['correctness']-aa[q]['correctness'] if aa[q]['correctness'] is not None and bb[q]['correctness'] is not None else None} for q in sorted(aa.keys()&bb.keys()))
            if source_groups:
                sensitivity=sj.paired_bootstrap([(source_groups[q],aa[q]['correctness'],bb[q]['correctness']) for q in sorted(aa.keys()&bb.keys())])
                comparisons[-1].update(source_cluster_n=sensitivity.get('n_groups'),source_cluster_ci_low=sensitivity['ci_low'],source_cluster_ci_high=sensitivity['ci_high'])
            for population,questions in [('all_motor',aa.keys()&bb.keys()),
                ('answerable',{q for q in aa.keys()&bb.keys() if aa[q]['answerable']}),
                ('answerable_both_attempted',{q for q in aa.keys()&bb.keys() if aa[q]['answerable'] and not aa[q]['abstention'] and not bb[q]['abstention']})]:
                values=[(aa[q]['question_group'],aa[q]['correctness'],bb[q]['correctness']) for q in sorted(questions)]
                row={'model':model,'before':before,'after':after,'population':population,'eligible_pairs':len(questions),
                     **sj.paired_bootstrap(values),**full_correct_change(values),'status':'preliminary_unvalidated'}
                if source_groups:
                    sc=sj.paired_bootstrap([(source_groups[q],aa[q]['correctness'],bb[q]['correctness']) for q in sorted(questions)])
                    row.update(source_cluster_n=sc.get('n_groups'),source_cluster_ci_low=sc['ci_low'],source_cluster_ci_high=sc['ci_high'])
                decomposition.append(row)
    public=Path(public);public.mkdir(parents=True,exist_ok=True)
    csv_write(public/'correctness_summary.csv',summaries);csv_write(public/'grounding_summary.csv',grounding)
    csv_write(public/'sft_rag_comparison.csv',comparisons)
    csv_write(public/'sft_accuracy_decomposition.csv',decomposition)
    strata=[]
    for protocol,items in [('three_vote',primary),('preliminary_vote_1',preliminary)]:
        for key in sorted({(r['experiment'],r['model'],r['condition']) for r in per}):
            for population in ('answerable','unanswerable','answerable_attempts'):
                eligible=[r for r in per if (r['experiment'],r['model'],r['condition'])==key and
                    ((not r['answerable']) if population=='unanswerable' else r['answerable'] and (population!='answerable_attempts' or not r['abstention']))]
                ids={r['evaluation_id'] for r in eligible}
                scores=[r['correctness'] for r in items if r['evaluation_id'] in ids and r['correctness'] is not None]
                strata.append({'experiment':key[0],'model':key[1],'condition':key[2],'protocol':protocol,'population':population,
                               'eligible_n':len(eligible),'scored_n':len(scores),'mean_correctness':statistics.mean(scores) if scores else None,
                               'status':'unjudged' if not scores else 'preliminary_unvalidated'})
    csv_write(public/'answerability_correctness_summary.csv',strata)
    csv_write(directory/'paired_question_differences.csv',paired_private)
    type_summary=[]
    for key in sorted({(r['experiment'],r['model'],r['condition'],r.get('question_type','unspecified')) for r in per}):
        rs=[r for r in per if (r['experiment'],r['model'],r['condition'],r.get('question_type','unspecified'))==key]
        values=[r['correctness'] for r in rs if r['correctness'] is not None]
        type_summary.append({'experiment':key[0],'model':key[1],'condition':key[2],'question_type':key[3],
            'population_n':len(rs),'judged_n':len(values),'mean_correctness':statistics.mean(values) if values else None,
            'status':'unjudged' if not values else 'preliminary_unvalidated'})
    csv_write(public/'question_type_summary.csv',type_summary)
    # Separate matched pilot tables use the SAME observed question IDs across
    # all eight historical arms. Never compare different complete-case samples.
    pilot_manifest={}
    historical_keys=sorted({r['condition'] for r in per if r['experiment']=='historical_sft21'})
    for protocol,items,stem in [('three_vote',primary,'three_vote'),('preliminary_single_vote',preliminary,'single_vote')]:
        eligible={c:{r['qid'] for r in items if r['experiment']=='historical_sft21' and r['condition']==c and r['correctness'] is not None} for c in historical_keys}
        common=set.intersection(*eligible.values()) if len(eligible)==8 else set()
        pilot_manifest[protocol]={'matched_questions':len(common),'question_ids_sha256':core.digest(sorted(common)),
             'population':'same complete-case questions across all eight historical arms; not the full frozen benchmark',
             'human_validated':False}
        ps=[]
        for c in historical_keys:
            rs=[r for r in items if r['experiment']=='historical_sft21' and r['condition']==c and r['qid'] in common]
            supports=[r['evidence_support'] for r in rs if r['evidence_support'] is not None]
            a=[r for r in rs if r['answerable']];u=[r for r in rs if not r['answerable']]
            claims=[r for r in rs if r['evidence_support'] is not None or r['abstention']]
            ps.append({'experiment':'historical_sft21_matched_pilot','model':c.split('-')[0],'condition':c,
                'protocol':protocol,'population_n':len(rs),'judged_n':len(rs),'answerable_n':len(a),'unanswerable_n':len(u),
                'mean_correctness':statistics.mean(r['correctness'] for r in rs) if rs else None,
                'mean_reference_support':statistics.mean(supports) if supports else None,'reference_support_judged_n':len(supports),
                'unsupported_material_claim_rate':statistics.mean(r['unsupported_material_claim'] for r in claims) if claims else None,
                'unsupported_claim_judged_n':len(claims),
                'unnecessary_refusal_rate':statistics.mean(r['abstention'] for r in a) if a else None,
                'correct_refusal_rate':statistics.mean(r['abstention'] for r in u) if u else None,
                'human_validated':False,'status':'preliminary_selected_complete_cases'})
        csv_write(public/f'matched_{stem}_summary.csv',ps)
        pc=[]
        for model in ('qwen3b','ministral3b'):
            for before,after in [('base-closedbook','sft-closedbook'),('base-openbook','sft-openbook'),('base-closedbook','base-openbook'),('sft-closedbook','sft-openbook')]:
                aa={r['qid']:r for r in items if r['model']==model and r['condition']==model+'-'+before and r['qid'] in common}
                bb={r['qid']:r for r in items if r['model']==model and r['condition']==model+'-'+after and r['qid'] in common}
                pairs=[(aa[q]['question_group'],aa[q]['correctness'],bb[q]['correctness']) for q in sorted(common)]
                pc.append({'model':model,'before':before,'after':after,'protocol':protocol,**sj.paired_bootstrap(pairs),
                           **full_correct_change(pairs),
                           'status':'preliminary_selected_complete_cases','human_validated':False})
                if source_groups:
                    sensitivity=sj.paired_bootstrap([(source_groups[q],aa[q]['correctness'],bb[q]['correctness']) for q in sorted(common)])
                    pc[-1].update(source_cluster_n=sensitivity.get('n_groups'),source_cluster_ci_low=sensitivity['ci_low'],source_cluster_ci_high=sensitivity['ci_high'])
        csv_write(public/f'matched_{stem}_comparison.csv',pc)
    core.write_json(public/'matched_pilot_manifest.json',pilot_manifest)
    attempts=core.rows(directory/'judge_attempts.jsonl');manifest=json.loads((directory/'judge_manifest.json').read_text())
    requests=core.rows(directory/'judge_requests.jsonl') if (directory/'judge_requests.jsonl').exists() else []
    adopted_requests=core.rows(directory/'adopted_request_receipts.jsonl') if (directory/'adopted_request_receipts.jsonl').exists() else []
    all_requests=adopted_requests+requests
    reported_usage=[r['usage']['usage'] for r in all_requests if r.get('usage') and type(r['usage'].get('usage',{}).get('totalTokenCount')) is int]
    human=[];human_pairs=[];primary_map={r['evaluation_id']:r for r in primary}
    with (directory/'human_review_sample.csv').open() as f:
        for r in csv.DictReader(f):
            if r.get('human_correctness','').strip() and r.get('human_reason','').strip():
                score=float(r['human_correctness'])
                if score not in sj.SCORES:raise ValueError('Invalid human label')
                human.append(r)
                merged=primary_map.get(r['evaluation_id'])
                if merged and merged['correctness'] is not None:human_pairs.append((int(score*2),int(merged['correctness']*2)))
    agreement=None
    if human_pairs:
        hm=core.load_module('_semantic_human_agreement','eval/human_agreement.py')
        agreement={'paired_n':len(human_pairs),'quadratic_weighted_kappa':hm.quadratic_weighted_kappa(human_pairs) if len({x for pair in human_pairs for x in pair})>1 else None}
    consistency=[]
    for va,vb in ((1,2),(1,3),(2,3)):
        pairs=[]
        for eid,vs in groups.items():
            by_vote={v['vote']:v for v in vs}
            if va in by_vote and vb in by_vote and by_vote[va]['correctness'] is not None and by_vote[vb]['correctness'] is not None:
                pairs.append((int(2*by_vote[va]['correctness']),int(2*by_vote[vb]['correctness'])))
        kappa=None
        if pairs and len({x for pair in pairs for x in pair})>1:
            hm=core.load_module('_semantic_repeat_agreement','eval/human_agreement.py')
            kappa=hm.quadratic_weighted_kappa(pairs)
        consistency.append({'vote_a':va,'vote_b':vb,'paired_scored_n':len(pairs),
            'exact_agreement':statistics.mean(a==b for a,b in pairs) if pairs else None,'quadratic_weighted_kappa':kappa})
    validation={'rubric_version':sj.VERSION,'population_n':len(tasks),'valid_votes':len(votes),'attempts':len(attempts),
                'three_vote_completed_responses':len(primary),'single_vote_responses':len(preliminary),
                'three_vote_scored_responses':sum(r['correctness'] is not None for r in primary),
                'single_vote_scored_responses':sum(r['correctness'] is not None for r in preliminary),
                'generation_requests':len(all_requests),'adopted_source_generation_requests':len(adopted_requests),
                'requests_without_token_receipt':len(all_requests)-len(reported_usage),
                'token_usage_complete':len(all_requests)==len(reported_usage),
                'source_dependence_groups':len(set(source_groups.values())) if source_groups else None,
                'within_judge_repeated_vote_agreement':consistency,
                'within_judge_fact_disagreement_n':sum(r['n_facts_disagreed'] for r in primary),
                'within_judge_fact_n':sum(len(r['fact_scores']) for r in primary),
                'invalid_attempts':sum(not a['valid'] for a in attempts),'human_review_completed':len(human),'human_agreement':agreement,
                'three_vote_disagreements':sum(r['disagreement'] for r in primary),'paid_api_enabled':False,
                'input_tokens':sum(a.get('input_tokens') or 0 for a in attempts),'output_tokens':sum(a.get('output_tokens') or 0 for a in attempts),
                'thinking_tokens':sum(a.get('thinking_tokens') or 0 for a in attempts),'reported_total_tokens':sum(a.get('total_tokens') or 0 for a in attempts),
                'estimated_cost_usd':0,'rankings_validated':False,'citation_grounding':'not_judged',
                'uncertainty':'Only nine unanswerable questions; source-document connected-component bootstrap sensitivity is reported separately where available; few independent sources limit inference.'}
    if all_requests:
        validation.update(input_tokens=sum(u.get('promptTokenCount',0) for u in reported_usage),
                          output_tokens=sum(u.get('candidatesTokenCount',0) for u in reported_usage),
                          thinking_tokens=sum(u.get('thoughtsTokenCount',0) for u in reported_usage),
                          reported_total_tokens=sum(u.get('totalTokenCount',0) for u in reported_usage),
                          token_accounting_scope='Current dispatch and adopted source-run requests; fictional calibration and other engineering trials are separate.')
    versions={'gemini-3.5-flash-lite':'3.5-flash-lite-07-2026','gemini-3.1-flash-lite':'3.1-flash-lite-05-2026'}
    validation.update(judge_provider=manifest.get('provider'),judge_model=manifest.get('model'),
                      judge_model_catalogue_version=manifest.get('model_catalogue_version',versions.get(manifest.get('model'))),thinking_level=manifest.get('thinking_level'),
                      transport_version=manifest.get('transport_version','indexed-pointwise-v3'),
                      permission_status='owner_authorised_full_semantic_payload')
    core.write_json(public/'judge_validation_summary.json',validation)
    core.write_json(directory/'judge_manifest.json',{**manifest,**validation})
    (directory/'judge_run_report.md').write_text('# Semantic judge run\n\n'+json.dumps(validation,indent=2)+'\n\nMissing scores remain pending. No superiority claim is supported.\n')
    print('Three-vote responses:',len(primary),'valid votes:',len(votes))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--home',required=True);q.add_argument('--output',required=True);q.add_argument('--new-predictions');q.add_argument('--exclude-history',action='store_true')
    q=sub.add_parser('calibrate');q.add_argument('--model-dir',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('run');q.add_argument('--directory',required=True);q.add_argument('--model-dir',required=True);q.add_argument('--calibration',required=True);q.add_argument('--limit',type=int,default=24)
    q=sub.add_parser('aggregate');q.add_argument('--directory',required=True);q.add_argument('--public',required=True)
    q=sub.add_parser('evidence-groups');q.add_argument('--home',required=True);q.add_argument('--directory',required=True)
    q=sub.add_parser('run-gemini')
    for name in ('directory','permission','free-receipt','shared-usage','catalogue','env-file'):q.add_argument('--'+name,required=True)
    q.add_argument('--model',default='gemini-3.5-flash-lite');q.add_argument('--limit',type=int,default=24)
    q.add_argument('--question-limit',type=int,default=76);q.add_argument('--reserve-requests',type=int,default=50)
    q.add_argument('--wait-for-minute-quota',action='store_true')
    a=vars(p.parse_args());command=a.pop('command');globals()[command.replace('-','_')](**a)
