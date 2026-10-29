"""Two blinded independent judges, original semantic rubric, deterministic citations."""
import asyncio
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
from experiments import core
from eval import semantic_judge as sj
from eval import judge_span_ids as indexed
from cloud.d1000.export import ARCHIVE,sha

EXTENSION='''
ADDITIONAL SEPARATE CLAIM-TO-CITATION AUDIT v1:
For each student citation listed in citations, assess the substantive claim
attributed to it against student_evidence, independently of ID resolution.
Return citation_assessments with citation, answer_span (verbatim attributed
claim), evidence_span (verbatim from student_evidence or empty), support
(0=unsupported/contradicted,.5=partial,1=fully supported,null=uncertain).
Every citation must have one assessment; no citations or no supplied student
evidence means an empty list. Existing overall/reference rubric is unchanged.
'''


def schema(task):
    s=indexed.schema([task]);props=s['properties']['evaluations']['items']['properties']
    props['citation_assessments']={'type':'array','items':{'type':'object','properties':{
        'citation':{'type':'string'},'answer_span':{'type':'string'},'evidence_span':{'type':'string'},
        'support':{'type':['number','null'],'enum':[0,.5,1,None]}},
        'required':['citation','answer_span','evidence_span','support'],'additionalProperties':False}}
    s['properties']['evaluations']['items']['required']=sorted(props)
    return s


def normalise(value,task):
    value=dict(value);citations=value.pop('citation_assessments')
    v=indexed.normalise(value,task)
    expected=task['citations'] if task['student_evidence_available'] else []
    assert len(citations)==len(expected)
    assert sorted(c['citation'] for c in citations)==sorted(expected)
    evidence='\n'.join(c['text'] for c in task['student_evidence'])
    for c in citations:
        assert c['answer_span'] in task['model_answer'] and c['answer_span'].strip()
        assert c['evidence_span'] in evidence
        assert c['support'] in (0,.5,1,None)
        if c['support'] in (.5,1):assert c['evidence_span'].strip()
    return {**v,'citation_assessments':citations}


def tasks(root):
    from scripts.run_semantic_judge import reference_package
    ab=core.load_module('_d1000_abstention','eval/abstention.py')
    gold={g['id']:g for g in core.rows(root/'evaluation/gold.jsonl')}
    contexts={r['qid']:r for r in core.rows(root/'evaluation/contexts.jsonl')}
    chunks=core.rows(root/'evaluation/chunks.jsonl');pairs=[]
    for path in sorted((root/'runs').glob('*/*responses.jsonl')):
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue # only incomplete last append is deferred
            g=gold[r['qid']];rag=r['condition'].endswith('-rag');answer=r['answer']
            if re.search(r'(?i)\b(qwen|ministral|smollm|deepseek|llama|gemma)[\d.-]*\b',answer):continue
            reference,complete=reference_package(g,chunks,contexts[r['qid']]['chunks'])
            task={'evaluation_id':core.digest({'model':r['model'],'condition':r['condition'],'qid':r['qid']})[:24],
                'question':g['question'],'gold_answerable':g['answerable'],'reference_answer':g['gold_answer'],
                'key_facts':g['key_facts'],'must_not_assert':g.get('must_not_assert',''),
                'reference_evidence':reference,'reference_complete':complete,
                'student_evidence':contexts[r['qid']]['chunks'] if rag else [],'student_evidence_available':rag,
                'model_answer':answer,'citations':[m.group(0) for m in core.common.CITATION_RE.finditer(answer)],
                'deterministic_abstention':ab.is_abstention_assertion(answer,abstain_sentence=core.common.ABSTAIN_SENTENCE,citation_re=core.common.CITATION_RE),
                'output_complete':r['final_answer_present'] and r['output_tokens']<700}
            pairs.append((task,{'qid':r['qid'],'condition':r['condition'],'model':r['model']}))
    return pairs


def provider_runtime(root):
    """Pinned existing transport, independent caches, inherited account reservations."""
    mount=next(p.parent for p in Path('/kaggle/input').rglob('migration_manifest.json') if json.loads(p.read_text()).get('archive_sha256')==ARCHIVE)
    m=json.loads((mount/'migration_manifest.json').read_text());assert sha(mount/'checkpoint.bin')==ARCHIVE
    runtime=root/'judge_runtime';runtime.mkdir(exist_ok=True)
    code=root/'production_transport';code.mkdir(exist_ok=True);(code/'__init__.py').touch()
    with tarfile.open(mount/'checkpoint.bin') as tar:
        def read(name):
            b=tar.extractfile(name).read();assert __import__('hashlib').sha256(b).hexdigest()==m['files'][name];return b
        for name in ('providers.py','common.py','kelly.py'):
            (code/name).write_bytes(read('uk-insurance-QA-data/src/'+name))
        prefix='uk-insurance-QA-data-local/production-v1/'
        (runtime/'provider_verification.json').write_bytes(read(prefix+'provider_verification.json'))
        for provider in ('groq','gemini'):
            dest=runtime/'provider_usage'/(provider+'.json');dest.parent.mkdir(exist_ok=True)
            if not dest.exists():dest.write_bytes(read(prefix+'provider_usage/'+provider+'.json'))
    sys.path.insert(0,str(root));return importlib.import_module('production_transport.providers'),runtime


async def run(root,stop,state):
    root=Path(root);out=root/'judging';out.mkdir(exist_ok=True)
    providers,runtime=provider_runtime(root)
    models={'groq':'openai/gpt-oss-120b','gemini':'gemini-3.5-flash-lite'}
    values={k:os.environ.get(k,'') for k in ('GROQ_API_KEY','GEMINI_API_KEY')}
    # Separate account ledgers/caches per provider; single caller for each.
    deadline=time.monotonic()+9*3600
    async def worker(provider):
        status=state.setdefault('judges',{}).setdefault(provider,{'completed':0})
        try:
            if not values[provider.upper()+'_API_KEY']:raise RuntimeError('Notebook secret missing')
            providers.verify_receipt(runtime,provider,models[provider])
            cat,_=await asyncio.to_thread(providers.transport,values,provider)
            assert providers.catalogue_contains(provider,models[provider],cat)
            status['model']=models[provider];status['catalogue_verified']=True
        except Exception as e:status.update(status='blocked_preflight',reason=str(e)[:500]);return
        journal=out/(provider+'.jsonl');done={r['evaluation_id'] for r in core.rows(journal)} if journal.exists() else set()
        failed=set();calibrated=False
        while True:
            from eval.semantic_fixtures import fixtures
            pending=[]
            if not calibrated:
                fixture_path=out/(provider+'-calibration.json')
                if fixture_path.exists():
                    c=json.loads(fixture_path.read_text());calibrated=c['passed']
                    if not calibrated:status.update(status='calibration_failed');return
                else:pending=[(dict(t,citations=[],student_evidence_available=False),expected) for t,expected in fixtures()]
            if calibrated:pending=[(t,meta) for t,meta in tasks(root) if t['evaluation_id'] not in done|failed]
            if not pending:
                if stop.is_set():status['status']='complete_available';return
                await asyncio.sleep(20);continue
            calibration_results=[]
            for task,meta in pending:
                try:
                    payload=indexed.prompt([task])+EXTENSION
                    result=await asyncio.to_thread(providers.complete,values,provider,models[provider],
                        [{'role':'user','content':payload}],runtime,2048,schema(task),'d1000-independent-dual-judge/1')
                    raw=result['answer'];assert len(raw['evaluations'])==1
                    v=normalise(raw['evaluations'][0],task)
                    if not calibrated:
                        # Original fictional anchors; valid-but-wrong scores fail closed.
                        ok=v['correctness']==meta
                        calibration_results.append({'id':task['evaluation_id'],'passed':ok})
                    else:
                        with journal.open('a') as f:f.write(json.dumps({**v,'provider':provider,'model':models[provider],
                            'request_sha256':result['request_sha256'],'usage':result['usage'],'input_sha256':core.digest(task)})+'\n')
                        done.add(task['evaluation_id']);status.update(status='running',completed=len(done))
                except providers.QuotaPause as e:
                    status.update(status='quota_wait',resume_after_epoch=e.until)
                    delay=e.until-time.time()
                    if time.monotonic()+delay>deadline:
                        status['status']='quota_exceeds_session';return
                    # Supported same-session wait; the other judge keeps working.
                    while time.time()<=e.until:await asyncio.sleep(min(60,max(1,e.until-time.time()+1)))
                    break
                except Exception as e:
                    if not calibrated:status.update(status='calibration_blocked',reason=str(e)[:500]);return
                    failed.add(task['evaluation_id']);status['failures']=len(failed)
                    with (out/(provider+'-failures.jsonl')).open('a') as f:f.write(json.dumps({'evaluation_id':task['evaluation_id'],'type':type(e).__name__,'reason':str(e)[:500]})+'\n')
                    # Unknown outcome is never replayed blindly by the cache.
            if not calibrated and len(calibration_results)==len(pending):
                passed=all(r['passed'] for r in calibration_results)
                core.write_json(out/(provider+'-calibration.json'),{'passed':passed,'results':calibration_results})
                if not passed:status['status']='calibration_failed';return
                calibrated=True
    await asyncio.gather(*(worker(p) for p in models))
