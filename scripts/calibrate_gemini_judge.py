"""Fictional-only calibration of the exact indexed pointwise judge transport."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core
from eval import semantic_judge as sj
from eval import judge_span_ids as indexed
from eval.gemini_judge import GeminiJudge,QuotaError,InvalidOutput
from eval.semantic_fixtures import fixtures
from scripts.run_semantic_judge import secure


def make_client(directory,free_receipt,shared_usage,catalogue,env_file,fictional=False,thinking_level='high'):
    key=None
    for line in Path(env_file).read_text().splitlines():
        if line.startswith('GEMINI_API_KEY='):key=line.split('=',1)[1].strip().strip('"').strip("'")
    if not key:raise ValueError('Missing GEMINI_API_KEY')
    free=json.loads(Path(free_receipt).read_text());free=free.get('gemini',free)
    return GeminiJudge(key,free['model'],json.loads((directory/'gemini_semantic_permission.json').read_text()),free,
        directory/'gemini_usage.jsonl',Path(shared_usage),json.loads(Path(catalogue).read_text()),fictional=fictional,reserve_requests=50,thinking_level=thinking_level)


def calibrate(directory,free_receipt,shared_usage,catalogue,env_file,thinking_level='high'):
    out=secure(directory);target=out/'calibration_indexed_v3.json'
    if target.exists():raise FileExistsError('Preserve completed calibration; choose a fresh run')
    client=make_client(out,free_receipt,shared_usage,catalogue,env_file,fictional=True,thinking_level=thinking_level)
    expected=fixtures();tasks=[t for t,e in expected];attempts=[];results=[]
    def call(items,seed):
        while True:
            try:return client.call_many(items,seed)
            except InvalidOutput as e:
                return [{'valid':False,'error':str(e),'raw':''} for t in items],{**e.usage,'raw_invalid_batch':e.raw}
            except QuotaError as e:
                if e.retry_after is None:raise
                print('Calibration waits for free minute quota',round(e.retry_after,1),flush=True);time.sleep(min(60,e.retry_after))
    first,usage=call(tasks,20261021);attempts.append({'seed':20261021,'results':first,'usage':usage});results=list(first)
    invalid=[i for i,((t,e),r) in enumerate(zip(expected,results)) if e is not None and not r['valid']]
    if invalid:
        retry,usage=call([tasks[i] for i in invalid],20261022);attempts.append({'seed':20261022,'results':retry,'usage':usage})
        for i,r in zip(invalid,retry):results[i]=r
    passed=all((r['valid'] and r['judgement']['correctness']==e) if e is not None else
               (not r['valid'] or r['judgement']['correctness'] is None) for r,(t,e) in zip(results,expected))
    report={'passed':passed,'provider':'gemini','model':client.model,'rubric_version':sj.VERSION,'transport_version':indexed.TRANSPORT,
        'thinking_level':thinking_level,'fixture_sha256':core.digest(expected),'scoreable_cases':8,'missing_reference_guards':1,
        'maximum_attempts_per_item':2,'prompt_sha256':core.digest(sj.PROMPT),'wrapper_sha256':core.digest(indexed.WRAPPER),
        'attempts':attempts,'final_results':results,'interpretation':'Eight scoreable cases must match exactly; missing-reference guard must return null or fail strict validation. No real benchmark examples used.'}
    with target.open('x') as f:f.write(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('Indexed fictional calibration passed:',passed,'gradeable cases:',8,'missing-reference guards:',1,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','free-receipt','shared-usage','catalogue','env-file'):p.add_argument('--'+name,required=True)
    p.add_argument('--thinking-level',choices=['minimal','low','medium','high'],default='high')
    calibrate(**vars(p.parse_args()))
