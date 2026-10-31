"""Read-only results diagnostic: one quota-reserved, wholly fictional request."""
import json,os,tarfile,time
from pathlib import Path
from cloud.d1000.campaign import ROOT,STORE,write
from cloud.d1000.export import sha
from cloud.d1000.storage import info
from cloud.d1000.judge_recovery import body,validate,complete,PROTOCOL
from cloud.d1000.judging import provider_runtime,core
from eval.semantic_fixtures import fixtures


def main():
    from kaggle_secrets import UserSecretsClient
    from kaggle import api
    from groq import Groq,APIStatusError
    import httpx
    secrets=UserSecretsClient()
    for key in ('KAGGLE_API_TOKEN','GROQ_API_KEY'):os.environ[key]=secrets.get_secret(key)
    api.authenticate();latest=info(api,STORE);assert latest.is_private
    temp=Path('/kaggle/working/probe');temp.mkdir(exist_ok=True)
    api.dataset_download_file(STORE+'/'+str(latest.current_version_number),'progress.json',path=str(temp),quiet=True,force=True,unzip=True)
    current=json.loads((temp/'progress.json').read_text())
    assert current['judges']['groq'].get('reason')=='groq HTTP 400','Groq worker must be stopped before probe'
    candidates=[p for p in Path('/kaggle/input').rglob('artifacts.bin') if (p.parent/'progress.json').exists()]
    assert len(candidates)==1;archive=candidates[0];saved=json.loads((archive.parent/'progress.json').read_text())
    assert sha(archive)==saved['artifacts_sha256']
    assert saved['judges']['groq'].get('reason')=='groq HTTP 400'
    ROOT.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        name='judge_runtime/provider_usage/groq.json'
        p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(name).read())
    providers,runtime=provider_runtime(ROOT)
    limits=providers.verify_receipt(runtime,'groq','openai/gpt-oss-120b')
    delta_file=Path('cloud/d1000/groq_probe_quota_deltas.json')
    if delta_file.exists():
        q=providers.Quotas(runtime/'provider_usage/groq.json','groq',limits)
        def merge(s):
            known={r['request_sha256'] for r in s['entries']}
            for r in json.loads(delta_file.read_text()):
                if r['request_sha256'] not in known:s['entries'].append(r);known.add(r['request_sha256'])
        q.update(merge)
    # Gemini may advance private versions; its separate ledger is never touched.
    task=fixtures()[0][0];actual=body('groq','openai/gpt-oss-120b',task)
    digest=core.digest;quota=providers.Quotas(runtime/'provider_usage/groq.json','groq',limits)
    start_entries=len(json.loads((runtime/'provider_usage/groq.json').read_text())['entries'])
    request_hash=digest({'diagnostic':'groq-schema-v1','body':actual})
    counted=actual['messages']+[{'role':'user','content':json.dumps(actual['response_format'])}]
    report={'purpose':'fictional schema diagnostic only','new_research_judgements':0,'results_dataset_writes':0,'request_sha256':request_hash}
    try:reservation=quota.reserve(request_hash,providers.token_reservation('groq',actual['model'],counted,2048))
    except providers.QuotaPause as e:
        report.update(status='quota_wait',resume_after_epoch=e.until);write('/kaggle/working/groq_probe_receipt.json',report);return
    try:
        with Groq(api_key=os.environ['GROQ_API_KEY'],max_retries=0,timeout=120,http_client=httpx.Client(follow_redirects=False)) as client:
            raw=client.chat.completions.create(**actual).model_dump(mode='json')
        quota.reconcile(reservation,raw.get('usage',{}));first=validate(raw,task,providers)
        report.update(status='valid_fictional_response',usage=raw.get('usage',{}))
        results=[{'id':task['evaluation_id'],'correctness':first['correctness'],'expected':fixtures()[0][1],'passed':first['correctness']==fixtures()[0][1],'evidence_support':first['evidence_support'],'abstention_category':first['abstention_category'],'request_sha256':request_hash}]
        deadline=time.time()+20*60
        for task,expected in fixtures()[1:]:
            while True:
                try:
                    record=complete({'GROQ_API_KEY':os.environ['GROQ_API_KEY']},'groq','openai/gpt-oss-120b',task,runtime,providers);break
                except providers.QuotaPause as e:
                    if e.until>deadline:raise
                    time.sleep(max(1,e.until-time.time()+1))
            value=record['answer'];passed=value['correctness']==expected
            if task['deterministic_abstention']:passed=passed and value['evidence_support'] is None and value['abstention_category']==('incorrect_abstention' if task['gold_answerable'] else 'correct_abstention')
            results.append({'id':task['evaluation_id'],'correctness':value['correctness'],'expected':expected,'passed':passed,'evidence_support':value['evidence_support'],'abstention_category':value['abstention_category'],'request_sha256':record['request_sha256']})
        report['calibration']={'protocol':PROTOCOL,'fixtures_sha256':core.digest(fixtures()),'passed':len(results)==9 and all(r['passed'] for r in results),'results':results}
        report['status']='calibration_passed' if report['calibration']['passed'] else 'calibration_failed'
    except APIStatusError as e:
        # Error body concerns only the public fictional schema, never candidate text.
        error=e.body.get('error',e.body) if isinstance(e.body,dict) else {}
        report.update(status='request_rejected',http_status=e.status_code,error={k:error.get(k) for k in ('type','code','param','message')})
        if e.status_code==429:quota.pause(providers.retry_after(e.response.headers))
    except providers.QuotaPause as e:report.update(status='quota_wait',resume_after_epoch=e.until)
    except Exception as e:report.update(status='uncertain_or_invalid',error_type=type(e).__name__,reason=str(e)[:180])
    ledger=json.loads((runtime/'provider_usage/groq.json').read_text())
    report['quota_delta']=ledger['entries'][reservation]
    report['quota_deltas']=ledger['entries'][start_entries:]
    write('/kaggle/working/groq_probe_receipt.json',report)
