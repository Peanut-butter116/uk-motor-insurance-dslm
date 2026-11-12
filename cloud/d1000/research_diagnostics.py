"""Read cached validation categories only; no judge calls or results writes."""
import collections,json,os,tarfile,time
from pathlib import Path
from cloud.d1000.campaign import write
from cloud.d1000.export import sha


def main():
    from kaggle import api
    from kaggle_secrets import UserSecretsClient
    os.environ['KAGGLE_API_TOKEN']=UserSecretsClient().get_secret('KAGGLE_API_TOKEN');api.authenticate()
    archives=[p for p in Path('/kaggle/input').rglob('artifacts.bin') if (p.parent/'progress.json').exists()]
    assert len(archives)==1;archive=archives[0];saved=json.loads((archive.parent/'progress.json').read_text())
    assert sha(archive)==saved['artifacts_sha256']
    report={'kind':'research_validation_categories','source_archive_sha256':saved['artifacts_sha256'],'source_judge_counts':saved['judge_counts'],'new_judge_calls':0,'results_dataset_writes':0,'providers':{}}
    with tarfile.open(archive) as tar:
        for provider in ('groq','gemini'):
            errors=collections.Counter();final_errors=collections.Counter();states=collections.Counter();tasks=0
            prefix='judge_runtime/evaluation_cache_v2/'+provider+'/'
            for member in tar.getmembers():
                if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith('.attempts.json'):continue
                history=json.loads(tar.extractfile(member).read());tasks+=1
                for h in history:
                    states[h['status']]+=1
                    if h.get('validation_error'):errors[h['validation_error']]+=1
                if history and history[-1]['status']=='known_invalid' and len(history)>=3:
                    final_errors[history[-1].get('validation_error','unknown')]+=1
            usage=json.loads(tar.extractfile('judge_runtime/provider_usage/'+provider+'.json').read())
            recent=[e for e in usage['entries'] if e['time']>time.time()-86400]
            report['providers'][provider]={'tasks_with_attempts':tasks,'attempt_states':dict(states),'validation_categories':dict(errors),'exhausted_task_categories':dict(final_errors),'requests_reserved_24h':len(recent),'conservative_reserved_tokens_24h':sum(e['quota_tokens'] for e in recent)}
    write('/kaggle/working/evaluation_diagnostic_receipt.json',report)
