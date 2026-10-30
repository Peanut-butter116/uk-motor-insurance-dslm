"""CPU-only recovery and diagnostics. Never imports a model training runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from cloud.d1000.storage import info
from cloud.d1000.export import sha
from cloud.d1000.campaign import STORE, ROOT, write, restore_runtime


def restore():
    from kaggle import api
    from kaggle_secrets import UserSecretsClient
    import os
    secrets=UserSecretsClient()
    for key in ('KAGGLE_API_TOKEN','GROQ_API_KEY','GEMINI_API_KEY'):
        os.environ[key]=secrets.get_secret(key)
    api.authenticate()
    assert api.kernels_status('scarletthe0116/uk-motor-d1000-train-eval/8').status.name=='COMPLETE'
    previous=info(api,STORE);assert previous.is_private
    temp=Path('/kaggle/working/resume');temp.mkdir(exist_ok=True)
    api.dataset_download_file(STORE+'/'+str(previous.current_version_number),'progress.json',path=str(temp),quiet=True,force=True,unzip=True)
    saved=json.loads((temp/'progress.json').read_text())
    matches=[p for p in Path('/kaggle/input').rglob('artifacts.bin') if (p.parent/'progress.json').exists() and json.loads((p.parent/'progress.json').read_text())==saved]
    assert len(matches)==1,'Latest verified private checkpoint must be mounted'
    archive=matches[0];assert sha(archive)==saved['artifacts_sha256']
    assert saved['responses_completed']==912
    assert all(saved['models'][k]['status']=='complete' for k in ('qwen3b','ministral3b','smollm1b7'))
    assert saved['models']['deepseek1b5']['status']=='tokenisation_failed'
    restore_runtime(archive,ROOT,saved)
    preservation={}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and (p.relative_to(ROOT).parts[0] in ('runs','evaluation','bundle') or p.name=='config.json'):
            preservation[str(p.relative_to(ROOT))]=sha(p)
    cfg=json.loads(Path('cloud/d1000/config.json').read_text())
    gold=[json.loads(s) for s in (ROOT/'evaluation/gold.jsonl').read_text().splitlines() if s.strip()]
    gold_ids={g['id'] for g in gold};assert len(gold_ids)==76
    assert sum(g['answerable'] for g in gold)==67
    populations={};n=0
    for p in sorted((ROOT/'runs').glob('*/*responses.jsonl')):
        rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        assert len(rows)==152
        assert len({(r['qid'],r['condition']) for r in rows})==152
        assert all(r['model']==cfg['models'][p.parent.name] for r in rows)
        for condition in {r['condition'] for r in rows}:
            assert {r['qid'] for r in rows if r['condition']==condition}==gold_ids
        for r in rows:
            key=p.parent.name+'/'+r['condition'];populations[key]=populations.get(key,0)+1
        n+=len(rows)
    assert n==912 and len(populations)==12 and set(populations.values())=={76}
    assert json.loads((ROOT/'config.json').read_text())==json.loads(Path('cloud/d1000/config.json').read_text())
    saved['last_persistence']={'dataset':STORE,'version':previous.current_version_number,'private':True,'readback_verified':True}
    write(ROOT/'evaluation_preservation.json',preservation)
    adapters={k:sorted(str(p.relative_to(ROOT)) for p in (ROOT/'runs'/k/'full').rglob('adapter*.safetensors')) for k in ('qwen3b','ministral3b','smollm1b7')}
    assert all(adapters.values()),'Missing completed adapter weights'
    adapter_hashes={}
    for key in adapters:
        receipt=json.loads((ROOT/'runs'/key/'full/receipt.json').read_text())
        complete=json.loads((ROOT/'runs'/key/'COMPLETE.json').read_text())
        actual=sha(ROOT/'runs'/key/'full/adapter/adapter_model.safetensors')
        assert receipt['passed'] and receipt['model']==cfg['models'][key]
        assert actual==receipt['adapter_sha256']==complete['adapter_sha256'] and complete['responses']==304
        adapter_hashes[key]=actual
    return saved,{'source_version':previous.current_version_number,'archive_hash_verified':True,'response_count':n,'condition_counts':populations,'immutable_files':len(preservation),'immutable_manifest_sha256':sha(ROOT/'evaluation_preservation.json'),'adapter_hashes_verified':adapter_hashes,'secrets_accessible':True,'previous_execution_complete':True}


def verify_unchanged(root=ROOT):
    root=Path(root)
    expected=json.loads((root/'evaluation_preservation.json').read_text())
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and (p.relative_to(root).parts[0] in ('runs','evaluation','bundle') or p.name=='config.json')}
    assert actual==expected,'Immutable training/inference/benchmark artifacts changed'
    return hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()


def checkpoint(state):
    """Only the CPU coordinator writes snapshots; fail closed on mutation/races."""
    from kaggle import api
    from cloud.d1000.storage import publish
    verify_unchanged()
    assert info(api,STORE).current_version_number==state['last_persistence']['version'],'Concurrent writer detected'
    stage=Path('/kaggle/working/publish');stage.mkdir(exist_ok=True)
    with tarfile.open(stage/'artifacts.bin','w:gz') as tar:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file() or p.suffix in ('.lock','.tmp'):continue
            if p.suffix=='.jsonl':
                import io
                b=p.read_bytes();b=b[:b.rfind(b'\n')+1]
                member=tarfile.TarInfo(str(p.relative_to(ROOT)));member.size=len(b);tar.addfile(member,io.BytesIO(b))
            else:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    state['judge_counts']={provider:sum(1 for _ in (ROOT/'judging'/(provider+'.jsonl')).open()) if (ROOT/'judging'/(provider+'.jsonl')).exists() else 0 for provider in ('groq','gemini')}
    state['artifacts_sha256']=sha(stage/'artifacts.bin');write(stage/'progress.json',state)
    state['last_persistence']=publish(stage,STORE)
    write('/kaggle/working/evaluation_receipt.json',state)


def diagnostics():
    """Read existing cached fictional results in cloud; publish only safe categories."""
    from cloud.d1000 import judging
    from eval.semantic_fixtures import fixtures
    import jsonschema
    state,audit=restore()
    providers,runtime=judging.provider_runtime(ROOT)
    fixture_tasks={t['evaluation_id']:dict(t,citations=[],student_evidence_available=False) for t,_ in fixtures()}
    report={'preservation':audit,'providers':{},'new_api_calls':0}
    for provider in ('groq','gemini'):
        folder=runtime/'provider_cache'/provider;rows=[]
        for path in sorted(folder.glob('*.attempts.json')):
            history=json.loads(path.read_text())
            for attempt in history:
                raw=folder/(attempt['request_sha256']+'.raw.json')
                if not raw.exists():continue
                result=json.loads(raw.read_text());choice=result.get('choices',[{}])[0];msg=choice.get('message',{})
                row={'attempt_status':attempt.get('status'),'kind':attempt.get('kind'),'finish_reason':choice.get('finish_reason'),'usage':result.get('usage',{}),'content_characters':len(msg.get('content') or ''),'reasoning_characters':len(msg.get('reasoning') or ''),'fixture_ids':[]}
                try:
                    value=providers.parse_json_content(msg.get('content'))
                    # Only inspect fictional calibration IDs; never report candidate text.
                    evaluations=value.get('evaluations',[]) if isinstance(value,dict) else []
                    ids=[v.get('evaluation_id') for v in evaluations if isinstance(v,dict)]
                    if not ids or any(i not in fixture_tasks for i in ids):continue
                    row['fixture_ids']=ids;row['schema_errors']=[];row['semantic_errors']=[]
                    task=fixture_tasks[ids[0]]
                    for e in jsonschema.Draft202012Validator(judging.schema(task)).iter_errors(value):
                        row['schema_errors'].append({'validator':e.validator,'path':list(e.absolute_path)})
                    for v in evaluations:
                        try:judging.normalise(v,fixture_tasks[v['evaluation_id']])
                        except Exception as e:row['semantic_errors'].append({'type':type(e).__name__,'reason':str(e)[:120]})
                    row['refusal_support_values']=[v.get('evidence_support') for v in evaluations if fixture_tasks[v['evaluation_id']]['deterministic_abstention']]
                except Exception as e:row['parse_failure']=getattr(e,'kind',type(e).__name__)
                rows.append(row)
        report['providers'][provider]={'cached_attempts':rows,'verification':providers.verify_receipt(runtime,provider,state['judges'][provider]['model'])}
    # Never emit account receipt contents or private data.
    for p in report['providers'].values():
        v=p.pop('verification');p['verified_free_quotas']={k:v[k] for k in ('rpm','rpd','tpm','tpd')}
    write('/kaggle/working/evaluation_diagnostic_receipt.json',report)


def main():
    import threading,asyncio,time
    from cloud.d1000.judge_recovery import run,PROTOCOL,verify_calibration
    from cloud.d1000.metrics import summarise
    from experiments import core
    state,preservation=restore()
    from kaggle import api
    assert api.kernels_status('scarletthe0116/uk-motor-d1000-train-eval/16').status.name in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'),'Existing evaluation writer16 is still active'
    assert api.kernels_status('scarletthe0116/uk-motor-d1000-train-eval/14').status.name=='COMPLETE','Groq calibration14 must have completed'
    audit=json.loads(Path('cloud/d1000/evaluation_repair_v1.json').read_text())
    assert preservation['immutable_manifest_sha256']==audit['immutable_manifest_sha256'],'Source93 immutable inventory changed'
    assert all(sha(Path(p))==h for p,h in audit['immutable_code_hashes'].items()),'Frozen rubric/fixtures/reference assembly changed'
    actual={str(p):sha(p) for p in Path('cloud/d1000').glob('*.py')}
    assert actual==audit['destination_hashes'],'Unaudited evaluation code'
    allowed=[audit['source_fingerprint'],audit['destination_fingerprint']]
    for prior in audit.get('prior_evaluation_transitions',[]):
        assert hashlib.sha256(''.join(prior['source_hashes'][p] for p in sorted(prior['source_hashes'])).encode()).hexdigest()==prior['source_fingerprint']
        allowed.append(prior['source_fingerprint'])
    assert state['code_fingerprint'] in allowed
    old=json.loads(Path('cloud/d1000/resume_repair_v1.json').read_text())['destination_hashes']
    for name in ('train_worker.py','export.py','campaign.py','judging.py','gpu_probe.py','storage.py'):
        assert actual['cloud/d1000/'+name]==old['cloud/d1000/'+name],'Forbidden training/data/transport change'
    from cloud.d1000.judging import provider_runtime
    providers,runtime=provider_runtime(ROOT)
    delta_file=Path('cloud/d1000/groq_probe_quota_deltas.json')
    assert sha(delta_file)==audit['quota_deltas_sha256']
    deltas=json.loads(delta_file.read_text())
    limits=providers.verify_receipt(runtime,'groq','openai/gpt-oss-120b')
    def merge(s):
        known={r['request_sha256'] for r in s['entries']}
        for r in deltas:
            if r['request_sha256'] not in known:s['entries'].append(r);known.add(r['request_sha256'])
    providers.Quotas(runtime/'provider_usage/groq.json','groq',limits).update(merge)
    calibration_file=Path('cloud/d1000/groq_calibration14_receipt.json')
    assert sha(calibration_file)==audit['groq_calibration_receipt_sha256']
    observed=json.loads(calibration_file.read_text());calibration=observed['calibration']
    from eval.semantic_fixtures import fixtures
    assert observed['status']=='calibration_passed' and calibration['passed'] is True
    verify_calibration(calibration)
    destination=ROOT/'judging/groq-calibration-v2.json';destination.parent.mkdir(exist_ok=True)
    if destination.exists() and json.loads(destination.read_text())!=calibration:
        history=destination.parent/'history';history.mkdir(exist_ok=True)
        (history/('groq-calibration-'+sha(destination)+'.json')).write_bytes(destination.read_bytes())
    write(destination,calibration)
    state['adopted_groq_calibration']={'source_execution':14,'receipt_sha256':sha(calibration_file),'quota_deltas_merged':len(deltas),'passed':True}
    state.update(code_fingerprint=audit['destination_fingerprint'],evaluation_repair=audit,
                 evaluation_only=True,status='evaluation_only_calibration',preservation=preservation,responses_completed=912,gpus=[])
    checkpoint(state) # Persistence/resume smoke before any new judge call.
    ready={p:threading.Event() for p in ('groq','gemini')}
    def worker():
        try:asyncio.run(run(ROOT,state,ready))
        except Exception as e:state['judging_failure']={'type':type(e).__name__,'reason':str(e)[:300]}
    thread=threading.Thread(target=worker,daemon=True);thread.start();last=time.monotonic()
    while thread.is_alive():
        passed=[p for p,s in state.get('judges',{}).items() if s.get('status')=='calibration_passed_waiting_persistence' and not ready[p].is_set()]
        if passed:
            checkpoint(state)
            for p in passed:ready[p].set()
            state['status']='evaluation_only_running'
        elif time.monotonic()-last>=120:checkpoint(state);last=time.monotonic()
        thread.join(timeout=5)
    verify_unchanged();state['results']=summarise(ROOT)
    state['metric_artifacts']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file()}
    state['paired_judgements']=len({r['evaluation_id'] for r in core.rows(ROOT/'judging/groq.jsonl')} & {r['evaluation_id'] for r in core.rows(ROOT/'judging/gemini.jsonl')}) if all((ROOT/'judging'/(p+'.jsonl')).exists() for p in ('groq','gemini')) else 0
    state['status']='evaluation_complete' if state['results']['judge_votes']=={'groq':912,'gemini':912} else 'evaluation_partial_or_blocked'
    checkpoint(state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnostics',action='store_true');args=p.parse_args()
    if args.diagnostics:diagnostics()
    else:main()
