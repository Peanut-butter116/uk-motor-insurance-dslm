"""Kaggle D1000 campaign coordinator. One writer, explicit GPU isolation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from cloud.d1000.export import sha,HASHES
from cloud.d1000.storage import info,publish,verify_file

ROOT=Path('/kaggle/working/d1000')
STORE='scarletthe0116/uk-motor-d1000-models-results'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def checkpoint(state):
    """Snapshot only complete, immutable model stages and checkpoint markers."""
    stage=Path('/kaggle/working/publish');stage.mkdir(exist_ok=True)
    state['judge_counts']={p:sum(1 for _ in (ROOT/'judging'/f'{p}.jsonl').open())
                          if (ROOT/'judging'/f'{p}.jsonl').exists() else 0 for p in ('groq','gemini')}
    with tarfile.open(stage/'artifacts.tar.gz','w:gz') as tar:
        for name in ('config.json','bundle','evaluation','production_transport'):
            if (ROOT/name).exists():tar.add(ROOT/name,arcname=name)
        for directory in ('judging','judge_runtime'):
            for p in (ROOT/directory).rglob('*') if (ROOT/directory).exists() else []:
                if not p.is_file() or p.suffix in ('.lock','.tmp'):continue
                import io
                b=p.read_bytes()
                if p.suffix=='.jsonl':b=b[:b.rfind(b'\n')+1]
                item=tarfile.TarInfo(str(p.relative_to(ROOT)));item.size=len(b);tar.addfile(item,io.BytesIO(b))
        for model in (ROOT/'runs').glob('*') if (ROOT/'runs').exists() else []:
            for p in model.rglob('*'):
                if not p.is_file():continue
                if 'checkpoints' in p.parts:
                    cp=next((a for a in p.parents if a.name.startswith('checkpoint-')),None)
                    if cp is None or not (cp/'COMPLETE.json').exists():continue
                elif any(x in p.parts for x in ('smoke','full')):
                    phase=model/('smoke' if 'smoke' in p.parts else 'full')
                    if not (phase/'receipt.json').exists():continue
                # Complete line snapshots only for append-only inference journals.
                if p.suffix=='.jsonl':
                    payload=p.read_bytes();payload=payload[:payload.rfind(b'\n')+1]
                    import io
                    item=tarfile.TarInfo(str(p.relative_to(ROOT)));item.size=len(payload);tar.addfile(item,io.BytesIO(payload))
                else:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    state['artifacts_sha256']=sha(stage/'artifacts.tar.gz')
    write(stage/'progress.json',state)
    if state.get('last_persistence'):
        from kaggle import api
        current=info(api,STORE)
        assert current.current_version_number==state['last_persistence']['version'],'Concurrent writer detected; stop'
    receipt=publish(stage,STORE,create=state.get('last_persistence') is None)
    state['last_persistence']=receipt;write('/kaggle/working/campaign_receipt.json',state)


def evaluation():
    from experiments import core
    # Same frozen 140-question tasks/contexts, filtered by immutable motor IDs.
    mount=next(p.parent for p in Path('/kaggle/input').rglob('INSURANCE_EXPERIMENT'))
    def payload(name):
        if (mount/'payload'/name).exists():return (mount/'payload'/name).read_bytes()
        with zipfile.ZipFile(mount/'payload.zip') as z:return z.read(name)
    tasks_bytes=payload('tasks.jsonl')
    assert __import__('hashlib').sha256(tasks_bytes).hexdigest()=='4d4191ab1f76e9ee2aa1492c48d9a4da0459854b35eec3472ac7b8a919365c4d'
    mpath=next(p for p in Path('/kaggle/input').rglob('migration_manifest.json') if json.loads(p.read_text()).get('archive_sha256')=='a88e3259cde5357f273f97a9ff2afa0821359815a7238c78332056d16511fe0e')
    m=json.loads(mpath.read_text())
    with tarfile.open(mpath.parent/'checkpoint.bin') as tar:
        name=next(n for n,h in m['files'].items() if h==HASHES['gold']);b=tar.extractfile(name).read()
        assert __import__('hashlib').sha256(b).hexdigest()==HASHES['gold']
    gold=[json.loads(s) for s in b.decode().splitlines() if s]
    gold=[g for g in gold if g['split']=='test' and g['line']=='motor'];assert len(gold)==76
    motor={g['id'] for g in gold};tasks=[json.loads(s) for s in tasks_bytes.decode().splitlines() if s]
    tasks=[t for t in tasks if t['qid'] in motor];assert len(tasks)==152
    output=ROOT/'evaluation';output.mkdir(exist_ok=True)
    core.write_rows(output/'tasks.jsonl',tasks);core.write_rows(output/'gold.jsonl',gold)
    frozen=next(Path('/kaggle/input').rglob('FREEZE_v2.json')).parent
    expected=json.loads((Path(__file__).parent/'frozen_evaluation_hashes.json').read_text())
    for name,h in expected.items():assert sha(frozen/name)==h
    for name in ('contexts.jsonl','chunks.jsonl'):shutil.copy2(frozen/name,output/name)
    write(output/'manifest.json',{'questions':76,'tasks':152,'gold_sha256':HASHES['gold'],
        'source_tasks_sha256':__import__('hashlib').sha256(tasks_bytes).hexdigest(),
        'reference_corpus_hashes':expected,
        'tasks_sha256':sha(output/'tasks.jsonl'),'selection_permitted':False})


def main():
    assert Path('/kaggle/working').is_dir(),'Cloud only'
    ROOT.mkdir(exist_ok=True)
    state={'protocol':'d1000-mixed-quality-qlora-dual-judge/1','status':'preflight','models':{},
        'responses_completed':0,'judge_counts':{'groq':0,'gemini':0},'last_persistence':None}
    write('/kaggle/working/campaign_receipt.json',state)
    from kaggle_secrets import UserSecretsClient
    secrets=UserSecretsClient()
    # Never print secrets or place them in source/artifacts.
    for name in ('KAGGLE_API_TOKEN','GROQ_API_KEY','GEMINI_API_KEY'):
        try:os.environ[name]=secrets.get_secret(name)
        except Exception:
            if name=='KAGGLE_API_TOKEN':
                state.update(status='blocked_secrets',required_action='Enable KAGGLE_API_TOKEN in this notebook and Save & Run All')
                write('/kaggle/working/campaign_receipt.json',state);return
    from kaggle import api
    api.authenticate()
    # Do not compete with the independently started hardware/QLoRA smoke.
    while api.kernels_status('scarletthe0116/uk-motor-d1000-gpu-smoke/2').status.name not in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'):
        state['status']='waiting_for_gpu_smoke';write('/kaggle/working/campaign_receipt.json',state);time.sleep(30)
    # Restore completed cloud stages/optimizer checkpoints instead of repeating
    # successful model runs or judge API requests on an editor restart.
    try:previous=info(api,STORE)
    except Exception as e:
        if getattr(getattr(e,'response',None),'status_code',None)!=404 and '404' not in str(e):raise
        previous=None
    if previous is not None:
        assert previous.is_private is True
        dest=Path('/kaggle/working/resume');dest.mkdir(exist_ok=True)
        for name in ('progress.json','artifacts.tar.gz'):
            api.dataset_download_file(STORE+'/'+str(previous.current_version_number),name,path=str(dest),quiet=True,force=True,unzip=True)
        saved=json.loads((dest/'progress.json').read_text())
        assert sha(dest/'artifacts.tar.gz')==saved['artifacts_sha256']
        with tarfile.open(dest/'artifacts.tar.gz') as tar:tar.extractall(ROOT,filter='data')
        state=saved;state['last_persistence']={'dataset':STORE,'version':previous.current_version_number,'private':True,'readback_verified':True}
    dataset='scarletthe0116/uk-motor-qa-d1000-training';d=info(api,dataset)
    assert d.is_private is True and d.current_version_number==1
    manifest=next(Path('/kaggle/input').rglob('uk_motor_qa_d1000_manifest.json'))
    export=manifest.parent;m=json.loads(manifest.read_text())
    for name,h in m['files'].items():assert sha(export/name)==h
    verify_file(dataset,1,'export/uk_motor_qa_d1000.csv',m['files']['uk_motor_qa_d1000.csv'])
    cfg=json.loads((Path(__file__).parent/'config.json').read_text())
    fingerprint=__import__('hashlib').sha256(''.join(sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()
    if state.get('code_fingerprint'):assert state['code_fingerprint']==fingerprint,'Resume code differs; require a separately audited repair'
    state['code_fingerprint']=fingerprint
    if (ROOT/'config.json').exists():assert json.loads((ROOT/'config.json').read_text())==cfg,'Restored experiment config differs; use separate versioned campaign'
    write(ROOT/'config.json',cfg)
    import torch
    state['gpus']=[{'index':i,'name':torch.cuda.get_device_name(i),'total_memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    assert state['gpus'],'No cloud CUDA allocation'
    state['export']={'dataset':dataset,'version':1,'records':1000,'csv_sha256':m['files']['uk_motor_qa_d1000.csv'],'readback_verified':True}
    evaluation()
    from experiments.prepare_milestone_sft import prepare
    if (ROOT/'bundle/manifest.json').exists():
        bundle=json.loads((ROOT/'bundle/manifest.json').read_text())
        assert bundle['source_csv_sha256']==m['files']['uk_motor_qa_d1000.csv'] and bundle['models']==cfg['models']
    else:
        from experiments import core
        bundle=prepare(export,ROOT/'bundle',cfg,core.rows(ROOT/'evaluation/gold.jsonl'))
    state['dataset_conversion']={k:bundle[k] for k in ('source_rows','eligible_rows','model_preflight_failures')}
    state['train_rows']=len(bundle['train_ids']);state['validation_rows']=len(bundle['validation_ids'])
    state['status']='gpu_smoke_then_training';checkpoint(state)
    import threading,asyncio
    from cloud.d1000.judging import run as judge_run
    stop=threading.Event()
    def judge_thread():
        try:asyncio.run(judge_run(ROOT,stop,state))
        except Exception as e:state['judging_failure']={'type':type(e).__name__,'reason':str(e)[:600]}
    judges=threading.Thread(target=judge_thread,daemon=True);judges.start()
    queue=[k for k in cfg['models'] if k not in bundle['model_preflight_failures'] and not (ROOT/'runs'/k/'COMPLETE.json').exists()]
    for k in cfg['models']:state['models'][k]={'status':'queued' if k in queue else 'complete' if (ROOT/'runs'/k/'COMPLETE.json').exists() else 'tokenisation_failed'}
    active={};free=list(range(min(2,torch.cuda.device_count())));last=time.monotonic()
    # One process/GPU. No unsafe four-way oversubscription on two T4s.
    while queue or active:
        while queue and free:
            key=queue.pop(0);gpu=free.pop(0);env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false'}
            log=(ROOT/(key+'.log')).open('w')
            proc=subprocess.Popen([sys.executable,'-m','cloud.d1000.train_worker','--root',str(ROOT),'--model',key],env=env,stdout=log,stderr=subprocess.STDOUT)
            active[key]=(proc,gpu,log);state['models'][key]={'status':'smoke_training_inference','gpu':gpu,'pid':proc.pid}
        for key,(proc,gpu,log) in list(active.items()):
            if proc.poll() is not None:
                log.close();state['models'][key].update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
                free.append(gpu);del active[key];checkpoint(state)
        if time.monotonic()-last>120:
            state['responses_completed']=sum(sum(1 for _ in p.open()) for p in (ROOT/'runs').glob('*/*responses.jsonl'))
            checkpoint(state);last=time.monotonic()
        write('/kaggle/working/campaign_receipt.json',state);time.sleep(10)
    state['status']='training_inference_finished';checkpoint(state);stop.set()
    while judges.is_alive():
        judges.join(timeout=30)
        checkpoint(state)
    from cloud.d1000.metrics import summarise
    summary=summarise(ROOT);state.update(status='finished_or_documented_partial',results=summary)
    checkpoint(state)


if __name__=='__main__':
    try:main()
    except Exception as e:
        write('/kaggle/working/FAILED.json',{'type':type(e).__name__,'reason':str(e)[:1000]})
        raise
