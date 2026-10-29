"""Real D1000 GPU smoke, no API secrets or production writes needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from cloud.d1000.campaign import ROOT,write,evaluation


def main():
    import torch
    from experiments.prepare_milestone_sft import prepare
    ROOT.mkdir(exist_ok=True)
    cfg=json.loads((Path(__file__).parent/'config.json').read_text());write(ROOT/'config.json',cfg)
    state={'status':'gpu_preflight','gpus':[{'index':i,'name':torch.cuda.get_device_name(i),
        'total_memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
        'models':{},'full_training_started':False,'model_calls':0}
    write('/kaggle/working/gpu_probe_receipt.json',state)
    assert state['gpus'],'No CUDA allocation'
    export=next(Path('/kaggle/input').rglob('uk_motor_qa_d1000_manifest.json')).parent
    # A training smoke must not read the held-out benchmark at all.
    m=prepare(export,ROOT/'bundle',cfg)
    state['conversion']={k:m[k] for k in ('source_rows','eligible_rows','model_preflight_failures')}
    state['train_rows']=len(m['train_ids']);state['validation_rows']=len(m['validation_ids'])
    pending=[k for k in cfg['models'] if k not in m['model_preflight_failures']]
    active={};free=list(range(min(2,len(state['gpus']))))
    while pending or active:
        while pending and free:
            key=pending.pop(0);gpu=free.pop(0);log=(ROOT/(key+'.log')).open('w')
            p=subprocess.Popen([sys.executable,'-m','cloud.d1000.train_worker','--root',str(ROOT),'--model',key,'--smoke-only'],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false'},stdout=log,stderr=subprocess.STDOUT)
            active[key]=(p,gpu,log);state['models'][key]={'status':'smoke_running','gpu':gpu}
        for key,(p,gpu,log) in list(active.items()):
            if p.poll() is None:continue
            log.close();state['models'][key].update(status='smoke_passed' if p.returncode==0 else 'smoke_failed',exit_code=p.returncode)
            metrics=ROOT/'runs'/key/'smoke/training.json'
            if metrics.exists():
                t=json.loads(metrics.read_text());state['models'][key]['metrics']={k:t[k] for k in ('loss','seconds','samples_per_second','peak_gpu_bytes','steps')}
            failure=ROOT/'runs'/key/'FAILED.json'
            if failure.exists():
                f=json.loads(failure.read_text());state['models'][key]['failure']={k:f[k] for k in ('type','reason')}
            free.append(gpu);del active[key]
        write('/kaggle/working/gpu_probe_receipt.json',state);time.sleep(10)
    state['status']='smoke_finished';write('/kaggle/working/gpu_probe_receipt.json',state)


if __name__=='__main__':
    try:main()
    except Exception as e:
        write('/kaggle/working/gpu_probe_failure.json',{'type':type(e).__name__,'reason':str(e)[:1000]});raise
