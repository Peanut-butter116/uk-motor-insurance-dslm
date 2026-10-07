"""Prepare private Kaggle dataset/kernel folders. This command never uploads or runs."""
import argparse
import json
from pathlib import Path
import shutil
from .core import ROOT, private_output, sha, write_json
from .prepare_sft import validate_bundle


def stage(bundle,output,username,kind,model='ministral3b'):
    bundle=Path(bundle)
    if kind=='smoke':
        validate_bundle(bundle);payload_files=['train.jsonl','manifest.json']
    else:
        m=json.loads((bundle/'manifest.json').read_text())
        if sha(bundle/'tasks.jsonl')!=m['tasks_sha256']:raise ValueError('Evaluation bundle changed')
        payload_files=['tasks.jsonl','manifest.json']  # scoring contexts and gold stay local
    out=private_output(output);dataset=out/'dataset';kernel=out/'kernel'
    payload=dataset/'payload';payload.mkdir(parents=True);kernel.mkdir()
    for name in payload_files:shutil.copy2(bundle/name,payload/name)
    code=dataset/'code'
    for folder in ('experiments','src','eval'):
        for p in (ROOT/folder).glob('*.py'):
            target=code/folder/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    (code/'configs').mkdir(parents=True)
    shutil.copy2(ROOT/'configs/insurance_3b.json',code/'configs/insurance_3b.json')
    shutil.copy2(ROOT/'requirements-experiment.txt',code/'requirements-experiment.txt')
    (dataset/'INSURANCE_EXPERIMENT').write_text(kind)
    slug=f'uk-insurance-{kind}-{model}'
    write_json(dataset/'dataset-metadata.json',{'id':f'{username}/{slug}-private',
        'title':f'{slug}-private','licenses':[{'name':'other'}]})
    command=(['-m','experiments.train','--mode','smoke'] if kind=='smoke' else
             ['-m','experiments.evaluate','run','--model',model])
    script='''import glob, json, pathlib, subprocess, sys, traceback
work=pathlib.Path('/kaggle/working')
try:
    hits=glob.glob('/kaggle/input/**/INSURANCE_EXPERIMENT',recursive=True)
    if len(hits)!=1: raise RuntimeError('Expected exactly one private experiment dataset')
    root=pathlib.Path(hits[0]).parent
    subprocess.run([sys.executable,'-m','pip','install','-r',str(root/'code/requirements-experiment.txt')],check=True)
    cmd=[sys.executable]+COMMAND+['--bundle',str(root/'payload'),'--output',str(work/'run')]
    subprocess.run(cmd,cwd=root/'code',check=True)
    (work/'SUCCESS.json').write_text(json.dumps({'kind':KIND}))
except Exception:
    (work/'FAILED.txt').write_text(traceback.format_exc())
    raise
'''.replace('COMMAND',repr(command)).replace('KIND',repr(kind))
    (kernel/'run.py').write_text(script)
    write_json(kernel/'kernel-metadata.json',{'id':f'{username}/{slug}','title':slug,'code_file':'run.py',
        'language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,
        'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':[f'{username}/{slug}-private'],'competition_sources':[],
        'kernel_sources':[],'model_sources':[]})
    print(f'Private staging only: {out}; no upload or GPU run performed.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--username',default='KAGGLE_USERNAME');p.add_argument('--kind',choices=['smoke','base-eval'],required=True)
    p.add_argument('--model',choices=['qwen3b','ministral3b'],default='ministral3b')
    stage(**vars(p.parse_args()))
