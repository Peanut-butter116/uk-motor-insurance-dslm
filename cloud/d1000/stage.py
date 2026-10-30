"""Code-only Kaggle staging. No private data passes through this machine."""
import argparse
import hashlib
import json
from pathlib import Path


def stage(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    code={}
    for directory in ('experiments','src','eval','cloud/d1000'):
        for p in Path(directory).glob('*.py'):code[str(p)]=p.read_text()
    code['cloud/d1000/config.json']=Path('cloud/d1000/config.json').read_text()
    code['cloud/d1000/resume_repair_v1.json']=Path('cloud/d1000/resume_repair_v1.json').read_text()
    code['cloud/d1000/frozen_evaluation_hashes.json']=Path('cloud/d1000/frozen_evaluation_hashes.json').read_text()
    code['scripts/run_semantic_judge.py']=Path('scripts/run_semantic_judge.py').read_text()
    code['configs/insurance_3b.json']=Path('configs/insurance_3b.json').read_text()
    bootstrap='''from pathlib import Path
import json,subprocess,sys
code=Path('/kaggle/working/code');code.mkdir(exist_ok=True)
'''
    bootstrap+='files='+repr(code)+'\n'
    bootstrap+='''for name,text in files.items():
    p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
with Path('/kaggle/working/install.log').open('w') as log:
    subprocess.run([sys.executable,'-m','pip','install','transformers==5.19.0','peft==0.21.2','accelerate==1.15.0',
        'bitsandbytes==0.50.2','sentencepiece','protobuf','mistral-common','jinja2==3.1.6','kaggle==2.2.5','matplotlib','groq==1.7.0','tiktoken==0.14.0','jsonschema'],
        stdout=log,stderr=subprocess.STDOUT,check=True)
sys.path.insert(0,str(code))
import os
os.chdir(code)
from cloud.d1000.campaign import main
try:
    main()
except Exception as e:
    Path('/kaggle/working/FAILED.json').write_text(json.dumps({'type':type(e).__name__,'reason':str(e)[:1000]}))
    raise
'''
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':['# D1000 QLoRA training and dual-judge evaluation\n',
            'Private cloud export v1; original verified data preserved. Enable existing KAGGLE_API_TOKEN, GROQ_API_KEY, GEMINI_API_KEY. ',
            'T4 x2: one isolated process per GPU. Smoke passes automatically unlock full fixed-epoch training. ',
            'No generation, paid fallback or benchmark tuning.\n']},
            {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':bootstrap.splitlines(True)}]}
    (output/'campaign.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    meta={'id':'scarletthe0116/uk-motor-d1000-train-eval','title':'uk-motor-d1000-train-eval',
        'code_file':'campaign.ipynb','language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':['scarletthe0116/uk-motor-qa-d1000-training','scarletthe0116/uk-motor-qa-production-v1',
                           'scarletthe0116/uk-insurance-base-eval-qwen3b-private','scarletthe0116/uk-motor-frozen-evaluation-v2','scarletthe0116/uk-motor-d1000-models-results'],
        'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return {'notebook':meta['id'],'code_sha256':hashlib.sha256(bootstrap.encode()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);print(json.dumps(stage(**vars(p.parse_args()))))
