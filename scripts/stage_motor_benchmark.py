"""Stage one frozen descriptive motor benchmark, reusing an approved Kaggle dataset.

No data upload is performed. Gold answers and scoring contexts remain local.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core


def stage(home, output, public):
    home=Path(home).resolve();out=core.private_output(output)
    cfg=json.loads((core.ROOT/'experiments/2026-10-20/configs/six_models.json').read_text())
    cfg.update(protocol='motor-frozen-test-descriptive-v1',status='protocol_frozen',
        loading_precision='float16',quantisation='None, except original Ministral FP8 weights dequantised to float16',
        selection_permitted=False,development_required=False,
        prompt_version='common-single-user-inherited-instructions-v1',
        selection_rule='Descriptive only. No winner, model selection or prompt adjustment from protected test results.')
    bundle=home/'experiments/20261007/test-eval'
    inherited=json.loads((bundle/'manifest.json').read_text())
    assert core.sha(bundle/'tasks.jsonl')==inherited['tasks_sha256']
    assert core.sha(bundle/'contexts.jsonl')==inherited['contexts_sha256']
    motor={r['id'] for r in core.benchmark(home) if r['split']=='test' and r['line']=='motor'}
    assert len(motor)==76
    tasks=[t for t in core.rows(bundle/'tasks.jsonl') if t['qid'] in motor]
    payload=out/'local-bundle';payload.mkdir()
    core.write_rows(payload/'tasks.jsonl',tasks)
    core.write_rows(payload/'contexts.jsonl',[r for r in core.rows(bundle/'contexts.jsonl') if r['qid'] in motor])
    manifest={'purpose':'frozen_motor_test_descriptive','split':'test','questions':len(motor),
        'selection_permitted':False,'protocol_frozen_before_submission':True,
        'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        'tasks_sha256':core.sha(payload/'tasks.jsonl'),'contexts_sha256':core.sha(payload/'contexts.jsonl'),
        'source_tasks_sha256':inherited['tasks_sha256'],
        'source_contexts_sha256':inherited['contexts_sha256'],
        'gold_sha256':inherited['gold_sha256'],
        'source_dataset':'scarletthe0116/uk-insurance-base-eval-qwen3b-private',
        'evaluation_population':'Protected motor test, descriptive only, no selection',
        'new_restricted_upload':False}
    core.write_json(payload/'manifest.json',manifest)
    kernel=out/'kernel';kernel.mkdir()
    runner=(core.ROOT/'experiments/six_model_benchmark.py').read_text()
    code='''from pathlib import Path
import json,hashlib,subprocess,sys,zipfile
root=Path('/kaggle/working/benchmark-input');root.mkdir()
'''
    code+=f"cfg={cfg!r}\nmanifest={manifest!r}\nmotor=set({sorted(motor)!r})\n"
    code+='''hits=list(Path('/kaggle/input').rglob('INSURANCE_EXPERIMENT'))
if len(hits)!=1:raise RuntimeError('Expected one existing approved private dataset')
mount=hits[0].parent
if (mount/'payload/tasks.jsonl').exists():source=(mount/'payload/tasks.jsonl').read_bytes()
else:
    with zipfile.ZipFile(mount/'payload.zip') as z:source=z.read('tasks.jsonl')
if hashlib.sha256(source).hexdigest()!=manifest['source_tasks_sha256']:raise RuntimeError('Existing private dataset version mismatch')
tasks=[json.loads(s) for s in source.decode().splitlines() if s]
data=''.join(json.dumps(t,ensure_ascii=False)+'\\n' for t in tasks if t['qid'] in motor)
if hashlib.sha256(data.encode()).hexdigest()!=manifest['tasks_sha256']:raise RuntimeError('Motor task population mismatch')
(root/'tasks.jsonl').write_text(data)
(root/'manifest.json').write_text(json.dumps(manifest))
(root/'config.json').write_text(json.dumps(cfg))
'''
    code+=f"(root/'runner.py').write_text({runner!r})\n"
    code+='''with Path('/kaggle/working/install.log').open('w') as f:
    subprocess.run([sys.executable,'-m','pip','install','transformers==5.19.0','accelerate==1.15.0','bitsandbytes==0.50.2','sentencepiece','protobuf','mistral-common','jinja2==3.1.6'],stdout=f,stderr=subprocess.STDOUT,check=True)
with Path('/kaggle/working/inference.log').open('w') as f:
    subprocess.run([sys.executable,str(root/'runner.py'),'--config',str(root/'config.json'),'--bundle',str(root),'--output','/kaggle/working/benchmark-output'],stdout=f,stderr=subprocess.STDOUT,check=True)
'''
    (kernel/'run.py').write_text(code)
    meta={'id':'scarletthe0116/motor-six-model-frozen-test-20261020','title':'motor-six-model-frozen-test-20261020',
        'code_file':'run.py','language':'python','kernel_type':'script','is_private':True,
        'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':[manifest['source_dataset']],'competition_sources':[],
        'kernel_sources':[],'model_sources':[]}
    core.write_json(kernel/'kernel-metadata.json',meta)
    public=Path(public);public.mkdir(parents=True,exist_ok=True)
    core.write_json(public/'six_model_test_protocol.json',manifest)
    (public.parent/'configs').mkdir(parents=True,exist_ok=True)
    core.write_json(public.parent/'configs/six_models_descriptive.json',cfg)
    core.write_json(public/'benchmark_launch.json',{'notebook':meta['id'],'status':'staged_not_submitted',
        'private_requested':True,'kernel_sha256':core.sha(kernel/'run.py'),
        'runner_sha256':core.sha(core.ROOT/'experiments/six_model_benchmark.py'),
        'manifest_sha256':core.sha(payload/'manifest.json'), 'new_restricted_upload':False})
    print('Frozen 76 questions / 152 matched tasks. Staged only:',kernel)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['home','output','public']:p.add_argument('--'+n,required=True)
    stage(**vars(p.parse_args()))
