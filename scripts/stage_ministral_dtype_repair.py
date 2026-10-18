"""Stage ONLY the failed pinned Ministral checkpoint; never resubmit successful models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core
from scripts.stage_motor_benchmark import stage


def prepare(home,output,public):
    out=core.private_output(output)
    stage(home,out/'preparation',out/'preparation-docs/manifests')
    prep=out/'preparation';old_cfg=json.loads((out/'preparation-docs/configs/six_models_descriptive.json').read_text())
    old_manifest=json.loads((prep/'local-bundle/manifest.json').read_text())
    selected=[m for m in old_cfg['models'] if m['id']=='mistralai/Ministral-3-3B-Instruct-2512']
    if len(selected)!=1:raise ValueError('Pinned Ministral configuration missing')
    cfg={**old_cfg,'models':selected,'post_dequant_fp16_cast':True,
         'protocol':'motor-frozen-test-descriptive-ministral-fp16-repair-v1'}
    manifest={**old_manifest,'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
              'repair':'FP8 dequantisation returns BF16 layers; cast all floating parameters to the originally requested FP16 before generation',
              'inference_prompt_changed':False,'checkpoint_changed':False}
    launch=out/'launch';payload=launch/'local-bundle';payload.mkdir(parents=True)
    for n in ['tasks.jsonl','contexts.jsonl']:shutil.copyfile(prep/'local-bundle'/n,payload/n)
    core.write_json(payload/'manifest.json',manifest)
    kernel=launch/'kernel';kernel.mkdir()
    code=(prep/'kernel/run.py').read_text()
    for before,after in [(f'cfg={old_cfg!r}',f'cfg={cfg!r}'),(f'manifest={old_manifest!r}',f'manifest={manifest!r}')]:
        if code.count(before)!=1:raise ValueError('Generated kernel serialization changed')
        code=code.replace(before,after)
    (kernel/'run.py').write_text(code)
    meta=json.loads((prep/'kernel/kernel-metadata.json').read_text())
    meta.update(id='scarletthe0116/motor-ministral-fp16-repair-20261020',title='motor-ministral-fp16-repair-20261020')
    core.write_json(kernel/'kernel-metadata.json',meta)
    public=Path(public);public.mkdir(parents=True,exist_ok=True)
    core.write_json(public/'protocol.json',manifest);core.write_json(public/'config.json',cfg)
    core.write_json(public/'launch.json',{'notebook':meta['id'],'status':'staged_not_submitted','private_requested':True,
        'kernel_sha256':core.sha(kernel/'run.py'),'runner_sha256':core.sha(core.ROOT/'experiments/six_model_benchmark.py'),
        'selected_checkpoint':selected[0],'questions':76,'paired_tasks':152,'new_restricted_upload':False})
    print('Staged only the failed Ministral checkpoint:',kernel)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['home','output','public']:p.add_argument('--'+n,required=True)
    prepare(**vars(p.parse_args()))
