"""Package full training or tuned evaluation without altering fingerprinted code.

Staging only: uploads and GPU launches remain explicit Kaggle CLI steps.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import core, train, reload_check, prepare_sft, kaggle_stage

GATES = ['leakage', 'tokenisation', 'masking', 'finite_loss', 'language_only', 'adapter_reload']


def verify_training(downloaded, launch, bundle, mode):
    downloaded, launch, bundle = map(Path, (downloaded, launch, bundle))
    if (downloaded/'FAILED.txt').exists():
        raise ValueError('GPU wrapper failed')
    success = json.loads((downloaded/'SUCCESS.json').read_text())
    expected = json.loads((launch/'launch.json').read_text())
    if success != {'kind': mode, 'nonce': expected['nonce']}:
        raise ValueError('Wrong or stale GPU launch')
    _, manifest = prepare_sft.validate_bundle(bundle)
    receipt = json.loads((downloaded/'run/receipt.json').read_text())
    if mode == 'smoke':
        train.require_smoke(downloaded/'run/receipt.json', manifest)
    if (receipt.get('mode') != mode or receipt.get('fingerprint') != train.fingerprint(manifest)
            or receipt.get('rows') != 21 or any(receipt.get('gates', {}).get(g) is not True for g in GATES)):
        raise ValueError('Training receipt does not match frozen experiment')
    if core.sha(downloaded/'run/reload_diagnostics.json') != receipt['diagnostics_sha256']:
        raise ValueError('Reload diagnostics changed')
    diagnostics = json.loads((downloaded/'run/reload_diagnostics.json').read_text())
    final = diagnostics['prepared_reload']
    reload_check.require_comparison(final)
    if final['atol'] != 0.02 or final['rtol'] != 0.01 or not final['tokenizer_equal']:
        raise ValueError('Reload tolerance or tokenisation changed')
    for check in [diagnostics['saved_adapter'], final['restored_adapter']]:
        if check['equal'] is not True or any(check[k] for k in ['missing_trained','missing','extra','different']):
            raise ValueError('Incomplete saved/restored adapter')
    if mode == 'full':
        smoke = json.loads((launch/'dataset/payload/smoke_receipt.json').read_text())
        if receipt['versions'] != smoke['versions']:
            raise ValueError('Runtime differs from passing smoke')
        log = json.loads((downloaded/'run/training.json').read_text())['log']
        if not log or log[-1].get('epoch') != core.config()['training']['epochs']:
            raise ValueError('Full training did not complete the fixed epochs')
    return receipt


def retarget(out, username, kind, before_command, after_command):
    out = Path(out); slug = 'uk-insurance-' + kind + '-ministral3b'
    core.write_json(out/'dataset/dataset-metadata.json', {'id':username+'/'+slug+'-private',
        'title':slug+'-private','licenses':[{'name':'other'}]})
    p = out/'kernel/kernel-metadata.json'; meta = json.loads(p.read_text())
    meta.update(id=username+'/'+slug,title=slug,dataset_sources=[username+'/'+slug+'-private'])
    core.write_json(p,meta)
    p = out/'kernel/run.py'; script = p.read_text()
    old_kind = 'smoke' if kind == 'full' else 'base-eval'
    if script.count(repr(before_command)) != 1 or script.count("'kind':"+repr(old_kind)) != 1:
        raise ValueError('Unexpected wrapper; refusing to patch launch command')
    script = script.replace(repr(before_command),repr(after_command)).replace("'kind':"+repr(old_kind),"'kind':"+repr(kind))
    p.write_text(script)
    (out/'dataset/INSURANCE_EXPERIMENT').write_text(kind)
    p = out/'launch.json'; meta = json.loads(p.read_text());meta['kind'] = kind;core.write_json(p,meta)


def provenance(bundle, receipt_path):
    return {'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=core.ROOT,text=True).strip(),
        'code_hash':train.code_hash(), 'config_sha256':core.sha(core.ROOT/'configs/insurance_3b.json'),
        'training_manifest_sha256':core.sha(Path(bundle)/'manifest.json'),
        'receipt_sha256':core.sha(receipt_path), 'receipt':json.loads(Path(receipt_path).read_text())}


def stage_full(bundle, completed, source_launch, output, username):
    verify_training(completed,source_launch,bundle,'smoke')
    receipt = Path(completed)/'run/receipt.json'
    kaggle_stage.stage(bundle,output,username,'smoke','ministral3b')
    out = Path(output)
    shutil.copy2(receipt,out/'dataset/payload/smoke_receipt.json')
    retarget(out,username,'full',['-m','experiments.train','--mode','smoke'],
        ['-m','experiments.train','--mode','full','--smoke-receipt',
         '/kaggle/working/input-copy/payload/smoke_receipt.json'])
    core.write_json(out/'provenance.json',provenance(bundle,receipt))


def stage_eval(bundle, training_bundle, completed, source_launch, output, username):
    verify_training(completed,source_launch,training_bundle,'full')
    adapter = Path(completed)/'run/adapter'
    ac = json.loads((adapter/'adapter_config.json').read_text())
    if ac['base_model_name_or_path'] != core.config()['models']['ministral3b']['id']:
        raise ValueError('Adapter model mismatch')
    kaggle_stage.stage(bundle,output,username,'base-eval','ministral3b')
    out = Path(output); dest = out/'dataset/payload/adapter';dest.mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors'):
        shutil.copy2(adapter/name,dest/name)
    retarget(out,username,'sft-eval',['-m','experiments.evaluate','run','--model','ministral3b'],
        ['-m','experiments.evaluate','run','--model','ministral3b','--adapter',
         '/kaggle/working/input-copy/payload/adapter'])
    meta = provenance(training_bundle,Path(completed)/'run/receipt.json')
    meta['adapter_sha256'] = {p.name:core.sha(p) for p in dest.iterdir()}
    core.write_json(out/'provenance.json',meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['full','sft-eval'])
    for name in ['bundle','completed','source-launch','output','username']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--training-bundle')
    args = vars(parser.parse_args());kind = args.pop('kind')
    if kind == 'full':
        args.pop('training_bundle');stage_full(**args)
    else:
        if not args['training_bundle']:parser.error('--training-bundle required for sft-eval')
        stage_eval(**args)
