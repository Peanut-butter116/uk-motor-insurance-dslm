"""CUDA smoke-first QLoRA. Full mode requires a matching successful GPU receipt."""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
from .core import (ROOT, config, digest, encode_training, generate, load_model,
    private_output, sha, token_ids, write_json)
from .prepare_sft import validate_bundle


def code_hash():
    return digest({str(p.relative_to(ROOT)):sha(p) for folder in ['experiments','src','eval']
        for p in sorted((ROOT/folder).glob('*.py'))})


def fingerprint(manifest):
    return digest({'training_manifest':manifest,'config':config(),'code':code_hash()})


def require_smoke(receipt,manifest):
    r=json.loads(Path(receipt).read_text())
    gates=['leakage','tokenisation','masking','finite_loss','language_only','adapter_reload']
    if r.get('mode')!='smoke' or r.get('fingerprint')!=fingerprint(manifest) or any(r.get('gates',{}).get(g) is not True for g in gates):
        raise ValueError('Full training blocked: smoke receipt missing, stale or failed')
    return r


def language_targets(model):
    suffixes={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    names=[name for name,_ in model.named_modules() if 'language_model' in name.split('.')
        and name.rsplit('.',1)[-1] in suffixes and not any(x in name.lower() for x in ('vision','visual','projector'))]
    if not names:raise ValueError('No language-only adapter targets found')
    return names


def collate(batch,pad_id):
    import torch
    n=max(len(x['input_ids']) for x in batch)
    return {k:torch.tensor([r[k]+[pad]*(n-len(r[k])) for r in batch])
        for k,pad in [('input_ids',pad_id),('attention_mask',0),('labels',-100)]}


def run(bundle,output,mode='smoke',receipt=None):
    data,manifest=validate_bundle(bundle)
    if manifest.get('tokenisation_checked') is not True:
        raise ValueError('Run experiments.tokenise before training')
    old_receipt=None
    if mode=='full':
        if not receipt:raise ValueError('Full training requires --smoke-receipt')
        old_receipt=require_smoke(receipt,manifest)
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel
    versions={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes','accelerate']}
    if old_receipt and versions!=old_receipt['versions']:
        raise ValueError('Runtime differs from the successful smoke; repeat smoke first')
    set_seed(config()['seed'])
    out=private_output(output);cfg=config()['training']
    model,tokenizer=load_model('ministral3b')
    # Preflight EVERY eligible row, even for a two-step smoke. Never truncate.
    encoded=[encode_training(tokenizer,r['messages'],cfg['max_length']) for r in data]
    write_json(out/'tokenisation.json',{'rows':len(data),'lengths':[len(r['input_ids']) for r in encoded],
        'supervised_tokens':[sum(t!=-100 for t in r['labels']) for r in encoded]})
    targets=language_targets(model)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model=get_peft_model(model,LoraConfig(r=cfg['r'],lora_alpha=cfg['alpha'],lora_dropout=cfg['dropout'],
        bias='none',target_modules=targets,task_type='CAUSAL_LM'))
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n or 'language_model' not in n or any(s in n for s in ('vision','projector')) for n in trainable):
        raise ValueError('Non-language adapter parameter became trainable')
    from peft import get_peft_model_state_dict
    initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    model.config.use_cache=False
    args=TrainingArguments(output_dir=str(out/'checkpoints'),per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],num_train_epochs=cfg['epochs'],
        max_steps=cfg['smoke_steps'] if mode=='smoke' else -1,learning_rate=cfg['learning_rate'],
        seed=config()['seed'],data_seed=config()['seed'],bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),gradient_checkpointing=True,
        save_strategy='no',logging_steps=1,report_to='none',remove_unused_columns=False)
    trainer=Trainer(model=model,args=args,train_dataset=encoded,
        data_collator=lambda b:collate(b,tokenizer.pad_token_id))
    result=trainer.train()
    if not math.isfinite(result.training_loss) or result.training_loss<=0:
        raise ValueError('Invalid smoke loss')
    updated=get_peft_model_state_dict(model)
    if not any(not torch.equal(initial[k],updated[k].detach().cpu()) for k in initial):
        raise ValueError('Smoke did not update any adapter weights')
    del initial,updated
    adapter=out/'adapter';model.save_pretrained(adapter);tokenizer.save_pretrained(adapter)
    write_json(out/'training.json',{'loss':result.training_loss,'log':trainer.state.log_history,'trainable':trainable})
    from . import reload_check as check
    from safetensors.torch import load_file
    import inspect
    probe_messages=data[0]['messages'][:-1]
    ids_cpu=torch.tensor([token_ids(tokenizer,probe_messages,True)])
    mask_cpu=torch.ones_like(ids_cpu)
    before=check.probe(model,tokenizer,ids_cpu,mask_cpu)
    saved={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    disk=load_file(str(adapter/'adapter_model.safetensors'))
    disk_check=check.adapter_check(saved,disk,trainable)
    report={'input_ids_sha256':digest(ids_cpu.tolist()), 'attention_mask_sha256':digest(mask_cpu.tolist()),
        'before_runtime':before['runtime'], 'before_generated_tokens':before['tokens'],
        'before_generated_text':before['text'], 'saved_adapter':disk_check}
    write_json(out/'reload_diagnostics.json',report)
    # Preserve the exact installed preparation code as private diagnostic evidence.
    (out/'prepare_model_for_kbit_training.txt').write_text(inspect.getsource(prepare_model_for_kbit_training))
    if not disk_check['equal']:raise ValueError('Saved adapter does not contain all trained weights')
    del trainer,model,disk;gc.collect();torch.cuda.empty_cache()
    comparisons={}
    # First reproduce the old reload, then isolate base preparation as the only
    # changed operation. Both paths use the same fixed IDs/mask and pinned loader.
    for label,prepare_base in [('unprepared_reload',False),('prepared_reload',True)]:
        fresh,fresh_tokenizer=load_model('ministral3b')
        tokenizer_equal=token_ids(fresh_tokenizer,probe_messages,True)==ids_cpu[0].tolist()
        if prepare_base:
            fresh=prepare_model_for_kbit_training(fresh,use_gradient_checkpointing=False)
        fresh=PeftModel.from_pretrained(fresh,adapter,is_trainable=False)
        restored_check=check.adapter_check(saved,get_peft_model_state_dict(fresh),trainable)
        after=check.probe(fresh,fresh_tokenizer,ids_cpu,mask_cpu)
        comparison=check.compare(before,after)
        comparison.update(tokenizer_equal=tokenizer_equal,restored_adapter=restored_check)
        comparisons[label]=comparison
        report[label]={**comparison,'runtime':after['runtime'],
            'generated_tokens':after['tokens'],'generated_text':after['text']}
        write_json(out/'reload_diagnostics.json',report)
        print(label,json.dumps(comparison),flush=True)
        del fresh,after;gc.collect();torch.cuda.empty_cache()
    final=comparisons['prepared_reload']
    if not final['tokenizer_equal'] or not final['restored_adapter']['equal']:
        raise ValueError('Reload tokenisation or adapter weights differ')
    check.require_comparison(final)
    write_json(out/'receipt.json',{'mode':mode,'fingerprint':fingerprint(manifest),'versions':versions,
        'gates':dict.fromkeys(['leakage','tokenisation','masking','finite_loss','language_only','adapter_reload'],True),
        'rows':len(data),'steps':trainer_steps(result),
        'reload_requires_kbit_preparation':True,
        'max_logit_difference':final['max_absolute_difference'],
        'mean_logit_difference':final['mean_absolute_difference'],
        'diagnostics_sha256':sha(out/'reload_diagnostics.json')})



def trainer_steps(result):return result.global_step

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke');p.add_argument('--smoke-receipt')
    a=p.parse_args();run(a.bundle,a.output,a.mode,a.smoke_receipt)
