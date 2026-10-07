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
    probe=data[0]['messages'][:-1]
    model.eval();model.config.use_cache=True
    ids=torch.tensor([token_ids(tokenizer,probe,True)],device=model.device)
    with torch.inference_mode():before=model(input_ids=ids).logits[:,-1,:].float().cpu()
    from peft import get_peft_model_state_dict
    saved={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    del trainer,model,ids;gc.collect();torch.cuda.empty_cache()
    fresh,fresh_tokenizer=load_model('ministral3b')
    fresh=PeftModel.from_pretrained(fresh,adapter,is_trainable=False);fresh.eval()
    if token_ids(fresh_tokenizer,probe,True)!=token_ids(tokenizer,probe,True):raise ValueError('Reload tokenisation drift')
    restored=get_peft_model_state_dict(fresh)
    if set(saved)!=set(restored) or any(not torch.equal(saved[k],restored[k].detach().cpu()) for k in saved):
        raise ValueError('Adapter weights differ after reload')
    ids=torch.tensor([token_ids(fresh_tokenizer,probe,True)],device=fresh.device)
    with torch.inference_mode():after=fresh(input_ids=ids).logits[:,-1,:].float().cpu()
    if not torch.allclose(before,after,atol=0.02,rtol=0.01):raise ValueError('Reload logits differ')
    answer,_=generate(fresh,fresh_tokenizer,probe,max_new_tokens=32)
    if not answer.strip():raise ValueError('Reload generation empty')
    write_json(out/'receipt.json',{'mode':mode,'fingerprint':fingerprint(manifest),'versions':versions,
        'gates':dict.fromkeys(['leakage','tokenisation','masking','finite_loss','language_only','adapter_reload'],True),
        'rows':len(data),'steps':trainer_steps(result),'max_logit_difference':(before-after).abs().max().item()})


def trainer_steps(result):return result.global_step

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke');p.add_argument('--smoke-receipt')
    a=p.parse_args();run(a.bundle,a.output,a.mode,a.smoke_receipt)
