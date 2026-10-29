"""One isolated GPU process: fresh QLoRA smoke -> full epochs -> reload -> inference."""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import time


def run(root,key,smoke_only=False):
    from experiments import core
    from experiments.train import collate,validate_trainable
    from experiments import reload_check as check
    from experiments.six_model_benchmark import normalise_answer
    import torch
    from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel,get_peft_model_state_dict
    root=Path(root);cfg=json.loads((root/'config.json').read_text());core.config=lambda:cfg
    check.config=lambda:cfg
    set_seed(cfg['seed']);out=root/'runs'/key;out.mkdir(parents=True,exist_ok=True)
    bundle=root/'bundle';manifest=json.loads((bundle/'manifest.json').read_text())
    hashes=json.loads((bundle/'bundle_hashes.json').read_text())
    for split in ('train','validation'):
        p=bundle/f'{key}-{split}.jsonl';assert core.sha(p)==hashes[p.name]
    data=core.rows(bundle/f'{key}-train.jsonl');valid=core.rows(bundle/f'{key}-validation.jsonl')
    for r in data+valid:r.pop('source_record_id')
    versions={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','accelerate')}
    tc=cfg['training'];started=time.monotonic()
    spec=cfg['models'][key];gates={}
    core.write_json(out/'configuration.json',{'config':cfg,'model':spec,'bundle':manifest,'versions':versions,
        'gpu':torch.cuda.get_device_name(0),'visible_gpu_count':torch.cuda.device_count(),
        'quantisation':'NF4,double quantisation; BF16 if supported else FP16','scheduler':'linear',
        'batch_size':1,'gradient_accumulation':tc['gradient_accumulation_steps'],
        'checkpoint_selection':'fixed final complete epoch; validation logging only'})
    assert torch.cuda.device_count()==1,'Each process must see exactly its assigned GPU'

    def fresh(adapter=None):
        model,tok=core.load_model(key)
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=adapter is None)
        if adapter:model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
        return model,tok

    def train(mode):
        model,tok=fresh()
        suffixes={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
        targets=[n for n,m in model.named_modules() if n.rsplit('.',1)[-1] in suffixes
                 and getattr(getattr(m,'weight',None),'ndim',0)==2
                 and not any(x in n.lower() for x in ('vision','visual','projector'))]
        if spec['class']=='mistral3':targets=[n for n in targets if 'language_model' in n.split('.')]
        assert targets,'No language-only linear projections'
        model=get_peft_model(model,LoraConfig(r=tc['r'],lora_alpha=tc['alpha'],
            lora_dropout=tc['dropout'],bias='none',target_modules=targets,task_type='CAUSAL_LM'))
        names=validate_trainable(model,targets);model.config.use_cache=False
        initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()} if mode=='smoke' else None
        dest=out/mode;dest.mkdir(exist_ok=True)
        args=TrainingArguments(output_dir=str(dest/'checkpoints'),per_device_train_batch_size=1,
            per_device_eval_batch_size=1,gradient_accumulation_steps=tc['gradient_accumulation_steps'],
            num_train_epochs=tc['epochs'],max_steps=tc['smoke_steps'] if mode=='smoke' else -1,
            learning_rate=tc['learning_rate'],seed=cfg['seed'],data_seed=cfg['seed'],
            bf16=torch.cuda.is_bf16_supported(),fp16=not torch.cuda.is_bf16_supported(),
            # Completed checkpoints remain immutable while the coordinator
            # archives them. Trainer deletion would race cloud snapshots.
            gradient_checkpointing=True,save_strategy='steps',save_steps=25,save_total_limit=None,
            eval_strategy='steps' if mode=='smoke' else 'epoch',eval_steps=1,
            logging_steps=1,report_to='none',remove_unused_columns=False,load_best_model_at_end=False)
        class CheckpointMarker(TrainerCallback):
            def on_save(self,args,state,control,**kw):
                core.write_json(Path(args.output_dir)/f'checkpoint-{state.global_step}'/'COMPLETE.json',
                                {'step':state.global_step,'complete':True})
        trainer=Trainer(model=model,args=args,train_dataset=data,eval_dataset=valid[:2] if mode=='smoke' else valid,callbacks=[CheckpointMarker()],
            data_collator=lambda batch:collate(batch,tok.pad_token_id))
        before=time.monotonic();torch.cuda.reset_peak_memory_stats()
        checkpoints=list((dest/'checkpoints').glob('checkpoint-*'))
        resume=str(max(checkpoints,key=lambda p:int(p.name.split('-')[-1]))) if checkpoints else None
        result=trainer.train(resume_from_checkpoint=resume)
        assert math.isfinite(result.training_loss) and result.training_loss>0
        if mode=='full':assert trainer.state.epoch>=tc['epochs']-.001,'Partial epochs cannot pass'
        validation=trainer.evaluate();assert math.isfinite(validation['eval_loss'])
        if initial:assert any(not torch.equal(initial[n],v.cpu()) for n,v in get_peft_model_state_dict(model).items())
        adapter=dest/'adapter';model.save_pretrained(adapter);tok.save_pretrained(adapter)
        core.write_json(dest/'training.json',{'loss':result.training_loss,'validation':validation,
            'log_history':trainer.state.log_history,'steps':result.global_step,'epoch':trainer.state.epoch,
            'seconds':time.monotonic()-before,'samples_per_second':result.metrics.get('train_samples_per_second'),
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'adapter_sha256':core.sha(adapter/'adapter_model.safetensors')})
        model=check.unwrap_for_reload(model,trainer.accelerator)
        # Use a training-only fixed prompt; never use the protected test for gates.
        sample=data[0];n=next(i for i,x in enumerate(sample['labels']) if x!=-100)
        ids=torch.tensor([sample['input_ids'][:n]]);mask=torch.ones_like(ids)
        pre=check.probe(model,tok,ids,mask,key)
        expected={n:v.cpu().clone() for n,v in get_peft_model_state_dict(model).items()}
        del model,trainer,initial;gc.collect();torch.cuda.empty_cache()
        restored,tok=fresh(str(adapter));post=check.probe(restored,tok,ids,mask,key)
        equality=check.adapter_check(expected,get_peft_model_state_dict(restored),names)
        comparison=check.compare(pre,post);core.write_json(dest/'reload.json',{'comparison':comparison,'weights':equality})
        assert equality['equal'];check.require_comparison(comparison)
        del restored,expected;gc.collect();torch.cuda.empty_cache()
        core.write_json(dest/'receipt.json',{'mode':mode,'passed':True,'model':spec,'versions':versions,
            'dataset_manifest_sha256':core.sha(bundle/'manifest.json'),'adapter_sha256':core.sha(adapter/'adapter_model.safetensors')})
        return adapter

    smoke=out/'smoke/receipt.json'
    if not smoke.exists():train('smoke')
    assert json.loads(smoke.read_text())['passed'] is True
    if smoke_only:return
    if not (out/'full/receipt.json').exists():train('full')
    # Independent base/SFT, identical saved tasks. No historical output adoption
    # without exact protocol equivalence; this campaign reruns all conditions.
    tasks=core.rows(root/'evaluation/tasks.jsonl')
    assert len(tasks)==152 and len({t['qid'] for t in tasks})==76
    for label,adapter in [('sft',out/'full/adapter'),('base',None)]:
        output=out/f'{label}-responses.jsonl'
        done={r['task_id'] for r in core.rows(output)} if output.exists() else set()
        if len(done)==152:continue
        model,tok=fresh(str(adapter)) if adapter else core.load_model(key)
        model.eval()
        with output.open('a') as f:
            for task in tasks:
                identity=task['qid']+'/'+task['mode']
                if identity in done:continue
                ids=core.token_ids(tok,task['messages'],True)
                assert len(ids)+cfg['evaluation']['max_new_tokens']<=cfg['evaluation']['max_total_tokens'],'No evidence truncation permitted'
                x=torch.tensor([ids],device=model.device);t=time.monotonic()
                with torch.inference_mode():y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,
                    max_new_tokens=700,pad_token_id=tok.pad_token_id)
                raw=tok.decode(y[0,len(ids):],skip_special_tokens=True)
                parsed=normalise_answer(raw,key=='deepseek1b5')
                row={'task_id':identity,'qid':task['qid'],'condition':label+('-rag' if task['mode']=='openbook' else ''),
                    'model':spec,'raw_answer':raw,**parsed,'seconds':time.monotonic()-t,
                    'prompt_tokens':len(ids),'output_tokens':len(y[0])-len(ids),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
        del model;gc.collect();torch.cuda.empty_cache()
    core.write_json(out/'COMPLETE.json',{'model':spec,'status':'complete','responses':304,
        'seconds':time.monotonic()-started,'adapter_sha256':core.sha(out/'full/adapter/adapter_model.safetensors')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',required=True);p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args()
    try:run(args.root,args.model,args.smoke_only)
    except Exception as e:
        import traceback
        out=Path(args.root)/'runs'/args.model;out.mkdir(parents=True,exist_ok=True)
        (out/'FAILED.json').write_text(json.dumps({'type':type(e).__name__,'reason':str(e),'traceback':traceback.format_exc()}))
        raise
