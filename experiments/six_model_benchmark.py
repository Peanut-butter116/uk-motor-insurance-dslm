"""Kaggle CUDA runner for reviewed motor development tasks or synthetic probes.

Never accepts test data for selection. Records raw outputs and all skips.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def run(config, bundle, output, compatibility_probe=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation required')
    cfg=json.loads(Path(config).read_text())
    bundle=Path(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text())
    tasks_bytes=(bundle/'tasks.jsonl').read_bytes()
    if hashlib.sha256(tasks_bytes).hexdigest()!=manifest['tasks_sha256']:
        raise ValueError('Task hash mismatch')
    tasks=[json.loads(x) for x in tasks_bytes.decode().splitlines() if x]
    if compatibility_probe:
        if manifest.get('purpose')!='public_synthetic_compatibility_probe':
            raise ValueError('Probe requires entirely synthetic inputs')
    elif (manifest.get('split')!='dev' or manifest.get('independent_review_complete') is not True
          or manifest.get('training_leakage_screen_complete') is not True
          or manifest.get('protected_semantic_screen_complete') is not True):
        raise ValueError('Reviewed independent motor dev population required')
    if not tasks or len({(x['qid'],x['mode']) for x in tasks})!=len(tasks):
        raise ValueError('Empty or duplicate tasks')
    qids={x['qid'] for x in tasks}
    if any({t['mode'] for t in tasks if t['qid']==q}!={'closedbook','openbook'} for q in qids):
        raise ValueError('Every QID requires matched closedbook and openbook tasks')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    runtime={p:importlib.metadata.version(p) for p in ('torch','transformers','accelerate','bitsandbytes')}
    statuses=[]
    for item in cfg['models']:
        key=item['id'].replace('/','--')
        try:
            # Do not accept licence terms automatically or substitute checkpoints.
            tok=AutoTokenizer.from_pretrained(item['id'],revision=item['revision'])
            cls=AutoModelForCausalLM
            if item['class']=='mistral3':
                from transformers import Mistral3ForConditionalGeneration
                cls=Mistral3ForConditionalGeneration
            model=cls.from_pretrained(item['id'],revision=item['revision'],device_map={'':0},
                dtype=torch.float16,attn_implementation='sdpa',
                quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
            model.eval()
            torch.manual_seed(cfg['generation']['seed'])
            torch.cuda.reset_peak_memory_stats()
            with (out/(key+'.jsonl')).open('x') as f:
                for t in tasks:
                    # Single user turn preserves task instructions for families
                    # whose templates disallow system messages, including DeepSeek.
                    prompt='\n\n'.join(m['content'] for m in t['messages'])
                    x=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                        add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
                    n=x['input_ids'].shape[-1]
                    limit=min(cfg['generation']['max_total_tokens'],getattr(model.config,'max_position_embeddings',8192))
                    if n+cfg['generation']['max_new_tokens']>limit:
                        raise ValueError('Prompt exceeds context budget; no evidence truncation allowed')
                    torch.cuda.synchronize();start=time.monotonic()
                    with torch.inference_mode():
                        y=model.generate(**x,do_sample=False,max_new_tokens=cfg['generation']['max_new_tokens'],
                            pad_token_id=tok.pad_token_id or tok.eos_token_id)
                    torch.cuda.synchronize()
                    answer=tok.decode(y[0,n:],skip_special_tokens=True)
                    f.write(json.dumps({'qid':t['qid'],'mode':t['mode'],'model':item,
                        'raw_answer':answer,'prompt_tokens':n,'generated_tokens':y.shape[-1]-n,
                        'latency_seconds':time.monotonic()-start,
                        'peak_gpu_allocated_gb':torch.cuda.max_memory_allocated()/2**30})+'\n');f.flush()
            statuses.append({'model':item['id'],'status':'compatibility_probe_complete' if compatibility_probe else 'inference_complete'})
        except Exception as exc:
            # Exception text can include private inputs or credential-bearing URLs.
            statuses.append({'model':item['id'],'status':'failed','exception_type':type(exc).__name__})
        finally:
            if 'model' in locals():del model
            import gc
            gc.collect();torch.cuda.empty_cache()
    result={'purpose':manifest.get('purpose','motor_dev'),'config':cfg,'input_manifest':manifest,
            'gpu':torch.cuda.get_device_name(0),'runtime':runtime,'models':statuses,
            'warning':'Compatibility probes cannot support model-quality comparisons'}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','bundle','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--compatibility-probe',action='store_true')
    run(**vars(p.parse_args()))
