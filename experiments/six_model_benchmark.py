"""Kaggle CUDA runner for reviewed motor development tasks or synthetic probes.

Never accepts test data for selection. Records raw outputs and all skips.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import re


def validate_population(cfg, manifest, tasks_bytes, compatibility_probe=False):
    """Keep protected-test descriptive evaluation distinct from dev selection."""
    if hashlib.sha256(tasks_bytes).hexdigest()!=manifest['tasks_sha256']:
        raise ValueError('Task hash mismatch')
    if compatibility_probe:
        if manifest.get('purpose')!='public_synthetic_compatibility_probe':
            raise ValueError('Probe requires entirely synthetic inputs')
    elif manifest.get('purpose')=='frozen_motor_test_descriptive':
        config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if (manifest.get('split')!='test' or manifest.get('selection_permitted') is not False
                or manifest.get('protocol_frozen_before_submission') is not True
                or manifest.get('config_sha256')!=config_hash):
            raise ValueError('Protected test requires a pre-frozen descriptive-only protocol')
    elif (manifest.get('split')!='dev' or manifest.get('independent_review_complete') is not True
          or manifest.get('training_leakage_screen_complete') is not True
          or manifest.get('protected_semantic_screen_complete') is not True):
        raise ValueError('Reviewed independent motor dev population required')
    tasks=[json.loads(x) for x in tasks_bytes.decode().splitlines() if x]
    if not tasks or len({(x['qid'],x['mode']) for x in tasks})!=len(tasks):
        raise ValueError('Empty or duplicate tasks')
    qids={x['qid'] for x in tasks}
    if len(qids)!=manifest['questions']:
        raise ValueError('Question count differs from frozen manifest')
    if any({t['mode'] for t in tasks if t['qid']==q}!={'closedbook','openbook'} for q in qids):
        raise ValueError('Every QID requires matched closedbook and openbook tasks')
    return tasks


def normalise_answer(raw, reasoning_expected=False):
    """Retain raw reasoning separately and never score an unfinished thought as final."""
    if (reasoning_expected or '<think>' in raw) and '</think>' not in raw:
        return {'answer':'', 'reasoning_truncated':True, 'final_answer_present':False}
    if '</think>' in raw:
        answer=raw.rsplit('</think>',1)[1].strip()
    else:
        answer=re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
    return {'answer':answer,'reasoning_truncated':False,'final_answer_present':bool(answer)}


def run(config, bundle, output, compatibility_probe=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation required')
    cfg=json.loads(Path(config).read_text())
    bundle=Path(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text())
    tasks_bytes=(bundle/'tasks.jsonl').read_bytes()
    tasks=validate_population(cfg,manifest,tasks_bytes,compatibility_probe)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    runtime={p:importlib.metadata.version(p) for p in ('torch','transformers','accelerate','bitsandbytes')}
    statuses=[]
    for item in cfg['models']:
        key=item['id'].replace('/','--')
        started=time.monotonic()
        model=None
        completed=0
        try:
            # Do not accept licence terms automatically or substitute checkpoints.
            tok=AutoTokenizer.from_pretrained(item['id'],revision=item['revision'])
            cls=AutoModelForCausalLM
            if item['class']=='mistral3':
                from transformers import Mistral3ForConditionalGeneration
                cls=Mistral3ForConditionalGeneration
            loading={'device_map':{'':0},'dtype':torch.float16,'attn_implementation':'sdpa'}
            if cfg.get('loading_precision')=='float16':
                if item['class']=='mistral3':
                    # Official FP8 weights require dequantisation on a T4.
                    # Preserve the exact requested checkpoint and revision.
                    from transformers import FineGrainedFP8Config
                    loading['quantization_config']=FineGrainedFP8Config(dequantize=True)
            else:
                loading['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
            model=cls.from_pretrained(item['id'],revision=item['revision'],**loading)
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
                    text_config=getattr(model.config,'text_config',model.config)
                    limit=min(cfg['generation']['max_total_tokens'],getattr(text_config,'max_position_embeddings',8192))
                    if n+cfg['generation']['max_new_tokens']>limit:
                        raise ValueError('Prompt exceeds context budget; no evidence truncation allowed')
                    torch.cuda.synchronize();start=time.monotonic()
                    with torch.inference_mode():
                        y=model.generate(**x,do_sample=False,max_new_tokens=cfg['generation']['max_new_tokens'],
                            pad_token_id=tok.pad_token_id or tok.eos_token_id)
                    torch.cuda.synchronize()
                    answer=tok.decode(y[0,n:],skip_special_tokens=True)
                    f.write(json.dumps({'qid':t['qid'],'mode':t['mode'],'model':item,
                        'raw_answer':answer,**normalise_answer(answer,'DeepSeek-R1' in item['id']),
                        'prompt_tokens':n,'generated_tokens':y.shape[-1]-n,
                        'hit_output_cap':y.shape[-1]-n>=cfg['generation']['max_new_tokens'],
                        'latency_seconds':time.monotonic()-start,
                        'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                        'tokenizer_class':type(tok).__name__,'loading_precision':cfg.get('loading_precision','nf4')})+'\n');f.flush()
                    completed+=1
            output_file=out/(key+'.jsonl')
            statuses.append({'model':item['id'],'revision':item['revision'],
                'status':'compatibility_probe_complete' if compatibility_probe else 'inference_complete',
                'tasks_completed':completed,'runtime_seconds':time.monotonic()-started,
                'answers_sha256':hashlib.sha256(output_file.read_bytes()).hexdigest()})
        except Exception as exc:
            # Exception text can include private inputs or credential-bearing URLs.
            # Detailed diagnostics stay with private Kaggle outputs, never summaries.
            import traceback
            (out/(key+'-failure.log')).write_text(traceback.format_exc())
            statuses.append({'model':item['id'],'revision':item['revision'],'status':'failed',
                'exception_type':type(exc).__name__,'tasks_completed':completed,
                'runtime_seconds':time.monotonic()-started})
        finally:
            del model
            import gc
            gc.collect();torch.cuda.empty_cache()
        # Per-model completion survives a later model failure or session timeout.
        (out/'model_status.json').write_text(json.dumps(statuses,indent=2)+'\n')
    result={'purpose':manifest.get('purpose','motor_dev'),'config':cfg,'input_manifest':manifest,
            'gpu':torch.cuda.get_device_name(0),'runtime':runtime,'models':statuses,
            'warning':'No model selection from protected-test descriptive results or compatibility probes'}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','bundle','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--compatibility-probe',action='store_true')
    run(**vars(p.parse_args()))
