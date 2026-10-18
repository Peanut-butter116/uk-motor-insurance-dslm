"""Audit pinned chat templates and private prompt lengths without model weights."""
import json,hashlib,importlib.metadata,time
from pathlib import Path
from transformers import AutoTokenizer
import argparse
p=argparse.ArgumentParser(description='CPU tokenizer/length audit only; no policy text leaves this process')
p.add_argument('--bundle',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
root=Path(__file__).resolve().parents[1];private=Path(args.bundle)
cfg=json.loads((root/'experiments/2026-10-20/configs/six_models_descriptive.json').read_text());tasks=[json.loads(x) for x in (private/'tasks.jsonl').read_text().splitlines()];results=[]
for m in cfg['models']:
 try:
  tok=AutoTokenizer.from_pretrained(m['id'],revision=m['revision']);lengths=[]
  for t in tasks:
   messages=[{'role':'user','content':'\n\n'.join(x['content'] for x in t['messages'])}]
   ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True)
   lengths.append(len(ids['input_ids']))
  results.append({'model':m['id'],'revision':m['revision'],'tokenizer_class':type(tok).__name__,'status':'tokenisation_passed' if max(lengths)+700<=8192 else 'context_budget_failed','tasks':len(tasks),'min_prompt_tokens':min(lengths),'max_prompt_tokens':max(lengths),'over_budget_tasks':sum(n+700>8192 for n in lengths),'chat_template_sha256':hashlib.sha256(str(tok.chat_template).encode()).hexdigest()})
 except Exception as e:results.append({'model':m['id'],'revision':m['revision'],'status':'access_or_dependency_blocked','exception_type':type(e).__name__})
 out={'runtime_transformers':importlib.metadata.version('transformers'),'purpose':'CPU tokenisation only, not GPU compatibility or quality','models':results};Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
 print(m['id'],results[-1]['status'],flush=True)
