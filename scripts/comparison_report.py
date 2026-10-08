"""Render aggregate-only tables; no case text or model rankings."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core

PRIMARY=[('qwen2.5-7b-bnb-openbook','Previous Qwen2.5-7B + RAG'),
    ('qwen2.5-7b-raft-openbook','Previous QLoRA-RAFT + RAG'),
    ('qwen3b-base-openbook','Qwen2.5-3B + RAG'),
    ('ministral3b-base-openbook','E2: Ministral-3-3B + RAG'),
    ('ministral3b-sft-openbook','E4: Ministral QLoRA-SFT + RAG')]
SECONDARY=[('ministral3b-base-closedbook','E1: Ministral without RAG'),
    ('ministral3b-sft-closedbook','E3: Ministral QLoRA-SFT without RAG')]


def render(summary):
    lines=['Aggregate results from the frozen test benchmark. A dash means unmeasured, not zero.',
           'Correctness/composite stay pending until all required judgements are complete.','']
    for slice_name,title in [('full','Full frozen benchmark'),('motor','Supplementary motor-only slice')]:
        n=summary['population' if slice_name=='full' else 'motor_population']
        lines += [f'**{title} (n={n})**','']
        for names,heading in [(PRIMARY,'Open-book comparison'),(SECONDARY,'Closed-book diagnostic')]:
            lines += [heading,'','| Model | Correctness | Citation resolution | Abstention | Composite | Parse rate | Status |',
                      '|---|---:|---:|---:|---:|---:|---|']
            for label,name in names:
                r=summary['models'].get(label,{}).get(slice_name,{})
                vals=[f'{r[k]:.6f}' if r.get(k) is not None else '—' for k in ['correctness','citation','abstention','composite','parse_rate']]
                lines.append('| '+name+' | '+' | '.join(vals)+' | '+r.get('status','not run')+' |')
            lines.append('')
    lines += ['Parse rate retains the inherited definition (parseable citation or recognised abstention).',
        'Citation resolution is metadata matching, not claim entailment. No winner is declared while judging is incomplete.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--aggregate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=core.private_output(a.output)
    (out/'comparison.md').write_text(render(json.loads(Path(a.aggregate).read_text())))
