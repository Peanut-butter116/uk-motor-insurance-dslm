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
    ('ministral3b-sft-openbook','E4: Ministral QLoRA-SFT + RAG'),
    ('qwen3b-sft-openbook','E6: Qwen2.5-3B QLoRA-SFT + RAG')]
SECONDARY=[('ministral3b-base-closedbook','E1: Ministral without RAG'),
    ('ministral3b-sft-closedbook','E3: Ministral QLoRA-SFT without RAG'),
    ('qwen3b-base-closedbook','Qwen2.5-3B without RAG'),
    ('qwen3b-sft-closedbook','E5: Qwen2.5-3B QLoRA-SFT without RAG')]


def render(summary):
    lines=['Aggregate results from the frozen benchmark. Pending is unmeasured, not zero.',
        'Full-test rows are the historical comparison; motor rows are supplementary.', '',
        '| Population | Model / condition | Correctness | Citation | Abstention | Composite | Parse rate | Status |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for slice_name,title in [('full','Full'),('motor','Motor')]:
        n=summary['population' if slice_name=='full' else 'motor_population']
        for label,name in PRIMARY+SECONDARY:
            r=summary['models'].get(label,{}).get(slice_name,{})
            vals=[f'{r[k]:.6f}' if r.get(k) is not None else 'Pending' for k in ['correctness','citation','abstention','composite','parse_rate']]
            lines.append('| '+f'{title} · {n}'+' | '+name+' | '+' | '.join(vals)+' | '+r.get('status','not run')+' |')
    lines += ['', 'Citation resolution is not claim entailment. No overall winner is declared while correctness is incomplete.']
    return '\n'.join(lines)+'\n'


def render_effects(summary,slice_name='full'):
    from scripts.error_analysis import sft_effects
    effects=sft_effects(summary['models'])[slice_name]
    lines=['| Metric | Ministral Base | Ministral SFT | Δ SFT | Qwen Base | Qwen SFT | Δ SFT |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for metric,cell in effects.items():
        vals=[]
        for family in ['ministral3b','qwen3b']:
            for k in ['base','sft','delta']:
                v=cell[family][k];vals.append('Pending' if v is None else (f'{v:+.6f}' if k=='delta' else f'{v:.6f}'))
        lines.append('| '+metric+' | '+' | '.join(vals)+' |')
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--aggregate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=core.private_output(a.output)
    (out/'comparison.md').write_text(render(json.loads(Path(a.aggregate).read_text())))
