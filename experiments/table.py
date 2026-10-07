"""Build the test-only comparison table; absent experiments remain explicitly unrun."""
import argparse
import json
from pathlib import Path
from .core import private_output,write_json


def table(home,metrics,output):
    baseline=json.loads((Path(home)/'baseline/abstention.json').read_text())
    result=[]
    for key in ['qwen2.5-7b-bnb-openbook','qwen2.5-7b-raft-openbook']:
        r=baseline['rows'][key]['adopted']
        if r['corr_lower']!=r['corr_upper']:raise ValueError('Historical baseline is not fully judged')
        result.append({'row':key,'composite':r['composite_lower'],'correctness':r['corr_lower'],
            'citation':r['citation'],'abstention':r['abstention'],'n':140,'status':'historical'})
    supplied={}
    for p in metrics:
        r=json.loads(Path(p).read_text())
        if r.get('split')!='test' or r.get('n')!=140:raise ValueError('Do not mix dev/smoke scores with historical test results')
        if r['row'] in supplied:raise ValueError('Duplicate model row')
        supplied[r['row']]=r
    for model in ['qwen3b-base','ministral3b-base','ministral3b-sft']:
        for mode in ['closedbook','openbook']:
            key=f'{model}-{mode}'
            result.append(supplied.pop(key,{'row':key,'status':'not run'}))
    if supplied:raise ValueError('Unexpected result rows')
    out=private_output(output);write_json(out/'comparison.json',result)
    lines=['| Experiment | Composite | Correctness | Citation | Abstention | Status |',
        '|---|---:|---:|---:|---:|---|']
    for r in result:
        vals=[f"{r[k]:.6f}" if k in r else '—' for k in ['composite','correctness','citation','abstention']]
        lines.append('| '+r['row']+' | '+' | '.join(vals)+' | '+r.get('status','scored')+' |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--metrics',nargs='*',default=[]);p.add_argument('--output',required=True)
    table(**vars(p.parse_args()))
