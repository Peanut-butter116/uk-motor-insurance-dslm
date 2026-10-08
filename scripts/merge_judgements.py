"""Merge exactly three blinded votes through the inherited merge rule."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core


def merge(tasks,mapping,votes,output):
    if len(votes)!=3 or len({str(Path(p).resolve()) for p in votes})!=3:
        raise ValueError('Exactly three separate vote files required')
    tasks=core.unique(core.rows(tasks),'task');mapping=core.unique(core.rows(mapping),'task')
    if set(tasks)!=set(mapping):raise ValueError('Task map differs from judging population')
    module=core.load_module('_merge_votes','eval/judge_votes.py')
    batches=[core.unique(core.rows(p),'task') for p in votes]
    if any(set(b)!=set(tasks) for b in batches):raise ValueError('Incomplete or extra vote population')
    result=[]
    for code,t in tasks.items():
        vv=[]
        for i,b in enumerate(batches,1):
            v=b[code];spans=v.get('evidence_spans')
            if not isinstance(spans,list) or any(not isinstance(s,str) or (s and s not in t['model_answer']) for s in spans):
                raise ValueError('Evidence spans must quote the model answer')
            vv.append({**v,'vote':i})
        merged=module.merge_task(code,vv,len(t['key_facts']),min_valid=3)
        result.append({**merged,'qid':mapping[code]['qid'],'row':mapping[code]['row']})
    out=core.private_output(output)
    core.write_rows(out/'scores.jsonl',result)
    core.write_json(out/'provenance.json',{'votes':[{'sha256':core.sha(p)} for p in votes],
        'tasks':len(tasks),'votes_per_task':3,'merge':'eval/judge_votes.py:merge_task; min_valid=3'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['tasks','mapping','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--votes',nargs=3,required=True)
    merge(**vars(p.parse_args()))
