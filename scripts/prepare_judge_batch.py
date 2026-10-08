"""Pool existing private correctness tasks, hiding model and condition metadata."""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core


def prepare(sources,output,seed=0):
    pending=[];seen=set()
    for source in sources:
        source=Path(source)
        tasks=core.unique(core.rows(source/'judge_tasks.jsonl'),'task')
        mapping=core.unique(core.rows(source/'judge_map.jsonl'),'task')
        if set(tasks)!=set(mapping):raise ValueError('Judging tasks and mapping differ')
        for key,t in tasks.items():
            m=mapping[key];identity=(m['qid'],m['row'])
            if identity in seen:raise ValueError('Duplicate answer in judge batch')
            seen.add(identity)
            # Explicit allowlist: QID, model, experiment/condition and source path
            # can never become task fields via an input schema extension.
            task={k:t[k] for k in ['question','gold_answer','key_facts','model_answer']}
            task['must_not_assert']=t.get('must_not_assert','')
            pending.append((task,{'qid':m['qid'],'row':m['row']}))
    if not pending:raise ValueError('Empty judging population')
    random.Random(seed).shuffle(pending)
    tasks=[];mapping=[]
    for i,(t,m) in enumerate(pending):
        code=f'j{i:05d}';tasks.append({'task':code,**t});mapping.append({'task':code,**m})
    out=core.private_output(output)
    core.write_rows(out/'tasks.jsonl',tasks);core.write_rows(out/'mapping.jsonl',mapping)
    (out/'rubric.md').write_text((core.ROOT/'eval/JUDGE_RUBRIC.md').read_text())
    core.write_json(out/'status.json',{'status':'awaiting_judging','tasks':len(tasks),
        'required_votes':3*len(tasks),'shuffle_seed':seed,
        'instructions':'Give each of three independent judge runs tasks.jsonl and rubric.md only. Never supply mapping.jsonl, source folders, model identities or other votes.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--output',required=True);prepare(**vars(p.parse_args()))
