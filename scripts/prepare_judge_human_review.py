"""Prepare a blinded 28-case historical review sheet, never fill human labels."""
import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core
from scripts.run_semantic_judge import secure,csv_write


def safe(value):
    value=str(value)
    return "'"+value if value.lstrip().startswith(('=','+','-','@')) else value


def prepare(directory):
    out=secure(directory);tasks=core.unique(core.rows(out/'judge_inputs.jsonl'),'evaluation_id')
    mapping=core.unique(core.rows(out/'private_mapping.jsonl'),'evaluation_id')
    votes=core.rows(out/'judge_votes.jsonl');by_task={}
    for v in votes:by_task.setdefault(v['evaluation_id'],[]).append(v)
    rng=random.Random(20261020);pool=[eid for eid in tasks if mapping[eid]['experiment']=='historical_sft21'];rng.shuffle(pool)
    chosen=[];qids=set()
    def choose(candidates):
        for eid in candidates:
            if eid not in chosen and mapping[eid]['qid'] not in qids:
                chosen.append(eid);qids.add(mapping[eid]['qid']);return True
        return False
    arms=sorted({mapping[eid]['condition'] for eid in pool})
    # One unanswerable and two answerable cases per arm; balance observed score
    # bins where possible, without showing scores or arm identities to reviewer.
    for arm in arms:
        candidates=[e for e in pool if mapping[e]['condition']==arm]
        choose([e for e in candidates if not tasks[e]['gold_answerable']])
        for desired in (0,1):
            preferred=[e for e in candidates if tasks[e]['gold_answerable'] and any(v['correctness']==desired for v in by_task.get(e,[]))]
            if not choose(preferred):choose([e for e in candidates if tasks[e]['gold_answerable']])
    extra=[e for e in pool if tasks[e]['gold_answerable'] and any(v['correctness']==.5 or v['needs_human_review'] for v in by_task.get(e,[]))]
    while len(chosen)<28:
        if not choose(extra) and not choose(pool):raise ValueError('Insufficient unique questions for human review')
    rng.shuffle(chosen)
    target=out/'human_review_sample.csv'
    if target.exists():
        with target.open() as f:existing=list(csv.DictReader(f))
        if any(r.get('human_correctness','').strip() for r in existing):
            raise ValueError('Human labels already present: preserve and review existing sheet')
        backup=out/'human_review_sample_initial_mixed.csv'
        if backup.exists():raise FileExistsError('Review sheet already staged; do not reshuffle it')
        shutil.copyfile(target,backup)
    rows=[]
    for eid in chosen:
        t=tasks[eid]
        evidence='\n\n'.join(f'Reference {i+1}: '+c['text'] for i,c in enumerate(t['reference_evidence']))
        row={'evaluation_id':eid,'question':t['question'],'gold_answerable':t['gold_answerable'],
            'reference_answer':t['reference_answer'],'key_facts':'\n'.join(f'{i+1}. {s}' for i,s in enumerate(t['key_facts'])),
            'reference_evidence':evidence,'model_answer':t['model_answer'],
            'student_evidence_available':t['student_evidence_available'],
            'human_correctness':'','human_evidence_support':'','human_reason':''}
        rows.append({k:safe(v) if isinstance(v,str) else v for k,v in row.items()})
    csv_write(target,rows)
    (out/'human_review_instructions.md').write_text('''# Independent blind human review

Do not open the model mapping or judge votes before labelling. All 28 cases are
historical motor responses; each condition is represented, with distinct questions.
The sampled score strata are hidden. Fill human_correctness (0, 0.5, 1), optional
human_evidence_support (0, 0.5, 1; blank for no factual claim), and human_reason.
Use the supplied policy evidence, material conditions/exclusions and reference
facts. Coverage/safety errors score 0. Missing/ambiguous reference information
should be explained and left unscored, not forced to 0. Model identities and all
LLM decisions remain hidden. No human labels have been supplied by this script.
The sheet escapes spreadsheet formula prefixes; canonical inputs remain lossless
in judge_inputs.jsonl. Add reviewer identity/date in a separate private note.
''')
    core.write_json(out/'human_review_selection.json',{'seed':20261020,'n':len(rows),'distinct_questions':len(qids),
        'historical_only':True,'selection':'all eight arms; answerability and observed score/uncertainty strata; not representative accuracy sampling',
        'evaluation_ids':chosen,'sheet_sha256_before_labels':core.sha(target),'labels_completed':0})
    print('Prepared 28 blinded historical cases; human labels completed: 0')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);prepare(**vars(p.parse_args()))
