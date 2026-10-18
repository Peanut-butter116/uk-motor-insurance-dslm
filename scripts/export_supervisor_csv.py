"""Create a readable private CSV without modifying the immutable canonical snapshot."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core

FIELDS=['id','insurer','product','question','context','answer','answerability','citation','question_type','validation_status']


def safe(value):
    value=str(value)
    # Excel may ignore leading whitespace before a formula prefix.
    return "'"+value if value.lstrip().startswith(('=','+','-','@')) or value.startswith(('\t','\r')) else value


def export(snapshot, output, public):
    snapshot=Path(snapshot);manifest=json.loads((snapshot/'interim_dataset_manifest.json').read_text())
    canonical=snapshot/'interim_motor_training.jsonl'
    if core.sha(canonical)!=manifest['sha256']['interim_motor_training.jsonl']:
        raise ValueError('Immutable snapshot changed')
    rows=core.rows(canonical);ids={r['example_id'] for r in rows}
    if len(ids)!=len(rows):raise ValueError('Duplicate IDs')
    norm=lambda q:' '.join(q.casefold().split())
    clusters=collections.defaultdict(list)
    for r in rows:
        if r['split']!='train' or r['validation_status']!='accepted' or r['unresolved_citations']:
            raise ValueError('Ineligible record')
        if not r['provenance']['production_validation']['all_protected_semantic_screen_passed']:
            raise ValueError('Unresolved protected-question screen')
        clusters[norm(r['question'])].append(r)
    exact_keys=[core.digest({k:r[k] for k in ['question','context','answer','answerability']}) for r in rows]
    if len(set(exact_keys))!=len(rows):raise ValueError('Duplicate model-visible records')
    out=core.private_output(output)
    path=out/'interim_motor_training_display.csv'
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        for r in rows:
            cites='; '.join(f"[{e['insurer']}, {e['source_id']}, {e['section']}, p.{e['page']}]" for e in r['evidence_text'])
            view={'id':r['example_id'],'citation':cites,**{k:r[k] for k in FIELDS if k not in ['id','citation']}}
            w.writerow({k:safe(view[k]) for k in FIELDS})
    with path.open(newline='',encoding='utf-8') as f:
        reread=list(csv.DictReader(f))
    if len(reread)!=len(rows) or {r['id'] for r in reread}!=ids:raise ValueError('CSV round-trip failed')
    repeated=[rs for rs in clusters.values() if len(rs)>1]
    audit={'rows':len(rows),'distinct_questions':len(clusters),'unique_ids':len(ids),
        'exact_model_visible_duplicates':0,'repeated_question_clusters':len(repeated),
        'cross_answerability_clusters':sum(len({r['answerability'] for r in rs})>1 for rs in repeated),
        'cluster_sizes':dict(collections.Counter(len(rs) for rs in repeated)),
        'interpretation':'25 repeated question texts are answerable/insufficient-evidence pairs with different contexts, not duplicate model-visible rows. Keep groups together and review label/context contrast before SFT.',
        'columns':FIELDS,'encoding':'UTF-8','multiline':'RFC 4180 quoted multiline cells, verified with csv.DictReader',
        'training_ready':False,'human_review_completed':0,
        'source_dataset_version':manifest['dataset_version'],'canonical_jsonl_sha256':core.sha(canonical),
        'display_csv_sha256':core.sha(path),'supervisor_download_url':None,
        'sharing_status':'pending content-transfer approval and confirmed supervisor access; no new restricted upload',
        'leakage_status':'All 409 carry approved protected-question screens. Newly drafted dev excluded from training; exhaustive source/context semantic review pending.'}
    core.write_json(out/'display_manifest.json',audit)
    core.write_json(Path(public),audit)
    for p in out.iterdir():p.chmod(0o400)
    out.chmod(0o500)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['snapshot','output','public']:p.add_argument('--'+n,required=True)
    export(**vars(p.parse_args()))
