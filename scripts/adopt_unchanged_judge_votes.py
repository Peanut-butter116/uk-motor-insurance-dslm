"""Preserve valid work across a reference-package repair, only for identical tasks."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core
from scripts.run_semantic_judge import secure


def unchanged_ids(old,new,old_mapping,new_mapping):
    return {eid for eid in old.keys()&new.keys() if old[eid]==new[eid] and old_mapping[eid]==new_mapping[eid]}


def adopt(source, target):
    source,target=secure(source),secure(target)
    old=core.unique(core.rows(source/'judge_inputs.jsonl'),'evaluation_id');new=core.unique(core.rows(target/'judge_inputs.jsonl'),'evaluation_id')
    old_mapping=core.unique(core.rows(source/'private_mapping.jsonl'),'evaluation_id');new_mapping=core.unique(core.rows(target/'private_mapping.jsonl'),'evaluation_id')
    unchanged=unchanged_ids(old,new,old_mapping,new_mapping)
    if (target/'judge_votes.jsonl').stat().st_size or (target/'judge_attempts.jsonl').stat().st_size:
        raise ValueError('Adoption requires a fresh target journal')
    accepted=[];attempts=[]
    for name,output in [('judge_votes.jsonl',accepted),('judge_attempts.jsonl',attempts)]:
        for r in core.rows(source/name):
            if r['evaluation_id'] in unchanged:
                output.append({**r,'request_id':'adopted:'+r.get('request_id','unknown'),
                    'adopted_from_run':source.name,'adopted_input_sha256':core.digest(new[r['evaluation_id']])})
    core.write_rows(target/'judge_votes.jsonl',accepted);core.write_rows(target/'judge_attempts.jsonl',attempts)
    (target/'judge_requests.jsonl').touch(mode=0o600)
    shutil.copyfile(source/'judge_requests.jsonl',target/'adopted_request_receipts.jsonl')
    for name in ['gemini_usage.jsonl','gemini_semantic_permission.json','gemini_free_quota_verified_20261018.json','evidence_groups.json','calibration_indexed_v3.json']:
        shutil.copyfile(source/name,target/name)
    manifest=json.loads((target/'judge_manifest.json').read_text())
    manifest.update(adopted_votes=len(accepted),adopted_attempts=len(attempts),unchanged_task_n=len(unchanged),
        changed_task_n=len(new)-len(unchanged),adopted_source_run=source.name,
        adopted_source_inputs_sha256=core.sha(source/'judge_inputs.jsonl'),
        adopted_source_votes_sha256=core.sha(source/'judge_votes.jsonl'),
        adoption_rule='Only exactly identical canonical judging inputs; no changed task votes transferred',
        permission_status='owner_authorised_full_semantic_payload')
    core.write_json(target/'judge_manifest.json',manifest)
    print('Adopted valid votes:',len(accepted),'unchanged tasks:',len(unchanged),'changed tasks:',len(new)-len(unchanged))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--target',required=True)
    adopt(**vars(p.parse_args()))
