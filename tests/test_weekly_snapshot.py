"""Acceptance and round-trip checks for protected experiment snapshots."""
import csv
import json
from pathlib import Path
import subprocess

import pytest

from experiments.weekly_snapshot import freeze, sha


def source(tmp_path):
    repo = tmp_path / 'source-repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.invalid', 'commit', '--allow-empty', '-qm', 'fixture'], check=True)
    export = tmp_path / 'version-1'
    export.mkdir()
    record = {'example_id':'demo-1', 'product':'motor', 'split':'train',
              'question':'=A fictional question?', 'answer':'An invented answer.',
              'validation_status':'accepted', 'unresolved_citations':[],
              'provenance':{'production_validation':{'all_protected_semantic_screen_passed':True}},
              'evidence_text':[{'insurer':'Demo','source_id':'fiction','section':'A'}],
              'answerability':'answerable','data_origin':'generated','validation_method':'fixture'}
    (export/'train.jsonl').write_text(json.dumps(record)+'\n')
    (export/'dataset_manifest.json').write_text(json.dumps({'accepted':1,
        'train_hashes':{'jsonl':sha(export/'train.jsonl')}, 'human_review_completed':0,
        'training_ready':False,'status':'fixture'}))
    return repo, export, record


def test_snapshot_roundtrip_and_false_readiness(tmp_path):
    repo, export, record = source(tmp_path)
    out = tmp_path/'snapshot'
    result = freeze(export, repo, out)
    assert result['training_ready'] is False
    with (out/'interim_motor_training.csv').open() as f:
        rows = [{k:json.loads(v) for k,v in r.items()} for r in csv.DictReader(f)]
    assert rows == [record]
    with (out/'interim_motor_training_display.csv').open() as f:
        assert next(csv.DictReader(f))['question'].startswith("'=")
    assert not ((out/'interim_motor_training.jsonl').stat().st_mode & 0o222)
    out.chmod(0o700)  # allow fixture cleanup
    for p in out.iterdir():p.chmod(0o600)


@pytest.mark.parametrize('failure',['tampered_export','unresolved_citation','unpassed_screen','reserved_seed'])
def test_snapshot_rejects_ineligible_records(tmp_path, failure):
    repo, export, record = source(tmp_path)
    if failure=='unresolved_citation':record['unresolved_citations']=['missing']
    elif failure=='unpassed_screen':record['provenance']['production_validation']['all_protected_semantic_screen_passed']=False
    elif failure=='reserved_seed':record['split']='seed'
    (export/'train.jsonl').write_text(json.dumps(record)+'\n'+' ')
    if failure!='tampered_export':
        p=export/'dataset_manifest.json';m=json.loads(p.read_text());m['train_hashes']['jsonl']=sha(export/'train.jsonl');p.write_text(json.dumps(m))
    with pytest.raises(ValueError):freeze(export,repo,tmp_path/'rejected')
