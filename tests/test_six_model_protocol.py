import hashlib
import json

import pytest

from experiments.six_model_benchmark import validate_population, normalise_answer
from scripts.export_supervisor_csv import safe


def population():
    cfg={'models':[{'id':'fictional/model','revision':'pinned'}]}
    tasks=''.join(json.dumps({'qid':'q','mode':mode,'messages':[]})+'\n' for mode in ['closedbook','openbook']).encode()
    manifest={'purpose':'frozen_motor_test_descriptive','split':'test','questions':1,
        'tasks_sha256':hashlib.sha256(tasks).hexdigest(),'selection_permitted':False,
        'protocol_frozen_before_submission':True,
        'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()}
    return cfg,manifest,tasks


def test_protected_descriptive_population_is_separate_from_selection():
    cfg,m,tasks=population()
    assert len(validate_population(cfg,m,tasks))==2
    m['selection_permitted']=True
    with pytest.raises(ValueError,match='descriptive-only'):validate_population(cfg,m,tasks)


def test_changed_protocol_or_population_is_rejected():
    cfg,m,tasks=population();cfg['models'][0]['revision']='different'
    with pytest.raises(ValueError):validate_population(cfg,m,tasks)
    cfg,m,tasks=population()
    with pytest.raises(ValueError,match='hash'):validate_population(cfg,m,tasks+b' ')


def test_unreviewed_dev_is_not_eligible_for_selection():
    cfg,m,tasks=population();m.update(purpose='motor_dev',split='dev',independent_review_complete=False)
    with pytest.raises(ValueError,match='Reviewed'):validate_population(cfg,m,tasks)


def test_missing_rag_pair_is_rejected():
    cfg,m,tasks=population();tasks=tasks.splitlines()[0]+b'\n';m['tasks_sha256']=hashlib.sha256(tasks).hexdigest()
    with pytest.raises(ValueError,match='matched'):validate_population(cfg,m,tasks)


def test_reasoning_is_not_scored_as_a_final_answer():
    assert normalise_answer('<think>unfinished')['answer']==''
    assert normalise_answer('reasoning without opening tag',True)['reasoning_truncated']
    assert normalise_answer('reasoning</think> Final answer.',True)['answer']=='Final answer.'
    assert normalise_answer('Plain answer.')['answer']=='Plain answer.'


@pytest.mark.parametrize('value',['=1+1',' +SUM(A1:A2)','\t@danger','\r=1','-2'])
def test_readable_csv_neutralises_formula_prefixes(value):
    assert safe(value).startswith("'")
