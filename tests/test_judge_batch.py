import pytest
from experiments import core
from scripts.prepare_judge_batch import prepare


def source(path,row):
    path.mkdir()
    core.write_rows(path/'judge_tasks.jsonl',[{'task':'t0','question':'synthetic question',
        'gold_answer':'synthetic answer','key_facts':['fact'],'model_answer':'fact',
        'qid':'SHOULD_NOT_LEAK','model':'SHOULD_NOT_LEAK','condition':'SHOULD_NOT_LEAK'}])
    core.write_rows(path/'judge_map.jsonl',[{'task':'t0','qid':'private-q','row':row}])
    return path


def test_blinded_batch_uses_unique_codes_and_explicit_field_allowlist(tmp_path):
    a=source(tmp_path/'a','model-a');b=source(tmp_path/'b','model-b')
    prepare([a,b],tmp_path/'out')
    tasks=core.rows(tmp_path/'out/tasks.jsonl')
    assert len({t['task'] for t in tasks})==2
    assert all(set(t)=={'task','question','gold_answer','key_facts','model_answer','must_not_assert'} for t in tasks)
    assert 'SHOULD_NOT_LEAK' not in (tmp_path/'out/tasks.jsonl').read_text()
    assert len(core.rows(tmp_path/'out/mapping.jsonl'))==2


def test_duplicate_answer_cannot_enter_blinded_batch(tmp_path):
    a=source(tmp_path/'a','model-a');b=source(tmp_path/'b','model-a')
    with pytest.raises(ValueError,match='Duplicate answer'):
        prepare([a,b],tmp_path/'out')
