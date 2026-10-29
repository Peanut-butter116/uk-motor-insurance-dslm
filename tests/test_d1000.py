import hashlib
import json
from pathlib import Path
from experiments.prepare_milestone_sft import messages,grouped_split


def test_targets_and_context_preserved():
    row={'question':'Is this covered?','context':'[Acorn, motor.pdf, p.2]\nQuoted wording',
         'answer':'The evidence is insufficient to decide.'}
    m=messages(row)
    assert m[-1]['content']==row['answer']
    assert row['context'] in m[0]['content']
    assert 'reply with exactly this sentence' not in m[0]['content']
    assert 'exact bracketed identifiers' in m[0]['content']


def test_grouped_split_reproducible_and_disjoint():
    rows=[{'example_id':str(i),'question':'Same question' if i<2 else str(i)} for i in range(50)]
    a,b=grouped_split(rows);assert (a,b)==grouped_split(rows)
    assert not set(a)&set(b) and set(a)|set(b)=={str(i) for i in range(50)}
    assert ('0' in a)==('1' in a)


def test_historical_loader_unchanged():
    expected={'experiments/prepare_sft.py':'604cd50693d174ddfc80f6f50d5acadc17b36314bcee27ddd654a567a72fc4fe',
              'experiments/tokenise.py':'ba035a86d228ce00a207cc76e89d03748981e63481166c58a5be35dfc716a8ea',
              'experiments/train.py':'a7601e43896f543f1ff70f802eb28b28985650073c3bd3b718569203d7e7df9a'}
    for name,h in expected.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h


def test_exact_models_preserved():
    cfg=json.loads(Path('cloud/d1000/config.json').read_text())
    old=json.loads(Path('configs/insurance_3b.json').read_text())
    assert cfg['models']['qwen3b']==old['models']['qwen3b']
    assert cfg['models']['ministral3b']==old['models']['ministral3b']
    assert len(cfg['models'])==4


def test_agreement_missing_and_degenerate():
    from cloud.d1000.metrics import agreement
    assert agreement([None],[1])['paired']==0
    assert agreement([1,1],[1,1])['kappa'] is None
    a=agreement([0,0,1,1],[0,1,1,1])
    assert a['agreement']==.75 and a['kappa']==.5 and a['disagreements']==1


def test_judge_extension_does_not_replace_frozen_rubric():
    from cloud.d1000.judging import schema,normalise
    from eval.semantic_fixtures import fixtures
    from eval import judge_span_ids as indexed
    task,_=fixtures()[0]
    s=schema(task)['properties']['evaluations']['items']['properties']
    assert 'correctness' in s and 'reference_span_ids' in s and 'citation_assessments' in s
    assert 'correctness' in indexed.schema([task])['properties']['evaluations']['items']['properties']
