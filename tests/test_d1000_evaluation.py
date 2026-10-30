import json
from pathlib import Path
import pytest
from cloud.d1000.evaluation_only import verify_unchanged
from cloud.d1000.judging import normalise
from eval.semantic_fixtures import fixtures


def refusal(task,correctness):
    return dict(evaluation_id=task['evaluation_id'],correctness=correctness,
        evidence_support=None,materially_incorrect=False,unsupported_material_claim=False,
        abstention_category='incorrect_abstention' if task['gold_answerable'] else 'correct_abstention',
        fact_scores=[0]*len(task['key_facts']),answer_span_ids=[None]*len(task['key_facts']),
        contradiction=False,material_conditions=[],contradictions=[],reference_span_ids=[],
        reason='The answer refuses to make a policy claim.',needs_human_review=False,
        judge_confidence='high',uncertainty_flags=[],citation_assessments=[])


def test_correct_and_incorrect_refusal_keep_null_support():
    assert len(fixtures())==9
    for i,score in [(3,0),(4,1)]:
        task,label=fixtures()[i];assert label==score
        task=dict(task,citations=[],student_evidence_available=False)
        v=refusal(task,score)
        assert normalise(v,task)['evidence_support'] is None
        v['evidence_support']=1
        with pytest.raises(ValueError,match='Refusal support must be N/A'):
            normalise(v,task)


def test_incomplete_and_extra_judgement_fields_rejected():
    task,_=fixtures()[4];v=refusal(task,1)
    del v['answer_span_ids']
    with pytest.raises(ValueError,match='Unexpected indexed schema'):normalise(v,task)
    v=refusal(task,1);v['invented']='field'
    with pytest.raises(ValueError,match='Unexpected indexed schema'):normalise(v,task)


def test_preservation_guard_detects_changed_or_added_responses(tmp_path):
    from cloud.d1000.export import sha
    p=tmp_path/'runs/model/base-responses.jsonl';p.parent.mkdir(parents=True);p.write_text('original\n')
    (tmp_path/'evaluation_preservation.json').write_text(json.dumps({str(p.relative_to(tmp_path)):sha(p)}))
    verify_unchanged(tmp_path)
    p.write_text('regenerated\n')
    with pytest.raises(AssertionError,match='Immutable'):verify_unchanged(tmp_path)


def test_cpu_diagnostics_staging_has_no_training_dispatch(tmp_path):
    from cloud.d1000.stage import stage
    stage(tmp_path,mode='evaluation-diagnostics')
    meta=json.loads((tmp_path/'kernel-metadata.json').read_text())
    nb=json.loads((tmp_path/'campaign.ipynb').read_text());code=''.join(nb['cells'][1]['source'])
    assert meta['enable_gpu'] is False and meta['machine_shape']=='None'
    assert '\nfrom cloud.d1000.evaluation_only import diagnostics as main\n' in code
    assert '\nfrom cloud.d1000.campaign import main\n' not in code


def test_schema_transport_retries_known_invalid_but_cache_avoids_replay(tmp_path):
    import types,jsonschema
    from cloud.d1000.judge_recovery import complete,body
    task,_=fixtures()[4];calls=[]
    invalid=refusal(task,1);invalid['evidence_support']=1
    valid=refusal(task,1)
    class Quotas:
        def __init__(self,*args):pass
        def reserve(self,*args):return len(calls)
        def reconcile(self,*args):pass
    def transport(values,provider,actual):
        calls.append(actual)
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'evaluations':[invalid if len(calls)==1 else valid]})}}],'usage':{}},{}
    fake=types.SimpleNamespace(verify_receipt=lambda *a:{},Quotas=Quotas,
        atomic_private=lambda p,v:p.write_text(json.dumps(v)),transport=transport,
        parse_json_content=json.loads,validate_answer=lambda v,s:(jsonschema.validate(v,s) or v),
        token_reservation=lambda *a:1,QuotaPause=type('QuotaPause',(Exception,),{}))
    r=complete({},'groq','openai/gpt-oss-120b',task,tmp_path,fake)
    assert r['answer']['evidence_support'] is None and len(calls)==2
    assert calls[0]['response_format']['type']=='json_schema'
    assert 'Refusal support must be N/A' in calls[1]['messages'][-1]['content']
    complete({},'groq','openai/gpt-oss-120b',task,tmp_path,fake)
    assert len(calls)==2
    history=next((tmp_path/'evaluation_cache_v2/groq').glob('*.attempts.json'))
    record=next(p for p in history.parent.glob('*.json') if not p.name.endswith(('.raw.json','.attempts.json')))
    record.unlink();h=json.loads(history.read_text());h[-1]['status']='transport_uncertain';history.write_text(json.dumps(h))
    (history.parent/(h[-1]['request_sha256']+'.raw.json')).unlink()
    with pytest.raises(RuntimeError,match='Unknown request outcome'):
        complete({},'groq','openai/gpt-oss-120b',task,tmp_path,fake)
    assert len(calls)==2


def test_truncated_and_malformed_responses_fail_closed():
    import types
    from cloud.d1000.judge_recovery import validate
    task,_=fixtures()[4]
    with pytest.raises(ValueError,match='incomplete'):
        validate({'choices':[{'finish_reason':'length','message':{'content':'{}'}}]},task,None)
    fake=types.SimpleNamespace(parse_json_content=json.loads)
    with pytest.raises(json.JSONDecodeError):
        validate({'choices':[{'finish_reason':'stop','message':{'content':'{"broken"'}}]},task,fake)
