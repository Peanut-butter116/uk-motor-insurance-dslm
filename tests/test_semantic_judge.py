"""Synthetic-only privacy, reliability and semantic evaluation regression tests."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import urllib.error
import pytest
from eval import semantic_judge as sj
from eval.semantic_fixtures import fixtures
from eval.gemini_judge import GeminiJudge, QuotaError
from scripts.run_semantic_judge import secure


def task_vote(index=0):
    t,expected=copy.deepcopy(fixtures()[index])
    abst=t['deterministic_abstention']
    category=('incorrect_abstention' if t['gold_answerable'] else 'correct_abstention') if abst else ('correct_answer' if expected else 'incorrect_answer')
    v={'evaluation_id':t['evaluation_id'],'correctness':expected,
       'evidence_support':None if abst else (1 if expected==1 else 0),
       'materially_incorrect':expected==0 and not abst,'unsupported_material_claim':expected==0 and not abst,
       'abstention_category':category,'fact_scores':[2 if expected==1 else 0]*len(t['key_facts']),
       'evidence_spans':[t['model_answer'] if expected==1 else '']*len(t['key_facts']),
       'contradiction':expected==0 and not abst,'material_conditions':['£150 excess'],
       'contradictions':[],'reference_spans':[] if abst else [t['reference_evidence'][0]['text']],
       'reason':'Fictional policy comparison.','needs_human_review':False,'judge_confidence':'high','uncertainty_flags':[]}
    return t,v


def test_blinding_and_duplicate_prevention():
    t,_=task_vote();r={**t,'qid':'secret-qid','model':'secret-model','condition':'secret-sft','revision':'secret-revision','scores':.9}
    tasks,mapping=sj.blind([r],4)
    assert not {'model','condition','revision','scores','qid'} & tasks[0].keys()
    assert 'secret-model' not in json.dumps(tasks)
    assert mapping[0]['model']=='secret-model'
    assert sj.blind([r],4)==sj.blind([r],4)
    with pytest.raises(ValueError):sj.blind([r,r],4)
    with pytest.raises(ValueError,match='self-identification'):sj.blind([{**r,'model_answer':'I am Qwen2.5.'}],4)


@pytest.mark.parametrize('bad',['```json\n{}\n```','{"a":1,"a":2}','{"a":','[]','NaN'])
def test_strict_json(bad):
    with pytest.raises((ValueError,TypeError)):sj.validate(bad,task_vote()[0])


def test_valid_vote_and_material_error():
    t,v=task_vote();assert sj.validate(json.dumps(v),t)==v
    v['contradiction']=True
    with pytest.raises(ValueError,match='zero'):sj.validate(json.dumps(v),t)


def test_refusal_rules():
    for i in (3,4):
        t,v=task_vote(i);assert sj.validate(json.dumps(v),t)['correctness']==(0 if i==3 else 1)
    t,v=task_vote(3);v['correctness']=1
    with pytest.raises(ValueError):sj.validate(json.dumps(v),t)


def test_missing_evidence_and_uncertainty():
    t,v=task_vote();t['reference_complete']=False
    with pytest.raises(ValueError):sj.validate(json.dumps(v),t)
    v.update(correctness=None,evidence_support=None,needs_human_review=True,uncertainty_flags=['missing_reference'])
    assert sj.validate(json.dumps(v),t)['correctness'] is None


def test_answer_and_reference_spans():
    t,v=task_vote();v['reference_spans']=['Invented evidence']
    with pytest.raises(ValueError):sj.validate(json.dumps(v),t)
    t,v=task_vote();v['evidence_spans']=['Invented answer']
    with pytest.raises(ValueError):sj.validate(json.dumps(v),t)


def test_three_votes_disagreement_null_and_duplicates():
    t,v=task_vote();votes=[{**v,'vote':i} for i in (1,2,3)]
    votes[1]={**votes[1],'correctness':.5}
    result=sj.merge(votes,t)
    assert result['correctness']==1 and result['disagreement']
    assert sj.merge(votes[:2],t) is None
    assert sj.merge(votes[:1],t,required=1)['protocol']=='preliminary_single_vote'
    votes[1]['correctness']=None;assert sj.merge(votes,t)['correctness'] is None
    with pytest.raises(ValueError):sj.merge([votes[0],votes[0],votes[2]],t)


def test_bootstrap_paired_missing_and_groups():
    result=sj.paired_bootstrap([('a',0,1),('a',.5,1),('b',None,0)],iterations=100)
    assert result['n_pairs']==2 and result['n_groups']==1 and result['difference']==.75
    assert result['ci_low'] is None and result['interval_status']=='insufficient_independent_groups'
    assert sj.paired_bootstrap([('b',None,0)])['difference'] is None


def test_permission_questions_only_not_semantic():
    r={'authorised':True,'provider':'gemini','questions_only':True}
    assert not sj.permission_allows(r,'gemini',['model_answer'],'motor_semantic_evaluation')
    r.update(questions_only=False,purpose='motor_semantic_evaluation',allowed_fields=['model_answer'])
    assert sj.permission_allows(r,'gemini',['model_answer'],'motor_semantic_evaluation')
    assert not sj.permission_allows(r,'gemini',['reference_answer'],'motor_semantic_evaluation')


def test_quota_and_unknown_free_plan():
    r={'free_plan_confirmed':True,'rpm':5,'rpd':500,'tpm':250000}
    assert sj.quota_available(r,[],100,100)
    assert not sj.quota_available(r,[(99,10)]*5,100,100)
    assert not sj.quota_available(r,[(99,249999)],100,2)
    with pytest.raises(PermissionError):sj.quota_available({},[],100,10)


def test_shared_research_quota_deduplicates_adopted_journals(tmp_path):
    from eval.gemini_judge import research_events
    a=tmp_path/'old';b=tmp_path/'new'
    reservation={'time':99,'reserved_tokens':300}
    a.write_text(json.dumps(reservation)+'\n')
    b.write_text(json.dumps(reservation)+'\n'+json.dumps({'request_time':99,'actual_tokens':200,'quota_actual_tokens':100})+'\n'+json.dumps({'time':100,'reserved_tokens':400})+'\n')
    assert sorted(research_events([a,b,b]))==[(99,100),(100,400)]


def client(tmp_path,fictional=False):
    shared=tmp_path/'shared.json';shared.write_text('{"entries":[],"blocked_until":0}')
    free={'model':'gemini-test-flash','free_plan_confirmed':True,'verified_at_epoch':100,'rpm':5,'rpd':500,'tpm':250000}
    return GeminiJudge('fictional-key','gemini-test-flash',{},free,tmp_path/'ledger',shared,
        [{'name':'models/gemini-test-flash','supportedGenerationMethods':['generateContent']}],fictional=fictional,clock=lambda:100)


def test_remote_privacy_gate_before_network(tmp_path):
    c=client(tmp_path)
    with patch('urllib.request.build_opener') as network:
        with pytest.raises(PermissionError):c.call(task_vote()[0],1)
        network.assert_not_called()
    c=client(tmp_path,fictional=True);t=task_vote()[0];t['question']='Arbitrary protected question'
    with pytest.raises(PermissionError):c.call(t,1)


def test_429_stops_without_retry(tmp_path):
    c=client(tmp_path,fictional=True)
    with patch('urllib.request.build_opener') as network:
        network.return_value.open.side_effect=urllib.error.HTTPError('https://example.invalid',429,'quota',{},None)
        with pytest.raises(QuotaError):c.call(fixtures()[0][0],1)
        assert network.return_value.open.call_count==1
    assert len((tmp_path/'ledger').read_text().splitlines())==1


def test_private_output_firewall():
    with pytest.raises(ValueError):secure(Path(__file__).resolve().parents[1]/'results')


def test_mixed_judges_and_invalid_vote_index():
    t,v=task_vote()
    with pytest.raises(ValueError):sj.merge([{**v,'vote':0}],t,required=1)
    with pytest.raises(ValueError):sj.merge([{**v,'vote':i,'provider':str(i)} for i in (1,2,3)],t)


def test_provider_schema_enforces_canonical_fact_arity():
    t,_=task_vote();schema=sj.response_schema(t)
    assert set(schema['required'])==sj.FIELDS
    assert schema['properties']['fact_scores']['minItems']==len(t['key_facts'])
    assert schema['properties']['evidence_spans']['maxItems']==len(t['key_facts'])
    assert schema['additionalProperties'] is False


def test_unsupported_absence_does_not_need_fabricated_quote():
    t,v=task_vote(5);v['reference_spans']=[]
    assert sj.validate(json.dumps(v),t)['evidence_support']==0
    v['unsupported_material_claim']=False;v['contradiction']=False
    with pytest.raises(ValueError):sj.validate(json.dumps(v),t)


def test_batch_pack_preserves_population_and_independent_questions():
    from scripts.run_gemini_batch_judge import pack
    tasks=[];mapping={}
    for q in range(4):
        for arm in range(8):
            t=copy.deepcopy(fixtures()[0][0]);t['evaluation_id']=f'q{q}-a{arm}';tasks.append(t)
            mapping[t['evaluation_id']]={'question_group':str(q)}
    batches=list(pack(tasks,mapping,42,size=10))
    assert len([t for batch in batches for t in batch])==32
    assert all(len({mapping[t['evaluation_id']]['question_group'] for t in batch})==len(batch) for batch in batches)
    assert list(pack(tasks,mapping,42,size=10))==batches


def test_indexed_transport_lossless_and_exact_resolution():
    from eval import judge_span_ids as ids
    text='First sentence.  Second sentence.\nThird line\n'
    spans=ids.segments(text,'A')
    assert ''.join(s['text'] for s in spans)==text
    assert ids.resolve('A0-A2',[spans])==text
    with pytest.raises(ValueError):ids.resolve('A2-A0',[spans])
    with pytest.raises(ValueError):ids.resolve('R0S0-R1S0',[[{'id':'R0S0','text':'one'}],[{'id':'R1S0','text':'two'}]])
    with pytest.raises(ValueError):ids.resolve('Invented',[spans])


def test_indexed_normalisation_preserves_legacy_scores():
    from eval import judge_span_ids as ids
    t,v=task_vote();transport={k:x for k,x in v.items() if k not in ('evidence_spans','reference_spans')}
    transport['answer_span_ids']=['A0'];transport['reference_span_ids']=['R0S0-R0S3']
    result=ids.normalise(transport,t)
    assert result['fact_scores']==v['fact_scores'] and result['correctness']==1
    assert result['evidence_spans']==v['evidence_spans']
    transport['answer_span_ids']=['A999']
    with pytest.raises(ValueError):ids.normalise(transport,t)


def test_unanswerable_reference_scope_and_missing_answerable_reference():
    from scripts.run_semantic_judge import reference_package
    context=[{'text':'Fictional Acorn policy wording.','meta':{'filename':'fictional.txt','section_id':'1'}}]
    gold={'answerable':False,'gold_citations':[],'review_status':'council_verified'}
    refs,complete=reference_package(gold,[],context)
    assert complete and refs[0]['text']==context[0]['text']
    assert 'not a positive gold citation' in refs[0]['scope']
    assert reference_package({**gold,'answerable':True},[],context)==([],False)
    assert reference_package({**gold,'review_status':'draft'},[],context)[1] is False


def test_adoption_requires_identical_input_and_identity_mapping():
    from scripts.adopt_unchanged_judge_votes import unchanged_ids
    old={'e0':{'model_answer':'Fictional answer'}};meta={'e0':{'model':'student-a','condition':'base'}}
    assert unchanged_ids(old,copy.deepcopy(old),meta,copy.deepcopy(meta))=={'e0'}
    assert not unchanged_ids(old,old,meta,{'e0':{'model':'student-b','condition':'base'}})
    assert not unchanged_ids(old,{'e0':{'model_answer':'Changed answer'}},meta,meta)


def test_explicit_judge_plan_does_not_allow_unregistered_models():
    t,v=task_vote();votes=[{**v,'vote':i,'provider':'gemini','model':'judge-a' if i!=3 else 'judge-b'} for i in (1,2,3)]
    plan={'1':'judge-a','2':'judge-a','3':'judge-b'}
    assert sj.merge(votes,t,judge_plan=plan)['n_votes']==3
    with pytest.raises(ValueError):sj.merge(votes,t,judge_plan={**plan,'3':'other-model'})


def test_worker_lock_prevents_duplicate_dispatch(tmp_path):
    import fcntl
    from scripts.run_gemini_batch_judge import execute
    with (tmp_path/'.judge_worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='already owns'):execute(directory=str(tmp_path))


def test_no_semantic_figures_from_pending_scores(tmp_path):
    from scripts.plot_semantic_results import make
    source=tmp_path/'results';source.mkdir();out=tmp_path/'figures'
    (source/'correctness_summary.csv').write_text('mean_correctness,judged_n,population_n\n,0,76\n')
    for name in ['grounding_summary.csv','sft_rag_comparison.csv']:(source/name).write_text('model\n')
    (source/'judge_validation_summary.json').write_text('{}')
    make(source,out)
    assert not list(out.glob('*.png')) and not list(out.glob('*.svg'))
    assert 'zero real responses' in (out/'README.md').read_text()


def test_aggregate_exact_denominators_and_pending(tmp_path):
    from scripts.run_semantic_judge import aggregate
    directory=tmp_path/'private';directory.mkdir();public=tmp_path/'public'
    tasks=[task_vote(0)[0],task_vote(4)[0]]
    def rows(name,items):(directory/name).write_text(''.join(json.dumps(r)+'\n' for r in items))
    rows('judge_inputs.jsonl',tasks)
    rows('private_mapping.jsonl',[{'evaluation_id':t['evaluation_id'],'qid':str(i),'model':'qwen3b',
        'experiment':'historical_sft21','model_id':'fictional/student','revision':'fixture-v1','condition':'qwen3b-base-closedbook','question_group':str(i),
        'parse_success':True,'latency_seconds':1,'citation_count':None,'resolved_citation_count':None} for i,t in enumerate(tasks)])
    # One response complete, the other missing: never count missing as incorrect.
    v=task_vote(0)[1];rows('judge_votes.jsonl',[{**v,'vote':i} for i in (1,2,3)])
    rows('judge_attempts.jsonl',[])
    (directory/'judge_manifest.json').write_text('{}')
    (directory/'human_review_sample.csv').write_text('evaluation_id,human_correctness,human_reason\n')
    aggregate(directory,public)
    import csv
    result=list(csv.DictReader((public/'correctness_summary.csv').open()))[0]
    assert result['population_n']=='2' and result['judged_n']=='1'
    assert float(result['mean_correctness'])==1 and result['pending_or_uncertain_n']=='1'
    single=list(csv.DictReader((public/'preliminary_single_vote_summary.csv').open()))[0]
    assert single['correctness_scored_n']=='1' and single['population_n']=='2'
    assert single['protocol']=='preliminary_vote_1' and single['revision']=='fixture-v1'
    validation=json.loads((public/'judge_validation_summary.json').read_text())
    assert validation['valid_votes']==3 and validation['human_review_completed']==0
