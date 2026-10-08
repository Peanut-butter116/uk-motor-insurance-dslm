import json
import pytest
from experiments import core
from scripts.error_analysis import aggregate, paired, evidence_presence, validate_answers, make_records
from scripts.faithfulness_audit import select_qids
from scripts.kaggle_followup import verify_training, retarget
from scripts.merge_judgements import merge


def record(q='a',corr=None,abst=False):
    return dict(qid=q,line='motor',answerable=True,abst=abst,n_sup=1,n_tot=2,
        corr_item=corr,judged=corr is not None,parse_ok=True,resolved_format_ok=True,
        evidence={'status':'all_gold_quotes_located'},categories=[],faithfulness='not_assessed')


def test_partial_correctness_cannot_be_published():
    m=aggregate([record('a',1.0),record('b')])
    assert 'composite' not in m and 'correctness' not in m
    assert m['missing_correctness']==1
    assert m['evidence_decomposition']['all_gold_quotes_located']['correctness'] is None
    assert m['categories']['CONTRADICTION'] is None


def test_complete_scores_reuse_historical_composite():
    m=aggregate([record('a',1.0)])
    assert m['composite']==pytest.approx(.5+.3*.5+.2)


def test_missing_paired_judgement_blocks_bootstrap():
    m,_=paired([record()],[record(corr=1)])
    assert m['paired_bootstrap']['status']=='awaiting_judging'
    assert m['correctness']['pending']==1


def test_paired_bootstrap_has_correct_direction_and_reproducible_seed():
    a=[record('a',0),record('b',.5)];b=[record('a',.5),record('b',1)]
    m,_=paired(a,b,iterations=30)
    assert m['correctness']['counts']=={'improves':2}
    assert m['paired_bootstrap']['metrics']['correctness']['ci95']==[.5,.5]
    assert m==paired(a,b,iterations=30)[0]


def test_mismatched_pairs_fail():
    with pytest.raises(ValueError):paired([record('a',1)],[record('b',1)])


def test_empty_analysis_fails():
    with pytest.raises(ValueError):aggregate([])


def test_evidence_requires_same_document_and_quote():
    g={'gold_citations':[{'doc_file':'a.pdf','insurer':'Test','quote':'synthetic support'}]}
    chunk={'text':'synthetic support','meta':{'filename':'b.pdf','insurer':'Test'}}
    assert evidence_presence(g,[chunk])['status']=='no_gold_quotes_located'
    chunk['meta']['filename']='a.pdf'
    assert evidence_presence(g,[chunk])['status']=='all_gold_quotes_located'
    assert evidence_presence({'gold_citations':[]},[chunk])['status']=='unassessable'


@pytest.mark.parametrize('data', [[],[{'qid':'a','row':'x','answer':''}],
    [{'qid':'a','row':'x','answer':'ok'},{'qid':'a','row':'x','answer':'ok'}]])
def test_bad_answer_population_fails(data):
    with pytest.raises(ValueError):validate_answers(data,{'a':{}},'x')


def test_two_vote_new_score_is_rejected():
    g={'a':dict(answerable=True,key_facts=['synthetic'],line='motor',gold_citations=[])}
    with pytest.raises(ValueError,match='three valid'):
        make_records([{'qid':'a','row':'x','answer':'synthetic'}],g,{'a':{'chunks':[]}},
            [{'qid':'a','row':'x','fact_scores':[2],'contradiction':False,'n_votes':2,'n_votes_cast':3}], 'x')


def test_faithfulness_sample_is_fixed_and_paired():
    gold={str(i):{} for i in range(60)}
    answers={q:{'answer':'Synthetic factual answer.'} for q in gold}
    pool,sample=select_qids(gold,answers,answers)
    assert len(pool)==60 and len(sample)==45 and len(set(sample))==45
    assert sample==select_qids(gold,answers,answers)[1]


def test_failed_gpu_output_never_unlocks_training(tmp_path):
    (tmp_path/'FAILED.txt').write_text('failed')
    with pytest.raises(ValueError,match='wrapper failed'):
        verify_training(tmp_path,tmp_path,tmp_path,'smoke')


def test_wrong_nonce_never_unlocks_training(tmp_path):
    core.write_json(tmp_path/'SUCCESS.json',{'kind':'smoke','nonce':'wrong'})
    core.write_json(tmp_path/'launch.json',{'nonce':'expected'})
    with pytest.raises(ValueError,match='stale'):
        verify_training(tmp_path,tmp_path,tmp_path,'smoke')


def test_one_vote_cannot_replace_three(tmp_path):
    with pytest.raises(ValueError,match='three separate'):
        merge('unused','unused',['vote'],tmp_path/'out')


def test_three_votes_merge_with_verbatim_evidence(tmp_path):
    tasks=tmp_path/'tasks.jsonl';mapping=tmp_path/'map.jsonl'
    core.write_rows(tasks,[{'task':'a','key_facts':['fact'],'model_answer':'fact'}])
    core.write_rows(mapping,[{'task':'a','qid':'q','row':'model'}])
    files=[]
    for i,v in enumerate([0,2,2]):
        f=tmp_path/f'v{i}.jsonl';files.append(f)
        core.write_rows(f,[{'task':'a','fact_scores':[v],'evidence_spans':['fact'],'contradiction':False}])
    merge(tasks,mapping,files,tmp_path/'out')
    row=core.rows(tmp_path/'out/scores.jsonl')[0]
    assert row['n_votes']==3 and row['fact_scores']==[2]


def test_missing_vote_population_fails(tmp_path):
    tasks=tmp_path/'tasks';mapping=tmp_path/'map'
    core.write_rows(tasks,[{'task':'a','key_facts':['fact'],'model_answer':'fact'}])
    core.write_rows(mapping,[{'task':'a','qid':'q','row':'model'}])
    files=[]
    for i in range(3):
        f=tmp_path/str(i);files.append(f);core.write_rows(f,[])
    with pytest.raises(ValueError):merge(tasks,mapping,files,tmp_path/'out')
