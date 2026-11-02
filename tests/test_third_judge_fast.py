import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from cloud.third_judge_fast.protocol import select, eligible, calibration_pass, MODEL
from cloud.third_judge_fast.engine import adapter, preserved, original_manifest


def test_historical_votes_take_precedence_and_unknowns_are_never_reassigned():
    votes={'groq':{'a':{'v':1}},'gemini':{'a':{'v':2},'b':{'v':3}},'gemini31':{'a':{'v':4},'c':{'v':5}}}
    assert select(votes)['a']==('groq',{'v':1})
    candidates=[({'evaluation_id':i},{}) for i in 'abcdef']
    assert [t['evaluation_id'] for t in eligible(candidates,votes,{'d','e'})]==['f']


def test_new_adapter_uses_existing_shared_quota_ledger_and_exact_identity(tmp_path):
    called=[]
    p=SimpleNamespace(Quotas=lambda *args:called.append(args))
    proof={'free_tier':True,'catalogue_verified':True,'model':MODEL,
           'limits':{'rpm':15,'rpd':500,'tpm':250000,'tpd':125000000}}
    a=adapter(p,tmp_path/'new',tmp_path/'old',proof)
    limits=a.verify_receipt(tmp_path/'new','gemini',MODEL)
    a.Quotas(tmp_path/'new/provider_usage/gemini.json','gemini',limits)
    assert called[0][0]==tmp_path/'old/provider_usage/gemini.json'
    with pytest.raises(RuntimeError):a.verify_receipt(tmp_path/'new','gemini','other')
    proof['free_tier']=False
    with pytest.raises(RuntimeError):a.verify_receipt(tmp_path/'new','gemini',MODEL)


def test_original_calibration_requires_null_refusal_support_and_exact_label():
    t={'deterministic_abstention':True,'gold_answerable':False}
    a={'correctness':1,'evidence_support':None,'abstention_category':'correct_abstention'}
    assert calibration_pass(t,1,a)
    assert not calibration_pass(t,1,{**a,'evidence_support':1})
    assert not calibration_pass(t,0,a)


def test_historical_cache_and_journal_modification_blocks(tmp_path):
    p=tmp_path/'judging/gemini.jsonl';p.parent.mkdir();p.write_text('original')
    manifest=original_manifest(tmp_path);preserved(tmp_path,manifest)
    p.write_text('changed')
    with pytest.raises(RuntimeError):preserved(tmp_path,manifest)


def test_failure_sealing_never_calls_provider_and_cannot_enable_research(tmp_path,monkeypatch):
    import asyncio
    from cloud.third_judge_fast import engine
    from cloud.d1000 import judge_recovery as jr
    folder=tmp_path/'third';folder.mkdir()
    runtime=folder/'runtime';runtime.mkdir()
    called=[]
    async def snapshot(recovery,sink,event):called.append(event)
    monkeypatch.setattr(engine,'checkpoint',snapshot)
    def forbidden(*args,**kwargs):raise AssertionError('No model calls allowed while sealing')
    monkeypatch.setattr(jr,'complete',forbidden)
    recovery=SimpleNamespace(third={'folder':folder,'runtime':runtime,'state':{},'protected':{'protected-a'},'active':0})
    outcome=asyncio.run(engine.seal_calibration_failure(recovery,lambda x:None))
    assert outcome=={'calibration':'FAIL','cases_passed':0,'research_calls':0}
    assert json.loads((folder/'calibration.json').read_text())['passed'] is False
    assert recovery.third['state']['phase']=='calibration_failed'
    assert not recovery.third.get('calibration_verified',False)
    assert called==['third_judge_calibration_failed_readback']
    with pytest.raises(RuntimeError):asyncio.run(engine.run(recovery,lambda x:None))
