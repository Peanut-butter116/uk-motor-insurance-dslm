"""Four-model preflight and handoff tests use fictional data and no network."""
import asyncio
import ast
import json
from pathlib import Path
import sys
import types
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cloud/qa'))
import four_campaign as four


def listings():
    return {'groq':{'openai/gpt-oss-120b','openai/gpt-oss-20b'},
            'gemini':{'gemini-3.5-flash-lite','gemini-3.1-flash-lite'}}


def test_four_distinct_models_preserve_originals():
    specs=four.select_specs(listings())
    assert len({s['model'] for s in specs.values()})==4
    assert specs['groq']['model']=='openai/gpt-oss-120b'
    assert specs['gemini']['model']=='gemini-3.5-flash-lite'


def test_only_documented_free_fallbacks_and_no_missing_existing():
    data=listings();data['gemini'].remove('gemini-3.1-flash-lite')
    data['gemini'].add('gemini-3.8-flash')
    assert four.select_specs(data)['gemini_extra']['model']=='gemini-3.8-flash'
    data['gemini'].remove('gemini-3.8-flash');data['gemini'].add('paid-only-pro')
    with pytest.raises(RuntimeError,match='No approved distinct free'):four.select_specs(data)
    data=listings();data['groq'].remove('openai/gpt-oss-120b')
    with pytest.raises(RuntimeError,match='Existing model missing'):four.select_specs(data)


def test_authorisation_keeps_account_limits_and_original_receipt(tmp_path):
    original={'model':'openai/gpt-oss-120b','rpm':30,'tpm':8000,'rpd':1000,'tpd':200000}
    provider=types.SimpleNamespace(verify_receipt=lambda *args:dict(original))
    prior=four.authorise_models(provider,tmp_path,four.select_specs(listings()),
                               {'catalogue_sha256':{'groq':'hash','gemini':'hash'}})
    receipt=provider.verify_receipt(tmp_path,'groq','openai/gpt-oss-20b')
    assert receipt['tpm']==8000 and receipt['shared_provider_quota']
    assert receipt['synthetic_json_probe_passed'] is False
    assert original['model']=='openai/gpt-oss-120b'
    with pytest.raises(RuntimeError,match='Unapproved'):provider.verify_receipt(tmp_path,'groq','paid')


@pytest.mark.parametrize('terminal',['COMPLETE','ERROR'])
def test_handoff_waits_without_writing_or_generating(tmp_path,monkeypatch,terminal):
    statuses=iter(['RUNNING',terminal]);calls=[]
    api=types.SimpleNamespace(kernels_status=lambda key:(calls.append(key) or types.SimpleNamespace(status=types.SimpleNamespace(name=next(statuses)))))
    monkeypatch.setattr(four,'Path',lambda p:tmp_path/Path(p).name)
    sleeps=[]
    async def sleep(seconds):sleeps.append(seconds)
    monkeypatch.setattr(four.asyncio,'sleep',sleep)
    if terminal=='ERROR':
        with pytest.raises(RuntimeError,match='did not finish cleanly'):asyncio.run(four.wait_for_predecessor(api))
    else:asyncio.run(four.wait_for_predecessor(api))
    assert sleeps==[30] and calls==[four.PREDECESSOR]*2
    receipt=json.loads((tmp_path/'four_handoff.json').read_text())
    assert receipt['new_writer_started'] is False and receipt['generation_calls']==0


def test_staged_entry_guard_and_notebook_event_loop(tmp_path):
    import stage_four
    stage_four.main(tmp_path)
    nb=json.loads((tmp_path/'resume.ipynb').read_text())
    code=''.join(nb['cells'][-1]['source'])
    assert 'from four_campaign import run\nawait run()' in code
    assert 'Read-only wait for execution14' in ''.join(nb['cells'][0]['source'])
    compile(code,'four-staged','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


@pytest.mark.parametrize('ambiguous',[False,True])
def test_known503_recovery_is_scoped_and_audited(tmp_path,ambiguous):
    run=tmp_path;part=run/'fast_unverified_v1';part.mkdir()
    state={'providers':{'groq':{},'gemini':{'status':'blocked','error':'gemini HTTP 503'}},
           'jobs':{'pkg':{'provider':'gemini','status':'assigned','started_epoch':1}}}
    if ambiguous:state['jobs']['other']=dict(state['jobs']['pkg'])
    statefile=part/'state.json';statefile.write_text(json.dumps(state))
    cache=run/'provider_cache/gemini';cache.mkdir(parents=True)
    history=[{'status':'transport_uncertain','request_sha256':'request'}]
    path=cache/'known.attempts.json';path.write_text(json.dumps(history))
    unrelated=cache/'unrelated.attempts.json';unrelated.write_text(json.dumps(history))
    bodies=[]
    provider=types.SimpleNamespace(digest=lambda body:(bodies.append(body) or 'known'))
    generation=types.SimpleNamespace(TEACHER_PROMPT='original prompt')
    packages=[{'id':key,'context':'fictional'} for key in state['jobs']]
    args=(run,generation,packages,provider,lambda *args:('generation',{}))
    if ambiguous:
        with pytest.raises(RuntimeError,match='does not match one'):four.recover_known_503(*args)
        assert json.loads(path.read_text())==history
    else:
        four.recover_known_503(*args)
        assert json.loads(path.read_text())==[]
        audit=json.loads((part/'gemini-503-recovery.json').read_text())
        assert audit['original_attempts']==history
        assert json.loads(statefile.read_text())['providers']['gemini']['status']=='quota_wait'
        assert bodies[0]['stage']=='fast_generation_unverified_v1'
        assert bodies[0]['body']['model']=='gemini-3.5-flash-lite'
    assert json.loads(unrelated.read_text())==history


def test_known503_retries_keep_extra_quota_reservations(tmp_path,monkeypatch):
    class Safe(Exception):pass
    calls=[];sleeps=[];reservations=[]
    def send(*args):
        calls.append(True)
        if len(calls)<3:raise Safe('groq HTTP 503')
        return {'ok':True},{}
    monkeypatch.setattr(four.fast.ReusedTransport,'__call__',send)
    monkeypatch.setattr(four.time,'sleep',lambda seconds:sleeps.append(seconds))
    transport=object.__new__(four.Retry503Transport);transport.safe_error=Safe;transport.run_path=tmp_path
    transport.providers=types.SimpleNamespace(verify_receipt=lambda *args:{},
        Quotas=lambda *args:types.SimpleNamespace(reserve=lambda *args:reservations.append(args)),
        digest=lambda body:'retry',token_reservation=lambda *args:100)
    assert transport({},'groq',{'model':'openai/gpt-oss-20b','messages':[],'max_tokens':1})[0]['ok']
    assert len(reservations)==2 and sleeps==[15,30]
