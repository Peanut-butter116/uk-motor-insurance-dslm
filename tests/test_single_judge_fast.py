from cloud.single_judge_fast import handoff
import types


def test_old_worker_detection_ignores_idle_and_other_threads(monkeypatch):
    def worker():pass
    def unrelated():pass
    threads=[types.SimpleNamespace(is_alive=lambda:True,_target=worker),
             types.SimpleNamespace(is_alive=lambda:False,_target=worker),
             types.SimpleNamespace(is_alive=lambda:True,_target=unrelated)]
    monkeypatch.setattr(handoff.threading,'enumerate',lambda:threads)
    assert handoff.old_workers()==threads[:1]


def test_groq_first_does_not_mutate_historical_votes():
    from cloud.single_judge_fast.protocol import select
    votes={'groq':{'a':{'score':1}},'gemini':{'a':{'score':0},'b':{'score':.5}}}
    chosen=select(votes)
    assert chosen['a']==('groq',{'score':1}) and chosen['b'][0]=='gemini'
    assert votes['gemini']['a']['score']==0


def test_quarantine_excludes_unknown_from_both_providers_without_mutation():
    import copy
    from cloud.single_judge_fast.protocol import eligible_tasks
    candidates=[({'evaluation_id':s},{}) for s in ('unknown','done','claimed','failed','large','safe')]
    unknown={'gemini':{'unknown'},'groq':set()};saved=copy.deepcopy(unknown)
    costs={t['evaluation_id']:100 for t,_ in candidates};costs['large']=9000
    for provider in ('groq','gemini'):
        result=eligible_tasks(candidates,{'done'},{'claimed':provider},unknown,{'failed'},costs,8000)
        assert [t['evaluation_id'] for t in result]==['safe']
    assert unknown==saved


def test_same_runtime_continuation_rejects_live_provider_before_mutation(monkeypatch,tmp_path):
    import asyncio,types,pytest
    from cloud.single_judge_fast import continuation,lifecycle
    monkeypatch.setattr(lifecycle,'provider_threads',lambda:[123])
    recovery=types.SimpleNamespace(fast_started=True,started=True,ready=True,evaluation=types.SimpleNamespace(ROOT=tmp_path))
    with pytest.raises(RuntimeError,match='not drained'):
        asyncio.run(continuation.resume_same_runtime(recovery,lambda x:None))


def test_same_runtime_resume_retains_lease_clock_and_protected_bytes(monkeypatch,tmp_path):
    import asyncio,json,types
    from cloud.single_judge_fast import continuation,lifecycle,engine
    folder=tmp_path/'single_judge_fast_v1';folder.mkdir()
    prior={'protocol':'d1000-single-judge-fast-v1','phase':'suspended','launch_epoch':1000,'deadline_epoch':73000,'baseline_unique':349}
    proof={**prior,'expected_checkpoint':294,'active_provider_threads':0,'active_provider_tasks':0,'unique_judged':469,'lease_version':1}
    (folder/'protocol.json').write_text(json.dumps(prior));(folder/'suspension.json').write_text(json.dumps(proof));(folder/'STOP_REQUESTED').touch()
    journal=tmp_path/'judge_runtime';journal.mkdir();protected=journal/'protected.json';protected.write_text('unchanged uncertain reservation')
    recovery=types.SimpleNamespace(fast_started=True,started=True,ready=True,lease_version=1,
      evaluation=types.SimpleNamespace(ROOT=tmp_path),state={'status':'single_judge_suspended_for_supported_runtime_renewal','last_persistence':{'version':294}},log=tmp_path/'private.log')
    monkeypatch.setattr(lifecycle,'provider_threads',lambda:[])
    monkeypatch.setattr(continuation.time,'time',lambda:40000)
    monkeypatch.setattr(engine,'writer_guard',lambda r:None)
    monkeypatch.setattr(engine,'prepare',lambda r:{'status':'READY_NOT_RUNNING'})
    async def coordinate(r,sink):
        assert r.fast_started and r.lease_version==1 and r.fast_resume_control==prior
        assert not (folder/'STOP_REQUESTED').exists()
        assert len(list(folder.glob('STOP_ACKNOWLEDGED-*')))==1
        assert protected.read_text()=='unchanged uncertain reservation'
        return 'resumed'
    monkeypatch.setattr(engine,'coordinate',coordinate)
    reports=[]
    assert asyncio.run(continuation.resume_same_runtime(recovery,reports.append))=='resumed'
    assert reports[0]['original_deadline_epoch']==73000
    # A second entry cannot consume the same suspension again.
    import pytest
    with pytest.raises(RuntimeError,match='marker missing'):
        asyncio.run(continuation.resume_same_runtime(recovery,reports.append))


def test_same_runtime_resume_rejects_preflight_ledger_mutation(monkeypatch,tmp_path):
    import asyncio,json,types,pytest
    from cloud.single_judge_fast import continuation,lifecycle,engine
    folder=tmp_path/'single_judge_fast_v1';folder.mkdir()
    prior={'protocol':'d1000-single-judge-fast-v1','phase':'suspended','launch_epoch':1000,'deadline_epoch':73000,'baseline_unique':349}
    proof={**prior,'expected_checkpoint':294,'active_provider_threads':0,'active_provider_tasks':0,'unique_judged':469,'lease_version':1}
    for name,value in [('protocol.json',prior),('suspension.json',proof)]: (folder/name).write_text(json.dumps(value))
    (folder/'STOP_REQUESTED').touch();runtime=tmp_path/'judge_runtime';runtime.mkdir();ledger=runtime/'ledger.json';ledger.write_text('reserved')
    recovery=types.SimpleNamespace(fast_started=True,started=True,ready=True,lease_version=1,evaluation=types.SimpleNamespace(ROOT=tmp_path),state={'status':'single_judge_suspended_for_supported_runtime_renewal','last_persistence':{'version':294}},log=tmp_path/'private.log')
    monkeypatch.setattr(lifecycle,'provider_threads',lambda:[]);monkeypatch.setattr(continuation.time,'time',lambda:40000)
    monkeypatch.setattr(engine,'writer_guard',lambda r:None)
    def mutate(r):ledger.write_text('cleared');return {}
    monkeypatch.setattr(engine,'prepare',mutate)
    with pytest.raises(RuntimeError,match='protected research state'):
        asyncio.run(continuation.resume_same_runtime(recovery,lambda x:None))
    assert (folder/'STOP_REQUESTED').exists()


def test_quota_audit_preserves_unknown_reservations_and_real_expiry(tmp_path):
    import json
    from cloud.single_judge_fast.protocol import quota_audit,feasibility
    folder=tmp_path/'provider_usage';folder.mkdir()
    state={'blocked_until':1060,'entries':[{'time':990,'quota_tokens':400,'status':'reserved'},
      {'time':-85000,'quota_tokens':200,'status':'usage_recorded','usage':{'prompt_tokens':100,'completion_tokens':20}}]}
    for p in ('groq','gemini'):(folder/(p+'.json')).write_text(json.dumps(state))
    before={p:p.read_bytes() for p in folder.iterdir()}
    class Providers:
        def verify_receipt(self,*args):return {'rpm':5,'rpd':500,'tpm':8000,'tpd':200000}
        def daily_tokens(self,e):return sum(e['usage'].values()) if 'usage' in e else e['quota_tokens']
    report=quota_audit(tmp_path,Providers(),now=1000)
    assert report['groq']['remaining_conservative_ledger']=={'rpm':4,'rpd':498,'tpm':7600,'tpd':199480}
    assert report['groq']['twenty_hour_request_slots_upper_bound']==499
    assert report['groq']['next_daily_entry_expiry']==1401
    assert report['groq']['blocked_until_epoch']==1060
    assert before=={p:p.read_bytes() for p in folder.iterdir()}
    projection=feasibility(report,563,{p:{'median_prompt_estimate':6000} for p in report})
    assert projection['quota_projection_feasible'] is False
    assert projection['request_slots_hard_upper_bound']==1498
    assert report['gemini']['twenty_hour_request_slots_upper_bound']==999


def test_unknown_task_filter_is_provider_scoped(tmp_path,monkeypatch):
    import json
    from cloud.single_judge_fast.protocol import unknown_ids
    from cloud.d1000 import judge_recovery as jr
    from experiments import core
    candidates=[({'evaluation_id':'a'},{}),({'evaluation_id':'b'}, {})]
    for p in ('groq','gemini'):(tmp_path/'judge_runtime/evaluation_cache_v2'/p).mkdir(parents=True)
    task=candidates[0][0]
    key=core.digest({'task':task,'provider':'gemini','model':'gemini-3.5-flash-lite','protocol':jr.PROTOCOL})
    history=tmp_path/'judge_runtime/evaluation_cache_v2/gemini'/(key+'.attempts.json')
    original=json.dumps([{'status':'transport_uncertain','request_sha256':'protected'}]);history.write_text(original)
    assert unknown_ids(tmp_path,candidates,'gemini')=={'a'}
    assert unknown_ids(tmp_path,candidates,'groq')==set()
    assert history.read_text()==original


def test_two_providers_claim_disjoint_work_and_persist_once(tmp_path,monkeypatch):
    import asyncio,json,threading,time,types
    from cloud.single_judge_fast import engine
    from cloud.d1000 import judging,judge_recovery as jr
    (tmp_path/'judging').mkdir()
    votes={'groq':{str(i):{} for i in range(909)},'gemini':{}}
    candidates=[({'evaluation_id':'new'+str(i)}, {}) for i in range(3)]
    costs={p:{t['evaluation_id']:100 for t,m in candidates} for p in votes}
    calls=[];inflight=set();peak=[0];lock=threading.Lock()
    def complete(values,p,model,task,*args,**kwargs):
        with lock:
            assert task['evaluation_id'] not in inflight
            inflight.add(task['evaluation_id']);peak[0]=max(peak[0],len(inflight));calls.append((p,task['evaluation_id']))
        time.sleep(.02)
        with lock:inflight.remove(task['evaluation_id'])
        return {'answer':{'evaluation_id':task['evaluation_id']},'request_sha256':'fictional','usage':{},'input_sha256':'fictional'}
    monkeypatch.setattr(jr,'complete',complete)
    monkeypatch.setattr(engine,'writer_guard',lambda r:None)
    monkeypatch.setattr(engine,'quota_audit',lambda *a,**k:{})
    monkeypatch.setattr(engine.metrics,'summarise',lambda *a:{'selected_provider_counts':{p:len(votes[p]) for p in votes}})
    providers=types.SimpleNamespace(verify_receipt=lambda *a:{'tpm':1000},QuotaPause=type('Pause',(Exception,),{}))
    monkeypatch.setattr(judging,'provider_runtime',lambda root:(providers,tmp_path))
    state={'last_persistence':{'version':249,'readback_verified':True},'artifacts_sha256':'fictional'}
    def checkpoint(s):s['last_persistence']['version']+=1
    recovery=types.SimpleNamespace(evaluation=types.SimpleNamespace(ROOT=tmp_path,checkpoint=checkpoint),state=state,lease_version=1,
        fast_prepared={'votes':votes,'candidates':candidates,'unknown':{p:set() for p in votes},'costs':costs},
        fast_archive_audit=lambda a:{'persisted_judge_counts':{p:len(votes[p]) for p in votes},'persisted_duplicate_votes':{p:0 for p in votes}})
    for k in ('GROQ_API_KEY','GEMINI_API_KEY'):monkeypatch.setenv(k,'fictional')
    reports=[];result=asyncio.run(engine.coordinate(recovery,reports.append))
    assert result['status']=='COMPLETE' and result['unique_judged']==912
    assert len(calls)==len({ident for p,ident in calls})==3
    assert peak[0]==2
    assert reports[-1]['remaining']==0 and reports[-1]['readback_verified']
    assert len((tmp_path/'single_judge_fast_v1/new_assignments.jsonl').read_text().splitlines())==3


def test_gemini_midnight_reset_preserves_all_unknown_reservations():
    import copy,datetime,types
    from cloud.single_judge_fast.quotas import install,pacific_window
    now=datetime.datetime(2026,11,2,8,0,2,tzinfo=datetime.timezone.utc).timestamp()
    start,reset=pacific_window(now)
    assert start==now-2
    assert reset-start==86400
    old=[{'time':start-30,'quota_tokens':200,'status':'reserved','request_sha256':'uncertain'}]
    state={'blocked_until':0,'entries':copy.deepcopy(old)}
    class Base:
        def __init__(self):self.provider='gemini';self.limits={'rpm':5,'rpd':1,'tpm':8000,'tpd':200000}
        def update(self,fn):return fn(state)
        def reserve(self,*args):raise AssertionError('Groq test must delegate separately')
    class Pause(Exception):pass
    module=types.SimpleNamespace(Quotas=Base,QuotaPause=Pause,SafeAPIError=ValueError,daily_tokens=lambda e:e['quota_tokens'])
    install(module);quota=module.Quotas();quota.reserve('new',100,now=now)
    assert state['entries'][0]==old[0] and len(state['entries'])==2
    try:quota.reserve('would_exceed',100,now=now)
    except Pause:pass
    else:raise AssertionError('Current calendar RPD must still block')
    assert len(state['entries'])==2


def test_pacific_reset_handles_25_hour_dst_day():
    from cloud.single_judge_fast.quotas import pacific_window
    from datetime import datetime,timezone
    now=datetime(2026,11,1,19,tzinfo=timezone.utc).timestamp()
    start,end=pacific_window(now)
    assert end-start==25*3600
    assert datetime.fromtimestamp(end,timezone.utc).isoformat()=='2026-11-02T08:00:00+00:00'


def test_async_entry_cannot_restart_launched_protocol(tmp_path):
    import asyncio,types,pytest
    from cloud.single_judge_fast.engine import run_async
    recovery=types.SimpleNamespace(fast_started=True,fast_control={})
    failure=RuntimeError('asyncio.run() cannot be called from a running event loop')
    with pytest.raises(RuntimeError,match='already launched'):asyncio.run(run_async(recovery,lambda x:None,failure,{'fictional':'hash'}))
    recovery=types.SimpleNamespace(fast_started=True)
    with pytest.raises(RuntimeError,match='already started'):asyncio.run(run_async(recovery,lambda x:None,RuntimeError('other')))
    path=tmp_path/'ledger.json';path.write_text('changed')
    with pytest.raises(RuntimeError,match='Quota state changed'):asyncio.run(run_async(recovery,lambda x:None,failure,{str(path):'different'}))


def test_resume_keeps_original_clock_and_rejects_arbitrary_interruption():
    import pytest
    from cloud.single_judge_fast.lifecycle import clock
    from cloud.single_judge_fast.resume import validate_suspension
    control={'protocol':'d1000-single-judge-fast-v1','phase':'suspended','launch_epoch':1000,'deadline_epoch':73000,'baseline_unique':349}
    proof={**control,'expected_checkpoint':300,'active_provider_threads':0,'active_provider_tasks':0,'unique_judged':850}
    state={'status':'single_judge_suspended_for_supported_runtime_renewal'}
    validate_suspension(state,control,proof,300,40000)
    assert clock(control,40000,850)==(1000,73000,349)
    with pytest.raises(RuntimeError,match='not a controlled'):validate_suspension({'status':'single_judge_partial'},control,proof,300,40000)
    proof['active_provider_threads']=1
    with pytest.raises(RuntimeError,match='not fully drained'):validate_suspension(state,control,proof,300,40000)
    with pytest.raises(RuntimeError,match='deadline changed'):clock({**control,'deadline_epoch':74000},40000,850)
    with pytest.raises(RuntimeError,match='Insufficient time'):clock(control,72500,850)


def test_suspension_waits_for_request_and_checkpoint_thread_stacks():
    from cloud.single_judge_fast.lifecycle import provider_threads
    request={'detect':provider_threads}
    exec(compile('def complete():\n    return detect()\n','cloud/d1000/judge_recovery.py','exec'),request)
    assert request['complete']()
    persistence={'detect':provider_threads}
    exec(compile('def checkpoint():\n    return detect()\n','original-audited-checkpoint','exec'),persistence)
    assert persistence['checkpoint']()
    assert not provider_threads()


def test_final_report_rejects_incomplete_checkpoint_evidence_before_reading(tmp_path):
    import pytest
    from cloud.single_judge_fast.final_report import report
    state={'last_persistence':{'version':300,'private':True,'readback_verified':True}}
    with pytest.raises(RuntimeError,match='did not finish'):report(tmp_path,{'status':'running'},state)
    with pytest.raises(RuntimeError,match='not verified'):report(tmp_path,{'status':'COMPLETE','checkpoint':299},state)


def idle_loss_fixture():
    from cloud.single_judge_fast.idle_loss import ENGINE_SHA,PROTOCOL_SHA
    control={'protocol':'d1000-single-judge-fast-v1','launch_epoch':1000,'deadline_epoch':73000,'baseline_unique':349,
             'same_lease_version':1,'source_hashes':{'engine.py':ENGINE_SHA,'protocol.py':PROTOCOL_SHA}}
    progress={'active_provider_calls':0,'launch_epoch':1000,'deadline_epoch':73000,'unique_judged':579,
              'provider_status':{'groq':{'status':'quota_wait','resume_after_epoch':80000},
              'gemini':{'status':'blocked','category':'transport_uncertain_provider_stopped'}}}
    state={'status':'single_judge_partial','last_persistence':{'version':369,'private':True,'readback_verified':True},
           'single_judge_fast':{**control,'progress':progress}}
    prior={'lease_version':1,'lease_owner_sha256':'fictional-owner-hash','launch_epoch':1000,'deadline_epoch':73000}
    return state,control,prior


def test_idle_vm_loss_requires_impossible_future_calls_and_retired_vm():
    import copy,pytest
    from cloud.single_judge_fast.idle_loss import validate
    from cloud.single_judge_fast.lifecycle import clock
    state,control,prior=idle_loss_fixture();before=copy.deepcopy((state,control,prior))
    with pytest.raises(RuntimeError,match='termination'):validate(state,control,prior,369,40000)
    result=validate(state,control,prior,369,40000,old_runtime_ended=True)
    assert result['no_post_checkpoint_requests_possible'] and result['source_checkpoint']==369
    assert (state,control,prior)==before
    assert clock({**control,'phase':'idle_loss_reconciled','idle_loss_reconciliation':result},40000,579)==(1000,73000,349)
    with pytest.raises(RuntimeError,match='evidence'):clock({**control,'phase':'idle_loss_reconciled'},40000,579)


def test_idle_vm_loss_rejects_any_possible_unsaved_request():
    import copy,pytest
    from cloud.single_judge_fast.idle_loss import validate
    original,control,prior=idle_loss_fixture()
    changes=[lambda s:s['single_judge_fast']['progress'].update(active_provider_calls=1),
      lambda s:s['single_judge_fast']['progress']['provider_status']['groq'].update(resume_after_epoch=50000),
      lambda s:s['single_judge_fast']['progress']['provider_status']['gemini'].update(status='running'),
      lambda s:s['single_judge_fast']['progress']['provider_status']['gemini'].update(status='quota_wait',resume_after_epoch=50000),
      lambda s:s['last_persistence'].update(readback_verified=False)]
    for change in changes:
        state=copy.deepcopy(original);change(state)
        with pytest.raises(RuntimeError):validate(state,control,prior,369,40000,old_runtime_ended=True)
    with pytest.raises(RuntimeError,match='not been reviewed'):
        validate(original,{**control,'source_hashes':{'engine.py':'different'}},prior,369,40000,old_runtime_ended=True)
    with pytest.raises(RuntimeError,match='lease ownership'):
        validate(original,control,{**prior,'lease_version':2},369,40000,old_runtime_ended=True)
    with pytest.raises(RuntimeError,match='Insufficient time'):
        validate(original,control,prior,369,72900,old_runtime_ended=True)


def test_fresh_restore_configures_colab_secrets_before_autoauth_sdk_import(monkeypatch):
    import builtins,os,types,pytest
    from pathlib import Path
    from cloud.single_judge_fast.resume import restore
    original_import=builtins.__import__;original_is_dir=Path.is_dir;original_exists=Path.exists
    monkeypatch.setattr(Path,'is_dir',lambda p:True if str(p)=='/content' else original_is_dir(p))
    monkeypatch.setattr(Path,'exists',lambda p:False if str(p).startswith('/kaggle/working/d1000') else original_exists(p))
    for name in ('KAGGLE_API_TOKEN','GROQ_API_KEY','GEMINI_API_KEY'):monkeypatch.delenv(name,raising=False)
    class ImportReached(Exception):pass
    def guarded_import(name,*args,**kwargs):
        if name=='google.colab':return types.SimpleNamespace(userdata=types.SimpleNamespace(get=lambda label:'fictional'))
        if name=='kaggle':
            assert all(os.environ.get(key)=='fictional' for key in ('KAGGLE_API_TOKEN','GROQ_API_KEY','GEMINI_API_KEY'))
            raise ImportReached()
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    with pytest.raises(ImportReached):restore({}, {}, None,private_colab_transfer_approved=True)
