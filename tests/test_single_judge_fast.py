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
