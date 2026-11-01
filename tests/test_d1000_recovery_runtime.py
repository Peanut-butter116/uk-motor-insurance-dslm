import ast
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
import pytest
from cloud.recovery_v1 import runtime, native_probe


def test_chained_http_status_is_not_lost():
    for status in (400,429,401,403,500,503):
        cause=HTTPError('https://example.invalid',status,'private body omitted',{},None)
        error=ConnectionError('wrapper'); error.__cause__=cause
        chain=native_probe.safe_error(error)
        assert chain[-1]['http_status']==status
        assert 'private body' not in json.dumps(chain) and 'example.invalid' not in json.dumps(chain)


def test_native_probe_stops_without_native_credential(monkeypatch):
    monkeypatch.delenv('KAGGLE_API_V1_TOKEN',raising=False)
    result=native_probe.run()
    assert result['status']=='BLOCKED'
    assert result['blocked_at']=='notebook_native_token_unavailable'
    assert result['secret_service_calls']==result['production_dataset_writes']==0


def test_transfer_gate_precedes_credentials_and_downloads():
    with pytest.raises(PermissionError):runtime.ColabRecovery().prepare()
    with pytest.raises(RuntimeError):runtime.ColabRecovery().resume()


def test_source_bundle_rejects_hash_change_and_path_escape(monkeypatch):
    def notebook(source):return {'cells':[{'cell_type':'code','source':[source]}]}
    with pytest.raises(ValueError,match='hash mismatch'):runtime.supported_bundle(notebook('files={}'))
    source="files={'../escape':'x'}"
    monkeypatch.setattr(runtime,'SOURCE_SHA',hashlib.sha256(source.encode()).hexdigest())
    with pytest.raises(ValueError,match='Unsafe'):runtime.supported_bundle(notebook(source))


def test_restore_adapter_changes_only_credential_prologue():
    source=Path('cloud/d1000/evaluation_only.py').read_text()
    adapted=runtime.restore_adapter(source)
    scope={};exec(adapted,scope)
    assert scope['restore'].__code__.co_names.count('UserSecretsClient')==0
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='restore')
    original=ast.get_source_segment(source,node)+'\n'
    expected=original.replace(runtime.CREDENTIAL_PROLOGUE,'    from kaggle import api\n    api.authenticate()\n',1)
    expected=expected.replace("Path('/kaggle/input')","Path('/content/d1000-private-input')",1)
    assert adapted.co_code==compile(expected,'<reviewed-colab-restore-v1>','exec').co_code
    with pytest.raises(ValueError,match='differs'):
        runtime.restore_adapter(source.replace('secrets=UserSecretsClient()','secrets = UserSecretsClient()'))


def write_state(root,duplicate=False,uncertain=True):
    (root/'judging').mkdir()
    for provider,n in [('groq',36),('gemini',345)]:
        ids=list(range(32)) + list(range(1000 if provider=='groq' else 2000,(1000 if provider=='groq' else 2000)+n-32))
        if duplicate:ids[-1]=ids[0]
        (root/'judging'/f'{provider}.jsonl').write_text(''.join(json.dumps({'evaluation_id':i})+'\n' for i in ids))
    folder=root/'judge_runtime/evaluation_cache_v2/gemini';folder.mkdir(parents=True)
    if uncertain:(folder/'task.attempts.json').write_text(json.dumps([{'status':'transport_uncertain','error_category':'gemini HTTP 503','request_sha256':'fixed'}]))
    return folder


def test_restore_retains_unknown_and_rejects_duplicates_or_changed_raw(tmp_path):
    folder=write_state(tmp_path)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert runtime.verify_journals(tmp_path)['paired']==32
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    (folder/'fixed.raw.json').write_text('{}')
    with pytest.raises(ValueError,match='durable reply'):runtime.verify_journals(tmp_path)


def test_duplicate_votes_fail_closed(tmp_path):
    write_state(tmp_path,duplicate=True)
    with pytest.raises(ValueError,match='population'):runtime.verify_journals(tmp_path)


def test_stage_bundle_reuses_exact_code_and_rejects_mutation(tmp_path):
    files={'cloud/d1000/example.py':'x=1\n','cloud/d1000/audit.json':'{}'}
    runtime.stage_bundle(tmp_path,files)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    runtime.stage_bundle(tmp_path,files)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    (tmp_path/'cloud/d1000/example.py').write_text('x=2\n')
    with pytest.raises(RuntimeError,match='differs'):runtime.stage_bundle(tmp_path,files)
    assert (tmp_path/'cloud/d1000/example.py').read_text()=='x=2\n'


def test_stage_bundle_rejects_unexpected_executable(tmp_path):
    (tmp_path/'injected.py').write_text('pass')
    with pytest.raises(RuntimeError,match='Unexpected'):runtime.stage_bundle(tmp_path,{})


def test_scratch_creates_once_then_waits_for_private_readback(monkeypatch):
    import types
    calls=[];ready=[]
    def create(folder,**kwargs):
        assert kwargs['public'] is False
        assert Path(folder,'nonce.txt').read_text()=='a'*32
        calls.append('create');return types.SimpleNamespace(error=None)
    def info(api,dataset):
        calls.append('info')
        if calls.count('info')==1:
            import requests
            response=requests.Response();response.status_code=403
            raise requests.HTTPError(response=response)
        return types.SimpleNamespace(is_private=True,current_version_number=1)
    api=types.SimpleNamespace(dataset_create_new=create,dataset_status=lambda dataset:'ready')
    monkeypatch.setattr(runtime.time,'sleep',lambda seconds:None)
    receipt=runtime.scratch_persistence(api,info,lambda *args:ready.append(args),'a'*32,attempts=2)
    assert calls==['create','info','info'] and receipt['readback_verified']
    assert ready[0][3]==hashlib.sha256(('a'*32).encode()).hexdigest()
    calls.clear()
    runtime.scratch_persistence(api,lambda *a:types.SimpleNamespace(is_private=True,current_version_number=1),lambda *a:None,'a'*32,resume_created=True)
    assert calls==[]
    with pytest.raises(ValueError):runtime.scratch_persistence(api,info,lambda *a:None,'invalid')


def test_checkpoint_cache_requires_exact_archive_and_refreshes_progress(tmp_path,monkeypatch):
    import types
    data=b'fictional archive';(tmp_path/'artifacts.bin').write_bytes(data)
    monkeypatch.setattr(runtime,'ARCHIVE_SHA',hashlib.sha256(data).hexdigest())
    calls=[];api=types.SimpleNamespace(dataset_download_file=lambda *a,**kw:calls.append((a,kw)))
    runtime.checkpoint_input(api,tmp_path,'artifacts.bin');assert calls==[]
    runtime.checkpoint_input(api,tmp_path,'progress.json');assert len(calls)==1
    (tmp_path/'artifacts.bin').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='differs'):runtime.checkpoint_input(api,tmp_path,'artifacts.bin')
    assert len(calls)==1


def test_progress_reports_only_after_original_private_readback():
    import types
    from cloud.recovery_v1.monitor import install
    state={'last_persistence':{'private':True,'readback_verified':False,'version':240},
           'judge_counts':{'groq':37,'gemini':345},'artifacts_sha256':'fixed',
           'responses_completed':912,'judges':{'groq':{'status':'running','completed':37,'reason':'private text excluded'}}}
    events=[]
    def checkpoint(value):value['last_persistence']['readback_verified']=True;return 'done'
    evaluation=types.SimpleNamespace(checkpoint=checkpoint);install(evaluation,events.append)
    assert evaluation.checkpoint(state)=='done'
    assert events[0]['judge_counts']['groq']==37
    assert 'private text' not in json.dumps(events)
    evaluation=types.SimpleNamespace(checkpoint=lambda state:None);install(evaluation,events.append)
    state['last_persistence']['readback_verified']=False
    with pytest.raises(RuntimeError):evaluation.checkpoint(state)
    assert len(events)==1


def test_saved_vote_monitor_counts_actual_archive_journals(tmp_path):
    import io,tarfile
    from cloud.recovery_v1.monitor import saved_vote_counts
    archive=tmp_path/'fictional.bin'
    with tarfile.open(archive,'w:gz') as tar:
        for provider,ids in [('groq',[1,2]),('gemini',[2,3])]:
            data=''.join(json.dumps({'evaluation_id':i})+'\n' for i in ids).encode()
            entry=tarfile.TarInfo('judging/'+provider+'.jsonl');entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
    result=saved_vote_counts(archive)
    assert result['persisted_judge_counts']=={'groq':2,'gemini':2}
    assert result['persisted_paired']==1
    assert result['persisted_duplicate_votes']=={'groq':0,'gemini':0}


def test_lease_creation_rejection_never_steals_or_retries(tmp_path):
    import types
    from cloud.recovery_v1 import lease
    calls=[]
    api=types.SimpleNamespace(dataset_create_new=lambda *a,**k:(calls.append('create') or types.SimpleNamespace(error='exists')))
    recovery=types.SimpleNamespace(api=api,session='a'*32,publish=lambda *a,**k:None,info=lambda *a:calls.append('info'))
    (tmp_path/'owner.json').write_text(json.dumps({'session':recovery.session,'source_checkpoint':239}))
    lease.install(recovery)
    with pytest.raises(RuntimeError,match='already exists'):recovery.publish(tmp_path,lease.LEASE,create=True)
    assert calls==['create']


def test_lease_rejects_wrong_owner_before_network(tmp_path):
    import types
    from cloud.recovery_v1 import lease
    recovery=types.SimpleNamespace(session='a'*32,publish=lambda *a,**k:None)
    (tmp_path/'owner.json').write_text(json.dumps({'session':'someone_else','source_checkpoint':239}))
    lease.install(recovery)
    with pytest.raises(RuntimeError,match='owner'):recovery.publish(tmp_path,lease.LEASE,create=True)


def test_lease_create_once_then_private_owner_readback(tmp_path,monkeypatch):
    import types,requests
    from cloud.recovery_v1 import lease
    from cloud.d1000 import storage
    calls=[]
    def create(*a,**kwargs):
        assert kwargs['public'] is False
        calls.append('create');return types.SimpleNamespace(error=None)
    def info(*a):
        calls.append('info')
        if calls.count('info')==1:
            response=requests.Response();response.status_code=403
            raise requests.HTTPError(response=response)
        return types.SimpleNamespace(is_private=True,current_version_number=1)
    recovery=types.SimpleNamespace(session='a'*32,api=types.SimpleNamespace(dataset_create_new=create,dataset_status=lambda dataset:'ready'),info=info,publish=lambda *a,**k:None)
    (tmp_path/'owner.json').write_text(json.dumps({'session':recovery.session,'source_checkpoint':239}))
    monkeypatch.setattr(lease.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(storage,'verify_file',lambda dataset,version,name,digest:calls.append(('readback',digest)))
    lease.install(recovery,attempts=2);result=recovery.publish(tmp_path,lease.LEASE,create=True)
    assert calls[:3]==['create','info','info']
    assert calls[-1]==('readback',hashlib.sha256((tmp_path/'owner.json').read_bytes()).hexdigest())
    assert result['readback_verified'] and result['private']


def test_archived_transport_checks_hashes_and_never_resets_ledgers(tmp_path,monkeypatch):
    import types
    from cloud.recovery_v1 import transport
    code=tmp_path/'production_transport/production_transport';code.mkdir(parents=True)
    work=tmp_path/'judge_runtime';(work/'provider_usage').mkdir(parents=True)
    files={}
    for name in ('providers.py','common.py','kelly.py'):
        (code/name).write_text('# fictional '+name)
        files['uk-insurance-QA-data/src/'+name]=hashlib.sha256((code/name).read_bytes()).hexdigest()
    (work/'provider_verification.json').write_text('{}')
    files['uk-insurance-QA-data-local/production-v1/provider_verification.json']=hashlib.sha256(b'{}').hexdigest()
    for provider in ('groq','gemini'):(work/'provider_usage'/f'{provider}.json').write_text('{"retained":true}')
    before={p:p.read_bytes() for p in (work/'provider_usage').iterdir()}
    module=types.SimpleNamespace(__file__=str(code/'providers.py'))
    monkeypatch.setattr(transport.importlib,'import_module',lambda name:module)
    judging=types.SimpleNamespace(transport_paths=lambda root:(code,work))
    result=transport.install_verified(judging,tmp_path,{'archive_sha256':transport.ARCHIVE,'files':files})
    assert result['transport_hashes_verified']==4
    judging.provider_runtime(tmp_path)
    assert before=={p:p.read_bytes() for p in (work/'provider_usage').iterdir()}
    (code/'providers.py').write_text('changed')
    with pytest.raises(RuntimeError,match='hash mismatch'):judging.provider_runtime(tmp_path)


def test_transport_rejects_wrong_original_archive_before_loading(tmp_path):
    from cloud.recovery_v1 import transport
    with pytest.raises(RuntimeError,match='identity'):transport.install_verified(None,tmp_path,{'archive_sha256':'wrong'})


def test_continuation_rejects_unrelated_failure_before_any_network():
    import types
    from cloud.recovery_v1.continue_startup import continue_verified
    recovery=types.SimpleNamespace(ready=True,started=True,lease_version=1)
    try:raise ValueError('fictional')
    except ValueError as failure:
        with pytest.raises(RuntimeError,match='Only the diagnosed'):continue_verified(recovery,failure,{},lambda *a:None,lambda *a:None)


def test_continuation_rejects_an_active_evaluation_worker(monkeypatch):
    import types
    from cloud.recovery_v1 import continue_startup
    scope={};exec(compile('def provider_runtime():\n    return next(iter(()))\n','judging.py','exec'),scope)
    def worker():pass
    monkeypatch.setattr(continue_startup.threading,'enumerate',lambda:[types.SimpleNamespace(is_alive=lambda:True,_target=worker)])
    try:scope['provider_runtime']()
    except StopIteration as failure:
        recovery=types.SimpleNamespace(ready=True,started=True,lease_version=1)
        with pytest.raises(RuntimeError,match='already active'):continue_startup.continue_verified(recovery,failure,{},lambda *a:None,lambda *a:None)
