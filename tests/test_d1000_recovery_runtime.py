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


def test_scratch_is_unique_private_create_only_without_lookup(tmp_path):
    calls=[]
    def publish(folder,dataset,create=False):
        calls.append((dataset,create))
        assert (folder/'nonce.txt').read_text()=='a'*32
        return {'dataset':dataset,'version':1,'private':True,'readback_verified':True}
    result=runtime.scratch_persistence(publish,'a'*32)
    assert calls==[('scarletthe0116/uk-motor-d1000-scratch-'+'a'*12,True)]
    assert result['readback_verified']
    with pytest.raises(ValueError):runtime.scratch_persistence(publish,'invalid')


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
