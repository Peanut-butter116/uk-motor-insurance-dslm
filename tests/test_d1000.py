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


def test_private_readback_uses_sdk_and_checks_exact_bytes(monkeypatch):
    import sys,types
    import pytest
    from cloud.d1000.storage import verify_file
    payload=b'{"status":"synthetic"}\n';calls=[]
    def download(dataset,name,**kwargs):
        calls.append((dataset,name,kwargs))
        (Path(kwargs['path'])/name).write_bytes(payload)
    monkeypatch.setitem(sys.modules,'kaggle',types.SimpleNamespace(api=types.SimpleNamespace(dataset_download_file=download)))
    verify_file('fictional/private',2,'progress.json',hashlib.sha256(payload).hexdigest())
    assert calls[0][0:2]==('fictional/private/2','progress.json')
    assert calls[0][2]['unzip'] is True
    assert not Path(calls[0][2]['path']).exists()
    with pytest.raises(AssertionError,match='Remote artifact hash mismatch'):
        verify_file('fictional/private',2,'progress.json','0'*64)
    with pytest.raises(AssertionError,match='cloud-only'):
        verify_file('fictional/private',2,'private-training.csv','0'*64)


def test_provider_runtime_outside_copied_source_guard(tmp_path):
    from cloud.d1000.judging import transport_paths
    code,runtime=transport_paths(tmp_path/'campaign')
    source_root=code.parent.resolve()
    assert runtime.resolve()!=source_root and source_root not in runtime.resolve().parents
    assert code.parent.name=='production_transport'
    assert runtime.parent==tmp_path/'campaign'


def test_exact_audited_resume_repair_rejects_unrelated_code(monkeypatch):
    import pytest
    from cloud.d1000 import campaign
    audit=json.loads(Path('cloud/d1000/resume_repair_v1.json').read_text())
    destination=hashlib.sha256(''.join(audit['destination_hashes'][p] for p in sorted(audit['destination_hashes'])).encode()).hexdigest()
    state={'code_fingerprint':audit['source_fingerprint']}
    campaign.audited_resume_repair(state,destination)
    assert state['resume_repairs']==[audit]
    assert audit['source_hashes']['cloud/d1000/train_worker.py']==audit['destination_hashes']['cloud/d1000/train_worker.py']
    with pytest.raises(AssertionError):
        campaign.audited_resume_repair({'code_fingerprint':'0'*64},destination)
    original=campaign.sha
    monkeypatch.setattr(campaign,'sha',lambda p:'0'*64 if p.name=='train_worker.py' else original(p))
    with pytest.raises(AssertionError,match='audited destination'):
        campaign.audited_resume_repair({'code_fingerprint':audit['source_fingerprint']},destination)


def test_restore_keeps_latest_unfinished_optimizer_without_altering_archive(tmp_path):
    import io,tarfile
    from cloud.d1000.campaign import restore_runtime
    archive=tmp_path/'artifacts.bin'
    names=['runs/qwen3b/full/checkpoints/checkpoint-25/optimizer.pt',
           'runs/qwen3b/full/adapter/adapter.safetensors',
           'runs/smollm1b7/full/checkpoints/checkpoint-25/optimizer.pt',
           'runs/smollm1b7/full/checkpoints/checkpoint-50/optimizer.pt',
           'runs/qwen3b/base-responses.jsonl']
    with tarfile.open(archive,'w:gz') as t:
        for name in names:
            m=tarfile.TarInfo(name);m.size=1;t.addfile(m,io.BytesIO(b'x'))
    h=hashlib.sha256(archive.read_bytes()).hexdigest();root=tmp_path/'restore'
    audit=restore_runtime(archive,root,{'models':{'qwen3b':{'status':'complete'}},'artifacts_sha256':h})
    assert (root/names[1]).exists() and (root/names[4]).exists()
    assert (root/names[3]).exists() and not (root/names[2]).exists() and not (root/names[0]).exists()
    assert audit['optimizer_members_not_extracted']==2
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==h
