"""Cloud recovery and immutable-state contracts, using fictional data only."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import pytest

QA = Path(__file__).resolve().parents[1]/'cloud/qa'
sys.path.insert(0, str(QA))
from state import freeze_milestones, verify_preserved, template_key
from runner import cap_source, load_archive, require_cloud


def row(i):
    return {'example_id': str(i), 'question': 'Synthetic question '+str(i),
            'answerability': 'answerable', 'source_id': 'fictional-policy',
            'generation_model': 'fictional-teacher', 'validation_method': 'fictional-independent-check'}


def test_preservation_checks_complete_record():
    with pytest.raises(ValueError, match='changed or missing'):
        verify_preserved([row(1)], [{**row(1), 'question': 'Changed'}])
    with pytest.raises(ValueError, match='Duplicate'):
        verify_preserved([row(1)], [row(1), row(1)])


def test_nested_milestones_are_immutable(tmp_path):
    baseline = [row(i) for i in range(409)]
    records = [row(i) for i in range(500)]
    freeze_milestones(tmp_path, records, baseline, 'fictional-date')
    first = (tmp_path/'milestones/D500/train.jsonl').read_bytes()
    freeze_milestones(tmp_path, [row(i) for i in range(1000)], baseline, 'later-date')
    assert (tmp_path/'milestones/D500/train.jsonl').read_bytes() == first
    thousand = (tmp_path/'milestones/D1000/train.jsonl').read_text().splitlines()
    assert thousand[:500] == first.decode().splitlines()
    changed = [row(i) for i in range(1000)]
    changed[410]['question'] = 'Altered after freeze'
    with pytest.raises(ValueError, match='Frozen milestone changed'):
        freeze_milestones(tmp_path, changed, baseline, 'later-date')


def test_template_substitutions():
    assert template_key('Is the limit £200?') == template_key('Is the limit £400?')


def test_cloud_execution_refuses_local_machine(monkeypatch):
    monkeypatch.delenv('KAGGLE_KERNEL_RUN_TYPE', raising=False)
    with pytest.raises(RuntimeError, match='only inside Kaggle'):
        require_cloud()


def test_target_transform_preserves_strings():
    source = 'prompt="Do not change 10000"\na=10000\nb=10000\nc=10000\nd=10000\n'
    namespace = {}; exec(cap_source(source, 500), namespace)
    assert namespace['prompt'] == 'Do not change 10000'
    assert namespace['a'] == 500
    with pytest.raises(ValueError, match='sites changed'):
        cap_source('x=10000', 500)


def test_checkpoint_refuses_corruption_and_path_traversal(tmp_path):
    archive = tmp_path/'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo('../escape'); info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    manifest = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'source_commit': 'aef4d8c947750bb2fe1b10530c7060e2ab5cbda6', 'files': {}}
    dest = tmp_path/'session'; dest.mkdir()
    with pytest.raises(tarfile.FilterError):
        load_archive(archive, manifest, dest)
    manifest['archive_sha256'] = 'bad'
    with pytest.raises(ValueError, match='mismatch'):
        load_archive(archive, manifest, dest)


def test_exact_inherited_http_recovery_keeps_audit_and_quota(tmp_path, monkeypatch):
    import runner
    path = tmp_path/'provider_cache/gemini/eb774bffd908a8f992743880d1b365fd0cf68dd173e904813d7a86602219cf7e.attempts.json'
    path.parent.mkdir(parents=True)
    history = [{'status':'transport_uncertain', 'request_sha256':'fictional-request', 'reservation':'keep-me'}]
    path.write_text(json.dumps(history))
    (tmp_path/'state.json').write_text(json.dumps({'stop_reason':'gemini HTTP 503'}))
    monkeypatch.setattr(runner, 'sha', lambda _: '4339a2ab3e8a81cea37e1008b5cd542f6ddee3e786112a2eed91a327c207ad75')
    runner.recover_inherited_503(tmp_path)
    assert json.loads(path.read_text()) == []
    assert json.loads((tmp_path/'cloud_http_503_recovery.json').read_text())['original_attempts'] == history
    # Generic unknown transports remain blocked, never cleared by a blanket reset.
    path.write_text(json.dumps(history))
    (tmp_path/'state.json').write_text(json.dumps({'stop_reason':'gemini transport timeout'}))
    with pytest.raises(RuntimeError, match='no longer matches'):
        runner.recover_inherited_503(tmp_path)
