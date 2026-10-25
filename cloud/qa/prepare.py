"""Package ONLY existing private state for one-way migration. Never generates QA.

Creates a temporary migration archive, not a new persistent local dataset.
"""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

PIN = 'aef4d8c947750bb2fe1b10530c7060e2ab5cbda6'
SKIP = {'__pycache__', '.git', '.venv'}


def prepare(source, local, legacy, output):
    source, local, legacy, output = map(Path, (source, local, legacy, output))
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise ValueError('Unexpected production code revision')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Production checkout must be clean')
    from state import read_rows, sha
    run = local/'production-v1'
    final = (run/'data/final').resolve()
    baseline = read_rows(final/'train.jsonl')
    if len(baseline) != 409 or sha(final/'train.jsonl') != '6ec5b3728bd6b25cae09d97c81974f371c76a47ce61d2d2795a6919257b44a8f':
        raise ValueError('Re-audit newer baseline before migration')
    # Never include credentials, raw PDFs, or unrelated private experiment data.
    files = {}
    for folder in ('src', 'scripts', 'config'):
        for path in (source/folder).rglob('*'):
            if path.is_file() and not SKIP.intersection(path.parts):
                files['uk-insurance-QA-data/'+str(path.relative_to(source))] = path
    for name in ('requirements.txt',):
        files['uk-insurance-QA-data/'+name] = source/name
    for path in run.rglob('*'):
        rel = path.relative_to(run)
        if not path.is_file() or path.is_symlink() or SKIP.intersection(rel.parts):
            continue
        if rel.parts[0] == 'data':
            if not path.is_relative_to(final):
                continue
        elif rel.parts[0].startswith('before_'):
            continue
        elif path.suffix in ('.log', '.lock') or path.name.endswith('_process.json') or path.name == 'process.json':
            continue
        files['uk-insurance-QA-data-local/production-v1/'+str(rel)] = path
    # Preserve the upstream immutable preservation baseline; avoid exporting all old versions.
    for name in ('train.jsonl',):
        files['uk-insurance-QA-data-local/production-v1/before_provider_migration/final/'+name] = run/'before_provider_migration/final'/name
    files['uk-insurance-QA-data-local/runs/20261017-motor-v1/legacy_motor_candidates.jsonl'] = local/'runs/20261017-motor-v1/legacy_motor_candidates.jsonl'
    files['uk-insurance-QA-data-local/kelly/cost_ledger.json'] = local/'kelly/cost_ledger.json'
    for name in ('documents.jsonl', 'gold/gold_v2.jsonl', 'gold/clusters.json'):
        files['uk-insurance-dslm-data/data/'+name] = legacy/'data'/name
    output.mkdir(parents=True, exist_ok=True)
    # Lock out the original writer while reading its multi-file checkpoint.
    import fcntl
    with (run/'process.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = {name: sha(path) for name, path in files.items()}
        with tarfile.open(output/'checkpoint.tar.gz', 'w:gz') as tar:
            for name, path in sorted(files.items()):
                tar.add(path, arcname=name, recursive=False)
            link = tarfile.TarInfo('uk-insurance-QA-data-local/production-v1/data/final')
            link.type = tarfile.SYMTYPE
            link.linkname = 'export_versions/'+final.name
            tar.addfile(link)
        if hashes != {name: sha(path) for name, path in files.items()}:
            raise ValueError('Checkpoint changed during migration')
    manifest = {'source_commit': PIN, 'files': hashes, 'baseline_count': len(baseline),
                'baseline_train_sha256': sha(final/'train.jsonl'),
                'archive_sha256': sha(output/'checkpoint.tar.gz'), 'format': 1}
    (output/'migration_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'dataset-metadata.json').write_text(json.dumps({
        'id': 'scarletthe0116/uk-motor-qa-production-v1',
        'title': 'UK Motor QA Production v1',
        'licenses': [{'name': 'other'}],
        'description': 'Private resumable QA checkpoint; restricted policy evidence. No redistribution license granted.'}, indent=2)+'\n')
    print(json.dumps({'files': len(files), 'baseline_count': len(baseline), 'archive_sha256': manifest['archive_sha256']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'local', 'legacy', 'output'):
        p.add_argument('--'+name, required=True)
    prepare(**vars(p.parse_args()))
