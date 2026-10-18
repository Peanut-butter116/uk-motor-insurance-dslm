"""Freeze an atomic accepted-data export outside the public repository.

CSV cells are JSON encoded for lossless round-trip. A separate display CSV
provides ordinary readable strings, with spreadsheet formula injection escaped.
No production process, state or live pointer is modified.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(source, source_repo, output):
    source = Path(source).resolve(strict=True)  # resolve live pointer exactly once
    output = Path(output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('Restricted snapshots must remain outside the public repository')
    manifest = json.loads((source / 'dataset_manifest.json').read_text())
    if sha(source / 'train.jsonl') != manifest['train_hashes']['jsonl']:
        raise ValueError('Source export hash mismatch')
    records = [json.loads(s) for s in (source / 'train.jsonl').read_text().splitlines() if s]
    if len(records) != manifest['accepted'] or not records:
        raise ValueError('Source population mismatch')
    if len({r['example_id'] for r in records}) != len(records):
        raise ValueError('Duplicate example IDs')
    for r in records:
        if (r['product'] != 'motor' or r['split'] != 'train'
                or r['validation_status'] != 'accepted' or r['unresolved_citations']
                or not r['provenance']['production_validation'].get('all_protected_semantic_screen_passed')):
            raise ValueError('Ineligible or unresolved record')
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    canonical = output / 'interim_motor_training.jsonl'
    canonical.write_bytes((source / 'train.jsonl').read_bytes())
    fields = sorted({k for r in records for k in r})
    csv_path = output / 'interim_motor_training.csv'
    with csv_path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(r.get(k), ensure_ascii=False) for k in fields} for r in records)
    with csv_path.open(newline='') as f:
        reread = [{k: json.loads(v) for k, v in r.items()} for r in csv.DictReader(f)]
    assert reread == records
    with (output / 'interim_motor_training_display.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            values = {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in r.items()}
            w.writerow({k: "'" + v if v.startswith(('=', '+', '-', '@', '\t', '\r')) else v for k, v in values.items()})
    norm = lambda q: ' '.join(q.casefold().split())
    result = {
        'source_repository': 'https://github.com/Peanut-butter116/uk-insurance-QA-data',
        'source_snapshot_path': str(source), 'dataset_version': source.name,
        'source_commit': subprocess.check_output(['git', '-C', str(source_repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'export_timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'training_examples': len(records), 'unique_questions': len({norm(r['question']) for r in records}),
        'insurers': sorted({e['insurer'] for r in records for e in r['evidence_text']}),
        'source_documents': len({e['source_id'] for r in records for e in r['evidence_text']}),
        'evidence_sections': len({(e['source_id'], e['section']) for r in records for e in r['evidence_text']}),
        'answerability': dict(collections.Counter(r['answerability'] for r in records)),
        'origins': dict(collections.Counter(r['data_origin'] for r in records)),
        'validation_methods': dict(collections.Counter(r['validation_method'] for r in records)),
        'human_review_completed': manifest['human_review_completed'],
        'training_ready': manifest['training_ready'], 'source_status': manifest['status'],
        'all_protected_semantic_screen_passed': True,
        'duplicate_question_rows': len(records) - len({norm(r['question']) for r in records}),
        'policy_versions': 'Source-content hashes, printed versions unverified',
        'csv_encoding': 'Canonical: JSON in every cell. Display: readable strings, nested values JSON, formula prefixes escaped.',
        'fields': fields,
        'sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        'limitations': ['Interim accepted export, not cleared for new full training',
                        'Human review outstanding', 'Repeated question variants require grouped deduplication before training'],
    }
    (output / 'interim_dataset_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    for p in output.iterdir():
        p.chmod(0o400)
    output.chmod(0o500)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--source-repo', required=True)
    p.add_argument('--output', required=True)
    print(json.dumps(freeze(**vars(p.parse_args())), indent=2))
