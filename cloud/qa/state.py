"""Cloud checkpoint utilities. No provider calls; tests use fictional records."""
import collections
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def verify_preserved(previous, current):
    lookup = {r['example_id']: r for r in current}
    if len(lookup) != len(current):
        raise ValueError('Duplicate accepted IDs')
    for row in previous:
        if lookup.get(row['example_id']) != row:
            raise ValueError('Previously accepted record changed or missing')


def report(run):
    run = Path(run)
    state = json.loads((run / 'state.json').read_text())
    records = read_rows(run / 'data/final/train.jsonl')
    generated = read_rows(run / 'generated_candidates.jsonl')
    ids = {r['example_id'] for r in generated}
    decisions = state['decisions']
    reasons = collections.Counter(reason for key in ids for reason in decisions.get(key, {}).get('reasons', []))
    labels = collections.Counter(r['answerability'] for r in records)
    baseline_path = run/'cloud_baseline.jsonl'
    baseline_ids = {r['example_id'] for r in read_rows(baseline_path)} if baseline_path.exists() else {r['example_id'] for r in records}
    new_records = [r for r in records if r['example_id'] not in baseline_ids]
    new_labels = collections.Counter(r['answerability'] for r in new_records)
    generated_accepted = sum(decisions.get(key, {}).get('status') == 'accepted' for key in ids)
    return {
        'total_accepted': len(records), 'target': 2000, 'remaining_to_target': max(0, 2000-len(records)),
        'answerability': dict(labels),
        'answerability_percent': {k: round(100*v/len(records), 2) for k, v in labels.items()} if records else {},
        'new_cloud_accepted': len(new_records), 'new_cloud_answerability': dict(new_labels),
        'new_cloud_insufficiency_percent': round(100*new_labels['unanswerable']/len(new_records), 2) if new_records else None,
        'new_cloud_insufficiency_target_percent': [20, 25],
        'candidates_generated': len(generated), 'generated_accepted': generated_accepted,
        'generated_acceptance_rate': generated_accepted/len(generated) if generated else None,
        'generated_dispositions': dict(collections.Counter(decisions.get(k, {}).get('status', 'pending') for k in ids)),
        'verification_reasons': dict(reasons),
        'duplicate_removals': sum(any('duplicate' in r for r in decisions.get(k, {}).get('reasons', [])) for k in ids),
        'leakage_exclusions': sum(any('protected' in r or 'heldout' in r for r in decisions.get(k, {}).get('reasons', [])) for k in ids),
        'by_insurer': dict(collections.Counter(r['insurer'] for r in records)),
        'last_completed_package': next((x['package_id'] for x in reversed(state['generation']) if x['status']=='judged'), None),
        'last_checkpoint_utc': state.get('updated_at_utc'), 'status': state['status'],
        'train_sha256': sha(run/'data/final/train.jsonl'),
        'human_review_completed': 0, 'training_ready': False,
        'statistics_scope': 'All recorded generated candidates, including historical providers; reasons may overlap. No package completion timestamp exists in inherited state.'}


def freeze_milestones(run, records, baseline, now):
    """Stable baseline order followed by deterministic append order; never replace a freeze."""
    run = Path(run)
    verify_preserved(baseline, records)
    order_path = run / 'cloud_acceptance_order.json'
    order = json.loads(order_path.read_text()) if order_path.exists() else [r['example_id'] for r in baseline]
    lookup = {r['example_id']: r for r in records}
    if len(order) != len(set(order)) or not set(order).issubset(lookup):
        raise ValueError('Acceptance order changed')
    order += sorted(set(lookup)-set(order))
    write_json(order_path, order)
    for n in (500, 1000, 2000):
        if len(order) < n:
            continue
        dest = run / 'milestones' / ('D'+str(n))
        selected = [lookup[k] for k in order[:n]]
        content = ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False)+'\n' for r in selected)
        checksum = hashlib.sha256(content.encode()).hexdigest()
        if dest.exists():
            manifest = json.loads((dest/'manifest.json').read_text())
            if manifest['sha256'] != checksum or sha(dest/'train.jsonl') != checksum:
                raise ValueError('Frozen milestone changed')
            continue
        dest.mkdir(parents=True)
        (dest/'train.jsonl').write_text(content)
        write_json(dest/'manifest.json', {
            'dataset_version': 'D'+str(n), 'record_ids': order[:n], 'accepted': n,
            'answerability': dict(collections.Counter(r['answerability'] for r in selected)),
            'source_documents': dict(collections.Counter(r['source_id'] for r in selected)),
            'generation_models': dict(collections.Counter(str(r['generation_model']) for r in selected)),
            'validation_methods': dict(collections.Counter(r['validation_method'] for r in selected)),
            'sha256': checksum, 'created_at_utc': now,
            'validation': {'prior_records_preserved': True, 'unique_ids': True},
            'training_ready': False, 'human_review_required': True,
            'selection': 'baseline order then sorted newly accepted IDs at each durable checkpoint'})


def template_key(question):
    text = re.sub(r'\b\d+(?:[.,]\d+)*\b', '<number>', question.casefold())
    text = re.sub(r'\b(?:hastings direct|admiral|ageas|aviva|axa|aa)\b', '<insurer>', text)
    return re.sub(r'[^\w<>]+', ' ', text).strip()
