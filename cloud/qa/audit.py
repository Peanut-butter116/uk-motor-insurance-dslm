"""Read existing state without changing it; publish aggregates only."""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys
import importlib.util
from types import SimpleNamespace
from state import report, read_rows, sha, write_json


def audit(source, local, legacy, output):
    source, local, legacy = map(Path, (source, local, legacy))
    import jsonschema
    run = local/'production-v1'
    final = (run/'data/final').resolve()
    rows = read_rows(final/'train.jsonl')
    manifest = json.loads((final/'dataset_manifest.json').read_text())
    state = json.loads((run/'state.json').read_text())
    schema = json.loads((source/'config/dataset_schema.json').read_text())
    for row in rows:
        jsonschema.validate(row, schema)
        if state['decisions'][row['example_id']]['status'] != 'accepted':
            raise ValueError('Export/decision mismatch')
        if row['unresolved_citations'] or not row['provenance']['production_validation']['all_protected_semantic_screen_passed']:
            raise ValueError('Accepted quality metadata missing')
    if len(rows) != manifest['accepted'] or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Accepted count or IDs mismatch')
    if sha(final/'train.jsonl') != manifest['train_hashes']['jsonl']:
        raise ValueError('Accepted checksum differs')
    if sha(legacy/'data/gold/gold_v2.jsonl') != manifest['original_gold_sha256']:
        raise ValueError('Frozen benchmark changed')
    result = report(run)
    sys.path.insert(0, str(source.resolve()))
    spec = importlib.util.spec_from_file_location('audited_generation', source/'scripts/generate_motor.py')
    generation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
    completed = {p['package_id'] for p in state['generation']}
    pending = [p for p in generation.packages(SimpleNamespace(run=run, docs={})) if p['id'] not in completed]
    estimated = round(4*len(pending)*result['generated_acceptance_rate'])
    result['remaining_source_queue'] = {
        'packages': len(pending), 'insurers': dict(collections.Counter(p['document']['insurer'] for p in pending)),
        'max_candidates': 4*len(pending), 'historical_yield_additional_estimate': estimated,
        'estimated_total_before_new_gates': len(rows)+estimated,
        'historical_package_yield_additional_estimate': round(len(pending)*result['generated_accepted']/len(state['generation'])),
        'limitation': 'Descriptive historical aggregate yield; not a forecast. Additional approved evidence likely required for 2000.'}
    result.update({'audited_on': '2026-10-25', 'production_commit': subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'schema_valid': True, 'accepted_ids_unique': True, 'frozen_gold_hash_verified': True,
        'original_gold_sha256': manifest['original_gold_sha256'],
        'unique_question_texts': len({r['question'] for r in rows}),
        'model_visible_duplicates': len(rows)-len({(r['question'], r['context']) for r in rows}),
        'validation_methods': dict(collections.Counter(r['validation_method'] for r in rows)),
        'generation_packages': dict(collections.Counter(p['status'] for p in state['generation'])),
        'cloud_generation_launched': False, 'blocked_on': ['Notebook secrets not configured'],
        'insufficiency_target_new_batches_percent': [20, 25],
        'deadline': '2026-11-18',
        'milestones_remaining': {str(n): max(0, n-len(rows)) for n in (500, 1000, 2000)}})
    write_json(output, result)
    print(json.dumps({'accepted': len(rows), 'generated': result['candidates_generated'], 'checks': 'passed'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('source', 'local', 'legacy', 'output'):
        p.add_argument('--'+k, required=True)
    audit(**vars(p.parse_args()))
