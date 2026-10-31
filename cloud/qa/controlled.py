"""One explicitly pinned source package; retain aggregate evidence in Kaggle only."""
import collections
import json
from pathlib import Path
import shutil
import time
from runner import main, DATASET, dataset_info, load_archive
from state import read_rows, report, sha, verify_preserved, write_json

PACKAGE = 'a710f17c1583c5d20f8de5a65eca41c351325123a9dde4484c7c130982842c9d:p42'


def run():
    result = {'test_success': False, 'controlled_package_id': PACKAGE, 'source_package_limit': 1}
    try:
        main(max_packages=1, target=500, package_id=PACKAGE)
    except Exception as error:
        result['exception_class'] = type(error).__name__
        # Provider errors are redacted upstream; do not serialize arbitrary SDK errors.
        if type(error).__name__ in ('SafeAPIError', 'QuotaPause'):
            result['safe_api_error'] = str(error)
        elif hasattr(error, 'code') and isinstance(error.code, int):
            result['http_status'] = error.code
    session = Path('/kaggle/working/session')
    run = session/'uk-insurance-QA-data-local/production-v1'
    if (run/'state.json').exists():
        state = json.loads((run/'state.json').read_text())
        before = json.loads((run/'controlled_batch_before.json').read_text())
        baseline = read_rows(run/'cloud_baseline.jsonl')
        current = read_rows(run/'data/final/train.jsonl')
        verify_preserved(baseline, current)
        result.update(report(run))
        result['accepted_before'] = before['progress']['total_accepted']
        result['accepted_delta'] = len(current)-result['accepted_before']
        result['generated_before'] = before['progress']['candidates_generated']
        result['generated_delta'] = result['candidates_generated']-result['generated_before']
        result['baseline_409_preserved'] = True
        gold = session/'uk-insurance-dslm-data/data/gold/gold_v2.jsonl'
        result['frozen_benchmark_unchanged'] = sha(gold) == before['frozen_gold_sha256']
        changed = {key: value for key, value in state['decisions'].items() if before['decisions'].get(key) != value}
        result['changed_candidate_dispositions'] = dict(collections.Counter(d['status'] for d in changed.values()))
        result['changed_candidate_failure_reasons'] = dict(collections.Counter(r for d in changed.values() for r in d.get('reasons', [])))
        result['changed_candidate_ids'] = sorted(changed)
        result['completed_controlled_package'] = [p for p in state['generation'] if p['package_id'] == PACKAGE]
        added_packages = state['generation'][len(before['generation']):]
        result['only_controlled_package_processed'] = all(p['package_id'] == PACKAGE for p in added_packages)
        result['api_stop_reason'] = state.get('stop_reason')
        result['resume_after_epoch'] = state.get('resume_after_epoch')
        result['current_accepted_ids_unique'] = len({r['example_id'] for r in current}) == len(current)
        result['accepted_decisions_match_export'] = {k for k,d in state['decisions'].items() if d['status']=='accepted'} == {r['example_id'] for r in current}
        import jsonschema
        schema = json.loads((session/'uk-insurance-QA-data/config/dataset_schema.json').read_text())
        for row in current:
            jsonschema.validate(row, schema)
        result['accepted_schema_valid'] = True
        baseline_ids = {r['example_id'] for r in baseline}
        new = [r for r in current if r['example_id'] not in baseline_ids]
        result['new_acceptances_independently_verified'] = all(
            r['provenance']['production_validation'].get('provider') == 'gemini' and
            r['provenance']['production_validation'].get('all_protected_semantic_screen_passed') is True and
            r['provenance']['production_validation'].get('training_semantic_screen', {}).get('verdict', {}).get('duplicate') is False
            for r in new)
        receipt_file = Path('/kaggle/working/persistence_receipt.json')
        if receipt_file.exists():
            from kaggle import api
            receipt = json.loads(receipt_file.read_text())
            info = dataset_info(api)
            result['persisted_dataset_version'] = receipt['version']
            result['dataset_private'] = info.is_private
            incoming = Path('/kaggle/working/restore-check'); incoming.mkdir(exist_ok=True)
            for name in ('migration_manifest.json', 'checkpoint.bin'):
                api.dataset_download_file(DATASET+'/'+str(receipt['version']), name,
                    path=str(incoming), force=True, quiet=True, unzip=True)
            manifest = json.loads((incoming/'migration_manifest.json').read_text())
            dest = incoming/'restored'; dest.mkdir(exist_ok=True)
            load_archive(incoming/'checkpoint.bin', manifest, dest)
            restored = dest/'uk-insurance-QA-data-local/production-v1'
            restored_rows = read_rows(restored/'data/final/train.jsonl')
            verify_preserved(current, restored_rows)
            result['persisted_restore_verified'] = len(restored_rows) == len(current) and sha(restored/'state.json') == sha(run/'state.json')
            result['persisted_archive_sha256'] = manifest['archive_sha256']
            shutil.rmtree(incoming)
        result['test_success'] = bool(result.get('persisted_restore_verified') and result.get('dataset_private') and
            result['completed_controlled_package'] and result['baseline_409_preserved'] and
            result['only_controlled_package_processed'] and result['frozen_benchmark_unchanged'] and
            result['accepted_decisions_match_export'] and result['new_acceptances_independently_verified'])
    write_json(Path('/kaggle/working/controlled_batch_report.json'), result)
    print(json.dumps(result, indent=2))
    return result
