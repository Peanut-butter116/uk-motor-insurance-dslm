"""Provider-free verification of the a pinned saved batch checkpoint, inside Kaggle."""
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
from difflib import SequenceMatcher

EXPECTED = 'cae9a99a112796a520d7256a7c35056e976bed2c025f2c03fd7c2fe558bfedd4'
VERSION = 9
ACCEPTED = 412


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit():
    manifests = list(Path('/kaggle/input').rglob('migration_manifest.json'))
    assert len(manifests) == 1
    root = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    assert manifest['archive_sha256'] == EXPECTED == sha(root/'checkpoint.bin')
    dest = Path('/kaggle/working/audit-scratch'); dest.mkdir()
    try:
        with tarfile.open(root/'checkpoint.bin') as tar:
            tar.extractall(dest, filter='data')
        assert all(sha(dest/name) == checksum for name, checksum in manifest['files'].items())
        run = dest/'uk-insurance-QA-data-local/production-v1'
        baseline = rows(run/'cloud_baseline.jsonl'); current = rows(run/'data/final/train.jsonl')
        lookup = {r['example_id']:r for r in current}; baseline_ids = {r['example_id'] for r in baseline}
        assert len(baseline) == 409 and len(current) == len(lookup) == ACCEPTED
        assert all(lookup[r['example_id']] == r for r in baseline)
        new = [r for r in current if r['example_id'] not in baseline_ids]
        gold = dest/'uk-insurance-dslm-data/data/gold/gold_v2.jsonl'
        state = json.loads((run/'state.json').read_text())
        before = json.loads((run/'controlled_batch_before.json').read_text())
        assert sha(gold) == before['frozen_gold_sha256']
        assert {key for key,d in state['decisions'].items() if d['status']=='accepted'} == set(lookup)
        comparisons = {}
        for row in new:
            screen = state['decisions'][row['example_id']]['training_semantic_screen']
            comparisons[row['example_id']] = set(screen['comparison_ids']) | set(screen.get('repair_comparison_ids', []))
        missing_baseline = sum(not baseline_ids <= comparisons[r['example_id']] for r in new)
        missing_pairs = [(a['example_id'],b['example_id']) for i,a in enumerate(new) for b in new[i+1:]
            if a['example_id'] not in comparisons[b['example_id']] and b['example_id'] not in comparisons[a['example_id']]]
        norm = lambda s: re.sub(r'[^\w]+',' ',s.casefold()).strip()
        duplicate_pairs = [(a['example_id'],b['example_id']) for a in new for b in current
            if a['example_id'] != b['example_id'] and SequenceMatcher(None,norm(a['question']),norm(b['question'])).ratio() >= .9]
        sys.path.insert(0,str(dest/'uk-insurance-QA-data'))
        from src.production import deterministic, judge_reasons, PAYLOAD
        from src.check_leakage import inspect
        from src.common import digest
        documents = {d['filename']:d for d in rows(dest/'uk-insurance-dslm-data/data/documents.jsonl')}
        for d in json.loads((run/'corpus/manifest.json').read_text()):
            if d['status'].startswith('extracted'):
                documents[d['filename']] = {'sha256':d['source_id']}
        findings,_ = inspect(new,dest/'uk-insurance-dslm-data',protected_splits=('test','dev','seed'))
        deterministic_failures = {}
        for row in new:
            decision = state['decisions'][row['example_id']]
            reasons = deterministic(row,documents,[f['reason'] for f in findings if f['example_id']==row['example_id']])
            reasons += judge_reasons(row,decision['judge'])
            if decision['input_sha256'] != digest({k:row[k] for k in PAYLOAD}):
                reasons.append('validation_payload_hash_mismatch')
            if reasons:
                deterministic_failures[row['example_id']] = sorted(set(reasons))
        assert sha(dest/'uk-insurance-QA-data-local/kelly/cost_ledger.json') == '539ac875377dd07ba575d538eeabc618a95233df74779b4887fd21036e438264'
        history = state['generation'][len(before['generation']):]
        assert len(history) == 1 and history[0]['package_id'] == before['package_id']
        result = {'dataset_version':VERSION,'archive_sha256':EXPECTED,'accepted':len(current),'baseline_preserved':409,
            'unique_question_texts':len({r['question'] for r in current}), 'new_accepted':len(new),'frozen_benchmark_unchanged':True,'files_verified':len(manifest['files']),
            'new_lexical_duplicate_pairs':duplicate_pairs,'missing_training_baseline_screens':missing_baseline,
            'missing_within_batch_semantic_pairs':missing_pairs,'deterministic_and_claim_failures':deterministic_failures,
            'only_one_package_completed':True,'paid_kelly_ledger_unchanged':True,'provider_calls':0}
        result['passed'] = not (duplicate_pairs or missing_baseline or missing_pairs or deterministic_failures)
        Path('/kaggle/working/post_batch_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        shutil.rmtree(dest)


if __name__ == '__main__':
    audit()
