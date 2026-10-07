"""Verify a downloaded GPU run; export private judge tasks and unjudged metrics.

Run from the repository root. All artifacts, including summaries, stay private.
This post-processing script does not change the smoke-tested training runtime.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import core, evaluate


def deterministic_metrics(records):
    judge = core.load_module('_report_judge', 'eval/judge.py')
    metrics = judge._row_metrics(records)
    # Correctness is incomplete: never publish the inherited partial composite.
    return {k: metrics[k] for k in ('citation', 'abstention', 'cite_sup', 'cite_tot',
        'over_refuse', 'ans_n', 'refuse_ok', 'refuse_n')}


def summarise(home, bundle, launch, downloaded, output):
    bundle, launch, downloaded = map(Path, (bundle, launch, downloaded))
    if (downloaded / 'FAILED.txt').exists():
        raise ValueError('GPU wrapper reported failure; inspect private diagnostics')
    receipt = json.loads((downloaded / 'SUCCESS.json').read_text())
    expected = json.loads((launch / 'launch.json').read_text())
    if receipt['nonce'] != expected['nonce'] or receipt['kind'] != 'base-eval':
        raise ValueError('Stale or wrong GPU completion receipt')
    manifest = json.loads((downloaded / 'run/manifest.json').read_text())
    input_manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['input'] != input_manifest:
        raise ValueError('GPU results are incomplete or use different benchmark inputs')
    if core.sha(bundle / 'tasks.jsonl') != input_manifest['tasks_sha256']:
        raise ValueError('Prompt bundle changed')
    if manifest['config'] != input_manifest['config'] or manifest['adapter'] is not None:
        raise ValueError('Unexpected configuration or adapter in base evaluation')
    if manifest['model_key'] != expected['model']:
        raise ValueError('Unexpected base model')
    gold = {g['id']: g for g in core.benchmark(home) if g['split'] == input_manifest['split']}
    ctx = core.unique(core.rows(bundle / 'contexts.jsonl'), 'qid')
    ab = core.load_module('_report_abstention', 'eval/abstention.py')
    out = core.private_output(output)
    results = []
    for mode in ('closedbook', 'openbook'):
        label = manifest['model_key'] + '-base-' + mode
        answers = downloaded / 'run' / ('answers_' + label + '.jsonl')
        data = core.rows(answers)
        if set(core.unique(data, 'qid')) != set(gold) or any(
            r['row'] != label or not r['answer'].strip() for r in data):
            raise ValueError('Wrong or empty answer population')
        # Existing scorer verifies gold/context hashes and emits blinded judge tasks.
        evaluate.score(home, bundle, answers, None, out / mode)
        records = []
        for row in data:
            abst = ab.is_abstention_assertion(row['answer'],
                abstain_sentence=core.common.ABSTAIN_SENTENCE, citation_re=core.common.CITATION_RE)
            check = core.rag.verify_citations(row['answer'], ctx[row['qid']]['chunks'])
            records.append({'qid': row['qid'], 'answerable': gold[row['qid']]['answerable'],
                'abst': abst, 'n_sup': check['n_supported'], 'n_tot': check['n_total'],
                'corr_item': None, 'judged': False, 'parse_ok': abst or check['n_total'] > 0})
        metrics = dict(row=label, n=len(records), split=input_manifest['split'],
            status='awaiting_judging', **deterministic_metrics(records),
            parse_rate=sum(r['parse_ok'] for r in records)/len(records),
            generation_seconds=sum(r['latency_s'] for r in data), answers_sha256=core.sha(answers))
        core.write_rows(out / mode / 'deterministic_records.jsonl', records)
        core.write_json(out / mode / 'deterministic_metrics.json', metrics)
        results.append(metrics)
    core.write_json(out / 'summary.json', {'nonce': receipt['nonce'],
        'versions': manifest['versions'], 'rows': results})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('home', 'bundle', 'launch', 'downloaded', 'output'):
        parser.add_argument('--' + name, required=True)
    summarise(**vars(parser.parse_args()))
