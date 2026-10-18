"""Verify historical GPU receipts and create private paired CSV/public aggregates."""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics
from experiments import core
from scripts.summarise_gpu_results import summarise
from scripts.error_analysis import make_records, aggregate


def write_csv(path, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def audit(home, output, public, qwen_download=None):
    home = Path(home).resolve()
    out = core.private_output(output)
    public = Path(public)
    public.mkdir(parents=True, exist_ok=True)
    exp = home / 'experiments'
    bundle = exp / '20261007/test-eval'
    runs = [
        ('qwen3b', exp/'20261007/launch-base-eval-qwen3b', exp/'20261007/output-qwen3b-v1', False),
        ('ministral3b', exp/'20261007/launch-base-eval-ministral3b', exp/'20261007/output-ministral3b-v1', False),
        ('ministral3b', exp/'20261008/launch-sft-eval-v1', exp/'20261011/output-sft-eval-v1', True),
    ]
    if qwen_download:
        runs.append(('qwen3b', exp/'20261013/launch-qwen-sft-eval-v1', Path(qwen_download), True))
    gold = {r['id']: r for r in core.benchmark(home) if r['split'] == 'test'}
    ctx = core.unique(core.rows(bundle/'contexts.jsonl'), 'qid')
    summary, paired, receipts, all_records = [], [], [], {}
    for key, launch, downloaded, tuned in runs:
        label = key + ('-sft' if tuned else '-base')
        scored = out / label
        summarise(home, bundle, launch, downloaded, scored, tuned)
        sm = json.loads((scored/'summary.json').read_text())
        receipts.append({'condition':label, 'success_sha256':core.sha(downloaded/'SUCCESS.json'),
                         'launch_sha256':core.sha(launch/'launch.json'),
                         'run_manifest_sha256':core.sha(downloaded/'run/manifest.json'),
                         'nonce':sm['nonce'], 'runtime':sm['versions']})
        for mode in ('closedbook', 'openbook'):
            row = label+'-'+mode
            answers = core.rows(downloaded/'run'/('answers_'+row+'.jsonl'))
            byid = core.unique(answers, 'qid')
            records = make_records(answers, gold, ctx, [], row)
            all_records[row] = {r['qid']:r for r in records}
            for population in ('historical_full', 'motor'):
                selected = [r for r in records if population == 'historical_full' or r['line']=='motor']
                metrics = aggregate(selected)
                summary.append({'population':population, 'model':key, 'parameters_billion':3,
                    'condition': 'SFT + RAG' if tuned and mode=='openbook' else 'SFT' if tuned else 'Base + RAG' if mode=='openbook' else 'Base',
                    'sft_examples':21 if tuned else 0, 'n':len(selected), 'correctness':None,
                    'balanced_abstention':metrics['abstention'],
                    'citation_resolution':metrics['citation'] if mode=='openbook' else None,
                    'claim_support':None, 'parse_rate':metrics['parse_rate'],
                    'over_refusals':metrics['over_refuse'], 'answerable_n':metrics['ans_n'],
                    'correct_refusals':metrics['refuse_ok'], 'unanswerable_n':metrics['refuse_n'],
                    'generation_seconds':sum(byid[r['qid']]['latency_s'] for r in selected),
                    'peak_gpu_memory_gb':None, 'composite':None, 'review_status':'awaiting_independent_judging'})
            for r in records:
                a = byid[r['qid']];g=gold[r['qid']]
                paired.append({'question_id':r['qid'],'insurance_topic':g['question_type'],
                    'population':g['line'],'answerability':g['answerable'],'model':key,'model_size':'3B',
                    'condition':row,'prediction':a['answer'],'reference_answer':g['gold_answer'],
                    'correctness':None,'abstention_decision':r['abst'],
                    'citation_resolution':r['n_sup']/r['n_tot'] if mode=='openbook' and r['n_tot'] else None,
                    'claim_support':None,'unsupported_claim_flag':None,'parse_success':r['parse_ok'],
                    'latency_seconds':a['latency_s'],'review_status':'awaiting_independent_judging'})
    write_csv(out/'paired_results.csv', paired)
    # QID pairs bootstrap the parse-rate delta, not independent condition samples.
    intervals=[]
    for model in ('ministral3b','qwen3b'):
        for mode in ('closedbook','openbook'):
            a=all_records.get(model+'-base-'+mode);b=all_records.get(model+'-sft-'+mode)
            if not a or not b:continue
            qs=[q for q in a if a[q]['line']=='motor']
            delta=[int(b[q]['parse_ok'])-int(a[q]['parse_ok']) for q in qs]
            rng=random.Random(20261020)
            draws=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(5000))
            intervals.append({'model':model,'condition':mode,'metric':'motor_parse_rate_sft_minus_base',
                'paired_n':len(delta),'estimate':statistics.mean(delta),'lower_95':draws[124],
                'upper_95':draws[4874], 'method':'paired QID percentile bootstrap, 5000 draws, seed 20261020',
                'limitations':'Exploratory, no multiplicity correction; parse success does not establish answer correctness'})
    write_csv(public/'model_summary.csv', summary)
    (public/'paired_uncertainty.json').write_text(json.dumps(intervals,indent=2)+'\n')
    (public/'verification_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    (public/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Verified',len(paired),'private paired rows;',len(summary),'aggregate rows')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('home','output','public'):p.add_argument('--'+name,required=True)
    p.add_argument('--qwen-download')
    audit(**vars(p.parse_args()))
