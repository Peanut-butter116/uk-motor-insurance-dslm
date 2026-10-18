"""Verify downloaded six-model outputs and export separate motor-only summaries.

Missing semantic judgments stay null. Partial models never enter comparison rows.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core
from experiments.six_model_benchmark import normalise_answer


def verify(home, launch, downloaded, output, public):
    home,launch,downloaded=map(Path,(home,launch,downloaded))
    public=Path(public)
    if any((public/name).exists() for name in ['six_model_summary.json','six_model_summary.csv','six_model_verification.json']):
        raise FileExistsError('Choose a fresh public result directory; verified results are never overwritten')
    expected=json.loads((launch/'local-bundle/manifest.json').read_text())
    result=json.loads((downloaded/'benchmark-output/receipt.json').read_text())
    if result['input_manifest']!=expected:raise ValueError('Downloaded input manifest mismatch')
    cfg=result['config']
    if hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()!=expected['config_sha256']:
        raise ValueError('Downloaded protocol changed')
    contexts=launch/'local-bundle/contexts.jsonl'
    if core.sha(contexts)!=expected['contexts_sha256']:raise ValueError('Scoring evidence changed')
    ctx=core.unique(core.rows(contexts),'qid')
    gold={g['id']:g for g in core.benchmark(home) if g['split']=='test' and g['line']=='motor'}
    if len(gold)!=76 or set(ctx)!=set(gold):raise ValueError('Motor population mismatch')
    if core.sha(home/'data/gold/gold_v2.jsonl')!=expected['gold_sha256']:raise ValueError('Protected gold changed')
    requested={m['id']:m for m in cfg['models']}
    statuses={r['model']:r for r in result['models']}
    if set(requested)!=set(statuses):raise ValueError('Missing model execution status')
    ab=core.load_module('_six_model_abstention','eval/abstention.py')
    out=core.private_output(output);summary=[];predictions=[];tasks=[];mapping=[];integrity=[]
    for model,status in statuses.items():
        spec=requested[model]
        if status['revision']!=spec['revision']:raise ValueError('Checkpoint revision changed')
        if status['status']!='inference_complete':continue
        path=downloaded/'benchmark-output'/(model.replace('/','--')+'.jsonl')
        if core.sha(path)!=status['answers_sha256']:raise ValueError('Raw answer file changed')
        rows=core.rows(path)
        if len(rows)!=152 or {(r['qid'],r['mode']) for r in rows}!={(q,m) for q in gold for m in ['closedbook','openbook']}:
            raise ValueError('Incomplete or duplicate matched motor answer population')
        if any(r['model']!=spec or not r['raw_answer'].strip() for r in rows):raise ValueError('Wrong checkpoint or empty raw answer')
        integrity.append({'model':model,'revision':spec['revision'],'raw_sha256':core.sha(path),'matched_questions':76,'tasks':152})
        for mode in ['closedbook','openbook']:
            metrics=[]
            for raw in [r for r in rows if r['mode']==mode]:
                norm=normalise_answer(raw['raw_answer'],'DeepSeek-R1' in model)
                a=norm['answer'];q=raw['qid'];g=gold[q]
                abst=ab.is_abstention_assertion(a,abstain_sentence=core.common.ABSTAIN_SENTENCE,citation_re=core.common.CITATION_RE)
                citations=core.rag.verify_citations(a,ctx[q]['chunks']) if mode=='openbook' else {'n_supported':0,'n_total':0}
                parsed=abst or bool(core.common.CITATION_RE.search(a))
                row={**raw,**norm,'abstention_decision':abst,'answerable':g['answerable'],
                    'peak_gpu_allocated_gib':raw.get('peak_gpu_allocated_gib',raw.get('peak_gpu_allocated_gb')),
                    'citation_count':citations['n_total'] if mode=='openbook' else None,
                    'resolved_citation_count':citations['n_supported'] if mode=='openbook' else None,
                    'parse_success':parsed,'correctness':None,'claim_support':None,
                    'unsupported_claim_flag':None,'review_status':'awaiting_independent_judging'}
                metrics.append(row);predictions.append(row)
                if g['answerable'] and not abst:
                    code=f'j{len(tasks):05d}'
                    tasks.append({'task':code,'question':g['question'],'gold_answer':g['gold_answer'],
                        'key_facts':g['key_facts'],'must_not_assert':g.get('must_not_assert',''),'model_answer':a})
                    mapping.append({'task':code,'qid':q,'model':model,'condition':mode})
            ans=[r for r in metrics if r['answerable']];unas=[r for r in metrics if not r['answerable']]
            refused=sum(r['abstention_decision'] for r in ans);correct_ref=sum(r['abstention_decision'] for r in unas)
            n_cites=sum(r['citation_count'] or 0 for r in metrics)
            summary.append({'experiment':'six_model_frozen_motor_descriptive_v1','model':model,'revision':spec['revision'],
                'parameters_billion':spec['parameters_billion_nominal'],'condition':'Base + RAG' if mode=='openbook' else 'Base',
                'n':76,'answerable_n':len(ans),'unanswerable_n':len(unas),'correctness':None,'claim_support':None,
                'unsupported_claim_rate':None,'balanced_abstention':.5*(correct_ref/len(unas)+1-refused/len(ans)),
                'unnecessary_refusal_rate':refused/len(ans),'correct_refusal_rate':correct_ref/len(unas),
                'citation_resolution':sum(r['resolved_citation_count'] or 0 for r in metrics)/n_cites if mode=='openbook' and n_cites else None,
                'parse_rate':statistics.mean(r['parse_success'] for r in metrics),
                'mean_latency_seconds':statistics.mean(r['latency_seconds'] for r in metrics),
                'peak_gpu_allocated_gib':max(r.get('peak_gpu_allocated_gib',r.get('peak_gpu_allocated_gb')) for r in rows),
                'peak_memory_scope':'whole model run, both conditions',
                'output_cap_rate':statistics.mean(r['hit_output_cap'] for r in metrics),
                'final_answer_missing_rate':statistics.mean(not r['final_answer_present'] for r in metrics),
                'precision':cfg['loading_precision'],'gpu':result['gpu'],'review_status':'awaiting_independent_judging'})
    core.write_rows(out/'normalised_predictions.jsonl',predictions)
    core.write_rows(out/'judge_tasks.jsonl',tasks);core.write_rows(out/'judge_map.jsonl',mapping)
    (out/'rubric.md').write_text((core.ROOT/'eval/JUDGE_RUBRIC.md').read_text())
    if predictions:
        with (out/'normalised_predictions.csv').open('x',newline='') as f:
            fields=['qid','mode','answer','answerable','abstention_decision','parse_success','latency_seconds','peak_gpu_allocated_gib','review_status']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in predictions)
    public.mkdir(parents=True,exist_ok=True)
    core.write_json(public/'six_model_summary.json',summary)
    if summary:
        with (public/'six_model_summary.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    core.write_json(public/'six_model_verification.json',{'status':'verified_download',
        'receipt_sha256':core.sha(downloaded/'benchmark-output/receipt.json'),'models':list(statuses.values()),
        'integrity':integrity,'correctness_pending_responses':len(tasks),'required_independent_votes':3*len(tasks),
        'selection_permitted':False,'semantic_support':'not_judged'})
    print('Verified completed models:',len(integrity),'private normalised predictions:',len(predictions))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['home','launch','downloaded','output','public']:p.add_argument('--'+n,required=True)
    verify(**vars(p.parse_args()))
