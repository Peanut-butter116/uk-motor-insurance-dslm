"""Private, paired post-hoc analysis using unchanged inherited metrics.

Configuration is a private JSON mapping labels to answer and optional score paths.
No correctness/composite or faithfulness is inferred from lexical similarity.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core

AB = core.load_module('_analysis_abstention','eval/abstention.py')
JUDGE = core.load_module('_analysis_judge','eval/judge.py')
VOTES = core.load_module('_analysis_votes','eval/judge_votes.py')


def evidence_presence(gold, chunks):
    """Exact normalized gold-quote location proxy, not semantic retrieval recall."""
    citations = gold.get('gold_citations') or []
    if not citations or any(not core.common.norm(c.get('quote','')) for c in citations):
        return {'status':'unassessable','found':0,'total':len(citations)}
    hits = []
    for c in citations:
        hits.append(any(core.common.norm(c['quote']) in core.common.norm(x['text'])
            and c['doc_file'] == x['meta']['filename']
            and c['insurer'].casefold() == x['meta']['insurer'].casefold()
            for x in chunks))
    count = sum(hits)
    status = 'all_gold_quotes_located' if count == len(hits) else (
        'some_gold_quotes_located' if count else 'no_gold_quotes_located')
    return {'status':status,'found':count,'total':len(hits)}


def validate_answers(data, gold, label):
    answer_map = core.unique(data,'qid')
    if set(answer_map) != set(gold) or any(a['row'] != label or not a['answer'].strip() for a in data):
        raise ValueError('Incomplete, duplicate, empty or mixed answer population')
    return answer_map


def make_records(data, gold, contexts, scores, label, historical=False):
    if historical and label not in ('qwen2.5-7b-bnb-openbook','qwen2.5-7b-raft-openbook'):
        raise ValueError('Historical vote exception is restricted to the two published baselines')
    answers = validate_answers(data,gold,label)
    score_map = {}
    for s in scores:
        if s['row'] != label:continue
        if s['qid'] in score_map:raise ValueError('Duplicate correctness score')
        if s['qid'] not in gold:raise ValueError('Unexpected correctness QID')
        score_map[s['qid']] = s
    records = []
    for q,g in gold.items():
        answer = answers[q]['answer']
        abst = AB.is_abstention_assertion(answer,abstain_sentence=core.common.ABSTAIN_SENTENCE,
                                         citation_re=core.common.CITATION_RE)
        citations = core.rag.verify_citations(answer,contexts[q]['chunks'])
        score = score_map.get(q); corr = None; judged = False
        if g['answerable']:
            if abst:corr = 0.0
            elif score is not None:
                ok,reason = VOTES.vote_valid(score,len(g['key_facts']))
                if not ok:raise ValueError('Invalid correctness vote: '+reason)
                if not historical:
                    if score.get('n_votes') != 3 or score.get('n_votes_cast') != 3:
                        raise ValueError('New correctness requires three valid independent votes')
                    spans=score.get('evidence_spans')
                    if not isinstance(spans,list) or any(not isinstance(s,str) or (s and s not in answer) for s in spans):
                        raise ValueError('Invalid correctness evidence spans')
                corr = 0.0 if score['contradiction'] else statistics.mean(score['fact_scores'])/2
                judged = True
        evidence = evidence_presence(g,contexts[q]['chunks']) if g['answerable'] else {'status':'not_applicable'}
        cats=[];openbook=label.endswith('openbook')
        if g['answerable'] and evidence['status'] in ('some_gold_quotes_located','no_gold_quotes_located'):
            cats.append('RETRIEVAL_EVIDENCE_NOT_LOCATED_PROXY')
        if openbook and evidence['status']=='all_gold_quotes_located' and corr is not None and corr < 1:
            cats.append('GENERATION_FAILURE_WITH_QUOTED_EVIDENCE_AVAILABLE')
        if judged and not score['contradiction'] and max(score['fact_scores'])>0 and min(score['fact_scores'])<2:
            cats.append('INCOMPLETE_ANSWER')
        if judged and score['contradiction']:cats.append('CONTRADICTION')
        if g['answerable'] and abst:cats.append('OVER_REFUSAL')
        if not g['answerable'] and not abst:cats.append('UNDER_REFUSAL_BY_READING_B')
        if (openbook and not abst and citations['n_total']==0) or citations['n_supported']<citations['n_total']:
            cats.append('CITATION_FAILURE')
        if not abst and citations['n_supported']==0:cats.append('FORMAT_FAILURE_RESOLVED')
        records.append({'qid':q,'line':g['line'],'answerable':g['answerable'],'abst':abst,
            'n_sup':citations['n_supported'],'n_tot':citations['n_total'],'corr_item':corr,
            'judged':judged,'parse_ok':abst or citations['n_total']>0,
            'resolved_format_ok':abst or citations['n_supported']>0,'evidence':evidence,
            'categories':cats,'faithfulness':'not_assessed'})
    return records


def aggregate(records):
    if not records:raise ValueError('Empty analysis population')
    metrics = JUDGE._row_metrics(records)
    missing = sum(r['answerable'] and r['corr_item'] is None for r in records)
    if missing:
        metrics.pop('correctness');metrics.pop('composite')
    evidence={}
    for status in ['all_gold_quotes_located','some_gold_quotes_located','no_gold_quotes_located','unassessable']:
        subset=[r for r in records if r['answerable'] and r['evidence']['status']==status]
        scored=[r['corr_item'] for r in subset if r['corr_item'] is not None]
        evidence[status]={'n':len(subset),'correctness':statistics.mean(scored) if len(scored)==len(subset) and scored else None,
            'unjudged':len(subset)-len(scored),'over_refusals':sum(r['abst'] for r in subset),
            'over_refusal_rate':sum(r['abst'] for r in subset)/len(subset) if subset else None}
    categories=dict(Counter(c for r in records for c in r['categories']))
    for name in ['RETRIEVAL_EVIDENCE_NOT_LOCATED_PROXY','GENERATION_FAILURE_WITH_QUOTED_EVIDENCE_AVAILABLE',
                 'INCOMPLETE_ANSWER','CONTRADICTION','OVER_REFUSAL','UNDER_REFUSAL_BY_READING_B',
                 'CITATION_FAILURE','FORMAT_FAILURE_RESOLVED']:
        categories.setdefault(name,0)
    categories['HALLUCINATION_FAITHFULNESS_FAILURE']=None
    for name in ['INCOMPLETE_ANSWER','CONTRADICTION','GENERATION_FAILURE_WITH_QUOTED_EVIDENCE_AVAILABLE']:
        if missing:categories[name]=None
    return dict(metrics,n=len(records),missing_correctness=missing,
        status='awaiting_judging' if missing else 'scored',
        parse_rate=sum(r['parse_ok'] for r in records)/len(records),
        resolved_format_rate=sum(r['resolved_format_ok'] for r in records)/len(records),
        categories=categories,evidence_decomposition=evidence,faithfulness_status='awaiting_judging')


def paired(left,right,seed=0,iterations=2000):
    a=core.unique(left,'qid');b=core.unique(right,'qid')
    if set(a)!=set(b):raise ValueError('Paired populations differ')
    out={}; cases=[]
    for metric in ['correctness','citation','abstention']:
        counts=Counter();eligible=0;pending=0
        for q in sorted(a):
            x,y=a[q],b[q]
            if metric=='correctness':
                if not x['answerable']:continue
                if x['corr_item'] is None or y['corr_item'] is None:
                    pending+=1;continue
                v,w=x['corr_item'],y['corr_item']
            elif metric=='citation':
                v=x['n_sup']/x['n_tot'] if x['n_tot'] else 0
                w=y['n_sup']/y['n_tot'] if y['n_tot'] else 0
            else:
                v=x['abst'] != x['answerable']; w=y['abst'] != y['answerable']
            category='improves' if w>v else 'worsens' if w<v else 'unchanged'
            counts[category]+=1;eligible+=1
            cases.append({'qid':q,'metric':metric,'change':category,'before':v,'after':w})
        out[metric]={'n':eligible,'pending':pending,'counts':dict(counts),
            'proportions':{k:counts[k]/eligible if eligible else None for k in ['improves','unchanged','worsens']}}
    complete=all(not r['answerable'] or r['corr_item'] is not None for r in left+right)
    if complete:
        rng=random.Random(seed);qids=sorted(a);boot={'correctness':[],'composite':[]}
        for _ in range(iterations):
            sample=[rng.choice(qids) for _ in qids]
            # Correctness is conditional on answerability; reject empty answerable resamples.
            if not any(a[q]['answerable'] for q in sample):continue
            ma=JUDGE._row_metrics([a[q] for q in sample]);mb=JUDGE._row_metrics([b[q] for q in sample])
            for metric in boot:boot[metric].append(mb[metric]-ma[metric])
        ma=JUDGE._row_metrics(left);mb=JUDGE._row_metrics(right)
        out['paired_bootstrap']={'seed':seed,'requested_resamples':iterations,'unit':'question',
            'metrics':{k:{'difference':mb[k]-ma[k],'ci95':[sorted(v)[int(.025*len(v))],sorted(v)[min(int(.975*len(v)),len(v)-1)]],
                          'resamples':len(v)} for k,v in boot.items() if v}}
    else:out['paired_bootstrap']={'status':'awaiting_judging'}
    return out,cases


def run(home,bundle,models,output):
    gold={g['id']:g for g in core.benchmark(home) if g['split']=='test'}
    bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    if manifest['split']!='test' or core.sha(bundle/'contexts.jsonl')!=manifest['contexts_sha256'] or core.sha(Path(home)/'data/gold/gold_v2.jsonl')!=manifest['gold_sha256']:
        raise ValueError('Frozen benchmark changed')
    contexts=core.unique(core.rows(bundle/'contexts.jsonl'),'qid')
    if set(contexts)!=set(gold):raise ValueError('Incomplete frozen evidence')
    specs=json.loads(Path(models).read_text());out=core.private_output(output)
    summary={'population':len(gold),'motor_population':sum(g['line']=='motor' for g in gold.values()),
        'evidence_method':'Exact normalized gold quote within a same-document/insurer frozen chunk; nonmatch is not proof of semantic absence.',
        'format_note':'Historical parse requires a parseable citation or reading-B refusal; resolved-format diagnostic additionally requires a resolved citation.',
        'abstention_note':'Inherited reading B may miss paraphrased refusals. Under-refusal flags need semantic review.',
        'models':{},'inputs':{'gold_sha256':manifest['gold_sha256'],'contexts_sha256':manifest['contexts_sha256']}}
    all_records={}
    for label,spec in specs.items():
        records=make_records(core.rows(spec['answers']),gold,contexts,
            core.rows(spec['scores']) if spec.get('scores') else [],label,spec.get('historical',False))
        all_records[label]=records;core.write_rows(out/(label+'.jsonl'),records)
        summary['models'][label]={'full':aggregate(records),'motor':aggregate([r for r in records if r['line']=='motor']),
            'answers_sha256':core.sha(spec['answers'])}
    for prefix in ['openbook','closedbook']:
        left='ministral3b-base-'+prefix;right='ministral3b-sft-'+prefix
        if left in all_records and right in all_records:
            summary['paired_'+prefix]={}
            for name in ['full','motor']:
                a=[r for r in all_records[left] if name=='full' or r['line']=='motor']
                b=[r for r in all_records[right] if name=='full' or r['line']=='motor']
                result,cases=paired(a,b);summary['paired_'+prefix][name]=result
                core.write_rows(out/('paired_'+prefix+'_'+name+'.jsonl'),cases)
    core.write_json(out/'aggregate.json',summary)
    print(json.dumps({'population':summary['population'],'motor_population':summary['motor_population'],
        'status':{k:v['full']['status'] for k,v in summary['models'].items()}}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['home','bundle','models','output']:p.add_argument('--'+n,required=True)
    run(**vars(p.parse_args()))
