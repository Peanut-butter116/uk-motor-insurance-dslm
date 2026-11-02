"""Established deterministic metrics plus separate, paired independent judge labels."""
import collections
import csv
import json
from pathlib import Path
import statistics
from experiments import core


def agreement(a,b):
    pairs=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    if not pairs:return {'paired':0,'agreement':None,'kappa':None,'disagreements':0}
    n=len(pairs);observed=sum(x==y for x,y in pairs)/n
    ac=collections.Counter(x for x,y in pairs);bc=collections.Counter(y for x,y in pairs)
    chance=sum(ac[k]*bc[k] for k in ac.keys()|bc.keys())/n**2
    return {'paired':n,'agreement':observed,'kappa':(observed-chance)/(1-chance) if chance<1 else None,
            'disagreements':sum(x!=y for x,y in pairs)}


def summarise(root):
    from cloud.d1000.judging import tasks
    root=Path(root);out=root/'results';out.mkdir(exist_ok=True)
    ab=core.load_module('_d1000_metric_abstention','eval/abstention.py')
    gold={g['id']:g for g in core.rows(root/'evaluation/gold.jsonl')};assert len(gold)==76
    contexts={r['qid']:r for r in core.rows(root/'evaluation/contexts.jsonl')}
    cfg=json.loads((root/'config.json').read_text())
    votes={p:{v['evaluation_id']:v for v in core.rows(root/'judging'/f'{p}.jsonl')} if (root/'judging'/f'{p}.jsonl').exists() else {} for p in ('groq','gemini')}
    disagreements=[];results=[]
    for key,spec in cfg['models'].items():
        for condition in ('base','base-rag','sft','sft-rag'):
            path=root/'runs'/key/(('sft' if condition.startswith('sft') else 'base')+'-responses.jsonl')
            responses={r['qid']:r for r in core.rows(path) if r['condition']==condition} if path.exists() else {}
            assert len(responses)==len(set(responses))
            rag=condition.endswith('-rag');values={p:[] for p in votes};support={p:[] for p in votes};citation={p:[] for p in votes}
            nparse=0;abstain=collections.Counter();resolved=ncite=0;latencies=[];failures=0
            for q,g in gold.items():
                r=responses.get(q)
                if r is None:continue
                answer=r['answer'];a=ab.is_abstention_assertion(answer,abstain_sentence=core.common.ABSTAIN_SENTENCE,citation_re=core.common.CITATION_RE)
                check=core.rag.verify_citations(answer,contexts[q]['chunks'])
                nparse+=int(a or check['n_total']>0);failures+=int(not r['final_answer_present'])
                abstain['answerable' if g['answerable'] else 'unanswerable']+=int(a)
                if rag:resolved+=check['n_supported'];ncite+=check['n_total']
                latencies.append(r['seconds'])
                eid=core.digest({'model':r['model'],'condition':condition,'qid':q})[:24]
                for p in votes:
                    v=votes[p].get(eid,{});values[p].append(v.get('correctness'));support[p].append(v.get('evidence_support'))
                    if rag:citation[p].extend(c['support'] for c in v.get('citation_assessments',[]))
                vs=[votes[p].get(eid) for p in votes]
                if all(vs) and any(vs[0].get(k)!=vs[1].get(k) for k in ('correctness','evidence_support','abstention_category')):
                    disagreements.append({'evaluation_id':eid,'model':key,'condition':condition,'qid':q,
                        'answerability':g['answerable'],'groq':{k:vs[0].get(k) for k in ('correctness','evidence_support','abstention_category')},
                        'gemini':{k:vs[1].get(k) for k in ('correctness','evidence_support','abstention_category')}})
            mean=lambda xs:statistics.mean(x for x in xs if x is not None) if any(x is not None for x in xs) else None
            agree=agreement(values['groq'],values['gemini'])
            row={'model':spec['id'],'revision':spec['revision'],'condition':condition,'training_set':'D1000-derived' if condition.startswith('sft') else None,
                'benchmark_questions':76,'responses':len(responses),'missing_responses':76-len(responses),
                'parsed_responses':nparse,'parse_rate':nparse/76,'final_answer_failures':failures,
                'answerable_denominator':67,'unanswerable_denominator':9,
                'unnecessary_refusals':abstain['answerable'],'appropriate_refusals':abstain['unanswerable'],
                'false_refusal_rate':abstain['answerable']/67 if len(responses)==76 else None,
                'deterministic_refusal_correctness':abstain['unanswerable']/(abstain['answerable']+abstain['unanswerable']) if abstain['answerable']+abstain['unanswerable'] else None,
                'balanced_abstention':.5*((67-abstain['answerable'])/67+abstain['unanswerable']/9) if len(responses)==76 else None,
                'citation_resolution':resolved/ncite if rag and ncite else None,'citation_denominator':ncite if rag else None,
                'groq_correctness':mean(values['groq']),'gemini_correctness':mean(values['gemini']),
                'groq_correctness_scored':sum(v is not None for v in values['groq']),
                'gemini_correctness_scored':sum(v is not None for v in values['gemini']),
                'groq_reference_support':mean(support['groq']),'gemini_reference_support':mean(support['gemini']),
                'groq_citation_support':mean(citation['groq']) if rag else None,
                'gemini_citation_support':mean(citation['gemini']) if rag else None,
                'correctness_agreement':agree['agreement'],'correctness_kappa':agree['kappa'],
                'judge_paired_denominator':agree['paired'],'inference_seconds':sum(latencies)}
            results.append(row)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    core.write_json(out/'results.json',results)
    core.write_json(out/'disagreements.json',disagreements)
    # Deterministic stratified human-review candidates, never fabricated reviews.
    groups=collections.defaultdict(list)
    for d in disagreements:groups[(d['model'],d['condition'],d['answerability'])].append(d)
    core.write_json(out/'human_review_sample.json',[sorted(v,key=lambda x:x['evaluation_id'])[0] for v in groups.values()])
    paired=sorted(set(votes['groq'])&set(votes['gemini']))
    agreement_report={k:agreement([votes['groq'][i].get(k) for i in paired],
                                [votes['gemini'][i].get(k) for i in paired])
                      for k in ('correctness','evidence_support','abstention_category')}
    agreement_report['overall']=agreement([
        tuple(votes['groq'][i].get(k) for k in ('correctness','evidence_support','abstention_category')) for i in paired],
        [tuple(votes['gemini'][i].get(k) for k in ('correctness','evidence_support','abstention_category')) for i in paired])
    agreement_report['interpretation']='LLM inter-judge consistency, not factual accuracy or human validation'
    core.write_json(out/'inter_judge_agreement.json',agreement_report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(cfg['models']);conditions=['base','base-rag','sft','sft-rag'];markers=['o','s','^','D']
    for metric in ('groq_correctness','gemini_correctness','balanced_abstention','citation_resolution','groq_citation_support','gemini_citation_support','parse_rate','correctness_agreement'):
        fig,ax=plt.subplots(figsize=(9,5))
        for cond,marker in zip(conditions,markers):
            ys=[next(r for r in results if r['model']==cfg['models'][k]['id'] and r['condition']==cond)[metric] for k in names]
            ys=[float('nan') if y is None else y for y in ys]
            ax.plot(names,ys,marker=marker,linestyle=['-','--',':','-.'][conditions.index(cond)],label=cond)
        ax.set_ylim(0,1.05);ax.set_ylabel(metric.replace('_',' '));ax.legend();ax.grid(alpha=.2)
        ax.set_title('D1000-derived descriptive comparison; missing scores remain missing')
        fig.tight_layout();fig.savefig(out/(metric+'.png'),dpi=180);fig.savefig(out/(metric+'.svg'));plt.close(fig)
    times=[]
    for k in names:
        p=root/'runs'/k/'full/training.json';times.append(json.loads(p.read_text())['seconds']/60 if p.exists() else float('nan'))
    fig,ax=plt.subplots(figsize=(8,4));bars=ax.bar(names,times)
    for b,h in zip(bars,['/','\\','x','.']):b.set_hatch(h)
    ax.set_ylabel('Training minutes');fig.tight_layout();fig.savefig(out/'training_time.png',dpi=180);plt.close(fig)
    return {'conditions':16,'responses':sum(r['responses'] for r in results),
            'judge_votes':{p:len(v) for p,v in votes.items()},'disagreements':len(disagreements),
            'human_reviews_completed':0,'primary_three_vote_results':False}
