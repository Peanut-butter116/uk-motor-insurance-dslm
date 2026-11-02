"""Unchanged deterministic definitions and selected semantic votes; no agreement calls."""
import collections
import csv
import json
from pathlib import Path
import statistics
from cloud.single_judge_fast.protocol import PROTOCOL,select,rows


def summarise(root,votes):
    from experiments import core
    root=Path(root);out=root/'results_single_judge_fast_v1';out.mkdir(exist_ok=True)
    ab=core.load_module('_fast_abstention','eval/abstention.py')
    gold={g['id']:g for g in rows(root/'evaluation/gold.jsonl')}
    contexts={r['qid']:r for r in rows(root/'evaluation/contexts.jsonl')}
    cfg=json.loads((root/'config.json').read_text());chosen=select(votes);results=[]
    mean=lambda xs:statistics.mean(x for x in xs if x is not None) if any(x is not None for x in xs) else None
    for key,spec in cfg['models'].items():
        for condition in ('base','base-rag','sft','sft-rag'):
            path=root/'runs'/key/(('sft' if condition.startswith('sft') else 'base')+'-responses.jsonl')
            responses={r['qid']:r for r in rows(path) if r['condition']==condition}
            parsed=resolved=ncite=answerable_refused=unanswerable_refused=covered=0
            correctness=[];support=[];citation=[];assign=collections.Counter();rag=condition.endswith('-rag')
            for q,g in gold.items():
                r=responses.get(q)
                if r is None:continue
                answer=r['answer'];refused=ab.is_abstention_assertion(answer,abstain_sentence=core.common.ABSTAIN_SENTENCE,citation_re=core.common.CITATION_RE)
                check=core.rag.verify_citations(answer,contexts[q]['chunks']);parsed+=int(refused or check['n_total']>0)
                if g['answerable']:answerable_refused+=int(refused)
                else:unanswerable_refused+=int(refused)
                if rag:resolved+=check['n_supported'];ncite+=check['n_total']
                ident=core.digest({'model':r['model'],'condition':condition,'qid':q})[:24]
                if ident in chosen:
                    provider,v=chosen[ident];covered+=1;assign[provider]+=1
                    correctness.append(v['correctness']);support.append(v['evidence_support'])
                    if rag:citation.extend(c['support'] for c in v.get('citation_assessments',[]))
            available=len(responses)==76;refusals=answerable_refused+unanswerable_refused
            row={'protocol':PROTOCOL,'model':spec['id'],'condition':condition,'available':available,
                 'benchmark_questions':76,'responses':len(responses),'missing_responses':76-len(responses),
                 'judged':covered,'missing_judgements':76-covered if available else None,
                 'groq_selected':assign['groq'],'gemini_selected':assign['gemini'],
                 'correctness':mean(correctness),'correctness_scored':sum(x is not None for x in correctness),
                 'reference_support':mean(support),'citation_support':mean(citation) if rag else None,
                 'parse_rate':parsed/76 if available else None,'citation_resolution':resolved/ncite if rag and ncite else None,
                 'citation_denominator':ncite if rag else None,'answerable_denominator':67,'unanswerable_denominator':9,
                 'appropriate_refusals':unanswerable_refused if available else None,
                 'unnecessary_refusals':answerable_refused if available else None,
                 'false_refusal_rate':answerable_refused/67 if available else None,
                 'refusal_correctness':unanswerable_refused/refusals if available and refusals else None,
                 'balanced_abstention':.5*((67-answerable_refused)/67+unanswerable_refused/9) if available else None}
            results.append(row)
    (out/'comparison.json').write_text(json.dumps(results,indent=2))
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    selected={ident:{'provider':p,'request_sha256':v['request_sha256']} for ident,(p,v) in chosen.items()}
    (out/'selected_votes.json').write_text(json.dumps(selected,sort_keys=True))
    return {'protocol':PROTOCOL,'unique_judged':len(chosen),'remaining':912-len(chosen),
            'selected_provider_counts':dict(collections.Counter(p for p,v in chosen.values())),
            'methodological_limitation':'Mixed-provider single valid judgement per response, Groq-first where both historical votes exist. Descriptive scores are not equivalent to the historical dual-judge design; no agreement analysis performed.',
            'conditions':16,'deepseek':'unavailable','full_coverage':len(chosen)==912}
