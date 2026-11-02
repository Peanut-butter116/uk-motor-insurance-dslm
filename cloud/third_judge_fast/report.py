"""Private aggregate comparisons/figures, including all missingness explicitly."""
import json
from pathlib import Path
from cloud.third_judge_fast.protocol import PROTOCOL, MODEL


def generate(recovery):
    from cloud.third_judge_fast.metrics import summarise
    ctx=recovery.third
    if ctx.get('active',0):raise RuntimeError('Drain requests before figures')
    summary=summarise(ctx['root'],ctx['votes'])
    folder=ctx['root']/'results_third_judge_fast_v1'
    table=json.loads((folder/'comparison.json').read_text())
    assert len(table)==16 and all(r['benchmark_questions']==76 for r in table)
    assert sum(r['judged'] for r in table)==summary['unique_judged']
    available=[r for r in table if r['available']]
    assert len(available)==12
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=[r['model'].split('/')[-1]+'\n'+r['condition'] for r in available]
    fig,axes=plt.subplots(3,1,figsize=(14,12),sharex=True,layout='constrained')
    for ax,metric,title in zip(axes,('correctness','citation_support','judged'),
                               ('Semantic correctness (available valid votes)','Citation support (RAG only)','Semantic judgement coverage / 76')):
        values=[r[metric] if r[metric] is not None else np.nan for r in available]
        ax.bar(range(12),values,color='#247887');ax.set_title(title);ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,76 if metric=='judged' else 1)
    axes[-1].set_xticks(range(12),labels,rotation=45,ha='right')
    fig.suptitle(f'Mixed-provider single-judge D1000: {summary["unique_judged"]}/912 judged\n'
                 'Groq GPT-OSS-120B; Gemini 3.5 Flash-Lite; calibrated Gemini 3.1 Flash-Lite')
    for suffix in ('png','pdf'):fig.savefig(folder/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    limitations={'protocol':PROTOCOL,'third_judge':{'provider':'gemini','model':MODEL},
                  'coverage':summary['unique_judged'],'missing_semantic_judgements':summary['remaining'],
                  'protected_historical_requests':4,'protected_unjudged':ctx['state']['protected_unjudged'],
                  'new_unknown_requests':ctx.get('new_unknown',0),'deepseek':'four conditions unavailable',
                  'interpretation':'Mixed-provider mixed-model single-judge descriptive evaluation; not dual-judge agreement or equivalence. '
                      'Historical Groq-first ties retained. Missing semantic votes excluded from score means, not counted as zero. '
                      'Every available model/condition retains 76 benchmark questions; score coverage reported separately.'}
    (folder/'methodology.json').write_text(json.dumps(limitations,indent=2))
    return {**summary,'figures_created':2,'aggregate_comparison_rows':len(table)}
