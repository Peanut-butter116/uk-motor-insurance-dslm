"""Publish semantic figures only when measured scores exist; never draw placeholders."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def make(results, output):
    results, output = Path(results), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names=['correctness_summary.csv','grounding_summary.csv','sft_rag_comparison.csv','judge_validation_summary.json']
    data={'source_sha256':{n:hashlib.sha256((results/n).read_bytes()).hexdigest() for n in names}}
    for n in names:
        data[n]=json.loads((results/n).read_text()) if n.endswith('.json') else list(csv.DictReader((results/n).open()))
    (output/'figure_data.json').write_text(json.dumps(data,indent=2)+'\n')
    rows=data['correctness_summary.csv']
    validation=data['judge_validation_summary.json']
    if not any(r['mean_correctness'] for r in rows):
        (output/'README.md').write_text('# Semantic evaluation figures\n\nNo correctness, RAG correctness improvement, correctness-versus-support, unsupported-answer or SFT-improvement figures were generated: **zero real responses have valid semantic judgements**.\n\nExisting behaviour figures measure citation resolution, parsing and refusal only. They cannot answer whether SFT improves accuracy.\n\n`figure_data.json` records the exact pending aggregate inputs and hashes. Regenerate after authorised judging: `python scripts/plot_semantic_results.py --results experiments/2026-10-20/results/semantic_judge_v2 --output experiments/2026-10-20/figures/semantic_judge_v2`.\n')
        print('No measured semantic scores; skipped all five semantic figures')
        return
    # Refuse misleading partially observed population comparisons.
    historical=[r for r in rows if r['experiment']=='historical_sft21']
    if any(int(r['judged_n'])!=int(r['population_n']) for r in historical):
        raise ValueError('Semantic figures require complete matched populations; partial results remain tables')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':11,'text.color':'black',
                         'axes.labelcolor':'black','savefig.facecolor':'white'})
    gs={(r['experiment'],r['model'],r['condition']):r for r in data['grounding_summary.csv']}
    names={'qwen3b':'Qwen2.5-3B','ministral3b':'Ministral-3B'}
    def save(fig,stem,title,foot):
        fig.suptitle(title,fontweight='bold');fig.text(.05,.02,foot,fontsize=9)
        fig.tight_layout(rect=(0,.1,1,.94))
        for ext in ('png','svg'):fig.savefig(output/(stem+'.'+ext),dpi=300)
        plt.close(fig)
    foot='Historical 21-example SFT; motor n=76 per arm. Three repeated judge votes; human validation n='+str(validation['human_review_completed'])+'.'
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    conditions=['base-closedbook','base-openbook','sft-closedbook','sft-openbook']
    lookup={(r['model'],r['condition']):r for r in historical}
    for ax,model in zip(axes,names):
        values=[float(lookup[(model,model+'-'+c)]['mean_correctness']) for c in conditions]
        bars=ax.bar(range(4),values,color=['#777777','#222222','#C8102E','#E78290'])
        ax.bar_label(bars,fmt='%.3f');ax.set_ylim(0,1.12);ax.set_xticks(range(4),['Base','Base + RAG','SFT','SFT + RAG'],rotation=20);ax.set_title(names[model]);ax.set_ylabel('Mean correctness (0–1)')
        ax.spines[['top','right']].set_visible(False)
    save(fig,'01_correctness','Answer correctness by model and condition',foot)
    for stem,kind in [('02_rag_effect','rag'),('05_sft_effect','sft')]:
        comps=[r for r in data['sft_rag_comparison.csv'] if (('sft-' in r['after'] and 'base-' in r['before']) if kind=='sft' else ('closedbook' in r['before'] and 'openbook' in r['after']))]
        if any(not r['difference'] or not r['ci_low'] or not r['ci_high'] for r in comps):continue
        fig,ax=plt.subplots(figsize=(11,6))
        for i,r in enumerate(comps):
            x=float(r['difference']);lo=float(r['ci_low']);hi=float(r['ci_high'])
            ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color='#C8102E',capsize=4)
        ax.set_yticks(range(len(comps)),[names[r['model']]+' / '+r['before']+' → '+r['after'] for r in comps]);ax.axvline(0,color='black',linewidth=.7);ax.set_xlim(-1,1);ax.set_xlabel('Paired mean correctness difference (95% group bootstrap CI)')
        save(fig,stem,'Effect of '+kind.upper()+' on correctness',foot)
    fig,ax=plt.subplots(figsize=(9,6))
    for r in historical:
        g=gs[(r['experiment'],r['model'],r['condition'])]
        if g['mean_reference_support']:
            x=float(r['mean_correctness']);y=float(g['mean_reference_support']);ax.scatter(x,y,color='#C8102E');ax.annotate(r['condition'],(x,y),fontsize=8)
    ax.set_xlim(0,1.05);ax.set_ylim(0,1.05);ax.set_xlabel('Mean correctness');ax.set_ylabel('Mean reference evidence support');save(fig,'03_correctness_support','Correctness and reference support',foot+' Support excludes refusals; denominators are in figure_data.json.')
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    labels=[r['condition'] for r in historical]
    for ax,values,title in zip(axes,[[float(r['correct_refusal_rate']) for r in historical],[float(r['unnecessary_refusal_rate']) for r in historical],[float(gs[(r['experiment'],r['model'],r['condition'])]['unsupported_material_claim_rate']) for r in historical]],['Correct refusal / 9 unanswerable','Unnecessary refusal / 67 answerable','Unsupported claim / judged responses']):
        bars=ax.bar(range(len(values)),values,color='#C8102E');ax.bar_label(bars,fmt='%.2f');ax.set_ylim(0,1.1);ax.set_xticks(range(len(values)),labels,rotation=75,fontsize=8);ax.set_title(title,fontsize=10)
    save(fig,'04_abstention_unsupported','Abstention and unsupported answers',foot)
    (output/'README.md').write_text('# Semantic evaluation figures\n\nPNG (300 dpi) and editable SVG; exact aggregate inputs and hashes in figure_data.json. All scores remain synthetic judge labels until independent human validation.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    make(**vars(p.parse_args()))
