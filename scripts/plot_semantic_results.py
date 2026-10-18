"""Publish semantic figures only when measured scores exist; never draw placeholders."""
import argparse
import csv
import hashlib
import json
import math
import textwrap
from pathlib import Path


def make(results, output):
    results, output = Path(results), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names=['correctness_summary.csv','grounding_summary.csv','sft_rag_comparison.csv','judge_validation_summary.json']
    names += [n for n in ['matched_three_vote_summary.csv','matched_single_vote_summary.csv',
                          'matched_three_vote_comparison.csv','matched_single_vote_comparison.csv','matched_pilot_manifest.json'] if (results/n).exists()]
    data={'source_sha256':{n:hashlib.sha256((results/n).read_bytes()).hexdigest() for n in names}}
    for n in names:
        data[n]=json.loads((results/n).read_text()) if n.endswith('.json') else list(csv.DictReader((results/n).open()))
    (output/'figure_data.json').write_text(json.dumps(data,indent=2)+'\n')
    rows=data['correctness_summary.csv']
    validation=data['judge_validation_summary.json']
    pilot=None
    historical=[r for r in rows if r.get('experiment')=='historical_sft21']
    if any(int(r['judged_n'])!=int(r['population_n']) for r in historical):
        for candidate in ('three_vote','single_vote'):
            available=data.get(f'matched_{candidate}_summary.csv',[])
            if len(available)==8 and all(int(r['population_n'])>=5 for r in available):
                pilot=candidate;historical=available;break
    if pilot is None and (not any(r['mean_correctness'] for r in rows) or any(int(r['judged_n'])!=int(r['population_n']) for r in historical)):
        message='zero real responses have valid semantic judgements' if not any(r['mean_correctness'] for r in rows) else 'fewer than five common scored questions across all eight arms'
        (output/'README.md').write_text('# Semantic evaluation figures\n\nNo correctness, RAG correctness improvement, correctness-versus-support, unsupported-answer or SFT-improvement figures were generated: **'+message+'**.\n\nExisting behaviour figures measure citation resolution, parsing and refusal only. They cannot answer whether SFT improves accuracy.\n\n`figure_data.json` records the exact pending aggregate inputs and hashes. Regenerate after authorised judging: `python scripts/plot_semantic_results.py --results experiments/2026-10-20/results/semantic_judge_v2 --output experiments/2026-10-20/figures/semantic_judge_v2`.\n')
        print('Semantic figures skipped:',message)
        return
    # Refuse misleading partially observed population comparisons.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'motor-semantic-20261020','font.size':11,'text.color':'black',
                         'axes.labelcolor':'black','savefig.facecolor':'white'})
    gs={(r['experiment'],r['model'],r['condition']):r for r in data['grounding_summary.csv']}
    if pilot:gs={(r['experiment'],r['model'],r['condition']):r for r in historical}
    names={'qwen3b':'Qwen2.5-3B','ministral3b':'Ministral-3B'}
    def save(fig,stem,title,foot):
        fig.suptitle(title,fontweight='bold');fig.text(.05,.02,textwrap.fill(foot,150),fontsize=9)
        fig.tight_layout(rect=(0,.1,1,.94))
        for ext in ('png','svg'):fig.savefig(output/(stem+'.'+ext),dpi=300)
        plt.close(fig)
    n=historical[0]['population_n']
    foot=('PRELIMINARY matched complete-case pilot, n='+str(n)+' per arm; '+pilot.replace('_',' ')+' protocol. ' if pilot else 'Full motor benchmark n=76 per arm; three votes. ')
    foot+='Historical SFT21; human validation n='+str(validation['human_review_completed'])+'. Not a validated ranking.'
    foot+=' Answerable/unanswerable n='+str(historical[0].get('answerable_n',67))+'/'+str(historical[0].get('unanswerable_n',9))+'.'
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
        comps=[r for r in data[f'matched_{pilot}_comparison.csv' if pilot else 'sft_rag_comparison.csv'] if (('sft-' in r['after'] and 'base-' in r['before']) if kind=='sft' else ('closedbook' in r['before'] and 'openbook' in r['after']))]
        if any(not r['difference'] or not r['ci_low'] or not r['ci_high'] for r in comps):continue
        fig,ax=plt.subplots(figsize=(11,6))
        for i,r in enumerate(comps):
            x=float(r['difference']);lo=float(r['ci_low']);hi=float(r['ci_high'])
            ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color='#C8102E',capsize=4)
        ax.set_yticks(range(len(comps)),[names[r['model']]+' / '+r['before']+' → '+r['after'] for r in comps]);ax.set_ylim(-.6,len(comps)-.4);ax.axvline(0,color='black',linewidth=.7);ax.set_xlim(-1,1);ax.set_xlabel('Paired mean correctness difference (95% group bootstrap CI)')
        save(fig,stem,'Effect of '+kind.upper()+' on correctness',foot)
    fig,ax=plt.subplots(figsize=(12,7))
    legend=[]
    for i,r in enumerate(historical,1):
        g=gs[(r['experiment'],r['model'],r['condition'])]
        if g['mean_reference_support']:
            x=float(r['mean_correctness']);y=float(g['mean_reference_support']);ax.scatter(x,y,color='#C8102E');ax.annotate(str(i),(x,y),xytext=(5,5),textcoords='offset points',fontsize=9)
            legend.append(str(i)+': '+r['condition'])
    ax.text(1.03,.98,'\n'.join(legend),transform=ax.transAxes,va='top',fontsize=9)
    ax.set_xlim(0,1.05);ax.set_ylim(0,1.05);ax.set_xlabel('Mean correctness');ax.set_ylabel('Mean reference evidence support');save(fig,'03_correctness_support','Correctness and reference support',foot+' Support excludes refusals; denominators are in figure_data.json.')
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    labels=[r['condition'] for r in historical]
    for ax,values,title in zip(axes,[[float(r['correct_refusal_rate']) if r['correct_refusal_rate'] else float('nan') for r in historical],[float(r['unnecessary_refusal_rate']) if r['unnecessary_refusal_rate'] else float('nan') for r in historical],[float(gs[(r['experiment'],r['model'],r['condition'])]['unsupported_material_claim_rate']) if gs[(r['experiment'],r['model'],r['condition'])]['unsupported_material_claim_rate'] else float('nan') for r in historical]],['Correct refusal / '+str(historical[0].get('unanswerable_n',9))+' unanswerable','Unnecessary refusal / '+str(historical[0].get('answerable_n',67))+' answerable','Unsupported claim / judged responses']):
        bars=ax.bar(range(len(values)),values,color='#C8102E');ax.bar_label(bars,fmt='%.2f');ax.set_ylim(0,1.1);ax.set_xticks(range(len(values)),labels,rotation=75,fontsize=8);ax.set_title(title,fontsize=10)
        for i,value in enumerate(values):
            if not math.isfinite(value):ax.text(i,.03,'N/A',ha='center',fontsize=8)
    save(fig,'04_abstention_unsupported','Abstention and unsupported answers',foot)
    for svg in output.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (output/'README.md').write_text('# Semantic evaluation figures\n\n'+foot+'\n\nPNG (300 dpi) and editable SVG; exact aggregate inputs and hashes in figure_data.json. Empty bars denote N/A, never zero. All scores remain synthetic judge labels until independent human validation. The pilot uses the same observed questions across all eight arms; excludes missing/uncertain scores and does not estimate full-test performance.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    make(**vars(p.parse_args()))
