#!/usr/bin/env python3
"""Presentation figures from the validated canonical aggregate table; no APIs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_d1000_results import validate,NAMES,CONDITIONS,MODEL_ORDER

COLORS={'base':'#0072B2','base-rag':'#009E73','sft':'#D89000','sft-rag':'#B4619A'}
MARKERS={'base':'o','base-rag':'s','sft':'D','sft-rag':'^'}


def generate(source,out):
    payload=validate(json.loads(Path(source).read_text()));out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=payload['provenance'];rows=[r for k in MODEL_ORDER for r in payload['comparison'] if r['model_key']==k and r['available']]
    lookup={(r['model_key'],r['condition']):r for r in rows};models=MODEL_ORDER[:3];conditions=list(CONDITIONS)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.titlesize':19,'axes.labelsize':15,
                         'xtick.labelsize':14,'ytick.labelsize':14,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','svg.hashsalt':'d1000-'+str(p['checkpoint']),
                         'pdf.fonttype':42,'savefig.facecolor':'white','axes.titleweight':'bold'})
    paths=[]
    def canvas(title,subtitle,ncols=1):
        fig,axes=plt.subplots(1,ncols,figsize=(16,9),squeeze=False)
        fig.subplots_adjust(left=.10,right=.97,bottom=.16,top=.78,wspace=.22)
        fig.text(.07,.94,title,fontsize=27,weight='bold',color='#142D4E')
        fig.text(.07,.885,subtitle,fontsize=15,color='#455468')
        fig.text(.07,.045,f'PROVISIONAL · private checkpoint v{p["checkpoint"]} · {p["unique_judged"]}/912 judged · mixed-provider single-judge · DeepSeek N/A',fontsize=12,color='#455468')
        return fig,list(axes[0])
    def save(fig,name):
        timestamp=datetime.fromtimestamp(p['verified_epoch'],timezone.utc)
        for ext in ('png','svg','pdf'):
            metadata=({'Date':timestamp.isoformat()} if ext=='svg' else
                      {'CreationDate':timestamp,'ModDate':timestamp} if ext=='pdf' else
                      {'Software':'D1000 reproducible aggregate plotting'})
            path=out/f'{name}.{ext}';fig.savefig(path,dpi=300,metadata=metadata);paths.append(path)
        plt.close(fig)
    legend=[Patch(facecolor=COLORS[c],label=CONDITIONS[c]) for c in conditions]

    fig,(ax,)=canvas('RAG improves observed correctness; SFT gains are mixed',
        'Mean semantic rubric score · 0–100% · labels show scored questions / 76 · whiskers: conditional 95% bootstrap intervals')
    x=np.arange(3);width=.18
    for j,c in enumerate(conditions):
        rr=[lookup[m,c] for m in models];ys=np.array([r['correctness']*100 for r in rr]);pos=x+(j-1.5)*width
        lows=np.array([r['correctness_ci_low']*100 for r in rr]);highs=np.array([r['correctness_ci_high']*100 for r in rr])
        bars=ax.bar(pos,ys,width,color=COLORS[c],edgecolor='white',label=CONDITIONS[c])
        ax.errorbar(pos,ys,yerr=[ys-lows,highs-ys],fmt='none',ecolor='#263746',capsize=4,lw=1.2)
        for r,b,y,hi in zip(rr,bars,ys,highs):
            if r['judged']<76:b.set_hatch('//')
            ax.text(b.get_x()+b.get_width()/2,hi+2.2,f'{y:.1f}%\n{r["correctness_denominator"]}/76',ha='center',va='bottom',fontsize=12)
    ax.set_xticks(x,[NAMES[m] for m in models]);ax.set_ylim(0,100);ax.yaxis.set_major_formatter(PercentFormatter(100));ax.set_ylabel('Observed semantic correctness');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=legend+[Patch(facecolor='white',edgecolor='#555555',hatch='//',label='Incomplete semantic coverage')],loc='upper center',bbox_to_anchor=(.52,.845),ncol=5,fontsize=13,frameon=False)
    fig.text(.10,.10,'Intervals condition on observed judged questions; they do not include judge error, missingness bias or model-training uncertainty.',fontsize=12,color='#455468')
    save(fig,'01_semantic_correctness')

    labels=[];pairs=[]
    for m in models:
        for before,after in [('base','sft'),('base-rag','sft-rag')]:
            labels.append(NAMES[m]+(' · no RAG' if before=='base' else ' · RAG'))
            pairs.append((lookup[m,before],lookup[m,after]))
    fig,axes=canvas('Fine-tuning is not a uniform improvement',
        'Circle = Base · diamond = SFT · connectors compare the same model and evidence setting · signed change in percentage points',2)
    fig.subplots_adjust(left=.23,right=.97,wspace=.22)
    for ax,metric,title in zip(axes,['correctness','balanced_abstention'],['Semantic correctness','Balanced abstention']):
        for y,(a,b) in enumerate(pairs):
            va=a[metric]*100;vb=b[metric]*100;color=COLORS[a['condition']]
            ax.plot([va,vb],[y,y],color='#708096',lw=3,zorder=1)
            ax.scatter(va,y,s=100,marker='o',color=color,zorder=3);ax.scatter(vb,y,s=110,marker='D',color=COLORS[b['condition']],zorder=3)
            ax.text(va,y-.15,f'{va:.1f}',ha='center',va='bottom',fontsize=12)
            ax.text(vb,y+.16,f'{vb:.1f}',ha='center',va='top',fontsize=12)
            ax.text(108,y,f'{vb-va:+.1f} pp',va='center',ha='right',fontsize=13,weight='bold')
        ax.set_yticks(range(6),labels if ax is axes[0] else ['']*6);ax.invert_yaxis();ax.set_ylim(5.65,-.65);ax.set_xlim(-3,112)
        ax.set_xticks([0,25,50,75,100]);ax.xaxis.set_major_formatter(PercentFormatter(100));ax.set_title(title);ax.grid(axis='x',alpha=.18)
    fig.text(.10,.10,'Correctness endpoints use unequal observed subsets. Companion sft_effects.json reports paired common-question changes and uncertainty.',fontsize=12,color='#455468')
    save(fig,'02_sft_effect')

    fig,axes=canvas('Refusal sensitivity and false refusals tell different stories',
        'All 912 saved responses evaluated deterministically · 67 answerable and 9 unanswerable questions per condition',3)
    fig.subplots_adjust(left=.23,right=.97,wspace=.25)
    labels=[NAMES[r['model_key']]+' · '+CONDITIONS[r['condition']] for r in rows]
    for ax,metric,title in zip(axes,['balanced_abstention','appropriate_refusal_rate','false_refusal_rate'],
                              ['Balanced abstention ↑','Unanswerable refusal ↑','False refusal ↓']):
        for y,r in enumerate(rows):
            v=r[metric]*100;ax.scatter(v,y,s=90,color=COLORS[r['condition']],marker=MARKERS[r['condition']])
            count='' if metric=='balanced_abstention' else f' ({int(r[metric+"_numerator"])}/{r[metric+"_denominator"]})'
            ax.text(105,y,f'{v:.1f}%'+count,ha='left',va='center',fontsize=11.5)
        ax.set_yticks(range(12),labels if ax is axes[0] else ['']*12);ax.set_ylim(11.6,-.6);ax.set_xlim(-5,163);ax.set_xticks([0,50,100]);ax.xaxis.set_major_formatter(PercentFormatter(100));ax.grid(axis='x',alpha=.18);ax.set_title(title,fontsize=17)
    fig.text(.10,.10,'Balanced score = ½(answerable non-refusal / 67 + unanswerable refusal / 9). Refusal recognition follows the frozen original rules.',fontsize=12,color='#455468')
    save(fig,'03_refusal_abstention')

    rag=[r for r in rows if r['condition'].endswith('-rag')]
    fig,axes=canvas('A resolving citation is not proof of supporting evidence',
        'RAG conditions only · resolution uses all generated citation IDs; support uses valid scored citation occurrences',2)
    fig.subplots_adjust(left=.23,right=.96,wspace=.25)
    for ax,metric,title in zip(axes,['citation_resolution','citation_support'],['Citation-ID resolution','Semantic citation support']):
        for y,r in enumerate(rag):
            v=r[metric]
            if v is not None:
                ax.barh(y,100*v,height=.5,color=COLORS[r['condition']],alpha=.95)
                suffix=f'{int(r[metric+"_numerator"])}/{r[metric+"_denominator"]}' if metric=='citation_resolution' else f'n={r[metric+"_denominator"]}'
                ax.text(104,y,f'{100*v:.1f}% · {suffix}',va='center',fontsize=12)
            else:ax.text(4,y,'N/A — no scored citation support',va='center',fontsize=12,color='#6A6A6A')
        ax.set_yticks(range(6),[NAMES[r['model_key']]+' · '+CONDITIONS[r['condition']] for r in rag] if ax is axes[0] else ['']*6)
        ax.set_ylim(5.6,-.6);ax.set_xlim(0,165);ax.set_xticks([0,25,50,75,100]);ax.xaxis.set_major_formatter(PercentFormatter(100));ax.grid(axis='x',alpha=.18);ax.set_title(title)
    fig.text(.10,.10,'SmolLM Base+RAG resolves 1/1 citation ID, with zero scored support occurrences. Its 100% resolution is not evidence of reliable grounding.',fontsize=12,color='#455468')
    save(fig,'04_citation_reliability')

    fig,(ax,)=canvas('Coverage is part of the result',
        f'{p["unique_judged"]} valid · {p["eligible_remaining"]} other missing · {p["quarantined_unjudged"]} quarantined · 76 questions per available model/condition')
    fig.subplots_adjust(left=.25,bottom=.21)
    for y,r in enumerate(rows):
        valid=r['judged'];missing=r['eligible_unjudged'];quarantine=r['quarantined']
        ax.barh(y,valid,color='#007F86',height=.65)
        ax.barh(y,missing,left=valid,color='#DDE2E7',edgecolor='#84909B',height=.65,hatch='//')
        ax.barh(y,quarantine,left=valid+missing,color='#D55E00',height=.65)
        ax.text(valid/2,y,str(valid),ha='center',va='center',color='white',fontsize=13,weight='bold')
        if missing:ax.text(valid+missing/2,y,str(missing),ha='center',va='center',fontsize=12)
        if quarantine:ax.text(77,y,f'{quarantine} quarantined',va='center',fontsize=11,color='#AD4400')
    ax.set_yticks(range(12),labels);ax.set_ylim(11.7,-.7);ax.set_xlim(0,89);ax.set_xticks([0,19,38,57,76]);ax.set_xlabel('Number of saved responses');ax.grid(axis='x',alpha=.12)
    fig.legend(handles=[Patch(facecolor='#007F86',label='Valid judgement'),Patch(facecolor='#DDE2E7',edgecolor='#84909B',hatch='//',label='Other missing'),Patch(facecolor='#D55E00',label='Quarantined')],loc='upper center',bbox_to_anchor=(.6,.84),ncol=3,frameon=False)
    fig.text(.10,.10,'Four protected requests affect three unjudged responses; one already has a valid historical vote. Missing semantic scores are not zeros.',fontsize=12,color='#455468')
    save(fig,'05_coverage')

    metrics=['correctness','balanced_abstention','citation_resolution','citation_support','parse_rate','coverage']
    matrix=np.array([[r[m]*100 if r[m] is not None else np.nan for m in metrics] for r in rows])
    fig,(ax,)=canvas('D1000 results at a glance',
        'Percentages · observed semantic means and deterministic outcomes are distinct · grey = not applicable or no scored support')
    fig.subplots_adjust(left=.25,right=.89,bottom=.18)
    cmap=plt.colormaps['viridis'].copy();cmap.set_bad('#E6E8EB')
    im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=100,cmap=cmap)
    ax.set_yticks(range(12),labels);ax.set_xticks(range(6),['Semantic\ncorrectness','Balanced\nabstention','Citation-ID\nresolution','Citation\nsupport','Inherited\nparse','Judgement\ncoverage'],fontsize=13)
    for y,r in enumerate(rows):
        for x,m in enumerate(metrics):
            v=matrix[y,x];text='N/A' if np.isnan(v) else f'{v:.1f}'
            if m=='correctness':text+='\n'+str(r['correctness_denominator'])+'/76'
            ax.text(x,y,text,ha='center',va='center',fontsize=11.5,color='#172B41' if np.isnan(v) or v>55 else 'white')
    fig.colorbar(im,ax=ax,fraction=.035,pad=.03,label='Percent')
    fig.text(.10,.10,'Scores are descriptive and provisional. Inherited parse = citation/refusal recognition; it does not measure structured JSON compliance.',fontsize=12,color='#455468')
    save(fig,'06_results_heatmap')
    manifest={'checkpoint':p['checkpoint'],'canonical_input_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'size_inches':[16,9],'png_dpi':300,
              'matplotlib':matplotlib.__version__,'numpy':np.__version__,
              'files':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='reports/d1000_final/metrics_summary.json');ap.add_argument('--output',default='reports/d1000_final/figures')
    args=ap.parse_args();result=generate(args.input,args.output);print(json.dumps({'checkpoint':result['checkpoint'],'files':len(result['files'])}))
