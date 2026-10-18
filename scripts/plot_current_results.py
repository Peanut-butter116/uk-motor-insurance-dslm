"""Generate PNG/SVG research figures from public verified aggregates only."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/2026-10-20'
RED='#C8102E';BLACK='#171717';GREY='#777777';PALE='#E8E8E8'


def make(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sources={name:EXP/name for name in ['results/summary.json','results/error_taxonomy.json',
        'manifests/interim_dataset_manifest.json','manifests/display_csv_manifest.json','configs/six_models.json',
        'reports/kaggle_status.json','results/six_model_v1/six_model_verification.json']}
    data={k:json.loads(p.read_text()) for k,p in sources.items()}
    data['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()}
    data['plotting_runtime']={'matplotlib':matplotlib.__version__,'numpy':np.__version__}
    (output/'figure_data.json').write_text(json.dumps(data,indent=2)+'\n')
    rows=[r for r in data['results/summary.json'] if r['population']=='motor']
    if len(rows)!=8 or any(r['n']!=76 or r['correctness'] is not None for r in rows):
        raise ValueError('Expected eight verified historical motor arms with correctness pending')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':BLACK,
        'axes.labelcolor':BLACK,'axes.edgecolor':BLACK,'xtick.color':BLACK,'ytick.color':BLACK,
        'svg.fonttype':'none','svg.hashsalt':'motor-research-20261020','savefig.facecolor':'white'})
    conditions=['Base','Base + RAG','SFT','SFT + RAG']
    by={(r['model'],r['condition']):r for r in rows}
    models=['qwen3b','ministral3b'];names=['Qwen2.5-3B','Ministral-3B']
    common='Historical matched 21-example SFT experiment; motor test n=76 (67 answerable, 9 unanswerable).'

    def polish(ax,ylabel=None,percent=False):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        if ylabel:ax.set_ylabel(ylabel)
        if percent:ax.set_ylim(0,110)
        else:ax.set_ylim(bottom=0)

    def save(fig,stem,title,foot):
        fig.suptitle(title,x=.07,y=.96,ha='left',fontsize=21,fontweight='bold')
        fig.text(.07,.035,foot,fontsize=11,va='bottom',wrap=True)
        fig.subplots_adjust(left=.09,right=.96,bottom=.22,top=.82,wspace=.3)
        for ext in ['png','svg']:fig.savefig(output/(stem+'.'+ext),dpi=300)
        plt.close(fig)

    # Six candidate statuses are a table, never invented performance bars.
    fig,axes=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1,1.4]})
    ax=axes[0];x=np.arange(2);width=.33
    for j,(cond,color) in enumerate([('Base',GREY),('Base + RAG',RED)]):
        vals=[100*by[(m,cond)]['parse_rate'] for m in models]
        bars=ax.bar(x+(j-.5)*width,vals,width,color=color,label=cond)
        ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=12)
    ax.set_xticks(x,names);polish(ax,'Inherited parse rate (%)',True);ax.legend(loc='upper left',frameon=False,fontsize=11)
    axes[1].axis('off')
    candidates=data['configs/six_models.json']['models']
    status=next(j['status'] for j in data['reports/kaggle_status.json']['jobs']
        if j['notebook'].endswith('motor-six-model-frozen-test-20261020'))
    verified={r['model']:r for r in data['results/six_model_v1/six_model_verification.json']['models']}
    values=[[m['id'].split('/')[-1],str(m['parameters_billion_nominal'])+'B',
        ('Inference complete;\njudging pending' if verified[m['id']]['status']=='inference_complete' else
         'Gated access failed' if m['gated'] else 'Compute dtype failed') if status=='COMPLETE' else
        'Access gated / '+status.lower() if m['gated'] else status.capitalize()+'; no verified outputs'] for m in candidates]
    tb=axes[1].table(cellText=values,colLabels=['New candidate','Size','Quality evaluation'],
        colWidths=[.55,.12,.38],cellLoc='left',loc='center')
    tb.auto_set_font_size(False);tb.set_fontsize(10);tb.scale(1,2.2)
    for (row,col),cell in tb.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        if row==0:cell.set_text_props(color=RED,weight='bold');cell.set_facecolor('#F3F3F3')
    save(fig,'01_base_models','Base-model evidence and six-candidate status',
        'Left: historical 3B base models, motor n=76. Right: three new candidates completed verified paired inference.\nCorrectness and claim support remain unjudged. Parse rate counts a citation parse or recognised abstention.')

    fig,axes=plt.subplots(1,2,figsize=(14,8))
    for ax,metric,title in zip(axes,['balanced_abstention','parse_rate'],['Balanced abstention','Inherited parse rate']):
        for j,(cond,color) in enumerate([('Base',GREY),('Base + RAG',RED)]):
            vals=[100*by[(m,cond)][metric] for m in models]
            bars=ax.bar(np.arange(2)+(j-.5)*.32,vals,.32,color=color,label=cond)
            ax.bar_label(bars,fmt='%.1f',padding=3)
        ax.set_xticks(np.arange(2),names);ax.set_title(title);polish(ax,'Score (%)',True)
    axes[0].legend(frameon=False)
    save(fig,'02_effect_of_rag','Effect of fixed RAG on historical base models',
        'Same 76 motor questions and frozen evidence. These are behaviour metrics, not answer correctness.\nNo-RAG citation quality is N/A. Latency and quality are not inferred from source resolution.')

    fig,axes=plt.subplots(1,3,figsize=(16,8))
    for ax,metric,title in zip(axes,['balanced_abstention','parse_rate','citation_resolution'],
        ['Balanced abstention','Inherited parse rate','Citation resolution']):
        for j,(m,name,color) in enumerate(zip(models,names,[BLACK,RED])):
            vals=[np.nan if by[(m,c)][metric] is None else 100*by[(m,c)][metric] for c in conditions]
            ax.plot(range(4),vals,'o-',color=color,label=name,linewidth=2)
            for i,v in enumerate(vals):
                if np.isfinite(v):
                    other=by[(models[1-j],conditions[i])][metric]
                    other=np.nan if other is None else 100*other
                    above=not np.isfinite(other) or (v>=other if j==0 else v>other)
                    ax.annotate(f'{v:.1f}',(i,v),xytext=(0,10 if above else -20),
                        textcoords='offset points',ha='center',fontsize=11,color=color,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
        ax.set_xticks(range(4),['Base','Base\n+ RAG','SFT','SFT\n+ RAG']);ax.set_title(title);polish(ax,'Score (%)',True)
        if metric=='citation_resolution':ax.text(.03,.08,'No-RAG citation: N/A',transform=ax.transAxes,fontsize=11)
    axes[0].legend(frameon=False,loc='lower left')
    save(fig,'03_historical_sft_rag','Historical four-arm SFT and RAG comparison',
        common+'\nQwen and Ministral share each model’s own checkpoint between base and SFT. No new interim-data SFT result.')

    fig,ax=plt.subplots(figsize=(14,8))
    labels=[name+' / '+c for m,name in zip(models,names) for c in conditions]
    vals=[by[(m,c)]['generation_seconds']/76 for m in models for c in conditions]
    bars=ax.barh(range(8),vals,color=[GREY,RED,BLACK,'#E78290']*2)
    ax.bar_label(bars,fmt='%.1f s',padding=5);ax.set_yticks(range(8),labels);ax.invert_yaxis()
    ax.set_xlim(0,max(vals)*1.22);ax.set_xlabel('Mean generation time per question (seconds)')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.28)
    fig.suptitle('Historical inference efficiency',x=.07,y=.96,ha='left',fontsize=21,fontweight='bold')
    fig.text(.07,.035,'All rows: historical 3B-class models, Tesla T4, motor n=76. Setup time excluded.\nHistorical peak GPU memory was not recorded. New-model timing is separate: see verified six_model_v1 tables.',fontsize=11)
    fig.subplots_adjust(left=.28,right=.91,bottom=.20,top=.83)
    for ext in ['png','svg']:fig.savefig(output/('04_model_efficiency.'+ext),dpi=300)
    plt.close(fig)

    manifest=data['manifests/interim_dataset_manifest.json']
    fig,axes=plt.subplots(1,3,figsize=(15,8))
    for ax,values,title in zip(axes,[manifest['origins'],manifest['answerability'],
        {'Insurers':len(manifest['insurers']),'Documents':manifest['source_documents'],'Evidence sections':manifest['evidence_sections']}],
        ['Data origin','Answerability','Source coverage']):
        bars=ax.bar(range(len(values)),list(values.values()),color=[BLACK,RED,GREY][:len(values)])
        ax.bar_label(bars,padding=4);ax.set_xticks(range(len(values)),[k.replace('unanswerable','Insufficient\nevidence').replace('answerable','Answerable').replace('generated','Generated').replace('legacy','Inherited').replace('Evidence sections','Evidence\nsections') for k in values]);ax.set_title(title);polish(ax,'Count');ax.set_ylim(0,max(values.values())*1.2)
    save(fig,'05_dataset_composition','Interim dataset: 409 accepted records, 384 distinct questions',
        'Verified snapshot 18 October 2026; 2,000 committed / 3,000 stretch target, freeze 18 November.\nHuman review: 0. training_ready=false. 25 deliberate answerability/context pairs need review. No growth trend inferred from one snapshot.')

    fig,axes=plt.subplots(1,3,figsize=(16,9))
    rag=[by[(m,c)] for m in models for c in ['Base + RAG','SFT + RAG']]
    labs=['Qwen Base + RAG','Qwen SFT + RAG','Ministral Base + RAG','Ministral SFT + RAG']
    tax={(r['condition'],r['category']):r['motor_question_count'] for r in data['results/error_taxonomy.json']['counts']}
    values=[[r['over_refusals']/r['answerable_n']*100 for r in rag],
        [(r['unanswerable_n']-r['correct_refusals'])/r['unanswerable_n']*100 for r in rag],
        [tax.get((r['model']+('-sft-openbook' if r['sft_examples'] else '-base-openbook'),'CITATION_FAILURE'),0)/76*100 for r in rag]]
    for ax,vals,title in zip(axes,values,['Unnecessary refusal\n(% of 67 answerable)','Missed refusal\n(% of 9 unanswerable)','Citation failures\n(% of 76 questions)']):
        bars=ax.bar(range(4),vals,color=[GREY,RED,GREY,RED]);ax.bar_label(bars,fmt='%.1f',padding=4)
        ax.set_xticks(range(4),labs,rotation=32,ha='right',fontsize=10);ax.set_title(title);polish(ax,'Rate (%)',True)
    save(fig,'06_error_analysis','Measured motor refusal and citation failures',
        'Historical RAG arms only. Citation failure includes missing or unresolved citations, not semantic contradiction.\nIncorrect-answer and unsupported-claim rates remain unjudged. Categories overlap and must not be summed.')
    files=sorted(p.name for p in output.iterdir() if p.suffix in ['.png','.svg'])
    lines=['# Current research figures','', 'Generated from verified public aggregates. White/black/red style, 300 dpi PNG and SVG with editable text. No private inputs or missing performance bars.', '',
        'These figures contain historical 21-example SFT results and current six-model execution status. They do not claim a completed six-model quality benchmark.', '',
        '| Figure | PNG | SVG |','|---|---|---|']
    for stem in sorted({Path(n).stem for n in files}):lines.append(f'| {stem} | [PNG]({stem}.png) | [SVG]({stem}.svg) |')
    lines+=['','Exact plotted inputs and source checksums: [figure_data.json](figure_data.json).',
        '', 'Regenerate from repository root: `MPLCONFIGDIR=/tmp/motor-mpl python scripts/plot_current_results.py`.',
        '', 'All performance populations are the same 76 motor questions. Correctness and semantic support remain missing, never zero. Efficiency is descriptive and GPU memory is unavailable. Historical and new checkpoint families must not be pooled.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('Generated',len(files),'PNG/SVG files from verified aggregates')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=EXP/'figures');make(p.parse_args().output)
