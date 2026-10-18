"""Regenerate seven public-only meeting figures (Matplotlib required).

Reads figures/aggregate_values.json only; never imports the experiment code or
opens private data. Run: python archive/2026-10-11/scripts/create_progress_figures.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
DATA = json.loads((OUT / 'aggregate_values.json').read_text())
NAVY = '#18324D'; TEAL = '#087F82'; BLUE = '#4776A4'; MUTED = '#526476'
PALE = '#EAF4F3'; LIGHT = '#F0F4F8'; AMBER = '#A46A13'; SAND = '#FFF3DD'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':17,
    'text.color':NAVY, 'axes.labelcolor':NAVY, 'axes.edgecolor':'#C5D0DB',
    'xtick.color':MUTED, 'ytick.color':NAVY, 'svg.fonttype':'none',
    'svg.hashsalt':'insurance-progress-20261011', 'savefig.facecolor':'white'})


def canvas(number, title, subtitle):
    fig = plt.figure(figsize=(16,9), facecolor='white')
    ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
    ax.text(.55,8.55,f'{number:02d}  /  UK INSURANCE DSLM',fontsize=12,color=TEAL,weight='bold')
    ax.text(.55,8.02,title,fontsize=29,weight='bold')
    ax.text(.55,7.55,subtitle,fontsize=16,color=MUTED)
    ax.text(.55,.23,'Tech Mahindra  •  Research progress  •  '+DATA['date'],fontsize=11,color=MUTED)
    ax.text(15.45,.23,'Public aggregates only  |  README '+DATA['source_commit'][:7],
            fontsize=11,color=MUTED,ha='right')
    return fig,ax


def box(ax,x,y,w,h,text,size=18,fill=LIGHT,edge=None,color=NAVY,bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle='round,pad=0.015,rounding_size=0.10',facecolor=fill,
        edgecolor=edge or fill,linewidth=1.5))
    ax.text(x,y,text,ha='center',va='center',fontsize=size,color=color,
            weight='bold' if bold else 'normal',linespacing=1.35)


def arrow(ax,x1,y1,x2,y2,color=MUTED):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
                 mutation_scale=17,linewidth=1.6,color=color))


def save(fig,name):
    for ext in ('svg','png'):
        meta={'Date':DATA['date'],'Creator':'Public aggregate progress figures'} if ext=='svg' else {'Software':'Public aggregate progress figures'}
        target = OUT/(name+'.'+ext)
        fig.savefig(target,dpi=160,metadata=meta)
        if ext == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


def extension():
    fig,ax=canvas(1,'Previous work → current extension',
        'Improve the previous insurance model using alternative adaptation techniques and increase answer accuracy.')
    ax.text(.65,6.83,'PREVIOUS INTERNS',fontsize=16,weight='bold',color=BLUE)
    xs=[2.25,6.1,9.95,13.8]
    texts=['Qwen2.5-7B','Existing RAG','QLoRA-RAFT\nexperiment','Frozen benchmark\n140 questions']
    for x,t in zip(xs,texts):box(ax,x,5.94,3.1,1.1,t,size=19)
    for a,b in zip(xs,xs[1:]):arrow(ax,a+1.57,5.94,b-1.57,5.94)
    ax.text(.7,4.98,'Historical composite:  Base Qwen7B + RAG  0.608431     |     QLoRA-RAFT + RAG  0.599115',fontsize=18)
    ax.text(.65,4.15,'MY EXTENSION',fontsize=16,weight='bold',color=TEAL)
    xs=[1.9,4.95,8.0,11.05,14.1]
    texts=['3B prototype\nCurrent compute\nconstraints','Qwen3B /\nMinistral-3B\ncomparison','Plain QLoRA-SFT\non Ministral','E2 vs E4\nControlled\ncomparison','Error +\nfaithfulness\nanalysis']
    for x,t in zip(xs,texts):box(ax,x,3.10,2.55,1.6,t,size=17,fill=PALE)
    for a,b in zip(xs,xs[1:]):arrow(ax,a+1.30,3.1,b-1.30,3.1,color=TEAL)
    box(ax,8,1.4,14.6,.85,'3B is used as a practical prototype; 7B remains desirable for later validation.',size=20,fill=LIGHT)
    save(fig,'01_project_extension')


def design():
    fig,ax=canvas(2,'Experimental design: E2 versus E4',
        'Main question: does plain QLoRA-SFT improve generation with the RAG evidence held fixed?')
    rows=[('E1','Base Ministral','Without RAG','Closed-book control',6.50,False),
          ('E2','Base Ministral','Frozen RAG','Main pre-SFT control',5.13,True),
          ('E3','QLoRA-SFT Ministral','Without RAG','Closed-book diagnostic',3.76,False),
          ('E4','QLoRA-SFT Ministral','Same frozen RAG','Main post-SFT experiment',2.39,True)]
    for lab,model,rag,role,y,main in rows:
        color=TEAL if main else MUTED;fill=PALE if main else LIGHT
        box(ax,1.1,y,1,.83,lab,22,fill=color,color='white',bold=True)
        box(ax,4.0,y,4.0,.83,model,19,fill=fill)
        arrow(ax,6.13,y,6.68,y,color)
        box(ax,8.2,y,2.9,.83,rag,19,fill=fill)
        arrow(ax,9.78,y,10.33,y,color)
        box(ax,12.75,y,4.5,.83,role,18,fill=fill,edge=color if main else None)
    ax.plot([15.3,15.55,15.55,15.3],[5.13,5.13,2.39,2.39],color=TEAL,lw=2)
    ax.text(8,1.38,'Same 140 questions  •  Same frozen evidence (E2 / E4)  •  Same prompting/evaluation',ha='center',fontsize=17)
    ax.text(8,.91,'Main changed variable: QLoRA-SFT',ha='center',fontsize=19,weight='bold',color=TEAL)
    ax.text(8,.54,'Runtime caveat: tuned inference prepares the base and upcasts some non-quantised parameters.',ha='center',fontsize=12,color=MUTED)
    save(fig,'02_experiment_design')


def timeline():
    fig,ax=canvas(3,'Week 2: from inherited pipeline to completed E3/E4',
        'Generation and engineering gates are complete. Correctness and faithfulness are still unmeasured.')
    stages=[('Inherited benchmark\nand RAG','DONE'),('Audited previous\nrepository','DONE'),('21 clean\nSFT examples','DONE'),('Base Qwen3B /\nMinistral evaluation','DONE'),('Smoke v1','GATE FAILED'),('Smoke v2','GATE FAILED'),
            ('Smoke v3','PASS'),('Full QLoRA-SFT','PASS'),('E3/E4 evaluation\n140 + 140 answers','DONE'),('Correctness\njudging','PENDING'),('Error / faithfulness\nanalysis','PENDING JUDGING')]
    for start,end,y in [(0,6,5.95),(6,11,2.85)]:
        xs=[1.75+2.5*i for i in range(end-start)]
        ax.plot([xs[0],xs[-1]],[y,y],color='#CBD5DF',lw=3,zorder=0)
        for x,idx in zip(xs,range(start,end)):
            title,status=stages[idx];pending='PENDING' in status;failed=status=='GATE FAILED'
            color=AMBER if failed else MUTED if pending else TEAL
            ax.scatter([x],[y],s=570,c='white' if pending else color,edgecolors=color,linewidths=2,zorder=3)
            ax.text(x,y,str(idx+1),color=color if pending else 'white',ha='center',va='center',fontsize=15,weight='bold')
            ax.text(x,y-.7,title,ha='center',va='top',fontsize=16,linespacing=1.3)
            ax.text(x,y-1.58,status,ha='center',fontsize=12,color=color,weight='bold')
    ax.plot([14.52,15.55,15.55,1.75],[5.95,5.95,3.95,3.95],color='#CBD5DF',lw=1.5)
    arrow(ax,1.75,3.95,1.75,3.15,color='#9BAAB9')
    ax.text(.7,.75,'Failed smokes were diagnosed before proceeding; the strict numerical reload tolerance was never relaxed.',fontsize=15,color=MUTED)
    save(fig,'03_week2_progress')


def reload_validation():
    fig,ax=canvas(4,'Adapter reload: numerical mismatch resolved',
        'Supporting engineering validation — not evidence of improved answer quality.')
    plot=fig.add_axes([.11,.30,.81,.46]);x=list(range(3));w=.27
    for off,key,color,title in [(-w/2,'max',BLUE,'Maximum absolute difference'),(w/2,'mean',TEAL,'Mean absolute difference')]:
        vals=[r[key] for r in DATA['reload']]
        plot.bar([i+off for i in x],vals,width=w,color=color,label=title)
        for i,v in enumerate(vals):
            if v==0:plot.scatter(i+off,0,s=38,marker='D',color=color,zorder=4,clip_on=False)
            plot.text(i+off,v+.018,f'{v:.6f}',ha='center',fontsize=14,color=color,weight='bold')
    plot.set(xticks=x,xticklabels=[r['label'] for r in DATA['reload']],ylim=(0,.56),ylabel='Absolute logit difference')
    plot.tick_params(axis='x',labelsize=17,length=0,pad=10);plot.tick_params(axis='y',labelsize=13)
    plot.spines[['top','right']].set_visible(False);plot.grid(axis='y',alpha=.2);plot.set_axisbelow(True)
    plot.legend(loc='upper right',frameon=False,fontsize=15)
    ax.text(8,1.47,'Smoke v3 identified the remaining mismatch as an inference-state issue caused by the',ha='center',fontsize=17)
    ax.text(8,1.06,'Accelerate autocast wrapper. The final fresh adapter reload matched exactly.',ha='center',fontsize=17)
    ax.text(8,.61,'Original tolerance retained: atol = 0.02, rtol = 0.01. All functional and tensor checks also passed.',ha='center',fontsize=13,color=MUTED)
    save(fig,'04_reload_validation')


def results():
    fig,ax=canvas(5,'Current benchmark results',
        'Full frozen benchmark · 140 questions · higher is better · E4 generation is now complete')
    metrics=[('correctness','Correctness'),('citation','Citation resolution'),('abstention','Balanced abstention'),('composite','Composite')]
    rr=DATA['results'];colors=[BLUE,BLUE,MUTED,TEAL,TEAL]
    for j,(key,title) in enumerate(metrics):
        plot=fig.add_axes([.245+j*.183,.26,.168,.49])
        plot.set(xlim=(0,1.22),ylim=(-.6,4.6),xticks=[0,.5,1],yticks=[])
        plot.invert_yaxis();plot.set_title(title,fontsize=16,pad=20,weight='bold')
        for i,row in enumerate(rr):
            v=row[key]
            if v is None:
                plot.text(.12,i,'Pending',ha='left',va='center',fontsize=15,color=MUTED,style='italic')
            else:
                plot.barh(i,v,height=.47,color=colors[i],alpha=.95 if i==4 else .8)
                plot.text(v+.025,i,f'{v:.3f}',ha='left',va='center',fontsize=13,color=NAVY)
        plot.spines[['top','right','left']].set_visible(False);plot.grid(axis='x',alpha=.16);plot.set_axisbelow(True)
        plot.tick_params(axis='x',labelsize=13)
        if j==0:
            for i,row in enumerate(rr):
                plot.text(-.10,i,row['label'],ha='right',va='center',fontsize=16,weight='bold' if i==4 else 'normal',clip_on=False)
    ax.text(8,1.55,'No overall winner is declared until correctness judging is complete.',ha='center',fontsize=20,weight='bold')
    ax.text(8,1.04,'Pending is unmeasured, not zero. Citation resolution is not semantic faithfulness.',ha='center',fontsize=16,color=MUTED)
    ax.text(8,.63,'Historical composite = 0.5 correctness + 0.3 citation resolution + 0.2 balanced abstention.',ha='center',fontsize=14,color=MUTED)
    save(fig,'05_current_results')


def decisions():
    fig,ax=canvas(6,'After E4: choose the next experiment from the evidence',
        'Future options only. Complete correctness judging first; do not tune on the frozen test results.')
    box(ax,8,6.72,8.8,.75,'E4 result: does plain QLoRA-SFT improve accuracy?',20,fill=NAVY,color='white',bold=True)
    arrow(ax,5.5,6.32,3.15,5.93,TEAL);arrow(ax,10.5,6.32,11.2,5.93,AMBER)
    box(ax,3.15,5.55,4.75,.70,'YES',19,fill=PALE,color=TEAL,bold=True)
    box(ax,11.2,5.55,7.65,.70,'LITTLE / NO IMPROVEMENT (OR WORSE)',17,fill=SAND,color=AMBER,bold=True)
    left=['Analyse which errors improved','Expand high-quality grounded SFT data','Test robustness','If compute is available:\nvalidate the method on a 7B model']
    for i,t in enumerate(left):
        y=4.5-i*1.04
        arrow(ax,3.15,y+.67,3.15,y+.4,TEAL)
        box(ax,3.15,y,5.45,.80,t,16,fill=PALE)
    box(ax,11.2,4.53,7.65,.73,'Inspect paired E2 vs E4 errors',19,fill=LIGHT)
    arrow(ax,11.2,5.17,11.2,4.92)
    ax.plot([11.2,11.2,6.4,6.4],[4.13,4.01,4.01,1.35],color=AMBER,lw=1.3)
    conditions=[('Data-limited','Increase verified / synthetic\ngrounded SFT examples'),
        ('Incomplete or\nunsupported generation','Test preference optimisation\n/ DPO'),
        ('Refusal weakness','Add targeted answerable /\nunanswerable training')]
    for i,(c,t) in enumerate(conditions):
        y=3.43-i*1.04
        arrow(ax,6.4,y,6.64,y,AMBER)
        box(ax,8.43,y,3.5,.85,c,15,fill=SAND)
        arrow(ax,10.23,y,10.73,y,AMBER)
        box(ax,13.02,y,4.43,.85,t,15,fill=LIGHT)
    ax.text(8,.65,'These are conditional research directions, not completed work or a recommendation selected before judging.',ha='center',fontsize=13,color=MUTED)
    save(fig,'06_next_step_decision')


def contribution():
    fig,ax=canvas(7,'Research contribution: what was inherited, added and next',
        'Extend the previous controlled experiment; preserve the benchmark and report neutral or negative findings honestly.')
    cols=[('INHERITED',BLUE,['Frozen benchmark','RAG pipeline','Qwen7B baseline','QLoRA-RAFT result']),
          ('ADDED THIS WEEK',TEAL,['Qwen3B / Ministral3B support','21-example plain SFT dataset','Strict leakage controls','Reproducible QLoRA reload','E2 vs E4 experiment','Motor-only slice (76 questions)','E3/E4: 140 + 140 answers']),
          ('NEXT',MUTED,['LLM correctness judging','Paired error analysis','Faithfulness / hallucination audit','Choose next adaptation technique'])]
    for x,(title,color,items) in zip([2.95,8,13.05],cols):
        box(ax,x,4.12,4.7,5.65,'',fill=LIGHT if title=='INHERITED' else PALE if title=='ADDED THIS WEEK' else '#F5F6F8')
        ax.text(x-2.08,6.45,title,fontsize=18,weight='bold',color=color)
        for i,item in enumerate(items):ax.text(x-2.08,5.68-i*.56,'• '+item,fontsize=16,va='center')
    ax.text(8,.86,'E3/E4 generation is complete. New-model correctness, composite and faithfulness remain pending.',ha='center',fontsize=16,color=MUTED)
    save(fig,'07_research_contribution')


if __name__=='__main__':
    if DATA['e3_e4_status']!='complete':raise ValueError('Update figure text for changed experiment status')
    for draw in [extension,design,timeline,reload_validation,results,decisions,contribution]:draw()
    print('Generated seven PNG and seven SVG figures from public aggregate_values.json')
