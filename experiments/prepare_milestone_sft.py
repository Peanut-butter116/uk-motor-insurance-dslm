"""Explicit opt-in mixed-quality milestone path; seed loader stays byte-identical."""
import collections
import hashlib
import json
import random
from pathlib import Path
from experiments import core

PROMPT_VERSION='production-context-citations-refusal/1'


def messages(row):
    # Preserve the established evidence-only instruction template, but adapt its
    # two incompatible clauses to the production targets without rewriting them.
    s=core.rag.SYSTEM_TEMPLATE.format(style=core.rag.PERSONA_STYLE['end_user'],
        abstain=core.common.ABSTAIN_SENTENCE,context=row['context'])
    s=s.replace('in the form [Insurer, Doc, Section, p.N].',
                'using the exact bracketed identifiers supplied with the extracts.')
    s=s.replace('reply with exactly this sentence and nothing else: "'+core.common.ABSTAIN_SENTENCE+'"',
                'state explicitly that the supplied evidence is insufficient; do not invent an answer')
    return [{'role':'system','content':s+' /no_think'},
            {'role':'user','content':row['question']},
            {'role':'assistant','content':row['answer']}]


def grouped_split(rows,seed=0,fraction=.1):
    groups=collections.defaultdict(list)
    for r in rows:groups[core.common.norm(r['question'])].append(r['example_id'])
    keys=sorted(groups);random.Random(seed).shuffle(keys)
    valid=set();target=max(1,round(len(rows)*fraction))
    for key in keys:
        if len(valid)>=target:break
        valid.update(groups[key])
    return [r['example_id'] for r in rows if r['example_id'] not in valid],sorted(valid)


def prepare(export,output,cfg,protected_gold=None):
    export,output=Path(export),Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((export/'uk_motor_qa_d1000_manifest.json').read_text())
    quality=json.loads((export/'uk_motor_qa_d1000_quality_report.json').read_text())
    assert manifest['records']==1000 and manifest['verification_counts']=={'verified':462,'unverified':538}
    assert quality['training_integrity_passed'] is True
    source=export/'uk_motor_qa_d1000.jsonl'
    assert core.sha(source)==manifest['files'][source.name]
    rows=core.rows(source);assert sorted(r['example_id'] for r in rows)==manifest['record_ids']
    blocked={}
    if protected_gold is not None:
        protected={(c['doc_file'],str(c['page'])) for g in protected_gold if g['split'] in ('test','dev') for c in g['gold_citations']}
        for r in rows:
            if any((e['policy_id'],str(e['page'])) in protected for e in r['evidence_text']):
                blocked[r['example_id']]='protected_supporting_page_overlap'
    from transformers import AutoTokenizer
    tokenizers={};failures={};encodings={};lengths={};exclusions=[]
    # All four enter tokenizer/access preflight before any training is scheduled.
    for key,spec in cfg['models'].items():
        try:
            kw={'fix_mistral_regex':True} if spec['class']=='mistral3' else {}
            tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],**kw)
            tokenizers[key]=tok;encodings[key]={};lengths[key]={}
            for r in rows:
                ms=messages(r);n=len(core.token_ids(tok,ms));lengths[key][r['example_id']]=n
                if n<=cfg['training']['max_length']:
                    encodings[key][r['example_id']]=core.encode_training(tok,ms,cfg['training']['max_length'])
        except Exception as e:failures[key]={'type':type(e).__name__,'reason':str(e)[:600]}
    good=[k for k in encodings if k not in failures]
    if not good:raise RuntimeError('No model passed tokenisation/access preflight')
    common=set.intersection(*(set(encodings[k]) for k in good))-set(blocked)
    for r in rows:
        if r['example_id'] not in common:exclusions.append({'record_id':r['example_id'],
            'reason':blocked.get(r['example_id'],'whole_record_over_2048_or_mask_incompatible'),'lengths':{k:lengths[k].get(r['example_id']) for k in good}})
    eligible=[r for r in rows if r['example_id'] in common]
    assert len(eligible)>=2,'No useful common population remains; no truncation allowed'
    train,val=grouped_split(eligible,cfg['seed'],cfg['validation_fraction'])
    result={'schema':'milestone-sft/1','source_manifest_sha256':core.sha(export/'uk_motor_qa_d1000_manifest.json'),
        'source_csv_sha256':manifest['files']['uk_motor_qa_d1000.csv'],'source_rows':1000,
        'eligible_rows':len(eligible),'train_ids':train,'validation_ids':val,'excluded':exclusions,
        'protected_page_exclusion_count':len(blocked),
        'model_preflight_failures':failures,'models':cfg['models'],'prompt_version':PROMPT_VERSION,
        'seed':cfg['seed'],'validation_policy':'Question-grouped 10%; logging only; no checkpoint selection or benchmark tuning',
        'verification_counts':manifest['verification_counts'],'selection_permitted':False}
    for key in good:
        for split,ids in [('train',train),('validation',val)]:
            core.write_rows(output/f'{key}-{split}.jsonl',[{'source_record_id':i,**encodings[key][i]} for i in ids])
    core.write_json(output/'manifest.json',result)
    core.write_json(output/'bundle_hashes.json',{p.name:core.sha(p) for p in output.glob('*.jsonl')})
    return result
