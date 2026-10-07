"""Seed-only SFT. Test/dev records are read ONLY as exclusion checks, never as examples."""
import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
from .core import benchmark, common, digest, private_output, rag, rows, sha, unique, write_json, write_rows

ACCEPTED={'verified','council_verified','council_adjudicated'}
HELD_OUT={'lv=','post office'}


def leakage_reasons(seed, forbidden, blocked_ids):
    reasons=[]
    if seed['split']!='seed':reasons.append('not_seed')
    if seed['id'] in blocked_ids:reasons.append('test_or_dev_intent_cluster')
    if seed.get('review_status') not in ACCEPTED:reasons.append('unverified')
    for other in forbidden:
        if seed['id']==other['id']:reasons.append('forbidden_id')
        q,a=common.norm(seed['question']),common.norm(seed['gold_answer'])
        oq,oa=common.norm(other['question']),common.norm(other['gold_answer'])
        if q==oq or SequenceMatcher(None,q,oq).ratio()>=0.85:reasons.append('question_overlap')
        if a==oa:reasons.append('answer_overlap')
    return sorted(set(reasons))


def select_evidence(seed,chunks):
    selected={}
    for citation in seed['gold_citations']:
        if citation['insurer'].casefold() in HELD_OUT:raise ValueError('held_out_insurer')
        candidates=[c for c in chunks if
            c['meta']['filename']==citation['doc_file'] and
            c['meta']['insurer'].casefold()==citation['insurer'].casefold() and
            abs(int(c['meta']['page'])-int(citation['page']))<=1 and
            common.norm(citation['quote']) and common.norm(citation['quote']) in common.norm(c['text'])]
        if not candidates:raise ValueError('supporting_quote_not_found')
        chosen=min(candidates,key=lambda c:(len(c['text']),c['id']))
        selected[chosen['id']]=chosen
    if not selected:raise ValueError('no_supporting_citations')
    return list(selected.values())


def build(home):
    home=Path(home);gold=benchmark(home)
    extra=rows(home/'data/gold/seed_extra_v2.jsonl')
    seeds=[g for g in gold if g['split']=='seed']+extra
    unique(seeds,'id')
    forbidden=[g for g in gold if g['split'] in ('test','dev')]
    clusters=json.loads((home/'data/gold/clusters.json').read_text())
    if clusters['gold_sha256']!=sha(home/'data/gold/gold_v2.jsonl'):raise ValueError('Stale cluster sidecar')
    blocked={i for c in clusters['straddles'] if set(c['splits'])&{'test','dev'} for i in c['ids']}
    chunks=rows(home/'data/chunks.jsonl');unique(chunks,'id')
    accepted=[];excluded=[]
    for seed in seeds:
        reasons=leakage_reasons(seed,forbidden,blocked)
        if not seed['answerable']:reasons.append('no_context_refusal_deferred')
        if reasons:
            excluded.append({'seed_id':seed['id'],'reasons':reasons});continue
        try: evidence=select_evidence(seed,chunks)
        except ValueError as e:
            excluded.append({'seed_id':seed['id'],'reasons':[str(e)]});continue
        # Keep the entire verified answer. Add resolvable pointers to its supplied evidence.
        tags=[common.chunk_header(c['meta']).split(']')[0]+']' for c in evidence]
        target=seed['gold_answer'].strip()+'\n\n'+' '.join(dict.fromkeys(tags))
        messages=rag.build_messages(seed['question'],seed.get('persona','end_user'),evidence)
        messages.append({'role':'assistant','content':target})
        blob=common.norm('\n'.join([seed['question'],target]+[c['text'] for c in evidence]))
        # Shared document text is allowed, but complete held-out QA strings are not.
        collision=any(common.norm(g['question']) in blob or common.norm(g['gold_answer']) in blob
                      or g['id'] in '\n'.join(m['content'] for m in messages) for g in forbidden)
        if collision:
            excluded.append({'seed_id':seed['id'],'reasons':['forbidden_qa_in_rendered_row']});continue
        check=rag.verify_citations(target,evidence)
        if not check['all_supported']:
            excluded.append({'seed_id':seed['id'],'reasons':['unresolved_target_citation']});continue
        accepted.append({'source_record_id':seed['id'],'messages':messages,'provenance':{
            'source_file':'seed_extra_v2.jsonl' if seed in extra else 'gold_v2.jsonl',
            'source_split':'seed','source_record_sha256':digest(seed),
            'review_status':seed['review_status'],'reviewer':seed.get('reviewer'),
            'selection':'seed citation quote -> shortest matching corpus chunk; no retrieval against test/dev',
            'evidence':[{'chunk_id':c['id'],'chunk_sha256':digest(c),'document':c['meta']['filename']} for c in evidence],
            'target_rule':'complete verified seed gold_answer plus evidence citation headers',
            'messages_sha256':digest(messages)}})
    if not accepted:raise ValueError('No eligible SFT examples')
    return accepted,excluded


def prepare(home,output):
    accepted,excluded=build(home)
    out=private_output(output);write_rows(out/'train.jsonl',accepted)
    home=Path(home)
    write_json(out/'manifest.json',{'schema':'seed-sft/1','leakage_checks_passed':True,
        'policy':'shared corpus; seed-derived evidence only; no test/dev QA, synthetic or frozen contexts',
        'train_rows':len(accepted),'excluded':excluded,
        'train_sha256':sha(out/'train.jsonl'),
        'source_hashes':{n:sha(home/n) for n in ['data/gold/gold_v2.jsonl','data/gold/seed_extra_v2.jsonl',
            'data/gold/FREEZE_v2.json','data/gold/clusters.json','data/chunks.jsonl']}})
    print(f'{len(accepted)} seed-only SFT rows; {len(excluded)} excluded. Private provenance written.')


def validate_bundle(bundle):
    bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    data=rows(bundle/'train.jsonl')
    if manifest.get('schema')!='seed-sft/1' or manifest.get('leakage_checks_passed') is not True:
        raise ValueError('Missing leakage audit')
    if sha(bundle/'train.jsonl')!=manifest['train_sha256'] or len(data)!=manifest['train_rows']:
        raise ValueError('Training bundle changed since leakage audit')
    unique(data,'source_record_id')
    for row in data:
        p=row['provenance']
        if p['source_split']!='seed' or p['source_file'] not in ('gold_v2.jsonl','seed_extra_v2.jsonl'):
            raise ValueError('Non-seed training provenance')
        if p['messages_sha256']!=digest(row['messages']) or not p['evidence']:
            raise ValueError('Missing or inconsistent evidence provenance')
    return data,manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.home,a.output)
