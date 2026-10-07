"""Export frozen prompts, run matched base/tuned rows, and reuse inherited scoring."""
import argparse
import json
from pathlib import Path
import random
import statistics
import time
from .core import (ROOT, benchmark, common, config, digest, generate, load_model,
    load_module, private_output, rag, rows, sha, unique, write_json, write_rows)


def prepare(home, output, split, index=None):
    gold = [g for g in benchmark(home) if g['split'] == split]
    home = Path(home)
    harness = load_module('_experiment_harness', 'eval/harness.py')
    if split == 'test':
        path = home / 'eval/results/frozen_contexts_v2.jsonl'
        if sha(path)!=config()['frozen_contexts_sha256']:
            raise ValueError('Historical frozen test contexts changed')
        contexts = unique(rows(path), 'qid')
        if set(contexts) != {g['id'] for g in gold}:
            raise ValueError('Frozen contexts do not cover exactly the test population')
    else:
        if not index:
            raise ValueError('Dev export requires --index: build a separate index from the frozen chunks')
        im=json.loads((Path(index)/'index_manifest.json').read_text())
        if im['chunks_sha256']!=sha(home/'data/chunks.jsonl') or im['embedding_model']!=common.EMBED_MODEL:
            raise ValueError('Dev index differs from frozen corpus or embedding model')
        rag.CHROMA_DIR=Path(index)/'chroma';rag._collection=None
        # Preserve retrieval settings, but reject the inherited silent fallback.
        if rag.reranker() is None:
            raise RuntimeError('Dev retrieval requires the original BGE reranker')
        contexts = {g['id']: {'qid': g['id'], 'chunks': rag.retrieve(g['question'], k=6)} for g in gold}
    tasks = []
    for g in gold:
        for mode in ('closedbook', 'openbook'):
            tasks.append({'qid': g['id'], 'mode': mode, 'split': split,
                'messages': harness.build_row_messages(g, contexts[g['id']], mode == 'openbook')})
    out = private_output(output)
    write_rows(out / 'tasks.jsonl', tasks)
    write_rows(out / 'contexts.jsonl', list(contexts.values()))
    write_json(out / 'manifest.json', {'split': split, 'questions': len(gold),
        'tasks_sha256': sha(out/'tasks.jsonl'), 'contexts_sha256': sha(out/'contexts.jsonl'),
        'gold_sha256': sha(home/'data/gold/gold_v2.jsonl'), 'config': config(),
        'source_context_sha256': sha(path) if split == 'test' else None})


def runtime_versions():
    import importlib.metadata
    import torch
    return {**{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes','accelerate']},
        'gpu':torch.cuda.get_device_name(0)}


def run(bundle, output, model_key, adapter=None):
    bundle = Path(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    tasks = rows(bundle/'tasks.jsonl')
    if sha(bundle/'tasks.jsonl') != manifest['tasks_sha256']:
        raise ValueError('Evaluation tasks changed after export')
    if len({(t['qid'],t['mode']) for t in tasks}) != len(tasks) or len(tasks) != manifest['questions']*2:
        raise ValueError('Incomplete or duplicate task population')
    out = private_output(output)
    model, tokenizer = load_model(model_key)
    if adapter:
        if model_key != 'ministral3b':
            raise ValueError('Adapter supported only for the pinned Ministral checkpoint')
        adapter_config=json.loads((Path(adapter)/'adapter_config.json').read_text())
        if adapter_config.get('base_model_name_or_path')!=config()['models'][model_key]['id']:
            raise ValueError('Adapter base checkpoint mismatch')
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    label = model_key + ('-sft' if adapter else '-base')
    for mode in ('closedbook','openbook'):
        with (out/f'answers_{label}-{mode}.jsonl').open('x') as f:
            for t in [x for x in tasks if x['mode']==mode]:
                start=time.monotonic()
                answer,n = generate(model,tokenizer,t['messages'])
                f.write(json.dumps({'qid':t['qid'],'row':f'{label}-{mode}',
                    'model':config()['models'][model_key], 'open_book':mode=='openbook',
                    'answer':answer,'prompt_tokens':n,'latency_s':time.monotonic()-start})+'\n')
                f.flush()
    write_json(out/'manifest.json', {'status':'complete','input':manifest,'model_key':model_key,
        'adapter': str(adapter) if adapter else None, 'config':config(),
        'versions':runtime_versions()})


def score(home, bundle, answers, scores, output):
    gold_all=benchmark(home)
    manifest=json.loads((Path(bundle)/'manifest.json').read_text())
    if sha(Path(home)/'data/gold/gold_v2.jsonl')!=manifest['gold_sha256']:
        raise ValueError('Gold changed since prompt export')
    if sha(Path(bundle)/'contexts.jsonl')!=manifest['contexts_sha256']:
        raise ValueError('Evaluation contexts changed')
    gold={g['id']:g for g in gold_all if g['split']==manifest['split']}
    ctx=unique(rows(Path(bundle)/'contexts.jsonl'),'qid')
    judge=load_module('_experiment_judge','eval/judge.py')
    ab=load_module('_experiment_abstention','eval/abstention.py')
    votes=load_module('_experiment_votes','eval/judge_votes.py')
    answer_rows=rows(answers); byid=unique(answer_rows,'qid')
    if set(byid)!=set(gold):raise ValueError('Incomplete answer population')
    row_names={a['row'] for a in answer_rows}
    if len(row_names)!=1:raise ValueError('Mixed model rows')
    row_name=next(iter(row_names))
    ss=rows(scores) if scores else []
    scoremap={(s['qid'],s['row']):s for s in ss}
    if len(scoremap)!=len(ss):raise ValueError('Duplicate judge scores')
    records=[];tasks=[]
    for q,g in gold.items():
        a=byid[q];text=a['answer']
        abst=ab.is_abstention_assertion(text,abstain_sentence=common.ABSTAIN_SENTENCE,citation_re=common.CITATION_RE)
        check=rag.verify_citations(text,ctx[q]['chunks'])
        corr=None;judged=False
        if g['answerable']:
            if abst:corr=0.0
            else:
                s=scoremap.get((q,row_name))
                if s is None:
                    tasks.append({'qid':q,'row':row_name,'question':g['question'],'gold_answer':g['gold_answer'],
                        'key_facts':g['key_facts'],'must_not_assert':g.get('must_not_assert',''),'model_answer':text})
                else:
                    ok,why=votes.vote_valid(s,len(g['key_facts']))
                    if not ok:raise ValueError(f'Invalid vote: {why}')
                    spans=s.get('evidence_spans')
                    if spans is None or any(not isinstance(v,str) or (v and v not in text) for v in spans):
                        raise ValueError('Judge evidence spans must be verbatim answer substrings')
                    corr=0.0 if s['contradiction'] else statistics.mean(s['fact_scores'])/2
                    judged=True
        records.append({'qid':q,'answerable':g['answerable'],'abst':abst,'n_sup':check['n_supported'],
            'n_tot':check['n_total'],'corr_item':corr,'judged':judged,'parse_ok':abst or check['n_total']>0})
    out=private_output(output)
    if tasks:
        random.Random(0).shuffle(tasks)
        mapping=[];blind=[]
        for i,t in enumerate(tasks):
            code=f't{i:04d}';mapping.append({'task':code,'qid':t['qid'],'row':t['row']})
            blind.append({'task':code,**{k:v for k,v in t.items() if k not in ('qid','row')}})
        write_rows(out/'judge_tasks.jsonl',blind);write_rows(out/'judge_map.jsonl',mapping)
        write_json(out/'status.json',{'status':'awaiting_judging','missing':len(tasks),
            'note':'Use inherited JUDGE_RUBRIC and judge_votes.merge_task for three votes. No composite until complete.'})
        return
    result=judge._row_metrics(records)
    result.update(row=row_name,n=len(records),split=manifest['split'],
        parse_rate=sum(r['parse_ok'] for r in records)/len(records),
        composite_ci=judge.bootstrap_ci(records,'composite'))
    result['rank_eligible']=result['parse_rate']>=0.9
    write_json(out/'metrics.json',result)
    write_rows(out/'per_question.jsonl',records)


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    x=s.add_parser('prepare');x.add_argument('--home',required=True);x.add_argument('--split',choices=['dev','test'],default='dev');x.add_argument('--index');x.add_argument('--output',required=True)
    x=s.add_parser('run');x.add_argument('--bundle',required=True);x.add_argument('--output',required=True);x.add_argument('--model',choices=['qwen3b','ministral3b'],required=True);x.add_argument('--adapter')
    x=s.add_parser('score');x.add_argument('--home',required=True);x.add_argument('--bundle',required=True);x.add_argument('--answers',required=True);x.add_argument('--scores');x.add_argument('--output',required=True)
    a=p.parse_args();kw=vars(a);cmd=kw.pop('cmd')
    if cmd=='run':kw['model_key']=kw.pop('model')
    globals()[cmd](**kw)

if __name__=='__main__':main()
