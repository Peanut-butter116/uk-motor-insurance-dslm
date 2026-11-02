"""Cloud-only aggregation. Export ONLY the returned safe aggregate payload.

No questions, answers, policy evidence, per-request IDs or quota entries leave
the approved private runtime. Metric definitions match single_judge_fast.metrics.
"""
import collections
import hashlib
import json
from pathlib import Path
import random
import statistics
import tarfile
import time
from cloud.single_judge_fast.protocol import PROTOCOL,MODELS,rows,select,load_validated,unknown_ids,quota_audit


def bootstrap_mean(values, seed=20261102, repeats=2000):
    values=[v for v in values if v is not None]
    if not values:return None,None
    rng=random.Random(seed);n=len(values)
    draws=sorted(sum(rng.choices(values,k=n))/n for _ in range(repeats))
    return draws[int(.025*(repeats-1))],draws[int(.975*(repeats-1))]


def mean_parts(values):
    values=[v for v in values if v is not None]
    return (sum(values)/len(values),sum(values),len(values)) if values else (None,0,0)


def build(recovery):
    import asyncio
    from experiments import core
    from cloud.d1000 import judging,judge_recovery as jr
    from cloud.single_judge_fast.engine import writer_guard
    from cloud.single_judge_fast.lifecycle import provider_threads
    from cloud.recovery_v1.monitor import saved_vote_counts
    root=Path(recovery.evaluation.ROOT)
    if provider_threads() or any(t.get_name() in ('fast-groq','fast-gemini') and not t.done() for t in asyncio.all_tasks()):
        raise RuntimeError('Drain coordinator before authoritative reporting snapshot')
    writer_guard(recovery)
    persistence=recovery.state['last_persistence'];version=persistence['version']
    if not persistence.get('readback_verified') or not persistence.get('private'):raise RuntimeError('Private readback required')
    archive=Path('/kaggle/working/publish/artifacts.bin')
    h=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    archive_sha=h.hexdigest()
    if archive_sha!=recovery.state['artifacts_sha256']:raise RuntimeError('Latest readback archive differs')
    # Strict saved vote identity/schema/citation validation, not journal counts alone.
    candidates=judging.tasks(root);votes=load_validated(root,candidates);chosen=select(votes)
    if len(candidates)!=912:raise RuntimeError('Expected 912 saved responses')
    counts=saved_vote_counts(archive)
    if counts['persisted_judge_counts']!={p:len(votes[p]) for p in MODELS} or any(counts['persisted_duplicate_votes'].values()):
        raise RuntimeError('Archive vote population differs')
    # Verify reporting inputs inside the independently readback archive.
    expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for directory in ('judging','judge_runtime/evaluation_cache_v2') for p in (root/directory).rglob('*')
              if p.is_file() and p.suffix not in ('.lock','.tmp')}
    seen=set()
    with tarfile.open(archive,'r|gz') as snapshot:
        for member in snapshot:
            if member.name not in expected:continue
            if hashlib.sha256(snapshot.extractfile(member).read()).hexdigest()!=expected[member.name]:
                raise RuntimeError('Archived reporting input hash differs')
            seen.add(member.name)
    if seen!=set(expected):raise RuntimeError('Reporting inputs absent from readback archive')
    providers,runtime=judging.provider_runtime(root)
    for p in MODELS:jr.verify_calibration(json.loads((root/'judging'/f'{p}-calibration-v2.json').read_text()))
    protected=set().union(*(unknown_ids(root,candidates,p) for p in MODELS))
    gold={g['id']:g for g in rows(root/'evaluation/gold.jsonl')}
    contexts={r['qid']:r for r in rows(root/'evaluation/contexts.jsonl')}
    cfg=json.loads((root/'config.json').read_text())
    assert len(gold)==76 and sum(g['answerable'] for g in gold.values())==67
    ab=core.load_module('_report_abstention','eval/abstention.py')
    comparison=[];long=[];scores={};missing_states=collections.Counter();protected_private=[]
    for task,_ in candidates:
        ident=task['evaluation_id']
        if ident in protected:protected_private.append(ident)
        if ident in chosen or ident in protected:continue
        key=core.digest({'task':task,'provider':'gemini','model':MODELS['gemini'],'protocol':jr.PROTOCOL})
        hp=Path(runtime)/'evaluation_cache_v2/gemini'/(key+'.attempts.json')
        history=json.loads(hp.read_text()) if hp.exists() else []
        state=history[-1]['status'] if history else 'not_attempted'
        missing_states[state]+=1
        if state=='known_invalid' and sum(x.get('transport_revision',2)==jr.TRANSPORT_REVISION for x in history)>=3:
            missing_states['known_invalid_attempt_bound_exhausted']+=1
    for model_key,spec in cfg['models'].items():
        model_id=spec['id']
        for condition in ('base','base-rag','sft','sft-rag'):
            response_path=root/'runs'/model_key/(('sft' if condition.startswith('sft') else 'base')+'-responses.jsonl')
            responses={r['qid']:r for r in rows(response_path) if r['condition']==condition}
            available=len(responses)==76;rag=condition.endswith('-rag')
            vals={'correctness':[],'reference_support':[],'citation_support':[]}
            n=collections.Counter();assign=collections.Counter();scores[(model_key,condition)]={}
            for q,g in gold.items():
                r=responses.get(q)
                if r is None:continue
                answer=r['answer']
                refused=ab.is_abstention_assertion(answer,abstain_sentence=core.common.ABSTAIN_SENTENCE,citation_re=core.common.CITATION_RE)
                check=core.rag.verify_citations(answer,contexts[q]['chunks'])
                n['parse']+=int(refused or check['n_total']>0)
                n['false_refusal' if g['answerable'] else 'appropriate_refusal']+=int(refused)
                if rag:n['resolved']+=check['n_supported'];n['citations']+=check['n_total']
                ident=core.digest({'model':r['model'],'condition':condition,'qid':q})[:24]
                if ident in protected and ident not in chosen:n['quarantined']+=1
                if ident not in chosen:continue
                p,v=chosen[ident];n['judged']+=1;assign[p]+=1
                vals['correctness'].append(v['correctness']);vals['reference_support'].append(v['evidence_support'])
                scores[(model_key,condition)][q]=v['correctness']
                if rag:vals['citation_support'].extend(c['support'] for c in v.get('citation_assessments',[]))
            row={'model_key':model_key,'model':model_id,'condition':condition,'available':available,'benchmark_questions':76,
                 'responses':len(responses),'judged':n['judged'],'missing':76-n['judged'] if available else None,
                 'quarantined':n['quarantined'],'eligible_unjudged':76-n['judged']-n['quarantined'] if available else None,
                 'groq_selected':assign['groq'],'gemini_selected':assign['gemini'],'checkpoint':version,'protocol':PROTOCOL}
            def metric(name,numerator,denominator,ci=None,description=''):
                value=numerator/denominator if available and denominator else None
                row[name]=value;row[name+'_numerator']=numerator if available else None;row[name+'_denominator']=denominator if available else 0
                lo,hi=ci if ci else (None,None)
                if ci:row[name+'_ci_low']=lo;row[name+'_ci_high']=hi
                long.append({'model':model_id,'model_key':model_key,'condition':condition,'metric':name,'value':value,
                             'numerator':numerator if available else None,'denominator':denominator if available else 0,
                             'benchmark_questions':76,'judged':n['judged'],'missing':row['missing'],'quarantined':n['quarantined'],
                             'available':available,'ci_low':lo,'ci_high':hi,'ci_method':'conditional IID question bootstrap; 2000 draws; seed20261102' if ci else None,
                             'definition':description,'checkpoint':version,'archive_sha256':archive_sha,'protocol':PROTOCOL})
            for name in ('correctness','reference_support','citation_support'):
                value,num,den=mean_parts(vals[name]);ci=bootstrap_mean(vals[name]) if name=='correctness' and available else None
                if name=='citation_support' and not rag:num,den=0,0
                metric(name,num,den,ci,'Mean of non-null valid semantic scores; missing/uncertain scores excluded, not zero')
            metric('parse_rate',n['parse'],76,'','Original inherited parse definition: recognised abstention OR at least one parseable citation; not JSON-output compliance')
            metric('citation_resolution',n['resolved'],n['citations'] if rag else 0,None,'Citation IDs resolving against fixed retrieved chunks; does not establish entailment')
            metric('appropriate_refusal_rate',n['appropriate_refusal'],9,None,'Unanswerable questions refused / 9')
            metric('false_refusal_rate',n['false_refusal'],67,None,'Answerable questions refused / 67')
            metric('refusal_correctness',n['appropriate_refusal'],n['appropriate_refusal']+n['false_refusal'],None,'Appropriate refusals / all refusals')
            metric('balanced_abstention',(67-n['false_refusal'])*9+n['appropriate_refusal']*67,2*67*9,None,'0.5 * (answerable non-refusal /67 + unanswerable refusal /9)')
            metric('coverage',n['judged'],76,None,'At least one accepted semantic judgement / 76')
            comparison.append(row)
    effects=[]
    for model_key,spec in cfg['models'].items():
        for before,after in (('base','sft'),('base-rag','sft-rag')):
            a=scores[(model_key,before)];b=scores[(model_key,after)]
            common=[q for q in a if q in b and a[q] is not None and b[q] is not None]
            delta=[b[q]-a[q] for q in common];value,num,den=mean_parts(delta);lo,hi=bootstrap_mean(delta)
            effects.append({'model_key':model_key,'model':spec['id'],'before':before,'after':after,'paired_scored_questions':den,
                            'correctness_delta':value,'delta_sum':num,'ci_low':lo,'ci_high':hi,'checkpoint':version,
                            'interpretation':'Common-question paired descriptive sensitivity analysis, not an extra judging protocol'})
    assert sum(r['judged'] for r in comparison)==len(chosen)
    assert sum(r['responses'] for r in comparison)==912
    private=root/'reporting_d1000';private.mkdir(exist_ok=True)
    (private/'protected_task_ids.json').write_text(json.dumps(protected_private))
    payload={'schema_version':'d1000-safe-aggregates-v1','provenance':{'checkpoint':version,'archive_sha256':archive_sha,
        'protocol':PROTOCOL,'status':'FULLY_JUDGED' if len(chosen)==912 else 'PARTIAL',
        'verified_epoch':time.time(),'reporting_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'responses':912,'unique_judged':len(chosen),'remaining':912-len(chosen),
        'eligible_remaining':912-len(chosen)-len(protected-set(chosen)),'quarantined_unjudged':len(protected-set(chosen)),
        'protected_requests':len(protected),'provider_votes':{p:len(votes[p]) for p in MODELS},
        'selected_provider_counts':dict(collections.Counter(p for p,v in chosen.values())),
        'active_judge_calls':0,'active_coordinators':0,'writer_lease_version':recovery.lease_version,
        'immutable_manifest_sha256':recovery.evaluation.verify_unchanged(),'benchmark_questions':76,
        'answerable':67,'unanswerable':9,'completed_adapters':3,'immutable_files':80,
        'calibration':{'groq':'PASS9/9','gemini':'PASS9/9','gemini31':'FAIL8/9; zero research votes'},
        'remaining_gemini_terminal_states':dict(missing_states),'quota_audit':quota_audit(runtime,providers),
        'training_config':cfg.get('training',{}),'source_quality':{'source_records':1000,'verified':462,'unverified':538,'train':751,'validation':83},
        'limitations':['Mixed-provider single-valid-judge evaluation; no dual-judge consensus or human review',
        'Semantic means condition on valid non-null scores; missingness may bias comparisons',
        'Bootstrap intervals describe sampled questions only, not judge/model uncertainty',
        'Inherited parse rate is citation/refusal recognition, not JSON structured-output compliance',
        'Four DeepSeek conditions unavailable; source quality mixed; frozen benchmark is small']},
        'comparison':comparison,'metrics':long,'sft_effects':effects}
    (private/'safe_aggregates.json').write_text(json.dumps(payload,sort_keys=True,indent=2))
    return payload
