"""Single-valid-judge selection and read-only quota planning."""
import collections
import json
from pathlib import Path
import statistics
import time

PROTOCOL='d1000-single-judge-fast-v1'
MODELS={'groq':'openai/gpt-oss-120b','gemini':'gemini-3.5-flash-lite'}
UNKNOWN=('in_flight','transport_uncertain','response_saved')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def select(votes):
    """Deterministic Groq-first selection. Retain both source journals unchanged."""
    selected={}
    for provider in ('gemini','groq'):
        for ident,vote in votes[provider].items():selected[ident]=(provider,vote)
    return selected


def load_validated(root,candidates):
    from cloud.d1000 import judging,judge_recovery as jr
    from experiments import core
    root=Path(root);tasks={t['evaluation_id']:t for t,_ in candidates};votes={p:{} for p in MODELS}
    for provider in MODELS:
        for vote in rows(root/'judging'/f'{provider}.jsonl'):
            ident=vote['evaluation_id']
            if ident not in tasks or ident in votes[provider]:raise RuntimeError('Unexpected/duplicate vote')
            task=tasks[ident]
            key=core.digest({'task':task,'provider':provider,'model':MODELS[provider],'protocol':jr.PROTOCOL})
            cache=root/'judge_runtime/evaluation_cache_v2'/provider/(key+'.json')
            record=json.loads(cache.read_text())
            if record['protocol']!=jr.PROTOCOL or record['input_sha256']!=core.digest(task) or record['generation_config']['model']!=MODELS[provider]:
                raise RuntimeError('Saved judgement identity mismatch')
            result={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'evaluations':[record['indexed_answer']]})}}]}
            canonical=jr.validate(result,task,judging.provider_runtime(root)[0])
            if canonical!=record['answer'] or any(vote.get(k)!=v for k,v in canonical.items()):
                raise RuntimeError('Journal and strictly validated cache differ')
            votes[provider][ident]=vote
    return votes


def unknown_ids(root,candidates,provider):
    from cloud.d1000 import judge_recovery as jr
    from experiments import core
    folder=Path(root)/'judge_runtime/evaluation_cache_v2'/provider;result=set()
    for task,_ in candidates:
        key=core.digest({'task':task,'provider':provider,'model':MODELS[provider],'protocol':jr.PROTOCOL})
        path=folder/(key+'.attempts.json')
        history=json.loads(path.read_text()) if path.exists() else []
        if history and history[-1]['status'] in UNKNOWN:result.add(task['evaluation_id'])
    return result


def quota_audit(runtime,providers,now=None,hours=20):
    from cloud.single_judge_fast.quotas import pacific_window
    now=time.time() if now is None else now;result={}
    for provider,model in MODELS.items():
        limits=providers.verify_receipt(runtime,provider,model)
        ledger=json.loads((Path(runtime)/'provider_usage'/f'{provider}.json').read_text())
        current=[e for e in ledger['entries'] if e['time']>now-86400]
        minute=[e for e in current if e['time']>now-60]
        usage=[providers.daily_tokens(e) for e in ledger['entries'] if e.get('status')=='usage_recorded']
        remaining={'rpm':max(0,limits['rpm']-len(minute)), 'rpd':max(0,limits['rpd']-len(current)),
                   'tpm':max(0,limits['tpm']-sum(e['quota_tokens'] for e in minute)),
                   'tpd':max(0,limits['tpd']-sum(providers.daily_tokens(e) for e in current))}
        expiring=[e for e in current if e['time']+86401<=now+hours*3600]
        slots=max(0,limits['rpd']-len(current))+len(expiring)
        tokens=max(0,limits['tpd']-sum(providers.daily_tokens(e) for e in current))+sum(providers.daily_tokens(e) for e in expiring)
        start,reset=pacific_window(now)
        if provider=='gemini':
            day=[e for e in ledger['entries'] if e['time']>=start]
            remaining['rpd']=max(0,limits['rpd']-len(day))
            slots=remaining['rpd']+(limits['rpd'] if reset+1<=now+hours*3600 else 0)
        result[provider]={'verified_limits':{k:limits[k] for k in ('rpm','rpd','tpm','tpd')},
             'remaining_conservative_ledger':remaining,'blocked_until_epoch':ledger.get('blocked_until',0),
             'next_minute_entry_expiry':min((e['time']+61 for e in minute),default=None),
             'next_daily_entry_expiry':min((e['time']+86401 for e in current),default=None),
             'twenty_hour_request_slots_upper_bound':slots,'twenty_hour_token_budget':tokens,
             'historical_mean_actual_tokens':statistics.mean(usage) if usage else None,
             'server_remaining_not_independently_known':True,
             'documented_calendar_reset_epoch':reset if provider=='gemini' else None,
             'quota_policy':'Gemini RPD midnight Pacific; RPM/TPM rolling60s and TPD rolling24h; all entries retained' if provider=='gemini' else 'existing conservative rolling60s/24h; reservations unchanged'}
    return result


def feasibility(audit,remaining,token_estimates):
    # Project with measured historical actual usage; do not equate projections to guarantees.
    capacity={}
    for p,q in audit.items():
        mean=q['historical_mean_actual_tokens']
        cost=max(mean or 0,token_estimates[p]['median_prompt_estimate'])
        capacity[p]=min(q['twenty_hour_request_slots_upper_bound'],int(q['twenty_hour_token_budget']/cost)) if cost else 0
    return {'remaining':remaining,'twenty_hour_projected_success_capacity_no_repairs':capacity,
            'quota_projection_feasible':sum(capacity.values())>=remaining,
            'request_slots_hard_upper_bound':sum(q['twenty_hour_request_slots_upper_bound'] for q in audit.values()),
            'interpretation':'Projection uses historical actual usage and current task estimates. Repairs, errors, shared account traffic and runtime limits can reduce coverage; not a completion promise.'}
