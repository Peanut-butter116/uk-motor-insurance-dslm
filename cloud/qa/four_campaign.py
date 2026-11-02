"""Guarded four-model extension of the existing unverified cloud coordinator."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import fast_campaign as fast
from runner import require_cloud, utc
from state import write_json

PREDECESSOR = 'scarletthe0116/uk-motor-qa-resume/14'
FREE_CANDIDATES = {
    'groq': ('openai/gpt-oss-20b', 'qwen/qwen3.8-27b'),
    'gemini': ('gemini-3.1-flash-lite', 'gemini-3.8-flash')}


def select_specs(listings):
    """No aliases or paid-model guesses: select from explicit free-tier allowlist."""
    existing = fast.MODELS
    for p, model in existing.items():
        if model not in listings[p]:
            raise RuntimeError('Existing model missing from authenticated catalogue: '+p)
    extras = {}
    for provider, candidates in FREE_CANDIDATES.items():
        extras[provider] = next((m for m in candidates if m in listings[provider]), None)
        if not extras[provider]:
            raise RuntimeError('No approved distinct free model in authenticated catalogue: '+provider)
    return {'groq': {'provider':'groq', 'model':existing['groq']},
            'gemini': {'provider':'gemini', 'model':existing['gemini']},
            'groq_extra': {'provider':'groq', 'model':extras['groq']},
            'gemini_extra': {'provider':'gemini', 'model':extras['gemini']}}


async def catalogue_preflight():
    """GET-only account-scoped listings; no policy text or generation request."""
    import httpx
    bases = {'groq':'https://api.groq.com/openai/v1',
             'gemini':'https://generativelanguage.googleapis.com/v1beta/openai'}
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        async def get(provider):
            response = await client.get(bases[provider]+'/models', headers={
                'Authorization':'Bearer '+os.environ[provider.upper()+'_API_KEY']})
            if response.status_code != 200:
                raise RuntimeError('Authenticated model listing failed: '+provider+' HTTP '+str(response.status_code))
            body = response.json()
            names = {m['id'].removeprefix('models/') for m in body.get('data', [])}
            return provider, names, hashlib.sha256(response.content).hexdigest()
        results = await asyncio.gather(*(get(p) for p in bases))
    listings = {p:names for p,names,_ in results}
    specs = select_specs(listings)
    receipt = {'checked_at_utc':utc(), 'authenticated_catalogue_verified':True,
               'workers':specs, 'catalogue_sha256':{p:h for p,_,h in results},
               'generation_or_judge_calls':0,
               'free_tier_sources':['https://console.groq.com/docs/rate-limits',
                                    'https://ai.google.dev/gemini-api/docs/pricing'],
               'substitutions':{p:s['model'] for w,s in specs.items() if w.endswith('_extra')
                                for p in [s['provider']] if s['model'] != FREE_CANDIDATES[p][0]}}
    write_json(Path('/kaggle/working/four_model_catalogue.json'), receipt)
    return specs, receipt


async def wait_for_predecessor(api, deadline_seconds=5*3600):
    """Read-only waiting is allowed alongside the old writer; never cancel it."""
    end = time.monotonic()+deadline_seconds
    while time.monotonic() < end:
        status = api.kernels_status(PREDECESSOR).status.name
        write_json(Path('/kaggle/working/four_handoff.json'), {
            'checked_at_utc':utc(), 'predecessor':PREDECESSOR, 'status':status,
            'new_writer_started':False, 'generation_calls':0})
        if status == 'COMPLETE':return
        if status not in ('RUNNING','QUEUED','CANCEL_REQUESTED'):
            raise RuntimeError('Predecessor did not finish cleanly; handoff refused: '+status)
        await asyncio.sleep(30)
    raise RuntimeError('Predecessor still active; read-only handoff wait ended')


def authorise_models(providers, run, specs, catalogue):
    """Preserve account attestations and all provider-wide limits across models.

    Original receipts stay unchanged. The new sidecar truthfully records live
    catalogue checks; structured generation compatibility is tested by smoke QA,
    not falsely inherited from the previous model's synthetic probe.
    """
    original = providers.verify_receipt
    parent = {p: original(run, p, model) for p,model in fast.MODELS.items()}
    allowed = {(s['provider'], s['model']) for s in specs.values()}
    def verify(path, provider, model):
        if (provider, model) not in allowed:
            raise RuntimeError('Unapproved four-worker model')
        account = original(path, provider, fast.MODELS[provider])
        return {**account, 'model':model, 'model_catalogue_verified':True,
                'synthetic_json_probe_passed':model == fast.MODELS[provider],
                'generation_compatibility':'controlled smoke generation; no QA judge',
                'catalogue_sha256':catalogue['catalogue_sha256'][provider],
                'shared_provider_quota':True}
    providers.verify_receipt = verify
    return original


class Retry503Transport(fast.ReusedTransport):
    """Retry only explicit HTTP503 replies; retain all additional reservations."""
    def __call__(self, values, provider, body=None):
        for attempt in range(3):
            if attempt:
                limits=self.providers.verify_receipt(self.run_path,provider,body['model'])
                quota=self.providers.Quotas(self.run_path/'provider_usage'/(provider+'.json'),provider,limits)
                quota.reserve(self.providers.digest({'http_503_retry':attempt,'body':body,'at':time.time()}),
                    self.providers.token_reservation(provider,body['model'],body['messages'],body['max_tokens']))
            try:return super().__call__(values,provider,body)
            except self.safe_error as error:
                if str(error)!=provider+' HTTP 503' or attempt==2:raise
                time.sleep(15*2**attempt)


def recover_known_503(run, generation, packages, providers, contract):
    """Match a known 503 to exactly one interrupted assigned request; never blanket-clear."""
    path=run/'fast_unverified_v1/state.json';state=json.loads(path.read_text())
    lookup={p['id']:p for p in packages}
    _,schema=contract('teacher',generation.TEACHER_PROMPT)
    for provider in fast.MODELS:
        ps=state['providers'][provider]
        if ps.get('error')!=provider+' HTTP 503' or ps.get('status')!='blocked':continue
        matches=[]
        for key,job in state['jobs'].items():
            if job['provider']!=provider or job['status']!='assigned' or not job.get('started_epoch'):continue
            package=lookup[key]
            body={'model':fast.MODELS[provider], 'messages':[{'role':'system','content':fast.SYSTEM},
                {'role':'user','content':generation.TEACHER_PROMPT+package['context']}],
                'temperature':0,'max_tokens':4096,'stream':False,'response_format':{'type':'json_object'}}
            ident=providers.digest({'provider':provider,'body':body,'schema':schema,'stage':'fast_generation_unverified_v1'})
            history_path=run/'provider_cache'/provider/(ident+'.attempts.json')
            if not history_path.exists():continue
            history=json.loads(history_path.read_text())
            if history and history[-1]['status']=='transport_uncertain':
                raw=history_path.parent/(history[-1]['request_sha256']+'.raw.json')
                if raw.exists():raise RuntimeError('Known503 has raw response; inspect before recovery')
                matches.append((key,history_path,history))
        if len(matches)!=1:raise RuntimeError('Known503 does not match one assigned request; recovery refused')
        key,history_path,history=matches[0]
        write_json(run/'fast_unverified_v1'/(provider+'-503-recovery.json'),{
            'provider':provider,'package_id':key,'old_error':ps['error'],'original_attempts':history,
            'authority':'Explicit HTTP503 is a rejected request; quota reservations retained',
            'recovered_at_utc':utc()})
        write_json(history_path,history[:-1])
        ps.update(status='quota_wait',resume_after_epoch=max(time.time()+15,ps.get('resume_after_epoch',0)))
        ps.pop('error',None)
    write_json(path,state)


async def run():
    require_cloud()
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for key in ('GROQ_API_KEY','GEMINI_API_KEY','KAGGLE_API_TOKEN'):
        os.environ[key] = secrets.get_secret(key)
        if not os.environ[key]:raise RuntimeError('Required Kaggle secret unavailable: '+key)
    from kaggle import api
    specs, catalogue = await catalogue_preflight()
    await wait_for_predecessor(api)
    # Always restore AFTER the old writer finishes, never the stale mounted input.
    session, version = fast.restore(api)
    source = session/'uk-insurance-QA-data';sys.path.insert(0,str(source))
    from src import providers
    from src.response_contracts import contract
    import importlib.util
    from types import SimpleNamespace
    run_path = session/'uk-insurance-QA-data-local/production-v1'
    state_path = run_path/'fast_unverified_v1/state.json'
    prior = json.loads(state_path.read_text())
    if prior.get('active_dispatch'):
        raise RuntimeError('Previous dispatch unresolved; no four-worker generation started')
    audit = run_path/'fast_unverified_v1/four_model_catalogue.json'
    write_json(audit,catalogue)
    spec=importlib.util.spec_from_file_location('four_recovery_generation',source/'scripts/generate_motor.py')
    generation=importlib.util.module_from_spec(spec);spec.loader.exec_module(generation)
    packages=generation.packages(SimpleNamespace(run=run_path,docs={}))
    recover_known_503(run_path,generation,packages,providers,contract)
    dest = session/'cloud-adapter';dest.mkdir(exist_ok=True)
    for name in ('four_campaign.py','fast_campaign.py'):
        shutil.copyfile(Path(__file__).parent/name,dest/name)
    original = authorise_models(providers,run_path,specs,catalogue)
    original_transport=fast.ReusedTransport
    fast.ReusedTransport=Retry503Transport
    try:
        await fast.campaign(session,api,version,worker_specs=specs)
    finally:
        providers.verify_receipt = original
        fast.ReusedTransport=original_transport


if __name__ == '__main__':asyncio.run(run())
