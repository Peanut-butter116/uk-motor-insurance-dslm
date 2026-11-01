"""One writer, disjoint provider queues, original strict completion and replay guards."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from contextlib import redirect_stdout,redirect_stderr
from cloud.single_judge_fast.protocol import PROTOCOL,MODELS,load_validated,unknown_ids,quota_audit,feasibility,select
from cloud.single_judge_fast import metrics
from cloud.single_judge_fast.handoff import old_workers


def writer_guard(recovery):
    from cloud.d1000.storage import verify_file
    if old_workers():raise RuntimeError('Historical worker still active')
    if not hasattr(recovery,'fast_handoff') or not recovery.started or not recovery.ready:
        raise RuntimeError('Verified drained handoff required')
    if not Path('/kaggle/working/d1000-authoritative-writer.lock').is_file():raise RuntimeError('Existing writer lock required')
    lease='scarletthe0116/uk-motor-d1000-evaluation-lease-v1'
    value=recovery.info(recovery.api,lease)
    if not value.is_private or value.current_version_number!=recovery.lease_version:raise RuntimeError('Lease changed')
    owner=json.dumps({'session':recovery.session,'source_checkpoint':239}).encode()
    verify_file(lease,recovery.lease_version,'owner.json',hashlib.sha256(owner).hexdigest())
    if recovery.info(recovery.api,'scarletthe0116/uk-motor-d1000-models-results').current_version_number!=recovery.state['last_persistence']['version']:
        raise RuntimeError('Concurrent checkpoint writer')
    if recovery.api.kernels_status('scarletthe0116/uk-motor-d1000-train-eval').status.name not in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'):
        raise RuntimeError('Active Kaggle writer')
    recovery.evaluation.verify_unchanged()


def prepare(recovery):
    from cloud.d1000 import judging,judge_recovery as jr
    writer_guard(recovery);root=Path(recovery.evaluation.ROOT)
    providers,runtime=judging.provider_runtime(root)
    from cloud.single_judge_fast.quotas import install
    install(providers)
    candidates=judging.tasks(root)
    if len(candidates)!=912 or len({t['evaluation_id'] for t,_ in candidates})!=912:raise RuntimeError('Population differs')
    votes=load_validated(root,candidates);covered=select(votes)
    unknown={p:unknown_ids(root,candidates,p) for p in MODELS}
    for p in MODELS:
        calibration=json.loads((root/'judging'/f'{p}-calibration-v2.json').read_text());jr.verify_calibration(calibration)
        catalogue,_=providers.transport({k:os.environ[k] for k in ('GROQ_API_KEY','GEMINI_API_KEY')},p)
        if not providers.catalogue_contains(p,MODELS[p],catalogue):raise RuntimeError('Exact model unavailable')
    costs={};estimates={}
    for p in MODELS:
        costs[p]={};prompts=[]
        for task,_ in candidates:
            if task['evaluation_id'] in covered:continue
            body=jr.body(p,MODELS[p],task)
            counted=body['messages']+[{'role':'user','content':json.dumps(body['response_format'])}]
            reservation=providers.token_reservation(p,MODELS[p],counted,2048)
            costs[p][task['evaluation_id']]=reservation
            # Planning estimate only. Actual calls still use exact original reservation.
            prompts.append(max(1,reservation-2048) if p=='groq' else max(1,(reservation-6144)/4))
        estimates[p]={'median_prompt_estimate':statistics.median(prompts) if prompts else 0,
                      'minimum_reservation':min(costs[p].values(),default=0),'maximum_reservation':max(costs[p].values(),default=0)}
    audit=quota_audit(runtime,providers);forecast=feasibility(audit,912-len(covered),estimates)
    report={'protocol':PROTOCOL,'status':'READY_NOT_RUNNING','source_checkpoint':recovery.state['last_persistence']['version'],
            'unique_judged':len(covered),'remaining':912-len(covered),'provider_votes':{p:len(votes[p]) for p in MODELS},
            'protected_task_counts':{p:len(unknown[p]) for p in MODELS},'quota_audit':audit,'token_estimates':estimates,
            'feasibility':forecast,'selection':'Groq first if both valid; otherwise Gemini; never request a second vote',
            'concurrency_per_provider':1,'judge_calls_in_preflight':0}
    recovery.fast_prepared={'candidates':candidates,'votes':votes,'unknown':unknown,'costs':costs,'report':report}
    return report


def run(recovery,sink):
    if getattr(recovery,'fast_started',False):raise RuntimeError('Fast coordinator already started')
    if not hasattr(recovery,'fast_prepared'):raise RuntimeError('Fast preflight required')
    writer_guard(recovery);recovery.fast_started=True
    with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        return asyncio.run(coordinate(recovery,sink))


async def run_async(recovery,sink,prelaunch_failure=None,ledger_hashes=None):
    """Colab already owns its event loop; continue only an untouched loop-entry failure."""
    if getattr(recovery,'fast_started',False):
        if not isinstance(prelaunch_failure,RuntimeError) or str(prelaunch_failure)!='asyncio.run() cannot be called from a running event loop':
            raise RuntimeError('Fast coordinator already started')
        if hasattr(recovery,'fast_control'):
            raise RuntimeError('Protocol already launched; cannot repeat entry')
        if not ledger_hashes or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v for p,v in ledger_hashes.items()):
            raise RuntimeError('Quota state changed; reconcile before continuation')
    elif prelaunch_failure is not None:
        raise RuntimeError('Unexpected prelaunch continuation')
    if not hasattr(recovery,'fast_prepared'):raise RuntimeError('Fast preflight required')
    writer_guard(recovery);recovery.fast_started=True
    with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        return await coordinate(recovery,sink)


async def coordinate(recovery,sink):
    from cloud.d1000 import judging,judge_recovery as jr
    from experiments import core
    root=Path(recovery.evaluation.ROOT);folder=root/'single_judge_fast_v1';folder.mkdir(exist_ok=True)
    prepared=recovery.fast_prepared;votes=prepared['votes'];candidates=prepared['candidates']
    unknown=prepared['unknown'];costs=prepared['costs'];covered=set(select(votes));baseline=len(covered)
    providers,runtime=judging.provider_runtime(root);values={k:os.environ[k] for k in ('GROQ_API_KEY','GEMINI_API_KEY')}
    launched=time.time();deadline=launched+20*3600
    control={'protocol':PROTOCOL,'launch_epoch':launched,'deadline_epoch':deadline,'baseline_unique':baseline,
             'source_checkpoint':recovery.state['last_persistence']['version'],'same_lease_version':recovery.lease_version,
             'historical_protocol':jr.PROTOCOL,'selection_rule':'Groq-first historical tie; otherwise first valid provider',
             'rubric_unchanged':True,'unknown_outcome_guard':'task/provider scoped; original complete() unchanged',
             'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (folder/'protocol.json').write_text(json.dumps(control,sort_keys=True,indent=2))
    gate=asyncio.Event();gate.set();claims={};active=set();failed={p:set() for p in MODELS};status={p:{'status':'ready'} for p in MODELS}
    recovery.fast_control=control;recovery.state['single_judge_fast']=control
    new_since_checkpoint=0;stopping=False
    def append(path,value):
        with path.open('a') as f:f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
    def progress():
        total=len(covered);elapsed=max(0,time.time()-launched);rate=(total-baseline)/(elapsed/3600) if elapsed else 0
        audit=quota_audit(runtime,providers)
        remaining=912-total;eta=remaining/rate if rate>0 else None
        current_projected=0
        for p,q in audit.items():
            cost=max(q.get('historical_mean_actual_tokens') or 0,prepared['report']['token_estimates'][p]['median_prompt_estimate'])
            current_projected+=min(q['remaining_conservative_ledger']['rpd'],int(q['remaining_conservative_ledger']['tpd']/cost)) if cost else 0
        reset_wait=max(0,audit.get('gemini',{}).get('documented_calendar_reset_epoch',time.time())-time.time())/3600 if audit else 0
        if eta is not None and remaining>current_projected:
            eta=max(eta,reset_wait+(remaining-current_projected)/rate)
        return {'protocol':PROTOCOL,'unique_judged':total,'remaining':912-total,
                'provider_votes':{p:len(votes[p]) for p in MODELS},'new_unique_since_launch':total-baseline,
                'successful_judgements_per_hour':rate,'estimated_hours_remaining':eta,
                'eta_interpretation':'Measured successful unique throughput plus projected calendar quota wait; errors, repairs and runtime availability may worsen this estimate',
                'elapsed_hours':elapsed/3600,'launch_epoch':launched,'deadline_epoch':deadline,
                'provider_status':status,'unresolved_requests':{p:len(unknown[p]) for p in MODELS},
                'quota_constraints':audit,'active_provider_calls':len(active)}
    async def checkpoint():
        nonlocal new_since_checkpoint
        gate.clear()
        while active:await asyncio.sleep(.2)
        report=progress();summary=metrics.summarise(root,votes)
        report['selected_provider_counts']=summary['selected_provider_counts']
        recovery.state['single_judge_fast'].update(progress=report,metrics=summary)
        recovery.state['status']='single_judge_complete' if len(covered)==912 else 'single_judge_partial'
        (folder/'progress.json').write_text(json.dumps(report,sort_keys=True,indent=2))
        writer_guard(recovery)
        # No provider calls or journal appends can race the snapshot.
        await asyncio.to_thread(recovery.evaluation.checkpoint,recovery.state)
        archive=Path('/kaggle/working/publish/artifacts.bin')
        stored=recovery.fast_archive_audit(archive)
        expected={p:len(votes[p]) for p in MODELS}
        if stored['persisted_judge_counts']!=expected or any(stored['persisted_duplicate_votes'].values()):raise RuntimeError('Snapshot vote journal mismatch')
        report.update(event='single_judge_private_checkpoint_readback',checkpoint=recovery.state['last_persistence']['version'],
                      private=True,readback_verified=True,archive_sha256=recovery.state['artifacts_sha256'],persisted_judge_counts=expected)
        sink(report);new_since_checkpoint=0;gate.set()
    async def worker(p):
        nonlocal new_since_checkpoint
        model=MODELS[p]
        while not stopping and len(covered)<912 and time.time()<deadline-900 and not (folder/'STOP_REQUESTED').exists():
            await gate.wait()
            if stopping:return
            limits=providers.verify_receipt(runtime,p,model)
            eligible=[t for t,m in candidates if t['evaluation_id'] not in covered and t['evaluation_id'] not in claims and t['evaluation_id'] not in unknown[p] and t['evaluation_id'] not in failed[p] and costs[p].get(t['evaluation_id'],10**20)<=limits['tpm']]
            if not eligible:
                status[p]={'status':'no_safe_unclaimed_tasks'};await asyncio.sleep(1);continue
            task=min(eligible,key=lambda t:(costs[p][t['evaluation_id']],t['evaluation_id']))
            ident=task['evaluation_id'];claims[ident]=p;active.add(p)
            try:
                # Strict original body, validators, cache identity, reservations and
                # durable raw-before-success journalling remain unchanged.
                record=await asyncio.to_thread(jr.complete,values,p,model,task,runtime,providers,
                        transport=lambda v,provider,actual:jr.observed_transport(v,provider,actual,runtime,providers))
                if ident in covered:raise RuntimeError('Duplicate cross-provider assignment')
                row={**record['answer'],'provider':p,'model':model,'request_sha256':record['request_sha256'],
                     'usage':record['usage'],'input_sha256':record['input_sha256'],'evaluation_protocol':PROTOCOL}
                append(root/'judging'/f'{p}.jsonl',row);votes[p][ident]=row;covered.add(ident)
                append(folder/'new_assignments.jsonl',{'evaluation_id':ident,'provider':p,'request_sha256':record['request_sha256'],'completed_epoch':time.time()})
                new_since_checkpoint+=1;status[p]={'status':'running','completed_votes':len(votes[p])}
            except providers.QuotaPause as exc:
                status[p]={'status':'quota_wait','resume_after_epoch':exc.until}
            except Exception as exc:
                reason=str(exc)
                unknown[p].update(unknown_ids(root,candidates,p));failed[p].add(ident)
                category=('unknown_outcome_protected' if 'Unknown request outcome' in reason else
                          'known_invalid_exhausted' if 'Known-invalid judgement exhausted' in reason else
                          'transport_uncertain_provider_stopped' if ident in unknown[p] else 'provider_or_validation_failure')
                append(folder/'safe_failures.jsonl',{'evaluation_id':ident,'provider':p,'category':category,'epoch':time.time()})
                status[p]={'status':'blocked' if category in ('transport_uncertain_provider_stopped','provider_or_validation_failure') else 'task_skipped','category':category}
                if status[p]['status']=='blocked':return
            finally:
                active.discard(p);claims.pop(ident,None)
            until=status[p].get('resume_after_epoch',0)
            while time.time()<until and time.time()<deadline-900 and not stopping and not (folder/'STOP_REQUESTED').exists():
                await asyncio.sleep(min(10,max(.2,until-time.time())))
    await checkpoint() # protocol, metrics, lease, baseline persisted before new calls
    workers=[asyncio.create_task(worker(p),name='fast-'+p) for p in MODELS];last=time.monotonic()
    try:
        while len(covered)<912 and time.time()<deadline-900 and not (folder/'STOP_REQUESTED').exists():
            if all(w.done() for w in workers):break
            if not claims and all(status[p]['status'] in ('blocked','no_safe_unclaimed_tasks') for p in MODELS):break
            if new_since_checkpoint>=5 or time.monotonic()-last>=300:
                await checkpoint();last=time.monotonic()
            await asyncio.sleep(1)
        stopping=True;gate.set()
        # Drain calls naturally, including their original timeout and durable
        # outcome handling; never cancel a running provider request.
        await asyncio.gather(*workers)
        await checkpoint()
        return {'status':'COMPLETE' if len(covered)==912 else 'PARTIAL_STOPPED','unique_judged':len(covered),
                'remaining':912-len(covered),'checkpoint':recovery.state['last_persistence']['version']}
    except BaseException:
        stopping=True;gate.set()
        await asyncio.gather(*workers,return_exceptions=True)
        raise
    finally:
        stopping=True;gate.set()
