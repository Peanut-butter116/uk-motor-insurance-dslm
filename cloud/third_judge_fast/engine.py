"""Same-lease third-model recovery with original strict complete()/calibration.

The new model uses a separate cache and journal, but the SAME Gemini account
quota ledger. No new quota allowance is inferred from changing model identity.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace
from contextlib import redirect_stdout, redirect_stderr
from cloud.third_judge_fast.protocol import PROTOCOL, MODEL, PROVIDER, select, eligible, calibration_pass
from cloud.single_judge_fast.protocol import MODELS, load_validated, unknown_ids, rows
from cloud.single_judge_fast.engine import writer_guard
from cloud.single_judge_fast.lifecycle import provider_threads


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value)+'\n'); stream.flush(); os.fsync(stream.fileno())


def original_manifest(root):
    return {str(p.relative_to(root)):sha(p) for directory in ('judging','judge_runtime/evaluation_cache_v2')
            for p in (root/directory).rglob('*') if p.is_file()}


def preserved(root, manifest):
    if any(not (root/p).is_file() or sha(root/p)!=h for p,h in manifest.items()):
        raise RuntimeError('Historical vote/cache state changed')


def adapter(providers, runtime, shared_runtime, proof):
    """An explicit new-model verification receipt, never an original receipt edit."""
    def verify(actual_runtime, provider, model):
        if Path(actual_runtime)!=runtime or provider!=PROVIDER or model!=MODEL:
            raise RuntimeError('Unreviewed third judge identity')
        if proof.get('free_tier') is not True or proof.get('catalogue_verified') is not True:
            raise RuntimeError('Free capacity and catalogue proof required')
        if proof.get('model')!=MODEL or proof.get('limits')!={'rpm':15,'rpd':500,'tpm':250000,'tpd':125000000}:
            raise RuntimeError('Verified limits differ')
        # TPD is the inherited conservative cap, not an independently verified
        # extra allowance. The complete old ledger is used by both models.
        return proof['limits']
    def quotas(path, provider, limits):
        if provider!=PROVIDER:raise RuntimeError('Unexpected quota provider')
        return providers.Quotas(shared_runtime/'provider_usage/gemini.json',provider,limits)
    result=SimpleNamespace(**{name:getattr(providers,name) for name in dir(providers) if not name.startswith('__')})
    result.verify_receipt=verify;result.Quotas=quotas
    return result


def prepare(recovery, proof):
    from cloud.d1000 import judging, judge_recovery as jr
    if recovery.fast_control.get('phase')!='suspended' or not hasattr(recovery,'fast_suspension'):
        raise RuntimeError('Controlled old coordinator suspension required')
    if provider_threads() or any(t.get_name() in ('fast-groq','fast-gemini') and not t.done() for t in asyncio.all_tasks()):
        raise RuntimeError('Old provider work remains')
    if getattr(recovery,'third_started',False):raise RuntimeError('Third coordinator already started')
    writer_guard(recovery)
    root=Path(recovery.evaluation.ROOT);folder=root/'third_judge_fast_v1';folder.mkdir(exist_ok=True)
    providers,shared=judging.provider_runtime(root)
    from cloud.single_judge_fast.quotas import install
    install(providers)
    candidates=judging.tasks(root)
    if len(candidates)!=912:raise RuntimeError('Response population differs')
    votes=load_validated(root,candidates);votes['gemini31']={}
    protected=set().union(*(unknown_ids(root,candidates,p) for p in MODELS))
    if len(unknown_ids(root,candidates,'gemini'))!=4:raise RuntimeError('Protected request population changed; reconcile')
    runtime=folder/'runtime';runtime.mkdir(exist_ok=True)
    bridge=adapter(providers,runtime,Path(shared),proof);bridge.verify_receipt(runtime,PROVIDER,MODEL)
    manifest=original_manifest(root)
    ledger=json.loads((Path(shared)/'provider_usage/gemini.json').read_text())
    state={'protocol':PROTOCOL,'provider':PROVIDER,'model':MODEL,'phase':'prepared',
           'source_checkpoint':recovery.state['last_persistence']['version'],
           'original_deadline_epoch':recovery.fast_control['deadline_epoch'],
           'lease_version':recovery.lease_version,'quota_policy':'same existing Gemini ledger; no quota reset',
           'protected_requests':4,'protected_unjudged':len(protected-set(select(votes))),
           'eligible':len(eligible(candidates,votes,protected)),
           'methodology':'Mixed-provider mixed-model single-judge; original rubric and strict validators unchanged',
           'source_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'capacity_evidence':proof}
    (folder/'protocol.json').write_text(json.dumps(state,indent=2,sort_keys=True))
    (folder/'historical_manifest.json').write_text(json.dumps(manifest,sort_keys=True))
    recovery.third={'root':root,'folder':folder,'runtime':runtime,'shared':Path(shared),'providers':bridge,
                    'candidates':candidates,'votes':votes,'protected':protected,'manifest':manifest,
                    'ledger_prefix':ledger['entries'],'state':state}
    return {k:state[k] for k in ('protocol','provider','model','source_checkpoint','eligible','protected_requests','protected_unjudged')}


def audit_archive(archive, folder, expected):
    """Audit every new recovery file against the independently read-back archive."""
    files={str(p.relative_to(folder.parent)):sha(p) for p in folder.rglob('*') if p.is_file()}
    seen=set()
    with tarfile.open(archive,'r|gz') as snapshot:
        for member in snapshot:
            if member.name not in files:continue
            stream=snapshot.extractfile(member)
            if hashlib.sha256(stream.read()).hexdigest()!=files[member.name]:raise RuntimeError('Third judge archive mismatch')
            seen.add(member.name)
    if seen!=set(files):raise RuntimeError('Third judge files missing from archive')
    journal=folder/'votes.jsonl';stored=rows(journal)
    if len(stored)!=expected or len({v['evaluation_id'] for v in stored})!=expected:
        raise RuntimeError('Third judge journal mismatch')


async def checkpoint(recovery,sink,event):
    from cloud.third_judge_fast import metrics
    ctx=recovery.third;root=ctx['root'];folder=ctx['folder']
    if ctx.get('active',0):raise RuntimeError('Cannot checkpoint active provider work')
    preserved(root,ctx['manifest']);writer_guard(recovery)
    ledger=json.loads((ctx['shared']/'provider_usage/gemini.json').read_text())
    prefix=ctx['ledger_prefix']
    if ledger['entries'][:len(prefix)]!=prefix:raise RuntimeError('Historical quota reservations changed')
    summary=metrics.summarise(root,ctx['votes'])
    ctx['state'].update(summary=summary,new_votes=len(ctx['votes']['gemini31']))
    (folder/'protocol.json').write_text(json.dumps(ctx['state'],indent=2,sort_keys=True))
    recovery.state['third_judge_recovery']=ctx['state']
    await asyncio.to_thread(recovery.evaluation.checkpoint,recovery.state)
    audit_archive('/kaggle/working/publish/artifacts.bin',folder,len(ctx['votes']['gemini31']))
    old=recovery.fast_archive_audit('/kaggle/working/publish/artifacts.bin')
    if old['persisted_judge_counts']!={p:len(ctx['votes'][p]) for p in MODELS} or any(old['persisted_duplicate_votes'].values()):
        raise RuntimeError('Historical archive counts differ')
    elapsed=max(.001,time.time()-ctx.get('started',time.time()))
    rate=len(ctx['votes']['gemini31'])*3600/elapsed
    report={'event':event,'protocol':PROTOCOL,'provider':PROVIDER,'model':MODEL,
            'checkpoint':recovery.state['last_persistence']['version'],'readback_verified':True,
            'archive_sha256':recovery.state['artifacts_sha256'],'unique_judged':summary['unique_judged'],
            'remaining':summary['remaining'],'new_votes':len(ctx['votes']['gemini31']),
            'eligible_remaining':len(eligible(ctx['candidates'],ctx['votes'],ctx['protected'])),
            'historical_votes':old['persisted_judge_counts'],'protected_requests':4,
            'new_unknown_requests':ctx.get('new_unknown',0),'successful_judgements_per_hour':rate,
            'deadline_epoch':ctx['state']['original_deadline_epoch'],'active_calls':0,'lease_version':recovery.lease_version}
    sink(report);ctx['last_report']=report
    return report


async def call(recovery,task):
    from cloud.d1000 import judge_recovery as jr
    ctx=recovery.third;ctx['active']=ctx.get('active',0)+1
    try:
        return await asyncio.to_thread(jr.complete, {k:os.environ[k] for k in ('GROQ_API_KEY','GEMINI_API_KEY')},
            PROVIDER,MODEL,task,ctx['runtime'],ctx['providers'])
    finally:ctx['active']-=1


async def calibrate(recovery,sink):
    from cloud.d1000 import judge_recovery as jr
    from eval.semantic_fixtures import fixtures
    from experiments import core
    ctx=recovery.third;results=[]
    # Exactly the original nine fixtures. Expected labels never reach call().
    for task,expected in fixtures():
        try:
            record=await call(recovery,dict(task,citations=[],student_evidence_available=False))
        except Exception:
            await seal_calibration_failure(recovery,sink)
            raise
        answer=record['answer'];passed=calibration_pass(task,expected,answer)
        results.append({'id':task['evaluation_id'],'passed':passed,'correctness':answer['correctness'],
                        'expected':expected,'evidence_support':answer['evidence_support'],
                        'abstention_category':answer['abstention_category'],'request_sha256':record['request_sha256']})
        sink({'event':'third_judge_calibration_case','cases_completed':len(results),'passed':passed,'model':MODEL})
    receipt={'protocol':jr.PROTOCOL,'recovery_protocol':PROTOCOL,'provider':PROVIDER,'model':MODEL,
             'fixtures_sha256':core.digest(fixtures()),'passed':len(results)==9 and all(v['passed'] for v in results),'results':results}
    (ctx['folder']/'calibration.json').write_text(json.dumps(receipt,indent=2))
    ctx['state']['calibration']='PASS' if receipt['passed'] else 'FAIL'
    await checkpoint(recovery,sink,'third_judge_calibration_readback')
    if not receipt['passed']:raise RuntimeError('Original nine-case calibration failed; no research calls')
    jr.verify_calibration(receipt);ctx['calibration_verified']=True
    return receipt['passed']


async def seal_calibration_failure(recovery,sink):
    """Seal actual cached calibration outcomes without making ANY API calls."""
    from cloud.d1000 import judge_recovery as jr
    from eval.semantic_fixtures import fixtures
    from experiments import core
    ctx=recovery.third;results=[]
    for original,expected in fixtures():
        task=dict(original,citations=[],student_evidence_available=False)
        key=core.digest({'task':task,'provider':PROVIDER,'model':MODEL,'protocol':jr.PROTOCOL})
        path=ctx['runtime']/'evaluation_cache_v2/gemini'/(key+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            reply={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'evaluations':[record['indexed_answer']]})}}]}
            if jr.validate(reply,task,ctx['providers'])!=record['answer']:raise RuntimeError('Calibration cache validation differs')
            answer=record['answer']
            results.append({'id':task['evaluation_id'],'passed':calibration_pass(task,expected,answer),
                            'correctness':answer['correctness'],'expected':expected,
                            'evidence_support':answer['evidence_support'],'abstention_category':answer['abstention_category'],
                            'request_sha256':record['request_sha256']})
        else:
            history_path=path.with_name(key+'.attempts.json')
            history=json.loads(history_path.read_text()) if history_path.exists() else []
            results.append({'id':task['evaluation_id'],'passed':False,'valid_result_available':False,
                            'attempts':len(history),'terminal_status':history[-1]['status'] if history else 'not_attempted'})
    receipt={'protocol':jr.PROTOCOL,'recovery_protocol':PROTOCOL,'provider':PROVIDER,'model':MODEL,
             'fixtures_sha256':core.digest(fixtures()),'passed':False,'results':results}
    (ctx['folder']/'calibration.json').write_text(json.dumps(receipt,indent=2))
    (ctx['folder']/'protected_task_ids.json').write_text(json.dumps(sorted(ctx['protected'])))
    source=ctx['folder']/'source';source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):(source/path.name).write_bytes(path.read_bytes())
    ctx['state'].update(phase='calibration_failed',calibration='FAIL',
                        calibration_cases_passed=sum(v['passed'] for v in results),
                        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    await checkpoint(recovery,sink,'third_judge_calibration_failed_readback')
    return {'calibration':'FAIL','cases_passed':sum(v['passed'] for v in results),'research_calls':0}


async def run(recovery,sink):
    """Bounded two-call batches; first batch one call measures actual latency.

    At any unknown outcome or quota pause stop dispatching, drain the bounded
    batch, checkpoint, then return. No repeated failed probes or automatic replay.
    """
    from cloud.d1000 import judge_recovery as jr
    ctx=recovery.third
    if not ctx.get('calibration_verified') or getattr(recovery,'third_started',False):
        raise RuntimeError('Verified calibration and exclusive entry required')
    writer_guard(recovery);preserved(ctx['root'],ctx['manifest'])
    recovery.third_started=True;ctx['started']=time.time();ctx['state']['phase']='running'
    failed=set();batch_limit=1;since=0
    await checkpoint(recovery,sink,'third_judge_research_start_readback')
    with recovery.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        while time.time()<ctx['state']['original_deadline_epoch']-900:
            pending=[t for t in eligible(ctx['candidates'],ctx['votes'],ctx['protected']) if t['evaluation_id'] not in failed]
            if not pending:break
            batch=pending[:batch_limit];before=time.time()
            results=await asyncio.gather(*(call(recovery,t) for t in batch),return_exceptions=True)
            stop=False
            for task,record in zip(batch,results):
                if isinstance(record,BaseException):
                    failed.add(task['evaluation_id'])
                    if isinstance(record,ctx['providers'].QuotaPause):
                        ctx['state'].update(phase='quota_wait',resume_after_epoch=record.until);stop=True
                    elif 'Known-invalid judgement exhausted' in str(record):
                        ctx['state']['known_invalid_exhausted']=ctx['state'].get('known_invalid_exhausted',0)+1
                    else:
                        # Preserve exact journal; do not retry or reassign this task.
                        key=__import__('experiments.core',fromlist=['digest']).digest({'task':task,'provider':PROVIDER,'model':MODEL,'protocol':jr.PROTOCOL})
                        history=json.loads((ctx['runtime']/'evaluation_cache_v2/gemini'/(key+'.attempts.json')).read_text())
                        if history and history[-1]['status'] in ('in_flight','transport_uncertain','response_saved'):
                            ctx['protected'].add(task['evaluation_id']);ctx['new_unknown']=ctx.get('new_unknown',0)+1
                        ctx['state'].update(phase='provider_blocked',safe_error_type=type(record).__name__);stop=True
                    continue
                value={**record['answer'],'provider':PROVIDER,'model':MODEL,'evaluation_protocol':PROTOCOL,
                       'request_sha256':record['request_sha256'],'input_sha256':record['input_sha256'],'usage':record['usage']}
                append(ctx['folder']/'votes.jsonl',value)
                ctx['votes']['gemini31'][task['evaluation_id']]=value;since+=1
            # Increase only after a measured successful first request. Two is the
            # hard bound; shared locked reservations still enforce 15 RPM/TPM/RPD.
            if len(ctx['votes']['gemini31']) and time.time()-before>=4:batch_limit=2
            if since>=4 or stop:
                await checkpoint(recovery,sink,'third_judge_votes_readback');since=0
            if stop:break
            await asyncio.sleep(max(0,10-(time.time()-before)))
        if ctx['state']['phase']=='running':ctx['state']['phase']='eligible_complete' if not eligible(ctx['candidates'],ctx['votes'],ctx['protected']) else 'deadline_stop'
        return await checkpoint(recovery,sink,'third_judge_terminal_readback')
