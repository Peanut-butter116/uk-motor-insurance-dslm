"""Renew account verification only after actual current checks; never renew quotas."""
import hashlib
import json
import os
from pathlib import Path
import time

MODELS={'groq':'openai/gpt-oss-120b','gemini':'gemini-3.5-flash-lite'}
REQUIRED=('free_plan_confirmed','model_catalogue_verified','synthetic_json_probe_passed',
          'policy_transfer_authorised','data_use_terms_compatible','training_dataset_use_compatible')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def renew(recovery, providers, runtime, verify_journals, account_evidence):
    from cloud.d1000.judge_recovery import verify_calibration,PROTOCOL
    from eval.semantic_fixtures import fixtures
    from experiments import core
    root=Path(recovery.evaluation.ROOT);runtime=Path(runtime)
    before=verify_journals(root)
    if before!=recovery.receipt['saved_judgements']:raise RuntimeError('Journal state changed')
    now=time.time()
    if set(account_evidence)!=set(MODELS):raise RuntimeError('Both account checks required')
    path=runtime/'provider_verification.json';original=path.read_bytes();old=json.loads(original)
    retained={p:sha(p) for directory in ('judging','judge_runtime') for p in (root/directory).rglob('*') if p.is_file() and p!=path}
    checks={}
    for provider,model in MODELS.items():
        evidence=account_evidence[provider]
        if evidence.get('free_plan_observed') is not True or evidence.get('terms_reviewed') is not True:
            raise RuntimeError('Current free account and data-use checks required')
        if not 0<=now-evidence.get('observed_at_epoch',0)<=3600:raise RuntimeError('Current account evidence required')
        item=old[provider]
        if item.get('model')!=model or not all(item.get(k) is True for k in REQUIRED):
            raise RuntimeError('Cannot create a missing original attestation')
        catalogue,_=providers.transport({k:os.environ[k] for k in ('GROQ_API_KEY','GEMINI_API_KEY')},provider)
        if not providers.catalogue_contains(provider,model,catalogue):raise RuntimeError('Pinned model absent')
        calibration_path=root/'judging'/f'{provider}-calibration-v2.json'
        calibration=json.loads(calibration_path.read_text());verify_calibration(calibration)
        if not (calibration['passed'] and calibration['protocol']==PROTOCOL and calibration['fixtures_sha256']==core.digest(fixtures())):
            raise RuntimeError('Original calibration cannot be reused')
        checks[provider]={'account':evidence,'model':model,'catalogue_verified':True,'original_calibration_sha256':sha(calibration_path),'inference_calls':0}
    # No cache, vote, reservation, limit, permission or methodology changes.
    current={p:sha(p) for directory in ('judging','judge_runtime') for p in (root/directory).rglob('*') if p.is_file() and p!=path}
    if current!=retained:raise RuntimeError('Request state changed during verification')
    new=json.loads(original)
    for provider in MODELS:new[provider]['verified_at_epoch']=now
    encoded=json.dumps(new,sort_keys=True,indent=2).encode()
    audit=root/'recovery_audits'/'provider_reverification_v1';audit.mkdir(parents=True,exist_ok=False);audit.chmod(0o700)
    backup=audit/'original_provider_verification.json';backup.write_bytes(original);backup.chmod(0o600)
    path.write_bytes(encoded);path.chmod(0o600)
    try:
        for provider,model in MODELS.items():providers.verify_receipt(runtime,provider,model)
    except BaseException:
        path.write_bytes(original);raise
    transition={'original_sha256':hashlib.sha256(original).hexdigest(),'renewed_sha256':sha(path),
                'backup':str(backup),'checks':checks,'quota_limits_unchanged':True,'request_state_unchanged':True}
    (audit/'transition.json').write_text(json.dumps(transition,sort_keys=True,indent=2))
    (audit/'transition.json').chmod(0o600)
    recovery.receipt['saved_judgements']=verify_journals(root)
    recovery.receipt['provider_reverification']=transition
    return transition


def validate_transition(path,original_hash,transition):
    if transition.get('original_sha256')!=original_hash or sha(transition['backup'])!=original_hash or sha(path)!=transition.get('renewed_sha256'):
        raise RuntimeError('Provider receipt transition identity differs')
    old=json.loads(Path(transition['backup']).read_text());new=json.loads(Path(path).read_text())
    for provider in MODELS:
        if new[provider].get('verified_at_epoch',0)<=old[provider].get('verified_at_epoch',0):raise RuntimeError('No current verification')
        new[provider]['verified_at_epoch']=old[provider]['verified_at_epoch']
        check=transition['checks'][provider]
        if check.get('catalogue_verified') is not True or check.get('inference_calls')!=0:raise RuntimeError('Missing verification evidence')
    if old!=new or not transition.get('request_state_unchanged') or not transition.get('quota_limits_unchanged'):
        raise RuntimeError('Receipt fields other than verified time changed')
