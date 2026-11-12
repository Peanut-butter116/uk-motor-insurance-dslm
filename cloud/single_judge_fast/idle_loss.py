"""Reconcile only a reviewed coordinator that could issue no further requests.

This does not recover arbitrary VM loss. Both workers must be provably unable
to issue any request between the durable snapshot and the original deadline.
The old VM must independently be absent from Colab's active session inventory.
"""
ENGINE_SHA='6e14157b9c61d81346ef7a659ac1fa132d9fbbbef92b932a9a9bf594d9bcc700'
PROTOCOL_SHA='eb2ec15c0e2cf49ed28cf7d61ebf1a6928aa9930fafc5bb68b7c940065a884b5'


def validate(state,control,previous_suspension,version,now,*,old_runtime_ended=False):
    from cloud.single_judge_fast.protocol import PROTOCOL
    from cloud.single_judge_fast.lifecycle import clock
    if old_runtime_ended is not True:raise RuntimeError('Old Colab runtime termination not independently verified')
    hashes=control.get('source_hashes',{})
    if hashes.get('engine.py')!=ENGINE_SHA or hashes.get('protocol.py')!=PROTOCOL_SHA:
        raise RuntimeError('Idle-loss worker semantics have not been reviewed for this source')
    if control.get('protocol')!=PROTOCOL or state.get('status')!='single_judge_partial':
        raise RuntimeError('Expected durable partial fast checkpoint')
    persisted=state['last_persistence']
    if persisted.get('version')!=version or persisted.get('private') is not True or persisted.get('readback_verified') is not True:
        raise RuntimeError('Latest private readback evidence required')
    progress=state['single_judge_fast']['progress']
    if progress.get('active_provider_calls')!=0:raise RuntimeError('Snapshot may contain active requests')
    statuses=progress['provider_status'];deadline=control['deadline_epoch']
    if statuses.get('groq',{}).get('status')!='quota_wait' or statuses['groq'].get('resume_after_epoch',0)<deadline:
        raise RuntimeError('Groq could have issued post-checkpoint requests')
    gemini=statuses.get('gemini',{})
    if gemini.get('status')!='blocked' or gemini.get('category') not in ('transport_uncertain_provider_stopped','provider_or_validation_failure'):
        raise RuntimeError('Gemini could have issued post-checkpoint requests')
    for field in ('launch_epoch','deadline_epoch'):
        if progress.get(field)!=control[field] or previous_suspension.get(field)!=control[field]:
            raise RuntimeError('Original campaign clock differs')
    if previous_suspension.get('lease_version')!=control.get('same_lease_version'):
        raise RuntimeError('Historical lease ownership differs')
    if not previous_suspension.get('lease_owner_sha256'):raise RuntimeError('Historical lease owner evidence missing')
    clock({**control,'phase':'suspended'},now,progress['unique_judged'])
    return {'method':'reviewed_idle_loss_v1','source_checkpoint':version,'old_runtime_ended':True,
            'no_post_checkpoint_requests_possible':True,'engine_sha256':ENGINE_SHA,
            'original_deadline_epoch':deadline,'unique_judged':progress['unique_judged'],
            'lease_version':previous_suspension['lease_version'],
            'lease_owner_sha256':previous_suspension['lease_owner_sha256']}
