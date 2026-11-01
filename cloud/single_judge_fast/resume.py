"""Restore only an independently verified, deliberately suspended fast checkpoint.

Never recover an arbitrary lost VM as if its last request were unprocessed.
Requires the same approved private Colab destination and supported existing secrets.
"""
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def validate_suspension(state,control,proof,version,now):
    from cloud.single_judge_fast.lifecycle import clock
    from cloud.single_judge_fast.protocol import PROTOCOL
    if version<250 or state.get('status')!='single_judge_suspended_for_supported_runtime_renewal':
        raise RuntimeError('Latest checkpoint is not a controlled suspension; reconcile unknown outcomes')
    if proof.get('protocol')!=PROTOCOL or proof.get('phase')!='suspended' or proof.get('expected_checkpoint')!=version:
        raise RuntimeError('Suspension checkpoint identity differs')
    if proof.get('active_provider_threads')!=0 or proof.get('active_provider_tasks')!=0:
        raise RuntimeError('Provider requests were not fully drained')
    if control.get('phase')!='suspended' or proof['launch_epoch']!=control['launch_epoch'] or proof['deadline_epoch']!=control['deadline_epoch']:
        raise RuntimeError('Suspension clock differs')
    clock(control,now,proof['unique_judged'])


def restore(runtime_scope,transport_scope,archive_audit,*,private_colab_transfer_approved=False):
    if not private_colab_transfer_approved:raise PermissionError('Existing project approval required before private processing')
    if not Path('/content').is_dir() or os.environ.get('COLAB_GPU') not in (None,'','0'):
        raise RuntimeError('Private Colab CPU runtime required')
    if Path('/kaggle/working/d1000-authoritative-writer.lock').exists() or Path('/kaggle/working/d1000').exists():
        raise RuntimeError('Use a fresh VM; do not replace an existing writer or restored state')
    from google.colab import userdata
    from kaggle import api
    for name in ('KAGGLE_API_TOKEN','GROQ_API_KEY','GEMINI_API_KEY'):
        value=userdata.get(name)
        if not isinstance(value,str) or not value:raise RuntimeError('Approved Colab secret unavailable')
        os.environ[name]=value;del value
    self=runtime_scope['ColabRecovery']()
    Path('/kaggle/working').mkdir(parents=True,exist_ok=True)
    self.log=Path('/kaggle/working/fast-resume-private.log');self.log.touch(mode=0o600);self.log.chmod(0o600)
    with self.log.open('a') as log,redirect_stdout(log),redirect_stderr(log):
        api.authenticate();self.api=api
        store=runtime_scope['STORE'];kernel=runtime_scope['KERNEL'];lease=runtime_scope['LEASE']
        if str(api.config_values.get(api.CONFIG_NAME_USER,'')).lower()!=store.split('/')[0]:raise RuntimeError('Storage owner authentication required')
        if api.kernels_status(kernel).status.name not in runtime_scope['TERMINAL']:raise RuntimeError('Kaggle writer active')
        with tempfile.TemporaryDirectory() as tmp:
            api.kernels_pull(kernel+'/18',tmp,metadata=True)
            notebooks=list(Path(tmp).glob('*.ipynb'))
            if len(notebooks)!=1:raise RuntimeError('Ambiguous audited source')
            files=runtime_scope['supported_bundle'](json.loads(notebooks[0].read_text()))
        code=Path('/kaggle/working/code');runtime_scope['stage_bundle'](code,files)
        sys.path.insert(0,str(code));os.chdir(code)
        from cloud.d1000 import evaluation_only as evaluation
        from cloud.d1000.storage import info,publish,verify_file
        self.info=info;self.publish=publish;self.evaluation=evaluation
        audit=json.loads((code/'cloud/d1000/evaluation_repair_v1.json').read_text())
        for p,h in {**audit['destination_hashes'],**audit['immutable_code_hashes']}.items():
            if runtime_scope['sha'](code/p)!=h:raise RuntimeError('Original source changed')
        for name,field in (('groq_probe_quota_deltas.json','quota_deltas_sha256'),('groq_calibration14_receipt.json','groq_calibration_receipt_sha256')):
            if runtime_scope['sha'](code/'cloud/d1000'/name)!=audit[field]:raise RuntimeError('Private recovery sideband changed')
        current=info(api,store)
        if not current.is_private:raise RuntimeError('Results must remain private')
        version=current.current_version_number
        mount=runtime_scope['INPUT_ROOT']/('fast-resume'+str(version));mount.mkdir(parents=True);mount.chmod(0o700)
        for name in ('progress.json','artifacts.bin'):
            api.dataset_download_file(store+'/'+str(version),name,path=str(mount),quiet=True,force=True,unzip=True)
            (mount/name).chmod(0o600)
        saved=json.loads((mount/'progress.json').read_text())
        if runtime_scope['sha'](mount/'artifacts.bin')!=saved['artifacts_sha256']:raise RuntimeError('Latest archive hash differs')
        scope=evaluation.__dict__.copy();exec(runtime_scope['restore_adapter'](files['cloud/d1000/evaluation_only.py']),scope)
        state,preservation=scope['restore']()
        if preservation['immutable_manifest_sha256']!=audit['immutable_manifest_sha256']:raise RuntimeError('Frozen data changed')
        root=evaluation.ROOT;folder=root/'single_judge_fast_v1'
        control=json.loads((folder/'protocol.json').read_text());proof=json.loads((folder/'suspension.json').read_text())
        validate_suspension(state,control,proof,version,time.time())
        self.state=state;self.preservation=preservation;self.receipt['preservation']=preservation
        # Audit the previously renewed receipt against its preserved original,
        # never silently rewrite verification dates during a runtime restart.
        self.receipt['provider_reverification']=json.loads((root/'recovery_audits/provider_reverification_v1/transition.json').read_text())
        transport_scope['prepare'](self)
        before=info(api,lease)
        if not before.is_private or before.current_version_number!=proof['lease_version']:raise RuntimeError('Prior lease changed')
        verify_file(lease,before.current_version_number,'owner.json',proof['lease_owner_sha256'])
        self.receipt['scratch_private_persistence']=runtime_scope['scratch_persistence'](api,info,verify_file,self.session)
        if info(api,store).current_version_number!=version or info(api,lease).current_version_number!=before.current_version_number:
            raise RuntimeError('Writer or source changed during renewal')
        with tempfile.TemporaryDirectory() as tmp:
            owner=Path(tmp)/'owner.json';owner.write_text(json.dumps({'session':self.session,'source_checkpoint':239}))
            result=publish(Path(tmp),lease,create=False)
            self.lease_version=result['version']
        fd=os.open('/kaggle/working/d1000-authoritative-writer.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
        self.ready=True;self.started=True;self.fast_handoff={'old_workers_drained':True,'source_checkpoint':version,'prior_suspended_lease':proof['lease_version']}
        self.fast_resume_control=control;self.fast_archive_audit=archive_audit
        original=evaluation.checkpoint
        def guarded_checkpoint(value):
            lease_now=info(api,lease)
            if not lease_now.is_private or lease_now.current_version_number!=self.lease_version:raise RuntimeError('Lease changed')
            if api.kernels_status(kernel).status.name not in runtime_scope['TERMINAL']:raise RuntimeError('Concurrent Kaggle writer')
            return original(value)
        evaluation.checkpoint=guarded_checkpoint
        stop=folder/'STOP_REQUESTED'
        if stop.exists():stop.rename(folder/('STOP_REQUESTED-consumed-at-checkpoint'+str(version)))
        from cloud.single_judge_fast import engine
        report=engine.prepare(self)
    return self,report
