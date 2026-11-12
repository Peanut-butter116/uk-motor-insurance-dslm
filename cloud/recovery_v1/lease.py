"""Single-create private lease with bounded metadata-provisioning checks."""
import hashlib
import json
from pathlib import Path
import time

LEASE='scarletthe0116/uk-motor-d1000-evaluation-lease-v1'


def install(recovery, *, attempts=18):
    original=recovery.publish
    def publish(folder,dataset,create=False):
        if dataset!=LEASE or create is not True:
            return original(folder,dataset,create=create)
        folder=Path(folder);owner=folder/'owner.json'
        expected={'session':recovery.session,'source_checkpoint':239}
        if json.loads(owner.read_text())!=expected:
            raise RuntimeError('Writer lease owner does not match this verified recovery')
        digest=hashlib.sha256(owner.read_bytes()).hexdigest()
        (folder/'dataset-metadata.json').write_text(json.dumps({'id':LEASE,
            'title':LEASE.split('/')[1],'licenses':[{'name':'other'}]}))
        result=recovery.api.dataset_create_new(str(folder),public=False,quiet=True,
            convert_to_csv=False,dir_mode='tar')
        if getattr(result,'error',None):
            raise RuntimeError('Writer lease already exists or creation was rejected')
        # One acknowledged create only. Repeated writes or lease stealing are forbidden.
        for attempt in range(attempts):
            try:
                d=recovery.info(recovery.api,LEASE)
                if not d.is_private:raise RuntimeError('Writer lease is not private')
                if recovery.api.dataset_status(LEASE)=='ready':
                    from cloud.d1000.storage import verify_file
                    verify_file(LEASE,d.current_version_number,'owner.json',digest)
                    return {'dataset':LEASE,'version':d.current_version_number,'private':True,
                            'readback_verified':True,'owner_sha256':digest}
            except Exception as exc:
                code=getattr(getattr(exc,'response',None),'status_code',None)
                if code not in (403,404) or attempt==attempts-1:raise
            if attempt+1<attempts:time.sleep(10)
        raise RuntimeError('Writer lease provisioning/readback unverified')
    recovery.publish=publish
