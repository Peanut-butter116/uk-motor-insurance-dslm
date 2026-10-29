"""Single-writer private Kaggle stage persistence; cloud-memory readback hashes."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def info(api,dataset):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    r=ApiGetDatasetRequest();r.owner_slug,r.dataset_slug=dataset.split('/')
    with api.build_kaggle_client() as c:return c.datasets.dataset_api_client.get_dataset(r)


def verify_file(dataset,version,name,expected):
    url=f'https://www.kaggle.com/api/v1/datasets/download/{dataset}/{name}?datasetVersionNumber={version}'
    # urllib strips Authorization on cross-host download redirects explicitly.
    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):
            if not newurl.startswith('https://'):raise RuntimeError('Insecure download redirect')
            r=super().redirect_request(req,fp,code,msg,headers,newurl);r.remove_header('Authorization');return r
    req=urllib.request.Request(url,headers={'Authorization':'Bearer '+os.environ['KAGGLE_API_TOKEN']})
    with urllib.request.build_opener(Redirect()).open(req,timeout=120) as stream:
        h=hashlib.sha256()
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    assert h.hexdigest()==expected,'Remote artifact hash mismatch'


def publish(folder,dataset,create=False):
    from kaggle import api
    from cloud.d1000.export import sha
    api.authenticate();folder=Path(folder)
    (folder/'dataset-metadata.json').write_text(json.dumps({'id':dataset,'title':dataset.split('/')[1],
        'licenses':[{'name':'other'}]}))
    if create:
        result=api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False,dir_mode='tar')
    else:
        previous=info(api,dataset);assert previous.is_private is True
        result=api.dataset_create_version(str(folder),version_notes='D1000 stage checkpoint',quiet=True,
            convert_to_csv=False,delete_old_versions=False,dir_mode='tar')
    if getattr(result,'error',None):raise RuntimeError('Private persistence upload rejected')
    for _ in range(60):
        d=info(api,dataset);assert d.is_private is True
        if api.dataset_status(dataset)=='ready' and (create or d.current_version_number>previous.current_version_number):
            for p in folder.iterdir():
                if p.is_file() and p.name!='dataset-metadata.json':verify_file(dataset,d.current_version_number,p.name,sha(p))
            return {'dataset':dataset,'version':d.current_version_number,'private':True,'readback_verified':True}
        time.sleep(10)
    raise RuntimeError('Persistence not verified; stop before more work')
