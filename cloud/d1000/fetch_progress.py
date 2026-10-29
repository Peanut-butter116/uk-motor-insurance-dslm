"""Aggregate-only monitor; never downloads datasets, adapters or raw responses."""
import argparse
import json
from pathlib import Path
from cloud.d1000.storage import info


def fetch(output,execution_version=None):
    from kaggle import api
    api.authenticate();out=Path(output);out.mkdir(exist_ok=True,parents=True)
    result={}
    campaign='uk-motor-d1000-train-eval'+('/'+str(execution_version) if execution_version is not None else '')
    for kind,slug in [('smoke','uk-motor-d1000-gpu-smoke/2'),('campaign',campaign)]:
        kernel='scarletthe0116/'+slug;status=api.kernels_status(kernel).status.name
        item={'notebook':kernel,'status':status};folder=out/kind;folder.mkdir(exist_ok=True)
        if status in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'):
            pattern=r'^(gpu_probe_receipt|gpu_probe_failure|campaign_receipt|FAILED)\.json$'
            try:
                paths,_=api.kernels_output(kernel,str(folder),force=True,quiet=True,file_pattern=pattern)
                for p in paths:
                    p=Path(p)
                    if p.name in ('gpu_probe_receipt.json','gpu_probe_failure.json','campaign_receipt.json','FAILED.json'):
                        item[p.name]=json.loads(p.read_text())
            except Exception as e:item['receipt_fetch_error']=type(e).__name__
        result[kind]=item
        if kind=='campaign':
            item['stage_receipt_observed']='campaign_receipt.json' in item
            item['execution_confirmed']=status=='RUNNING' or item['stage_receipt_observed']
            if status=='COMPLETE' and not item['execution_confirmed']:item['interpretation']='saved_only_no_execution_receipt'
    dataset='scarletthe0116/uk-motor-d1000-models-results'
    try:
        d=info(api,dataset);assert d.is_private is True
        if api.dataset_status(dataset)=='ready':
            folder=out/'persistent';folder.mkdir(exist_ok=True)
            api.dataset_download_file(dataset+'/'+str(d.current_version_number),'progress.json',path=str(folder),quiet=True,force=True,unzip=True)
            result['persistent']={'dataset':dataset,'version':d.current_version_number,'private':True,
                                  'progress':json.loads((folder/'progress.json').read_text())}
    except Exception as e:
        missing=getattr(getattr(e,'response',None),'status_code',None)==404 or '404' in str(e)
        result['persistent_status']='not_created_yet' if missing else type(e).__name__
    (out/'monitor.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--execution-version',type=int);fetch(**vars(p.parse_args()))
