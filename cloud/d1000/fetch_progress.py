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
            try:
                # Do not use kernels_output: it saves response.log even when
                # file_pattern excludes logs. Select exact aggregate files only.
                from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
                import requests
                request=ApiListKernelSessionOutputRequest()
                request.user_name='scarletthe0116'
                parts=slug.split('/');request.kernel_slug=parts[0]
                if len(parts)>1:request.version_label='v'+parts[1]
                with api.build_kaggle_client() as client:
                    listing=client.kernels.kernels_api_client.list_kernel_session_output(request)
                allowed={'gpu_probe_receipt.json','gpu_probe_failure.json','campaign_receipt.json','FAILED.json'}
                for entry in listing.files or []:
                    if entry.file_name not in allowed:continue
                    response=requests.get(entry.url,timeout=60)
                    response.raise_for_status()
                    if not response.content.strip():
                        item.setdefault('empty_receipts',[]).append(entry.file_name);continue
                    value=response.json();item[entry.file_name]=value
                    (folder/entry.file_name).write_text(json.dumps(value,indent=2)+'\n')
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
