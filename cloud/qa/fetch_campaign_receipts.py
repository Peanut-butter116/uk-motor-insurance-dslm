"""Read aggregate cloud state only; never download generated datasets or raw logs."""
import argparse,json
from pathlib import Path
from runner import DATASET,dataset_info


def fetch(output,execution_version=None):
    from kaggle import api
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    kernel='scarletthe0116/uk-motor-qa-resume'
    if execution_version is not None:kernel+='/'+str(execution_version)
    status=api.kernels_status(kernel).status.name
    info=dataset_info(api)
    if info.is_private is not True:raise RuntimeError('Private storage requirement failed')
    dataset_status=api.dataset_status(DATASET)
    result={'notebook_status':status,'latest_dataset_version':info.current_version_number,'dataset_private':True,'dataset_status':dataset_status}
    if execution_version is not None:result['execution_version']=execution_version
    if dataset_status=='ready':
        folder=output/('v'+str(info.current_version_number));folder.mkdir(exist_ok=True)
        api.dataset_download_file(DATASET+'/'+str(info.current_version_number),'progress.json',path=str(folder),quiet=True,force=True,unzip=True)
        progress=json.loads((folder/'progress.json').read_text())
        result.update({k:progress.get(k) for k in ('total_accepted','answerability','candidates_generated','generated_dispositions','status','last_checkpoint_utc')})
        if 'fast_campaign' in progress:
            result['fast_campaign']=progress['fast_campaign']
    if status in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'):
        paths,_=api.kernels_output(kernel,str(output),force=True,quiet=True,page_token='',page_size=200,
            file_pattern=r'^(campaign_report|persistence_receipt|fast_campaign_report|fast_persistence_receipt)\.json$')
        result['aggregate_files']=[Path(p).name for p in paths]
        if (output/'campaign_report.json').exists():
            campaign=json.loads((output/'campaign_report.json').read_text())
            result.update({k:campaign.get(k) for k in ('fresh_test_passed','new_candidates_generated','new_accepted','packages_completed_since_start','packages_remaining','stop_reason','latest_verified_receipt','child_exit_code')})
        for name in ('fast_campaign_report','fast_persistence_receipt'):
            path=output/(name+'.json')
            if path.exists():result[name]=json.loads(path.read_text())
    (output/'monitor.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--execution-version',type=int);fetch(**vars(p.parse_args()))
