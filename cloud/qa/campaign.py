"""Bounded Kaggle-session continuation; no external scheduler or paid fallback."""
import collections
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from runner import require_cloud, dataset_info, DATASET, utc
from state import read_rows, report, write_json

TARGET=1000
# Conservative wall-clock cap, not a claim about Kaggle's platform limit.
SESSION_SECONDS=4*60*60


def run():
    require_cloud()
    from kaggle_secrets import UserSecretsClient
    secrets=UserSecretsClient()
    for name in ('GROQ_API_KEY','GEMINI_API_KEY','KAGGLE_API_TOKEN'):
        value=secrets.get_secret(name)
        if not value:raise RuntimeError('Required Kaggle secret is unavailable: '+name)
        if name=='KAGGLE_API_TOKEN':os.environ[name]=value
        del value
    from kaggle import api
    info=dataset_info(api)
    if info.is_private is not True:raise RuntimeError('Private checkpoint required')
    campaign={'started_at_utc':utc(),'input_version':info.current_version_number,'target':TARGET,
              'execution_started':True,'automatic_continuation':'same Kaggle session only',
              'session_limit_seconds':SESSION_SECONDS,'fresh_test_passed':False,'batches':[]}
    end=time.monotonic()+SESSION_SECONDS
    root=Path('/kaggle/working');adapter=Path(__file__).parent
    output=root/'campaign_report.json'
    last_batch_safe=False
    write_json(output,campaign)
    while time.monotonic()<end-300:
        # Fresh processes prevent patched production modules leaking across batches.
        for name in ('session','incoming','persist'):
            path=root/name
            if path.exists():shutil.rmtree(path)
        (root/'persistence_receipt.json').unlink(missing_ok=True)
        cap=5 if campaign['fresh_test_passed'] else 1
        last_batch_safe=False
        child=subprocess.run([sys.executable,'-c',
            'import sys;sys.path.insert(0,'+repr(str(adapter))+');from runner import main;main(max_packages='+str(cap)+',target=1000)'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        run_path=root/'session/uk-insurance-QA-data-local/production-v1'
        receipt_path=root/'persistence_receipt.json'
        if not receipt_path.exists():
            campaign.update(stop_reason='No verified persistence receipt; inspect private notebook execution',child_exit_code=child.returncode)
            break
        receipt=json.loads(receipt_path.read_text())
        state=json.loads((run_path/'state.json').read_text())
        measured=report(run_path)
        if receipt['accepted']!=measured['total_accepted'] or child.returncode:
            campaign.update(stop_reason='Child failed or final export differs from verified checkpoint; stop for investigation',
                            child_exit_code=child.returncode,latest_verified_receipt=receipt)
            break
        qc=json.loads((run_path/'cloud_quality_report.json').read_text())
        if qc.get('passed') is not True or qc.get('accepted')!=measured['total_accepted']:
            campaign['stop_reason']='Missing matching acceptance audit; stop for investigation';break
        last_batch_safe=True
        origin=json.loads((run_path/'campaign_start.json').read_text())
        packages=state['generation'][origin['generation_count']:]
        new_candidates=[r for r in read_rows(run_path/'generated_candidates.jsonl')
                        if r['example_id'] not in set(origin['generated_ids'])]
        accepted_new=[r for r in new_candidates if state['decisions'].get(r['example_id'],{}).get('status')=='accepted']
        fresh_pass=bool(accepted_new) and all(state['decisions'][r['example_id']].get('judge') for r in accepted_new)
        campaign['fresh_test_passed']=campaign['fresh_test_passed'] or fresh_pass
        campaign.update(measured)
        campaign.update(target=TARGET,remaining_to_target=max(0,TARGET-measured['total_accepted']),
            latest_verified_receipt=receipt,new_candidates_generated=len(new_candidates),
            new_accepted=len(accepted_new),packages_completed_since_start=len(packages),
            packages_remaining=max(0,origin['eligible_package_count']-len(packages)),
            new_candidate_dispositions=dict(collections.Counter(state['decisions'].get(r['example_id'],{}).get('status','pending') for r in new_candidates)),
            new_candidate_failure_reasons=dict(collections.Counter(reason for r in new_candidates for reason in state['decisions'].get(r['example_id'],{}).get('reasons',[]))),
            quota_provider=state.get('paused_provider'),resume_after_epoch=state.get('resume_after_epoch'))
        campaign['batches'].append({'completed_at_utc':utc(),'version':receipt['version'],'accepted':measured['total_accepted'],
            'generated':measured['candidates_generated'],'status':state['status'],'package_cap':cap})
        write_json(output,campaign)
        # Main already verified every saved checkpoint. Keep campaign summaries in
        # the next checkpoint as well as private notebook outputs.
        write_json(run_path/'campaign_report.json',campaign)
        if measured['total_accepted']>=TARGET:
            campaign['stop_reason']='target_reached';break
        if state['status']=='blocked_source_exhaustion':
            campaign['stop_reason']='eligible_sources_exhausted';break
        if state['status']=='paused_provider_quota':
            wait=max(0,float(state.get('resume_after_epoch',0))-time.time())+5
            if wait>max(0,end-time.monotonic()-300):
                campaign['stop_reason']='Quota wait exceeds remaining session; restart from newest checkpoint after resume_after_epoch';break
            # Waiting runs remotely; do not bypass or clear provider reservations.
            campaign.update(waiting_for_quota=True,quota_wait_seconds=wait)
            write_json(output,campaign)
            time.sleep(wait)
            campaign['waiting_for_quota']=False
            continue
        if state['status'] not in ('paused_cloud_batch_limit','generating'):
            campaign['stop_reason']='Unresolved operational stop: '+state['status'];break
    else:
        campaign['stop_reason']='Session time cap reached; no cross-session scheduler configured'
    campaign['finished_at_utc']=utc()
    write_json(output,campaign)
    # A final summary is persisted without running any provider calls.
    if (root/'session/uk-insurance-QA-data-local/production-v1/state.json').exists() and campaign.get('latest_verified_receipt'):
        from runner import persist
        write_json(root/'session/uk-insurance-QA-data-local/production-v1/campaign_report.json',campaign)
        if last_batch_safe:
            persist(root/'session',api)
            campaign['latest_verified_receipt']=json.loads(receipt_path.read_text())
            write_json(output,campaign)
    print(json.dumps(campaign,indent=2))


if __name__=='__main__':run()
