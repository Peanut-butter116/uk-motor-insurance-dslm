"""Read job state and free GPU quota; never submit jobs or expose credentials."""
import argparse
import datetime
import json
from pathlib import Path


def audit(owner, output):
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    names=['motor-six-model-compatibility-20261020','motor-six-model-frozen-test-20261020']
    names += ['uk-insurance-'+kind+'-'+model for kind in ['base-eval','smoke','full','sft-eval'] for model in ['qwen3b','ministral3b']]
    jobs=[]
    for name in names:
        s=api.kernels_status(owner+'/'+name)
        jobs.append({'notebook':owner+'/'+name,'status':s.status.name,
            'failure_message_present':bool(s.failure_message),'url':'https://www.kaggle.com/code/'+owner+'/'+name})
    with api.build_kaggle_client() as c:
        q=c.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
        g=q.gpu_quota
        quota={k:getattr(g,k).total_seconds()/3600 for k in ['time_used','time_reserved','total_time_allowed']}
        quota['paid_scaling_enabled']=g.is_pay_to_scale_enabled
        quota['refresh_utc']=str(q.quota_refresh_time)
    result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'jobs':jobs,'gpu_quota_hours':quota,
        'interpretation':'Queued is not running or complete. Free quota does not guarantee accelerator availability. No paid judging or new SFT run submitted.'}
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner',default='scarletthe0116');p.add_argument('--output',required=True)
    audit(**vars(p.parse_args()))
