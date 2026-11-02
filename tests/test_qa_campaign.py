"""Exercise remote continuation decisions with fictional checkpoints, never API calls."""
import json
from pathlib import Path
import sys
import types
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cloud/qa'))
import campaign


@pytest.mark.parametrize('scenario', ['target','unsafe_child','daily_quota'])
def test_campaign_only_continues_after_verified_batch(tmp_path,monkeypatch,scenario):
    root=tmp_path/'cloud';root.mkdir();calls=[];sleeps=[];published=[]
    real_path=Path
    monkeypatch.setattr(campaign,'Path',lambda value:root if str(value)=='/kaggle/working' else real_path(value))
    monkeypatch.setattr(campaign,'require_cloud',lambda:None)
    monkeypatch.setattr(campaign,'dataset_info',lambda _:types.SimpleNamespace(is_private=True,current_version_number=10))
    monkeypatch.setitem(sys.modules,'kaggle_secrets',types.SimpleNamespace(UserSecretsClient=lambda:types.SimpleNamespace(get_secret=lambda _: 'fictional-secret')))
    monkeypatch.setitem(sys.modules,'kaggle',types.SimpleNamespace(api=object()))
    monkeypatch.setenv('KAGGLE_API_TOKEN','fictional-previous')
    monkeypatch.setattr(campaign.time,'sleep',lambda duration:sleeps.append(duration))
    def child(argv,**kwargs):
        calls.append(argv[-1]);index=len(calls)
        run=root/'session/uk-insurance-QA-data-local/production-v1';run.mkdir(parents=True)
        total=1000 if index==3 else 412+index
        status='target_reached_pending_human_review' if total==1000 else 'paused_cloud_batch_limit'
        if scenario=='daily_quota':status='paused_provider_quota'
        state={'status':status,'generation':[{'package_id':'fictional-package'}],
            'decisions':{'fictional-row':{'status':'accepted','judge':{'fictional':True}}},
            'resume_after_epoch':campaign.time.time()+86400,'paused_provider':'gemini'}
        def write(name,value): (run/name).write_text(json.dumps(value))
        write('state.json',state);write('campaign_start.json',{'generation_count':0,'generated_ids':[],'eligible_package_count':536})
        (run/'generated_candidates.jsonl').write_text(json.dumps({'example_id':'fictional-row'})+'\n')
        write('cloud_quality_report.json',{'passed':True,'accepted':total})
        (root/'persistence_receipt.json').write_text(json.dumps({'version':10+index,'accepted':total,'is_private':True}))
        return types.SimpleNamespace(returncode=1 if scenario=='unsafe_child' else 0)
    monkeypatch.setattr(campaign.subprocess,'run',child)
    monkeypatch.setattr(campaign,'report',lambda _: {'total_accepted':1000 if len(calls)==3 else 412+len(calls),'candidates_generated':608+len(calls)})
    import runner
    def persist(*_):
        published.append(True)
        receipt=json.loads((root/'persistence_receipt.json').read_text());receipt['version']+=1
        (root/'persistence_receipt.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(runner,'persist',persist)
    campaign.run()
    result=json.loads((root/'campaign_report.json').read_text())
    if scenario=='target':
        assert len(calls)==3 and 'max_packages=1' in calls[0] and 'max_packages=5' in calls[1]
        assert result['stop_reason']=='target_reached' and result['remaining_to_target']==0
        assert published==[True]
    elif scenario=='unsafe_child':
        assert len(calls)==1 and not published
        assert 'failed' in result['stop_reason']
    else:
        assert len(calls)==1 and not sleeps
        assert 'Quota wait exceeds remaining session' in result['stop_reason']
