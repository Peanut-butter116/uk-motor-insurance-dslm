import time
import pytest
from cloud.single_judge_fast.after_calibration import validate_transition


def valid():
    now=time.time();deadline=now+3600;launch=deadline-72000
    control={'protocol':'d1000-single-judge-fast-v1','phase':'suspended','launch_epoch':launch,'deadline_epoch':deadline,'baseline_unique':349}
    proof={'phase':'suspended','active_provider_threads':0,'active_provider_tasks':0,'expected_checkpoint':411,'lease_version':2,'unique_judged':774}
    alternative={'phase':'calibration_failed','calibration':'FAIL','new_votes':0,'source_checkpoint':411,'lease_version':2,'original_deadline_epoch':deadline}
    persistence={'private':True,'readback_verified':True,'version':412}
    return control,proof,alternative,persistence,2


def test_failed_alternative_can_return_only_to_original_protocol():
    validate_transition(*valid())


@pytest.mark.parametrize('target,key,value',[(1,'active_provider_threads',1),(2,'new_votes',1),
    (2,'phase','running'),(2,'source_checkpoint',410),(2,'lease_version',3),
    (3,'readback_verified',False),(3,'version',411),(2,'original_deadline_epoch',0)])
def test_unsafe_transition_is_rejected(target,key,value):
    args=list(valid());args[target][key]=value
    with pytest.raises(RuntimeError):validate_transition(*args)
