import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time
import xml.etree.ElementTree as ET
import pytest
from cloud.single_judge_fast.reporting import bootstrap_mean,mean_parts
from cloud.single_judge_fast.presentation_continuation import validate_entry,NOT_BEFORE,ORIGINAL_DEADLINE

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('d1000_report_builder',ROOT/'scripts/build_d1000_results.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def payload():return json.loads((ROOT/'reports/d1000_final/verified_aggregates.json').read_text())


def test_missing_semantic_scores_are_not_zero_and_denominator_is_observed():
    assert mean_parts([1,None,0.5])==(.75,1.5,2)
    assert mean_parts([None])==(None,0,0)
    p=builder.validate(payload())
    q=next(r for r in p['comparison'] if r['model_key']=='qwen3b' and r['condition']=='base-rag')
    assert q['correctness']==29.5/62 and q['correctness']!=29.5/76
    assert q['benchmark_questions']==76 and q['missing']==14


def test_absent_support_is_na_even_when_one_citation_resolves():
    q=next(r for r in payload()['comparison'] if r['model_key']=='smollm1b7' and r['condition']=='base-rag')
    assert q['citation_resolution']==1 and q['citation_resolution_denominator']==1
    assert q['citation_support'] is None and q['citation_support_denominator']==0


def test_deterministic_metrics_keep_all_76_questions():
    for r in payload()['comparison']:
        if not r['available']:continue
        assert r['parse_rate_denominator']==76
        assert r['appropriate_refusal_rate_denominator']==9
        assert r['false_refusal_rate_denominator']==67
        ar=r['appropriate_refusal_rate_numerator'];fr=r['false_refusal_rate_numerator']
        assert r['balanced_abstention']==pytest.approx(.5*((67-fr)/67+ar/9),abs=1e-14)


def test_deepseek_is_unavailable_not_failed_zero_score():
    rr=[r for r in payload()['comparison'] if not r['available']]
    assert len(rr)==4
    assert all(r['correctness'] is None and r['responses']==0 and r['missing'] is None for r in rr)


@pytest.mark.parametrize('key,value',[('value',0),('denominator',76),('checkpoint',1)])
def test_inconsistent_canonical_metric_is_rejected(key,value):
    p=payload();r=next(r for r in p['metrics'] if r['model_key']=='qwen3b' and r['condition']=='base-rag' and r['metric']=='correctness')
    r[key]=value
    with pytest.raises(AssertionError):builder.validate(p)


def test_bootstrap_empty_single_and_reproducible():
    assert bootstrap_mean([])==(None,None)
    assert bootstrap_mean([.5])==(.5,.5)
    assert bootstrap_mean([0,1,.5,None])==bootstrap_mean([0,1,.5,None])


def test_artifacts_use_real_table_and_high_resolution_vectors():
    folder=ROOT/'reports/d1000_final/figures';manifest=json.loads((folder/'manifest.json').read_text())
    assert len(manifest['files'])==18
    for name,digest in manifest['files'].items():
        path=folder/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        if name.endswith('.png'):
            assert struct.unpack('>II',path.read_bytes()[16:24])==(4800,2700)
        if name.endswith('.svg'):
            svg=ET.fromstring(path.read_text());assert svg.attrib['width']=='1152pt' and svg.attrib['height']=='648pt'
    assert manifest['canonical_input_sha256']==hashlib.sha256((ROOT/'reports/d1000_final/metrics_summary.json').read_bytes()).hexdigest()


def test_safe_aggregate_has_no_raw_private_fields():
    banned={'question','model_answer','reference_answer','key_facts','reference_evidence','request_sha256','authorization','GROQ_API_KEY','GEMINI_API_KEY','KAGGLE_API_TOKEN'}
    def walk(value):
        if isinstance(value,dict):
            assert not banned.intersection(value)
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    walk(payload())


def test_historical_readme_snapshots_unchanged():
    folder=ROOT/'archive/d1000-predecessor';m=json.loads((folder/'manifest.json').read_text())
    assert all(hashlib.sha256((folder/name).read_bytes()).hexdigest()==h for name,h in m['files'].items())


def test_continuation_cannot_start_early_or_change_historical_deadline():
    state={'last_persistence':{'version':428,'private':True,'readback_verified':True},'single_judge_fast':{'deadline_epoch':ORIGINAL_DEADLINE}}
    result={'status':'PARTIAL_STOPPED','checkpoint':427}
    validate_entry(result,state,NOT_BEFORE,NOT_BEFORE+3600,NOT_BEFORE-500)
    with pytest.raises(RuntimeError):validate_entry(result,state,NOT_BEFORE-1,NOT_BEFORE+3600,NOT_BEFORE-500)
    state['single_judge_fast']['deadline_epoch']+=1
    with pytest.raises(RuntimeError):validate_entry(result,state,NOT_BEFORE,NOT_BEFORE+3600,NOT_BEFORE-500)
