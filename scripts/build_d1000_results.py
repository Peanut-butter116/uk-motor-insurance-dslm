#!/usr/bin/env python3
"""Validate cloud-derived safe aggregates and build one canonical reporting table.

No API access or raw research data is required by this local reporting step.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MODEL_ORDER=('qwen3b','ministral3b','smollm1b7','deepseek1b5')
NAMES={'qwen3b':'Qwen2.5-3B','ministral3b':'Ministral-3B','smollm1b7':'SmolLM2-1.7B','deepseek1b5':'DeepSeek (unavailable)'}
CONDITIONS={'base':'Base','base-rag':'Base + RAG','sft':'SFT','sft-rag':'SFT + RAG'}


def validate(payload):
    p=payload['provenance'];table=payload['metrics'];wide=payload['comparison']
    assert payload['schema_version']=='d1000-safe-aggregates-v1'
    assert p['protocol']=='d1000-single-judge-fast-v1'
    assert p['responses']==912 and p['benchmark_questions']==76
    assert len(wide)==16 and sum(r['available'] for r in wide)==12
    assert sum(r['judged'] for r in wide)==p['unique_judged']
    assert p['remaining']==912-p['unique_judged']==p['eligible_remaining']+p['quarantined_unjudged']
    assert sum(r['quarantined'] for r in wide)==p['quarantined_unjudged']
    assert len({(r['model_key'],r['condition'],r['metric']) for r in table})==len(table)==160
    index={(r['model_key'],r['condition']):r for r in wide}
    for r in table:
        assert r['checkpoint']==p['checkpoint'] and r['archive_sha256']==p['archive_sha256']
        w=index[r['model_key'],r['condition']]
        assert r['benchmark_questions']==76 and 0<=r['judged']<=76
        assert r['quarantined']<=76-r['judged']
        value=r['value'];den=r['denominator'];num=r['numerator']
        if not r['available']:
            assert value is None and num is None and den==0 and not w['available']
        elif den:
            assert value is not None and math.isclose(value,num/den,abs_tol=1e-12)
            assert 0<=value<=1 and math.isfinite(value)
        else:assert value is None
        if r['metric'] in ('correctness','reference_support'):assert den<=r['judged']
        if r['metric']=='parse_rate' and r['available']:assert den==76
        if r['metric']=='appropriate_refusal_rate' and r['available']:assert den==9
        if r['metric']=='false_refusal_rate' and r['available']:assert den==67
        assert w[r['metric']]==value
    return payload


def pct(value):return 'N/A' if value is None else f'{100*value:.1f}%'


def comparison_markdown(payload):
    lines=['| Model | Condition | Correctness¹ | Judged / 76 | Balanced abstention² | Citation support³ | Citation-ID resolution | Parse⁴ | False refusal² |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    rows=sorted(payload['comparison'],key=lambda r:(MODEL_ORDER.index(r['model_key']),list(CONDITIONS).index(r['condition'])))
    for r in rows:
        if not r['available']:continue
        lines.append('| '+ ' | '.join([NAMES[r['model_key']],CONDITIONS[r['condition']],pct(r['correctness']),str(r['judged'])+'/76',
                                      pct(r['balanced_abstention']),pct(r['citation_support']),pct(r['citation_resolution']),pct(r['parse_rate']),pct(r['false_refusal_rate'])])+' |')
    lines.extend(['','¹ Mean rubric score (0 / 0.5 / 1), conditional on valid non-null semantic judgements; missing is not zero. See canonical table for score sums, denominators and conditional bootstrap intervals.',
                  '² Deterministic metrics use all 76 saved responses: 67 answerable and 9 unanswerable. Balanced abstention = ½(answerable non-refusal rate + unanswerable refusal rate).',
                  '³ Citation support uses only supplied, scored citation occurrences; denominators vary. No-RAG is N/A. SmolLM Base+RAG has one resolving ID but zero scored support occurrences, so support is N/A.',
                  '⁴ The inherited parse metric recognises a refusal or a parseable citation. It is **not** a measure of structured JSON generation compliance.',
                  '', 'DeepSeek’s four conditions are unavailable (N/A), excluded from the 912-response population.'])
    return '\n'.join(lines)+'\n'


def build(source,out):
    payload=validate(json.loads(Path(source).read_text()));out=Path(out);out.mkdir(parents=True,exist_ok=True)
    payload['report_build']={'input_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                           'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'metrics_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    for name,data in [('metrics_summary',payload['metrics']),('comparison',payload['comparison'])]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            fields=list(dict.fromkeys(k for r in data for k in r))
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    (out/'sft_effects.json').write_text(json.dumps(payload['sft_effects'],indent=2)+'\n')
    (out/'coverage_summary.json').write_text(json.dumps(payload['provenance'],indent=2,sort_keys=True)+'\n')
    (out/'comparison.md').write_text(comparison_markdown(payload))
    return payload


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='reports/d1000_final/verified_aggregates.json');ap.add_argument('--output',default='reports/d1000_final')
    args=ap.parse_args();p=build(args.input,args.output)
    print(json.dumps({k:p['provenance'][k] for k in ('checkpoint','status','unique_judged','remaining')}))
