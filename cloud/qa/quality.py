"""Provider-free acceptance audit, preserving the existing QC rubric."""
import re
from difflib import SequenceMatcher


def validate_new_records(build, records, baseline):
    import jsonschema
    from src.production import deterministic, judge_reasons, PAYLOAD
    from src.check_leakage import inspect
    from src.common import digest, norm
    original={r['example_id'] for r in baseline}
    new=[r for r in records if r['example_id'] not in original]
    decisions=build.state['decisions']
    findings,_=inspect(new,build.legacy,protected_splits=('test','dev','seed'))
    failures=[]; comparisons={}
    for row in new:
        key=row['example_id'];decision=decisions[key]
        jsonschema.validate(row,build.schema)
        reasons=deterministic(row,build.docs,[f['reason'] for f in findings if f['example_id']==key])
        reasons+=judge_reasons(row,decision['judge'])
        if decision['input_sha256']!=digest({k:row[k] for k in PAYLOAD}):reasons.append('validation_payload_hash_mismatch')
        if decision.get('all_protected_semantic_screen_passed') is not True:reasons.append('protected_screen_missing')
        if row['answerability']=='unanswerable' and decision.get('full_document_absence_verified') is not True:reasons.append('full_document_check_missing')
        screen=decision.get('training_semantic_screen',{})
        comparisons[key]=set(screen.get('comparison_ids',[]))|set(screen.get('repair_comparison_ids',[]))
        if not original<=comparisons[key]:reasons.append('training_baseline_comparison_missing')
        if reasons:failures.append({'id':key,'reasons':sorted(set(reasons))})
    missing=[(a['example_id'],b['example_id']) for i,a in enumerate(new) for b in new[i+1:]
             if a['example_id'] not in comparisons[b['example_id']] and b['example_id'] not in comparisons[a['example_id']]]
    normalized={r['example_id']:norm(r['question']) for r in records}
    duplicates=[]
    for row in new:
        key=row['example_id']
        for other in records:
            if key==other['example_id']:continue
            match=SequenceMatcher(None,normalized[key],normalized[other['example_id']])
            if match.quick_ratio()>=.9 and match.ratio()>=.9:duplicates.append((key,other['example_id']))
    result={'passed':not(failures or missing or duplicates),'accepted':len(records),'baseline_preserved':len(baseline),
            'new_records_checked':len(new),'grounding_failures':failures,'missing_semantic_pairs':missing,
            'lexical_duplicate_pairs':duplicates,'provider_calls':0}
    if not result['passed']:raise ValueError('Provider-free acceptance audit failed; no further generation permitted')
    return result
