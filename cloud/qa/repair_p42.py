"""Finish only missing within-package duplicate checks; never generate candidates."""
import ast
import collections
import json
import os
from pathlib import Path
import sys
from runner import DATASET, dataset_info, load_archive, persist, require_cloud
from state import read_rows, report, sha, verify_preserved, write_json


def comparisons(decision):
    screen = decision['training_semantic_screen']
    return set(screen['comparison_ids']) | set(screen.get('repair_comparison_ids', []))


def missing_pairs(records, decisions):
    return [(a['example_id'], b['example_id']) for i,a in enumerate(records) for b in records[i+1:]
            if a['example_id'] not in comparisons(decisions[b['example_id']])
            and b['example_id'] not in comparisons(decisions[a['example_id']])]


def repair():
    require_cloud()
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    values = {key:secrets.get_secret(key) for key in ('GEMINI_API_KEY','KAGGLE_API_TOKEN')}
    os.environ['KAGGLE_API_TOKEN'] = values.pop('KAGGLE_API_TOKEN')
    from kaggle import api
    info = dataset_info(api)
    if info.is_private is not True or info.current_version_number != 9:
        raise ValueError('Repair requires private checkpoint v9; no provider call made')
    incoming = Path('/kaggle/working/repair-input'); incoming.mkdir()
    for name in ('checkpoint.bin','migration_manifest.json'):
        api.dataset_download_file(DATASET+'/9',name,path=str(incoming),quiet=True,force=True,unzip=True)
    manifest = json.loads((incoming/'migration_manifest.json').read_text())
    if manifest['archive_sha256'] != 'cae9a99a112796a520d7256a7c35056e976bed2c025f2c03fd7c2fe558bfedd4':
        raise ValueError('Unexpected v9 checkpoint')
    session = Path('/kaggle/working/session'); session.mkdir()
    load_archive(incoming/'checkpoint.bin',manifest,session)
    source = session/'uk-insurance-QA-data'; run = session/'uk-insurance-QA-data-local/production-v1'
    sys.path.insert(0,str(source))
    from src import kelly,production,providers
    values.update(KELLY_BASE_URL='https://ai.kelly.cloud/v1',KELLY_MAX_SPEND_USD='50',
        KELLY_TEACHER_MODEL='claude-sonnet-4-6',KELLY_JUDGE_MODEL='gpt-5.4',
        QA_TEACHER_PROVIDER='groq',QA_JUDGE_PROVIDER='gemini',
        GROQ_MODEL='openai/gpt-oss-120b',GEMINI_MODEL='gemini-3.5-flash-lite')
    kelly.read_env = production.read_env = lambda *args:values.copy()
    build = production.Build(session/'uk-insurance-dslm-data',session/'uk-insurance-QA-data-local')
    providers.verify_receipt(run,'gemini',build.cfg['judge'])
    baseline = read_rows(run/'cloud_baseline.jsonl'); baseline_ids = {r['example_id'] for r in baseline}
    original = read_rows(run/'data/final/train.jsonl')
    records = [r for r in original if r['example_id'] not in baseline_ids]
    if len(baseline) != 409 or len(records) != 3 or any(r['provenance']['package_id'].split(':')[-1] != 'p42' for r in records):
        raise ValueError('Unexpected repair scope')
    pairs = missing_pairs(records,build.state['decisions'])
    if len(pairs) != 2:
        raise ValueError('Repair is scoped to the two audited missing comparisons')
    anchor = next(key for key in pairs[0] if all(key in pair for pair in pairs))
    reference = next(r for r in records if r['example_id']==anchor)
    candidates = [r for r in records if r['example_id']!=anchor]
    gold = session/'uk-insurance-dslm-data/data/gold/gold_v2.jsonl'; gold_sha = sha(gold)
    constants = [n.value for n in ast.walk(ast.parse((source/'src/protected_screen.py').read_text()))
        if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    prefix = next(value for value in constants if value.startswith('Check each candidate question'))
    data = {'protected_questions':[{'index':0,'question':reference['question']}],
            'candidates':[{'id':r['example_id'],'question':r['question']} for r in candidates]}
    output = {'repair_success':False,'input_version':9,'source_package':'p42','new_candidates_generated':0,
              'missing_pairs_before':len(pairs),'teacher_api_calls':0}
    try:
        response = build.call('judge',prefix+json.dumps(data),4096)
        verdicts = production.parse_batch(response['answer'],[r['example_id'] for r in candidates])
        for row in candidates:
            key=row['example_id'];verdict=verdicts[key];decision=build.state['decisions'][key]
            passed = verdict.get('compared_all') is True and verdict.get('duplicate') is False and verdict.get('protected_matches')==[] and bool(verdict.get('rationale'))
            decision['within_batch_resume_repair']={'model':build.cfg['judge'],'provider':'gemini',
                'comparison_id':anchor,'verdict':verdict,'request_sha256':response['request_sha256'],
                'response_sha256':response['response_sha256']}
            if passed:
                decision['training_semantic_screen'].setdefault('repair_comparison_ids',[]).append(anchor)
            else:
                decision['status']='needs_review'
                decision['reasons']=sorted(set(decision.get('reasons',[])+['within_batch_semantic_duplicate_or_uncertain']))
        for package in build.state['generation']:
            if package['package_id'].split(':')[-1]=='p42':
                package['accepted']=sum(build.state['decisions'][r['example_id']]['status']=='accepted' for r in records)
        build.state['within_batch_repair']={'input_version':9,'missing_pairs_before':2,'model':build.cfg['judge'],
            'request_sha256':response['request_sha256'],'response_sha256':response['response_sha256']}
        build.save('paused_cloud_batch_limit');m=build.exports()
        m.update(target=2000,cloud_run_target=500,complete=False)
        write_json(run/'data/final/dataset_manifest.json',m)
        current=read_rows(run/'data/final/train.jsonl');verify_preserved(baseline,current)
        current_new=[r for r in current if r['example_id'] not in baseline_ids]
        # Preserve canonical QA fields even when adding validation provenance.
        fields=('example_id','question','context','answer','answerability','evidence_ids','citation_ids')
        previous={r['example_id']:r for r in original}
        assert all(all(r[k]==previous[r['example_id']][k] for k in fields) for r in current)
        assert sha(gold)==gold_sha
        assert not (run/'milestones').exists(), 'Never alter an already frozen milestone'
        write_json(run/'cloud_acceptance_order.json',[r['example_id'] for r in baseline]+sorted(r['example_id'] for r in current_new))
        missing=missing_pairs(current_new,build.state['decisions'])
        output.update(report(run))
        output.update(missing_pairs_after=len(missing),baseline_409_preserved=True,frozen_benchmark_unchanged=True,
            changed_candidate_dispositions=dict(collections.Counter(build.state['decisions'][r['example_id']]['status'] for r in records)),
            changed_candidate_failure_reasons=dict(collections.Counter(reason for r in records for reason in build.state['decisions'][r['example_id']].get('reasons',[]))),
            original_qa_fields_preserved=True,repair_success=not missing)
    except kelly.SafeAPIError as error:
        output.update(safe_api_error=str(error))
        build.save('controlled_repair_api_pause')
    finally:
        persist(session,api)
    receipt=json.loads(Path('/kaggle/working/persistence_receipt.json').read_text())
    output.update(persisted_dataset_version=receipt['version'],dataset_private=receipt['is_private'],
                  persisted_archive_sha256=receipt['archive_sha256'])
    write_json(Path('/kaggle/working/controlled_batch_report.json'),output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    repair()
