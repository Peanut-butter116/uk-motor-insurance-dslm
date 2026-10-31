"""Generation-only concurrency and preservation contracts with fictional records."""
import asyncio
import ast
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import pytest

QA = Path(__file__).resolve().parents[1]/'cloud/qa'
sys.path.insert(0, str(QA))
import fast_campaign as fast


def test_exclusive_assignment_and_resume():
    queue = [{'id': str(i)} for i in range(8)]; jobs = {}
    a = fast.assign(queue, jobs, 'groq', 3)
    b = fast.assign(queue, jobs, 'gemini', 3)
    assert not set(a)&set(b)
    assert fast.assign(queue, jobs, 'groq', 3) == a
    jobs[a[0]]['status'] = 'completed'
    assert a[0] not in fast.assign(queue, jobs, 'groq', 3)


def test_staged_fast_mode_leaves_legacy_available(tmp_path):
    import stage_fast
    stage_fast.main(tmp_path)
    nb = json.loads((tmp_path/'resume.ipynb').read_text())
    code = ''.join(nb['cells'][-1]['source'])
    assert 'from fast_campaign import run\nawait run()' in code
    assert 'UNVERIFIED' in ''.join(nb['cells'][0]['source'])
    assert 'main(max_packages=1, target=500)' not in code
    assert 'from campaign import run' not in code
    assert (QA/'campaign.py').exists()
    compile(code, 'staged-fast', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_notebook_entry_uses_existing_event_loop(monkeypatch, tmp_path):
    """Kaggle/IPython already runs an event loop; the entry must be awaitable."""
    import inspect
    assert inspect.iscoroutinefunction(fast.run)
    monkeypatch.setattr(fast, 'require_cloud', lambda: None)
    monkeypatch.setitem(sys.modules, 'kaggle_secrets', types.SimpleNamespace(
        UserSecretsClient=lambda:types.SimpleNamespace(get_secret=lambda key:'fictional')))
    monkeypatch.setitem(sys.modules, 'kaggle', types.SimpleNamespace(api=object()))
    session = tmp_path/'session';session.mkdir()
    monkeypatch.setattr(fast, 'restore', lambda api:(session,217))
    calls=[]
    async def fake_campaign(*args):calls.append(args)
    monkeypatch.setattr(fast, 'campaign', fake_campaign)
    monkeypatch.setattr(fast.shutil, 'copyfile', lambda *args:None)
    for key in ('GROQ_API_KEY','GEMINI_API_KEY','KAGGLE_API_TOKEN'):
        monkeypatch.setenv(key, 'fictional')
    async def existing_notebook_loop():await fast.run()
    asyncio.run(existing_notebook_loop())
    assert calls and calls[0][2] == 217


@pytest.mark.parametrize('quota_on_groq,four_mode', [(False,False),(True,False),(False,True)])
def test_two_workers_overlap_preserve_verified_and_cap_target(tmp_path, monkeypatch, quota_on_groq, four_mode):
    import runner
    from state import sha
    session = tmp_path/'session'
    run = session/'uk-insurance-QA-data-local/production-v1';run.mkdir(parents=True)
    final = run/'data/final'; final.mkdir(parents=True)
    verified = [{'example_id': 'old'+str(i), 'question': 'old question '+str(i)} for i in range(990 if four_mode else 996)]
    train = final/'train.jsonl'; train.write_text(''.join(json.dumps(r)+'\n' for r in verified))
    gold = session/'uk-insurance-dslm-data/data/gold/gold_v2.jsonl';gold.parent.mkdir(parents=True);gold.write_text('')
    (run/'state.json').write_text(json.dumps({'generation': []}))
    if four_mode:
        prior = run/'fast_unverified_v1';prior.mkdir()
        old_fast=[{'example_id':'prior-'+p,'question':'prior '+p,'answerability':'answerable',
                   'generation_model':m,'provenance':{'teacher_provider':p}} for p,m in fast.MODELS.items()]
        (prior/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old_fast))
    before = sha(train), sha(gold)
    working = tmp_path/'working';working.mkdir()
    monkeypatch.setattr(fast, 'Path', lambda p: working/str(p).removeprefix('/kaggle/working').lstrip('/')
                        if str(p).startswith('/kaggle/working') else Path(p))
    monkeypatch.setattr(fast, 'technical_reasons', lambda row, docs, legacy, questions, pages:
                        ['normalised_duplicate_question'] if row['question'] in questions else [])
    class Safe(Exception): pass
    class Structured(Exception): pass
    class Quota(Safe):
        def __init__(self, provider, until):self.provider, self.until = provider, until
    calls = []
    def complete(values, provider, model, messages, run, max_tokens, schema, stage):
        assert stage == 'fast_generation_unverified_v1'
        assert messages[1]['content'].startswith('original teacher prompt')
        calls.append(provider);time.sleep(.05)
        if quota_on_groq and provider == 'groq':
            raise Quota(provider, time.time()+3600)
        return {'provider': provider, 'model': model, 'request_sha256': provider+'req',
                'response_sha256': provider+'resp', 'answer': {'records': [{'question': model+str(i)} for i in range(4 if quota_on_groq else 2)]}}
    providers = types.ModuleType('src.providers')
    providers.verify_receipt = lambda *args: {}
    providers.complete = complete;providers.QuotaPause = Quota
    src = types.ModuleType('src');src.providers = providers
    modules = {'src': src, 'src.providers': providers,
               'src.kelly': types.SimpleNamespace(SafeAPIError=Safe, StructuredOutputError=Structured),
               'src.response_contracts': types.SimpleNamespace(contract=lambda *args: ('generation', {})),
               'src.common': types.SimpleNamespace(norm=str.casefold,
                  write_rows=lambda p, rows: p.write_text(''.join(json.dumps(r)+'\n' for r in rows)))}
    for name, value in modules.items():monkeypatch.setitem(sys.modules, name, value)
    packages = [{'id': str(i), 'document': {'insurer': 'Hastings Direct'}, 'context': 'fictional evidence'} for i in range(4)]
    gen = types.SimpleNamespace(packages=lambda _: packages, TEACHER_PROMPT='original teacher prompt',
        candidate=lambda package, item, result: {'example_id': item['question'], 'question': item['question'],
            'answerability': 'answerable', 'generation_model':result['model'],
            'provenance': {'teacher_provider': result['provider']}})
    monkeypatch.setattr(fast.importlib.util, 'spec_from_file_location',
        lambda *args: types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None)))
    monkeypatch.setattr(fast.importlib.util, 'module_from_spec', lambda _: gen)
    monkeypatch.setattr(fast, 'ReusedTransport', lambda *args: types.SimpleNamespace(close=lambda: None))
    version = [1];published = []
    monkeypatch.setattr(fast, 'dataset_info', lambda _: types.SimpleNamespace(current_version_number=version[0]))
    monkeypatch.setattr(runner, 'report', lambda _: {'total_accepted': len(verified)})
    def persist(*args):
        version[0] += 1
        assert (sha(train), sha(gold)) == before
        measured = runner.report(run)['fast_campaign']
        published.append(measured)
        (working/'persistence_receipt.json').write_text(json.dumps({'version': version[0], 'accepted':len(verified)}))
    monkeypatch.setattr(runner, 'persist', persist)
    for key in ('GROQ_API_KEY', 'GEMINI_API_KEY'):monkeypatch.setenv(key, 'fictional')
    specs={p:{'provider':p,'model':m} for p,m in fast.MODELS.items()} if four_mode else None
    if four_mode:
        specs.update(groq_extra={'provider':'groq','model':'openai/gpt-oss-20b'},
                     gemini_extra={'provider':'gemini','model':'gemini-3.1-flash-lite'})
    asyncio.run(fast.campaign(session, object(), 1, worker_specs=specs))
    report = json.loads((working/'fast_campaign_report.json').read_text())
    assert sorted(calls) == (['gemini','gemini','groq','groq'] if four_mode else ['gemini', 'groq'])
    assert report['combined_total'] == 1000
    if quota_on_groq:
        assert report['fast_by_generator'] == {'gemini':4}
        assert report['provider_state']['groq']['status'] == 'quota_wait'
    elif four_mode:
        assert report['fast_by_generator']=={'groq':5,'gemini':5}
        assert report['four_smoke_passed']
        assert all(m['admitted']==2 for m in report['worker_metrics'].values())
        current=json.loads((run/'fast_unverified_v1/train.jsonl').read_text().splitlines()[0])
        assert current==old_fast[0]
    else:
        assert report['fast_by_generator'] == {'groq':2, 'gemini':2}
        assert report['smoke_passed'] and report['concurrent_request_overlap_observed']
    assert report['independent_llm_verification_calls'] == 0
    assert report['status'] == 'target_reached'
    assert published[0]['fast_unverified_count'] == (2 if four_mode else 0)
    assert (sha(train), sha(gold)) == before


def test_duplicate_and_benchmark_checks_are_provider_free(monkeypatch):
    production = types.SimpleNamespace(deterministic=lambda *args: [])
    leakage = types.SimpleNamespace(inspect=lambda *args, **kwargs: ([{'reason':'heldout_question_exact'}], {}))
    monkeypatch.setitem(sys.modules, 'src.production', production)
    monkeypatch.setitem(sys.modules, 'src.check_leakage', leakage)
    monkeypatch.setitem(sys.modules, 'src.common', types.SimpleNamespace(norm=str.casefold))
    production.deterministic = lambda row, docs, findings: list(findings)
    row = {'question':'Repeated question', 'answerability':'answerable',
           'evidence_text':[{'policy_id':'fictional.pdf','page':4}]}
    reasons = fast.technical_reasons(row, {}, Path('/fictional'), {'repeated question'}, {('fictional.pdf','4')})
    assert set(reasons) == {'heldout_question_exact','normalised_duplicate_question','protected_supporting_page_overlap'}
