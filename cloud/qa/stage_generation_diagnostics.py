"""Private, read-only cloud audit of generation failure metadata; no credentials."""
import argparse
import json
from pathlib import Path

CODE = '''import collections, hashlib, json, tarfile
import jsonschema
from pathlib import Path
roots=list(Path('/kaggle/input').rglob('migration_manifest.json'))
assert len(roots)==1
manifest=json.loads(roots[0].read_text());archive=roots[0].parent/'checkpoint.bin'
h=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
assert h.hexdigest()==manifest['archive_sha256']
with tarfile.open(archive) as tar:
    base='uk-insurance-QA-data-local/production-v1/'
    def read(name):
        payload=tar.extractfile(name).read()
        assert hashlib.sha256(payload).hexdigest()==manifest['files'][name]
        return json.loads(payload)
    state=read(base+'fast_unverified_v1/state.json')
    schema_name='uk-insurance-QA-data/src/response_contracts.py'
    schema_source=tar.extractfile(schema_name).read()
    assert hashlib.sha256(schema_source).hexdigest()==manifest['files'][schema_name]
    namespace={};exec(schema_source,namespace)
    _,schema=namespace['contract']('teacher','')
    assert state.get('execution_mode')=='four_generator_unverified_v1'
    workers=state['worker_specs']; failures=collections.defaultdict(collections.Counter)
    rawstats=collections.defaultdict(lambda:collections.Counter())
    for job in state['jobs'].values():
        if job.get('worker_id') in workers and job.get('failure'):
            failures[job['worker_id']][job['failure']]+=1
    since=state['upgrade_started_epoch']
    for member in tar.getmembers():
        if '/provider_cache/' not in member.name or not member.name.endswith('.attempts.json'):continue
        if member.mtime<since:continue
        for attempt in read(member.name):
            if attempt.get('status')!='structured_failure':continue
            rawname=str(Path(member.name).parent/(attempt['request_sha256']+'.raw.json'))
            try:rawmember=tar.getmember(rawname)
            except KeyError:continue
            if rawmember.mtime<since:continue
            response=read(rawname);model=response.get('model','unknown')
            counter=rawstats[model];counter['failure:'+attempt.get('kind','unknown')]+=1
            for choice in response.get('choices',[]):
                counter['finish:'+str(choice.get('finish_reason'))]+=1
                content=choice.get('message',{}).get('content')
                counter['content_characters']+=len(content) if isinstance(content,str) else 0
                if attempt.get('kind')=='schema_validation_failed' and isinstance(content,str):
                    text=content.strip()
                    if text.startswith('```'):
                        text=text.split('\\n',1)[-1].rsplit('```',1)[0].strip()
                    try:answer=json.loads(text)
                    except ValueError:counter['diagnostic_parse_failure']+=1;continue
                    for error in jsonschema.Draft202012Validator(schema).iter_errors(answer):
                        label='schema:'+str(error.validator)+':'+ '/'.join(map(str,error.schema_path))
                        if error.validator=='required' and isinstance(error.instance,dict):
                            for field in error.validator_value:
                                if field not in error.instance:counter[label+':missing-'+field]+=1
                        else:counter[label+':actual-'+type(error.instance).__name__]+=1
            usage=response.get('usage',{});counter['completion_tokens']+=usage.get('completion_tokens',0)
            details=usage.get('completion_tokens_details') or {}
            counter['reasoning_tokens']+=details.get('reasoning_tokens',0)
result={'archive_sha256':manifest['archive_sha256'],'manifest_created_at_utc':manifest['created_at_utc'],
        'worker_specs':workers,'worker_package_failures':dict(failures),'new_structured_failures_by_response_model':dict(rawstats),
        'production_writes':0,'model_calls':0,'raw_records_downloaded_locally':False}
Path('/kaggle/working/generation_diagnostics.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result,indent=2))
'''


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':CODE.splitlines(True)}]}
    (output/'audit.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    meta={'id':'scarletthe0116/uk-motor-qa-batch-audit','title':'uk-motor-qa-batch-audit','code_file':'audit.ipynb',
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_internet':False,
          'dataset_sources':['scarletthe0116/uk-motor-qa-production-v1'],
          'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(**vars(p.parse_args()))
