"""Cloud-only immutable D1000 export. Never changes production or benchmark."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import tarfile
from difflib import SequenceMatcher

ARCHIVE = 'a88e3259cde5357f273f97a9ff2afa0821359815a7238c78332056d16511fe0e'
HASHES = {
    'verified': '9946b87682442ab0948639e80c57a3179059cd563331f1d58fcb8b4ee39448c3',
    'fast': '50e025e4e0204a10dbbc961a12d844c8c99731252b87054f094e49e10f63dd72',
    'original': '6ec5b3728bd6b25cae09d97c81974f371c76a47ce61d2d2795a6919257b44a8f',
    'gold': '233befbf343d93fcef52aa6983e7c34508d531ca203ae9efcd0a7b3ede678234'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def norm(s): return ' '.join(s.casefold().split())


def export(mount, out):
    import jsonschema
    mount, out = Path(mount), Path(out)
    m = json.loads((mount/'migration_manifest.json').read_text())
    assert m['archive_sha256'] == ARCHIVE and sha(mount/'checkpoint.bin') == ARCHIVE
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(mount/'checkpoint.bin') as tar:
        def read(name):
            b = tar.extractfile(name).read()
            assert hashlib.sha256(b).hexdigest() == m['files'][name]
            return b
        def byhash(key):
            name = next(n for n,h in m['files'].items() if h == HASHES[key])
            return [json.loads(s) for s in read(name).decode().splitlines() if s]
        verified, fast, original, gold = [byhash(k) for k in ('verified','fast','original','gold')]
        schema = json.loads(read('uk-insurance-QA-data/config/dataset_schema.json'))
    assert (len(verified),len(fast)) == (462,538)
    rows = verified + fast
    ids = {r['example_id']:r for r in rows}
    assert len(ids) == 1000 and all(ids[r['example_id']] == r for r in original)
    assert all(r['provenance'].get('verification_status') == 'unverified' for r in fast)
    assert dict(collections.Counter(r['generation_model'] for r in fast)) == {
        'openai/gpt-oss-120b':126,'openai/gpt-oss-20b':28,
        'gemini-3.5-flash-lite':158,'gemini-3.1-flash-lite':226}
    for r in rows: jsonschema.validate(r,schema)
    assert len({(r['question'],r['context']) for r in rows}) == 1000
    # Historical context contrast pairs remain intact; question equality alone
    # is not permission to drop a previously accepted row.
    protected = [g for g in gold if g['split'] in ('test','dev')]
    findings=[]
    for r in rows:
        q=norm(r['question'])
        for g in protected:
            oq=norm(g['question'])
            if q==oq or (SequenceMatcher(None,q,oq).quick_ratio()>=.85 and
                         SequenceMatcher(None,q,oq).ratio()>=.85):
                findings.append({'record_id':r['example_id'],'protected_id':g['id'],
                                 'reason':'protected_question_overlap'})
    labels={r['example_id']:('verified' if i<462 else 'unverified') for i,r in enumerate(rows)}
    columns=list(schema['properties'])+['verification_status','source_package','dataset_version']
    file_hashes={}
    for name, subset in [('d1000',rows),('verified_462',verified),('unverified_538',fast)]:
        subset=sorted(subset,key=lambda r:r['example_id'])
        path=out/f'uk_motor_qa_{name}.csv'
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns,quoting=csv.QUOTE_ALL);w.writeheader()
            for r in subset:
                x={k:(json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else '' if v is None else v) for k,v in r.items()}
                x.update(verification_status=labels[r['example_id']],source_package=r['provenance'].get('package_id') or '',dataset_version='production-v1/319')
                w.writerow(x)
        with path.open(encoding='utf-8',newline='') as f: assert len(list(csv.DictReader(f)))==len(subset)
        j=out/f'uk_motor_qa_{name}.jsonl'
        j.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in subset))
        file_hashes.update({path.name:sha(path),j.name:sha(j)})
    report={'records':1000,'verified':462,'unverified':538,'schema_valid':True,
            'unique_ids':True,'unique_model_visible_pairs':True,
            'unique_questions':len({norm(r['question']) for r in rows}),
            'protected_overlap_count':len(findings),'training_integrity_passed':not findings,
            'independent_verification_claim_for_fast':False,'historical_hashes_preserved':True}
    (out/'uk_motor_qa_d1000_quality_report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Restricted findings and IDs stay in the private export only.
    (out/'leakage_findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    from datetime import datetime,timezone
    manifest={'schema':'motor-d1000-export/1','dataset_version':'D1000-production319',
        'source_dataset':'scarletthe0116/uk-motor-qa-production-v1/319','source_archive_sha256':ARCHIVE,
        'source_hashes':HASHES,'records':1000,'verification_counts':{'verified':462,'unverified':538},
        'record_ids':sorted(ids),'verified_ids':sorted(r['example_id'] for r in verified),
        'files':file_hashes,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'csv_contract':'UTF-8; all cells quoted; nested fields JSON; null cells empty; companion JSONL is authoritative/lossless. Import identifiers as text. No values rewritten for spreadsheet formulas.',
        'training_ready':False,'training_clearance':'Owner authorised descriptive mixed-quality training; requires separate tokenisation/leakage/smoke gates.'}
    (out/'uk_motor_qa_d1000_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return report,manifest


def main():
    assert Path('/kaggle/working').is_dir(), 'Cloud execution required'
    hits=list(Path('/kaggle/input').rglob('migration_manifest.json'))
    mount=next(p.parent for p in hits if json.loads(p.read_text()).get('archive_sha256')==ARCHIVE)
    report,manifest=export(mount,'/kaggle/working/export')
    summary={**report,'csv_sha256':manifest['files']['uk_motor_qa_d1000.csv'],
             'source_version':319,'status':'exported_private_kernel_output'}
    Path('/kaggle/working/export_receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__': main()
