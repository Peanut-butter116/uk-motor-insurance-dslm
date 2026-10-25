"""Stage a cloud-only, provider-free migration integrity check."""
import argparse
import json
from pathlib import Path
from stage_notebook import stage


def main(output):
    stage(output)
    output = Path(output)
    metadata = json.loads((output/'kernel-metadata.json').read_text())
    metadata.update(id='scarletthe0116/uk-motor-qa-checkpoint-check', title='uk-motor-qa-checkpoint-check')
    (output/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    code = '''import hashlib, json, tarfile
from pathlib import Path
root = Path('/kaggle/input/uk-motor-qa-production-v1')
if not root.exists():
    matches = list(Path('/kaggle/input').rglob('migration_manifest.json'))
    assert len(matches) == 1
    root = matches[0].parent
manifest = json.loads((root/'migration_manifest.json').read_text())
archive = root/'checkpoint.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256']
dest = Path('/kaggle/working/checked'); dest.mkdir()
with tarfile.open(archive) as tar:
    tar.extractall(dest, filter='data')
for name, checksum in manifest['files'].items():
    assert hashlib.sha256((dest/name).read_bytes()).hexdigest() == checksum
run = dest/'uk-insurance-QA-data-local/production-v1'
rows = [json.loads(line) for line in (run/'data/final/train.jsonl').read_text().splitlines()]
assert len(rows) == 409 and len({r['example_id'] for r in rows}) == 409
state = json.loads((run/'state.json').read_text())
assert sum(d['status']=='accepted' for d in state['decisions'].values()) == 409
original_manifest = json.loads((run/'data/final/dataset_manifest.json').read_text())
gold = dest/'uk-insurance-dslm-data/data/gold/gold_v2.jsonl'
assert hashlib.sha256(gold.read_bytes()).hexdigest() == original_manifest['original_gold_sha256']
result = {'status':'verified', 'accepted':409, 'files_verified':len(manifest['files']), 'archive_sha256':manifest['archive_sha256'], 'frozen_gold_unchanged':True, 'provider_calls':0}
Path('/kaggle/working/migration_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
# Restricted extracted state is temporary cloud scratch, not notebook output.
import shutil
shutil.rmtree(dest)
'''
    notebook = json.loads((output/'resume.ipynb').read_text())
    notebook['cells'] = [{'cell_type': 'code', 'execution_count': None, 'outputs': [], 'metadata': {}, 'source': code.splitlines(True)}]
    (output/'resume.ipynb').write_text(json.dumps(notebook, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    main(**vars(parser.parse_args()))
