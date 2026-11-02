"""Read-only cloud check that Kaggle retains the opaque checkpoint bytes."""
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p=next(Path('/kaggle/input').rglob('artifacts.bin'))
    state=json.loads((p.parent/'progress.json').read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==state['artifacts_sha256']
    with tarfile.open(p) as tar:members=tar.getnames()
    assert state['bootstrap_contains_models_or_data'] is False and not members
    result={'status':'passed','dataset':'scarletthe0116/uk-motor-d1000-models-results',
        'version':2,'opaque_artifact_retained':True,'archive_sha256_verified':True,
        'bootstrap_only':True,'trained_adapters':0,'records':0}
    Path('/kaggle/working/storage_probe_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
