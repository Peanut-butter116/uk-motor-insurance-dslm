"""Stage an independent provider-free restore audit pinned to an aggregate receipt."""
import argparse
import json
from pathlib import Path


def stage(output, receipt):
    receipt = json.loads(Path(receipt).read_text())
    assert receipt['dataset_private'] is True
    version = receipt['persisted_dataset_version']
    checksum = receipt['persisted_archive_sha256']
    accepted = receipt['total_accepted']
    assert isinstance(version,int) and isinstance(accepted,int) and len(checksum)==64
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=(Path(__file__).parent/'audit_saved_batch.py').read_text()
    # Override only pinned aggregate expectations before calling the audit.
    source=source[:source.index("if __name__ == '__main__':")]
    source+=f'EXPECTED = {checksum!r}\nVERSION = {version!r}\nACCEPTED = {accepted!r}\naudit()\n'
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':['# Provider-free saved-batch audit\n',f'Private checkpoint v{version}; no model calls, network disabled.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(True)}]}
    (output/'audit.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    meta={'id':'scarletthe0116/uk-motor-qa-batch-audit','title':'uk-motor-qa-batch-audit','code_file':'audit.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_internet':False,
        'dataset_sources':['scarletthe0116/uk-motor-qa-production-v1'],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--receipt',required=True)
    stage(**vars(parser.parse_args()))
