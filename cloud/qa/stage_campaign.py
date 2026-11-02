"""Stage the authorised 1,000-record campaign in the existing private notebook."""
import argparse,json
from pathlib import Path
from stage_notebook import stage


def main(output):
    stage(output);output=Path(output);here=Path(__file__).parent
    p=output/'resume.ipynb';nb=json.loads(p.read_text())
    code=''.join(nb['cells'][-1]['source'])
    old='from runner import main\nmain(max_packages=1, target=500)\n'
    assert code.count(old)==1
    code=code.replace(old,'(p / "campaign.py").write_text('+repr((here/'campaign.py').read_text())+')\nfrom campaign import run\nrun()\n')
    nb['cells'][-1]['source']=code.splitlines(True)
    nb['cells'][0]['source']=['# Authorised QA generation to 1,000 accepted records\n',
        'Fresh package test, then batches of five. Private checkpoints after every export. ',
        'Groq GPT-OSS → Gemini; free-tier quotas only. Same-session quota waits and restarts. ',
        'Four-hour session cap; no cross-session scheduler. Existing three Kaggle secrets required.\n']
    p.write_text(json.dumps(nb,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(**vars(p.parse_args()))
