"""Stage the explicitly authorised generation-only mode, without starting it."""
import argparse
import json
from pathlib import Path
from stage_notebook import stage


def main(output):
    stage(output)
    path = Path(output)/'resume.ipynb'
    nb = json.loads(path.read_text())
    code = ''.join(nb['cells'][-1]['source'])
    old = 'from runner import main\nmain(max_packages=1, target=500)\n'
    assert code.count(old) == 1
    source = (Path(__file__).parent/'fast_campaign.py').read_text()
    code = code.replace(old, '(p / "fast_campaign.py").write_text('+repr(source)+')\nfrom fast_campaign import run\nrun()\n')
    nb['cells'][-1]['source'] = code.splitlines(True)
    nb['cells'][0]['source'] = ['# Parallel Groq + Gemini generation — UNVERIFIED\n',
        'Owner-authorised generation-only campaign to 1,000 combined records. ',
        'No independent LLM judge. Historical verified export unchanged. ',
        'Separate private fast_unverified_v1 partition; deterministic safeguards only. ',
        'Two independent workers, exclusive packages, free quotas, four-hour cap. ',
        'Enable the existing three Kaggle secrets and Save & Run All.\n']
    path.write_text(json.dumps(nb, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    main(**vars(parser.parse_args()))
