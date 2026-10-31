"""Stage the authorized one-package test in the existing secret-enabled notebook."""
import argparse
import json
from pathlib import Path
from stage_notebook import stage


def main(output):
    stage(output)
    output = Path(output); here = Path(__file__).resolve().parent
    notebook = json.loads((output/'resume.ipynb').read_text())
    cell = ''.join(notebook['cells'][-1]['source'])
    cell = cell.replace('from runner import main\nmain(max_packages=1, target=500)\n',
        '(p / "controlled.py").write_text('+repr((here/'controlled.py').read_text())+')\nfrom controlled import run\nrun()\n')
    notebook['cells'][-1]['source'] = cell.splitlines(True)
    (output/'resume.ipynb').write_text(json.dumps(notebook, indent=2)+'\n')
    meta = json.loads((output/'kernel-metadata.json').read_text())
    meta.update(id='scarletthe0116/uk-motor-qa-checkpoint-check', title='uk-motor-qa-checkpoint-check')
    (output/'kernel-metadata.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    main(**vars(parser.parse_args()))
