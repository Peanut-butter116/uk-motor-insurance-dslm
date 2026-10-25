"""Save a two-comparison repair notebook; never start execution through this tool."""
import argparse
import json
from pathlib import Path
from stage_notebook import stage


def main(output):
    stage(output)
    output = Path(output); here = Path(__file__).resolve().parent
    notebook = json.loads((output/'resume.ipynb').read_text())
    cell = ''.join(notebook['cells'][-1]['source'])
    original = 'from runner import main\nmain(max_packages=1, target=500)\n'
    assert cell.count(original) == 1
    cell = cell.replace(original, '(p / "repair_p42.py").write_text('+repr((here/'repair_p42.py').read_text())+')\nfrom repair_p42 import repair\nrepair()\n')
    notebook['cells'][-1]['source'] = cell.splitlines(True)
    notebook['cells'][0]['source'] = ['# Finish p42 duplicate screening only\n',
        'Requires private checkpoint v9. One Gemini request compares the two missing pairs. ',
        'No teacher calls, no new questions, no new source packages. Existing secrets required.\n']
    (output/'resume.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    meta = json.loads((output/'kernel-metadata.json').read_text())
    meta.update(id='scarletthe0116/uk-motor-qa-checkpoint-check',title='uk-motor-qa-checkpoint-check')
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(**vars(parser.parse_args()))
