"""Stage the four-model upgrade separately; existing cloud writer stays untouched."""
import argparse
import json
from pathlib import Path
from stage_fast import main as stage_fast


def main(output):
    stage_fast(output)
    path=Path(output)/'resume.ipynb';nb=json.loads(path.read_text())
    source=(Path(__file__).parent/'four_campaign.py').read_text()
    code=''.join(nb['cells'][-1]['source'])
    old='from fast_campaign import run\nawait run()\n'
    assert code.count(old)==1
    code=code.replace(old,'(p / "four_campaign.py").write_text('+repr(source)+')\nfrom four_campaign import run\nawait run()\n')
    nb['cells'][-1]['source']=code.splitlines(True)
    nb['cells'][0]['source']=['# Four distinct generators — guarded handoff, UNVERIFIED\n',
        'Authenticated model-list checks first. Read-only wait for execution14 to finish. ',
        'No second writer or premature generation. Then restore newest private checkpoint, ',
        'smoke all four workers, adaptive provider concurrency, four-hour generation cap. ',
        'Existing Groq120B/Gemini3.5 retained; add Groq20B and a distinct supported free Gemini. ',
        'Same three secrets. No LLM judge. All historical rows and benchmark preserved.\n']
    path.write_text(json.dumps(nb,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(**vars(parser.parse_args()))
