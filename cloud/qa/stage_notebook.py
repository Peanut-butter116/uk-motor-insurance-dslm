"""Create a code-only Kaggle notebook; upload with --no-run until secrets exist."""
import argparse
import json
from pathlib import Path


def stage(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# Resume existing UK motor QA production\n',
        'Private, CPU-only; Groq GPT-OSS → Gemini. Add and enable `GROQ_API_KEY`, ',
        '`GEMINI_API_KEY`, `KAGGLE_API_TOKEN` in Add-ons → Secrets before Run All.\n',
        'First run: one source package, cap 500. No scheduler. Do not run concurrent sessions. ',
        'All outputs persist as versioned private Kaggle dataset checkpoints.\n']},
        {'cell_type': 'code', 'execution_count': None, 'outputs': [], 'metadata': {},
         'source': ["%pip install -q kaggle==2.2.5 groq==1.7.0 tiktoken==0.14.0 jsonschema\n"]}]
    code = "from pathlib import Path\nimport sys\np = Path('/kaggle/working/qa_adapter'); p.mkdir(exist_ok=True)\n"
    for name in ('state.py', 'runner.py', 'quality.py'):
        code += '(p / '+repr(name)+').write_text('+repr((here/name).read_text())+')\n'
    code += "sys.path.insert(0, str(p))\nfrom runner import main\nmain(max_packages=1, target=500)\n"
    cells.append({'cell_type': 'code', 'execution_count': None, 'outputs': [], 'metadata': {}, 'source': code.splitlines(True)})
    (output/'resume.ipynb').write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}}, 'cells': cells}, indent=2)+'\n')
    (output/'kernel-metadata.json').write_text(json.dumps({
        'id': 'scarletthe0116/uk-motor-qa-resume', 'title': 'uk-motor-qa-resume',
        'code_file': 'resume.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': False, 'enable_internet': True,
        'dataset_sources': ['scarletthe0116/uk-motor-qa-production-v1'],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': []}, indent=2)+'\n')
    print('Code-only notebook staged')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    stage(**vars(p.parse_args()))
