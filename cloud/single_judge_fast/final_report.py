"""Read aggregate tables only after a normal, readback-verified coordinator finish."""
import json
from pathlib import Path
from cloud.single_judge_fast.protocol import PROTOCOL


def report(root,result,state):
    if result.get('status') not in ('COMPLETE','PARTIAL_STOPPED'):
        raise RuntimeError('Coordinator did not finish through a verified checkpoint')
    persistence=state['last_persistence']
    if not persistence.get('private') or not persistence.get('readback_verified') or result['checkpoint']!=persistence['version']:
        raise RuntimeError('Final checkpoint readback not verified')
    table=json.loads((Path(root)/'results_single_judge_fast_v1/comparison.json').read_text())
    if len(table)!=16 or any(r['protocol']!=PROTOCOL or r['benchmark_questions']!=76 for r in table):
        raise RuntimeError('Comparison population differs')
    covered=sum(r['judged'] for r in table)
    if covered!=result['unique_judged'] or covered+result['remaining']!=912:
        raise RuntimeError('Comparison coverage differs from verified result')
    unavailable=[r for r in table if not r['available']]
    if len(unavailable)!=4 or any(r['responses']!=0 or r['correctness'] is not None for r in unavailable):
        raise RuntimeError('Unavailable model conditions must remain N/A')
    return {'protocol':PROTOCOL,'complete':covered==912,'unique_judged':covered,'remaining':912-covered,
            'checkpoint':persistence['version'],'private_readback_verified':True,
            'comparison':table,
            'methodological_limitation':'Mixed-provider single valid judgement, deterministic Groq-first historical tie rule; not dual-judge equivalence or inter-judge agreement. Missing semantic votes are not zero; unavailable DeepSeek conditions remain N/A.'}
