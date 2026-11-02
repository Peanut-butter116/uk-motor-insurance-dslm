"""A separate mixed-model extension, retaining the original transport validators."""
PROTOCOL = 'd1000-single-judge-third-recovery-v1'
MODEL = 'gemini-3.1-flash-lite'
PROVIDER = 'gemini'


def select(votes):
    # The third model can never displace any valid historical judgement.
    chosen = {i: ('gemini31', v) for i, v in votes.get('gemini31', {}).items()}
    chosen.update({i: ('gemini', v) for i, v in votes.get('gemini', {}).items()})
    chosen.update({i: ('groq', v) for i, v in votes.get('groq', {}).items()})
    return chosen


def eligible(candidates, votes, protected):
    covered = set(select(votes))
    return [t for t, _ in candidates if t['evaluation_id'] not in covered | set(protected)]


def calibration_pass(task, expected, answer):
    passed = answer['correctness'] == expected
    if task['deterministic_abstention']:
        return passed and answer['evidence_support'] is None and answer['abstention_category'] == (
            'incorrect_abstention' if task['gold_answerable'] else 'correct_abstention')
    return passed and answer['abstention_category'] not in ('correct_abstention', 'incorrect_abstention')
