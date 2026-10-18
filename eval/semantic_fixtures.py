"""Wholly fictional policies, fixed before any protected-label inspection."""
from eval.semantic_judge import TASK_FIELDS


def fixtures():
    policy = ('Fictional Acorn policy: Windscreen replacement has a £150 excess. '
              'Repair has no excess. Business delivery use is excluded. '
              'Personal belongings cover is limited to £200 and excludes cash.')
    rows = [
        ('What is the windscreen replacement excess?', 'The excess is £150.', ['£150 replacement excess'], 'The replacement excess is £150.', True, False, 1),
        ('What is the windscreen replacement excess?', 'The excess is £150.', ['£150 replacement excess'], 'Replacement has no excess.', True, False, 0),
        ('Are personal belongings including cash covered?', 'Belongings up to £200, excluding cash.', ['£200 limit', 'Cash excluded'], 'Belongings are covered up to £200, including cash.', True, False, 0),
        ('What is the repair excess?', 'There is no repair excess.', ['No repair excess'], 'I cannot answer from the provided information.', True, True, 0),
        ('What is the renewal premium?', 'The supplied policy does not specify it.', [], 'I cannot answer from the provided information.', False, True, 1),
        ('What is the renewal premium?', 'The supplied policy does not specify it.', [], 'Your renewal premium is £300.', False, False, 0),
        ('Are personal belongings covered and what is the cash exclusion?', 'Belongings up to £200, excluding cash.', ['£200 limit', 'Cash excluded'], 'Belongings are covered up to £200.', True, False, .5),
        ('Is business delivery use covered?', 'Business delivery is excluded.', ['Business delivery excluded'], 'Business delivery use is excluded.', True, False, 1),
    ]
    result = []
    for i, (q, ref, facts, ans, able, abst, expected) in enumerate(rows):
        task = dict(zip(TASK_FIELDS, [q, able, ref, facts, '', [{'text': policy}],
                                    [], False, ans, [], abst, True, True]))
        task['evaluation_id'] = f'fictional-{i}'
        result.append((task, expected))
    return result
