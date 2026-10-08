import pytest
from experiments.reload_check import require_comparison, mismatches

GATES = ['logits_within_original_tolerance', 'top1_equal',
         'generated_tokens_equal', 'generated_text_equal', 'generation_nonempty', 'runtime_equal']

@pytest.mark.parametrize('failed', GATES)
def test_reload_requires_every_numerical_and_functional_gate(failed):
    report = dict.fromkeys(GATES, True)
    report[failed] = False
    with pytest.raises(ValueError, match='reload gate failed'):
        require_comparison(report)


def test_incomplete_diagnostics_do_not_pass():
    with pytest.raises(ValueError):
        require_comparison({})


def test_complete_diagnostics_pass():
    require_comparison(dict.fromkeys(GATES, True))


def test_dtype_comparison_includes_missing_and_extra_parameters():
    assert mismatches({'norm': 'float32', 'old': 'float16'},
                      {'norm': 'float16', 'new': 'float32'}) == ['new', 'norm', 'old']
