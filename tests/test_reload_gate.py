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


def test_training_autocast_is_removed_before_reload_comparison():
    from experiments.reload_check import unwrap_for_reload
    class Model:
        def __init__(self): self._original_forward = object()
        def named_modules(self): return [('', self)]
    class Accelerator:
        def unwrap_model(self, model, keep_fp32_wrapper=True):
            if not keep_fp32_wrapper: del model._original_forward
            return model
    model = Model()
    assert unwrap_for_reload(model, Accelerator()) is model
    assert '_original_forward' not in model.__dict__


def test_remaining_training_autocast_fails_closed():
    from experiments.reload_check import unwrap_for_reload
    class Model:
        _original_forward = None
        def __init__(self): self._original_forward = object()
        def named_modules(self): return [('', self)]
    class Accelerator:
        def unwrap_model(self, model, **kw): return model
    with pytest.raises(ValueError, match='wrapper remains'):
        unwrap_for_reload(Model(), Accelerator())
