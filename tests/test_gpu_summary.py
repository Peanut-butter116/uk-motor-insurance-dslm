from scripts.summarise_gpu_results import deterministic_metrics


def test_pending_judging_never_emits_correctness_or_composite():
    records = [dict(answerable=True, corr_item=None, n_sup=1, n_tot=2,
                    abst=False, judged=False),
               dict(answerable=False, corr_item=None, n_sup=0, n_tot=0,
                    abst=True, judged=False)]
    result = deterministic_metrics(records)
    assert result['citation'] == 0.5
    assert result['abstention'] == 1.0
    assert 'correctness' not in result and 'composite' not in result
