"""Gate half A for T-075 — eval/judge_votes.py.

ZERO data-dependent tests, deliberately. CI checks out without the `policy_qa` submodule and
pytest exits 0 when every test skips, so a data-dependent half would be a false green
(eval/reranker_ab.py:620 rejects that pattern in writing). All data dependence lives in
`judge_votes.py --adopt-t062`, which never skips.

`check_merge` is fed VIOLATING inputs throughout — a gate nobody has watched fail is not a
gate, and this repo has caught five gate-proxy defects on that principle.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MOD_PATH = Path(__file__).resolve().parents[1] / "eval" / "judge_votes.py"


def _load(name="_jv_under_test"):
    spec = importlib.util.spec_from_file_location(name, MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


jv = _load()


def _v(scores, contra=False, vote=1, spans=None):
    r = {"task": "t000", "fact_scores": scores, "contradiction": contra, "vote": vote}
    if spans is not None:
        r["evidence_spans"] = spans
    return r


# ------------------------------------------------------------------------- import surface

def test_module_imports_without_the_data_repo():
    """The pure core must not need `common`, `rag` or the submodule at import time. If this
    fails, the pytest half has silently become data-dependent and will start SKIPPING in CI
    rather than failing — the exact false green this file exists to prevent."""
    poisoned = {}
    for n in ("common", "rag", "judge", "chromadb", "openai", "torch"):
        poisoned[n] = sys.modules.get(n)
        sys.modules[n] = None
    try:
        m = _load("_jv_isolated")
        assert callable(m.merge_task)
    finally:
        for n, old in poisoned.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old


# ----------------------------------------------------------------------- parse_vote_path

@pytest.mark.parametrize("name,want", [
    ("shard_00_v1.jsonl", ("shard_00", 1)),
    ("shard_19_v3.jsonl", ("shard_19", 3)),
    ("shard_00.jsonl", None),
    ("votes.jsonl", None),
    ("shard_00_v1.json", None),
])
def test_parse_vote_path(name, want):
    assert jv.parse_vote_path(name) == want


def test_filename_is_the_authority_for_the_vote_index():
    """A judge agent can write any `vote` value into its JSON; it cannot choose which file
    the harness appends to. The parse must come from the name."""
    assert jv.parse_vote_path("shard_07_v2.jsonl") == ("shard_07", 2)


# ---------------------------------------------------------------------------- vote_valid

def test_vote_valid_accepts_a_well_formed_vote():
    assert jv.vote_valid(_v([0, 1, 2]), 3)[0] is True


@pytest.mark.parametrize("row,n,reason", [
    (_v([0, 1]), 3, "arity"),
    (_v([0, 1, 2, 2]), 3, "arity"),
    (_v([0, 3, 2]), 3, "range"),
    (_v([0, -1, 2]), 3, "range"),
    (_v(["2", 1, 0]), 3, "range"),
    (_v([0, 1, 2], contra="yes"), 3, "contradiction_type"),
    (_v([0, 1, 2], spans=["a", "b"]), 3, "spans_arity"),
])
def test_vote_valid_rejects_malformed_votes(row, n, reason):
    ok, why = jv.vote_valid(row, n)
    assert ok is False and why == reason


def test_bool_is_not_accepted_as_a_fact_score():
    """bool subclasses int in Python, so `True in (0,1,2)` is True via 1 == True. A JSON
    `true` must not silently score as a 1."""
    assert jv.vote_valid(_v([True, 1, 2]), 3)[0] is False


def test_missing_fact_scores_is_invalid_not_a_crash():
    assert jv.vote_valid({"task": "t000", "contradiction": False}, 3)[0] is False


# ---------------------------------------------------------------------------- merge_task

def test_median_low_returns_a_score_a_judge_actually_cast():
    """statistics.median([0,2]) is 1.0 — a FLOAT, and a score nobody gave. On an even number
    of valid votes the merged value must still be an int that was actually cast."""
    m = jv.merge_task("t000", [_v([0], vote=1), _v([2], vote=2)], 1)
    assert m["fact_scores"] == [0]
    assert isinstance(m["fact_scores"][0], int)
    assert not isinstance(m["fact_scores"][0], bool)


def test_three_votes_give_the_ordinary_median():
    m = jv.merge_task("t000", [_v([2], vote=1), _v([0], vote=2), _v([2], vote=3)], 1)
    assert m["fact_scores"] == [2] and m["n_votes"] == 3


def test_contradiction_majority():
    votes = [_v([2], True, 1), _v([2], False, 2), _v([2], True, 3)]
    assert jv.merge_task("t000", votes, 1)["contradiction"] is True
    votes = [_v([2], False, 1), _v([2], False, 2), _v([2], True, 3)]
    assert jv.merge_task("t000", votes, 1)["contradiction"] is False


def test_contradiction_tie_goes_to_true_and_is_recorded():
    """The rule the hand-merge actually applied at t175 but never wrote down. Codifying it
    is what takes the T-062 replay from 349/350 to 350/350."""
    m = jv.merge_task("t000", [_v([0], True, 1), _v([0], False, 2)], 1)
    assert m["contradiction"] is True
    assert m["tie_break"] == "contradiction_tie_to_true"


def test_both_tie_breaks_push_the_score_down():
    """One principle: when the judges cannot agree, the model does not get the benefit of the
    doubt. median_low takes the lower score; the contradiction tie takes the harsher verdict."""
    assert jv.merge_task("t000", [_v([0], vote=1), _v([2], vote=2)], 1)["fact_scores"] == [0]
    assert jv.merge_task("t000", [_v([2], True, 1), _v([2], False, 2)], 1)["contradiction"]


def test_n_votes_is_the_valid_count_not_a_constant():
    """One vote miscounts arity -> it is rejected, n_votes is 2, and n_votes_cast is 3 so
    "3 cast / 2 used" is legible rather than looking like a short run."""
    votes = [_v([2, 2], vote=1), _v([2], vote=2), _v([0, 2], vote=3)]
    m = jv.merge_task("t000", votes, 2)
    assert m["n_votes"] == 2 and m["n_votes_cast"] == 3 and m["rejected"] == ["arity"]


def test_one_valid_vote_is_a_hard_failure_not_a_degradation():
    """A single vote is not an x3 median and must never enter the board."""
    with pytest.raises(ValueError):
        jv.merge_task("t000", [_v([2], vote=1), _v([2, 2], vote=2)], 1)


def test_evidence_span_is_carried_by_a_vote_that_gave_the_merged_score():
    """The published span must corroborate the published score, not contradict it."""
    votes = [_v([0], vote=1, spans=["low"]), _v([2], vote=2, spans=["high"]),
             _v([0], vote=3, spans=["also-low"])]
    assert jv.merge_task("t000", votes, 1)["evidence_spans"] == ["low"]


def test_missing_evidence_spans_do_not_crash_the_merge():
    m = jv.merge_task("t000", [_v([2], vote=1), _v([2], vote=2)], 1)
    assert m["evidence_spans"] == [""]


def test_disagreement_is_counted_per_fact():
    votes = [_v([2, 2], vote=1), _v([0, 2], vote=2), _v([2, 2], vote=3)]
    assert jv.merge_task("t000", votes, 2)["n_facts_disagreed"] == 1


# ---------------------------------------------------------------------------- todo_units

def test_todo_units_is_a_set_difference_over_what_is_on_disk():
    todo = jv.todo_units(["a", "b"], 3, {("a", 1), ("a", 2)})
    assert todo == {("a", 3), ("b", 1), ("b", 2), ("b", 3)}


def test_todo_units_is_empty_when_complete():
    done = {(c, v) for c in ("a", "b") for v in (1, 2, 3)}
    assert jv.todo_units(["a", "b"], 3, done) == set()


def test_a_half_complete_shard_is_not_a_special_case():
    """Only the missing (code, vote) pairs return to the todo set — a shard is never
    re-judged wholesale, which would waste tokens AND change the vote population."""
    assert jv.todo_units(["a", "b", "c"], 1, {("a", 1), ("c", 1)}) == {("b", 1)}


# --------------------------------------------------------------------- fact_disagreement

def test_fact_disagreement_reports_the_fact_denominator_not_the_task_count():
    """eval/leaderboard.md:7 reports '0/350 facts' where 350 is the TASK count and the fact
    count is 1,472. Both numbers must be emitted with their own denominators."""
    merged = [{"fact_scores": [0, 1, 2], "n_facts_disagreed": 1},
              {"fact_scores": [2, 2], "n_facts_disagreed": 0}]
    fd = jv.fact_disagreement(merged)
    assert fd["n_facts"] == 5 and fd["n_tasks"] == 2 and fd["any_split"] == 1


# --------------------------------------------------------- check_merge: MUST BE ABLE TO FAIL

def _merged(codes, n_votes=3):
    return [{"task": c, "fact_scores": [2], "n_votes": n_votes, "n_facts_disagreed": 0}
            for c in codes]


def _units(codes, n_votes=3):
    return {(c, v) for c in codes for v in range(1, n_votes + 1)}


def test_check_merge_passes_on_a_complete_run():
    codes = ["a", "b"]
    assert jv.check_merge(codes, _merged(codes), _units(codes), 3, 5) == []


def test_check_merge_catches_a_SHORT_RUN():
    """THE test. judge.py is silent here — _row_metrics takes the mean of whatever it is
    handed and prints a perfectly plausible number, so a run that finished 300 of 350 would
    publish a smaller-denominator score with nothing going red."""
    codes = ["a", "b", "c"]
    errs = jv.check_merge(codes, _merged(["a", "b"]), _units(["a", "b"]), 3, 5)
    assert any("missing from the merge" in e for e in errs)


def test_check_merge_catches_an_empty_population():
    """A validator that finds nothing to check has failed, not passed (the T-011 disease)."""
    assert jv.check_merge([], [], set(), 3, 5) == ["population is EMPTY — a merge over zero "
                                                  "tasks is not a result"]


def test_check_merge_catches_extra_tasks():
    codes = ["a"]
    errs = jv.check_merge(codes, _merged(["a", "z"]), _units(["a"]), 3, 5)
    assert any("not in the population" in e for e in errs)


def test_check_merge_catches_duplicate_rows():
    errs = jv.check_merge(["a"], _merged(["a"]) + _merged(["a"]), _units(["a"]), 3, 5)
    assert any("duplicate rows" in e for e in errs)


def test_check_merge_catches_a_missing_vote_pass():
    codes = ["a", "b"]
    errs = jv.check_merge(codes, _merged(codes), _units(codes, 2), 3, 5)
    assert any("votes on disk" in e for e in errs)


def test_check_merge_catches_votes_for_unknown_codes():
    codes = ["a"]
    errs = jv.check_merge(codes, _merged(codes), _units(codes) | {("ghost", 1)}, 3, 5)
    assert any("unknown codes" in e for e in errs)


def test_check_merge_catches_mass_silent_degradation():
    codes = ["a", "b", "c"]
    errs = jv.check_merge(codes, _merged(codes, n_votes=2), _units(codes), 3, max_degraded=1)
    assert any("degraded" in e for e in errs)


def test_check_merge_tolerates_degradation_within_budget():
    codes = ["a", "b", "c"]
    merged = _merged(["a", "b"]) + _merged(["c"], n_votes=2)
    assert jv.check_merge(codes, merged, _units(codes), 3, max_degraded=1) == []


# ------------------------------------------------------------------------------ read_rows

def test_read_rows_survives_a_torn_last_line(tmp_path):
    """A session killed mid-append leaves a partial line. The prefix must still load, so the
    torn judgement is simply absent from `done` and gets re-dispatched."""
    p = tmp_path / "shard_00_v1.jsonl"
    p.write_text('{"task":"a","fact_scores":[2]}\n{"task":"b","fact_sc')
    rows = jv.read_rows(p)
    assert len(rows) == 1 and rows[0]["task"] == "a"


def test_read_rows_on_a_missing_file_is_empty_not_an_error(tmp_path):
    assert jv.read_rows(tmp_path / "nope.jsonl") == []


def test_load_votes_first_write_wins_and_is_order_independent(tmp_path):
    """Deterministic under re-dispatch. 'Last wins' is not."""
    (tmp_path / "shard_00_v1.jsonl").write_text(
        '{"task":"a","fact_scores":[2],"contradiction":false}\n'
        '{"task":"a","fact_scores":[0],"contradiction":false}\n')
    got = jv.load_votes(tmp_path)
    assert got[("a", 1)]["fact_scores"] == [2]


def test_load_votes_ignores_files_that_are_not_vote_files(tmp_path):
    (tmp_path / "notes.jsonl").write_text('{"task":"x","fact_scores":[2]}\n')
    assert jv.load_votes(tmp_path) == {}
