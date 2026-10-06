"""Tests for eval/abstention.py — the assertion-based abstention rule (T-089).

ZERO DATA-DEPENDENT TESTS, and that is a hard requirement rather than a style choice. CI checks
out without the `policy_qa` submodule, and pytest exits 0 when every test skips, so a
data-dependent half would be a false green (eval/reranker_ab.py:620 rejects exactly this in
writing). All data dependence lives in `eval/abstention.py --verify`, which never skips.

Modules are loaded by EXPLICIT PATH. policy_qa/eval/ held stale shadows of five eval modules
that won on sys.path until T-071 deleted them; explicit-path loading is immune to their return.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab = _load("abstention_under_test", "eval/abstention.py")
aa = _load("abstention_audit_under_test", "eval/abstention_audit.py")
common = _load("common_under_test", "src/common.py")

AS = common.ABSTAIN_SENTENCE
KEY = AS.rstrip(" .")
CIT = common.CITATION_RE
TAG = "[Aviva, Motor Wording, Section 3, p.14]"
TAG2 = "[Admiral, Car Policy Booklet, Section 7, p.2]"


def loose(t):
    return ab.is_abstention_loose(t, abstain_sentence=AS)


def asserts(t, reading=ab.ADOPTED_READING):
    return ab.asserts(t, abstain_sentence=AS, citation_re=CIT, reading=reading)


def is_abst(t, reading=ab.ADOPTED_READING):
    return ab.is_abstention_assertion(t, abstain_sentence=AS, citation_re=CIT, reading=reading)


# ------------------------------------------------------------------------ import surface

def test_module_imports_without_the_data_repo():
    """The pure core may not need `common`, `rag`, or any heavy dependency. Poison them all and
    re-load: if this fails, some import crept above the IO boundary and the pytest half of the
    gate has quietly become skippable."""
    poisoned = {}
    for name in ("common", "rag", "chromadb", "openai", "torch", "sentence_transformers"):
        poisoned[name] = sys.modules.get(name)
        sys.modules[name] = None
    try:
        mod = _load("abstention_reimport_probe", "eval/abstention.py")
        assert callable(mod.is_abstention_assertion)
        assert callable(mod.reclassify)
        assert mod.ADOPTED_READING in mod.READING_RULES
    finally:
        for name, old in poisoned.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_loose_rule_is_DEFINED_IN_the_sibling_not_respelled_here():
    """judge.py:80's rule already exists twice (judge.py and abstention_audit.py). A THIRD
    spelling is how a rule drifts -- eval/error_decomposition.py:255 records that a
    reimplementation using the bare sentence is a strictly weaker prefix match, identical on
    today's data and silently divergent on a truncated refusal.

    Identity (`is`) is the WRONG assertion here and asserting it was a mistake: both modules are
    loaded by explicit path, so this test's `aa` and the copy abstention.py loads for itself are
    two distinct module objects with two distinct function objects. What actually matters is
    which FILE the code came from -- that is what "no third copy" means.
    """
    assert ab.is_abstention_loose.__code__.co_filename.endswith("abstention_audit.py")


def test_the_axis_math_is_also_reused_not_restated():
    for fn in ("abstention_axis", "correctness_bounds", "composite_bounds", "gap_bounds"):
        got = getattr(ab, fn).__code__.co_filename
        assert got.endswith("abstention_audit.py"), (fn, got)


def test_the_reused_loose_rule_behaves_identically_to_the_sibling_s():
    """Belt and braces on the file check: same verdict on every adversarial string.

    Scope, stated honestly because the first draft of this docstring over-claimed: this catches
    DIVERGENCE, not duplication. A verbatim copy of the function body into this module passes
    here and is caught only by the `co_filename` test above. Both are kept -- one for "is it the
    same code", one for "does it still behave the same".
    """
    for text in (BARE, BARE_QUOTED, BARE_NO_STOP, CITED_ANSWER_THEN_SENTENCE,
                 UNCITED_ASSERTION_THEN_SENTENCE, SENTENCE_THEN_CITATION_ONLY,
                 DIGITS_ONLY_RESIDUE, UNRELATED, EMPTY):
        assert ab.is_abstention_loose(text, abstain_sentence=AS) == \
            aa.is_abstention_loose(text, abstain_sentence=AS), text[:40]


# ------------------------------------------------------------------- the readings, on strings

BARE = AS
BARE_QUOTED = f'"{AS}"'
BARE_NO_STOP = KEY
CITED_ANSWER_THEN_SENTENCE = f"Windscreen claims carry a GBP 75 excess {TAG}. {AS}"
UNCITED_ASSERTION_THEN_SENTENCE = f"Windscreen claims carry a GBP 75 excess. {AS}"
SENTENCE_THEN_CITATION_ONLY = f"{AS} {TAG}"
SENTENCE_THEN_TWO_CITATIONS = f"{AS} {TAG} {TAG2}"
DIGITS_ONLY_RESIDUE = f"{AS} 500."
UNRELATED = f"Your excess is GBP 250 {TAG}."
EMPTY = ""


@pytest.mark.parametrize("text,want", [
    (BARE, True), (BARE_QUOTED, True), (BARE_NO_STOP, True),
    (CITED_ANSWER_THEN_SENTENCE, True), (UNCITED_ASSERTION_THEN_SENTENCE, True),
    (SENTENCE_THEN_CITATION_ONLY, True), (DIGITS_ONLY_RESIDUE, True),
    (UNRELATED, False), (EMPTY, False),
])
def test_loose_flags_exactly_what_it_should(text, want):
    assert loose(text) is want


def test_a_bare_refusal_is_an_abstention_under_every_reading():
    for reading in ab.READING_RULES:
        for text in (BARE, BARE_QUOTED, BARE_NO_STOP):
            assert is_abst(text, reading) is True, (reading, text[:40])


def test_a_cited_answer_carrying_the_sentence_is_never_an_abstention():
    """The T-084 defect itself: this is a real answer with a real citation that happens to end
    with the refusal sentence. Every reading must recover it."""
    for reading in ab.READING_RULES:
        assert is_abst(CITED_ANSWER_THEN_SENTENCE, reading) is False, reading


def test_an_UNCITED_assertion_is_the_case_that_separates_A_from_B():
    """THE DISCRIMINATING CASE, and the reason reading B was adopted.

    An uncited claim is strictly WORSE than a cited one -- it violates rule 2 of
    src/rag.py's SYSTEM_TEMPLATE. Reading A requires a citation before it will call anything an
    assertion, so it hands abstention credit to exactly the ungrounded answers. B does not.

    tests/test_abstention_detector.py pins this shape as a KNOWN FALSE POSITIVE of the length
    rule. Here it is asserted FIXED, which is what makes the replacement worth shipping.
    """
    assert asserts(UNCITED_ASSERTION_THEN_SENTENCE, "A") is False   # A cannot see it
    assert asserts(UNCITED_ASSERTION_THEN_SENTENCE, "B") is True    # B does
    assert asserts(UNCITED_ASSERTION_THEN_SENTENCE, "C") is True

    assert is_abst(UNCITED_ASSERTION_THEN_SENTENCE, "A") is True    # ...and mis-scores it
    assert is_abst(UNCITED_ASSERTION_THEN_SENTENCE, "B") is False   # FIXED under the adopted rule
    assert is_abst(UNCITED_ASSERTION_THEN_SENTENCE, "C") is False


def test_a_citation_is_a_pointer_not_a_proposition():
    """The rationale for the adopted rule, made executable. A refusal that also carries a
    citation asserts nothing -- the bracket names a clause, it does not claim anything about it.
    Readings A and B agree; C is the one that counts the bracket itself as an assertion."""
    for text in (SENTENCE_THEN_CITATION_ONLY, SENTENCE_THEN_TWO_CITATIONS):
        assert is_abst(text, "A") is True
        assert is_abst(text, "B") is True
        assert is_abst(text, "C") is False


def test_a_non_alphabetic_residue_is_not_an_assertion_limit_c():
    """Documented limit (c): "prose" is [A-Za-z], so digits and punctuation alone do not count.
    Asserted rather than left implicit, so it stays a KNOWN limit rather than a surprise. The
    alternative -- a word-count threshold -- is the fitted constant this rule exists to remove.
    """
    assert is_abst(DIGITS_ONLY_RESIDUE, "B") is True


def test_text_without_the_sentence_is_never_an_abstention():
    for reading in ab.READING_RULES:
        assert is_abst(UNRELATED, reading) is False
        assert is_abst(EMPTY, reading) is False


def test_the_adopted_rule_is_a_strict_subset_of_loose():
    """Not a detail: it means this correction can only move records OUT of the over-refusal
    bucket, never in. Anything else is a new abstention rule, not a correction of the old one."""
    corpus = [BARE, BARE_QUOTED, BARE_NO_STOP, CITED_ANSWER_THEN_SENTENCE,
              UNCITED_ASSERTION_THEN_SENTENCE, SENTENCE_THEN_CITATION_ONLY,
              SENTENCE_THEN_TWO_CITATIONS, DIGITS_ONLY_RESIDUE, UNRELATED, EMPTY]
    for reading in ab.READING_RULES:
        for text in corpus:
            if is_abst(text, reading):
                assert loose(text), (reading, text[:40])


# ------------------------------------------------------------------------ primitive behaviour

def test_residual_removes_both_spellings_of_the_sentence():
    """The 84-char key is a PREFIX of the 85-char sentence, so the full string must be removed
    first. Removing the key first would strand a trailing "." and, on a variant that appends
    prose, would leave punctuation that no [A-Za-z] test cares about -- but the ordering is
    pinned anyway because a future prose test might."""
    assert AS not in ab.residual(BARE, abstain_sentence=AS, citation_re=CIT)
    assert KEY not in ab.residual(BARE_NO_STOP, abstain_sentence=AS, citation_re=CIT)
    r = ab.residual(f"{AS} {AS}", abstain_sentence=AS, citation_re=CIT)
    assert not ab.ALPHA_RE.search(r), r


def test_residual_removes_every_citation_not_just_the_first():
    r = ab.residual(SENTENCE_THEN_TWO_CITATIONS, abstain_sentence=AS, citation_re=CIT)
    assert "Aviva" not in r and "Admiral" not in r, r


def test_citation_re_has_no_default_anywhere():
    """Injection is the guarantee that the DATA path binds the canonical common.CITATION_RE --
    the same object rag.verify_citations uses -- rather than eval/coverage_proxy.py's
    byte-for-byte copy. A default would quietly reintroduce the second pattern."""
    with pytest.raises(TypeError):
        ab.strip_citations("x")
    with pytest.raises(TypeError):
        ab.residual("x", abstain_sentence=AS)
    with pytest.raises(TypeError):
        ab.asserts("x", abstain_sentence=AS)


def test_an_unknown_reading_raises_rather_than_defaulting():
    """Silently falling back to the adopted reading would make a typo in a sensitivity sweep
    look like agreement between readings."""
    with pytest.raises(ValueError):
        asserts(BARE, "D")
    with pytest.raises(ValueError):
        asserts(BARE, "b")


def test_the_adopted_reading_is_B_and_is_declared():
    assert ab.ADOPTED_READING == "B"
    assert set(ab.READING_RULES) == {"A", "B", "C"}


# ------------------------------------------------------------------------------- reclassify

def _recs():
    return [
        {"id": "PQ-0001", "text": BARE, "answerable": True},
        {"id": "PQ-0002", "text": CITED_ANSWER_THEN_SENTENCE, "answerable": True},
        {"id": "PQ-0003", "text": UNCITED_ASSERTION_THEN_SENTENCE, "answerable": True},
        {"id": "PQ-0004", "text": UNRELATED, "answerable": True},
        {"id": "PQ-0005", "text": BARE, "answerable": False},
        {"id": "PQ-0006", "text": CITED_ANSWER_THEN_SENTENCE, "answerable": False},
        {"id": "PQ-0007", "text": UNRELATED, "answerable": False},
    ]


def test_reclassify_buckets_exactly():
    c = ab.reclassify(_recs(), abstain_sentence=AS, citation_re=CIT)
    assert c["ans_n"] == 4 and c["refuse_n"] == 3
    assert c["n_loose_over"] == 3 and c["n_adopted_over"] == 1
    assert c["recovered_ids"] == ["PQ-0002", "PQ-0003"]
    assert c["adopted_over_ids"] == ["PQ-0001"]
    assert c["n_loose_refuse_ok"] == 2 and c["n_adopted_refuse_ok"] == 1
    assert c["lost_refusal_ids"] == ["PQ-0006"]


def test_reclassify_partitions_both_sides_exactly():
    """loose == adopted + recovered on the answerable side, and loose == adopted + lost on the
    unanswerable side. A record that fell out of both buckets would silently shrink a
    denominator, which is the shape of defect check_merge exists to catch elsewhere."""
    for reading in ab.READING_RULES:
        c = ab.reclassify(_recs(), abstain_sentence=AS, citation_re=CIT, reading=reading)
        assert c["n_loose_over"] == c["n_adopted_over"] + c["n_recovered"], reading
        assert c["n_loose_refuse_ok"] == c["n_adopted_refuse_ok"] + c["n_lost_refusal"], reading


def test_reclassify_reports_the_UNFAVOURABLE_half_too():
    """An audit that emitted recovered_ids and not lost_refusal_ids would be advocacy. On this
    corpus the adopted rule genuinely costs the row a correct refusal, and that must show."""
    c = ab.reclassify(_recs(), abstain_sentence=AS, citation_re=CIT)
    assert c["n_lost_refusal"] > 0
    assert c["lost_refusal_ids"]


def test_reclassify_leaks_no_answer_text():
    """FIREWALL. reclassify takes text and returns IDS. If any answer string reached the report
    the whole artifact would become undistributable."""
    blob = repr(ab.reclassify(_recs(), abstain_sentence=AS, citation_re=CIT))
    for text in (BARE, CITED_ANSWER_THEN_SENTENCE, UNCITED_ASSERTION_THEN_SENTENCE, UNRELATED):
        assert text not in blob
    assert "Windscreen" not in blob and "Aviva" not in blob


def test_reading_sensitivity_carries_every_reading():
    s = ab.reading_sensitivity(_recs(), abstain_sentence=AS, citation_re=CIT)
    assert set(s) == set(ab.READING_RULES)
    # and the readings are genuinely different on this corpus, so the table is not decorative
    assert s["A"] != s["B"] or s["B"] != s["C"]


# ----------------------------------------------------------------- the point-estimate guard

def test_a_bound_may_not_carry_a_point_estimate_while_anything_is_unjudged():
    block = {"corr_lower": 0.3, "corr_upper": 0.5, "n_recovered_unjudged": 19,
             "composite": 0.61}
    assert ab.point_estimate_leaks(block) == ["composite"]


def test_a_bound_MAY_carry_a_point_estimate_once_nothing_is_unjudged():
    """The exception that makes T-091 meaningful, and it is keyed on a COUNT rather than a flag
    anyone can set. Judge every recovered record and the interval has collapsed to a
    measurement; publishing it is then honest, not an over-claim."""
    block = {"corr_lower": 0.42, "corr_upper": 0.42, "n_recovered_unjudged": 0,
             "composite": 0.61}
    assert ab.point_estimate_leaks(block) == []


@pytest.mark.parametrize("value,unlocks", [
    (0, True),          # the integer zero, and ONLY the integer zero
    (False, False),     # `False == 0` in Python -- a literal flag unlocking a point estimate
    (0.0, False),       # so does 0.0
    ("0", False),
    (None, False),
    (19, False),
])
def test_only_a_true_integer_zero_unlocks_a_point_estimate(value, unlocks):
    """MUST-FAIL guard on a real hole found by adversarial review before this landed.

    The guard was written as `if unjudged != 0`. Python's `False == 0` and `0.0 == 0`, so the
    boolean `False` — exactly the "flag anyone can set" the docstring promised was impossible —
    silently unlocked a published point estimate. `is_zero_count` type-checks instead.
    """
    block = {"corr_lower": 0.3, "corr_upper": 0.5, "n_recovered_unjudged": value,
             "composite": 0.61}
    assert ab.is_zero_count(value) is unlocks
    assert (ab.point_estimate_leaks(block) == []) is unlocks


def test_the_exception_does_not_fire_on_a_missing_count():
    """Absent is not zero. A block that forgot to report n_recovered_unjudged must be treated as
    unjudged, never as measured -- otherwise dropping a key becomes a way to unlock a point
    estimate."""
    block = {"corr_lower": 0.3, "corr_upper": 0.5, "composite": 0.61}
    assert ab.point_estimate_leaks(block) == ["composite"]


def test_the_guard_walks_nested_structures():
    report = {"rows": {"r1": {"adopted": {"corr_lower": 0.1, "corr_upper": 0.9,
                                          "n_recovered_unjudged": 4, "correctness": 0.5}}}}
    assert ab.point_estimate_leaks(report) == ["rows.r1.adopted.correctness"]


def test_booleans_are_not_point_estimates():
    block = {"corr_lower": 0.3, "corr_upper": 0.5, "n_recovered_unjudged": 2,
             "composite": True}
    assert ab.point_estimate_leaks(block) == []


# --------------------------------------------------------------------------- check_report

def _row(loose_over=36, adopted_over=17, recovered=19, loose_ok=16, adopted_ok=13, lost=3,
         rec_scored=0, rec_ids=("PQ-0001", "PQ-0002"), lost_ids=("PQ-0038",),
         abst=0.7234, corr_lo=0.3218, corr_hi=0.4828, comp_lo=0.5750, comp_hi=0.6555,
         imputed=0.3964):
    return {
        "ans_n": 118, "refuse_n": 22,
        "n_loose_over": loose_over, "n_adopted_over": adopted_over, "n_recovered": recovered,
        "n_loose_refuse_ok": loose_ok, "n_adopted_refuse_ok": adopted_ok,
        "n_lost_refusal": lost, "n_recovered_with_judge_score": rec_scored,
        "recovered_ids": list(rec_ids), "lost_refusal_ids": list(lost_ids),
        "loose": {"abstention": 0.7111, "corr_lower": 0.3218, "corr_upper": 0.3218,
                  "imputed_at_judged_mean": 0.3218, "n_recovered_unjudged": 0,
                  "composite_lower": 0.5725, "composite_upper": 0.5725},
        "adopted": {"abstention": abst, "corr_lower": corr_lo, "corr_upper": corr_hi,
                    "imputed_at_judged_mean": imputed, "n_recovered_unjudged": recovered,
                    "composite_lower": comp_lo, "composite_upper": comp_hi},
    }


def _sens(a_rec=19, b_rec=19, c_rec=20):
    def cell(rec):
        return {"adopted_over": 36 - rec, "recovered": rec, "adopted_refuse_ok": 13,
                "lost_refusal": 3, "loose_over": 36, "loose_refuse_ok": 16}
    return {"qwen2.5-7b-bnb-openbook": {"A": cell(a_rec), "B": cell(b_rec), "C": cell(c_rec)}}


def _gap(lo=-0.056350, hi=0.024158, sign="indeterminate"):
    return {"tuned": "qwen2.5-7b-raft-openbook", "base": "qwen2.5-7b-bnb-openbook",
            "loose": {"gap_lower": 0.026623, "gap_upper": 0.026623, "sign": "positive",
                      "sign_determined": True},
            "adopted": {"gap_lower": lo, "gap_upper": hi, "sign": sign,
                        "sign_determined": sign != "indeterminate"}}


def _report(sens=None, gap=None, **kw):
    return {"schema": "abstention/1", "adopted_reading": "B", "abstain_sentence_len": 85,
            "rows": {"qwen2.5-7b-bnb-openbook": _row(**kw)},
            "reading_sensitivity": sens if sens is not None else _sens(),
            "gap_tuned_minus_base": gap if gap is not None else _gap()}


def test_check_passes_on_an_identical_report():
    assert ab.check_report(_report(), _report()) == []


@pytest.mark.parametrize("kw,needle", [
    ({"loose_over": 35}, "n_loose_over"),
    ({"adopted_over": 18}, "n_adopted_over"),
    ({"recovered": 20}, "n_recovered"),
    ({"loose_ok": 15}, "n_loose_refuse_ok"),
    ({"adopted_ok": 14}, "n_adopted_refuse_ok"),
    ({"lost": 4}, "n_lost_refusal"),
    ({"rec_scored": 19}, "n_recovered_with_judge_score"),
    ({"rec_ids": ("PQ-0001", "PQ-0009")}, "recovered_ids"),
    ({"lost_ids": ("PQ-0099",)}, "lost_refusal_ids"),
    ({"abst": 0.70}, "adopted.abstention"),
    ({"corr_lo": 0.33}, "adopted.corr_lower"),
    ({"corr_hi": 0.51}, "adopted.corr_upper"),
    ({"imputed": 0.45}, "adopted.imputed_at_judged_mean"),
    ({"comp_lo": 0.58}, "adopted.composite_lower"),
    ({"comp_hi": 0.67}, "adopted.composite_upper"),
])
def test_check_fails_on_every_kind_of_drift(kw, needle):
    errs = ab.check_report(_report(), _report(**kw))
    assert errs, f"drift {kw} was NOT caught"
    assert any(needle in e for e in errs), f"{needle!r} not in {errs}"


@pytest.mark.parametrize("key,value", [
    ("schema", "abstention/2"),
    ("adopted_reading", "C"),
    ("abstain_sentence_len", 84),
])
def test_check_fails_when_the_rule_identity_moves(key, value):
    """Changing the adopted reading is a metric change and must never be silent -- it is exactly
    the move a hostile reader accuses us of ("you changed the rule after seeing the numbers")."""
    live = _report()
    live[key] = value
    errs = ab.check_report(_report(), live)
    assert any(key in e for e in errs), errs


def test_check_fails_when_a_rejected_reading_is_deleted():
    """The reading table is GATED, not merely published. Ungated, deleting readings A and C --
    or collapsing them into B -- would move no number at all. That is the same hole T-084
    measured in its own sensitivity sweep, closed the same way."""
    live = _report()
    del live["reading_sensitivity"]["qwen2.5-7b-bnb-openbook"]["C"]
    errs = ab.check_report(_report(), live)
    assert any("reading_sensitivity" in e and "missing" in e for e in errs), errs


def test_the_reading_guard_is_not_self_referential(monkeypatch):
    """MUST-FAIL guard on a real hole found by adversarial review before this landed.

    The check was `set(READING_RULES) - set(live[row])` — keyed to a constant IN THE MODULE
    UNDER TEST. Deleting a rejected reading from READING_RULES therefore deleted it from the
    expectation too, and `--verify` stayed GREEN: the classic "relax the constant and the guard
    relaxes with it" mutation that TASKS.md T-088 boards as measured mutation M4.

    The committed report is the external witness READING_RULES cannot edit, so the fix is a
    committed-vs-live set comparison — which is what abstention_audit.check_audit already does
    for its own sweep.
    """
    monkeypatch.setattr(ab, "READING_RULES", {"B": ab.READING_RULES["B"]})
    live = _report()
    del live["reading_sensitivity"]["qwen2.5-7b-bnb-openbook"]["A"]
    del live["reading_sensitivity"]["qwen2.5-7b-bnb-openbook"]["C"]
    errs = ab.check_report(_report(), live)
    assert any("reading set moved" in e for e in errs), errs


def test_check_gates_the_unlock_count_itself():
    """`n_recovered_unjudged` is the key that unlocks a point estimate, so leaving it ungated
    made the smuggle end-to-end: set it to 0 and both the guard AND the gate went quiet."""
    live = _report()
    live["rows"]["qwen2.5-7b-bnb-openbook"]["adopted"]["n_recovered_unjudged"] = 0
    errs = ab.check_report(_report(), live)
    assert any("n_recovered_unjudged" in e for e in errs), errs


def test_check_cross_checks_the_unlock_count_against_the_row_totals():
    """Gating the number pins it to the committed report; this pins it to reality. Without it,
    a report could be regenerated with a doctored count and stay internally consistent."""
    committed = _report(rec_scored=19)
    live = _report(rec_scored=19)
    # 19 recovered, 19 judged => unjudged must be 0; claim 5 and it must be caught
    for r in (committed, live):
        r["rows"]["qwen2.5-7b-bnb-openbook"]["adopted"]["n_recovered_unjudged"] = 5
    errs = ab.check_report(committed, live)
    assert any("but n_recovered - n_recovered_with_judge_score" in e for e in errs), errs


def test_check_fails_when_a_reading_silently_changes_its_split():
    errs = ab.check_report(_report(), _report(sens=_sens(a_rec=4)))
    assert any("reading_sensitivity[A]" in e for e in errs), errs


def test_check_fails_on_an_empty_reading_table():
    errs = ab.check_report(_report(), _report(sens={}))
    assert any("NO reading_sensitivity" in e for e in errs), errs


def test_check_fails_on_a_vacuous_zero_row_report():
    """House rule 6: a validator that silently checks ZERO rows has failed, not passed."""
    live = _report()
    live["rows"] = {}
    errs = ab.check_report(_report(), live)
    assert any("ZERO rows" in e for e in errs), errs


@pytest.mark.parametrize("kw", [
    {"lo": -0.01}, {"hi": 0.09}, {"sign": "positive"},
])
def test_check_fails_on_gap_drift_including_the_sign(kw):
    """The sign is the sentence a reader quotes. It is compared directly rather than
    transitively through the composite bounds, because it is the number that was over-claimed
    three times."""
    errs = ab.check_report(_report(), _report(gap=_gap(**kw)))
    assert any("gap.adopted" in e for e in errs), errs


def test_check_fails_when_a_point_estimate_is_injected_while_unjudged():
    live = _report()
    live["rows"]["qwen2.5-7b-bnb-openbook"]["adopted"]["composite"] = 0.61
    errs = ab.check_report(_report(), live)
    assert any("point estimate" in e for e in errs), errs


def test_check_tolerance_is_calibrated():
    """1e-7 is representation noise from the report's 6dp rounding; 1e-4 is a real move."""
    assert ab.check_report(_report(), _report(abst=0.7234 + 1e-7)) == []
    assert ab.check_report(_report(), _report(abst=0.7234 + 1e-4)) != []


def test_check_handles_a_missing_report():
    assert ab.check_report({}, _report()) == ["missing committed or live report"]
    assert ab.check_report(_report(), {}) == ["missing committed or live report"]
