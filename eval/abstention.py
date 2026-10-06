"""The assertion-based abstention detector (T-089) — the rule that replaces the fitted one.

WHAT THIS REPLACES, AND WHY

`eval/judge.py:80` calls a text a refusal by SUBSTRING (`ABSTAIN_SENTENCE.rstrip(" .") in text`).
T-084 measured the damage and shipped `eval/abstention_audit.py`, whose STRICT rule is

    loose AND no CITATION_RE match AND len(text.strip()) <= 1.5 * len(abstain_sentence)

That rule works, but it carries a **fitted constant** — the 1.5 — and a constant chosen after the
numbers were seen is the thing a hostile reader attacks first. This module drops it. The rule is

    abstention = emitted the sentence AND did not ASSERT

and `assert` is decided by looking at what is LEFT once the sentence and every citation bracket
are removed. If alphabetic prose remains, the model said something; saying something is not
refusing. Rationale for stripping the brackets rather than counting them: **a citation is a
pointer, not a proposition.** `[Aviva, Motor Wording, Section 3, p.14]` asserts nothing on its
own; the sentence around it does.

No length bound, no citation count, no threshold. One primitive, and it is a property of the
text rather than of a number someone picked.

THE PREDICATE WAS AMBIGUOUS AND THE READINGS WERE MEASURED BEFORE ONE WAS ADOPTED

The rule as originally stated ("residual alphabetic prose remains after stripping the sentence
and all CITATION_RE brackets AND >=1 citation match") admits three readings. All three were
implemented and run over the four frozen rows BEFORE any was chosen, and `reading_sensitivity`
in the report keeps all three visible permanently:

    A  assert = residual prose AND >=1 citation
    B  assert = residual prose                      <- ADOPTED
    C  assert = residual prose OR  >=1 citation

Measured on the frozen answers (2026-07-30):

    row                        A            B            C
    raft-openbook (tuned)      0 recovered  0 recovered  0 recovered   abstention 0.7461 in ALL
    bnb-openbook  (base)      19 / 0.7234  19 / 0.7234  20 / 0.6140
    bnb-closedbook             4 recovered 24 recovered 24 recovered

Three things that decided it, all of them measurements rather than preferences:

  1. **The invariance evidence does not discriminate.** The tuned row is EXACTLY 0.7461 under all
     three readings, and under T-084's length rule at every one of its eight multiples. That the
     rule leaves the tuned row untouched is real and worth stating -- but it is not evidence FOR
     any particular reading, and quoting it as such would be the kind of slip this repo punishes.
  2. **Reading C changes nothing at all.** It reproduces T-084's length rule on all four rows
     exactly. It would remove the fitted constant at zero cost -- an attractive property, and the
     strongest possible "this was not a metric change" defence -- but it also keeps the citation
     test, so it is a re-derivation rather than a correction.
  3. **Reading A cannot see an uncited assertion, and that is disqualifying.** A and B agree on
     both open-book rows and diverge only on closed-book (4 vs 24), because closed-book answers
     carry no citations at all. Under A, an answer that states an uncited fact is classified as
     "not asserting" and KEEPS its abstention credit -- yet an uncited claim is strictly worse
     than a cited one: it violates rule 2 of `src/rag.py`'s SYSTEM_TEMPLATE. A rule that is
     blindest exactly where the model is least grounded is the wrong rule.
     `tests/test_abstention_detector.py` already pins that case as a KNOWN FALSE POSITIVE of the
     length rule; reading B fixes it, reading A structurally cannot. `tests/test_abstention.py`
     asserts it fixed.

WHAT THE ADOPTED RULE DOES TO THE PUBLISHED BOARD -- and one finding that is bigger than the fix

Under reading B the base row's over-refusals fall **36/118 -> 17/118**, and the tuned row's stay
**17/118**. The two rows are then IDENTICAL on that count. So the over-refusal gap between base
and tuned that the board has been reporting is, in its entirety, a measurement artifact of the
substring test -- not a behavioural difference. Anything that quotes "base 36 vs tuned 17" as a
model property (T-076's board text does) is quoting the detector, not the model.

The correction stays BIDIRECTIONAL, which is what keeps it from being advocacy: the same rule
also withdraws credit on the unanswerable side, where the base row's correct refusals fall
16/22 -> 13/22. Net, its abstention axis moves 0.7111 -> 0.7234.

BOUNDS, NOT POINTS -- and the one condition under which that changes

The recovered records were never judged: `judge.py:92-93` skips them at prepare time, so no
`judge_scores.jsonl` row exists for any of them. Their correctness is UNKNOWN. This module
therefore reports an interval, via `abstention_audit.correctness_bounds`, and
`point_estimate_leaks` here refuses to let a single corrected number appear -- WITH ONE
EXCEPTION, which is the whole point of boarding T-091:

    a point estimate is permitted for a block if and only if `n_recovered_unjudged` is the
    INTEGER 0 -- i.e. every record the rule recovered now carries a real judge score.

That condition is mechanical and gate-checked at three levels, because two of them were shown
insufficient by adversarial review before this landed: the count must be a true integer zero
(`False` and `0.0` compare equal to 0 in Python and were both able to unlock it), the count is
itself a gated field, and `check_report` cross-checks it against `n_recovered -
n_recovered_with_judge_score` so it cannot simply be written. Judge the recovered records
and the bound legitimately collapses to a measurement; until then it cannot, however convenient
that would be. `judge.py aggregate` will NOT reflect those judgements on its own -- it computes
`abst` with the loose rule at :190 and hard-zeros before the score lookup at :194-195 -- so the
corrected axes are computed HERE, which is also why this module reimplements them rather than
importing the aggregate.

WHAT THIS CANNOT DO
  (a) It is a detector, not a judge. Recovering a record says it deserves to be judged, not that
      it would score well. The expected direction is mildly UNFAVOURABLE to our own fine-tune.
  (b) It inherits judge.py's exact substring key, so a model that PARAPHRASES the refusal is
      invisible to it, exactly as it is invisible to judge.py.
  (c) "Prose" is `[A-Za-z]`, so a residue of pure punctuation or digits does not count as an
      assertion. A bare "£500." appended to the refusal sentence reads as a refusal. This is a
      deliberate floor, not an oversight: the alternative is a word-count threshold, which is
      the fitted constant this module exists to remove.
  (d) It does not edit `eval/judge.py`. That file is on the tier2 hot list (OWNERS.toml:64) and
      needs the other human. This is a sibling that measures; changing the published metric is a
      separate, reviewed decision, raised as `req:S->A`.
  (e) It does not touch `eval/harness.py` either -- also tier2, and modified by the open PR #17.

FIREWALL: the report is ids, counts and floats. `reclassify` takes text and returns IDS; no
answer, question or clause text is ever read into it. The abstain sentence is OUR string
(src/common.py:29), which is what lets the pytest half build adversarial cases without the
corpus.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# PURE CORE. No IO and no heavy imports -- the citation regex is INJECTED rather than
# imported, so this half needs neither `common` nor the submodule. That is what lets
# tests/test_abstention.py contain ZERO data-dependent tests: CI checks out without
# policy_qa, and pytest exits 0 when every test skips, so a skippable half is a false green
# (eval/reranker_ab.py:620 rejects exactly this in writing). All data dependence lives in
# `--verify`, which never skips -- measured, not assumed: the suite passes with an absent
# submodule, a bogus POLICY_QA_HOME and an unset one, with zero skips in all three.
#
# NOT "no globals", and the earlier draft of this comment said so wrongly. Loading the T-084
# sibling below executes its whole module body, IO layer included, which prepends up to six
# entries to `sys.path`. Nothing is READ, so the CI property above holds -- but the import is
# not side-effect-free and claiming it was is the kind of unchecked adjective this repo's
# review pass exists to catch.
# --------------------------------------------------------------------------------------

ADOPTED_READING = "B"

READING_RULES = {
    "A": "residual alphabetic prose AND >=1 citation match",
    "B": "residual alphabetic prose (citations stripped, then ignored)",
    "C": "residual alphabetic prose OR >=1 citation match",
}

ALPHA_RE = re.compile(r"[A-Za-z]")

# Reused verbatim from the T-084 sibling rather than restated. eval/error_decomposition.py:258-262
# records why a re-spelling is not acceptable here: a reimplementation using the bare sentence
# would be a strictly weaker prefix match -- identical on today's data and silently divergent on
# a truncated refusal. One rule, one definition, imported by explicit path.
_AUDIT_PATH = Path(__file__).resolve().parent / "abstention_audit.py"


def _load(name: str, path: Path):
    """Load a sibling module by EXPLICIT PATH. Never by name off sys.path.

    policy_qa/eval/ held 2026-07-03 shadows of five eval modules that won on sys.path until
    T-071 deleted them. Explicit-path loading is immune to their return.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:            # pragma: no cover - unreachable on disk
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_audit = _load("_abstention_audit_for_abstention", _AUDIT_PATH)

is_abstention_loose = _audit.is_abstention_loose
abstention_axis = _audit.abstention_axis
correctness_bounds = _audit.correctness_bounds
composite_bounds = _audit.composite_bounds
gap_bounds = _audit.gap_bounds
POINT_ESTIMATE_KEYS = _audit.POINT_ESTIMATE_KEYS
_corr_item = _audit._corr_item          # judge.py:200-201 for one non-refusing answerable record


def strip_citations(text: str, *, citation_re) -> str:
    """Remove every bracketed citation. `citation_re` has NO DEFAULT, deliberately.

    `eval/coverage_proxy.py:171` has a function of the same name, and it is NOT imported here:
    it binds its own byte-for-byte COPY of the pattern (justified there -- its pure core may not
    touch the submodule). This module's data path must bind the CANONICAL `common.CITATION_RE`,
    the same object `rag.verify_citations` uses for the citation axis, so the detector and the
    axis can never disagree about what a citation is. Injection is how that is guaranteed, and
    an accidental default would quietly reintroduce the second copy.
    """
    return citation_re.sub(" ", text)


def residual(text: str, *, abstain_sentence: str, citation_re) -> str:
    """What is left after removing every citation and every copy of the refusal sentence.

    Both spellings of the sentence are removed -- the full string and judge.py's 84-character
    `rstrip(" .")` key -- longest first, because the key is a PREFIX of the sentence and
    removing it first would strand the trailing ".".
    """
    out = strip_citations(text, citation_re=citation_re)
    out = out.replace(abstain_sentence, " ")
    return out.replace(abstain_sentence.rstrip(" ."), " ")


def n_citations(text: str, *, citation_re) -> int:
    return len(citation_re.findall(text))


def asserts(text: str, *, abstain_sentence: str, citation_re,
            reading: str = ADOPTED_READING) -> bool:
    """Did the model state something, over and above emitting the refusal sentence?

    `reading` is explicit so all three candidates stay executable forever. The report's
    `reading_sensitivity` block runs every one of them on every row; a rule defended only at its
    chosen reading is a rule nobody can audit.
    """
    if reading not in READING_RULES:
        raise ValueError(f"unknown reading {reading!r}; expected one of {sorted(READING_RULES)}")
    prose = bool(ALPHA_RE.search(residual(
        text, abstain_sentence=abstain_sentence, citation_re=citation_re)))
    if reading == "B":
        return prose
    cited = n_citations(text, citation_re=citation_re) >= 1
    return (prose and cited) if reading == "A" else (prose or cited)


def is_abstention_assertion(text: str, *, abstain_sentence: str, citation_re,
                            reading: str = ADOPTED_READING) -> bool:
    """The adopted rule. A SUBSET of loose by construction, exactly as T-084's strict rule is.

    That subset property is not a detail: it means this correction can only ever move records OUT
    of the over-refusal bucket, never in. Anything else would be a NEW abstention rule rather
    than a correction of the existing one, and would need its own pre-registration.
    """
    if not is_abstention_loose(text, abstain_sentence=abstain_sentence):
        return False
    return not asserts(text, abstain_sentence=abstain_sentence,
                       citation_re=citation_re, reading=reading)


def reclassify(records: list[dict], *, abstain_sentence: str, citation_re,
               reading: str = ADOPTED_READING) -> dict:
    """Bucket the records under loose and under the adopted rule. Returns IDS AND COUNTS ONLY.

    `records` are {"id", "text", "answerable"}. `lost_refusal_ids` is the mirror image of
    `recovered_ids` on the unanswerable side -- records loose counted as a correct refusal and
    the adopted rule does not. It is reported because the correction is BIDIRECTIONAL, and an
    audit that showed only the favourable half would be advocacy.
    """
    loose_over, adopted_over, recovered = [], [], []
    loose_ok, adopted_ok, lost = [], [], []
    ans_n = refuse_n = 0
    for r in records:
        text = r["text"]
        lo = is_abstention_loose(text, abstain_sentence=abstain_sentence)
        ad = lo and not asserts(text, abstain_sentence=abstain_sentence,
                                citation_re=citation_re, reading=reading)
        if r["answerable"]:
            ans_n += 1
            if lo:
                loose_over.append(r["id"])
                (adopted_over if ad else recovered).append(r["id"])
        else:
            refuse_n += 1
            if lo:
                loose_ok.append(r["id"])
                (adopted_ok if ad else lost).append(r["id"])
    return {
        "ans_n": ans_n, "refuse_n": refuse_n,
        "n_loose_over": len(loose_over), "n_adopted_over": len(adopted_over),
        "n_recovered": len(recovered),
        "n_loose_refuse_ok": len(loose_ok), "n_adopted_refuse_ok": len(adopted_ok),
        "n_lost_refusal": len(lost),
        "loose_over_ids": sorted(loose_over),
        "adopted_over_ids": sorted(adopted_over),
        "recovered_ids": sorted(recovered),
        "lost_refusal_ids": sorted(lost),
    }


def reading_sensitivity(records: list[dict], *, abstain_sentence: str, citation_re) -> dict:
    """Every reading's split, for every row, permanently.

    This is the analogue of T-084's length sweep and it exists for the same reason: the choice
    between readings must remain inspectable by someone who does not trust the chooser. It is
    GATED by check_report, so silently collapsing two readings into one is a red gate.
    """
    out = {}
    for name in sorted(READING_RULES):
        r = reclassify(records, abstain_sentence=abstain_sentence,
                       citation_re=citation_re, reading=name)
        out[name] = {"adopted_over": r["n_adopted_over"], "recovered": r["n_recovered"],
                     "adopted_refuse_ok": r["n_adopted_refuse_ok"],
                     "lost_refusal": r["n_lost_refusal"],
                     "loose_over": r["n_loose_over"],
                     "loose_refuse_ok": r["n_loose_refuse_ok"]}
    return out


def is_zero_count(value) -> bool:
    """Strictly the INTEGER zero. Not False, not 0.0, not "0", not None.

    This is not pedantry, it is the guard. In Python `False == 0` and `0.0 == 0`, so writing the
    exception as a bare `!= 0` would let the boolean `False` -- literally a flag -- unlock a
    published point estimate, which is precisely what the docstring above promises is impossible.
    Caught by adversarial review before this landed, and it is the same bool/int conflation the
    vote validator in eval/judge_votes.py already type-checks for ("so a JSON `true` does not
    score as a 1").
    """
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def point_estimate_leaks(obj, path: str = "") -> list[str]:
    """No BOUNDED quantity may also be published as a single number -- unless it is measured.

    Same structural guard as `abstention_audit.point_estimate_leaks`, with the one exception
    that makes T-091 meaningful: a block whose `n_recovered_unjudged` is integer 0 has no
    unjudged population left, so its interval has collapsed to a measurement and a point estimate
    is legitimate. The exception is keyed on a COUNT rather than a flag -- see is_zero_count for
    why that distinction needs enforcing rather than merely asserting -- and `check_report` gates
    the count itself AND cross-checks it against the row's own totals, so it cannot be set
    freely either.
    """
    leaks: list[str] = []
    if isinstance(obj, dict):
        if any(k.endswith("_lower") for k in obj):
            if not is_zero_count(obj.get("n_recovered_unjudged")):
                for k, v in obj.items():
                    if k in POINT_ESTIMATE_KEYS and isinstance(v, (int, float)) \
                            and not isinstance(v, bool):
                        leaks.append(f"{path}.{k}" if path else k)
        for k, v in obj.items():
            leaks += point_estimate_leaks(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaks += point_estimate_leaks(v, f"{path}[{i}]")
    return leaks


def check_report(committed: dict, live: dict, tol: float = 1e-6) -> list[str]:
    """Enforcement. Returns human-readable breaches; empty list == pass.

    Pure, so the pytest half can feed it VIOLATING inputs and watch it go red. This repo has
    caught nine gate-proxy defects that were satisfied by an empty file, an early exit or a
    filename spelling, so a gate nobody has seen fail is not a gate.
    """
    errs: list[str] = []
    if not committed or not live:
        return ["missing committed or live report"]

    for key in ("schema", "adopted_reading", "abstain_sentence_len"):
        if committed.get(key) != live.get(key):
            errs.append(f"{key}: committed {committed.get(key)!r} != live {live.get(key)!r}")

    cr, lr = committed.get("rows") or {}, live.get("rows") or {}
    if set(cr) != set(lr):
        errs.append(f"row set moved: {sorted(cr)} -> {sorted(lr)}")
    if not lr:
        errs.append("live report contains ZERO rows -- vacuous audit (house rule 6)")

    # `n_recovered_unjudged` is GATED, and that is load-bearing rather than tidy: it is the key
    # that unlocks a point estimate (see point_estimate_leaks). Ungated, it could be set to 0 --
    # or deleted, or set to 999 -- with no gated number moving. Adversarial review demonstrated
    # exactly that end-to-end smuggle before this line existed.
    _BLOCK_KEYS = ("abstention", "corr_lower", "corr_upper", "imputed_at_judged_mean",
                   "n_recovered_unjudged", "composite_lower", "composite_upper")
    for row in sorted(set(cr) & set(lr)):
        c, l = cr[row], lr[row]
        # ...and it must AGREE with the row's own totals. Gating the number only pins it to the
        # committed report; this pins it to reality, so the unlock condition cannot be reached by
        # editing one field.
        lb = (l.get("adopted") or {})
        if "n_recovered_unjudged" in lb:
            want = (l.get("n_recovered") or 0) - (l.get("n_recovered_with_judge_score") or 0)
            if lb["n_recovered_unjudged"] != want:
                errs.append(f"{row}.adopted.n_recovered_unjudged: {lb['n_recovered_unjudged']} "
                            f"but n_recovered - n_recovered_with_judge_score = {want}")
        for k in ("ans_n", "refuse_n", "n_loose_over", "n_adopted_over", "n_recovered",
                  "n_loose_refuse_ok", "n_adopted_refuse_ok", "n_lost_refusal",
                  "n_recovered_with_judge_score"):
            if c.get(k) != l.get(k):
                errs.append(f"{row}.{k}: {c.get(k)} -> {l.get(k)}")
        for ids in ("recovered_ids", "lost_refusal_ids"):
            if sorted(c.get(ids) or []) != sorted(l.get(ids) or []):
                errs.append(f"{row}.{ids}: membership changed")
        for block in ("loose", "adopted"):
            cb, lb = (c.get(block) or {}), (l.get(block) or {})
            for k in _BLOCK_KEYS:
                cv, lv = cb.get(k), lb.get(k)
                if isinstance(cv, (int, float)) and isinstance(lv, (int, float)):
                    if abs(cv - lv) > tol:
                        errs.append(f"{row}.{block}.{k}: {cv} -> {lv}")
                elif cv != lv:
                    errs.append(f"{row}.{block}.{k}: {cv!r} -> {lv!r}")

    # ---- the reading table is GATED, not merely published. Ungated, a refactor could delete
    # readings A and C -- or silently make two of them identical -- and no number would move.
    # That is precisely the hole T-084 measured in its own sensitivity table (deleting the
    # citation criterion left --verify green) and it is closed the same way.
    cs, ls = committed.get("reading_sensitivity") or {}, live.get("reading_sensitivity") or {}
    if set(cs) != set(ls):
        errs.append(f"reading_sensitivity row set moved: {sorted(cs)} -> {sorted(ls)}")
    if not ls:
        errs.append("live report carries NO reading_sensitivity table -- the choice of reading "
                    "would be unauditable")
    for row in sorted(set(cs) & set(ls)):
        # TWO checks, and the second is the one that matters. Comparing against READING_RULES
        # alone is SELF-REFERENTIAL: it is a constant in the module under test, so deleting a
        # rejected reading from READING_RULES deletes it from the expectation too and --verify
        # stays green. Measured -- that is the "relax the constant and the guard relaxes with it"
        # mutation family TASKS.md T-088 boards as M4, and abstention_audit.check_audit already
        # guards its own sweep the right way (a committed-vs-live set comparison). Do both: the
        # committed report is an external witness that READING_RULES cannot edit.
        if set(cs[row]) != set(ls[row]):
            errs.append(f"{row}.reading_sensitivity: reading set moved "
                        f"{sorted(cs[row])} -> {sorted(ls[row])} -- the rejected readings are "
                        f"what make the adopted one auditable")
        missing = set(READING_RULES) - set(ls.get(row) or {})
        if missing:
            errs.append(f"{row}.reading_sensitivity: reading(s) {sorted(missing)} missing -- "
                        f"the rejected readings are what make the adopted one auditable")
        for name in sorted(set(cs[row]) & set(ls[row])):
            for k in ("adopted_over", "recovered", "adopted_refuse_ok", "lost_refusal",
                      "loose_over", "loose_refuse_ok"):
                if cs[row][name].get(k) != ls[row][name].get(k):
                    errs.append(f"{row}.reading_sensitivity[{name}].{k}: "
                                f"{cs[row][name].get(k)} -> {ls[row][name].get(k)}")

    cg, lg = committed.get("gap_tuned_minus_base") or {}, live.get("gap_tuned_minus_base") or {}
    for name in ("loose", "adopted"):
        c, l = cg.get(name) or {}, lg.get(name) or {}
        if bool(c) != bool(l):
            errs.append(f"gap.{name}: present in one report only")
            continue
        for k in ("gap_lower", "gap_upper"):
            cv, lv = c.get(k), l.get(k)
            if isinstance(cv, (int, float)) and isinstance(lv, (int, float)):
                if abs(cv - lv) > tol:
                    errs.append(f"gap.{name}.{k}: {cv} -> {lv}")
            elif cv != lv:
                errs.append(f"gap.{name}.{k}: {cv!r} -> {lv!r}")
        if c.get("sign") != l.get("sign"):
            errs.append(f"gap.{name}.sign: {c.get('sign')!r} -> {l.get('sign')!r}")

    leaks = point_estimate_leaks(live)
    if leaks:
        errs.append("point estimate of a BOUNDED quantity in the live report: "
                    + ", ".join(sorted(leaks)))
    return errs


# --------------------------------------------------------------------------------------
# IO HALF. Everything below reads the private data repo. --verify never skips.
# --------------------------------------------------------------------------------------

PROJ = Path(__file__).resolve().parents[1]
_candidates = [os.environ.get("POLICY_QA_HOME"), PROJ / "policy_qa",
               PROJ.parent / "policy_qa", PROJ]
_home = next((Path(c) for c in _candidates if c and Path(c, "src", "common.py").exists()), PROJ)
for _sub in ("eval", "src"):
    if (_home / _sub).exists():
        sys.path.insert(0, str(_home / _sub))
# ...then THIS script's own directory, inserted LAST so it is searched FIRST. Same comment as
# eval/abstention_audit.py:485 and eval/error_decomposition.py:246.
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPORT_PATH = PROJ / "eval" / "abstention.json"
ROWS = ("qwen2.5-7b-bnb-openbook", "qwen2.5-7b-bnb-closedbook",
        "qwen2.5-7b-raft-openbook", "qwen2.5-7b-raft-closedbook")
BASE_ROW = "qwen2.5-7b-bnb-openbook"
TUNED_ROW = "qwen2.5-7b-raft-openbook"


def build(*, reading: str = ADOPTED_READING) -> dict:
    from common import ABSTAIN_SENTENCE, CITATION_RE, RESULTS_DIR, gold_path, read_jsonl
    import rag

    gold_p = gold_path()
    sco_p = RESULTS_DIR / "judge_scores.jsonl"
    ctx_p = RESULTS_DIR / ("frozen_contexts_v2.jsonl"
                           if gold_p.name == "gold_v2.jsonl" else "frozen_contexts.jsonl")
    for p in (gold_p, sco_p, ctx_p):
        if not Path(p).exists():
            sys.exit(f"[1] missing required input: {p}")

    gold = {g["id"]: g for g in read_jsonl(gold_p)}
    if not gold:
        sys.exit("[2] gold set is EMPTY -- refusing to audit zero records (house rule 6).")
    contexts = {c["qid"]: c for c in read_jsonl(ctx_p)}
    scores = {(s["qid"], s["row"]): s for s in read_jsonl(sco_p)}

    rows_out: dict[str, dict] = {}
    sens_out: dict[str, dict] = {}
    for row in ROWS:
        ans_p = RESULTS_DIR / f"answers_{row}.jsonl"
        if not Path(ans_p).exists():
            sys.exit(f"[1] missing required input: {ans_p}")
        recs, judged, cite_sup, cite_tot = [], [], 0, 0
        for a in read_jsonl(ans_p):
            g = gold.get(a["qid"])
            if not g:
                continue
            answerable = g.get("answerable", True)
            recs.append({"id": a["qid"], "text": a["answer"], "answerable": answerable})
            if a["qid"] in contexts:
                chk = rag.verify_citations(a["answer"], contexts[a["qid"]]["chunks"])
                cite_sup += chk["n_supported"]
                cite_tot += chk["n_total"]
        if not recs:
            sys.exit(f"[2] row {row}: ZERO records matched gold -- refusing to report.")

        cls = reclassify(recs, abstain_sentence=ABSTAIN_SENTENCE,
                         citation_re=CITATION_RE, reading=reading)

        # The judged population is the answerable records the ADOPTED rule does not call a
        # refusal AND that were not recovered -- i.e. exactly judge.py's population, since a
        # recovered record has no score to contribute. Computed from the buckets rather than
        # re-derived, so it cannot drift from the classification above.
        excluded = set(cls["adopted_over_ids"]) | set(cls["recovered_ids"])
        for r in recs:
            if r["answerable"] and r["id"] not in excluded:
                v = _corr_item(scores.get((r["id"], row)))
                if v is not None:
                    judged.append(v)

        # The condition that governs whether a point estimate is legitimate. It is a COUNT of
        # ACTUALLY SCORED recovered records, not a promise and not a flag: T-091 takes it to
        # 19/19 and the bound collapses to a measurement on its own.
        #
        # A record only counts as scored if _corr_item returns a real value. A judge_scores row
        # can exist and still be unscoreable (empty `fact_scores`), in which case _corr_item is
        # None -- and treating None as 0.0 would fabricate a hard zero for a record nobody
        # scored, which is precisely the defect this whole module exists to correct.
        rec_scores = {}
        for q in cls["recovered_ids"]:
            v = _corr_item(scores.get((q, row)))
            if v is not None:
                rec_scores[q] = v
        citation = (cite_sup / cite_tot) if cite_tot else 0.0

        blocks = {}
        for name, over, refuse_ok, vals, n_rec in (
                # Under loose the recovered records ARE over-refusals, folded in at a hard 0.
                ("loose", cls["n_loose_over"], cls["n_loose_refuse_ok"], judged, 0),
                # Under the adopted rule they leave that bucket. Any that have since been judged
                # join the scored population; the rest stay unjudged and keep the interval open.
                ("adopted", cls["n_adopted_over"], cls["n_adopted_refuse_ok"],
                 judged + sorted(rec_scores.values()),
                 cls["n_recovered"] - len(rec_scores))):
            corr = correctness_bounds(vals, over, n_rec)
            abst = abstention_axis(refuse_ok, cls["refuse_n"], over, cls["ans_n"])
            blocks[name] = {**corr, **composite_bounds(corr, citation, abst),
                            "over": over, "refuse_ok": refuse_ok}

        rows_out[row] = {
            "ans_n": cls["ans_n"], "refuse_n": cls["refuse_n"], "n_records": len(recs),
            "n_loose_over": cls["n_loose_over"], "n_adopted_over": cls["n_adopted_over"],
            "n_recovered": cls["n_recovered"],
            "n_loose_refuse_ok": cls["n_loose_refuse_ok"],
            "n_adopted_refuse_ok": cls["n_adopted_refuse_ok"],
            "n_lost_refusal": cls["n_lost_refusal"],
            "n_recovered_with_judge_score": len(rec_scores),
            "recovered_ids": cls["recovered_ids"],
            "lost_refusal_ids": cls["lost_refusal_ids"],
            "adopted_over_ids": cls["adopted_over_ids"],
            "cite_sup": cite_sup, "cite_tot": cite_tot,
            **blocks,
        }
        sens_out[row] = reading_sensitivity(recs, abstain_sentence=ABSTAIN_SENTENCE,
                                            citation_re=CITATION_RE)

    gaps = {}
    if TUNED_ROW in rows_out and BASE_ROW in rows_out:
        for name in ("loose", "adopted"):
            gaps[name] = gap_bounds(rows_out[TUNED_ROW][name], rows_out[BASE_ROW][name])

    return {
        "schema": "abstention/1",
        "task": "T-089",
        "adopted_reading": reading,
        "readings": READING_RULES,
        "rule": "loose AND NOT asserts; asserts = " + READING_RULES[reading],
        "loose_rule": "eval/judge.py:80 -- ABSTAIN_SENTENCE.rstrip(' .') substring test",
        "abstain_sentence_len": len(ABSTAIN_SENTENCE),
        "replaces": "eval/abstention_audit.py's strict rule (drops its 1.5 length multiple, "
                    "the only fitted constant in either rule)",
        "inputs": {"gold": Path(gold_p).name, "contexts": Path(ctx_p).name,
                   "scores": Path(sco_p).name,
                   "answers": [f"answers_{r}.jsonl" for r in ROWS]},
        "rows": rows_out,
        "reading_sensitivity": sens_out,
        "gap_tuned_minus_base": {"tuned": TUNED_ROW, "base": BASE_ROW, **gaps},
    }


def _print(live: dict) -> None:
    print(f"adopted reading {live['adopted_reading']}: assert = "
          f"{live['readings'][live['adopted_reading']]}")
    print(f"abstain sentence: {live['abstain_sentence_len']} chars")
    for row, r in live["rows"].items():
        print(f"\n{row}  (answerable {r['ans_n']}, unanswerable {r['refuse_n']})")
        print(f"  over-refusals  loose {r['n_loose_over']:3d} -> adopted "
              f"{r['n_adopted_over']:3d}   recovered {r['n_recovered']}"
              f"  (judged {r['n_recovered_with_judge_score']}/{r['n_recovered']})")
        print(f"  refusals ok    loose {r['n_loose_refuse_ok']:3d} -> adopted "
              f"{r['n_adopted_refuse_ok']:3d}   lost {r['n_lost_refusal']}")
        for name in ("loose", "adopted"):
            b = r[name]
            print(f"  {name:<9s} abst={b['abstention']:.4f} cite={b['citation']:.4f} "
                  f"corr [{b['corr_lower']:.4f}, {b['corr_upper']:.4f}] "
                  f"composite [{b['composite_lower']:.4f}, {b['composite_upper']:.4f}]")
    g = live["gap_tuned_minus_base"]
    print(f"\ngap {g['tuned']} minus {g['base']}:")
    for name in ("loose", "adopted"):
        if name in g:
            b = g[name]
            print(f"  {name:<9s} [{b['gap_lower']:+.4f}, {b['gap_upper']:+.4f}]  sign {b['sign']}")
    print("\nreading sensitivity (adopted_over / recovered / lost_refusal):")
    for row, s in live["reading_sensitivity"].items():
        cells = "  ".join(f"{n}:{v['adopted_over']}/{v['recovered']}/{v['lost_refusal']}"
                          for n, v in s.items())
        print(f"  {row:<28s} {cells}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write the tracked report")
    ap.add_argument("--verify", action="store_true",
                    help="recompute live and compare against the committed report")
    ap.add_argument("--reading", choices=sorted(READING_RULES), default=ADOPTED_READING)
    a = ap.parse_args()

    live = build(reading=a.reading)

    if a.verify:
        if not REPORT_PATH.exists():
            print(f"FAIL: no committed report at {REPORT_PATH}")
            return 1
        try:
            committed = json.loads(REPORT_PATH.read_text())
        except json.JSONDecodeError as exc:
            print(f"FAIL: committed report is not valid JSON: {exc}")
            return 1
        errs = check_report(committed, live)
        if errs:
            print("ABSTENTION VERIFY FAILED:")
            for e in errs:
                print("  -", e)
            return 1
        b, t = live["rows"][BASE_ROW], live["rows"][TUNED_ROW]
        g = live["gap_tuned_minus_base"]["adopted"]
        print(f"ABSTENTION VERIFY OK -- reading {live['adopted_reading']}; over-refusals "
              f"base {b['n_loose_over']}->{b['n_adopted_over']}, "
              f"tuned {t['n_loose_over']}->{t['n_adopted_over']}; gap "
              f"[{g['gap_lower']:+.4f}, {g['gap_upper']:+.4f}] sign {g['sign']}")
        return 0

    _print(live)
    if a.write:
        REPORT_PATH.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
