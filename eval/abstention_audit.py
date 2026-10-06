"""Is a "refusal" really a refusal? The strict abstention detector and audit (T-084).

THE DEFECT

`eval/judge.py:80` decides whether a model refused with a SUBSTRING test:

    ABSTAIN_SENTENCE.rstrip(" .") in text or text.strip() == ABSTAIN_SENTENCE

An answer that answers the question, cites clauses, and then appends the refusal sentence as a
caveat about some *other* part of the question therefore counts as an over-refusal. That costs
the row twice: a hard 0 on correctness with no judge call at all (judge.py:195, and
judge.py:92-93 means the record was never even sent to the judge), plus a hit on the abstention
axis (judge.py:122,128). Two axes, 0.5 + 0.2 of the composite, charged for a formatting habit.

MEASURED ON DISK (2026-07-30, this module's own `--verify` output, not inherited):
  qwen2.5-7b-bnb-openbook   36 loose over-refusals, 20 of them carrying citations (median 121
                            words, max 293) -> 16 strict, 20 recovered
  qwen2.5-7b-raft-openbook  17 loose, 17 strict, 0 recovered
  qwen2.5-7b-bnb-closedbook 69 loose -> 45 strict, 24 recovered
  qwen2.5-7b-raft-closedbook 0 loose, 0 strict
The asymmetry is expected: the RAFT tune taught the model to emit the sentence BARE, the base
model uses it as a hedge. So the artifact penalises the BASE rows only, which is exactly the
direction that flatters the headline "tuning helped" claim.

PRE-REGISTERED CRITERIA (fixed BEFORE the sensitivity sweep below was run; the sweep is
reported in full precisely so the reader can see the choice was not fitted to it).
`is_abstention_strict` calls a text an abstention when ALL THREE hold:
  (i)   it satisfies judge.py's loose test -- strict is a SUBSET of loose BY CONSTRUCTION, so
        this audit can only ever move records OUT of the over-refusal bucket, never in. Anything
        else would be a new abstention rule rather than a correction of this one.
  (ii)  it contains NO citation matching `src/common.py`'s CITATION_RE. Justification: the
        contract in `src/rag.py` is "every factual claim ends with a citation". A citation is
        therefore the model's own marker that it asserted something. An answer that asserts
        something is not a refusal, whatever else it also says.
  (iii) `len(text.strip()) <= max_len_multiple * len(abstain_sentence)`. Justification: a
        genuine refusal IS the sentence; (iii) bounds the residual material at
        (multiple - 1) x the sentence's own length. Length is used, not word count, so the
        threshold is expressed in the sentence's own units and needs no separate constant.

WHY multiple = 1.5, and the sensitivity that makes the choice visible (`sensitivity` in the
report; sweep over 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0):
  * On the DECISION row (bnb-openbook) the 36 -> 16/20 split is INVARIANT across the whole
    sweep, including the unbounded column. Its loose-flagged population is perfectly bimodal:
    16 answers at exactly 1.00x the sentence, 20 at 3.29x-21.14x, all 20 carrying citations.
    So (ii) and (iii) are CONFOUNDED on that row -- each on its own rejects the same 20 -- and
    the headline number is not a function of this constant at all. (The confounding is the
    reason the unbounded column exists at all: see SENSITIVITY_MULTIPLES and check_audit. An
    earlier draft asserted here that "(ii) does all the work and (iii) does none"; measured on
    disk that is false at the adopted multiple, where either criterion alone suffices. (ii)
    only becomes the sole discriminator above 3.29x.)
  * The multiple only bites on bnb-closedbook, where 1.0 gives 4 strict and 1.1-3.0 gives 45.
    The cliff at 1.0 is a quoting habit: 41 answers are the bare sentence inside quotation
    marks, 87 chars against the sentence's 85 (1.02x). Excluding those would be the same
    substring-literalism this module exists to fix. On the unanswerable side of the same row the
    plateau is 1.1-2.0 (9 refusals). 1.5 is the interior of the widest jointly-flat interval.
  * A single constant is used for all four rows on purpose. Per-row tuning of a detector whose
    output feeds a between-row comparison is how you manufacture the comparison.

WHAT THIS MODULE REFUSES TO DO -- read this before quoting any number out of it.
The recovered records were NEVER JUDGED. judge.py:92-93 skips them at `prepare` time, so no
`judge_scores.jsonl` row exists for any of them (verified: 0 of the 36 / 69 flagged records on
the two base rows carry a score). Their correctness is therefore UNKNOWN, not zero and not
average. This module emits `corr_lower` (recovered score 0 -- which is arithmetically identical
to the published number, so the published correctness is exactly the LOWER bound) and
`corr_upper` (recovered score 1.0, the arithmetic ceiling of mean(fact_scores)/2), and
`point_estimate_leaks()` makes it a hard `--verify` failure for the report to carry a single
corrected correctness or composite anywhere. Emitting one would be the exact defect being
corrected here: an unmeasured quantity dressed as a measurement.

CORRECTED 2026-07-30, AND IT MOVED A PUBLISHED SIGN. `corr_upper` originally imputed the row's
own judged mean. That is a central estimate, not an upper bound -- see MAX_RECORD_CORRECTNESS
for the full record. The imputation survives as `imputed_at_judged_mean`, gated but never an
endpoint. Everything downstream moved: base composite [0.5531, 0.5923] -> [0.5531, 0.6378],
gap [+0.0068, +0.0460] -> [-0.0387, +0.0460], `sign` positive -> INDETERMINATE.

A CORRECTION TO THE T-084 BRIEF, measured, stated loudly. The brief's impact figures
(0.5894 abstention-only, 0.6287 recovered-at-mean, gap -0.0296..+0.0097) apply the strict rule
to the ANSWERABLE side while leaving the UNANSWERABLE side loose. They are reproduced here as
`answerable_side_only` and are NOT adopted. Strict applied consistently to both sides also
removes credit for hedged refusals on unanswerable questions -- bnb-openbook drops from 16/22
correct refusals to 8/22 -- which pushes the base row's abstention axis DOWN, from 0.7111 to
0.6140, not up to 0.7959. Under the consistent rule the base composite lands in
[0.5531, 0.6378] and the tuned-minus-base gap in [-0.0387, +0.0460]: it widens, and its SIGN
IS INDETERMINATE. (An earlier version of this paragraph read "[+0.0068, +0.0460]: it widens,
and it does not flip sign." That was the mean-imputation bug, not a finding; it is left visible
here rather than edited out.) The one-sided rule is a separate error: there is no principled
reason an answer that asserts cited content counts as a refusal on an unanswerable question but
not on an answerable one.

WHAT ELSE THIS CANNOT DO. (a) It is a detector, not a judge: recovering a record says the
record deserves to be judged, not that it would score well. (b) It inherits judge.py's exact
substring key, so a model that paraphrases the refusal is invisible to loose AND to strict.
(c) The citation criterion is a regex, so a claim asserted WITHOUT a citation and under the
length bound still reads as a refusal. (d) It does not edit judge.py: that file is on the
tier2 hot list (OWNERS.toml) and its reviewer is mid-queue. This is a sibling that measures the
damage; changing the published metric is a separate, reviewed decision.

FIREWALL: the report is ids, counts and floats only. No answer text, no question text, no
clause text is read into it -- `reclassify` takes text and returns IDS. The abstention sentence
itself is OUR string (src/common.py:29), not insurer text, which is what makes the pytest half
able to build adversarial cases without touching the corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# PURE CORE. No IO, no globals, no heavy imports -- `re` objects are INJECTED rather than
# imported so this half needs neither `common` nor the submodule. That is what lets
# tests/test_abstention_detector.py contain ZERO data-dependent tests: CI checks out without
# policy_qa, and pytest exits 0 when every test skips, so a skippable half is a false green
# (eval/reranker_ab.py:620). All data dependence lives in `--verify`, which never skips.
# --------------------------------------------------------------------------------------

STRICT_LEN_MULTIPLE = 1.5
# `inf` is not decoration and not a 'why not'. It is the ONLY column in which criterion (ii)
# is observable on its own: at an unbounded length the strict rule reduces to "loose AND no
# citation", so the column moves if and only if (ii) changes. Measured 2026-07-30 by deleting
# criterion (ii) from is_abstention_strict: with the sweep ending at 4.0 and `sensitivity`
# ungated, `--verify` stayed GREEN -- criterion (iii) alone reproduces the adopted split on all
# four rows (24/54/31/0 of the loose-flagged records), so the deletion moved no gated number.
# The pytest half caught it; the data half did not, and a criterion the artifact gate cannot
# see is a criterion that can be deleted in a refactor. See check_audit.
SENSITIVITY_MULTIPLES = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, float("inf"))
UNBOUNDED_KEY = f"{float('inf'):g}"          # "inf" -- the criterion-(ii) probe column

# Keys that would constitute a point estimate of a quantity this module has bounded rather
# than measured. Enforced by point_estimate_leaks() and by --verify.
POINT_ESTIMATE_KEYS = (
    "corr", "correctness", "corr_corrected", "corrected_correctness",
    "composite", "corrected_composite", "corr_point", "composite_point",
)

# The largest correctness a single answerable record can carry. judge.py:200-201 scores a
# record as mean(fact_scores)/2 with fact_scores drawn from {0,1,2}, so the per-record range
# is exactly [0, 1] and 1.0 is the arithmetic ceiling -- not a modelling assumption.
#
# THIS CONSTANT IS THE FIX FOR A REAL DEFECT, recorded because it moved a published sign.
# Until 2026-07-30 `corr_upper` imputed the row's OWN JUDGED MEAN to the recovered records.
# That is a central estimate wearing an `_upper` suffix: it answers "what if the unjudged
# records are typical?", which is a guess, not a bound. Every consumer -- composite_bounds,
# gap_bounds, `sign_determined`, and the leaderboard note quoting them -- inherited the
# too-narrow interval, and the base row's gap was published as `positive` on the strength of
# it. With the real ceiling the base composite upper moves 0.5923 -> 0.6378 and the gap's
# lower end moves +0.0068 -> -0.0387: the sign is INDETERMINATE. The mean imputation is still
# reported, as `imputed_at_judged_mean`, because it is genuinely informative -- but it is
# named for what it is and it is never an endpoint.
MAX_RECORD_CORRECTNESS = 1.0


def is_abstention_loose(text: str, *, abstain_sentence: str) -> bool:
    """judge.py:80, restated so the pure core needs no import of judge.py or `common`.

    Restated rather than imported ON PURPOSE, and the restatement is not trusted: the pytest
    half loads eval/judge.py by explicit path and asserts byte-for-byte agreement with
    `judge.is_abstention` over the whole adversarial battery. That is stronger than a comment
    promising the two match, and it does not drag `common`/`rag` into the pure core.
    """
    return abstain_sentence.rstrip(" .") in text or text.strip() == abstain_sentence


def is_abstention_strict(text: str, *, abstain_sentence: str, citation_re,
                         max_len_multiple: float = STRICT_LEN_MULTIPLE) -> bool:
    """The answer must BE the refusal, not merely contain it. Criteria pre-registered above.

    `citation_re` is a REQUIRED keyword with no default. A default of None would let a caller
    silently run without criterion (ii) -- the criterion that does all the work on the decision
    row -- and get a number that looks like a strict count and is not one. Pass
    `src/common.py`'s CITATION_RE; never a local re-spelling of it (house rule 5).

    Returns False for text that is not a loose abstention at all, so
    `strict(t) implies loose(t)` holds for every string by construction.
    """
    if not is_abstention_loose(text, abstain_sentence=abstain_sentence):
        return False
    if citation_re.search(text):
        return False
    return len(text.strip()) <= max_len_multiple * len(abstain_sentence)


def reclassify(records: list[dict], *, abstain_sentence: str, citation_re,
               max_len_multiple: float = STRICT_LEN_MULTIPLE) -> dict:
    """Split one row's records into the four buckets the two axes are built from.

    `records`: [{"id", "text", "answerable"}]. Returns IDS and counts only -- no text escapes
    this function, which is what keeps the report firewall-clean.

    `recovered_ids` are the answerable records that loose calls an over-refusal and strict does
    not. They are the whole subject of this audit: each one was denied a judge call by
    judge.py:92-93, so its correctness is unknown rather than zero.

    `lost_refusal_ids` is the mirror image on the unanswerable side -- records that loose
    counted as a correct refusal and strict does not. Reported because the correction is
    BIDIRECTIONAL and an audit that showed only the favourable half would be advocacy.
    """
    out = {
        "ans_n": 0, "refuse_n": 0,
        "loose_over_ids": [], "strict_over_ids": [], "recovered_ids": [],
        "loose_refuse_ok_ids": [], "strict_refuse_ok_ids": [], "lost_refusal_ids": [],
    }
    for r in records:
        loose = is_abstention_loose(r["text"], abstain_sentence=abstain_sentence)
        strict = is_abstention_strict(r["text"], abstain_sentence=abstain_sentence,
                                      citation_re=citation_re,
                                      max_len_multiple=max_len_multiple)
        if r["answerable"]:
            out["ans_n"] += 1
            if loose:
                out["loose_over_ids"].append(r["id"])
                (out["strict_over_ids"] if strict else out["recovered_ids"]).append(r["id"])
        else:
            out["refuse_n"] += 1
            if loose:
                out["loose_refuse_ok_ids"].append(r["id"])
                (out["strict_refuse_ok_ids"] if strict
                 else out["lost_refusal_ids"]).append(r["id"])
    for k in list(out):
        if k.endswith("_ids"):
            out[k].sort()
            out["n_" + k[:-4]] = len(out[k])
    return out


def abstention_axis(refuse_ok: int, refuse_n: int, over: int, ans_n: int) -> float:
    """judge.py:127-129 exactly: mean of (correct-refusal rate, non-over-refusal rate).

    The empty-denominator fallbacks are judge.py's, deliberately: 1.0, not 0.0. Changing them
    here would make this module disagree with the published metric for a reason unrelated to
    the defect it exists to measure.
    """
    refusal_acc = (refuse_ok / refuse_n) if refuse_n else 1.0
    non_over = 1 - (over / ans_n) if ans_n else 1.0
    return (refusal_acc + non_over) / 2


def correctness_bounds(judged_vals: list[float], n_over: int, n_recovered: int) -> dict:
    """BOUNDS, never a point estimate, on the corrected correctness axis.

    The denominator mirrors judge.py:117-118: answerable records whose corr_item is not None,
    i.e. the judged ones plus the ones scored 0 for over-refusing. Answerable records that are
    neither refusals nor judged carry corr_item None and are excluded from BOTH bounds, exactly
    as judge.py excludes them.

    corr_lower  -- the n_recovered unjudged records all score 0. This is arithmetically
                   IDENTICAL to the published number (loose folds them into n_over at 0), so
                   the published correctness is the lower bound, not a central estimate.
    corr_upper  -- they score MAX_RECORD_CORRECTNESS (1.0) each. A record that was never sent
                   to the judge could have scored anything in [0, 1], so this is the only
                   arithmetically defensible ceiling. It is deliberately wide: the width IS
                   the finding, and narrowing it needs judgements, not assumptions.

    imputed_at_judged_mean -- they score at this row's own judged mean. Reported because "what
                   if they are typical?" is a fair question, and suppressing the answer would
                   be its own kind of dishonesty. It is NOT an endpoint and is deliberately
                   named so it cannot be mistaken for one: it is a guess about unmeasured
                   records, and this module exists because such a guess was once published as
                   a bound (see MAX_RECORD_CORRECTNESS).

    Returns no key that could be read as a single corrected value -- see POINT_ESTIMATE_KEYS
    and point_estimate_leaks(). Raises rather than returning a bound when the population is
    empty: house rule 6, a validator that silently scores zero records has failed.
    """
    denom = len(judged_vals) + n_over + n_recovered
    if denom == 0:
        raise ValueError("correctness_bounds over an EMPTY population -- refusing to "
                         "report. Zero scored records is the T-011 false green, not a 0.0.")
    total = sum(judged_vals)
    mean = (total / len(judged_vals)) if judged_vals else None
    lower = total / denom
    upper = (total + MAX_RECORD_CORRECTNESS * n_recovered) / denom
    imputed = (total + (mean or 0.0) * n_recovered) / denom
    return {
        "n_judged": len(judged_vals),
        "judged_mean": round(mean, 6) if mean is not None else None,
        "n_over": n_over,
        "n_recovered_unjudged": n_recovered,
        "denominator": denom,
        "corr_lower": round(lower, 6),
        "corr_upper": round(upper, 6),
        "imputed_at_judged_mean": round(imputed, 6),
        "bound_width": round(upper - lower, 6),
        "note": "recovered records were never judged (judge.py:92-93); their scores are "
                "UNKNOWN. corr_lower == the published correctness by construction; corr_upper "
                "scores every unjudged record at the 1.0 ceiling. imputed_at_judged_mean is a "
                "GUESS, not an endpoint -- it was once published as `corr_upper` and that "
                "narrowed the gap enough to report a sign that the data does not support.",
    }


def composite_bounds(corr: dict, citation: float, abstention: float) -> dict:
    """0.5*correctness + 0.3*citation + 0.2*abstention (judge.py:131), carried as an interval.

    The citation axis is untouched by this correction: judge.py:119-120 sums citations over
    every record regardless of abstention, so a recovered record's citations were already
    counted. It is passed in rather than recomputed so this stays pure.
    """
    base = 0.3 * citation + 0.2 * abstention
    return {
        "composite_lower": round(0.5 * corr["corr_lower"] + base, 6),
        "composite_upper": round(0.5 * corr["corr_upper"] + base, 6),
        "citation": round(citation, 6),
        "abstention": round(abstention, 6),
    }


def gap_bounds(a: dict, b: dict) -> dict:
    """Interval for composite(a) - composite(b) from two `composite_bounds` dicts.

    Interval arithmetic, not a difference of midpoints: lo = a_lo - b_hi, hi = a_hi - b_lo.
    `sign_determined` is the only honest way to answer "did tuning help" from bounded inputs --
    it is True only when the whole interval sits on one side of 0.
    """
    lo = a["composite_lower"] - b["composite_upper"]
    hi = a["composite_upper"] - b["composite_lower"]
    return {"gap_lower": round(lo, 6), "gap_upper": round(hi, 6),
            "sign_determined": bool(lo > 0 or hi < 0),
            "sign": ("positive" if lo > 0 else "negative" if hi < 0 else "indeterminate")}


def sensitivity(records: list[dict], multiples, *, abstain_sentence: str, citation_re) -> dict:
    """The strict split at each candidate length multiple, so the constant is visible.

    Reported for every row and every multiple rather than for the chosen one, because a
    threshold defended only at its chosen value is a threshold nobody can audit.

    The `inf` column carries a second, separate job: it is the only place criterion (ii) is
    observable independently of criterion (iii), so gating this table is what makes deleting
    the citation test a RED gate rather than a silent no-op (see SENSITIVITY_MULTIPLES).
    """
    out = {}
    for m in multiples:
        r = reclassify(records, abstain_sentence=abstain_sentence, citation_re=citation_re,
                       max_len_multiple=m)
        out[f"{m:g}"] = {"strict_over": r["n_strict_over"],
                         "recovered": r["n_recovered"],
                         "strict_refuse_ok": r["n_strict_refuse_ok"],
                         "loose_over": r["n_loose_over"],
                         "loose_refuse_ok": r["n_loose_refuse_ok"]}
    return out


def point_estimate_leaks(obj, path: str = "") -> list[str]:
    """Structural guard: no bounded quantity may also be published as a single number.

    Walks the report and flags any dict that carries a `*_lower`/`*_upper` pair AND a scalar
    key from POINT_ESTIMATE_KEYS. `--verify` exits non-zero on a non-empty result, so the
    invariant is enforced by the gate rather than by reviewer discipline. This is (d) of T-084
    made structural: overclaiming here would be the exact defect being corrected.
    """
    leaks: list[str] = []
    if isinstance(obj, dict):
        bounded = any(k.endswith("_lower") for k in obj)
        if bounded:
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


def check_audit(committed: dict, live: dict, tol: float = 1e-6) -> list[str]:
    """Enforcement. Returns a list of human-readable breaches; empty list == pass.

    Pure so the pytest half can feed it VIOLATING inputs and watch it go red -- this repo has
    caught seven gate-proxy defects that were satisfied by an empty file, an early exit, or a
    filename spelling, so a gate nobody has seen fail is not a gate.
    """
    errs: list[str] = []
    if not committed or not live:
        return ["missing committed or live report"]

    for key in ("schema", "abstain_sentence_len", "len_multiple"):
        if committed.get(key) != live.get(key):
            errs.append(f"{key}: committed {committed.get(key)!r} != live {live.get(key)!r}")

    cr, lr = committed.get("rows") or {}, live.get("rows") or {}
    if set(cr) != set(lr):
        errs.append(f"row set moved: {sorted(cr)} -> {sorted(lr)}")
    if not lr:
        errs.append("live report contains ZERO rows -- vacuous audit (house rule 6)")

    for row in sorted(set(cr) & set(lr)):
        c, l = cr[row], lr[row]
        for k in ("n_loose_over", "n_strict_over", "n_recovered",
                  "n_loose_refuse_ok", "n_strict_refuse_ok", "ans_n", "refuse_n"):
            if c.get(k) != l.get(k):
                errs.append(f"{row}.{k}: {c.get(k)} -> {l.get(k)}")
        if sorted(c.get("recovered_ids") or []) != sorted(l.get("recovered_ids") or []):
            errs.append(f"{row}.recovered_ids: membership changed")
        # `imputed_at_judged_mean` is gated alongside the endpoints on purpose. It is the
        # quantity that was once mislabelled `corr_upper`, so leaving it ungated would let a
        # refactor silently reinstate the imputation as a bound -- the exact defect this
        # module now records. Gating it makes that swap a RED gate.
        _BLOCK_KEYS = ("abstention", "corr_lower", "corr_upper", "imputed_at_judged_mean",
                       "composite_lower", "composite_upper")
        for block, keys in (("loose", _BLOCK_KEYS),
                            ("strict", _BLOCK_KEYS),
                            # The one-sided variant is gated too. It is the variant the T-084
                            # brief quotes, so it is the one a reader is most likely to lift.
                            ("answerable_side_only", _BLOCK_KEYS)):
            cb, lb = (c.get(block) or {}), (l.get(block) or {})
            for k in keys:
                cv, lv = cb.get(k), lb.get(k)
                if isinstance(cv, (int, float)) and isinstance(lv, (int, float)):
                    if abs(cv - lv) > tol:
                        errs.append(f"{row}.{block}.{k}: {cv} -> {lv}")
                elif cv != lv:
                    errs.append(f"{row}.{block}.{k}: {cv!r} -> {lv!r}")

    # ---- the sensitivity table is GATED, not merely published.
    # Measured hole (2026-07-30): with this table ungated, deleting criterion (ii) from
    # is_abstention_strict left every gated number identical and --verify passed. Criterion
    # (iii) alone reproduces the adopted split on all four rows, so the two criteria are
    # confounded at the adopted multiple and only separate at the top of the sweep. Gating the
    # table -- and requiring the unbounded column, where strict reduces to "loose AND no
    # citation" -- is what makes each criterion independently load-bearing at the gate.
    cs, ls = committed.get("sensitivity") or {}, live.get("sensitivity") or {}
    if set(cs) != set(ls):
        errs.append(f"sensitivity row set moved: {sorted(cs)} -> {sorted(ls)}")
    if not ls:
        errs.append("live report carries NO sensitivity table -- the length multiple would be "
                    "unauditable and criterion (ii) ungated")
    for row in sorted(set(cs) & set(ls)):
        if UNBOUNDED_KEY not in (ls.get(row) or {}):
            errs.append(f"{row}.sensitivity: the {UNBOUNDED_KEY!r} column is missing -- it is "
                        f"the only column in which the citation criterion is observable alone")
        if set(cs[row]) != set(ls[row]):
            errs.append(f"{row}.sensitivity: multiples moved "
                        f"{sorted(cs[row])} -> {sorted(ls[row])}")
        for m in sorted(set(cs[row]) & set(ls[row])):
            for k in ("strict_over", "recovered", "strict_refuse_ok",
                      "loose_over", "loose_refuse_ok"):
                if cs[row][m].get(k) != ls[row][m].get(k):
                    errs.append(f"{row}.sensitivity[{m}].{k}: "
                                f"{cs[row][m].get(k)} -> {ls[row][m].get(k)}")

    # ---- the headline interval. It is a pure function of the composite bounds above, but it
    # is the number a reader quotes, so it is compared directly rather than transitively.
    cg, lg = committed.get("gap_tuned_minus_base") or {}, live.get("gap_tuned_minus_base") or {}
    for name in ("loose", "strict", "answerable_side_only"):
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
# IO layer. Everything below touches disk; nothing above does.
# --------------------------------------------------------------------------------------

PROJ = Path(__file__).resolve().parents[1]
_candidates = [os.environ.get("POLICY_QA_HOME"),
               PROJ / "policy_qa",
               PROJ.parent / "policy_qa", PROJ]
_home = next((Path(c) for c in _candidates if c and Path(c, "src", "common.py").exists()), PROJ)
for _sub in ("eval", "src"):
    if (_home / _sub).exists():
        sys.path.insert(0, str(_home / _sub))
# ...then THIS script's own directory, inserted LAST so it is searched FIRST.
# policy_qa/eval/ holds 2026-07-03 SHADOWS of five eval modules that otherwise win at
# sys.path position 0 (T-071). Placed BEFORE the resolver loop it would be shadowed itself,
# which is the whole trap. Same comment as eval/error_decomposition.py:246.
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPORT_PATH = PROJ / "eval" / "abstention_audit.json"
ROWS = ("qwen2.5-7b-bnb-openbook", "qwen2.5-7b-bnb-closedbook",
        "qwen2.5-7b-raft-openbook", "qwen2.5-7b-raft-closedbook")
BASE_ROW = "qwen2.5-7b-bnb-openbook"
TUNED_ROW = "qwen2.5-7b-raft-openbook"


def _corr_item(score: dict):
    """judge.py:200-201 for a NON-refusing answerable record. None when unscoreable."""
    if not score:
        return None
    if score.get("contradiction"):
        return 0.0
    fs = score.get("fact_scores")
    return statistics.mean(fs) / 2 if fs else None


def build(*, len_multiple: float = STRICT_LEN_MULTIPLE) -> dict:
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
        answers = read_jsonl(ans_p)
        recs, judged, cite_sup, cite_tot, missing_ctx = [], [], 0, 0, []
        for a in answers:
            g = gold.get(a["qid"])
            if not g:
                continue
            answerable = g.get("answerable", True)
            recs.append({"id": a["qid"], "text": a["answer"], "answerable": answerable})
            if a["qid"] in contexts:
                chk = rag.verify_citations(a["answer"], contexts[a["qid"]]["chunks"])
                cite_sup += chk["n_supported"]
                cite_tot += chk["n_total"]
            else:
                missing_ctx.append(a["qid"])
            if answerable and not is_abstention_loose(
                    a["answer"], abstain_sentence=ABSTAIN_SENTENCE):
                v = _corr_item(scores.get((a["qid"], row)))
                if v is not None:
                    judged.append(v)
        if not recs:
            sys.exit(f"[2] row {row}: ZERO records matched gold -- refusing to report.")

        cls = reclassify(recs, abstain_sentence=ABSTAIN_SENTENCE, citation_re=CITATION_RE,
                         max_len_multiple=len_multiple)
        citation = (cite_sup / cite_tot) if cite_tot else 0.0

        # Sanity that costs nothing and would have caught a whole class of mistake: a record
        # judge.py never sent to the judge cannot have a score. If this ever fires, `recovered`
        # is not the unjudged population and every bound below is wrong.
        scored_flagged = sorted(q for q in cls["loose_over_ids"] if (q, row) in scores)

        blocks = {}
        for name, over, refuse_ok in (
                ("loose", cls["n_loose_over"], cls["n_loose_refuse_ok"]),
                ("strict", cls["n_strict_over"], cls["n_strict_refuse_ok"]),
                # The T-084 brief's variant: strict on answerables, loose on unanswerables.
                # Reproduced so the difference from `strict` is visible; NOT adopted.
                ("answerable_side_only", cls["n_strict_over"], cls["n_loose_refuse_ok"])):
            n_rec = 0 if name == "loose" else cls["n_recovered"]
            corr = correctness_bounds(judged, over, n_rec)
            abst = abstention_axis(refuse_ok, cls["refuse_n"], over, cls["ans_n"])
            blocks[name] = {**corr, **composite_bounds(corr, citation, abst),
                            "over": over, "refuse_ok": refuse_ok}

        rows_out[row] = {
            "ans_n": cls["ans_n"], "refuse_n": cls["refuse_n"], "n_records": len(recs),
            "n_loose_over": cls["n_loose_over"], "n_strict_over": cls["n_strict_over"],
            "n_recovered": cls["n_recovered"],
            "n_loose_refuse_ok": cls["n_loose_refuse_ok"],
            "n_strict_refuse_ok": cls["n_strict_refuse_ok"],
            "n_lost_refusal": cls["n_lost_refusal"],
            "recovered_ids": cls["recovered_ids"],
            "lost_refusal_ids": cls["lost_refusal_ids"],
            "strict_over_ids": cls["strict_over_ids"],
            "cite_sup": cite_sup, "cite_tot": cite_tot,
            "n_flagged_with_judge_score": len(scored_flagged),
            "flagged_with_judge_score_ids": scored_flagged,
            "missing_context_ids": sorted(missing_ctx),
            **blocks,
        }
        sens_out[row] = sensitivity(recs, SENSITIVITY_MULTIPLES,
                                    abstain_sentence=ABSTAIN_SENTENCE,
                                    citation_re=CITATION_RE)

    gaps = {}
    if TUNED_ROW in rows_out and BASE_ROW in rows_out:
        for name in ("loose", "strict", "answerable_side_only"):
            gaps[name] = gap_bounds(rows_out[TUNED_ROW][name], rows_out[BASE_ROW][name])

    return {
        "schema": "abstention_audit/1",
        "task": "T-084",
        "loose_rule": "eval/judge.py:80 -- ABSTAIN_SENTENCE.rstrip(' .') substring test",
        "strict_rule": "loose AND no CITATION_RE match AND len(strip) <= "
                       "len_multiple * len(abstain_sentence)",
        "abstain_sentence_len": len(ABSTAIN_SENTENCE),
        "len_multiple": len_multiple,
        "adopted": "strict (applied to BOTH the answerable and unanswerable sides)",
        "inputs": {"gold": Path(gold_p).name, "contexts": Path(ctx_p).name,
                   "scores": Path(sco_p).name,
                   "answers": [f"answers_{r}.jsonl" for r in ROWS]},
        "rows": rows_out,
        "sensitivity": sens_out,
        "gap_tuned_minus_base": {"tuned": TUNED_ROW, "base": BASE_ROW, **gaps},
    }


def _print(live: dict) -> None:
    print(f"abstain sentence: {live['abstain_sentence_len']} chars   "
          f"len_multiple={live['len_multiple']}   adopted: {live['adopted']}")
    for row, r in live["rows"].items():
        print(f"\n{row}  (answerable {r['ans_n']}, unanswerable {r['refuse_n']})")
        print(f"  over-refusals  loose {r['n_loose_over']:3d} -> strict {r['n_strict_over']:3d}"
              f"   recovered {r['n_recovered']}")
        print(f"  refusals ok    loose {r['n_loose_refuse_ok']:3d} -> strict "
              f"{r['n_strict_refuse_ok']:3d}   lost {r['n_lost_refusal']}")
        for name in ("loose", "strict", "answerable_side_only"):
            b = r[name]
            print(f"  {name:<20s} abst={b['abstention']:.4f} cite={b['citation']:.4f} "
                  f"corr [{b['corr_lower']:.4f}, {b['corr_upper']:.4f}] "
                  f"composite [{b['composite_lower']:.4f}, {b['composite_upper']:.4f}]")
    g = live["gap_tuned_minus_base"]
    print(f"\ngap {g['tuned']} minus {g['base']}:")
    for name in ("loose", "strict", "answerable_side_only"):
        if name in g:
            b = g[name]
            print(f"  {name:<20s} [{b['gap_lower']:+.4f}, {b['gap_upper']:+.4f}]  "
                  f"sign {b['sign']}")
    print("\nsensitivity (strict_over / recovered / strict_refuse_ok) by length multiple:")
    for row, s in live["sensitivity"].items():
        cells = "  ".join(f"{m}:{v['strict_over']}/{v['recovered']}/{v['strict_refuse_ok']}"
                          for m, v in s.items())
        print(f"  {row:<28s} {cells}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write the tracked report")
    ap.add_argument("--verify", action="store_true",
                    help="recompute live and compare against the committed report")
    ap.add_argument("--len-multiple", type=float, default=STRICT_LEN_MULTIPLE)
    a = ap.parse_args()

    live = build(len_multiple=a.len_multiple)

    if a.verify:
        if not REPORT_PATH.exists():
            print(f"FAIL: no committed report at {REPORT_PATH}")
            return 1
        try:
            committed = json.loads(REPORT_PATH.read_text())
        except json.JSONDecodeError as exc:
            print(f"FAIL: committed report is not valid JSON: {exc}")
            return 1
        errs = check_audit(committed, live)
        if errs:
            print("ABSTENTION-AUDIT VERIFY FAILED:")
            for e in errs:
                print("  -", e)
            return 1
        b = live["rows"][BASE_ROW]
        print(f"ABSTENTION-AUDIT VERIFY OK -- {len(live['rows'])} rows; {BASE_ROW}: "
              f"{b['n_loose_over']} loose over-refusals -> {b['n_strict_over']} strict, "
              f"{b['n_recovered']} recovered; composite "
              f"[{b['strict']['composite_lower']:.4f}, "
              f"{b['strict']['composite_upper']:.4f}]")
        return 0

    _print(live)
    if a.write:
        REPORT_PATH.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
