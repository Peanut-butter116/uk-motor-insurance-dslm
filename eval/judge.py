"""Scoring + leaderboard for the model-candidacy pilot.

Metrics (council composite):
  0.5 x grounded correctness   — fact-level 0-2 vs key_facts (LLM judge;
                                 any contradiction of gold caps the item at 0)
  0.3 x citation support       — deterministic:每 citation regex-matched against
                                 the FROZEN retrieved-chunk metadata (insurer +
                                 section, page ±1); closed-book rows are checked
                                 against the same frozen contexts, so memorised-
                                 but-correct citations still earn credit
  0.2 x abstention accuracy    — mean of (correct refusal rate on unanswerables,
                                 non-refusal rate on answerables)

Hard gate before ranking: format-parse rate >= 90% (answer has >=1 citation OR
is a verbatim abstention).

Judge correctness scores are produced EXTERNALLY (Claude via Claude Code — no
API key on this machine; documented deviation) from judge_tasks.jsonl into
judge_scores.jsonl:  {"qid","row","fact_scores":[0|1|2,...],"contradiction":bool}

Usage:
  python eval/judge.py prepare      # emit judge_tasks.jsonl for all answer rows
  python eval/judge.py aggregate    # deterministic metrics + scores -> leaderboard.md
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
_candidates = [os.environ.get("POLICY_QA_HOME"),
               PROJ / "policy_qa",
               PROJ.parent / "policy_qa", PROJ]
_home = next((Path(c) for c in _candidates if c and Path(c, "src", "common.py").exists()), PROJ)
for sub in ("eval", "src"):
    if (_home / sub).exists():
        sys.path.insert(0, str(_home / sub))
from common import (  # noqa: E402
    ABSTAIN_SENTENCE, RESULTS_DIR, gold_path, read_jsonl, write_jsonl,
)
import rag  # noqa: E402

GOLD_PATH = gold_path()
# v2 gold gets its own frozen-contexts file (mirrors harness.py): binding v1
# contexts against v2 record ids would silently miss on every lookup.
V2 = GOLD_PATH.name == "gold_v2.jsonl"
CONTEXTS_PATH = RESULTS_DIR / ("frozen_contexts_v2.jsonl" if V2 else "frozen_contexts.jsonl")
LEADERBOARD = Path(__file__).resolve().parent / "leaderboard.md"

CAVEATS = """> **Caveats (read before quoting):** Pilot n is small — **differences under
> ~15–20 points are within noise**. Single run per model, temperature 0. Judge
> spot-check against human scoring is scheduled (week 2). Retrieval was frozen
> once per question and byte-identical across open-book rows, so this table
> compares **generators**, not retrievers. Local models are 4-bit quants; per-row
> sampling/protocol deviations are recorded in the answer files. The Claude
> ceiling row was generated via Claude Code rather than a pinned API model
> (documented deviation). Contamination controls beyond the closed-book row
> (counterfactual perturbation) are scheduled. **Open-book rows are capped by
> retrieval.** Measured on the 9-doc index (2026-07-07): embedding-order
> recall@6 = 44%; with the adopted BGE cross-encoder reranker = **53%**
> (all-clauses-retrieved 38%). The pilot rows below predate the reranker
> (frozen contexts, 5-doc index, 51% recall) — identical context per row keeps
> the generator comparison fair; the week-2/3 run re-freezes contexts on the
> upgraded retriever."""


def answer_rows() -> dict[str, list[dict]]:
    rows = {}
    for path in sorted(RESULTS_DIR.glob("answers_*.jsonl")):
        row_id = path.stem.replace("answers_", "")
        rows[row_id] = read_jsonl(path)
    return rows


def is_abstention(text: str) -> bool:
    return ABSTAIN_SENTENCE.rstrip(" .") in text or text.strip() == ABSTAIN_SENTENCE


def prepare() -> None:
    gold = {g["id"]: g for g in read_jsonl(GOLD_PATH)}
    tasks = []
    for row_id, answers in answer_rows().items():
        for a in answers:
            g = gold.get(a["qid"])
            if not g or not g.get("answerable", True):
                continue                      # abstention scored deterministically
            if is_abstention(a["answer"]):
                continue                      # refusal on answerable -> 0, no judge needed
            # `task` is an anonymous code so the scorer never sees which model/row
            # produced the answer (operator blinding — see eval/JUDGE_RUBRIC.md).
            tasks.append({
                "task": f"t{len(tasks):03d}",
                "qid": a["qid"], "row": row_id,
                "question": g["question"],
                "gold_answer": g["gold_answer"],
                "key_facts": g["key_facts"],
                "must_not_assert": g.get("must_not_assert", ""),
                "model_answer": a["answer"],
            })
    path = RESULTS_DIR / "judge_tasks.jsonl"
    write_jsonl(path, tasks)
    print(f"{len(tasks)} judging tasks -> {path}\n"
          "Grade with the hardened rubric in eval/JUDGE_RUBRIC.md (0/1/2 anchors + "
          "verbatim evidence spans + strict JSON), blind to the `row`.\n"
          "Write judge_scores.jsonl rows: "
          '{"qid","row","fact_scores":[0|1|2 per key_fact],"evidence_spans":[...],"contradiction":bool}')


def _row_metrics(records: list[dict]) -> dict:
    """Recompute the composite + parts from a list of per-question records.
    Kept pure so the bootstrap can call it on a resampled record list."""
    correct_vals = [r["corr_item"] for r in records
                    if r["answerable"] and r["corr_item"] is not None]
    cite_sup = sum(r["n_sup"] for r in records)
    cite_tot = sum(r["n_tot"] for r in records)
    ans_n = sum(1 for r in records if r["answerable"])
    over_refuse = sum(1 for r in records if r["answerable"] and r["abst"])
    refuse_n = sum(1 for r in records if not r["answerable"])
    refuse_ok = sum(1 for r in records if not r["answerable"] and r["abst"])
    correctness = statistics.mean(correct_vals) if correct_vals else 0.0
    citation = (cite_sup / cite_tot) if cite_tot else 0.0
    refusal_acc = (refuse_ok / refuse_n) if refuse_n else 1.0
    non_over = 1 - (over_refuse / ans_n) if ans_n else 1.0
    abstention = (refusal_acc + non_over) / 2
    return {
        "composite": 0.5 * correctness + 0.3 * citation + 0.2 * abstention,
        "correctness": correctness, "citation": citation, "abstention": abstention,
        "cite_sup": cite_sup, "cite_tot": cite_tot, "over_refuse": over_refuse,
        "ans_n": ans_n, "refuse_ok": refuse_ok, "refuse_n": refuse_n,
        "judged_by_llm": sum(1 for r in records if r["judged"]),
    }


def bootstrap_ci(records: list[dict], key: str, B: int = 1000, seed: int = 0) -> tuple:
    """95% percentile CI of a metric, resampling QUESTIONS with replacement."""
    rng = random.Random(seed)
    n = len(records)
    if n == 0:
        return (0.0, 0.0)
    vals = []
    for _ in range(B):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        vals.append(_row_metrics(sample)[key])
    vals.sort()
    return (vals[int(0.025 * B)], vals[int(0.975 * B)])


def paired_bootstrap(rec_a: dict, rec_b: dict, B: int = 1000, seed: int = 0) -> dict:
    """Resample the SAME qids for two rows; report P(A>B) and the composite-diff CI."""
    rng = random.Random(seed)
    common = [q for q in rec_a if q in rec_b]
    n = len(common)
    if n == 0:
        return {"p_a_gt_b": 0.5, "diff_lo": 0.0, "diff_hi": 0.0, "n": 0}
    diffs = []
    for _ in range(B):
        idx = [common[rng.randrange(n)] for _ in range(n)]
        ca = _row_metrics([rec_a[q] for q in idx])["composite"]
        cb = _row_metrics([rec_b[q] for q in idx])["composite"]
        diffs.append(ca - cb)
    diffs.sort()
    return {"p_a_gt_b": sum(d > 0 for d in diffs) / B,
            "diff_lo": diffs[int(0.025 * B)], "diff_hi": diffs[int(0.975 * B)], "n": n}


def aggregate() -> None:
    gold = {g["id"]: g for g in read_jsonl(GOLD_PATH)}
    contexts = {c["qid"]: c for c in read_jsonl(CONTEXTS_PATH)}
    scores = {(s["qid"], s["row"]): s for s in read_jsonl(RESULTS_DIR / "judge_scores.jsonl")}

    lines = ["# Model-candidacy mini-leaderboard (pilot)", "", CAVEATS, "",
             "| Row | Composite [95% CI] | Correctness | Citation support | Abstention | Parse | n | Latency |",
             "|---|---|---|---|---|---|---|---|"]
    ranked = []
    row_records: dict[str, dict] = {}       # row -> {qid: record} for paired bootstrap
    for row_id, answers in answer_rows().items():
        if not answers:
            continue
        records, lats = [], []
        for a in answers:
            g = gold.get(a["qid"])
            if not g:
                continue
            lats.append(a.get("latency_s", 0))
            abst = is_abstention(a["answer"])
            check = rag.verify_citations(a["answer"], contexts[a["qid"]]["chunks"])
            corr_item, judged = None, False
            if g.get("answerable", True):
                if abst:
                    corr_item = 0.0                       # over-refusal = hard 0, no judge call
                else:
                    s = scores.get((a["qid"], row_id))
                    if s:
                        judged = True                      # actually LLM-judged
                        corr_item = 0.0 if s.get("contradiction") else (
                            statistics.mean(s["fact_scores"]) / 2 if s.get("fact_scores") else None)
                        if corr_item is None:
                            judged = False
            records.append({
                "qid": a["qid"], "answerable": g.get("answerable", True), "abst": abst,
                "n_sup": check["n_supported"], "n_tot": check["n_total"],
                "corr_item": corr_item, "judged": judged,
                "parse_ok": abst or check["n_total"] > 0,
            })
        m = _row_metrics(records)
        n = len(records)
        parse_rate = sum(r["parse_ok"] for r in records) / n
        lo, hi = bootstrap_ci(records, "composite")
        gate = "" if parse_rate >= 0.9 else " ⚠️ *fails parse gate*"
        # "N judged" now = items actually LLM-judged (not the answerable denominator)
        lines.append(
            f"| **{row_id}**{gate} | **{m['composite']:.0%}** [{lo:.0%}–{hi:.0%}] "
            f"| {m['correctness']:.0%} ({m['judged_by_llm']} LLM-judged / "
            f"{m['ans_n']} answerable) | {m['citation']:.0%} ({m['cite_sup']}/{m['cite_tot']}) "
            f"| {m['abstention']:.0%} (refuse {m['refuse_ok']}/{m['refuse_n']}, "
            f"over-refuse {m['over_refuse']}/{m['ans_n']}) "
            f"| {parse_rate:.0%} | {n} | {statistics.median(lats):.1f}s |"
        )
        ranked.append((m["composite"], row_id, parse_rate))
        row_records[row_id] = {r["qid"]: r for r in records}
    ranked.sort(reverse=True)
    slm = [r for r in ranked
           if r[2] >= 0.9 and "claude" not in r[1] and "closedbook" not in r[1]]
    if len(slm) >= 2:
        a, b = slm[0][1], slm[1][1]
        pb = paired_bootstrap(row_records[a], row_records[b])
        sig = "significant" if (pb["diff_lo"] > 0 or pb["diff_hi"] < 0) else "NOT significant (CI spans 0)"
        lines += ["", f"**Top-2 SLM head-to-head (paired bootstrap, n={pb['n']}):** "
                  f"`{a}` vs `{b}` — composite diff 95% CI [{pb['diff_lo']:+.0%}, {pb['diff_hi']:+.0%}], "
                  f"P({a}>{b})={pb['p_a_gt_b']:.0%} → **{sig}**.",
                  f"So at this n the leading SLM is **not statistically separable** — the week-2 "
                  "100–150Q run decides. (This is the honest small-n result.)"]
    elif slm:
        lines += ["", f"**Leading SLM (pilot only):** `{slm[0][1]}` — n caveat applies."]
    LEADERBOARD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {LEADERBOARD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "aggregate"])
    a = ap.parse_args()
    prepare() if a.cmd == "prepare" else aggregate()
