"""Human-agreement spot-check for the LLM judge (Council-B validity gate).

The composite trusts an LLM (Claude) to score correctness. This measures whether
that judge agrees with a HUMAN on a small blind sample, via chance-corrected
Cohen's kappa — the number the supervisor will ask for. Report kappa WITH its n
and a bootstrap CI (NOT raw %: raw agreement overstates chance-corrected
agreement by ~30-40pp). Decision rule: trust the judge's ranking only if
quadratic-weighted kappa >= ~0.6 (Landis-Koch "substantial"); below that,
downgrade judge scores to a directional signal and lean on the deterministic
citation + abstention axes.

Workflow:
  python eval/human_agreement.py sheet      # -> human_label_sheet.jsonl (blind; ~20 items)
  #  ... a human fills "human_fact_scores" (0/1/2 per key_fact), blind to the judge ...
  python eval/human_agreement.py kappa       # -> quadratic-weighted kappa + 95% CI
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
from common import RESULTS_DIR, read_jsonl, write_jsonl  # noqa: E402

TASKS = RESULTS_DIR / "judge_tasks.jsonl"
SCORES = RESULTS_DIR / "judge_scores.jsonl"
SHEET = RESULTS_DIR / "human_label_sheet.jsonl"
N_ITEMS = 20


def make_sheet() -> None:
    """Emit a BLIND labelling sheet: question + key_facts + model answer, NO judge
    scores shown (operator blinding). One intern fills human_fact_scores; the
    OTHER intern (or the supervisor) should not have authored the answer."""
    tasks = read_jsonl(TASKS)
    if not tasks:
        sys.exit(f"No {TASKS} — run `python eval/judge.py prepare` first.")
    rng = random.Random(0)
    sample = rng.sample(tasks, min(N_ITEMS, len(tasks)))
    sheet = []
    for t in sample:
        sheet.append({
            "qid": t["qid"], "row": t["row"],
            "question": t["question"],
            "key_facts": t["key_facts"],
            "model_answer": t["model_answer"],
            "human_fact_scores": [None] * len(t["key_facts"]),   # FILL: 0/1/2 per fact
            "human_contradiction": None,                          # FILL: true/false
            "_instructions": "Score each key_fact 0=absent/wrong, 1=partial, 2=fully correct, "
                             "reading ONLY the model_answer. Set human_contradiction if the "
                             "answer contradicts a key_fact. Do NOT look at the judge scores.",
        })
    write_jsonl(SHEET, sheet)
    print(f"{len(sheet)} blind items -> {SHEET}\nFill human_fact_scores (0/1/2 per fact), "
          "blind to the judge, then run `python eval/human_agreement.py kappa`.")


def _confusion(pairs: list[tuple[int, int]], k: int = 3) -> list[list[int]]:
    m = [[0] * k for _ in range(k)]
    for a, b in pairs:
        m[a][b] += 1
    return m


def quadratic_weighted_kappa(pairs: list[tuple[int, int]], k: int = 3) -> float:
    """Cohen's kappa with quadratic weights for ordinal 0..k-1 labels."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    O = _confusion(pairs, k)
    r = [sum(O[i]) for i in range(k)]                     # rater A marginals
    c = [sum(O[i][j] for i in range(k)) for j in range(k)]  # rater B marginals
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            e = r[i] * c[j] / n
            num += w * O[i][j]
            den += w * e
    return 1 - num / den if den else 1.0


def compute_kappa() -> None:
    sheet = read_jsonl(SHEET)
    scores = {(s["qid"], s["row"]): s for s in read_jsonl(SCORES)}
    pairs: list[tuple[int, int]] = []            # (human, judge) per fact
    used = 0
    for item in sheet:
        h = item.get("human_fact_scores")
        s = scores.get((item["qid"], item["row"]))
        if not h or None in h or not s or not s.get("fact_scores"):
            continue
        judge = s["fact_scores"]
        if len(judge) != len(h):
            continue
        used += 1
        for hf, jf in zip(h, judge):
            pairs.append((int(hf), int(jf)))
    if not pairs:
        sys.exit("No filled items yet. Fill human_fact_scores in "
                 f"{SHEET} and re-run.")
    kappa = quadratic_weighted_kappa(pairs)
    # bootstrap CI over the fact-pairs
    rng = random.Random(0)
    boot = []
    for _ in range(2000):
        s = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        boot.append(quadratic_weighted_kappa(s))
    boot = [b for b in boot if b == b]
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    band = ("near-perfect" if kappa >= 0.8 else "substantial" if kappa >= 0.6
            else "moderate" if kappa >= 0.4 else "fair/poor")
    verdict = "TRUST the judge's ranking" if kappa >= 0.6 else \
        "DOWNGRADE judge to a directional signal; lean on the deterministic axes"
    print(f"Quadratic-weighted Cohen's kappa = {kappa:.2f}  [95% CI {lo:.2f}–{hi:.2f}]")
    print(f"  n = {used} items ({len(pairs)} fact-level judgements), Landis-Koch: {band}")
    print(f"  Decision (>=0.60 gate): {verdict}")
    print("  Note: kappa on ~20 items is fragile (one/two disagreements swing it) — "
          "treat as indicative; re-confirm on the week-2 150-item run.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "kappa"])
    a = ap.parse_args()
    make_sheet() if a.cmd == "sheet" else compute_kappa()
