# LLM-judge rubric (hardened) — how to score correctness

*Use this exact rubric whenever grading answers (Council-B best practice). It
makes scores auditable, mechanically parseable, and bias-controlled. The tasks
in `judge_tasks.jsonl` carry an anonymous `task` code, not the model name —
**keep the model identity hidden from whoever/whatever scores** (operator
blinding: the intern who runs the judge should not reveal the row, and should
shuffle order).*

## The prompt to give the judge

> You are a strict, style-blind evaluator. For each task you are given a
> `question`, a list of `key_facts` (the grading rubric), an optional
> `gold_answer` reference, and a `model_answer`. Score the **model_answer**
> against the **key_facts only** — ignore length, tone, formatting, citations,
> and which model wrote it.
>
> For **each** key_fact, output an integer score with these **anchors**:
> - **2** — the fact is stated fully and correctly.
> - **1** — the fact is partially or vaguely conveyed.
> - **0** — the fact is absent, **or the answer contradicts it**.
>
> For each score, also return a **verbatim evidence span** copied from the
> model_answer that justifies the score (or `""` if absent). This kills
> hallucinated grading and makes every score checkable.
>
> Set `contradiction: true` if the answer materially contradicts the gold /
> a key_fact (e.g. says "covered" when it isn't, wrong amount, wrong time-bar).
>
> Output **strict JSON only**, no prose:
> `{"task": "<code>", "fact_scores": [0|1|2, ...], "evidence_spans": ["...", ...], "contradiction": true|false}`

## Bias controls (why this design)
- **Reference-guided + fact-decomposed** (score per key_fact, contradiction caps
  at 0) — the design MT-Bench found most human-aligned; suppresses verbosity /
  impression scoring.
- **Family-disjoint judge** — the judge (Claude) must not be the candidate's
  family (Qwen); the Claude ceiling row is scored but reported **unranked**.
- **Operator blinding** — one intern strips the model identity + shuffles order
  before the other runs the judge; the `task` code (not `row`) is what the judge
  sees.
- **Pointwise** for the composite; add **pairwise A-vs-B** (both orders) only as a
  tiebreaker between the top-2 SLMs, never involving the Claude row.
- **At scale (week 2):** ≥3 judge runs per item (take the median), a second
  non-Claude judge on a ~25–30% audit subset (log inter-judge κ), and the
  human-agreement κ gate (`eval/human_agreement.py`, trust only if κ ≥ ~0.6).
  This needs an **API/batch judge** — the single biggest enabler.
