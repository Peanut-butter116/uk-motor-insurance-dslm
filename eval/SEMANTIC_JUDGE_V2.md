# UK motor semantic judge — motor-semantic-v2.0

This is an explicit extension of [JUDGE_RUBRIC.md](JUDGE_RUBRIC.md), not a replacement
for historical scores. The frozen prompt lives in `semantic_judge.PROMPT` and its
hash is recorded before execution. Calibration uses eight fictional Acorn policy
examples in `semantic_fixtures.py`; protected labels must never tune the prompt.

## Inputs and blinding

Only a shuffled opaque evaluation ID, question, gold answerability, reference
answer/key facts, gold policy clauses, student-visible evidence, answer, citations
and deterministic refusal decision enter the prompt. Model, family, adaptation,
condition, question IDs, source output paths and existing metrics remain in a
separate private mapping. Shuffle seed: 20261020. Reference clauses resolve by
frozen document and section; all matching chunks are retained without truncation.
Gold answers are grading references, not evidence. Unsupported gold assumptions
must be flagged. Historical inference did not record truncation flags; completeness
is unknown there. New reasoning/final-answer and output-cap flags are preserved.

## Scores and abstention

Overall correctness is 1 (all essential facts), 0.5 (partial with material omission
or limited noncritical error), or 0 (materially wrong). Material coverage/safety
errors, decisive exclusions missed and contradictions always score 0. Fluency and
reference-answer resemblance are insufficient. Evaluate coverage, excess, limits,
conditions, exceptions, eligibility and customer circumstances.

Reference evidence support is 1 (all material claims), 0.5 (some unsupported claims
without contradiction), or 0 (central unsupported/contradicted claim). Report an
unsupported-material-claim flag. Verbatim source spans are validated mechanically;
the span match alone does not verify entailment. Student **citation grounding**
requires a separate claim-to-cited-passage review and remains pending. It is N/A
without student-visible evidence. Reference support can be judged in either arm.

Use the inherited deterministic refusal detector. An answerable refusal scores
correctness 0; an appropriately calibrated unanswerable refusal scores 1. Refusals
have no factual-claim evidence-support score. A detector/gold disagreement needs
review. An invented answer to an unanswerable question scores 0.

Missing/ambiguous/conflicting evidence or unsupported gold assumptions produce
null scores and human-review flags. Borderline and incomplete output are explicit
uncertainty flags. No absent, invalid or uncertain output becomes a zero.

## Compatibility and reliability

The strict v2 JSON also retains legacy `fact_scores` (0/1/2), verbatim answer
`evidence_spans` and `contradiction`. These are aggregated by existing
`judge_votes.merge_task`: median-low per fact, majority contradiction (ties true).
Legacy mean-fact correctness and v2 overall correctness are different metrics.
No historical result or composite is overwritten.

Primary v2 results require three valid distinct vote indices. Overall scores use
median-low; any null vote leaves that metric null. Majority contradiction caps
correctness at zero. Disagreements and any review flag remain visible. Repeated
calls to one model measure within-model consistency, not independent human truth.
One-vote outputs are separately labelled preliminary and never fill primary tables.

The initial 28-item blind review sheet stratifies arms/families/answerability before
scores exist. After actual scoring, supplement it with correct/incorrect and
borderline/disagreement cases. Blank review fields are not completed human reviews.
Human agreement remains null until actual independent labels are supplied; use
`human_agreement.quadratic_weighted_kappa` on ordinal 0/1/2 mapped from 0/.5/1.
Do not trust rankings without independent validation (inherited kappa gate ≈0.6).

Paired differences use common question IDs with observed scores in both arms;
bootstrap exact repeated-question groups jointly, seed 20261020, 4,000 resamples.
Report complete-case coverage. Dependence across different questions sharing policy
sources is not yet modelled. Nine unanswerable motor questions provide imprecise
refusal sensitivity. Missing correctness means differences stay pending.

## Provider and privacy gates

Gemini catalogue discovery sends no research content. `gemini_judge.py` requires
explicit permission for every payload field and the semantic evaluation purpose;
questions-only leakage permission is insufficient. Free-plan/model/quota receipt
must be current, shared production usage is read-only, and evaluation reservations
are separate. There is no paid fallback or automatic retry on HTTP 429. Run only
one evaluation worker; shared usage snapshots cannot guarantee cross-process quota
reservations against an independently changing production pipeline.

Fictional mode accepts only the fixed public fixtures, not arbitrary content marked
fictional. For local fallback, pinned Phi-4-mini 4-bit weights run through native
MLX without transmitting task content. All eight fictional calibration examples
must pass strict schema and expected correctness before protected evaluation.
This small engineering gate is not a substitute for independent human calibration.

## Reproduction

Install `mlx-lm` in a separate native Apple Python environment. Download
`mlx-community/Phi-4-mini-instruct-4bit` revision
`ac1c269cb4222a4e136a3d09edad301056c1f36a` with Hugging Face snapshot metadata intact.
Keep weights and all following outputs outside the public repository.

```bash
python scripts/run_semantic_judge.py prepare --home "$POLICY_QA_HOME" \
  --output "$PRIVATE_JUDGE_RUN" --new-predictions "$VERIFIED_NEW_PREDICTIONS"
python scripts/run_semantic_judge.py calibrate --model-dir "$LOCAL_PHI" \
  --output "$FRESH_PRIVATE_CALIBRATION"
python scripts/run_semantic_judge.py run --directory "$PRIVATE_JUDGE_RUN" \
  --model-dir "$LOCAL_PHI" --calibration "$FRESH_PRIVATE_CALIBRATION/calibration.json" --limit 24
python scripts/run_semantic_judge.py aggregate --directory "$PRIVATE_JUDGE_RUN" \
  --public experiments/2026-10-20/results/semantic_judge_v2
```

The run journal is append-only and resumes valid vote units. Invalid attempts remain
auditable. Aggregation revalidates every vote and rejects duplicates. Private
outputs contain raw attempts, votes, mappings, per-response results and the blind
review sheet. Only aggregate tables and the validation summary are publishable.
