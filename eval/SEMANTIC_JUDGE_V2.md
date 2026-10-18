# UK motor semantic judge — motor-semantic-v2.2

This version extends [JUDGE_RUBRIC.md](JUDGE_RUBRIC.md). Historical predictions,
reference answers, benchmark hashes and published legacy metrics are preserved.
Overall correctness below is a separate metric from the legacy mean key-fact score.

## Frozen rubric and inputs

Correctness is **1** (all essential facts correct), **0.5** (useful but incomplete
or with a limited noncritical error), or **0** (materially wrong). Invented cover,
material safety/coverage errors, decisive exclusions missed and material
contradictions score zero. Check coverage, exclusions, excesses, limits, conditions,
exceptions, eligibility and customer circumstances. Fluency and answer similarity
are insufficient.

Reference support is **1** (all material claims supported), **0.5** (some unsupported
claims without contradiction), or **0** (central unsupported/contradicted claim).
Record `unsupported_material_claim`. Positive support requires verifiable source
spans. An unsupported claim may have no supporting span; it must then carry an
unsupported/contradiction flag. Refusals have no factual-claim support score.

Use the inherited deterministic refusal detector. An answerable refusal scores
correctness zero. An appropriately calibrated unanswerable refusal may score one;
an invented answer scores zero. Ambiguous/missing/conflicting evidence or an
unsupported gold assumption yields null scores and human-review flags. Incomplete
outputs and borderline decisions are explicitly flagged. Invalid or absent votes
never become zeros.

The private payload contains only opaque ID, question, gold answerability,
reference answer/key facts, policy evidence, student-visible evidence,
answer/citations and engineering refusal/completeness flags. A separate private
mapping retains model, revision, condition, experiment and question ID. Shuffle
seed: **20261020**. Student names and performance scores are not supplied. Potential
self-identification in an answer blocks preparation pending an audited blind copy.

Gold cited document/sections supply reference evidence for answerable questions.
Verified unanswerable questions deliberately lack a positive gold citation: their
reference package retains the **frozen policy context**, labelled as context rather
than a positive gold citation, alongside the frozen answerability/reference answer.
This package repair is versioned `frozen-policy-scope-v2.2.1`. It does not change
benchmark labels, questions, references or student outputs. Votes were adopted
only for identical canonical inputs and retained source provenance.

Reference evidence and student-visible evidence are distinct. Reference support
can be assessed in either inference arm. **Student citation grounding is N/A
without student evidence and remains a separate pending claim-to-citation audit
for RAG.** Source ID existence or citation resolution alone is not entailment.

## Strict indexed transport

`judge_span_ids.py` supplies lossless ordered text segments: `A0` for answer spans,
`R0S0` for reference spans. Their concatenation exactly reconstructs the original
text. The judge selects IDs/ranges and still decides whether the segment supports
the claim. Local code resolves them to exact original substrings. Invented IDs,
reversed ranges and ranges across source chunks are rejected. No quotation is
fabricated or paraphrased by post-processing.

`indexed-pointwise-v3` uses a shared JSON schema for a batch of at most ten
**different question groups**. Exact IDs, uniqueness, fact arity, types, scores,
verbatim resolved spans, refusal rules and uncertainty constraints are validated
per item. An item is one vote, not an entire batch. The same response's three
votes always occur in three separate stateless API requests, seeds 20261021–23;
no other votes or previous decisions enter any prompt.

Nested fixed array grammars exceeded provider schema limits. Shared schemas and
post-generation arity validation preserve strictness without truncating inputs.
Each request reserves a conservative UTF-8-byte/token upper bound, at most 175,000,
and up to 32,768 output tokens including thinking. The actual model limit is
checked through the authenticated catalogue. No evidence truncation is allowed.
See Google's [structured output documentation](https://ai.google.dev/gemini-api/docs/structured-output)
and [thinking documentation](https://ai.google.dev/gemini-api/docs/generate-content/thinking).

## Compatibility and aggregation

| Transport field | Canonical / historical mapping |
|---|---|
| `answer_span_ids` | Resolve into legacy verbatim `evidence_spans`, one per key fact |
| `reference_span_ids` | Resolve into canonical `reference_spans` |
| `fact_scores` | Legacy integers 0/1/2, unchanged |
| `contradiction` | Legacy material-contradiction cap, unchanged |
| `correctness` | New overall 0/.5/1 rubric; **not** mean fact correctness |
| `evidence_support` | Reference semantic support; **not** citation resolution |
| `evaluation_id` | Opaque task ID; join private mapping for historical `(qid, row)` |

Primary results require three valid distinct vote indices. The existing
`judge_votes.merge_task` supplies median-low fact scores and majority contradiction
(ties true). Overall scores use median-low; any null vote leaves that metric null.
Majority contradiction caps correctness at zero. A full merged correctness score
with a missing/partial merged required fact is inconsistent: correctness stays
null and needs human review. Disagreements and low confidence remain visible.
One-vote tables use **vote 1 only** and are explicitly preliminary. An explicit
per-vote model plan is required before combining different judge identities;
the current research run uses Gemini 3.5 Flash-Lite only.

These are repeated runs of one model, not three independent human experts.
Within-model ordinal kappa and exact agreement measure consistency, not truth.
Human agreement remains null until actual independent labels exist. The inherited
human-validation trust gate is approximately weighted kappa ≥0.6, with sample size
and coverage considered. No ranking is validated without independent review.

Paired differences use common question IDs, including separate fully-correct
percentage-point differences. Bootstrap exact repeated-question groups jointly,
seed 20261020, 4,000 resamples. A second sensitivity resamples connected components
sharing any reference source document. Fewer than two independent groups yields
**no interval**, not a zero-width interval. Four source components and only nine
unanswerable questions limit inference. Complete-case plots use identical observed
questions across all eight historical arms; their population is explicitly a
selected subset, not a full-test performance estimate.

## Providers, permissions and quota

The owner explicitly authorised the full motor semantic payload through the
**official Gemini API, free quota only**, after the original questions-only
leakage-check receipt was found insufficient. Approval is recorded privately in
`gemini_semantic_permission.json`. No production permission or usage file was
modified. Current judge: **`gemini-3.5-flash-lite`**, catalogue version
`3.5-flash-lite-07-2026`, high thinking. The authenticated AI Studio free-tier model
row showed 15 RPM / 250K input TPM / 500 RPD on 18 October.

The adapter requires exact purpose/field approval, a current model-matched free
receipt and a model in the authenticated catalogue. Production usage is read
only; evaluation reservations use a separate ledger. It reserves 50 daily requests,
one request/minute and 50K input tokens/minute for other work. Completed requests
replace conservative TPM reservations with reported input tokens; all token
categories remain recorded for accounting. HTTP 429 stops without a paid fallback.
Only one worker may use a given evaluation ledger.

Eight scoreable fictional Acorn cases must match the rubric exactly; a ninth
missing-reference guard must return null or fail strict validation. Only invalid
outputs get one predefined retry; valid-but-wrong calibration scores are not
retried. All calibration attempts are retained. Earlier quoted-span/local Phi
trials and failed secondary-model trials are not mixed into research scores.

## Reproduction

Prepare outside the public repository, verify model catalogue/free quota/permission,
then calibrate before any research calls. No paid SDK is required: the adapter
uses the official REST endpoint and Python standard library.

```bash
python scripts/run_semantic_judge.py prepare --home "$POLICY_QA_HOME" \
  --output "$PRIVATE_JUDGE_RUN" --new-predictions "$VERIFIED_NEW_PREDICTIONS"
python scripts/run_semantic_judge.py evidence-groups --home "$POLICY_QA_HOME" \
  --directory "$PRIVATE_JUDGE_RUN"
python scripts/calibrate_gemini_judge.py --directory "$PRIVATE_JUDGE_RUN" \
  --free-receipt "$FREE_QUOTA_RECEIPT" --shared-usage "$READ_ONLY_PRODUCTION_USAGE" \
  --catalogue "$AUTHENTICATED_CATALOGUE" --env-file "$PRIVATE_ENV_FILE"
python scripts/run_gemini_batch_judge.py --directory "$PRIVATE_JUDGE_RUN" \
  --free-receipt "$FREE_QUOTA_RECEIPT" --shared-usage "$READ_ONLY_PRODUCTION_USAGE" \
  --catalogue "$AUTHENTICATED_CATALOGUE" --env-file "$PRIVATE_ENV_FILE" --include-new
python scripts/run_semantic_judge.py aggregate --directory "$PRIVATE_JUDGE_RUN" \
  --public experiments/2026-10-20/results/semantic_judge_v2
python scripts/plot_semantic_results.py --results experiments/2026-10-20/results/semantic_judge_v2 \
  --output experiments/2026-10-20/figures/semantic_judge_v2
```

The manifest contains prompt/rubric/transport versions and hashes, dataset/corpus
hashes, model IDs, seeds, actual attempts, token receipts, costs and review coverage.
Raw attempts, per-question results, model mapping and the blind human sheet stay
private. Public output contains only approved aggregate tables and figures.
