# Independent semantic evaluation — 18 October 2026

Citation resolution checks whether a citation matches a supplied passage. It cannot establish whether that passage supports the claim or whether the answer is correct. The [versioned evaluator](../../../eval/SEMANTIC_JUDGE_V2.md) uses **`motor-semantic-v2.2`**, preserving the inherited key-fact rubric and adding overall correctness (0, 0.5, 1), reference-evidence support, material errors and explicit uncertainty. Material coverage/safety errors score zero; missing or ambiguous references leave scores null.

**Judge:** official Gemini API, **`gemini-3.5-flash-lite`**, catalogue version `3.5-flash-lite-07-2026`, high thinking. The owner explicitly authorised protected questions, reference answers/key facts, policy wording and model answers/citations for this purpose. Actual free-tier limits were verified: 15 RPM / 250K input TPM / 500 RPD. Paid calls are disabled. Production is read only and retains a quota reserve. The fictional calibration gate passed before protected judging. Local Phi-4-mini and secondary Gemini 3.1 trials failed calibration and contribute no research labels.

Inputs are blinded, shuffled with seed 20261020, and omit student model names, Base/SFT/RAG labels and existing scores. They retain the question, answerability, gold reference/key facts, reference policy evidence, separately identified student-visible evidence, model answer/citations and deterministic refusal decision. Exact source spans are validated locally. No-RAG answers may be assessed against gold references, but citation grounding remains N/A. **Reference support is not claim-to-citation entailment; the latter remains pending for RAG.**

Primary results require three valid repeated stateless requests, predefined median aggregation and contradiction caps. These measure within-model consistency, not independent human truth. **535 valid votes** are saved: **502 responses have a valid first vote, 475 have a non-null first-vote correctness score, and 0 have completed all three votes**. Persistent provider HTTP 503 overload interrupted execution; no paid fallback was used. Human review completed: **0** of a prepared blinded 28-case sample. Human agreement is unmeasured. Repeat-vote agreement on 26 scoreable pairs is 80.8%, weighted kappa 0.778; this is not validation accuracy.

**Preliminary vote-1 results below are incomplete and unvalidated.** Each arm targets the same 76 questions (67 answerable, nine unanswerable); the displayed denominators are the available scoreable subset and differ between arms. Support excludes refusals and uncertain support labels. Do not infer effects by subtracting these unpaired means.

| Model | Condition | Correctness (scored n) | Reference support (scored n) |
|---|---|---:|---:|
| Ministral-3B | Base | 0.172 (64) | 0.262 (21) |
| Ministral-3B | Base + RAG | 0.558 (60) | 0.808 (52) |
| Ministral-3B | SFT | 0.152 (56) | 0.118 (34) |
| Ministral-3B | SFT + RAG | 0.455 (56) | 0.672 (61) |
| Qwen2.5-3B | Base | 0.087 (63) | 0.091 (55) |
| Qwen2.5-3B | Base + RAG | 0.364 (55) | 0.595 (58) |
| Qwen2.5-3B | SFT | 0.065 (69) | 0.077 (65) |
| Qwen2.5-3B | SFT + RAG | 0.288 (52) | 0.491 (55) |

Paired vote-1 comparisons use common scored question IDs and 4,000 question-group bootstrap resamples; missing scores are excluded, never zero. These selected complete-case intervals condition on one judge's labels, have no multiplicity correction, and do not represent human-validated full-test effects. Source-component sensitivity is reported separately (only three or four independent components).

| Model | SFT comparison | Scored pairs | Mean difference [95% paired CI] |
|---|---|---:|---:|
| Qwen2.5-3B | Without RAG | 57 | +0.000 [-0.061, +0.070] |
| Qwen2.5-3B | With RAG | 41 | +0.024 [-0.085, +0.122] |
| Ministral-3B | Without RAG | 50 | -0.030 [-0.110, +0.040] |
| Ministral-3B | With RAG | 47 | -0.149 [-0.277, -0.032] |

The available pairs do **not establish a consistent SFT accuracy improvement**. Both completion of the three-vote protocol and independent review remain necessary. No winner or new composite is reported. The original 484 non-abstaining answerable historical responses require 1,452 votes; this expanded historical protocol covers **all 608 responses / 1,824 votes**, including refusals and unanswerable items. The full 140-question historical benchmark stays separate.

[Primary correctness](../results/semantic_judge_v2/correctness_summary.csv) · [Primary grounding](../results/semantic_judge_v2/grounding_summary.csv) · [Preliminary vote-1 summary](../results/semantic_judge_v2/preliminary_single_vote_summary.csv) · [Paired vote-1 comparisons](../results/semantic_judge_v2/preliminary_single_vote_comparison.csv) · [Coverage/reliability](../results/semantic_judge_v2/judge_validation_summary.json) · [Execution report](../reports/semantic_judge_execution.md).

[Five semantic PNG/SVG figures](../figures/semantic_judge_v2/README.md) use the **same 13 scoreable questions across all eight arms**, a deliberately labelled matched preliminary subset (11 answerable, two unanswerable). Their means differ from the partial-arm table above and must not be described as full 76-question results. Exact inputs and hashes accompany the figures.


## Verified inference and private artefacts

Historical 1,120 predictions were reverified without rerunning them. The 76-question motor population gives 608 historical responses; SFT used 21 training records. The current six-model benchmark and probe are COMPLETE. Qwen1.5B, DeepSeek1.5B and SmolLM2 each completed 152 verified paired responses; the failed-only pinned Ministral repair completed another 152. Llama/Gemma remain gated. No interim-data SFT began.

Restricted inputs, raw attempts, votes, disagreements, per-question differences, human review sample and manifest remain in the private sibling experiment directory `llm_judge_v22_policy_scope/`. The later official Ministral package is separate, `llm_judge_ministral_repair_v22/`; it has 152 prepared responses and no votes. The active 1,064-response package requires 3,192 votes; the later extension requires another 456. Never join opaque IDs across runs without their own mappings.

The first transport/reference package was superseded by `frozen-policy-scope-v2.2.1`: verified unanswerable questions retain frozen policy context rather than inventing positive gold citations. Exactly 117 votes were adopted only where the canonical task and private identity mapping were identical. Earlier attempts, failed fictional calibrations and all provider failure receipts remain private.

## Blocker and continuation

Official Gemini returned repeated HTTP 503 UNAVAILABLE responses stating temporary high demand, including after a backoff and smaller batches. These are provider availability failures, not a permission denial or evidence of exhausted free quota. Completed votes are retained; each response/vote is resumable with at most two invalid-output attempts. Paid fallback is disabled. Stop on HTTP 429 or the conservative free-quota guard; retain the production reserve. Recheck availability before resuming the single-worker command in the versioned protocol. The new-model judge package is prepared but remains unexecuted while historical judging is incomplete.
