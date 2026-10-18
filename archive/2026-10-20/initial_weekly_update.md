# Supervisor update for 20 October 2026

Prepared from evidence verified on 18 October. Main objective: test whether SFT improves UK motor policy Q&A independently and with fixed retrieval.

## Completed work

The immutable private export contains 409 accepted records, 384 distinct questions, six insurers, 27 documents and 501 sections. Origins: 135 inherited, 274 generated. Answerability: 335 answerable, 74 insufficient-evidence records. Validation provenance: 325 GPT two-stage and 84 Gemini two-stage, all with passed protected-question screening. Human review is zero. Source training readiness is false and 25 rows repeat question text. Production reported paused_provider_quota; no production state was modified.

Verified 1,120 answers across eight historical conditions, including newly recovered Qwen SFT inference. Both SFT runs used 21 seed examples, four optimiser steps and two epochs, with validated saved-adapter reloads. Frozen contexts, launch nonces, checkpoint configuration, populations and adapter hashes passed checks. New dataset snapshot did not supply those historical experiments.

Created private paired-result CSV, public full/motor aggregate tables, exploratory paired bootstrap intervals and a blinded 850-response judging package. The regression suite passed 224 tests before this continuation. See the current execution report for later checks.

Prepared 20 new private development drafts from eight groups, with 16 answerable and four unanswerable questions. The insurer is absent from training and selected source groups are absent from inherited gold citations. Independent review and semantic independence remain pending; legal-expenses-heavy coverage needs expansion. No quality model selection used these drafts or the protected test.

## Observed results

On 76 motor questions, Ministral with RAG changes from citation resolution 0.8125/base to 0.9254/SFT; balanced abstention 0.6036 to 0.6592; parse 75.0% to 76.3%.

Qwen with RAG changes from citation resolution 0.8889/base to 0.7857/SFT; balanced abstention 0.5000 to 0.4925; parse 69.7% to 88.2%. The exploratory paired bootstrap for Qwen parse improvement is +18.4 percentage points (95% interval +7.9 to +28.9). It measures formatting behaviour, not correctness. SFT + RAG Qwen refuses zero of nine unanswerable motor questions and one of 67 answerable questions.

No new correctness, claim-support, unsupported-claim or composite score is available. Do not declare a winner. Full 140-question historical results remain separate from the 76-question motor slice. Historical 7B baselines are retained in README and are not directly pooled into the new six-model screening protocol.

## Six-model and compute status

Pinned official revisions for all requested candidates. Llama and Gemma are gated, DeepSeek is Qwen-derived and official Ministral differs from the inherited Unsloth quantisation. Submitted a private code-only T4 probe using fictional evidence. Submission is not completion. No new restricted snapshot upload and no new 409-record SFT run occurred. Existing GPU outputs establish historical T4 operation, not current free quota availability.

## Discussion with Rama and Tushar

Agree independent judging/review ownership, additional dev topics and source groups, and GPU allocation. Prioritise finishing existing correctness assessment and reviewing dev data before new quality screening. Reach 2,000 accepted examples (3,000 stretch), freeze on 18 November and deliver final research/model/repository by 18 December.
