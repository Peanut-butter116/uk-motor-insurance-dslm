# D1000 training and dual-judge campaign

This opt-in campaign is separate from the unchanged historical 21-seed loader.
It never starts QA generation. The owner explicitly authorised descriptive SFT
using the mixed population: 462 historical verified and 538 unverified records.
This authorisation does not constitute human review or factual certification.

## Executed dataset work

Private production checkpoint 319 is immutable. CPU export notebook
`scarletthe0116/uk-motor-d1000-export`, execution 1, completed the export.
The dedicated private dataset is
<https://www.kaggle.com/datasets/scarletthe0116/uk-motor-qa-d1000-training/versions/1>.
Its independent CPU readback audit completed in
`scarletthe0116/uk-motor-d1000-export-audit`, execution 1. Every CSV field matches
the companion canonical JSONL. All six data-file hashes and the frozen reference
files match. See `export_audit_receipt.json` for safe aggregates.

CSV SHA-256:
`faec491822ce60758d1705fe0f1b71621eaf8dd34f9f9b8168eca001b1acd6b3`.
The UTF-8 CSV quotes every cell; nested fields are JSON. Import IDs as text.
Nulls have empty CSV cells; lossless companion JSONL preserves their types and
all original fields. Values are never rewritten for Excel formula escaping.
Export order is stable by `example_id`. There are 1,000 IDs/model-visible pairs,
975 normalised questions, and zero protected lexical question overlaps. The 25
historical context-contrast pairs are retained, grouped together for splitting.

The existing frozen corpus/reference files were uploaded directly from their
unchanged private source paths to private dataset
`scarletthe0116/uk-motor-frozen-evaluation-v2`, version 1. Temporary upload staging
used symlinks, not new local raw-data copies. `frozen_evaluation_hashes.json`
pins all four files. Their readback was independently checked in cloud.

## Model identity and training changes

`config.json` resolves the requested Qwen3B/Ministral3B to the historical pinned
QLoRA checkpoints, not the separate Qwen1.5B/official Ministral FP16 screening
population. DeepSeek and SmolLM use their exact screened pinned revisions. No
model substitutions occur. Public config access is recorded in
`model_access_metadata.json`; it is not a claim that GPU loading passed.

The original NF4/double quantisation, rank16/alpha32/dropout.05, LR1e-4, two epochs,
batch1, accumulation8 and 2048-token budget are retained. Each model uses its own
pinned tokenizer/chat template. All models enter access/tokenisation preflight;
incompatible models fail separately. Targets and contexts are never truncated.
A common eligible population is frozen after tokenizer/masking checks. Protected
supporting-page overlaps are excluded using the generation pipeline's deterministic
page rule. Every exclusion and actual train/validation ID is private. The frozen
benchmark is used only for exclusions and final evaluation, never selection.

The production target contract uses a separately versioned adaptation of the
historical evidence-only system template: exact production citation headers and
evidence-limited prose abstentions replace contradictory four-part-citation and
fixed-refusal requirements. Original targets remain unchanged. Historical prompts,
seed loading, tokenisation and training files remain byte-identical. A grouped,
seed0 90/10 split is for validation logging only; the fixed final epoch is used.
Never describe a filtered D1000-derived train subset as 1,000 training examples.

Fresh two-step smoke requires finite train/validation loss, language-only LoRA
weights, changed adapter tensors, saved exact adapter weights and matching freshly
reloaded logits/greedy outputs under the original strict tolerances. Only then does
that model continue to full epochs and inference. Runtime versions, GPU, peak
allocation, durations, losses and trainable tensors remain private. A model failure
does not prevent other models running. Existing completed stages resume.

T4 x2 was actually allocated by smoke execution 1. Default execution uses one
isolated process per GPU and queues the other models; no four-process speedup is
claimed. Four-way GPU sharing has not yet been benchmarked. Inference starts in
the completed model's process while another model may still train. Historical
responses are not adopted: all four conditions rerun from exact frozen messages,
greedy decoding and 700 output tokens, with no evidence truncation. Raw DeepSeek
reasoning, final parsing and incomplete outputs are preserved separately.

## Judging and metrics

Two provider workers independently judge identical opaque canonical inputs using
the existing `motor-semantic-v2.2` rubric and indexed span validation. Exact judges:
Groq `openai/gpt-oss-120b` and Gemini `gemini-3.5-flash-lite`. They do not receive
model/condition identities or each other's votes. Self-identifying answers are
deferred for audited blinding, not silently redacted. Each provider must pass the
existing nine fictional correctness anchors before research calls.

This is a separately versioned one-vote-per-provider descriptive analysis. It does
not replace or claim completion of the historical three-vote Gemini primary
protocol. A distinct claim-to-citation extension records exact answer/evidence
spans and support per citation. Citation ID resolution stays deterministic and
separate from semantic reference support and claim-to-citation support. Closed-book
citation support is N/A. Unsupported, ambiguous and missing evidence leaves nulls.

The pinned production transport supplies independent durable request caches,
inherited provider quota reservations and official provider routes. The existing
free-account receipts must remain model-matched and current; no paid fallback is
available. Unknown request outcomes are never replayed blindly. Short and daily
quota waits proceed independently within the session budget. Completion can require
multiple sessions/days; 2,432 evaluations are not assumed to fit one day's quotas.

All 16 result rows retain the 76-question denominator, missing/invalid responses,
parse rate, nine-unanswerable/67-answerable refusal counts, balanced abstention,
citation resolution, each judge's correctness/support and paired agreement. Missing
judge scores are not zeros and not silently averaged. Cohen's kappa, disagreements
and a stratified human-review sample are separate; no human review is invented.
Figures contain accessible markers/line styles and missing values remain missing.

## Storage, starting and restart

Full campaign notebook: `scarletthe0116/uk-motor-d1000-train-eval`.
It requires the existing three Kaggle secrets and an actual editor **Save Version →
Save & Run All** with T4 x2. API saves do not attach secrets; a saved/COMPLETE empty
version is not evidence of training. Read exact aggregate receipts before claims.

The coordinator waits for GPU smoke execution2 to terminate before loading models.
It verifies private dataset v1 by authenticated cloud readback. One coordinator
publishes private `uk-motor-d1000-models-results` stages and verifies every top-level
artifact hash by cloud-memory streaming before more work. This dataset is created
only after secret attachment/preflight, so a URL is not proof it already exists.

Immutable completed optimizer checkpoints have completion markers; they are never
deleted while snapshots are read. Stable adapters/stages, complete response journal
lines, independent judge caches/quotas and progress are versioned together. A restart
restores the newest private stage and checks config/code identity. It skips completed
model runs, resumes optimizer checkpoints and skips successful judge calls. Changed
code requires a separately audited repair, not a forged old smoke receipt. Never
start two production coordinators. Check latest status before editor restart.

Kaggle saved executions run remotely. There is no cross-session GPU scheduler or
claim that a desktop monitor runs while a laptop is closed. GPU training speed,
four-way performance, full training and judging completion require real receipts.

Validation: 301 local regression tests passed, including unchanged historical
loader hashes, grouped split determinism, preserved targets, exact checkpoint
identities, frozen judge schema and missing/degenerate agreement cases. CPU tests
are not substitutes for the running cloud CUDA smoke.
