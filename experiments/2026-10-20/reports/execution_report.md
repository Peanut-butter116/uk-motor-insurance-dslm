# Current execution report for the 20 October meeting

Evidence checked on 18 October 2026. This report supersedes the archived initial weekly update. No new presentation materials were produced.

## Kaggle execution

Last authoritative status check: `2026-10-18T14:18:53.767786+00:00`.

| Notebook | Status | Interpretation |
|---|---|---|
| [motor-six-model-compatibility-20261020](https://www.kaggle.com/code/scarletthe0116/motor-six-model-compatibility-20261020) | QUEUED | No execution logs or output artefacts available |
| [motor-six-model-frozen-test-20261020](https://www.kaggle.com/code/scarletthe0116/motor-six-model-frozen-test-20261020) | QUEUED | No execution logs or output artefacts available |
| [uk-insurance-base-eval-qwen3b](https://www.kaggle.com/code/scarletthe0116/uk-insurance-base-eval-qwen3b) | COMPLETE | Historical job ended; inference/training receipts verified separately |
| [uk-insurance-base-eval-ministral3b](https://www.kaggle.com/code/scarletthe0116/uk-insurance-base-eval-ministral3b) | COMPLETE | Historical job ended; inference/training receipts verified separately |
| [uk-insurance-smoke-qwen3b](https://www.kaggle.com/code/scarletthe0116/uk-insurance-smoke-qwen3b) | COMPLETE | Historical job ended; inference/training receipts verified separately |
| [uk-insurance-smoke-ministral3b](https://www.kaggle.com/code/scarletthe0116/uk-insurance-smoke-ministral3b) | COMPLETE | Historical job ended; inference/training receipts verified separately |
| [uk-insurance-full-qwen3b](https://www.kaggle.com/code/scarletthe0116/uk-insurance-full-qwen3b) | COMPLETE | Historical job ended; inference/training receipts verified separately |
| [uk-insurance-full-ministral3b](https://www.kaggle.com/code/scarletthe0116/uk-insurance-full-ministral3b) | COMPLETE | Historical job ended; inference/training receipts verified separately |
| [uk-insurance-sft-eval-qwen3b](https://www.kaggle.com/code/scarletthe0116/uk-insurance-sft-eval-qwen3b) | COMPLETE | Historical job ended; inference/training receipts verified separately |
| [uk-insurance-sft-eval-ministral3b](https://www.kaggle.com/code/scarletthe0116/uk-insurance-sft-eval-ministral3b) | COMPLETE | Historical job ended; inference/training receipts verified separately |

The actual benchmark is a distinct 76-question / 152-task-per-model run using the already approved private evaluation dataset. It is not another compatibility probe. The original queued probe was preserved. Neither queued job has a verified loading result, measured runtime, adapter or evaluation output. No duplicate benchmark and no new SFT job were launched.

Kaggle reports 30 GPU hours allowed, zero used and zero reserved in the current window, with paid scaling disabled. This establishes available quota, not accelerator capacity or a start-time guarantee. No failure message is reported for either queued job.

## Model and prompt checks

All six original checkpoint revisions are retained. Llama and Gemma unauthenticated config requests return HTTP 401. No licence gate was bypassed. The other four configs and pinned tokenizers are accessible. All 152 motor tasks fit the 8,192-token budget, including the 700-token generation allowance, for those four tokenizers. Maximum prompt lengths: Qwen 2,979, DeepSeek 2,955, SmolLM2 3,244 and Ministral 3,612 tokens. CPU checks used Transformers 5.5.3, while the submitted GPU runtime requests 5.19.0. These are tokenizer checks, not GPU compatibility results.

Official Ministral weights are FP8. The actual benchmark uses the model-card dequantisation path to float16 instead of applying an incompatible NF4 override to the FP8 checkpoint. All new candidates request float16 compute/weights, with that documented storage exception. Historical Unsloth NF4 Ministral and historical 3B Qwen remain distinct from official Ministral and new 1.5B Qwen.

Protocol and selection rules were frozen before submission. Protected motor test results are descriptive only: no prompt tuning or model selection. The 20 draft dev questions remain unreviewed and ineligible for definitive selection. The verifier retains raw reasoning and normalises final answers separately. Unfinished DeepSeek reasoning cannot supply an apparent final answer.

## Completed historical evaluation

Re-ran receipt/hash/population verification and CPU scoring from saved outputs, without rerunning model inference. All 1,120 answers across eight historical conditions remain valid. Reproduced aggregate CSV/JSON, receipt summaries and paired uncertainty files are byte-identical to the published files.

Automatic citation resolution, inherited parse adherence, adopted-reading-B abstention and historical timing are complete. Citation resolution remains a metadata check. No new correctness, semantic claim-support, unsupported-claim rate or composite score is available. Historical 21-example SFT results remain labelled separately from the queued six-model run and any future interim-data SFT.

The complete blinded historical batch contains 850 responses needing 2,550 independent votes. Its motor slice contains 484 responses needing 1,452 votes. Paid APIs remain disabled. Approval for a new gold-answer/model-prediction transfer to an external semantic judge has not been established, and the separate production provider quota was not used. Current data-upload approval covers private Kaggle storage, not an unrelated judging provider. No independent votes or human reviews were invented.

## Dataset and SFT readiness

The latest resolved atomic export remains 409 accepted records / 384 distinct question texts. Counts: 135 inherited, 274 generated; 335 answerable, 74 insufficient-evidence cases; six insurers, 27 source documents, 501 evidence sections. All accepted records retain protected-question screen provenance. Human review remains zero and training_ready=false.

The 25 repeated-question clusters all have two distinct contexts and opposite answerability labels. Exact model-visible duplicates: zero. We retained all 409 IDs and require grouped handling and label/evidence review instead of arbitrarily deleting contrastive examples. No automatically accepted record was promoted to human-reviewed ground truth.

No new eligible reviewed production subset was certified. New SFT is blocked by outstanding quality review, unresolved independent dev review/leakage assessment and absence of a reviewed dev-based model choice. Historical 21-example runs do not satisfy a new interim-data experiment.

## Private supervisor CSV

[Training Dataset — CSV](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-interim-20261020-409) — private Kaggle dataset ID 12682948, version 1. Server-side privacy confirmed. All five owner-authenticated per-file downloads match their uploaded SHA-256 hashes. The full-archive endpoint returned HTTP 404 during checking, so file downloads are the verified access route.

Owner local display CSV: `../uk-motor-insurance-dslm-private-data/experiments/2026-10-20/data-display-v2/interim_motor_training_display.csv`.

Rows: 409. SHA-256: `2e368db03e8d0ab3015b56c573154a83c4c5b07496da4fc5a5c8776df5eed4cc`.

Ten requested columns, UTF-8, valid quoted multiline cells, formula-prefix escaping, unique IDs and no exact model-visible duplicates. Canonical lossless CSV/JSONL preserve all provenance separately. Supervisor access remains unverified until usernames are supplied and the owner invites them. No claim of public download access is made.

## Figures

Six figures are available as 300-dpi PNG and SVG with editable text in [figures/](../figures/README.md). The exact public inputs and source SHA-256 hashes accompany them. They show measured historical base/RAG/SFT behaviour, descriptive timing, current dataset composition and automatic refusal/citation failure rates. The six-candidate panel shows execution status, not placeholder performance. Correctness and unsupported-claim bars are deliberately absent because those metrics remain unjudged. Only one timestamped dataset snapshot is used, so no growth trend is invented.

## Repository cleanup and validation

Archived the legacy README, 11 October figures and generator, early meeting notes, the superseded weekly note and existing presentation materials. Retained shared inference, training, retrieval, scoring and testing code in its existing locations. No unique research evidence was deleted. A pre-existing untracked figure script was left untouched.

The rewritten root README covers the current objective, data, real private CSV link, all six checkpoints, protocol, pending semantics, measured historical comparisons, reproduction and roadmap. The current regression suite passes 234 tests, including protected-test selection safeguards, complete task-pair checks, reasoning normalisation and spreadsheet formula-prefix checks.

## Next executable actions

1. When either Kaggle job leaves QUEUED, inspect its terminal status and private logs. Download to a fresh private directory and verify before publishing results.
2. Run the provided verifier only for complete checkpoint populations. Publish new six-model aggregates separately. Repair a failed model only after inspecting its private failure log; never substitute checkpoints.
3. Complete independently approved correctness/claim review and motor dev review without paid API usage or interference with production.
4. Invite the supplied supervisor usernames to the private dataset and verify their access.
5. Certify a reviewed training subset before starting one controlled SFT comparison.
