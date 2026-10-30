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
Execution5 is confirmed RUNNING with verified repaired code hash
`5dfd45d97e5f2b3b6778bea7bf1076f7c88be576de3395c20285907b7c93c27b`.
The agent started it through the owner's loaded editor with existing secrets.
Execution4 was safely stopped at its read-only guard before any model loading;
its receipt confirms zero models/responses/judge calls. An independent private
CPU storage probe proved `.bin` retention/readback before execution5 started.
The current private models/results version2 is an empty metadata bootstrap,
not evidence of successful training or saved adapters.
It requires the existing three Kaggle secrets and an actual editor **Save Version →
Save & Run All** with T4 x2. API saves do not attach secrets; a saved/COMPLETE empty
version is not evidence of training. Read exact aggregate receipts before claims.

The coordinator waits for GPU smoke execution2 to terminate before loading models.
It verifies private dataset v1 by authenticated cloud readback. One coordinator
publishes private `uk-motor-d1000-models-results` stages as an opaque `artifacts.bin`
archive (Kaggle otherwise unpacks `.tar.gz` uploads) and verifies every top-level
artifact hash by cloud-memory streaming before more work. The bootstrap alone
does not establish that any model artifacts have been produced.

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

### 29 October smoke completion and private readback repair

Standalone GPU smoke 2 completed: Qwen, Ministral and SmolLM passed two-step training with finite loss and reload gates. Peak allocated GPU memory was 7,168,736,256 / 9,259,468,800 / 3,584,092,160 bytes respectively. DeepSeek failed the unchanged prompt-prefix masking safety guard during tokenisation; no checkpoint substitution or unsafe masking was applied. Standalone eligibility was 893/1,000 rows (804 train, 89 validation); full protected-support-page exclusions still apply separately. The recorded smoke timing includes evaluation overhead and is not a full training estimate.

Campaign 5 stopped with HTTP404 after its smoke guard and before training, responses or judge calls. Its restore used the SDK successfully, but the legacy private-file hash readback URL was unsupported. The repair uses the authenticated Kaggle SDK for temporary cloud readback, including server ZIP envelope unwrapping, then hashes the exact file and removes the temporary copy. Research-artifact readback is explicitly cloud-only. Empty private bootstrap v2 and every historical dataset remain intact. Repaired private campaign 6 is confirmed RUNNING with exact code hash `5c544a287b007e4da3dd5abc341821bdecc54b458186b1861d0f1962e48b2249`, T4 x2, existing secrets. It independently repeats fresh gates and continues supported models to fixed-epoch training; successful full training/adapters/inference/judging are not yet confirmed.

### Campaign 6 private checkpoint v5 (29 October 12:56 UTC)

Private ready v5 confirms export readback and preparation completed, with 834 eligible rows: 751 training and 83 validation after protected-page and sequence-length checks. Qwen GPU0 and Ministral GPU1 workers are active (combined smoke/training/inference stage); SmolLM queued and DeepSeek tokenisation blocked. No full adapter, inference response or judge evaluation has yet been confirmed. Production D1000 remains 1,000 unchanged source rows.

Both judges stopped before calibration with `Restricted outputs must be outside Git repository`: copied production `common.py` derived its protected source parent from the old transport placement, which included its runtime cache. A narrow repair nests the copied transport source one directory deeper and leaves the cache as a sibling outside that protected source tree; the original privacy guard remains unchanged. The source/runtime separation test passes (eight targeted D1000 tests). This repair is PREPARED ONLY and is not injected into active execution 6. Do not cancel active model work or launch a second writer. After a safe terminal checkpoint, audit the exact code-fingerprint migration and resume completed model/call journals with this repair. Actual judge API calls and future calibration success remain unconfirmed.

### Recovery execution 7 (30 October)

Kaggle reported execution 6 `CANCEL_ACKNOWLEDGED` after approximately twelve hours; the actor/cause is not established. Exact terminal aggregate receipt confirms private v58 was read back successfully, two completed workers and 608 responses preserved, SmolLM unfinished, DeepSeek tokenisation blocked and both judges still at zero.

The existing secret-enabled editor started private T4x2 execution 7. Its source hash is `8ada4766fea2e506e7c1745810a76ab6447f951469ef60648fc867eeeb4ab75f`; API independently confirms RUNNING. Source notebook 6 was hashed and its full cloud-Python component hash map audited. Only judging directory layout, coordinator exact-hash migration and code staging of the audit changed; model/training/data/prompt/rubric code remains unchanged. `resume_repair_v1.json` binds every source and destination Python file. The coordinator rejects unrecognised changes and records the transition in private progress. Nine targeted tests pass, including unrelated-code rejection and privacy guard isolation.

Recovery restores the newest private checkpoint, skips both completed workers and resumes saved SmolLM optimizer/inference state. Startup restoration and successful judge calibration are not yet confirmed. Do not launch another writer. Review-stage dependency: judge task assembly imports `scripts.run_semantic_judge.reference_package`; the current stage file list does not explicitly bundle scripts. Verify that import before claiming real judging works; prepare a code-only packaging repair at a safe boundary if missing, retaining exact historical reference assembly.

### Recovery 8: bounded cloud working storage and exact receipt fetching

Execution 7 failed in preflight restore with `OSError: [Errno 28] No space left on device`, before resumed models or judging. Its FAILED.json was zero bytes; the specific error was confirmed in Kaggle's UI, without downloading raw failure logs. Private ready v58 remained unchanged (7,075,889,157 archive bytes).

Recovery 8 mounts that private dataset read-only, verifies its archive hash and exact progress against current remote metadata, and extracts runtime files without copying the archive into working disk. Completed worker adapters/receipts/journals are retained; their redundant runtime optimizer copies are skipped, and only the latest unfinished optimizer checkpoint per phase is extracted. Original optimizer history remains immutable in prior private versions. A restore audit records this selection. Ten targeted tests pass, including original-archive immutability and latest-optimizer selection. The exact existing `scripts/run_semantic_judge.py` is bundled for canonical reference assembly. Model/training, schema, benchmark, dataset, prompts and judging rubric remain unchanged.

The monitoring SDK helper was also found to write notebook logs despite its filename filter. It has been replaced with direct metadata listing and exact aggregate-JSON fetches, including empty receipt handling. Three unintended local log copies were removed unread. No unrestricted logs are fetched or written by the corrected monitor.

### Evaluation-only recovery (30 October)

Execution 8 completed with three successful model workers and 912 responses in private checkpoint 93. DeepSeek remains unavailable. CPU diagnostic execution 9 independently hashed and restored that archive, verified all twelve available conditions contain the same 76 benchmark IDs, and inspected the existing fictional calibration caches without new API calls. No raw outputs were downloaded. Groq's three failures were schema failures, not truncation: each stopped normally but omitted `citation_assessments` and included an extra root property. Gemini's refusal case returned support 1 instead of null; earlier attempts also contained invalid legacy fact-score types.

`python -m cloud.d1000.stage --output /private/tmp/d1000-evaluation --mode evaluation-only` stages a CPU-only notebook using `evaluation_only.main`, with no training or inference dispatch. It verifies every adapter against its existing receipt, all 912 responses and frozen question populations, and an exact evaluation source audit. Every results checkpoint rechecks an immutable file manifest and the authoritative cloud version. The single coordinator preserves completed data/adapters and privately persists caches, ledgers, calibration receipts, individual votes, errors, metrics and figures with readback verification.

The new judge transport sends the schema on wire, retains strict local semantic validation, and explains the existing distinction between refusal correctness and null factual evidence support. Known malformed or cross-field-invalid replies receive at most three durable attempts with validator feedback, never expected calibration scores. A wrong calibration correctness score fails the unchanged nine-case gate. Unknown outcomes require recovery from saved raw replies or remain blocked; they are not replayed. Original provider receipts, conservative free quotas and account ledgers are retained. Judges run independently, and either passing judge can proceed while the other remains explicitly incomplete. Quota waits can span further sessions; no paid fallback or automatic quota reset is used.

Sixteen targeted tests pass, including refusal validation, malformed/incomplete replies, bounded retries, cache reuse, unknown-request blocking, immutable artifact changes and CPU staging. Execution 10 was started on CPU with exact source hash `8b6f55db4d50525ad9b52afa055ba08ee2ffd04ae38e91f4139383d71e75bc5c`. Calibration outcomes and research progress must be read from its actual private receipts; launch alone does not establish success.

Execution 10 was stopped by the exact source audit before API calls or persistence: a broad bootstrap replacement also modified bundled stage.py. The builder now replaces only the standalone dispatch line; the staging test verifies every embedded source file against its local hash. Corrected CPU execution 11 source hash: `2af4b53b81549c605ef648fba4e334783747b8596fda8c6fe74c204eb5db8bdf`. All sixteen targeted tests pass. Calibration/result success is still receipt-bound.

### Verified calibration repairs and safe next handoff

Gemini passed all nine unchanged anchors in execution11 and is evaluating saved responses as the single results writer. Read-only fictional Groq probes12/13 identified a provider schema compiler incompatibility with integral/fractional numeric enums, including all-float literals. Groq wire schemas now use number/null fields with explicit allowed-score descriptions; the unchanged local canonical schema still enforces exactly0/.5/1/null. Integer legacy fact-score enums remain. Groq CPU probe14 passed all nine anchors, including both refusal decisions and missing-reference uncertainty. Its public summary excludes account ledger entries; the full receipt and exact quota deltas remain private.

Read-only diagnostic15 found research rejections concentrated in citation assertions, span grounding and array arity. The prepared transport revision3 adds specific diagnostics, Gemini task-derived array lengths, and only the same judge's previous invalid reply as bounded repair context. It never supplies expected calibration labels. Known-invalid attempts are bounded to three per audited transport revision; unknown outcomes remain blocked or are recovered from durable replies. Already valid votes remain untouched. Fifty-one relevant semantic/D1000 tests cover both known-invalid repair limits and falsified/duplicate calibration receipts.

The prepared next evaluation coordinator refuses to start while writer11 is active. After its verified terminal boundary, it restores the newest private checkpoint, preserves Gemini journals/caches and the source93 immutable inventory, adopts Groq's actual calibration14 receipt and merges all real diagnostic quota reservations without resets. It never trains or regenerates candidate responses. It may require multiple free-quota windows; incomplete votes stay missing. Code currently prepared in GitHub intentionally differs from immutable active11 source.

Stage the next CPU notebook with `python -m cloud.d1000.stage --output /private/tmp/d1000-dual-next --mode evaluation-only --recovery-receipts /private/tmp/motor-d1000-private-recovery-receipts`. The receipt directory is private staging metadata, not Git content. Do not publish individual quota ledger entries. Update the mounted models/results dataset to the newest verified private version, confirm old writer termination and no overlapping provider caller, then use the existing secret-enabled editor. Publish only safe aggregates and code. Exact source and audit hashes must be checked again after launch.
