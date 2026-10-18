# Current 20 October motor research experiment

Evidence checked on 18 October 2026. [Semantic execution report](reports/semantic_judge_execution.md) and [Kaggle receipts](reports/kaggle_status.json) give the current status. Historical 21-example SFT, new six-model base inference and future interim-data SFT remain separate experiments.

## Completed and pending

- Reverified 1,120 saved historical answers without rerunning inference. Existing aggregate results reproduce byte-for-byte.
- Created and verified a ten-column, 409-row private display CSV. The 25 repeated questions form deliberate answerability/context pairs, not exact duplicate records.
- Uploaded the owner-approved snapshot to a new **private** Kaggle dataset, version 1. All five individual file downloads match their SHA-256 checksums. Supervisor invitations await usernames.
- Completed the original six-model descriptive run and compatibility probe. Qwen1.5B, DeepSeek1.5B and SmolLM2 each passed verification for 152 responses; a failed-only Ministral dtype repair subsequently completed another 152. Llama/Gemma remain gated.
- Verified pinned tokenizer templates and prompt lengths for four open candidates. Llama/Gemma require gated access. CPU preflight is not GPU evaluation.
- Produced six measured PNG/SVG figures and archived obsolete root progress/presentation material. No new slides were created.
- Authorised official Gemini free-quota semantic judging is underway with rubric motor-semantic-v2.2. Full correctness and reference-support results require three valid repeated votes. No new interim-data SFT began.

## Private data access

[Training Dataset — CSV](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-interim-20261020-409) (private, version 1). Use an invited Kaggle account. Owner access and private visibility are verified; supervisor access awaits usernames/invitations.

Per-file download, verified with the owner account:

```bash
kaggle datasets download -d scarletthe0116/uk-motor-interim-20261020-409/1 \
  -f interim_motor_training_display.csv -p NEW_PRIVATE_DOWNLOAD
```

The full-archive download returned HTTP 404 during checking. The per-file command is the verified route. [Dataset privacy/integrity receipt](manifests/private_dataset_receipt.json).

Private experiment root is the sibling `uk-motor-insurance-dslm-private-data/experiments/2026-10-20/`:

| Path | Purpose |
|---|---|
| `data/` | Original immutable 409-record canonical CSV/JSONL and full manifest |
| `data-display-v2/` | Requested ten-column readable CSV and display manifest |
| `audit/paired_results.csv` | Restricted historical per-question predictions/references |
| `audit/behavioural_case_examples.csv` | Ten automatic citation/refusal illustrations; semantic correctness unjudged |
| `correctness-judge-batch/` | 850 blinded tasks and separate private mapping |
| `dev-draft/` | 20 unreviewed motor development drafts |
| `benchmark-v1/` | Frozen 76-question local scoring bundle and exact private submitted kernel |
| `kaggle-snapshot-v1/` | Owner-approved dataset upload package |
| `reverified-history-20261018/` | Independent recheck of saved historical artefacts, not new inference |

No private examples, retrieved passages, gold questions, adapters, judge tasks or raw model outputs belong in this public folder.

## Current Kaggle jobs

- [Actual six-model motor benchmark](https://www.kaggle.com/code/scarletthe0116/motor-six-model-frozen-test-20261020): COMPLETE; three models produced verified paired outputs. Existing private dataset mounted, no new evaluation text uploaded.
- [Earlier compatibility probe](https://www.kaggle.com/code/scarletthe0116/motor-six-model-compatibility-20261020): preserved, COMPLETE.
- [All eleven relevant job statuses and GPU quota](reports/kaggle_status.json).

[Failed-only Ministral repair](https://www.kaggle.com/code/scarletthe0116/motor-ministral-fp16-repair-20261020): COMPLETE; 152 responses verified, original pinned checkpoint retained. Post-dequantisation floating parameters explicitly cast to FP16. [Repair results](results/ministral_fp16_repair/six_model_summary.csv).

Do not resubmit completed jobs. The actual job requests a free T4 and float16. Official Ministral FP8 weights use documented dequantisation. Llama/Gemma licence gates remain intact. Check current status before any action.

## Reproduction

Run from repository root. All raw outputs must use fresh directories outside the public repository.

```bash
# Copy accepted records from a single resolved atomic export
python -m experiments.weekly_snapshot --source PRIVATE_ATOMIC_EXPORT \
  --source-repo DATA_REPO --output NEW_PRIVATE_SNAPSHOT
# Create the readable display while preserving canonical data
python scripts/export_supervisor_csv.py --snapshot PRIVATE_SNAPSHOT \
  --output NEW_PRIVATE_DISPLAY --public NEW_SAFE_DISPLAY_MANIFEST.json
# Freeze both no-RAG/RAG arms; this command does not submit anything
python scripts/stage_motor_benchmark.py --home "$POLICY_QA_HOME" \
  --output NEW_PRIVATE_STAGE --public NEW_PUBLIC_MANIFEST_DIRECTORY
# Current existing job status, with a current Kaggle client
python scripts/audit_kaggle_status.py --output NEW_STATUS_REPORT.json
# Local tokenisation only; no private text sent to model hosting
python scripts/check_motor_tokenizers.py --bundle PRIVATE_LOCAL_BUNDLE \
  --output NEW_TOKENIZER_REPORT.json
```

The actual run already exists. For a future completed job, download once to a fresh private path:

```bash
kaggle kernels output scarletthe0116/motor-six-model-frozen-test-20261020 \
  -p NEW_PRIVATE_GPU_DOWNLOAD
python scripts/verify_motor_benchmark.py --home "$POLICY_QA_HOME" \
  --launch PRIVATE_BENCHMARK_STAGE --downloaded NEW_PRIVATE_GPU_DOWNLOAD \
  --output NEW_PRIVATE_SCORED --public NEW_PUBLIC_SIX_MODEL_RESULTS
```

The verifier checks checkpoint revisions, exact manifest/input hashes, every paired QID and all 152 tasks for each complete model. Failed/partial models never enter comparison rows. Raw reasoning stays private; final-answer parsing and output-cap failures are explicit. Missing correctness/support stays null. Keep result versions create-only.

The exact submitted inference runner is [kaggle/benchmark_runner_v1.py](kaggle/benchmark_runner_v1.py), matching the launch SHA-256. The shared runner and verifier include a later final-answer normalisation safeguard for DeepSeek. This does not change frozen prompts, generation settings or checkpoint revisions. Private kernel source remains preserved with the launch.

For historical verification use `experiments.weekly_audit` with the saved launch/output folders and a new output directory. [Historical recheck](reports/historical_reverification.json) records byte-identical summaries. Historical training fingerprints must use launch-time code, not the changed current checkout.

QLoRA training remains in `experiments/train.py`, `tokenise.py`, `reload_check.py` and `scripts/kaggle_followup.py`. Its historical 21-record gates are not a ready 409-record training path. Certify a new reviewed dataset, adapt the row-count receipt gates, run a fresh smoke for the changed dataset/model/runtime, complete optimisation, verify the saved adapter reload, then finish both inference conditions. No new SFT success is claimed.

## Evaluation and figures

[Evaluation protocol](../../docs/evaluation_protocol.md) defines isolation and judging. The [versioned semantic judge protocol](../../eval/SEMANTIC_JUDGE_V2.md) extends the inherited key-fact rubric without changing historical predictions. The owner explicitly authorised full evaluation payloads to official Gemini under verified free quota. Paid usage remains disabled. Three valid repeated requests per response are required for primary results; vote-1 analyses are separate and preliminary. Reference support is distinct from claim-to-citation grounding. Human validation remains pending.

[Historical aggregate CSV](results/model_summary.csv), [paired uncertainty](results/paired_uncertainty.json), [automatic error taxonomy](results/error_taxonomy.json), [verified training receipts](manifests/historical_training_receipts.json) and [six PNG/SVG figures](figures/README.md) contain public summaries only.

```bash
MPLCONFIGDIR=/tmp/motor-mpl python scripts/plot_current_results.py
python -m pytest -q
```

The plotting script reads public aggregate inputs only and saves exact data/source hashes. New six-model quality figures require verified completed inference outputs. Correctness and unsupported-claim bars must never be inferred from citation resolution.

## Semantic judging reproduction

Use the commands in [motor-semantic-v2.2](../../eval/SEMANTIC_JUDGE_V2.md). The frozen historical/new-three-model judging package is private `llm_judge_v22_policy_scope/`; the later verified Ministral repair is prepared separately in `llm_judge_ministral_repair_v22/` with `prepare --exclude-history`. Separate run directories have separate opaque-ID namespaces and must be joined through their own private mappings. Do not replace the active package to add new predictions. All inputs, raw votes, disagreements and human review sheets remain private. Public [semantic results](results/semantic_judge_v2/judge_validation_summary.json) record actual coverage, missing votes and validation status.
