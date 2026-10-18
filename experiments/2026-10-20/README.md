# 20 October supervisor experiment package

Evidence audit: 18 October 2026. Public files contain code, aggregate counts and hashes only.

## Completed

- Resolved and copied one atomic accepted export: 409 records, 384 distinct questions, 135 inherited, 274 generated. Canonical CSV and JSONL round-trip exactly. Read-only private files preserve all provenance. training_ready=false and human review remains zero.
- Verified four historical GPU runs: Qwen and Ministral base and 21-example SFT, each with and without fixed RAG. Total 1,120 unique nonempty answers. Recovered the completed Qwen SFT output from Kaggle.
- Verified both full training runs and adapter reloads against frozen launch-time code.
- Produced 1,120 private paired CSV rows, 16 public aggregate rows and exploratory paired bootstrap intervals.
- Prepared a blinded 850-response correctness batch requiring 2,550 independent votes.
- Authored 20 private motor dev drafts. Review and semantic independence remain incomplete.
- Pinned all six candidate model revisions. Submitted a synthetic T4 compatibility probe with no restricted upload.

## Exact private paths

Use the sibling directory `uk-motor-insurance-dslm-private-data/experiments/2026-10-20/`:

- `data/interim_motor_training.csv`: lossless JSON-cell canonical CSV.
- `data/interim_motor_training_display.csv`: readable filterable display view.
- `data/interim_motor_training.jsonl` and `data/interim_dataset_manifest.json`.
- `audit/paired_results.csv`: restricted predictions and references.
- `audit/behavioural_case_examples.csv`: ten automatic citation/refusal illustrations, with correctness unassessed.
- `correctness-judge-batch/`: blinded tasks and separate mapping, not sent to a provider.
- `dev-draft/questions.jsonl`: new unreviewed development questions.
- `output-qwen-sft-recovered/`: downloaded Kaggle outputs and adapters.

## Kaggle traceability

Existing private jobs (owner access required):

- [Qwen base](https://www.kaggle.com/code/scarletthe0116/uk-insurance-base-eval-qwen3b)
- [Ministral base](https://www.kaggle.com/code/scarletthe0116/uk-insurance-base-eval-ministral3b)
- [Qwen SFT inference](https://www.kaggle.com/code/scarletthe0116/uk-insurance-sft-eval-qwen3b)
- [Ministral SFT inference](https://www.kaggle.com/code/scarletthe0116/uk-insurance-sft-eval-ministral3b)
- [Six-model synthetic compatibility probe](https://www.kaggle.com/code/scarletthe0116/motor-six-model-compatibility-20261020)

The probe is not a development quality comparison. No new restricted training snapshot or dev data was uploaded. Licence-gated checkpoints require authorised access. Installed system Kaggle was too old for access_token; a temporary current client used the existing token without printing it. No credential changed.

## Reproduction

From the repository root, use `python -m experiments.weekly_snapshot --source PRIVATE_ATOMIC_EXPORT --source-repo DATA_REPO --output NEW_PRIVATE_SNAPSHOT`. Never use a live file for training.

Use `python -m experiments.weekly_audit --home PRIVATE_HOME --output NEW_PRIVATE_AUDIT --public NEW_AGGREGATE_DIR --qwen-download VERIFIED_QWEN_DOWNLOAD` to verify outputs. Outputs are create-only and private paths must be outside this public repository.

`experiments/six_model_benchmark.py` is the new CUDA runner. It rejects unreviewed dev bundles, never silently truncates, and records loading failures. `kaggle/` contains a self-contained code-only probe. It is GPU compatibility preparation, not a six-model quality result. Reuse the existing train.py/tokenise.py path only after adapting its 21-row receipt gates to a reviewed new dataset and running a fresh smoke. No new large-data SFT implementation or completed run is claimed.

## Remaining dependencies

Correctness and claim-support judging, reviewed diverse motor dev benchmark, confirmed licence access, and reviewed/deduplicated training data. Full six-model evaluation and new interim-data SFT remain incomplete. See the nine-slide presentation and weekly update.
