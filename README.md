# UK Motor Insurance DSLM

Lean starting point for comparing small base models and QLoRA supervised fine-tuning with the existing insurance RAG pipeline.


## Experiment progress — 7 October 2026

The following table compares the same frozen 140-question test population.
Historical results use the corrected abstention reading B. A dash means no
measurement is available; it is not zero. New model outputs have not yet been
generated or judged.

| Experiment | Composite | Correctness | Citation resolution | Abstention | Status |
|---|---:|---:|---:|---:|---|
| Previous Qwen2.5-7B + RAG | 0.608431 | 0.388771 | 0.897872 | 0.723421 | Reproduced from saved outputs |
| Previous QLoRA-RAFT + RAG | 0.599115 | 0.348093 | 0.919463 | 0.746148 | Reproduced from saved outputs |
| Qwen2.5-3B without RAG | — | — | — | — | Private GPU evaluation prepared |
| Qwen2.5-3B + RAG | — | — | — | — | Private GPU evaluation prepared |
| E1: Ministral-3-3B without RAG | — | — | — | — | Private GPU evaluation prepared |
| E2: Ministral-3-3B + RAG | — | — | — | — | Private GPU evaluation prepared |
| E3: Ministral QLoRA-SFT without RAG | — | — | — | — | Waiting for training smoke |
| E4: Ministral QLoRA-SFT + RAG | — | — | — | — | Waiting for training smoke |

Steps completed, in simple words:

1. Checked the old benchmark and reproduced both historical RAG scores.
2. Added both small models using the same frozen evidence and generation limits.
3. Built training examples only from eligible seed records, recording where each
   question, answer and supporting passage came from. Dev/test examples were excluded.
4. Checked real tokenisation and assistant-only labels. Twenty-one examples fit
   the 2,048-token training limit; two longer examples were excluded whole.
5. Ran 168 local regression and safety tests. All 42 protected input files remain
   unchanged. None of the 280 evaluation prompts needs evidence truncation.
6. Authenticated Kaggle locally and prepared separate private evaluation and
   training packages. Upload is pending explicit private-data export approval.

No GPU result or adapter-reload success is claimed yet. Full training remains
blocked until the CUDA smoke passes every gate. The small, answerable-only SFT
population is a limitation; no new human entailment review has been performed.
Benchmark text, raw answers, training examples, adapters, logs and credentials
stay outside this public repository.

## Layout

- `src/`: retained retrieval, generation, citation checking, and ingestion.
- `eval/`: retained comparison and judging components; `JUDGE_RUBRIC.md` defines the inherited grading rubric.
- `train/kaggle/`: training, packaging, and submission components to adapt for plain SFT.
- `tests/`: five retained regression-test modules.
- `data/gold/SCHEMA.md`: schema used by the persona regression test, not a results report.

## Private data

The essential benchmark, frozen test contexts, historical answers, source PDFs and corrected baseline scores are stored outside this repository:

`/absolute/path/to/uk-motor-insurance-dslm-private-data`

Before invoking inherited scripts, set the data path in the shell (the scripts do not automatically read `.env`):

```bash
export POLICY_QA_HOME="/absolute/path/to/uk-motor-insurance-dslm-private-data"
```

The private folder contains one `src/common.py` configuration mirror required by the inherited path resolver. Keep it aligned with the code copy until the resolver is refactored. Its `baseline/abstention.json` records the historical corrected scores. Do not overwrite the 140 frozen test contexts or original answer files. Store new experiment outputs separately.

## Adapt before training

The inherited scripts remain available for reproduction. Use the controlled 3B extension below for the new experiment; the following cautions apply to the legacy entry points.

- Build a new SFT dataset with supported answers; do not reuse the RAFT-specific packaging split unchanged.
- Add model identifiers and separate dev/test context and result paths. The current harness can overwrite test contexts when freezing dev questions.
- Use adopted abstention reading B. The old `judge.py` entry point still uses the retired detector and must be adapted.
- Adapt tokenizer/chat-template and assistant-loss masking for the chosen model.
- Measure sequence lengths before choosing a 2,048-token limit; never silently truncate supporting evidence.
- The 30-record dev set has 22 travel and eight home questions, no motor questions. Keep it out of training while using it for validation.

## Attribution and baseline

Core code derives from the UK Insurance DSLM project by Sumer Sener and Ahmad Bafakih, Tech Mahindra Makers Lab. Source code commit: `be4bedd`; source data pin: `811173d`. No new license is asserted by this cleanup.

Historical test composites: base Qwen2.5-7B + RAG **0.608431**; QLoRA-RAFT + RAG **0.599115**. These use 50% correctness, 30% citation resolution and 20% balanced abstention. Citation resolution is not claim entailment.

This repository starts with a clean snapshot of the retained code. Historical Git metadata and removed content are preserved in local backups; private data is not included.

## Controlled 3B extension

`experiments/` adds Qwen2.5-3B and Ministral-3-3B base evaluation, plus a
smoke-first Ministral QLoRA-SFT path. Inherited RAG prompts, citation resolution,
composite weights and adopted abstention reading B are reused. Both Ministral
conditions use the pinned `unsloth/Ministral-3-3B-Instruct-2512-bnb-4bit` checkpoint.
The Qwen checkpoint and both revisions are in `configs/insurance_3b.json`.
This quantised Ministral baseline must be labelled as such, not as an FP8 run.

### Private data preparation

Run commands from this repository root. All output paths must be outside it;
use a fresh directory for each run. Existing outputs are never overwritten.

```bash
export POLICY_QA_HOME="/absolute/path/to/uk-motor-insurance-dslm-private-data"
export EXPERIMENT_DIR="/absolute/path/to/private-experiments"
python -m experiments.prepare_sft --home "$POLICY_QA_HOME" --output "$EXPERIMENT_DIR/sft-audited"
# Tokenizer dependencies only for this local step; no model weights required:
# pip install transformers==5.19.0 jinja2==3.1.6 sentencepiece protobuf
python -m experiments.tokenise --bundle "$EXPERIMENT_DIR/sft-audited" --output "$EXPERIMENT_DIR/sft-ready"
```

Only verified seed records are candidates. Dev/test QA, frozen contexts and
historical RAFT/synthetic files never supply training examples. Evidence is
selected deterministically from seed citation quotes in the shared frozen
corpus. Shared policy passages are permitted; held-out insurers and annotated
seed intent clusters crossing into dev/test are excluded. Exact/near-question,
complete-answer and rendered-content checks supplement provenance; these
mechanical checks cannot prove the absence of every semantic paraphrase.
Every row preserves source and evidence hashes, selection method, review status
and target construction. Targets retain the complete verified seed answer plus
resolvable evidence citations. This relies on inherited council verification,
not a new human entailment assessment. Unanswerable seeds are deferred because
this first SFT dataset requires supporting evidence. Whole overlength examples
are excluded with a recorded reason; prompts and targets are never truncated.
The fixed abstention instruction is inherited prompt boilerplate, not copied
from a test answer.

### Base evaluation first

Frozen test prompts can be exported without a retrieval service:

```bash
python -m experiments.evaluate prepare --home "$POLICY_QA_HOME" --split test --output "$EXPERIMENT_DIR/test-eval"
```

This copies the exact saved test contexts into a NEW private evaluation bundle;
it never regenerates or overwrites them. For dev evaluation first, restore LM
Studio with the original Nomic embedding model and install
`requirements-retrieval.txt`, then reproduce an index from the frozen chunks:

```bash
python -m experiments.index --home "$POLICY_QA_HOME" --output "$EXPERIMENT_DIR/dev-index"
python -m experiments.evaluate prepare --home "$POLICY_QA_HOME" --split dev --index "$EXPERIMENT_DIR/dev-index" --output "$EXPERIMENT_DIR/dev-eval"
```

Dev retrieval requires BGE reranking; silent fallback is rejected. The corpus
and retrieval parameters remain unchanged. Keep all tuning decisions on dev;
this inherited dev population contains no motor questions. The full 140-test
population is the historical comparison; a motor-only score is a separate slice.

On CUDA after installing `requirements-experiment.txt`:

```bash
python -m experiments.evaluate run --bundle "$EXPERIMENT_DIR/dev-eval" --model qwen3b --output "$EXPERIMENT_DIR/qwen-dev"
python -m experiments.evaluate run --bundle "$EXPERIMENT_DIR/dev-eval" --model ministral3b --output "$EXPERIMENT_DIR/ministral-dev"
```

Each run generates both without-RAG and with-RAG answers, greedy decoding and
700 new tokens. An 8,192-token inference budget preserves full evidence; the
2,048 limit applies to training. Overflow fails rather than truncating.
Model revision and quantisation are fixed. This is generator comparison, not
retriever tuning. Use `--adapter /private/path/adapter` with `--model ministral3b`
for E3/E4 after a valid trained adapter exists.

### Private Kaggle smoke

```bash
python -m experiments.kaggle_stage --kind smoke --username YOUR_KAGGLE_USERNAME --bundle "$EXPERIMENT_DIR/sft-ready" --output "$EXPERIMENT_DIR/kaggle-smoke"
python -m experiments.kaggle_stage --kind base-eval --model qwen3b --username YOUR_KAGGLE_USERNAME --bundle "$EXPERIMENT_DIR/dev-eval" --output "$EXPERIMENT_DIR/kaggle-qwen"
python -m experiments.kaggle_stage --kind base-eval --model ministral3b --username YOUR_KAGGLE_USERNAME --bundle "$EXPERIMENT_DIR/dev-eval" --output "$EXPERIMENT_DIR/kaggle-ministral"
```

Staging never uploads or runs. After connecting your account, create a **private**
Kaggle dataset from the selected `dataset/` folder (preserve nested folders),
and use `kernel/` to create its private T4 script. Training and evaluation are
separate datasets: the smoke training upload includes only audited seeds and
code, never test prompts, frozen test contexts or gold evaluation answers.
Inspect `FAILED.txt` on failure. A successful training kernel must produce
`run/receipt.json` with every gate true, not merely a successful process status.

The two-optimizer-step smoke checks every training row's real tokenisation,
assistant-only labels and sequence length, then checks finite loss, changed
adapter weights, language-only trainable parameters, fresh-base adapter reload,
weight equality, logits agreement and generation. Only language-model attention
and MLP projections receive LoRA; vision and projector parameters stay frozen.
No fallback to full-text loss is permitted.

Full training is deliberately not launched by the Kaggle staging command.
`experiments.train --mode full` additionally requires `--smoke-receipt` matching
training data, code, configuration and dependency versions. Changing any of
these requires another smoke. Default configuration is rank 16, alpha 32,
dropout .05, LR 1e-4, two epochs, sequence 2048 and seed 0. GPU compatibility and
adapter reload are unverified until an actual CUDA smoke receipt exists.

### Scoring and results

```bash
python -m experiments.evaluate score --home "$POLICY_QA_HOME" --bundle "$EXPERIMENT_DIR/test-eval" --answers /private/path/answers.jsonl --output "$EXPERIMENT_DIR/judging"
```

Missing correctness scores produce anonymous private judge tasks and a separate
mapping, never a partial composite. Apply `eval/JUDGE_RUBRIC.md`, use three
independent votes and `eval/judge_votes.py:merge_task`, then map task codes back
to `{qid,row}`. Supply the merged scores with `--scores` and a new output path.
The existing merge rule and evidence-span checks are preserved. Human spot
checks remain necessary; citation resolution is not a hallucination metric.

```bash
python -m experiments.table --home "$POLICY_QA_HOME" --metrics /private/path/metrics.json --output "$EXPERIMENT_DIR/comparison"
```

The table includes the corrected historical Qwen7B and RAFT results and marks
missing new experiments `not run`. It rejects dev/smoke populations in a test
comparison. No new measured performance is claimed before generation and judging.
