# UK Motor Insurance DSLM

Lean starting point for comparing small base models and QLoRA supervised fine-tuning with the existing insurance RAG pipeline.

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

This cleanup preserves inherited implementations; it does not implement the new SFT experiment.

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
