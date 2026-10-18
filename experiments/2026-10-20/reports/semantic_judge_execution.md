# Independent semantic judge execution — 18 October 2026

## Verified execution

Both Kaggle notebooks are COMPLETE. The actual frozen motor benchmark has three successful checkpoints: Qwen2.5-1.5B, DeepSeek-R1-Distill-Qwen-1.5B and SmolLM2-1.7B. Each has 76 Base and 76 Base + RAG responses. Exact manifests, configuration hashes, pinned revisions, output hashes and 152 unique paired tasks per model passed `verify_motor_benchmark.py`. Llama/Gemma failed gated access; official Ministral failed `BFloat16 != Half` matrix compute. No notebook was restarted. Raw downloads are private.

Historical audit: 1,120 verified predictions, 608 motor responses (76 × eight arms), including 484 non-abstaining answerable responses. Historical SFT used 21 examples. The 140-question full benchmark remains separate. New-model normalized outputs preserve final-answer and output-cap failures, particularly relevant to DeepSeek. None is a correctness result.

## Permission and provider

The authoritative private receipt `uk-insurance-QA-data-local/production-v1/protected_question_permission.json` explicitly permits Gemini **questions-only leakage checking**, excludes gold answers and frozen contexts, and prohibits teacher/training use. It does not authorise transmitting complete semantic evaluation packages. The training CSV's private Kaggle storage approval does not extend that permission.

The official authenticated catalogue confirms `models/gemini-3.5-flash-lite`, catalogue version `3.5-flash-lite-07-2026`. The account free-plan receipt specifies 5 RPM, 250,000 TPM and 500 RPD, shared with the live data pipeline. One fixed fictional API smoke test passed strict JSON validation: 814 input / 149 output / 963 total reported tokens; estimated API cost $0 under the verified free quota. No restricted content was sent; production files and usage ledger were read only. Full historical three-vote judging needs 1,824 requests, so it exceeds a single free daily quota even before production usage.

Local fallback: `mlx-community/Phi-4-mini-instruct-4bit`, revision `ac1c269cb4222a4e136a3d09edad301056c1f36a`, loaded on Apple M1 Pro / MLX GPU with 16 GiB system RAM. All eight fictional calibration cases failed strict validation, including incorrect fact-score schemas and abstention labels. No protected response was submitted to this judge. Its scores were not salvaged or used as research results.

## Current coverage and interpretation

Real successful votes: **0**. Real three-vote responses: **0**. Human review: **0**. Agreement: **not measured**. Correctness, reference support, unsupported-material-claim rates and paired accuracy improvements are pending for every arm. The central SFT accuracy question remains unanswered; existing parsing/citation/refusal differences do not establish accuracy improvements.

The private `llm_judge_v2/` directory contains 1,064 shuffled inputs, a separate identity mapping, empty vote/attempt journals, pending result/per-question/disagreement tables, a blind 28-item human sheet, hashes, manifest and run report. Total full-batch vote requirement: 3,192. Historical responses are scheduled before new-model responses when Gemini permission is granted. Additional actual-score strata must supplement the initial review sheet after scoring.

Public tables contain only aggregate counts/known deterministic metrics and blank semantic scores. Five semantic PNG/SVG figures were correctly skipped. Their plotting script and exact pending input data are saved. Historical behaviour figures remain separately labelled.

## Resume

See [semantic protocol](../../../eval/SEMANTIC_JUDGE_V2.md). Gemini requires a separately recorded approval covering every field and `motor_semantic_evaluation`; the integration enforces this before reading the API key. Use `run-gemini --help` for the private permission, free receipt, read-only shared ledger, authenticated catalogue and env-file arguments. Only one worker may run. HTTP 429 stops without retry or paid fallback. Do not mix judges or weaken the fictional local calibration gate. After actual judging, aggregate, independently review and generate figures before any accuracy claim.
