# Small Language Models for UK Motor Insurance Policy-Grounded Question Answering

**Tech Mahindra Makers Lab · D1000 experiment · presentation: 3 November 2026**

Can relatively small language models answer UK motor insurance questions reliably, and how do supervised fine-tuning (SFT) and retrieval-augmented generation (RAG) affect correctness, grounding and refusal behaviour?

**Latest verified evaluation snapshot: private checkpoint v{{CHECKPOINT}} — {{STATUS}}, {{JUDGED}}/912 valid semantic judgements ({{COVERAGE_PERCENT}}%).** All 912 model responses and three trained adapters are saved and verified. Of the {{MISSING}} missing semantic judgements, **{{EXHAUSTED}} exhausted Gemini’s existing validation-attempt bound; {{QUARANTINED}} are quarantined**. Four protected transport-uncertain requests are retained, including one response already covered by a historical valid vote. No unknown request is replayed or reassigned. This is not a completed 912-response semantic evaluation.

**Private reporting checkpoint v445** independently preserves the same 833 judged responses plus refreshed canonical tables and figures. The figure/table provenance is verified evaluation snapshot 444; checkpoint 445 adds reporting artifacts, not new votes.

The D1000 findings below supersede the October 20 interim experiment as this repository’s primary results. Historical data, protocols and evidence remain preserved under [Historical experiments](#historical-experiments).

## Research questions and conditions

The frozen 76-question benchmark is reused across three completed model families and four conditions: **3 × 4 × 76 = 912 responses**. It contains 67 answerable and nine unanswerable questions. The same retrieved passages are used in the two RAG arms. No benchmark-based model selection is permitted.

| Condition | Student weights | Student-visible policy evidence |
|---|---|---|
| Base | Pinned instruct model | None |
| Base + RAG | Same base model | Fixed retrieved passages |
| SFT | QLoRA-adapted model | None |
| SFT + RAG | Same adapted model | The same fixed passages |

## Dataset and training

D1000 contains **1,000 source QA records: 462 independently verified and 538 explicitly unverified**. This is mixed-quality data, not 1,000 verified training examples. Only **834 eligible converted records** entered the frozen split: **751 training / 83 validation**. The split is grouped, seed 0, for validation logging; the fixed final training epoch is used. Human review remains uncompleted. [Dataset export and training audit](cloud/d1000/README.md).

QLoRA retains NF4/double quantisation, LoRA rank 16, alpha 32, dropout 0.05, learning rate 1e-4, two epochs, batch size 1, accumulation 8 and a 2,048-token training budget. Generation is greedy, seed 0, up to 700 new tokens within an 8,192-token total budget. Targets and evidence are not silently truncated. The frozen benchmark is excluded from training/selection. [Exact model revisions and configuration](cloud/d1000/config.json).

## Model completion

| Model family | Exact student checkpoint | D1000 status |
|---|---|---|
| Qwen2.5-3B | `Qwen/Qwen2.5-3B-Instruct` | Adapter verified; all four conditions, 304 responses saved |
| Ministral-3B | `unsloth/Ministral-3-3B-Instruct-2512-bnb-4bit` | Adapter verified; all four conditions, 304 responses saved |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Adapter verified; all four conditions, 304 responses saved |
| DeepSeek-R1-Distill-Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | **Unavailable:** prompt-prefix masking incompatibility; four conditions N/A |

The D1000 Qwen3B and Unsloth Ministral checkpoints are distinct from the separate October 20 Qwen1.5B/official-Ministral screening population. No DeepSeek result is inferred or fabricated.

## Evaluation protocol and coverage

The owner-authorised **`d1000-single-judge-fast-v1`** protocol requires one valid semantic judgement per response. Original judges: Groq `openai/gpt-oss-120b` and Gemini `gemini-3.5-flash-lite`; both passed all nine unchanged fictional calibration cases. Historical ties use Groq first; otherwise an existing valid vote is reused. Existing votes, the frozen rubric, citation validation and request journals are retained. No new paired-judgement collection or cross-judge agreement analysis is performed.

There are **{{GROQ_VOTES}} Groq and {{GEMINI_VOTES}} Gemini saved votes**, including 48 historical overlaps. Deterministic selection uses **{{GROQ_SELECTED}} Groq + {{GEMINI_SELECTED}} Gemini = {{JUDGED}}** responses. These are mixed-provider single-judge descriptive results, not dual-judge consensus or human-reviewed accuracy. A Gemini3.1 fallback failed calibration (8/9) and contributes **zero** research votes.

Every accepted semantic result passes the original schema, score/null constraints, citation occurrence and verbatim-span validation. Missing scores remain missing. Deterministic refusal/citation/parse metrics use all saved responses independently of semantic API availability. Strict journal and cache identities, archive hash, 80-file preservation manifest, all 912 responses, adapters and writer lease were reverified before aggregation.

## Latest verified results — provisional / {{STATUS}}

{{COMPARISON_TABLE}}

{{FINDINGS}}

## Presentation figures

All figures are 16:9, exported as4800×2700 PNG, editable vector SVG and vector PDF. Values come from the canonical aggregate table; no manually transcribed plot values are used.

![D1000 semantic correctness](reports/d1000_final/figures/01_semantic_correctness.png)

![D1000 coverage](reports/d1000_final/figures/05_coverage.png)

| Figure | Intended slide | PNG / SVG / PDF |
|---|---|---|
| 1. Semantic correctness | Main model/condition comparison, score coverage and conditional intervals | [PNG](reports/d1000_final/figures/01_semantic_correctness.png) · [SVG](reports/d1000_final/figures/01_semantic_correctness.svg) · [PDF](reports/d1000_final/figures/01_semantic_correctness.pdf) |
| 2. SFT effect | Positive and negative Base→SFT changes, with/without RAG | [PNG](reports/d1000_final/figures/02_sft_effect.png) · [SVG](reports/d1000_final/figures/02_sft_effect.svg) · [PDF](reports/d1000_final/figures/02_sft_effect.pdf) |
| 3. Refusal behaviour | Appropriate refusal, false refusal and their balanced score | [PNG](reports/d1000_final/figures/03_refusal_abstention.png) · [SVG](reports/d1000_final/figures/03_refusal_abstention.svg) · [PDF](reports/d1000_final/figures/03_refusal_abstention.pdf) |
| 4. Citation reliability | Metadata resolution versus semantic evidence support | [PNG](reports/d1000_final/figures/04_citation_reliability.png) · [SVG](reports/d1000_final/figures/04_citation_reliability.svg) · [PDF](reports/d1000_final/figures/04_citation_reliability.pdf) |
| 5. Coverage | Valid, missing and quarantined responses across 12 conditions | [PNG](reports/d1000_final/figures/05_coverage.png) · [SVG](reports/d1000_final/figures/05_coverage.svg) · [PDF](reports/d1000_final/figures/05_coverage.pdf) |
| 6. Results heatmap | Compact overview with N/A and semantic sample sizes | [PNG](reports/d1000_final/figures/06_results_heatmap.png) · [SVG](reports/d1000_final/figures/06_results_heatmap.svg) · [PDF](reports/d1000_final/figures/06_results_heatmap.pdf) |

[Slide-ready presentation summary](reports/d1000_final/presentation_summary.md) · [Reporting guide](reports/d1000_final/README.md).

## Reproduction

Public reporting requires only aggregate results. Raw QA, policy wording, reference answers, model responses, adapter weights, vote journals and quota entries remain private.

```bash
python -m pip install -r reports/d1000_final/requirements.txt
python scripts/build_d1000_results.py
python scripts/plot_d1000_results.py
python scripts/render_d1000_readme.py
python -m pytest -q tests/test_d1000_reporting.py
```

The input [verified_aggregates.json](reports/d1000_final/verified_aggregates.json) was generated inside the approved private Colab VM from the independently readback archive. The canonical [metrics_summary.json](reports/d1000_final/metrics_summary.json) and [CSV](reports/d1000_final/metrics_summary.csv) contain model, condition, metric, value, numerator, denominator, coverage, uncertainty and checkpoint provenance. [Coverage/provenance](reports/d1000_final/coverage_summary.json), [figure hashes](reports/d1000_final/figures/manifest.json) and the cloud-only [aggregate builder](cloud/single_judge_fast/reporting.py) make regeneration traceable. Private runtime aggregation requires approved credentials and a stopped, exclusively leased coordinator; the local scripts make no API calls.

Evaluation/training modules remain in [cloud/d1000](cloud/d1000/README.md); secure Colab restoration in [cloud/recovery_v1](cloud/recovery_v1/README.md); single-judge continuation in [cloud/single_judge_fast](cloud/single_judge_fast/README.md). Reporting does not start training, inference or generation. Do not use Run all on historical cloud notebooks.

## Limitations and remaining work

This snapshot is **PARTIAL**, not fully judged or eligible-complete. {{EXHAUSTED}} currently missing responses exhausted Gemini’s bounded validation attempts; {{QUARANTINED}} unjudged responses are protected by unknown outcomes. Missingness is concentrated in RAG conditions and may bias semantic comparisons. A valid judge record is not the same as factual ground truth. There is no completed human review, no multi-seed training uncertainty and no independent human/judge agreement assessment. Conditional bootstrap intervals do not account for judge disagreement or missingness and are not evidence of significant universal model differences.

D1000 mixes verified and unverified source examples, and only its eligible converted subset was used. The frozen test has 76 questions, including only 9 unanswerable items. Citation-support denominators are scored occurrences, whereas resolution uses all emitted IDs; they must not be conflated. The inherited parse metric is citation/refusal recognition rather than JSON structured-output compliance. DeepSeek’s unavailable arm remains N/A.

All four unknown requests and quota reservations are preserved. The original calibrated Groq continuation ran after the recorded 18:40:33UTC boundary and added **eight valid judgements**, independently persisted and read back. It then stopped on a provider error with zero active calls; the safe terminal output does not identify the exact error category. No restart or additional judge request was made during reporting. Gemini’s known-invalid bounds and failed third-model calibration remain unchanged. [Current run receipt](reports/d1000_final/run_receipt.json).

The historical capacity audit at snapshot 427 found 19 non-quarantined tasks fitting Groq’s 8,000-token/minute reservation limit and 65 exceeding it. This is historical capacity, not a claim of currently usable work. At terminal snapshot 444, 76 non-quarantined responses remain missing and three are quarantined. The final conservative ledger is available in the coverage report; server remaining quota is not independently known. No prompt was truncated and no attempt bound reset. [Historical capacity audit](reports/d1000_final/capacity_audit.json).

## Private research assets

Authorised account access is required; no private records are embedded in this repository.

- [Private D1000 training export](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-qa-d1000-training/versions/1).
- [Private frozen evaluation assets](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-frozen-evaluation-v2/versions/1).
- [Private results checkpoint {{CHECKPOINT}}](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-d1000-models-results/versions/{{CHECKPOINT}}).
- [Private CPU Colab evaluation](https://colab.research.google.com/drive/1g_rir-DPeeoZG3fAqNZxoY_JRXuRcbbD).
- [Working research PR #2](https://github.com/Peanut-butter116/uk-motor-insurance-dslm/pull/2); no merge is implied by this reporting update.

## Historical experiments

The earlier 409-record interim dataset, October 20 screening results, three-repeat-judge protocol and 21-seed SFT experiments are historical populations, not the current D1000 results. They remain available in [October 20 experiment](experiments/2026-10-20/README.md), [archive index](archive/README.md) and [exact predecessor README snapshots](archive/d1000-predecessor/README.md), with original manifests and hashes preserved. Historical three-vote results have not been overwritten or relabelled as the D1000 single-judge experiment.
