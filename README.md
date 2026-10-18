# UK Motor Insurance SLM Research

Tech Mahindra Makers Lab research for the **20 October 2026** supervisor meeting. Evidence checked on **18 October 2026**. Final project deadline: **18 December 2026**.

**Current execution:** both Kaggle jobs completed. Qwen1.5B, DeepSeek1.5B and SmolLM2 completed verified inference on 76 questions in each of two conditions (456 responses). Llama/Gemma failed on gated access; Ministral failed on a compute-dtype mismatch. Historical Qwen/Ministral comparisons contain 1,120 verified answers. Correctness and semantic support remain unjudged; the independent judging pipeline is implemented and privacy gated. [Live job links and last checked state](experiments/2026-10-20/reports/kaggle_status.json).

## 1. Project overview

Investigate whether small language models can answer UK motor insurance questions using the relevant policy evidence. Questions cover coverage, exclusions, conditions, limits and customer scenarios. The model should cite supporting clauses and decline to guess when information is missing.

The research focuses on practical 1B–3B models and reproducible experiments. It extends the predecessor's insurance RAG and fine-tuning work. This repository owns model experiments, evaluation and reporting. The separate data-production repository owns document collection and Q&A acceptance.

## 2. Research questions

Does supervised fine-tuning improve correctness, grounding, citation support and appropriate abstention? Four conditions separate adaptation from evidence access:

| Condition | Student weights | Evidence at inference |
|---|---|---|
| Base | Original instruct checkpoint | None |
| Base + RAG | Original instruct checkpoint | Fixed retrieved policy passages |
| SFT | Same checkpoint after QLoRA SFT | None |
| SFT + RAG | Same fine-tuned checkpoint | The same fixed passages |

Base versus SFT without RAG comes first. Those arms diagnose guessing and refusal when insurer-specific evidence is unavailable. Citation-support metrics are N/A without supplied evidence. The RAG pair tests how the model uses identical evidence.

## 3. Dataset

Authoritative source: [uk-insurance-QA-data](https://github.com/Peanut-butter116/uk-insurance-QA-data). Restricted records remain in private local storage, outside both public repositories.

| Verified interim snapshot | Count/status |
|---|---:|
| Automatically accepted records | 409 |
| Distinct question texts | 384 |
| Inherited / generated | 135 / 274 |
| Answerable / insufficient evidence | 335 / 74 |
| Insurers / documents / evidence sections | 6 / 27 / 501 |
| Completed human reviews | **0** |
| Training ready | **False** |

The latest resolved production export still matches the immutable snapshot. Dataset version: `63a736cd768267114bf229097d05b77c8f0fbc7ae4d9126bc512e4ae3fa36083`. [Snapshot manifest](experiments/2026-10-20/manifests/interim_dataset_manifest.json).

Validation combines deterministic checks, an independent evidence-only answer, reference/claim/citation auditing and protected-question screening. Provenance records 325 two-stage GPT validations and 84 Gemini validations. Automatic acceptance is distinct from human-reviewed ground truth. Printed policy editions remain unverified; source-content hashes identify versions.

The 25 repeated-question clusters are answerable/insufficient-evidence pairs using different contexts. There are **zero exact model-visible duplicates** and 409 unique IDs. Retain each pair as a group and review its evidence/label contrast before SFT. [CSV and cluster audit](experiments/2026-10-20/manifests/display_csv_manifest.json).

## 4. Downloadable training CSV

**[Training Dataset — CSV](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-interim-20261020-409)** — PRIVATE Kaggle dataset, version 1. Sign in with an authorised account. Supervisor invitations are pending usernames.

### Training dataset access

The new readable CSV contains **409 records** and these columns: `id`, `insurer`, `product`, `question`, `context`, `answer`, `answerability`, `citation`, `question_type`, `validation_status`.

Owner's verified private file:

```text
../uk-motor-insurance-dslm-private-data/experiments/2026-10-20/data-display-v2/interim_motor_training_display.csv
```

SHA-256: `2e368db03e8d0ab3015b56c573154a83c4c5b07496da4fc5a5c8776df5eed4cc`.

UTF-8 and quoted multiline cells round-trip successfully. Formula prefixes are escaped. The original 409-record lossless CSV/JSONL and full provenance remain immutable in the sibling `data/` directory. [Display manifest](experiments/2026-10-20/manifests/display_csv_manifest.json).

**For Rama, Tushar and other supervisors:** sign in to Kaggle and ask the dataset owner to invite your account. The owner approved this upload and Kaggle confirms `is_private=true`. Supervisor access cannot be verified until usernames are supplied and invitations are accepted. Owner-authenticated downloads of all five files matched their SHA-256 checksums, including all 409 display rows. Download `interim_motor_training_display.csv` from dataset version 1. The full-archive endpoint returned HTTP 404 during checking, so use the file download action or the per-file CLI command in the experiment instructions. Do not commit the CSV to this public repository. [Privacy/version receipt](experiments/2026-10-20/manifests/private_dataset_receipt.json).

## 5. Base models

All revisions are pinned in the [six-model configuration](experiments/2026-10-20/configs/six_models_descriptive.json). No checkpoint substitutions are permitted.

| Exact checkpoint | Nominal parameters | Current status |
|---|---:|---|
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1B | Failed: gated checkpoint access |
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 1B | Failed: gated checkpoint access |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | Verified Base / Base + RAG: 76 each |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.5B | Verified Base / Base + RAG: 76 each; Qwen-derived architecture |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | Verified Base / Base + RAG: 76 each |
| [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | 3B | Failed: BF16 / FP16 compute mismatch during original FP8 loading |

The official Ministral candidate differs from the historical Unsloth 4-bit checkpoint. The historical Qwen is 3B, while the new candidate is 1.5B. These experiment populations and checkpoints remain separate.

## 6. Methodology

The new descriptive run freezes **76 protected motor questions / 152 paired tasks** before submission. It reuses the already approved private Kaggle evaluation dataset and the exact historical retrieved passages. Gold answers and scoring contexts remain local. [Frozen protocol](experiments/2026-10-20/manifests/six_model_test_protocol.json).

All candidates use their pinned tokenizer/chat template with the same task instructions in one user turn, greedy generation, seed 0 and a 700-token output cap within an 8,192-token total budget. No evidence truncation is allowed. The new run uses float16 weights/compute, with original Ministral FP8 weights dequantised on loading as documented by its [official model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512). This differs from historical NF4 experiments, so timing and quality comparisons across protocols are not controlled.

The protected test provides descriptive results only. No prompt tuning, model winner or SFT selection will use it. The 20 new motor development drafts require independent answer review, broader source/topic coverage and semantic leakage checks before selection. [Evaluation protocol](docs/evaluation_protocol.md).

QLoRA training retains assistant-only loss masking, whole-record exclusions for overlength examples, complete-epoch checks and saved-adapter reload verification. Historical SFT used 21 seed examples. **No new interim-data SFT has started:** the accepted snapshot lacks completed quality review, development independence is unresolved and no reviewed dev-based model choice is available.

## 7. Validation and evaluation

- **Correctness:** blinded key-fact rubric with three independent votes. Historical motor subset: 484 non-abstaining answerable responses need 1,452 votes; full historical set: 850 responses need 2,550 votes. [Rubric](eval/JUDGE_RUBRIC.md).
- **Citation resolution:** automatic metadata matching against frozen passages. A resolved citation does not prove correctness or semantic support.
- **Claim support and unsupported claims:** require evidence-based semantic review. Both remain unjudged.
- **Abstention:** report refusal sensitivity on nine unanswerable motor items, unnecessary refusals on 67 answerable items and their balanced score.
- **Parsing:** inherited metric counts a parseable citation or recognised abstention. New raw outputs retain reasoning, final-answer presence and token-cap failures separately.
- **Efficiency:** record generation time and peak GPU allocation where measured. Historical latency is descriptive; historical peak memory is unavailable.

Paid API judging remains disabled. Existing production-provider permissions do not establish approval to transfer protected gold answers and model predictions for a new judging batch. No such payload was sent. Missing scores stay null, and no new composite or winner is reported. The historical composite remains `0.5 × correctness + 0.3 × citation resolution + 0.2 × balanced abstention`, only when all components exist.

### LLM-as-a-Judge Correctness Evaluation

Citation resolution checks whether a citation matches a supplied passage; it does not establish whether the passage supports a claim or whether the answer is correct. The new [versioned semantic evaluator](eval/SEMANTIC_JUDGE_V2.md), `motor-semantic-v2.0`, preserves the historical fact rubric while adding overall correctness (0, 0.5, 1), reference evidence support, material-error flags and explicit uncertainty.

The authenticated official Gemini catalogue confirms **`gemini-3.5-flash-lite`** is available. Its free-plan receipt and shared quota were checked, and one wholly fictional structured-output smoke test passed. **No protected evaluation payload was sent.** Existing permission allows questions-only leakage checks and explicitly excludes gold answers/frozen contexts; semantic processing needs a separate field-and-purpose approval. An independent local **Phi-4-mini-instruct 4-bit** fallback was attempted but failed all eight fictional calibration cases, so it was blocked from protected judging. [Execution report](experiments/2026-10-20/reports/semantic_judge_execution.md).

Private inputs are shuffled with seed 20261020 and contain only an opaque ID, question, answerability, reference answer/key facts, gold evidence, student-visible evidence, answer/citations and deterministic refusal decision. Student identities and Base/SFT/RAG labels remain in a separate private mapping. Reference evidence is distinct from what the student saw: no-RAG citation grounding is N/A. Supported source spans must be verbatim; factual entailment still needs semantic validation. Refusals are scored against answerability; missing/ambiguous/conflicting references produce null scores and review flags.

Primary outputs require **three valid repeated votes** with predefined median aggregation and material-contradiction caps. These are within-judge consistency checks, not independent human ground truth. Single-vote outputs are separate and preliminary. The blind 28-item review sheet is prepared; **human reviews completed: 0; agreement: not measured**. Paired differences use common question IDs and a grouped question-level bootstrap; source dependence and only nine unanswerable questions limit inference.

| Evaluation population | Responses prepared | Real responses judged | Required votes |
|---|---:|---:|---:|
| Historical motor, 21-example SFT, eight arms | 608 | **0** | 1,824 |
| New three-model Base / Base + RAG | 456 | **0** | 1,368 |

The earlier 484-answer / 1,452-vote count covers only non-abstaining answerable historical responses; the new batch includes refusals and all unanswerable questions. No correctness gain, grounding gain, composite or overall winner can yet be claimed. [Correctness table](experiments/2026-10-20/results/semantic_judge_v2/correctness_summary.csv) · [Grounding table](experiments/2026-10-20/results/semantic_judge_v2/grounding_summary.csv) · [Paired SFT/RAG comparisons](experiments/2026-10-20/results/semantic_judge_v2/sft_rag_comparison.csv) · [Validation coverage](experiments/2026-10-20/results/semantic_judge_v2/judge_validation_summary.json). Blank values mean pending, never zero. The five semantic figures await actual scores.

## 8. This week's results

**New six-model experiment:** the private [actual benchmark](https://www.kaggle.com/code/scarletthe0116/motor-six-model-frozen-test-20261020) and [compatibility probe](https://www.kaggle.com/code/scarletthe0116/motor-six-model-compatibility-20261020) completed. Three checkpoints produced all 152 required paired responses each; file hashes, revisions, frozen input manifests and question/condition uniqueness passed verification. Llama and Gemma could not access gated checkpoints. Official Ministral failed with a BF16/FP16 matrix dtype mismatch. No successful run was restarted. [New-model measured behaviour, latency and memory](experiments/2026-10-20/results/six_model_v1/six_model_summary.csv) remain separate from historical 3B SFT results. Correctness and semantic grounding are pending for every new condition.

**Directly relevant historical evidence:** 1,120 verified answers, comprising four arms each for Qwen2.5-3B and Ministral-3B. SFT used 21 examples. Below is the common 76-question motor slice, with correctness and claim support pending throughout.

| Model | Condition | Balanced abstention | Citation resolution | Parse rate |
|---|---|---:|---:|---:|
| Qwen2.5-3B | Base | 0.493 | N/A | 25.0% |
| Qwen2.5-3B | Base + RAG | 0.500 | 0.889 | 69.7% |
| Ministral-3B | Base | 0.454 | N/A | 36.8% |
| Ministral-3B | Base + RAG | 0.604 | 0.812 | 75.0% |
| Ministral-3B | SFT | 0.695 | N/A | 47.4% |
| Ministral-3B | SFT + RAG | 0.659 | 0.925 | 76.3% |
| Qwen2.5-3B | SFT | 0.493 | N/A | 30.3% |
| Qwen2.5-3B | SFT + RAG | 0.493 | 0.786 | 88.2% |

SFT behaviour is mixed: Ministral RAG citation resolution increases, while Qwen RAG citation resolution decreases despite a higher parse rate. This does not establish an answer-accuracy gain. Paired uncertainty and all 140-question historical aggregates remain available in the complete tables.

[Complete aggregate table](experiments/2026-10-20/results/model_summary.csv) · [Verification receipts](experiments/2026-10-20/results/verification_receipts.json) · [Paired uncertainty](experiments/2026-10-20/results/paired_uncertainty.json) · [PNG/SVG figures and exact plotted data](experiments/2026-10-20/figures/README.md) · [Execution and evaluation report](experiments/2026-10-20/reports/execution_report.md).

## 9. Reproduction

```bash
# Regression suite
python -m pytest -q
# Figures from public aggregates only
MPLCONFIGDIR=/tmp/motor-mpl python scripts/plot_current_results.py
# Stage the frozen benchmark in a NEW private directory; no upload occurs here
python scripts/stage_motor_benchmark.py --home "$POLICY_QA_HOME" \
  --output "$NEW_PRIVATE_RUN" --public experiments/2026-10-20/manifests
```

Check job state before submitting anything. The actual run already exists: **do not launch it again**. Private Kaggle uses the existing approved evaluation dataset. The completed outputs have been privately downloaded and verified. For future runs, download to a new private directory and run `scripts/verify_motor_benchmark.py` to validate the receipt, hashes, checkpoint revisions and all 152 task pairs before adding any results.

[Detailed reproduction](experiments/2026-10-20/README.md) covers snapshot loading, readable CSV export, paired base/RAG inference, QLoRA training gates, output verification and judging. Reusable code remains in `experiments/`, `src/`, `eval/`, `train/` and `scripts/`. Restricted data and adapters always stay outside the public repository.

## 10. Future roadmap

Reach **2,000 accepted examples**, with **3,000** as a stretch. Complete review and freeze the dataset by **18 November**. Finish reviewed dev-based model selection and the full SFT/RAG comparison, then deliver the model, reproducible repository and final research report by **18 December 2026**. [Roadmap](docs/project_roadmap.md).

## 11. Archived experiments

[Historical research, reproduction notes, figures and earlier presentation materials](archive/README.md).
