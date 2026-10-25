# UK Motor Insurance SLM Research

Tech Mahindra Makers Lab research for the **20 October 2026** supervisor meeting. Evidence checked on **19 October 2026 (London; 18 October UTC)**. Final project deadline: **18 December 2026**.

**Current execution:** four new models completed verified Base/Base+RAG inference (608 responses); Llama/Gemma remain gated. Historical predictions are verified (1,120 full-benchmark / 608 motor). Authorised free-tier Gemini judging saved 535 valid votes, but persistent HTTP 503 overload interrupted execution: 0 three-vote responses and 0 human reviews. Current correctness estimates are explicitly preliminary. [Job receipts](experiments/2026-10-20/reports/kaggle_status.json).

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
| [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | 3B | Verified Base / Base + RAG: 76 each after failed-only FP16 repair |

The official Ministral candidate differs from the historical Unsloth 4-bit checkpoint. The historical Qwen is 3B, while the new candidate is 1.5B. These experiment populations and checkpoints remain separate.

## 6. Methodology

The new descriptive run freezes **76 protected motor questions / 152 paired tasks** before submission. It reuses the already approved private Kaggle evaluation dataset and the exact historical retrieved passages. Gold answers and scoring contexts remain local. [Frozen protocol](experiments/2026-10-20/manifests/six_model_test_protocol.json).

All candidates use their pinned tokenizer/chat template with the same task instructions in one user turn, greedy generation, seed 0 and a 700-token output cap within an 8,192-token total budget. No evidence truncation is allowed. The new run uses float16 weights/compute, with original Ministral FP8 weights dequantised on loading as documented by its [official model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512). This differs from historical NF4 experiments, so timing and quality comparisons across protocols are not controlled.

The protected test provides descriptive results only. No prompt tuning, model winner or SFT selection will use it. The 20 new motor development drafts require independent answer review, broader source/topic coverage and semantic leakage checks before selection. [Evaluation protocol](docs/evaluation_protocol.md).

QLoRA training retains assistant-only loss masking, whole-record exclusions for overlength examples, complete-epoch checks and saved-adapter reload verification. Historical SFT used 21 seed examples. **No new interim-data SFT has started:** the accepted snapshot lacks completed quality review, development independence is unresolved and no reviewed dev-based model choice is available.

## 7. Validation and evaluation

- **Correctness:** blinded key-fact and overall correctness rubric with three repeated valid votes for primary results; separate preliminary vote-1 results below. [Inherited rubric](eval/JUDGE_RUBRIC.md) and [versioned extension](eval/SEMANTIC_JUDGE_V2.md).
- **Citation resolution:** automatic metadata matching against frozen passages. A resolved citation does not prove correctness or semantic support.
- **Reference support and unsupported claims:** preliminary semantic labels are available for part of the historical motor set. Claim-to-citation entailment remains pending.
- **Abstention:** report refusal sensitivity on nine unanswerable motor items, unnecessary refusals on 67 answerable items and their balanced score.
- **Parsing:** inherited metric counts a parseable citation or recognised abstention. New raw outputs retain reasoning, final-answer presence and token-cap failures separately.
- **Efficiency:** record generation time and peak GPU allocation where measured. Historical latency is descriptive; historical peak memory is unavailable.

Paid API judging remains disabled. Full semantic payload processing is explicitly owner-authorised for official Gemini free quota. Missing scores stay null; no winner is reported. Historical metric definitions are preserved.

### LLM-as-a-Judge Correctness Evaluation

Citation resolution checks whether a citation matches a supplied passage. It cannot establish whether that passage supports the claim or whether the answer is correct. The [versioned evaluator](eval/SEMANTIC_JUDGE_V2.md) uses **`motor-semantic-v2.2`**, preserving the inherited key-fact rubric and adding overall correctness (0, 0.5, 1), reference-evidence support, material errors and explicit uncertainty. Material coverage/safety errors score zero; missing or ambiguous references leave scores null.

**Judge:** official Gemini API, **`gemini-3.5-flash-lite`**, catalogue version `3.5-flash-lite-07-2026`, high thinking. The owner explicitly authorised protected questions, reference answers/key facts, policy wording and model answers/citations for this purpose. Actual free-tier limits were verified: 15 RPM / 250K input TPM / 500 RPD. Paid calls are disabled. Production is read only and retains a quota reserve. The fictional calibration gate passed before protected judging. Local Phi-4-mini and secondary Gemini 3.1 trials failed calibration and contribute no research labels.

Inputs are blinded, shuffled with seed 20261020, and omit student model names, Base/SFT/RAG labels and existing scores. They retain the question, answerability, gold reference/key facts, reference policy evidence, separately identified student-visible evidence, model answer/citations and deterministic refusal decision. Exact source spans are validated locally. No-RAG answers may be assessed against gold references, but citation grounding remains N/A. **Reference support is not claim-to-citation entailment; the latter remains pending for RAG.**

Primary results require three valid repeated stateless requests, predefined median aggregation and contradiction caps. These measure within-model consistency, not independent human truth. **535 valid votes** are saved: **502 responses have a valid first vote, 475 have a non-null first-vote correctness score, and 0 have completed all three votes**. Persistent provider HTTP 503 overload interrupted execution; no paid fallback was used. Human review completed: **0** of a prepared blinded 28-case sample. Human agreement is unmeasured. Repeat-vote agreement on 26 scoreable pairs is 80.8%, weighted kappa 0.778; this is not validation accuracy.

**Preliminary vote-1 results below are incomplete and unvalidated.** Each arm targets the same 76 questions (67 answerable, nine unanswerable); the displayed denominators are the available scoreable subset and differ between arms. Support excludes refusals and uncertain support labels. Do not infer effects by subtracting these unpaired means.

| Model | Condition | Correctness (scored n) | Reference support (scored n) |
|---|---|---:|---:|
| Ministral-3B | Base | 0.172 (64) | 0.262 (21) |
| Ministral-3B | Base + RAG | 0.558 (60) | 0.808 (52) |
| Ministral-3B | SFT | 0.152 (56) | 0.118 (34) |
| Ministral-3B | SFT + RAG | 0.455 (56) | 0.672 (61) |
| Qwen2.5-3B | Base | 0.087 (63) | 0.091 (55) |
| Qwen2.5-3B | Base + RAG | 0.364 (55) | 0.595 (58) |
| Qwen2.5-3B | SFT | 0.065 (69) | 0.077 (65) |
| Qwen2.5-3B | SFT + RAG | 0.288 (52) | 0.491 (55) |

Paired vote-1 comparisons use common scored question IDs and 4,000 question-group bootstrap resamples; missing scores are excluded, never zero. These selected complete-case intervals condition on one judge's labels, have no multiplicity correction, and do not represent human-validated full-test effects. Source-component sensitivity is reported separately (only three or four independent components).

| Model | SFT comparison | Scored pairs | Mean difference [95% paired CI] |
|---|---|---:|---:|
| Qwen2.5-3B | Without RAG | 57 | +0.000 [-0.061, +0.070] |
| Qwen2.5-3B | With RAG | 41 | +0.024 [-0.085, +0.122] |
| Ministral-3B | Without RAG | 50 | -0.030 [-0.110, +0.040] |
| Ministral-3B | With RAG | 47 | -0.149 [-0.277, -0.032] |

The available pairs do **not establish a consistent SFT accuracy improvement**. Both completion of the three-vote protocol and independent review remain necessary. No winner or new composite is reported. The original 484 non-abstaining answerable historical responses require 1,452 votes; this expanded historical protocol covers **all 608 responses / 1,824 votes**, including refusals and unanswerable items. The full 140-question historical benchmark stays separate.

[Primary correctness](experiments/2026-10-20/results/semantic_judge_v2/correctness_summary.csv) · [Primary grounding](experiments/2026-10-20/results/semantic_judge_v2/grounding_summary.csv) · [Preliminary vote-1 summary](experiments/2026-10-20/results/semantic_judge_v2/preliminary_single_vote_summary.csv) · [Paired vote-1 comparisons](experiments/2026-10-20/results/semantic_judge_v2/preliminary_single_vote_comparison.csv) · [Coverage/reliability](experiments/2026-10-20/results/semantic_judge_v2/judge_validation_summary.json) · [Execution report](experiments/2026-10-20/reports/semantic_judge_execution.md).

[Five semantic PNG/SVG figures](experiments/2026-10-20/figures/semantic_judge_v2/README.md) use the **same 13 scoreable questions across all eight arms**, a deliberately labelled matched preliminary subset (11 answerable, two unanswerable). Their means differ from the partial-arm table above and must not be described as full 76-question results. Exact inputs and hashes accompany the figures.

## 8. This week's results

**New six-model experiment:** the private [actual benchmark](https://www.kaggle.com/code/scarletthe0116/motor-six-model-frozen-test-20261020) and [compatibility probe](https://www.kaggle.com/code/scarletthe0116/motor-six-model-compatibility-20261020) completed. Three checkpoints produced all 152 required paired responses each; file hashes, revisions, frozen input manifests and question/condition uniqueness passed verification. Llama and Gemma could not access gated checkpoints. Official Ministral initially failed a BF16/FP16 mismatch; a separate [failed-only repair](https://www.kaggle.com/code/scarletthe0116/motor-ministral-fp16-repair-20261020) completed another 152 verified responses with the same pinned checkpoint. No successful model was rerun. [Repair results](experiments/2026-10-20/results/ministral_fp16_repair/six_model_summary.csv). [New-model measured behaviour, latency and memory](experiments/2026-10-20/results/six_model_v1/six_model_summary.csv) remain separate from historical 3B SFT results. Correctness and semantic grounding are pending for every new condition. The original three-model judge package and the separate 152-response Ministral extension are prepared; historical judging has priority. The original three-model judge package and the separate 152-response Ministral extension are prepared; historical judging has priority. The original three-model judge package and the separate 152-response Ministral extension are prepared; historical judging has priority.

**Directly relevant historical evidence:** 1,120 verified answers, comprising four arms each for Qwen2.5-3B and Ministral-3B. SFT used 21 examples. Below is the common 76-question motor slice for deterministic behaviour. Partial semantic results and their exact observed denominators are reported separately above.

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

**25 October controlled QA resumption:** verified private checkpoint v10 contains **412 accepted records** (338 answerable / 74 insufficient evidence), preserving all 409 originals and the frozen benchmark. Four cached p42 candidates were independently checked: three accepted, one retained for review; none regenerated. An independent restore audit passed after repairing two missing within-batch duplicate comparisons across a quota restart. The Groq GPT-OSS → Gemini workflow remains unchanged. The remaining 536 Hastings packages historically imply roughly 420–550 further acceptances; more eligible evidence is likely needed for 2,000. Free-tier continuation to 1,000 is now authorised; Kaggle execution v8 passed fresh generation and verification, reached **421 accepted**, and continues automatically; latest observed private checkpoint is v43, with 644 generated candidates. [Audit, receipts and restart instructions](cloud/qa/README.md) · [Progress](cloud/qa/progress.json) · [Opt-in SFT loader proposal](cloud/qa/SFT_LOADER_PROPOSAL.md).

## 11. Archived experiments

[Historical research, reproduction notes, figures and earlier presentation materials](archive/README.md).
