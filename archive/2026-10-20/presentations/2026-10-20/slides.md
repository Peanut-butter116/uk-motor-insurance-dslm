# 20 October 2026 supervisor presentation

Style: white background, black body text, red accents, TechM on every slide. Evidence cutoff: 18 October 2026.

## Slide 1 — Small Language Models for UK Motor Insurance Policy Q&A
Dataset Expansion, Base-Model Comparison and Fine-Tuning Progress

- 20 October 2026

- Tech Mahindra Makers Lab

- Evidence verified on 18 October 2026

Speaker notes: Today I will show what we have completed, what the measurements tell us and what still needs work. The research asks whether supervised fine-tuning improves small models on questions grounded in UK motor insurance wording. We are keeping base versus SFT without retrieval separate from the comparison with retrieval. I have checked saved GPU outputs instead of rerunning completed experiments. The progress includes a private dataset snapshot, matched Qwen and Ministral comparisons, and a new development draft. Correctness judging remains unfinished, so I will not claim an overall winner. The six smaller candidates are the next screening experiment, rather than six completed quality evaluations. This distinction matters because model loading, completed inference and a reliable accuracy result are different milestones. We can use this meeting to agree the review priorities that make the next experiments meaningful.

## Slide 2 — Research direction

- Inherited: policy corpus, RAG pipeline and 140-question test benchmark

- Motor focus: consistent policy questions, exclusions and realistic scenarios

- 1B–3B candidates: practical GPU experiments and deployment efficiency

- Core question: does SFT improve correctness, grounding and appropriate abstention?

Speaker notes: The previous interns built the retrieval system and investigated fine-tuning on a broader insurance population. Their historical seven-billion-parameter comparison did not establish a statistically reliable fine-tuning advantage. We retain their code and protected benchmark, while narrowing the new research to motor insurance. Coverage can depend on a specific insurer, wording version, limit or exclusion. A plausible general answer may therefore be wrong for the supplied policy. Small models allow more controlled comparisons within the available compute. Retrieval gives the model policy evidence. Fine-tuning aims to improve how it uses evidence and when it declines to guess. We need to test those contributions separately. The inherited test includes 76 motor questions. Its original development split contains no motor questions, so it cannot support motor model selection. We have prepared a separate draft and kept the test protected from prompt tuning. Source: repository README and verified experiment manifests.

## Slide 3 — Data pipeline and current snapshot

- 409 accepted records, 384 distinct question texts

- 135 inherited and 274 generated records

- 6 insurers, 27 source documents, 501 evidence sections

- 335 answerable and 74 insufficient-evidence records

- 2,000 committed target, 3,000 stretch. Freeze: 18 November

Interim only: zero human reviews, 25 repeated-question rows, training_ready = false

Speaker notes: These are counts from one resolved atomic export, copied into a new private snapshot on 18 October. They are accepted examples rather than generated candidates. The snapshot contains 135 inherited records and 274 generated records. Generation uses the production workflow, including Groq. Validation history is mixed: 325 records use the independent two-stage GPT validator and 84 use Gemini. All accepted rows carry a passed protected-question screen. This does not replace human review. The source explicitly marks the dataset as not training-ready. There are 25 additional rows sharing question text, so we need grouped deduplication and review before full training. We preserved every technical field in the canonical JSONL and lossless CSV, with a separate readable CSV view. The production status at extraction was paused for provider quota. We did not stop, restart or alter that process. Our revised research target is 2,000 accepted records, with 3,000 as a stretch. Source: interim dataset manifest.

## Slide 4 — Training schema and evidence validation

- Canonical fields: ID, insurer, policy/version, question, evidence, answer

- Also retained: citations, answerability, origin, validation and provenance

- Fictional demo: “Glass repair is covered up to 123 demo units.”

- Question: “What glass repair limit does this invented policy state?”

- Target: “123 demo units. [Demo, invented-policy, Glass, p.1]”

Illustration only. No restricted policy example has public content clearance.

Speaker notes: The CSV is ready to open privately and filter by insurer, question type, answerability and review status. Its canonical version JSON-encodes every cell so nested evidence and provenance remain lossless. The display version uses readable text and escapes spreadsheet formula prefixes. The example on this slide is entirely invented and does not describe a real policy. We have no content-level clearance for publishing a real accepted example, so it cannot serve as a validated training case. For real records, an independent first stage answers from the evidence without seeing the reference answer. A second stage checks the reference, claims, numbers, conditions, citations and answerability. The protected-question screen checks potential leakage. Human reviewers still need to verify a sample and resolve repeated-question variants. A negative coverage answer is different from insufficient evidence: the first needs a supporting exclusion, while the second should explain what is missing. Source: exporter schema and interim manifest.

## Slide 5 — Six base-model candidates

| Checkpoint | Nominal size | Family | Access |
| --- | --- | --- | --- |
| Llama 3.2 Instruct | 1B | Llama | Gated |
| Gemma 3 IT | 1B | Gemma | Gated |
| Qwen2.5 Instruct | 1.5B | Qwen | Open |
| DeepSeek R1 Distill | 1.5B | Qwen-derived | Open |
| SmolLM2 Instruct | 1.7B | SmolLM2 | Open |
| Ministral 3 Instruct | 3B | Ministral | Open |

Pinned revisions. T4 NF4 probe submitted. Quality screening awaits reviewed motor dev data.

Speaker notes: The six requested checkpoints now have pinned revision identifiers from their official Hugging Face metadata. Llama and Gemma require licence access, which we have not bypassed. Qwen and the DeepSeek distillation are related, so this is not six independent architectures. Ministral is a larger reference. The official Ministral candidate also differs from the Unsloth quantised checkpoint used in the historical experiment, so we will not combine those results. We have submitted a private T4 compatibility probe with entirely fictional text and no restricted dataset. This checks actual loading and generation where access permits. The screening runner records model and tokenizer revisions, generation settings, runtime, latency and GPU memory. It refuses unreviewed development inputs and never silently truncates evidence. Before quality screening, the new motor development draft needs independent reference review, semantic leakage checks and broader policy coverage. Sources: official Hugging Face model cards linked in six_models.json.

## Slide 6 — Verified base-model behaviour on motor questions

| Model / condition | Balanced abstention | Citation resolution | Parse rate |
| --- | --- | --- | --- |
| Qwen 3B Base | 0.493 | N/A | 25.0% |
| Qwen 3B Base + RAG | 0.500 | 0.889 | 69.7% |
| Ministral 3B Base | 0.454 | N/A | 36.8% |
| Ministral 3B Base + RAG | 0.604 | 0.812 | 75.0% |

76 paired questions. Correctness and claim support pending. Citation metric N/A without RAG.

Speaker notes: These are verified historical results on the same 76 motor questions, rather than new results for all six candidates. The Qwen model here is 3B, while the new candidate is 1.5B. We retained the original prompts, evidence contexts and generation settings. Retrieval substantially increases the inherited parse metric for both models. That metric counts a parseable citation or a recognised refusal. It is not a general JSON-format score. Citation resolution checks whether a citation matches source metadata. It does not establish that the source supports a claim. Without retrieval, the model has no supplied policy text, so citation quality is not applicable. Those arms help diagnose guessing and refusal behaviour. Balanced abstention combines refusing unanswerable questions with answering answerable questions. The motor population has 67 answerable and nine unanswerable items, so uncertainty is substantial. No correctness, claim-support or composite score is inferred from these automatic measures. Source: verified model_summary.csv and launch receipts.

## Slide 7 — Previous 21-example SFT experiments

| Model / condition | Balanced abstention | Citation resolution | Parse rate |
| --- | --- | --- | --- |
| Ministral 3B SFT | 0.695 | N/A | 47.4% |
| Ministral 3B SFT + RAG | 0.659 | 0.925 | 76.3% |
| Qwen 3B SFT | 0.493 | N/A | 30.3% |
| Qwen 3B SFT + RAG | 0.493 | 0.786 | 88.2% |

Four arms per model: Base, Base + RAG, SFT, SFT + RAG. No new 409-record training run.

Speaker notes: Both historical fine-tuning runs used 21 eligible seed examples. They did not use the new interim production snapshot. We verified full training, saved adapter hashes and successful reload checks against the launch-time code. The Qwen SFT inference had completed on Kaggle but its outputs had not been incorporated in the README. We recovered and verified them. With RAG, Ministral citation resolution increases from 0.813 to 0.925 on the motor subset. Qwen shows a different pattern: parse rate increases from 69.7% to 88.2%, while citation resolution decreases from 0.889 to 0.786. Qwen refuses none of the nine unanswerable motor items in that SFT-plus-RAG arm. This is evidence of mixed behavioural changes. It does not establish an accuracy benefit. A paired bootstrap estimates the Qwen parse improvement at 18.4 percentage points, with an exploratory 95% interval from 7.9 to 28.9 points. Source: verified summary and paired uncertainty report.

## Slide 8 — Findings and remaining limitations

- Verified: 1,120 answers across eight matched historical conditions

- SFT changes citation and refusal behaviour differently across models

- Correctness, claim support and unsupported-claim rates remain unjudged

- 20 new private dev drafts require independent review and expansion

- 850 correctness tasks prepared, requiring 2,550 independent votes

No overall winner. A completed compatibility probe will not establish model quality.

Speaker notes: The main finding is that we now have a verified comparison covering four conditions for each of two models. Fine-tuning behaviour differs between them, which supports the need for architecture comparison and proper judging. The historical seed dataset was small and answerable-only, so it gives limited training coverage for abstention. The new accepted export adds insufficient-evidence cases but remains uncleared for full training. The development draft uses an insurer absent from the snapshot and source groups absent from protected gold citations. It is mainly legal-expenses material and still needs independent review and semantic screening. We have prepared a blinded batch for existing correctness assessment, without sending protected answers to a new provider. Until that judging completes, success and failure cases can only describe citation or refusal behaviour, not factual accuracy. The small unanswerable motor sample limits confidence. Latencies are descriptive because these historical jobs were not a controlled speed comparison. Source: audit, dev draft manifest and judge batch status.

## Slide 9 — Roadmap to 18 December

| Period | Deliverable |
| --- | --- |
| 18–20 October | Verified progress deck, snapshot and experiment audit |
| 21 October–6 November | Review motor dev set, screen models, expand accepted data |
| 7–18 November | Human review, deduplication and final dataset freeze |
| 19 November–4 December | Full QLoRA SFT and fixed-retrieval four-arm evaluation |
| 5–18 December | Final model, reproducible repository, report and handover |

Rama and Tushar: agree review ownership, dev coverage and compute priorities.

Speaker notes: The deadline remains 18 December, with a dataset freeze on 18 November. The first priority after this meeting is to finish judging the answers that already exist. That makes model conclusions much more useful than another unjudged GPU run. In parallel, we need independent review and expansion of the new motor development population. Once the selection protocol is fixed, we can screen the six candidates and then evaluate selected conditions on the protected motor test. The training data should reach at least 2,000 accepted, reviewed and deduplicated records, with 3,000 as a stretch. We will then run full QLoRA training, retain the successful smoke and reload safeguards, and keep retrieval evidence identical between base-plus-RAG and SFT-plus-RAG. I would like Rama and Tushar to agree who reviews the development answers, which policy topics need stronger representation and how much GPU time we should allocate to screening versus final training. Source: project roadmap.
