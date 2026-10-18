# Motor evaluation protocol, draft v1

## Populations and leakage

The inherited benchmark has 140 test questions, including 76 motor items (67 answerable, nine unanswerable), and zero motor development items. It remains protected. Historical results are descriptive and must not drive revised prompts or selection rules.

A new private development draft contains 20 freshly authored questions, 16 answerable and four insufficient-evidence cases, across eight LV= source groups. LV= is absent from the interim accepted training snapshot. Selected groups are absent from every inherited gold citation. Maximum question similarity under the recorded character comparison is 0.6495. These checks do not establish semantic independence or rule out incidental training-context overlap. Independent answer review, semantic leakage review, topic expansion and frozen retrieval export remain pending. This draft is ineligible for model selection.

Review the question, full supporting context, answerability, target, exclusions and citation entailment independently. Record reviewer identity/date, decisions and hashes privately. Expand beyond legal expenses and a single insurer. Reserve source groups before any new generation or training. Freeze reviewed QIDs, source/evidence hashes, prompts, retrieval pipeline and selection criteria before screening. Never obtain development questions by carving accepted training examples into a dev split.

## Conditions

E1 Base and E3 SFT receive no retrieved evidence. E2 Base + RAG and E4 SFT + RAG receive identical frozen evidence per question. Each uses the same original checkpoint and revision within a matched comparison. Preserve historical prompts for reproduction. The new six-model runner uses a shared single-user-turn wrapper with the same task instructions, the checkpoint's own chat template, greedy generation, seed 0 and a 700-token output cap. This prompt version is separate from historical evaluation.

No silent evidence truncation or checkpoint substitution. All models must receive identical source passages and compatible token budgets. If a task does not fit, stop that condition and report it rather than selectively dropping evidence. Preserve DeepSeek reasoning in raw outputs and report final-answer parsing and token-cap failures separately. Official Ministral and historical Unsloth quantised results cannot be pooled.

## Measures and decisions

Independent correctness uses the inherited key-fact rubric and three valid blinded votes. Citation resolution checks source metadata and is distinct from claim-level support. Assess claim support from frozen context independently. Report abstention sensitivity, unnecessary refusal rate and balanced abstention. Preserve inherited parse metric: citation parse or recognised abstention. Record latency, actual token lengths, GPU peak allocation, checkpoint/tokenizer revision, dtype, quantisation and software versions.

No new-model winner or composite until correctness is complete. The historical composite is 0.5 correctness + 0.3 citation resolution + 0.2 balanced abstention. Closed-book citation metrics and composites are N/A. Use correctness and grounding first, abstention next and efficiency as a practical tie-breaker. Freeze exact selection rules on reviewed dev data before protected test evaluation.

For paired binary outcomes bootstrap QID pairs (5,000 draws, seed 20261020), reporting sample size and uncertainty. The present parse-rate intervals are exploratory with no multiplicity correction. Do not interpret them as correctness effects. Citation ratios pool citations and do not represent independent trials. Avoid false precision with only nine unanswerable motor questions.

## New SFT gate

Current production snapshot is not training-ready. Complete human review and grouped deduplication first. Record token lengths, exclude whole overlength records and apply assistant-only loss masking. Keep answerability metadata out of model inputs. Reuse smoke, complete-epoch, saved-adapter and exact reload checks. Verify historical fingerprints against frozen launch code rather than a changed working tree. A new smoke is required for changed data, model or runtime. Full training is complete only after adapter save/reload, inference and saved outputs.
