# Slide-ready D1000 summary — 3 November 2026

**Status: PARTIAL, provisional.** Verified checkpoint v444: 833/912 judged; 79 missing, including 3 quarantined unjudged responses. No full-coverage claim.

**Research question:** Can relatively small language models reliably answer UK motor insurance questions, and what benefits do SFT and RAG provide?

**Method:** Three successfully trained families (Qwen2.5-3B, Ministral-3B, SmolLM2-1.7B), four conditions (Base, Base+RAG, SFT, SFT+RAG), one frozen 76-question test (67 answerable, 9 unanswerable), 912 saved responses. Mixed-provider single-valid-judge evaluation using calibrated Groq GPT-OSS-120B and Gemini3.5 Flash-Lite, unchanged rubric/schema and original nine-case calibration. Deterministic metrics cover all saved answers. D1000 source: 1,000 records, 462 verified and 538 unverified; actual split 751 training / 83 validation.

### Findings supported by this snapshot

- **RAG raises observed correctness in each model family, with and without SFT** in this snapshot. Semantic coverage differs between arms; these are observed-subset comparisons, not unbiased population effects.
- **Highest observed correctness: Ministral-3B SFT + RAG, 62.7%** (42 rubric score points / 67 judged questions; conditional question-bootstrap interval 52.2%–72.4%). Its balanced abstention is 83.7% across all 76 saved answers. This is an observed ranking, not a statistically established universal winner.
- **SFT is not a uniform correctness improvement.** Qwen2.5-3B Base→SFT: 13.2%→20.3%; Qwen2.5-3B Base + RAG→SFT + RAG: 46.9%→45.2%; Ministral-3B Base→SFT: 18.4%→15.1%; SmolLM2-1.7B Base + RAG→SFT + RAG: 20.0%→40.0%. [Paired common-question sensitivity analysis](reports/d1000_final/sft_effects.json) reports matched-question deltas and intervals separately.
- **Refusal sensitivity can hide over-refusal.** Ministral SFT refuses 8/9 unanswerable questions but also 54/67 answerable ones (80.6% false refusal). With RAG this becomes 7/9 appropriate refusals and 7/67 false refusals; balanced abstention is 83.7%.
- **Citation matching is not semantic support.** Ministral SFT+RAG resolves 71/71 IDs, while its available semantic citation-support mean is 90.5% across 58 scored occurrences. SmolLM Base+RAG has only one resolving ID and zero scored support occurrences; its resolution is not a grounding-reliability finding.
- **Lowest observed correctness: SmolLM2-1.7B Base, 9.3% (75/76 judged).** A balanced-abstention score of 50% can result from answering every question and refusing none.

**Limitations:** Mixed-provider judging; unequal, potentially nonrandom semantic missingness; no completed human review; mixed-quality source examples; a small benchmark with only nine unanswerable items; one fixed training run; unavailable DeepSeek arm. Conditional bootstrap intervals do not capture judge error, training variation or missing-data bias. Four historical unknown requests remain protected.

**Takeaway:** The evidence supports supplying retrieved policy passages, but fine-tuning must be checked for correctness and over-refusal. Citation matching and semantic support answer different questions. Present the valid evidence with sample sizes and missingness.

**Suggested slides:** Figure1 main correctness; Figure2 positive/negative SFT effects; Figure3 refusal trade-offs; Figure4 citation reliability; Figure5 coverage/limitations. Figure6 is an optional compact summary. PNGs are 16:9; SVG/PDF are vector alternatives. No PowerPoint file was generated.
