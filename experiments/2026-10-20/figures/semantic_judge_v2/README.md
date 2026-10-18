# Semantic evaluation figures

No correctness, RAG correctness improvement, correctness-versus-support, unsupported-answer or SFT-improvement figures were generated: **zero real responses have valid semantic judgements**.

Existing behaviour figures measure citation resolution, parsing and refusal only. They cannot answer whether SFT improves accuracy.

`figure_data.json` records the exact pending aggregate inputs and hashes. Regenerate after authorised judging: `python scripts/plot_semantic_results.py --results experiments/2026-10-20/results/semantic_judge_v2 --output experiments/2026-10-20/figures/semantic_judge_v2`.
