# Current research figures

Generated from verified public aggregates. White/black/red style, 300 dpi PNG and SVG with editable text. No private inputs or missing performance bars.

These figures contain historical 21-example SFT results and current six-model execution status. They do not claim a completed six-model quality benchmark.

| Figure | PNG | SVG |
|---|---|---|
| 01_base_models | [PNG](01_base_models.png) | [SVG](01_base_models.svg) |
| 02_effect_of_rag | [PNG](02_effect_of_rag.png) | [SVG](02_effect_of_rag.svg) |
| 03_historical_sft_rag | [PNG](03_historical_sft_rag.png) | [SVG](03_historical_sft_rag.svg) |
| 04_model_efficiency | [PNG](04_model_efficiency.png) | [SVG](04_model_efficiency.svg) |
| 05_dataset_composition | [PNG](05_dataset_composition.png) | [SVG](05_dataset_composition.svg) |
| 06_error_analysis | [PNG](06_error_analysis.png) | [SVG](06_error_analysis.svg) |

Exact plotted inputs and source checksums: [figure_data.json](figure_data.json).

Regenerate from repository root: `MPLCONFIGDIR=/tmp/motor-mpl python scripts/plot_current_results.py`.

All performance populations are the same 76 motor questions. Correctness and semantic support remain missing, never zero. Efficiency is descriptive and GPU memory is unavailable. Historical and new checkpoint families must not be pooled.
