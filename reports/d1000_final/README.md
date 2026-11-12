# D1000 presentation reporting package

**Provisional reporting package.** Current verified coverage, snapshot version and execution classification are recorded in `coverage_summary.json`, `comparison.md` and `run_receipt.json`. Incomplete semantic coverage is never labelled full completion.

## Files and source of truth

- `verified_aggregates.json`: checksum-verified, cloud-derived safe input; contains no raw questions/answers or per-request IDs.
- `metrics_summary.json`: canonical long-form metric table plus provenance and validated convenience views. `metrics_summary.csv` is the same long table as CSV.
- `comparison.csv` / `comparison.md`:12 available model-condition rows derived from the canonical table; DeepSeek remains N/A in the machine-readable table.
- `coverage_summary.json`: checkpoint, preserved population, quotas and missingness.
- `sft_effects.json`: common-question paired correctness differences, paired sample sizes and conditional percentile-bootstrap intervals.
- `figures/`: six16:9 figures, each as300dpi PNG, SVG and PDF, plus checksums in `manifest.json`.
- `presentation_summary.md`: slide-ready language for3 November.
- `run_receipt.json`: private snapshot verification and public build/test traceability.

## Reproduce without private data or API calls

```bash
python -m pip install -r reports/d1000_final/requirements.txt
python scripts/build_d1000_results.py
MPLCONFIGDIR=/tmp/d1000-mpl python scripts/plot_d1000_results.py
python scripts/render_d1000_readme.py
python -m pytest -q tests/test_d1000_reporting.py
```

To refresh from a newer private checkpoint, stop/drain the single writer, use `cloud.single_judge_fast.reporting.build(recovery)` only inside the approved private Colab VM, and verify its source/archive/population/journal checks. Export only its safe aggregate return value, verify its transfer checksum, replace `verified_aggregates.json`, and rerun the two local scripts. Never download private archives, raw answers, policy evidence or individual quota entries locally. A new presentation snapshot requires a fresh verified checkpoint; do not hand-edit plotted numbers.

## Definitions and uncertainty

Correctness is the mean0/0.5/1 rubric score across non-null valid semantic judgements. Numerators are score sums, not counts of fully correct answers. Citation support uses non-null scored citation occurrences; resolution uses all emitted citation IDs. Refusal/parse metrics use all 76 saved answers, with 67 answerable and 9 unanswerable questions. Balanced abstention is½(non-refusal rate on answerable + refusal rate on unanswerable). The inherited parse metric recognises a refusal or citation, not JSON compliance. Missing semantic judgements are not zeros. N/A is used for unavailable models, absent student evidence and zero scored support occurrences.

Correctness intervals use2000 deterministic question bootstrap resamples, seed20261102, conditional on available scores. Paired SFT sensitivity intervals resample common scored questions. They do not capture model-training variation, judge/provider uncertainty, nonrandom missingness or establish a universal model ranking. No new LLM requests are used for any report metric.

## Figure guide

1. `01_semantic_correctness`: observed correctness, coverage labels and conditional intervals.
2. `02_sft_effect`: observed Base→SFT changes for correctness and balanced abstention; unequal coverage disclosed.
3. `03_refusal_abstention`: all-response balanced score, unanswerable refusal and false refusal with counts.
4. `04_citation_reliability`: resolution and semantic support shown separately, with denominators.
5. `05_coverage`: valid/other missing/quarantined counts in all 12 available conditions.
6. `06_results_heatmap`: compact descriptive comparison, explicit N/A and correctness sample sizes.

The exact predecessor README and historical evidence are preserved under `archive/d1000-predecessor` and existing experiment directories. Main-branch merge is not performed.
