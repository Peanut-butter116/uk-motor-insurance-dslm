# Supervisor meeting figures

Seven presentation figures, each supplied as a **2560 × 1440 PNG** and an
editable-text SVG. Snapshot date: **11 October 2026**.

All figures use the public [README at `1e1ff8d`](https://github.com/Peanut-butter116/uk-motor-insurance-dslm/blob/1e1ff8d2d2875c0e19baed98964d20c865bda57b/README.md).
The complete source commit and README SHA256 are recorded in
[aggregate_values.json](aggregate_values.json). The figures are a dated snapshot,
not a live dashboard. New-model correctness, composite and faithfulness remain
pending. E3/E4 generation is complete: **140 answers each**.

The generator reads only the public aggregate configuration in this folder. It
does not import experiment modules, access private paths, contact Kaggle, read
credentials or consume case-level records. Diagram narrative describes the
public experiment design and proposed next-stage options; it is not a measured
performance claim. No protected policy text, question, answer, training example,
adapter, judge task or private log is included.

## Figures and provenance

| Figure | What it shows and public sources | Pending / interpretation | Files |
|---|---|---|---|
| 01 — Project extension | README research aim, inherited pipeline, 140-question benchmark; historical composites **0.608431** and **0.599115**; practical 3B prototype and later 7B validation. | Error/faithfulness analysis and 7B validation are future work; does not claim smaller is better. | [PNG](01_project_extension.png) · [SVG](01_project_extension.svg) |
| 02 — Experiment design | README controlled 3B extension: E1–E4, same **140** questions, frozen evidence, prompts and evaluation; highlights E2/E4. | Correctness comparison pending; discloses the prepared-base dtype difference in tuned inference. | [PNG](02_experiment_design.png) · [SVG](02_experiment_design.svg) |
| 03 — Week 2 progress | README progress, **21** SFT examples, failed Smoke v1/v2 gates, Smoke v3 pass, full-training pass and **140 + 140** E3/E4 outputs. | Completed failed smokes are labelled failed, not successful. Correctness and faithfulness judging remain pending. | [PNG](03_week2_progress.png) · [SVG](03_week2_progress.svg) |
| 04 — Reload validation | README Smoke v2 diagnosis and Smoke v3 pass: maximum differences **0.468750**, **0.119458**, **0.000000**; means **0.076282**, **0.015449**, **0.000000**; unchanged atol **0.02**, rtol **0.01**. | Engineering validation, not a benchmark quality result. Values rounded to six decimals; full adapter also reloaded exactly. | [PNG](04_reload_validation.png) · [SVG](04_reload_validation.svg) |
| 05 — Current results | README full-benchmark RAG rows, reproduced below. Includes newly measured E4 alongside the four requested reference rows. | New correctness/composite shown as **Pending**, never zero. No winner declared. | [PNG](05_current_results.png) · [SVG](05_current_results.svg) |
| 06 — Next-step decision | Public research direction: use paired E2/E4 errors to consider grounded SFT expansion, DPO, targeted refusal data, robustness and later 7B validation. No new numerical result. | Conditional proposals only. Complete judging first; no next experiment is selected or implemented. | [PNG](06_next_step_decision.png) · [SVG](06_next_step_decision.svg) |
| 07 — Research contribution | README inherited assets, new model support, **21** SFT examples, leakage/reload controls, E2/E4, **76** motor questions and completed **140 + 140** outputs. | E3/E4 moved from “Next” to completed work after verification. Judging, semantic paired analysis and adaptation choice remain next. | [PNG](07_research_contribution.png) · [SVG](07_research_contribution.svg) |

## Exact chart values

These are the six-decimal public README values; chart labels use three decimals
for slide readability. All rows below use the full frozen 140-question benchmark.

| Model + RAG | Correctness | Citation resolution | Balanced abstention | Composite |
|---|---:|---:|---:|---:|
| Qwen2.5-7B · Base + RAG | 0.388771 | 0.897872 | 0.723421 | 0.608431 |
| Qwen2.5-7B · QLoRA-RAFT + RAG | 0.348093 | 0.919463 | 0.746148 | 0.599115 |
| Qwen2.5-3B · Base + RAG | Pending | 0.852761 | 0.522727 | Pending |
| E2 · Ministral-3B · Base + RAG | Pending | 0.842697 | 0.596687 | Pending |
| E4 · Ministral-3B · QLoRA-SFT + RAG | Pending | 0.914729 | 0.696071 | Pending |

Citation resolution is not semantic faithfulness. The historical composite is
0.5 × correctness + 0.3 × citation resolution + 0.2 × balanced abstention.
The motor-only results and closed-book diagnostics remain in the root README's
single complete comparison table, rather than being mixed into the main chart.

## Regeneration

From the repository root, in a separate plotting environment:

```bash
python -m venv /tmp/insurance-figures
/tmp/insurance-figures/bin/python -m pip install matplotlib==3.11.2
/tmp/insurance-figures/bin/python archive/2026-10-11/scripts/create_progress_figures.py
```

The script overwrites only the 14 named figure images. It uses DejaVu Sans,
16:9 canvases, 160-dpi PNG and SVG with editable text. Numeric configuration is
public and manually auditable. When updating a future snapshot, first publish
the verified aggregate results in the root README, then update the configuration
and relevant status/narrative in the script. Never substitute zero for null scores.

All seven PNGs were visually inspected, including the corrected timeline and
decision-tree connectors. SVGs use the same drawing code and were parsed for
validity. The existing **199 tests pass**; all **42 protected input files** and
protected benchmark/training/evaluation code/configuration hashes are unchanged.
