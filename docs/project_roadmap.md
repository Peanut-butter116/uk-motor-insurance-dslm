# Research roadmap

| Period | Planned milestone |
|---|---|
| 18–20 October | Verify completed results, snapshot data, prepare supervisor deck and GPU compatibility checks |
| 21 October–6 November | Complete judging, independently review motor dev data, screen six candidates, expand towards 1,000–2,000 accepted records |
| 7–18 November | Finish human review and deduplication, freeze at least 2,000 accepted records; 3,000 stretch |
| 19 November–4 December | Full QLoRA SFT, fixed-retrieval four-arm comparison and uncertainty/error analysis |
| 5–18 December | Final model/adapter, reproducible code, research report, presentation and handover |

The current 409-record snapshot is an interim acceptance checkpoint, not a completed training dataset. Production still records the previous 10,000 target. Repository B adopts the revised targets without editing live production configuration.

Immediate dependencies: independent review of the 20 motor dev drafts, broader topic/source coverage, 850 existing correctness tasks, private-upload clearance for any new restricted Kaggle dataset, and successful GPU compatibility checks. Do not use frozen test performance to adjust prompts or select models.
