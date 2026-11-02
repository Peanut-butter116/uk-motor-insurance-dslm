| Model | Condition | Correctness¹ | Judged / 76 | Balanced abstention² | Citation support³ | Citation-ID resolution | Parse⁴ | False refusal² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B | Base | 13.2% | 76/76 | 49.3% | N/A | N/A | 25.0% | 1.5% |
| Qwen2.5-3B | Base + RAG | 46.9% | 64/76 | 50.0% | 57.2% | 88.9% | 69.7% | 0.0% |
| Qwen2.5-3B | SFT | 20.3% | 74/76 | 50.0% | N/A | N/A | 43.4% | 0.0% |
| Qwen2.5-3B | SFT + RAG | 45.2% | 63/76 | 58.9% | 75.4% | 92.4% | 92.1% | 4.5% |
| Ministral-3B | Base | 18.4% | 76/76 | 45.4% | N/A | N/A | 36.8% | 31.3% |
| Ministral-3B | Base + RAG | 58.8% | 57/76 | 60.4% | 91.4% | 81.2% | 75.0% | 1.5% |
| Ministral-3B | SFT | 15.1% | 76/76 | 54.1% | N/A | N/A | 89.5% | 80.6% |
| Ministral-3B | SFT + RAG | 62.7% | 67/76 | 83.7% | 90.5% | 100.0% | 96.1% | 10.4% |
| SmolLM2-1.7B | Base | 9.3% | 75/76 | 50.0% | N/A | N/A | 15.8% | 0.0% |
| SmolLM2-1.7B | Base + RAG | 20.0% | 70/76 | 50.0% | N/A | 100.0% | 1.3% | 0.0% |
| SmolLM2-1.7B | SFT | 14.7% | 75/76 | 50.0% | N/A | N/A | 9.2% | 0.0% |
| SmolLM2-1.7B | SFT + RAG | 40.0% | 60/76 | 48.5% | 66.7% | 83.3% | 71.1% | 3.0% |

¹ Mean rubric score (0 / 0.5 / 1), conditional on valid non-null semantic judgements; missing is not zero. See canonical table for score sums, denominators and conditional bootstrap intervals.
² Deterministic metrics use all 76 saved responses: 67 answerable and 9 unanswerable. Balanced abstention = ½(answerable non-refusal rate + unanswerable refusal rate).
³ Citation support uses only supplied, scored citation occurrences; denominators vary. No-RAG is N/A. SmolLM Base+RAG has one resolving ID but zero scored support occurrences, so support is N/A.
⁴ The inherited parse metric recognises a refusal or a parseable citation. It is **not** a measure of structured JSON generation compliance.

DeepSeek’s four conditions are unavailable (N/A), excluded from the 912-response population.
