# Third-judge recovery — blocked by calibration, 2 November 2026

The old Colab coordinator was interrupted only while idle with zero provider calls and both workers blocked. Controlled suspension drained all workers and independently read back private checkpoint **411**, retaining the same authoritative lease version **2**. No second writer was created.

The existing Groq key's model catalogue does **not** contain `llama-3.3-70b-versatile`; no generation probe was spent on that unavailable candidate. The existing Gemini key's catalogue contains `gemini-3.1-flash-lite`. AI Studio reports the project as Free tier and that model's limits as **15 RPM / 250K TPM / 500 RPD**, with 0 usage in the selected one-day window. The recovery implementation nevertheless shares the entire existing Gemini ledger and its conservative daily token cap; it neither resets reservations nor assumes another independent project-wide budget.

**Gemini 3.1 calibration failed.** Eight original fictional cases passed. The missing-reference case exhausted the unchanged three-attempt bound with strict validation categories `Missing evidence cannot be scored` (twice) and `Material error must score zero` (once). Eleven total calibration requests produced eight accepted calibration results and one exhausted invalid case. No research judgement was requested, no new unknown outcome occurred, no fixture label/rubric/schema was changed, and no failed calibration was treated as a pass.

Private checkpoint **412** was independently read back and audited in the Colab VM, including every third-judge file hash and unchanged historical journal counts. A transient browser connection loss during publication recovered; both the Colab archive receipt and a separate metadata/aggregate-progress check confirm version 412. Archive SHA-256:

`c2fa6663acc4e85b93c52c30103cf852ae163c945a00d011e497f196c8eaf320`

| Aggregate | Verified state |
|---|---:|
| Saved responses | 912 |
| Unique valid semantic judgements | 774 / 912 |
| Historical Groq votes | 67 |
| Historical Gemini 3.5 votes | 755 |
| New third-judge research votes | 0 |
| Remaining eligible responses | 135 |
| Quarantined unjudged responses | 3 |
| Historical protected Gemini requests | 4 |
| New transport-uncertain requests | 0 |
| Active research coordinators / provider calls | 0 / 0 |
| Authoritative writer lease | Existing version 2, retained |

All 912 responses, three adapters, frozen benchmark/80-file immutable manifest, completed votes, private caches/journals and historical quota reservations remain intact. Three unjudged responses cannot be reassigned under the protected-request rule; the fourth protected request already has a historical valid vote. Maximum eligible coverage without authoritative unknown-outcome recovery is therefore **909/912**.

The deadline remains **2 November 2026 15:15:57 UTC**. No calibrated usable alternative is currently available, so no supported completion projection can be given. Original Groq's recorded reset is **18:40:33 UTC**, after that deadline. Do not repeatedly relaunch either alternative or reset the calibration attempt bound. The aggregate comparison/figure generator is prepared; final complete results are not claimed while 138 semantic judgements are missing.

Validation: **145 tests passed**, including historical precedence, global unknown-task exclusion, exact model binding, reuse of the original quota path, strict refusal calibration and no-call failure sealing. Only code and safe aggregate evidence are published; raw records, quota entries and credential values remain private.
