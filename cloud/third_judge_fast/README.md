# D1000 third-judge recovery v1

Separate protocol `d1000-single-judge-third-recovery-v1`, extending the authorised mixed-provider single-valid-judge evaluation. Original code, fixture labels, rubric, model responses, journals and benchmark remain unchanged.

The existing Colab coordinator must be fully drained and privately checkpointed by `lifecycle.suspend`. `prepare` reuses the same verified writer lease and original deadline. It rejects any changed historical unknown population. Historical Groq then Gemini votes take precedence; the third judge evaluates only globally uncovered responses outside the union of protected requests.

Candidate selection on 2 November 2026: the existing Groq key catalogue does not contain `llama-3.3-70b-versatile`; the existing Gemini key catalogue contains `gemini-3.1-flash-lite`. The existing project's AI Studio **Free tier** dashboard shows Gemini 3.1 at **15 RPM / 250K TPM / 500 RPD**, zero usage in the selected one-day window. Dashboard data can lag 15 minutes. The implementation conservatively uses the **same original Gemini account quota ledger**, retaining every historical reservation and its existing 125M-token daily conservative cap. This does not assert an independent extra project quota or a newly verified TPD allowance.

`adapter` explicitly binds the new model and observed limits without changing the original provider verification receipt. Original `judge_recovery.complete` supplies exact task/model cache identities, durable request histories/raw replies, bounded same-judge repair and strict original schema/citation validation. New caches and vote journals live separately under `third_judge_fast_v1`; all calls still reserve against the original Gemini ledger.

All nine original fictional calibration cases must pass, and their actual receipt/cache must be privately checkpointed and read back before research. One call measures latency, then at most two disjoint calls run concurrently. Quota pauses or uncertain outcomes stop dispatch and drain the bounded batch before checkpointing. Every four successful votes triggers a private publish/readback plus full third-judge file hash audit and unchanged historical vote audit.

Results retain exact provider/model/protocol identity. Tables include all 16 conditions, four unavailable DeepSeek conditions, 76-question denominators and explicit semantic missingness. Figures and raw records remain private. No cross-judge agreement claim is made.

Official references: [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), [Gemini 3.1 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite), [OpenAI-compatible transport](https://ai.google.dev/gemini-api/docs/openai).
