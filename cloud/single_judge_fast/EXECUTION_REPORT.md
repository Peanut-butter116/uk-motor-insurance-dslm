# D1000 single-judge fast execution —1 November2026

Protocol `d1000-single-judge-fast-v1` is genuinely executing in the owner-approved Restricted private CPU Colab notebook: https://colab.research.google.com/drive/1g_rir-DPeeoZG3fAqNZxoY_JRXuRcbbD . One coordinator holds the existing private lease version1; the old dual-judge coordinator drained to zero workers and final checkpoint249 before transition. The scheduled desktop monitor remains PAUSED.

## Latest observed durable milestone

Private checkpoint257 is independently readback verified; archive SHAf7d337a687d7cf73e82c35e734e670ea532412c52ba4828b9c956875fbcb1296. It contains386/912 unique valid judgements,526 unjudged responses, Groq56 votes and Gemini378 votes. Historical pairs remain48, and duplicate votes remain zero. The protocol has added37 unique valid results. Groq-first deterministic historical selection chooses56 Groq and330 Gemini records. All912 responses,3 adapters,80 immutable files and frozen76-question benchmark remain preserved.

Launch:2026-11-01T19:15:57.112748+00:00; deadline:2026-11-02T15:15:57.112748+00:00 (UK time is UTC on these dates). The clock is never restarted by supported runtime renewal. Snapshot-capture throughput was174.2 new unique results/hour. A more conservative observation rate including startup/readback was127.3/hour. Rates are early measurements, not completion guarantees. The snapshot's quota-aware ETA was about12.8 hours and depends on the next Gemini daily allowance.

## Quotas and Gemini recovery

Groq retains30RPM/1000RPD/8KTPM/200KTPD. At this checkpoint, tracked daily tokens remaining were79068. Gemini retains the stricter5RPM,500RPD,250KTPM settings and its original configured125,000,000TPD cap. The TPD value is existing receipt configuration, not a newly independently verified service entitlement; earlier narrative1.25M/125M abbreviations must not override the actual pinned receipt. Server-side429 and pause evidence remain authoritative. At the snapshot, Gemini had465 tracked RPD slots remaining in the current Pacific day. External shared-account consumption is not independently known.

Google documents Gemini RPD reset at midnight Pacific: https://ai.google.dev/gemini-api/docs/rate-limits . The next reset is2 November08:00UTC, within the original20-hour window. The separately reviewed adapter changes only that accounting window; it does not erase, reset or rewrite existing ledger bytes. RPM/TPM and TPD remain conservatively tracked. Preflight projected sufficient capacity across current and next daily allowances, conditional on successful calls and runtime availability.

Gemini's original transport-uncertain503 remains protected, with no durable reply and no replay. The guard is task/provider scoped, so other independent Gemini tasks are now successfully judged. New failures with uncertain outcomes stop that provider. Shared claims prevent any response already judged or assigned to the other provider from being requested again. No new paired-judgement collection or agreement analysis occurs.

## Method and outputs

This is a mixed-provider single-valid-judge descriptive evaluation, not equivalent to the historical dual-judge design. The original rubric, citation/abstention validators, exact models and both original nine-case calibration receipts are retained. Deterministic parse, citation-ID resolution, gold-label refusal/abstention, coverage and per-model/condition counts use no additional LLM calls. Separate private comparison JSON/CSV and selected-vote records are updated at every checkpoint. DeepSeek remains N/A; full76-question denominators remain and missing semantic votes are not zero.

Active runtime code revision0e4488eb6d077ff65f26f1bb87ba4109729dea4b. An initial Colab asyncio.run loop-entry failure made no judge calls; the supported awaited continuation reused the same untouched lease and quota state. The new131-test code additionally prepares controlled runtime renewal and more conservative reporting. It has not replaced the active worker code while requests are in flight.

## Runtime continuity

Free Colab may terminate before the quota projection completes; https://research.google.com/colaboratory/faq.html documents a maximum12-hour free runtime. No paid resources, keepalive workaround or new account is authorised. The prepared renewal path accepts only a deliberately suspended, fully drained and independently verified checkpoint, verifies the original hashes and private scratch persistence again, and versions the same private lease. It preserves the original deadline, votes, caches, protected histories and quota ledgers. An arbitrary VM loss with possibly unsaved outcomes remains blocked pending reconciliation. The controlled fresh-runtime renewal path is prepared/tested but has not been executed on a second VM.

The full evaluation is INCOMPLETE. No final complete comparison is claimed. Only safe source, tests, documentation and aggregate receipts enter GitHub; raw research assets and credentials remain private.
