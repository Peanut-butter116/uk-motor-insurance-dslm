# D1000 speed-first execution —1 November2026

The owner-authorised `d1000-single-judge-fast-v1` is genuinely executing in the Restricted private CPU notebook https://colab.research.google.com/drive/1g_rir-DPeeoZG3fAqNZxoY_JRXuRcbbD . One coordinator holds the same private lease version1. The old dual-judge worker was drained to zero and final private checkpoint249 independently read back before transition. The old scheduled desktop monitor remains PAUSED.

## Verified progress

Latest observed archive: private checkpoint260, SHAb2d555510d3e8f18fda224f6b048ffc972ada225cb2564cd79eb577e57083c31. Unique coverage404/912; remaining508. Groq58 valid votes; Gemini394. Historical pairs remain48 and duplicate votes remain zero. The fast protocol has durably added55 unique valid judgements. Deterministic selection chooses58 Groq and346 Gemini records.

Launch1 November19:15:57UTC; fixed deadline2 November15:15:57UTC (UK time is UTC on these dates). Snapshot-capture throughput:172.3 new unique results/hour. Conservative observation rate including startup and readbacks:145.5/hour. The snapshot quota-aware projection was about12.7 hours. These early measurements are not a completion guarantee.

Both providers have produced new valid first judgements. Gemini's original HTTP503 transport-uncertain task remains protected and is never replayed. Its task/provider-scoped guard allows independent tasks to proceed. One worker per provider, a shared claim map, existing valid-vote reuse and disjoint queues prevent new paired collection. No inter-judge agreement analysis is generated.

## Quotas and deadline

Groq remains30RPM/1000RPD/8KTPM/200KTPD. Its tracked daily token budget remaining at this snapshot was66039. Gemini remains at the stricter5RPM/500RPD/250KTPM settings, with448 tracked requests left in its current Pacific day. The existing configured TPD cap is125,000,000, not a newly verified service entitlement. Earlier narrative abbreviations do not override the actual receipt. Server remaining/shared-account consumption is not independently known; real429 pauses remain authoritative.

Google documents Gemini RPD resetting at midnight Pacific: https://ai.google.dev/gemini-api/docs/rate-limits . The next reset is2 November08:00UTC. The adapter retains every ledger entry and reservation; only the new protocol's RPD accounting window follows that documented boundary. RPM/TPM and TPD remain conservative. Preflight projected enough free capacity across current and next daily allowances, conditional on actual errors, repairs and runtime availability. No paid fallback, new account or artificial ledger reset occurs.

## Integrity and metrics

All912 responses,3 adapters,80 immutable files and frozen76-question benchmark remain verified. Exact models, rubric, citation support, abstention rules and both original nine-case calibration receipts are unchanged. Groq wins deterministic historical ties; both historical votes are retained. This is a mixed-provider single-valid-judge descriptive evaluation, not equivalent to the historical dual-judge design.

Separate private comparison JSON/CSV, provider assignment and coverage artifacts are updated at each checkpoint. Parse rate, citation-ID resolution, gold-label refusal/abstention, counts and missingness require no additional LLM calls. Full76-question denominators remain; missing semantic votes are not zero. DeepSeek's four conditions remain N/A. The read-only final report cell is queued after the coordinator and prints tables only after a normal readback-verified finish; it marks completion only at912 valid judgements.

## Runtime continuity

The active worker is revision0e4488eb6d077ff65f26f1bb87ba4109729dea4b. Prepared renewal/final-report changes pass133 tests and have not replaced code while requests are in flight. Free Colab may terminate before the quota projection completes; its FAQ documents a maximum12-hour free VM: https://research.google.com/colaboratory/faq.html . No keepalive/resource bypass or paid compute is used.

`lifecycle.suspend` waits for all provider tasks, actual completion threads and checkpoint/publish/readback threads to drain; recovers only fully validated cached successes, then independently persists a suspension proof. `resume.restore` accepts only that deliberately suspended latest checkpoint in a fresh private CPU VM, re-verifies archive/inputs/votes/calibration, tests scratch persistence and versions the same private lease. The original launch time and deadline are preserved. An arbitrary lost VM with potentially unsaved outcomes remains blocked pending reconciliation. The renewal path is prepared/tested, not yet executed on a second VM.

Full evaluation remains INCOMPLETE. Only safe source, tests, docs and aggregate receipts are public. All private archives, policy evidence, benchmark questions, model replies, journals and credentials stay in approved private storage.
