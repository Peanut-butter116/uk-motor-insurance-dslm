# D1000 single-judge recovery: verified Gemini continuation

The private CPU Colab notebook https://colab.research.google.com/drive/1g_rir-DPeeoZG3fAqNZxoY_JRXuRcbbD resumed the owner-authorised `d1000-single-judge-fast-v1` under the same exclusive writer lease version1. The old scheduled desktop monitor remains PAUSED.

## Actual verified progress

Private checkpoint296 passed archive readback and actual vote-journal auditing:475/912 unique valid judged responses,437 unjudged,67 Groq votes and456 Gemini votes. Six new valid first judgements have been durably saved since this recovery. Historical overlap remains48; duplicate votes remain zero. Selected provider counts are67 Groq and408 Gemini. Archive SHA256:abb587e66bf55bf7371a843ade25724ba135e446c24c658f8f7c36b816aa867b.

Gemini is genuinely producing valid new judgements. One coordinator is active with one worker per provider and shared disjoint claims; Groq independently remains quota-waiting. Checkpoints stop new claims, drain active requests, persist, independently read back and audit actual journal counts. A checkpoint captures zero active requests; the coordinator can resume the next batch after readback.

## Confirmed failure and quarantine

Read-only inspection of both exact attempt journals confirmed HTTP503, not429 or a parsing error. Both original statuses remain `transport_uncertain`, with one attempt each and no durable raw reply or validated cache. Google documents503 as service unavailability; the exact backend cause and whether either request was processed remain unknown: https://ai.google.dev/gemini-api/docs/troubleshooting . No platform-wide outage or credential failure is asserted.

The idle old control cell was interrupted at its sleep boundary after checkpoint293 readback and zero active calls. The observed exit was CancelledError at asyncio.sleep. No provider tasks or completion/persistence threads remained. Controlled suspension checkpoint294 then passed private readback with469 unique votes. A supported read-only model catalogue request passed; generation health was considered unproven until the six new validated results passed checkpoint296 readback.

The tested correction excludes the union of protected unknown response IDs from both provider queues. Neither request is replayed, reassigned, cleared, reclassified or counted as a successful vote. One affected response already has a valid historical Groq judgement; the other has no valid judgement. Its exact reply cannot be recovered from the available durable records. Therefore maximum safe coverage is911/912 unless that outcome is authoritatively recovered. There are436 eligible unjudged responses at checkpoint296.

`continuation.resume_same_runtime` requires a deliberately drained, readback-verified suspension, unchanged private lease/latest checkpoint, immutable files, valid saved votes and calibrations, and zero old provider work. Existing journal/cache/quota files are hash-checked across preflight. The stop marker is archived only after checks pass; no started flag, lease, ledger or campaign clock is reset. The protected-ID/hash audit stays private. A new transport-uncertain failure still stops that provider for the session; there is no automatic outage retry loop.

## Quotas and unchanged deadline

Original launch:1 November2026 19:15:57UTC. Fixed deadline:2 November15:15:57UTC. New calls stop15 minutes earlier to drain and save. No extension has been assumed.

Groq remains30RPM/1000RPD/8000TPM/200000TPD. Its tracked remaining TPD is6343, below the minimum eligible original reservation6519. Recorded next availability is2 November18:40:33UTC, after the deadline. No smaller reservation, truncation or quota reset is introduced.

Gemini retains the stricter5RPM/500RPD/250000TPM configuration. The original configured TPD cap is125000000, not a new independently established service entitlement. Its checkpoint296 conservative ledger has374 current Pacific-day request slots left. The documented RPD reset is2 November08:00UTC: https://ai.google.dev/gemini-api/docs/rate-limits . Reservations, including both unknown requests, are retained. Shared-account/server remaining consumption is not independently known; actual429 pauses remain authoritative.

Measured new-success throughput between readbacks295 and296 is134/hour, including private publish/readback. Whole-campaign throughput, including the blockage, is37/hour. About27/hour is required for the remaining eligible population by the fixed deadline. Rate scenarios with the current quota/reset imply eligible completion around2 November08:30–10:30UTC, conditional on service health, repairs, shared usage and Colab runtime continuity. This is not a guarantee. The complete912 target is not currently feasible because one response remains quarantined;911 is the safe ceiling. The engine's generic remaining-count ETA does not resolve that exception.

## Integrity, validation and reports

137 tests pass across fast orchestration and original evaluation/recovery/semantic validation. Tests include cross-provider quarantine, disjoint claims, protected-byte checks, refusal to resume live workers, preservation of original lease/deadline, rejection of preflight ledger mutation and one-use suspension markers. The prepared fresh-VM renewal notebook embeds the updated reviewed public source/hash bundle and has no saved outputs.

All912 responses, three adapters,80 immutable files, frozen76-question benchmark, original exact judge models, rubric and nine-case calibration requirements remain unchanged. Missing semantic votes are never zero; every available model/condition retains76 questions. DeepSeek stays N/A. Mixed-provider single-valid-judge results are not equivalent to the historical dual-judge experiment. No new paired collection or agreement analysis occurs.

A read-only report cell is queued behind the sole coordinator. Full final comparisons require912 verified valid judgements; a normal partial stop is clearly labelled incomplete. Deterministic and partial semantic comparison JSON/CSV stay private. Only safe code/tests/docs/aggregate receipts are public. No training, inference, QA generation, paid fallback, new account, credential exposure or raw private artifact publication occurs.

Free Colab may terminate before the quota window completes. The prepared renewal path requires a deliberately drained suspension and independently verified private checkpoint before a fresh VM versions the same lease; it preserves the original deadline and all replay protections. Arbitrary VM loss is not treated as proof that unsaved requests were unprocessed. No second VM/coordinator has been started during this recovery.
