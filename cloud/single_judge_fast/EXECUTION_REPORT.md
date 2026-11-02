# D1000 fresh-VM recovery —2 November2026

Evaluation-only execution genuinely resumed in the same Restricted private CPU notebook https://colab.research.google.com/drive/1g_rir-DPeeoZG3fAqNZxoY_JRXuRcbbD . Private checkpoint371 independently passed readback and actual journal-count auditing:585/912 unique valid judged responses,67 Groq votes and566 Gemini votes. Six new valid first judgements were durably saved after this fresh-VM recovery. Remaining327 responses comprise325 eligible and two quarantined. Historical overlap remains48 and duplicate votes remain zero. Selected provider counts are67 Groq and518 Gemini.

Checkpoint371 archive SHA256:112cde02f3328641f16d396c1556d5afd46cdd47e76c0abd1c635ef6a765d883. One coordinator is active under version2 of the existing private writer lease. The old scheduled desktop monitor remains PAUSED.

## Observed interruption and reconciliation

The old browser output ended at readback-verified private369 with579 unique votes,67 Groq and560 Gemini votes, zero active provider calls, Gemini stopped after a third transport-uncertain outcome, and Groq quota-waiting until2 November18:40:33UTC. Colab had disconnected. Reconnection allocated a fresh CPU VM with no old recovery object, research root or writer lock; its session inventory independently showed only that fresh current session, with no other sessions. The precise reason the former VM ended is not established.

This was not treated as permission to replay possibly unsaved work. `idle_loss.validate` admits only a narrowly proved idle state: exact reviewed engine/protocol hashes, latest private checkpoint readback, zero active calls, Gemini's worker already stopped, and Groq's resume time beyond the original deadline. In the exact reviewed worker implementation neither provider could submit any later request. Any active call, running/quota-paused Gemini worker, earlier Groq availability, unreviewed source, missing retired-VM evidence or changed lease fails closed. Arbitrary VM loss remains blocked.

Fresh restoration pulled the exact original18 source, verified the369 archive and frozen inputs, validated all saved votes/calibration and unchanged quota ledgers, verified the old lease owner, and completed a new private scratch upload/independent checksum readback. It then versioned the same lease to2 and acquired a new exclusive local lock in the fresh VM. The original launch time/baseline/deadline were preserved. The private reconciliation proof was persisted in initial checkpoint370 before any new judging. No second coordinator ran.

Kaggle's SDK can authenticate during import; Colab's existing secrets are now loaded before that import on a fresh VM. No token value is printed, embedded or written into artifacts. Storage authentication and private production writes passed in this VM; the old Kaggle internal Secrets service was not used or probed again.

## Unresolved requests

The restored journals independently confirm three Gemini HTTP503 outcomes, each still transport_uncertain with no durable raw reply or validated cache. Exact server processing outcomes remain unknown. Google documents503 as service unavailability, not proof that a request was unprocessed: https://ai.google.dev/gemini-api/docs/troubleshooting . No credential failure,429 quota exhaustion or platform-wide outage is asserted.

All three stable request identities, statuses and reservations remain preserved. Their response IDs are quarantined across both providers. One affected response already has a valid historical Groq judgement; two remain unjudged. No replay, reassignment, deletion or reclassification occurs. Current safe coverage ceiling is910/912 unless exact outcomes are authoritatively recovered. A new transport-uncertain outcome still stops that provider for the session.

## Quotas, measurement and fixed deadline

Original protocol `d1000-single-judge-fast-v1` launch:1 November19:15:57UTC. Fixed deadline:2 November15:15:57UTC. New requests stop15 minutes earlier to drain/persist. No extension or reset is assumed.

Gemini's documented Pacific-day allowance reset at2 November08:00UTC. Fresh preflight recorded500 tracked request slots available; checkpoint371 has490 remaining. Limits remain the stricter5RPM/500RPD/250000TPM; the original configured TPD cap125000000 is not a newly established service entitlement. Next RPD reset is3 November08:00UTC. Existing entries and unknown reservations remain in the ledgers; no ledger reset occurred. Shared-account/server remaining consumption is not independently known, and actual429 pauses remain authoritative. Source: https://ai.google.dev/gemini-api/docs/rate-limits .

Groq remains30RPM/1000RPD/8000TPM/200000TPD, with6343 tracked daily tokens remaining and a minimum eligible original reservation6664. Recorded availability2 November18:40:33UTC is after the fixed deadline. No smaller reservation, truncation, paid fallback or account change is used.

The first post-resume readback interval370→371 yielded six new successes at114.8/hour including publish/readback. About68/hour is needed for eligible coverage by the deadline, or about70/hour allowing the final15-minute drain. Throughput is provisionally sufficient, but the first six successes consumed ten tracked request slots. At that small-sample repair ratio490 slots would yield about294 additional successes, below325 eligible remaining; repair rates may vary, so full eligible coverage is not promised. Any time estimate must be conditional on repairs, provider availability and actual quotas. The generic whole-campaign ETA includes the overnight blockage and is not a credible active-service forecast. Full912 is presently infeasible because two responses remain quarantined.

## Integrity and validation

140 tests pass across fast orchestration and original evaluation/recovery/semantic validators. New tests cover the narrowly admitted idle-loss proof, refusal of every possible post-checkpoint request case, unchanged original clock, old lease ownership, and secure credential acquisition before autoauth SDK import. The prepared renewal notebook embeds the updated public source/hash bundle and has zero saved outputs. Its default entry still requires deliberate suspension; the idle-loss path is selected only after independent termination and exact source/state evidence.

All912 responses, three adapters,80 immutable files and frozen76-question benchmark verified. Exact judge models, prompts, rubric, strict response schema, citation/refusal rules and both original nine-case calibration requirements are unchanged. Existing valid votes are reused; shared claims prevent new paired collection and duplicates. DeepSeek remains N/A. Missing semantic votes are never zero, and each available model/condition retains76 questions. Results remain mixed-provider single-valid-judge descriptive results, not full dual-judge equivalence or agreement analysis.

A read-only completion-gated report cell is queued behind the sole coordinator. A normal partial stop is labelled incomplete; full final comparisons require912 valid judgements. Private partial comparison artifacts continue persisting at checkpoints. Only safe code/tests/docs/aggregate receipts enter GitHub; raw archives, evidence, questions, responses, journals and credentials stay in approved private storage. No training, inference, QA generation, paid compute or new account was started.
