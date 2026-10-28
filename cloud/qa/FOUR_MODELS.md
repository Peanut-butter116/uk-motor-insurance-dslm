# Four distinct generators: guarded upgrade

Owner authorised four distinct models on 28 October 2026. The existing exact
models remain Groq `openai/gpt-oss-120b` and Gemini `gemini-3.5-flash-lite`.
New candidates are Groq `openai/gpt-oss-20b` and Gemini `gemini-3.1-flash-lite`.
The latter replaces the duplicate Flash-Lite request following the owner's
explicit choice of a distinct Gemini model. Public provider documentation lists
these models and free-tier support; this does not establish availability to the
owner's keys. Authenticated live model-list checks in Kaggle are mandatory.

If absent from the authenticated catalogue, explicitly allowlisted free-tier
fallbacks are `qwen/qwen3.8-27b` and `gemini-3.8-flash`, respectively. The actual
selection and substitution are recorded in `four_model_catalogue.json`; an
unsupported original model or absent free fallback stops the upgrade. No aliases,
paid fallbacks or new credentials are introduced. A listing is not a successful
generation; all four workers must admit smoke records before scaling.

Sources checked 28 October:

- https://console.groq.com/docs/models
- https://console.groq.com/docs/rate-limits
- https://ai.google.dev/gemini-api/docs/models
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/rate-limits

## Safe handoff

`four_campaign.run()` loads the existing Kaggle secrets and performs GET-only
model listing requests. It waits read-only for **exact execution14** to finish
with `COMPLETE`; it never cancels it or writes checkpoints while waiting.
An error/cancelled predecessor refuses handoff. After clean completion it loads
the newest private checkpoint, verifies the archive/files, and refuses to start
if any old `active_dispatch` remains unresolved. Thus no fixed starting version
can overwrite progress accumulated during preparation or waiting.

This guarded wait is part of a running Kaggle job, not a recurring scheduler.
It is bounded to five hours; generation then has its own four-hour application
cap. Kaggle's runtime/session limits still apply. No cross-session launch or
secret attachment is claimed. Do not start multiple upgrade jobs.

The known Gemini HTTP503 observed in execution14 is handled only if the cloud
state's exact `gemini HTTP 503` block matches exactly one assigned request's
cache history. The original history and quota reservations are retained in a
private recovery audit. Zero or multiple matches, existing raw responses or
other unknown transport outcomes stop recovery; there is no blanket reset.

## Coordinator and safeguards

The existing `fast_campaign.campaign()` is reused, with four worker specs and
the same source order, original generation prompt, canonical constructor and
deterministic checks. No LLM assessor, judge, answerability verifier or semantic
duplicate model is called. All additions stay unverified. Existing verified and
unverified rows are preserved exactly as record objects, and the original
verified export/benchmark hashes remain checked at every save.

Jobs are assigned by the single coordinator, retain worker identity on restart,
and cannot be assigned twice. New provenance includes provider/model, worker ID,
generation timestamp, source package and request/response hashes. Historical
rows are not rewritten to add new metadata. The same `fast_unverified_v1`
partition and private versioned dataset retain all populations separately from
verified data; per-model counters distinguish their origins.

Four asynchronous workers share two provider ledgers and HTTP clients. Each
provider starts with at most two simultaneous requests. A quota collision drops
its next cohort's request semaphore to one; both model workers remain eligible,
with rotated allocation. The other provider continues independently. More
workers do not create additional quota. Existing account/transfer attestations
and conservative provider-wide limits are retained; model catalogue evidence is
stored separately rather than fabricating a new model's synthetic probe result.

Smoke assigns one distinct package per worker; afterward small cohorts assign
two per worker, at most eight packages. Recoverable short quota waits occur
inside the cohort, avoiding two full archive uploads for every minute-window
pause. Provider semaphore limits and durable quota reservations still apply.
Long quota waits return to the coordinator. Known HTTP503 replies get at most
two retries with exponential backoff and additional quota reservations; unknown
transport errors are not blindly retried.

Exclusive assignments persist before dispatch; completed cohorts persist at a
quiescent barrier. No provider thread writes while the checkpoint is archived.
Accepted/rejected work and cached responses live in cloud scratch during a
cohort, then in the private dataset after verified publication. Interruption can
lose an in-flight cohort; unresolved requests are not silently regenerated.

Metrics include per-worker generated/admitted/failed, completion attempts,
admissions/hour, request durations, quota pauses and worker wait seconds;
provider rolling60s/24h request/token reservations; aggregate cumulative and
upgrade-only admissions/hour. Failed counts include technical exclusions and
failed package requests, so units are documented rather than equated with a
quality-judge rejection rate. Concurrent worker waits can overlap.

## Start and monitor

Stage with `python cloud/qa/stage_four.py --output /tmp/motor-four-stage`, then
save privately without running with `kaggle kernels push -p /tmp/motor-four-stage
--no-run`. Saved version15 contains this upgrade. In the existing editor keep
the three existing secrets selected and choose **Save Version → Save & Run
All**. Confirm **Four distinct generators — guarded handoff, UNVERIFIED**.
The cloud entry may safely wait while execution14 remains active.

While a newer saved version or waiting job exists, monitor the predecessor with
`fetch_campaign_receipts.py --execution-version 14 --output /tmp/motor-fast-v14`.
Do not infer old execution completion from the notebook's latest saved version.
After the new execution is confirmed, monitor its exact version. Root accepted
counts remain the legacy verified population; `fast_campaign` contains combined
counts, per-model/worker metrics and `four_smoke_passed`. No four-model cloud
smoke success or throughput improvement is claimed until measured in its
persisted receipts. The comparison baseline is approximately81 admissions/hour.
