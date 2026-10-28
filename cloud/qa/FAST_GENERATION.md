# Opt-in accelerated campaign — unverified

The owner authorised this methodology change on 28 October 2026. It is a
separate experimental population, not an extension of independently verified
training data. `campaign.py` / `runner.py` remain the legacy verified workflow.

`fast_campaign.py` loads the newest private production checkpoint, verifies its
archive and every file hash, and preserves the verified `data/final/train.jsonl`
and protected benchmark hashes. The existing source selector, canonical row
constructor and original `TEACHER_PROMPT` are reused for both existing models:
Groq `openai/gpt-oss-120b` and Gemini `gemini-3.5-flash-lite`.

There are **no LLM source assessments, independent solves, answer audits,
semantic duplicate comparisons or full-document absence judgements** in this
path. Existing deterministic schema/source/citation, lexical benchmark/section/
supporting-page exclusions and normalised question deduplication remain. These
checks do not establish factual correctness or genuine unanswerability.

New canonical rows use `validation_status=schema_checked`,
`validation_method=deterministic_only_no_independent_verification`, and
`provenance.verification_status=unverified`. They retain the actual generator
model, provider, request/response hashes and package ID. Historical rows are not
relabelled. This data is not automatically training-ready; the historical SFT
loader is unchanged.

Private partition: `production-v1/fast_unverified_v1/` inside each versioned
checkpoint archive. It contains `train.jsonl`, `excluded.jsonl`, `state.json`,
`manifest.json` and aggregate `progress.json`. The original verified export is
separate and unchanged. Root aggregate progress contains a `fast_campaign`
object with verified, unverified, per-generator and combined counts.

The coordinator exclusively assigns distinct remaining Hastings packages. Each
provider has one asynchronous worker backed by a reused HTTP client. Smoke mode
assigns one package each; after both have admitted records and overlapping
request intervals are observed, waves assign up to five packages per provider.
The original durable request caches, structured retries and free-tier account /
quota receipts are reused. Account/terms receipt checks are local authorisation
checks, not QA judgements. No paid route exists.

Quota pauses are independent: a paused worker returns control while the other
finishes its assigned work. Quiescent barriers publish exclusive assignments
before requests and completed results afterward, with authenticated cloud
streaming verification before more calls. No archive is taken during request
writes. A killed session with dispatched but unpersisted responses quarantines
those source jobs rather than regenerating them, and conservatively blocks the
affected provider for 24 hours to retain unknown quota usage. The archived state
records these interruptions. This trades some lost in-flight work for safe
resumption without duplicated requests or rate-limit bypass.

Execution remains in Kaggle, with the existing three Kaggle secrets, CPU only
and a four-hour application session cap. The job stops at 1,000 combined records,
source exhaustion, session cap or an unresolved operational blocker. Persistent
data stays private on Kaggle; local tests use fictional records only.

Stage code with `python cloud/qa/stage_fast.py --output /tmp/motor-fast-stage`,
then `kaggle kernels push -p /tmp/motor-fast-stage --no-run`. In the existing
notebook editor confirm the **Parallel Groq + Gemini generation — UNVERIFIED**
heading and the existing three secret selections, then **Save Version → Save &
Run All**. Do not start it alongside the legacy writer. API starts previously
failed to attach secrets, so no cross-session generation scheduler is claimed.

Cloud execution emits only aggregate `fast_campaign_report.json` and
`fast_persistence_receipt.json` as monitoring receipts. Root private checkpoint
`progress.json.fast_campaign` supplies live monitoring without downloading rows.
Do not infer execution, smoke success or throughput from code upload alone.
