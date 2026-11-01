# D1000 secure CPU recovery v1

This adapter is separate from the audited Kaggle pipeline. It never trains, infers,
generates QA, changes the rubric, or replaces a judge. It uses only the existing
Groq GPT-OSS-120B and Gemini 3.5 Flash-Lite protocol and original quota ledgers.

## Current evidence and remaining gates

Execution28 returned HTTP400 for all three secret labels. The runtime confirmed
HTTPError subclasses URLError, and the URLError handler precedes HTTPError. Thus
HTTP errors are wrapped in ConnectionError before the HTTP-specific branch runs.
The endpoint is `/requests/GetUserSecretByLabelRequest`, with JSON `Label` and the
runtime-managed authorization headers. Client construction succeeded, so an
internal runtime credential was present; server acceptance of that credential is
not proven. Failure occurred after sending the internal Secrets request and before
retrieving a value or authenticating the public Kaggle API. The replacement token
contents were never tested by that failure.

All three configured labels fail in this notebook. This does not establish that
all users or notebook modes are affected. Attachments were visibly enabled, but a
saved API execution's session/attachment authorization could still differ from
an interactive editor. The runtime source has the same exception-order defect as
the official source. A protocol/version mismatch or session issue is possible;
no evidence identifies either as the specific cause of400. Do not patch internal
JWTs, IAP headers, endpoints, or authorization. No raw sensitive body is exported.

Official source:
https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_web_client.py
https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py

SDK2.2.5 `get_access_token_from_env` supports a notebook token file through the
environment variable **KAGGLE_API_V1_TOKEN** (its Python constant is named
KAGGLE_API_V1_TOKEN_PATH). This can avoid the explicit Kaggle Secrets lookup, if the
file exists and its permissions include private dataset writes. It cannot supply
the Groq/Gemini keys. `native_probe.py` makes no Secrets calls and tests private
metadata/progress plus nonce-only private scratch write/readback if native auth
exists. It never writes the production dataset. A missing token fails closed.
Launchers29/30 failed before Python execution because their notebook metadata
lacked a kernelspec. The corrected31 completed: native token file present,
authentication completed and private results metadata readable at239. Runtime SDK
was2.0.2; the progress download then raised TypeError before scratch creation.
The2.2.5 download interface used by the probe is not verified on2.0.2. Thus archive
restoration and scratch write/readback remain unverified; do not retry merely to
repeat Secrets400. The Colab environment explicitly pins2.2.5. See native31_receipt.json. No additional Secrets requests are needed to reproduce28.

## Colab preparation: exact minimal user actions

1. Open `colab_cpu_v1.ipynb` in **your own Google Colab account**. Keep the notebook
   private. Select Runtime > Change runtime type > CPU / hardware accelerator None.
   Do not purchase compute. Use the interactive Colab notebook, not remote workers.
2. In Colab's left **Secrets** panel, configure these three names using the existing
   credentials directly in Colab: `KAGGLE_API_TOKEN`, `GROQ_API_KEY`, `GEMINI_API_KEY`.
   Enable access for this notebook. Credentials are not transferred automatically
   from Kaggle. Never put values in a cell, output, GitHub, or chat. If creating a
   secret requires human credential entry, the human performs it.
3. Confirm the project's confidentiality rules permit the existing private239
   archive to be processed in your private Colab VM. Only then set
   `PRIVATE_COLAB_TRANSFER_APPROVED = True` in the preflight cell. Until this is
   confirmed, the code stops before credentials or private downloads. No Drive
   storage or new storage destination is used: Kaggle remains authoritative.
4. Run the installation/bootstrap and **preflight** cells. Preflight securely uses
   Colab userdata, reads private Kaggle notebook18 code, validates exact source SHA,
   restores exact239 in the cloud via the authenticated HTTPS SDK, verifies the
   archive,912 responses, three adapter hashes, all80 immutable files/frozen
   benchmark, saved36/345/32 votes, private audit sidebands and Gemini unknown503.
   It verifies original code/rubric hashes and tests a separate private nonce-only
   scratch dataset write/readback. Any failure stops before judge calls.
5. Only a `READY_NOT_RUNNING` preflight permits the final resume cell. It creates
   a fixed private writer lease; an existing lease fails closed. Do not run the
   Kaggle coordinator or a second Colab notebook. The wrapper rechecks the lease,
   Kaggle execution status and production dataset version at each original
   checkpoint. On interruption, reconcile the latest checkpoint and lease before
   another start; never remove a lease or reservations merely to force a restart.

The notebook contains only public code. Private notebook18 carries the exact
private calibration/quota sidebands; these are pulled **inside Colab only** and
must never be added to the public notebook or source. No private artifacts are
transferred through this desktop. Raw artifacts, journals and logs stay in the
private cloud runtime and private Kaggle dataset. Authentication uses supported
SDK APIs and Colab Secrets only. Encryption is HTTPS with normal certificate
verification. Project permission and account privacy must be confirmed first.

## Preservation and execution

`runtime.py` compiles a strictly matched copy of the original `restore` function,
removing only its Kaggle Secrets credential prologue. It supplies credentials via
Colab userdata and keeps all original restore validation. Audited source files on
disk remain unchanged, so the original evaluation's code-fingerprint checks still
run. The original `main` handles calibration, quota accounting, cached completions,
unknown outcomes, concurrent providers, metrics and private checkpoints. The
Gemini uncertain503 remains blocked unless separately reconciled from an exact
durable reply; Groq may proceed independently. Original120-second checkpointing
and mandatory calibration persistence barriers remain in force. A Colab session
ending can interrupt work; do not represent prepared code as durable new results.

Preflight is deliberately pinned to239 and its last verified counts. If Kaggle
reports a newer checkpoint, stop and reconcile it rather than selecting239 or
silently discarding newer work. Each successful original checkpoint writes the
whole journal/cache/ledger state and independently reads it back. The receipt
contains aggregate status only. Do not download private raw outputs locally.

Free limits remain Groq30RPM/1000RPD/8KTPM/200KTPD and
Gemini5RPM/500RPD/250KTPM/1.25MTPD, subject to the preserved actual receipts.
No paid fallback or quota reset is enabled. DeepSeek remains unavailable. Missing
votes remain missing; all76-question denominators and historical metrics remain.

Official Colab Secrets implementation:
https://github.com/googlecolab/colabtools/blob/main/google/colab/userdata.py
Colab runtime/privacy/resource limits:
https://research.google.com/colaboratory/faq.html
SDK authentication reference:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md

## Validation

Run `python -m pytest -q tests/test_d1000_recovery_runtime.py` and existing semantic
and D1000 recovery tests. Tests cover chained HTTP statuses without sensitive
messages, transfer/credential gates, exact-source/path rejection, narrow restore
adaptation, duplicate-vote rejection and preservation of the uncertain request.
Cloud end-to-end restoration and persistence remain unverified until Colab is
securely configured and preflight actually succeeds.
