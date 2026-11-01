# D1000 recovery status — 1 November 2026

Evaluation is BLOCKED; no active coordinator, new votes or production writes.
Scheduled monitoring remains PAUSED at the owner's request.

## Root cause evidence

Diagnostic28 confirmed HTTP400 for each existing secret. Kaggle's
`/requests/GetUserSecretByLabelRequest` returns an HTTPError that its client wraps
as ConnectionError because URLError is caught before its HTTPError subclass.
The failure is after an internal Secrets request is sent, before any secret value
is returned or public Kaggle API token is used. Internal credential presence is
established by successful client construction; its server-side acceptance is not.
The request shape matches official client code. All three labels fail in this
notebook; token invalidity,429 rate limiting and a platform-wide outage are not
established. Exact400 cause, session/mode/attachment behaviour and protocol
compatibility remain uncertain. No sensitive response body was exported.

## Plan A — BLOCKED

Corrected bounded probe31 completed privately on CPU with exact source verification,
zero Secrets calls, zero judges and zero production writes. Native token-file
presence and successful authentication/private dataset metadata access are confirmed.
Private checkpoint239 is still current. Runtime Kaggle SDK was2.0.2. A TypeError
occurred on the subsequent progress-download call, which used2.2.5 parameters;
no private archive restore or scratch write/readback was established. Independent
code pull confirmed private CPU/no dataset mounts. Launchers29/30 had failed before
Python execution due to missing kernelspec; the notebook metadata was corrected
before31. No repeated secret request or full evaluation was made.

Native authentication alone cannot recover the judge keys. Both Groq and Gemini
remain blocked by their last verified400 secret retrievals. Further Kaggle attempts
stop here. No credentials, internal JWTs, IAP headers or endpoints were modified.

## Plan B — prepared; secure human setup required

`colab_cpu_v1.ipynb` embeds the public, reviewed `runtime.py` adapter. It installs
SDK2.2.5 and CPU judging dependencies, loads credentials once from Colab Secrets,
and preserves the exact supported18 research source on disk. It retrieves private
code/audit sidebands and checkpoint only inside the approved cloud VM, through the
authenticated SDK. No private transfer has started. Colab destination permission
and all three secrets are pending direct human configuration.

The adapter retains original restoration/hash/response/adapter/benchmark checks,
adds exact36/345/32 vote/duplicate checks and the existing uncertain Gemini503 guard,
tests a separate nonce-only private scratch dataset, then creates a private writer
lease before invoking the unchanged evaluation. Existing lease, active Kaggle job,
newer checkpoint, mismatched source, absent credentials or failed persistence all
block judging. Kaggle remains the only persistent research store. No Drive transfer,
training, inference, generation, paid fallback or quota reset is included.

## Evaluation and limits

Last verified private checkpoint239:36/912 Groq,345/912 Gemini,32 paired,912 saved
responses and3 adapters. Existing quota receipts and ledgers stay authoritative:
Groq30RPM/1000RPD/8KTPM/200KTPD; Gemini5RPM/500RPD/250KTPM/1.25MTPD. Actual provider
availability has not been re-probed. Gemini's one transport-uncertain503 with no
durable raw reply is not replayed. DeepSeek remains unavailable and votes remain
missing rather than zero. Final metrics are not claimed.

## Validation and next action

108 targeted tests passed (new recovery safeguards plus D1000 and semantic judge
regressions). All Colab code cells compile, have a Python kernelspec, contain no
stored outputs, and embed the exact reviewed adapter. Cloud end-to-end preflight
and resumed judging remain unverified until secure setup succeeds.

Open the prepared notebook in your own private Colab CPU runtime, enter the three
existing credentials only in Colab Secrets, enable notebook access, and confirm
project permission for private Colab processing before setting the transfer gate.
Run preflight; only READY_NOT_RUNNING permits the sole-writer resume cell. Exact
instructions are in README.md. If project permission is unavailable, stop before
any transfer. Do not paste credentials into chat.
