# D1000 recovery status — 1 November 2026

Colab recovery succeeded and evaluation-only execution genuinely resumed. Three new successful Groq votes are present in independently readback-verified private checkpoint243. The same exclusive writer lease remains held by one active coordinator. Scheduled monitoring stays PAUSED at the owner's request.

## Kaggle diagnosis

Diagnostic28 confirmed HTTP400 for all three secret labels at `/requests/GetUserSecretByLabelRequest`. The internal client catches HTTPError as URLError and wraps it as ConnectionError. Exact400 cause remains unknown; invalid user tokens,429 and a platform-wide outage are not established. No further full Kaggle evaluation was launched against the failing service.

## Colab recovery evidence

Private Restricted CPU notebook: https://colab.research.google.com/drive/1g_rir-DPeeoZG3fAqNZxoY_JRXuRcbbD . The owner approved private Colab processing and access to all three existing secrets. All three actual retrievals PASS without credential output. Kaggle SDK2.2.5 authenticated the dataset owner and restored239 inside the private VM: archive hash,912 responses,3 adapters,80 immutable files, frozen benchmark,36 Groq/345 Gemini/32 paired votes and the unresolved Gemini503 all verified.

A separate private nonce scratch dataset was created, independently downloaded and checksum verified. The private exclusive lease version1 and owner checksum were verified. The same local exclusive lock remains held; no second lease or writer was started.

## Reviewed compatibility and freshness repairs

The Colab adapter uses a writable private cloud input folder, exact archived transport source/component hashes and supported SDK2.2.5 downloads. Original research source18 remains unchanged. An initial missing Kaggle mount failed before checkpoint creation or workers; continuation permits only that exact failure, same lease, unchanged journals and no production checkpoint.

The transport account receipt was over seven days old. Current account UI confirms Groq Free with30RPM/1000RPD/8KTPM/200KTPD and Gemini Free with15RPM/500RPD/250KTPM. Existing stricter Gemini5RPM/1.25MTPD project caps remain unchanged. Live catalogue GETs verify both exact models, and the original9-case calibration receipts pass unchanged. Current primary data-use documentation was reviewed, including Google's UK/EEA regional exception. Re-verification preserves the original private receipt and records an old/new hash transition permitting only the verification date to change. All quotas, reservations, caches and votes remain byte-identical; the original seven-day expiry guard is retained. No inference calls were made for this re-verification.

Recovery helper revision36a023c48dd9511f326104fbc458fd6b9033e0e3;122 relevant tests pass. Only code, tests and aggregate receipts are public. Private archives, records, responses, keys and per-request quota entries remain private.

## Evaluation

Private checkpoint243 is independently readback verified with archive SHA55941235c9b63cb4c7b6592c7cb71984e97c9e8753089cf734ff738c584ac34f. Its actual archived journals contain Groq39/912, Gemini345/912 and35 paired votes, with zero duplicate votes. This establishes three new successful Groq judgements persisted after recovery. Live progress recorded40 Groq while snapshot capture contained39; only the archived39 are claimed durable. All912 responses remain preserved. The same private writer lease version1 remains held; one coordinator is executing in Colab. No second coordinator was launched.

Groq is observing short quota waits under retained30RPM/1000RPD/8KTPM/200KTPD limits. The observed reset epoch1793558803.1988807 was a transient pacing boundary, not a new daily entitlement. Gemini credentials/catalogue/calibration PASS, but its worker is blocked; the existing unresolved503 with no durable reply remains protected and was not replayed. Existing Gemini5RPM/500RPD/250KTPM/1.25MTPD limits are retained. No training, inference, generation, DeepSeek repair, paid fallback or quota reset occurred. Evaluation is partial, not final metrics.

The live notebook must stay connected; do not run another resume cell or clear its lock/lease. If interrupted, reconcile the latest private version and held lease before a reviewed continuation. Never rerun recovery239 over a newer production checkpoint. The scheduled desktop monitor remains PAUSED; it was not re-enabled.
