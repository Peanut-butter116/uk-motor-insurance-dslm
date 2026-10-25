# Resume the existing QA pipeline in Kaggle

Prepared 25 October 2026. Dataset freeze deadline: **18 November 2026**.

**Generation has not been launched.** The owner chose to configure notebook secrets later. A separate provider-free Kaggle job checks the migrated checkpoint. Its receipt is recorded separately; it is not a generation success.

## Authoritative pipeline and audit

The production repository is [uk-insurance-QA-data](https://github.com/Peanut-butter116/uk-insurance-QA-data), pinned at `aef4d8c947750bb2fe1b10530c7060e2ab5cbda6`. This research repository owns experiments and the cloud adapter. No upstream production file or existing private local dataset was modified.

- `scripts/generate_motor.py`: actual source-package selector, `SOURCE_PROMPT`, `TEACHER_PROMPT`, content-derived stable IDs and resumable generation loop. `src/generate_qa.py` is an earlier helper, not the production entry point.
- `src/production.py`: canonical Build loader, schema validation, two-stage independent judge, seven acceptance axes, exact claim-quote checks, decision checkpoint and atomic export pointer.
- `src/providers.py`: explicit free-provider routing, verified account receipts, persistent quota reservations, 429 pauses, structured-response retries, request caches and uncertain-transport stop.
- `src/protected_screen.py`: exhaustive questions-only semantic screening against all protected test/dev/seed questions. Gold answers and contexts never enter this screening request or the teacher prompt.
- `src/check_leakage.py`: exact/lexical question, section, long-quote and gold-answer overlap checks. `config/dataset_schema.json` is canonical; `config/generation.yaml` is planning configuration, while the actual old runner hardcodes 10,000.
- Private authority: `uk-insurance-QA-data-local/production-v1/state.json`; resolve `production-v1/data/final` ONCE to its immutable export directory. Candidates: `generated_candidates.jsonl`; dispositions: state and `review_and_rejection_queue.jsonl`; API responses/attempts/quotas: `provider_cache/`, `api_cache/`, `provider_usage/`. Do not merge old root exports, backups, or the dated README's 337-row snapshot.

The current export resolves to `52902d2e4bd0ff464e10538fc6d2134ac8ab821f9b4e63ecbcc49aed196cb7ec`. Its 409 records match the research snapshot byte for byte, SHA-256 `6ec5b3728bd6b25cae09d97c81974f371c76a47ce61d2d2795a6919257b44a8f`. There are 335 answerable and 74 `unanswerable` records, 384 distinct question texts and no identical question/context pairs. Existing repeated-question context variants remain untouched.

Historical provenance contains **325 GPT-5.4 validations and 84 Gemini validations**; it is not an all-Gemini dataset. The current configured workflow is **Groq `openai/gpt-oss-120b` → Gemini `gemini-3.5-flash-lite`**. The owner explicitly confirmed preserving Groq after clarifying the initial “Grok” spelling. No xAI model was substituted.

The last checkpoint is `2026-10-18T22:55:37Z`, stopped with `gemini HTTP 503`. The last judged package ends in `:p40`; `:p41` exhausted structured retries. There are **608 generated candidates: 274 accepted (45.07%), 266 review, 49 rejected, 15 structured failures and 4 pending**. The other 135 accepted records are inherited. Twelve generated candidates were flagged for near-duplicates and 41 for exhaustive protected-screen failure/uncertainty. Reason counts overlap. [Reproducible aggregate report](progress.json).

One exact Gemini request was marked uncertain after the known HTTP 503 rejection. The adapter preserves its original attempt in an audit file and retains its quota reservation before permitting the exact request to retry. It does not clear other uncertain or in-flight requests. A missing raw response after an unknown transport remains a blocker requiring investigation.

## Actual private cloud locations

- Original immutable [409-row display/training snapshot, version 1](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-interim-20261020-409/versions/1) stays unchanged.
- [Private production checkpoint](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-qa-production-v1) stores `checkpoint.bin` and `migration_manifest.json`. Subsequent versions also contain `progress.json`. The archive retains accepted/pending/rejected records, reasons, state, evidence, historical response caches, quota ledgers, frozen protected data and pinned production code. It grants no redistribution rights.
- [Prepared private resumption notebook](https://www.kaggle.com/code/scarletthe0116/uk-motor-qa-resume) is saved **without execution**.
- [Provider-free migration check](https://www.kaggle.com/code/scarletthe0116/uk-motor-qa-checkpoint-check) validates archive/file hashes, 409 IDs, accepted decisions and the frozen-gold hash inside Kaggle. Only its aggregate receipt is retained as notebook output.

The old Kaggle dataset contains a training/display snapshot, not enough state to resume generation. The separate production dataset on the same private platform preserves that immutable snapshot and avoids mixing its display files with operational caches. Only existing data was temporarily archived for one-way migration; no new QA was generated or downloaded locally. Migration scratch is removed after validation.

Privacy was checked through authenticated Kaggle metadata. GitHub reports the data-production repository private and this research repository public. Public files contain code, schemas, hashes and aggregates only. Existing source-permission owner attestations remain private, scoped to the already staged material and provider transfers; no new source rights, public redistribution, provider plan or training clearance is inferred.

## Enable and run the first cloud batch

1. Open the private **resumption notebook** above and select **Edit**.
2. Under **Add-ons → Secrets**, add and enable `GROQ_API_KEY`, `GEMINI_API_KEY`, and `KAGGLE_API_TOKEN` from the existing authorised accounts. Enter secrets in Kaggle, never in GitHub code, notebook cells, logs or chat. The Kaggle token must be the existing owner's token with permission to version the checkpoint dataset. No new account is needed.
3. Keep Internet enabled and accelerator **None**. Confirm both provider projects remain on their authorised free plans; no paid usage is authorised.
4. Keep the last cell at `main(max_packages=1, target=500)` and **Save Version → Save & Run All**. Do not start a second session or another writer against the same checkpoint store.
5. The job loads the newest checkpoint version, verifies hashes, refreshes the original catalogue/synthetic-JSON readiness checks and resumes four pending candidates before at most one new eligible source package. The provider models must still exist: an unavailable pinned model stops the job; no replacement is selected.
6. Completion requires a new PRIVATE dataset version and a matching `/kaggle/working/persistence_receipt.json`, including count, version, timestamp and archive hash. Notebook `COMPLETE` alone does not prove generation succeeded: inspect `finished_status`, `progress.json`, the receipt and accepted-count delta. If no candidate passes all gates, the batch may save zero new accepted records. Do not call that an increase.

Missing secrets stop before provider calls. Provider readiness refresh is seven-day scoped; account/permission declarations are preserved, never fabricated. Free quotas remain the existing private receipt's conservative limits. Current documentation says Gemini quotas depend on the project; confirm actual account limits in [AI Studio](https://ai.google.dev/gemini-api/docs/rate-limits). [Groq rate-limit documentation](https://console.groq.com/docs/rate-limits) likewise does not replace the account receipt.

## Restart and scale

Start a **fresh** Kaggle session, then run the same notebook. It fetches the newest persistent version rather than trusting a stale mounted attachment. Completed package IDs, accepted decisions, content-derived candidate IDs, validated responses and quota reservations are reused. Immutable baseline and all records accepted before the session must survive exactly; otherwise persistence fails.

After the first successful generation/persistence batch is inspected, set `max_packages=5` (maximum 20) and retain `target=500`. At D500, change only `target=1000`; at D1000, change to `target=2000`. The default stops after one package; the adapter records `paused_cloud_batch_limit` when eligible work remains. A 429 records a durable pause and exits; restart after the recorded `resume_after_epoch`. Known 503 responses receive bounded 15/30/60-second backoff; unknown transports stop for recovery. Stop at a package boundary when possible. Hard interruption may require recovery of an in-flight request; no blind reissue is permitted.

Every export is versioned and synchronised before further work. Uploads are verified with pinned-version reads in cloud memory, without downloading datasets to the laptop. Upload/verification failure stops generation. Exclusive local cloud locks and a single notebook writer are required: Kaggle dataset updates are not a distributed compare-and-swap lock. **No scheduler or unattended continuation has been configured.**

## Method retained and additional exclusions

Generator, source-usability prompt, independent-solve prompt, seven-axis reference/claim audit prompt, protected-question prompt, response schemas, citation identifiers, generation parameters and accepted record schema are retained. A guarded AST transformation changes only four integer target constants in the pinned orchestration script; prompt strings are unchanged. `Build` and its caches/exports are reused rather than reimplemented. API keys exist only in Kaggle secrets/process memory.

The adapter adds explicitly documented checks for NEW records only:

- Use the same independent-solve prompt and Gemini model over the entire extracted source policy for new insufficient-evidence candidates. Incomplete or oversized documents go to review; available substantive answers exclude the insufficiency label. The supplied training context and abstention text are never rewritten.
- Reuse the original exclusion-only semantic rubric against accepted training questions, plus earlier passing candidates in the batch. Preserve the original separate full frozen-set screen. Lexical/normalised near-duplicate checks remain and trivial numerical templates are additionally quarantined.
- Exclude new records sharing an identified protected supporting page, in addition to the existing section/quote checks. Ambiguous old section/page mappings are a limitation; no printed-version independence is claimed.
- Preserve every previously accepted record exactly; historical records are not retroactively relabelled or certified by these extra checks.

These stricter exclusion gates may lower yield and change the composition of new data; their outcomes and original model provenance remain auditable. They do not relax or replace the original acceptance rubric. Source order remains the existing insurer-interleaving order. The 20–25% new insufficiency share is an operational monitoring target, never padding or a forced absence label. Insufficiency diversity and share require inspection after actual batches; they are not yet demonstrated.

## Milestones and SFT limitations

D500/D1000/D2000 freeze the original 409-record order followed by sorted newly accepted IDs at each durable checkpoint. Prefix selection guarantees nested milestone IDs. A frozen milestone cannot be changed. Each contains canonical `train.jsonl` and a manifest with IDs, label/source distributions, generation/validation provenance summaries, checksum, date and preservation checks. Original complete row provenance is retained. Human review remains outstanding; `training_ready=false` is preserved.

The record format remains compatible with the production canonical schema. **The research repository's current SFT loader is still the historical 21-seed loader.** It does not automatically consume these milestones. Reviewed training clearance, grouped context-variant decisions, independent development data and a separately tested loader change remain prerequisites to new QLoRA training. This task does not launch SFT or alter the frozen 76-question benchmark.

## Feasibility and remaining risks

There are **537 uncompleted packages**, all from Hastings Direct, at most **2,148** further candidates before rejection. At the historical aggregate 45.07% yield, that is approximately **968** additional acceptances (about **1,377** total, plus any recovered pending records), before the stricter new gates. Achieving 2,000 from this queue would require approximately 74% acceptance, so the present corpus queue does not reliably support the main target. That estimate assumes every remaining package produces four candidates. Including historical source-review/failed packages gives a lower observed package-yield estimate of about **347** further acceptances. Both estimates are descriptive, not forecasts, and historical providers differ.

Broader source representation and 2,000 accepted records will likely require additional eligible page packages from approved policies or newly authorised evidence, prepared through the existing extraction/source-review process. Do not regenerate completed packages or relax gates to fill the gap. D3000 is not enabled. No completion date is promised without measuring the first cloud batches and account quota availability.

## Local checks (no provider calls)

```bash
python -m pytest tests/ -q
python cloud/qa/audit.py --source ../uk-insurance-QA-data \
  --local ../uk-insurance-QA-data-local --legacy ../uk-insurance-dslm-data \
  --output cloud/qa/progress.json
python cloud/qa/stage_notebook.py --output /tmp/motor-qa-notebook
# Saves code only; does not execute the notebook:
kaggle kernels push -p /tmp/motor-qa-notebook --no-run
```

`prepare.py` is a one-time migration helper. Do not rerun it over a newer cloud checkpoint or upload its original archive as a replacement version. Future checkpoint creation and generation are cloud-only. Utility tests use fictional data. The provider-free cloud migration check does not establish end-to-end generation, full-document judge performance, or restart after a provider failure; those remain to be validated after secrets are configured.
