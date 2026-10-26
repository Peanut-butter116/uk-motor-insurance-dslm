# Resume the existing QA pipeline in Kaggle

Prepared 25 October 2026. Dataset freeze deadline: **18 November 2026**.

**Controlled p42 resumption completed and independently audited.** Private checkpoint **v10** contains **412 accepted records: 338 answerable and 74 insufficient evidence**. The four cached Groq candidates were resumed without regeneration: three accepted, one held for review. All 409 originals and the frozen benchmark are unchanged. [Final aggregate progress](progress.json) · [Independent v10 restore/QC audit](reports/post-batch-audit-v10.json). The owner has now authorised free-tier continuation to 1,000, beginning with a fresh package. **Execution v8 is COMPLETE.** Verified private **v154** contains **450 accepted (376 answerable / 74 insufficient)** and 744 cumulative candidates. The session processed 52 source packages and generated 136 fresh candidates: 38 accepted, 84 review, 12 rejected and two structured-output failures. There are 484 packages left. The Groq quota wait exceeded the remaining four-hour session window; it was saved and has now elapsed. A fresh editor start is required to continue; the completed-session monitor is paused. Paid fallback and QLoRA training remain excluded.

## Monitoring connectivity — recovered 25 October, 20:39 UTC

The earlier local Kaggle connection failure has recovered. The new authenticated aggregate check confirms private v106 ready and the campaign RUNNING at 440 accepted records. The historical [connectivity failure receipt](reports/monitor-connectivity-20261025T1911.json) remains retained; no cloud restart was performed.

## Authoritative pipeline and audit

The production repository is [uk-insurance-QA-data](https://github.com/Peanut-butter116/uk-insurance-QA-data), pinned at `aef4d8c947750bb2fe1b10530c7060e2ab5cbda6`. This research repository owns experiments and the cloud adapter. No upstream production file or existing private local dataset was modified.

- `scripts/generate_motor.py`: actual source-package selector, `SOURCE_PROMPT`, `TEACHER_PROMPT`, content-derived stable IDs and resumable generation loop. `src/generate_qa.py` is an earlier helper, not the production entry point.
- `src/production.py`: canonical Build loader, schema validation, two-stage independent judge, seven acceptance axes, exact claim-quote checks, decision checkpoint and atomic export pointer.
- `src/providers.py`: explicit free-provider routing, verified account receipts, persistent quota reservations, 429 pauses, structured-response retries, request caches and uncertain-transport stop.
- `src/protected_screen.py`: exhaustive questions-only semantic screening against all protected test/dev/seed questions. Gold answers and contexts never enter this screening request or the teacher prompt.
- `src/check_leakage.py`: exact/lexical question, section, long-quote and gold-answer overlap checks. `config/dataset_schema.json` is canonical; `config/generation.yaml` is planning configuration, while the actual old runner hardcodes 10,000.
- Private authority: `uk-insurance-QA-data-local/production-v1/state.json`; resolve `production-v1/data/final` ONCE to its immutable export directory. Candidates: `generated_candidates.jsonl`; dispositions: state and `review_and_rejection_queue.jsonl`; API responses/attempts/quotas: `provider_cache/`, `api_cache/`, `provider_usage/`. Do not merge old root exports, backups, or the dated README's 337-row snapshot.

The pre-cloud baseline export resolves to `52902d2e4bd0ff464e10538fc6d2134ac8ab821f9b4e63ecbcc49aed196cb7ec`. Its 409 records match the research snapshot byte for byte, SHA-256 `6ec5b3728bd6b25cae09d97c81974f371c76a47ce61d2d2795a6919257b44a8f`. There are 335 answerable and 74 `unanswerable` records, 384 distinct question texts and no identical question/context pairs. Existing repeated-question context variants remain untouched.

Historical provenance contains **325 GPT-5.4 validations and 84 Gemini validations**; it is not an all-Gemini dataset. The current configured workflow is **Groq `openai/gpt-oss-120b` → Gemini `gemini-3.5-flash-lite`**. The owner explicitly confirmed preserving Groq after clarifying the initial “Grok” spelling. No xAI model was substituted.

The pre-cloud checkpoint is `2026-10-18T22:55:37Z`, stopped with `gemini HTTP 503`. The last judged package ends in `:p40`; `:p41` exhausted structured retries. At that pre-cloud checkpoint there were **608 generated candidates: 274 accepted (45.07%), 266 review, 49 rejected, 15 structured failures and 4 pending**. Final v10 has 277 generated acceptances (45.56%), 267 review, 49 rejected and 15 structured failures, with no pending candidates; the 135 inherited accepted records give 412 total. The other 135 accepted records are inherited. Twelve generated candidates were flagged for near-duplicates and 41 for exhaustive protected-screen failure/uncertainty. Reason counts overlap. [Reproducible aggregate report](progress.json).

One exact Gemini request was marked uncertain after the known HTTP 503 rejection. The adapter preserves its original attempt in an audit file and retains its quota reservation before permitting the exact request to retry. It does not clear other uncertain or in-flight requests. A missing raw response after an unknown transport remains a blocker requiring investigation.

## Actual private cloud locations

- Original immutable [409-row display/training snapshot, version 1](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-interim-20261020-409/versions/1) stays unchanged.
- [Private production checkpoint](https://www.kaggle.com/datasets/scarletthe0116/uk-motor-qa-production-v1) stores `checkpoint.bin` and `migration_manifest.json`. Subsequent versions also contain `progress.json`. The archive retains accepted/pending/rejected records, reasons, state, evidence, historical response caches, quota ledgers, frozen protected data and pinned production code. It grants no redistribution rights.
- [Prepared private resumption notebook](https://www.kaggle.com/code/scarletthe0116/uk-motor-qa-resume) is saved **without execution**.
- [Secret-enabled controlled notebook](https://www.kaggle.com/code/scarletthe0116/uk-motor-qa-checkpoint-check) originally performed migration validation and now runs the pinned p42 test/repair. [Independent provider-free batch audit](https://www.kaggle.com/code/scarletthe0116/uk-motor-qa-batch-audit) verifies restored private checkpoints; only aggregate audit results are retrieved locally.

The old Kaggle dataset contains a training/display snapshot, not enough state to resume generation. The separate production dataset on the same private platform preserves that immutable snapshot and avoids mixing its display files with operational caches. Only existing data was temporarily archived for one-way migration; no new QA was generated or downloaded locally. Migration scratch is removed after validation.

The verified migration baseline is **version 3**; subsequent controlled-batch versions and clearance status are recorded below. Versions 1–2 are retained for migration history and must not be used: archive expansion and an omitted symlink target were detected and corrected before any generation. Privacy was checked through authenticated Kaggle metadata. [Setup receipt](cloud_receipt.json). GitHub reports the data-production repository private and this research repository public. Public files contain code, schemas, hashes and aggregates only. Existing source-permission owner attestations remain private, scoped to the already staged material and provider transfers; no new source rights, public redistribution, provider plan or training clearance is inferred.

## Controlled one-package execution — 25 October

The secret-enabled existing notebook is `uk-motor-qa-checkpoint-check`. Its earlier provider-free validation versions are preserved. The controlled generation code pins package `p42`, including its four already-generated pending candidates. Restarts can finish that package but cannot advance to `p43` or another package.

Start secret-using runs from **Kaggle’s notebook editor**, using **Save Version → Save & Run All** with the existing three secret selections enabled. The API-triggered attempt failed with a startup `ConnectionError` before any provider call; the editor-triggered run successfully accessed secrets and persisted checkpoints. The CLI does not provide a supported secret-attachment mechanism: [official documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#using-secrets-in-kernels). Do not put credentials in a dataset or notebook cells as a workaround.

The first editor run saved private checkpoint v5 and verified a full restore. It paused at the conservative Gemini quota guard after two-stage validation: three candidates remained provisional and one insufficiency candidate required review because full-document extraction was incomplete. All 409 accepted records and the frozen benchmark remained intact. [Intermediate aggregate receipt](reports/controlled-p42-quota.json). This intermediate pause is not a completed test. The authorized continuation remains pinned to the same package.

The continuation exported 412 accepted records in private checkpoint v9: three answerable additions, with the insufficient-evidence candidate left for review. No new teacher candidates were generated: all four were cached p42 candidates. [Primary batch receipt](reports/controlled-p42-final.json). An independent provider-free restore verified every file, preserved baseline and grounding checks, but found **two missing semantic duplicate comparisons among the new records** after the quota restart. [Independent v9 audit](reports/post-batch-audit-v9.json). Thus the primary receipt's `test_success=true` is superseded by this QC finding; v9 is provisional, not cleared for training.

`repair_p42.py` is pinned to private v9 and only those two missing comparisons, using the existing Gemini exclusion-only prompt. It never calls the teacher or selects a source package. A failed or uncertain match quarantines only a new record; all 409 originals and previous cloud versions remain intact. `stage_repair.py` saves that code to checkpoint-check without executing it. The editor-started repair completed as notebook v8, saved private dataset v10 and passed both missing comparisons. No candidate was demoted. [Repair receipt](reports/controlled-p42-repair-v10.json) · [Persistence receipt](reports/persistence-v10.json). A separate provider-free audit restored v10 and verified all 5,139 files, baseline preservation, all new duplicate comparisons and original deterministic/claim gates. [Final audit](reports/post-batch-audit-v10.json). The main resumption adapter now includes earlier passing provisional candidates in comparisons across restarts; a regression test covers that bug.

If Kaggle reports `ConcurrencyViolation` with expected/actual draft sequence numbers, do not upload another API version or repeatedly save the stale editor. Close other editing tabs, reopen the saved notebook, choose Edit and refresh to load the current draft. Confirm the heading says “Finish p42 duplicate screening only”, then select Save Version → **Save & Run All**, not Quick Save. A save-only `COMPLETE` version can have no executed cells or receipts; it is not a repair execution. This resolved the saved-only v7 draft conflict; the subsequent editor-started v8 executed successfully. Do not run the repair again: it is intentionally pinned to input v9 and will stop before provider calls against v10.

Use `python cloud/qa/fetch_controlled_receipts.py --output /tmp/motor-qa-receipts` to retrieve aggregate results only. It makes a bounded first-page output listing; using the CLI output command with a regex can still paginate through thousands of private scratch files and hit Kaggle monitoring rate limits.

The [SFT loader proposal](SFT_LOADER_PROPOSAL.md) is separate from execution. Historical loader modules are unchanged; [byte-identity and regression evidence](sft_backward_compatibility.json) is recorded. The proposal is not implemented and no training is launched.

## General notebook setup (future batches require approval)

1. Open the approved private notebook and select **Edit**. The checkpoint-check notebook contains the completed targeted repair, not general generation; follow the restaging instructions below only after approval.
2. Under **Add-ons → Secrets**, add and enable `GROQ_API_KEY`, `GEMINI_API_KEY`, and `KAGGLE_API_TOKEN` from the existing authorised accounts. Enter secrets in Kaggle, never in GitHub code, notebook cells, logs or chat. The Kaggle token must be the existing owner's token with permission to version the checkpoint dataset. No new account is needed.
3. Keep Internet enabled and accelerator **None**. Confirm both provider projects remain on their authorised free plans; no paid usage is authorised.
4. Keep the last cell at `main(max_packages=1, target=500)` and **Save Version → Save & Run All**. Do not start a second session or another writer against the same checkpoint store.
5. The job loads the newest checkpoint version, verifies hashes, refreshes the original catalogue/synthetic-JSON readiness checks and resumes any pending verification before selecting at most one new eligible source package. At verified v10, no pending candidates remain. The provider models must still exist: an unavailable pinned model stops the job; no replacement is selected.
6. Completion requires a new PRIVATE dataset version and a matching `/kaggle/working/persistence_receipt.json`, including count, version, timestamp and archive hash. Notebook `COMPLETE` alone does not prove generation succeeded: inspect `finished_status`, `progress.json`, the receipt and accepted-count delta. If no candidate passes all gates, the batch may save zero new accepted records. Do not call that an increase.

Missing secrets stop before provider calls. Provider readiness refresh is seven-day scoped; account/permission declarations are preserved, never fabricated. Free quotas remain the existing private receipt's conservative limits. Current documentation says Gemini quotas depend on the project; confirm actual account limits in [AI Studio](https://ai.google.dev/gemini-api/docs/rate-limits). [Groq rate-limit documentation](https://console.groq.com/docs/rate-limits) likewise does not replace the account receipt.

## Authorised target-1,000 campaign and restart

The owner has authorised continuing existing eligible packages to **1,000 accepted**, with no additional approval per batch. Start [uk-motor-qa-resume](https://www.kaggle.com/code/scarletthe0116/uk-motor-qa-resume) from a fresh editor with the existing three secrets enabled, using **Save Version → Save & Run All**. Version 7 staged the code; editor-started **execution v8 completed** after safe automatic continuation. Final verified private **v154** contains 450 accepted records and 744 total generated candidates. It processed 52 packages, generated 136 new candidates, and accepted 38 (27.94%); 84 remain in review, 12 were rejected and two exhausted structured retries. There are 484 eligible packages remaining and 550 acceptances needed for 1,000. [Final aggregate session receipt](reports/campaign-session-v8-final.json) · [Verified persistence receipt](reports/persistence-v154.json).

The original 409 and campaign-starting 412 records were preserved by every export check, as was the frozen benchmark. All 38 additions are answerable; insufficient-evidence coverage remains 74/450 (16.44%). Twenty-four new candidates were flagged for incomplete full-document extraction and stayed in review; reason counts overlap. The 20–25% new insufficiency target was not achieved, and no labels or gates were relaxed.

**Restart now:** open the existing notebook in a fresh Edit session, keep the existing three secrets enabled, then **Save Version → Save & Run All**. No code upload, remigration, API-key sharing or further generation approval is needed within the existing free-tier authorisation. The recorded Groq retry time was 25 October at 22:19:42 London time and has elapsed. The notebook loads the latest private checkpoint (v154 unless a newer valid version exists) and preserves all 450 records. It does not rely on an old mounted attachment. The session monitor is paused; monitoring can be re-enabled after an actual restart is confirmed. Do not run the old p42-only repair notebook.

`campaign.py` invokes the pinned production adapter in isolated processes: one fresh package at a time until new generation produces a verified accepted candidate, then batches of five. Source-review failures do not count as a fresh end-to-end success. Models, prompts, evidence and thresholds remain unchanged. Each export adds a provider-free audit of grounding, schema, protected screening and complete training duplicate comparisons before private persistence. D750 joins the existing deterministic nested D500/D750/D1000/D2000 freezes; all remain `training_ready=false`.

Each process restores the newest private checkpoint, verifies every archived file, preserves all 409 originals and the campaign's starting 412 records, and excludes completed packages. Accepted/pending/review records and provider caches remain cloud-only. Storage uploads are checked by pinned-version streaming reads before more provider calls. The parent refuses continuation after a child error or a missing/mismatched verified receipt. Original provider verification receipts remain valid for seven days; restarts reuse them and rerun the existing synthetic readiness checks only if a receipt no longer passes the original verification rules. This avoids consuming quota on identical probes at each batch restart.

A recorded provider quota pause is waited out **inside the same Kaggle session** if it fits the remaining four-hour session window. Reservations and caches remain intact; no rate-limit bypass or paid fallback is enabled. Stop at 1,000, actual source exhaustion, an unresolved operational failure, or the session cap. **There is no configured cross-session generation scheduler.** The separate thread monitor (`uk-motor-qa-cloud-progress`) checked aggregates every 15 minutes during execution and is now paused after the completed session; it does not restart a secret-using Kaggle session. Desktop task scheduling follows the [official scheduled-task documentation](https://learn.chatgpt.com/docs/automations?surface=app). A quota wait beyond the remaining session is saved with its restart time rather than claiming unattended resumption.

To resume after a session ends, start the same campaign notebook through Kaggle's editor in a fresh session after any recorded `resume_after_epoch`. It loads the latest persistent version automatically; never remigrate v3/v10 over a newer version. Use only one writer. A Kaggle stale-draft `ConcurrencyViolation` requires closing other editor tabs and reopening/refreshing the saved notebook before Save & Run All.

`campaign_report.json` and `persistence_receipt.json` are private notebook aggregate outputs. During execution, the private dataset's `progress.json` exposes actual persisted counts. Download aggregate receipts only; do not download `checkpoint.bin` or raw records to the laptop. A saved notebook or a generic COMPLETE status alone does not establish fresh generation success.

## Method retained and additional exclusions

Generator, source-usability prompt, independent-solve prompt, seven-axis reference/claim audit prompt, protected-question prompt, response schemas, citation identifiers, generation parameters and accepted record schema are retained. A guarded AST transformation changes only four integer target constants in the pinned orchestration script; prompt strings are unchanged. `Build` and its caches/exports are reused rather than reimplemented. API keys exist only in Kaggle secrets/process memory.

The adapter adds explicitly documented checks for NEW records only:

- Use the same independent-solve prompt and Gemini model over the entire extracted source policy for new insufficient-evidence candidates. Incomplete or oversized documents go to review; available substantive answers exclude the insufficiency label. The supplied training context and abstention text are never rewritten.
- Reuse the original exclusion-only semantic rubric against accepted training questions, plus earlier passing candidates in the batch. Preserve the original separate full frozen-set screen. Lexical/normalised near-duplicate checks remain and trivial numerical templates are additionally quarantined.
- Exclude new records sharing an identified protected supporting page, in addition to the existing section/quote checks. Ambiguous old section/page mappings are a limitation; no printed-version independence is claimed.
- Preserve every previously accepted record exactly; historical records are not retroactively relabelled or certified by these extra checks.

These stricter exclusion gates may lower yield and change the composition of new data; their outcomes and original model provenance remain auditable. They do not relax or replace the original acceptance rubric. Source order remains the existing insurer-interleaving order. The 20–25% new insufficiency share is an operational monitoring target, never padding or a forced absence label. Insufficiency diversity and share require inspection after actual batches; they are not yet demonstrated.

## Milestones and SFT limitations

D500/D750/D1000/D2000 freeze the original 409-record order followed by sorted newly accepted IDs at each durable checkpoint. Prefix selection guarantees nested milestone IDs. A frozen milestone cannot be changed. Each contains canonical `train.jsonl` and a manifest with IDs, label/source distributions, generation/validation provenance summaries, checksum, date and preservation checks. Original complete row provenance is retained. Human review remains outstanding; `training_ready=false` is preserved.

The record format remains compatible with the production canonical schema. **The research repository's current SFT loader is still the historical 21-seed loader.** It does not automatically consume these milestones. Reviewed training clearance, grouped context-variant decisions, independent development data and a separately tested loader change remain prerequisites to new QLoRA training. This task does not launch SFT or alter the frozen 76-question benchmark.

## Feasibility and remaining risks

After completing p42, **536 uncompleted packages** remain, all Hastings Direct, with a ceiling of 2,144 generated candidates before rejection. Use completed-package yield rather than multiplying this ceiling by candidate acceptance rate: source-review and structured-output failures also consume packages. Pre-test Hastings history produced 164 accepted records from 210 completed packages across providers, while the recent free Groq workflow produced 84 from 82 packages. Applied to the remaining queue, these give approximately **419–549 additional acceptances**, about **830–960 total** from the verified 412. [Historical counts and scope](hastings_capacity_baseline.json). These are descriptive estimates, not forecasts: ordered overlapping policy pages are correlated and stricter exclusions may reduce yield. The one-package trial is too small to extrapolate from its 3/4 acceptance rate.

Broader source representation and 2,000 accepted records will likely require additional eligible page packages from approved policies or newly authorised evidence, prepared through the existing extraction/source-review process. Do not regenerate completed packages or relax gates to fill the gap. D3000 is not enabled. No completion date is promised without measuring the first cloud batches and account quota availability.

## Local checks (no provider calls)

```bash
python -m pytest tests/ -q
python cloud/qa/audit.py --source ../uk-insurance-QA-data \
  --local ../uk-insurance-QA-data-local --legacy ../uk-insurance-dslm-data \
  --output /tmp/motor-local-baseline-audit.json
python cloud/qa/stage_notebook.py --output /tmp/motor-qa-notebook
# Saves code only; does not execute the notebook:
kaggle kernels push -p /tmp/motor-qa-notebook --no-run
```

`prepare.py` is a one-time migration helper. Do not rerun it over a newer cloud checkpoint or upload its original archive as a replacement version. Future checkpoint creation and generation are cloud-only. Utility tests use fictional data. Provider-free migration checks establish checkpoint integrity only. Controlled p42 receipts separately record model verification, a quota pause/restart and persistence. The independent v9 audit found two missing within-batch comparisons; the targeted repair and independent v10 restore audit now establish batch completion. Full-document insufficiency validation did not clear the incomplete-extraction candidate, which remains under review.
