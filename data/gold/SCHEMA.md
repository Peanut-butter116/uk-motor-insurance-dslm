# Gold FAQ dataset — record schemas (v2)

Current file: `gold_v2.jsonl` — **200 records: test 140 / dev 30 / seed 30**
(earlier versions are retained immutable; see `CHANGELOG.md` for the additive
migration rules). Every record is verified against the source PDF
by its author and cross-reviewed by the other intern before the split freezes
(author ≠ reviewer, `review_status` tracks this). `eval/validate_gold.py`
mechanically validates all three files below and doubles as the **re-ingest
drift gate**: run it after every index rebuild; a citation section that no
longer resolves is a blocking failure.

| Field | Type | Meaning |
|---|---|---|
| `id` | str | `PQ-0001` … stable, never reused (erratum log instead of edits after freeze) |
| `version` | str | dataset version the record entered at (`v0`) |
| `split` | str | `test` / `dev` / `seed` — assigned at creation, frozen before any tuning |
| `persona` | str | `end_user`, `handler`, or `buyer` (pre-purchase guidance) |
| `line` | str | `travel` / `home` / `motor` / `general` — v1 field |
| `intent` | str | optional; manager's 2026-07-07 intent taxonomy (motor records) |
| `question` | str | exactly as a real user/handler would ask |
| `gold_answer` | str | correct, complete answer grounded in the wording |
| `answer_short` | str | A free-text, **human-facing gist** of the answer — conventionally `yes` / `no` / `depends` or a scalar amount, but **the convention is NOT enforced on answerable records and is not an enum**. See the note below. **The one hard rule: `answerable:false` → MUST be exactly `not_addressed`** (enforced by `validate_gold.py`). |
| `key_facts` | list[str] | 2–6 atomic facts = the judge's grading rubric |
| `gold_citations` | list[obj] | `{insurer, doc_file, section, page, quote}` — `quote` is a **verbatim ≤50-word span** from the PDF (what makes faithfulness mechanically checkable). **Type contract (T-028): `page` here is a `str`; chunk `meta.page` is an `int`. This asymmetry is FROZEN into the 140 hash-pinned test records, so it is normalised at COMPARISON SITES, never in the data: any code comparing a citation page to a chunk page must coerce both via `int(...)` (see `src/rag.py` `_norm_doc`/resolver, which also allows ±1 page tolerance for PDF-viewer off-by-ones). New comparison code copies that pattern; do NOT "fix" the records.** |
| `question_type` | str | coverage_in / exclusion / limits_excess / claims_process / definition / cancellation / eligibility / multi_clause / unanswerable |
| `difficulty` | int | 1 (single clause) · 2 (clause + condition) · 3 (multi-clause reasoning) |
| `answerable` | bool | false → gold behaviour is the fixed abstention sentence |
| `unanswerable_reason` | str | `absent_from_wording` / `needs_personal_facts` / `out_of_scope_advice` / `insurer_not_in_corpus` (v1: e.g. Direct Line — wording exists but is excluded from the index pending storage sign-off) |
| `must_not_assert` | str | the tempting wrong claim (e.g. the plausible-but-excluded reading) |
| `author` / `reviewer` | str | initials; reviewer must differ |
| `review_status` | str | `pending` → `verified` |
| `_held_out_insurer` | str | **Assembly-time annotation** (leading `_` = added by the pipeline, not hand-authored). Marks a record drawn from a **held-out insurer** — Post Office (travel) or LV= (motor) — whose records exist only in the frozen `test` split so the held-out delta measures generalisation to an unseen insurer. Present on **44 records, all `test`** (`LV=` 26, `Post Office` 18). See the note below. |

### `answer_short` — what it is, and what it is not

`answer_short` is a **convenience gist for humans reading the dataset. No scorer
reads it.** The eval composite's 0.2 abstention term is computed in
`eval/judge.py` (`_row_metrics`) from the record's `answerable` **boolean** plus
whether the model emitted the canonical abstain sentence — never from this
field. The judge payload carries only `question` / `gold_answer` / `key_facts` /
`must_not_assert` / `model_answer`. Verify it yourself:

```
grep -c answer_short eval/judge.py eval/harness.py     # -> 0 and 0
```

`train/raft_build_data.py` likewise builds every answerable target's conclusion
from `key_facts[0]`, not from this field.

**The `yes` / `no` / `depends` / amount convention is NOT enforced on answerable
records** — `validate_gold.py` applies no enum there. As of 2026-07-16 the live
`gold_v2.jsonl` carries **23 distinct values**: the four conventional ones
(`no` 66, `yes` 49, `depends` 36, `not_addressed` 30) plus a **19-value
free-text tail** (`£25`, `3 years`, `within 24 hours`, `as soon as you can`,
`twice the excess`, …). Treat a mismatch here as a **documentation/consistency
nit, not a metric defect** — it cannot move a score.

The **one** mechanically enforced rule is the unanswerable sentinel:
`answerable:false` ⇒ `answer_short == "not_addressed"` (currently 30/30). That
gate exists to keep the sentinel canonical so downstream consumers can rely on
it as a synonym for `answerable == false` — **not** because any scorer reads it.

### `_held_out_insurer` — why it exists separately from the citations

The held-out **purity checks** in `eval/assemble_gold.py` and
`eval/check_freeze.py` key on **`gold_citations[].insurer`** — they assert that
no non-`test` record cites a held-out insurer. That check is blind to a record
with **no citations**, and the **6 unanswerable held-out records** carry none by
construction (`answerable:false` ⇒ no `gold_citations`). Those 6 are therefore
marked **only** by `_held_out_insurer`. Do not delete the field on the
assumption it is derivable from the citations: for those records, it is not.

Taxonomy targets for the full 200-set: coverage 25 · exclusions 25 ·
limits/excess 25 · claims-process 20 · definitions 20 · cancellation 15 ·
eligibility 20 · multi-clause 20 · **unanswerable 30 (15%)**. Personas ~50/50
with ~30 paired questions (same clause, both registers).

**Motor slice (+87, v1 targets):** cover types 8 · excess 7 · NCD 6 · use
classes 6 · add-ons 8 · European cover 5 · modifications 5 · claims 6 ·
repairs 4 · named drivers 6 · mileage 4 · change-fees 3 · discounts 3 ·
payment-mechanics 4 (mechanics printed in T&Cs are answerable-with-citation;
only personal pricing refuses) · unanswerable 12 (incl. `insurer_not_in_corpus`
Direct Line/Churchill probes). New single-turn total: **287**.

## Conversation records — `conversations_v1.jsonl` (buyer mode)

| Field | Type | Meaning |
|---|---|---|
| `id` | str | `CONV-0001` … |
| `version` / `split` | str | as above; **all manager-derived items are `test`** |
| `scenario_facts` | dict | the simulated customer's ground-truth card — keys from the slot schema (`vehicle_type, driver_age_band, licence_years, annual_mileage_band, usage_class, overnight_parking, named_drivers, modifications, ncd_years`) + optional `context` |
| `opening_message` | str | the customer's first utterance |
| `intent` | str | gold intent label |
| `required_clarifications` | list[str] | slots the assistant must ask before its first citation-bearing turn (Clarification F1 is computed deterministically from template logs) |
| `expected_coverage_points` | list[str] | facts a complete substantive answer must convey (judge rubric, 0–2 each) |
| `gold_citations` | list[obj] | same shape as single-turn; validated verbatim |
| `must_not` | list[str] | assertions/behaviours that cap the transcript at fail |
| `personalization_rubric` | str | PERG-safe: rewards *including* all relevant cited facts for the collected slots — never narrowing/recommending |
| `author` / `reviewer` / `review_status` | str | as above |

## Safety probes — `safety_probes_v1.jsonl` (buyer mode)

| Field | Type | Meaning |
|---|---|---|
| `id` | str | `PROBE-0001` … |
| `kind` | str | `negative_paraphrase` (manager's 5 cases × 3, authored post-guard-freeze by the non-implementing intern) / `advice_boundary` / `implicit_steering` / `balanced_presentation` |
| `prompt` | str | the user message (or short scripted lead-in) |
| `pass_criterion` | obj | `{tag: deterministic\|judged, rule: "..."}` — deterministic rules are regex/constant checks (SIGNPOST emission, lexicon absence, all-three-enumeration); judged rules go to the LLM judge |
| `must_not` | list[str] | optional; explicit forbidden assertions |

Safety scoring: hard gate = **0 deterministic violations measured guard-OFF**
(ranked rows must pass); judged violations are a separate reported column; the
guard-on vs guard-off delta is itself a reported result.
