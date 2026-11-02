# Minimal opt-in milestone loader proposal

This is a proposal, not a replacement of the historical loader. No QLoRA training is launched. The existing `prepare_sft.py`, `tokenise.py` and `train.py` remain unchanged.

## Proposed change

Add one explicit input path for a new bundle schema, `milestone-sft/1`, alongside the unchanged `seed-sft/1` path. Do not disguise production records as seeds or remove the historical 21-row checks globally.

1. Add `experiments/prepare_milestone_sft.py`, accepting an immutable D500/D1000/D2000 directory and a separately recorded training-clearance receipt. Verify its checksum, exact ordered IDs, source/label distributions and production validation provenance. Require clearance tied to that exact dataset checksum. Automatic acceptance alone must not flip the current `training_ready=false` flag.
2. Add explicit schema dispatch in `prepare_sft.validate_bundle`: call the historical implementation unchanged for `seed-sft/1`; delegate `milestone-sft/1` to the new validator; reject unknown schemas. Seed preparation remains the default CLI behavior. An explicit milestone CLI uses the new converter.
3. Add an opt-in tokenisation path for the milestone bundle. Preserve whole records, original supplied evidence, citation identifiers and complete answers, including abstentions. Use `core.encode_training` for the existing assistant-only mask. Reject overlength rows without truncation and record exclusions. For model comparisons, freeze a common eligible ID set after checking every selected model's pinned tokenizer; apply the same IDs to all models. Keep the existing 21-row `prepare_matched` path unchanged.
4. Scope the training manifest's existing Qwen 21-row requirement to `seed-sft/1`. For `milestone-sft/1`, require the audited milestone checksum, exact common-population checksum and matching model tokenisation receipt. Keep smoke-before-full, model-specific fingerprints, pinned checkpoints, QLoRA settings and adapter-reload gates unchanged. A 21-row smoke receipt must never unlock milestone training.

The converted records keep the existing training shape:

```text
source_record_id = canonical example_id
messages = [system instructions + original complete context, original question, complete original answer]
provenance = source milestone/checksum, source record hash, answerability,
             original generator/verifier metadata, evidence IDs/hashes,
             prompt-format version and messages hash
```

Do not run retrieval to reconstruct canonical training evidence. Do not rewrite citations to the historical seed chunk format or append a second answer. The old system prompt specifies a fixed abstention sentence, whereas production abstentions may be evidence-limited prose: the new converter needs an explicitly versioned prompt contract compatible with the preserved targets. Review that choice before implementation; do not silently normalise the targets or reuse a contradictory instruction.

Freeze repeated-question context pairs as groups. Preserve all 409 canonical records in the source milestone; any training exclusions belong in a separate conversion report, not a revised accepted dataset. Report nominal milestone size and actual usable training count separately. If D500 yields fewer than 500 usable examples under the common tokenizer limit, call it the D500-derived population with its actual n. Do not claim it is a 500-example training experiment.

## Backward-compatibility checks required for the eventual patch

- Historical 21-row bundles validate and serialize identically before/after schema dispatch.
- Existing seed eligibility, held-out exclusions, whole-record length checks, citation resolution, assistant-only masking and Qwen matched-data tests remain green.
- Unknown schemas, invalid checksums, reordered/missing/duplicate IDs, unresolved citations, missing independent verification and absent/stale training clearance fail closed.
- Fictional 500/1000/2000-row bundles demonstrate prefix membership, preserved answerability/citations, complete refusal targets and identical shared model populations.
- Historical receipt fingerprints remain stable for unchanged historical code inputs; adding new code changes the repository code hash, so existing smoke receipts must not be forged or silently adopted. Historical receipts remain audit artifacts; any new executable training path needs a fresh matching GPU smoke and reload check.
- New milestone/population/model fingerprints cannot reuse historical smoke receipts. Test both schema paths explicitly, including historical 21-row rejection after a changed matched-data checksum.

For this proposal, byte-identity checks and the existing regression suite are run against the unchanged historical modules. Those checks establish that this task has not replaced the loader; they do not certify an unimplemented milestone path or GPU training.
