"""Package the private Kaggle dataset for the QLoRA run (DATA_SYNC plane 4).

What ships (staged to POLICY_QA_HOME/kaggle_staging/dataset/, then uploaded as a
PRIVATE, DISPOSABLE Kaggle dataset — delete it after the experiment):
  raft_train_split.jsonl   the RAFT rows (count in raft_manifest.json) + a clause-disjoint train/val split
  smoltalk_replay.jsonl    64-row pinned replay slice (make_smoltalk_slice.py)
  eval_tasks_kaggle.jsonl  pre-rendered eval prompts (harness.py export-kaggle)
  raft_manifest.json       provenance (bucket mix, seed, pending-gold count)

The split happens HERE, locally — it is a correctness-critical decision that
must be auditable before upload: bucket-C refusal rows are minimal-pair twins
of A/B rows sharing a source_record_id; a naive random split would leak the
paired clause across train/val. Grouping by source_record_id prevents that.

⚠ THE STAGING DIR IS TRACKED IN THE DATA REPO, so every run of this script — even
`--stage-only`, even with no upload — REWRITES TRACKED FILES under
`POLICY_QA_HOME/kaggle_staging/dataset/`. Measured 2026-08-19: a single
`--train-file synth_train.verified.jsonl --stage-only` left `raft_train_split.jsonl`
and `raft_manifest.json` modified in the submodule. It is harmless and reversible
(`git -C policy_qa checkout -- kaggle_staging/`), but it dirties the submodule, and a
dirty submodule next to a deliberately-pinned gitlink is exactly the state
`tests/test_submodule_pin.py` exists to keep legible. CHECK `git -C policy_qa status`
AFTER STAGING, and restore it unless you intend to commit the new staging set.

Run in the working-project checkout (or set POLICY_QA_HOME):
  python train/kaggle/package_dataset.py            # stage + create (first time)
  python train/kaggle/package_dataset.py --version "msg"   # subsequent updates
  python train/kaggle/package_dataset.py --stage-only      # no upload
  python train/kaggle/package_dataset.py --adapter artifacts/kaggle_runs/<ts>/qwen25-7b-raft-lora \
      --version "add adapter"   # ship a fetched adapter for --eval-only recovery
  python train/kaggle/package_dataset.py --train-file synth_train.verified.jsonl \
      --version "synth challenger"   # package the synthetic set instead of the 224-row RAFT set
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The `kaggle` CLI is NOT on the PATH here, and assuming it is cost a blocked GPU pipeline on
# 2026-08-25: `sh([KAGGLE, ...])` died with FileNotFoundError after staging, which reads like a
# missing install rather than a PATH problem. Measured: `which kaggle` -> not found, while
# `<conda-env>/bin/kaggle --version` -> Kaggle CLI 2.2.3.
#
# Resolve it as a SIBLING OF THE RUNNING INTERPRETER. Whichever env's python runs this script is
# the env whose `kaggle` we want, so the two can never drift apart -- and it needs no hardcoded
# home path. `shutil.which` is the fallback for a system install. Same precedent as
# `eval/demo_preflight.py:104`, which pins `lms` by absolute path rather than trusting the PATH.
def _kaggle_bin() -> str:
    sibling = Path(sys.executable).parent / "kaggle"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("kaggle")
    if found:
        return found
    sys.exit("cannot find the `kaggle` CLI. It is not beside this interpreter "
             f"({sibling}) and not on the PATH. Install it into the policyqa env: "
             "pip install kaggle")


# RESOLVED LAZILY, ON FIRST USE. Same defect as train/kaggle/run_kaggle.py, same commit
# (b1cf5be), same fix: `KAGGLE = _kaggle_bin()` at module scope means importing this file
# calls sys.exit on any machine without the CLI, which kills pytest COLLECTION and stops the
# whole suite. No test imports this module today — that is the only reason it was not a
# second CI outage, and it is not a property worth relying on. The loud failure is unchanged
# and still fires on first real use. Gate: tests/test_no_import_time_exit.py.
_KAGGLE: str | None = None


def kaggle_bin() -> str:
    """The CLI path, resolved once and cached. Call this; never resolve at module scope."""
    global _KAGGLE
    if _KAGGLE is None:
        _KAGGLE = _kaggle_bin()
    return _KAGGLE

PROJ = Path(__file__).resolve().parents[2]
_candidates = [os.environ.get("POLICY_QA_HOME"),
               PROJ / "policy_qa",
               PROJ.parent / "policy_qa", PROJ]
HOME = next((Path(c) for c in _candidates if c and Path(c, "src", "common.py").exists()), PROJ)

TRAIN_DIR = HOME / "data" / "train"
STAGE = HOME / "kaggle_staging" / "dataset"
DATASET_SLUG = "uk-dslm-raft-private"
SEED = 0
N_HOLDOUT_IDS = 5           # ~17-19 rows (~9%) held out as val, clause-disjoint
AGGREGATE_IDS = {"motor-faq", "d-seed"}   # umbrella ids — always stay in train


def kaggle_username(required: bool = True) -> str | None:
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        if required:
            sys.exit("No ~/.kaggle/kaggle.json — create an API token at "
                     "kaggle.com Settings -> API -> Create New Token, save it there, "
                     "then: chmod 600 ~/.kaggle/kaggle.json")
        return None
    return json.loads(cred.read_text())["username"]


def make_split(rows: list[dict]) -> list[dict]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_id[r["source_record_id"]].append(r)

    # candidates: real per-record ids that have BOTH an answerable row and a
    # C-refusal twin (so val exercises commit AND refuse on unseen clauses)
    candidates = sorted(
        rid for rid, rs in by_id.items()
        if rid not in AGGREGATE_IDS
        and any(r["bucket"] in ("A", "B") for r in rs)
        and any(r["bucket"] == "C" for r in rs)
    )
    rng = random.Random(SEED)
    val_ids = set(rng.sample(candidates, min(N_HOLDOUT_IDS, len(candidates))))

    out = []
    for r in rows:
        r = dict(r)
        r["split"] = "val" if r["source_record_id"] in val_ids else "train"
        out.append(r)

    train_ids = {r["source_record_id"] for r in out if r["split"] == "train"}
    leak = train_ids & val_ids
    assert not leak, f"clause leak across split: {leak}"

    table = Counter((r["bucket"], r["split"]) for r in out)
    print(f"split (seed={SEED}): {len(val_ids)} held-out record ids -> "
          f"{sum(1 for r in out if r['split'] == 'val')} val rows")
    for b in sorted({k[0] for k in table}):
        print(f"  bucket {b}: train={table.get((b, 'train'), 0):3d}  "
              f"val={table.get((b, 'val'), 0):3d}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", metavar="MSG", help="update an existing dataset")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--adapter", metavar="DIR",
                    help="also ship a fetched LoRA adapter dir (as adapter/) so the "
                         "kernel's --eval-only recovery mode can load it")
    ap.add_argument("--train-file", metavar="NAME", default="raft_train.jsonl",
                    help="training rows to package, relative to POLICY_QA_HOME/data/train/ "
                         "(default: raft_train.jsonl, the 224-row set behind every published "
                         "number). Use synth_train.verified.jsonl to package the synthetic set.")
    ap.add_argument("--manifest-file", metavar="NAME", default=None,
                    help="provenance manifest to ship (default: raft_manifest.json for the "
                         "default train file, synth_manifest.json for a synth_* train file)")
    args = ap.parse_args()

    # WHY THIS IS A FLAG AND NOT AN EDIT TO THE CONSTANT, measured 2026-08-19.
    # raft_train.jsonl is the 224-row set behind every published row, and its sha256 is pinned
    # in the freeze receipt (FREEZE_v2.json .raft_receipt.raft_train_sha256) -- overwriting it
    # to train on something else turns check_freeze.py verify RED and needs a --force re-baseline.
    # The synthetic set is a DROP-IN: both files carry exactly
    # {bucket, distractor_ids, messages, oracle_present, source_record_id} with the same
    # system/user/assistant roles, so make_split()'s source_record_id grouping works unchanged
    # (verified: raft 224 rows / 56 distinct source ids; synth 1,871 rows / 374).
    raft_path = TRAIN_DIR / args.train_file
    if args.manifest_file:
        manifest_name = args.manifest_file
    else:
        manifest_name = ("synth_manifest.json" if args.train_file.startswith("synth")
                         else "raft_manifest.json")
    replay_path = TRAIN_DIR / "smoltalk_replay.jsonl"
    tasks_path = HOME / "eval" / "results" / "eval_tasks_kaggle.jsonl"
    manifest_path = TRAIN_DIR / manifest_name
    for p, hint in ((raft_path, f"python train/raft_build_data.py  (or pick another --train-file)"),
                    (replay_path, "python train/make_smoltalk_slice.py"),
                    (tasks_path, "python eval/harness.py export-kaggle"),
                    (manifest_path, f"no manifest {manifest_name} — pass --manifest-file")):
        if not p.exists():
            sys.exit(f"Missing {p} — run: {hint}")
    if args.train_file != "raft_train.jsonl":
        print(f"NOTE: packaging {args.train_file}, NOT the published 224-row raft_train.jsonl. "
              f"Any row trained from this is a CHALLENGER arm and must be named as one.")

    rows = [json.loads(l) for l in open(raft_path) if l.strip()]
    split_rows = make_split(rows)

    STAGE.mkdir(parents=True, exist_ok=True)
    for old in STAGE.iterdir():
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    with open(STAGE / "raft_train_split.jsonl", "w") as f:
        for r in split_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(replay_path, STAGE / "smoltalk_replay.jsonl")
    shutil.copy(tasks_path, STAGE / "eval_tasks_kaggle.jsonl")
    shutil.copy(manifest_path, STAGE / "raft_manifest.json")
    if args.adapter:
        src = Path(args.adapter)
        if not (src / "adapter_model.safetensors").exists():
            sys.exit(f"--adapter: no adapter_model.safetensors in {src}")
        shutil.copytree(src, STAGE / "adapter")
        print(f"staged adapter from {src} -> adapter/ (for --eval-only)")

    user = kaggle_username(required=not args.stage_only) or "KAGGLE_USER"
    (STAGE / "dataset-metadata.json").write_text(json.dumps({
        "id": f"{user}/{DATASET_SLUG}",
        "title": "uk-dslm RAFT private (disposable, plane-4)",
        "licenses": [{"name": "other"}],
    }, indent=1))
    print(f"staged {len(list(STAGE.iterdir()))} files -> {STAGE}")

    if args.stage_only:
        print("stage-only: not uploading. Upload later with --version or a create run."
              + (" (metadata has a placeholder username — re-run with kaggle.json in "
                 "place before uploading)" if user == "KAGGLE_USER" else ""))
        return

    cmd = ([kaggle_bin(), "datasets", "version", "-p", str(STAGE), "-m", args.version]
           if args.version else
           [kaggle_bin(), "datasets", "create", "-p", str(STAGE)])
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    combined = f"{r.stdout}\n{r.stderr}"
    print(combined.strip())
    # CLI 2.2.3 reports API failures as "Dataset creation error: …" on stdout
    # WITH exit code 0 — the exit code alone is not a success signal.
    if r.returncode != 0 or re.search(r"(?i)\berror\b\s*:|\berror\b(?!s)", combined):
        sys.exit("kaggle upload failed (see output above)")
    print(f"\nPRIVATE dataset: https://www.kaggle.com/datasets/{user}/{DATASET_SLUG}"
          "\nRemember: this is DISPOSABLE plane-4 storage (verbatim quotes inside)"
          " — delete it from the Kaggle website after the experiment.")


if __name__ == "__main__":
    main()
