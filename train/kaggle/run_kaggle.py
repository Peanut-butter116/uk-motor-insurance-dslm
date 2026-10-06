"""Headless Kaggle train loop: stage -> push -> poll -> fetch -> verify.

The whole fine-tune runs on Kaggle's free T4 with zero browser interaction
(after the one-time token + phone verification). This driver:
  1. stages kernel_train_eval.py (rewriting RUN_MODE per flag) + metadata;
  2. pushes with --accelerator NvidiaTeslaT4 AND machine_shape in the metadata
     (belt-and-braces: with enable_gpu and no machine_shape Kaggle defaults to
     P100, which hard-fails Unsloth — cc 6.0 < 7.0);
  3. polls `kaggle kernels status` until terminal;
  4. fetches /kaggle/working into the working project's artifacts/ dir;
  5. verifies the run actually produced what it should, exit non-zero if not.

Usage (policyqa env, ~/.kaggle/kaggle.json in place):
  python train/kaggle/run_kaggle.py --smoke      # ~20-min end-to-end sanity loop
  python train/kaggle/run_kaggle.py --full       # the real run
  python train/kaggle/run_kaggle.py --eval-only  # recovery: adapter from dataset
Smoke first, always. After the first green smoke run, copy the resolved
versions from versions.json into PINS in kernel_train_eval.py and commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
PROJ = Path(__file__).resolve().parents[2]
_candidates = [os.environ.get("POLICY_QA_HOME"),
               PROJ / "policy_qa",
               PROJ.parent / "policy_qa", PROJ]
HOME = next((Path(c) for c in _candidates if c and Path(c, "src", "common.py").exists()), PROJ)

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


# RESOLVED LAZILY, ON FIRST USE — and the laziness is the whole point.
#
# This used to be `KAGGLE = _kaggle_bin()`, evaluated at IMPORT time. b1cf5be fixed a bad
# assumption (the CLI is on the PATH) and shipped a harder one (the CLI exists on any machine
# that so much as IMPORTS this file). tests/test_kaggle_status_parse.py imports it during
# collection, GitHub's runners have no kaggle CLI, so pytest died with
#   INTERNALERROR> SystemExit: cannot find the `kaggle` CLI ...   /   no tests ran   /   exit 3
# and the suite stopped running on main entirely. A suite that never runs does not look red,
# it looks absent — every open PR still showed green, measuring a tree without the bad line.
#
# The failure behaviour is UNCHANGED and deliberately so: _kaggle_bin() still exits loudly,
# with the same message, the moment anything actually needs the CLI. Importing this module is
# now free. Gate: tests/test_no_import_time_exit.py.
_KAGGLE: str | None = None


def kaggle_bin() -> str:
    """The CLI path, resolved once and cached. Call this; never resolve at module scope."""
    global _KAGGLE
    if _KAGGLE is None:
        _KAGGLE = _kaggle_bin()
    return _KAGGLE


KERNEL_SLUG = "uk-dslm-raft-qlora"
DATASET_SLUG = "uk-dslm-raft-private"
POLL_S = 60
# Poller deadlines, in minutes. RAISED 2026-08-19 from full=360 after measuring the two real
# runs -- 360 would have abandoned a healthy kernel mid-run on any set larger than ~500 rows.
#
# THE COST IS NOT WHAT THE ROW COUNT SUGGESTS. Measured from the committed train_log.txt of the
# two full runs, the wall clock splits three ways and only ONE of them scales with the data:
#     setup + model load + quantisation   ~1h20m   FIXED
#     training                            ~25 min at 244-270 rows   SCALES
#     eval generation, 280 tasks          ~2h00m-2h30m   FIXED
#   run 20260722T080000Z (244 rows): 23:49:29 -> 01:30:59 train done -> 04:02:18  = 4h12m49s
#   run 20260723T072126Z (270 rows): 03:30:46 -> 05:17:42 train done -> 07:20:03  = 3h49m17s
# So scaling the training set 7.6x to the ~2,050-row union costs roughly three EXTRA hours of
# training, not 7.6x of everything:  1h20m + ~3h10m + ~2h15m = ~6h45m.
# 660 min leaves headroom under Kaggle's ~12h session cap without waiting all day on a hung run.
# If you shrink the eval task list, the fixed ~2h eval term shrinks with it.
TIMEOUT_MIN = {"smoke": 40, "full": 660, "eval_only": 180}


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def kaggle_username() -> str:
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        sys.exit("No ~/.kaggle/kaggle.json — create an API token at kaggle.com "
                 "Settings -> API, save it there, chmod 600 it, and re-run.")
    return json.loads(cred.read_text())["username"]


def stage(run_mode: str, user: str, nonce: str) -> Path:
    stage_dir = HOME / "kaggle_staging" / "kernel"
    stage_dir.mkdir(parents=True, exist_ok=True)
    src = (KAGGLE_DIR / "kernel_train_eval.py").read_text()
    src, n = re.subn(r'^RUN_MODE = "\w+"', f'RUN_MODE = "{run_mode}"', src,
                     count=1, flags=re.M)
    assert n == 1, "RUN_MODE line not found in kernel_train_eval.py"
    src, n = re.subn(r'^RUN_NONCE = "\w+"', f'RUN_NONCE = "{nonce}"', src,
                     count=1, flags=re.M)
    assert n == 1, "RUN_NONCE line not found in kernel_train_eval.py"
    (stage_dir / "kernel_train_eval.py").write_text(src)
    (stage_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}",
        # Title must slugify to KERNEL_SLUG: CLI >=2.x derives the kernel's URL
        # slug from the TITLE (not the id), so "uk-dslm RAFT QLoRA (T4)" pushed
        # to .../uk-dslm-raft-qlora-t4 while status/output polled the id slug —
        # 40 min of "unparseable status" then a failed fetch (2026-07-21 smoke).
        "title": KERNEL_SLUG,
        "code_file": "kernel_train_eval.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=1))
    print(f"staged kernel (RUN_MODE={run_mode}) -> {stage_dir}")
    return stage_dir


def push(stage_dir: Path) -> None:
    r = sh([kaggle_bin(), "kernels", "push", "-p", str(stage_dir),
            "--accelerator", "NvidiaTeslaT4"])
    if r.returncode != 0 and ("--accelerator" in (r.stderr or "")
                              or "unrecognized" in (r.stderr or "")):
        print("! this kaggle CLI lacks --accelerator; relying on machine_shape "
              "in kernel-metadata.json (pip install -U kaggle to fix)")
        r = sh([kaggle_bin(), "kernels", "push", "-p", str(stage_dir)])
    combined = f"{r.stdout}\n{r.stderr}"
    # CLI 2.2.3 reports API-level push failures as "Kernel push error: …" on
    # stdout WITH exit code 0 — the exit code alone is not a success signal.
    if r.returncode != 0 or re.search(r"(?i)\b(push|kernel)\s+error\b|error:", combined):
        sys.exit(f"push failed:\n{combined.strip()}")
    print(r.stdout.strip())


def _status_state(line: str) -> str:
    """Extract the normalized kernel state from a `has status "…"` line.

    CLI v2 drift #4 (2026-07-23): the status token became a dotted enum repr —
    `"KernelWorkerStatus.COMPLETE"` where it used to be `"complete"`. A bare
    [A-Za-z]+ capture stopped at the dot, returned 'kernelworkerstatus' for
    EVERY state, and the poller ran every run to its timeout (the T-062 smoke
    completed in 17 min and 'timed out' at 40). Take the token after the last
    dot; both formats normalize to the same state words."""
    m = re.search(r'status\s+"?([A-Za-z.]+)"?', line)
    return m.group(1).lower().rsplit(".", 1)[-1] if m else ""


def poll(user: str, timeout_min: int) -> str:
    """Parse the literal `has status "<state>"` token — substring matching the
    whole output would misread transient CLI/network error text as terminal,
    and 'complete' could shadow an error line printed alongside it."""
    ref = f"{user}/{KERNEL_SLUG}"
    deadline = time.time() + timeout_min * 60
    last, seen_active = "", False
    while time.time() < deadline:
        r = sh([kaggle_bin(), "kernels", "status", ref])
        line = (r.stdout or r.stderr).strip()
        if line != last:
            print(f"  [{dt.datetime.now():%H:%M:%S}] {line}")
            last = line
        state = _status_state(line)
        if state in ("running", "queued", "kernelworkerstarted", "kernelworkerpending"):
            seen_active = True
        elif state == "complete":
            if seen_active:
                return "complete"
            # complete on the FIRST poll = probably a stale previous version;
            # the nonce check in fetch_and_verify is the real guard — note it.
            print("  (status already 'complete' before this run was seen active "
                  "— relying on the manifest nonce to reject stale artifacts)")
            return "complete"
        elif state in ("error", "cancelacknowledged", "cancelrequested"):
            return "error"
        elif not state:
            print("  (unparseable status output — transient? continuing to poll)")
        time.sleep(POLL_S)
    return "timeout"


def fetch_and_verify(user: str, run_mode: str, nonce: str) -> Path:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = HOME / "artifacts" / "kaggle_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    r = sh([kaggle_bin(), "kernels", "output", f"{user}/{KERNEL_SLUG}",
            "-p", str(out_dir)])
    print(r.stdout.strip() or r.stderr.strip())

    problems = []
    # GUARD_FAIL first — with the kernel's exit-0 contract this file, not the
    # version status, is the failure signal.
    if (out_dir / "GUARD_FAIL.txt").exists():
        problems.append("GUARD_FAIL.txt present: " + (out_dir / "GUARD_FAIL.txt").read_text().strip())
    log_path = out_dir / "train_log.txt"
    log_text = log_path.read_text() if log_path.exists() else ""
    if not log_text:
        problems.append("train_log.txt missing/empty")
    else:
        for marker in ("GPU_GUARD_OK", "TRAIN COMPLETE"):
            if marker not in log_text:
                problems.append(f"log lacks {marker!r}")
    # bind the artifacts to THIS push (status/output always point at the
    # latest version — a stale previous run must not pass verification)
    mani_path = out_dir / "MANIFEST.json"
    if mani_path.exists():
        mani = json.loads(mani_path.read_text())
        if mani.get("run_nonce") != nonce:
            problems.append(f"nonce mismatch: artifacts are from a DIFFERENT run "
                            f"({mani.get('run_nonce')} != {nonce})")
        if mani.get("run_mode") != run_mode:
            problems.append(f"run_mode mismatch: {mani.get('run_mode')} != {run_mode}")
    if run_mode != "eval_only":
        adapter = out_dir / "qwen25-7b-raft-lora"
        if not (adapter / "adapter_model.safetensors").exists():
            problems.append("adapter_model.safetensors missing")
        for f in ("versions.json", "loss_history.json"):
            if not (out_dir / f).exists():
                problems.append(f"{f} missing")
    expect_rows = 2 if run_mode == "smoke" else None
    answer_files = sorted(out_dir.glob("answers_*.jsonl"))
    expected_n = 2 if run_mode == "eval_only" else 4
    if len(answer_files) != expected_n:
        problems.append(f"{len(answer_files)} answers files (expected {expected_n})")
    for af in answer_files:
        n = sum(1 for l in open(af) if l.strip())
        if expect_rows and n != expect_rows:
            problems.append(f"{af.name}: {n} rows (smoke expects {expect_rows})")
        print(f"  {af.name}: {n} rows")
    if not (out_dir / "MANIFEST.json").exists():
        problems.append("MANIFEST.json missing")

    if problems:
        print("\nVERIFY FAILED:")
        for p in problems:
            print("  -", p)
        if log_text:
            print("\n--- last 30 log lines ---")
            print("\n".join(log_text.splitlines()[-30:]))
        sys.exit(1)
    print(f"\nVERIFY OK — artifacts in {out_dir}")
    if run_mode == "full":
        print("Next: copy the four answers_*.jsonl into eval/results/, then\n"
              "  python eval/judge.py prepare   (judge via Claude Code)\n"
              "  python eval/judge.py aggregate (leaderboard)\n"
              "And pin versions.json into PINS if not yet done.")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    run_mode = "smoke" if args.smoke else "full" if args.full else "eval_only"

    user = kaggle_username()
    nonce = dt.datetime.now(dt.timezone.utc).strftime("n%Y%m%d%H%M%S")
    stage_dir = stage(run_mode, user, nonce)
    push(stage_dir)
    print(f"pushed (nonce {nonce}); polling every {POLL_S}s "
          f"(timeout {TIMEOUT_MIN[run_mode]} min)…")
    status = poll(user, TIMEOUT_MIN[run_mode])
    print(f"terminal status: {status}")
    # fetch even on error/timeout — whatever published may tell us what broke.
    # NOTE the kernel always exits 0 (see its EXIT CONTRACT), so 'error' here
    # means an infra-level failure (image, quota, verification) — and success
    # is decided by fetch_and_verify's artifact checks, not by the status.
    fetch_and_verify(user, run_mode, nonce)
    if status != "complete":
        sys.exit(1)


if __name__ == "__main__":
    main()
