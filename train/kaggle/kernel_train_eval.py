"""Self-contained Kaggle kernel: QLoRA-train Qwen2.5-7B on the RAFT set, then
generate the full 4-row eval board (base/tuned x closedbook/openbook) in one
session. Pushed headlessly via train/kaggle/run_kaggle.py — see README there.

Design (council-specified):
  - GPU GUARD before anything heavy: P100 (cc 6.0) hard-fails Unsloth; T4 is 7.5.
  - Own log file (/kaggle/working/train_log.txt): the CLI-fetched .log is often
    empty (kaggle-cli issue #603), so stdout is not trusted to survive.
  - Base rows A (closedbook) + B' (openbook) are generated BEFORE the LoRA is
    attached — same bnb-4bit backend as the tuned rows C/D, which removes the
    MLX-vs-bnb quantisation confound the pilot rows carry.
  - Prompts come PRE-RENDERED from eval_tasks_kaggle.jsonl (built by
    eval/harness.py export-kaggle with the real harness code) — the kernel never
    re-implements prompt construction, so rows stay byte-comparable to the pilot.
  - Record-then-pin installs: PINS starts loose; after the first green smoke run
    the resolved versions from versions.json get hard-coded here.
  - EXIT CONTRACT: the kernel ALWAYS exits 0, even on failure. Kaggle does not
    publish the /kaggle/working snapshot for FAILED kernel versions, so a
    non-zero exit would destroy exactly the diagnostics (train_log.txt,
    GUARD_FAIL.txt) a headless run needs. Failure is signalled by artifacts:
    GUARD_FAIL.txt present = failed; MANIFEST.json present = succeeded.
    run_kaggle.py verifies on artifacts, never on the version status.
  - RUN_MODE / RUN_NONCE are rewritten by run_kaggle.py in the STAGED copy only
    (the nonce binds fetched artifacts to THIS push — stale-version guard):
      smoke     : max_steps=8, 2 eval tasks per row  (~20 min end-to-end)
      full      : the real run
      eval_only : skip training; load adapter from the input dataset (recovery)
"""
RUN_MODE = "full"
RUN_NONCE = "dev"

import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # single-GPU Unsloth

WORKING = "/kaggle/working"
# DATA is resolved by glob AFTER the excepthook is installed (see below) — a
# hardcoded ref-slug mount guess killed two 2026-07-21 smokes: Kaggle derives
# the /kaggle/input/<dir> name from the dataset TITLE slug, not the ref.
DATA = None  # assigned post-excepthook; every use sits below that point
OUT_DIR = f"{WORKING}/qwen25-7b-raft-lora"
LOG_PATH = f"{WORKING}/train_log.txt"
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
# 5120, not 4096: the longest exported eval prompt measures 3635 tokens and
# generation adds up to 700 more (4335 total) — 4096 would overflow the KV cache
# on the 4 longest open-book tasks. Training rows max ~1.8k tokens either way.
MAX_SEQ_LEN = 5120
GEN_BATCH = 8

# Record-then-pin: pinned 2026-07-21 from the first GREEN smoke's versions.json
# (run 20260721T230536Z, nonce n20260721230536 — T4, loss 0.899 @ step 8).
# torch/transformers etc. ride the Kaggle image; we pin what pip resolves.
PINS = ["unsloth==2026.7.4", "unsloth_zoo==2026.7.4", "trl==0.24.0",
        "peft==0.19.1", "bitsandbytes==0.49.2", "accelerate==1.13.0",
        "xformers==0.0.35", "datasets==4.3.0"]
FALLBACK_PINS = ["unsloth", "xformers", "trl", "peft", "accelerate", "bitsandbytes"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def fail(msg: str) -> None:
    """Exit 0 ON PURPOSE: only 'complete' versions publish /kaggle/working, and
    the guard file + log ARE the failure signal (see EXIT CONTRACT above)."""
    log(f"FATAL: {msg}")
    with open(f"{WORKING}/GUARD_FAIL.txt", "w") as f:
        f.write(msg + "\n")
    sys.exit(0)


def _excepthook(exc_type, exc, tb):
    """Uncaught exception -> log + GUARD_FAIL + exit 0, so the output snapshot
    (and with it the diagnostics) still publishes. os._exit forces status 0 —
    a plain exception would end the process 1 and Kaggle would drop the files."""
    import traceback
    log("FATAL (uncaught): " + "".join(traceback.format_exception(exc_type, exc, tb))[-4000:])
    with open(f"{WORKING}/GUARD_FAIL.txt", "w") as f:
        f.write(f"uncaught: {exc_type.__name__}: {exc}\n")
    os._exit(0)


sys.excepthook = _excepthook

# Locate the dataset by its CONTENT, not a mount-name guess — glob survives any
# title/ref slug derivation Kaggle applies. Raising here is safe: the excepthook
# above turns it into GUARD_FAIL.txt + exit 0, preserving the output snapshot.
import glob as _glob                                        # noqa: E402
# Recursive: the 2026 layout nests mounts (observed: /kaggle/input/datasets/…),
# and single-level globbing missed it (smoke 3's diagnostic listing proved it).
_hits = _glob.glob("/kaggle/input/**/raft_train_split.jsonl", recursive=True)
if not _hits:
    raise FileNotFoundError(
        "raft_train_split.jsonl nowhere under /kaggle/input; tree (2 levels): "
        f"{sorted(_glob.glob('/kaggle/input/*/*'))}")
DATA = _hits[0].rsplit("/", 1)[0]
log(f"dataset mount resolved: {DATA}")

# ---------------------------------------------------------------- GPU guard
import torch  # noqa: E402  (present in the Kaggle base image)

if not torch.cuda.is_available():
    fail("no CUDA device — was the accelerator set?")
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
if cap[0] < 7:
    fail(f"GPU {name} has compute capability {cap[0]}.{cap[1]} < 7.0 — "
         "Unsloth cannot run (P100 trap). Select the T4 accelerator.")
log(f"GPU_GUARD_OK {name} cc={cap[0]}.{cap[1]} | RUN_MODE={RUN_MODE}")

# ------------------------------------------------------------------ install
def pip_install(pkgs: list[str], extra: list[str] | None = None) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + (extra or []) + pkgs
    log(f"pip install: {' '.join(pkgs)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"pip failed (rc={r.returncode}): {r.stderr[-2000:]}")
    return r.returncode == 0


t0 = time.time()
if os.environ.get("UK_DSLM_FALLBACK_DONE") != "1":
    if not pip_install(PINS):
        fail("primary pip install failed")
try:
    from unsloth import FastLanguageModel
except Exception as e:                                        # noqa: BLE001
    if os.environ.get("UK_DSLM_FALLBACK_DONE") == "1":
        fail(f"unsloth import failed even after fallback restart: {e}")
    log(f"unsloth import failed ({e}); installing fallback set + RESTARTING "
        "the interpreter (already-imported modules like torch stay cached in "
        "sys.modules — an in-process re-import would not pick up new wheels)")
    if not pip_install(FALLBACK_PINS):
        fail("fallback pip install failed")
    os.environ["UK_DSLM_FALLBACK_DONE"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)     # never returns

import importlib.metadata as _md                              # noqa: E402

versions = {}
for pkg in ("unsloth", "unsloth_zoo", "trl", "peft", "transformers",
            "bitsandbytes", "accelerate", "torch", "xformers", "datasets"):
    try:
        versions[pkg] = _md.version(pkg)
    except _md.PackageNotFoundError:
        versions[pkg] = None
# runtime truth beats dist-info for anything imported before pip ran (torch):
versions["torch_runtime"] = torch.__version__
with open(f"{WORKING}/versions.json", "w") as f:
    json.dump(versions, f, indent=1)
log(f"install done in {time.time() - t0:.0f}s | versions: {versions}")

# --------------------------------------------------------------------- data
def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


raft = read_jsonl(f"{DATA}/raft_train_split.jsonl")
replay = read_jsonl(f"{DATA}/smoltalk_replay.jsonl")
eval_tasks = read_jsonl(f"{DATA}/eval_tasks_kaggle.jsonl")
train_rows = [r for r in raft if r.get("split") == "train"] + replay
val_rows = [r for r in raft if r.get("split") == "val"]
log(f"data: {len(train_rows)} train ({len(replay)} replay) | {len(val_rows)} val | "
    f"{len(eval_tasks)} eval tasks")
if RUN_MODE == "smoke":
    by_mode: dict[str, list] = {}
    for t in eval_tasks:
        by_mode.setdefault(t["mode"], []).append(t)
    eval_tasks = [t for m in by_mode.values() for t in m[:2]]
    log(f"smoke: eval tasks cut to {len(eval_tasks)}")

# -------------------------------------------------------------------- model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True, dtype=None,               # None -> fp16 on T4 (pre-Ampere)
)
log(f"base model loaded: {BASE_MODEL} "
    f"(commit {getattr(model.config, '_commit_hash', 'unknown')})")

# ---------------------------------------------------------------- eval gen
def generate_rows(row_name: str, model_label: str) -> None:
    """Batched greedy generation over the pre-rendered eval tasks; harness schema."""
    try:
        FastLanguageModel.for_inference(model)
    except Exception as e:                                    # noqa: BLE001
        log(f"for_inference unavailable ({e}) — plain eval() mode")
        model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = {"openbook": [], "closedbook": []}
    tasks = sorted(eval_tasks, key=lambda t: t["mode"])
    for i in range(0, len(tasks), GEN_BATCH):
        batch = tasks[i:i + GEN_BATCH]
        prompts = [tokenizer.apply_chat_template(t["messages"], tokenize=False,
                                                 add_generation_prompt=True)
                   for t in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        t_gen = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=700, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=tokenizer.pad_token_id)
        dt_batch = time.time() - t_gen
        for j, t in enumerate(batch):
            text = tokenizer.decode(gen[j][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            # defensive <think> strip (should never trigger on Qwen2.5)
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            out[t["mode"]].append({
                "qid": t["qid"],
                "row": f"{row_name}-{t['mode']}",
                "model": model_label,
                "open_book": t["mode"] == "openbook",
                "answer": text.strip(),
                "latency_s": round(dt_batch / len(batch), 2),
                "deviations": ["bnb-4bit on Kaggle T4 (in-kernel row)",
                               "greedy decode, batched"],
            })
        log(f"  [{row_name}] {min(i + GEN_BATCH, len(tasks))}/{len(tasks)} "
            f"({dt_batch / len(batch):.1f}s/answer)")
    for mode, rows in out.items():
        path = f"{WORKING}/answers_{row_name}-{mode}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"  wrote {len(rows)} rows -> {path}")


if RUN_MODE != "eval_only":
    log("generating BASE rows (A closedbook, B' openbook) before LoRA…")
    generate_rows("qwen2.5-7b-bnb", BASE_MODEL)

# -------------------------------------------------------------------- train
if RUN_MODE == "eval_only":
    from peft import PeftModel
    adapter_dir = f"{DATA}/adapter"
    log(f"eval_only: loading adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
else:
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset
    import inspect

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )

    def to_text(rows: list[dict]) -> Dataset:
        return Dataset.from_list([{
            "text": tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                                  add_generation_prompt=False)
        } for r in rows])

    train_ds, val_ds = to_text(train_rows), to_text(val_rows)

    # --- TRL API-drift shim: build kwargs only from accepted parameters ---
    cfg_params = set(inspect.signature(SFTConfig.__init__).parameters)
    cfg_kwargs = dict(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,   # eff ~16
        num_train_epochs=2, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.05,
        fp16=True, bf16=False,                     # T4: fp16
        optim="adamw_8bit", logging_steps=5, seed=0,
        report_to="none",
    )
    if RUN_MODE == "smoke":
        cfg_kwargs["max_steps"] = 8
    if val_rows:
        for key, val in (("eval_strategy", "epoch"), ("evaluation_strategy", "epoch")):
            if key in cfg_params:
                cfg_kwargs[key] = val
                break
    if "dataset_text_field" in cfg_params:
        cfg_kwargs["dataset_text_field"] = "text"
    for key in ("max_seq_length", "max_length"):
        if key in cfg_params:
            cfg_kwargs[key] = MAX_SEQ_LEN
            break
    cfg = SFTConfig(**{k: v for k, v in cfg_kwargs.items() if k in cfg_params or k == "output_dir"})

    tr_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    tr_kwargs = dict(model=model, args=cfg, train_dataset=train_ds)
    if val_rows:
        tr_kwargs["eval_dataset"] = val_ds
    tr_kwargs["processing_class" if "processing_class" in tr_params else "tokenizer"] = tokenizer
    if "dataset_text_field" in tr_params and "dataset_text_field" not in cfg_kwargs:
        tr_kwargs["dataset_text_field"] = "text"
    trainer = SFTTrainer(**tr_kwargs)

    # Assistant-only loss where the API allows it (right objective for form-training).
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        log("train_on_responses_only: ON (assistant-only loss)")
    except Exception as e:                                    # noqa: BLE001
        log(f"train_on_responses_only unavailable ({e}) — full-text loss fallback")

    t0 = time.time()
    result = trainer.train()
    log(f"TRAINING done in {(time.time() - t0) / 60:.1f} min | "
        f"final loss {result.training_loss:.4f}")
    with open(f"{WORKING}/loss_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)

    model.save_pretrained(OUT_DIR)                # adapter only — base stays frozen
    tokenizer.save_pretrained(OUT_DIR)
    log(f"adapter saved -> {OUT_DIR}")

# --------------------------------------------------------------- tuned rows
log("generating TUNED rows (C closedbook, D openbook)…")
generate_rows("qwen2.5-7b-raft", f"{BASE_MODEL}+raft-lora")

# ----------------------------------------------------------------- manifest
manifest = {
    "run_mode": RUN_MODE,
    "run_nonce": RUN_NONCE,
    "base_model": BASE_MODEL,
    "base_commit": getattr(model.config, "_commit_hash", None),
    "gpu": name, "compute_capability": f"{cap[0]}.{cap[1]}",
    "train_rows": len(train_rows), "replay_rows": len(replay),
    "val_rows": len(val_rows), "eval_tasks": len(eval_tasks),
    "versions": versions,
    "note": ("Rung-1 provisional: gold verification pending on a subset — see "
             "raft_manifest.json in the input dataset for the exact count. "
             "Forgetting probe suite deferred to session 2 (pre-registration prep)."),
}
with open(f"{WORKING}/MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=1)
log("TRAIN COMPLETE")

# ALLOW-IMPORT-TIME-EXIT: this file is not a module, it is the script Kaggle executes top to
# bottom inside the kernel. Its module-level fail() calls ARE its control flow — the GPU guard
# has to stop the run before a two-hour job starts on a P100. It is never imported: it cannot
# be, off-Kaggle, since it imports torch and unsloth and globs /kaggle/input, and
# tests/test_no_import_time_exit.py separately asserts no test references it. Waiver placed at
# EOF on purpose: paper/appendix_reproducibility.md anchors :47 and :326 and
# docs/T118_PREREG_KAGGLE_EVAL_ONLY.md anchors :245-249 and :332, so nothing above may shift.
