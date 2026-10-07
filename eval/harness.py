"""Model-comparison harness — the "model candidacy" run.

Fairness protocol (council-specified):
  - retrieval is frozen ONCE per question (freeze_contexts) and the identical
    byte-for-byte context is given to every model row;
  - one OpenAI-compatible client for all rows; temp 0, equal max_tokens;
  - local rows run strictly SEQUENTIALLY with `lms unload --all` between model
    switches (16 GB M2 — two resident models would swap-thrash);
  - <think> traces stripped defensively from EVERY row;
  - all raw outputs cached as JSONL (resumable; leaderboard recomputable).

Rows:
  qwen3-8b-openbook      local, frozen context
  qwen3-8b-closedbook    local, NO context — the RAG-delta + contamination row
  qwen2.5-7b-openbook    local, frozen context (second model)
  claude-openbook        EXTERNAL: prompts exported to external_tasks_claude.jsonl,
                         answered via Claude Code (no API key on this machine);
                         recorded protocol deviation: not a pinned API model.

Usage:
  python eval/harness.py freeze          # freeze retrieval contexts for gold_v0
  python eval/harness.py run --row qwen3-8b-openbook
  python eval/harness.py run --all-local
  python eval/harness.py export-external # write Claude-row prompt tasks
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJ = Path(__file__).resolve().parents[1]
_candidates = [os.environ.get("POLICY_QA_HOME"),
               PROJ / "policy_qa",
               PROJ.parent / "policy_qa", PROJ]
_home = next((Path(c) for c in _candidates if c and Path(c, "src", "common.py").exists()), PROJ)
for sub in ("eval", "src"):
    if (_home / sub).exists():
        sys.path.insert(0, str(_home / sub))
from common import (  # noqa: E402
    ABSTAIN_SENTENCE, LMSTUDIO_BASE_URL, RESULTS_DIR,
    gold_path, read_jsonl, strip_think, write_jsonl,
)
import rag  # noqa: E402

GOLD_PATH = gold_path()
# v2 gold gets its own frozen-contexts file (the v1 pilot file stays untouched
# so the pilot leaderboard remains reproducible); default split under v2 = the
# frozen test set. Use --split dev for smoke runs.
V2 = GOLD_PATH.name == "gold_v2.jsonl"
CONTEXTS_PATH = RESULTS_DIR / ("frozen_contexts_v2.jsonl" if V2 else "frozen_contexts.jsonl")
DEFAULT_SPLIT = "test" if V2 else None
MAX_TOKENS = 700
K = 6


def gold_records(split: str | None = None) -> list[dict]:
    recs = read_jsonl(GOLD_PATH)
    split = split or DEFAULT_SPLIT
    if split and split != "all":
        recs = [g for g in recs if g.get("split") == split]
    return recs

ROWS = {
    "qwen3-8b-openbook": {
        "model": "qwen/qwen3-8b", "open_book": True, "provider": "lmstudio",
        "deviations": ["temp 0 in non-thinking mode (card suggests 0.7/0.8) — deviation #1",
                       "4-bit MLX quant"],
    },
    "qwen3-8b-closedbook": {
        "model": "qwen/qwen3-8b", "open_book": False, "provider": "lmstudio",
        "deviations": ["closed-book control: no retrieved context — RAG-delta/contamination row"],
    },
    "qwen2.5-7b-openbook": {
        "model": "qwen2.5-7b-instruct", "open_book": True, "provider": "lmstudio",
        "deviations": ["4-bit MLX quant"],
    },
    "claude-openbook": {
        "model": "claude-via-claude-code", "open_book": True, "provider": "external",
        "deviations": ["ceiling row generated through Claude Code, not a pinned API model — documented"],
    },
}

CLOSED_BOOK_SYSTEM = """You are a UK insurance policy-wording assistant. {style}

No policy extracts are provided. Answer from your own knowledge of the CURRENT policy wording of the insurer named in the question. If you cite, use the format [Insurer, Doc, Section, p.N] only for sections you are genuinely sure exist. If you do not reliably know what this insurer's current wording says, reply with exactly this sentence and nothing else: "{abstain}"
Never give advice, recommendations, pricing, or eligibility judgements."""


def freeze_contexts(split: str | None = None) -> None:
    gold = gold_records(split)
    if not gold:
        sys.exit(f"No gold questions at {GOLD_PATH}")
    rows = []
    for g in gold:
        chunks = rag.retrieve(g["question"], k=K)
        rows.append({
            "qid": g["id"],
            "chunks": chunks,
            "context_text": rag.format_context(chunks),
        })
        print(f"  froze {g['id']} ({len(chunks)} chunks)")
    write_jsonl(CONTEXTS_PATH, rows)
    print(f"Frozen contexts for {len(rows)} questions -> {CONTEXTS_PATH}")


def build_row_messages(g: dict, ctx: dict | None, open_book: bool) -> list[dict]:
    persona = g.get("persona", "end_user")
    if open_book:
        return rag.build_messages(g["question"], persona, ctx["chunks"])
    system = CLOSED_BOOK_SYSTEM.format(
        style=rag.PERSONA_STYLE[persona], abstain=ABSTAIN_SENTENCE
    ) + " /no_think"
    return [{"role": "system", "content": system},
            {"role": "user", "content": g["question"]}]


def unload_all() -> None:
    try:
        subprocess.run([str(Path.home() / ".lmstudio/bin/lms"), "unload", "--all"],
                       capture_output=True, timeout=60)
        print("  (unloaded all local models)")
    except Exception as e:
        print(f"  ! unload failed: {e}")


def run_row(row_id: str, split: str | None = None) -> None:
    row = ROWS[row_id]
    if row["provider"] != "lmstudio":
        sys.exit(f"{row_id} is external — use export-external, then answer via Claude Code.")
    gold = gold_records(split)
    contexts = {c["qid"]: c for c in read_jsonl(CONTEXTS_PATH)}
    out_path = RESULTS_DIR / f"answers_{row_id}.jsonl"
    done = {r["qid"] for r in read_jsonl(out_path)}
    answers = read_jsonl(out_path)
    from openai import OpenAI
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")

    for g in gold:
        if g["id"] in done:
            continue
        ctx = contexts.get(g["id"])
        if row["open_book"] and not ctx:
            sys.exit(f"No frozen context for {g['id']} — run freeze first.")
        messages = build_row_messages(g, ctx, row["open_book"])
        t0 = time.time()
        resp = client.chat.completions.create(
            model=row["model"], messages=messages,
            temperature=0.0, max_tokens=MAX_TOKENS,
        )
        text = strip_think(resp.choices[0].message.content or "")
        answers.append({
            "qid": g["id"], "row": row_id, "model": row["model"],
            "open_book": row["open_book"], "answer": text,
            "latency_s": round(time.time() - t0, 2),
            "deviations": row["deviations"],
        })
        write_jsonl(out_path, answers)          # write-through: resumable
        print(f"  [{row_id}] {g['id']} done in {answers[-1]['latency_s']}s "
              f"({len(answers)}/{len(gold)})")
    print(f"Row {row_id} complete -> {out_path}")


def export_external(split: str | None = None) -> None:
    gold = gold_records(split)
    contexts = {c["qid"]: c for c in read_jsonl(CONTEXTS_PATH)}
    tasks = []
    for g in gold:
        messages = build_row_messages(g, contexts.get(g["id"]), open_book=True)
        tasks.append({"qid": g["id"], "row": "claude-openbook",
                      "system": messages[0]["content"], "user": messages[1]["content"]})
    path = RESULTS_DIR / "external_tasks_claude-openbook.jsonl"
    write_jsonl(path, tasks)
    print(f"{len(tasks)} external tasks -> {path}\n"
          "Answer them via Claude Code into answers_claude-openbook.jsonl "
          "(same schema as local rows).")


def export_kaggle(split: str | None = None) -> None:
    """Pre-render BOTH prompt styles for the Kaggle train+eval kernel.

    The kernel must not re-implement prompt construction (drift risk): rows
    generated on Kaggle stay byte-identical to the pilot's open-book row B and
    the closed-book control, because the exact harness code renders them here.
    Gold answers are deliberately NOT exported — the kernel never needs them
    (scoring happens locally), minimising licensed text shipped to plane 4.
    """
    gold = gold_records(split)
    contexts = {c["qid"]: c for c in read_jsonl(CONTEXTS_PATH)}
    tasks = []
    for g in gold:
        ctx = contexts.get(g["id"])
        if not ctx:
            sys.exit(f"No frozen context for {g['id']} — run freeze first.")
        for mode, open_book in (("openbook", True), ("closedbook", False)):
            messages = build_row_messages(g, ctx if open_book else None, open_book)
            tasks.append({"qid": g["id"], "mode": mode,
                          "messages": messages, "max_tokens": MAX_TOKENS})
    path = RESULTS_DIR / "eval_tasks_kaggle.jsonl"
    write_jsonl(path, tasks)
    print(f"{len(tasks)} kernel eval tasks ({len(gold)} questions x 2 modes) -> {path}\n"
          "Ship this file in the private Kaggle dataset (train/kaggle/package_dataset.py).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", help="test | dev | seed | all (v2 default: test)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    runp = sub.add_parser("run")
    runp.add_argument("--row", choices=list(ROWS))
    runp.add_argument("--all-local", action="store_true")
    sub.add_parser("export-external")
    sub.add_parser("export-kaggle")
    args, s = ap.parse_known_args(), None
    args = args[0] if isinstance(args, tuple) else args
    s = args.split

    if args.cmd == "freeze":
        freeze_contexts(s)
    elif args.cmd == "export-external":
        export_external(s)
    elif args.cmd == "export-kaggle":
        export_kaggle(s)
    elif args.cmd == "run":
        if args.all_local:
            local = [r for r, cfg in ROWS.items() if cfg["provider"] == "lmstudio"]
            for i, r in enumerate(local):
                if i:
                    unload_all()
                run_row(r, s)
            unload_all()
        elif args.row:
            run_row(args.row, s)
        else:
            sys.exit("run needs --row or --all-local")


if __name__ == "__main__":
    main()
