"""Ingest UK insurer policy-wording PDFs into a ChromaDB index.

Pipeline (per the council-approved design):
  pymupdf4llm per-page markdown
  -> strip headers/footers recurring on >50% of pages
  -> section-first chunking (~300-450 tokens, 50-token overlap, never across sections)
  -> "Insurer | Doc | Section" context header + nomic "search_document: " prefix
  -> ChromaDB with PRECOMPUTED embeddings (never Chroma's default embedder)

Also maintains data/documents.jsonl: URL + date + SHA256 + licence-note per file
(the provenance manifest the council required), and data/chunks.jsonl for eyeballing.

Run:  python src/ingest.py            # ingest everything in data/raw_pdfs
      python src/ingest.py --spot-check   # 10 retrieval spot-checks after indexing
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf  # PyMuPDF
import pymupdf4llm
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHROMA_DIR, COLLECTION, DATA_DIR, EMBED_MODEL, LMSTUDIO_BASE_URL,
    RAW_PDF_DIR, TIER_RE, chunk_header, doc_facts_for, read_jsonl,
    sha256_of, write_jsonl,
)

CHUNK_TARGET_CHARS = 1700   # ~420 tokens at ~4 chars/token
CHUNK_MIN_CHARS = 350
OVERLAP_CHARS = 220         # ~50 tokens

# Section headings across insurers: markdown headings from pymupdf4llm, plus
# common explicit styles ("Section B2 — Baggage", "SECTION 3: ...", bold lines).
SEP_CHARS = ":\\-–—·•|."
HEADING_RES = [
    re.compile(r"^#{1,4}\s+(?P<title>.+)$"),
    re.compile(rf"^\**\s*Section\s+(?P<sec>[A-Z]?\d*[A-Za-z]?\d*)\s*[{SEP_CHARS}\s]*(?P<title>.*?)\**$", re.IGNORECASE),
    re.compile(r"^\*\*(?P<title>[^*]{4,80})\*\*$"),
]
SECTION_ID_RE = re.compile(rf"^Section\s+(?P<id>[A-Z]?\d+[A-Za-z]?\d*|[A-Z]\d*)\b\s*[{SEP_CHARS}\s]*(?P<rest>.*)$", re.IGNORECASE)
_STRIP = " *" + SEP_CHARS.replace("\\", "")


def detect_heading(line: str) -> tuple[str | None, str | None]:
    """Return (section_id, section_title) if the line is a heading, else (None, None)."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None, None
    for rx in HEADING_RES:
        m = rx.match(stripped)
        if not m:
            continue
        gd = m.groupdict()
        title = (gd.get("title") or "").strip(_STRIP)
        sec = (gd.get("sec") or "").strip()
        if sec:
            return sec.upper(), title or f"Section {sec.upper()}"
        if title:
            # "Section B2 — Baggage" arriving as a markdown/bold heading:
            # canonicalize the id so citations stay short and matchable.
            m2 = SECTION_ID_RE.match(title)
            if m2 and m2.group("id"):
                rest = m2.group("rest").strip(_STRIP)
                return m2.group("id").upper(), rest or title
            return title[:40], title
    return None, None


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Drop short lines that recur on >50% of pages (headers/footers/slogans)."""
    if len(pages) < 6:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        seen = set()
        for line in page.splitlines():
            key = re.sub(r"\d+", "#", line.strip().lower())
            if 3 <= len(key) <= 80 and key not in seen:
                counts[key] += 1
                seen.add(key)
    threshold = len(pages) * 0.5
    repeated = {k for k, c in counts.items() if c > threshold}
    cleaned = []
    for page in pages:
        kept = [
            line for line in page.splitlines()
            if re.sub(r"\d+", "#", line.strip().lower()) not in repeated
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def sections_from_pages(pages: list[str]) -> list[dict]:
    """Walk pages line-by-line, opening a new section at each heading."""
    sections: list[dict] = []
    current = {"section_id": "Preamble", "section_title": "Preamble", "page": 1, "lines": []}
    for pageno, page in enumerate(pages, start=1):
        for line in page.splitlines():
            sec_id, sec_title = detect_heading(line)
            if sec_id:
                if current["lines"] and any(l.strip() for l in current["lines"]):
                    sections.append(current)
                current = {"section_id": sec_id, "section_title": sec_title,
                           "page": pageno, "lines": []}
            else:
                current["lines"].append(line)
    if current["lines"] and any(l.strip() for l in current["lines"]):
        sections.append(current)
    for s in sections:
        s["text"] = re.sub(r"\n{3,}", "\n\n", "\n".join(s.pop("lines"))).strip()
    return [s for s in sections if len(s["text"]) >= 40]


def chunk_section(section: dict) -> list[str]:
    """Pack a section's paragraphs into ~CHUNK_TARGET_CHARS chunks with overlap.

    Markdown table rows are kept atomic (a split table scrambles limits/excesses).
    """
    text = section["text"]
    if len(text) <= CHUNK_TARGET_CHARS:
        return [text]
    blocks: list[str] = []
    for para in text.split("\n\n"):
        if para.strip().startswith("|"):          # whole table = one atomic block
            blocks.append(para)
        else:
            blocks.extend(p for p in [para] if p.strip())
    chunks, buf = [], ""
    for block in blocks:
        if buf and len(buf) + len(block) + 2 > CHUNK_TARGET_CHARS:
            chunks.append(buf.strip())
            buf = buf[-OVERLAP_CHARS:] + "\n\n" if len(buf) > OVERLAP_CHARS else ""
        buf += block + "\n\n"
    if buf.strip() and (len(buf.strip()) >= CHUNK_MIN_CHARS or not chunks):
        chunks.append(buf.strip())
    elif buf.strip() and chunks:
        chunks[-1] += "\n\n" + buf.strip()
    return chunks


def raw_page_markdown(page) -> str:
    """Raw-fallback page text with structure recovered from the PDF itself:
    font-size-aware headings (>=16pt -> ##, >=9.5pt -> ###) and column-aware
    block ordering (plain sort=True interleaves two-column layouts line-by-line
    — observed on the Post Office wording, which collapsed to 17 junk sections)."""
    d = page.get_text("dict")
    blocks = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        lines_out, max_size = [], 0.0
        for l in b.get("lines", []):
            txt = "".join(s["text"] for s in l.get("spans", []))
            if txt.strip():
                lines_out.append(txt)
                max_size = max(max_size, max((s["size"] for s in l["spans"]
                                              if s["text"].strip()), default=0.0))
        if lines_out:
            blocks.append((b["bbox"], "\n".join(lines_out), max_size))
    if not blocks:
        return page.get_text("text", sort=True)
    w = page.rect.width
    left = [b for b in blocks if b[0][2] < w * 0.55]
    right = [b for b in blocks if b not in left]
    if len(left) >= 2 and len(right) >= 2:
        ordered = sorted(left, key=lambda b: b[0][1]) + sorted(right, key=lambda b: b[0][1])
    else:
        ordered = sorted(blocks, key=lambda b: (round(b[0][1]), b[0][0]))
    out = []
    for _bbox, text, size in ordered:
        flat = " ".join(text.split())
        if size >= 16 and len(flat) < 80:
            out.append("## " + flat)
        elif size >= 9.5 and len(flat) < 80:
            out.append("### " + flat)
        else:
            out.append(text)
    return "\n\n".join(out)


def extract_pdf(path: Path) -> tuple[list[str], int]:
    """Per-page markdown; falls back to structure-recovering raw extraction if
    pymupdf4llm fails OR silently loses text on a page (observed in smoke
    testing: its layout analysis can drop body text — compare char counts per
    page and take the raw extraction for any page where markdown lost >35%)."""
    doc = pymupdf.open(path)
    npages = doc.page_count
    raw_pages = [raw_page_markdown(doc[i]) for i in range(npages)]
    try:
        page_dicts = pymupdf4llm.to_markdown(doc, page_chunks=True)
        pages = [p["text"] for p in page_dicts]
        for i in range(min(len(pages), npages)):
            raw_len = len(raw_pages[i].strip())
            if raw_len > 200 and len(pages[i].strip()) < 0.65 * raw_len:
                print(f"  ! {path.name} p.{i+1}: markdown lost text "
                      f"({len(pages[i].strip())} vs {raw_len} chars) — raw fallback for this page")
                pages[i] = raw_pages[i]
    except Exception as e:                        # per-doc fallback, pre-agreed
        print(f"  ! pymupdf4llm failed on {path.name} ({e}); raw-text fallback")
        pages = raw_pages
    doc.close()
    return pages, npages


def build_manifest(pdfs: list[Path]) -> list[dict]:
    manifest_path = DATA_DIR / "documents.jsonl"
    existing = {r["filename"]: r for r in read_jsonl(manifest_path)}
    rows = []
    for pdf in pdfs:
        facts = doc_facts_for(pdf.name)
        row = existing.get(pdf.name, {})
        row.update({
            "filename": pdf.name,
            "sha256": sha256_of(pdf),
            "bytes": pdf.stat().st_size,
            "downloaded": row.get("downloaded")
            or dt.datetime.fromtimestamp(pdf.stat().st_mtime).isoformat(timespec="seconds"),
            "acquisition": "manual browser download (terms-safe; no automated fetching)",
            **facts,
        })
        rows.append(row)
    write_jsonl(manifest_path, rows)
    return rows


def embed_batches(client: OpenAI, texts: list[str], batch: int = 32) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
        out.extend(d.embedding for d in resp.data)
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot-check", action="store_true")
    ap.add_argument("--chunks-only", action="store_true",
                    help="stop after writing data/chunks.jsonl — skips embedding "
                         "and the Chroma index, so no LM Studio is needed. This is "
                         "all review_gold.py requires, so a reviewer can set up "
                         "with just the PDFs + this repo.")
    args = ap.parse_args()

    pdfs = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {RAW_PDF_DIR} — download the corpus first.")
    print(f"{len(pdfs)} PDFs found; building manifest…")
    manifest = build_manifest(pdfs)
    for m in manifest:
        print(f"  {m['insurer']:<12} {m['doc_short']:<24} {m['filename']}")

    all_chunks: list[dict] = []
    for pdf in pdfs:
        facts = doc_facts_for(pdf.name)
        pages, npages = extract_pdf(pdf)
        total_chars = sum(len(p) for p in pages)
        if total_chars < npages * 100:
            print(f"  ! WARNING {pdf.name}: thin text layer "
                  f"({total_chars} chars / {npages} pages) — check extraction quality")
        pages = strip_repeated_lines(pages)
        sections = sections_from_pages(pages)
        n_before = len(all_chunks)
        for sec in sections:
            for j, text in enumerate(chunk_section(sec)):
                meta = {
                    "insurer": facts["insurer"],
                    "doc_short": facts["doc_short"],
                    "doc_title": facts["doc_title"],
                    "line": facts["line"],
                    "filename": pdf.name,
                    "section_id": sec["section_id"],
                    "section_title": sec["section_title"],
                    "page": sec["page"],
                    "part": j,
                }
                tiers = sorted(set(TIER_RE.findall(sec["section_title"] + " " + text[:400])))
                if tiers and facts["insurer"] in ("Admiral", "AA"):
                    meta["tiers"] = ", ".join(tiers)
                all_chunks.append({
                    "id": f"{pdf.stem}::{sec['section_id']}::{j}",
                    "text": text,
                    "meta": meta,
                })
        print(f"  {pdf.name}: {npages}pp -> {len(sections)} sections "
              f"-> {len(all_chunks) - n_before} chunks")

    # de-dup ids (same heading text can repeat across a doc)
    seen: Counter[str] = Counter()
    for c in all_chunks:
        seen[c["id"]] += 1
        if seen[c["id"]] > 1:
            c["id"] += f"~{seen[c['id']]}"

    write_jsonl(DATA_DIR / "chunks.jsonl", all_chunks)
    print(f"{len(all_chunks)} chunks total -> data/chunks.jsonl")

    if args.chunks_only:
        print("--chunks-only: stopping before embedding (no LM Studio needed).\n"
              "chunks.jsonl is everything eval/review_gold.py and "
              "eval/validate_gold.py need. Run without this flag to build the "
              "Chroma index for retrieval/RAG.")
        return

    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
    embed_texts = [
        "search_document: " + chunk_header(c["meta"]) + "\n" + c["text"]
        for c in all_chunks
    ]
    print("Embedding via LM Studio…")
    vectors = embed_batches(client, embed_texts)

    import chromadb
    cdb = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        cdb.delete_collection(COLLECTION)
    except Exception:
        pass
    col = cdb.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    B = 256
    for i in range(0, len(all_chunks), B):
        part = all_chunks[i:i + B]
        col.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=vectors[i:i + B],
            metadatas=[c["meta"] for c in part],
        )
    print(f"Indexed {col.count()} chunks into {CHROMA_DIR}/{COLLECTION}")

    if args.spot_check:
        from rag import retrieve
        queries = [
            "Is my mobile phone covered if it is stolen abroad?",
            "What is the single item limit for valuables?",
            "Am I covered if I cancel my trip because of illness?",
            "What excess do I pay on a baggage claim?",
            "Is accidental damage to my TV covered at home?",
            "Are golf clubs covered?",
            "What happens if I miss my flight departure?",
            "Is emergency medical treatment abroad covered?",
            "How do I make a claim?",
            "Is my bicycle covered away from home?",
        ]
        for q in queries:
            hits = retrieve(q, k=3)
            top = hits[0]["meta"] if hits else {}
            print(f"\nQ: {q}\n   -> {top.get('insurer')} | {top.get('section_id')} | "
                  f"{top.get('section_title')} (p.{top.get('page')})")


if __name__ == "__main__":
    main()
