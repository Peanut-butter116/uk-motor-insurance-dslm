"""Retrieval + citation-constrained generation for the Policy-Wording Q&A demo.

Two personas share one contract:
  - every factual claim ends with a citation [Insurer, Doc, Section, p.N]
    copied EXACTLY from a supplied extract header;
  - if the extracts don't support an answer, reply with the fixed abstention
    sentence (verbatim — the UI badge and the judge detect it);
  - never advice, recommendations, pricing or eligibility judgements.

verify_citations() regex-checks every citation against the retrieved chunk
metadata — the green badge in the app, and the embryo of the project's
citation-faithfulness metric.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ABSTAIN_SENTENCE, BUYER_ABSTAIN_SENTENCE, CHROMA_DIR, CITATION_RE, COLLECTION,
    EMBED_MODEL, GEN_MODEL, LMSTUDIO_BASE_URL, SIGNPOST_SENTENCE, chunk_header,
    norm, strip_think,
)

_client = None
_collection = None
_reranker = None


def client():
    """Lazy LM Studio client — same idiom as collection() below.

    Deliberately lazy: verify_citations() is pure logic and needs no LLM, so
    importing this module must not require `openai`. CI installs pytest alone
    and imports this module to test the citation resolver; a module-level
    `from openai import OpenAI` turned that into a collection error.
    """
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio", timeout=180.0)
    return _client

# Reranker choice is MEASURED, not assumed (A/B on the 21 answerable gold
# records, 2026-07-07): baseline embedding-order recall@6 = 44%; the top-24
# pool holds 68% (ranking headroom); ms-marco MiniLM HURT recall (42%);
# a keyword multi-query variant polluted the pool (56% ceiling); BGE base
# lifted recall@6 to 53% (k=8: 56%). So: single query -> pool 24 -> BGE.
RERANKER_MODEL = "BAAI/bge-reranker-base"


def collection():
    global _collection
    if _collection is None:
        import chromadb
        _collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    return _collection


def reranker():
    """Lazy-loaded cross-encoder; None if unavailable (falls back gracefully)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            print(f"! reranker unavailable ({e}) — falling back to embedding order")
            _reranker = False
    return _reranker or None


def _embed_search(query: str, n: int, where: dict | None) -> list[dict]:
    vec = client().embeddings.create(
        model=EMBED_MODEL, input=["search_query: " + query]
    ).data[0].embedding
    res = collection().query(query_embeddings=[vec], n_results=n, where=where)
    return [
        {"id": i, "text": d, "meta": m, "distance": dist}
        for i, d, m, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def retrieve(question: str, k: int = 6, insurers: list[str] | None = None,
             rerank: bool = True, pool: int = 24) -> list[dict]:
    """Single-query embedding pool -> cross-encoder rerank -> top-k.

    rerank=False reproduces the original embedding-order pipeline (used for
    A/B measurement and as the fallback when the reranker is absent).
    """
    where = {"insurer": {"$in": insurers}} if insurers else None
    if not rerank:
        return _embed_search(question, k, where)
    candidates = _embed_search(question, pool, where)
    ce = reranker()
    if ce is None:
        return candidates[:k]
    pairs = [(question, chunk_header(c["meta"]) + "\n" + c["text"][:1500]) for c in candidates]
    scores = ce.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:k]


PERSONA_STYLE = {
    "end_user": (
        "You are answering a POLICYHOLDER with no insurance background. Use plain,"
        " warm, everyday English. Short sentences. Explain any insurance term you"
        " must use (e.g. 'excess — the part of a claim you pay yourself')."
        " Address them as 'you'."
    ),
    "handler": (
        "You are answering a POLICY HANDLER (insurer staff). Be precise and"
        " technical. Quote the operative clause text verbatim in double quotation"
        " marks before interpreting it. List ALL relevant sections, including"
        " exclusions, conditions and cross-references. Note any per-tier"
        " differences (e.g. Admiral vs Gold vs Platinum)."
    ),
    # PERG-safe pre-purchase framing (SCHEMA.md persona enum): information about
    # what the wording says, never a recommendation or suitability judgement —
    # rule 4 of the system template still governs.
    "buyer": (
        "You are answering a PROSPECTIVE BUYER comparing policies before"
        " purchase. Use plain English and explain insurance terms. Lay out what"
        " the wording actually covers, excludes and limits — including"
        " differences between tiers or insurers when asked — so they can decide"
        " for themselves. Never recommend a product, judge suitability, or say"
        " what they should choose."
    ),
}

SYSTEM_TEMPLATE = """You are a UK insurance policy-wording assistant used inside a research demo. {style}

STRICT RULES — these override everything else:
1. Answer ONLY from the numbered policy extracts provided below. Your own knowledge of insurance must never supply a fact.
2. Every factual claim MUST end with a citation in square brackets copied EXACTLY from the header of the extract that supports it, in the form [Insurer, Doc, Section, p.N]. A sentence without a citation is a rule violation.
3. If the extracts do not contain enough information to answer, reply with exactly this sentence and nothing else: "{abstain}"
4. Never give advice, recommendations, opinions on whether to claim, pricing, or eligibility judgements. If asked for any of these, use rule 3's sentence.
5. If different insurers or cover tiers differ, say so explicitly rather than averaging them.
6. Answer about the specific insurer(s) asked about; if extracts from other insurers are present, ignore them unless the question compares insurers.

POLICY EXTRACTS:
{context}"""

NO_CONTEXT_PLACEHOLDER = "(no policy extracts provided)"


def format_context(chunks: list[dict], max_chars: int | None = None) -> str:
    """max_chars truncates each extract (buyer mode trims prefill for latency;
    Q&A personas pass None and keep full clause text)."""
    if not chunks:
        return NO_CONTEXT_PLACEHOLDER
    blocks = []
    for n, c in enumerate(chunks, 1):
        text = c["text"][:max_chars] if max_chars else c["text"]
        blocks.append(f"--- Extract {n}: {chunk_header(c['meta'])}\n{text}")
    return "\n\n".join(blocks)


def build_messages(question: str, persona: str, chunks: list[dict],
                   no_think: bool = True) -> list[dict]:
    system = SYSTEM_TEMPLATE.format(
        style=PERSONA_STYLE[persona],
        abstain=ABSTAIN_SENTENCE,
        context=format_context(chunks),
    )
    if no_think:
        system += " /no_think"       # Qwen3 soft switch; harmless elsewhere
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def answer(question: str, persona: str = "end_user", k: int = 6,
           insurers: list[str] | None = None, model: str = GEN_MODEL,
           stream: bool = False, temperature: float = 0.0,
           max_tokens: int = 600):
    """Returns (text, chunks) — or (generator, chunks) when stream=True."""
    chunks = retrieve(question, k=k, insurers=insurers)
    messages = build_messages(question, persona, chunks)
    if stream:
        resp = client().chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=True,
        )

        def gen():
            for part in resp:
                delta = part.choices[0].delta.content
                if delta:
                    yield delta
        return gen(), chunks
    resp = client().chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens,
    )
    return strip_think(resp.choices[0].message.content or ""), chunks


def _norm_doc(s: str) -> str:
    """Normalise a document token: strip any directory, a trailing .pdf, and case.

    Model-emitted doc tokens don't byte-match metadata (a model may write
    'IP-TR-1-014' where the filename is 'IP-TR-1-014.pdf', or vary case), so
    this is applied to BOTH sides of the lookup.
    """
    s = str(s).strip().strip("\"'").strip("_*# ").strip()
    s = s.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if s.lower().endswith(".pdf"):
        s = s[:-len(".pdf")]
    return " ".join(s.split()).casefold()


def _fold(s: str) -> str:
    """`norm()` plus hyphen-run collapsing, for fixed-sentence detection only.

    `norm()` already folds case, curly quotes, en/em dashes and whitespace, but it
    leaves `--` as two characters, and a generator that renders the em dash as `--`
    is still abstaining.
    """
    return re.sub(r"-{2,}", "-", norm(s))


# The three fixed refusal sentences, pre-folded once at import.
_REFUSAL_FOLDED = tuple(
    _fold(s).rstrip(" .")
    for s in (ABSTAIN_SENTENCE, BUYER_ABSTAIN_SENTENCE, SIGNPOST_SENTENCE)
)


def is_refusal(text: str) -> bool:
    """Did the model emit one of the three fixed refusal sentences?

    Compared under `_fold`, NOT as an exact substring. The exact test this replaces
    (`ABSTAIN_SENTENCE.rstrip("。 .") in text`) missed every dash, curly-quote,
    case and whitespace variant: measured on the published base row it scored
    **6 of 140** correct refusals as `abstained=False`, which the app then rendered
    as a red "No citations found — rule violation" — the system accusing itself of
    a rule violation for behaving correctly.

    It also recognises BUYER_ABSTAIN_SENTENCE and SIGNPOST_SENTENCE, which the old
    test never did, so buyer-mode abstention rendered no badge at all.

    This is the APP-SIDE detector. It is deliberately NOT the metric: the published
    abstention axis is computed by `eval/abstention.py` (adopted reading B), and
    `eval/judge.py:80` still ships a third, retired rule. Do not quote a badge in
    this app as evidence for a leaderboard number.
    """
    folded = _fold(text)
    return any(s in folded for s in _REFUSAL_FOLDED)


def verify_citations(text: str, chunks: list[dict]) -> dict:
    """Check every [Insurer, Doc, Section, p.N] citation against retrieved metadata.

    A citation is 'supported' when insurer + DOCUMENT + section match a retrieved
    chunk (page allowed ±1 — section text can span a page boundary).

    The document is part of the key on purpose (fixed 2026-07-15). Keying on
    insurer+section alone collapsed 60 distinct key-groups in the current corpus:
    Admiral's two travel IPIDs both carry section 'What is insured?' p.1, so a
    model could cite the single-trip IPID's £1,000/£200 limits against the annual
    IPID's £2,000/£300 and still score full citation_support (0.3 of the eval
    composite) — see gold record PQ-0051, which asks exactly that tier question.

    A chunk is citable by EITHER identifier the pipeline exposes for its document:
      - doc_short — what chunk_header() prints, so what a prompted model copies;
        since T-062 it is also what RAFT targets stamp (filename tags collapsed
        citation-support 90->15 on leaderboard v2 — the tuned model could not
        reproduce filenames it never saw in context);
      - filename  — kept for adapters trained on pre-T-062 RAFT data, whose
        answers legitimately cite filenames.
    The corpus maps these 1:1 (verified: no alias resolves to two documents), so
    accepting both cannot reintroduce the collapse.
    """
    def norm_section(s: str) -> str:
        s = s.strip().lower().strip("_*# ").strip()
        return s[len("section "):] if s.startswith("section ") else s

    known = {}
    for c in chunks:
        m = c["meta"]
        insurer, section = m["insurer"].lower(), norm_section(str(m["section_id"]))
        for token in (m.get("doc_short"), m.get("filename")):
            doc = _norm_doc(token) if token else ""
            if doc:
                known.setdefault((insurer, doc, section), set()).add(int(m["page"]))
    results = []
    for match in CITATION_RE.finditer(text):
        insurer, doc, section, page = (g.strip() for g in match.groups())
        doc = _norm_doc(doc)
        # Fail CLOSED: an empty/unparseable document token is UNSUPPORTED. Never
        # fall back to an (insurer, section) lookup — that is the collapse above.
        key = (insurer.lower(), doc, norm_section(section))
        pages = known.get(key, set()) if doc else set()
        ok = any(abs(int(page) - p) <= 1 for p in pages) if pages else False
        results.append({"citation": match.group(0), "supported": ok})
    abstained = is_refusal(text)
    return {
        "citations": results,
        "n_total": len(results),
        "n_supported": sum(r["supported"] for r in results),
        "abstained": abstained,
        "all_supported": bool(results) and all(r["supported"] for r in results),
    }


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Is my phone covered if it's stolen abroad?"
    persona = sys.argv[2] if len(sys.argv) > 2 else "end_user"
    text, chunks = answer(q, persona)
    print(text)
    print("\n--- citation check:", verify_citations(text, chunks))
