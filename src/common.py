"""Shared config for the Policy-Wording Q&A pipeline.

All components talk to LM Studio's OpenAI-compatible server on localhost:1234.
Embeddings use nomic-embed-text-v1.5, which REQUIRES task prefixes:
"search_document: " for indexed text, "search_query: " for queries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(_h).resolve() if (_h := os.environ.get("POLICY_QA_HOME")) and (Path(_h) / "src" / "common.py").exists() else Path(__file__).resolve().parents[1]
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
DATA_DIR = PROJECT_ROOT / "data"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
CHROMA_DIR = PROJECT_ROOT / "chroma"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
# The demo generator (LM Studio identifier). Two measured reasons this is
# qwen2.5-7b-instruct and NOT the qwen/qwen3-8b it defaulted to until 2026-08-07:
#
#  (a) Consistency. Every published row on eval/leaderboard.md was generated on
#      qwen2.5-7b-instruct. While the app defaulted to qwen/qwen3-8b, the live
#      demo and the published numbers described DIFFERENT models — so the first
#      question a supervisor asks, "is this the model you measured?", had the
#      answer "no". Matching the default to the board makes it "yes", and that
#      is the whole value of the demo.
#
#  (b) Latency. Measured end-to-end on this M2 through LM Studio: qwen/qwen3-8b
#      is ~63s per answer, and a live 22-question sweep averaged 77s with a
#      worst case of 246s — long enough on stage that a question reads as hung.
#      qwen2.5-7b-instruct is roughly 2.5x faster on the same box.
#      CAUTION: the 21.6s on eval/leaderboard.md is the KAGGLE T4 harness, not
#      LM Studio on this laptop. It is not live demo latency and must never be
#      quoted as such.
#
# POLICY_QA_GEN_MODEL overrides at runtime, so a side-by-side against qwen3-8b
# is an env var rather than an edit to a file that has to stay byte-identical
# across the code repo and the data repo. Harness and conversation-eval rows
# pass `model=` explicitly (eval/harness.py ROWS, eval/conversation_eval.py),
# so they are pinned by their own row config and unaffected by either the
# default or the override.
GEN_MODEL = os.environ.get("POLICY_QA_GEN_MODEL", "qwen2.5-7b-instruct")
COLLECTION = "policy_chunks"

# The abstention sentence is fixed verbatim so the app badge, the judge and
# the harness can all detect it deterministically.
ABSTAIN_SENTENCE = (
    "This isn't addressed in the policy wording I have — please refer to a policy handler."
)

# Buyer-mode (pre-purchase guidance) fixed sentences — same verbatim-detection
# contract as ABSTAIN_SENTENCE, but in the register of someone who has no
# policy yet: signpost to a broker/insurer, never to "a handler".
BUYER_ABSTAIN_SENTENCE = (
    "The policy documents I have don't answer that — please check with an insurer or broker."
)
SIGNPOST_SENTENCE = (
    "I can't recommend a policy or tell you what to buy — for a decision like that, "
    "please speak to an insurer or an independent broker."
)
BUYER_SCOPE_SENTENCE = (
    "I can only describe the three insurers' documents in my index — this is not a "
    "market comparison."
)
BUYER_DISCLAIMER = (
    "Research prototype — information from public policy documents only, not advice or "
    "a personal recommendation. Nothing you type is stored. A decision to buy any policy "
    "should be made with an insurer or independent broker."
)

# Recommendation-lexicon guard for buyer mode: any hit on a substantive answer
# triggers one constrained retry, then the hard SIGNPOST_SENTENCE fallback.
# Deliberately narrow, high-precision phrases (advice speech-acts), so the
# guard-off violation count measures the model, not incidental wording.
RECOMMENDATION_RE = re.compile(
    r"\b(?:"
    r"you should (?:go with|choose|pick|buy|take out|opt for)"
    r"|i(?:'d| would) (?:recommend|suggest|go with|choose|pick)"
    r"|(?:i|we) recommend"
    r"|(?:the )?best (?:option|choice|policy|insurer|cover) for you"
    r"|your best bet"
    r"|you(?:'d| would) be better off (?:with|choosing)"
    r"|most people (?:your age|like you) (?:pick|choose|go with)"
    r"|(?:is|would be) (?:the right|the best|ideal|perfect) (?:choice|policy|option) for you"
    r")\b",
    re.IGNORECASE,
)

# Citation format: [Insurer, DocShort, Section <id>, p.<page>]
# Parsing tolerates stray quotation marks inside the brackets and a trailing
# detail segment ("..., p.23, point 5").
CITATION_RE = re.compile(
    r"""\[\s*["']?\s*([^,\[\]"']+?)\s*,\s*["']?([^,\[\]]+?)\s*,\s*([^,\[\]]+?)\s*,\s*p\.?\s*(\d+)[^\]]*\]"""
)


def norm(s: str) -> str:
    """Canonical text normaliser for quote/section matching.

    LIVES HERE, not in eval/validate_gold.py, because four modules across three
    directories need it — train/raft_build_data.py, eval/review_gold.py,
    eval/council_verify.py and eval/coverage_report.py — and until 2026-07-30 they all
    got it via `from validate_gold import norm`, which resolved off
    `$POLICY_QA_HOME/eval` on sys.path. That path held a STALE 2026-07-03 shadow of
    validate_gold.py, so every one of them was silently binding the data repo's copy
    (T-071). Deleting the shadow then broke the import outright, because none of those
    four directories is the repo's eval/.

    src/common.py is the right home precisely because it is the one file the rule keeps
    BYTE-IDENTICAL between the code repo and the data repo, so it resolves the same way
    from either — which is exactly the property the old arrangement lacked.

    Behaviour is unchanged from eval/validate_gold.py's implementation, and
    tests/test_norm_single_source.py pins the two against each other so they cannot
    drift while both exist.
    """
    s = s.lower()
    s = re.sub(r"<br\s*/?>|&lt;br\s*/?&gt;", " ", s)   # pymupdf4llm line-break artifacts
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[\s*_#|•·]+", " ", s)   # fold whitespace AND markdown noise
    return s.strip()

# Hand-curated manifest facts per known file (browser-downloaded by the user;
# URL + licence note recorded for provenance — see documents.jsonl).
# Direct Line and FOS are deliberately absent pending supervisor sign-off.
KNOWN_DOCS = [
    {
        "pattern": r"NTRTG10145",
        "insurer": "Aviva", "doc_short": "Travel Wording", "line": "travel",
        "doc_title": "Aviva Single/Annual Multi-trip Travel Insurance policy wording (NTRTG10145)",
        "source_url": "https://www.aviva.co.uk/static/library/pdfs/travel/NTRTG10145.pdf",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"aa-travel-insurance-essential.*202602|aa[-_ ]?travel",
        "insurer": "AA", "doc_short": "Essential Travel Wording", "line": "travel",
        "doc_title": "AA Travel Insurance Essential policy wording (Feb 2026)",
        "source_url": "https://www.theaa.com/media/the-aa/pdf/insurance/aa-travel-insurance-essential-policy-wording-202602.pdf",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"PostOffice|direct_policy_wording|post[-_ ]?office",
        "insurer": "Post Office", "doc_short": "Travel Wording", "line": "travel",
        "doc_title": "Post Office Travel Insurance policy wording (current edition)",
        "source_url": "https://www.postoffice.co.uk/travel-insurance (documents section)",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"TR-APB-\d+",
        "insurer": "Admiral", "doc_short": "Travel Policy Book", "line": "travel",
        "doc_title": "Admiral Travel Insurance policy book (multi-tier: Admiral/Gold/Platinum)",
        "source_url": "https://www.admiral.com/existing-customers/policy-documents.php",
        "licence_note": "Public insurer document; internal research use; do not redistribute. Multi-tier book — chunks carry tier metadata.",
    },
    {
        "pattern": r"IP-TR-1-\d+",
        "insurer": "Admiral", "doc_short": "Travel IPID (single trip)", "line": "travel",
        "doc_title": "Admiral Travel Insurance IPID — single trip (IP-TR-1)",
        "source_url": "https://www.admiral.com/existing-customers/policy-documents.php",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    {
        "pattern": r"IP-TR-2-\d+",
        "insurer": "Admiral", "doc_short": "Travel IPID (annual)", "line": "travel",
        "doc_title": "Admiral Travel Insurance IPID — annual multi-trip (IP-TR-2)",
        "source_url": "https://www.admiral.com/existing-customers/policy-documents.php",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    # --- motor (added 2026-07-07; files arrive via manual browser download) ---
    {
        "pattern": r"Insurance_policy_default",
        "insurer": "Aviva", "doc_short": "Motor Wording", "line": "motor",
        "doc_title": "Aviva motor insurance policy booklet (Online & Premium)",
        "source_url": "https://www.online.aviva.co.uk/StaticDocsAV/Insurance_policy_default_v15.pdf",
        "licence_note": "Public insurer document; internal research use; do not redistribute. PAIRING RULE: IPID must come from the same product page.",
    },
    {
        "pattern": r"0042731",
        "insurer": "LV=", "doc_short": "Motor T&Cs", "line": "motor",
        "doc_title": "LV= Motor Insurance Terms & Conditions (0042731-2025)",
        "source_url": "https://www.lv.com/car-insurance (policy documents section)",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"40383",
        "insurer": "LV=", "doc_short": "Cover & Limits", "line": "motor",
        "doc_title": "LV= Car Insurance Cover & Limits (40383-2025 v4-1)",
        "source_url": "https://www.lv.com/car-insurance (policy documents section)",
        "licence_note": "Public insurer document; internal research use; do not redistribute. TABLE-HEAVY: spot-check every limits/excess table after ingest.",
    },
    {
        "pattern": r"0042748",
        "insurer": "LV=", "doc_short": "Car IPID", "line": "motor",
        "doc_title": "LV= Car Insurance IPID (0042748-2025)",
        "source_url": "https://www.lv.com/car-insurance (policy documents section)",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    {
        "pattern": r"0042719",
        "insurer": "LV=", "doc_short": "Motor Legal Expenses", "line": "motor",
        "doc_title": "LV= Motor Legal Expenses cover document (0042719-2025)",
        "source_url": "https://www.lv.com/car-insurance (policy documents section)",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"40596",
        "insurer": "LV=", "doc_short": "Breakdown Cover", "line": "motor",
        "doc_title": "LV= Britannia Rescue breakdown document (40596-2025)",
        "source_url": "https://www.lv.com/car-insurance (policy documents section)",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        # Admiral car docs are click-through downloads; the exact AD-xxx code is
        # recorded from the downloaded file, so match generically on 'car'.
        "pattern": r"(?i)AD-\d{3}-\d{3}.*car|car.*insurance.*guide|admiral.*car",
        "insurer": "Admiral", "doc_short": "Car Policy Booklet", "line": "motor",
        "doc_title": "Admiral Car Insurance policy booklet (multi-tier: Essential/Admiral/Gold/Platinum)",
        "source_url": "https://www.admiral.com/existing-customers/policy-documents.php",
        "licence_note": "Public insurer document; internal research use; do not redistribute. Multi-tier book — chunks carry tier metadata.",
    },
    {
        "pattern": r"Insurance_product_important_information",
        "insurer": "Aviva", "doc_short": "Motor IPID", "line": "motor",
        "doc_title": "Aviva Motor Insurance IPID",
        "source_url": "https://www.online.aviva.co.uk/sales/DocLink.aspx?name=CAR_INSURANCEPRODUCTINFORMATION",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    {
        "pattern": r"(?i)AD-004|renewal-brochure",
        "insurer": "Admiral", "doc_short": "Car Guide Amendments", "line": "motor",
        "doc_title": "Admiral 'Amendments to Your Car Insurance Guide' booklet",
        "source_url": "https://www.admiral.com/existing-customers/policy-documents.php",
        "licence_note": "Public insurer document; internal research use; do not redistribute. AMENDS the Car Insurance Guide — cite alongside it.",
    },
    {
        "pattern": r"(?i)AD046|claims-flyer",
        "insurer": "Admiral", "doc_short": "Claims Flyer", "line": "motor",
        "doc_title": "Admiral motor Claim Information Flyer",
        "source_url": "https://www.admiral.com/existing-customers/policy-documents.php",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"(?i)IP-CA|IP-MO|car.*ipid|ipid.*car",   # Admiral motor IPID assets are IP-MO-1-00x
        "insurer": "Admiral", "doc_short": "Car IPID", "line": "motor",
        "doc_title": "Admiral Car Insurance IPID",
        "source_url": "https://www.admiral.com/existing-customers/policy-documents.php",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    {
        "pattern": r"Home_wording_default",
        "insurer": "Aviva", "doc_short": "Home Wording", "line": "home",
        "doc_title": "Aviva Online & Premium Home Insurance policy booklet",
        "source_url": "https://www.online.aviva.co.uk/StaticDocsAV/Home_wording_default_v2.pdf",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"NTRTG14137",
        "insurer": "Aviva", "doc_short": "Travel IPID", "line": "travel",
        "doc_title": "Aviva Travel Insurance IPID (NTRTG14137)",
        "source_url": "https://static.aviva.io/content/dam/aviva-public/gb/pdfs/personal/insurance/travel/NTRTG14137.pdf",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    {
        "pattern": r"Home_insurance_product_information",
        "insurer": "Aviva", "doc_short": "Home IPID", "line": "home",
        "doc_title": "Aviva Home Insurance IPID",
        "source_url": "https://www.online.aviva.co.uk/StaticDocsAV/Home_insurance_product_information_default_v4.pdf",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    # --- T-046 additions (M1 AA car, M2 Aviva breakdown, T-040 Admiral home + PO travel).
    # Appended at the END on purpose: doc_facts_for is FIRST-MATCH-WINS, so every pattern
    # above keeps priority and no existing document can be re-labelled by these entries.
    # Patterns are pinned to asset codes, not generic words, for the same reason.
    # Guarded by tests/test_known_docs_covers_corpus.py.
    {
        "pattern": r"aa-car-insurance-policy-booklet",
        "insurer": "AA", "doc_short": "Car Wording", "line": "motor",
        "doc_title": "AA Car Insurance policy booklet (December 2023; Silver/Gold/Platinum tiers)",
        "source_url": "https://www.theaa.com/media/the-aa/pdf/insurance/tiers/aa-car-insurance-policy-booklet-december-2023.pdf",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"comprehensive-silver-\d{6}",
        "insurer": "AA", "doc_short": "Car IPID", "line": "motor",
        "doc_title": "AA Car Insurance IPID — Silver Comprehensive (May 2025)",
        "source_url": "https://www.theaa.com/media/the-aa/pdf/ipids/motor/tiers/comprehensive-silver-202505.pdf",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    {
        "pattern": r"NMDOR0064",
        "insurer": "Aviva", "doc_short": "Breakdown Wording", "line": "motor",
        "doc_title": "Aviva Rescue (motor breakdown) policy wording (NMDOR0064)",
        "source_url": "https://static.aviva.io/content/dam/aviva-public/gb/pdfs/personal/insurance/motor/breakdown/insurance-motor-breakdown-policy-wording-261017-NMDOR0064.pdf",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"NMDOR14136",
        "insurer": "Aviva", "doc_short": "Breakdown IPID", "line": "motor",
        "doc_title": "Aviva Motor Breakdown Insurance IPID (NMDOR14136)",
        "source_url": "https://static.aviva.io/content/dam/aviva-public/gb/pdfs/personal/insurance/motor/breakdown/insurance-motor-breakdown-insurance-product-information-document-NMDOR14136.pdf",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    {
        "pattern": r"HH-010-018",
        "insurer": "Admiral", "doc_short": "Home Policy Booklet", "line": "home",
        "doc_title": "Admiral Guide to your Home Insurance Cover (HH-010-018)",
        "source_url": "https://mktgblobpubaccess1.blob.core.windows.net/eui-pdf-assets/admiral/HH-010-018-Guide-to-your-Home-Insurance-Cover.pdf",
        "licence_note": "Public insurer document; internal research use; do not redistribute.",
    },
    {
        "pattern": r"(?i)IP-HO",   # Admiral home IPID assets are IP-HO-1-0xx (cf. IP-MO motor, IP-TR travel)
        "insurer": "Admiral", "doc_short": "Home IPID", "line": "home",
        "doc_title": "Admiral Home Insurance IPID (IP-HO-1-011)",
        "source_url": "https://mktgblobpubaccess1.blob.core.windows.net/eui-pdf-assets/admiral/IP-HO-1-011.pdf",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
    {
        "pattern": r"standard_ipid_\d{4}",
        "insurer": "Post Office", "doc_short": "Travel IPID", "line": "travel",
        "doc_title": "Post Office Travel Insurance IPID — Standard (March 2026)",
        "source_url": "https://www.postoffice.co.uk/dam6/jcr:96434c31-bc4b-4a7e-be02-2441061f3cb3/standard_ipid_2603.2026-03-09-11-20-50.pdf",
        "licence_note": "IPID: regulator-mandated summary intended for free distribution (ICOBS 6 Annex 3).",
    },
]

TIER_RE = re.compile(r"\b(Admiral Platinum|Admiral Gold|Platinum|Gold|Essential|Standard|Premier|Premium)\b")


def doc_facts_for(filename: str) -> dict:
    for d in KNOWN_DOCS:
        if re.search(d["pattern"], filename, re.IGNORECASE):
            return {k: v for k, v in d.items() if k != "pattern"}
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return {
        "insurer": stem.split()[0].title() if stem else "Unknown",
        "doc_short": "Document", "line": "unknown",
        "doc_title": filename,
        "source_url": "MANUAL — record the source URL in documents.jsonl",
        "licence_note": "UNVERIFIED — check terms before indexing beyond internal research use.",
    }


def gold_path() -> Path:
    """Current single-turn gold file: highest available version."""
    for name in ("gold_v2.jsonl", "gold_v1.jsonl", "gold_v0.jsonl"):
        p = GOLD_DIR / name
        if p.exists():
            return p
    return GOLD_DIR / "gold_v0.jsonl"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Defensively remove reasoning traces from ANY model's output."""
    text = THINK_RE.sub("", text)
    # Unclosed <think> (truncated generation): drop everything from the tag on.
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def chunk_header(meta: dict) -> str:
    """The context header prepended to every chunk — citations must copy it."""
    return (
        f"[{meta['insurer']}, {meta['doc_short']}, Section {meta['section_id']}, "
        f"p.{meta['page']}] {meta.get('section_title', '')}".strip()
    )
