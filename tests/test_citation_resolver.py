"""CI gate: a citation must name the RIGHT DOCUMENT, not just the right insurer.

Why this exists (a real bug, found by review council 2026-07-15): verify_citations()
built its lookup keyed on (insurer, section) only, and threw away the document group
that CITATION_RE already captures. Admiral's two travel IPIDs both carry
insurer="Admiral", section="What is insured?", page=1 — so they collapsed to ONE key
(60 such collapses in the corpus). On gold record PQ-0051, which asks which Admiral
travel tier pays what, a model could cite the single-trip IPID's limits against the
annual IPID's and still earn full citation_support — 0.3 of the eval composite,
silently inflated. Nothing caught it: every citation "resolved".

The document token is keyed on BOTH identifiers the pipeline exposes:
  - doc_short — chunk_header() prints it, so a prompted model copies it; since
    T-062, RAFT targets stamp it too (filename tags collapsed citation-support
    on leaderboard v2);
  - filename  — kept for adapters trained on pre-T-062 RAFT data, which
    legitimately emit filenames.

Every fixture here is 100% SYNTHETIC — fake insurer "TestSure", following the
precedent in data/gold/sample_synthetic.jsonl. No byte of licensed insurer text or
of any real corpus/gold record appears in this file, so it runs anywhere, including
CI on a fresh clone with no PDFs.

    pytest tests/test_citation_resolver.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rag import verify_citations  # noqa: E402


def chunk(doc_short: str, section: str, page: int, *,
          insurer: str = "TestSure", filename: str | None = None) -> dict:
    """A synthetic retrieved chunk — same meta shape as the real pipeline emits."""
    return {
        "text": "Synthetic clause text — not from any real policy wording.",
        "meta": {
            "insurer": insurer,
            "doc_short": doc_short,
            "filename": filename or f"{doc_short.replace(' ', '-')}-SAMPLE.pdf",
            "section_id": section,
            "section_title": section,
            "page": page,
            "line": "travel",
            "doc_title": f"{insurer} {doc_short} (synthetic)",
            "part": "",
        },
    }


def supported(text: str, chunks: list[dict]) -> list[bool]:
    return [c["supported"] for c in verify_citations(text, chunks)["citations"]]


# --- the bug itself: PQ-0051's key shape, synthesised -------------------------
# Two sibling IPIDs, identical insurer + section + page, different documents.
SINGLE_TRIP = chunk("Trip IPID (single trip)", "What is insured?", 1,
                    filename="IP-SYN-1-001.pdf")
ANNUAL = chunk("Trip IPID (annual)", "What is insured?", 1,
               filename="IP-SYN-2-001.pdf")


def test_citation_naming_the_wrong_sibling_document_is_not_supported():
    """The old (insurer, section) key collapsed these two into one entry."""
    text = "Belongings are covered up to the stated limit [TestSure, Trip IPID (single trip), Section What is insured?, p.1]."
    assert supported(text, [ANNUAL]) == [False], (
        "A citation naming the single-trip IPID was marked supported by a retrieved "
        "set containing ONLY the annual IPID. Insurer, section and page are identical "
        "across the two documents — this is exactly the PQ-0051 tier collapse."
    )


def test_citation_naming_the_right_document_is_still_supported():
    """The happy path must not regress — the fix must not fail everything closed."""
    text = "Belongings are covered up to the stated limit [TestSure, Trip IPID (single trip), Section What is insured?, p.1]."
    assert supported(text, [SINGLE_TRIP]) == [True]


def test_both_siblings_retrieved_resolves_each_to_its_own_document():
    """With both in the pool, each citation still binds to the document it names."""
    text = ("Single-trip limit [TestSure, Trip IPID (single trip), Section What is insured?, p.1]. "
            "Annual limit [TestSure, Trip IPID (annual), Section What is insured?, p.1]. "
            "Ghost limit [TestSure, Trip IPID (platinum), Section What is insured?, p.1].")
    assert supported(text, [SINGLE_TRIP, ANNUAL]) == [True, True, False]


def test_same_section_name_in_a_different_product_line_is_not_supported():
    """'What is insured?' appears in every IPID — a motor citation must not resolve
    against a travel IPID chunk just because the section heading matches."""
    text = "Windscreen cover applies [TestSure, Car IPID, Section What is insured?, p.2]."
    assert supported(text, [chunk("Trip IPID (single trip)", "What is insured?", 1)]) == [False]


# --- both identifiers, since the two generation paths emit different ones -----
def test_filename_token_resolves_for_the_raft_trained_path():
    """RAFT targets carry doc_file (the filename), so a tuned model emits it."""
    text = "The wording states the limit [TestSure, IP-SYN-1-001.pdf, Section What is insured?, p.1]."
    assert supported(text, [SINGLE_TRIP]) == [True]
    wrong = "The wording states the limit [TestSure, IP-SYN-2-001.pdf, Section What is insured?, p.1]."
    assert supported(wrong, [SINGLE_TRIP]) == [False], "filename path must discriminate siblings too"


def test_doc_token_normalisation_tolerates_case_and_a_missing_extension():
    """Models drop the .pdf and vary case; that must not read as a different doc."""
    text = "Limit applies [TestSure, ip-syn-1-001, Section What is insured?, p.1]."
    assert supported(text, [SINGLE_TRIP]) == [True]


# --- fail closed --------------------------------------------------------------
def test_page_tolerance_still_applies_within_the_named_document():
    """Page ±1 (a section spanning a boundary) survives, but only for the right doc."""
    text = "Restrictions apply [TestSure, Trip IPID (annual), Section Restrictions, p.3]."
    assert supported(text, [chunk("Trip IPID (annual)", "Restrictions", 2)]) == [True]
    assert supported(text, [chunk("Trip IPID (single trip)", "Restrictions", 2)]) == [False]


def test_unknown_document_is_not_supported():
    text = "Cover applies [TestSure, Some Wording We Never Retrieved, Section What is insured?, p.1]."
    assert supported(text, [SINGLE_TRIP]) == [False]


def test_uncopied_prompt_template_placeholder_is_not_supported():
    """A model echoing the literal [Insurer, Doc, Section, p.N] template earns nothing."""
    text = "Cover applies [TestSure, Doc, Section What is insured?, p.1]."
    assert supported(text, [SINGLE_TRIP]) == [False]


def test_all_supported_is_false_when_any_citation_names_the_wrong_document():
    """The aggregate the badge and the judge read must reflect the miss."""
    text = ("Single-trip limit [TestSure, Trip IPID (single trip), Section What is insured?, p.1]. "
            "Annual limit [TestSure, Trip IPID (annual), Section What is insured?, p.1].")
    check = verify_citations(text, [SINGLE_TRIP])
    assert check["n_total"] == 2
    assert check["n_supported"] == 1
    assert check["all_supported"] is False
