"""CI gate: the app-side refusal detector must survive rendering variants.

Why this exists. `rag.verify_citations` used to decide "did the model abstain?" with an EXACT
substring test:

    abstained = ABSTAIN_SENTENCE.rstrip("。 .") in text

That is brittle in the worst possible direction. When it says False on a genuine refusal, the
app falls through to its zero-citation branch and paints a banner saying the answer cites
nothing — i.e. in front of the room the demo accuses the system of breaking its own rule at
the exact moment the model obeyed it. An em dash rendered as `-`, `--` or `–`, a curly
apostrophe, a case change, `**bold**`, or a line wrap was enough.

It was also only ever taught ONE sentence. `BUYER_ABSTAIN_SENTENCE` and `SIGNPOST_SENTENCE`
were never recognised at all, so buyer mode — one of the app's three personas — showed no
badge for its own headline behaviour, which is the behaviour a compliance-minded supervisor
is most likely to ask about.

HONEST SCOPE OF THE FIX — measured 2026-08-07 over the four frozen answer files in
`policy_qa/eval/results/`, 140 rows each, retired exact rule -> new `is_refusal`:

    qwen2.5-7b-bnb-openbook    BASE, the row the live demo drives     52 -> 52   (+0)
    qwen2.5-7b-raft-openbook   TUNED, the row on the leaderboard      31 -> 31   (+0)
    qwen2.5-7b-bnb-closedbook                                         82 -> 84   (+2)
    qwen2.5-7b-raft-closedbook                                         0 ->  0   (+0)

So on the committed OPEN-BOOK data — the published rows, and the ones anyone will ask about —
this fix recovers ZERO additional abstentions. It is not a metric improvement and must never
be presented as one. Its two real values are:
  (a) the two buyer-mode sentences, which were genuinely unhandled, so buyer abstention went
      from "no badge, ever" to "badge";
  (b) robustness against LIVE generation, which is not frozen and is exactly what the demo
      runs. The +2 on the closed-book base row (PQ-0159, PQ-0037) is the only direct evidence
      we have that a real local decode does produce a variant rendering — two rows out of 560.
No row LOSES a detection under the new rule; it is a strict superset of the old one on all
four files, and `test_new_rule_is_a_superset_of_the_retired_one` keeps it that way.

WHAT IT SPECIFICALLY DOES NOT FIX. Six base-row answers render the "cites nothing" banner:
PQ-0058, PQ-0084, PQ-0097, PQ-0099, PQ-0134, PQ-0135. Re-checked 2026-08-07: they are 401 to
2,443 characters long, contain zero `[` characters and zero parseable citations, and are not
refusals under either rule. They are genuine uncited answers, the banner is CORRECT on all
six, and nothing in this change touches them. Do not let a demo script or a slide imply
otherwise.

NOT THE PUBLISHED METRIC. This is the APP-SIDE detector. The published abstention axis comes
from `eval/abstention.py` (adopted reading B) and `eval/judge.py:80` still carries a third,
retired rule. Three detectors coexist on purpose; a badge in the app is not evidence for a
leaderboard number.

Loads `src/rag.py` by explicit path from the CODE repo — no submodule, no data, no network,
no LM Studio — so it runs for real on a fresh CI clone rather than skipping.

    pytest tests/test_refusal_detection.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CODE_SRC = REPO / "src"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_rag():
    """Import the code repo's `src/rag.py` by explicit path.

    Deliberately NO `sys.path.insert` here. `src/rag.py` line 21 already does its own
    own-directory insert before `from common import ...` — the correct in-package pattern that
    `tests/test_no_bare_src_path_insert.py:29-33` explains is exempt from that lint — so the
    import resolves without this file touching `sys.path` at all. Adding one anyway would
    leak `src/` onto the path for every test module that runs after this one, which is the
    shape of leak that once left the suite green while a module was broken standalone.
    """
    return _load("_rag_under_test", CODE_SRC / "rag.py")


rag = _load_rag()

# The sentences read from src/common.py directly, under a private module name. Compared below
# against what rag.py actually bound: if a stale `common` were sitting in sys.modules from an
# earlier test file, rag.py's bare `from common import ...` would silently bind THAT one and
# this whole file would be testing the wrong constants — T-071's failure mode, in miniature.
_common = _load("_common_for_refusal", CODE_SRC / "common.py")

SENTENCES = {
    "ABSTAIN_SENTENCE": _common.ABSTAIN_SENTENCE,
    "BUYER_ABSTAIN_SENTENCE": _common.BUYER_ABSTAIN_SENTENCE,
    "SIGNPOST_SENTENCE": _common.SIGNPOST_SENTENCE,
}


def _retired_rule(text: str) -> bool:
    """The exact-substring test `is_refusal` replaced, kept so its behaviour stays pinned."""
    return _common.ABSTAIN_SENTENCE.rstrip("。 .") in text


def _variants(s: str) -> dict[str, str]:
    """Renderings a local generator plausibly produces for the same fixed sentence.

    Every one of these is a rewrite the OLD exact test would have scored as "not a refusal".
    The composite at the end matters most: real decodes stack these, they do not arrive one at
    a time.
    """
    return {
        "verbatim": s,
        "hyphen for em dash": s.replace("—", "-"),
        "double hyphen for em dash": s.replace("—", "--"),
        "triple hyphen for em dash": s.replace("—", "---"),
        "en dash for em dash": s.replace("—", "–"),
        "spaced hyphen for em dash": s.replace("—", " - "),
        "curly apostrophe": s.replace("'", "’"),
        "uppercased": s.upper(),
        "title cased": s.title(),
        "whitespace expanded": "  ".join(s.split()),
        "newline wrapped": s.replace(" ", "\n", 3),
        "markdown bold": f"**{s}**",
        "leading/trailing space": f"   {s}   ",
        "appended after prose": "Here is what I found in the extracts.\n\n" + s,
        "wrapped in prose": f"I checked all six extracts. {s} Sorry I cannot be more use.",
        "composite (upper + double hyphen + curly + bold)": (
            f"**{s.replace('—', '--').replace(chr(39), '’').upper()}**"
        ),
    }


# Parametrised by NAME, not by the sentence itself: pytest builds test IDs out of the
# parameter values, and a whole sentence in every ID makes a failure line unreadable.
@pytest.mark.parametrize("name", sorted(SENTENCES))
def test_every_fixed_refusal_sentence_is_recognised(name: str) -> None:
    """All three fixed sentences, under every rendering variant."""
    for label, text in _variants(SENTENCES[name]).items():
        assert rag.is_refusal(text), (
            f"{name} was NOT recognised when {label}. The app falls through to its "
            "zero-citation branch and tells the room the system violated its own citation "
            "rule, on an answer that abstained correctly.\n"
            f"    text: {text[:100]!r}"
        )


def test_buyer_sentences_are_the_regression_this_closes() -> None:
    """Pin the two sentences the retired rule never knew about at all.

    Asserting the retired rule does NOT match them is not decoration: it records that the gap
    was real, so if someone ever "simplifies" `is_refusal` back towards the old shape this
    test names what would be lost rather than just going red.
    """
    for name in ("BUYER_ABSTAIN_SENTENCE", "SIGNPOST_SENTENCE"):
        s = SENTENCES[name]
        assert not _retired_rule(s), f"premise changed: the retired rule now matches {name}"
        assert rag.is_refusal(s), (
            f"{name} must be recognised — without it buyer mode renders no badge for its own "
            "headline behaviour, which is the one a compliance question lands on."
        )


@pytest.mark.parametrize(
    "label,text",
    [
        ("a normal cited answer", "You are covered under [AA, Car Wording, Section 4, p.12]."),
        ("empty string", ""),
        ("whitespace only", "   \n  "),
        ("unrelated prose", "The excess is stated in your schedule."),
        (
            "shares the abstain sentence's tail",
            "Please refer to a policy handler if you would like this double-checked.",
        ),
        (
            "near-inverse of the abstain sentence's head",
            "This is addressed in the policy wording I have.",
        ),
        (
            "one word off the abstain sentence",
            "This isn't addressed in the policy wording I have — please refer to a claims handler.",
        ),
        (
            "shares the buyer sentence's vocabulary",
            "The policy documents I have do answer that — see the two sections quoted above.",
        ),
    ],
)
def test_non_refusals_are_not_flagged(label: str, text: str) -> None:
    """The detector must not become so permissive that it swallows real answers.

    The last four cases are the ones that matter: they share vocabulary with the fixed
    sentences without BEING them. A detector loose enough to fire on those silently converts
    answered questions into abstentions, which inflates the app's refusal badge in the
    direction that flatters the project — the worst direction for a measurement to be wrong in.
    """
    assert not rag.is_refusal(text), f"false positive on {label}: {text[:90]!r}"


def test_new_rule_is_a_superset_of_the_retired_one() -> None:
    """Whatever the old exact test caught, the new one must still catch.

    Measured on the four frozen answer files (see the module docstring): no row loses a
    detection, on any of them. This pins that direction, because a fold-based rule CAN lose
    matches — `norm()` collapses `*_#|•·` to spaces, so a normaliser change is capable of
    turning a hit into a miss without anyone noticing.
    """
    corpus = [t for s in SENTENCES.values() for t in _variants(s).values()]
    corpus += [
        "Some prose. " + _common.ABSTAIN_SENTENCE,
        _common.ABSTAIN_SENTENCE + " [AA, Car Wording, Section 4, p.12]",
        "",
        "The excess is stated in your schedule.",
    ]
    lost = [t for t in corpus if _retired_rule(t) and not rag.is_refusal(t)]
    assert not lost, (
        f"{len(lost)} text(s) the retired exact rule flagged are no longer flagged, e.g. "
        f"{lost[0][:100]!r}. The replacement was meant to be strictly more permissive; "
        "losing a match is a regression the old code did not have."
    )


def test_verify_citations_actually_uses_the_detector() -> None:
    """The integration point, not just the helper.

    `is_refusal` being correct is worthless if `verify_citations` still computes `abstained`
    some other way — and `verify_citations`'s result is what the app renders the badge from.
    Driven with `chunks=[]`, so this stays data-free.
    """
    for name, sentence in sorted(SENTENCES.items()):
        variant = sentence.replace("—", "--").upper()
        res = rag.verify_citations(variant, [])
        assert res["abstained"] is True, (
            f"verify_citations did not flag a {name} variant as abstained — the badge is "
            "wired to this dict, not to is_refusal directly, so the fix has not landed where "
            "the app reads it."
        )
        assert res["n_total"] == 0 and res["all_supported"] is False

    cited = "You are covered under [AA, Car Wording, Section 4, p.12]."
    res = rag.verify_citations(cited, [])
    assert res["abstained"] is False, "verify_citations flagged a cited answer as an abstention"
    assert res["n_total"] == 1, "the citation regex no longer parses a well-formed citation"


def test_rag_bound_the_code_repo_common() -> None:
    """Guard against this whole file testing the wrong constants.

    `src/rag.py` imports `common` by BARE NAME off its own-directory sys.path insert. If any
    module named `common` is already in `sys.modules` when this file is collected — a stale
    submodule copy, another test's fixture — that one wins, silently, and every assertion here
    would be measuring sentences that are not the ones the app ships. That is T-071 exactly:
    eight imports bound a stale shadow and the only reason the harm was zero was luck.
    """
    for name, sentence in sorted(SENTENCES.items()):
        assert getattr(rag, name) == sentence, (
            f"rag.py's {name} is not src/common.py's {name}. rag.py bound a DIFFERENT "
            "`common` module — check what put one in sys.modules ahead of it."
        )


def test_detector_is_not_the_published_metric() -> None:
    """Guard the boundary, so nobody wires the app badge to the leaderboard by accident."""
    assert hasattr(rag, "is_refusal"), "is_refusal is the app-side detector's public name"
    doc = (rag.is_refusal.__doc__ or "").lower()
    assert "eval/abstention.py" in doc, (
        "is_refusal's docstring must keep naming eval/abstention.py as the real metric. Three "
        "abstention detectors coexist in this repo and conflating them has already cost a "
        "published number once."
    )
