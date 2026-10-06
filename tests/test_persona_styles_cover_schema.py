"""Every persona the SCHEMA defines must have a PERSONA_STYLE entry.

Why (2026-07-21): `eval/harness.py export-kaggle` crashed with KeyError:
'buyer' — SCHEMA.md has defined the `buyer` persona since v2, and 32 of the
140 frozen test records use it, but src/rag.py's PERSONA_STYLE only ever grew
`end_user` and `handler`. The style map silently lagged the schema until a
buyer record flowed through build_messages. This test pins the two together:
it parses the persona enum row out of the tracked data/gold/SCHEMA.md, so a
future persona added to the schema without a style (or vice versa) fails CI
on a fresh clone with no data submodule present.

    pytest tests/test_persona_styles_cover_schema.py -q
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("rag", REPO / "src" / "rag.py")
rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag)


def schema_personas() -> set[str]:
    schema = (REPO / "data" / "gold" / "SCHEMA.md").read_text()
    row = next(l for l in schema.splitlines() if l.startswith("| `persona`"))
    return set(re.findall(r"`([a-z_]+)`", row)) - {"persona"}


def test_schema_declares_the_known_personas():
    assert schema_personas() == {"end_user", "handler", "buyer"}


def test_persona_style_covers_every_schema_persona():
    missing = schema_personas() - set(rag.PERSONA_STYLE)
    assert not missing, f"PERSONA_STYLE lacks schema personas: {sorted(missing)}"


def test_no_orphan_styles_outside_the_schema():
    orphans = set(rag.PERSONA_STYLE) - schema_personas()
    assert not orphans, f"PERSONA_STYLE has personas the schema does not define: {sorted(orphans)}"
