"""The README generator must refuse to publish an inconsistent registry.

``test_readme_results.py`` proves the happy path: the committed block matches
the committed ledgers. These tests prove the other half -- the fail-closed
behavior that is the tool's entire point. Each corrupts one input the way it
has actually gone wrong in this repository (a median that stops matching its
per-image evidence, a ledger that loses corpus coverage, a ledger that is
simply absent) and asserts the tool raises instead of rendering a
plausible-looking table.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "update_readme", REPO / "tools" / "update_readme.py"
)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def _corrupted_registry(tmp_path: Path, mutate) -> Path:
    data = json.loads(tool.FIDELITY.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "compositor-fidelity.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_the_committed_ledgers_pass_every_cross_check() -> None:
    block = tool.build_block()
    assert block.startswith(tool.BEGIN)
    assert block.endswith(tool.END)


def test_a_median_that_contradicts_its_per_image_evidence_is_refused(
    tmp_path, monkeypatch
) -> None:
    def bump(data) -> None:
        data["formats"]["svg"]["expectation"]["deployed"]["lpips_median"] += 0.01

    monkeypatch.setattr(tool, "FIDELITY", _corrupted_registry(tmp_path, bump))
    with pytest.raises(tool.LedgerError, match="deployed LPIPS median"):
        tool.declarative_expectations()


def test_a_p90_that_contradicts_its_per_image_evidence_is_refused(
    tmp_path, monkeypatch
) -> None:
    def bump(data) -> None:
        data["formats"]["css"]["expectation"]["deployed"]["lpips_p90"] -= 0.02

    monkeypatch.setattr(tool, "FIDELITY", _corrupted_registry(tmp_path, bump))
    with pytest.raises(tool.LedgerError, match="deployed LPIPS p90"):
        tool.declarative_expectations()


def test_lost_corpus_coverage_is_refused(tmp_path, monkeypatch) -> None:
    def drop_one(data) -> None:
        del data["per_image"]["svg"][0]

    monkeypatch.setattr(tool, "FIDELITY", _corrupted_registry(tmp_path, drop_one))
    with pytest.raises(tool.LedgerError, match="expected 21 corpus images"):
        tool.declarative_expectations()


def test_a_missing_ledger_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tool, "SVG_SIZES", tool.REPO / "does-not-exist.json")
    with pytest.raises(tool.LedgerError, match="missing ledger"):
        tool.svg_size_medians_kb()
