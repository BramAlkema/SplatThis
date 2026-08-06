"""The committed corpus gallery must match its assets and provenance ledgers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "tools" / "build_corpus_gallery.py"


def test_corpus_gallery_matches_assets_and_ledgers() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\n"
        "Regenerate with: python tools/build_corpus_gallery.py"
    )
