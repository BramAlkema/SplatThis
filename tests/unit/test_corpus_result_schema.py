"""The corpus result schema must not silently mix evidence levels.

`tools/corpus_benchmark.py` is content-addressed and resumable, which is what
makes long corpus runs practical -- and also what let rows written under an
older schema persist untouched beside newer ones. The stored `results.jsonl`
carries four renderers (rsvg-convert, proxy-srgb, canvas-linear and Chrome)
distinguished only by a free-text string, so an aggregate over rows mixes
deployed-artifact evidence with proxies without saying so.

That is the exact failure the project's evidence model exists to prevent, and
it cost a real measurement: a content-versus-fidelity correlation computed
across those rows came out at r=+0.456, against +0.863 on governing Chromium
data alone.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_benchmark():
    """Import tools/corpus_benchmark.py, which is a script rather than a package.

    It must be registered in ``sys.modules`` before execution: the module
    defines dataclasses, and ``@dataclass`` resolves annotations through
    ``sys.modules[cls.__module__].__dict__``, which fails with a bare
    ``AttributeError`` if the module is not yet there.
    """
    spec = importlib.util.spec_from_file_location(
        "corpus_benchmark", REPO / "tools" / "corpus_benchmark.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["corpus_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def test_every_scorer_declares_its_evidence_level():
    """A score without an evidence level cannot be filtered later.

    `is_deployed_artifact` is what separates a governing render from a
    diagnostic proxy. If a scorer omits it, its rows become indistinguishable
    downstream and the distinction survives only in whoever wrote the code.
    """
    benchmark = _load_benchmark()
    source = (REPO / "tools" / "corpus_benchmark.py").read_text(encoding="utf-8")

    assert benchmark.RESULT_SCHEMA_VERSION >= 2

    # Each scorer returns a dict literal; all of them must carry both keys.
    for scorer in ("score_svg", "score_pptx_proxy", "score_canvas_capture"):
        start = source.index(f"def {scorer}(")
        body = source[start : start + 3000]
        assert '"render_kind"' in body, f"{scorer} omits render_kind"
        assert '"is_deployed_artifact"' in body, f"{scorer} omits is_deployed_artifact"


def test_stale_schema_rows_are_not_treated_as_done(tmp_path):
    """Resume must re-score rows written before the current schema.

    Otherwise the run completes 'successfully' while leaving exactly the rows
    that lack evidence levels in place -- which is how the committed corpus
    ended up mixing schema versions.
    """
    benchmark = _load_benchmark()
    results = tmp_path / "results.jsonl"

    rows = [
        {"key": "old-row", "returncode": 0, "lpips": 0.4},
        {
            "key": "current-row",
            "returncode": 0,
            "lpips": 0.4,
            "schema_version": benchmark.RESULT_SCHEMA_VERSION,
        },
    ]
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))

    done = benchmark.load_done(results)

    assert "current-row" in done
    assert "old-row" not in done, "a pre-schema row must be re-scored, not skipped"
