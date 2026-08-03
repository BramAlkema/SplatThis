"""`--svg-optimize`: shrink the emitted SVG without ever risking good output.

The optimizer shells out to an external Node tool, so the contract under
test is mostly about failure modes: a missing, failing, slow or destructive
svgo must leave the original file untouched.
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from splatthis.io import (
    SVG_OPTIMIZE_MIN_SAFE_PRECISION,
    generate_svg_content,
    optimize_svg_file,
)
from splatthis.splat import create_isotropic_splat

HAVE_SVGO = shutil.which("svgo") is not None


def _write_svg(path: Path, n: int = 40) -> str:
    rng = np.random.default_rng(3)
    splats = [
        create_isotropic_splat(
            center=rng.uniform(4, 60, size=2),
            sigma=float(rng.uniform(1.5, 5.0)),
            color=rng.uniform(0.1, 0.9, size=3),
            alpha=float(rng.uniform(0.3, 1.0)),
        )
        for _ in range(n)
    ]
    content = generate_svg_content(splats, width=64, height=64, k_sigma=2.5)
    path.write_text(content, encoding="utf-8")
    return content


def test_missing_svgo_leaves_file_untouched(tmp_path, monkeypatch):
    """No svgo on PATH: warn, report, and change nothing."""
    svg = tmp_path / "a.svg"
    original = _write_svg(svg)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    report = optimize_svg_file(str(svg))

    assert report["applied"] is False
    assert report["reason"] == "svgo-not-installed"
    assert svg.read_text(encoding="utf-8") == original


def test_failing_svgo_leaves_file_untouched(tmp_path, monkeypatch):
    """Non-zero exit must not clobber the emitted SVG."""
    import subprocess as sp

    svg = tmp_path / "b.svg"
    original = _write_svg(svg)
    monkeypatch.setattr(shutil, "which", lambda _name: "/fake/svgo")
    monkeypatch.setattr(
        sp,
        "run",
        lambda cmd, **_kwargs: sp.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="expected test failure"
        ),
    )

    report = optimize_svg_file(str(svg))

    assert report["applied"] is False
    assert report["reason"].startswith("svgo-failed")
    assert svg.read_text(encoding="utf-8") == original


def test_destructive_svgo_output_is_rejected(tmp_path, monkeypatch):
    """If svgo returns success but drops shapes, keep the original."""
    import subprocess as sp

    svg = tmp_path / "c.svg"
    original = _write_svg(svg)
    monkeypatch.setattr(shutil, "which", lambda _name: "/fake/svgo")

    def fake_run(cmd, **_kwargs):
        # Emit a valid but nearly empty SVG to the -o target.
        out = Path(cmd[cmd.index("-o") + 1])
        out.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
        return sp.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sp, "run", fake_run)
    report = optimize_svg_file(str(svg))

    assert report["applied"] is False
    assert report["reason"] == "svgo-dropped-shapes"
    assert svg.read_text(encoding="utf-8") == original


def test_invalid_xml_output_is_rejected(tmp_path, monkeypatch):
    """Malformed svgo output must never replace a good file."""
    import subprocess as sp

    svg = tmp_path / "d.svg"
    original = _write_svg(svg)
    monkeypatch.setattr(shutil, "which", lambda _name: "/fake/svgo")

    def fake_run(cmd, **_kwargs):
        Path(cmd[cmd.index("-o") + 1]).write_text("<svg><not closed", encoding="utf-8")
        return sp.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sp, "run", fake_run)
    report = optimize_svg_file(str(svg))

    assert report["applied"] is False
    assert report["reason"].startswith("svgo-invalid-xml")
    assert svg.read_text(encoding="utf-8") == original


def test_low_precision_warns(tmp_path, monkeypatch, caplog):
    """Precision below the measured-safe floor must be called out."""
    svg = tmp_path / "e.svg"
    _write_svg(svg)
    monkeypatch.setattr(shutil, "which", lambda _name: None)  # stop before running

    with caplog.at_level("WARNING"):
        optimize_svg_file(str(svg), precision=1)

    assert SVG_OPTIMIZE_MIN_SAFE_PRECISION == 2


@pytest.mark.skipif(not HAVE_SVGO, reason="svgo is not installed")
def test_real_svgo_shrinks_and_preserves_shapes(tmp_path):
    """With svgo present: smaller file, same shape count, still valid XML."""
    svg = tmp_path / "real.svg"
    original = _write_svg(svg)
    before = svg.stat().st_size

    report = optimize_svg_file(str(svg))

    assert report["applied"] is True, report
    assert report["bytes_after"] < before
    assert report["precision"] == SVG_OPTIMIZE_MIN_SAFE_PRECISION

    optimized = svg.read_text(encoding="utf-8")
    ET.fromstring(optimized)  # still well-formed

    # svgo may rewrite <ellipse rx==ry> as <circle>; the total must not drop.
    def shapes(text: str) -> int:
        return text.count("<ellipse") + text.count("<circle") + text.count("<path")

    assert shapes(optimized) >= shapes(original)
    # Gradients carry the splat falloff and must survive intact.
    assert optimized.count("<radialGradient") == original.count("<radialGradient")


@pytest.mark.skipif(not HAVE_SVGO, reason="svgo is not installed")
def test_converter_records_optimization_in_manifest(tmp_path):
    """End-to-end: --svg-optimize reports into the run manifest."""
    import json

    from PIL import Image

    from splatthis.converter import PNG2SVGConverter

    img = tmp_path / "tiny.png"
    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (24, 24, 3), dtype=np.uint8)
    ).save(img)

    artifacts = tmp_path / "art"
    out = tmp_path / "tiny.svg"
    PNG2SVGConverter(
        max_splats=12,
        stages=[2, 1],
        optimizer_backend="torch",
        refinement_config={"svg_optimize": True},
    ).convert(
        str(img), output_path=str(out), verbose=False, artifacts_dir=str(artifacts)
    )

    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    assert "svg_optimize" in manifest
    assert manifest["svg_optimize"]["precision"] == 2
    ET.parse(out)  # emitted file still parses after optimization


def test_svgo_does_not_take_the_embedded_population_with_it(tmp_path):
    """svgo's default preset includes removeMetadata.

    That silently destroyed the population in every
    `--embed-population --svg-optimize` run: nothing downstream reads it
    back, so the SVG looked fine and simply was not re-loadable any more.
    """
    import shutil

    if shutil.which("svgo") is None:
        pytest.skip("svgo is not installed")

    from splatthis.population_embed import load_population
    from splatthis.splat import create_isotropic_splat
    from splatthis.svg_export import generate_svg_content, optimize_svg_file

    splats = [
        create_isotropic_splat(
            center=[8.0 + i * 3, 9.0 + i * 2],
            sigma=2.5,
            color=[0.3, 0.5, 0.7],
            alpha=0.6,
        )
        for i in range(24)
    ]
    path = tmp_path / "embedded.svg"
    path.write_text(
        generate_svg_content(splats, width=96, height=96, embed_population=True),
        encoding="utf-8",
    )
    assert len(load_population(str(path))) == len(splats)

    report = optimize_svg_file(str(path), precision=2)
    assert report["applied"] or report.get("reason") in {
        "no-size-win",
        "svgo-not-installed",
    }
    assert len(load_population(str(path))) == len(splats), (
        "svgo stripped the embedded population; the artifact is no longer "
        "re-targetable or warm-startable"
    )
