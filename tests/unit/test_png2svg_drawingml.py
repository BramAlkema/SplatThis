"""Tests for PNG->DrawingML generation utilities."""

from pathlib import Path

from png2svg_gs.io import (
    EMU_PER_PX,
    generate_drawingml_slide_content,
    px_to_emu,
    save_drawingml,
)
from png2svg_gs.splat import create_isotropic_splat


def test_px_to_emu_conversion():
    """Pixel to EMU conversion should follow OOXML ratio."""
    assert px_to_emu(1.0) == EMU_PER_PX
    assert px_to_emu(10.0) == 10 * EMU_PER_PX
    assert px_to_emu(-5.0) == 0


def test_generate_drawingml_slide_content_basic():
    """Generated DrawingML should contain expected slide and shape tags.

    Default style is now 'gradient' (radial gradient with per-stop alpha);
    'soft-edge' is opt-in via splat_style="soft-edge". See
    test_generate_drawingml_slide_content_soft_edge_style for the soft-edge path.
    """
    splat = create_isotropic_splat(
        center=[20.0, 30.0], sigma=5.0, color=[1.0, 0.0, 0.0], alpha=0.5
    )
    content = generate_drawingml_slide_content(
        [splat], width=100, height=80, k_sigma=2.5
    )

    assert content.startswith('<?xml version="1.0"')
    assert "<p:sld " in content
    assert "<p:spTree>" in content
    assert 'name="Splat Group"' in content
    assert "<p:grpSp>" in content
    assert 'name="Splat 3"' in content
    assert '<a:prstGeom prst="ellipse">' in content
    assert 'val="FF0000"' in content
    assert "<a:gradFill>" in content
    assert "<a:gsLst>" in content
    assert "<a:alpha val=" in content
    assert "<a:softEdge" not in content


def test_generate_drawingml_slide_content_soft_edge_style():
    """Soft-edge DrawingML path remains available via explicit splat_style."""
    splat = create_isotropic_splat(
        center=[20.0, 30.0], sigma=5.0, color=[1.0, 0.0, 0.0], alpha=0.5
    )
    content = generate_drawingml_slide_content(
        [splat],
        width=100,
        height=80,
        k_sigma=2.5,
        splat_style="soft-edge",
    )

    assert "<a:solidFill>" in content
    assert "<a:softEdge" in content
    assert 'val="FF0000"' in content
    assert "<a:gradFill>" not in content


def test_generate_drawingml_slide_content_gradient_style():
    """The radial-gradient DrawingML path is the default; explicit selection matches."""
    splat = create_isotropic_splat(
        center=[20.0, 30.0], sigma=5.0, color=[1.0, 0.0, 0.0], alpha=0.5
    )
    content = generate_drawingml_slide_content(
        [splat],
        width=100,
        height=80,
        k_sigma=2.5,
        splat_style="gradient",
    )

    assert "<a:gradFill>" in content
    assert "<a:gsLst>" in content
    assert '<a:path path="shape">' in content
    assert "<a:fillToRect" not in content
    assert 'val="FF0000"' in content  # splat color baked into gradient stops
    assert "<a:alpha val=" in content  # per-stop opacity present
    assert "<a:softEdge" not in content


def test_save_drawingml_writes_file(tmp_path: Path):
    """save_drawingml should persist slide XML to disk."""
    splat = create_isotropic_splat(
        center=[10.0, 10.0], sigma=2.0, color=[0.2, 0.4, 0.6], alpha=0.75
    )
    out = tmp_path / "slide1.xml"
    save_drawingml([splat], width=64, height=64, output_path=str(out), k_sigma=2.5)

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "<p:sld " in text
    assert "<p:sp>" in text
