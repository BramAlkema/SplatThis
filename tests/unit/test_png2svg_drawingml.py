"""Tests for PNG->DrawingML generation utilities."""

import inspect
import re
from pathlib import Path

from splatthis.io import (
    EMU_PER_PX,
    PPTX_PAINTER_ORDER_BACK_TO_FRONT,
    PPTX_PAINTER_ORDER_LEGACY,
    generate_drawingml_slide_content,
    px_to_emu,
    save_drawingml,
)
from splatthis.splat import create_isotropic_splat


def test_px_to_emu_conversion():
    """Pixel to EMU conversion should follow OOXML ratio."""
    assert px_to_emu(1.0) == EMU_PER_PX
    assert px_to_emu(10.0) == 10 * EMU_PER_PX
    # Negative offsets must survive: border-overlapping splats have
    # legitimately negative a:off values (fix for the left/top-edge clamp).
    assert px_to_emu(-5.0) == -5 * EMU_PER_PX


def test_border_overlap_splat_keeps_negative_offset():
    """A splat overlapping the left/top slide edge must emit a negative a:off.

    Regression test for the px_to_emu clamp that displaced every
    border-overlapping splat inward while keeping its full a:ext size.
    """
    splat = create_isotropic_splat(
        center=[2.0, 3.0], sigma=8.0, color=[0.0, 1.0, 0.0], alpha=0.8
    )
    # 128x96 stays at emu_scale 1.0 (the OOXML minimum-slide guard kicks in
    # below 96px), so the expected offsets below are exact.
    content = generate_drawingml_slide_content(
        [splat], width=128, height=96, k_sigma=2.5
    )
    import re

    offsets = [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r'<a:off x="(-?\d+)" y="(-?\d+)"/>', content)
    ]
    # rx = 1.15 * 2.5 * 8 = 23px, so x = 2 - 23 = -21px and y = 3 - 23 = -20px.
    assert any(x < 0 and y < 0 for x, y in offsets), offsets
    assert (px_to_emu(2.0 - 23.0), px_to_emu(3.0 - 23.0)) in offsets


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


def test_drawingml_painter_order_reverses_shape_emission() -> None:
    """Corrected DrawingML must emit the front-to-back input in reverse."""

    red = create_isotropic_splat(
        center=[20.0, 20.0], sigma=5.0, color=[1.0, 0.0, 0.0], alpha=0.7
    )
    blue = create_isotropic_splat(
        center=[30.0, 30.0], sigma=5.0, color=[0.0, 0.0, 1.0], alpha=0.7
    )

    legacy = generate_drawingml_slide_content(
        [red, blue],
        width=64,
        height=64,
        painter_order=PPTX_PAINTER_ORDER_LEGACY,
    )
    # The corrected order is the default since the 21-image real-PowerPoint
    # corpus selected it (median LPIPS 0.320 vs 0.375).
    corrected = generate_drawingml_slide_content([red, blue], width=64, height=64)

    assert legacy.index('val="FF0000"') < legacy.index('val="0000FF"')
    assert corrected.index('val="0000FF"') < corrected.index('val="FF0000"')


def test_corrected_drawingml_reverses_within_layers_not_layer_groups() -> None:
    """Depth groups stay back-to-front while splats reverse inside each group."""

    red = create_isotropic_splat(
        center=[10.0, 10.0], sigma=3.0, color=[1.0, 0.0, 0.0], alpha=0.7
    )
    blue = create_isotropic_splat(
        center=[20.0, 20.0], sigma=3.0, color=[0.0, 0.0, 1.0], alpha=0.7
    )
    green = create_isotropic_splat(
        center=[30.0, 30.0], sigma=3.0, color=[0.0, 1.0, 0.0], alpha=0.7
    )
    red.layer = blue.layer = 0
    green.layer = 1

    corrected = generate_drawingml_slide_content(
        [red, blue, green],
        width=64,
        height=64,
        painter_order=PPTX_PAINTER_ORDER_BACK_TO_FRONT,
    )

    assert corrected.index('val="0000FF"') < corrected.index('val="FF0000"')
    assert corrected.index('val="FF0000"') < corrected.index('val="00FF00"')


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


def test_pptx_gradient_stops_follow_the_shared_opacity_curve() -> None:
    """The emitted stops must equal the curve the training proxy models.

    ``_PPTXGradientProxyRenderer`` reproduces the deck by scaling the alpha
    column by ``PPTX_GRADIENT_ALPHA_SCALE``, which is only correct while the
    emitter's stops follow ``_gaussian_opacity_curve`` at that scaled alpha.
    Asserting against the shared helper -- rather than a retyped copy of the
    formula -- means a change to the curve fails here instead of silently
    splitting the fit from the deck.
    """
    import numpy as np

    from splatthis.export_common import (
        ELLIPSE_OVERLAP_BOOST,
        PPTX_GRADIENT_ALPHA_SCALE,
        _gaussian_opacity_curve,
    )

    stop_pattern = re.compile(
        r'<a:gs pos="(\d+)">\s*<a:srgbClr val="[0-9A-F]{6}">'
        r'(?:<a:alpha val="(\d+)"/>)?'
    )
    # Track the emitter's own default rather than restating it.
    k_sigma = (
        inspect.signature(generate_drawingml_slide_content)
        .parameters["k_sigma"]
        .default
    )
    footprint = ELLIPSE_OVERLAP_BOOST * k_sigma

    for alpha in (0.05, 0.2, 0.5, 0.9):
        splat = create_isotropic_splat(
            center=[50.0, 50.0], sigma=6.0, color=[1.0, 0.0, 0.0], alpha=alpha
        )
        xml = generate_drawingml_slide_content([splat], width=100, height=100)
        shape = re.search(r"<p:sp>.*?</p:sp>", xml, re.S).group(0)
        stops = stop_pattern.findall(shape)
        positions = np.array([int(pos) / 100000 for pos, _ in stops])
        emitted = np.array([(int(a) / 100000 if a else 1.0) for _, a in stops])

        modelled = _gaussian_opacity_curve(
            positions, alpha * PPTX_GRADIENT_ALPHA_SCALE, footprint
        )
        assert np.allclose(modelled, emitted, atol=2e-4), (
            f"alpha={alpha}: emitted stops diverge from the shared curve the "
            f"training proxy models"
        )


def test_pptx_effective_alpha_matches_each_primitive_law() -> None:
    """Each splat style must map alpha the way its emitter does.

    The gradient deck writes ``1 - exp(-alpha * scale * G)``, so the plain
    Gaussian renderer needs ``alpha * scale``. Applying the soft-edge law
    there instead -- the right constant with the wrong curve -- under-models
    opacity by up to 27% at high alpha, and the post-fit stage that used it
    made real PowerPoint captures measurably worse than no post-fit at all.
    """
    import torch

    from splatthis.export_common import (
        PPTX_GRADIENT_ALPHA_SCALE,
        PPTX_SOFT_EDGE_ALPHA_SCALE,
    )
    from splatthis.proxies import pptx_effective_alpha

    alpha = torch.tensor([0.05, 0.2, 0.5, 0.9, 1.0])

    gradient = pptx_effective_alpha(
        alpha, splat_style="gradient", alpha_scale=PPTX_GRADIENT_ALPHA_SCALE
    )
    assert torch.allclose(gradient, alpha * PPTX_GRADIENT_ALPHA_SCALE)

    soft = pptx_effective_alpha(
        alpha, splat_style="soft-edge", alpha_scale=PPTX_SOFT_EDGE_ALPHA_SCALE
    )
    expected_centre = (1.0 - torch.exp(-alpha)) * PPTX_SOFT_EDGE_ALPHA_SCALE
    assert torch.allclose(1.0 - torch.exp(-soft), expected_centre, atol=1e-6)

    # The laws agree near zero and diverge with alpha -- the reason the
    # mix-up survived. Pin that, so a future "simplification" to one law
    # fails here instead of silently in a deck.
    both_at_low_alpha = pptx_effective_alpha(
        torch.tensor([0.01]), splat_style="gradient", alpha_scale=0.4
    ) - pptx_effective_alpha(
        torch.tensor([0.01]), splat_style="soft-edge", alpha_scale=0.4
    )
    assert abs(float(both_at_low_alpha)) < 1e-3
    at_full = pptx_effective_alpha(
        torch.tensor([1.0]), splat_style="gradient", alpha_scale=0.4
    ) - pptx_effective_alpha(
        torch.tensor([1.0]), splat_style="soft-edge", alpha_scale=0.4
    )
    assert float(at_full) > 0.1
