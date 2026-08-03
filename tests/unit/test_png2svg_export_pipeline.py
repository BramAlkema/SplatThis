"""Tests for SVG/PPTX export pipeline helpers."""

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import splatthis.browser_capture as browser_capture
from splatthis.converter import PNG2SVGConverter
from splatthis.io import (
    PPTX_GRADIENT_ALPHA_SCALE,
    atomic_output_path,
    generate_svg_content,
    save_pptx_with_splat_png,
    save_pptx_with_splats,
)
from splatthis.renderer import render_splats_numpy
from splatthis.splat import create_isotropic_splat


def test_generate_svg_content_emits_per_splat_gradients():
    """SVG export should include one radial gradient per splat."""
    splats = [
        create_isotropic_splat(
            center=np.array([10.0, 10.0]),
            sigma=2.0,
            color=np.array([1.0, 0.2, 0.2]),
            alpha=0.8,
        ),
        create_isotropic_splat(
            center=np.array([20.0, 12.0]),
            sigma=3.0,
            color=np.array([0.1, 0.7, 0.4]),
            alpha=0.6,
        ),
    ]
    svg = generate_svg_content(splats, width=32, height=24, k_sigma=2.5)
    assert svg.count("<radialGradient") == 2
    assert svg.count('fill="url(#splat_grad_') == 2


def _overlapping_ordered_splats():
    front = create_isotropic_splat(
        center=np.array([8.0, 8.0]),
        sigma=3.0,
        color=np.array([1.0, 0.0, 0.0]),
        alpha=0.85,
    )
    back = create_isotropic_splat(
        center=np.array([18.0, 8.0]),
        sigma=3.0,
        color=np.array([0.0, 0.0, 1.0]),
        alpha=0.75,
    )
    front.importance = 0.1
    back.importance = 0.9
    return [front, back]


@pytest.mark.parametrize("recipe", ["standard", "browser-compatible"])
def test_static_svg_emits_front_to_back_input_in_reverse_painter_order(recipe):
    """The first alpha-over splat must be the last painted SVG element."""
    svg = generate_svg_content(
        _overlapping_ordered_splats(), width=26, height=16, export_recipe=recipe
    )

    ellipse_ids = re.findall(r'<ellipse id="(splat_\d+)"', svg)

    assert ellipse_ids == ["splat_1", "splat_0"]


def test_static_svg_can_reproduce_legacy_forward_order_explicitly():
    svg = generate_svg_content(
        _overlapping_ordered_splats(),
        width=26,
        height=16,
        painter_order="legacy",
    )

    ellipse_ids = re.findall(r'<ellipse id="(splat_\d+)"', svg)

    assert ellipse_ids == ["splat_0", "splat_1"]


def test_scripted_svg_stores_rows_in_reverse_painter_order():
    svg = generate_svg_content(
        _overlapping_ordered_splats(),
        width=26,
        height=16,
        export_recipe="scripted-matrix",
    )

    payload = re.search(r'<script id="splat-data"[^>]*>([^<]+)</script>', svg).group(1)
    centers = [float(row.split(",")[4]) for row in payload.split(";")]

    assert centers == [18.0, 8.0]


@pytest.mark.parametrize("recipe", ["palette-quantized", "blur"])
def test_compact_svg_recipes_emit_reverse_painter_order(recipe):
    svg = generate_svg_content(
        _overlapping_ordered_splats(), width=26, height=16, export_recipe=recipe
    )

    centers = [float(value) for value in re.findall(r'<ellipse cx="([\d.]+)"', svg)]

    assert centers == [18.0, 8.0]


def test_high_svg_gradient_quality_is_stricter_but_bounded():
    splats = _overlapping_ordered_splats()
    standard = generate_svg_content(splats, width=26, height=16)
    high = generate_svg_content(splats, width=26, height=16, gradient_quality="high")

    standard_stops = standard.count("<stop ")
    high_stops = high.count("<stop ")

    assert standard_stops < high_stops <= 9 * len(splats)
    assert re.search(r'stop-opacity="\d\.\d{4}"', high)


def test_generate_svg_content_can_embed_background_rect():
    """SVG export should include an explicit background when requested."""
    splats = [
        create_isotropic_splat(
            center=np.array([12.0, 10.0]),
            sigma=2.0,
            color=np.array([0.5, 0.3, 0.2]),
            alpha=0.7,
        )
    ]
    svg = generate_svg_content(
        splats,
        width=32,
        height=24,
        k_sigma=2.5,
        background_linear_rgb=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert 'class="background"' in svg
    assert '<rect x="0" y="0" width="32" height="24"' in svg


def test_max_fidelity_default_svg_is_static_not_scripted():
    """Default SVG output should remain usable in static consumers such as <img>."""
    splats = [
        create_isotropic_splat(
            center=np.array([12.0, 10.0]),
            sigma=2.0,
            color=np.array([0.5, 0.3, 0.2]),
            alpha=0.7,
        )
    ]
    converter = PNG2SVGConverter(
        max_splats=1,
        quality_profile="max-fidelity",
        apple_silicon_splat_cap=None,
    )

    svg = converter._generate_svg(splats, width=32, height=24)

    assert converter.svg_export_recipe == "standard"
    assert converter.svg_gradient_quality == "standard"
    assert converter.svg_compositor_gate is True
    assert 'id="splat-data"' not in svg
    assert "<script" not in svg
    assert "<radialGradient" in svg
    assert 'fill="url(#splat_grad_' in svg


def test_browser_compatible_svg_recipe_feathers_and_clamps_background_alpha():
    """Browser recipe should expand splats and cap safe-background opacity."""
    splat = create_isotropic_splat(
        center=np.array([8.0, 8.0]),
        sigma=2.0,
        color=np.array([0.2, 0.2, 0.2]),
        alpha=0.8,
    )
    background_safe = np.ones((20, 20), dtype=bool)
    foreground = np.zeros((20, 20), dtype=bool)
    edge_band = np.zeros((20, 20), dtype=bool)

    svg = generate_svg_content(
        [splat],
        width=20,
        height=20,
        k_sigma=2.5,
        background_linear_rgb=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        export_recipe="browser-compatible",
        foreground_mask=foreground,
        background_safe_mask=background_safe,
        edge_band_mask=edge_band,
    )

    assert 'rx="11.50" ry="11.50"' in svg
    assert 'offset="50.0%"' in svg
    assert 'offset="75.0%" stop-color=' in svg
    # opacity emitted at 2-decimal precision; LPIPS-confirmed visually identical
    # at this precision (see commit 3fba7e2).
    assert 'stop-opacity="0.18"' in svg


def test_scripted_matrix_svg_recipe_stores_compact_splat_rows():
    """Scripted SVG recipe should store matrix rows, not expanded gradients."""
    splats = [
        create_isotropic_splat(
            center=np.array([8.0, 8.0]),
            sigma=2.0,
            color=np.array([0.2, 0.3, 0.4]),
            alpha=0.8,
        ),
        create_isotropic_splat(
            center=np.array([14.0, 10.0]),
            sigma=1.5,
            color=np.array([0.7, 0.2, 0.1]),
            alpha=0.6,
        ),
    ]

    svg = generate_svg_content(
        splats,
        width=20,
        height=20,
        k_sigma=2.5,
        background_linear_rgb=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        export_recipe="scripted-matrix",
    )

    assert 'id="splat-data"' in svg
    assert "data-rendered" in svg
    assert svg.count("<radialGradient") == 0
    assert svg.count("<ellipse") == 0
    assert svg.count(";") >= 1


def test_numpy_renderer_returns_background_when_no_splats():
    """Numpy renderer should emit requested background for empty splat sets."""
    rendered = render_splats_numpy(
        splats=[],
        width=5,
        height=4,
        background_linear_rgb=np.array([0.25, 0.5, 0.75], dtype=np.float32),
    )
    assert rendered.shape == (4, 5, 3)
    assert np.allclose(rendered[0, 0], np.array([0.25, 0.5, 0.75], dtype=np.float32))


def test_save_pptx_with_splat_png_creates_minimal_package(tmp_path: Path):
    """Raster PPTX helper remains available as an explicit fallback."""
    splats = [
        create_isotropic_splat(
            center=np.array([8.0, 8.0]),
            sigma=2.5,
            color=np.array([0.9, 0.1, 0.2]),
            alpha=0.7,
        )
    ]
    out = tmp_path / "slide.pptx"
    save_pptx_with_splat_png(splats=splats, width=32, height=24, output_path=str(out))

    assert out.exists()
    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())
    required = {
        "[Content_Types].xml",
        "_rels/.rels",
        "ppt/presentation.xml",
        "ppt/_rels/presentation.xml.rels",
        "ppt/slides/slide1.xml",
        "ppt/slides/_rels/slide1.xml.rels",
        "ppt/media/image1.png",
    }
    assert required.issubset(names)


def test_save_pptx_with_splats_creates_native_shape_package(tmp_path: Path):
    """Default PPTX export should contain native DrawingML splat shapes, not PNG media."""
    splats = [
        create_isotropic_splat(
            center=np.array([8.0, 8.0]),
            sigma=2.5,
            color=np.array([0.9, 0.1, 0.2]),
            alpha=0.7,
        ),
        create_isotropic_splat(
            center=np.array([18.0, 12.0]),
            sigma=1.5,
            color=np.array([0.1, 0.6, 0.9]),
            alpha=0.5,
        ),
    ]
    splats[0].layer = 0
    splats[0].importance = 0.1
    splats[1].layer = 2
    splats[1].importance = 2.4
    out = tmp_path / "slide_shapes.pptx"
    save_pptx_with_splats(
        splats=splats,
        width=32,
        height=24,
        output_path=str(out),
        background_linear_rgb=np.array([0.05, 0.04, 0.03], dtype=np.float32),
    )

    assert out.exists()
    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())
        slide_xml = zf.read("ppt/slides/slide1.xml").decode("utf-8")
        rels_xml = zf.read("ppt/slides/_rels/slide1.xml.rels").decode("utf-8")

    assert "ppt/media/image1.png" not in names
    assert "<p:pic>" not in slide_xml
    assert slide_xml.count("<p:grpSp>") == 3
    assert 'name="Splat Group"' in slide_xml
    assert 'name="Base Layer"' in slide_xml
    assert 'name="Detail Layer"' in slide_xml
    assert slide_xml.count("<p:sp>") == 3  # background + two splats
    assert 'name="Splat Background"' in slide_xml
    # Default PPTX splat style is now 'gradient' (radial gradient with per-stop
    # alpha) rather than 'soft-edge'. See DEFAULT_PPTX_SPLAT_STYLE.
    assert "<a:gradFill>" in slide_xml
    assert "<a:softEdge" not in slide_xml
    assert "relationships/image" not in rels_xml


def test_converter_exports_pptx_and_comparison_artifacts(tmp_path: Path):
    """Converter should emit PPTX, preview PNG, side-by-side HTML, and manifest metrics."""
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, :12, 0] = 255
    image[:, 12:, 1] = 255
    input_path = tmp_path / "input.png"
    Image.fromarray(image).save(input_path)

    output_path = tmp_path / "output.pptx"
    preview_path = tmp_path / "output_preview.png"
    side_by_side_path = tmp_path / "comparison.html"
    artifacts_path = tmp_path / "artifacts"

    converter = PNG2SVGConverter(
        max_splats=36,
        stages=[1],
        target_size=(24, 24),
        seed=19,
        device="cpu",
        blend_mode="alpha-over",
        layered_saliency=True,
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        output_format="pptx",
        save_json=True,
        verbose=False,
        artifacts_dir=str(artifacts_path),
        preview_png_path=str(preview_path),
        side_by_side_html=str(side_by_side_path),
    )

    assert output_path.exists()
    assert preview_path.exists()
    assert side_by_side_path.exists()
    with zipfile.ZipFile(output_path, "r") as zf:
        names = set(zf.namelist())
        slide_xml = zf.read("ppt/slides/slide1.xml").decode("utf-8")
    assert "ppt/media/image1.png" not in names
    assert "<p:pic>" not in slide_xml
    assert "<p:grpSp>" in slide_xml
    assert 'name="Splat Group"' in slide_xml
    assert 'name="Base Layer"' in slide_xml
    assert 'name="Mass Layer"' in slide_xml
    assert "<p:sp>" in slide_xml
    # Default PPTX splat style flipped from 'soft-edge' to 'gradient'.
    assert "<a:gradFill>" in slide_xml
    assert "<a:softEdge" not in slide_xml
    manifest = json.loads(
        (artifacts_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert "internal_metrics" in manifest
    assert "export_quality" in manifest
    assert manifest["config"]["pptx_export_mode"] == "drawingml-splats"
    assert manifest["config"]["pptx_splat_style"] == "gradient"
    # Corrected order is the default since the real-PowerPoint corpus
    # selected it (median LPIPS 0.320 vs 0.375).
    assert manifest["config"]["pptx_painter_order"] == "back-to-front"
    assert manifest["config"]["layered_saliency"] is True
    assert manifest["layered_saliency"]["enabled"] is True
    assert manifest["artifact_evaluation"] == {
        "render_kind": "pptx-proxy",
        "renderer": "internal-splat-renderer",
        "is_deployed_artifact": False,
        "metric_source": "internal",
    }
    assert manifest["artifacts"]["splat_proxy"]["path"] == str(preview_path)
    assert manifest["artifacts"]["splat_proxy"]["is_deployed_artifact"] is False


def test_converter_can_postfit_scripted_svg_proxy(tmp_path: Path):
    """SVG output can run a tiny browser-proxy color/alpha post-fit stage."""
    image = np.zeros((14, 14, 3), dtype=np.uint8)
    image[:, :7, 0] = 230
    image[:, 7:, 1] = 210
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.svg"
    artifacts_path = tmp_path / "artifacts"
    Image.fromarray(image).save(input_path)

    converter = PNG2SVGConverter(
        max_splats=10,
        stages=[1],
        target_size=(14, 14),
        seed=29,
        device="cpu",
        blend_mode="alpha-over",
        quality_profile="fast",
        refinement_config={
            "svg_export_recipe": "scripted-matrix",
            "svg_proxy_postfit_iters": 1,
        },
        apple_silicon_splat_cap=None,
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        output_format="svg",
        save_json=True,
        verbose=False,
        artifacts_dir=str(artifacts_path),
    )

    svg = output_path.read_text(encoding="utf-8")
    manifest = json.loads(
        (artifacts_path / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert 'id="splat-data"' in svg
    assert any(
        stage.get("stage_type") == "svg_proxy_postfit" for stage in manifest["stages"]
    )
    assert (artifacts_path / "svg-postfit.raw.json").exists()
    assert manifest["export_quality"]["method"].startswith("playwright-chromium/")


def test_svg_acceptance_fails_closed_when_browser_capture_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4, 0] = 220
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.svg"
    artifacts_path = tmp_path / "artifacts"
    Image.fromarray(image).save(input_path)
    monkeypatch.setattr(
        browser_capture,
        "render_svg_in_browser_to_linear_rgb",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("test")),
    )

    converter = PNG2SVGConverter(
        max_splats=4,
        stages=[1],
        target_size=(8, 8),
        seed=37,
        quality_profile="fast",
        device="cpu",
        apple_silicon_splat_cap=None,
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        output_format="svg",
        verbose=False,
        artifacts_dir=str(artifacts_path),
    )

    manifest = json.loads((artifacts_path / "run_manifest.json").read_text())
    assert output_path.exists()
    assert manifest["export_quality"]["available"] is False
    assert manifest["export_quality"]["method"] == "proxy-fallback"
    assert manifest["acceptance_metric_source"] == "unavailable"
    assert manifest["artifact_evaluation"]["render_kind"] == ("svg-browser-unavailable")
    assert manifest["acceptance"]["checks"]["governing_browser_render"] is False
    assert manifest["acceptance"]["pass"] is False
    assert manifest["acceptance"]["reason"] == ("governing-browser-render-unavailable")


def test_converter_can_postfit_pptx_proxy(tmp_path: Path):
    """PPTX output can run a tiny soft-edge proxy color/alpha post-fit stage."""
    image = np.zeros((14, 14, 3), dtype=np.uint8)
    image[:, :7, 0] = 230
    image[:, 7:, 1] = 210
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.pptx"
    artifacts_path = tmp_path / "artifacts"
    Image.fromarray(image).save(input_path)

    converter = PNG2SVGConverter(
        max_splats=10,
        stages=[1],
        target_size=(14, 14),
        seed=31,
        device="cpu",
        blend_mode="alpha-over",
        quality_profile="fast",
        refinement_config={"pptx_proxy_postfit_iters": 1},
        apple_silicon_splat_cap=None,
        layered_saliency=True,
        pptx_splat_style="soft-edge",
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        output_format="pptx",
        save_json=True,
        verbose=False,
        artifacts_dir=str(artifacts_path),
    )

    with zipfile.ZipFile(output_path, "r") as zf:
        names = set(zf.namelist())
        slide_xml = zf.read("ppt/slides/slide1.xml").decode("utf-8")
    manifest = json.loads(
        (artifacts_path / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert "ppt/media/image1.png" not in names
    assert "<p:pic>" not in slide_xml
    assert "<a:softEdge" in slide_xml
    assert any(
        stage.get("stage_type") == "pptx_proxy_postfit" for stage in manifest["stages"]
    )
    assert (artifacts_path / "pptx-postfit.raw.json").exists()


def test_converter_uses_gradient_proxy_defaults_for_gradient_pptx_postfit(
    tmp_path: Path,
):
    """Gradient PPTX post-fit should use the PowerPoint-tuned gradient proxy."""
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, :6, 0] = 220
    image[:, 6:, 2] = 220
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.pptx"
    artifacts_path = tmp_path / "artifacts"
    Image.fromarray(image).save(input_path)

    converter = PNG2SVGConverter(
        max_splats=8,
        stages=[1],
        target_size=(12, 12),
        seed=37,
        device="cpu",
        blend_mode="alpha-over",
        quality_profile="fast",
        refinement_config={"pptx_proxy_postfit_iters": 1},
        apple_silicon_splat_cap=None,
        pptx_splat_style="gradient",
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        output_format="pptx",
        save_json=True,
        verbose=False,
        artifacts_dir=str(artifacts_path),
    )

    with zipfile.ZipFile(output_path, "r") as zf:
        slide_xml = zf.read("ppt/slides/slide1.xml").decode("utf-8")
    manifest = json.loads(
        (artifacts_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    stage = next(
        stage
        for stage in manifest["stages"]
        if stage.get("stage_type") == "pptx_proxy_postfit"
    )

    assert "<a:gradFill>" in slide_xml
    assert '<a:path path="shape">' in slide_xml
    assert "<a:softEdge" not in slide_xml
    assert stage["pptx_splat_style"] == "gradient"
    assert np.isclose(stage["alpha_scale"], PPTX_GRADIENT_ALPHA_SCALE)
    assert np.isclose(stage["sigma_scale"], 1.0)


def test_converter_can_train_against_pptx_proxy(tmp_path: Path):
    """PPTX output can optimize from the start against the soft-edge proxy."""
    image = np.zeros((14, 14, 3), dtype=np.uint8)
    image[:, :7, 0] = 230
    image[:, 7:, 1] = 210
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.pptx"
    artifacts_path = tmp_path / "artifacts"
    Image.fromarray(image).save(input_path)

    converter = PNG2SVGConverter(
        max_splats=10,
        stages=[1],
        target_size=(14, 14),
        seed=37,
        device="cpu",
        blend_mode="alpha-over",
        quality_profile="fast",
        refinement_config={"training_export_target": "pptx-softedge"},
        apple_silicon_splat_cap=None,
        layered_saliency=True,
        pptx_splat_style="soft-edge",
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        output_format="pptx",
        save_json=True,
        verbose=False,
        artifacts_dir=str(artifacts_path),
    )

    with zipfile.ZipFile(output_path, "r") as zf:
        names = set(zf.namelist())
        slide_xml = zf.read("ppt/slides/slide1.xml").decode("utf-8")
    manifest = json.loads(
        (artifacts_path / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["config"]["training_export_target"] == "pptx-softedge"
    assert "ppt/media/image1.png" not in names
    assert "<p:pic>" not in slide_xml
    assert "<a:softEdge" in slide_xml
    assert manifest["stages"][0]["iterations"] == 1


def test_oklab_transform_reference_values():
    """torch_linear_rgb_to_oklab matches Ottosson reference points."""
    import torch

    from splatthis.renderer import torch_linear_rgb_to_oklab

    rgb = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])  # linear white, black
    lab = torch_linear_rgb_to_oklab(rgb)
    # White -> L=1, a=b=0 ; black -> L=a=b=0
    assert torch.allclose(lab[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-3)
    # Black maps near the origin (small clamp floor on the cube root).
    assert torch.allclose(lab[1], torch.tensor([0.0, 0.0, 0.0]), atol=5e-3)


def test_oklab_loss_runs_and_is_differentiable():
    """L1SSIMLoss in oklab space produces a finite, backprop-able scalar."""
    import torch

    from splatthis.renderer import L1SSIMLoss

    rendered = torch.rand(16, 16, 3, requires_grad=True)
    target = torch.rand(16, 16, 3)
    loss = L1SSIMLoss(color_space="oklab")(rendered, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()


def test_spatial_weighted_l1_prioritizes_weighted_pixels():
    """Spatial weights should affect the L1 term while leaving the API differentiable."""
    import torch

    from splatthis.renderer import L1SSIMLoss

    target = torch.zeros(2, 2, 3)
    rendered = torch.zeros(2, 2, 3, requires_grad=True)
    rendered.data[0, 0, :] = 1.0
    rendered.data[1, 1, :] = 1.0
    weights = torch.tensor([[0.1, 1.0], [1.0, 1.0]])

    weighted_loss = L1SSIMLoss(
        l1_weight=1.0, ssim_weight=0.0, spatial_weight_map=weights
    )(
        rendered,
        target,
    )
    unweighted_loss = L1SSIMLoss(l1_weight=1.0, ssim_weight=0.0)(rendered, target)

    assert weighted_loss < unweighted_loss
    weighted_loss.backward()
    assert rendered.grad is not None


def test_luminance_gradient_loss_penalizes_soft_edges():
    """The optional gradient term should push local edge sharpness."""
    import torch

    from splatthis.renderer import L1SSIMLoss

    target = torch.zeros(8, 8, 3)
    target[:, 4:, :] = 1.0
    rendered = target.clone()
    rendered[:, 3, :] = 0.35
    rendered[:, 4, :] = 0.65
    rendered.requires_grad_(True)

    loss = L1SSIMLoss(l1_weight=0.0, ssim_weight=0.0, gradient_weight=1.0)(
        rendered, target
    )

    assert float(loss.detach()) > 0.0
    loss.backward()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()


def _demo_splats(n=6):
    rng = np.random.default_rng(3)
    from splatthis.splat import create_anisotropic_splat

    splats = []
    for i in range(n):
        if i % 2 == 0:
            splats.append(
                create_isotropic_splat(
                    center=rng.uniform(4, 28, size=2),
                    sigma=float(rng.uniform(1.5, 4.0)),
                    color=rng.uniform(0.1, 0.9, size=3),
                    alpha=float(rng.uniform(0.3, 1.0)),
                )
            )
        else:
            angle = float(rng.uniform(0, np.pi))
            vecs = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ],
                dtype=np.float32,
            )
            splats.append(
                create_anisotropic_splat(
                    center=rng.uniform(4, 28, size=2),
                    eigenvals=np.array([9.0, 2.0], dtype=np.float32),
                    eigenvecs=vecs,
                    color=rng.uniform(0.1, 0.9, size=3),
                    alpha=float(rng.uniform(0.3, 1.0)),
                )
            )
    return splats


def test_palette_quantized_recipe_shares_gradients_and_scales_opacity():
    """Palette recipe: <= palette_size shared gradients, exact element opacity."""
    from splatthis.io import generate_palette_quantized_svg_content

    splats = _demo_splats(8)
    svg = generate_palette_quantized_svg_content(
        splats, width=32, height=32, palette_size=4
    )
    assert svg.count("<radialGradient") <= 4
    # Every emitted ellipse references a shared palette gradient and carries
    # the true alpha-over center opacity 1 - exp(-alpha).
    import re

    ellipses = re.findall(r'<ellipse[^>]*opacity="([0-9.]+)"[^>]*url\(#p(\d+)\)', svg)
    assert len(ellipses) == len(splats)
    expected = sorted(
        round(1.0 - np.exp(-min(max(s.alpha, 0.0), 1.0)), 4) for s in splats
    )
    assert sorted(float(op) for op, _ in ellipses) == expected
    # Deterministic across calls (fixed kmeans seed).
    assert svg == generate_palette_quantized_svg_content(
        splats, width=32, height=32, palette_size=4
    )


def test_blur_recipe_filter_region_and_peak_opacity_target():
    """Blur recipe: widened filter region + (1-exp(-a))/mass_fraction opacity."""
    import re

    from splatthis.io import SVG_BLUR_CORE_K_SIGMA, generate_blur_svg_content

    splats = _demo_splats(5)
    svg = generate_blur_svg_content(splats, width=32, height=32)
    assert 'x="-300%"' in svg and 'width="700%"' in svg
    assert "<feGaussianBlur" in svg
    mass_fraction = 1.0 - np.exp(-0.5 * SVG_BLUR_CORE_K_SIGMA**2)
    opacities = sorted(
        float(m) for m in re.findall(r'opacity="([0-9.]+)" filter=', svg)
    )
    expected = sorted(
        round(min((1.0 - np.exp(-min(max(s.alpha, 0.0), 1.0))) / mass_fraction, 1.0), 4)
        for s in splats
    )
    assert opacities == expected


def test_parallax_canvas_html_embeds_strength_and_canvas():
    from splatthis.io import generate_parallax_pixel_runtime_html

    splats = _demo_splats(4)
    html = generate_parallax_pixel_runtime_html(
        splats, width=32, height=32, parallax_strength=7.5
    )
    # Layer canvases are created by the JS runtime, not as static tags.
    assert "canvas" in html.lower()
    assert "const STRENGTH = 7.500;" in html
    assert 'data-layers="1"' in html
    assert "splats.sort" not in html


def test_css_compositor_is_scriptless_and_uses_dom_gradient_splats():
    from splatthis.io import generate_css_splat_html

    splats = _demo_splats(4)
    html = generate_css_splat_html(
        list(reversed(splats)),
        width=32,
        height=24,
        title='unsafe <title> & "quote"',
    )

    assert '<main id="scene" data-compositor="css-splats"' in html
    assert 'data-splat-count="4"' in html
    assert html.count('class="splat"') == 4
    assert "radial-gradient(ellipse 50% 50% at center" in html
    assert "background:transparent}#scene{" in html
    assert "background:transparent}}#scene{" not in html
    assert "<script" not in html.lower()
    assert "<canvas" not in html.lower()
    assert "<svg" not in html.lower()
    assert "<title>unsafe &lt;title&gt; &amp; &quot;quote&quot;</title>" in html


def test_css_compositor_parallax_uses_10x10_hover_grid_without_script():
    from splatthis.io import generate_css_splat_html

    splats = _demo_splats(4)
    for splat, layer in zip(splats, (0, 1, 2, 3)):
        splat.layer = layer
    html = generate_css_splat_html(
        splats,
        width=32,
        height=24,
        parallax_strength=10.0,
        hover_grid_size=10,
    )

    assert 'data-grid="10"' in html
    assert html.count('class="depth-hit h') == 100
    assert ".h0:hover~.plane-midground" in html
    assert ".h99:hover~.plane-foreground" in html
    assert html.count('class="plane plane-') == 3
    assert "<script" not in html.lower()


def test_native_canvas_compositor_submits_gradient_splats_not_imagedata():
    from splatthis.io import generate_native_canvas_html

    splats = _demo_splats(4)
    html = generate_native_canvas_html(splats, width=32, height=24)

    assert 'data-compositor="canvas-api-splats"' in html
    assert 'data-splat-count="4"' in html
    assert "createRadialGradient" in html
    assert "ctx.arc(0, 0, 1" in html
    assert "globalCompositeOperation = 'source-over'" in html
    assert "createImageData" not in html
    assert "putImageData" not in html


def test_native_canvas_parallax_uses_three_browser_drawn_planes():
    from splatthis.io import generate_native_canvas_html

    splats = _demo_splats(4)
    for splat, layer in zip(splats, (0, 1, 2, 3)):
        splat.layer = layer
    html = generate_native_canvas_html(
        splats, width=32, height=24, parallax_strength=10.0
    )

    assert 'data-parallax="true"' in html
    assert '"name":"background"' in html
    assert '"name":"midground"' in html
    assert '"name":"foreground"' in html
    assert "canvas.style.transform" in html
    assert "putImageData" not in html


def test_pixel_runtime_has_accelerated_and_exact_fallback_chain():
    from splatthis.io import generate_webgl_pixel_runtime_html

    html = generate_webgl_pixel_runtime_html(_demo_splats(4), width=32, height=24)

    assert 'data-compositor="pixel-runtime"' in html
    assert "EXT_color_buffer_float" in html
    assert "EXT_float_blend" in html
    assert "gl.RGBA32F" in html
    assert "gl.RGBA16F" in html
    assert "gl.blendFuncSeparate(gl.DST_ALPHA,gl.ONE" in html
    assert "gl.drawArraysInstanced" in html
    assert "webglcontextlost" in html
    assert "document.createElement('canvas')" in html
    assert "OffscreenCanvas" in html
    assert "main-thread-fallback" in html
    assert "splatthisPixelBackend" in html
    assert "checkHalfFloatQuality" in html
    assert "maxError<=2 && meanError<=0.5" in html
    assert "__SPLATTHIS_GPU_QUALITY" in html
    assert "createImageData" in html
    assert "putImageData" in html


def test_pixel_runtime_rejects_unknown_backend():
    from splatthis.io import generate_webgl_pixel_runtime_html

    with pytest.raises(ValueError, match="Unsupported pixel runtime backend"):
        generate_webgl_pixel_runtime_html(
            _demo_splats(1), width=32, height=24, backend="webgpu"
        )


@pytest.mark.parametrize("width,height", [(0, 32), (32, 0), (-1, 32)])
def test_canvas_html_rejects_non_positive_dimensions(width, height):
    from splatthis.io import (
        generate_css_splat_html,
        generate_native_canvas_html,
        generate_parallax_pixel_runtime_html,
        generate_pixel_runtime_html,
    )

    for generator in (
        generate_pixel_runtime_html,
        generate_parallax_pixel_runtime_html,
        generate_css_splat_html,
        generate_native_canvas_html,
    ):
        with pytest.raises(ValueError, match="must be positive"):
            generator([], width=width, height=height)


def test_canvas_html_escapes_title_and_embeds_presorted_splats():
    from splatthis.io import generate_pixel_runtime_html

    splats = _demo_splats(4)
    html = generate_pixel_runtime_html(
        list(reversed(splats)),
        width=32,
        height=32,
        title='unsafe <title> & "quote"',
    )

    assert "<title>unsafe &lt;title&gt; &amp; &quot;quote&quot;</title>" in html
    assert "SPLATS.sort" not in html


def test_atomic_output_keeps_existing_destination_on_failure(tmp_path):
    destination = tmp_path / "nested" / "artifact.svg"
    destination.parent.mkdir()
    destination.write_text("old")

    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_output_path(destination) as temporary:
            temporary.write_text("new")
            raise RuntimeError("interrupted")

    assert destination.read_text() == "old"
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_canvas_html_compositing_space_flag_and_color_encoding():
    from splatthis.io import generate_pixel_runtime_html, linear_to_srgb

    splat = create_isotropic_splat(
        center=np.array([16.0, 16.0]),
        sigma=3.0,
        color=np.array([0.2, 0.2, 0.2]),
        alpha=0.9,
    )
    html_lin = generate_pixel_runtime_html([splat], width=32, height=32)
    html_srgb = generate_pixel_runtime_html(
        [splat], width=32, height=32, compositing_space="srgb"
    )
    assert "const SRGB_IN = false;" in html_lin
    assert "const SRGB_IN = true;" in html_srgb
    assert "new Worker(workerUrl)" in html_lin
    assert "new OffscreenCanvas" in html_lin
    assert "new Uint8ClampedArray(event.data.pixels)" in html_lin
    assert "main-thread-fallback" in html_lin
    assert 'data-execution="pending"' in html_lin
    assert "dataset.splatthisRenderDone = 'true'" in html_lin
    assert "Math.sqrt((sx*ct)*(sx*ct) + (sy*st)*(sy*st))" in html_lin
    encoded = float(linear_to_srgb(np.array([0.2], dtype=np.float32))[0])
    assert f"{encoded:.4f}"[:5] in html_srgb or f"{encoded:.6f}"[:6] in html_srgb


def test_canvas_postprocess_gate_reverts_a_quality_regression():
    converter = PNG2SVGConverter(
        max_splats=2,
        quality_profile="max-fidelity",
        apple_silicon_splat_cap=None,
    )
    optimized = [
        create_isotropic_splat(
            center=np.array([5.0, 8.0]),
            sigma=3.0,
            color=np.array([0.9, 0.1, 0.1]),
            alpha=0.8,
        ),
        create_isotropic_splat(
            center=np.array([11.0, 8.0]),
            sigma=3.0,
            color=np.array([0.1, 0.2, 0.9]),
            alpha=0.8,
        ),
    ]
    target = render_splats_numpy(
        optimized,
        width=16,
        height=16,
        background_linear_rgb=np.zeros(3, dtype=np.float32),
    )
    destructive_candidate = [optimized[0]]

    selected, gate = converter._select_monotonic_canvas_postprocess(
        optimized_splats=optimized,
        postprocessed_splats=destructive_candidate,
        image=target,
    )

    assert selected is optimized
    assert gate["accepted"] is False
    assert gate["decision"] == "revert"
    assert gate["before_count"] == 2
    assert gate["candidate_count"] == 1
    assert gate["selected_count"] == 2
    assert gate["candidate"]["ssim_srgb"] < gate["before"]["ssim_srgb"]


def test_canvas_checkpoint_selection_requires_gain_or_smaller_equivalent():
    converter = PNG2SVGConverter(
        max_splats=4000,
        quality_profile="max-fidelity",
        apple_silicon_splat_cap=None,
    )
    incumbent = {"ssim_srgb": 0.9794, "psnr_srgb": 44.73}

    assert not converter._prefer_canvas_checkpoint(
        candidate={"ssim_srgb": 0.9722, "psnr_srgb": 44.07},
        candidate_count=3680,
        incumbent=incumbent,
        incumbent_count=2000,
    )
    assert converter._prefer_canvas_checkpoint(
        candidate={"ssim_srgb": 0.9810, "psnr_srgb": 44.70},
        candidate_count=3680,
        incumbent=incumbent,
        incumbent_count=2000,
    )
    assert converter._prefer_canvas_checkpoint(
        candidate={"ssim_srgb": 0.9792, "psnr_srgb": 44.70},
        candidate_count=1500,
        incumbent=incumbent,
        incumbent_count=2000,
    )


def test_pixel_runtime_adaptive_compute_stops_before_densification_and_residual(
    tmp_path,
):
    img_path = tmp_path / "tiny.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)).save(
        img_path
    )
    artifacts = tmp_path / "artifacts"
    converter = PNG2SVGConverter(
        max_splats=16,
        stages=[1, 1, 1],
        optimizer_backend="torch",
        apple_silicon_splat_cap=None,
        refinement_config={
            "adaptive_compute_enabled": True,
            "adaptive_compute_target_ssim_srgb": 0.0,
            "adaptive_compute_min_checkpoints": 2,
            "residual_detail_enabled": True,
            "residual_detail_passes": 1,
            "residual_detail_iters": 1,
        },
    )

    converter.convert(
        str(img_path),
        output_path=str(tmp_path / "tiny.html"),
        output_format="pixel-runtime",
        artifacts_dir=str(artifacts),
        verbose=False,
    )

    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    adaptive = next(
        stage
        for stage in manifest["stages"]
        if stage.get("stage_type") == "canvas_adaptive_compute"
    )
    assert adaptive["stopped_early"] is True
    assert adaptive["reason"] == "quality-target"
    assert adaptive["checkpoints_observed"] == 2
    assert adaptive["skipped_main_stages"] == 1
    assert adaptive["skipped_main_stage_iterations"] == 1
    assert adaptive["skipped_residual_detail"] is True
    assert (artifacts / "iter-2.metrics.json").is_file()
    assert not (artifacts / "iter-3.metrics.json").exists()
    assert not (artifacts / "residual-1.metrics.json").exists()


def test_adaptive_compute_rejects_non_pixel_runtime_output(tmp_path):
    img_path = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)
    converter = PNG2SVGConverter(
        max_splats=4,
        stages=[1],
        optimizer_backend="torch",
        apple_silicon_splat_cap=None,
        refinement_config={"adaptive_compute_enabled": True},
    )

    with pytest.raises(ValueError, match="only pixel-runtime"):
        converter.convert(
            str(img_path),
            output_path=str(tmp_path / "tiny.svg"),
            output_format="svg",
            verbose=False,
        )


def test_drawingml_blur_style_alpha_uses_peak_opacity_target():
    import re

    from splatthis.io import PPTX_BLUR_CORE_K_SIGMA, generate_drawingml_slide_content

    splat = create_isotropic_splat(
        center=np.array([16.0, 16.0]),
        sigma=4.0,
        color=np.array([0.8, 0.1, 0.1]),
        alpha=0.7,
    )
    content = generate_drawingml_slide_content(
        [splat], width=32, height=32, splat_style="blur"
    )
    assert "<a:blur rad=" in content
    mass_fraction = 1.0 - np.exp(-0.5 * PPTX_BLUR_CORE_K_SIGMA**2)
    expected_units = int(
        np.clip(
            round(min((1.0 - np.exp(-0.7)) / mass_fraction, 1.0) * 100000), 0, 100000
        )
    )
    alphas = [int(m) for m in re.findall(r'<a:alpha val="(\d+)"/>', content)]
    assert expected_units in alphas


def test_preview_png_srgb_mode_differs_from_linear(tmp_path):
    from splatthis.io import render_splats_preview_png

    splat = create_isotropic_splat(
        center=np.array([16.0, 16.0]),
        sigma=5.0,
        color=np.array([0.6, 0.3, 0.1]),
        alpha=0.5,
    )
    lin_path = tmp_path / "lin.png"
    srgb_path = tmp_path / "srgb.png"
    render_splats_preview_png(
        splats=[splat],
        width=32,
        height=32,
        output_path=str(lin_path),
        background_linear_rgb=np.array([0.9, 0.9, 0.9], dtype=np.float32),
    )
    render_splats_preview_png(
        splats=[splat],
        width=32,
        height=32,
        output_path=str(srgb_path),
        background_linear_rgb=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        compositing_space="srgb",
    )
    lin_img = np.asarray(Image.open(lin_path), dtype=np.int16)
    srgb_img = np.asarray(Image.open(srgb_path), dtype=np.int16)
    assert np.abs(lin_img - srgb_img).max() > 1


def test_cached_linear_framebuffer_writer_matches_preview_renderer(tmp_path):
    from splatthis.io import render_splats_preview_png, save_linear_rgb_png
    from splatthis.renderer import render_splats_numpy

    splats = _demo_splats(4)
    background = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    rendered = render_splats_numpy(
        splats,
        width=32,
        height=24,
        background_linear_rgb=background,
        compositing_space="srgb",
    )
    cached_path = tmp_path / "cached.png"
    direct_path = tmp_path / "direct.png"

    save_linear_rgb_png(rendered, str(cached_path))
    render_splats_preview_png(
        splats,
        width=32,
        height=24,
        output_path=str(direct_path),
        background_linear_rgb=background,
        compositing_space="srgb",
    )

    assert cached_path.read_bytes() == direct_path.read_bytes()


def test_convert_restores_run_mutated_config(tmp_path):
    """convert() must not leave run-mutated config on the instance."""
    import copy

    img_path = tmp_path / "tiny.png"
    rng = np.random.default_rng(0)
    Image.fromarray((rng.uniform(0, 255, size=(16, 16, 3))).astype(np.uint8)).save(
        img_path
    )

    converter = PNG2SVGConverter(
        max_splats=16,
        stages=[2, 1],
        optimizer_backend="torch",
        time_budget="smoke",
    )
    before = (
        converter.max_splats,
        list(converter.stages),
        copy.deepcopy(converter.refinement_config),
        copy.deepcopy(converter.acceptance_criteria),
    )
    out = tmp_path / "tiny.svg"
    converter.convert(str(img_path), output_path=str(out), verbose=False)
    after = (
        converter.max_splats,
        list(converter.stages),
        converter.refinement_config,
        converter.acceptance_criteria,
    )
    assert after == before


def test_embedded_population_round_trips_without_changing_the_drawing() -> None:
    """The population must survive the artifact and not alter a pixel.

    Embedding exists so an SVG can be re-targeted, warm-started, or handed to
    another fitter to beat at the same splat budget. All three depend on the
    payload surviving verbatim; none of them are worth a rendering change, so
    the drawn markup is asserted byte-identical.
    """
    import re

    from splatthis.population_embed import decode_population, population_from_svg
    from splatthis.splat import create_isotropic_splat
    from splatthis.svg_export import generate_svg_content

    splats = [
        create_isotropic_splat(
            center=[10.0 + 3 * i, 12.0 + 2 * i],
            sigma=1.5 + 0.1 * i,
            color=[0.2 + 0.01 * i, 0.5, 0.7],
            alpha=0.3 + 0.02 * i,
        )
        for i in range(12)
    ]

    for recipe in ("standard", "palette-quantized", "blur", "scripted-matrix"):
        plain = generate_svg_content(splats, 64, 64, export_recipe=recipe)
        embedded = generate_svg_content(
            splats, 64, 64, export_recipe=recipe, embed_population=True
        )
        assert "@@METADATA@@" not in embedded, f"{recipe}: placeholder left unfilled"

        recovered = population_from_svg(embedded)
        assert len(recovered) == len(splats), recipe
        for before, after in zip(splats, recovered):
            assert float(after.mu[0]) == pytest.approx(float(before.mu[0]), abs=1e-4)
            assert float(after.alpha) == pytest.approx(float(before.alpha), abs=1e-4)

        stripped = re.sub(r"\n?\s*<metadata>.*?</metadata>", "", embedded, flags=re.S)
        assert (
            stripped.strip() == plain.strip()
        ), f"{recipe}: embedding changed the drawn markup"

    assert "<metadata>" not in generate_svg_content(splats, 64, 64)

    with pytest.raises(ValueError, match="not a splatthis population envelope"):
        decode_population('{"schema": "something.else/1"}')


def test_load_population_accepts_both_artifact_kinds(tmp_path) -> None:
    """A population must load from either artifact this project writes.

    Embedding is only useful if something can read it back: without this the
    `--embed-population` envelope is write-only and the re-target and
    warm-start claims made for it are aspirational.
    """
    from splatthis.population_embed import load_population
    from splatthis.splat import create_isotropic_splat
    from splatthis.storage import save_splats_json
    from splatthis.svg_export import generate_svg_content

    splats = [
        create_isotropic_splat(
            center=[8.0 + i, 9.0 + i], sigma=1.2, color=[0.3, 0.4, 0.5], alpha=0.4
        )
        for i in range(7)
    ]

    svg_path = tmp_path / "embedded.svg"
    svg_path.write_text(
        generate_svg_content(splats, 48, 48, embed_population=True), encoding="utf-8"
    )
    from_svg = load_population(str(svg_path))
    assert len(from_svg) == len(splats)

    json_path = tmp_path / "population.json"
    save_splats_json(splats, str(json_path))
    from_json = load_population(str(json_path))
    assert len(from_json) == len(splats)

    for a, b in zip(from_svg, from_json):
        assert float(a.mu[0]) == pytest.approx(float(b.mu[0]), abs=1e-4)
        assert float(a.alpha) == pytest.approx(float(b.alpha), abs=1e-4)

    plain = tmp_path / "plain.svg"
    plain.write_text(generate_svg_content(splats, 48, 48), encoding="utf-8")
    with pytest.raises(ValueError, match="no embedded splatthis population"):
        load_population(str(plain))


def test_population_survives_every_carrier_without_changing_the_artifact(
    tmp_path,
) -> None:
    """SVG, PPTX and PNG must all carry a population inertly.

    Each format hides the payload somewhere its readers are required to
    ignore -- an SVG <metadata> element, an unreferenced OOXML package part,
    a PNG text chunk. The point of the feature is that the artifact is
    unchanged as an artifact, so that is what gets asserted.
    """
    import numpy as np
    from PIL import Image

    from splatthis.population_embed import (
        load_population,
        png_population_chunk,
        pptx_population_part,
    )
    from splatthis.pptx_export import (
        generate_drawingml_slide_content,
        save_pptx_with_drawingml_content,
    )
    from splatthis.splat import create_isotropic_splat

    splats = [
        create_isotropic_splat(
            center=[6.0 + i, 7.0 + i], sigma=1.1, color=[0.5, 0.3, 0.2], alpha=0.5
        )
        for i in range(9)
    ]

    # PNG: pixels must be byte-identical with and without the chunk.
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[4:12, 4:12] = (200, 120, 60)
    plain_png = tmp_path / "plain.png"
    embedded_png = tmp_path / "embedded.png"
    Image.fromarray(pixels).save(plain_png)
    Image.fromarray(pixels).save(embedded_png, pnginfo=png_population_chunk(splats))
    assert np.array_equal(
        np.asarray(Image.open(plain_png)), np.asarray(Image.open(embedded_png))
    )
    assert len(load_population(str(embedded_png))) == len(splats)

    # PPTX: the part must be declared in [Content_Types].xml, or PowerPoint
    # treats the package as damaged and offers to repair it.
    import zipfile

    slide = generate_drawingml_slide_content(splats, width=32, height=32)
    deck = tmp_path / "embedded.pptx"
    save_pptx_with_drawingml_content(
        slide_xml=slide,
        width=32,
        height=32,
        output_path=str(deck),
        splat_count=len(splats),
        embedded_population=pptx_population_part(splats),
    )
    with zipfile.ZipFile(deck) as package:
        content_types = package.read("[Content_Types].xml").decode("utf-8")
    assert (
        "splatthis/population.json" in content_types
    ), "undeclared package part: PowerPoint will offer to repair this deck"
    assert len(load_population(str(deck))) == len(splats)


def _steg_splats(count: int = 40):
    from splatthis.splat import create_isotropic_splat

    return [
        create_isotropic_splat(
            center=[1.0 + i * 0.25, 2.0 + i * 0.5],
            sigma=1.3,
            color=[0.4, 0.6, 0.2],
            alpha=0.7,
        )
        for i in range(count)
    ]


def _carrier_image(width: int, height: int):
    """A textured RGB image; a flat one hides the payload's cost in PNG size."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    pixels = rng.integers(40, 210, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_population_survives_the_pixel_carrier_round_trip(tmp_path):
    """The in-pixel carrier round-trips through a real PNG file.

    The depth is not stored anywhere, so this covers both rungs of the
    ladder: a large image takes the payload at 1 bit, a smaller one needs 2,
    and recovery has to find each without being told which.
    """
    pytest.importorskip("stego_lsb")
    import math

    from PIL import Image

    from splatthis.population_embed import (
        STEG_BIT_DEPTHS,
        embed_population_in_pixels,
        encode_population,
        population_from_pixels,
        steg_capacity_bytes,
    )

    splats = _steg_splats()
    payload = len(encode_population(splats).encode("utf-8"))
    # Sized from the real payload rather than guessed: one image comfortably
    # takes it at 1 bit, the other is deliberately just too small to.
    roomy = int(math.sqrt(payload * 8 / 3)) * 2
    tight = int(math.sqrt(payload * 8 / 3) * 0.85)

    seen_depths = set()
    for side in (roomy, tight):
        image = _carrier_image(side, side)
        seen_depths.add(
            next(d for d in STEG_BIT_DEPTHS if steg_capacity_bytes(image, d) >= payload)
        )
        path = tmp_path / f"carrier-{side}.png"
        embed_population_in_pixels(image, splats).save(path)
        with Image.open(path) as reread:
            assert len(population_from_pixels(reread)) == len(splats)

    assert seen_depths == set(STEG_BIT_DEPTHS), (
        f"both rungs of the depth ladder must be exercised, only hit "
        f"{sorted(seen_depths)}"
    )


def test_explicit_depth_four_can_still_be_read_back():
    """The write ladder stops at 2; the read ladder must not.

    `bits_per_channel=4` is a documented escape hatch for images too small
    to carry the payload otherwise. If recovery only tried the depths that
    embedding picks automatically, that path would write files nothing
    ships can open.
    """
    pytest.importorskip("stego_lsb")
    from splatthis.population_embed import (
        STEG_BIT_DEPTHS,
        STEG_READ_DEPTHS,
        embed_population_in_pixels,
        population_from_pixels,
    )

    assert set(STEG_BIT_DEPTHS) < set(STEG_READ_DEPTHS), (
        "recovery must try at least every depth embedding can write, "
        "including the explicit-only ones"
    )

    import math

    from splatthis.population_embed import encode_population

    splats = _steg_splats()
    # Sized so 2 bits genuinely fall short and 4 genuinely suffice: at side
    # ~= sqrt(payload) the capacities are 0.75x and 1.5x the payload. Being
    # too small for the automatic ladder is the only reason to reach for
    # depth 4 at all, so the test has to actually be in that regime.
    side = int(math.sqrt(len(encode_population(splats).encode("utf-8"))))
    image = _carrier_image(side, side)
    with pytest.raises(ValueError, match="cannot carry"):
        embed_population_in_pixels(image, splats)

    carrier = embed_population_in_pixels(image, splats, bits_per_channel=4)
    assert len(population_from_pixels(carrier)) == len(splats)


def test_pixel_carrier_does_not_mutate_the_caller_image():
    """stego-lsb writes through putdata(); this project scores those images.

    An in-place rewrite of a loaded source would shift every measurement
    taken from it later in the same process.
    """
    pytest.importorskip("stego_lsb")
    import numpy as np

    from splatthis.population_embed import embed_population_in_pixels

    image = _carrier_image(256, 256)
    before = np.asarray(image).copy()
    carrier = embed_population_in_pixels(image, _steg_splats())

    assert np.array_equal(np.asarray(image), before), "caller's image was mutated"
    assert not np.array_equal(np.asarray(carrier), before)
    assert int(np.abs(np.asarray(carrier, int) - before.astype(int)).max()) <= 3


def test_pixel_carrier_refuses_bad_modes_and_overflow():
    """Failing loudly beats degrading the picture without saying so."""
    pytest.importorskip("stego_lsb")
    from splatthis.population_embed import embed_population_in_pixels

    splats = _steg_splats()
    for mode in ("RGBA", "L", "P"):
        with pytest.raises(ValueError, match="needs an RGB image"):
            embed_population_in_pixels(_carrier_image(64, 64).convert(mode), splats)

    # Too small at every rung of the ladder: an error, never a silent
    # escalation to a depth that visibly damages the image.
    with pytest.raises(ValueError, match="cannot carry"):
        embed_population_in_pixels(_carrier_image(16, 16), _steg_splats(400))


def test_png_reader_falls_back_to_pixels_when_the_chunk_is_stripped(tmp_path):
    """The reason the pixel carrier exists, asserted directly.

    Re-saving through PIL drops the text chunk, which is what optimisers and
    metadata strippers do. The population has to survive that.
    """
    pytest.importorskip("stego_lsb")
    from PIL import Image

    from splatthis.population_embed import (
        PNG_POPULATION_KEY,
        embed_population_in_pixels,
        png_population_chunk,
        population_from_png,
    )

    splats = _steg_splats()
    both = tmp_path / "both.png"
    embed_population_in_pixels(_carrier_image(256, 256), splats).save(
        both, pnginfo=png_population_chunk(splats)
    )
    assert len(population_from_png(str(both))) == len(splats)

    stripped = tmp_path / "stripped.png"
    with Image.open(both) as opened:
        opened.save(stripped)
    with Image.open(stripped) as check:
        assert PNG_POPULATION_KEY not in (check.text or {}), "chunk should be gone"
    assert len(population_from_png(str(stripped))) == len(splats)


def test_pixel_carrier_reports_absence_rather_than_guessing(tmp_path):
    pytest.importorskip("stego_lsb")
    from splatthis.population_embed import population_from_png

    plain = tmp_path / "plain.png"
    _carrier_image(128, 128).save(plain)
    with pytest.raises(ValueError, match="no embedded splatthis population"):
        population_from_png(str(plain))


def test_email_safe_css_variant_stands_alone_and_fits_the_budget():
    """The email variant must survive a stripped <style> block.

    Gmail's app removes <style> for non-Gmail accounts, so anything the
    standard recipe keeps in the shared stylesheet -- the ellipse shape, the
    positioning context, the backdrop -- has to be inline or the build
    collapses in exactly the client it exists for.
    """
    import numpy as np

    from splatthis.browser_export import (
        CSS_EMAIL_GRADIENT_STOPS,
        generate_css_splat_html,
    )
    from splatthis.splat import create_isotropic_splat

    splats = [
        create_isotropic_splat(
            center=[10.0 + i, 12.0 + (i % 7)],
            sigma=3.0,
            color=[0.4, 0.2, 0.6],
            alpha=0.55,
        )
        for i in range(40)
    ]
    background = np.asarray([0.5, 0.2, 0.1], dtype=np.float32)
    html = generate_css_splat_html(
        splats,
        width=120,
        height=100,
        background_linear_rgb=background,
        email_safe=True,
    )

    assert "<style" not in html, "a stripped <style> block must cost nothing"
    for absent in ("mask-image", "color(srgb-linear", 'class="splat"'):
        assert absent not in html, f"{absent} is not available in mail clients"

    # Every declaration the stylesheet used to provide, now per element.
    assert html.count("position:absolute") >= len(splats)
    assert html.count("border-radius:50%") >= len(splats)
    # The scene keeps its positioning context and backdrop inline.
    assert "position:relative" in html and "background:rgb(" in html
    # Gradient size stated explicitly; the default is farthest-corner, which
    # is sqrt(2) larger and silently changes every splat's footprint.
    assert html.count("radial-gradient(ellipse 50% 50% at center") == len(splats)
    assert html.count("rgba(") == len(splats) * CSS_EMAIL_GRADIENT_STOPS

    standard = generate_css_splat_html(
        splats, width=120, height=100, background_linear_rgb=background
    )
    assert len(html) < len(standard), "the email variant must be the smaller one"
