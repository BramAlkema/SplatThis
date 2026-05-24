"""Tests for SVG/PPTX export pipeline helpers."""

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.io import (
    PPTX_GRADIENT_ALPHA_SCALE,
    generate_svg_content,
    save_pptx_with_splat_png,
    save_pptx_with_splats,
)
from png2svg_gs.renderer import render_splats_numpy
from png2svg_gs.splat import create_isotropic_splat


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
    assert 'stop-opacity="0.18127"' in svg


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
    assert manifest["config"]["layered_saliency"] is True
    assert manifest["layered_saliency"]["enabled"] is True


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

    from png2svg_gs.renderer import torch_linear_rgb_to_oklab

    rgb = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])  # linear white, black
    lab = torch_linear_rgb_to_oklab(rgb)
    # White -> L=1, a=b=0 ; black -> L=a=b=0
    assert torch.allclose(lab[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-3)
    # Black maps near the origin (small clamp floor on the cube root).
    assert torch.allclose(lab[1], torch.tensor([0.0, 0.0, 0.0]), atol=5e-3)


def test_oklab_loss_runs_and_is_differentiable():
    """L1SSIMLoss in oklab space produces a finite, backprop-able scalar."""
    import torch

    from png2svg_gs.renderer import L1SSIMLoss

    rendered = torch.rand(16, 16, 3, requires_grad=True)
    target = torch.rand(16, 16, 3)
    loss = L1SSIMLoss(color_space="oklab")(rendered, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()


def test_spatial_weighted_l1_prioritizes_weighted_pixels():
    """Spatial weights should affect the L1 term while leaving the API differentiable."""
    import torch

    from png2svg_gs.renderer import L1SSIMLoss

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

    from png2svg_gs.renderer import L1SSIMLoss

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
