"""Contracts for intentionally separated low-level modules."""

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from splatthis.color import (
    linear_to_srgb,
    linear_to_srgb_float32,
    srgb_to_linear,
    srgb_to_linear_float32,
)
from splatthis.pixel_runtime import (
    generate_parallax_pixel_runtime_html,
    generate_pixel_runtime_html,
    generate_webgl_pixel_runtime_html,
)


def test_color_roundtrip_and_scorer_dtype_contract() -> None:
    linear = np.linspace(0.0, 1.0, 33, dtype=np.float32)

    assert np.allclose(srgb_to_linear(linear_to_srgb(linear)), linear, atol=1e-6)
    assert linear_to_srgb_float32(linear).dtype == np.float32
    assert srgb_to_linear_float32(linear).dtype == np.float32


def test_io_retains_established_pixel_runtime_imports() -> None:
    from splatthis import io

    assert io.generate_pixel_runtime_html is generate_pixel_runtime_html
    assert io.generate_webgl_pixel_runtime_html is generate_webgl_pixel_runtime_html
    assert (
        io.generate_parallax_pixel_runtime_html is generate_parallax_pixel_runtime_html
    )


def test_io_is_a_thin_compatibility_facade() -> None:
    from splatthis import artifact_io, browser_export, io, pptx_export, svg_export

    assert io.atomic_write_text is artifact_io.atomic_write_text
    assert io.generate_css_splat_html is browser_export.generate_css_splat_html
    assert io.generate_svg_content is svg_export.generate_svg_content
    assert io.save_pptx_with_splats is pptx_export.save_pptx_with_splats
    assert len(Path(io.__file__).read_text(encoding="utf-8").splitlines()) < 200
    assert (
        len(Path(artifact_io.__file__).read_text(encoding="utf-8").splitlines()) < 100
    )


def test_production_modules_do_not_import_the_io_compatibility_facade() -> None:
    package_dir = Path(__file__).parents[2] / "src" / "splatthis"
    offenders = []
    for source_path in package_dir.rglob("*.py"):
        if source_path.name == "io.py":
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level > 0
                and node.module == "io"
            ):
                offenders.append(f"{source_path.name}:{node.lineno}")

    assert offenders == []


def test_production_modules_do_not_import_artifact_io_facade() -> None:
    package_dir = Path(__file__).parents[2] / "src" / "splatthis"
    allowed = {"artifact_io.py", "io.py"}
    offenders = []
    for source_path in package_dir.rglob("*.py"):
        if source_path.name in allowed:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level > 0
                and node.module == "artifact_io"
            ):
                offenders.append(f"{source_path.name}:{node.lineno}")

    assert offenders == []


def test_converter_coordinator_stays_thin() -> None:
    import inspect

    from splatthis.conversion_engine import ConversionEngine
    from splatthis.converter import PNG2SVGConverter, _PPTXSoftEdgeProxyRenderer
    from splatthis.proxies import _PPTXSoftEdgeProxyRenderer as ProxyRenderer

    facade_path = Path(inspect.getfile(PNG2SVGConverter))
    source_lines = inspect.getsourcelines(PNG2SVGConverter._convert_impl)[0]

    assert PNG2SVGConverter.__module__ == "splatthis.converter"
    assert issubclass(PNG2SVGConverter, ConversionEngine)
    assert _PPTXSoftEdgeProxyRenderer is ProxyRenderer
    assert len(facade_path.read_text(encoding="utf-8").splitlines()) < 25
    assert len(source_lines) < 100


def test_conversion_engine_is_only_a_composition_root() -> None:
    import inspect

    from splatthis.conversion_engine import ConversionEngine

    engine_path = Path(inspect.getfile(ConversionEngine))
    own_methods = {
        name
        for name, value in ConversionEngine.__dict__.items()
        if inspect.isfunction(value)
    }
    responsibility_modules = {
        ConversionEngine.__init__.__module__,
        ConversionEngine._initialize_splats.__module__,
        ConversionEngine._optimize_splats.__module__,
        ConversionEngine._add_error_driven_splats.__module__,
        ConversionEngine._run_color_alpha_postfit.__module__,
        ConversionEngine._generate_svg.__module__,
        ConversionEngine._compute_region_guidance.__module__,
    }

    assert len(engine_path.read_text(encoding="utf-8").splitlines()) < 200
    assert own_methods == {"convert", "_convert_impl"}
    assert responsibility_modules == {
        "splatthis.engine_configuration",
        "splatthis.engine_initialization",
        "splatthis.engine_optimization",
        "splatthis.engine_densification",
        "splatthis.engine_postfit",
        "splatthis.engine_artifacts",
        "splatthis.engine_guidance",
    }


def test_immutable_run_config_and_backend_registry() -> None:
    from splatthis.artifact_backends import get_artifact_backend
    from splatthis.config import (
        SUPPORTED_OUTPUT_FORMATS,
        ConversionRequest,
        ConverterConfig,
    )
    from splatthis.converter import PNG2SVGConverter

    converter = PNG2SVGConverter(
        max_splats=4,
        stages=[1],
        quality_profile="fast",
        refinement_config={"scientific-scalar": np.float32(0.25)},
        apple_silicon_splat_cap=None,
    )
    config = ConverterConfig.from_converter(converter)
    fingerprint = config.fingerprint()
    converter.refinement_config["new-key"] = "caller-mutation"

    assert "new-key" not in config.refinement
    assert config.fingerprint() == fingerprint
    assert {
        get_artifact_backend(name).output_format for name in SUPPORTED_OUTPUT_FORMATS
    } == SUPPORTED_OUTPUT_FORMATS
    with pytest.raises(TypeError):
        config.refinement["forbidden"] = True
    with pytest.raises(ValueError, match="Unsupported output format"):
        ConversionRequest("input.png", output_format="unknown")


def test_vector_markup_lives_in_packaged_templates() -> None:
    import splatthis

    package_dir = Path(splatthis.__file__).parent
    forbidden_markup = re.compile(
        r"<(?:\?xml|!DOCTYPE|/?svg\b|/?ellipse\b|/?radialGradient\b|/?p:|/?a:)"
    )
    offenders = []
    for source_path in package_dir.rglob("*.py"):
        if forbidden_markup.search(source_path.read_text(encoding="utf-8")):
            offenders.append(source_path.relative_to(package_dir).as_posix())

    assert offenders == []
    assert (package_dir / "templates/svg/standard_document.svg").is_file()
    assert (package_dir / "templates/drawingml/shape.xml").is_file()
    assert (package_dir / "templates/pptx/theme.xml").is_file()
    assert (package_dir / "templates/reporting/side_by_side.html").is_file()


def test_template_renderer_rejects_missing_values() -> None:
    from splatthis.template_assets import render_template

    with pytest.raises(ValueError, match="HEIGHT"):
        render_template(
            "svg/empty_document.svg",
            width=10,
            background="",
        )
