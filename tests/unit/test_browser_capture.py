"""Dimension and metadata contracts for Playwright SVG capture."""

from pathlib import Path

import numpy as np
import pytest

import splatthis.browser_capture as browser_capture
from splatthis.browser_capture import (
    PlaywrightSvgRenderer,
    SvgCaptureResult,
    read_svg_pixel_size,
    resolve_browser_executable,
)
from splatthis.io import (
    atomic_write_text,
    evaluate_css_export_quality,
    evaluate_native_canvas_export_quality,
    evaluate_pixel_runtime_export_quality,
    evaluate_svg_export_quality,
    generate_svg_content,
    generate_webgl_pixel_runtime_html,
    linear_to_srgb,
)
from splatthis.renderer import render_splats_numpy
from splatthis.splat import GaussianSplat


def test_read_svg_pixel_size_prefers_explicit_dimensions(tmp_path: Path) -> None:
    svg = tmp_path / "fixed.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="384px" height="335" '
        'viewBox="0 0 10 10"/>'
    )

    assert read_svg_pixel_size(svg) == (384, 335)


def test_read_svg_pixel_size_accepts_origin_zero_viewbox(tmp_path: Path) -> None:
    svg = tmp_path / "viewbox.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 48"/>')

    assert read_svg_pixel_size(svg) == (64, 48)


@pytest.mark.parametrize(
    "content, message",
    [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="10"/>',
            "unitless or px",
        ),
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="1 0 10 10"/>',
            "start at 0 0",
        ),
    ],
)
def test_read_svg_pixel_size_rejects_context_dependent_geometry(
    tmp_path: Path, content: str, message: str
) -> None:
    svg = tmp_path / "ambiguous.svg"
    svg.write_text(content)

    with pytest.raises(ValueError, match=message):
        read_svg_pixel_size(svg)


def test_capture_result_reports_repeat_stability(tmp_path: Path) -> None:
    svg = tmp_path / "sample.svg"
    output = tmp_path / "sample.png"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>')
    result = SvgCaptureResult(
        svg=svg,
        output=output,
        browser_version="140.0",
        browser_executable=Path("/browser"),
        width=1,
        height=1,
        warmup_captures=1,
        capture_time_ms_samples=(4.0, 2.0, 3.0),
        sample_sha256=("same", "same", "same"),
        sample_outputs=(),
    )

    metadata = result.as_dict()

    assert metadata["capture_time_ms"] == 3.0
    assert metadata["pixel_stable"] is True
    assert metadata["device_scale_factor"] == 1
    assert metadata["warmup_captures"] == 1


def test_configured_browser_executable_overrides_platform_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "chrome"
    executable.touch()
    monkeypatch.setenv("SPLATTHIS_BROWSER_EXECUTABLE", str(executable))

    assert resolve_browser_executable() == executable.resolve()


def test_export_quality_keeps_proxy_diagnostic_but_fails_browser_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = np.zeros((2, 2, 3), dtype=np.float32)
    monkeypatch.setattr(
        browser_capture,
        "render_svg_in_browser_to_linear_rgb",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("test")),
    )

    result = evaluate_svg_export_quality(
        target_linear_rgb=target,
        svg_path="missing.svg",
        fallback_linear_rgb=target,
    )

    assert result["available"] is False
    assert result["method"] == "proxy-fallback"
    assert result["governing_method"] == "unavailable:test"
    assert result["used_fallback"] is True
    assert result["metrics"]["ssim_srgb"] == pytest.approx(1.0)


def test_css_export_quality_requires_governing_browser_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = np.zeros((2, 2, 3), dtype=np.float32)
    monkeypatch.setattr(
        browser_capture,
        "render_css_html_in_browser_to_linear_rgb",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("test")),
    )

    result = evaluate_css_export_quality(
        target_linear_rgb=target,
        html_path="missing.html",
        fallback_linear_rgb=target,
    )

    assert result["available"] is False
    assert result["method"] == "proxy-fallback"
    assert result["governing_method"] == "unavailable:test"
    assert result["used_fallback"] is True


def test_native_canvas_quality_requires_governing_browser_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = np.zeros((2, 2, 3), dtype=np.float32)
    monkeypatch.setattr(
        browser_capture,
        "render_canvas_html_in_browser_to_linear_rgb",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("test")),
    )

    result = evaluate_native_canvas_export_quality(
        target_linear_rgb=target,
        html_path="missing.html",
        fallback_linear_rgb=target,
    )

    assert result["available"] is False
    assert result["method"] == "proxy-fallback"
    assert result["governing_method"] == "unavailable:test"


def test_pixel_runtime_quality_requires_governing_browser_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = np.zeros((2, 2, 3), dtype=np.float32)
    monkeypatch.setattr(
        browser_capture,
        "render_pixel_runtime_html_in_browser_to_linear_rgb",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("test")),
    )

    result = evaluate_pixel_runtime_export_quality(
        target_linear_rgb=target,
        html_path="missing.html",
        fallback_linear_rgb=target,
    )

    assert result["available"] is False
    assert result["method"] == "proxy-fallback"
    assert result["governing_method"] == "unavailable:test"


@pytest.mark.skipif(
    not browser_capture.browser_capture_configured(),
    reason="local Chromium capture is unavailable",
)
def test_browser_svg_center_pixel_matches_front_to_back_alpha_over(
    tmp_path: Path,
) -> None:
    """Catch accidental forward DOM emission with two strongly overlapping colors."""
    front = GaussianSplat(
        mu=np.array([16.0, 16.0], dtype=np.float32),
        sigma=np.array([[100.0, 0.0], [0.0, 100.0]], dtype=np.float32),
        color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        alpha=0.9,
        importance=0.1,
    )
    back = GaussianSplat(
        mu=np.array([16.0, 16.0], dtype=np.float32),
        sigma=np.array([[100.0, 0.0], [0.0, 100.0]], dtype=np.float32),
        color=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        alpha=0.9,
        importance=0.9,
    )
    splats = [front, back]
    background = np.zeros(3, dtype=np.float32)
    svg = tmp_path / "overlap.svg"
    atomic_write_text(
        svg,
        generate_svg_content(
            splats,
            width=32,
            height=32,
            background_linear_rgb=background,
            gradient_quality="high",
        ),
    )

    rendered, _ = browser_capture.render_svg_in_browser_to_linear_rgb(svg, 32, 32)
    expected = render_splats_numpy(
        splats,
        width=32,
        height=32,
        background_linear_rgb=background,
        compositing_space="srgb",
    )

    np.testing.assert_allclose(
        linear_to_srgb(rendered[16, 16]),
        linear_to_srgb(expected[16, 16]),
        atol=3.0 / 255.0,
    )


@pytest.mark.skipif(
    not browser_capture.browser_capture_configured(),
    reason="local Chromium capture is unavailable",
)
def test_forced_pixel_runtime_backends_render_with_exact_cpu_fallbacks(
    tmp_path: Path,
) -> None:
    splats = [
        GaussianSplat(
            mu=np.array([8.0, 8.0], dtype=np.float32),
            sigma=np.array([[9.0, 0.0], [0.0, 4.0]], dtype=np.float32),
            color=np.array([0.8, 0.2, 0.1], dtype=np.float32),
            alpha=0.7,
        )
    ]
    paths = {
        name: tmp_path / f"{name}.html"
        for name in ("rgba32f", "rgba16f", "worker", "main")
    }
    for backend, path in paths.items():
        atomic_write_text(
            str(path),
            generate_webgl_pixel_runtime_html(
                splats, width=16, height=16, backend=backend
            ),
        )

    worker, worker_label = (
        browser_capture.render_pixel_runtime_html_in_browser_to_linear_rgb(
            paths["worker"], 16, 16
        )
    )
    main, main_label = (
        browser_capture.render_pixel_runtime_html_in_browser_to_linear_rgb(
            paths["main"], 16, 16
        )
    )

    assert worker_label.endswith(":worker-offscreen")
    assert main_label.endswith(":main-thread-fallback")
    np.testing.assert_array_equal(worker, main)

    for backend in ("rgba32f", "rgba16f"):
        rendered, label = (
            browser_capture.render_pixel_runtime_html_in_browser_to_linear_rgb(
                paths[backend], 16, 16
            )
        )
        assert label.endswith(
            (f":webgl2-{backend}", ":worker-offscreen", ":main-thread-fallback")
        )
        # A supported GPU path is quality-bounded; an unsupported one reaches
        # one of the exact CPU fallbacks asserted above.
        assert float(np.max(np.abs(rendered - main))) < 0.02


def test_renderer_reuses_one_page_and_resizes_it() -> None:
    class FakePage:
        def __init__(self) -> None:
            self.resizes: list[dict[str, int]] = []
            self.closed = False

        def emulate_media(self, **kwargs: str) -> None:
            assert kwargs == {"reduced_motion": "reduce"}

        def set_viewport_size(self, size: dict[str, int]) -> None:
            self.resizes.append(size)

        def close(self) -> None:
            self.closed = True

    class FakeBrowser:
        def __init__(self) -> None:
            self.page = FakePage()
            self.new_page_calls: list[dict[str, object]] = []

        def new_page(self, **kwargs: object) -> FakePage:
            self.new_page_calls.append(kwargs)
            return self.page

    browser = FakeBrowser()
    renderer = PlaywrightSvgRenderer()
    renderer._browser = browser

    first = renderer._page_for(384, 256)
    same = renderer._page_for(384, 256)
    resized = renderer._page_for(400, 300)

    assert first is same is resized
    assert len(browser.new_page_calls) == 1
    assert browser.page.resizes == [{"width": 400, "height": 300}]
    renderer._close_page()
    assert browser.page.closed is True
