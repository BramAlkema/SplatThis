"""Deterministic, dimension-locked browser-artifact capture with Chromium.

The browser dependency is imported lazily so normal conversion remains usable
without the optional ``capture`` extra. Callers keep one renderer open across
many artifacts to avoid measuring browser startup for every capture.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import os
import re
import shutil
import statistics
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, Optional, Self

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .color import srgb_to_linear
from .storage import atomic_output_path

DEFAULT_CHROME_EXECUTABLE = Path(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
_PIXEL_LENGTH = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:px)?\s*$")
_SHARED_RENDERER: Optional["PlaywrightSvgRenderer"] = None
_SHARED_RENDERER_LOCK = threading.Lock()


def resolve_browser_executable(explicit: Optional[Path] = None) -> Optional[Path]:
    """Resolve an installed Chromium executable or defer to Playwright."""

    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.environ.get("SPLATTHIS_BROWSER_EXECUTABLE")
    if configured:
        return Path(configured).expanduser().resolve()
    if DEFAULT_CHROME_EXECUTABLE.is_file():
        return DEFAULT_CHROME_EXECUTABLE.resolve()
    if os.name == "nt":
        for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            root = os.environ.get(variable)
            if root:
                candidate = (
                    Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe"
                )
                if candidate.is_file():
                    return candidate.resolve()
    for command in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        executable = shutil.which(command)
        if executable is not None:
            return Path(executable).resolve()
    # Playwright may have a managed Chromium installation. Launching it will
    # produce the authoritative availability error when it does not.
    return None


def browser_capture_configured() -> bool:
    """Return whether the Playwright client and a plausible browser are present."""

    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return resolve_browser_executable() is not None


def _integer_pixels(value: str, *, attribute: str) -> int:
    match = _PIXEL_LENGTH.fullmatch(value)
    if match is None:
        raise ValueError(
            f"SVG {attribute} must use unitless or px dimensions, got {value!r}"
        )
    pixels = float(match.group(1))
    rounded = round(pixels)
    if pixels <= 0 or abs(pixels - rounded) > 1e-6:
        raise ValueError(f"SVG {attribute} must resolve to positive whole pixels")
    return int(rounded)


def read_svg_pixel_size(svg_path: Path) -> tuple[int, int]:
    """Read deterministic pixel dimensions from an SVG root element.

    Width and height take precedence. A unitless/px viewBox is accepted as a
    fallback when it starts at the origin. Percentage and physical units are
    intentionally rejected because their result depends on a containing page.
    """

    root = next(ET.iterparse(svg_path, events=("start",)))[1]
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise ValueError(f"not an SVG root element: {svg_path}")
    width = root.get("width")
    height = root.get("height")
    if width is not None and height is not None:
        return (
            _integer_pixels(width, attribute="width"),
            _integer_pixels(height, attribute="height"),
        )

    view_box = root.get("viewBox")
    if view_box is None:
        raise ValueError("SVG needs width/height or an origin-zero viewBox")
    fields = view_box.replace(",", " ").split()
    if len(fields) != 4:
        raise ValueError(f"invalid SVG viewBox: {view_box!r}")
    x, y = (float(field) for field in fields[:2])
    if abs(x) > 1e-6 or abs(y) > 1e-6:
        raise ValueError("viewBox fallback must start at 0 0")
    return (
        _integer_pixels(fields[2], attribute="viewBox width"),
        _integer_pixels(fields[3], attribute="viewBox height"),
    )


def _validate_capture_geometry(
    geometry: Mapping[str, Any], *, width: int, height: int
) -> None:
    expected = {
        "tag": "svg",
        "x": 0,
        "y": 0,
        "width": width,
        "height": height,
        "dpr": 1,
        "viewportWidth": width,
        "viewportHeight": height,
    }
    for key, expected_value in expected.items():
        actual = geometry.get(key)
        if actual != expected_value:
            raise RuntimeError(
                f"SVG capture geometry mismatch for {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )


def _validate_html_element_geometry(
    geometry: Mapping[str, Any], *, width: int, height: int, selector: str
) -> None:
    expected = {
        "count": 1,
        "x": 0,
        "y": 0,
        "width": width,
        "height": height,
        "dpr": 1,
        "viewportWidth": width,
        "viewportHeight": height,
    }
    for key, expected_value in expected.items():
        actual = geometry.get(key)
        if actual != expected_value:
            raise RuntimeError(
                f"HTML capture geometry mismatch for {selector} {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )


def _validate_png_size(payload: bytes, *, width: int, height: int) -> None:
    with Image.open(BytesIO(payload)) as image:
        if image.size != (width, height):
            raise RuntimeError(
                f"captured PNG is {image.size}, expected {(width, height)}"
            )


def _capture_page_png(
    page: Any,
    *,
    svg_uri: str,
    width: int,
    height: int,
    timeout_ms: int,
    validate_geometry: bool = True,
) -> tuple[bytes, float]:
    started = time.perf_counter()
    page.goto(svg_uri, wait_until="load", timeout=timeout_ms)
    page.evaluate("""async () => {
            if (document.fonts) await document.fonts.ready;
            await new Promise(resolve => requestAnimationFrame(
                () => requestAnimationFrame(resolve)));
        }""")
    if validate_geometry:
        geometry = page.evaluate("""() => {
                const root = document.documentElement;
                const rect = root.getBoundingClientRect();
                return {
                    tag: root.localName,
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height,
                    dpr: window.devicePixelRatio,
                    viewportWidth: window.innerWidth,
                    viewportHeight: window.innerHeight,
                };
            }""")
        _validate_capture_geometry(geometry, width=width, height=height)
    payload = page.screenshot(
        type="png",
        animations="disabled",
        scale="css",
        clip={"x": 0, "y": 0, "width": width, "height": height},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _validate_png_size(payload, width=width, height=height)
    return payload, elapsed_ms


def _capture_html_element_page_png(
    page: Any,
    *,
    html_uri: str,
    selector: str,
    width: int,
    height: int,
    timeout_ms: int,
    validate_geometry: bool = True,
) -> tuple[bytes, float]:
    """Capture one exact-size HTML element with interaction held at rest."""

    started = time.perf_counter()
    page.goto(html_uri, wait_until="load", timeout=timeout_ms)
    geometry = page.evaluate(
        """async selector => {
            if (document.fonts) await document.fonts.ready;
            // Score the compositor's neutral frame. In particular, an old
            // mouse position must not accidentally activate a parallax cell.
            document.documentElement.style.pointerEvents = 'none';
            document.querySelectorAll('.plane').forEach(
                element => element.style.transition = 'none');
            await new Promise(resolve => requestAnimationFrame(
                () => requestAnimationFrame(resolve)));
            const elements = document.querySelectorAll(selector);
            const rect = elements.length === 1
                ? elements[0].getBoundingClientRect()
                : {x: null, y: null, width: null, height: null};
            return {
                count: elements.length,
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
                dpr: window.devicePixelRatio,
                viewportWidth: window.innerWidth,
                viewportHeight: window.innerHeight,
            };
        }""",
        selector,
    )
    if validate_geometry:
        _validate_html_element_geometry(
            geometry, width=width, height=height, selector=selector
        )
    payload = page.screenshot(
        type="png",
        animations="disabled",
        scale="css",
        clip={"x": 0, "y": 0, "width": width, "height": height},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _validate_png_size(payload, width=width, height=height)
    return payload, elapsed_ms


def _capture_pixel_runtime_page_png(
    page: Any,
    *,
    html_uri: str,
    width: int,
    height: int,
    timeout_ms: int,
) -> tuple[bytes, float, str]:
    """Read the completed pixel runtime's own canvas buffer as PNG bytes."""

    started = time.perf_counter()
    page.goto(html_uri, wait_until="load", timeout=timeout_ms)
    page.wait_for_function(
        """() => {
            const root = document.documentElement.dataset;
            return root.splatthisRenderDone === 'true' ||
                Boolean(root.splatthisRenderError);
        }""",
        timeout=timeout_ms,
    )
    result = page.evaluate("""() => {
            const elements = document.querySelectorAll('#c');
            const canvas = elements.length === 1 ? elements[0] : null;
            const root = document.documentElement.dataset;
            return {
                count: elements.length,
                width: canvas ? canvas.width : null,
                height: canvas ? canvas.height : null,
                compositor: canvas ? canvas.dataset.compositor : null,
                execution: canvas ? canvas.dataset.execution : null,
                error: root.splatthisRenderError || null,
                png: canvas ? canvas.toDataURL('image/png').split(',', 2)[1] : null,
            };
        }""")
    if result.get("error"):
        raise RuntimeError(f"pixel runtime failed: {result['error']}")
    expected = {
        "count": 1,
        "width": width,
        "height": height,
        "compositor": "pixel-runtime",
    }
    for key, expected_value in expected.items():
        actual = result.get(key)
        if actual != expected_value:
            raise RuntimeError(
                f"pixel runtime canvas mismatch for {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    execution = str(result.get("execution") or "")
    if execution in {"", "pending"}:
        raise RuntimeError("pixel runtime did not report its execution backend")
    try:
        payload = base64.b64decode(str(result["png"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise RuntimeError("pixel runtime returned an invalid canvas PNG") from exc
    _validate_png_size(payload, width=width, height=height)
    return payload, (time.perf_counter() - started) * 1000.0, execution


@dataclass(frozen=True)
class SvgCaptureResult:
    """Provenance and repeatability data for one browser-rendered SVG."""

    svg: Path
    output: Path
    browser_version: str
    browser_executable: Optional[Path]
    width: int
    height: int
    warmup_captures: int
    capture_time_ms_samples: tuple[float, ...]
    sample_sha256: tuple[str, ...]
    sample_outputs: tuple[Path, ...]

    @property
    def capture_time_ms(self) -> float:
        return float(statistics.median(self.capture_time_ms_samples))

    @property
    def pixel_stable(self) -> bool:
        return len(set(self.sample_sha256)) == 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "splatthis.svg-browser-capture/1",
            "svg": str(self.svg),
            "svg_sha256": hashlib.sha256(self.svg.read_bytes()).hexdigest(),
            "output": str(self.output),
            "browser": self.browser_version,
            "browser_executable": (
                str(self.browser_executable)
                if self.browser_executable is not None
                else "playwright-managed chromium"
            ),
            "capture_method": "Playwright Chromium viewport-clipped PNG screenshot",
            "resource_wait": "load, document.fonts.ready, two animation frames",
            "animations": "disabled during screenshot",
            "device_scale_factor": 1,
            "width": self.width,
            "height": self.height,
            "warmup_captures": self.warmup_captures,
            "capture_time_ms": self.capture_time_ms,
            "capture_time_ms_samples": list(self.capture_time_ms_samples),
            "sample_sha256": list(self.sample_sha256),
            "sample_outputs": [str(path) for path in self.sample_outputs],
            "pixel_stable": self.pixel_stable,
        }


class PlaywrightSvgRenderer:
    """Reuse one headless Chromium process for exact-size browser captures."""

    def __init__(
        self,
        *,
        browser_executable: Optional[Path] = None,
        timeout_ms: int = 120_000,
    ) -> None:
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        if browser_executable is not None:
            browser_executable = browser_executable.expanduser().resolve()
            if not browser_executable.is_file():
                raise FileNotFoundError(
                    f"Chromium executable not found: {browser_executable}"
                )
        self.browser_executable = browser_executable
        self.timeout_ms = timeout_ms
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._page_size: Optional[tuple[int, int]] = None

    @property
    def browser_version(self) -> str:
        if self._browser is None:
            raise RuntimeError("renderer has not been started")
        return str(self._browser.version)

    def __enter__(self) -> Self:
        return self.start()

    def start(self) -> Self:
        """Start Chromium once; repeated calls are idempotent."""

        if self._browser is not None:
            return self
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                'Playwright is required for browser capture; install ".[capture]"'
            ) from exc
        self._playwright = sync_playwright().start()
        launch_options: dict[str, Any] = {"headless": True}
        if self.browser_executable is not None:
            launch_options["executable_path"] = str(self.browser_executable)
        try:
            self._browser = self._playwright.chromium.launch(**launch_options)
        except Exception:
            self._playwright.stop()
            self._playwright = None
            raise
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close Chromium and the Playwright driver if they were started."""

        self._close_page()
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    @property
    def renderer_label(self) -> str:
        return f"playwright-chromium/{self.browser_version}"

    def _close_page(self) -> None:
        page, self._page = self._page, None
        self._page_size = None
        if page is not None:
            try:
                page.close()
            except Exception:
                # The browser may already have torn down a crashed page.
                pass

    def _page_for(self, width: int, height: int) -> Any:
        if self._browser is None:
            raise RuntimeError("renderer has not been started")
        size = (width, height)
        if self._page is None:
            self._page = self._browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
            )
            self._page.emulate_media(reduced_motion="reduce")
            self._page_size = size
        elif self._page_size != size:
            self._page.set_viewport_size({"width": width, "height": height})
            self._page_size = size
        return self._page

    def _capture_payloads(
        self,
        svg_path: Path,
        *,
        width: int,
        height: int,
        repeats: int,
    ) -> tuple[list[bytes], list[float]]:
        if width < 1 or height < 1:
            raise ValueError("capture dimensions must be positive")
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        svg_path = svg_path.resolve()
        if not svg_path.is_file():
            raise FileNotFoundError(svg_path)

        page = self._page_for(width, height)
        svg_uri = svg_path.as_uri()
        payloads: list[bytes] = []
        timings: list[float] = []
        try:
            # The first GPU/filter-pipeline draw can differ by one 8-bit value
            # on a handful of pixels. Warm every artifact, then measure only
            # fresh navigations that must agree byte-for-byte.
            _capture_page_png(
                page,
                svg_uri=svg_uri,
                width=width,
                height=height,
                timeout_ms=self.timeout_ms,
            )
            for _ in range(repeats):
                payload, elapsed_ms = _capture_page_png(
                    page,
                    svg_uri=svg_uri,
                    width=width,
                    height=height,
                    timeout_ms=self.timeout_ms,
                    validate_geometry=False,
                )
                payloads.append(payload)
                timings.append(elapsed_ms)
        except Exception:
            # A failed navigation can leave a page unusable. Keep Chromium but
            # force a clean page on the next capture.
            self._close_page()
            raise
        return payloads, timings

    def render_linear_rgb(
        self,
        svg_path: Path,
        *,
        width: int,
        height: int,
    ) -> NDArray[np.float32]:
        """Capture one governing frame directly to linear RGB in memory."""

        payloads, _ = self._capture_payloads(
            svg_path,
            width=width,
            height=height,
            repeats=1,
        )
        with Image.open(BytesIO(payloads[0])) as image:
            srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return srgb_to_linear(srgb)

    def render_html_element_linear_rgb(
        self,
        html_path: Path,
        *,
        selector: str,
        width: int,
        height: int,
    ) -> NDArray[np.float32]:
        """Capture a dimension-locked HTML element to linear RGB in memory."""

        if width < 1 or height < 1:
            raise ValueError("capture dimensions must be positive")
        html_path = html_path.resolve()
        if not html_path.is_file():
            raise FileNotFoundError(html_path)

        page = self._page_for(width, height)
        html_uri = html_path.as_uri()
        try:
            _capture_html_element_page_png(
                page,
                html_uri=html_uri,
                selector=selector,
                width=width,
                height=height,
                timeout_ms=self.timeout_ms,
            )
            payload, _ = _capture_html_element_page_png(
                page,
                html_uri=html_uri,
                selector=selector,
                width=width,
                height=height,
                timeout_ms=self.timeout_ms,
                validate_geometry=False,
            )
        except Exception:
            self._close_page()
            raise
        with Image.open(BytesIO(payload)) as image:
            srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return srgb_to_linear(srgb)

    def render_pixel_runtime_linear_rgb(
        self,
        html_path: Path,
        *,
        width: int,
        height: int,
    ) -> tuple[NDArray[np.float32], str]:
        """Capture the completed static pixel runtime and selected backend."""

        if width < 1 or height < 1:
            raise ValueError("capture dimensions must be positive")
        html_path = html_path.resolve()
        if not html_path.is_file():
            raise FileNotFoundError(html_path)

        page = self._page_for(width, height)
        html_uri = html_path.as_uri()
        try:
            # Warm shader compilation and browser image encoding, then grade a
            # fresh navigation just as the SVG capture path does.
            _capture_pixel_runtime_page_png(
                page,
                html_uri=html_uri,
                width=width,
                height=height,
                timeout_ms=self.timeout_ms,
            )
            payload, _, execution = _capture_pixel_runtime_page_png(
                page,
                html_uri=html_uri,
                width=width,
                height=height,
                timeout_ms=self.timeout_ms,
            )
        except Exception:
            self._close_page()
            raise
        with Image.open(BytesIO(payload)) as image:
            srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return srgb_to_linear(srgb), execution

    def capture(
        self,
        svg_path: Path,
        output_path: Path,
        *,
        width: int,
        height: int,
        repeats: int = 3,
        samples_dir: Optional[Path] = None,
    ) -> SvgCaptureResult:
        """Capture an SVG without viewport padding or device-scale ambiguity."""

        svg_path = svg_path.resolve()
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if samples_dir is not None:
            samples_dir = samples_dir.resolve()
            samples_dir.mkdir(parents=True, exist_ok=True)

        payloads, timings = self._capture_payloads(
            svg_path,
            width=width,
            height=height,
            repeats=repeats,
        )
        sample_outputs: list[Path] = []
        if samples_dir is not None:
            for repeat, payload in enumerate(payloads):
                sample_path = samples_dir / f"repeat-{repeat:03d}.png"
                with atomic_output_path(sample_path) as temporary:
                    temporary.write_bytes(payload)
                sample_outputs.append(sample_path)

        with atomic_output_path(output_path) as temporary:
            temporary.write_bytes(payloads[-1])
        return SvgCaptureResult(
            svg=svg_path,
            output=output_path,
            browser_version=self.browser_version,
            browser_executable=self.browser_executable,
            width=width,
            height=height,
            warmup_captures=1,
            capture_time_ms_samples=tuple(timings),
            sample_sha256=tuple(
                hashlib.sha256(payload).hexdigest() for payload in payloads
            ),
            sample_outputs=tuple(sample_outputs),
        )


def get_shared_svg_renderer() -> PlaywrightSvgRenderer:
    """Return the process-wide Chromium renderer used by synchronous tooling."""

    global _SHARED_RENDERER
    with _SHARED_RENDERER_LOCK:
        if _SHARED_RENDERER is None:
            renderer = PlaywrightSvgRenderer(
                browser_executable=resolve_browser_executable()
            )
            renderer.start()
            _SHARED_RENDERER = renderer
        return _SHARED_RENDERER


def close_shared_svg_renderer() -> None:
    """Close the process-wide renderer, primarily for orderly interpreter exit."""

    global _SHARED_RENDERER
    with _SHARED_RENDERER_LOCK:
        if _SHARED_RENDERER is not None:
            try:
                _SHARED_RENDERER.close()
            except Exception:
                # Interpreter shutdown can tear the driver down before atexit.
                pass
            finally:
                _SHARED_RENDERER = None


def render_svg_in_browser_to_linear_rgb(
    svg_path: str | Path,
    width: int,
    height: int,
) -> tuple[NDArray[np.float32], str]:
    """Render one SVG with the shared governing Chromium target or fail.

    No library rasterizer or NumPy proxy is attempted here. Callers that permit
    diagnostic proxies must catch the explicit RuntimeError themselves.
    """

    try:
        renderer = get_shared_svg_renderer()
        rendered = renderer.render_linear_rgb(
            Path(svg_path), width=width, height=height
        )
        return rendered, renderer.renderer_label
    except Exception as exc:
        close_shared_svg_renderer()
        raise RuntimeError(
            f"governing Chromium SVG capture failed: {type(exc).__name__}: {exc}"
        ) from exc


def render_css_html_in_browser_to_linear_rgb(
    html_path: str | Path,
    width: int,
    height: int,
) -> tuple[NDArray[np.float32], str]:
    """Render the CSS compositor's scene in shared governing Chromium."""

    try:
        renderer = get_shared_svg_renderer()
        rendered = renderer.render_html_element_linear_rgb(
            Path(html_path), selector="#scene", width=width, height=height
        )
        return rendered, renderer.renderer_label
    except Exception as exc:
        close_shared_svg_renderer()
        raise RuntimeError(
            f"governing Chromium CSS capture failed: {type(exc).__name__}: {exc}"
        ) from exc


def render_canvas_html_in_browser_to_linear_rgb(
    html_path: str | Path,
    width: int,
    height: int,
) -> tuple[NDArray[np.float32], str]:
    """Render a native Canvas-splat scene in shared governing Chromium."""

    try:
        renderer = get_shared_svg_renderer()
        rendered = renderer.render_html_element_linear_rgb(
            Path(html_path), selector="#scene", width=width, height=height
        )
        return rendered, renderer.renderer_label
    except Exception as exc:
        close_shared_svg_renderer()
        raise RuntimeError(
            f"governing Chromium Canvas capture failed: {type(exc).__name__}: {exc}"
        ) from exc


def render_pixel_runtime_html_in_browser_to_linear_rgb(
    html_path: str | Path,
    width: int,
    height: int,
) -> tuple[NDArray[np.float32], str]:
    """Render the selected static pixel-runtime backend in Chromium."""

    try:
        renderer = get_shared_svg_renderer()
        rendered, execution = renderer.render_pixel_runtime_linear_rgb(
            Path(html_path), width=width, height=height
        )
        return rendered, f"{renderer.renderer_label}:{execution}"
    except Exception as exc:
        close_shared_svg_renderer()
        raise RuntimeError(
            "governing Chromium pixel-runtime capture failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


atexit.register(close_shared_svg_renderer)
