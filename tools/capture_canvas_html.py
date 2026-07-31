#!/usr/bin/env python3
"""Capture the exact browser canvas pixel buffer from a SplatThis HTML file.

Run this with a Python environment that provides Playwright. The caller may
point Playwright at an already-installed Chrome, so no downloaded browser is
required.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--browser-executable",
        default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Render the page repeatedly and report the median runtime",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Optionally retain every repeated canvas PNG for noise calibration",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    from playwright.sync_api import sync_playwright

    html_path = args.html.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_dir = args.samples_dir.resolve() if args.samples_dir else None
    if samples_dir is not None:
        samples_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=args.browser_executable,
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            render_samples = []
            png_samples = []
            sample_outputs = []
            data_url = ""
            canvas = None
            for repeat in range(args.repeats):
                page.goto(
                    html_path.as_uri(),
                    wait_until="domcontentloaded",
                    timeout=args.timeout_ms,
                )
                page.wait_for_function(
                    """() =>
                        document.documentElement.dataset.splatthisRenderDone === 'true'
                        || (document.querySelector('#status')?.textContent || '')
                            .startsWith('rendered ')""",
                    timeout=args.timeout_ms,
                )
                canvas = page.locator("#c")
                data_url = canvas.evaluate("(node) => node.toDataURL('image/png')")
                prefix = "data:image/png;base64,"
                if not data_url.startswith(prefix):
                    raise RuntimeError("canvas did not return a PNG data URL")
                png_bytes = base64.b64decode(data_url[len(prefix) :])
                png_samples.append(png_bytes)
                if samples_dir is not None:
                    sample_path = samples_dir / f"repeat-{repeat:03d}.png"
                    sample_path.write_bytes(png_bytes)
                    sample_outputs.append(str(sample_path))
                render_ms = page.evaluate("() => window.__SPLATTHIS_RENDER_MS")
                if render_ms is None:
                    status_text = page.locator("#status").text_content() or ""
                    import re

                    match = re.search(r"\bin ([0-9.]+)ms\b", status_text)
                    render_ms = float(match.group(1)) if match else 0.0
                render_samples.append(float(render_ms))

            assert canvas is not None
            output_path.write_bytes(png_samples[-1])
            metadata = {
                "schema": "splatthis.canvas-capture/1",
                "html": str(html_path),
                "output": str(output_path),
                "browser": browser.version,
                "capture_method": "browser canvas.toDataURL",
                "render_ms": float(statistics.median(render_samples)),
                "render_ms_samples": render_samples,
                "sample_sha256": [
                    hashlib.sha256(payload).hexdigest() for payload in png_samples
                ],
                "sample_outputs": sample_outputs,
                "width": int(canvas.get_attribute("width") or 0),
                "height": int(canvas.get_attribute("height") or 0),
            }
            print(json.dumps(metadata, sort_keys=True))
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
