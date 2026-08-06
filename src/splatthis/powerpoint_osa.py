"""Capture a generated PPTX through native Microsoft PowerPoint on macOS.

This is deliberately a small operating-system adapter. It uses Apple's Open
Scripting Architecture (OSA) to drive PowerPoint and the system
``screencapture`` utility to record the rendered window. It has no Python
package dependencies and is invoked only when PPTX capture is requested.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .storage import atomic_output_path

OSA_CAPTURE_SCRIPT = r"""
on run argv
    set sourcePath to item 1 of argv
    set capturePath to item 2 of argv
    set renderDelay to (item 3 of argv) as real
    set openedDeck to missing value

    try
        tell application "Microsoft PowerPoint"
            activate
            open (POSIX file sourcePath)
            set openedDeck to active presentation
        end tell

        delay renderDelay

        tell application "System Events"
            tell process "Microsoft PowerPoint"
                set frontmost to true
                set windowPosition to position of front window
                set windowSize to size of front window
            end tell
        end tell

        set captureRegion to ((item 1 of windowPosition) as text) & "," & ¬
            ((item 2 of windowPosition) as text) & "," & ¬
            ((item 1 of windowSize) as text) & "," & ¬
            ((item 2 of windowSize) as text)
        do shell script "/usr/sbin/screencapture -x -t png -R" & captureRegion & ¬
            " " & quoted form of capturePath

        if openedDeck is not missing value then
            tell application "Microsoft PowerPoint" to close openedDeck saving no
        end if
        return captureRegion
    on error errorMessage number errorNumber
        if openedDeck is not missing value then
            try
                tell application "Microsoft PowerPoint" to close openedDeck saving no
            end try
        end if
        error errorMessage number errorNumber
    end try
end run
"""


def _require_macos_tools() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("PowerPoint OSA capture is available only on macOS")
    missing = [
        executable
        for executable in ("/usr/bin/osascript", "/usr/sbin/screencapture")
        if not Path(executable).is_file()
    ]
    if missing:
        raise RuntimeError(f"required macOS capture tool not found: {missing[0]}")
    app_locations = (
        Path("/Applications/Microsoft PowerPoint.app"),
        Path.home() / "Applications/Microsoft PowerPoint.app",
    )
    if not any(path.is_dir() for path in app_locations):
        raise RuntimeError("Microsoft PowerPoint is required for PPTX capture")


def capture_pptx_with_powerpoint(
    pptx_path: str | Path,
    output_path: str | Path,
    *,
    render_delay: float = 3.0,
) -> dict[str, Any]:
    """Open one PPTX in PowerPoint and capture its front window as a PNG.

    PowerPoint stays running, but the presentation opened for this capture is
    closed without saving. macOS may ask the caller
    to grant Automation, Accessibility, and Screen Recording permissions.
    """

    _require_macos_tools()
    source = Path(pptx_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PPTX not found: {source}")
    if source.suffix.lower() != ".pptx":
        raise ValueError("PowerPoint OSA capture requires a .pptx artifact")
    if render_delay < 0:
        raise ValueError("render_delay must be non-negative")

    with atomic_output_path(destination) as temporary:
        # macOS screencapture exits successfully but refuses dot-prefixed output
        # names, while the atomic publisher intentionally creates one. Stage to
        # a sibling PNG with the same random token, then atomically replace it.
        temporary_png = temporary.with_name(f"{temporary.name.lstrip('.')}.png")
        try:
            command = [
                "/usr/bin/osascript",
                "-l",
                "AppleScript",
                "-e",
                OSA_CAPTURE_SCRIPT,
                str(source),
                str(temporary_png),
                str(float(render_delay)),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=max(30.0, render_delay + 20.0),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"PowerPoint OSA capture timed out after {exc.timeout:g} seconds"
                ) from exc
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                permission_hint = (
                    " Grant Automation, Accessibility, and Screen Recording "
                    "permissions to the calling terminal or application."
                )
                raise RuntimeError(
                    f"PowerPoint OSA capture failed: {detail or 'unknown OSA error'}."
                    f"{permission_hint}"
                )
            try:
                with Image.open(temporary_png) as image:
                    image.verify()
                with Image.open(temporary_png) as image:
                    width, height = image.size
            except Exception as exc:
                raise RuntimeError(
                    "PowerPoint OSA capture did not produce a valid PNG"
                ) from exc
            if width <= 0 or height <= 0:
                raise RuntimeError("PowerPoint OSA capture produced an empty PNG")
            os.replace(temporary_png, temporary)
        finally:
            temporary_png.unlink(missing_ok=True)

    return {
        "schema": "splatthis.powerpoint-osa-capture/1",
        "artifact": str(source),
        "output": str(destination),
        "format": "pptx",
        "capture_method": "Microsoft PowerPoint window via OSA and screencapture",
        "width": int(width),
        "height": int(height),
        "window_region": completed.stdout.strip(),
    }
