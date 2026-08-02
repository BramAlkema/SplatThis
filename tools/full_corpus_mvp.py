#!/usr/bin/env python3
"""Run and report the full 21-image Top-K and mixed-primitives MVPs.

The runner deliberately reuses the deployed seed-0 SVG/PPTX artifacts already
stored below ``result/corpus``. Top-K arms are trained at the corpus resolution;
mixed paths are selected against a native-dimension Chromium capture. Results
are written below ``./tmp/full-corpus-mvp`` by default and are resumable.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "result" / "corpus"
DEFAULT_OUTPUT = REPO / "tmp" / "full-corpus-mvp"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _csv(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _manifest(root: Path) -> dict[str, dict[str, Any]]:
    return json.loads((root / "corpus.json").read_text())["images"]


def _selected_images(root: Path, only: str | None) -> list[tuple[str, dict[str, Any]]]:
    images = _manifest(root)
    names = list(images)
    if only:
        requested = _csv(only)
        missing = sorted(set(requested) - set(images))
        if missing:
            raise SystemExit(f"unknown corpus image(s): {', '.join(missing)}")
        names = requested
    return [(name, images[name]) for name in names]


def _run_logged(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(result.stdout)
    return int(result.returncode), float(elapsed)


def run_topk(
    corpus_root: Path,
    output_root: Path,
    *,
    only: str | None,
    budgets: Iterable[int],
    seeds: Iterable[int],
    iterations: int,
    force: bool,
) -> None:
    images = _selected_images(corpus_root, only)
    jobs = [
        (name, meta, budget, seed)
        for budget in budgets
        for seed in seeds
        for name, meta in images
    ]
    for index, (name, meta, budget, seed) in enumerate(jobs, 1):
        out = output_root / "topk" / f"n{budget}" / f"s{seed}" / name
        comparison = out / f"{name}_comparison.json"
        if comparison.exists() and not force:
            print(f"[topk {index}/{len(jobs)}] {name} n={budget} s={seed}: cached")
            continue
        out.mkdir(parents=True, exist_ok=True)
        source = corpus_root / meta["path"]
        command = [
            sys.executable,
            str(REPO / "tools" / "topk_distillation_mvp.py"),
            str(source),
            "--output-dir",
            str(out),
            "--max-edge",
            str(max(meta["size"])),
            "--splats",
            str(budget),
            "--teacher-iters",
            str(iterations),
            "--student-iters",
            str(iterations),
            "--seed",
            str(seed),
            "--device",
            "mps",
            "--renderer-backend",
            "torch-batched",
            "--tile-size",
            "32",
            "--batch-tile-count",
            "16",
        ]
        print(
            f"[topk {index}/{len(jobs)}] {name} n={budget} s={seed} ... ",
            end="",
            flush=True,
        )
        returncode, elapsed = _run_logged(command, out / "run.log")
        if returncode:
            print(f"FAILED ({elapsed:.1f}s; see {out / 'run.log'})")
        else:
            print(f"done ({elapsed:.1f}s)")


def run_mixed(
    corpus_root: Path,
    output_root: Path,
    *,
    only: str | None,
    force: bool,
) -> None:
    images = _selected_images(corpus_root, only)
    for index, (name, meta) in enumerate(images, 1):
        out = output_root / "mixed" / name
        comparison = out / "comparison.json"
        if comparison.exists() and not force:
            print(f"[mixed {index}/{len(images)}] {name}: cached")
            continue
        out.mkdir(parents=True, exist_ok=True)
        source = corpus_root / meta["path"]
        baseline_svg = corpus_root / "runs" / f"{name}_svg_s0.svg"
        baseline_pptx = corpus_root / "runs" / f"{name}_pptx_s0.pptx"
        command = [
            sys.executable,
            str(REPO / "tools" / "mixed_primitives_mvp.py"),
            str(source),
            str(baseline_svg),
            "--baseline-pptx",
            str(baseline_pptx),
            "--output-dir",
            str(out),
            "--counts",
            "16,32,64",
            "--lengths",
            "12,24",
            "--widths",
            "1.0,2.0",
            "--opacities",
            "0.65",
            "--min-ssim-gain",
            "0.001",
            "--compact",
        ]
        print(f"[mixed {index}/{len(images)}] {name} ... ", end="", flush=True)
        returncode, elapsed = _run_logged(command, out / "run.log")
        if returncode:
            print(f"FAILED ({elapsed:.1f}s; see {out / 'run.log'})")
        else:
            print(f"done ({elapsed:.1f}s)")


def _screen_to_srgb(image):
    """Decode a macOS screenshot into sRGB via its embedded display profile.

    screencapture records the display framebuffer (Display P3 on modern
    Macs) and tags the PNG with that profile. Reading the raw numbers as
    sRGB desaturates every primary -- FF0000 comes back as (234, 51, 35),
    measured in docs/pptx-colorspace.md -- so the screenshot is converted
    to sRGB here, once, before any cropping or scoring.
    """
    icc = image.info.get("icc_profile")
    if not icc:
        return image
    import io as _io

    from PIL import ImageCms

    source_profile = ImageCms.ImageCmsProfile(_io.BytesIO(icc))
    return ImageCms.profileToProfile(
        image.convert("RGB"), source_profile, ImageCms.createProfile("sRGB")
    )


def _capture_powerpoint_slideshow(
    pptx: Path,
    output: Path,
    width: int,
    height: int,
) -> tuple[int, str]:
    from PIL import Image

    pptx = pptx.resolve()
    pptx_text = str(pptx).replace('"', '\\"')
    screen = output.with_name(f"{output.stem}-screen.png")
    script = f"""
with timeout of 60 seconds
tell application "Microsoft PowerPoint"
    activate
    open POSIX file "{pptx_text}"
    set showSettings to slide show settings of active presentation
    set show with presenter of showSettings to false
    set show type of showSettings to slide show type speaker
    set loop until stopped of showSettings to true
    set advance mode of showSettings to slide show advance manual advance
    run slide show showSettings
    set pointer type of slideshow view of slide show window 1 to slide show pointer always hidden
end tell
tell application "System Events"
    set frontmost of process "Microsoft PowerPoint" to true
end tell
delay 2
tell application "Microsoft PowerPoint"
    return bounds of slide show window 1
end tell
end timeout
"""
    result = subprocess.run(
        ["osascript", "-e", script],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        return int(result.returncode), result.stdout
    try:
        window_bounds = tuple(
            int(part.strip()) for part in result.stdout.strip().split(",")
        )
        if len(window_bounds) != 4:
            raise ValueError
    except ValueError:
        window_bounds = (0, 0, 0, 0)
    menu_bar = subprocess.run(
        [
            "osascript",
            "-e",
            (
                'tell application "System Events" to tell process '
                '"Microsoft PowerPoint" to return item 2 of '
                "size of menu bar 1"
            ),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        menu_bar_height = int(menu_bar.stdout.strip())
    except ValueError:
        menu_bar_height = 0
    subprocess.run(
        [
            "osascript",
            "-e",
            (
                'tell application "System Events" to set frontmost of process '
                '"Microsoft PowerPoint" to true'
            ),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    # Move the real system cursor away from PowerPoint's editing controls.
    # Hiding the slideshow pointer alone does not dismiss an already-visible
    # UI tooltip (e.g. "Insert new slide"), which then contaminates the first
    # capture and its artifact metrics.
    subprocess.run(
        [
            "swift",
            "-e",
            ("import CoreGraphics; " "CGWarpMouseCursorPosition(CGPoint(x: 1, y: 1))"),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    time.sleep(1.0)
    window_id_result = subprocess.run(
        [
            "swift",
            "-e",
            (
                "import CoreGraphics; "
                "let info = CGWindowListCopyWindowInfo("
                "[.optionOnScreenOnly, .excludeDesktopElements], "
                "kCGNullWindowID) as! [[String: Any]]; "
                "for window in info { "
                'if (window[kCGWindowOwnerName as String] as? String) == "Microsoft PowerPoint" '
                "&& ((window[kCGWindowName as String] as? String)?"
                '.hasPrefix("PowerPoint Slide Show") ?? false), '
                "let id = window[kCGWindowNumber as String] as? Int { "
                "print(id); break } }"
            ),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        window_id = int(window_id_result.stdout.strip())
    except ValueError:
        window_id = 0
    capture_command = ["screencapture", "-x"]
    if window_id > 0:
        capture_command.extend(["-l", str(window_id)])
    capture_command.append(str(screen.resolve()))
    capture = subprocess.run(
        capture_command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if capture.returncode and window_id > 0:
        capture = subprocess.run(
            ["screencapture", "-x", str(screen.resolve())],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    cleanup_script = """
with timeout of 30 seconds
tell application "Microsoft PowerPoint"
    if (count of slide show windows) > 0 then
        exit slide show slideshow view of slide show window 1
    end if
    if exists active presentation then close active presentation saving no
end tell
end timeout
"""
    cleanup = subprocess.run(
        ["osascript", "-e", cleanup_script],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    messages = result.stdout + window_id_result.stdout + capture.stdout + cleanup.stdout
    if capture.returncode or not screen.exists():
        return int(capture.returncode or 1), messages

    with Image.open(screen) as raw_screen:
        image = _screen_to_srgb(raw_screen)
        crop_box = _powerpoint_slide_crop_box(
            image.size,
            window_bounds,
            menu_bar_height,
            (width, height),
        )
        cropped = image.crop(crop_box).convert("RGB")
        matte_box = _powerpoint_matte_crop_box(cropped, (width, height))
        cropped = cropped.crop(matte_box)
        cropped.resize((width, height), Image.Resampling.LANCZOS).save(output)
    messages += f"\nslide crop: {crop_box}; matte trim: {matte_box}\n"
    screen.unlink(missing_ok=True)
    return 0, messages


def _powerpoint_slide_crop_box(
    screen_size: tuple[int, int],
    window_bounds: tuple[int, int, int, int],
    menu_bar_height: int,
    slide_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Fit a slide into PowerPoint's queried full-screen presentation surface.

    This is the current-app equivalent of the sibling svg2pptx screenshot
    utility's window capture. Current PowerPoint does not expose a usable
    slideshow window ID, so the actual full-screen capture is cropped from the
    window bounds and Accessibility-reported menu-bar safe area.
    """

    screen_width, screen_height = screen_size
    left, top, right, bottom = window_bounds
    logical_width = right - left
    logical_height = bottom - top
    if logical_width <= 0 or logical_height <= 0:
        left, top, right, bottom = 0, 0, screen_width, screen_height
        logical_width, logical_height = screen_width, screen_height

    scale_x = screen_width / float(logical_width)
    scale_y = screen_height / float(logical_height)
    surface_left = int(round(left * scale_x))
    surface_right = int(round(right * scale_x))
    # AX menu-bar size includes its boundary pixel; subtracting one matches the
    # first drawable row in a Retina screencapture.
    safe_top = max(top, max(menu_bar_height - 1, 0))
    surface_top = int(round(safe_top * scale_y))
    surface_bottom = int(round(bottom * scale_y))

    usable_width = surface_right - surface_left
    usable_height = surface_bottom - surface_top
    slide_width, slide_height = slide_size
    aspect = float(slide_width) / float(max(slide_height, 1))
    if usable_width / float(max(usable_height, 1)) > aspect:
        crop_height = usable_height
        crop_width = int(round(crop_height * aspect))
        crop_left = surface_left + (usable_width - crop_width) // 2
        crop_top = surface_top
    else:
        crop_width = usable_width
        crop_height = int(round(crop_width / aspect))
        crop_left = surface_left
        crop_top = surface_top + (usable_height - crop_height) // 2
    return (
        crop_left,
        crop_top,
        crop_left + crop_width,
        crop_top + crop_height,
    )


def _powerpoint_matte_crop_box(
    image: Any,
    slide_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Trim a thin black slideshow matte left inside the queried surface.

    PowerPoint's AppleScript window bounds include a small asymmetric black
    presentation margin on current macOS builds. Only accept a trim when the
    detected non-matte rectangle still occupies most of the preliminary crop
    and closely matches the requested slide aspect ratio.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    full = (0, 0, width, height)
    if width == 0 or height == 0:
        return full

    active = np.max(rgb, axis=2) > 8
    active_rows = np.flatnonzero(np.mean(active, axis=1) > 0.02)
    active_cols = np.flatnonzero(np.mean(active, axis=0) > 0.02)
    if active_rows.size == 0 or active_cols.size == 0:
        return full

    left = int(active_cols[0])
    top = int(active_rows[0])
    right = int(active_cols[-1]) + 1
    bottom = int(active_rows[-1]) + 1
    detected_width = right - left
    detected_height = bottom - top
    if detected_width < 0.85 * width or detected_height < 0.85 * height:
        return full

    slide_width, slide_height = slide_size
    target_aspect = float(slide_width) / float(max(slide_height, 1))
    detected_aspect = float(detected_width) / float(max(detected_height, 1))
    relative_aspect_error = abs(detected_aspect / target_aspect - 1.0)
    if relative_aspect_error > 0.01:
        return full
    return (left, top, right, bottom)


def capture_mixed_powerpoint(
    corpus_root: Path,
    output_root: Path,
    *,
    only: str | None,
    force: bool,
) -> None:
    images = dict(_selected_images(corpus_root, only))
    comparisons = [
        path
        for path in sorted((output_root / "mixed").glob("*/comparison.json"))
        if path.parent.name in images
    ]
    accepted = [
        (path, _load_json(path))
        for path in comparisons
        if _load_json(path).get("winner_pptx")
    ]
    for index, (comparison_path, comparison) in enumerate(accepted, 1):
        name = comparison_path.parent.name
        width, height = map(int, comparison["size"])
        baseline = corpus_root / "runs" / f"{name}_pptx_s0.pptx"
        candidate = Path(comparison["winner_pptx"]["path"])
        if not candidate.is_absolute():
            candidate = REPO / candidate
        print(
            f"[powerpoint {index}/{len(accepted)}] {name} ... ",
            end="",
            flush=True,
        )
        failed = False
        messages = []
        for label, pptx in (("baseline", baseline), ("candidate", candidate)):
            png = comparison_path.parent / f"powerpoint-{label}.png"
            if png.exists() and not force:
                continue
            returncode, message = _capture_powerpoint_slideshow(
                pptx, png, width, height
            )
            if returncode or not png.exists():
                failed = True
                messages.append(f"{label} slideshow capture: {message.strip()}")
                break
        if failed:
            (comparison_path.parent / "powerpoint-capture.log").write_text(
                "\n".join(messages) + "\n"
            )
            print("FAILED")
        else:
            print("done")


def _grid_rois(
    height: int, width: int, tile: int = 64
) -> list[tuple[int, int, int, int]]:
    return [
        (y, x, min(y + tile, height), min(x + tile, width))
        for y in range(0, height, tile)
        for x in range(0, width, tile)
    ]


def _artifact_metrics(
    source: Path,
    artifact: Path,
    width: int,
    height: int,
    *,
    shape_count: int,
) -> dict[str, Any]:
    from splatthis.browser_capture import render_svg_in_browser_to_linear_rgb
    from splatthis.fidelity.metrics import compute_fidelity_metrics
    from splatthis.io import load_png

    target = load_png(str(source), target_size=(width, height))[..., :3]
    rendered, renderer = render_svg_in_browser_to_linear_rgb(
        str(artifact), width, height
    )
    metrics = compute_fidelity_metrics(
        target,
        rendered,
        fixed_rois=_grid_rois(height, width),
        splat_count=shape_count,
        file_size_bytes=artifact.stat().st_size,
        render_method=renderer,
    ).as_dict()
    for key, value in list(metrics.items()):
        if isinstance(value, float) and not np.isfinite(value):
            metrics[key] = None
    return metrics


def _raster_metrics(
    source: Path,
    raster: Path,
    width: int,
    height: int,
    *,
    shape_count: int,
    artifact_size_bytes: int,
    render_method: str,
) -> dict[str, Any]:
    from splatthis.fidelity.metrics import compute_fidelity_metrics
    from splatthis.io import load_png

    target = load_png(str(source), target_size=(width, height))[..., :3]
    rendered = load_png(str(raster), target_size=(width, height))[..., :3]
    metrics = compute_fidelity_metrics(
        target,
        rendered,
        fixed_rois=_grid_rois(height, width),
        splat_count=shape_count,
        file_size_bytes=artifact_size_bytes,
        render_method=render_method,
    ).as_dict()
    for key, value in list(metrics.items()):
        if isinstance(value, float) and not np.isfinite(value):
            metrics[key] = None
    return metrics


HIGHER_IS_BETTER = {"psnr_srgb", "ssim_srgb", "ms_ssim_luma"}
LOWER_IS_BETTER = {
    "lpips",
    "delta_e_ok_mean",
    "delta_e_ok_p95",
    "edge_chamfer",
    "edge_gradient_l1",
    "worst_roi_error",
}


def _deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER):
        before = baseline.get(key)
        after = candidate.get(key)
        result[key] = (
            None if before is None or after is None else float(after) - float(before)
        )
    result["file_size_bytes"] = float(candidate["file_size_bytes"]) - float(
        baseline["file_size_bytes"]
    )
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_summary(corpus_root: Path, output_root: Path) -> dict[str, Any]:
    meta = _manifest(corpus_root)
    topk_records: list[dict[str, Any]] = []
    for comparison_path in sorted(
        (output_root / "topk").glob("n*/s*/*/*_comparison.json")
    ):
        comparison = _load_json(comparison_path)
        name = Path(comparison["input"]).stem
        width, height = map(int, comparison["size"])
        budget = int(comparison["splats"])
        seed = int(comparison["seed"])
        source = corpus_root / meta[name]["path"]
        direct_path = Path(comparison["artifacts"]["direct"]["svg"])
        student_path = Path(comparison["artifacts"]["student"]["svg"])
        direct = _artifact_metrics(
            source, direct_path, width, height, shape_count=budget
        )
        student = _artifact_metrics(
            source, student_path, width, height, shape_count=budget
        )
        topk_records.append(
            {
                "image": name,
                "content_class": meta[name]["content_class"],
                "size": [width, height],
                "budget": budget,
                "seed": seed,
                "source": str(source),
                "direct_svg": str(direct_path),
                "student_svg": str(student_path),
                "teacher_proxy": comparison["proxy"]["teacher"],
                "direct": direct,
                "student": student,
                "delta": _deltas(direct, student),
                "decision": comparison["decision"],
            }
        )

    mixed_records: list[dict[str, Any]] = []
    for comparison_path in sorted((output_root / "mixed").glob("*/comparison.json")):
        comparison = _load_json(comparison_path)
        name = Path(comparison["source"]).stem
        width, height = map(int, comparison["size"])
        source = corpus_root / meta[name]["path"]
        baseline_path = Path(comparison["baseline_svg"])
        raw_path = corpus_root / "runs" / f"{name}_svg_s0_art" / "final.raw.json"
        shape_count = int(_load_json(raw_path)["num_splats"])
        baseline = _artifact_metrics(
            source, baseline_path, width, height, shape_count=shape_count
        )
        winner = comparison.get("winner")
        candidate = None
        delta = None
        if winner:
            candidate_path = Path(winner["path"])
            candidate = _artifact_metrics(
                source,
                candidate_path,
                width,
                height,
                shape_count=shape_count + int(winner["stroke_count"]),
            )
            delta = _deltas(baseline, candidate)
        powerpoint = None
        baseline_capture = comparison_path.parent / "powerpoint-baseline.png"
        candidate_capture = comparison_path.parent / "powerpoint-candidate.png"
        winner_pptx = comparison.get("winner_pptx")
        if (
            winner
            and winner_pptx
            and baseline_capture.exists()
            and candidate_capture.exists()
        ):
            baseline_pptx = corpus_root / "runs" / f"{name}_pptx_s0.pptx"
            candidate_pptx = Path(winner_pptx["path"])
            if not candidate_pptx.is_absolute():
                candidate_pptx = REPO / candidate_pptx
            baseline_powerpoint = _raster_metrics(
                source,
                baseline_capture,
                width,
                height,
                shape_count=shape_count,
                artifact_size_bytes=baseline_pptx.stat().st_size,
                render_method="Microsoft PowerPoint slideshow",
            )
            candidate_powerpoint = _raster_metrics(
                source,
                candidate_capture,
                width,
                height,
                shape_count=shape_count + int(winner_pptx["native_segment_shapes"]),
                artifact_size_bytes=candidate_pptx.stat().st_size,
                render_method="Microsoft PowerPoint slideshow",
            )
            powerpoint = {
                "baseline_capture": str(baseline_capture),
                "candidate_capture": str(candidate_capture),
                "baseline": baseline_powerpoint,
                "candidate": candidate_powerpoint,
                "delta": _deltas(baseline_powerpoint, candidate_powerpoint),
            }
        mixed_records.append(
            {
                "image": name,
                "content_class": meta[name]["content_class"],
                "size": [width, height],
                "source": str(source),
                "baseline_svg": str(baseline_path),
                "candidate_svg": None if winner is None else winner["path"],
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
                "winner": winner,
                "winner_pptx": winner_pptx,
                "powerpoint": powerpoint,
                "decision": comparison["decision"],
            }
        )

    summary = {
        "schema": "splatthis.full-corpus-mvp/1",
        "corpus_images": len(meta),
        "topk": topk_records,
        "mixed": mixed_records,
        "aggregates": {
            "topk": _aggregate(topk_records, ("budget", "seed")),
            "mixed": _aggregate(
                [record for record in mixed_records if record["delta"]], ()
            ),
            "mixed_powerpoint": _aggregate(
                [
                    {
                        **record,
                        "delta": record["powerpoint"]["delta"],
                    }
                    for record in mixed_records
                    if record.get("powerpoint")
                ],
                (),
            ),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _aggregate(
    records: list[dict[str, Any]], group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        if not record.get("delta"):
            continue
        key = tuple(record[field] for field in group_keys)
        groups.setdefault(key, []).append(record)
    output = []
    for key, group in sorted(groups.items()):
        row = {field: value for field, value in zip(group_keys, key)}
        row["n"] = len(group)
        for metric in sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER):
            values = [
                float(record["delta"][metric])
                for record in group
                if record["delta"].get(metric) is not None
            ]
            if not values:
                continue
            improvement_count = sum(
                value > 0 if metric in HIGHER_IS_BETTER else value < 0
                for value in values
            )
            row[f"{metric}_median_delta"] = float(statistics.median(values))
            row[f"{metric}_improved"] = int(improvement_count)
        row["file_size_median_delta"] = float(
            statistics.median(record["delta"]["file_size_bytes"] for record in group)
        )
        output.append(row)
    return output


def _data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }[suffix]
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _delta_class(value: Any, *, lower: bool = False) -> str:
    if value is None or abs(float(value)) < 1e-12:
        return ""
    improved = float(value) < 0 if lower else float(value) > 0
    return "good" if improved else "bad"


def _statline(record: dict[str, Any], side: str) -> str:
    return _metrics_line(record[side])


def _metrics_line(metrics: dict[str, Any]) -> str:
    return (
        f"SSIM {_fmt(metrics['ssim_srgb'])} · "
        f"MS-SSIM {_fmt(metrics['ms_ssim_luma'])} · "
        f"LPIPS {_fmt(metrics['lpips'])} · "
        f"PSNR {_fmt(metrics['psnr_srgb'], 2)} · "
        f"{metrics['file_size_bytes'] / 1024:.0f} KB"
    )


def _powerpoint_panel(record: dict[str, Any]) -> str:
    powerpoint = record.get("powerpoint")
    if not powerpoint:
        return ""
    baseline_uri = _data_uri(Path(powerpoint["baseline_capture"]))
    candidate_uri = _data_uri(Path(powerpoint["candidate_capture"]))
    delta = powerpoint["delta"]
    return f"""
  <div class="powerpoint">
    <h4>Actual Microsoft PowerPoint slideshow capture</h4>
    <div class="pair">
      <figure><img src="{baseline_uri}" alt="baseline rendered by PowerPoint"><figcaption>Baseline PPTX<br>{_metrics_line(powerpoint['baseline'])}</figcaption></figure>
      <figure><img src="{candidate_uri}" alt="mixed candidate rendered by PowerPoint"><figcaption>Mixed PPTX<br>{_metrics_line(powerpoint['candidate'])}</figcaption></figure>
    </div>
    <div class="deltas">
      <span class="{_delta_class(delta['ssim_srgb'])}">ΔSSIM {_fmt(delta['ssim_srgb'], 5)}</span>
      <span class="{_delta_class(delta['ms_ssim_luma'])}">ΔMS-SSIM {_fmt(delta['ms_ssim_luma'], 5)}</span>
      <span class="{_delta_class(delta['lpips'], lower=True)}">ΔLPIPS {_fmt(delta['lpips'], 5)}</span>
      <span class="{_delta_class(delta['edge_chamfer'], lower=True)}">Δedge {_fmt(delta['edge_chamfer'], 3)}</span>
      <span>Δbytes {delta['file_size_bytes'] / 1024:+.1f} KB</span>
    </div>
  </div>"""


def generate_html(summary: dict[str, Any], output_root: Path, output: Path) -> None:
    topk = summary["topk"]
    mixed = summary["mixed"]
    topk_sections = []
    for budget in sorted({int(record["budget"]) for record in topk}):
        budget_records = [r for r in topk if int(r["budget"]) == budget]
        records_by_image: dict[str, list[dict[str, Any]]] = {}
        for record in budget_records:
            records_by_image.setdefault(record["image"], []).append(record)
        cards = []
        for image_name, seed_records in sorted(records_by_image.items()):
            seed_records.sort(key=lambda record: int(record["seed"]))
            record = seed_records[0]
            delta = record["delta"]
            source_uri = _data_uri(Path(record["source"]))
            direct_uri = _data_uri(Path(record["direct_svg"]))
            student_uri = _data_uri(Path(record["student_svg"]))
            seed_rows = "".join(
                f"""
      <tr>
        <td>{seed_record['seed']}</td>
        <td class="{_delta_class(seed_record['delta']['ssim_srgb'])}">{_fmt(seed_record['delta']['ssim_srgb'], 5)}</td>
        <td class="{_delta_class(seed_record['delta']['ms_ssim_luma'])}">{_fmt(seed_record['delta']['ms_ssim_luma'], 5)}</td>
        <td class="{_delta_class(seed_record['delta']['lpips'], lower=True)}">{_fmt(seed_record['delta']['lpips'], 5)}</td>
        <td class="{_delta_class(seed_record['delta']['worst_roi_error'], lower=True)}">{_fmt(seed_record['delta']['worst_roi_error'], 5)}</td>
        <td class="{_delta_class(seed_record['delta']['edge_chamfer'], lower=True)}">{_fmt(seed_record['delta']['edge_chamfer'], 3)}</td>
        <td>{seed_record['delta']['file_size_bytes'] / 1024:+.1f} KB</td>
      </tr>"""
                for seed_record in seed_records
            )
            cards.append(
                f"""
<article class="card">
  <header><h3>{html.escape(image_name)}</h3><span>{html.escape(record['content_class'])}</span></header>
  <div class="triptych">
    <figure><img src="{source_uri}" alt="source"><figcaption>Source</figcaption></figure>
    <figure><img src="{direct_uri}" alt="direct SVG"><figcaption>Seed {record['seed']} direct SVG<br>{_statline(record, 'direct')}</figcaption></figure>
    <figure><img src="{student_uri}" alt="Top-K student SVG"><figcaption>Seed {record['seed']} Top-K student SVG<br>{_statline(record, 'student')}</figcaption></figure>
  </div>
  <div class="deltas">
    <span class="{_delta_class(delta['ssim_srgb'])}">ΔSSIM {_fmt(delta['ssim_srgb'], 5)}</span>
    <span class="{_delta_class(delta['ms_ssim_luma'])}">ΔMS-SSIM {_fmt(delta['ms_ssim_luma'], 5)}</span>
    <span class="{_delta_class(delta['lpips'], lower=True)}">ΔLPIPS {_fmt(delta['lpips'], 5)}</span>
    <span class="{_delta_class(delta['edge_chamfer'], lower=True)}">Δedge {_fmt(delta['edge_chamfer'], 3)}</span>
    <span>Δbytes {delta['file_size_bytes'] / 1024:+.1f} KB</span>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>seed</th><th>ΔSSIM</th><th>ΔMS-SSIM</th><th>ΔLPIPS</th><th>Δworst ROI</th><th>Δedge</th><th>Δbytes</th></tr></thead>
    <tbody>{seed_rows}</tbody>
  </table></div>
</article>"""
            )
        aggregate_rows = [
            row for row in summary["aggregates"]["topk"] if int(row["budget"]) == budget
        ]
        aggregate_table_rows = "".join(
            f"""
    <tr>
      <td>{row['seed']}</td><td>{row['n']}</td>
      <td>{_fmt(row.get('ssim_srgb_median_delta'), 5)} ({row.get('ssim_srgb_improved', 0)}/{row['n']})</td>
      <td>{_fmt(row.get('ms_ssim_luma_median_delta'), 5)} ({row.get('ms_ssim_luma_improved', 0)}/{row['n']})</td>
      <td>{_fmt(row.get('lpips_median_delta'), 5)} ({row.get('lpips_improved', 0)}/{row['n']})</td>
      <td>{row.get('file_size_median_delta', 0) / 1024:+.1f} KB</td>
    </tr>"""
            for row in aggregate_rows
        )
        seed_label = ", ".join(str(row["seed"]) for row in aggregate_rows)
        topk_sections.append(
            f"""
<section>
  <h2>Top-K teacher → alpha-over student · {budget} splats · seed(s) {seed_label}</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>seed</th><th>images</th><th>median ΔSSIM (improved)</th><th>median ΔMS-SSIM (improved)</th><th>median ΔLPIPS (improved)</th><th>median Δbytes</th></tr></thead>
    <tbody>{aggregate_table_rows}</tbody>
  </table></div>
  {''.join(cards)}
</section>"""
        )

    mixed_cards = []
    for record in mixed:
        source_uri = _data_uri(Path(record["source"]))
        baseline_uri = _data_uri(Path(record["baseline_svg"]))
        if record["candidate"] is None:
            candidate_uri = baseline_uri
            candidate_caption = "No candidate cleared the +0.001 SSIM gate"
            deltas = '<span class="bad">reverted to baseline</span>'
        else:
            candidate_uri = _data_uri(Path(record["candidate_svg"]))
            candidate_caption = "Mixed native-path SVG<br>" + _statline(
                record, "candidate"
            )
            delta = record["delta"]
            deltas = (
                f'<span class="{_delta_class(delta["ssim_srgb"])}">ΔSSIM {_fmt(delta["ssim_srgb"], 5)}</span>'
                f'<span class="{_delta_class(delta["ms_ssim_luma"])}">ΔMS-SSIM {_fmt(delta["ms_ssim_luma"], 5)}</span>'
                f'<span class="{_delta_class(delta["lpips"], lower=True)}">ΔLPIPS {_fmt(delta["lpips"], 5)}</span>'
                f'<span class="{_delta_class(delta["edge_chamfer"], lower=True)}">Δedge {_fmt(delta["edge_chamfer"], 3)}</span>'
                f'<span>Δbytes {delta["file_size_bytes"] / 1024:+.1f} KB</span>'
            )
        mixed_cards.append(
            f"""
<article class="card">
  <header><h3>{html.escape(record['image'])}</h3><span>{html.escape(record['content_class'])}</span></header>
  <div class="triptych">
    <figure><img src="{source_uri}" alt="source"><figcaption>Source</figcaption></figure>
    <figure><img src="{baseline_uri}" alt="baseline SVG"><figcaption>Baseline SVG<br>{_statline(record, 'baseline')}</figcaption></figure>
    <figure><img src="{candidate_uri}" alt="mixed SVG"><figcaption>{candidate_caption}</figcaption></figure>
  </div>
  <div class="deltas">{deltas}</div>
  {_powerpoint_panel(record)}
</article>"""
        )
    mixed_aggregate = (
        summary["aggregates"]["mixed"][0] if summary["aggregates"]["mixed"] else {}
    )
    powerpoint_aggregate = (
        summary["aggregates"]["mixed_powerpoint"][0]
        if summary["aggregates"]["mixed_powerpoint"]
        else {}
    )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SplatThis full-corpus MVPs</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin:0; background:#090b10; color:#e8edf6; }}
main {{ max-width:1680px; margin:auto; padding:32px; }}
h1 {{ font-size:clamp(2rem,5vw,4.5rem); margin:.2em 0; }}
h2 {{ margin-top:64px; border-top:1px solid #283040; padding-top:32px; }}
.lede,.summary {{ color:#aeb8c8; max-width:1000px; line-height:1.6; }}
.card {{ background:#111722; border:1px solid #283040; border-radius:18px; padding:18px; margin:20px 0; }}
.card header {{ display:flex; justify-content:space-between; align-items:baseline; }}
.card header span,figcaption {{ color:#aeb8c8; }}
.triptych {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
.pair {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.powerpoint {{ margin-top:20px; padding-top:14px; border-top:1px solid #283040; }}
figure {{ margin:0; }}
img {{ width:100%; height:clamp(180px,28vw,440px); object-fit:contain; background:#05070a; border-radius:10px; }}
figcaption {{ font-size:.82rem; line-height:1.45; margin-top:8px; }}
.deltas {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
.deltas span {{ background:#202938; border-radius:999px; padding:6px 10px; font-variant-numeric:tabular-nums; }}
.table-wrap {{ overflow-x:auto; margin-top:14px; }}
table {{ width:100%; border-collapse:collapse; font-size:.82rem; font-variant-numeric:tabular-nums; }}
th,td {{ padding:7px 9px; text-align:right; border-bottom:1px solid #283040; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
.good {{ color:#6de3a5; }} .bad {{ color:#ff8d91; }}
@media(max-width:850px) {{ .triptych,.pair {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<p class="lede">SplatThis research report</p>
<h1>Full-corpus MVPs</h1>
<p class="lede">All {summary['corpus_images']} corpus images, scored over the full deployed frame.
SVG panels below embed the real SVG artifacts; they are not PNG proxy previews.
SSIM is standard windowed SSIM in display sRGB. MS-SSIM is a three-scale
windowed luma score. LPIPS, PSNR, edge distance, worst 64×64 tile error,
artifact bytes, and per-image deltas are retained in <code>summary.json</code>.
The deployed baselines were optimized with MLX; the Top-K experiment ran with
Torch on MPS. The completed Top-K run is a low-budget screen: 1,024 splats is
below the deployed corpus median of 1,389, and its 40 total direct-arm
iterations are far shorter than the deployed schedule. Top-K tables compare
governing Chromium SVG captures at every completed seed, while the visual
triptychs show seed 0.</p>
{''.join(topk_sections)}
<section>
  <h2>Mixed native paths on deployed MLX SVG baselines</h2>
  <p class="summary">Accepted {len([r for r in mixed if r['candidate'] is not None])}/{len(mixed)} images at the +0.001 SSIM gate.
  Among accepted candidates: median ΔSSIM {_fmt(mixed_aggregate.get('ssim_srgb_median_delta'), 5)} ·
  median ΔLPIPS {_fmt(mixed_aggregate.get('lpips_median_delta'), 5)}. In actual
  Microsoft PowerPoint: median ΔSSIM {_fmt(powerpoint_aggregate.get('ssim_srgb_median_delta'), 5)}
  ({powerpoint_aggregate.get('ssim_srgb_improved', 0)}/{powerpoint_aggregate.get('n', 0)} improved) ·
  median ΔLPIPS {_fmt(powerpoint_aggregate.get('lpips_median_delta'), 5)}
  ({powerpoint_aggregate.get('lpips_improved', 0)}/{powerpoint_aggregate.get('n', 0)} improved).
  PowerPoint comparisons use clean full-screen captures from PowerPoint's own
  slideshow renderer, never Quick Look, LibreOffice, or an internal proxy.</p>
  {''.join(mixed_cards)}
</section>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    print(f"wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only")
    parser.add_argument("--topk", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--capture-mixed-pptx", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--budgets", default="256,1024")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.topk:
        run_topk(
            args.corpus_root,
            args.output_root,
            only=args.only,
            budgets=_csv(args.budgets, int),
            seeds=_csv(args.seeds, int),
            iterations=args.iterations,
            force=args.force,
        )
    if args.mixed:
        run_mixed(
            args.corpus_root,
            args.output_root,
            only=args.only,
            force=args.force,
        )
    if args.capture_mixed_pptx:
        capture_mixed_powerpoint(
            args.corpus_root,
            args.output_root,
            only=args.only,
            force=args.force,
        )
    summary = None
    if args.summarize or args.html:
        summary = build_summary(args.corpus_root, args.output_root)
        print(f"wrote {args.output_root / 'summary.json'}")
    if args.html:
        generate_html(
            summary or build_summary(args.corpus_root, args.output_root),
            args.output_root,
            args.output_root / "index.html",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
