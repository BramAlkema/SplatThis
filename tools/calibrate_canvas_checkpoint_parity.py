#!/usr/bin/env python3
"""Calibrate historical pixel-runtime scores against Chrome pixel buffers.

The tool reconstructs every available stage HTML from its canonical raw splat
artifact, captures the canvas with real Chrome, and compares both the
source-relative score and the model/browser pixel buffers. It never retrains a
checkpoint, so model-to-browser bias stays separate from optimizer variance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from splatthis.canvas_parity import (  # noqa: E402
    CanvasParityObservation,
    summarize_canvas_parity,
)
from splatthis.io import (  # noqa: E402
    atomic_write_text,
    compute_quality_metrics,
    load_png,
    load_splats_json,
)
from splatthis.pixel_runtime import generate_pixel_runtime_html  # noqa: E402
from splatthis.renderer import (  # noqa: E402
    render_pixel_runtime_numpy,
    render_splats_numpy,
)

DEFAULT_CORPUS = REPO / "result" / "corpus"
DEFAULT_OUTPUT = REPO / "tmp" / "canvas-checkpoint-parity"
DEFAULT_CAPTURE_PYTHON = Path(sys.executable)
DEFAULT_BROWSER = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


@dataclass(frozen=True)
class CheckpointArtifact:
    image: str
    label: str
    raw_path: Path
    model_ssim_srgb: float
    model_psnr_srgb: float
    splat_count: int


def _json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _checkpoint_artifacts(image: str, manifest_path: Path) -> list[CheckpointArtifact]:
    manifest = json.loads(manifest_path.read_text())
    artifacts: list[CheckpointArtifact] = []
    for stage in manifest.get("stages", []):
        if (
            stage.get("deployed_ssim_srgb") is None
            or stage.get("deployed_psnr_srgb") is None
        ):
            continue
        stage_type = stage.get("stage_type")
        if stage_type == "residual_detail":
            pass_index = int(stage.get("residual_pass", 1))
            label = f"residual-{pass_index}"
        else:
            stage_index = int(stage.get("stage", 0))
            if stage_index <= 0:
                continue
            label = f"iter-{stage_index}"
        raw_path = manifest_path.parent / f"{label}.raw.json"
        if not raw_path.exists():
            continue
        artifacts.append(
            CheckpointArtifact(
                image=image,
                label=label,
                raw_path=raw_path,
                model_ssim_srgb=float(stage["deployed_ssim_srgb"]),
                model_psnr_srgb=float(stage["deployed_psnr_srgb"]),
                splat_count=int(stage.get("splat_count", 0)),
            )
        )
    return artifacts


def _latest_curves(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    selected_images: set[str],
) -> dict[str, tuple[Path, list[CheckpointArtifact]]]:
    candidates: dict[str, list[Path]] = {}
    for record in reversed(records):
        image = record.get("image")
        artifact_path = record.get("artifacts_path")
        if (
            record.get("format") not in {"canvas", "pixel-runtime"}
            or record.get("seed") != 0
            or not isinstance(image, str)
            or image not in selected_images
            or not isinstance(artifact_path, str)
        ):
            continue
        manifest_path = root / artifact_path / "run_manifest.json"
        paths = candidates.setdefault(image, [])
        if manifest_path.exists() and manifest_path not in paths:
            paths.append(manifest_path)

    curves: dict[str, tuple[Path, list[CheckpointArtifact]]] = {}
    for image, manifest_paths in candidates.items():
        for manifest_path in manifest_paths:
            checkpoints = _checkpoint_artifacts(image, manifest_path)
            if len(checkpoints) >= 2:
                curves[image] = (manifest_path, checkpoints)
                break
    return curves


def _capture_canvas(
    *,
    capture_python: Path,
    browser_executable: Path,
    html_path: Path,
    capture_path: Path,
    repeats: int,
) -> dict[str, Any]:
    command = [
        str(capture_python),
        str(REPO / "tools" / "capture_canvas_html.py"),
        str(html_path),
        str(capture_path),
        "--browser-executable",
        str(browser_executable),
        "--repeats",
        str(repeats),
    ]
    result = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=180,
    )
    if result.returncode != 0 or not capture_path.exists():
        raise RuntimeError(result.stdout.strip() or "Chrome capture failed")
    try:
        metadata = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("capture returned invalid metadata") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("capture metadata must be a JSON object")
    return metadata


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    """Return a stable repo-relative path when possible."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def _absolute_preserving_symlink(path: Path) -> Path:
    """Make a command path absolute without escaping its virtual environment."""

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def _validate_capture_environment(capture_python: Path) -> None:
    """Fail once, before the corpus loop, when Playwright is unavailable."""

    result = subprocess.run(
        [
            str(capture_python),
            "-c",
            "from playwright.sync_api import sync_playwright",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stdout.strip() or "Playwright import failed"
        raise ValueError(f"capture Python cannot import Playwright: {detail}")


def _write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _render_report(summary: Mapping[str, Any]) -> str:
    calibration = summary["calibration"]
    continuous = summary.get("continuous_model_baseline")
    lines = [
        "# Pixel-runtime checkpoint model-to-Chrome parity",
        "",
        "Every row is one unchanged full-frame stage artifact. No checkpoint",
        "was retrained for this calibration.",
        "",
        f"- Images: {calibration['image_count']}",
        f"- Checkpoints: {calibration['observation_count']}",
        "- Recommended SSIM safety margin: "
        f"{calibration['recommended_ssim_safety_margin']:.6f}",
        "- Recommended PSNR safety margin: "
        f"{calibration['recommended_psnr_safety_margin']:.3f} dB",
        "- Worst model SSIM overstatement: "
        f"{calibration['ssim_overstatement']['max']:+.6f}",
        "- Minimum direct model/browser parity SSIM: "
        f"{calibration['pixel_parity_ssim_srgb']['min']:.8f}",
        "",
        "| Image | Checkpoint | Model SSIM | Chrome SSIM | Overstatement | Pixel parity SSIM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if continuous:
        lines[10:10] = [
            "- Previous continuous-model maximum SSIM overstatement: "
            f"{continuous['ssim_overstatement']['max']:+.6f}",
            "- Previous continuous-model minimum parity SSIM: "
            f"{continuous['pixel_parity_ssim_srgb']['min']:.8f}",
        ]
    for item in summary["observations"]:
        lines.append(
            f"| {item['image']} | {item['checkpoint']} | "
            f"{item['model_ssim_srgb']:.6f} | "
            f"{item['browser_ssim_srgb']:.6f} | "
            f"{item['ssim_overstatement']:+.6f} | "
            f"{item['pixel_parity_ssim_srgb']:.8f} |"
        )
    missing = summary.get("missing_images", [])
    failures = summary.get("failures", [])
    if missing:
        lines.extend(["", "Missing historical curves: " + ", ".join(missing)])
    if failures:
        lines.extend(["", f"Capture failures: {len(failures)}"])
    lines.append("")
    return "\n".join(lines)


def _parse_selected(value: Optional[str], known: set[str]) -> set[str]:
    if value is None:
        return set(known)
    selected = {part.strip() for part in value.split(",") if part.strip()}
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError(f"unknown corpus images: {', '.join(unknown)}")
    return selected


def main() -> int:  # noqa: C901 - orchestration is intentionally linear
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--canvas-capture-python", type=Path, default=DEFAULT_CAPTURE_PYTHON
    )
    parser.add_argument("--browser-executable", type=Path, default=DEFAULT_BROWSER)
    parser.add_argument("--only", help="comma-separated corpus image names")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    root = args.corpus_root.resolve()
    output_dir = args.output_dir.resolve()
    capture_python = _absolute_preserving_symlink(args.canvas_capture_python)
    browser_executable = args.browser_executable.resolve()
    if not (root / "corpus.json").is_file():
        parser.error(f"corpus metadata not found: {root / 'corpus.json'}")
    if not capture_python.is_file():
        parser.error(f"capture Python not found: {capture_python}")
    if not browser_executable.is_file():
        parser.error(f"browser executable not found: {browser_executable}")
    try:
        _validate_capture_environment(capture_python)
    except ValueError as exc:
        parser.error(str(exc))
    corpus = json.loads((root / "corpus.json").read_text())
    image_meta = corpus["images"]
    selected_images = _parse_selected(args.only, set(image_meta))
    records = _json_lines(root / "results.jsonl")
    curves = _latest_curves(root, records, selected_images)
    missing_images = sorted(selected_images - set(curves))

    observations: list[CanvasParityObservation] = []
    continuous_observations: list[CanvasParityObservation] = []
    details: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    total = sum(len(checkpoints) for _, checkpoints in curves.values())
    completed = 0
    for image, (manifest_path, checkpoints) in sorted(curves.items()):
        manifest = json.loads(manifest_path.read_text())
        config = manifest.get("config", {})
        width, height = config.get("resolved_target_size", image_meta[image]["size"])
        training_target = str(config.get("training_export_target", "canvas"))
        compositing_space = str(config.get("compositing_space", "linear"))
        if training_target in {"svg", "pptx-softedge"}:
            compositing_space = "srgb"
        background = np.asarray(
            config.get("background_linear_rgb", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        source_path = root / image_meta[image]["path"]
        target = load_png(str(source_path))[..., :3]
        for checkpoint in checkpoints:
            completed += 1
            checkpoint_dir = output_dir / "captures" / image
            html_path = checkpoint_dir / f"{checkpoint.label}.html"
            capture_path = checkpoint_dir / f"{checkpoint.label}.png"
            metadata_path = checkpoint_dir / f"{checkpoint.label}.capture.json"
            try:
                splats = load_splats_json(str(checkpoint.raw_path))
                html = generate_pixel_runtime_html(
                    splats,
                    int(width),
                    int(height),
                    background_linear_rgb=background,
                    title=f"{image} {checkpoint.label} parity",
                    compositing_space=compositing_space,
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_text(html_path, html)
                html_sha256 = _sha256(html_path)
                metadata: dict[str, Any] = {}
                if metadata_path.exists():
                    cached = json.loads(metadata_path.read_text())
                    if isinstance(cached, dict):
                        metadata = cached
                cache_valid = (
                    capture_path.exists() and metadata.get("html_sha256") == html_sha256
                )
                if args.force or not cache_valid:
                    metadata = _capture_canvas(
                        capture_python=capture_python,
                        browser_executable=browser_executable,
                        html_path=html_path,
                        capture_path=capture_path,
                        repeats=args.repeats,
                    )
                    metadata["html_sha256"] = html_sha256
                    _write_json(metadata_path, metadata)

                browser = load_png(str(capture_path))[..., :3]
                continuous_model = render_splats_numpy(
                    splats,
                    width=int(width),
                    height=int(height),
                    background_linear_rgb=background,
                    compositing_space=compositing_space,
                )
                model = render_pixel_runtime_numpy(
                    splats,
                    width=int(width),
                    height=int(height),
                    background_linear_rgb=background,
                    compositing_space=compositing_space,
                )
                if (
                    target.shape != browser.shape
                    or model.shape != browser.shape
                    or continuous_model.shape != browser.shape
                ):
                    raise ValueError(
                        f"shape mismatch target={target.shape} model={model.shape} "
                        f"browser={browser.shape}"
                    )
                model_quality = compute_quality_metrics(target, model)
                continuous_quality = compute_quality_metrics(target, continuous_model)
                browser_quality = compute_quality_metrics(target, browser)
                parity_quality = compute_quality_metrics(model, browser)
                continuous_parity_quality = compute_quality_metrics(
                    continuous_model, browser
                )
                absolute = np.abs(model - browser)
                continuous_absolute = np.abs(continuous_model - browser)
                observation = CanvasParityObservation(
                    image=image,
                    checkpoint=checkpoint.label,
                    model_ssim_srgb=float(model_quality["ssim_srgb"]),
                    browser_ssim_srgb=float(browser_quality["ssim_srgb"]),
                    model_psnr_srgb=float(model_quality["psnr_srgb"]),
                    browser_psnr_srgb=float(browser_quality["psnr_srgb"]),
                    pixel_parity_ssim_srgb=float(parity_quality["ssim_srgb"]),
                    pixel_mae_linear=float(np.mean(absolute)),
                    pixel_max_abs_linear=float(np.max(absolute)),
                )
                observations.append(observation)
                continuous_observation = CanvasParityObservation(
                    image=image,
                    checkpoint=checkpoint.label,
                    model_ssim_srgb=float(continuous_quality["ssim_srgb"]),
                    browser_ssim_srgb=float(browser_quality["ssim_srgb"]),
                    model_psnr_srgb=float(continuous_quality["psnr_srgb"]),
                    browser_psnr_srgb=float(browser_quality["psnr_srgb"]),
                    pixel_parity_ssim_srgb=float(
                        continuous_parity_quality["ssim_srgb"]
                    ),
                    pixel_mae_linear=float(np.mean(continuous_absolute)),
                    pixel_max_abs_linear=float(np.max(continuous_absolute)),
                )
                continuous_observations.append(continuous_observation)
                details.append(
                    {
                        **observation.as_dict(),
                        "continuous_model_ssim_srgb": float(
                            continuous_quality["ssim_srgb"]
                        ),
                        "continuous_model_psnr_srgb": float(
                            continuous_quality["psnr_srgb"]
                        ),
                        "continuous_model_ssim_overstatement": (
                            continuous_observation.ssim_overstatement
                        ),
                        "continuous_model_pixel_parity_ssim_srgb": float(
                            continuous_parity_quality["ssim_srgb"]
                        ),
                        "stored_model_ssim_srgb": checkpoint.model_ssim_srgb,
                        "stored_model_psnr_srgb": checkpoint.model_psnr_srgb,
                        "stored_model_semantics": "historical-continuous-model",
                        "stored_vs_recomputed_ssim_delta": float(
                            checkpoint.model_ssim_srgb - observation.model_ssim_srgb
                        ),
                        "splat_count": len(splats),
                        "raw_sha256": _sha256(checkpoint.raw_path),
                        "html_sha256": html_sha256,
                        "capture_sha256": _sha256(capture_path),
                        "browser": metadata.get("browser"),
                        "browser_render_ms": metadata.get("render_ms"),
                        "manifest": _display_path(manifest_path),
                        "raw_artifact": _display_path(checkpoint.raw_path),
                    }
                )
                print(
                    f"[{completed}/{total}] {image} {checkpoint.label}: "
                    f"continuous {continuous_observation.model_ssim_srgb:.6f} -> "
                    f"deployed {observation.model_ssim_srgb:.6f} -> "
                    f"Chrome {observation.browser_ssim_srgb:.6f} "
                    f"({observation.ssim_overstatement:+.6f})"
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(
                    {
                        "image": image,
                        "checkpoint": checkpoint.label,
                        "error": str(exc),
                    }
                )
                print(f"[{completed}/{total}] {image} {checkpoint.label}: FAILED {exc}")

    if not observations:
        raise SystemExit("no checkpoint parity observations were captured")
    calibration = summarize_canvas_parity(observations)
    continuous_calibration = summarize_canvas_parity(continuous_observations)
    summary = {
        "schema": "splatthis.canvas-checkpoint-parity/2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(root),
        "selection": sorted(selected_images),
        "evidence": {
            "full_frame": True,
            "retrained": False,
            "capture_method": "Google Chrome canvas.toDataURL",
            "runtime_scorer": "canvas-image-data-byte-v1",
            "runtime_semantics": (
                "JavaScript double compute, Float32Array accumulation, "
                "8-bit sRGB ImageData"
            ),
            "pixel_exact_observations": sum(
                item.pixel_max_abs_linear == 0.0 for item in observations
            ),
            "optimizer_variance_in_scope": False,
        },
        "calibration": calibration,
        "continuous_model_baseline": continuous_calibration,
        "missing_images": missing_images,
        "failures": failures,
        "observations": details,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    atomic_write_text(output_dir / "report.md", _render_report(summary))
    print(f"wrote {output_dir / 'summary.json'}")
    print(f"wrote {output_dir / 'report.md'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
