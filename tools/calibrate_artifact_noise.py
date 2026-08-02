#!/usr/bin/env python3
"""Repeat deployed-artifact renders and derive target-specific noise floors.

Capture is explicit because the PowerPoint target controls the desktop viewer.
Without ``--capture`` the command only re-analyzes an existing observations
file.

Examples:

    PYTHONPATH=src python tools/calibrate_artifact_noise.py \
      --capture --targets svg --only chameleon

    PYTHONPATH=src python tools/calibrate_artifact_noise.py \
      --capture --targets pixel-runtime,svg,pptx --repeats 5
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from splatthis.artifact_gates import (
    ARTIFACT_TARGETS,
    ArtifactGateCalibration,
    calibrate_artifact_observations,
)
from splatthis.browser_capture import (
    get_shared_svg_renderer,
    render_css_html_in_browser_to_linear_rgb,
    render_svg_in_browser_to_linear_rgb,
)
from splatthis.fidelity.metrics import compute_fidelity_metrics
from splatthis.io import linear_to_srgb, load_png

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "result" / "corpus"
DEFAULT_OUTPUT = REPO / "tmp" / "artifact-noise"
ArtifactSpec = Tuple[str, str, Path, str]
ObservationKey = Tuple[str, str, int]
FloatArray = NDArray[np.float32]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        records.append(value)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    temporary.replace(path)


def _latest_seed_zero_runs(root: Path) -> Dict[Tuple[str, str], dict[str, Any]]:
    selected: Dict[Tuple[str, str], dict[str, Any]] = {}
    for record in _json_lines(root / "results.jsonl"):
        if record.get("seed") != 0 or record.get("returncode", 0) != 0:
            continue
        image = record.get("image")
        target = record.get("format")
        if isinstance(image, str) and isinstance(target, str):
            selected[(image, target)] = record
    return selected


def _artifact_path(
    root: Path,
    image: str,
    target: str,
    run: Optional[Mapping[str, Any]],
) -> Path:
    if target == "css":
        # CSS artifacts are the committed corpus-gallery builds, emitted by
        # the shipped emitter from the same populations the fidelity
        # registry measured (tools/build_corpus_gallery.py --emit).
        return REPO / "docs" / "corpus" / "css" / f"{image}.html"
    recorded = run.get("output_path") if run is not None else None
    if isinstance(recorded, str) and recorded:
        return root / recorded
    extension = {"pixel-runtime": "html", "svg": "svg", "pptx": "pptx"}[target]
    artifact_target = "canvas" if target == "pixel-runtime" else target
    return root / "runs" / f"{image}_{artifact_target}_s0.{extension}"


def _grid_rois(
    height: int, width: int, tile: int = 64
) -> list[tuple[int, int, int, int]]:
    return [
        (y, x, min(y + tile, height), min(x + tile, width))
        for y in range(0, height, tile)
        for x in range(0, width, tile)
    ]


def _finite_metrics(metrics: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    output: Dict[str, Optional[float]] = {}
    for name, value in metrics.items():
        if name in {"splat_count", "file_size_bytes", "render_method"}:
            continue
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            output[name] = float(value)
        elif value is None or isinstance(value, (int, float)):
            output[name] = None
    return output


def _score(
    source_linear: FloatArray,
    rendered_linear: FloatArray,
    *,
    renderer: str,
) -> Dict[str, Optional[float]]:
    height, width = source_linear.shape[:2]
    metrics = compute_fidelity_metrics(
        source_linear,
        rendered_linear,
        fixed_rois=_grid_rois(height, width),
        render_method=renderer,
    )
    return _finite_metrics(metrics.as_dict())


def _load_rendered_png(path: Path) -> FloatArray:
    return np.asarray(load_png(str(path))[..., :3], dtype=np.float32)


def _save_linear_png(path: Path, linear_rgb: FloatArray) -> None:
    srgb = np.clip(linear_to_srgb(linear_rgb), 0.0, 1.0)
    pixels = np.rint(srgb * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def _powerpoint_version() -> str:
    result = subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "Microsoft PowerPoint" to return version',
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _observation(
    *,
    target: str,
    image: str,
    artifact: Path,
    artifact_id: str,
    source: Path,
    repeat: int,
    renderer: str,
    renderer_version: str,
    capture_method: str,
    capture_path: Path,
    render_time_ms: float,
    metrics: Mapping[str, Optional[float]],
) -> Dict[str, Any]:
    return {
        "schema": "splatthis.artifact-observation/1",
        "target": target,
        "image": image,
        "artifact_id": artifact_id,
        "artifact": _display_path(artifact),
        "artifact_sha256": _sha256(artifact),
        "source": _display_path(source),
        "repeat": repeat,
        "renderer": renderer,
        "renderer_version": renderer_version,
        "capture_method": capture_method,
        "capture": _display_path(capture_path),
        "capture_sha256": _sha256(capture_path),
        "render_time_ms": float(render_time_ms),
        "metrics": dict(metrics),
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO)
        rendered = str(relative)
        return f"./{rendered}" if relative.parts[:1] == ("tmp",) else rendered
    except ValueError:
        return str(resolved)


def _capture_svg(
    *,
    image: str,
    source: Path,
    artifact: Path,
    artifact_id: str,
    output_dir: Path,
    repeats: int,
    force: bool,
) -> list[dict[str, Any]]:
    source_linear = np.asarray(load_png(str(source))[..., :3], dtype=np.float32)
    height, width = source_linear.shape[:2]
    records = []
    renderer_version = get_shared_svg_renderer().browser_version
    renderer = f"playwright-chromium/{renderer_version}"
    renderer_cache_key = f"chromium-{renderer_version.replace('.', '-')}"
    for repeat in range(repeats):
        capture_path = (
            output_dir
            / "captures"
            / "svg"
            / renderer_cache_key
            / image
            / f"{repeat:03d}.png"
        )
        started = time.perf_counter()
        if force or not capture_path.exists():
            rendered, actual_renderer = render_svg_in_browser_to_linear_rgb(
                str(artifact), width, height
            )
            if actual_renderer != renderer:
                raise RuntimeError(
                    f"SVG renderer changed during capture: {actual_renderer}"
                )
            _save_linear_png(capture_path, rendered)
            rendered = _load_rendered_png(capture_path)
        else:
            rendered = _load_rendered_png(capture_path)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        records.append(
            _observation(
                target="svg",
                image=image,
                artifact=artifact,
                artifact_id=artifact_id,
                source=source,
                repeat=repeat,
                renderer=renderer,
                renderer_version=renderer_version,
                capture_method="Playwright Chromium native-dimension PNG screenshot",
                capture_path=capture_path,
                render_time_ms=elapsed_ms,
                metrics=_score(source_linear, rendered, renderer=renderer),
            )
        )
    return records


def _capture_css(
    *,
    image: str,
    source: Path,
    artifact: Path,
    artifact_id: str,
    output_dir: Path,
    repeats: int,
    force: bool,
) -> list[dict[str, Any]]:
    source_linear = np.asarray(load_png(str(source))[..., :3], dtype=np.float32)
    height, width = source_linear.shape[:2]
    records = []
    renderer_version = get_shared_svg_renderer().browser_version
    renderer = f"playwright-chromium/{renderer_version}"
    renderer_cache_key = f"chromium-{renderer_version.replace('.', '-')}"
    for repeat in range(repeats):
        capture_path = (
            output_dir
            / "captures"
            / "css"
            / renderer_cache_key
            / image
            / f"{repeat:03d}.png"
        )
        started = time.perf_counter()
        if force or not capture_path.exists():
            rendered, actual_renderer = render_css_html_in_browser_to_linear_rgb(
                str(artifact), width, height
            )
            if actual_renderer != renderer:
                raise RuntimeError(
                    f"CSS renderer changed during capture: {actual_renderer}"
                )
            _save_linear_png(capture_path, rendered)
            rendered = _load_rendered_png(capture_path)
        else:
            rendered = _load_rendered_png(capture_path)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        records.append(
            _observation(
                target="css",
                image=image,
                artifact=artifact,
                artifact_id=artifact_id,
                source=source,
                repeat=repeat,
                renderer=renderer,
                renderer_version=renderer_version,
                capture_method=(
                    "Playwright Chromium native-dimension PNG screenshot "
                    "of the scriptless CSS build"
                ),
                capture_path=capture_path,
                render_time_ms=elapsed_ms,
                metrics=_score(source_linear, rendered, renderer=renderer),
            )
        )
    return records


def _capture_canvas(
    *,
    image: str,
    source: Path,
    artifact: Path,
    artifact_id: str,
    output_dir: Path,
    repeats: int,
    force: bool,
    capture_python: Path,
    browser_executable: Path,
) -> list[dict[str, Any]]:
    sample_dir = output_dir / "captures" / "pixel-runtime" / image
    expected = [sample_dir / f"repeat-{repeat:03d}.png" for repeat in range(repeats)]
    metadata_path = sample_dir / "capture.json"
    if (
        force
        or not metadata_path.exists()
        or not all(path.exists() for path in expected)
    ):
        command = [
            str(capture_python),
            str(REPO / "tools" / "capture_canvas_html.py"),
            str(artifact),
            str(sample_dir / "last.png"),
            "--browser-executable",
            str(browser_executable),
            "--repeats",
            str(repeats),
            "--samples-dir",
            str(sample_dir),
        ]
        result = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=max(180, repeats * 120),
        )
        (sample_dir / "capture.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout.strip())
        try:
            metadata = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError("canvas capture emitted invalid metadata") from exc
        _write_json(metadata_path, metadata)
    else:
        metadata = json.loads(metadata_path.read_text())

    source_linear = np.asarray(load_png(str(source))[..., :3], dtype=np.float32)
    render_samples = metadata.get("render_ms_samples", [])
    records = []
    for repeat, capture_path in enumerate(expected):
        rendered = _load_rendered_png(capture_path)
        render_ms = (
            float(render_samples[repeat]) if repeat < len(render_samples) else 0.0
        )
        records.append(
            _observation(
                target="pixel-runtime",
                image=image,
                artifact=artifact,
                artifact_id=artifact_id,
                source=source,
                repeat=repeat,
                renderer="Google Chrome pixel-runtime canvas.toDataURL",
                renderer_version=str(metadata.get("browser", "unknown")),
                capture_method="browser canvas.toDataURL",
                capture_path=capture_path,
                render_time_ms=render_ms,
                metrics=_score(
                    source_linear,
                    rendered,
                    renderer="Google Chrome pixel-runtime canvas.toDataURL",
                ),
            )
        )
    return records


def _capture_pptx(
    *,
    image: str,
    source: Path,
    artifact: Path,
    artifact_id: str,
    output_dir: Path,
    repeats: int,
    force: bool,
) -> list[dict[str, Any]]:
    capture_powerpoint = getattr(
        importlib.import_module("full_corpus_mvp"),
        "_capture_powerpoint_slideshow",
    )

    source_linear = np.asarray(load_png(str(source))[..., :3], dtype=np.float32)
    height, width = source_linear.shape[:2]
    version = _powerpoint_version()
    records = []
    for repeat in range(repeats):
        capture_path = output_dir / "captures" / "pptx" / image / f"{repeat:03d}.png"
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if force or not capture_path.exists():
            returncode, message = capture_powerpoint(
                artifact, capture_path, width, height
            )
            (capture_path.with_suffix(".log")).write_text(message)
            if returncode or not capture_path.exists():
                raise RuntimeError(message.strip())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rendered = _load_rendered_png(capture_path)
        records.append(
            _observation(
                target="pptx",
                image=image,
                artifact=artifact,
                artifact_id=artifact_id,
                source=source,
                repeat=repeat,
                renderer="Microsoft PowerPoint slideshow",
                renderer_version=version,
                capture_method="PowerPoint slideshow window capture",
                capture_path=capture_path,
                render_time_ms=elapsed_ms,
                metrics=_score(
                    source_linear,
                    rendered,
                    renderer="Microsoft PowerPoint slideshow",
                ),
            )
        )
    return records


def _report_markdown(
    calibration: ArtifactGateCalibration,
    *,
    errors: Sequence[Mapping[str, str]],
    selection: Sequence[str],
) -> str:
    lines = [
        "# Artifact gate calibration",
        "",
        f"- Required repeats: {calibration.required_repeats}",
        f"- Noise multiplier: {calibration.noise_multiplier:.2f}",
        f"- Selection: {', '.join(selection) if selection else 'full corpus'}",
        f"- Platform: {platform.platform()}",
        "",
        "| Target | Complete artifacts | Expected | Observations | Complete |",
        "|---|---:|---:|---:|---|",
    ]
    for target in ARTIFACT_TARGETS:
        result = calibration.targets[target]
        lines.append(
            f"| {target} | {result.complete_artifact_count} | "
            f"{result.artifact_count} | {result.observation_count} | "
            f"{'yes' if result.complete else 'no'} |"
        )
    for target in ARTIFACT_TARGETS:
        result = calibration.targets[target]
        lines.extend(
            [
                "",
                f"## {target}",
                "",
                "| Metric | Median span | p95 span | Max span | Recommended minimum delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        if not result.metrics:
            lines.append("| No repeated observations | — | — | — | — |")
        for metric, estimate in sorted(result.metrics.items()):
            lines.append(
                f"| {metric} | {estimate.median_span:.8g} | "
                f"{estimate.p95_span:.8g} | {estimate.max_span:.8g} | "
                f"{estimate.recommended_min_delta:.8g} |"
            )
    if errors:
        lines.extend(["", "## Capture errors", ""])
        for error in errors:
            lines.append(f"- `{error['target']}/{error['image']}`: {error['error']}")
    lines.extend(
        [
            "",
            "A zero span means the repeated pixel metrics were deterministic in",
            "this run. It does not calibrate algorithmic or cross-version variance.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", default="pixel-runtime,svg,pptx")
    parser.add_argument("--only", help="comma-separated corpus image names")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--noise-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--merge-observations",
        type=Path,
        action="append",
        default=[],
        help="include another JSONL capture session in the noise analysis",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="perform renderer captures before analyzing observations",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace cached repeat captures for the selected artifacts",
    )
    parser.add_argument(
        "--canvas-capture-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter that provides Playwright",
    )
    parser.add_argument(
        "--browser-executable",
        type=Path,
        default=Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    )
    return parser


def _build_artifact_specs(
    *,
    corpus_root: Path,
    targets: Sequence[str],
    selected_names: Sequence[str],
) -> Tuple[Dict[str, list[str]], list[ArtifactSpec]]:
    latest_runs = _latest_seed_zero_runs(corpus_root)
    expected_artifacts: Dict[str, list[str]] = {
        target: [] for target in ARTIFACT_TARGETS
    }
    specs = []
    for target in targets:
        for image in selected_names:
            run_target = "canvas" if target == "pixel-runtime" else target
            artifact = _artifact_path(
                corpus_root, image, target, latest_runs.get((image, run_target))
            )
            artifact_hash = _sha256(artifact) if artifact.exists() else "missing"
            artifact_id = f"{target}:{image}:{artifact_hash[:16]}"
            expected_artifacts[target].append(artifact_id)
            specs.append((target, image, artifact, artifact_id))
    return expected_artifacts, specs


def _capture_one(
    *,
    target: str,
    common: Mapping[str, Any],
    capture_python: Path,
    browser_executable: Path,
) -> list[dict[str, Any]]:
    if target == "svg":
        return _capture_svg(**common)
    if target == "css":
        return _capture_css(**common)
    if target == "pixel-runtime":
        return _capture_canvas(
            **common,
            capture_python=capture_python,
            browser_executable=browser_executable,
        )
    return _capture_pptx(**common)


def _capture_specs(
    *,
    specs: Sequence[ArtifactSpec],
    corpus_root: Path,
    images: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    repeats: int,
    force: bool,
    capture_python: Path,
    browser_executable: Path,
    observation_index: Dict[ObservationKey, dict[str, Any]],
    observations_path: Path,
) -> list[dict[str, str]]:
    errors = []
    for index, (target, image, artifact, artifact_id) in enumerate(specs, 1):
        source = corpus_root / str(images[image]["path"])
        print(
            f"[{index}/{len(specs)}] {target}/{image} ... ",
            end="",
            flush=True,
        )
        if not artifact.exists():
            message = f"missing artifact: {artifact}"
            errors.append({"target": target, "image": image, "error": message})
            print("MISSING")
            continue
        try:
            captured = _capture_one(
                target=target,
                common={
                    "image": image,
                    "source": source,
                    "artifact": artifact,
                    "artifact_id": artifact_id,
                    "output_dir": output_dir,
                    "repeats": repeats,
                    "force": force,
                },
                capture_python=capture_python,
                browser_executable=browser_executable,
            )
            for record in captured:
                key = (
                    str(record["target"]),
                    str(record["artifact_id"]),
                    int(record["repeat"]),
                )
                observation_index[key] = record
            observations = [observation_index[key] for key in sorted(observation_index)]
            _write_jsonl(observations_path, observations)
            print("done")
        except Exception as exc:
            errors.append({"target": target, "image": image, "error": str(exc)})
            print("FAILED")
    return errors


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    targets = _parse_csv(args.targets)
    invalid_targets = sorted(set(targets) - set(ARTIFACT_TARGETS))
    if invalid_targets:
        parser.error(f"unsupported targets: {', '.join(invalid_targets)}")

    corpus_root = args.corpus_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((corpus_root / "corpus.json").read_text())
    images = corpus["images"]
    selected_names = _parse_csv(args.only) if args.only else list(images)
    unknown_images = sorted(set(selected_names) - set(images))
    if unknown_images:
        parser.error(f"unknown corpus images: {', '.join(unknown_images)}")

    expected_artifacts, specs = _build_artifact_specs(
        corpus_root=corpus_root,
        targets=targets,
        selected_names=selected_names,
    )
    observations_path = output_dir / "observations.jsonl"
    observations = _json_lines(observations_path)
    observation_index: Dict[ObservationKey, dict[str, Any]] = {
        (
            str(record.get("target")),
            str(record.get("artifact_id")),
            int(record.get("repeat", -1)),
        ): record
        for record in observations
    }
    errors = (
        _capture_specs(
            specs=specs,
            corpus_root=corpus_root,
            images=images,
            output_dir=output_dir,
            repeats=args.repeats,
            force=args.force,
            capture_python=args.canvas_capture_python,
            browser_executable=args.browser_executable,
            observation_index=observation_index,
            observations_path=observations_path,
        )
        if args.capture
        else []
    )

    observations = [observation_index[key] for key in sorted(observation_index)]
    merged_observations = list(observations)
    for merge_path in args.merge_observations:
        merged_observations.extend(_json_lines(merge_path.resolve()))
    calibration = calibrate_artifact_observations(
        merged_observations,
        required_repeats=args.repeats,
        noise_multiplier=args.noise_multiplier,
        expected_targets=ARTIFACT_TARGETS,
        expected_artifacts=expected_artifacts,
    )
    _write_json(output_dir / "artifact-gates.json", calibration.as_dict())
    summary = {
        "schema": "splatthis.artifact-calibration-run/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_root": _display_path(corpus_root),
        "selection": selected_names,
        "targets_requested": targets,
        "capture_performed": bool(args.capture),
        "observations": _display_path(observations_path),
        "merged_observations": [
            _display_path(path.resolve()) for path in args.merge_observations
        ],
        "errors": errors,
        "calibration": calibration.as_dict(),
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        _report_markdown(
            calibration,
            errors=errors,
            selection=[] if args.only is None else selected_names,
        )
    )
    print(f"wrote {_display_path(output_dir / 'artifact-gates.json')}")
    print(f"wrote {_display_path(output_dir / 'report.md')}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
