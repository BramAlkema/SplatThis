#!/usr/bin/env python3
"""Run the native-PPTX painter-order MVP over the canonical corpus."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from splatthis.io import atomic_output_path, atomic_write_text

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "result" / "corpus"
DEFAULT_OUTPUT = REPO / "tmp" / "pptx-order-compositor-corpus"
RUNNER = Path(__file__).with_name("pptx_order_compositor_mvp.py")

METRIC_KEYS = (
    "ssim_srgb",
    "ms_ssim_luma",
    "psnr_srgb",
    "lpips",
    "delta_e_ok_mean",
    "delta_e_ok_p95",
    "edge_chamfer",
    "edge_gradient_l1",
    "worst_roi_error",
    "file_size_bytes",
    "render_time_sec",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", default=None, help="Comma-separated image names")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def _corpus_images(corpus_root: Path, only: str | None) -> list[dict[str, str]]:
    corpus = json.loads((corpus_root / "corpus.json").read_text(encoding="utf-8"))
    selected = None
    if only:
        selected = {name.strip() for name in only.split(",") if name.strip()}
    image_records = corpus["images"]
    if isinstance(image_records, dict):
        image_records = image_records.values()
    images = [
        image
        for image in image_records
        if selected is None or image["name"] in selected
    ]
    if selected is not None:
        found = {image["name"] for image in images}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"Unknown corpus image(s): {', '.join(missing)}")
    return images


def _run_one(
    corpus_root: Path,
    output_root: Path,
    image: Mapping[str, str],
    *,
    repeats: int,
    force: bool,
) -> dict[str, Any]:
    name = image["name"]
    output_dir = output_root / name
    result_path = output_dir / "results.json"
    if result_path.exists() and not force:
        report = json.loads(result_path.read_text(encoding="utf-8"))
        _ensure_selected_artifact(output_dir, report)
        atomic_write_text(result_path, json.dumps(report, indent=2))
        return report

    source = corpus_root / image["path"]
    run_dir = corpus_root / "runs" / f"{name}_pptx_s0_art"
    command = [
        sys.executable,
        str(RUNNER),
        str(source),
        "--splats-json",
        str(run_dir / "final.raw.json"),
        "--manifest",
        str(run_dir / "run_manifest.json"),
        "--output-dir",
        str(output_dir),
        "--repeats",
        str(repeats),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "run.log", completed.stdout)
    if completed.returncode or not result_path.exists():
        raise RuntimeError(
            f"{name} failed after {time.perf_counter() - started:.1f}s; "
            f"see {output_dir / 'run.log'}"
        )
    report = json.loads(result_path.read_text(encoding="utf-8"))
    _ensure_selected_artifact(output_dir, report)
    atomic_write_text(result_path, json.dumps(report, indent=2))
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _ensure_selected_artifact(output_dir: Path, report: dict[str, Any]) -> Path:
    """Materialize the recorded winner for fresh and resumed corpus runs."""

    selected_recipe = str(report["selection"]["selected_recipe"])
    source = output_dir / f"{selected_recipe}.pptx"
    destination = output_dir / "selected.pptx"
    if not source.exists():
        raise FileNotFoundError(f"Missing selected PPTX candidate: {source}")
    if not destination.exists() or _sha256(destination) != _sha256(source):
        with atomic_output_path(destination) as temporary:
            shutil.copyfile(source, temporary)
    report["selected_artifact"] = {
        "recipe": selected_recipe,
        "path": str(destination),
        "sha256": _sha256(destination),
    }
    report["schema"] = "splatthis.pptx-order-compositor-mvp/2"
    return destination


def _powerpoint_version() -> str:
    completed = subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "Microsoft PowerPoint" to return version',
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    version = completed.stdout.strip()
    return version if completed.returncode == 0 and version else "unknown"


def _median(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return None if not values else float(statistics.median(values))


def _result_by_recipe(report: Mapping[str, Any], recipe: str) -> dict[str, Any]:
    return next(
        dict(result) for result in report["results"] if result["recipe"] == recipe
    )


def _delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        before = baseline.get(key)
        after = candidate.get(key)
        output[key] = (
            None if before is None or after is None else float(after) - float(before)
        )
    return output


def _aggregate(
    images: Sequence[Mapping[str, str]], reports: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    per_image = []
    legacy_rows = []
    corrected_rows = []
    selected_rows = []
    for image, report in zip(images, reports):
        legacy = _result_by_recipe(report, "legacy-order")
        corrected = _result_by_recipe(report, "corrected-order")
        selected_recipe = str(report["selection"]["selected_recipe"])
        selected = legacy if selected_recipe == "legacy-order" else corrected
        decision = report["selection"]["decisions"][0]
        legacy_rows.append(legacy)
        corrected_rows.append(corrected)
        selected_rows.append(selected)
        per_image.append(
            {
                "image": image["name"],
                "content_class": image.get("content_class"),
                "size": [report["width"], report["height"]],
                "splats": report["splat_count"],
                "pptx_splat_style": report["pptx_splat_style"],
                "legacy_order": legacy,
                "corrected_order": corrected,
                "delta_corrected_minus_legacy": _delta(legacy, corrected),
                "selection": selected_recipe,
                "gate_failures": decision["failures"],
                "captures_successful": all(
                    len(item["captures"]) >= 1 for item in report["capture"].values()
                ),
            }
        )

    corrected_count = sum(
        record["selection"] == "corrected-order" for record in per_image
    )
    medians = {}
    for name, rows in (
        ("legacy_order", legacy_rows),
        ("corrected_order", corrected_rows),
        ("artifact_gate_selection", selected_rows),
    ):
        medians[name] = {key: _median(rows, key) for key in METRIC_KEYS}
    delta_rows = [record["delta_corrected_minus_legacy"] for record in per_image]
    delta_medians = {key: _median(delta_rows, key) for key in METRIC_KEYS}

    return {
        "schema": "splatthis.pptx-order-compositor-corpus/1",
        "date": "2026-07-31",
        "renderer": "Microsoft PowerPoint slideshow",
        "scope": {
            "images": len(per_image),
            "populations": len(per_image),
            "seed": 0,
            "capture_repeats": min(
                len(item["captures"])
                for report in reports
                for item in report["capture"].values()
            ),
        },
        "selection_counts": {
            "corrected-order": corrected_count,
            "legacy-order": len(per_image) - corrected_count,
        },
        "medians": medians,
        "median_delta_corrected_minus_legacy": delta_medians,
        "ssim_improvements_over_0_002": sum(
            float(record["delta_corrected_minus_legacy"]["ssim_srgb"]) > 0.002
            for record in per_image
        ),
        "ssim_regressions_over_0_002": sum(
            float(record["delta_corrected_minus_legacy"]["ssim_srgb"]) < -0.002
            for record in per_image
        ),
        "per_image": per_image,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_overview(
    output_root: Path,
    corpus_root: Path,
    images: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
) -> None:
    cards = []
    per_image = {record["image"]: record for record in summary["per_image"]}
    for image in images:
        name = image["name"]
        result = per_image[name]
        legacy = result["legacy_order"]
        corrected = result["corrected_order"]
        delta = result["delta_corrected_minus_legacy"]
        source = os.path.relpath(corpus_root / image["path"], output_root)
        cards.append(
            f"<section><h2>{html.escape(name)}</h2>"
            f'<p class="decision">selected <strong>{html.escape(result["selection"])}</strong> · '
            f'ΔSSIM {_fmt(delta["ssim_srgb"], 5)} · ΔLPIPS {_fmt(delta["lpips"], 5)} · '
            f'<a href="{html.escape(name)}/selected.pptx">native PPTX</a></p>'
            '<div class="triptych">'
            f'<figure><img src="{html.escape(source)}"><figcaption>source</figcaption></figure>'
            f'<figure><img src="{html.escape(name)}/captures/legacy-order-1.png">'
            f'<figcaption>legacy<br>SSIM {_fmt(legacy["ssim_srgb"], 5)} · '
            f'LPIPS {_fmt(legacy["lpips"], 5)}</figcaption></figure>'
            f'<figure><img src="{html.escape(name)}/captures/corrected-order-1.png">'
            f'<figcaption>corrected<br>SSIM {_fmt(corrected["ssim_srgb"], 5)} · '
            f'LPIPS {_fmt(corrected["lpips"], 5)}</figcaption></figure>'
            "</div></section>"
        )
    counts = summary["selection_counts"]
    medians = summary["medians"]
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PPTX painter-order corpus</title>
<style>
:root{{color-scheme:dark;font:15px system-ui,sans-serif}}
body{{margin:0;padding:28px;background:#101014;color:#eee}}
header,main{{max-width:1500px;margin:auto}}header{{margin-bottom:30px}}
.stats{{display:flex;gap:24px;flex-wrap:wrap;color:#bbb}}
section{{padding:18px;margin:0 0 24px;background:#19191e;border:1px solid #303038;border-radius:12px}}
h2{{margin:0}}.decision{{color:#bbb}}.triptych{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
figure{{margin:0}}img{{display:block;width:100%;height:auto;border-radius:6px}}
figcaption{{margin-top:8px;color:#aaa}}strong{{color:#fff}}
@media(max-width:800px){{.triptych{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Native PowerPoint painter-order corpus</h1>
<div class="stats"><span>{summary['scope']['images']} complete images</span>
<span>corrected selected {counts['corrected-order']}</span>
<span>legacy retained {counts['legacy-order']}</span>
<span>legacy median SSIM {_fmt(medians['legacy_order']['ssim_srgb'])}</span>
<span>corrected median SSIM {_fmt(medians['corrected_order']['ssim_srgb'])}</span>
<span>gated median SSIM {_fmt(medians['artifact_gate_selection']['ssim_srgb'])}</span></div>
</header><main>{''.join(cards)}</main></body></html>
"""
    atomic_write_text(output_root / "index.html", document)


def main() -> int:
    args = _parse_args()
    corpus_root = args.corpus_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    existing_summary_path = output_root / "results.json"
    existing_summary = None
    if existing_summary_path.exists() and not args.force:
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
    images = _corpus_images(corpus_root, args.only)
    reports = []
    started = time.perf_counter()
    for index, image in enumerate(images, 1):
        name = image["name"]
        item_started = time.perf_counter()
        print(f"[{index}/{len(images)}] {name}", flush=True)
        report = _run_one(
            corpus_root,
            output_root,
            image,
            repeats=args.repeats,
            force=args.force,
        )
        reports.append(report)
        selected = report["selection"]["selected_recipe"]
        legacy = _result_by_recipe(report, "legacy-order")
        corrected = _result_by_recipe(report, "corrected-order")
        print(
            f"  {selected}; ΔSSIM "
            f"{float(corrected['ssim_srgb']) - float(legacy['ssim_srgb']):+.5f}; "
            f"ΔLPIPS {float(corrected['lpips']) - float(legacy['lpips']):+.5f}; "
            f"{time.perf_counter() - item_started:.1f}s",
            flush=True,
        )
    summary = _aggregate(images, reports)
    summary["renderer_version"] = _powerpoint_version()
    aggregation_elapsed = time.perf_counter() - started
    summary["elapsed_sec"] = (
        existing_summary.get("elapsed_sec", aggregation_elapsed)
        if existing_summary is not None
        else aggregation_elapsed
    )
    summary["aggregation_elapsed_sec"] = aggregation_elapsed
    summary_path = output_root / "results.json"
    atomic_write_text(summary_path, json.dumps(summary, indent=2))
    if args.summary_output is not None:
        atomic_write_text(args.summary_output.resolve(), json.dumps(summary, indent=2))
    _write_overview(output_root, corpus_root, images, summary)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
