"""Regression tests for content-addressed corpus benchmark runs."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from png2svg_gs.io import save_splats_json
from png2svg_gs.splat import create_isotropic_splat

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "corpus_benchmark", REPO / "tools" / "corpus_benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
corpus_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = corpus_benchmark
SPEC.loader.exec_module(corpus_benchmark)


def test_run_identity_changes_with_source_and_optimizer_config(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"first")
    base = corpus_benchmark._run_config(
        source=source,
        fmt="svg",
        seed=0,
        splats=2000,
        stages="1000,500,250",
        profile="max-fidelity",
        optimizer_backend="mlx",
        full_geometry=False,
    )
    base_hash = corpus_benchmark._config_hash(base)

    source.write_bytes(b"second")
    changed_source = corpus_benchmark._run_config(
        source=source,
        fmt="svg",
        seed=0,
        splats=2000,
        stages="1000,500,250",
        profile="max-fidelity",
        optimizer_backend="mlx",
        full_geometry=False,
    )
    assert corpus_benchmark._config_hash(changed_source) != base_hash

    changed_optimizer = dict(base, optimizer_backend="torch")
    assert corpus_benchmark._config_hash(changed_optimizer) != base_hash

    changed_initial_population = corpus_benchmark._run_config(
        source=source,
        fmt="svg",
        seed=0,
        splats=2000,
        stages="1000,500,250",
        profile="max-fidelity",
        optimizer_backend="mlx",
        full_geometry=False,
        initial_splat_cap=2000,
        initial_splat_fraction=0.75,
    )
    assert changed_initial_population["initial_splat_cap"] == 2000
    assert changed_initial_population["initial_splat_fraction"] == 0.75
    assert corpus_benchmark._config_hash(changed_initial_population) != (
        corpus_benchmark._config_hash(changed_source)
    )


def test_run_key_uses_config_hash_not_human_label() -> None:
    key = corpus_benchmark.run_key("rocket", "pptx", 0, "abc123")
    assert key == "rocket|pptx|seed0|cfg-abc123"


def test_overview_scopes_runtime_to_canvas_and_compares_artifacts(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    source = tmp_path / "sample.png"
    Image.fromarray(np.full((8, 10, 3), 128, dtype=np.uint8)).save(source)
    corpus = {
        "images": {
            "sample": {
                "path": "sample.png",
                "size": [10, 8],
                "bytes": source.stat().st_size,
                "content_class": "fixture",
                "note": "comparison fixture",
            }
        }
    }
    (tmp_path / "corpus.json").write_text(json.dumps(corpus))

    splat = create_isotropic_splat(
        center=np.array([5.0, 4.0]),
        sigma=2.0,
        color=np.array([0.5, 0.5, 0.5]),
        alpha=0.8,
    )
    artifact_dir = runs / "sample_canvas_s0_art"
    artifact_dir.mkdir()
    save_splats_json([splat], str(artifact_dir / "final.raw.json"))
    (artifact_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "resolved_target_size": [10, 8],
                    "training_export_target": "canvas",
                    "background_linear_rgb": [0.0, 0.0, 0.0],
                },
                "acceptance": {"measured": {"runtime_sec": 12.0}},
            }
        )
    )

    (runs / "sample_canvas_s0.html").write_text("<canvas></canvas>")
    (runs / "sample_svg_s0.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="8"/>'
    )
    (runs / "sample_pptx_s0.pptx").write_bytes(b"pptx-fixture")
    Image.fromarray(np.full((8, 10, 3), 120, dtype=np.uint8)).save(
        runs / "sample_pptx_s0_powerpoint_slide.png"
    )
    records = [
        {
            "key": "sample|canvas|seed0|fixture",
            "image": "sample",
            "format": "canvas",
            "seed": 0,
            "returncode": 0,
            "artifact_bytes": 100,
            "splats_final": 1,
            "runtime_sec": 12.0,
            "lpips": 0.1,
            "ssim_srgb": 0.9,
            "psnr_srgb": 30.0,
            "renderer": "canvas-linear",
        },
        {
            "key": "sample|svg|seed0|fixture",
            "image": "sample",
            "format": "svg",
            "seed": 0,
            "returncode": 0,
            "artifact_bytes": 200,
            "splats_final": 1,
            "runtime_sec": 14.0,
            "lpips": 0.2,
            "ssim_srgb": 0.8,
            "psnr_srgb": 25.0,
            "renderer": "rsvg-convert",
        },
        {
            "key": "sample|pptx|seed0|fixture",
            "image": "sample",
            "format": "pptx",
            "seed": 0,
            "returncode": 0,
            "artifact_bytes": 300,
            "splats_final": 1,
            "runtime_sec": 16.0,
        },
    ]
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (tmp_path / "powerpoint_results.jsonl").write_text(
        json.dumps(
            {
                "image": "sample",
                "lpips": 0.3,
                "ssim_srgb": 0.7,
                "psnr_srgb": 20.0,
            }
        )
        + "\n"
    )

    output = tmp_path / "index.html"
    corpus_benchmark.generate_canvas_corpus_html(tmp_path, output)
    document = output.read_text(encoding="utf-8")

    assert '<figure class="canvas-panel">' in document
    assert '<span id="status-0">queued · 1 canvas-trained splats</span>' in document
    assert '<footer id="status-' not in document
    assert "<caption>Artifact comparison</caption>" in document
    assert "actual slideshow capture" in document
    assert "<td>0.3000</td>" in document
    assert '<td class="best">0.1000</td>' in document
    assert "Math.sqrt((sx*ct)*(sx*ct)+(sy*st)*(sy*st))" in document
    assert 'data-model-index="0"' in document
    assert "IntersectionObserver" in document
    assert "requestIdleCallback" in document
    assert "requestAnimationFrame(renderNext)" not in document
    assert "model.splats.sort" not in document
