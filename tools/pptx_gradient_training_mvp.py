#!/usr/bin/env python3
"""Hybrid-aware fit of the *existing* PPTX gradient primitive.

The ring experiment showed that fitting against a compositor model which
actually matches PowerPoint recovers a large colour gain -- but it paid for
that with eight times the shapes and a 34-second slide (docs/pptx-ring-stack.md).
This tool tests whether the colour half of that result survives at zero
shape cost, by fitting the shipped gradient primitive against an exact model
of itself.

The gradient splat turns out to be cleanly modelable, contrary to the
assumption behind the ring detour. Its DrawingML ``gradFill`` always carries
eight evenly spaced stops whose opacities follow

    op(t) = 1 - exp(-scale * alpha * exp(-0.5 * (t * footprint)^2))

with ``scale`` and ``footprint`` extracted from the emitter at run time
(measured exact to 1e-5 across alpha in [0.02, 0.97]). PowerPoint
interpolates those stops linearly and composites the result alpha-over in
display sRGB, per the probe in docs/pptx-colorspace.md. This proxy renders
exactly that -- piecewise-linear stop ramp, half-pixel edge antialiasing,
sRGB alpha-over, back-to-front -- and is differentiable in position, scale,
rotation, colour and alpha.

Emission uses the ordinary gradient emitter, so the deck is byte-comparable
to the shipping one: same shape count, same size, same open time. Only the
fitted parameters change.

Usage::

    PYTHONPATH=src python tools/pptx_gradient_training_mvp.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from full_corpus_mvp import _capture_powerpoint_slideshow  # noqa: E402

from splatthis.fidelity.metrics import compute_fidelity_metrics  # noqa: E402
from splatthis.io import load_png  # noqa: E402
from splatthis.pptx_export import (  # noqa: E402
    generate_drawingml_slide_content,
    save_pptx_with_drawingml_content,
)
from splatthis.splat import GaussianSplat  # noqa: E402
from splatthis.splat import RawSplat, create_isotropic_splat  # noqa: E402
from splatthis.storage import load_splats_json  # noqa: E402

RUNS = REPO / "result" / "corpus" / "runs"
SOURCES = REPO / "result" / "corpus" / "images"
WORK = REPO / "tmp" / "pptx-gradient-training-mvp"

IMAGES = tuple(os.environ.get("TRAIN_IMAGES", "chameleon").split(","))
ITERS = int(os.environ.get("TRAIN_ITERS", "300"))
ROW_CHUNK = 24
N_STOPS = 8


def emitter_constants() -> Dict[str, float]:
    """Extract the deck's geometry and opacity-curve constants from the emitter."""
    sigma, alpha = 6.0, 0.5
    probe = create_isotropic_splat(
        center=[50.0, 50.0], sigma=sigma, color=[1.0, 0.0, 0.0], alpha=alpha
    )
    xml = generate_drawingml_slide_content([probe], width=100, height=100)
    shape = re.search(r"<p:sp>.*?</p:sp>", xml, re.S).group(0)
    ext = re.search(r'<a:ext cx="(\d+)" cy="(\d+)"/>', shape)
    gs = re.findall(
        r'<a:gs pos="(\d+)">\s*<a:srgbClr val="[0-9A-F]{6}">'
        r'(?:<a:alpha val="(\d+)"/>)?',
        shape,
    )
    positions = np.array([int(p) / 100000 for p, _ in gs])
    opacities = np.array([(int(a) / 100000 if a else 1.0) for _, a in gs])
    if len(positions) != N_STOPS:
        raise RuntimeError(f"expected {N_STOPS} gradient stops, got {len(positions)}")

    # op0 = 1 - exp(-scale * alpha)  ->  scale
    scale = -math.log(1.0 - opacities[0]) / alpha
    # op1 at t = 1/7  ->  footprint
    t1 = positions[1] / positions[-1]
    inner = -math.log(1.0 - opacities[1]) / (scale * alpha)
    footprint = math.sqrt(-2.0 * math.log(inner)) / t1
    return {
        "radius_per_sigma": int(ext.group(1)) / 9525 / 2 / sigma,
        "opacity_scale": scale,
        "footprint": footprint,
    }


def refine(image: str, constants: Dict[str, float]):
    import torch

    torch.manual_seed(0)
    run = RUNS / f"{image}_pptx_s0_art"
    splats = load_splats_json(str(run / "final.raw.json"))
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    bg_linear = np.asarray(
        manifest["config"]["background_linear_rgb"], dtype=np.float32
    )
    source_linear = np.asarray(
        load_png(str(SOURCES / f"{image}.png"))[..., :3], dtype=np.float32
    )
    height, width = source_linear.shape[:2]

    def to_srgb(t):
        return torch.where(
            t <= 0.0031308, t * 12.92, 1.055 * t.clamp(min=1e-8) ** (1 / 2.4) - 0.055
        )

    target = to_srgb(torch.from_numpy(source_linear))
    bg_srgb = to_srgb(torch.from_numpy(bg_linear))

    ordered = sorted(splats, key=lambda s: s.importance)  # back-to-front
    raws = [s.to_raw_splat().to_dict() for s in ordered]
    xy = torch.tensor([[r["x"], r["y"]] for r in raws], requires_grad=True)
    log_s = torch.tensor(
        [[math.log(max(r["sx"], 0.3)), math.log(max(r["sy"], 0.3))] for r in raws],
        requires_grad=True,
    )
    theta = torch.tensor([r["theta"] for r in raws], requires_grad=True)
    color_linear = torch.tensor(
        [[r["r"], r["g"], r["b"]] for r in raws], requires_grad=True
    )
    alpha_logit = torch.tensor(
        [
            math.log(min(max(r["a"], 0.02), 0.97) / (1 - min(max(r["a"], 0.02), 0.97)))
            for r in raws
        ],
        requires_grad=True,
    )

    stop_t = torch.linspace(0.0, 1.0, N_STOPS)
    gauss_at_stop = torch.exp(-0.5 * (stop_t * constants["footprint"]) ** 2)
    radius_per_sigma = constants["radius_per_sigma"]
    opacity_scale = constants["opacity_scale"]

    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width, dtype=torch.float32)

    def render(y_start: int = 0, y_stop: int = None):
        y_stop = height if y_stop is None else y_stop
        alpha = torch.sigmoid(alpha_logit)
        # Exact emitter stop opacities, differentiable in alpha.
        stop_op = 1.0 - torch.exp(
            -opacity_scale * alpha[:, None] * gauss_at_stop[None, :]
        )
        color = to_srgb(color_linear.clamp(0.0, 1.0))
        radius = torch.exp(log_s) * radius_per_sigma
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        # Half-pixel edge antialiasing, in normalized-radius units.
        aa = (0.5 / radius.mean(dim=1)).clamp(1e-3, 0.5)

        rows = []
        for y0 in range(y_start, y_stop, ROW_CHUNK):
            gy = ys[y0 : min(y0 + ROW_CHUNK, y_stop)]
            dy = gy[None, :, None] - xy[:, 1][:, None, None]
            dx = xs[None, None, :] - xy[:, 0][:, None, None]
            u = (dx * cos_t[:, None, None] + dy * sin_t[:, None, None]) / radius[:, 0][
                :, None, None
            ]
            v = (-dx * sin_t[:, None, None] + dy * cos_t[:, None, None]) / radius[:, 1][
                :, None, None
            ]
            f = torch.sqrt(u * u + v * v + 1e-8)
            # Piecewise-linear stop ramp via triangular basis functions.
            node = f * (N_STOPS - 1)
            a = torch.zeros_like(f)
            for i in range(N_STOPS):
                hat = (1.0 - (node - i).abs()).clamp(0.0, 1.0)
                a = a + stop_op[:, i][:, None, None] * hat
            edge = 0.5 * (
                1.0 + torch.erf((1.0 - f) / (math.sqrt(2.0) * aa[:, None, None]))
            )
            a = (a * edge).clamp(0.0, 0.995)
            transmittance = torch.flip(
                torch.cumprod(torch.flip(1.0 - a, dims=[0]), dim=0), dims=[0]
            )
            in_front = torch.cat([transmittance[1:], torch.ones_like(a[:1])], dim=0)
            chunk = ((a * in_front)[..., None] * color[:, None, None, :]).sum(dim=0)
            chunk = chunk + transmittance[0][..., None] * bg_srgb[None, None, :]
            rows.append(chunk)
        return torch.cat(rows, dim=0)

    optimizer = torch.optim.Adam(
        [
            {"params": [xy], "lr": 3e-2},
            {"params": [log_s], "lr": 4e-3},
            {"params": [theta], "lr": 4e-3},
            {"params": [color_linear], "lr": 6e-3},
            {"params": [alpha_logit], "lr": 8e-3},
        ]
    )
    band = min(96, height)
    generator = torch.Generator().manual_seed(1)
    for iteration in range(ITERS):
        optimizer.zero_grad()
        if iteration >= ITERS - 10:
            y0, y1 = 0, height
        else:
            y0 = int(
                torch.randint(0, max(1, height - band + 1), (1,), generator=generator)
            )
            y1 = y0 + band
        loss = (render(y0, y1) - target[y0:y1]).abs().mean()
        loss.backward()
        optimizer.step()
        if iteration % 25 == 0 or iteration == ITERS - 1:
            print(f"  iter {iteration:4d}  L1 {loss.item():.5f}", flush=True)

    refined = []
    with torch.no_grad():
        alpha = torch.sigmoid(alpha_logit)
        for i, raw in enumerate(raws):
            item = dict(raw)
            item["x"], item["y"] = float(xy[i, 0]), float(xy[i, 1])
            item["sx"] = float(torch.exp(log_s[i, 0]))
            item["sy"] = float(torch.exp(log_s[i, 1]))
            item["theta"] = float(theta[i])
            item["r"] = float(color_linear[i, 0].clamp(0, 1))
            item["g"] = float(color_linear[i, 1].clamp(0, 1))
            item["b"] = float(color_linear[i, 2].clamp(0, 1))
            item["a"] = float(alpha[i])
            refined.append(GaussianSplat.from_raw_splat(RawSplat.from_dict(item)))
    return refined, bg_linear, width, height


def score(image: str, capture_path: Path) -> Dict[str, float]:
    source = np.asarray(
        load_png(str(SOURCES / f"{image}.png"))[..., :3], dtype=np.float32
    )
    rendered = np.asarray(load_png(str(capture_path))[..., :3], dtype=np.float32)
    height, width = source.shape[:2]
    rois = [
        (y, x, min(y + 64, height), min(x + 64, width))
        for y in range(0, height, 64)
        for x in range(0, width, 64)
    ]
    metrics = compute_fidelity_metrics(
        source, rendered, fixed_rois=rois, render_method="pp"
    ).as_dict()
    return {
        name: round(float(metrics[name]), 4)
        for name in ("ssim_srgb", "lpips", "delta_e_ok_mean", "delta_e_ok_p95")
    }


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    constants = emitter_constants()
    print(f"emitter constants: {constants}")
    baseline = {
        row["image"]: row
        for row in (
            json.loads(line)
            for line in (REPO / "result" / "corpus" / "powerpoint_results.jsonl")
            .read_text()
            .splitlines()
            if line
        )
    }
    results: Dict[str, Any] = {}
    for index, image in enumerate(IMAGES, 1):
        deck = WORK / f"{image}-gradtrained.pptx"
        capture = WORK / f"{image}-gradtrained-powerpoint.png"
        if not deck.exists():
            print(
                f"[{index}/{len(IMAGES)}] {image}: hybrid-aware refinement "
                f"({ITERS} iters) ...",
                flush=True,
            )
            refined, bg_linear, width, height = refine(image, constants)
            slide_xml = generate_drawingml_slide_content(
                refined,
                width=width,
                height=height,
                background_linear_rgb=bg_linear,
            )
            save_pptx_with_drawingml_content(
                slide_xml=slide_xml,
                width=width,
                height=height,
                output_path=str(deck),
                splat_count=len(refined),
            )
            print(
                f"[{index}/{len(IMAGES)}] {image}: deck emitted "
                f"({slide_xml.count('<p:sp>')} shapes, "
                f"{deck.stat().st_size / 1024:.0f} KB)",
                flush=True,
            )
        if not capture.exists():
            source = load_png(str(SOURCES / f"{image}.png"))
            height, width = source.shape[:2]
            print(f"[{index}/{len(IMAGES)}] {image}: capturing ...", flush=True)
            returncode, message = _capture_powerpoint_slideshow(
                deck, capture, width, height
            )
            if returncode or not capture.exists():
                print(f"capture failed: {message.strip()[-300:]}", file=sys.stderr)
                return 2
        results[image] = {
            "hybrid_trained": score(image, capture),
            "baseline": {
                "ssim_srgb": round(float(baseline[image]["ssim_srgb"]), 4),
                "lpips": round(float(baseline[image]["lpips"]), 4),
            },
            "deck_bytes": deck.stat().st_size,
        }
        print(f"[{index}/{len(IMAGES)}] {image}: scored", flush=True)

    (WORK / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"\n{'image':20s}{'variant':>16s}{'ssim':>9s}{'lpips':>9s}{'deck KB':>10s}")
    for image, entry in results.items():
        b, h = entry["baseline"], entry["hybrid_trained"]
        print(
            f"{image:20s}{'gradient base':>16s}{b['ssim_srgb']:>9.4f}"
            f"{b['lpips']:>9.4f}{'161':>10s}"
        )
        print(
            f"{image:20s}{'hybrid-trained':>16s}{h['ssim_srgb']:>9.4f}"
            f"{h['lpips']:>9.4f}{entry['deck_bytes'] / 1024:>10.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
