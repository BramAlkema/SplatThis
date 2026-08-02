#!/usr/bin/env python3
"""Ring-aware training: fit the splats for the primitive that ships.

The ring-stack emitter is exactly modelable -- quantile radii are universal
constants, ring alphas are a closed-form differentiable function of the
splat's alpha, the feather is an erf step of known width, and PowerPoint
composites solid-alpha shapes in display sRGB (probe: within 0.006). This
tool exploits that: it renders the exact feathered ring composite in torch,
fine-tunes each stored pptx population against the source image under that
model, emits the refined population through the same ring emitter the MVP
uses, and captures the deck in real PowerPoint.

This is the closed loop the gradient style can never have: the optimizer
sees precisely what PowerPoint will draw, steps included, and learns to
place splats that hide them.

Usage::

    PYTHONPATH=src python tools/pptx_ring_training_mvp.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

os.environ.setdefault("RING_COUNT", "8")
os.environ.setdefault("RING_LAYOUT", "quantile")
os.environ.setdefault("RING_BLUR", "1.0")

import pptx_ring_stack_mvp as ring_tool  # noqa: E402  (reads RING_* env)
from full_corpus_mvp import _capture_powerpoint_slideshow  # noqa: E402

from splatthis.io import load_png  # noqa: E402
from splatthis.pptx_export import (  # noqa: E402
    generate_drawingml_slide_content,
    save_pptx_with_drawingml_content,
)
from splatthis.splat import GaussianSplat, RawSplat  # noqa: E402
from splatthis.storage import load_splats_json  # noqa: E402

RUNS = REPO / "result" / "corpus" / "runs"
SOURCES = REPO / "result" / "corpus" / "images"
WORK = REPO / "tmp" / "pptx-ring-training-mvp"

IMAGES = tuple(os.environ.get("TRAIN_IMAGES", "chameleon,colorwheel").split(","))
ITERS = int(os.environ.get("TRAIN_ITERS", "250"))
ROW_CHUNK = 24
_K = int(os.environ["RING_COUNT"])
_BLUR = float(os.environ["RING_BLUR"])
BLUR_SIGMA_DIVISOR = 3.25  # calibrated DrawingML blur: sigma = rad / 3.25


def emitter_constants() -> Dict[str, float]:
    """Extract the deck's geometry and peak-alpha mapping from the emitter."""
    from splatthis.splat import create_isotropic_splat

    sigma, alpha = 6.0, 0.5
    probe = create_isotropic_splat(
        center=[50.0, 50.0], sigma=sigma, color=[1.0, 0.0, 0.0], alpha=alpha
    )
    xml = generate_drawingml_slide_content([probe], width=100, height=100)
    shape = re.search(r"<p:sp>.*?</p:sp>", xml, re.S).group(0)
    ext = re.search(r'<a:ext cx="(\d+)" cy="(\d+)"/>', shape)
    peak = re.search(
        r'<a:gs pos="0">\s*<a:srgbClr val="[0-9A-F]+"><a:alpha val="(\d+)"/>', shape
    )
    radius_px = int(ext.group(1)) / 9525 / 2
    return {
        "radius_per_sigma": radius_px / sigma,
        "peak_per_alpha": int(peak.group(1)) / 100000.0 / alpha,
    }


def quantile_fractions() -> List[float]:
    """Universal ring radii in extent fractions (alpha-independent)."""
    fractions = []
    previous = 1.0
    for i in range(1, _K + 1):
        q = (i - 0.5) / _K
        f = min(1.0, math.sqrt(-2.0 * math.log(q)) / ring_tool.K_SIGMA)
        f = min(f, previous - 1e-3)
        fractions.append(f)
        previous = f
    return fractions


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

    fractions = torch.tensor(quantile_fractions())
    quantiles = torch.tensor([(i - 0.5) / _K for i in range(1, _K + 1)])
    gaps = torch.tensor(
        [
            quantile_fractions()[i]
            - (quantile_fractions()[i + 1] if i + 1 < _K else 0.0)
            for i in range(_K)
        ]
    )
    feather_sigma = (_BLUR * gaps / BLUR_SIGMA_DIVISOR).clamp(min=1e-3)

    radius_per_sigma = constants["radius_per_sigma"]
    peak_per_alpha = constants["peak_per_alpha"]

    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width, dtype=torch.float32)

    def render(y_start: int = 0, y_stop: int = None) -> "torch.Tensor":
        y_stop = height if y_stop is None else y_stop
        alpha = torch.sigmoid(alpha_logit)
        a0 = (peak_per_alpha * alpha).clamp(1e-4, 0.995)
        # Closed-form sequential ring alphas: T_i = a0 * q_i.
        targets = a0[:, None] * quantiles[None, :]
        prev = torch.cat([torch.zeros_like(a0)[:, None], targets[:, :-1]], dim=1)
        ring_a = ((targets - prev) / (1.0 - prev)).clamp(0.0, 0.995)

        color = to_srgb(color_linear.clamp(0.0, 1.0))
        radius = torch.exp(log_s) * radius_per_sigma  # [N,2] px at f=1
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

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
            survivor = torch.ones_like(f)
            for k in range(_K):
                step = 0.5 * (
                    1.0
                    + torch.erf(
                        (fractions[k] - f) / (math.sqrt(2.0) * feather_sigma[k])
                    )
                )
                survivor = survivor * (1.0 - ring_a[:, k][:, None, None] * step)
            a = 1.0 - survivor  # [N, R, W]
            transmittance = torch.flip(
                torch.cumprod(torch.flip(1.0 - a, dims=[0]), dim=0), dims=[0]
            )
            in_front = torch.cat([transmittance[1:], torch.ones_like(a[:1])], dim=0)
            weights = a * in_front
            chunk = (weights[..., None] * color[:, None, None, :]).sum(dim=0)
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
    # Stochastic row bands: each step renders a random horizontal band,
    # cutting per-iteration cost ~4x; the last steps run full-frame.
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
            print(
                f"  iter {iteration:4d}  L1 {loss.item():.5f} " f"(rows {y0}-{y1})",
                flush=True,
            )

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
        deck = WORK / f"{image}-ringtrained.pptx"
        capture = WORK / f"{image}-ringtrained-powerpoint.png"
        if not deck.exists():
            print(
                f"[{index}/{len(IMAGES)}] {image}: ring-aware refinement "
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
            counter = [50000]
            transformed = ring_tool.SP_PATTERN.sub(
                lambda m: ring_tool.transform_shape(m, counter), slide_xml
            )
            save_pptx_with_drawingml_content(
                slide_xml=transformed,
                width=width,
                height=height,
                output_path=str(deck),
                splat_count=len(refined),
            )
            print(
                f"[{index}/{len(IMAGES)}] {image}: deck emitted "
                f"({counter[0] - 50000} rings)",
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
            "ring_trained": ring_tool.score(image, capture),
            "baseline_gradient": {
                "ssim_srgb": round(float(baseline[image]["ssim_srgb"]), 4),
                "lpips": round(float(baseline[image]["lpips"]), 4),
            },
        }
        print(f"[{index}/{len(IMAGES)}] {image}: scored", flush=True)

    (WORK / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"\n{'image':22s}{'variant':>14s}{'ssim':>10s}{'lpips':>10s}")
    for image, entry in results.items():
        b, r = entry["baseline_gradient"], entry["ring_trained"]
        print(
            f"{image:22s}{'gradient':>14s}{b['ssim_srgb']:>10.4f}"
            f"{b['lpips']:>10.4f}"
        )
        print(
            f"{image:22s}{'ring-trained':>14s}{r['ssim_srgb']:>10.4f}"
            f"{r['lpips']:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
