#!/usr/bin/env python3
"""Does a PPTX rendering model predict the real PowerPoint capture?

The check that should precede any PPTX proxy or training work: compare the
exact feathered ring-stack model with a real ring-deck capture, and the
plain Gaussian model with its gradient-deck capture. Both land near 0.08-0.10
LPIPS -- see docs/pptx-ring-stack.md. Run before trusting any claim that a
primitive is "exactly modelable".

The whole ring-aware training thesis rests on the proxy being faithful. If
proxy-vs-capture loss is small, training against it is training against
PowerPoint. If it is as large as the gradient emitter's loss, the premise
collapses and the optimizer is chasing a fiction.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault("RING_COUNT", "8")
os.environ.setdefault("RING_LAYOUT", "quantile")
os.environ.setdefault("RING_BLUR", "1.0")

import pptx_ring_training_mvp as trainer
import torch

from splatthis.fidelity.metrics import compute_fidelity_metrics
from splatthis.io import load_png
from splatthis.renderer import render_splats_numpy
from splatthis.storage import load_splats_json

IMAGE = "chameleon"
K = 8
BLUR = 1.0
consts = trainer.emitter_constants()
fr = trainer.quantile_fractions()

run = REPO / "result/corpus/runs" / f"{IMAGE}_pptx_s0_art"
splats = load_splats_json(str(run / "final.raw.json"))
manifest = json.loads((run / "run_manifest.json").read_text())
bg_lin = np.asarray(manifest["config"]["background_linear_rgb"], dtype=np.float32)
src_lin = np.asarray(
    load_png(str(REPO / f"result/corpus/images/{IMAGE}.png"))[..., :3], dtype=np.float32
)
H, W = src_lin.shape[:2]


def to_srgb(t):
    return torch.where(
        t <= 0.0031308, t * 12.92, 1.055 * t.clamp(min=1e-8) ** (1 / 2.4) - 0.055
    )


ordered = sorted(splats, key=lambda s: s.importance)
raws = [s.to_raw_splat().to_dict() for s in ordered]
xy = torch.tensor([[r["x"], r["y"]] for r in raws])
rad = torch.tensor([[r["sx"], r["sy"]] for r in raws]) * consts["radius_per_sigma"]
th = torch.tensor([r["theta"] for r in raws])
col = to_srgb(torch.tensor([[r["r"], r["g"], r["b"]] for r in raws]).clamp(0, 1))
alpha = torch.tensor([r["a"] for r in raws])

fractions = torch.tensor(fr)
q = torch.tensor([(i - 0.5) / K for i in range(1, K + 1)])
gaps = torch.tensor([fr[i] - (fr[i + 1] if i + 1 < K else 0.0) for i in range(K)])
fsig = (BLUR * gaps / 3.25).clamp(min=1e-3)

a0 = (consts["peak_per_alpha"] * alpha).clamp(1e-4, 0.995)
tg = a0[:, None] * q[None, :]
prev = torch.cat([torch.zeros_like(a0)[:, None], tg[:, :-1]], dim=1)
ring_a = ((tg - prev) / (1.0 - prev)).clamp(0.0, 0.995)

ys = torch.arange(H, dtype=torch.float32)
xs = torch.arange(W, dtype=torch.float32)
ct, st = torch.cos(th), torch.sin(th)
bg_srgb = to_srgb(torch.from_numpy(bg_lin))
rows = []
with torch.no_grad():
    for y0 in range(0, H, 16):
        gy = ys[y0 : y0 + 16]
        dy = gy[None, :, None] - xy[:, 1][:, None, None]
        dx = xs[None, None, :] - xy[:, 0][:, None, None]
        u = (dx * ct[:, None, None] + dy * st[:, None, None]) / rad[:, 0][:, None, None]
        v = (-dx * st[:, None, None] + dy * ct[:, None, None]) / rad[:, 1][
            :, None, None
        ]
        f = torch.sqrt(u * u + v * v + 1e-8)
        surv = torch.ones_like(f)
        for k in range(K):
            step = 0.5 * (
                1.0 + torch.erf((fractions[k] - f) / (math.sqrt(2.0) * fsig[k]))
            )
            surv = surv * (1.0 - ring_a[:, k][:, None, None] * step)
        a = 1.0 - surv
        T = torch.flip(torch.cumprod(torch.flip(1.0 - a, dims=[0]), dim=0), dims=[0])
        infront = torch.cat([T[1:], torch.ones_like(a[:1])], dim=0)
        chunk = ((a * infront)[..., None] * col[:, None, None, :]).sum(dim=0)
        chunk = chunk + T[0][..., None] * bg_srgb[None, None, :]
        rows.append(chunk)
proxy_srgb = torch.cat(rows, dim=0).clamp(0, 1).numpy()
proxy_lin = np.where(
    proxy_srgb <= 0.04045, proxy_srgb / 12.92, ((proxy_srgb + 0.055) / 1.055) ** 2.4
).astype(np.float32)

cap = np.asarray(
    load_png(
        str(REPO / "tmp/pptx-ring-stack-mvp" / f"{IMAGE}-rings-k8-q-b1-powerpoint.png")
    )[..., :3],
    dtype=np.float32,
)
rois = [(0, 0, H, W)]


def sc(a_, b_):
    m = compute_fidelity_metrics(a_, b_, fixed_rois=rois, render_method="x").as_dict()
    return float(m["lpips"]), float(m["ssim_srgb"])


print("=== PROXY FAITHFULNESS (chameleon, untrained population) ===")
l, s = sc(proxy_lin, cap)
print(f"ring proxy      vs real ring capture : LPIPS {l:.4f}  SSIM {s:.4f}")

# Reference points for scale
grad_cap = np.asarray(
    load_png(
        str(REPO / "result/corpus/runs" / f"{IMAGE}_pptx_s0_powerpoint_slide.png")
    )[..., :3],
    dtype=np.float32,
)
for space in ("linear", "srgb"):
    model = render_splats_numpy(
        splats, W, H, background_linear_rgb=bg_lin, compositing_space=space
    )[..., :3].astype(np.float32)
    l2, s2 = sc(model, grad_cap)
    print(f"gaussian {space:6s} vs real grad capture : LPIPS {l2:.4f}  SSIM {s2:.4f}")
