"""Self-contained browser pixel runtimes for Gaussian splat artifacts."""

from __future__ import annotations

from html import escape as escape_html
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from .color import linear_to_srgb
from .splat import GaussianSplat, render_importance_for_raw


def generate_parallax_pixel_runtime_html(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    title: str = "SplatThis Parallax",
    parallax_strength: float = 28.0,
) -> str:
    """Parallax ImageData runtime: per-layer canvases driven by mouse position.

    Splats with ``raw.layer`` set (via ``--layered-saliency``) get bucketed
    into base/mass/detail/edge canvases. Each canvas runs the same
    linear-light alpha-over render as ``generate_pixel_runtime_html`` but only on
    its own splats. The canvases stack absolutely; on mousemove a
    ``translate3d`` is applied per canvas scaled by its depth (base
    stationary, edge moves the most). Background-rect plate is painted
    behind layer 0 so areas revealed by foreground translation show the
    scene background color, not black/empty.

    Quality caveat: each layer composites linear-light internally, but the
    DOM composites the layers in sRGB display space. Tiny color drift vs
    the single-buffer pixel runtime at static rest. The parallax effect itself
    is the goal here, not pixel-perfect render parity.

    Splats without a layer tag fall back to layer 1 ("mass") so they get
    a modest parallax offset.
    """

    import json

    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("Canvas width and height must be positive integers")

    bg_lin = (
        [0.0, 0.0, 0.0]
        if background_linear_rgb is None
        else [
            float(np.clip(c, 0.0, 1.0))
            for c in np.asarray(background_linear_rgb).reshape(-1)[:3]
        ]
    )
    bg_srgb = linear_to_srgb(np.array(bg_lin, dtype=np.float32))
    bg_rgb = tuple(int(np.clip(np.round(c * 255), 0, 255)) for c in bg_srgb)
    bg_css = f"rgb({bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]})"

    # Bucket splats into three parallax planes (not four). The saliency
    # layers (base=0, mass=1, detail=2, edge=3) get collapsed:
    #   base (0)    -> background plane (stationary)
    #   mass (1)    -> midground plane
    #   detail (2)  -> foreground plane     (merged with edge)
    #   edge (3)    -> foreground plane     (merged with detail)
    # Merging detail+edge avoids visible tearing between near-foreground
    # planes (the man's eye+glasses-rim splats would otherwise track at
    # different speeds than the face-skin splats) and eliminates one
    # cross-layer sRGB compositing seam at the most detailed region.
    # Untagged splats fall back to midground.
    PLANE_BACKGROUND = "background"
    PLANE_MIDGROUND = "midground"
    PLANE_FOREGROUND = "foreground"
    PLANE_DEPTHS = {
        PLANE_BACKGROUND: 0.0,
        PLANE_MIDGROUND: 0.4,
        PLANE_FOREGROUND: 1.0,
    }

    def _layer_to_plane(layer_id: Optional[int]) -> str:
        if layer_id is None:
            return PLANE_MIDGROUND
        if layer_id <= 0:
            return PLANE_BACKGROUND
        if layer_id == 1:
            return PLANE_MIDGROUND
        return PLANE_FOREGROUND

    buckets: Dict[str, List[List[float]]] = {
        PLANE_BACKGROUND: [],
        PLANE_MIDGROUND: [],
        PLANE_FOREGROUND: [],
    }
    for splat in splats:
        raw = splat.to_raw_splat()
        plane = _layer_to_plane(raw.layer)
        buckets[plane].append(
            [
                float(raw.x),
                float(raw.y),
                float(raw.sx),
                float(raw.sy),
                float(raw.theta),
                float(raw.r),
                float(raw.g),
                float(raw.b),
                float(raw.a),
                render_importance_for_raw(raw),
            ]
        )

    layer_records = []
    for plane in (PLANE_BACKGROUND, PLANE_MIDGROUND, PLANE_FOREGROUND):
        if not buckets[plane]:
            continue
        # Sort once while emitting instead of making every browser repeat it.
        buckets[plane].sort(key=lambda row: row[9])
        layer_records.append(
            {
                "layer": plane,
                "depth": PLANE_DEPTHS[plane],
                "splats": buckets[plane],
            }
        )
    layer_data_json = json.dumps(layer_records, separators=(",", ":"))

    js = (
        r"""
(function(){
  const t0 = performance.now();
  const W = __W__, H = __H__;
  const BG = __BG__;
  const STRENGTH = __STRENGTH__;
  const LAYERS = __LAYERS__;
  const status = document.getElementById('status');
  const stack = document.getElementById('stack');

  function renderLayer(canvas, splats) {
    const ctx = canvas.getContext('2d', { willReadFrequently: false });
    const lin = new Float32Array(W * H * 3);
    const T = new Float32Array(W * H).fill(1);
    const FOOTPRINT = 3.0;
    for (let si = 0; si < splats.length; si++) {
      const s = splats[si];
      const x = s[0], y = s[1];
      const sx = Math.max(s[2], 1e-4), sy = Math.max(s[3], 1e-4);
      const theta = s[4];
      const r = s[5], g = s[6], b = s[7];
      const a = Math.min(1, Math.max(0, s[8]));
      const ct = Math.cos(theta), st = Math.sin(theta);
      const rx = Math.max(1, Math.ceil(FOOTPRINT * Math.sqrt((sx*ct)*(sx*ct) + (sy*st)*(sy*st))));
      const ry = Math.max(1, Math.ceil(FOOTPRINT * Math.sqrt((sx*st)*(sx*st) + (sy*ct)*(sy*ct))));
      const x0 = Math.max(0, Math.floor(x - rx));
      const x1 = Math.min(W, Math.ceil(x + rx + 1));
      const y0 = Math.max(0, Math.floor(y - ry));
      const y1 = Math.min(H, Math.ceil(y + ry + 1));
      if (x0 >= x1 || y0 >= y1) continue;
      const invSx2 = 1 / (sx * sx), invSy2 = 1 / (sy * sy);
      for (let py = y0; py < y1; py++) {
        const baseRow = py * W;
        for (let px = x0; px < x1; px++) {
          const dx = px - x, dy = py - y;
          const u = ct * dx + st * dy;
          const v = -st * dx + ct * dy;
          const q = u * u * invSx2 + v * v * invSy2;
          const w = Math.exp(-0.5 * q);
          const la = 1 - Math.exp(-a * w);
          const idx = baseRow + px;
          const tt = T[idx];
          const contrib = tt * la;
          const j = idx * 3;
          lin[j]     += contrib * r;
          lin[j + 1] += contrib * g;
          lin[j + 2] += contrib * b;
          T[idx] = tt * (1 - la);
        }
      }
    }
    const img = ctx.createImageData(W, H);
    const out = img.data;
    const THR = 0.0031308;
    // Per-layer canvases are stacked over the bg plate; transparent pixels
    // (T near 1) reveal the layer below, so write alpha = 1 - T.
    for (let i = 0; i < W * H; i++) {
      const j = i * 3, k = i * 4;
      const tt = T[i];
      let rL = lin[j], gL = lin[j + 1], bL = lin[j + 2];
      const denom = (1 - tt) > 1e-6 ? (1 - tt) : 1;
      rL = rL / denom;
      gL = gL / denom;
      bL = bL / denom;
      if (rL < 0) rL = 0; else if (rL > 1) rL = 1;
      if (gL < 0) gL = 0; else if (gL > 1) gL = 1;
      if (bL < 0) bL = 0; else if (bL > 1) bL = 1;
      const rS = rL <= THR ? 12.92 * rL : 1.055 * Math.pow(rL, 1/2.4) - 0.055;
      const gS = gL <= THR ? 12.92 * gL : 1.055 * Math.pow(gL, 1/2.4) - 0.055;
      const bS = bL <= THR ? 12.92 * bL : 1.055 * Math.pow(bL, 1/2.4) - 0.055;
      out[k]     = (rS * 255 + 0.5) | 0;
      out[k + 1] = (gS * 255 + 0.5) | 0;
      out[k + 2] = (bS * 255 + 0.5) | 0;
      out[k + 3] = ((1 - tt) * 255 + 0.5) | 0;
    }
    ctx.putImageData(img, 0, 0);
  }

  const layerEls = [];
  for (const ld of LAYERS) {
    const cnv = document.createElement('canvas');
    cnv.width = W; cnv.height = H;
    cnv.className = 'layer';
    cnv.dataset.depth = ld.depth;
    stack.appendChild(cnv);
    renderLayer(cnv, ld.splats);
    layerEls.push(cnv);
  }

  let total = 0;
  for (const ld of LAYERS) total += ld.splats.length;
  status.textContent = 'parallax ready: ' + LAYERS.length + ' layers, ' + total + ' splats, rendered in ' + (performance.now() - t0).toFixed(0) + 'ms';

  function onMove(e) {
    const rect = stack.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width  - 0.5) * 2;
    const my = ((e.clientY - rect.top)  / rect.height - 0.5) * 2;
    for (const el of layerEls) {
      const d = parseFloat(el.dataset.depth);
      // Foreground tracks the mouse, background stays still. We translate
      // OPPOSITE the mouse so the parallax feels like looking through the scene.
      const tx = -mx * d * STRENGTH;
      const ty = -my * d * STRENGTH;
      el.style.transform = 'translate3d(' + tx.toFixed(2) + 'px,' + ty.toFixed(2) + 'px,0)';
    }
  }
  stack.addEventListener('mousemove', onMove);
  stack.addEventListener('mouseleave', () => {
    for (const el of layerEls) el.style.transform = 'translate3d(0,0,0)';
  });
})();
""".replace(
            "__W__", str(int(width))
        )
        .replace("__H__", str(int(height)))
        .replace("__BG__", f"[{bg_lin[0]:.6f},{bg_lin[1]:.6f},{bg_lin[2]:.6f}]")
        .replace("__STRENGTH__", f"{float(parallax_strength):.3f}")
        .replace("__LAYERS__", layer_data_json)
    )

    safe_title = escape_html(title)
    return (
        "<!doctype html>\n"
        f'<html><head><meta charset="utf-8"><title>{safe_title}</title>\n'
        "<style>\n"
        "  body { margin: 0; background: #111; color: #eee;"
        "    font: 14px -apple-system, sans-serif;"
        "    display: flex; flex-direction: column; align-items: center; padding: 16px; }\n"
        f"  #stack {{ position: relative; width: {int(width)}px; height: {int(height)}px;"
        f"    background: {bg_css}; overflow: hidden; border: 1px solid #333; border-radius: 6px; }}\n"
        "  #stack .layer { position: absolute; top: 0; left: 0;"
        "    transition: transform 0.06s cubic-bezier(0.2,0.7,0.3,1.0);"
        "    pointer-events: none; image-rendering: pixelated; }\n"
        "  #status { color: #7fd17f; font-family: ui-monospace, monospace;"
        "    font-size: 12px; margin: 8px 0; }\n"
        "</style></head>\n"
        "<body>\n"
        f'<div id="stack" data-compositor="pixel-runtime-parallax" '
        f'data-layers="{len(layer_records)}"></div>\n'
        '<div id="status">rendering...</div>\n'
        "<script>\n" + js + "\n</script>\n"
        "</body></html>\n"
    )


def generate_pixel_runtime_html(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    title: str = "SplatThis Pixel Runtime",
    compositing_space: str = "linear",
) -> str:
    """Self-contained HTML that software-rasterizes splats into ImageData.

    The static runtime executes in a Web Worker on an OffscreenCanvas when the
    browser supports both APIs, then transfers the exact packed pixel bytes to
    the visible canvas. A main-thread implementation of the same function is
    retained as a compatibility fallback. ``__SPLATTHIS_RENDER_MS`` measures
    end-to-end completion while ``__SPLATTHIS_COMPUTE_MS`` isolates raster
    computation.

    Shares its serialized inputs with `render_pixel_runtime_numpy`, whose
    double-precision math, Float32Array writeback, and 8-bit ImageData packing
    are calibrated byte-for-byte against Chrome. Both use a 3σ footprint,
    importance ascending = back-to-front ordering, and per-splat alpha-over with
    `layer_alpha = 1 - exp(-a * exp(-0.5 * q))`). In "linear" mode the
    accumulator is linear-RGB and the output is gamma-encoded at the end; in
    "srgb" mode colors/background are pre-encoded host-side and composited
    directly in display space, matching models trained for SVG/PPTX deploy
    targets. The browser's gamma-space SVG compositing and gradient-stop
    discretization are the things you can't reproduce with `radialGradient`;
    a JS canvas can. The continuous optimizer forward remains unquantized; the
    deployed scorer models the actual displayed bytes.
    """
    import json

    from .renderer import prepare_pixel_runtime_data

    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("Pixel runtime width and height must be positive integers")

    rows, serialized_background, srgb_mode = prepare_pixel_runtime_data(
        splats,
        background_linear_rgb=background_linear_rgb,
        compositing_space=compositing_space,
    )
    bg_lin = [float(channel) for channel in serialized_background]
    splats_json = json.dumps(rows, separators=(",", ":"))

    js = (
        r"""
(function(){
  const t0 = performance.now();
  const W = __W__, H = __H__;
  const BG = __BG__;
  const SPLATS = __SPLATS__;
  const SRGB_IN = __SRGB_IN__;
  const status = document.getElementById('status');
  const canvas = document.getElementById('c');
  let finished = false;

  function rasterizePixelRuntime(targetCanvas, splats, width, height, background, srgbIn) {
    const computeStarted = performance.now();
    const FOOTPRINT = 3.0;
    const ctx = targetCanvas.getContext('2d', { willReadFrequently: false });
    const lin = new Float32Array(width * height * 3);
    const T = new Float32Array(width * height).fill(1);

    for (let si = 0; si < splats.length; si++) {
      const s = splats[si];
      const x = s[0], y = s[1];
      const sx = Math.max(s[2], 1e-4), sy = Math.max(s[3], 1e-4);
      const theta = s[4];
      const r = s[5], g = s[6], b = s[7];
      const a = Math.min(1, Math.max(0, s[8]));
      const ct = Math.cos(theta), st = Math.sin(theta);
      const rx = Math.max(1, Math.ceil(FOOTPRINT * Math.sqrt((sx*ct)*(sx*ct) + (sy*st)*(sy*st))));
      const ry = Math.max(1, Math.ceil(FOOTPRINT * Math.sqrt((sx*st)*(sx*st) + (sy*ct)*(sy*ct))));
      const x0 = Math.max(0, Math.floor(x - rx));
      const x1 = Math.min(width, Math.ceil(x + rx + 1));
      const y0 = Math.max(0, Math.floor(y - ry));
      const y1 = Math.min(height, Math.ceil(y + ry + 1));
      if (x0 >= x1 || y0 >= y1) continue;
      const invSx2 = 1 / (sx * sx), invSy2 = 1 / (sy * sy);
      for (let py = y0; py < y1; py++) {
        const baseRow = py * width;
        for (let px = x0; px < x1; px++) {
          const dx = px - x, dy = py - y;
          const u = ct * dx + st * dy;
          const v = -st * dx + ct * dy;
          const q = u * u * invSx2 + v * v * invSy2;
          const w = Math.exp(-0.5 * q);
          const la = 1 - Math.exp(-a * w);
          const idx = baseRow + px;
          const tt = T[idx];
          const contrib = tt * la;
          const j = idx * 3;
          lin[j]     += contrib * r;
          lin[j + 1] += contrib * g;
          lin[j + 2] += contrib * b;
          T[idx] = tt * (1 - la);
        }
      }
    }

    // Pack into ImageData. srgbIn means colors were pre-encoded host-side and
    // composited directly in display space, so no gamma encode here.
    const img = ctx.createImageData(width, height);
    const out = img.data;
    const THR = 0.0031308;
    for (let i = 0; i < width * height; i++) {
      const j = i * 3, k = i * 4;
      const tt = T[i];
      let rL = lin[j]     + tt * background[0];
      let gL = lin[j + 1] + tt * background[1];
      let bL = lin[j + 2] + tt * background[2];
      if (rL < 0) rL = 0; else if (rL > 1) rL = 1;
      if (gL < 0) gL = 0; else if (gL > 1) gL = 1;
      if (bL < 0) bL = 0; else if (bL > 1) bL = 1;
      const rS = srgbIn ? rL : (rL <= THR ? 12.92 * rL : 1.055 * Math.pow(rL, 1/2.4) - 0.055);
      const gS = srgbIn ? gL : (gL <= THR ? 12.92 * gL : 1.055 * Math.pow(gL, 1/2.4) - 0.055);
      const bS = srgbIn ? bL : (bL <= THR ? 12.92 * bL : 1.055 * Math.pow(bL, 1/2.4) - 0.055);
      out[k]     = (rS * 255 + 0.5) | 0;
      out[k + 1] = (gS * 255 + 0.5) | 0;
      out[k + 2] = (bS * 255 + 0.5) | 0;
      out[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return {
      computeMs: performance.now() - computeStarted,
      pixels: img.data
    };
  }

  function finish(mode, computeMs) {
    if (finished) return;
    finished = true;
    const renderMs = performance.now() - t0;
    canvas.dataset.execution = mode;
    window.__SPLATTHIS_COMPUTE_MS = computeMs;
    window.__SPLATTHIS_RENDER_MS = renderMs;
    window.__SPLATTHIS_RENDER_MODE = mode;
    document.documentElement.dataset.splatthisExecution = mode;
    document.documentElement.dataset.splatthisRenderDone = 'true';
    status.textContent = 'software-rasterized ' + SPLATS.length + ' splats into ' + W + '×' + H + ' ImageData pixels in ' + renderMs.toFixed(0) + 'ms total / ' + computeMs.toFixed(0) + 'ms compute (' + mode + ', ' + (SRGB_IN ? 'srgb' : 'linear') + '-space alpha-over)';
  }

  function renderOnMainThread(reason) {
    const result = rasterizePixelRuntime(canvas, SPLATS, W, H, BG, SRGB_IN);
    finish('main-thread-fallback:' + reason, result.computeMs);
  }

  if (typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined') {
    renderOnMainThread('unsupported');
    return;
  }

  const workerSource = [
    rasterizePixelRuntime.toString(),
    "self.onmessage = function(event) {",
    "  try {",
    "    const p = event.data;",
    "    const target = new OffscreenCanvas(p.width, p.height);",
    "    const result = rasterizePixelRuntime(target, p.splats, p.width, p.height, p.background, p.srgbIn);",
    "    self.postMessage({pixels: result.pixels.buffer, computeMs: result.computeMs}, [result.pixels.buffer]);",
    "  } catch (error) {",
    "    self.postMessage({error: String(error && error.stack || error)});",
    "  }",
    "};"
  ].join('\n');
  const workerUrl = URL.createObjectURL(new Blob([workerSource], {type: 'text/javascript'}));
  const worker = new Worker(workerUrl);
  let workerSettled = false;

  function fallBackFromWorker(reason) {
    if (workerSettled) return;
    workerSettled = true;
    worker.terminate();
    URL.revokeObjectURL(workerUrl);
    renderOnMainThread(reason);
  }

  worker.onerror = function(event) {
    event.preventDefault();
    fallBackFromWorker('worker-error');
  };
  worker.onmessage = function(event) {
    if (event.data && event.data.error) {
      fallBackFromWorker('worker-runtime-error');
      return;
    }
    if (workerSettled) return;
    workerSettled = true;
    // Transfer the exact packed bytes rather than an ImageBitmap. ImageBitmap
    // may receive browser color conversion in transit, while ImageData is the
    // byte-level deployment contract calibrated by this runtime.
    const pixels = new Uint8ClampedArray(event.data.pixels);
    const displayContext = canvas.getContext('2d', { willReadFrequently: false });
    displayContext.putImageData(new ImageData(pixels, W, H), 0, 0);
    worker.terminate();
    URL.revokeObjectURL(workerUrl);
    finish('worker-offscreen', Number(event.data.computeMs));
  };
  worker.postMessage({
    width: W,
    height: H,
    background: BG,
    splats: SPLATS,
    srgbIn: SRGB_IN
  });
})();
""".replace(
            "__W__", str(int(width))
        )
        .replace("__H__", str(int(height)))
        .replace("__BG__", f"[{bg_lin[0]:.6f},{bg_lin[1]:.6f},{bg_lin[2]:.6f}]")
        .replace("__SRGB_IN__", "true" if srgb_mode else "false")
        .replace("__SPLATS__", splats_json)
    )

    safe_title = escape_html(title)
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8"><title>' + safe_title + "</title>\n"
        "<style>\n"
        "  body{margin:0;background:#111;color:#eee;font:14px -apple-system,sans-serif;"
        "display:flex;flex-direction:column;align-items:center;padding:16px}\n"
        "  #c{image-rendering:pixelated;border:1px solid #333;border-radius:6px;max-width:100%}\n"
        "  #status{color:#7fd17f;font-family:ui-monospace,monospace;font-size:12px;margin:8px 0}\n"
        "</style></head>\n"
        "<body>\n"
        f'<canvas id="c" data-compositor="pixel-runtime" data-execution="pending" '
        f'width="{int(width)}" height="{int(height)}"></canvas>\n'
        '<div id="status">rendering...</div>\n'
        "<script>\n" + js + "\n</script>\n"
        "</body></html>\n"
    )


def generate_webgl_pixel_runtime_html(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    title: str = "SplatThis Accelerated Pixel Runtime",
    compositing_space: str = "linear",
    backend: str = "auto",
) -> str:
    """Emit the accelerated pixel runtime with exact CPU fallbacks.

    The browser tries RGBA32F WebGL2, then RGBA16F WebGL2, then the exact
    Worker/OffscreenCanvas software renderer, and finally the same exact
    renderer on the main thread. GPU paths render on a temporary canvas and
    transfer packed bytes to the visible 2D canvas, so failed contexts never
    prevent a later fallback from acquiring its required context.
    """
    import json

    from .renderer import prepare_pixel_runtime_data

    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("Pixel runtime width and height must be positive")
    normalized_backend = str(backend).strip().lower()
    allowed_backends = {"auto", "rgba32f", "rgba16f", "worker", "main"}
    if normalized_backend not in allowed_backends:
        raise ValueError(
            f"Unsupported pixel runtime backend {backend!r}; "
            f"expected one of {sorted(allowed_backends)}"
        )

    rows, serialized_background, srgb_mode = prepare_pixel_runtime_data(
        splats,
        background_linear_rgb=background_linear_rgb,
        compositing_space=compositing_space,
    )
    bg = [float(channel) for channel in serialized_background]
    splats_json = json.dumps(rows, separators=(",", ":"))
    safe_title = escape_html(title)
    js = (
        r"""
(function(){
  const started = performance.now();
  const W = __W__, H = __H__;
  const BG = __BG__;
  const SPLATS = __SPLATS__;
  const SRGB_IN = __SRGB_IN__;
  const canvas = document.getElementById('c');
  const status = document.getElementById('status');
  const configuredBackend = __BACKEND__;
  const forcedBackend = new URLSearchParams(location.search).get('splatthisPixelBackend') ||
    (configuredBackend==='auto' ? '' : configuredBackend);
  const failures = [];
  let finished = false;
  let gpuQuality = null;

  function finish(mode, computeMs, pixels) {
    if (finished) return;
    finished = true;
    if (pixels) {
      const displayContext = canvas.getContext('2d', {willReadFrequently:false});
      displayContext.putImageData(new ImageData(pixels, W, H), 0, 0);
    }
    const renderMs = performance.now() - started;
    canvas.dataset.execution = mode;
    window.__SPLATTHIS_COMPUTE_MS = computeMs;
    window.__SPLATTHIS_RENDER_MS = renderMs;
    window.__SPLATTHIS_RENDER_MODE = mode;
    window.__SPLATTHIS_FAST_PATH_FAILURES = failures.slice();
    window.__SPLATTHIS_GPU_QUALITY = gpuQuality;
    document.documentElement.dataset.splatthisExecution = mode;
    document.documentElement.dataset.splatthisRenderDone = 'true';
    status.textContent = 'rendered ' + SPLATS.length + ' splats at ' + W + '×' + H +
      ' in ' + renderMs.toFixed(1) + 'ms total / ' + computeMs.toFixed(1) +
      'ms compute (' + mode + ', ' + (SRGB_IN ? 'srgb' : 'linear') + ' alpha-over)';
  }

  function rasterizePixelRuntime(targetCanvas, splats, width, height, background, srgbIn) {
    const computeStarted = performance.now();
    const FOOTPRINT = 3.0;
    const ctx = targetCanvas.getContext('2d', {willReadFrequently:false});
    const lin = new Float32Array(width * height * 3);
    const transmittance = new Float32Array(width * height).fill(1);
    for (let si=0; si<splats.length; si++) {
      const s=splats[si];
      const x=s[0], y=s[1];
      const sx=Math.max(s[2],1e-4), sy=Math.max(s[3],1e-4);
      const theta=s[4], r=s[5], g=s[6], b=s[7];
      const alpha=Math.min(1,Math.max(0,s[8]));
      const ct=Math.cos(theta), st=Math.sin(theta);
      const rx=Math.max(1,Math.ceil(FOOTPRINT*Math.sqrt((sx*ct)*(sx*ct)+(sy*st)*(sy*st))));
      const ry=Math.max(1,Math.ceil(FOOTPRINT*Math.sqrt((sx*st)*(sx*st)+(sy*ct)*(sy*ct))));
      const x0=Math.max(0,Math.floor(x-rx));
      const x1=Math.min(width,Math.ceil(x+rx+1));
      const y0=Math.max(0,Math.floor(y-ry));
      const y1=Math.min(height,Math.ceil(y+ry+1));
      const invSx2=1/(sx*sx), invSy2=1/(sy*sy);
      for (let py=y0; py<y1; py++) {
        const row=py*width;
        for (let px=x0; px<x1; px++) {
          const dx=px-x, dy=py-y;
          const u=ct*dx+st*dy, v=-st*dx+ct*dy;
          const weight=Math.exp(-0.5*(u*u*invSx2+v*v*invSy2));
          const layerAlpha=1-Math.exp(-alpha*weight);
          const index=row+px, t=transmittance[index], contribution=t*layerAlpha;
          const target=index*3;
          lin[target]+=contribution*r;
          lin[target+1]+=contribution*g;
          lin[target+2]+=contribution*b;
          transmittance[index]=t*(1-layerAlpha);
        }
      }
    }
    const image=ctx.createImageData(width,height), out=image.data;
    const threshold=0.0031308;
    for (let index=0; index<width*height; index++) {
      const source=index*3, target=index*4, t=transmittance[index];
      let r=lin[source]+t*background[0];
      let g=lin[source+1]+t*background[1];
      let b=lin[source+2]+t*background[2];
      r=Math.min(1,Math.max(0,r));
      g=Math.min(1,Math.max(0,g));
      b=Math.min(1,Math.max(0,b));
      if (!srgbIn) {
        r=r<=threshold ? 12.92*r : 1.055*Math.pow(r,1/2.4)-0.055;
        g=g<=threshold ? 12.92*g : 1.055*Math.pow(g,1/2.4)-0.055;
        b=b<=threshold ? 12.92*b : 1.055*Math.pow(b,1/2.4)-0.055;
      }
      out[target]=(r*255+0.5)|0;
      out[target+1]=(g*255+0.5)|0;
      out[target+2]=(b*255+0.5)|0;
      out[target+3]=255;
    }
    ctx.putImageData(image,0,0);
    return {computeMs:performance.now()-computeStarted,pixels:image.data};
  }

  function exactPixelBytes(px,py) {
    const FOOTPRINT=3.0;
    let r=0, g=0, b=0, transmittance=1;
    for (let index=0; index<SPLATS.length; index++) {
      const s=SPLATS[index];
      const x=s[0], y=s[1];
      const sx=Math.max(s[2],1e-4), sy=Math.max(s[3],1e-4);
      const ct=Math.cos(s[4]), st=Math.sin(s[4]);
      const rx=Math.max(1,Math.ceil(FOOTPRINT*Math.sqrt((sx*ct)*(sx*ct)+(sy*st)*(sy*st))));
      const ry=Math.max(1,Math.ceil(FOOTPRINT*Math.sqrt((sx*st)*(sx*st)+(sy*ct)*(sy*ct))));
      const x0=Math.max(0,Math.floor(x-rx));
      const x1=Math.min(W,Math.ceil(x+rx+1));
      const y0=Math.max(0,Math.floor(y-ry));
      const y1=Math.min(H,Math.ceil(y+ry+1));
      if (px<x0 || px>=x1 || py<y0 || py>=y1) continue;
      const dx=px-x, dy=py-y;
      const u=ct*dx+st*dy, v=-st*dx+ct*dy;
      const weight=Math.exp(-0.5*(u*u/(sx*sx)+v*v/(sy*sy)));
      const layerAlpha=1-Math.exp(-Math.min(1,Math.max(0,s[8]))*weight);
      const contribution=transmittance*layerAlpha;
      r+=contribution*s[5]; g+=contribution*s[6]; b+=contribution*s[7];
      transmittance*=1-layerAlpha;
    }
    r=Math.min(1,Math.max(0,r+transmittance*BG[0]));
    g=Math.min(1,Math.max(0,g+transmittance*BG[1]));
    b=Math.min(1,Math.max(0,b+transmittance*BG[2]));
    if (!SRGB_IN) {
      const threshold=0.0031308;
      r=r<=threshold ? 12.92*r : 1.055*Math.pow(r,1/2.4)-0.055;
      g=g<=threshold ? 12.92*g : 1.055*Math.pow(g,1/2.4)-0.055;
      b=b<=threshold ? 12.92*b : 1.055*Math.pow(b,1/2.4)-0.055;
    }
    return [(r*255+0.5)|0,(g*255+0.5)|0,(b*255+0.5)|0];
  }

  function checkHalfFloatQuality(pixels) {
    // A fixed 4x4 grid is cheap (16 * splat count) and detects broken or
    // excessively lossy half-float accumulation without paying for a second
    // full CPU frame. Normal IEEE binary16 drift is allowed up to two bytes.
    let channels=0, totalError=0, maxError=0;
    for (let gy=0; gy<4; gy++) {
      const py=Math.min(H-1,Math.floor((gy+0.5)*H/4));
      for (let gx=0; gx<4; gx++) {
        const px=Math.min(W-1,Math.floor((gx+0.5)*W/4));
        const expected=exactPixelBytes(px,py);
        const offset=(py*W+px)*4;
        for (let channel=0; channel<3; channel++) {
          const error=Math.abs(pixels[offset+channel]-expected[channel]);
          totalError+=error; maxError=Math.max(maxError,error); channels++;
        }
      }
    }
    const meanError=totalError/channels;
    return {
      samples:16,
      maxByteError:maxError,
      meanAbsByteError:meanError,
      accepted:maxError<=2 && meanError<=0.5
    };
  }

  function renderOnMainThread(reason) {
    failures.push(reason);
    const result=rasterizePixelRuntime(canvas,SPLATS,W,H,BG,SRGB_IN);
    finish('main-thread-fallback',result.computeMs,null);
  }

  function renderInWorker(reason) {
    if (reason) failures.push(reason);
    if (typeof Worker==='undefined' || typeof OffscreenCanvas==='undefined') {
      renderOnMainThread('worker-or-offscreen-unsupported');
      return;
    }
    const workerSource=[
      rasterizePixelRuntime.toString(),
      "self.onmessage=function(event){try{const p=event.data;const target=new OffscreenCanvas(p.width,p.height);const result=rasterizePixelRuntime(target,p.splats,p.width,p.height,p.background,p.srgbIn);self.postMessage({pixels:result.pixels.buffer,computeMs:result.computeMs},[result.pixels.buffer]);}catch(error){self.postMessage({error:String(error&&error.stack||error)});}};"
    ].join('\n');
    let workerUrl;
    let worker;
    try {
      workerUrl=URL.createObjectURL(new Blob([workerSource],{type:'text/javascript'}));
      worker=new Worker(workerUrl);
    } catch (error) {
      if (workerUrl) URL.revokeObjectURL(workerUrl);
      renderOnMainThread('worker-construction-failed');
      return;
    }
    let settled=false;
    function workerFailed(message) {
      if (settled) return;
      settled=true;
      failures.push(message);
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      renderOnMainThread('worker-failed');
    }
    worker.onerror=function(event){event.preventDefault();workerFailed('worker-error');};
    worker.onmessage=function(event){
      if (event.data&&event.data.error) {workerFailed('worker-runtime-error');return;}
      if (settled) return;
      settled=true;
      const pixels=new Uint8ClampedArray(event.data.pixels);
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      finish('worker-offscreen',Number(event.data.computeMs),pixels);
    };
    worker.postMessage({width:W,height:H,background:BG,splats:SPLATS,srgbIn:SRGB_IN});
  }

  function shader(gl, type, source) {
      const value = gl.createShader(type);
      gl.shaderSource(value, source);
      gl.compileShader(value);
      if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(value) || 'shader compile failed');
      }
      return value;
  }

  function program(gl, vertexSource, fragmentSource) {
      const value = gl.createProgram();
      gl.attachShader(value, shader(gl,gl.VERTEX_SHADER,vertexSource));
      gl.attachShader(value, shader(gl,gl.FRAGMENT_SHADER,fragmentSource));
      gl.linkProgram(value);
      if (!gl.getProgramParameter(value, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(value) || 'program link failed');
      }
      return value;
  }

    const splatVertex = `#version 300 es
precision highp float;
layout(location=0) in vec4 aRect;
layout(location=1) in vec4 aGeometry;
layout(location=2) in vec2 aRotation;
layout(location=3) in vec4 aColorAlpha;
uniform vec2 uSize;
flat out vec4 vGeometry;
flat out vec2 vRotation;
flat out vec4 vColorAlpha;
void main() {
  vec2 corner;
  if (gl_VertexID == 0) corner=vec2(0.0,0.0);
  else if (gl_VertexID == 1) corner=vec2(1.0,0.0);
  else if (gl_VertexID == 2) corner=vec2(0.0,1.0);
  else if (gl_VertexID == 3) corner=vec2(0.0,1.0);
  else if (gl_VertexID == 4) corner=vec2(1.0,0.0);
  else corner=vec2(1.0,1.0);
  vec2 pixel=mix(aRect.xy,aRect.zw,corner);
  vec2 ndc=vec2(pixel.x/uSize.x*2.0-1.0,1.0-pixel.y/uSize.y*2.0);
  gl_Position=vec4(ndc,0.0,1.0);
  vGeometry=aGeometry;
  vRotation=aRotation;
  vColorAlpha=aColorAlpha;
}`;
    const splatFragment = `#version 300 es
precision highp float;
uniform vec2 uSize;
flat in vec4 vGeometry;
flat in vec2 vRotation;
flat in vec4 vColorAlpha;
out vec4 outputColor;
void main() {
  vec2 pixel=vec2(gl_FragCoord.x-0.5,uSize.y-gl_FragCoord.y-0.5);
  vec2 delta=pixel-vGeometry.xy;
  float u=vRotation.x*delta.x+vRotation.y*delta.y;
  float v=-vRotation.y*delta.x+vRotation.x*delta.y;
  float q=u*u*vGeometry.z+v*v*vGeometry.w;
  float weight=exp(-0.5*q);
  float layerAlpha=1.0-exp(-vColorAlpha.a*weight);
  outputColor=vec4(vColorAlpha.rgb*layerAlpha,layerAlpha);
}`;
    const displayVertex = `#version 300 es
precision highp float;
void main() {
  vec2 p=gl_VertexID==0 ? vec2(-1.0,-1.0) :
         (gl_VertexID==1 ? vec2(3.0,-1.0) : vec2(-1.0,3.0));
  gl_Position=vec4(p,0.0,1.0);
}`;
    const displayFragment = `#version 300 es
precision highp float;
uniform sampler2D uAccum;
uniform vec3 uBackground;
uniform bool uSrgbIn;
out vec4 outputColor;
vec3 encodeSrgb(vec3 value) {
  bvec3 low=lessThanEqual(value,vec3(0.0031308));
  vec3 lo=12.92*value;
  vec3 hi=1.055*pow(value,vec3(1.0/2.4))-0.055;
  return mix(hi,lo,low);
}
void main() {
  ivec2 coordinate=ivec2(gl_FragCoord.xy);
  vec4 accum=texelFetch(uAccum,coordinate,0);
  vec3 color=clamp(accum.rgb+accum.a*uBackground,0.0,1.0);
  outputColor=vec4(uSrgbIn ? color : encodeSrgb(color),1.0);
}`;

  const instance = new Float32Array(SPLATS.length * 14);
    const FOOTPRINT = 3.0;
    for (let index=0; index<SPLATS.length; index++) {
      const s=SPLATS[index];
      const x=s[0], y=s[1];
      const sx=Math.max(s[2],1e-4), sy=Math.max(s[3],1e-4);
      const ct=Math.cos(s[4]), st=Math.sin(s[4]);
      const rx=Math.max(1,Math.ceil(FOOTPRINT*Math.sqrt((sx*ct)*(sx*ct)+(sy*st)*(sy*st))));
      const ry=Math.max(1,Math.ceil(FOOTPRINT*Math.sqrt((sx*st)*(sx*st)+(sy*ct)*(sy*ct))));
      const base=index*14;
      instance[base]=Math.max(0,Math.floor(x-rx));
      instance[base+1]=Math.max(0,Math.floor(y-ry));
      instance[base+2]=Math.min(W,Math.ceil(x+rx+1));
      instance[base+3]=Math.min(H,Math.ceil(y+ry+1));
      instance[base+4]=x; instance[base+5]=y;
      instance[base+6]=1/(sx*sx); instance[base+7]=1/(sy*sy);
      instance[base+8]=ct; instance[base+9]=st;
      instance[base+10]=s[5]; instance[base+11]=s[6]; instance[base+12]=s[7];
      instance[base+13]=Math.min(1,Math.max(0,s[8]));
    }

  function tryWebGL(format) {
    const computeStarted=performance.now();
    const surface=document.createElement('canvas');
    surface.width=W; surface.height=H;
    let contextLost=false;
    surface.addEventListener('webglcontextlost',function(event){
      event.preventDefault(); contextLost=true;
    });
    const gl=surface.getContext('webgl2',{
      alpha:false,antialias:false,depth:false,stencil:false,
      premultipliedAlpha:false,preserveDrawingBuffer:true
    });
    if (!gl) {failures.push(format+':webgl2-unavailable');return null;}
    const colorFloat=gl.getExtension('EXT_color_buffer_float');
    let internalFormat, pixelType, mode;
    if (format==='rgba32f') {
      if (!colorFloat || !gl.getExtension('EXT_float_blend')) {
        failures.push('rgba32f:extensions-unavailable'); return null;
      }
      internalFormat=gl.RGBA32F; pixelType=gl.FLOAT; mode='webgl2-rgba32f';
    } else {
      // Attempt RGBA16F directly; framebuffer completeness is the
      // authoritative capability probe across WebGL2 implementations.
      internalFormat=gl.RGBA16F; pixelType=gl.HALF_FLOAT; mode='webgl2-rgba16f';
    }
    try {
    const accumulationTexture=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,accumulationTexture);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D,0,internalFormat,W,H,0,gl.RGBA,pixelType,null);
    const framebuffer=gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER,framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,accumulationTexture,0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('float framebuffer incomplete');
    }

    const splatProgram=program(gl,splatVertex,splatFragment);
    gl.useProgram(splatProgram);
    gl.uniform2f(gl.getUniformLocation(splatProgram,'uSize'),W,H);
    const buffer=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    gl.bufferData(gl.ARRAY_BUFFER,instance,gl.STATIC_DRAW);
    const stride=14*4;
    const attributes=[[0,4,0],[1,4,4],[2,2,8],[3,4,10]];
    for (const [location,size,offset] of attributes) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location,size,gl.FLOAT,false,stride,offset*4);
      gl.vertexAttribDivisor(location,1);
    }
    gl.viewport(0,0,W,H);
    gl.clearColor(0,0,0,1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFuncSeparate(gl.DST_ALPHA,gl.ONE,gl.ZERO,gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLES,0,6,SPLATS.length);

    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    gl.disable(gl.BLEND);
    const displayProgram=program(gl,displayVertex,displayFragment);
    gl.useProgram(displayProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D,accumulationTexture);
    gl.uniform1i(gl.getUniformLocation(displayProgram,'uAccum'),0);
    gl.uniform3f(gl.getUniformLocation(displayProgram,'uBackground'),BG[0],BG[1],BG[2]);
    gl.uniform1i(gl.getUniformLocation(displayProgram,'uSrgbIn'),SRGB_IN ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES,0,3);
    gl.finish();
    if (contextLost || gl.isContextLost()) throw new Error('context-lost');
    const bottomUp=new Uint8Array(W*H*4);
    gl.readPixels(0,0,W,H,gl.RGBA,gl.UNSIGNED_BYTE,bottomUp);
    if (gl.getError()!==gl.NO_ERROR) throw new Error('readPixels-failed');
    const pixels=new Uint8ClampedArray(bottomUp.length), rowBytes=W*4;
    for (let row=0; row<H; row++) {
      const source=(H-1-row)*rowBytes;
      pixels.set(bottomUp.subarray(source,source+rowBytes),row*rowBytes);
    }
    if (format==='rgba16f') {
      gpuQuality=checkHalfFloatQuality(pixels);
      if (!gpuQuality.accepted) {
        throw new Error('half-float-quality-gate-failed:'+
          gpuQuality.maxByteError+'/'+gpuQuality.meanAbsByteError.toFixed(3));
      }
    }
    return {mode:mode,pixels:pixels,computeMs:performance.now()-computeStarted};
    } catch (error) {
      failures.push(format+':'+String(error&&error.message||error));
      return null;
    }
  }

  if (forcedBackend==='main') {renderOnMainThread('forced-main');return;}
  if (forcedBackend==='worker') {renderInWorker('forced-worker');return;}
  let result=null;
  if (forcedBackend==='rgba16f') result=tryWebGL('rgba16f');
  else if (forcedBackend==='rgba32f') result=tryWebGL('rgba32f');
  else result=tryWebGL('rgba32f') || tryWebGL('rgba16f');
  if (result) finish(result.mode,result.computeMs,result.pixels);
  else renderInWorker('webgl-fast-paths-unavailable');
})();
""".replace(
            "__W__", str(int(width))
        )
        .replace("__H__", str(int(height)))
        .replace("__BG__", f"[{bg[0]:.6f},{bg[1]:.6f},{bg[2]:.6f}]")
        .replace("__SRGB_IN__", "true" if srgb_mode else "false")
        .replace("__BACKEND__", json.dumps(normalized_backend))
        .replace("__SPLATS__", splats_json)
    )
    return (
        "<!doctype html>\n"
        f'<html><head><meta charset="utf-8"><title>{safe_title}</title>\n'
        "<style>html,body{margin:0;background:#111;color:#eee;font:14px "
        "-apple-system,sans-serif}body{display:flex;flex-direction:column;"
        "align-items:center;padding:16px}#c{border:1px solid #333;max-width:100%}"
        "#status{color:#7fd17f;font:12px ui-monospace,monospace;margin:8px}</style>"
        "</head><body>"
        f'<canvas id="c" data-compositor="pixel-runtime" '
        f'data-execution="pending" width="{int(width)}" height="{int(height)}"></canvas>'
        '<div id="status">rendering pixel runtime...</div>'
        f"<script>{js}</script></body></html>\n"
    )
