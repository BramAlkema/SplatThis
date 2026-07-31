# WebGL2 Pixel-Runtime MVP

**Status:** integrated; 21-image Chrome gate passed, cross-browser qualification pending
**Date:** 2026-07-31

## Question

Can the exact CPU `pixel-runtime` formula move to the GPU without silently
changing the visible result?

## Implementation

`generate_webgl_pixel_runtime_html` submits one bounded rectangle per splat
with WebGL2 instancing. A fragment shader evaluates
`1 - exp(-alpha * exp(-0.5 * q))`. A float framebuffer stores accumulated
premultiplied color and remaining transmittance. Float blending implements the
same front-to-back recurrence as the CPU runtime:

```text
color += transmittance * layer_alpha * splat_color
transmittance *= 1 - layer_alpha
```

A final shader adds the background and performs the optional linear-to-sRGB
encoding. The emitted artifact selects:

1. RGBA32F WebGL2 with `EXT_color_buffer_float` and `EXT_float_blend`;
2. RGBA16F WebGL2 when half-float rendering is available and a fixed 4 x 4
   sample remains within two bytes maximum and 0.5 byte mean absolute error
   of the exact formula;
3. the exact rasterizer in a Worker with OffscreenCanvas; or
4. the same exact rasterizer on the main thread.

GPU work happens on a temporary canvas. The packed result is copied into the
visible 2D canvas, so a failed or lost WebGL context does not consume the
visible canvas context needed by a later fallback. The selected backend,
failure reasons, quality sample, compute time, and total time are exposed as
runtime metadata. `?splatthisPixelBackend=rgba32f|rgba16f|worker|main` can force
each backend for diagnostics.

The governing Playwright capture waits for `splatthisRenderDone`, reads the
runtime's own canvas PNG, records the selected backend in the renderer label,
and uses that frame for export acceptance. It does not substitute an internal
proxy when Chromium is unavailable.

## Chameleon result

The integrated-path test uses a 1,788-splat pixel-runtime checkpoint at
476 x 502 in Chrome 140.0.7339.81. Seven fresh navigations followed two warm-up
navigations per backend.

| Runtime | Median completion | Pixel relation to CPU |
|---|---:|---|
| Main-thread CPU baseline | 126.8 ms | Reference |
| Worker plus OffscreenCanvas | 133.3 ms | Byte-identical |
| WebGL2 RGBA32F blend | 18.5 ms | Six pixels differ by one 8-bit value |
| WebGL2 RGBA16F blend | about 20 ms including sample gate | Maximum two-byte error on this frame |

RGBA32F versus CPU scored SSIM_sRGB 0.99999995 and PSNR_sRGB 98.90 dB.
RGBA16F versus CPU scored SSIM_sRGB 0.99823 and PSNR_sRGB 54.08 dB; versus the
source, its SSIM_sRGB was 0.90486 rather than 0.90494. The Worker improves UI
responsiveness but not end-to-end latency; WebGL2 is the measured speed path.
The combined fast-path and fallback code added 14.4 KB to this self-contained
HTML file, an increase of 4.7% over the earlier Worker/main-only artifact.

## Chrome corpus gate

The stored seed-0 historical Canvas (now pixel-runtime) checkpoints for all 21
corpus images were replayed at their complete native benchmark dimensions.
Each backend received one warm-up and one measured navigation in the same
Chrome 140 process.

| Check | Result |
|---|---:|
| 32F selected | 21 / 21 |
| 32F maximum error versus exact CPU | 1 byte |
| 32F worst source SSIM_sRGB delta | -0.0000014 |
| 16F accepted by runtime sample | 20 / 21 |
| 16F worst source SSIM_sRGB delta, including fallback | -0.000337 |
| Worker versus main CPU | byte-identical on 21 / 21 |
| Median total 32F / 16F / Worker / main | 19.1 / 21.5 / 112.5 / 93.6 ms |

Checkerboard exceeded the half-float sample's mean-error threshold and
automatically used exact Worker CPU. On the accepted images, the largest
full-frame 16F error was three bytes even though the sampled maximum was at
most two. The sample is therefore a rejection heuristic, not a proof of a
global per-pixel bound. The deployed source-metric gate is the final authority.

## Verdict

The chain is safe to emit because every acceleration failure has an exact CPU
fallback and final acceptance captures the selected Chrome canvas buffer. It
does not yet establish portable GPU equivalence. Remaining qualification is:

- current Safari, Firefox, and Edge behavior where each path is supported;
- memory and first-load warm-up reporting; and
- confirmation that the 4 x 4 half-float sample remains conservative across
  other GPUs and browser versions.

The versioned aggregate is
[`data/pixel-runtime-webgl-corpus.json`](../data/pixel-runtime-webgl-corpus.json).
Local detailed measurements are under `./tmp/pixel-runtime-chain/`.
