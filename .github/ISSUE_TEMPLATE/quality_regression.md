---
name: Quality regression
about: Output got measurably worse, or a deployed artifact looks wrong
labels: quality
---

## What got worse

<!-- Which output format, and against which version or commit. -->

## Measured, not eyeballed

<!-- This project grades artifacts in their governing renderer: Chromium for
     SVG/CSS/Canvas/pixel-runtime, real PowerPoint for decks. Internal proxy
     metrics and librsvg/CairoSVG renders cannot support a fidelity claim.

     If you have numbers, give SSIM_sRGB or LPIPS and say how you obtained
     them. If you do not, attach the artifact and the source image and say what
     looks wrong — that is genuinely useful too. -->

| | before | after |
|---|---|---|
| SSIM_sRGB | | |
| LPIPS | | |
| splat count | | |

## Artifacts

<!-- The source image, and the output from each version. -->

## Note on LPIPS vs SSIM

<!-- SSIM systematically over-rewards the blur recipe's smoothness; LPIPS
     reverses several apparent "blur wins". If the two disagree, LPIPS is the
     one this project trusts. -->
