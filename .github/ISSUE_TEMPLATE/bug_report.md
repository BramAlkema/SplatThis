---
name: Bug report
about: Something produced the wrong result, crashed, or refused to run
labels: bug
---

## What happened

<!-- The observed behaviour. If it is a quality problem, say which output
     format and attach the artifact — an SVG or PPTX is far more useful than a
     screenshot of one. -->

## The command

```bash
splatthis ... 
```

## Environment

- `splatthis --version`:
- OS and CPU (e.g. macOS 26 arm64, Ubuntu 24.04 x86_64):
- Backend: `mlx` (default on Apple Silicon) or `torch`
- Chrome installed, and was the `capture` extra used?

## Run manifest

<!-- Re-run with `--artifacts-dir ./debug-run` and attach
     `debug-run/run_manifest.json`. It records the config fingerprint, the
     renderer identity, the acceptance decision, and whether the host-memory
     guard reduced the population — which is usually the answer. -->

## Reproducibility

<!-- Does it happen every time? Note that only `--optimizer-backend torch`
     produces byte-identical output under a fixed seed; MLX orders float32
     reductions on the Metal device nondeterministically. If you are comparing
     two runs byte-for-byte, pin torch first. -->
