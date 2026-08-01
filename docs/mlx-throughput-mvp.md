# MLX throughput MVP

**Date:** 2026-08-01

**Environment:** Apple Silicon, MLX 0.31.2, Python 3.13.14

**Evidence:** [`data/mlx-batch-tile-mvp.json`](../data/mlx-batch-tile-mvp.json)

## Question

Can corpus parallelism, caching, or larger MLX tile batches shorten runs without
quietly weakening the seeded artifact contract?

## Concurrent corpus processes

Two small full-frame pixel-runtime jobs were measured serially and with two MLX
subprocesses. Shared-Metal execution reduced wall time by 29–31%, but final raw
splat parameters no longer matched the serial seed-identical runs. The largest
observed parameter difference was about 0.002. This is small visually but makes
seeded corpus evidence irreproducible.

`tools/corpus_benchmark.py` therefore permits `--jobs N` for Torch/CPU and
rejects `--jobs > 1` for MLX. JSONL writes and scoring remain single-writer.

## Exact run cache

Corpus identity already includes source bytes, configuration, and executable
code. Cache validation now additionally requires a successful record and a
primary artifact matching its recorded size and, for new records, SHA-256. A
post-change smoke run recorded the digest and skipped an unchanged conversion
in 0.89 seconds. Failed, corrupted, and partially deleted runs are retried.

## MLX tile-batch sweep

The governing sweep used all 21 full source frames, seed 0, the explicit `fast`
profile, 512 requested splats, one 100-iteration stage, and pixel-runtime output.
It compared 8 versus 32 tiles per MLX render batch.

- Optimizer time improved on 21/21 images; median change was −42.2%.
- Internal pipeline wall time improved on 21/21; median change was −28.5%.
- External process wall time improved by a median 22.8%.
- Median ΔSSIM was +0.000013, but results moved both ways.
- Maximum absolute ΔSSIM was 0.00253 and maximum absolute ΔPSNR was 0.315 dB.

The batch size changes floating-point grouping and therefore the optimization
trajectory. It is not a behavior-neutral kernel setting. Batch 32 is adopted
only for the explicitly speed-oriented `fast` profile. `balanced` and
`max-fidelity` retain batch 8 until their own deployed-artifact corpus gates
justify a change. These are low-budget internal metrics, not browser or
PowerPoint fidelity claims.
