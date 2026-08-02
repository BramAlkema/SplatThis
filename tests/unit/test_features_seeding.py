"""Seeded reproducibility of the seeding stage.

The README publishes that a fixed ``--seed`` reproduces the reported metrics
on both backends. That guarantee starts here, in the numpy seeding path;
these tests pin determinism and the geometric contracts (in-bounds seeds,
Poisson minimum spacing) that the fitting stages assume.
"""

from __future__ import annotations

import numpy as np

from splatthis.features import init_seeds_content_adaptive, poisson_disk_sampling


def _image() -> np.ndarray:
    rng = np.random.default_rng(123)
    image = (rng.random((32, 32, 3)) * 0.2).astype(np.float32)
    image[8:24, 8:24] = 0.9  # a bright square, so gradients exist to attract seeds
    return image


def test_same_rng_seed_reproduces_content_adaptive_seeds() -> None:
    first = init_seeds_content_adaptive(_image(), 40, rng=np.random.default_rng(7))
    second = init_seeds_content_adaptive(_image(), 40, rng=np.random.default_rng(7))
    assert first == second
    other = init_seeds_content_adaptive(_image(), 40, rng=np.random.default_rng(8))
    assert first != other


def test_content_adaptive_seeds_lie_inside_the_image() -> None:
    seeds = init_seeds_content_adaptive(_image(), 40, rng=np.random.default_rng(7))
    assert len(seeds) == 40
    for x, y in seeds:
        assert 0 <= x < 32
        assert 0 <= y < 32


def test_poisson_disk_is_deterministic_and_respects_min_distance() -> None:
    first = poisson_disk_sampling(48, 40, 6.0, rng=np.random.default_rng(7))
    second = poisson_disk_sampling(48, 40, 6.0, rng=np.random.default_rng(7))
    assert first == second

    points = np.asarray(first, dtype=np.float64)
    assert len(points) >= 2, "a 48x40 field at spacing 6 must fit several points"
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 6.0 - 1e-6

    for x, y in first:
        assert 0 <= x < 48
        assert 0 <= y < 40
