import numpy as np

from splatthis.features import (
    compute_gradient_magnitude,
    compute_structure_field,
    edge_tangent_angle,
)


def test_gradient_and_structure_maps_need_only_numpy():
    ramp = np.tile(np.linspace(0, 1, 24, dtype=np.float32), (16, 1))
    image = np.repeat(ramp[..., None], 3, axis=2)

    magnitude = compute_gradient_magnitude(image)
    directions, anisotropy = compute_structure_field(image)

    assert magnitude.shape == ramp.shape
    assert directions.shape == (*ramp.shape, 2)
    assert anisotropy.shape == ramp.shape
    assert float(np.mean(magnitude[:, 2:-2])) > 0
    assert np.all(anisotropy >= 1)


def test_splat_major_axis_follows_edge_tangent():
    gradient_direction = np.array([1.0, 0.0], dtype=np.float32)
    assert np.isclose(edge_tangent_angle(gradient_direction), np.pi / 2)
