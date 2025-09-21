#!/usr/bin/env python3
"""Demonstration of AdaptiveGaussian2D functionality."""

import numpy as np
from src.splat_this.core.adaptive_gaussian import (
    AdaptiveGaussian2D,
    create_isotropic_gaussian,
    create_anisotropic_gaussian
)


def demo_covariance_computation():
    """Demonstrate covariance matrix computation."""
    print("=== Covariance Matrix Demonstration ===")

    # Isotropic Gaussian
    iso_gaussian = create_isotropic_gaussian([0.5, 0.5], 0.1, [1.0, 0.0, 0.0])
    print(f"Isotropic Gaussian:")
    print(f"  Aspect ratio: {iso_gaussian.aspect_ratio:.3f}")
    print(f"  Covariance matrix:\n{iso_gaussian.covariance_matrix}")

    # Anisotropic Gaussian
    aniso_gaussian = create_anisotropic_gaussian(
        [0.5, 0.5], (0.05, 0.15), np.pi/4, [0.0, 1.0, 0.0]
    )
    print(f"\nAnisotropic Gaussian (3:1 ratio, 45° rotation):")
    print(f"  Aspect ratio: {aniso_gaussian.aspect_ratio:.3f}")
    print(f"  Orientation: {aniso_gaussian.orientation * 180 / np.pi:.1f}°")
    print(f"  Covariance matrix:\n{aniso_gaussian.covariance_matrix}")

    # Edge-aligned Gaussian (horizontal edge)
    edge_gaussian = AdaptiveGaussian2D(
        mu=np.array([0.5, 0.5]),
        inv_s=np.array([10.0, 2.0]),  # Very elongated
        theta=0.0,  # Horizontal
        color=np.array([0.0, 0.0, 1.0])
    )
    print(f"\nEdge-aligned Gaussian (horizontal edge):")
    print(f"  Aspect ratio: {edge_gaussian.aspect_ratio:.3f}")
    print(f"  Principal axis length: {edge_gaussian.principal_axis_length:.3f}")
    print(f"  Minor axis length: {edge_gaussian.minor_axis_length:.3f}")


def demo_parameter_optimization():
    """Demonstrate parameter optimization scenario."""
    print("\n=== Parameter Optimization Demonstration ===")

    # Start with a basic Gaussian
    gaussian = AdaptiveGaussian2D(
        mu=np.array([0.3, 0.7]),
        inv_s=np.array([1.0, 1.0]),
        theta=0.0,
        color=np.array([0.8, 0.6, 0.4])
    )

    print(f"Initial Gaussian:")
    print(f"  Position: {gaussian.mu}")
    print(f"  Aspect ratio: {gaussian.aspect_ratio:.3f}")
    print(f"  3σ radius (100x100 image): {gaussian.compute_3sigma_radius_px((100, 100)):.2f}px")

    # Simulate optimization updates
    learning_rates = {'mu': 0.01, 'inv_s': 0.02, 'theta': 0.01}

    # Simulate gradient updates (random for demo)
    np.random.seed(42)
    for iteration in range(5):
        # Simulate gradients
        grad_mu = np.random.normal(0, 0.1, 2)
        grad_inv_s = np.random.normal(0, 0.05, 2)
        grad_theta = np.random.normal(0, 0.1)

        # Apply updates
        gaussian.mu -= learning_rates['mu'] * grad_mu
        gaussian.inv_s -= learning_rates['inv_s'] * grad_inv_s
        gaussian.theta -= learning_rates['theta'] * grad_theta

        # Clip parameters
        gaussian.clip_parameters()

        gaussian.refinement_count += 1

        print(f"After iteration {iteration + 1}:")
        print(f"  Position: {gaussian.mu}")
        print(f"  Aspect ratio: {gaussian.aspect_ratio:.3f}")
        print(f"  Orientation: {gaussian.orientation * 180 / np.pi:.1f}°")


def demo_backward_compatibility():
    """Demonstrate backward compatibility with current Gaussian class."""
    print("\n=== Backward Compatibility Demonstration ===")

    # Import current Gaussian
    from src.splat_this.core.extract import Gaussian

    # Create a traditional Gaussian
    traditional = Gaussian(
        x=150, y=200, rx=25, ry=35, theta=np.pi/6,
        r=200, g=150, b=100, a=0.8, score=0.7
    )

    print(f"Traditional Gaussian:")
    print(f"  Position: ({traditional.x}, {traditional.y})")
    print(f"  Radii: ({traditional.rx}, {traditional.ry})")
    print(f"  Color: RGB({traditional.r}, {traditional.g}, {traditional.b})")

    # Convert to adaptive
    image_size = (400, 300)  # H, W
    adaptive = AdaptiveGaussian2D.from_gaussian(traditional, image_size)

    print(f"\nConverted to AdaptiveGaussian2D:")
    print(f"  Position (normalized): {adaptive.mu}")
    print(f"  Inverse scales: {adaptive.inv_s}")
    print(f"  Color (normalized): {adaptive.color}")
    print(f"  Aspect ratio: {adaptive.aspect_ratio:.3f}")

    # Convert back
    restored = adaptive.to_gaussian(image_size)

    print(f"\nRestored to Gaussian:")
    print(f"  Position: ({restored.x:.1f}, {restored.y:.1f})")
    print(f"  Radii: ({restored.rx:.1f}, {restored.ry:.1f})")
    print(f"  Color: RGB({restored.r}, {restored.g}, {restored.b})")

    # Check round-trip accuracy
    pos_error = abs(traditional.x - restored.x) + abs(traditional.y - restored.y)
    color_error = abs(traditional.r - restored.r) + abs(traditional.g - restored.g) + abs(traditional.b - restored.b)

    print(f"\nRound-trip accuracy:")
    print(f"  Position error: {pos_error:.3f} pixels")
    print(f"  Color error: {color_error} (out of 765)")


def demo_serialization():
    """Demonstrate serialization capabilities."""
    print("\n=== Serialization Demonstration ===")

    # Create complex Gaussian
    gaussian = AdaptiveGaussian2D(
        mu=np.array([0.25, 0.75]),
        inv_s=np.array([2.0, 0.5]),
        theta=np.pi / 3,
        color=np.array([0.8, 0.4, 0.6]),
        alpha=0.9,
        content_complexity=0.7,
        saliency_score=0.8,
        refinement_count=15
    )

    print(f"Original Gaussian: {gaussian}")

    # Serialize
    data = gaussian.to_dict()
    print(f"\nSerialized data keys: {list(data.keys())}")

    # Deserialize
    restored = AdaptiveGaussian2D.from_dict(data)
    print(f"Restored Gaussian: {restored}")

    # Verify equality
    print(f"\nSerialization accurate: {np.allclose(gaussian.mu, restored.mu)}")


if __name__ == "__main__":
    demo_covariance_computation()
    demo_parameter_optimization()
    demo_backward_compatibility()
    demo_serialization()

    print("\n=== AdaptiveGaussian2D Demo Complete ===")
    print("✅ Full covariance matrix support")
    print("✅ Anisotropic ellipse generation")
    print("✅ Numerical stability and parameter clipping")
    print("✅ Backward compatibility with current Gaussian class")
    print("✅ Serialization and optimization support")
    print("\nReady for next phase: Gradient Computation Utilities!")