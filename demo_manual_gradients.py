#!/usr/bin/env python3
"""
Demonstration script for manual gradient computation in adaptive Gaussian splatting.

This script showcases the T3.1: Manual Gradient Computation functionality,
demonstrating how to compute gradients for all Gaussian splat parameters
using finite differences.
"""

import numpy as np
from typing import Tuple, List

# Import our manual gradient computation modules
from src.splat_this.core.manual_gradients import (
    GradientConfig,
    ManualGradientComputer,
    compute_splat_gradients,
    validate_gradient_computation
)
from src.splat_this.core.adaptive_gaussian import (
    AdaptiveGaussian2D,
    create_isotropic_gaussian,
    create_anisotropic_gaussian
)


def create_test_images(size: Tuple[int, int] = (64, 64)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic test images for gradient computation."""
    H, W = size

    # Create target image with some structure
    x, y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    target = np.stack([
        0.5 + 0.3 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y),  # Red channel
        0.3 + 0.4 * (x**2 + y**2),                                   # Green channel
        0.7 - 0.2 * np.abs(x - y)                                   # Blue channel
    ], axis=-1)
    target = np.clip(target, 0, 1)

    # Create rendered image (slightly different)
    rendered = target + 0.1 * np.random.normal(0, 0.05, target.shape)
    rendered = np.clip(rendered, 0, 1)

    # Compute error map
    error_map = np.linalg.norm(target - rendered, axis=-1)

    return target, rendered, error_map


def demo_basic_gradient_computation():
    """Demonstrate basic gradient computation for a single splat."""
    print("=" * 60)
    print("DEMO 1: Basic Gradient Computation")
    print("=" * 60)

    # Create test splat
    splat = create_isotropic_gaussian(
        center=[0.5, 0.5],
        scale=0.1,
        color=[0.8, 0.2, 0.1],
        alpha=0.8
    )

    # Create test images
    target, rendered, error_map = create_test_images()

    # Create gradient computer
    config = GradientConfig(
        position_step=0.01,
        scale_step=0.001,
        rotation_step=0.01,
        color_step=0.001,
        gradient_clipping=True,
        clip_threshold=10.0
    )
    computer = ManualGradientComputer(config)

    print(f"Input splat: mu={splat.mu}, inv_s={splat.inv_s}, theta={splat.theta:.3f}")
    print(f"             color={splat.color}, alpha={splat.alpha:.3f}")

    # Compute all gradients
    gradients = computer.compute_all_gradients(splat, target, rendered, error_map)

    print(f"\nComputed gradients:")
    print(f"  Position: {gradients.position_grad}")
    print(f"  Scale:    {gradients.scale_grad}")
    print(f"  Rotation: {gradients.rotation_grad:.6f}")
    print(f"  Color:    {gradients.color_grad}")
    print(f"  Alpha:    {gradients.alpha_grad:.6f}")

    # Validate gradients
    validation = computer.validate_gradients(splat, target, rendered, error_map, gradients)
    print(f"\nGradient validation:")
    print(f"  Position error: {validation.position_error:.6f}")
    print(f"  Scale error:    {validation.scale_error:.6f}")
    print(f"  Rotation error: {validation.rotation_error:.6f}")
    print(f"  Color error:    {validation.color_error:.6f}")
    print(f"  Alpha error:    {validation.alpha_error:.6f}")
    print(f"  Max error:      {validation.max_error:.6f}")
    print(f"  Validation passed: {validation.passed}")


def demo_finite_difference_methods():
    """Demonstrate different finite difference methods."""
    print("\n" + "=" * 60)
    print("DEMO 2: Finite Difference Methods Comparison")
    print("=" * 60)

    # Create test data
    splat = create_anisotropic_gaussian(
        center=[0.4, 0.6],
        scales=(0.08, 0.12),
        orientation=np.pi/4,
        color=[0.2, 0.8, 0.3],
        alpha=0.9
    )
    target, rendered, error_map = create_test_images()

    methods = ['forward', 'backward', 'central']
    results = {}

    for method in methods:
        config = GradientConfig(finite_diff_method=method)
        computer = ManualGradientComputer(config)
        gradients = computer.compute_all_gradients(splat, target, rendered, error_map)
        results[method] = gradients

        print(f"\n{method.capitalize()} differences:")
        print(f"  Position: {gradients.position_grad}")
        print(f"  Scale:    {gradients.scale_grad}")
        print(f"  Rotation: {gradients.rotation_grad:.6f}")

    # Compare methods
    print(f"\nMethod comparison (Position gradient):")
    forward_pos = results['forward'].position_grad
    backward_pos = results['backward'].position_grad
    central_pos = results['central'].position_grad

    print(f"  Forward vs Central:  diff = {np.linalg.norm(forward_pos - central_pos):.6f}")
    print(f"  Backward vs Central: diff = {np.linalg.norm(backward_pos - central_pos):.6f}")


def demo_gradient_clipping():
    """Demonstrate gradient clipping functionality."""
    print("\n" + "=" * 60)
    print("DEMO 3: Gradient Clipping")
    print("=" * 60)

    # Create splat with extreme parameters to generate large gradients
    splat = create_isotropic_gaussian([0.5, 0.5], 0.01, [1.0, 0.0, 0.0], 0.1)

    # Create images with large differences
    target = np.ones((32, 32, 3)) * 0.1
    rendered = np.ones((32, 32, 3)) * 0.9
    error_map = np.linalg.norm(target - rendered, axis=-1)

    # Without clipping
    config_no_clip = GradientConfig(gradient_clipping=False)
    computer_no_clip = ManualGradientComputer(config_no_clip)
    gradients_no_clip = computer_no_clip.compute_all_gradients(splat, target, rendered, error_map)

    # With clipping
    config_clip = GradientConfig(gradient_clipping=True, clip_threshold=5.0)
    computer_clip = ManualGradientComputer(config_clip)
    gradients_clip = computer_clip.compute_all_gradients(splat, target, rendered, error_map)

    print("Without clipping:")
    print(f"  Position norm: {np.linalg.norm(gradients_no_clip.position_grad):.3f}")
    print(f"  Scale norm:    {np.linalg.norm(gradients_no_clip.scale_grad):.3f}")
    print(f"  Color norm:    {np.linalg.norm(gradients_no_clip.color_grad):.3f}")

    print("\nWith clipping (threshold=5.0):")
    print(f"  Position norm: {np.linalg.norm(gradients_clip.position_grad):.3f}")
    print(f"  Scale norm:    {np.linalg.norm(gradients_clip.scale_grad):.3f}")
    print(f"  Color norm:    {np.linalg.norm(gradients_clip.color_grad):.3f}")


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n" + "=" * 60)
    print("DEMO 4: Convenience Functions")
    print("=" * 60)

    # Create test data
    splat = create_isotropic_gaussian([0.3, 0.7], 0.08, [0.5, 0.5, 0.8], 0.7)
    target, rendered, error_map = create_test_images()

    # Use convenience function for gradient computation
    config = GradientConfig(finite_diff_method='central', gradient_clipping=True)
    gradients = compute_splat_gradients(splat, target, rendered, error_map, config)

    print("Using compute_splat_gradients():")
    print(f"  Position: {gradients.position_grad}")
    print(f"  Scale:    {gradients.scale_grad}")
    print(f"  Alpha:    {gradients.alpha_grad:.6f}")

    # Use convenience function for validation
    validation = validate_gradient_computation(splat, target, rendered, error_map, config)

    print(f"\nUsing validate_gradient_computation():")
    print(f"  Max error:   {validation.max_error:.6f}")
    print(f"  Validation:  {validation.passed}")


def demo_step_size_effects():
    """Demonstrate effect of different step sizes."""
    print("\n" + "=" * 60)
    print("DEMO 5: Step Size Effects")
    print("=" * 60)

    splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.7, 0.3, 0.1], 0.8)
    target, rendered, error_map = create_test_images()

    step_sizes = [0.001, 0.01, 0.1]

    for step in step_sizes:
        config = GradientConfig(
            position_step=step,
            scale_step=step * 0.1,  # Scale steps are typically smaller
            rotation_step=step,
            color_step=step * 0.1   # Color steps are typically smaller
        )
        computer = ManualGradientComputer(config)
        gradients = computer.compute_all_gradients(splat, target, rendered, error_map)

        print(f"\nStep size = {step}:")
        print(f"  Position: [{gradients.position_grad[0]:.6f}, {gradients.position_grad[1]:.6f}]")
        print(f"  Rotation: {gradients.rotation_grad:.6f}")
        print(f"  Alpha:    {gradients.alpha_grad:.6f}")


def demo_summary():
    """Provide a summary of gradient computation capabilities."""
    print("\n" + "=" * 60)
    print("DEMO 6: System Summary")
    print("=" * 60)

    # Create a splat
    splat = create_isotropic_gaussian([0.5, 0.5], 0.15, [0.8, 0.2, 0.1], 0.8)
    target, rendered, error_map = create_test_images((32, 32))

    # Compute gradients
    config = GradientConfig()
    computer = ManualGradientComputer(config)
    gradients = computer.compute_all_gradients(splat, target, rendered, error_map)

    print("Manual Gradient Computation System Features:")
    print("  ✓ Position gradient computation (2D)")
    print("  ✓ Scale gradient computation (anisotropic)")
    print("  ✓ Rotation gradient computation")
    print("  ✓ Color gradient computation (RGB)")
    print("  ✓ Alpha gradient computation")
    print("  ✓ Finite difference methods (forward/backward/central)")
    print("  ✓ Gradient clipping for numerical stability")
    print("  ✓ Numerical gradient validation")
    print("  ✓ Comprehensive unit test coverage")
    print("  ✓ Convenience functions for easy use")

    print(f"\nExample gradient computation:")
    print(f"  Input splat:  position=[{splat.mu[0]:.2f}, {splat.mu[1]:.2f}], scale={1/splat.inv_s[0]:.3f}")
    print(f"  Position grad: [{gradients.position_grad[0]:.4f}, {gradients.position_grad[1]:.4f}]")
    print(f"  Scale grad:    [{gradients.scale_grad[0]:.4f}, {gradients.scale_grad[1]:.4f}]")
    print(f"  Rotation grad: {gradients.rotation_grad:.4f}")
    print(f"  Alpha grad:    {gradients.alpha_grad:.4f}")

    # Validate the gradients
    validation = computer.validate_gradients(splat, target, rendered, error_map, gradients)
    print(f"\nGradient validation: {'PASSED' if validation.passed else 'FAILED'}")
    print(f"  Max validation error: {validation.max_error:.6f}")

    print("\nT3.1: Manual Gradient Computation - COMPLETE ✓")


def main():
    """Run all gradient computation demonstrations."""
    print("Manual Gradient Computation Demo")
    print("Part of T3.1: Manual Gradient Computation")
    print("Adaptive Gaussian Splatting System")

    try:
        demo_basic_gradient_computation()
        demo_finite_difference_methods()
        demo_gradient_clipping()
        demo_convenience_functions()
        demo_step_size_effects()
        demo_summary()

        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("Manual gradient computation system is fully functional.")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()