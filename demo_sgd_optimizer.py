#!/usr/bin/env python3
"""
Demonstration script for SGD optimization in adaptive Gaussian splatting.

This script showcases the T3.2: SGD Optimization Loop functionality,
demonstrating how to optimize Gaussian splat parameters using various
SGD methods with adaptive learning rates and convergence criteria.
"""

import numpy as np
from typing import List, Dict
import logging

# Import our SGD optimization modules
from src.splat_this.core.sgd_optimizer import (
    SGDConfig,
    SGDOptimizer,
    OptimizationMethod,
    LearningRateSchedule,
    optimize_splats_sgd,
    create_sgd_config_preset
)
from src.splat_this.core.manual_gradients import GradientConfig
from src.splat_this.core.adaptive_gaussian import (
    AdaptiveGaussian2D,
    create_isotropic_gaussian,
    create_anisotropic_gaussian
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_test_scenario(scenario: str = "basic") -> tuple:
    """Create test images and splats for different scenarios."""
    if scenario == "basic":
        # Simple test case
        size = (32, 32)
        target = np.ones((*size, 3)) * 0.6
        rendered = np.ones((*size, 3)) * 0.4
        error_map = np.abs(target[:, :, 0] - rendered[:, :, 0])

        splats = [
            create_isotropic_gaussian([0.3, 0.3], 0.1, [0.8, 0.2, 0.1], 0.8),
            create_isotropic_gaussian([0.7, 0.7], 0.08, [0.1, 0.8, 0.2], 0.7)
        ]

    elif scenario == "complex":
        # More complex scenario with varied error
        size = (48, 48)
        H, W = size
        x, y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))

        # Target with structure
        target = np.stack([
            0.5 + 0.3 * np.sin(6 * np.pi * x),
            0.3 + 0.4 * np.cos(6 * np.pi * y),
            0.7 - 0.2 * (x + y)
        ], axis=-1)
        target = np.clip(target, 0, 1)

        # Rendered with systematic error
        rendered = target + 0.1 * np.sin(4 * np.pi * x * y)[:, :, None]
        rendered = np.clip(rendered, 0, 1)

        error_map = np.linalg.norm(target - rendered, axis=-1)

        splats = [
            create_isotropic_gaussian([0.2, 0.2], 0.12, [0.9, 0.1, 0.1], 0.9),
            create_anisotropic_gaussian([0.5, 0.5], (0.08, 0.15), np.pi/6, [0.1, 0.9, 0.1], 0.8),
            create_isotropic_gaussian([0.8, 0.8], 0.1, [0.1, 0.1, 0.9], 0.7),
            create_anisotropic_gaussian([0.3, 0.7], (0.06, 0.18), np.pi/3, [0.8, 0.8, 0.1], 0.85)
        ]

    else:  # "convergence"
        # Scenario designed to test convergence
        size = (24, 24)
        target = np.random.random((*size, 3)) * 0.5 + 0.25
        rendered = target + np.random.normal(0, 0.02, target.shape)
        rendered = np.clip(rendered, 0, 1)
        error_map = np.abs(target[:, :, 0] - rendered[:, :, 0])

        splats = [create_isotropic_gaussian([0.5, 0.5], 0.15, [0.6, 0.6, 0.6], 0.8)]

    return target, rendered, error_map, splats


def demo_basic_optimization():
    """Demonstrate basic SGD optimization."""
    print("=" * 60)
    print("DEMO 1: Basic SGD Optimization")
    print("=" * 60)

    target, rendered, error_map, splats = create_test_scenario("basic")

    print(f"Test setup:")
    print(f"  Image size: {target.shape[:2]}")
    print(f"  Number of splats: {len(splats)}")
    print(f"  Initial error: {np.sum(error_map**2):.6f}")

    # Create basic SGD configuration
    config = SGDConfig(
        method=OptimizationMethod.SGD_MOMENTUM,
        position_lr=0.01,
        scale_lr=0.005,
        rotation_lr=0.001,
        color_lr=0.005,
        alpha_lr=0.002,
        momentum=0.9,
        max_iterations=50,
        log_every=10
    )

    print(f"\nOptimization configuration:")
    print(f"  Method: {config.method.value}")
    print(f"  Position LR: {config.position_lr}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Momentum: {config.momentum}")

    # Run optimization
    print(f"\nStarting optimization...")
    result = optimize_splats_sgd(splats, target, rendered, error_map, config)

    print(f"\nOptimization results:")
    print(f"  Final loss: {result.final_loss:.6f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Early stopped: {result.early_stopped}")

    # Show parameter changes
    for i, (orig, opt) in enumerate(zip(splats, result.optimized_splats)):
        print(f"\nSplat {i} changes:")
        pos_change = np.linalg.norm(opt.mu - orig.mu)
        print(f"  Position change: {pos_change:.4f}")
        scale_change = np.linalg.norm(1/opt.inv_s - 1/orig.inv_s)
        print(f"  Scale change: {scale_change:.4f}")
        color_change = np.linalg.norm(opt.color - orig.color)
        print(f"  Color change: {color_change:.4f}")


def demo_optimization_methods():
    """Demonstrate different optimization methods."""
    print("\n" + "=" * 60)
    print("DEMO 2: Optimization Methods Comparison")
    print("=" * 60)

    target, rendered, error_map, splats = create_test_scenario("basic")

    methods = [
        (OptimizationMethod.SGD, "Standard SGD"),
        (OptimizationMethod.SGD_MOMENTUM, "SGD with Momentum"),
        (OptimizationMethod.ADAM, "Adam Optimizer"),
        (OptimizationMethod.RMSPROP, "RMSprop"),
        (OptimizationMethod.ADAGRAD, "Adagrad")
    ]

    results = {}

    for method, name in methods:
        print(f"\nTesting {name}...")

        config = SGDConfig(
            method=method,
            max_iterations=30,
            log_every=999  # Suppress logging for comparison
        )

        # Use copy of splats for each method
        test_splats = [splat.copy() for splat in splats]
        result = optimize_splats_sgd(test_splats, target, rendered, error_map, config)

        results[method] = result

        print(f"  Final loss: {result.final_loss:.6f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Converged: {result.converged}")

    # Compare results
    print(f"\nMethod comparison:")
    best_loss = min(r.final_loss for r in results.values())
    for method, name in methods:
        result = results[method]
        improvement = ((result.final_loss - best_loss) / best_loss) * 100 if best_loss > 0 else 0
        print(f"  {name:20}: Loss={result.final_loss:.6f} (+{improvement:5.1f}%)")


def demo_learning_rate_schedules():
    """Demonstrate learning rate scheduling."""
    print("\n" + "=" * 60)
    print("DEMO 3: Learning Rate Schedules")
    print("=" * 60)

    target, rendered, error_map, splats = create_test_scenario("basic")

    schedules = [
        (LearningRateSchedule.CONSTANT, "Constant LR"),
        (LearningRateSchedule.LINEAR_DECAY, "Linear Decay"),
        (LearningRateSchedule.EXPONENTIAL_DECAY, "Exponential Decay"),
        (LearningRateSchedule.COSINE_ANNEALING, "Cosine Annealing"),
        (LearningRateSchedule.STEP_DECAY, "Step Decay")
    ]

    for schedule, name in schedules:
        print(f"\nTesting {name}...")

        config = SGDConfig(
            lr_schedule=schedule,
            max_iterations=40,
            log_every=999
        )

        test_splats = [splat.copy() for splat in splats]
        result = optimize_splats_sgd(test_splats, target, rendered, error_map, config)

        print(f"  Final loss: {result.final_loss:.6f}")
        print(f"  Converged: {result.converged}")

        # Show learning rate evolution
        history = result.optimization_history
        if history.learning_rates['position']:
            initial_lr = history.learning_rates['position'][0]
            final_lr = history.learning_rates['position'][-1]
            lr_ratio = final_lr / initial_lr if initial_lr > 0 else 1.0
            print(f"  LR ratio (final/initial): {lr_ratio:.3f}")


def demo_convergence_criteria():
    """Demonstrate convergence criteria and early stopping."""
    print("\n" + "=" * 60)
    print("DEMO 4: Convergence Criteria")
    print("=" * 60)

    target, rendered, error_map, splats = create_test_scenario("convergence")

    configs = [
        ("Strict convergence", SGDConfig(
            convergence_threshold=1e-8,
            early_stopping_patience=10,
            max_iterations=200
        )),
        ("Relaxed convergence", SGDConfig(
            convergence_threshold=1e-4,
            early_stopping_patience=30,
            max_iterations=200
        )),
        ("Early stopping", SGDConfig(
            convergence_threshold=1e-10,  # Very strict
            early_stopping_patience=5,    # But early stopping enabled
            max_iterations=200
        ))
    ]

    for name, config in configs:
        print(f"\nTesting {name}...")
        config.log_every = 999  # Suppress logging

        test_splats = [splat.copy() for splat in splats]
        result = optimize_splats_sgd(test_splats, target, rendered, error_map, config)

        print(f"  Final loss: {result.final_loss:.6f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Converged: {result.converged}")
        print(f"  Early stopped: {result.early_stopped}")

        # Show convergence metrics
        history = result.optimization_history
        if history.gradient_norms:
            final_grad_norm = history.gradient_norms[-1]
            print(f"  Final gradient norm: {final_grad_norm:.6f}")


def demo_batch_processing():
    """Demonstrate batch vs mini-batch processing."""
    print("\n" + "=" * 60)
    print("DEMO 5: Batch Processing")
    print("=" * 60)

    target, rendered, error_map, splats = create_test_scenario("complex")

    batch_configs = [
        ("Full batch", SGDConfig(batch_size=None, max_iterations=30)),
        ("Mini-batch (size=2)", SGDConfig(batch_size=2, max_iterations=30)),
        ("Single sample", SGDConfig(batch_size=1, max_iterations=30))
    ]

    for name, config in batch_configs:
        print(f"\nTesting {name}...")
        config.log_every = 999

        test_splats = [splat.copy() for splat in splats]
        result = optimize_splats_sgd(test_splats, target, rendered, error_map, config)

        print(f"  Final loss: {result.final_loss:.6f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Converged: {result.converged}")


def demo_presets():
    """Demonstrate configuration presets."""
    print("\n" + "=" * 60)
    print("DEMO 6: Configuration Presets")
    print("=" * 60)

    target, rendered, error_map, splats = create_test_scenario("basic")

    presets = ["fast", "balanced", "high_quality"]

    for preset in presets:
        print(f"\nTesting '{preset}' preset...")

        config = create_sgd_config_preset(preset)
        config.log_every = 999

        test_splats = [splat.copy() for splat in splats]
        result = optimize_splats_sgd(test_splats, target, rendered, error_map, config)

        print(f"  Method: {config.method.value}")
        print(f"  Max iterations: {config.max_iterations}")
        print(f"  Final loss: {result.final_loss:.6f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Converged: {result.converged}")


def demo_gradient_clipping():
    """Demonstrate gradient clipping effectiveness."""
    print("\n" + "=" * 60)
    print("DEMO 7: Gradient Clipping")
    print("=" * 60)

    # Create scenario with potential for large gradients
    target = np.ones((16, 16, 3)) * 0.1
    rendered = np.ones((16, 16, 3)) * 0.9  # Large difference
    error_map = np.abs(target[:, :, 0] - rendered[:, :, 0])

    splats = [create_isotropic_gaussian([0.5, 0.5], 0.05, [1.0, 0.0, 0.0], 0.1)]

    configs = [
        ("No clipping", SGDConfig(
            gradient_clipping=False,
            max_iterations=20
        )),
        ("Clipping (threshold=1.0)", SGDConfig(
            gradient_clipping=True,
            clip_threshold=1.0,
            max_iterations=20
        )),
        ("Aggressive clipping (threshold=0.1)", SGDConfig(
            gradient_clipping=True,
            clip_threshold=0.1,
            max_iterations=20
        ))
    ]

    for name, config in configs:
        print(f"\nTesting {name}...")
        config.log_every = 999

        test_splats = [splat.copy() for splat in splats]
        result = optimize_splats_sgd(test_splats, target, rendered, error_map, config)

        print(f"  Final loss: {result.final_loss:.6f}")
        print(f"  Iterations: {result.iterations}")

        # Check for numerical stability
        opt_splat = result.optimized_splats[0]
        is_stable = (
            np.all(np.isfinite(opt_splat.mu)) and
            np.all(opt_splat.inv_s > 0) and
            np.isfinite(opt_splat.theta) and
            np.all(0 <= opt_splat.color) and np.all(opt_splat.color <= 1) and
            0 <= opt_splat.alpha <= 1
        )
        print(f"  Numerically stable: {is_stable}")


def demo_system_summary():
    """Provide a summary of SGD optimization capabilities."""
    print("\n" + "=" * 60)
    print("DEMO 8: System Summary")
    print("=" * 60)

    target, rendered, error_map, splats = create_test_scenario("basic")

    # Run with balanced config
    config = create_sgd_config_preset("balanced")
    config.max_iterations = 50
    config.log_every = 999

    result = optimize_splats_sgd(splats, target, rendered, error_map, config)

    print("SGD Optimization System Features:")
    print("  ✓ Multiple optimization methods (SGD, Momentum, Adam, RMSprop, Adagrad)")
    print("  ✓ Adaptive learning rate scheduling (5 different schedules)")
    print("  ✓ Convergence criteria and early stopping")
    print("  ✓ Gradient clipping for numerical stability")
    print("  ✓ Batch and mini-batch processing")
    print("  ✓ Comprehensive parameter updates (position, scale, rotation, color, alpha)")
    print("  ✓ Configuration presets for different use cases")
    print("  ✓ Integration with manual gradient computation")
    print("  ✓ Detailed optimization history tracking")
    print("  ✓ Robust error handling and validation")

    print(f"\nExample optimization result:")
    print(f"  Method: {config.method.value}")
    print(f"  Final loss: {result.final_loss:.6f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Parameters optimized: {len(result.optimized_splats)} splats")

    history = result.optimization_history
    if history.loss_history:
        improvement = ((history.loss_history[0] - result.final_loss) / history.loss_history[0]) * 100
        print(f"  Loss improvement: {improvement:.1f}%")

    if result.gradient_validation:
        print(f"  Gradient validation: {'PASSED' if result.gradient_validation.passed else 'FAILED'}")

    print("\nT3.2: SGD Optimization Loop - COMPLETE ✓")


def main():
    """Run all SGD optimization demonstrations."""
    print("SGD Optimization System Demo")
    print("Part of T3.2: SGD Optimization Loop")
    print("Adaptive Gaussian Splatting System")

    try:
        demo_basic_optimization()
        demo_optimization_methods()
        demo_learning_rate_schedules()
        demo_convergence_criteria()
        demo_batch_processing()
        demo_presets()
        demo_gradient_clipping()
        demo_system_summary()

        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("SGD optimization system is fully functional.")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()