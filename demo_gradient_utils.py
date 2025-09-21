#!/usr/bin/env python3
"""Demonstration of gradient computation utilities for adaptive Gaussian splatting."""

import numpy as np
from src.splat_this.core.gradient_utils import (
    GradientAnalyzer,
    ProbabilityMapGenerator,
    SpatialSampler,
    EdgeDetector,
    visualize_gradient_analysis,
    sample_adaptive_positions
)


def create_test_image():
    """Create a synthetic test image with various features."""
    # Create base image
    image = np.zeros((100, 100, 3), dtype=np.float32)

    # Add vertical stripes
    for i in range(10, 90, 15):
        image[:, i:i+3, :] = [0.8, 0.6, 0.4]

    # Add horizontal edges
    image[20:25, :, :] = [0.9, 0.2, 0.1]
    image[70:75, :, :] = [0.1, 0.8, 0.3]

    # Add circular feature
    center_y, center_x = 50, 50
    y, x = np.ogrid[:100, :100]
    mask = (x - center_x)**2 + (y - center_y)**2 < 15**2
    image[mask] = [0.2, 0.3, 0.9]

    # Add noise for texture
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)

    return image


def demo_gradient_analysis():
    """Demonstrate gradient analysis capabilities."""
    print("=== Gradient Analysis Demo ===")

    image = create_test_image()
    print(f"Created test image: {image.shape}")

    # Initialize analyzer
    analyzer = GradientAnalyzer(sigma=1.0, method='sobel')

    # Compute gradients
    grad_x, grad_y = analyzer.compute_gradients(image)
    print(f"Gradient shapes: {grad_x.shape}, {grad_y.shape}")

    # Compute gradient magnitude and orientation
    magnitude = analyzer.compute_gradient_magnitude(image)
    orientation = analyzer.compute_gradient_orientation(image)

    print(f"Gradient magnitude - min: {np.min(magnitude):.4f}, max: {np.max(magnitude):.4f}")
    print(f"Gradient orientation - min: {np.min(orientation):.4f}, max: {np.max(orientation):.4f}")

    # Structure tensor analysis
    structure_tensor = analyzer.compute_structure_tensor(image)
    edge_strength, principal_orientation, coherence = analyzer.analyze_local_structure(structure_tensor)

    print(f"Edge strength - min: {np.min(edge_strength):.4f}, max: {np.max(edge_strength):.4f}")
    print(f"Mean coherence: {np.mean(coherence):.4f}")

    # Find strongest edges
    strong_edges = edge_strength > np.percentile(edge_strength, 90)
    num_strong_edges = np.sum(strong_edges)
    print(f"Strong edge pixels (top 10%): {num_strong_edges}")


def demo_probability_maps():
    """Demonstrate probability map generation."""
    print("\n=== Probability Map Generation Demo ===")

    image = create_test_image()

    # Initialize generator
    generator = ProbabilityMapGenerator(gradient_weight=0.7, uniform_weight=0.3)
    analyzer = GradientAnalyzer()

    # Create different types of probability maps
    grad_magnitude = analyzer.compute_gradient_magnitude(image)

    # Pure gradient-based map
    grad_prob = generator.create_gradient_probability_map(grad_magnitude)
    print(f"Gradient probability map - sum: {np.sum(grad_prob):.6f}")

    # Mixed map
    mixed_prob = generator.create_mixed_probability_map(image, analyzer)
    print(f"Mixed probability map - sum: {np.sum(mixed_prob):.6f}")

    # Saliency-based map
    try:
        saliency_prob = generator.create_saliency_probability_map(image)
        print(f"Saliency probability map - sum: {np.sum(saliency_prob):.6f}")
    except Exception as e:
        print(f"Saliency map generation skipped: {e}")

    # Compare concentration
    grad_entropy = -np.sum(grad_prob * np.log(grad_prob + 1e-10))
    mixed_entropy = -np.sum(mixed_prob * np.log(mixed_prob + 1e-10))

    print(f"Gradient map entropy: {grad_entropy:.4f}")
    print(f"Mixed map entropy: {mixed_entropy:.4f}")
    print("(Lower entropy = more concentrated)")


def demo_spatial_sampling():
    """Demonstrate spatial sampling strategies."""
    print("\n=== Spatial Sampling Demo ===")

    image = create_test_image()

    # Create probability map
    analyzer = GradientAnalyzer()
    generator = ProbabilityMapGenerator()
    prob_map = generator.create_mixed_probability_map(image, analyzer)

    # Initialize sampler
    sampler = SpatialSampler(seed=42)

    # Different sampling strategies
    n_samples = 50

    # Basic sampling
    positions_basic = sampler.sample_from_probability_map(prob_map, n_samples)
    print(f"Basic sampling: {len(positions_basic)} positions")

    # Minimum distance sampling
    positions_min_dist = sampler.sample_with_minimum_distance(
        prob_map, n_samples, min_distance=8.0, max_attempts=200
    )
    print(f"Min distance sampling: {len(positions_min_dist)} positions")

    # Stratified sampling
    positions_stratified = sampler.sample_stratified(
        prob_map, n_samples, grid_divisions=5
    )
    print(f"Stratified sampling: {len(positions_stratified)} positions")

    # Analyze spatial distribution
    def compute_spatial_stats(positions):
        positions_array = np.array(positions)
        mean_y, mean_x = np.mean(positions_array, axis=0)
        std_y, std_x = np.std(positions_array, axis=0)
        return mean_y, mean_x, std_y, std_x

    print("\nSpatial statistics:")
    for name, positions in [
        ("Basic", positions_basic),
        ("Min distance", positions_min_dist),
        ("Stratified", positions_stratified)
    ]:
        mean_y, mean_x, std_y, std_x = compute_spatial_stats(positions)
        print(f"  {name}: center=({mean_y:.1f}, {mean_x:.1f}), spread=({std_y:.1f}, {std_x:.1f})")


def demo_edge_detection():
    """Demonstrate edge detection utilities."""
    print("\n=== Edge Detection Demo ===")

    image = create_test_image()

    # Initialize detector
    detector = EdgeDetector()

    # Canny edge detection
    try:
        edges_canny = detector.detect_edges_canny(
            image, sigma=1.0, low_threshold=0.1, high_threshold=0.2
        )
        print(f"Canny edges: {np.sum(edges_canny)} edge pixels")
    except Exception as e:
        print(f"Canny detection skipped: {e}")

    # Gradient-based edge detection
    edges_gradient = detector.detect_edges_gradient(image, threshold=0.3)
    print(f"Gradient edges: {np.sum(edges_gradient)} edge pixels")

    # Gaussian smoothing
    smoothed = detector.apply_gaussian_smoothing(image, sigma=2.0)
    print(f"Smoothed image variance reduction: {np.var(image):.6f} -> {np.var(smoothed):.6f}")

    # Edge orientation map
    orientation_map = detector.compute_edge_orientation_map(image, sigma=1.0)
    print(f"Edge orientation range: [{np.min(orientation_map):.3f}, {np.max(orientation_map):.3f}] rad")


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n=== Convenience Functions Demo ===")

    image = create_test_image()

    # Adaptive position sampling with different methods
    methods = ['mixed', 'uniform']
    n_samples = 30

    for method in methods:
        try:
            positions = sample_adaptive_positions(image, n_samples, method=method)
            print(f"{method.capitalize()} sampling: {len(positions)} positions")

            # Compute coverage distribution
            y_coords = [pos[0] for pos in positions]
            x_coords = [pos[1] for pos in positions]
            y_range = max(y_coords) - min(y_coords)
            x_range = max(x_coords) - min(x_coords)
            print(f"  Coverage: Y range = {y_range}, X range = {x_range}")

        except Exception as e:
            print(f"{method} sampling failed: {e}")


def demo_comprehensive_analysis():
    """Demonstrate comprehensive gradient analysis."""
    print("\n=== Comprehensive Analysis Demo ===")

    image = create_test_image()

    try:
        # Run full analysis
        results = visualize_gradient_analysis(image)

        print(f"Analysis results available:")
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")

        # Summary statistics
        grad_mag = results['gradient_magnitude']
        edge_strength = results['edge_strength']
        coherence = results['coherence']
        prob_map = results['probability_map']

        print(f"\nSummary statistics:")
        print(f"  Max gradient magnitude: {np.max(grad_mag):.4f}")
        print(f"  Mean edge strength: {np.mean(edge_strength):.4f}")
        print(f"  Mean coherence: {np.mean(coherence):.4f}")
        print(f"  Probability map entropy: {-np.sum(prob_map * np.log(prob_map + 1e-10)):.4f}")

    except Exception as e:
        print(f"Comprehensive analysis failed: {e}")


if __name__ == "__main__":
    print("🎯 SplatThis Gradient Utilities Demonstration")
    print("=" * 50)

    demo_gradient_analysis()
    demo_probability_maps()
    demo_spatial_sampling()
    demo_edge_detection()
    demo_convenience_functions()
    demo_comprehensive_analysis()

    print("\n" + "=" * 50)
    print("✅ Gradient Utilities Demo Complete!")
    print("🔄 Ready for content-adaptive Gaussian splat placement")
    print("📊 All gradient analysis tools operational")