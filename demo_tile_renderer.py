#!/usr/bin/env python3
"""Demonstration of tile-based rendering framework for adaptive Gaussian splatting."""

import numpy as np
import time
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian, create_anisotropic_gaussian
from src.splat_this.core.tile_renderer import TileRenderer, RenderConfig, create_tile_renderer


def create_test_gaussians(image_size: tuple) -> list:
    """Create diverse test Gaussians for demonstration."""
    H, W = image_size
    gaussians = []

    # Central isotropic Gaussian
    gaussians.append(create_isotropic_gaussian(
        center=[0.5, 0.5],
        scale=0.1,
        color=[0.8, 0.2, 0.2],
        alpha=0.7
    ))

    # Anisotropic edge-like Gaussians
    gaussians.append(create_anisotropic_gaussian(
        center=[0.2, 0.8],
        scales=(0.05, 0.2),  # Elongated vertically
        orientation=np.pi/2,
        color=[0.2, 0.8, 0.2],
        alpha=0.8
    ))

    gaussians.append(create_anisotropic_gaussian(
        center=[0.8, 0.3],
        scales=(0.15, 0.03),  # Elongated horizontally
        orientation=0.0,
        color=[0.2, 0.2, 0.8],
        alpha=0.6
    ))

    # Diagonal edge
    gaussians.append(create_anisotropic_gaussian(
        center=[0.7, 0.7],
        scales=(0.02, 0.12),  # Thin diagonal
        orientation=np.pi/4,
        color=[0.8, 0.8, 0.2],
        alpha=0.9
    ))

    # Multiple small overlapping Gaussians
    for i in range(5):
        x = 0.3 + i * 0.08
        y = 0.4
        gaussians.append(create_isotropic_gaussian(
            center=[x, y],
            scale=0.03,
            color=[0.8, 0.4, 0.8],
            alpha=0.5
        ))

    # Corner Gaussians to test boundary handling
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    colors = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]]

    for (x, y), color in zip(corners, colors):
        gaussians.append(create_isotropic_gaussian(
            center=[x, y],
            scale=0.08,
            color=color,
            alpha=0.6
        ))

    return gaussians


def demo_basic_rendering():
    """Demonstrate basic tile-based rendering."""
    print("=== Basic Tile Rendering Demo ===")

    image_size = (128, 128)
    gaussians = create_test_gaussians(image_size)

    print(f"Created {len(gaussians)} test Gaussians")
    print(f"Image size: {image_size}")

    # Create renderer with default settings
    renderer = TileRenderer(image_size)
    print(f"Tile grid: {renderer.tiles_x}x{renderer.tiles_y} tiles ({renderer.total_tiles} total)")

    # Render image
    start_time = time.time()
    rendered = renderer.render_full_image(gaussians)
    render_time = time.time() - start_time

    print(f"Rendering completed in {render_time:.3f} seconds")
    print(f"Output shape: {rendered.shape}")
    print(f"Alpha range: [{np.min(rendered[:,:,3]):.3f}, {np.max(rendered[:,:,3]):.3f}]")
    print(f"Color range: [{np.min(rendered[:,:,:3]):.3f}, {np.max(rendered[:,:,:3]):.3f}]")

    # Show rendering statistics
    stats = renderer.get_rendering_stats()
    print(f"\nRendering Statistics:")
    print(f"  Non-empty tiles: {stats['non_empty_tiles']}/{stats['total_tiles']}")
    print(f"  Total Gaussian assignments: {stats['total_gaussian_assignments']}")
    print(f"  Max Gaussians in tile: {stats['max_gaussians_in_tile']}")
    print(f"  Avg Gaussians per tile: {stats['avg_gaussians_per_tile']:.2f}")


def demo_configuration_comparison():
    """Compare different rendering configurations."""
    print("\n=== Configuration Comparison Demo ===")

    image_size = (64, 64)
    gaussians = create_test_gaussians(image_size)

    configs = [
        ("Default", RenderConfig()),
        ("Small Tiles", RenderConfig(tile_size=8)),
        ("Large Tiles", RenderConfig(tile_size=32)),
        ("High Top-K", RenderConfig(top_k=16)),
        ("Low Top-K", RenderConfig(top_k=4)),
        ("Debug Mode", RenderConfig(debug_mode=True))
    ]

    for name, config in configs:
        print(f"\n{name} Configuration:")
        print(f"  Tile size: {config.tile_size}")
        print(f"  Top-K: {config.top_k}")
        print(f"  Max Gaussians/tile: {config.max_gaussians_per_tile}")

        renderer = TileRenderer(image_size, config)

        start_time = time.time()
        rendered = renderer.render_full_image(gaussians)
        render_time = time.time() - start_time

        stats = renderer.get_rendering_stats()

        print(f"  Render time: {render_time:.4f}s")
        print(f"  Tiles: {stats['total_tiles']} ({stats['non_empty_tiles']} non-empty)")
        print(f"  Max Gaussians in tile: {stats['max_gaussians_in_tile']}")

        # Quality metrics
        total_alpha = np.sum(rendered[:,:,3])
        coverage = np.mean(rendered[:,:,3] > 0.01)

        print(f"  Total alpha: {total_alpha:.1f}")
        print(f"  Coverage: {coverage:.1%}")


def demo_performance_scaling():
    """Demonstrate performance scaling with Gaussian count."""
    print("\n=== Performance Scaling Demo ===")

    image_size = (128, 128)
    gaussian_counts = [10, 25, 50, 100, 200]

    for count in gaussian_counts:
        # Create Gaussians
        gaussians = []
        np.random.seed(42)  # Reproducible results

        for i in range(count):
            # Random position
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)

            # Random anisotropy
            scale_x = np.random.uniform(0.02, 0.08)
            scale_y = np.random.uniform(0.02, 0.08)
            theta = np.random.uniform(0, np.pi)

            # Random color
            color = np.random.uniform(0.2, 1.0, 3)
            alpha = np.random.uniform(0.3, 0.8)

            gaussian = create_anisotropic_gaussian(
                center=[x, y],
                scales=(scale_x, scale_y),
                orientation=theta,
                color=color,
                alpha=alpha
            )
            gaussians.append(gaussian)

        # Render and time
        renderer = TileRenderer(image_size)

        start_time = time.time()
        rendered = renderer.render_full_image(gaussians)
        render_time = time.time() - start_time

        stats = renderer.get_rendering_stats()

        print(f"{count:3d} Gaussians: {render_time:.4f}s "
              f"({render_time/count*1000:.2f}ms/Gaussian), "
              f"assignments: {stats['total_gaussian_assignments']}, "
              f"max/tile: {stats['max_gaussians_in_tile']}")


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("\n=== Edge Cases Demo ===")

    test_cases = [
        ("Tiny Image", (8, 8)),
        ("Tall Image", (128, 32)),
        ("Wide Image", (32, 128)),
        ("Single Pixel", (1, 1))
    ]

    for name, image_size in test_cases:
        print(f"\n{name}: {image_size}")

        try:
            # Create appropriate Gaussians
            if image_size[0] >= 8 and image_size[1] >= 8:
                gaussians = create_test_gaussians(image_size)[:3]  # Just a few
            else:
                # Single central Gaussian for tiny images
                gaussians = [create_isotropic_gaussian(
                    center=[0.5, 0.5],
                    scale=0.2,
                    color=[1.0, 0.5, 0.0],
                    alpha=0.8
                )]

            renderer = TileRenderer(image_size)
            rendered = renderer.render_full_image(gaussians)

            stats = renderer.get_rendering_stats()
            print(f"  Success: {stats['total_tiles']} tiles, "
                  f"{stats['total_gaussian_assignments']} assignments")

        except Exception as e:
            print(f"  Error: {e}")


def demo_gaussian_properties():
    """Demonstrate how different Gaussian properties affect rendering."""
    print("\n=== Gaussian Properties Demo ===")

    image_size = (64, 64)

    property_tests = [
        ("High Anisotropy", create_anisotropic_gaussian([0.5, 0.5], (0.01, 0.2), 0, [1,0,0], 0.8)),
        ("Low Anisotropy", create_anisotropic_gaussian([0.5, 0.5], (0.08, 0.1), 0, [0,1,0], 0.8)),
        ("High Alpha", create_isotropic_gaussian([0.5, 0.5], 0.1, [0,0,1], 0.95)),
        ("Low Alpha", create_isotropic_gaussian([0.5, 0.5], 0.1, [1,1,0], 0.2)),
        ("Large Scale", create_isotropic_gaussian([0.5, 0.5], 0.3, [1,0,1], 0.6)),
        ("Small Scale", create_isotropic_gaussian([0.5, 0.5], 0.02, [0,1,1], 0.9))
    ]

    renderer = TileRenderer(image_size)

    for name, gaussian in property_tests:
        print(f"\n{name}:")
        print(f"  Aspect ratio: {gaussian.aspect_ratio:.2f}")
        print(f"  Alpha: {gaussian.alpha:.2f}")

        # Compute 3σ radius
        radius = renderer.compute_3sigma_radius_px(gaussian)
        print(f"  3σ radius: {radius:.1f} pixels")

        # Render single Gaussian
        rendered = renderer.render_full_image([gaussian])

        # Analyze coverage
        alpha_channel = rendered[:,:,3]
        coverage = np.mean(alpha_channel > 0.01)
        max_alpha = np.max(alpha_channel)
        mean_alpha = np.mean(alpha_channel[alpha_channel > 0])

        print(f"  Coverage: {coverage:.1%}")
        print(f"  Max alpha: {max_alpha:.3f}")
        print(f"  Mean alpha (non-zero): {mean_alpha:.3f}")


def demo_tile_assignment_details():
    """Demonstrate detailed tile assignment analysis."""
    print("\n=== Tile Assignment Analysis Demo ===")

    image_size = (64, 64)

    # Create Gaussians with known properties
    gaussians = [
        # Central large Gaussian
        create_isotropic_gaussian([0.5, 0.5], 0.15, [1,0,0], 0.8),
        # Corner small Gaussian
        create_isotropic_gaussian([0.1, 0.1], 0.05, [0,1,0], 0.9),
        # Edge elongated Gaussian
        create_anisotropic_gaussian([0.8, 0.5], (0.02, 0.12), np.pi/2, [0,0,1], 0.7)
    ]

    renderer = TileRenderer(image_size, RenderConfig(debug_mode=True))

    print(f"Analyzing {len(gaussians)} Gaussians on {renderer.tiles_x}x{renderer.tiles_y} tile grid")

    # Assign and analyze
    renderer.assign_gaussians_to_tiles(gaussians)

    for i, gaussian in enumerate(gaussians):
        radius = renderer.compute_3sigma_radius_px(gaussian)
        center_px = (gaussian.mu[0] * image_size[1], gaussian.mu[1] * image_size[0])

        print(f"\nGaussian {i}:")
        print(f"  Position: ({center_px[0]:.1f}, {center_px[1]:.1f}) px")
        print(f"  3σ radius: {radius:.1f} px")
        print(f"  Aspect ratio: {gaussian.aspect_ratio:.2f}")

        # Count tile assignments
        assigned_tiles = []
        for y, tile_row in enumerate(renderer.tiles):
            for x, tile in enumerate(tile_row):
                if i in tile.gaussian_indices:
                    assigned_tiles.append((x, y))

        print(f"  Assigned to {len(assigned_tiles)} tiles: {assigned_tiles}")

    # Overall statistics
    stats = renderer.get_rendering_stats()
    print(f"\nOverall Assignment Statistics:")
    print(f"  Total assignments: {stats['total_gaussian_assignments']}")
    print(f"  Non-empty tiles: {stats['non_empty_tiles']}/{stats['total_tiles']}")
    print(f"  Avg assignments per tile: {stats['avg_gaussians_per_tile']:.2f}")
    print(f"  Max assignments in one tile: {stats['max_gaussians_in_tile']}")


if __name__ == "__main__":
    print("🎨 SplatThis Tile-Based Rendering Demonstration")
    print("=" * 60)

    demo_basic_rendering()
    demo_configuration_comparison()
    demo_performance_scaling()
    demo_edge_cases()
    demo_gaussian_properties()
    demo_tile_assignment_details()

    print("\n" + "=" * 60)
    print("✅ Tile Renderer Demo Complete!")
    print("🔧 Spatial binning: Efficient Gaussian-to-tile mapping")
    print("📊 Top-K blending: Quality-preserving pixel evaluation")
    print("⚡ Performance: Scalable rendering for adaptive splats")
    print("🎯 Ready for next phase: Error Computation and Analysis")