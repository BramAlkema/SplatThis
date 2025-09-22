#!/usr/bin/env python3
"""
Visualization utilities for error maps and progressive allocation debugging.

This module provides functions for visualizing reconstruction error maps,
probability distributions, and splat placements for debugging and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from typing import List, Tuple, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _canvas_to_rgb_array(fig) -> np.ndarray:
    """Convert matplotlib figure canvas to RGB array.

    This function handles different matplotlib backends that may not support
    the tostring_rgb() method.

    Args:
        fig: Matplotlib figure

    Returns:
        RGB array (H, W, 3) as uint8
    """
    fig.canvas.draw()
    try:
        # Try the standard method first
        rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        # Fallback for macOS and other backends that don't support tostring_rgb
        try:
            # Use buffer_rgba and convert to rgb
            buf = fig.canvas.buffer_rgba()
            rgb_array = np.asarray(buf)
            # Convert RGBA to RGB
            if rgb_array.shape[-1] == 4:
                rgb_array = rgb_array[..., :3]
        except AttributeError:
            # Final fallback: use print_to_buffer
            buf, size = fig.canvas.print_to_buffer()
            rgb_array = np.frombuffer(buf, dtype=np.uint8)
            # This may need manual reshaping based on the backend
            try:
                rgb_array = rgb_array.reshape(size[1], size[0], -1)
                if rgb_array.shape[-1] == 4:
                    rgb_array = rgb_array[..., :3]
            except ValueError:
                # If reshaping fails, create a dummy array
                logger.warning("Failed to convert canvas to RGB array, returning dummy array")
                rgb_array = np.zeros((100, 100, 3), dtype=np.uint8)

    return rgb_array.astype(np.uint8)


def visualize_error_map(
    error_map: np.ndarray,
    title: str = "Reconstruction Error Map",
    colormap: str = "hot",
    save_path: Optional[str] = None,
    show_colorbar: bool = True,
    figsize: Tuple[int, int] = (8, 6)
) -> np.ndarray:
    """Visualize reconstruction error map as a heatmap.

    Args:
        error_map: Per-pixel error map (H, W)
        title: Plot title
        colormap: Matplotlib colormap name
        save_path: Optional path to save the figure
        show_colorbar: Whether to show colorbar
        figsize: Figure size in inches

    Returns:
        RGB image array (H, W, 3) as uint8 for further processing
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create the heatmap
    im = ax.imshow(error_map, cmap=colormap, aspect='equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Add colorbar with error statistics
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Error Value', rotation=270, labelpad=20)

        # Add error statistics to colorbar
        max_error = np.max(error_map)
        mean_error = np.mean(error_map)
        cbar.ax.text(1.05, 0.95, f'Max: {max_error:.3f}', transform=cbar.ax.transAxes)
        cbar.ax.text(1.05, 0.90, f'Mean: {mean_error:.3f}', transform=cbar.ax.transAxes)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error map visualization saved to {save_path}")

    # Convert plot to RGB array
    rgb_array = _canvas_to_rgb_array(fig)

    plt.close(fig)
    return rgb_array


def visualize_probability_map(
    prob_map: np.ndarray,
    sampled_positions: Optional[List[Tuple[int, int]]] = None,
    title: str = "Probability Distribution",
    colormap: str = "viridis",
    save_path: Optional[str] = None,
    marker_size: int = 20,
    figsize: Tuple[int, int] = (8, 6)
) -> np.ndarray:
    """Visualize probability distribution with optional sample points.

    Args:
        prob_map: Probability distribution (H, W)
        sampled_positions: Optional list of (y, x) sampled positions
        title: Plot title
        colormap: Matplotlib colormap name
        save_path: Optional path to save the figure
        marker_size: Size of sample point markers
        figsize: Figure size in inches

    Returns:
        RGB image array (H, W, 3) as uint8
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create the probability heatmap
    im = ax.imshow(prob_map, cmap=colormap, aspect='equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Add sampled positions if provided
    if sampled_positions:
        y_coords, x_coords = zip(*sampled_positions)
        ax.scatter(x_coords, y_coords, c='red', s=marker_size,
                  marker='x', linewidth=2, label=f'{len(sampled_positions)} Samples')
        ax.legend(loc='upper right')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Probability', rotation=270, labelpad=20)

    # Add probability statistics
    prob_sum = np.sum(prob_map)
    max_prob = np.max(prob_map)
    cbar.ax.text(1.05, 0.95, f'Sum: {prob_sum:.3f}', transform=cbar.ax.transAxes)
    cbar.ax.text(1.05, 0.90, f'Max: {max_prob:.3f}', transform=cbar.ax.transAxes)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Probability map visualization saved to {save_path}")

    # Convert plot to RGB array
    rgb_array = _canvas_to_rgb_array(fig)

    plt.close(fig)
    return rgb_array


def visualize_side_by_side_comparison(
    target: np.ndarray,
    rendered: np.ndarray,
    error_map: np.ndarray,
    title: str = "Image Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> np.ndarray:
    """Create side-by-side comparison of target, rendered, and error images.

    Args:
        target: Target image (H, W) or (H, W, C)
        rendered: Rendered image (H, W) or (H, W, C)
        error_map: Error map (H, W)
        title: Overall plot title
        save_path: Optional path to save the figure
        figsize: Figure size in inches

    Returns:
        RGB image array of the comparison plot
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Target image
    if len(target.shape) == 3:
        axes[0].imshow(target)
    else:
        axes[0].imshow(target, cmap='gray')
    axes[0].set_title('Target Image', fontweight='bold')
    axes[0].axis('off')

    # Rendered image
    if len(rendered.shape) == 3:
        axes[1].imshow(rendered)
    else:
        axes[1].imshow(rendered, cmap='gray')
    axes[1].set_title('Rendered Image', fontweight='bold')
    axes[1].axis('off')

    # Error map
    im = axes[2].imshow(error_map, cmap='hot')
    axes[2].set_title('Error Map', fontweight='bold')
    axes[2].axis('off')

    # Add colorbar for error map
    cbar = fig.colorbar(im, ax=axes[2], shrink=0.8)
    cbar.set_label('Error', rotation=270, labelpad=15)

    # Overall title and statistics
    mean_error = np.mean(error_map)
    max_error = np.max(error_map)
    fig.suptitle(f'{title} (Mean Error: {mean_error:.3f}, Max Error: {max_error:.3f})',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Side-by-side comparison saved to {save_path}")

    # Convert plot to RGB array
    rgb_array = _canvas_to_rgb_array(fig)

    plt.close(fig)
    return rgb_array


def visualize_splat_placement(
    image: np.ndarray,
    splat_positions: List[Tuple[float, float]],
    splat_scales: Optional[List[float]] = None,
    title: str = "Splat Placement",
    save_path: Optional[str] = None,
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (10, 8)
) -> np.ndarray:
    """Visualize splat placements overlaid on the original image.

    Args:
        image: Background image (H, W) or (H, W, C)
        splat_positions: List of (x, y) splat center positions
        splat_scales: Optional list of splat scales for size visualization
        title: Plot title
        save_path: Optional path to save the figure
        alpha: Transparency for splat overlays
        figsize: Figure size in inches

    Returns:
        RGB image array of the visualization
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Display background image
    if len(image.shape) == 3:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')

    # Plot splat positions
    if splat_scales is not None:
        # Draw ellipses for splats with scale information
        for i, ((x, y), scale) in enumerate(zip(splat_positions, splat_scales)):
            ellipse = Ellipse((x, y), width=scale*2, height=scale*2,
                            facecolor='red', alpha=alpha, edgecolor='white', linewidth=1)
            ax.add_patch(ellipse)
    else:
        # Simple scatter plot for positions
        x_coords, y_coords = zip(*splat_positions) if splat_positions else ([], [])
        ax.scatter(x_coords, y_coords, c='red', s=20, marker='o', alpha=alpha,
                  edgecolors='white', linewidth=1)

    ax.set_title(f'{title} ({len(splat_positions)} splats)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Add legend
    if splat_scales is not None:
        scale_range = f"{min(splat_scales):.1f}-{max(splat_scales):.1f}"
        ax.text(0.02, 0.98, f'Scale range: {scale_range}',
               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Splat placement visualization saved to {save_path}")

    # Convert plot to RGB array
    rgb_array = _canvas_to_rgb_array(fig)

    plt.close(fig)
    return rgb_array


def visualize_progressive_allocation_steps(
    steps_data: List[dict],
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> List[np.ndarray]:
    """Visualize progressive allocation steps showing error evolution.

    Args:
        steps_data: List of dictionaries with keys: 'iteration', 'error_map',
                   'splat_count', 'mean_error', 'added_positions'
        save_dir: Optional directory to save individual step visualizations
        figsize: Figure size in inches

    Returns:
        List of RGB image arrays for each step
    """
    rgb_arrays = []

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, step in enumerate(steps_data):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Error map
        im1 = axes[0].imshow(step['error_map'], cmap='hot')
        axes[0].set_title(f"Error Map - Iteration {step['iteration']}", fontweight='bold')
        axes[0].axis('off')
        cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8)
        cbar1.set_label('Error', rotation=270, labelpad=15)

        # Error evolution plot
        iterations = [s['iteration'] for s in steps_data[:i+1]]
        errors = [s['mean_error'] for s in steps_data[:i+1]]
        splat_counts = [s['splat_count'] for s in steps_data[:i+1]]

        ax2 = axes[1]
        ax2.plot(iterations, errors, 'b-o', linewidth=2, markersize=6, label='Mean Error')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Mean Error', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True, alpha=0.3)

        # Secondary axis for splat count
        ax3 = ax2.twinx()
        ax3.plot(iterations, splat_counts, 'r-s', linewidth=2, markersize=4, label='Splat Count')
        ax3.set_ylabel('Splat Count', color='r')
        ax3.tick_params(axis='y', labelcolor='r')

        # Add current step info
        ax2.set_title(f"Progress - Step {i+1}/{len(steps_data)}", fontweight='bold')

        # Highlight current iteration
        if len(iterations) > 1:
            ax2.axvline(x=step['iteration'], color='gray', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save if requested
        if save_dir:
            save_path = Path(save_dir) / f"step_{i+1:03d}_iter_{step['iteration']:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.debug(f"Step visualization saved to {save_path}")

        # Convert to RGB array
        rgb_array = _canvas_to_rgb_array(fig)
        rgb_arrays.append(rgb_array)

        plt.close(fig)

    logger.info(f"Generated {len(rgb_arrays)} progressive allocation visualizations")
    return rgb_arrays


def create_error_histogram(
    error_map: np.ndarray,
    title: str = "Error Distribution",
    bins: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> np.ndarray:
    """Create histogram of error values for analysis.

    Args:
        error_map: Per-pixel error map (H, W)
        title: Plot title
        bins: Number of histogram bins
        save_path: Optional path to save the figure
        figsize: Figure size in inches

    Returns:
        RGB image array of the histogram plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Flatten error map and remove zeros for better visualization
    flat_errors = error_map.flatten()
    non_zero_errors = flat_errors[flat_errors > 0]

    # Create histogram
    n, bins_edges, patches = ax.hist(flat_errors, bins=bins, alpha=0.7,
                                   color='skyblue', edgecolor='black', linewidth=0.5)

    # Highlight non-zero errors
    if len(non_zero_errors) > 0:
        ax.hist(non_zero_errors, bins=bins, alpha=0.9,
               color='orange', edgecolor='black', linewidth=0.5,
               label=f'Non-zero errors ({len(non_zero_errors)} pixels)')
        ax.legend()

    # Add statistics
    mean_error = np.mean(flat_errors)
    std_error = np.std(flat_errors)
    max_error = np.max(flat_errors)

    ax.axvline(mean_error, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_error:.3f}')
    ax.axvline(mean_error + std_error, color='red', linestyle=':', alpha=0.7,
              label=f'Mean + Std: {mean_error + std_error:.3f}')

    ax.set_xlabel('Error Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title}\nMax: {max_error:.3f}, Std: {std_error:.3f}',
                fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error histogram saved to {save_path}")

    # Convert plot to RGB array
    rgb_array = _canvas_to_rgb_array(fig)

    plt.close(fig)
    return rgb_array


def export_to_png(
    image_array: np.ndarray,
    output_path: str,
    normalize: bool = False
) -> None:
    """Export numpy array as PNG image.

    Args:
        image_array: Image array (H, W) or (H, W, C)
        output_path: Output file path
        normalize: Whether to normalize values to [0, 255]
    """
    try:
        import imageio
    except ImportError:
        logger.error("imageio not available, cannot export PNG")
        return

    if normalize:
        # Normalize to [0, 255] range
        image_norm = ((image_array - image_array.min()) /
                     (image_array.max() - image_array.min()) * 255)
        image_norm = image_norm.astype(np.uint8)
    else:
        image_norm = image_array.astype(np.uint8)

    imageio.imwrite(output_path, image_norm)
    logger.info(f"Image exported to {output_path}")


def create_debug_summary(
    target: np.ndarray,
    rendered: np.ndarray,
    error_map: np.ndarray,
    prob_map: np.ndarray,
    sampled_positions: List[Tuple[int, int]],
    save_dir: str,
    prefix: str = "debug"
) -> dict:
    """Create comprehensive debug summary with all visualizations.

    Args:
        target: Target image
        rendered: Rendered image
        error_map: Error map
        prob_map: Probability map
        sampled_positions: Sampled positions
        save_dir: Directory to save all outputs
        prefix: Filename prefix

    Returns:
        Dictionary with paths to saved visualizations
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Error map visualization
    error_path = Path(save_dir) / f"{prefix}_error_map.png"
    visualize_error_map(error_map, save_path=str(error_path))
    saved_files['error_map'] = str(error_path)

    # Probability map visualization
    prob_path = Path(save_dir) / f"{prefix}_probability_map.png"
    visualize_probability_map(prob_map, sampled_positions, save_path=str(prob_path))
    saved_files['probability_map'] = str(prob_path)

    # Side-by-side comparison
    comparison_path = Path(save_dir) / f"{prefix}_comparison.png"
    visualize_side_by_side_comparison(target, rendered, error_map, save_path=str(comparison_path))
    saved_files['comparison'] = str(comparison_path)

    # Error histogram
    histogram_path = Path(save_dir) / f"{prefix}_error_histogram.png"
    create_error_histogram(error_map, save_path=str(histogram_path))
    saved_files['histogram'] = str(histogram_path)

    # Export raw arrays
    raw_error_path = Path(save_dir) / f"{prefix}_error_raw.png"
    export_to_png(error_map, str(raw_error_path), normalize=True)
    saved_files['raw_error'] = str(raw_error_path)

    raw_prob_path = Path(save_dir) / f"{prefix}_prob_raw.png"
    export_to_png(prob_map, str(raw_prob_path), normalize=True)
    saved_files['raw_probability'] = str(raw_prob_path)

    logger.info(f"Debug summary created in {save_dir} with {len(saved_files)} files")
    return saved_files