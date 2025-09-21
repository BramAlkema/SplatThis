"""Error computation and analysis for adaptive Gaussian splatting optimization.

This module provides comprehensive error metrics and analysis tools for evaluating
reconstruction quality and guiding optimization in adaptive Gaussian splatting.
"""

import math
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.filters import gaussian
from skimage.color import rgb2gray

logger = logging.getLogger(__name__)


@dataclass
class ErrorMetrics:
    """Container for various error metrics."""

    l1_error: float = 0.0           # Mean absolute error
    l2_error: float = 0.0           # Mean squared error
    rmse: float = 0.0               # Root mean squared error
    ssim_score: float = 0.0         # Structural similarity index
    psnr: float = 0.0               # Peak signal-to-noise ratio
    mse: float = 0.0                # Mean squared error (alias for l2_error)
    mae: float = 0.0                # Mean absolute error (alias for l1_error)

    # Perceptual metrics
    edge_error: float = 0.0         # Error in edge regions
    smooth_error: float = 0.0       # Error in smooth regions
    gradient_error: float = 0.0     # Gradient magnitude error

    # Coverage metrics
    coverage_ratio: float = 0.0     # Fraction of pixels with non-zero alpha
    alpha_mean: float = 0.0         # Mean alpha value
    alpha_std: float = 0.0          # Standard deviation of alpha

    def __post_init__(self):
        """Compute derived metrics."""
        # Aliases
        self.mae = self.l1_error
        self.mse = self.l2_error

        # RMSE from L2
        self.rmse = math.sqrt(self.l2_error) if self.l2_error >= 0 else 0.0

        # PSNR from MSE (assuming pixel values in [0,1])
        if self.mse > 1e-10:
            self.psnr = -10 * math.log10(self.mse)
        else:
            self.psnr = float('inf')

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging/serialization."""
        return {
            'l1_error': self.l1_error,
            'l2_error': self.l2_error,
            'rmse': self.rmse,
            'ssim_score': self.ssim_score,
            'psnr': self.psnr,
            'edge_error': self.edge_error,
            'smooth_error': self.smooth_error,
            'gradient_error': self.gradient_error,
            'coverage_ratio': self.coverage_ratio,
            'alpha_mean': self.alpha_mean,
            'alpha_std': self.alpha_std
        }


@dataclass
class ErrorRegion:
    """High-error region information."""

    center: Tuple[int, int]         # Region center (y, x)
    bbox: Tuple[int, int, int, int] # Bounding box (y1, x1, y2, x2)
    area: int                       # Region area in pixels
    mean_error: float               # Mean error in region
    max_error: float                # Maximum error in region
    error_type: str                 # Type of error ('edge', 'smooth', 'general')
    priority: float                 # Priority for addressing (higher = more urgent)

    @property
    def size(self) -> Tuple[int, int]:
        """Return region size as (height, width)."""
        y1, x1, y2, x2 = self.bbox
        return (y2 - y1, x2 - x1)


class ErrorAnalyzer:
    """Comprehensive error analysis for adaptive Gaussian splatting."""

    def __init__(self, window_size: int = 7, edge_threshold: float = 0.1):
        """
        Initialize error analyzer.

        Args:
            window_size: Window size for SSIM computation
            edge_threshold: Threshold for edge detection
        """
        self.window_size = window_size
        self.edge_threshold = edge_threshold

        # Error history for convergence tracking
        self.error_history: List[ErrorMetrics] = []

    def compute_basic_metrics(self, target: np.ndarray, rendered: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> ErrorMetrics:
        """
        Compute basic error metrics between target and rendered images.

        Args:
            target: Target image (H, W, C) or (H, W)
            rendered: Rendered image (H, W, C) or (H, W, 4)
            mask: Optional mask for error computation (H, W)

        Returns:
            ErrorMetrics instance with computed values
        """
        # Ensure compatible shapes and formats
        target_rgb, rendered_rgb = self._prepare_images(target, rendered)

        # Apply mask if provided
        if mask is not None:
            target_rgb = target_rgb * mask[..., np.newaxis]
            rendered_rgb = rendered_rgb * mask[..., np.newaxis]

        # Compute basic error metrics
        diff = target_rgb - rendered_rgb
        abs_diff = np.abs(diff)
        squared_diff = diff ** 2

        # L1 and L2 errors
        l1_error = np.mean(abs_diff)
        l2_error = np.mean(squared_diff)

        # Coverage metrics (if rendered has alpha channel)
        if rendered.ndim == 3 and rendered.shape[2] == 4:
            alpha_channel = rendered[:, :, 3]
            coverage_ratio = np.mean(alpha_channel > 0.01)
            alpha_mean = np.mean(alpha_channel)
            alpha_std = np.std(alpha_channel)
        else:
            coverage_ratio = 1.0  # Assume full coverage
            alpha_mean = 1.0
            alpha_std = 0.0

        metrics = ErrorMetrics(
            l1_error=l1_error,
            l2_error=l2_error,
            coverage_ratio=coverage_ratio,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std
        )

        return metrics

    def compute_ssim(self, target: np.ndarray, rendered: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Structural Similarity Index (SSIM).

        Args:
            target: Target image
            rendered: Rendered image
            mask: Optional mask for computation

        Returns:
            SSIM score [0, 1] (1 = perfect similarity)
        """
        target_rgb, rendered_rgb = self._prepare_images(target, rendered)

        # Convert to grayscale for SSIM
        if target_rgb.shape[2] == 3:
            target_gray = rgb2gray(target_rgb)
            rendered_gray = rgb2gray(rendered_rgb)
        else:
            target_gray = target_rgb[:, :, 0]
            rendered_gray = rendered_rgb[:, :, 0]

        # Apply mask if provided
        if mask is not None:
            target_gray = target_gray * mask
            rendered_gray = rendered_gray * mask

        try:
            ssim_score = ssim(target_gray, rendered_gray,
                            win_size=self.window_size,
                            data_range=1.0)
        except Exception as e:
            logger.warning(f"SSIM computation failed: {e}")
            ssim_score = 0.0

        return float(ssim_score)

    def compute_perceptual_metrics(self, target: np.ndarray, rendered: np.ndarray,
                                 mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute perceptual error metrics including edge and gradient errors.

        Args:
            target: Target image
            rendered: Rendered image
            mask: Optional mask for computation

        Returns:
            Dictionary of perceptual metrics
        """
        target_rgb, rendered_rgb = self._prepare_images(target, rendered)

        # Convert to grayscale for edge analysis
        target_gray = rgb2gray(target_rgb) if target_rgb.shape[2] == 3 else target_rgb[:, :, 0]
        rendered_gray = rgb2gray(rendered_rgb) if rendered_rgb.shape[2] == 3 else rendered_rgb[:, :, 0]

        # Detect edges in both images
        try:
            from skimage.feature import canny
            target_edges = canny(target_gray, sigma=1.0, low_threshold=self.edge_threshold/2,
                               high_threshold=self.edge_threshold)
            rendered_edges = canny(rendered_gray, sigma=1.0, low_threshold=self.edge_threshold/2,
                                 high_threshold=self.edge_threshold)
        except Exception:
            # Fallback to gradient-based edge detection
            target_edges = self._gradient_edges(target_gray)
            rendered_edges = self._gradient_edges(rendered_gray)

        # Compute gradients
        target_grad_x = ndimage.sobel(target_gray, axis=1)
        target_grad_y = ndimage.sobel(target_gray, axis=0)
        target_grad_mag = np.sqrt(target_grad_x**2 + target_grad_y**2)

        rendered_grad_x = ndimage.sobel(rendered_gray, axis=1)
        rendered_grad_y = ndimage.sobel(rendered_gray, axis=0)
        rendered_grad_mag = np.sqrt(rendered_grad_x**2 + rendered_grad_y**2)

        # Apply mask if provided
        if mask is not None:
            target_edges = target_edges * mask
            rendered_edges = rendered_edges * mask
            target_grad_mag = target_grad_mag * mask
            rendered_grad_mag = rendered_grad_mag * mask

        # Edge error (difference in edge maps)
        edge_error = np.mean(np.abs(target_edges.astype(float) - rendered_edges.astype(float)))

        # Smooth region error (error in non-edge regions)
        smooth_mask = ~(target_edges | rendered_edges)
        if np.any(smooth_mask):
            if mask is not None:
                smooth_mask = smooth_mask & mask
            target_smooth = target_rgb[smooth_mask]
            rendered_smooth = rendered_rgb[smooth_mask]
            smooth_error = np.mean(np.abs(target_smooth - rendered_smooth))
        else:
            smooth_error = 0.0

        # Gradient magnitude error
        gradient_error = np.mean(np.abs(target_grad_mag - rendered_grad_mag))

        return {
            'edge_error': edge_error,
            'smooth_error': smooth_error,
            'gradient_error': gradient_error
        }

    def compute_comprehensive_metrics(self, target: np.ndarray, rendered: np.ndarray,
                                    mask: Optional[np.ndarray] = None) -> ErrorMetrics:
        """
        Compute comprehensive error metrics combining all approaches.

        Args:
            target: Target image
            rendered: Rendered image
            mask: Optional mask for computation

        Returns:
            Complete ErrorMetrics instance
        """
        # Basic metrics
        metrics = self.compute_basic_metrics(target, rendered, mask)

        # SSIM
        metrics.ssim_score = self.compute_ssim(target, rendered, mask)

        # Perceptual metrics
        perceptual = self.compute_perceptual_metrics(target, rendered, mask)
        metrics.edge_error = perceptual['edge_error']
        metrics.smooth_error = perceptual['smooth_error']
        metrics.gradient_error = perceptual['gradient_error']

        # Update derived metrics
        metrics.__post_init__()

        return metrics

    def create_error_map(self, target: np.ndarray, rendered: np.ndarray,
                        error_type: str = 'l1') -> np.ndarray:
        """
        Create spatial error map showing per-pixel errors.

        Args:
            target: Target image
            rendered: Rendered image
            error_type: Type of error ('l1', 'l2', 'ssim_local')

        Returns:
            Error map (H, W) with per-pixel error values
        """
        target_rgb, rendered_rgb = self._prepare_images(target, rendered)

        if error_type == 'l1':
            # L1 (absolute) error per pixel
            diff = np.abs(target_rgb - rendered_rgb)
            error_map = np.mean(diff, axis=2)  # Average across channels

        elif error_type == 'l2':
            # L2 (squared) error per pixel
            diff = (target_rgb - rendered_rgb) ** 2
            error_map = np.mean(diff, axis=2)  # Average across channels

        elif error_type == 'ssim_local':
            # Local SSIM error
            target_gray = rgb2gray(target_rgb) if target_rgb.shape[2] == 3 else target_rgb[:, :, 0]
            rendered_gray = rgb2gray(rendered_rgb) if rendered_rgb.shape[2] == 3 else rendered_rgb[:, :, 0]

            # Compute local SSIM using sliding window
            error_map = self._compute_local_ssim_error(target_gray, rendered_gray)

        else:
            raise ValueError(f"Unknown error type: {error_type}")

        return error_map

    def detect_high_error_regions(self, error_map: np.ndarray,
                                threshold: Optional[float] = None,
                                min_area: int = 10) -> List[ErrorRegion]:
        """
        Detect regions with high reconstruction error.

        Args:
            error_map: Per-pixel error map
            threshold: Error threshold (auto-computed if None)
            min_area: Minimum region area in pixels

        Returns:
            List of high-error regions
        """
        if threshold is None:
            # Auto-compute threshold as mean + 2*std
            threshold = np.mean(error_map) + 2 * np.std(error_map)

        # Create binary mask of high-error pixels
        high_error_mask = error_map > threshold

        # Find connected components
        try:
            from skimage.measure import label, regionprops
            labeled_regions = label(high_error_mask)
            props = regionprops(labeled_regions, intensity_image=error_map)
        except ImportError:
            # Fallback to scipy
            labeled_regions, num_regions = ndimage.label(high_error_mask)
            props = []
            for i in range(1, num_regions + 1):
                region_mask = labeled_regions == i
                if np.sum(region_mask) >= min_area:
                    coords = np.where(region_mask)
                    y_coords, x_coords = coords

                    # Create a simple region object
                    class SimpleRegion:
                        def __init__(self):
                            self.area = len(y_coords)
                            self.centroid = (np.mean(y_coords), np.mean(x_coords))
                            self.bbox = (np.min(y_coords), np.min(x_coords),
                                       np.max(y_coords) + 1, np.max(x_coords) + 1)
                            self.mean_intensity = np.mean(error_map[region_mask])
                            self.max_intensity = np.max(error_map[region_mask])

                    props.append(SimpleRegion())

        # Convert to ErrorRegion objects
        regions = []
        for prop in props:
            if prop.area >= min_area:
                # Determine error type based on region characteristics
                error_type = self._classify_error_region(error_map, prop)

                # Compute priority based on area and error magnitude
                priority = prop.area * prop.mean_intensity

                region = ErrorRegion(
                    center=(int(prop.centroid[0]), int(prop.centroid[1])),
                    bbox=prop.bbox,
                    area=prop.area,
                    mean_error=prop.mean_intensity,
                    max_error=prop.max_intensity,
                    error_type=error_type,
                    priority=priority
                )
                regions.append(region)

        # Sort by priority (highest first)
        regions.sort(key=lambda r: r.priority, reverse=True)

        return regions

    def track_error_history(self, metrics: ErrorMetrics) -> None:
        """
        Add error metrics to history for convergence analysis.

        Args:
            metrics: Error metrics to add to history
        """
        self.error_history.append(metrics)

        # Keep reasonable history length
        max_history = 1000
        if len(self.error_history) > max_history:
            self.error_history = self.error_history[-max_history:]

    def analyze_convergence(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze error convergence from history.

        Args:
            window_size: Window size for trend analysis

        Returns:
            Convergence analysis results
        """
        if len(self.error_history) < window_size:
            return {
                'converged': False,
                'trend': 'insufficient_data',
                'improvement_rate': 0.0,
                'plateau_detected': False
            }

        # Extract recent error values
        recent_errors = [m.l2_error for m in self.error_history[-window_size:]]

        # Compute trend
        x = np.arange(len(recent_errors))
        coeffs = np.polyfit(x, recent_errors, 1)
        slope = coeffs[0]

        # Analyze improvement
        improvement_rate = -slope / recent_errors[0] if recent_errors[0] > 0 else 0.0

        # Detect plateau (small changes)
        error_std = np.std(recent_errors)
        error_mean = np.mean(recent_errors)
        relative_std = error_std / error_mean if error_mean > 0 else float('inf')
        plateau_detected = relative_std < 0.01  # Less than 1% relative variation

        # Convergence criteria
        converged = plateau_detected and improvement_rate < 0.001

        return {
            'converged': converged,
            'trend': 'decreasing' if slope < -1e-6 else ('increasing' if slope > 1e-6 else 'stable'),
            'improvement_rate': improvement_rate,
            'plateau_detected': plateau_detected,
            'recent_error_std': error_std,
            'recent_error_mean': error_mean,
            'slope': slope
        }

    def create_quality_report(self, target: np.ndarray, rendered: np.ndarray,
                            include_regions: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive quality assessment report.

        Args:
            target: Target image
            rendered: Rendered image
            include_regions: Whether to include high-error region analysis

        Returns:
            Comprehensive quality report
        """
        # Compute comprehensive metrics
        metrics = self.compute_comprehensive_metrics(target, rendered)

        # Track in history
        self.track_error_history(metrics)

        # Create error maps
        l1_map = self.create_error_map(target, rendered, 'l1')
        l2_map = self.create_error_map(target, rendered, 'l2')

        report = {
            'metrics': metrics.to_dict(),
            'error_statistics': {
                'l1_map_mean': float(np.mean(l1_map)),
                'l1_map_std': float(np.std(l1_map)),
                'l1_map_max': float(np.max(l1_map)),
                'l2_map_mean': float(np.mean(l2_map)),
                'l2_map_std': float(np.std(l2_map)),
                'l2_map_max': float(np.max(l2_map))
            }
        }

        # High-error regions analysis
        if include_regions:
            regions = self.detect_high_error_regions(l1_map)
            report['high_error_regions'] = {
                'count': len(regions),
                'total_area': sum(r.area for r in regions),
                'regions': [
                    {
                        'center': r.center,
                        'area': r.area,
                        'mean_error': r.mean_error,
                        'error_type': r.error_type,
                        'priority': r.priority
                    }
                    for r in regions[:10]  # Top 10 regions
                ]
            }

        # Convergence analysis
        if len(self.error_history) > 1:
            convergence = self.analyze_convergence()
            report['convergence'] = convergence

        return report

    def _prepare_images(self, target: np.ndarray, rendered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare images for error computation (ensure compatible formats)."""
        # Handle grayscale vs color
        if target.ndim == 2:
            target = target[..., np.newaxis]
        if rendered.ndim == 2:
            rendered = rendered[..., np.newaxis]

        # Extract RGB channels from rendered (drop alpha if present)
        if rendered.shape[2] == 4:
            rendered_rgb = rendered[:, :, :3]
        else:
            rendered_rgb = rendered

        # Ensure target has same number of channels
        if target.shape[2] == 1 and rendered_rgb.shape[2] == 3:
            target_rgb = np.repeat(target, 3, axis=2)
        elif target.shape[2] == 3 and rendered_rgb.shape[2] == 1:
            rendered_rgb = np.repeat(rendered_rgb, 3, axis=2)
        else:
            target_rgb = target

        # Ensure same spatial dimensions
        if target_rgb.shape[:2] != rendered_rgb.shape[:2]:
            raise ValueError(f"Image dimensions must match: {target_rgb.shape[:2]} vs {rendered_rgb.shape[:2]}")

        # Ensure float type and [0,1] range
        target_rgb = np.clip(target_rgb.astype(np.float32), 0.0, 1.0)
        rendered_rgb = np.clip(rendered_rgb.astype(np.float32), 0.0, 1.0)

        return target_rgb, rendered_rgb

    def _gradient_edges(self, image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Simple gradient-based edge detection fallback."""
        grad_x = ndimage.sobel(image, axis=1)
        grad_y = ndimage.sobel(image, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return grad_mag > threshold

    def _compute_local_ssim_error(self, target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """Compute local SSIM error using sliding window."""
        H, W = target.shape
        error_map = np.zeros((H, W))
        win_size = self.window_size
        half_win = win_size // 2

        for y in range(half_win, H - half_win):
            for x in range(half_win, W - half_win):
                # Extract local windows
                target_win = target[y-half_win:y+half_win+1, x-half_win:x+half_win+1]
                rendered_win = rendered[y-half_win:y+half_win+1, x-half_win:x+half_win+1]

                # Compute local SSIM
                try:
                    local_ssim = ssim(target_win, rendered_win, data_range=1.0)
                    error_map[y, x] = 1.0 - local_ssim  # Convert to error
                except Exception:
                    error_map[y, x] = np.mean(np.abs(target_win - rendered_win))

        return error_map

    def _classify_error_region(self, error_map: np.ndarray, region) -> str:
        """Classify error region type based on characteristics."""
        # Simple classification based on region size and error pattern
        if region.area > 100:
            return 'smooth'  # Large regions likely in smooth areas
        elif region.max_intensity > 2 * region.mean_intensity:
            return 'edge'    # High peak error likely at edges
        else:
            return 'general' # General reconstruction error


# Convenience functions
def compute_reconstruction_error(target: np.ndarray, rendered: np.ndarray,
                               mask: Optional[np.ndarray] = None) -> ErrorMetrics:
    """Convenience function to compute basic reconstruction error."""
    analyzer = ErrorAnalyzer()
    return analyzer.compute_comprehensive_metrics(target, rendered, mask)


def create_error_visualization(target: np.ndarray, rendered: np.ndarray,
                             error_type: str = 'l1') -> Dict[str, np.ndarray]:
    """Convenience function to create error visualization."""
    analyzer = ErrorAnalyzer()
    error_map = analyzer.create_error_map(target, rendered, error_type)

    return {
        'error_map': error_map,
        'target': target,
        'rendered': rendered,
        'difference': np.abs(target - rendered) if target.shape == rendered.shape else None
    }