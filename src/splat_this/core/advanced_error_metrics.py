"""Advanced error metrics for adaptive Gaussian splatting optimization.

This module extends the basic error analysis with state-of-the-art perceptual
and frequency-domain metrics for comprehensive quality assessment.
"""

import math
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq
from skimage.metrics import structural_similarity as ssim
from skimage.filters import gaussian, sobel_h, sobel_v
from skimage.color import rgb2gray
from skimage.segmentation import felzenszwalb, slic

from .error_analysis import ErrorMetrics, ErrorAnalyzer, ErrorRegion

logger = logging.getLogger(__name__)


class PerceptualMetric(Enum):
    """Available perceptual metrics."""
    LPIPS_VGG = "lpips_vgg"           # LPIPS with VGG features
    LPIPS_ALEX = "lpips_alex"         # LPIPS with AlexNet features
    SSIM_MULTISCALE = "ssim_ms"       # Multi-scale SSIM
    GRADIENT_SIMILARITY = "grad_sim"   # Gradient-based similarity
    TEXTURE_SIMILARITY = "texture_sim" # Texture-based similarity


class FrequencyBand(Enum):
    """Frequency bands for spectral analysis."""
    LOW = "low"                       # 0-25% of Nyquist frequency
    MID_LOW = "mid_low"              # 25-50% of Nyquist frequency
    MID_HIGH = "mid_high"            # 50-75% of Nyquist frequency
    HIGH = "high"                     # 75-100% of Nyquist frequency


@dataclass
class AdvancedErrorMetrics:
    """Container for advanced error metrics."""

    # Perceptual metrics
    lpips_score: float = 0.0                    # LPIPS perceptual distance
    ms_ssim_score: float = 0.0                  # Multi-scale SSIM
    gradient_similarity: float = 0.0            # Gradient-based similarity
    texture_similarity: float = 0.0             # Texture-based similarity
    edge_coherence: float = 0.0                 # Edge structure preservation

    # Frequency-domain metrics
    frequency_response: Dict[str, float] = field(default_factory=dict)  # Per-band frequency errors
    spectral_distortion: float = 0.0            # Overall spectral distortion
    high_freq_preservation: float = 0.0         # High-frequency detail preservation

    # Content-aware metrics
    content_weighted_error: float = 0.0         # Content-importance weighted error
    saliency_weighted_error: float = 0.0        # Saliency-weighted error
    semantic_consistency: float = 0.0           # Semantic region consistency

    # Comparative metrics
    quality_rank: Optional[int] = None          # Quality ranking (1=best)
    preference_score: float = 0.0               # Preference probability

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = {
            'lpips_score': self.lpips_score,
            'ms_ssim_score': self.ms_ssim_score,
            'gradient_similarity': self.gradient_similarity,
            'texture_similarity': self.texture_similarity,
            'edge_coherence': self.edge_coherence,
            'spectral_distortion': self.spectral_distortion,
            'high_freq_preservation': self.high_freq_preservation,
            'content_weighted_error': self.content_weighted_error,
            'saliency_weighted_error': self.saliency_weighted_error,
            'semantic_consistency': self.semantic_consistency,
            'quality_rank': self.quality_rank,
            'preference_score': self.preference_score
        }
        result.update(self.frequency_response)
        return result


@dataclass
class ContentRegion:
    """Content-aware region classification."""

    region_id: int                               # Unique region identifier
    bbox: Tuple[int, int, int, int]             # Bounding box (y1, x1, y2, x2)
    area: int                                   # Region area in pixels
    content_type: str                           # Content classification
    importance_weight: float                     # Visual importance weight
    texture_complexity: float                   # Local texture complexity
    edge_density: float                         # Edge density in region

    @property
    def size(self) -> Tuple[int, int]:
        """Return region size as (height, width)."""
        y1, x1, y2, x2 = self.bbox
        return (y2 - y1, x2 - x1)


class LPIPSCalculator:
    """LPIPS (Learned Perceptual Image Patch Similarity) calculator.

    This is a simplified implementation that uses hand-crafted features
    instead of deep learning features for compatibility.
    """

    def __init__(self, metric_type: PerceptualMetric = PerceptualMetric.LPIPS_VGG):
        """Initialize LPIPS calculator."""
        self.metric_type = metric_type
        self.patch_size = 64
        self.stride = 32

        # Feature extraction weights (simplified VGG-like features)
        self.conv_weights = self._init_feature_weights()

    def compute_lpips(self, target: np.ndarray, rendered: np.ndarray) -> float:
        """
        Compute LPIPS score between target and rendered images.

        Args:
            target: Target image (H, W, C)
            rendered: Rendered image (H, W, C)

        Returns:
            LPIPS score (lower is better, 0 = identical)
        """
        # Convert to grayscale if needed
        target_gray = rgb2gray(target) if target.ndim == 3 and target.shape[2] == 3 else target
        rendered_gray = rgb2gray(rendered) if rendered.ndim == 3 and rendered.shape[2] == 3 else rendered

        # Extract patches
        target_patches = self._extract_patches(target_gray)
        rendered_patches = self._extract_patches(rendered_gray)

        # Compute features for each patch
        target_features = [self._extract_features(patch) for patch in target_patches]
        rendered_features = [self._extract_features(patch) for patch in rendered_patches]

        # Compute perceptual distances
        distances = []
        for tf, rf in zip(target_features, rendered_features):
            # Cosine similarity in feature space
            dot_product = np.sum(tf * rf)
            norm_product = np.linalg.norm(tf) * np.linalg.norm(rf)

            if norm_product > 1e-8:
                similarity = dot_product / norm_product
                distance = 1.0 - similarity
            else:
                distance = 0.0

            distances.append(max(0.0, distance))  # Ensure non-negative

        # Average distance across patches
        lpips_score = np.mean(distances) if distances else 0.0

        return float(max(0.0, lpips_score))  # Ensure non-negative

    def _init_feature_weights(self) -> List[np.ndarray]:
        """Initialize simplified feature extraction weights."""
        # Simplified Gabor-like filters for texture analysis
        weights = []

        # Different orientations and frequencies
        for angle in [0, 45, 90, 135]:
            for freq in [0.1, 0.2, 0.3]:
                kernel = self._gabor_kernel(angle, freq, size=5)
                weights.append(kernel)

        return weights

    def _gabor_kernel(self, angle: float, frequency: float, size: int = 5) -> np.ndarray:
        """Create Gabor filter kernel."""
        angle_rad = np.deg2rad(angle)
        sigma = size / 6.0

        # Create coordinate grids (ensure odd size)
        if size % 2 == 0:
            size += 1
        half_size = size // 2
        x, y = np.meshgrid(np.arange(-half_size, half_size+1), np.arange(-half_size, half_size+1))

        # Rotate coordinates
        x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
        y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)

        # Gabor function
        gaussian = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * frequency * x_rot)

        gabor = gaussian * sinusoid

        # Normalize
        gabor = gabor / np.sum(np.abs(gabor))

        return gabor

    def _extract_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract overlapping patches from image."""
        patches = []
        H, W = image.shape[:2]

        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        return patches

    def _extract_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract features from a patch using simplified VGG-like processing."""
        features = []

        # Apply each filter
        for kernel in self.conv_weights:
            # Convolve with filter
            response = ndimage.convolve(patch, kernel, mode='constant')

            # Pool (average)
            pooled = np.mean(np.abs(response))
            features.append(pooled)

        # Add some basic statistics
        features.extend([
            np.mean(patch),
            np.std(patch),
            np.min(patch),
            np.max(patch)
        ])

        return np.array(features)


class FrequencyAnalyzer:
    """Frequency-domain error analysis."""

    def __init__(self):
        """Initialize frequency analyzer."""
        self.frequency_bands = {
            FrequencyBand.LOW: (0.0, 0.25),
            FrequencyBand.MID_LOW: (0.25, 0.5),
            FrequencyBand.MID_HIGH: (0.5, 0.75),
            FrequencyBand.HIGH: (0.75, 1.0)
        }

    def compute_frequency_metrics(self, target: np.ndarray, rendered: np.ndarray) -> Dict[str, float]:
        """
        Compute frequency-domain error metrics.

        Args:
            target: Target image
            rendered: Rendered image

        Returns:
            Dictionary of frequency metrics
        """
        # Convert to grayscale for frequency analysis
        target_gray = rgb2gray(target) if target.ndim == 3 and target.shape[2] == 3 else target
        rendered_gray = rgb2gray(rendered) if rendered.ndim == 3 and rendered.shape[2] == 3 else rendered

        # Compute FFT
        target_fft = fft2(target_gray)
        rendered_fft = fft2(rendered_gray)

        # Get frequency coordinates
        H, W = target_gray.shape
        freq_y = fftfreq(H)
        freq_x = fftfreq(W)
        fy, fx = np.meshgrid(freq_y, freq_x, indexing='ij')
        freq_magnitude = np.sqrt(fy**2 + fx**2)

        # Normalize frequency magnitude to [0, 1]
        max_freq = np.sqrt(0.5**2 + 0.5**2)  # Maximum possible frequency
        freq_magnitude_norm = freq_magnitude / max_freq

        # Compute per-band errors
        band_errors = {}
        for band, (low_freq, high_freq) in self.frequency_bands.items():
            # Create band mask
            band_mask = (freq_magnitude_norm >= low_freq) & (freq_magnitude_norm < high_freq)

            if np.any(band_mask):
                # Extract band components
                target_band = target_fft * band_mask
                rendered_band = rendered_fft * band_mask

                # Compute error in frequency domain
                freq_error = np.mean(np.abs(target_band - rendered_band)**2)
                band_errors[f"freq_{band.value}"] = float(freq_error)
            else:
                band_errors[f"freq_{band.value}"] = 0.0

        # Overall spectral distortion
        spectral_error = np.mean(np.abs(target_fft - rendered_fft)**2)

        # High-frequency preservation
        high_freq_mask = freq_magnitude_norm >= 0.5
        if np.any(high_freq_mask):
            target_high = np.abs(target_fft * high_freq_mask)
            rendered_high = np.abs(rendered_fft * high_freq_mask)

            target_high_energy = np.sum(target_high)
            rendered_high_energy = np.sum(rendered_high)

            if target_high_energy > 1e-8:
                high_freq_preservation = rendered_high_energy / target_high_energy
                high_freq_preservation = min(high_freq_preservation, 1.0)  # Cap at 1.0
            else:
                high_freq_preservation = 1.0
        else:
            high_freq_preservation = 1.0

        metrics = {
            'spectral_distortion': float(spectral_error),
            'high_freq_preservation': float(high_freq_preservation)
        }
        metrics.update(band_errors)

        return metrics


class ContentAwareAnalyzer:
    """Content-aware error analysis with semantic region classification."""

    def __init__(self, num_segments: int = 50):
        """Initialize content-aware analyzer."""
        self.num_segments = num_segments
        self.edge_detector_sigma = 1.0

    def analyze_content_regions(self, image: np.ndarray) -> List[ContentRegion]:
        """
        Analyze image content and classify regions.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            List of content regions
        """
        # Convert to appropriate format for segmentation
        if image.ndim == 3 and image.shape[2] == 3:
            # Use color image for segmentation
            seg_image = image
        else:
            # Convert grayscale to RGB for segmentation
            if image.ndim == 2:
                seg_image = np.stack([image, image, image], axis=2)
            else:
                seg_image = image

        # Perform superpixel segmentation
        try:
            segments = slic(seg_image, n_segments=self.num_segments, compactness=10, start_label=1)
        except Exception:
            # Fallback to simpler segmentation
            segments = felzenszwalb(seg_image, scale=100, sigma=0.5, min_size=50)

        # Analyze each segment
        regions = []
        unique_segments = np.unique(segments)

        for seg_id in unique_segments:
            if seg_id == 0:  # Skip background
                continue

            # Extract segment mask
            mask = segments == seg_id
            coords = np.where(mask)

            if len(coords[0]) == 0:
                continue

            # Compute bounding box
            y_coords, x_coords = coords
            bbox = (np.min(y_coords), np.min(x_coords),
                   np.max(y_coords) + 1, np.max(x_coords) + 1)

            # Compute region properties
            area = len(y_coords)

            # Content classification
            content_type = self._classify_content(seg_image, mask)

            # Importance weighting
            importance_weight = self._compute_importance_weight(seg_image, mask, content_type)

            # Texture complexity
            texture_complexity = self._compute_texture_complexity(seg_image, mask)

            # Edge density
            edge_density = self._compute_edge_density(seg_image, mask)

            region = ContentRegion(
                region_id=int(seg_id),
                bbox=bbox,
                area=area,
                content_type=content_type,
                importance_weight=importance_weight,
                texture_complexity=texture_complexity,
                edge_density=edge_density
            )
            regions.append(region)

        return regions

    def compute_content_weighted_error(self, target: np.ndarray, rendered: np.ndarray,
                                     regions: List[ContentRegion]) -> float:
        """
        Compute content-weighted error using region importance.

        Args:
            target: Target image
            rendered: Rendered image
            regions: Content regions with importance weights

        Returns:
            Content-weighted error
        """
        # Prepare images
        target_gray = rgb2gray(target) if target.ndim == 3 and target.shape[2] == 3 else target
        rendered_gray = rgb2gray(rendered) if rendered.ndim == 3 and rendered.shape[2] == 3 else rendered

        # Compute per-pixel errors
        pixel_errors = np.abs(target_gray - rendered_gray)

        # Create importance weight map
        H, W = target_gray.shape
        weight_map = np.ones((H, W))

        # Reconstruct segmentation from regions (simplified)
        for region in regions:
            y1, x1, y2, x2 = region.bbox
            weight_map[y1:y2, x1:x2] *= region.importance_weight

        # Compute weighted error
        weighted_errors = pixel_errors * weight_map
        total_weight = np.sum(weight_map)

        if total_weight > 0:
            content_weighted_error = np.sum(weighted_errors) / total_weight
        else:
            content_weighted_error = np.mean(pixel_errors)

        return float(content_weighted_error)

    def _classify_content(self, image: np.ndarray, mask: np.ndarray) -> str:
        """Classify content type of a region."""
        # Extract region pixels
        if image.ndim == 3:
            region_pixels = image[mask]
        else:
            region_pixels = image[mask, np.newaxis]

        # Compute statistics
        mean_intensity = np.mean(region_pixels)
        std_intensity = np.std(region_pixels)

        # Simple heuristic classification (order matters)
        if mean_intensity > 0.7:
            return "bright"       # High intensity = bright region
        elif mean_intensity < 0.3:
            return "dark"         # Low intensity = dark region
        elif std_intensity > 0.3:
            return "textured"     # High variation = textured region
        elif std_intensity < 0.1:
            return "smooth"       # Low variation = smooth region
        else:
            return "general"      # Default classification

    def _compute_importance_weight(self, image: np.ndarray, mask: np.ndarray, content_type: str) -> float:
        """Compute visual importance weight for a region."""
        # Base weights by content type
        type_weights = {
            "textured": 1.5,      # Textured regions are visually important
            "bright": 1.2,        # Bright regions draw attention
            "smooth": 0.8,        # Smooth regions less important
            "dark": 0.9,          # Dark regions moderately important
            "general": 1.0        # Default weight
        }

        base_weight = type_weights.get(content_type, 1.0)

        # Adjust based on region properties
        region_size = np.sum(mask)
        size_factor = min(region_size / 1000.0, 1.0)  # Larger regions slightly more important

        return base_weight * (0.8 + 0.2 * size_factor)

    def _compute_texture_complexity(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Compute texture complexity for a region."""
        # Convert to grayscale for texture analysis
        if image.ndim == 3:
            gray_image = rgb2gray(image)
        else:
            gray_image = image

        # Compute gradients
        grad_x = sobel_h(gray_image)
        grad_y = sobel_v(gray_image)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Extract region gradients
        region_gradients = grad_magnitude[mask]

        # Texture complexity as gradient variation
        if len(region_gradients) > 0:
            complexity = np.std(region_gradients)
        else:
            complexity = 0.0

        return float(complexity)

    def _compute_edge_density(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Compute edge density for a region."""
        # Convert to grayscale for edge detection
        if image.ndim == 3:
            gray_image = rgb2gray(image)
        else:
            gray_image = image

        # Detect edges
        try:
            from skimage.feature import canny
            edges = canny(gray_image, sigma=self.edge_detector_sigma)
        except Exception:
            # Fallback to gradient-based edges
            grad_x = sobel_h(gray_image)
            grad_y = sobel_v(gray_image)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            edges = grad_magnitude > 0.1

        # Compute edge density in region
        region_edges = edges[mask]
        if len(region_edges) > 0:
            edge_density = np.mean(region_edges)
        else:
            edge_density = 0.0

        return float(edge_density)


class ComparativeQualityAssessment:
    """Framework for comparing multiple reconstruction methods."""

    def __init__(self):
        """Initialize comparative assessment framework."""
        self.methods_compared = 0
        self.comparison_history: List[Dict[str, Any]] = []

    def compare_methods(self, target: np.ndarray,
                       reconstructions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple reconstruction methods.

        Args:
            target: Target image
            reconstructions: Dictionary mapping method names to reconstructed images

        Returns:
            Comparison results with rankings and statistics
        """
        if len(reconstructions) < 2:
            raise ValueError("Need at least 2 methods to compare")

        # Initialize advanced analyzer
        lpips_calc = LPIPSCalculator()
        freq_analyzer = FrequencyAnalyzer()
        content_analyzer = ContentAwareAnalyzer()

        # Analyze content regions (once for target)
        content_regions = content_analyzer.analyze_content_regions(target)

        # Compute metrics for each method
        method_metrics = {}

        for method_name, reconstruction in reconstructions.items():
            # Basic error analyzer
            basic_analyzer = ErrorAnalyzer()
            basic_metrics = basic_analyzer.compute_comprehensive_metrics(target, reconstruction)

            # Advanced metrics
            lpips_score = lpips_calc.compute_lpips(target, reconstruction)
            freq_metrics = freq_analyzer.compute_frequency_metrics(target, reconstruction)
            content_weighted_error = content_analyzer.compute_content_weighted_error(
                target, reconstruction, content_regions)

            # Combine all metrics
            advanced_metrics = AdvancedErrorMetrics(
                lpips_score=lpips_score,
                spectral_distortion=freq_metrics['spectral_distortion'],
                high_freq_preservation=freq_metrics['high_freq_preservation'],
                content_weighted_error=content_weighted_error
            )
            advanced_metrics.frequency_response = {
                k: v for k, v in freq_metrics.items()
                if k.startswith('freq_')
            }

            method_metrics[method_name] = {
                'basic': basic_metrics.to_dict(),
                'advanced': advanced_metrics.to_dict(),
                'combined_score': self._compute_combined_score(basic_metrics, advanced_metrics)
            }

        # Rank methods
        rankings = self._rank_methods(method_metrics)

        # Update rankings in metrics
        for method_name, rank in rankings.items():
            method_metrics[method_name]['advanced']['quality_rank'] = rank

        # Store comparison history
        comparison_record = {
            'methods': list(reconstructions.keys()),
            'rankings': rankings,
            'timestamp': self.methods_compared
        }
        self.comparison_history.append(comparison_record)
        self.methods_compared += 1

        return method_metrics

    def _compute_combined_score(self, basic_metrics: ErrorMetrics,
                              advanced_metrics: AdvancedErrorMetrics) -> float:
        """Compute combined quality score from all metrics."""
        # Weighted combination of different metric types
        score = 0.0

        # Basic metrics (40% weight)
        score += 0.2 * (1.0 - basic_metrics.l1_error)       # Lower error is better
        score += 0.1 * basic_metrics.ssim_score             # Higher SSIM is better
        score += 0.1 * min(basic_metrics.psnr / 40.0, 1.0)  # Normalize PSNR to [0,1]

        # Advanced metrics (60% weight)
        score += 0.3 * (1.0 - advanced_metrics.lpips_score)           # Lower LPIPS is better
        score += 0.1 * advanced_metrics.high_freq_preservation        # Higher preservation is better
        score += 0.2 * (1.0 - advanced_metrics.content_weighted_error) # Lower weighted error is better

        return float(max(0.0, min(1.0, score)))  # Clamp to [0,1] and ensure Python float type

    def _rank_methods(self, method_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Rank methods by combined score."""
        # Extract scores
        scores = [(name, metrics['combined_score']) for name, metrics in method_metrics.items()]

        # Sort by score (higher is better)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Assign ranks
        rankings = {}
        for rank, (method_name, score) in enumerate(scores, 1):
            rankings[method_name] = rank

        return rankings


class AdvancedErrorAnalyzer(ErrorAnalyzer):
    """Extended error analyzer with advanced metrics."""

    def __init__(self, window_size: int = 7, edge_threshold: float = 0.1):
        """Initialize advanced error analyzer."""
        super().__init__(window_size, edge_threshold)

        # Initialize sub-analyzers
        self.lpips_calculator = LPIPSCalculator()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.content_analyzer = ContentAwareAnalyzer()
        self.comparative_assessor = ComparativeQualityAssessment()

    def compute_advanced_metrics(self, target: np.ndarray, rendered: np.ndarray,
                               mask: Optional[np.ndarray] = None) -> AdvancedErrorMetrics:
        """
        Compute comprehensive advanced error metrics.

        Args:
            target: Target image
            rendered: Rendered image
            mask: Optional mask for computation

        Returns:
            Advanced error metrics
        """
        # LPIPS perceptual metric
        lpips_score = self.lpips_calculator.compute_lpips(target, rendered)

        # Frequency-domain metrics
        freq_metrics = self.frequency_analyzer.compute_frequency_metrics(target, rendered)

        # Content-aware metrics
        content_regions = self.content_analyzer.analyze_content_regions(target)
        content_weighted_error = self.content_analyzer.compute_content_weighted_error(
            target, rendered, content_regions)

        # Multi-scale SSIM
        ms_ssim_score = self._compute_multiscale_ssim(target, rendered, mask)

        # Gradient similarity
        gradient_similarity = self._compute_gradient_similarity(target, rendered, mask)

        # Texture similarity
        texture_similarity = self._compute_texture_similarity(target, rendered, mask)

        # Edge coherence
        edge_coherence = self._compute_edge_coherence(target, rendered, mask)

        # Create advanced metrics object
        advanced_metrics = AdvancedErrorMetrics(
            lpips_score=lpips_score,
            ms_ssim_score=ms_ssim_score,
            gradient_similarity=gradient_similarity,
            texture_similarity=texture_similarity,
            edge_coherence=edge_coherence,
            spectral_distortion=freq_metrics['spectral_distortion'],
            high_freq_preservation=freq_metrics['high_freq_preservation'],
            content_weighted_error=content_weighted_error
        )

        # Add frequency response
        advanced_metrics.frequency_response = {
            k: v for k, v in freq_metrics.items()
            if k.startswith('freq_')
        }

        return advanced_metrics

    def create_advanced_error_map(self, target: np.ndarray, rendered: np.ndarray,
                                error_type: str = 'content_weighted') -> np.ndarray:
        """
        Create advanced error map with content-aware weighting.

        Args:
            target: Target image
            rendered: Rendered image
            error_type: Type of advanced error map

        Returns:
            Advanced error map
        """
        if error_type == 'content_weighted':
            # Basic L1 error
            basic_error_map = self.create_error_map(target, rendered, 'l1')

            # Content-aware weighting
            content_regions = self.content_analyzer.analyze_content_regions(target)

            # Create weight map
            H, W = basic_error_map.shape
            weight_map = np.ones((H, W))

            for region in content_regions:
                y1, x1, y2, x2 = region.bbox
                weight_map[y1:y2, x1:x2] *= region.importance_weight

            # Apply weighting
            weighted_error_map = basic_error_map * weight_map

            return weighted_error_map

        elif error_type == 'frequency_weighted':
            # Weight errors by local frequency content
            target_gray = rgb2gray(target) if target.ndim == 3 and target.shape[2] == 3 else target

            # Compute local frequency content
            grad_x = sobel_h(target_gray)
            grad_y = sobel_v(target_gray)
            local_freq = np.sqrt(grad_x**2 + grad_y**2)

            # Normalize to create weights
            if np.max(local_freq) > 0:
                freq_weights = 1.0 + 2.0 * (local_freq / np.max(local_freq))
            else:
                freq_weights = np.ones_like(local_freq)

            # Basic error map
            basic_error_map = self.create_error_map(target, rendered, 'l1')

            # Apply frequency weighting
            freq_weighted_map = basic_error_map * freq_weights

            return freq_weighted_map

        else:
            # Fall back to basic error map
            return self.create_error_map(target, rendered, error_type)

    def _compute_multiscale_ssim(self, target: np.ndarray, rendered: np.ndarray,
                               mask: Optional[np.ndarray] = None) -> float:
        """Compute multi-scale SSIM."""
        scales = [1.0, 0.5, 0.25]  # Different scales
        ssim_scores = []

        for scale in scales:
            if scale < 1.0:
                # Downsample images
                from skimage.transform import rescale
                target_scaled = rescale(target, scale, anti_aliasing=True, channel_axis=-1 if target.ndim == 3 else None)
                rendered_scaled = rescale(rendered, scale, anti_aliasing=True, channel_axis=-1 if rendered.ndim == 3 else None)
                mask_scaled = rescale(mask, scale, anti_aliasing=True) if mask is not None else None
            else:
                target_scaled = target
                rendered_scaled = rendered
                mask_scaled = mask

            # Compute SSIM at this scale
            scale_ssim = self.compute_ssim(target_scaled, rendered_scaled, mask_scaled)
            ssim_scores.append(scale_ssim)

        # Weighted average (give more weight to original scale)
        weights = [0.5, 0.3, 0.2]
        ms_ssim = sum(w * s for w, s in zip(weights, ssim_scores))

        return float(ms_ssim)

    def _compute_gradient_similarity(self, target: np.ndarray, rendered: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> float:
        """Compute gradient-based similarity."""
        # Convert to grayscale
        target_gray = rgb2gray(target) if target.ndim == 3 and target.shape[2] == 3 else target
        rendered_gray = rgb2gray(rendered) if rendered.ndim == 3 and rendered.shape[2] == 3 else rendered

        # Compute gradients
        target_grad_x = sobel_h(target_gray)
        target_grad_y = sobel_v(target_gray)
        target_grad_mag = np.sqrt(target_grad_x**2 + target_grad_y**2)

        rendered_grad_x = sobel_h(rendered_gray)
        rendered_grad_y = sobel_v(rendered_gray)
        rendered_grad_mag = np.sqrt(rendered_grad_x**2 + rendered_grad_y**2)

        # Apply mask if provided
        if mask is not None:
            target_grad_mag = target_grad_mag * mask
            rendered_grad_mag = rendered_grad_mag * mask

        # Compute similarity
        grad_diff = np.abs(target_grad_mag - rendered_grad_mag)
        max_grad = np.maximum(target_grad_mag, rendered_grad_mag)

        # Avoid division by zero and ensure non-negative result
        similarity_map = np.ones_like(grad_diff)
        valid_mask = max_grad > 1e-8
        similarity_map[valid_mask] = 1.0 - (grad_diff[valid_mask] / max_grad[valid_mask])

        # Ensure values are in [0, 1] range
        similarity_map = np.clip(similarity_map, 0.0, 1.0)
        gradient_similarity = np.mean(similarity_map)

        return float(gradient_similarity)

    def _compute_texture_similarity(self, target: np.ndarray, rendered: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> float:
        """Compute texture-based similarity using local binary patterns."""
        from skimage.feature import local_binary_pattern

        # Convert to grayscale
        target_gray = rgb2gray(target) if target.ndim == 3 and target.shape[2] == 3 else target
        rendered_gray = rgb2gray(rendered) if rendered.ndim == 3 and rendered.shape[2] == 3 else rendered

        # Compute LBP
        radius = 3
        n_points = 8 * radius

        try:
            target_lbp = local_binary_pattern(target_gray, n_points, radius, method='uniform')
            rendered_lbp = local_binary_pattern(rendered_gray, n_points, radius, method='uniform')
        except Exception:
            # Fallback to simple texture measure
            return self._compute_simple_texture_similarity(target_gray, rendered_gray, mask)

        # Apply mask if provided
        if mask is not None:
            target_lbp = target_lbp * mask
            rendered_lbp = rendered_lbp * mask

        # Compute histogram similarity
        target_hist, _ = np.histogram(target_lbp, bins=n_points + 2, range=(0, n_points + 2))
        rendered_hist, _ = np.histogram(rendered_lbp, bins=n_points + 2, range=(0, n_points + 2))

        # Normalize histograms
        target_hist = target_hist.astype(float)
        rendered_hist = rendered_hist.astype(float)

        if np.sum(target_hist) > 0:
            target_hist /= np.sum(target_hist)
        if np.sum(rendered_hist) > 0:
            rendered_hist /= np.sum(rendered_hist)

        # Compute chi-squared distance
        chi2_dist = np.sum((target_hist - rendered_hist)**2 / (target_hist + rendered_hist + 1e-8))

        # Convert to similarity (lower distance = higher similarity)
        texture_similarity = np.exp(-chi2_dist)

        return float(texture_similarity)

    def _compute_simple_texture_similarity(self, target: np.ndarray, rendered: np.ndarray,
                                         mask: Optional[np.ndarray] = None) -> float:
        """Simple texture similarity fallback."""
        # Use variance as texture measure
        target_var = ndimage.generic_filter(target, np.var, size=5)
        rendered_var = ndimage.generic_filter(rendered, np.var, size=5)

        if mask is not None:
            target_var = target_var * mask
            rendered_var = rendered_var * mask

        # Compute similarity
        var_diff = np.abs(target_var - rendered_var)
        max_var = np.maximum(target_var, rendered_var)

        similarity_map = np.ones_like(var_diff)
        valid_mask = max_var > 1e-8
        similarity_map[valid_mask] = 1.0 - (var_diff[valid_mask] / max_var[valid_mask])

        return float(np.mean(similarity_map))

    def _compute_edge_coherence(self, target: np.ndarray, rendered: np.ndarray,
                              mask: Optional[np.ndarray] = None) -> float:
        """Compute edge structure coherence."""
        # Convert to grayscale
        target_gray = rgb2gray(target) if target.ndim == 3 and target.shape[2] == 3 else target
        rendered_gray = rgb2gray(rendered) if rendered.ndim == 3 and rendered.shape[2] == 3 else rendered

        # Detect edges
        try:
            from skimage.feature import canny
            target_edges = canny(target_gray, sigma=1.0)
            rendered_edges = canny(rendered_gray, sigma=1.0)
        except Exception:
            # Fallback to gradient-based edges
            target_grad_x = sobel_h(target_gray)
            target_grad_y = sobel_v(target_gray)
            target_grad_mag = np.sqrt(target_grad_x**2 + target_grad_y**2)
            target_edges = target_grad_mag > 0.1

            rendered_grad_x = sobel_h(rendered_gray)
            rendered_grad_y = sobel_v(rendered_gray)
            rendered_grad_mag = np.sqrt(rendered_grad_x**2 + rendered_grad_y**2)
            rendered_edges = rendered_grad_mag > 0.1

        # Apply mask if provided
        if mask is not None:
            target_edges = target_edges * mask
            rendered_edges = rendered_edges * mask

        # Compute edge overlap metrics
        intersection = np.sum(target_edges & rendered_edges)
        union = np.sum(target_edges | rendered_edges)

        if union > 0:
            jaccard_index = intersection / union
        else:
            jaccard_index = 1.0  # No edges in either image

        return float(jaccard_index)


# Convenience functions
def compute_advanced_reconstruction_error(target: np.ndarray, rendered: np.ndarray,
                                        mask: Optional[np.ndarray] = None) -> AdvancedErrorMetrics:
    """Convenience function to compute advanced reconstruction error."""
    analyzer = AdvancedErrorAnalyzer()
    return analyzer.compute_advanced_metrics(target, rendered, mask)


def compare_reconstruction_methods(target: np.ndarray,
                                 reconstructions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """Convenience function to compare multiple reconstruction methods."""
    assessor = ComparativeQualityAssessment()
    return assessor.compare_methods(target, reconstructions)