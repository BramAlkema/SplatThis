"""
Anisotropic Refinement System for Adaptive Gaussian Splatting.

T4.1: Advanced edge-aware anisotropic refinement that enhances splat elongation
and orientation based on local image structure to improve edge representation.

This module provides:
- Edge-aware anisotropy enhancement
- Dynamic aspect ratio optimization
- Orientation fine-tuning for edge alignment
- Anisotropy constraints and validation
- Edge-following quality metrics
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum

from .adaptive_gaussian import AdaptiveGaussian2D
from .gradient_utils import GradientAnalyzer
from .progressive_refinement import RefinementConfig, RefinementResult, RefinementState
from ..utils.math import safe_eigendecomposition, clamp_value, normalize_angle

logger = logging.getLogger(__name__)


class AnisotropyStrategy(Enum):
    """Strategies for anisotropic refinement."""
    EDGE_FOLLOWING = "edge_following"      # Align with edge direction
    STRUCTURE_TENSOR = "structure_tensor"  # Use structure tensor analysis
    GRADIENT_BASED = "gradient_based"      # Use gradient information
    HYBRID = "hybrid"                      # Combine multiple approaches


class AnisotropyOperation(Enum):
    """Types of anisotropic operations."""
    ASPECT_RATIO_ENHANCEMENT = "aspect_ratio_enhancement"
    ORIENTATION_ALIGNMENT = "orientation_alignment"
    EDGE_SHARPENING = "edge_sharpening"
    COHERENCE_OPTIMIZATION = "coherence_optimization"


@dataclass
class AnisotropicConfig:
    """Configuration for anisotropic refinement."""

    # Strategy and operations
    strategy: AnisotropyStrategy = AnisotropyStrategy.HYBRID
    enabled_operations: List[AnisotropyOperation] = field(default_factory=lambda: [
        AnisotropyOperation.ASPECT_RATIO_ENHANCEMENT,
        AnisotropyOperation.ORIENTATION_ALIGNMENT,
        AnisotropyOperation.EDGE_SHARPENING
    ])

    # Gradient analysis
    gradient_method: str = "sobel"
    gradient_sigma: float = 1.0
    structure_tensor_sigma: float = 2.0

    # Anisotropy parameters
    max_aspect_ratio: float = 8.0
    min_aspect_ratio: float = 1.0
    aspect_ratio_step: float = 0.2
    orientation_tolerance: float = np.pi / 12  # 15 degrees

    # Edge detection thresholds
    edge_strength_threshold: float = 0.1
    coherence_threshold: float = 0.3
    gradient_magnitude_threshold: float = 0.05

    # Optimization parameters
    max_refinement_iterations: int = 5
    convergence_threshold: float = 1e-4
    learning_rate: float = 0.1

    # Quality constraints
    preserve_area: bool = True
    maintain_smoothness: bool = True
    overlap_penalty_weight: float = 0.1

    # Validation weights
    edge_alignment_weight: float = 0.4
    coherence_weight: float = 0.3
    aspect_ratio_weight: float = 0.2
    smoothness_weight: float = 0.1

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_aspect_ratio <= self.min_aspect_ratio:
            raise ValueError("max_aspect_ratio must be greater than min_aspect_ratio")
        if self.edge_strength_threshold < 0:
            raise ValueError("edge_strength_threshold must be non-negative")
        if self.coherence_threshold < 0 or self.coherence_threshold > 1:
            raise ValueError("coherence_threshold must be in [0, 1]")


@dataclass
class AnisotropicAnalysis:
    """Results of anisotropic structure analysis."""
    edge_strength: np.ndarray           # Edge strength field
    orientation: np.ndarray             # Edge orientation field
    coherence: np.ndarray              # Structure coherence field
    gradient_magnitude: np.ndarray      # Gradient magnitude field
    structure_tensor: np.ndarray        # Full structure tensor field
    quality_map: np.ndarray            # Quality assessment map


@dataclass
class AnisotropicRefinementResult:
    """Result of anisotropic refinement."""
    refined_splats: List[AdaptiveGaussian2D]
    iterations: int
    converged: bool
    final_quality: float
    quality_improvement: float
    refinement_history: List[Dict[str, Any]]
    anisotropic_analysis: AnisotropicAnalysis


class AnisotropicRefiner:
    """Anisotropic refinement system for edge-aware Gaussian splat optimization."""

    def __init__(self, config: AnisotropicConfig):
        """
        Initialize anisotropic refiner.

        Args:
            config: Anisotropic refinement configuration
        """
        self.config = config
        self.gradient_analyzer = GradientAnalyzer(
            sigma=config.gradient_sigma,
            method=config.gradient_method
        )
        self.refinement_history = []

    def analyze_anisotropic_structure(self, image: np.ndarray) -> AnisotropicAnalysis:
        """
        Analyze image structure for anisotropic refinement.

        Args:
            image: Input image for analysis

        Returns:
            Anisotropic analysis results
        """
        logger.info("Computing anisotropic structure analysis...")

        # Compute gradients and structure tensor
        grad_x, grad_y = self.gradient_analyzer.compute_gradients(image)
        gradient_magnitude = self.gradient_analyzer.compute_gradient_magnitude(image)
        structure_tensor = self.gradient_analyzer.compute_structure_tensor(
            image, smoothing_sigma=self.config.structure_tensor_sigma
        )

        # Analyze local structure
        edge_strength, orientation, coherence = self.gradient_analyzer.analyze_local_structure(
            structure_tensor
        )

        # Compute quality map
        quality_map = self._compute_quality_map(
            edge_strength, coherence, gradient_magnitude
        )

        return AnisotropicAnalysis(
            edge_strength=edge_strength,
            orientation=orientation,
            coherence=coherence,
            gradient_magnitude=gradient_magnitude,
            structure_tensor=structure_tensor,
            quality_map=quality_map
        )

    def _compute_quality_map(self, edge_strength: np.ndarray, coherence: np.ndarray,
                           gradient_magnitude: np.ndarray) -> np.ndarray:
        """Compute quality map for anisotropic refinement guidance."""
        # Combine multiple quality indicators
        edge_quality = np.clip(edge_strength / (self.config.edge_strength_threshold + 1e-8), 0, 1)
        coherence_quality = np.clip(coherence / (self.config.coherence_threshold + 1e-8), 0, 1)
        gradient_quality = np.clip(
            gradient_magnitude / (self.config.gradient_magnitude_threshold + 1e-8), 0, 1
        )

        # Weighted combination
        quality_map = (
            self.config.edge_alignment_weight * edge_quality +
            self.config.coherence_weight * coherence_quality +
            (1.0 - self.config.edge_alignment_weight - self.config.coherence_weight) * gradient_quality
        )

        return np.clip(quality_map, 0, 1)

    def refine_splat_anisotropy(self, splat: AdaptiveGaussian2D,
                              analysis: AnisotropicAnalysis,
                              image_shape: Tuple[int, int]) -> AdaptiveGaussian2D:
        """
        Refine a single splat's anisotropic properties.

        Args:
            splat: Gaussian splat to refine
            analysis: Anisotropic structure analysis
            image_shape: Image dimensions (height, width)

        Returns:
            Refined splat with updated anisotropic properties
        """
        # Convert normalized coordinates to pixel coordinates
        pixel_x = int(splat.mu[0] * (image_shape[1] - 1))
        pixel_y = int(splat.mu[1] * (image_shape[0] - 1))

        # Clamp to image bounds
        pixel_x = np.clip(pixel_x, 0, image_shape[1] - 1)
        pixel_y = np.clip(pixel_y, 0, image_shape[0] - 1)

        # Extract local structure information
        local_edge_strength = analysis.edge_strength[pixel_y, pixel_x]
        local_orientation = analysis.orientation[pixel_y, pixel_x]
        local_coherence = analysis.coherence[pixel_y, pixel_x]
        local_quality = analysis.quality_map[pixel_y, pixel_x]

        # Skip refinement if quality is too low
        if local_quality < self.config.gradient_magnitude_threshold:
            return splat

        # Create refined splat
        refined_splat = AdaptiveGaussian2D(
            mu=splat.mu.copy(),
            inv_s=splat.inv_s.copy(),
            theta=splat.theta,
            color=splat.color.copy(),
            alpha=splat.alpha,
            content_complexity=splat.content_complexity,
            saliency_score=splat.saliency_score,
            refinement_count=splat.refinement_count + 1
        )

        # Apply anisotropic operations
        if AnisotropyOperation.ASPECT_RATIO_ENHANCEMENT in self.config.enabled_operations:
            refined_splat = self._enhance_aspect_ratio(
                refined_splat, local_edge_strength, local_coherence
            )

        if AnisotropyOperation.ORIENTATION_ALIGNMENT in self.config.enabled_operations:
            refined_splat = self._align_orientation(
                refined_splat, local_orientation, local_coherence
            )

        if AnisotropyOperation.EDGE_SHARPENING in self.config.enabled_operations:
            refined_splat = self._sharpen_edges(
                refined_splat, local_edge_strength, local_quality
            )

        # Validate and constrain the refined splat
        refined_splat = self._apply_anisotropic_constraints(refined_splat)

        return refined_splat

    def _enhance_aspect_ratio(self, splat: AdaptiveGaussian2D,
                            edge_strength: float, coherence: float) -> AdaptiveGaussian2D:
        """Enhance aspect ratio based on edge strength and coherence."""
        if edge_strength < self.config.edge_strength_threshold:
            return splat

        # Compute current aspect ratio
        current_scales = 1.0 / splat.inv_s
        current_aspect_ratio = max(current_scales) / min(current_scales)

        # Determine target aspect ratio based on edge strength and coherence
        enhancement_factor = min(coherence * edge_strength, 1.0)
        target_aspect_ratio = 1.0 + enhancement_factor * (self.config.max_aspect_ratio - 1.0)
        target_aspect_ratio = clamp_value(
            target_aspect_ratio, self.config.min_aspect_ratio, self.config.max_aspect_ratio
        )

        # Apply gradual aspect ratio change
        if target_aspect_ratio > current_aspect_ratio:
            new_aspect_ratio = current_aspect_ratio + self.config.aspect_ratio_step
            new_aspect_ratio = min(new_aspect_ratio, target_aspect_ratio)

            # Adjust scales while preserving area if configured
            if self.config.preserve_area:
                # Keep geometric mean constant
                geo_mean = np.sqrt(current_scales[0] * current_scales[1])
                new_major_scale = geo_mean * np.sqrt(new_aspect_ratio)
                new_minor_scale = geo_mean / np.sqrt(new_aspect_ratio)
            else:
                # Stretch major axis, keep minor axis
                major_idx = np.argmax(current_scales)
                new_scales = current_scales.copy()
                new_scales[major_idx] = new_scales[major_idx] * (new_aspect_ratio / current_aspect_ratio)
                new_major_scale = new_scales[major_idx]
                new_minor_scale = new_scales[1 - major_idx]

            # Update inverse scales
            splat.inv_s[0] = 1.0 / new_major_scale
            splat.inv_s[1] = 1.0 / new_minor_scale

            # Sort to maintain consistent ordering (larger scale first)
            if new_minor_scale > new_major_scale:
                splat.inv_s[0], splat.inv_s[1] = splat.inv_s[1], splat.inv_s[0]

        return splat

    def _align_orientation(self, splat: AdaptiveGaussian2D,
                         edge_orientation: float, coherence: float) -> AdaptiveGaussian2D:
        """Align splat orientation with local edge direction."""
        if coherence < self.config.coherence_threshold:
            return splat

        # Compute orientation difference
        current_orientation = splat.theta
        orientation_diff = normalize_angle(edge_orientation - current_orientation)

        # Apply gradual orientation adjustment
        if abs(orientation_diff) > self.config.orientation_tolerance:
            adjustment_strength = coherence * self.config.learning_rate
            orientation_adjustment = adjustment_strength * orientation_diff

            # Apply the adjustment
            new_orientation = current_orientation + orientation_adjustment
            splat.theta = normalize_angle(new_orientation)

        return splat

    def _sharpen_edges(self, splat: AdaptiveGaussian2D,
                      edge_strength: float, quality: float) -> AdaptiveGaussian2D:
        """Sharpen edges by reducing scale in direction perpendicular to edge."""
        if edge_strength < self.config.edge_strength_threshold:
            return splat

        # Reduce minor axis scale to sharpen edges
        sharpening_factor = 1.0 - (edge_strength * quality * self.config.learning_rate)
        sharpening_factor = clamp_value(sharpening_factor, 0.5, 1.0)

        # Apply sharpening to minor axis (larger inv_s value)
        minor_axis_idx = np.argmax(splat.inv_s)
        splat.inv_s[minor_axis_idx] *= (1.0 / sharpening_factor)

        return splat

    def _apply_anisotropic_constraints(self, splat: AdaptiveGaussian2D) -> AdaptiveGaussian2D:
        """Apply constraints to ensure valid anisotropic parameters."""
        # Constrain inverse scales to reasonable ranges
        min_inv_scale = 1.0 / 10.0  # Max scale of 10 pixels
        max_inv_scale = 1.0 / 0.01  # Min scale of 0.01 pixels

        splat.inv_s = np.clip(splat.inv_s, min_inv_scale, max_inv_scale)

        # Constrain aspect ratio
        current_scales = 1.0 / splat.inv_s
        current_aspect_ratio = max(current_scales) / min(current_scales)

        if current_aspect_ratio > self.config.max_aspect_ratio:
            # Scale down the major axis
            major_idx = np.argmax(current_scales)
            target_major_scale = current_scales[1 - major_idx] * self.config.max_aspect_ratio
            splat.inv_s[major_idx] = 1.0 / target_major_scale

        # Constrain other parameters
        splat.alpha = clamp_value(splat.alpha, 0.0, 1.0)
        splat.color = np.clip(splat.color, 0.0, 1.0)
        splat.theta = normalize_angle(splat.theta)

        return splat

    def refine_splats_anisotropically(self, splats: List[AdaptiveGaussian2D],
                                    target_image: np.ndarray) -> AnisotropicRefinementResult:
        """
        Perform anisotropic refinement on a collection of splats.

        Args:
            splats: List of Gaussian splats to refine
            target_image: Target image for structure analysis

        Returns:
            Anisotropic refinement result
        """
        logger.info(f"Starting anisotropic refinement with {len(splats)} splats")

        # Analyze image structure
        analysis = self.analyze_anisotropic_structure(target_image)

        # Initialize refinement
        refined_splats = [splat.__class__(**splat.__dict__) for splat in splats]  # Deep copy
        initial_quality = self._compute_overall_quality(refined_splats, analysis)

        refinement_history = []
        image_shape = target_image.shape[:2]

        # Iterative refinement
        for iteration in range(self.config.max_refinement_iterations):
            logger.info(f"Anisotropic refinement iteration {iteration + 1}/{self.config.max_refinement_iterations}")

            iteration_improved = False

            # Refine each splat
            for i, splat in enumerate(refined_splats):
                original_splat = AdaptiveGaussian2D(**splat.__dict__)

                # Apply anisotropic refinement
                refined_splat = self.refine_splat_anisotropy(splat, analysis, image_shape)

                # Check if refinement improved quality
                if self._validate_refinement(original_splat, refined_splat, analysis):
                    refined_splats[i] = refined_splat
                    iteration_improved = True
                else:
                    # Revert to original if refinement didn't help
                    refined_splats[i] = original_splat

            # Compute quality and check convergence
            current_quality = self._compute_overall_quality(refined_splats, analysis)
            quality_improvement = current_quality - initial_quality

            refinement_history.append({
                'iteration': iteration + 1,
                'quality': current_quality,
                'improvement': quality_improvement,
                'num_refined': sum(1 for s in refined_splats if s.refinement_count > 0)
            })

            logger.info(f"Iteration {iteration + 1}: Quality = {current_quality:.6f}, "
                       f"Improvement = {quality_improvement:.6f}")

            # Check convergence
            if iteration > 0:
                prev_quality = refinement_history[-2]['quality']
                if abs(current_quality - prev_quality) < self.config.convergence_threshold:
                    logger.info(f"Converged at iteration {iteration + 1}")
                    break

            if not iteration_improved:
                logger.info(f"No improvements in iteration {iteration + 1}, stopping")
                break

        final_quality = self._compute_overall_quality(refined_splats, analysis)

        logger.info(f"Anisotropic refinement completed: {iteration + 1} iterations")
        logger.info(f"Final quality: {final_quality:.6f}")
        logger.info(f"Quality improvement: {final_quality - initial_quality:.6f}")

        return AnisotropicRefinementResult(
            refined_splats=refined_splats,
            iterations=iteration + 1,
            converged=(iteration + 1 < self.config.max_refinement_iterations),
            final_quality=final_quality,
            quality_improvement=final_quality - initial_quality,
            refinement_history=refinement_history,
            anisotropic_analysis=analysis
        )

    def _compute_overall_quality(self, splats: List[AdaptiveGaussian2D],
                               analysis: AnisotropicAnalysis) -> float:
        """Compute overall quality metric for splat configuration."""
        if not splats:
            return 0.0

        total_quality = 0.0
        image_shape = analysis.edge_strength.shape

        for splat in splats:
            # Convert to pixel coordinates
            pixel_x = int(splat.mu[0] * (image_shape[1] - 1))
            pixel_y = int(splat.mu[1] * (image_shape[0] - 1))
            pixel_x = np.clip(pixel_x, 0, image_shape[1] - 1)
            pixel_y = np.clip(pixel_y, 0, image_shape[0] - 1)

            # Extract local quality
            local_quality = analysis.quality_map[pixel_y, pixel_x]

            # Weight by splat properties
            aspect_ratio_quality = self._compute_aspect_ratio_quality(splat, analysis, (pixel_y, pixel_x))
            orientation_quality = self._compute_orientation_quality(splat, analysis, (pixel_y, pixel_x))

            splat_quality = (
                self.config.edge_alignment_weight * local_quality +
                self.config.aspect_ratio_weight * aspect_ratio_quality +
                self.config.coherence_weight * orientation_quality
            )

            total_quality += splat_quality

        return total_quality / len(splats)

    def _compute_aspect_ratio_quality(self, splat: AdaptiveGaussian2D,
                                    analysis: AnisotropicAnalysis,
                                    pixel_pos: Tuple[int, int]) -> float:
        """Compute aspect ratio quality for a splat."""
        pixel_y, pixel_x = pixel_pos

        current_scales = 1.0 / splat.inv_s
        current_aspect_ratio = max(current_scales) / min(current_scales)

        # Get local edge properties
        edge_strength = analysis.edge_strength[pixel_y, pixel_x]
        coherence = analysis.coherence[pixel_y, pixel_x]

        # Ideal aspect ratio based on local structure
        if edge_strength > self.config.edge_strength_threshold and coherence > self.config.coherence_threshold:
            ideal_aspect_ratio = 1.0 + coherence * edge_strength * (self.config.max_aspect_ratio - 1.0)
        else:
            ideal_aspect_ratio = 1.0

        # Quality decreases with distance from ideal
        aspect_ratio_error = abs(current_aspect_ratio - ideal_aspect_ratio)
        quality = np.exp(-aspect_ratio_error)

        return quality

    def _compute_orientation_quality(self, splat: AdaptiveGaussian2D,
                                   analysis: AnisotropicAnalysis,
                                   pixel_pos: Tuple[int, int]) -> float:
        """Compute orientation quality for a splat."""
        pixel_y, pixel_x = pixel_pos

        local_orientation = analysis.orientation[pixel_y, pixel_x]
        coherence = analysis.coherence[pixel_y, pixel_x]

        if coherence < self.config.coherence_threshold:
            return 1.0  # No strong orientation preference

        # Compute orientation alignment
        orientation_diff = abs(normalize_angle(splat.theta - local_orientation))
        orientation_diff = min(orientation_diff, np.pi - orientation_diff)  # Consider symmetry

        # Quality decreases with orientation misalignment
        max_diff = np.pi / 2
        quality = 1.0 - (orientation_diff / max_diff)

        return max(0.0, quality)

    def _validate_refinement(self, original: AdaptiveGaussian2D,
                           refined: AdaptiveGaussian2D,
                           analysis: AnisotropicAnalysis) -> bool:
        """Validate that refinement improved the splat."""
        image_shape = analysis.edge_strength.shape

        # Convert to pixel coordinates
        pixel_x = int(original.mu[0] * (image_shape[1] - 1))
        pixel_y = int(original.mu[1] * (image_shape[0] - 1))
        pixel_x = np.clip(pixel_x, 0, image_shape[1] - 1)
        pixel_y = np.clip(pixel_y, 0, image_shape[0] - 1)

        # Compute quality for both splats
        original_quality = (
            self._compute_aspect_ratio_quality(original, analysis, (pixel_y, pixel_x)) *
            self._compute_orientation_quality(original, analysis, (pixel_y, pixel_x))
        )

        refined_quality = (
            self._compute_aspect_ratio_quality(refined, analysis, (pixel_y, pixel_x)) *
            self._compute_orientation_quality(refined, analysis, (pixel_y, pixel_x))
        )

        return refined_quality > original_quality


# Convenience functions for easy usage

def create_anisotropic_config_preset(preset: str = "balanced") -> AnisotropicConfig:
    """Create predefined anisotropic refinement configurations."""
    if preset == "conservative":
        return AnisotropicConfig(
            strategy=AnisotropyStrategy.EDGE_FOLLOWING,
            max_aspect_ratio=3.0,
            learning_rate=0.05,
            max_refinement_iterations=3,
            edge_strength_threshold=0.2,
            coherence_threshold=0.5
        )
    elif preset == "balanced":
        return AnisotropicConfig(
            strategy=AnisotropyStrategy.HYBRID,
            max_aspect_ratio=6.0,
            learning_rate=0.1,
            max_refinement_iterations=5,
            edge_strength_threshold=0.1,
            coherence_threshold=0.3
        )
    elif preset == "aggressive":
        return AnisotropicConfig(
            strategy=AnisotropyStrategy.HYBRID,
            max_aspect_ratio=10.0,
            learning_rate=0.2,
            max_refinement_iterations=8,
            edge_strength_threshold=0.05,
            coherence_threshold=0.2
        )
    elif preset == "experimental":
        return AnisotropicConfig(
            strategy=AnisotropyStrategy.STRUCTURE_TENSOR,
            enabled_operations=[
                AnisotropyOperation.ASPECT_RATIO_ENHANCEMENT,
                AnisotropyOperation.ORIENTATION_ALIGNMENT,
                AnisotropyOperation.EDGE_SHARPENING,
                AnisotropyOperation.COHERENCE_OPTIMIZATION
            ],
            max_aspect_ratio=15.0,
            learning_rate=0.3,
            max_refinement_iterations=10,
            edge_strength_threshold=0.02,
            coherence_threshold=0.1
        )
    else:
        raise ValueError(f"Unknown preset: {preset}. Available: conservative, balanced, aggressive, experimental")


def refine_splats_anisotropically(splats: List[AdaptiveGaussian2D],
                                target_image: np.ndarray,
                                config: Optional[AnisotropicConfig] = None) -> AnisotropicRefinementResult:
    """
    Convenience function for anisotropic splat refinement.

    Args:
        splats: List of Gaussian splats to refine
        target_image: Target image for structure analysis
        config: Optional configuration (defaults to balanced preset)

    Returns:
        Anisotropic refinement result
    """
    if config is None:
        config = create_anisotropic_config_preset("balanced")

    refiner = AnisotropicRefiner(config)
    return refiner.refine_splats_anisotropically(splats, target_image)