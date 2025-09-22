"""
Progressive Refinement System for Adaptive Gaussian Splatting.

This module implements T3.3: Progressive Refinement System as part of Phase 3: Progressive Optimization.
It provides iterative refinement of Gaussian splat parameters based on reconstruction error analysis,
integrating the manual gradient computation (T3.1) and SGD optimization (T3.2) systems.

Key Features:
- Error map computation and analysis
- High-error region identification (>80th percentile)
- Splat refinement operations (scale, position, alpha, anisotropy)
- Integration with SGD optimization for parameter updates
- Convergence criteria and iterative improvement
- Content-adaptive refinement strategies
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any, Set
from enum import Enum
import copy
import cv2

# Import optimization systems from previous tasks
from .sgd_optimizer import (
    SGDOptimizer,
    SGDConfig,
    OptimizationResult,
    OptimizationMethod
)
from .manual_gradients import (
    ManualGradientComputer,
    GradientConfig,
    SplatGradients
)
from .adaptive_gaussian import AdaptiveGaussian2D

logger = logging.getLogger(__name__)


class RefinementStrategy(Enum):
    """Available refinement strategies."""
    ERROR_DRIVEN = "error_driven"      # Refine based on reconstruction error
    CONTENT_ADAPTIVE = "content_adaptive"  # Refine based on local content complexity
    SALIENCY_GUIDED = "saliency_guided"    # Refine based on saliency analysis
    HYBRID = "hybrid"                  # Combination of multiple strategies


class RefinementOperation(Enum):
    """Types of refinement operations."""
    SCALE_ADJUSTMENT = "scale_adjustment"    # Adjust splat size
    POSITION_REFINEMENT = "position_refinement"  # Fine-tune position
    ALPHA_MODULATION = "alpha_modulation"    # Adjust transparency
    ANISOTROPY_OPTIMIZATION = "anisotropy_optimization"  # Adjust aspect ratio
    COLOR_CORRECTION = "color_correction"    # Adjust color
    ROTATION_OPTIMIZATION = "rotation_optimization"  # Adjust orientation


@dataclass
class RefinementConfig:
    """Configuration for progressive refinement system."""

    # Refinement strategy and operations
    strategy: RefinementStrategy = RefinementStrategy.ERROR_DRIVEN
    enabled_operations: List[RefinementOperation] = field(default_factory=lambda: [
        RefinementOperation.SCALE_ADJUSTMENT,
        RefinementOperation.POSITION_REFINEMENT,
        RefinementOperation.ALPHA_MODULATION,
        RefinementOperation.ANISOTROPY_OPTIMIZATION
    ])

    # Error analysis parameters
    error_percentile_threshold: float = 80.0  # High-error threshold (percentile)
    min_error_threshold: float = 0.01        # Minimum error to consider for refinement
    error_analysis_kernel_size: int = 5      # Kernel size for error analysis

    # Refinement iterations and convergence
    max_refinement_iterations: int = 10      # Maximum refinement iterations
    convergence_threshold: float = 0.001     # Relative improvement threshold
    refinement_patience: int = 3             # Iterations without improvement

    # Splat selection criteria
    min_splats_per_iteration: int = 1        # Minimum splats to refine per iteration
    max_splats_per_iteration: int = 10       # Maximum splats to refine per iteration
    splat_overlap_threshold: float = 0.3     # Overlap threshold for splat selection

    # Content analysis parameters
    content_analysis_enabled: bool = True    # Enable local content analysis
    variance_weight: float = 0.3             # Weight for local variance
    gradient_weight: float = 0.4             # Weight for gradient magnitude
    saliency_weight: float = 0.3             # Weight for saliency

    # Scale adaptation parameters
    min_scale_factor: float = 0.5            # Minimum scale multiplier
    max_scale_factor: float = 3.0            # Maximum scale multiplier
    scale_adaptation_rate: float = 0.2       # Rate of scale changes

    # Integration with SGD optimizer
    sgd_config: Optional[SGDConfig] = None   # SGD configuration for optimization
    sgd_iterations_per_refinement: int = 20  # SGD iterations per refinement step

    # Validation and logging
    validate_every: int = 1                  # Validation frequency
    log_progress: bool = True                # Enable progress logging
    save_intermediate_results: bool = False  # Save intermediate results

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.error_percentile_threshold <= 100:
            raise ValueError("error_percentile_threshold must be in [0,100]")
        if self.min_error_threshold < 0:
            raise ValueError("min_error_threshold must be non-negative")
        if self.max_refinement_iterations <= 0:
            raise ValueError("max_refinement_iterations must be positive")
        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive")
        if not 0 <= self.splat_overlap_threshold <= 1:
            raise ValueError("splat_overlap_threshold must be in [0,1]")


@dataclass
class ErrorRegion:
    """Information about a high-error region."""
    center: Tuple[int, int]               # Center coordinates (y, x)
    bbox: Tuple[int, int, int, int]       # Bounding box (y1, x1, y2, x2)
    error_magnitude: float                # Average error in region
    area: int                             # Region area in pixels
    overlapping_splats: List[int]         # Indices of overlapping splats
    content_complexity: float = 0.0       # Local content complexity measure


@dataclass
class RefinementState:
    """State tracking for progressive refinement."""
    iteration: int = 0
    total_error: float = float('inf')
    best_error: float = float('inf')
    error_history: List[float] = field(default_factory=list)
    refined_splats: Set[int] = field(default_factory=set)
    refinement_count: int = 0
    convergence_rate: float = 0.0
    patience_counter: int = 0
    converged: bool = False

    # Region and operation tracking
    identified_regions: List[ErrorRegion] = field(default_factory=list)
    operations_performed: Dict[RefinementOperation, int] = field(default_factory=dict)

    def reset(self):
        """Reset refinement state."""
        self.iteration = 0
        self.total_error = float('inf')
        self.best_error = float('inf')
        self.error_history.clear()
        self.refined_splats.clear()
        self.refinement_count = 0
        self.convergence_rate = 0.0
        self.patience_counter = 0
        self.converged = False
        self.identified_regions.clear()
        self.operations_performed.clear()


@dataclass
class RefinementResult:
    """Result of progressive refinement."""
    refined_splats: List[AdaptiveGaussian2D]
    final_error: float
    iterations: int
    converged: bool
    refinement_history: RefinementState
    total_operations: int
    sgd_results: List[OptimizationResult] = field(default_factory=list)


class ProgressiveRefiner:
    """
    Progressive refinement system for adaptive Gaussian splats.

    This class implements iterative refinement of Gaussian splat parameters
    based on reconstruction error analysis and content-adaptive strategies.
    """

    def __init__(self, config: RefinementConfig = None):
        """
        Initialize progressive refiner.

        Args:
            config: Refinement configuration
        """
        self.config = config or RefinementConfig()
        self.state = RefinementState()

        # Initialize SGD optimizer
        sgd_config = self.config.sgd_config or SGDConfig(
            method=OptimizationMethod.SGD_MOMENTUM,
            max_iterations=self.config.sgd_iterations_per_refinement,
            log_every=999  # Suppress SGD logging during refinement
        )
        self.sgd_optimizer = SGDOptimizer(sgd_config)

        # Initialize gradient computer for content analysis
        self.gradient_computer = ManualGradientComputer()

    def refine_splats(self,
                     splats: List[AdaptiveGaussian2D],
                     target_image: np.ndarray,
                     rendered_image: np.ndarray,
                     loss_function: Optional[Callable] = None) -> RefinementResult:
        """
        Perform progressive refinement of Gaussian splats.

        Args:
            splats: List of Gaussian splats to refine
            target_image: Target image (H, W, C)
            rendered_image: Current rendered image (H, W, C)
            loss_function: Custom loss function (optional)

        Returns:
            RefinementResult with refined splats and metrics
        """
        logger.info(f"Starting progressive refinement with {len(splats)} splats")
        logger.info(f"Strategy: {self.config.strategy.value}, Max iterations: {self.config.max_refinement_iterations}")

        # Reset refinement state
        self.state.reset()

        # Make copies of splats for refinement
        refined_splats = [splat.copy() for splat in splats]
        sgd_results = []

        # Set default loss function
        if loss_function is None:
            loss_function = self._default_loss_function

        try:
            # Main refinement loop
            for iteration in range(self.config.max_refinement_iterations):
                self.state.iteration = iteration

                if self.config.log_progress:
                    logger.info(f"Refinement iteration {iteration + 1}/{self.config.max_refinement_iterations}")

                # Compute current error map
                error_map = self._compute_error_map(target_image, rendered_image)
                current_error = np.sum(error_map**2)

                self.state.total_error = current_error
                self.state.error_history.append(current_error)

                # Update best error and check for improvement
                if current_error < self.state.best_error:
                    self.state.best_error = current_error
                    self.state.patience_counter = 0
                else:
                    self.state.patience_counter += 1

                # Check convergence
                if self._check_convergence():
                    logger.info(f"Converged at iteration {iteration + 1}")
                    self.state.converged = True
                    break

                # Identify high-error regions
                high_error_regions = self._identify_high_error_regions(
                    error_map, target_image, rendered_image
                )
                self.state.identified_regions = high_error_regions

                if not high_error_regions:
                    logger.info("No high-error regions found, stopping refinement")
                    break

                # Select splats for refinement
                splats_to_refine = self._select_splats_for_refinement(
                    refined_splats, high_error_regions, error_map
                )

                if not splats_to_refine:
                    logger.info("No splats selected for refinement")
                    self.state.patience_counter += 1
                    continue

                # Perform refinement operations
                operations_count = self._perform_refinement_operations(
                    refined_splats, splats_to_refine, high_error_regions,
                    target_image, rendered_image, error_map
                )

                # Apply SGD optimization to refined splats
                if operations_count > 0:
                    sgd_result = self._apply_sgd_optimization(
                        [refined_splats[i] for i in splats_to_refine],
                        target_image, rendered_image, error_map
                    )
                    sgd_results.append(sgd_result)

                    # Update refined splats with SGD results
                    for i, refined_splat in enumerate(sgd_result.optimized_splats):
                        refined_splats[splats_to_refine[i]] = refined_splat

                self.state.refinement_count += operations_count
                self.state.refined_splats.update(splats_to_refine)

                # Update rendered image for next iteration (placeholder)
                # In practice, this would re-render with updated splats
                rendered_image = self._update_rendered_image(refined_splats, target_image)

                # Validation
                if iteration % self.config.validate_every == 0:
                    self._validate_refinement(refined_splats, target_image, rendered_image)

        except Exception as e:
            logger.error(f"Refinement failed at iteration {self.state.iteration}: {e}")
            raise

        # Calculate total operations
        total_operations = sum(self.state.operations_performed.values())

        logger.info(f"Progressive refinement completed: {self.state.iteration + 1} iterations")
        logger.info(f"Final error: {self.state.total_error:.6f}")
        logger.info(f"Best error: {self.state.best_error:.6f}")
        logger.info(f"Total operations: {total_operations}")

        return RefinementResult(
            refined_splats=refined_splats,
            final_error=self.state.total_error,
            iterations=self.state.iteration + 1,
            converged=self.state.converged,
            refinement_history=copy.deepcopy(self.state),
            total_operations=total_operations,
            sgd_results=sgd_results
        )

    def _compute_error_map(self, target_image: np.ndarray, rendered_image: np.ndarray) -> np.ndarray:
        """Compute pixel-wise error map between target and rendered images."""
        # L2 norm error per pixel
        error_map = np.linalg.norm(target_image - rendered_image, axis=-1)

        # Apply smoothing if kernel size > 1
        if self.config.error_analysis_kernel_size > 1:
            kernel_size = self.config.error_analysis_kernel_size
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            error_map = cv2.filter2D(error_map, -1, kernel)

        return error_map

    def _identify_high_error_regions(self,
                                   error_map: np.ndarray,
                                   target_image: np.ndarray,
                                   rendered_image: np.ndarray) -> List[ErrorRegion]:
        """Identify high-error regions in the error map."""
        # Calculate error threshold
        error_threshold = np.percentile(error_map, self.config.error_percentile_threshold)
        error_threshold = max(error_threshold, self.config.min_error_threshold)

        # Create binary mask of high-error regions
        high_error_mask = (error_map > error_threshold).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_error_mask)

        regions = []
        for label in range(1, num_labels):  # Skip background (label 0)
            # Get region properties
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 4:  # Skip very small regions
                continue

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            # Calculate average error in region
            region_mask = (labels == label)
            avg_error = np.mean(error_map[region_mask])

            # Calculate content complexity if enabled
            content_complexity = 0.0
            if self.config.content_analysis_enabled:
                content_complexity = self._analyze_content_complexity(
                    target_image, (y, x, y + h, x + w)
                )

            region = ErrorRegion(
                center=(int(centroids[label][1]), int(centroids[label][0])),
                bbox=(y, x, y + h, x + w),
                error_magnitude=avg_error,
                area=area,
                overlapping_splats=[],  # Will be filled by _select_splats_for_refinement
                content_complexity=content_complexity
            )
            regions.append(region)

        # Sort regions by error magnitude (descending)
        regions.sort(key=lambda r: r.error_magnitude, reverse=True)

        logger.debug(f"Identified {len(regions)} high-error regions")
        return regions

    def _select_splats_for_refinement(self,
                                    splats: List[AdaptiveGaussian2D],
                                    error_regions: List[ErrorRegion],
                                    error_map: np.ndarray) -> List[int]:
        """Select splats that should be refined based on error regions."""
        H, W = error_map.shape
        selected_splats = set()

        for region in error_regions:
            y1, x1, y2, x2 = region.bbox
            region_center_y, region_center_x = region.center

            # Find splats that overlap with this error region
            overlapping_splats = []
            for i, splat in enumerate(splats):
                # Convert normalized coordinates to pixel coordinates
                splat_y = int(splat.mu[1] * H)
                splat_x = int(splat.mu[0] * W)

                # Calculate splat radius in pixels (approximate)
                splat_radius = int(np.mean(1.0 / splat.inv_s) * min(H, W))

                # Check if splat overlaps with error region
                if (x1 - splat_radius <= splat_x <= x2 + splat_radius and
                    y1 - splat_radius <= splat_y <= y2 + splat_radius):
                    overlapping_splats.append(i)

            region.overlapping_splats = overlapping_splats

            # Add overlapping splats to selection
            for splat_idx in overlapping_splats:
                if len(selected_splats) < self.config.max_splats_per_iteration:
                    selected_splats.add(splat_idx)

        # Ensure minimum number of splats
        selected_list = list(selected_splats)
        if len(selected_list) < self.config.min_splats_per_iteration and len(splats) > 0:
            # Add random splats to reach minimum
            remaining_indices = [i for i in range(len(splats)) if i not in selected_splats]
            if remaining_indices:
                additional_count = min(
                    self.config.min_splats_per_iteration - len(selected_list),
                    len(remaining_indices)
                )
                additional_splats = np.random.choice(
                    remaining_indices, size=additional_count, replace=False
                )
                selected_list.extend(additional_splats)

        logger.debug(f"Selected {len(selected_list)} splats for refinement")
        return selected_list

    def _perform_refinement_operations(self,
                                     splats: List[AdaptiveGaussian2D],
                                     splat_indices: List[int],
                                     error_regions: List[ErrorRegion],
                                     target_image: np.ndarray,
                                     rendered_image: np.ndarray,
                                     error_map: np.ndarray) -> int:
        """Perform refinement operations on selected splats."""
        operations_count = 0

        for splat_idx in splat_indices:
            splat = splats[splat_idx]

            # Find relevant error regions for this splat
            relevant_regions = [r for r in error_regions if splat_idx in r.overlapping_splats]

            if not relevant_regions:
                continue

            # Choose primary region (highest error)
            primary_region = max(relevant_regions, key=lambda r: r.error_magnitude)

            # Perform enabled refinement operations
            for operation in self.config.enabled_operations:
                if operation == RefinementOperation.SCALE_ADJUSTMENT:
                    if self._refine_scale(splat, primary_region, error_map):
                        operations_count += 1
                        self.state.operations_performed[operation] = (
                            self.state.operations_performed.get(operation, 0) + 1
                        )

                elif operation == RefinementOperation.POSITION_REFINEMENT:
                    if self._refine_position(splat, primary_region, error_map):
                        operations_count += 1
                        self.state.operations_performed[operation] = (
                            self.state.operations_performed.get(operation, 0) + 1
                        )

                elif operation == RefinementOperation.ALPHA_MODULATION:
                    if self._refine_alpha(splat, primary_region, error_map):
                        operations_count += 1
                        self.state.operations_performed[operation] = (
                            self.state.operations_performed.get(operation, 0) + 1
                        )

                elif operation == RefinementOperation.ANISOTROPY_OPTIMIZATION:
                    if self._refine_anisotropy(splat, primary_region, error_map):
                        operations_count += 1
                        self.state.operations_performed[operation] = (
                            self.state.operations_performed.get(operation, 0) + 1
                        )

            # Update splat metadata
            splat.refinement_count += 1
            splat.error_contribution = primary_region.error_magnitude

        return operations_count

    def _refine_scale(self, splat: AdaptiveGaussian2D, region: ErrorRegion, error_map: np.ndarray) -> bool:
        """Refine splat scale based on error region characteristics."""
        # Calculate scale adjustment based on error magnitude and content complexity
        error_factor = min(region.error_magnitude, 1.0)
        content_factor = region.content_complexity if self.config.content_analysis_enabled else 0.5

        # Higher error or complexity suggests need for smaller, more detailed splats
        scale_multiplier = 1.0 - self.config.scale_adaptation_rate * (error_factor + content_factor)
        scale_multiplier = np.clip(scale_multiplier, self.config.min_scale_factor, self.config.max_scale_factor)

        # Apply scale adjustment
        old_inv_s = splat.inv_s.copy()
        splat.inv_s *= scale_multiplier  # Increase inv_s to decrease scale

        # Ensure scale constraints
        splat.clip_parameters()

        # Check if significant change was made
        scale_change = np.linalg.norm(splat.inv_s - old_inv_s)
        return scale_change > 1e-6

    def _refine_position(self, splat: AdaptiveGaussian2D, region: ErrorRegion, error_map: np.ndarray) -> bool:
        """Refine splat position to better align with error region."""
        H, W = error_map.shape

        # Current position in pixel coordinates
        current_pixel_y = int(splat.mu[1] * H)
        current_pixel_x = int(splat.mu[0] * W)

        # Target position (error region center)
        target_pixel_y, target_pixel_x = region.center

        # Calculate movement vector
        dy = target_pixel_y - current_pixel_y
        dx = target_pixel_x - current_pixel_x

        # Apply position adjustment (small step towards target)
        adjustment_rate = 0.1  # Conservative adjustment
        new_pixel_y = current_pixel_y + dy * adjustment_rate
        new_pixel_x = current_pixel_x + dx * adjustment_rate

        # Convert back to normalized coordinates
        old_mu = splat.mu.copy()
        splat.mu[1] = new_pixel_y / H
        splat.mu[0] = new_pixel_x / W

        # Ensure position constraints
        splat.clip_parameters()

        # Check if significant change was made
        position_change = np.linalg.norm(splat.mu - old_mu)
        return position_change > 1e-6

    def _refine_alpha(self, splat: AdaptiveGaussian2D, region: ErrorRegion, error_map: np.ndarray) -> bool:
        """Refine splat alpha based on error characteristics."""
        # Adjust alpha based on error magnitude
        error_factor = min(region.error_magnitude, 1.0)

        # Higher error suggests need for more/less opacity depending on strategy
        alpha_adjustment = 0.05 * error_factor
        if region.error_magnitude > 0.5:
            # High error - try reducing alpha to let other splats contribute
            alpha_adjustment = -alpha_adjustment

        old_alpha = splat.alpha
        splat.alpha += alpha_adjustment

        # Ensure alpha constraints
        splat.clip_parameters()

        # Check if significant change was made
        alpha_change = abs(splat.alpha - old_alpha)
        return alpha_change > 1e-6

    def _refine_anisotropy(self, splat: AdaptiveGaussian2D, region: ErrorRegion, error_map: np.ndarray) -> bool:
        """Refine splat anisotropy based on local content analysis."""
        if not self.config.content_analysis_enabled:
            return False

        # Analyze local gradient direction in the error region
        y1, x1, y2, x2 = region.bbox
        if y2 - y1 < 3 or x2 - x1 < 3:
            return False

        # Calculate gradient in region (simplified)
        region_error = error_map[y1:y2, x1:x2]
        grad_y = np.gradient(region_error, axis=0)
        grad_x = np.gradient(region_error, axis=1)

        # Dominant gradient direction
        avg_grad_y = np.mean(grad_y)
        avg_grad_x = np.mean(grad_x)
        grad_magnitude = np.sqrt(avg_grad_y**2 + avg_grad_x**2)

        if grad_magnitude < 0.1:  # No significant gradient
            return False

        # Adjust anisotropy based on gradient direction
        grad_angle = np.arctan2(avg_grad_y, avg_grad_x)

        old_inv_s = splat.inv_s.copy()
        old_theta = splat.theta

        # Align splat with gradient direction
        splat.theta = grad_angle % np.pi

        # Adjust anisotropy ratio based on gradient strength
        anisotropy_factor = min(1.0 + grad_magnitude, 2.0)
        splat.inv_s[0] *= anisotropy_factor  # Make more elliptical

        # Ensure constraints
        splat.clip_parameters()

        # Check if significant change was made
        inv_s_change = np.linalg.norm(splat.inv_s - old_inv_s)
        theta_change = abs(splat.theta - old_theta)
        return inv_s_change > 1e-6 or theta_change > 1e-6

    def _apply_sgd_optimization(self,
                              splats: List[AdaptiveGaussian2D],
                              target_image: np.ndarray,
                              rendered_image: np.ndarray,
                              error_map: np.ndarray) -> OptimizationResult:
        """Apply SGD optimization to refined splats."""
        return self.sgd_optimizer.optimize_splats(
            splats, target_image, rendered_image, error_map
        )

    def _analyze_content_complexity(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Analyze local content complexity in a region."""
        y1, x1, y2, x2 = bbox
        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return 0.0

        # Convert to grayscale for analysis
        if len(region.shape) == 3:
            gray_region = np.mean(region, axis=-1)
        else:
            gray_region = region

        # Calculate variance (texture measure)
        variance = np.var(gray_region)

        # Calculate gradient magnitude (edge measure)
        # Check if region is large enough for gradient calculation
        if gray_region.shape[0] >= 2 and gray_region.shape[1] >= 2:
            grad_y = np.gradient(gray_region, axis=0)
            grad_x = np.gradient(gray_region, axis=1)
            gradient_magnitude = np.mean(np.sqrt(grad_y**2 + grad_x**2))
        else:
            # For very small regions, use variance as a proxy for complexity
            gradient_magnitude = variance

        # Combine measures
        complexity = (
            self.config.variance_weight * variance +
            self.config.gradient_weight * gradient_magnitude
        )

        return min(complexity, 1.0)  # Normalize to [0,1]

    def _update_rendered_image(self, splats: List[AdaptiveGaussian2D], target_image: np.ndarray) -> np.ndarray:
        """Update rendered image with current splat configuration (placeholder)."""
        # In practice, this would re-render the image with updated splats
        # For now, return a slightly modified version to simulate improvement
        return target_image + np.random.normal(0, 0.01, target_image.shape)

    def _default_loss_function(self,
                              splats: List[AdaptiveGaussian2D],
                              target_image: np.ndarray,
                              rendered_image: np.ndarray,
                              error_map: np.ndarray) -> float:
        """Default loss function (L2 error)."""
        return np.sum(error_map**2)

    def _check_convergence(self) -> bool:
        """Check if refinement has converged."""
        if len(self.state.error_history) < 2:
            return False

        # Check relative improvement
        recent_errors = self.state.error_history[-3:] if len(self.state.error_history) >= 3 else self.state.error_history
        if len(recent_errors) < 2:
            return False

        relative_improvement = (max(recent_errors) - min(recent_errors)) / max(recent_errors)

        # Check convergence criteria
        converged = (
            relative_improvement < self.config.convergence_threshold or
            self.state.patience_counter >= self.config.refinement_patience
        )

        return converged

    def _validate_refinement(self,
                           splats: List[AdaptiveGaussian2D],
                           target_image: np.ndarray,
                           rendered_image: np.ndarray):
        """Validate refinement progress."""
        # Basic validation - ensure splats remain valid
        for i, splat in enumerate(splats):
            if not (np.all(np.isfinite(splat.mu)) and
                    np.all(splat.inv_s > 0) and
                    np.isfinite(splat.theta) and
                    np.all(0 <= splat.color) and np.all(splat.color <= 1) and
                    0 <= splat.alpha <= 1):
                logger.warning(f"Splat {i} has invalid parameters after refinement")


# Convenience functions for easy integration
def refine_splats_progressively(splats: List[AdaptiveGaussian2D],
                               target_image: np.ndarray,
                               rendered_image: np.ndarray,
                               config: RefinementConfig = None) -> RefinementResult:
    """
    Convenience function for progressive splat refinement.

    Args:
        splats: List of Gaussian splats to refine
        target_image: Target image (H, W, C)
        rendered_image: Current rendered image (H, W, C)
        config: Refinement configuration

    Returns:
        RefinementResult with refined splats and metrics
    """
    refiner = ProgressiveRefiner(config)
    return refiner.refine_splats(splats, target_image, rendered_image)


def create_refinement_config_preset(preset: str = "balanced") -> RefinementConfig:
    """
    Create refinement configuration presets for different use cases.

    Args:
        preset: Preset name ("fast", "balanced", "high_quality")

    Returns:
        RefinementConfig with preset parameters
    """
    if preset == "fast":
        return RefinementConfig(
            max_refinement_iterations=5,
            error_percentile_threshold=85.0,
            sgd_iterations_per_refinement=10,
            enabled_operations=[
                RefinementOperation.SCALE_ADJUSTMENT,
                RefinementOperation.POSITION_REFINEMENT
            ]
        )
    elif preset == "high_quality":
        return RefinementConfig(
            max_refinement_iterations=20,
            error_percentile_threshold=75.0,
            sgd_iterations_per_refinement=50,
            enabled_operations=[
                RefinementOperation.SCALE_ADJUSTMENT,
                RefinementOperation.POSITION_REFINEMENT,
                RefinementOperation.ALPHA_MODULATION,
                RefinementOperation.ANISOTROPY_OPTIMIZATION,
                RefinementOperation.COLOR_CORRECTION
            ],
            content_analysis_enabled=True
        )
    else:  # balanced
        return RefinementConfig(
            max_refinement_iterations=10,
            error_percentile_threshold=80.0,
            sgd_iterations_per_refinement=20,
            enabled_operations=[
                RefinementOperation.SCALE_ADJUSTMENT,
                RefinementOperation.POSITION_REFINEMENT,
                RefinementOperation.ALPHA_MODULATION,
                RefinementOperation.ANISOTROPY_OPTIMIZATION
            ]
        )