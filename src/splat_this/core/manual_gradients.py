#!/usr/bin/env python3
"""Manual gradient computation for adaptive Gaussian splat optimization."""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class GradientConfig:
    """Configuration for gradient computation."""
    position_step: float = 0.1              # Step size for position gradients
    scale_step: float = 0.001               # Step size for scale gradients
    rotation_step: float = 0.01             # Step size for rotation gradients (radians)
    color_step: float = 0.001               # Step size for color gradients
    gradient_clipping: bool = True          # Enable gradient clipping
    clip_threshold: float = 10.0            # Gradient clipping threshold
    numerical_validation: bool = True       # Enable numerical gradient validation
    finite_diff_method: str = 'central'    # 'forward', 'backward', 'central'
    stability_epsilon: float = 1e-8        # Small value for numerical stability

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.position_step <= 0:
            raise ValueError("position_step must be positive")
        if self.scale_step <= 0:
            raise ValueError("scale_step must be positive")
        if self.rotation_step <= 0:
            raise ValueError("rotation_step must be positive")
        if self.color_step <= 0:
            raise ValueError("color_step must be positive")
        if self.clip_threshold <= 0:
            raise ValueError("clip_threshold must be positive")
        if self.stability_epsilon <= 0:
            raise ValueError("stability_epsilon must be positive")

        valid_methods = ['forward', 'backward', 'central']
        if self.finite_diff_method not in valid_methods:
            raise ValueError(f"finite_diff_method must be one of {valid_methods}")


@dataclass
class SplatGradients:
    """Gradients for a single Gaussian splat."""
    position_grad: np.ndarray    # Position gradient (2,) for 2D
    scale_grad: np.ndarray      # Scale gradient (2,) for anisotropic
    rotation_grad: float        # Rotation gradient (scalar)
    color_grad: np.ndarray      # Color gradient (3,) for RGB
    alpha_grad: float          # Alpha gradient (scalar)

    def __post_init__(self):
        """Validate gradient shapes."""
        if self.position_grad.shape != (2,):
            raise ValueError("position_grad must have shape (2,)")
        if self.scale_grad.shape != (2,):
            raise ValueError("scale_grad must have shape (2,)")
        if self.color_grad.shape != (3,):
            raise ValueError("color_grad must have shape (3,)")
        if not isinstance(self.rotation_grad, (float, int, np.number)):
            raise ValueError("rotation_grad must be scalar")
        if not isinstance(self.alpha_grad, (float, int, np.number)):
            raise ValueError("alpha_grad must be scalar")


@dataclass
class GradientValidation:
    """Results from numerical gradient validation."""
    position_error: float       # Error between analytical and numerical position gradients
    scale_error: float         # Error between analytical and numerical scale gradients
    rotation_error: float      # Error between analytical and numerical rotation gradients
    color_error: float         # Error between analytical and numerical color gradients
    alpha_error: float         # Error between analytical and numerical alpha gradients
    max_error: float          # Maximum error across all parameters
    passed: bool              # Whether validation passed


class ManualGradientComputer:
    """Compute gradients manually for Gaussian splat parameters."""

    def __init__(self, config: Optional[GradientConfig] = None):
        """Initialize gradient computer.

        Args:
            config: Gradient computation configuration
        """
        self.config = config or GradientConfig()

    def compute_position_gradient(self, splat, target_image: np.ndarray,
                                rendered_image: np.ndarray,
                                error_map: np.ndarray) -> np.ndarray:
        """Compute position gradient from error map.

        Args:
            splat: Gaussian splat object with position, scale, rotation, color, alpha
            target_image: Target image (H, W, C)
            rendered_image: Current rendered image (H, W, C)
            error_map: Pixel-wise error map (H, W)

        Returns:
            Position gradient (2,) as [dy, dx]
        """
        h, w = error_map.shape
        current_pos = np.array(splat.mu)

        # Compute gradient using finite differences
        if self.config.finite_diff_method == 'central':
            grad = self._central_difference_position(
                splat, target_image, rendered_image, error_map
            )
        elif self.config.finite_diff_method == 'forward':
            grad = self._forward_difference_position(
                splat, target_image, rendered_image, error_map
            )
        else:  # backward
            grad = self._backward_difference_position(
                splat, target_image, rendered_image, error_map
            )

        # Apply gradient clipping if enabled
        if self.config.gradient_clipping:
            grad = self._clip_gradient(grad, self.config.clip_threshold)

        return grad

    def compute_scale_gradient(self, splat, target_image: np.ndarray,
                             rendered_image: np.ndarray,
                             error_map: np.ndarray) -> np.ndarray:
        """Compute scale gradient for anisotropy optimization.

        Args:
            splat: Gaussian splat object
            target_image: Target image (H, W, C)
            rendered_image: Current rendered image (H, W, C)
            error_map: Pixel-wise error map (H, W)

        Returns:
            Scale gradient (2,) as [scale_x_grad, scale_y_grad]
        """
        current_scale = 1.0 / np.array(splat.inv_s)  # Convert from inverse scales

        if self.config.finite_diff_method == 'central':
            grad = self._central_difference_scale(
                splat, target_image, rendered_image, error_map
            )
        elif self.config.finite_diff_method == 'forward':
            grad = self._forward_difference_scale(
                splat, target_image, rendered_image, error_map
            )
        else:  # backward
            grad = self._backward_difference_scale(
                splat, target_image, rendered_image, error_map
            )

        # Apply gradient clipping
        if self.config.gradient_clipping:
            grad = self._clip_gradient(grad, self.config.clip_threshold)

        return grad

    def compute_rotation_gradient(self, splat, target_image: np.ndarray,
                                rendered_image: np.ndarray,
                                error_map: np.ndarray) -> float:
        """Compute rotation gradient for orientation refinement.

        Args:
            splat: Gaussian splat object
            target_image: Target image (H, W, C)
            rendered_image: Current rendered image (H, W, C)
            error_map: Pixel-wise error map (H, W)

        Returns:
            Rotation gradient (scalar)
        """
        current_rotation = splat.theta

        if self.config.finite_diff_method == 'central':
            grad = self._central_difference_rotation(
                splat, target_image, rendered_image, error_map
            )
        elif self.config.finite_diff_method == 'forward':
            grad = self._forward_difference_rotation(
                splat, target_image, rendered_image, error_map
            )
        else:  # backward
            grad = self._backward_difference_rotation(
                splat, target_image, rendered_image, error_map
            )

        # Apply gradient clipping
        if self.config.gradient_clipping:
            grad = self._clip_scalar_gradient(grad, self.config.clip_threshold)

        return grad

    def compute_color_gradient(self, splat, target_image: np.ndarray,
                             rendered_image: np.ndarray,
                             error_map: np.ndarray) -> np.ndarray:
        """Compute color gradient for appearance optimization.

        Args:
            splat: Gaussian splat object
            target_image: Target image (H, W, C)
            rendered_image: Current rendered image (H, W, C)
            error_map: Pixel-wise error map (H, W)

        Returns:
            Color gradient (3,) as [r_grad, g_grad, b_grad]
        """
        current_color = np.array(splat.color)

        if self.config.finite_diff_method == 'central':
            grad = self._central_difference_color(
                splat, target_image, rendered_image, error_map
            )
        elif self.config.finite_diff_method == 'forward':
            grad = self._forward_difference_color(
                splat, target_image, rendered_image, error_map
            )
        else:  # backward
            grad = self._backward_difference_color(
                splat, target_image, rendered_image, error_map
            )

        # Apply gradient clipping
        if self.config.gradient_clipping:
            grad = self._clip_gradient(grad, self.config.clip_threshold)

        return grad

    def compute_alpha_gradient(self, splat, target_image: np.ndarray,
                             rendered_image: np.ndarray,
                             error_map: np.ndarray) -> float:
        """Compute alpha gradient for opacity optimization.

        Args:
            splat: Gaussian splat object
            target_image: Target image (H, W, C)
            rendered_image: Current rendered image (H, W, C)
            error_map: Pixel-wise error map (H, W)

        Returns:
            Alpha gradient (scalar)
        """
        current_alpha = splat.alpha

        if self.config.finite_diff_method == 'central':
            grad = self._central_difference_alpha(
                splat, target_image, rendered_image, error_map
            )
        elif self.config.finite_diff_method == 'forward':
            grad = self._forward_difference_alpha(
                splat, target_image, rendered_image, error_map
            )
        else:  # backward
            grad = self._backward_difference_alpha(
                splat, target_image, rendered_image, error_map
            )

        # Apply gradient clipping
        if self.config.gradient_clipping:
            grad = self._clip_scalar_gradient(grad, self.config.clip_threshold)

        return grad

    def compute_all_gradients(self, splat, target_image: np.ndarray,
                            rendered_image: np.ndarray,
                            error_map: np.ndarray) -> SplatGradients:
        """Compute all gradients for a single splat.

        Args:
            splat: Gaussian splat object
            target_image: Target image (H, W, C)
            rendered_image: Current rendered image (H, W, C)
            error_map: Pixel-wise error map (H, W)

        Returns:
            SplatGradients containing all parameter gradients
        """
        position_grad = self.compute_position_gradient(
            splat, target_image, rendered_image, error_map
        )
        scale_grad = self.compute_scale_gradient(
            splat, target_image, rendered_image, error_map
        )
        rotation_grad = self.compute_rotation_gradient(
            splat, target_image, rendered_image, error_map
        )
        color_grad = self.compute_color_gradient(
            splat, target_image, rendered_image, error_map
        )
        alpha_grad = self.compute_alpha_gradient(
            splat, target_image, rendered_image, error_map
        )

        return SplatGradients(
            position_grad=position_grad,
            scale_grad=scale_grad,
            rotation_grad=rotation_grad,
            color_grad=color_grad,
            alpha_grad=alpha_grad
        )

    def validate_gradients(self, splat, target_image: np.ndarray,
                         rendered_image: np.ndarray,
                         error_map: np.ndarray,
                         analytical_gradients: SplatGradients) -> GradientValidation:
        """Validate analytical gradients against numerical gradients.

        Args:
            splat: Gaussian splat object
            target_image: Target image (H, W, C)
            rendered_image: Current rendered image (H, W, C)
            error_map: Pixel-wise error map (H, W)
            analytical_gradients: Computed analytical gradients

        Returns:
            GradientValidation with error metrics
        """
        # Compute numerical gradients with smaller step sizes
        small_config = GradientConfig(
            position_step=self.config.position_step * 0.1,
            scale_step=self.config.scale_step * 0.1,
            rotation_step=self.config.rotation_step * 0.1,
            color_step=self.config.color_step * 0.1,
            finite_diff_method='central',
            gradient_clipping=False  # No clipping for validation
        )

        numerical_computer = ManualGradientComputer(small_config)
        numerical_gradients = numerical_computer.compute_all_gradients(
            splat, target_image, rendered_image, error_map
        )

        # Compute errors
        position_error = np.linalg.norm(
            analytical_gradients.position_grad - numerical_gradients.position_grad
        )
        scale_error = np.linalg.norm(
            analytical_gradients.scale_grad - numerical_gradients.scale_grad
        )
        rotation_error = abs(
            analytical_gradients.rotation_grad - numerical_gradients.rotation_grad
        )
        color_error = np.linalg.norm(
            analytical_gradients.color_grad - numerical_gradients.color_grad
        )
        alpha_error = abs(
            analytical_gradients.alpha_grad - numerical_gradients.alpha_grad
        )

        max_error = max(position_error, scale_error, rotation_error, color_error, alpha_error)

        # Check if validation passes (relative error < 10%)
        tolerance = 0.1
        passed = (
            position_error < tolerance * np.linalg.norm(analytical_gradients.position_grad + self.config.stability_epsilon) and
            scale_error < tolerance * np.linalg.norm(analytical_gradients.scale_grad + self.config.stability_epsilon) and
            rotation_error < tolerance * abs(analytical_gradients.rotation_grad + self.config.stability_epsilon) and
            color_error < tolerance * np.linalg.norm(analytical_gradients.color_grad + self.config.stability_epsilon) and
            alpha_error < tolerance * abs(analytical_gradients.alpha_grad + self.config.stability_epsilon)
        )

        return GradientValidation(
            position_error=position_error,
            scale_error=scale_error,
            rotation_error=rotation_error,
            color_error=color_error,
            alpha_error=alpha_error,
            max_error=max_error,
            passed=bool(passed)
        )

    def _compute_error_change(self, splat, target_image: np.ndarray,
                            rendered_image: np.ndarray) -> float:
        """Compute total error for the current splat configuration."""
        # This is a placeholder - in practice, this would render the splat
        # and compute the L2 error against the target
        from .error_analysis import ErrorAnalyzer

        analyzer = ErrorAnalyzer()
        metrics = analyzer.compute_basic_metrics(target_image, rendered_image)
        return metrics.l2_error

    def _central_difference_position(self, splat, target_image: np.ndarray,
                                   rendered_image: np.ndarray,
                                   error_map: np.ndarray) -> np.ndarray:
        """Compute position gradient using central differences."""
        grad = np.zeros(2)
        step = self.config.position_step

        for i in range(2):  # x and y
            # Create perturbed splat copies
            splat_pos = splat.mu.copy()

            # Forward step
            splat_pos[i] += step
            splat_forward = self._create_perturbed_splat(splat, position=splat_pos)
            error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

            # Backward step
            splat_pos[i] -= 2 * step
            splat_backward = self._create_perturbed_splat(splat, position=splat_pos)
            error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

            # Central difference
            grad[i] = (error_forward - error_backward) / (2 * step)

        return grad

    def _central_difference_scale(self, splat, target_image: np.ndarray,
                                rendered_image: np.ndarray,
                                error_map: np.ndarray) -> np.ndarray:
        """Compute scale gradient using central differences."""
        grad = np.zeros(2)
        step = self.config.scale_step

        for i in range(2):  # scale_x and scale_y
            splat_scale = (1.0 / splat.inv_s).copy()

            # Forward step
            splat_scale[i] += step
            splat_forward = self._create_perturbed_splat(splat, scale=splat_scale)
            error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

            # Backward step
            splat_scale[i] -= 2 * step
            splat_backward = self._create_perturbed_splat(splat, scale=splat_scale)
            error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

            # Central difference
            grad[i] = (error_forward - error_backward) / (2 * step)

        return grad

    def _central_difference_rotation(self, splat, target_image: np.ndarray,
                                   rendered_image: np.ndarray,
                                   error_map: np.ndarray) -> float:
        """Compute rotation gradient using central differences."""
        step = self.config.rotation_step

        # Forward step
        splat_forward = self._create_perturbed_splat(splat, rotation=splat.theta + step)
        error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

        # Backward step
        splat_backward = self._create_perturbed_splat(splat, rotation=splat.theta - step)
        error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

        # Central difference
        grad = (error_forward - error_backward) / (2 * step)

        return grad

    def _central_difference_color(self, splat, target_image: np.ndarray,
                                rendered_image: np.ndarray,
                                error_map: np.ndarray) -> np.ndarray:
        """Compute color gradient using central differences."""
        grad = np.zeros(3)
        step = self.config.color_step

        for i in range(3):  # R, G, B
            splat_color = splat.color.copy()

            # Forward step
            splat_color[i] += step
            splat_forward = self._create_perturbed_splat(splat, color=splat_color)
            error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

            # Backward step
            splat_color[i] -= 2 * step
            splat_backward = self._create_perturbed_splat(splat, color=splat_color)
            error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

            # Central difference
            grad[i] = (error_forward - error_backward) / (2 * step)

        return grad

    def _central_difference_alpha(self, splat, target_image: np.ndarray,
                                rendered_image: np.ndarray,
                                error_map: np.ndarray) -> float:
        """Compute alpha gradient using central differences."""
        step = self.config.color_step  # Use same step as color

        # Forward step
        splat_forward = self._create_perturbed_splat(splat, alpha=splat.alpha + step)
        error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

        # Backward step
        splat_backward = self._create_perturbed_splat(splat, alpha=splat.alpha - step)
        error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

        # Central difference
        grad = (error_forward - error_backward) / (2 * step)

        return grad

    def _forward_difference_position(self, splat, target_image: np.ndarray,
                                   rendered_image: np.ndarray,
                                   error_map: np.ndarray) -> np.ndarray:
        """Compute position gradient using forward differences."""
        grad = np.zeros(2)
        step = self.config.position_step

        # Current error
        error_current = self._compute_error_change(splat, target_image, rendered_image)

        for i in range(2):
            # Forward step
            splat_pos = splat.mu.copy()
            splat_pos[i] += step
            splat_forward = self._create_perturbed_splat(splat, position=splat_pos)
            error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

            # Forward difference
            grad[i] = (error_forward - error_current) / step

        return grad

    def _forward_difference_scale(self, splat, target_image: np.ndarray,
                                rendered_image: np.ndarray,
                                error_map: np.ndarray) -> np.ndarray:
        """Compute scale gradient using forward differences."""
        grad = np.zeros(2)
        step = self.config.scale_step

        error_current = self._compute_error_change(splat, target_image, rendered_image)

        for i in range(2):
            splat_scale = (1.0 / splat.inv_s).copy()
            splat_scale[i] += step
            splat_forward = self._create_perturbed_splat(splat, scale=splat_scale)
            error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

            grad[i] = (error_forward - error_current) / step

        return grad

    def _forward_difference_rotation(self, splat, target_image: np.ndarray,
                                   rendered_image: np.ndarray,
                                   error_map: np.ndarray) -> float:
        """Compute rotation gradient using forward differences."""
        step = self.config.rotation_step

        error_current = self._compute_error_change(splat, target_image, rendered_image)

        splat_forward = self._create_perturbed_splat(splat, rotation=splat.theta + step)
        error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

        grad = (error_forward - error_current) / step

        return grad

    def _forward_difference_color(self, splat, target_image: np.ndarray,
                                rendered_image: np.ndarray,
                                error_map: np.ndarray) -> np.ndarray:
        """Compute color gradient using forward differences."""
        grad = np.zeros(3)
        step = self.config.color_step

        error_current = self._compute_error_change(splat, target_image, rendered_image)

        for i in range(3):
            splat_color = splat.color.copy()
            splat_color[i] += step
            splat_forward = self._create_perturbed_splat(splat, color=splat_color)
            error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

            grad[i] = (error_forward - error_current) / step

        return grad

    def _forward_difference_alpha(self, splat, target_image: np.ndarray,
                                rendered_image: np.ndarray,
                                error_map: np.ndarray) -> float:
        """Compute alpha gradient using forward differences."""
        step = self.config.color_step

        error_current = self._compute_error_change(splat, target_image, rendered_image)

        splat_forward = self._create_perturbed_splat(splat, alpha=splat.alpha + step)
        error_forward = self._compute_error_change(splat_forward, target_image, rendered_image)

        grad = (error_forward - error_current) / step

        return grad

    def _backward_difference_position(self, splat, target_image: np.ndarray,
                                    rendered_image: np.ndarray,
                                    error_map: np.ndarray) -> np.ndarray:
        """Compute position gradient using backward differences."""
        grad = np.zeros(2)
        step = self.config.position_step

        error_current = self._compute_error_change(splat, target_image, rendered_image)

        for i in range(2):
            splat_pos = splat.mu.copy()
            splat_pos[i] -= step
            splat_backward = self._create_perturbed_splat(splat, position=splat_pos)
            error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

            grad[i] = (error_current - error_backward) / step

        return grad

    def _backward_difference_scale(self, splat, target_image: np.ndarray,
                                 rendered_image: np.ndarray,
                                 error_map: np.ndarray) -> np.ndarray:
        """Compute scale gradient using backward differences."""
        grad = np.zeros(2)
        step = self.config.scale_step

        error_current = self._compute_error_change(splat, target_image, rendered_image)

        for i in range(2):
            splat_scale = (1.0 / splat.inv_s).copy()
            splat_scale[i] -= step
            splat_backward = self._create_perturbed_splat(splat, scale=splat_scale)
            error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

            grad[i] = (error_current - error_backward) / step

        return grad

    def _backward_difference_rotation(self, splat, target_image: np.ndarray,
                                    rendered_image: np.ndarray,
                                    error_map: np.ndarray) -> float:
        """Compute rotation gradient using backward differences."""
        step = self.config.rotation_step

        error_current = self._compute_error_change(splat, target_image, rendered_image)

        splat_backward = self._create_perturbed_splat(splat, rotation=splat.theta - step)
        error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

        grad = (error_current - error_backward) / step

        return grad

    def _backward_difference_color(self, splat, target_image: np.ndarray,
                                 rendered_image: np.ndarray,
                                 error_map: np.ndarray) -> np.ndarray:
        """Compute color gradient using backward differences."""
        grad = np.zeros(3)
        step = self.config.color_step

        error_current = self._compute_error_change(splat, target_image, rendered_image)

        for i in range(3):
            splat_color = splat.color.copy()
            splat_color[i] -= step
            splat_backward = self._create_perturbed_splat(splat, color=splat_color)
            error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

            grad[i] = (error_current - error_backward) / step

        return grad

    def _backward_difference_alpha(self, splat, target_image: np.ndarray,
                                 rendered_image: np.ndarray,
                                 error_map: np.ndarray) -> float:
        """Compute alpha gradient using backward differences."""
        step = self.config.color_step

        error_current = self._compute_error_change(splat, target_image, rendered_image)

        splat_backward = self._create_perturbed_splat(splat, alpha=splat.alpha - step)
        error_backward = self._compute_error_change(splat_backward, target_image, rendered_image)

        grad = (error_current - error_backward) / step

        return grad

    def _create_perturbed_splat(self, original_splat, **kwargs):
        """Create a copy of splat with perturbed parameters."""
        from .adaptive_gaussian import AdaptiveGaussian2D

        # Create a copy with potentially modified parameters
        position = kwargs.get('position', original_splat.mu)
        scale = kwargs.get('scale', 1.0 / original_splat.inv_s)  # Convert from inverse scales
        rotation = kwargs.get('rotation', original_splat.theta)
        color = kwargs.get('color', original_splat.color)
        alpha = kwargs.get('alpha', original_splat.alpha)

        # Pre-clip values to ensure they are within valid ranges
        alpha = np.clip(alpha, 0.0, 1.0)
        if hasattr(color, '__len__'):
            color = np.clip(color, 0.0, 1.0)

        # Convert scale back to inverse scale for constructor
        if 'scale' in kwargs:
            # Protect against zero or negative scales
            scale_array = np.array(scale)
            scale_array = np.clip(scale_array, 1e-6, 1e6)
            inv_s = 1.0 / scale_array
        else:
            inv_s = original_splat.inv_s.copy()

        # Create new splat with perturbed parameters
        perturbed_splat = AdaptiveGaussian2D(
            mu=position,
            inv_s=inv_s,
            theta=rotation,
            color=color,
            alpha=alpha,
            content_complexity=original_splat.content_complexity,
            saliency_score=original_splat.saliency_score,
            refinement_count=original_splat.refinement_count
        )

        # Clip parameters to valid ranges for numerical stability
        perturbed_splat.clip_parameters()

        return perturbed_splat

    def _clip_gradient(self, gradient: np.ndarray, threshold: float) -> np.ndarray:
        """Clip gradient magnitude to threshold."""
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > threshold:
            return gradient * (threshold / grad_norm)
        return gradient

    def _clip_scalar_gradient(self, gradient: float, threshold: float) -> float:
        """Clip scalar gradient to threshold."""
        return np.clip(gradient, -threshold, threshold)


def compute_splat_gradients(splat, target_image: np.ndarray,
                          rendered_image: np.ndarray,
                          error_map: np.ndarray,
                          config: Optional[GradientConfig] = None) -> SplatGradients:
    """Convenience function to compute all gradients for a splat.

    Args:
        splat: Gaussian splat object
        target_image: Target image (H, W, C)
        rendered_image: Current rendered image (H, W, C)
        error_map: Pixel-wise error map (H, W)
        config: Gradient computation configuration

    Returns:
        SplatGradients containing all parameter gradients
    """
    computer = ManualGradientComputer(config)
    return computer.compute_all_gradients(splat, target_image, rendered_image, error_map)


def validate_gradient_computation(splat, target_image: np.ndarray,
                                rendered_image: np.ndarray,
                                error_map: np.ndarray,
                                config: Optional[GradientConfig] = None) -> GradientValidation:
    """Convenience function to validate gradient computation.

    Args:
        splat: Gaussian splat object
        target_image: Target image (H, W, C)
        rendered_image: Current rendered image (H, W, C)
        error_map: Pixel-wise error map (H, W)
        config: Gradient computation configuration

    Returns:
        GradientValidation with error metrics
    """
    computer = ManualGradientComputer(config)
    gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)
    return computer.validate_gradients(splat, target_image, rendered_image, error_map, gradients)