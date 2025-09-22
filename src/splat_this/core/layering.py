"""Depth scoring and layer assignment for splats."""

from typing import Dict, List
import numpy as np
import logging
from scipy import ndimage

# Import OpenCV with error handling for optional dependency
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

if False:  # TYPE_CHECKING
    from .extract import Gaussian
else:
    from .extract import Gaussian

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """Score splats by visual importance for depth assignment."""

    def __init__(
        self,
        area_weight: float = 0.3,
        edge_weight: float = 0.5,
        color_weight: float = 0.2,
    ):
        self.area_weight = area_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight

    def score_splats(self, splats: List["Gaussian"], image: np.ndarray) -> None:
        """Update splat scores based on importance factors."""
        if not splats:
            return

        logger.info(f"Scoring {len(splats)} splats using multi-factor analysis")

        # Pre-compute image-level metrics for efficiency
        image_area = image.shape[0] * image.shape[1]
        edge_map = self._compute_edge_map(image) if HAS_OPENCV else None

        for splat in splats:
            area_score = self._calculate_area_score(splat, image_area)
            edge_score = self._calculate_edge_score(splat, image, edge_map)
            color_score = self._calculate_color_score(splat, image)

            # Weighted combination with normalization
            weighted_score = (
                area_score * self.area_weight
                + edge_score * self.edge_weight
                + color_score * self.color_weight
            )

            # Ensure score is in [0, 1] range
            splat.score = max(0.0, min(1.0, weighted_score))

        logger.debug(
            f"Scored splats - range: {min(s.score for s in splats):.3f} to {max(s.score for s in splats):.3f}"
        )

    def _compute_edge_map(self, image: np.ndarray) -> np.ndarray:
        """Pre-compute edge strength map for the entire image."""
        if not HAS_OPENCV:
            logger.warning("OpenCV not available, using simplified edge detection")
            # Fallback: use scipy gradient
            gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
            gradient_x = ndimage.sobel(gray, axis=1)
            gradient_y = ndimage.sobel(gray, axis=0)
            return np.sqrt(gradient_x**2 + gradient_y**2)

        # Import cv2 locally to avoid issues during testing
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV import failed, falling back to scipy")
            gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
            gradient_x = ndimage.sobel(gray, axis=1)
            gradient_y = ndimage.sobel(gray, axis=0)
            return np.sqrt(gradient_x**2 + gradient_y**2)

        # Convert to grayscale for edge detection
        gray = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        )

        # Calculate Laplacian for edge strength
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Compute local variance to get edge strength map
        edge_variance_map = ndimage.generic_filter(laplacian, np.var, size=3)

        return edge_variance_map

    def _calculate_area_score(self, splat: "Gaussian", image_area: float) -> float:
        """Score based on relative area (larger splats are more important)."""
        splat_area = splat.area()
        relative_area = splat_area / image_area

        # Use log scaling to prevent very large splats from dominating
        # Score peaks around 1% of image area
        optimal_ratio = 0.01
        if relative_area <= optimal_ratio:
            return relative_area / optimal_ratio
        else:
            # Diminishing returns for very large splats, but keep positive
            log_factor = max(0.1, 1.0 - 0.3 * np.log(relative_area / optimal_ratio))
            return max(0.0, log_factor)

    def _calculate_edge_score(
        self, splat: "Gaussian", image: np.ndarray, edge_map: np.ndarray = None
    ) -> float:
        """Score based on edge strength in splat region."""
        x, y = int(splat.x), int(splat.y)

        # Check bounds
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            return 0.0

        if edge_map is not None:
            # Use pre-computed edge map for efficiency
            rx, ry = int(splat.rx), int(splat.ry)

            # Sample edge strength in elliptical region around splat
            x1, x2 = max(0, x - rx), min(image.shape[1], x + rx)
            y1, y2 = max(0, y - ry), min(image.shape[0], y + ry)

            if x1 >= x2 or y1 >= y2:
                return 0.0

            region_edge_map = edge_map[y1:y2, x1:x2]
            if region_edge_map.size == 0:
                return 0.0

            # Calculate mean edge strength in region
            edge_strength = np.mean(region_edge_map)
            return min(1.0, edge_strength / 255.0)  # Normalize

        else:
            # Fallback: compute edge strength directly
            rx, ry = int(splat.rx), int(splat.ry)
            x1, x2 = max(0, x - rx), min(image.shape[1], x + rx)
            y1, y2 = max(0, y - ry), min(image.shape[0], y + ry)

            if x1 >= x2 or y1 >= y2:
                return 0.0

            region = image[y1:y2, x1:x2]
            if region.size == 0:
                return 0.0

            # Simple gradient-based edge detection
            gray = np.mean(region, axis=2) if len(region.shape) == 3 else region

            # Check if region is large enough for gradient calculation
            if gray.shape[0] < 2 or gray.shape[1] < 2:
                return 0.0

            gradient_x = np.gradient(gray, axis=1)
            gradient_y = np.gradient(gray, axis=0)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

            return min(1.0, np.mean(gradient_magnitude) / 128.0)  # Normalize

    def _calculate_color_score(self, splat: "Gaussian", image: np.ndarray) -> float:
        """Score based on color variance/complexity in splat region."""
        x, y = int(splat.x), int(splat.y)
        rx, ry = int(splat.rx), int(splat.ry)

        # Define bounding box
        x1, x2 = max(0, x - rx), min(image.shape[1], x + rx)
        y1, y2 = max(0, y - ry), min(image.shape[0], y + ry)

        if x1 >= x2 or y1 >= y2:
            return 0.0

        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        # Calculate color variance across all channels
        if len(region.shape) == 3:
            # Multi-channel (RGB)
            color_variance = np.var(region, axis=(0, 1))  # Variance per channel
            total_variance = np.mean(color_variance)  # Average across channels
        else:
            # Grayscale
            total_variance = np.var(region)

        # Normalize (max variance for 8-bit image is 255^2)
        normalized_variance = min(1.0, total_variance / (255.0**2))

        return normalized_variance

    def score_splats_vectorized(
        self, splats: List["Gaussian"], image: np.ndarray
    ) -> None:
        """Optimized vectorized scoring for large splat counts."""
        if not splats:
            return

        logger.info(f"Vectorized scoring for {len(splats)} splats")

        # Pre-compute edge map for entire image
        edge_map = self._compute_edge_map(image)
        image_area = image.shape[0] * image.shape[1]

        # Vectorized coordinate extraction
        positions = np.array([(int(s.x), int(s.y)) for s in splats])

        # Vectorized area scoring
        areas = np.array([s.area() for s in splats])
        relative_areas = areas / image_area
        optimal_ratio = 0.01

        # Vectorized area score calculation
        area_scores = np.where(
            relative_areas <= optimal_ratio,
            relative_areas / optimal_ratio,
            np.maximum(
                0.0, np.maximum(0.1, 1.0 - 0.3 * np.log(relative_areas / optimal_ratio))
            ),
        )

        # Apply scores to splats
        for i, splat in enumerate(splats):
            x, y = positions[i]

            # Sample edge strength at splat location
            if 0 <= y < edge_map.shape[0] and 0 <= x < edge_map.shape[1]:
                edge_score = min(1.0, edge_map[y, x] / 255.0)
            else:
                edge_score = 0.0

            area_score = area_scores[i]
            color_score = self._calculate_color_score(splat, image)

            # Weighted combination
            weighted_score = (
                area_score * self.area_weight
                + edge_score * self.edge_weight
                + color_score * self.color_weight
            )

            splat.score = max(0.0, min(1.0, weighted_score))

        logger.debug(
            f"Vectorized scoring complete - range: {min(s.score for s in splats):.3f} to {max(s.score for s in splats):.3f}"
        )


class LayerAssigner:
    """Assign splats to depth layers based on importance scores."""

    def __init__(self, n_layers: int = 4):
        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers}")
        self.n_layers = n_layers

    def _calculate_depth_value(self, layer_idx: int) -> float:
        """Calculate depth value for a layer index, handling n_layers=1 case."""
        if self.n_layers == 1:
            # When there's only one layer, use middle depth value
            return 0.6  # midpoint between 0.2 and 1.0

        # Normal case: interpolate between 0.2 (back) and 1.0 (front)
        return 0.2 + (layer_idx / (self.n_layers - 1)) * 0.8

    def assign_layers(self, splats: List["Gaussian"]) -> Dict[int, List["Gaussian"]]:
        """Assign splats to depth layers based on scores."""
        if not splats:
            return {}

        logger.info(f"Assigning {len(splats)} splats to {self.n_layers} depth layers")

        # Sort by score (ascending, so highest scores go to front layers)
        sorted_splats = sorted(splats, key=lambda s: s.score)

        # Calculate layer boundaries using percentiles
        layer_size = len(sorted_splats) // self.n_layers
        remainder = len(sorted_splats) % self.n_layers

        layers = {}
        start_idx = 0

        for layer_idx in range(self.n_layers):
            # Distribute remainder across layers (back layers get extra splats first)
            current_layer_size = layer_size + (1 if layer_idx < remainder else 0)
            end_idx = start_idx + current_layer_size

            # Get splats for this layer
            layer_splats = sorted_splats[start_idx:end_idx]

            # Assign depth values (0.2 for back layer, 1.0 for front layer)
            # Higher layer_idx = higher scores = front layers = higher depth values
            depth_value = self._calculate_depth_value(layer_idx)

            for splat in layer_splats:
                splat.depth = depth_value

            layers[layer_idx] = layer_splats
            start_idx = end_idx

        logger.debug(f"Layer assignment complete - {len(layers)} layers created")
        for layer_idx, layer_splats in layers.items():
            if layer_splats:
                scores = [s.score for s in layer_splats]
                logger.debug(f"Layer {layer_idx}: {len(layer_splats)} splats, "
                           f"depth={layer_splats[0].depth:.2f}, "
                           f"score_range=[{min(scores):.3f}, {max(scores):.3f}]")

        return layers

    def get_layer_statistics(self, layers: Dict[int, List["Gaussian"]]) -> Dict[int, dict]:
        """Get statistics for each layer."""
        stats = {}
        for layer_idx, layer_splats in layers.items():
            if layer_splats:
                scores = [s.score for s in layer_splats]
                areas = [s.area() for s in layer_splats]
                stats[layer_idx] = {
                    'count': len(layer_splats),
                    'depth': layer_splats[0].depth,
                    'score_range': (min(scores), max(scores)),
                    'avg_score': sum(scores) / len(scores),
                    'area_range': (min(areas), max(areas)),
                    'avg_area': sum(areas) / len(areas)
                }
            else:
                stats[layer_idx] = {
                    'count': 0,
                    'depth': self._calculate_depth_value(layer_idx),
                    'score_range': (0, 0),
                    'avg_score': 0,
                    'area_range': (0, 0),
                    'avg_area': 0
                }
        return stats

    def balance_layers(self, layers: Dict[int, List["Gaussian"]],
                      min_per_layer: int = 10) -> Dict[int, List["Gaussian"]]:
        """Ensure minimum splats per layer for visual balance."""
        if not layers:
            return layers

        total_splats = sum(len(layer) for layer in layers.values())

        if total_splats < min_per_layer * self.n_layers:
            # Not enough splats for minimum distribution
            logger.warning(f"Not enough splats ({total_splats}) for minimum distribution "
                         f"({min_per_layer} per layer × {self.n_layers} layers)")
            return layers

        logger.info(f"Balancing layers with minimum {min_per_layer} splats per layer")

        # Check if any layer is too small
        needs_balancing = any(len(layer) < min_per_layer for layer in layers.values())

        if not needs_balancing:
            logger.debug("All layers already meet minimum requirements")
            return layers

        # Redistribute if any layer is too small
        balanced_layers = {}
        all_splats = []

        # Collect all splats
        for layer_splats in layers.values():
            all_splats.extend(layer_splats)

        # Re-sort by score
        all_splats.sort(key=lambda s: s.score)

        # Redistribute with minimum guarantees
        target_per_layer = max(min_per_layer, len(all_splats) // self.n_layers)

        for layer_idx in range(self.n_layers):
            start_idx = layer_idx * target_per_layer
            end_idx = min((layer_idx + 1) * target_per_layer, len(all_splats))

            # Handle remainder splats in the last layer
            if layer_idx == self.n_layers - 1:
                end_idx = len(all_splats)

            layer_splats = all_splats[start_idx:end_idx]

            # Assign depth values
            depth_value = self._calculate_depth_value(layer_idx)

            for splat in layer_splats:
                splat.depth = depth_value

            balanced_layers[layer_idx] = layer_splats

        logger.debug(f"Layer balancing complete - new distribution: "
                   f"{[len(layer) for layer in balanced_layers.values()]}")

        return balanced_layers

    def validate_layers(self, layers: Dict[int, List["Gaussian"]]) -> bool:
        """Validate layer assignment consistency."""
        try:
            # Check layer indices are sequential
            expected_indices = set(range(self.n_layers))
            actual_indices = set(layers.keys())

            if expected_indices != actual_indices:
                logger.error(f"Missing layer indices: {expected_indices - actual_indices}")
                return False

            # Check depth values are correctly assigned
            for layer_idx, layer_splats in layers.items():
                expected_depth = self._calculate_depth_value(layer_idx)

                for splat in layer_splats:
                    if abs(splat.depth - expected_depth) > 1e-6:
                        logger.error(f"Incorrect depth in layer {layer_idx}: "
                                   f"expected {expected_depth:.3f}, got {splat.depth:.3f}")
                        return False

            # Check score ordering between layers
            layer_score_ranges = []
            for layer_idx in range(self.n_layers):
                if layers[layer_idx]:
                    scores = [s.score for s in layers[layer_idx]]
                    layer_score_ranges.append((min(scores), max(scores)))
                else:
                    layer_score_ranges.append((0, 0))

            # Verify that later layers generally have higher scores
            for i in range(len(layer_score_ranges) - 1):
                current_max = layer_score_ranges[i][1]
                next_min = layer_score_ranges[i + 1][0]

                # Allow some overlap but warn if ordering is completely wrong
                if current_max > next_min and layer_score_ranges[i + 1][1] > 0:
                    logger.warning(f"Score overlap between layers {i} and {i + 1}: "
                                 f"[{layer_score_ranges[i][0]:.3f}, {current_max:.3f}] vs "
                                 f"[{next_min:.3f}, {layer_score_ranges[i + 1][1]:.3f}]")

            logger.debug("Layer validation passed")
            return True

        except Exception as e:
            logger.error(f"Layer validation failed: {e}")
            return False


class QualityController:
    """Apply quality control and filtering to splat collections."""

    def __init__(
        self,
        target_count: int,
        k_multiplier: float = 2.5,
        min_area_threshold: float = 1.0,
        max_alpha: float = 1.0,
        alpha_adjustment: bool = True,
    ):
        self.target_count = target_count
        self.k_multiplier = k_multiplier
        self.min_area_threshold = min_area_threshold
        self.max_alpha = max_alpha
        self.alpha_adjustment = alpha_adjustment

    def optimize_splats(self, splats: List["Gaussian"]) -> List["Gaussian"]:
        """Apply quality control and filtering to splat collection."""
        if not splats:
            return []

        logger.info(f"Starting quality control on {len(splats)} splats")

        # Step 1: Remove invalid splats
        valid_splats = self._validate_and_cleanup(splats)
        logger.debug(f"After validation: {len(valid_splats)} splats")

        # Step 2: Cull micro-regions (very small splats)
        culled_splats = self._cull_micro_regions(valid_splats)
        logger.debug(f"After micro-region culling: {len(culled_splats)} splats")

        # Step 3: Apply size-based filtering with k_multiplier
        filtered_splats = self._apply_size_filtering(culled_splats)
        logger.debug(f"After size filtering: {len(filtered_splats)} splats")

        # Step 4: Achieve target count through score-based selection
        target_splats = self._achieve_target_count(filtered_splats)
        logger.debug(f"After target count filtering: {len(target_splats)} splats")

        # Step 5: Adjust alpha transparency
        if self.alpha_adjustment:
            final_splats = self._adjust_alpha_transparency(target_splats)
        else:
            final_splats = target_splats

        logger.info(f"Quality control complete: {len(final_splats)} final splats")
        return final_splats

    def _validate_and_cleanup(self, splats: List["Gaussian"]) -> List["Gaussian"]:
        """Remove invalid splats and fix minor issues."""
        valid_splats = []

        for splat in splats:
            try:
                # Check for valid radii
                if splat.rx <= 0 or splat.ry <= 0:
                    continue

                # Check for valid colors
                if not (0 <= splat.r <= 255 and 0 <= splat.g <= 255 and 0 <= splat.b <= 255):
                    continue

                # Check for valid alpha
                if not (0.0 <= splat.a <= 1.0):
                    # Try to fix alpha if it's slightly out of bounds
                    splat.a = max(0.0, min(1.0, splat.a))

                # Check for valid coordinates (not NaN or infinite)
                if not (np.isfinite(splat.x) and np.isfinite(splat.y)):
                    continue

                # Check for valid depth
                if not np.isfinite(splat.depth):
                    splat.depth = 0.5  # Default depth

                # Check for valid score
                if not np.isfinite(splat.score):
                    splat.score = 0.0  # Default score

                valid_splats.append(splat)

            except Exception as e:
                logger.warning(f"Invalid splat encountered: {e}")
                continue

        return valid_splats

    def _cull_micro_regions(self, splats: List["Gaussian"]) -> List["Gaussian"]:
        """Remove splats that are too small to be visually meaningful."""
        if not splats:
            return []

        # Calculate area for each splat
        areas = [splat.area() for splat in splats]

        if not areas:
            return []

        # Use adaptive threshold based on image statistics
        median_area = np.median(areas)
        adaptive_threshold = max(self.min_area_threshold, median_area * 0.01)

        culled_splats = []
        for splat in splats:
            if splat.area() >= adaptive_threshold:
                culled_splats.append(splat)

        logger.debug(f"Micro-region culling: threshold={adaptive_threshold:.2f}, "
                   f"removed {len(splats) - len(culled_splats)} micro-splats")

        return culled_splats

    def _apply_size_filtering(self, splats: List["Gaussian"]) -> List["Gaussian"]:
        """Apply size-based filtering using k_multiplier parameter."""
        if not splats or self.k_multiplier <= 0:
            return splats

        # Calculate areas and sort by size
        splat_areas = [(splat, splat.area()) for splat in splats]
        splat_areas.sort(key=lambda x: x[1], reverse=True)  # Largest first

        # Calculate how many splats to keep based on k_multiplier
        # k_multiplier acts as a factor for the target count
        max_keep = int(self.target_count * self.k_multiplier)

        # Keep top splats by size, but don't exceed the limit
        keep_count = min(len(splats), max_keep)
        filtered_splats = [splat for splat, _ in splat_areas[:keep_count]]

        logger.debug(f"Size filtering: k_multiplier={self.k_multiplier}, "
                   f"max_keep={max_keep}, kept {len(filtered_splats)} splats")

        return filtered_splats

    def _achieve_target_count(self, splats: List["Gaussian"]) -> List["Gaussian"]:
        """Reduce splats to target count using score-based selection."""
        if not splats or len(splats) <= self.target_count:
            return splats

        # Sort by score (descending) to keep highest-scoring splats
        scored_splats = sorted(splats, key=lambda s: s.score, reverse=True)

        # Take top target_count splats
        target_splats = scored_splats[:self.target_count]

        logger.debug(f"Target count filtering: from {len(splats)} to {len(target_splats)} splats")

        return target_splats

    def _adjust_alpha_transparency(self, splats: List["Gaussian"]) -> List["Gaussian"]:
        """Adjust alpha values for optimal visual blending."""
        if not splats:
            return []

        # Calculate alpha adjustment based on splat density and overlap
        adjusted_splats = []

        for splat in splats:
            # Base alpha on splat score and size
            # Higher scoring, larger splats get higher alpha
            score_factor = splat.score
            area_factor = min(1.0, splat.area() / 100.0)  # Normalize to reasonable range

            # Combine factors with some base alpha
            base_alpha = 0.7  # Ensure some transparency for blending
            adjusted_alpha = base_alpha + (1.0 - base_alpha) * (score_factor * 0.7 + area_factor * 0.3)

            # Apply max alpha limit
            final_alpha = min(self.max_alpha, adjusted_alpha)

            # Create new splat with adjusted alpha
            adjusted_splat = Gaussian(
                x=splat.x,
                y=splat.y,
                rx=splat.rx,
                ry=splat.ry,
                theta=splat.theta,
                r=splat.r,
                g=splat.g,
                b=splat.b,
                a=final_alpha,
                score=splat.score,
                depth=splat.depth,
            )
            adjusted_splats.append(adjusted_splat)

        logger.debug(f"Alpha adjustment complete: alpha range "
                   f"[{min(s.a for s in adjusted_splats):.3f}, "
                   f"{max(s.a for s in adjusted_splats):.3f}]")

        return adjusted_splats

    def get_quality_statistics(self, original_splats: List["Gaussian"],
                             final_splats: List["Gaussian"]) -> Dict[str, float]:
        """Get statistics about the quality control process."""
        if not original_splats:
            return {}

        stats = {
            'original_count': len(original_splats),
            'final_count': len(final_splats),
            'reduction_ratio': len(final_splats) / len(original_splats),
            'target_achievement': len(final_splats) / self.target_count if self.target_count > 0 else 0.0,
        }

        if final_splats:
            areas = [s.area() for s in final_splats]
            scores = [s.score for s in final_splats]
            alphas = [s.a for s in final_splats]

            stats.update({
                'avg_area': np.mean(areas),
                'area_std': np.std(areas),
                'avg_score': np.mean(scores),
                'score_std': np.std(scores),
                'avg_alpha': np.mean(alphas),
                'alpha_range': (min(alphas), max(alphas)),
            })

        return stats
