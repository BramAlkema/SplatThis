"""Optimized depth scoring and layer assignment with parallel processing."""

from typing import Dict, List, Tuple
import numpy as np
import logging
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Import OpenCV with error handling for optional dependency
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

from .extract import Gaussian
from ..utils.profiler import global_profiler, MemoryEfficientProcessor

logger = logging.getLogger(__name__)


class OptimizedImportanceScorer:
    """Optimized splat importance scorer with parallel processing."""

    def __init__(
        self,
        area_weight: float = 0.3,
        edge_weight: float = 0.5,
        color_weight: float = 0.2,
        max_workers: int = None,
    ):
        self.area_weight = area_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight

        # Determine optimal worker count
        if max_workers is None:
            cpu_count = psutil.cpu_count(logical=False) or 4
            self.max_workers = min(cpu_count, 8)  # Cap to avoid memory pressure
        else:
            self.max_workers = max_workers

    @global_profiler.profile_function("parallel_scoring")
    def score_splats(self, splats: List[Gaussian], image: np.ndarray) -> None:
        """Update splat scores using parallel processing."""
        if not splats:
            return

        logger.info(f"Scoring {len(splats)} splats using parallel multi-factor analysis")

        # Pre-compute image-level metrics once
        image_area = image.shape[0] * image.shape[1]
        edge_map = self._compute_edge_map(image) if HAS_OPENCV else None

        # Decide whether to use parallel processing
        if len(splats) > 100:
            self._score_splats_parallel(splats, image, image_area, edge_map)
        else:
            self._score_splats_sequential(splats, image, image_area, edge_map)

        logger.info(f"Completed scoring with range: "
                   f"{min(s.score for s in splats):.3f} - {max(s.score for s in splats):.3f}")

    def _score_splats_parallel(
        self,
        splats: List[Gaussian],
        image: np.ndarray,
        image_area: float,
        edge_map: np.ndarray
    ) -> None:
        """Score splats using parallel processing."""
        # Split splats into chunks for workers
        chunk_size = max(10, len(splats) // self.max_workers)
        splat_chunks = [
            splats[i:i + chunk_size]
            for i in range(0, len(splats), chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunks to workers
            future_to_chunk = {
                executor.submit(
                    self._score_splat_chunk,
                    chunk, image, image_area, edge_map
                ): chunk
                for chunk in splat_chunks
            }

            # Collect results and update splats
            for future in as_completed(future_to_chunk):
                try:
                    chunk_scores = future.result()
                    chunk = future_to_chunk[future]

                    # Update splat scores
                    for splat, score in zip(chunk, chunk_scores):
                        splat.score = score

                except Exception as e:
                    logger.warning(f"Failed to score splat chunk: {e}")

    def _score_splats_sequential(
        self,
        splats: List[Gaussian],
        image: np.ndarray,
        image_area: float,
        edge_map: np.ndarray
    ) -> None:
        """Score splats sequentially for smaller counts."""
        for splat in splats:
            area_score = self._calculate_area_score(splat, image_area)
            edge_score = self._calculate_edge_score(splat, image, edge_map)
            color_score = self._calculate_color_score(splat, image)

            # Weighted combination
            splat.score = (
                self.area_weight * area_score +
                self.edge_weight * edge_score +
                self.color_weight * color_score
            )

    def _score_splat_chunk(
        self,
        splats: List[Gaussian],
        image: np.ndarray,
        image_area: float,
        edge_map: np.ndarray
    ) -> List[float]:
        """Score a chunk of splats."""
        scores = []

        for splat in splats:
            area_score = self._calculate_area_score(splat, image_area)
            edge_score = self._calculate_edge_score(splat, image, edge_map)
            color_score = self._calculate_color_score(splat, image)

            # Weighted combination
            score = (
                self.area_weight * area_score +
                self.edge_weight * edge_score +
                self.color_weight * color_score
            )
            scores.append(score)

        return scores

    def _compute_edge_map(self, image: np.ndarray) -> np.ndarray:
        """Compute edge map using optimized method."""
        if not HAS_OPENCV:
            # Fallback to scipy for edge detection
            gray = np.mean(image, axis=2).astype(np.uint8)
            return ndimage.laplace(gray)

        # Use OpenCV for better performance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Use Laplacian for edge detection (faster than Canny for our purposes)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

        # Normalize to 0-1 range
        edge_map = np.abs(laplacian)
        if edge_map.max() > 0:
            edge_map = edge_map / edge_map.max()

        return edge_map

    def _calculate_area_score(self, splat: Gaussian, image_area: float) -> float:
        """Calculate area-based importance score."""
        # Ellipse area = π * rx * ry
        splat_area = np.pi * splat.rx * splat.ry

        # Logarithmic scaling for better distribution
        log_factor = np.log10(1 + splat_area)
        normalized_area = splat_area / image_area

        # Combine log scaling with normalization
        return max(0.0, log_factor * normalized_area * 100)

    def _calculate_edge_score(
        self,
        splat: Gaussian,
        image: np.ndarray,
        edge_map: np.ndarray
    ) -> float:
        """Calculate edge strength score efficiently."""
        if edge_map is None:
            # Fallback: use alpha as proxy for edge importance
            return splat.a

        height, width = edge_map.shape

        # Clamp coordinates to image bounds
        x = max(0, min(int(splat.x), width - 1))
        y = max(0, min(int(splat.y), height - 1))

        # Sample edge strength in small region around splat center
        region_size = max(1, int(min(splat.rx, splat.ry) // 2))

        y_start = max(0, y - region_size)
        y_end = min(height, y + region_size + 1)
        x_start = max(0, x - region_size)
        x_end = min(width, x + region_size + 1)

        if y_start < y_end and x_start < x_end:
            region = edge_map[y_start:y_end, x_start:x_end]
            return float(np.mean(region))
        else:
            return 0.0

    def _calculate_color_score(self, splat: Gaussian, image: np.ndarray) -> float:
        """Calculate color variance score efficiently."""
        height, width = image.shape[:2]

        # Clamp coordinates
        x = max(0, min(int(splat.x), width - 1))
        y = max(0, min(int(splat.y), height - 1))

        # Sample small region for color variance
        region_size = max(1, int(min(splat.rx, splat.ry) // 3))

        y_start = max(0, y - region_size)
        y_end = min(height, y + region_size + 1)
        x_start = max(0, x - region_size)
        x_end = min(width, x + region_size + 1)

        if y_start < y_end and x_start < x_end:
            region = image[y_start:y_end, x_start:x_end]

            # Calculate color variance
            if region.size > 0:
                color_var = np.var(region, axis=(0, 1))
                return float(np.mean(color_var)) / 255.0  # Normalize

        return 0.0


class ParallelQualityController:
    """Optimized quality controller with parallel processing."""

    def __init__(
        self,
        target_count: int = 1500,
        k_multiplier: float = 2.5,
        alpha_adjustment: bool = True,
        max_workers: int = None
    ):
        self.target_count = target_count
        self.k_multiplier = k_multiplier
        self.alpha_adjustment = alpha_adjustment

        if max_workers is None:
            cpu_count = psutil.cpu_count(logical=False) or 4
            self.max_workers = min(cpu_count, 6)
        else:
            self.max_workers = max_workers

    @global_profiler.profile_function("parallel_quality_control")
    def optimize_splats(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Optimize splats using parallel quality control pipeline."""
        if not splats:
            return []

        logger.info(f"Starting parallel quality control for {len(splats)} splats")

        # Step 1: Remove invalid splats (parallel)
        valid_splats = self._filter_valid_splats_parallel(splats)
        logger.debug(f"Valid splats after filtering: {len(valid_splats)}")

        # Step 2: Score-based filtering
        if len(valid_splats) > self.target_count:
            filtered_splats = self._filter_by_score(valid_splats)
        else:
            filtered_splats = valid_splats

        # Step 3: Size-based filtering (parallel)
        size_filtered = self._filter_by_size_parallel(filtered_splats)

        # Step 4: Alpha adjustment (parallel if many splats)
        if self.alpha_adjustment:
            if len(size_filtered) > 200:
                final_splats = self._adjust_alpha_parallel(size_filtered)
            else:
                final_splats = self._adjust_alpha_sequential(size_filtered)
        else:
            final_splats = size_filtered

        # Step 5: Final validation
        validated_splats = self._final_validation(final_splats)

        logger.info(f"Quality control complete: {len(splats)} → {len(validated_splats)} splats")
        return validated_splats

    def _filter_valid_splats_parallel(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Filter invalid splats using parallel processing."""
        if len(splats) < 100:
            return [s for s in splats if self._is_valid_splat(s)]

        # Process in chunks
        chunk_size = max(20, len(splats) // self.max_workers)
        splat_chunks = [
            splats[i:i + chunk_size]
            for i in range(0, len(splats), chunk_size)
        ]

        valid_splats = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._filter_chunk_validity, chunk): chunk
                for chunk in splat_chunks
            }

            for future in as_completed(future_to_chunk):
                try:
                    chunk_valid = future.result()
                    valid_splats.extend(chunk_valid)
                except Exception as e:
                    logger.warning(f"Failed to filter chunk: {e}")

        return valid_splats

    def _filter_chunk_validity(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Filter validity for a chunk of splats."""
        return [s for s in splats if self._is_valid_splat(s)]

    def _is_valid_splat(self, splat: Gaussian) -> bool:
        """Check if splat has valid parameters."""
        return (
            splat.rx > 0 and splat.ry > 0 and
            0 <= splat.r <= 255 and
            0 <= splat.g <= 255 and
            0 <= splat.b <= 255 and
            0.0 <= splat.a <= 1.0
        )

    def _filter_by_score(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Filter splats by score to reach target count."""
        if len(splats) <= self.target_count:
            return splats

        # Sort by score (descending) and take top splats
        sorted_splats = sorted(splats, key=lambda s: s.score, reverse=True)
        return sorted_splats[:self.target_count]

    def _filter_by_size_parallel(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Remove tiny splats using parallel processing."""
        if len(splats) < 100:
            return [s for s in splats if self._is_reasonable_size(s)]

        chunk_size = max(20, len(splats) // self.max_workers)
        splat_chunks = [
            splats[i:i + chunk_size]
            for i in range(0, len(splats), chunk_size)
        ]

        filtered_splats = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._filter_chunk_size, chunk): chunk
                for chunk in splat_chunks
            }

            for future in as_completed(future_to_chunk):
                try:
                    chunk_filtered = future.result()
                    filtered_splats.extend(chunk_filtered)
                except Exception as e:
                    logger.warning(f"Failed to filter chunk by size: {e}")

        return filtered_splats

    def _filter_chunk_size(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Filter size for a chunk of splats."""
        return [s for s in splats if self._is_reasonable_size(s)]

    def _is_reasonable_size(self, splat: Gaussian) -> bool:
        """Check if splat has reasonable size."""
        min_radius = 1.0
        max_radius = 200.0
        return (min_radius <= splat.rx <= max_radius and
                min_radius <= splat.ry <= max_radius)

    def _adjust_alpha_parallel(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Adjust alpha values using parallel processing."""
        chunk_size = max(50, len(splats) // self.max_workers)
        splat_chunks = [
            splats[i:i + chunk_size]
            for i in range(0, len(splats), chunk_size)
        ]

        adjusted_splats = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._adjust_alpha_chunk, chunk): chunk
                for chunk in splat_chunks
            }

            for future in as_completed(future_to_chunk):
                try:
                    chunk_adjusted = future.result()
                    adjusted_splats.extend(chunk_adjusted)
                except Exception as e:
                    logger.warning(f"Failed to adjust alpha for chunk: {e}")

        return adjusted_splats

    def _adjust_alpha_chunk(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Adjust alpha for a chunk of splats."""
        for splat in splats:
            # Score-based alpha adjustment
            alpha_boost = min(0.3, splat.score * 0.5)
            splat.a = min(1.0, splat.a + alpha_boost)
        return splats

    def _adjust_alpha_sequential(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Adjust alpha values sequentially."""
        for splat in splats:
            alpha_boost = min(0.3, splat.score * 0.5)
            splat.a = min(1.0, splat.a + alpha_boost)
        return splats

    def _final_validation(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Final validation pass."""
        return [s for s in splats if self._is_valid_splat(s)]

    def get_quality_statistics(
        self,
        original_splats: List[Gaussian],
        final_splats: List[Gaussian]
    ) -> Dict[str, float]:
        """Get quality control statistics."""
        if not original_splats:
            return {}

        original_count = len(original_splats)
        final_count = len(final_splats)

        # Calculate statistics
        reduction_ratio = 1.0 - (final_count / original_count)

        # Average area and score
        if final_splats:
            avg_area = np.mean([np.pi * s.rx * s.ry for s in final_splats])
            avg_score = np.mean([s.score for s in final_splats])
        else:
            avg_area = 0.0
            avg_score = 0.0

        return {
            'original_count': original_count,
            'final_count': final_count,
            'reduction_ratio': reduction_ratio,
            'avg_area': float(avg_area),
            'avg_score': float(avg_score),
        }