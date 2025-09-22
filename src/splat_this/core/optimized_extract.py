"""Optimized splat extraction with performance improvements."""

import logging
import numpy as np
from typing import List, Optional, Tuple
from skimage.segmentation import slic
from skimage.color import rgb2lab
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from .extract import Gaussian
from ..utils.profiler import global_profiler, MemoryEfficientProcessor
from ..utils.math import safe_eigendecomposition, clamp_value

logger = logging.getLogger(__name__)


class OptimizedSplatExtractor:
    """Optimized Gaussian splat extractor with performance monitoring."""

    def __init__(self, k: float = 2.5, base_alpha: float = 0.65, max_workers: Optional[int] = None, max_memory_mb: Optional[int] = None):
        self.k = k
        self.base_alpha = base_alpha
        self.memory_processor = MemoryEfficientProcessor(max_memory_mb=max_memory_mb) if max_memory_mb else MemoryEfficientProcessor()

        # Determine optimal number of workers
        if max_workers is None:
            # Use number of CPU cores, but limit to avoid memory issues
            cpu_count = psutil.cpu_count(logical=False) or 4
            self.max_workers = min(cpu_count, 8)  # Cap at 8 to avoid memory pressure
        else:
            self.max_workers = max_workers

    @global_profiler.profile_function("slic_segmentation")
    def extract_splats(self, image: np.ndarray, n_splats: int) -> List[Gaussian]:
        """Extract Gaussian splats with performance optimization."""
        height, width = image.shape[:2]
        logger.info(
            f"Extracting {n_splats} splats from {width}×{height} image using optimized SLIC"
        )

        # Check if we should downsample the image BEFORE enforcing memory limit
        should_downsample, new_size = self.memory_processor.should_downsample_image(
            (width, height), n_splats
        )

        if should_downsample:
            scale_x = new_size[0] / width
            scale_y = new_size[1] / height
            logger.info(f"Downsampling image from {width}×{height} to {new_size[0]}×{new_size[1]} for memory efficiency")

            # Use optimized downsampling - this replaces the large image with a smaller one
            image = self._downsample_image(image, new_size)
            # Force garbage collection to free the original large image from memory
            import gc
            gc.collect()

            # Update dimensions after downsampling
            height, width = image.shape[:2]
            logger.debug(f"Image downsampled to actual shape: {width}×{height}")
        else:
            scale_x = scale_y = 1.0

        # Check memory AFTER potential downsampling
        # At this point, the image has been downsampled if needed
        self.memory_processor.ensure_memory_limit("extract_splats_after_downsample")

        # Perform optimized SLIC segmentation
        segments = self._optimized_slic(image, n_splats)

        self.memory_processor.ensure_memory_limit("post_slic")

        # Extract splats using parallel processing for large segment counts
        unique_segments = np.unique(segments)
        logger.debug(f"SLIC generated {len(unique_segments)} unique segments")

        if len(unique_segments) > 100:  # Use parallel processing for many segments
            splats = self._extract_splats_parallel(image, segments, unique_segments, scale_x, scale_y)
        else:
            splats = self._extract_splats_sequential(image, segments, unique_segments, scale_x, scale_y)

        logger.info(f"Successfully extracted {len(splats)} splats")
        return splats

    def _downsample_image(self, image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """Efficiently downsample image."""
        from skimage.transform import resize

        # Use anti-aliasing for better quality
        return (resize(image, (new_size[1], new_size[0]), anti_aliasing=True) * 255).astype(np.uint8)

    @global_profiler.profile_function("slic_algorithm")
    def _optimized_slic(self, image: np.ndarray, n_splats: int) -> np.ndarray:
        """Optimized SLIC segmentation with better parameters."""
        # Convert to LAB color space for better perceptual uniformity
        lab_image = rgb2lab(image)

        # Optimized SLIC parameters for better performance/quality balance
        segments = slic(
            lab_image,
            n_segments=n_splats,
            compactness=10.0,
            sigma=1.0,
            start_label=1,
            convert2lab=False,  # Already converted
            max_num_iter=10,  # Limit iterations for performance
            enforce_connectivity=True,  # Better segments
        )

        return segments

    @global_profiler.profile_function("parallel_splat_extraction")
    def _extract_splats_parallel(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        unique_segments: np.ndarray,
        scale_x: float,
        scale_y: float
    ) -> List[Gaussian]:
        """Extract splats using parallel processing."""
        splats = []

        # Split segments into chunks for workers
        chunk_size = max(10, len(unique_segments) // self.max_workers)
        segment_chunks = [
            unique_segments[i:i + chunk_size]
            for i in range(0, len(unique_segments), chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunks to workers
            future_to_chunk = {
                executor.submit(
                    self._process_segment_chunk,
                    image, segments, chunk, scale_x, scale_y
                ): chunk
                for chunk in segment_chunks
            }

            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_splats = future.result()
                    splats.extend(chunk_splats)
                except Exception as e:
                    logger.warning(f"Failed to process segment chunk: {e}")

        return splats

    def _process_segment_chunk(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        segment_ids: np.ndarray,
        scale_x: float,
        scale_y: float
    ) -> List[Gaussian]:
        """Process a chunk of segments."""
        chunk_splats = []

        for segment_id in segment_ids:
            if segment_id == 0:  # Skip background
                continue

            # Create mask for this segment
            mask = segments == segment_id

            # Skip very small segments (optimized check)
            pixel_count = np.sum(mask)
            if pixel_count < 10:
                continue

            try:
                splat = self._segment_to_gaussian_optimized(
                    image, mask, segment_id, scale_x, scale_y
                )
                if splat is not None:
                    chunk_splats.append(splat)
            except Exception as e:
                logger.debug(f"Failed to extract splat from segment {segment_id}: {e}")
                continue

        return chunk_splats

    def _extract_splats_sequential(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        unique_segments: np.ndarray,
        scale_x: float,
        scale_y: float
    ) -> List[Gaussian]:
        """Extract splats sequentially (for smaller segment counts)."""
        splats = []

        for segment_id in unique_segments:
            if segment_id == 0:  # Skip background
                continue

            # Create mask for this segment
            mask = segments == segment_id

            # Skip very small segments
            pixel_count = np.sum(mask)
            if pixel_count < 10:
                continue

            try:
                splat = self._segment_to_gaussian_optimized(
                    image, mask, segment_id, scale_x, scale_y
                )
                if splat is not None:
                    splats.append(splat)
            except Exception as e:
                logger.warning(f"Failed to extract splat from segment {segment_id}: {e}")
                continue

        return splats

    def _segment_to_gaussian_optimized(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        segment_id: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0
    ) -> Optional[Gaussian]:
        """Optimized conversion of image segment to Gaussian splat."""
        # Get pixel coordinates where mask is True (vectorized)
        y_coords, x_coords = np.where(mask)

        if len(x_coords) == 0:
            return None

        # Calculate centroid (vectorized)
        cx = float(np.mean(x_coords))
        cy = float(np.mean(y_coords))

        # Calculate covariance matrix for ellipse parameters (optimized)
        try:
            # Center the coordinates
            x_centered = x_coords - cx
            y_centered = y_coords - cy

            # Compute covariance matrix elements directly
            cov_xx = np.mean(x_centered * x_centered)
            cov_yy = np.mean(y_centered * y_centered)
            cov_xy = np.mean(x_centered * y_centered)

            # Create covariance matrix
            cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

            # Safe eigendecomposition
            eigenvalues, eigenvectors = safe_eigendecomposition(cov_matrix)

            # Clamp eigenvalues to prevent negative values (numerical noise) causing NaN
            eigenvalues = np.maximum(eigenvalues, 1e-6)  # Small positive minimum

            # Scale eigenvalues by k factor and ensure minimum size
            rx = max(1.0, np.sqrt(eigenvalues[0]) * self.k)
            ry = max(1.0, np.sqrt(eigenvalues[1]) * self.k)

            # Calculate rotation angle from principal eigenvector (largest eigenvalue)
            principal_idx = int(np.argmax(eigenvalues))
            theta = float(
                np.arctan2(
                    eigenvectors[1, principal_idx],
                    eigenvectors[0, principal_idx],
                )
            )

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug(f"Eigendecomposition failed for segment {segment_id}: {e}")
            # Fallback to bounding box approximation
            rx = max(1.0, (np.max(x_coords) - np.min(x_coords)) / 2.0)
            ry = max(1.0, (np.max(y_coords) - np.min(y_coords)) / 2.0)
            theta = 0.0

        # Calculate average color (vectorized)
        masked_pixels = image[mask]
        avg_color = np.mean(masked_pixels, axis=0)

        # Scale coordinates back to original image size if downsampled
        cx_scaled = cx / scale_x
        cy_scaled = cy / scale_y
        rx_scaled = rx / scale_x
        ry_scaled = ry / scale_y

        # Clamp values to valid ranges
        r = clamp_value(int(avg_color[0]), 0, 255)
        g = clamp_value(int(avg_color[1]), 0, 255)
        b = clamp_value(int(avg_color[2]), 0, 255)

        # Calculate alpha based on segment size (larger segments = more opaque)
        pixel_count = len(x_coords)
        area_factor = min(1.0, pixel_count / 1000.0)  # Normalize by 1000 pixels
        alpha = clamp_value(self.base_alpha + (area_factor * 0.35), 0.1, 1.0)

        return Gaussian(
            x=cx_scaled,
            y=cy_scaled,
            rx=rx_scaled,
            ry=ry_scaled,
            theta=theta,
            r=r,
            g=g,
            b=b,
            a=alpha,
            score=0.0,  # Will be set by importance scoring
            depth=0.5,  # Will be set by layer assignment
        )


class BatchSplatExtractor:
    """Batch processor for multiple images with shared optimization."""

    def __init__(self, max_memory_mb: Optional[int] = None, **extractor_kwargs):
        # Pass max_memory_mb to the OptimizedSplatExtractor
        if max_memory_mb is not None:
            extractor_kwargs['max_memory_mb'] = max_memory_mb
        self.extractor = OptimizedSplatExtractor(**extractor_kwargs)
        self.memory_processor = MemoryEfficientProcessor(max_memory_mb=max_memory_mb) if max_memory_mb else MemoryEfficientProcessor()

    @global_profiler.profile_function("batch_extraction")
    def extract_batch(self, images: List[np.ndarray], n_splats_per_image: List[int]) -> List[List[Gaussian]]:
        """Extract splats from multiple images efficiently."""
        if len(images) != len(n_splats_per_image):
            raise ValueError("Number of images must match number of splat counts")

        results = []

        for i, (image, n_splats) in enumerate(zip(images, n_splats_per_image)):
            logger.info(f"Processing batch image {i+1}/{len(images)}")

            try:
                splats = self.extractor.extract_splats(image, n_splats)
                results.append(splats)

                # Memory check between images
                self.memory_processor.ensure_memory_limit(f"batch_image_{i}")

            except Exception as e:
                logger.error(f"Failed to process batch image {i}: {e}")
                results.append([])  # Empty result for failed image

        return results