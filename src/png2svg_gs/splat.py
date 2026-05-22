"""
Gaussian Splat data structure and utilities.

Streamlined splat representation optimized for PNG→SVG conversion.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

RAW_SPLAT_SCHEMA_VERSION = "png2splat.raw/1"
MIN_SCALE = 1e-4


@dataclass
class RawSplat:
    """
    Canonical raw splat schema used as the pipeline interchange format.

    Coordinates are image-space pixels.
    """

    x: float
    y: float
    sx: float
    sy: float
    theta: float
    r: float
    g: float
    b: float
    a: float
    importance: float = 0.0
    source: Optional[str] = None
    score: Optional[float] = None
    layer: Optional[int] = None

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)
        self.sx = float(self.sx)
        self.sy = float(self.sy)
        self.theta = float(self.theta)
        self.r = float(self.r)
        self.g = float(self.g)
        self.b = float(self.b)
        self.a = float(self.a)
        self.importance = float(self.importance)
        if self.score is not None:
            self.score = float(self.score)
        if self.layer is not None:
            self.layer = int(self.layer)
        self.validate()

    def validate(self) -> None:
        """Validate and normalize to safe numeric bounds."""
        if self.sx < MIN_SCALE:
            raise ValueError(f"sx must be >= {MIN_SCALE}, got {self.sx}")
        if self.sy < MIN_SCALE:
            raise ValueError(f"sy must be >= {MIN_SCALE}, got {self.sy}")
        if not np.isfinite(self.theta):
            raise ValueError(f"theta must be finite, got {self.theta}")

        self.r = float(np.clip(self.r, 0.0, 1.0))
        self.g = float(np.clip(self.g, 0.0, 1.0))
        self.b = float(np.clip(self.b, 0.0, 1.0))
        self.a = float(np.clip(self.a, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize raw splat for JSON output."""
        data: Dict[str, Any] = {
            "x": self.x,
            "y": self.y,
            "sx": self.sx,
            "sy": self.sy,
            "theta": self.theta,
            "r": self.r,
            "g": self.g,
            "b": self.b,
            "a": self.a,
            "importance": self.importance,
        }
        if self.source is not None:
            data["source"] = self.source
        if self.score is not None:
            data["score"] = self.score
        if self.layer is not None:
            data["layer"] = self.layer
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawSplat":
        """Create validated RawSplat from dictionary."""
        required = ["x", "y", "sx", "sy", "theta", "r", "g", "b", "a"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing raw splat fields: {', '.join(missing)}")
        return cls(
            x=data["x"],
            y=data["y"],
            sx=data["sx"],
            sy=data["sy"],
            theta=data["theta"],
            r=data["r"],
            g=data["g"],
            b=data["b"],
            a=data["a"],
            importance=data.get("importance", 0.0),
            source=data.get("source"),
            score=data.get("score"),
            layer=data.get("layer"),
        )


@dataclass
class GaussianSplat:
    """
    Anisotropic 2D Gaussian splat for PNG→SVG conversion.

    Simplified from AdaptiveGaussian2D for practical use.
    """
    # Core parameters
    mu: np.ndarray      # mean μᵢ = (x, y) in image coordinates
    sigma: np.ndarray   # covariance Σᵢ ∈ ℝ²ˣ² (positive-definite)
    color: np.ndarray   # color cᵢ = (r, g, b) in [0,1]
    alpha: float        # opacity αᵢ ∈ [0,1]

    # Optional metadata
    importance: float = 0.0    # For LOD and pruning
    _raw_cache: Optional[RawSplat] = field(default=None, init=False, repr=False, compare=False)
    _raw_cache_key: Optional[Tuple[float, ...]] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate and ensure proper types."""
        self.mu = np.array(self.mu, dtype=np.float32)
        self.sigma = np.array(self.sigma, dtype=np.float32)
        self.color = np.array(self.color, dtype=np.float32)
        self.alpha = float(self.alpha)
        self.importance = float(self.importance)

        self._validate()

    def _validate(self):
        """Validate splat parameters."""
        if self.mu.shape != (2,):
            raise ValueError(f"mu must be 2D, got shape {self.mu.shape}")
        if self.sigma.shape != (2, 2):
            raise ValueError(f"sigma must be 2x2, got shape {self.sigma.shape}")
        if self.color.shape[0] < 3:
            raise ValueError(f"color must have at least 3 components, got {len(self.color)}")
        if not (0 <= self.alpha <= 1):
            raise ValueError(f"alpha must be in [0,1], got {self.alpha}")

        # Check positive definiteness
        if not is_positive_definite(self.sigma):
            logger.warning("Covariance matrix is not positive definite, clamping")
            self.sigma = clamp_positive_definite(self.sigma)

    def evaluate_at(self, point: np.ndarray) -> float:
        """
        Evaluate Gaussian at given point.

        Args:
            point: [x, y] coordinates in image space

        Returns:
            Gaussian value at point
        """
        delta = point - self.mu
        try:
            sigma_inv = np.linalg.inv(self.sigma)
            quadratic_form = delta.T @ sigma_inv @ delta
            return np.exp(-0.5 * quadratic_form)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix in evaluation")
            return 0.0

    def eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of covariance matrix.

        Returns:
            eigenvalues, eigenvectors for SVG transform generation
        """
        try:
            eigenvals, eigenvecs = np.linalg.eigh(self.sigma)
            # Sort by eigenvalue magnitude (descending)
            idx = np.argsort(eigenvals)[::-1]
            return eigenvals[idx], eigenvecs[:, idx]
        except np.linalg.LinAlgError:
            logger.warning("Failed eigendecomposition, using identity")
            return np.array([1.0, 1.0]), np.eye(2)

    def _raw_state_key(self) -> Tuple[float, ...]:
        """Return compact key for raw parameter cache invalidation."""
        return (
            float(self.mu[0]),
            float(self.mu[1]),
            float(self.sigma[0, 0]),
            float(self.sigma[0, 1]),
            float(self.sigma[1, 0]),
            float(self.sigma[1, 1]),
            float(self.color[0]),
            float(self.color[1]),
            float(self.color[2]),
            float(self.alpha),
            float(self.importance),
        )

    def _principal_params_from_sigma(self) -> Tuple[float, float, float]:
        """
        Convert 2x2 covariance to (sx, sy, theta) without eigendecomposition.

        Uses the closed-form eigenvalues/eigenvector angle of a symmetric 2x2 matrix.
        """
        a = float(self.sigma[0, 0])
        b = float(0.5 * (self.sigma[0, 1] + self.sigma[1, 0]))
        c = float(self.sigma[1, 1])

        trace = a + c
        discriminant = max((a - c) * (a - c) + 4.0 * b * b, 0.0)
        root = float(np.sqrt(discriminant))

        lambda_major = max(0.5 * (trace + root), MIN_SCALE)
        lambda_minor = max(0.5 * (trace - root), MIN_SCALE)
        theta = float(0.5 * np.arctan2(2.0 * b, a - c))

        sx = float(np.sqrt(lambda_major))
        sy = float(np.sqrt(lambda_minor))
        return sx, sy, theta

    def compute_3sigma_radius(self) -> float:
        """Compute 3σ radius for spatial binning."""
        sx, _, _ = self._principal_params_from_sigma()
        return 3.0 * float(max(sx, 0.0))

    def area(self) -> float:
        """Compute ellipse area (π * sqrt(det(Σ)))."""
        det = np.linalg.det(self.sigma)
        return np.pi * np.sqrt(max(det, 1e-8))  # Prevent negative determinant

    def copy(self) -> 'GaussianSplat':
        """Create deep copy."""
        cloned = GaussianSplat(
            mu=self.mu.copy(),
            sigma=self.sigma.copy(),
            color=self.color.copy(),
            alpha=self.alpha,
            importance=self.importance
        )
        if self._raw_cache is not None:
            cloned._raw_cache = RawSplat(
                x=float(self._raw_cache.x),
                y=float(self._raw_cache.y),
                sx=float(self._raw_cache.sx),
                sy=float(self._raw_cache.sy),
                theta=float(self._raw_cache.theta),
                r=float(self._raw_cache.r),
                g=float(self._raw_cache.g),
                b=float(self._raw_cache.b),
                a=float(self._raw_cache.a),
                importance=float(self._raw_cache.importance),
                source=self._raw_cache.source,
                score=self._raw_cache.score,
                layer=self._raw_cache.layer,
            )
            cloned._raw_cache_key = self._raw_cache_key
        return cloned

    def __str__(self) -> str:
        eigenvals, _ = self.eigendecomposition()
        return (f"GaussianSplat(mu={self.mu}, eigenvals={eigenvals}, "
                f"alpha={self.alpha:.3f}, area={self.area():.3f})")

    def to_raw_splat(self) -> RawSplat:
        """Convert Gaussian representation to canonical raw schema."""
        cache_key = self._raw_state_key()
        if self._raw_cache is not None and self._raw_cache_key == cache_key:
            return self._raw_cache

        sx, sy, theta = self._principal_params_from_sigma()
        raw = RawSplat(
            x=float(self.mu[0]),
            y=float(self.mu[1]),
            sx=sx,
            sy=sy,
            theta=theta,
            r=float(self.color[0]),
            g=float(self.color[1]),
            b=float(self.color[2]),
            a=float(self.alpha),
            importance=float(self.importance),
        )
        self._raw_cache = raw
        self._raw_cache_key = cache_key
        return raw

    @classmethod
    def from_raw_splat(cls, raw: RawSplat) -> "GaussianSplat":
        """Create Gaussian splat from canonical raw schema."""
        theta = float(raw.theta)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        scales_sq = np.diag(
            np.array([max(raw.sx, MIN_SCALE) ** 2, max(raw.sy, MIN_SCALE) ** 2], dtype=np.float32)
        )
        sigma = rotation @ scales_sq @ rotation.T

        splat = cls(
            mu=np.array([raw.x, raw.y], dtype=np.float32),
            sigma=sigma.astype(np.float32),
            color=np.array([raw.r, raw.g, raw.b], dtype=np.float32),
            alpha=float(raw.a),
            importance=float(raw.importance),
        )
        splat._raw_cache = RawSplat(
            x=float(raw.x),
            y=float(raw.y),
            sx=float(raw.sx),
            sy=float(raw.sy),
            theta=float(raw.theta),
            r=float(raw.r),
            g=float(raw.g),
            b=float(raw.b),
            a=float(raw.a),
            importance=float(raw.importance),
            source=raw.source,
            score=raw.score,
            layer=raw.layer,
        )
        splat._raw_cache_key = splat._raw_state_key()
        return splat

    @classmethod
    def from_raw_dict(cls, data: Dict[str, Any]) -> "GaussianSplat":
        """Convenience constructor from raw splat dictionary."""
        return cls.from_raw_splat(RawSplat.from_dict(data))


def create_isotropic_splat(center: np.ndarray, sigma: float,
                          color: np.ndarray, alpha: float = 0.8) -> GaussianSplat:
    """
    Create isotropic (circular) Gaussian splat.

    Args:
        center: [x, y] position in image coordinates
        sigma: Standard deviation (radius)
        color: RGB color in [0,1]
        alpha: Opacity

    Returns:
        Isotropic GaussianSplat
    """
    covariance = np.eye(2, dtype=np.float32) * (sigma ** 2)
    return GaussianSplat(
        mu=np.array(center, dtype=np.float32),
        sigma=covariance,
        color=np.array(color, dtype=np.float32),
        alpha=alpha
    )


def create_anisotropic_splat(center: np.ndarray, eigenvals: np.ndarray,
                           eigenvecs: np.ndarray, color: np.ndarray,
                           alpha: float = 0.8) -> GaussianSplat:
    """
    Create anisotropic Gaussian splat from eigendecomposition.

    Args:
        center: [x, y] position
        eigenvals: [λ1, λ2] eigenvalues
        eigenvecs: 2x2 eigenvector matrix
        color: RGB color in [0,1]
        alpha: Opacity

    Returns:
        Anisotropic GaussianSplat
    """
    # Reconstruct covariance: Σ = V * Λ * V^T
    Lambda = np.diag(eigenvals)
    covariance = eigenvecs @ Lambda @ eigenvecs.T

    return GaussianSplat(
        mu=np.array(center, dtype=np.float32),
        sigma=covariance.astype(np.float32),
        color=np.array(color, dtype=np.float32),
        alpha=alpha
    )


def is_positive_definite(matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
    """Check if matrix is positive definite."""
    try:
        eigenvals = np.linalg.eigvals(matrix)
        return np.all(eigenvals > tolerance)
    except np.linalg.LinAlgError:
        return False


def clamp_positive_definite(matrix: np.ndarray, min_eigenval: float = 1e-6) -> np.ndarray:
    """
    Clamp matrix to be positive definite.

    Args:
        matrix: 2x2 matrix to clamp
        min_eigenval: Minimum eigenvalue

    Returns:
        Clamped positive definite matrix
    """
    try:
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        # Clamp eigenvalues to minimum
        eigenvals = np.maximum(eigenvals, min_eigenval)
        # Reconstruct matrix
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    except np.linalg.LinAlgError:
        # Fallback to identity with minimum eigenvalue
        return np.eye(2) * min_eigenval


def merge_nearby_splats(splats: list[GaussianSplat],
                       distance_threshold: float = 2.0) -> list[GaussianSplat]:
    """
    Merge splats that are very close together.

    Args:
        splats: List of splats to merge
        distance_threshold: Maximum distance for merging

    Returns:
        List of merged splats
    """
    if len(splats) <= 1:
        return splats

    merged = []
    used = set()

    for i, splat1 in enumerate(splats):
        if i in used:
            continue

        # Find nearby splats
        cluster = [splat1]
        used.add(i)

        for j, splat2 in enumerate(splats[i+1:], start=i+1):
            if j in used:
                continue

            distance = np.linalg.norm(splat1.mu - splat2.mu)
            if distance < distance_threshold:
                cluster.append(splat2)
                used.add(j)

        # Merge cluster
        if len(cluster) == 1:
            merged.append(cluster[0])
        else:
            merged_splat = _merge_splat_cluster(cluster)
            merged.append(merged_splat)

    return merged


def _merge_splat_cluster(splats: list[GaussianSplat]) -> GaussianSplat:
    """Merge a cluster of splats into a single splat."""
    # Weight by alpha (importance)
    weights = np.array([s.alpha for s in splats])
    weights = weights / np.sum(weights)

    # Weighted average position
    mu = np.sum([w * s.mu for w, s in zip(weights, splats)], axis=0)

    # Weighted average color
    color = np.sum([w * s.color for w, s in zip(weights, splats)], axis=0)

    # Sum alpha (with clipping)
    alpha = min(1.0, np.sum([s.alpha for s in splats]))

    # Average covariance (simple approach)
    sigma = np.mean([s.sigma for s in splats], axis=0)
    sigma = clamp_positive_definite(sigma)

    # Max importance
    importance = max(s.importance for s in splats)

    return GaussianSplat(
        mu=mu,
        sigma=sigma,
        color=color,
        alpha=alpha,
        importance=importance
    )
