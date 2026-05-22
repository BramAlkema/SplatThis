"""
Differentiable renderer for Gaussian splats.

Includes:
- `GaussianRenderer`: pure PyTorch fallback renderer.
- `GsplatRenderer`: optional gsplat-backed renderer (legacy 2D ops).
"""

import logging
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .splat import GaussianSplat

logger = logging.getLogger(__name__)


def _splats_to_numpy_table(splats: List[GaussianSplat]) -> np.ndarray:
    """Convert splats to a compact float32 table [N, 11]."""
    if not splats:
        return np.zeros((0, 11), dtype=np.float32)

    rows = np.empty((len(splats), 11), dtype=np.float32)
    for idx, splat in enumerate(splats):
        raw = splat.to_raw_splat()
        rows[idx, 0] = float(raw.x)
        rows[idx, 1] = float(raw.y)
        rows[idx, 2] = float(raw.sx)
        rows[idx, 3] = float(raw.sy)
        rows[idx, 4] = float(raw.theta)
        rows[idx, 5] = 0.0  # reserved for future parameterization flags
        rows[idx, 6] = float(raw.r)
        rows[idx, 7] = float(raw.g)
        rows[idx, 8] = float(raw.b)
        rows[idx, 9] = float(raw.a)
        rows[idx, 10] = float(raw.importance)
    return rows


def _normalize_backend_name(backend: str) -> Literal["auto", "torch", "gsplat"]:
    normalized = str(backend).strip().lower()
    if normalized not in {"auto", "torch", "gsplat"}:
        raise ValueError(f"Unsupported renderer backend: {backend}")
    return normalized  # type: ignore[return-value]


def _legacy_gsplat_ops():
    """
    Resolve legacy 2D gsplat ops used by GaussianImage/image-gs style pipelines.

    Returns:
        `(project_gaussians_2d_scale_rot, rasterize_gaussians_sum)` or `(None, None)`.
    """
    try:
        from gsplat.project_gaussians_2d_scale_rot import (  # type: ignore
            project_gaussians_2d_scale_rot,
        )
        from gsplat.rasterize_sum import rasterize_gaussians_sum  # type: ignore
        return project_gaussians_2d_scale_rot, rasterize_gaussians_sum
    except Exception:
        return None, None


def can_use_gsplat(device: torch.device) -> bool:
    """Return whether gsplat backend is usable in this runtime."""
    project_fn, rasterize_fn = _legacy_gsplat_ops()
    if project_fn is None or rasterize_fn is None:
        return False
    return device.type == "cuda" and torch.cuda.is_available()


def resolve_renderer_backend(
    backend: str, device: torch.device
) -> Literal["torch", "gsplat"]:
    """
    Resolve backend mode against runtime constraints.

    `auto` picks `gsplat` when available on CUDA, otherwise falls back to `torch`.
    """
    normalized = _normalize_backend_name(backend)
    if normalized == "auto":
        return "gsplat" if can_use_gsplat(device) else "torch"
    if normalized == "gsplat" and not can_use_gsplat(device):
        raise RuntimeError(
            "Requested backend 'gsplat' is unavailable. "
            "Install a gsplat build exposing legacy 2D ops and use CUDA."
        )
    return normalized  # type: ignore[return-value]


def create_renderer(
    backend: str,
    width: int,
    height: int,
    device: torch.device,
    tile_size: int = 16,
    blend_mode: str = "weighted",
    background_color: Optional[Union[torch.Tensor, np.ndarray, List[float], Tuple[float, float, float]]] = None,
    compositing_space: str = "linear",
) -> nn.Module:
    """Factory for renderer backends."""
    resolved = resolve_renderer_backend(backend, device)
    if resolved == "gsplat":
        renderer = GsplatRenderer(
            width=width,
            height=height,
            tile_size=tile_size,
            background_color=background_color,
        )
    else:
        renderer = GaussianRenderer(
            width=width,
            height=height,
            tile_size=tile_size,
            blend_mode=blend_mode,
            background_color=background_color,
            compositing_space=compositing_space,
        )
    return renderer.to(device)


def torch_linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Differentiable linear-RGB -> sRGB (gamma encode), values in [0,1]."""
    x = torch.clamp(x, 0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.clamp(x, min=1e-12).pow(1.0 / 2.4) - 0.055)


def torch_srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB -> linear-RGB (gamma decode), values in [0,1]."""
    x = torch.clamp(x, 0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def torch_linear_rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable linear-sRGB -> OKLab (Björn Ottosson). Input/output [...,3].

    OKLab is perceptually uniform, so L1 distance in OKLab approximates perceived
    color difference far better than L1 in linear or sRGB RGB.
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    # l,m,s are nonneg for in-gamut colors; clamp guards tiny negatives and keeps
    # the cube-root gradient finite near zero.
    l_ = torch.clamp(l, min=1e-8).pow(1.0 / 3.0)
    m_ = torch.clamp(m, min=1e-8).pow(1.0 / 3.0)
    s_ = torch.clamp(s, min=1e-8).pow(1.0 / 3.0)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return torch.stack([L, a, bb], dim=-1)


class GaussianRenderer(nn.Module):
    """
    Differentiable Gaussian splat renderer using PyTorch.

    Simplified renderer optimized for PNG→SVG conversion without CUDA.
    """

    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int = 16,
        blend_mode: str = "weighted",
        background_color: Optional[Union[torch.Tensor, np.ndarray, List[float], Tuple[float, float, float]]] = None,
        enable_tile_culling: bool = True,
        culling_sigma: float = 3.0,
        compositing_space: str = "linear",
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.enable_tile_culling = bool(enable_tile_culling)
        self.culling_sigma = float(max(culling_sigma, 1.0))
        self.blend_mode = str(blend_mode).strip().lower()
        # "linear": composite in linear RGB (physically correct).
        # "srgb": composite in gamma-encoded sRGB, matching how browsers blend
        # overlapping SVG shapes -- so the optimizer's solution matches the
        # rendered SVG. Inputs/outputs stay in linear RGB either way.
        self.compositing_space = str(compositing_space).strip().lower()
        if self.compositing_space not in {"linear", "srgb"}:
            raise ValueError(f"Unsupported compositing space: {compositing_space}")
        if self.blend_mode not in {"weighted", "alpha-over"}:
            raise ValueError(f"Unsupported blend mode: {blend_mode}")
        if background_color is None:
            background = torch.zeros(3, dtype=torch.float32)
        else:
            background = torch.as_tensor(background_color, dtype=torch.float32).flatten()
            if background.numel() != 3:
                raise ValueError("background_color must have exactly 3 values")
            background = torch.clamp(background, 0.0, 1.0)
        self.register_buffer("background", background)
        self._black_background = bool(torch.max(torch.abs(background)).item() <= 1e-8)

        # Pre-compute pixel coordinates
        self.register_buffer('pixel_coords', self._create_pixel_coords())

    def _create_pixel_coords(self) -> torch.Tensor:
        """Create meshgrid of pixel coordinates."""
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing='ij'
        )
        # Stack to [H, W, 2] format
        coords = torch.stack([x_coords, y_coords], dim=-1)
        return coords

    def forward(self, splats_tensor: torch.Tensor) -> torch.Tensor:
        """
        Render splats to image.

        Args:
            splats_tensor: [N, 11] tensor where each row is:
                [mu_x, mu_y, sx, sy, theta, reserved, r, g, b, alpha, importance]

        Returns:
            Rendered image [H, W, 3]
        """
        if len(splats_tensor) == 0:
            return self.background.view(1, 1, 3).expand(self.height, self.width, 3).to(
                splats_tensor.device
            )

        # Extract parameters
        mu = splats_tensor[:, :2]  # [N, 2]
        sx = torch.clamp(splats_tensor[:, 2], min=1e-4)  # [N]
        sy = torch.clamp(splats_tensor[:, 3], min=1e-4)  # [N]
        theta = splats_tensor[:, 4]  # [N]
        colors = splats_tensor[:, 6:9]  # [N, 3]
        alphas = splats_tensor[:, 9]  # [N]
        importance = splats_tensor[:, 10]  # [N]

        if self.compositing_space == "srgb":
            # Composite in gamma-encoded sRGB to mirror browser SVG blending,
            # then decode back to linear so all external interfaces are unchanged.
            colors = torch_linear_to_srgb(colors)
            saved_bg = self.background
            try:
                self.background = torch_linear_to_srgb(saved_bg)
                rendered = self._render_tiled(mu, sx, sy, theta, colors, alphas, importance)
            finally:
                self.background = saved_bg
            return torch_srgb_to_linear(rendered)

        # Render using tile-based approach for memory efficiency
        return self._render_tiled(mu, sx, sy, theta, colors, alphas, importance)

    def _render_tiled(
        self,
        mu: torch.Tensor,
        sx: torch.Tensor,
        sy: torch.Tensor,
        theta: torch.Tensor,
        colors: torch.Tensor,
        alphas: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        """Render using tiles to manage memory usage."""
        device = mu.device
        background_tile = self.background.view(1, 1, 3).to(device)
        output = background_tile.expand(self.height, self.width, 3).clone()

        # Higher-importance splats are composited later (front-most).
        order = torch.argsort(importance, descending=False)
        mu = mu[order]
        sx = sx[order]
        sy = sy[order]
        theta = theta[order]
        colors = colors[order]
        alphas = alphas[order]

        tile_bins: Optional[List[Optional[torch.Tensor]]] = None
        tiles_x = 0
        if self.enable_tile_culling:
            bins, tiles_x = self._build_tile_splat_bins(mu=mu, sx=sx, sy=sy)
            tile_bins = []
            for indices in bins:
                if not indices:
                    tile_bins.append(None)
                    continue
                tile_bins.append(torch.tensor(indices, dtype=torch.long, device=device))

        # Process in tiles
        for tile_y in range(0, self.height, self.tile_size):
            for tile_x in range(0, self.width, self.tile_size):
                # Tile bounds
                y1 = tile_y
                y2 = min(tile_y + self.tile_size, self.height)
                x1 = tile_x
                x2 = min(tile_x + self.tile_size, self.width)

                # Get tile pixel coordinates
                tile_coords = self.pixel_coords[y1:y2, x1:x2]  # [tile_h, tile_w, 2]

                tile_mu = mu
                tile_sx = sx
                tile_sy = sy
                tile_theta = theta
                tile_colors = colors
                tile_alphas = alphas
                if tile_bins is not None:
                    tile_ix = (tile_y // self.tile_size) * tiles_x + (tile_x // self.tile_size)
                    active = tile_bins[tile_ix]
                    if active is None:
                        continue
                    tile_mu = mu.index_select(0, active)
                    tile_sx = sx.index_select(0, active)
                    tile_sy = sy.index_select(0, active)
                    tile_theta = theta.index_select(0, active)
                    tile_colors = colors.index_select(0, active)
                    tile_alphas = alphas.index_select(0, active)

                # Render tile
                tile_output = self._render_tile(
                    tile_coords,
                    tile_mu,
                    tile_sx,
                    tile_sy,
                    tile_theta,
                    tile_colors,
                    tile_alphas,
                )
                output[y1:y2, x1:x2] = tile_output

        return output

    def _build_tile_splat_bins(
        self,
        mu: torch.Tensor,
        sx: torch.Tensor,
        sy: torch.Tensor,
    ) -> Tuple[List[List[int]], int]:
        """Assign splats to tiles using conservative axis-aligned 3σ footprint."""
        tiles_x = (self.width + self.tile_size - 1) // self.tile_size
        tiles_y = (self.height + self.tile_size - 1) // self.tile_size
        bins: List[List[int]] = [[] for _ in range(tiles_x * tiles_y)]
        if len(mu) == 0:
            return bins, tiles_x

        with torch.no_grad():
            radius = self.culling_sigma * torch.maximum(sx, sy)
            x_min = torch.floor((mu[:, 0] - radius) / float(self.tile_size)).to(torch.int64)
            x_max = torch.floor((mu[:, 0] + radius) / float(self.tile_size)).to(torch.int64)
            y_min = torch.floor((mu[:, 1] - radius) / float(self.tile_size)).to(torch.int64)
            y_max = torch.floor((mu[:, 1] + radius) / float(self.tile_size)).to(torch.int64)

            x_min = torch.clamp(x_min, 0, max(tiles_x - 1, 0))
            x_max = torch.clamp(x_max, 0, max(tiles_x - 1, 0))
            y_min = torch.clamp(y_min, 0, max(tiles_y - 1, 0))
            y_max = torch.clamp(y_max, 0, max(tiles_y - 1, 0))

            x_min_np = x_min.detach().cpu().numpy()
            x_max_np = x_max.detach().cpu().numpy()
            y_min_np = y_min.detach().cpu().numpy()
            y_max_np = y_max.detach().cpu().numpy()

        for splat_idx in range(len(mu)):
            tx0 = int(x_min_np[splat_idx])
            tx1 = int(x_max_np[splat_idx])
            ty0 = int(y_min_np[splat_idx])
            ty1 = int(y_max_np[splat_idx])
            for ty in range(ty0, ty1 + 1):
                base = ty * tiles_x
                for tx in range(tx0, tx1 + 1):
                    bins[base + tx].append(splat_idx)

        return bins, tiles_x

    def _render_tile(
        self,
        tile_coords: torch.Tensor,
        mu: torch.Tensor,
        sx: torch.Tensor,
        sy: torch.Tensor,
        theta: torch.Tensor,
        colors: torch.Tensor,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Render a single tile."""
        tile_h, tile_w = tile_coords.shape[:2]
        num_splats = len(mu)
        device = mu.device

        if num_splats == 0:
            return self.background.view(1, 1, 3).expand(tile_h, tile_w, 3).to(device)

        # Reshape tile coordinates for broadcasting: [tile_h, tile_w, 1, 2]
        coords = tile_coords.unsqueeze(2)  # [tile_h, tile_w, 1, 2]

        # Compute distances from each pixel to each splat center
        # mu: [1, 1, N, 2], coords: [tile_h, tile_w, 1, 2]
        delta = coords - mu.view(1, 1, num_splats, 2)  # [tile_h, tile_w, N, 2]

        # Compute Gaussian weights
        weights = self._compute_gaussian_weights(delta, sx, sy, theta)  # [tile_h, tile_w, N]

        # Convert gaussian responses to per-layer alpha and composite front-to-back.
        if self.blend_mode == "weighted":
            weighted = weights * alphas.view(1, 1, -1)
            total_weight = torch.sum(weighted, dim=-1, keepdim=True)
            weighted_colors = weighted.unsqueeze(-1) * colors.view(1, 1, num_splats, 3)
            color_sum = torch.sum(weighted_colors, dim=2)
            normalized = color_sum / torch.clamp(total_weight, min=1e-8)
            if self._black_background:
                # Keep legacy weighted behavior for black background compatibility.
                return torch.clamp(normalized, 0.0, 1.0)
            coverage = torch.clamp(total_weight, min=0.0, max=1.0)
            blended = coverage * normalized + (1.0 - coverage) * self.background.view(1, 1, 3).to(
                device
            )
            return torch.clamp(blended, 0.0, 1.0)

        # Alpha-over compositing path.
        density = torch.clamp(weights * alphas.view(1, 1, -1), min=0.0)
        alpha_layers = 1.0 - torch.exp(-density)
        one_minus_alpha = torch.clamp(1.0 - alpha_layers, min=1e-6, max=1.0)
        transmittance_prefix = torch.cumprod(
            torch.cat([torch.ones(tile_h, tile_w, 1, device=device), one_minus_alpha], dim=-1),
            dim=-1,
        )[..., :-1]
        contributions = transmittance_prefix * alpha_layers
        output_colors = torch.sum(
            contributions.unsqueeze(-1) * colors.view(1, 1, num_splats, 3), dim=2
        )
        remaining_transmittance = torch.prod(one_minus_alpha, dim=-1, keepdim=True)
        output_colors = output_colors + remaining_transmittance * self.background.view(1, 1, 3)
        return torch.clamp(output_colors, 0.0, 1.0)

    def _compute_gaussian_weights(
        self,
        delta: torch.Tensor,
        sx: torch.Tensor,
        sy: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian weights for pixel distances.

        Args:
            delta: [tile_h, tile_w, N, 2] - distance vectors
            sx: [N] major-axis scales
            sy: [N] minor-axis scales
            theta: [N] in-plane rotation

        Returns:
            [tile_h, tile_w, N] - Gaussian weights
        """
        dx = delta[..., 0]
        dy = delta[..., 1]
        cos_t = torch.cos(theta).view(1, 1, -1)
        sin_t = torch.sin(theta).view(1, 1, -1)

        # Rotate local coordinates into each splat's principal frame.
        u = cos_t * dx + sin_t * dy
        v = -sin_t * dx + cos_t * dy

        inv_sx2 = 1.0 / torch.clamp(sx, min=1e-4).view(1, 1, -1).pow(2)
        inv_sy2 = 1.0 / torch.clamp(sy, min=1e-4).view(1, 1, -1).pow(2)
        quadratic = u * u * inv_sx2 + v * v * inv_sy2

        weights = torch.exp(-0.5 * quadratic)
        return weights


class GsplatRenderer(nn.Module):
    """
    Optional gsplat-backed 2D renderer.

    This adapter targets legacy 2D ops commonly used by GaussianImage/image-gs:
    `project_gaussians_2d_scale_rot` + `rasterize_gaussians_sum`.
    """

    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int = 16,
        background_color: Optional[Union[torch.Tensor, np.ndarray, List[float], Tuple[float, float, float]]] = None,
    ):
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(tile_size)
        self.project_gaussians_2d_scale_rot, self.rasterize_gaussians_sum = _legacy_gsplat_ops()
        if self.project_gaussians_2d_scale_rot is None or self.rasterize_gaussians_sum is None:
            raise RuntimeError(
                "gsplat backend requested, but legacy 2D ops are unavailable "
                "(project_gaussians_2d_scale_rot / rasterize_gaussians_sum)."
            )
        if background_color is None:
            background = torch.zeros(3, dtype=torch.float32)
        else:
            background = torch.as_tensor(background_color, dtype=torch.float32).flatten()
            if background.numel() != 3:
                raise ValueError("background_color must have exactly 3 values")
            background = torch.clamp(background, 0.0, 1.0)
        self.register_buffer("background", background)

    def forward(self, splats_tensor: torch.Tensor) -> torch.Tensor:
        if len(splats_tensor) == 0:
            return self.background.view(1, 1, 3).expand(self.height, self.width, 3).to(
                splats_tensor.device
            )

        if splats_tensor.device.type != "cuda":
            raise RuntimeError("GsplatRenderer requires CUDA tensors.")

        mu = splats_tensor[:, :2]
        scales = torch.clamp(splats_tensor[:, 2:4], min=1e-4)
        rotation = torch.remainder(splats_tensor[:, 4:5], 2.0 * torch.pi)
        colors = splats_tensor[:, 6:9]
        opacities = splats_tensor[:, 9:10]

        means = self._pixel_to_ndc(mu)

        tile_bounds = (
            (self.width + self.tile_size - 1) // self.tile_size,
            (self.height + self.tile_size - 1) // self.tile_size,
            1,
        )

        xys, depths, radii, conics, num_tiles_hit = self.project_gaussians_2d_scale_rot(
            means, scales, rotation, self.height, self.width, tile_bounds
        )
        out_img = self.rasterize_gaussians_sum(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            colors,
            opacities,
            self.height,
            self.width,
            self.tile_size,
            self.tile_size,
            background=self.background.to(splats_tensor.device),
            return_alpha=False,
        )
        return torch.clamp(out_img, 0.0, 1.0)

    def _pixel_to_ndc(self, mu: torch.Tensor) -> torch.Tensor:
        width_denom = max(self.width - 1, 1)
        height_denom = max(self.height - 1, 1)
        x_ndc = (mu[:, 0] / float(width_denom)) * 2.0 - 1.0
        y_ndc = (mu[:, 1] / float(height_denom)) * 2.0 - 1.0
        return torch.stack([x_ndc, y_ndc], dim=-1)

def splats_to_tensor(
    splats: List[GaussianSplat], device: Optional[Union[torch.device, str]] = None
) -> torch.Tensor:
    """
    Convert list of splats to tensor format.

    Args:
        splats: List of GaussianSplat objects

    Returns:
        [N, 11] tensor with splat parameters:
        [x, y, sx, sy, theta, reserved, r, g, b, alpha, importance]
    """
    table = _splats_to_numpy_table(splats)
    tensor = torch.from_numpy(table)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_splats(tensor: torch.Tensor) -> List[GaussianSplat]:
    """
    Convert tensor format back to list of splats.

    Args:
        tensor: [N, 11] tensor with splat parameters
            [x, y, sx, sy, theta, reserved, r, g, b, alpha, importance]

    Returns:
        List of GaussianSplat objects
    """
    splats = []
    for i in range(len(tensor)):
        row = tensor[i].detach().cpu().numpy()
        theta = float(np.remainder(row[4], 2.0 * np.pi))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        scales_sq = np.diag(
            np.array(
                [
                    max(float(row[2]), 1e-4) ** 2,
                    max(float(row[3]), 1e-4) ** 2,
                ],
                dtype=np.float32,
            )
        )
        sigma = rotation @ scales_sq @ rotation.T
        splat = GaussianSplat(
            mu=row[:2],
            sigma=sigma,
            color=row[6:9],
            alpha=row[9],
            importance=row[10]
        )
        splats.append(splat)
    return splats


class SimpleLoss(nn.Module):
    """
    Simple loss function for PNG→SVG conversion.

    Combines MSE, total variation, and area penalty.
    """

    def __init__(self, mse_weight: float = 1.0, tv_weight: float = 0.1,
                 area_weight: float = 0.01):
        super().__init__()
        self.mse_weight = mse_weight
        self.tv_weight = tv_weight
        self.area_weight = area_weight

    def forward(self, rendered: torch.Tensor, target: torch.Tensor,
                splats_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            rendered: Rendered image [H, W, 3]
            target: Target image [H, W, 3]
            splats_tensor: Splat parameters [N, 11]

        Returns:
            Combined loss scalar
        """
        # MSE loss
        mse_loss = torch.mean((rendered - target) ** 2)

        # Total variation loss on alpha (encourage smoothness)
        if len(splats_tensor) > 0:
            alphas = splats_tensor[:, 9]  # [N]
            tv_loss = torch.mean(torch.abs(alphas[1:] - alphas[:-1]))
        else:
            tv_loss = torch.tensor(0.0, device=rendered.device)

        # Area penalty (encourage smaller splats)
        if len(splats_tensor) > 0:
            # Approximate area from determinant of covariance
            areas = torch.clamp(splats_tensor[:, 2], min=1e-4) * torch.clamp(
                splats_tensor[:, 3], min=1e-4
            )
            area_loss = torch.mean(areas)
        else:
            area_loss = torch.tensor(0.0, device=rendered.device)

        # Combined loss
        total_loss = (self.mse_weight * mse_loss +
                     self.tv_weight * tv_loss +
                     self.area_weight * area_loss)

        return total_loss


class L1SSIMLoss(nn.Module):
    """
    Baseline reconstruction loss for Phase 1.

    loss = l1_weight * L1 + ssim_weight * (1 - SSIM)
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.2, color_space: str = "linear"):
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.ssim_weight = float(ssim_weight)
        # "linear": L1/SSIM on linear RGB. "oklab": transform both images to
        # perceptually-uniform OKLab first, so the optimizer prioritizes errors
        # the eye actually notices (chroma/lightness) rather than linear-RGB MSE.
        self.color_space = str(color_space).strip().lower()
        if self.color_space not in {"linear", "oklab"}:
            raise ValueError(f"Unsupported loss color space: {color_space}")
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.color_space == "oklab":
            rendered = torch_linear_rgb_to_oklab(rendered)
            target = torch_linear_rgb_to_oklab(target)
        l1_loss = torch.mean(torch.abs(rendered - target))
        ssim = self._global_ssim(rendered, target)
        dssim = 1.0 - ssim
        return self.l1_weight * l1_loss + self.ssim_weight * dssim

    def _global_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a simple global SSIM over spatial dimensions per channel.

        Returns scalar in [-1, 1] (clamped).
        """
        mu_x = torch.mean(x, dim=(0, 1))
        mu_y = torch.mean(y, dim=(0, 1))

        x_centered = x - mu_x.view(1, 1, -1)
        y_centered = y - mu_y.view(1, 1, -1)

        sigma_x = torch.mean(x_centered * x_centered, dim=(0, 1))
        sigma_y = torch.mean(y_centered * y_centered, dim=(0, 1))
        sigma_xy = torch.mean(x_centered * y_centered, dim=(0, 1))

        numerator = (2.0 * mu_x * mu_y + self.c1) * (2.0 * sigma_xy + self.c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + self.c1) * (sigma_x + sigma_y + self.c2)
        ssim_per_channel = numerator / torch.clamp(denominator, min=1e-8)
        ssim = torch.mean(ssim_per_channel)
        return torch.clamp(ssim, min=-1.0, max=1.0)


def render_splats_numpy(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    background_linear_rgb: Optional[np.ndarray] = None,
    footprint_sigma: float = 3.0,
) -> np.ndarray:
    """
    Simple NumPy renderer for validation and debugging.

    Args:
        splats: List of splats
        width: Image width
        height: Image height

    Returns:
        Rendered image [H, W, 3]
    """
    if background_linear_rgb is None:
        background = np.zeros(3, dtype=np.float32)
    else:
        background = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
        if background.size != 3:
            raise ValueError("background_linear_rgb must have 3 values")
        background = np.clip(background, 0.0, 1.0).astype(np.float32)

    if not splats:
        return np.broadcast_to(background.reshape(1, 1, 3), (height, width, 3)).astype(np.float32).copy()

    # Create output image and transmittance for alpha-over.
    output = np.zeros((height, width, 3), dtype=np.float32)
    transmittance = np.ones((height, width), dtype=np.float32)

    # Composite low-importance -> high-importance.
    ordered = sorted(splats, key=lambda s: float(s.importance))

    for splat in ordered:
        raw = splat.to_raw_splat()
        cx = float(np.clip(raw.x, 0.0, width - 1.0))
        cy = float(np.clip(raw.y, 0.0, height - 1.0))
        sx = max(float(raw.sx), 1e-4)
        sy = max(float(raw.sy), 1e-4)
        theta = float(raw.theta)

        radius_x = max(1, int(np.ceil(max(1.0, footprint_sigma) * sx)))
        radius_y = max(1, int(np.ceil(max(1.0, footprint_sigma) * sy)))
        x0 = max(0, int(np.floor(cx - radius_x)))
        x1 = min(width, int(np.ceil(cx + radius_x + 1)))
        y0 = max(0, int(np.floor(cy - radius_y)))
        y1 = min(height, int(np.ceil(cy + radius_y + 1)))
        if x0 >= x1 or y0 >= y1:
            continue

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        dx = gx - cx
        dy = gy - cy
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        u = cos_t * dx + sin_t * dy
        v = -sin_t * dx + cos_t * dy
        quadratic = (u * u) / (sx * sx) + (v * v) / (sy * sy)

        density = np.clip(float(raw.a), 0.0, 1.0) * np.exp(-0.5 * quadratic)
        layer_alpha = 1.0 - np.exp(-density)
        local_trans = transmittance[y0:y1, x0:x1]
        contrib = local_trans * layer_alpha
        color = np.array([raw.r, raw.g, raw.b], dtype=np.float32).reshape(1, 1, 3)

        output[y0:y1, x0:x1] += contrib[..., None] * color
        transmittance[y0:y1, x0:x1] = local_trans * (1.0 - layer_alpha)

    output += transmittance[..., None] * background.reshape(1, 1, 3)
    return np.clip(output, 0.0, 1.0).astype(np.float32)
