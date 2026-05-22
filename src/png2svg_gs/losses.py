#!/usr/bin/env python3
"""
Advanced Loss Functions for PNG→SVG Quality Optimization

Implements perceptual and content-aware loss functions that go beyond simple MSE
to optimize for human visual perception and structural preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import torchvision.transforms as transforms
from torchvision.models import vgg16


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for visually-aligned optimization.
    Uses pre-trained VGG features to compare images in perceptual space.
    """

    def __init__(self,
                 feature_layers: list = [3, 8, 15, 22],
                 weights: list = [1.0, 1.0, 1.0, 1.0],
                 device: str = 'cpu'):
        super().__init__()

        self.device = device
        self.feature_layers = feature_layers
        self.weights = weights

        # Load VGG16 weights with offline-safe fallback.
        try:
            from torchvision.models import VGG16_Weights  # type: ignore

            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        except Exception:
            try:
                vgg = vgg16(pretrained=True).features.to(device).eval()
            except Exception:
                vgg = vgg16(weights=None).features.to(device).eval()

        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

        # Normalization for ImageNet pre-trained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def extract_features(self, x: torch.Tensor) -> list:
        """Extract multi-layer VGG features."""
        # Ensure 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 4:  # RGBA
            x = x[:, :3]  # Drop alpha

        # VGG pooling collapses very small inputs; upsample tiny tensors.
        if x.shape[-2] < 32 or x.shape[-1] < 32:
            x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

        # Normalize
        x = self.normalize(x)

        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)

        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between predictions and targets."""
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        loss = 0.0
        for i, (pred_feat, target_feat, weight) in enumerate(
            zip(pred_features, target_features, self.weights)
        ):
            loss += weight * F.mse_loss(pred_feat, target_feat)

        return loss


class EdgePreservingLoss(nn.Module):
    """
    Edge-preserving loss that penalizes smoothing of important edges.
    Uses Sobel gradients to detect and preserve structural boundaries.
    """

    def __init__(self, edge_weight: float = 1.0, device: str = 'cpu'):
        super().__init__()
        self.edge_weight = edge_weight
        self.device = device

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel operators."""
        # Convert to grayscale if needed
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge-preserving loss."""
        pred_edges = self.compute_gradients(pred)
        target_edges = self.compute_gradients(target)

        return self.edge_weight * F.mse_loss(pred_edges, target_edges)


class ContentAdaptiveLoss(nn.Module):
    """
    Content-adaptive loss that weights different regions based on visual importance.
    Uses saliency and texture complexity to focus optimization on important areas.
    """

    def __init__(self,
                 saliency_weight: float = 2.0,
                 texture_weight: float = 1.5,
                 device: str = 'cpu'):
        super().__init__()
        self.saliency_weight = saliency_weight
        self.texture_weight = texture_weight
        self.device = device

    def compute_saliency_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute simple center-surround saliency."""
        # Convert to grayscale
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x

        # Center-surround using Gaussian differences
        sigma1, sigma2 = 1.0, 3.0

        # Create Gaussian kernels
        def gaussian_kernel(sigma: float, size: int = 7) -> torch.Tensor:
            coords = torch.arange(size, dtype=torch.float32, device=self.device)
            coords = coords - size // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g = g / g.sum()
            return g.unsqueeze(0) * g.unsqueeze(1)

        kernel1 = gaussian_kernel(sigma1).unsqueeze(0).unsqueeze(0)
        kernel2 = gaussian_kernel(sigma2, size=15).unsqueeze(0).unsqueeze(0)

        blur1 = F.conv2d(gray, kernel1, padding=kernel1.shape[-1]//2)
        blur2 = F.conv2d(gray, kernel2, padding=kernel2.shape[-1]//2)

        saliency = torch.abs(blur1 - blur2)
        return saliency / (saliency.max() + 1e-8)

    def compute_texture_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local texture complexity using variance."""
        # Convert to grayscale
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x

        # Local variance using unfold
        patches = F.unfold(gray, kernel_size=5, padding=2)  # 5x5 patches
        patches = patches.view(gray.shape[0], 25, gray.shape[2], gray.shape[3])

        variance = torch.var(patches, dim=1, keepdim=True)
        return variance / (variance.max() + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute content-adaptive weighted loss."""
        # Base L2 loss
        base_loss = (pred - target) ** 2

        # Compute importance maps
        saliency = self.compute_saliency_map(target)
        texture = self.compute_texture_complexity(target)

        # Combine importance maps
        importance = 1.0 + self.saliency_weight * saliency + self.texture_weight * texture

        # Weight loss by importance
        weighted_loss = base_loss * importance

        return weighted_loss.mean()


class AdvancedLoss(nn.Module):
    """
    Combined advanced loss function that integrates multiple quality metrics.
    Balances pixel accuracy, perceptual quality, edge preservation, and content adaptation.
    """

    def __init__(self,
                 mse_weight: float = 1.0,
                 perceptual_weight: float = 0.5,
                 edge_weight: float = 0.3,
                 content_weight: float = 0.2,
                 device: str = 'cpu'):
        super().__init__()

        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.content_weight = content_weight

        # Initialize component losses
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(device=device)
        if edge_weight > 0:
            self.edge_loss = EdgePreservingLoss(device=device)
        if content_weight > 0:
            self.content_loss = ContentAdaptiveLoss(device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute combined advanced loss with component breakdown."""
        losses = {}
        total_loss = 0.0

        # MSE loss
        if self.mse_weight > 0:
            mse_loss = F.mse_loss(pred, target)
            losses['mse'] = mse_loss
            total_loss += self.mse_weight * mse_loss

        # Perceptual loss
        if self.perceptual_weight > 0 and hasattr(self, 'perceptual_loss'):
            perc_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perc_loss
            total_loss += self.perceptual_weight * perc_loss

        # Edge preservation loss
        if self.edge_weight > 0 and hasattr(self, 'edge_loss'):
            edge_loss = self.edge_loss(pred, target)
            losses['edge'] = edge_loss
            total_loss += self.edge_weight * edge_loss

        # Content-adaptive loss
        if self.content_weight > 0 and hasattr(self, 'content_loss'):
            content_loss = self.content_loss(pred, target)
            losses['content'] = content_loss
            total_loss += self.content_weight * content_loss

        losses['total'] = total_loss
        return losses


def create_advanced_loss(config: Dict[str, Any]) -> AdvancedLoss:
    """Factory function to create advanced loss from configuration."""
    return AdvancedLoss(
        mse_weight=config.get('mse_weight', 1.0),
        perceptual_weight=config.get('perceptual_weight', 0.5),
        edge_weight=config.get('edge_weight', 0.3),
        content_weight=config.get('content_weight', 0.2),
        device=config.get('device', 'cpu')
    )


if __name__ == "__main__":
    # Test the advanced loss functions
    device = 'cpu'

    # Create sample images
    pred = torch.randn(1, 3, 64, 64, device=device)
    target = torch.randn(1, 3, 64, 64, device=device)

    # Test individual losses
    print("Testing Advanced Loss Functions:")

    # Perceptual loss
    perc_loss = PerceptualLoss(device=device)
    perc_val = perc_loss(pred, target)
    print(f"Perceptual Loss: {perc_val.item():.6f}")

    # Edge loss
    edge_loss = EdgePreservingLoss(device=device)
    edge_val = edge_loss(pred, target)
    print(f"Edge Loss: {edge_val.item():.6f}")

    # Content loss
    content_loss = ContentAdaptiveLoss(device=device)
    content_val = content_loss(pred, target)
    print(f"Content Loss: {content_val.item():.6f}")

    # Combined loss
    combined_loss = AdvancedLoss(device=device)
    combined_vals = combined_loss(pred, target)
    print(f"Combined Loss Breakdown:")
    for name, val in combined_vals.items():
        print(f"  {name}: {val.item():.6f}")
