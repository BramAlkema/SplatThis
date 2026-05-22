#!/usr/bin/env python3
"""
Enhanced PNG→SVG Converter with Advanced Quality Features

Integrates perceptual loss, edge-preserving initialization, content-adaptive
splat placement, and multi-scale optimization for superior visual quality.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.splat import GaussianSplat, create_isotropic_splat
from png2svg_gs.io import load_png, save_svg
from png2svg_gs.renderer import create_renderer, render_splats_numpy, splats_to_tensor, tensor_to_splats
from png2svg_gs.losses import AdvancedLoss, create_advanced_loss
from png2svg_gs.advanced_features import (
    ContentAdaptiveSplatPlacer,
    EdgeDetectionConfig,
    StructureTensorConfig
)


class EnhancedPNG2SVGConverter(PNG2SVGConverter):
    """
    Enhanced PNG→SVG converter with advanced quality features.

    Improvements over base converter:
    - Perceptual loss functions (VGG-based)
    - Edge-preserving initialization
    - Content-adaptive splat placement
    - Structure tensor-guided anisotropic splats
    - Multi-scale optimization
    """

    def __init__(self,
                 max_splats: int = 300,
                 stages: List[int] = [30, 20, 15, 10],
                 device: str = 'cpu',
                 advanced_loss_config: Optional[Dict[str, Any]] = None,
                 edge_config: Optional[EdgeDetectionConfig] = None,
                 structure_config: Optional[StructureTensorConfig] = None,
                 multi_scale: bool = True,
                 quality_preset: str = 'balanced'):

        # Initialize base converter
        super().__init__(max_splats=max_splats, stages=stages, device=device)

        # Advanced features
        self.multi_scale = multi_scale
        self.quality_preset = quality_preset

        # Configure advanced loss
        self.advanced_loss_config = advanced_loss_config or self._get_loss_config(quality_preset)

        # Initialize advanced components
        self.content_placer = ContentAdaptiveSplatPlacer(edge_config, structure_config)

        print(f"Enhanced converter initialized with {quality_preset} preset")
        print(f"Advanced features: perceptual loss, edge detection, structure analysis")

    def _get_loss_config(self, preset: str) -> Dict[str, Any]:
        """Get loss configuration for quality preset."""
        configs = {
            'fast': {
                'mse_weight': 1.0,
                'perceptual_weight': 0.1,
                'edge_weight': 0.1,
                'content_weight': 0.0,
                'device': self.device
            },
            'balanced': {
                'mse_weight': 1.0,
                'perceptual_weight': 0.3,
                'edge_weight': 0.2,
                'content_weight': 0.1,
                'device': self.device
            },
            'quality': {
                'mse_weight': 1.0,
                'perceptual_weight': 0.5,
                'edge_weight': 0.3,
                'content_weight': 0.2,
                'device': self.device
            }
        }
        return configs.get(preset, configs['balanced'])

    def _initialize_splats(
        self, image: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> List[GaussianSplat]:
        """Enhanced splat initialization using content-adaptive placement."""
        height, width = image.shape[:2]

        # Determine splat count based on image complexity
        complexity_factor = self._estimate_complexity(image)
        base_count = min(self.max_splats, max(100, width * height // 160))
        initial_count = int(base_count * (0.8 + 0.4 * complexity_factor))

        print(f"Image complexity: {complexity_factor:.3f}")
        print(f"Creating {initial_count} content-adaptive splats for {width}x{height} image")

        # Use content-adaptive placement
        candidates = self.content_placer.generate_splat_candidates(
            image, initial_count, strategy='adaptive'
        )

        # Convert to splats
        splats = self.content_placer.create_splats_from_candidates(candidates)

        print(f"Generated {len(splats)} splats:")
        splat_types = {}
        for splat in splats:
            stype = getattr(splat, 'type', 'unknown')
            splat_types[stype] = splat_types.get(stype, 0) + 1

        for stype, count in splat_types.items():
            print(f"  {stype}: {count} splats")

        return splats

    def _estimate_complexity(self, image: np.ndarray) -> float:
        """Estimate image complexity for adaptive splat count."""
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image

        # Edge density
        from scipy.ndimage import sobel
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_density = np.mean(edge_magnitude > 0.1)

        # Color variance
        color_variance = np.var(image.reshape(-1, image.shape[-1]), axis=0).mean()

        # Spatial frequency
        fft = np.fft.fft2(gray)
        freq_magnitude = np.abs(fft)
        high_freq_energy = np.sum(freq_magnitude[freq_magnitude.shape[0]//4:, freq_magnitude.shape[1]//4:])
        total_energy = np.sum(freq_magnitude)
        freq_ratio = high_freq_energy / (total_energy + 1e-8)

        # Combine metrics
        complexity = (edge_density + color_variance + freq_ratio) / 3.0
        return np.clip(complexity, 0.0, 1.0)

    def _create_loss_function(self, target_tensor: torch.Tensor) -> AdvancedLoss:
        """Create advanced loss function."""
        return create_advanced_loss(self.advanced_loss_config)

    def _optimize_stage(self,
                       splats_tensor: torch.Tensor,
                       target_tensor: torch.Tensor,
                       iterations: int,
                       stage_name: str) -> Dict[str, List[float]]:
        """Enhanced optimization with advanced loss."""

        # Create advanced loss function
        loss_fn = self._create_loss_function(target_tensor)

        # Optimizer
        optimizer = torch.optim.Adam([splats_tensor], lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5
        )

        # Track metrics
        metrics = {
            'total_loss': [],
            'mse_loss': [],
            'perceptual_loss': [],
            'edge_loss': [],
            'content_loss': []
        }

        print(f"\n{stage_name} - Advanced optimization ({iterations} iterations)")

        for i in range(iterations):
            optimizer.zero_grad()

            # Render current splats
            rendered_hwc = self.renderer(splats_tensor)
            rendered = rendered_hwc.permute(2, 0, 1).unsqueeze(0)

            # Compute advanced loss
            losses = loss_fn(rendered, target_tensor)

            # Backward pass
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([splats_tensor], max_norm=1.0)

            # Update
            optimizer.step()

            # Clamp parameters
            self._clamp_splat_parameters(splats_tensor)

            # Update scheduler
            scheduler.step(losses['total'])

            # Track metrics
            for key in metrics:
                if key in losses:
                    metrics[key].append(losses[key].item())

            # Progress reporting
            if (i + 1) % max(1, iterations // 5) == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  Iter {i+1:3d}/{iterations}: "
                      f"Total={losses['total'].item():.6f}, "
                      f"MSE={losses.get('mse', torch.tensor(0)).item():.6f}, "
                      f"LR={lr:.6f}")

        return metrics

    def _multi_scale_optimization(self,
                                 image: np.ndarray,
                                 splats: List[GaussianSplat]) -> List[GaussianSplat]:
        """Multi-scale optimization for fine detail preservation."""
        if not self.multi_scale:
            return splats

        print("\nApplying multi-scale optimization...")

        # Create multiple scales
        scales = [1.0, 0.5, 0.25]
        H, W = image.shape[:2]
        base_renderer = getattr(self, "renderer", None)

        for scale in scales:
            if scale == 1.0:
                continue  # Skip full resolution (already optimized)

            scale_h, scale_w = int(H * scale), int(W * scale)

            # Resize image
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray((image * 255).astype(np.uint8))
            scaled_image = pil_image.resize((scale_w, scale_h), PILImage.LANCZOS)
            scaled_array = np.array(scaled_image).astype(np.float32) / 255.0

            print(f"  Scale {scale}: {scale_w}x{scale_h}")

            # Scale splat positions
            scaled_splats = []
            for splat in splats:
                scaled_splat = GaussianSplat(
                    mu=splat.mu * scale,
                    sigma=splat.sigma * (scale**2),
                    color=splat.color,
                    alpha=splat.alpha,
                    importance=splat.importance
                )
                scaled_splats.append(scaled_splat)

            # Convert to tensor
            scaled_tensor = splats_to_tensor(scaled_splats, device=self.device)
            scaled_tensor.requires_grad_(True)
            scaled_target = torch.from_numpy(scaled_array.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            self.renderer = create_renderer(
                backend=self.renderer_backend,
                width=scale_w,
                height=scale_h,
                device=self.device,
                blend_mode=self.blend_mode,
            )

            # Quick optimization at this scale
            self.target_size = (scale_h, scale_w)
            self._optimize_stage(scaled_tensor, scaled_target, 10, f"Scale {scale}")

            # Update original splats with scaled improvements
            updated_splats = tensor_to_splats(scaled_tensor)
            for i, (original, updated) in enumerate(zip(splats, updated_splats)):
                # Scale back up
                splats[i] = GaussianSplat(
                    mu=updated.mu / scale,
                    sigma=updated.sigma / (scale**2),
                    color=updated.color,
                    alpha=updated.alpha,
                    importance=updated.importance
                )

        # Reset target size
        self.target_size = (H, W)
        if base_renderer is not None:
            self.renderer = base_renderer
        return splats

    def convert(self,
               input_path: str,
               output_path: str,
               save_json: bool = False,
               verbose: bool = True) -> Dict[str, Any]:
        """Enhanced conversion with advanced features."""

        print(f"Enhanced PNG→SVG conversion: {input_path} → {output_path}")
        print(f"Quality preset: {self.quality_preset}")
        print(f"Multi-scale optimization: {self.multi_scale}")

        # Load image
        image = load_png(input_path)
        H, W = image.shape[:2]
        self.target_size = (H, W)

        # Initialize with advanced placement
        splats = self._initialize_splats(image)

        # Prepare tensors
        splats_tensor = splats_to_tensor(splats, device=self.device)
        splats_tensor.requires_grad_(True)
        target_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        self.renderer = create_renderer(
            backend=self.renderer_backend,
            width=W,
            height=H,
            device=self.device,
            blend_mode=self.blend_mode,
        )

        # Multi-stage optimization with advanced loss
        all_metrics = {}

        for stage_idx, iterations in enumerate(self.stages):
            stage_name = f"Stage {stage_idx + 1}"
            metrics = self._optimize_stage(splats_tensor, target_tensor, iterations, stage_name)
            all_metrics[stage_name] = metrics

        # Apply multi-scale refinement
        final_splats = tensor_to_splats(splats_tensor)
        if self.multi_scale:
            final_splats = self._multi_scale_optimization(image, final_splats)

        # Generate SVG
        svg_content = self._generate_svg(final_splats, W, H)

        # Save SVG
        with open(output_path, 'w') as f:
            f.write(svg_content)

        # Quality assessment
        rendered = render_splats_numpy(final_splats, W, H)
        mse = np.mean((rendered - image[:, :, :3]) ** 2)
        coverage = np.sum(rendered > 0.01) / rendered.size

        results = {
            'input_path': input_path,
            'output_path': output_path,
            'splat_count': len(final_splats),
            'mse': float(mse),
            'coverage': float(coverage),
            'optimization_metrics': all_metrics,
            'quality_preset': self.quality_preset,
            'multi_scale': self.multi_scale
        }

        if verbose:
            print(f"\nConversion completed!")
            print(f"  Final splats: {len(final_splats)}")
            print(f"  MSE: {mse:.6f}")
            print(f"  Coverage: {coverage:.1%}")

        # Save JSON if requested
        if save_json:
            json_path = output_path.replace('.svg', '.json')

            json_data = {
                'metadata': results,
                'splats': []
            }

            for splat in final_splats:
                json_data['splats'].append({
                    'mu': splat.mu.tolist(),
                    'sigma': splat.sigma.tolist(),
                    'color': splat.color.tolist(),
                    'alpha': float(splat.alpha),
                    'importance': float(splat.importance)
                })

            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            if verbose:
                print(f"  Saved JSON: {json_path}")

        return results


def create_enhanced_converter(preset: str = 'balanced',
                            max_splats: int = 500,
                            device: str = 'cpu') -> EnhancedPNG2SVGConverter:
    """Factory function to create enhanced converter with preset configurations."""

    presets = {
        'fast': {
            'stages': [20, 15, 10],
            'multi_scale': False,
            'advanced_loss_config': {
                'mse_weight': 1.0,
                'perceptual_weight': 0.1,
                'edge_weight': 0.1,
                'content_weight': 0.0,
                'device': device
            }
        },
        'balanced': {
            'stages': [40, 30, 20, 10],
            'multi_scale': True,
            'advanced_loss_config': {
                'mse_weight': 1.0,
                'perceptual_weight': 0.3,
                'edge_weight': 0.2,
                'content_weight': 0.1,
                'device': device
            }
        },
        'quality': {
            'stages': [60, 40, 30, 20, 10],
            'multi_scale': True,
            'advanced_loss_config': {
                'mse_weight': 1.0,
                'perceptual_weight': 0.5,
                'edge_weight': 0.3,
                'content_weight': 0.2,
                'device': device
            }
        }
    }

    config = presets.get(preset, presets['balanced'])

    return EnhancedPNG2SVGConverter(
        max_splats=max_splats,
        stages=config['stages'],
        device=device,
        advanced_loss_config=config['advanced_loss_config'],
        multi_scale=config['multi_scale'],
        quality_preset=preset
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python enhanced_converter.py <input.png> <output.svg> [preset]")
        print("Presets: fast, balanced, quality")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    preset = sys.argv[3] if len(sys.argv) > 3 else 'balanced'

    # Create enhanced converter
    converter = create_enhanced_converter(preset=preset)

    # Convert
    results = converter.convert(
        input_path=input_path,
        output_path=output_path,
        save_json=True,
        verbose=True
    )
