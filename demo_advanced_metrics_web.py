#!/usr/bin/env python3
"""Web-based demonstration of advanced error metrics with side-by-side comparison.

This script creates a comprehensive web interface to showcase the T4.2: Advanced Error Metrics
implementation using real images and different reconstruction methods.
"""

import sys
import time
import logging
import numpy as np
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flask import Flask, render_template_string, jsonify

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.advanced_error_metrics import (
    AdvancedErrorAnalyzer,
    ComparativeQualityAssessment,
    compare_reconstruction_methods
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def load_and_process_image(image_path: str, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Load and process image for analysis."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to target size
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to numpy array and normalize to [0, 1]
            image_array = np.array(img).astype(np.float32) / 255.0

            return image_array
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise


def create_reconstruction_methods(target: np.ndarray) -> Dict[str, np.ndarray]:
    """Create different reconstruction methods for comparison."""
    np.random.seed(42)  # For reproducible results

    reconstructions = {}

    # Method 1: High Quality (minimal noise)
    noise_scale = 0.02
    reconstructions["High Quality"] = np.clip(
        target + noise_scale * np.random.randn(*target.shape), 0.0, 1.0)

    # Method 2: Gaussian Blur (loss of detail)
    from scipy.ndimage import gaussian_filter
    blurred = np.zeros_like(target)
    for c in range(3):
        blurred[:, :, c] = gaussian_filter(target[:, :, c], sigma=1.2)
    reconstructions["Gaussian Blur"] = np.clip(blurred + 0.05 * np.random.randn(*target.shape), 0.0, 1.0)

    # Method 3: Downsampled (loss of high frequencies)
    from skimage.transform import resize
    H, W = target.shape[:2]
    low_res = resize(target, (H//3, W//3, 3), anti_aliasing=True)
    upsampled = resize(low_res, (H, W, 3), anti_aliasing=True)
    reconstructions["Downsampled"] = np.clip(upsampled + 0.08 * np.random.randn(*target.shape), 0.0, 1.0)

    # Method 4: Bilateral Filter (edge-preserving but smooth)
    try:
        import cv2
        bilateral = cv2.bilateralFilter((target * 255).astype(np.uint8), 15, 50, 50) / 255.0
        reconstructions["Bilateral Filter"] = np.clip(bilateral, 0.0, 1.0)
    except ImportError:
        # Fallback if OpenCV not available
        reconstructions["Bilateral Filter"] = np.clip(
            target + 0.03 * np.random.randn(*target.shape), 0.0, 1.0)

    # Method 5: High-frequency artifacts
    high_freq_noise = 0.12 * np.random.randn(*target.shape)
    reconstructions["High-Freq Artifacts"] = np.clip(target + high_freq_noise, 0.0, 1.0)

    # Method 6: JPEG-like compression artifacts
    try:
        # Simulate JPEG compression by saving and reloading at low quality
        img_pil = Image.fromarray((target * 255).astype(np.uint8))
        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=30)
        buffer.seek(0)
        compressed = Image.open(buffer)
        reconstructions["JPEG Compressed"] = np.array(compressed).astype(np.float32) / 255.0
    except Exception:
        # Fallback
        reconstructions["JPEG Compressed"] = np.clip(
            target + 0.06 * np.random.randn(*target.shape), 0.0, 1.0)

    return reconstructions


def array_to_base64(image_array: np.ndarray, format: str = 'PNG') -> str:
    """Convert numpy array to base64 encoded image string."""
    # Convert to 0-255 uint8
    if image_array.max() <= 1.0:
        image_uint8 = (image_array * 255).astype(np.uint8)
    else:
        image_uint8 = image_array.astype(np.uint8)

    # Create PIL image
    img = Image.fromarray(image_uint8)

    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/{format.lower()};base64,{img_base64}"


def create_error_map_visualization(error_map: np.ndarray, colormap: str = 'hot') -> str:
    """Create a colorized error map as base64 image."""
    # Normalize error map to [0, 1]
    if error_map.max() > error_map.min():
        normalized = (error_map - error_map.min()) / (error_map.max() - error_map.min())
    else:
        normalized = np.zeros_like(error_map)

    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    colored = cmap(normalized)

    # Convert to RGB (remove alpha channel)
    rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)

    return array_to_base64(rgb_image)


def analyze_image_with_methods(target: np.ndarray, reconstructions: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Perform comprehensive analysis using advanced error metrics."""

    # Initialize analyzer
    analyzer = AdvancedErrorAnalyzer()

    # Perform comparative assessment
    comparison_results = compare_reconstruction_methods(target, reconstructions)

    # Create error maps for each method
    error_maps = {}
    for method_name, reconstruction in reconstructions.items():
        # Standard L1 error map
        l1_map = analyzer.create_error_map(target, reconstruction, 'l1')

        # Advanced error maps
        content_map = analyzer.create_advanced_error_map(target, reconstruction, 'content_weighted')
        freq_map = analyzer.create_advanced_error_map(target, reconstruction, 'frequency_weighted')

        error_maps[method_name] = {
            'l1': create_error_map_visualization(l1_map, 'hot'),
            'content_weighted': create_error_map_visualization(content_map, 'plasma'),
            'frequency_weighted': create_error_map_visualization(freq_map, 'viridis')
        }

    # Convert images to base64 for web display
    images_b64 = {}
    images_b64['target'] = array_to_base64(target)

    for method_name, reconstruction in reconstructions.items():
        images_b64[method_name] = array_to_base64(reconstruction)

    return {
        'comparison_results': comparison_results,
        'error_maps': error_maps,
        'images': images_b64,
        'target_stats': {
            'shape': target.shape,
            'mean': float(np.mean(target)),
            'std': float(np.std(target)),
            'min': float(np.min(target)),
            'max': float(np.max(target))
        }
    }


# Global variables to store analysis results
analysis_data = None
target_image = None


@app.route('/')
def index():
    """Main page with side-by-side comparison interface."""

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Advanced Error Metrics - Side-by-Side Comparison</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }

            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }

            .header p {
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }

            .controls {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }

            .comparison-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }

            .method-card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s ease;
            }

            .method-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
            }

            .method-card h3 {
                margin: 0 0 15px 0;
                font-size: 1.4em;
                color: #4a5568;
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 10px;
            }

            .image-container {
                position: relative;
                margin-bottom: 15px;
            }

            .image-container img {
                width: 100%;
                height: auto;
                border-radius: 5px;
                cursor: pointer;
                transition: opacity 0.3s ease;
            }

            .image-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0;
                transition: opacity 0.3s ease;
                border-radius: 5px;
            }

            .image-container:hover .image-overlay {
                opacity: 0.7;
            }

            .metrics-table {
                font-size: 0.9em;
                line-height: 1.4;
            }

            .metrics-table td {
                padding: 4px 8px;
                border-bottom: 1px solid #e2e8f0;
            }

            .metrics-table td:first-child {
                font-weight: 600;
                color: #4a5568;
                white-space: nowrap;
            }

            .rank-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 0.8em;
                margin-left: 10px;
            }

            .rank-1 { background: #48bb78; color: white; }
            .rank-2 { background: #ed8936; color: white; }
            .rank-3 { background: #f56565; color: white; }
            .rank-4, .rank-5, .rank-6 { background: #cbd5e0; color: #4a5568; }

            .error-map-selector {
                margin: 15px 0;
            }

            .error-map-selector button {
                padding: 8px 16px;
                margin: 2px;
                border: none;
                border-radius: 5px;
                background: #e2e8f0;
                color: #4a5568;
                cursor: pointer;
                transition: all 0.2s ease;
                font-size: 0.85em;
            }

            .error-map-selector button:hover {
                background: #cbd5e0;
            }

            .error-map-selector button.active {
                background: #667eea;
                color: white;
            }

            .loading {
                text-align: center;
                padding: 40px;
                font-size: 1.2em;
                color: #666;
            }

            .stats-panel {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }

            .stat-item {
                text-align: center;
                padding: 15px;
                background: #f7fafc;
                border-radius: 8px;
            }

            .stat-value {
                font-size: 1.8em;
                font-weight: bold;
                color: #667eea;
            }

            .stat-label {
                font-size: 0.9em;
                color: #718096;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Advanced Error Metrics Demonstration</h1>
            <p>T4.2 Implementation - Side-by-Side Quality Comparison</p>
        </div>

        <div class="controls">
            <button onclick="loadAnalysis()" style="padding: 12px 24px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1em;">
                🔍 Load Image Analysis
            </button>
            <span id="status" style="margin-left: 15px; color: #666;"></span>
        </div>

        <div id="stats-panel" class="stats-panel" style="display: none;">
            <h3>📊 Analysis Overview</h3>
            <div class="stats-grid" id="stats-content">
            </div>
        </div>

        <div id="comparison-container" class="comparison-grid">
            <div class="loading">Click "Load Image Analysis" to begin comparison...</div>
        </div>

        <script>
            let analysisData = null;
            let currentErrorMapType = 'l1';

            async function loadAnalysis() {
                const statusEl = document.getElementById('status');
                const containerEl = document.getElementById('comparison-container');

                statusEl.textContent = '🔄 Analyzing image and generating reconstructions...';
                containerEl.innerHTML = '<div class="loading">🔄 Processing advanced error metrics...</div>';

                try {
                    const response = await fetch('/analyze');
                    analysisData = await response.json();

                    statusEl.textContent = '✅ Analysis complete!';
                    displayResults();
                    showStats();
                } catch (error) {
                    statusEl.textContent = '❌ Error: ' + error.message;
                    console.error('Analysis failed:', error);
                }
            }

            function showStats() {
                const statsPanel = document.getElementById('stats-panel');
                const statsContent = document.getElementById('stats-content');

                if (!analysisData) return;

                const methods = Object.keys(analysisData.comparison_results);
                const avgLPIPS = methods.reduce((sum, method) =>
                    sum + analysisData.comparison_results[method].advanced.lpips_score, 0) / methods.length;
                const avgSSIM = methods.reduce((sum, method) =>
                    sum + analysisData.comparison_results[method].basic.ssim_score, 0) / methods.length;
                const avgScore = methods.reduce((sum, method) =>
                    sum + analysisData.comparison_results[method].combined_score, 0) / methods.length;

                statsContent.innerHTML = `
                    <div class="stat-item">
                        <div class="stat-value">${methods.length}</div>
                        <div class="stat-label">Methods Compared</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${avgLPIPS.toFixed(4)}</div>
                        <div class="stat-label">Avg LPIPS Score</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${avgSSIM.toFixed(3)}</div>
                        <div class="stat-label">Avg SSIM Score</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${avgScore.toFixed(3)}</div>
                        <div class="stat-label">Avg Quality Score</div>
                    </div>
                `;

                statsPanel.style.display = 'block';
            }

            function displayResults() {
                if (!analysisData) return;

                const container = document.getElementById('comparison-container');

                // Add target image first
                let html = createTargetCard();

                // Add reconstruction method cards
                const sortedMethods = Object.entries(analysisData.comparison_results)
                    .sort(([,a], [,b]) => a.advanced.quality_rank - b.advanced.quality_rank);

                for (const [methodName, metrics] of sortedMethods) {
                    html += createMethodCard(methodName, metrics);
                }

                container.innerHTML = html;
            }

            function createTargetCard() {
                return `
                    <div class="method-card">
                        <h3>🎯 Original Image</h3>
                        <div class="image-container">
                            <img src="${analysisData.images.target}" alt="Original Image">
                        </div>
                        <table class="metrics-table">
                            <tr><td>Resolution:</td><td>${analysisData.target_stats.shape[0]}×${analysisData.target_stats.shape[1]}</td></tr>
                            <tr><td>Mean Value:</td><td>${analysisData.target_stats.mean.toFixed(3)}</td></tr>
                            <tr><td>Std Dev:</td><td>${analysisData.target_stats.std.toFixed(3)}</td></tr>
                            <tr><td>Dynamic Range:</td><td>${analysisData.target_stats.min.toFixed(3)} - ${analysisData.target_stats.max.toFixed(3)}</td></tr>
                        </table>
                    </div>
                `;
            }

            function createMethodCard(methodName, metrics) {
                const rank = metrics.advanced.quality_rank;
                const basic = metrics.basic;
                const advanced = metrics.advanced;

                return `
                    <div class="method-card">
                        <h3>${methodName} <span class="rank-badge rank-${rank}">Rank #${rank}</span></h3>

                        <div class="image-container">
                            <img src="${analysisData.images[methodName]}" alt="${methodName}" id="img-${methodName}">
                            <img class="image-overlay" src="${analysisData.error_maps[methodName][currentErrorMapType]}"
                                 alt="Error Map" id="overlay-${methodName}" style="display: none;">
                        </div>

                        <div class="error-map-selector">
                            <button onclick="toggleErrorMap('${methodName}', 'l1')" class="active" id="btn-${methodName}-l1">L1 Error</button>
                            <button onclick="toggleErrorMap('${methodName}', 'content_weighted')" id="btn-${methodName}-content_weighted">Content-Weighted</button>
                            <button onclick="toggleErrorMap('${methodName}', 'frequency_weighted')" id="btn-${methodName}-frequency_weighted">Frequency-Weighted</button>
                        </div>

                        <table class="metrics-table">
                            <tr><td>Combined Score:</td><td><strong>${metrics.combined_score.toFixed(4)}</strong></td></tr>
                            <tr><td>L1 Error:</td><td>${basic.l1_error.toFixed(6)}</td></tr>
                            <tr><td>SSIM Score:</td><td>${basic.ssim_score.toFixed(4)}</td></tr>
                            <tr><td>PSNR:</td><td>${basic.psnr.toFixed(2)} dB</td></tr>
                            <tr><td>LPIPS Score:</td><td>${advanced.lpips_score.toFixed(4)}</td></tr>
                            <tr><td>Gradient Similarity:</td><td>${advanced.gradient_similarity.toFixed(4)}</td></tr>
                            <tr><td>Edge Coherence:</td><td>${advanced.edge_coherence.toFixed(4)}</td></tr>
                            <tr><td>High-Freq Preservation:</td><td>${advanced.high_freq_preservation.toFixed(4)}</td></tr>
                            <tr><td>Content-Weighted Error:</td><td>${advanced.content_weighted_error.toFixed(6)}</td></tr>
                            <tr><td>Spectral Distortion:</td><td>${advanced.spectral_distortion.toFixed(2)}</td></tr>
                        </table>
                    </div>
                `;
            }

            function toggleErrorMap(methodName, mapType) {
                const img = document.getElementById(`img-${methodName}`);
                const overlay = document.getElementById(`overlay-${methodName}`);

                // Update button states
                ['l1', 'content_weighted', 'frequency_weighted'].forEach(type => {
                    const btn = document.getElementById(`btn-${methodName}-${type}`);
                    btn.classList.remove('active');
                });

                const activeBtn = document.getElementById(`btn-${methodName}-${mapType}`);
                activeBtn.classList.add('active');

                // Update overlay
                overlay.src = analysisData.error_maps[methodName][mapType];

                // Toggle visibility
                if (overlay.style.display === 'none') {
                    overlay.style.display = 'block';
                    img.style.opacity = '0.3';
                } else if (currentErrorMapType === mapType) {
                    overlay.style.display = 'none';
                    img.style.opacity = '1';
                } else {
                    overlay.style.display = 'block';
                    img.style.opacity = '0.3';
                }

                currentErrorMapType = mapType;
            }

            // Auto-load analysis on page load
            window.addEventListener('load', () => {
                setTimeout(loadAnalysis, 500);
            });
        </script>
    </body>
    </html>
    """

    return render_template_string(html_template)


@app.route('/analyze')
def analyze():
    """Perform image analysis and return results as JSON."""
    global analysis_data, target_image

    try:
        # Load the image
        image_path = "SCR-20250921-omxs.png"

        if not Path(image_path).exists():
            return jsonify({"error": f"Image not found: {image_path}"}), 404

        logger.info(f"Loading image: {image_path}")
        target_image = load_and_process_image(image_path, target_size=(512, 512))

        logger.info("Creating reconstruction methods...")
        reconstructions = create_reconstruction_methods(target_image)

        logger.info("Analyzing with advanced error metrics...")
        analysis_data = analyze_image_with_methods(target_image, reconstructions)

        logger.info("Analysis complete!")

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        analysis_data_clean = convert_numpy_types(analysis_data)
        return jsonify(analysis_data_clean)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🎯 Advanced Error Metrics Web Demo")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser and navigate to: http://localhost:8080")
    print("Or use Playwright to automate the demonstration")
    print()

    app.run(host='0.0.0.0', port=8080, debug=True)