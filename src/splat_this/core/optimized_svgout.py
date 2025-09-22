"""Optimized SVG generation with performance improvements for large splat counts."""

import math
import logging
from typing import Dict, List, Optional, TextIO
from pathlib import Path
from xml.sax.saxutils import escape
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

from .extract import Gaussian
from ..utils.profiler import global_profiler, MemoryEfficientProcessor

logger = logging.getLogger(__name__)


class OptimizedSVGGenerator:
    """Optimized SVG generator for large splat counts."""

    def __init__(
        self,
        width: int,
        height: int,
        precision: int = 3,
        parallax_strength: int = 40,
        interactive_top: int = 0,
        chunk_size: int = 500,  # Process splats in chunks
        max_memory_mb: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.precision = precision
        self.parallax_strength = parallax_strength
        self.interactive_top = interactive_top
        self.chunk_size = chunk_size
        self.memory_processor = MemoryEfficientProcessor(max_memory_mb=max_memory_mb) if max_memory_mb else MemoryEfficientProcessor()

    @global_profiler.profile_function("svg_generation")
    def generate_svg(
        self,
        layers: Dict[int, List[Gaussian]],
        gaussian_mode: bool = False,
        title: Optional[str] = None,
    ) -> str:
        """Generate optimized SVG with chunked processing for large splat counts."""
        if not layers:
            logger.warning("No layers provided for SVG generation")
            return self._generate_empty_svg()

        total_splats = sum(len(layer_splats) for layer_splats in layers.values())
        logger.info(f"Generating optimized SVG with {len(layers)} layers, "
                   f"{total_splats} splats, gaussian_mode={gaussian_mode}")

        # Check memory requirements
        self.memory_processor.ensure_memory_limit("svg_generation_start")

        # Use streaming approach for very large splat counts
        if total_splats > 5000:
            return self._generate_streaming_svg(layers, gaussian_mode, title)
        else:
            return self._generate_standard_svg(layers, gaussian_mode, title)

    def _generate_streaming_svg(
        self,
        layers: Dict[int, List[Gaussian]],
        gaussian_mode: bool,
        title: Optional[str] = None,
    ) -> str:
        """Generate SVG using streaming approach for large splat counts."""
        logger.info("Using streaming SVG generation for large splat count")

        # Use StringIO for efficient string building
        output = io.StringIO()

        # Write header
        self._write_header(output, title)

        # Write definitions
        self._write_defs(output, gaussian_mode, layers)

        # Process layers in chunks
        sorted_layers = sorted(layers.items(), key=lambda x: x[0])

        for layer_idx, splats in sorted_layers:
            if not splats:
                continue

            self._write_layer_chunked(output, layer_idx, splats, gaussian_mode)

        # Write styles and scripts
        self._write_styles(output)
        self._write_scripts(output)

        # Write footer
        self._write_footer(output)

        result = output.getvalue()
        output.close()

        # Validate result
        if not self._validate_svg_structure(result):
            logger.error("Generated streaming SVG failed basic validation")
            raise ValueError("Invalid SVG structure generated")

        return result

    def _generate_standard_svg(
        self,
        layers: Dict[int, List[Gaussian]],
        gaussian_mode: bool,
        title: Optional[str] = None,
    ) -> str:
        """Generate SVG using standard approach for smaller splat counts."""
        # Generate SVG components
        header = self._generate_header(title)
        defs = self._generate_defs(gaussian_mode, layers)
        layer_groups = self._generate_layer_groups_optimized(layers, gaussian_mode)
        styles = self._generate_styles()
        scripts = self._generate_scripts()
        footer = self._generate_footer()

        # Combine all parts
        svg_content = f"{header}\n{defs}\n{layer_groups}\n{styles}\n{scripts}\n{footer}"

        # Validate basic structure
        if not self._validate_svg_structure(svg_content):
            logger.error("Generated standard SVG failed basic validation")
            raise ValueError("Invalid SVG structure generated")

        return svg_content

    def _write_header(self, output: TextIO, title: Optional[str] = None) -> None:
        """Write SVG header to output stream."""
        title_elem = f'\n    <title>{escape(title)}</title>' if title else ''

        header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 {self.width} {self.height}"
     xmlns="http://www.w3.org/2000/svg"
     style="width: 100%; height: 100vh; background: #000;"
     class="splat-svg"
     data-parallax-strength="{self.parallax_strength}"
     data-interactive-top="{self.interactive_top}">{title_elem}'''

        output.write(header)

    def _write_defs(self, output: TextIO, gaussian_mode: bool, layers: Optional[Dict[int, List[Gaussian]]] = None) -> None:
        """Write SVG definitions to output stream."""
        if not gaussian_mode:
            output.write("\n    <defs></defs>")
            return

        if not layers:
            # Fallback to simple single gradient
            gradient_def = '''
    <defs>
        <radialGradient id="gaussianGradient" cx="50%" cy="50%" r="50%" gradientUnits="objectBoundingBox">
            <stop offset="0%" stop-color="currentColor" stop-opacity="1"/>
            <stop offset="70%" stop-color="currentColor" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="currentColor" stop-opacity="0"/>
        </radialGradient>
    </defs>'''
            output.write(gradient_def)
            return

        # Collect unique colors from all splats
        unique_colors = set()
        for layer_splats in layers.values():
            for splat in layer_splats:
                rgb = f"rgb({splat.r}, {splat.g}, {splat.b})"
                unique_colors.add(rgb)

        # Generate individual gradient for each color
        output.write("\n    <defs>")
        for color in unique_colors:
            # Create safe ID from color
            color_id = color.replace("rgb(", "").replace(")", "").replace(", ", "_").replace(" ", "")
            gradient = f'''
        <radialGradient id="grad_{color_id}" cx="50%" cy="50%" r="50%" gradientUnits="objectBoundingBox">
            <stop offset="0%" stop-color="{color}" stop-opacity="1"/>
            <stop offset="70%" stop-color="{color}" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
        </radialGradient>'''
            output.write(gradient)
        output.write("\n    </defs>")

    @global_profiler.profile_function("chunked_layer_writing")
    def _write_layer_chunked(
        self,
        output: TextIO,
        layer_idx: int,
        splats: List[Gaussian],
        gaussian_mode: bool
    ) -> None:
        """Write layer with chunked splat processing."""
        if not splats:
            return

        depth = splats[0].depth if splats else 0.5

        # Write group header
        output.write(f'\n    <g class="layer" data-depth="{self._format_number(depth)}" data-layer="{layer_idx}">')

        # Process splats in chunks to manage memory
        for chunk_start in range(0, len(splats), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(splats))
            chunk_splats = splats[chunk_start:chunk_end]

            # Process chunk of splats
            for splat in chunk_splats:
                splat_svg = self._generate_splat_element_optimized(splat, gaussian_mode)
                output.write(f"\n        {splat_svg}")

            # Memory check after each chunk
            if chunk_end < len(splats):  # Not the last chunk
                self.memory_processor.ensure_memory_limit(f"layer_{layer_idx}_chunk_{chunk_start}")

        # Write group footer
        output.write("\n    </g>")

    def _write_styles(self, output: TextIO) -> None:
        """Write CSS styles to output stream."""
        styles = '''
    <style><![CDATA[
        .splat-svg {
            overflow: hidden;
            cursor: crosshair;
        }

        .layer {
            transform-style: preserve-3d;
            transition: transform 0.1s ease-out;
            will-change: transform;
        }

        .layer ellipse {
            vector-effect: non-scaling-stroke;
        }

        .interactive-splat {
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .interactive-splat:hover {
            transform: scale(1.1);
        }

        @media (prefers-reduced-motion: reduce) {
            .layer, .interactive-splat {
                transition: none !important;
                animation: none !important;
            }
        }

        @media (max-width: 768px) {
            .splat-svg {
                cursor: default;
            }
        }
    ]]></style>'''
        output.write(styles)

    def _write_scripts(self, output: TextIO) -> None:
        """Write JavaScript to output stream."""
        script = f'''
    <script><![CDATA[
        (function() {{
            'use strict';

            // Check for reduced motion preference
            const prefersReducedMotion = window.matchMedia &&
                window.matchMedia('(prefers-reduced-motion: reduce)').matches;

            if (prefersReducedMotion) {{
                console.log('SplatThis: Animations disabled due to user preference');
                return;
            }}

            const svg = document.querySelector('.splat-svg');
            const layers = document.querySelectorAll('.layer');
            const parallaxStrength = {self.parallax_strength};
            const interactiveTop = {self.interactive_top};

            let isGyroSupported = false;
            let isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

            // Optimized mouse parallax for desktop
            let animationFrameId = null;
            function handleMouseMove(e) {{
                if (isMobile || !svg) return;

                // Throttle using requestAnimationFrame
                if (animationFrameId) return;

                animationFrameId = requestAnimationFrame(() => {{
                    const rect = svg.getBoundingClientRect();
                    const centerX = rect.width / 2;
                    const centerY = rect.height / 2;

                    const mouseX = e.clientX - rect.left;
                    const mouseY = e.clientY - rect.top;

                    const deltaX = (mouseX - centerX) / centerX;
                    const deltaY = (mouseY - centerY) / centerY;

                    updateLayerTransforms(deltaX, deltaY);
                    animationFrameId = null;
                }});
            }}

            // Optimized gyroscope parallax for mobile
            function handleDeviceOrientation(e) {{
                if (!isMobile || !isGyroSupported) return;

                const beta = e.beta || 0;   // X-axis rotation
                const gamma = e.gamma || 0; // Y-axis rotation

                const deltaX = Math.max(-1, Math.min(1, gamma / 30));
                const deltaY = Math.max(-1, Math.min(1, beta / 30));

                updateLayerTransforms(deltaX, deltaY);
            }}

            function updateLayerTransforms(deltaX, deltaY) {{
                layers.forEach((layer, index) => {{
                    const depth = parseFloat(layer.getAttribute('data-depth')) || 0.5;
                    const parallaxFactor = (depth - 0.6) * parallaxStrength;

                    const translateX = deltaX * parallaxFactor;
                    const translateY = deltaY * parallaxFactor;

                    layer.style.transform = `translate(${{translateX}}px, ${{translateY}}px)`;

                    // Add interactive effects for top splats
                    if (interactiveTop > 0 && index < interactiveTop) {{
                        layer.classList.add('interactive-splat');
                    }}
                }});
            }}

            // Initialize event listeners
            if (!isMobile) {{
                svg.addEventListener('mousemove', handleMouseMove, {{ passive: true }});
                svg.addEventListener('mouseleave', () => {{
                    updateLayerTransforms(0, 0);
                }});
            }} else {{
                // Request gyroscope permission on iOS 13+
                if (typeof DeviceOrientationEvent !== 'undefined' &&
                    typeof DeviceOrientationEvent.requestPermission === 'function') {{
                    DeviceOrientationEvent.requestPermission()
                        .then(response => {{
                            if (response === 'granted') {{
                                isGyroSupported = true;
                                window.addEventListener('deviceorientation', handleDeviceOrientation, {{ passive: true }});
                            }}
                        }})
                        .catch(console.error);
                }} else if (typeof DeviceOrientationEvent !== 'undefined') {{
                    isGyroSupported = true;
                    window.addEventListener('deviceorientation', handleDeviceOrientation, {{ passive: true }});
                }}
            }}

            console.log(`SplatThis SVG loaded - ${{layers.length}} layers, parallax: ${{parallaxStrength}}`);
        }})();
    ]]></script>'''
        output.write(script)

    def _write_footer(self, output: TextIO) -> None:
        """Write SVG footer to output stream."""
        output.write("\n</svg>")

    @global_profiler.profile_function("optimized_layer_groups")
    def _generate_layer_groups_optimized(self, layers: Dict[int, List[Gaussian]], gaussian_mode: bool) -> str:
        """Generate layer groups with optimization for large splat counts."""
        layer_groups = []

        # Sort layers by depth (back to front)
        sorted_layers = sorted(layers.items(), key=lambda x: x[0])

        for layer_idx, splats in sorted_layers:
            if not splats:
                continue

            # Use parallel processing for layers with many splats
            if len(splats) > 200:
                layer_content = self._generate_large_layer_parallel(layer_idx, splats, gaussian_mode)
            else:
                layer_content = self._generate_layer_sequential(layer_idx, splats, gaussian_mode)

            layer_groups.append(layer_content)

        return "\n".join(layer_groups)

    def _generate_large_layer_parallel(
        self,
        layer_idx: int,
        splats: List[Gaussian],
        gaussian_mode: bool
    ) -> str:
        """Generate layer content using parallel processing for large layers."""
        depth = splats[0].depth if splats else 0.5

        # Split splats into chunks for parallel processing
        chunk_size = 100
        splat_chunks = [
            splats[i:i + chunk_size]
            for i in range(0, len(splats), chunk_size)
        ]

        # Process chunks in parallel
        splat_elements = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_chunk = {
                executor.submit(self._process_splat_chunk, chunk, gaussian_mode): chunk
                for chunk in splat_chunks
            }

            for future in as_completed(future_to_chunk):
                try:
                    chunk_elements = future.result()
                    splat_elements.extend(chunk_elements)
                except Exception as e:
                    logger.warning(f"Failed to process splat chunk: {e}")

        # Generate group
        group_header = f'    <g class="layer" data-depth="{self._format_number(depth)}" data-layer="{layer_idx}">'
        group_footer = "    </g>"

        formatted_elements = [f"        {elem}" for elem in splat_elements]
        layer_content = f"{group_header}\n" + "\n".join(formatted_elements) + f"\n{group_footer}"

        return layer_content

    def _generate_layer_sequential(
        self,
        layer_idx: int,
        splats: List[Gaussian],
        gaussian_mode: bool
    ) -> str:
        """Generate layer content sequentially for smaller layers."""
        depth = splats[0].depth if splats else 0.5

        # Generate group header
        group_header = f'    <g class="layer" data-depth="{self._format_number(depth)}" data-layer="{layer_idx}">'

        # Generate splats for this layer
        splat_elements = []
        for splat in splats:
            splat_svg = self._generate_splat_element_optimized(splat, gaussian_mode)
            splat_elements.append(f"        {splat_svg}")

        # Generate group footer
        group_footer = "    </g>"

        # Combine layer
        layer_content = f"{group_header}\n" + "\n".join(splat_elements) + f"\n{group_footer}"
        return layer_content

    def _process_splat_chunk(self, splats: List[Gaussian], gaussian_mode: bool) -> List[str]:
        """Process a chunk of splats in parallel."""
        return [
            self._generate_splat_element_optimized(splat, gaussian_mode)
            for splat in splats
        ]

    def _generate_splat_element_optimized(self, splat: Gaussian, gaussian_mode: bool) -> str:
        """Generate optimized SVG ellipse element for a single splat."""
        # Pre-compute and cache formatted values
        cx = self._format_number(splat.x)
        cy = self._format_number(splat.y)
        rx = self._format_number(splat.rx)
        ry = self._format_number(splat.ry)

        # Only include rotation if significant (optimization)
        if abs(splat.theta) > 1e-6:
            rotation_deg = self._format_number(math.degrees(splat.theta))
            transform = f' transform="rotate({rotation_deg} {cx} {cy})"'
        else:
            transform = ''

        # Optimize color formatting
        if gaussian_mode:
            # Use gradient fill for gaussian appearance with color-specific gradient
            color_rgb = f"rgb({splat.r}, {splat.g}, {splat.b})"
            color_id = color_rgb.replace("rgb(", "").replace(")", "").replace(", ", "_").replace(" ", "")
            style = (
                f'color: {color_rgb}; '
                f'fill: url(#grad_{color_id}); '
                f'fill-opacity: {self._format_number(splat.a)}; '
                'stroke: none;'
            )
            color_attr = f' data-color="{color_rgb}"'
        else:
            # Use optimized RGBA format
            style = f'fill: rgba({splat.r}, {splat.g}, {splat.b}, {self._format_number(splat.a)}); stroke: none;'
            color_attr = ''

        # Build element efficiently
        element_parts = [
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"',
            f' style="{style}"'
        ]

        if color_attr:
            element_parts.append(color_attr)

        if transform:
            element_parts.append(transform)

        element_parts.append('/>')

        return ''.join(element_parts)

    # Copy other methods from original SVGGenerator with minimal changes
    def _generate_header(self, title: Optional[str] = None) -> str:
        """Generate SVG header with proper viewBox and namespace."""
        title_elem = f'\n    <title>{escape(title)}</title>' if title else ''

        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 {self.width} {self.height}"
     xmlns="http://www.w3.org/2000/svg"
     style="width: 100%; height: 100vh; background: #000;"
     class="splat-svg"
     data-parallax-strength="{self.parallax_strength}"
     data-interactive-top="{self.interactive_top}">{title_elem}'''

    def _generate_defs(self, gaussian_mode: bool, layers: Dict[int, List[Gaussian]] = None) -> str:
        """Generate SVG definitions including gradients for gaussian mode."""
        if not gaussian_mode:
            return "    <defs></defs>"

        if not layers:
            # Fallback to simple approach
            return '''    <defs></defs>'''

        # Collect unique colors from all splats
        unique_colors = set()
        for layer_splats in layers.values():
            for splat in layer_splats:
                rgb = f"rgb({splat.r}, {splat.g}, {splat.b})"
                unique_colors.add(rgb)

        # Generate individual gradient for each color
        gradients = []
        for color in unique_colors:
            # Create safe ID from color
            color_id = color.replace("rgb(", "").replace(")", "").replace(", ", "_").replace(" ", "")
            gradient = f'''        <radialGradient id="grad_{color_id}" cx="50%" cy="50%" r="50%" gradientUnits="objectBoundingBox">
            <stop offset="0%" stop-color="{color}" stop-opacity="1"/>
            <stop offset="70%" stop-color="{color}" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
        </radialGradient>'''
            gradients.append(gradient)

        defs_content = "\n".join(gradients)
        return f'''    <defs>
{defs_content}
    </defs>'''

    def _generate_styles(self) -> str:
        """Generate CSS styles for layers and animation."""
        return '''    <style><![CDATA[
        .splat-svg {
            overflow: hidden;
            cursor: crosshair;
        }

        .layer {
            transform-style: preserve-3d;
            transition: transform 0.1s ease-out;
            will-change: transform;
        }

        .layer ellipse {
            vector-effect: non-scaling-stroke;
        }

        .interactive-splat {
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .interactive-splat:hover {
            transform: scale(1.1);
        }

        @media (prefers-reduced-motion: reduce) {
            .layer, .interactive-splat {
                transition: none !important;
                animation: none !important;
            }
        }

        @media (max-width: 768px) {
            .splat-svg {
                cursor: default;
            }
        }
    ]]></style>'''

    def _generate_scripts(self) -> str:
        """Generate JavaScript for parallax interaction."""
        return f'''    <script><![CDATA[
        (function() {{
            'use strict';

            // Check for reduced motion preference
            const prefersReducedMotion = window.matchMedia &&
                window.matchMedia('(prefers-reduced-motion: reduce)').matches;

            if (prefersReducedMotion) {{
                console.log('SplatThis: Animations disabled due to user preference');
                return;
            }}

            const svg = document.querySelector('.splat-svg');
            const layers = document.querySelectorAll('.layer');
            const parallaxStrength = {self.parallax_strength};
            const interactiveTop = {self.interactive_top};

            let isGyroSupported = false;
            let isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

            // Mouse parallax for desktop
            function handleMouseMove(e) {{
                if (isMobile || !svg) return;

                const rect = svg.getBoundingClientRect();
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;

                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                const deltaX = (mouseX - centerX) / centerX;
                const deltaY = (mouseY - centerY) / centerY;

                updateLayerTransforms(deltaX, deltaY);
            }}

            // Gyroscope parallax for mobile
            function handleDeviceOrientation(e) {{
                if (!isMobile || !isGyroSupported) return;

                const beta = e.beta || 0;   // X-axis rotation
                const gamma = e.gamma || 0; // Y-axis rotation

                const deltaX = Math.max(-1, Math.min(1, gamma / 30));
                const deltaY = Math.max(-1, Math.min(1, beta / 30));

                updateLayerTransforms(deltaX, deltaY);
            }}

            function updateLayerTransforms(deltaX, deltaY) {{
                layers.forEach((layer, index) => {{
                    const depth = parseFloat(layer.getAttribute('data-depth')) || 0.5;
                    const parallaxFactor = (depth - 0.6) * parallaxStrength;

                    const translateX = deltaX * parallaxFactor;
                    const translateY = deltaY * parallaxFactor;

                    layer.style.transform = `translate(${{translateX}}px, ${{translateY}}px)`;

                    // Add interactive effects for top splats
                    if (interactiveTop > 0 && index < interactiveTop) {{
                        layer.classList.add('interactive-splat');
                    }}
                }});
            }}

            // Initialize event listeners
            if (!isMobile) {{
                svg.addEventListener('mousemove', handleMouseMove);
                svg.addEventListener('mouseleave', () => {{
                    updateLayerTransforms(0, 0);
                }});
            }} else {{
                // Request gyroscope permission on iOS 13+
                if (typeof DeviceOrientationEvent !== 'undefined' &&
                    typeof DeviceOrientationEvent.requestPermission === 'function') {{
                    DeviceOrientationEvent.requestPermission()
                        .then(response => {{
                            if (response === 'granted') {{
                                isGyroSupported = true;
                                window.addEventListener('deviceorientation', handleDeviceOrientation);
                            }}
                        }})
                        .catch(console.error);
                }} else if (typeof DeviceOrientationEvent !== 'undefined') {{
                    isGyroSupported = true;
                    window.addEventListener('deviceorientation', handleDeviceOrientation);
                }}
            }}

            console.log(`SplatThis SVG loaded - ${{layers.length}} layers, parallax: ${{parallaxStrength}}`);
        }})();
    ]]></script>'''

    def _generate_footer(self) -> str:
        """Generate SVG footer."""
        return "</svg>"

    def _generate_empty_svg(self) -> str:
        """Generate empty SVG when no layers are provided."""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 {self.width} {self.height}"
     xmlns="http://www.w3.org/2000/svg"
     style="width: 100%; height: 100vh; background: #000;">
    <text x="{self.width//2}" y="{self.height//2}"
          text-anchor="middle" dominant-baseline="middle"
          fill="rgba(255,255,255,0.5)" font-family="sans-serif" font-size="24">
        No splats to display
    </text>
</svg>'''

    def _format_number(self, num: float) -> str:
        """Format number with specified precision."""
        return f"{num:.{self.precision}f}"

    def _validate_svg_structure(self, svg_content: str) -> bool:
        """Perform basic validation of SVG structure."""
        try:
            # Check for required elements
            required_elements = [
                '<?xml version="1.0"',
                '<svg',
                'viewBox=',
                'xmlns="http://www.w3.org/2000/svg"',
                '</svg>'
            ]

            for element in required_elements:
                if element not in svg_content:
                    logger.error(f"Missing required SVG element: {element}")
                    return False

            # Check for balanced tags (basic)
            svg_count = svg_content.count('<svg')
            svg_close_count = svg_content.count('</svg>')

            if svg_count != svg_close_count:
                logger.error(f"Unbalanced SVG tags: {svg_count} open, {svg_close_count} close")
                return False

            return True

        except Exception as e:
            logger.error(f"SVG validation error: {e}")
            return False

    @global_profiler.profile_function("svg_file_save")
    def save_svg(self, svg_content: str, output_path: Path) -> None:
        """Save SVG content to file with optimization."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use buffered writing for large SVG files
            with open(output_path, "w", encoding="utf-8", buffering=8192) as f:
                f.write(svg_content)

            logger.info(f"SVG saved successfully to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save SVG to {output_path}: {e}")
            raise

    def get_svg_info(self, layers: Dict[int, List[Gaussian]]) -> Dict[str, int]:
        """Get information about the SVG that would be generated."""
        total_splats = sum(len(layer_splats) for layer_splats in layers.values())

        return {
            'width': self.width,
            'height': self.height,
            'precision': self.precision,
            'layer_count': len(layers),
            'total_splats': total_splats,
            'parallax_strength': self.parallax_strength,
            'interactive_top': self.interactive_top,
            'estimated_size_kb': self._estimate_svg_size(total_splats),
        }

    def _estimate_svg_size(self, total_splats: int) -> int:
        """Estimate SVG file size in KB."""
        # Base SVG overhead
        base_size = 2  # KB

        # Per-splat overhead (ellipse element + attributes)
        per_splat = 0.15  # ~150 bytes per splat

        # Styles and scripts overhead
        script_overhead = 3  # KB

        total_kb = base_size + (total_splats * per_splat) + script_overhead
        return int(total_kb)