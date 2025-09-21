"""SVG generation with inline animation."""

import math
import logging
from typing import Dict, List, Optional
from pathlib import Path
from xml.sax.saxutils import escape

if False:  # TYPE_CHECKING
    from .extract import Gaussian
else:
    from .extract import Gaussian

logger = logging.getLogger(__name__)


class SVGGenerator:
    """Generate animated SVG from layered splats."""

    def __init__(
        self,
        width: int,
        height: int,
        precision: int = 3,
        parallax_strength: int = 40,
        interactive_top: int = 0,
    ):
        self.width = width
        self.height = height
        self.precision = precision
        self.parallax_strength = parallax_strength
        self.interactive_top = interactive_top
        self._gradient_counter = 0
        self._gradient_defs: List[str] = []

    def generate_svg(
        self,
        layers: Dict[int, List["Gaussian"]],
        gaussian_mode: bool = False,
        title: Optional[str] = None,
    ) -> str:
        """Generate complete SVG with layers and animation."""
        if not layers:
            logger.warning("No layers provided for SVG generation")
            return self._generate_empty_svg()

        logger.info(f"Generating SVG with {len(layers)} layers, "
                   f"gaussian_mode={gaussian_mode}, size={self.width}×{self.height}")

        self._reset_gradient_registry()

        # Generate SVG components
        header = self._generate_header(title)
        layer_groups = self._generate_layer_groups(layers, gaussian_mode)
        defs = self._generate_defs(gaussian_mode)
        styles = self._generate_styles()
        scripts = self._generate_scripts()
        footer = self._generate_footer()

        # Combine all parts
        svg_content = f"{header}\n{defs}\n{layer_groups}\n{styles}\n{scripts}\n{footer}"

        # Validate basic structure
        if not self._validate_svg_structure(svg_content):
            logger.error("Generated SVG failed basic validation")
            raise ValueError("Invalid SVG structure generated")

        return svg_content

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

    def _reset_gradient_registry(self) -> None:
        """Reset gradient definitions for a new SVG generation."""
        self._gradient_counter = 0
        self._gradient_defs = []

    def _register_gradient(self, splat: "Gaussian") -> str:
        """Register a gaussian gradient for the given splat and return its ID."""
        gradient_id = f"gaussian-gradient-{self._gradient_counter}"
        self._gradient_counter += 1

        color = f"rgb({splat.r}, {splat.g}, {splat.b})"
        gradient_def = (
            "        <radialGradient id=\"{gradient_id}\" cx=\"50%\" cy=\"50%\" r=\"50%\" "
            "gradientUnits=\"objectBoundingBox\">\n"
            "            <stop offset=\"0%\" stop-color=\"{color}\" stop-opacity=\"1\"/>\n"
            "            <stop offset=\"70%\" stop-color=\"{color}\" stop-opacity=\"0.7\"/>\n"
            "            <stop offset=\"100%\" stop-color=\"{color}\" stop-opacity=\"0\"/>\n"
            "        </radialGradient>"
        ).format(gradient_id=gradient_id, color=color)

        self._gradient_defs.append(gradient_def)
        return gradient_id

    def _generate_defs(self, gaussian_mode: bool) -> str:
        """Generate SVG definitions including gradients for gaussian mode."""
        if not gaussian_mode or not self._gradient_defs:
            return "    <defs></defs>"

        gradient_defs = "\n".join(self._gradient_defs)
        return f"    <defs>\n{gradient_defs}\n    </defs>"

    def _generate_layer_groups(self, layers: Dict[int, List["Gaussian"]], gaussian_mode: bool) -> str:
        """Generate layer groups with proper depth attributes."""
        layer_groups = []

        # Sort layers by depth (back to front)
        sorted_layers = sorted(layers.items(), key=lambda x: x[0])

        for layer_idx, splats in sorted_layers:
            if not splats:
                continue

            # Calculate depth value for this layer
            depth = splats[0].depth if splats else 0.5

            # Generate group header
            group_header = f'    <g class="layer" data-depth="{self._format_number(depth)}" data-layer="{layer_idx}">'

            # Generate splats for this layer
            splat_elements = []
            for splat in splats:
                splat_svg = self._generate_splat_element(splat, gaussian_mode)
                splat_elements.append(f"        {splat_svg}")

            # Generate group footer
            group_footer = "    </g>"

            # Combine layer
            layer_content = f"{group_header}\n" + "\n".join(splat_elements) + f"\n{group_footer}"
            layer_groups.append(layer_content)

        return "\n".join(layer_groups)

    def _generate_splat_element(self, splat: "Gaussian", gaussian_mode: bool) -> str:
        """Generate SVG ellipse element for a single splat."""
        # Format coordinates and dimensions with precision
        cx = self._format_number(splat.x)
        cy = self._format_number(splat.y)
        rx = self._format_number(splat.rx)
        ry = self._format_number(splat.ry)

        # Convert rotation from degrees to radians for transform
        rotation_deg = self._format_number(math.degrees(splat.theta))

        # Format alpha
        alpha = self._format_number(splat.a)

        # Create rotation transform if needed
        transform = f' transform="rotate({rotation_deg} {cx} {cy})"' if abs(splat.theta) > 1e-6 else ''

        if gaussian_mode:
            gradient_id = self._register_gradient(splat)
            fill = f'url(#{gradient_id})'
            style = f'fill: {fill}; fill-opacity: {alpha}; stroke: none;'

            ellipse = f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"'
            ellipse += f' style="{style}"'
        else:
            # Use solid color fill
            fill = f'rgba({splat.r}, {splat.g}, {splat.b}, {alpha})'
            style = f'fill: {fill}; stroke: none;'
            ellipse = f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"'
            ellipse += f' style="{style}"'

        if transform:
            ellipse += transform

        ellipse += '/>'

        return ellipse

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

    def save_svg(self, svg_content: str, output_path: Path) -> None:
        """Save SVG content to file with validation."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(svg_content)

            logger.info(f"SVG saved successfully to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save SVG to {output_path}: {e}")
            raise

    def get_svg_info(self, layers: Dict[int, List["Gaussian"]]) -> Dict[str, int]:
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
        }
