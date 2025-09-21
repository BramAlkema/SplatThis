"""Optimized command-line interface for SplatThis with performance monitoring."""

import sys
import time
import logging
from pathlib import Path

import click

from .utils.image import load_image, validate_image_dimensions
from .utils.profiler import global_profiler, MemoryEfficientProcessor, estimate_memory_usage
from .core.optimized_extract import OptimizedSplatExtractor
from .core.layering import ImportanceScorer, LayerAssigner, QualityController
from .core.optimized_svgout import OptimizedSVGGenerator


class OptimizedProgressBar:
    """Enhanced progress bar with performance monitoring."""

    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []

    def update(self, step_name: str) -> None:
        """Update progress bar with current step and performance info."""
        current_time = time.time()
        if self.step_times:
            step_duration = current_time - self.step_times[-1]
        else:
            step_duration = current_time - self.start_time

        self.step_times.append(current_time)
        self.current_step += 1
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = current_time - self.start_time

        # Enhanced progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        # Add step timing info
        if self.current_step > 1:
            step_info = f" ({step_duration:.1f}s)"
        else:
            step_info = ""

        click.echo(
            f"\r{self.description}: [{bar}] {percentage:.1f}% - {step_name}{step_info}",
            nl=False,
        )

        if self.current_step == self.total_steps:
            click.echo(f" ✓ Complete ({elapsed:.1f}s)")


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--frame", default=0, help="GIF frame number (default: 0)", type=int)
@click.option(
    "--splats",
    default=1500,
    help="Target splat count (default: 1500)",
    type=click.IntRange(50, 10000),
)
@click.option(
    "--layers",
    default=5,
    help="Number of depth layers (default: 5)",
    type=click.IntRange(2, 10),
)
@click.option(
    "--k",
    default=2.5,
    help="Splat size multiplier (default: 2.5)",
    type=click.FloatRange(1.0, 5.0),
)
@click.option(
    "--alpha",
    default=0.65,
    help="Base alpha transparency (default: 0.65)",
    type=click.FloatRange(0.1, 1.0),
)
@click.option(
    "--parallax-strength",
    default=40,
    help="Parallax animation strength (default: 40)",
    type=click.IntRange(0, 100),
)
@click.option(
    "--interactive-top",
    default=0,
    help="Number of interactive top splats (default: 0)",
    type=click.IntRange(0, 50),
)
@click.option(
    "--gaussian", is_flag=True, help="Use gradient mode for higher fidelity"
)
@click.option(
    "--verbose", is_flag=True, help="Enable verbose output with performance metrics"
)
@click.option(
    "--max-memory",
    default=1024,
    help="Maximum memory usage in MB (default: 1024)",
    type=click.IntRange(256, 8192),
)
@click.option(
    "--profile",
    is_flag=True,
    help="Enable detailed performance profiling"
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output SVG file path",
)
def main(
    input_file: Path,
    output: Path,
    frame: int,
    splats: int,
    layers: int,
    k: float,
    alpha: float,
    parallax_strength: int,
    interactive_top: int,
    gaussian: bool,
    verbose: bool,
    max_memory: int,
    profile: bool,
) -> None:
    """Convert image to parallax-animated SVG splats with performance optimization.

    Examples:
        splatlify photo.jpg -o parallax.svg --profile
        splatlify animation.gif --frame 5 --splats 2000 -o output.svg --max-memory 2048
        splatlify image.png --gaussian --layers 6 -o high-quality.svg --verbose
    """
    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    # Display banner
    click.echo("🎨 SplatThis v0.1.0 - Image to Parallax SVG Converter (Optimized)")
    click.echo()

    # Initialize memory processor
    memory_processor = MemoryEfficientProcessor(max_memory_mb=max_memory)

    # Pre-flight checks
    if verbose:
        estimated_memory = estimate_memory_usage(splats, layers)
        click.echo(f"📊 Pre-flight checks:")
        click.echo(f"   Estimated memory usage: {estimated_memory:.0f}MB")
        click.echo(f"   Memory limit: {max_memory}MB")

        if estimated_memory > max_memory * 0.8:
            click.echo(f"⚠️  Warning: Estimated memory usage is high!", err=True)
            if not click.confirm("Continue anyway?"):
                raise click.Abort()
        click.echo()

    # Validate output path
    if output.exists():
        if not click.confirm(f"Output file {output} exists. Overwrite?"):
            click.echo("Aborted.")
            return

    try:
        # Initialize progress bar
        progress = OptimizedProgressBar(6, "Converting")

        # Step 1: Load and validate image
        progress.update("Loading image")
        image, dimensions = load_image(input_file, frame=frame)
        validate_image_dimensions(dimensions)

        if verbose:
            click.echo(f"📷 Loading image:")
            click.echo(f"   File: {input_file}")
            click.echo(f"   Dimensions: {dimensions[0]}×{dimensions[1]}")
            click.echo(f"   Frame: {frame}")

        memory_processor.ensure_memory_limit("post_image_load")

        # Check if image should be downsampled
        should_downsample, new_size = memory_processor.should_downsample_image(
            (dimensions[0], dimensions[1]), splats
        )

        if should_downsample and verbose:
            click.echo(f"🔧 Image will be downsampled to {new_size[0]}×{new_size[1]} for memory efficiency")

        # Step 2: Extract splats using optimized extractor
        progress.update("Extracting splats")
        extractor = OptimizedSplatExtractor(k=k, base_alpha=alpha)
        extracted_splats = extractor.extract_splats(image, splats)

        if verbose:
            click.echo(f"🎯 Extracting splats:")
            click.echo(f"   Target count: {splats}")
            click.echo(f"   Extracted: {len(extracted_splats)}")
            click.echo(f"   K factor: {k}")
            click.echo(f"   Base alpha: {alpha}")

        memory_processor.ensure_memory_limit("post_extraction")

        # Step 3: Score splats for importance
        progress.update("Scoring importance")
        scorer = ImportanceScorer(area_weight=0.3, edge_weight=0.5, color_weight=0.2)
        scorer.score_splats(extracted_splats, image)

        if verbose:
            scores = [s.score for s in extracted_splats]
            click.echo(f"📈 Importance scoring:")
            click.echo(f"   Score range: {min(scores):.3f} - {max(scores):.3f}")
            click.echo(f"   Average score: {sum(scores)/len(scores):.3f}")

        # Step 4: Quality control and filtering
        progress.update("Applying quality control")
        controller = QualityController(
            target_count=min(splats, len(extracted_splats)),
            k_multiplier=k,
            alpha_adjustment=True
        )
        final_splats = controller.optimize_splats(extracted_splats)

        if verbose:
            click.echo(f"🎛️  Quality control:")
            click.echo(f"   Original splats: {len(extracted_splats)}")
            click.echo(f"   Final splat count: {len(final_splats)}")
            click.echo(f"   Reduction: {(1 - len(final_splats)/len(extracted_splats))*100:.1f}%")

        memory_processor.ensure_memory_limit("post_quality_control")

        # Step 5: Assign depth layers
        progress.update("Assigning layers")
        layer_assigner = LayerAssigner(n_layers=layers)
        layer_data = layer_assigner.assign_layers(final_splats)

        if verbose:
            click.echo(f"🏗️  Layer assignment:")
            for layer_idx, layer_splats in layer_data.items():
                if layer_splats:
                    avg_depth = sum(s.depth for s in layer_splats) / len(layer_splats)
                    click.echo(f"   Layer {layer_idx}: {len(layer_splats)} splats (depth: {avg_depth:.2f})")

        # Step 6: Generate optimized SVG
        progress.update("Generating SVG")
        generator = OptimizedSVGGenerator(
            width=dimensions[0],
            height=dimensions[1],
            parallax_strength=parallax_strength,
            interactive_top=interactive_top,
        )

        svg_content = generator.generate_svg(
            layer_data, gaussian_mode=gaussian, title=f"SplatThis: {input_file.name}"
        )

        memory_processor.ensure_memory_limit("post_svg_generation")

        # Save output with optimized I/O
        generator.save_svg(svg_content, output)

        # Final statistics
        file_size = output.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        click.echo(f"\n✅ Successfully created {output}")
        click.echo("📊 Final statistics:")
        click.echo(f"   Splats: {len(final_splats)}")
        click.echo(f"   Layers: {len(layer_data)}")
        click.echo(f"   File size: {file_size_mb:.2f} MB")

        if verbose:
            click.echo(f"   Mode: {'Gradient' if gaussian else 'Solid'}")
            click.echo(f"   Parallax strength: {parallax_strength}px")
            click.echo(f"   Interactive splats: {interactive_top}")
            click.echo(f"   Memory limit: {max_memory}MB")

        # Show performance profile if requested
        if profile:
            click.echo()
            global_profiler.print_summary("Performance Profile")

    except Exception as e:
        progress.update("Error")
        click.echo(f"\n❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()

        if profile:
            click.echo()
            global_profiler.print_summary("Performance Profile (Error)")

        sys.exit(1)


if __name__ == "__main__":
    main()