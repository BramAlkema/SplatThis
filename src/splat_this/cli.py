"""Command-line interface for SplatThis."""

import sys
import time
from pathlib import Path

import click

from .utils.image import load_image, validate_image_dimensions
from .core.extract import SplatExtractor
from .core.layering import ImportanceScorer, LayerAssigner, QualityController
from .core.svgout import SVGGenerator


class ProgressBar:
    """Simple progress bar for CLI operations."""

    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()

    def update(self, step_name: str) -> None:
        """Update progress bar with current step."""
        self.current_step += 1
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time

        # Simple progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        click.echo(
            f"\r{self.description}: [{bar}] {percentage:.1f}% - {step_name}",
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
    type=click.IntRange(100, 10000),
)
@click.option(
    "--layers",
    default=4,
    help="Depth layers (default: 4)",
    type=click.IntRange(2, 8),
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
    help="Base alpha (default: 0.65)",
    type=click.FloatRange(0.1, 1.0),
)
@click.option(
    "--parallax-strength",
    default=40,
    help="Parallax strength (default: 40)",
    type=click.IntRange(0, 200),
)
@click.option(
    "--interactive-top",
    default=0,
    help="Interactive splats (default: 0)",
    type=click.IntRange(0, 5000),
)
@click.option("--gaussian", is_flag=True, help="Enable gradient mode")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
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
) -> None:
    """Convert image to parallax-animated SVG splats.

    Examples:
        splatlify photo.jpg -o parallax.svg
        splatlify animation.gif --frame 5 --splats 2000 -o output.svg
        splatlify image.png --gaussian --layers 6 -o high-quality.svg
    """
    # Display banner
    click.echo("🎨 SplatThis v0.1.0 - Image to Parallax SVG Converter")
    click.echo()

    # Validate output path
    if output.exists():
        if not click.confirm(f"Output file {output} exists. Overwrite?"):
            click.echo("Aborted.")
            return

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize progress tracking
    total_steps = 6  # Load, Extract, Score, Layer, Generate, Save
    progress = ProgressBar(total_steps, "Converting")

    try:
        # Step 1: Load image
        progress.update("Loading image")
        if verbose:
            click.echo(f"\nLoading image: {input_file}")

        image, (height, width) = load_image(input_file, frame)
        validate_image_dimensions(image)

        if verbose:
            click.echo(f"Image dimensions: {width}x{height}")
            click.echo(f"Image size: {image.nbytes / (1024*1024):.1f} MB")

        # Step 2: Extract splats
        progress.update("Extracting splats")
        extractor = SplatExtractor(k=k, base_alpha=alpha)
        raw_splats = extractor.extract_splats(image, splats)

        if verbose:
            click.echo(f"Extracted {len(raw_splats)} raw splats")

        # Step 3: Score splats
        progress.update("Scoring importance")
        scorer = ImportanceScorer()
        scorer.score_splats(raw_splats, image)

        # Step 4: Assign layers and filter
        progress.update("Assigning layers")
        controller = QualityController(target_count=splats, k_multiplier=k)
        final_splats = controller.optimize_splats(raw_splats)

        assigner = LayerAssigner(n_layers=layers)
        layer_data = assigner.assign_layers(final_splats)

        if verbose:
            click.echo(f"Final splat count: {len(final_splats)}")

        # Step 5: Generate SVG
        progress.update("Generating SVG")
        generator = SVGGenerator(
            width,
            height,
            parallax_strength=parallax_strength,
            interactive_top=interactive_top
        )

        svg_content = generator.generate_svg(layer_data, gaussian_mode=gaussian)

        # Step 6: Save output
        progress.update("Saving file")
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

    except Exception as e:
        progress.update("Error")
        click.echo(f"\n❌ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
