"""Unit tests for CLI functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import numpy as np

from splat_this.cli import main, ProgressBar
from splat_this.core.extract import Gaussian


class TestProgressBar:
    """Test progress bar functionality."""

    def test_progress_bar_initialization(self):
        """Test progress bar initialization."""
        progress = ProgressBar(5, "Testing")
        assert progress.total_steps == 5
        assert progress.current_step == 0
        assert progress.description == "Testing"

    @patch('splat_this.cli.click.echo')
    def test_progress_bar_update(self, mock_echo):
        """Test progress bar update functionality."""
        progress = ProgressBar(3, "Testing")

        # First update
        progress.update("Step 1")
        assert progress.current_step == 1
        mock_echo.assert_called()

        # Check that the call contains progress info
        call_args = mock_echo.call_args[0][0]
        assert "Testing:" in call_args
        assert "33.3%" in call_args
        assert "Step 1" in call_args

    @patch('splat_this.cli.click.echo')
    def test_progress_bar_completion(self, mock_echo):
        """Test progress bar completion."""
        progress = ProgressBar(2, "Testing")

        progress.update("Step 1")
        progress.update("Step 2")

        assert progress.current_step == 2
        # Should have completion message
        call_args = mock_echo.call_args[0][0]
        assert "✓ Complete" in call_args


class TestCLIMain:
    """Test main CLI functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

        # Create mock image data
        self.mock_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        self.mock_dimensions = (100, 150)  # height, width

        # Create mock splats
        self.mock_splats = [
            Gaussian(x=50, y=40, rx=5, ry=5, theta=0, r=255, g=100, b=50, a=0.8),
            Gaussian(x=75, y=60, rx=8, ry=6, theta=0.5, r=100, g=200, b=150, a=0.7),
        ]

        # Mock layer data
        self.mock_layers = {0: [self.mock_splats[0]], 1: [self.mock_splats[1]]}

    def _setup_svg_generator_mock(self, mock_svg_gen):
        """Helper to set up SVG generator mock with file creation."""
        mock_generator_instance = Mock()
        mock_generator_instance.generate_svg.return_value = "<svg>test</svg>"

        # Create a side effect for save_svg that actually creates the file
        def save_svg_side_effect(content, path):
            path.write_text(content)
        mock_generator_instance.save_svg.side_effect = save_svg_side_effect

        mock_svg_gen.return_value = mock_generator_instance
        return mock_generator_instance

    def test_cli_help(self):
        """Test CLI help output."""
        result = self.runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert "Convert image to parallax-animated SVG splats" in result.output
        assert "--splats" in result.output
        assert "--layers" in result.output
        assert "--gaussian" in result.output
        assert "--verbose" in result.output

    def test_cli_missing_required_args(self):
        """Test CLI with missing required arguments."""
        result = self.runner.invoke(main, [])

        assert result.exit_code != 0
        assert "Usage:" in result.output

    def test_cli_missing_output(self):
        """Test CLI with missing output argument."""
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
            result = self.runner.invoke(main, [temp_file.name])

            assert result.exit_code != 0
            assert "Missing option" in result.output or "Error" in result.output

    def test_cli_nonexistent_input_file(self):
        """Test CLI with nonexistent input file."""
        result = self.runner.invoke(main, [
            'nonexistent_file.jpg',
            '-o', 'output.svg'
        ])

        assert result.exit_code != 0

    @patch('splat_this.cli.load_image')
    @patch('splat_this.cli.validate_image_dimensions')
    @patch('splat_this.cli.SplatExtractor')
    @patch('splat_this.cli.ImportanceScorer')
    @patch('splat_this.cli.QualityController')
    @patch('splat_this.cli.LayerAssigner')
    @patch('splat_this.cli.SVGGenerator')
    def test_cli_successful_execution(self, mock_svg_gen, mock_layer_assigner,
                                     mock_quality_controller, mock_scorer,
                                     mock_extractor, mock_validate, mock_load_image):
        """Test successful CLI execution with mocked components."""

        # Setup mocks
        mock_load_image.return_value = (self.mock_image, self.mock_dimensions)
        mock_validate.return_value = None

        # Mock extractor
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract_splats.return_value = self.mock_splats
        mock_extractor.return_value = mock_extractor_instance

        # Mock scorer
        mock_scorer_instance = Mock()
        mock_scorer.return_value = mock_scorer_instance

        # Mock quality controller
        mock_controller_instance = Mock()
        mock_controller_instance.optimize_splats.return_value = self.mock_splats
        mock_quality_controller.return_value = mock_controller_instance

        # Mock layer assigner
        mock_assigner_instance = Mock()
        mock_assigner_instance.assign_layers.return_value = self.mock_layers
        mock_layer_assigner.return_value = mock_assigner_instance

        # Mock SVG generator
        mock_generator_instance = self._setup_svg_generator_mock(mock_svg_gen)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.jpg"
            input_file.touch()  # Create empty file
            output_file = Path(temp_dir) / "output.svg"

            result = self.runner.invoke(main, [
                str(input_file),
                '-o', str(output_file)
            ])

            assert result.exit_code == 0
            assert "Successfully created" in result.output
            assert "Final statistics:" in result.output

            # Verify mocks were called
            mock_load_image.assert_called_once()
            mock_extractor_instance.extract_splats.assert_called_once()
            mock_scorer_instance.score_splats.assert_called_once()
            mock_controller_instance.optimize_splats.assert_called_once()
            mock_assigner_instance.assign_layers.assert_called_once()
            mock_generator_instance.generate_svg.assert_called_once()
            mock_generator_instance.save_svg.assert_called_once()

    @patch('splat_this.cli.load_image')
    @patch('splat_this.cli.validate_image_dimensions')
    @patch('splat_this.cli.SplatExtractor')
    @patch('splat_this.cli.ImportanceScorer')
    @patch('splat_this.cli.QualityController')
    @patch('splat_this.cli.LayerAssigner')
    @patch('splat_this.cli.SVGGenerator')
    def test_cli_verbose_mode(self, mock_svg_gen, mock_layer_assigner,
                             mock_quality_controller, mock_scorer,
                             mock_extractor, mock_validate, mock_load_image):
        """Test CLI verbose mode output."""

        # Setup mocks (same as successful execution)
        mock_load_image.return_value = (self.mock_image, self.mock_dimensions)
        mock_validate.return_value = None

        mock_extractor_instance = Mock()
        mock_extractor_instance.extract_splats.return_value = self.mock_splats
        mock_extractor.return_value = mock_extractor_instance

        mock_scorer_instance = Mock()
        mock_scorer.return_value = mock_scorer_instance

        mock_controller_instance = Mock()
        mock_controller_instance.optimize_splats.return_value = self.mock_splats
        mock_quality_controller.return_value = mock_controller_instance

        mock_assigner_instance = Mock()
        mock_assigner_instance.assign_layers.return_value = self.mock_layers
        mock_layer_assigner.return_value = mock_assigner_instance

        mock_generator_instance = self._setup_svg_generator_mock(mock_svg_gen)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.jpg"
            input_file.touch()
            output_file = Path(temp_dir) / "output.svg"

            result = self.runner.invoke(main, [
                str(input_file),
                '-o', str(output_file),
                '--verbose'
            ])

            assert result.exit_code == 0
            assert "Loading image:" in result.output
            assert "Image dimensions:" in result.output
            assert "Extracted" in result.output
            assert "Final splat count:" in result.output
            assert "Mode:" in result.output

    @patch('splat_this.cli.load_image')
    @patch('splat_this.cli.validate_image_dimensions')
    @patch('splat_this.cli.SplatExtractor')
    @patch('splat_this.cli.ImportanceScorer')
    @patch('splat_this.cli.QualityController')
    @patch('splat_this.cli.LayerAssigner')
    @patch('splat_this.cli.SVGGenerator')
    def test_cli_gaussian_mode(self, mock_svg_gen, mock_layer_assigner,
                              mock_quality_controller, mock_scorer,
                              mock_extractor, mock_validate, mock_load_image):
        """Test CLI with gaussian mode enabled."""

        # Setup mocks
        mock_load_image.return_value = (self.mock_image, self.mock_dimensions)
        mock_validate.return_value = None

        mock_extractor_instance = Mock()
        mock_extractor_instance.extract_splats.return_value = self.mock_splats
        mock_extractor.return_value = mock_extractor_instance

        mock_scorer_instance = Mock()
        mock_scorer.return_value = mock_scorer_instance

        mock_controller_instance = Mock()
        mock_controller_instance.optimize_splats.return_value = self.mock_splats
        mock_quality_controller.return_value = mock_controller_instance

        mock_assigner_instance = Mock()
        mock_assigner_instance.assign_layers.return_value = self.mock_layers
        mock_layer_assigner.return_value = mock_assigner_instance

        mock_generator_instance = self._setup_svg_generator_mock(mock_svg_gen)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.jpg"
            input_file.touch()
            output_file = Path(temp_dir) / "output.svg"

            result = self.runner.invoke(main, [
                str(input_file),
                '-o', str(output_file),
                '--gaussian'
            ])

            assert result.exit_code == 0

            # Verify gaussian mode was passed to generate_svg
            mock_generator_instance.generate_svg.assert_called_once()
            call_kwargs = mock_generator_instance.generate_svg.call_args[1]
            assert call_kwargs['gaussian_mode'] is True

    @patch('splat_this.cli.load_image')
    @patch('splat_this.cli.validate_image_dimensions')
    @patch('splat_this.cli.SplatExtractor')
    @patch('splat_this.cli.ImportanceScorer')
    @patch('splat_this.cli.QualityController')
    @patch('splat_this.cli.LayerAssigner')
    @patch('splat_this.cli.SVGGenerator')
    def test_cli_custom_parameters(self, mock_svg_gen, mock_layer_assigner,
                                  mock_quality_controller, mock_scorer,
                                  mock_extractor, mock_validate, mock_load_image):
        """Test CLI with custom parameters."""

        # Setup mocks
        mock_load_image.return_value = (self.mock_image, self.mock_dimensions)
        mock_validate.return_value = None

        mock_extractor_instance = Mock()
        mock_extractor_instance.extract_splats.return_value = self.mock_splats
        mock_extractor.return_value = mock_extractor_instance

        mock_scorer_instance = Mock()
        mock_scorer.return_value = mock_scorer_instance

        mock_controller_instance = Mock()
        mock_controller_instance.optimize_splats.return_value = self.mock_splats
        mock_quality_controller.return_value = mock_controller_instance

        mock_assigner_instance = Mock()
        mock_assigner_instance.assign_layers.return_value = self.mock_layers
        mock_layer_assigner.return_value = mock_assigner_instance

        mock_generator_instance = self._setup_svg_generator_mock(mock_svg_gen)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.jpg"
            input_file.touch()
            output_file = Path(temp_dir) / "output.svg"

            result = self.runner.invoke(main, [
                str(input_file),
                '-o', str(output_file),
                '--splats', '2000',
                '--layers', '6',
                '--k', '3.0',
                '--alpha', '0.8',
                '--parallax-strength', '60',
                '--interactive-top', '5'
            ])

            assert result.exit_code == 0

            # Verify parameters were passed correctly
            mock_extractor.assert_called_with(k=3.0, base_alpha=0.8)
            mock_quality_controller.assert_called_with(target_count=2000, k_multiplier=3.0)
            mock_layer_assigner.assert_called_with(n_layers=6)

            # Check SVG generation parameters
            call_kwargs = mock_generator_instance.generate_svg.call_args[1]
            assert call_kwargs['parallax_strength'] == 60

    def test_cli_parameter_validation(self):
        """Test CLI parameter validation."""
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
            # Test invalid splat count
            result = self.runner.invoke(main, [
                temp_file.name,
                '-o', 'output.svg',
                '--splats', '50'  # Below minimum
            ])
            assert result.exit_code != 0

            # Test invalid layer count
            result = self.runner.invoke(main, [
                temp_file.name,
                '-o', 'output.svg',
                '--layers', '1'  # Below minimum
            ])
            assert result.exit_code != 0

            # Test invalid k value
            result = self.runner.invoke(main, [
                temp_file.name,
                '-o', 'output.svg',
                '--k', '0.5'  # Below minimum
            ])
            assert result.exit_code != 0

    @patch('splat_this.cli.click.confirm')
    def test_cli_overwrite_confirmation(self, mock_confirm):
        """Test CLI overwrite confirmation."""
        mock_confirm.return_value = False  # User says no

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.jpg"
            input_file.touch()
            output_file = Path(temp_dir) / "output.svg"
            output_file.touch()  # File already exists

            result = self.runner.invoke(main, [
                str(input_file),
                '-o', str(output_file)
            ])

            assert result.exit_code == 0
            assert "Aborted" in result.output
            mock_confirm.assert_called_once()

    @patch('splat_this.cli.load_image')
    def test_cli_error_handling(self, mock_load_image):
        """Test CLI error handling."""
        # Mock an exception during image loading
        mock_load_image.side_effect = Exception("Test error")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.jpg"
            input_file.touch()
            output_file = Path(temp_dir) / "output.svg"

            result = self.runner.invoke(main, [
                str(input_file),
                '-o', str(output_file)
            ])

            assert result.exit_code == 1
            assert "❌ Error:" in result.output

    @patch('splat_this.cli.load_image')
    @patch('traceback.print_exc')
    def test_cli_verbose_error_handling(self, mock_traceback, mock_load_image):
        """Test CLI error handling in verbose mode."""
        mock_load_image.side_effect = Exception("Test error")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.jpg"
            input_file.touch()
            output_file = Path(temp_dir) / "output.svg"

            result = self.runner.invoke(main, [
                str(input_file),
                '-o', str(output_file),
                '--verbose'
            ])

            assert result.exit_code == 1
            assert "❌ Error:" in result.output
            mock_traceback.assert_called_once()

    @patch('splat_this.cli.load_image')
    def test_cli_banner_display(self, mock_load_image):
        """Test CLI banner display."""
        # Mock load_image to fail quickly so we can see the banner
        mock_load_image.side_effect = Exception("Test error")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.jpg"
            input_file.touch()  # Create valid file so Click validation passes
            output_file = Path(temp_dir) / "output.svg"

            result = self.runner.invoke(main, [
                str(input_file),
                '-o', str(output_file)
            ])

            assert "🎨 SplatThis v0.1.0" in result.output
            assert "Image to Parallax SVG Converter" in result.output

    @patch('splat_this.cli.load_image')
    @patch('splat_this.cli.validate_image_dimensions')
    @patch('splat_this.cli.SplatExtractor')
    @patch('splat_this.cli.ImportanceScorer')
    @patch('splat_this.cli.QualityController')
    @patch('splat_this.cli.LayerAssigner')
    @patch('splat_this.cli.SVGGenerator')
    def test_cli_gif_frame_parameter(self, mock_svg_gen, mock_layer_assigner,
                                    mock_quality_controller, mock_scorer,
                                    mock_extractor, mock_validate, mock_load_image):
        """Test CLI with GIF frame parameter."""

        # Setup mocks
        mock_load_image.return_value = (self.mock_image, self.mock_dimensions)
        mock_validate.return_value = None

        mock_extractor_instance = Mock()
        mock_extractor_instance.extract_splats.return_value = self.mock_splats
        mock_extractor.return_value = mock_extractor_instance

        mock_scorer_instance = Mock()
        mock_scorer.return_value = mock_scorer_instance

        mock_controller_instance = Mock()
        mock_controller_instance.optimize_splats.return_value = self.mock_splats
        mock_quality_controller.return_value = mock_controller_instance

        mock_assigner_instance = Mock()
        mock_assigner_instance.assign_layers.return_value = self.mock_layers
        mock_layer_assigner.return_value = mock_assigner_instance

        mock_generator_instance = self._setup_svg_generator_mock(mock_svg_gen)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.gif"
            input_file.touch()
            output_file = Path(temp_dir) / "output.svg"

            result = self.runner.invoke(main, [
                str(input_file),
                '-o', str(output_file),
                '--frame', '3'
            ])

            assert result.exit_code == 0

            # Verify frame parameter was passed to load_image
            mock_load_image.assert_called_once_with(input_file, 3)

    def test_cli_output_directory_creation(self):
        """Test that CLI creates output directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "test.jpg"
            input_file.touch()

            # Output in nested directory that doesn't exist
            output_file = Path(temp_dir) / "nested" / "dir" / "output.svg"

            # Mock load_image to fail early, but after directory creation
            with patch('splat_this.cli.load_image', side_effect=Exception("Test")):
                result = self.runner.invoke(main, [
                    str(input_file),
                    '-o', str(output_file)
                ])

                # Directory should be created even though execution failed
                assert output_file.parent.exists()

    def test_cli_statistics_display(self):
        """Test CLI statistics display."""
        with patch('splat_this.cli.load_image') as mock_load_image, \
             patch('splat_this.cli.validate_image_dimensions'), \
             patch('splat_this.cli.SplatExtractor') as mock_extractor, \
             patch('splat_this.cli.ImportanceScorer') as mock_scorer, \
             patch('splat_this.cli.QualityController') as mock_controller, \
             patch('splat_this.cli.LayerAssigner') as mock_assigner, \
             patch('splat_this.cli.SVGGenerator') as mock_svg_gen:

            # Setup mocks
            mock_load_image.return_value = (self.mock_image, self.mock_dimensions)

            mock_extractor_instance = Mock()
            mock_extractor_instance.extract_splats.return_value = self.mock_splats
            mock_extractor.return_value = mock_extractor_instance

            mock_scorer_instance = Mock()
            mock_scorer.return_value = mock_scorer_instance

            mock_controller_instance = Mock()
            mock_controller_instance.optimize_splats.return_value = self.mock_splats
            mock_controller.return_value = mock_controller_instance

            mock_assigner_instance = Mock()
            mock_assigner_instance.assign_layers.return_value = self.mock_layers
            mock_assigner.return_value = mock_assigner_instance

            mock_generator_instance = self._setup_svg_generator_mock(mock_svg_gen)

            with tempfile.TemporaryDirectory() as temp_dir:
                input_file = Path(temp_dir) / "test.jpg"
                input_file.touch()
                output_file = Path(temp_dir) / "output.svg"

                result = self.runner.invoke(main, [
                    str(input_file),
                    '-o', str(output_file)
                ])

                assert result.exit_code == 0
                assert "📊 Final statistics:" in result.output
                assert "Splats:" in result.output
                assert "Layers:" in result.output
                assert "File size:" in result.output