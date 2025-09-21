# SplatThis Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using SplatThis.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Image Loading Problems](#image-loading-problems)
3. [Processing Errors](#processing-errors)
4. [SVG Output Issues](#svg-output-issues)
5. [Performance Problems](#performance-problems)
6. [Browser Compatibility](#browser-compatibility)
7. [API Usage Issues](#api-usage-issues)
8. [Platform-Specific Issues](#platform-specific-issues)

## Installation Issues

### Problem: `pip install` fails

**Error Messages:**
```
ERROR: Could not find a version that satisfies the requirement splat-this
```

**Solutions:**
```bash
# Install from source (current method)
git clone https://github.com/BramAlkema/SplatThis.git
cd SplatThis
pip install -e ".[dev]"

# Verify Python version (requires 3.8+)
python --version

# Update pip
pip install --upgrade pip
```

### Problem: Missing dependencies

**Error Messages:**
```
ModuleNotFoundError: No module named 'skimage'
ImportError: No module named 'cv2'
```

**Solutions:**
```bash
# Install missing dependencies
pip install scikit-image>=0.19.0
pip install opencv-python  # Optional, for advanced features

# Or reinstall with all dependencies
pip install -e ".[dev]"

# Check installed packages
pip list | grep -E "(scikit-image|Pillow|numpy|click)"
```

### Problem: Permission errors on installation

**Error Messages:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Or install for user only
pip install --user -e ".[dev]"
```

## Image Loading Problems

### Problem: "File not found" error

**Error Messages:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'image.jpg'
```

**Solutions:**
```bash
# Check file exists
ls -la image.jpg

# Use absolute path
splatlify /full/path/to/image.jpg -o output.svg

# Check current directory
pwd
```

### Problem: Unsupported image format

**Error Messages:**
```
ValueError: Invalid image format or corrupted file
PIL.UnidentifiedImageError: cannot identify image file
```

**Solutions:**
```bash
# Check file format
file image.unknown

# Convert to supported format (JPG, PNG, GIF)
convert image.bmp image.jpg  # Using ImageMagick

# Verify image integrity
python -c "from PIL import Image; Image.open('image.jpg').verify()"
```

### Problem: Corrupted or incomplete image

**Error Messages:**
```
OSError: broken data stream when reading image file
ValueError: Invalid image array format or dimensions
```

**Solutions:**
```python
# Test image loading
from splat_this.utils.image import ImageLoader
loader = ImageLoader()

try:
    image = loader.load_image("problematic.jpg")
    print(f"Image loaded successfully: {image.shape}")
except Exception as e:
    print(f"Image loading failed: {e}")
    # Try with PIL directly
    from PIL import Image
    img = Image.open("problematic.jpg")
    print(f"PIL loaded: {img.size}, {img.mode}")
```

## Processing Errors

### Problem: Memory error with large images

**Error Messages:**
```
MemoryError: Unable to allocate array
RuntimeError: out of memory
```

**Solutions:**
```bash
# Reduce image size automatically
splatlify large_image.jpg --splats 1000 -o output.svg

# Manual resizing with Python
python -c "
from PIL import Image
img = Image.open('large.jpg')
img.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
img.save('resized.jpg')
"

# Use optimized components
python -c "
from splat_this.core.optimized_extract import OptimizedSplatExtractor
extractor = OptimizedSplatExtractor(max_size_limit=1280)
"
```

### Problem: Invalid splat parameters

**Error Messages:**
```
ValueError: Invalid radii: rx=-1.0, ry=5.0
ValueError: Invalid RGB: (256, 100, 50)
ValueError: Invalid alpha: 1.5
```

**Solutions:**
```python
# Debug splat extraction
from splat_this.core.extract import SplatExtractor

extractor = SplatExtractor()
try:
    splats = extractor.extract_splats(image, n_splats=1500)
    print(f"Extracted {len(splats)} valid splats")
except Exception as e:
    print(f"Extraction failed: {e}")

    # Try with more conservative settings
    splats = extractor.extract_splats(image, n_splats=500, min_size=10)
```

### Problem: No splats extracted

**Error Messages:**
```
RuntimeWarning: No valid splats extracted from image
```

**Solutions:**
```python
# Check image properties
import numpy as np
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Image range: {image.min()} - {image.max()}")

# Try different extraction parameters
extractor = SplatExtractor(
    n_segments=1000,    # Fewer segments
    compactness=5.0,    # Less compact
    sigma=2.0           # More smoothing
)
splats = extractor.extract_splats(image, n_splats=500, min_size=3)
```

## SVG Output Issues

### Problem: SVG file is empty or corrupted

**Error Messages:**
```
SVG file exists but shows no content
Browser shows "This XML file does not appear to have any style information"
```

**Solutions:**
```python
# Validate SVG generation
from splat_this.core.svgout import SVGGenerator

generator = SVGGenerator(800, 600)

# Check if layers are empty
if not layers:
    print("No layers provided - this will create empty SVG")

# Generate with debugging
svg_content = generator.generate_svg(layers, gaussian_mode=False)

# Check SVG content
print(f"SVG length: {len(svg_content)} characters")
print("SVG preview:", svg_content[:200])

# Validate basic structure
required_elements = ['<svg', '</svg>', 'viewBox', '<g']
for element in required_elements:
    if element not in svg_content:
        print(f"WARNING: Missing {element}")
```

### Problem: SVG displays but no animation

**Possible Causes:**
- Browser JavaScript disabled
- CSS/JS features not supported
- Motion preferences set to reduced

**Solutions:**
```bash
# Test with simpler output
splatlify image.jpg --parallax-strength 0 -o static.svg

# Check browser console for errors (F12 → Console)

# Test with different browsers
# Chrome, Firefox, Safari, Edge
```

### Problem: SVG too large or slow

**Error Messages:**
```
File size exceeds reasonable limits
Browser becomes unresponsive
```

**Solutions:**
```bash
# Reduce complexity
splatlify image.jpg --splats 800 --layers 3 -o smaller.svg

# Lower precision
splatlify image.jpg --precision 2 -o compact.svg

# Disable advanced features
splatlify image.jpg --parallax-strength 20 --interactive-top 0 -o simple.svg
```

## Performance Problems

### Problem: Processing takes too long

**Symptoms:**
- Command hangs for minutes
- High CPU usage
- No progress output

**Solutions:**
```bash
# Use verbose mode to see progress
splatlify image.jpg -v -o output.svg

# Use optimized components automatically
splatlify image.jpg --splats 1200 -o output.svg

# Reduce complexity
splatlify image.jpg --splats 800 --layers 3 -o faster.svg

# Check system resources
top -p $(pgrep -f splatlify)
```

### Problem: High memory usage

**Solutions:**
```python
# Monitor memory during processing
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Process with memory monitoring
check_memory()
# ... run splatlify ...
check_memory()

# Use memory-efficient settings
splatlify image.jpg --splats 1000 -o output.svg
```

### Problem: Optimization not working

**Error Messages:**
```
ImportError: cannot import name 'OptimizedSplatExtractor'
Performance seems same as standard components
```

**Solutions:**
```python
# Verify optimized components are available
try:
    from splat_this.core.optimized_extract import OptimizedSplatExtractor
    print("Optimized components available")
except ImportError as e:
    print(f"Optimized components not available: {e}")

# Check if optimization is being used
import logging
logging.basicConfig(level=logging.INFO)

# Should see optimization messages in logs
```

## Browser Compatibility

### Problem: SVG works in Chrome but not Safari

**Solutions:**
```python
# Generate with maximum compatibility
generator = SVGGenerator(
    width=800,
    height=600,
    precision=2,
    parallax_strength=30
)

svg_content = generator.generate_svg(
    layers,
    gaussian_mode=False,  # Use simple circles
    title="Cross-browser SVG"
)
```

### Problem: No parallax effect on mobile

**Expected Behavior:**
- Mouse parallax disabled on mobile
- Gyroscope parallax should work

**Solutions:**
```bash
# Test gyroscope support
# Open SVG on mobile device
# Tilt device to see motion

# Check if device motion is enabled in browser settings
```

### Problem: SVG not displaying in email

**Solutions:**
```bash
# Email clients have limited SVG support
# Consider generating static version for email
splatlify image.jpg --parallax-strength 0 --interactive-top 0 -o email.svg

# Or convert to image
# Use external tools to convert SVG to PNG for email
```

## API Usage Issues

### Problem: Import errors

**Error Messages:**
```
ImportError: cannot import name 'SplatExtractor'
ModuleNotFoundError: No module named 'splat_this'
```

**Solutions:**
```python
# Check installation
import sys
print(sys.path)

# Try different import styles
try:
    from splat_this.core.extract import SplatExtractor
except ImportError:
    try:
        from splat_this import SplatExtractor
    except ImportError as e:
        print(f"Import failed: {e}")

# Verify package installation
import pkg_resources
try:
    pkg_resources.get_distribution('splat-this')
    print("Package is installed")
except pkg_resources.DistributionNotFound:
    print("Package not found - reinstall required")
```

### Problem: Type errors with numpy arrays

**Error Messages:**
```
TypeError: 'int' object has no attribute 'shape'
ValueError: Input arrays must be numpy arrays
```

**Solutions:**
```python
# Ensure proper numpy array format
import numpy as np
from PIL import Image

# Convert PIL to numpy correctly
pil_image = Image.open("image.jpg")
if pil_image.mode != 'RGB':
    pil_image = pil_image.convert('RGB')
image = np.array(pil_image)

# Validate array format
print(f"Shape: {image.shape}")
print(f"Dtype: {image.dtype}")
print(f"Range: {image.min()}-{image.max()}")

# Should be (height, width, 3) with dtype uint8 and range 0-255
```

## Platform-Specific Issues

### Windows Issues

**Problem: Path separators**
```bash
# Use forward slashes or raw strings
splatlify "C:/Users/Name/image.jpg" -o "C:/Users/Name/output.svg"

# Or use pathlib
python -c "
from pathlib import Path
input_path = Path('C:/Users/Name/image.jpg')
output_path = Path('C:/Users/Name/output.svg')
"
```

**Problem: Virtual environment activation**
```cmd
# Use .bat file on Windows
venv\Scripts\activate.bat

# Or use PowerShell
venv\Scripts\Activate.ps1
```

### macOS Issues

**Problem: Permission denied for image files**
```bash
# Check file permissions
ls -la image.jpg

# Fix permissions
chmod 644 image.jpg

# Grant terminal access to files if needed
# System Preferences → Security & Privacy → Privacy → Files and Folders
```

### Linux Issues

**Problem: Missing system dependencies**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# For image processing libraries
sudo apt-get install libjpeg-dev libpng-dev

# CentOS/RHEL
sudo yum install python3-devel python3-pip
sudo yum install libjpeg-devel libpng-devel
```

## Getting Help

### Collecting Debug Information

```python
def collect_debug_info():
    """Collect system information for bug reports."""
    import sys
    import platform
    import numpy as np
    from PIL import Image

    print("System Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Pillow: {Image.__version__}")

    try:
        import skimage
        print(f"scikit-image: {skimage.__version__}")
    except ImportError:
        print("scikit-image: NOT INSTALLED")

    try:
        from splat_this import __version__
        print(f"SplatThis: {__version__}")
    except ImportError:
        print("SplatThis: NOT INSTALLED")

# Run this and include output in bug reports
collect_debug_info()
```

### Enabling Debug Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now run your SplatThis commands with full logging
```

### Performance Profiling

```python
from splat_this.utils.profiler import PerformanceProfiler
import cProfile

# Method 1: Using built-in profiler
profiler = PerformanceProfiler()
# ... run your code ...
stats = profiler.get_stats()
print(stats)

# Method 2: Using cProfile
cProfile.run('your_splatlify_code_here()', 'profile_output.prof')

# Analyze with
# python -m pstats profile_output.prof
```

If you're still experiencing issues after trying these solutions, please create an issue on the [GitHub repository](https://github.com/BramAlkema/SplatThis/issues) with:

1. Your system information (from `collect_debug_info()`)
2. Complete error messages
3. Steps to reproduce the issue
4. Sample image (if possible)
5. Command line or code you're using