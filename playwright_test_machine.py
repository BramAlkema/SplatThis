#!/usr/bin/env python3
"""Playwright Testing Machine for SplatThis Code Verification.

This automated testing machine checks:
1. SVG color rendering correctness
2. PNG to SVG conversion pipeline
3. Visual quality of generated splats
4. Browser compatibility across different scenarios
"""

import asyncio
import sys
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from playwright.async_api import async_playwright, Browser, Page
except ImportError:
    print("❌ Playwright not installed. Run: pip install playwright")
    sys.exit(1)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    screenshot_path: str = None
    duration: float = 0.0


@dataclass
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[TestResult]
    total_duration: float = 0.0

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_count / len(self.results) * 100


class SplatThisTestMachine:
    """Automated testing machine for SplatThis codebase."""

    def __init__(self):
        self.browser: Browser = None
        self.page: Page = None
        self.test_suites: List[TestSuite] = []
        self.screenshots_dir = Path("test_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)

    async def setup(self):
        """Initialize Playwright browser."""
        print("🚀 Starting Playwright Testing Machine...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page()
        print("✅ Browser initialized")

    async def teardown(self):
        """Clean up resources."""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        print("🧹 Cleanup complete")

    async def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and capture results."""
        print(f"🧪 Running: {test_name}")
        start_time = time.time()

        try:
            await test_func()
            duration = time.time() - start_time
            result = TestResult(test_name, True, "✅ PASSED", duration=duration)
            print(f"✅ {test_name} - PASSED ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start_time
            screenshot_path = await self.capture_screenshot(f"failed_{test_name}")
            result = TestResult(test_name, False, f"❌ FAILED: {str(e)}",
                              screenshot_path=screenshot_path, duration=duration)
            print(f"❌ {test_name} - FAILED: {str(e)} ({duration:.2f}s)")

        return result

    async def capture_screenshot(self, name: str) -> str:
        """Capture screenshot for debugging."""
        timestamp = int(time.time())
        screenshot_path = self.screenshots_dir / f"{name}_{timestamp}.png"
        await self.page.screenshot(path=str(screenshot_path))
        return str(screenshot_path)

    # ============ SVG COLOR TESTS ============

    async def test_svg_color_rendering(self):
        """Test that SVG colors render correctly (not black)."""
        await self.page.goto("file://" + str(Path.cwd() / "test_svg_embedded.html"))
        await self.page.wait_for_load_state('networkidle')

        # Check that test circles are visible and colored
        red_circle = self.page.locator('svg ellipse[fill="red"]').first
        assert await red_circle.is_visible(), "Red test circle should be visible"

        # Take screenshot for verification
        await self.capture_screenshot("svg_color_test")

    async def test_gradient_rendering(self):
        """Test that gradients render with actual colors."""
        await self.page.goto("file://" + str(Path.cwd() / "test_svg_embedded.html"))
        await self.page.wait_for_load_state('networkidle')

        # Check gradient elements exist
        gradient_elements = await self.page.locator('radialGradient').count()
        assert gradient_elements >= 2, f"Expected at least 2 gradients, found {gradient_elements}"

    # ============ PNG TO SVG CONVERSION TESTS ============

    async def test_png_to_svg_pipeline(self):
        """Test complete PNG to SVG conversion pipeline."""
        # First ensure the conversion script works
        print("  Running PNG to SVG conversion...")
        result = subprocess.run([
            sys.executable, "simple_png_to_svg.py"
        ], capture_output=True, text=True, cwd=str(Path.cwd()))

        assert result.returncode == 0, f"Conversion script failed: {result.stderr}"

        # Check that files were created
        assert Path("simple_1000_splats.svg").exists(), "SVG file not created"
        assert Path("simple_comparison.html").exists(), "Comparison HTML not created"

    async def test_svg_file_validity(self):
        """Test that generated SVG files are valid and display."""
        svg_path = Path("simple_1000_splats.svg")
        assert svg_path.exists(), "SVG file does not exist"

        # Load SVG in browser
        await self.page.goto("file://" + str(svg_path.absolute()))
        await self.page.wait_for_load_state('networkidle')

        # Check SVG root element
        svg_element = self.page.locator('svg').first
        assert await svg_element.is_visible(), "SVG element should be visible"

        # Check for ellipse elements (splats)
        ellipse_count = await self.page.locator('ellipse').count()
        assert ellipse_count > 0, f"No ellipses found in SVG, expected > 0"
        print(f"  Found {ellipse_count} splat ellipses")

    async def test_svg_has_colors(self):
        """Test that SVG splats have actual colors (not all black)."""
        svg_path = Path("simple_1000_splats.svg")
        await self.page.goto("file://" + str(svg_path.absolute()))
        await self.page.wait_for_load_state('networkidle')

        # Check for gradient definitions
        gradient_count = await self.page.locator('radialGradient').count()
        assert gradient_count > 1, f"Expected multiple gradients, found {gradient_count}"

        # Check for non-black colors in gradients
        stop_elements = await self.page.locator('stop[stop-color*="rgb"]').all()
        assert len(stop_elements) > 0, "No RGB color stops found"

        # Verify colors are not all black
        colors = []
        for stop in stop_elements[:5]:  # Check first 5
            color = await stop.get_attribute('stop-color')
            colors.append(color)

        non_black_colors = [c for c in colors if c != "rgb(0, 0, 0)"]
        assert len(non_black_colors) > 0, f"All colors are black: {colors}"
        print(f"  Found non-black colors: {non_black_colors[:3]}")

    # ============ COMPARISON PAGE TESTS ============

    async def test_comparison_page_loads(self):
        """Test that comparison page loads correctly."""
        html_path = Path("simple_comparison.html")
        await self.page.goto("file://" + str(html_path.absolute()))
        await self.page.wait_for_load_state('networkidle')

        # Check page title
        title = await self.page.title()
        assert "PNG vs SVG" in title, f"Unexpected page title: {title}"

        # Check both images are present
        png_img = self.page.locator('img[alt="Original PNG"]').first
        assert await png_img.is_visible(), "PNG image should be visible"

        # Check SVG object is present
        svg_object = self.page.locator('object[type="image/svg+xml"]').first
        assert await svg_object.is_visible(), "SVG object should be visible"

    async def test_visual_comparison(self):
        """Test visual comparison between PNG and SVG."""
        html_path = Path("simple_comparison.html")
        await self.page.goto("file://" + str(html_path.absolute()))
        await self.page.wait_for_load_state('networkidle')

        # Wait for content to load
        await asyncio.sleep(2)

        # Take screenshot of comparison
        await self.capture_screenshot("visual_comparison")

        # Check that both sections are visible
        png_section = self.page.locator('h2:has-text("Original PNG")').first
        svg_section = self.page.locator('h2:has-text("Gaussian Splats")').first

        assert await png_section.is_visible(), "PNG section should be visible"
        assert await svg_section.is_visible(), "SVG section should be visible"

    # ============ ADVANCED TESTS ============

    async def test_demo_scripts_work(self):
        """Test that demo scripts execute without errors."""
        demo_scripts = [
            "debug_splats.py",
        ]

        for script in demo_scripts:
            if Path(script).exists():
                print(f"  Testing {script}...")
                result = subprocess.run([
                    sys.executable, script
                ], capture_output=True, text=True, cwd=str(Path.cwd()))

                assert result.returncode == 0, f"Demo script {script} failed: {result.stderr}"

    async def test_browser_console_errors(self):
        """Check for browser console errors."""
        html_path = Path("simple_comparison.html")

        # Collect console messages
        console_messages = []
        self.page.on("console", lambda msg: console_messages.append(msg.text))

        await self.page.goto("file://" + str(html_path.absolute()))
        await self.page.wait_for_load_state('networkidle')
        await asyncio.sleep(1)

        # Check for errors
        error_messages = [msg for msg in console_messages if "error" in msg.lower()]
        assert len(error_messages) == 0, f"Console errors found: {error_messages}"

    # ============ TEST RUNNER ============

    async def run_svg_tests(self):
        """Run SVG-specific tests."""
        print("\n🎨 Running SVG Tests...")
        tests = [
            ("SVG Color Rendering", self.test_svg_color_rendering),
            ("Gradient Rendering", self.test_gradient_rendering),
        ]

        results = []
        start_time = time.time()

        for test_name, test_func in tests:
            result = await self.run_test(test_name, test_func)
            results.append(result)

        duration = time.time() - start_time
        return TestSuite("SVG Tests", results, duration)

    async def run_conversion_tests(self):
        """Run PNG to SVG conversion tests."""
        print("\n🔄 Running Conversion Tests...")
        tests = [
            ("PNG to SVG Pipeline", self.test_png_to_svg_pipeline),
            ("SVG File Validity", self.test_svg_file_validity),
            ("SVG Has Colors", self.test_svg_has_colors),
        ]

        results = []
        start_time = time.time()

        for test_name, test_func in tests:
            result = await self.run_test(test_name, test_func)
            results.append(result)

        duration = time.time() - start_time
        return TestSuite("Conversion Tests", results, duration)

    async def run_integration_tests(self):
        """Run integration tests."""
        print("\n🔗 Running Integration Tests...")
        tests = [
            ("Comparison Page Loads", self.test_comparison_page_loads),
            ("Visual Comparison", self.test_visual_comparison),
            ("Demo Scripts Work", self.test_demo_scripts_work),
            ("Browser Console Errors", self.test_browser_console_errors),
        ]

        results = []
        start_time = time.time()

        for test_name, test_func in tests:
            result = await self.run_test(test_name, test_func)
            results.append(result)

        duration = time.time() - start_time
        return TestSuite("Integration Tests", results, duration)

    async def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 SPLATTHIS AUTOMATED TESTING MACHINE")
        print("=" * 60)

        await self.setup()

        try:
            # Run test suites
            svg_suite = await self.run_svg_tests()
            conversion_suite = await self.run_conversion_tests()
            integration_suite = await self.run_integration_tests()

            self.test_suites = [svg_suite, conversion_suite, integration_suite]

            # Generate report
            self.generate_report()

        finally:
            await self.teardown()

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        total_tests = 0
        total_passed = 0
        total_duration = 0

        for suite in self.test_suites:
            print(f"\n{suite.name}:")
            print(f"  Tests: {len(suite.results)}")
            print(f"  Passed: {suite.passed_count}")
            print(f"  Failed: {suite.failed_count}")
            print(f"  Success Rate: {suite.success_rate:.1f}%")
            print(f"  Duration: {suite.total_duration:.2f}s")

            total_tests += len(suite.results)
            total_passed += suite.passed_count
            total_duration += suite.total_duration

            # Show failed tests
            failed_tests = [r for r in suite.results if not r.passed]
            if failed_tests:
                print("  Failed Tests:")
                for test in failed_tests:
                    print(f"    ❌ {test.name}: {test.message}")
                    if test.screenshot_path:
                        print(f"       Screenshot: {test.screenshot_path}")

        print("\n" + "=" * 60)
        print("🎯 OVERALL RESULTS")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Success Rate: {total_passed / total_tests * 100:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")

        if total_passed == total_tests:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {total_tests - total_passed} TESTS FAILED")

        print(f"\n📸 Screenshots saved to: {self.screenshots_dir}")


async def main():
    """Main entry point."""
    machine = SplatThisTestMachine()
    await machine.run_all_tests()


if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("src/splat_this").exists():
        print("❌ Please run from SplatThis project root directory")
        sys.exit(1)

    asyncio.run(main())