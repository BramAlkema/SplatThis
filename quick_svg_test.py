#!/usr/bin/env python3
"""
Quick SVG color test to check for black rendering issues
"""
import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright

async def quick_svg_test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1200, 'height': 800})
        page = await context.new_page()
        
        # Set shorter timeout
        page.set_default_timeout(10000)  # 10 seconds
        
        print("=== Quick SVG Color Test ===")
        
        # Test 1: Simple color test first
        test_html = """
<!DOCTYPE html>
<html>
<head>
    <style>body { margin: 20px; background: white; }</style>
</head>
<body>
    <h1>SVG Color Test</h1>
    <svg width="300" height="200" viewBox="0 0 300 200">
        <circle cx="50" cy="50" r="30" fill="red" opacity="0.8"/>
        <circle cx="150" cy="50" r="30" fill="green" opacity="0.8"/>
        <circle cx="250" cy="50" r="30" fill="blue" opacity="0.8"/>
        <circle cx="100" cy="120" r="30" fill="#ff6600" opacity="0.6"/>
        <circle cx="200" cy="120" r="30" fill="rgb(255,0,255)" opacity="0.6"/>
    </svg>
</body>
</html>
"""
        
        # Write and test simple HTML
        test_path = "/Users/ynse/projects/SplatThis/quick_color_test.html"
        with open(test_path, 'w') as f:
            f.write(test_html)
        
        await page.goto(f"file://{test_path}")
        await page.wait_for_load_state('domcontentloaded')
        
        await page.screenshot(path="/Users/ynse/projects/SplatThis/test_screenshots/quick_color_test.png")
        print("✓ Basic color test completed - check quick_color_test.png")
        
        # Test 2: Check one of the actual SVG files (smallest one first)
        svg_files = [
            "/Users/ynse/projects/SplatThis/intermediate_outputs/step3_initial_splats.svg",
        ]
        
        for svg_file in svg_files:
            if os.path.exists(svg_file):
                filename = Path(svg_file).stem
                print(f"\nTesting {filename}...")
                
                try:
                    await page.goto(f"file://{svg_file}")
                    await page.wait_for_load_state('domcontentloaded', timeout=5000)
                    
                    # Check SVG element properties
                    svg_element = await page.query_selector('svg')
                    if svg_element:
                        # Get basic info
                        viewbox = await svg_element.get_attribute('viewBox')
                        print(f"  ViewBox: {viewbox}")
                        
                        # Count circles
                        circles = await page.query_selector_all('circle')
                        print(f"  Circle count: {len(circles)}")
                        
                        # Sample first few circles for color info
                        if circles:
                            for i in range(min(3, len(circles))):
                                circle = circles[i]
                                fill = await circle.get_attribute('fill')
                                opacity = await circle.get_attribute('opacity')
                                print(f"    Circle {i}: fill={fill}, opacity={opacity}")
                        
                        # Take a smaller screenshot (not full page)
                        await page.set_viewport_size({'width': 800, 'height': 600})
                        screenshot_path = f"/Users/ynse/projects/SplatThis/test_screenshots/{filename}_quick.png"
                        await page.screenshot(path=screenshot_path)
                        print(f"✓ Screenshot saved: {filename}_quick.png")
                    else:
                        print(f"  ❌ No SVG element found")
                        
                except Exception as e:
                    print(f"  ❌ Error testing {filename}: {str(e)}")
        
        # Test 3: Visual inspection website
        print("\n=== Testing Visual Inspection Website ===")
        visual_path = "/Users/ynse/projects/SplatThis/e2e_results/final_visual_inspection.html"
        
        if os.path.exists(visual_path):
            try:
                await page.goto(f"file://{visual_path}")
                await page.wait_for_load_state('domcontentloaded', timeout=5000)
                
                # Check for SVG elements
                svg_elements = await page.query_selector_all('svg, object[type="image/svg+xml"], img[src$=".svg"]')
                print(f"Found {len(svg_elements)} SVG-related elements")
                
                # Take screenshot
                await page.screenshot(path="/Users/ynse/projects/SplatThis/test_screenshots/visual_inspection_quick.png")
                print("✓ Visual inspection screenshot saved")
                
            except Exception as e:
                print(f"❌ Error testing visual inspection: {str(e)}")
        
        await browser.close()
        print("\n=== Test Summary ===")
        print("Check test_screenshots/ for results:")
        print("- quick_color_test.png: Basic SVG color rendering test")
        print("- step3_initial_splats_quick.png: Actual SVG file rendering")
        print("- visual_inspection_quick.png: Website rendering")

if __name__ == "__main__":
    os.makedirs("/Users/ynse/projects/SplatThis/test_screenshots", exist_ok=True)
    asyncio.run(quick_svg_test())
