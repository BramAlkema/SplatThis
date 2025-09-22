import asyncio
from playwright.async_api import async_playwright
import os
from pathlib import Path

async def test_svg_rendering():
    """Test SVG rendering to verify colors are displaying properly (not black)"""
    
    async with async_playwright() as p:
        # Launch browser with visible mode for debugging
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Test 1: Load the final visual inspection HTML page
        html_path = Path("/Users/ynse/projects/SplatThis/e2e_results/final_visual_inspection.html").resolve()
        print(f"Loading HTML file: {html_path}")
        
        await page.goto(f"file://{html_path}")
        await page.wait_for_load_state('networkidle')
        
        # Wait for SVGs to load
        await page.wait_for_timeout(3000)
        
        # Take full page screenshot
        await page.screenshot(path='svg_rendering_full_page.png', full_page=True)
        print("✅ Full page screenshot taken: svg_rendering_full_page.png")
        
        # Test 2: Check each SVG individually
        svg_files = [
            '/Users/ynse/projects/SplatThis/intermediate_outputs/step3_initial_splats.svg',
            '/Users/ynse/projects/SplatThis/intermediate_outputs/step4_refinement.svg', 
            '/Users/ynse/projects/SplatThis/intermediate_outputs/step5_scale_optimization.svg',
            '/Users/ynse/projects/SplatThis/intermediate_outputs/step6_final_output.svg'
        ]
        
        for i, svg_path in enumerate(svg_files, 1):
            print(f"\nTesting SVG {i}: {Path(svg_path).name}")
            
            # Navigate directly to the SVG file
            await page.goto(f"file://{svg_path}")
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(2000)
            
            # Take screenshot of individual SVG
            screenshot_name = f'svg_individual_{i}_{Path(svg_path).stem}.png'
            await page.screenshot(path=screenshot_name)
            print(f"✅ Individual SVG screenshot: {screenshot_name}")
            
            # Check if SVG content is loaded by evaluating SVG elements
            svg_elements = await page.query_selector_all('circle, ellipse, rect, path')
            print(f"   Found {len(svg_elements)} SVG elements")
            
            # Get SVG dimensions
            svg_element = await page.query_selector('svg')
            if svg_element:
                bbox = await svg_element.bounding_box()
                if bbox:
                    print(f"   SVG dimensions: {bbox['width']}x{bbox['height']}")
        
        # Test 3: Create a simple test page to verify individual SVG rendering
        test_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SVG Rendering Test</title>
    <style>
        body {{ margin: 20px; background: white; }}
        .svg-container {{ 
            margin: 20px 0; 
            border: 2px solid #ccc; 
            padding: 20px;
            background: #f0f0f0;
        }}
        h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>SVG Color Rendering Test</h1>
    <div class="svg-container">
        <h2>Step 3: Initial Splats</h2>
        <object data="file:///Users/ynse/projects/SplatThis/intermediate_outputs/step3_initial_splats.svg" 
                type="image/svg+xml" width="400" height="400"></object>
    </div>
    <div class="svg-container">
        <h2>Step 6: Final Output</h2>
        <object data="file:///Users/ynse/projects/SplatThis/intermediate_outputs/step6_final_output.svg" 
                type="image/svg+xml" width="400" height="400"></object>
    </div>
</body>
</html>
"""
        
        # Write test HTML to a temporary file
        test_html_path = '/Users/ynse/projects/SplatThis/svg_test_page.html'
        with open(test_html_path, 'w') as f:
            f.write(test_html)
        
        print(f"\nTesting custom SVG test page...")
        await page.goto(f"file://{test_html_path}")
        await page.wait_for_load_state('networkidle')
        await page.wait_for_timeout(3000)
        
        await page.screenshot(path='svg_custom_test_page.png', full_page=True)
        print("✅ Custom test page screenshot: svg_custom_test_page.png")
        
        # Test 4: Examine the SVG content directly
        print(f"\nAnalyzing SVG structure...")
        
        # Read one SVG file to check its structure
        sample_svg = svg_files[0]
        with open(sample_svg, 'r') as f:
            svg_content = f.read()
        
        # Count color elements
        import re
        color_patterns = re.findall(r'fill="[^"]*"', svg_content)
        non_black_colors = [c for c in color_patterns if 'rgb(' in c and c != 'fill="rgb(0,0,0)"']
        
        print(f"   Total fill attributes: {len(color_patterns)}")
        print(f"   Non-black colors: {len(non_black_colors)}")
        
        if non_black_colors:
            print(f"   Sample colors: {non_black_colors[:5]}")
        
        # Check for circle/ellipse elements with size info
        circles = re.findall(r'<(?:circle|ellipse)[^>]+r[xy]?="([^"]+)"', svg_content)
        if circles:
            circle_sizes = [float(c) for c in circles[:10] if c.replace('.', '').isdigit()]
            print(f"   Circle/ellipse sizes (first 10): {circle_sizes}")
            print(f"   Size range: {min(circle_sizes):.2f} - {max(circle_sizes):.2f}")
        
        await browser.close()
        
        # Clean up test file
        os.remove(test_html_path)
        
        print(f"\n🎉 SVG rendering test completed!")
        print(f"Screenshots saved to current directory")
        
        return {
            'html_loaded': True,
            'svg_files_tested': len(svg_files),
            'total_colors': len(color_patterns),
            'non_black_colors': len(non_black_colors),
            'size_range': f"{min(circle_sizes):.2f} - {max(circle_sizes):.2f}" if circles else "N/A"
        }

if __name__ == "__main__":
    result = asyncio.run(test_svg_rendering())
    print(f"\nTest Results: {result}")
