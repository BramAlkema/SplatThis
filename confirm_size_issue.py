#!/usr/bin/env python3
"""
Test to confirm the splat size issue
"""
import asyncio
import os
from playwright.async_api import async_playwright

async def test_size_comparison():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 800, 'height': 600})
        page = await context.new_page()
        page.set_default_timeout(5000)
        
        test_files = [
            ("/Users/ynse/projects/SplatThis/test_minimal_splats.svg", "minimal_splats_comparison"),
            ("/Users/ynse/projects/SplatThis/test_scaled_splats.svg", "scaled_up_original"),
            ("/Users/ynse/projects/SplatThis/intermediate_outputs/step3_initial_splats.svg", "original_tiny_splats")
        ]
        
        for svg_file, name in test_files:
            if os.path.exists(svg_file):
                print(f"Testing {name}...")
                try:
                    await page.goto(f"file://{svg_file}")
                    await page.wait_for_load_state('domcontentloaded')
                    
                    screenshot_path = f"/Users/ynse/projects/SplatThis/test_screenshots/{name}.png"
                    await page.screenshot(path=screenshot_path)
                    print(f"✓ Screenshot saved: {name}.png")
                    
                except Exception as e:
                    print(f"❌ Error testing {name}: {e}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_size_comparison())
