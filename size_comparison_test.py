#!/usr/bin/env python3
"""
Test to confirm the splat size issue by creating comparison SVGs
"""
import re

# Read the original SVG
with open('/Users/ynse/projects/SplatThis/intermediate_outputs/step3_initial_splats.svg', 'r') as f:
    original_content = f.read()

# Create a version with scaled-up splats (multiply all rx/ry by 10)
def scale_splat_sizes(content, scale_factor=10):
    def replace_ellipse(match):
        full_match = match.group(0)
        rx = float(match.group(1))
        ry = float(match.group(2))
        
        # Scale up the radii
        new_rx = rx * scale_factor
        new_ry = ry * scale_factor
        
        # Replace the rx and ry values
        result = re.sub(r'rx="[^"]*"', f'rx="{new_rx:.3f}"', full_match)
        result = re.sub(r'ry="[^"]*"', f'ry="{new_ry:.3f}"', result)
        
        return result
    
    # Find and replace all ellipse elements
    scaled_content = re.sub(
        r'<ellipse[^>]*rx="([^"]*?)"[^>]*ry="([^"]*?)"[^>]*/>',
        replace_ellipse,
        content
    )
    
    return scaled_content

# Create scaled version
scaled_content = scale_splat_sizes(original_content, scale_factor=10)

# Save the scaled version
with open('/Users/ynse/projects/SplatThis/test_scaled_splats.svg', 'w') as f:
    f.write(scaled_content)

print("Created test_scaled_splats.svg with 10x larger splats")

# Also create a minimal test with just a few large splats
minimal_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg" 
     style="width: 100%; height: 100vh; background: #000;">
    <defs>
        <radialGradient id="grad_red" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="rgb(255, 100, 100)" stop-opacity="1"/>
            <stop offset="70%" stop-color="rgb(255, 100, 100)" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="rgb(255, 100, 100)" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="grad_blue" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="rgb(100, 150, 255)" stop-opacity="1"/>
            <stop offset="70%" stop-color="rgb(100, 150, 255)" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="rgb(100, 150, 255)" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="grad_green" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="rgb(100, 255, 100)" stop-opacity="1"/>
            <stop offset="70%" stop-color="rgb(100, 255, 100)" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="rgb(100, 255, 100)" stop-opacity="0"/>
        </radialGradient>
    </defs>
    
    <!-- Large visible splats for testing -->
    <ellipse cx="128" cy="128" rx="20" ry="20" fill="url(#grad_red)" fill-opacity="0.8"/>
    <ellipse cx="256" cy="256" rx="25" ry="15" fill="url(#grad_blue)" fill-opacity="0.7" transform="rotate(45 256 256)"/>
    <ellipse cx="384" cy="128" rx="18" ry="30" fill="url(#grad_green)" fill-opacity="0.9" transform="rotate(30 384 128)"/>
    <ellipse cx="128" cy="384" rx="15" ry="25" fill="url(#grad_red)" fill-opacity="0.6" transform="rotate(-30 128 384)"/>
    <ellipse cx="384" cy="384" rx="22" ry="18" fill="url(#grad_blue)" fill-opacity="0.8" transform="rotate(60 384 384)"/>
    
    <!-- Show what tiny splats look like -->
    <ellipse cx="200" cy="50" rx="0.5" ry="0.5" fill="url(#grad_red)" fill-opacity="0.8"/>
    <ellipse cx="210" cy="50" rx="1" ry="1" fill="url(#grad_blue)" fill-opacity="0.8"/>
    <ellipse cx="220" cy="50" rx="2" ry="2" fill="url(#grad_green)" fill-opacity="0.8"/>
    <ellipse cx="230" cy="50" rx="3" ry="3" fill="url(#grad_red)" fill-opacity="0.8"/>
    <ellipse cx="240" cy="50" rx="5" ry="5" fill="url(#grad_blue)" fill-opacity="0.8"/>
    
</svg>'''

with open('/Users/ynse/projects/SplatThis/test_minimal_splats.svg', 'w') as f:
    f.write(minimal_svg)

print("Created test_minimal_splats.svg with size comparison")
