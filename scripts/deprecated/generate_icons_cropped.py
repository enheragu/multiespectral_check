#!/usr/bin/env python3
"""
Generate icon variants from logo.png with cropped margins (no padding).
Crops to content bounding box, then regenerates all sizes and .ico format.
"""
import os
import sys
from pathlib import Path
from PIL import Image

# Setup paths
repo_root = Path(__file__).parent.parent
media_dir = repo_root / "src/frontend/resources/media"
os.chdir(repo_root)

# Load source PNG
source_path = media_dir / "logo.png"
print(f"Loading {source_path}...")
img = Image.open(source_path).convert('RGBA')
original_size = img.size
print(f"  Original size: {original_size}")

# Crop to bounding box (remove margins)
bbox = img.getbbox()
if bbox:
    x0, y0, x1, y1 = bbox
    content_width = x1 - x0
    content_height = y1 - y0
    print(f"  Bounding box: ({x0}, {y0}) to ({x1}, {y1})")
    print(f"  Content size: {content_width}x{content_height}")
    
    # Crop to content
    img_cropped = img.crop(bbox)
    
    # Make square (if not already) with small uniform padding for breathing room
    max_dim = max(content_width, content_height)
    square = Image.new('RGBA', (max_dim, max_dim), (0, 0, 0, 0))
    
    # Center content in square
    offset_x = (max_dim - content_width) // 2
    offset_y = (max_dim - content_height) // 2
    square.paste(img_cropped, (offset_x, offset_y), img_cropped)
    
    # Scale up to 1250x1250 for consistency with original dimensions
    img_final = square.resize((1250, 1250), Image.Resampling.LANCZOS)
    print(f"  Rescaled to: {img_final.size}")
    
    # Save as new source
    source_backup = media_dir / "logo_original_padded.png"
    if not source_backup.exists():
        img.save(source_backup)
        print(f"  Backed up original to {source_backup.name}")
    
    img.save(source_path)
    print(f"  Saved cropped logo to {source_path.name}")
else:
    print("  ERROR: Could not determine bounding box!")
    sys.exit(1)

# Now generate all icon sizes from cropped version
sizes = [16, 24, 32, 48, 64, 128, 256, 512]
print("\nGenerating PNG variants...")

for size in sizes:
    resized = img_final.resize((size, size), Image.Resampling.LANCZOS)
    png_path = media_dir / f"logo_{size}x{size}.png"
    resized.save(png_path)
    print(f"  {png_path.name}")

# Generate .ico from the cropped version (multiple sizes)
print("\nGenerating logo.ico...")
ico_sizes = [16, 24, 32, 48, 64, 128, 256]
ico_images = [
    img_final.resize((size, size), Image.Resampling.LANCZOS)
    for size in ico_sizes
]
ico_path = media_dir / "logo.ico"
ico_images[0].save(ico_path, format='ICO', sizes=[(size, size) for size in ico_sizes], append_images=ico_images[1:])
print(f"  {ico_path.name} ({', '.join(map(str, ico_sizes))} px)")

print("\n✓ Icon generation complete!")
print("  Icons are loaded from disk via frontend.resources.icon_path —")
print("  no qrc/rcc step needed. Just commit the regenerated files.")
