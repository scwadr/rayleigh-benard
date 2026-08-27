"""
Stitch PNG frames into an MP4 video (Debug Version).

Usage:
    stitch_video_debug.py <input_dir> <output_file> [--fps=<fps>]

Options:
    --fps=<fps>     Frames per second [default: 10]
"""

import imageio.v2 as imageio
import pathlib
import sys
from docopt import docopt

def main():
    args = docopt(__doc__)
    input_dir = pathlib.Path(args['<input_dir>']).absolute()
    output_str = args['<output_file>']
    fps = int(args['--fps'])
    
    # Force .png extension for Animated PNG (APNG)
    # You can also change this to .webp if you prefer smaller file sizes
    output_path = pathlib.Path(output_str)
    if output_path.suffix.lower() not in ['.png', '.webp']:
        print(f"Warning: Changing extension from {output_path.suffix} to .png for High Quality.")
        output_path = output_path.with_suffix('.png')

    print(f"--- Stitching High-Quality Animation ---")
    print(f"Source: {input_dir}")
    print(f"Target: {output_path} (Format: {output_path.suffix[1:].upper()})")

    frames = sorted(list(input_dir.glob('*.png')))
    if not frames:
        print("Error: No PNG files found.")
        sys.exit(1)

    print(f"Found {len(frames)} frames.")
    
    # Read images
    images = []
    for i, filename in enumerate(frames):
        if i % 10 == 0:
            print(f"  Reading {i}/{len(frames)}")
        images.append(imageio.imread(filename))

    print("Saving... (This might take a moment for large files)")
    
    # Save as APNG or WebP (Both support TrueColor)
    # duration is in milliseconds for some backends, or seconds for others. 
    # ImageIO v2 uses 'duration' in seconds (per frame) standardly for these plugins.
    imageio.mimsave(output_path, images, fps=fps, loop=0)
    
    print("-" * 40)
    print("Done! To view this file:")
    print("  > Drag and drop it into a Web Browser (Chrome/Firefox)")
    print("  > Or open with a viewer that supports APNG/WebP")
    print("-" * 40)

if __name__ == "__main__":
    main()