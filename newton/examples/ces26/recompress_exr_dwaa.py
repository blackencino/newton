# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Recompress EXR files with DWAA compression.

Scans a directory for EXR files, opens each one, and saves it back out
with DWAA compression for much smaller file sizes.

Usage: uv run python newton/examples/ces26/recompress_exr_dwaa.py
"""

import sys
from pathlib import Path

import OpenEXR


def recompress_exr_dwaa(input_path: Path, output_path: Path | None = None) -> None:
    """
    Recompress an EXR file with DWAA compression.
    
    Args:
        input_path: Path to input EXR file
        output_path: Path to output EXR file (defaults to overwriting input)
    """
    if output_path is None:
        output_path = input_path
    
    # Read the EXR file
    exr_file = OpenEXR.File(str(input_path))
    header = dict(exr_file.header())
    channels = exr_file.channels()
    
    # Update compression to DWAA
    header["compression"] = OpenEXR.DWAA_COMPRESSION
    
    # Write out with new compression
    new_file = OpenEXR.File(header, channels)
    new_file.write(str(output_path))


def main():
    # Configuration
    input_dir = Path(r"D:\ces26_data")
    
    # Find all EXR files
    exr_files = sorted(input_dir.glob("*.exr"))
    
    if not exr_files:
        print(f"No EXR files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(exr_files)} EXR files to recompress")
    print(f"Recompressing with DWAA compression...")
    print()
    
    for i, exr_path in enumerate(exr_files):
        # Get file size before
        size_before = exr_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"[{i+1}/{len(exr_files)}] {exr_path.name} ({size_before:.1f} MB)", end="", flush=True)
        
        try:
            recompress_exr_dwaa(exr_path)
            
            # Get file size after
            size_after = exr_path.stat().st_size / (1024 * 1024)  # MB
            ratio = size_after / size_before * 100 if size_before > 0 else 0
            
            print(f" -> {size_after:.1f} MB ({ratio:.0f}%)")
        except Exception as e:
            print(f" ERROR: {e}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()

