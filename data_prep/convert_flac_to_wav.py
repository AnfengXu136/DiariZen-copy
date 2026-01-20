#!/usr/bin/env python3
"""
Convert FLAC files to mono WAV files using sox.
Processes AISHELL4 dataset splits: test, train_L, train_M, train_S
"""

import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_BASE = "/scratch1/anfengxu/public_datasets/diarization/AISHELL4"
OUTPUT_BASE = "/scratch1/anfengxu/public_datasets/diarization/AISHELL4_wav"
SPLITS = ["test", "train_L", "train_M", "train_S"]


def convert_flac_to_mono_wav(input_flac, output_wav):
    """
    Convert a FLAC file to mono WAV using sox.
    
    Args:
        input_flac: Path to input FLAC file
        output_wav: Path to output WAV file
    """
    # Create output directory if it doesn't exist
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    
    # Use sox to convert: flac -> wav and make mono (-c 1)
    cmd = ["sox", str(input_flac), "-c", "1", str(output_wav)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_flac}: {e.stderr}")
        return False


def process_split(split_name):
    """
    Process all FLAC files in a given split.
    
    Args:
        split_name: Name of the split (e.g., 'test', 'train_L')
    """
    input_dir = Path(INPUT_BASE) / split_name
    output_dir = Path(OUTPUT_BASE) / split_name
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    # Find all FLAC files recursively
    flac_files = list(input_dir.rglob("*.flac"))
    
    if not flac_files:
        print(f"No FLAC files found in {input_dir}")
        return
    
    print(f"\nProcessing split: {split_name}")
    print(f"Found {len(flac_files)} FLAC files")
    
    # Process each file
    success_count = 0
    for flac_file in tqdm(flac_files, desc=f"Converting {split_name}"):
        # Preserve directory structure
        relative_path = flac_file.relative_to(input_dir)
        output_file = output_dir / relative_path.with_suffix(".wav")
        
        if convert_flac_to_mono_wav(flac_file, output_file):
            success_count += 1
    
    print(f"Successfully converted {success_count}/{len(flac_files)} files")


def main():
    """Main function to process all splits."""
    print("Starting FLAC to mono WAV conversion")
    print(f"Input base: {INPUT_BASE}")
    print(f"Output base: {OUTPUT_BASE}")
    print(f"Splits to process: {', '.join(SPLITS)}")
    
    # Check if sox is available
    try:
        subprocess.run(["sox", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: sox is not installed or not in PATH")
        print("Please install sox first (e.g., 'apt-get install sox' or 'brew install sox')")
        return
    
    # Create output base directory
    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in SPLITS:
        process_split(split)
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()