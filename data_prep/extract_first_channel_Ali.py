
"""
Script to extract the first channel from AliMeeting audio files and copy them
to a new directory structure.
"""

import os
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Source base directory
SOURCE_BASE = "/scratch1/anfengxu/public_datasets/diarization/AliMeeting"

# Destination base directory
DEST_BASE = "/scratch1/anfengxu/public_datasets/diarization/AliMeeting_mono"

# Subdirectories to process
SUBDIRS = [
    "Eval_Ali/Eval_Ali_far/audio_dir",
    "Test_Ali/Test_Ali_far/audio_dir",
    "Train_Ali_far/audio_dir",
]


def extract_first_channel(source_path, dest_path):
    """
    Extract the first channel from an audio file and save to destination.
    
    Args:
        source_path: Path to source audio file
        dest_path: Path to save the mono audio file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the audio file
        audio, samplerate = sf.read(source_path)
        
        # Check if audio is multi-channel
        if len(audio.shape) == 1:
            # Already mono, just copy
            mono_audio = audio
        else:
            # Extract first channel
            mono_audio = audio[:, 0]
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Write the mono audio
        sf.write(dest_path, mono_audio, samplerate)
        return True
        
    except Exception as e:
        print(f"Error processing {source_path}: {e}")
        return False


def process_subdirectory(subdir):
    """
    Process all audio files in a subdirectory.
    
    Args:
        subdir: Relative path to subdirectory (e.g., "Eval_Ali/Eval_Ali_far/audio_dir")
    """
    source_dir = os.path.join(SOURCE_BASE, subdir)
    
    # For destination, we need to preserve the structure
    # e.g., Eval_Ali/Eval_Ali_far/audio_dir -> Eval_Ali/Eval_Ali_far/audio_dir
    dest_dir = os.path.join(DEST_BASE, subdir)
    
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return
    
    print(f"\nProcessing: {source_dir}")
    print(f"Destination: {dest_dir}")
    
    # Get all wav files in the source directory
    wav_files = list(Path(source_dir).glob("*.wav"))
    
    if not wav_files:
        print(f"No wav files found in {source_dir}")
        return
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Process each file with progress bar
    for wav_file in tqdm(wav_files, desc=f"Processing {subdir}"):
        dest_path = os.path.join(dest_dir, wav_file.name)
        
        # Skip if already exists
        if os.path.exists(dest_path):
            skipped_count += 1
            continue
        
        if extract_first_channel(str(wav_file), dest_path):
            success_count += 1
        else:
            failed_count += 1
    
    # Print summary for this subdirectory
    print(f"\nSummary for {subdir}:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Already exist (skipped): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total files: {len(wav_files)}")


def main():
    """Main function to process all subdirectories."""
    print("="*60)
    print("AliMeeting First Channel Extraction")
    print("="*60)
    
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    for subdir in SUBDIRS:
        process_subdirectory(subdir)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
