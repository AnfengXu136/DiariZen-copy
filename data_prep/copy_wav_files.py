
"""
Script to read wav file names from wav.scp and copy them from source diarization datasets
to the destination paths specified in wav.scp.
"""

import os
import shutil
from pathlib import Path

# Source directories to search
SOURCE_DATASETS = [
    "/scratch1/anfengxu/public_datasets/diarization/AISHELL4_wav",
    "/scratch1/anfengxu/public_datasets/diarization/AMI-far",
    "/scratch1/anfengxu/public_datasets/diarization/AliMeeting/Eval_Ali/Eval_Ali_far/audio_dir",
    "/scratch1/anfengxu/public_datasets/diarization/AliMeeting/Test_Ali/Test_Ali_far/audio_dir",
    "/scratch1/anfengxu/public_datasets/diarization/AliMeeting/Train_Ali_far/audio_dir",
]

def find_wav_file(wav_name, source_dirs):
    """
    Search for a wav file in the given source directories.
    
    Args:
        wav_name: Name of the wav file (e.g., "ES2011a.wav")
        source_dirs: List of directories to search
        
    Returns:
        Full path to the file if found, None otherwise
    """
    for source_dir in source_dirs:
        # Search recursively in each source directory
        for root, dirs, files in os.walk(source_dir):
            if wav_name in files:
                return os.path.join(root, wav_name)
    return None

def copy_wav_files(scp_file):
    """
    Read wav.scp file and copy files from source to destination.
    
    Args:
        scp_file: Path to the wav.scp file
    """
    if not os.path.exists(scp_file):
        print(f"Error: {scp_file} does not exist")
        return
    
    copied_count = 0
    not_found_count = 0
    already_exist_count = 0
    
    with open(scp_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # Parse the line: ID destination_path
            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: Invalid line {line_num}: {line}")
                continue
                
            file_id, dest_path = parts
            wav_name = f"{file_id}.wav"
            
            # Check if destination already exists
            if os.path.exists(dest_path):
                print(f"Already exists: {dest_path}")
                already_exist_count += 1
                continue
            
            # Find the source file
            source_path = find_wav_file(wav_name, SOURCE_DATASETS)
            
            if source_path is None:
                print(f"Not found: {wav_name}")
                not_found_count += 1
                continue
            
            # Create destination directory if it doesn't exist
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy the file
            try:
                shutil.copy2(source_path, dest_path)
                print(f"Copied: {source_path} -> {dest_path}")
                copied_count += 1
            except Exception as e:
                print(f"Error copying {source_path} to {dest_path}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Files copied: {copied_count}")
    print(f"  Files already exist: {already_exist_count}")
    print(f"  Files not found: {not_found_count}")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    scp_file = "/project2/shrikann_35/anfengxu/DiariZen-copy/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4/train/wav.scp"
    copy_wav_files(scp_file)
