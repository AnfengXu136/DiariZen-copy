"""Generate AMI IHM mix wavs from headset channels.

Iterates through a root AMI directory and runs SoX to create an
``<meeting>.IHM-mix.wav`` for each meeting that has headset channel wavs.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CHANNEL_PATTERN = re.compile(r"(.+)\.Headset-(\d+)\.wav$")


def find_headset_groups(meeting_dir: Path) -> Dict[str, List[Tuple[int, Path]]]:
    """Return mapping of base meeting name to list of (channel, path)."""
    groups: Dict[str, List[Tuple[int, Path]]] = {}
    for wav_path in meeting_dir.glob("*.Headset-*.wav"):
        match = CHANNEL_PATTERN.match(wav_path.name)
        if not match:
            continue
        base, channel = match.group(1), int(match.group(2))
        groups.setdefault(base, []).append((channel, wav_path))
    return groups


def build_sox_command(channels: Iterable[Path], output_path: Path) -> List[str]:
    return [
        "sox",
        "-m",
        *[str(p) for p in channels],
        str(output_path),
        "gain",
        "-n",
    ]


def process_meeting(
    meeting_dir: Path,
    input_root: Path,
    output_root: Path,
    min_channels: int,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    """Process a single meeting directory. Returns (made, skipped)."""
    made = skipped = 0
    groups = find_headset_groups(meeting_dir)
    for base, channel_pairs in groups.items():
        if len(channel_pairs) < min_channels:
            skipped += 1
            continue

        sorted_channels = [p for _, p in sorted(channel_pairs, key=lambda x: x[0])]
        rel_dir = meeting_dir.relative_to(input_root)
        output_path = output_root / rel_dir / f"{base}.wav"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_sox_command(sorted_channels, output_path)
        if dry_run:
            print("DRY RUN:", " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)
            print(f"Created {output_path}")
        made += 1
    return made, skipped


def process_root(
    input_root: Path,
    output_root: Path,
    min_channels: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    total_made = total_skipped = 0
    for meeting_dir in tqdm(sorted(p for p in input_root.glob("**") if p.is_dir())):
        made, skipped = process_meeting(
            meeting_dir,
            input_root,
            output_root,
            min_channels,
            overwrite,
            dry_run,
        )
        total_made += made
        total_skipped += skipped
    print(
        f"Done. Created {total_made} file(s); skipped {total_skipped} (missing channels or exists)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AMI IHM mixes from headset wav channels using SoX.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        # default=Path("/scratch1/anfengxu/public_datasets/diarization/AMI"),
        default=Path("/scratch1/anfengxu/public_datasets/diarization/AliMeeting"),
        help="Root directory containing AMI meeting folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        # default=Path("/scratch1/anfengxu/public_datasets/diarization/AMI-IHM-mix"),
        default=Path("/scratch1/anfengxu/public_datasets/diarization/AliMeeting-IHM-mix"),
        help="Directory where IHM-mix wavs will be written (mirrors input structure).",
    )
    parser.add_argument(
        "--min-channels",
        type=int,
        default=1,
        help="Minimum number of headset channels required to build a mix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running SoX.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise SystemExit(f"Input root does not exist: {args.input_root}")
    process_root(
        input_root=args.input_root,
        output_root=args.output_root,
        min_channels=args.min_channels,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()