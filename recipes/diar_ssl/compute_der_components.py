#!/usr/bin/env python3
"""Compute and print DER component breakdown: MD, FA, SC.

Uses the dscore Python API (same pipeline as score.py) to extract global
Missed Detection (MD), False Alarm (FA), and Speaker Confusion (SC).
"""
import argparse
import glob
import os
import sys

# Make dscore importable regardless of working directory.
RECIPE_DIR = os.path.dirname(os.path.abspath(__file__))
DSCORE_DIR = os.path.normpath(os.path.join(RECIPE_DIR, '..', '..', 'dscore'))
sys.path.insert(0, DSCORE_DIR)

from scorelib.rttm import load_rttm
from scorelib.turn import merge_turns, trim_turns
from scorelib.uem import gen_uem, load_uem
from scorelib import metrics


def load_rttms(rttm_fns):
    """Load turns from multiple RTTM files (mirrors score.py helper)."""
    turns = []
    file_ids = set()
    for rttm_fn in rttm_fns:
        new_turns, _, new_file_ids = load_rttm(rttm_fn)
        turns.extend(new_turns)
        file_ids.update(new_file_ids)
    return turns, file_ids


def parse_args():
    p = argparse.ArgumentParser(
        description='Print MD / FA / SC breakdown from diarization output.')
    p.add_argument('-r', required=True, metavar='REF_RTTM',
                   help='Reference RTTM file')
    p.add_argument('-s', required=True, metavar='SYS_RTTM_GLOB',
                   help='System RTTM file or glob pattern (quote in shell)')
    p.add_argument('--collar', type=float, default=0.0,
                   help='Collar size in seconds (default: 0.0)')
    p.add_argument('--uem', default=None, metavar='UEM',
                   help='Optional UEM file')
    return p.parse_args()


def main():
    args = parse_args()

    sys_files = sorted(glob.glob(args.s))
    if not sys_files:
        print(f'[compute_der_components] No system RTTM files found: {args.s}')
        return

    # Load reference turns.
    ref_turns, _ = load_rttms([args.r])

    # Load system turns (handles multiple files).
    sys_turns, _ = load_rttms(sys_files)

    # Build or load UEM.
    if args.uem:
        uem = load_uem(args.uem)
    else:
        uem = gen_uem(ref_turns, sys_turns)

    # Trim and merge turns — same pre-processing as score.py.
    ref_turns = merge_turns(trim_turns(ref_turns, uem))
    sys_turns = merge_turns(trim_turns(sys_turns, uem))

    # Compute DER breakdown via the new metrics function.
    breakdown = metrics.der_breakdown(
        ref_turns, sys_turns, uem=uem, collar=args.collar)

    print(
        f'DER breakdown (collar={args.collar}s):  '
        f'MD={breakdown["md"]:.2f}%  '
        f'FA={breakdown["fa"]:.2f}%  '
        f'SC={breakdown["sc"]:.2f}%  '
        f'DER={breakdown["der"]:.2f}%'
    )


if __name__ == '__main__':
    main()
