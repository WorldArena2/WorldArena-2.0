#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# SPDX-License-Identifier: Apache-2.0

"""Delete immediate subdirectories under a root path if total size < threshold.

Only considers direct children of ``root`` (not nested subfolders individually).
Default threshold is 15 GiB (1024-based), overridable with ``--threshold-gb``.

Examples::

    # Preview what would be removed
    python toolkits/delete_subdirs_below_size.py /path/to/parent --dry-run

    # Actually delete subdirs under 15 GiB
    python toolkits/delete_subdirs_below_size.py /path/to/parent
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def directory_size_bytes(path: Path, follow_symlinks: bool = False) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=follow_symlinks):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                if not follow_symlinks and fp.is_symlink():
                    continue
                total += fp.stat(follow_symlinks=follow_symlinks).st_size
            except OSError:
                continue
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root",
        type=Path,
        help="Parent directory whose immediate subdirectories are checked.",
    )
    p.add_argument(
        "--threshold-gb",
        type=float,
        default=15.0,
        help="Subdirs with total size strictly below this many GiB (1024^3) are removed.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not delete.",
    )
    p.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinks when summing sizes (default: off).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root.resolve()
    threshold = int(args.threshold_gb * (1024**3))

    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    candidates: list[tuple[Path, int]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        size = directory_size_bytes(child, follow_symlinks=args.follow_symlinks)
        candidates.append((child, size))

    to_delete = [(p, s) for p, s in candidates if s < threshold]
    keep = [(p, s) for p, s in candidates if s >= threshold]

    for p, s in keep:
        print(f"keep  {s / (1024**3):.3f} GiB  {p}")

    for p, s in to_delete:
        gib = s / (1024**3)
        if args.dry_run:
            print(f"[dry-run] would delete  {gib:.3f} GiB  (< {args.threshold_gb} GiB)  {p}")
        else:
            print(f"delete  {gib:.3f} GiB  {p}")
            shutil.rmtree(p, ignore_errors=False)

    print(
        f"summary: {len(to_delete)} removed / would remove, {len(keep)} kept "
        f"(threshold {args.threshold_gb} GiB, root {root})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
