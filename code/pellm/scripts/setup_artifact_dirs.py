#!/usr/bin/env python3
"""Create the canonical PELLM artifact directories on the data drive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_paths import ARTIFACT_DIRS, DATA_ROOT, ensure_artifact_dirs, env_defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create / print canonical PELLM training artifact directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the directories and environment exports without creating them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = ARTIFACT_DIRS if args.dry_run else ensure_artifact_dirs()

    action = "Would create" if args.dry_run else "Created/verified"
    print(f"{action} PELLM artifact root: {DATA_ROOT}")
    for name, path in dirs.items():
        print(f"  {name:20s} {path}")

    print("\nRecommended environment:")
    for key, value in env_defaults().items():
        print(f"  export {key}={value}")


if __name__ == "__main__":
    main()
