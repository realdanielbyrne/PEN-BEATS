#!/usr/bin/env python3
"""Prune large generated experiment artifacts without touching source files.

Safe by default:
- activation caches under scripts/experiments/results/**
- dry-run unless --apply is passed

Opt-in:
- trainedmodels/<experiment>/ checkpoints via --include-checkpoints
- local wandb/ run directories via --include-wandb
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "scripts" / "experiments" / "results"
MODELS_ROOT = REPO_ROOT / "trainedmodels"
WANDB_ROOT = REPO_ROOT / "wandb"


@dataclass(frozen=True)
class Candidate:
    path: Path
    kind: str


def dir_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def matches_experiment(path: Path, experiments: set[str] | None, root: Path) -> bool:
    if not experiments:
        return True
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return bool(rel.parts) and rel.parts[0] in experiments


def find_cache_candidates(experiments: set[str] | None) -> list[Candidate]:
    candidates: list[Candidate] = []
    if not RESULTS_ROOT.exists():
        return candidates

    for path in RESULTS_ROOT.rglob("*"):
        if not path.is_dir():
            continue
        if not matches_experiment(path, experiments, RESULTS_ROOT):
            continue
        name = path.name
        if name in {"attn_activation_cache", "ae_activation_cache"} or name.startswith("ae_cache_"):
            candidates.append(Candidate(path=path, kind="cache"))
    return candidates


def find_checkpoint_candidates(experiments: set[str] | None) -> list[Candidate]:
    candidates: list[Candidate] = []
    if not MODELS_ROOT.exists():
        return candidates

    if experiments:
        for experiment in sorted(experiments):
            path = MODELS_ROOT / experiment
            if path.exists():
                candidates.append(Candidate(path=path, kind="checkpoint"))
        return candidates

    for path in sorted(MODELS_ROOT.iterdir()):
        if path.is_dir():
            candidates.append(Candidate(path=path, kind="checkpoint"))
    return candidates


def find_wandb_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    if not WANDB_ROOT.exists():
        return candidates

    for path in sorted(WANDB_ROOT.iterdir()):
        if path.name == "latest-run":
            continue
        if path.is_dir():
            candidates.append(Candidate(path=path, kind="wandb"))
    return candidates


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def prune_empty_dirs(base: Path) -> int:
    removed = 0
    if not base.exists():
        return removed

    for path in sorted((p for p in base.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()
            removed += 1
        except OSError:
            continue
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune generated experiment caches, optional checkpoints, and local wandb runs.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        metavar="NAME",
        help="Limit pruning to one experiment name. Repeat for multiple experiments.",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Also remove trainedmodels/<experiment>/ directories. Use with care.",
    )
    parser.add_argument(
        "--include-wandb",
        action="store_true",
        help="Also remove local wandb/ run directories after runs are synced.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, the script performs a dry run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = set(args.experiment) or None

    candidates = find_cache_candidates(experiments)
    if args.include_checkpoints:
        candidates.extend(find_checkpoint_candidates(experiments))
    if args.include_wandb:
        candidates.extend(find_wandb_candidates())

    unique_candidates = sorted({candidate.path: candidate for candidate in candidates}.values(), key=lambda c: str(c.path))

    if not unique_candidates:
        print("No matching generated artifacts found.")
        return 0

    total_bytes = 0
    print("Planned removals:")
    for candidate in unique_candidates:
        size_bytes = dir_size_bytes(candidate.path)
        total_bytes += size_bytes
        print(f"  {candidate.kind:10s} {format_bytes(size_bytes):>8s}  {candidate.path.relative_to(REPO_ROOT)}")

    print(f"\nTotal reclaimable: {format_bytes(total_bytes)}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete these paths.")
        return 0

    for candidate in unique_candidates:
        remove_path(candidate.path)

    removed_empty = prune_empty_dirs(RESULTS_ROOT)
    removed_empty += prune_empty_dirs(MODELS_ROOT)
    if args.include_wandb:
        removed_empty += prune_empty_dirs(WANDB_ROOT)

    print(f"\nRemoved {len(unique_candidates)} path(s).")
    if removed_empty:
        print(f"Removed {removed_empty} empty parent directorie(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
