#!/usr/bin/env python3
"""Canonical filesystem layout for PELLM training artifacts.

Heavy artifacts should live outside the git checkout by default.  The
``PELLM_DATA_ROOT`` environment variable can override the root for machines
that do not mount the large data drive at ``<pellm_data_root>``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

DATA_ROOT = Path(os.environ.get("PELLM_DATA_ROOT", "<pellm_data_root>")).expanduser()

DATASETS_DIR = DATA_ROOT / "datasets"
CHECKPOINTS_DIR = DATA_ROOT / "checkpoints"
TRAINED_MODELS_DIR = DATA_ROOT / "trainedmodels"
RUNS_DIR = DATA_ROOT / "runs"
EVALS_DIR = DATA_ROOT / "evals"
HF_STAGING_DIR = DATA_ROOT / "hf_staging"
TMP_DIR = DATA_ROOT / "tmp"
HF_HOME_DIR = DATA_ROOT / "hf_home"

ARTIFACT_DIRS: dict[str, Path] = {
    "datasets": DATASETS_DIR,
    "hf_datasets_cache": DATASETS_DIR / "hf_cache",
    "transformers_cache": DATASETS_DIR / "transformers_cache",
    "tokenized_shards": DATASETS_DIR / "tokenized_shards",
    "checkpoints": CHECKPOINTS_DIR,
    "trainedmodels": TRAINED_MODELS_DIR,
    "runs": RUNS_DIR,
    "evals": EVALS_DIR,
    "hf_staging": HF_STAGING_DIR,
    "tmp": TMP_DIR,
    "hf_home": HF_HOME_DIR,
}

HF_MODEL_REPOS: dict[str, str] = {
    "baseline": "anon/pellm-smol-135m-baseline",
    "ae_mlp": "anon/pellm-smol-135m-ae-mlp",
    "trendwavelet": "anon/pellm-smol-135m-trendwavelet",
    "ae_tw": "anon/pellm-smol-135m-ae-tw",
}


def env_defaults() -> dict[str, str]:
    """Return default environment variables for data-drive backed training."""

    return {
        "PELLM_DATA_ROOT": str(DATA_ROOT),
        "HF_HOME": str(HF_HOME_DIR),
        "HF_DATASETS_CACHE": str(DATASETS_DIR / "hf_cache"),
        "TRANSFORMERS_CACHE": str(DATASETS_DIR / "transformers_cache"),
        "TMPDIR": str(TMP_DIR),
    }


def apply_data_drive_env_defaults(env: Mapping[str, str]) -> dict[str, str]:
    """Merge artifact environment defaults without overriding user choices."""

    merged = {str(k): str(v) for k, v in env.items()}
    for key, value in env_defaults().items():
        merged.setdefault(key, value)
    return merged


def ensure_artifact_dirs() -> dict[str, Path]:
    """Create all canonical artifact directories and return them by name."""

    for path in ARTIFACT_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIRS


def repo_id_for_variant(variant: str) -> str:
    """Return the default Hugging Face repo id for a known paper variant."""

    try:
        return HF_MODEL_REPOS[variant]
    except KeyError as exc:
        valid = ", ".join(sorted(HF_MODEL_REPOS))
        raise ValueError(f"Unknown variant {variant!r}; expected one of: {valid}") from exc
