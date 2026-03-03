"""
WaveletV3AE Study — Successive Halving Search + Cross-Dataset Benchmark

Config-driven architecture search for alternating TrendAE + WaveletV3AE stacks,
followed by global top-10 cross-dataset validation.

CLI modes:
    search --config <dataset_yaml>
    cross  --configs <yaml1> <yaml2> <yaml3> <yaml4>
    all    --configs <yaml1> <yaml2> <yaml3> <yaml4>

Common CLI scripts:
    # Search one dataset (full 3-round successive halving)
    python experiments/run_wavelet_v3ae_study.py search \
        --config experiments/configs/wavelet_v3ae_study_m4.yaml

    # Search one dataset, a single round only
    python experiments/run_wavelet_v3ae_study.py search \
        --config experiments/configs/wavelet_v3ae_study_traffic.yaml \
        --round 1

    # Analyze/rank existing search results (no new training)
    python experiments/run_wavelet_v3ae_study.py search \
        --config experiments/configs/wavelet_v3ae_study_weather.yaml \
        --round all --analyze

    # Run all 4 dataset searches then cross-dataset benchmark
    python experiments/run_wavelet_v3ae_study.py all \
        --configs \
        experiments/configs/wavelet_v3ae_study_m4.yaml \
        experiments/configs/wavelet_v3ae_study_tourism.yaml \
        experiments/configs/wavelet_v3ae_study_traffic.yaml \
        experiments/configs/wavelet_v3ae_study_weather.yaml

    # Run cross-dataset benchmark only (uses completed round-3 CSVs)
    python experiments/run_wavelet_v3ae_study.py cross \
        --configs \
        experiments/configs/wavelet_v3ae_study_m4.yaml \
        experiments/configs/wavelet_v3ae_study_tourism.yaml \
        experiments/configs/wavelet_v3ae_study_traffic.yaml \
        experiments/configs/wavelet_v3ae_study_weather.yaml

    # Optional hardware overrides
    python experiments/run_wavelet_v3ae_study.py search \
        --config experiments/configs/wavelet_v3ae_study_m4.yaml \
        --accelerator cuda --n-gpus 2
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

# Allow running from project root or experiments/
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXPERIMENTS_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", "src"))

from run_unified_benchmark import (  # noqa: E402
    run_single_experiment,
    result_exists,
    get_batch_size,
    init_csv,
    load_dataset,
    resolve_n_gpus,
    CSV_COLUMNS,
    FORECAST_MULTIPLIERS,
    DATASET_PERIODS,
    EARLY_STOPPING_PATIENCE,
    _shutdown_requested,
)
from tools.meta_forecaster import MetaForecaster  # noqa: E402


torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WAVELET_V3AE_FAMILIES = [
    "HaarWaveletV3AE",
    "DB2WaveletV3AE",
    "DB3WaveletV3AE",
    "DB4WaveletV3AE",
    "DB10WaveletV3AE",
    "DB20WaveletV3AE",
    "Coif1WaveletV3AE",
    "Coif2WaveletV3AE",
    "Coif3WaveletV3AE",
    "Coif10WaveletV3AE",
    "Symlet2WaveletV3AE",
    "Symlet3WaveletV3AE",
    "Symlet10WaveletV3AE",
    "Symlet20WaveletV3AE",
]

SEARCH_EXTRA_COLUMNS = [
    "search_round",
    "basis_dim",
    "basis_offset",
    "trend_thetas_dim_cfg",
    "wavelet_family",
    "bd_label",
    "latent_dim_cfg",
    "meta_predicted_best",
    "meta_convergence_score",
]

CROSS_EXTRA_COLUMNS = SEARCH_EXTRA_COLUMNS + [
    "source_datasets",
    "global_selection_score",
    "canonical_config_id",
]

SEARCH_CSV_COLUMNS = CSV_COLUMNS + SEARCH_EXTRA_COLUMNS
CROSS_CSV_COLUMNS = CSV_COLUMNS + CROSS_EXTRA_COLUMNS

DEFAULT_META_TRAINING_CSVS = [
    os.path.join(_EXPERIMENTS_DIR, "results", "m4", "unified_benchmark_results.csv"),
    os.path.join(_EXPERIMENTS_DIR, "results", "m4", "block_benchmark_results.csv"),
    os.path.join(_EXPERIMENTS_DIR, "results", "traffic", "block_benchmark_results.csv"),
    os.path.join(_EXPERIMENTS_DIR, "results", "m4", "convergence_study_results_v1.csv"),
    os.path.join(_EXPERIMENTS_DIR, "results", "weather", "convergence_study_results_v1.csv"),
    os.path.join(_EXPERIMENTS_DIR, "results", "traffic", "convergence_study_results_v1.csv"),
]

SUPPORTED_DATASETS = {"m4", "tourism", "traffic", "weather"}


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


@dataclass
class StudyConfig:
    path: str
    dataset: str
    period: str
    architecture: dict
    training: dict
    lr_scheduler: dict
    search_space: dict
    search_rounds: list[dict]
    output: dict
    hardware: dict
    runs: dict
    meta_forecaster: dict
    logging: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YAML Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, val in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return cfg


_LOGGING_CONFIGURED = False
_LOG_PATH = None


class _TeeStream:
    """Write to both the original stream and a log file stream."""

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()

    def isatty(self):
        return self.primary.isatty()

    def fileno(self):
        return self.primary.fileno()


def _configure_process_logging(log_path: str) -> None:
    """Mirror stdout/stderr to a shared log file path."""
    global _LOGGING_CONFIGURED, _LOG_PATH
    if _LOGGING_CONFIGURED:
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_fh)
    sys.stderr = _TeeStream(sys.stderr, log_fh)
    _LOGGING_CONFIGURED = True
    _LOG_PATH = log_path


def _configure_process_logging_from_env() -> None:
    log_path = os.environ.get("WAVELET_V3AE_LOG_PATH")
    if log_path:
        _configure_process_logging(log_path)


def load_study_config(path: str) -> StudyConfig:
    raw = _load_yaml(path)

    dataset = str(raw.get("dataset", "")).strip().lower()
    period = str(raw.get("period", "")).strip()

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}' in {path}; expected one of {sorted(SUPPORTED_DATASETS)}"
        )
    if period not in DATASET_PERIODS[dataset]:
        raise ValueError(
            f"Invalid period '{period}' for dataset '{dataset}' in {path}; "
            f"expected one of {list(DATASET_PERIODS[dataset].keys())}"
        )

    defaults = {
        "architecture": {
            "trend_block": "TrendAE",
            "repeats": 5,
        },
        "training": {
            "active_g": False,
            "sum_losses": False,
            "activation": "ReLU",
            "n_blocks_per_stack": 1,
            "share_weights": True,
            "loss": "SMAPELoss",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "basis_offset": 0,
        },
        "lr_scheduler": {
            "warmup_epochs": 15,
            "eta_min": 1e-6,
            "T_max": None,
        },
        "search_space": {
            "wavelets": list(WAVELET_V3AE_FAMILIES),
            "basis_labels": ["eq_fcast", "lt_fcast", "eq_bcast", "lt_bcast"],
            "trend_thetas_dims": [3, 5],
            "latent_dims": [2, 5, 8],
        },
        "search": {
            "rounds": [
                {"max_epochs": 10, "n_runs": 3, "keep_fraction": 0.67},
                {"max_epochs": 15, "n_runs": 3, "keep_fraction": 0.67},
                {"max_epochs": 50, "n_runs": 3, "top_k": 10},
            ]
        },
        "output": {
            "results_dir": "experiments/results",
            "search_csv_filename": "wavelet_v3ae_study_results.csv",
            "cross_csv_path": "experiments/results/wavelet_v3ae_cross_dataset_results.csv",
        },
        "hardware": {
            "accelerator": "auto",
            "num_workers": 0,
            "n_gpus": None,
            "batch_size": None,
        },
        "logging": {
            "wandb": True,
            "wandb_project": "nbeats-lightning",
        },
        "runs": {
            "base_seed": 42,
        },
        "meta_forecaster": {
            "enabled": True,
            "cache_dir": "experiments/results/.meta_cache",
            "training_csvs": list(DEFAULT_META_TRAINING_CSVS),
        },
    }

    architecture = _deep_merge(defaults["architecture"], raw.get("architecture") or {})
    training = _deep_merge(defaults["training"], raw.get("training") or {})
    lr_scheduler = _deep_merge(defaults["lr_scheduler"], raw.get("lr_scheduler") or {})
    search_space = _deep_merge(defaults["search_space"], raw.get("search_space") or {})
    search_rounds = list((_deep_merge(defaults["search"], raw.get("search") or {})).get("rounds") or [])
    output = _deep_merge(defaults["output"], raw.get("output") or {})
    hardware = _deep_merge(defaults["hardware"], raw.get("hardware") or {})
    logging_cfg = _deep_merge(defaults["logging"], raw.get("logging") or {})
    runs = _deep_merge(defaults["runs"], raw.get("runs") or {})
    meta_forecaster = _deep_merge(defaults["meta_forecaster"], raw.get("meta_forecaster") or {})

    if bool(training.get("active_g", False)):
        raise ValueError("This study enforces active_g=false (one-pass only).")

    return StudyConfig(
        path=os.path.abspath(path),
        dataset=dataset,
        period=period,
        architecture=architecture,
        training=training,
        lr_scheduler=lr_scheduler,
        search_space=search_space,
        search_rounds=search_rounds,
        output=output,
        hardware=hardware,
        logging=logging_cfg,
        runs=runs,
        meta_forecaster=meta_forecaster,
    )


# ---------------------------------------------------------------------------
# Config Expansion
# ---------------------------------------------------------------------------


def compute_basis_dim(label: str, forecast_length: int, backcast_length: int) -> int:
    if label == "eq_fcast":
        return forecast_length
    if label == "lt_fcast":
        return max(forecast_length // 2, forecast_length - 2)
    if label == "eq_bcast":
        return backcast_length
    if label == "lt_bcast":
        return backcast_length // 2
    raise ValueError(f"Unknown basis label: {label}")


def wavelet_short_name(wavelet_family: str) -> str:
    return wavelet_family.replace("WaveletV3AELG", "").replace("WaveletV3AE", "")


def canonical_config_id(wavelet_family: str, bd_label: str, trend_thetas_dim: int, latent_dim: int) -> str:
    return f"{wavelet_family}|{bd_label}|ttd{int(trend_thetas_dim)}|ld{int(latent_dim)}"


def config_name_from_canonical(canonical_id: str) -> str:
    wavelet_family, bd_label, ttd_tag, ld_tag = canonical_id.split("|")
    return f"{wavelet_short_name(wavelet_family)}_{bd_label}_{ttd_tag}_{ld_tag}"


def _forecast_backcast_lengths(dataset_name: str, period: str, forecast_multiplier_override: int | None = None) -> tuple[int, int]:
    forecast_length = int(DATASET_PERIODS[dataset_name][period]["horizon"])
    if forecast_multiplier_override is None:
        forecast_multiplier = int(FORECAST_MULTIPLIERS[dataset_name])
    else:
        forecast_multiplier = int(forecast_multiplier_override)
    return forecast_length, int(forecast_length * forecast_multiplier)


def generate_search_configs(study_cfg: StudyConfig, promoted_config_names: set[str] | None = None) -> dict[str, dict]:
    dataset_name = study_cfg.dataset
    period = study_cfg.period

    forecast_length, backcast_length = _forecast_backcast_lengths(
        dataset_name,
        period,
        forecast_multiplier_override=study_cfg.training.get("forecast_multiplier"),
    )

    trend_block = str(study_cfg.architecture.get("trend_block", "TrendAE"))
    repeats = int(study_cfg.architecture.get("repeats", 5))

    wavelets = list(study_cfg.search_space.get("wavelets") or [])
    basis_labels = list(study_cfg.search_space.get("basis_labels") or [])
    trend_thetas_dims = list(study_cfg.search_space.get("trend_thetas_dims") or [])
    latent_dims = list(study_cfg.search_space.get("latent_dims") or [])

    configs: dict[str, dict] = {}
    for wavelet_family in wavelets:
        for bd_label in basis_labels:
            basis_dim = compute_basis_dim(bd_label, forecast_length, backcast_length)
            for trend_thetas_dim in trend_thetas_dims:
                for latent_dim in latent_dims:
                    canonical_id = canonical_config_id(
                        wavelet_family,
                        bd_label,
                        int(trend_thetas_dim),
                        int(latent_dim),
                    )
                    config_name = config_name_from_canonical(canonical_id)

                    if promoted_config_names is not None and config_name not in promoted_config_names:
                        continue

                    configs[config_name] = {
                        "config_name": config_name,
                        "canonical_config_id": canonical_id,
                        "wavelet_family": wavelet_family,
                        "bd_label": bd_label,
                        "basis_dim": int(basis_dim),
                        "basis_offset": int(study_cfg.training.get("basis_offset", 0)),
                        "trend_thetas_dim": int(trend_thetas_dim),
                        "latent_dim": int(latent_dim),
                        "stack_types": [trend_block, wavelet_family] * repeats,
                        "n_blocks_per_stack": int(study_cfg.training.get("n_blocks_per_stack", 1)),
                        "share_weights": bool(study_cfg.training.get("share_weights", True)),
                    }

    return configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_keep_n(total: int, keep_fraction: float) -> int:
    return max(1, int(math.floor(total * float(keep_fraction))))


def resolve_lr_scheduler(lr_scheduler_cfg: dict, max_epochs: int) -> dict | None:
    if not lr_scheduler_cfg:
        return None
    warmup_epochs = int(lr_scheduler_cfg.get("warmup_epochs", 15))
    if max_epochs <= warmup_epochs:
        return None
    t_max_raw = lr_scheduler_cfg.get("T_max")
    t_max = int(t_max_raw) if t_max_raw is not None else max(max_epochs - warmup_epochs, 1)
    eta_min = float(lr_scheduler_cfg.get("eta_min", 1e-6))
    return {
        "warmup_epochs": warmup_epochs,
        "T_max": t_max,
        "eta_min": eta_min,
    }


def _resolve_seed(base_seed: int, run_idx: int) -> int:
    return int(base_seed) + int(run_idx)


def _resolve_wandb(study_cfg: StudyConfig, cli_args) -> tuple[bool, str]:
    enabled = bool(study_cfg.logging.get("wandb", True))
    if getattr(cli_args, "wandb", False):
        enabled = True
    if getattr(cli_args, "no_wandb", False):
        enabled = False
    project = str(getattr(cli_args, "wandb_project", None) or study_cfg.logging.get("wandb_project", "nbeats-lightning"))
    return enabled, project


def _search_csv_path(study_cfg: StudyConfig) -> str:
    return os.path.join(
        study_cfg.output["results_dir"],
        study_cfg.dataset,
        study_cfg.output.get("search_csv_filename", "wavelet_v3ae_study_results.csv"),
    )


def _cross_csv_path(study_cfg: StudyConfig) -> str:
    return os.path.abspath(study_cfg.output.get("cross_csv_path", "experiments/results/wavelet_v3ae_cross_dataset_results.csv"))


def _rows_for_round(csv_path: str, round_num: int) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    out: list[dict] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sr = row.get("search_round", "")
            if sr == str(round_num):
                out.append(row)
    return out


def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _update_meta_predictions(csv_path: str, round_num: int, meta_forecaster: MetaForecaster) -> None:
    if not os.path.exists(csv_path):
        return

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if row.get("search_round", "") == str(round_num):
                raw_curve = row.get("val_loss_curve", "")
                try:
                    parsed = json.loads(raw_curve)
                    curve = [float(v) for v in parsed]
                    if len(curve) >= MetaForecaster.BACKCAST_LENGTH:
                        pred = meta_forecaster.predict(curve)
                        pb = pred.get("predicted_best")
                        cs = pred.get("convergence_score")
                        row["meta_predicted_best"] = (
                            f"{pb:.8f}" if pb is not None and math.isfinite(float(pb)) else ""
                        )
                        row["meta_convergence_score"] = (
                            f"{cs:.8f}" if cs is not None and math.isfinite(float(cs)) else ""
                        )
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_and_promote(
    csv_path: str,
    round_num: int,
    keep_fraction: float,
    top_k: int | None = None,
    use_meta: bool = False,
) -> tuple[list[str], list[dict]]:
    rows = _rows_for_round(csv_path, round_num)
    if not rows:
        return [], []

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["config_name"]].append(row)

    rankings = []
    for config_name, rs in grouped.items():
        val_losses = []
        meta_preds = []
        n_diverged = 0

        for r in rs:
            bvl = _safe_float(r.get("best_val_loss"), default=float("inf"))
            if math.isfinite(bvl):
                val_losses.append(bvl)

            mp = _safe_float(r.get("meta_predicted_best"), default=float("nan"))
            if math.isfinite(mp):
                meta_preds.append(mp)

            if str(r.get("diverged", "")).lower() in ("true", "1"):
                n_diverged += 1

        median_bvl = float(np.median(val_losses)) if val_losses else float("inf")
        min_meta = min(meta_preds) if meta_preds else float("inf")
        divergence_rate = n_diverged / len(rs) if rs else 0.0

        if use_meta and math.isfinite(min_meta):
            rank_metric = min_meta
        else:
            rank_metric = median_bvl

        rankings.append(
            {
                "config_name": config_name,
                "median_best_val_loss": median_bvl,
                "meta_predicted_best": min_meta,
                "divergence_rate": divergence_rate,
                "rank_metric": rank_metric,
                "n_rows": len(rs),
            }
        )

    rankings.sort(
        key=lambda r: (
            r["divergence_rate"] > 0.5,
            r["rank_metric"] if math.isfinite(r["rank_metric"]) else 1e18,
            r["config_name"],
        )
    )

    total = len(rankings)
    if top_k is not None:
        keep_n = min(int(top_k), total)
    else:
        keep_n = compute_keep_n(total, keep_fraction)

    promoted = [r["config_name"] for r in rankings[:keep_n]]

    print(f"\n  {'=' * 80}")
    print(
        f"  Round {round_num} ranking: {total} configs, "
        f"promoting {keep_n} ({'top_k' if top_k is not None else f'floor({keep_fraction:.2f})'})"
    )
    print(f"  {'=' * 80}")
    print(f"  {'Rank':<6} {'Config':<42} {'ValLoss':>10} {'Meta':>10} {'Div%':>6}")
    print(f"  {'-' * 80}")
    for i, r in enumerate(rankings[: min(40, total)]):
        marker = "*" if r["config_name"] in promoted else " "
        meta_str = f"{r['meta_predicted_best']:.4f}" if math.isfinite(r["meta_predicted_best"]) else "--"
        print(
            f"  {i + 1:<5}{marker} {r['config_name']:<42} "
            f"{r['median_best_val_loss']:>10.4f} {meta_str:>10} {r['divergence_rate'] * 100:>5.1f}%"
        )

    return promoted, rankings


# ---------------------------------------------------------------------------
# Single Experiment Wrapper
# ---------------------------------------------------------------------------


def _run_single_search_job(
    study_cfg: StudyConfig,
    config: dict,
    dataset,
    train_series_list,
    csv_path: str,
    round_num: int,
    run_idx: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    accelerator: str,
    num_workers: int,
    seed: int,
    wandb_enabled: bool,
    wandb_project: str,
    gpu_id: int | None = None,
):
    lr_scheduler_config = resolve_lr_scheduler(study_cfg.lr_scheduler, max_epochs)

    extra_row = {
        "search_round": round_num,
        "basis_dim": int(config["basis_dim"]),
        "basis_offset": int(config["basis_offset"]),
        "trend_thetas_dim_cfg": int(config["trend_thetas_dim"]),
        "wavelet_family": config["wavelet_family"],
        "bd_label": config["bd_label"],
        "latent_dim_cfg": int(config["latent_dim"]),
        "meta_predicted_best": "",
        "meta_convergence_score": "",
    }

    experiment_name = f"wavelet_v3ae_search_r{round_num}"

    run_single_experiment(
        experiment_name=experiment_name,
        config_name=config["config_name"],
        category="wavelet_v3ae_search",
        stack_types=config["stack_types"],
        period=study_cfg.period,
        run_idx=run_idx,
        dataset=dataset,
        train_series_list=train_series_list,
        csv_path=csv_path,
        n_blocks_per_stack=int(config["n_blocks_per_stack"]),
        share_weights=bool(config["share_weights"]),
        active_g=False,
        sum_losses=bool(study_cfg.training.get("sum_losses", False)),
        activation=str(study_cfg.training.get("activation", "ReLU")),
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        accelerator_override=accelerator,
        forecast_multiplier=int(study_cfg.training.get("forecast_multiplier") or FORECAST_MULTIPLIERS[study_cfg.dataset]),
        num_workers=num_workers,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        save_predictions=False,
        gpu_id=gpu_id,
        basis_dim=int(config["basis_dim"]),
        basis_offset=int(config["basis_offset"]),
        extra_row=extra_row,
        csv_columns=SEARCH_CSV_COLUMNS,
        trend_thetas_dim=int(config["trend_thetas_dim"]),
        latent_dim_override=int(config["latent_dim"]),
        lr_scheduler_config=lr_scheduler_config,
        optimizer_name=str(study_cfg.training.get("optimizer", "Adam")),
        learning_rate=float(study_cfg.training.get("learning_rate", 0.001)),
        loss_override=str(study_cfg.training.get("loss", "SMAPELoss")),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Search Runner
# ---------------------------------------------------------------------------


def _resolve_gpu_count(study_cfg: StudyConfig, cli_args) -> int:
    class _GPUArgs:
        def __init__(self, accelerator: str, n_gpus: int | None):
            self.accelerator = accelerator
            self.n_gpus = n_gpus

    accelerator = cli_args.accelerator or study_cfg.hardware.get("accelerator", "auto")
    requested = cli_args.n_gpus if cli_args.n_gpus is not None else study_cfg.hardware.get("n_gpus")
    return resolve_n_gpus(_GPUArgs(accelerator=accelerator, n_gpus=requested))


def _run_search_round_sequential(
    study_cfg: StudyConfig,
    configs: dict[str, dict],
    round_num: int,
    round_spec: dict,
    csv_path: str,
    cli_args,
):
    dataset = load_dataset(study_cfg.dataset, study_cfg.period)
    train_series_list = dataset.get_training_series()

    max_epochs = int(round_spec.get("max_epochs", 10))
    n_runs = int(round_spec.get("n_runs", 3))
    patience = max_epochs if round_num == 1 else min(max_epochs, EARLY_STOPPING_PATIENCE)

    accelerator = cli_args.accelerator or study_cfg.hardware.get("accelerator", "auto")
    num_workers = int(cli_args.num_workers if cli_args.num_workers is not None else study_cfg.hardware.get("num_workers", 0))
    batch_size_override = cli_args.batch_size if cli_args.batch_size is not None else study_cfg.hardware.get("batch_size")
    batch_size = get_batch_size(study_cfg.dataset, study_cfg.period, batch_size_override)
    base_seed = int(study_cfg.runs.get("base_seed", 42))
    wandb_enabled, wandb_project = _resolve_wandb(study_cfg, cli_args)

    total = len(configs) * n_runs
    done = 0

    print(f"\n  {'-' * 64}")
    print(f"  Round {round_num}: {len(configs)} configs × {n_runs} runs × {max_epochs} epochs")
    print(f"  {'-' * 64}")

    for config_name, config in configs.items():
        if _shutdown_requested:
            print("[SHUTDOWN] stopping search round")
            return

        for run_idx in range(n_runs):
            done += 1
            experiment_name = f"wavelet_v3ae_search_r{round_num}"
            if result_exists(csv_path, experiment_name, config_name, study_cfg.period, run_idx):
                continue

            seed = _resolve_seed(base_seed, run_idx)
            print(f"  [{done}/{total}] {config_name} / run {run_idx} (seed={seed})")
            _run_single_search_job(
                study_cfg=study_cfg,
                config=config,
                dataset=dataset,
                train_series_list=train_series_list,
                csv_path=csv_path,
                round_num=round_num,
                run_idx=run_idx,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                accelerator=accelerator,
                num_workers=num_workers,
                seed=seed,
                wandb_enabled=wandb_enabled,
                wandb_project=wandb_project,
            )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gpu_worker_search(gpu_id: int, job_queue, shutdown_event, worker_args: dict):
    _configure_process_logging_from_env()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    study_cfg: StudyConfig = worker_args["study_cfg"]
    round_num = worker_args["round_num"]
    max_epochs = worker_args["max_epochs"]
    patience = worker_args["patience"]
    csv_path = worker_args["csv_path"]
    batch_size = worker_args["batch_size"]
    num_workers = worker_args["num_workers"]
    base_seed = worker_args["base_seed"]
    wandb_enabled = worker_args["wandb_enabled"]
    wandb_project = worker_args["wandb_project"]

    dataset = load_dataset(study_cfg.dataset, study_cfg.period)
    train_series_list = dataset.get_training_series()

    while not shutdown_event.is_set():
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break

        _run_single_search_job(
            study_cfg=study_cfg,
            config=job["config"],
            dataset=dataset,
            train_series_list=train_series_list,
            csv_path=csv_path,
            round_num=round_num,
            run_idx=job["run_idx"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            accelerator="cuda",
            num_workers=num_workers,
            seed=_resolve_seed(base_seed, job["run_idx"]),
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            gpu_id=gpu_id,
        )


def _run_search_round_parallel(
    study_cfg: StudyConfig,
    configs: dict[str, dict],
    round_num: int,
    round_spec: dict,
    csv_path: str,
    n_gpus: int,
    cli_args,
):
    max_epochs = int(round_spec.get("max_epochs", 10))
    n_runs = int(round_spec.get("n_runs", 3))
    patience = max_epochs if round_num == 1 else min(max_epochs, EARLY_STOPPING_PATIENCE)
    num_workers = int(cli_args.num_workers if cli_args.num_workers is not None else study_cfg.hardware.get("num_workers", 0))
    batch_size_override = cli_args.batch_size if cli_args.batch_size is not None else study_cfg.hardware.get("batch_size")
    batch_size = get_batch_size(study_cfg.dataset, study_cfg.period, batch_size_override)
    base_seed = int(study_cfg.runs.get("base_seed", 42))
    wandb_enabled, wandb_project = _resolve_wandb(study_cfg, cli_args)

    jobs = []
    for config in configs.values():
        for run_idx in range(n_runs):
            if result_exists(
                csv_path,
                f"wavelet_v3ae_search_r{round_num}",
                config["config_name"],
                study_cfg.period,
                run_idx,
            ):
                continue
            jobs.append({"config": config, "run_idx": run_idx})

    print(f"\n  Round {round_num} parallel: {len(jobs)} pending jobs across {n_gpus} GPUs")
    if not jobs:
        return

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for job in jobs:
        q.put(job)

    shutdown_event = ctx.Event()
    worker_args = {
        "study_cfg": study_cfg,
        "round_num": round_num,
        "max_epochs": max_epochs,
        "patience": patience,
        "csv_path": csv_path,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "base_seed": base_seed,
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
    }

    workers = []
    for gid in range(n_gpus):
        p = ctx.Process(target=_gpu_worker_search, args=(gid, q, shutdown_event, worker_args))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    for p in workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_meta_forecaster(study_cfg: StudyConfig) -> MetaForecaster | None:
    if not bool(study_cfg.meta_forecaster.get("enabled", True)):
        return None

    training_csvs = [
        os.path.abspath(p) for p in list(study_cfg.meta_forecaster.get("training_csvs") or [])
    ]
    existing = [p for p in training_csvs if os.path.exists(p)]
    if not existing:
        print("  [INFO] no existing CSVs for meta-forecaster; fallback to val_loss ranking")
        return None

    cache_dir = os.path.abspath(str(study_cfg.meta_forecaster.get("cache_dir", "experiments/results/.meta_cache")))
    mf = MetaForecaster(cache_dir=cache_dir)
    try:
        print(f"  [META] training/loading meta-forecaster from {len(existing)} CSVs")
        mf.train(existing)
    except ValueError as exc:
        print(f"  [WARN] meta-forecaster unavailable: {exc}")
        return None
    return mf


def run_search_mode(study_cfg: StudyConfig, cli_args) -> None:
    search_csv = _search_csv_path(study_cfg)
    init_csv(search_csv, columns=SEARCH_CSV_COLUMNS)

    full_configs = generate_search_configs(study_cfg)
    print(f"\n{'=' * 78}")
    print(f"WaveletV3AE search — dataset={study_cfg.dataset} period={study_cfg.period}")
    print(f"  Config space: {len(full_configs)}")
    print(f"  Stack length: {len(next(iter(full_configs.values()))['stack_types'])}")
    print(f"  Output CSV:   {search_csv}")
    print(f"{'=' * 78}")

    n_gpus = _resolve_gpu_count(study_cfg, cli_args)
    rounds = study_cfg.search_rounds
    if not rounds:
        raise ValueError("search.rounds is empty")

    if cli_args.round is None or cli_args.round == "all":
        rounds_to_run = list(range(1, len(rounds) + 1))
    else:
        rounds_to_run = [int(cli_args.round)]

    meta_forecaster = _train_meta_forecaster(study_cfg)

    promoted: list[str] | None = None

    for round_num in rounds_to_run:
        if _shutdown_requested:
            print("[SHUTDOWN] stopping before next round")
            break

        round_spec = rounds[round_num - 1]

        if round_num == 1:
            configs = generate_search_configs(study_cfg)
        else:
            if promoted is None:
                prev_round = round_num - 1
                prev_spec = rounds[prev_round - 1]
                promoted, _ = rank_and_promote(
                    csv_path=search_csv,
                    round_num=prev_round,
                    keep_fraction=float(prev_spec.get("keep_fraction", 0.67)),
                    top_k=prev_spec.get("top_k"),
                    use_meta=(prev_round == 1),
                )
                if not promoted:
                    raise RuntimeError(
                        f"No promoted configs found from round {prev_round}. "
                        f"Run prior rounds or check results CSV: {search_csv}"
                    )
            configs = generate_search_configs(study_cfg, promoted_config_names=set(promoted))

        if getattr(cli_args, "analyze", False):
            print(f"  [ANALYZE] skipping run for round {round_num}")
        else:
            if n_gpus >= 2:
                _run_search_round_parallel(
                    study_cfg=study_cfg,
                    configs=configs,
                    round_num=round_num,
                    round_spec=round_spec,
                    csv_path=search_csv,
                    n_gpus=n_gpus,
                    cli_args=cli_args,
                )
            else:
                _run_search_round_sequential(
                    study_cfg=study_cfg,
                    configs=configs,
                    round_num=round_num,
                    round_spec=round_spec,
                    csv_path=search_csv,
                    cli_args=cli_args,
                )

        if round_num == 1 and meta_forecaster is not None:
            _update_meta_predictions(search_csv, round_num=1, meta_forecaster=meta_forecaster)

        promoted, _rankings = rank_and_promote(
            csv_path=search_csv,
            round_num=round_num,
            keep_fraction=float(round_spec.get("keep_fraction", 0.67)),
            top_k=round_spec.get("top_k"),
            use_meta=(round_num == 1),
        )

    print(f"\n{'=' * 78}")
    print(f"Search complete — results: {search_csv}")
    print(f"{'=' * 78}")


# ---------------------------------------------------------------------------
# Cross-Dataset Selection + Benchmark
# ---------------------------------------------------------------------------


def _canonical_from_row(row: dict) -> str:
    ttd = _safe_float(row.get("trend_thetas_dim_cfg"), default=float("nan"))
    ld = _safe_float(row.get("latent_dim_cfg"), default=float("nan"))
    if not math.isfinite(ttd) or not math.isfinite(ld):
        return ""
    return canonical_config_id(
        row.get("wavelet_family", ""),
        row.get("bd_label", ""),
        int(ttd),
        int(ld),
    )


def _round3_rankings_for_dataset(search_csv: str) -> tuple[list[tuple[str, float]], dict[str, float], set[str]]:
    rows = _rows_for_round(search_csv, round_num=3)
    if not rows:
        return [], {}, set()

    grouped_vals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        cid = row.get("canonical_config_id") or _canonical_from_row(row)
        if not cid:
            continue
        bvl = _safe_float(row.get("best_val_loss"), default=float("inf"))
        if math.isfinite(bvl):
            grouped_vals[cid].append(bvl)

    ranked = sorted(
        [(cid, float(np.median(vals))) for cid, vals in grouped_vals.items() if vals],
        key=lambda x: (x[1], x[0]),
    )

    percentiles: dict[str, float] = {}
    denom = max(1, len(ranked) - 1)
    for idx, (cid, _metric) in enumerate(ranked):
        percentiles[cid] = idx / denom if len(ranked) > 1 else 0.0

    top10 = {cid for cid, _ in ranked[:10]}
    return ranked, percentiles, top10


def select_global_top10(dataset_rankings: dict[str, tuple[list[tuple[str, float]], dict[str, float], set[str]]]) -> list[dict]:
    candidate_pool: set[str] = set()
    for _dataset, (_ranked, _pct, top10) in dataset_rankings.items():
        candidate_pool.update(top10)

    if not candidate_pool:
        return []

    selected_rows = []
    for cid in sorted(candidate_pool):
        pct_vals = []
        source = []
        for dataset_name, (_ranked, pcts, top10) in dataset_rankings.items():
            if cid in top10:
                source.append(dataset_name)
            pct_vals.append(float(pcts.get(cid, 1.0)))

        selected_rows.append(
            {
                "canonical_config_id": cid,
                "mean_percentile": float(np.mean(pct_vals)),
                "std_percentile": float(np.std(pct_vals)),
                "source_datasets": sorted(source),
            }
        )

    selected_rows.sort(
        key=lambda r: (r["mean_percentile"], r["std_percentile"], r["canonical_config_id"])
    )

    return selected_rows[:10]


def _parse_canonical(canonical_id: str) -> dict:
    wavelet_family, bd_label, ttd_tag, ld_tag = canonical_id.split("|")
    return {
        "wavelet_family": wavelet_family,
        "bd_label": bd_label,
        "trend_thetas_dim": int(ttd_tag.replace("ttd", "")),
        "latent_dim": int(ld_tag.replace("ld", "")),
    }


def _run_single_cross_job(
    dataset_cfg: StudyConfig,
    candidate: dict,
    dataset,
    train_series_list,
    run_idx: int,
    csv_path: str,
    batch_size: int,
    accelerator: str,
    num_workers: int,
    seed: int,
    max_epochs: int,
    patience: int,
    wandb_enabled: bool,
    wandb_project: str,
    gpu_id: int | None = None,
):
    parsed = _parse_canonical(candidate["canonical_config_id"])

    forecast_length, backcast_length = _forecast_backcast_lengths(
        dataset_cfg.dataset,
        dataset_cfg.period,
        forecast_multiplier_override=dataset_cfg.training.get("forecast_multiplier"),
    )
    basis_dim = compute_basis_dim(parsed["bd_label"], forecast_length, backcast_length)

    repeats = int(dataset_cfg.architecture.get("repeats", 5))
    trend_block = str(dataset_cfg.architecture.get("trend_block", "TrendAE"))
    config_name = config_name_from_canonical(candidate["canonical_config_id"])

    extra_row = {
        "search_round": "",
        "basis_dim": int(basis_dim),
        "basis_offset": int(dataset_cfg.training.get("basis_offset", 0)),
        "trend_thetas_dim_cfg": int(parsed["trend_thetas_dim"]),
        "wavelet_family": parsed["wavelet_family"],
        "bd_label": parsed["bd_label"],
        "latent_dim_cfg": int(parsed["latent_dim"]),
        "meta_predicted_best": "",
        "meta_convergence_score": "",
        "source_datasets": ",".join(candidate.get("source_datasets", [])),
        "global_selection_score": f"{candidate['mean_percentile']:.8f}",
        "canonical_config_id": candidate["canonical_config_id"],
    }

    lr_scheduler_config = resolve_lr_scheduler(dataset_cfg.lr_scheduler, max_epochs)

    run_single_experiment(
        experiment_name="wavelet_v3ae_cross_dataset",
        config_name=config_name,
        category="wavelet_v3ae_cross_dataset",
        stack_types=[trend_block, parsed["wavelet_family"]] * repeats,
        period=dataset_cfg.period,
        run_idx=run_idx,
        dataset=dataset,
        train_series_list=train_series_list,
        csv_path=csv_path,
        n_blocks_per_stack=int(dataset_cfg.training.get("n_blocks_per_stack", 1)),
        share_weights=bool(dataset_cfg.training.get("share_weights", True)),
        active_g=False,
        sum_losses=bool(dataset_cfg.training.get("sum_losses", False)),
        activation=str(dataset_cfg.training.get("activation", "ReLU")),
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        accelerator_override=accelerator,
        forecast_multiplier=int(dataset_cfg.training.get("forecast_multiplier") or FORECAST_MULTIPLIERS[dataset_cfg.dataset]),
        num_workers=num_workers,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        save_predictions=False,
        gpu_id=gpu_id,
        basis_dim=int(basis_dim),
        basis_offset=int(dataset_cfg.training.get("basis_offset", 0)),
        extra_row=extra_row,
        csv_columns=CROSS_CSV_COLUMNS,
        trend_thetas_dim=int(parsed["trend_thetas_dim"]),
        latent_dim_override=int(parsed["latent_dim"]),
        lr_scheduler_config=lr_scheduler_config,
        optimizer_name=str(dataset_cfg.training.get("optimizer", "Adam")),
        learning_rate=float(dataset_cfg.training.get("learning_rate", 0.001)),
        loss_override=str(dataset_cfg.training.get("loss", "SMAPELoss")),
        seed=seed,
    )


def run_cross_mode(study_cfgs: list[StudyConfig], cli_args) -> None:
    by_dataset = {cfg.dataset: cfg for cfg in study_cfgs}
    required = {"m4", "tourism", "traffic", "weather"}
    missing = sorted(required - set(by_dataset.keys()))
    if missing:
        raise ValueError(f"cross mode requires configs for {sorted(required)}; missing {missing}")

    dataset_rankings = {}
    for dataset_name, cfg in sorted(by_dataset.items()):
        search_csv = _search_csv_path(cfg)
        ranked, percentiles, top10 = _round3_rankings_for_dataset(search_csv)
        dataset_rankings[dataset_name] = (ranked, percentiles, top10)
        print(
            f"  {dataset_name:<8} round3 ranked={len(ranked):<4} top10={len(top10):<2} csv={search_csv}"
        )

    global_top10 = select_global_top10(dataset_rankings)
    if not global_top10:
        raise RuntimeError("No global top-10 could be selected; ensure round-3 search results exist.")

    print(f"\n{'=' * 78}")
    print("Global top-10 candidates (mean percentile across datasets)")
    print(f"{'=' * 78}")
    for idx, c in enumerate(global_top10, start=1):
        print(
            f"  {idx:>2}. {c['canonical_config_id']} "
            f"mean={c['mean_percentile']:.4f} std={c['std_percentile']:.4f} "
            f"src={','.join(c['source_datasets'])}"
        )

    cross_csv = _cross_csv_path(study_cfgs[0])
    init_csv(cross_csv, columns=CROSS_CSV_COLUMNS)

    n_runs = 3
    max_epochs = 50
    patience = min(max_epochs, EARLY_STOPPING_PATIENCE)

    for dataset_name, cfg in sorted(by_dataset.items()):
        if _shutdown_requested:
            print("[SHUTDOWN] stopping cross benchmark")
            break

        dataset = load_dataset(cfg.dataset, cfg.period)
        train_series_list = dataset.get_training_series()

        accelerator = cli_args.accelerator or cfg.hardware.get("accelerator", "auto")
        num_workers = int(cli_args.num_workers if cli_args.num_workers is not None else cfg.hardware.get("num_workers", 0))
        batch_size_override = cli_args.batch_size if cli_args.batch_size is not None else cfg.hardware.get("batch_size")
        batch_size = get_batch_size(cfg.dataset, cfg.period, batch_size_override)
        base_seed = int(cfg.runs.get("base_seed", 42))
        wandb_enabled, wandb_project = _resolve_wandb(cfg, cli_args)

        print(f"\n  Dataset={dataset_name} period={cfg.period}")
        for candidate in global_top10:
            parsed = _parse_canonical(candidate["canonical_config_id"])
            config_name = config_name_from_canonical(candidate["canonical_config_id"])

            for run_idx in range(n_runs):
                if result_exists(
                    cross_csv,
                    "wavelet_v3ae_cross_dataset",
                    config_name,
                    cfg.period,
                    run_idx,
                ):
                    continue

                print(f"    {config_name} / run {run_idx}")
                _run_single_cross_job(
                    dataset_cfg=cfg,
                    candidate=candidate,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    run_idx=run_idx,
                    csv_path=cross_csv,
                    batch_size=batch_size,
                    accelerator=accelerator,
                    num_workers=num_workers,
                    seed=_resolve_seed(base_seed, run_idx),
                    max_epochs=max_epochs,
                    patience=patience,
                    wandb_enabled=wandb_enabled,
                    wandb_project=wandb_project,
                )

            # Keep linter from considering parsed unused if optimizations change.
            _ = parsed

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 78}")
    print(f"Cross benchmark complete — results: {cross_csv}")
    print(f"{'=' * 78}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_many_configs(config_paths: list[str]) -> list[StudyConfig]:
    loaded = [load_study_config(path) for path in config_paths]
    unique = {cfg.dataset: cfg for cfg in loaded}
    return list(unique.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WaveletV3AE successive-halving + cross-dataset runner"
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    p_search = sub.add_parser("search", help="Run successive halving for one dataset config")
    p_search.add_argument("--config", required=True, help="Path to dataset YAML config")
    p_search.add_argument("--round", default=None, help="Specific round to run (1/2/3) or 'all'")
    p_search.add_argument("--analyze", action="store_true", help="Only rank existing results")
    p_search.add_argument("--accelerator", default=None, choices=["auto", "cuda", "mps", "cpu"])
    p_search.add_argument("--n-gpus", type=int, default=None)
    p_search.add_argument("--batch-size", type=int, default=None)
    p_search.add_argument("--num-workers", type=int, default=None)
    p_search.add_argument("--wandb", action="store_true", help="Force-enable W&B logging")
    p_search.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    p_search.add_argument("--wandb-project", default=None, help="Override W&B project name")
    p_search.add_argument("--log-path", default=None, help="Optional explicit log file path (default: /tmp/*.log)")
    p_search.add_argument("--no-tmp-log", action="store_true", help="Disable auto /tmp log teeing")

    p_cross = sub.add_parser("cross", help="Run cross-dataset benchmark from 4 dataset configs")
    p_cross.add_argument("--configs", nargs="+", required=True, help="List of dataset YAML configs")
    p_cross.add_argument("--accelerator", default=None, choices=["auto", "cuda", "mps", "cpu"])
    p_cross.add_argument("--batch-size", type=int, default=None)
    p_cross.add_argument("--num-workers", type=int, default=None)
    p_cross.add_argument("--wandb", action="store_true", help="Force-enable W&B logging")
    p_cross.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    p_cross.add_argument("--wandb-project", default=None, help="Override W&B project name")
    p_cross.add_argument("--log-path", default=None, help="Optional explicit log file path (default: /tmp/*.log)")
    p_cross.add_argument("--no-tmp-log", action="store_true", help="Disable auto /tmp log teeing")

    p_all = sub.add_parser("all", help="Run search for all configs then run cross benchmark")
    p_all.add_argument("--configs", nargs="+", required=True, help="List of dataset YAML configs")
    p_all.add_argument("--round", default=None, help="Specific round for search stage or 'all'")
    p_all.add_argument("--accelerator", default=None, choices=["auto", "cuda", "mps", "cpu"])
    p_all.add_argument("--n-gpus", type=int, default=None)
    p_all.add_argument("--batch-size", type=int, default=None)
    p_all.add_argument("--num-workers", type=int, default=None)
    p_all.add_argument("--wandb", action="store_true", help="Force-enable W&B logging")
    p_all.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    p_all.add_argument("--wandb-project", default=None, help="Override W&B project name")
    p_all.add_argument("--log-path", default=None, help="Optional explicit log file path (default: /tmp/*.log)")
    p_all.add_argument("--no-tmp-log", action="store_true", help="Disable auto /tmp log teeing")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "no_tmp_log", False):
        log_path = args.log_path
        if not log_path:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"/tmp/wavelet_v3ae_study_{ts}_{os.getpid()}.log"
        _configure_process_logging(log_path)
        os.environ["WAVELET_V3AE_LOG_PATH"] = log_path
        print(f"[LOG] Writing mirrored run logs to: {log_path}")

    if args.mode == "search":
        cfg = load_study_config(args.config)
        run_search_mode(cfg, args)
        return

    if args.mode == "cross":
        cfgs = _load_many_configs(args.configs)
        run_cross_mode(cfgs, args)
        return

    if args.mode == "all":
        cfgs = _load_many_configs(args.configs)
        for cfg in sorted(cfgs, key=lambda c: c.dataset):
            if _shutdown_requested:
                print("[SHUTDOWN] stopping all-mode search stage")
                return
            print(f"\n{'#' * 78}")
            print(f"Search stage for dataset={cfg.dataset}")
            print(f"{'#' * 78}")
            run_search_mode(cfg, args)
        run_cross_mode(cfgs, args)
        return

    parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
