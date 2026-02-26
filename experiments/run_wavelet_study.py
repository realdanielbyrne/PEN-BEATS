"""
Hierarchical Wavelet Frequency-Band Experiment

Pilot study (n_runs=5) testing three hypotheses:
  A) Frequency window: which SVD-offset band works best in isolation?
  B) Targeted Trend+Wavelet: does shifting the wavelet away from DC help?
  C) N-HiTS-style hierarchical tiers: three non-overlapping frequency bands
     stacked in the N-BEATS doubly-residual architecture.

All configs use basis_dim=16 (narrow window), contrast with the unified
benchmark's basis_dim=128. Datasets: M4-Yearly (fast baseline), Traffic-96,
Weather-96.

Results → experiments/results/<dataset>/wavelet_study_results.csv

Usage:
    # Single dataset
    python experiments/run_wavelet_study.py --dataset m4 --periods Yearly
    python experiments/run_wavelet_study.py --dataset traffic --periods Traffic-96
    python experiments/run_wavelet_study.py --dataset weather --periods Weather-96

    # All three (M4-Yearly, Traffic-96, Weather-96)
    python experiments/run_wavelet_study.py --dataset all

    # Smoke test (1 run, 2 epochs)
    python experiments/run_wavelet_study.py --dataset traffic --periods Traffic-96 \\
        --n-runs 1 --max-epochs 2
"""

import argparse
import csv
import gc
import json
import math
import os
import queue
import signal
import sys
import time

import numpy as np
import torch
import lightning.pytorch as pl

# Allow running from project root or experiments/
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXPERIMENTS_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", "src"))

# Import shared infrastructure from the unified benchmark
from run_unified_benchmark import (
    run_single_experiment,
    result_exists,
    get_batch_size,
    init_csv,
    append_result,
    load_dataset,
    resolve_n_gpus,
    CSV_COLUMNS,
    N_RUNS_DEFAULT,
    MAX_EPOCHS,
    PATIENCE,
    THETAS_DIM,
    LOSS,
    LEARNING_RATE,
    LATENT_DIM,
    BASIS_DIM,
    BASE_SEED,
    FORECAST_MULTIPLIERS,
    DATASET_PERIODS,
    TRAFFIC_PERIODS,
    WEATHER_PERIODS,
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    _shutdown_requested,
)
import multiprocessing as mp

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Wavelet Study Constants
# ---------------------------------------------------------------------------

WAVELET_STUDY_N_RUNS = 5
WAVELET_BASIS_DIM = 16    # narrow window — study basis_offset sensitivity

RESULTS_DIR = os.path.join(_EXPERIMENTS_DIR, "results")

# Extended CSV schema — adds basis_dim and basis_offset columns
WAVELET_CSV_COLUMNS = CSV_COLUMNS + ["basis_dim", "basis_offset", "forecast_basis_dim"]

# Datasets covered by this study (subset of DATASET_PERIODS)
WAVELET_DATASETS = {
    "m4": {
        "periods": {"Yearly": DATASET_PERIODS["m4"]["Yearly"]},
        "forecast_multiplier": FORECAST_MULTIPLIERS["m4"],
    },
    "traffic": {
        "periods": {"Traffic-96": TRAFFIC_PERIODS["Traffic-96"]},
        "forecast_multiplier": FORECAST_MULTIPLIERS["traffic"],
    },
    "weather": {
        "periods": {"Weather-96": WEATHER_PERIODS["Weather-96"]},
        "forecast_multiplier": FORECAST_MULTIPLIERS["weather"],
    },
}

# ---------------------------------------------------------------------------
# 16 Wavelet Study Configs
# ---------------------------------------------------------------------------
# All configs: n_blocks_per_stack=1, share_weights=True (unless noted).
# basis_dim=16 throughout. stack_basis_offsets is per-stack; basis_offset is
# model-level (used when stack_basis_offsets is None).
#
# Frequency bands (basis_dim=16 each, non-overlapping):
#   LF: offset=0  → SVD rows  0-15  (smoothest / near-DC)
#   MF: offset=16 → SVD rows 16-31  (seasonal / oscillatory)
#   HF: offset=32 → SVD rows 32-47  (high-freq detail)

WAVELET_STUDY_CONFIGS = {

    # -------------------------------------------------------------------
    # Group A — Offset sweep (8 configs)
    # Single-band homogeneous 30-stack. Establishes which frequency window
    # is most predictive in isolation.
    # -------------------------------------------------------------------
    "Coif2_off0": {
        "category": "group_a",
        "stack_types": ["Coif2WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 0,
        "stack_basis_offsets": None,
    },
    "Coif2_off8": {
        "category": "group_a",
        "stack_types": ["Coif2WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 8,
        "stack_basis_offsets": None,
    },
    "Coif2_off16": {
        "category": "group_a",
        "stack_types": ["Coif2WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 16,
        "stack_basis_offsets": None,
    },
    "Coif2_off24": {
        "category": "group_a",
        "stack_types": ["Coif2WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 24,
        "stack_basis_offsets": None,
    },
    "DB3_off0": {
        "category": "group_a",
        "stack_types": ["DB3WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 0,
        "stack_basis_offsets": None,
    },
    "DB3_off8": {
        "category": "group_a",
        "stack_types": ["DB3WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 8,
        "stack_basis_offsets": None,
    },
    "DB3_off16": {
        "category": "group_a",
        "stack_types": ["DB3WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 16,
        "stack_basis_offsets": None,
    },
    "DB3_off24": {
        "category": "group_a",
        "stack_types": ["DB3WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 24,
        "stack_basis_offsets": None,
    },

    # -------------------------------------------------------------------
    # Group B — Trend + targeted frequency (4 configs)
    # Bridges from current best (Trend+Coif2/DB3 at offset=0) toward
    # 3-tier design. Tests whether non-DC targeting improves things.
    # -------------------------------------------------------------------
    "Trend+Coif2_LF": {
        "category": "group_b",
        "stack_types": ["Trend", "Coif2WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 0,
        "stack_basis_offsets": None,
    },
    "Trend+Coif2_MF": {
        "category": "group_b",
        "stack_types": ["Trend", "Coif2WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 16,
        "stack_basis_offsets": None,
    },
    "Trend+DB3_LF": {
        "category": "group_b",
        "stack_types": ["Trend", "DB3WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 0,
        "stack_basis_offsets": None,
    },
    "Trend+DB3_MF": {
        "category": "group_b",
        "stack_types": ["Trend", "DB3WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 16,
        "stack_basis_offsets": None,
    },

    # -------------------------------------------------------------------
    # Group C — 3-tier hierarchical stacking (4 configs)
    # N-HiTS-style: LF / MF / HF bands handled by successive tier groups.
    # Variant 1 (pure): all wavelet, non-overlapping bands.
    # Variant 2 (stable): one Trend per triplet guards against divergence.
    # -------------------------------------------------------------------

    # Pure: 30 stacks, 10 per tier
    "Coif2_3tier_pure": {
        "category": "group_c",
        "stack_types": ["Coif2WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 0,
        "stack_basis_offsets": [0] * 10 + [16] * 10 + [32] * 10,
    },
    "DB3_3tier_pure": {
        "category": "group_c",
        "stack_types": ["DB3WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 0,
        "stack_basis_offsets": [0] * 10 + [16] * 10 + [32] * 10,
    },

    # Stable: 45 stacks (3 × [Trend, Wav, Wav] × 5), Trend slots ignore offset.
    # stack_basis_offsets assigns LF to tier-1 wavelet stacks, MF to tier-2,
    # HF to tier-3. Trend stacks receive 0 (ignored by the block).
    "Coif2_3tier_stable": {
        "category": "group_c",
        "stack_types": (
            ["Trend", "Coif2WaveletV3", "Coif2WaveletV3"] * 5
            + ["Trend", "Coif2WaveletV3", "Coif2WaveletV3"] * 5
            + ["Trend", "Coif2WaveletV3", "Coif2WaveletV3"] * 5
        ),
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 0,
        "stack_basis_offsets": [0, 0, 0] * 5 + [0, 16, 16] * 5 + [0, 32, 32] * 5,
    },
    "DB3_3tier_stable": {
        "category": "group_c",
        "stack_types": (
            ["Trend", "DB3WaveletV3", "DB3WaveletV3"] * 5
            + ["Trend", "DB3WaveletV3", "DB3WaveletV3"] * 5
            + ["Trend", "DB3WaveletV3", "DB3WaveletV3"] * 5
        ),
        "n_blocks_per_stack": 1,
        "share_weights": True,
        "basis_dim": 16,
        "basis_offset": 0,
        "stack_basis_offsets": [0, 0, 0] * 5 + [0, 16, 16] * 5 + [0, 32, 32] * 5,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wavelet_csv_path(dataset_name):
    return os.path.join(RESULTS_DIR, dataset_name, "wavelet_study_results.csv")


def _get_batch_size(dataset_name, period, override=None):
    """Look up per-dataset/period batch size, with optional override."""
    tuned = BATCH_SIZES.get((dataset_name, period))
    if tuned is not None:
        return tuned
    if override is not None:
        return override
    return DEFAULT_BATCH_SIZE


# ---------------------------------------------------------------------------
# Single Wavelet Experiment Wrapper
# ---------------------------------------------------------------------------

def run_wavelet_experiment(
    config_name,
    cfg,
    period,
    run_idx,
    dataset,
    train_series_list,
    csv_path,
    pass_name,
    active_g,
    max_epochs,
    patience,
    batch_size,
    accelerator_override,
    forecast_multiplier,
    num_workers=0,
    gpu_id=None,
):
    """Wrapper around run_single_experiment for wavelet study configs.

    Extracts wavelet-specific params from cfg and passes them as extra_row
    so they appear in the extended CSV schema.
    """
    basis_dim = cfg.get("basis_dim", WAVELET_BASIS_DIM)
    basis_offset = cfg.get("basis_offset", 0)
    stack_offsets = cfg.get("stack_basis_offsets", None)
    forecast_basis_dim = cfg.get("forecast_basis_dim", None)

    extra_row = {
        "basis_dim": basis_dim,
        "basis_offset": basis_offset,
        "forecast_basis_dim": forecast_basis_dim if forecast_basis_dim is not None else "",
    }

    run_single_experiment(
        experiment_name=pass_name,
        config_name=config_name,
        category=cfg["category"],
        stack_types=cfg["stack_types"],
        period=period,
        run_idx=run_idx,
        dataset=dataset,
        train_series_list=train_series_list,
        csv_path=csv_path,
        n_blocks_per_stack=cfg["n_blocks_per_stack"],
        share_weights=cfg["share_weights"],
        active_g=active_g,
        sum_losses=False,
        activation="ReLU",
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        accelerator_override=accelerator_override,
        forecast_multiplier=forecast_multiplier,
        num_workers=num_workers,
        wandb_enabled=False,
        save_predictions=False,
        gpu_id=gpu_id,
        basis_dim=basis_dim,
        basis_offset=basis_offset,
        stack_basis_offsets=stack_offsets,
        forecast_basis_dim=forecast_basis_dim,
        extra_row=extra_row,
        csv_columns=WAVELET_CSV_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Job List Builder
# ---------------------------------------------------------------------------

def _build_wavelet_job_list(periods, n_runs, dataset_name, batch_size_override):
    """Build flat list of job dicts for all (period, pass, config, run_idx)."""
    passes = [
        ("baseline", False),
        ("activeG_fcast", "forecast"),
    ]
    jobs = []
    for period in periods:
        batch_size = _get_batch_size(dataset_name, period, batch_size_override)
        for pass_name, active_g in passes:
            for config_name, cfg in WAVELET_STUDY_CONFIGS.items():
                for run_idx in range(n_runs):
                    jobs.append({
                        "period": period,
                        "pass_name": pass_name,
                        "active_g": active_g,
                        "config_name": config_name,
                        "cfg": cfg,
                        "run_idx": run_idx,
                        "batch_size": batch_size,
                    })
    return jobs


# ---------------------------------------------------------------------------
# GPU Worker (parallel execution)
# ---------------------------------------------------------------------------

def _wavelet_gpu_worker(gpu_id, job_queue, shutdown_event, worker_args):
    """Worker process: pins to GPU gpu_id, pulls jobs from shared queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    torch.set_float32_matmul_precision("medium")

    prefix = f"[GPU {gpu_id}]"
    print(f"{prefix} Wavelet worker started (CUDA_VISIBLE_DEVICES={gpu_id}).")

    dataset_cache = {}
    series_cache = {}

    while not shutdown_event.is_set():
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break

        dataset_name = worker_args["dataset_name"]
        period = job["period"]

        if period not in dataset_cache:
            dataset_cache[period] = load_dataset(dataset_name, period)
            series_cache[period] = dataset_cache[period].get_training_series()
        dataset = dataset_cache[period]
        train_series_list = series_cache[period]

        run_wavelet_experiment(
            config_name=job["config_name"],
            cfg=job["cfg"],
            period=period,
            run_idx=job["run_idx"],
            dataset=dataset,
            train_series_list=train_series_list,
            csv_path=worker_args["csv_path"],
            pass_name=job["pass_name"],
            active_g=job["active_g"],
            max_epochs=worker_args["max_epochs"],
            patience=worker_args["patience"],
            batch_size=job["batch_size"],
            accelerator_override="cuda",
            forecast_multiplier=worker_args["forecast_multiplier"],
            num_workers=worker_args["num_workers"],
            gpu_id=gpu_id,
        )

    print(f"{prefix} Wavelet worker finished.")


# ---------------------------------------------------------------------------
# Sequential Runner
# ---------------------------------------------------------------------------

def _run_wavelet_sequential(
    dataset_name, periods, n_runs, max_epochs, patience,
    forecast_multiplier, batch_size_override, csv_path, accelerator,
    num_workers,
):
    passes = [
        ("baseline", False),
        ("activeG_fcast", "forecast"),
    ]

    for period in periods:
        print(f"\n{'='*60}")
        print(f"  Wavelet Study — {dataset_name.upper()} / {period}")
        print(f"{'='*60}")

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()
        batch_size = _get_batch_size(dataset_name, period, batch_size_override)

        for pass_name, active_g in passes:
            print(f"\n  --- Pass: {pass_name} (active_g={active_g}) ---")

            for config_name, cfg in WAVELET_STUDY_CONFIGS.items():
                for run_idx in range(n_runs):
                    run_wavelet_experiment(
                        config_name=config_name,
                        cfg=cfg,
                        period=period,
                        run_idx=run_idx,
                        dataset=dataset,
                        train_series_list=train_series_list,
                        csv_path=csv_path,
                        pass_name=pass_name,
                        active_g=active_g,
                        max_epochs=max_epochs,
                        patience=patience,
                        batch_size=batch_size,
                        accelerator_override=accelerator,
                        forecast_multiplier=forecast_multiplier,
                        num_workers=num_workers,
                    )


# ---------------------------------------------------------------------------
# Parallel Runner
# ---------------------------------------------------------------------------

def _run_wavelet_parallel(
    dataset_name, periods, n_runs, max_epochs, patience,
    forecast_multiplier, batch_size_override, csv_path, n_gpus, num_workers,
):
    jobs = _build_wavelet_job_list(periods, n_runs, dataset_name, batch_size_override)

    pending_jobs = [
        job for job in jobs
        if not result_exists(csv_path, job["pass_name"], job["config_name"],
                             job["period"], job["run_idx"])
    ]

    n_complete = len(jobs) - len(pending_jobs)
    print(f"  Jobs: {len(jobs)} total, {len(pending_jobs)} pending, "
          f"{n_complete} already complete")

    if not pending_jobs:
        print("  All jobs already complete!")
        return

    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    for job in pending_jobs:
        job_queue.put(job)

    shutdown_event = ctx.Event()

    worker_args = {
        "dataset_name": dataset_name,
        "csv_path": csv_path,
        "max_epochs": max_epochs,
        "patience": patience,
        "forecast_multiplier": forecast_multiplier,
        "num_workers": num_workers,
    }

    workers = []
    for gid in range(n_gpus):
        p = ctx.Process(
            target=_wavelet_gpu_worker,
            args=(gid, job_queue, shutdown_event, worker_args),
        )
        p.start()
        workers.append(p)

    print(f"  Spawned {n_gpus} GPU worker processes.")

    for p in workers:
        p.join()

    for p in workers:
        if p.is_alive():
            print(f"  [WARN] Terminating worker PID {p.pid}")
            p.terminate()
            p.join(timeout=10)


# ---------------------------------------------------------------------------
# Per-Dataset Orchestrator
# ---------------------------------------------------------------------------

def _run_wavelet_dataset(dataset_name, periods_filter, args, n_gpus):
    """Run wavelet study for a single dataset."""

    if dataset_name == "m4":
        all_periods = {"Yearly": DATASET_PERIODS["m4"]["Yearly"]}
        forecast_multiplier = FORECAST_MULTIPLIERS["m4"]
    elif dataset_name == "traffic":
        all_periods = {"Traffic-96": TRAFFIC_PERIODS["Traffic-96"]}
        forecast_multiplier = FORECAST_MULTIPLIERS["traffic"]
    elif dataset_name == "weather":
        all_periods = {"Weather-96": WEATHER_PERIODS["Weather-96"]}
        forecast_multiplier = FORECAST_MULTIPLIERS["weather"]
    else:
        print(f"[ERROR] dataset '{dataset_name}' not supported in wavelet study. "
              f"Choose from: m4, traffic, weather.")
        return

    if periods_filter:
        periods = [p for p in periods_filter if p in all_periods]
        if not periods:
            print(f"[ERROR] No valid periods for '{dataset_name}'. "
                  f"Available: {list(all_periods.keys())}")
            return
    else:
        periods = list(all_periods.keys())

    n_runs = args.n_runs if args.n_runs is not None else WAVELET_STUDY_N_RUNS
    max_epochs = args.max_epochs
    patience = EARLY_STOPPING_PATIENCE

    csv_path = _wavelet_csv_path(dataset_name)
    init_csv(csv_path, columns=WAVELET_CSV_COLUMNS)

    total_configs = len(WAVELET_STUDY_CONFIGS)
    total_passes = 2

    print(f"\n{'='*70}")
    print(f"Wavelet Study — {dataset_name.upper()}")
    print(f"  Periods:  {periods}")
    print(f"  Configs:  {total_configs}  |  Passes: {total_passes}  |  Runs/config: {n_runs}")
    print(f"  Max epochs: {max_epochs}  |  Patience: {patience}")
    print(f"  Total runs per period: {total_configs * total_passes * n_runs}")
    print(f"  Results: {csv_path}")
    if n_gpus >= 2:
        print(f"  GPUs: {n_gpus} (parallel execution)")
    else:
        print(f"  Mode: sequential")
    print(f"{'='*70}")

    if n_gpus >= 2:
        _run_wavelet_parallel(
            dataset_name=dataset_name,
            periods=periods,
            n_runs=n_runs,
            max_epochs=max_epochs,
            patience=patience,
            forecast_multiplier=forecast_multiplier,
            batch_size_override=args.batch_size,
            csv_path=csv_path,
            n_gpus=n_gpus,
            num_workers=args.num_workers,
        )
    else:
        _run_wavelet_sequential(
            dataset_name=dataset_name,
            periods=periods,
            n_runs=n_runs,
            max_epochs=max_epochs,
            patience=patience,
            forecast_multiplier=forecast_multiplier,
            batch_size_override=args.batch_size,
            csv_path=csv_path,
            accelerator=args.accelerator,
            num_workers=args.num_workers,
        )

    print(f"\n{'='*70}")
    print(f"Wavelet Study COMPLETE — {dataset_name.upper()}")
    print(f"Results: {csv_path}")
    print(f"{'='*70}")


# ===========================================================================
# SEARCH MODE — Hyperparameter Search with N-BEATS Meta-Forecaster
# ===========================================================================

from meta_forecaster import MetaForecaster

# ---------------------------------------------------------------------------
# Search Constants
# ---------------------------------------------------------------------------

SEARCH_WAVELETS = [
    "HaarWaveletV3", "DB2WaveletV3", "DB3WaveletV3", "DB4WaveletV3",
    "DB10WaveletV3", "Coif2WaveletV3", "Coif3WaveletV3",
    "Symlet3WaveletV3", "Symlet10WaveletV3",
]
SEARCH_BASIS_DIMS = [4, 8, 16, 32, 48, 64, 96]
SEARCH_BASIS_OFFSETS = [0, 8, 16, 32, 48]
SEARCH_N_STACKS = [10, 20, 30]
SEARCH_ARCH_PATTERNS = ["homogeneous", "trend_mixed", "trend_season_mixed"]

# Successive halving schedule (aggressive: keep top 33%)
ROUND_SCHEDULE = {
    1: {"max_epochs": 6,   "n_runs": 1, "keep_fraction": 0.33},
    2: {"max_epochs": 15,  "n_runs": 1, "keep_fraction": 0.33},
    3: {"max_epochs": 30,  "n_runs": 3, "keep_fraction": 0.33},
    4: {"max_epochs": 100, "n_runs": 5, "keep_fraction": 1.0},
}

# Extended CSV schema for search mode
SEARCH_CSV_COLUMNS = WAVELET_CSV_COLUMNS + [
    "search_round", "arch_pattern", "wavelet_family",
    "n_stacks_requested", "meta_predicted_best", "meta_convergence_score",
]

# Known existing CSVs with val_loss curves for meta-forecaster training
_META_TRAINING_CSVS = [
    os.path.join(RESULTS_DIR, "m4", "unified_benchmark_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "block_benchmark_results.csv"),
    os.path.join(RESULTS_DIR, "traffic", "block_benchmark_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "convergence_study_results_v1.csv"),
    os.path.join(RESULTS_DIR, "weather", "convergence_study_results_v1.csv"),
    os.path.join(RESULTS_DIR, "traffic", "convergence_study_results_v1.csv"),
]

META_CACHE_DIR = os.path.join(RESULTS_DIR, ".meta_cache")


# ---------------------------------------------------------------------------
# Search Helpers
# ---------------------------------------------------------------------------

def _search_csv_path(dataset_name):
    return os.path.join(RESULTS_DIR, dataset_name, "wavelet_search_results.csv")


def is_valid_config(basis_dim, basis_offset, forecast_length):
    """Check if a (basis_dim, basis_offset) pair is valid for a given forecast_length.

    Filters out degenerate configs where the effective dimension after SVD
    clamping would be less than 25% of the requested basis_dim or less than 2.
    """
    if basis_offset >= forecast_length:
        return False
    effective_dim = min(basis_dim, forecast_length - basis_offset)
    if effective_dim < max(2, basis_dim * 0.25):
        return False
    return True


def _build_stack_types(wavelet_family, arch_pattern, n_stacks):
    """Build a stack_types list for a given architecture pattern."""
    if arch_pattern == "homogeneous":
        return [wavelet_family] * n_stacks
    elif arch_pattern == "trend_mixed":
        return ["Trend", wavelet_family] * (n_stacks // 2)
    elif arch_pattern == "trend_season_mixed":
        return ["Trend", "Seasonality", wavelet_family] * (n_stacks // 3)
    else:
        raise ValueError(f"Unknown arch_pattern: {arch_pattern}")


def _short_wavelet_name(wavelet_family):
    """Shorten 'DB3WaveletV3' to 'DB3' for config naming."""
    return wavelet_family.replace("WaveletV3", "")


_ARCH_ABBREVS = {
    "homogeneous": "homog",
    "trend_mixed": "trmix",
    "trend_season_mixed": "tsmix",
}


def generate_search_configs(forecast_length, round_num=1, promoted_configs=None):
    """Generate wavelet search configs programmatically.

    Parameters
    ----------
    forecast_length : int
        Forecast length for the target dataset (used for validity filtering).
    round_num : int
        Search round (1-4). Round 1 generates the full grid; rounds 2-4
        filter to promoted_configs.
    promoted_configs : list[str] or None
        Config names promoted from the prior round (rounds 2-4 only).

    Returns
    -------
    dict[str, dict]
        Config name -> config dict.
    """
    configs = {}

    for wavelet in SEARCH_WAVELETS:
        short_name = _short_wavelet_name(wavelet)
        for arch in SEARCH_ARCH_PATTERNS:
            arch_abbrev = _ARCH_ABBREVS[arch]

            # trend_season_mixed requires n_stacks divisible by 3
            stacks_list = SEARCH_N_STACKS
            if arch == "trend_season_mixed":
                stacks_list = [s for s in SEARCH_N_STACKS if s % 3 == 0]

            for n_stacks in stacks_list:
                for basis_dim in SEARCH_BASIS_DIMS:
                    for basis_offset in SEARCH_BASIS_OFFSETS:
                        if not is_valid_config(basis_dim, basis_offset, forecast_length):
                            continue

                        config_name = (
                            f"{short_name}_{arch_abbrev}_s{n_stacks}"
                            f"_d{basis_dim}_o{basis_offset}"
                        )

                        # For rounds 2+, only include promoted configs
                        if promoted_configs is not None and config_name not in promoted_configs:
                            continue

                        stack_types = _build_stack_types(wavelet, arch, n_stacks)

                        configs[config_name] = {
                            "category": f"search_round{round_num}",
                            "stack_types": stack_types,
                            "n_blocks_per_stack": 1,
                            "share_weights": True,
                            "basis_dim": basis_dim,
                            "basis_offset": basis_offset,
                            "stack_basis_offsets": None,
                            "arch_pattern": arch,
                            "wavelet_family": wavelet,
                            "n_stacks_requested": n_stacks,
                        }

    return configs


# ---------------------------------------------------------------------------
# Search Job Builder
# ---------------------------------------------------------------------------

def _build_search_job_list(configs, periods, n_runs, dataset_name, batch_size_override,
                           round_num):
    """Build flat list of job dicts for search mode."""
    jobs = []
    for period in periods:
        batch_size = _get_batch_size(dataset_name, period, batch_size_override)
        for config_name, cfg in configs.items():
            for run_idx in range(n_runs):
                jobs.append({
                    "period": period,
                    "pass_name": f"search_r{round_num}",
                    "active_g": False,
                    "config_name": config_name,
                    "cfg": cfg,
                    "run_idx": run_idx,
                    "batch_size": batch_size,
                    "round_num": round_num,
                })
    return jobs


def run_search_experiment(
    config_name, cfg, period, run_idx, dataset, train_series_list,
    csv_path, round_num, active_g, max_epochs, patience, batch_size,
    accelerator_override, forecast_multiplier, num_workers=0, gpu_id=None,
):
    """Wrapper around run_single_experiment for search mode configs."""
    basis_dim = cfg.get("basis_dim", WAVELET_BASIS_DIM)
    basis_offset = cfg.get("basis_offset", 0)
    stack_offsets = cfg.get("stack_basis_offsets", None)
    forecast_basis_dim = cfg.get("forecast_basis_dim", None)

    extra_row = {
        "basis_dim": basis_dim,
        "basis_offset": basis_offset,
        "forecast_basis_dim": forecast_basis_dim if forecast_basis_dim is not None else "",
        "search_round": round_num,
        "arch_pattern": cfg.get("arch_pattern", ""),
        "wavelet_family": cfg.get("wavelet_family", ""),
        "n_stacks_requested": cfg.get("n_stacks_requested", len(cfg["stack_types"])),
        "meta_predicted_best": "",
        "meta_convergence_score": "",
    }

    run_single_experiment(
        experiment_name=f"search_r{round_num}",
        config_name=config_name,
        category=cfg["category"],
        stack_types=cfg["stack_types"],
        period=period,
        run_idx=run_idx,
        dataset=dataset,
        train_series_list=train_series_list,
        csv_path=csv_path,
        n_blocks_per_stack=cfg["n_blocks_per_stack"],
        share_weights=cfg["share_weights"],
        active_g=active_g,
        sum_losses=False,
        activation="ReLU",
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        accelerator_override=accelerator_override,
        forecast_multiplier=forecast_multiplier,
        num_workers=num_workers,
        wandb_enabled=False,
        save_predictions=False,
        gpu_id=gpu_id,
        basis_dim=basis_dim,
        basis_offset=basis_offset,
        stack_basis_offsets=stack_offsets,
        forecast_basis_dim=forecast_basis_dim,
        extra_row=extra_row,
        csv_columns=SEARCH_CSV_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Ranking & Promotion
# ---------------------------------------------------------------------------

def _load_search_results(csv_path, round_num):
    """Load results for a specific search round from the CSV."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("search_round", "") == str(round_num):
                rows.append(row)
    return rows


def rank_and_promote(csv_path, round_num, keep_fraction, meta_forecaster=None,
                     top_k_override=None):
    """Rank configs from a search round and select top configs for promotion.

    Parameters
    ----------
    csv_path : str
        Path to wavelet_search_results.csv.
    round_num : int
        Which round's results to rank.
    keep_fraction : float
        Fraction of configs to keep (e.g. 0.33 for top 33%).
    meta_forecaster : MetaForecaster or None
        If provided and round_num == 1, use meta-predictions as the ranking metric.
    top_k_override : int or None
        Override the number of configs to keep (instead of keep_fraction).

    Returns
    -------
    list[str]
        Config names promoted to the next round.
    """
    rows = _load_search_results(csv_path, round_num)
    if not rows:
        print(f"  [RANK] No results found for round {round_num}")
        return []

    # Group by config_name, compute median best_val_loss
    from collections import defaultdict
    config_metrics = defaultdict(list)
    config_curves = defaultdict(list)

    for row in rows:
        name = row["config_name"]
        bvl = row.get("best_val_loss", "nan")
        try:
            bvl_f = float(bvl)
        except (ValueError, TypeError):
            bvl_f = float("nan")

        if math.isfinite(bvl_f):
            config_metrics[name].append(bvl_f)

        # Collect val_loss curves for meta-forecaster
        if meta_forecaster is not None and round_num == 1:
            raw_curve = row.get("val_loss_curve", "")
            try:
                parsed = json.loads(raw_curve)
                curve = [float(v) for v in parsed]
                if len(curve) >= MetaForecaster.BACKCAST_LENGTH:
                    config_curves[name].append(curve)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    if not config_metrics:
        print(f"  [RANK] No valid metrics for round {round_num}")
        return []

    # Compute ranking metric for each config
    rankings = []
    for name, vals in config_metrics.items():
        median_bvl = float(np.median(vals))
        divergence_rate = sum(1 for v in vals if v > 1000) / len(vals)

        meta_best = float("nan")
        meta_score = float("nan")

        if meta_forecaster is not None and round_num == 1 and name in config_curves:
            curves = config_curves[name]
            preds = meta_forecaster.predict_batch(curves)
            valid_preds = [p["predicted_best"] for p in preds
                          if math.isfinite(p["predicted_best"])]
            if valid_preds:
                meta_best = float(np.median(valid_preds))
                meta_score = float(np.median(
                    [p["convergence_score"] for p in preds
                     if math.isfinite(p["convergence_score"])]
                ))

        # Primary ranking metric
        if math.isfinite(meta_best) and round_num == 1:
            rank_metric = meta_best
        else:
            rank_metric = median_bvl

        rankings.append({
            "config_name": name,
            "median_best_val_loss": median_bvl,
            "n_runs": len(vals),
            "divergence_rate": divergence_rate,
            "meta_predicted_best": meta_best,
            "meta_convergence_score": meta_score,
            "rank_metric": rank_metric,
        })

    # Sort by rank_metric (lower is better), divergent configs go to bottom
    rankings.sort(key=lambda r: (
        r["divergence_rate"] > 0.5,  # penalize > 50% divergence
        r["rank_metric"] if math.isfinite(r["rank_metric"]) else 1e9,
    ))

    total_configs = len(rankings)
    if top_k_override is not None:
        keep_n = min(top_k_override, total_configs)
    else:
        keep_n = max(1, int(total_configs * keep_fraction))

    promoted = [r["config_name"] for r in rankings[:keep_n]]

    # Print summary
    print(f"\n  {'='*65}")
    print(f"  Round {round_num} Ranking — {total_configs} configs, "
          f"promoting top {keep_n}")
    print(f"  {'='*65}")
    print(f"  {'Rank':<5} {'Config':<45} {'ValLoss':>8} "
          f"{'Meta':>8} {'Div%':>5}")
    print(f"  {'-'*65}")

    for i, r in enumerate(rankings[:min(30, total_configs)]):
        marker = " *" if r["config_name"] in promoted else "  "
        meta_str = (f"{r['meta_predicted_best']:.2f}"
                    if math.isfinite(r["meta_predicted_best"]) else "   --")
        div_str = f"{r['divergence_rate']*100:.0f}%"
        print(f"  {i+1:<5}{marker} {r['config_name']:<43} "
              f"{r['median_best_val_loss']:>8.2f} {meta_str:>8} {div_str:>5}")

    if total_configs > 30:
        print(f"  ... ({total_configs - 30} more configs)")

    # Marginal analysis
    _print_marginal_analysis(rankings)

    return promoted


def _print_marginal_analysis(rankings):
    """Print factor analysis: mean metric by each parameter dimension."""
    from collections import defaultdict

    # Parse config names to extract parameters
    param_metrics = {
        "wavelet": defaultdict(list),
        "arch": defaultdict(list),
        "stacks": defaultdict(list),
        "dim": defaultdict(list),
        "offset": defaultdict(list),
    }

    for r in rankings:
        name = r["config_name"]
        metric = r["rank_metric"]
        if not math.isfinite(metric):
            continue

        parts = name.split("_")
        # Parse: {wavelet}_{arch}_s{stacks}_d{dim}_o{offset}
        if len(parts) >= 5:
            wavelet = parts[0]
            arch = parts[1]
            stacks = parts[2].replace("s", "")
            dim = parts[3].replace("d", "")
            offset = parts[4].replace("o", "")

            param_metrics["wavelet"][wavelet].append(metric)
            param_metrics["arch"][arch].append(metric)
            param_metrics["stacks"][stacks].append(metric)
            param_metrics["dim"][dim].append(metric)
            param_metrics["offset"][offset].append(metric)

    print(f"\n  --- Marginal Analysis (mean rank_metric, lower is better) ---")
    for param_name, groups in param_metrics.items():
        if not groups:
            continue
        print(f"\n  {param_name}:")
        sorted_groups = sorted(groups.items(),
                               key=lambda kv: np.mean(kv[1]))
        for key, vals in sorted_groups:
            print(f"    {key:<15} mean={np.mean(vals):>8.2f}  "
                  f"median={np.median(vals):>8.2f}  n={len(vals)}")


# ---------------------------------------------------------------------------
# Search Sequential Runner
# ---------------------------------------------------------------------------

def _run_search_sequential(
    dataset_name, periods, configs, n_runs, round_num, max_epochs, patience,
    forecast_multiplier, batch_size_override, csv_path, accelerator,
    num_workers,
):
    for period in periods:
        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()
        batch_size = _get_batch_size(dataset_name, period, batch_size_override)

        for config_name, cfg in configs.items():
            for run_idx in range(n_runs):
                run_search_experiment(
                    config_name=config_name,
                    cfg=cfg,
                    period=period,
                    run_idx=run_idx,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    round_num=round_num,
                    active_g=False,
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    accelerator_override=accelerator,
                    forecast_multiplier=forecast_multiplier,
                    num_workers=num_workers,
                )


# ---------------------------------------------------------------------------
# Search GPU Worker
# ---------------------------------------------------------------------------

def _search_gpu_worker(gpu_id, job_queue, shutdown_event, worker_args):
    """Worker process for search mode GPU parallel execution."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    torch.set_float32_matmul_precision("medium")

    prefix = f"[GPU {gpu_id}]"
    print(f"{prefix} Search worker started (CUDA_VISIBLE_DEVICES={gpu_id}).")

    dataset_cache = {}
    series_cache = {}

    while not shutdown_event.is_set():
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break

        dataset_name = worker_args["dataset_name"]
        period = job["period"]

        if period not in dataset_cache:
            dataset_cache[period] = load_dataset(dataset_name, period)
            series_cache[period] = dataset_cache[period].get_training_series()
        dataset = dataset_cache[period]
        train_series_list = series_cache[period]

        run_search_experiment(
            config_name=job["config_name"],
            cfg=job["cfg"],
            period=period,
            run_idx=job["run_idx"],
            dataset=dataset,
            train_series_list=train_series_list,
            csv_path=worker_args["csv_path"],
            round_num=job["round_num"],
            active_g=job["active_g"],
            max_epochs=worker_args["max_epochs"],
            patience=worker_args["patience"],
            batch_size=job["batch_size"],
            accelerator_override="cuda",
            forecast_multiplier=worker_args["forecast_multiplier"],
            num_workers=worker_args["num_workers"],
            gpu_id=gpu_id,
        )

    print(f"{prefix} Search worker finished.")


def _run_search_parallel(
    dataset_name, periods, configs, n_runs, round_num, max_epochs, patience,
    forecast_multiplier, batch_size_override, csv_path, n_gpus, num_workers,
):
    jobs = _build_search_job_list(
        configs, periods, n_runs, dataset_name, batch_size_override, round_num,
    )

    pending_jobs = [
        job for job in jobs
        if not result_exists(csv_path, job["pass_name"], job["config_name"],
                             job["period"], job["run_idx"])
    ]

    n_complete = len(jobs) - len(pending_jobs)
    print(f"  Jobs: {len(jobs)} total, {len(pending_jobs)} pending, "
          f"{n_complete} already complete")

    if not pending_jobs:
        print("  All jobs already complete!")
        return

    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    for job in pending_jobs:
        job_queue.put(job)

    shutdown_event = ctx.Event()

    worker_args = {
        "dataset_name": dataset_name,
        "csv_path": csv_path,
        "max_epochs": max_epochs,
        "patience": patience,
        "forecast_multiplier": forecast_multiplier,
        "num_workers": num_workers,
    }

    workers = []
    for gid in range(n_gpus):
        p = ctx.Process(
            target=_search_gpu_worker,
            args=(gid, job_queue, shutdown_event, worker_args),
        )
        p.start()
        workers.append(p)

    print(f"  Spawned {n_gpus} GPU worker processes.")
    for p in workers:
        p.join()

    for p in workers:
        if p.is_alive():
            print(f"  [WARN] Terminating worker PID {p.pid}")
            p.terminate()
            p.join(timeout=10)


# ---------------------------------------------------------------------------
# Search Orchestrator
# ---------------------------------------------------------------------------

def _get_forecast_length(dataset_name, period):
    """Get forecast_length for a dataset/period combo."""
    ds = load_dataset(dataset_name, period)
    return ds.forecast_length


def _run_search_round(dataset_name, periods, round_num, configs, args, n_gpus,
                      forecast_multiplier, csv_path):
    """Run a single search round."""
    schedule = ROUND_SCHEDULE[round_num]
    max_epochs = args.search_max_epochs or schedule["max_epochs"]
    n_runs = args.search_n_runs or schedule["n_runs"]
    patience = min(max_epochs, EARLY_STOPPING_PATIENCE)

    n_configs = len(configs)

    print(f"\n  {'─'*60}")
    print(f"  ROUND {round_num}: {n_configs} configs × {n_runs} runs × "
          f"{max_epochs} epochs")
    print(f"  {'─'*60}")

    if n_configs == 0:
        print("  No configs to run!")
        return

    if n_gpus >= 2:
        _run_search_parallel(
            dataset_name=dataset_name,
            periods=periods,
            configs=configs,
            n_runs=n_runs,
            round_num=round_num,
            max_epochs=max_epochs,
            patience=patience,
            forecast_multiplier=forecast_multiplier,
            batch_size_override=args.batch_size,
            csv_path=csv_path,
            n_gpus=n_gpus,
            num_workers=args.num_workers,
        )
    else:
        _run_search_sequential(
            dataset_name=dataset_name,
            periods=periods,
            configs=configs,
            n_runs=n_runs,
            round_num=round_num,
            max_epochs=max_epochs,
            patience=patience,
            forecast_multiplier=forecast_multiplier,
            batch_size_override=args.batch_size,
            csv_path=csv_path,
            accelerator=args.accelerator,
            num_workers=args.num_workers,
        )


def run_search(dataset_name, args, n_gpus):
    """Main search orchestrator: runs successive halving with meta-forecaster."""

    if dataset_name == "m4":
        all_periods = {"Yearly": DATASET_PERIODS["m4"]["Yearly"]}
        forecast_multiplier = FORECAST_MULTIPLIERS["m4"]
    elif dataset_name == "traffic":
        all_periods = {"Traffic-96": TRAFFIC_PERIODS["Traffic-96"]}
        forecast_multiplier = FORECAST_MULTIPLIERS["traffic"]
    elif dataset_name == "weather":
        all_periods = {"Weather-96": WEATHER_PERIODS["Weather-96"]}
        forecast_multiplier = FORECAST_MULTIPLIERS["weather"]
    else:
        print(f"[ERROR] dataset '{dataset_name}' not supported.")
        return

    periods_filter = args.periods
    if periods_filter:
        periods = [p for p in periods_filter if p in all_periods]
        if not periods:
            print(f"[ERROR] No valid periods for '{dataset_name}'.")
            return
    else:
        periods = list(all_periods.keys())

    csv_path = _search_csv_path(dataset_name)
    init_csv(csv_path, columns=SEARCH_CSV_COLUMNS)

    # Get forecast_length for validity filtering
    forecast_length = _get_forecast_length(dataset_name, periods[0])

    # Determine which rounds to run
    round_spec = args.round if hasattr(args, "round") and args.round else "all"
    if round_spec == "all":
        rounds_to_run = [1, 2, 3, 4]
    else:
        rounds_to_run = [int(round_spec)]

    # Step 0: Train / load meta-forecaster
    print(f"\n{'='*70}")
    print(f"Wavelet Search — {dataset_name.upper()}")
    print(f"  Periods: {periods}")
    print(f"  Rounds:  {rounds_to_run}")
    print(f"{'='*70}")

    existing_csvs = [p for p in _META_TRAINING_CSVS if os.path.exists(p)]
    meta_forecaster = None
    if existing_csvs:
        print(f"\n  Step 0: Training meta-forecaster on {len(existing_csvs)} existing CSVs...")
        meta_forecaster = MetaForecaster(META_CACHE_DIR)
        try:
            meta_forecaster.train(existing_csvs)
        except ValueError as e:
            print(f"  [WARN] Meta-forecaster training failed: {e}")
            print(f"  [WARN] Falling back to val_loss ranking only.")
            meta_forecaster = None
    else:
        print(f"\n  [INFO] No existing result CSVs found for meta-forecaster training.")
        print(f"  [INFO] Using val_loss ranking only.")

    # Run each round
    promoted = None
    for round_num in rounds_to_run:
        # Generate configs for this round
        if round_num == 1:
            configs = generate_search_configs(forecast_length, round_num)
        else:
            # Load promotions from prior round
            if promoted is None:
                prior_round = round_num - 1
                prior_schedule = ROUND_SCHEDULE[prior_round]
                promoted = rank_and_promote(
                    csv_path, prior_round,
                    prior_schedule["keep_fraction"],
                    meta_forecaster=meta_forecaster if prior_round == 1 else None,
                    top_k_override=args.top_k,
                )
                if not promoted:
                    print(f"  [ERROR] No configs promoted from round {prior_round}. "
                          f"Run round {prior_round} first.")
                    return

            if round_num == 4:
                # Final round also runs active_g="forecast" pass
                configs = generate_search_configs(
                    forecast_length, round_num, promoted_configs=set(promoted)
                )
            else:
                configs = generate_search_configs(
                    forecast_length, round_num, promoted_configs=set(promoted)
                )

        _run_search_round(
            dataset_name, periods, round_num, configs, args, n_gpus,
            forecast_multiplier, csv_path,
        )

        # After running, rank and promote for the next round
        schedule = ROUND_SCHEDULE[round_num]
        promoted = rank_and_promote(
            csv_path, round_num,
            schedule["keep_fraction"],
            meta_forecaster=meta_forecaster if round_num == 1 else None,
            top_k_override=args.top_k,
        )

        # Update meta-forecaster predictions in CSV for round 1
        if meta_forecaster is not None and round_num == 1:
            _update_meta_predictions(csv_path, round_num, meta_forecaster)

    # Final round 4: also run active_g="forecast" for promoted configs
    if 4 in rounds_to_run and promoted:
        print(f"\n  Running active_g='forecast' pass for top {len(promoted)} configs...")
        configs_ag = generate_search_configs(
            forecast_length, round_num=4, promoted_configs=set(promoted)
        )
        # Re-run with active_g=forecast — modify the configs' category
        for name, cfg in configs_ag.items():
            cfg["category"] = "search_round4_ag"

        ag_schedule = ROUND_SCHEDULE[4]
        ag_max_epochs = args.search_max_epochs or ag_schedule["max_epochs"]
        ag_n_runs = args.search_n_runs or ag_schedule["n_runs"]
        ag_patience = min(ag_max_epochs, EARLY_STOPPING_PATIENCE)

        for period in periods:
            dataset = load_dataset(dataset_name, period)
            train_series_list = dataset.get_training_series()
            batch_size = _get_batch_size(dataset_name, period, args.batch_size)

            for config_name, cfg in configs_ag.items():
                for run_idx in range(ag_n_runs):
                    # Check if already done
                    if result_exists(csv_path, "search_r4_ag", config_name, period, run_idx):
                        continue

                    basis_dim = cfg.get("basis_dim", WAVELET_BASIS_DIM)
                    basis_offset = cfg.get("basis_offset", 0)
                    extra_row = {
                        "basis_dim": basis_dim,
                        "basis_offset": basis_offset,
                        "search_round": "4_ag",
                        "arch_pattern": cfg.get("arch_pattern", ""),
                        "wavelet_family": cfg.get("wavelet_family", ""),
                        "n_stacks_requested": cfg.get("n_stacks_requested", ""),
                        "meta_predicted_best": "",
                        "meta_convergence_score": "",
                    }
                    run_single_experiment(
                        experiment_name="search_r4_ag",
                        config_name=config_name,
                        category="search_round4_ag",
                        stack_types=cfg["stack_types"],
                        period=period,
                        run_idx=run_idx,
                        dataset=dataset,
                        train_series_list=train_series_list,
                        csv_path=csv_path,
                        n_blocks_per_stack=cfg["n_blocks_per_stack"],
                        share_weights=cfg["share_weights"],
                        active_g="forecast",
                        sum_losses=False,
                        activation="ReLU",
                        max_epochs=ag_max_epochs,
                        patience=ag_patience,
                        batch_size=batch_size,
                        accelerator_override=args.accelerator,
                        forecast_multiplier=forecast_multiplier,
                        num_workers=args.num_workers,
                        wandb_enabled=False,
                        save_predictions=False,
                        basis_dim=basis_dim,
                        basis_offset=basis_offset,
                        stack_basis_offsets=cfg.get("stack_basis_offsets"),
                        extra_row=extra_row,
                        csv_columns=SEARCH_CSV_COLUMNS,
                    )

    print(f"\n{'='*70}")
    print(f"Wavelet Search COMPLETE — {dataset_name.upper()}")
    print(f"Results: {csv_path}")
    print(f"{'='*70}")


def _update_meta_predictions(csv_path, round_num, meta_forecaster):
    """Update meta prediction columns in the CSV for a given round."""
    if not os.path.exists(csv_path):
        return

    rows_all = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get("search_round", "") == str(round_num):
                raw_curve = row.get("val_loss_curve", "")
                try:
                    parsed = json.loads(raw_curve)
                    curve = [float(v) for v in parsed]
                    if len(curve) >= MetaForecaster.BACKCAST_LENGTH:
                        pred = meta_forecaster.predict(curve)
                        row["meta_predicted_best"] = (
                            f"{pred['predicted_best']:.6f}"
                            if math.isfinite(pred["predicted_best"]) else ""
                        )
                        row["meta_convergence_score"] = (
                            f"{pred['convergence_score']:.6f}"
                            if math.isfinite(pred["convergence_score"]) else ""
                        )
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            rows_all.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_all)


def analyze_search(dataset_name, round_num):
    """Analyze existing search results and print a summary."""
    csv_path = _search_csv_path(dataset_name)
    if not os.path.exists(csv_path):
        print(f"[ERROR] No search results at {csv_path}")
        return

    rows = _load_search_results(csv_path, round_num)
    if not rows:
        print(f"[ERROR] No results for round {round_num} in {csv_path}")
        return

    print(f"\n{'='*70}")
    print(f"Search Analysis — {dataset_name.upper()} — Round {round_num}")
    print(f"  Total rows: {len(rows)}")
    print(f"{'='*70}")

    # Use rank_and_promote just for analysis (keep_fraction=1.0 to show all)
    rank_and_promote(csv_path, round_num, keep_fraction=1.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Wavelet Frequency-Band Experiment for N-BEATS Lightning"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["m4", "traffic", "weather", "all"],
        help=(
            "Dataset to run (use 'all' to run M4-Yearly, Traffic-96, Weather-96 "
            "sequentially)"
        ),
    )
    parser.add_argument(
        "--periods", nargs="+", default=None,
        help=(
            "Filter to specific periods. "
            "m4: Yearly. traffic: Traffic-96. weather: Weather-96."
        ),
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help=f"Runs per config (default: {WAVELET_STUDY_N_RUNS})",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=MAX_EPOCHS,
        help=f"Maximum training epochs (default: {MAX_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override default batch size",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=None,
        help="Number of GPUs for parallel execution (default: auto-detect)",
    )
    parser.add_argument(
        "--accelerator", default="auto", choices=["auto", "cuda", "mps", "cpu"],
        help="Accelerator (default: auto)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0)",
    )

    # Search mode arguments
    parser.add_argument(
        "--mode", choices=["study", "search"], default="study",
        help=(
            "'study' runs the 16 hardcoded configs (default). "
            "'search' runs hyperparameter search with meta-forecaster."
        ),
    )
    parser.add_argument(
        "--round", default="all",
        help=(
            "Search round to run: 1 (coarse, 6 epochs), 2 (medium, 15 epochs), "
            "3 (fine, 30 epochs), 4 (full, 100 epochs), or 'all'. "
            "Only used with --mode search."
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override number of configs to promote between rounds.",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze existing search results (no training). Use with --round.",
    )
    parser.add_argument(
        "--search-max-epochs", type=int, default=None,
        help="Override max epochs for search round.",
    )
    parser.add_argument(
        "--search-n-runs", type=int, default=None,
        help="Override runs per config for search round.",
    )

    args = parser.parse_args()

    # Resolve GPU count (reuse unified benchmark helper via a minimal namespace)
    class _FakeArgs:
        accelerator = args.accelerator
        n_gpus = args.n_gpus
    n_gpus = resolve_n_gpus(_FakeArgs())

    if args.mode == "search":
        if args.analyze:
            # Analysis-only mode
            round_num = int(args.round) if args.round != "all" else 1
            if args.dataset == "all":
                for ds in ["m4", "traffic", "weather"]:
                    analyze_search(ds, round_num)
            else:
                analyze_search(args.dataset, round_num)
        else:
            # Run search
            if args.dataset == "all":
                print(f"\n{'#'*70}")
                print(f"Wavelet Search — ALL DATASETS")
                print(f"{'#'*70}")
                for ds in ["m4", "traffic", "weather"]:
                    run_search(ds, args, n_gpus)
                print(f"\n{'#'*70}")
                print(f"Wavelet Search COMPLETE — ALL DATASETS")
                print(f"{'#'*70}")
            else:
                run_search(args.dataset, args, n_gpus)
    else:
        # Original study mode
        if args.dataset == "all":
            study_datasets = ["m4", "traffic", "weather"]
            print(f"\n{'#'*70}")
            print(f"Wavelet Study — ALL DATASETS (M4-Yearly, Traffic-96, Weather-96)")
            print(f"{'#'*70}")
            for ds in study_datasets:
                _run_wavelet_dataset(ds, args.periods, args, n_gpus)
            print(f"\n{'#'*70}")
            print(f"Wavelet Study COMPLETE — ALL DATASETS")
            print(f"{'#'*70}")
        else:
            _run_wavelet_dataset(args.dataset, args.periods, args, n_gpus)


if __name__ == "__main__":
    main()
