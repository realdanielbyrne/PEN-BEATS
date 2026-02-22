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
import gc
import os
import queue
import signal
import sys
import time

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
WAVELET_CSV_COLUMNS = CSV_COLUMNS + ["basis_dim", "basis_offset"]

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

    extra_row = {
        "basis_dim": basis_dim,
        "basis_offset": basis_offset,
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

    args = parser.parse_args()

    # Resolve GPU count (reuse unified benchmark helper via a minimal namespace)
    class _FakeArgs:
        accelerator = args.accelerator
        n_gpus = args.n_gpus
    n_gpus = resolve_n_gpus(_FakeArgs())

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
