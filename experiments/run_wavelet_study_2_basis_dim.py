"""
Wavelet Study 2 — Basis Dimension Sweep

Factorial study testing how wavelet basis_dim interacts with Trend polynomial
degree (thetas_dim) in a Trend+Wavelet alternating architecture.

Architecture: [Trend, <WaveletV3>] * 5  (10 stacks total)
  - n_blocks_per_stack=1, share_weights=True, active_g=False, activation=ReLU

Hyperparameter grid (24 configs):
  - Wavelet family:  HaarWaveletV3, DB3WaveletV3, Coif2WaveletV3  (3)
  - basis_dim:       forecast_length, max(forecast_length//2, forecast_length-2),
                     backcast_length, backcast_length//2                        (4)
  - basis_offset:    0 (fixed)
  - thetas_dim:      3, 5                                                      (2)

Dataset: M4-Yearly only (forecast_length=6, backcast_length=30)
Runs: 3 seeds per config (72 total)

Results → experiments/results/m4/wavelet_study_2_basis_dim_results.csv

Usage:
    python experiments/run_wavelet_study_2_basis_dim.py
    python experiments/run_wavelet_study_2_basis_dim.py --n-runs 1 --max-epochs 2   # smoke test
    python experiments/run_wavelet_study_2_basis_dim.py --batch-size 512
"""

import argparse
import os
import queue
import signal
import sys

import torch
import multiprocessing as mp

# Allow running from project root or experiments/
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXPERIMENTS_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", "src"))

from run_unified_benchmark import (
    run_single_experiment,
    result_exists,
    get_batch_size,
    init_csv,
    load_dataset,
    resolve_n_gpus,
    CSV_COLUMNS,
    MAX_EPOCHS,
    THETAS_DIM,
    BASE_SEED,
    FORECAST_MULTIPLIERS,
    DATASET_PERIODS,
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
)

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Study Constants
# ---------------------------------------------------------------------------

STUDY_N_RUNS = 3
STUDY_MAX_EPOCHS = 100
STUDY_PATIENCE = EARLY_STOPPING_PATIENCE  # early stopping patience
STUDY_BATCH_SIZE = 1024

RESULTS_DIR = os.path.join(_EXPERIMENTS_DIR, "results")

# Extended CSV schema
STUDY_CSV_COLUMNS = CSV_COLUMNS + [
    "basis_dim", "basis_offset", "thetas_dim", "backcast_length",
]

# ---------------------------------------------------------------------------
# Wavelet families under test
# ---------------------------------------------------------------------------

WAVELET_FAMILIES = ["HaarWaveletV3", "DB3WaveletV3", "Coif2WaveletV3"]

# ---------------------------------------------------------------------------
# Config Generator
# ---------------------------------------------------------------------------

def _build_configs():
    """Build the factorial config dictionary.

    For M4-Yearly: forecast_length=6, backcast_length=5*6=30.
    basis_dim values:
      - forecast_length          = 6
      - max(forecast_length//2, forecast_length-2) = max(3, 4) = 4
      - backcast_length          = 30
      - backcast_length // 2     = 15
    """
    forecast_length = DATASET_PERIODS["m4"]["Yearly"]["horizon"]  # 6
    forecast_multiplier = FORECAST_MULTIPLIERS["m4"]              # 5
    backcast_length = forecast_length * forecast_multiplier       # 30

    basis_dim_less_than_forecast = max(forecast_length // 2, forecast_length - 2)

    basis_dims = [
        ("eq_fcast", forecast_length),
        ("lt_fcast", basis_dim_less_than_forecast),
        ("eq_bcast", backcast_length),
        ("lt_bcast", backcast_length // 2),
    ]

    thetas_dims = [3, 5]
    basis_offset = 0

    configs = {}
    for wavelet in WAVELET_FAMILIES:
        wavelet_short = wavelet.replace("WaveletV3", "")
        for bd_label, bd_value in basis_dims:
            for td in thetas_dims:
                config_name = f"{wavelet_short}_bd{bd_value}_{bd_label}_td{td}"
                stack_types = ["Trend", wavelet] * 5  # 10 stacks

                configs[config_name] = {
                    "category": "basis_dim_sweep",
                    "stack_types": stack_types,
                    "n_blocks_per_stack": 1,
                    "share_weights": True,
                    "basis_dim": bd_value,
                    "basis_offset": basis_offset,
                    "thetas_dim": td,
                    "backcast_length": backcast_length,
                    "wavelet_family": wavelet,
                    "bd_label": bd_label,
                }

    return configs


STUDY_CONFIGS = _build_configs()

# ---------------------------------------------------------------------------
# CSV path
# ---------------------------------------------------------------------------

def _csv_path():
    return os.path.join(RESULTS_DIR, "m4", "wavelet_study_2_basis_dim_results.csv")


# ---------------------------------------------------------------------------
# Experiment Wrapper
# ---------------------------------------------------------------------------

def run_basis_dim_experiment(
    config_name, cfg, period, run_idx, dataset, train_series_list,
    csv_path, max_epochs, patience, batch_size, accelerator_override,
    forecast_multiplier, num_workers=0, gpu_id=None, wandb_enabled=False,
    wandb_project="nbeats-lightning",
):
    """Wrapper around run_single_experiment for basis_dim study configs."""
    basis_dim = cfg["basis_dim"]
    basis_offset = cfg["basis_offset"]
    thetas_dim = cfg["thetas_dim"]
    backcast_length = cfg["backcast_length"]

    extra_row = {
        "basis_dim": basis_dim,
        "basis_offset": basis_offset,
        "thetas_dim": thetas_dim,
        "backcast_length": backcast_length,
    }

    run_single_experiment(
        experiment_name="baseline",
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
        active_g=False,
        sum_losses=False,
        activation="ReLU",
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        accelerator_override=accelerator_override,
        forecast_multiplier=forecast_multiplier,
        num_workers=num_workers,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        save_predictions=True,
        gpu_id=gpu_id,
        basis_dim=basis_dim,
        basis_offset=basis_offset,
        extra_row=extra_row,
        csv_columns=STUDY_CSV_COLUMNS,
        trend_thetas_dim=thetas_dim,
    )


# ---------------------------------------------------------------------------
# Job List Builder
# ---------------------------------------------------------------------------

def _build_job_list(n_runs, batch_size_override):
    """Build flat list of job dicts for all (config, run_idx)."""
    period = "Yearly"
    batch_size = batch_size_override if batch_size_override else STUDY_BATCH_SIZE

    jobs = []
    for config_name, cfg in STUDY_CONFIGS.items():
        for run_idx in range(n_runs):
            jobs.append({
                "period": period,
                "config_name": config_name,
                "cfg": cfg,
                "run_idx": run_idx,
                "batch_size": batch_size,
            })
    return jobs


# ---------------------------------------------------------------------------
# GPU Worker (parallel execution)
# ---------------------------------------------------------------------------

def _gpu_worker(gpu_id, job_queue, shutdown_event, worker_args):
    """Worker process: pins to GPU gpu_id, pulls jobs from shared queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    torch.set_float32_matmul_precision("medium")

    prefix = f"[GPU {gpu_id}]"
    print(f"{prefix} Basis-dim study worker started (CUDA_VISIBLE_DEVICES={gpu_id}).")

    dataset = load_dataset("m4", "Yearly")
    train_series_list = dataset.get_training_series()

    while not shutdown_event.is_set():
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break

        run_basis_dim_experiment(
            config_name=job["config_name"],
            cfg=job["cfg"],
            period=job["period"],
            run_idx=job["run_idx"],
            dataset=dataset,
            train_series_list=train_series_list,
            csv_path=worker_args["csv_path"],
            max_epochs=worker_args["max_epochs"],
            patience=worker_args["patience"],
            batch_size=job["batch_size"],
            accelerator_override="cuda",
            forecast_multiplier=worker_args["forecast_multiplier"],
            num_workers=worker_args["num_workers"],
            gpu_id=gpu_id,
            wandb_enabled=worker_args["wandb_enabled"],
            wandb_project=worker_args["wandb_project"],
        )

    print(f"{prefix} Basis-dim study worker finished.")


# ---------------------------------------------------------------------------
# Sequential Runner
# ---------------------------------------------------------------------------

def _run_sequential(
    n_runs, max_epochs, patience, forecast_multiplier,
    batch_size_override, csv_path, accelerator, num_workers,
    wandb_enabled, wandb_project,
):
    period = "Yearly"

    print(f"\n{'='*60}")
    print(f"  Wavelet Study 2 (Basis Dim) — M4 / {period}")
    print(f"{'='*60}")

    dataset = load_dataset("m4", period)
    train_series_list = dataset.get_training_series()
    batch_size = batch_size_override if batch_size_override else STUDY_BATCH_SIZE

    for config_name, cfg in STUDY_CONFIGS.items():
        for run_idx in range(n_runs):
            run_basis_dim_experiment(
                config_name=config_name,
                cfg=cfg,
                period=period,
                run_idx=run_idx,
                dataset=dataset,
                train_series_list=train_series_list,
                csv_path=csv_path,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                accelerator_override=accelerator,
                forecast_multiplier=forecast_multiplier,
                num_workers=num_workers,
                wandb_enabled=wandb_enabled,
                wandb_project=wandb_project,
            )


# ---------------------------------------------------------------------------
# Parallel Runner
# ---------------------------------------------------------------------------

def _run_parallel(
    n_runs, max_epochs, patience, forecast_multiplier,
    batch_size_override, csv_path, n_gpus, num_workers,
    wandb_enabled, wandb_project,
):
    jobs = _build_job_list(n_runs, batch_size_override)

    pending_jobs = [
        job for job in jobs
        if not result_exists(csv_path, "baseline", job["config_name"],
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
        "csv_path": csv_path,
        "max_epochs": max_epochs,
        "patience": patience,
        "forecast_multiplier": forecast_multiplier,
        "num_workers": num_workers,
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
    }

    workers = []
    for gid in range(n_gpus):
        p = ctx.Process(
            target=_gpu_worker,
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wavelet Study 2 — Basis Dimension Sweep (M4-Yearly)",
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help=f"Runs per config (default: {STUDY_N_RUNS})",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=STUDY_MAX_EPOCHS,
        help=f"Maximum training epochs (default: {STUDY_MAX_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Override batch size (default: {STUDY_BATCH_SIZE})",
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
    parser.add_argument(
        "--wandb", action="store_true", default=True,
        help="Enable W&B logging (default: True)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--wandb-project", default="nbeats-lightning",
        help="W&B project name (default: nbeats-lightning)",
    )

    args = parser.parse_args()

    wandb_enabled = args.wandb and not args.no_wandb

    # Resolve GPU count
    class _FakeArgs:
        accelerator = args.accelerator
        n_gpus = args.n_gpus
    n_gpus = resolve_n_gpus(_FakeArgs())

    n_runs = args.n_runs if args.n_runs is not None else STUDY_N_RUNS
    max_epochs = args.max_epochs
    patience = STUDY_PATIENCE
    forecast_multiplier = FORECAST_MULTIPLIERS["m4"]

    csv_path = _csv_path()
    init_csv(csv_path, columns=STUDY_CSV_COLUMNS)

    total_configs = len(STUDY_CONFIGS)

    print(f"\n{'='*70}")
    print(f"Wavelet Study 2 — Basis Dimension Sweep")
    print(f"  Dataset:  M4-Yearly (forecast_length=6, backcast_length=30)")
    print(f"  Configs:  {total_configs}  |  Runs/config: {n_runs}")
    print(f"  Max epochs: {max_epochs}  |  Patience: {patience}")
    print(f"  Batch size: {args.batch_size or STUDY_BATCH_SIZE}")
    print(f"  W&B: {wandb_enabled}")
    print(f"  Total runs: {total_configs * n_runs}")
    print(f"  Results: {csv_path}")
    if n_gpus >= 2:
        print(f"  GPUs: {n_gpus} (parallel execution)")
    else:
        print(f"  Mode: sequential")
    print(f"{'='*70}")

    if n_gpus >= 2:
        _run_parallel(
            n_runs=n_runs,
            max_epochs=max_epochs,
            patience=patience,
            forecast_multiplier=forecast_multiplier,
            batch_size_override=args.batch_size,
            csv_path=csv_path,
            n_gpus=n_gpus,
            num_workers=args.num_workers,
            wandb_enabled=wandb_enabled,
            wandb_project=args.wandb_project,
        )
    else:
        _run_sequential(
            n_runs=n_runs,
            max_epochs=max_epochs,
            patience=patience,
            forecast_multiplier=forecast_multiplier,
            batch_size_override=args.batch_size,
            csv_path=csv_path,
            accelerator=args.accelerator,
            num_workers=args.num_workers,
            wandb_enabled=wandb_enabled,
            wandb_project=args.wandb_project,
        )

    print(f"\n{'='*70}")
    print(f"Wavelet Study 2 COMPLETE — Basis Dimension Sweep")
    print(f"Results: {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

