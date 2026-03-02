"""
TrendAE + AE Architecture Search Experiment

Identifies best-performing AE (AutoEncoder, GenericAE, BottleneckGenericAE)
+ TrendAE configurations for parameter-efficient architectures using successive
halving with a meta-forecaster.

Architecture: ["TrendAE", AE_variant] * 5  (10 stacks total, alternating)

Hyperparameter search space:
  - AE variants: AutoEncoder, GenericAE, BottleneckGenericAE
  - latent_dim: [2, 8, 16]  (AERootBlock backbone bottleneck)
  - thetas_dim: [5, 10]     (BottleneckGenericAE projection bottleneck)
  - active_g: [False, "forecast"]  (GenericAE, BottleneckGenericAE only)
  - Fixed: n_blocks_per_stack=1, share_weights=True, activation=ReLU

Successive halving (3 rounds, keep top 33%):
  Round 1: 10 epochs, 3 runs/config → meta-forecaster ranking → top 33%
  Round 2: 15 epochs, 3 runs/config → early stopping + divergence → top 33%
  Round 3: 30 epochs, 3 runs/config → final validation → top 2

Datasets: M4-Yearly (primary), Tourism-Yearly, Weather-96.

Success criteria: OWA < 0.85 on M4-Yearly with <5M parameters.

Usage:
    # Full pipeline (all 3 rounds, M4-Yearly)
    python experiments/run_trendae_study.py --dataset m4

    # Single round
    python experiments/run_trendae_study.py --dataset m4 --round 1

    # Validate top configs on Tourism
    python experiments/run_trendae_study.py --dataset tourism --round 3

    # Run on Weather-96
    python experiments/run_trendae_study.py --dataset weather --round 1

    # Analyze results
    python experiments/run_trendae_study.py --dataset m4 --round 1 --analyze

    # Smoke test
    python experiments/run_trendae_study.py --dataset m4 --round 1 --search-max-epochs 2
"""

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
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
    BASE_SEED,
    FORECAST_MULTIPLIERS,
    DATASET_PERIODS,
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    TOURISM_PERIODS,
    M4_PERIODS,
    _shutdown_requested,
)
from meta_forecaster import MetaForecaster

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# TrendAE+AE Study Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(_EXPERIMENTS_DIR, "results")

# Extended CSV schema for TrendAE+AE search
TRENDAE_CSV_COLUMNS = CSV_COLUMNS + [
    "search_round", "arch_pattern", "ae_variant",
    "latent_dim_cfg", "thetas_dim_cfg", "trend_thetas_dim_cfg",
    "meta_predicted_best", "meta_convergence_score",
]

# AE variants to search
AE_VARIANTS = [
    "AutoEncoder", "GenericAE", "BottleneckGenericAE",
    "GenericAEBackcast", "AutoEncoderAE", "GenericAEBackcastAE",
]

# RootBlock-based variants (no latent_dim param; search latent_dim maps to thetas_dim)
_ROOTBLOCK_AE_VARIANTS = {"AutoEncoder", "GenericAEBackcast"}

# Variants that use thetas_dim as an internal bottleneck (search over SEARCH_THETAS_DIMS)
_BOTTLENECK_AE_VARIANTS = {"BottleneckGenericAE", "GenericAEBackcastAE", "AutoEncoderAE"}

# Generic-type variants where active_g can help convergence
_ACTIVE_G_AE_VARIANTS = {"GenericAE", "BottleneckGenericAE", "GenericAEBackcast", "GenericAEBackcastAE"}

# Hyperparameter search space
SEARCH_LATENT_DIMS = [2, 8, 16]
SEARCH_THETAS_DIMS = [5, 10]  # For bottleneck variants
SEARCH_ACTIVE_G = [False, "forecast"]  # For Generic-type variants

# Fixed architecture
N_STACKS = 10  # ["TrendAE", AE_variant] * 5
SEARCH_TREND_THETAS_DIMS = [3, 5]  # Trend polynomial degree: 3=cubic, 5=degree-4

# Successive halving schedule (3 rounds, 3 runs each, keep top 33% / top 2 final)
ROUND_SCHEDULE = {
    1: {"max_epochs": 10, "n_runs": 3, "keep_fraction": 0.33},
    2: {"max_epochs": 15, "n_runs": 3, "keep_fraction": 0.33},
    3: {"max_epochs": 30, "n_runs": 3, "keep_fraction": 0.33, "top_k": 2},
}

# Known existing CSVs for meta-forecaster training
_META_TRAINING_CSVS = [
    os.path.join(RESULTS_DIR, "m4", "unified_benchmark_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "block_benchmark_results.csv"),
    os.path.join(RESULTS_DIR, "traffic", "block_benchmark_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "convergence_study_results_v1.csv"),
    os.path.join(RESULTS_DIR, "weather", "convergence_study_results_v1.csv"),
    os.path.join(RESULTS_DIR, "traffic", "convergence_study_results_v1.csv"),
    os.path.join(RESULTS_DIR, "convergence_study_v2_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "wavelet_study_3_successive_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "ae_trend_search_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "wavelet_trendae_comparison_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "wavelet_study_2_basis_dim_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "generic_ae_pure_stack_results.csv"),
]

META_CACHE_DIR = os.path.join(RESULTS_DIR, ".meta_cache")

# Datasets supported by this study
TRENDAE_DATASETS = {
    "m4":      {"periods": ["Yearly"]},
    "tourism": {"periods": ["Tourism-Yearly"]},
    "weather": {"periods": ["Weather-96"]},
}


# ---------------------------------------------------------------------------
# Config Generation
# ---------------------------------------------------------------------------

def _build_stack_types(ae_variant):
    """Build alternating TrendAE + AE stack pattern (10 stacks total)."""
    return ["TrendAE", ae_variant] * 5


def generate_trendae_configs(round_num=1, promoted_configs=None):
    """Generate TrendAE+AE search configs programmatically.

    Parameters
    ----------
    round_num : int
        Search round (1-3). Round 1 generates the full grid; rounds 2-3
        filter to promoted_configs only.
    promoted_configs : set[str] or None
        Config names promoted from the prior round (rounds 2-3 only).

    Returns
    -------
    dict[str, dict]
        Config name -> config dict with keys: category, stack_types,
        n_blocks_per_stack, share_weights, ae_variant, latent_dim,
        thetas_dim, active_g.
    """
    configs = {}

    for ae_variant in AE_VARIANTS:
        for latent_dim in SEARCH_LATENT_DIMS:
            # Determine thetas_dim search space
            if ae_variant in _BOTTLENECK_AE_VARIANTS:
                thetas_dims = SEARCH_THETAS_DIMS
            else:
                # GenericAE ignores thetas_dim (direct projection)
                # RootBlock AE variants: search latent_dim maps to thetas_dim
                thetas_dims = [THETAS_DIM]

            # Determine active_g search space
            if ae_variant in _ACTIVE_G_AE_VARIANTS:
                active_g_values = SEARCH_ACTIVE_G
            else:
                # AutoEncoder / AutoEncoderAE: only test active_g=False
                active_g_values = [False]

            for thetas_dim in thetas_dims:
                for trend_td in SEARCH_TREND_THETAS_DIMS:
                    for active_g in active_g_values:
                        # Build config name
                        ag_str = "agF" if active_g == "forecast" else "ag0"
                        config_name = (
                            f"{ae_variant}_ld{latent_dim}_td{thetas_dim}_ttd{trend_td}_{ag_str}"
                        )

                        # For rounds 2+, only include promoted configs
                        if promoted_configs is not None and config_name not in promoted_configs:
                            continue

                        stack_types = _build_stack_types(ae_variant)

                        # RootBlock-based variants (AutoEncoder, GenericAEBackcast)
                        # have no latent_dim param; the search latent_dim maps to
                        # thetas_dim since that's their internal bottleneck.
                        if ae_variant in _ROOTBLOCK_AE_VARIANTS:
                            effective_thetas_dim = latent_dim  # bottleneck = thetas_dim
                            effective_latent_dim = LATENT_DIM  # unused by RootBlock
                        else:
                            effective_thetas_dim = thetas_dim
                            effective_latent_dim = latent_dim

                        configs[config_name] = {
                            "category": f"trendae_search_round{round_num}",
                            "stack_types": stack_types,
                            "n_blocks_per_stack": 1,
                            "share_weights": True,
                            "ae_variant": ae_variant,
                            "latent_dim": effective_latent_dim,
                            "thetas_dim": effective_thetas_dim,
                            "trend_thetas_dim": trend_td,
                            "active_g": active_g,
                            "arch_pattern": "trendae_ae_alternating",
                            "latent_dim_cfg": latent_dim,    # original search value
                            "thetas_dim_cfg": thetas_dim,    # original search value
                            "trend_thetas_dim_cfg": trend_td,
                        }

    return configs


# ---------------------------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------------------------

def _search_csv_path(dataset_name):
    """Return path to the TrendAE+AE search results CSV."""
    return os.path.join(RESULTS_DIR, dataset_name, "trendae_study_results.csv")


def _get_batch_size(dataset_name, period, override=None):
    """Get batch size for a dataset/period combo."""
    return get_batch_size(dataset_name, period, override)


def _get_forecast_length(dataset_name, period):
    """Get forecast length for a dataset/period."""
    ds = load_dataset(dataset_name, period)
    return ds.forecast_length



# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_trendae_experiment(
    config_name, cfg, period, run_idx, dataset, train_series_list,
    csv_path, round_num, max_epochs, patience, batch_size,
    accelerator_override, forecast_multiplier, num_workers=0, gpu_id=None,
):
    """Wrapper around run_single_experiment for TrendAE+AE configs."""
    active_g = cfg.get("active_g", False)
    thetas_dim = cfg.get("thetas_dim", THETAS_DIM)
    latent_dim = cfg.get("latent_dim", LATENT_DIM)
    trend_td = cfg.get("trend_thetas_dim", 3)

    # Cosine annealing LR scheduler: hold constant for 15 epochs, then decay
    warmup_epochs = 15
    if max_epochs > warmup_epochs:
        lr_scheduler_config = {
            "warmup_epochs": warmup_epochs,
            "T_max": max_epochs - warmup_epochs,
            "eta_min": 1e-6,
        }
    else:
        lr_scheduler_config = None  # too few epochs for annealing

    extra_row = {
        "search_round": round_num,
        "arch_pattern": cfg.get("arch_pattern", "trendae_ae_alternating"),
        "ae_variant": cfg.get("ae_variant", ""),
        "latent_dim_cfg": cfg.get("latent_dim_cfg", ""),
        "thetas_dim_cfg": cfg.get("thetas_dim_cfg", ""),
        "trend_thetas_dim_cfg": cfg.get("trend_thetas_dim_cfg", ""),
        "meta_predicted_best": "",
        "meta_convergence_score": "",
    }

    run_single_experiment(
        experiment_name=f"trendae_search_r{round_num}",
        config_name=config_name,
        category=cfg.get("category", f"trendae_search_round{round_num}"),
        stack_types=cfg["stack_types"],
        period=period,
        run_idx=run_idx,
        dataset=dataset,
        train_series_list=train_series_list,
        csv_path=csv_path,
        n_blocks_per_stack=cfg.get("n_blocks_per_stack", 1),
        share_weights=cfg.get("share_weights", True),
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
        extra_row=extra_row,
        csv_columns=TRENDAE_CSV_COLUMNS,
        thetas_dim_override=thetas_dim,
        latent_dim_override=latent_dim,
        trend_thetas_dim=trend_td,
        lr_scheduler_config=lr_scheduler_config,
    )


# ---------------------------------------------------------------------------
# Ranking & Promotion (mirrors wavelet study pattern)
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

    Returns list of promoted config_name strings.
    """
    rows = _load_search_results(csv_path, round_num)
    if not rows:
        return []

    # Group by config_name
    from collections import defaultdict
    config_results = defaultdict(list)
    for row in rows:
        config_results[row["config_name"]].append(row)

    rankings = []
    for name, result_rows in config_results.items():
        val_losses = []
        n_diverged = 0
        meta_best = float("inf")
        meta_score = float("inf")

        for r in result_rows:
            bvl = r.get("best_val_loss", "")
            try:
                bvl_val = float(bvl)
                if math.isfinite(bvl_val):
                    val_losses.append(bvl_val)
            except (ValueError, TypeError):
                pass

            if r.get("diverged", "").lower() in ("true", "1"):
                n_diverged += 1

            # Meta predictions (if available)
            mp = r.get("meta_predicted_best", "")
            try:
                mp_val = float(mp)
                if math.isfinite(mp_val):
                    meta_best = min(meta_best, mp_val)
            except (ValueError, TypeError):
                pass

            ms = r.get("meta_convergence_score", "")
            try:
                ms_val = float(ms)
                if math.isfinite(ms_val):
                    meta_score = min(meta_score, ms_val)
            except (ValueError, TypeError):
                pass

        if not val_losses:
            median_bvl = float("inf")
        else:
            median_bvl = float(np.median(val_losses))

        divergence_rate = n_diverged / len(result_rows) if result_rows else 0.0

        # Primary ranking metric
        if math.isfinite(meta_best) and round_num == 1:
            rank_metric = meta_best
        else:
            rank_metric = median_bvl

        rankings.append({
            "config_name": name,
            "median_best_val_loss": median_bvl,
            "n_runs": len(val_losses),
            "divergence_rate": divergence_rate,
            "meta_predicted_best": meta_best,
            "meta_convergence_score": meta_score,
            "rank_metric": rank_metric,
        })

    # Sort by rank_metric (lower is better), penalize divergent configs
    rankings.sort(key=lambda r: (
        r["divergence_rate"] > 0.5,
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

    return promoted


# ---------------------------------------------------------------------------
# Meta-Forecaster Prediction Update
# ---------------------------------------------------------------------------

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

    # Rewrite CSV with updated predictions
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_all)


# ---------------------------------------------------------------------------
# Round Runner (sequential)
# ---------------------------------------------------------------------------

def _run_search_round_sequential(dataset_name, periods, round_num, configs,
                                 args, forecast_multiplier, csv_path):
    """Run a single search round sequentially on one device."""
    schedule = ROUND_SCHEDULE[round_num]
    max_epochs = args.search_max_epochs or schedule["max_epochs"]
    n_runs = schedule["n_runs"]

    # Early stopping only in rounds 2+
    if round_num >= 2:
        patience = min(max_epochs, EARLY_STOPPING_PATIENCE)
    else:
        patience = max_epochs  # No early stopping in round 1

    n_configs = len(configs)

    print(f"\n  {'─'*60}")
    print(f"  ROUND {round_num}: {n_configs} configs × {n_runs} runs × "
          f"{max_epochs} epochs  (sequential)")
    print(f"  {'─'*60}")

    if n_configs == 0:
        print("  No configs to run!")
        return

    for period in periods:
        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()
        batch_size = _get_batch_size(dataset_name, period, args.batch_size)

        for config_name, cfg in configs.items():
            if _shutdown_requested:
                print("[SHUTDOWN] Exiting search round.")
                return

            for run_idx in range(n_runs):
                if result_exists(csv_path, f"trendae_search_r{round_num}",
                                 config_name, period, run_idx):
                    continue

                run_trendae_experiment(
                    config_name=config_name,
                    cfg=cfg,
                    period=period,
                    run_idx=run_idx,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    round_num=round_num,
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    accelerator_override=args.accelerator,
                    forecast_multiplier=forecast_multiplier,
                    num_workers=args.num_workers,
                )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-GPU Parallel Round Runner
# ---------------------------------------------------------------------------

def _build_search_jobs(periods, round_num, configs, dataset_name,
                       batch_size_override):
    """Build a flat list of job dicts for all (period, config, run_idx) combos."""
    schedule = ROUND_SCHEDULE[round_num]
    n_runs = schedule["n_runs"]
    jobs = []
    for period in periods:
        batch_size = _get_batch_size(dataset_name, period, batch_size_override)
        for config_name, cfg in configs.items():
            for run_idx in range(n_runs):
                jobs.append({
                    "period": period,
                    "config_name": config_name,
                    "cfg": cfg,
                    "run_idx": run_idx,
                    "batch_size": batch_size,
                })
    return jobs


def _gpu_worker(gpu_id, job_queue, shutdown_event, worker_args):
    """Worker process: pins to GPU gpu_id, pulls jobs from shared queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    torch.set_float32_matmul_precision("medium")

    prefix = f"[GPU {gpu_id}]"
    print(f"{prefix} TrendAE study worker started "
          f"(CUDA_VISIBLE_DEVICES={gpu_id}).")

    dataset_name = worker_args["dataset_name"]
    round_num = worker_args["round_num"]

    # Cache datasets per period to avoid redundant loading
    dataset_cache = {}
    series_cache = {}

    while not shutdown_event.is_set():
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break

        period = job["period"]
        if period not in dataset_cache:
            dataset_cache[period] = load_dataset(dataset_name, period)
            series_cache[period] = dataset_cache[period].get_training_series()

        run_trendae_experiment(
            config_name=job["config_name"],
            cfg=job["cfg"],
            period=period,
            run_idx=job["run_idx"],
            dataset=dataset_cache[period],
            train_series_list=series_cache[period],
            csv_path=worker_args["csv_path"],
            round_num=round_num,
            max_epochs=worker_args["max_epochs"],
            patience=worker_args["patience"],
            batch_size=job["batch_size"],
            accelerator_override="cuda",
            forecast_multiplier=worker_args["forecast_multiplier"],
            num_workers=worker_args["num_workers"],
            gpu_id=gpu_id,
        )

    print(f"{prefix} TrendAE study worker finished.")


def _run_search_round_parallel(dataset_name, periods, round_num, configs,
                                args, forecast_multiplier, csv_path, n_gpus):
    """Run a single search round in parallel across multiple GPUs."""
    schedule = ROUND_SCHEDULE[round_num]
    max_epochs = args.search_max_epochs or schedule["max_epochs"]

    # Early stopping only in rounds 2+
    if round_num >= 2:
        patience = min(max_epochs, EARLY_STOPPING_PATIENCE)
    else:
        patience = max_epochs

    n_configs = len(configs)

    print(f"\n  {'─'*60}")
    print(f"  ROUND {round_num}: {n_configs} configs × {schedule['n_runs']} runs × "
          f"{max_epochs} epochs  ({n_gpus} GPUs)")
    print(f"  {'─'*60}")

    if n_configs == 0:
        print("  No configs to run!")
        return

    # Build flat job list and filter completed
    jobs = _build_search_jobs(periods, round_num, configs, dataset_name,
                              args.batch_size)
    pending_jobs = [
        job for job in jobs
        if not result_exists(csv_path, f"trendae_search_r{round_num}",
                             job["config_name"], job["period"], job["run_idx"])
    ]

    n_complete = len(jobs) - len(pending_jobs)
    print(f"  Jobs: {len(jobs)} total, {len(pending_jobs)} pending, "
          f"{n_complete} already complete")

    if not pending_jobs:
        print("  All jobs already complete!")
        return

    # Set up multiprocessing with spawn context (clean CUDA state per worker)
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    for job in pending_jobs:
        job_queue.put(job)

    shutdown_event = ctx.Event()

    worker_args = {
        "dataset_name": dataset_name,
        "round_num": round_num,
        "csv_path": csv_path,
        "max_epochs": max_epochs,
        "patience": patience,
        "forecast_multiplier": forecast_multiplier,
        "num_workers": args.num_workers,
    }

    # Spawn one worker per GPU
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

    # Clean up any workers still alive
    for p in workers:
        if p.is_alive():
            print(f"  [WARN] Terminating worker PID {p.pid}")
            p.terminate()
            p.join(timeout=10)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_search_round(dataset_name, periods, round_num, configs, args,
                       forecast_multiplier, csv_path, n_gpus=0):
    """Dispatch a search round to sequential or parallel execution."""
    if n_gpus >= 2:
        _run_search_round_parallel(
            dataset_name, periods, round_num, configs, args,
            forecast_multiplier, csv_path, n_gpus,
        )
    else:
        _run_search_round_sequential(
            dataset_name, periods, round_num, configs, args,
            forecast_multiplier, csv_path,
        )


# ---------------------------------------------------------------------------
# Main Search Pipeline
# ---------------------------------------------------------------------------

def run_trendae_search(args):
    """Run the full TrendAE+AE successive halving search."""
    dataset_name = args.dataset

    if dataset_name not in TRENDAE_DATASETS:
        print(f"[ERROR] Dataset '{dataset_name}' not supported. "
              f"Choose from: {list(TRENDAE_DATASETS.keys())}")
        return

    periods = TRENDAE_DATASETS[dataset_name]["periods"]
    forecast_multiplier = FORECAST_MULTIPLIERS[dataset_name]

    csv_path = _search_csv_path(dataset_name)
    init_csv(csv_path, columns=TRENDAE_CSV_COLUMNS)

    # Resolve GPU count for parallel execution
    n_gpus = resolve_n_gpus(args)

    # Determine which rounds to run
    round_spec = args.round if hasattr(args, "round") and args.round else "all"
    if round_spec == "all":
        rounds_to_run = [1, 2, 3]
    else:
        rounds_to_run = [int(round_spec)]

    # Print header
    print(f"\n{'='*70}")
    print(f"TrendAE+AE Architecture Search — {dataset_name.upper()}")
    print(f"  Periods: {periods}")
    print(f"  Rounds:  {rounds_to_run}")
    print(f"  Config space: {len(generate_trendae_configs())} total configs")
    if n_gpus >= 2:
        print(f"  GPUs: {n_gpus} (parallel execution)")
    else:
        print(f"  Mode: sequential")
    print(f"{'='*70}")

    # Step 0: Train / load meta-forecaster
    existing_csvs = [p for p in _META_TRAINING_CSVS if os.path.exists(p)]
    meta_forecaster = None
    if existing_csvs:
        print(f"\n  Step 0: Training meta-forecaster on {len(existing_csvs)} existing CSVs...")
        meta_forecaster = MetaForecaster(META_CACHE_DIR)
        try:
            meta_forecaster.train(existing_csvs, force_retrain=True)
        except ValueError as e:
            print(f"  [WARN] Meta-forecaster training failed: {e}")
            print(f"  [WARN] Falling back to val_loss ranking only.")
            meta_forecaster = None
    else:
        print(f"\n  [INFO] No existing CSVs found for meta-forecaster training.")
        print(f"  [INFO] Will use val_loss ranking only.")

    # Run each round
    promoted = None
    for round_num in rounds_to_run:
        # Analyze-only mode
        if args.analyze:
            schedule = ROUND_SCHEDULE[round_num]
            effective_top_k = args.top_k or schedule.get("top_k")
            promoted = rank_and_promote(
                csv_path, round_num,
                schedule["keep_fraction"],
                meta_forecaster=meta_forecaster if round_num == 1 else None,
                top_k_override=effective_top_k,
            )
            continue

        # Generate configs for this round
        if round_num == 1:
            configs = generate_trendae_configs(round_num)
        else:
            # Load promotions from prior round
            if promoted is None:
                prior_round = round_num - 1
                prior_schedule = ROUND_SCHEDULE[prior_round]
                effective_top_k = args.top_k or prior_schedule.get("top_k")
                promoted = rank_and_promote(
                    csv_path, prior_round,
                    prior_schedule["keep_fraction"],
                    meta_forecaster=meta_forecaster if prior_round == 1 else None,
                    top_k_override=effective_top_k,
                )
                if not promoted:
                    print(f"  [ERROR] No configs promoted from round {prior_round}. "
                          f"Run round {prior_round} first.")
                    return

            configs = generate_trendae_configs(
                round_num, promoted_configs=set(promoted)
            )

        _run_search_round(
            dataset_name, periods, round_num, configs, args,
            forecast_multiplier, csv_path, n_gpus=n_gpus,
        )

        # After running, rank and promote for the next round
        schedule = ROUND_SCHEDULE[round_num]
        effective_top_k = args.top_k or schedule.get("top_k")
        promoted = rank_and_promote(
            csv_path, round_num,
            schedule["keep_fraction"],
            meta_forecaster=meta_forecaster if round_num == 1 else None,
            top_k_override=effective_top_k,
        )

        # Update meta-forecaster predictions in CSV for round 1
        if meta_forecaster is not None and round_num == 1:
            _update_meta_predictions(csv_path, round_num, meta_forecaster)

    print(f"\n{'='*70}")
    print(f"TrendAE+AE Search COMPLETE — {dataset_name.upper()}")
    print(f"Results: {csv_path}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TrendAE+AE Architecture Search — successive halving with meta-forecaster"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["m4", "tourism", "weather"],
        help="Dataset to search (m4=Yearly, tourism=Tourism-Yearly, weather=Weather-96)"
    )
    parser.add_argument(
        "--round", default=None,
        help="Run a specific round (1, 2, or 3). Default: all rounds."
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze existing results without running new experiments."
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override number of configs to promote (instead of keep_fraction)."
    )
    parser.add_argument(
        "--search-max-epochs", type=int, default=None,
        help="Override max epochs for the current round."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size (dataset-tuned values take precedence)."
    )
    parser.add_argument(
        "--accelerator", default="auto",
        help="Accelerator override: auto, cuda, mps, cpu."
    )
    parser.add_argument(
        "--n-gpus", type=int, default=None,
        help="Number of GPUs for parallel execution (default: auto-detect)."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader num_workers."
    )

    args = parser.parse_args()
    run_trendae_search(args)


if __name__ == "__main__":
    main()
