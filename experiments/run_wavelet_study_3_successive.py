"""
Wavelet Study 3 — Successive Halving Search

Applies successive halving to find the best Trend+WaveletV3 configurations
across all 14 wavelet families, 4 basis_dim levels, and 2 trend_thetas_dim
values (3 vs 5), with a two-pass design (baseline + active_g='forecast').

Config space per pass (112 configs):
  - Wavelet family:      All 14 WaveletV3 variants                         (14)
  - basis_dim:           forecast_length, max(fl//2, fl-2),
                         backcast_length, backcast_length//2                 (4)
  - trend_thetas_dim:    3, 5                                                (2)

Two-pass design (runs both passes through successive halving independently):
  Pass 1 ("baseline"):      active_g=False,      sum_losses=False
  Pass 2 ("activeG_fcast"):  active_g="forecast", sum_losses=False

Architecture:
  M4 / Tourism:             [Trend, <WaveletV3>] * 5   (10 stacks total)
  Traffic-96 / Weather-96:  [Trend, <WaveletV3>] * 10  (20 stacks total)

Successive halving (3 rounds, 3 runs each, keep top 33%):
  Round 1: 7 epochs,  3 runs/config → meta-forecaster ranking → top 33%
  Round 2: 15 epochs, 3 runs/config → median val_loss + early stopping → top 33%
  Round 3: 50 epochs, 3 runs/config → final validation → top 2

Datasets: M4-Yearly (primary), Weather-96, Traffic-96 (cross-dataset validation)

Results → experiments/results/<dataset>/wavelet_study_3_successive_results.csv

Usage:
    # Full pipeline (all 3 rounds, M4-Yearly)
    python experiments/run_wavelet_study_3_successive.py --dataset m4

    # Weather-96 cross-dataset validation
    python experiments/run_wavelet_study_3_successive.py --dataset weather

    # Traffic-96 cross-dataset validation
    python experiments/run_wavelet_study_3_successive.py --dataset traffic

    # Single round
    python experiments/run_wavelet_study_3_successive.py --dataset m4 --round 1

    # Single pass only (skip active_g pass)
    python experiments/run_wavelet_study_3_successive.py --dataset m4 --pass baseline

    # Analyze results
    python experiments/run_wavelet_study_3_successive.py --dataset m4 --round 1 --analyze

    # Smoke test
    python experiments/run_wavelet_study_3_successive.py --dataset m4 --round 1 --search-max-epochs 2
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
# Wavelet Study 3 Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(_EXPERIMENTS_DIR, "results")

# All 14 WaveletV3 families
SEARCH_WAVELETS = [
    "HaarWaveletV3",
    "DB2WaveletV3",
    "DB3WaveletV3",
    "DB4WaveletV3",
    "DB10WaveletV3",
    "DB20WaveletV3",
    "Coif1WaveletV3",
    "Coif2WaveletV3",
    "Coif3WaveletV3",
    "Coif10WaveletV3",
    "Symlet2WaveletV3",
    "Symlet3WaveletV3",
    "Symlet10WaveletV3",
    "Symlet20WaveletV3",
]

SEARCH_TREND_THETAS_DIMS = [3, 5]

# Extended CSV schema
WAVELET_STUDY3_CSV_COLUMNS = CSV_COLUMNS + [
    "search_round", "basis_dim", "basis_offset", "trend_thetas_dim_cfg",
    "wavelet_family", "bd_label", "active_g_cfg",
    "meta_predicted_best", "meta_convergence_score",
]

# Two-pass design: baseline and active_g='forecast'
SEARCH_PASSES = [
    ("baseline", False),
    ("activeG_fcast", "forecast"),
]

# Successive halving schedule (3 rounds, 3 runs each, keep top 33% / top 2 final)
ROUND_SCHEDULE = {
    1: {"max_epochs": 7,  "n_runs": 3, "keep_fraction": 0.33},
    2: {"max_epochs": 15, "n_runs": 3, "keep_fraction": 0.33},
    3: {"max_epochs": 50, "n_runs": 3, "keep_fraction": 0.33, "top_k": 2},
}

# Known existing CSVs for meta-forecaster training
_META_TRAINING_CSVS = [
    os.path.join(RESULTS_DIR, "m4", "unified_benchmark_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "block_benchmark_results.csv"),
    os.path.join(RESULTS_DIR, "traffic", "block_benchmark_results.csv"),
    os.path.join(RESULTS_DIR, "m4", "convergence_study_results_v1.csv"),
    os.path.join(RESULTS_DIR, "weather", "convergence_study_results_v1.csv"),
    os.path.join(RESULTS_DIR, "traffic", "convergence_study_results_v1.csv"),
]

META_CACHE_DIR = os.path.join(RESULTS_DIR, ".meta_cache")

# Datasets supported by this study
WAVELET_STUDY3_DATASETS = {
    "m4":      {"periods": ["Yearly"]},
    "tourism": {"periods": ["Tourism-Yearly"]},
    "weather": {"periods": ["Weather-96"]},
    "traffic": {"periods": ["Traffic-96"]},
}

# Datasets that use 20 stacks (10 Trend+Wavelet pairs) instead of the default 10
LARGE_STACK_DATASETS = {"traffic", "weather"}


# ---------------------------------------------------------------------------
# Config Generation
# ---------------------------------------------------------------------------

def _wavelet_short_name(wavelet):
    """Convert e.g. 'DB3WaveletV3' → 'DB3'."""
    return wavelet.replace("WaveletV3", "")


def _compute_basis_dims(dataset_name, period):
    """Compute basis_dim levels from dataset dimensions, deduplicated by value.

    Returns list of (label, value) tuples with unique values only (first label
    wins when two levels collide, e.g. Tourism-Yearly where eq_fcast and
    lt_bcast both resolve to 4).
    """
    horizon = DATASET_PERIODS[dataset_name][period]["horizon"]
    forecast_length = horizon
    forecast_multiplier = FORECAST_MULTIPLIERS[dataset_name]
    backcast_length = forecast_length * forecast_multiplier

    basis_dim_less_than_forecast = max(forecast_length // 2, forecast_length - 2)

    candidates = [
        ("eq_fcast", forecast_length),
        ("lt_fcast", basis_dim_less_than_forecast),
        ("eq_bcast", backcast_length),
        ("lt_bcast", backcast_length // 2),
    ]

    seen_values = set()
    unique = []
    for label, value in candidates:
        if value not in seen_values:
            seen_values.add(value)
            unique.append((label, value))

    return unique


def generate_wavelet_study3_configs(dataset_name, period, round_num=1,
                                     promoted_configs=None):
    """Generate Trend+WaveletV3 search configs.

    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g. "m4", "tourism").
    period : str
        Period name (e.g. "Yearly", "Tourism-Yearly").
    round_num : int
        Search round (1-3). Round 1 generates the full grid; rounds 2-3
        filter to promoted_configs only.
    promoted_configs : set[str] or None
        Config names promoted from the prior round (rounds 2-3 only).

    Returns
    -------
    dict[str, dict]
        Config name -> config dict.
    """
    basis_dims = _compute_basis_dims(dataset_name, period)
    configs = {}

    for wavelet in SEARCH_WAVELETS:
        wavelet_short = _wavelet_short_name(wavelet)
        for bd_label, bd_value in basis_dims:
            for trend_td in SEARCH_TREND_THETAS_DIMS:
                config_name = (
                    f"{wavelet_short}_bd{bd_value}_{bd_label}_ttd{trend_td}"
                )

                # For rounds 2+, only include promoted configs
                if promoted_configs is not None and config_name not in promoted_configs:
                    continue

                n_stack_pairs = 10 if dataset_name in LARGE_STACK_DATASETS else 5
                stack_types = ["Trend", wavelet] * n_stack_pairs  # 20 or 10 stacks

                configs[config_name] = {
                    "category": f"wavelet_search_round{round_num}",
                    "stack_types": stack_types,
                    "n_blocks_per_stack": 1,
                    "share_weights": True,
                    "basis_dim": bd_value,
                    "basis_offset": 0,
                    "trend_thetas_dim": trend_td,
                    "wavelet_family": wavelet,
                    "bd_label": bd_label,
                }

    return configs


# ---------------------------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------------------------

def _search_csv_path(dataset_name):
    """Return path to the wavelet study 3 search results CSV."""
    return os.path.join(
        RESULTS_DIR, dataset_name, "wavelet_study_3_successive_results.csv"
    )


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_wavelet_study3_experiment(
    config_name, cfg, period, run_idx, dataset, train_series_list,
    csv_path, round_num, max_epochs, patience, batch_size,
    accelerator_override, forecast_multiplier, num_workers=0, gpu_id=None,
    active_g=False, pass_name="baseline",
):
    """Wrapper around run_single_experiment for wavelet study 3 configs."""
    basis_dim = cfg["basis_dim"]
    basis_offset = cfg["basis_offset"]
    trend_td = cfg["trend_thetas_dim"]

    extra_row = {
        "search_round": round_num,
        "basis_dim": basis_dim,
        "basis_offset": basis_offset,
        "trend_thetas_dim_cfg": trend_td,
        "wavelet_family": cfg["wavelet_family"],
        "bd_label": cfg["bd_label"],
        "active_g_cfg": str(active_g),
        "meta_predicted_best": "",
        "meta_convergence_score": "",
    }

    experiment_name = f"wavelet_search_r{round_num}_{pass_name}"

    # Cosine annealing LR scheduler: hold constant for 15 epochs, then decay
    cosine_warmup_epochs = 15
    if max_epochs > cosine_warmup_epochs:
        lr_scheduler_config = {
            "warmup_epochs": cosine_warmup_epochs,
            "T_max": max_epochs - cosine_warmup_epochs,
            "eta_min": 1e-6,
        }
    else:
        lr_scheduler_config = None  # too few epochs for annealing

    run_single_experiment(
        experiment_name=experiment_name,
        config_name=config_name,
        category=cfg.get("category", f"wavelet_search_round{round_num}"),
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
        extra_row=extra_row,
        csv_columns=WAVELET_STUDY3_CSV_COLUMNS,
        trend_thetas_dim=trend_td,
        lr_scheduler_config=lr_scheduler_config,
    )


# ---------------------------------------------------------------------------
# Ranking & Promotion
# ---------------------------------------------------------------------------

def _load_search_results(csv_path, round_num, pass_name=None):
    """Load results for a specific search round (and optionally pass) from the CSV."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("search_round", "") != str(round_num):
                continue
            if pass_name is not None:
                # Filter by experiment name which includes pass
                exp = row.get("experiment", "")
                expected_tag = f"wavelet_search_r{round_num}_{pass_name}"
                if exp != expected_tag:
                    continue
            rows.append(row)
    return rows


def rank_and_promote(csv_path, round_num, keep_fraction, meta_forecaster=None,
                     top_k_override=None, pass_name=None):
    """Rank configs from a search round and select top configs for promotion.

    When pass_name is provided, only results from that pass are considered.
    When pass_name is None, all results from the round are ranked together
    (backward compatible with single-pass runs).

    Returns list of promoted config_name strings.
    """
    rows = _load_search_results(csv_path, round_num, pass_name=pass_name)
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

        # Primary ranking metric: meta-forecaster in round 1, val_loss otherwise
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
    print(f"\n  {'='*75}")
    print(f"  Round {round_num} Ranking — {total_configs} configs, "
          f"promoting top {keep_n}")
    print(f"  {'='*75}")
    print(f"  {'Rank':<5} {'Config':<45} {'ValLoss':>8} "
          f"{'Meta':>8} {'Div%':>5}")
    print(f"  {'-'*75}")

    for i, r in enumerate(rankings[:min(40, total_configs)]):
        marker = " *" if r["config_name"] in promoted else "  "
        meta_str = (f"{r['meta_predicted_best']:.2f}"
                    if math.isfinite(r["meta_predicted_best"]) else "   --")
        div_str = f"{r['divergence_rate']*100:.0f}%"
        print(f"  {i+1:<5}{marker} {r['config_name']:<43} "
              f"{r['median_best_val_loss']:>8.2f} {meta_str:>8} {div_str:>5}")

    if total_configs > 40:
        print(f"  ... ({total_configs - 40} more configs not shown)")

    # Print marginal analysis
    _print_marginal_analysis(rankings)

    return promoted


# ---------------------------------------------------------------------------
# Marginal Factor Analysis
# ---------------------------------------------------------------------------

def _print_marginal_analysis(rankings):
    """Print factor-level analysis by wavelet family, basis_dim, trend_thetas_dim."""
    from collections import defaultdict

    # Parse config names to extract factors
    factor_metrics = {
        "wavelet_family": defaultdict(list),
        "bd_label": defaultdict(list),
        "trend_thetas_dim": defaultdict(list),
    }

    for r in rankings:
        name = r["config_name"]
        metric = r["median_best_val_loss"]
        if not math.isfinite(metric):
            continue

        # Parse: {WaveletShort}_bd{value}_{label}_ttd{value}
        parts = name.split("_")
        if len(parts) < 5:
            continue

        wavelet_short = parts[0]
        # bd_label is parts[2] (or parts[2]_parts[3] for two-word labels like eq_fcast)
        # Find the ttd part
        ttd_idx = None
        for i, p in enumerate(parts):
            if p.startswith("ttd"):
                ttd_idx = i
                break

        if ttd_idx is None:
            continue

        bd_label = "_".join(parts[2:ttd_idx])
        ttd_value = parts[ttd_idx]

        factor_metrics["wavelet_family"][wavelet_short].append(metric)
        factor_metrics["bd_label"][bd_label].append(metric)
        factor_metrics["trend_thetas_dim"][ttd_value].append(metric)

    print(f"\n  {'─'*60}")
    print(f"  Marginal Analysis (median val_loss by factor level)")
    print(f"  {'─'*60}")

    for factor_name, level_metrics in factor_metrics.items():
        if not level_metrics:
            continue
        print(f"\n  {factor_name}:")
        sorted_levels = sorted(
            level_metrics.items(),
            key=lambda x: float(np.median(x[1]))
        )
        for level, metrics in sorted_levels:
            med = float(np.median(metrics))
            print(f"    {level:<20} median={med:.4f}  (n={len(metrics)})")


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
# Round Runner
# ---------------------------------------------------------------------------

def _resolve_passes(args):
    """Determine which passes to run based on CLI --pass argument."""
    pass_filter = getattr(args, "pass_filter", None)
    if pass_filter is None:
        return SEARCH_PASSES
    return [(name, ag) for name, ag in SEARCH_PASSES if name == pass_filter]


def _run_search_round_sequential(dataset_name, periods, round_num, configs, args,
                                 forecast_multiplier, csv_path):
    """Run a single search round sequentially on one device (both passes unless filtered)."""
    schedule = ROUND_SCHEDULE[round_num]
    max_epochs = args.search_max_epochs or schedule["max_epochs"]
    n_runs = schedule["n_runs"]
    passes = _resolve_passes(args)

    # Early stopping only in rounds 2+
    if round_num >= 2:
        patience = min(max_epochs, EARLY_STOPPING_PATIENCE)
    else:
        patience = max_epochs  # No early stopping in round 1

    n_configs = len(configs)
    n_passes = len(passes)

    print(f"\n  {'─'*60}")
    print(f"  ROUND {round_num}: {n_configs} configs × {n_passes} passes × "
          f"{n_runs} runs × {max_epochs} epochs")
    for pass_name, active_g in passes:
        print(f"    Pass: {pass_name} (active_g={active_g})")
    print(f"  {'─'*60}")

    if n_configs == 0:
        print("  No configs to run!")
        return

    for period in periods:
        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()
        batch_size = get_batch_size(dataset_name, period, args.batch_size)

        for pass_name, active_g in passes:
            if _shutdown_requested:
                print("[SHUTDOWN] Exiting search round.")
                return

            print(f"\n    --- Pass: {pass_name} (active_g={active_g}) ---")

            for config_name, cfg in configs.items():
                if _shutdown_requested:
                    print("[SHUTDOWN] Exiting search round.")
                    return

                experiment_tag = f"wavelet_search_r{round_num}_{pass_name}"
                for run_idx in range(n_runs):
                    if result_exists(csv_path, experiment_tag,
                                     config_name, period, run_idx):
                        continue

                    run_wavelet_study3_experiment(
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
                        active_g=active_g,
                        pass_name=pass_name,
                    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-GPU Parallel Round Runner
# ---------------------------------------------------------------------------

def _build_search_jobs(periods, round_num, configs, passes, dataset_name,
                       batch_size_override):
    """Build a flat list of job dicts for all (period, pass, config, run_idx) combos."""
    schedule = ROUND_SCHEDULE[round_num]
    n_runs = schedule["n_runs"]
    jobs = []
    for period in periods:
        batch_size = get_batch_size(dataset_name, period, batch_size_override)
        for pass_name, active_g in passes:
            for config_name, cfg in configs.items():
                for run_idx in range(n_runs):
                    jobs.append({
                        "period": period,
                        "config_name": config_name,
                        "cfg": cfg,
                        "run_idx": run_idx,
                        "batch_size": batch_size,
                        "pass_name": pass_name,
                        "active_g": active_g,
                    })
    return jobs


def _gpu_worker(gpu_id, job_queue, shutdown_event, worker_args):
    """Worker process: pins to GPU gpu_id, pulls jobs from shared queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    torch.set_float32_matmul_precision("medium")

    prefix = f"[GPU {gpu_id}]"
    print(f"{prefix} Wavelet Study 3 worker started "
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

        run_wavelet_study3_experiment(
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
            active_g=job["active_g"],
            pass_name=job["pass_name"],
        )

    print(f"{prefix} Wavelet Study 3 worker finished.")


def _run_search_round_parallel(dataset_name, periods, round_num, configs, args,
                                forecast_multiplier, csv_path, n_gpus):
    """Run a single search round in parallel across multiple GPUs."""
    schedule = ROUND_SCHEDULE[round_num]
    max_epochs = args.search_max_epochs or schedule["max_epochs"]
    passes = _resolve_passes(args)

    # Early stopping only in rounds 2+
    if round_num >= 2:
        patience = min(max_epochs, EARLY_STOPPING_PATIENCE)
    else:
        patience = max_epochs  # No early stopping in round 1

    n_configs = len(configs)
    n_passes = len(passes)

    print(f"\n  {'─'*60}")
    print(f"  ROUND {round_num}: {n_configs} configs × {n_passes} passes × "
          f"{schedule['n_runs']} runs × {max_epochs} epochs  ({n_gpus} GPUs)")
    for pass_name, active_g in passes:
        print(f"    Pass: {pass_name} (active_g={active_g})")
    print(f"  {'─'*60}")

    if n_configs == 0:
        print("  No configs to run!")
        return

    # Build flat job list and filter completed
    all_jobs = _build_search_jobs(periods, round_num, configs, passes,
                                  dataset_name, args.batch_size)
    pending_jobs = [
        job for job in all_jobs
        if not result_exists(
            csv_path,
            f"wavelet_search_r{round_num}_{job['pass_name']}",
            job["config_name"], job["period"], job["run_idx"],
        )
    ]

    n_complete = len(all_jobs) - len(pending_jobs)
    print(f"  Jobs: {len(all_jobs)} total, {len(pending_jobs)} pending, "
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
# Per-Pass Promotion (union across passes)
# ---------------------------------------------------------------------------

def _promote_union(csv_path, round_num, schedule, passes,
                   meta_forecaster=None, top_k_override=None):
    """Rank each pass independently and return union of promoted config names.

    This ensures that a config promoted by *either* pass survives to the next
    round, so both passes always operate on the same config set.
    """
    all_promoted = set()
    for pass_name, _active_g in passes:
        promoted = rank_and_promote(
            csv_path, round_num,
            schedule["keep_fraction"],
            meta_forecaster=meta_forecaster,
            top_k_override=top_k_override,
            pass_name=pass_name,
        )
        if promoted:
            print(f"    Pass '{pass_name}': promoted {len(promoted)} configs")
            all_promoted.update(promoted)
        else:
            print(f"    Pass '{pass_name}': no results found for round {round_num}")

    if all_promoted:
        print(f"    Union: {len(all_promoted)} unique promoted configs")
    return list(all_promoted)


# ---------------------------------------------------------------------------
# Main Search Pipeline
# ---------------------------------------------------------------------------

def run_wavelet_study3_search(args):
    """Run the full wavelet study 3 successive halving search."""
    dataset_name = args.dataset

    if dataset_name not in WAVELET_STUDY3_DATASETS:
        print(f"[ERROR] Dataset '{dataset_name}' not supported. "
              f"Choose from: {list(WAVELET_STUDY3_DATASETS.keys())}")
        return

    periods = WAVELET_STUDY3_DATASETS[dataset_name]["periods"]
    forecast_multiplier = FORECAST_MULTIPLIERS[dataset_name]

    csv_path = _search_csv_path(dataset_name)
    init_csv(csv_path, columns=WAVELET_STUDY3_CSV_COLUMNS)

    # Use first period for config generation (basis_dim depends on dataset dims)
    primary_period = periods[0]

    # Determine which rounds to run
    round_spec = args.round if hasattr(args, "round") and args.round else "all"
    if round_spec == "all":
        rounds_to_run = [1, 2, 3]
    else:
        rounds_to_run = [int(round_spec)]

    # Count total configs
    full_configs = generate_wavelet_study3_configs(
        dataset_name, primary_period, round_num=1
    )

    passes = _resolve_passes(args)

    # Resolve GPU count for parallel execution
    n_gpus = resolve_n_gpus(args)

    # Print header
    print(f"\n{'='*70}")
    print(f"Wavelet Study 3 — Successive Halving Search — {dataset_name.upper()}")
    print(f"  Periods: {periods}")
    print(f"  Rounds:  {rounds_to_run}")
    print(f"  Passes:  {[p[0] for p in passes]}")
    print(f"  Config space: {len(full_configs)} total configs")
    print(f"  Wavelets: {len(SEARCH_WAVELETS)} families")
    print(f"  Trend thetas dims: {SEARCH_TREND_THETAS_DIMS}")
    n_pairs = 10 if dataset_name in LARGE_STACK_DATASETS else 5
    print(f"  Stack pairs: {n_pairs} ([Trend, WaveletV3] × {n_pairs} = {n_pairs * 2} stacks)")
    if n_gpus >= 2:
        print(f"  GPUs: {n_gpus} (parallel execution)")
    else:
        print(f"  Mode: sequential (GPUs available: {max(n_gpus, 0)})")
    print(f"{'='*70}")

    # Step 0: Train / load meta-forecaster
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
        print(f"\n  [INFO] No existing CSVs found for meta-forecaster training.")
        print(f"  [INFO] Will use val_loss ranking only.")

    # Run each round
    promoted = None
    for round_num in rounds_to_run:
        # Analyze-only mode
        if args.analyze:
            schedule = ROUND_SCHEDULE[round_num]
            effective_top_k = args.top_k or schedule.get("top_k")
            # Rank per-pass and take union for promotion
            promoted = _promote_union(
                csv_path, round_num, schedule, passes,
                meta_forecaster=meta_forecaster if round_num == 1 else None,
                top_k_override=effective_top_k,
            )
            continue

        # Generate configs for this round
        if round_num == 1:
            configs = generate_wavelet_study3_configs(
                dataset_name, primary_period, round_num
            )
        else:
            # Load promotions from prior round
            if promoted is None:
                prior_round = round_num - 1
                prior_schedule = ROUND_SCHEDULE[prior_round]
                effective_top_k = args.top_k or prior_schedule.get("top_k")
                promoted = _promote_union(
                    csv_path, prior_round, prior_schedule, passes,
                    meta_forecaster=meta_forecaster if prior_round == 1 else None,
                    top_k_override=effective_top_k,
                )
                if not promoted:
                    print(f"  [ERROR] No configs promoted from round {prior_round}. "
                          f"Run round {prior_round} first.")
                    return

            configs = generate_wavelet_study3_configs(
                dataset_name, primary_period, round_num,
                promoted_configs=set(promoted),
            )

        _run_search_round(
            dataset_name, periods, round_num, configs, args,
            forecast_multiplier, csv_path, n_gpus=n_gpus,
        )

        # After running, rank and promote for the next round
        schedule = ROUND_SCHEDULE[round_num]
        effective_top_k = args.top_k or schedule.get("top_k")
        promoted = _promote_union(
            csv_path, round_num, schedule, passes,
            meta_forecaster=meta_forecaster if round_num == 1 else None,
            top_k_override=effective_top_k,
        )

        # Update meta-forecaster predictions in CSV for round 1
        if meta_forecaster is not None and round_num == 1:
            _update_meta_predictions(csv_path, round_num, meta_forecaster)

    print(f"\n{'='*70}")
    print(f"Wavelet Study 3 COMPLETE — {dataset_name.upper()}")
    print(f"Results: {csv_path}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wavelet Study 3 — Successive Halving Search"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(WAVELET_STUDY3_DATASETS.keys()),
        help="Dataset to search (m4, tourism, weather, traffic)"
    )
    parser.add_argument(
        "--round", default=None,
        help="Run a specific round (1, 2, or 3). Default: all rounds."
    )
    parser.add_argument(
        "--pass", dest="pass_filter", default=None,
        choices=["baseline", "activeG_fcast"],
        help="Run only a specific pass. Default: both passes."
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
    run_wavelet_study3_search(args)


if __name__ == "__main__":
    main()
