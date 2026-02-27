"""
GenericAE & BottleneckGenericAE Pure-Stack Successive Halving Study

Searches hyperparameters for pure homogeneous GenericAE and BottleneckGenericAE
stacks (no Trend anchors) to find optimal latent_dim, thetas_dim, and active_g
configurations.

Architecture: [GenericAE] * 10  or  [BottleneckGenericAE] * 10
Dataset: M4-Yearly or Weather-96

Hyperparameter search space:
  - latent_dim: [2, 4, 8, 16]     (AERootBlock bottleneck)
  - thetas_dim: [3, 5, 8, 10]     (BottleneckGenericAE projection bottleneck only)
  - active_g: [False, "forecast"]
  - Fixed: n_blocks_per_stack=1, share_weights=True, activation=ReLU, sum_losses=False
  - LR scheduler: warmup_epochs=15, CosineAnnealingLR after

Config count:
  - GenericAE: 4 latent_dim × 2 active_g = 8 configs
  - BottleneckGenericAE: 4 latent_dim × 4 thetas_dim × 2 active_g = 32 configs
  - Total: 40 configs

Successive halving (3 rounds):
  Round 1: 8 epochs, 3 runs/config, keep 50%   → 20 configs
  Round 2: 15 epochs, 3 runs/config, keep 50%  → 10 configs
  Round 3: 50 epochs, 3 runs/config, top 5 + NBEATS-I+G 10-stack baseline

Usage:
    # Full pipeline (all 3 rounds)
    python experiments/run_generic_ae_study.py --round all
    python experiments/run_generic_ae_study.py --dataset weather --round all

    # Single round
    python experiments/run_generic_ae_study.py --round 1

    # Analyze results
    python experiments/run_generic_ae_study.py --analyze

    # Smoke test
    python experiments/run_generic_ae_study.py --round 1 --max-epochs 2
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

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
    CSV_COLUMNS,
    THETAS_DIM,
    LATENT_DIM,
    FORECAST_MULTIPLIERS,
    EARLY_STOPPING_PATIENCE,
    _shutdown_requested,
)
from meta_forecaster import MetaForecaster

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(_EXPERIMENTS_DIR, "results")
EXPERIMENT_NAME = "generic_ae_pure_stack"

# Extended CSV schema
STUDY_CSV_COLUMNS = CSV_COLUMNS + [
    "search_round", "block_type", "latent_dim_cfg", "thetas_dim_cfg",
]

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

# Block types under study
BLOCK_TYPES = ["GenericAE", "BottleneckGenericAE"]

STUDY_DATASETS = {
    "m4": "Yearly",
    "weather": "Weather-96",
}

# Hyperparameter search space
SEARCH_LATENT_DIMS = [2, 4, 8, 16]
SEARCH_THETAS_DIMS = [3, 5, 8, 10]  # BottleneckGenericAE only
SEARCH_ACTIVE_G = [False, "forecast"]

# Architecture
N_STACKS = 10

# 10-stack NBEATS-I+G baseline — trained only in the final round for comparison
BASELINE_CONFIG_NAME = "NBEATS-I+G_10stack"
BASELINE_CONFIG = {
    "category": "baseline",
    "stack_types": ["Trend", "Seasonality"] + ["Generic"] * 8,
    "n_blocks_per_stack": 1,
    "share_weights": True,
    "block_type": "I+G",
    "latent_dim": LATENT_DIM,
    "thetas_dim": THETAS_DIM,
    "trend_thetas_dim": 3,  # paper-faithful cubic polynomial
    "active_g": False,
    "latent_dim_cfg": "",
    "thetas_dim_cfg": "",
}

# Successive halving schedule
ROUND_SCHEDULE = {
    1: {"max_epochs": 8,  "n_runs": 3, "keep_fraction": 0.50},
    2: {"max_epochs": 15, "n_runs": 3, "keep_fraction": 0.50},
    3: {"max_epochs": 50, "n_runs": 3, "top_k": 5},
}


# ---------------------------------------------------------------------------
# Config Generation
# ---------------------------------------------------------------------------

def generate_configs(round_num=1, promoted_configs=None):
    """Generate pure-stack GenericAE/BottleneckGenericAE search configs.

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
        Config name -> config dict.
    """
    configs = {}

    for block_type in BLOCK_TYPES:
        for latent_dim in SEARCH_LATENT_DIMS:
            # Determine thetas_dim search space
            if block_type == "BottleneckGenericAE":
                thetas_dims = SEARCH_THETAS_DIMS
            else:
                # GenericAE ignores thetas_dim (direct projection)
                thetas_dims = [THETAS_DIM]

            for thetas_dim in thetas_dims:
                for active_g in SEARCH_ACTIVE_G:
                    ag_str = "agF" if active_g == "forecast" else "ag0"
                    config_name = (
                        f"{block_type}_ld{latent_dim}_td{thetas_dim}_{ag_str}"
                    )

                    # For rounds 2+, only include promoted configs
                    if promoted_configs is not None and config_name not in promoted_configs:
                        continue

                    stack_types = [block_type] * N_STACKS

                    configs[config_name] = {
                        "category": f"ae_pure_round{round_num}",
                        "stack_types": stack_types,
                        "n_blocks_per_stack": 1,
                        "share_weights": True,
                        "block_type": block_type,
                        "latent_dim": latent_dim,
                        "thetas_dim": thetas_dim,
                        "active_g": active_g,
                        "latent_dim_cfg": latent_dim,
                        "thetas_dim_cfg": thetas_dim,
                    }

    return configs


# ---------------------------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------------------------

def _csv_path(dataset_name):
    """Return path to the study results CSV."""
    return os.path.join(RESULTS_DIR, dataset_name, "generic_ae_pure_stack_results.csv")


# ---------------------------------------------------------------------------
# Ranking & Promotion
# ---------------------------------------------------------------------------

def _load_round_results(csv_path, round_num):
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


def rank_and_promote(csv_path, round_num, keep_fraction, meta_forecaster=None, top_k_override=None):
    """Rank configs from a search round and select top configs for promotion.

    Returns list of promoted config_name strings.
    """
    rows = _load_round_results(csv_path, round_num)
    if not rows:
        print(f"  [WARN] No results found for round {round_num}.")
        return []

    # Group by config_name
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

            if meta_forecaster is not None and round_num == 1:
                raw_curve = r.get("val_loss_curve", "")
                try:
                    parsed = json.loads(raw_curve)
                    curve = [float(v) for v in parsed]
                    if len(curve) >= MetaForecaster.BACKCAST_LENGTH:
                        pred = meta_forecaster.predict(curve)
                        pred_best = float(pred.get("predicted_best", float("inf")))
                        pred_score = float(pred.get("convergence_score", float("inf")))
                        if math.isfinite(pred_best):
                            meta_best = min(meta_best, pred_best)
                        if math.isfinite(pred_score):
                            meta_score = min(meta_score, pred_score)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        if not val_losses:
            median_bvl = float("inf")
        else:
            median_bvl = float(np.median(val_losses))

        divergence_rate = n_diverged / len(result_rows) if result_rows else 0.0
        rank_metric = meta_best if (round_num == 1 and math.isfinite(meta_best)) else median_bvl

        rankings.append({
            "config_name": name,
            "median_best_val_loss": median_bvl,
            "n_runs": len(val_losses),
            "divergence_rate": divergence_rate,
            "meta_predicted_best": meta_best,
            "meta_convergence_score": meta_score,
            "rank_metric": rank_metric,
        })

    # Sort: penalize >50% divergence, then by meta metric (round 1) or median val_loss
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
    print(f"  Round {round_num} Ranking - {total_configs} configs, "
          f"promoting top {keep_n}")
    print(f"  {'='*65}")
    print(f"  {'Rank':<5} {'Config':<42} {'ValLoss':>9} {'Meta':>9} "
          f"{'Div%':>5} {'Runs':>4}")
    print(f"  {'-'*65}")

    for i, r in enumerate(rankings[:min(40, total_configs)]):
        marker = " *" if r["config_name"] in promoted else "  "
        div_str = f"{r['divergence_rate']*100:.0f}%"
        meta_str = (f"{r['meta_predicted_best']:.4f}"
                    if math.isfinite(r["meta_predicted_best"]) else "   --")
        print(f"  {i+1:<5}{marker} {r['config_name']:<40} "
              f"{r['median_best_val_loss']:>9.4f} {meta_str:>9} {div_str:>5} {r['n_runs']:>4}")

    return promoted


# ---------------------------------------------------------------------------
# Round Runner
# ---------------------------------------------------------------------------

def _run_search_round(round_num, configs, args, csv_path, dataset_name, period):
    """Run a single search round."""
    schedule = ROUND_SCHEDULE[round_num]
    max_epochs = args.max_epochs if args.max_epochs is not None else schedule["max_epochs"]
    n_runs = schedule["n_runs"]

    forecast_multiplier = FORECAST_MULTIPLIERS[dataset_name]

    # Early stopping only in rounds 2+
    if round_num >= 2:
        patience = min(max_epochs, EARLY_STOPPING_PATIENCE)
    else:
        patience = max_epochs  # No early stopping in round 1

    # LR scheduler (used in all rounds, but most impactful in round 3)
    warmup_epochs = 15
    scheduler_cfg = {
        "warmup_epochs": warmup_epochs,
        "T_max": max(max_epochs - warmup_epochs, 1),
        "eta_min": 1e-6,
    }

    n_configs = len(configs)
    config_items = list(configs.items())
    if args.reverse:
        config_items.reverse()

    print(f"\n  {'='*60}")
    print(f"  ROUND {round_num}: {n_configs} configs x {n_runs} runs x "
          f"{max_epochs} epochs")
    if args.reverse:
        print("  Order: REVERSE")
    print(f"  {'='*60}")

    if n_configs == 0:
        print("  No configs to run!")
        return

    dataset = load_dataset(dataset_name, period)
    train_series_list = dataset.get_training_series()
    batch_size = get_batch_size(dataset_name, period, args.batch_size)

    completed = 0
    total = n_configs * n_runs

    for config_name, cfg in config_items:
        if _shutdown_requested:
            print("[SHUTDOWN] Exiting search round.")
            return

        for run_idx in range(n_runs):
            experiment_tag = f"ae_pure_r{round_num}"
            if result_exists(csv_path, experiment_tag, config_name, period, run_idx):
                completed += 1
                continue

            extra_row = {
                "search_round": round_num,
                "block_type": cfg["block_type"],
                "latent_dim_cfg": cfg["latent_dim_cfg"],
                "thetas_dim_cfg": cfg["thetas_dim_cfg"],
            }

            print(f"\n  [{completed+1}/{total}] {config_name} / run {run_idx}")

            run_single_experiment(
                experiment_name=experiment_tag,
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
                active_g=cfg["active_g"],
                sum_losses=False,
                activation="ReLU",
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                accelerator_override=args.accelerator,
                forecast_multiplier=forecast_multiplier,
                num_workers=args.num_workers,
                wandb_enabled=False,
                save_predictions=False,
                extra_row=extra_row,
                csv_columns=STUDY_CSV_COLUMNS,
                thetas_dim_override=cfg["thetas_dim"],
                latent_dim_override=cfg["latent_dim"],
                trend_thetas_dim=cfg.get("trend_thetas_dim", 5),
                lr_scheduler_config=scheduler_cfg,
            )

            completed += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main Study Pipeline
# ---------------------------------------------------------------------------

def run_study(args):
    """Run the full successive halving search."""
    dataset_name = args.dataset
    period = STUDY_DATASETS[dataset_name]
    csv_path = _csv_path(dataset_name)
    init_csv(csv_path, columns=STUDY_CSV_COLUMNS)

    # Determine which rounds to run
    round_spec = args.round
    if round_spec == "all":
        rounds_to_run = [1, 2, 3]
    else:
        rounds_to_run = [int(round_spec)]

    all_configs = generate_configs()
    print(f"\n{'='*70}")
    print(f"GenericAE & BottleneckGenericAE Pure-Stack Study - {dataset_name}/{period}")
    print(f"  Total config space: {len(all_configs)} configs")
    print(f"  Rounds to run:      {rounds_to_run}")
    if args.reverse:
        print("  Order:             REVERSE")
    print(f"{'='*70}")

    # Step 0: Train / load meta-forecaster for round 1 ranking.
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

    promoted = None
    for round_num in rounds_to_run:
        if _shutdown_requested:
            break

        # Generate configs for this round
        if round_num == 1:
            configs = generate_configs(round_num)
        else:
            # Load promotions from prior round
            if promoted is None:
                prior_round = round_num - 1
                prior_schedule = ROUND_SCHEDULE[prior_round]
                promoted = rank_and_promote(
                    csv_path, prior_round,
                    prior_schedule.get("keep_fraction"),
                    meta_forecaster=meta_forecaster if prior_round == 1 else None,
                    top_k_override=prior_schedule.get("top_k"),
                )
                if not promoted:
                    print(f"  [ERROR] No configs promoted from round {prior_round}. "
                          f"Run round {prior_round} first.")
                    return

            configs = generate_configs(
                round_num, promoted_configs=set(promoted)
            )

        # Add NBEATS-I+G baseline in the final round for comparison
        if round_num == 3:
            configs[BASELINE_CONFIG_NAME] = BASELINE_CONFIG

        _run_search_round(round_num, configs, args, csv_path, dataset_name, period)

        # Rank and promote for the next round
        schedule = ROUND_SCHEDULE[round_num]
        promoted = rank_and_promote(
            csv_path, round_num,
            schedule.get("keep_fraction"),
            meta_forecaster=meta_forecaster if round_num == 1 else None,
            top_k_override=schedule.get("top_k"),
        )

    print(f"\n{'='*70}")
    print(f"GenericAE Pure-Stack Study COMPLETE")
    print(f"Results: {csv_path}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(dataset_name="m4"):
    """Comprehensive analysis of study results."""
    period = STUDY_DATASETS[dataset_name]
    csv_path = _csv_path(dataset_name)
    if not os.path.exists(csv_path):
        print(f"[ERROR] No results file found: {csv_path}")
        return

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[ERROR] No results found in CSV.")
        return

    # Find the highest round with data
    rounds_present = set()
    for row in rows:
        try:
            rounds_present.add(int(row.get("search_round", 0)))
        except (ValueError, TypeError):
            pass

    final_round = max(rounds_present) if rounds_present else 0

    # Group by config_name (use final round results when available, else latest)
    config_results = defaultdict(list)
    for row in rows:
        try:
            rnd = int(row.get("search_round", 0))
        except (ValueError, TypeError):
            rnd = 0
        if rnd == final_round:
            config_results[row["config_name"]].append(row)

    owa_values = []
    for row in config_results.values():
        for r in row:
            try:
                val = float(r.get("owa", "nan"))
                if math.isfinite(val):
                    owa_values.append(val)
            except (ValueError, TypeError):
                pass
    has_owa = len(owa_values) > 0
    metric_name = "OWA" if has_owa else "best_val_loss"

    # Compute summary stats
    summaries = []
    for name, result_rows in config_results.items():
        owa_vals, smape_vals, mase_vals, bvl_vals = [], [], [], []
        n_diverged = 0
        n_params = None

        for r in result_rows:
            for metric, vals in [("owa", owa_vals), ("smape", smape_vals), ("mase", mase_vals)]:
                try:
                    v = float(r.get(metric, "nan"))
                    if math.isfinite(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    pass

            try:
                bvl = float(r.get("best_val_loss", "nan"))
                if math.isfinite(bvl):
                    bvl_vals.append(bvl)
            except (ValueError, TypeError):
                pass

            if r.get("diverged", "").lower() in ("true", "1"):
                n_diverged += 1

            if n_params is None:
                try:
                    n_params = int(r.get("n_params", 0))
                except (ValueError, TypeError):
                    pass

        metric_vals = owa_vals if has_owa else bvl_vals
        if not metric_vals:
            continue

        convergence_rate = (len(result_rows) - n_diverged) / len(result_rows) if result_rows else 0

        # Parse config components
        block_type = result_rows[0].get("block_type", "")
        latent_dim = result_rows[0].get("latent_dim_cfg", "")
        thetas_dim = result_rows[0].get("thetas_dim_cfg", "")
        active_g_str = "forecast" if "_agF" in name else "False"

        summaries.append({
            "config_name": name,
            "block_type": block_type,
            "latent_dim": latent_dim,
            "thetas_dim": thetas_dim,
            "active_g": active_g_str,
            "mean_metric": float(np.mean(metric_vals)),
            "std_metric": float(np.std(metric_vals)),
            "mean_owa": float(np.mean(owa_vals)),
            "std_owa": float(np.std(owa_vals)),
            "mean_smape": float(np.mean(smape_vals)) if smape_vals else float("nan"),
            "mean_mase": float(np.mean(mase_vals)) if mase_vals else float("nan"),
            "mean_best_val_loss": float(np.mean(bvl_vals)) if bvl_vals else float("nan"),
            "n_runs": len(metric_vals),
            "n_params": n_params or 0,
            "convergence_rate": convergence_rate,
        })

    summaries.sort(key=lambda s: s["mean_metric"])

    print(f"\n{'='*90}")
    print(f"GenericAE & BottleneckGenericAE Pure-Stack Study - {dataset_name}/{period} Results")
    print(f"  Final round: {final_round}  |  Configs with results: {len(summaries)}")
    print(f"  Primary metric: {metric_name}")
    print(f"{'='*90}")

    print(f"\n  {'Rank':<5} {'Config':<42} {metric_name:>14} {'+/-':>6} "
          f"{'SMAPE':>7} {'MASE':>7} {'Params':>8} {'Conv%':>5} {'n':>3}")
    print(f"  {'-'*90}")

    for i, s in enumerate(summaries):
        conv_str = f"{s['convergence_rate']*100:.0f}%"
        params_str = f"{s['n_params']:,}" if s['n_params'] else "?"
        print(
            f"  {i+1:<5} {s['config_name']:<42} {s['mean_metric']:>14.4f} "
            f"{s['std_metric']:>6.4f} {s['mean_smape']:>7.2f} {s['mean_mase']:>7.4f} "
            f"{params_str:>8} {conv_str:>5} {s['n_runs']:>3}"
        )

    print(f"\n  {'='*60}")
    print("  Best config per block type:")
    print(f"  {'='*60}")

    block_best = {}
    for s in summaries:
        bt = s["block_type"]
        if bt not in block_best or s["mean_metric"] < block_best[bt]["mean_metric"]:
            block_best[bt] = s

    for bt in BLOCK_TYPES + ["I+G"]:
        if bt not in block_best:
            print(f"    {bt:<25} (no results)")
            continue
        s = block_best[bt]
        if bt == "I+G":
            print(
                f"    {bt:<25} {s['config_name']:<35} {metric_name}={s['mean_metric']:.4f} "
                f"(10-stack baseline)"
            )
        else:
            print(
                f"    {bt:<25} {s['config_name']:<35} {metric_name}={s['mean_metric']:.4f} "
                f"(ld={s['latent_dim']}, td={s['thetas_dim']}, ag={s['active_g']})"
            )

    print(f"\n  {'='*60}")
    print("  Marginal analysis by latent_dim:")
    print(f"  {'='*60}")
    ld_groups = defaultdict(list)
    for s in summaries:
        if s["block_type"] != "I+G":
            ld_groups[s["latent_dim"]].append(s["mean_metric"])
    for ld in sorted(ld_groups.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        vals = ld_groups[ld]
        print(
            f"    ld={str(ld):<4}  mean_{metric_name}={np.mean(vals):.4f}  "
            f"median={np.median(vals):.4f}  n_configs={len(vals)}"
        )

    print(f"\n  {'='*60}")
    print("  Marginal analysis by thetas_dim (BottleneckGenericAE only):")
    print(f"  {'='*60}")
    td_groups = defaultdict(list)
    for s in summaries:
        if s["block_type"] == "BottleneckGenericAE":
            td_groups[s["thetas_dim"]].append(s["mean_metric"])
    if td_groups:
        for td in sorted(td_groups.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
            vals = td_groups[td]
            print(
                f"    td={str(td):<4}  mean_{metric_name}={np.mean(vals):.4f}  "
                f"median={np.median(vals):.4f}  n_configs={len(vals)}"
            )
    else:
        print("    (no BottleneckGenericAE results)")

    print(f"\n  {'='*60}")
    print("  Marginal analysis by active_g:")
    print(f"  {'='*60}")
    ag_groups = defaultdict(list)
    for s in summaries:
        if s["block_type"] != "I+G":
            ag_groups[s["active_g"]].append(s["mean_metric"])
    for ag in sorted(ag_groups.keys()):
        vals = ag_groups[ag]
        print(
            f"    active_g={ag:<10}  mean_{metric_name}={np.mean(vals):.4f}  "
            f"median={np.median(vals):.4f}  n_configs={len(vals)}"
        )

    if has_owa and dataset_name == "m4":
        unified_csv = os.path.join(RESULTS_DIR, dataset_name, "unified_benchmark_results.csv")
        if os.path.exists(unified_csv):
            print(f"\n  {'='*60}")
            print("  Comparison vs Part 1 benchmark (30-stack, fixed params):")
            print(f"  {'='*60}")

            p1_results = defaultdict(list)
            with open(unified_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("period") != period:
                        continue
                    cname = row.get("config_name", "")
                    if cname in ("GenericAE", "BottleneckGenericAE", "NBEATS-G", "NBEATS-I", "NBEATS-I+G"):
                        p1_results[cname].append(row)

            print(f"\n  {'Config':<30} {'Arch':>12} {'OWA':>7} {'Params':>10} {'Note'}")
            print(f"  {'-'*75}")

            for cname in ["NBEATS-G", "NBEATS-I", "NBEATS-I+G", "GenericAE", "BottleneckGenericAE"]:
                if cname in p1_results:
                    owas = []
                    for r in p1_results[cname]:
                        try:
                            o = float(r.get("owa", "nan"))
                            if math.isfinite(o):
                                owas.append(o)
                        except (ValueError, TypeError):
                            pass
                    if owas:
                        params = p1_results[cname][0].get("n_params", "?")
                        print(
                            f"  {cname:<30} {'30-stack':>12} {np.mean(owas):>7.4f} "
                            f"{params:>10} Part 1 baseline"
                        )

            for bt in BLOCK_TYPES:
                if bt in block_best:
                    s = block_best[bt]
                    print(
                        f"  {s['config_name']:<30} {'10-stack':>12} {s['mean_metric']:>7.4f} "
                        f"{s['n_params']:>10,} This study best"
                    )

        ae_trend_csv = os.path.join(RESULTS_DIR, dataset_name, "ae_trend_search_results.csv")
        if os.path.exists(ae_trend_csv):
            print(f"\n  {'='*60}")
            print("  Comparison vs ae_trend_study ([Trend, X] x 5):")
            print(f"  {'='*60}")

            at_results = defaultdict(list)
            with open(ae_trend_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ae_var = row.get("ae_variant", "")
                    if ae_var in ("GenericAE", "BottleneckGenericAE"):
                        at_results[ae_var].append(row)

            for bt in BLOCK_TYPES:
                at_owas = []
                for r in at_results.get(bt, []):
                    try:
                        o = float(r.get("owa", "nan"))
                        if math.isfinite(o):
                            at_owas.append(o)
                    except (ValueError, TypeError):
                        pass

                if at_owas:
                    at_best_owa = min(at_owas)
                    at_mean = float(np.mean(at_owas))
                    print(
                        f"    {bt:<25} ae_trend best_OWA={at_best_owa:.4f} "
                        f"mean_OWA={at_mean:.4f}  (n={len(at_owas)})"
                    )
                    if bt in block_best:
                        delta = block_best[bt]["mean_metric"] - at_best_owa
                        label = "pure-stack better" if delta < 0 else "Trend-anchored better"
                        print(
                            f"    {'':<25} pure-stack best_OWA={block_best[bt]['mean_metric']:.4f} "
                            f"delta={delta:+.4f}  ({label})"
                        )

    if not has_owa:
        print(f"\n  [INFO] OWA is unavailable for dataset '{dataset_name}'.")
        print("  [INFO] Rankings and marginals above use best_val_loss instead.")

    print(f"\n{'='*70}")
    print(f"Analysis complete. Results: {csv_path}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GenericAE & BottleneckGenericAE Pure-Stack Successive Halving Study"
    )
    parser.add_argument(
        "--dataset", default="m4", choices=sorted(STUDY_DATASETS.keys()),
        help="Dataset to run/analyze."
    )
    parser.add_argument(
        "--round", default="all",
        help="Run a specific round (1, 2, or 3) or 'all'. Default: all."
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze existing results without running new experiments."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override max epochs for the current round."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size."
    )
    parser.add_argument(
        "--accelerator", default="auto",
        help="Accelerator override: auto, cuda, mps, cpu."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader num_workers."
    )
    parser.add_argument(
        "--reverse", action="store_true", default=False,
        help="Iterate configs in reverse order (for parallel execution with a second process)."
    )

    args = parser.parse_args()

    if args.analyze:
        analyze_results(args.dataset)
    else:
        run_study(args)
        analyze_results(args.dataset)


if __name__ == "__main__":
    main()
