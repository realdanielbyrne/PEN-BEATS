"""
Wavelet TrendAE Comparison — Top 10 WaveletV3 configs with TrendAE backbone

Takes the top 10 configurations from wavelet_study_3 round 3 and replaces the
Trend block with TrendAE to evaluate whether the autoencoder trend backbone
improves wavelet-based architectures on M4-Yearly.

Architecture: ["TrendAE", <WaveletV3>] * 5  (10 stacks total)

Fixed from wavelet_study_3 round 3 winners:
  - active_g=False, sum_losses=False, activation=ReLU
  - n_blocks_per_stack=1, share_weights=True
  - trend_thetas_dim=3 (except Symlet10 config which uses ttd=5)

New search dimension:
  - latent_dim: [2, 5, 8]  (AERootBlock bottleneck width)

Training: 30 epochs with early stopping (matches round 3 budget).
Seeds: 3 runs per config (seeds 42, 43, 44).

Results → experiments/results/m4/wavelet_trendae_comparison_results.csv

Usage:
    # Run full comparison
    python experiments/run_wavelet_trendae_comparison.py

    # Smoke test (2 epochs)
    python experiments/run_wavelet_trendae_comparison.py --max-epochs 2

    # Analyze results
    python experiments/run_wavelet_trendae_comparison.py --analyze
"""

import argparse
import csv
import gc
import math
import os
import sys

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
    LOSS,
    FORECAST_MULTIPLIERS,
    EARLY_STOPPING_PATIENCE,
    _shutdown_requested,
)

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(_EXPERIMENTS_DIR, "results")
EXPERIMENT_NAME = "wavelet_trendae_comparison"
N_RUNS = 3
MAX_EPOCHS = 30
LATENT_DIMS = [2, 5, 8]

# Extended CSV schema
COMPARISON_CSV_COLUMNS = CSV_COLUMNS + [
    "basis_dim", "basis_offset", "trend_thetas_dim_cfg",
    "wavelet_family", "bd_label", "latent_dim_cfg", "trend_block",
]

# ---------------------------------------------------------------------------
# Top 10 from wavelet_study_3 round 3 (ranked by mean OWA)
# ---------------------------------------------------------------------------

TOP10_CONFIGS = [
    {"name": "DB20_bd15_lt_bcast_ttd3",       "wavelet": "DB20WaveletV3",    "basis_dim": 15, "bd_label": "lt_bcast", "ttd": 3},
    {"name": "Coif2_bd4_lt_fcast_ttd3",       "wavelet": "Coif2WaveletV3",   "basis_dim": 4,  "bd_label": "lt_fcast", "ttd": 3},
    {"name": "Coif3_bd4_lt_fcast_ttd3",       "wavelet": "Coif3WaveletV3",   "basis_dim": 4,  "bd_label": "lt_fcast", "ttd": 3},
    {"name": "Haar_bd4_lt_fcast_ttd3",        "wavelet": "HaarWaveletV3",    "basis_dim": 4,  "bd_label": "lt_fcast", "ttd": 3},
    {"name": "Coif3_bd6_eq_fcast_ttd3",       "wavelet": "Coif3WaveletV3",   "basis_dim": 6,  "bd_label": "eq_fcast", "ttd": 3},
    {"name": "Symlet10_bd6_eq_fcast_ttd5",    "wavelet": "Symlet10WaveletV3","basis_dim": 6,  "bd_label": "eq_fcast", "ttd": 5},
    {"name": "DB10_bd15_lt_bcast_ttd3",       "wavelet": "DB10WaveletV3",    "basis_dim": 15, "bd_label": "lt_bcast", "ttd": 3},
    {"name": "Coif2_bd30_eq_bcast_ttd3",      "wavelet": "Coif2WaveletV3",   "basis_dim": 30, "bd_label": "eq_bcast", "ttd": 3},
    {"name": "Symlet3_bd4_lt_fcast_ttd3",     "wavelet": "Symlet3WaveletV3", "basis_dim": 4,  "bd_label": "lt_fcast", "ttd": 3},
    {"name": "Coif1_bd15_lt_bcast_ttd3",      "wavelet": "Coif1WaveletV3",   "basis_dim": 15, "bd_label": "lt_bcast", "ttd": 3},
]


def _csv_path():
    return os.path.join(RESULTS_DIR, "m4", "wavelet_trendae_comparison_results.csv")


def generate_configs():
    """Generate TrendAE configs for each top-10 wavelet config × latent_dim."""
    configs = {}
    for entry in TOP10_CONFIGS:
        for latent_dim in LATENT_DIMS:
            config_name = f"{entry['name']}_ld{latent_dim}"
            stack_types = ["TrendAE", entry["wavelet"]] * 5  # 10 stacks
            configs[config_name] = {
                "stack_types": stack_types,
                "n_blocks_per_stack": 1,
                "share_weights": True,
                "basis_dim": entry["basis_dim"],
                "basis_offset": 0,
                "trend_thetas_dim": entry["ttd"],
                "wavelet_family": entry["wavelet"],
                "bd_label": entry["bd_label"],
                "latent_dim": latent_dim,
            }
    return configs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_comparison(args):
    dataset_name = "m4"
    period = "Yearly"
    forecast_multiplier = FORECAST_MULTIPLIERS[dataset_name]
    max_epochs = args.max_epochs
    patience = min(max_epochs, EARLY_STOPPING_PATIENCE)

    csv_path = _csv_path()
    init_csv(csv_path, columns=COMPARISON_CSV_COLUMNS)

    configs = generate_configs()

    print(f"\n{'='*70}")
    print(f"Wavelet TrendAE Comparison — M4-Yearly")
    print(f"  Configs:     {len(configs)} (top 10 × {len(LATENT_DIMS)} latent_dims)")
    print(f"  Runs/config: {N_RUNS}")
    print(f"  Max epochs:  {max_epochs}")
    print(f"  Latent dims: {LATENT_DIMS}")
    print(f"  Total runs:  {len(configs) * N_RUNS}")
    print(f"{'='*70}")

    dataset = load_dataset(dataset_name, period)
    train_series_list = dataset.get_training_series()
    batch_size = get_batch_size(dataset_name, period, args.batch_size)

    completed = 0
    total = len(configs) * N_RUNS

    for config_name, cfg in configs.items():
        if _shutdown_requested:
            print("[SHUTDOWN] Exiting.")
            break

        for run_idx in range(N_RUNS):
            if result_exists(csv_path, EXPERIMENT_NAME, config_name, period, run_idx):
                completed += 1
                continue

            extra_row = {
                "basis_dim": cfg["basis_dim"],
                "basis_offset": cfg["basis_offset"],
                "trend_thetas_dim_cfg": cfg["trend_thetas_dim"],
                "wavelet_family": cfg["wavelet_family"],
                "bd_label": cfg["bd_label"],
                "latent_dim_cfg": cfg["latent_dim"],
                "trend_block": "TrendAE",
            }

            print(f"\n  [{completed+1}/{total}] {config_name} / run {run_idx}")

            run_single_experiment(
                experiment_name=EXPERIMENT_NAME,
                config_name=config_name,
                category="wavelet_trendae_comparison",
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
                accelerator_override=args.accelerator,
                forecast_multiplier=forecast_multiplier,
                num_workers=args.num_workers,
                wandb_enabled=False,
                save_predictions=False,
                basis_dim=cfg["basis_dim"],
                basis_offset=cfg["basis_offset"],
                extra_row=extra_row,
                csv_columns=COMPARISON_CSV_COLUMNS,
                trend_thetas_dim=cfg["trend_thetas_dim"],
                latent_dim_override=cfg["latent_dim"],
            )

            completed += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Wavelet TrendAE Comparison COMPLETE")
    print(f"Results: {csv_path}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results():
    """Print summary comparison of TrendAE results vs original Trend baselines."""
    from collections import defaultdict

    csv_path = _csv_path()
    if not os.path.exists(csv_path):
        print(f"[ERROR] No results file found: {csv_path}")
        return

    # Load TrendAE results
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[ERROR] No results found in CSV.")
        return

    # Group by config_name
    config_results = defaultdict(list)
    for row in rows:
        config_results[row["config_name"]].append(row)

    # Compute summary stats
    summaries = []
    for name, result_rows in config_results.items():
        owa_vals = []
        smape_vals = []
        mase_vals = []
        for r in result_rows:
            try:
                owa = float(r.get("owa", "nan"))
                if math.isfinite(owa):
                    owa_vals.append(owa)
            except (ValueError, TypeError):
                pass
            try:
                smape = float(r.get("smape", "nan"))
                if math.isfinite(smape):
                    smape_vals.append(smape)
            except (ValueError, TypeError):
                pass
            try:
                mase = float(r.get("mase", "nan"))
                if math.isfinite(mase):
                    mase_vals.append(mase)
            except (ValueError, TypeError):
                pass

        if not owa_vals:
            continue

        # Parse base config and latent_dim from name (e.g. DB20_bd15_lt_bcast_ttd3_ld5)
        parts = name.rsplit("_ld", 1)
        base_config = parts[0] if len(parts) == 2 else name
        latent_dim = int(parts[1]) if len(parts) == 2 else "?"

        summaries.append({
            "config_name": name,
            "base_config": base_config,
            "latent_dim": latent_dim,
            "mean_owa": float(np.mean(owa_vals)),
            "std_owa": float(np.std(owa_vals)),
            "mean_smape": float(np.mean(smape_vals)) if smape_vals else float("nan"),
            "mean_mase": float(np.mean(mase_vals)) if mase_vals else float("nan"),
            "n_runs": len(owa_vals),
        })

    summaries.sort(key=lambda s: s["mean_owa"])

    # --- Overall ranking ---
    print(f"\n{'='*80}")
    print(f"Wavelet TrendAE Comparison — M4-Yearly Results")
    print(f"{'='*80}")
    print(f"\n  {'Rank':<5} {'Config':<40} {'LD':>3} {'OWA':>8} {'±std':>7} "
          f"{'SMAPE':>8} {'MASE':>8} {'n':>3}")
    print(f"  {'-'*80}")

    for i, s in enumerate(summaries):
        print(f"  {i+1:<5} {s['config_name']:<40} {s['latent_dim']:>3} "
              f"{s['mean_owa']:>8.4f} {s['std_owa']:>7.4f} "
              f"{s['mean_smape']:>8.2f} {s['mean_mase']:>8.4f} {s['n_runs']:>3}")

    # --- Best latent_dim per base config ---
    print(f"\n  {'─'*60}")
    print(f"  Best latent_dim per base config:")
    print(f"  {'─'*60}")

    base_best = {}
    for s in summaries:
        bc = s["base_config"]
        if bc not in base_best or s["mean_owa"] < base_best[bc]["mean_owa"]:
            base_best[bc] = s

    for bc, s in sorted(base_best.items(), key=lambda x: x[1]["mean_owa"]):
        print(f"    {bc:<38} ld={s['latent_dim']:<3} OWA={s['mean_owa']:.4f}")

    # --- Marginal analysis by latent_dim ---
    print(f"\n  {'─'*60}")
    print(f"  Marginal analysis by latent_dim:")
    print(f"  {'─'*60}")

    ld_groups = defaultdict(list)
    for s in summaries:
        ld_groups[s["latent_dim"]].append(s["mean_owa"])

    for ld in sorted(ld_groups.keys()):
        vals = ld_groups[ld]
        print(f"    ld={ld:<3}  mean_OWA={np.mean(vals):.4f}  "
              f"median={np.median(vals):.4f}  n={len(vals)}")

    # --- Reference: original Trend baselines from wavelet_study_3 ---
    ws3_path = os.path.join(RESULTS_DIR, "m4", "wavelet_study_3_successive_results.csv")
    if os.path.exists(ws3_path):
        print(f"\n  {'─'*60}")
        print(f"  Comparison vs original Trend baselines (wavelet_study_3 R3):")
        print(f"  {'─'*60}")

        # Load round 3 results for matching base configs
        trend_results = defaultdict(list)
        with open(ws3_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("search_round") != "3":
                    continue
                trend_results[row["config_name"]].append(row)

        print(f"\n  {'Base Config':<38} {'Trend OWA':>10} {'TrendAE OWA':>12} {'Δ':>8}")
        print(f"  {'-'*70}")

        for bc in sorted(base_best.keys(), key=lambda x: base_best[x]["mean_owa"]):
            tae_owa = base_best[bc]["mean_owa"]
            tae_ld = base_best[bc]["latent_dim"]

            if bc in trend_results:
                trend_owas = []
                for r in trend_results[bc]:
                    try:
                        o = float(r.get("owa", "nan"))
                        if math.isfinite(o):
                            trend_owas.append(o)
                    except (ValueError, TypeError):
                        pass

                if trend_owas:
                    trend_mean = float(np.mean(trend_owas))
                    delta = tae_owa - trend_mean
                    arrow = "▼" if delta < 0 else "▲" if delta > 0 else "="
                    print(f"  {bc:<38} {trend_mean:>10.4f} {tae_owa:>10.4f} (ld={tae_ld}) "
                          f"{arrow}{abs(delta):>6.4f}")
                else:
                    print(f"  {bc:<38} {'N/A':>10} {tae_owa:>10.4f} (ld={tae_ld})")
            else:
                print(f"  {bc:<38} {'N/A':>10} {tae_owa:>10.4f} (ld={tae_ld})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wavelet TrendAE Comparison — Top 10 WaveletV3 configs with TrendAE"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=MAX_EPOCHS,
        help=f"Max training epochs (default: {MAX_EPOCHS})."
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze existing results without running new experiments."
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

    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    else:
        run_comparison(args)
        analyze_results()


if __name__ == "__main__":
    main()
