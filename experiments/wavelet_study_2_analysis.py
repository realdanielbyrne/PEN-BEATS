"""
Wavelet Study 2 — Basis Dimension Sweep Analysis

Analyzes results from run_wavelet_study_2_basis_dim.py:
  - Factorial design: 3 wavelet families × 4 basis_dim levels × 2 thetas_dim levels
  - 3 seeds per config (72 total runs)
  - Dataset: M4-Yearly (forecast_length=6, backcast_length=30)

Produces:
  1. Overall summary statistics
  2. Marginal effects of each factor (wavelet, basis_dim, thetas_dim)
  3. Two-way interaction tables (wavelet×basis_dim, wavelet×thetas_dim, basis_dim×thetas_dim)
  4. Top/bottom config rankings
  5. ANOVA-style variance decomposition
  6. Convergence diagnostics (epochs, early stopping, val_loss curves)

Usage:
    python experiments/wavelet_study_2_analysis.py
"""

import ast
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.float_format", "{:.4f}".format)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_SCRIPT_DIR, "results", "m4", "wavelet_study_2_basis_dim_results.csv")

METRICS = ["smape", "mase", "owa", "best_val_loss"]
NUMERIC_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "basis_dim", "basis_offset", "thetas_dim",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title, char="=", width=80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def _parse_wavelet(config_name):
    """Extract wavelet family short name from config_name like 'Haar_bd6_eq_fcast_td3'."""
    return config_name.split("_")[0]


def _parse_bd_label(config_name):
    """Extract basis_dim label (eq_fcast, lt_fcast, eq_bcast, lt_bcast)."""
    parts = config_name.split("_")
    return f"{parts[2]}_{parts[3]}"


def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Results file not found: {CSV_PATH}")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["wavelet"] = df["config_name"].apply(_parse_wavelet)
    df["bd_label"] = df["config_name"].apply(_parse_bd_label)
    df["thetas_dim"] = df["thetas_dim"].astype(int)
    df["basis_dim"] = df["basis_dim"].astype(int)
    return df


# ---------------------------------------------------------------------------
# 1. Overall Summary
# ---------------------------------------------------------------------------

def section_overall_summary(df):
    _banner("1. OVERALL SUMMARY")
    print(f"Total runs:       {len(df)}")
    print(f"Unique configs:   {df['config_name'].nunique()}")
    print(f"Runs per config:  {df.groupby('config_name').size().unique()}")
    print(f"Wavelet families: {sorted(df['wavelet'].unique())}")
    print(f"basis_dim values: {sorted(df['basis_dim'].unique())}")
    print(f"thetas_dim vals:  {sorted(df['thetas_dim'].unique())}")
    print()
    print("── Global metric statistics ──")
    print(df[METRICS].describe().T.to_string())
    print()


# ---------------------------------------------------------------------------
# 2. Marginal Effects
# ---------------------------------------------------------------------------

def section_marginal_effects(df):
    _banner("2. MARGINAL EFFECTS (mean ± std across seeds)")

    for factor, label in [("wavelet", "Wavelet Family"),
                          ("basis_dim", "Basis Dimension"),
                          ("thetas_dim", "Trend Thetas Dim")]:
        print(f"\n── {label} ──")
        agg = df.groupby(factor)[METRICS].agg(["mean", "std", "count"])
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        for m in METRICS:
            agg[f"{m}_summary"] = agg.apply(
                lambda r: f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}", axis=1
            )
        summary_cols = [f"{m}_summary" for m in METRICS]
        print(agg[summary_cols + [f"{METRICS[0]}_count"]].rename(
            columns={f"{METRICS[0]}_count": "n_runs"}
        ).to_string())
    print()


# ---------------------------------------------------------------------------
# 3. Two-Way Interaction Tables
# ---------------------------------------------------------------------------

def section_interactions(df):
    _banner("3. TWO-WAY INTERACTION TABLES (mean sMAPE)")

    pairs = [
        ("wavelet", "basis_dim", "Wavelet × Basis Dim"),
        ("wavelet", "thetas_dim", "Wavelet × Thetas Dim"),
        ("basis_dim", "thetas_dim", "Basis Dim × Thetas Dim"),
    ]
    for row_factor, col_factor, title in pairs:
        print(f"\n── {title} ──")
        pivot = df.pivot_table(
            values="smape", index=row_factor, columns=col_factor, aggfunc="mean"
        )
        print(pivot.to_string())
        print()

    # Same for OWA
    print("\n── Wavelet × Basis Dim (mean OWA) ──")
    pivot_owa = df.pivot_table(
        values="owa", index="wavelet", columns="basis_dim", aggfunc="mean"
    )
    print(pivot_owa.to_string())
    print()


# ---------------------------------------------------------------------------
# 4. Config Rankings
# ---------------------------------------------------------------------------

def section_rankings(df):
    _banner("4. CONFIG RANKINGS")

    agg = df.groupby("config_name").agg(
        wavelet=("wavelet", "first"),
        basis_dim=("basis_dim", "first"),
        bd_label=("bd_label", "first"),
        thetas_dim=("thetas_dim", "first"),
        smape_mean=("smape", "mean"),
        smape_std=("smape", "std"),
        mase_mean=("mase", "mean"),
        owa_mean=("owa", "mean"),
        owa_std=("owa", "std"),
        val_loss_mean=("best_val_loss", "mean"),
        epochs_mean=("epochs_trained", "mean"),
        n_params=("n_params", "first"),
        runs=("smape", "count"),
    ).sort_values("smape_mean")

    print("\n── Top 10 configs by mean sMAPE ──")
    print(agg.head(10).to_string())
    print()
    print("── Bottom 5 configs by mean sMAPE ──")
    print(agg.tail(5).to_string())
    print()

    # Best single run
    best_row = df.loc[df["smape"].idxmin()]
    print(f"── Best single run ──")
    print(f"  Config:    {best_row['config_name']}")
    print(f"  Run/Seed:  {best_row['run']}/{best_row['seed']}")
    print(f"  sMAPE:     {best_row['smape']:.4f}")
    print(f"  MASE:      {best_row['mase']:.4f}")
    print(f"  OWA:       {best_row['owa']:.4f}")
    print()


# ---------------------------------------------------------------------------
# 5. Statistical Tests
# ---------------------------------------------------------------------------

def section_statistical_tests(df):
    _banner("5. STATISTICAL TESTS")

    # Kruskal-Wallis for each factor on sMAPE
    for factor in ["wavelet", "basis_dim", "thetas_dim"]:
        groups = [g["smape"].values for _, g in df.groupby(factor)]
        stat, p = stats.kruskal(*groups)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  Kruskal-Wallis on sMAPE by {factor:12s}: H={stat:8.3f}  p={p:.4f}  {sig}")

    print()

    # Pairwise Mann-Whitney for basis_dim (the main factor of interest)
    bd_values = sorted(df["basis_dim"].unique())
    print("── Pairwise Mann-Whitney U (basis_dim on sMAPE) ──")
    for i in range(len(bd_values)):
        for j in range(i + 1, len(bd_values)):
            g1 = df[df["basis_dim"] == bd_values[i]]["smape"]
            g2 = df[df["basis_dim"] == bd_values[j]]["smape"]
            stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  bd={bd_values[i]:2d} vs bd={bd_values[j]:2d}: U={stat:6.1f}  p={p:.4f}  {sig}")
    print()

    # Two-way ANOVA approximation via OLS (if enough data)
    try:
        from scipy.stats import f_oneway
        print("── One-way ANOVA on sMAPE by basis_dim ──")
        groups = [g["smape"].values for _, g in df.groupby("basis_dim")]
        f_stat, p_val = f_oneway(*groups)
        print(f"  F={f_stat:.3f}  p={p_val:.4f}")
    except Exception as e:
        print(f"  ANOVA skipped: {e}")
    print()



# ---------------------------------------------------------------------------
# 6. Convergence Diagnostics
# ---------------------------------------------------------------------------

def section_convergence(df):
    _banner("6. CONVERGENCE DIAGNOSTICS")

    # Epochs trained summary by factor
    print("── Epochs trained by basis_dim ──")
    ep_bd = df.groupby("basis_dim")["epochs_trained"].agg(["mean", "std", "min", "max"])
    print(ep_bd.to_string())
    print()

    print("── Epochs trained by wavelet ──")
    ep_wv = df.groupby("wavelet")["epochs_trained"].agg(["mean", "std", "min", "max"])
    print(ep_wv.to_string())
    print()

    # Training time
    print("── Training time (seconds) by basis_dim ──")
    tt = df.groupby("basis_dim")["training_time_seconds"].agg(["mean", "std"])
    print(tt.to_string())
    print()

    # Parameter count by basis_dim
    print("── Parameter count by basis_dim ──")
    params = df.groupby("basis_dim")["n_params"].agg(["mean", "min", "max"])
    print(params.to_string())
    print()

    # Loss ratio (final_val_loss / best_val_loss) — measures overfitting
    print("── Loss ratio (final_val / best_val) by basis_dim ──")
    lr = df.groupby("basis_dim")["loss_ratio"].agg(["mean", "std", "max"])
    print(lr.to_string())
    print()

    # Val loss curve analysis
    if "val_loss_curve" in df.columns:
        print("── Val loss curve analysis ──")
        curve_lengths = []
        final_improvements = []
        for _, row in df.iterrows():
            try:
                curve = ast.literal_eval(row["val_loss_curve"])
                curve = [float(v) for v in curve]
                curve_lengths.append(len(curve))
                if len(curve) >= 2:
                    final_improvements.append(curve[0] - curve[-1])
            except (ValueError, SyntaxError):
                pass
        if curve_lengths:
            print(f"  Curve lengths: mean={np.mean(curve_lengths):.1f}, "
                  f"min={min(curve_lengths)}, max={max(curve_lengths)}")
            print(f"  Improvement (first - last val_loss): "
                  f"mean={np.mean(final_improvements):.3f}, "
                  f"std={np.std(final_improvements):.3f}")
    print()


# ---------------------------------------------------------------------------
# 7. Effect Size & Practical Significance
# ---------------------------------------------------------------------------

def section_effect_sizes(df):
    _banner("7. EFFECT SIZES & PRACTICAL SIGNIFICANCE")

    # Range of mean sMAPE across configs
    config_means = df.groupby("config_name")["smape"].mean()
    smape_range = config_means.max() - config_means.min()
    print(f"  sMAPE range across configs: {smape_range:.4f}")
    print(f"  Best config mean sMAPE:     {config_means.min():.4f} ({config_means.idxmin()})")
    print(f"  Worst config mean sMAPE:    {config_means.max():.4f} ({config_means.idxmax()})")
    print()

    # Effect size per factor (eta-squared approximation)
    grand_mean = df["smape"].mean()
    ss_total = ((df["smape"] - grand_mean) ** 2).sum()

    print("── Eta-squared (proportion of variance explained) ──")
    for factor in ["wavelet", "basis_dim", "thetas_dim"]:
        group_means = df.groupby(factor)["smape"].transform("mean")
        ss_between = ((group_means - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        print(f"  {factor:12s}: η² = {eta_sq:.4f}  ({eta_sq*100:.1f}%)")
    print()

    # Basis dim label analysis (semantic grouping)
    print("── sMAPE by basis_dim semantic label ──")
    bd_label_agg = df.groupby("bd_label")["smape"].agg(["mean", "std", "count"])
    print(bd_label_agg.sort_values("mean").to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()

    section_overall_summary(df)
    section_marginal_effects(df)
    section_interactions(df)
    section_rankings(df)
    section_statistical_tests(df)
    section_convergence(df)
    section_effect_sizes(df)

    _banner("ANALYSIS COMPLETE", char="=")
    print(f"  Data: {CSV_PATH}")
    print(f"  Rows: {len(df)}")
    print()


if __name__ == "__main__":
    main()
