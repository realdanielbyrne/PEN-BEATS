"""Wavelet Study 3 — Successive Halving Comprehensive Analysis

Analyzes results from the WaveletV3 successive halving search:
  - 14 wavelet families × 4 basis_dim labels × 2 trend_thetas_dim × 2 active_g passes
  - 3 rounds (7→15→30 epochs), pruning 112→58→29 configs
  - 1,194 total rows (3 seeds × 2 passes per config per round)
  - Dataset: M4-Yearly (forecast_length=6, backcast_length=30)

Produces 14 analysis sections covering factor effects, elimination patterns,
baseline comparisons, stability, convergence, and final recommendations.

Usage:
    python experiments/wavelet_study_3_analysis.py
"""

import ast
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.float_format", "{:.4f}".format)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_SCRIPT_DIR, "results", "m4", "wavelet_study_3_successive_results.csv")
BASELINE_CSV = os.path.join(_SCRIPT_DIR, "results", "m4", "block_benchmark_results.csv")
AE_TREND_CSV = os.path.join(_SCRIPT_DIR, "results", "m4", "ae_trend_search_results.csv")

METRICS = ["smape", "mase", "owa", "best_val_loss"]
NUMERIC_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "basis_dim", "basis_offset",
    "trend_thetas_dim_cfg", "meta_predicted_best", "meta_convergence_score",
]

# Hardcoded baseline reference values (M4-Yearly, 30-stack)
BASELINES = {
    "NBEATS-I+G":  {"owa": 0.8057, "owa_std": 0.008, "smape": 13.53, "params": 35_900_000},
    "GenericAE":   {"owa": 0.8063, "owa_std": 0.007, "smape": 13.57, "params": 4_800_000},
    "AutoEncoder": {"owa": 0.8075, "owa_std": 0.012, "smape": 13.56, "params": 24_900_000},
    "NBEATS-I":    {"owa": 0.8132, "owa_std": 0.009, "smape": 13.67, "params": 12_900_000},
    "NBEATS-G":    {"owa": 0.8198, "owa_std": 0.013, "smape": 13.70, "params": 24_700_000},
    "AE+Trend":    {"owa": 0.8015, "owa_std": 0.006, "smape": 13.53, "params": 5_200_000},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title, char="=", width=90):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def _sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Results file not found: {CSV_PATH}")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Derive short wavelet name
    df["wavelet"] = df["wavelet_family"].str.replace("WaveletV3", "", regex=False)
    # Integer alias for trend_thetas_dim
    df["ttd"] = df["trend_thetas_dim_cfg"].astype(int)
    # Ensure active_g_cfg is string
    df["active_g_cfg"] = df["active_g_cfg"].astype(str)
    df["search_round"] = pd.to_numeric(df["search_round"], errors="coerce").astype(int)
    return df


def load_baselines():
    """Try to load baseline CSVs, return (block_df|None, ae_trend_df|None)."""
    block_df, ae_df = None, None
    if os.path.exists(BASELINE_CSV):
        try:
            bdf = pd.read_csv(BASELINE_CSV)
            bdf = bdf[bdf["period"] == "Yearly"]
            for c in ["smape", "mase", "owa", "n_params", "training_time_seconds"]:
                if c in bdf.columns:
                    bdf[c] = pd.to_numeric(bdf[c], errors="coerce")
            # Filter to valid baselines only (exclude blown-up wavelet V1 runs)
            valid = bdf.groupby("config_name")["owa"].mean()
            valid = valid[valid < 2.0].index
            block_df = bdf[bdf["config_name"].isin(valid)]
        except Exception as e:
            print(f"  [WARN] Could not load block baselines: {e}")
    if os.path.exists(AE_TREND_CSV):
        try:
            adf = pd.read_csv(AE_TREND_CSV)
            for c in ["smape", "mase", "owa", "n_params", "training_time_seconds"]:
                if c in adf.columns:
                    adf[c] = pd.to_numeric(adf[c], errors="coerce")
            adf["search_round"] = pd.to_numeric(adf["search_round"], errors="coerce").astype(int)
            ae_df = adf[adf["search_round"] == adf["search_round"].max()]
        except Exception as e:
            print(f"  [WARN] Could not load AE+Trend baselines: {e}")
    return block_df, ae_df


# ---------------------------------------------------------------------------
# 1. Overview & Data Summary
# ---------------------------------------------------------------------------

def section_overview(df):
    _banner("1. OVERVIEW & DATA SUMMARY")
    n_wavelets = df["wavelet"].nunique()
    n_bd = df["bd_label"].nunique()
    n_ttd = df["ttd"].nunique()
    n_ag = df["active_g_cfg"].nunique()
    print(f"  CSV:              {CSV_PATH}")
    print(f"  Total rows:       {len(df)}")
    print(f"  Unique configs:   {df['config_name'].nunique()}")
    print(f"  Search rounds:    {sorted(df['search_round'].unique())}")
    print(f"  Config space:     {n_wavelets} wavelets x {n_bd} bd_labels x "
          f"{n_ttd} ttd x {n_ag} active_g = {n_wavelets * n_bd * n_ttd} base configs")
    print(f"  Wavelet families: {sorted(df['wavelet'].unique())}")
    print(f"  basis_dim labels: {sorted(df['bd_label'].unique())}")
    print(f"  basis_dim values: {sorted(df['basis_dim'].unique())}")
    print(f"  ttd values:       {sorted(df['ttd'].unique())}")
    print(f"  active_g_cfg:     {sorted(df['active_g_cfg'].unique())}")
    print()

    print("── Per-round breakdown ──")
    print(f"  {'Round':>5s}  {'Configs':>8s}  {'Rows':>6s}  {'Epochs':>8s}  {'Passes':>8s}")
    print(f"  {'-'*42}")
    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        ep_range = f"{rdf['epochs_trained'].min()}-{rdf['epochs_trained'].max()}"
        passes = ", ".join(sorted(rdf["active_g_cfg"].unique()))
        print(f"  {r:5d}  {rdf['config_name'].nunique():8d}  {len(rdf):6d}  {ep_range:>8s}  {passes}")
    print()

    print("── Global metric statistics ──")
    print(df[METRICS].describe().T.to_string())
    print()


# ---------------------------------------------------------------------------
# 2. Successive Halving Funnel
# ---------------------------------------------------------------------------

def section_funnel(df):
    _banner("2. SUCCESSIVE HALVING FUNNEL")
    rounds = sorted(df["search_round"].unique())
    print(f"  {'Round':>5s}  {'Configs':>8s}  {'Rows':>6s}  {'Epochs':>7s}  "
          f"{'Best OWA':>9s}  {'Worst OWA':>10s}  {'Median OWA':>11s}")
    print(f"  {'-'*65}")
    prev_configs = None
    for r in rounds:
        rdf = df[df["search_round"] == r]
        n_cfg = rdf["config_name"].nunique()
        config_owas = rdf.groupby("config_name")["owa"].mean()
        max_ep = rdf["epochs_trained"].max()
        ratio = ""
        if prev_configs is not None:
            ratio = f"  (kept {n_cfg}/{prev_configs} = {n_cfg/prev_configs:.0%})"
        print(f"  {r:5d}  {n_cfg:8d}  {len(rdf):6d}  {max_ep:7d}  "
              f"{config_owas.min():9.4f}  {config_owas.max():10.4f}  "
              f"{config_owas.median():11.4f}{ratio}")
        prev_configs = n_cfg
    print()
    print("  Interpretation: Each round doubles epochs and prunes ~50% of configs.")
    print("  The funnel narrows from the full factorial grid to the best ~25% of configs.")
    print()


# ---------------------------------------------------------------------------
# 3. Per-Round Leaderboards
# ---------------------------------------------------------------------------

def section_leaderboards(df):
    _banner("3. PER-ROUND LEADERBOARDS")

    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        for pass_name in ["False", "forecast"]:
            pass_label = "baseline" if pass_name == "False" else "active_g=forecast"
            pdf = rdf[rdf["active_g_cfg"] == pass_name]
            if pdf.empty:
                continue

            agg = (
                pdf.groupby("config_name")
                .agg(
                    mean_owa=("owa", "mean"),
                    std_owa=("owa", "std"),
                    mean_smape=("smape", "mean"),
                    mean_mase=("mase", "mean"),
                    n_params=("n_params", "first"),
                    mean_time=("training_time_seconds", "mean"),
                )
                .sort_values("mean_owa")
            )
            n_show = 20 if r == 1 else len(agg)
            n_cfg = len(agg)

            print(f"\n── Round {r} / {pass_label} ({n_cfg} configs, "
                  f"showing {'top ' + str(n_show) if n_show < n_cfg else 'all'}) ──")
            print(f"  {'#':>3s}  {'Config':<40s} {'OWA':>7s} {'±':>6s} "
                  f"{'sMAPE':>7s} {'MASE':>7s} {'Params':>10s} {'Time':>6s}")
            print(f"  {'-'*85}")

            for rank, (name, row) in enumerate(agg.head(n_show).iterrows(), 1):
                std = row.std_owa if pd.notna(row.std_owa) else 0
                print(f"  {rank:3d}  {name:<40s} {row.mean_owa:7.4f} {std:6.4f} "
                      f"{row.mean_smape:7.2f} {row.mean_mase:7.3f} "
                      f"{row.n_params:10,.0f} {row.mean_time:5.1f}s")

            if n_show < n_cfg:
                worst = agg.tail(3)
                print(f"  {'...'}")
                for rank, (name, row) in zip(
                    range(n_cfg - 2, n_cfg + 1), worst.iterrows()
                ):
                    std = row.std_owa if pd.notna(row.std_owa) else 0
                    print(f"  {rank:3d}  {name:<40s} {row.mean_owa:7.4f} {std:6.4f} "
                          f"{row.mean_smape:7.2f} {row.mean_mase:7.3f} "
                          f"{row.n_params:10,.0f} {row.mean_time:5.1f}s")
    print()


# ---------------------------------------------------------------------------
# 4. Factor Marginals (Round 1 Only)
# ---------------------------------------------------------------------------

def section_marginals(df):
    _banner("4. FACTOR MARGINALS (Round 1 — balanced factorial grid)")
    r1 = df[df["search_round"] == 1]
    print(f"  Using Round 1 only ({len(r1)} rows, {r1['config_name'].nunique()} configs)")
    print(f"  This is the only round with a balanced design for unbiased marginals.\n")

    factors = [
        ("wavelet", "Wavelet Family"),
        ("bd_label", "Basis Dim Label"),
        ("ttd", "Trend Thetas Dim"),
        ("active_g_cfg", "active_g Config"),
    ]
    for factor, label in factors:
        print(f"── {label} ──")
        agg = r1.groupby(factor)[["owa", "smape", "mase"]].agg(["mean", "std", "count"])
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        for m in ["owa", "smape", "mase"]:
            agg[f"{m}_disp"] = agg.apply(
                lambda row: f"{row[f'{m}_mean']:.4f} +/- {row[f'{m}_std']:.4f}", axis=1
            )
        disp = agg[[f"{m}_disp" for m in ["owa", "smape", "mase"]] + ["owa_count"]].rename(
            columns={"owa_count": "n_runs", "owa_disp": "OWA", "smape_disp": "sMAPE", "mase_disp": "MASE"}
        )
        print(disp.sort_values("OWA").to_string())
        print()


# ---------------------------------------------------------------------------
# 5. Two-Way Interactions
# ---------------------------------------------------------------------------

def section_interactions(df):
    _banner("5. TWO-WAY INTERACTIONS (Round 1 — mean values)")
    r1 = df[df["search_round"] == 1]

    pairs = [
        ("wavelet", "bd_label", "Wavelet x Basis Dim Label"),
        ("wavelet", "ttd", "Wavelet x Trend Thetas Dim"),
        ("bd_label", "active_g_cfg", "Basis Dim Label x active_g"),
        ("bd_label", "ttd", "Basis Dim Label x Trend Thetas Dim"),
        ("wavelet", "active_g_cfg", "Wavelet x active_g"),
    ]

    for row_fac, col_fac, title in pairs:
        for metric in ["owa", "smape"]:
            print(f"\n── {title} (mean {metric.upper()}) ──")
            pivot = r1.pivot_table(
                values=metric, index=row_fac, columns=col_fac, aggfunc="mean"
            )
            print(pivot.to_string(float_format="{:.4f}".format))
    print()


# ---------------------------------------------------------------------------
# 6. Variance Decomposition
# ---------------------------------------------------------------------------

def section_variance_decomposition(df):
    _banner("6. VARIANCE DECOMPOSITION (Round 1)")
    r1 = df[df["search_round"] == 1]
    target = "owa"
    grand_mean = r1[target].mean()
    ss_total = ((r1[target] - grand_mean) ** 2).sum()

    factors = ["wavelet", "bd_label", "ttd", "active_g_cfg"]

    print(f"  Grand mean OWA: {grand_mean:.4f}")
    print(f"  SS_total:       {ss_total:.4f}\n")

    print("── Eta-squared (proportion of variance explained) ──")
    print(f"  {'Factor':<18s} {'eta-sq':>8s} {'%':>7s}  {'Kruskal H':>10s} {'p-value':>10s}  {'Sig':>4s}")
    print(f"  {'-'*65}")

    eta_results = []
    for factor in factors:
        group_means = r1.groupby(factor)[target].transform("mean")
        ss_between = ((group_means - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        groups = [g[target].values for _, g in r1.groupby(factor)]
        h_stat, p_val = stats.kruskal(*groups)
        sig = _sig(p_val)
        eta_results.append((factor, eta_sq, h_stat, p_val, sig))
        print(f"  {factor:<18s} {eta_sq:8.4f} {eta_sq*100:6.1f}%  "
              f"{h_stat:10.3f} {p_val:10.6f}  {sig:>4s}")

    # Rank by eta-squared
    eta_results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Ranking: {' > '.join(f'{f} ({e*100:.1f}%)' for f, e, *_ in eta_results)}")
    print()


# ---------------------------------------------------------------------------
# 7. Cross-Metric Covariance
# ---------------------------------------------------------------------------

def section_cross_metric(df):
    _banner("7. CROSS-METRIC CORRELATION")
    cols = ["smape", "mase", "owa", "best_val_loss", "training_time_seconds", "n_params"]
    avail = [c for c in cols if c in df.columns and df[c].notna().sum() > 10]

    print("── Pearson correlation ──")
    print(df[avail].corr().to_string(float_format="{:.3f}".format))
    print()

    print("── Spearman rank correlation ──")
    print(df[avail].corr(method="spearman").to_string(float_format="{:.3f}".format))
    print()

    # Key observations
    pearson = df[avail].corr()
    if "smape" in pearson.columns and "mase" in pearson.columns:
        print(f"  sMAPE-MASE Pearson: {pearson.loc['smape', 'mase']:.3f}")
    if "owa" in pearson.columns and "best_val_loss" in pearson.columns:
        print(f"  OWA-val_loss Pearson: {pearson.loc['owa', 'best_val_loss']:.3f}")
    print()


# ---------------------------------------------------------------------------
# 8. Stability Analysis
# ---------------------------------------------------------------------------

def section_stability(df):
    _banner("8. STABILITY ANALYSIS (OWA spread across seeds)")

    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        print(f"\n── Round {r} ──")

        for pass_name in ["False", "forecast"]:
            pass_label = "baseline" if pass_name == "False" else "active_g=forecast"
            pdf = rdf[rdf["active_g_cfg"] == pass_name]
            if pdf.empty:
                continue
            spread = (
                pdf.groupby("config_name")["owa"]
                .agg(["mean", "std", "min", "max"])
                .assign(range=lambda x: x["max"] - x["min"])
            )
            print(f"\n  {pass_label}:")
            print(f"    Mean spread (max-min):  {spread['range'].mean():.4f}")
            print(f"    Mean std:               {spread['std'].mean():.4f}")
            print(f"    Max spread:             {spread['range'].max():.4f}  "
                  f"({spread['range'].idxmax()})")

            # Top 5 most stable
            most_stable = spread.sort_values("range").head(5)
            print(f"    Top 5 most stable:")
            for name, row in most_stable.iterrows():
                print(f"      {name:<40s} range={row['range']:.4f}  "
                      f"OWA={row['mean']:.4f}")

            # Top 5 least stable
            least_stable = spread.sort_values("range", ascending=False).head(5)
            print(f"    Top 5 least stable:")
            for name, row in least_stable.iterrows():
                print(f"      {name:<40s} range={row['range']:.4f}  "
                      f"OWA={row['mean']:.4f}")

    # R3 pass comparison
    r3 = df[df["search_round"] == 3]
    if not r3.empty:
        print(f"\n── R3 pass comparison ──")
        for pass_name, label in [("False", "baseline"), ("forecast", "active_g=forecast")]:
            pdf = r3[r3["active_g_cfg"] == pass_name]
            spread = pdf.groupby("config_name")["owa"].agg(["std", "mean"])
            print(f"  {label:20s}: mean_std={spread['std'].mean():.4f}, "
                  f"mean_OWA={spread['mean'].mean():.4f}")
    print()


# ---------------------------------------------------------------------------
# 9. Round-over-Round Progression
# ---------------------------------------------------------------------------

def section_progression(df):
    _banner("9. ROUND-OVER-ROUND PROGRESSION (surviving configs)")
    rounds = sorted(df["search_round"].unique())
    if len(rounds) < 2:
        print("  (need >= 2 rounds)")
        return

    # Configs in final round
    last_round = rounds[-1]
    final_configs = set(df[df["search_round"] == last_round]["config_name"].unique())
    print(f"  Tracking {len(final_configs)} configs that survived to Round {last_round}")
    print()

    # Use baseline pass for consistency
    base = df[df["active_g_cfg"] == "False"]
    medians = {}
    for r in rounds:
        rdf = base[(base["search_round"] == r) & (base["config_name"].isin(final_configs))]
        if rdf.empty:
            continue
        medians[r] = rdf.groupby("config_name")["owa"].mean()

    prog = pd.DataFrame(medians)
    prog.columns = [f"R{r}" for r in prog.columns]
    if len(prog.columns) >= 2:
        first_col, last_col = prog.columns[0], prog.columns[-1]
        prog["Delta"] = prog[last_col] - prog[first_col]
        prog["Delta%"] = (prog["Delta"] / prog[first_col] * 100).round(2)
    prog = prog.sort_values(prog.columns[-2] if "Delta" not in prog.columns else prog.columns[-3])

    print(prog.round(4).to_string())
    print()

    # Summary
    if "Delta" in prog.columns:
        improved = (prog["Delta"] < 0).sum()
        worsened = (prog["Delta"] > 0).sum()
        print(f"  Improved R1->R3: {improved}/{len(prog)}  "
              f"(mean Delta: {prog['Delta'].mean():.4f})")
        print(f"  Worsened R1->R3: {worsened}/{len(prog)}")

    # Per-transition average
    cols = [c for c in prog.columns if c.startswith("R")]
    for i in range(len(cols) - 1):
        delta = prog[cols[i+1]] - prog[cols[i]]
        print(f"  {cols[i]}->{cols[i+1]}: mean change = {delta.mean():.4f}")
    print()


# ---------------------------------------------------------------------------
# 10. Elimination Analysis
# ---------------------------------------------------------------------------

def section_elimination(df):
    _banner("10. ELIMINATION ANALYSIS")
    base = df[df["active_g_cfg"] == "False"]  # Use baseline pass for factor analysis

    rounds = sorted(df["search_round"].unique())
    for r_idx in range(len(rounds) - 1):
        r_curr, r_next = rounds[r_idx], rounds[r_idx + 1]
        curr_configs = set(base[base["search_round"] == r_curr]["config_name"].unique())
        next_configs = set(base[base["search_round"] == r_next]["config_name"].unique())
        dropped = curr_configs - next_configs
        survived = curr_configs & next_configs

        print(f"\n── R{r_curr} -> R{r_next}: {len(dropped)} dropped, {len(survived)} survived ──")

        # Factor distribution of dropped vs survived
        r_data = base[base["search_round"] == r_curr].drop_duplicates("config_name")
        for factor in ["wavelet", "bd_label", "ttd"]:
            surv_data = r_data[r_data["config_name"].isin(survived)]
            drop_data = r_data[r_data["config_name"].isin(dropped)]

            surv_counts = surv_data[factor].value_counts().sort_index()
            drop_counts = drop_data[factor].value_counts().sort_index()
            total_counts = r_data[factor].value_counts().sort_index()

            print(f"\n  {factor} elimination rates:")
            print(f"    {'Level':<18s} {'Total':>6s} {'Survived':>9s} {'Dropped':>8s} {'Elim%':>6s}")
            print(f"    {'-'*50}")
            for level in total_counts.index:
                t = total_counts.get(level, 0)
                s = surv_counts.get(level, 0)
                d = drop_counts.get(level, 0)
                pct = d / t * 100 if t > 0 else 0
                print(f"    {str(level):<18s} {t:6d} {s:9d} {d:8d} {pct:5.0f}%")

    # Dedicated: ttd=5 near-total elimination
    print(f"\n── ttd=5 near-total elimination ──")
    r1_base = base[base["search_round"] == 1]
    r3_configs = set(base[base["search_round"] == 3]["config_name"].unique())
    r3_data = r1_base[r1_base["config_name"].isin(r3_configs)]
    ttd5_in_r3 = r3_data[r3_data["ttd"] == 5]["config_name"].nunique()
    ttd3_in_r3 = r3_data[r3_data["ttd"] == 3]["config_name"].nunique()
    r1_ttd5 = r1_base[r1_base["ttd"] == 5]["config_name"].nunique()
    r1_ttd3 = r1_base[r1_base["ttd"] == 3]["config_name"].nunique()
    print(f"  R1: ttd=3 had {r1_ttd3} configs, ttd=5 had {r1_ttd5}")
    print(f"  R3: ttd=3 has {ttd3_in_r3} configs, ttd=5 has {ttd5_in_r3}")
    print(f"  ttd=5 survival rate: {ttd5_in_r3}/{r1_ttd5} = "
          f"{ttd5_in_r3/r1_ttd5*100:.0f}%" if r1_ttd5 > 0 else "  (no ttd=5)")

    # Mann-Whitney U test: ttd=3 vs ttd=5 in R1
    ttd3_owa = r1_base[r1_base["ttd"] == 3]["owa"].dropna()
    ttd5_owa = r1_base[r1_base["ttd"] == 5]["owa"].dropna()
    if len(ttd3_owa) > 0 and len(ttd5_owa) > 0:
        u_stat, p_val = stats.mannwhitneyu(ttd3_owa, ttd5_owa, alternative="two-sided")
        print(f"  Mann-Whitney U (ttd=3 vs ttd=5, R1 OWA): U={u_stat:.1f}, "
              f"p={p_val:.6f} {_sig(p_val)}")
        print(f"  ttd=3 mean OWA: {ttd3_owa.mean():.4f}, ttd=5 mean OWA: {ttd5_owa.mean():.4f}")

    # Dedicated: Coif10 family — entirely eliminated by R3
    print(f"\n── Coif10 elimination ──")
    coif10_data = base[base["wavelet"] == "Coif10"]
    if not coif10_data.empty:
        last_round_coif10 = coif10_data["search_round"].max()
        c10_r1 = coif10_data[coif10_data["search_round"] == 1]
        print(f"  Coif10 last appeared in Round {last_round_coif10}")
        print(f"  R1 mean OWA: {c10_r1['owa'].mean():.4f} "
              f"(rank among families: see marginals)")
        in_r3 = coif10_data[coif10_data["search_round"] == 3]["config_name"].nunique()
        print(f"  Configs in R3: {in_r3} (entirely eliminated: {'Yes' if in_r3 == 0 else 'No'})")
    else:
        print("  Coif10 not found in data.")

    # Dedicated: DB5-DB20 higher-order attrition
    print(f"\n── Higher-order Daubechies attrition ──")
    db_families = [w for w in sorted(df["wavelet"].unique()) if w.startswith("DB")]
    for wv in db_families:
        wv_data = base[base["wavelet"] == wv]
        r1_n = wv_data[wv_data["search_round"] == 1]["config_name"].nunique()
        r3_n = wv_data[wv_data["search_round"] == 3]["config_name"].nunique()
        r3_mean = ""
        if r3_n > 0:
            r3_sub = wv_data[wv_data["search_round"] == 3]
            r3_mean = f"  R3 mean OWA: {r3_sub['owa'].mean():.4f}"
        print(f"  {wv:<8s}: R1={r1_n} configs -> R3={r3_n} configs  "
              f"(survival {r3_n/r1_n*100:.0f}%){r3_mean}" if r1_n > 0 else f"  {wv}: no R1 data")
    print()


# ---------------------------------------------------------------------------
# 11. Baseline Comparisons
# ---------------------------------------------------------------------------

def section_baseline_comparisons(df):
    _banner("11. BASELINE COMPARISONS")
    block_df, ae_df = load_baselines()

    # Top 10 R3 configs (mean across both passes)
    r3 = df[df["search_round"] == 3]
    r3_agg = (
        r3.groupby(["config_name", "active_g_cfg"])
        .agg(mean_owa=("owa", "mean"), std_owa=("owa", "std"),
             mean_smape=("smape", "mean"), n_params=("n_params", "first"))
        .reset_index()
    )

    for pass_name, label in [("False", "baseline"), ("forecast", "active_g=forecast")]:
        pass_agg = r3_agg[r3_agg["active_g_cfg"] == pass_name].sort_values("mean_owa").head(10)

        print(f"\n── Top 10 R3 configs ({label}) vs baselines ──")
        print(f"  {'Config':<40s} {'OWA':>7s} {'+-':>6s} {'sMAPE':>7s} "
              f"{'D I+G':>7s} {'D G':>7s} {'D AE+T':>7s} {'Params':>10s}")
        print(f"  {'-'*92}")

        for _, row in pass_agg.iterrows():
            d_ig = row.mean_owa - BASELINES["NBEATS-I+G"]["owa"]
            d_g = row.mean_owa - BASELINES["NBEATS-G"]["owa"]
            d_aet = row.mean_owa - BASELINES["AE+Trend"]["owa"]
            std = row.std_owa if pd.notna(row.std_owa) else 0
            print(f"  {row.config_name:<40s} {row.mean_owa:7.4f} {std:6.4f} "
                  f"{row.mean_smape:7.2f} {d_ig:+7.4f} {d_g:+7.4f} {d_aet:+7.4f} "
                  f"{row.n_params:10,.0f}")

        # Baselines reference
        print(f"\n  Reference baselines:")
        for name, vals in BASELINES.items():
            print(f"    {name:<16s}: OWA={vals['owa']:.4f} +/- {vals['owa_std']:.3f}  "
                  f"sMAPE={vals['smape']:.2f}  Params={vals['params']:,}")

    # Mann-Whitney U tests using raw data from baseline CSVs
    print(f"\n── Statistical tests (Mann-Whitney U, OWA) ──")
    r3_base_owa = r3[r3["active_g_cfg"] == "False"]
    r3_ag_owa = r3[r3["active_g_cfg"] == "forecast"]

    # Best R3 configs for each pass
    for pass_name, label, pass_df in [
        ("False", "baseline", r3_base_owa),
        ("forecast", "active_g=forecast", r3_ag_owa),
    ]:
        best_config = pass_df.groupby("config_name")["owa"].mean().idxmin()
        best_owa_vals = pass_df[pass_df["config_name"] == best_config]["owa"].values
        print(f"\n  Best R3 {label}: {best_config} (n={len(best_owa_vals)})")

        if block_df is not None:
            for baseline_name in ["NBEATS-I+G", "NBEATS-G", "NBEATS-I", "GenericAE", "AutoEncoder"]:
                bl_vals = block_df[block_df["config_name"] == baseline_name]["owa"].dropna().values
                if len(bl_vals) > 0:
                    u, p = stats.mannwhitneyu(best_owa_vals, bl_vals, alternative="two-sided")
                    print(f"    vs {baseline_name:<16s}: U={u:6.1f}  p={p:.4f}  {_sig(p)}  "
                          f"(baseline mean={bl_vals.mean():.4f}, n={len(bl_vals)})")

        if ae_df is not None:
            ae_best = ae_df.groupby("config_name")["owa"].mean().idxmin()
            ae_vals = ae_df[ae_df["config_name"] == ae_best]["owa"].dropna().values
            if len(ae_vals) > 0:
                u, p = stats.mannwhitneyu(best_owa_vals, ae_vals, alternative="two-sided")
                print(f"    vs AE+Trend best : U={u:6.1f}  p={p:.4f}  {_sig(p)}  "
                      f"(AE mean={ae_vals.mean():.4f}, n={len(ae_vals)})")
    print()


# ---------------------------------------------------------------------------
# 12. Parameter Efficiency
# ---------------------------------------------------------------------------

def section_param_efficiency(df):
    _banner("12. PARAMETER EFFICIENCY")
    params_range = df["n_params"].agg(["min", "max"])
    print(f"  WS3 param range: {params_range['min']:,.0f} - {params_range['max']:,.0f}")
    print(f"  Reduction vs NBEATS-G (24.7M):  "
          f"{(1 - params_range['max']/24_700_000)*100:.0f}%-{(1 - params_range['min']/24_700_000)*100:.0f}%")
    print(f"  Reduction vs NBEATS-I+G (35.9M): "
          f"{(1 - params_range['max']/35_900_000)*100:.0f}%-{(1 - params_range['min']/35_900_000)*100:.0f}%")
    print()

    # OWA per million params
    r3 = df[df["search_round"] == 3]
    r3_agg = (
        r3.groupby(["config_name", "active_g_cfg"])
        .agg(mean_owa=("owa", "mean"), n_params=("n_params", "first"))
        .reset_index()
    )
    r3_agg["owa_per_Mparam"] = r3_agg["mean_owa"] / (r3_agg["n_params"] / 1_000_000)

    print("── OWA per million params (R3, lower is better) ──")
    print(f"  {'Config':<40s} {'Pass':<10s} {'OWA':>7s} {'Params':>10s} {'OWA/Mparam':>11s}")
    print(f"  {'-'*82}")

    for _, row in r3_agg.sort_values("owa_per_Mparam").head(10).iterrows():
        label = "baseline" if row.active_g_cfg == "False" else "activeG"
        print(f"  {row.config_name:<40s} {label:<10s} {row.mean_owa:7.4f} "
              f"{row.n_params:10,.0f} {row.owa_per_Mparam:11.4f}")

    # Comparison to AE baselines
    print(f"\n── Baseline OWA/Mparam comparison ──")
    for name, vals in BASELINES.items():
        eff = vals["owa"] / (vals["params"] / 1_000_000)
        print(f"  {name:<16s}: OWA/Mparam = {eff:.4f}  "
              f"(OWA={vals['owa']:.4f}, {vals['params']/1e6:.1f}M params)")
    print()


# ---------------------------------------------------------------------------
# 13. Convergence Diagnostics
# ---------------------------------------------------------------------------

def section_convergence(df):
    _banner("13. CONVERGENCE DIAGNOSTICS")

    # Per-round epoch distribution
    print("── Epochs trained per round ──")
    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        ep = rdf["epochs_trained"]
        es_count = (rdf["stopping_reason"] == "EARLY_STOPPED").sum()
        max_ep_count = (rdf["stopping_reason"] == "MAX_EPOCHS").sum()
        print(f"  R{r}: epochs={ep.min()}-{ep.max()} (mean={ep.mean():.1f}), "
              f"early_stopped={es_count}, max_epochs={max_ep_count}")
    print()

    # R3 detailed: loss ratio by factor
    r3 = df[df["search_round"] == 3]
    if not r3.empty and "loss_ratio" in r3.columns:
        print("── R3 loss ratio (final_val/best_val) by factor ──")
        for factor in ["wavelet", "bd_label", "ttd", "active_g_cfg"]:
            if factor in r3.columns:
                lr = r3.groupby(factor)["loss_ratio"].agg(["mean", "std", "max"])
                print(f"\n  {factor}:")
                print(f"  {lr.to_string()}")
        print()

    # best_epoch distribution in R3
    if not r3.empty and "best_epoch" in r3.columns:
        print("── R3 best_epoch distribution ──")
        be = r3["best_epoch"]
        print(f"  mean={be.mean():.1f}, std={be.std():.1f}, "
              f"min={be.min()}, max={be.max()}, median={be.median():.0f}")
        print()

    # Val loss curve analysis in R3
    if "val_loss_curve" in r3.columns:
        print("── R3 val_loss_curve analysis ──")
        improvements = []
        non_monotone = 0
        total_parsed = 0
        for _, row in r3.iterrows():
            try:
                curve = ast.literal_eval(row["val_loss_curve"])
                curve = [float(v) for v in curve]
                total_parsed += 1
                if len(curve) >= 2:
                    improvements.append((curve[0] - curve[-1]) / curve[0] * 100)
                    # Check non-monotonicity
                    diffs = np.diff(curve)
                    if np.any(diffs > 0):
                        non_monotone += 1
            except (ValueError, SyntaxError, TypeError):
                pass
        if improvements:
            print(f"  Parsed {total_parsed} curves")
            print(f"  Relative improvement (first->last): "
                  f"mean={np.mean(improvements):.1f}%, std={np.std(improvements):.1f}%")
            print(f"  Non-monotone curves: {non_monotone}/{total_parsed} "
                  f"({non_monotone/total_parsed*100:.0f}%)")
    print()


# ---------------------------------------------------------------------------
# 14. Final Verdict & Interpretation
# ---------------------------------------------------------------------------

def section_verdict(df):
    _banner("14. FINAL VERDICT & INTERPRETATION", char="*")

    r3 = df[df["search_round"] == 3]
    r1 = df[df["search_round"] == 1]

    # Best overall configs
    overall_agg = (
        r3.groupby(["config_name", "active_g_cfg"])
        .agg(
            mean_owa=("owa", "mean"),
            std_owa=("owa", "std"),
            mean_smape=("smape", "mean"),
            mean_mase=("mase", "mean"),
            n_params=("n_params", "first"),
        )
        .reset_index()
        .sort_values("mean_owa")
    )

    print("\n── KEY FINDINGS ──\n")

    # 1. Factor importance
    grand_mean = r1["owa"].mean()
    ss_total = ((r1["owa"] - grand_mean) ** 2).sum()
    eta_results = {}
    for factor in ["wavelet", "bd_label", "ttd", "active_g_cfg"]:
        r1_base = r1[r1["active_g_cfg"] == "False"] if factor == "active_g_cfg" else r1
        if factor == "active_g_cfg":
            gm = r1["owa"].mean()
            sst = ((r1["owa"] - gm) ** 2).sum()
        else:
            gm, sst = grand_mean, ss_total
        group_means = r1.groupby(factor)["owa"].transform("mean") if sst > 0 else 0
        if sst > 0:
            ss_b = ((group_means - gm) ** 2).sum()
            eta_results[factor] = ss_b / sst
        else:
            eta_results[factor] = 0
    top_factor = max(eta_results, key=eta_results.get)
    print(f"  1. DOMINANT FACTOR: {top_factor} (eta-sq={eta_results[top_factor]:.3f}, "
          f"{eta_results[top_factor]*100:.1f}% of variance)")

    # 2. ttd finding
    ttd3_mean = r1[r1["ttd"] == 3]["owa"].mean()
    ttd5_mean = r1[r1["ttd"] == 5]["owa"].mean()
    r3_ttd5 = r3[r3["ttd"] == 5]["config_name"].nunique()
    print(f"\n  2. TREND THETAS DIM: ttd=3 ({ttd3_mean:.4f}) decisively beats ttd=5 "
          f"({ttd5_mean:.4f}). Only {r3_ttd5} ttd=5 config(s) survived to R3.")

    # 3. Wavelet family ranking
    r1_base = r1[r1["active_g_cfg"] == "False"]
    wv_means = r1_base.groupby("wavelet")["owa"].mean().sort_values()
    print(f"\n  3. WAVELET RANKING (R1 mean OWA):")
    for i, (wv, val) in enumerate(wv_means.items(), 1):
        r3_present = "R3" if wv in r3["wavelet"].values else "eliminated"
        print(f"     {i:2d}. {wv:<10s}: {val:.4f}  ({r3_present})")

    # 4. Basis dim
    bd_means = r1_base.groupby("bd_label")["owa"].mean().sort_values()
    print(f"\n  4. BASIS DIM LABEL RANKING: {' < '.join(f'{k}({v:.4f})' for k, v in bd_means.items())}")

    # 5. Baseline comparison verdict
    best_row = overall_agg.iloc[0]
    best_owa = best_row.mean_owa
    best_std = best_row.std_owa if pd.notna(best_row.std_owa) else 0
    print(f"\n  5. BEST CONFIG: {best_row.config_name} ({best_row.active_g_cfg})")
    print(f"     OWA: {best_owa:.4f} +/- {best_std:.4f}, "
          f"sMAPE: {best_row.mean_smape:.2f}, Params: {best_row.n_params:,.0f}")

    beat_count = 0
    for name, vals in BASELINES.items():
        delta = best_owa - vals["owa"]
        verdict = "BEATS" if delta < -0.001 else "MATCHES" if abs(delta) < 0.005 else "LOSES TO"
        if delta < -0.001:
            beat_count += 1
        print(f"     vs {name:<16s}: {delta:+.4f}  ({verdict})")

    # 6. Parameter efficiency
    print(f"\n  6. PARAMETER EFFICIENCY: ~{best_row.n_params/1e6:.1f}M params = "
          f"{(1 - best_row.n_params/24_700_000)*100:.0f}% fewer than NBEATS-G, "
          f"{(1 - best_row.n_params/35_900_000)*100:.0f}% fewer than NBEATS-I+G")

    # 7. Top recommended configs table
    print(f"\n── RECOMMENDED CONFIGS ──\n")
    print(f"  {'#':>3s}  {'Config':<40s} {'Pass':<10s} {'OWA':>7s} {'+-':>6s} "
          f"{'sMAPE':>7s} {'MASE':>7s} {'Params':>10s}")
    print(f"  {'-'*90}")
    for rank, (_, row) in enumerate(overall_agg.head(10).iterrows(), 1):
        label = "baseline" if row.active_g_cfg == "False" else "activeG"
        std = row.std_owa if pd.notna(row.std_owa) else 0
        print(f"  {rank:3d}  {row.config_name:<40s} {label:<10s} {row.mean_owa:7.4f} "
              f"{std:6.4f} {row.mean_smape:7.2f} {row.mean_mase:7.3f} "
              f"{row.n_params:10,.0f}")

    # Overall verdict
    ig_owa = BASELINES["NBEATS-I+G"]["owa"]
    aet_owa = BASELINES["AE+Trend"]["owa"]
    if best_owa < ig_owa - 0.005:
        overall = "WaveletV3+Trend BEATS the NBEATS-I+G baseline with ~80% fewer params."
    elif best_owa < ig_owa + 0.005:
        overall = "WaveletV3+Trend MATCHES NBEATS-I+G with ~80% fewer params."
    else:
        overall = "WaveletV3+Trend does NOT match NBEATS-I+G but may beat simpler baselines."

    print(f"\n  OVERALL VERDICT: {overall}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()

    section_overview(df)
    section_funnel(df)
    section_leaderboards(df)
    section_marginals(df)
    section_interactions(df)
    section_variance_decomposition(df)
    section_cross_metric(df)
    section_stability(df)
    section_progression(df)
    section_elimination(df)
    section_baseline_comparisons(df)
    section_param_efficiency(df)
    section_convergence(df)
    section_verdict(df)

    _banner("ANALYSIS COMPLETE")
    print(f"  Data: {CSV_PATH}")
    print(f"  Rows: {len(df)}")
    print(f"  Sections: 14")
    print()


if __name__ == "__main__":
    main()
