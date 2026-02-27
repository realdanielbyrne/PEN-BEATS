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

try:
    from llm_commentary import generate_commentary
    _LLM = True
except ImportError:
    _LLM = False

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

def _section(title):
    """Emit a Markdown section heading (## level)."""
    print()
    print(f"## {title}")
    print()


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
    _section("1. Overview & Data Summary")
    n_wavelets = df["wavelet"].nunique()
    n_bd = df["bd_label"].nunique()
    n_ttd = df["ttd"].nunique()
    n_ag = df["active_g_cfg"].nunique()
    print(f"- **CSV:** `{CSV_PATH}`")
    print(f"- **Total rows:** {len(df)}")
    print(f"- **Unique configs:** {df['config_name'].nunique()}")
    print(f"- **Search rounds:** {sorted(df['search_round'].unique())}")
    print(f"- **Config space:** {n_wavelets} wavelets × {n_bd} bd_labels × "
          f"{n_ttd} ttd × {n_ag} active_g = {n_wavelets * n_bd * n_ttd} base configs")
    print(f"- **Wavelet families:** {sorted(df['wavelet'].unique())}")
    print(f"- **basis_dim labels:** {sorted(df['bd_label'].unique())}")
    print(f"- **ttd values:** {sorted(df['ttd'].unique())}")
    print(f"- **active_g_cfg:** {sorted(df['active_g_cfg'].unique())}")
    print()

    print("### Per-Round Breakdown\n")
    round_rows = []
    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        ep_range = f"{rdf['epochs_trained'].min()}-{rdf['epochs_trained'].max()}"
        passes = ", ".join(sorted(rdf["active_g_cfg"].unique()))
        round_rows.append({"Round": r, "Configs": rdf['config_name'].nunique(),
                           "Rows": len(rdf), "Epochs": ep_range, "Passes": passes})
    print(pd.DataFrame(round_rows).to_markdown(index=False))
    print()

    print("### Global Metric Statistics\n")
    print(df[METRICS].describe().T.to_markdown(floatfmt=".4f"))
    print()


# ---------------------------------------------------------------------------
# 2. Successive Halving Funnel
# ---------------------------------------------------------------------------

def section_funnel(df):
    _section("2. Successive Halving Funnel")
    print("Each round doubles the epoch budget and prunes ~50% of configurations. "
          "The funnel narrows from the full factorial grid to the best ~25%.\n")
    rounds = sorted(df["search_round"].unique())
    rows = []
    prev_configs = None
    for r in rounds:
        rdf = df[df["search_round"] == r]
        n_cfg = rdf["config_name"].nunique()
        config_owas = rdf.groupby("config_name")["owa"].mean()
        max_ep = rdf["epochs_trained"].max()
        kept = f"{n_cfg}/{prev_configs} ({n_cfg/prev_configs:.0%})" if prev_configs else "—"
        rows.append({"Round": r, "Configs": n_cfg, "Rows": len(rdf), "Epochs": max_ep,
                      "Best OWA": f"{config_owas.min():.4f}",
                      "Worst OWA": f"{config_owas.max():.4f}",
                      "Median OWA": f"{config_owas.median():.4f}", "Kept": kept})
        prev_configs = n_cfg
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


# ---------------------------------------------------------------------------
# 3. Per-Round Leaderboards
# ---------------------------------------------------------------------------

def section_leaderboards(df):
    _section("3. Per-Round Leaderboards")

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

            show_label = f"top {n_show}" if n_show < n_cfg else "all"
            print(f"\n### Round {r} / {pass_label} ({n_cfg} configs, showing {show_label})\n")

            tbl = agg.head(n_show).reset_index()
            tbl = tbl.rename(columns={
                "config_name": "Config", "mean_owa": "OWA", "std_owa": "±",
                "mean_smape": "sMAPE", "mean_mase": "MASE",
                "n_params": "Params", "mean_time": "Time(s)"})
            tbl["±"] = tbl["±"].fillna(0)
            tbl.insert(0, "#", range(1, len(tbl) + 1))
            print(tbl.to_markdown(index=False, floatfmt=(".0f", "", ".4f", ".4f", ".2f", ".3f", ",.0f", ".1f")))

            if n_show < n_cfg:
                print("\n*… bottom 3:*\n")
                worst = agg.tail(3).reset_index()
                worst = worst.rename(columns={
                    "config_name": "Config", "mean_owa": "OWA", "std_owa": "±",
                    "mean_smape": "sMAPE", "mean_mase": "MASE",
                    "n_params": "Params", "mean_time": "Time(s)"})
                worst["±"] = worst["±"].fillna(0)
                worst.insert(0, "#", range(n_cfg - 2, n_cfg + 1))
                print(worst.to_markdown(index=False, floatfmt=(".0f", "", ".4f", ".4f", ".2f", ".3f", ",.0f", ".1f")))
    print()


# ---------------------------------------------------------------------------
# 4. Factor Marginals (Round 1 Only)
# ---------------------------------------------------------------------------

def section_marginals(df):
    _section("4. Factor Marginals (Round 1 — Balanced Factorial Grid)")
    r1 = df[df["search_round"] == 1]
    print(f"Using Round 1 only ({len(r1)} rows, {r1['config_name'].nunique()} configs). "
          f"This is the only round with a balanced design for unbiased marginals.\n")

    factors = [
        ("wavelet", "Wavelet Family"),
        ("bd_label", "Basis Dim Label"),
        ("ttd", "Trend Thetas Dim"),
        ("active_g_cfg", "active_g Config"),
    ]
    for factor, label in factors:
        print(f"### {label}\n")
        agg = r1.groupby(factor)[["owa", "smape", "mase"]].agg(["mean", "std", "count"])
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        for m in ["owa", "smape", "mase"]:
            agg[f"{m}_disp"] = agg.apply(
                lambda row: f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}", axis=1
            )
        disp = agg[[f"{m}_disp" for m in ["owa", "smape", "mase"]] + ["owa_count"]].rename(
            columns={"owa_count": "n_runs", "owa_disp": "OWA", "smape_disp": "sMAPE", "mase_disp": "MASE"}
        )
        print(disp.sort_values("OWA").to_markdown())
        owa_agg = agg["owa_mean"].sort_values()
        best_val = str(owa_agg.index[0])
        worst_val = str(owa_agg.index[-1])
        delta = float(owa_agg.iloc[-1] - owa_agg.iloc[0])
        llm_ctx = {
            "parameter_name": factor,
            "best_value": best_val,
            "best_owa": float(owa_agg.iloc[0]),
            "worst_value": worst_val,
            "worst_owa": float(owa_agg.iloc[-1]),
            "delta": delta,
            "all_values": [
                {"value": str(v), "med_owa": float(owa)}
                for v, owa in owa_agg.items()
            ],
        }
        llm_text = generate_commentary("hyperparameter_marginal", llm_ctx) if _LLM else None
        if llm_text:
            print(f"\n{llm_text}")
        print()


# ---------------------------------------------------------------------------
# 5. Two-Way Interactions
# ---------------------------------------------------------------------------

def section_interactions(df):
    _section("5. Two-Way Interactions (Round 1)")
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
            print(f"\n### {title} (mean {metric.upper()})\n")
            pivot = r1.pivot_table(
                values=metric, index=row_fac, columns=col_fac, aggfunc="mean"
            )
            print(pivot.to_markdown(floatfmt=".4f"))
    print()


# ---------------------------------------------------------------------------
# 6. Variance Decomposition
# ---------------------------------------------------------------------------

def section_variance_decomposition(df):
    _section("6. Variance Decomposition (Round 1)")
    r1 = df[df["search_round"] == 1]
    target = "owa"
    grand_mean = r1[target].mean()
    ss_total = ((r1[target] - grand_mean) ** 2).sum()

    factors = ["wavelet", "bd_label", "ttd", "active_g_cfg"]

    print(f"- **Grand mean OWA:** {grand_mean:.4f}")
    print(f"- **SS_total:** {ss_total:.4f}\n")

    print("### Eta-squared (proportion of variance explained)\n")

    eta_results = []
    for factor in factors:
        group_means = r1.groupby(factor)[target].transform("mean")
        ss_between = ((group_means - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        groups = [g[target].values for _, g in r1.groupby(factor)]
        h_stat, p_val = stats.kruskal(*groups)
        sig = _sig(p_val)
        eta_results.append({"Factor": factor, "η²": eta_sq, "%": f"{eta_sq*100:.1f}",
                            "Kruskal H": h_stat, "p-value": p_val, "Sig": sig})

    eta_df = pd.DataFrame(eta_results)
    print(eta_df.to_markdown(index=False, floatfmt=("", ".4f", "", ".3f", ".6f", "")))

    # Rank by eta-squared
    eta_df_sorted = eta_df.sort_values("η²", ascending=False)
    ranking_parts = [f"{row['Factor']} ({row['%']}%)" for _, row in eta_df_sorted.iterrows()]
    print(f"\n**Ranking:** {' > '.join(ranking_parts)}")
    print()


# ---------------------------------------------------------------------------
# 7. Cross-Metric Covariance
# ---------------------------------------------------------------------------

def section_cross_metric(df):
    _section("7. Cross-Metric Correlation")
    cols = ["smape", "mase", "owa", "best_val_loss", "training_time_seconds", "n_params"]
    avail = [c for c in cols if c in df.columns and df[c].notna().sum() > 10]

    print("### Pearson Correlation\n")
    print(df[avail].corr().to_markdown(floatfmt=".3f"))
    print()

    print("### Spearman Rank Correlation\n")
    print(df[avail].corr(method="spearman").to_markdown(floatfmt=".3f"))
    print()

    pearson = df[avail].corr()
    if "smape" in pearson.columns and "mase" in pearson.columns:
        print(f"- **sMAPE–MASE Pearson:** {pearson.loc['smape', 'mase']:.3f}")
    if "owa" in pearson.columns and "best_val_loss" in pearson.columns:
        print(f"- **OWA–val_loss Pearson:** {pearson.loc['owa', 'best_val_loss']:.3f}")
    print()


# ---------------------------------------------------------------------------
# 8. Stability Analysis
# ---------------------------------------------------------------------------

def section_stability(df):
    _section("8. Stability Analysis (OWA Spread Across Seeds)")

    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        print(f"\n### Round {r}\n")

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
            print(f"**{pass_label}:**\n")
            print(f"- Mean spread (max−min): {spread['range'].mean():.4f}")
            print(f"- Mean std: {spread['std'].mean():.4f}")
            print(f"- Max spread: {spread['range'].max():.4f} ({spread['range'].idxmax()})\n")

            most_stable = spread.sort_values("range").head(5).reset_index()
            most_stable = most_stable.rename(columns={"config_name": "Config", "range": "Range", "mean": "OWA"})
            print("*Top 5 most stable:*\n")
            print(most_stable[["Config", "Range", "OWA"]].to_markdown(index=False, floatfmt=".4f"))
            print()

            least_stable = spread.sort_values("range", ascending=False).head(5).reset_index()
            least_stable = least_stable.rename(columns={"config_name": "Config", "range": "Range", "mean": "OWA"})
            print("*Top 5 least stable:*\n")
            print(least_stable[["Config", "Range", "OWA"]].to_markdown(index=False, floatfmt=".4f"))
            print()

    # R3 pass comparison
    r3 = df[df["search_round"] == 3]
    if not r3.empty:
        print("### R3 Pass Comparison\n")
        cmp_rows = []
        for pass_name, label in [("False", "baseline"), ("forecast", "active_g=forecast")]:
            pdf = r3[r3["active_g_cfg"] == pass_name]
            spread = pdf.groupby("config_name")["owa"].agg(["std", "mean"])
            cmp_rows.append({"Pass": label, "Mean Std": spread['std'].mean(),
                             "Mean OWA": spread['mean'].mean()})
        print(pd.DataFrame(cmp_rows).to_markdown(index=False, floatfmt=".4f"))

        # LLM stability summary for R3
        r3_base = r3[r3["active_g_cfg"] == "False"]
        if not r3_base.empty:
            r3_spread = (
                r3_base.groupby("config_name")["owa"]
                .agg(["mean", "std", "min", "max"])
                .assign(range=lambda x: x["max"] - x["min"])
            )
            llm_ctx = {
                "mean_spread": float(r3_spread["range"].mean()),
                "max_spread": float(r3_spread["range"].max()),
                "most_stable": list(r3_spread.sort_values("range").head(3).index),
                "most_volatile": list(r3_spread.sort_values("range", ascending=False).head(3).index),
            }
            llm_text = generate_commentary("stability_analysis", llm_ctx) if _LLM else None
            if llm_text:
                print(f"\n{llm_text}")
    print()


# ---------------------------------------------------------------------------
# 9. Round-over-Round Progression
# ---------------------------------------------------------------------------

def section_progression(df):
    _section("9. Round-over-Round Progression (Surviving Configs)")
    rounds = sorted(df["search_round"].unique())
    if len(rounds) < 2:
        print("  (need >= 2 rounds)")
        return

    # Configs in final round
    last_round = rounds[-1]
    final_configs = set(df[df["search_round"] == last_round]["config_name"].unique())
    print(f"Tracking **{len(final_configs)}** configs that survived to Round {last_round}.\n")

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

    print(prog.round(4).to_markdown())
    print()

    if "Delta" in prog.columns:
        improved = (prog["Delta"] < 0).sum()
        worsened = (prog["Delta"] > 0).sum()
        print(f"- **Improved R1→R3:** {improved}/{len(prog)} (mean Δ: {prog['Delta'].mean():.4f})")
        print(f"- **Worsened R1→R3:** {worsened}/{len(prog)}")

    cols = [c for c in prog.columns if c.startswith("R")]
    for i in range(len(cols) - 1):
        delta = prog[cols[i+1]] - prog[cols[i]]
        print(f"- {cols[i]}→{cols[i+1]}: mean change = {delta.mean():.4f}")
    print()


# ---------------------------------------------------------------------------
# 10. Elimination Analysis
# ---------------------------------------------------------------------------

def section_elimination(df):
    _section("10. Elimination Analysis")
    base = df[df["active_g_cfg"] == "False"]  # Use baseline pass for factor analysis

    rounds = sorted(df["search_round"].unique())
    for r_idx in range(len(rounds) - 1):
        r_curr, r_next = rounds[r_idx], rounds[r_idx + 1]
        curr_configs = set(base[base["search_round"] == r_curr]["config_name"].unique())
        next_configs = set(base[base["search_round"] == r_next]["config_name"].unique())
        dropped = curr_configs - next_configs
        survived = curr_configs & next_configs

        print(f"\n### R{r_curr} → R{r_next}: {len(dropped)} dropped, {len(survived)} survived\n")

        r_data = base[base["search_round"] == r_curr].drop_duplicates("config_name")
        for factor in ["wavelet", "bd_label", "ttd"]:
            surv_data = r_data[r_data["config_name"].isin(survived)]
            drop_data = r_data[r_data["config_name"].isin(dropped)]

            surv_counts = surv_data[factor].value_counts().sort_index()
            drop_counts = drop_data[factor].value_counts().sort_index()
            total_counts = r_data[factor].value_counts().sort_index()

            elim_rows = []
            for level in total_counts.index:
                t = total_counts.get(level, 0)
                s = surv_counts.get(level, 0)
                d = drop_counts.get(level, 0)
                pct = d / t * 100 if t > 0 else 0
                elim_rows.append({"Level": str(level), "Total": t, "Survived": s,
                                  "Dropped": d, "Elim%": f"{pct:.0f}%"})
            print(f"**{factor} elimination rates:**\n")
            print(pd.DataFrame(elim_rows).to_markdown(index=False))
            print()

    # ttd=5 near-total elimination
    print("### ttd=5 Near-Total Elimination\n")
    r1_base = base[base["search_round"] == 1]
    r3_configs = set(base[base["search_round"] == 3]["config_name"].unique())
    r3_data = r1_base[r1_base["config_name"].isin(r3_configs)]
    ttd5_in_r3 = r3_data[r3_data["ttd"] == 5]["config_name"].nunique()
    ttd3_in_r3 = r3_data[r3_data["ttd"] == 3]["config_name"].nunique()
    r1_ttd5 = r1_base[r1_base["ttd"] == 5]["config_name"].nunique()
    r1_ttd3 = r1_base[r1_base["ttd"] == 3]["config_name"].nunique()
    print(f"- R1: ttd=3 had {r1_ttd3} configs, ttd=5 had {r1_ttd5}")
    print(f"- R3: ttd=3 has {ttd3_in_r3} configs, ttd=5 has {ttd5_in_r3}")
    if r1_ttd5 > 0:
        print(f"- **ttd=5 survival rate:** {ttd5_in_r3}/{r1_ttd5} = "
              f"{ttd5_in_r3/r1_ttd5*100:.0f}%")

    ttd3_owa = r1_base[r1_base["ttd"] == 3]["owa"].dropna()
    ttd5_owa = r1_base[r1_base["ttd"] == 5]["owa"].dropna()
    if len(ttd3_owa) > 0 and len(ttd5_owa) > 0:
        u_stat, p_val = stats.mannwhitneyu(ttd3_owa, ttd5_owa, alternative="two-sided")
        print(f"- **Mann-Whitney U** (ttd=3 vs ttd=5, R1 OWA): U={u_stat:.1f}, "
              f"p={p_val:.6f} {_sig(p_val)}")
        print(f"- ttd=3 mean OWA: {ttd3_owa.mean():.4f}, ttd=5 mean OWA: {ttd5_owa.mean():.4f}")
    print()

    # Coif10 elimination
    print("### Coif10 Elimination\n")
    coif10_data = base[base["wavelet"] == "Coif10"]
    if not coif10_data.empty:
        last_round_coif10 = coif10_data["search_round"].max()
        c10_r1 = coif10_data[coif10_data["search_round"] == 1]
        print(f"- Coif10 last appeared in Round {last_round_coif10}")
        print(f"- R1 mean OWA: {c10_r1['owa'].mean():.4f}")
        in_r3 = coif10_data[coif10_data["search_round"] == 3]["config_name"].nunique()
        print(f"- Configs in R3: {in_r3} (**entirely eliminated:** {'Yes' if in_r3 == 0 else 'No'})")
    else:
        print("Coif10 not found in data.")
    print()

    # Higher-order Daubechies attrition
    print("### Higher-Order Daubechies Attrition\n")
    db_families = [w for w in sorted(df["wavelet"].unique()) if w.startswith("DB")]
    db_rows = []
    for wv in db_families:
        wv_data = base[base["wavelet"] == wv]
        r1_n = wv_data[wv_data["search_round"] == 1]["config_name"].nunique()
        r3_n = wv_data[wv_data["search_round"] == 3]["config_name"].nunique()
        r3_owa = ""
        if r3_n > 0:
            r3_owa = f"{wv_data[wv_data['search_round'] == 3]['owa'].mean():.4f}"
        surv = f"{r3_n/r1_n*100:.0f}%" if r1_n > 0 else "—"
        db_rows.append({"Wavelet": wv, "R1 configs": r1_n, "R3 configs": r3_n,
                        "Survival": surv, "R3 mean OWA": r3_owa})
    print(pd.DataFrame(db_rows).to_markdown(index=False))
    print()


# ---------------------------------------------------------------------------
# 11. Baseline Comparisons
# ---------------------------------------------------------------------------

def section_baseline_comparisons(df):
    _section("11. Baseline Comparisons")
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

        print(f"\n### Top 10 R3 Configs ({label}) vs Baselines\n")
        tbl = pass_agg.copy()
        tbl["±"] = tbl["std_owa"].fillna(0)
        tbl["Δ I+G"] = tbl["mean_owa"] - BASELINES["NBEATS-I+G"]["owa"]
        tbl["Δ G"] = tbl["mean_owa"] - BASELINES["NBEATS-G"]["owa"]
        tbl["Δ AE+T"] = tbl["mean_owa"] - BASELINES["AE+Trend"]["owa"]
        tbl = tbl.rename(columns={"config_name": "Config", "mean_owa": "OWA",
                                   "mean_smape": "sMAPE", "n_params": "Params"})
        print(tbl[["Config", "OWA", "±", "sMAPE", "Δ I+G", "Δ G", "Δ AE+T", "Params"]].to_markdown(
            index=False, floatfmt=("", ".4f", ".4f", ".2f", "+.4f", "+.4f", "+.4f", ",.0f")))

        print(f"\n**Reference baselines:**\n")
        bl_rows = [{"Baseline": name, "OWA": vals['owa'], "± OWA": vals['owa_std'],
                     "sMAPE": vals['smape'], "Params": vals['params']}
                    for name, vals in BASELINES.items()]
        print(pd.DataFrame(bl_rows).to_markdown(index=False, floatfmt=("", ".4f", ".3f", ".2f", ",.0f")))

    # Statistical tests
    print(f"\n### Statistical Tests (Mann-Whitney U, OWA)\n")
    r3_base_owa = r3[r3["active_g_cfg"] == "False"]
    r3_ag_owa = r3[r3["active_g_cfg"] == "forecast"]

    for pass_name, label, pass_df in [
        ("False", "baseline", r3_base_owa),
        ("forecast", "active_g=forecast", r3_ag_owa),
    ]:
        best_config = pass_df.groupby("config_name")["owa"].mean().idxmin()
        best_owa_vals = pass_df[pass_df["config_name"] == best_config]["owa"].values
        print(f"**Best R3 {label}:** `{best_config}` (n={len(best_owa_vals)})\n")

        test_rows = []
        if block_df is not None:
            for baseline_name in ["NBEATS-I+G", "NBEATS-G", "NBEATS-I", "GenericAE", "AutoEncoder"]:
                bl_vals = block_df[block_df["config_name"] == baseline_name]["owa"].dropna().values
                if len(bl_vals) > 0:
                    u, p = stats.mannwhitneyu(best_owa_vals, bl_vals, alternative="two-sided")
                    test_rows.append({"vs": baseline_name, "U": f"{u:.1f}", "p": f"{p:.4f}",
                                      "Sig": _sig(p), "Baseline Mean": f"{bl_vals.mean():.4f}",
                                      "n": len(bl_vals)})

        if ae_df is not None:
            ae_best = ae_df.groupby("config_name")["owa"].mean().idxmin()
            ae_vals = ae_df[ae_df["config_name"] == ae_best]["owa"].dropna().values
            if len(ae_vals) > 0:
                u, p = stats.mannwhitneyu(best_owa_vals, ae_vals, alternative="two-sided")
                test_rows.append({"vs": f"AE+Trend best ({ae_best})", "U": f"{u:.1f}",
                                  "p": f"{p:.4f}", "Sig": _sig(p),
                                  "Baseline Mean": f"{ae_vals.mean():.4f}", "n": len(ae_vals)})
        if test_rows:
            print(pd.DataFrame(test_rows).to_markdown(index=False))
        print()


# ---------------------------------------------------------------------------
# 12. Parameter Efficiency
# ---------------------------------------------------------------------------

def section_param_efficiency(df):
    _section("12. Parameter Efficiency")
    params_range = df["n_params"].agg(["min", "max"])
    print(f"- **WS3 param range:** {params_range['min']:,.0f} – {params_range['max']:,.0f}")
    print(f"- **Reduction vs NBEATS-G (24.7M):** "
          f"{(1 - params_range['max']/24_700_000)*100:.0f}%–{(1 - params_range['min']/24_700_000)*100:.0f}%")
    print(f"- **Reduction vs NBEATS-I+G (35.9M):** "
          f"{(1 - params_range['max']/35_900_000)*100:.0f}%–{(1 - params_range['min']/35_900_000)*100:.0f}%")
    print()

    r3 = df[df["search_round"] == 3]
    r3_agg = (
        r3.groupby(["config_name", "active_g_cfg"])
        .agg(mean_owa=("owa", "mean"), n_params=("n_params", "first"))
        .reset_index()
    )
    r3_agg["owa_per_Mparam"] = r3_agg["mean_owa"] / (r3_agg["n_params"] / 1_000_000)

    print("### OWA per Million Params (R3, lower is better)\n")
    top_eff = r3_agg.sort_values("owa_per_Mparam").head(10).copy()
    top_eff["Pass"] = top_eff["active_g_cfg"].map({"False": "baseline", "forecast": "activeG"})
    top_eff = top_eff.rename(columns={"config_name": "Config", "mean_owa": "OWA",
                                       "n_params": "Params", "owa_per_Mparam": "OWA/Mparam"})
    print(top_eff[["Config", "Pass", "OWA", "Params", "OWA/Mparam"]].to_markdown(
        index=False, floatfmt=("", "", ".4f", ",.0f", ".4f")))

    print(f"\n### Baseline OWA/Mparam Comparison\n")
    bl_eff = [{"Baseline": name, "OWA": vals['owa'],
               "Params (M)": f"{vals['params']/1e6:.1f}",
               "OWA/Mparam": vals['owa'] / (vals['params'] / 1_000_000)}
              for name, vals in BASELINES.items()]
    print(pd.DataFrame(bl_eff).to_markdown(index=False, floatfmt=("", ".4f", "", ".4f")))
    print()


# ---------------------------------------------------------------------------
# 13. Convergence Diagnostics
# ---------------------------------------------------------------------------

def section_convergence(df):
    _section("13. Convergence Diagnostics")

    print("### Epochs Trained per Round\n")
    ep_rows = []
    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        ep = rdf["epochs_trained"]
        es_count = int((rdf["stopping_reason"] == "EARLY_STOPPED").sum())
        max_ep_count = int((rdf["stopping_reason"] == "MAX_EPOCHS").sum())
        ep_rows.append({"Round": f"R{r}", "Epochs": f"{ep.min()}–{ep.max()}",
                        "Mean": f"{ep.mean():.1f}", "Early Stopped": es_count,
                        "Max Epochs": max_ep_count})
    print(pd.DataFrame(ep_rows).to_markdown(index=False))
    print()

    r3 = df[df["search_round"] == 3]
    if not r3.empty and "loss_ratio" in r3.columns:
        print("### R3 Loss Ratio (final_val / best_val) by Factor\n")
        for factor in ["wavelet", "bd_label", "ttd", "active_g_cfg"]:
            if factor in r3.columns:
                lr = r3.groupby(factor)["loss_ratio"].agg(["mean", "std", "max"])
                print(f"**{factor}:**\n")
                print(lr.to_markdown(floatfmt=".4f"))
                print()

    if not r3.empty and "best_epoch" in r3.columns:
        print("### R3 best_epoch Distribution\n")
        be = r3["best_epoch"]
        print(f"- mean={be.mean():.1f}, std={be.std():.1f}, "
              f"min={be.min()}, max={be.max()}, median={be.median():.0f}")
        print()

    if "val_loss_curve" in r3.columns:
        print("### R3 Val-Loss Curve Analysis\n")
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
                    diffs = np.diff(curve)
                    if np.any(diffs > 0):
                        non_monotone += 1
            except (ValueError, SyntaxError, TypeError):
                pass
        if improvements:
            print(f"- Parsed **{total_parsed}** curves")
            print(f"- Relative improvement (first→last): "
                  f"mean={np.mean(improvements):.1f}%, std={np.std(improvements):.1f}%")
            print(f"- Non-monotone curves: {non_monotone}/{total_parsed} "
                  f"({non_monotone/total_parsed*100:.0f}%)")
    print()


# ---------------------------------------------------------------------------
# 14. Final Verdict & Interpretation
# ---------------------------------------------------------------------------

def section_verdict(df):
    _section("14. Final Verdict & Interpretation")

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

    print("### Key Findings\n")

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
    print(f"1. **Dominant Factor:** `{top_factor}` (η²={eta_results[top_factor]:.3f}, "
          f"{eta_results[top_factor]*100:.1f}% of variance)")

    ttd3_mean = r1[r1["ttd"] == 3]["owa"].mean()
    ttd5_mean = r1[r1["ttd"] == 5]["owa"].mean()
    r3_ttd5 = r3[r3["ttd"] == 5]["config_name"].nunique()
    print(f"2. **Trend Thetas Dim:** ttd=3 ({ttd3_mean:.4f}) decisively beats ttd=5 "
          f"({ttd5_mean:.4f}). Only {r3_ttd5} ttd=5 config(s) survived to R3.")

    r1_base = r1[r1["active_g_cfg"] == "False"]
    wv_means = r1_base.groupby("wavelet")["owa"].mean().sort_values()
    print(f"3. **Wavelet Ranking (R1 mean OWA):**")
    for i, (wv, val) in enumerate(wv_means.items(), 1):
        r3_present = "✓ R3" if wv in r3["wavelet"].values else "✗ eliminated"
        print(f"   {i}. `{wv}`: {val:.4f}  ({r3_present})")

    bd_means = r1_base.groupby("bd_label")["owa"].mean().sort_values()
    print(f"4. **Basis Dim Ranking:** {' < '.join(f'{k} ({v:.4f})' for k, v in bd_means.items())}")

    best_row = overall_agg.iloc[0]
    best_owa = best_row.mean_owa
    best_std = best_row.std_owa if pd.notna(best_row.std_owa) else 0
    print(f"5. **Best Config:** `{best_row.config_name}` (active_g={best_row.active_g_cfg})")
    print(f"   OWA: {best_owa:.4f} ± {best_std:.4f}, "
          f"sMAPE: {best_row.mean_smape:.2f}, Params: {best_row.n_params:,.0f}")

    for name, vals in BASELINES.items():
        delta = best_owa - vals["owa"]
        verdict = "**BEATS**" if delta < -0.001 else "MATCHES" if abs(delta) < 0.005 else "LOSES TO"
        print(f"   - vs {name}: Δ={delta:+.4f} ({verdict})")

    print(f"6. **Parameter Efficiency:** ~{best_row.n_params/1e6:.1f}M params = "
          f"{(1 - best_row.n_params/24_700_000)*100:.0f}% fewer than NBEATS-G, "
          f"{(1 - best_row.n_params/35_900_000)*100:.0f}% fewer than NBEATS-I+G")

    # Recommended configs table
    print(f"\n### Recommended Configs\n")
    rec = overall_agg.head(10).copy()
    rec.insert(0, "#", range(1, len(rec) + 1))
    rec["Pass"] = rec["active_g_cfg"].map({"False": "baseline", "forecast": "activeG"})
    rec["std_owa"] = rec["std_owa"].fillna(0)
    rec = rec.rename(columns={"config_name": "Config", "mean_owa": "OWA", "std_owa": "±",
                               "mean_smape": "sMAPE", "mean_mase": "MASE", "n_params": "Params"})
    print(rec[["#", "Config", "Pass", "OWA", "±", "sMAPE", "MASE", "Params"]].to_markdown(
        index=False, floatfmt=(".0f", "", "", ".4f", ".4f", ".2f", ".3f", ",.0f")))

    ig_owa = BASELINES["NBEATS-I+G"]["owa"]
    if best_owa < ig_owa - 0.005:
        overall = "WaveletV3+Trend **BEATS** the NBEATS-I+G baseline with ~80% fewer params."
    elif best_owa < ig_owa + 0.005:
        overall = "WaveletV3+Trend **MATCHES** NBEATS-I+G with ~80% fewer params."
    else:
        overall = "WaveletV3+Trend does **NOT** match NBEATS-I+G but may beat simpler baselines."

    print(f"\n> **OVERALL VERDICT:** {overall}")

    # LLM variant comparison
    top_wavelet = r1.groupby("wavelet")["owa"].mean().sort_values()
    llm_ctx = {
        "variants": list(top_wavelet.index[:5]),
        "round_results": {
            "R1_wavelet_means": {str(k): float(v) for k, v in top_wavelet.items()},
            "best_r3_config": str(best_row.config_name),
            "best_r3_owa": float(best_owa),
            "vs_NBEATS_I+G": float(best_owa - ig_owa),
            "vs_AE+Trend": float(best_owa - BASELINES["AE+Trend"]["owa"]),
        },
    }
    llm_text = generate_commentary("variant_comparison", llm_ctx) if _LLM else None
    if llm_text:
        print(f"\n{llm_text}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()

    print("# Wavelet Study 3 — Successive Halving Analysis\n")

    # --- Abstract ---
    r3 = df[df["search_round"] == 3]
    r3_agg = r3.groupby(["config_name", "active_g_cfg"]).agg(
        mean_owa=("owa", "mean"), n_params=("n_params", "first")).reset_index()
    best_row = r3_agg.sort_values("mean_owa").iloc[0]
    n_wavelets = df["wavelet"].nunique()
    n_r1 = df[df["search_round"] == 1]["config_name"].nunique()
    n_r3 = r3["config_name"].nunique()

    print("## Abstract\n")
    print(f"This study applies successive-halving search to **WaveletV3+Trend** stacks on "
          f"M4-Yearly, exploring {n_wavelets} wavelet families across basis dimensions, "
          f"trend thetas dimensions, and active_g configurations. The search narrows "
          f"{n_r1} initial configurations to {n_r3} across 3 rounds of increasing epoch "
          f"budgets (7→15→30).\n")
    print("**Key Takeaways:**\n")
    print(f"- **Best configuration:** `{best_row.config_name}` "
          f"(active_g={best_row.active_g_cfg}) achieves mean OWA = **{best_row.mean_owa:.4f}** "
          f"with {best_row.n_params:,.0f} parameters.")
    ig_owa = BASELINES["NBEATS-I+G"]["owa"]
    delta = best_row.mean_owa - ig_owa
    verdict = "beats" if delta < -0.001 else "matches" if abs(delta) < 0.005 else "trails"
    print(f"- **vs NBEATS-I+G ({ig_owa:.4f}):** {verdict} (Δ = {delta:+.4f}).")
    param_red = (1 - best_row.n_params / 24_700_000) * 100
    print(f"- **Parameter efficiency:** ~{param_red:.0f}% fewer parameters than NBEATS-G.")
    print(f"- **Total rows analysed:** {len(df)} ({df['config_name'].nunique()} configs, "
          f"{df['search_round'].nunique()} rounds).\n")

    print(f"- **CSV:** `{CSV_PATH}`\n")

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


if __name__ == "__main__":
    main()
