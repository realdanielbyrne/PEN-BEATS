"""Wavelet + TrendAE Combination Study — Comprehensive Analysis

Reads experiments/results/m4/wavelet_trendae_comparison_results.csv and produces:
  1. Overview: dataset summary
  2. Overall leaderboard (median OWA across runs)
  3. Wavelet family marginals
  4. Latent dim (TrendAE) marginals
  5. Basis dim marginals
  6. Head-to-head: backcast vs forecast basis offset
  7. Stability analysis (OWA spread across seeds)
  8. Parameter efficiency (Pareto frontier)
  9. Comparison vs standalone WaveletV3 and AE+Trend baselines
 10. Notable takeaways and recommended configurations

Usage:
    python experiments/wavelet_trendae_comparison_analysis.py
    python experiments/wavelet_trendae_comparison_analysis.py > experiments/analysis_reports/wavelet_trendae_comparison_analysis.md
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 55)
pd.set_option("display.float_format", "{:.4f}".format)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_SCRIPT_DIR, "results", "m4", "wavelet_trendae_comparison_results.csv")
BASELINE_CSV = os.path.join(_SCRIPT_DIR, "results", "m4", "block_benchmark_results.csv")

NBEATS_G_PARAMS = 24_700_000

PUBLISHED_BASELINES = {
    "NBEATS-I+G":   {"owa": 0.8057, "smape": 13.53, "params": 35_900_000},
    "AE+Trend":     {"owa": 0.8015, "smape": 13.53, "params": 5_200_000},
    "GenericAE":    {"owa": 0.8063, "smape": 13.57, "params": 4_800_000},
    "NBEATS-G":     {"owa": 0.8198, "smape": 13.70, "params": 24_700_000},
}

NUM_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "basis_dim", "trend_thetas_dim_cfg", "latent_dim_cfg",
]


def _section(title):
    print(f"\n## {title}\n")


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Not found: {CSV_PATH}")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["wavelet"] = df["wavelet_family"].str.replace("WaveletV3", "", regex=False)
    df["diverged"] = df["diverged"].astype(str).str.lower().isin(["true", "1"])
    return df


def overall_leaderboard(df):
    agg = (
        df.groupby("config_name")
        .agg(
            med_owa=("owa", "median"), min_owa=("owa", "min"), max_owa=("owa", "max"),
            med_smape=("smape", "median"), med_mase=("mase", "median"),
            n_params=("n_params", "first"), wavelet=("wavelet", "first"),
            latent_dim=("latent_dim_cfg", "first"), basis_dim=("basis_dim", "first"),
            bd_label=("bd_label", "first") if "bd_label" in df.columns else ("basis_dim", "first"),
        )
        .sort_values("med_owa")
    )
    n_cfg = df["config_name"].nunique()
    n_runs = df.groupby("config_name")["run"].nunique().iloc[0] if n_cfg > 0 else 0
    print(f"{n_cfg} configs × {n_runs} runs\n")
    tbl = agg.reset_index().copy()
    tbl.insert(0, "#", range(1, len(tbl) + 1))
    tbl["±"] = tbl["max_owa"] - tbl["min_owa"]
    tbl = tbl.rename(columns={"config_name": "Config", "wavelet": "Wavelet", "med_owa": "OWA",
                               "med_smape": "sMAPE", "med_mase": "MASE", "n_params": "Params"})
    print(tbl[["#", "Config", "Wavelet", "OWA", "±", "sMAPE", "MASE", "Params"]].to_markdown(
        index=False, floatfmt=(".0f", "", "", ".4f", ".4f", ".2f", ".4f", ",.0f")))
    print()
    best = agg.iloc[0]
    worst = agg.iloc[-1]
    print(f"The best configuration achieves a median OWA of **{best.med_owa:.4f}** while the worst "
          f"reaches **{worst.med_owa:.4f}**, a spread of {worst.med_owa - best.med_owa:.4f}. "
          f"The top config uses the **{best.wavelet}** wavelet family with **{int(best.n_params):,}** parameters.")
    return agg


def marginals(df):
    for col, label in [
        ("wavelet", "Wavelet Family"),
        ("latent_dim_cfg", "Latent Dim (TrendAE)"),
        ("basis_dim", "Basis Dim (WaveletV3)"),
        ("bd_label", "Basis Offset (lt_bcast vs eq_fcast)"),
    ]:
        if col not in df.columns:
            continue
        grp = (
            df.groupby(col)
            .agg(med_owa=("owa", "median"), mean_owa=("owa", "mean"),
                 std_owa=("owa", "std"), n=("owa", "count"),
                 med_params=("n_params", "median"))
            .sort_values("med_owa")
        )
        print(f"\n### {label}\n")
        tbl = grp.reset_index().rename(columns={col: "Value", "med_owa": "Med OWA", "mean_owa": "Mean",
                                                  "std_owa": "Std", "n": "N", "med_params": "Med Params"})
        print(tbl.to_markdown(index=False, floatfmt=("", ".4f", ".4f", ".4f", ".0f", ",.0f")))
        best_val = grp.index[0]
        worst_val = grp.index[-1]
        print(f"\n`{best_val}` is the strongest setting (median OWA {grp.iloc[0].med_owa:.4f}) "
              f"while `{worst_val}` is the weakest ({grp.iloc[-1].med_owa:.4f}).")


def latent_dim_discussion(df):
    """Data-driven discussion on selecting the optimal latent dimension for TrendAE."""
    col = "latent_dim_cfg"
    if col not in df.columns or df[col].dropna().empty:
        print("(latent_dim_cfg column not found — skipping)\n")
        return

    grp = (
        df.groupby(col)
        .agg(
            med_owa=("owa", "median"), mean_owa=("owa", "mean"),
            std_owa=("owa", "std"), n=("owa", "count"),
            med_params=("n_params", "median"),
        )
        .sort_values(col)
    )
    dims = grp.index.tolist()
    best_dim = grp["med_owa"].idxmin()
    worst_dim = grp["med_owa"].idxmax()
    best_owa = grp.loc[best_dim, "med_owa"]
    worst_owa = grp.loc[worst_dim, "med_owa"]

    print("In this hybrid stack, the **TrendAE** component uses an AERootBlock "
          "backbone whose bottleneck width is controlled by `latent_dim`. "
          "The encoder path is `backcast_length → units/2 → latent_dim` and the "
          "decoder expands back via `latent_dim → units/2 → units`, after which "
          "the trend head applies a Vandermonde polynomial basis expansion. "
          "A smaller latent_dim increases regularisation while a larger value "
          "preserves more signal for the trend polynomial to fit.\n")

    bcl = int(df["backcast_length"].iloc[0]) if "backcast_length" in df.columns else 30
    print(f"With backcast_length = {bcl}, the tested latent dimensions are: "
          f"**{', '.join(str(int(d)) for d in dims)}**.\n")

    for d in dims:
        r = grp.loc[d]
        tag = " ← best" if d == best_dim else (" ← worst" if d == worst_dim else "")
        print(f"- **latent_dim = {int(d)}:** median OWA = {r.med_owa:.4f}, "
              f"std = {r.std_owa:.4f}, params ≈ {r.med_params:,.0f}{tag}")
    print()

    delta = worst_owa - best_owa
    print(f"The optimal setting is **latent_dim = {int(best_dim)}** "
          f"(median OWA {best_owa:.4f}), outperforming the worst "
          f"(latent_dim = {int(worst_dim)}) by Δ = {delta:.4f}. ")

    if best_dim == min(dims):
        print("The tightest bottleneck wins. The TrendAE head already imposes "
              "strong inductive bias via its polynomial basis, so the backbone "
              "needs only a minimal latent representation. Combined with the "
              "WaveletV3 stack handling oscillatory components, the TrendAE "
              "can afford aggressive compression for the slowly-varying residual.\n")
    elif best_dim == max(dims):
        print("The widest bottleneck wins, suggesting the trend polynomial "
              "benefits from richer backbone features. In this wavelet+TrendAE "
              "combination, the wavelet stack absorbs oscillatory patterns, "
              "leaving the TrendAE to model a potentially complex residual "
              "that benefits from a wider information path.\n")
    else:
        print("A mid-range bottleneck provides the best trade-off. "
              "The wavelet stack handles oscillatory components, so the TrendAE "
              "residual is relatively smooth but not trivially simple — a moderate "
              "bottleneck captures enough structure without overfitting.\n")

    print("**Practical recommendation:** Use `latent_dim = "
          f"{int(best_dim)}` for Wavelet+TrendAE stacks on M4-Yearly. "
          "Since the TrendAE only needs to model the slowly-varying residual "
          "after the wavelet stack, the latent dimension can be kept small. "
          "For longer forecast horizons, experiment with slightly larger values.\n")


def stability_analysis(df):
    spread = (
        df.groupby("config_name")["owa"]
        .agg(["median", "min", "max", "std"])
        .assign(range=lambda x: x["max"] - x["min"])
        .sort_values("median")
    )
    print(f"- **Mean spread (max−min):** {spread['range'].mean():.4f}")
    print(f"- **Max spread (max−min):** {spread['range'].max():.4f} (`{spread['range'].idxmax()}`)")
    print(f"- **Mean std:** {spread['std'].mean():.4f}")
    most_stable = spread.sort_values("range").head(5).reset_index()
    most_stable.columns = ["Config", "Median OWA", "Min", "Max", "Std", "Range"]
    print(f"\n### Most Stable Configs (smallest max−min spread)\n")
    print(most_stable[["Config", "Median OWA", "Range", "Std"]].to_markdown(
        index=False, floatfmt=("", ".4f", ".4f", ".4f")))


def param_efficiency(df):
    agg = (
        df.groupby("config_name")
        .agg(med_owa=("owa", "median"), n_params=("n_params", "first"),
             wavelet=("wavelet", "first"))
        .sort_values("n_params")
    )
    pareto = []
    best_owa = float("inf")
    for name, r in agg.iterrows():
        if r.med_owa < best_owa:
            pareto.append(name)
            best_owa = r.med_owa
    tbl = agg.reset_index().copy()
    tbl["Reduction"] = (1 - tbl["n_params"] / NBEATS_G_PARAMS) * 100
    tbl["Pareto"] = tbl["config_name"].apply(lambda n: "◀" if n in pareto else "")
    tbl = tbl.rename(columns={"config_name": "Config", "wavelet": "Wavelet",
                               "n_params": "Params", "med_owa": "Med OWA"})
    print(tbl[["Config", "Wavelet", "Params", "Reduction", "Med OWA", "Pareto"]].to_markdown(
        index=False, floatfmt=("", "", ",.0f", ".1f", ".4f", "")))
    print(f"\n**{len(pareto)} Pareto-optimal** configurations identified where no other config "
          f"achieves both lower OWA and fewer parameters.")


def baseline_comparison(df):
    bl_rows = [{"Source": name, "Med OWA": vals["owa"], "Params": vals["params"]}
               for name, vals in sorted(PUBLISHED_BASELINES.items(), key=lambda x: x[1]["owa"])]
    print(pd.DataFrame(bl_rows).to_markdown(index=False, floatfmt=("", ".4f", ",.0f")))
    print()
    best = df.groupby("config_name")["owa"].median().sort_values()
    top5 = best.head(5)
    print("### Top-5 Wavelet+TrendAE Configs (this study)\n")
    top5_rows = [{"Config": name, "Med OWA": owa,
                  "Δ vs AE+Trend": owa - PUBLISHED_BASELINES["AE+Trend"]["owa"]}
                 for name, owa in top5.items()]
    print(pd.DataFrame(top5_rows).to_markdown(index=False, floatfmt=("", ".4f", "+.4f")))
    best_name = top5.index[0]
    best_delta = top5.iloc[0] - PUBLISHED_BASELINES["AE+Trend"]["owa"]
    verdict = "improves upon" if best_delta < -0.001 else "matches" if abs(best_delta) < 0.005 else "falls short of"
    print(f"\nThe best Wavelet+TrendAE config (`{best_name}`) {verdict} the AE+Trend baseline "
          f"(Δ = {best_delta:+.4f}).")


def training_stability(df):
    n_total = len(df)
    n_div = int(df["diverged"].sum())
    early_stopped = int((df["stopping_reason"] == "EARLY_STOPPED").sum())
    max_epochs = int((df["stopping_reason"] == "MAX_EPOCHS").sum())
    stab_rows = [
        {"Metric": "Total runs", "Count": n_total, "%": ""},
        {"Metric": "Diverged", "Count": n_div, "%": f"{n_div/n_total*100:.1f}%"},
        {"Metric": "Early stopped", "Count": early_stopped, "%": f"{early_stopped/n_total*100:.1f}%"},
        {"Metric": "Hit max epoch", "Count": max_epochs, "%": f"{max_epochs/n_total*100:.1f}%"},
    ]
    print(pd.DataFrame(stab_rows).to_markdown(index=False))
    if n_div > 0:
        print(f"\n⚠️ **{n_div} runs diverged** ({n_div/n_total*100:.1f}%), suggesting some "
              f"configurations may require learning rate tuning or gradient clipping.")
    else:
        print(f"\nAll {n_total} runs converged without divergence — the architecture is stable across seeds.")


def main():
    df = load()
    total_runs = len(df)
    total_configs = df["config_name"].nunique()
    total_time = df["training_time_seconds"].sum()

    # --- H1 Title ---
    print("# Wavelet + TrendAE Combination Study — Results Analysis\n")

    # --- Abstract ---
    best_cfg = df.groupby("config_name")["owa"].median().idxmin()
    best_owa = df.groupby("config_name")["owa"].median().min()
    best_params = int(df[df["config_name"] == best_cfg]["n_params"].iloc[0])
    aet_owa = PUBLISHED_BASELINES["AE+Trend"]["owa"]
    delta_aet = best_owa - aet_owa
    n_div = int(df["diverged"].sum())
    param_red = (1 - best_params / NBEATS_G_PARAMS) * 100

    print("## Abstract\n")
    print(f"This study evaluates **Wavelet+TrendAE** hybrid stacks on the M4-Yearly benchmark, "
          f"exploring {total_configs} configurations across {total_runs} total runs "
          f"({total_time / 60:.1f} min total training time). "
          f"The best configuration, `{best_cfg}`, achieves a median OWA of **{best_owa:.4f}** "
          f"(Δ = {delta_aet:+.4f} vs AE+Trend baseline at {aet_owa:.4f}) "
          f"with **{best_params:,}** parameters ({param_red:.0f}% fewer than NBEATS-G). "
          f"{'All runs converged successfully.' if n_div == 0 else f'{n_div} runs diverged.'}\n")

    print("**Key Takeaways:**\n")
    wv_best = df.groupby("wavelet")["owa"].median().idxmin()
    print(f"1. **Best wavelet family:** `{wv_best}`")
    beat_str = "beats" if delta_aet < -0.001 else "matches" if abs(delta_aet) < 0.005 else "falls short of"
    print(f"2. **vs AE+Trend baseline:** {beat_str} (Δ = {delta_aet:+.4f})")
    print(f"3. **Parameter efficiency:** {param_red:.0f}% reduction vs NBEATS-G")
    print()

    _section("1. Overview")
    print(f"- **CSV:** `{CSV_PATH}`")
    print(f"- **Rows:** {total_runs} ({total_configs} unique configs)")
    print(f"- **Total training time:** {total_time / 60:.1f} min")
    print(f"- **Wavelet families:** {sorted(df['wavelet'].unique())}")
    print(f"- **OWA range:** {df['owa'].min():.4f} – {df['owa'].max():.4f}\n")
    bl_rows = [{"Baseline": name, "OWA": vals["owa"], "sMAPE": vals["smape"], "Params": vals["params"]}
               for name, vals in PUBLISHED_BASELINES.items()]
    print(pd.DataFrame(bl_rows).to_markdown(index=False, floatfmt=("", ".4f", ".2f", ",.0f")))

    _section("2. Overall Leaderboard")
    overall_leaderboard(df)

    _section("3. Hyperparameter Marginals")
    marginals(df)

    _section("3b. Selecting the Optimal Latent Dimension (TrendAE)")
    latent_dim_discussion(df)

    _section("4. Stability Analysis (OWA spread across seeds)")
    stability_analysis(df)

    _section("5. Parameter Efficiency")
    param_efficiency(df)

    _section("6. Baseline Comparison")
    baseline_comparison(df)

    _section("7. Training Stability (divergence / stopping)")
    training_stability(df)


if __name__ == "__main__":
    main()

