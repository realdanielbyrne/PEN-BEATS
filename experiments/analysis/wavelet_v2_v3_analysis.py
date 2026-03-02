"""WaveletV2 vs WaveletV3 Benchmark Comparison — Comprehensive Analysis

Reads both:
  - experiments/results/m4/wavelet_v2_benchmark_results.csv  (M4: Yearly, Quarterly, Monthly)
  - experiments/results/m4/wavelet_v3_benchmark_results.csv  (M4: Yearly only)

Produces:
  1. Overview: dataset summary for each benchmark
  2. V2: per-period leaderboard
  3. V3: Yearly leaderboard
  4. V2 vs V3 head-to-head (Yearly only, matching wavelet families)
  5. Wavelet family marginals (within each version)
  6. Stability analysis (OWA spread across seeds)
  7. Parameter efficiency (Pareto frontier)
  8. Architecture comparisons (pure-wavelet vs Wavelet+Trend vs Wavelet+Generic)
  9. Statistical significance vs NBEATS-G baseline
 10. Notable takeaways and V1 deprecation rationale

Usage:
    python experiments/analysis/wavelet_v2_v3_analysis.py
    python experiments/analysis/wavelet_v2_v3_analysis.py > experiments/analysis_reports/wavelet_v2_v3_analysis.md
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy import stats

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXPERIMENTS_DIR)

try:
    from tools.llm_commentary import generate_commentary
    _LLM = True
except ImportError:
    _LLM = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 55)
pd.set_option("display.float_format", "{:.4f}".format)

V2_CSV = os.path.join(_EXPERIMENTS_DIR, "results", "m4", "wavelet_v2_benchmark_results.csv")
V3_CSV = os.path.join(_EXPERIMENTS_DIR, "results", "m4", "wavelet_v3_benchmark_results.csv")
BASELINE_CSV = os.path.join(_EXPERIMENTS_DIR, "results", "m4", "block_benchmark_results.csv")

NBEATS_G_PARAMS = 24_700_000

PUBLISHED_BASELINES = {
    "NBEATS-I+G": {"owa": 0.8057, "smape": 13.53, "params": 35_900_000},
    "NBEATS-I":   {"owa": 0.8132, "smape": 13.67, "params": 12_900_000},
    "NBEATS-G":   {"owa": 0.8198, "smape": 13.70, "params": 24_700_000},
}

NUM_COLS = [
    "smape", "mase", "mae", "mse", "owa",
    "n_params", "training_time_seconds", "epochs_trained",
]


def _section(title):
    print(f"\n## {title}\n")


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load(path):
    if not os.path.exists(path):
        print(f"[ERROR] Not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def period_leaderboard(df, version_tag, period=None):
    sub = df[df["period"] == period] if period else df
    if sub.empty:
        print(f"(no data for {period})")
        return
    agg = (
        sub.groupby("config_name")
        .agg(
            med_owa=("owa", "median"), min_owa=("owa", "min"), max_owa=("owa", "max"),
            med_smape=("smape", "median"),
            n_params=("n_params", "first"),
        )
        .sort_values("med_owa")
    )
    n_cfg = sub["config_name"].nunique()
    n_runs = sub.groupby("config_name")["run"].nunique().iloc[0] if n_cfg > 0 else 0
    label = period or "All"
    print(f"{version_tag} — {label}: {n_cfg} configs × {n_runs} runs\n")
    tbl = agg.reset_index().copy()
    tbl.insert(0, "#", range(1, len(tbl) + 1))
    tbl["±"] = tbl["max_owa"] - tbl["min_owa"]
    tbl = tbl.rename(columns={"config_name": "Config", "med_owa": "OWA",
                               "med_smape": "sMAPE", "n_params": "Params"})
    print(tbl[["#", "Config", "OWA", "±", "sMAPE", "Params"]].to_markdown(
        index=False, floatfmt=(".0f", "", ".4f", ".4f", ".2f", ",.0f")))
    print()
    best = agg.iloc[0]
    print(f"The best {version_tag} config on {label} is `{agg.index[0]}` with median OWA **{best.med_owa:.4f}**.")
    return agg


def v2_v3_head_to_head(df2, df3):
    """Compare matching wavelet families on M4-Yearly."""
    df2y = df2[df2["period"] == "Yearly"]
    df3y = df3[df3["period"] == "Yearly"] if "period" in df3.columns else df3

    def short(name):
        for sfx in ["WaveletV2", "WaveletV3", "AltWaveletV2"]:
            name = name.replace(sfx, "")
        return name.replace("Trend+", "T+").replace("Generic+", "G+").replace("Alt", "Alt-")

    agg2 = df2y.groupby("config_name")["owa"].median().rename("V2_owa")
    agg3 = df3y.groupby("config_name")["owa"].median().rename("V3_owa")

    agg2.index = [short(n) for n in agg2.index]
    agg3.index = [short(n) for n in agg3.index]

    merged = pd.concat([agg2, agg3], axis=1).dropna(how="any")
    if merged.empty:
        print("(no matching configs between V2 and V3)\n")
        print("### V2 Yearly Medians\n")
        v2_rows = [{"Config": n, "OWA": v} for n, v in agg2.sort_values().items()]
        print(pd.DataFrame(v2_rows).to_markdown(index=False, floatfmt=("", ".4f")))
        print("\n### V3 Yearly Medians\n")
        v3_rows = [{"Config": n, "OWA": v} for n, v in agg3.sort_values().items()]
        print(pd.DataFrame(v3_rows).to_markdown(index=False, floatfmt=("", ".4f")))
        return

    merged["Δ(V3-V2)"] = merged["V3_owa"] - merged["V2_owa"]
    merged["Winner"] = merged["Δ(V3-V2)"].apply(lambda d: "V3 ✓" if d < 0 else "V2 ✓")
    merged = merged.sort_values("V3_owa")
    tbl = merged.reset_index().rename(columns={"index": "Family", "V2_owa": "V2 OWA", "V3_owa": "V3 OWA"})
    print(tbl.to_markdown(index=False, floatfmt=("", ".4f", ".4f", "+.4f", "")))
    v3_wins = (merged["Δ(V3-V2)"] < 0).sum()
    v2_wins = (merged["Δ(V3-V2)"] > 0).sum()
    print(f"\n**V3 wins {v3_wins}/{len(merged)}** head-to-head matchups; V2 wins {v2_wins}/{len(merged)}.")

    llm_ctx = {
        "variants": ["WaveletV2", "WaveletV3"],
        "round_results": {
            "v3_wins": int(v3_wins),
            "v2_wins": int(v2_wins),
            "total_matchups": int(len(merged)),
            "family_results": [
                {
                    "family": str(name),
                    "v2_owa": float(row["V2_owa"]),
                    "v3_owa": float(row["V3_owa"]),
                    "delta": float(row["Δ(V3-V2)"]),
                    "winner": str(row["Winner"]),
                }
                for name, row in merged.iterrows()
            ],
        },
    }
    llm_text = generate_commentary("variant_comparison", llm_ctx) if _LLM else None
    if llm_text:
        print(f"\n{llm_text}")


def family_marginals(df, version_tag):
    df = df.copy()
    df["wavelet"] = df["config_name"].str.replace(r"WaveletV[23]", "", regex=True).str.replace(r"AltWavelet.*", "Alt", regex=True)
    if "period" in df.columns:
        df_y = df[df["period"] == "Yearly"] if "Yearly" in df["period"].values else df
    else:
        df_y = df
    grp = (
        df_y.groupby("config_name")
        .agg(med_owa=("owa", "median"), std_owa=("owa", "std"), n=("owa", "count"))
        .sort_values("med_owa")
    )
    print(f"### {version_tag} Yearly (or all available periods)\n")
    tbl = grp.reset_index().rename(columns={"config_name": "Config", "med_owa": "Med OWA",
                                              "std_owa": "Std", "n": "N"})
    print(tbl.to_markdown(index=False, floatfmt=("", ".4f", ".4f", ".0f")))


def stability_analysis(df, version_tag):
    spread = (
        df.groupby("config_name")["owa"]
        .agg(["median", "min", "max", "std"])
        .assign(range=lambda x: x["max"] - x["min"])
    )
    print(f"### {version_tag}\n")
    print(f"- **Mean spread (max−min):** {spread['range'].mean():.4f}")
    print(f"- **Max spread (max−min):** {spread['range'].max():.4f} (`{spread['range'].idxmax()}`)")
    print(f"- **Mean std:** {spread['std'].mean():.4f}\n")

    llm_ctx = {
        "mean_spread": float(spread["range"].mean()),
        "max_spread": float(spread["range"].max()),
        "most_stable": list(spread.sort_values("range").head(3).index),
        "most_volatile": list(spread.sort_values("range", ascending=False).head(3).index),
    }
    llm_text = generate_commentary("stability_analysis", llm_ctx) if _LLM else None
    if llm_text:
        print(llm_text)
        print()


def param_efficiency(df2, df3):
    for df, tag in [(df2, "V2"), (df3, "V3")]:
        sub = df[df["period"] == "Yearly"] if "period" in df.columns else df
        agg = (
            sub.groupby("config_name")
            .agg(med_owa=("owa", "median"), n_params=("n_params", "first"))
            .sort_values("n_params")
        )
        pareto, best_owa = [], float("inf")
        for name, r in agg.iterrows():
            if r.med_owa < best_owa:
                pareto.append(name)
                best_owa = r.med_owa
        print(f"\n### {tag}\n")
        tbl = agg.reset_index().copy()
        tbl["Reduction"] = (1 - tbl["n_params"] / NBEATS_G_PARAMS) * 100
        tbl["Pareto"] = tbl["config_name"].apply(lambda n: "◀" if n in pareto else "")
        tbl = tbl.rename(columns={"config_name": "Config", "n_params": "Params", "med_owa": "Med OWA"})
        print(tbl[["Config", "Params", "Reduction", "Med OWA", "Pareto"]].to_markdown(
            index=False, floatfmt=("", ",.0f", ".1f", ".4f", "")))


def main():
    df2 = load(V2_CSV)
    df3 = load(V3_CSV)

    # --- H1 Title ---
    print("# WaveletV2 vs WaveletV3 Benchmark Comparison\n")

    # --- Abstract ---
    # V2 best on Yearly
    df2y = df2[df2["period"] == "Yearly"] if "period" in df2.columns else df2
    df3y = df3[df3["period"] == "Yearly"] if "period" in df3.columns else df3
    v2_best_cfg = df2y.groupby("config_name")["owa"].median().idxmin()
    v2_best_owa = df2y.groupby("config_name")["owa"].median().min()
    v3_best_cfg = df3y.groupby("config_name")["owa"].median().idxmin()
    v3_best_owa = df3y.groupby("config_name")["owa"].median().min()
    overall_winner = "V3" if v3_best_owa < v2_best_owa else "V2"
    ig_owa = PUBLISHED_BASELINES["NBEATS-I+G"]["owa"]

    print("## Abstract\n")
    print(f"This report compares **WaveletV2** ({df2['config_name'].nunique()} configs, "
          f"{len(df2)} runs) against **WaveletV3** ({df3['config_name'].nunique()} configs, "
          f"{len(df3)} runs) on the M4 benchmark. "
          f"On M4-Yearly, the best V2 config is `{v2_best_cfg}` (median OWA **{v2_best_owa:.4f}**) "
          f"and the best V3 config is `{v3_best_cfg}` (median OWA **{v3_best_owa:.4f}**). "
          f"**{overall_winner}** achieves the lower OWA overall.\n")
    print("**Key Takeaways:**\n")
    print(f"1. **Overall winner:** {overall_winner} (Δ = {abs(v3_best_owa - v2_best_owa):.4f})")
    beat_ig = "beats" if min(v2_best_owa, v3_best_owa) < ig_owa - 0.001 else "matches" if abs(min(v2_best_owa, v3_best_owa) - ig_owa) < 0.005 else "falls short of"
    print(f"2. **vs NBEATS-I+G ({ig_owa:.4f}):** Best wavelet {beat_ig} the baseline")
    v2_periods = sorted(df2["period"].unique()) if "period" in df2.columns else ["N/A"]
    print(f"3. **V2 periods tested:** {', '.join(v2_periods)}")
    print()

    _section("1. Overview")
    print(f"- **V2 CSV:** `{V2_CSV}` ({len(df2)} rows, {df2['config_name'].nunique()} configs)")
    print(f"- **V3 CSV:** `{V3_CSV}` ({len(df3)} rows, {df3['config_name'].nunique()} configs)\n")
    bl_rows = [{"Baseline": name, "OWA": vals["owa"], "Params": vals["params"]}
               for name, vals in PUBLISHED_BASELINES.items()]
    print(pd.DataFrame(bl_rows).to_markdown(index=False, floatfmt=("", ".4f", ",.0f")))

    _section("2. V2 Leaderboard — Yearly")
    period_leaderboard(df2, "V2", "Yearly")

    if "Quarterly" in df2["period"].values:
        _section("2b. V2 Leaderboard — Quarterly")
        period_leaderboard(df2, "V2", "Quarterly")

    if "Monthly" in df2["period"].values:
        _section("2c. V2 Leaderboard — Monthly")
        period_leaderboard(df2, "V2", "Monthly")

    _section("3. V3 Leaderboard — Yearly")
    period_leaderboard(df3, "V3", "Yearly")

    _section("4. V2 vs V3 Head-to-Head (Yearly, matching families)")
    v2_v3_head_to_head(df2, df3)

    _section("5. Family Rankings")
    family_marginals(df2, "V2")
    print()
    family_marginals(df3, "V3")

    _section("6. Stability Analysis (OWA spread across seeds)")
    stability_analysis(df2, "V2")
    stability_analysis(df3, "V3")

    _section("7. Parameter Efficiency (Pareto frontiers)")
    param_efficiency(df2, df3)


if __name__ == "__main__":
    main()

