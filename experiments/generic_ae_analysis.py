"""Generic AE Pure-Stack Study — Comprehensive Analysis

Reads experiments/results/m4/generic_ae_pure_stack_results.csv and produces:
  1. Overview / dataset summary
  2. Successive halving funnel
  3. Per-round leaderboards
  4. Hyperparameter marginals (block_type, latent_dim, thetas_dim, active_g)
  5. Block-type head-to-head (GenericAE vs BottleneckGenericAE vs I+G baseline)
  6. Stability analysis (OWA spread across seeds)
  7. Round-over-round progression for promoted configs
  8. Parameter efficiency (OWA vs n_params, Pareto frontier)
  9. Statistical significance vs NBEATS-I+G baseline (Mann-Whitney U)
 10. Training stability (divergence rates, early stopping)
 11. Notable takeaways and recommended configurations
 12. Caveats and limitations

Usage:
    python experiments/generic_ae_analysis.py
    python experiments/generic_ae_analysis.py > experiments/analysis_reports/generic_ae_analysis.md
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy import stats

try:
    from llm_commentary import generate_commentary
    _LLM = True
except ImportError:
    _LLM = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 55)
pd.set_option("display.float_format", "{:.4f}".format)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_SCRIPT_DIR, "results", "m4", "generic_ae_pure_stack_results.csv")
BASELINE_CSV = os.path.join(_SCRIPT_DIR, "results", "m4", "block_benchmark_results.csv")

TARGET_OWA = 0.85
NBEATS_G_PARAMS = 24_700_000  # 30-stack Generic baseline

# Published M4-Yearly paper baselines (30-stack, 10 seeds)
PUBLISHED_BASELINES = {
    "NBEATS-I+G":  {"owa": 0.8057, "smape": 13.53, "params": 35_900_000},
    "NBEATS-I":    {"owa": 0.8132, "smape": 13.67, "params": 12_900_000},
    "NBEATS-G":    {"owa": 0.8198, "smape": 13.70, "params": 24_700_000},
}

NUM_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "latent_dim_cfg", "thetas_dim_cfg",
]


def section(title, char="="):
    """Emit a Markdown section heading (## level)."""
    print()
    print(f"## {title}")
    print()


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
    if "search_round" in df.columns:
        df["search_round"] = pd.to_numeric(df["search_round"], errors="coerce").fillna(1).astype(int)
    df["active_g"] = df["active_g"].astype(str)
    df["diverged"] = df["diverged"].astype(str).str.lower().isin(["true", "1"])
    return df


def funnel_summary(df):
    rounds = sorted(df["search_round"].unique())
    rows = []
    for r in rounds:
        rdf = df[df["search_round"] == r]
        n_cfg = rdf["config_name"].nunique()
        n_runs = len(rdf)
        max_ep = rdf["epochs_trained"].max()
        best = rdf.groupby("config_name")["owa"].median().min()
        rows.append({"Round": r, "Configs": n_cfg, "Runs": n_runs,
                      "Epochs": max_ep, "Best Med OWA": f"{best:.4f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()
    if len(rounds) >= 2:
        print(f"The successive halving procedure narrowed from {rows[0]['Configs']} to "
              f"{rows[-1]['Configs']} configurations across {len(rounds)} rounds. "
              f"Each round increased the training budget while eliminating weaker candidates.")
    print()


def round_leaderboard(df, round_num):
    rdf = df[df["search_round"] == round_num]
    if rdf.empty:
        print(f"(no data for round {round_num})")
        return

    agg_cols = {
        "med_owa": ("owa", "median"), "min_owa": ("owa", "min"), "max_owa": ("owa", "max"),
        "med_smape": ("smape", "median"), "med_mase": ("mase", "median"),
        "n_params": ("n_params", "first"), "block_type": ("block_type", "first"),
        "latent_dim": ("latent_dim_cfg", "first"), "thetas_dim": ("thetas_dim_cfg", "first"),
        "active_g": ("active_g", "first"),
    }
    agg = rdf.groupby("config_name").agg(**agg_cols).sort_values("med_owa")

    n_cfg = rdf["config_name"].nunique()
    n_runs = rdf.groupby("config_name")["run"].nunique().iloc[0] if n_cfg > 0 else 0
    max_ep = rdf["epochs_trained"].max()
    print(f"{n_cfg} configs × {n_runs} runs, {max_ep} epochs each\n")

    rows = []
    for rank, (name, r) in enumerate(agg.iterrows(), 1):
        spread = r.max_owa - r.min_owa
        rows.append({"Rank": rank, "Config": name, "Block": str(r.block_type),
                      "OWA": f"{r.med_owa:.4f}", "±": f"{spread:.4f}",
                      "sMAPE": f"{r.med_smape:.2f}", "Params": f"{r.n_params:,.0f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()
    best = agg.iloc[0]
    print(f"The top configuration achieves median OWA = {best.med_owa:.4f} "
          f"with {best.n_params:,.0f} parameters.")
    print()


def main():
    df = load()
    total_runs = len(df)
    total_configs = df["config_name"].nunique()
    total_rounds = df["search_round"].nunique()
    total_time = df["training_time_seconds"].sum()

    print("# Generic AE Pure-Stack Study — Results Analysis\n")

    # --- Compute abstract key takeaways ---
    last_round = df["search_round"].max()
    r_final = df[df["search_round"] == last_round]
    best_cfg_owa = r_final.groupby("config_name")["owa"].median().sort_values()
    best_name = best_cfg_owa.index[0]
    best_owa = best_cfg_owa.iloc[0]
    best_params = r_final[r_final["config_name"] == best_name]["n_params"].iloc[0]
    param_reduction = (1 - best_params / NBEATS_G_PARAMS) * 100
    ig_owa = PUBLISHED_BASELINES["NBEATS-I+G"]["owa"]

    print("## Abstract\n")
    print("This study evaluates GenericAE and BottleneckGenericAE as pure-stack "
          "architectures on M4-Yearly using successive-halving search. GenericAE "
          "replaces the standard FC backbone with an encoder-decoder structure, "
          "producing a rank-constrained latent representation before projecting "
          "to backcast and forecast outputs.\n")
    print("**Key Takeaways:**\n")
    print(f"- **Best configuration:** `{best_name}` achieves median OWA = **{best_owa:.4f}** "
          f"with {best_params:,.0f} parameters ({param_reduction:.0f}% fewer than NBEATS-G).")
    delta_ig = best_owa - ig_owa
    verdict = "beats" if delta_ig < -0.001 else "matches" if abs(delta_ig) < 0.005 else "trails"
    print(f"- **vs NBEATS-I+G ({ig_owa:.4f}):** {verdict} (Δ = {delta_ig:+.4f}).")
    n_r1 = df[df["search_round"] == 1]["config_name"].nunique()
    n_rf = r_final["config_name"].nunique()
    print(f"- **Search scope:** {n_r1} initial configs → {n_rf} survivors across "
          f"{total_rounds} rounds.")
    print(f"- **Block types tested:** {sorted(df['block_type'].unique())}")
    print(f"- **Total compute:** {total_time / 60:.1f} minutes across {total_runs} runs.\n")

    print(f"- **CSV:** `{CSV_PATH}`")
    print(f"- **Rows:** {total_runs} ({total_configs} unique configs, {total_rounds} rounds)")
    print()
    print("### Published Baselines (M4-Yearly, 30-stack)\n")
    rows = [{"Config": name, "OWA": f"{vals['owa']:.4f}", "sMAPE": f"{vals['smape']:.2f}",
             "Params": f"{vals['params']:,}"} for name, vals in PUBLISHED_BASELINES.items()]
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()

    section("1. Successive Halving Funnel")
    funnel_summary(df)

    for r in sorted(df["search_round"].unique()):
        section(f"2.{r} Round {r} Leaderboard")
        round_leaderboard(df, r)

    section("3. Hyperparameter Marginals (Round 1 — Full Grid)")
    hyperparameter_marginals(df)

    section("3b. Selecting the Optimal Latent Dimension")
    latent_dim_discussion(df)

    section("4. Stability Analysis (OWA Spread Across Seeds)")
    stability_analysis(df)

    section("5. Round-over-Round Progression (Final Configs)")
    round_progression(df)

    section("6. Parameter Efficiency (Final Round)")
    param_efficiency(df)

    section("7. Statistical Significance vs NBEATS-I+G")
    statistical_significance(df)

    section("8. Training Stability (Divergence / Stopping)")
    training_stability(df)


def hyperparameter_marginals(df):
    """Marginal effect of each hyperparameter on median OWA (Round 1 = full grid)."""
    r1 = df[df["search_round"] == 1]
    print(f"Round 1 provides a balanced factorial grid ({len(r1)} rows, "
          f"{r1['config_name'].nunique()} configs) for unbiased marginal estimates.\n")
    for col, label in [
        ("block_type", "Block Type"),
        ("latent_dim_cfg", "Latent Dim"),
        ("thetas_dim_cfg", "Thetas Dim"),
        ("active_g", "active_g"),
    ]:
        if col not in r1.columns:
            continue
        grp = (
            r1.groupby(col)
            .agg(med_owa=("owa", "median"), mean_owa=("owa", "mean"),
                 std_owa=("owa", "std"), n=("owa", "count"),
                 med_params=("n_params", "median"))
            .sort_values("med_owa")
        )
        print(f"### {label}\n")
        rows = []
        for val, r in grp.iterrows():
            rows.append({label: str(val), "Med OWA": f"{r.med_owa:.4f}",
                         "Mean OWA": f"{r.mean_owa:.4f}", "Std": f"{r.std_owa:.4f}",
                         "N": int(r.n), "Med Params": f"{r.med_params:,.0f}"})
        print(pd.DataFrame(rows).to_markdown(index=False))
        best_val = grp.index[0]
        worst_val = grp.index[-1]
        gap = grp.iloc[-1].med_owa - grp.iloc[0].med_owa
        llm_ctx = {
            "parameter_name": col,
            "best_value": str(best_val),
            "best_owa": float(grp.iloc[0].med_owa),
            "worst_value": str(worst_val),
            "worst_owa": float(grp.iloc[-1].med_owa),
            "delta": float(gap),
            "all_values": [
                {"value": str(v), "med_owa": float(r.med_owa)}
                for v, r in grp.iterrows()
            ],
        }
        llm_text = generate_commentary("hyperparameter_marginal", llm_ctx) if _LLM else None
        if llm_text:
            print(f"\n{llm_text}\n")
        else:
            print(f"\n**{best_val}** leads with the lowest median OWA; "
                  f"the gap to the worst level ({worst_val}) is {gap:.4f}.\n")


def latent_dim_discussion(df):
    """Data-driven discussion on selecting the optimal latent dimension."""
    col = "latent_dim_cfg"
    r1 = df[df["search_round"] == 1]
    if col not in r1.columns or r1[col].dropna().empty:
        print("(latent_dim_cfg column not found or empty — skipping)\n")
        return

    valid = r1.dropna(subset=[col])
    grp = (
        valid.groupby(col)
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

    print("The **latent dimension** sets the information bottleneck width in the "
          "AERootBlock backbone used by GenericAE and BottleneckGenericAE. The "
          "encoder compresses each block's input along the path "
          "`backcast_length → units/2 → latent_dim`, and the decoder expands it "
          "back via `latent_dim → units/2 → units`. The head layers then project "
          "to backcast and forecast outputs. A smaller latent_dim enforces "
          "stronger compression (regularisation), while a larger value preserves "
          "more information at the risk of overfitting.\n")

    bcl = int(valid["backcast_length"].iloc[0]) if "backcast_length" in valid.columns else 30
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

    llm_ctx = {
        "parameter_name": "latent_dim",
        "architecture_name": "GenericAE",
        "backcast_length": bcl,
        "forecast_length": 6,
        "best_value": int(best_dim),
        "best_owa": float(best_owa),
        "worst_value": int(worst_dim),
        "worst_owa": float(worst_owa),
        "delta": float(delta),
        "all_values": [
            {"value": int(d), "med_owa": float(grp.loc[d, "med_owa"]),
             "std_owa": float(grp.loc[d, "std_owa"]),
             "med_params": float(grp.loc[d, "med_params"])}
            for d in dims
        ],
    }
    llm_text = generate_commentary("hyperparameter_discussion", llm_ctx) if _LLM else None
    if llm_text:
        print(f"\n{llm_text}")
    else:
        if best_dim == min(dims):
            print("The smallest bottleneck wins, suggesting aggressive compression "
                  "is beneficial. Since GenericAE uses direct linear projections to "
                  "target lengths (no structured basis), the backbone's inductive bias "
                  "comes entirely from the AE bottleneck. A narrow latent prevents the "
                  "network from memorising noise in the lookback window.\n")
        elif best_dim == max(dims):
            print("The widest bottleneck wins, indicating the generic head layers "
                  "need richer features from the backbone. Unlike structured blocks "
                  "(Trend, Seasonality) that constrain the output space, GenericAE's "
                  "direct projection benefits from a larger latent representation.\n")
        else:
            print("A mid-range bottleneck strikes the best balance. Too narrow forces "
                  "excessive information loss; too wide provides diminishing returns "
                  "as the generic head already has unconstrained capacity.\n")

        print("**Practical recommendation:** Use `latent_dim = "
              f"{int(best_dim)}` for GenericAE / BottleneckGenericAE stacks on "
              "M4-Yearly. For longer horizons or more complex datasets, consider "
              "scaling latent_dim proportionally to backcast_length "
              "(e.g. latent_dim ≈ backcast_length / 5–10) and re-evaluating "
              "via a small grid search.\n")


def stability_analysis(df):
    print("Stability is measured by the OWA spread (max − min) across random seeds for each configuration.\n")
    last_spread = None
    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        spread = (
            rdf.groupby("config_name")["owa"]
            .agg(["median", "min", "max", "std"])
            .assign(range=lambda x: x["max"] - x["min"])
        )
        last_spread = spread
        print(f"### Round {r}\n")
        print(f"- **Mean spread (max−min):** {spread['range'].mean():.4f}")
        print(f"- **Max spread:** {spread['range'].max():.4f} ({spread['range'].idxmax()})")
        print(f"- **Mean std:** {spread['std'].mean():.4f}\n")
        most_stable = spread.sort_values("range").head(5).reset_index()
        most_stable.columns = ["Config", "Median OWA", "Min", "Max", "Std", "Range"]
        print("**Most stable configs:**\n")
        print(most_stable[["Config", "Median OWA", "Range", "Std"]].to_markdown(index=False, floatfmt=".4f"))
        print()

    if last_spread is not None:
        llm_ctx = {
            "mean_spread": float(last_spread["range"].mean()),
            "max_spread": float(last_spread["range"].max()),
            "most_stable": list(last_spread.sort_values("range").head(3).index),
            "most_volatile": list(last_spread.sort_values("range", ascending=False).head(3).index),
        }
        llm_text = generate_commentary("stability_analysis", llm_ctx) if _LLM else None
        if llm_text:
            print(llm_text)
            print()


def param_efficiency(df):
    last_round = df["search_round"].max()
    rdf = df[df["search_round"] == last_round]
    agg = (
        rdf.groupby("config_name")
        .agg(med_owa=("owa", "median"), n_params=("n_params", "first"),
             block_type=("block_type", "first"))
        .sort_values("n_params")
    )
    print(f"Reference: NBEATS-G 30-stack baseline has {NBEATS_G_PARAMS:,} parameters.\n")
    # Identify Pareto front (lower params AND lower OWA)
    pareto = []
    best_owa_so_far = float("inf")
    for name, r in agg.iterrows():
        if r.med_owa < best_owa_so_far:
            pareto.append(name)
            best_owa_so_far = r.med_owa
    rows = []
    for name, r in agg.iterrows():
        reduction = (1 - r.n_params / NBEATS_G_PARAMS) * 100
        pareto_tag = "◀ PARETO" if name in pareto else ""
        rows.append({"Config": name, "Block": str(r.block_type),
                      "Params": f"{r.n_params:,.0f}", "Reduction": f"{reduction:.1f}%",
                      "Med OWA": f"{r.med_owa:.4f}", "Pareto": pareto_tag})
    print(pd.DataFrame(rows).to_markdown(index=False))
    best_cfg_name = agg.iloc[agg["med_owa"].values.argmin()].name if hasattr(agg, "name") else agg["med_owa"].idxmin()
    best_row = agg.loc[best_cfg_name]
    llm_ctx = {
        "baseline_params": NBEATS_G_PARAMS,
        "best_config": {
            "name": str(best_cfg_name),
            "params": int(best_row.n_params),
            "owa": float(best_row.med_owa),
            "reduction_pct": float((1 - best_row.n_params / NBEATS_G_PARAMS) * 100),
        },
        "configs": [
            {"name": str(n), "params": int(r.n_params), "owa": float(r.med_owa),
             "reduction_pct": float((1 - r.n_params / NBEATS_G_PARAMS) * 100),
             "pareto": n in pareto}
            for n, r in agg.iterrows()
        ],
    }
    llm_text = generate_commentary("param_efficiency", llm_ctx) if _LLM else None
    if llm_text:
        print(f"\n{llm_text}")
    else:
        print(f"\nConfigurations on the Pareto frontier achieve the best OWA for their "
              f"parameter budget — they cannot be improved on one axis without regressing on the other.\n")


def statistical_significance(df):
    """Mann-Whitney U test of final-round configs vs NBEATS-I+G OWA reference."""
    last_round = df["search_round"].max()
    rdf = df[df["search_round"] == last_round]
    ref_owa = PUBLISHED_BASELINES["NBEATS-I+G"]["owa"]

    # Load block benchmark if available for true reference distribution
    ref_samples = None
    if os.path.exists(BASELINE_CSV):
        bdf = pd.read_csv(BASELINE_CSV)
        bdf_ig = bdf[(bdf["config_name"] == "NBEATS-I+G") & (bdf["period"] == "Yearly")]
        if not bdf_ig.empty:
            ref_samples = pd.to_numeric(bdf_ig["owa"], errors="coerce").dropna().values

    print(f"Reference: NBEATS-I+G OWA = {ref_owa:.4f}")
    if ref_samples is not None:
        print(f"Using {len(ref_samples)} empirical reference samples from block benchmark.\n")
    else:
        print("(No empirical reference available; skipping Mann-Whitney U test.)\n")
        return

    print("A one-sided Mann-Whitney U test (alternative = 'less') checks whether each "
          "configuration's OWA distribution is stochastically lower than NBEATS-I+G.\n")
    rows = []
    configs = rdf.groupby("config_name")["owa"].median().sort_values()
    for name, med_owa in configs.items():
        cfg_vals = pd.to_numeric(rdf[rdf["config_name"] == name]["owa"], errors="coerce").dropna().values
        if len(cfg_vals) < 2:
            continue
        stat, p = stats.mannwhitneyu(cfg_vals, ref_samples, alternative="less")
        better = "YES ✓" if med_owa < ref_owa else "no"
        rows.append({"Config": name, "Med OWA": f"{med_owa:.4f}", "p-value": f"{p:.4f}",
                      "Sig": sig_stars(p), "Better?": better})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def training_stability(df):
    print("Training stability tracks divergence rates and early-stopping behaviour across rounds.\n")
    rows = []
    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        n_total = len(rdf)
        n_div = rdf["diverged"].sum()
        early_stopped = (rdf["stopping_reason"] == "EARLY_STOPPED").sum()
        max_epochs = (rdf["stopping_reason"] == "MAX_EPOCHS").sum()
        rows.append({"Round": r, "Runs": n_total,
                      "Diverged": f"{n_div} ({n_div/n_total*100:.1f}%)",
                      "Early Stopped": f"{early_stopped} ({early_stopped/n_total*100:.1f}%)",
                      "Max Epochs": f"{max_epochs} ({max_epochs/n_total*100:.1f}%)"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def round_progression(df):
    rounds = sorted(df["search_round"].unique())
    if len(rounds) < 2:
        print("(Need ≥ 2 rounds for progression analysis.)\n")
        return
    last_round = rounds[-1]
    final_configs = set(df[df["search_round"] == last_round]["config_name"].unique())
    print(f"Tracking the {len(final_configs)} configurations that survived to Round {last_round} "
          f"across all earlier rounds.\n")
    medians = {}
    for r in rounds:
        rdf = df[(df["search_round"] == r) & (df["config_name"].isin(final_configs))]
        if not rdf.empty:
            medians[r] = rdf.groupby("config_name")["owa"].median()
    prog = pd.DataFrame(medians)
    prog.columns = [f"R{r}" for r in prog.columns]
    if len(prog.columns) >= 2:
        first_col, last_col = prog.columns[0], prog.columns[-1]
        prog["Δ"] = prog[last_col] - prog[first_col]
        prog["Δ%"] = (prog["Δ"] / prog[first_col] * 100).round(1)
    prog = prog.sort_values(prog.columns[-2] if len(prog.columns) >= 2 else prog.columns[0])
    print(prog.round(4).reset_index().rename(columns={"index": "Config"}).to_markdown(index=False))
    if "Δ" in prog.columns:
        improved = (prog["Δ"] < 0).sum()
        llm_ctx = {
            "n_improved": int(improved),
            "n_total": int(len(prog)),
            "progression_data": [
                {"config": str(name), "delta": float(row["Δ"]), "delta_pct": float(row["Δ%"])}
                for name, row in prog.iterrows()
                if "Δ" in row.index
            ],
        }
        llm_text = generate_commentary("round_progression", llm_ctx) if _LLM else None
        if llm_text:
            print(f"\n{llm_text}\n")
        else:
            print(f"\n{improved} of {len(prog)} configs improved with additional training epochs. "
                  f"Mean Δ = {prog['Δ'].mean():.4f}.\n")

if __name__ == "__main__":
    main()

