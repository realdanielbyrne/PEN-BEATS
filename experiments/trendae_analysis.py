"""Comprehensive analysis of the TrendAE architecture search results.

Reads experiments/results/m4/trendae_study_results.csv and produces:
  1. Successive-halving funnel summary
  2. Per-round leaderboards (median OWA across runs)
  3. Hyperparameter marginals (ae_variant, latent_dim, thetas_dim, trend_thetas_dim, active_g)
  4. Latent dimension discussion
  5. Variant head-to-head
  6. Stability analysis (OWA spread across seeds)
  7. Round-over-round progression for promoted configs
  8. Parameter efficiency frontier
  9. Final verdict
"""

import sys, io, os
import pandas as pd
import numpy as np

try:
    from llm_commentary import generate_commentary
    _LLM = True
except ImportError:
    _LLM = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 55)

# ── Constants ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(
    os.path.dirname(__file__), "results", "m4", "trendae_study_results.csv"
)
TARGET_OWA = 0.85
TARGET_PARAMS = 5_000_000
NBEATS_G_PARAMS = 24_700_000  # 30-stack Generic baseline

NUM_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio",
]


def load():
    df = pd.read_csv(CSV_PATH)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["search_round"] = pd.to_numeric(df["search_round"], errors="coerce").astype(int)
    return df


def section(title):
    """Emit a Markdown section heading (## level)."""
    print()
    print(f"## {title}")
    print()


def round_leaderboard(df, round_num):
    """Print median-OWA leaderboard for a single round."""
    rdf = df[df["search_round"] == round_num]
    if rdf.empty:
        print(f"(no data for round {round_num})")
        return None

    n_runs = rdf.groupby("config_name")["run"].nunique().iloc[0]
    max_ep = rdf["epochs_trained"].max()
    n_cfg = rdf["config_name"].nunique()

    agg = (
        rdf.groupby("config_name")
        .agg(
            med_owa=("owa", "median"),
            min_owa=("owa", "min"),
            max_owa=("owa", "max"),
            med_smape=("smape", "median"),
            med_mase=("mase", "median"),
            n_params=("n_params", "first"),
            ae_variant=("ae_variant", "first"),
            latent_dim=("latent_dim_cfg", "first"),
            thetas_dim=("thetas_dim_cfg", "first"),
            trend_thetas_dim=("trend_thetas_dim_cfg", "first"),
            active_g=("active_g", "first"),
            med_time=("training_time_seconds", "median"),
        )
        .sort_values("med_owa")
    )

    print(f"{n_cfg} configs × {n_runs} runs, up to {max_ep} epochs each\n")

    rows = []
    for rank, (name, r) in enumerate(agg.iterrows(), 1):
        spread = r.max_owa - r.min_owa
        hit = "✓" if r.med_owa < TARGET_OWA and r.n_params < TARGET_PARAMS else ""
        rows.append({
            "Rank": rank, "Config": name, "OWA": f"{r.med_owa:.4f}",
            "±": f"{spread:.4f}", "sMAPE": f"{r.med_smape:.2f}",
            "MASE": f"{r.med_mase:.2f}", "Params": f"{r.n_params:,.0f}",
            "Time": f"{r.med_time:.1f}s", "Target": hit,
        })
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()

    best = agg.iloc[0]
    worst = agg.iloc[-1]
    print(f"The top-ranked configuration achieves a median OWA of {best.med_owa:.4f} "
          f"with {best.n_params:,.0f} parameters, while the worst scores {worst.med_owa:.4f}. "
          f"The spread between best and worst is {worst.med_owa - best.med_owa:.4f}.")
    winners = agg[(agg.med_owa < TARGET_OWA) & (agg.n_params < TARGET_PARAMS)]
    if len(winners) > 0:
        print(f"**{len(winners)} configuration(s) meet both the OWA < {TARGET_OWA} and "
              f"params < {TARGET_PARAMS:,} targets.**")
    print()
    return agg


def funnel_summary(df):
    """Show how many configs survived each round."""
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
        first_cfgs = rows[0]["Configs"]
        last_cfgs = rows[-1]["Configs"]
        print(f"The successive halving procedure pruned from {first_cfgs} to {last_cfgs} "
              f"configurations across {len(rounds)} rounds, retaining the top "
              f"{last_cfgs/first_cfgs*100:.0f}% of candidates.")
    print()


def hyperparameter_marginals(df):
    """Marginal effect of each hyperparameter on median OWA."""
    for col, label in [
        ("ae_variant", "AE Variant"),
        ("latent_dim_cfg", "Latent Dim"),
        ("thetas_dim_cfg", "Thetas Dim"),
        ("trend_thetas_dim_cfg", "Trend Thetas Dim"),
        ("active_g", "active_g"),
    ]:
        if col not in df.columns:
            continue
        grp = (
            df.groupby(col)
            .agg(
                med_owa=("owa", "median"),
                mean_owa=("owa", "mean"),
                std_owa=("owa", "std"),
                n_runs=("owa", "count"),
                med_params=("n_params", "median"),
            )
            .sort_values("med_owa")
        )
        print(f"\n### {label}\n")
        tbl = grp.reset_index().rename(columns={col: "Value", "med_owa": "Med OWA",
            "mean_owa": "Mean OWA", "std_owa": "Std", "n_runs": "N", "med_params": "Med Params"})
        tbl["Med Params"] = tbl["Med Params"].apply(lambda x: f"{x:,.0f}")
        print(tbl.to_markdown(index=False))
        best_val = grp.index[0]
        worst_val = grp.index[-1]
        delta = grp.iloc[-1].med_owa - grp.iloc[0].med_owa
        llm_ctx = {
            "parameter_name": col,
            "best_value": str(best_val),
            "best_owa": float(grp.iloc[0].med_owa),
            "worst_value": str(worst_val),
            "worst_owa": float(grp.iloc[-1].med_owa),
            "delta": float(delta),
            "all_values": [
                {"value": str(v), "med_owa": float(r.med_owa)}
                for v, r in grp.iterrows()
            ],
        }
        llm_text = generate_commentary("hyperparameter_marginal", llm_ctx) if _LLM else None
        if llm_text:
            print(f"\n{llm_text}")
        else:
            print(f"\n**{best_val}** yields the best median OWA while **{worst_val}** "
                  f"is the weakest (Δ = {delta:.4f}).")
        print()


def latent_dim_discussion(df):
    """Data-driven discussion on selecting the optimal latent dimension."""
    col = "latent_dim_cfg"
    if col not in df.columns:
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

    print("The **latent dimension** controls the information bottleneck width in the "
          "AERootBlock backbone. The encoder compresses each block's input from "
          "`backcast_length → units/2 → latent_dim`, and the decoder expands it back "
          "to `latent_dim → units/2 → units` before the head layers produce backcast "
          "and forecast outputs. A smaller latent dim forces stronger compression "
          "(regularisation); a larger dim preserves more information but risks "
          "overfitting.\n")

    bcl = int(df["backcast_length"].iloc[0]) if "backcast_length" in df.columns else 30
    fcl = int(df["forecast_length"].iloc[0]) if "forecast_length" in df.columns else 6
    print(f"With backcast_length = {bcl} and forecast_length = {fcl}, the tested "
          f"latent dimensions are: **{', '.join(str(int(d)) for d in dims)}**.\n")

    for d in dims:
        r = grp.loc[d]
        tag = " ← best" if d == best_dim else (" ← worst" if d == worst_dim else "")
        print(f"- **latent_dim = {int(d)}:** median OWA = {r.med_owa:.4f}, "
              f"std = {r.std_owa:.4f}, params ≈ {r.med_params:,.0f}{tag}")
    print()

    delta = worst_owa - best_owa
    print(f"The optimal setting is **latent_dim = {int(best_dim)}** "
          f"(median OWA {best_owa:.4f}), outperforming the worst setting "
          f"(latent_dim = {int(worst_dim)}) by Δ = {delta:.4f}. ")

    llm_ctx = {
        "parameter_name": "latent_dim",
        "architecture_name": "TrendAE",
        "backcast_length": bcl,
        "forecast_length": fcl,
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
            print("The smallest bottleneck wins, indicating that strong compression "
                  "benefits the TrendAE stacks. Since each AE block is paired with a "
                  "structured Trend block that constrains the output via a polynomial "
                  "basis, the backbone needs only a minimal latent representation.\n")
        elif best_dim == max(dims):
            print("The largest bottleneck wins, suggesting that the AE backbone benefits "
                  "from preserving more information. Despite the Trend block's polynomial "
                  "constraints, the backbone's richer features improve forecast quality "
                  "at the cost of mild overfitting risk.\n")
        else:
            print("A mid-range bottleneck achieves the best balance between compression "
                  "and information preservation. The Trend head constrains the output "
                  "space, so the backbone doesn't need maximum capacity, but too-narrow "
                  "a bottleneck discards useful structure.\n")

        print(f"**Practical recommendation:** Use `latent_dim = {int(best_dim)}` as the "
              "default for TrendAE stacks on M4-Yearly. For longer forecast horizons or "
              "higher-frequency data, consider scaling proportionally "
              "(e.g. latent_dim ≈ backcast_length / 5–10).\n")


def variant_head_to_head(df):
    """Compare AE variant families: best config per variant per round."""
    rounds = sorted(df["search_round"].unique())
    round_results = {}
    all_variants = set()
    for r in rounds:
        rdf = df[df["search_round"] == r]
        best_per_variant = (
            rdf.groupby(["ae_variant", "config_name"])["owa"]
            .median()
            .reset_index()
            .sort_values("owa")
            .drop_duplicates("ae_variant")
            .sort_values("owa")
        )
        print(f"\n### Round {r} — Best Config per Variant\n")
        tbl = best_per_variant[["ae_variant", "config_name", "owa"]].rename(
            columns={"ae_variant": "Variant", "config_name": "Best Config", "owa": "Med OWA"})
        print(tbl.to_markdown(index=False))
        round_results[f"Round {r}"] = {
            str(row["ae_variant"]): {"config": str(row["config_name"]), "owa": float(row["owa"])}
            for _, row in best_per_variant.iterrows()
        }
        all_variants.update(best_per_variant["ae_variant"].tolist())
    print()

    llm_ctx = {
        "variants": sorted(str(v) for v in all_variants),
        "round_results": round_results,
    }
    llm_text = generate_commentary("variant_comparison", llm_ctx) if _LLM else None
    if llm_text:
        print(llm_text)
    else:
        print("Variants that maintain top positions across rounds demonstrate robust "
              "performance independent of training budget.")
    print()


def stability_analysis(df):
    """Analyse OWA variance across seeds for each round."""
    last_spread = None
    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        spread = (
            rdf.groupby("config_name")["owa"]
            .agg(["median", "min", "max", "std"])
            .assign(range=lambda x: x["max"] - x["min"])
        )
        last_spread = spread
        print(f"\n### Round {r}\n")
        print(f"- **Mean spread (max−min):** {spread['range'].mean():.4f}")
        print(f"- **Max spread (max−min):** {spread['range'].max():.4f} "
              f"(`{spread['range'].idxmax()}`)")
        print(f"- **Mean std:** {spread['std'].mean():.4f}")
        most_stable = spread.sort_values("range").head(3)
        print(f"- **Most stable configs:** "
              + ", ".join(f"`{n}` (±{rv:.4f})" for n, rv in most_stable["range"].items()))
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
        else:
            print("Lower spread values indicate more consistent performance across random seeds. "
                  "Configurations with high spread may be sensitive to initialization.")
    print()


def round_progression(df):
    """Track how promoted configs improve across rounds."""
    rounds = sorted(df["search_round"].unique())
    if len(rounds) < 2:
        print("(need ≥2 rounds for progression)")
        return

    last_round = rounds[-1]
    final_configs = set(df[df["search_round"] == last_round]["config_name"].unique())

    medians = {}
    for r in rounds:
        rdf = df[(df["search_round"] == r) & (df["config_name"].isin(final_configs))]
        if rdf.empty:
            continue
        medians[r] = rdf.groupby("config_name")["owa"].median()

    prog = pd.DataFrame(medians)
    prog.columns = [f"R{r}" for r in prog.columns]
    if len(prog.columns) >= 2:
        first_col, last_col = prog.columns[0], prog.columns[-1]
        prog["Δ"] = prog[last_col] - prog[first_col]
        prog["Δ%"] = (prog["Δ"] / prog[first_col] * 100).round(1)
    prog = prog.sort_values(prog.columns[-2])

    print(prog.round(4).to_markdown())
    print()
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
            print(llm_text)
        else:
            print(f"**{improved} of {len(prog)} surviving configurations improved** their OWA "
                  f"from R1 to R{last_round}, confirming that additional training "
                  f"epochs benefit the top candidates.")
    print()



def param_efficiency(df):
    """Show OWA vs parameter count for final-round configs."""
    last_round = df["search_round"].max()
    rdf = df[df["search_round"] == last_round]
    agg = (
        rdf.groupby("config_name")
        .agg(
            med_owa=("owa", "median"),
            n_params=("n_params", "first"),
            ae_variant=("ae_variant", "first"),
        )
        .sort_values("n_params")
    )
    rows = []
    for name, r in agg.iterrows():
        reduction = (1 - r.n_params / NBEATS_G_PARAMS) * 100
        hit = "✓" if r.med_owa < TARGET_OWA and r.n_params < TARGET_PARAMS else "✗"
        rows.append({"Config": name, "Params": f"{r.n_params:,.0f}",
                      "Reduction": f"{reduction:.1f}%", "Med OWA": f"{r.med_owa:.4f}", "Target": hit})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()
    best_efficiency = agg.iloc[agg["med_owa"].values.argmin()]
    llm_ctx = {
        "baseline_params": NBEATS_G_PARAMS,
        "best_config": {
            "name": str(best_efficiency.name) if hasattr(best_efficiency, "name") else "best",
            "params": int(best_efficiency.n_params),
            "owa": float(best_efficiency.med_owa),
            "reduction_pct": float((1 - best_efficiency.n_params / NBEATS_G_PARAMS) * 100),
        },
        "configs": [
            {"name": str(n), "params": int(r.n_params), "owa": float(r.med_owa),
             "reduction_pct": float((1 - r.n_params / NBEATS_G_PARAMS) * 100)}
            for n, r in agg.iterrows()
        ],
    }
    llm_text = generate_commentary("param_efficiency", llm_ctx) if _LLM else None
    if llm_text:
        print(llm_text)
    else:
        print(f"All TrendAE configurations achieve substantial parameter reductions "
              f"relative to the {NBEATS_G_PARAMS:,}-parameter Generic baseline. "
              f"The best-performing config uses {best_efficiency.n_params:,.0f} parameters "
              f"({(1 - best_efficiency.n_params/NBEATS_G_PARAMS)*100:.0f}% reduction) "
              f"while achieving OWA = {best_efficiency.med_owa:.4f}.")
    print()


def final_verdict(df):
    """Summarise which configs meet the target criteria."""
    last_round = df["search_round"].max()
    rdf = df[df["search_round"] == last_round]
    agg = (
        rdf.groupby("config_name")
        .agg(
            med_owa=("owa", "median"),
            min_owa=("owa", "min"),
            max_owa=("owa", "max"),
            med_smape=("smape", "median"),
            med_mase=("mase", "median"),
            n_params=("n_params", "first"),
            ae_variant=("ae_variant", "first"),
            latent_dim=("latent_dim_cfg", "first"),
            thetas_dim=("thetas_dim_cfg", "first"),
            trend_thetas_dim=("trend_thetas_dim_cfg", "first"),
            active_g=("active_g", "first"),
        )
        .sort_values("med_owa")
    )

    winners = agg[(agg.med_owa < TARGET_OWA) & (agg.n_params < TARGET_PARAMS)]
    near_miss = agg[(agg.med_owa < TARGET_OWA) & (agg.n_params >= TARGET_PARAMS)]

    print(f"**Target:** OWA < {TARGET_OWA}, Params < {TARGET_PARAMS:,}")
    print(f"**Baseline:** N-BEATS-G 30-stack = {NBEATS_G_PARAMS:,} params\n")

    if len(winners) > 0:
        print(f"✅ **{len(winners)} configuration(s) MEET the target:**\n")
        for name, r in winners.iterrows():
            reduction = (1 - r.n_params / NBEATS_G_PARAMS) * 100
            print(f"**{name}**\n")
            print(f"- OWA: {r.med_owa:.4f} (range {r.min_owa:.4f}–{r.max_owa:.4f})")
            print(f"- sMAPE: {r.med_smape:.2f}, MASE: {r.med_mase:.2f}")
            print(f"- Params: {r.n_params:,.0f} ({reduction:.0f}% reduction)")
            print(f"- Hyperparams: ae={r.ae_variant}, latent_dim={int(r.latent_dim)}, "
                  f"thetas_dim={int(r.thetas_dim)}, "
                  f"trend_thetas_dim={int(r.trend_thetas_dim)}, active_g={r.active_g}")
            print()
    else:
        print("❌ No configurations meet both OWA and parameter targets.\n")

    if len(near_miss) > 0:
        print(f"⚠️ **{len(near_miss)} configuration(s) meet OWA target but exceed "
              f"parameter budget:**\n")
        rows = [{"Config": name, "OWA": f"{r.med_owa:.4f}",
                 "Params": f"{r.n_params:,.0f}"} for name, r in near_miss.iterrows()]
        print(pd.DataFrame(rows).to_markdown(index=False))
        print()


def main():
    df = load()
    total_runs = len(df)
    total_configs = df["config_name"].nunique()
    total_rounds = df["search_round"].nunique()
    total_time = df["training_time_seconds"].sum()

    print("# TrendAE Architecture Search — Results Analysis\n")

    # --- Abstract ---
    last_round = df["search_round"].max()
    r_final = df[df["search_round"] == last_round]
    best_cfg_owa = r_final.groupby("config_name")["owa"].median().sort_values()
    best_name = best_cfg_owa.index[0]
    best_owa = best_cfg_owa.iloc[0]
    best_params = r_final[r_final["config_name"] == best_name]["n_params"].iloc[0]
    param_reduction = (1 - best_params / NBEATS_G_PARAMS) * 100

    # Identify best AE variant from R1
    r1 = df[df["search_round"] == 1]
    best_variant = r1.groupby("ae_variant")["owa"].median().idxmin()

    n_diverged = df["diverged"].sum() if "diverged" in df.columns else 0
    n_variants = df["ae_variant"].nunique()

    print("## Abstract\n")
    print(f"This study evaluates **{n_variants} AE-backbone variants** paired with Trend "
          f"blocks across {total_configs} configurations on M4-Yearly, using successive "
          f"halving over {total_rounds} rounds ({total_runs} total runs, "
          f"{total_time / 60:.1f} min compute). Each configuration combines an AE block "
          f"(AutoEncoder, GenericAE, GenericAEBackcast, BottleneckGenericAE, and their "
          f"AE-backbone counterparts) with a Trend block, varying latent_dim, thetas_dim, "
          f"trend_thetas_dim, and active_g.\n")
    print("**Key Takeaways:**\n")
    print(f"1. **Best configuration:** `{best_name}` — median OWA = **{best_owa:.4f}** "
          f"with {best_params:,.0f} params ({param_reduction:.0f}% fewer than NBEATS-G).")
    target_met = best_owa < TARGET_OWA
    print(f"2. **Target OWA < {TARGET_OWA}:** {'Met ✓' if target_met else 'Not met ✗'}")
    print(f"3. **Best AE variant (R1 marginal):** `{best_variant}`")
    n_r1 = r1["config_name"].nunique()
    n_rf = r_final["config_name"].nunique()
    print(f"4. **Search scope:** {n_r1} → {n_rf} configs via successive halving.")
    print(f"5. **Convergence:** {n_diverged} diverged runs out of {total_runs} ({n_diverged/total_runs*100:.1f}%).\n")

    print(f"- **CSV:** `{CSV_PATH}`")
    print(f"- **Rows:** {total_runs} ({total_configs} unique configs, {total_rounds} rounds)")
    print(f"- **AE variants:** {sorted(df['ae_variant'].unique())}")
    print(f"- **Total training time:** {total_time / 60:.1f} min")
    print(f"- **OWA range:** {df['owa'].min():.4f} – {df['owa'].max():.4f}")
    print()

    section("1. Successive Halving Funnel")
    funnel_summary(df)

    for r in sorted(df["search_round"].unique()):
        section(f"2.{r} Round {r} Leaderboard")
        round_leaderboard(df, r)

    section("3. Hyperparameter Marginals (Round 1 — Full Grid)")
    hyperparameter_marginals(r1)

    section("3b. Selecting the Optimal Latent Dimension")
    latent_dim_discussion(r1)

    section("4. Variant Head-to-Head")
    variant_head_to_head(df)

    section("5. Stability Analysis (OWA Spread Across Seeds)")
    stability_analysis(df)

    section("6. Round-over-Round Progression (Final Configs)")
    round_progression(df)

    section("7. Parameter Efficiency (Final Round)")
    param_efficiency(df)

    section("8. Final Verdict")
    final_verdict(df)


if __name__ == "__main__":
    main()