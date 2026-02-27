"""Comprehensive analysis of AE+Trend architecture search results.

Reads experiments/results/m4/ae_trend_search_results.csv and produces:
  1. Per-round leaderboards (median OWA across runs)
  2. Successive-halving funnel summary
  3. Hyperparameter marginals (ae_variant, latent_dim, thetas_dim, active_g)
  4. Parameter-efficiency frontier (OWA vs n_params)
  5. Stability analysis (OWA spread across seeds)
  6. Round-over-round improvement for promoted configs
  7. Final verdict against target criteria
"""

import sys, io, os
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 55)

# Track section numbering for Markdown heading levels
_section_counter = 0

# ── Constants ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(
    os.path.dirname(__file__), "results", "m4", "ae_trend_search_results.csv"
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
    # Ensure search_round is int
    df["search_round"] = pd.to_numeric(df["search_round"], errors="coerce").astype(int)
    return df


def section(title, char="="):
    """Emit a Markdown section heading (## level)."""
    global _section_counter
    _section_counter += 1
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
            active_g=("active_g", "first"),
            med_time=("training_time_seconds", "median"),
        )
        .sort_values("med_owa")
    )

    print(f"{n_cfg} configs × {n_runs} runs, {max_ep} epochs each\n")

    rows = []
    for rank, (name, r) in enumerate(agg.iterrows(), 1):
        spread = r.max_owa - r.min_owa
        hit = "✓" if r.med_owa < TARGET_OWA and r.n_params < TARGET_PARAMS else ""
        rows.append({
            "Rank": rank, "Config": name, "OWA": f"{r.med_owa:.3f}",
            "±": f"{spread:.3f}", "sMAPE": f"{r.med_smape:.2f}",
            "MASE": f"{r.med_mase:.2f}", "Params": f"{r.n_params:,.0f}",
            "Time": f"{r.med_time:.1f}s", "Target": hit,
        })
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()

    # Commentary
    best = agg.iloc[0]
    worst = agg.iloc[-1]
    print(f"The top-ranked configuration achieves a median OWA of {best.med_owa:.3f} "
          f"with {best.n_params:,.0f} parameters, while the worst scores {worst.med_owa:.3f}. "
          f"The spread between best and worst is {worst.med_owa - best.med_owa:.3f}.")
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
                      "Epochs": max_ep, "Best Med OWA": f"{best:.3f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()
    if len(rounds) >= 2:
        first_cfgs = rows[0]["Configs"]
        last_cfgs = rows[-1]["Configs"]
        print(f"The successive halving procedure pruned from {first_cfgs} to {last_cfgs} "
              f"configurations across {len(rounds)} rounds, retaining the top "
              f"{last_cfgs/first_cfgs*100:.0f}% of candidates. Each round increased "
              f"the training budget while eliminating weaker configurations.")
    print()


def hyperparameter_marginals(df):
    """Marginal effect of each hyperparameter on median OWA."""
    for col, label in [
        ("ae_variant", "AE Variant"),
        ("latent_dim_cfg", "Latent Dim (search)"),
        ("thetas_dim_cfg", "Thetas Dim (search)"),
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
        # Commentary
        best_val = grp.index[0]
        worst_val = grp.index[-1]
        delta = grp.iloc[-1].med_owa - grp.iloc[0].med_owa
        print(f"\n**{best_val}** yields the best median OWA while **{worst_val}** "
              f"is the weakest (Δ = {delta:.3f}).")
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
          "and forecast outputs. A smaller latent dim forces stronger compression, "
          "which acts as a regulariser but may discard useful signal; a larger latent "
          "dim preserves more information but risks overfitting.\n")

    print(f"Across this experiment (backcast_length = {int(df['backcast_length'].iloc[0])}, "
          f"forecast_length = {int(df['forecast_length'].iloc[0] if 'forecast_length' in df.columns else 6)}), "
          f"three latent dimensions were tested: **{', '.join(str(int(d)) for d in dims)}**.\n")

    # Per-dim summary
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

    # Interpretation
    if best_dim == min(dims):
        print("The smallest bottleneck wins, indicating that strong compression "
              "acts as beneficial regularisation for this architecture on M4-Yearly. "
              "The AE+Trend stack already has a structured trend head that constrains "
              "the forecast space, so the backbone needs only a narrow latent "
              "representation to capture residual patterns.\n")
    elif best_dim == max(dims):
        print("The largest bottleneck wins, suggesting that the AE backbone benefits "
              "from preserving more information before handing off to the trend head. "
              "The trend polynomial basis already provides strong inductive bias, "
              "so an over-compressed latent may strip useful structure.\n")
    else:
        print("A mid-range bottleneck achieves the best balance between compression "
              "and information preservation, suggesting diminishing returns at "
              "higher dimensions while the lowest dimension is too aggressive.\n")

    print("**Practical recommendation:** Use `latent_dim = "
          f"{int(best_dim)}` as the default for AE+Trend configurations on "
          "M4-Yearly. When adapting to datasets with longer backcast windows "
          "or more complex seasonal patterns, consider scaling latent_dim "
          "proportionally (e.g. latent_dim ≈ backcast_length / 5–10).\n")


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
                      "Reduction": f"{reduction:.1f}%", "Med OWA": f"{r.med_owa:.3f}", "Target": hit})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()
    best_efficiency = agg.iloc[agg["med_owa"].values.argmin()]
    print(f"All AE+Trend configurations achieve substantial parameter reductions "
          f"relative to the {NBEATS_G_PARAMS:,}-parameter Generic baseline. "
          f"The best-performing config uses {best_efficiency.n_params:,.0f} parameters "
          f"({(1 - best_efficiency.n_params/NBEATS_G_PARAMS)*100:.0f}% reduction) "
          f"while achieving OWA = {best_efficiency.med_owa:.3f}.")
    print()


def stability_analysis(df):
    """Analyse OWA variance across seeds for each round."""
    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        spread = (
            rdf.groupby("config_name")["owa"]
            .agg(["median", "min", "max", "std"])
            .assign(range=lambda x: x["max"] - x["min"])
        )
        print(f"\n### Round {r}\n")
        print(f"- **Mean spread (max−min):** {spread['range'].mean():.3f}")
        print(f"- **Max spread (max−min):** {spread['range'].max():.3f} "
              f"(`{spread['range'].idxmax()}`)")
        print(f"- **Mean std:** {spread['std'].mean():.3f}")
        most_stable = spread.sort_values("range").head(3)
        print(f"- **Most stable configs:** "
              + ", ".join(f"`{n}` (±{r:.3f})" for n, r in most_stable["range"].items()))
    print()
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

    print(prog.round(3).to_markdown())
    print()
    if "Δ" in prog.columns:
        improved = (prog["Δ"] < 0).sum()
        print(f"**{improved} of {len(prog)} surviving configurations improved** their OWA "
              f"from the first to the last round, confirming that additional training "
              f"epochs benefit the top candidates.")
    print()


def variant_head_to_head(df):
    """Compare AE variant families: best config per variant per round."""
    rounds = sorted(df["search_round"].unique())
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
    print()
    print("This head-to-head comparison reveals which AE backbone architectures are "
          "most competitive at each stage of the search. Variants that maintain top "
          "positions across rounds demonstrate robust performance independent of "
          "training budget.")
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
            print(f"- OWA: {r.med_owa:.3f} (range {r.min_owa:.3f}–{r.max_owa:.3f})")
            print(f"- sMAPE: {r.med_smape:.2f}, MASE: {r.med_mase:.2f}")
            print(f"- Params: {r.n_params:,.0f} ({reduction:.0f}% reduction)")
            print(f"- Hyperparams: ae={r.ae_variant}, latent_dim={r.latent_dim}, "
                  f"thetas_dim={r.thetas_dim}, active_g={r.active_g}")
            print()
    else:
        print("❌ No configurations meet both OWA and parameter targets.\n")

    if len(near_miss) > 0:
        print(f"⚠️ **{len(near_miss)} configuration(s) meet OWA target but exceed "
              f"parameter budget:**\n")
        rows = [{"Config": name, "OWA": f"{r.med_owa:.3f}",
                 "Params": f"{r.n_params:,.0f}"} for name, r in near_miss.iterrows()]
        print(pd.DataFrame(rows).to_markdown(index=False))
        print()


def main():
    df = load()
    total_runs = len(df)
    total_configs = df["config_name"].nunique()
    total_rounds = df["search_round"].nunique()
    total_time = df["training_time_seconds"].sum()

    print("# AE+Trend Architecture Search — Results Analysis\n")

    # --- Compute abstract key takeaways ---
    last_round = df["search_round"].max()
    r_final = df[df["search_round"] == last_round]
    best_cfg_owa = r_final.groupby("config_name")["owa"].median().sort_values()
    best_name = best_cfg_owa.index[0]
    best_owa = best_cfg_owa.iloc[0]
    best_params = r_final[r_final["config_name"] == best_name]["n_params"].iloc[0]
    param_reduction = (1 - best_params / NBEATS_G_PARAMS) * 100

    print("## Abstract\n")
    print("This study applies successive-halving architecture search to AE+Trend hybrid "
          "configurations on the M4-Yearly dataset. The AE+Trend stack pairs a compact "
          "autoencoder block with a trend block, aiming to match or beat established "
          "N-BEATS baselines at a fraction of the parameter cost.\n")
    print("**Key Takeaways:**\n")
    print(f"- **Best configuration:** `{best_name}` achieves median OWA = **{best_owa:.4f}** "
          f"with only {best_params:,.0f} parameters ({param_reduction:.0f}% fewer than NBEATS-G).")
    target_met = best_owa < TARGET_OWA
    print(f"- **Target OWA < {TARGET_OWA}:** {'Met ✓' if target_met else 'Not met'}")
    n_r1 = df[df["search_round"] == 1]["config_name"].nunique()
    n_rf = r_final["config_name"].nunique()
    print(f"- **Search scope:** {n_r1} initial configs narrowed to {n_rf} across "
          f"{total_rounds} rounds of successive halving.")
    print(f"- **Total compute:** {total_time / 60:.1f} minutes across {total_runs} training runs.\n")

    print(f"- **CSV:** `{CSV_PATH}`")
    print(f"- **Rows:** {total_runs} ({total_configs} unique configs, {total_rounds} rounds)")
    print(f"- **Total training time:** {total_time / 60:.1f} min")
    print()

    section("1. Successive Halving Funnel")
    funnel_summary(df)

    for r in sorted(df["search_round"].unique()):
        section(f"2.{r} Round {r} Leaderboard")
        round_leaderboard(df, r)

    section("3. Hyperparameter Marginals (Round 1 — Full Grid)")
    r1 = df[df["search_round"] == 1]
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
