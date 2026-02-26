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
    w = 90
    print()
    print(char * w)
    print(f"  {title}")
    print(char * w)


def round_leaderboard(df, round_num):
    """Print median-OWA leaderboard for a single round."""
    rdf = df[df["search_round"] == round_num]
    if rdf.empty:
        print(f"  (no data for round {round_num})")
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

    print(f"  {n_cfg} configs × {n_runs} runs, {max_ep} epochs each\n")
    hdr = (
        f"  {'#':>3s}  {'Config':<50s} {'OWA':>7s} {'±':>7s} "
        f"{'sMAPE':>7s} {'MASE':>7s} {'Params':>10s} {'Time':>6s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for rank, (name, r) in enumerate(agg.iterrows(), 1):
        spread = r.max_owa - r.min_owa
        hit = "✓" if r.med_owa < TARGET_OWA and r.n_params < TARGET_PARAMS else " "
        print(
            f"  {rank:3d}  {name:<50s} {r.med_owa:7.3f} {spread:7.3f} "
            f"{r.med_smape:7.2f} {r.med_mase:7.2f} {r.n_params:10,.0f} {r.med_time:5.1f}s {hit}"
        )
    return agg


def funnel_summary(df):
    """Show how many configs survived each round."""
    rounds = sorted(df["search_round"].unique())
    print(f"  {'Round':>6s}  {'Configs':>8s}  {'Runs':>6s}  {'Epochs':>7s}  {'Best Med OWA':>13s}")
    print("  " + "-" * 55)
    for r in rounds:
        rdf = df[df["search_round"] == r]
        n_cfg = rdf["config_name"].nunique()
        n_runs = len(rdf)
        max_ep = rdf["epochs_trained"].max()
        best = rdf.groupby("config_name")["owa"].median().min()
        print(f"  {r:6d}  {n_cfg:8d}  {n_runs:6d}  {max_ep:7d}  {best:13.3f}")


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
        print(f"\n  ── {label} ──")
        print(f"  {'Value':<25s} {'Med OWA':>8s} {'Mean':>8s} {'Std':>7s} {'N':>5s} {'Med Params':>12s}")
        print("  " + "-" * 70)
        for val, r in grp.iterrows():
            print(
                f"  {str(val):<25s} {r.med_owa:8.3f} {r.mean_owa:8.3f} "
                f"{r.std_owa:7.3f} {r.n_runs:5.0f} {r.med_params:12,.0f}"
            )


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
    print(f"  {'Config':<50s} {'Params':>10s} {'Reduction':>10s} {'Med OWA':>8s} {'Target':>7s}")
    print("  " + "-" * 90)
    for name, r in agg.iterrows():
        reduction = (1 - r.n_params / NBEATS_G_PARAMS) * 100
        hit = "✓" if r.med_owa < TARGET_OWA and r.n_params < TARGET_PARAMS else "✗"
        print(
            f"  {name:<50s} {r.n_params:10,.0f} {reduction:9.1f}% {r.med_owa:8.3f} {hit:>7s}"
        )


def stability_analysis(df):
    """Analyse OWA variance across seeds for each round."""
    for r in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == r]
        spread = (
            rdf.groupby("config_name")["owa"]
            .agg(["median", "min", "max", "std"])
            .assign(range=lambda x: x["max"] - x["min"])
        )
        print(f"\n  Round {r}:")
        print(f"    Mean spread (max-min): {spread['range'].mean():.3f}")
        print(f"    Max  spread (max-min): {spread['range'].max():.3f}  "
              f"({spread['range'].idxmax()})")
        print(f"    Mean std:              {spread['std'].mean():.3f}")
        most_stable = spread.sort_values("range").head(3)
        print(f"    Most stable configs:   "
              + ", ".join(f"{n} (±{r:.3f})" for n, r in most_stable["range"].items()))


def round_progression(df):
    """Track how promoted configs improve across rounds."""
    rounds = sorted(df["search_round"].unique())
    if len(rounds) < 2:
        print("  (need ≥2 rounds for progression)")
        return

    # Find configs present in the last round
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

    print(prog.round(3).to_string())


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
        print(f"\n  Round {r} — Best config per variant:")
        print(f"  {'Variant':<25s} {'Best Config':<45s} {'Med OWA':>8s}")
        print("  " + "-" * 80)
        for _, row in best_per_variant.iterrows():
            print(f"  {row.ae_variant:<25s} {row.config_name:<45s} {row.owa:8.3f}")


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

    print(f"  Target: OWA < {TARGET_OWA}, Params < {TARGET_PARAMS:,}")
    print(f"  Baseline: N-BEATS-G 30-stack = {NBEATS_G_PARAMS:,} params\n")

    if len(winners) > 0:
        print(f"  ✓ {len(winners)} config(s) MEET the target:\n")
        for name, r in winners.iterrows():
            reduction = (1 - r.n_params / NBEATS_G_PARAMS) * 100
            print(f"    {name}")
            print(f"      OWA:  {r.med_owa:.3f}  (range {r.min_owa:.3f}–{r.max_owa:.3f})")
            print(f"      sMAPE: {r.med_smape:.2f}   MASE: {r.med_mase:.2f}")
            print(f"      Params: {r.n_params:,.0f}  ({reduction:.0f}% reduction)")
            print(f"      Hyperparams: ae={r.ae_variant}, ld={r.latent_dim}, "
                  f"td={r.thetas_dim}, active_g={r.active_g}")
            print()
    else:
        print("  ✗ No configs meet both OWA and param targets.\n")

    if len(near_miss) > 0:
        print(f"  ~ {len(near_miss)} config(s) meet OWA target but exceed param budget:\n")
        for name, r in near_miss.iterrows():
            print(f"    {name}: OWA={r.med_owa:.3f}, Params={r.n_params:,.0f}")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    df = load()
    total_runs = len(df)
    total_configs = df["config_name"].nunique()
    total_rounds = df["search_round"].nunique()
    total_time = df["training_time_seconds"].sum()

    section("AE+TREND ARCHITECTURE SEARCH — RESULTS ANALYSIS")
    print(f"  CSV:     {CSV_PATH}")
    print(f"  Rows:    {total_runs}  ({total_configs} unique configs, "
          f"{total_rounds} rounds)")
    print(f"  Total training time: {total_time / 60:.1f} min")

    # 1. Funnel
    section("1. SUCCESSIVE HALVING FUNNEL")
    funnel_summary(df)

    # 2. Per-round leaderboards
    for r in sorted(df["search_round"].unique()):
        section(f"2.{r}  ROUND {r} LEADERBOARD")
        round_leaderboard(df, r)

    # 3. Hyperparameter marginals (Round 1 only — full grid)
    section("3. HYPERPARAMETER MARGINALS (Round 1 — full grid)")
    r1 = df[df["search_round"] == 1]
    hyperparameter_marginals(r1)

    # 4. Variant head-to-head
    section("4. VARIANT HEAD-TO-HEAD")
    variant_head_to_head(df)

    # 5. Stability
    section("5. STABILITY ANALYSIS (OWA spread across seeds)")
    stability_analysis(df)

    # 6. Round progression
    section("6. ROUND-OVER-ROUND PROGRESSION (final configs)")
    round_progression(df)

    # 7. Parameter efficiency
    section("7. PARAMETER EFFICIENCY (final round)")
    param_efficiency(df)

    # 8. Final verdict
    section("8. FINAL VERDICT", char="*")
    final_verdict(df)


if __name__ == "__main__":
    main()
