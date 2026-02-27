"""TrendAE architecture search analysis across runner-targeted datasets.

Behavior:
- Discovers datasets from run_trendae_study.TRENDAE_DATASETS
- Resolves CSV paths from run_trendae_study._search_csv_path
- Supports --dataset {all,<dataset>}
- Skips missing/empty CSVs
- Uses OWA when finite values exist, else falls back to best_val_loss
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

from run_trendae_study import TRENDAE_DATASETS, _search_csv_path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 55)

TARGET_OWA = 0.85
TARGET_PARAMS = 5_000_000
NBEATS_G_PARAMS = 24_700_000

NUM_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "latent_dim_cfg", "thetas_dim_cfg",
    "trend_thetas_dim_cfg", "search_round",
]


def section(title):
    print()
    print(f"## {title}")
    print()


def _resolve_datasets(dataset_arg):
    available = list(TRENDAE_DATASETS.keys())
    if dataset_arg == "all":
        return available
    return [dataset_arg]


def _load_df(csv_path):
    df = pd.read_csv(csv_path)
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "search_round" in df.columns:
        df["search_round"] = df["search_round"].fillna(1).astype(int)
    return df


def _metric_profile(df):
    owa = pd.to_numeric(df["owa"], errors="coerce") if "owa" in df.columns else pd.Series(dtype=float)
    has_owa = bool(len(owa) and owa.notna().any())
    primary_col = "owa" if has_owa else "best_val_loss"
    if primary_col not in df.columns:
        return None
    vals = pd.to_numeric(df[primary_col], errors="coerce")
    if not vals.notna().any():
        return None
    return {
        "has_owa": has_owa,
        "primary_col": primary_col,
        "primary_label": "OWA" if has_owa else "best_val_loss",
        "target_enabled": has_owa,
    }


def funnel_summary(df, m):
    rounds = sorted(df["search_round"].dropna().unique())
    rows = []
    for r in rounds:
        rdf = df[df["search_round"] == r]
        n_cfg = rdf["config_name"].nunique()
        n_runs = len(rdf)
        max_ep = rdf["epochs_trained"].max() if "epochs_trained" in rdf.columns else np.nan
        best = rdf.groupby("config_name")[m["primary_col"]].median().min()
        rows.append(
            {
                "Round": int(r),
                "Configs": int(n_cfg),
                "Runs": int(n_runs),
                "Epochs": int(max_ep) if pd.notna(max_ep) else "N/A",
                f"Best Med {m['primary_label']}": f"{best:.4f}",
            }
        )
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def round_leaderboard(df, round_num, m):
    rdf = df[df["search_round"] == round_num]
    if rdf.empty:
        print(f"(no data for round {round_num})")
        return None

    agg = (
        rdf.groupby("config_name")
        .agg(
            med_metric=(m["primary_col"], "median"),
            min_metric=(m["primary_col"], "min"),
            max_metric=(m["primary_col"], "max"),
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
        .sort_values("med_metric")
    )

    n_runs = rdf.groupby("config_name")["run"].nunique().iloc[0] if "run" in rdf.columns else 0
    max_ep = rdf["epochs_trained"].max() if "epochs_trained" in rdf.columns else np.nan
    print(f"{rdf['config_name'].nunique()} configs x {n_runs} runs, up to {max_ep} epochs each\n")

    rows = []
    for rank, (name, row) in enumerate(agg.iterrows(), 1):
        spread = row.max_metric - row.min_metric
        hit = ""
        if m["target_enabled"]:
            hit = "YES" if row.med_metric < TARGET_OWA and row.n_params < TARGET_PARAMS else ""
        rows.append(
            {
                "Rank": rank,
                "Config": name,
                m["primary_label"]: f"{row.med_metric:.4f}",
                "Spread": f"{spread:.4f}",
                "sMAPE": f"{row.med_smape:.2f}" if pd.notna(row.med_smape) else "N/A",
                "MASE": f"{row.med_mase:.2f}" if pd.notna(row.med_mase) else "N/A",
                "Params": f"{row.n_params:,.0f}" if pd.notna(row.n_params) else "N/A",
                "Time": f"{row.med_time:.1f}s" if pd.notna(row.med_time) else "N/A",
                "Target": hit,
            }
        )
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()

    best = agg.iloc[0]
    worst = agg.iloc[-1]
    print(
        f"Top config median {m['primary_label']}={best.med_metric:.4f}; "
        f"worst={worst.med_metric:.4f}; delta={worst.med_metric - best.med_metric:.4f}."
    )
    print()
    return agg


def hyperparameter_marginals(df, m):
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
                med_metric=(m["primary_col"], "median"),
                mean_metric=(m["primary_col"], "mean"),
                std_metric=(m["primary_col"], "std"),
                n_runs=(m["primary_col"], "count"),
                med_params=("n_params", "median"),
            )
            .sort_values("med_metric")
        )
        print(f"\n### {label}\n")
        tbl = grp.reset_index().rename(
            columns={
                col: "Value",
                "med_metric": f"Med {m['primary_label']}",
                "mean_metric": f"Mean {m['primary_label']}",
                "std_metric": "Std",
                "n_runs": "N",
                "med_params": "Med Params",
            }
        )
        if "Med Params" in tbl.columns:
            tbl["Med Params"] = tbl["Med Params"].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
            )
        print(tbl.to_markdown(index=False))
        print()


def latent_dim_discussion(df, m):
    if "latent_dim_cfg" not in df.columns:
        print("(latent_dim_cfg not found)")
        return

    grp = (
        df.groupby("latent_dim_cfg")
        .agg(
            med_metric=(m["primary_col"], "median"),
            std_metric=(m["primary_col"], "std"),
            n=(m["primary_col"], "count"),
            med_params=("n_params", "median"),
        )
        .sort_values("med_metric")
    )
    print(grp.to_markdown())
    print()
    print(
        f"Best latent_dim={int(grp.index[0])} by median {m['primary_label']} "
        f"({grp.iloc[0].med_metric:.4f})."
    )
    print()


def variant_head_to_head(df, m):
    rounds = sorted(df["search_round"].dropna().unique())
    for r in rounds:
        rdf = df[df["search_round"] == r]
        best_per_variant = (
            rdf.groupby(["ae_variant", "config_name"])[m["primary_col"]]
            .median()
            .reset_index()
            .sort_values(m["primary_col"])
            .drop_duplicates("ae_variant")
            .sort_values(m["primary_col"])
        )
        print(f"\n### Round {int(r)} - Best Config per Variant\n")
        tbl = best_per_variant[["ae_variant", "config_name", m["primary_col"]]].rename(
            columns={
                "ae_variant": "Variant",
                "config_name": "Best Config",
                m["primary_col"]: f"Med {m['primary_label']}",
            }
        )
        print(tbl.to_markdown(index=False))
    print()


def stability_analysis(df, m):
    for r in sorted(df["search_round"].dropna().unique()):
        rdf = df[df["search_round"] == r]
        spread = (
            rdf.groupby("config_name")[m["primary_col"]]
            .agg(["median", "min", "max", "std"])
            .assign(range=lambda x: x["max"] - x["min"])
        )
        print(f"\n### Round {int(r)}\n")
        print(f"- Mean spread: {spread['range'].mean():.4f}")
        print(f"- Max spread:  {spread['range'].max():.4f} ({spread['range'].idxmax()})")
        print(f"- Mean std:    {spread['std'].mean():.4f}")
    print()


def round_progression(df, m):
    rounds = sorted(df["search_round"].dropna().unique())
    if len(rounds) < 2:
        print("(need >=2 rounds for progression)")
        return

    last_round = rounds[-1]
    final_configs = set(df[df["search_round"] == last_round]["config_name"].unique())

    medians = {}
    for r in rounds:
        rdf = df[(df["search_round"] == r) & (df["config_name"].isin(final_configs))]
        if rdf.empty:
            continue
        medians[int(r)] = rdf.groupby("config_name")[m["primary_col"]].median()

    prog = pd.DataFrame(medians)
    prog.columns = [f"R{c}" for c in prog.columns]
    if len(prog.columns) >= 2:
        first_col = prog.columns[0]
        last_col = prog.columns[-1]
        prog["Delta"] = prog[last_col] - prog[first_col]
        prog["DeltaPct"] = (prog["Delta"] / prog[first_col] * 100).round(1)
    print(prog.sort_values(prog.columns[-1]).round(4).to_markdown())
    print()


def param_efficiency(df, m):
    last_round = df["search_round"].max()
    rdf = df[df["search_round"] == last_round]
    agg = (
        rdf.groupby("config_name")
        .agg(
            med_metric=(m["primary_col"], "median"),
            n_params=("n_params", "first"),
            ae_variant=("ae_variant", "first"),
        )
        .sort_values("n_params")
    )
    rows = []
    for name, row in agg.iterrows():
        reduction = (1 - row.n_params / NBEATS_G_PARAMS) * 100 if pd.notna(row.n_params) else np.nan
        rows.append(
            {
                "Config": name,
                "Params": f"{row.n_params:,.0f}" if pd.notna(row.n_params) else "N/A",
                "Reduction": f"{reduction:.1f}%" if pd.notna(reduction) else "N/A",
                f"Med {m['primary_label']}": f"{row.med_metric:.4f}",
            }
        )
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def final_verdict(df, m):
    last_round = df["search_round"].max()
    rdf = df[df["search_round"] == last_round]
    agg = (
        rdf.groupby("config_name")
        .agg(
            med_metric=(m["primary_col"], "median"),
            min_metric=(m["primary_col"], "min"),
            max_metric=(m["primary_col"], "max"),
            n_params=("n_params", "first"),
            ae_variant=("ae_variant", "first"),
        )
        .sort_values("med_metric")
    )

    if m["target_enabled"]:
        winners = agg[(agg.med_metric < TARGET_OWA) & (agg.n_params < TARGET_PARAMS)]
        print(f"Target: OWA < {TARGET_OWA}, Params < {TARGET_PARAMS:,}")
        if winners.empty:
            print("No configurations meet both targets.")
        else:
            print(f"{len(winners)} configurations meet both targets.")
    else:
        print("Primary metric: median best_val_loss (lower is better).")
        print("OWA target checks are skipped because OWA is unavailable for this dataset.")

    best_name = agg.index[0]
    best_row = agg.iloc[0]
    print(
        f"Best final-round config: {best_name} with median {m['primary_label']}="
        f"{best_row.med_metric:.4f}."
    )
    print()


def analyze_dataset(dataset_name, csv_path):
    df = _load_df(csv_path)
    if df.empty:
        return False, "empty_csv"

    metric = _metric_profile(df)
    if metric is None:
        return False, "no_usable_metric"

    print(f"\n## Dataset: {dataset_name}\n")
    print(f"- CSV: `{csv_path}`")
    print(f"- Rows: {len(df)}")
    print(f"- Primary metric: `{metric['primary_col']}`")
    print()

    total_runs = len(df)
    total_configs = df["config_name"].nunique()
    total_rounds = df["search_round"].nunique()
    total_time = df["training_time_seconds"].sum() if "training_time_seconds" in df.columns else np.nan

    print("### Abstract\n")
    print(
        f"This analysis covers {total_configs} configurations over {total_rounds} rounds "
        f"({total_runs} runs)."
    )
    if pd.notna(total_time):
        print(f"Total training time: {total_time / 60:.1f} min.")
    if metric["has_owa"]:
        rng_min = pd.to_numeric(df["owa"], errors="coerce").min()
        rng_max = pd.to_numeric(df["owa"], errors="coerce").max()
        print(f"OWA range: {rng_min:.4f} - {rng_max:.4f}.")
    else:
        print("OWA unavailable; using best_val_loss throughout.")
    print()

    section("1. Successive Halving Funnel")
    funnel_summary(df, metric)

    for r in sorted(df["search_round"].dropna().unique()):
        section(f"2.{int(r)} Round {int(r)} Leaderboard")
        round_leaderboard(df, int(r), metric)

    r1 = df[df["search_round"] == df["search_round"].min()]
    section("3. Hyperparameter Marginals (Round 1)")
    hyperparameter_marginals(r1, metric)

    section("3b. Latent Dimension Discussion")
    latent_dim_discussion(r1, metric)

    section("4. Variant Head-to-Head")
    variant_head_to_head(df, metric)

    section("5. Stability Analysis")
    stability_analysis(df, metric)

    section("6. Round-over-Round Progression")
    round_progression(df, metric)

    section("7. Parameter Efficiency")
    param_efficiency(df, metric)

    section("8. Final Verdict")
    final_verdict(df, metric)

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="TrendAE multi-dataset analysis")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *list(TRENDAE_DATASETS.keys())],
        help="Dataset to analyze or 'all'",
    )
    args = parser.parse_args()

    print("# TrendAE Architecture Search - Multi-Dataset Analysis\n")

    requested = _resolve_datasets(args.dataset)
    analyzed = []
    skipped = []

    for ds in requested:
        csv_path = _search_csv_path(ds)
        if not os.path.exists(csv_path):
            print(f"[SKIP] dataset={ds} reason=missing_csv path={csv_path}")
            skipped.append((ds, "missing_csv", csv_path))
            continue

        ok, reason = analyze_dataset(ds, csv_path)
        if ok:
            analyzed.append(ds)
        else:
            print(f"[SKIP] dataset={ds} reason={reason} path={csv_path}")
            skipped.append((ds, reason, csv_path))

    print("\n# Summary\n")
    print(f"- analyzed_count: {len(analyzed)}")
    print(f"- skipped_count: {len(skipped)}")
    print(f"- analyzed: {analyzed}")
    if skipped:
        print("- skipped:")
        for ds, reason, path in skipped:
            print(f"  - dataset={ds} reason={reason} path={path}")

    if not analyzed:
        print("No datasets were analyzed successfully.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
