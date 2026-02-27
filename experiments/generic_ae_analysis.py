"""Generic AE Pure-Stack Study — Dataset-Aware Analysis

Reads experiments/results/<dataset>/generic_ae_pure_stack_results.csv and produces:
  1. Dataset + run summary
  2. Successive-halving funnel by round
  3. Per-round leaderboard (median primary metric)
  4. Round-1 hyperparameter marginals
  5. Final-round stability and training behavior
  6. Optional M4 OWA significance vs NBEATS-I+G baseline

Metric policy:
  - Use OWA when available (M4)
  - Fallback to best_val_loss when OWA is unavailable (Weather)

Usage:
    python experiments/generic_ae_analysis.py --dataset m4
    python experiments/generic_ae_analysis.py --dataset weather
    python experiments/generic_ae_analysis.py --dataset all
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from run_generic_ae_study import STUDY_DATASETS, _csv_path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.float_format", "{:.4f}".format)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "latent_dim_cfg", "thetas_dim_cfg",
]


def _baseline_csv(dataset_name):
    return os.path.join(_SCRIPT_DIR, "results", dataset_name, "block_benchmark_results.csv")


def _section(title):
    print(f"\n## {title}\n")


def _print_table(df):
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))


def _sig_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _load_df(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[ERROR] Empty CSV: {csv_path}")
        return None

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "search_round" in df.columns:
        df["search_round"] = pd.to_numeric(df["search_round"], errors="coerce").fillna(1).astype(int)
    else:
        df["search_round"] = 1

    if "active_g" in df.columns:
        df["active_g"] = df["active_g"].astype(str)
    else:
        df["active_g"] = "False"

    if "diverged" in df.columns:
        df["diverged"] = (
            df["diverged"].astype(str).str.lower().isin(["true", "1"])
        )
    else:
        df["diverged"] = False

    return df


def _metric_profile(df):
    has_owa = "owa" in df.columns and pd.to_numeric(df["owa"], errors="coerce").notna().any()
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
    }


def _round_funnel(df, metric):
    rows = []
    rounds = sorted(df["search_round"].unique())
    for round_num in rounds:
        rdf = df[df["search_round"] == round_num]
        best_med = (
            rdf.groupby("config_name")[metric["primary_col"]]
            .median()
            .min()
        )
        rows.append({
            "Round": round_num,
            "Configs": rdf["config_name"].nunique(),
            "Runs": len(rdf),
            "Max Epochs": int(rdf["epochs_trained"].max()) if "epochs_trained" in rdf.columns else np.nan,
            f"Best Med {metric['primary_label']}": f"{best_med:.4f}",
        })

    _print_table(pd.DataFrame(rows))


def _round_leaderboard(df, round_num, metric):
    rdf = df[df["search_round"] == round_num]
    if rdf.empty:
        print(f"(no rows for round {round_num})")
        return

    agg = (
        rdf.groupby("config_name")
        .agg(
            med_metric=(metric["primary_col"], "median"),
            min_metric=(metric["primary_col"], "min"),
            max_metric=(metric["primary_col"], "max"),
            med_smape=("smape", "median"),
            med_mase=("mase", "median"),
            n_params=("n_params", "first"),
            block_type=("block_type", "first"),
            latent_dim=("latent_dim_cfg", "first"),
            thetas_dim=("thetas_dim_cfg", "first"),
            active_g=("active_g", "first"),
        )
        .sort_values("med_metric")
    )

    rows = []
    for rank, (name, row) in enumerate(agg.iterrows(), 1):
        rows.append({
            "Rank": rank,
            "Config": name,
            "Block": row.block_type,
            metric["primary_label"]: f"{row.med_metric:.4f}",
            "Spread": f"{(row.max_metric - row.min_metric):.4f}",
            "sMAPE": f"{row.med_smape:.2f}",
            "MASE": f"{row.med_mase:.4f}",
            "Params": f"{int(row.n_params):,}" if pd.notna(row.n_params) else "?",
        })

    _print_table(pd.DataFrame(rows))


def _marginals_round1(df, metric):
    r1 = df[df["search_round"] == 1]
    if r1.empty:
        print("(round 1 data missing)")
        return

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
            .agg(
                med_metric=(metric["primary_col"], "median"),
                mean_metric=(metric["primary_col"], "mean"),
                std_metric=(metric["primary_col"], "std"),
                n=(metric["primary_col"], "count"),
            )
            .sort_values("med_metric")
        )

        rows = []
        for val, row in grp.iterrows():
            rows.append({
                label: str(val),
                f"Med {metric['primary_label']}": f"{row.med_metric:.4f}",
                f"Mean {metric['primary_label']}": f"{row.mean_metric:.4f}",
                "Std": f"{row.std_metric:.4f}",
                "N": int(row.n),
            })

        print(f"### {label}\n")
        _print_table(pd.DataFrame(rows))
        print()


def _stability_final_round(df, metric):
    last_round = int(df["search_round"].max())
    rdf = df[df["search_round"] == last_round]

    spread = (
        rdf.groupby("config_name")[metric["primary_col"]]
        .agg(["median", "min", "max", "std"])
        .assign(range=lambda x: x["max"] - x["min"])
        .sort_values("range")
    )

    if spread.empty:
        print("(no stability rows)")
        return

    print(f"Final round analyzed: {last_round}")
    print(f"Mean spread: {spread['range'].mean():.4f}")
    print(f"Max spread: {spread['range'].max():.4f}")

    stable = spread.head(5).reset_index()
    stable.columns = [
        "Config",
        f"Median {metric['primary_label']}",
        "Min",
        "Max",
        "Std",
        "Range",
    ]
    print("\nMost stable configs:\n")
    _print_table(stable[["Config", f"Median {metric['primary_label']}", "Range", "Std"]])


def _training_stability(df):
    rows = []
    for round_num in sorted(df["search_round"].unique()):
        rdf = df[df["search_round"] == round_num]
        n_total = len(rdf)
        n_div = int(rdf["diverged"].sum())

        if "stopping_reason" in rdf.columns:
            early_stopped = int((rdf["stopping_reason"] == "EARLY_STOPPED").sum())
            max_epochs = int((rdf["stopping_reason"] == "MAX_EPOCHS").sum())
        else:
            early_stopped = 0
            max_epochs = 0

        rows.append({
            "Round": round_num,
            "Runs": n_total,
            "Diverged": f"{n_div} ({(n_div / n_total * 100) if n_total else 0:.1f}%)",
            "Early Stopped": f"{early_stopped} ({(early_stopped / n_total * 100) if n_total else 0:.1f}%)",
            "Max Epochs": f"{max_epochs} ({(max_epochs / n_total * 100) if n_total else 0:.1f}%)",
        })

    _print_table(pd.DataFrame(rows))


def _significance_m4(df, metric, dataset_name, period):
    if dataset_name != "m4" or not metric["has_owa"]:
        print("(significance skipped: requires M4 OWA)")
        return

    baseline_csv = _baseline_csv(dataset_name)
    if not os.path.exists(baseline_csv):
        print(f"(significance skipped: missing baseline CSV: {baseline_csv})")
        return

    bdf = pd.read_csv(baseline_csv)
    bdf = bdf[(bdf["config_name"] == "NBEATS-I+G") & (bdf["period"] == period)]
    ref_samples = pd.to_numeric(bdf["owa"], errors="coerce").dropna().values

    if len(ref_samples) < 2:
        print("(significance skipped: insufficient NBEATS-I+G baseline samples)")
        return

    last_round = int(df["search_round"].max())
    rdf = df[df["search_round"] == last_round]

    rows = []
    for cfg_name, cfg_rows in rdf.groupby("config_name"):
        cfg_owa = pd.to_numeric(cfg_rows["owa"], errors="coerce").dropna().values
        if len(cfg_owa) < 2:
            continue

        stat, p_value = stats.mannwhitneyu(cfg_owa, ref_samples, alternative="less")
        med_owa = float(np.median(cfg_owa))
        rows.append({
            "Config": cfg_name,
            "Med OWA": f"{med_owa:.4f}",
            "U": f"{stat:.1f}",
            "p-value": f"{p_value:.4f}",
            "Sig": _sig_stars(p_value),
        })

    if rows:
        _print_table(pd.DataFrame(rows).sort_values("Med OWA"))
    else:
        print("(no configs with enough OWA samples for significance test)")


def analyze_dataset(dataset_name):
    period = STUDY_DATASETS[dataset_name]
    csv_path = _csv_path(dataset_name)

    df = _load_df(csv_path)
    if df is None:
        return

    metric = _metric_profile(df)
    if metric is None:
        print("[ERROR] Could not determine usable metric column (owa or best_val_loss).")
        return

    print(f"# Generic AE Pure-Stack Analysis — {dataset_name}/{period}\n")
    print(f"- CSV: {csv_path}")
    print(f"- Rows: {len(df)}")
    print(f"- Configs: {df['config_name'].nunique()}")
    print(f"- Rounds: {df['search_round'].nunique()}")
    print(f"- Primary metric: {metric['primary_label']}")

    _section("1. Successive Halving Funnel")
    _round_funnel(df, metric)

    for round_num in sorted(df["search_round"].unique()):
        _section(f"2.{round_num} Round {round_num} Leaderboard")
        _round_leaderboard(df, round_num, metric)

    _section("3. Hyperparameter Marginals (Round 1)")
    _marginals_round1(df, metric)

    _section(f"4. Stability ({metric['primary_label']} spread)")
    _stability_final_round(df, metric)

    _section("5. Training Stability")
    _training_stability(df)

    _section("6. Significance vs NBEATS-I+G")
    _significance_m4(df, metric, dataset_name, period)


def main():
    parser = argparse.ArgumentParser(description="Analyze GenericAE pure-stack study results.")
    parser.add_argument(
        "--dataset",
        default="m4",
        choices=["m4", "weather", "all"],
        help="Dataset to analyze.",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for dataset_name in sorted(STUDY_DATASETS.keys()):
            analyze_dataset(dataset_name)
            print("\n" + "-" * 90 + "\n")
    else:
        analyze_dataset(args.dataset)


if __name__ == "__main__":
    main()
