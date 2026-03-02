"""Wavelet Study 3 successive-halving analysis across runner-targeted datasets.

Behavior:
- Discovers datasets from run_wavelet_study_3_successive.WAVELET_STUDY3_DATASETS
- Resolves CSV paths from run_wavelet_study_3_successive._search_csv_path
- Supports --dataset {all,<dataset>}
- Skips missing/empty CSVs
- Uses OWA when finite values exist, else falls back to best_val_loss
- Runs M4-only baseline comparison section only for dataset=m4 with finite OWA
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXPERIMENTS_DIR)

from run_wavelet_study_3_successive import WAVELET_STUDY3_DATASETS, _search_csv_path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.float_format", "{:.4f}".format)

BASELINE_CSV = os.path.join(_EXPERIMENTS_DIR, "results", "m4", "block_benchmark_results.csv")
AE_TREND_CSV = os.path.join(_EXPERIMENTS_DIR, "results", "m4", "ae_trend_search_results.csv")

BASELINES = {
    "NBEATS-I+G": {"owa": 0.8057, "smape": 13.53, "params": 35_900_000},
    "GenericAE": {"owa": 0.8063, "smape": 13.57, "params": 4_800_000},
    "AutoEncoder": {"owa": 0.8075, "smape": 13.56, "params": 24_900_000},
    "NBEATS-I": {"owa": 0.8132, "smape": 13.67, "params": 12_900_000},
    "NBEATS-G": {"owa": 0.8198, "smape": 13.70, "params": 24_700_000},
    "AE+Trend": {"owa": 0.8015, "smape": 13.53, "params": 5_200_000},
}

NUMERIC_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "basis_dim", "basis_offset",
    "trend_thetas_dim_cfg", "meta_predicted_best", "meta_convergence_score",
    "search_round",
]


def section(title):
    print()
    print(f"## {title}")
    print()


def _resolve_datasets(dataset_arg):
    available = list(WAVELET_STUDY3_DATASETS.keys())
    if dataset_arg == "all":
        return available
    return [dataset_arg]


def _load_df(csv_path):
    df = pd.read_csv(csv_path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "search_round" in df.columns:
        df["search_round"] = df["search_round"].fillna(1).astype(int)
    if "wavelet_family" in df.columns:
        df["wavelet"] = df["wavelet_family"].astype(str).str.replace("WaveletV3", "", regex=False)
    if "trend_thetas_dim_cfg" in df.columns:
        df["ttd"] = pd.to_numeric(df["trend_thetas_dim_cfg"], errors="coerce")
    if "active_g_cfg" in df.columns:
        df["active_g_cfg"] = df["active_g_cfg"].astype(str)
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
    }


def section_overview(df, csv_path, m):
    section("1. Overview & Data Summary")
    print(f"- CSV: `{csv_path}`")
    print(f"- Total rows: {len(df)}")
    print(f"- Unique configs: {df['config_name'].nunique()}")
    print(f"- Search rounds: {sorted(df['search_round'].dropna().unique().tolist())}")
    print(f"- Primary metric: {m['primary_col']}")
    print()

    rows = []
    for r in sorted(df["search_round"].dropna().unique()):
        rdf = df[df["search_round"] == r]
        rows.append(
            {
                "Round": int(r),
                "Configs": rdf["config_name"].nunique(),
                "Rows": len(rdf),
                "Epochs": f"{int(rdf['epochs_trained'].min())}-{int(rdf['epochs_trained'].max())}" if "epochs_trained" in rdf.columns and not rdf.empty else "N/A",
                "Passes": ", ".join(sorted(rdf["active_g_cfg"].dropna().astype(str).unique().tolist())) if "active_g_cfg" in rdf.columns else "N/A",
            }
        )
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def section_funnel(df, m):
    section("2. Successive Halving Funnel")
    rows = []
    prev_cfgs = None
    for r in sorted(df["search_round"].dropna().unique()):
        rdf = df[df["search_round"] == r]
        cfg_count = rdf["config_name"].nunique()
        best = rdf.groupby("config_name")[m["primary_col"]].median().min()
        keep = "-"
        if prev_cfgs:
            keep = f"{cfg_count}/{prev_cfgs} ({cfg_count/prev_cfgs:.0%})"
        rows.append(
            {
                "Round": int(r),
                "Configs": cfg_count,
                "Rows": len(rdf),
                f"Best Med {m['primary_label']}": f"{best:.4f}",
                "Kept": keep,
            }
        )
        prev_cfgs = cfg_count
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def section_leaderboards(df, m):
    section("3. Round Leaderboards")
    for r in sorted(df["search_round"].dropna().unique()):
        rdf = df[df["search_round"] == r].copy()
        print(f"### Round {int(r)}\n")

        group_cols = ["config_name"]
        if "active_g_cfg" in rdf.columns:
            group_cols.append("active_g_cfg")

        agg = (
            rdf.groupby(group_cols)
            .agg(
                mean_metric=(m["primary_col"], "mean"),
                std_metric=(m["primary_col"], "std"),
                mean_smape=("smape", "mean"),
                mean_mase=("mase", "mean"),
                n_params=("n_params", "first"),
            )
            .sort_values("mean_metric")
            .reset_index()
            .head(15)
        )

        rename = {
            "config_name": "Config",
            "mean_metric": m["primary_label"],
            "std_metric": "Std",
            "mean_smape": "sMAPE",
            "mean_mase": "MASE",
            "n_params": "Params",
        }
        if "active_g_cfg" in agg.columns:
            rename["active_g_cfg"] = "Pass"
        agg = agg.rename(columns=rename)

        display_cols = ["Config"]
        if "Pass" in agg.columns:
            display_cols.append("Pass")
        display_cols += [m["primary_label"], "Std", "sMAPE", "MASE", "Params"]
        print(agg[display_cols].to_markdown(index=False, floatfmt=".4f"))
        print()


def section_marginals(df, m):
    section("4. Hyperparameter Marginals (Round 1)")
    r1 = df[df["search_round"] == df["search_round"].min()].copy()

    for factor in ["wavelet", "bd_label", "ttd", "active_g_cfg"]:
        if factor not in r1.columns:
            continue
        grp = (
            r1.groupby(factor)
            .agg(
                mean_metric=(m["primary_col"], "mean"),
                std_metric=(m["primary_col"], "std"),
                n=(m["primary_col"], "count"),
            )
            .sort_values("mean_metric")
            .reset_index()
        )
        grp = grp.rename(
            columns={
                factor: "Value",
                "mean_metric": f"Mean {m['primary_label']}",
                "std_metric": "Std",
                "n": "N",
            }
        )
        print(f"### {factor}\n")
        print(grp.to_markdown(index=False, floatfmt=".4f"))
        print()


def section_stability(df, m):
    section("5. Stability Analysis")
    for r in sorted(df["search_round"].dropna().unique()):
        rdf = df[df["search_round"] == r]
        spread = (
            rdf.groupby("config_name")[m["primary_col"]]
            .agg(["mean", "std", "min", "max"])
            .assign(range=lambda x: x["max"] - x["min"])
        )
        print(f"### Round {int(r)}\n")
        print(f"- Mean spread: {spread['range'].mean():.4f}")
        print(f"- Max spread: {spread['range'].max():.4f} ({spread['range'].idxmax()})")
        print(f"- Mean std: {spread['std'].mean():.4f}")
        print()


def section_progression(df, m):
    section("6. Round-over-Round Progression")
    rounds = sorted(df["search_round"].dropna().unique())
    if len(rounds) < 2:
        print("(need >=2 rounds)")
        print()
        return

    final_round = rounds[-1]
    finalists = set(df[df["search_round"] == final_round]["config_name"].unique())

    medians = {}
    for r in rounds:
        rdf = df[(df["search_round"] == r) & (df["config_name"].isin(finalists))]
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


def section_baseline_comparisons(df, m, dataset_name):
    section("7. Baseline Comparisons")
    if not (dataset_name == "m4" and m["has_owa"]):
        print("Section skipped (M4-specific baseline references).")
        print()
        return

    r3 = df[df["search_round"] == df["search_round"].max()]
    top = (
        r3.groupby(["config_name", "active_g_cfg"], dropna=False)
        .agg(mean_owa=("owa", "mean"), n_params=("n_params", "first"), smape=("smape", "mean"))
        .sort_values("mean_owa")
        .head(10)
        .reset_index()
    )
    top["vs NBEATS-I+G"] = top["mean_owa"] - BASELINES["NBEATS-I+G"]["owa"]
    top = top.rename(columns={"config_name": "Config", "active_g_cfg": "Pass", "mean_owa": "OWA", "n_params": "Params", "smape": "sMAPE"})
    print(top[["Config", "Pass", "OWA", "sMAPE", "Params", "vs NBEATS-I+G"]].to_markdown(index=False, floatfmt=".4f"))
    print()

    base_rows = [
        {"Baseline": name, "OWA": vals["owa"], "sMAPE": vals["smape"], "Params": vals["params"]}
        for name, vals in BASELINES.items()
    ]
    print(pd.DataFrame(base_rows).sort_values("OWA").to_markdown(index=False, floatfmt=".4f"))
    print()

    if os.path.exists(BASELINE_CSV):
        bdf = pd.read_csv(BASELINE_CSV)
        if "period" in bdf.columns:
            bdf = bdf[bdf["period"] == "Yearly"]
        if not bdf.empty and "owa" in bdf.columns:
            print("Loaded M4 block baseline CSV for reference.")
    if os.path.exists(AE_TREND_CSV):
        adf = pd.read_csv(AE_TREND_CSV)
        if not adf.empty:
            print("Loaded M4 AE+Trend CSV for reference.")
    print()


def section_final_verdict(df, m):
    section("8. Final Verdict")
    r_final = df[df["search_round"] == df["search_round"].max()].copy()
    agg = (
        r_final.groupby(["config_name", "active_g_cfg"], dropna=False)
        .agg(
            med_metric=(m["primary_col"], "median"),
            std_metric=(m["primary_col"], "std"),
            n_params=("n_params", "first"),
            smape=("smape", "median"),
            mase=("mase", "median"),
        )
        .sort_values("med_metric")
        .reset_index()
    )
    best = agg.iloc[0]

    print(
        f"Best configuration: {best['config_name']} (pass={best.get('active_g_cfg', 'N/A')}) with "
        f"median {m['primary_label']}={best['med_metric']:.4f}."
    )
    if m["has_owa"]:
        delta = best["med_metric"] - BASELINES["NBEATS-I+G"]["owa"]
        verdict = "beats" if delta < -0.001 else "matches" if abs(delta) < 0.005 else "trails"
        print(f"vs NBEATS-I+G ({BASELINES['NBEATS-I+G']['owa']:.4f}): {verdict} (delta={delta:+.4f}).")
    else:
        print("Primary metric: best_val_loss (lower is better). OWA-based baseline comparisons are not applicable.")

    top_cols = ["config_name", "active_g_cfg", "med_metric", "std_metric", "n_params", "smape", "mase"]
    show = agg[top_cols].head(10).rename(
        columns={
            "config_name": "Config",
            "active_g_cfg": "Pass",
            "med_metric": f"Med {m['primary_label']}",
            "std_metric": "Std",
            "n_params": "Params",
            "smape": "sMAPE",
            "mase": "MASE",
        }
    )
    print()
    print(show.to_markdown(index=False, floatfmt=".4f"))
    print()


def analyze_dataset(dataset_name, csv_path):
    df = _load_df(csv_path)
    if df.empty:
        return False, "empty_csv"

    m = _metric_profile(df)
    if m is None:
        return False, "no_usable_metric"

    print(f"\n## Dataset: {dataset_name}\n")
    print(f"- CSV: `{csv_path}`")
    print(f"- Rows: {len(df)}")
    print(f"- Primary metric: `{m['primary_col']}`")
    print()

    section_overview(df, csv_path, m)
    section_funnel(df, m)
    section_leaderboards(df, m)
    section_marginals(df, m)
    section_stability(df, m)
    section_progression(df, m)
    section_baseline_comparisons(df, m, dataset_name)
    section_final_verdict(df, m)

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Wavelet Study 3 multi-dataset analysis")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *list(WAVELET_STUDY3_DATASETS.keys())],
        help="Dataset to analyze or 'all'",
    )
    args = parser.parse_args()

    print("# Wavelet Study 3 - Multi-Dataset Analysis\n")

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
