"""LG/VAE Block Study — Multi-Dataset Analysis & Report

Analyzes successive-halving results from run_lg_vae_study.py across datasets.

Sections:
  1. Overview & Data Summary
  2. Successive Halving Funnel
  3. Round Leaderboards
  4. Hyperparameter Marginals (Round 1)
  5. Stability Analysis
  6. Round-over-Round Progression
  7. Baseline Comparisons (M4 only)
  8. LG vs VAE Head-to-Head
  9. Final Verdict

Usage:
    python experiments/analysis/lg_vae_study_analysis.py --dataset all
    python experiments/analysis/lg_vae_study_analysis.py --dataset m4
    python experiments/analysis/lg_vae_study_analysis.py --dataset weather
    python experiments/analysis/lg_vae_study_analysis.py --dataset m4 --llm
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXPERIMENTS_DIR)

from run_lg_vae_study import LG_VAE_STUDY_DATASETS, _search_csv_path

try:
    from tools.llm_commentary import generate_commentary
    _LLM = True
except ImportError:
    _LLM = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.float_format", "{:.4f}".format)

BASELINE_CSV = os.path.join(_EXPERIMENTS_DIR, "results", "m4", "block_benchmark_results.csv")

BASELINES = {
    "NBEATS-I+G": {"owa": 0.8057, "smape": 13.53, "params": 35_900_000},
    "GenericAE":  {"owa": 0.8063, "smape": 13.57, "params": 4_800_000},
    "AutoEncoder": {"owa": 0.8075, "smape": 13.56, "params": 24_900_000},
    "NBEATS-I":   {"owa": 0.8132, "smape": 13.67, "params": 12_900_000},
    "NBEATS-G":   {"owa": 0.8198, "smape": 13.70, "params": 24_700_000},
    "AE+Trend":   {"owa": 0.8015, "smape": 13.53, "params": 5_200_000},
}

NUMERIC_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio",
    "meta_predicted_best", "meta_convergence_score",
    "search_round",
]

# LG/VAE head-to-head matched pairs: (LG block, VAE block)
LG_VAE_PAIRS = [
    ("GenericAELG",          "GenericVAE"),
    ("BottleneckGenericAELG", "BottleneckGenericVAE"),
    ("GenericAEBackcastAELG", "GenericAEBackcastVAE"),
    ("AutoEncoderAELG",      "AutoEncoderVAE"),
    ("TrendAELG+Haar",       "TrendVAE+Haar"),
    ("TrendAELG+DB4",        "TrendVAE+DB4"),
    ("TrendAELG+Coif2",      "TrendVAE+Coif2"),
    ("TrendAELG+Symlet3",    "TrendVAE+Symlet3"),
    ("NBEATS-I-LG",          "NBEATS-I-VAE"),
]


def section(title):
    print()
    print(f"## {title}")
    print()


def _resolve_datasets(dataset_arg):
    available = list(LG_VAE_STUDY_DATASETS.keys())
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
    if "active_g_cfg" in df.columns:
        df["active_g_cfg"] = df["active_g_cfg"].astype(str)
    if "backbone_family" in df.columns:
        df["backbone_family"] = df["backbone_family"].astype(str)
    if "category" in df.columns:
        df["category"] = df["category"].astype(str)
    if "wavelet_family" in df.columns:
        df["wavelet_family"] = df["wavelet_family"].astype(str)
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


# ---------------------------------------------------------------------------
# Section 1: Overview & Data Summary
# ---------------------------------------------------------------------------

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
        rows.append({
            "Round": int(r),
            "Configs": rdf["config_name"].nunique(),
            "Rows": len(rdf),
            "Epochs": (
                f"{int(rdf['epochs_trained'].min())}-{int(rdf['epochs_trained'].max())}"
                if "epochs_trained" in rdf.columns and not rdf.empty else "N/A"
            ),
            "Passes": (
                ", ".join(sorted(rdf["active_g_cfg"].dropna().astype(str).unique().tolist()))
                if "active_g_cfg" in rdf.columns else "N/A"
            ),
        })
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


# ---------------------------------------------------------------------------
# Section 2: Successive Halving Funnel
# ---------------------------------------------------------------------------

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
        rows.append({
            "Round": int(r),
            "Configs": cfg_count,
            "Rows": len(rdf),
            f"Best Med {m['primary_label']}": f"{best:.4f}",
            "Kept": keep,
        })
        prev_cfgs = cfg_count
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


# ---------------------------------------------------------------------------
# Section 3: Round Leaderboards
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Section 4: Hyperparameter Marginals (Round 1)
# ---------------------------------------------------------------------------

def section_marginals(df, m, use_llm=False):
    section("4. Hyperparameter Marginals (Round 1)")
    r1 = df[df["search_round"] == df["search_round"].min()].copy()

    factors = [
        ("backbone_family", "backbone_family"),
        ("category", "category"),
        ("active_g_cfg", "active_g_cfg"),
    ]

    # Wavelet family only for trend_wavelet category
    has_wavelet = (
        "wavelet_family" in r1.columns
        and "category" in r1.columns
        and (r1["category"] == "trend_wavelet").any()
    )
    if has_wavelet:
        factors.append(("wavelet_family (trend_wavelet only)", "wavelet_family"))

    for label, factor in factors:
        col = factor
        if col not in r1.columns:
            continue

        subset = r1
        if factor == "wavelet_family":
            subset = r1[r1["category"] == "trend_wavelet"]
            subset = subset[subset["wavelet_family"].astype(str).str.len() > 0]

        grp = (
            subset.groupby(col)
            .agg(
                mean_metric=(m["primary_col"], "mean"),
                std_metric=(m["primary_col"], "std"),
                n=(m["primary_col"], "count"),
            )
            .sort_values("mean_metric")
            .reset_index()
        )
        grp = grp.rename(columns={
            col: "Value",
            "mean_metric": f"Mean {m['primary_label']}",
            "std_metric": "Std",
            "n": "N",
        })
        print(f"### {label}\n")
        print(grp.to_markdown(index=False, floatfmt=".4f"))
        print()

    if use_llm and _LLM:
        llm_ctx = {
            "study": "LG/VAE Block Study",
            "factors": [f[0] for f in factors],
            "description": (
                "Comparing three backbone families: LG (learned-gate AE), "
                "VAE (variational AE), and RootBlock (standard VAE), across "
                "pure homogeneous, trend+wavelet, and NBEATS-I style categories."
            ),
        }
        llm_text = generate_commentary("hyperparameter_marginal", llm_ctx)
        if llm_text:
            print(llm_text)
            print()


# ---------------------------------------------------------------------------
# Section 5: Stability Analysis
# ---------------------------------------------------------------------------

def section_stability(df, m, use_llm=False):
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

    if use_llm and _LLM:
        llm_ctx = {
            "study": "LG/VAE Block Study",
            "description": "Stability of LG/VAE block configurations across runs.",
        }
        llm_text = generate_commentary("stability_analysis", llm_ctx)
        if llm_text:
            print(llm_text)
            print()


# ---------------------------------------------------------------------------
# Section 6: Round-over-Round Progression
# ---------------------------------------------------------------------------

def section_progression(df, m, use_llm=False):
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

    if use_llm and _LLM:
        llm_ctx = {
            "study": "LG/VAE Block Study",
            "description": "Round-over-round progression of LG/VAE finalist configs.",
        }
        llm_text = generate_commentary("round_progression", llm_ctx)
        if llm_text:
            print(llm_text)
            print()


# ---------------------------------------------------------------------------
# Section 7: Baseline Comparisons (M4 only)
# ---------------------------------------------------------------------------

def section_baseline_comparisons(df, m, dataset_name):
    section("7. Baseline Comparisons")
    if not (dataset_name == "m4" and m["has_owa"]):
        print("Section skipped (M4-specific baseline references).")
        print()
        return

    r_final = df[df["search_round"] == df["search_round"].max()]
    top = (
        r_final.groupby(["config_name", "active_g_cfg"], dropna=False)
        .agg(
            mean_owa=("owa", "mean"),
            n_params=("n_params", "first"),
            smape=("smape", "mean"),
        )
        .sort_values("mean_owa")
        .head(10)
        .reset_index()
    )
    top["vs NBEATS-I+G"] = top["mean_owa"] - BASELINES["NBEATS-I+G"]["owa"]
    top = top.rename(columns={
        "config_name": "Config",
        "active_g_cfg": "Pass",
        "mean_owa": "OWA",
        "n_params": "Params",
        "smape": "sMAPE",
    })
    print(top[["Config", "Pass", "OWA", "sMAPE", "Params", "vs NBEATS-I+G"]].to_markdown(
        index=False, floatfmt=".4f"
    ))
    print()

    base_rows = [
        {"Baseline": name, "OWA": vals["owa"], "sMAPE": vals["smape"], "Params": vals["params"]}
        for name, vals in BASELINES.items()
    ]
    print(pd.DataFrame(base_rows).sort_values("OWA").to_markdown(index=False, floatfmt=".4f"))
    print()


# ---------------------------------------------------------------------------
# Section 8: LG vs VAE Head-to-Head
# ---------------------------------------------------------------------------

def section_lg_vae_head_to_head(df, m, use_llm=False):
    section("8. LG vs VAE Head-to-Head")

    # Use final round if available, else latest
    final_round = df["search_round"].max()
    rdf = df[df["search_round"] == final_round].copy()

    # Group by (config_name, active_g_cfg) and compute median metric
    group_cols = ["config_name"]
    if "active_g_cfg" in rdf.columns:
        group_cols.append("active_g_cfg")

    agg = (
        rdf.groupby(group_cols)[m["primary_col"]]
        .median()
        .reset_index()
        .rename(columns={m["primary_col"]: "median_metric"})
    )

    rows = []
    for lg_name, vae_name in LG_VAE_PAIRS:
        for pass_val in agg["active_g_cfg"].unique() if "active_g_cfg" in agg.columns else [None]:
            if pass_val is not None:
                lg_row = agg[(agg["config_name"] == lg_name) & (agg["active_g_cfg"] == pass_val)]
                vae_row = agg[(agg["config_name"] == vae_name) & (agg["active_g_cfg"] == pass_val)]
            else:
                lg_row = agg[agg["config_name"] == lg_name]
                vae_row = agg[agg["config_name"] == vae_name]

            if lg_row.empty or vae_row.empty:
                continue

            lg_val = lg_row["median_metric"].iloc[0]
            vae_val = vae_row["median_metric"].iloc[0]
            delta = vae_val - lg_val
            winner = "LG" if lg_val < vae_val else "VAE" if vae_val < lg_val else "Tie"

            row = {
                "LG Config": lg_name,
                "VAE Config": vae_name,
                f"LG {m['primary_label']}": lg_val,
                f"VAE {m['primary_label']}": vae_val,
                "Delta (VAE-LG)": delta,
                "Winner": winner,
            }
            if pass_val is not None:
                row["Pass"] = pass_val
            rows.append(row)

    if not rows:
        print("No matched LG/VAE pairs found in the final round.")
        print()
        return

    pairs_df = pd.DataFrame(rows)
    print(pairs_df.to_markdown(index=False, floatfmt=".4f"))
    print()

    # Summary
    if rows:
        lg_wins = sum(1 for r in rows if r["Winner"] == "LG")
        vae_wins = sum(1 for r in rows if r["Winner"] == "VAE")
        ties = sum(1 for r in rows if r["Winner"] == "Tie")
        print(f"**Score: LG {lg_wins} — VAE {vae_wins} — Tie {ties}**")
        print()

    if use_llm and _LLM:
        llm_ctx = {
            "study": "LG/VAE Block Study",
            "description": (
                "Head-to-head comparison of matched LG (learned-gate) vs "
                "VAE (variational) block pairs across configurations."
            ),
        }
        llm_text = generate_commentary("variant_comparison", llm_ctx)
        if llm_text:
            print(llm_text)
            print()


# ---------------------------------------------------------------------------
# Section 9: Final Verdict
# ---------------------------------------------------------------------------

def section_final_verdict(df, m):
    section("9. Final Verdict")
    r_final = df[df["search_round"] == df["search_round"].max()].copy()

    group_cols = ["config_name"]
    if "active_g_cfg" in r_final.columns:
        group_cols.append("active_g_cfg")

    agg = (
        r_final.groupby(group_cols, dropna=False)
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

    if agg.empty:
        print("No final round results available.")
        print()
        return

    best = agg.iloc[0]

    print(
        f"Best configuration: **{best['config_name']}** "
        f"(pass={best.get('active_g_cfg', 'N/A')}) with "
        f"median {m['primary_label']}={best['med_metric']:.4f}."
    )
    if m["has_owa"]:
        delta = best["med_metric"] - BASELINES["NBEATS-I+G"]["owa"]
        verdict = "beats" if delta < -0.001 else "matches" if abs(delta) < 0.005 else "trails"
        print(
            f"vs NBEATS-I+G ({BASELINES['NBEATS-I+G']['owa']:.4f}): "
            f"{verdict} (delta={delta:+.4f})."
        )
    else:
        print(
            "Primary metric: best_val_loss (lower is better). "
            "OWA-based baseline comparisons are not applicable."
        )

    top_cols = list(group_cols) + ["med_metric", "std_metric", "n_params", "smape", "mase"]
    show = agg[top_cols].head(10).rename(columns={
        "config_name": "Config",
        "active_g_cfg": "Pass",
        "med_metric": f"Med {m['primary_label']}",
        "std_metric": "Std",
        "n_params": "Params",
        "smape": "sMAPE",
        "mase": "MASE",
    })
    print()
    print(show.to_markdown(index=False, floatfmt=".4f"))
    print()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def analyze_dataset(dataset_name, csv_path, use_llm=False):
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
    section_marginals(df, m, use_llm=use_llm)
    section_stability(df, m, use_llm=use_llm)
    section_progression(df, m, use_llm=use_llm)
    section_baseline_comparisons(df, m, dataset_name)
    section_lg_vae_head_to_head(df, m, use_llm=use_llm)
    section_final_verdict(df, m)

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="LG/VAE Block Study — Multi-Dataset Analysis"
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *list(LG_VAE_STUDY_DATASETS.keys())],
        help="Dataset to analyze or 'all'",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Enable optional LLM commentary (requires llm_commentary module).",
    )
    args = parser.parse_args()

    use_llm = args.llm

    print("# LG/VAE Block Study - Multi-Dataset Analysis\n")

    requested = _resolve_datasets(args.dataset)
    analyzed = []
    skipped = []

    for ds in requested:
        csv_path = _search_csv_path(ds)
        if not os.path.exists(csv_path):
            print(f"[SKIP] dataset={ds} reason=missing_csv path={csv_path}")
            skipped.append((ds, "missing_csv", csv_path))
            continue

        ok, reason = analyze_dataset(ds, csv_path, use_llm=use_llm)
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
