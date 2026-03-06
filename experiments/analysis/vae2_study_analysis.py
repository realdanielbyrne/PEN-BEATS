"""VAE2 Block Study — Multi-Dataset Analysis & Report

Analyzes successive-halving results from the VAE2 study across datasets.
Compares three backbone families: VAE2 (new), VAE1 (variational AE), and
AELG (learned-gate AE).

Sections:
  1. Overview & Data Summary
  2. Successive Halving Funnel
  3. Round Leaderboards
  4. Hyperparameter Marginals
  5. Stability Analysis
  6. Round-over-Round Progression
  7. Baseline Comparisons (M4 only)
  8. Backbone Family Head-to-Head
  9. Final Verdict

Usage:
    python experiments/analysis/vae2_study_analysis.py --dataset all
    python experiments/analysis/vae2_study_analysis.py --dataset m4
    python experiments/analysis/vae2_study_analysis.py --dataset weather
    python experiments/analysis/vae2_study_analysis.py --dataset all --output report.md
"""

import argparse
import io
import os
import re
import sys

import nbformat as nbf
import numpy as np
import pandas as pd

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.float_format", "{:.4f}".format)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

VAE2_DATASETS = {
    "m4": os.path.join(_EXPERIMENTS_DIR, "results", "m4", "vae2_study_results.csv"),
    "weather": os.path.join(_EXPERIMENTS_DIR, "results", "weather", "vae2_study_results.csv"),
}

# ---------------------------------------------------------------------------
# Reference baselines (M4 OWA; lower is better)
# ---------------------------------------------------------------------------

BASELINES = {
    "Naive2":   {"owa": 1.0,    "smape": None,  "params": 0},
    "NBEATS-G": {"owa": 0.862,  "smape": 13.70, "params": 24_700_000},
}

NUMERIC_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "search_round", "latent_dim",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass_from_experiment(exp_str):
    """Derive pass_type from experiment column value.

    experiment prefix 'baseline_r*'      -> 'baseline'
    experiment prefix 'activeG_fcast_r*' -> 'activeG_fcast'
    """
    if isinstance(exp_str, str):
        if exp_str.startswith("baseline"):
            return "baseline"
        if exp_str.startswith("activeG_fcast"):
            return "activeG_fcast"
    return "unknown"


def _backbone_base(config_name):
    """Strip backbone family suffixes to obtain a canonical base config name.

    E.g. 'GenericVAE2_ld8' -> 'GenericBACKBONE_ld8'
         'GenericAELG_ld8' -> 'GenericBACKBONE_ld8'
         'GenericVAE_ld8'  -> 'GenericBACKBONE_ld8'
    """
    s = config_name
    # Order matters: replace VAE2 before VAE so 'VAE2' is not partially matched
    s = re.sub(r"VAE2", "BACKBONE", s)
    s = re.sub(r"AELG", "BACKBONE", s)
    s = re.sub(r"VAE1?", "BACKBONE", s)  # VAE or VAE1
    return s


def section(title):
    print()
    print(f"## {title}")
    print()


def _load_df(csv_path):
    df = pd.read_csv(csv_path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "search_round" in df.columns:
        df["search_round"] = df["search_round"].fillna(1).astype(int)
    if "backbone_family" in df.columns:
        df["backbone_family"] = df["backbone_family"].astype(str)
    if "category_label" in df.columns:
        df["category_label"] = df["category_label"].astype(str)
    if "wavelet_family" in df.columns:
        df["wavelet_family"] = df["wavelet_family"].astype(str)
    if "latent_dim" in df.columns:
        df["latent_dim"] = df["latent_dim"].astype(str)

    # Derive pass_type from experiment column
    if "experiment" in df.columns:
        df["pass_type"] = df["experiment"].apply(_pass_from_experiment)
    else:
        df["pass_type"] = "unknown"

    # Derive backbone-agnostic base config for head-to-head matching
    if "config_name" in df.columns:
        df["base_config"] = df["config_name"].apply(_backbone_base)

    return df


def _metric_profile(df, dataset_name):
    """Determine the primary evaluation metric.

    M4 OWA is preferred when available and non-NaN.
    Weather OWA is always NaN — fall back to norm_mae.
    """
    owa = (
        pd.to_numeric(df["owa"], errors="coerce")
        if "owa" in df.columns
        else pd.Series(dtype=float)
    )
    has_owa = bool(len(owa) and owa.notna().any())

    if has_owa:
        return {"has_owa": True, "primary_col": "owa", "primary_label": "OWA"}

    # Fall back: try norm_mae, then best_val_loss
    for fallback in ("norm_mae", "best_val_loss"):
        if fallback in df.columns:
            vals = pd.to_numeric(df[fallback], errors="coerce")
            if vals.notna().any():
                return {
                    "has_owa": False,
                    "primary_col": fallback,
                    "primary_label": fallback,
                }
    return None


# ---------------------------------------------------------------------------
# Section 1: Overview & Data Summary
# ---------------------------------------------------------------------------

def section_overview(df, csv_path, m):
    section("1. Overview & Data Summary")
    print(f"- CSV: `{csv_path}`")
    print(f"- Total rows: {len(df)}")
    print(f"- Unique configs: {df['config_name'].nunique()}")
    print(f"- Search rounds: {sorted(df['search_round'].dropna().unique().tolist())}")
    print(f"- Primary metric: `{m['primary_col']}`")
    print(f"- Backbone families: {sorted(df['backbone_family'].dropna().unique().tolist())}")
    print(f"- Latent dims: {sorted(df['latent_dim'].dropna().unique().tolist())}")
    print()

    rows = []
    for r in sorted(df["search_round"].dropna().unique()):
        rdf = df[df["search_round"] == r]
        passes = sorted(rdf["pass_type"].dropna().unique().tolist())
        rows.append({
            "Round": int(r),
            "Configs": rdf["config_name"].nunique(),
            "Rows": len(rdf),
            "Epochs": (
                f"{int(rdf['epochs_trained'].min())}-{int(rdf['epochs_trained'].max())}"
                if "epochs_trained" in rdf.columns and not rdf.empty
                else "N/A"
            ),
            "Passes": ", ".join(passes),
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
        if prev_cfgs is not None:
            keep = f"{cfg_count}/{prev_cfgs} ({cfg_count / prev_cfgs:.0%})"
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

        group_cols = ["config_name", "pass_type"]

        agg = (
            rdf.groupby(group_cols, dropna=False)
            .agg(
                mean_metric=(m["primary_col"], "mean"),
                std_metric=(m["primary_col"], "std"),
                mean_smape=("smape", "mean"),
                mean_mase=("mase", "mean"),
                n_params=("n_params", "first"),
                backbone_family=("backbone_family", "first"),
                latent_dim=("latent_dim", "first"),
            )
            .sort_values("mean_metric")
            .reset_index()
            .head(15)
        )

        agg = agg.rename(columns={
            "config_name": "Config",
            "pass_type": "Pass",
            "mean_metric": m["primary_label"],
            "std_metric": "Std",
            "mean_smape": "sMAPE",
            "mean_mase": "MASE",
            "n_params": "Params",
            "backbone_family": "Backbone",
            "latent_dim": "LD",
        })

        display_cols = ["Config", "Pass", "Backbone", "LD", m["primary_label"], "Std", "sMAPE", "MASE", "Params"]
        print(agg[display_cols].to_markdown(index=False, floatfmt=".4f"))
        print()


# ---------------------------------------------------------------------------
# Section 4: Hyperparameter Marginals
# ---------------------------------------------------------------------------

def section_marginals(df, m):
    section("4. Hyperparameter Marginals (Round 1)")
    r1 = df[df["search_round"] == df["search_round"].min()].copy()

    factors = [
        ("backbone_family", "backbone_family"),
        ("category_label", "category_label"),
        ("latent_dim", "latent_dim"),
    ]

    # Wavelet family only for trend_wavelet categories
    has_wavelet = (
        "wavelet_family" in r1.columns
        and "category_label" in r1.columns
        and r1["category_label"].str.contains("wavelet", na=False).any()
    )
    if has_wavelet:
        factors.append(("wavelet_family (wavelet configs only)", "wavelet_family"))

    for label, factor in factors:
        col = factor
        if col not in r1.columns:
            continue

        subset = r1.copy()
        if factor == "wavelet_family":
            subset = r1[r1["category_label"].str.contains("wavelet", na=False)]
            subset = subset[subset["wavelet_family"].astype(str).str.lower() != "nan"]
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


# ---------------------------------------------------------------------------
# Section 5: Stability Analysis
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Section 6: Round-over-Round Progression
# ---------------------------------------------------------------------------

def section_progression(df, m):
    section("6. Round-over-Round Progression")
    rounds = sorted(df["search_round"].dropna().unique())
    if len(rounds) < 2:
        print("(need >=2 rounds to show progression)")
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


# ---------------------------------------------------------------------------
# Section 7: Baseline Comparisons (M4 only)
# ---------------------------------------------------------------------------

def section_baseline_comparisons(df, m, dataset_name):
    section("7. Baseline Comparisons")
    if not (dataset_name == "m4" and m["has_owa"]):
        print("Section skipped (M4 OWA baselines not applicable for this dataset).")
        print()
        return

    r_final = df[df["search_round"] == df["search_round"].max()].copy()
    top = (
        r_final.groupby(["config_name", "pass_type"], dropna=False)
        .agg(
            mean_owa=("owa", "mean"),
            n_params=("n_params", "first"),
            smape=("smape", "mean"),
            backbone_family=("backbone_family", "first"),
        )
        .sort_values("mean_owa")
        .head(10)
        .reset_index()
    )
    top["vs NBEATS-G"] = top["mean_owa"] - BASELINES["NBEATS-G"]["owa"]
    top = top.rename(columns={
        "config_name": "Config",
        "pass_type": "Pass",
        "mean_owa": "OWA",
        "n_params": "Params",
        "smape": "sMAPE",
        "backbone_family": "Backbone",
    })
    print("### Top-10 VAE2-Study Configs (Final Round)\n")
    print(
        top[["Config", "Pass", "Backbone", "OWA", "sMAPE", "Params", "vs NBEATS-G"]].to_markdown(
            index=False, floatfmt=".4f"
        )
    )
    print()

    print("### Reference Baselines\n")
    base_rows = [
        {"Baseline": name, "OWA": vals["owa"], "sMAPE": vals.get("smape") or "N/A", "Params": vals["params"]}
        for name, vals in BASELINES.items()
    ]
    print(pd.DataFrame(base_rows).sort_values("OWA").to_markdown(index=False, floatfmt=".4f"))
    print()


# ---------------------------------------------------------------------------
# Section 8: Backbone Family Head-to-Head
# ---------------------------------------------------------------------------

def section_backbone_head_to_head(df, m):
    section("8. Backbone Family Head-to-Head")
    """Compare VAE2 vs VAE1, VAE2 vs AELG, VAE1 vs AELG by matching on base_config + pass_type."""

    final_round = df["search_round"].max()
    rdf = df[df["search_round"] == final_round].copy()

    # Aggregate median metric per (config_name, pass_type)
    agg = (
        rdf.groupby(["config_name", "pass_type", "backbone_family", "base_config"], dropna=False)[m["primary_col"]]
        .median()
        .reset_index()
        .rename(columns={m["primary_col"]: "median_metric"})
    )

    pairs = [
        ("VAE2", "VAE1"),
        ("VAE2", "AELG"),
        ("VAE1", "AELG"),
    ]

    for fam_a, fam_b in pairs:
        print(f"### {fam_a} vs {fam_b}\n")

        a_df = agg[agg["backbone_family"] == fam_a][["base_config", "pass_type", "median_metric"]].rename(
            columns={"median_metric": f"{fam_a}_{m['primary_label']}"}
        )
        b_df = agg[agg["backbone_family"] == fam_b][["base_config", "pass_type", "median_metric"]].rename(
            columns={"median_metric": f"{fam_b}_{m['primary_label']}"}
        )

        merged = pd.merge(a_df, b_df, on=["base_config", "pass_type"], how="inner")
        if merged.empty:
            print(f"No matched pairs found for {fam_a} vs {fam_b}.")
            print()
            continue

        a_col = f"{fam_a}_{m['primary_label']}"
        b_col = f"{fam_b}_{m['primary_label']}"
        merged["Delta (B-A)"] = merged[b_col] - merged[a_col]
        merged["Winner"] = merged.apply(
            lambda row: fam_a if row[a_col] < row[b_col]
            else (fam_b if row[b_col] < row[a_col] else "Tie"),
            axis=1,
        )
        merged = merged.rename(columns={"base_config": "Base Config", "pass_type": "Pass"})
        print(merged.to_markdown(index=False, floatfmt=".4f"))
        print()

        a_wins = (merged["Winner"] == fam_a).sum()
        b_wins = (merged["Winner"] == fam_b).sum()
        ties = (merged["Winner"] == "Tie").sum()
        print(f"**Score: {fam_a} {a_wins} — {fam_b} {b_wins} — Tie {ties}**")
        print()


# ---------------------------------------------------------------------------
# Section 9: Final Verdict
# ---------------------------------------------------------------------------

def section_final_verdict(df, m):
    section("9. Final Verdict")
    r_final = df[df["search_round"] == df["search_round"].max()].copy()

    agg = (
        r_final.groupby(["config_name", "pass_type", "backbone_family"], dropna=False)
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
        f"(backbone={best['backbone_family']}, pass={best['pass_type']}) "
        f"with median {m['primary_label']}={best['med_metric']:.4f}."
    )
    if m["has_owa"]:
        delta_g = best["med_metric"] - BASELINES["NBEATS-G"]["owa"]
        verdict_g = "beats" if delta_g < -0.001 else "matches" if abs(delta_g) < 0.005 else "trails"
        print(
            f"vs NBEATS-G ({BASELINES['NBEATS-G']['owa']:.4f}): "
            f"{verdict_g} (delta={delta_g:+.4f})."
        )
    else:
        print(
            f"Primary metric: {m['primary_col']} (lower is better). "
            "OWA baselines not available for this dataset."
        )

    top_cols = ["config_name", "pass_type", "backbone_family", "med_metric", "std_metric", "n_params", "smape", "mase"]
    show = agg[top_cols].head(10).rename(columns={
        "config_name": "Config",
        "pass_type": "Pass",
        "backbone_family": "Backbone",
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
# Notebook generation
# ---------------------------------------------------------------------------

def _generate_notebook(datasets_analyzed, nb_path):
    """Build and write a self-contained Jupyter notebook mirroring the analysis."""
    nb = nbf.v4.new_notebook()
    cells = []

    # ---- Title cell ----
    cells.append(nbf.v4.new_markdown_cell(
        "# VAE2 Block Study — Analysis Notebook\n\n"
        "Generated automatically from `vae2_study_analysis.py`.\n\n"
        "Compares three backbone families: **VAE2** (new), **VAE1** (variational AE), "
        "and **AELG** (learned-gate AE) across datasets using successive-halving results."
    ))

    # ---- Setup cell ----
    setup_code = """\
import os, sys, re
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

pd.set_option('display.max_colwidth', 80)
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_rows', 50)

# Relative paths from notebooks/ directory to results CSVs
VAE2_DATASETS = {
    "m4":     "../../results/m4/vae2_study_results.csv",
    "weather": "../../results/weather/vae2_study_results.csv",
}

BASELINES = {
    "Naive2":   {"owa": 1.0,    "smape": None,  "params": 0},
    "NBEATS-G": {"owa": 0.862,  "smape": 13.70, "params": 24_700_000},
}

NUMERIC_COLS = [
    "smape", "mase", "mae", "mse", "owa", "norm_mae", "norm_mse",
    "n_params", "training_time_seconds", "epochs_trained",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "search_round", "latent_dim",
]

def _pass_from_experiment(exp_str):
    if isinstance(exp_str, str):
        if exp_str.startswith("baseline"):
            return "baseline"
        if exp_str.startswith("activeG_fcast"):
            return "activeG_fcast"
    return "unknown"

def _backbone_base(config_name):
    s = config_name
    s = re.sub(r"VAE2", "BACKBONE", s)
    s = re.sub(r"AELG", "BACKBONE", s)
    s = re.sub(r"VAE1?", "BACKBONE", s)
    return s

def _load_df(csv_path):
    df = pd.read_csv(csv_path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "search_round" in df.columns:
        df["search_round"] = df["search_round"].fillna(1).astype(int)
    for str_col in ("backbone_family", "category_label", "wavelet_family", "latent_dim"):
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(str)
    if "experiment" in df.columns:
        df["pass_type"] = df["experiment"].apply(_pass_from_experiment)
    else:
        df["pass_type"] = "unknown"
    if "config_name" in df.columns:
        df["base_config"] = df["config_name"].apply(_backbone_base)
    return df

def _metric_profile(df):
    owa = pd.to_numeric(df["owa"], errors="coerce") if "owa" in df.columns else pd.Series(dtype=float)
    has_owa = bool(len(owa) and owa.notna().any())
    if has_owa:
        return {"has_owa": True, "primary_col": "owa", "primary_label": "OWA"}
    for fallback in ("norm_mae", "best_val_loss"):
        if fallback in df.columns:
            vals = pd.to_numeric(df[fallback], errors="coerce")
            if vals.notna().any():
                return {"has_owa": False, "primary_col": fallback, "primary_label": fallback}
    return None

print("Setup complete.")
"""
    cells.append(nbf.v4.new_code_cell(setup_code))

    for ds in datasets_analyzed:
        cells.append(nbf.v4.new_markdown_cell(f"---\n## Dataset: `{ds}`"))

        # Data loading cell
        load_code = f"""\
csv_path = VAE2_DATASETS["{ds}"]
df = _load_df(csv_path)
m = _metric_profile(df)

display(Markdown(f"**Rows:** {{len(df)}} | **Unique configs:** {{df['config_name'].nunique()}} | **Primary metric:** `{{m['primary_col']}}`"))
display(df.head())
"""
        cells.append(nbf.v4.new_code_cell(load_code))

        # Section 1: Overview
        cells.append(nbf.v4.new_markdown_cell("### 1. Overview & Data Summary"))
        s1_code = """\
rows = []
for r in sorted(df["search_round"].dropna().unique()):
    rdf = df[df["search_round"] == r]
    passes = sorted(rdf["pass_type"].dropna().unique().tolist())
    rows.append({
        "Round": int(r),
        "Configs": rdf["config_name"].nunique(),
        "Rows": len(rdf),
        "Epochs": (
            f"{int(rdf['epochs_trained'].min())}-{int(rdf['epochs_trained'].max())}"
            if "epochs_trained" in rdf.columns and not rdf.empty else "N/A"
        ),
        "Passes": ", ".join(passes),
    })
overview_df = pd.DataFrame(rows)
display(overview_df)
"""
        cells.append(nbf.v4.new_code_cell(s1_code))

        # Section 2: Funnel
        cells.append(nbf.v4.new_markdown_cell("### 2. Successive Halving Funnel"))
        s2_code = """\
funnel_rows = []
prev_cfgs = None
for r in sorted(df["search_round"].dropna().unique()):
    rdf = df[df["search_round"] == r]
    cfg_count = rdf["config_name"].nunique()
    best = rdf.groupby("config_name")[m["primary_col"]].median().min()
    keep = "-"
    if prev_cfgs is not None:
        keep = f"{cfg_count}/{prev_cfgs} ({cfg_count / prev_cfgs:.0%})"
    funnel_rows.append({
        "Round": int(r),
        "Configs": cfg_count,
        "Rows": len(rdf),
        f"Best Med {m['primary_label']}": round(best, 4),
        "Kept": keep,
    })
    prev_cfgs = cfg_count
display(pd.DataFrame(funnel_rows))
"""
        cells.append(nbf.v4.new_code_cell(s2_code))

        # Section 3: Round Leaderboards
        cells.append(nbf.v4.new_markdown_cell("### 3. Round Leaderboards"))
        s3_code = """\
for r in sorted(df["search_round"].dropna().unique()):
    rdf = df[df["search_round"] == r].copy()
    display(Markdown(f"#### Round {int(r)}"))
    group_cols = ["config_name", "pass_type"]
    agg = (
        rdf.groupby(group_cols, dropna=False)
        .agg(
            mean_metric=(m["primary_col"], "mean"),
            std_metric=(m["primary_col"], "std"),
            mean_smape=("smape", "mean"),
            mean_mase=("mase", "mean"),
            n_params=("n_params", "first"),
            backbone_family=("backbone_family", "first"),
            latent_dim=("latent_dim", "first"),
        )
        .sort_values("mean_metric")
        .reset_index()
        .head(15)
    )
    agg = agg.rename(columns={
        "config_name": "Config", "pass_type": "Pass",
        "mean_metric": m["primary_label"], "std_metric": "Std",
        "mean_smape": "sMAPE", "mean_mase": "MASE",
        "n_params": "Params", "backbone_family": "Backbone", "latent_dim": "LD",
    })
    display_cols = ["Config", "Pass", "Backbone", "LD", m["primary_label"], "Std", "sMAPE", "MASE", "Params"]
    display(agg[[c for c in display_cols if c in agg.columns]])
"""
        cells.append(nbf.v4.new_code_cell(s3_code))

        # Section 4: Hyperparameter Marginals
        cells.append(nbf.v4.new_markdown_cell("### 4. Hyperparameter Marginals (Round 1)"))
        s4_code = """\
r1 = df[df["search_round"] == df["search_round"].min()].copy()
factors = [
    ("backbone_family", "backbone_family"),
    ("category_label", "category_label"),
    ("latent_dim", "latent_dim"),
]
has_wavelet = (
    "wavelet_family" in r1.columns
    and "category_label" in r1.columns
    and r1["category_label"].str.contains("wavelet", na=False).any()
)
if has_wavelet:
    factors.append(("wavelet_family (wavelet configs only)", "wavelet_family"))

for label, factor in factors:
    col = factor
    if col not in r1.columns:
        continue
    subset = r1.copy()
    if factor == "wavelet_family":
        subset = r1[r1["category_label"].str.contains("wavelet", na=False)]
        subset = subset[subset["wavelet_family"].astype(str).str.lower() != "nan"]
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
        .rename(columns={col: "Value", "mean_metric": f"Mean {m['primary_label']}", "std_metric": "Std", "n": "N"})
    )
    display(Markdown(f"**{label}**"))
    display(grp)
"""
        cells.append(nbf.v4.new_code_cell(s4_code))

        # Section 5: Stability
        cells.append(nbf.v4.new_markdown_cell("### 5. Stability Analysis"))
        s5_code = """\
stability_summary = []
for r in sorted(df["search_round"].dropna().unique()):
    rdf = df[df["search_round"] == r]
    spread = (
        rdf.groupby("config_name")[m["primary_col"]]
        .agg(["mean", "std", "min", "max"])
        .assign(range=lambda x: x["max"] - x["min"])
    )
    stability_summary.append({
        "Round": int(r),
        "Mean spread": round(spread["range"].mean(), 4),
        "Max spread": round(spread["range"].max(), 4),
        "Config with max spread": spread["range"].idxmax(),
        "Mean std": round(spread["std"].mean(), 4),
    })
display(pd.DataFrame(stability_summary))
"""
        cells.append(nbf.v4.new_code_cell(s5_code))

        # Section 6: Progression
        cells.append(nbf.v4.new_markdown_cell("### 6. Round-over-Round Progression"))
        s6_code = """\
rounds = sorted(df["search_round"].dropna().unique())
if len(rounds) < 2:
    display(Markdown("*(need >= 2 rounds to show progression)*"))
else:
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
    display(prog.sort_values(prog.columns[-1]).round(4))
"""
        cells.append(nbf.v4.new_code_cell(s6_code))

        # Section 7: Baseline Comparisons
        cells.append(nbf.v4.new_markdown_cell("### 7. Baseline Comparisons (M4 only)"))
        s7_code = f"""\
dataset_name = "{ds}"
if not (dataset_name == "m4" and m["has_owa"]):
    display(Markdown("*Section skipped — M4 OWA baselines not applicable for this dataset.*"))
else:
    r_final = df[df["search_round"] == df["search_round"].max()].copy()
    top = (
        r_final.groupby(["config_name", "pass_type"], dropna=False)
        .agg(
            mean_owa=("owa", "mean"),
            n_params=("n_params", "first"),
            smape=("smape", "mean"),
            backbone_family=("backbone_family", "first"),
        )
        .sort_values("mean_owa")
        .head(10)
        .reset_index()
    )
    top["vs NBEATS-G"] = top["mean_owa"] - BASELINES["NBEATS-G"]["owa"]
    top = top.rename(columns={{
        "config_name": "Config", "pass_type": "Pass", "mean_owa": "OWA",
        "n_params": "Params", "smape": "sMAPE", "backbone_family": "Backbone",
    }})
    display(Markdown("**Top-10 VAE2-Study Configs (Final Round)**"))
    display(top[["Config", "Pass", "Backbone", "OWA", "sMAPE", "Params", "vs NBEATS-G"]])
    display(Markdown("**Reference Baselines**"))
    base_rows = [
        {{"Baseline": name, "OWA": vals["owa"], "sMAPE": vals.get("smape") or "N/A", "Params": vals["params"]}}
        for name, vals in BASELINES.items()
    ]
    display(pd.DataFrame(base_rows).sort_values("OWA"))
"""
        cells.append(nbf.v4.new_code_cell(s7_code))

        # Section 8: Backbone Head-to-Head
        cells.append(nbf.v4.new_markdown_cell("### 8. Backbone Family Head-to-Head"))
        s8_code = """\
final_round = df["search_round"].max()
rdf = df[df["search_round"] == final_round].copy()

agg_h2h = (
    rdf.groupby(["config_name", "pass_type", "backbone_family", "base_config"], dropna=False)[m["primary_col"]]
    .median()
    .reset_index()
    .rename(columns={m["primary_col"]: "median_metric"})
)

pairs = [("VAE2", "VAE1"), ("VAE2", "AELG"), ("VAE1", "AELG")]

for fam_a, fam_b in pairs:
    display(Markdown(f"#### {fam_a} vs {fam_b}"))
    a_df = agg_h2h[agg_h2h["backbone_family"] == fam_a][["base_config", "pass_type", "median_metric"]].rename(
        columns={"median_metric": f"{fam_a}_{m['primary_label']}"}
    )
    b_df = agg_h2h[agg_h2h["backbone_family"] == fam_b][["base_config", "pass_type", "median_metric"]].rename(
        columns={"median_metric": f"{fam_b}_{m['primary_label']}"}
    )
    merged = pd.merge(a_df, b_df, on=["base_config", "pass_type"], how="inner")
    if merged.empty:
        display(Markdown(f"*No matched pairs found for {fam_a} vs {fam_b}.*"))
        continue
    a_col = f"{fam_a}_{m['primary_label']}"
    b_col = f"{fam_b}_{m['primary_label']}"
    merged["Delta (B-A)"] = merged[b_col] - merged[a_col]
    merged["Winner"] = merged.apply(
        lambda row: fam_a if row[a_col] < row[b_col] else (fam_b if row[b_col] < row[a_col] else "Tie"),
        axis=1,
    )
    merged = merged.rename(columns={"base_config": "Base Config", "pass_type": "Pass"})
    display(merged)
    a_wins = (merged["Winner"] == fam_a).sum()
    b_wins = (merged["Winner"] == fam_b).sum()
    ties = (merged["Winner"] == "Tie").sum()
    display(Markdown(f"**Score: {fam_a} {a_wins} — {fam_b} {b_wins} — Tie {ties}**"))
"""
        cells.append(nbf.v4.new_code_cell(s8_code))

        # Section 9: Final Verdict
        cells.append(nbf.v4.new_markdown_cell("### 9. Final Verdict"))
        s9_code = """\
r_final = df[df["search_round"] == df["search_round"].max()].copy()
agg_verdict = (
    r_final.groupby(["config_name", "pass_type", "backbone_family"], dropna=False)
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

if agg_verdict.empty:
    display(Markdown("*No final round results available.*"))
else:
    best = agg_verdict.iloc[0]
    verdict_msg = (
        f"**Best configuration:** `{best['config_name']}` "
        f"(backbone={best['backbone_family']}, pass={best['pass_type']}) "
        f"with median {m['primary_label']}={best['med_metric']:.4f}."
    )
    if m["has_owa"]:
        delta_g = best["med_metric"] - BASELINES["NBEATS-G"]["owa"]
        verdict_g = "beats" if delta_g < -0.001 else ("matches" if abs(delta_g) < 0.005 else "trails")
        verdict_msg += f"  \\nvs NBEATS-G ({BASELINES['NBEATS-G']['owa']:.4f}): {verdict_g} (delta={delta_g:+.4f})."
    display(Markdown(verdict_msg))
    show = agg_verdict[["config_name", "pass_type", "backbone_family", "med_metric", "std_metric", "n_params", "smape", "mase"]].head(10).rename(columns={
        "config_name": "Config", "pass_type": "Pass", "backbone_family": "Backbone",
        "med_metric": f"Med {m['primary_label']}", "std_metric": "Std",
        "n_params": "Params", "smape": "sMAPE", "mase": "MASE",
    })
    display(show)
"""
        cells.append(nbf.v4.new_code_cell(s9_code))

    nb["cells"] = cells
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[notebook] Written to {nb_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main per-dataset analysis
# ---------------------------------------------------------------------------

def analyze_dataset(dataset_name, csv_path):
    if not os.path.exists(csv_path):
        print(f"[SKIP] dataset={dataset_name} reason=missing_csv path={csv_path}")
        return False, "missing_csv"

    df = _load_df(csv_path)
    if df.empty:
        print(f"[SKIP] dataset={dataset_name} reason=empty_csv path={csv_path}")
        return False, "empty_csv"

    m = _metric_profile(df, dataset_name)
    if m is None:
        print(f"[SKIP] dataset={dataset_name} reason=no_usable_metric")
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
    section_backbone_head_to_head(df, m)
    section_final_verdict(df, m)

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="VAE2 Block Study — Multi-Dataset Analysis"
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *list(VAE2_DATASETS.keys())],
        help="Dataset to analyze or 'all'",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path (default: stdout)",
    )
    args = parser.parse_args()

    # Redirect to file if requested
    if args.output:
        fh = open(args.output, "w", encoding="utf-8")
        sys.stdout = fh

    print("# VAE2 Block Study - Multi-Dataset Analysis\n")

    requested = list(VAE2_DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    analyzed = []
    skipped = []

    for ds in requested:
        csv_path = VAE2_DATASETS[ds]
        ok, reason = analyze_dataset(ds, csv_path)
        if ok:
            analyzed.append(ds)
        else:
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

    # Always generate companion notebook
    notebook_dir = os.path.join(_EXPERIMENTS_DIR, "analysis", "notebooks")
    os.makedirs(notebook_dir, exist_ok=True)
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    nb_path = os.path.join(notebook_dir, f"{script_stem}.ipynb")
    _generate_notebook(analyzed, nb_path)


if __name__ == "__main__":
    main()
