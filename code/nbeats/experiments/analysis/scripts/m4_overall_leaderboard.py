"""m4_overall_leaderboard.py

Builds the unified M4 leaderboard across every CSV under experiments/results/m4/.
Treats paper-sample (`nbeats_paper`) and sliding (`sliding`) as separate tracks.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experiments" / "results" / "m4"

PERIODS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]


# ---------- protocol classification --------------------------------------------------

# Empirically verified mapping. Files with "paper_sample" or "tiered_offset" always
# use the nbeats_paper protocol. Files with "comprehensive_sweep" use sliding. Older
# omnibus / unified / study CSVs predate the new protocol field and were all sliding.
PAPER_SAMPLE_FILES = {
    "comprehensive_m4_paper_sample_results.csv",
    "comprehensive_m4_paper_sample_sym10_fills_results.csv",
    "comprehensive_m4_paper_sample_plateau_results.csv",
    "tiered_offset_m4_allperiods_results.csv",
    "tiered_offset_m4_allperiods_paperlr_results.csv",
    "tiered_offset_m4_weekly_plateau_validation_results.csv",
    "m4_hourly_sym10_tiered_offset_results.csv",
    "m4_hourly_sym10_tiered_offset_paperlr_results.csv",
    "test_earlystop_fix_results.csv",
}

SLIDING_FILES = {
    "comprehensive_sweep_m4_results.csv",
}

# legacy / specialty files (sliding protocol, but heterogeneous schemas / partial coverage)
LEGACY_SLIDING_FILES = {
    "ablation_results.csv",
    "ae_trend_search_results.csv",
    "block_benchmark_results.csv",
    "convergence_study_results_v1.csv",
    "ensemble_individual_results.csv",
    "ensemble_summary_results.csv",
    "gate_fn_study_results.csv",
    "generic_ae_pure_stack_results.csv",
    "generic_dim_sweep_results.csv",
    "kl_weight_sweep_m4_results.csv",
    "lg_vae_study_results.csv",
    "omnibus_benchmark_results.csv",
    "resnet_skip_study_results.csv",
    "resnet_skip_study_v2_results.csv",
    "trend_seas_wav_comparison_results.csv",
    "trendae_study_results.csv",
    "trendwavelet_generic_effectiveness_results.csv",
    "trendwavelet_twidth_validation_m4_results.csv",
    "trendwaveletae_pure_study_results.csv",
    "trendwaveletae_v2_study_results.csv",
    "trendwaveletaelg_pure_study_results.csv",
    "trendwaveletaelg_pure_v2_study_results.csv",
    "unified_benchmark_results.csv",
    "v3aelg_stackheight_sweep_results.csv",
    "vae2_study_results.csv",
    "wavelet_search_results.csv",
    "wavelet_study_2_basis_dim_results.csv",
    "wavelet_study_3_successive_results.csv",
    "wavelet_study_3_successive_trendae_results.csv",
    "wavelet_study_results.csv",
    "wavelet_trendae_comparison_results.csv",
    "wavelet_v2_benchmark_results.csv",
    "wavelet_v3_benchmark_results.csv",
    "wavelet_v3ae_study_results.csv",
    "wavelet_v3aelg_trendaelg_study_results.csv",
}


# ---------- loading ------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:  # pragma: no cover
        print(f"  skip {path.name}: {e}", file=sys.stderr)
        return None
    if df.empty:
        return None
    df["_source_file"] = path.name
    return df


def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def apply_strict_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "diverged" in df.columns:
        diverged = df["diverged"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        diverged = pd.Series(False, index=df.index)

    smape = pd.to_numeric(df.get("smape", pd.Series(np.nan, index=df.index)), errors="coerce")
    best_ep = pd.to_numeric(
        df.get("best_epoch", pd.Series(np.nan, index=df.index)), errors="coerce"
    )

    bad = (
        diverged
        | (smape > 100)
        | ((best_ep == 0) & (smape > 50))
        | smape.isna()
    )
    return df.loc[~bad].copy()


# ---------- ingestion ----------------------------------------------------------------

def load_all() -> pd.DataFrame:
    frames = []
    files = sorted(RESULTS_DIR.glob("*.csv"))
    for p in files:
        if p.name.endswith(".lock"):
            continue
        df = load_csv(p)
        if df is None:
            continue
        # tag protocol
        if p.name in PAPER_SAMPLE_FILES:
            proto = "paper_sample"
        elif p.name in SLIDING_FILES:
            proto = "sliding"
        elif p.name in LEGACY_SLIDING_FILES:
            proto = "legacy_sliding"
        else:
            proto = "unknown"
        df["_protocol"] = proto

        # coerce key numerics
        df = coerce_numeric(
            df,
            [
                "smape",
                "owa",
                "mase",
                "mae",
                "mse",
                "n_params",
                "best_epoch",
                "epochs_trained",
            ],
        )
        # lr scheduler tag
        if p.name == "comprehensive_m4_paper_sample_plateau_results.csv":
            df["_lr_sched"] = "plateau"
        elif p.name == "tiered_offset_m4_allperiods_paperlr_results.csv":
            df["_lr_sched"] = "step_paper"
        elif p.name == "tiered_offset_m4_allperiods_results.csv":
            df["_lr_sched"] = "plateau"
        elif p.name == "tiered_offset_m4_weekly_plateau_validation_results.csv":
            df["_lr_sched"] = "plateau"
        elif p.name == "m4_hourly_sym10_tiered_offset_paperlr_results.csv":
            df["_lr_sched"] = "step_paper"
        elif p.name == "m4_hourly_sym10_tiered_offset_results.csv":
            df["_lr_sched"] = "plateau"
        elif p.name == "comprehensive_m4_paper_sample_results.csv":
            df["_lr_sched"] = "step_paper"
        elif p.name == "comprehensive_m4_paper_sample_sym10_fills_results.csv":
            df["_lr_sched"] = "step_paper"
        elif p.name == "comprehensive_sweep_m4_results.csv":
            df["_lr_sched"] = "cosine_warmup"
        else:
            df["_lr_sched"] = "unknown"

        frames.append(df)

    full = pd.concat(frames, ignore_index=True, sort=False)
    return full


# ---------- canonicalization ---------------------------------------------------------

# The dedicated Hourly tiered files (m4_hourly_sym10_tiered_offset{,_paperlr}_results.csv)
# encode tiering direction in the config name suffix (e.g. T+Sym10V3_10s_bdEQ_ascend),
# whereas the all-periods tiered files (tiered_offset_m4_allperiods{,_paperlr}_results.csv)
# encode it as a separate `tiered` column on a single canonical config name
# (e.g. T+Sym10V3_10s_tiered_ag0). Without canonicalization, identical architectures
# appear under different config_name values, breaking generalist mean-rank computations.
#
# Map below is from Hourly-file naming -> all-periods canonical naming. Applied only
# to rows from the Hourly tiered files.
HOURLY_TO_ALLPERIODS_NAME = {
    "T+Sym10V3_10s_bdEQ_ascend": "T+Sym10V3_10s_tiered_ag0",
    "T+Sym10V3_10s_bdEQ_descend": "T+Sym10V3_10s_tiered_ag0",
    "TAE+Sym10V3AE_10s_ld32_bdEQ_ascend": "TAE+Sym10V3AE_10s_tiered",
    "TAE+Sym10V3AE_10s_ld32_bdEQ_descend": "TAE+Sym10V3AE_10s_tiered",
    # The TW / TWAE unified blocks have no tiered all-periods sibling; we keep them
    # under a canonical "_tiered" name for cross-period grouping that mirrors the
    # all-periods naming convention.
    "TW_10s_td3_sym10_bdEQ_ascend": "TW_10s_td3_sym10_tiered",
    "TW_10s_td3_sym10_bdEQ_descend": "TW_10s_td3_sym10_tiered",
    "TWAE_10s_td3_sym10_ld16_bdEQ_ascend": "TWAE_10s_td3_sym10_ld16_tiered",
    "TWAE_10s_td3_sym10_ld16_bdEQ_descend": "TWAE_10s_td3_sym10_ld16_tiered",
}

HOURLY_TIERED_FILES = {
    "m4_hourly_sym10_tiered_offset_results.csv",
    "m4_hourly_sym10_tiered_offset_paperlr_results.csv",
}


def canonicalize_config_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite config_name on Hourly tiered file rows to match all-periods naming.

    Preserves the original name in `_orig_config_name` and the direction in
    `_tiered_direction` for traceability."""
    df = df.copy()
    df["_orig_config_name"] = df["config_name"]
    if "_tiered_direction" not in df.columns:
        df["_tiered_direction"] = pd.NA

    # propagate direction from tiered/offset_direction columns
    if "offset_direction" in df.columns:
        mask_od = df["offset_direction"].notna()
        df.loc[mask_od, "_tiered_direction"] = df.loc[mask_od, "offset_direction"]
    if "tiered" in df.columns:
        mask_t = df["tiered"].notna() & df["_tiered_direction"].isna()
        df.loc[mask_t, "_tiered_direction"] = df.loc[mask_t, "tiered"]

    # canonicalize Hourly-tiered names
    is_hourly_tiered = df["_source_file"].isin(HOURLY_TIERED_FILES)
    if is_hourly_tiered.any():
        renamed = df.loc[is_hourly_tiered, "config_name"].map(HOURLY_TO_ALLPERIODS_NAME)
        # only overwrite where mapping exists; leave others unchanged
        df.loc[is_hourly_tiered & renamed.notna(), "config_name"] = renamed[renamed.notna()]
    return df


# ---------- analysis -----------------------------------------------------------------

def _aggregate_with_best_direction(
    sub: pd.DataFrame,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate by (config_name, protocol, source_file, lr_sched), preserving
    direction-as-disambiguator: aggregate first by direction within each canonical
    (config, protocol, source, lr) cell, then keep the best-SMAPE direction.

    This is what the user requested: 'taking the best direction's SMAPE per LR'.
    Configs without tiered direction (most of them) pass through unchanged because
    `_tiered_direction` is NaN/null and the inner groupby produces a single row."""
    extra_cols = extra_cols or []
    group_keys = ["config_name", "_protocol", "_source_file", "_lr_sched"]
    inner_keys = group_keys + ["_tiered_direction"]
    inner = (
        sub.groupby(inner_keys, dropna=False)
        .agg(
            smape_mean=("smape", "mean"),
            smape_std=("smape", "std"),
            owa_mean=("owa", "mean"),
            params=("n_params", "median"),
            n=("smape", "count"),
        )
        .reset_index()
    )
    # Within each (config, protocol, source, lr), keep the best-SMAPE direction.
    inner = inner.sort_values("smape_mean")
    best = inner.drop_duplicates(subset=group_keys, keep="first").reset_index(drop=True)
    return best


def per_period_leaderboard(df: pd.DataFrame, top_k: int = 10) -> dict[str, pd.DataFrame]:
    out = {}
    for period in PERIODS:
        sub = df[df["period"] == period]
        if sub.empty:
            continue
        agg = _aggregate_with_best_direction(sub)
        agg = agg[agg["n"] >= 3]  # require min coverage
        agg = agg.sort_values("smape_mean").reset_index(drop=True)
        out[period] = agg.head(top_k)
    return out


def per_period_full(df: pd.DataFrame, min_n: int = 5) -> dict[str, pd.DataFrame]:
    out = {}
    for period in PERIODS:
        sub = df[df["period"] == period]
        if sub.empty:
            continue
        agg = _aggregate_with_best_direction(sub)
        agg = agg[agg["n"] >= min_n]
        out[period] = agg.sort_values("smape_mean").reset_index(drop=True)
    return out


def generalist_ranking(per_period_full_map: dict[str, pd.DataFrame], protocol: str) -> pd.DataFrame:
    """Compute mean rank across all 6 periods for a given protocol.

    For configs that appear in multiple source files (e.g. NBEATS-IG_30s_ag0 in both
    comprehensive_m4_paper_sample_results and the plateau companion), we KEEP THE
    BEST mean SMAPE per period — i.e. the single best run pool wins."""
    proto_filter = {"paper_sample"} if protocol == "paper_sample" else {"sliding"}
    period_ranks = {}
    for period, df_p in per_period_full_map.items():
        d = df_p[df_p["_protocol"].isin(proto_filter)].copy()
        # collapse duplicate config_name to its best SMAPE
        d = d.sort_values("smape_mean").drop_duplicates("config_name", keep="first")
        d["rank"] = d["smape_mean"].rank(method="min")
        period_ranks[period] = d.set_index("config_name")["rank"]

    rank_df = pd.DataFrame(period_ranks)
    rank_df = rank_df.dropna(thresh=4)  # require at least 4/6 periods
    rank_df["mean_rank"] = rank_df.mean(axis=1, skipna=True)
    rank_df["n_periods"] = rank_df.notna().sum(axis=1) - 1
    return rank_df.sort_values("mean_rank")


# ---------- main ---------------------------------------------------------------------

def main():
    print("Loading all M4 CSVs...", file=sys.stderr)
    raw = load_all()
    print(f"  raw rows: {len(raw):,}", file=sys.stderr)
    print(f"  protocols: {raw['_protocol'].value_counts().to_dict()}", file=sys.stderr)

    # canonicalize Hourly tiered config names BEFORE filtering / aggregation so
    # that the same architecture spans Hourly and the rest of the periods.
    raw = canonicalize_config_names(raw)
    n_canonicalized = (raw["_orig_config_name"] != raw["config_name"]).sum()
    print(f"  canonicalized rows: {n_canonicalized:,}", file=sys.stderr)

    # apply strict divergence filter
    df = apply_strict_filter(raw)
    print(f"  filtered rows: {len(df):,} (dropped {len(raw) - len(df):,})", file=sys.stderr)

    # only the modern, well-documented sweeps (paper_sample, sliding).
    # legacy_sliding is reported but excluded from leaderboards (heterogeneous schemas/protocols).
    main_df = df[df["_protocol"].isin({"paper_sample", "sliding"})].copy()
    print(f"  main analysis rows: {len(main_df):,}", file=sys.stderr)

    # PAPER-SAMPLE leaderboard
    print("\n" + "=" * 80)
    print("PAPER-SAMPLE PROTOCOL — per-period top-10")
    print("=" * 80)
    ps = main_df[main_df["_protocol"] == "paper_sample"]
    ps_lb = per_period_leaderboard(ps, top_k=10)
    for period, table in ps_lb.items():
        print(f"\n{period}")
        cols = ["config_name", "_source_file", "_lr_sched", "smape_mean", "smape_std", "params", "n"]
        print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # SLIDING leaderboard
    print("\n" + "=" * 80)
    print("SLIDING PROTOCOL — per-period top-10")
    print("=" * 80)
    sl = main_df[main_df["_protocol"] == "sliding"]
    sl_lb = per_period_leaderboard(sl, top_k=10)
    for period, table in sl_lb.items():
        print(f"\n{period}")
        cols = ["config_name", "_source_file", "_lr_sched", "smape_mean", "smape_std", "params", "n"]
        print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # generalist rankings
    full = per_period_full(main_df, min_n=5)

    print("\n" + "=" * 80)
    print("PAPER-SAMPLE generalist mean-rank top-15")
    print("=" * 80)
    gen_ps = generalist_ranking(full, protocol="paper_sample")
    print(gen_ps.head(15).round(2).to_string())

    print("\n" + "=" * 80)
    print("SLIDING generalist mean-rank top-15")
    print("=" * 80)
    gen_sl = generalist_ranking(full, protocol="sliding")
    print(gen_sl.head(15).round(2).to_string())

    # Sub-1M / sub-5M champions per period (paper_sample)
    print("\n" + "=" * 80)
    print("PAPER-SAMPLE sub-1M / sub-5M champions per period")
    print("=" * 80)
    for period in PERIODS:
        sub = ps[ps["period"] == period]
        if sub.empty:
            continue
        agg = _aggregate_with_best_direction(sub)
        agg = agg[agg["n"] >= 5]
        sub1m = agg[agg["params"] < 1e6].nsmallest(3, "smape_mean")
        sub5m = agg[(agg["params"] >= 1e6) & (agg["params"] < 5e6)].nsmallest(3, "smape_mean")
        print(f"\n{period} (paper_sample)")
        print(" sub-1M:")
        if not sub1m.empty:
            print(sub1m.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            print("  (none)")
        print(" sub-5M:")
        if not sub5m.empty:
            print(sub5m.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            print("  (none)")

    # LR scheduler comparison on the tiered files (have step vs plateau matched runs)
    print("\n" + "=" * 80)
    print("LR scheduler head-to-head — tiered all-periods (plateau vs step_paper)")
    print("=" * 80)
    tiered = df[
        df["_source_file"].isin(
            {
                "tiered_offset_m4_allperiods_results.csv",
                "tiered_offset_m4_allperiods_paperlr_results.csv",
            }
        )
    ]
    if not tiered.empty:
        # Aggregate with best-direction so ascend/descend collapse to one row.
        rows = []
        for period in tiered["period"].unique():
            agg_p = _aggregate_with_best_direction(tiered[tiered["period"] == period])
            agg_p["period"] = period
            rows.append(agg_p)
        wide = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        # pivot
        pv = wide.pivot_table(
            index=["config_name", "period"],
            columns="_lr_sched",
            values="smape_mean",
        ).reset_index()
        pv["delta_plateau_minus_step"] = pv["plateau"] - pv["step_paper"]
        per_period_lr = (
            pv.groupby("period")["delta_plateau_minus_step"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        per_period_lr.columns = ["period", "mean_delta", "median_delta", "n_pairs"]
        print(per_period_lr.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        print("\n  Negative delta = plateau better than step_paper.")

    # plateau vs step paper-sample comprehensive baseline
    print("\n" + "=" * 80)
    print("LR scheduler — comprehensive_m4_paper_sample plateau vs step")
    print("=" * 80)
    psall = df[
        df["_source_file"].isin(
            {
                "comprehensive_m4_paper_sample_results.csv",
                "comprehensive_m4_paper_sample_plateau_results.csv",
            }
        )
    ]
    if not psall.empty:
        wide = (
            psall.groupby(["config_name", "period", "_lr_sched"], dropna=False)
            .agg(smape_mean=("smape", "mean"), n=("smape", "count"))
            .reset_index()
        )
        pv = wide.pivot_table(
            index=["config_name", "period"],
            columns="_lr_sched",
            values="smape_mean",
        ).reset_index()
        if "plateau" in pv.columns and "step_paper" in pv.columns:
            pv["delta_plateau_minus_step"] = pv["plateau"] - pv["step_paper"]
            per_period_lr = (
                pv.dropna(subset=["delta_plateau_minus_step"])
                .groupby("period")["delta_plateau_minus_step"]
                .agg(["mean", "median", "count"])
                .reset_index()
            )
            per_period_lr.columns = ["period", "mean_delta", "median_delta", "n_pairs"]
            print(per_period_lr.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Hourly tiered head-to-head
    print("\n" + "=" * 80)
    print("Hourly tiered — top configs by source")
    print("=" * 80)
    h = df[(df["period"] == "Hourly")]
    if not h.empty:
        h_agg = _aggregate_with_best_direction(h)
        h_agg = h_agg[h_agg["n"] >= 5].sort_values("smape_mean").head(20)
        print(h_agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Also print the canonicalized Hourly tiered with direction breakout, for
        # traceability against the original CSV names.
        print("\n  --- Hourly tiered (direction breakout, original config_name preserved) ---")
        h_tiered = h[h["_source_file"].isin(HOURLY_TIERED_FILES)]
        if not h_tiered.empty:
            ht = (
                h_tiered.groupby(
                    ["config_name", "_orig_config_name", "_tiered_direction", "_lr_sched"],
                    dropna=False,
                )
                .agg(
                    smape_mean=("smape", "mean"),
                    smape_std=("smape", "std"),
                    params=("n_params", "median"),
                    n=("smape", "count"),
                )
                .reset_index()
                .sort_values("smape_mean")
            )
            print(ht.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
