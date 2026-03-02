"""Milk Convergence 10-Stack Study — Comprehensive Analysis

Reads experiments/results/milk_convergence_10stack/milk_convergence_10stack_results.csv
Compares Generic 10-stack baseline vs active_g=True on a simple univariate series.

Produces:
  1. Overview / dataset summary
  2. Convergence rates (divergence / healthy / early-stopped)
  3. Val-loss distribution (best, final, loss ratio)
  4. Best-epoch distribution (how many epochs to converge)
  5. Training speed comparison
  6. Stability: val-loss spread across seeds
  7. Active_g effect head-to-head
  8. Notable takeaways

Usage:
    python experiments/analysis/milk_convergence_10stack_analysis.py
    python experiments/analysis/milk_convergence_10stack_analysis.py > experiments/analysis_reports/milk_convergence_10stack_analysis.md
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy import stats

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EXPERIMENTS_DIR)

try:
    from llm_commentary import generate_commentary
    _LLM = True
except ImportError:
    _LLM = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.4f}".format)

CSV_PATH = os.path.join(
    _EXPERIMENTS_DIR, "results", "milk_convergence_10stack", "milk_convergence_10stack_results.csv"
)

NUM_COLS = [
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "epochs_trained", "loss_ratio",
    "n_params", "training_time_seconds",
]


def section(title):
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
    df["diverged"] = df["diverged"].astype(str).str.lower().isin(["true", "1"])
    df["healthy"] = df["healthy"].astype(str).str.lower().isin(["true", "1"])
    return df


def convergence_rates(df):
    print("Convergence rates indicate how many runs completed training without divergence.\n")
    rows = []
    for cfg, grp in df.groupby("config_name"):
        n = len(grp)
        n_div = grp["diverged"].sum()
        n_healthy = grp["healthy"].sum()
        n_early = (grp["stopping_reason"] == "EARLY_STOPPED").sum()
        n_max = (grp["stopping_reason"] == "MAX_EPOCHS").sum()
        rows.append({"Config": cfg, "N": n,
                      "Healthy": f"{n_healthy} ({n_healthy/n*100:.0f}%)",
                      "Diverged": f"{n_div} ({n_div/n*100:.0f}%)",
                      "Early Stopped": f"{n_early} ({n_early/n*100:.0f}%)",
                      "Max Epochs": f"{n_max} ({n_max/n*100:.0f}%)"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    # Commentary
    total_div = df["diverged"].sum()
    total = len(df)
    print(f"\nOverall divergence rate: {total_div}/{total} ({total_div/total*100:.1f}%). "
          f"{'All runs healthy — training is stable.' if total_div == 0 else 'Some runs diverged; consider learning-rate tuning.'}\n")


def val_loss_stats(df):
    print("The loss ratio (final / best) measures overfitting: values near 1.0 indicate stable convergence, "
          "while values >> 1.0 suggest the model overfit past its best epoch.\n")
    rows = []
    for cfg, grp in df.groupby("config_name"):
        healthy = grp[grp["healthy"]]
        if healthy.empty:
            rows.append({"Config": cfg, "Med Best Loss": "—", "Med Final Loss": "—",
                          "Med Ratio": "—", "Std Ratio": "—"})
            continue
        rows.append({"Config": cfg,
                      "Med Best Loss": f"{healthy['best_val_loss'].median():.4f}",
                      "Med Final Loss": f"{healthy['final_val_loss'].median():.4f}",
                      "Med Ratio": f"{healthy['loss_ratio'].median():.4f}",
                      "Std Ratio": f"{healthy['loss_ratio'].std():.4f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def best_epoch_distribution(df):
    print("The best epoch indicates how quickly each configuration converges to its optimal validation loss.\n")
    rows = []
    for cfg, grp in df.groupby("config_name"):
        healthy = grp[grp["healthy"]]
        if healthy.empty:
            continue
        ep = healthy["best_epoch"]
        rows.append({"Config": cfg, "Med Epoch": f"{ep.median():.1f}",
                      "Min": f"{ep.min():.0f}", "Max": f"{ep.max():.0f}",
                      "Std": f"{ep.std():.2f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def training_speed(df):
    print("Training speed comparison (wall-clock time per configuration).\n")
    rows = []
    for cfg, grp in df.groupby("config_name"):
        med_time = grp["training_time_seconds"].median()
        med_ep = grp["epochs_trained"].median()
        per_epoch = med_time / med_ep if med_ep > 0 else float("nan")
        rows.append({"Config": cfg, "Med Time (s)": f"{med_time:.1f}",
                      "Med Epochs": f"{med_ep:.1f}", "s/epoch": f"{per_epoch:.2f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def activeg_head_to_head(df):
    print("A one-sided Mann-Whitney U test checks whether `active_g=True` produces "
          "stochastically lower validation loss than the baseline.\n")
    baseline = df[df["config_name"] == "Milk10_baseline"]["best_val_loss"].dropna()
    activeg = df[df["config_name"] == "Milk10_activeG"]["best_val_loss"].dropna()
    if baseline.empty or activeg.empty:
        print("(Missing data for comparison.)\n")
        return
    stat, p = stats.mannwhitneyu(activeg, baseline, alternative="less")
    delta = activeg.median() - baseline.median()
    sign = "+" if delta >= 0 else ""
    rows = [
        {"Condition": "baseline", "Med best_val_loss": f"{baseline.median():.4f}", "N": len(baseline)},
        {"Condition": "active_g", "Med best_val_loss": f"{activeg.median():.4f}", "N": len(activeg)},
    ]
    print(pd.DataFrame(rows).to_markdown(index=False))
    print(f"\n- **Δ (active_g − baseline):** {sign}{delta:.4f}")
    print(f"- **Mann-Whitney U p-value:** {p:.4f} {sig_stars(p)}")
    verdict = "Active_g produces significantly lower loss." if delta < 0 and p < 0.05 else \
              "No significant difference; baseline is comparable or better."
    print(f"- **Verdict:** {verdict}")

    llm_ctx = {
        "epoch_stats": {
            "baseline_med_loss": float(baseline.median()),
            "activeg_med_loss": float(activeg.median()),
            "delta": float(delta),
            "p_value": float(p),
            "significant": bool(delta < 0 and p < 0.05),
        },
        "context_extra": "Milk univariate 10-stack convergence study comparing baseline vs active_g=True",
    }
    llm_text = generate_commentary("convergence_analysis", llm_ctx) if _LLM else None
    if llm_text:
        print(f"\n{llm_text}")
    print()


def main():
    df = load()
    total_runs = len(df)
    total_configs = df["config_name"].nunique()
    total_time = df["training_time_seconds"].sum()

    print("# Milk Convergence 10-Stack Study — Results Analysis\n")

    # --- Abstract ---
    healthy = df[df["healthy"]]
    configs = list(df["config_name"].unique())
    best_cfg = healthy.groupby("config_name")["best_val_loss"].median().idxmin() if not healthy.empty else "N/A"
    best_loss = healthy.groupby("config_name")["best_val_loss"].median().min() if not healthy.empty else float("nan")
    div_rate = df["diverged"].sum() / len(df) * 100

    print("## Abstract\n")
    print("This convergence study compares a 10-stack Generic N-BEATS baseline against "
          "the same architecture with `active_g=True` on the Milk univariate time series. "
          "Unlike the M4 benchmark studies, this experiment uses **validation loss** as the "
          "primary metric (no OWA) and focuses on training stability and convergence behaviour.\n")
    print("**Key Takeaways:**\n")
    print(f"- **Best configuration:** `{best_cfg}` achieves median best val_loss = **{best_loss:.4f}**.")
    print(f"- **Divergence rate:** {div_rate:.0f}% of runs diverged across all conditions.")
    print(f"- **Configurations tested:** {configs}")
    print(f"- **Total compute:** {total_time / 60:.1f} minutes across {total_runs} runs.\n")

    print(f"- **CSV:** `{CSV_PATH}`")
    print(f"- **Rows:** {total_runs} ({total_configs} unique configs)")
    print(f"- **Parameters per model:** {df['n_params'].iloc[0]:,.0f}")
    print()

    section("1. Convergence Rates")
    convergence_rates(df)

    section("2. Validation Loss Statistics (Healthy Runs)")
    val_loss_stats(df)

    section("3. Best-Epoch Distribution (Healthy Runs)")
    best_epoch_distribution(df)

    section("4. Training Speed")
    training_speed(df)

    section("5. active_g Head-to-Head (Mann-Whitney U)")
    activeg_head_to_head(df)


if __name__ == "__main__":
    main()

