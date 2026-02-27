"""Unbalanced active_g Study — Comprehensive Analysis

Reads experiments/results/milk_convergence/milk_convergence_results.csv
Compares four active_g conditions on the Milk6 univariate forecasting task:
  - False      (baseline: no activation on output layers)
  - True       (balanced: both backcast and forecast activated)
  - forecast   (forecast-only activation)
  - backcast   (backcast-only activation)

Produces:
  1. Overview / dataset summary
  2. Convergence rates per condition
  3. Val-loss distribution (best, final, ratio)
  4. Best-epoch distribution
  5. Statistical significance (pairwise Mann-Whitney U)
  6. Stability analysis (val-loss spread across seeds)
  7. Training speed comparison
  8. Notable takeaways and recommendations

Usage:
    python experiments/unbalanced_active_g_analysis.py
    python experiments/unbalanced_active_g_analysis.py > experiments/analysis_reports/unbalanced_active_g_analysis.md
"""

import io, os, sys
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

try:
    from llm_commentary import generate_commentary
    _LLM = True
except ImportError:
    _LLM = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.4f}".format)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(
    _SCRIPT_DIR, "results", "milk_convergence", "milk_convergence_results.csv"
)

NUM_COLS = [
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "epochs_trained", "loss_ratio",
    "n_params", "training_time_seconds",
]

CONFIG_ORDER = [
    "Milk6_baseline",
    "Milk6_activeG",
    "Milk6_activeG_forecastOnly",
    "Milk6_activeG_backcastOnly",
]
ACTIVE_G_LABELS = {
    "Milk6_baseline": "False",
    "Milk6_activeG": "True (balanced)",
    "Milk6_activeG_forecastOnly": "forecast-only",
    "Milk6_activeG_backcastOnly": "backcast-only",
}


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
    print("Convergence rates per condition show how training stability varies with active_g mode.\n")
    rows = []
    for cfg in CONFIG_ORDER:
        grp = df[df["config_name"] == cfg]
        if grp.empty:
            continue
        label = ACTIVE_G_LABELS.get(cfg, cfg)
        n = len(grp)
        n_div = grp["diverged"].sum()
        n_healthy = grp["healthy"].sum()
        n_early = (grp["stopping_reason"] == "EARLY_STOPPED").sum()
        rows.append({"Config": cfg, "active_g": label, "N": n,
                      "Healthy": f"{n_healthy} ({n_healthy/n*100:.0f}%)",
                      "Diverged": f"{n_div} ({n_div/n*100:.0f}%)",
                      "Early Stopped": f"{n_early} ({n_early/n*100:.0f}%)"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def val_loss_stats(df):
    print("Validation loss statistics for healthy runs. The loss ratio (final / best) indicates "
          "overfitting tendency.\n")
    rows = []
    for cfg in CONFIG_ORDER:
        grp = df[(df["config_name"] == cfg) & df["healthy"]]
        if grp.empty:
            rows.append({"Config": cfg, "active_g": ACTIVE_G_LABELS.get(cfg, cfg),
                          "Med Best Loss": "—", "Med Ratio": "—", "Std Ratio": "—"})
            continue
        label = ACTIVE_G_LABELS.get(cfg, cfg)
        rows.append({"Config": cfg, "active_g": label,
                      "Med Best Loss": f"{grp['best_val_loss'].median():.4f}",
                      "Med Ratio": f"{grp['loss_ratio'].median():.4f}",
                      "Std Ratio": f"{grp['loss_ratio'].std():.4f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def best_epoch_distribution(df):
    print("Best-epoch distribution indicates convergence speed for each active_g mode.\n")
    rows = []
    for cfg in CONFIG_ORDER:
        grp = df[(df["config_name"] == cfg) & df["healthy"]]
        if grp.empty:
            continue
        label = ACTIVE_G_LABELS.get(cfg, cfg)
        ep = grp["best_epoch"]
        rows.append({"Config": cfg, "active_g": label,
                      "Med Epoch": f"{ep.median():.1f}", "Min": f"{ep.min():.0f}",
                      "Max": f"{ep.max():.0f}", "Std": f"{ep.std():.2f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def pairwise_significance(df):
    """Pairwise Mann-Whitney U on best_val_loss (all healthy runs)."""
    print("Each pair is tested with a one-sided Mann-Whitney U to determine which condition "
          "produces stochastically lower validation loss.\n")
    samples = {}
    for cfg in CONFIG_ORDER:
        grp = df[(df["config_name"] == cfg) & df["healthy"]]["best_val_loss"].dropna()
        if not grp.empty:
            samples[cfg] = grp.values

    rows = []
    for a, b in combinations(CONFIG_ORDER, 2):
        if a not in samples or b not in samples:
            continue
        sa, sb = samples[a], samples[b]
        med_a, med_b = np.median(sa), np.median(sb)
        delta = med_b - med_a
        alt = "less" if delta < 0 else "greater"
        stat, p = stats.mannwhitneyu(sb, sa, alternative=alt)
        sign = "+" if delta >= 0 else ""
        rows.append({"Config A": a, "Config B": b, "Δ med": f"{sign}{delta:.4f}",
                      "p-value": f"{p:.4f}", "Sig": sig_stars(p)})
    print(pd.DataFrame(rows).to_markdown(index=False))
    sig_pairs = sum(1 for r in rows if r["Sig"] != "ns")
    print(f"\n{sig_pairs} of {len(rows)} pairs show statistically significant differences (p < 0.05).")

    # LLM interpretation of active_g modes
    val_stats = {
        cfg: {
            "label": ACTIVE_G_LABELS.get(cfg, cfg),
            "med_loss": float(np.median(samples[cfg])) if cfg in samples else None,
        }
        for cfg in CONFIG_ORDER if cfg in samples
    }
    best_cfg = min(val_stats, key=lambda c: val_stats[c]["med_loss"] or float("inf"))
    llm_ctx = {
        "parameter_name": "active_g",
        "architecture_name": "Generic 6-stack",
        "backcast_length": 36,
        "forecast_length": 6,
        "best_value": ACTIVE_G_LABELS.get(best_cfg, best_cfg),
        "best_owa": val_stats[best_cfg]["med_loss"] or 0.0,
        "worst_value": ACTIVE_G_LABELS.get(
            max(val_stats, key=lambda c: val_stats[c]["med_loss"] or 0.0), ""
        ),
        "worst_owa": max(v["med_loss"] or 0.0 for v in val_stats.values()),
        "delta": float(
            max(v["med_loss"] or 0.0 for v in val_stats.values()) -
            min(v["med_loss"] or 0.0 for v in val_stats.values())
        ),
        "all_values": [
            {"value": v["label"], "med_owa": v["med_loss"]}
            for v in val_stats.values()
            if v["med_loss"] is not None
        ],
    }
    llm_text = generate_commentary("hyperparameter_discussion", llm_ctx) if _LLM else None
    if llm_text:
        print(f"\n{llm_text}")
    print()


def stability(df):
    print("Stability is measured by the spread (max − min) and standard deviation of best_val_loss "
          "across random seeds.\n")
    rows = []
    for cfg in CONFIG_ORDER:
        grp = df[(df["config_name"] == cfg) & df["healthy"]]["best_val_loss"].dropna()
        if grp.empty:
            continue
        label = ACTIVE_G_LABELS.get(cfg, cfg)
        rows.append({"Config": cfg, "active_g": label,
                      "Med Loss": f"{grp.median():.4f}",
                      "Range": f"{grp.max()-grp.min():.4f}",
                      "Std": f"{grp.std():.4f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def training_speed(df):
    print("Wall-clock training time per condition.\n")
    rows = []
    for cfg in CONFIG_ORDER:
        grp = df[df["config_name"] == cfg]
        if grp.empty:
            continue
        label = ACTIVE_G_LABELS.get(cfg, cfg)
        rows.append({"Config": cfg, "active_g": label,
                      "Med Time (s)": f"{grp['training_time_seconds'].median():.1f}",
                      "Med Epochs": f"{grp['epochs_trained'].median():.1f}"})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()


def main():
    df = load()
    total_runs = len(df)
    total_configs = df["config_name"].nunique()
    total_time = df["training_time_seconds"].sum()

    print("# Unbalanced active_g Study — Results Analysis\n")

    # --- Abstract ---
    healthy = df[df["healthy"]]
    best_cfg = healthy.groupby("config_name")["best_val_loss"].median().idxmin() if not healthy.empty else "N/A"
    best_loss = healthy.groupby("config_name")["best_val_loss"].median().min() if not healthy.empty else float("nan")

    print("## Abstract\n")
    print("This study investigates four `active_g` modes — False (baseline), True (balanced), "
          "forecast-only, and backcast-only — on the Milk6 univariate forecasting task with "
          "a 6-stack Generic architecture. The experiment measures convergence stability, "
          "validation loss, and pairwise statistical significance to determine whether "
          "asymmetric activation on the backcast or forecast output layers affects training.\n")
    print("**Key Takeaways:**\n")
    print(f"- **Best condition:** `{best_cfg}` achieves median best_val_loss = **{best_loss:.4f}**.")
    print(f"- **Conditions tested:** {list(ACTIVE_G_LABELS.values())}")
    print(f"- **Total compute:** {total_time / 60:.1f} minutes across {total_runs} runs "
          f"({total_runs // total_configs} runs per condition).\n")

    print("**Condition Mapping:**\n")
    cond_rows = [{"Config": cfg, "active_g": label} for cfg, label in ACTIVE_G_LABELS.items()]
    print(pd.DataFrame(cond_rows).to_markdown(index=False))
    print()
    print(f"- **CSV:** `{CSV_PATH}`")
    print(f"- **Rows:** {total_runs} ({total_configs} configs)")
    print(f"- **Primary metric:** best_val_loss (lower = better; no OWA for this dataset).")
    print()

    section("1. Convergence Rates")
    convergence_rates(df)

    section("2. Validation Loss Statistics (Healthy Runs)")
    val_loss_stats(df)

    section("3. Best-Epoch Distribution (Healthy Runs)")
    best_epoch_distribution(df)

    section("4. Stability Across Seeds")
    stability(df)

    section("5. Training Speed")
    training_speed(df)

    section("6. Pairwise Statistical Significance (Mann-Whitney U)")
    pairwise_significance(df)


if __name__ == "__main__":
    main()

