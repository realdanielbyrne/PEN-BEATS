# Trend Backbone Selection for Wavelet Stacks

## Core Guideline

When building Trend+Wavelet stacks, the **backbone class hierarchy** matters more than individual block choice:

**RootBlock (non-AE) > AERootBlockLG (learned-gate AE) >> AERootBlock (plain AE)**

**Important exception (gate fn study, 2026-03-15):** On **Weather-96**, the hierarchy reverses for AE vs AELG: plain AE significantly outperforms AELG (MWU p=0.036). The top 2 Weather configs are AE controls (no learned gate). The learned gate may over-constrain the latent space for long-horizon forecasting. **For Weather-96 alternating stacks, prefer AE over AELG.**

| Full Stack | Backbone | M4-Yearly SMAPE | OWA |
|---|---|---|---|
| Trend + WaveletV3 | RootBlock | **13.410** | **0.794** |
| TrendAELG + WaveletV3AELG | AERootBlockLG | 13.438 | 0.795 |
| TrendAE + WaveletV3AE | AERootBlock | 15.020 | 0.894 |

The plain AE bottleneck (`AERootBlock`) is **not viable** for alternating wavelet stacks — it destroys information and homogenizes wavelet families (η²=0.003, ns). The learned gate in `AERootBlockLG` is essential for AE-family competitiveness in alternating configurations.

**Depth scaling exception (skip study v2):** For GenericAE (homogeneous stacks), `AERootBlock` is actually **more depth-stable** than `AERootBlockLG`. GenericAE maintains SMAPE ~15.2 at 30 stacks, while GenericAELG collapses to SMAPE ~36 without skip connections. The learned gate amplifies gradient issues at depth. For unified TrendWavelet blocks, both AE and AELG perform equivalently at all depths (10-30 stacks).

Within the top two tiers (non-AE and AELG), **both Trend and TrendAE companion blocks reach comparable top-tier accuracy**. Prefer TrendAE when parameter efficiency matters; prefer Trend for marginally better average-case performance. This is a **soft guideline** based on paired statistical tests across two datasets (Cohen's d < 0.2).

---

## Recommendations

### Default: TrendAE
- ~16% fewer parameters (4.25M vs 5.10M on M4-Yearly)
- R3 best OWA competitive with or slightly better than Trend's R3 best
- Preferred when compute budget, model size, or deployment constraints matter

### Alternative: Trend
- Slightly better average-case performance across all configs
- Statistically significant on M4-Yearly (p=0.000034) but negligible effect size (d=0.14)
- Not significant on Weather-96 (p=0.15)
- Preferred for maximum expected performance when parameter count is not a concern

---

## Evidence Summary

### Head-to-head (seed-matched Wilcoxon signed-rank on R1)

| Dataset | Pairs | Direction | p-value | Cohen's d | Significant? |
|---|---|---|---|---|---|
| M4-Yearly (OWA) | 662 | Trend better on average | 0.000034 | 0.14 | Yes (alpha=0.025) but d < 0.2 |
| Weather-96 (best_val_loss) | 502 | Trend slightly better | 0.15 | 0.06 | No |

### R3 Best Configs

| Dataset | Trend Best | TrendAE Best |
|---|---|---|
| M4-Yearly | DB20_bd15_lt_bcast_ttd3 (OWA 0.806, 5.10M params) | DB4_bd6_eq_fcast_ttd5 (OWA 0.800, 4.25M params) |
| Weather-96 | Symlet2_bd96_eq_fcast_ttd5 (bvl 42.75, 6.17M params) | DB3_bd96_eq_fcast_ttd3 (bvl 42.77, 5.22M params) |

### Key Observations
- TrendAE wins at the top of the leaderboard on M4 despite slightly worse average performance
- The backbone choice is secondary to hyperparameter selection (wavelet family, basis_dim, ttd)
- High R3 finalist overlap between backbones: the same configs tend to survive successive halving regardless of backbone

---

## Study Design Implication

When running successive halving searches:
1. **Fixed budget:** Pick one backbone (TrendAE recommended) and spend budget on HP search
2. **Large budget:** Run both backbones and compare R3 finalists
3. **Deployment-constrained:** Use TrendAE for ~16% parameter savings at negligible accuracy cost

---

## Source / Reproducibility

| Evidence | Report | Data |
|---|---|---|
| Study 3 cross-backbone (662+502 paired obs) | `experiments/analysis/notebooks/wavelet_study_3_successive_insights.ipynb` | See data sources in notebook |
| M4 Trend | - | `experiments/results/m4/wavelet_study_3_successive_results.csv` |
| M4 TrendAE | - | `experiments/results/m4/wavelet_study_3_successive_trendae_results.csv` |
| Weather Trend | - | `experiments/results/weather/wavelet_study_3_successive_results.csv` |
| Weather TrendAE | - | `experiments/results/weather/wavelet_study_3_successive_trendae_results.csv` |
| V3AE backbone hierarchy (332 configs, 995 runs) | `experiments/analysis/analysis_reports/wavelet_v3ae_study_analysis.md` | `experiments/results/m4/wavelet_v3ae_study_results.csv` |
| V3AELG cross-dataset (4 datasets) | `experiments/analysis/analysis_reports/wavelet_v3aelg_study_analysis.md` | `experiments/results/m4/wavelet_v3aelg_trendaelg_study_results.csv` + 3 others |
| Gate Function Study (AE vs AELG, M4+Weather, 303 runs) | `experiments/analysis/analysis_reports/gate_fn_study_analysis.md` | `experiments/results/m4/gate_fn_study_results.csv`, `experiments/results/weather/gate_fn_study_results.csv` |
