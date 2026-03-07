# Wavelet Basis Dimension Selection

## Core Guideline

When configuring `basis_dim` for WaveletV3 blocks, use **`eq_fcast`** as the default. Avoid `lt_fcast` on short forecast horizons.

This recommendation is supported by **five independent studies** across 4 backbones (Trend, TrendAE, TrendAELG, TrendAE+V3AE) and 4 datasets. The V3AELG cross-dataset study (R3 converged data) strengthened this from a soft guideline to a **confident recommendation**: `eq_fcast` is optimal or co-optimal on all viable datasets at convergence.

**Key insight from V3AELG study:** R1 (10-epoch) data showed apparent advantages for larger basis_dim labels (eq_bcast). At R3 convergence (50 epochs), these advantages vanished -- `eq_fcast` caught up and matched or beat all other labels. The R1 advantage of larger labels is a convergence speed artifact, not a quality difference.

---

## Label Definitions

The `compute_basis_dim()` function (in `run_wavelet_v3aelg_study.py`, `run_wavelet_v3ae_study.py`, `run_trendwaveletae_study.py`) maps labels to values:

| Label | Formula | Description |
|---|---|---|
| `eq_fcast` | `forecast_length` | Basis dim equals forecast length |
| `lt_fcast` | `max(forecast_length // 2, forecast_length - 2)` | Smaller than forecast length |
| `eq_bcast` | `backcast_length` | Basis dim equals backcast length |
| `lt_bcast` | `backcast_length // 2` | Half the backcast length |

### Resolved Values by Dataset

| Label | M4-Yearly (f=6, b=30) | Tourism-Yearly (f=4, b=8) | Traffic-96 (f=96, b=384) | Weather-96 (f=96, b=384) |
|---|---|---|---|---|
| `eq_fcast` | 6 | **4** | 96 | 96 |
| `lt_fcast` | 4 | 2 | 94 | 94 |
| `eq_bcast` | 30 | 8 | 384 | 384 |
| `lt_bcast` | 15 | **4** | 192 | 192 |

**Note:** When `backcast_length = 2 * forecast_length` (e.g., Tourism-Yearly), `eq_fcast` and `lt_bcast` collapse to the same value. This is confirmed by identical SMAPE outputs in the AELG sweep.

---

## Tiered Recommendations

### Best default (use this)
- **`eq_fcast`** -- Basis dim matches forecast length. Top-1 or co-top-1 on all viable datasets at R3 convergence (M4-Yearly, Tourism-Yearly, Weather-96). Natural match: exactly enough basis functions to represent the output. The V3AELG study confirmed this across 14 wavelet families and 3 datasets.

### Acceptable alternatives (include if search budget allows)
- **`lt_bcast`** -- Half the backcast length. Consistently top-2. On Tourism-Yearly and Weather-96, it resolves to the same value as `eq_fcast` (degeneracy when `backcast_length = 2 * forecast_length`).
- **`eq_bcast`** -- Full backcast length. Converges faster (more basis functions) but does not improve final quality. At R3, equivalent to `eq_fcast` on M4-Yearly (spread < 0.014 SMAPE). Useful when training epochs are limited.

### Avoid on short horizons
- **`lt_fcast`** -- Too few basis functions for short forecasts. 0% Round 3 survival on M4-Yearly in the AELG sweep (basis_dim=4 for forecast_length=6). On Tourism-Yearly (basis_dim=2), only 2 configs survived. On Weather-96 (basis_dim=94), it performs acceptably -- the problem is specific to short horizons where the halving produces too few coefficients.

---

## Study Design Implication

When designing successive halving searches over wavelet configurations:

1. **Default search space:** Use `basis_labels: [eq_fcast, lt_bcast]` (2 labels instead of 4)
2. **Budget permits:** Add `eq_bcast` as a third option
3. **Never include `lt_fcast` for forecast_length <= 8** -- it wastes search budget on configs that will be eliminated early
4. **Long horizons (forecast >= 48):** All four labels are viable since the absolute differences are small

This halves the basis_dim search dimension, doubling the budget available for other hyperparameters (wavelet family, stack height, latent_dim).

---

## Statistical Caveat

No study found statistically significant pairwise differences between the top three labels (eq_fcast, lt_bcast, eq_bcast) at convergence. The recommendation is based on **consistent directional trends** across 5 studies and **one clear negative signal** (lt_fcast fails on short horizons).

| Study | Test | p-value | Result |
|---|---|---|---|
| Study 2 (72 runs, M4, Trend+WaveletV3) | Kruskal-Wallis on SMAPE by basis_dim | p=0.29 | Not significant |
| Study 3 (112 configs, M4, Trend+WaveletV3) | Round-by-round survival | All 4 labels in R3 | Mixed |
| Study 3 cross-backbone (M4, Trend vs TrendAE) | Rank correlation of bd_label medians | rho=0.60 | eq_fcast and lt_bcast top-2 for both backbones |
| Study 3 cross-backbone (Weather, Trend vs TrendAE) | Rank correlation of bd_label medians | rho=-0.50 | eq_fcast top-1 for Trend; eq_bcast top-1 for TrendAE |
| AELG Sweep (144 configs, M4+Tourism, TrendAELG+WaveletV3AELG) | Mann-Whitney on top labels | p>0.05 | Not significant |
| **V3AELG Study (R3, M4+Tourism+Weather)** | **R3 converged ranking** | -- | **eq_fcast top-1 or co-top-1 on all 3 datasets** |

### V3AELG convergence insight (2026-03-07)

The V3AELG study revealed why prior R1-based analyses showed ambiguous results: **larger basis_dim labels converge faster but do not produce better final forecasts**. At R1 (10 epochs), `eq_bcast` appeared competitive or better. At R3 (50 epochs), `eq_fcast` matched or surpassed all others, with the spread between top-3 labels narrowing to < 0.014 SMAPE on M4-Yearly.

This convergence speed artifact also explains the Weather cross-backbone disagreement (Study 3): at R1, TrendAE preferred `eq_bcast` because the AE bottleneck slowed convergence of smaller-basis configs. With sufficient training, this difference would likely vanish.

### Cross-backbone note (Study 3)

The bd_label ranking is broadly consistent across Trend, TrendAE, and TrendAELG backbones:
- **M4:** eq_fcast top-1 or top-2 for all three backbones
- **Weather:** eq_fcast top-1 for Trend and TrendAELG; TrendAE slightly prefers eq_bcast at R1 but likely converges to eq_fcast
- **Conclusion:** The backbone choice does not change the bd_label recommendation

### TrendWaveletAELG exception: bd_label sensitivity is HIGHER

In TrendWaveletAELG (unified trend+wavelet block), bd_label differences do NOT vanish at convergence:
- eq_fcast (13.571) vs lt_fcast (13.599): p=0.017 **
- eq_fcast (13.571) vs eq_bcast (13.943): p=0.004 ***
- Spread: 0.372 SMAPE at R3 (vs 0.014 in V3AELG alternating study)

The unified AE bottleneck makes basis dimension selection more consequential. **eq_fcast is even more critical for TrendWaveletAELG than for alternating stacks.** See TrendWaveletAELG Pure Study analysis.

---

## Source / Reproducibility

| Evidence | Report | Data |
|---|---|---|
| **V3AELG Study (112 configs, 14 families, R3 converged, M4+Tourism+Weather)** | `experiments/analysis/analysis_reports/wavelet_v3aelg_study_analysis.md` | `experiments/results/m4/wavelet_v3aelg_trendaelg_study_results.csv`, `experiments/results/tourism/wavelet_v3aelg_trendaelg_study_results.csv`, `experiments/results/weather/wavelet_v3aelg_trendaelg_study_results.csv` |
| V3AELG Notebook (Section 4: Basis Dim Effect) | `experiments/analysis/notebooks/wavelet_v3aelg_trendaelg_study_analysis.ipynb` | See notebook data sources |
| Study 2 (72 runs, Trend+WaveletV3, M4) | `experiments/analysis/analysis_reports/wavelet_study_2_analysis.md` | `experiments/results/m4/wavelet_study_2_basis_dim_results.csv` |
| Study 3 (112 configs, Trend+WaveletV3, M4+Weather) | `experiments/analysis/analysis_reports/wavelet_study_3_analysis.md` | `experiments/results/m4/wavelet_study_3_successive_results.csv` |
| Study 3 cross-backbone (Trend vs TrendAE, M4+Weather) | `experiments/analysis/notebooks/wavelet_study_3_successive_insights.ipynb` (Section 10C) | `experiments/results/m4/wavelet_study_3_successive_trendae_results.csv`, `experiments/results/weather/wavelet_study_3_successive_trendae_results.csv` |
| AELG Sweep (144 configs, TrendAELG+WaveletV3AELG, M4+Tourism) | `experiments/analysis/analysis_reports/v3aelg_stackheight_sweep_analysis.md` (Section 5) | `experiments/results/m4/v3aelg_stackheight_sweep_results.csv`, `experiments/results/tourism/v3aelg_stackheight_sweep_results.csv` |
| Label formula | `experiments/run_wavelet_v3aelg_study.py:378` (`compute_basis_dim`) | -- |
