# Wavelet Basis Dimension Selection

## Core Guideline

When configuring `basis_dim` for WaveletV3 blocks, use **`eq_fcast`** or **`lt_bcast`** as the default labels. Avoid `lt_fcast` on short forecast horizons.

This is a **soft guideline** based on consistent trends across four independent studies (including cross-backbone validation with both Trend and TrendAE), not a statistically significant hard rule. Its primary value is narrowing search spaces from 4 labels to 2, saving compute in successive halving studies.

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

### Safe defaults (use these)
- **`eq_fcast`** -- Basis dim matches forecast length. Consistently top-2 across all studies. Natural match: exactly enough basis functions to represent the output.
- **`lt_bcast`** -- Half the backcast length. Consistently top-2. Provides moderate regularization.

### Acceptable (include if search budget allows)
- **`eq_bcast`** -- Full backcast length. Slightly over-parameterized on M4-Yearly (SMAPE 13.690 vs 13.615 for eq_fcast/lt_bcast in AELG sweep), but acceptable. On Weather-96, it performs comparably to eq_fcast.

### Avoid on short horizons
- **`lt_fcast`** -- Too few basis functions for short forecasts. 0% Round 3 survival on M4-Yearly in the AELG sweep (basis_dim=4 for forecast_length=6). On Tourism-Yearly (basis_dim=2), only 2 configs survived. On Weather-96 (basis_dim=94), it performs well -- the problem is specific to short horizons where the halving produces too few coefficients.

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

No study found statistically significant pairwise differences between the top three labels (eq_fcast, lt_bcast, eq_bcast):

| Study | Test | p-value | Result |
|---|---|---|---|
| Study 2 (72 runs, M4, Trend+WaveletV3) | Kruskal-Wallis on SMAPE by basis_dim | p=0.29 | Not significant |
| Study 3 (112 configs, M4, Trend+WaveletV3) | Round-by-round survival | All 4 labels in R3 | Mixed |
| Study 3 cross-backbone (M4, Trend vs TrendAE) | Rank correlation of bd_label medians | rho=0.60 | eq_fcast and lt_bcast top-2 for both backbones |
| Study 3 cross-backbone (Weather, Trend vs TrendAE) | Rank correlation of bd_label medians | rho=-0.50 | eq_fcast top-1 for Trend; eq_bcast top-1 for TrendAE |
| AELG Sweep (144 configs, M4+Tourism, TrendAELG+WaveletV3AELG) | Mann-Whitney on top labels | p>0.05 | Not significant |

The recommendation is based on **consistent directional trends** (eq_fcast and lt_bcast always rank top-2 across studies) and **one clear negative signal** (lt_fcast fails on short horizons), not on statistically proven superiority.

### Cross-backbone note (Study 3)

The bd_label ranking is broadly consistent across Trend and TrendAE backbones:
- **M4:** Both backbones rank eq_fcast and lt_bcast in the top-2 (rho=0.60)
- **Weather:** eq_fcast remains top-1 for Trend; TrendAE slightly prefers eq_bcast but differences are negligible (< 0.1 best_val_loss)
- **Conclusion:** The backbone choice does not change the bd_label recommendation

---

## Source / Reproducibility

| Evidence | Report | Data |
|---|---|---|
| Study 2 (72 runs, Trend+WaveletV3, M4) | `experiments/analysis/analysis_reports/wavelet_study_2_analysis.md` | `experiments/results/m4/wavelet_study_2_basis_dim_results.csv` |
| Study 3 (112 configs, Trend+WaveletV3, M4+Weather) | `experiments/analysis/analysis_reports/wavelet_study_3_analysis.md` | `experiments/results/m4/wavelet_study_3_successive_results.csv` |
| Study 3 cross-backbone (Trend vs TrendAE, M4+Weather) | `experiments/analysis/notebooks/wavelet_study_3_successive_insights.ipynb` (Section 10C) | `experiments/results/m4/wavelet_study_3_successive_trendae_results.csv`, `experiments/results/weather/wavelet_study_3_successive_trendae_results.csv` |
| AELG Sweep (144 configs, TrendAELG+WaveletV3AELG, M4+Tourism) | `experiments/analysis/analysis_reports/v3aelg_stackheight_sweep_analysis.md` (Section 5) | `experiments/results/m4/v3aelg_stackheight_sweep_results.csv`, `experiments/results/tourism/v3aelg_stackheight_sweep_results.csv` |
| Label formula | `experiments/run_wavelet_v3aelg_study.py:378` (`compute_basis_dim`) | -- |
