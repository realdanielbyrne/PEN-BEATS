# Trend Thetas Dim Selection for Wavelet Stacks

## Core Rule

Use **`trend_thetas_dim=3`** for the Trend/TrendAE head in Trend+Wavelet stacks. This is consistently and significantly better than `trend_thetas_dim=5` across all tested backbones and datasets.

---

## Recommendations

### Always use: `trend_thetas_dim=3`
- Consistently best across all 4 backbone/dataset combinations
- Statistically significant in all 4 tests (all p < 0.01)
- Lower polynomial degree provides better regularization for the trend component

### Avoid: `trend_thetas_dim=5`
- Higher polynomial degree adds capacity without improving forecasts
- May overfit the trend component, leaving less useful residual for the wavelet stack

---

## Evidence Summary

### Mann-Whitney U tests (R1 data, ttd=3 vs ttd=5)

| Dataset | Backbone | ttd=3 median | ttd=5 median | p-value | Best |
|---|---|---|---|---|---|
| M4-Yearly (OWA) | Trend | 0.9427 | 0.9637 | 0.0012 | ttd=3 |
| M4-Yearly (OWA) | TrendAE | 0.9738 | 0.9803 | 0.0068 | ttd=3 |
| Weather-96 (bvl) | Trend | 43.389 | 43.476 | 0.0012 | ttd=3 |
| Weather-96 (bvl) | TrendAE | 43.384 | 43.554 | 0.0000 | ttd=3 |

### Strength Assessment
- **Consistent direction:** ttd=3 wins in all 4 comparisons
- **Significant:** All p-values < 0.01 (well below Bonferroni-corrected threshold)
- **Backbone-independent:** Result holds for both Trend and TrendAE
- **Dataset-independent:** Result holds for both M4-Yearly and Weather-96
- **No contradicting evidence**

This meets all skill creation thresholds and is classified as a **strong recommendation**.

---

## Rationale

The trend thetas dim controls the polynomial degree of the trend basis expansion:
- `ttd=3`: degree-2 polynomial (constant + linear + quadratic)
- `ttd=5`: degree-4 polynomial (adds cubic + quartic terms)

For short-to-medium forecast horizons (6–96 steps), a quadratic trend is sufficient. Higher-degree polynomials risk fitting noise in the trend residual, reducing the signal available for the wavelet stack to model.

---

## Source / Reproducibility

| Evidence | Report | Data |
|---|---|---|
| Study 3 cross-backbone (4 tests) | `experiments/analysis/notebooks/wavelet_study_3_successive_insights.ipynb` (Section 10B) | See notebook data sources |
| Study 3 M4 Trend | `experiments/analysis/analysis_reports/wavelet_study_3_analysis.md` | `experiments/results/m4/wavelet_study_3_successive_results.csv` |
| Study 3 M4 TrendAE | - | `experiments/results/m4/wavelet_study_3_successive_trendae_results.csv` |
| Study 3 Weather Trend | - | `experiments/results/weather/wavelet_study_3_successive_results.csv` |
| Study 3 Weather TrendAE | - | `experiments/results/weather/wavelet_study_3_successive_trendae_results.csv` |
