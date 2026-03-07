# Trend Thetas Dim Selection for Wavelet Stacks

## Core Rule

The optimal `trend_thetas_dim` depends on the **forecast horizon**:

- **Short horizons (H <= ~10):** Use `trend_thetas_dim=3` (quadratic polynomial)
- **Long horizons (H >= ~50):** Use `trend_thetas_dim=5` (quartic polynomial)
- **Medium horizons (10 < H < 50):** Not yet tested; default to ttd=3

This is a **horizon-dependent rule** that supersedes the prior recommendation of "always ttd=3".

---

## Recommendations

### Short horizons (H <= 10): `trend_thetas_dim=3`
- Statistically significant on Tourism-Yearly (H=4, p=0.008)
- Directionally better on M4-Yearly (H=6, p=0.16 at R3 convergence)
- Lower polynomial degree provides better regularization when there are few forecast steps

### Long horizons (H >= 50): `trend_thetas_dim=5`
- Statistically significant on Weather-96 (H=96, p=0.042)
- Higher-degree polynomial captures more complex trend shapes over longer horizons
- The quartic term adds meaningful capacity that the wavelet stack cannot compensate for

### Medium horizons (10 < H < 50): `trend_thetas_dim=3` (tentative default)
- No direct evidence in this range
- The crossover point has not been identified -- it likely lies somewhere in this range
- Future experiments on M4-Quarterly (H=8), M4-Monthly (H=18), or Traffic-192 are needed

---

## Evidence Summary

### V3AELG Study (R3 converged data, 50 epochs) -- Mann-Whitney U tests

| Dataset | Horizon | ttd=3 mean | ttd=5 mean | Winner | p-value | Significant? |
|---|---|---|---|---|---|---|
| M4-Yearly (SMAPE) | H=6 | 13.476 | 13.484 | ttd=3 | 0.16 | No (at convergence) |
| Tourism-Yearly (SMAPE) | H=4 | 20.98 | 21.12 | ttd=3 | 0.008 | Yes |
| Weather-96 (MSE) | H=96 | 2088.5 | 2075.1 | ttd=5 | 0.042 | Yes |
| Traffic-96 | H=96 | -- | -- | -- | -- | 86% divergence, not testable |

### Prior Study 3 (R1 data, 10 epochs) -- Mann-Whitney U tests

| Dataset | Backbone | ttd=3 median | ttd=5 median | p-value | Best |
|---|---|---|---|---|---|
| M4-Yearly (OWA) | Trend | 0.9427 | 0.9637 | 0.0012 | ttd=3 |
| M4-Yearly (OWA) | TrendAE | 0.9738 | 0.9803 | 0.0068 | ttd=3 |
| Weather-96 (bvl) | Trend | 43.389 | 43.476 | 0.0012 | ttd=3 |
| Weather-96 (bvl) | TrendAE | 43.384 | 43.554 | 0.0000 | ttd=3 |

### Reconciling the two studies

The prior Study 3 found ttd=3 better on Weather-96 at R1 (10 epochs). The V3AELG study found ttd=5 better at R3 (50 epochs). This is a **convergence speed artifact**: ttd=3 converges faster (fewer parameters to optimize), creating a misleading early advantage. With sufficient training, ttd=5 catches up and surpasses ttd=3 on long horizons.

On M4-Yearly (short horizon), the R1 advantage of ttd=3 narrows to non-significance at R3 (p=0.16), but ttd=3 remains directionally better. The convergence speed advantage happens to align with the true final ranking on short horizons but not on long horizons.

**Lesson:** Always evaluate trend_thetas_dim with converged (R3) data, not early-round data.

---

## Rationale

The trend thetas dim controls the polynomial degree of the trend basis expansion:
- `ttd=3`: degree-2 polynomial (constant + linear + quadratic)
- `ttd=5`: degree-4 polynomial (adds cubic + quartic terms)

For short forecast horizons (H <= ~10), a quadratic trend is sufficient to capture the trend shape in a few steps. Higher-degree polynomials risk overfitting the trend component, absorbing signal that should be modeled by the wavelet stack.

For long forecast horizons (H >= ~50), the additional polynomial terms capture meaningful curvature in the trend that cannot be represented by a quadratic. The wavelet stack handles high-frequency residuals, but the trend component needs enough capacity for low-frequency shape.

---

## Source / Reproducibility

| Evidence | Report | Data |
|---|---|---|
| V3AELG Study (R3 converged, 3 datasets) | `experiments/analysis/analysis_reports/wavelet_v3aelg_study_analysis.md` | `experiments/results/m4/wavelet_v3aelg_trendaelg_study_results.csv`, `experiments/results/tourism/wavelet_v3aelg_trendaelg_study_results.csv`, `experiments/results/weather/wavelet_v3aelg_trendaelg_study_results.csv` |
| V3AELG Notebook (Section 5: TTD Reversal) | `experiments/analysis/notebooks/wavelet_v3aelg_trendaelg_study_analysis.ipynb` | See notebook data sources |
| Study 3 cross-backbone (R1 data, 4 tests) | `experiments/analysis/notebooks/wavelet_study_3_successive_insights.ipynb` (Section 10B) | See notebook data sources |
| Study 3 M4 Trend | `experiments/analysis/analysis_reports/wavelet_study_3_analysis.md` | `experiments/results/m4/wavelet_study_3_successive_results.csv` |
| Study 3 M4 TrendAE | - | `experiments/results/m4/wavelet_study_3_successive_trendae_results.csv` |
| Study 3 Weather Trend | - | `experiments/results/weather/wavelet_study_3_successive_results.csv` |
| Study 3 Weather TrendAE | - | `experiments/results/weather/wavelet_study_3_successive_trendae_results.csv` |
