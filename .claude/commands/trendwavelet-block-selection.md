# TrendWavelet Block Selection: Unified vs Alternating

## Core Rule

When choosing between TrendWaveletAELG (unified block) and TrendAELG + WaveletV3AELG (alternating stacks):

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| **Tourism-Yearly** | **TrendWaveletAELG** | New SOTA (20.681 vs 20.930) |
| Short horizon (H<=10), simplicity preferred | **TrendWaveletAELG** | Simpler architecture, competitive results |
| M4-Yearly, maximum accuracy | Trend+WaveletV3 (non-AE) | SOTA 13.410, unified is +0.40% |
| Traffic-96 or Weather-96 | **Neither** -- use Generic/GenericAELG | Both trend+wavelet architectures fail |

---

## TrendWaveletAELG Configuration

When using TrendWaveletAELG:

### Hyperparameter defaults
- `latent_dim=16` (significantly better than ld=8, p=0.042)
- `trend_thetas_dim=3` for H<=10 (confirmed, p=0.255 directional)
- `basis_dim_label=eq_fcast` (significantly best, p=0.017)
- `n_stacks=10`, `n_blocks_per_stack=1`, `share_weights=True`

### Wavelet family: does NOT matter
Wavelet family is a non-factor for TrendWaveletAELG (Kruskal-Wallis p=0.107 across 14 families). The AE bottleneck homogenizes basis representations. Use any family; coif3 has best cross-dataset average rank.

**This is the opposite of alternating stacks** where sym20 is the universal best family.

### Known instability
**Avoid DB4 + eq_fcast + ld=16.** This specific combination catastrophically fails (SMAPE ~76). DB4 with other bd_labels or at ld=8 is fine.

---

## Key Differences from Alternating Stacks

| Property | TrendWaveletAELG (unified) | TrendAELG + WaveletV3AELG (alternating) |
|----------|---------------------------|----------------------------------------|
| Wavelet family effect | Non-significant (p=0.107) | Significant (sym20 best) |
| Basis label sensitivity | High (0.37 SMAPE spread) | Low (0.014 spread at convergence) |
| Cross-dataset family consistency | Uncorrelated (rho=-0.1) | Correlated (sym20 universal) |
| Traffic-96 | 100% failure | 86% divergence |
| Tourism-Yearly | **SOTA** (20.681) | 20.930 |
| M4-Yearly | 13.463 (+0.40%) | 13.438 (+0.21%) |
| Architecture simplicity | 1 block type | 2 block types alternating |

---

## Evidence

| Study | Report | Data |
|-------|--------|------|
| TrendWaveletAELG Pure v1 (M4, 112 configs) | `experiments/analysis/analysis_reports/trendwaveletaelg_pure_study_analysis.md` | `experiments/results/m4/trendwaveletaelg_pure_study_results.csv` |
| TrendWaveletAELG Pure v2 (4 datasets) | Same report | `experiments/results/*/trendwaveletaelg_pure_v2_study_results.csv` |
| Notebook | `experiments/analysis/notebooks/trendwaveletaelg_pure_study_insights.ipynb` | See notebook |
