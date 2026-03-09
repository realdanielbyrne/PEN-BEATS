# TrendWaveletAELG Pure Study Analysis

## Overview

Homogeneous stacks of `TrendWaveletAELG` blocks -- a single block type combining AERootBlockLG backbone (encoder-decoder with sigmoid-gated latent) with dual Vandermonde polynomial trend basis and orthonormal DWT wavelet basis expansion.

**Two study versions:**

- **V1:** 14 wavelet families, td={3,5}, ld=8, M4-Yearly. 112 configs, 3-round successive halving.
- **V2:** 6 wavelet families, td=3, ld=16, across M4-Yearly + Tourism-Yearly + Traffic-96 + Weather-96. 24 configs per dataset.

**Total data:** 1,102 rows across 5 CSV files.

## Key Findings

### 1. NEW Tourism-Yearly SOTA

TrendWaveletAELG sets a new Tourism-Yearly best:

- **coif3_eq_bcast_td3_ld16: SMAPE 20.681** vs prior best 20.930 (TrendAELG+WaveletV3AELG Coif1)
- Improvement: -0.249 SMAPE (-1.19%)
- The simpler unified block outperforms the more complex alternating architecture

### 2. Competitive on M4-Yearly (Doesn't Beat SOTA)

- v2 best: db3_eq_fcast_td3_ld16 (SMAPE 13.463, +0.40% above non-AE SOTA 13.410)
- v1 best: db10_eq_fcast_td3_ld8 (SMAPE 13.506, +0.71% above SOTA)
- All configs ~1.5M params

### 3. Wavelet Family is a Non-Factor

Kruskal-Wallis p=0.107 across 13 families on M4-Yearly v1 R3. All families cluster within 0.18 SMAPE (13.552-13.735). No pairwise comparison reaches significance.

This contrasts sharply with TrendAELG+WaveletV3AELG (alternating) where sym20 was universally best. The unified AE bottleneck homogenizes basis representations.

### 4. Basis Label Matters More Than in Alternating Stacks

eq_fcast (13.571) is significantly best:

- vs lt_fcast: p=0.017
- vs eq_bcast: p=0.004
- Spread: 0.372 SMAPE (does NOT vanish at convergence, unlike V3AELG study's 0.014)

### 5. ld=16 Better Than ld=8 (with Caveats)

v2 (ld=16) top-5 significantly better than v1 (ld=8) top-5 (p=0.042). However, DB4+eq_fcast catastrophically fails at ld=16 (SMAPE ~76) while working fine at ld=8. DB4 with other bd_labels at ld=16 is fine.

### 6. Traffic-96 Divergence (Protocol Failure, Not Architectural)

100% of runs produce SMAPE=200 with val_loss flatlined from epoch 1. **Root cause: insufficient backcast horizon.** This study used bl=192 (L=2H, `forecast_multiplier=2`), which is inadequate for Traffic-96. A subsequent study (AsymWavelet Diagnostic, 2026-03-08) using bl=480 (L=5H, `forecast_multiplier=5`) and 8 stacks achieved 80-100% convergence for TrendAELG+WaveletV3AELG. This is a protocol failure, not an architectural incompatibility.

### 7. Cross-Dataset Family Rankings Uncorrelated

Spearman rho=-0.100 (p=0.873) between M4 and Tourism family rankings. No universal family recommendation for TrendWaveletAELG. Coif3 has the best cross-dataset average rank (1.5).

## Best Configs

| Dataset | Config | SMAPE | OWA | Params |
|---------|--------|-------|-----|--------|
| M4-Yearly (v1) | db10_eq_fcast_td3_ld8 | 13.506 | 0.799 | 1,485,050 |
| M4-Yearly (v2) | db3_eq_fcast_td3_ld16 | 13.463 | 0.797 | ~1.5M |
| Tourism-Yearly | coif3_eq_bcast_td3_ld16 | **20.681** | N/A | ~1.5M |
| Traffic-96 | N/A for this study (L=2H insufficient; rerun with L≥5H) | 200.0 | N/A | ~4.8M |

## Architecture Selection

| Scenario | Recommendation |
|----------|---------------|
| Short horizon, simplicity preferred | TrendWaveletAELG (unified) |
| Short horizon, max M4 accuracy | Trend+WaveletV3 non-AE (13.410) |
| Tourism-Yearly | TrendWaveletAELG coif3_eq_bcast_td3_ld16 |
| Traffic | Use with L≥5H (`forecast_multiplier=5`) and ≤8-10 stacks |

## Detailed Notebook

See `experiments/analysis/notebooks/trendwaveletaelg_pure_study_insights.ipynb` for full analysis with visualizations and statistical tests.

## Data Sources

- `experiments/results/m4/trendwaveletaelg_pure_study_results.csv` (711 rows, v1)
- `experiments/results/m4/trendwaveletaelg_pure_v2_study_results.csv` (150 rows, v2)
- `experiments/results/tourism/trendwaveletaelg_pure_v2_study_results.csv` (150 rows, v2)
- `experiments/results/traffic/trendwaveletaelg_pure_v2_study_results.csv` (80 rows, v2)
- `experiments/results/weather/trendwaveletaelg_pure_v2_study_results.csv` (10 rows, v2)

---

## Correction Addendum (2026-03-09)

**The Traffic-96 "complete failure" conclusion is incorrect.** The 100% divergence observed in this study was caused by the evaluation protocol (bl=192, L=2H; 20 stacks), not by inherent block-dataset incompatibility.

A subsequent study (AsymWavelet Diagnostic, 2026-03-08) using L=5H (bl=480) and 8 stacks demonstrated 80-100% convergence for alternating TrendAELG+WaveletV3AELG on Traffic-96, achieving MSE ~0.0006. The recommendation "Traffic/Weather: Do NOT use trend+wavelet" should be revised to: **use trend+wavelet with adequate lookback (L≥5H) and moderate stack depth (≤8-10 stacks).**
