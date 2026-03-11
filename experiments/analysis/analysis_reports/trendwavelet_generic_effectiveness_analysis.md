# TrendWaveletGeneric Effectiveness Study -- M4-Yearly

**Date:** 2026-03-11
**Dataset:** M4-Yearly (H=6, L=30)
**Notebook:** `experiments/analysis/notebooks/trendwavelet_generic_effectiveness.ipynb`
**Results CSV:** `experiments/results/m4/trendwavelet_generic_effectiveness_results.csv`
**Config:** `experiments/configs/trendwavelet_generic_effectiveness.yaml`

## Executive Summary

TrendWaveletGeneric blocks add a third learned low-rank generic branch (rank `generic_dim`) to the existing polynomial trend + DWT wavelet decomposition in TrendWavelet blocks. This study tests whether the additional representational capacity improves forecasting quality on M4-Yearly.

**The answer is no.** The generic branch provides no statistically significant improvement on any backbone (all MWU p > 0.16). The study's most useful finding is a dramatic parameter efficiency gain: TWG-AE and TWG-AELG variants use 445K parameters vs 1.5M for their TW-AE/AELG counterparts (3.4x reduction) while matching quality. M4-Yearly SOTA (SMAPE=13.410) is not beaten.

## Experimental Design

- **7 configs:** 4 TrendWaveletGeneric variants (RootBlock, AE, AELG, VAE) + 3 TrendWavelet baselines (RootBlock, AE, AELG)
- **2 passes:** baseline (active_g=false), activeG_fcast (active_g=forecast)
- **5 seeds** per config-pass: 42, 43, 44, 45, 46
- **70 total runs**, zero divergences
- **Shared settings:** coif2 wavelet, trend_thetas_dim=3, basis_dim=4, generic_dim=5, latent_dim=16, 10 stacks, max 100 epochs, patience 10

## Full Ranking (Config x Pass)

| Rank | Config | Pass | SMAPE | Std | OWA | Params | Delta | Delta% |
|------|--------|------|-------|-----|-----|--------|-------|--------|
| 1 | TWGAELG | baseline | 13.509 | 0.028 | 0.801 | 445K | -- | -- |
| 2 | TWAELG | activeG_fcast | 13.519 | 0.046 | 0.803 | 1.52M | +0.010 | +0.08% |
| 3 | TWGAELG | activeG_fcast | 13.551 | 0.085 | 0.804 | 445K | +0.043 | +0.32% |
| 4 | TWAE | activeG_fcast | 13.561 | 0.038 | 0.808 | 1.52M | +0.053 | +0.39% |
| 5 | TWGAE | baseline | 13.567 | 0.041 | 0.806 | 445K | +0.058 | +0.43% |
| 6 | TWAELG | baseline | 13.596 | 0.172 | 0.808 | 1.52M | +0.088 | +0.65% |
| 7 | TW | baseline | 13.627 | 0.110 | 0.812 | 2.07M | +0.119 | +0.88% |
| 8 | TWGAE | activeG_fcast | 13.634 | 0.125 | 0.810 | 445K | +0.126 | +0.93% |
| 9 | TW | activeG_fcast | 13.638 | 0.277 | 0.813 | 2.07M | +0.129 | +0.96% |
| 10 | TWG | activeG_fcast | 13.654 | 0.064 | 0.812 | 2.09M | +0.145 | +1.08% |
| 11 | TWAE | baseline | 13.672 | 0.216 | 0.814 | 1.52M | +0.163 | +1.21% |
| 12 | TWG | baseline | 13.760 | 0.295 | 0.821 | 2.09M | +0.251 | +1.86% |
| 13 | TWGVAE | baseline | 16.067 | 1.602 | 0.965 | 600K | +2.558 | +18.9% |
| 14 | TWGVAE | activeG_fcast | 16.205 | 0.616 | 0.986 | 600K | +2.696 | +20.0% |

## Paired Comparisons: TWG vs TW (Matched by Seed)

### Baseline Pass

| Backbone | TWG SMAPE | TW SMAPE | Diff | 95% CI | MWU p | Wilcoxon p | Cohen's d | TWG wins |
|----------|-----------|----------|------|--------|-------|------------|-----------|----------|
| RootBlock | 13.760 (0.264) | 13.627 (0.098) | +0.133 | [-0.022, +0.343] | 0.421 | 0.312 | +0.67 | 1/5 |
| AERootBlock | 13.567 (0.037) | 13.672 (0.194) | -0.105 | [-0.244, +0.028] | 1.000 | 0.438 | -0.75 | 3/5 |
| AERootBlockLG | 13.509 (0.025) | 13.596 (0.153) | -0.088 | [-0.221, +0.023] | 0.548 | 0.438 | -0.80 | 3/5 |

### activeG_fcast Pass

| Backbone | TWG SMAPE | TW SMAPE | Diff | 95% CI | MWU p | Wilcoxon p | Cohen's d | TWG wins |
|----------|-----------|----------|------|--------|-------|------------|-----------|----------|
| RootBlock | 13.654 (0.057) | 13.638 (0.247) | +0.016 | [-0.209, +0.182] | 0.310 | 0.625 | +0.09 | 1/5 |
| AERootBlock | 13.634 (0.112) | 13.561 (0.034) | +0.073 | [-0.015, +0.160] | 0.690 | 0.312 | +0.89 | 2/5 |
| AERootBlockLG | 13.551 (0.076) | 13.519 (0.041) | +0.032 | [-0.038, +0.107] | 0.841 | 0.438 | +0.53 | 1/5 |

**Verdict:** No paired comparison reaches statistical significance. All 95% CIs straddle zero. The generic branch is neither consistently helpful nor harmful on M4-Yearly.

Interesting pattern: On AE/AELG backbones, TWG shows a small advantage in the baseline pass (where active_g is off), but this advantage reverses with activeG_fcast. This suggests the learned generic branch and active_g may be partially redundant -- both provide a flexible learned component on top of the structured polynomial+wavelet basis.

## Parameter Efficiency

| Config | SMAPE (pooled) | Params | SMAPE/M-param |
|--------|---------------|--------|---------------|
| TWGAELG | 13.530 | 445K | 30.4 |
| TWGAE | 13.601 | 445K | 30.6 |
| TWGVAE | 16.136 | 600K | 26.9 |
| TWAELG | 13.558 | 1.52M | 8.9 |
| TWAE | 13.616 | 1.52M | 9.0 |
| TWG (RootBlock) | 13.707 | 2.09M | 6.6 |
| TW (RootBlock) | 13.633 | 2.07M | 6.6 |

The TWG-AE variants achieve the best parameter efficiency: 445K params for SMAPE ~13.5. This is 3.4x fewer parameters than TW-AE/AELG (1.52M) and 4.7x fewer than TW/TWG RootBlock (~2.1M), with equivalent or better quality.

## active_g Effect

| Config | Baseline | activeG_fcast | Diff | MWU p |
|--------|----------|---------------|------|-------|
| TWG | 13.760 | 13.654 | -0.106 | 1.000 |
| TWGAE | 13.567 | 13.634 | +0.068 | 0.690 |
| TWGAELG | 13.509 | 13.551 | +0.043 | 0.548 |
| TWGVAE | 16.067 | 16.205 | +0.138 | 0.548 |
| TW | 13.627 | 13.638 | +0.011 | 0.421 |
| TWAE | 13.672 | 13.561 | -0.110 | 1.000 |
| TWAELG | 13.596 | 13.519 | -0.077 | 0.548 |

No config shows a significant active_g effect. All MWU p > 0.4. active_g=forecast is a non-factor for all TrendWavelet family blocks.

## VAE Catastrophe (Continued)

TrendWaveletGenericVAE SMAPE=16.136 (pooled) vs best non-VAE 13.509. The +19% penalty and high variance (std=1.60 in baseline pass) continue the established pattern that AERootBlockVAE is never competitive for N-BEATS forecasting.

## Backbone Hierarchy

| Backbone | SMAPE (mean) | Std | Min | n |
|----------|-------------|-----|-----|---|
| AERootBlockLG | 13.544 | 0.098 | 13.442 | 20 |
| AERootBlock | 13.609 | 0.127 | 13.476 | 20 |
| RootBlock | 13.670 | 0.202 | 13.434 | 20 |

Confirmed hierarchy: AERootBlockLG > AERootBlock > RootBlock. MWU tests: LG vs RootBlock p=0.0136 (significant), LG vs AE p=0.1339, AE vs RootBlock p=0.3043.

## Comparison to M4-Yearly SOTA

| Config | SMAPE | OWA | Params | Source |
|--------|-------|-----|--------|--------|
| **Trend+WaveletV3 (Coif2_bd6_eq_fcast_td3)** | **13.410** | **0.794** | **1.4M** | **Wavelet Study 2** |
| TrendAELG+WaveletV3AELG (sym20_eq_fcast_td3_ld16) | 13.438 | 0.795 | 4.3M | V3AELG Study |
| TrendAE+HaarWavV3AE_Alt_ld12 | 13.496 | 0.800 | 965K | Trend-Seas-Wav Comparison |
| **TWGAELG baseline (this study)** | **13.509** | **0.801** | **445K** | **This study** |
| TWAELG activeG (this study) | 13.519 | 0.803 | 1.52M | This study |

The study winner (TWGAELG, 13.509) does not beat SOTA but achieves the lowest parameter count of any competitive config (445K, 3.1x fewer than SOTA).

## Recommendations

### Current Best Configuration (M4-Yearly)
**Unchanged:** Trend+WaveletV3, Coif2_bd6_eq_fcast_td3. SMAPE=13.410, OWA=0.794.

### What to Test Next

1. **generic_dim sweep on TWGAELG:** Test gd=2,3,5,8,10,15 to find the optimal rank for the learned generic branch. Current gd=5 was a default.

2. **Multi-dataset validation:** Run TWGAELG (best from this study) on Tourism-Yearly, Traffic-96 (L=5H), and Weather-96 to confirm parameter efficiency holds across datasets.

3. **Alternating Trend + WaveletV3Generic:** Separate the polynomial and wavelet+generic branches into alternating stacks, as the Trend+WaveletV3 alternating architecture beats unified TrendWavelet on M4. This is the most promising avenue for beating SOTA with the generic branch.

4. **basis_dim=6 (eq_fcast) for TWG:** The SOTA uses bd=6 (eq_fcast for H=6), but this study used bd=4 (lt_fcast). The generic branch may interact differently with basis_dim.

### Open Questions

- Why does TWG-AE have 3.4x fewer params than TW-AE? The architectural source of this reduction needs investigation -- it may reveal a more general principle about how the generic branch reshapes parameter allocation in AE backbones.
- Is the generic branch more useful on longer horizons where there is more residual signal beyond polynomial trend + wavelet bases?
- Could a larger generic_dim (> 5) unlock the branch's potential, or does the N-BEATS residual architecture already capture residuals through stacking?
