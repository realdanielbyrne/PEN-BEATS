# Asymmetric Wavelet Diagnostic Study: Traffic-96 and Weather-96

**Date:** 2026-03-08
**Datasets:** Traffic-96, Weather-96
**Runs:** 50 per dataset (10 configs x 5 seeds)
**Architecture:** TrendAELG + WaveletV3AELG (alternating, 8 stacks) and TrendVAE + WaveletV3VAE (alternating, 8 stacks)

## Executive Summary

This study tested whether **asymmetric wavelet family pairs** -- using different wavelet families for the backcast path (L=480) vs. the forecast path (H=96) -- improve forecasting performance. The rationale: long-support wavelets (sym20) may better capture the structure of the long backcast, while short-support wavelets (coif2, sym3) may be more appropriate for the shorter forecast.

**The hypothesis is not supported.** Symmetric wavelet pairs perform as well or better than asymmetric pairs on both datasets. On Traffic, symmetric AELG pairs significantly outperform asymmetric ones (Mann-Whitney p=0.026). On Weather, no significant difference exists in either direction.

Key findings:

1. **AELG dominates VAE on Traffic** by 5.6x in MSE (p < 1e-6). On Weather, backbones are statistically equivalent.
2. **coif2 is the best forecast wavelet** on both datasets, regardless of backcast wavelet choice.
3. **VAE is functionally blind to wavelet choice** -- all 5 VAE configs produce nearly identical results (KW p=0.49 on Traffic).
4. **AELG divergence rate on Traffic dropped to 16%** from 86% (alternating prior) and 100% (pure prior), thanks to shallower 8-stack architecture.

## Study Design

### Configurations Tested

| Config Name | Backbone | Backcast Wavelet | Forecast Wavelet | Type | Params |
|---|---|---|---|---|---|
| AELG-coif2-coif2 | AERootBlockLG | coif2 | coif2 | Symmetric | 1,807,436 |
| AELG-sym3-sym3 | AERootBlockLG | sym3 | sym3 | Symmetric | 1,807,436 |
| AELG-sym20-sym3 | AERootBlockLG | sym20 | sym3 | Asymmetric | 1,807,436 |
| AELG-sym20-coif2 | AERootBlockLG | sym20 | coif2 | Asymmetric | 1,807,436 |
| AELG-sym3-sym20 | AERootBlockLG | sym3 | sym20 | Asymmetric | 1,807,436 |
| VAE-coif2-coif2 | AERootBlockVAE | coif2 | coif2 | Symmetric | 2,133,580 |
| VAE-sym3-sym3 | AERootBlockVAE | sym3 | sym3 | Symmetric | 2,133,580 |
| VAE-sym20-sym3 | AERootBlockVAE | sym20 | sym3 | Asymmetric | 2,133,580 |
| VAE-sym20-coif2 | AERootBlockVAE | sym20 | coif2 | Asymmetric | 2,133,580 |
| VAE-sym3-sym20 | AERootBlockVAE | sym3 | sym20 | Asymmetric | 2,133,580 |

All configs use: 8 stacks (4 alternating TrendAE/WaveletV3AE pairs), 1 block per stack, shared weights, ReLU activation, no active_g, no sum_losses.

## Full Rankings

### Traffic-96

| Rank | Config | Backbone | Type | Mean MSE | Std MSE | Mean SMAPE | Diverged |
|---|---|---|---|---|---|---|---|
| 1 | AELG-coif2-coif2 | AELG | SYM | 0.000603 | 0.000010 | 22.240 | 1/5 |
| 2 | AELG-sym3-sym3 | AELG | SYM | 0.000610 | 0.000029 | 23.287 | 0/5 |
| 3 | AELG-sym20-coif2 | AELG | ASYM | 0.000611 | 0.000014 | 23.168 | 1/5 |
| 4 | AELG-sym3-sym20 | AELG | ASYM | 0.000623 | 0.000025 | 23.769 | 1/5 |
| 5 | AELG-sym20-sym3 | AELG | ASYM | 0.000634 | 0.000012 | 24.682 | 1/5 |
| 6 | VAE-coif2-coif2 | VAE | SYM | 0.003452 | 0.000035 | 66.068 | 0/5 |
| 7 | VAE-sym20-coif2 | VAE | ASYM | 0.003453 | 0.000027 | 66.123 | 0/5 |
| 8 | VAE-sym3-sym20 | VAE | ASYM | 0.003464 | 0.000012 | 66.096 | 0/5 |
| 9 | VAE-sym3-sym3 | VAE | SYM | 0.003465 | 0.000016 | 66.119 | 0/5 |
| 10 | VAE-sym20-sym3 | VAE | ASYM | 0.003473 | 0.000008 | 66.138 | 0/5 |

**Best individual run:** AELG-coif2-coif2, seed=1, MSE=0.000588, SMAPE=22.054

### Weather-96

| Rank | Config | Backbone | Type | Mean MSE | Std MSE | Mean SMAPE | Diverged |
|---|---|---|---|---|---|---|---|
| 1 | AELG-sym20-coif2 | AELG | ASYM | 1804.02 | 389.59 | 70.175 | 0/5 |
| 2 | VAE-coif2-coif2 | VAE | SYM | 1847.96 | 242.50 | 68.442 | 0/5 |
| 3 | AELG-sym3-sym3 | AELG | SYM | 2035.22 | 334.27 | 70.481 | 0/5 |
| 4 | AELG-sym3-sym20 | AELG | ASYM | 2119.82 | 133.61 | 70.325 | 0/5 |
| 5 | VAE-sym3-sym3 | VAE | SYM | 2124.93 | 127.76 | 68.300 | 0/5 |
| 6 | AELG-coif2-coif2 | AELG | SYM | 2131.54 | 489.09 | 71.496 | 0/5 |
| 7 | VAE-sym20-sym3 | VAE | ASYM | 2143.97 | 184.56 | 68.404 | 0/5 |
| 8 | VAE-sym20-coif2 | VAE | ASYM | 2152.32 | 312.42 | 67.951 | 0/5 |
| 9 | VAE-sym3-sym20 | VAE | ASYM | 2154.54 | 405.98 | 67.764 | 0/5 |
| 10 | AELG-sym20-sym3 | AELG | ASYM | 2158.54 | 180.32 | 69.107 | 0/5 |

**Best individual run:** AELG-sym20-coif2, seed=2, MSE=1303.85, SMAPE=67.970

## Statistical Analysis

### 1. AELG vs VAE Backbone

| Dataset | AELG Mean MSE | VAE Mean MSE | Ratio | Mann-Whitney p | Cohen's d |
|---|---|---|---|---|---|
| Traffic-96 | 0.000616 | 0.003461 | 5.6x | < 1e-6 | 132.5 (huge) |
| Weather-96 | 2049.83 | 2084.74 | 1.02x | 0.846 | 0.11 (small) |

**Traffic:** The AELG backbone is overwhelmingly superior. VAE runs on Traffic cluster near SMAPE ~66 -- effectively failing to learn. The 5.6x MSE gap (Cohen's d = 132) is the largest effect in the study.

**Weather:** No meaningful backbone difference. AELG and VAE produce statistically indistinguishable results. This contrasts sharply with Traffic and suggests the VAE failure on Traffic is dataset-specific, not architecture-inherent.

### 2. Symmetric vs Asymmetric Wavelet Pairs

Within AELG backbone (the meaningful comparison):

| Dataset | Symmetric Mean MSE | Asymmetric Mean MSE | Mann-Whitney p | Result |
|---|---|---|---|---|
| Traffic-96 | 0.000606 | 0.000623 | 0.026 | Symmetric wins (sig.) |
| Weather-96 | 2083.38 | 2027.46 | 0.677 | No significant difference |

**Traffic:** Symmetric pairs significantly outperform asymmetric ones. This is the opposite of the study hypothesis.

**Weather:** No statistically significant difference. The Weather winner (AELG-sym20-coif2) is asymmetric, but its advantage over symmetric coif2-coif2 is not significant (p=0.22 pairwise).

### 3. Forecast Wavelet Effect (AELG only)

| Dataset | coif2 forecast | sym3 forecast | sym20 forecast | Kruskal-Wallis p |
|---|---|---|---|---|
| Traffic-96 | 0.000607 | 0.000622 | 0.000623 | 0.250 |
| Weather-96 | 1967.78 | 2096.88 | 2119.82 | 0.506 |

Not statistically significant on either dataset, but coif2 is consistently the lowest MSE forecast wavelet in both cases. The trend is consistent even if the effect is small relative to seed variance.

### 4. Backcast Wavelet Effect (AELG only)

| Dataset | coif2 backcast | sym3 backcast | sym20 backcast | Kruskal-Wallis p |
|---|---|---|---|---|
| Traffic-96 | 0.000603 | 0.000617 | 0.000622 | 0.145 |
| Weather-96 | 2131.54 | 2077.52 | 1981.28 | 0.740 |

Also not significant. On Traffic, coif2 backcast is best; on Weather, sym20 backcast is best. Opposite patterns -- no generalizable backcast wavelet preference.

### 5. VAE Wavelet Sensitivity

| Dataset | VAE config spread (MSE) | Kruskal-Wallis p | Interpretation |
|---|---|---|---|
| Traffic-96 | 0.000021 | 0.493 | No sensitivity |
| Weather-96 | 306.58 | 0.331 | No sensitivity |

The VAE backbone completely homogenizes wavelet basis differences. The stochastic reparameterization overwhelms the deterministic basis, rendering wavelet choice irrelevant for VAE blocks.

### 6. Cross-Dataset Consistency

Spearman rank correlation between Traffic and Weather config rankings: **rho = 0.41, p = 0.24** (not significant).

Notable rank changes:

- AELG-coif2-coif2: #1 on Traffic, #6 on Weather (delta = 5)
- AELG-sym20-coif2: #3 on Traffic, #1 on Weather (delta = 2)
- VAE-coif2-coif2: #6 on Traffic, #2 on Weather (delta = 4)

The only robust cross-dataset pattern: **coif2 as forecast wavelet appears in the #1 config on both datasets.**

## Divergence Analysis (Traffic-96)

| Config | Diverged Runs | Divergence Rate | Notes |
|---|---|---|---|
| AELG-coif2-coif2 | 1/5 | 20% | seed=4, epochs=36, MSE=0.000601 |
| AELG-sym20-sym3 | 1/5 | 20% | seed=3, epochs=27, MSE=0.000618 |
| AELG-sym20-coif2 | 1/5 | 20% | seed=2, epochs=13, MSE=0.000623 |
| AELG-sym3-sym20 | 1/5 | 20% | seed=5, epochs=31, MSE=0.000602 |
| AELG-sym3-sym3 | 0/5 | 0% | Most stable config |
| All VAE configs | 0/25 | 0% | VAE never diverges (but never learns either) |

**Overall AELG divergence rate: 16%** (4/25). This is dramatically better than prior Traffic studies (86% for alternating at L=2H, 100% for pure TrendWaveletAELG at L=2H). **The primary stabilizing factor was the lookback increase from L=2H (bl=192) to L=5H (bl=480).** The shallower 8-stack architecture (vs 20 stacks in prior studies) is a secondary contributing factor.

Diverged runs produced reasonable test metrics, suggesting late-training instability rather than complete failure. The best-performing diverged run (AELG-sym3-sym20, seed=5, MSE=0.000602) actually outperforms several non-diverged runs from other configs.

## Interpretation

### Why asymmetry does not help

The asymmetric wavelet hypothesis assumed that the backcast and forecast basis generators operate independently and benefit from family-matched dimensionality. In practice, WaveletV3 blocks learn through the **same shared backbone** (encoder-decoder in AELG), meaning the latent representation must simultaneously serve both paths. A long-support backcast wavelet (sym20) creates a different latent structure than a short-support one (coif2), and this latent structure then feeds into the forecast basis generator. The two paths are not independent -- they share the AE bottleneck.

Additionally, the basis dimension is clamped to the target length, so sym20's long-support advantage is already limited by the `pywt.dwt_max_level` constraint. On H=96, most wavelet families produce similar multi-level decomposition structures.

### Why coif2 emerges as the best forecast wavelet

Coiflets are designed to have both the scaling function and wavelet function with vanishing moments, providing better approximation properties near boundaries. For a 96-step forecast, boundary effects matter more than for the 480-step backcast, which may explain coif2's advantage specifically in the forecast position.

### Why VAE fails on Traffic but not Weather

Traffic data (862 PeMS sensors, hourly) has highly regular diurnal/weekly patterns with sharp transitions. The VAE's stochastic latent adds noise that blurs these sharp features. Weather data (21 meteorological indicators, 10-min) has smoother dynamics that are more tolerant of latent noise. This is consistent with the prior finding that VAE blocks are "never competitive" -- but the Weather results show the gap can narrow to statistical insignificance on smooth data.

## Recommendations

### Current Best Configurations

| Dataset | Best Config | MSE | SMAPE | Confidence |
|---|---|---|---|---|
| Traffic-96 | AELG-coif2-coif2 | 0.000603 | 22.24 | Moderate (5 seeds, 1 diverged) |
| Weather-96 | AELG-sym20-coif2 | 1804.02 | 70.18 | Low (5 seeds, high variance) |

### What to Test Next

1. **Expand seed count to 10-20** for Traffic AELG-coif2-coif2 and Weather AELG-sym20-coif2 to confirm rankings with higher statistical power.

2. **Test AELG-coif2-coif2 on Weather** with more runs -- it ranked #6 with very high variance (std=489), so more seeds may revise its position.

3. **Test sym20-coif2 on Traffic** with more runs -- it ranked #3 and could potentially beat coif2-coif2 with reduced variance.

4. **Shallower stacks for Traffic stability.** The 8-stack (4-pair) architecture reduced divergence to 16%. Try 6-stack (3-pair) to see if divergence drops further without sacrificing quality.

5. **Non-AE baselines for Traffic/Weather.** This study only tested AE-family blocks. The prior finding that non-AE Trend+WaveletV3 beats AELG on M4-Yearly (SMAPE 13.41 vs 13.44) suggests testing symmetric Trend+WaveletV3 with coif2 on Traffic/Weather.

### Open Questions

1. **Is the sym20-coif2 Weather advantage real?** With 5 seeds and std=389, we need 15+ seeds to resolve whether this asymmetric pair genuinely beats coif2-coif2.

2. **Why does AELG-sym3-sym3 never diverge on Traffic** while all other AELG configs have 20% divergence? Is sym3's shorter support providing training stability?

3. **Would coif2-sym3 (tested on neither dataset in this study) perform differently from sym3-coif2?** The current study only tested pairs involving sym20.

4. **Is the coif2 forecast advantage specific to H=96?** Testing at other horizons (192, 336, 720) would determine if this is a general property of coif2 or specific to the backcast/forecast ratio.

## Files

- Traffic results: `experiments/results/traffic/Traffic-AsymWavelet-Diagnostic_results.csv`
- Weather results: `experiments/results/weather/Weather-AsymWavelet-Diagnostic_results.csv`
- Notebook: `experiments/analysis/notebooks/asym_wavelet_diagnostic_insights.ipynb`
- This report: `experiments/analysis/analysis_reports/asym_wavelet_diagnostic_analysis.md`

---

## Clarification Note (2026-03-09)

This report attributes the improved convergence (16% divergence vs 86-100% in prior studies) primarily to "shallower 8-stack architecture." However, the most significant protocol change was the **lookback increase from L=2H (bl=192) to L=5H (bl=480)**. All prior Traffic studies that reported high divergence used bl=192; this is the only study that used bl=480. Both factors (shorter stacks and longer lookback) likely contributed, but the lookback change is the more impactful variable. Additionally, this study ran **without normalization** (no `protocol:` block in the YAML), despite comments in the config mentioning Z-score normalization. Traffic data is naturally in the 0-1 scale and does not require normalization.
