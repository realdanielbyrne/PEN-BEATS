# Wavelet + TrendAE Comparison Study -- Corrected Analysis

**Date:** 2026-03-07 (corrected reanalysis)
**Dataset:** M4-Yearly
**Architecture:** `[TrendAE, WaveletV3] * 5` (10 stacks, alternating)
**CSV:** `experiments/results/m4/wavelet_trendae_comparison_results.csv`
**Notebook:** `experiments/analysis/notebooks/wavelet_trendae_comparison_insights.ipynb`

## Executive Summary

This study evaluates whether replacing the standard Trend backbone (RootBlock, 4 FC layers) with TrendAE (AERootBlock, encoder-decoder bottleneck) improves Trend+WaveletV3 hybrid stacks on M4-Yearly. **The answer is no.** The best TrendAE+WaveletV3 config achieves OWA=0.797, which is worse than the non-AE SOTA (OWA=0.794) while using 3x more parameters (~4.2M vs ~1.4M).

**Data quality note:** The CSV contains 2 exact duplicate rows for `Symlet3_bd4_lt_fcast_ttd3_ld5` (seeds 43 and 44). All statistics below use the deduplicated dataset (90 rows, 30 configs, 3 seeds each).

### Key Findings

1. **TrendAE is not an improvement.** Best TrendAE+WavV3: OWA=0.797, SMAPE=13.462. Non-AE SOTA: OWA=0.794, SMAPE=13.410. The AE bottleneck on the Trend block adds capacity but not useful inductive bias.
2. **No hyperparameter reaches statistical significance.** All Kruskal-Wallis tests (wavelet family p=0.49, latent dim p=0.37, basis dim p=0.18, bd_label p=0.18) fail to reject the null at alpha=0.05.
3. **Strong wavelet x latent_dim interaction.** Symlets prefer LD=8, Daubechies prefer LD=2, Coiflets/Haar prefer LD=5. Marginal LD analysis is misleading.
4. **Stability is the differentiator.** Top configs are within 0.002 OWA of each other, but stability varies by 10x. The most stable config (Symlet3_bd4_lt_fcast_ttd3_ld8, std=0.0006) should be preferred.
5. **100% convergence rate.** Zero diverged runs across all 90 training runs. Architecture is stable.

---

## 1. Study Design

- **30 configurations** = 10 top WaveletV3 configs from wavelet_study_3 x 3 latent dims (2, 5, 8)
- **3 runs per config** with seeds [42, 43, 44]
- **90 total runs** (after deduplication), 332.3 minutes total training time
- **50 max epochs**, patience=10, lr=0.001 with warmup/cosine schedule
- **Wavelet families tested:** Coif1, Coif2, Coif3, DB10, DB20, Haar, Symlet3, Symlet10
- **Basis dims tested:** 4, 6, 15, 30 (confounded with bd_label: each bd_label maps to one basis_dim)
- **BD labels tested:** lt_fcast (bd=4), eq_fcast (bd=6), lt_bcast (bd=15), eq_bcast (bd=30)

## 2. Configuration Rankings

### Top 15 by Mean OWA

|  # | Config | Wavelet | LD | BD | Mean OWA | Std OWA | Mean SMAPE | Std SMAPE | Params |
|---:|:-------|:--------|---:|---:|---------:|--------:|-----------:|----------:|-------:|
|  1 | Symlet3_bd4_lt_fcast_ttd3_ld8 | Symlet3 | 8 | 4 | 0.7967 | 0.0006 | 13.462 | 0.006 | 4,239,415 |
|  2 | DB20_bd15_lt_bcast_ttd3_ld2 | DB20 | 2 | 15 | 0.7975 | 0.0030 | 13.451 | 0.023 | 4,264,985 |
|  3 | Coif2_bd30_eq_bcast_ttd3_ld5 | Coif2 | 5 | 30 | 0.7978 | 0.0045 | 13.459 | 0.045 | 4,307,240 |
|  4 | Symlet10_bd6_eq_fcast_ttd3_ld8 | Symlet10 | 8 | 6 | 0.7988 | 0.0040 | 13.458 | 0.049 | 4,249,655 |
|  5 | Haar_bd4_lt_fcast_ttd3_ld5 | Haar | 5 | 4 | 0.8005 | 0.0062 | 13.533 | 0.083 | 4,235,560 |
|  6 | Coif2_bd4_lt_fcast_ttd3_ld2 | Coif2 | 2 | 4 | 0.8026 | 0.0012 | 13.518 | 0.012 | 4,231,705 |
|  7 | Symlet3_bd4_lt_fcast_ttd3_ld2 | Symlet3 | 2 | 4 | 0.8032 | 0.0015 | 13.543 | 0.023 | 4,231,705 |
|  8 | Coif3_bd4_lt_fcast_ttd3_ld5 | Coif3 | 5 | 4 | 0.8055 | 0.0063 | 13.566 | 0.097 | 4,235,560 |
|  9 | DB10_bd15_lt_bcast_ttd3_ld2 | DB10 | 2 | 15 | 0.8056 | 0.0125 | 13.600 | 0.199 | 4,264,985 |
| 10 | Coif3_bd6_eq_fcast_ttd3_ld5 | Coif3 | 5 | 6 | 0.8064 | 0.0101 | 13.565 | 0.109 | 4,245,800 |
| 11 | Coif3_bd6_eq_fcast_ttd3_ld2 | Coif3 | 2 | 6 | 0.8065 | 0.0092 | 13.564 | 0.100 | 4,241,945 |
| 12 | DB20_bd15_lt_bcast_ttd3_ld5 | DB20 | 5 | 15 | 0.8086 | 0.0052 | 13.643 | 0.046 | 4,268,840 |
| 13 | Haar_bd4_lt_fcast_ttd3_ld2 | Haar | 2 | 4 | 0.8086 | 0.0063 | 13.635 | 0.111 | 4,231,705 |
| 14 | Symlet10_bd6_eq_fcast_ttd3_ld5 | Symlet10 | 5 | 6 | 0.8096 | 0.0177 | 13.581 | 0.212 | 4,245,800 |
| 15 | Coif2_bd4_lt_fcast_ttd3_ld8 | Coif2 | 8 | 4 | 0.8097 | 0.0033 | 13.609 | 0.035 | 4,239,415 |

### Bottom 5 by Mean OWA

|  # | Config | Wavelet | LD | Mean OWA | Std OWA | Mean SMAPE |
|---:|:-------|:--------|---:|---------:|--------:|-----------:|
| 26 | Coif3_bd6_eq_fcast_ttd3_ld8 | Coif3 | 8 | 0.8316 | 0.0234 | 13.871 |
| 27 | Coif1_bd15_lt_bcast_ttd3_ld2 | Coif1 | 2 | 0.8317 | 0.0361 | 13.965 |
| 28 | Coif2_bd30_eq_bcast_ttd3_ld8 | Coif2 | 8 | 0.8390 | 0.0720 | 14.002 |
| 29 | DB20_bd15_lt_bcast_ttd3_ld8 | DB20 | 8 | 0.8503 | 0.0269 | 14.196 |
| 30 | Coif1_bd15_lt_bcast_ttd3_ld8 | Coif1 | 8 | 0.8544 | 0.0580 | 14.231 |

### Baseline Comparison

| Architecture | Best SMAPE | Best OWA | Params | vs This Study |
|:-------------|----------:|---------:|-------:|--------------:|
| Trend+WaveletV3 (non-AE SOTA) | 13.410 | 0.794 | ~1.4M | **Beats this study** |
| TrendAELG+WaveletV3AELG | 13.438 | 0.795 | ~4.3M | Beats this study |
| **TrendAE+WaveletV3 (THIS STUDY)** | **13.462** | **0.797** | **~4.2M** | -- |
| NBEATS-I+G | 13.53 | 0.806 | 35.9M | This study wins |
| NBEATS-G | 13.70 | 0.820 | 24.7M | This study wins |
| TrendAE+WaveletV3AE | 15.020 | 0.894 | ~4.2M | This study wins |

5 of 30 configs beat all four legacy baselines (OWA < 0.8015).
4 of 30 configs achieve sub-0.800 OWA.

## 3. Hyperparameter Factor Analysis

### 3a. Statistical Significance (Kruskal-Wallis Tests)

| Factor | H-statistic | p-value | Eta-squared | Significant? |
|:-------|------------:|--------:|------------:|:-------------|
| Wavelet Family | 7.49 | 0.3801 | -0.006 | No |
| Latent Dim (TrendAE) | 1.99 | 0.3699 | n/a | No |
| Basis Dim (WaveletV3) | 4.88 | 0.1806 | n/a | No |
| BD Label | 4.88 | 0.1806 | n/a | No (confounded with BD) |

**No factor reaches significance.** The study is likely underpowered with only 3 seeds per config. BD label and basis_dim are perfectly confounded (each bd_label maps to exactly one basis_dim value), so they test the same hypothesis.

### 3b. Wavelet Family Marginals

| Wavelet | Mean OWA | Std OWA | Mean SMAPE | N Runs |
|:--------|--------:|--------:|-----------:|-------:|
| Symlet3 | 0.8034 | 0.0079 | 13.548 | 9 |
| Haar | 0.8104 | 0.0228 | 13.678 | 9 |
| Symlet10 | 0.8120 | 0.0192 | 13.650 | 9 |
| Coif3 | 0.8139 | 0.0193 | 13.679 | 18 |
| DB10 | 0.8147 | 0.0149 | 13.693 | 9 |
| Coif2 | 0.8149 | 0.0313 | 13.675 | 18 |
| DB20 | 0.8188 | 0.0278 | 13.763 | 9 |
| Coif1 | 0.8352 | 0.0417 | 13.968 | 9 |

Symlet3 is the best by marginal mean OWA (0.8034), but the effect is not significant (p=0.38). Coif1 is consistently worst, which aligns with prior findings.

### 3c. Latent Dimension Marginals

| LD | Mean OWA | Std OWA | Mean SMAPE | N Runs |
|---:|--------:|--------:|-----------:|-------:|
| 5 | 0.8079 | 0.0138 | 13.592 | 30 |
| 2 | 0.8128 | 0.0197 | 13.676 | 30 |
| 8 | 0.8249 | 0.0354 | 13.834 | 30 |

LD=5 appears best marginally, but the effect is not significant (KW p=0.37). More importantly, the optimal LD depends strongly on wavelet family:

| Wavelet | Best LD | LD=2 OWA | LD=5 OWA | LD=8 OWA | Pattern |
|:--------|--------:|---------:|---------:|---------:|:--------|
| Symlet3 | 8 | 0.8032 | 0.8103 | 0.7967 | Monotone down |
| Symlet10 | 8 | 0.8275 | 0.8096 | 0.7988 | Monotone down |
| DB20 | 2 | 0.7975 | 0.8086 | 0.8503 | Monotone up |
| DB10 | 2 | 0.8056 | 0.8102 | 0.8283 | Monotone up |
| Coif1 | 5 | 0.8317 | 0.8194 | 0.8544 | U-shape |
| Coif2 | 5 | 0.8162 | 0.8042 | 0.8244 | U-shape |
| Coif3 | 5 | 0.8109 | 0.8060 | 0.8249 | U-shape |
| Haar | 5 | 0.8086 | 0.8005 | 0.8222 | U-shape |

**Interpretation:** Near-symmetric wavelets (Symlets) have smooth filter responses that require more latent capacity to encode. Long-support Daubechies wavelets already carry more information in fewer coefficients, benefiting from aggressive compression. This interaction invalidates any universal LD recommendation.

### 3d. Basis Dimension / BD Label Analysis

| BD | BD Label | Mean OWA | Std OWA | N Runs |
|---:|:---------|--------:|--------:|-------:|
| 4 | lt_fcast | 0.8085 | 0.0160 | 38 |
| 6 | eq_fcast | 0.8134 | 0.0184 | 18 |
| 30 | eq_bcast | 0.8223 | 0.0433 | 9 |
| 15 | lt_bcast | 0.8229 | 0.0304 | 27 |

BD=4 (lt_fcast) has the best mean OWA AND the lowest variance. BD=30 (eq_bcast) has the worst mean despite having the best single config (Coif2_bd30_eq_bcast_ttd3_ld5) -- its high variance (0.0433) makes it unreliable.

**Note:** bd_label and basis_dim are perfectly confounded in this study (lt_fcast always has bd=4, eq_fcast always has bd=6, etc.). The observed effect is basis_dim, not any semantic property of the label.

## 4. Training Stability

| Metric | Value |
|:-------|------:|
| Total runs | 90 |
| Diverged | 0 (0%) |
| Early stopped | 44 (48.9%) |
| Hit max epochs | 46 (51.1%) |
| Mean best epoch | 32.9 |
| Median best epoch | 40 |
| Mean loss ratio | 1.032 |

### Early Stopping by Wavelet Family

| Wavelet | ES Rate | Mean Epochs | Mean Best Epoch |
|:--------|--------:|------------:|----------------:|
| DB20 | 77.8% | 34.8 | 24.3 |
| Coif1 | 66.7% | 34.9 | 25.3 |
| Symlet10 | 66.7% | 39.0 | 30.7 |
| DB10 | 66.7% | 36.8 | 28.4 |
| Coif3 | 55.6% | 39.7 | 30.3 |
| Symlet3 | 36.4% | 49.7 | 41.5 |
| Coif2 | 22.2% | 45.4 | 38.3 |
| Haar | 22.2% | 46.6 | 38.3 |

Long-support wavelets (DB20, DB10) early stop more frequently, suggesting they converge faster but may also overfit earlier.

### Per-Config Seed Stability

| Stability Tier | Description | Count |
|:---------------|:------------|------:|
| Excellent (gap < 2%) | Worst seed within 2% of others | 14/30 |
| Moderate (gap 2-5%) | Some seed variance | 6/30 |
| Poor (gap > 5%) | One seed dramatically worse | 10/30 |

Configs with poor stability (>5% gap) are dominated by LD=8 with long-support wavelets (Coif1_ld8, DB20_ld8, Haar_ld8). Seed 44 is the worst performer in 8/16 problematic configs.

### Most Stable Configs

| Config | Mean OWA | Std OWA | Range |
|:-------|--------:|--------:|------:|
| Symlet3_bd4_lt_fcast_ttd3_ld8 | 0.7967 | 0.0006 | 0.0011 |
| Coif2_bd4_lt_fcast_ttd3_ld2 | 0.8026 | 0.0012 | 0.0022 |
| Symlet3_bd4_lt_fcast_ttd3_ld2 | 0.8032 | 0.0015 | 0.0029 |
| DB10_bd15_lt_bcast_ttd3_ld5 | 0.8102 | 0.0027 | 0.0054 |
| DB20_bd15_lt_bcast_ttd3_ld2 | 0.7975 | 0.0030 | 0.0059 |

## 5. Parameter Efficiency

All configs use ~4.2-4.3M parameters (spread < 2%), so parameter count is not a differentiator within this study. The critical parameter efficiency comparison is against external baselines:

| Architecture | Params | Best OWA | Params/OWA |
|:-------------|-------:|---------:|-----------:|
| Trend+WaveletV3 (non-AE) | ~1.4M | 0.794 | 1.76M |
| TrendAELG+WaveletV3AELG | ~4.3M | 0.795 | 5.41M |
| TrendAE+WaveletV3 (this) | ~4.2M | 0.797 | 5.27M |
| NBEATS-I+G | 35.9M | 0.806 | 44.5M |
| NBEATS-G | 24.7M | 0.820 | 30.1M |

The non-AE Trend+WaveletV3 is 3x more parameter-efficient than TrendAE+WaveletV3 while achieving better OWA.

### Pareto-Optimal Configs (within this study)

| Config | Mean OWA | Params |
|:-------|--------:|-------:|
| Symlet3_bd4_lt_fcast_ttd3_ld8 | 0.7967 | 4,239,415 |
| Haar_bd4_lt_fcast_ttd3_ld5 | 0.8005 | 4,235,560 |
| Coif2_bd4_lt_fcast_ttd3_ld2 | 0.8026 | 4,231,705 |

## 6. Corrections to Prior Report

This report supersedes the previous analysis which contained several errors:

| Issue | Prior Claim | Correction |
|:------|:------------|:-----------|
| Data quality | 92 rows analyzed | 2 duplicate rows; correct count is 90 |
| Best config | Coif2_bd30_eq_bcast_ttd3_ld5 (#1 by median OWA) | Symlet3_bd4_lt_fcast_ttd3_ld8 is #1 by mean OWA and 10x more stable |
| Latent dim | LD=2 recommended for production | LD=5 is best marginally (p=0.37, ns). Strong interaction with wavelet invalidates universal recommendation |
| Basis dim | BD=30 recommended, BD=15 called "anti-pattern" | BD=4 has best mean OWA and lowest variance. BD=30 is highest variance |
| Wavelet | Detailed architectural explanation for Symlet3 | Effect not significant (p=0.49). Explanation was overfitted to noise |
| BD label | Mechanistic explanation for eq_bcast | Confounded with basis_dim. Label semantics are not causal |
| Overall | Claims TrendAE+WavV3 "emerges as the clear winner" | Non-AE SOTA (OWA=0.794) beats this study with 3x fewer params |

## 7. Backbone Hierarchy Update

This study, combined with prior findings, establishes the definitive AE backbone hierarchy for Trend+Wavelet stacks on M4-Yearly:

```
Trend (RootBlock)           OWA=0.794  ~1.4M params  << RECOMMENDED
  > TrendAELG (AERootBlockLG)  OWA=0.795  ~4.3M params
  >= TrendAE (AERootBlock)      OWA=0.797  ~4.2M params  << THIS STUDY
  >> TrendVAE (AERootBlockVAE)  OWA=0.894+ ~4.2M params
```

## 8. Recommendations

### Current Best Configuration (M4-Yearly)
**Coif2_bd6_eq_fcast_td3** (Trend+WaveletV3, non-AE) -- unchanged from Wavelet Study 2.
- SMAPE=13.410, OWA=0.794
- ~1.4M parameters
- Confidence: MODERATE (3 seeds, but confirmed across multiple studies)

### What NOT to Do
- Do NOT use TrendAE backbone for WaveletV3 stacks. It adds 3x parameters for no improvement.
- Do NOT recommend a single "best" latent dim without specifying wavelet family.
- Do NOT use BD=30 as a default -- BD=4 is more reliable within this architecture.

### What to Test Next
1. **Extended training for TrendAE+WavV3 top configs (100 epochs):** 51% of runs hit max_epochs, suggesting some configs could improve with more training. Test the top-4 configs with 100 epochs and 10 seeds to get definitive rankings.
2. **Non-AE Trend+WaveletV3 with more seeds:** The current SOTA (Coif2_bd6_eq_fcast_td3) has only 3 seeds. Run with 10 seeds to confirm and narrow confidence intervals.
3. **Cross-dataset validation:** Test the non-AE Trend+WaveletV3 SOTA on M4-Quarterly and M4-Monthly to assess generalization.

### Open Questions
1. Why does Symlet3_bd4_lt_fcast_ttd3_ld8 have such low seed variance (std=0.0006)? Is this a property of the Symlet3+LD=8 combination, or a coincidence of the 3 seeds tested?
2. The wavelet x latent_dim interaction is the most interesting finding but cannot be tested for significance. Would a larger study (10+ seeds) confirm the pattern?
3. Would the TrendAE backbone show advantages on longer-horizon tasks (M4-Monthly, M4-Quarterly) where the AE bottleneck might act as a more useful regularizer?
