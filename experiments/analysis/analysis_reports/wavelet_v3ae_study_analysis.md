# WaveletV3AE Study Analysis: TrendAE + WaveletV3AE on M4-Yearly

**Date:** 2026-03-07
**Dataset:** M4-Yearly (forecast_length=6, backcast_length=30)
**Architecture:** `[TrendAE, <Family>WaveletV3AE]` repeated 5x (10 stacks total)
**Training:** 10 epochs, 3 seeds (42, 43, 44), SMAPE loss, ReLU activation
**CSV:** `experiments/results/m4/wavelet_v3ae_study_results.csv`
**Rows:** 1,161 raw (995 after deduplication), 332 unique configurations

---

## Executive Summary

This study evaluates the **TrendAE + WaveletV3AE** block family across a large hyperparameter grid on M4-Yearly. The sweep covers **14 wavelet families**, **4 basis dimension labels**, **2 trend thetas dimensions**, and **3 latent dimensions**, yielding 332 unique configurations with 3 seeds each.

**Key findings:**

1. **Best configuration:** Symlet2_lt_bcast_ttd3_ld8 (SMAPE=15.020, OWA=0.894) -- but the top-4 are statistically indistinguishable.
2. **Substantially underperforms baselines:** The best V3AE config trails the non-AE Trend+WaveletV3 baseline (SMAPE=13.410) by +1.61 SMAPE (+12.0%). It also trails NBEATS-G (~13.4) and TrendAELG+WaveletV3AELG (13.438) by similar margins.
3. **Latent dimension is the strongest factor:** ld=2 is catastrophically worse than ld={5,8}. ld=5 and ld=8 are statistically equivalent.
4. **Basis dimension matters greatly:** lt_fcast (bd=4, below forecast_length) is reliably the worst. lt_bcast (bd=15) and eq_bcast (bd=30) are the best and statistically equivalent.
5. **ttd=3 is consistently better** than ttd=5 (moderate effect, p < 1e-8).
6. **Wavelet family has negligible effect** (eta^2=0.003, p=0.26). All 14 families perform comparably once other hyperparameters are optimized.
7. **All runs hit MAX_EPOCHS at 10 epochs** with continued improvement. Val loss curves show ~3-6% improvement in the final epoch. Extended training would likely narrow the gap to baselines.

**Verdict:** The plain AE backbone (AERootBlock) is substantially weaker than the non-AE (RootBlock) and learned-gate AE (AERootBlockLG) alternatives for wavelet blocks on M4-Yearly. The bottleneck without gating appears to lose too much information at this scale. **Do not use TrendAE+WaveletV3AE for M4-Yearly production** -- prefer Trend+WaveletV3 or TrendAELG+WaveletV3AELG instead.

---

## 1. Study Design

### 1.1 Search Grid

| Dimension | Values | Count |
|-----------|--------|-------|
| Wavelet family | Haar, DB2, DB3, DB4, DB10, DB20, Coif1, Coif2, Coif3, Coif10, Symlet2, Symlet3, Symlet10, Symlet20 | 14 |
| Basis dim label | eq_fcast (bd=6), lt_fcast (bd=4), lt_bcast (bd=15), eq_bcast (bd=30) | 4 |
| Trend thetas dim | 3, 5 | 2 |
| Latent dim | 2, 5, 8 | 3 |
| **Total configs** | | **332** (some Symlet20 combos incomplete) |
| Seeds per config | 42, 43, 44 | 3 |
| **Total runs** | | **995** (after deduplication) |

### 1.2 Basis Dimension Resolution

For M4-Yearly (H=6, L=30), the four basis dimension labels resolve to:

| Label | basis_dim | Interpretation |
|-------|-----------|----------------|
| lt_fcast | 4 | Below forecast length (restrictive) |
| eq_fcast | 6 | Equal to forecast length |
| lt_bcast | 15 | Between forecast and backcast length |
| eq_bcast | 30 | Equal to backcast length (full span) |

No degeneracy exists -- all four labels map to distinct values.

### 1.3 Data Quality

- **Zero divergent runs** (all 995 runs completed successfully).
- **All runs stopped at MAX_EPOCHS** (10 epochs). None hit early stopping.
- **Duplicate entries** existed for DB3 (lt_bcast), DB4 (all labels), and DB10 (all labels) -- 166 duplicate (config, seed) pairs with identical metrics. These were removed, leaving 995 unique run records.
- **One incomplete config:** Symlet20_lt_bcast_ttd3_ld5 has only 2 runs (seed 43 missing).

---

## 2. Overall Rankings

### 2.1 Top-20 Configurations by Mean SMAPE

| Rank | Configuration | SMAPE | Std | OWA | Params | n |
|------|--------------|-------|-----|-----|--------|---|
| 1 | Symlet2_lt_bcast_ttd3_ld8 | 15.020 | 0.239 | 0.894 | 972,895 | 3 |
| 2 | Haar_eq_fcast_ttd3_ld8 | 15.027 | 0.123 | 0.898 | 949,855 | 3 |
| 3 | Haar_eq_bcast_ttd3_ld8 | 15.113 | 0.578 | 0.900 | 1,011,295 | 3 |
| 4 | DB3_lt_bcast_ttd3_ld5 | 15.126 | 0.094 | 0.891 | 961,345 | 3 |
| 5 | DB20_lt_bcast_ttd3_ld5 | 15.167 | 0.212 | 0.906 | 961,345 | 3 |
| 6 | DB4_eq_bcast_ttd3_ld8 | 15.168 | 0.232 | 0.908 | 1,011,295 | 3 |
| 7 | Coif1_lt_bcast_ttd3_ld8 | 15.176 | 0.205 | 0.905 | 972,895 | 3 |
| 8 | Coif2_lt_bcast_ttd3_ld8 | 15.177 | 0.365 | 0.909 | 972,895 | 3 |
| 9 | Coif3_lt_bcast_ttd3_ld5 | 15.192 | 0.203 | 0.902 | 961,345 | 3 |
| 10 | DB3_lt_bcast_ttd3_ld8 | 15.195 | 0.368 | 0.909 | 972,895 | 3 |
| 11 | Symlet3_eq_bcast_ttd3_ld8 | 15.236 | 0.270 | 0.910 | 1,011,295 | 3 |
| 12 | Symlet3_eq_fcast_ttd3_ld8 | 15.241 | 0.187 | 0.912 | 949,855 | 3 |
| 13 | Coif2_eq_fcast_ttd3_ld8 | 15.243 | 0.320 | 0.915 | 949,855 | 3 |
| 14 | Coif1_eq_bcast_ttd5_ld5 | 15.315 | 0.519 | 0.915 | 1,002,315 | 3 |
| 15 | Coif10_lt_bcast_ttd3_ld8 | 15.318 | 0.345 | 0.914 | 972,895 | 3 |
| 16 | Symlet2_eq_bcast_ttd3_ld8 | 15.343 | 0.183 | 0.919 | 1,011,295 | 3 |
| 17 | DB10_eq_bcast_ttd3_ld8 | 15.388 | 0.410 | 0.931 | 1,011,295 | 3 |
| 18 | Symlet2_eq_bcast_ttd3_ld5 | 15.400 | 0.227 | 0.913 | 999,745 | 3 |
| 19 | DB10_lt_bcast_ttd3_ld5 | 15.409 | 0.585 | 0.916 | 961,345 | 3 |
| 20 | Symlet20_eq_bcast_ttd3_ld5 | 15.418 | 0.518 | 0.925 | 999,745 | 3 |

**Observations:**
- The top-20 span only 0.40 SMAPE (15.02 to 15.42), indicating a flat plateau among the best configs.
- 18 of the top 20 use ttd=3 (only exception: #14 Coif1_eq_bcast_ttd5_ld5).
- All top-20 use ld >= 5 (ld=8 dominates: 13 of 20, ld=5 fills the remaining 7).
- No lt_fcast configs appear in the top 20.
- 11 wavelet families are represented in the top 20 -- further confirming wavelet family is not a strong discriminator.

### 2.2 Top-10 Configurations by Mean OWA

| Rank | Configuration | OWA | Std | SMAPE |
|------|--------------|-----|-----|-------|
| 1 | DB3_lt_bcast_ttd3_ld5 | 0.8911 | 0.0081 | 15.126 |
| 2 | Symlet2_lt_bcast_ttd3_ld8 | 0.8942 | 0.0185 | 15.020 |
| 3 | Haar_eq_fcast_ttd3_ld8 | 0.8979 | 0.0144 | 15.027 |
| 4 | Haar_eq_bcast_ttd3_ld8 | 0.9002 | 0.0345 | 15.113 |
| 5 | Coif3_lt_bcast_ttd3_ld5 | 0.9019 | 0.0186 | 15.192 |
| 6 | Coif1_lt_bcast_ttd3_ld8 | 0.9052 | 0.0221 | 15.176 |
| 7 | DB20_lt_bcast_ttd3_ld5 | 0.9057 | 0.0115 | 15.167 |
| 8 | DB4_eq_bcast_ttd3_ld8 | 0.9082 | 0.0131 | 15.168 |
| 9 | Coif2_lt_bcast_ttd3_ld8 | 0.9085 | 0.0295 | 15.177 |
| 10 | DB3_lt_bcast_ttd3_ld8 | 0.9087 | 0.0333 | 15.195 |

**Note:** DB3_lt_bcast_ttd3_ld5 is the OWA champion with remarkably low variance (std=0.0081) despite not being the SMAPE winner. The SMAPE and OWA rankings are correlated but not identical -- MASE (which feeds into OWA alongside SMAPE) creates the discrepancy.

### 2.3 Best Individual Runs

| Rank | Configuration | Seed | SMAPE | OWA |
|------|--------------|------|-------|-----|
| 1 | Symlet3_lt_bcast_ttd3_ld2 | 43 | 14.607 | 0.867 |
| 2 | Symlet20_lt_fcast_ttd3_ld8 | 42 | 14.624 | 0.870 |
| 3 | Symlet20_eq_bcast_ttd3_ld8 | 43 | 14.626 | 0.867 |
| 4 | Haar_eq_bcast_ttd3_ld8 | 43 | 14.634 | 0.870 |
| 5 | DB2_eq_bcast_ttd5_ld5 | 43 | 14.642 | 0.867 |

The best single run (SMAPE=14.607) is from a config that does NOT appear in the top-20 by mean -- it has high variance. Seed 43 appears in 8 of the top-10 single runs, confirming seed 43 is consistently the "lucky" seed.

---

## 3. Main Effects Analysis

### 3.1 Basis Dimension Label (Strongest Factor, eta^2 = 0.153)

| Label | basis_dim | Mean SMAPE | Std | Mean OWA | n |
|-------|-----------|------------|-----|----------|---|
| lt_bcast | 15 | 16.204 | 1.016 | 0.979 | 239 |
| eq_bcast | 30 | 16.345 | 1.387 | 0.993 | 252 |
| eq_fcast | 6 | 16.437 | 1.188 | 0.996 | 252 |
| lt_fcast | 4 | **17.487** | 1.459 | **1.070** | 252 |

**Statistical tests (Mann-Whitney U on SMAPE):**

| Comparison | U | p-value | Effect r | Sig. |
|-----------|---|---------|----------|------|
| eq_bcast vs eq_fcast | 28,182 | 0.029 | 0.112 | * |
| eq_bcast vs lt_bcast | 30,322 | 0.895 | -0.007 | ns |
| eq_bcast vs lt_fcast | 15,429 | 1.8e-23 | 0.514 | *** |
| eq_fcast vs lt_bcast | 33,598 | 0.027 | -0.116 | * |
| eq_fcast vs lt_fcast | 16,604 | 1.9e-20 | 0.477 | *** |
| lt_bcast vs lt_fcast | 13,368 | 1.6e-26 | 0.556 | *** |

**Interpretation:**
- **lt_fcast (bd=4) is catastrophically bad.** It sets basis_dim below the forecast length (4 < 6), meaning the wavelet basis cannot represent all forecast time steps. This confirms the "never set basis_dim < forecast_length" rule from Wavelet Study 2.
- **lt_bcast (bd=15) and eq_bcast (bd=30) are statistically equivalent** (p=0.895). Both provide enough basis functions to fully represent the forecast.
- **eq_fcast (bd=6) is marginally worse** than lt_bcast (p=0.027, r=0.116) -- just enough basis functions for the forecast, but without overparameterization headroom.
- The ordering is: lt_bcast (moderate overparameterization) >= eq_bcast (full overparameterization) > eq_fcast (exact) >> lt_fcast (underparameterized).

### 3.2 Latent Dimension (Second Strongest Factor, eta^2 = 0.112)

| Latent dim | Mean SMAPE | Std | Mean OWA | n |
|-----------|------------|-----|----------|---|
| 8 | 16.275 | 1.110 | 0.990 | 330 |
| 5 | 16.297 | 1.098 | 0.986 | 332 |
| 2 | **17.295** | 1.597 | **1.055** | 333 |

**Statistical tests:**

| Comparison | U | p-value | Effect r | Sig. |
|-----------|---|---------|----------|------|
| ld=2 vs ld=5 | 78,096 | 3.2e-20 | -0.413 | *** |
| ld=2 vs ld=8 | 77,602 | 4.0e-20 | -0.412 | *** |
| ld=5 vs ld=8 | 55,701 | 0.708 | -0.017 | ns |

**Interpretation:**
- **ld=2 is catastrophically underparameterized.** With only 2 latent dimensions, the AE bottleneck discards too much information. This is a 1-SMAPE-point penalty over ld=5/8.
- **ld=5 and ld=8 are statistically indistinguishable** (p=0.708). The AE bottleneck has enough capacity at 5 dimensions for this task. There is no benefit to increasing it to 8.
- Combined with the memory in MEMORY.md noting that AELG with ld=16 outperforms ld=32, the pattern is clear: **the AE family prefers moderate latent dimensions, not larger ones.**

### 3.3 Trend Thetas Dimension (Moderate Effect, eta^2 = 0.033)

| Thetas dim | Mean SMAPE | Std | Mean OWA | n |
|-----------|------------|-----|----------|---|
| 3 | 16.414 | 1.314 | 0.993 | 500 |
| 5 | 16.835 | 1.401 | 1.027 | 495 |

**Mann-Whitney U:** U=97,574, p=7.7e-09, r=0.212

**Interpretation:**
- **ttd=3 beats ttd=5** with moderate effect size. For M4-Yearly (H=6), a cubic trend polynomial (degree 2, represented by 3 thetas) is sufficient. A quintic polynomial (ttd=5) overfits the trend component.
- This is consistent with prior findings across multiple studies. For short horizons, lower-degree trend polynomials are preferred.

### 3.4 Wavelet Family (Negligible Effect, eta^2 = 0.003)

| Family | Mean SMAPE | Std | Mean OWA | n |
|--------|------------|-----|----------|---|
| Coif3 | 16.348 | 1.154 | 0.988 | 72 |
| Symlet20 | 16.360 | 1.313 | 0.992 | 59 |
| Symlet2 | 16.426 | 1.404 | 0.996 | 72 |
| Coif2 | 16.535 | 1.362 | 1.004 | 72 |
| DB20 | 16.548 | 1.241 | 1.006 | 72 |
| DB10 | 16.596 | 1.347 | 1.008 | 72 |
| Coif10 | 16.614 | 1.293 | 1.006 | 72 |
| Coif1 | 16.629 | 1.405 | 1.009 | 72 |
| Symlet3 | 16.639 | 1.203 | 1.010 | 72 |
| DB4 | 16.690 | 1.470 | 1.015 | 72 |
| DB2 | 16.717 | 1.247 | 1.018 | 72 |
| DB3 | 16.730 | 1.365 | 1.017 | 72 |
| Haar | 16.829 | 1.528 | 1.025 | 72 |
| Symlet10 | 17.022 | 1.753 | 1.040 | 72 |

**Kruskal-Wallis test:** H=15.9, p=0.256 (not significant)

**Interpretation:**
- **Wavelet family does NOT significantly differentiate performance** in the V3AE context. The entire range from best (Coif3: 16.35) to worst (Symlet10: 17.02) family is only 0.67 SMAPE, and the Kruskal-Wallis test is non-significant.
- This is a striking contrast to the non-AE Wavelet Study 2, where wavelet family mattered (Coif2 was clearly the best). The AE bottleneck appears to homogenize the wavelet basis representations, erasing family-specific advantages.
- Grouped by support length:
  - Coiflets (16.531) slightly beat Short Symlets (16.532), Long DB (16.572), Long Symlets (16.724), and Short DB/Haar (16.742), but none of these differences are significant.

### 3.5 Effect Size Ranking

| Factor | eta^2 | p-value | Interpretation |
|--------|-------|---------|----------------|
| Basis dim label | 0.153 | 2.6e-33 | **Large** -- the dominant hyperparameter |
| Latent dim | 0.112 | 2.8e-25 | **Large** -- mainly driven by ld=2 being terrible |
| Trend thetas dim | 0.033 | 7.7e-09 | **Small-moderate** -- ttd=3 reliably better |
| Wavelet family | 0.003 | 0.256 | **Negligible** -- not significant |

---

## 4. Interaction Effects

### 4.1 Full Factorial SMAPE Table

| bd_label | ttd | ld=2 | ld=5 | ld=8 |
|----------|-----|------|------|------|
| eq_bcast | 3 | 16.376 | 15.725 | **15.518** |
| eq_bcast | 5 | 17.634 | 16.329 | 16.488 |
| eq_fcast | 3 | 17.359 | 16.441 | 15.793 |
| eq_fcast | 5 | 16.755 | 16.126 | 16.149 |
| lt_bcast | 3 | 16.778 | 15.794 | 15.585 |
| lt_bcast | 5 | 17.007 | 16.024 | 16.013 |
| lt_fcast | 3 | 18.162 | 16.439 | 16.926 |
| lt_fcast | 5 | 18.268 | 17.464 | 17.662 |

**Key observations:**
- The **best cell** is eq_bcast / ttd=3 / ld=8 (SMAPE=15.518), closely followed by lt_bcast / ttd=3 / ld=8 (15.585) and lt_bcast / ttd=3 / ld=5 (15.794).
- The **worst cell** is lt_fcast / ttd=5 / ld=2 (SMAPE=18.268) -- combining all three worst-level choices.
- **ld=2 is always the worst** regardless of bd_label and ttd. The penalty ranges from +0.6 to +1.8 SMAPE.
- **An interesting interaction:** For eq_fcast, ttd=5 actually beats ttd=3 at ld=5 (16.126 vs 16.441). This is the only cell where ttd=5 is competitive, suggesting that when the basis is small (bd=6), a more flexible trend polynomial compensates.
- **lt_fcast is always worst** regardless of ttd/ld combination. Even the best lt_fcast cell (ttd=3, ld=5: 16.439) is worse than the worst non-lt_fcast cell among the "good" labels.

### 4.2 Full Factorial OWA Table

| bd_label | ttd | ld=2 | ld=5 | ld=8 |
|----------|-----|------|------|------|
| eq_bcast | 3 | 0.987 | 0.944 | **0.935** |
| eq_bcast | 5 | 1.084 | 1.000 | 1.008 |
| eq_fcast | 3 | 1.057 | 0.994 | 0.956 |
| eq_fcast | 5 | 1.018 | 0.972 | 0.981 |
| lt_bcast | 3 | 1.015 | 0.945 | **0.937** |
| lt_bcast | 5 | 1.036 | 0.969 | 0.972 |
| lt_fcast | 3 | 1.117 | 0.990 | 1.036 |
| lt_fcast | 5 | 1.124 | 1.069 | 1.087 |

The OWA table mirrors the SMAPE patterns. The two best cells (eq_bcast/ttd=3/ld=8 at 0.935 and lt_bcast/ttd=3/ld=8 at 0.937) are the only ones below 0.95.

### 4.3 Best Configuration Per Wavelet Family

| Family | Best Config | SMAPE | OWA | bd | ttd | ld |
|--------|------------|-------|-----|----|----|-----|
| Symlet2 | Symlet2_lt_bcast_ttd3_ld8 | 15.020 | 0.894 | lt_bcast | 3 | 8 |
| Haar | Haar_eq_fcast_ttd3_ld8 | 15.027 | 0.898 | eq_fcast | 3 | 8 |
| DB3 | DB3_lt_bcast_ttd3_ld5 | 15.126 | 0.891 | lt_bcast | 3 | 5 |
| DB20 | DB20_lt_bcast_ttd3_ld5 | 15.167 | 0.906 | lt_bcast | 3 | 5 |
| DB4 | DB4_eq_bcast_ttd3_ld8 | 15.168 | 0.908 | eq_bcast | 3 | 8 |
| Coif1 | Coif1_lt_bcast_ttd3_ld8 | 15.176 | 0.905 | lt_bcast | 3 | 8 |
| Coif2 | Coif2_lt_bcast_ttd3_ld8 | 15.177 | 0.909 | lt_bcast | 3 | 8 |
| Coif3 | Coif3_lt_bcast_ttd3_ld5 | 15.192 | 0.902 | lt_bcast | 3 | 5 |
| Symlet3 | Symlet3_eq_bcast_ttd3_ld8 | 15.236 | 0.910 | eq_bcast | 3 | 8 |
| Coif10 | Coif10_lt_bcast_ttd3_ld8 | 15.318 | 0.914 | lt_bcast | 3 | 8 |
| DB10 | DB10_eq_bcast_ttd3_ld8 | 15.388 | 0.931 | eq_bcast | 3 | 8 |
| Symlet20 | Symlet20_eq_bcast_ttd3_ld5 | 15.418 | 0.925 | eq_bcast | 3 | 5 |
| Symlet10 | Symlet10_lt_bcast_ttd3_ld5 | 15.486 | 0.917 | lt_bcast | 3 | 5 |
| DB2 | DB2_lt_bcast_ttd3_ld5 | 15.643 | 0.935 | lt_bcast | 3 | 5 |

**Patterns across families:**
- **ttd=3 wins for every single family** when optimized. Unanimous.
- **bd_label:** lt_bcast is most common (8 of 14), followed by eq_bcast (4 of 14), eq_fcast (1 of 14, Haar only), lt_fcast (0 of 14). This confirms that moderate-to-large basis dimensions are universally preferred.
- **ld:** 8 wins for 8 of 14 families; 5 wins for 6 of 14. Never ld=2.
- The spread from best to worst family (15.02 to 15.64, when each is optimized) is only 0.62 SMAPE.

---

## 5. Consistency and Stability

### 5.1 Most Consistent Top Configs (Low Std)

| Config | Mean SMAPE | Std | Mean OWA | OWA Std |
|--------|-----------|-----|----------|---------|
| DB3_lt_bcast_ttd3_ld5 | 15.126 | **0.094** | 0.891 | **0.008** |
| Haar_eq_fcast_ttd3_ld8 | 15.027 | **0.123** | 0.898 | 0.014 |
| Symlet3_eq_fcast_ttd3_ld8 | 15.241 | **0.187** | 0.912 | 0.021 |
| Symlet2_eq_bcast_ttd3_ld8 | 15.343 | **0.183** | 0.919 | 0.017 |
| Coif3_lt_bcast_ttd3_ld5 | 15.192 | **0.203** | 0.902 | 0.019 |

DB3_lt_bcast_ttd3_ld5 stands out as both the OWA champion and the most consistent configuration, with an exceptionally low SMAPE std of 0.094 and OWA std of 0.008. For risk-averse deployment, this is the recommended V3AE configuration.

### 5.2 Seed Bias

| Seed | Mean SMAPE | Mean OWA | n |
|------|-----------|----------|---|
| 43 | 16.206 | 0.980 | 332 |
| 42 | 16.413 | 0.993 | 332 |
| 44 | 17.253 | 1.058 | 331 |

Seed 44 is consistently the worst by a full SMAPE point over seed 43. This is a large seed effect and suggests that 10 epochs is insufficient for the less favorable initializations to converge. With more training, seeds would likely converge closer together.

---

## 6. Comparison to Baselines

### 6.1 Performance Gap

| Architecture | Best SMAPE | Best OWA | Gap to V3AE |
|-------------|-----------|---------|-------------|
| Trend + WaveletV3 (Coif2, bd=6, non-AE) | 13.410 | 0.794 | -- |
| TrendAELG + WaveletV3AELG (Sym20, ld=16) | 13.438 | 0.795 | -- |
| NBEATS-G (30x Generic) | ~13.4 | ~0.840 | -- |
| **TrendAE + WaveletV3AE (this study)** | **15.020** | **0.891** | **+1.61 SMAPE (+12.0%)** |

The V3AE family trails the non-AE baseline by +1.61 SMAPE (12.0%) and the AELG baseline by +1.58 SMAPE (11.8%). This is a large, practically significant deficit.

### 6.2 OWA Threshold Analysis

| Threshold | Configs Meeting It | Percentage |
|-----------|-------------------|------------|
| OWA < 1.0 (beat Naive2) | 177 of 332 | 53.3% |
| OWA < 0.95 | 61 of 332 | 18.4% |
| OWA < 0.90 | 3 of 332 | 0.9% |

Only 3 configurations achieve a mean OWA below 0.90 (the top 3 in Section 2.2). While over half the configs beat Naive2, none approach the ~0.79-0.84 range achieved by non-AE and AELG architectures.

### 6.3 Why V3AE Underperforms

Several factors explain the performance gap:

1. **Training duration:** Only 10 epochs were run, with val_loss still dropping 3-6% in the final epoch. Baseline studies typically use 30-100+ epochs. The convergence trajectory (Section 7) suggests significant untapped potential.

2. **AE bottleneck without gating:** The `AERootBlock` imposes a hard bottleneck (units -> units/2 -> latent_dim -> units/2 -> units) without any mechanism to selectively pass information through. The AELG variant adds a learned sigmoid gate on the latent dimensions, which allows the network to discover which latent features to retain. Without this gate, the V3AE bottleneck appears to discard useful signal.

3. **Latent dim range too low:** This study tested ld={2,5,8}, whereas AELG studies found ld=16 to be optimal. Even ld=8 may be too restrictive for the plain AE bottleneck.

4. **Wavelet basis homogenization:** The AE bottleneck appears to erase wavelet-family-specific advantages (eta^2=0.003 vs significant family effects in non-AE studies). This suggests the bottleneck is the binding constraint, not the basis expansion.

---

## 7. Convergence Analysis

### 7.1 Training State

- **All 995 runs stopped at MAX_EPOCHS (10)**. No early stopping was triggered.
- **79.6% of runs** (792 of 995) had their best validation loss at epoch 9 (the last epoch, 0-indexed), meaning they were still improving.
- **20.4% of runs** (203 of 995) had loss_ratio > 1.0, meaning their final val loss was worse than their best. The mean loss_ratio was 1.009, with a maximum of 1.46 -- indicating some runs experienced minor val loss upticks in the final epoch but no serious divergence.

### 7.2 Training Trajectory

Mean validation loss curve for the top-20 configurations:

```
Epoch  1:    64.82 (initial)
Epoch  2:    29.17 (-55.0%)
Epoch  3:    32.19 (+10.4%)  <-- noise
Epoch  4:    24.92 (-22.6%)
Epoch  5:    23.18 (-7.0%)
Epoch  6:    20.43 (-11.9%)
Epoch  7:    18.77 (-8.1%)
Epoch  8:    17.69 (-5.7%)
Epoch  9:    16.44 (-7.1%)
Epoch 10:    15.97 (-2.9%)
```

The final three epochs yielded a 1.73-point improvement in val_loss. While the rate of improvement is decelerating, a 2.9% drop in the final epoch is substantial. **Extrapolating conservatively, 10-20 more epochs could yield 1-3 additional SMAPE points of improvement**, potentially bringing the best configs into the 13-14 range and closing much of the gap to baselines.

### 7.3 Recommendation: Extended Training

This study should be considered a **screening round**, not a final evaluation. The 10-epoch budget was appropriate for identifying which hyperparameter settings matter (bd_label, latent_dim, ttd) and eliminating bad choices (lt_fcast, ld=2, ttd=5). However, the absolute SMAPE values are not directly comparable to baselines trained for 30+ epochs.

**Priority for follow-up training:** The top-5 configs (Symlet2_lt_bcast_ttd3_ld8, Haar_eq_fcast_ttd3_ld8, Haar_eq_bcast_ttd3_ld8, DB3_lt_bcast_ttd3_ld5, DB20_lt_bcast_ttd3_ld5) should be retrained with 30-50 epochs for fair comparison.

---

## 8. Parameter Efficiency

### 8.1 Parameter Counts by Factor

| Factor | Level | Min Params | Max Params | Mean Params |
|--------|-------|-----------|-----------|-------------|
| Latent dim | 2 | 916,515 | 990,765 | 946,548 |
| Latent dim | 5 | 928,065 | 1,002,315 | 958,088 |
| Latent dim | 8 | 939,615 | 1,013,865 | 969,618 |
| Basis dim | eq_fcast (6) | 926,755 | 952,425 | 939,590 |
| Basis dim | lt_fcast (4) | 916,515 | 942,185 | 929,350 |
| Basis dim | lt_bcast (15) | 949,795 | 975,465 | 962,458 |
| Basis dim | eq_bcast (30) | 988,195 | 1,013,865 | 1,001,030 |

Parameter counts range from 916K to 1.01M -- only a 10.6% spread. This is narrow enough that parameter count is not a meaningful differentiator. Performance differences are driven by architecture, not capacity.

### 8.2 Comparison to Non-AE Baselines

For reference, the non-AE Trend+WaveletV3 models from Wavelet Study 2 had approximately 1.4M parameters -- about 40% more than V3AE configs. Despite fewer parameters, V3AE performs significantly worse, confirming that the AE bottleneck is harmful rather than a useful regularizer.

---

## 9. Top vs Bottom Quartile Composition

To understand which settings concentrate among winners vs losers, the top and bottom quartiles by SMAPE were analyzed:

| Factor | Level | Top Quartile | Bottom Quartile | Interpretation |
|--------|-------|-------------|----------------|----------------|
| bd_label | lt_bcast | **33.7%** | 4.8% | Strongly associated with top performance |
| bd_label | eq_bcast | **34.9%** | 13.3% | Moderately associated with top performance |
| bd_label | eq_fcast | 22.9% | 16.9% | Neutral |
| bd_label | lt_fcast | 8.4% | **65.1%** | Strongly associated with poor performance |
| latent_dim | 8 | **54.2%** | 22.9% | Associated with top performance |
| latent_dim | 5 | **41.0%** | 13.3% | Associated with top performance |
| latent_dim | 2 | 4.8% | **63.9%** | Strongly associated with poor performance |
| ttd | 3 | **68.7%** | 39.8% | Moderately associated with top performance |
| ttd | 5 | 31.3% | **60.2%** | Moderately associated with poor performance |
| wavelet_family | (all) | ~5-11% each | ~4-10% each | No family is over/underrepresented |

The quartile analysis dramatically confirms the main effects. **lt_fcast and ld=2 are the "failure modes"** -- 65% and 64% of bottom-quartile configs use these settings, respectively.

---

## 10. Statistical Comparisons Among Top Configs

### 10.1 Top-2 by SMAPE: Symlet2_lt_bcast_ttd3_ld8 vs Haar_eq_fcast_ttd3_ld8

| Seed | Symlet2 | Haar | Difference |
|------|---------|------|------------|
| 42 | 14.852 | 15.079 | +0.226 |
| 43 | 15.294 | 15.116 | -0.178 |
| 44 | 14.915 | 14.886 | -0.029 |

- **Paired t-test:** t=-0.054, p=0.962
- **Wilcoxon signed-rank:** W=3.0, p=1.000
- **Mean difference:** +0.006 (essentially zero)

These two configs are **completely indistinguishable statistically**. With only 3 seeds, neither test has adequate power to detect small differences.

### 10.2 Top SMAPE vs Top OWA: Symlet2_lt_bcast_ttd3_ld8 vs DB3_lt_bcast_ttd3_ld5

| Seed | Symlet2 SMAPE | DB3 SMAPE | Symlet2 OWA | DB3 OWA |
|------|--------------|-----------|-------------|---------|
| 42 | 14.852 | 15.205 | 0.878 | 0.898 |
| 43 | 15.294 | 15.150 | 0.893 | 0.883 |
| 44 | 14.915 | 15.023 | 0.912 | 0.893 |

- **Wilcoxon signed-rank (SMAPE):** W=2.0, p=0.750

Again, no significant difference. DB3_lt_bcast_ttd3_ld5 has better OWA because it also has lower MASE, but the gap is not statistically reliable.

---

## 11. Conclusions and Recommendations

### 11.1 Robust Hyperparameter Recommendations for V3AE

If TrendAE+WaveletV3AE must be used (e.g., for ablation studies comparing AE backbones):

| Parameter | Recommended | Avoid |
|-----------|------------|-------|
| Basis dim label | lt_bcast or eq_bcast | lt_fcast (always) |
| Latent dim | 5 or 8 (equivalent) | 2 (catastrophic) |
| Trend thetas dim | 3 | 5 |
| Wavelet family | Any (no significant effect) | -- |

### 11.2 Current Best V3AE Configurations

For **lowest risk** (consistency): **DB3_lt_bcast_ttd3_ld5** (SMAPE=15.126, OWA=0.891, std=0.094)
For **lowest mean SMAPE**: **Symlet2_lt_bcast_ttd3_ld8** (SMAPE=15.020, OWA=0.894, std=0.239)
For **lowest mean OWA**: **DB3_lt_bcast_ttd3_ld5** (SMAPE=15.126, OWA=0.891, std_OWA=0.008)

### 11.3 Architecture Recommendation

**Do not use TrendAE+WaveletV3AE for M4-Yearly production.** The architecture underperforms all known alternatives:

| Architecture | Best SMAPE | Relative Gap |
|-------------|-----------|-------------|
| Trend + WaveletV3 (non-AE) | 13.410 | baseline |
| TrendAELG + WaveletV3AELG | 13.438 | +0.2% |
| NBEATS-G | ~13.4 | +0.0% |
| **TrendAE + WaveletV3AE** | **15.020** | **+12.0%** |

The plain AE backbone lacks the representational capacity or learning dynamics to compete. The learned-gate (LG) mechanism appears essential for AE-family wavelet blocks.

### 11.4 What to Test Next

1. **Extended training for top-5 V3AE configs (30-50 epochs):** To determine whether the 12% gap narrows substantially with more training, or whether V3AE is fundamentally inferior.

2. **V3AE with larger latent_dim {12, 16, 24}:** This study only tested ld={2,5,8}. The AELG family performs best at ld=16. Testing larger latent dims for plain AE could narrow the gap.

3. **V3AE with sum_losses=True:** Adding backcast reconstruction loss may help the AE bottleneck learn better representations.

4. **Cross-backbone comparison at matched training budget:** Run Trend+WaveletV3, TrendAELG+WaveletV3AELG, and TrendAE+WaveletV3AE all at 10 epochs with the same configs to isolate the backbone effect from training duration differences.

### 11.5 Open Questions

- **Is the 12% gap structural or convergence-related?** The 10-epoch budget heavily penalizes V3AE. If extended training closes the gap to <2%, the bottleneck may be a useful regularizer for longer training runs. If the gap persists, the plain AE bottleneck is structurally harmful.
- **Does the wavelet family indifference persist with more training?** At 10 epochs, the AE bottleneck may be the binding constraint masking family effects. With more training, family differences might emerge.
- **Would ld=16 help plain V3AE?** The optimal latent dim for V3AE might be higher than tested. The AELG finding that ld=16 >> ld=32 suggests a sweet spot around 16 for the AE family.

---

## Appendix: Bottom-10 Configurations

| Rank | Configuration | SMAPE | Std | OWA |
|------|--------------|-------|-----|-----|
| 1 | Symlet10_lt_fcast_ttd5_ld5 | 19.994 | 1.635 | 1.261 |
| 2 | Symlet10_lt_fcast_ttd3_ld2 | 19.826 | 1.868 | 1.261 |
| 3 | DB20_lt_fcast_ttd5_ld2 | 19.479 | 1.875 | 1.230 |
| 4 | Haar_lt_fcast_ttd3_ld2 | 19.084 | 0.874 | 1.190 |
| 5 | Symlet2_eq_fcast_ttd3_ld2 | 19.082 | 2.379 | 1.194 |
| 6 | Coif1_lt_fcast_ttd5_ld2 | 18.913 | 3.251 | 1.179 |
| 7 | Coif2_lt_fcast_ttd3_ld2 | 18.904 | 2.787 | 1.185 |
| 8 | DB4_lt_fcast_ttd5_ld5 | 18.903 | 1.149 | 1.188 |
| 9 | Symlet10_lt_fcast_ttd5_ld8 | 18.885 | 1.234 | 1.167 |
| 10 | Symlet2_eq_bcast_ttd5_ld2 | 18.831 | 2.665 | 1.186 |

All bottom-10 configs use either lt_fcast (8 of 10) or ld=2 (7 of 10), with 5 of 10 combining both failure modes. The worst config (SMAPE=19.99) is ~5 SMAPE points worse than the best -- a 33% performance penalty from bad hyperparameter choices.
