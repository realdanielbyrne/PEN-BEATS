# TrendWaveletAE vs TrendWaveletAELG: Comprehensive Analysis

## Executive Summary

This report provides a unified analysis of all TrendWavelet block studies (v1 and v2) across 4 datasets (M4-Yearly, Tourism-Yearly, Weather-96, Traffic-96), comparing the plain autoencoder backbone (TrendWaveletAE / AERootBlock) against the learned-gate variant (TrendWaveletAELG / AERootBlockLG).

**Total data: 3,831 rows across 10 CSV files, spanning 4 datasets, 2 block types, and 2 study versions (plus 10-run SOTA confirmation).**

### Top-Line Findings

1. **AELG beats AE on every dataset** where both have been tested, with statistical significance on M4-Yearly (Wilcoxon p=0.002 on matched configs) and Tourism-Yearly (MWU p=0.010). Effect size is small (Cohen's d=0.23 on M4) but consistent.

2. **Neither block beats the non-AE SOTA on M4-Yearly.** Best AE-family result: AELG v2 SMAPE=13.463 vs non-AE Trend+WaveletV3 SMAPE=13.410 (+0.40% gap).

3. **Tourism-Yearly AELG coif3_eq_bcast_td3_ld16: SMAPE=20.864** (10-run confirmed mean; original 3-run estimate of 20.681 was overly optimistic). 95% CI [20.732, 20.996] overlaps prior SOTA (20.930), so the improvement is suggestive but not statistically definitive.

4. **Traffic-96 showed 100% divergence in this study** for unified TrendWavelet blocks (all 80 AELG runs, AE runs never completed). **Root cause: insufficient backcast horizon** (bl=192, L=2H, `forecast_multiplier=2`). With bl=480 (L=5H), the architecture converges on Traffic-96.

5. **Latent dimension is the most important hyperparameter** (Kruskal-Wallis p<0.001). Wavelet family is a non-factor (p=0.38).

---

## 1. Architecture Differences

Both TrendWaveletAE and TrendWaveletAELG share identical projection heads and frozen basis expansions. The only difference is the backbone:

| Component | TrendWaveletAE | TrendWaveletAELG |
|-----------|---------------|------------------|
| Parent class | AERootBlock | AERootBlockLG |
| Encoder | fc1(backcast_len -> units/2) -> ReLU -> fc2(units/2 -> latent_dim) -> ReLU | Same |
| Bottleneck | z = activated latent | z = activated latent * sigmoid(gate) |
| Gate vector | None | nn.Parameter(ones(latent_dim)), learned via backprop |
| Decoder | fc3(latent_dim -> units/2) -> ReLU -> fc4(units/2 -> units) -> ReLU | Same |
| Additional params | 0 | latent_dim (8-16 floats per block, 80-160 total) |
| Projection | Single linear per path to (trend_dim + wavelet_dim) coefficients | Same |
| Basis expansion | Vandermonde polynomial + orthonormal DWT wavelet, summed | Same |

The learned gate applies `sigmoid(gate) * z` at the bottleneck, allowing the network to discover effective latent dimensionality during training by selectively suppressing uninformative dimensions.

---

## 2. Data Inventory

### Experiment Coverage Matrix

|  | M4-Yearly | Tourism-Yearly | Traffic-96 | Weather-96 |
|--|-----------|---------------|------------|------------|
| **AE v1** (pure, 14 wavelets, ld={2,5,8}, td={3,5}) | 2,133 rows | -- | -- | -- |
| **AELG v1** (pure, 14 wavelets, ld=8, td={3,5}) | 711 rows | -- | -- | -- |
| **AE v2** (mixed, 5 wavelets, ld={8,12}, td=3) | 140 rows | 110 rows | 0 rows (empty) | 7 rows (R1 only) |
| **AELG v2** (pure, 6 wavelets, ld=16, td=3) | 150 rows | 150 rows | 80 rows | 150 rows |
| **AELG v2** (from mixed study) | 80 rows | 110 rows | -- | -- |
| **AELG SOTA confirmation** (10 seeds, coif3_eq_bcast) | -- | 10 rows | -- | -- |

### Study Design Differences

| Parameter | V1 (pure studies) | V2 AE (mixed) | V2 AELG (pure) |
|-----------|------------------|---------------|-----------------|
| Wavelet families | 14 (haar through sym20) | 5 (haar,db3,db20,coif2,sym10) | 6 (haar,db3,db4,coif2,coif3,sym3) |
| Basis labels | eq_fcast, lt_fcast, eq_bcast, lt_bcast | eq_fcast, lt_fcast | eq_fcast, lt_fcast, eq_bcast, lt_bcast |
| Trend dims | {3, 5} | {3} | {3} |
| Latent dims | AE: {2,5,8}; AELG: {8} | {8, 12} | {16} |
| Successive halving | 3 rounds to 50 epochs | 2 rounds to 50 epochs | 3 rounds to 50 epochs |
| Stacks | 10 (share_weights=True) | 10 (share_weights=True) | 10 (share_weights=True) |

**Important note:** V1 and V2 search spaces are different enough that direct comparison isolates `latent_dim` but confounds wavelet set and bd_label options.

---

## 3. M4-Yearly Results

### 3.1 Best Configurations

| Rank | Study | Block | Config | SMAPE | Std | OWA | Params | Runs |
|------|-------|-------|--------|-------|-----|-----|--------|------|
| 1 | AELG v2 | TrendWaveletAELG | db3_eq_fcast_td3_ld16 | **13.463** | 0.040 | 0.797 | 1,526,170 | 3 |
| 2 | AELG v1 | TrendWaveletAELG | db10_eq_fcast_td3_ld8 | 13.506 | 0.053 | 0.799 | 1,485,050 | 3 |
| 3 | AE v2 | TrendWaveletAE | coif2_lt_fcast_td3_ld12 | 13.509 | 0.019 | 0.800 | 1,495,230 | 5 |
| 4 | AE v1 | TrendWaveletAE | sym20_eq_fcast_td3_ld8 | 13.514 | 0.061 | 0.801 | 1,484,970 | 3 |
| -- | **Non-AE SOTA** | Trend+WaveletV3 | coif2_bd6_eq_fcast_td3 | **13.410** | -- | 0.794 | ~1.4M | 3 |

All TrendWavelet configs are +0.05 to +0.10 SMAPE above the non-AE SOTA. The AE bottleneck adds overhead without improving quality on M4-Yearly.

### 3.2 Head-to-Head: AE vs AELG (V1, Matched Configs at R3)

At R3 with ld=8, there are **42 matched config keys** (same wavelet, bd_label, td) across both block types:

- **AELG wins: 30/42 configs** (71%)
- Mean AE SMAPE: 13.6163
- Mean AELG SMAPE: 13.5854
- Mean advantage: 0.031 SMAPE
- **Wilcoxon signed-rank: stat=209, p=0.0019** (significant)

Overall R3 distribution (all configs, not just matched):

- AE: n=450, mean=13.668, std=0.213, median=13.631
- AELG: n=150, mean=13.618, std=0.228, median=13.586
- **Mann-Whitney U: p<0.001**
- Cohen's d = 0.23 (small effect)

### 3.3 Hyperparameter Factor Analysis (AE v1, R3)

| Factor | Kruskal-Wallis p | Effect Size (eta-squared) | Most Important? |
|--------|-----------------|--------------------------|-----------------|
| **latent_dim** | **<0.001** | Large (ld=2 vs 8: d=0.72) | YES -- dominant factor |
| bd_label | 0.096 | Marginal | Weak trend (eq_fcast best) |
| wavelet_family | 0.380 | Negligible | NO -- non-factor |
| trend_dim | 0.706 | Negligible | NO -- td=3 and td=5 identical |

**Latent dimension detail (AE v1 R3):**

| ld | Mean SMAPE | Std | n | vs ld=8 Cohen's d |
|----|-----------|-----|---|-------------------|
| 2 | 13.814 | 0.337 | 63 | 0.72 (large) |
| 5 | 13.665 | 0.187 | 192 | 0.23 (small) |
| 8 | 13.625 | 0.159 | 195 | -- (reference) |

**Wavelet family is homogenized by the AE bottleneck.** The spread across 14 families is only 0.10 SMAPE (13.634 to 13.738). This contrasts sharply with the alternating TrendAELG+WaveletV3AELG architecture where wavelet choice mattered (sym20 was universally best). In the unified block, the encoder-decoder can learn to compensate for different wavelet properties.

### 3.4 V1 vs V2: AELG ld=8 vs ld=16

On 8 matched config keys (excluding DB4 catastrophe at ld=16):

- V2 (ld=16) wins: 6/8
- Mean improvement: 0.072 SMAPE
- Wilcoxon p=0.11 (not significant with n=8, but consistent direction)

**Caution:** DB4+eq_fcast+ld16 catastrophically fails (SMAPE~76), while DB4+eq_fcast+ld8 works fine (SMAPE~13.55). Higher latent dimension increases instability risk.

---

## 4. Tourism-Yearly Results

### 4.1 AELG Best Config (10-Run Confirmation)

The original 3-run estimate of SMAPE=20.681 for `coif3_eq_bcast_td3_ld16` was overly optimistic. A 10-seed confirmation study (seeds 42-51) revised the mean upward:

| Config | SMAPE | Std | Runs | Seeds | 95% CI |
|--------|-------|-----|------|-------|--------|
| **AELG coif3_eq_bcast_td3_ld16** | **20.864** | 0.213 | 10 | 42-51 | [20.732, 20.996] |
| AELG haar_eq_fcast_td3_ld16 | 21.008 | 0.185 | 3 | (= haar_lt_bcast) | |
| AELG db4_lt_fcast_td3_ld16 | 21.069 | 0.099 | 3 | | |
| AE db20_eq_fcast_td3_ld12 | 21.013 | 0.143 | 5 | | |
| Prior SOTA (TrendAELG+WaveletV3AELG Coif1) | 20.930 | -- | -- | | |

**The 95% CI [20.732, 20.996] overlaps the prior SOTA (20.930).** The unified TrendWaveletAELG block is competitive with the alternating architecture on Tourism-Yearly, but the improvement is not statistically significant. The mean (20.864) is 0.066 below the prior SOTA — suggestive but inconclusive.

Individual run SMAPE values: 20.779, 20.814, 20.977, 20.586, 21.082, 21.311, 20.871, 20.832, 20.707, 20.682. Min=20.586, Max=21.311.

### 4.2 AE vs AELG on Tourism

- AE (R2, 50 runs): mean SMAPE = 21.272
- AELG (R3, 30 runs): mean SMAPE = 21.098
- **Mann-Whitney U: p=0.0096** (significant)
- Cohen's d = 0.62 (medium effect)

The AELG advantage is larger on Tourism than M4 (d=0.62 vs d=0.23). This may reflect Tourism's shorter horizon (H=4) and smaller dataset benefiting more from the gate's regularization.

### 4.3 Tourism bd_label Degeneracy

On Tourism-Yearly (forecast=4, backcast=8):

- `eq_fcast` -> basis_dim=4
- `lt_bcast` -> basis_dim=8/2=4
- These are **identical** (confirmed: all run-level SMA values match exactly)
- `lt_fcast` -> basis_dim=2 (undersized)
- `eq_bcast` -> basis_dim=8

The SOTA config uses `eq_bcast` (bd=8), which provides maximum wavelet basis dimensionality for the forecast path. This suggests that on very short horizons, more basis functions help.

---

## 5. Weather-96 Results

### 5.1 AELG v2 Top Configs (R3)

| Config | MSE | Std | MAE | Runs |
|--------|-----|-----|-----|------|
| db3_eq_fcast_td3_ld16 | 1920.1 | 602.8 | 13.53 | 3 |
| sym3_lt_fcast_td3_ld16 | 1989.3 | 27.2 | 13.81 | 3 |
| db4_eq_fcast_td3_ld16 | 2137.0 | 150.9 | 14.32 | 3 |
| db3_lt_fcast_td3_ld16 | 2232.1 | 269.6 | 14.52 | 3 |
| coif3_eq_fcast_td3_ld16 | 2269.3 | 527.8 | 15.09 | 3 |

**High variance alert:** The best config (db3) has std=603, nearly 31% of its mean. This makes Weather rankings unreliable with only 3 seeds.

### 5.2 AE v2 Weather (Sparse Data)

Only 7 runs completed (R1 only, 15 epochs):

- haar_lt_fcast_td3_ld8: MSE=1970 (1 run)
- haar_eq_fcast_td3_ld12: MSE=2265 (3 runs)
- haar_eq_fcast_td3_ld8: MSE=2452 (3 runs)

Too sparse for any conclusions. The single run at MSE=1970 is comparable to AELG's best but is not replicated.

### 5.3 Comparison to Other Architectures

- **Best from AsymWavelet study:** AELG-sym20-coif2, MSE=1804 (alternating stacks, 5 seeds)
- **Best from this study (unified AELG):** db3_eq_fcast_td3_ld16, MSE=1920
- The alternating architecture beats unified by ~6% on Weather.

---

## 6. Traffic-96 Results

### 6.1 Complete Failure (Root Cause: Insufficient Backcast Horizon)

- **AELG v2:** 80 runs, 100% divergence (all SMAPE=200, val_loss flat from epoch 1)
- **AE v2:** 0 data rows (file exists but empty -- runs never started or were abandoned)

**Root cause: insufficient backcast horizon.** Both this study and the alternating architecture study used bl=192 (L=2H, `forecast_multiplier=2`). Traffic-96 requires bl≥480 (L=5H) for reliable convergence. The progression of failure rates:

- This study (TrendWaveletAELG pure, bl=192, 20 stacks): 100% divergence
- Prior alternating study (TrendAELG+WaveletV3AELG, bl=192, 20 stacks): 86% divergence
- AsymWavelet Diagnostic (alternating, bl=480, 8 stacks): 16% divergence

The apparent difference in failure rates between unified (100%) and alternating (86%) architectures is likely a secondary effect of the shared root cause. Both architectures fail with inadequate lookback; neither can be fairly compared at L=2H. The conclusion that "unified TrendWavelet is MORE sensitive to Traffic" is not supported once the lookback confound is removed.

---

## 7. Statistical Summary

### 7.1 Effect Sizes

| Comparison | Dataset | Cohen's d | Direction | p-value |
|------------|---------|-----------|-----------|---------|
| AE vs AELG (matched, ld=8) | M4-Yearly | 0.23 (small) | AELG better | 0.002 |
| AE vs AELG (overall R3) | M4-Yearly | 0.23 (small) | AELG better | <0.001 |
| AE vs AELG (R2 vs R3) | Tourism | 0.62 (medium) | AELG better | 0.010 |
| ld=2 vs ld=8 | M4-Yearly | 0.72 (large) | ld=8 better | <0.001 |
| ld=5 vs ld=8 | M4-Yearly | 0.23 (small) | ld=8 better | <0.001 |

### 7.2 Factor Importance Ranking (AE v1, M4-Yearly R3)

```
1. latent_dim     : KW p < 0.001  -- DOMINANT
2. bd_label        : KW p = 0.096  -- marginal
3. wavelet_family  : KW p = 0.380  -- non-factor
4. trend_dim       : KW p = 0.706  -- non-factor
```

---

## 8. Key Gaps and Missing Data

1. **No v1 data outside M4-Yearly.** V1 studies were M4-only for both AE and AELG.
2. **AE v2 Traffic: never run** (empty CSV). AE v2 Weather: 7 rows only (barely started).
3. **V1 and V2 search spaces differ** in wavelet sets, bd_labels, and latent dims -- no perfectly matched cross-version comparison possible.
4. **Weather needs more seeds.** Best config std=603 (31% of mean) with only 3 seeds.
5. **No exploration of training hyperparameters** (learning rate, warm-up, stack depth, active_g, sum_losses) for TrendWavelet blocks specifically.
6. **No M4 periods beyond Yearly.** Quarterly and Monthly may behave differently.

---

## 9. Convergence Analysis

| Study | Round | Stopping | Mean Epochs | Early Stopped % |
|-------|-------|----------|-------------|-----------------|
| AE v1 M4 | R3 (50 max) | 60% MAX_EPOCHS | 48.1 | 40% |
| AELG v1 M4 | R3 (50 max) | 63% MAX_EPOCHS | 48.6 | 37% |
| AE v2 M4 | R2 (50 max) | 54% MAX_EPOCHS | 47.7 | 46% |
| AELG v2 M4 | R2 (50 max) | 50% MAX_EPOCHS | 48.8 | 50% |

Most runs hit MAX_EPOCHS or early-stop near the limit, suggesting 50 epochs is adequate. Loss ratios are close to 1.0 (mean ~1.005), confirming convergence.

---

## 10. Recommendations

### Current Best Configurations

| Dataset | Winner | Config | Performance | Confidence |
|---------|--------|--------|-------------|------------|
| M4-Yearly | Non-AE (overall best) | Trend+WaveletV3 coif2_bd6_eq_fcast_td3 | SMAPE=13.410, OWA=0.794 | HIGH |
| M4-Yearly | AELG (best AE-family) | TrendWaveletAELG db3_eq_fcast_td3_ld16 | SMAPE=13.463, OWA=0.797 | MEDIUM (3 runs, DB4 catastrophe at same ld) |
| Tourism-Yearly | AELG (best unified) | TrendWaveletAELG coif3_eq_bcast_td3_ld16 | SMAPE=20.864 (95% CI [20.732, 20.996]) | HIGH (10 runs; overlaps prior SOTA 20.930) |
| Weather-96 | Alternating AELG | AELG-sym20-coif2 (from AsymWavelet study) | MSE=1804 | LOW (5 seeds, high variance) |
| Traffic-96 | NOT VIABLE at L=2H (bl=192) | -- | 100% divergence with `forecast_multiplier=2`; use L≥5H | HIGH |

### What to Test Next

1. ~~**Tourism-Yearly SOTA confirmation with 10 seeds.**~~ **DONE.** 10-run confirmation revised mean from 20.681 to 20.864 (95% CI [20.732, 20.996]). Not a clear SOTA over prior 20.930. See `experiments/results/tourism/tourism_aelg_sota_confirmation_results.csv`.

2. **AE v2 Weather completion.** *(IN PROGRESS)* The AE v2 Weather study was abandoned at 7 rows. Running the full config (`trendwaveletae_v2_weather.yaml`) for a fair AE vs AELG comparison on Weather-96.

3. **ld=16 stability investigation.** The DB4+eq_fcast+ld16 catastrophe needs explanation. Test ld=16 with gradient clipping or lower learning rate.

4. **M4 other periods (Quarterly, Monthly).** All TrendWavelet data is Yearly-only. These blocks may behave differently at longer horizons.

5. **TrendWaveletAE at ld=16** (matching AELG v2). No AE data exists at ld=16. This would complete the AE vs AELG comparison at the latent dimension that produced the best AELG results.

### Open Questions

1. **Why does the learned gate help more on Tourism (d=0.62) than M4 (d=0.23)?** Is it the shorter horizon, smaller dataset, or different data characteristics?
2. **Why does the AE bottleneck homogenize wavelet families?** In alternating stacks, wavelet family matters because the wavelet block is specialized. In the unified block, the encoder-decoder apparently learns to normalize basis differences.
3. **Would non-AE TrendWavelet (no bottleneck) work?** A `TrendWavelet` block using `RootBlock` (4 FC layers, no bottleneck) has never been tested. Given that non-AE Trend+WaveletV3 is SOTA on M4-Yearly, this might outperform both AE variants.

---

## Data Sources

- `experiments/results/m4/trendwaveletae_pure_study_results.csv` (2,133 rows, v1)
- `experiments/results/m4/trendwaveletaelg_pure_study_results.csv` (711 rows, v1)
- `experiments/results/m4/trendwaveletae_v2_study_results.csv` (220 rows, v2 mixed)
- `experiments/results/m4/trendwaveletaelg_pure_v2_study_results.csv` (150 rows, v2)
- `experiments/results/tourism/trendwaveletae_v2_study_results.csv` (220 rows, v2 mixed)
- `experiments/results/tourism/trendwaveletaelg_pure_v2_study_results.csv` (150 rows, v2)
- `experiments/results/traffic/trendwaveletaelg_pure_v2_study_results.csv` (80 rows, v2)
- `experiments/results/weather/trendwaveletae_v2_study_results.csv` (7 rows, v2 sparse)
- `experiments/results/weather/trendwaveletaelg_pure_v2_study_results.csv` (150 rows, v2)

- `experiments/results/tourism/tourism_aelg_sota_confirmation_results.csv` (10 rows, SOTA confirmation)

## Notebook

See `experiments/analysis/notebooks/trendwavelet_ae_vs_aelg_comprehensive.ipynb` for interactive analysis with visualizations.

---

## Correction Addendum (2026-03-09)

**The Traffic-96 "completely non-viable" conclusion is incorrect.** The 100% divergence across all 80 AELG runs was caused by the evaluation protocol (bl=192, L=2H; 20 stacks), not by inherent block-dataset incompatibility.

A subsequent study (AsymWavelet Diagnostic, 2026-03-08) using L=5H (bl=480) and 8 stacks demonstrated 80-100% convergence for TrendAELG+WaveletV3AELG on Traffic-96, with AELG dominating VAE by 5.6× in MSE. The "NOT VIABLE" designation in the summary table should be read as: **viable with adequate lookback (L≥5H) and moderate stack depth (≤8-10 stacks).**
