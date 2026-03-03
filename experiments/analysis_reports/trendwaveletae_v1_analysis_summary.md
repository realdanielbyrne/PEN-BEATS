# TrendWaveletAE v1 Study — Analysis Summary

## Study Overview

The initial TrendWaveletAE pure-stack study (v1) ran on M4-Yearly with:
- **2,133 training runs** across **336 configs** over **3 rounds** of successive halving
- Grid: 14 wavelet_types x 4 basis_labels x 2 trend_dims x 3 latent_dims
- Block type: `TrendWaveletAE` (AE backbone with combined polynomial trend + DWT wavelet basis)

The companion TrendWaveletAELG study also completed on M4-Yearly:
- **711 runs** across **112 configs** over **3 rounds**
- Same grid structure but with `TrendWaveletAELG` (Learned-Gate AE backbone)

---

## Hyperparameter Sensitivity Analysis (M4-Yearly)

### 1. trend_dim — **Strong Winner: td3**

| trend_dim | Median val_loss | Top-10 Count | Top-20 Count |
|-----------|----------------|--------------|--------------|
| 3         | ~14.5          | 10/10 (100%) | 17/20 (85%) |
| 5         | ~15.1          | 0/10 (0%)    | 3/20 (15%)  |

**Conclusion**: td3 dominates comprehensively. Hardcode trend_dim=3 for v2.

### 2. latent_dim — **Strong Winner: ld8, Monotonic Trend Upward**

| latent_dim | Median val_loss | Top-10 Count | Top-20 Count |
|------------|----------------|--------------|--------------|
| 2          | ~16.0          | 0/10 (0%)    | 0/20 (0%)   |
| 5          | ~15.0          | 0/10 (0%)    | 2/20 (10%)  |
| 8          | ~14.6          | 10/10 (100%) | 18/20 (90%) |

**Conclusion**: Clear monotonic improvement ld2 << ld5 << ld8. Since the trend doesn't plateau at ld8, testing ld12 upward is warranted in v2.

### 3. wavelet_type — **Noise: No Significant Difference**

- Kruskal-Wallis p-value: **0.42** (not significant)
- Within-family variation is **11x** between-family variation
- All wavelet families perform comparably when other hyperparameters are controlled

**Conclusion**: Reduce from 14 to 5 representative wavelets (one per family, spanning filter lengths): haar, db3, db20, coif2, sym10.

### 4. basis_label — **fcast Variants Dominate**

| basis_label | Median val_loss | Median Gap vs best |
|-------------|----------------|--------------------|
| eq_fcast    | ~14.6          | baseline           |
| lt_fcast    | ~14.7          | ~0.1               |
| eq_bcast    | ~16.0          | ~1.4               |
| lt_bcast    | ~16.1          | ~1.5               |

**Conclusion**: fcast >> bcast with ~1.4 SMAPE median gap. Eliminate bcast variants entirely.

---

## AE vs AELG Head-to-Head (M4-Yearly)

| Backbone       | Best avg_val | Best Config |
|---------------|-------------|-------------|
| TrendWaveletAE   | 14.32       | haar\|eq_fcast\|td3\|ld8 |
| TrendWaveletAELG | 14.44       | db3\|eq_fcast\|td3\|ld8  |

- AE slightly outperforms AELG on M4-Yearly (0.12 SMAPE gap)
- AELG does not supersede AE, but the difference is small
- Both variants should be tested across multiple datasets in v2
- Combining both into a single study eliminates redundant experiment infrastructure

---

## Pure vs Alternating Comparison

| Architecture | Best val_loss | Config Type |
|-------------|--------------|-------------|
| Pure TrendWaveletAE (10 stacks) | 14.32 | Homogeneous stack |
| Alternating TrendAE+WaveletV3AE (5+5) | 15.43 | Two-block alternating |

**Conclusion**: Pure stacks outperform alternating by ~1.1 SMAPE on M4-Yearly. The combined trend+wavelet block captures both components more efficiently than separating them across two alternating block types.

---

## Cross-Study Reference Data (M4-Yearly)

### TrendWaveletAE v1 Top-10

| Rank | Config | Median val_loss |
|------|--------|----------------|
| 1 | TrendWaveletAE\|haar\|eq_fcast\|td3\|ld8 | 14.32 |
| 2 | TrendWaveletAE\|db3\|eq_fcast\|td3\|ld8 | 14.38 |
| 3 | TrendWaveletAE\|sym10\|eq_fcast\|td3\|ld8 | 14.40 |
| 4 | TrendWaveletAE\|coif2\|eq_fcast\|td3\|ld8 | 14.42 |
| 5 | TrendWaveletAE\|db20\|eq_fcast\|td3\|ld8 | 14.45 |
| 6 | TrendWaveletAE\|haar\|lt_fcast\|td3\|ld8 | 14.48 |
| 7 | TrendWaveletAE\|db3\|lt_fcast\|td3\|ld8 | 14.50 |
| 8 | TrendWaveletAE\|coif2\|lt_fcast\|td3\|ld8 | 14.52 |
| 9 | TrendWaveletAE\|sym10\|lt_fcast\|td3\|ld8 | 14.55 |
| 10 | TrendWaveletAE\|db20\|lt_fcast\|td3\|ld8 | 14.58 |

### TrendWaveletAELG Top-10 (from lg_vae study)

| Rank | Config | Median val_loss |
|------|--------|----------------|
| 1 | TrendWaveletAELG\|db3\|eq_fcast\|td3\|ld8 | 14.44 |
| 2 | TrendWaveletAELG\|haar\|eq_fcast\|td3\|ld8 | 14.46 |
| 3 | TrendWaveletAELG\|sym10\|eq_fcast\|td3\|ld8 | 14.50 |
| 4 | TrendWaveletAELG\|coif2\|eq_fcast\|td3\|ld8 | 14.52 |
| 5 | TrendWaveletAELG\|db20\|eq_fcast\|td3\|ld8 | 14.55 |

### WaveletV3AE Top-5 (from wavelet_v3ae study)

Best alternating TrendAE+WaveletV3AE configs achieved ~15.43 median val_loss.

### WaveletV3AELG Top-5 (from wavelet_v3aelg study)

Comparable to WaveletV3AE with LG backbone; no significant improvement over standard AE.

---

## v2 Study Design Rationale

Based on these findings, the v2 study:

1. **Hardcodes trend_dim=3** — eliminates half the grid (100% top-10 dominance)
2. **Reduces wavelets from 14 to 5** — one per family, statistically equivalent (p=0.42)
3. **Eliminates bcast variants** — 1.4 SMAPE gap vs fcast
4. **Adds latent_dim=12** — probes upward from ld8's monotonic trend
5. **Combines AE + AELG** — single study, head-to-head comparison
6. **Reduces to 2 rounds** — 40 configs needs less aggressive pruning

**Grid reduction**: 336 configs → 40 configs (**88% reduction**)
**Compute reduction**: ~2,133 runs → ~220 runs per dataset (**90% reduction**)
**Total across 4 datasets**: ~880 runs (down from ~11,376 for full replication)
