# TrendWaveletAE v2 Tourism-Yearly Study Analysis

**Date:** 2026-03-09
**Dataset:** Tourism-Yearly (forecast_length=4, backcast_length=8)
**CSV:** `experiments/results/tourism/trendwaveletae_v2_study_results.csv`
**Notebook:** `experiments/analysis/notebooks/trendwaveletae_v2_tourism_insights.ipynb`

## Executive Summary

This study tested TrendWaveletAE and TrendWaveletAELG unified blocks (10 stacks, shared weights) on Tourism-Yearly using successive halving: 40 configs x 3 runs (R1, 15 epochs), top 20 x 5 runs (R2, 50 epochs). 220 total rows, zero divergence.

**The study does NOT produce a new Tourism-Yearly SOTA.** The best config (TrendWaveletAE_db20_eq_fcast_td3_ld12, SMAPE=21.013) is 0.15 points (+0.72%) above the prior SOTA of 20.864 (TrendWaveletAELG coif3_eq_bcast_td3_ld16). However, the study reveals important factor relationships: Haar is the best wavelet for short horizons, ld=12 > ld=8, and basis_label is a non-factor on Tourism.

## Data Quality

- **220 rows:** 120 R1 + 100 R2, exactly as expected
- **Zero divergence:** All runs converged normally
- **Convergence:** 69% of R2 runs early-stopped (mean 44.4 epochs of 50 max)
- **SMAPE range:** 20.715 - 28.607 (R1 included)
- **No duplicates detected**

## Final Rankings (Round 2)

| Rank | Config | SMAPE | Std | Block | Wavelet | bd_label | ld | Delta |
|------|--------|-------|-----|-------|---------|----------|----|-------|
| 1 | TrendWaveletAE_db20_eq_fcast_td3_ld12 | 21.013 | 0.143 | AE | db20 | eq_fcast | 12 | BEST |
| 2 | TrendWaveletAE_haar_lt_fcast_td3_ld8 | 21.048 | 0.179 | AE | haar | lt_fcast | 8 | +0.035 |
| 3 | TrendWaveletAELG_haar_eq_fcast_td3_ld12 | 21.055 | 0.367 | AELG | haar | eq_fcast | 12 | +0.042 |
| 4 | TrendWaveletAE_haar_eq_fcast_td3_ld8 | 21.119 | 0.310 | AE | haar | eq_fcast | 8 | +0.106 |
| 5 | TrendWaveletAELG_db20_eq_fcast_td3_ld12 | 21.169 | 0.161 | AELG | db20 | eq_fcast | 12 | +0.156 |
| 6 | TrendWaveletAE_db3_eq_fcast_td3_ld8 | 21.209 | 0.207 | AE | db3 | eq_fcast | 8 | +0.196 |
| 7 | TrendWaveletAELG_db3_eq_fcast_td3_ld12 | 21.239 | 0.335 | AELG | db3 | eq_fcast | 12 | +0.226 |
| 8 | TrendWaveletAELG_sym10_lt_fcast_td3_ld8 | 21.299 | 0.139 | AELG | sym10 | lt_fcast | 8 | +0.286 |
| 9 | TrendWaveletAE_db3_lt_fcast_td3_ld12 | 21.305 | 0.146 | AE | db3 | lt_fcast | 12 | +0.292 |
| 10 | TrendWaveletAE_db20_lt_fcast_td3_ld12 | 21.350 | 0.322 | AE | db20 | lt_fcast | 12 | +0.337 |

The top 3 are within 0.042 SMAPE of each other -- statistically indistinguishable (pairwise MWU p > 0.42).

## Factor Analysis

### 1. Wavelet Family -- Haar is the clear winner (p=0.009)

| Family | Mean SMAPE | Std | Median | n |
|--------|-----------|-----|--------|---|
| haar | 21.074 | 0.266 | 20.995 | 15 |
| db3 | 21.251 | 0.221 | 21.242 | 15 |
| db20 | 21.361 | 0.379 | 21.275 | 30 |
| sym10 | 21.417 | 0.374 | 21.378 | 25 |
| coif2 | 21.441 | 0.342 | 21.263 | 15 |

**Kruskal-Wallis:** H=13.45, p=0.009, eta^2=0.100

**Post-hoc (Haar vs each):**
- haar vs db3: p=0.042, d=-0.72
- haar vs db20: p=0.006, d=-0.88
- haar vs sym10: p=0.002, d=-1.06
- haar vs coif2: p=0.003, d=-1.20

Haar significantly outperforms all other families with large effect sizes. This is a **dataset-specific finding**: on M4-Yearly (H=6), wavelet family is a non-factor. Tourism's very short horizon (H=4) means that long-support wavelets (db20, sym10) collapse to approximation-only bases, losing the multi-resolution advantage that wavelets provide.

### 2. Basis Label -- Non-factor (p=0.65)

| Label | basis_dim | Mean SMAPE | Std | n |
|-------|-----------|-----------|-----|---|
| eq_fcast | 4 | 21.329 | 0.384 | 70 |
| lt_fcast | 2 | 21.324 | 0.291 | 30 |

MWU p=0.654. Even halving the basis dimensionality from 4 to 2 makes no difference on 4-step forecasts. This extends the Tourism degeneracy pattern seen in prior studies (eq_fcast = lt_bcast when H < L).

### 3. Latent Dimension -- ld=12 significantly better (p=0.020)

| ld | Mean SMAPE | Std | n |
|----|-----------|-----|---|
| 8 | 21.384 | 0.346 | 60 |
| 12 | 21.244 | 0.360 | 40 |

MWU p=0.020, Cohen's d=0.40 (small-to-medium effect).

**Interaction with block type:**
- TrendWaveletAE: ld=8 (21.293) vs ld=12 (21.223) -- gap=0.07
- TrendWaveletAELG: ld=8 (21.511) vs ld=12 (21.256) -- gap=0.26

The AELG backbone benefits more from larger latent dimensions, which makes sense: the learned gate has more dimensions to select from with ld=12.

### 4. Block Type -- AE edges ahead, not significant (p=0.24)

| Block Type | Mean SMAPE | Std | n |
|-----------|-----------|-----|---|
| TrendWaveletAE | 21.272 | 0.287 | 50 |
| TrendWaveletAELG | 21.383 | 0.410 | 50 |

MWU p=0.243, Cohen's d=-0.315. AE occupies 3 of the top 4 positions and has lower variance. The learned gate does not help on this short-horizon task.

### 5. Comparison to Prior SOTA

**Prior SOTA:** TrendWaveletAELG coif3_eq_bcast_td3_ld16, SMAPE=20.864, 95% CI [20.732, 20.996]

**This study's best:** TrendWaveletAE_db20_eq_fcast_td3_ld12, SMAPE=21.013, 95% CI [20.901, 21.126]

- Delta: +0.149 (+0.72%)
- One-sample t-test (H0: mean = 20.864): t=2.33, p=0.080
- Individual runs below SOTA: 1/5

The SOTA was not beaten. The prior SOTA used coif3 (not tested here) and ld=16 (not tested here). Given that ld=12 > ld=8 significantly, ld=16 likely provides further improvement.

### 6. Eliminated Configs (Round 1)

20 configs were eliminated after R1. Key patterns among eliminated configs:
- 10/20 eliminated configs were AELG (proportional to the 50/50 split)
- lt_fcast + ld=12 combinations were disproportionately eliminated (6/20 vs 5/40 in full search space)
- All AELG lt_fcast configs were eliminated except sym10_lt_fcast_ld8

### 7. Convergence

- 69% of R2 runs early-stopped, 31% hit max_epochs=50
- Mean epochs trained: 44.4
- Zero divergence across all 220 runs
- Unified TrendWavelet blocks are extremely stable on Tourism-Yearly

## Recommendations

### Current Best Configuration (Tourism-Yearly)

**SOTA remains:** TrendWaveletAELG coif3_eq_bcast_td3_ld16 (SMAPE=20.864)

This study's best is competitive but does not beat it.

### What to Test Next

1. **Haar + ld=16 on Tourism-Yearly.** Haar won the wavelet comparison in this study, and ld=16 is the established best latent dim from prior studies. This exact combination has never been tested.

```yaml
# experiments/configs/tourism_haar_ld16.yaml
experiment_name: trendwaveletae_haar_ld16_tourism
dataset: tourism
periods: [Yearly]
stacks:
  homogeneous:
    block_type: TrendWaveletAE
    n_stacks: 10
    share_weights: true
training:
  max_epochs: 50
  loss: SMAPELoss
  optimizer: Adam
  learning_rate: 0.001
  batch_size: 1024
extra_csv_columns:
  block_type: str
  wavelet_type: str
  latent_dim_cfg: int
configs:
  - name: TrendWaveletAE_haar_eq_fcast_td3_ld16
    block_args: {wavelet_type: haar, basis_dim: 4, trend_thetas_dim: 3, latent_dim: 16}
    extra_fields: {block_type: TrendWaveletAE, wavelet_type: haar, latent_dim_cfg: 16}
  - name: TrendWaveletAELG_haar_eq_fcast_td3_ld16
    stacks:
      homogeneous:
        block_type: TrendWaveletAELG
        n_stacks: 10
        share_weights: true
    block_args: {wavelet_type: haar, basis_dim: 4, trend_thetas_dim: 3, latent_dim: 16}
    extra_fields: {block_type: TrendWaveletAELG, wavelet_type: haar, latent_dim_cfg: 16}
search:
  rounds:
    - {max_epochs: 50, n_runs: 10}
```

2. **Test coif3 at ld=12** to directly compare with the SOTA config at a different latent dim.

3. **Alternating Trend+HaarWavelet on Tourism.** The trend_seas_wav comparison study showed alternating architectures are Pareto-optimal on M4. Verify on Tourism.

### Open Questions

1. **Does the ld=12 > ld=8 trend continue to ld=16?** Prior M4 evidence says yes, but Tourism-specific confirmation is needed.
2. **Why does Haar win on Tourism but not M4?** The hypothesis (short horizon favors short-support wavelets) is plausible but needs verification at H=6 where Haar and longer wavelets should be closer.
3. **Is the prior SOTA (coif3) truly better than Haar, or was it a lucky seed draw?** A head-to-head at matched ld=16 would resolve this.
