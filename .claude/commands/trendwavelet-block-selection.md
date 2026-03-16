# TrendWavelet Block Selection: Unified vs Alternating

## Core Rule

When choosing between TrendWaveletAELG (unified block) and TrendAELG + WaveletV3AELG (alternating stacks):

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| **Tourism-Yearly** | **TrendWaveletAELG** (or test GAE10) | Confirmed SMAPE=20.864 (10-run); GAE10_no_skip achieves 20.526 (needs head-to-head ≥10 seeds to confirm SOTA) |
| Short horizon (H<=10), simplicity preferred | **TrendWaveletAELG** | Simpler architecture, competitive results |
| M4-Yearly, maximum accuracy | Trend+WaveletV3 (non-AE) | SOTA 13.410, unified is +0.40% |
| **Traffic-96** | **TrendAELG+WaveletV3AELG (alternating)** | Requires L≥5H lookback and ≤8-10 stacks; converges with MSE ~0.0006 on Traffic |
| **Weather-96 (alternating, preferred)** | **TrendAE+WaveletV3AE (alternating, AE not AELG)** | Gate fn study FM5: MSE=0.121. AE beats AELG (MWU p=0.036). Use `forecast_multiplier=5` |
| **Weather-96 (non-alternating)** | **TrendVAE+HaarWaveletV3 at 10-20 stacks** | ResNet skip v2 R3: TVH20_skip5_a01 MSE=0.133 (bl=480); not sig. vs TVH10_no_skip |

---

## TrendWaveletAELG Configuration

When using TrendWaveletAELG:

### Hyperparameter defaults

- `latent_dim=16` (significantly better than ld=8, p=0.042)
- `trend_thetas_dim=3` for H<=10 (confirmed, p=0.255 directional)
- `basis_dim_label=eq_fcast` (significantly best, p=0.017)
- `forecast_multiplier=5` (FM5 beats FM7 on both M4 and Weather, p<0.001 each; gate fn study 2026-03-15)
- `n_stacks=10`, `n_blocks_per_stack=1`, `share_weights=True`

### Wavelet family: does NOT matter

Wavelet family is a non-factor for TrendWaveletAELG (Kruskal-Wallis p=0.107 across 14 families). The AE bottleneck homogenizes basis representations. Use any family; coif3 has best cross-dataset average rank.

**This is the opposite of alternating stacks** where sym20 is the universal best family.

### Depth scaling (skip study v2, updated 2026-03-10)

- **TrendWaveletAELG is depth-stable from 10 to 30 stacks on M4.** SMAPE 13.57 at both 10 and 30 stacks (no degradation). CV < 1%.
- **TrendWaveletAE is equally depth-stable on M4** from 10 to 30 stacks. SMAPE 13.58 at 30 stacks.
- **Skip connections are NOT needed on M4** and slightly hurt at 30 stacks (+0.09 SMAPE). The integrated polynomial+DWT basis provides sufficient inductive bias to prevent residual decay.
- **On Tourism, skip actively hurts unified TrendWaveletAE** (MWU p=0.001). Do not use skip for unified TrendWavelet on Tourism. Note: alternating TrendAELG+WaveletV3AELG v1 on Tourism did see marginal skip benefit (p=0.016) — different architecture.
- **Both AE and AELG backbones perform equivalently** for the unified TrendWavelet block at all depths (unlike alternating stacks where LG > non-LG).

### Known instability

**Avoid DB4 + eq_fcast + ld=16.** This specific combination catastrophically fails (SMAPE ~76). DB4 with other bd_labels or at ld=8 is fine.

---

## Key Differences from Alternating Stacks

| Property | TrendWaveletAELG (unified) | TrendAELG + WaveletV3AELG (alternating) |
|----------|---------------------------|----------------------------------------|
| Wavelet family effect | Non-significant (p=0.107) | Significant (sym20 best) |
| Basis label sensitivity | High (0.37 SMAPE spread) | Low (0.014 spread at convergence) |
| Cross-dataset family consistency | Uncorrelated (rho=-0.1) | Correlated (sym20 universal) |
| Traffic-96 (L=2H, 20 stacks) | 100% failure | 86% divergence |
| Traffic-96 (L=5H, 8 stacks) | Not tested | 16% divergence, MSE ~0.0006 |
| Tourism-Yearly | **SOTA** (20.681) | 20.930 |
| M4-Yearly | 13.463 (+0.40%) | 13.438 (+0.21%) |
| Architecture simplicity | 1 block type | 2 block types alternating |

### Alternating stacks: AELG vs AE backbone is dataset-dependent (gate fn study, 2026-03-15)

| Dataset | Better backbone | Evidence |
|---------|-----------------|----------|
| M4-Yearly | AELG (weakly) | All 4 architectures favor AELG, none significant |
| Weather-96 | **AE (significantly)** | MWU p=0.036. Top 2 Weather configs are AE controls (no gate) |

On Weather-96, the learned gate appears to over-constrain the latent space for long-horizon forecasting. Prefer AE for Weather alternating stacks. The gate function itself (sigmoid vs wavy_sigmoid vs wavelet_sigmoid) is a non-factor on both datasets (KW p>0.20).

### Lookback multiplier: FM5 > FM7 (gate fn study, 2026-03-15)

`forecast_multiplier=5` significantly outperforms `forecast_multiplier=7` on **both** datasets:

- M4-Yearly: FM5 wins 16/16 matched configs (Wilcoxon p<0.001), -0.09 SMAPE (-0.66%)
- Weather-96: FM5 wins 13/13 matched configs (Wilcoxon p<0.001), -0.035 MSE (-17.5%)

Use FM=5 as the default for AE/AELG architectures on these datasets.

### T+W+G does not improve over T+W

Adding a Generic block to the alternating T+W pattern adds parameters without improving accuracy on either dataset. Stick with T+W alternating for parameter efficiency.

---

## Evidence

| Study | Report | Data |
|-------|--------|------|
| TrendWaveletAELG Pure v1 (M4, 112 configs) | `experiments/analysis/analysis_reports/trendwaveletaelg_pure_study_analysis.md` | `experiments/results/m4/trendwaveletaelg_pure_study_results.csv` |
| TrendWaveletAELG Pure v2 (4 datasets) | Same report | `experiments/results/*/trendwaveletaelg_pure_v2_study_results.csv` |
| Skip Study v2 (36 configs, depth scaling) | `experiments/analysis/analysis_reports/resnet_skip_study_v2_analysis.md` | `experiments/results/m4/resnet_skip_study_v2_results.csv` |
| Skip Study v2 Notebook | `experiments/analysis/analysis_reports/resnet_skip_study_v2_analysis.ipynb` | See notebook |
| Notebook | `experiments/analysis/notebooks/trendwaveletaelg_pure_study_insights.ipynb` | See notebook |
| Gate Function Study (M4+Weather, 303 runs) | `experiments/analysis/analysis_reports/gate_fn_study_analysis.md` | `experiments/results/m4/gate_fn_study_results.csv`, `experiments/results/weather/gate_fn_study_results.csv` |
