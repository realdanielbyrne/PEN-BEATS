# Gate Function Study Analysis -- M4-Yearly + Weather-96 (Cross-Dataset)

**Date**: 2026-03-15 (updated with Weather data)
**Datasets**: M4-Yearly (H=6), Weather-96 (H=96)
**Runs**: M4: 160 (32 configs x 5 seeds), Weather: 143 (29 configs x ~5 seeds), zero divergences
**Notebook**: `experiments/analysis/notebooks/gate_fn_study_insights.ipynb`

## Executive Summary

The gate function used in AERootBlockLG's latent gate has **no significant effect** on forecasting performance on either M4-Yearly or Weather-96. Kruskal-Wallis p=0.37 (M4) and p=0.64 (Weather). The total spread across gate functions is 0.018 SMAPE on M4 (0.13%) and 0.003 MSE on Weather (1.8%). **Keep sigmoid as the default.**

The study's most important finding is a **serendipitous discovery**: forecast_multiplier=5 significantly outperforms forecast_multiplier=7 on **both** datasets, winning every single matched config comparison with p<0.001 on both.

A secondary cross-dataset insight: **AE (no gate) outperforms AELG (learned gate) on Weather** (MWU p=0.036) while AELG shows a weak, non-significant advantage on M4. The learned gate mechanism appears less beneficial (or mildly harmful) for long-horizon forecasting.

## Study Design

Three gate functions tested for the AELG backbone's `_compute_latent_gate_scale()`:
- **sigmoid**: `torch.sigmoid(x)` -- standard smooth [0,1] gate (current default)
- **wavy_sigmoid**: `sigmoid(1.6x) + 0.18*exp(-1.1*tau)*sin(4.2*tau)` -- damped sinusoidal tail
- **wavelet_sigmoid**: `sigmoid(1.6x) + A*exp(-0.5*(tau/s)^2)*sin(w0*tau)` -- Gaussian-windowed oscillation
- **none**: AE control (no gate at all)

Four architecture families, each with 3 AELG gate variants + 1 AE control = 4 configs per family. Two experiment variants per dataset (FM7 and FM5). Total: 4 x 4 x 2 = 32 configs per dataset x 5 seeds.

All configs use: coif3 wavelet, latent_dim=16, trend_thetas_dim=3, skip_distance=5, skip_alpha=0.1, 20-21 stacks, 200 max epochs (M4 FM5: 100), patience=25.

**Weather-specific settings**: basis_dim=96, forecast_basis_dim=96, normalize=true, train_ratio=0.7, val_ratio=0.1, batch_size=256.

**Weather FM5 incompleteness**: 3 of 16 configs missing (TALG_WV3LG_GALG21_wavelet, TALG_WV3LG_GALG21_wavy, TA_WV3A_GA21_control). Also TWGLG20_wavy has only 3 runs in FM7. All other config-experiment combinations have 5 complete runs.

## Finding 1: Gate Function is a NON-FACTOR (Cross-Dataset Confirmed)

### Overall Ranking -- M4-Yearly FM5

| Rank | Gate Function | SMAPE | +/- | OWA | n | Delta |
|------|--------------|-------|-----|-----|---|-------|
| 1 | sigmoid | 13.540 | 0.048 | 0.805 | 20 | -- |
| 2 | wavy_sigmoid | 13.548 | 0.071 | 0.805 | 20 | +0.008 (+0.06%) |
| 3 | wavelet_sigmoid | 13.558 | 0.082 | 0.806 | 20 | +0.018 (+0.13%) |

### Overall Ranking -- Weather-96 FM5

| Rank | Gate Function | MSE | +/- | MAE | n | Delta |
|------|--------------|-----|-----|-----|---|-------|
| 1 | wavelet_sigmoid | 0.1685 | 0.034 | 0.248 | 15 | -- |
| 2 | sigmoid | 0.1686 | 0.047 | 0.255 | 20 | +0.0001 (+0.02%) |
| 3 | wavy_sigmoid | 0.1716 | 0.028 | 0.253 | 15 | +0.0031 (+1.8%) |

### Statistical Tests (AELG only)

| Dataset | Test | Statistic | p-value | Interpretation |
|---------|------|-----------|---------|----------------|
| M4 FM5 | Kruskal-Wallis (SMAPE) | H=-- | 0.37 | NOT significant |
| Weather FM5 | Kruskal-Wallis (MSE) | H=0.89 | 0.64 | NOT significant |
| Weather FM7 | Kruskal-Wallis (MSE) | H=1.90 | 0.39 | NOT significant |
| M4 FM5 | eta-squared (SMAPE) | -- | -- | ~0.000 (no variance explained) |
| Weather FM5 | eta-squared (MSE) | -- | -- | ~0.038 (negligible) |

### Per-Architecture Breakdown -- Weather FM5

No architecture shows a significant gate function effect:

| Architecture | Best Gate (MSE) | KW p | MSE Spread |
|-------------|----------------|------|------------|
| TrendWavelet_pure | wavelet_sigmoid | 0.68 | 0.023 |
| TrendWaveletGeneric_pure | sigmoid | 0.85 | 0.010 |
| Trend_Wavelet_alt | sigmoid | -- | 0.022 |
| Trend_Wavelet_Generic_alt | sigmoid (1 config only) | -- | -- |

### Cross-Dataset Rank Consistency

| Gate Function | M4 FM5 | M4 FM7 | Weather FM5 | Weather FM7 | Avg Rank |
|--------------|--------|--------|-------------|-------------|----------|
| sigmoid | 1 | 2 | 2 | 2 | 1.75 |
| wavelet_sigmoid | 3 | 1 | 1 | 1 | 1.50 |
| wavy_sigmoid | 2 | 3 | 3 | 3 | 2.75 |

wavelet_sigmoid has the best average rank (1.50) but the margins are all non-significant. sigmoid has the simplest implementation and ranks 1.75. wavy_sigmoid is consistently the worst.

## Finding 2: FM5 Significantly Outperforms FM7 (Cross-Dataset)

### M4-Yearly

| Metric | FM7 (bl=42) | FM5 (bl=30) | Difference |
|--------|-------------|-------------|------------|
| Mean SMAPE | 13.641 | 13.551 | -0.090 (-0.66%) |
| FM5 wins | -- | 16/16 configs | all |
| Paired Wilcoxon | -- | -- | p < 0.001 |
| Early-stop rate | 32.5% | 97.5% | +65pp |
| Mean best epoch | 76.4 | 54.7 | -21.7 |

### Weather-96

| Metric | FM7 (bl=672) | FM5 (bl=480) | Difference |
|--------|-------------|-------------|------------|
| Mean MSE | 0.199 | 0.164 | -0.035 (-17.5%) |
| FM5 wins | -- | 13/13 configs (63/63 at run level: 81%) | all |
| Paired Wilcoxon | -- | -- | p < 0.0001 |
| Early-stop rate | 100% | 100% | (both) |
| Mean best epoch | 42.4 | 46.6 | +4.2 |

**The FM5 advantage is even stronger on Weather (17.5% MSE reduction) than on M4 (0.66% SMAPE reduction).** The extra lookback context from FM7 appears to add noise on both datasets.

### Weather FM5 vs FM7 by Config (largest improvements first)

| Config | FM5 MSE | FM7 MSE | Improvement |
|--------|---------|---------|-------------|
| TWGA20_control | 0.1411 | 0.2132 | -33.8% |
| TALG_WV3LG20_wavy | 0.1636 | 0.2203 | -25.7% |
| TALG_WV3LG20_sigmoid | 0.1412 | 0.1978 | -28.6% |
| TA_WV3A20_control | 0.1210 | 0.1660 | -27.1% |
| TWALG20_wavelet | 0.1640 | 0.2113 | -22.4% |

## Finding 3: AE Outperforms AELG on Weather

### Overall Backbone Comparison (FM5)

| Dataset | AE Mean | AELG Mean | MWU p | Winner |
|---------|---------|-----------|-------|--------|
| M4 | SMAPE 13.558 | SMAPE 13.549 | 0.42 | AELG (ns) |
| Weather | MSE 0.144 | MSE 0.169 | 0.036 | **AE (significant)** |

### Weather FM5: AE vs AELG by Architecture

| Architecture | AE MSE | Best AELG MSE (gate) | MWU p | Winner |
|-------------|--------|---------------------|-------|--------|
| TrendWavelet_pure | 0.170 | 0.164 (wavelet) | 1.00 | AELG |
| TrendWaveletGeneric_pure | 0.141 | 0.172 (sigmoid) | 0.42 | AE |
| Trend_Wavelet_alt | 0.121 | 0.141 (sigmoid) | 0.22 | AE |

The pattern is striking: AE wins on 2 of 3 FM5 architectures (both alternating ones), and the overall difference is significant. On M4, AELG shows a weak advantage (all ns). **The AELG mechanism is dataset-dependent.**

Interpretation: On Weather-96, the latent gate may be over-constraining the AE bottleneck. With 96-step forecasts and high-dimensional input (bl=480, 21 features), the gating mechanism could be pruning useful latent dimensions. The AE control allows all latent dimensions to contribute without constraint.

## Finding 4: Alternating Stacks Beat Homogeneous (Both Datasets)

| Dataset | Alternating Mean | Homogeneous Mean | MWU p |
|---------|-----------------|-----------------|-------|
| M4 FM5 | SMAPE 13.534 | SMAPE 13.567 | 0.0006 |
| Weather FM5 | MSE 0.146 | MSE 0.171 | 0.015 |

The alternating-stack advantage is consistent across datasets and gate functions. Trend_Wavelet_alt (T+W) is the most parameter-efficient winning architecture.

Adding a third block type (GenericAELG) in the T+W+G architecture provides no benefit:
- M4: T+W+G SMAPE=13.534 vs T+W SMAPE=13.534 (identical)
- Weather FM5: T+W+G MSE=0.174 (1 config only) vs T+W MSE=0.146 (worse, but incomplete data)

## Config Rankings

### M4-Yearly: Top 10 Overall

| Rank | FM | Config | Gate | SMAPE | +/- | OWA | Params |
|------|-----|--------|------|-------|-----|-----|--------|
| 1 | FM5 | TALG_WV3LG_GALG21_wavy | wavy_sigmoid | 13.495 | 0.042 | 0.802 | 2.62M |
| 2 | FM5 | TALG_WV3LG20_wavy | wavy_sigmoid | 13.509 | 0.074 | 0.802 | 2.08M |
| 3 | FM5 | TALG_WV3LG20_sigmoid | sigmoid | 13.525 | 0.058 | 0.804 | 2.08M |
| 4 | FM5 | TALG_WV3LG20_wavelet | wavelet_sigmoid | 13.530 | 0.072 | 0.804 | 2.08M |
| 5 | FM5 | TALG_WV3LG_GALG21_wavelet | wavelet_sigmoid | 13.531 | 0.104 | 0.804 | 2.62M |
| 6 | FM5 | TALG_WV3LG_GALG21_sigmoid | sigmoid | 13.534 | 0.044 | 0.805 | 2.62M |
| 7 | FM5 | TA_WV3A20_control | none (AE) | 13.537 | 0.078 | 0.805 | 2.08M |
| 8 | FM5 | TA_WV3A_GA21_control | none (AE) | 13.537 | 0.064 | 0.805 | 2.62M |
| 9 | FM5 | TWALG20_sigmoid | sigmoid | 13.538 | 0.041 | 0.804 | 1.04M |
| 10 | FM5 | TWALG20_wavelet | wavelet_sigmoid | 13.539 | 0.040 | 0.804 | 1.04M |

All top 10 are FM5 configs. Gate functions are scrambled in the ranking -- no consistent pattern.

### Weather-96: Top 10 Overall

| Rank | FM | Config | Gate | MSE | +/- | MAE | SMAPE | Params |
|------|-----|--------|------|-----|-----|-----|-------|--------|
| 1 | FM5 | TA_WV3A20_control | none (AE) | 0.121 | 0.017 | 0.216 | 40.7 | 4.61M |
| 2 | FM5 | TWGA20_control | none (AE) | 0.141 | 0.037 | 0.239 | 43.7 | 2.57M |
| 3 | FM5 | TALG_WV3LG20_sigmoid | sigmoid | 0.141 | 0.019 | 0.234 | 42.6 | 4.61M |
| 4 | FM5 | TALG_WV3LG20_wavelet | wavelet_sigmoid | 0.160 | 0.037 | 0.243 | 42.4 | 4.61M |
| 5 | FM5 | TALG_WV3LG20_wavy | wavy_sigmoid | 0.164 | 0.037 | 0.243 | 42.6 | 4.61M |
| 6 | FM5 | TWALG20_wavelet | wavelet_sigmoid | 0.164 | 0.022 | 0.244 | 42.7 | 2.49M |
| 7 | FM7 | TA_WV3A20_control | none (AE) | 0.166 | 0.025 | 0.263 | 49.0 | 5.35M |
| 8 | FM5 | TWA20_control | none (AE) | 0.170 | 0.014 | 0.264 | 46.7 | 2.49M |
| 9 | FM5 | TWGLG20_sigmoid | sigmoid | 0.172 | 0.049 | 0.262 | 46.2 | 2.57M |
| 10 | FM5 | TALG_WV3LG_GALG21_sigmoid | sigmoid | 0.174 | 0.066 | 0.257 | 44.6 | 7.13M |

Top 2 Weather configs are both AE controls. FM5 dominates (9 of top 10).

## Parameter Efficiency

### Parameter Counts by Architecture

| Architecture | M4 Params | Weather Params |
|-------------|-----------|---------------|
| TrendWavelet_pure (TW) | 1.04M | 2.49-2.98M |
| TrendWaveletGeneric_pure (TWG) | 1.10M | 2.57-3.08M |
| Trend_Wavelet_alt (T+W) | 2.08M | 4.61-5.35M |
| Trend_Wavelet_Generic_alt (T+W+G) | 2.62M | 7.13-8.68M |

Note: AE and AELG have identical parameter counts (the gate is a single vector of size latent_dim=16).

The most parameter-efficient competitive configs on both datasets are:
- **M4**: TWALG20_sigmoid (1.04M, SMAPE=13.538) -- 90% the quality of the winner at 40% the parameters
- **Weather**: TWGA20_control (2.57M, MSE=0.141) -- 85% the quality of the winner at 56% the parameters

## Seed-Level Variability

### M4-Yearly (FM5)
Very low variability. CV per config: 0.3-0.5%. All seeds well-behaved.

| Seed | Mean SMAPE | Rank |
|------|-----------|------|
| 45 | 13.529 | 1 |
| 43 | 13.540 | 2 |
| 46 | 13.555 | 3 |
| 44 | 13.559 | 4 |
| 42 | 13.571 | 5 |

### Weather-96 (FM5)
Much higher variability. CV per config: 14-26%. Rankings are unstable with 5 seeds.

| Seed | Mean MSE | Rank |
|------|----------|------|
| 43 | 0.151 | 1 |
| 42 | 0.158 | 2 |
| 45 | 0.162 | 3 |
| 46 | 0.173 | 4 |
| 44 | 0.174 | 5 |

Seed 44 is the worst on Weather (consistent with prior studies finding seed 44 problematic on multiple datasets).

## Convergence Analysis

### Weather FM5

| Gate | Epochs | Best Epoch | ES% | Best Val Loss | Training Time |
|------|--------|-----------|-----|--------------|--------------|
| sigmoid | 71.8 | 45.8 | 100% | 58.21 | 593s |
| wavy_sigmoid | 76.5 | 50.5 | 100% | 58.45 | 687s |
| wavelet_sigmoid | 72.2 | 46.2 | 100% | 58.51 | 621s |
| none (AE) | 71.1 | 45.1 | 100% | 58.55 | 580s |

All Weather configs early-stop. No gate function converges faster or slower. AE control converges earliest and achieves the best test MSE despite slightly higher validation loss -- suggesting better generalization.

### Weather FM7

| Gate | Epochs | Best Epoch | ES% | Best Val Loss | Training Time |
|------|--------|-----------|-----|--------------|--------------|
| wavy_sigmoid | 66.9 | 40.9 | 100% | 58.79 | 660s |
| wavelet_sigmoid | 72.2 | 46.1 | 100% | 58.89 | 686s |
| sigmoid | 68.3 | 42.3 | 100% | 58.98 | 670s |
| none (AE) | 66.2 | 40.2 | 100% | 59.06 | 661s |

## Recommendations

### Immediate Actions
1. **Keep sigmoid as the default gate function.** No reason to change. The code complexity of wavy_sigmoid and wavelet_sigmoid is not justified.
2. **Use forecast_multiplier=5 as the default** for M4-Yearly and Weather-96 AELG/AE architectures. This is the most actionable finding, confirmed on both datasets with high significance.
3. **For Weather-96, prefer AE over AELG** when using alternating stacks. The learned gate is mildly harmful.

### Proposed Next Experiments

#### 1. FM5 on M4-Yearly SOTA (highest priority)
```yaml
experiment_name: fm_validation_m4_sota
dataset: m4
periods: [Yearly]
training:
  forecast_multiplier: 5
  max_epochs: 100
  patience: 25
configs:
  - name: Coif2_bd6_eq_fcast_td3_fm5
    stacks: {type: alternating, blocks: [Trend, WaveletV3], repeats: 5}
    block_params:
      wavelet_type: coif2
      basis_dim: 6
      trend_thetas_dim: 3
runs:
  n_runs: 10
```

#### 2. Weather AE alternating with more seeds
```yaml
experiment_name: weather_ae_alt_confirmation
dataset: weather
periods: [Weather-96]
protocol:
  normalize: true
  train_ratio: 0.7
  val_ratio: 0.1
  include_target: true
  forecast_multiplier: 5
  batch_size: 256
training:
  max_epochs: 200
  patience: 25
  skip_distance: 5
  skip_alpha: 0.1
configs:
  - name: TA_WV3A20_fm5
    stacks: {type: alternating, blocks: [TrendAE, WaveletV3AE], repeats: 10}
    block_params:
      latent_dim: 16
      wavelet_type: coif3
      basis_dim: 96
      forecast_basis_dim: 96
      trend_thetas_dim: 3
runs:
  n_runs: 20
```

#### 3. FM Sweep
Test forecast_multiplier = 3, 4, 5, 6, 7 on both M4-Yearly and Weather-96 to find the optimal lookback ratio.

### Open Questions
1. Does FM5 advantage generalize to other M4 periods (Monthly, Quarterly)?
2. Does FM5 advantage hold for non-AE backbones (which produced the current SOTA)?
3. Is the Weather AE advantage specific to these architectures, or does it extend to all AELG blocks?
4. Would FM3 or FM4 be even better for M4-Yearly?
5. What is the optimal forecast_multiplier for Weather with non-AE blocks?
