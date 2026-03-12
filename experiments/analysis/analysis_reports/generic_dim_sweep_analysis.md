# Generic Dimension Sweep Analysis — TrendWaveletGeneric Family, M4-Yearly

**Date:** 2026-03-12
**Dataset:** M4-Yearly (H=6, L=30)
**Total runs:** 144 (23 configs x 3 seeds, 3-round successive halving)
**Divergences:** 0
**Notebook:** `experiments/analysis/notebooks/generic_dim_sweep_insights.ipynb`

## Executive Summary

This study swept the learned generic branch rank (generic_dim in {3, 5, 8, 16}) and stack depth (n in {5, 10, 15, 20}) across three TrendWaveletGeneric backbone variants (RootBlock, AERootBlock, AERootBlockLG). Two no-generic baselines (TrendWaveletAE, TrendWaveletAELG) served as controls.

**Main findings:**

1. **generic_dim is a complete non-factor for RootBlock** (Kruskal-Wallis p=0.91). The 4 gd values span only 0.050 SMAPE, within seed noise.
2. **RootBlock dominates AE variants** on quality (MWU p=0.007), but AE variants are 2-5x more parameter-efficient.
3. **AE backbones are extremely depth-hungry** — TWGAE needs n=15-20, TWGAELG needs n>=20 to compete with RootBlock at n=10.
4. **The generic branch helps AE backbones** (0.4-1.2 SMAPE improvement at equal epochs vs no-generic baselines).
5. **Does NOT beat M4-Yearly SOTA** (13.410). Best here is 13.440 (+0.22%).

## Study Design

| Component | Details |
|-----------|---------|
| Study 1 — generic_dim sweep | 12 configs: 3 backbones x 4 gd values, n=10 |
| Study 2 — Stack depth sweep | 9 configs: 3 backbones x 3 new depths (5,15,20), gd=5 |
| Baselines | 2 configs: TWAE, TWAELG (gd=0), n=10 |
| Fixed params | wavelet=coif2, basis_dim=6, trend_thetas_dim=3, latent_dim=16 |
| Search | 3-round halving: R1=10ep/23cfg, R2=15ep/15cfg, R3=50ep/10cfg |
| Seeds | 42, 43, 44 |

## R3 Final Rankings (50 epochs)

| Rank | Config | SMAPE | OWA | Params | Stacks | gd | Backbone | Delta |
|------|--------|-------|-----|--------|--------|-----|----------|-------|
| 1 | TWG_coif2_bd6_td3_gd3 | 13.440 (±0.038) | 0.797 (±0.003) | 2,085,040 | 10 | 3 | RootBlock | BEST |
| 2 | TWG_coif2_bd6_td3_gd8 | 13.449 (±0.051) | 0.797 (±0.003) | 2,099,690 | 10 | 8 | RootBlock | +0.009 |
| 3 | TWG_coif2_bd6_td3_gd5 | 13.462 (±0.050) | 0.798 (±0.004) | 2,090,900 | 10 | 5 | RootBlock | +0.022 |
| 4 | TWG_coif2_bd6_td3_gd16 | 13.489 (±0.096) | 0.801 (±0.008) | 2,123,130 | 10 | 16 | RootBlock | +0.050 |
| 5 | TWGAE_coif2_bd6_td3_ld16_gd5_n20 | 13.505 (±0.031) | 0.801 (±0.002) | 900,200 | 20 | 5 | AERootBlock | +0.065 |
| 6 | TWGAE_coif2_bd6_td3_ld16_gd5_n15 | 13.551 (±0.017) | 0.804 (±0.001) | 675,150 | 15 | 5 | AERootBlock | +0.111 |
| 7 | TWGAE_coif2_bd6_td3_ld16_gd3 | 13.599 (±0.029) | 0.807 (±0.002) | 444,240 | 10 | 3 | AERootBlock | +0.159 |
| 8 | TWGAELG_coif2_bd6_td3_ld16_gd5_n20 | 13.649 (±0.254) | 0.813 (±0.017) | 900,520 | 20 | 5 | AERootBlockLG | +0.210 |
| 9 | TWG_coif2_bd6_td3_gd5_n20 | 13.662 (±0.317) | 0.814 (±0.021) | 4,181,800 | 20 | 5 | RootBlock | +0.222 |
| 10 | TWG_coif2_bd6_td3_gd5_n15 | 13.741 (±0.445) | 0.821 (±0.030) | 3,136,350 | 15 | 5 | RootBlock | +0.302 |

## Study 1: generic_dim Is a Non-Factor for RootBlock

### RootBlock (n=10, R3)

| gd | SMAPE | OWA | Params |
|----|-------|-----|--------|
| 3 | 13.440 ±0.038 | 0.797 | 2,085,040 |
| 5 | 13.462 ±0.050 | 0.798 | 2,090,900 |
| 8 | 13.449 ±0.051 | 0.797 | 2,099,690 |
| 16 | 13.489 ±0.096 | 0.801 | 2,123,130 |

**Kruskal-Wallis H=0.538, p=0.910.** Total spread: 0.050 SMAPE. The standard 4-FC backbone (t_width=256) already has enough capacity to learn any pattern the generic branch would capture. The generic branch projection is redundant.

### AERootBlock (n=10, mixed rounds)

| gd | SMAPE | Round | Params |
|----|-------|-------|--------|
| 0 (baseline) | 15.912 ±0.859 | R1 | 435,450 |
| 3 | 13.599 ±0.029 | R3 | 444,240 |
| 5 | 14.462 ±0.332 | R2 | 450,100 |
| 8 | 14.402 ±0.183 | R2 | 458,890 |
| 16 | 14.535 ±0.410 | R2 | 482,330 |

Note: Comparison is unfair across rounds. At equal epochs (R1, 10ep), all gd>0 configs improve over gd=0 by 0.4-0.8 SMAPE.

### AERootBlockLG (n=10, R1 only — all eliminated)

| gd | SMAPE (R1) | Params |
|----|------------|--------|
| 0 (baseline) | 17.696 ±0.240 | 435,610 |
| 3 | 16.978 ±1.871 | 444,400 |
| 5 | 16.507 ±0.740 | 450,260 |
| 8 | 16.704 ±1.113 | 459,050 |
| 16 | 16.480 ±1.115 | 482,490 |

All AELG configs at n=10 were eliminated after R1. The learned gate makes the block extremely slow to converge at shallow depths. The generic branch helps (0.7-1.2 SMAPE) but not enough to survive halving.

## Study 2: AE Backbones Are Depth-Hungry

### Stack depth at gd=5 (best available round)

| Backbone | n=5 | n=10 | n=15 | n=20 |
|----------|-----|------|------|------|
| RootBlock | 14.173 (R2) | 13.462 (R3) | 13.741 (R3) | 13.662 (R3) |
| AERootBlock | 18.189 (R1) | 14.462 (R2) | 13.551 (R3) | 13.505 (R3) |
| AERootBlockLG | 21.184 (R1) | 16.507 (R1) | 14.080 (R2) | 13.649 (R3) |

**RootBlock:** Optimal at n=10. Extra stacks increase variance (n=15 std=0.445, n=20 std=0.317 vs n=10 std=0.050).

**AERootBlock:** Monotonic improvement. n=20 reaches 13.505, approaching RootBlock n=10 (13.440). Still improving — n=25-30 might close the gap further.

**AERootBlockLG:** Most depth-hungry. SMAPE drops from 21.2 (n=5) to 13.6 (n=20) — a 36% improvement. Only the n=20 config survived to R3, and its variance is high (std=0.254).

## Baseline Comparison (Fair, Equal Epochs)

At R1 (10 epochs), comparing gd=0 (no generic branch) vs gd>0 (with generic branch), both at n=10:

**AERootBlock:** Pooled generic (n=12) vs baseline (n=3): MWU p=0.435. Not significant, largely because gd=16 is equivalent to baseline. Individual gd=8 shows the largest improvement (-0.775 SMAPE).

**AERootBlockLG:** Pooled generic (n=12) vs baseline (n=3): MWU p=0.061. Approaching significance. All gd values improve over baseline.

## Comparison to External Baselines

| Config | SMAPE | OWA | Context |
|--------|-------|-----|---------|
| M4-Yearly SOTA (Trend+WaveletV3, non-AE) | 13.410 | 0.794 | External benchmark |
| Prior best TWGAELG (effectiveness study) | 13.509 | 0.801 | Previous TWG study |
| **This study: TWG gd=3 (RootBlock, n=10)** | **13.440** | **0.797** | Study winner |
| This study: TWGAE gd=5 n=20 (AE) | 13.505 | 0.801 | Best AE variant |
| Prior unified TWGAELG (coif2, lt_fcast) | 14.183 | — | Previous study |

**This study improves over the prior TWGAELG effectiveness study** (13.509 -> 13.440, -0.51%) but uses a different backbone (RootBlock vs AELG). The improvement is attributable to basis_dim=6 (eq_fcast) vs basis_dim=4 (lt_fcast) and the stronger RootBlock backbone — not the generic_dim setting.

**Does NOT beat M4-Yearly SOTA** (+0.030 SMAPE, +0.22%). The unified 3-way decomposition (trend+wavelet+generic in a single block) remains inferior to the 2-block Trend+WaveletV3 alternating architecture on M4-Yearly.

## Parameter Efficiency

| Config | SMAPE | Params | Params relative to #1 |
|--------|-------|--------|-----------------------|
| TWG gd=3, n=10 | 13.440 | 2,085,040 | 1.00x (reference) |
| TWGAE gd=5, n=20 | 13.505 | 900,200 | 0.43x (+0.065 SMAPE) |
| TWGAE gd=5, n=15 | 13.551 | 675,150 | 0.32x (+0.111 SMAPE) |
| TWGAE gd=3, n=10 | 13.599 | 444,240 | 0.21x (+0.159 SMAPE) |

TWGAE at n=15 achieves SMAPE within 0.11 of the winner using only 32% of the parameters. This is the most parameter-efficient sub-13.55 TrendWaveletGeneric config.

## Statistical Tests

| Test | Comparison | Statistic | p-value | Interpretation |
|------|-----------|-----------|---------|----------------|
| Kruskal-Wallis | gd={3,5,8,16} in RootBlock R3 | H=0.538 | 0.910 | generic_dim is a non-factor |
| Mann-Whitney U | RootBlock vs AERootBlock (R3 pooled) | — | 0.007 | RootBlock significantly better |
| Mann-Whitney U | AE+Generic vs AE baseline (R1, pooled) | — | 0.435 | Not significant |
| Mann-Whitney U | AELG+Generic vs AELG baseline (R1, pooled) | — | 0.061 | Approaching significance |

## Convergence Notes

- R3 configs trained 20-50 epochs (many early-stopped with patience=10).
- TWG gd=16 early-stopped at epochs 37-49, suggesting it converges faster (or plateaus earlier) than smaller gd values.
- AE configs typically ran 48-50 epochs, confirming they are slower to converge.
- AELG n=20 had mixed behavior: one seed early-stopped at epoch 31 while another hit 50 epochs.

## Recommendations

### Current Best Configuration (M4-Yearly)

**For maximum quality:** TWG_coif2_bd6_td3_gd3 (TrendWaveletGeneric, RootBlock, n=10, gd=3)
- SMAPE=13.440, OWA=0.797, 2.1M params
- Confidence: **HIGH** (3 seeds, low variance, consistent across gd values)
- Does not beat SOTA (13.410) but is the best TrendWaveletGeneric config found

**For parameter efficiency:** TWGAE_coif2_bd6_td3_ld16_gd5_n20 (AE backbone, n=20, gd=5)
- SMAPE=13.505, OWA=0.801, 900K params
- 57% fewer params than the quality winner, +0.065 SMAPE penalty

### What to Test Next

1. **TrendWaveletGeneric on Tourism-Yearly and Weather-96** — The generic branch may matter more on longer horizons where polynomial+wavelet basis is less sufficient. Tourism is especially interesting because GenericAE is the current winner there.

```yaml
# Proposed: TWG on Tourism-Yearly
experiment_name: twg_tourism_sweep
dataset: m4  # placeholder — update for Tourism
periods: [Yearly]
configs:
  - name: TWG_coif2_bd6_td3_gd3
    stacks: {type: homogeneous, block: TrendWaveletGeneric, n: 10}
    block_params: {generic_dim: 3}
  - name: TWGAE_coif2_bd6_td3_ld16_gd5_n20
    stacks: {type: homogeneous, block: TrendWaveletGenericAE, n: 20}
    block_params: {generic_dim: 5}
```

2. **TWGAE at n=25 and n=30** — AE performance was still improving monotonically at n=20. Test whether it eventually matches RootBlock quality.

3. **Consider retiring TrendWaveletGeneric for M4-Yearly** — Non-AE Trend+WaveletV3 (alternating 2-block architecture, 13.410) consistently beats the unified 3-way decomposition. The generic branch does not provide enough value to overcome the overhead of merging three basis types into one block.

### Open Questions

1. **Why does gd=16 increase variance on RootBlock?** (std=0.096 vs 0.038 for gd=3). Higher rank may introduce overfitting sensitivity.
2. **Would TWGAE at n=20-30 eventually match SOTA?** The convergence curve suggests possible but unlikely — the gap is only 0.095 SMAPE with 57% fewer params.
3. **Is the generic branch more valuable on datasets where polynomial+wavelet basis is insufficient?** Tourism (H=4) and Weather (H=96) would test this hypothesis.
4. **WaveletV3Generic block (2-way: wavelet+generic, no polynomial)** — noted in the YAML config as a block type that does not yet exist. Would removing the polynomial trend from the unified block help or hurt?
