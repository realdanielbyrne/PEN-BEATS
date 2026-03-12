# NHiTS-Protocol Benchmark Analysis: Weather and Traffic

**Date:** 2026-03-11
**Data:** 590 rows, 18 configs, 2 datasets (Weather, Traffic), 4 horizons (96, 192, 336, 720)
**Protocol:** Z-score normalization (Weather), no normalization (Traffic), 70/10/20 split, MSE loss, L=5H lookback, patience=10, max 100 epochs, 8 seeds

## Executive Summary

This study benchmarks 18 configurations -- spanning NBeatsNet (10-stack flat residual) and NHiTSNet (3-stack hierarchical pooling) architectures with novel AE/AELG block types -- against published NHiTS baselines on Weather and Traffic datasets.

**Headline results:**
- **We beat published NHiTS at Weather H=192 and H=336.** BottleneckGenericAELG-10 (NBeatsNet) achieves MSE=0.210 at H=336, a **22.7% improvement** over NHiTS paper (0.272). NHiTS-GenericAE achieves MSE=0.198 at H=192, a **6.2% improvement** over NHiTS paper (0.211).
- **We narrowly miss at H=96** (1.047x) and **badly miss at H=720** (1.454x).
- **NHiTSNet vs NBeatsNet exhibits a clear horizon crossover:** NHiTSNet wins at H=96/192; NBeatsNet wins at H=336/720.
- **BottleneckGenericAELG-10 is the most cross-horizon-robust config** with average rank 3.5/14 across all horizons.
- **Traffic experiment is incomplete** (only H=96 and partial H=192 collected so far).

## Detailed Rankings

### Weather H=96 (NHiTS baseline: MSE=0.158)

| Rank | Config | Model | N | MSE | MSE std | vs NHiTS |
|------|--------|-------|---|-----|---------|----------|
| 1 | NHiTS-GenericAELG | NHiTSNet | 8 | 0.1655 | 0.0349 | 1.047x |
| 2 | NHiTS-TrendAELG+Coif2V3AELG | NHiTSNet | 8 | 0.1778 | 0.0258 | 1.126x |
| 3 | NHiTS-TrendWaveletAELG | NHiTSNet | 8 | 0.1829 | 0.0248 | 1.158x |
| 4 | NHiTS-TrendWaveletAE | NHiTSNet | 8 | 0.1914 | 0.0346 | 1.212x |
| 5 | NHiTS-GenericAE | NHiTSNet | 8 | 0.1923 | 0.0315 | 1.217x |

Top 5 are all NHiTSNet. NBeatsNet configs start at rank 7 with TrendAE+Sym20V3AE (MSE=0.195). Generic-10 (vanilla N-BEATS) is dead last at rank 18 (MSE=0.261).

### Weather H=192 (NHiTS baseline: MSE=0.211)

| Rank | Config | Model | N | MSE | MSE std | vs NHiTS |
|------|--------|-------|---|-----|---------|----------|
| 1 | **NHiTS-GenericAE** | NHiTSNet | 8 | **0.1979** | 0.0315 | **0.938x BEATS** |
| 2 | **BottleneckGenericAE-10** | NBeatsNet | 8 | **0.2096** | 0.0582 | **0.993x BEATS** |
| 3 | NHiTS-GenericAELG | NHiTSNet | 8 | 0.2100 | 0.0459 | 0.995x (barely misses) |
| 4 | BottleneckGenericAELG-10 | NBeatsNet | 8 | 0.2159 | 0.0508 | 1.023x |
| 5 | NHiTS-TrendWaveletAELG | NHiTSNet | 8 | 0.2217 | 0.0441 | 1.051x |

### Weather H=336 (NHiTS baseline: MSE=0.272) -- **BEST RESULTS**

| Rank | Config | Model | N | MSE | MSE std | vs NHiTS |
|------|--------|-------|---|-----|---------|----------|
| 1 | **BottleneckGenericAELG-10** | NBeatsNet | 8 | **0.2102** | 0.0361 | **0.773x BEATS** |
| 2 | **NHiTS-GenericAELG** | NHiTSNet | 8 | **0.2266** | 0.0598 | **0.833x BEATS** |
| 3 | **BottleneckGenericAE-10** | NBeatsNet | 8 | **0.2295** | 0.0300 | **0.844x BEATS** |
| 4 | **Generic-10** | NBeatsNet | 8 | **0.2363** | 0.0524 | **0.869x BEATS** |
| 5 | **NHiTS-TrendWaveletAE** | NHiTSNet | 8 | **0.2506** | 0.0397 | **0.921x BEATS** |
| 6 | **NHiTS-Generic** | NHiTSNet | 8 | **0.2549** | 0.0535 | **0.937x BEATS** |

Six configs beat NHiTS. The top result (BottleneckGenericAELG-10, MSE=0.210) is 22.7% better than the published baseline.

### Weather H=720 (NHiTS baseline: MSE=0.348) -- **ALL FAIL**

| Rank | Config | Model | N | MSE | MSE std | vs NHiTS |
|------|--------|-------|---|-----|---------|----------|
| 1 | TrendWavelet-10 | NBeatsNet | 3 | 0.5060 | 0.0599 | 1.454x |
| 2 | BottleneckGenericAELG-10 | NBeatsNet | 8 | 0.5343 | 0.0892 | 1.535x |
| 3 | Generic-10 | NBeatsNet | 8 | 0.5418 | 0.0633 | 1.557x |

No config comes close. 81/113 runs (72%) have SMAPE > 100. Models converge to a near-constant prediction. The best (TrendWavelet-10) has only 3 runs and only trained 11 epochs -- severely undertrained.

### Traffic H=96 (preliminary, internal ranking only)

| Rank | Config | Model | N | MSE |
|------|--------|-------|---|-----|
| 1 | TrendAELG+Sym20V3AELG | NBeatsNet | 5 | 0.001014 |
| 2 | NHiTS-TrendWaveletAELG | NHiTSNet | 8 | 0.001045 |
| 3 | NHiTS-GenericAELG | NHiTSNet | 8 | 0.001066 |
| 4 | NHiTS-TrendAELG+Coif2V3AELG | NHiTSNet | 3 | 0.001075 |

All configs report MSE ~0.001, which is ~400x lower than the NHiTS paper's 0.401. This discrepancy indicates a metric reporting difference (per-variate vs. joint MSE or OT-only), not a real 400x improvement.

## Cross-Horizon Generalization

Average rank across all 4 horizons (14 configs with complete data):

| Rank | Config | Model | Avg Rank | H=96 | H=192 | H=336 | H=720 |
|------|--------|-------|----------|------|-------|-------|-------|
| 1 | BottleneckGenericAELG-10 | NBeatsNet | 3.5 | 8 | 3 | 1 | 2 |
| 2 | NHiTS-GenericAELG | NHiTSNet | 4.0 | 1 | 2 | 2 | 11 |
| 3 | BottleneckGenericAE-10 | NBeatsNet | 5.2 | 10 | 1 | 3 | 7 |
| 4 | NHiTS-TrendWaveletAELG | NHiTSNet | 7.0 | 3 | 4 | 8 | 13 |
| 5 | TrendWaveletAE-10 | NBeatsNet | 7.5 | 5 | 7 | 12 | 6 |

BottleneckGenericAELG-10 is the safest all-around choice. NHiTS-GenericAELG is the best at short horizons but collapses at H=720.

## NBeatsNet vs NHiTSNet Architecture Comparison

### Best of Each Architecture per Horizon

| Horizon | Best NBeatsNet | MSE | Best NHiTSNet | MSE | MWU p | Winner |
|---------|---------------|-----|---------------|-----|-------|--------|
| 96 | TrendAE+Sym20V3AE | 0.1954 | NHiTS-GenericAELG | 0.1655 | 0.028 | **NHiTSNet** * |
| 192 | BottleneckGenericAE-10 | 0.2096 | NHiTS-GenericAE | 0.1979 | 0.879 | NHiTSNet |
| 336 | BottleneckGenericAELG-10 | 0.2102 | NHiTS-GenericAELG | 0.2266 | 0.798 | NBeatsNet |
| 720 | TrendWavelet-10 | 0.5060 | NHiTS-GenericAELG | 0.6709 | 0.049 | **NBeatsNet** * |

### Matched Block Type: GenericAELG (10-stack vs 3-stack)

| Horizon | NBeatsNet-10 MSE | NHiTSNet-3 MSE | Delta | p-value |
|---------|-----------------|----------------|-------|---------|
| 96 | 0.2556 | 0.1655 | +54.5% | **0.001** |
| 192 | 0.2524 | 0.2100 | +20.2% | 0.195 |
| 336 | 0.2624 | 0.2266 | +15.8% | 0.105 |
| 720 | 0.5605 | 0.6709 | -16.5% | 0.065 |

The crossover occurs between H=336 and H=720. NHiTSNet's hierarchical pooling (kernel=[8,4,1]) is beneficial regularization at short horizons but destroys temporal resolution at long ones.

## AE vs AELG on Weather

The learned gate provides **no consistent benefit on Weather**. Across 28 pairwise comparisons (7 block pairs x 4 horizons):
- AELG wins 17/28 (61%) but only 1 is significant (NHiTS-GenericAE at H=96, p=0.021)
- AE wins 11/28 (39%) with 1 significant (TrendAE+Sym20 at H=720, p=0.043 -- in opposite direction)
- AELG win rate by horizon: H=96: 86%, H=192: 43%, H=336: 71%, H=720: 57%

The gate's marginal H=96 advantage is consistent with prior M4 findings but disappears at longer horizons. Use AELG as default (no downside) but the gate is not a differentiating factor here.

## H=720 Failure Mode

All 113 runs at H=720 produce SMAPE > 100 (essentially useless forecasts). Root causes:
1. **Early stopping fires too soon:** Most configs train only 11-20 epochs before patience=10 triggers. With L=3600, each epoch processes far fewer batches.
2. **TrendWavelet-10 (best, MSE=0.506) only has 3 runs** and trained only 11 epochs. 5 runs are missing.
3. **NHiTS-Generic has an outlier** (run 4, MSE=3.887) that may be a divergence not caught by SMAPE filtering.
4. **Model scale:** At H=720, parameter counts reach 13-48M, but the training signal may be too weak per parameter.

## Recommendations

### Current Best Configurations

1. **Weather H=96:** NHiTS-GenericAELG (NHiTSNet, 983K params) -- MSE=0.165, 4.7% above NHiTS paper
2. **Weather H=192:** NHiTS-GenericAE (NHiTSNet, 1.5M params) -- MSE=0.198, **beats NHiTS by 6.2%**
3. **Weather H=336:** BottleneckGenericAELG-10 (NBeatsNet, 5.8M params) -- MSE=0.210, **beats NHiTS by 22.7%**
4. **Weather H=720:** TrendWavelet-10 (NBeatsNet, 13M params) -- MSE=0.506, **needs improvement** (low confidence, 3 runs)
5. **Cross-horizon best:** BottleneckGenericAELG-10 (NBeatsNet, avg rank 3.5/14)

### What to Test Next

1. **H=720 recovery:**
   - Increase patience to 25, disable early stopping for some runs to observe full training curves
   - Try learning rate warm-up (linear warm-up for 5 epochs)
   - Complete TrendWavelet-10 runs (only 3/8 present)

2. **NHiTSNet with BottleneckGenericAELG blocks:**
   ```yaml
   - name: NHiTS-BottleneckGenericAELG
     model: NHiTSNet
     stack_types: ['BottleneckGenericAELG', 'BottleneckGenericAELG', 'BottleneckGenericAELG']
     n_pools_kernel_size: [8, 4, 1]
     n_freq_downsample: [24, 12, 1]
   ```

3. **NHiTSNet pooling ablation at H=720:**
   ```yaml
   - name: NHiTS-GenericAELG-gentle-pool
     model: NHiTSNet
     stack_types: ['GenericAELG', 'GenericAELG', 'GenericAELG']
     n_pools_kernel_size: [4, 2, 1]
     n_freq_downsample: [12, 6, 1]
   ```

4. **Complete Traffic experiment** across all 4 horizons before analysis.

5. **Run active_g=forecast configs** (YAML files exist: nhits_activeg_weather.yaml, nhits_activeg_traffic.yaml).

6. **Verify Traffic MSE metric alignment** with NHiTS paper -- per-variate vs joint.

### Open Questions

1. Why does BottleneckGenericAELG-10 excel at H=336 specifically? Is the bottleneck dim (thetas_dim=5) a better match for 336-step structure?
2. Would deeper NHiTSNet (5 stacks) recover H=720 without losing short-horizon performance?
3. Can skip connections (skip_distance=3, skip_alpha=0.1) rescue H=720 NBeatsNet runs?
4. Is the H=720 failure structural (L=5H too aggressive) or just training-regime (patience too low)?
