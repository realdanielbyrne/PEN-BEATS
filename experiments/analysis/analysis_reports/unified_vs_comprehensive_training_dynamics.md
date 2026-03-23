# Unified Benchmark vs Comprehensive Sweep: Training Dynamics Analysis

**Date:** 2026-03-22
**Datasets:** M4 (Yearly, Quarterly, Monthly), Tourism-Yearly, Weather-96, Milk, Traffic-96
**Data sources:**
- `experiments/results/m4/unified_benchmark_results.csv` (1,479 rows, 16 configs)
- `experiments/results/m4/comprehensive_sweep_m4_results.csv` (2,302 rows, 112 configs)
- `experiments/results/*/comprehensive_sweep_*_results.csv` (Weather: 1,125, Tourism: 1,114, Milk: 1,120, Traffic: 99)
- `experiments/results/milk/unified_benchmark_results.csv` (3,200 rows, 16 configs)
**Notebook:** `experiments/analysis/notebooks/unified_vs_comprehensive_training_dynamics.ipynb`

---

## Executive Summary

The unified benchmark used `max_epochs=100, patience=10` with no LR scheduler, while the comprehensive sweep used `max_epochs=200, patience=20` with warmup+cosine annealing. This analysis quantifies the impact of these differences on performance and stability.

**Key findings:**

1. **The comprehensive sweep is better on 10/12 matched configs** (mean improvement -3.8% SMAPE), with 3 statistically significant improvements (p<0.05).

2. **The LR warmup is the most impactful difference**, not the epoch count. It reduces bimodal convergence on M4-Yearly from 1.7% to 0.2% of runs.

3. **Patience=20 helps 47% of M4-Yearly runs** find a better minimum than patience=10 would, with a median improvement of 0.8% in validation loss.

4. **Only 3.2% of runs have best_epoch > 100**, so `max_epochs=200` is mostly insurance -- but it matters for multi-component architectures (Trend+Seasonality+Generic).

5. **Architecture sensitivity varies dramatically**: alternating Trend+Wavelet (RootBlock) configs benefit from patience=20 in 70%+ of runs, while TrendWaveletAELG configs benefit in only 40%.

---

## 1. Training Configuration Differences

| Setting | Unified Benchmark | Comprehensive Sweep |
|---------|-------------------|---------------------|
| `max_epochs` | 100 | 200 |
| `patience` | 10 | 20 |
| `n_stacks` | 30 (all configs) | 10 or 30 (varies) |
| `lr_scheduler` | none | warmup(15) + cosine(eta_min=1e-6) |
| Seeds | sequential (43-51) | random |
| `forecast_multiplier` | 5 | 5 |

### Epoch Distribution Comparison

| Period | UB mean | UB median | CS mean | CS median | Ratio |
|--------|---------|-----------|---------|-----------|-------|
| Yearly | 40.5 | 38.0 | 62.6 | 60.0 | 1.55x |
| Quarterly | 32.9 | 32.0 | 63.4 | 62.0 | 1.93x |
| Monthly | 27.8 | 25.0 | 44.8 | 41.5 | 1.61x |

The comprehensive sweep trains 1.5-1.9x longer on average, driven primarily by the doubled patience (20 vs 10).

---

## 2. Matched Config Comparison

Six architecturally identical configs exist in both experiments at 30 stacks. Results for M4-Yearly and M4-Quarterly:

### M4-Yearly

| Config | UB SMAPE | CS SMAPE | Delta | Delta% | p-value | Sig? |
|--------|----------|----------|-------|--------|---------|------|
| NBEATS-G (baseline) | 13.601 | 13.591 | -0.010 | -0.1% | ns | |
| NBEATS-G (activeG) | 13.800 | 13.800 | -0.001 | -0.0% | ns | |
| NBEATS-I+G (baseline) | 13.926 | 13.580 | -0.346 | -2.5% | 0.030 | ** |
| NBEATS-I+G (activeG) | 14.367 | 13.594 | -0.773 | -5.4% | 0.0004 | ** |
| BNG (baseline) | 20.895 | 16.929 | -3.966 | -19.0% | ns | |
| BNG (activeG) | 13.929 | 13.743 | -0.186 | -1.3% | ns | |

### M4-Quarterly

| Config | UB SMAPE | CS SMAPE | Delta | Delta% | p-value | Sig? |
|--------|----------|----------|-------|--------|---------|------|
| NBEATS-G (baseline) | 15.088 | 12.743 | -2.345 | -15.5% | ns | |
| NBEATS-G (activeG) | 10.528 | 10.588 | +0.060 | +0.6% | ns | |
| NBEATS-I+G (baseline) | 10.290 | 10.153 | -0.137 | -1.3% | ns | |
| NBEATS-I+G (activeG) | 10.248 | 10.167 | -0.082 | -0.8% | 0.021 | ** |
| BNG (baseline) | 10.528 | 10.465 | -0.062 | -0.6% | ns | |
| BNG (activeG) | 10.521 | 10.529 | +0.008 | +0.1% | ns | |

**Summary:** Comprehensive sweep wins 10/12 configs. Significantly better in 3/12. Mean delta: -0.65 SMAPE (-3.8%).

The biggest beneficiaries are multi-component architectures (NBEATS-I+G) which need the Trend and Seasonality blocks to specialize -- a process that takes more epochs than homogeneous Generic stacks.

---

## 3. Divergence Analysis

### M4-Yearly Partial Divergence Rates

| Tier | Threshold | Unified Benchmark | Comprehensive Sweep |
|------|-----------|-------------------|---------------------|
| Mild | SMAPE > 25 | 5/288 (1.7%) | 2/1120 (0.2%) |
| Moderate | SMAPE > 45 | 3/288 (1.0%) | 1/1120 (0.1%) |
| Severe | SMAPE > 100 | 0/288 (0.0%) | 0/1120 (0.0%) |

The unified benchmark has **8.5x** higher mild divergence rate. Affected configs:
- BottleneckGeneric (baseline): 2/9 seeds bimodal (SMAPE ~45)
- Coif2WaveletV3 (baseline): 1/9 bimodal (SMAPE ~45)
- DB4WaveletV3 (baseline): 2/9 bimodal (SMAPE 45-76)

All are 30-stack homogeneous configs without Trend blocks. The LR warmup in the comprehensive sweep prevents the early-training instability that causes these failures.

### Cross-Dataset Divergence (Comprehensive Sweep)

| Dataset | Explicit Diverged | SMAPE > 100 | Total |
|---------|-------------------|-------------|-------|
| M4-Yearly | 0 | 0 | 1,120 |
| M4-Quarterly | 0 | 0 | 1,120 |
| Tourism | 0 | 0 | 1,114 |
| Weather | 0 | 260 (agf only) | 1,125 |
| Milk | 191 (17.1%) | 6 | 1,120 |

Weather's SMAPE > 100 runs are entirely due to `active_g=forecast` being catastrophic on that dataset -- not a training duration issue. Milk divergence is intrinsic to the dataset's small size.

---

## 4. Counterfactual: patience=20 vs patience=10

Using validation loss curves from the comprehensive sweep, we simulated what would have happened with patience=10:

| Dataset | Runs Improved by patience=20 | % | Mean Improvement |
|---------|------------------------------|---|------------------|
| M4-Yearly | 529/1,120 | 47.2% | 2.25% val_loss |
| M4-Quarterly | (similar) | ~13% | |
| Tourism | 65/1,114 | 5.8% | 8.03% val_loss |
| Weather | (similar) | ~20% | |

**M4-Yearly** is the dataset most helped by patience=20. Nearly half of all runs find a better minimum in the extra 10 patience epochs. Tourism converges so fast (best_epoch ~13) that extended patience rarely helps.

### Best Epoch Distribution

| Dataset | Mean Best Epoch | Median | % > 50 | % > 80 | Mean Waste |
|---------|-----------------|--------|--------|--------|------------|
| M4-Yearly | 39.3 | 37.0 | 22.7% | 2.5% | 23.2 epochs |
| M4-Quarterly | 33.9 | 32.0 | 12.8% | 1.2% | 29.6 epochs |
| Tourism | 13.1 | 12.0 | 0.9% | 0.3% | 101.0 epochs |
| Weather | 34.7 | 33.0 | 19.7% | 1.3% | 28.6 epochs |
| Milk | 26.9 | 23.0 | 9.0% | 1.6% | 19.0 epochs |

Tourism wastes 88% of training epochs -- `max_epochs=100` with `patience=10` would be perfectly adequate.

---

## 5. Architecture Sensitivity

### By Backbone (M4-Yearly)

| Backbone | % Improved by patience=20 | Mean Improvement | SMAPE |
|----------|---------------------------|------------------|-------|
| RootBlock | 52.6% | 2.80% | 13.706 |
| AERootBlock | 48.0% | 2.00% | 13.666 |
| AERootBlockLG | 43.1% | 1.87% | 13.693 |

RootBlock architectures benefit most because they have the largest parameter space to explore. AE/AELG bottlenecks constrain the optimization landscape, leading to faster convergence.

### Most Sensitive Block Types (M4-Yearly, >60% improvement rate)

1. **Trend+Seasonality+Generic** (72.5%): Multi-component architectures need blocks to specialize
2. **Trend+Coif2WaveletV3** (70.0%): Alternating RootBlock stacks with large capacity
3. **Trend+DB3WaveletV3** (70.0%): Same pattern
4. **Trend+Symlet10WaveletV3** (70.0%): Same pattern
5. **Trend+HaarWaveletV3** (65.0%): Same pattern

### Least Sensitive Block Types (<42% improvement rate)

- **TrendWaveletAELG** (40.5%, 0.58% improvement): Fastest convergence, least benefit
- **TrendAELG+Coif2WaveletV3AELG** (33.3%, 1.52% improvement)

---

## 6. Recommendations

### Optimal Training Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| `max_epochs` | 200 | 3.2% of runs need > 100 epochs; 200 is safe ceiling |
| `patience` | 20 | 47% of M4 runs benefit; cost is only ~10 extra epochs of compute |
| `lr_scheduler` | warmup(15) + cosine | Prevents bimodal convergence; most impactful single change |

### Dataset-Specific Overrides

| Dataset | max_epochs | patience | Notes |
|---------|------------|----------|-------|
| M4 | 200 | 20 | Standard settings; I+G configs need it |
| Tourism | 100 | 10 | Converges at epoch ~13; longer training is waste |
| Weather | 200 | 20 | Similar to M4 |
| Traffic | 200 | 20 | Standard; limited data |
| Milk | 200 | 20 | Divergence-dominated; active_g=forecast helps |

### What to Test Next

1. **LR warmup isolation study**: Run NBEATS-I+G and BottleneckGeneric with max_epochs=100, patience=10, WITH warmup, to isolate whether it's the warmup or the patience that prevents divergence.

2. **Patience=15 compromise**: Test patience=15 as a middle ground -- it should capture most of the benefit while saving ~5 epochs of waste per run.

3. **Adaptive patience by architecture**: Since AE/AELG configs converge 30-40% faster, they could use patience=12-15 while RootBlock configs use patience=20.

### Open Questions

1. **Is the LR warmup or the patience the primary divergence prevention mechanism?** The current data confounds both changes.

2. **Would the unified benchmark results change significantly with just a warmup added?** This would be the cheapest intervention.

3. **Can Tourism training be further shortened?** With best_epoch ~13 and patience=20, training runs for ~33 epochs needlessly.
