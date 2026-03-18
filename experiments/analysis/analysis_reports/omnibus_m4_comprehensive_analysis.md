# M4 Omnibus Benchmark: Comprehensive Analysis

**Date:** 2026-03-17
**Data:** `experiments/results/m4/omnibus_benchmark_results.csv`
**Notebook:** `experiments/analysis/notebooks/omnibus_m4_comprehensive_analysis.ipynb`

## Executive Summary

871 runs across 69 named configurations (52 unique after deduplication), covering M4-Yearly (689 runs, 52 configs) and M4-Quarterly (182 runs, 16 configs). Zero divergences on Yearly; 1 bimodal failure on Quarterly (NBEATS-G-baseline, seed 50). All runs used early stopping.

**Headline finding:** Novel wavelet-trend architectures match or beat paper baselines (NBEATS-G, NBEATS-I+G) at 10-80x fewer parameters. The top 10 Yearly configs are within OWA 0.003 of each other -- a statistical tie (all pairwise MWU p > 0.23).

## Data Quality Issues

1. **forecast_basis_dim sweep failed:** Configs with `_bd32`, `_bd64`, `_bd128` suffixes produce run-level identical results. The parameter did not take effect. 6 config groups (18 names) are affected. Only `_bd32` variants are retained.
2. **Naming aliases:** `NBEATS-G-baseline10` = `NBEATS-G-baseline`, `NBEATS-G-activeG10/20` = `NBEATS-G-activeG`, `NBEATS-I-activeG` = `NBEATS-I-baseline` (active_g has no effect on Trend/Seasonality blocks), `TrendAELG+SynWaveletV3AELG-30-skip` = `TrendAELG+DB3WaveletV3AELG-30-skip`.

## M4-Yearly Rankings (Top 20 by OWA)

| Rank | Config | SMAPE | +/- | MASE | OWA | +/- | N | Params | Family |
|------|--------|-------|-----|------|-----|-----|---|--------|--------|
| 1 | NBEATS-I+G-activeG | 13.508 | 0.125 | 3.094 | 0.8026 | 0.0100 | 10 | 36.0M | paper_baseline |
| 2 | TrendAELG+Coif2WaveletV3AELG-30 | 13.521 | 0.119 | 3.092 | 0.8027 | 0.0099 | 10 | 2.9M | novel_aelg |
| 3 | TrendAELG+Coif2WaveletV3AELG-30-activeG | 13.515 | 0.096 | 3.095 | 0.8028 | 0.0089 | 10 | 2.9M | novel_aelg |
| 4 | Trend+Coif2WaveletV3-30 | 13.530 | 0.154 | 3.091 | 0.8029 | 0.0129 | 10 | 15.2M | novel_nonae |
| 5 | TrendAELG+HaarWaveletV3AELG-30 | 13.521 | 0.146 | 3.100 | 0.8037 | 0.0137 | 10 | 3.1M | novel_aelg |
| 6 | TrendAE+DB3WaveletV3AE-30-activeG | 13.533 | 0.054 | 3.098 | 0.8038 | 0.0045 | 10 | 2.9M | novel_ae |
| 7 | TrendWaveletGeneric-10 | 13.545 | 0.183 | 3.103 | 0.8049 | 0.0147 | 10 | 2.1M | novel_nonae |
| 8 | TrendWaveletAELG-20 | 13.542 | 0.097 | 3.110 | 0.8056 | 0.0087 | 10 | 0.9M | novel_aelg |
| 9 | TrendAELG+Symlet3WaveletV3AELG-30 | 13.550 | 0.123 | 3.110 | 0.8058 | 0.0105 | 10 | 3.1M | novel_aelg |
| 10 | TrendAE+DB3WaveletV3AE-30 | 13.571 | 0.129 | 3.106 | 0.8060 | 0.0117 | 10 | 2.9M | novel_ae |
| 11 | TrendWaveletGenericAELG-10-activeG | 13.567 | 0.163 | 3.110 | 0.8064 | 0.0130 | 10 | 0.5M | novel_aelg |
| 12 | TrendAELG+Coif2WaveletV3AELG-30-skip | 13.560 | 0.097 | 3.117 | 0.8070 | 0.0076 | 10 | 2.9M | novel_aelg |
| 13 | TrendWaveletGeneric-30 | 13.587 | 0.161 | 3.110 | 0.8070 | 0.0126 | 10 | 6.3M | novel_nonae |
| 14 | TrendAELG+HaarWaveletV3AELG-20 | 13.562 | 0.132 | 3.119 | 0.8074 | 0.0116 | 10 | 2.1M | novel_aelg |
| 15 | NBEATS-I-baseline | 13.601 | 0.122 | 3.110 | 0.8074 | 0.0107 | 10 | 12.9M | paper_baseline |
| 16 | TrendWaveletAELG-10 | 13.579 | 0.089 | 3.116 | 0.8075 | 0.0077 | 10 | 0.4M | novel_aelg |
| 17 | TrendAE+Coif2WaveletV3AE-30 | 13.591 | 0.138 | 3.116 | 0.8079 | 0.0112 | 10 | 2.9M | novel_ae |
| 18 | Trend+Coif2WaveletV3-30-activeG_bd32 | 13.571 | 0.126 | 3.125 | 0.8084 | 0.0100 | 10 | 15.4M | novel_nonae |
| 19 | TrendAELG+DB3WaveletV3AELG-30-skip | 13.585 | 0.065 | 3.122 | 0.8084 | 0.0060 | 10 | 2.9M | novel_aelg |
| 20 | TrendAELG+DB3WaveletV3AELG-30 | 13.599 | 0.163 | 3.125 | 0.8093 | 0.0145 | 10 | 2.9M | novel_aelg |

**Paper baselines for reference:**

| Config | OWA | Rank | Params |
|--------|-----|------|--------|
| NBEATS-I+G-activeG | 0.8026 | 1 | 36.0M |
| NBEATS-I-baseline | 0.8074 | 15 | 12.9M |
| NBEATS-G-baseline | 0.8119 | 39 | 24.7M |
| NBEATS-I+G-baseline | 0.8148 | 50 | 36.0M |
| NBEATS-G-activeG | 0.8278 | 63 | 24.7M |

## M4-Quarterly Rankings

| Rank | Config | SMAPE | OWA | +/- | Params |
|------|--------|-------|-----|-----|--------|
| 1 | Trend+Coif2WaveletV3-30 | 10.128 | 0.8850 | 0.0037 | 15.4M |
| 2 | TrendWaveletAELG-10 | 10.187 | 0.8883 | 0.0020 | 0.5M |
| 3 | NBEATS-I+G-activeG | 10.140 | 0.8898 | 0.0092 | 36.3M |
| 4 | NBEATS-I+G-baseline | 10.151 | 0.8907 | 0.0095 | 36.3M |
| 5 | NBEATS-I+GenericAE | 10.181 | 0.8924 | 0.0053 | 17.8M |
| 6 | TrendWaveletGeneric-10 | 10.211 | 0.8942 | 0.0104 | 2.1M |
| ... | NBEATS-G-baseline | 12.740 | 1.1634 | 0.7758 | 25.0M |

NBEATS-G-baseline has 1/10 runs diverged (seed 50, SMAPE=34.06). activeG prevents this divergence.

## Backbone Comparison

| Backbone | N runs | Mean OWA | Median OWA | Best Config OWA |
|----------|--------|----------|------------|-----------------|
| RootBlock | 159 | 0.8111 | 0.8099 | 0.8029 |
| AERootBlockLG | 240 | 0.8119 | 0.8096 | 0.8027 |
| AERootBlock | 100 | 0.8188 | 0.8165 | 0.8038 |

- Kruskal-Wallis: H=18.76, p=0.000084
- RootBlock vs AERootBlockLG: MWU p=0.61 (equivalent)
- RootBlock vs AERootBlock: MWU p=0.0001 (RootBlock better)
- AERootBlockLG vs AERootBlock: MWU p=0.0001 (AELG better)

**Conclusion:** RootBlock = AERootBlockLG >> AERootBlock. The learned gate is essential for AE-family competitiveness.

## active_g Effect

15 paired comparisons (Yearly):
- active_g helps: 4/15
- active_g hurts: 11/15
- Significant at p<0.05: 0/15

**Verdict:** active_g is a non-factor. Default should be False.

## Skip Connection Effect

Two paired comparisons at 30 stacks:
- Coif2 AELG: no_skip 0.8027 vs skip 0.8070 (p=0.52, no_skip better)
- DB3 AELG: no_skip 0.8093 vs skip 0.8084 (p=0.57, skip marginally better)

Skip reduces variance (std drops by ~35%) but does not improve mean OWA. Not recommended for stable architectures.

## Wavelet Family Comparison (AELG backbone, 30 stacks)

| Wavelet | OWA | Std |
|---------|-----|-----|
| Coif2 | 0.8027 | 0.0099 |
| Haar | 0.8037 | 0.0137 |
| Symlet3 | 0.8058 | 0.0105 |
| DB3 | 0.8093 | 0.0145 |

Kruskal-Wallis: H=1.04, p=0.79. **Wavelet family is a non-factor** for AELG on M4-Yearly.

## Pareto-Optimal Configurations (OWA vs Parameters)

| Config | OWA | Params | OWA gap vs best | Param ratio |
|--------|-----|--------|----------------|-------------|
| TrendWaveletAELG-10 | 0.8075 | 0.44M | +0.6% | 82x fewer |
| TrendWaveletGenericAELG-10-activeG | 0.8064 | 0.45M | +0.5% | 80x fewer |
| TrendWaveletAELG-20 | 0.8056 | 0.87M | +0.4% | 41x fewer |
| TrendWaveletGeneric-10 | 0.8049 | 2.09M | +0.3% | 17x fewer |
| TrendAE+DB3WaveletV3AE-30-activeG | 0.8038 | 2.94M | +0.2% | 12x fewer |
| TrendAELG+Coif2WaveletV3AELG-30 | 0.8027 | 2.94M | +0.01% | 12x fewer |
| NBEATS-I+G-activeG | 0.8026 | 35.95M | BEST | 1x |

## Cross-Period Consistency

15 configs tested on both Yearly and Quarterly. Spearman rank correlation (excluding diverged NBEATS-G): rho ~ 0.6-0.7 (moderate).

**Cross-period champion:** Trend+Coif2WaveletV3-30 -- top 4 on Yearly, #1 on Quarterly.

## Best Individual Runs

| Config | Seed | SMAPE | OWA |
|--------|------|-------|-----|
| Trend+Coif2WaveletV3-30 | 44 | 13.298 | 0.7842 |
| TrendWaveletGeneric-10 | 43 | 13.317 | 0.7860 |
| TrendAELG+Coif2WaveletV3AELG-30 | 45 | 13.334 | 0.7872 |
| TrendWaveletGeneric-30 | 46 | 13.317 | 0.7874 |
| NBEATS-I+G-activeG | 44 | 13.326 | 0.7879 |

## Most Stable Configurations (Lowest OWA Std)

| Config | Mean OWA | Std | CV |
|--------|----------|-----|-----|
| TrendAE+DB3WaveletV3AE-30-activeG | 0.8038 | 0.0045 | 0.0056 |
| TrendAELG+DB3WaveletV3AELG-30-skip | 0.8084 | 0.0060 | 0.0074 |
| TrendWaveletAELG-10-activeG | 0.8117 | 0.0065 | 0.0080 |
| TrendWaveletAELG-10 | 0.8075 | 0.0077 | 0.0095 |

AE-family configs are consistently more stable -- the bottleneck regularizes training.

## Recommendations

### Current Best Configs

| Use Case | Config | OWA | Params |
|----------|--------|-----|--------|
| Quality-first | TrendAELG+Coif2WaveletV3AELG-30 | 0.8027 | 2.9M |
| Cross-period | Trend+Coif2WaveletV3-30 | 0.8029/0.8850 | 15.2M |
| Efficient | TrendWaveletAELG-10 | 0.8075 | 0.44M |
| Stability | TrendAE+DB3WaveletV3AE-30-activeG | 0.8038 | 2.9M |

### What to Test Next

1. Complete Quarterly sweep for top Yearly configs
2. Extend to Monthly, Weekly, Daily, Hourly periods
3. Fix `forecast_basis_dim` implementation and re-run sweep
4. `forecast_multiplier` sweep on M4
5. Ensemble of top 3-5 statistically tied configs

### Open Questions

- Can ensemble methods break the OWA 0.803 barrier?
- Why does NBEATS-I+G-activeG win on Yearly but not Quarterly?
- Would TrendWaveletAELG-30 with eq_fcast bd match alternating architecture?
