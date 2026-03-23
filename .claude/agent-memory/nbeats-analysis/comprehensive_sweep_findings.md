---
name: comprehensive_sweep_cross_dataset
description: Major cross-dataset findings from the paper-ready 112-config comprehensive sweep (M4-Y, M4-Q, Tourism, Weather, Milk). Overturns many prior M4-only conclusions.
type: project
---

## Comprehensive Sweep Cross-Dataset Analysis (2026-03-22)

See `experiments/analysis/analysis_reports/comprehensive_sweep_cross_dataset_analysis.md`
See notebook: `experiments/analysis/notebooks/comprehensive_sweep_cross_dataset.ipynb`

### Critical New Findings

1. **active_g=forecast is CATASTROPHIC on Weather-96.** ALL 26 tested agf configs produce SMAPE ~100 vs ~42-46 for ag0. Universal failure across all block types. On Milk it's the opposite -- agf rescues from divergence. **active_g is a dataset-level setting, not block-level.**

2. **Only 2 of 12 prior findings confirmed (F9, F12).** 7 denied, 3 partially confirmed. Prior conclusions were artifacts of M4-only studies.

3. **Best generalist: TALG+DB3V3ALG_10s_ag0** (mean rank 14.6/112 across 5 datasets, 2.4M params). Weather winner, Milk #2, M4-Q #7.

4. **Wavelet family matters** (KW p<0.05 on 3/4 datasets). Best: db3 cross-dataset. coif2 for M4-Y, haar for Milk. Prior "barely matters" claim is wrong.

5. **td5 beats td3 on M4-Yearly** (p=0.009). Prior "always td3" is wrong for M4-Y. td3 still safe default elsewhere.

### Overturned Priors
- "active_g stabilizes Generic" --> DENIED (dataset-dependent)
- "Wavelet type barely matters" --> DENIED (significant on 3/4 datasets)
- "Backbone hierarchy AELG >= RB > AE" --> DENIED (no consistent hierarchy)
- "Higher ld hurts AE, helps AELG" --> DENIED (no pattern)
- "TrendAE+WaveletV3AE = most stable" --> DENIED (AELG is more stable)

### Dataset-Specific Winners
- M4-Yearly: TW_10s_td3_bdeq_coif2 (SMAPE=13.499, 2.1M params)
- M4-Quarterly: NBEATS-IG_10s_ag0 (SMAPE=10.126, 19.6M params)
- Tourism: NBEATS-G_10s_ag0 (SMAPE=21.672, 8.1M params)
- Weather: TALG+DB3V3ALG_10s_ag0 (SMAPE=41.540, 2.4M params)
- Milk: T+Sym10V3_30s_bdeq (SMAPE=1.262, 15.2M params, 50% divergence)

### Safe Defaults
- active_g=False (NEVER agf on Weather)
- skip_distance=0 (disabled)
- latent_dim=8 or 16
- basis_dim=eq_fcast
- trend_thetas_dim=3
- wavelet_family=db3
- n_stacks=10
- share_weights=True, n_blocks_per_stack=1
