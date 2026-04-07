---
name: comprehensive_sweep_cross_dataset
description: Major cross-dataset findings from the 112-config comprehensive sweep (M4 all 6 periods, Tourism, Weather, Milk). Updated 2026-04-06 with all M4 periods and refined active_g Weather findings.
type: project
---

## Comprehensive Sweep Cross-Dataset Analysis (2026-04-06, updated from 2026-03-22)

See `experiments/analysis/analysis_reports/comprehensive_sweep_cross_dataset_analysis.md`
See notebook: `experiments/analysis/notebooks/comprehensive_sweep_cross_dataset.ipynb`

### Critical Findings

1. **Wavelet architectures beat baselines on 6/9 dataset-periods.** Baselines only win M4-Q (tie), M4-Daily (under-tested, only 14 configs), M4-Hourly.

2. **active_g is dataset AND architecture dependent.** NOT simply "catastrophic on Weather":
   - **Weather unified/homogeneous stacks:** Catastrophic (SMAPE ~100)
   - **Weather alternating stacks:** Benign or beneficial (TWGAELG_10s significantly BETTER with agf, p=0.045)
   - **Tourism:** Essential (Wilcoxon p=0.0002), eliminates all bimodal failures
   - **Milk:** Critical for Generic blocks (SMAPE 19->2.4), marginal for wavelet
   - **M4:** Mixed (helps Yearly/Hourly, hurts Monthly)

3. **Backbone hierarchy is dataset-dependent (reverses on Weather/Milk):**
   - M4/Tourism: RootBlock > AELG > AE
   - Weather: AE > AELG > RootBlock
   - Milk: AE > AELG

4. **Latent dim preference reverses on Weather:** ld=8 > ld=16 > ld=32 (KW p=0.010). Elsewhere ld=16 safe.

5. **Sub-1M TWAELG/TWAE models within 0.5% of winners** on M4-Y/Q/M/W, Tourism (F12 confirmed).

6. **db3 is safest cross-dataset wavelet.** coif2 for M4-Y, Haar for Milk/M4-Hourly.

7. **Unified TrendWavelet wins short horizons (H=4-8).** Alternating wins Weather and M4-Weekly.

8. **10 stacks optimal for short horizons (Tourism p<0.001, Milk p=0.007). 30 stacks for long (M4-Daily, M4-Hourly).**

### Dataset-Specific Winners (All Periods)

| Dataset | Winner | SMAPE | Params |
|---------|--------|-------|--------|
| M4-Yearly | TW_10s_td3_bdeq_coif2 | 13.499 | 2.1M |
| M4-Quarterly | NBEATS-IG_10s_ag0 | 10.126 | 19.6M |
| M4-Monthly | TW_30s_td3_bd2eq_coif2 | 13.279 | 7.1M |
| M4-Weekly | T+Db3V3_30s_bdeq | 6.671 | 15.8M |
| M4-Daily | NBEATS-G_30s_ag0 | 2.603 | 26.0M (only 14 configs!) |
| M4-Hourly | NBEATS-IG_30s_agf | 8.587 | 43.6M |
| Tourism-Y | TW_10s_td3_bdeq_db3 | 21.773 | 2.0M |
| Weather-96 | TAE+DB3V3AE_30s_ld8_ag0 | MSE 0.138 | 7.1M |
| Milk | TALG+DB3V3ALG_10s_ag0 | 1.512 | 1.0M |

### Best Generalist
TALG+DB3V3ALG_10s_ag0 (2.4M params, mean rank 14.6/112)

### Finding Verification (F1-F12)
- **CONFIRMED:** F9 (novel arches beat baselines at fewer params), F12 (TWAELG ~400K Pareto optimal)
- **PARTIALLY CONFIRMED:** F2 (td3 safe default), F4 (eq_fcast reasonable default), F6 (skip marginal), F10 (alt AELG top on Weather/Milk)
- **DENIED/REVISED:** F1 (active_g dataset+arch dependent), F3 (wavelet matters), F5 (ld no pattern), F7 (active_g rescue dataset-dependent), F8 (backbone hierarchy dataset-dependent), F11 (AE stability confirmed on Milk/Weather only)

### FM7 Lookback Experiment (2026-03-24)

bl=672 (L=7H) is strictly worse than bl=480 (L=5H) on Weather-96. Mean degradation +6.51 SMAPE. Optimal lookback: fm5.

### Coverage Gaps (Priority Next Experiments)

1. **M4-Daily** -- only 14 configs tested, no AE/AELG variants
2. **Tourism coif3 + eq_bcast** -- SOTA (20.864) settings not in sweep grid
3. **M4-Hourly larger AE/AELG** -- TAE+DB3V3AE already #3
4. **Weather TAE+DB3V3AE at 10s/20s** -- find depth sweet spot

### Safe Defaults
- active_g=False (safe everywhere; use agf only on Tourism or alternating stacks)
- skip_distance=0 (disabled)
- latent_dim=16 for M4, ld=8 for Weather/Milk
- basis_dim=eq_fcast (short horizons), bd2eq (long horizons)
- trend_thetas_dim=3 (safe default; td5 for M4-Yearly)
- wavelet_family=db3 (cross-dataset); coif2 for M4-Y, Haar for Milk
- n_stacks=10 (short horizons), 30 (long horizons)
- forecast_multiplier=5 for Weather-96
- share_weights=True, n_blocks_per_stack=1
