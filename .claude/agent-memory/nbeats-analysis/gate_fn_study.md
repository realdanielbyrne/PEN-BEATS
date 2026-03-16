---
name: gate-fn-study-findings
description: Gate function ablation for AELG backbone (sigmoid vs wavy_sigmoid vs wavelet_sigmoid) on M4-Yearly and Weather-96. Gate fn is a non-factor on both datasets. FM5 beats FM7 cross-dataset. AE beats AELG on Weather.
type: project
---

## Gate Function Study (M4-Yearly + Weather-96, 2026-03-15)

- See `experiments/analysis/analysis_reports/gate_fn_study_analysis.md`
- See notebook: `experiments/analysis/notebooks/gate_fn_study_insights.ipynb`
- **M4: 160 runs (32 configs x 5 seeds), Weather: 143 runs (29 configs x ~5 seeds), zero divergences.**

### Key Findings

1. **Gate function is a NON-FACTOR (cross-dataset confirmed).**
   - M4: KW p=0.37, spread=0.018 SMAPE (0.13%). Weather: KW p=0.64, spread=0.003 MSE (1.8%).
   - No per-architecture significance on either dataset (all KW p > 0.20).
   - Cross-dataset avg rank: wavelet_sigmoid 1.50, sigmoid 1.75, wavy_sigmoid 2.75 -- but differences are noise.
   - **Recommendation:** Keep sigmoid as default. Remove wavy/wavelet_sigmoid from production consideration.

2. **FM5 significantly beats FM7 on BOTH datasets** (paired Wilcoxon p<0.001 each).
   - M4: FM5 wins 16/16 configs, mean improvement -0.090 SMAPE (-0.66%).
   - Weather: FM5 wins 13/13 configs, mean improvement -0.035 MSE (-17.5%).
   - FM5 converges faster on M4 (best epoch ~55 vs ~75), always early-stops.
   - **How to apply:** Use forecast_multiplier=5 as default for M4-Yearly and Weather-96.

3. **AE vs AELG is DATASET-DEPENDENT.**
   - M4: AELG weakly better (all 4 architectures, none significant).
   - Weather: AE significantly better overall (MWU p=0.036). Top 2 Weather configs are AE controls.
   - The learned gate may over-constrain long-horizon Weather bottleneck.
   - **How to apply:** For Weather-96, prefer AE over AELG for alternating stacks.

4. **Alternating stacks > homogeneous (confirmed on both datasets).**
   - M4: MWU p=0.0006. Weather: MWU p=0.015.
   - T+W+G does NOT improve over T+W (extra Generic adds params without accuracy).

### Study Winners

- **M4:** TALG_WV3LG_GALG21_wavy FM5 (SMAPE=13.495, OWA=0.802, 2.62M params). Does NOT beat SOTA (13.410).
- **Weather:** TA_WV3A20_control FM5 (MSE=0.121, MAE=0.216, 4.61M params). AE control, not AELG.

### Weather-Specific Notes

- Weather FM5 incomplete: 3 of 16 configs missing from FM5, TWGLG20_wavy has 3 runs in FM7.
- Weather variance is high: CV=14-26% per config vs M4's 0.3-0.5%. Need 10+ seeds for stable rankings.
- Seed 44 consistently worst on Weather (confirming prior cross-study pattern).
