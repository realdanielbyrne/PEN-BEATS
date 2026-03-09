# TrendWaveletAE Pure-Stack Study Analysis

**Status:** Superseded by the comprehensive analysis.

See `trendwavelet_ae_vs_aelg_comprehensive_analysis.md` for the unified analysis of all TrendWaveletAE and TrendWaveletAELG studies (v1 and v2) across all datasets.

## Quick Reference

### TrendWaveletAE Best Results

| Dataset | Config | Performance | Study | Runs |
|---------|--------|-------------|-------|------|
| M4-Yearly | sym20_eq_fcast_td3_ld8 | SMAPE=13.514 | v1 R3 | 3 |
| M4-Yearly | coif2_lt_fcast_td3_ld12 | SMAPE=13.509 | v2 R2 | 5 |
| Tourism-Yearly | db20_eq_fcast_td3_ld12 | SMAPE=21.013 | v2 R2 | 5 |
| Weather-96 | haar_lt_fcast_td3_ld8 | MSE=1970 | v2 R1 | 1 |
| Traffic-96 | N/A (empty file) | -- | -- | 0 |

### Key Finding

TrendWaveletAE is consistently outperformed by TrendWaveletAELG (Wilcoxon p=0.002 on M4-Yearly matched configs, MWU p=0.010 on Tourism). The learned gate adds negligible parameter overhead but provides meaningful regularization. Use AELG instead.

## Data Sources

- `experiments/results/m4/trendwaveletae_pure_study_results.csv` (2,133 rows, v1)
- `experiments/results/m4/trendwaveletae_v2_study_results.csv` (220 rows, v2 mixed AE+AELG)
- `experiments/results/tourism/trendwaveletae_v2_study_results.csv` (220 rows, v2 mixed)
- `experiments/results/weather/trendwaveletae_v2_study_results.csv` (7 rows, v2 sparse)
- `experiments/results/traffic/trendwaveletae_v2_study_results.csv` (0 rows, empty)
