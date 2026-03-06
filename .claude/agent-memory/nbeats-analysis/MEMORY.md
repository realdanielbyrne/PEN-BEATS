# Persistent Notes

## Block Architecture
- WaveletV3 now respects `pywt.dwt_max_level(...)=0` and keeps `level=0` instead of forcing an invalid level-1 decomposition.
- For short targets, prefer short-support wavelets (`haar`, `db2`, `db3`); long-support families (`db20`, `sym20`, `coif10`) can collapse to approximation-only bases on short horizons.

## TrendAELG + WaveletV3AELG Stack Height Findings (2026-03-05)
- See `experiments/analysis/analysis_reports/v3aelg_stackheight_sweep_analysis.md`
- **Stack height:** repeats=2 (4 stacks) is always insufficient; repeats=4 (8) and repeats=5 (10) are equivalent after extended training (p>0.1). repeats=4 is more parameter-efficient.
- **Latent dim:** latent_dim=16 significantly outperforms latent_dim=32 on short-horizon tasks (M4-Yearly, Tourism-Yearly). V1 study (ld=16) top-5 means beat V2 sweep (ld=32) top-5 means (p<0.02 both datasets).
- **Best known configs (TrendAELG+WaveletV3AELG family):**
  - M4-Yearly: Symlet20_eq_fcast_ttd3_ld16 (SMAPE=13.438, OWA=0.795) from V1 study
  - Tourism-Yearly: Coif1_eq_fcast_ttd3_ld16 (SMAPE=20.930) from V1 study
- **Wavelet families:** Coif1 and Coif2 are the most consistent across datasets. Symlet3 consistently underperforms.
- **Basis labels:** `eq_fcast` (basis_dim=forecast_length) and `lt_bcast` are strongest. `lt_fcast` is too restrictive for short forecasts.
- **Tourism degeneracy:** On Tourism-Yearly (fcast=4, bcast=8), `eq_fcast` and `lt_bcast` both resolve to basis_dim=4, producing identical results.
- **Next experiments needed:** Stack height sweep at latent_dim=16, latent_dim grid search {8,16,24,32}.