# Persistent Notes

## Block Architecture
- WaveletV3 now respects `pywt.dwt_max_level(...)=0` and keeps `level=0` instead of forcing an invalid level-1 decomposition.
- For short targets, prefer short-support wavelets (`haar`, `db2`, `db3`); long-support families (`db20`, `sym20`, `coif10`) can collapse to approximation-only bases on short horizons.

## TrendAELG + WaveletV3AELG Stack Height Findings (2026-03-05)
- See `experiments/analysis/analysis_reports/v3aelg_stackheight_sweep_analysis.md`
- **Stack height:** repeats=2 (4 stacks) is always insufficient; repeats=4 (8) and repeats=5 (10) are equivalent after extended training (p>0.1). repeats=4 is more parameter-efficient.
- **Latent dim:** latent_dim=16 significantly outperforms latent_dim=32 on short-horizon tasks (M4-Yearly, Tourism-Yearly). V1 study (ld=16) top-5 means beat V2 sweep (ld=32) top-5 means (p<0.02 both datasets).
- **Best known configs (TrendAELG+WaveletV3AELG family):**
  - M4-Yearly: Symlet20_eq_fcast_ttd3_ld16 (SMAPE=13.438, OWA=0.795) -- replicated in V3AELG study
  - Tourism-Yearly: Coif1_eq_fcast_ttd3_ld16 (SMAPE=20.930) -- replicated in V3AELG study
  - Weather-96: Symlet20_eq_fcast_ttd5_ld16 (MSE=2070.61) -- from V3AELG study (3 seeds, high variance)
  - Traffic-96: NOT VIABLE (86% divergence rate)
- **Wavelet families (updated 2026-03-07):** Symlet20 is the best cross-dataset wavelet (avg rank 2.3 across M4/Tourism/Weather). DB4 excels on short horizons but fails on long ones (#14/14 on Weather). Coif2 and DB10 consistently underperform.
- **Basis labels:** `eq_fcast` (basis_dim=forecast_length) is optimal or co-optimal on all datasets. At convergence (R3), bd_label differences nearly vanish on M4 (spread <0.014 SMAPE).
- **Tourism/Weather degeneracy:** On Tourism-Yearly (fcast=4, bcast=8) and Weather-96 (fcast=96, bcast=192), `eq_fcast` and `lt_bcast` both resolve to the same basis_dim, producing identical results.
- **Trend thetas dim is dataset-dependent:** ttd=3 for short horizons (Tourism p=0.008), ttd=5 for long horizons (Weather p=0.042), no difference on M4-Yearly at convergence (p=0.16).
- **Traffic-96: architecture not viable.** 86.2% divergence rate. DB20 100% diverged, Haar 96%. Only DB2/DB3/DB4 occasionally converge. Do NOT use TrendAELG+WaveletV3AELG for Traffic without major modifications.
- **Next experiments needed:** Weather confirmation with 10 seeds, M4 other periods, Traffic recovery ablation, ttd crossover point study.

## Wavelet Study 2: Basis Dimension Sweep (Trend+WaveletV3, M4-Yearly) (2026-03-07)
- See `experiments/analysis/analysis_reports/wavelet_study_2_basis_dim_analysis.md`
- **Best config (non-AE):** Coif2_bd6_eq_fcast_td3 (SMAPE=13.410, OWA=0.794) -- only sub-0.800 mean OWA in sweep
- **Best single run:** Coif2_bd6_eq_fcast_td3, seed 44, SMAPE=13.293, OWA=0.785
- **Key rule:** Never set basis_dim < forecast_length (`lt_fcast`). It reliably hurts, especially for Haar.
- **Smooth wavelets prefer eq_fcast (bd=H), coarse wavelets prefer eq_bcast (bd=L):** Coif2 best at bd=6, Haar/DB3 best at bd=30.
- **Thetas dim interaction:** td=3 better for Coif2/DB3; td=5 better for Haar (compensates for coarse wavelet basis).
- **Non-AE Trend+WaveletV3 (SMAPE 13.410) beats TrendAELG+WaveletV3AELG (SMAPE 13.438)** on M4-Yearly, suggesting AE bottleneck may not help for this config.
- **Parameter counts nearly identical** across all basis_dims (~1.4% spread); select bd purely on quality.

## WaveletV3AE Study: TrendAE + WaveletV3AE on M4-Yearly (2026-03-07)
- See `experiments/analysis/analysis_reports/wavelet_v3ae_study_analysis.md`
- **332 configs, 995 runs (14 wavelet families x 4 bd_labels x 2 ttd x 3 ld), 10 epochs only**
- **V3AE substantially underperforms baselines:** Best SMAPE=15.020 vs non-AE 13.410 (+12%). Plain AE bottleneck is harmful.
- **Factor importance:** bd_label (eta^2=0.153) > latent_dim (0.112) > ttd (0.033) >> wavelet_family (0.003, ns)
- **Wavelet family does NOT matter** for V3AE (p=0.256). AE bottleneck homogenizes basis representations.
- **Best V3AE configs:** Symlet2_lt_bcast_ttd3_ld8 (SMAPE=15.020), DB3_lt_bcast_ttd3_ld5 (OWA=0.891, most consistent)
- **Failure modes:** lt_fcast (bd<H) and ld=2 are catastrophic. 65% and 64% of bottom-quartile configs use these.
- **ld=5 and ld=8 are equivalent** (p=0.71). ld=2 is ~1 SMAPE worse.
- **All runs hit MAX_EPOCHS at 10 epochs** with 3-6% improvement in final epoch. Extended training needed for fair comparison.
- **Recommendation:** Do NOT use TrendAE+WaveletV3AE for production. Use Trend+WaveletV3 or TrendAELG+WaveletV3AELG.

## AE Backbone Hierarchy (M4-Yearly)
- **RootBlock (non-AE) > AERootBlockLG > AERootBlock** for wavelet blocks.
- Non-AE Trend+WaveletV3: SMAPE=13.410, OWA=0.794
- TrendAELG+WaveletV3AELG: SMAPE=13.438, OWA=0.795
- TrendAE+WaveletV3AE: SMAPE=15.020, OWA=0.894 (at 10 epochs; gap may narrow with training)
- The learned gate in AELG is essential for AE-family competitiveness.

## ResNet Skip Connection Study (M4-Yearly) (2026-03-07)
- See `experiments/analysis/analysis_reports/resnet_skip_study_analysis.md`
- **Skip connections should NOT be a default setting.** Only beneficial for unstable architectures.
- **GenericAELG collapses at depth >= 20** (bimodal convergence: ~2/3 seeds stuck at SMAPE ~48). Skip connections (skip_distance=5, alpha=0.1) rescue convergence, reducing SMAPE from 36 to 13.8. Skip_distance=10 is insufficient.
- **TrendWav and Generic do NOT benefit from skip connections.** Both architectures are stable at 16-30 stacks without skip.
- **Fixed alpha (0.1) beats learnable alpha** in 4/5 comparisons. Learnable adds complexity without benefit.
- **Optimal skip_distance = floor(n_stacks / 6)** when skip is needed.
- **Legacy Generic (30x, 24.7M params) is NOT rehabilitated** by skip. TrendWav at 1.5M params achieves better SMAPE.
- **Study winner:** TW16_skip4_learn (SMAPE=13.521, OWA=0.802) -- does NOT beat prior SOTA (Coif2_bd6: 13.410, 0.794).
- **M4-Yearly SOTA unchanged:** Coif2_bd6_eq_fcast_td3 (Trend+WaveletV3, non-AE) SMAPE=13.410, OWA=0.794.
- **MetaForecaster:** Moderate predictive value (Spearman rho=0.49, p=0.015), 75% top-12 overlap, but ranked eventual winner 15th/24.
- **Next:** Test skip on non-AE Trend+WaveletV3 SOTA config; test on Traffic/Weather (longer sequences).

## Skill File Updates (2026-03-07)
- **wavelet-family-selection.md:** Updated from "Haar/DB2/DB3 as safe defaults" to "Symlet20 as universal best (avg rank 2.3/14)" with horizon-dependent tier recommendations.
- **trend-thetas-dim-selection.md:** Overhauled from "always ttd=3" to horizon-dependent: ttd=3 for H<=10, ttd=5 for H>=50. Prior "always ttd=3" was based on R1 data that mixed convergence speed with final quality.
- **wavelet-basis-dim-selection.md:** Strengthened from "soft guideline" to "confident recommendation" for eq_fcast. Added V3AELG R3 convergence evidence showing R1 bd_label differences vanish at convergence.

## Critical Methodology Lesson
- **R1 (early training) data can produce misleading factor rankings.** Both ttd and bd_label showed R1 advantages that reversed or vanished at R3 convergence. Always validate hyperparameter recommendations with converged data. This affected two prior skill recommendations (ttd and bd_label).