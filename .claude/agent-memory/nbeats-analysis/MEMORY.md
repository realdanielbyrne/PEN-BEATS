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

## AE Backbone Hierarchy (M4-Yearly, updated 2026-03-07)
- **RootBlock (non-AE) > AERootBlockLG > AERootBlock >> AERootBlockVAE** for wavelet blocks.
- Non-AE Trend+WaveletV3: SMAPE=13.410, OWA=0.794, ~1.4M params
- TrendAELG+WaveletV3AELG: SMAPE=13.438, OWA=0.795, ~4.3M params
- TrendAE+WaveletV3 (non-AE wavelet): SMAPE=13.462, OWA=0.797, ~4.2M params (TrendAE comparison study)
- TrendAE+WaveletV3AE: SMAPE=15.020, OWA=0.894 (at 10 epochs; gap may narrow with training)
- The learned gate in AELG is essential for AE-family competitiveness.
- TrendAE backbone does NOT improve over plain Trend even with non-AE WaveletV3 blocks.

## TrendAE + WaveletV3 Comparison Study (M4-Yearly) (2026-03-07)
- See `experiments/analysis/analysis_reports/wavelet_trendae_comparison_analysis.md`
- **30 configs, 90 runs (8 wavelet families x ~3 bd_labels x 3 latent_dims), 50 epochs**
- **TrendAE does NOT improve over plain Trend.** Best OWA=0.797 vs non-AE SOTA 0.794, with 3x more params.
- **CSV has 2 duplicate rows** for Symlet3_bd4_lt_fcast_ttd3_ld5. Always deduplicate before analysis.
- **No hyperparameter factor is significant** (all KW p>0.18). Study underpowered with 3 seeds.
- **Strong wavelet x latent_dim interaction:** Symlets prefer LD=8, Daubechies prefer LD=2, Coiflets/Haar prefer LD=5. Marginal LD analysis is misleading.
- **Most stable config:** Symlet3_bd4_lt_fcast_ttd3_ld8 (OWA=0.797, std=0.0006, range=0.0011).
- **10/30 configs have >5% outlier seed gap.** Seed 44 is worst in 8/16 problematic cases.
- **Mean vs median OWA rank correlation is only rho=0.65** (3 seeds insufficient for stable ranking).
- **bd_label and basis_dim are perfectly confounded** in this study; cannot separate their effects.
- **Prior analysis report was unreliable:** Recommended LD=2 (wrong), BD=30 (unreliable), claimed TrendAE was "clear winner" (wrong vs non-AE SOTA).

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

## Pure VAE (AERootBlockVAE) Performance Audit (2026-03-07)
- **Pure VAE stacks are never competitive.** Across all 4 datasets, no pure-VAE config comes close to winning.
- **M4-Yearly:** Best pure VAE = TrendVAE+SeasonalityVAE+GenericVAE_ld16 (SMAPE=19.77) vs overall best 13.41. +47% gap. Rank 1547/1842.
- **Tourism-Yearly:** Best pure VAE = NBEATS-I-VAE (SMAPE=108.83) vs overall best 21.32. +410% gap. Near-total failure for generic VAE blocks (SMAPE~190).
- **Traffic-96:** Best pure VAE = GenericVAE_ld16 (MSE=0.000701) vs overall best 0.000603. +16% gap. Rank 25/250. This is VAE's best relative showing.
- **Weather-96:** Best pure VAE = VAE-coif2-coif2/TrendVAE+WaveletV3VAE (MSE=1848) vs overall best 1518. +22% gap. **Rank 4/588** -- VAE's single strongest result. But note: overall #1 (Generic10_activeG) has 50 runs vs VAE's 5.
- **TrendVAE + deterministic wavelet combos** also underperform: best on M4 is TrendVAE+Haar (SMAPE=15.41, +15% vs best). On Tourism they catastrophically fail (SMAPE 65-72 vs best 21.3).
- **Backbone hierarchy extended:** RootBlock > AERootBlockLG > AERootBlock >> AERootBlockVAE. VAE is strictly worst across all datasets.
- **One exception worth noting:** On Weather, TrendVAE+WaveletV3VAE (VAE-coif2-coif2, MSE=1848) beat the AELG equivalent (AELG-coif2-coif2, MSE=2132) in the same AsymWavelet study. But AELG-sym20-coif2 (MSE=1804) still beat VAE.
- **KL divergence penalty likely too aggressive** for the residual N-BEATS architecture. The stochastic latent disrupts the precise backcast subtraction that N-BEATS relies on.

## TrendWaveletAELG Pure Study (2026-03-07)
- See `experiments/analysis/analysis_reports/trendwaveletaelg_pure_study_analysis.md`
- **NEW Tourism-Yearly SOTA:** TrendWaveletAELG coif3_eq_bcast_td3_ld16, SMAPE=20.681 (beats prior 20.930 from TrendAELG+WaveletV3AELG Coif1)
- **Best on M4-Yearly:** v2 db3_eq_fcast_td3_ld16 (SMAPE=13.463, +0.40% above non-AE SOTA 13.410), v1 db10_eq_fcast_td3_ld8 (SMAPE=13.506)
- **Wavelet family is a NON-FACTOR:** Kruskal-Wallis p=0.107 with 14 families on M4. AE bottleneck homogenizes bases. 0.18 SMAPE spread across all families.
- **bd_label matters MORE in TrendWaveletAELG** than alternating stacks: eq_fcast significantly best (p=0.017), 0.37 SMAPE spread at R3.
- **ld=16 > ld=8** (p=0.042). But DB4+eq_fcast+ld=16 catastrophic (~76 SMAPE).
- **Traffic-96:** 100% SMAPE=200. Worse than alternating's 86% divergence.
- **Cross-dataset family rankings uncorrelated** (rho=-0.1). Coif3 best avg rank (1.5).
- **Architecture recommendation:** TrendWaveletAELG preferred for short-horizon simplicity; beats alternating on Tourism.

## Critical Methodology Lesson
- **R1 (early training) data can produce misleading factor rankings.** Both ttd and bd_label showed R1 advantages that reversed or vanished at R3 convergence. Always validate hyperparameter recommendations with converged data. This affected two prior skill recommendations (ttd and bd_label).