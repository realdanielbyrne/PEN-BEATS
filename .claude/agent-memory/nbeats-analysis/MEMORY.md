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
- **Wavelet families (updated 2026-03-08):** Symlet20 is the best cross-dataset wavelet for M4/Tourism/Weather (avg rank 2.3/14). DB4 excels on short horizons but fails on long ones (#14/14 on Weather). DB10 consistently underperforms. Coif2 underperforms as a backcast wavelet on M4/Tourism but is the **best forecast wavelet** on Traffic-96 and Weather-96 (AsymWavelet study). Context matters: coif2's boundary-vanishing-moment properties benefit short forecast horizons.
- **Basis labels:** `eq_fcast` (basis_dim=forecast_length) is optimal or co-optimal on all datasets. At convergence (R3), bd_label differences nearly vanish on M4 (spread <0.014 SMAPE).
- **Tourism/Weather degeneracy:** On Tourism-Yearly (fcast=4, bcast=8) and Weather-96 (fcast=96, bcast=192), `eq_fcast` and `lt_bcast` both resolve to the same basis_dim, producing identical results.
- **Trend thetas dim is dataset-dependent:** ttd=3 for short horizons (Tourism p=0.008), ttd=5 for long horizons (Weather p=0.042), no difference on M4-Yearly at convergence (p=0.16).
- **Traffic-96: architecture viability is depth-dependent.** Deep configs (repeats=4+) have 86%+ divergence. But 8-stack (repeats=2, 4 alternating pairs) drops divergence to 16% and achieves MSE=0.000603 (AsymWavelet Diagnostic study, 2026-03-08).
- **Next experiments needed:** Weather confirmation with 10 seeds, M4 other periods, Traffic recovery ablation, ttd crossover point study.

## Asymmetric Wavelet Diagnostic Study (Traffic-96, Weather-96) (2026-03-08)
- See `experiments/analysis/analysis_reports/asym_wavelet_diagnostic_analysis.md`
- **Asymmetric wavelet pairs do NOT help.** Symmetric pairs match or beat asymmetric on both datasets. Traffic AELG: symmetric significantly better (MWU p=0.026).
- **coif2 is the best forecast wavelet** on both Traffic (MSE=0.000607) and Weather (MSE=1968). Appears in #1 config on both datasets.
- **AELG dominates VAE on Traffic** (5.6x MSE, p<1e-6) but backbones are equivalent on Weather (p=0.85).
- **VAE is wavelet-blind:** Kruskal-Wallis p=0.49 (Traffic), p=0.33 (Weather). Stochastic bottleneck overwhelms basis.
- **8-stack AELG on Traffic is viable** (16% divergence vs 86%+ at deeper stacks). Best: AELG-coif2-coif2 MSE=0.000603.
- **Best configs:** Traffic: AELG-coif2-coif2 (MSE=0.000603), Weather: AELG-sym20-coif2 (MSE=1804, high variance, needs more seeds).

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

## ResNet Skip Connection Study v1 (M4-Yearly) (2026-03-07)
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

## ResNet Skip Connection Study v2 (M4-Yearly) (2026-03-07)
- See `experiments/analysis/analysis_reports/resnet_skip_study_v2_analysis.md`
- See notebook: `experiments/analysis/analysis_reports/resnet_skip_study_v2_analysis.ipynb`
- **36 configs, 207 runs, 3-round successive halving (15/30/100 epochs).** Tests GenericAE, TrendWaveletAE, TrendWaveletAELG, double-VAE, double-VAE2, and TrendVAE+HaarWaveletV3.
- **Winner: TWALG30_no_skip** (TrendWaveletAELG, 30 stacks, NO skip) SMAPE=13.568 ± 0.131, OWA=0.805. Matches v1 winner (13.521) within noise.
- **TrendWavelet blocks are depth-stable:** No degradation from 10 to 30 stacks for both TrendWaveletAE and TrendWaveletAELG. Skip connections don't help (and slightly hurt at 30 stacks).
- **GenericAE does NOT degrade at depth** (unlike GenericAELG): SMAPE 15.2 at 30 stacks vs GenericAELG's collapse to 36. The learned gate (AERootBlockLG) is the source of depth instability, not the AE bottleneck itself.
- **Double-VAE pairing is catastrophically bad:** TrendVAE+WaveletV3VAE (SMAPE 29-31) and TrendVAE2+WaveletV3VAE2 (SMAPE 37-44). Compounded reparameterization noise corrupts backcast residuals. Skip connections cannot rescue this.
- **Single-VAE + deterministic wavelet works but underperforms:** TrendVAE+HaarWaveletV3 (SMAPE 15.3 at 10 stacks) is 2x better than double-VAE but still ~12% worse than TrendWaveletAE (13.6). Uses 3x more parameters (4.4M vs 1.5M).
- **skip_distance=2 is too frequent.** Eliminated in R2 for all architectures. Sweet spot remains 3-5.
- **Unified TrendWavelet matches alternating architecture:** SMAPE 13.57 (unified, 4.5M params) vs 13.52 (alternating TrendAELG+WaveletV3AELG, ~13M params at matched depth). 3x parameter efficiency advantage.
- **All R3 configs early-stopped** (39-64 epochs of 100 max). Most stable: TWA30_no_skip (CV=0.5%).
- **Backbone hierarchy updated:** RootBlock > AERootBlockLG ≈ AERootBlock (at depth) >> AERootBlockVAE. Key insight: GenericAE (AERootBlock) is depth-stable while GenericAELG (AERootBlockLG) collapses, reversing the usual LG > non-LG ordering specifically for depth scaling.

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

## TrendWaveletAE vs TrendWaveletAELG Comprehensive Study (2026-03-08)
- See `experiments/analysis/analysis_reports/trendwavelet_ae_vs_aelg_comprehensive_analysis.md`
- See notebook: `experiments/analysis/notebooks/trendwavelet_ae_vs_aelg_comprehensive.ipynb`
- **3,821 rows across 9 CSV files, 4 datasets, 2 block types, 2 study versions**
- **AELG beats AE on M4** (Wilcoxon p=0.002, d=0.23). **On Tourism, AELG advantage is ld-dependent:** significant at ld=16 (MWU p=0.010 in comprehensive study) but vanishes at ld=8/12 (p=0.24 in v2 tourism study). AE slightly better at smaller latent dims.
- **Best configs per dataset:**
  - M4-Yearly: AELG v2 db3_eq_fcast_td3_ld16 (SMAPE=13.463, +0.40% vs non-AE SOTA 13.410)
  - Tourism-Yearly: AELG coif3_eq_bcast_td3_ld16 (SMAPE=20.864 confirmed with 10 runs, original 3-run estimate was 20.681)
  - Weather-96: AELG db3_eq_fcast_td3_ld16 (MSE=1920, high variance std=603, 3 seeds)
  - Traffic-96: 100% divergence (ALL 80 runs SMAPE=200)
- **Latent dim is the dominant hyperparameter** (KW p<0.001). ld=2 is catastrophic (d=0.72 vs ld=8). ld=5~ld=8 (d=0.23). ld=12 and ld=16 can improve further but introduce instability (DB4+eq_fcast+ld16 catastrophe).
- **Wavelet family is a NON-FACTOR** for both AE (p=0.38) and AELG (p=0.11). AE bottleneck homogenizes bases. Spread <0.10 SMAPE across 14 families.
- **Trend dim is a non-factor** on M4-Yearly (p=0.71). td=3 and td=5 identical.
- **bd_label is marginal** for AE (p=0.096) but significant for AELG (p=0.017). eq_fcast best for both.
- **Tourism bd_label degeneracy confirmed:** eq_fcast and lt_bcast resolve to identical bd=4, producing identical results (verified at run level).
- **V1 vs V2:** V2 (ld=16) beats V1 (ld=8) on 6/8 matched AELG configs, but not significant (p=0.11, n=8).
- **Architecture recommendation:** AELG preferred over AE universally. For Tourism, use unified TrendWaveletAELG. For M4 max accuracy, use non-AE Trend+WaveletV3.
- **Key gaps:** No v1 data outside M4. AE v2 Traffic empty, AE v2 Weather 7 rows only. No TrendWaveletAE at ld=16.

## Trend-Seasonality-Wavelet Comparison Study (M4-Yearly) (2026-03-08)
- See `experiments/analysis/analysis_reports/trend_seas_wav_comparison_analysis.md`
- See notebook: `experiments/analysis/notebooks/trend_seas_wav_comparison_insights.ipynb`
- **200 runs, 40 configs, 5 seeds.** Trend(AE/AELG) + Second(SeasonalityAE/AELG/VAE, HaarWavAE, Coif2WavAE, HaarWavAELG, Coif2WavAELG) x I-style(2 stacks) vs alternating(10 stacks) x ld(8,12).
- **Wavelet blocks are highly depth-sensitive (20x more than seasonality).** Wavelet I-style SMAPE=14.08 (worst), wavelet alternating SMAPE=13.54 (best). Seasonality: 13.85 vs 13.83 (negligible). Never use wavelet blocks in <6 stacks.
- **Alternating wavelet configs are Pareto-optimal:** SMAPE 13.50 with 965K params vs seasonality's 13.54 with 11.3M params (12x more).
- **arch_style is the dominant factor** (KW H=50.3, p<0.0001, eta^2=0.25). second_backbone significant (p=0.0001). All other factors ns.
- **TrendAE vs TrendAELG: non-factor** (p=0.85). Haar vs Coif2: non-factor (p>0.5). ld 8 vs 12: non-factor (p=0.59).
- **Best config:** TrendAE+HaarWaveletV3AE_Alt_ld12 (SMAPE=13.496, OWA=0.800, 965K params). Does NOT beat M4-Yearly SOTA (13.410).
- **Most parameter-efficient sub-13.55 config to date.**

## TrendWaveletAE v2 Tourism-Yearly Study (2026-03-09)
- See `experiments/analysis/analysis_reports/trendwaveletae_v2_tourism_analysis.md`
- See notebook: `experiments/analysis/notebooks/trendwaveletae_v2_tourism_insights.ipynb`
- **220 rows, 40 configs (R1) -> 20 configs (R2), zero divergence.**
- **Does NOT beat Tourism SOTA (20.864).** Best: TrendWaveletAE_db20_eq_fcast_td3_ld12 (SMAPE=21.013, +0.72%).
- **Haar is the best wavelet for Tourism-Yearly** (KW p=0.009, all post-hoc p<0.05). Short-support wavelets suit H=4. Dataset-specific: on M4-Yearly wavelet family is a non-factor.
- **ld=12 > ld=8** (MWU p=0.020, d=0.40). Continues the pattern: larger ld helps for unified TrendWavelet.
- **bd_label is a non-factor** (p=0.65). Even bd=2 (lt_fcast) works for 4-step forecasts.
- **AE vs AELG: no significant difference** (p=0.24). AE edges ahead (21.27 vs 21.38). Reverses the usual "AELG > AE" pattern seen on M4.
- **Prior comprehensive study finding "AELG beats AE on Tourism" (p=0.010) may have been driven by ld=16 configs** not present in this study. At ld=8/12, AE is equivalent or slightly better.
- **Next test needed:** ~~Haar at ld=16 on Tourism~~ DONE -- see Haar ld=16 confirmation below.

## Tourism-Yearly Haar ld=16 Confirmation Study (2026-03-09)
- See `experiments/analysis/analysis_reports/tourism_haar_ld16_confirmation_analysis.md`
- See notebook: `experiments/analysis/notebooks/tourism_haar_ld16_confirmation.ipynb`
- **20 rows, 2 configs (AE, AELG) x 10 runs, 50 epochs, zero divergences.**
- **Does NOT beat Tourism SOTA (20.864).** AE_haar: 20.996 (+0.63%), AELG_haar: 21.057 (+0.93%).
- **AE vs AELG: no difference** (Wilcoxon p=0.32, MWU p=0.62). Confirms AE=AELG on Tourism at ld>=12.
- **AE_haar vs SOTA:** MWU p=0.104 (not significant), bootstrap P(better)=5.8%. SOTA wins 8/10 seeds.
- **AELG_haar vs SOTA:** MWU p=0.031 (significant), bootstrap P(better)=1.2%. SOTA wins 9/10 seeds.
- **ld=16 IS the best latent dim for Haar on Tourism** (vs ld=12: MWU p=0.0002, d=-1.80). But even at best ld, Haar does not reach coif3 quality.
- **Coif3's advantage may come from eq_bcast (bd=8) vs eq_fcast (bd=4)**, not the wavelet family itself. Needs ablation.
- **Tourism-Yearly SOTA confirmed:** TrendWaveletAELG_coif3_eq_bcast_td3_ld16, SMAPE=20.864, 95% CI [20.712, 21.016].

## Critical Methodology Lesson
- **R1 (early training) data can produce misleading factor rankings.** Both ttd and bd_label showed R1 advantages that reversed or vanished at R3 convergence. Always validate hyperparameter recommendations with converged data. This affected two prior skill recommendations (ttd and bd_label).