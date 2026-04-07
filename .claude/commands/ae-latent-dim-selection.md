# Latent Dimension Selection for AE Wavelet Blocks

## Core Rule

When configuring `latent_dim` for AE-family wavelet blocks (`AERootBlock`, `AERootBlockLG`, `AERootBlockVAE`), follow these guidelines:

| Backbone | Recommended `latent_dim` | Avoid | Evidence Strength |
|----------|------------------------|-------|-------------------|
| AERootBlockLG | **16** | >= 32 | Strong (2 datasets, p < 0.02) |
| AERootBlock | **5 or 8** (equivalent) | 2 | Strong (995 runs, p < 3e-20) |
| AERootBlockVAE | 5-8 (tentative) | 2 | Extrapolated from AE/AELG |

**General principle:** Moderate latent dimensions outperform both extremes. Too small destroys information; too large removes the regularization benefit of the bottleneck.

---

## Detailed Recommendations

### AERootBlockLG (Learned-Gate AE)

- **Use `latent_dim=16`.** This was the best setting across M4-Yearly and Tourism-Yearly in the V1 AELG study.
- **Avoid `latent_dim=32`.** The V2 sweep (ld=32) top-5 means were significantly worse than V1 (ld=16) top-5 means (p < 0.02 on both datasets).
- The learned gate allows the network to discover effective dimensionality within the latent space, so a moderate dimension (16) gives the gate enough room to work without being overwhelmed by unused dimensions.

### AERootBlock (Plain AE)

- **Use `latent_dim=5` or `latent_dim=8`.** These are statistically equivalent (Mann-Whitney p=0.71, n=330 runs each).
- **Never use `latent_dim=2`.** It is catastrophically underparameterized:
  - ~1 SMAPE worse than ld=5/8 (17.295 vs 16.275/16.297)
  - Mann-Whitney p < 3e-20, effect size r=0.41
  - 64% of bottom-quartile configs use ld=2
  - Only 5% of top-quartile configs use ld=2
- The plain AE lacks a gating mechanism, so it needs enough latent dimensions to avoid discarding useful signal. But the tested range only went up to 8 -- **ld=16 may be better** (untested for plain AE, but optimal for AELG).

### TrendWaveletAELG (Unified Block)

- **Use `latent_dim=16`.** v2 (ld=16) top-5 configs significantly outperform v1 (ld=8) top-5 (Mann-Whitney p=0.042) on M4-Yearly.
- **Caution with DB4+eq_fcast at ld=16:** This specific combination catastrophically fails (SMAPE ~76). DB4 with other basis labels or at ld=8 is fine.
- Evidence: TrendWaveletAELG Pure Study. See `experiments/analysis/analysis_reports/trendwaveletaelg_pure_study_analysis.md`.

### AERootBlock on Weather-96: AE outperforms AELG (gate fn study, 2026-03-15)

On Weather-96 with alternating stacks (TrendAE + WaveletV3AE), plain AE significantly outperforms AELG (MWU p=0.036). The top 2 Weather configs are AE controls with no learned gate. The gate function choice (sigmoid vs wavy_sigmoid vs wavelet_sigmoid) is a non-factor on both M4 and Weather (KW p>0.20). This suggests `latent_dim=16` with plain AE (no gate) is worth testing on long-horizon datasets.

### Updated: Latent Dim is Dataset-Dependent (Comprehensive Sweep, 2026-04-06)

The comprehensive sweep (112 configs x 10 runs) found that **latent dim preferences reverse on Weather**:

| Dataset | Best ld | KW p-value | Notes |
|---------|---------|-----------|-------|
| M4-Yearly | 16 | 0.001 | ld=16 best, consistent with prior |
| M4-Quarterly | 8 | 0.0003 | Smaller bottleneck preferred |
| M4-Hourly | 16 | 0.002 | ld=16 best |
| Tourism | 8/16/32 equiv | ns | No meaningful difference |
| Weather-96 | **8 > 16 > 32** | **0.010** | **Reversed!** Smaller = better |
| Milk | 8 sufficient | -- | Simple series needs less |

**Revised recommendation:** Use ld=16 as default for M4; use ld=8 for Weather and Milk. The prior "always use ld=16" advice does not generalize to normalized multivariate series (Weather) or simple univariate series (Milk), where stronger regularization from a smaller bottleneck helps.

### Untested but Recommended: ld=16 for Plain AE

Given that:
1. AELG is optimal at ld=16
2. Plain AE is optimal at ld=8 (the maximum tested)
3. Performance is still improving from ld=5 to ld=8 (slightly, not significantly)
4. Plain AE outperforms AELG on Weather-96 (gate fn study), suggesting the gate may not be needed

It is plausible that plain AE would benefit from ld=12 or ld=16. This has not been tested on M4 but is supported indirectly by Weather-96 results.

---

## Study Design Implication

When designing AE wavelet block studies:
1. **Default:** Set `latent_dim=16` for AELG, `latent_dim=8` for plain AE
2. **Search budget available:** Test `latent_dim` in {5, 8, 16} for plain AE; {8, 16, 24} for AELG
3. **Never include `latent_dim=2`** -- it wastes search budget on configs that will be eliminated

---

## Evidence Summary

### AERootBlock (V3AE Study, M4-Yearly)

332 configs, 995 runs after deduplication, 14 wavelet families, 10 epochs.

| latent_dim | Mean SMAPE | Std | Mean OWA | n |
|-----------|-----------|-----|----------|---|
| 2 | 17.295 | 1.597 | 1.055 | 333 |
| 5 | 16.297 | 1.098 | 0.986 | 332 |
| 8 | 16.275 | 1.110 | 0.990 | 330 |

Pairwise Mann-Whitney U tests:

| Comparison | p-value | Effect r | Significant? |
|-----------|---------|----------|-------------|
| ld=2 vs ld=5 | 3.2e-20 | 0.413 | Yes (***) |
| ld=2 vs ld=8 | 4.0e-20 | 0.412 | Yes (***) |
| ld=5 vs ld=8 | 0.708 | 0.017 | No |

Kruskal-Wallis eta-squared = 0.112 (second-strongest factor after bd_label).

### AERootBlockLG (AELG Stack Height Sweep, M4-Yearly + Tourism-Yearly)

V1 study (ld=16) top-5 means beat V2 sweep (ld=32) top-5 means with p < 0.02 on both datasets. See `experiments/analysis/analysis_reports/v3aelg_stackheight_sweep_analysis.md`.

---

## Source / Reproducibility

| Evidence | Report | Data |
|---|---|---|
| V3AE Study (332 configs, AERootBlock, M4) | `experiments/analysis/analysis_reports/wavelet_v3ae_study_analysis.md` | `experiments/results/m4/wavelet_v3ae_study_results.csv` |
| V3AE Notebook | `experiments/analysis/notebooks/wavelet_v3ae_study_insights.ipynb` | Same CSV |
| AELG Stack Height Sweep (AERootBlockLG, M4+Tourism) | `experiments/analysis/analysis_reports/v3aelg_stackheight_sweep_analysis.md` | `experiments/results/m4/v3aelg_stackheight_sweep_results.csv` |
