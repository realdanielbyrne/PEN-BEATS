# Tourism-Yearly Haar + ld=16 Confirmation Analysis

**Date:** 2026-03-09
**Dataset:** Tourism-Yearly (forecast=4, backcast=8)
**CSV:** `experiments/results/tourism/tourism_haar_ld16_confirmation_results.csv`
**Notebook:** `experiments/analysis/notebooks/tourism_haar_ld16_confirmation.ipynb`

## Executive Summary

**SOTA unchanged.** Neither TrendWaveletAE nor TrendWaveletAELG with Haar wavelet at ld=16 beats the current Tourism-Yearly SOTA (TrendWaveletAELG_coif3_eq_bcast_td3_ld16, SMAPE=20.864). The best Haar config (AE backbone) achieves SMAPE=20.996, +0.132 above SOTA. While the CIs overlap, bootstrap analysis gives only a 5.8% probability that Haar AE is truly better than coif3 AELG. The AELG Haar variant is significantly worse (MWU p=0.031).

However, ld=16 is confirmed as the best latent dim for Haar on Tourism (MWU p=0.0002 vs ld=12), and both AE and AELG backbones are equivalent at this latent dim (Wilcoxon p=0.32).

## Experiment Design

- 2 configs: TrendWaveletAE and TrendWaveletAELG, both with haar_eq_fcast_td3_ld16
- 10 runs per config (seeds 42-51), 50 epochs each
- Zero divergences
- 1.46M parameters (AE) / 1.46M parameters (AELG)

## Results

### Summary Statistics

| Rank | Config | Mean SMAPE | Std | 95% CI | vs SOTA |
|------|--------|-----------|-----|--------|---------|
| 1 | **SOTA:** AELG_coif3_eq_bcast_td3_ld16 | 20.864 | 0.212 | [20.712, 21.016] | -- |
| 2 | AE_haar_eq_fcast_td3_ld16 | 20.996 | 0.172 | [20.874, 21.119] | +0.132 (+0.63%) |
| 3 | AELG_haar_eq_fcast_td3_ld16 | 21.057 | 0.183 | [20.926, 21.188] | +0.193 (+0.93%) |

### Per-Seed Comparison

| Seed | AE_haar | AELG_haar | SOTA_coif3 | AE < SOTA | AELG < SOTA |
|------|---------|-----------|------------|-----------|-------------|
| 42 | 21.106 | 20.806 | 20.779 | no | no |
| 43 | 21.076 | 21.170 | 20.814 | no | no |
| 44 | 21.086 | 21.049 | 20.977 | no | no |
| 45 | 20.634 | 20.927 | 20.586 | no | no |
| 46 | 21.098 | 21.126 | 21.082 | no | no |
| 47 | 20.891 | 21.108 | 21.311 | YES | YES |
| 48 | 21.182 | 21.121 | 20.871 | no | no |
| 49 | 20.812 | 20.877 | 20.832 | YES | no |
| 50 | 20.957 | 20.941 | 20.707 | no | no |
| 51 | 21.122 | 21.446 | 20.682 | no | no |

SOTA wins on 8/10 seeds vs AE_haar, 9/10 seeds vs AELG_haar.

## Statistical Tests

### AE vs AELG (within this study)

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Mann-Whitney U | U=43.0 | 0.623 | Not significant |
| Wilcoxon signed-rank | W=17.0 | 0.322 | Not significant |
| Cohen's d | -0.344 | -- | Small effect (AE trivially better) |

**Conclusion:** AE and AELG are equivalent on Tourism-Yearly at ld=16. This confirms the v2 Tourism study finding that the AE-vs-AELG advantage seen on M4 does not transfer to Tourism at larger latent dims.

### AE_haar vs SOTA

| Test | Statistic | p-value | Effect Size |
|------|-----------|---------|-------------|
| Mann-Whitney U | U=72.0 | 0.104 | Cohen's d = 0.685 |
| Bootstrap P(AE < SOTA) | -- | -- | 5.8% |

**Conclusion:** Not significant at alpha=0.05 but the effect size is medium-large. The CI overlap (SOTA upper 21.016, AE lower 20.874) is real but misleading -- on a per-seed basis SOTA dominates 8/10.

### AELG_haar vs SOTA

| Test | Statistic | p-value | Effect Size |
|------|-----------|---------|-------------|
| Mann-Whitney U | U=79.0 | **0.031** | Cohen's d = 0.974 |
| Bootstrap P(AELG < SOTA) | -- | -- | 1.2% |

**Conclusion:** Significantly worse than SOTA. Large effect size.

### Latent Dim Scaling (Haar, TrendWaveletAE)

| Latent Dim | N | Mean SMAPE | Std | 95% CI |
|-----------|---|-----------|-----|--------|
| ld=8 | 16 | 21.572 | 0.885 | [21.101, 22.043] |
| ld=12 | 6 | 23.234 | 1.753 | [21.395, 25.073] |
| **ld=16** | **10** | **20.996** | **0.172** | **[20.874, 21.119]** |

- ld=16 vs ld=12: MWU p=0.0002, Cohen's d = -1.80 (massive improvement)
- ld=16 vs ld=8: ~0.58 SMAPE improvement, variance drops 5x

**ld=16 is definitively best for Haar on Tourism.** The monotonic improvement with increasing ld and dramatic variance reduction suggest the model is better identified at ld=16.

## Interpretation

### Why does coif3 beat Haar despite Haar's v2 family win?

Three factors likely explain the discrepancy:

1. **The v2 study tested ld=8/12 only.** At small latent dims, Haar's simplicity (1 coefficient, approximation-only) reduces the bottleneck's job. At ld=16, the bottleneck has enough capacity that coif3's richer basis (6 vanishing moments) provides better signal reconstruction.

2. **bd_label confound.** The SOTA uses eq_bcast (bd=8) while this study used eq_fcast (bd=4). This gives coif3 a 2x larger basis, enabling genuine multi-level decomposition. Haar with eq_bcast would also have bd=8, but since Haar at bd=8 is still approximation-only for an 8-sample input, it likely does not benefit as much.

3. **v2 family ranking was underpowered.** 8 runs per config with high variance (std 0.9-1.8) at ld=8/12. The "Haar wins" conclusion may reflect Haar's stability advantage at small latent dims rather than true quality superiority.

## Updated Tourism-Yearly Knowledge

- **SOTA:** TrendWaveletAELG_coif3_eq_bcast_td3_ld16, SMAPE=20.864, 95% CI [20.712, 21.016]
- **ld=16 is confirmed best** for unified TrendWavelet blocks on Tourism
- **AE = AELG** on Tourism at ld >= 12 (reverses M4 pattern)
- **Haar is NOT the best wavelet for Tourism** when latent dim and bd_label are properly controlled; coif3 with eq_bcast retains the lead

## Next Experiments (Priority Order)

1. **Non-AE Trend + HaarWaveletV3** (no AE bottleneck): On M4-Yearly, non-AE beats all AE variants. If the pattern holds on Tourism, this could leap past 20.864. Highest expected impact.

2. **TrendWaveletAE_coif3_eq_bcast_td3_ld16**: Swap AELG to AE on the winning SOTA config. Given AE's equivalence/slight edge on Tourism, this might shave 0.05-0.10 SMAPE.

3. **Alternating TrendAE + HaarWaveletV3AE at ld=16**: Tests whether architecture separation helps on Tourism's very short horizon.

4. **Haar_eq_bcast_td3_ld16** (both AE and AELG): Isolate whether coif3's advantage is the wavelet family or the larger basis_dim from eq_bcast.

## Open Questions

1. Does the non-AE backbone dominate on Tourism as it does on M4? Tourism's extremely short series (H=4, L=8) might favor AE regularization more than M4-Yearly (H=6, L=30).

2. Is coif3's advantage over Haar driven by the wavelet basis or the basis_dim (eq_bcast=8 vs eq_fcast=4)? A direct Haar_eq_bcast test would answer this.

3. Why is ld=12 dramatically worse than both ld=8 and ld=16 for Haar on Tourism? The ld=12 results show extreme instability (std=1.75) suggesting a training pathology at that specific dimension. Low priority but scientifically interesting.
