# Wavelet Family Selection for N-BEATS Wavelet Stacks

## Core Rule

Choose wavelet families based on **two considerations**:

1. **Target length compatibility** -- the wavelet filter must be short enough for the target length to permit real multilevel decomposition
2. **Cross-dataset performance** -- some families generalize better than others, independent of filter-length constraints

### Updated Recommendation (Comprehensive Sweep, 2026-04-06)

The comprehensive sweep (112 configs x 10 runs, M4 all periods + Tourism + Weather + Milk) found that **db3 is the safest cross-dataset default** (ranks 1st or 2nd on 3/4 datasets). The earlier Symlet20 recommendation was based on a smaller V3AELG study with fewer configs. Key findings:

| Dataset | Best Wavelet | KW p-value |
|---------|-------------|-----------|
| M4-Yearly | coif2 | 0.013 |
| M4-Monthly/Hourly | Haar | <0.01 |
| Tourism | db3 (non-AE), sym10 (AELG) | 0.049 |
| Weather | db3 | 0.036 |
| Milk | Haar | 0.458 (ns) |

**Short horizons (H<=8):** Haar or db3. **Medium/long horizons:** db3 or coif2. **Cross-dataset default:** db3.

When Symlet20's filter is compatible with the target length, it remains a strong choice for long-horizon datasets, but db3 is now the recommended first-choice for new experiments.

### Refinement (M4 Paper-Sample Sweep, 2026-04-27)

53 configs × 6 periods × 10 runs under `sampling_style=nbeats_paper`. See `experiments/analysis/analysis_reports/comprehensive_m4_paper_sample_analysis.md`.

| M4 Period | Best wavelet (paper-sample) | Best wavelet (sliding) | Agreement |
|---|---|---|---|
| Yearly    | db3 (`TALG+DB3V3ALG_10s_ag0`) | coif2 | family-level (alt-T+W) |
| Quarterly | n/a (paper baseline wins) | n/a (paper baseline wins) | yes |
| Monthly   | haar (`TW_30s_td3_bdeq_haar`) | coif2 | weak — both small wins |
| Weekly    | coif2 (`T+Coif2V3_30s_bdeq`) | db3 | family-level (T+W RB) |
| Daily     | coif2 (`TAELG+Coif2V3ALG_30s_ag0`) | (paper baseline) | n/a |
| Hourly    | n/a (paper baseline wins) | n/a (paper baseline wins) | yes |

**Best M4 generalist (paper-sample):** `T+Sym10V3_30s_bdeq` (mean rank 6.83/53, top-3 on 3 periods, top-15 on all 6). Corroborates `T+HaarV3_30s_bd2eq` from the sliding sweep — **haar / sym10 are the universal M4 generalist wavelets**.

**M4 default shortlist (both protocols): haar, db3, coif2, sym10.**

### Coif3 verdict on M4 (2026-04-27)

**Coif3 unlocks NO new per-period SOTA.** Tested on M4 paper-sample sweep across `TW_{10s,30s}_td3_bdeq_coif3` and `T+Coif3V3_30s_bdeq`. Best result: family-internal wins on TW-10s for Monthly (+0.09 vs db3) and Weekly (+0.05 vs db3) — both within 1 std and beaten outright by `TW_30s_td3_bdeq_haar` and `T+Coif2V3_30s_bdeq` at the cross-family level. Coif3's 6 vanishing moments and longer support add no value over coif2 (2 VM) or sym10 on M4 horizons.

**Drop coif3 from default M4 wavelet shortlist.** (Tourism question still open — separate AELG_coif3_eq_bcast_td3_ld16 SOTA finding has not been retested.)

---

## Cross-Dataset Performance Rankings (V3AELG Study, 2026-03-07)

Ranked by average rank across 3 datasets (M4-Yearly, Tourism-Yearly, Weather-96), R3 converged data (50 epochs), 14 families tested:

| Rank | Family | M4 Rank | Tourism Rank | Weather Rank | Avg Rank | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | **Symlet20** | 3 | 2 | 2 | **2.3** | Universal best -- near-linear phase preserves temporal structure |
| 2 | DB4 | 1 | 1 | 14 | 5.3 | Excellent on short horizons, catastrophic on long |
| 3 | Symlet3 | 4 | 5 | 7 | 5.3 | Consistent mid-tier |
| 4 | Coif3 | 6 | 4 | 8 | 6.0 | |
| 5 | DB3 | 5 | 6 | 9 | 6.7 | |
| 6-14 | Others | -- | -- | -- | 7.0-11.3 | See full report |

**Key pattern:** DB4 is #1 on both short-horizon datasets (M4 H=6, Tourism H=4) but dead last (#14) on Weather-96 (H=96). DB20 shows the inverse (#1 Weather, #12-13 short-horizon). Symlet20 transcends this tradeoff due to its near-linear phase response.

---

## Practical Thresholds (Filter Length Compatibility)

Rule of thumb: you need `target_length >= 2 * (dec_len - 1)` for at least one non-boundary DWT level.

| Family | Example blocks | Filter length (`dec_len`) | Rough minimum target length for level >= 1 |
|---|---|---:|---:|
| Haar | `HaarWaveletV3*` | 2 | 2 |
| DB2 / Sym2 | `DB2WaveletV3*`, `Symlet2WaveletV3*` | 4 | 6 |
| DB3 / Sym3 / Coif1 | `DB3WaveletV3*`, `Symlet3WaveletV3*`, `Coif1WaveletV3*` | 6 | 10 |
| DB4 | `DB4WaveletV3*` | 8 | 14 |
| Coif2 | `Coif2WaveletV3*` | 12 | 22 |
| Coif3 | `Coif3WaveletV3*` | 18 | 34 |
| DB10 / Sym10 | `DB10WaveletV3*`, `Symlet10WaveletV3*` | 20 | 38 |
| DB20 / Sym20 | `DB20WaveletV3*`, `Symlet20WaveletV3*` | 40 | 78 |
| Coif10 | `Coif10WaveletV3*` | 60 | 118 |

In WaveletV3, if `pywt.dwt_max_level(target_length, dec_len) == 0`, the code keeps `level=0` instead of forcing an invalid level-1 decomposition. Short targets with long filters therefore get an approximation-only orthonormal basis rather than a true multiscale wavelet decomposition.

---

## Selection Guide by Forecast Horizon

### Very short horizons / targets (<= 8)
- **Best:** DB4 (rank #1 on M4 and Tourism), DB3, Symlet3
- **Safe fallback:** Haar, DB2, Sym2
- **Avoid:** DB10+, Sym10+, DB20+, Sym20+, Coif2+ (filter too long for real decomposition)
- **Note:** Symlet20 still works (approximation-only basis) and ranks #2-3 on these horizons despite not getting multiscale decomposition

### Short targets (9-16)
- Good defaults: DB3, DB4, Sym3, Coif1
- Symlet20 viable if backcast_length >= 78

### Medium targets (17-37)
- Safe: DB3, DB4, Sym3, Coif1, Coif2, Coif3
- DB10 / Sym10 become reasonable near the top of this range

### Long targets (38-77)
- DB10 and Sym10 are appropriate
- DB20 / Sym20 viable near the top of this range

### Very long targets (>= 78)
- **Best:** Symlet20 (rank #1-2 universally), DB20 (rank #1 on Weather-96)
- DB10, Sym10, Coif3 also strong
- Coif10 still needs ~118+

---

## Important Practical Note

N-BEATS often uses `backcast_length = 4 * forecast_length` or `backcast_length = 5 * forecast_length`.

So a configuration can have:
- a **reasonable backcast wavelet basis**, but
- a **forecast basis that is too short** for the same family to behave multiscale

Example:
- `forecast_length=6`, `backcast_length=24`
- `DB20WaveletV3` will still be far too long for the forecast basis

When in doubt, pick the family using the **shorter of the two target lengths**.

---

## Recommendation Summary

### If you must pick ONE family (any horizon):
- **db3** -- safest cross-dataset default (comprehensive sweep, 2026-04-06). Ranks 1st or 2nd on Tourism, Weather, Milk.

### If you are optimizing for a specific horizon:
- **Short (H <= 8):** Haar or db3 (Haar wins Milk; db3 wins Tourism)
- **Medium (H = 13-18):** coif2 or db3 (coif2 wins M4-Yearly/Monthly)
- **Long (H >= 48):** Haar or db3 (Haar wins M4-Hourly; Symlet20 viable if filter fits)
- **Mixed/unknown:** db3 (safest overall per comprehensive sweep)

### Families to generally avoid:
- **Coif10** -- filter too long for most practical horizons
- **sym10** -- inconsistent (good on Tourism AELG, poor elsewhere)

---

## Exception: TrendWaveletAELG (Unified Block)

**Wavelet family selection does NOT apply to TrendWaveletAELG.** In the unified block (which combines trend + wavelet in a single AE bottleneck), wavelet family is a non-factor (Kruskal-Wallis p=0.107 across 14 families on M4-Yearly). The AE bottleneck homogenizes basis representations.

For TrendWaveletAELG, use any reasonable family. Coif3 has the best cross-dataset average rank (1.5). See the `trendwavelet-block-selection` skill for details.

---

## Source / Reproducibility

| Evidence | Report | Data |
|---|---|---|
| V3AELG Study (112 configs, 14 families, 4 datasets, R3 converged) | `experiments/analysis/analysis_reports/wavelet_v3aelg_study_analysis.md` | `experiments/results/m4/wavelet_v3aelg_trendaelg_study_results.csv`, `experiments/results/tourism/wavelet_v3aelg_trendaelg_study_results.csv`, `experiments/results/weather/wavelet_v3aelg_trendaelg_study_results.csv` |
| V3AELG Notebook | `experiments/analysis/notebooks/wavelet_v3aelg_trendaelg_study_analysis.ipynb` (Section 2) | See notebook data sources |
| Study 3 cross-backbone (Trend vs TrendAE) | `experiments/analysis/notebooks/wavelet_study_3_successive_insights.ipynb` | `experiments/results/m4/wavelet_study_3_successive_results.csv` |
| AELG Sweep (M4+Tourism) | `experiments/analysis/analysis_reports/v3aelg_stackheight_sweep_analysis.md` | See report data sources |
