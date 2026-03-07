# Wavelet Family Selection for N-BEATS Wavelet Stacks

## Core Rule

Choose wavelet families based on **two considerations**:

1. **Target length compatibility** -- the wavelet filter must be short enough for the target length to permit real multilevel decomposition
2. **Cross-dataset performance** -- some families generalize better than others, independent of filter-length constraints

When both considerations are satisfied, **Symlet20 is the universal best choice** (avg rank 2.3/14 across M4-Yearly, Tourism-Yearly, Weather-96). When Symlet20's filter is too long for the target, fall back to the horizon-appropriate family.

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
- **Symlet20** -- avg rank 2.3/14, never worse than 3rd across 3 datasets

### If you are optimizing for a specific horizon:
- **Short (H <= 8):** DB4 > Symlet3 > DB3
- **Long (H >= 48):** Symlet20 > DB20 > Symlet10
- **Mixed/unknown:** Symlet20 (safest overall)

### Families to generally avoid:
- **DB2, Haar** -- mid-to-low performers (rank 8-10 avg), outclassed by DB3/DB4/Symlet20
- **Coif2, DB10** -- consistently underperform their respective complexity tiers
- **Coif10** -- filter too long for most practical horizons

---

## Source / Reproducibility

| Evidence | Report | Data |
|---|---|---|
| V3AELG Study (112 configs, 14 families, 4 datasets, R3 converged) | `experiments/analysis/analysis_reports/wavelet_v3aelg_study_analysis.md` | `experiments/results/m4/wavelet_v3aelg_trendaelg_study_results.csv`, `experiments/results/tourism/wavelet_v3aelg_trendaelg_study_results.csv`, `experiments/results/weather/wavelet_v3aelg_trendaelg_study_results.csv` |
| V3AELG Notebook | `experiments/analysis/notebooks/wavelet_v3aelg_trendaelg_study_analysis.ipynb` (Section 2) | See notebook data sources |
| Study 3 cross-backbone (Trend vs TrendAE) | `experiments/analysis/notebooks/wavelet_study_3_successive_insights.ipynb` | `experiments/results/m4/wavelet_study_3_successive_results.csv` |
| AELG Sweep (M4+Tourism) | `experiments/analysis/analysis_reports/v3aelg_stackheight_sweep_analysis.md` | See report data sources |
