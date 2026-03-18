---
name: omnibus_benchmark_m4_findings
description: Comprehensive omnibus benchmark of 52 unique configs on M4-Yearly (689 runs) and M4-Quarterly (182 runs) with 10 seeds each under identical training
type: project
---

## Omnibus Benchmark M4 Comprehensive (2026-03-17)

- See `experiments/analysis/analysis_reports/omnibus_m4_comprehensive_analysis.md`
- See notebook: `experiments/analysis/notebooks/omnibus_m4_comprehensive_analysis.ipynb`
- **871 rows, 69 named configs (52 unique after dedup), Yearly + Quarterly, zero divergences on Yearly, 1 bimodal on Quarterly.**

**Data quality issues:**
- **forecast_basis_dim sweep FAILED:** `_bd32/_bd64/_bd128` produce run-level identical results. Parameter not applied. 6 config groups affected. Implementation bug needs fixing.
- **Naming aliases:** `NBEATS-I-activeG` = `NBEATS-I-baseline` (active_g has no effect on I blocks). `SynWavelet` skip = DB3 skip (identical runs).

**Key findings (updated from prior analysis):**
- Top 10 Yearly within OWA 0.003. Statistical tie (all MWU p>0.23).
- **M4-Yearly best:** NBEATS-I+G-activeG (OWA=0.8026, 36M) and TrendAELG+Coif2WaveletV3AELG-30 (OWA=0.8027, 2.9M) -- equivalent quality, 12x param difference.
- **M4-Quarterly best:** Trend+Coif2WaveletV3-30 (OWA=0.8850, 15.4M).
- **Cross-period champion:** Trend+Coif2WaveletV3-30 (top 4 Yearly, #1 Quarterly).
- **active_g is a non-factor:** Hurts 11/15 comparisons, 0/15 significant. Do NOT enable by default.
- **Backbone: RootBlock = AELG >> AE** (KW p<0.0001). AELG learned gate essential.
- **Wavelet family: non-factor for AELG** (KW p=0.79). Bottleneck homogenizes basis.
- **Skip connections: reduce variance, do not improve mean.** Not recommended for stable architectures.
- **NBEATS-G unstable on Quarterly:** 1/10 bimodal divergence (seed 50, SMAPE=34). activeG prevents it.
- **Pareto frontier:** TrendWaveletAELG-10 (0.44M, OWA=0.808) is 82x more param-efficient than NBEATS-I+G with only +0.6% OWA gap.
- **Most stable:** TrendAE+DB3WaveletV3AE-30-activeG (std=0.0045). AE bottleneck regularizes training.

**Why:** First apples-to-apples comparison across both M4-Yearly and M4-Quarterly under uniform training.
**How to apply:** Use Trend+Coif2WaveletV3-30 for cross-period robustness. Use TrendAELG+Coif2WaveletV3AELG-30 for quality-first M4-Yearly. Use TrendWaveletAELG-10 for resource-constrained.
