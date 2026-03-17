---
name: omnibus_benchmark_m4_findings
description: First head-to-head omnibus benchmark of 36 configs on M4-Yearly with 10 seeds each under identical training (200ep, patience=20)
type: project
---

## Omnibus Benchmark M4-Yearly (2026-03-16)

- See `experiments/analysis/analysis_reports/omnibus_benchmark_m4_analysis.md`
- See notebook: `experiments/analysis/notebooks/omnibus_benchmark_m4_insights.ipynb`
- **356 runs, 36 configs, 10 seeds, zero divergences. Only Yearly complete (no Quarterly yet).**
- **1 config incomplete:** TrendAELG+Coif2WaveletV3AELG-30-activeG (1/10 runs). 11 YAML configs have no results.

**Key findings:**
- Top 10 separated by only 0.097 SMAPE (13.508-13.605). No pairwise top-5 differences are significant (all MWU p>0.57).
- **NBEATS-I+G-activeG is best paper baseline** (SMAPE=13.508, 36M params).
- **TrendAELG+Coif2WavV3AELG-30 is best novel config** (SMAPE=13.521, 2.9M params, 10 runs).
- **TrendWaveletAELG-10 is most parameter-efficient** (SMAPE=13.579, 436K params = 82x fewer than I+G).
- **active_g is NOT beneficial for most configs.** Hurts NBEATS-G significantly (p=0.049). Zero effect on Trend/Seasonality (bit-identical runs). Only potentially helps NBEATS-I+G (p=0.16, ns).
- **Family hierarchy: novel_nonae >= paper_baseline ~ novel_aelg >> novel_ae.**
- **Backbone hierarchy confirmed: AELG >= RootBlock > AERootBlock** (matched Coif2 30-stack comparison).
- **DB3 vs Coif2: no significant difference** on M4-Yearly.
- **Stability: wavelet+AE configs are most stable** (std 0.054-0.097). Generic/Bottleneck AE are most unstable (std 0.215-0.337).
- **No config reaches prior SOTA of 13.410** (wavelet_study_2, 50 fixed epochs, 3 seeds). Training protocol difference (early stopping) is likely cause.

**Why:** This is the first apples-to-apples comparison under uniform training conditions. Prior studies used different epoch counts, seeds, and stopping criteria.
**How to apply:** Use these rankings as the ground truth for M4-Yearly architecture recommendations. Prior SOTA of 13.410 may be a protocol artifact — needs 50-epoch fixed-training confirmation run.
