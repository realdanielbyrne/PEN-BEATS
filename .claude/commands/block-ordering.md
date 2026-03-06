# N-BEATS Block Ordering Guidelines

## Rule: Always Put Trend Blocks First

**Evidence**: VAE2 study (M4-Yearly + Weather-96), 11 matched ordering pairs across AELG, VAE1, and VAE2 backbone families. Wilcoxon signed-rank tests with Bonferroni correction (α = 0.05/11 = 0.00455).

| Dataset | Pairs significant (α=0.05) | Survive Bonferroni | Direction |
|---|---|---|---|
| M4-Yearly | 5/11 | 4/11 | All favour Trend-first |
| Weather-96 | 3/11 | 2/11 | All favour Trend-first |

**No pair in any configuration showed Generic/Wavelet-first to be significantly better.**

---

## Ordering Recommendations by Stack Type

### Two-block stacks
| Correct order | Wrong order | Effect (M4 OWA delta) |
|---|---|---|
| `TrendAELG → GenericAELG` | `GenericAELG → TrendAELG` | ~0.013 OWA |
| `TrendAELG → DB4WaveletV3AELG` | `DB4V3AELG → TrendAELG` | ~0.036 OWA (**Bonferroni-significant**, r=0.83) |

### Three-block stacks
| Correct order | Wrong order | Effect |
|---|---|---|
| `Trend → Seasonality → Generic` | `Generic → Seasonality → Trend` | ~0.016 OWA |
| `Trend → Wavelet → Generic` | `Generic → Wavelet → Trend` | Moderate |

### VAE/VAE2 backbones — ordering is **critical**
For VAE-family blocks, wrong ordering can 2–3× the OWA on M4 (e.g., mean OWA 3.34 reversed vs 2.55 correct). The stochastic bottleneck amplifies ordering errors.

---

## Why This Happens (Mechanism)

N-BEATS uses residual connections: each block receives `input - sum(previous backcasts)`. Trend-first is correct because:

1. Trend blocks have strong inductive bias (polynomial basis) — they efficiently extract low-frequency structure from the raw signal
2. Subsequent blocks (Generic, Wavelet) then model the detrended residual, which is a better-conditioned problem
3. Generic/Wavelet-first forces a flexible block to "guess" which part of the raw signal to model, leaving a messier residual for the Trend block

---

## Practical Implication for Study Design

**Block ordering is not a hyperparameter to search over.** Don't waste successive halving rounds on reversed configs. When designing a new study:

- Always place `Trend*` stacks before `Generic*`, `Wavelet*`, or `Seasonality*` stacks
- The N-BEATS-I ordering (`Trend → Seasonality → Generic`) is the correct default for interpretable stacks
- This guideline holds across all tested backbone families: AELG, VAE1, VAE2

---

## Source / Reproducibility

- **Config**: `experiments/configs/vae2_study_m4_weather.yaml`
- **Data**: `experiments/results/m4/vae2_study_results.csv`, `experiments/results/weather/vae2_study_results.csv`
- **Notebook**: `experiments/analysis/notebooks/vae2_study_insights.ipynb` — Section "Block Ordering: Does Sequence Matter?" (cells `q11-ordering-*`)
- **Report**: `experiments/analysis/analysis_reports/vae2_study_insights.md`
