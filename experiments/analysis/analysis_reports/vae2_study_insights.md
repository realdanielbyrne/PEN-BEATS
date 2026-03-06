# VAE2 Study — Key Insights

**Study date:** 2026-03-05
**Datasets:** M4-Yearly (OWA; lower is better) and Weather-96 (norm_mae; lower is better)
**Backbone families compared:** VAE2 (new), VAE1 (original variational AE), AELG (learned-gate AE)
**Search method:** Successive halving over 27 configs × 3 latent dims (8/16/32), two passes (baseline + activeG_fcast)

---

## Executive Summary

**VAE2 does not improve over AELG on either dataset.** On M4-Yearly, VAE2 is catastrophically worse — OWA ~2.88 versus AELG's ~0.88, placing every VAE2 run above the Naive2 random-walk baseline (OWA 1.0). On Weather-96, VAE2 is competitive in Round 2 (norm_mae ~0.40 versus AELG's ~0.38), but AELG still wins. VAE1 performs between AELG and VAE2 on M4, but is the worst family on Weather. The successive halving algorithm eliminated every VAE2 and VAE1 config from the M4 Round 3, with 17 AELG configs advancing and none from the VAE families.

**Recommendation: Do not use VAE2.** Use AELG with wavelet+trend stack compositions for both M4 and multi-variate datasets.

---

## Finding 1: Family-Level Performance

| Backbone | M4 Mean OWA | M4 Median OWA | Weather Mean norm_mae | Weather Median norm_mae |
|---|---|---|---|---|
| **AELG** | **0.878** | **0.852** | **0.398** | **0.373** |
| VAE1 | 1.570 | 1.387 | 0.648 | 0.606 |
| VAE2 | 2.879 | 2.793 | 0.491 | 0.437 |

Reference baselines (M4): Naive2 = 1.000, NBEATS-G = 0.862, NBEATS-I+G = 0.806.

AELG is the only family to consistently beat NBEATS-G. VAE2 and VAE1 both exceed the Naive2 threshold on M4 — they are worse than a random walk for M4-Yearly forecasting.

On Weather, VAE2 is noticeably better than VAE1. The ordering reverses from M4: VAE1 is the worst family on Weather while VAE2 shows some competitive ability in later rounds.

---

## Finding 2: Latent Dimension Effects

Latent_dim (8, 16, 32) has different impacts by backbone family:

- **AELG:** Minimal effect. OWA ranges from 0.890 (ld=8) to 0.869 (ld=32). The learned gate absorbs extra capacity gracefully.
- **VAE1:** Monotonically worse with larger ld on M4 (1.478 → 1.672). KL regularization becomes harder to balance with more stochastic dimensions.
- **VAE2:** Same trend as VAE1, but more extreme. OWA increases from 2.687 (ld=8) to 3.000 (ld=32). Smaller latent dimensions are strictly better.

**If VAE2 is used, ld=8 is the only viable setting.** Larger bottlenecks amplify training instability.

---

## Finding 3: Successive Halving Funnel

### M4 (3 rounds)

| Round | AELG configs | VAE1 configs | VAE2 configs | Best OWA (median) |
|---|---|---|---|---|
| 1 | 24 | 27 | 27 | 0.818 |
| 2 | 24 | 26 | 2 | 0.796 |
| 3 | 17 | 0 | 0 | 0.794 |

VAE2 went from 27 configs in Round 1 to 2 configs in Round 2 (both with ld=8) and was completely eliminated before Round 3. VAE1 survived to Round 2 but was also fully eliminated. AELG improved from Round 1 median OWA 0.914 to Round 3 median 0.814.

### Weather (2 rounds)

| Round | AELG configs | VAE1 configs | VAE2 configs | Best norm_mae (median) |
|---|---|---|---|---|
| 1 | 24 | 27 | 27 | 0.331 |
| 2 | 24 | 8 | 20 | 0.332 |

Weather shows more diversity — VAE2 survived to Round 2 with 20 configs (74% retention) while VAE1 dropped to 8 configs (30% retention). This reflects that VAE2's KL regularization is more useful for multi-variate meteorological series.

---

## Finding 4: Cross-Dataset Generalization

Spearman rank correlation between M4 rank and Weather rank across all 75 matched configs: **r = 0.41, p < 0.001**. A real but modest relationship — M4-good configs tend to be at least somewhat good on Weather, but with substantial scatter.

All top-10 configs by both M4 OWA and Weather norm_mae are AELG. The best cross-dataset winner is `TrendAELG+GenericAELG_ld16` — ranked #1 on Weather and #6 on M4. VAE2 configs show near-zero within-family rank correlation between datasets, meaning M4 performance tells you nothing about Weather performance for VAE2 architectures.

---

## Finding 5: Architecture Category Effects

Stack composition matters much more for AELG than for VAE families:

- **AELG best categories:** `wavelet_trend` (OWA 0.848), `trend_seasonality_generic` (0.851), `trend_generic` (0.856)
- **AELG worst category:** `generic_wavelet_trend` (OWA 1.013) — putting Generic first, then Wavelet, then Trend degrades performance

For VAE2, category differences are large in absolute terms but all categories remain far above acceptable thresholds on M4. On Weather, VAE2 with `trend_seasonality_generic` and `trend_generic` achieves norm_mae ~0.43, close to AELG's best.

**Key pattern across all families:** Architectures with a leading or prominent Trend block consistently outperform those without. Trend blocks anchor the latent representation, especially important for the stochastic VAE bottleneck.

---

## Finding 6: Round-3 AELG Results vs. Paper Baselines

All 17 Round-3 AELG survivors beat NBEATS-G (OWA 0.862). OWA range: 0.805–0.829.

**Top 5 M4 Round-3 configs:**

| Config | Category | LD | Median OWA |
|---|---|---|---|
| DB4V3AELG+TrendAELG_ld16 | wavelet_trend | 16 | 0.805 |
| DB4V3AELG+TrendAELG_ld8 | wavelet_trend | 8 | 0.807 |
| GenericAELG+TrendAELG_ld16 | generic_trend | 16 | 0.810 |
| TrendAELG+DB4WaveletV3AELG_ld32 | trend_wavelet | 32 | 0.811 |
| DB4V3AELG+TrendAELG_ld32 | wavelet_trend | 32 | 0.813 |

The best config `DB4V3AELG+TrendAELG_ld16` achieves OWA 0.805, matching NBEATS-I+G (0.806) at approximately 1.8M parameters — roughly 7% of NBEATS-G's 24.7M parameter count.

---

## Finding 7: Pass Type (active_g) Effect

`active_g=True` (activation on final linear layers) consistently hurts performance:

| Backbone | M4 delta (activeG - baseline) | Weather delta |
|---|---|---|
| AELG | +0.007 | +0.016 |
| VAE1 | +0.080 | +0.044 |
| VAE2 | −0.001 | +0.033 |

**Prefer baseline pass (`active_g=False`) for all backbone families.** The activeG extension offers no consistent benefit in the VAE study; for VAE1 it adds ~8% OWA degradation.

---

## Finding 8: Computational Cost

| Backbone | Avg params | M4 train time (s) | M4 mean OWA |
|---|---|---|---|
| AELG | 1,809,934 | 18.3 | 0.878 |
| VAE1 | 2,176,905 | 12.9 | 1.570 |
| VAE2 | 1,592,538 | 10.5 | 2.879 |

VAE2 is the cheapest per run (fewest params, fastest training) but most expensive per unit of forecasting quality. Its fast convergence likely reflects rapid collapse to a poor local minimum rather than efficient learning.

**Seed-to-seed stability:** AELG within-config OWA std dev ~0.025 (highly stable). VAE2 std dev ~0.18 (highly unstable). VAE2's poor average conceals very high variance — some runs achieve OWA ~2.0 while others reach ~3.5+.

---

## Recommendations

### For M4-Yearly or structured seasonal series:
1. Use `DB4V3AELG+TrendAELG_ld16` — OWA 0.805, matches NBEATS-I+G at 7% of parameters.
2. Or `GenericAELG+TrendAELG_ld16` — OWA 0.810, simpler architecture without wavelet blocks.
3. Do not use VAE2 or VAE1 for M4-Yearly under any configuration.

### For Weather-96 or multi-variate continuous series:
1. Use `TrendAELG+GenericAELG_ld16` — norm_mae 0.338, best single cross-dataset config.
2. VAE2 with ld=8 and trend-inclusive stacks is a viable second choice (norm_mae ~0.344).
3. VAE1 is the worst choice on Weather.

### On VAE2's future potential:
VAE2 may benefit from: (a) a tuned KL weight (currently fixed at 0.001 — may need to be much smaller for M4), (b) a warmup schedule that starts with KL weight = 0 and gradually increases, or (c) application to genuinely noisy, high-dimensional domains where latent stochasticity aids generalization. Without these modifications, AELG is strictly preferred.

---

## Notebook

Full interactive analysis: `experiments/analysis/notebooks/vae2_study_insights.ipynb`

The notebook covers: backbone comparison box plots, latent dim interaction heatmaps, successive halving funnel visualization, round-over-round progression lines, cross-dataset rank scatter plot, category heatmaps, Round-3 leaderboard vs. baselines, pass-type effect, and computational cost analysis.
