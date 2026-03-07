# ResNet Skip Connection Study Analysis

**Dataset:** M4-Yearly (H=6, L=30)
**Date:** 2026-03-07
**Method:** 3-round successive halving (24 configs -> 12 -> 6)
**Results CSV:** `experiments/results/m4/resnet_skip_study_results.csv`
**Notebook:** `experiments/analysis/notebooks/resnet_skip_study_insights.ipynb`

---

## Executive Summary

This study tests whether ResNet-style skip connections (periodic re-injection of `alpha * original_input` into the residual stream) improve forecasting for deep N-BEATS architectures. Three architectures were tested across stack depths of 8-30, with skip_distance from 4-10 and both fixed (0.1) and learnable alpha.

**Main findings:**
1. Skip connections provide **dramatic benefit** for GenericAELG at depth >= 20, rescuing it from bimodal convergence failure (SMAPE 36 -> 13.8).
2. Skip connections provide **negligible benefit** for TrendAELG+WaveletV3AELG and Generic, which do not suffer from depth degradation on this dataset.
3. The study winner is TW16_skip4_learn (SMAPE 13.521, OWA 0.802), but it does not surpass the prior M4-Yearly SOTA of SMAPE 13.410 (Coif2_bd6_eq_fcast_td3 from the basis dimension study).
4. Skip connections should **not be a default setting** -- they are a targeted remedy for architecturally unstable configurations.

---

## 1. Successive Halving Funnel

| Round | Configs | Seeds | Epochs | Description |
|-------|---------|-------|--------|-------------|
| 1 | 24 | 3 | 10 | All configs; keep top 50% |
| 2 | 12 | 3 | 25 | Survivors; keep top 50% |
| 3 | 6 | 5 | ~68 | Finalists; early stopping patience=10 |

### Round 1 -> 2 Elimination

TrendWav configs dominated Round 1 (top 5 positions). GenericAELG no-skip baselines at depth >= 20 scored catastrophically high SMAPE (36-49) due to bimodal convergence failure. Three GAELG30 skip configs survived to Round 2 despite high mean SMAPE, because their "good" seeds showed strong convergence potential.

### Round 2 -> 3 Elimination

By Round 2, the field narrowed to TW and G30 configs, plus GAELG30_skip5_a01/learn. The GAELG30 skip configs achieved competitive SMAPE (~13.8) but were eliminated in favor of slightly better TW configs. The final 6 comprised 4 TW and 2 G30 configs.

---

## 2. Depth Degradation

### GenericAELG: Catastrophic Instability

| Depth | Round | SMAPE Mean | Std | Individual Seeds | Convergence |
|-------|-------|-----------|-----|-----------------|-------------|
| 10 | R1 | 19.136 | 1.615 | 21.0, 18.5, 17.9 | Stable |
| 20 | R1 | 36.543 | 18.491 | 48.5, 15.2, 45.9 | **Bimodal** |
| 30 | R1 | 36.027 | 18.164 | 43.9, 15.3, 48.9 | **Bimodal** |

At depth >= 20, GenericAELG exhibits a bimodal failure mode: approximately 2/3 of seeds get trapped in a loss plateau at SMAPE ~45-49, while 1/3 converge normally to ~15. This is not random noise -- it is a systematic convergence failure linked to the AE bottleneck + learned gate architecture at depth.

### TrendAELG + WaveletV3AELG: Monotonic Improvement

| Depth | Best Round | SMAPE Mean | Std | Params |
|-------|-----------|-----------|-----|--------|
| 8 | R1 | 17.519 | 1.752 | 759,948 |
| 16 | R2 | 13.755 | 0.276 | 1,519,896 |
| 24 | R3 | 13.557 | 0.058 | 2,279,844 |

Performance improves monotonically with depth. The interpretable block structure (alternating Trend polynomial + Wavelet DWT bases) provides stable gradient pathways regardless of depth.

### Generic (30 stacks): Stable

G30_no_skip reached Round 3 with SMAPE 13.617 -- fully competitive. The paper-standard Generic architecture does not suffer from signal decay at 30 stacks on M4-Yearly (H=6, L=30).

---

## 3. Skip Connection Effectiveness

### GenericAELG: Skip Connections as Rescue

| Config | R1 SMAPE | R2 SMAPE | Status |
|--------|---------|---------|--------|
| GAELG30_no_skip | 36.027 +/- 18.164 | eliminated | Catastrophic |
| GAELG30_skip5_a01 | 26.268 +/- 17.727 | **13.792 +/- 0.099** | Rescued |
| GAELG30_skip5_learn | 25.860 +/- 18.143 | **13.879 +/- 0.043** | Rescued |
| GAELG30_skip10_learn | 25.534 +/- 18.080 | 24.299 +/- 17.893 | Partial rescue |

**Skip=5 fully rescues GAELG30** by providing 5 injection points (stacks 5,10,15,20,25). By 25 epochs, all 3 seeds converge. Skip=10 (only 2 injection points) is insufficient -- one seed remains stuck.

### TrendWav: No Meaningful Benefit

At Round 3 (5 seeds, ~68 epochs):

| Config | SMAPE | Delta vs No-Skip |
|--------|-------|-----------------|
| TW16_skip4_learn | 13.521 | n/a (no TW16 no-skip in R3) |
| TW24_no_skip | 13.557 | baseline |
| TW24_skip8_learn | 13.614 | +0.058 (+0.4%) |
| TW24_skip4_a01 | 13.630 | +0.073 (+0.5%) |

Skip connections do not improve TW24 performance. The TW16_skip4_learn win over TW24_no_skip is within noise (p=0.42) and may simply reflect that 16 stacks is sufficient.

### Generic: No Benefit

| Config | R3 SMAPE |
|--------|---------|
| G30_no_skip | 13.617 |
| G30_skip10_learn | 13.653 |

Skip connections marginally hurt Generic performance.

---

## 4. Optimal Skip Distance

For GenericAELG 30-stack (the only architecture where skip matters):

| Skip Distance | Injections | R2 SMAPE | All Seeds Converged? |
|--------------|-----------|---------|---------------------|
| 5 | 5 | 13.792 | Yes (all 3) |
| 10 | 2 | 24.299 | No (1/3 stuck) |

**Rule of thumb:** `skip_distance = floor(n_stacks / 6)` provides adequate injection density.

---

## 5. Fixed vs Learnable Alpha

| Comparison | Round | Fixed (0.1) | Learnable | Winner |
|-----------|-------|------------|-----------|--------|
| TW16 skip=4 | R2 | 13.750 | 13.780 | Fixed (+0.030) |
| TW24 skip=4 | R1 | 14.587 | 15.106 | Fixed (+0.520) |
| GAELG30 skip=5 | R2 | 13.792 | 13.879 | Fixed (+0.087) |
| G30 skip=5 | R1 | 16.630 | 16.570 | Learnable (-0.060) |
| G30 skip=10 | R1 | 15.144 | 25.097 | Fixed (+9.953) |

**Fixed alpha wins 4/5 comparisons.** Learnable alpha adds an optimization degree of freedom that does not pay off within the training budgets tested. The G30_skip10 learnable case is extreme: the learned alpha destabilized one seed.

**Recommendation:** Use `skip_alpha = 0.1` (fixed).

---

## 6. Legacy Generic Rehabilitation

**Result: Failed.** Skip connections do not rehabilitate the 30x Generic architecture:

| Config | SMAPE | Params | SMAPE/MParam |
|--------|-------|--------|-------------|
| TW16_skip4_learn | 13.521 | 1.5M | 8.90 |
| TW24_no_skip | 13.557 | 2.3M | 5.95 |
| G30_no_skip | 13.617 | 24.7M | 0.55 |
| G30_skip10_learn | 13.653 | 24.7M | 0.55 |

G30 is 16x less parameter-efficient than TW16 while achieving worse SMAPE. The fundamental problem is not signal decay -- it is the absence of inductive bias in Generic blocks.

---

## 7. MetaForecaster Accuracy

| Metric | Value |
|--------|-------|
| Spearman rho (meta rank vs actual R1 rank) | 0.479 (p=0.018) |
| Spearman rho (meta rank vs actual best rank) | 0.489 (p=0.015) |
| Top-12 overlap with actual R2 survivors | 9/12 (75%) |
| Rank of eventual winner (TW16_skip4_learn) | 15th of 24 |

The MetaForecaster shows moderate predictive power but significantly underestimated slow-converging configs. It ranked the eventual study winner 15th. The MetaForecaster is useful as a soft prior but should not drive elimination decisions alone.

---

## 8. Final Leaderboard

| Rank | Config | SMAPE | Std | OWA | Params | Architecture |
|------|--------|-------|-----|-----|--------|-------------|
| 1 | TW16_skip4_learn | 13.521 | 0.057 | 0.802 | 1.5M | TrendAELG+WaveletV3AELG |
| 2 | TW24_no_skip | 13.557 | 0.058 | 0.804 | 2.3M | TrendAELG+WaveletV3AELG |
| 3 | TW24_skip8_learn | 13.614 | 0.089 | 0.809 | 2.3M | TrendAELG+WaveletV3AELG |
| 4 | G30_no_skip | 13.617 | 0.092 | 0.811 | 24.7M | Generic |
| 5 | TW24_skip4_a01 | 13.630 | 0.137 | 0.810 | 2.3M | TrendAELG+WaveletV3AELG |
| 6 | G30_skip10_learn | 13.653 | 0.258 | 0.814 | 24.7M | Generic |

No pairwise comparison reaches p < 0.05 significance (Mann-Whitney U). All 6 finalists are within a 0.132 SMAPE range.

**Comparison to prior SOTA:** The study winner (SMAPE 13.521, OWA 0.802) does not beat the prior M4-Yearly SOTA from the basis dimension study: Coif2_bd6_eq_fcast_td3 (SMAPE 13.410, OWA 0.794). The prior SOTA used non-AE Trend+WaveletV3 blocks.

---

## 9. Recommendations

### Current Best Configuration (M4-Yearly)

**Unchanged:** Coif2_bd6_eq_fcast_td3 (Trend+WaveletV3, non-AE) remains the M4-Yearly SOTA at SMAPE 13.410, OWA 0.794.

### Skip Connection Usage Guidelines

1. **Default: OFF.** Skip connections should not be enabled by default.
2. **Enable for GenericAELG at depth >= 16:** Use `skip_distance = floor(n_stacks / 6)`, `skip_alpha = 0.1`.
3. **Do not use for TrendWav or Generic:** No measurable benefit.
4. **If in doubt:** The cost of enabling skip connections is near-zero (1 extra parameter for learnable, 0 for fixed), so enabling them on deep stacks is a low-risk hedge.

### What to Test Next

1. **Skip connections on non-AE Trend+WaveletV3 at 16-24 stacks** with Coif2 wavelet, bd=6, td=3. This would test whether skip connections can push the actual SOTA config further.

2. **Longer training for GenericAELG skip configs.** The GAELG30_skip5_a01 config (SMAPE 13.792 at Round 2, 25 epochs) was eliminated before reaching full convergence. A dedicated run at 100+ epochs might show competitive results.

3. **Skip connections on Traffic/Weather datasets.** These datasets have much longer sequences (L=480, H=96 for Traffic-96) where signal decay through 16-24 stacks may be a real issue.

### Open Questions

- **Why is GenericAELG bimodally unstable at depth?** The learned gate + AE bottleneck combination creates a sharp loss landscape where gradient flow depends critically on initialization. Skip connections smooth this landscape by providing "shortcut" gradient pathways. A more targeted fix might be better initialization or gate warmup.

- **Would `sum_losses=True` help stability?** The backcast reconstruction loss (Section 3.3 of the paper) explicitly encourages gradient flow to all stacks. This might achieve the same stabilization as skip connections without the architectural modification.

- **Are 68 epochs enough?** Round 3 ran ~68 epochs with patience=10. The cosine annealing schedule with warmup_epochs=15 might benefit from 100+ epochs. TW16_skip4_learn peaked at epoch ~47, suggesting it was not fully converged.
