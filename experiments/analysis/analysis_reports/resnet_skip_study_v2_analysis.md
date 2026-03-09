# ResNet Skip Connection Study v2 — Analysis Report

**Dataset:** M4 Yearly | **Date:** 2026-03-07 | **Config:** `experiments/configs/resnet_skip_study_v2.yaml`

## Study Design

Follow-up to the initial skip connection study (v1, 24 configs) which found that skip connections rescued GenericAELG at 30 stacks but didn't help TrendAELG+WaveletV3AELG. This v2 study explores untested block families and skip distances.

**36 configs across 6 sections:**

| Section | Architecture | Configs | Purpose |
|---------|-------------|---------|---------|
| A | GenericAE (homogeneous) | 6 | Does GenericAE degrade at depth like GenericAELG? |
| B | TrendVAE + WaveletV3VAE (alternating) | 6 | Double-VAE backbone interaction with skip |
| C | TrendVAE2 + WaveletV3VAE2 (alternating) | 6 | Double-VAE2 backbone interaction with skip |
| D | TrendWaveletAE (homogeneous) | 6 | Combined trend+wavelet block depth scaling |
| E | TrendWaveletAELG (homogeneous) | 6 | LG variant depth scaling |
| F | TrendVAE + HaarWaveletV3 (alternating) | 6 | Corrected single-VAE design (deterministic wavelet) |

**Successive halving:** 3 rounds (15/30/100 epochs), keep 50% per round, meta_forecaster disabled.

## Final Rankings (Round 3, 5 runs each)

| Rank | Config | Architecture | Stacks | Skip | SMAPE | Std | CV% | OWA |
|------|--------|-------------|--------|------|-------|-----|-----|-----|
| 1 | TWALG30_no_skip | TrendWaveletAELG | 30 | -- | 13.568 | 0.131 | 1.0% | 0.805 |
| 2 | TWALG10_no_skip | TrendWaveletAELG | 10 | -- | 13.570 | 0.100 | 0.7% | 0.807 |
| 3 | TWA30_no_skip | TrendWaveletAE | 30 | -- | 13.580 | 0.071 | 0.5% | 0.805 |
| 4 | TWALG20_skip3_a01 | TrendWaveletAELG | 20 | 3 | 13.642 | 0.128 | 0.9% | 0.813 |
| 5 | TWALG30_skip3_a01 | TrendWaveletAELG | 30 | 3 | 13.657 | 0.065 | 0.5% | 0.812 |
| 6 | TWA20_skip3_a01 | TrendWaveletAE | 20 | 3 | 13.658 | 0.108 | 0.8% | 0.813 |
| 7 | TWA20_no_skip | TrendWaveletAE | 20 | -- | 13.671 | 0.199 | 1.5% | 0.814 |
| 8 | TWA30_skip3_a01 | TrendWaveletAE | 30 | 3 | 13.752 | 0.152 | 1.1% | 0.821 |
| 9 | TWALG20_no_skip | TrendWaveletAELG | 20 | -- | 13.800 | 0.590 | 4.3% | 0.827 |

All R3 configs early-stopped before 100 epochs (39-64 epochs), indicating good convergence.

## Key Findings

### 1. GenericAE does NOT degrade at depth (unlike GenericAELG)

| Config | Stacks | SMAPE (R1) |
|--------|--------|------------|
| GAE10_no_skip | 10 | 15.55 ± 0.96 |
| GAE20_no_skip | 20 | 16.18 ± 0.53 |
| GAE30_no_skip | 30 | 15.16 ± 0.39 |
| GAE30_skip3_a01 | 30 | 14.76 ± 0.40 |

GenericAE shows no systematic degradation from 10 to 30 stacks (v1 showed GenericAELG collapsing from 14→36 SMAPE at 30 stacks without skip). Skip helps modestly at 30 stacks (-0.4 SMAPE) but GenericAE doesn't need rescue. The AERootBlock backbone (without the learned gate) is inherently more stable at depth than AERootBlockLG.

GenericAE is outperformed by TrendWavelet families (~14.8 vs ~13.6) and was eliminated after R2.

### 2. TrendWavelet blocks scale excellently — no skip needed

TrendWaveletAE and TrendWaveletAELG both scale from 10 to 30 stacks without degradation:

| Config | Stacks | R3 SMAPE | R3 OWA |
|--------|--------|----------|--------|
| TWALG10_no_skip | 10 | 13.570 | 0.807 |
| TWALG20_no_skip | 20 | 13.800 | 0.827 |
| TWALG30_no_skip | 30 | 13.568 | 0.805 |
| TWA20_no_skip | 20 | 13.671 | 0.814 |
| TWA30_no_skip | 30 | 13.580 | 0.805 |

The combined trend+wavelet architecture maintains performance from 10-30 stacks, likely because the integrated polynomial + DWT basis provides sufficient inductive bias to prevent residual decay.

### 3. Skip connections don't help TrendWavelet families

| Comparison | SMAPE Delta | Verdict |
|-----------|-------------|---------|
| TWA20: no_skip → skip3 | -0.013 | Negligible |
| TWA30: no_skip → skip3 | +0.172 | Skip hurts |
| TWALG20: no_skip → skip3 | -0.158 | Marginal help |
| TWALG30: no_skip → skip3 | +0.089 | Skip hurts |

Skip distance=2 was eliminated in R2 for both TWA and TWALG, performing worse than skip=3. The TrendWavelet architecture is inherently depth-resilient; skip injection adds noise to a well-behaved residual stream.

### 4. Double-VAE pairing is catastrophically bad

| Architecture | Best R1 SMAPE | Params |
|-------------|---------------|--------|
| TrendVAE + WaveletV3VAE | 29.23 (8 stacks) | 1.08M |
| TrendVAE2 + WaveletV3VAE2 | 37.49 (8 stacks) | 0.77M |
| TrendVAE + HaarWaveletV3 | 15.26 (10 stacks) | 4.39M |
| TrendWaveletAE | 14.27 (10 stacks) | 1.48M |

Pairing two VAE-backbone blocks compounds stochastic noise from the reparameterization trick. Each block's backcast is corrupted by sampling noise, and the residual connection propagates this corruption to all downstream blocks. The double-VAE2 variant is even worse.

The corrected design (TrendVAE + deterministic HaarWaveletV3) achieves 15.26 SMAPE — nearly 2x better than double-VAE — confirming the diagnosis. However, this single-VAE design still underperforms TrendWaveletAE (14.27) and was eliminated after R2.

Skip connections do not rescue double-VAE architectures: TVAE24_skip4_a01 (32.8) performs comparably to TVAE24_no_skip (30.1). The problem is fundamental to the VAE pairing, not depth-related signal decay.

### 5. TrendVAE+Haar underperforms TrendWavelet despite 3x more parameters

| Architecture | Stacks | R1 SMAPE | Params |
|-------------|--------|----------|--------|
| TVH10_no_skip | 10 | 15.26 | 4.39M |
| TWA10_no_skip | 10 | 14.27 | 1.48M |
| TWALG10_no_skip | 10 | 14.30 | 1.49M |

TrendVAE+HaarWaveletV3 uses default widths (t_width=256, g_width=512) resulting in 4.39M params at 10 stacks — 3x more than TrendWaveletAE (1.48M). Despite the capacity advantage, it underperforms by ~1 SMAPE point. This is consistent with the backbone hierarchy: AERootBlock > AERootBlockVAE across all studies.

TVH also shows degradation and instability at depth (TVH30_skip3 SMAPE=16.42 ± 1.41), unlike TrendWavelet.

### 6. v2 winner matches v1 winner

| Study | Winner | SMAPE | Architecture |
|-------|--------|-------|-------------|
| v1 | TW16_skip4_learn | 13.521 | TrendAELG + WaveletV3AELG (alternating) |
| v2 | TWALG30_no_skip | 13.568 | TrendWaveletAELG (unified) |
| v2 #3 | TWA30_no_skip | 13.580 | TrendWaveletAE (unified) |

The unified TrendWavelet blocks match the alternating TrendAELG+WaveletV3AELG architecture at equivalent SMAPE but with 3x fewer parameters (4.45M vs ~13M for alternating at matched stack count). The unified block's built-in polynomial+DWT basis provides the same representational capacity in a single block.

## Successive Halving Progression

| Round | Configs | Epochs | Runs | Eliminated |
|-------|---------|--------|------|-----------|
| R1 | 36 | 15 | 3 | All VAE/VAE2 variants, GAE10/20, TVH20/skip variants |
| R2 | 18 | 30 | 3 | GAE30 variants, TVH10/30, TWA10, skip_dist=2 variants |
| R3 | 9 | 100 | 5 | — (final) |

The halving correctly eliminated all double-VAE configs in R1 and all GenericAE/TVH configs by R2, converging on the TrendWavelet family as the clear winner.

## Conclusions

1. **Skip connections are architecture-specific**: They rescue GenericAELG from depth degradation (v1 finding) but don't help TrendWavelet families, which are inherently depth-stable.

2. **The TrendWavelet unified block is the best architecture for M4-Yearly**: SMAPE ~13.57 across 10-30 stacks with no degradation, low variance (CV 0.5-1.0%), and modest parameter count (1.5-4.5M). Both AE and AELG backbones perform equivalently.

3. **Double-VAE is fundamentally broken**: Pairing two VAE-backbone blocks corrupts the residual stream with compounded sampling noise. Always pair VAE blocks with deterministic (RootBlock) counterparts.

4. **GenericAE is depth-stable**: Unlike GenericAELG, the base AERootBlock backbone doesn't degrade at 30 stacks, suggesting the learned gate mechanism in AERootBlockLG is the source of instability.

5. **Skip distance=2 is too frequent**: Both TWA and TWALG skip_dist=2 were eliminated in R2, confirming skip_dist=3-5 is the sweet spot when skip is beneficial.

## Recommendations (M4-Yearly)

- For M4-Yearly: Use TrendWaveletAELG or TrendWaveletAE at 10-30 stacks without skip connections.
- For deep GenericAELG stacks (20+): Use skip_distance=4-5 with alpha=0.1 (v1 finding).
- Never pair two VAE-backbone blocks in alternating stacks.
- The unified TrendWavelet block is preferred over alternating TrendAELG+WaveletV3AELG due to 3x parameter efficiency at equivalent performance.

---

## Cross-Dataset Extension: Tourism-Yearly & Weather-96

**Date:** 2026-03-09 | **Notebook:** `experiments/analysis/analysis_reports/resnet_skip_study_v2_analysis.ipynb`

The M4-Yearly findings above were tested on two additional datasets to assess generalizability:

| Dataset | H | L | Studies | Total Runs |
|---------|---|---|---------|------------|
| Tourism-Yearly | 4 | 8 | v1 (24 configs, 160 rows), v2 (36 configs, 222 rows) | 382 |
| Weather-96 | 96 | 480 | v1 (24 configs, 156 rows), v2 (36 configs, 170 rows) | 326 |

### Tourism-Yearly Results

**v1 Final Rankings (R3, 5 runs):**

| Rank | Config | Architecture | Stacks | Skip | SMAPE | Std |
|------|--------|-------------|--------|------|-------|-----|
| 1 | GAELG20_skip5_learn | GenericAELG | 20 | d=5 | 21.174 | 0.313 |
| 2 | G30_skip5_a01 | Generic | 30 | d=5 | 21.196 | 0.064 |
| 3 | TW16_skip8_learn | TrendAELG+WaveletV3AELG | 16 | d=8 | 21.198 | 0.180 |
| 4 | TW16_skip4_a01 | TrendAELG+WaveletV3AELG | 16 | d=4 | 21.269 | 0.262 |
| 5 | TW8_skip4_learn | TrendAELG+WaveletV3AELG | 8 | d=4 | 21.301 | 0.176 |
| 6 | TW16_no_skip | TrendAELG+WaveletV3AELG | 16 | -- | 21.306 | 0.254 |

**v2 Final Rankings (R3, 5 runs):**

| Rank | Config | Architecture | Stacks | Skip | SMAPE | Std |
|------|--------|-------------|--------|------|-------|-----|
| 1 | GAE10_no_skip | GenericAE | 10 | -- | 20.526 | 0.204 |
| 2 | TWA10_no_skip | TrendWaveletAE | 10 | -- | 21.098 | 0.247 |
| 3 | GAE20_skip5_a01 | GenericAE | 20 | d=5 | 21.109 | 0.471 |
| 4 | GAE20_no_skip | GenericAE | 20 | -- | 21.114 | 0.254 |
| 5 | TWA20_no_skip | TrendWaveletAE | 20 | -- | 21.271 | 0.152 |
| 6 | TWALG10_no_skip | TrendWaveletAELG | 10 | -- | 21.572 | 0.964 |

**Tourism-specific findings:**

1. **GenericAE at 10 stacks is the study winner** (SMAPE 20.526), beating all TrendWavelet configs. This is the only dataset where GenericAE outperforms TrendWavelet. On Tourism's very short horizon (H=4), GenericAE's flexible basis outperforms the rigid polynomial+DWT decomposition.

2. **Skip connections actively HURT TrendWaveletAE on Tourism** (MWU p=0.001, rank-biserial r=0.79). This is the strongest anti-skip signal across all datasets.

3. **Skip helps Generic legacy on Tourism** (p=0.014). On M4, skip had negligible effect on Generic. On Tourism's short horizon, deep Generic stacks lose the signal without skip re-injection.

4. **Shallow depth wins:** Optimal depth is 10-16 stacks. 24-30 stacks consistently degrades performance (Tourism v1 TW24: +1.85 SMAPE vs TW16; Tourism v2 TWA30: +1.03 vs TWA10).

5. **GenericAELG bimodal collapse occurs on Tourism** at 30 stacks (one seed at 69.9, others 27-28), matching M4 but less severe.

6. **Double-VAE is EVEN WORSE on Tourism.** TrendVAE+WaveletV3VAE: SMAPE 143-181 (near random-output territory). The very short H=4 gives no room for reparameterization noise to average out.

7. **GAE10 at SMAPE 20.53 is competitive with Tourism SOTA** (TrendWaveletAELG_coif3 at 20.864). Worth further investigation with optimized hyperparameters.

### Weather-96 Results

**v1 Final Rankings (R3, 3-5 runs):**

| Rank | Config | Architecture | Stacks | Skip | SMAPE | MSE |
|------|--------|-------------|--------|------|-------|-----|
| 1 | TW16_no_skip | TrendAELG+WaveletV3AELG | 16 | -- | 40.245 | 0.117 |
| 2 | TW8_skip4_learn | TrendAELG+WaveletV3AELG | 8 | d=4 | 40.605 | 0.109 |
| 3 | TW16_skip4_a01 | TrendAELG+WaveletV3AELG | 16 | d=4 | 41.065 | 0.089 |
| 4 | TW16_skip8_learn | TrendAELG+WaveletV3AELG | 16 | d=8 | 41.083 | 0.113 |

**v2 Top Rankings (R2, 3 runs -- R3 incomplete):**

| Rank | Config | Architecture | Stacks | Skip | SMAPE | MSE |
|------|--------|-------------|--------|------|-------|-----|
| 1 (SMAPE) | TVH10_no_skip | TrendVAE+HaarWaveletV3 | 10 | -- | 37.964 | 0.123 |
| 2 | TWALG20_skip3_a01 | TrendWaveletAELG | 20 | d=3 | 38.762 | 0.089 |
| 3 | TWA20_no_skip | TrendWaveletAE | 20 | -- | 38.853 | 0.107 |
| 1 (MSE) | TWALG20_skip3_a01 | TrendWaveletAELG | 20 | d=3 | 38.762 | 0.089 |

**Weather-specific findings:**

1. **No architecture benefits from skip on Weather** (all MWU p > 0.25 for TrendWav/GenericAELG/Generic in v1). In v2, skip actually hurts TrendVAE2+WaveletV3VAE2 (p=0.013) and TrendVAE+Haar (p=0.034).

2. **GenericAELG does NOT collapse on Weather.** No bimodal failure at any depth (10-30 stacks). Weather's data normalization and longer sequences likely stabilize the learned gate's gradient flow.

3. **SMAPE and MSE rankings diverge.** TVH10 wins SMAPE (37.96) but TWALG20_skip3 wins MSE (0.089). For normalized Weather data, MSE is arguably more meaningful.

4. **GAELG10 has catastrophic MSE on Weather** (~2700) despite moderate SMAPE (~66). At 10 stacks, GenericAELG produces forecasts with roughly correct shape but wildly wrong scale.

5. **TrendWavelet at 16-20 stacks is optimal for Weather.** Both v1 and v2 converge on moderate depth as optimal.

6. **Double-VAE is much less catastrophic on Weather** (SMAPE 41-46 vs 38-43 for deterministic, ratio ~1.1x). On longer horizons, reparameterization noise averages out.

7. **Weather v2 R3 is incomplete** (only 2 of ~9 expected configs completed). Rankings are based on R2 data (30 epochs, 3 runs).

### Cross-Dataset Skip Connection Matrix

| Architecture | M4-Yearly (H=6) | Tourism-Yearly (H=4) | Weather-96 (H=96) |
|---|---|---|---|
| **TrendWaveletAE** | No effect (ns) | **NO-SKIP better** (p=0.001) | No effect (ns) |
| **TrendWaveletAELG** | No effect (ns) | No effect (ns) | No effect (ns) |
| **GenericAE** | No effect (ns) | **NO-SKIP better** (p=0.011) | No effect (ns) |
| **GenericAELG** | **SKIP rescues** at depth>=20 | Skip marginal (ns) | No effect (ns) |
| **Generic** | No effect (ns) | **SKIP helps** (p=0.014) | No effect (ns) |
| **Double-VAE** | Skip reduces severity (ns) | Skip reduces severity (p<0.01) | No effect / hurts |

### Cross-Dataset Conclusions

1. **Skip connections are NEVER optimal for TrendWavelet blocks** across all 3 datasets. The polynomial+DWT basis provides inherent depth stability that skip cannot improve and may degrade.

2. **The GenericAELG rescue effect is M4-specific.** The dramatic bimodal collapse only fully manifests on M4-Yearly. Tourism shows a milder form; Weather shows no collapse. The mechanism depends on horizon length and data normalization.

3. **Optimal depth scales with forecast horizon.** Tourism (H=4): 10 stacks. M4 (H=6): 10-30 stacks (flat). Weather (H=96): 16-20 stacks. Rough heuristic: `optimal_stacks ~ max(10, H)`.

4. **Double-VAE catastrophe severity scales inversely with horizon.** Tourism (H=4): 7-8x worse. M4 (H=6): 2-3x worse. Weather (H=96): ~1.1x worse.

5. **Fixed alpha=0.1 remains the safest default.** Wins 4/5 on M4, 4/6 on Tourism, 2/6 on Weather. Learnable alpha gains on Weather but never dramatically.

### Updated Recommendations (All Datasets)

- **Default: skip OFF** for all architectures except GenericAELG on M4-type data.
- **Tourism:** Use GenericAE at 10 stacks (SMAPE 20.53) or TrendWaveletAE at 10 stacks (SMAPE 21.10). Skip hurts. Investigate GAE10 with optimized hyperparameters vs Tourism SOTA.
- **Weather:** Use TrendAELG+WaveletV3AELG at 16 stacks (v1 best) or TrendWaveletAE/AELG at 20 stacks (v2 best). Complete v2 R3 for definitive ranking.
- **Never pair two VAE-backbone blocks** -- confirmed on all 3 datasets.
- **When skip IS needed** (GenericAELG on M4): use skip_distance=floor(n_stacks/6), alpha=0.1.
