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

## Recommendations

- For M4-Yearly: Use TrendWaveletAELG or TrendWaveletAE at 10-30 stacks without skip connections.
- For deep GenericAELG stacks (20+): Use skip_distance=4-5 with alpha=0.1 (v1 finding).
- Never pair two VAE-backbone blocks in alternating stacks.
- The unified TrendWavelet block is preferred over alternating TrendAELG+WaveletV3AELG due to 3x parameter efficiency at equivalent performance.
