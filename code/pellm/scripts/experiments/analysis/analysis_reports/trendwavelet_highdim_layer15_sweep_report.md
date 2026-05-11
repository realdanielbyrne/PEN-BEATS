# TrendWavelet High-Dimension Layer 15 Sweep -- Analysis Report

**Date:** 2026-03-16
**Experiment:** `trendwavelet_highdim_layer15_sweep`
**Notebook:** `scripts/experiments/analysis/notebooks/trendwavelet_highdim_layer15_sweep_analysis.ipynb`

## Executive Summary

This sweep tests TrendWavelet attention replacements on layer 15 with higher wavelet dimensions (64, 128, 256) and a new `active_g` feature (SiLU activation on TrendWavelet output) applied independently during pre-training and fine-tuning. 124 of 240 planned runs completed, covering `trend_wavelet` and `trend_wavelet_generic` fully, with only 4 runs for `trend_wavelet_lg` and 0 for `trend_wavelet_generic_lg`.

**The single most important finding is that `active_g_finetune=True` is transformative.** It drops mean final PPL by 1.8 points (15.47 vs 17.28) with a Cohen's d of 2.6 and p < 1e-11. This effect is larger than wavelet_dim, attn_init, and pe_attn_mode combined. The best run achieves **14.52 PPL** -- 23% better than the prior TrendWavelet best (18.96) and well below vanilla Llama (18.90).

## Completion Status

| pe_attn_mode | Runs Complete | Expected | Coverage |
|---|---|---|---|
| trend_wavelet | 60 | 60 | 100% |
| trend_wavelet_generic | 60 | 60 | 100% |
| trend_wavelet_lg | 4 | 60 | 7% (wd=64, lstsq only) |
| trend_wavelet_generic_lg | 0 | 60 | 0% |
| **Total** | **124** | **240** | **52%** |

All 120 runs for the two complete modes have `status=ok`. Coverage is balanced across wavelet_dim, attn_init, and active_g combinations within these modes.

## Rankings

### Top 10 Configurations by Final PPL

| Rank | Final PPL | wavelet_dim | pe_attn_mode | attn_init | active_g_ft | Compression | Improvement vs Vanilla |
|---|---|---|---|---|---|---|---|
| 1 | 14.521 | 256 | tw_generic | random | True | 7.4x | 23.2% |
| 2 | 14.759 | 128 | tw_generic | random | True | 13.9x | 21.9% |
| 3 | 14.906 | 64 | tw_generic | random | True | 24.6x | 21.1% |
| 4 | 14.918 | 256 | tw_generic | fourier | True | 7.4x | 21.1% |
| 5 | 14.933 | 256 | tw | svd | True | 7.7x | 21.0% |
| 6 | 14.933 | 256 | tw | pretrained | True | 7.7x | 21.0% |
| 7 | 14.951 | 256 | tw_generic | pretrained | True | 7.4x | 20.9% |
| 8 | 14.952 | 256 | tw | pretrained | True | 7.7x | 20.9% |
| 9 | 14.990 | 256 | tw_generic | pretrained | True | 7.4x | 20.7% |
| 10 | 15.019 | 256 | tw | lstsq | True | 7.7x | 20.5% |

All top 10 runs have `active_g_finetune=True`. 8 of 10 are at wd=256. Random init appears 3 times (including #1).

### Factor Rankings

**1. active_g_finetune (DOMINANT)**
- True: mean 15.47 PPL, std 0.49
- False: mean 17.28 PPL, std 0.86
- Paired Wilcoxon: p = 1.1e-11, Cohen's d = 2.6
- Effect scales with wavelet_dim: +1.2 PPL at wd=64, +1.9 at wd=128, +2.3 at wd=256

**2. wavelet_dim (moderate)**
- wd=64: mean 15.94 (gf=True), 27x compression
- wd=128: mean 15.44, 14x compression
- wd=256: mean 15.03, 7.7x compression
- Kruskal-Wallis within gf=True: p = 0.093 (marginal)

**3. attn_init (small)**
- random: 15.22 mean (gf=True) -- BEST
- fourier: 15.48
- svd: 15.62
- lstsq: 15.53
- pretrained: 15.49
- Random significantly better than each alternative (p < 0.035, one-sided)

**4. pe_attn_mode (not significant)**
- trend_wavelet: 15.50 mean (gf=True)
- trend_wavelet_generic: 15.42 mean
- Paired Wilcoxon: p > 0.4

**5. active_g_pretrain (not significant)**
- True: 15.48 mean, False: 15.46 mean
- Mann-Whitney: p = 0.88

## Pre-training Convergence

MSE drops only 5-7% across 10 epochs of attention pre-training. Higher wavelet_dim achieves lower absolute MSE (1.32 at wd=256 vs 1.81 at wd=64) because more basis components provide better representational capacity. However, **MSE reduction does not predict final PPL** -- random init has the worst starting MSE but the best final PPL after LM fine-tuning.

Pre-training PPL recovery from baseline is modest: 1.0 PPL at wd=64, 2.0 at wd=128, 6.0 at wd=256. The pretrained init causes extreme baseline PPL at wd=256 (up to 204 PPL) but this is fully recovered by LM fine-tuning.

## Comparison to Prior Sweep (wd=16/28/40)

| Metric | Prior Best (wd=40) | This Sweep Best (wd=256) | Delta |
|---|---|---|---|
| Final PPL | 18.96 | 14.52 | -4.44 (23.4%) |
| vs Vanilla | +0.06 (+0.3%) | -4.38 (-23.2%) | Major improvement |
| active_g_finetune | Not tested | True | New feature |
| Compression (layer 15 attn) | 48x | 7.4x | Less compressed |

The dramatic improvement is primarily due to `active_g_finetune`, not wavelet_dim scaling. To verify, a follow-up with active_g_finetune=True at wd=40 is recommended.

## Recommendations

### Current Best Configuration (Layer 15 TrendWavelet Attention)

```yaml
pe_attn_mode: "trend_wavelet_generic"  # or "trend_wavelet" -- no significant difference
wavelet_dim: 256                        # best quality; use 64 for max compression
attn_init: "random"
active_g_pretrain: false
active_g_finetune: true
attn_pretrain_epochs: 0                 # wasteful, skip
epochs: 1
lr: 1.0e-3
freeze_base: true
```

**Confidence: HIGH** -- 120 complete runs, consistent across all factor combinations.

### Should the Sweep Continue?

**No.** The remaining 116 runs (`trend_wavelet_lg` and `trend_wavelet_generic_lg`) are low priority because:
1. `active_g_finetune` already provides output-level nonlinear gating
2. The 4 completed `trend_wavelet_lg` runs show no advantage over `trend_wavelet`
3. The `_lg` mode adds coefficient-level gating that is redundant with active_g

### What to Test Next

1. **Validate active_g_finetune at lower wavelet_dims** (16, 28, 40) -- confirm the effect generalizes and is not specific to high dims
2. **Multi-layer TrendWavelet with active_g** (layers 12-15) -- test whether active_g helps when replacing multiple layers
3. **Combined TrendWavelet attention + AE MLP** -- the best attention config (tw_generic, wd=256, random, active_g_finetune=True) combined with the best MLP config (ae, ld=512, pretrained, svd inner) should be tested as a full replacement
4. **Higher wavelet_dims** (384, 512) -- wavelet_dim scaling is not plateaued; push higher if compression is less critical

### Open Questions

1. Does active_g_finetune help when combined with AE MLP replacement? The SiLU may interact differently when the MLP is also compressed.
2. Is the active_g effect specific to 1 epoch of LM fine-tuning, or does it persist with more epochs? At convergence, the gap may narrow.
3. What is the mechanism? The SiLU acts on reconstructed projection outputs -- does it learn to suppress poorly-reconstructed components, or does it enable a fundamentally different representation?
