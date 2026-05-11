# Contextual Teacher V2: KD + Attention Pre-training Analysis

**Date**: 2026-03-25
**Experiments**: `mlp_attn_two_wave_layers_14_15_ctxteacher_v2` (V2), `..._ctxteacher` (V1), `mlp_attn_two_wave_layers_14_15` (Original)
**Notebook**: `scripts/experiments/analysis/notebooks/ctxteacher_v2_analysis.ipynb`

## Experiment Overview

This report analyzes the `mlp_attn_two_wave_layers_14_15_ctxteacher_v2` experiment. The experiment was designed to test two additions to a two-wave parameter-efficient replacement pipeline on decoder layers 14 and 15:

1. The use of knowledge distillation (`kd_alpha=0.7`) during Wave 1 MLP replacement fine-tuning.
2. The use of contextual attention pre-training (using the Wave 1 compressed model as the teacher) for 3 epochs before Wave 2 attention fine-tuning.

## Executive Summary

V2 tested two new features on the MLP-first two-wave pipeline (layers 14-15): **knowledge distillation** (kd_alpha=0.7) and **contextual attention pre-training** (3 epochs using the wave-1 MLP-compressed model as teacher). Both features failed to improve over V1's simpler configuration.

**V2 final PPL: 23.42 -- worse than V1's 21.97 by +1.45 PPL.** Knowledge distillation is the primary cause: it degraded wave 1 by +2.72 PPL, which propagated through the entire pipeline. Contextual attention pre-training provided a real but ephemeral 3.37 PPL improvement that was entirely absorbed by LM fine-tuning.

## Final Rankings

```
1. V1 (ld=512, no KD, no attn PT):  final_ppl=21.97 (1.13x vanilla) -- BEST
2. V2 (ld=512, KD=0.7, ctx attn PT): final_ppl=23.42 (1.20x vanilla) -- +1.45 PPL
3. Original (ld=256, no KD, 5ep FT):  final_ppl=24.94 (1.18x vanilla) -- +2.97 PPL
```

Note: Original uses a different FineWeb sample (original_ppl=21.14 vs 19.45 for V1/V2), so the ratio comparison is more meaningful than raw PPL.

## Detailed Results

### Wave 1 (MLP Replacement)

| Metric | Original (ld=256) | V1 (ld=512, no KD) | V2 (ld=512, KD=0.7) |
|--------|-------------------|---------------------|----------------------|
| original_ppl | 21.14 | 19.45 | 19.45 |
| baseline_ppl | 120.81 | 109.19 | 109.19 |
| ae_pretrain_ppl | 35.24 | 32.08 | 33.16 |
| final_ppl | 22.26 | **19.86** | 22.58 |
| ae_pretrain_epochs | 5 | 5 | 3 |
| kd_alpha | 1.0 | 1.0 | 0.7 |
| train_loss (ep1) | 3.252 | 3.232 | 3.356 |
| val_ppl (ep1) | 22.83 | 23.85 | 26.38 |

**Key finding: KD at alpha=0.7 costs +2.72 PPL in wave 1** (22.58 vs 19.86). While the AE-compressed MLP *can* approximate the local SwiGLU block (achieving ~0.02 MSE in pre-training), during end-to-end fine-tuning with a frozen base model, the KD loss weight acts as a harmful global constraint. It forces the network to try and exactly match vanilla teacher logits, which the frozen student lacks the degrees of freedom to do. Train loss is higher (3.356 vs 3.232) because the KD term adds an irreducible component.

### AE Pre-training MSE

| Epoch | Original (ld=256) | V1 (ld=512) | V2 (ld=512) |
|-------|-------------------|-------------|-------------|
| 1 | 0.1967 | 0.1964 | 0.1962 |
| 2 | 0.0207 | 0.0211 | 0.0214 |
| 3 | 0.0190 | 0.0197 | 0.0205 |
| 4 | 0.0173 | 0.0174 | - |
| 5 | 0.0169 | 0.0169 | - |

3 epochs captures 96% of the MSE reduction. Epochs 4-5 contribute only 14% marginal improvement. **3 AE pretrain epochs is sufficient.**

### Wave 2 (Attention Replacement)

| Metric | Original (ld=256) | V1 (no KD, no attn PT) | V2 (KD=0.7, ctx attn PT) |
|--------|-------------------|------------------------|--------------------------|
| baseline_ppl | 32.87 | 28.80 | 36.17 |
| attn_pretrain_ppl | - | - | 32.80 |
| final_ppl | 24.94 | **21.97** | 23.42 |
| attn_pretrain_epochs | 0 | 0 | 3 |
| kd_alpha | 1.0 | 1.0 | 0.7 |
| train_loss (ep1) | 2.889 | 2.865 | 3.370 |
| val_ppl (ep1) | 25.33 | 26.61 | 27.37 |

**V2's higher baseline (36.17 vs 28.80) is caused by V2's worse wave 1 result** (22.58 vs 19.86 feeds into wave 2). The attention swap disruption ratio is similar: V2=1.60x, V1=1.45x.

### Contextual Attention Pre-training

V2's 3 epochs of contextual attn pre-training reduced baseline from 36.17 to 32.80 (3.37 PPL, 9.3%). This is notably better than prior vanilla-teacher attn pre-training experiments where MSE barely moved. The contextual teacher (MLP-compressed model) provides activations from the actual operating distribution.

However, **the MSE reduction is still tiny** (2.021 -> 1.996, only 1.3%), and the **final PPL after LM FT does not benefit**. The pre-training gain is entirely absorbed by one epoch of end-to-end training.

### Parameter Counts

| Configuration | Total Params | Reduction |
|---|---|---|
| Vanilla Llama-3.2-1B | ~1,236,000,000 | - |
| W1 MLP-only (ld=256) | 1,144,597,504 | 7.4% |
| W1 MLP-only (ld=512) | 1,145,647,104 | 7.3% |
| W2 Both axes (ld=256) | 1,124,134,136 | 9.1% |
| W2 Both axes (ld=512) | 1,125,183,736 | 9.0% |

ld=256 vs ld=512 differs by only ~1M params (0.08% of vanilla) but ld=512 is substantially better for quality.

## Causal Decomposition

| Source | PPL Impact |
|--------|-----------|
| KD penalty on W1 | +2.72 PPL |
| W1 penalty propagation to W2 | +7.37 PPL higher baseline |
| Ctx attn PT benefit (pre-FT) | -3.37 PPL |
| Net V2 regression vs V1 | +1.45 PPL |

KD is the dominant cause. The contextual attn PT partially compensates but cannot overcome the wave 1 deficit.

## Overfitting Dynamics

The original experiment (5 LM FT epochs) showed severe overfitting in both waves:

- W1: val PPL rises from 22.83 (ep1) to 36.03 (ep4), early stopped
- W2: val PPL rises from 25.33 (ep1) to 50.34 (ep4), early stopped

V1 and V2 wisely limited to 1 epoch. The 1-epoch regime captures the majority of the improvement with zero overfitting risk.

## Recommendations

### Current Best Configuration (Two-Wave, Layers 14-15)

```yaml
# Wave 1: MLP replacement
pe_mlp_mode: ae_lg
ae_latent_dim: 512
ae_init: pretrained
ae_pretrain_epochs: 3  # reduced from 5, sufficient
kd_alpha: 1.0  # KD disabled
epochs: 1
lr: 1.0e-4
freeze_base: true

# Wave 2: Attention replacement
pe_attn_mode: trend_wavelet_lg
trend_dim: 3
wavelet_dim: 28
attn_init: pretrained
attn_pretrain_epochs: 0  # skip, wasteful
kd_alpha: 1.0  # KD disabled
epochs: 1
```

### What to Test Next

1. **Higher wavelet_dim (64-128) in wave 2.** wd=28 is aggressive; prior single-layer results show wd=64 achieves 14.9 PPL (0.79x vanilla). This is the highest-impact change available.

2. **KD at higher alpha (0.95) or wave-2-only.** Remember kd_alpha=1.0 = pure CE (KD disabled); lower alpha = MORE KD. Alpha=0.7 (30% KD) already hurt. If testing KD, try alpha=0.95 (only 5% KD signal), or apply KD only in wave 2 after both axes are assembled.

3. **Lower lr (5e-5) with 2-3 epochs.** The 1-epoch constraint is conservative. With a halved learning rate, 2-3 epochs might be productive before overfitting.

4. **Scale to layers 12-15** using V1's configuration as template (no KD, no attn PT, ld=512, 1ep FT).

### Open Questions

1. **Is KD fundamentally incompatible with AE bottleneck MLP?** The bottleneck prevents exact teacher matching, so KD may always add noise. Testing KD on attention-only waves (no MLP bottleneck) would isolate this.

2. **Would KD help with more LM FT epochs?** With only 1 epoch, the model barely adapts. KD might become beneficial over 3-5 epochs by regularizing against overfitting -- but overfitting already occurs at epoch 2 without KD.

3. **Does contextual attn PT help when starting from a stronger base?** V2's wave 1 was degraded by KD. If the wave 1 model were V1-quality (19.86 PPL), the contextual attn PT might have a different effect.
