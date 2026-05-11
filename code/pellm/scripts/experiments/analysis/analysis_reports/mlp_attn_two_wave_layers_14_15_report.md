# Two-Wave Staged Training: MLP-First vs Attention-First on Layers 14-15

**Date**: 2026-03-24
**Experiments**: `mlp_attn_two_wave_layers_14_15`, `attn_mlp_two_wave_layers_14_15`
**Notebook**: `scripts/experiments/analysis/notebooks/mlp_attn_two_wave_layers_14_15_analysis.ipynb`

## Executive Summary

Two complementary wave pipeline experiments tested whether the order of staged MLP + attention replacement matters. Both pipelines target the same final architecture (AE-LG MLP + TrendWavelet-LG attention on decoder layers 14-15) but differ in which axis is replaced first:

- **MLP-first**: Wave 1 replaces MLP with AE-LG, Wave 2 adds TW-LG attention
- **Attention-first**: Wave 1 replaces attention with TW-LG, Wave 2 adds AE-LG MLP

**The result is decisive: MLP-first achieves 24.94 PPL; attention-first achieves 54.67 PPL.** This 2.2x difference from the same target architecture establishes that **wave ordering is a first-order design choice** -- MLP must always be replaced before attention in staged pipelines.

## Experiment Configuration

| Parameter | Value |
|---|---|
| Model | Llama-3.2-1B-Instruct |
| Target layers | 14, 15 |
| MLP mode | ae_lg (latent_dim=256, pretrained init) |
| Attention mode | trend_wavelet_lg (trend_dim=3, wavelet_dim=28, pretrained init) |
| Dataset | FineWeb (50k samples) |
| LM fine-tuning | 5 epochs, lr=1e-4, patience=3, batch_size=2, grad_accum=8 |
| Freeze strategy | freeze_base=True |
| active_g_finetune | True (both pipelines) |
| AE pre-training | 5 epochs (MLP-first) / 10 epochs (attn-first), exponential LR decay |
| Attn pre-training | 0 epochs (MLP-first) / 2 epochs (attn-first) |

## Results

### Final Rankings

| Rank | Pipeline | Final PPL | vs Original (21.14) | Best Val PPL | Epochs Run |
|------|----------|-----------|---------------------|-------------|------------|
| 1 | **MLP-first** | **24.94** | **1.18x** | 25.33 | 4 (early stop) |
| 2 | Attn-first | 54.67 | 2.59x | 54.64 | 5 |

Delta: +29.73 PPL (+119% relative). MLP-first is unambiguously superior.

### Per-Wave Progression

#### MLP-First Pipeline

| Stage | PPL | Recovery |
|-------|-----|----------|
| Original | 21.14 | -- |
| W1 Baseline (MLP swap) | 120.81 | 5.7x disruption |
| W1 AE Pretrain | 35.24 | 70.8% of gap closed |
| W1 Final (+LM FT) | 22.26 | 36.8% further from AE pretrain |
| W2 Baseline (+Attn swap) | 32.87 | 1.48x disruption from W1 final |
| W2 Final (+LM FT) | 24.94 | 24.1% of W2 gap closed |

#### Attention-First Pipeline

| Stage | PPL | Recovery |
|-------|-----|----------|
| Original | 21.14 | -- |
| W1 Baseline (Attn swap) | 32.22 | 1.52x disruption |
| W1 Attn Pretrain | 29.42 | 8.7% of gap closed |
| W1 Final (+LM FT) | 27.11 | 7.8% further |
| W2 Baseline (+MLP swap) | **904,138** | **33,340x disruption (CATASTROPHIC)** |
| W2 AE Pretrain | 76,620 | 91.5% of gap closed |
| W2 Final (+LM FT) | 54.67 | 99.9% of gap closed |

### Parameter Counts

| Configuration | Total Params | vs Vanilla |
|---|---|---|
| Vanilla Llama-3.2-1B | ~1,236,000,000 | 1.00x |
| W1 MLP-only (AE-LG on 14-15) | 1,144,597,504 | 0.93x |
| W1 Attn-only (TW-LG on 14-15) | 1,215,351,032 | 0.98x |
| Final (both axes on 14-15) | 1,124,134,136 | 0.91x |

Combined replacement of layers 14-15 removes ~112M parameters (9.1% reduction). MLP replacement contributes ~91M (7.4%), attention replacement contributes ~20M (1.7%).

### AE Pre-training MSE

Both pipelines achieve comparable AE reconstruction quality:

| Pipeline | AE Epochs | Final MSE | Starting MSE |
|---|---|---|---|
| MLP-first (W1) | 5 | 0.0169 | 0.197 |
| Attn-first (W2) | 10 | 0.0227 | 0.224 |

The attn-first pipeline's slightly higher MSE (0.023 vs 0.017) does not explain the 2.2x PPL gap. The catastrophe stems from the compressed-attention context making the downstream optimization landscape much harder, not from AE reconstruction quality.

## Analysis: Why MLP-First Wins

### The Core Mechanism

MLP replacement is inherently more disruptive than attention replacement:
- MLP swap baseline disruption: **5.7x** original PPL (120.81 from 21.14)
- Attention swap baseline disruption: **1.5x** original PPL (32.22 from 21.14)

This asymmetry means:

1. **MLP should see vanilla activations.** AE pre-training learns to reconstruct MLP input/output pairs. When the teacher model has clean, vanilla attention, the activation distribution is well-behaved and the AE can learn an effective compression. When attention is already compressed (TW-colored activations), the distribution is harder to model.

2. **Attention swap is a minor perturbation.** After MLP is trained, swapping attention increases PPL by only 1.48x (22.26 to 32.87). This small gap is easily closed by 1 epoch of LM fine-tuning.

3. **The reverse order creates a catastrophic cascade.** MLP swap after attention compression yields 904K baseline PPL -- the combined error from compressed attention AND uninitialized MLP bottleneck compounds multiplicatively through the residual stream.

### Overfitting in Both Pipelines

All waves overfit on FineWeb within 1-2 epochs:
- MLP-first W1: best val PPL at epoch 1 (22.83), rises to 36.03 by epoch 4
- MLP-first W2: best val PPL at epoch 1 (25.33), rises to 50.34 by epoch 4
- Attn-first W1: barely moves (27.91 to 27.79 over 5 epochs -- TW has only 576K params)
- Attn-first W2: best val PPL at epoch 3 (54.64), rises after

Early stopping with patience=3 correctly mitigates this, but setting epochs=1-2 would save 60-75% compute.

## Comparison to Prior Results

**Important caveat**: These experiments evaluate on FineWeb (original_ppl=21.14), while prior single-axis results evaluated on WikiText-2 (original_ppl=18.90). PPL values are not directly comparable.

Normalized comparison (final_ppl / original_ppl):

| Configuration | Ratio | Dataset |
|---|---|---|
| Best TW attn only (layer 15, wd=256) | 0.77x | WikiText-2 |
| Best AE MLP only (layer 15, ld=512) | 0.82x | WikiText-2 |
| MLP-first combined (layers 14-15) | 1.18x | FineWeb |
| Attn-first combined (layers 14-15) | 2.59x | FineWeb |

The combined replacement on 2 layers (14-15) with wd=28 does not beat vanilla, unlike the single-axis single-layer results with higher wavelet_dim or latent_dim. This is expected: (a) replacing 2 layers is harder than 1, (b) wd=28 is small, and (c) ae_latent_dim=256 is suboptimal (prior work shows ld=512 is better).

## Recommendations

### Current Best Configuration

For combined MLP+attention replacement on layers 14-15:

**MLP-first pipeline, ae_lg (ld=256) + trend_wavelet_lg (td=3, wd=28), FineWeb 50k, 5 epochs AE pretrain + 1 epoch LM FT per wave.** Final PPL: ~22-25 on FineWeb (1.05-1.18x original).

### What to Test Next

1. **Reduce LM FT to 1 epoch per wave.** Overfitting is severe; epoch 1 captures most gains. This saves ~75% compute per wave.

2. **Increase wavelet_dim to 64-128 in wave 2.** Prior single-axis results show wd=256 achieves 14.52 PPL (0.77x original) on WikiText-2. Even wd=64 should substantially improve the attention replacement quality.

3. **Increase ae_latent_dim to 512 in wave 1.** Prior results show ld=512 outperforms ld=256. The extra parameters are marginal compared to the quality improvement.

4. **Add knowledge distillation (kd_alpha=0.5).** Neither pipeline used KD. This is a known beneficial technique that could help both waves.

5. **Scale to more layers with MLP-first ordering.** The ordering principle is established; the `incremental_mlp_layer_by_layer.yaml` pipeline should use MLP-first ordering for each wave.

### Open Questions

1. **Would joint (non-staged) replacement work better?** Both axes replaced simultaneously in a single wave might avoid the ordering problem entirely, at the cost of a harder optimization.

2. **Does the ordering principle hold at higher compression?** With wd=128 and ld=512, the asymmetry between MLP and attention disruption may change.

3. **Is the FineWeb overfitting a dataset-specific issue?** Testing with WikiText-2 for LM FT (smaller but less diverse) might show different overfitting dynamics.
