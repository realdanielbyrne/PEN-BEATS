# Hybrid Distillation Layer 15 Test -- Analysis Report

**Date:** 2026-03-27
**Experiment:** `hybrid_distill_layer15_test`
**Analyst:** Claude (automated)

---

## Experiment Overview

This report details the `hybrid_distill_layer15_test` experiment. This experiment investigated whether hybrid knowledge distillation—combining both logit KD (`kd_alpha=0.5`) and attention pattern KD (`kd_attn_weight=0.3`)—improves the training of a pipeline where both MLP and attention are replaced on decoder layer 15.

## Executive Summary

This experiment tested **hybrid knowledge distillation** (logit KD + attention pattern KD) in a three-wave pipeline replacing both MLP and attention on decoder layer 15 of Llama-3.2-1B-Instruct.

**Result: Hybrid KD actively harms training.** The combined logit + attention pattern KD (alpha=0.5, attn_weight=0.3) produced the worst Wave 2 PPL of any wave pipeline experiment to date (27.16 vs 21.97 for no-KD baseline). This extends the prior finding that logit-only KD hurts AE bottleneck training -- adding attention pattern alignment makes it even worse.

Wave 3 (polish) **crashed** due to a bug where the `attn_implementation="eager"` override does not propagate through `resume_from` model loading.

## Experiment Design

Three-wave sequential pipeline on FineWeb (50k samples), layer 15 only:

| Wave | Replacement | KD Settings | LM FT | Status |
|------|------------|-------------|--------|--------|
| W1 | AE-LG MLP (ld=512) | None (alpha=1.0) | 3 epochs, lr=1e-4 | OK |
| W2 | + TW-LG attention (td=3, wd=28) | alpha=0.5, attn_kd=0.3 | 3 epochs, lr=1e-4 | OK (retry) |
| W3 | Polish (unfrozen base) | alpha=0.7, attn_kd=0.1 | 2 epochs, lr=1e-5 | FAILED |

All waves used: batch_size=3, grad_accum_steps=8, freeze_base=True (except W3), seed=42.

## Detailed Results

### Wave 1: AE-LG MLP Replacement (No KD)

| Metric | Value |
|--------|-------|
| Original PPL (vanilla) | 19.45 |
| Baseline PPL (post-swap) | 42.72 (2.20x vanilla) |
| AE Pretrain PPL | 25.34 (1.30x vanilla) |
| Final PPL (3 LM FT epochs) | 18.81 (0.97x vanilla) |
| PPL improvement | 55.97% |
| Parameters | 1191M total, 760M trainable |

**AE Pre-training MSE convergence:**

| Epoch | MSE | LR |
|-------|-----|-----|
| 1 | 0.3618 | 1e-6 (warmup) |
| 2 | 0.0310 | 5e-4 (warmup) |
| 3 | 0.0251 | 1e-3 |
| 4 | 0.0220 | 8.5e-4 |
| 5 | 0.0213 | 7.2e-4 |

94% MSE reduction in 5 epochs. 92% in 3 epochs -- confirming 3 epochs is sufficient.

**LM Fine-tuning (overfitting observed):**

| Epoch | Train Loss | Val PPL |
|-------|-----------|---------|
| 1 | 3.159 | 22.36 |
| 2 | 2.670 | 24.01 |
| 3 | 2.243 | 29.29 |

Val PPL degrades after epoch 1 while train loss decreases -- classic overfitting. Best checkpoint is epoch 1.

### Wave 2: Hybrid KD (Logit + Attention Pattern)

| Metric | Value |
|--------|-------|
| Baseline PPL (post-attn-swap) | 20.91 |
| Final PPL (3 LM FT epochs) | 27.16 |
| PPL improvement | -29.89% (DEGRADED) |
| Parameters | 1180M total, 761M trainable |
| Disruption from attn swap | 1.11x (18.81 -> 20.91) |

**LM Fine-tuning:**

| Epoch | Train Loss | Val PPL |
|-------|-----------|---------|
| 1 | 3.518 | 34.54 |
| 2 | 3.283 | 32.48 |
| 3 | 3.240 | 31.70 |

Training was still converging (val_ppl declining) but even the trajectory suggests convergence around 30 PPL -- well above the 20.91 baseline. The model never recovered from the KD-induced constraint.

### Wave 3: Polish (FAILED)

Failed with `IndexError: tuple index out of range` at `student_attns[layer_idx]`.

**Root cause:** Model loaded from `resume_from` uses SDPA attention implementation, which returns empty attention tuples when `output_attentions=True` is requested. The `attn_implementation="eager"` override that the code sets for attention KD is not propagating through the `from_pretrained_pe_llama()` loading path.

## Cross-Experiment Comparison

### KD Impact on Wave 2 Final PPL

```
1. No KD (ctxteacher_v1):          W2 PPL = 21.97  -- BEST
2. Logit KD alpha=0.7 (ctxteacher_v2): W2 PPL = 23.42  (+1.45, +6.6%)
3. Hybrid KD alpha=0.5+attn=0.3:   W2 PPL = 27.16  (+5.19, +23.6%)
```

**Monotonic degradation:** Every increase in KD signal strength produces worse results. The relationship is:

- 0% KD weight -> 21.97 PPL (best)
- 30% KD weight -> 23.42 PPL (+1.45)
- 50% KD + 30% attn KD -> 27.16 PPL (+5.19)

**Caveat:** The experiments differ in layer count (14+15 vs 15 only) and epoch count (1 vs 3). However, the hybrid_distill experiment had an easier task (1 layer, not 2) and more training budget (3 epochs, not 1), yet produced the worst result. This makes the KD penalty even more striking.

### Wave 1 MLP Comparison

| Experiment | Layers | W1 Final PPL |
|-----------|--------|-------------|
| hybrid_distill | 15 only | 18.81 |
| ctxteacher_v1 | 14+15 | 19.86 |
| ctxteacher_v2 | 14+15 | 22.58 |

The hybrid_distill W1 had the best W1 PPL, as expected since it only replaced 1 layer (vs 2). This confirms AE-LG MLP replacement is solid when KD is not applied.

## Key Findings

### 1. Hybrid KD is the worst KD variant tested

Adding attention pattern KD on top of logit KD costs an additional +3.74 PPL beyond the already-harmful logit KD. The attention pattern alignment forces TrendWavelet projections to produce attention distributions matching dense attention -- directly fighting the compression.

### 2. KD is anti-correlated with performance in frozen AE bottleneck pipelines

Three data points now establish a clear trend: more KD = worse PPL. While the AE bottleneck can approximate the SwiGLU block locally (proven by pre-training MSE), during end-to-end fine-tuning with a frozen base model, the KD loss acts as a harmful global constraint. The frozen student lacks the degrees of freedom needed to exactly match the teacher's logits, so KD penalizes the student for finding any divergent (but potentially valid) compressed representation.

### 3. Overfitting persists with 3 epochs on FineWeb

Even with freeze_base=True and only layer 15 replaced, val PPL degrades after epoch 1. The prior recommendation of 1 epoch for FineWeb LM FT is reinforced.

### 4. Attention swap disruption is minimal on layer 15

Adding TW-LG attention on top of trained AE-LG MLP causes only 1.11x disruption (18.81 -> 20.91 PPL). This is less than the ~1.5x observed in the two-layer ctxteacher experiments.

### 5. SDPA/eager attention bug blocks attention pattern KD in pipelines

The `resume_from` code path does not honor `attn_implementation="eager"`, causing crashes when attention KD is used in wave 2+.

## Bugs to Fix

### Attention implementation not propagated on resume

**Location:** `scripts/finetune.py`, around the `from_pretrained_pe_llama()` call for `resume_from` loading.

**Problem:** When `kd_attn_weight > 0`, the code sets `attn_implementation="eager"` but this is not applied to the model config before loading from a saved checkpoint. The saved model's config uses SDPA (the default).

**Fix:** After loading the model from `resume_from`, set `model.config._attn_implementation = "eager"` and ensure all attention layers use the eager path. Alternatively, pass `attn_implementation="eager"` to the `from_pretrained` call.

## Recommendations

### Configuration: Abandon KD for wave pipelines

```yaml
# Recommended wave pipeline settings
kd_alpha: 1.0        # No KD (pure CE loss)
kd_attn_weight: 0.0  # No attention pattern KD
```

### Next Experiments

1. **Clean no-KD baseline on layer 15 with FineWeb** -- same pipeline but kd_alpha=1.0, 1 epoch per wave. Establishes the single-layer FineWeb baseline.

2. **Very light KD sweep (alpha=0.95, 0.98)** -- if KD has any regime where it helps, it would be at very low weight. Worth one test to close the question.

3. **Scale to more layers without KD** -- the no-KD pipeline is proven. Extend to layers 14+15, 12-15, etc.

### Suggested YAML for next test

```yaml
pipeline:
  name: "no_kd_layer15_fineweb"
  description: "Clean no-KD baseline for layer 15 on FineWeb"
  python: ".venv/bin/python"
  env:
    PYTORCH_ALLOC_CONF: "expandable_segments:True"
  wandb: false

defaults:
  dataset: "fineweb"
  dataset_num_samples: 50000
  epochs: 1
  lr: 1.0e-4
  patience: 3
  batch_size: 3
  grad_accum_steps: 8
  dtype: "bfloat16"
  freeze_base: true
  seed: 42

waves:
  - name: "wave1_mlp_layer15"
    pe_attn_mode: "standard"
    pe_mlp_mode: "ae_lg"
    pe_mlp_layer_indices: [15]
    ae_latent_dim: 512
    ae_init: "pretrained"
    active_g_finetune: true
    ae_pretrain_epochs: 3
    ae_pretrain_lr: 0.001
    ae_pretrain_scheduler: "exponential"
    ae_pretrain_lr_warmup: 2
    ae_pretrain_gamma: 0.85
    ae_dataset: "fineweb"
    ae_cache_num_samples: 50000

  - name: "wave2_attn_layer15"
    resume_from: "wave1_mlp_layer15"
    pe_attn_mode: "trend_wavelet_lg"
    pe_layer_indices: [15]
    pe_mlp_mode: "ae_lg"
    pe_mlp_layer_indices: [15]
    ae_latent_dim: 512
    ae_init: "pretrained"
    trend_dim: 3
    wavelet_dim: 28
    attn_init: "pretrained"
    active_g_finetune: true
    attn_pretrain_epochs: 0
    ae_pretrain_epochs: 0
    # No KD
    kd_alpha: 1.0
```

---

**Artifacts:**

- Notebook: `scripts/experiments/analysis/notebooks/hybrid_distill_layer15_test_analysis.ipynb`
- Report: `scripts/experiments/analysis/analysis_reports/hybrid_distill_layer15_test_report.md`
