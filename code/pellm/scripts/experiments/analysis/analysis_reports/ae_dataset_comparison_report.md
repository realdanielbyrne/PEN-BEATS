# AE Dataset Comparison: FineWeb vs WikiText-2 for Activation Caching

**Date:** 2026-03-14
**Experiment:** `ae_dataset_comparison`
**Analyst:** Claude (automated analysis)

## Executive Summary

This experiment tested whether using diverse FineWeb activations for AE pre-training improves final model quality compared to narrow WikiText-2. The answer is **no** -- dataset choice for AE activation caching has negligible effect on final perplexity (0.007 PPL difference, Cohen's d = 0.15).

The dominant finding is that **1 epoch of LM fine-tuning after AE pre-training is transformative**, reducing perplexity from ~27 to ~15.5 and pushing the single-layer-compressed model **below vanilla Llama-3.2-1B-Instruct perplexity** (15.44 vs 18.90). Every single one of 480 runs beats vanilla, regardless of configuration.

### New Best Configuration

| Setting | Value |
|---------|-------|
| pe_mlp_mode | ae_lg (or ae -- negligible difference) |
| ae_init | pretrained |
| ae_inner_init | svd |
| ae_latent_dim | 512 |
| ae_dataset | wikitext2 (simpler, equally effective) |
| ae_pretrain_epochs | 5 |
| epochs | 1 |
| lr | 1e-4 |
| **Final PPL** | **15.44 +/- 0.05 (0.82x vanilla)** |

Previous best (no LM fine-tuning): 26.54 PPL (1.40x vanilla).

---

## Experiment Design

**Grid:** 2 datasets (wikitext2, fineweb) x 4 inits (pretrained, fourier, cur, svd) x 3 latent dims (32, 256, 512) x 2 MLP modes (ae, ae_lg) x 2 inner inits (svd, match) x 5 seeds = 480 runs

**Fixed settings:** Layer 15 only, lr=1e-4, batch_size=2, grad_accum=8, ae_pretrain_epochs=5, epochs=1, freeze_base=true

**Note:** The grid is slightly imbalanced -- wikitext2+pretrained is missing ld=32 (0 runs) and has extra ld=512 runs (40 vs expected 20). The fineweb side is perfectly balanced. This does not affect paired comparisons.

---

## Detailed Results

### 1. PPL Pipeline Progression (non-SVD inits, means)

| Stage | PPL | Description |
|-------|-----|-------------|
| Vanilla Llama | 18.90 | Original model |
| After layer replacement | 42.6 | Baseline PPL with AE bottleneck |
| After AE pre-training (5 epochs) | 27.9 | Reconstruction pre-training |
| **After 1 epoch LM fine-tuning** | **15.6** | End-to-end language modeling |

The LM fine-tuning epoch reduces PPL by 44% from the AE pre-training level, and pushes below vanilla.

### 2. Factor Importance Ranking

| Rank | Factor | PPL Range | Effect Size |
|------|--------|-----------|-------------|
| 1 | ae_init | 0.244 | pretrained > fourier > cur >> svd |
| 2 | ae_latent_dim | 0.095 | 512 > 256 ~ 32 |
| 3 | ae_inner_init | 0.020 | svd > match (d=-0.45) |
| 4 | pe_mlp_mode | 0.007 | ae_lg ~ ae (negligible) |
| 5 | ae_dataset | 0.007 | wikitext2 ~ fineweb (negligible) |

### 3. Dataset Comparison (Primary Question)

| Metric | WikiText-2 | FineWeb | Difference |
|--------|-----------|---------|------------|
| Final PPL (mean) | 15.617 | 15.625 | +0.007 |
| AE Pretrain PPL | 27.78 | 28.68 | +0.90 |
| Final MSE | 0.0444 | 0.0438 | -0.0006 |
| Win rate | 50.8% | 49.2% | -- |

- Paired t-test: t=2.32, p=0.021 (marginal)
- Wilcoxon: W=13009, p=0.178 (not significant)
- Cohen's d: 0.15 (negligible)

**Conclusion:** FineWeb activations do not improve final quality. AE pre-training PPL is worse with FineWeb (training on out-of-distribution activations), but LM fine-tuning erases this gap completely. Use WikiText-2 for simplicity.

### 4. Init Strategy Hierarchy (Confirmed)

| Init | Mean Final PPL | Pairwise vs pretrained |
|------|---------------|----------------------|
| pretrained | 15.521 | -- |
| fourier | 15.579 | pretrained wins 88/120, p<0.0001 |
| cur | 15.630 | pretrained wins 119/120, p<0.0001 |
| svd | 15.765 | pretrained wins 120/120, p<0.0001 |

The hierarchy is the same as prior experiments, but gaps are compressed by LM fine-tuning (0.24 PPL total spread vs 3+ PPL without fine-tuning).

### 5. ae_inner_init: SVD > Match (New Finding)

SVD inner init for fc2/fc3 layers wins 67.3% of paired comparisons (p<1e-11, d=-0.45). This **reverses** the prior finding from the fc2fc3_strategy experiment where algorithm-specific init was tested without LM fine-tuning. With LM fine-tuning, SVD provides a better starting point for the bottleneck layers.

### 6. ae vs ae_lg: Effectively Tied (Updated Finding)

ae_lg wins 55.8% of paired comparisons (p=0.0004, d=0.22). This reverses the prior finding where ae won 30/32 comparisons without LM fine-tuning. With LM fine-tuning, the learned gate provides a negligible benefit. Use either; ae is simpler.

### 7. Latent Dimension: 512 Best, But 32 is Surprisingly Close

| ld | Mean Final PPL | Delta from 512 |
|----|---------------|----------------|
| 512 | 15.565 | -- |
| 256 | 15.657 | +0.092 |
| 32 | 15.660 | +0.095 |

ld=32 achieves only 0.095 PPL worse than ld=512 after LM fine-tuning. This is remarkable given the 16x difference in bottleneck capacity. With more LM fine-tuning epochs, the gap may close further.

### 8. AE Pre-training MSE

- All runs completed 5 epochs (no early stopping triggered)
- MSE-PPL correlation: r=0.77 (p<1e-96), confirming MSE is a good proxy
- Pretrained init achieves lowest final MSE (0.038), followed by fourier (0.042), cur (0.043), svd (0.053)
- FineWeb starts with lower MSE but converges to the same level as WikiText-2

---

## Recommendations

### Current Best Configuration (High Confidence)

```yaml
pe_mlp_mode: ae          # or ae_lg, negligible difference
ae_init: pretrained
ae_inner_init: svd
ae_latent_dim: 512        # 256 or even 32 acceptable with more fine-tuning
ae_pretrain_epochs: 5
epochs: 1                 # minimum; more epochs likely better
lr: 1e-4
ae_dataset: wikitext2     # simpler, no benefit from FineWeb
pe_mlp_layer_indices: [15]
freeze_base: true
```

### What to Test Next

1. **More LM fine-tuning epochs (highest priority):**
   ```yaml
   epochs: [1, 2, 3, 5]
   ae_pretrain_epochs: 5
   ae_init: pretrained
   ae_latent_dim: [32, 256, 512]
   ```
   Rationale: best_val_ppl averages 13.4 vs test 15.4 -- model has not converged.

2. **Multi-layer replacement:**
   ```yaml
   pe_mlp_layer_indices: [[15], [14,15], [13,14,15], [12,13,14,15]]
   ae_init: pretrained
   ae_latent_dim: 256
   epochs: 3
   ```
   Rationale: Single-layer achieves 0.82x vanilla. Can we compress more layers and stay below 1.0x?

3. **Skip AE pre-training:**
   ```yaml
   ae_pretrain_epochs: [0, 5]
   epochs: [1, 3]
   ae_init: pretrained
   ```
   Rationale: If LM fine-tuning is this powerful, AE pre-training may be unnecessary overhead.

4. **ld=32 with more fine-tuning:**
   ```yaml
   ae_latent_dim: 32
   epochs: [1, 3, 5, 10]
   ae_init: pretrained
   ```
   Rationale: ld=32 is only 0.095 PPL worse after 1 epoch. Extremely high compression ratio (~99% MLP param reduction). Could be the sweet spot with more training.

### Open Questions

1. **Why does 1 epoch of LM fine-tuning have such a massive effect?** The AE pre-training only optimizes reconstruction; LM fine-tuning allows the compressed layer to adapt its outputs to minimize cross-entropy directly. This suggests the AE bottleneck retains enough information that the model can "route around" reconstruction imperfections.

2. **Will the init hierarchy persist with more fine-tuning?** The gap between pretrained and svd init compressed from 3+ PPL to 0.24 PPL with just 1 epoch. More epochs may equalize all inits.

3. **Is AE pre-training necessary at all?** With random init + LM fine-tuning, would we get similar results?

4. **How many layers can be compressed below vanilla PPL?** The compression budget per layer is generous with LM fine-tuning.

---

## Analysis Artifacts

- **Notebook:** `scripts/experiments/analysis/notebooks/ae_dataset_comparison_analysis.ipynb`
- **This report:** `scripts/experiments/analysis/analysis_reports/ae_dataset_comparison_report.md`
- **Experiment config:** `scripts/experiments/ae_dataset_comparison.yaml`
- **Raw data:** `scripts/experiments/results/ae_dataset_comparison/logs/`
