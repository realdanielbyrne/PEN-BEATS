# TrendWavelet Layer 15 Sweep Analysis Report

**Date:** 2026-03-16
**Experiment:** `trendwavelet_layer15_sweep`
**Notebook:** `scripts/experiments/analysis/notebooks/trendwavelet_layer15_sweep_analysis.ipynb`

## Executive Summary

216 configurations were swept across 4 TrendWavelet attention modes, 3 wavelet dimensions (16/28/40), 3 wavelet types (haar/db3/sym10), and 6 initialization modes (pretrained/lstsq/svd/cur/fourier/random) on decoder layer 15 only. All runs used 5 epochs of attention reconstruction pre-training followed by 1 epoch of LM fine-tuning.

**Key findings:**

1. **Attention pre-training is wasteful, not destructive.** For lstsq/svd/fourier inits, MSE moves <1% during pre-training -- it is a complete no-op. For pretrained/random inits, pre-training recovers ~2.7 PPL but the final PPL after LM FT is actually worse than inits that skip the recovery. Recommendation: set `attn_pretrain_epochs=0`.

2. **CUR initialization is catastrophic and dimension-dependent.** At wavelet_dim=40, CUR produces baseline PPL up to 4,598 and final PPL up to 1,471. It must be excluded from all TrendWavelet experiments. The failure is caused by CUR selecting rows/columns in weight space, which have no valid interpretation in the TrendWavelet basis domain.

3. **Larger basis dimensions consistently improve perplexity** (p < 1e-24). The best run at wavelet_dim=40 achieves 18.96 PPL (only +0.06 from vanilla) with 48x compression. The scaling curve has not plateaued -- higher dimensions should be tested.

4. **Attention mode, wavelet type, and learned gating are all non-significant** (p > 0.15). The factor hierarchy is: wavelet_dim >> attn_init >> everything else.

## Detailed Rankings

### Top 10 Configurations

| Rank | Mode | WD | Wavelet | Init | Final PPL | Gap from Vanilla | Compression |
|------|------|----|---------|------|-----------|-----------------|-------------|
| 1 | tw_generic | 40 | db3 | random | 18.962 | +0.063 | 48x |
| 2 | tw_generic | 40 | sym10 | fourier | 19.023 | +0.124 | 48x |
| 3 | tw_generic | 40 | sym10 | random | 19.053 | +0.154 | 48x |
| 4 | tw | 40 | sym10 | svd | 19.069 | +0.170 | 48x |
| 5 | tw_generic | 40 | sym10 | svd | 19.090 | +0.191 | 48x |
| 6 | tw | 40 | sym10 | fourier | 19.114 | +0.215 | 48x |
| 7 | tw | 40 | sym10 | lstsq | 19.153 | +0.254 | 48x |
| 8 | tw_generic | 28 | haar | random | 19.153 | +0.254 | 66x |
| 9 | tw_generic | 40 | db3 | lstsq | 19.200 | +0.301 | 48x |
| 10 | tw_generic | 40 | sym10 | lstsq | 19.204 | +0.305 | 48x |

All top-10 configs use wavelet_dim=40 (except #8 at wd=28). The top 7 are all within 0.25 PPL of vanilla.

### Bottom 5 Configurations (all CUR)

| Rank | Mode | WD | Wavelet | Init | Final PPL | Baseline PPL |
|------|------|----|---------|------|-----------|-------------|
| 216 | tw | 40 | sym10 | cur | 1471.3 | 4598.3 |
| 215 | tw_generic | 40 | sym10 | cur | 1457.5 | 4596.2 |
| 214 | tw | 40 | haar | cur | 747.5 | 3257.8 |
| 213 | tw_generic | 40 | haar | cur | 719.9 | 3259.0 |
| 212 | tw_generic_lg | 40 | sym10 | cur | 218.5 | 552.7 |

## Statistical Tests

| Test | Factor | Statistic | p-value | Significant? |
|------|--------|-----------|---------|-------------|
| Kruskal-Wallis | wavelet_dim (excl CUR) | H=108.4 | 2.8e-24 | YES |
| Kruskal-Wallis | attn_init (excl CUR) | H=34.3 | 6.5e-7 | YES |
| Kruskal-Wallis | pe_attn_mode (excl CUR) | H=4.3 | 0.23 | No |
| Kruskal-Wallis | wavelet_type (excl CUR) | H=2.4 | 0.30 | No |
| Mann-Whitney | CUR vs non-CUR | U=6480 | 3.0e-21 | YES |
| Mann-Whitney | _lg vs non-_lg (excl CUR) | U=4552 | 0.15 | No |
| Mann-Whitney | generic vs non-generic (excl CUR) | U=3546 | 0.15 | No |
| Wilcoxon | pretrained vs lstsq (matched) | W=0 | 2.9e-11 | YES |

### Pairwise wavelet_dim comparisons (Mann-Whitney, excl CUR):
- wd=16 vs wd=28: p=2.4e-15, delta=0.57 PPL
- wd=28 vs wd=40: p=2.1e-6, delta=0.31 PPL
- wd=16 vs wd=40: p=1.9e-19, delta=0.88 PPL

## Attention Pre-training Analysis

### PPL Recovery Decomposition

| Init | Baseline PPL | After Pretrain | Final PPL | Pretrain Recovery | LM FT Recovery | Pretrain % of Cost |
|------|-------------|---------------|-----------|-------------------|----------------|-------------------|
| lstsq | 22.01 | 22.01 | 19.89 | 0.001 | 2.12 | 0.0% |
| svd | 22.02 | 22.01 | 19.85 | 0.004 | 2.16 | 0.1% |
| fourier | 22.02 | 22.01 | 19.83 | 0.004 | 2.19 | 0.1% |
| pretrained | 24.77 | 22.09 | 20.35 | 2.680 | 1.75 | 45.6% |
| random | 24.89 | 22.03 | 19.77 | 2.859 | 2.26 | 47.7% |

For lstsq/svd/fourier, the attention pre-training phase contributes effectively 0% of the PPL recovery. These inits already produce near-optimal reconstructions (baseline ~22.0 vs vanilla 18.9). The entire recovery comes from 1 epoch of LM fine-tuning.

For pretrained/random, pre-training does work (recovering ~2.7-2.9 PPL), but the final result is worse: pretrained ends at 20.35 (worst non-CUR init), while random ends at 19.77 (best init). The pre-training may be pushing weights into an MSE-optimal but LM-suboptimal basin.

### MSE Convergence

| Init | MSE Start | MSE End | Reduction % |
|------|-----------|---------|------------|
| lstsq | 2.112 | 2.112 | 0.02% |
| svd | 2.119 | 2.115 | 0.19% |
| fourier | 2.132 | 2.118 | 0.65% |
| pretrained | 2.441 | 2.251 | 7.71% |
| random | 2.160 | 2.113 | 2.18% |
| cur | 47.906 | 47.177 | 3.32% |

All runs completed the full 5 pre-training epochs (early stopping never triggered), confirming that the pre-training objective is poorly matched to the task.

## Basis Dimension Scaling

| Total Basis | Wavelet Dim | Mean Final PPL | PPL Gap | Compression | Trainable Params |
|------------|-------------|---------------|---------|-------------|-----------------|
| 19 | 16 | 20.42 | +1.52 | ~108x | ~224K |
| 31 | 28 | 19.85 | +0.95 | ~66x | ~322K |
| 43 | 40 | 19.54 | +0.65 | ~48x | ~420K |

The relationship is monotonically improving with no sign of plateauing. Each step adds ~100K trainable params while recovering ~0.3-0.6 PPL. The marginal cost is trivial (0.008% of total model params per step).

## Init Mode Rankings (excluding CUR)

| Rank | Init | Mean Final PPL | Delta from Best |
|------|------|---------------|----------------|
| 1 | random | 19.771 | -- |
| 2 | fourier | 19.826 | +0.055 |
| 3 | svd | 19.851 | +0.080 |
| 4 | lstsq | 19.889 | +0.118 |
| 5 | pretrained | 20.347 | +0.576 |

The `random` init winning is counterintuitive. Hypothesis: lstsq/svd/fourier start near the MSE-optimal reconstruction, which is a local minimum that the 1-epoch LM FT cannot fully escape. Random init starts in a more exploratory part of parameter space, allowing LM FT to find a better LM-optimal point. Pretrained init is worst because it copies raw weight rows as coefficients (wrong domain for the basis) and both pre-training and LM FT cannot fully correct this.

## Recommendations

### Current Best Configuration

```yaml
pe_attn_mode: trend_wavelet_generic  # marginal edge, not significant
wavelet_dim: 40                       # or higher
wavelet_type: db3                     # any works
trend_dim: 3
attn_init: random                     # or fourier/svd -- all within 0.1 PPL
pe_layer_indices: [15]
attn_pretrain_epochs: 0               # SKIP -- wasteful
epochs: 1
lr: 1e-4
freeze_base: true
```

Expected: ~18.96-19.07 PPL (48x compression on layer 15 attention, +0.06-0.17 from vanilla).

### Proposed Next Experiments

**1. Larger basis dimensions (high priority):**
```yaml
configs:
  - name: "tw_large_basis"
    pe_attn_mode: "trend_wavelet_generic"
    wavelet_dim: [40, 56, 64, 80, 96]
    wavelet_type: ["db3"]
    attn_init: ["random", "svd"]
    attn_pretrain_epochs: 0
    epochs: 1
```

**2. Pre-training ablation (confirmation):**
```yaml
configs:
  - name: "tw_no_pretrain"
    pe_attn_mode: ["trend_wavelet", "trend_wavelet_generic"]
    wavelet_dim: [40]
    wavelet_type: ["db3", "sym10"]
    attn_init: ["random", "svd", "lstsq"]
    attn_pretrain_epochs: [0, 5]
    epochs: 1
```

**3. Multi-layer TrendWavelet (layers 12-15):**
```yaml
configs:
  - name: "tw_multilayer"
    pe_attn_mode: "trend_wavelet_generic"
    wavelet_dim: [40, 64]
    attn_init: ["random", "svd"]
    pe_layer_indices: [[15], [14, 15], [12, 13, 14, 15]]
    attn_pretrain_epochs: 0
    epochs: [1, 3]
```

**4. Combined TrendWavelet + AE MLP on layer 15:**
```yaml
configs:
  - name: "tw_plus_ae"
    pe_attn_mode: "trend_wavelet_generic"
    pe_mlp_mode: "ae"
    wavelet_dim: [40]
    attn_init: "random"
    ae_latent_dim: [256, 512]
    ae_init: "pretrained"
    ae_inner_init: "svd"
    attn_pretrain_epochs: 0
    ae_pretrain_epochs: 10
    epochs: 1
```

### Open Questions

1. Where does the basis dimension scaling curve flatten? Need wavelet_dim > 40 data.
2. Is attn pre-training truly unnecessary, or would it help with more epochs / better learning rate? (Low prior -- MSE is already near floor for good inits.)
3. Does random init's advantage persist at larger basis dimensions and multi-layer settings?
4. Are TrendWavelet attention compression and AE MLP compression additive or do they interact?
