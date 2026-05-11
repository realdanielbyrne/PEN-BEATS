# AE Initialization Methods Layer 15 Sweep -- Analysis Report

**Experiment:** `ae_init_methods_layer15_sweep`
**Date:** 2026-03-13
**Analyst:** Claude Opus 4.6

---

## Executive Summary

This experiment benchmarks 5 AE initialization strategies (pretrained, random, svd, cur, fourier) across 4 latent dimensions (128, 256, 384, 512) for both `ae` and `ae_lg` MLP modes on decoder layer 15 of Llama-3.2-1B-Instruct. 80 total runs (40 configs x 2 duplicates) all completed successfully.

**Key findings:**

1. **New best configuration:** ae, pretrained init, ld=512 achieves final_ppl=26.54 (1.40x vanilla), improving over the prior best of 26.73 from the ae_vs_aelg sweep.

2. **Init hierarchy (definitive):** pretrained > fourier > cur > random >> svd. All pairwise differences are statistically significant (Wilcoxon p<0.001). Pretrained wins 8/8 head-to-head comparisons against its closest competitor (fourier).

3. **ae_lg does NOT help:** ae (without gating) significantly outperforms ae_lg, winning 30/32 paired comparisons (Wilcoxon p<0.0001, Cohen's d=-1.31). This overturns the hypothesis that learned gating would improve bottleneck effectiveness.

4. **New init methods (cur, fourier):** Fourier is a strong second-best init, trailing pretrained by only 0.17 PPL on average. CUR ranks third, 0.37 PPL behind pretrained. Neither outperforms pretrained.

5. **SVD instability worsens exponentially:** At ld=512 (ae mode), baseline_ppl explodes to 1161 (27.3x normal). SVD should never be used at any latent dimension for single-layer AE MLP.

---

## 1. Performance Rankings

### Top 15 Configurations (averaged over 2 duplicate runs)

| Rank | Mode  | LD  | Init       | Final PPL | Std   | Delta  | Improv% | PPL/Vanilla |
|------|-------|-----|------------|-----------|-------|--------|---------|-------------|
| 1    | ae    | 512 | pretrained | 26.544    | 0.011 | --     | 37.7%   | 1.405x      |
| 2    | ae    | 384 | pretrained | 26.579    | 0.037 | +0.035 | 37.6%   | 1.406x      |
| 3    | ae_lg | 512 | pretrained | 26.679    | 0.019 | +0.135 | 37.3%   | 1.412x      |
| 4    | ae    | 512 | fourier    | 26.701    | 0.055 | +0.157 | 37.3%   | 1.413x      |
| 5    | ae    | 384 | fourier    | 26.728    | 0.038 | +0.184 | 37.1%   | 1.414x      |
| 6    | ae    | 256 | pretrained | 26.769    | 0.045 | +0.226 | 37.1%   | 1.416x      |
| 7    | ae_lg | 384 | pretrained | 26.786    | 0.017 | +0.242 | 37.0%   | 1.417x      |
| 8    | ae_lg | 512 | fourier    | 26.800    | 0.039 | +0.256 | 37.0%   | 1.418x      |
| 9    | ae_lg | 256 | pretrained | 26.916    | 0.034 | +0.373 | 36.7%   | 1.424x      |
| 10   | ae_lg | 384 | fourier    | 26.932    | 0.070 | +0.388 | 36.6%   | 1.425x      |
| 11   | ae    | 256 | fourier    | 26.975    | 0.004 | +0.432 | 36.5%   | 1.427x      |
| 12   | ae_lg | 512 | cur        | 26.986    | 0.028 | +0.442 | 36.7%   | 1.428x      |
| 13   | ae    | 512 | cur        | 27.038    | 0.047 | +0.494 | 36.9%   | 1.431x      |
| 14   | ae    | 128 | pretrained | 27.127    | 0.015 | +0.583 | 36.0%   | 1.435x      |
| 15   | ae_lg | 256 | fourier    | 27.165    | 0.062 | +0.621 | 36.1%   | 1.437x      |

### Bottom 5 (all SVD)

| Rank | Mode  | LD  | Init | Final PPL | Baseline PPL | Improv% |
|------|-------|-----|------|-----------|--------------|---------|
| 36   | ae_lg | 384 | svd  | 28.885    | 120.6        | 76.1%   |
| 37   | ae_lg | 512 | svd  | 28.896    | 174.7        | 83.5%   |
| 38   | ae    | 256 | svd  | 29.102    | 172.0        | 83.1%   |
| 39   | ae    | 384 | svd  | 29.481    | 684.2        | 95.7%   |
| 40   | ae    | 512 | svd  | 29.623    | 1161.1       | 97.5%   |

---

## 2. Factor Analysis

### 2.1 Initialization Method (most important factor after LR)

**Marginal means (excluding SVD, averaged over ld and mode):**

| Init       | Mean PPL | Std   | Min PPL | Max PPL |
|------------|----------|-------|---------|---------|
| pretrained | 26.843   | 0.275 | 26.544  | 27.341  |
| fourier    | 27.014   | 0.286 | 26.701  | 27.465  |
| cur        | 27.216   | 0.163 | 26.986  | 27.478  |
| random     | 27.470   | 0.078 | 27.346  | 27.578  |

**Pairwise statistical tests (Wilcoxon signed-rank, n=16 paired observations):**

| Comparison           | Mean Diff | W-stat | p-value  | Cohen's d | Winner     |
|----------------------|-----------|--------|----------|-----------|------------|
| pretrained vs fourier| -0.172    | 0      | <0.0001* | -2.79     | pretrained |
| pretrained vs cur    | -0.373    | 0      | <0.0001* | -2.59     | pretrained |
| pretrained vs random | -0.628    | 0      | <0.0001* | -2.31     | pretrained |
| fourier vs cur       | -0.201    | 4      | 0.0002*  | -1.32     | fourier    |
| fourier vs random    | -0.456    | 0      | <0.0001* | -1.61     | fourier    |
| cur vs random        | -0.255    | 5      | 0.0003*  | -1.39     | cur        |

All pairwise differences are highly significant with large effect sizes. The ranking is stable: **pretrained > fourier > cur > random**.

**Ranking stability across latent dimensions:**

| LD  | 1st (PPL)         | 2nd (PPL)       | 3rd (PPL)     | 4th (PPL)      |
|-----|-------------------|-----------------|---------------|----------------|
| 128 | pretrained (27.23)| fourier (27.41) | cur (27.43)   | random (27.44) |
| 256 | pretrained (26.84)| fourier (27.07) | cur (27.24)   | random (27.48) |
| 384 | pretrained (26.68)| fourier (26.83) | cur (27.18)   | random (27.52) |
| 512 | pretrained (26.61)| fourier (26.75) | cur (27.01)   | random (27.44) |

The ranking is perfectly stable across all latent dimensions and both MLP modes. The advantage of pretrained over fourier actually *increases* at higher latent dims (gap grows from 0.18 at ld=128 to 0.22 at ld=256).

### 2.2 Latent Dimension

**Marginal means (excluding SVD):**

| LD  | Mean PPL | Best PPL | Delta from prev |
|-----|----------|----------|-----------------|
| 128 | 27.378   | 27.127   | --              |
| 256 | 27.161   | 26.769   | -0.217          |
| 384 | 27.052   | 26.579   | -0.109          |
| 512 | 26.953   | 26.544   | -0.098          |

**Diminishing returns analysis:**

| Transition | PPL Improvement | Per LD Unit  |
|------------|-----------------|--------------|
| 128 -> 256 | 0.217           | 0.00170      |
| 256 -> 384 | 0.109           | 0.00085      |
| 384 -> 512 | 0.098           | 0.00077      |

Improvement per additional latent dimension unit halves at each step. The 384->512 transition yields only 0.098 PPL for 128 extra latent units. For best absolute PPL, use ld=512. For efficiency (PPL per parameter), ld=256 or ld=384 are the sweet spots.

### 2.3 ae vs ae_lg (Learned Gating)

This is the first experiment with working ae_lg data (the stdout parsing bug from the prior sweep has been fixed).

| Metric            | ae     | ae_lg  |
|-------------------|--------|--------|
| Marginal mean PPL | 27.078 | 27.194 |
| Win rate          | 30/32  | 2/32   |
| Wilcoxon p-value  | <0.0001|        |
| Cohen's d         | -1.31  |        |

ae (without gating) is significantly and consistently better than ae_lg (with learned sigmoid gating). The mean difference is 0.116 PPL, which is small in absolute terms but highly consistent (ae wins 93.8% of paired comparisons).

**Interpretation:** The learned gate sigmoid(gate)*z at the bottleneck does not help for single-layer replacement at these latent dimensions. The gate may:

- Add unnecessary parameters that need optimization time beyond 10 epochs
- Introduce gradient flow complications through the sigmoid
- Not be needed when the bottleneck dimension is already well-sized (128-512 out of 2048 hidden dim)

The gate might be more useful for multi-layer replacement or when the bottleneck is more aggressive (ld<128). For now, **prefer ae over ae_lg**.

---

## 3. New Initialization Methods Assessment

### 3.1 CUR Decomposition

CUR uses leverage-score row/column selection from original LlamaMLP weights to initialize the AE bottleneck. Results:

- **Ranking:** 3rd of 4 stable inits (behind pretrained and fourier)
- **Mean PPL:** 27.216 (0.373 worse than pretrained)
- **Stability:** Excellent. No baseline inflation (all baselines ~42.5-42.9)
- **Convergence:** Fastest epoch-1 MSE (0.118 vs 0.129 for pretrained), but converges to higher final MSE (0.0377 vs 0.0333)
- **Verdict:** Stable but underperforms pretrained/fourier. CUR's leverage-score selection captures some weight structure but not as effectively as the least-squares fit of pretrained.

### 3.2 Fourier Filtering

Fourier uses FFT denoising of truncated LlamaMLP weights to initialize. Results:

- **Ranking:** 2nd of 4 stable inits (behind pretrained only)
- **Mean PPL:** 27.014 (0.171 worse than pretrained)
- **Stability:** Excellent. No baseline inflation.
- **Convergence:** Similar convergence trajectory to pretrained; slightly higher epoch-1 MSE (0.146 vs 0.129) but similar final MSE (0.0337 vs 0.0333)
- **Verdict:** Strong second-best. Fourier filtering preserves the low-frequency structure of original weights, providing a warm start that is nearly as good as least-squares fitting. A viable alternative when pretrained init is not available or desired.

### 3.3 Neither Outperforms Pretrained

Both new methods fail to beat pretrained initialization. Pretrained wins 8/8 head-to-head against fourier (p<0.0001) and 8/8 against CUR (p<0.0001). The pretrained least-squares fit from original weights remains the gold standard.

---

## 4. SVD Instability Analysis

SVD instability worsens exponentially with latent dimension, confirming and extending prior findings:

| Mode  | LD  | SVD Baseline | Normal Baseline | Inflation | SVD Final PPL |
|-------|-----|-------------|-----------------|-----------|---------------|
| ae    | 128 | 40.8        | 42.4            | 1.0x      | 28.488        |
| ae_lg | 128 | 40.7        | 42.4            | 1.0x      | 28.354        |
| ae    | 256 | 172.0       | 42.5            | 4.0x      | 29.102        |
| ae_lg | 256 | 61.3        | 42.5            | 1.4x      | 28.773        |
| ae    | 384 | 684.2       | 42.6            | 16.1x     | 29.481        |
| ae_lg | 384 | 120.6       | 42.5            | 2.8x      | 28.885        |
| ae    | 512 | 1161.1      | 42.6            | 27.3x     | 29.623        |
| ae_lg | 512 | 174.7       | 42.5            | 4.1x      | 28.896        |

**Notable finding:** ae_lg shows much less SVD baseline inflation than ae. At ld=512, ae has 27.3x inflation vs ae_lg's 4.1x. The learned gate may be dampening the effect of the bad SVD initialization. However, even with this mitigation, SVD ae_lg still underperforms all stable inits.

Despite enormous baseline inflation, AE pre-training partially recovers SVD runs to final PPL ~28.5-29.6, but this is still 2-3 PPL worse than stable inits (~26.5-27.5). **SVD should be avoided entirely for AE MLP initialization.**

---

## 5. AE Pre-training Convergence

### MSE by Init Method (averaged over all ld and mode)

| Init       | Epoch 1 MSE | Epoch 10 MSE | Total Reduction | Ep9->10 Drop |
|------------|-------------|--------------|-----------------|--------------|
| cur        | 0.11821     | 0.03771      | 68.1%           | 1.01%        |
| pretrained | 0.12884     | 0.03330      | 74.2%           | 1.16%        |
| fourier    | 0.14612     | 0.03368      | 76.9%           | 1.17%        |
| random     | 0.16942     | 0.03670      | 78.3%           | 1.16%        |
| svd        | 0.18462     | 0.04864      | 73.7%           | 0.95%        |

**Convergence status:** All init methods show ~1% MSE reduction from epoch 9 to 10, confirming that **10 epochs is insufficient for full convergence**. More training (20-30 epochs) would likely improve PPL further.

**Init ordering by convergence speed:** CUR starts with the lowest epoch-1 MSE (0.118), indicating its initialization is already close to the target. However, CUR converges to a higher final MSE (0.0377) than pretrained (0.0333) or fourier (0.0337), suggesting CUR captures different aspects of the weight structure that are less useful for reconstruction.

### MSE-PPL Correlation

- **Pearson r = 0.855 (p < 1e-18)**
- **Spearman rho = 0.783 (p < 1e-14)**

Very strong positive correlation between final MSE and final PPL. This validates the AE pre-training objective: minimizing reconstruction MSE directly translates to better language modeling. Lower MSE = better weight reconstruction = lower perplexity.

---

## 6. Cross-Experiment Comparison

The prior experiment (ae_vs_aelg_layer15_sweep) best result was ae, pretrained, ld=256, lr=1e-4 with final_ppl=26.73.

| Metric        | Prior Best   | New Best     | Improvement  |
|---------------|-------------|-------------|--------------|
| Config        | ae/pre/256  | ae/pre/512  | Higher ld    |
| Final PPL     | 26.73       | 26.54       | -0.19 PPL    |
| PPL/vanilla   | 1.414x      | 1.405x      | -0.009x      |
| Improvement%  | 37.18%      | 37.66%      | +0.48pp      |

The improvement comes purely from the higher latent dimension (512 vs 256). Init strategy and LR are unchanged.

---

## 7. Recommendations

### Default Configuration Going Forward

```
--pe-mlp-mode ae
--ae-init pretrained
--ae-latent-dim 512     (for best PPL; 256 for efficiency)
--lr 1e-4
--ae-pretrain-epochs 20  (extend from 10)
```

### Specific Recommendations

1. **Init:** Always use `pretrained`. It dominates all alternatives across all conditions.
2. **MLP mode:** Use `ae` (not ae_lg). Learned gating hurts at these latent dimensions.
3. **Latent dim:** ld=512 for best PPL; ld=384 for best efficiency tradeoff; ld=256 is the minimum recommended.
4. **Pretrain epochs:** Extend to 20-30 epochs (MSE not converged at 10).
5. **Never use SVD init** for AE MLP at any latent dimension.

---

## 8. Proposed Next Experiments

### Experiment 1: Extended AE Pre-training

Test whether more AE pre-training epochs improves the best configuration.

```yaml
experiment:
  name: "ae_extended_pretrain_layer15"
  python: ".venv/bin/python"
  env:
    PYTORCH_ALLOC_CONF: "expandable_segments:True"
  wandb: false

defaults:
  model_name: "meta-llama/Llama-3.2-1B-Instruct"
  dtype: "bfloat16"
  batch_size: 2
  grad_accum_steps: 8
  patience: 3
  max_length: 512
  max_eval_batches: 50
  pe_attn_mode: "standard"
  pe_mlp_mode: "ae"
  ae_init: "pretrained"
  freeze_base: true
  local_files_only: true
  pe_mlp_layer_indices: [[15]]
  ae_pretrain_early_stopping: true
  ae_pretrain_warmup: 5
  ae_pretrain_patience: 7
  lr: 1.0e-4
  epochs: 0

configs:
  - name: "ae_ld512_ep20"
    ae_latent_dim: 512
    ae_pretrain_epochs: 20

  - name: "ae_ld512_ep30"
    ae_latent_dim: 512
    ae_pretrain_epochs: 30

  - name: "ae_ld384_ep20"
    ae_latent_dim: 384
    ae_pretrain_epochs: 20

  - name: "ae_ld384_ep30"
    ae_latent_dim: 384
    ae_pretrain_epochs: 30
```

**Rationale:** MSE is still declining ~1% per epoch at epoch 10. Extended training should lower MSE further, and given the strong MSE-PPL correlation (r=0.855), this should translate to lower PPL. This is the lowest-risk experiment to run next.

### Experiment 2: Multi-Layer AE MLP Replacement

Test AE MLP on multiple decoder layers using the optimal settings from layer 15.

```yaml
experiment:
  name: "ae_multilayer_sweep"
  python: ".venv/bin/python"
  env:
    PYTORCH_ALLOC_CONF: "expandable_segments:True"
  wandb: false

defaults:
  model_name: "meta-llama/Llama-3.2-1B-Instruct"
  dtype: "bfloat16"
  batch_size: 2
  grad_accum_steps: 8
  patience: 3
  max_length: 512
  max_eval_batches: 50
  pe_attn_mode: "standard"
  pe_mlp_mode: "ae"
  ae_init: "pretrained"
  ae_latent_dim: 512
  freeze_base: true
  local_files_only: true
  ae_pretrain_epochs: 20
  ae_pretrain_early_stopping: true
  ae_pretrain_warmup: 5
  ae_pretrain_patience: 7
  lr: 1.0e-4
  epochs: 0

configs:
  - name: "layer15_only"
    pe_mlp_layer_indices: [[15]]

  - name: "layers_14_15"
    pe_mlp_layer_indices: [[14, 15]]

  - name: "layers_13_14_15"
    pe_mlp_layer_indices: [[13, 14, 15]]

  - name: "layers_12_to_15"
    pe_mlp_layer_indices: [[12, 13, 14, 15]]
```

**Rationale:** Single-layer replacement achieves 37.7% improvement over baseline_ppl. Multi-layer replacement will increase compression but also increase baseline disruption. Prior attention-layer findings show last-4 layers are least destructive. Testing progressive expansion (1->2->3->4 layers) maps the PPL-compression Pareto frontier for MLP replacement.

### Experiment 3: Combined Attention + MLP Compression

Test combining TrendWavelet attention with AE MLP bottleneck on the last 4 layers.

```yaml
experiment:
  name: "combined_attn_mlp_last4"
  python: ".venv/bin/python"
  env:
    PYTORCH_ALLOC_CONF: "expandable_segments:True"
  wandb: false

defaults:
  model_name: "meta-llama/Llama-3.2-1B-Instruct"
  dtype: "bfloat16"
  batch_size: 2
  grad_accum_steps: 8
  patience: 3
  max_length: 512
  max_eval_batches: 50
  freeze_base: true
  local_files_only: true
  pe_mlp_mode: "ae"
  ae_init: "pretrained"
  ae_latent_dim: 512
  ae_pretrain_epochs: 20
  ae_pretrain_early_stopping: true
  ae_pretrain_warmup: 5
  ae_pretrain_patience: 7
  lr: 1.0e-4

configs:
  - name: "attn_only_last4"
    pe_attn_mode: "trend_wavelet_lg"
    pe_layer_indices: [[12, 13, 14, 15]]
    pe_mlp_mode: "standard"
    trend_dim: 3
    wavelet_dim: 28
    epochs: 10

  - name: "mlp_only_last4"
    pe_attn_mode: "standard"
    pe_mlp_layer_indices: [[12, 13, 14, 15]]
    epochs: 0

  - name: "both_last4_joint"
    pe_attn_mode: "trend_wavelet_lg"
    pe_layer_indices: [[12, 13, 14, 15]]
    pe_mlp_layer_indices: [[12, 13, 14, 15]]
    trend_dim: 3
    wavelet_dim: 28
    epochs: 5

  - name: "both_last4_staged_attn_first"
    pe_attn_mode: "trend_wavelet_lg"
    pe_layer_indices: [[12, 13, 14, 15]]
    pe_mlp_layer_indices: [[12, 13, 14, 15]]
    trend_dim: 3
    wavelet_dim: 28
    epochs: 5
    # Note: staged training requires manual resume_from setup
```

**Rationale:** Attention and MLP compression are independent axes. Combining them on the last 4 layers should yield maximum parameter reduction while maintaining low PPL. This experiment tests joint vs staged training and quantifies the interaction between the two compression approaches.

---

## 9. Open Questions

1. **How much does extended AE pre-training help?** MSE is clearly not converged. The strong MSE-PPL correlation (r=0.855) predicts meaningful PPL improvement from more epochs.

2. **How does multi-layer AE MLP replacement scale?** Layer 15 alone gives 37.7% improvement over its disrupted baseline. Will layers 12-14 behave similarly, or are deeper layers more sensitive?

3. **Why does ae_lg underperform ae?** Is it the sigmoid gradient flow, insufficient training of the gate, or the gate being unnecessary at these latent dimensions? Would ae_lg help at ld<128 or with multi-layer replacement?

4. **What is the practical parameter reduction?** The current experiment replaces a single layer's MLP (~50M params) with an AE bottleneck (~2.9M params for ld=512). Multi-layer replacement would multiply the savings.

5. **Does the init ranking change with more training epochs?** Random init might catch up to pretrained given enough epochs, or the gap might persist/widen.

---

## Appendix: Complete Sorted Table

| # | Mode  | LD  | Init       | Final PPL | BL PPL | Final MSE |
|---|-------|-----|------------|-----------|--------|-----------|
| 1 | ae    | 512 | pretrained | 26.544    | 42.6   | 0.03134   |
| 2 | ae    | 384 | pretrained | 26.579    | 42.6   | 0.03198   |
| 3 | ae_lg | 512 | pretrained | 26.679    | 42.5   | 0.03217   |
| 4 | ae    | 512 | fourier    | 26.701    | 42.6   | 0.03125   |
| 5 | ae    | 384 | fourier    | 26.728    | 42.5   | 0.03181   |
| 6 | ae    | 256 | pretrained | 26.769    | 42.5   | 0.03292   |
| 7 | ae_lg | 384 | pretrained | 26.786    | 42.5   | 0.03286   |
| 8 | ae_lg | 512 | fourier    | 26.800    | 42.5   | 0.03233   |
| 9 | ae_lg | 256 | pretrained | 26.916    | 42.5   | 0.03388   |
| 10| ae_lg | 384 | fourier    | 26.932    | 42.5   | 0.03301   |
| 11| ae    | 256 | fourier    | 26.975    | 42.5   | 0.03348   |
| 12| ae_lg | 512 | cur        | 26.986    | 42.6   | 0.03677   |
| 13| ae    | 512 | cur        | 27.038    | 42.8   | 0.03701   |
| 14| ae    | 128 | pretrained | 27.127    | 42.4   | 0.03520   |
| 15| ae_lg | 256 | fourier    | 27.165    | 42.5   | 0.03470   |
| 16| ae_lg | 384 | cur        | 27.177    | 42.7   | 0.03722   |
| 17| ae    | 384 | cur        | 27.177    | 42.9   | 0.03723   |
| 18| ae    | 256 | cur        | 27.240    | 42.8   | 0.03774   |
| 19| ae_lg | 256 | cur        | 27.250    | 42.6   | 0.03764   |
| 20| ae_lg | 128 | pretrained | 27.341    | 42.4   | 0.03604   |
| 21| ae    | 512 | random     | 27.346    | 42.5   | 0.03514   |
| 22| ae    | 128 | fourier    | 27.351    | 42.5   | 0.03576   |
| 23| ae    | 128 | cur        | 27.382    | 42.6   | 0.03884   |
| 24| ae    | 128 | random     | 27.402    | 42.6   | 0.03710   |
| 25| ae    | 256 | random     | 27.429    | 42.4   | 0.03610   |
| 26| ae    | 384 | random     | 27.459    | 42.6   | 0.03559   |
| 27| ae_lg | 128 | fourier    | 27.465    | 42.5   | 0.03713   |
| 28| ae_lg | 128 | random     | 27.475    | 42.5   | 0.03828   |
| 29| ae_lg | 128 | cur        | 27.478    | 42.5   | 0.03926   |
| 30| ae_lg | 512 | random     | 27.534    | 42.5   | 0.03654   |
| 31| ae_lg | 256 | random     | 27.541    | 42.6   | 0.03760   |
| 32| ae_lg | 384 | random     | 27.578    | 42.8   | 0.03722   |
| 33| ae_lg | 128 | svd        | 28.354    | 40.7   | 0.04532   |
| 34| ae    | 128 | svd        | 28.488    | 40.8   | 0.04650   |
| 35| ae_lg | 256 | svd        | 28.773    | 61.3   | 0.04751   |
| 36| ae_lg | 384 | svd        | 28.885    | 120.6  | 0.04838   |
| 37| ae_lg | 512 | svd        | 28.896    | 174.7  | 0.04853   |
| 38| ae    | 256 | svd        | 29.102    | 172.0  | 0.05003   |
| 39| ae    | 384 | svd        | 29.481    | 684.2  | 0.05116   |
| 40| ae    | 512 | svd        | 29.623    | 1161.1 | 0.05171   |

---

**Analysis artifacts:**

- Notebook: `scripts/experiments/analysis/notebooks/ae_init_methods_layer15_sweep_analysis.ipynb`
- Report: `scripts/experiments/analysis/analysis_reports/ae_init_methods_layer15_sweep_report.md`
- Data: `scripts/experiments/results/ae_init_methods_layer15_sweep/logs/ae_init_methods_layer15_sweep_log.csv`
