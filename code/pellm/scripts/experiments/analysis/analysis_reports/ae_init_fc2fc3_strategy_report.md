# AE Init FC2/FC3 Strategy Analysis Report

**Date**: 2026-03-13
**Experiment**: `ae_init_fc2fc3_strategy`
**Analyst**: Claude Opus 4.6

## Executive Summary

This experiment tested whether using algorithm-specific initialization for the inner bottleneck layers (fc2/fc3) of the AE MLP -- instead of defaulting to SVD decomposition -- would improve perplexity. **The answer is definitively no.** Algorithm-specific fc2/fc3 init degrades PPL by 0.24-0.42 across all non-SVD configurations compared to the prior SVD fallback approach. The SVD fallback should be reinstated.

## Experiment Design

**What changed**: Previously, all init modes (pretrained, CUR, Fourier) used SVD decomposition as a fallback for the inner bottleneck layers fc2 (`hidden/2 -> latent`) and fc3 (`latent -> hidden/2`). Only the outer layers fc1 (`hidden -> hidden/2`) and fc4 (`hidden/2 -> hidden`) used the algorithm-specific strategy. This experiment extends each algorithm's strategy to all 4 AE layers.

**Setup**: 2 modes (ae, ae_lg) x 4 inits (pretrained, fourier, cur, svd) x 3 seeds = 24 runs. Fixed: ld=512, layer 15, lr=1e-4, 10 AE pretrain epochs, no LM fine-tuning.

**Data**: 35 successful runs (1 failed). ae_lg runs were duplicated between two config blocks; deduplicated to 24 unique runs.

## Results

### 1. FC2/FC3 Strategy Impact (Primary Finding)

Comparison of this experiment (algorithm-specific fc2/fc3) vs prior sweep (SVD fallback for fc2/fc3):

| Mode   | Init       | Prior PPL (SVD fb) | New PPL (algo-specific) | Delta   | Direction |
|--------|------------|--------------------:|------------------------:|--------:|-----------|
| ae     | pretrained |               26.544 |                  26.889 |  +0.345 | WORSE     |
| ae     | fourier    |               26.701 |                  27.045 |  +0.344 | WORSE     |
| ae     | cur        |               27.038 |                  27.298 |  +0.260 | WORSE     |
| ae     | svd        |               29.623 |                  29.539 |  -0.084 | ~SAME     |
| ae_lg  | pretrained |               26.679 |                  27.073 |  +0.394 | WORSE     |
| ae_lg  | fourier    |               26.800 |                  27.215 |  +0.416 | WORSE     |
| ae_lg  | cur        |               26.986 |                  27.228 |  +0.242 | WORSE     |
| ae_lg  | svd        |               28.896 |                  28.882 |  -0.013 | ~SAME     |

The pattern is unambiguous: **6/6 non-SVD configurations are worse**, with degradation of +0.24 to +0.42 ppl. The SVD control runs are unchanged (as expected, since their fc2/fc3 strategy did not change).

**Why SVD is better for fc2/fc3**: The inner bottleneck layers map between `hidden/2` (1024) and `latent` (512) -- a moderate compression. SVD provides the mathematically optimal rank-k approximation (Eckart-Young theorem), which is especially effective for these moderately-compressed linear mappings. Truncation, CUR, and Fourier filtering are heuristic approximations that sacrifice this optimality.

### 2. Within-Experiment Rankings

```
1. ae+pretrained:  ppl=26.889 +/- 0.052 (n=3) -- BEST
2. ae+fourier:     ppl=27.045 +/- 0.028 (n=3) -- +0.156
3. ae_lg+pretrained: ppl=27.073 +/- 0.048 (n=3) -- +0.184
4. ae_lg+fourier:  ppl=27.215 +/- 0.023 (n=3) -- +0.326
5. ae_lg+cur:      ppl=27.228 +/- 0.018 (n=3) -- +0.339
6. ae+cur:         ppl=27.298 +/- 0.019 (n=3) -- +0.409
7. ae_lg+svd:      ppl=28.882 +/- 0.058 (n=3) -- +1.993
8. ae+svd:         ppl=29.539 +/- 0.091 (n=3) -- +2.651
```

Init ranking is identical to the prior sweep: **pretrained > fourier > cur >> svd**.

### 3. ae vs ae_lg Comparison

| Init       | ae PPL  | ae_lg PPL | Diff    | Winner |
|------------|--------:|----------:|--------:|--------|
| pretrained |  26.889 |    27.073 |  +0.184 | ae     |
| fourier    |  27.045 |    27.215 |  +0.171 | ae     |
| cur        |  27.298 |    27.228 |  -0.070 | ae_lg  |
| svd        |  29.539 |    28.882 |  -0.657 | ae_lg  |

ae wins on the two best inits (pretrained, fourier). ae_lg wins on CUR (marginally) and SVD (due to lower baseline disruption from the gate). For practical use, **ae remains the recommended mode**.

### 4. MSE Convergence

| Mode | Init       | Epoch 1 MSE | Final MSE | PPL    |
|------|------------|------------:|----------:|-------:|
| ae   | pretrained |       0.122 |     0.033 | 26.889 |
| ae   | fourier    |       0.129 |     0.033 | 27.045 |
| ae_lg| pretrained |       0.130 |     0.034 | 27.073 |
| ae_lg| fourier    |       0.139 |     0.035 | 27.215 |
| ae_lg| cur        |       0.113 |     0.038 | 27.228 |
| ae   | cur        |       0.109 |     0.038 | 27.298 |
| ae_lg| svd        |       0.184 |     0.049 | 28.882 |
| ae   | svd        |       0.205 |     0.052 | 29.539 |

MSE-PPL Pearson correlation: r=0.985 (p=3.8e-18). MSE remains an excellent proxy for PPL.

### 5. Statistical Notes

With n=3 per group, Mann-Whitney U tests yield p=0.10 for all pairwise comparisons (the minimum achievable p-value for n=3 vs n=3). However, **all 6 non-SVD comparisons show degradation in the same direction**, and the effect sizes are 4-8x larger than the within-group standard deviation. The consistency across conditions compensates for the limited per-condition power.

## Recommendations

### Current Best Configuration (Unchanged)

```yaml
pe_mlp_mode: ae
ae_init: pretrained       # with SVD fallback for fc2/fc3 (prior behavior)
ae_latent_dim: 512
lr: 1.0e-4
pe_mlp_layer_indices: [15]
ae_pretrain_epochs: 10
```

**Expected PPL**: ~26.54 (1.40x vanilla Llama)

### Action: Revert FC2/FC3 Strategy

The `ae_init` code should be reverted to use SVD decomposition for fc2/fc3 regardless of the chosen init mode. Algorithm-specific strategies should only apply to fc1/fc4.

### What to Test Next

1. **More AE pre-training epochs (20-30)**: MSE still declining ~1%/epoch at epoch 10. The strong MSE-PPL correlation predicts proportional PPL gains.

2. **Multi-layer AE replacement**: Test layers [14, 15], then [13, 14, 15]. Single-layer experiments have converged on optimal settings; the next frontier is scaling.

3. **Separate AE pretrain learning rate**: Try ae_pretrain_lr=3e-4 while keeping LM lr=1e-4. Faster AE convergence could close the remaining gap to vanilla.

4. **LM fine-tuning after AE pre-training**: All current experiments use epochs=0. Even 1-2 epochs of end-to-end training might provide additional recovery.

### Open Questions

- Why does ae_lg mitigate SVD baseline inflation (175 vs 1161)? The learned gate appears to dampen unstable initializations, but this has not been formally analyzed.
- Would the SVD fallback also be optimal for fc2/fc3 at smaller latent dimensions (128, 256)? The current experiment only tests ld=512.
- Is the fc2/fc3 finding specific to layer 15, or does it generalize across decoder layers?

## Artifacts

- **Notebook**: `scripts/experiments/analysis/notebooks/ae_init_fc2fc3_strategy_analysis.ipynb`
- **Data**: `scripts/experiments/results/ae_init_fc2fc3_strategy/logs/ae_init_fc2fc3_strategy_log.csv`
- **YAML config**: `scripts/experiments/ae_init_fc2fc3_strategy.yaml`
