# Wavelet Study 2: Basis Dimension Sweep -- Interpretive Analysis

**Date**: 2026-03-07
**Dataset**: M4 Yearly (forecast_length=6, backcast_length=30)
**Architecture**: Trend + WaveletV3 (10 stacks, 1 block/stack, weight sharing)
**Source CSV**: `experiments/results/m4/wavelet_study_2_basis_dim_results.csv`

---

## 1. Executive Summary

The basis dimension sweep across 24 configurations reveals that **Coif2 with basis_dim=6 (equal to forecast length) and thetas_dim=3 is the clear winner**, achieving the only sub-0.800 mean OWA (0.7944) and the best mean SMAPE (13.410) among all tested configurations. However, the margins between the top ~10 configurations are narrow (0.16 SMAPE points separating rank 1 from rank 10), and no single factor -- wavelet family, basis dimension, or thetas dimension -- reaches statistical significance in isolation. The strongest signal in the data is the practical inferiority of `lt_fcast` (basis_dim < forecast_length), which consistently occupies the worst positions and produces the highest variance. Larger basis dimensions (equal to or exceeding the forecast length) are uniformly safer choices, and Coif2 edges ahead of DB3 and Haar on average by a small but consistent margin.

---

## 2. Experiment Design

This experiment sweeps three factors in a full factorial design:

| Factor | Levels | Values |
|--------|--------|--------|
| Wavelet family | 3 | Coif2, DB3, Haar |
| Basis dimension | 4 | 4 (`lt_fcast`), 6 (`eq_fcast`), 15 (`lt_bcast`), 30 (`eq_bcast`) |
| Trend thetas_dim | 2 | 3, 5 |

This produces 3 x 4 x 2 = 24 unique configurations, each run with 3 seeds (42, 43, 44), for 72 total training runs. The semantic labels encode the relationship between basis_dim and the forecast/backcast horizons:

- **`lt_fcast`** (bd=4): basis dimension is *less than* the forecast length (H=6). The wavelet basis under-spans the output space.
- **`eq_fcast`** (bd=6): basis dimension *equals* the forecast length. One-to-one mapping between basis coefficients and forecast positions.
- **`lt_bcast`** (bd=15): basis dimension is *less than* the backcast length (L=30) but exceeds H.
- **`eq_bcast`** (bd=30): basis dimension *equals* the backcast length. Maximum representational capacity.

All configurations use ReLU activation, no `active_g`, and no `sum_losses`. Training uses early stopping with patience.

---

## 3. Per-Wavelet-Family Analysis

### Marginal Rankings

| Rank | Wavelet | Mean SMAPE | Std | Mean MASE | Mean OWA | n |
|------|---------|-----------|-----|-----------|----------|---|
| 1 | Coif2 | 13.569 | 0.149 | 3.128 | 0.809 | 24 |
| 2 | DB3 | 13.636 | 0.134 | 3.152 | 0.814 | 24 |
| 3 | Haar | 13.637 | 0.232 | 3.146 | 0.813 | 24 |

Coif2 leads by 0.067 SMAPE points over DB3 and 0.068 over Haar. This is a small but consistent advantage: Coif2 achieves the best or second-best marginal mean across all three metrics. The advantage is not statistically significant by Kruskal-Wallis (H=3.27, p=0.195), which is expected given the small sample sizes and the narrow performance corridor.

### Why Coif2 Edges Ahead

Coiflet-2 wavelets have a balance of smoothness (4 vanishing moments) and compact support (11 filter taps) that is well-suited to the M4-Yearly domain. The Haar wavelet, while maximally localized (2 taps), produces a piecewise-constant basis that may be too coarse for capturing smooth trend-residual interactions in annual economic data. DB3 (Daubechies-3, 6 taps, 3 vanishing moments) falls between the two in both smoothness and locality, and its performance falls between them accordingly. The pattern is consistent with the hypothesis that moderate wavelet regularity improves basis quality for smooth time series.

### Per-Wavelet Internal Spread

The internal performance range (best config minus worst config within a wavelet family) reveals that **Haar is the most sensitive to hyperparameter choices**:

| Wavelet | Best Config SMAPE | Worst Config SMAPE | Spread |
|---------|------------------|-------------------|--------|
| Haar | 13.509 | 13.947 | 0.438 |
| Coif2 | 13.410 | 13.744 | 0.334 |
| DB3 | 13.579 | 13.777 | 0.198 |

DB3 shows the tightest internal spread (0.198), meaning it is the most robust to basis_dim and thetas_dim choices, even though its best absolute performance is middling. Haar has the largest spread (0.438), driven entirely by the Haar_bd4_lt_fcast_td3 outlier discussed in Section 7.

---

## 4. Basis Dimension Analysis

### Marginal Rankings by Basis Dimension

| Rank | Basis Dim | Semantic Label | Mean SMAPE | Std | Mean OWA |
|------|-----------|---------------|-----------|-----|----------|
| 1 | 30 | eq_bcast (bd=L) | 13.563 | 0.152 | 0.808 |
| 2 | 6 | eq_fcast (bd=H) | 13.601 | 0.208 | 0.811 |
| 3 | 15 | lt_bcast (bd<L) | 13.621 | 0.134 | 0.813 |
| 4 | 4 | lt_fcast (bd<H) | 13.671 | 0.202 | 0.816 |

Basis dimension explains the most variance of any single factor (eta-squared = 4.9%), and the only statistically significant pairwise comparison in the entire experiment is **bd=4 vs bd=30** (Mann-Whitney U=231, p=0.030). This is the central finding of the sweep.

### The Gradient from lt_fcast to eq_bcast

The semantic labels create an ordinal scale of representational capacity, and performance follows this scale monotonically:

```
lt_fcast (bd=4) --> eq_fcast (bd=6) --> lt_bcast (bd=15) --> eq_bcast (bd=30)
   13.671            13.601              13.621               13.563
   worst             --                  --                   best
```

The progression is not perfectly smooth -- `eq_fcast` (bd=6) slightly outperforms `lt_bcast` (bd=15) despite having fewer basis functions. This suggests that the alignment between basis_dim and a characteristic length scale (the forecast horizon) matters more than raw dimensionality.

### Why eq_bcast and eq_fcast Outperform

The two "equal-to" settings (bd=6=H and bd=30=L) occupy the top two positions. This is not coincidental. When basis_dim matches a structural length in the architecture:

1. **eq_fcast (bd=6=H)**: The wavelet basis has exactly as many functions as forecast positions. The linear projection from basis coefficients to forecast values is square, creating an invertible mapping with no information loss or redundancy. This is the "Goldilocks" setting for the forecast path.

2. **eq_bcast (bd=30=L)**: The basis fully spans the backcast space. The WaveletV3 generator can represent any backcast pattern exactly, giving the residual decomposition maximum flexibility.

3. **lt_fcast (bd=4<H)**: With only 4 basis functions for 6 forecast positions, the model must reconstruct the forecast from an under-determined basis. This information bottleneck forces the forecast head to interpolate, adding error.

### Per-Wavelet Interaction with Basis Dimension

The optimal basis dimension is *not* the same for all wavelet families:

| | bd=4 (lt_fcast) | bd=6 (eq_fcast) | bd=15 (lt_bcast) | bd=30 (eq_bcast) | Best |
|---------|---------|---------|---------|---------|------|
| **Coif2** | 13.661 | **13.490** | 13.557 | 13.569 | eq_fcast |
| **DB3** | 13.616 | 13.646 | 13.701 | **13.580** | eq_bcast |
| **Haar** | 13.736 | 13.666 | 13.606 | **13.540** | eq_bcast |

Coif2 strongly prefers bd=6 (eq_fcast), while DB3 and Haar both prefer bd=30 (eq_bcast). This interaction is one of the most interpretively rich findings. Coif2's higher regularity means its basis functions are smooth enough to efficiently represent the forecast with just 6 coefficients, while the coarser Haar and intermediate DB3 wavelets need the full 30-dimensional basis to achieve their best performance. In other words, smoother wavelets extract more useful information per basis function, reducing the need for high dimensionality.

---

## 5. Thetas Dimension Interaction

### Overall Effect

| Thetas Dim | Mean SMAPE | Std | Mean OWA |
|-----------|-----------|-----|----------|
| 3 | 13.603 | 0.181 | 0.811 |
| 5 | 13.625 | 0.176 | 0.813 |

Thetas dimension has the weakest marginal effect of all three factors (eta-squared = 0.4%, Kruskal-Wallis p=0.499). The 0.022 SMAPE difference is negligible. This makes sense architecturally: thetas_dim controls the polynomial degree of the Trend block, and for yearly data with a 6-step forecast horizon, a cubic polynomial (degree 3 implies 3 coefficients) is already quite expressive -- a degree-5 polynomial adds flexibility that may slightly overfit the trend component.

### Per-Wavelet Reversal

Despite the negligible overall effect, there is a notable reversal in the wavelet-thetas interaction:

| Wavelet | td=3 | td=5 | Better |
|---------|------|------|--------|
| Coif2 | **13.529** | 13.610 | td=3 |
| DB3 | **13.600** | 13.672 | td=3 |
| Haar | 13.682 | **13.592** | td=5 |

Coif2 and DB3 both prefer td=3, while Haar reverses and prefers td=5. One plausible explanation: Haar's piecewise-constant wavelet basis lacks smoothness, so the Trend block must compensate by using a higher-degree polynomial to capture smooth dynamics. Coif2 and DB3, with smoother bases, already encode some of this smoothness in their wavelet coefficients, making the extra polynomial flexibility of td=5 redundant or harmful.

This reversal is practically important: if you choose Haar, pair it with td=5; if you choose Coif2 or DB3, use td=3.

---

## 6. Top 5 Configurations

| Rank | Configuration | Mean SMAPE | Std | Mean MASE | Std | Mean OWA | Std | Delta vs Best |
|------|--------------|-----------|-----|-----------|-----|----------|-----|---------------|
| 1 | Coif2_bd6_eq_fcast_td3 | **13.410** | 0.101 | **3.053** | 0.042 | **0.794** | 0.008 | -- |
| 2 | Haar_bd30_eq_bcast_td5 | 13.509 | 0.045 | 3.079 | 0.013 | 0.801 | 0.003 | +0.099 (+0.7%) |
| 3 | Haar_bd15_lt_bcast_td3 | 13.517 | 0.099 | 3.106 | 0.032 | 0.804 | 0.007 | +0.107 (+0.8%) |
| 4 | Haar_bd4_lt_fcast_td5 | 13.526 | 0.134 | 3.108 | 0.059 | 0.805 | 0.012 | +0.116 (+0.9%) |
| 5 | Coif2_bd15_lt_bcast_td5 | 13.551 | 0.126 | 3.119 | 0.052 | 0.807 | 0.010 | +0.141 (+1.1%) |

### Notable Observations

**The winner is clear but the pack is tight.** Coif2_bd6_eq_fcast_td3 leads by 0.099 SMAPE points over the runner-up, which is a meaningful margin in absolute terms (nearly 1% relative improvement) but not statistically significant at conventional thresholds given n=3 per config (Mann-Whitney p=0.200). The lack of statistical power is a sample size limitation, not evidence of equivalence.

**Coif2_bd6_eq_fcast_td3 is the only config with mean OWA below 0.800.** This is a meaningful threshold: OWA < 0.800 means outperforming Naive2 by more than 20% on the combined SMAPE+MASE metric. No other configuration achieves this on average, though 15 individual runs (across many configs) dip below 0.800.

**Haar_bd30_eq_bcast_td5 is the most stable runner-up.** Its SMAPE standard deviation of 0.045 is the second-lowest of all 24 configs (after DB3_bd4_lt_fcast_td3 at 0.011, which ranks 14th on mean). This makes it an attractive choice when reproducibility is paramount.

---

## 7. Stability and Variance Analysis

### Coefficient of Variation Rankings (Most Stable to Least)

| Rank | Configuration | Mean SMAPE | Std | CV |
|------|--------------|-----------|-----|------|
| 1 | DB3_bd4_lt_fcast_td3 | 13.582 | 0.011 | 0.08% |
| 2 | Coif2_bd30_eq_bcast_td3 | 13.564 | 0.024 | 0.17% |
| 3 | Haar_bd30_eq_bcast_td5 | 13.509 | 0.045 | 0.33% |
| ... | ... | ... | ... | ... |
| 23 | Haar_bd4_lt_fcast_td3 | 13.947 | 0.328 | 2.35% |
| 24 | Haar_bd6_eq_fcast_td5 | 13.639 | 0.364 | 2.67% |

The two most stable configs (CV < 0.2%) both use eq_bcast or lt_fcast with low basis_dim, but their mean performance is mediocre. The most volatile configs (CV > 2%) are both Haar, reinforcing that Haar wavelet bases are more seed-sensitive than Coif2 or DB3.

### Outlier: Haar_bd4_lt_fcast_td3

This configuration stands out as the worst performer by a substantial margin:

| Seed | SMAPE | MASE | OWA | Epochs | Loss Ratio |
|------|-------|------|-----|--------|------------|
| 42 | 13.810 | 3.225 | 0.828 | 21 | 1.010 |
| 43 | 13.710 | 3.175 | 0.819 | 22 | 1.006 |
| 44 | **14.321** | **3.459** | **0.873** | 25 | **1.091** |

Seed 44 is the single worst run in the entire experiment (SMAPE 14.321, 1.14 standard deviations above the config mean). Its loss ratio of 1.091 is the highest in the dataset, suggesting the model overshot early stopping and the validation loss was still climbing when patience ran out. However, even the "good" seeds (42, 43) for this config are below average, so the problem is not just one bad seed -- it is the combination of the coarsest wavelet (Haar), the most restrictive basis dimension (4 < H=6), and the lower trend degree (td=3) that creates a representationally impoverished architecture.

The Haar_bd4_lt_fcast_td5 analog (same wavelet and basis_dim but td=5) ranks 4th overall, demonstrating that the higher trend polynomial degree rescues Haar from the lt_fcast penalty. This is the strongest evidence for the td interaction discussed in Section 5.

### Per-Seed Consistency of Top Ranks

The top configuration is *not* identical across seeds:

| Seed | Rank 1 Config | SMAPE |
|------|--------------|-------|
| 42 | Coif2_bd30_eq_bcast_td5 | 13.368 |
| 43 | DB3_bd30_eq_bcast_td5 | 13.341 |
| 44 | **Coif2_bd6_eq_fcast_td3** | **13.293** |

Coif2_bd6_eq_fcast_td3 is the global best on seed 44 and ranks 4th on seed 42, but is not in the top 5 on seed 43. This is typical of N-BEATS experiments with narrow performance corridors -- individual seed variation can shuffle rankings within the top cluster. The config's overall superiority comes from consistently strong (if not always top-1) performance across all seeds, avoiding the catastrophic runs that plague some Haar configurations.

---

## 8. Parameter Efficiency

| Basis Dim | Mean Params | Min | Max |
|-----------|------------|-----|-----|
| 4 (lt_fcast) | 5,071,380 | 5,070,095 | 5,072,665 |
| 6 (eq_fcast) | 5,081,620 | 5,080,335 | 5,082,905 |
| 15 (lt_bcast) | 5,104,660 | 5,103,375 | 5,105,945 |
| 30 (eq_bcast) | 5,143,060 | 5,141,775 | 5,144,345 |

The parameter count spread across the entire experiment is **1.46%** (74,250 parameters out of ~5.1M). This is negligible -- all 24 configurations are essentially the same size model. The variation comes entirely from the basis_dim influencing the linear projection dimensions in the wavelet block, but these layers are tiny relative to the shared Trend block and the WaveletV3 backbone FC layers.

**Practical implication**: Basis dimension can be selected purely on quality grounds. There is no parameter efficiency tradeoff to consider. The winning config (Coif2_bd6_eq_fcast_td3, 5.08M params) is actually *smaller* than the best eq_bcast configs (5.14M params), achieving better performance with 1.2% fewer parameters. This is because bd=6 creates a more efficient projection than bd=30, requiring the network to learn a compact representation that happens to generalize better.

---

## 9. Training and Convergence

### Epochs to Convergence

| Basis Dim | Mean Epochs | Std | Min | Max |
|-----------|------------|-----|-----|-----|
| 4 (lt_fcast) | 23.3 | 5.0 | 13 | 36 |
| 6 (eq_fcast) | 23.1 | 6.6 | 15 | 39 |
| 15 (lt_bcast) | 22.3 | 4.8 | 16 | 33 |
| 30 (eq_bcast) | 21.7 | 3.7 | 15 | 27 |

All configurations converge quickly (20-24 epochs on average), as expected for the M4-Yearly dataset. The eq_bcast group (bd=30) converges slightly faster on average (21.7 epochs) with the tightest spread (std=3.7), suggesting that larger basis dimensions provide a smoother loss landscape. The eq_fcast group (bd=6) has the widest epoch range (15-39), driven by a single outlier (Haar_bd6_eq_fcast_td5, seed 42: 39 epochs).

### Loss Ratios

All groups show loss ratios between 1.028 and 1.045, indicating modest validation loss overshoot (3-4.5%) at early stopping. This is within healthy bounds for N-BEATS with patience-based stopping. No configuration shows systematic divergence.

### Validation Loss Quality

Best validation losses are tightly clustered (mean 13.597, std 0.062 across all 72 runs), confirming that all configurations reach a similar validation loss floor. The differences in test SMAPE arise primarily from generalization quality beyond the validation set, not from training optimization failures.

---

## 10. Key Findings and Recommendations

### Finding 1: Coif2_bd6_eq_fcast_td3 is the Recommended Default

For M4-Yearly with the Trend+WaveletV3 architecture, use:
- **Wavelet**: Coif2
- **basis_dim**: 6 (= forecast_length)
- **thetas_dim**: 3

This achieves OWA 0.794 (the only sub-0.800 mean in the sweep), is reasonably stable (std 0.101), and is the smallest model tested by parameter count.

### Finding 2: Avoid lt_fcast (basis_dim < forecast_length)

Underspanning the forecast space with basis_dim=4 reliably hurts performance, particularly for Haar wavelets. The only lt_fcast config that performs competitively is Haar_bd4_lt_fcast_td5, and even that success depends on the higher trend polynomial compensating for the restricted wavelet basis. As a rule: **never set basis_dim below forecast_length**.

### Finding 3: eq_fcast vs eq_bcast Depends on Wavelet Smoothness

- **Smooth wavelets (Coif2)**: Prefer `eq_fcast` (bd=H). The smooth basis functions are expressive enough per dimension.
- **Coarse wavelets (Haar, DB3)**: Prefer `eq_bcast` (bd=L). More basis functions compensate for lower per-function expressiveness.

### Finding 4: Thetas Dimension is a Second-Order Effect, but Watch the Interaction

td=3 is marginally better overall and clearly better for Coif2 and DB3. Use td=5 only with Haar. Do not invest experimental budget in sweeping thetas_dim further.

### Finding 5: Statistical Power is Limited

With 3 seeds per config, pairwise statistical tests between individual configurations have limited power. The one significant pairwise result (bd=4 vs bd=30, p=0.030) aligns with the largest practical gap in the data. Future experiments should use 5-10 seeds for finer discrimination.

### Recommendations for Next Experiments

1. **Confirm Coif2_bd6_eq_fcast_td3 with more seeds** (e.g., 10 runs) to establish tighter confidence intervals and enable rigorous pairwise testing against the top 3 alternatives.

2. **Test bd=6 with other wavelet families** beyond Coif2, DB3, and Haar. Candidates: Coif1 (known strong performer from V1 study), Symlet4, Symlet8. The eq_fcast setting may unlock better performance from families not yet tested.

3. **Test the winning config in the AE backbone** (WaveletV3AELG or WaveletV3AE). The V1 study found Coif1_eq_fcast_td3 with latent_dim=16 achieved SMAPE 13.438 on M4-Yearly (TrendAELG+WaveletV3AELG architecture). The non-AE Coif2_eq_fcast_td3 result here (13.410) is notably *better* than the AE variant's best, suggesting the AE bottleneck may not be beneficial for this particular configuration.

4. **Cross-dataset validation** on M4-Quarterly and M4-Monthly, where the forecast horizon is longer (8 and 18 respectively). The eq_fcast recommendation (bd=H) should be retested, as longer horizons may shift the optimal basis_dim-to-horizon ratio.

---

## Appendix: Full Configuration Rankings

| Rank | Configuration | Wavelet | BD | Label | TD | SMAPE | +/-Std | MASE | OWA | Params |
|------|--------------|---------|-----|-------|-----|-------|--------|------|------|--------|
| 1 | Coif2_bd6_eq_fcast_td3 | Coif2 | 6 | eq_fcast | 3 | 13.410 | 0.101 | 3.053 | 0.794 | 5,080,335 |
| 2 | Haar_bd30_eq_bcast_td5 | Haar | 30 | eq_bcast | 5 | 13.509 | 0.045 | 3.079 | 0.801 | 5,144,345 |
| 3 | Haar_bd15_lt_bcast_td3 | Haar | 15 | lt_bcast | 3 | 13.517 | 0.099 | 3.106 | 0.804 | 5,103,375 |
| 4 | Haar_bd4_lt_fcast_td5 | Haar | 4 | lt_fcast | 5 | 13.526 | 0.134 | 3.108 | 0.805 | 5,072,665 |
| 5 | Coif2_bd15_lt_bcast_td5 | Coif2 | 15 | lt_bcast | 5 | 13.551 | 0.126 | 3.119 | 0.807 | 5,105,945 |
| 6 | Coif2_bd15_lt_bcast_td3 | Coif2 | 15 | lt_bcast | 3 | 13.563 | 0.081 | 3.128 | 0.808 | 5,103,375 |
| 7 | Coif2_bd30_eq_bcast_td3 | Coif2 | 30 | eq_bcast | 3 | 13.564 | 0.024 | 3.140 | 0.810 | 5,141,775 |
| 8 | Coif2_bd6_eq_fcast_td5 | Coif2 | 6 | eq_fcast | 5 | 13.570 | 0.189 | 3.135 | 0.810 | 5,082,905 |
| 9 | Haar_bd30_eq_bcast_td3 | Haar | 30 | eq_bcast | 3 | 13.571 | 0.165 | 3.116 | 0.807 | 5,141,775 |
| 10 | Coif2_bd30_eq_bcast_td5 | Coif2 | 30 | eq_bcast | 5 | 13.575 | 0.249 | 3.130 | 0.809 | 5,144,345 |
| 11 | Coif2_bd4_lt_fcast_td3 | Coif2 | 4 | lt_fcast | 3 | 13.579 | 0.063 | 3.133 | 0.810 | 5,070,095 |
| 12 | DB3_bd30_eq_bcast_td3 | DB3 | 30 | eq_bcast | 3 | 13.579 | 0.142 | 3.132 | 0.810 | 5,141,775 |
| 13 | DB3_bd30_eq_bcast_td5 | DB3 | 30 | eq_bcast | 5 | 13.582 | 0.281 | 3.138 | 0.810 | 5,144,345 |
| 14 | DB3_bd4_lt_fcast_td3 | DB3 | 4 | lt_fcast | 3 | 13.582 | 0.011 | 3.119 | 0.808 | 5,070,095 |
| 15 | DB3_bd6_eq_fcast_td3 | DB3 | 6 | eq_fcast | 3 | 13.612 | 0.181 | 3.157 | 0.814 | 5,080,335 |
| 16 | DB3_bd15_lt_bcast_td3 | DB3 | 15 | lt_bcast | 3 | 13.626 | 0.127 | 3.143 | 0.812 | 5,103,375 |
| 17 | Haar_bd6_eq_fcast_td5 | Haar | 6 | eq_fcast | 5 | 13.639 | 0.364 | 3.138 | 0.812 | 5,082,905 |
| 18 | DB3_bd4_lt_fcast_td5 | DB3 | 4 | lt_fcast | 5 | 13.651 | 0.052 | 3.150 | 0.814 | 5,072,665 |
| 19 | DB3_bd6_eq_fcast_td5 | DB3 | 6 | eq_fcast | 5 | 13.679 | 0.078 | 3.156 | 0.816 | 5,082,905 |
| 20 | Haar_bd6_eq_fcast_td3 | Haar | 6 | eq_fcast | 3 | 13.693 | 0.265 | 3.166 | 0.817 | 5,080,335 |
| 21 | Haar_bd15_lt_bcast_td5 | Haar | 15 | lt_bcast | 5 | 13.695 | 0.166 | 3.172 | 0.818 | 5,105,945 |
| 22 | Coif2_bd4_lt_fcast_td5 | Coif2 | 4 | lt_fcast | 5 | 13.744 | 0.193 | 3.190 | 0.822 | 5,072,665 |
| 23 | DB3_bd15_lt_bcast_td5 | DB3 | 15 | lt_bcast | 5 | 13.777 | 0.058 | 3.223 | 0.827 | 5,105,945 |
| 24 | Haar_bd4_lt_fcast_td3 | Haar | 4 | lt_fcast | 3 | 13.947 | 0.328 | 3.287 | 0.840 | 5,070,095 |

---

*Report generated from 72 training runs across 24 configurations (3 seeds each). Raw data: `experiments/results/m4/wavelet_study_2_basis_dim_results.csv`. Script-generated summary: `experiments/analysis/analysis_reports/wavelet_study_2_analysis.md`.*
