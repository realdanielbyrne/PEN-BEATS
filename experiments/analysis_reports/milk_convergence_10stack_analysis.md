# Milk Convergence 10-Stack Study — Results Analysis

## Abstract

This convergence study compares a 10-stack Generic N-BEATS baseline against the same architecture with `active_g=True` on the Milk univariate time series. Unlike the M4 benchmark studies, this experiment uses **validation loss** as the primary metric (no OWA) and focuses on training stability and convergence behaviour.

**Key Takeaways:**

- **Best configuration:** `Milk10_baseline` achieves median best val_loss = **1.5428**.
- **Divergence rate:** 0% of runs diverged across all conditions.
- **Configurations tested:** ['Milk10_baseline', 'Milk10_activeG']
- **Total compute:** 45.7 minutes across 200 runs.

- **CSV:** `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/milk_convergence_10stack/milk_convergence_10stack_results.csv`
- **Rows:** 200 (2 unique configs)
- **Parameters per model:** 8,161,280


## 1. Convergence Rates

Convergence rates indicate how many runs completed training without divergence.

| Config          |   N | Healthy    | Diverged   | Early Stopped   | Max Epochs   |
|:----------------|----:|:-----------|:-----------|:----------------|:-------------|
| Milk10_activeG  | 100 | 100 (100%) | 0 (0%)     | 100 (100%)      | 0 (0%)       |
| Milk10_baseline | 100 | 67 (67%)   | 1 (1%)     | 99 (99%)        | 0 (0%)       |

Overall divergence rate: 1/200 (0.5%). Some runs diverged; consider learning-rate tuning.


## 2. Validation Loss Statistics (Healthy Runs)

The loss ratio (final / best) measures overfitting: values near 1.0 indicate stable convergence, while values >> 1.0 suggest the model overfit past its best epoch.

| Config          |   Med Best Loss |   Med Final Loss |   Med Ratio |   Std Ratio |
|:----------------|----------------:|-----------------:|------------:|------------:|
| Milk10_activeG  |          2.4012 |           3.1361 |      1.3024 |      0.2616 |
| Milk10_baseline |          1.5428 |           1.9304 |      1.2168 |      0.2509 |


## 3. Best-Epoch Distribution (Healthy Runs)

The best epoch indicates how quickly each configuration converges to its optimal validation loss.

| Config          |   Med Epoch |   Min |   Max |   Std |
|:----------------|------------:|------:|------:|------:|
| Milk10_activeG  |          56 |    22 |   147 | 22.76 |
| Milk10_baseline |          63 |    28 |   144 | 26.68 |


## 4. Training Speed

Training speed comparison (wall-clock time per configuration).

| Config          |   Med Time (s) |   Med Epochs |   s/epoch |
|:----------------|---------------:|-------------:|----------:|
| Milk10_activeG  |           13.5 |           76 |      0.18 |
| Milk10_baseline |           12.9 |           86 |      0.15 |


## 5. active_g Head-to-Head (Mann-Whitney U)

A one-sided Mann-Whitney U test checks whether `active_g=True` produces stochastically lower validation loss than the baseline.

| Condition   |   Med best_val_loss |   N |
|:------------|--------------------:|----:|
| baseline    |              1.6962 | 100 |
| active_g    |              2.4012 | 100 |

- **Δ (active_g − baseline):** +0.7050
- **Mann-Whitney U p-value:** 1.0000 ns
- **Verdict:** No significant difference; baseline is comparable or better.

