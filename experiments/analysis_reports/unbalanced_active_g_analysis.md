# Unbalanced active_g Study — Results Analysis

## Abstract

This study investigates four `active_g` modes — False (baseline), True (balanced), forecast-only, and backcast-only — on the Milk6 univariate forecasting task with a 6-stack Generic architecture. The experiment measures convergence stability, validation loss, and pairwise statistical significance to determine whether asymmetric activation on the backcast or forecast output layers affects training.

**Key Takeaways:**

- **Best condition:** `Milk6_baseline` achieves median best_val_loss = **1.4726**.
- **Conditions tested:** ['False', 'True (balanced)', 'forecast-only', 'backcast-only']
- **Total compute:** 67.5 minutes across 400 runs (100 runs per condition).

**Condition Mapping:**

| Config                     | active_g        |
|:---------------------------|:----------------|
| Milk6_baseline             | False           |
| Milk6_activeG              | True (balanced) |
| Milk6_activeG_forecastOnly | forecast-only   |
| Milk6_activeG_backcastOnly | backcast-only   |

- **CSV:** `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/milk_convergence/milk_convergence_results.csv`
- **Rows:** 400 (4 configs)
- **Primary metric:** best_val_loss (lower = better; no OWA for this dataset).


## 1. Convergence Rates

Convergence rates per condition show how training stability varies with active_g mode.

| Config                     | active_g        |   N | Healthy    | Diverged   | Early Stopped   |
|:---------------------------|:----------------|----:|:-----------|:-----------|:----------------|
| Milk6_baseline             | False           | 100 | 70 (70%)   | 1 (1%)     | 99 (99%)        |
| Milk6_activeG              | True (balanced) | 100 | 100 (100%) | 0 (0%)     | 100 (100%)      |
| Milk6_activeG_forecastOnly | forecast-only   | 100 | 100 (100%) | 0 (0%)     | 100 (100%)      |
| Milk6_activeG_backcastOnly | backcast-only   | 100 | 63 (63%)   | 3 (3%)     | 97 (97%)        |


## 2. Validation Loss Statistics (Healthy Runs)

Validation loss statistics for healthy runs. The loss ratio (final / best) indicates overfitting tendency.

| Config                     | active_g        |   Med Best Loss |   Med Ratio |   Std Ratio |
|:---------------------------|:----------------|----------------:|------------:|------------:|
| Milk6_baseline             | False           |          1.4726 |      1.283  |      0.216  |
| Milk6_activeG              | True (balanced) |          2.6172 |      1.2743 |      0.2819 |
| Milk6_activeG_forecastOnly | forecast-only   |          2.2391 |      1.2712 |      0.2686 |
| Milk6_activeG_backcastOnly | backcast-only   |          2.1625 |      1.255  |      0.2406 |


## 3. Best-Epoch Distribution (Healthy Runs)

Best-epoch distribution indicates convergence speed for each active_g mode.

| Config                     | active_g        |   Med Epoch |   Min |   Max |   Std |
|:---------------------------|:----------------|------------:|------:|------:|------:|
| Milk6_baseline             | False           |          92 |    47 |   190 | 36.11 |
| Milk6_activeG              | True (balanced) |          57 |    29 |   129 | 21.92 |
| Milk6_activeG_forecastOnly | forecast-only   |          56 |    26 |   145 | 25.06 |
| Milk6_activeG_backcastOnly | backcast-only   |          91 |    36 |   209 | 40.47 |


## 4. Stability Across Seeds

Stability is measured by the spread (max − min) and standard deviation of best_val_loss across random seeds.

| Config                     | active_g        |   Med Loss |   Range |    Std |
|:---------------------------|:----------------|-----------:|--------:|-------:|
| Milk6_baseline             | False           |     1.4726 |  1.104  | 0.2083 |
| Milk6_activeG              | True (balanced) |     2.6172 |  1.4361 | 0.2759 |
| Milk6_activeG_forecastOnly | forecast-only   |     2.2391 |  1.7625 | 0.3597 |
| Milk6_activeG_backcastOnly | backcast-only   |     2.1625 |  1.5725 | 0.4211 |


## 5. Training Speed

Wall-clock training time per condition.

| Config                     | active_g        |   Med Time (s) |   Med Epochs |
|:---------------------------|:----------------|---------------:|-------------:|
| Milk6_baseline             | False           |           11.6 |        106.5 |
| Milk6_activeG              | True (balanced) |            9.1 |         77   |
| Milk6_activeG_forecastOnly | forecast-only   |            7.2 |         76   |
| Milk6_activeG_backcastOnly | backcast-only   |           10.4 |        111   |


## 6. Pairwise Statistical Significance (Mann-Whitney U)

Each pair is tested with a one-sided Mann-Whitney U to determine which condition produces stochastically lower validation loss.

| Config A                   | Config B                   |   Δ med |   p-value | Sig   |
|:---------------------------|:---------------------------|--------:|----------:|:------|
| Milk6_baseline             | Milk6_activeG              |  1.1447 |    0      | ***   |
| Milk6_baseline             | Milk6_activeG_forecastOnly |  0.7665 |    0      | ***   |
| Milk6_baseline             | Milk6_activeG_backcastOnly |  0.6899 |    0      | ***   |
| Milk6_activeG              | Milk6_activeG_forecastOnly | -0.3782 |    0      | ***   |
| Milk6_activeG              | Milk6_activeG_backcastOnly | -0.4548 |    0      | ***   |
| Milk6_activeG_forecastOnly | Milk6_activeG_backcastOnly | -0.0766 |    0.0152 | *     |

6 of 6 pairs show statistically significant differences (p < 0.05).

