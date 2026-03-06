# V3AELG Stack Height Sweep Analysis

**Date:** 2026-03-05
**Study:** `v3aelg_stackheight_sweep` (configs in `experiments/configs/wavelet_v3aelg_trendaelg_*.yaml`)
**Datasets Analyzed:** M4-Yearly (complete), Tourism-Yearly (complete), Traffic-96 (in progress)
**Prior Study:** `wavelet_v3aelg_trendaelg_study` (V1, fixed repeats=5, latent_dim=16)

---

## Executive Summary

This study sweeps over TrendAELG + WaveletV3AELG alternating stacks at four stack heights (repeats: 2, 3, 4, 5 = 4, 6, 8, 10 total stacks) across 9 wavelet families and 4 basis-dimension labels, yielding 144 configs per dataset. Three-round successive halving (50% elimination per round) narrows the field from 144 to 36 configs.

**Key Findings:**

1. **Deeper is better, but diminishing returns past 8 stacks.** Repeats=2 (4 stacks) was completely eliminated in Round 1 on both datasets. Repeats=4 and repeats=5 dominate the final round. On M4-Yearly, the best config is repeats=4 (8 stacks); on Tourism-Yearly, the best is repeats=5 (10 stacks). The difference between repeats=4 and repeats=5 is not statistically significant (M4: p=0.86, Tourism: p=0.10).

2. **No single wavelet family dominates.** All 9 families survived to Round 3 on both datasets. In the final round, Coif1 and Coif2 are consistent top performers across datasets, but the performance spread across families is narrow (~0.13 SMAPE on M4, ~0.20 on Tourism).

3. **The prior V1 study (latent_dim=16) outperforms this sweep (latent_dim=32).** V1 achieved SMAPE=13.438 on M4 vs 13.527 here (+0.66%), and SMAPE=20.930 on Tourism vs 21.037 here (+0.51%). This is a statistically significant difference for the top-5 means (p<0.02 on both datasets). The latent_dim=16 to latent_dim=32 change is the likely explanation, suggesting latent_dim=32 is over-parameterized for these relatively short forecast horizons.

4. **Basis labels `eq_fcast` and `lt_bcast` are the strongest.** They have the highest survival rates and produce the best SMAPE scores. On Tourism, these two labels are functionally identical (both resolve to basis_dim=4).

5. **All Round 3 configs converged well** (100% early-stopped, zero divergence, mean loss_ratio ~1.03), indicating the architecture is stable across all tested configurations.

---

## 1. Study Design

### Architecture
- **Stack pattern:** Alternating `[TrendAELG, WaveletV3AELG]` repeated `N` times
- **Block config:** 1 block per stack, shared weights, ReLU activation, no active_g, no sum_losses
- **Optimizer:** Adam, lr=0.001 with cosine annealing (warmup=15 epochs, eta_min=1e-6)
- **Loss:** SMAPELoss

### Search Space (144 configs per dataset)
| Dimension | Values | Count |
|---|---|---|
| Wavelet families | Haar, DB2, DB3, DB4, Coif1, Coif2, Coif3, Symlet2, Symlet3 | 9 |
| Basis labels | eq_fcast, lt_fcast, eq_bcast, lt_bcast | 4 |
| Repeats (stack height) | 2, 3, 4, 5 (= 4, 6, 8, 10 stacks) | 4 |
| Trend thetas dim | 3 (fixed) | 1 |
| Latent dim | 32 (fixed) | 1 |

### Basis Label Resolution
The basis labels resolve to different `basis_dim` values depending on the dataset's `forecast_length` and `backcast_length`:

| Label | M4-Yearly (fcast=6, bcast=30) | Tourism-Yearly (fcast=4, bcast=8) |
|---|---|---|
| eq_fcast | basis_dim=6 | basis_dim=4 |
| lt_fcast | basis_dim=4 | basis_dim=2 |
| eq_bcast | basis_dim=30 | basis_dim=8 |
| lt_bcast | basis_dim=15 | basis_dim=4 |

**Note:** On Tourism-Yearly, `eq_fcast` and `lt_bcast` both resolve to `basis_dim=4`, making them functionally identical. This is confirmed by identical SMAPE values across all runs for paired configs (e.g., `Symlet2_eq_fcast_ttd3_ld32_r5` and `Symlet2_lt_bcast_ttd3_ld32_r5` produce the same predictions). This means Tourism effectively has only 3 distinct basis_dim settings per wavelet/repeat combination, not 4.

### Successive Halving Schedule
| Round | Max Epochs | Runs/Config | Configs | Elimination |
|---|---|---|---|---|
| 1 | 15 | 3 | 144 | Keep top 50% |
| 2 | 25 | 3 | 72 | Keep top 50% |
| 3 | 100 (86 actual M4, 52 Tourism) | 3 | 36 | Final ranking |

---

## 2. Round-by-Round Summary

### M4-Yearly
| Round | Configs | Mean SMAPE Range | Median SMAPE |
|---|---|---|---|
| 1 | 144 | [14.122, 18.331] | 14.944 |
| 2 | 72 | [13.577, 14.216] | 13.761 |
| 3 | 36 | [13.527, 13.982] | 13.607 |

The SMAPE range compresses dramatically from Round 1 (4.2 spread) to Round 3 (0.46 spread), confirming that successive halving effectively removes the worst performers early while leaving a competitive final field.

### Tourism-Yearly
| Round | Configs | Mean SMAPE Range | Median SMAPE |
|---|---|---|---|
| 1 | 144 | [23.835, 83.724] | 28.178 |
| 2 | 72 | [21.444, 81.314] | 21.894 |
| 3 | 36 | [21.037, 21.399] | 21.169 |

Tourism shows a much wider initial range (59.9 spread) due to some basis configurations being poorly suited to the short Tourism horizon. The final round is remarkably tight (0.36 spread).

### Traffic-96 (In Progress)
Only 7 of 144 configs completed in Round 1 (all Haar wavelets with eq_fcast or lt_fcast). Early results show a clear stack height effect: MSE drops from 0.000790 (repeats=2) to 0.000702 (repeats=5).

---

## 3. Stack Height (Repeats) Analysis

### 3.1 Survival Rate by Repeats

| Repeats | Total Stacks | M4 R1 | M4 R2 | M4 R3 | Tourism R1 | Tourism R2 | Tourism R3 |
|---|---|---|---|---|---|---|---|
| 2 | 4 | 36 | **0** (0%) | -- | 36 | **0** (0%) | -- |
| 3 | 6 | 36 | 10 (28%) | 2 (6%) | 36 | 14 (39%) | **0** (0%) |
| 4 | 8 | 36 | 28 (78%) | 11 (31%) | 36 | 29 (81%) | 9 (25%) |
| 5 | 10 | 36 | 34 (94%) | 23 (64%) | 36 | 29 (81%) | 27 (75%) |

**Critical finding: Repeats=2 (4 stacks) is insufficient.** All 36 repeats=2 configs were eliminated in Round 1 on both datasets. On M4, the best repeats=2 validation loss (15.959) was worse than the 72nd-best config cutoff (15.169). The architecture simply needs more than 4 stacks to learn the TrendAELG + WaveletV3AELG decomposition.

**Repeats=5 dominates numerically** in the final round (23/36 on M4, 27/36 on Tourism), but this partly reflects its stronger Round 1 performance giving it more configs to promote.

### 3.2 Performance by Repeats

#### Round 1 (All Configs, Quick Evaluation)

| Repeats | Stacks | M4 Mean SMAPE | M4 Best SMAPE | M4 Mean OWA | Tourism Mean SMAPE | Tourism Best SMAPE |
|---|---|---|---|---|---|---|
| 2 | 4 | 16.346 | 15.432 | 0.991 | 29.573 | 27.914 |
| 3 | 6 | 15.205 | 14.672 | 0.916 | 28.775 | 26.317 |
| 4 | 8 | 14.796 | 14.321 | 0.890 | 28.917 | 25.810 |
| 5 | 10 | 14.602 | 14.122 | 0.878 | 25.490 | 23.835 |

Performance improves monotonically with stack height in Round 1. The improvement from repeats=4 to repeats=5 is smaller than from repeats=3 to repeats=4, suggesting diminishing returns.

#### Round 3 (Survivors Only, Extended Training)

| Repeats | Stacks | M4 Configs | M4 Mean SMAPE | M4 Best SMAPE | Tourism Configs | Tourism Mean SMAPE | Tourism Best SMAPE |
|---|---|---|---|---|---|---|---|
| 3 | 6 | 2 | 13.563 | 13.556 | 0 | -- | -- |
| 4 | 8 | 11 | 13.636 | 13.527 | 9 | 21.220 | 21.091 |
| 5 | 10 | 23 | 13.649 | 13.532 | 27 | 21.159 | 21.037 |

After extended training, the performance gap between repeats=4 and repeats=5 is negligible:
- **M4:** The best repeats=4 config (13.527) actually beats the best repeats=5 config (13.532).
- **Tourism:** The best repeats=5 config (21.037) slightly beats repeats=4 (21.091).

### 3.3 Statistical Tests: Repeats=4 vs Repeats=5

| Dataset | Repeats=4 Mean | Repeats=5 Mean | Mann-Whitney U | p-value | Interpretation |
|---|---|---|---|---|---|
| M4-Yearly | 13.636 (n=33) | 13.649 (n=69) | 1164.0 | 0.858 | No significant difference |
| Tourism-Yearly | 21.220 (n=27) | 21.159 (n=81) | 1323.0 | 0.104 | Marginal Tourism advantage for r=5 |

**Conclusion:** Repeats=4 (8 stacks) and repeats=5 (10 stacks) perform equivalently after extended training. The 8-stack architecture achieves comparable performance with ~20% fewer parameters (834K vs 1.04M on M4), making it more parameter-efficient.

### 3.4 Parameter Count by Repeats

| Repeats | Stacks | M4 Param Range | Tourism Param Range |
|---|---|---|---|
| 3 | 6 | 625,545 -- 662,409 | -- (eliminated) |
| 4 | 8 | 834,060 -- 883,212 | 792,076 -- 800,268 |
| 5 | 10 | 1,042,575 -- 1,104,015 | 979,855 -- 1,000,335 |

---

## 4. Wavelet Family Ranking

### 4.1 Round 3 Survival Counts

| Wavelet | M4 R3 Configs | Tourism R3 Configs | Combined |
|---|---|---|---|
| Coif1 | 3 | 6 | 9 |
| Coif2 | 6 | 1 | 7 |
| Symlet3 | 4 | 6 | 10 |
| DB4 | 5 | 4 | 9 |
| Symlet2 | 4 | 4 | 8 |
| DB2 | 4 | 4 | 8 |
| Haar | 4 | 4 | 8 |
| DB3 | 3 | 4 | 7 |
| Coif3 | 3 | 3 | 6 |

### 4.2 Round 3 SMAPE Rankings

#### M4-Yearly (Round 3)
| Rank | Wavelet | Mean SMAPE | Std | Best SMAPE | Configs |
|---|---|---|---|---|---|
| 1 | Coif1 | 13.587 | 0.068 | 13.461 | 3 |
| 2 | Coif2 | 13.587 | 0.091 | 13.449 | 6 |
| 3 | DB2 | 13.598 | 0.101 | 13.459 | 4 |
| 4 | DB3 | 13.608 | 0.089 | 13.402 | 3 |
| 5 | Symlet2 | 13.625 | 0.187 | **13.356** | 4 |
| 6 | DB4 | 13.680 | 0.263 | 13.441 | 5 |
| 7 | Haar | 13.680 | 0.172 | 13.464 | 4 |
| 8 | Coif3 | 13.688 | 0.169 | 13.496 | 3 |
| 9 | Symlet3 | 13.716 | 0.358 | 13.424 | 4 |

#### Tourism-Yearly (Round 3)
| Rank | Wavelet | Mean SMAPE | Std | Best SMAPE | Configs |
|---|---|---|---|---|---|
| 1 | Symlet2 | 21.087 | 0.213 | **20.790** | 4 |
| 2 | Coif2 | 21.090 | 0.159 | 20.928 | 1 |
| 3 | Coif1 | 21.100 | 0.148 | 20.896 | 6 |
| 4 | DB3 | 21.131 | 0.255 | 20.773 | 4 |
| 5 | Coif3 | 21.136 | 0.225 | 20.902 | 3 |
| 6 | Haar | 21.181 | 0.178 | 20.985 | 4 |
| 7 | DB4 | 21.208 | 0.170 | 20.961 | 4 |
| 8 | Symlet3 | 21.263 | 0.210 | 20.909 | 6 |
| 9 | DB2 | 21.289 | 0.196 | 21.019 | 4 |

### 4.3 Cross-Dataset Wavelet Consistency

| Wavelet | M4 Rank | Tourism Rank | Avg Rank |
|---|---|---|---|
| Coif1 | 1 | 3 | 2.0 |
| Coif2 | 2 | 2 | 2.0 |
| DB3 | 4 | 4 | 4.0 |
| DB2 | 3 | 9 | 6.0 |
| Symlet2 | 5 | 1 | 3.0 |
| DB4 | 6 | 7 | 6.5 |
| Haar | 7 | 6 | 6.5 |
| Coif3 | 8 | 5 | 6.5 |
| Symlet3 | 9 | 8 | 8.5 |

**Spearman rank correlation:** rho=0.433, p=0.244 (not significant, but moderate positive trend).

**Consistently strong:** Coif1 (avg rank 2.0) and Coif2 (avg rank 2.0) are the most reliable across datasets. Symlet2 is the single-best on Tourism but middle-of-pack on M4.

**Consistently weak:** Symlet3 ranks 9th on M4 and 8th on Tourism (avg 8.5).

**Important context:** The performance spread across wavelet families is small. On M4, the gap between the best (Coif1, 13.587) and worst (Symlet3, 13.716) wavelet family mean is only 0.129 SMAPE (0.95%). On Tourism, the gap is 0.202 SMAPE (0.96%). The choice of wavelet family matters less than the choice of stack height or basis dimension.

---

## 5. Basis Label Analysis

### 5.1 Survival Rates

| Label | M4 Basis Dim | M4 R1->R2 | M4 R2->R3 | Tourism Basis Dim | Tourism R1->R2 | Tourism R2->R3 |
|---|---|---|---|---|---|---|
| eq_fcast | 6 | 64% | 70% | 4 | 61% | 45% |
| eq_bcast | 30 | 53% | 63% | 8 | 69% | 56% |
| lt_bcast | 15 | 47% | 47% | 4 | 58% | 48% |
| lt_fcast | 4 | 36% | **0%** | 2 | 11% | 50% |

**`lt_fcast` is strongly disfavored on M4** (no Round 3 survivors), consistent with the very small basis_dim=4 being too restrictive for M4-Yearly's forecast_length=6. On Tourism, lt_fcast barely survives (only 2 configs in R3) with its basis_dim=2.

**`eq_fcast` is the strongest on M4** with 16/36 final survivors (44%). It offers a natural match: basis_dim equals forecast_length, giving the wavelet exactly the right number of basis functions to represent the output.

### 5.2 Round 3 Performance

#### M4-Yearly
| Label | Basis Dim | Mean SMAPE | Best SMAPE | Configs |
|---|---|---|---|---|
| eq_fcast | 6 | 13.615 | **13.356** | 16 |
| lt_bcast | 15 | 13.615 | 13.402 | 8 |
| eq_bcast | 30 | 13.690 | 13.446 | 12 |

`eq_fcast` and `lt_bcast` perform identically on average. `eq_bcast` (basis_dim=30) is slightly worse, suggesting that an overly large basis dimension can hurt by introducing too many coefficients to learn.

#### Tourism-Yearly
| Label | Basis Dim | Mean SMAPE | Best SMAPE | Configs |
|---|---|---|---|---|
| lt_fcast | 2 | 21.084 | 20.909 | 2 |
| eq_fcast / lt_bcast | 4 | 21.172 | **20.790** | 20 (tied) |
| eq_bcast | 8 | 21.190 | 20.773 | 14 |

On Tourism, `lt_fcast` (basis_dim=2) has the best mean but with only 2 configs, its reliability is uncertain. The eq_fcast/lt_bcast pair (basis_dim=4) produces the best individual run (20.790) and dominates by volume.

### 5.3 Basis Label Interpretation

The results suggest a **moderate basis dimension is optimal**: large enough to capture the signal but small enough to regularize. For M4-Yearly (forecast=6), `eq_fcast` (basis_dim=6) works best. For Tourism-Yearly (forecast=4), basis_dim=4 is the sweet spot. The `eq_bcast` label (basis_dim=30 on M4, basis_dim=8 on Tourism) is slightly over-parameterized.

---

## 6. Final Round Top-10 Analysis

### M4-Yearly Top 10 (Round 3)

| Rank | Config | SMAPE | Std | OWA | Params | Repeats | Stacks | Delta |
|---|---|---|---|---|---|---|---|---|
| 1 | DB2_eq_fcast_ttd3_ld32_r4 | 13.527 | 0.073 | 0.803 | 834,060 | 4 | 8 | -- |
| 2 | Coif2_eq_bcast_ttd3_ld32_r5 | 13.532 | 0.038 | 0.803 | 1,104,015 | 5 | 10 | +0.03% |
| 3 | DB3_lt_bcast_ttd3_ld32_r5 | 13.540 | 0.120 | 0.803 | 1,065,615 | 5 | 10 | +0.10% |
| 4 | Coif1_eq_bcast_ttd3_ld32_r5 | 13.540 | 0.096 | 0.805 | 1,104,015 | 5 | 10 | +0.10% |
| 5 | Symlet3_eq_fcast_ttd3_ld32_r5 | 13.543 | 0.039 | 0.804 | 1,042,575 | 5 | 10 | +0.12% |
| 6 | DB4_eq_fcast_ttd3_ld32_r5 | 13.549 | 0.071 | 0.805 | 1,042,575 | 5 | 10 | +0.16% |
| 7 | Symlet2_lt_bcast_ttd3_ld32_r5 | 13.553 | 0.112 | 0.805 | 1,065,615 | 5 | 10 | +0.20% |
| 8 | Coif2_eq_fcast_ttd3_ld32_r3 | 13.556 | 0.090 | 0.806 | 625,545 | 3 | 6 | +0.21% |
| 9 | DB4_lt_bcast_ttd3_ld32_r4 | 13.558 | 0.124 | 0.805 | 852,492 | 4 | 8 | +0.23% |
| 10 | Coif2_eq_fcast_ttd3_ld32_r5 | 13.566 | 0.123 | 0.806 | 1,042,575 | 5 | 10 | +0.29% |

The M4 winner is a repeats=4 config (DB2_eq_fcast), which is notable since repeats=5 dominates the field numerically. The spread from rank 1 to rank 10 is only 0.039 SMAPE (0.29%), indicating a very competitive final field. All top-10 configs achieve OWA in the 0.803--0.806 range, solidly beating the Naive2 baseline.

Rank 8 deserves attention: **Coif2_eq_fcast_ttd3_ld32_r3** achieves SMAPE=13.556 with only 625,545 parameters (6 stacks), making it the most parameter-efficient top-10 config -- 40% fewer parameters than the repeats=5 configs.

### Tourism-Yearly Top 10 (Round 3)

| Rank | Config | SMAPE | Std | Params | Repeats | Stacks | Delta |
|---|---|---|---|---|---|---|---|
| 1 | Symlet2_eq_fcast_ttd3_ld32_r5 | 21.037 | 0.308 | 990,095 | 5 | 10 | -- |
| 2 | Symlet2_lt_bcast_ttd3_ld32_r5 | 21.037 | 0.308 | 990,095 | 5 | 10 | +0.00% |
| 3 | DB3_lt_bcast_ttd3_ld32_r5 | 21.042 | 0.115 | 990,095 | 5 | 10 | +0.02% |
| 4 | DB3_eq_fcast_ttd3_ld32_r5 | 21.042 | 0.115 | 990,095 | 5 | 10 | +0.02% |
| 5 | Coif1_eq_fcast_ttd3_ld32_r5 | 21.043 | 0.088 | 990,095 | 5 | 10 | +0.03% |
| 6 | Coif1_lt_bcast_ttd3_ld32_r5 | 21.043 | 0.088 | 990,095 | 5 | 10 | +0.03% |
| 7 | Coif3_eq_fcast_ttd3_ld32_r5 | 21.044 | 0.222 | 990,095 | 5 | 10 | +0.03% |
| 8 | Coif3_lt_bcast_ttd3_ld32_r5 | 21.044 | 0.222 | 990,095 | 5 | 10 | +0.03% |
| 9 | Symlet3_lt_fcast_ttd3_ld32_r5 | 21.082 | 0.174 | 979,855 | 5 | 10 | +0.21% |
| 10 | Symlet2_lt_fcast_ttd3_ld32_r5 | 21.086 | 0.145 | 979,855 | 5 | 10 | +0.23% |

**All Tourism top-10 configs use repeats=5 (10 stacks).** Ranks 1-2, 3-4, 5-6, and 7-8 are identical pairs (eq_fcast = lt_bcast, both resolving to basis_dim=4 on Tourism). The effective top 5 unique configs are: Symlet2, DB3, Coif1, Coif3, and Symlet3 (lt_fcast). The spread from rank 1 to rank 10 is only 0.049 SMAPE (0.23%).

### Convergence Quality (Round 3)

| Metric | M4-Yearly | Tourism-Yearly |
|---|---|---|
| Stopping reason | 100% EARLY_STOPPED | 100% EARLY_STOPPED |
| Best epoch range | [12, 75] | [18, 41] |
| Mean best epoch | 38.2 | 30.2 |
| Mean loss ratio | 1.040 | 1.017 |
| Diverged configs | 0 | 0 |

All configs converged cleanly. The slightly higher loss ratio on M4 (1.040 vs 1.017) suggests mild overfitting on M4, but early stopping captured the optimum.

---

## 7. Cross-Dataset Consistency

### Top-10 Overlap

| Overlap Level | Count | Details |
|---|---|---|
| Top-10 overlap | 2 / 10 | DB3_lt_bcast_ttd3_ld32_r5, Symlet2_lt_bcast_ttd3_ld32_r5 |
| Top-20 overlap | 9 / 20 | Strong convergence at moderate depth |
| Total R3 overlap | 24 / 36 | 67% of final configs appear in both datasets |

The two configs that appear in both top-10 lists:

| Config | M4 Rank | M4 SMAPE | Tourism Rank | Tourism SMAPE |
|---|---|---|---|---|
| DB3_lt_bcast_ttd3_ld32_r5 | 3 | 13.540 | 3 | 21.042 |
| Symlet2_lt_bcast_ttd3_ld32_r5 | 7 | 13.553 | 1 | 21.037 |

**DB3_lt_bcast_ttd3_ld32_r5** is the most consistent cross-dataset performer, ranking 3rd on both datasets. It uses DB3 wavelets with basis_dim matched to the backcast half-length, giving it a general-purpose decomposition that works across different forecast horizons.

### Repeats Agreement

Both datasets strongly agree that repeats >= 4 is necessary. Repeats=2 was universally eliminated. Repeats=5 dominates both final rounds (64% of M4 R3, 75% of Tourism R3).

### Basis Label Agreement

Both datasets favor `eq_fcast` and `lt_bcast`. Both datasets penalize `lt_fcast` (too few basis functions). The `eq_bcast` label performs acceptably but never tops the rankings.

---

## 8. Comparison to Prior V1 Study

### V1 Study Design

| Parameter | V1 Study | V2 Stack Height Sweep |
|---|---|---|
| Stack height | Fixed: 10 stacks (repeats=5) | Sweep: 4, 6, 8, 10 stacks |
| Latent dim | 16 | 32 |
| Trend thetas dim | 3, 5 | 3 only |
| Wavelet families | 14 (incl. DB10, DB20, Symlet10, Symlet20, Coif10) | 9 (short-support only) |
| Configs searched | 112 | 144 |
| R3 survivors | 50 | 36 |
| R1/R2/R3 epochs | 10/15/50 | 15/25/86-100 |

### Head-to-Head Comparison

| Dataset | V1 Best Config | V1 SMAPE | V2 Best Config | V2 SMAPE | Delta |
|---|---|---|---|---|---|
| M4-Yearly | Symlet20_eq_fcast_ttd3_ld16 | **13.438** | DB2_eq_fcast_ttd3_ld32_r4 | 13.527 | +0.089 (+0.66%) |
| Tourism-Yearly | Coif1_eq_fcast_ttd3_ld16 | **20.930** | Symlet2_eq_fcast_ttd3_ld32_r5 | 21.037 | +0.107 (+0.51%) |

### Statistical Significance

| Test | M4-Yearly | Tourism-Yearly |
|---|---|---|
| V1 best vs V2 best (single config, MWU) | U=2.0, p=0.400 | U=4.0, p=1.000 |
| V1 top-5 means vs V2 top-5 means (MWU) | U=0.0, **p=0.008** | U=0.0, **p=0.011** |

**Individual best configs:** Not significantly different (p>0.05), but with only 3 runs per config, power is limited.

**Top-5 aggregate comparison:** V1 is significantly better than V2 on both datasets (p<0.02). This is not a single lucky config -- the entire V1 top-5 is better than the entire V2 top-5.

### Why Does V1 Outperform V2?

The primary difference is **latent_dim=16 (V1) vs latent_dim=32 (V2)**. For these short-horizon forecasting tasks (M4-Yearly: forecast=6, Tourism-Yearly: forecast=4), a 32-dimensional latent space is likely over-parameterized:

1. **Excess capacity:** With forecast_length=6, the wavelet basis only needs ~6 coefficients. A latent_dim=32 bottleneck provides far more capacity than the task requires, potentially allowing the model to memorize noise rather than learning generalizable patterns.

2. **Gate efficiency:** The AELG learned gates must learn to suppress more dimensions with latent_dim=32, requiring more training signal to converge to the effective dimensionality.

3. **V1 included long-support wavelets:** Symlet20 (the V1 M4 winner) is not in the V2 search space. However, this is unlikely the main factor, since V1's short-support wavelets (Coif1, DB2, Haar) at latent_dim=16 also outperform V2's best.

4. **V1 also searched trend_thetas_dim=5:** Some V1 winners use ttd=5 (Symlet3_lt_bcast_ttd5_ld16 is #2 on M4). V2 fixed ttd=3.

### V1 vs V2: Matched Comparison (Same Wavelet, Same Repeats)

To isolate the latent_dim effect, we can compare configs that differ only in latent_dim. The V1 study used n_stacks=10 (repeats=5), so comparing against V2 repeats=5 configs with the same wavelet, basis_label, and ttd=3. Two V1 configs (DB3_lt_bcast, Symlet3_lt_bcast) had a catastrophic run (SMAPE=200) and are excluded.

| Config Pattern | V1 (ld=16) R3 SMAPE | V2 (ld=32) R3 SMAPE | Delta |
|---|---|---|---|
| M4: Coif2_eq_fcast_ttd3 | 13.524 | 13.566 | +0.042 |
| M4: Haar_eq_fcast_ttd3 | 13.482 | 13.692 | +0.210 |
| M4: Symlet2_eq_fcast_ttd3 | 13.498 | 13.596 | +0.099 |
| M4: DB2_eq_bcast_ttd3 | 13.475 | 13.663 | +0.188 |
| M4: Coif1_lt_bcast_ttd3 | 13.561 | 13.617 | +0.055 |
| Tourism: Coif1_eq_fcast_ttd3 | 20.930 | 21.043 | +0.113 |
| Tourism: Symlet2_eq_fcast_ttd3 | 21.019 | 21.037 | +0.018 |
| Tourism: DB4_eq_fcast_ttd3 | 20.947 | 21.211 | +0.264 |
| Tourism: DB3_eq_fcast_ttd3 | 21.171 | 21.042 | -0.129 |
| Tourism: Coif1_eq_bcast_ttd3 | 21.107 | 21.129 | +0.021 |

Of the 10 matched comparisons, V1 (ld=16) outperforms V2 (ld=32) in 9 out of 10 cases, with a median advantage of +0.077 SMAPE. The single exception (Tourism: DB3_eq_fcast) shows V2 winning by 0.129. The latent_dim=16 to latent_dim=32 change is the dominant factor in V2's relative underperformance.

---

## 9. Traffic-96 Early Results (In Progress)

Only 7 of 144 configs have completed Round 1 (all Haar wavelets). Preliminary findings:

| Repeats | Stacks | Mean MSE | Mean MAE | Mean SMAPE | Params |
|---|---|---|---|---|---|
| 2 | 4 | 0.000790 | 0.01161 | 18.628 | ~723K |
| 3 | 6 | 0.000764 | 0.01127 | 18.109 | ~1.09M |
| 4 | 8 | 0.000718 | 0.01089 | 17.616 | ~1.45M |
| 5 | 10 | 0.000702 | 0.01086 | 17.662 | ~1.81M |

The same pattern holds: deeper stacks improve MSE monotonically, though the MSE gap between repeats=4 and repeats=5 is small (2.3%). The `eq_fcast` label consistently outperforms `lt_fcast` on Traffic.

---

## 10. Recommendations

### 10.1 Current Best Configurations

**For M4-Yearly:**
The V1 study winner (`Symlet20_eq_fcast_ttd3_ld16`, SMAPE=13.438, OWA=0.795) remains the best known TrendAELG+WaveletV3AELG configuration for M4-Yearly. From this sweep, `DB2_eq_fcast_ttd3_ld32_r4` (SMAPE=13.527) is the best at latent_dim=32 but does not improve upon V1.

**For Tourism-Yearly:**
The V1 study winner (`Coif1_eq_fcast_ttd3_ld16`, SMAPE=20.930) remains the best. From this sweep, `Symlet2_eq_fcast_ttd3_ld32_r5` (SMAPE=21.037) is the best at latent_dim=32.

**Cross-dataset robust choice:**
`DB3_lt_bcast_ttd3_ld32_r5` ranks 3rd on both M4 and Tourism, making it the most consistent across datasets from this sweep.

### 10.2 What to Test Next

**Priority 1: latent_dim=16 + stack height sweep**

The clear takeaway from this study is that latent_dim=32 underperforms latent_dim=16. The stack height sweep should be re-run with latent_dim=16 to combine the best of both studies. This would test whether the optimal stack height shifts at a smaller latent dimension.

```yaml
# Proposed: v3aelg_stackheight_ld16_sweep
study_name: v3aelg_stackheight_ld16_sweep_m4
dataset: m4
period: Yearly

search_space:
  wavelets:
    - Coif1WaveletV3AELG
    - Coif2WaveletV3AELG
    - DB2WaveletV3AELG
    - DB3WaveletV3AELG
    - Symlet2WaveletV3AELG
    - Symlet20WaveletV3AELG  # V1 winner
  basis_labels: [eq_fcast, lt_bcast]  # Clear best two
  trend_thetas_dims: [3, 5]  # Re-include ttd=5
  latent_dim: 16
  repeats: [3, 4, 5]  # Drop repeats=2 (proven insufficient)
```

This reduces the search space to 6 x 2 x 2 x 3 = 72 configs (vs 144), allowing faster iteration.

**Priority 2: Focused repeats=3/4 study at latent_dim=16**

Since repeats=4 can match repeats=5 with fewer parameters, a targeted study could determine whether the parameter savings hold at latent_dim=16:

```yaml
search_space:
  wavelets: [Coif1WaveletV3AELG, Coif2WaveletV3AELG, DB3WaveletV3AELG, Symlet20WaveletV3AELG]
  basis_labels: [eq_fcast, lt_bcast]
  trend_thetas_dims: [3]
  latent_dim: 16
  repeats: [3, 4, 5]
```

With 4 x 2 x 1 x 3 = 24 configs, this could use more runs per config (5 instead of 3) for higher statistical power.

**Priority 3: latent_dim grid search**

Test latent_dim in {8, 16, 24, 32} for the top-3 configs to map the latent dimension vs performance curve:

```yaml
search_space:
  wavelets: [Coif1WaveletV3AELG, DB3WaveletV3AELG, Symlet20WaveletV3AELG]
  basis_labels: [eq_fcast]
  trend_thetas_dims: [3]
  latent_dim: [8, 16, 24, 32]  # Grid search over latent_dim
  repeats: [5]
```

### 10.3 Open Questions

1. **Is latent_dim=16 universally better, or only for short horizons?** Traffic-96 (forecast=96) and Weather-96 have much longer forecast horizons. The optimal latent_dim may be larger there. The ongoing Traffic sweep (latent_dim=32) will provide partial answers.

2. **Why does Symlet20 win on M4 but not Tourism?** Symlet20 was not included in this sweep, but it was the V1 M4 winner. Its long-support wavelet basis may be particularly well-suited to M4-Yearly's longer backcast window (30 vs 8 for Tourism).

3. **Would repeats=4 with latent_dim=16 beat repeats=5?** This sweep showed repeats=4 and repeats=5 are tied at latent_dim=32. At latent_dim=16 (which reduces per-stack parameters), the balance might shift further toward repeats=4 or even repeats=3.

4. **Is trend_thetas_dim=5 better than 3?** V1 found several ttd=5 configs in its top-10 (M4 #2, #4, #5, #7, #9). This sweep fixed ttd=3 and cannot answer this question. The Priority 1 experiment above includes both.

5. **How do the best wavelet-AELG configs compare to non-wavelet baselines?** A direct comparison against NBEATS-G (30xGeneric) and the best GenericAE/GenericAELG configs would establish whether wavelet basis expansion adds value over learned generic projections.

---

## Appendix: Data Quality Notes

- **M4-Yearly:** 756 rows, 3 complete rounds, all configs have exactly 3 runs. No missing values. OWA available.
- **Tourism-Yearly:** 756 rows, 3 complete rounds. OWA not available (Tourism lacks Naive2 baselines). The eq_fcast/lt_bcast degeneracy (both = basis_dim=4) inflates apparent config diversity -- effectively 108 unique configs, not 144.
- **Traffic-96:** 20 rows, Round 1 only (7 of 144 configs, all Haar). Too early for meaningful analysis. OWA not available.
- **Weather-96:** Not yet started.
- **V1 Study:** Complete for M4 and Tourism with 50 final-round configs each. Uses latent_dim=16, trend_thetas_dim in {3,5}, and includes long-support wavelets not in V2.
