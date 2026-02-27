# Generic AE Pure-Stack Study — Results Analysis

## Abstract

This study evaluates GenericAE and BottleneckGenericAE as pure-stack architectures on M4-Yearly using successive-halving search. GenericAE replaces the standard FC backbone with an encoder-decoder structure, producing a rank-constrained latent representation before projecting to backcast and forecast outputs.

**Key Takeaways:**

- **Best configuration:** `GenericAE_ld16_td5_agF` achieves median OWA = **0.7988** with 1,664,160 parameters (93% fewer than NBEATS-G).
- **vs NBEATS-I+G (0.8057):** beats (Δ = -0.0069).
- **Search scope:** 40 initial configs → 11 survivors across 3 rounds.
- **Block types tested:** ['BottleneckGenericAE', 'GenericAE', 'I+G']
- **Total compute:** 33.9 minutes across 213 runs.

- **CSV:** `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/generic_ae_pure_stack_results.csv`
- **Rows:** 213 (41 unique configs, 3 rounds)

### Published Baselines (M4-Yearly, 30-stack)

| Config     |    OWA |   sMAPE | Params     |
|:-----------|-------:|--------:|:-----------|
| NBEATS-I+G | 0.8057 |   13.53 | 35,900,000 |
| NBEATS-I   | 0.8132 |   13.67 | 12,900,000 |
| NBEATS-G   | 0.8198 |   13.7  | 24,700,000 |


## 1. Successive Halving Funnel

|   Round |   Configs |   Runs |   Epochs |   Best Med OWA |
|--------:|----------:|-------:|---------:|---------------:|
|       1 |        40 |    120 |        8 |         1.1456 |
|       2 |        20 |     60 |       15 |         0.8971 |
|       3 |        11 |     33 |       50 |         0.7988 |

The successive halving procedure narrowed from 40 to 11 configurations across 3 rounds. Each round increased the training budget while eliminating weaker candidates.


## 2.1 Round 1 Leaderboard

40 configs × 3 runs, 8 epochs each

|   Rank | Config                            | Block               |    OWA |      ± |   sMAPE | Params    |
|-------:|:----------------------------------|:--------------------|-------:|-------:|--------:|:----------|
|      1 | GenericAE_ld16_td5_ag0            | GenericAE           | 1.1456 | 0.118  |   18.25 | 1,664,160 |
|      2 | BottleneckGenericAE_ld4_td3_agF   | BottleneckGenericAE | 1.2083 | 0.071  |   18.95 | 1,434,750 |
|      3 | GenericAE_ld8_td5_ag0             | GenericAE           | 1.2188 | 0.153  |   18.93 | 1,623,120 |
|      4 | BottleneckGenericAE_ld2_td10_agF  | BottleneckGenericAE | 1.2251 | 0.1283 |   19.25 | 1,462,920 |
|      5 | GenericAE_ld16_td5_agF            | GenericAE           | 1.2273 | 0.2524 |   19.07 | 1,664,160 |
|      6 | GenericAE_ld2_td5_ag0             | GenericAE           | 1.2385 | 0.2267 |   18.96 | 1,592,340 |
|      7 | BottleneckGenericAE_ld2_td5_agF   | BottleneckGenericAE | 1.2614 | 0.1268 |   19.57 | 1,435,470 |
|      8 | BottleneckGenericAE_ld8_td8_agF   | BottleneckGenericAE | 1.2671 | 0.055  |   19.83 | 1,482,720 |
|      9 | BottleneckGenericAE_ld2_td5_ag0   | BottleneckGenericAE | 1.2744 | 0.0795 |   20.02 | 1,435,470 |
|     10 | GenericAE_ld8_td5_agF             | GenericAE           | 1.2814 | 0.2175 |   19.67 | 1,623,120 |
|     11 | BottleneckGenericAE_ld16_td5_agF  | BottleneckGenericAE | 1.2905 | 0.0021 |   20.02 | 1,507,290 |
|     12 | BottleneckGenericAE_ld16_td10_agF | BottleneckGenericAE | 1.3031 | 0.0399 |   20.35 | 1,534,740 |
|     13 | GenericAE_ld2_td5_agF             | GenericAE           | 1.3036 | 0.1459 |   20.39 | 1,592,340 |
|     14 | BottleneckGenericAE_ld4_td5_agF   | BottleneckGenericAE | 1.3049 | 0.1094 |   20.07 | 1,445,730 |
|     15 | BottleneckGenericAE_ld8_td5_agF   | BottleneckGenericAE | 1.3061 | 0.0723 |   20.54 | 1,466,250 |
|     16 | GenericAE_ld4_td5_agF             | GenericAE           | 1.3086 | 0.0544 |   20.37 | 1,602,600 |
|     17 | BottleneckGenericAE_ld2_td10_ag0  | BottleneckGenericAE | 1.3171 | 0.1882 |   20.81 | 1,462,920 |
|     18 | BottleneckGenericAE_ld16_td3_agF  | BottleneckGenericAE | 1.3219 | 0.2151 |   20.51 | 1,496,310 |
|     19 | BottleneckGenericAE_ld16_td8_agF  | BottleneckGenericAE | 1.3377 | 0.1285 |   20.63 | 1,523,760 |
|     20 | BottleneckGenericAE_ld2_td3_agF   | BottleneckGenericAE | 1.3516 | 0.1057 |   20.57 | 1,424,490 |
|     21 | BottleneckGenericAE_ld2_td8_agF   | BottleneckGenericAE | 1.3637 | 0.1246 |   20.59 | 1,451,940 |
|     22 | BottleneckGenericAE_ld8_td3_agF   | BottleneckGenericAE | 1.3714 | 0.1512 |   20.73 | 1,455,270 |
|     23 | BottleneckGenericAE_ld4_td8_agF   | BottleneckGenericAE | 1.3723 | 0.1407 |   21.47 | 1,462,200 |
|     24 | BottleneckGenericAE_ld4_td10_agF  | BottleneckGenericAE | 1.4024 | 0.2046 |   21.34 | 1,473,180 |
|     25 | BottleneckGenericAE_ld8_td10_agF  | BottleneckGenericAE | 1.4229 | 0.1394 |   21.35 | 1,493,700 |
|     26 | BottleneckGenericAE_ld16_td10_ag0 | BottleneckGenericAE | 1.4253 | 1.6822 |   21.64 | 1,534,740 |
|     27 | BottleneckGenericAE_ld2_td8_ag0   | BottleneckGenericAE | 1.432  | 1.6963 |   21.85 | 1,451,940 |
|     28 | BottleneckGenericAE_ld4_td3_ag0   | BottleneckGenericAE | 1.4327 | 0.2889 |   21.73 | 1,434,750 |
|     29 | BottleneckGenericAE_ld8_td10_ag0  | BottleneckGenericAE | 1.4363 | 1.4446 |   22.08 | 1,493,700 |
|     30 | BottleneckGenericAE_ld8_td8_ag0   | BottleneckGenericAE | 1.4699 | 2.7749 |   22.15 | 1,482,720 |
|     31 | BottleneckGenericAE_ld16_td8_ag0  | BottleneckGenericAE | 1.4809 | 2.0208 |   22.71 | 1,523,760 |
|     32 | BottleneckGenericAE_ld16_td5_ag0  | BottleneckGenericAE | 1.5126 | 1.9912 |   22.77 | 1,507,290 |
|     33 | BottleneckGenericAE_ld4_td10_ag0  | BottleneckGenericAE | 1.523  | 1.5044 |   23.01 | 1,473,180 |
|     34 | GenericAE_ld4_td5_ag0             | GenericAE           | 1.548  | 1.4136 |   22.7  | 1,602,600 |
|     35 | BottleneckGenericAE_ld2_td3_ag0   | BottleneckGenericAE | 1.65   | 0.4947 |   25.69 | 1,424,490 |
|     36 | BottleneckGenericAE_ld4_td5_ag0   | BottleneckGenericAE | 1.6552 | 1.8477 |   24.51 | 1,445,730 |
|     37 | BottleneckGenericAE_ld4_td8_ag0   | BottleneckGenericAE | 3.0219 | 1.3755 |   50.27 | 1,462,200 |
|     38 | BottleneckGenericAE_ld16_td3_ag0  | BottleneckGenericAE | 3.0461 | 1.7427 |   50.18 | 1,496,310 |
|     39 | BottleneckGenericAE_ld8_td5_ag0   | BottleneckGenericAE | 3.3275 | 2.0435 |   52.38 | 1,466,250 |
|     40 | BottleneckGenericAE_ld8_td3_ag0   | BottleneckGenericAE | 3.5811 | 2.1622 |   49.27 | 1,455,270 |

The top configuration achieves median OWA = 1.1456 with 1,664,160 parameters.


## 2.2 Round 2 Leaderboard

20 configs × 3 runs, 15 epochs each

|   Rank | Config                           | Block               |    OWA |      ± |   sMAPE | Params    |
|-------:|:---------------------------------|:--------------------|-------:|-------:|--------:|:----------|
|      1 | GenericAE_ld16_td5_agF           | GenericAE           | 0.8971 | 0.022  |   14.87 | 1,664,160 |
|      2 | GenericAE_ld2_td5_agF            | GenericAE           | 0.9037 | 0.0891 |   15.07 | 1,592,340 |
|      3 | GenericAE_ld2_td5_ag0            | GenericAE           | 0.9081 | 0.1    |   15.07 | 1,592,340 |
|      4 | GenericAE_ld8_td5_agF            | GenericAE           | 0.9182 | 0.128  |   15.14 | 1,623,120 |
|      5 | BottleneckGenericAE_ld8_td10_ag0 | BottleneckGenericAE | 0.9255 | 0.0833 |   15.16 | 1,493,700 |
|      6 | BottleneckGenericAE_ld4_td10_agF | BottleneckGenericAE | 0.9336 | 0.1078 |   15.41 | 1,473,180 |
|      7 | BottleneckGenericAE_ld2_td8_agF  | BottleneckGenericAE | 0.9347 | 0.2199 |   15.38 | 1,451,940 |
|      8 | BottleneckGenericAE_ld8_td10_agF | BottleneckGenericAE | 0.9359 | 0.0792 |   15.29 | 1,493,700 |
|      9 | GenericAE_ld8_td5_ag0            | GenericAE           | 0.9463 | 0.0519 |   15.57 | 1,623,120 |
|     10 | BottleneckGenericAE_ld4_td5_agF  | BottleneckGenericAE | 0.9591 | 0.0905 |   15.62 | 1,445,730 |
|     11 | GenericAE_ld16_td5_ag0           | GenericAE           | 0.9593 | 0.164  |   15.63 | 1,664,160 |
|     12 | BottleneckGenericAE_ld2_td10_agF | BottleneckGenericAE | 0.9599 | 0.1041 |   15.8  | 1,462,920 |
|     13 | BottleneckGenericAE_ld2_td5_ag0  | BottleneckGenericAE | 0.9641 | 0.0528 |   15.67 | 1,435,470 |
|     14 | BottleneckGenericAE_ld2_td5_agF  | BottleneckGenericAE | 0.968  | 0.3122 |   15.79 | 1,435,470 |
|     15 | BottleneckGenericAE_ld4_td3_agF  | BottleneckGenericAE | 0.98   | 0.0263 |   15.8  | 1,434,750 |
|     16 | GenericAE_ld4_td5_agF            | GenericAE           | 0.9806 | 0.0883 |   15.99 | 1,602,600 |
|     17 | BottleneckGenericAE_ld4_td5_ag0  | BottleneckGenericAE | 1.0068 | 0.514  |   16.31 | 1,445,730 |
|     18 | BottleneckGenericAE_ld16_td5_ag0 | BottleneckGenericAE | 1.0231 | 0.3444 |   16.46 | 1,507,290 |
|     19 | GenericAE_ld4_td5_ag0            | GenericAE           | 1.0461 | 0.0821 |   16.64 | 1,602,600 |
|     20 | BottleneckGenericAE_ld2_td8_ag0  | BottleneckGenericAE | 1.0725 | 1.9226 |   17.16 | 1,451,940 |

The top configuration achieves median OWA = 0.8971 with 1,664,160 parameters.


## 2.3 Round 3 Leaderboard

11 configs × 3 runs, 50 epochs each

|   Rank | Config                           | Block               |    OWA |      ± |   sMAPE | Params     |
|-------:|:---------------------------------|:--------------------|-------:|-------:|--------:|:-----------|
|      1 | GenericAE_ld16_td5_agF           | GenericAE           | 0.7988 | 0.0073 |   13.49 | 1,664,160  |
|      2 | NBEATS-I+G_10stack               | I+G                 | 0.8023 | 0.057  |   13.49 | 19,506,435 |
|      3 | GenericAE_ld16_td5_ag0           | GenericAE           | 0.8074 | 0.0084 |   13.66 | 1,664,160  |
|      4 | GenericAE_ld8_td5_agF            | GenericAE           | 0.8096 | 0.0275 |   13.64 | 1,623,120  |
|      5 | BottleneckGenericAE_ld8_td10_ag0 | BottleneckGenericAE | 0.8096 | 0.0101 |   13.67 | 1,493,700  |
|      6 | BottleneckGenericAE_ld8_td10_agF | BottleneckGenericAE | 0.81   | 0.0147 |   13.65 | 1,493,700  |
|      7 | GenericAE_ld4_td5_agF            | GenericAE           | 0.8111 | 0.0067 |   13.68 | 1,602,600  |
|      8 | GenericAE_ld2_td5_agF            | GenericAE           | 0.8115 | 0.0104 |   13.71 | 1,592,340  |
|      9 | BottleneckGenericAE_ld16_td5_ag0 | BottleneckGenericAE | 0.8143 | 0.0137 |   13.72 | 1,507,290  |
|     10 | GenericAE_ld2_td5_ag0            | GenericAE           | 0.8199 | 0.0163 |   13.79 | 1,592,340  |
|     11 | BottleneckGenericAE_ld2_td8_agF  | BottleneckGenericAE | 0.8276 | 0.0194 |   13.93 | 1,451,940  |

The top configuration achieves median OWA = 0.7988 with 1,664,160 parameters.


## 3. Hyperparameter Marginals (Round 1 — Full Grid)

Round 1 provides a balanced factorial grid (120 rows, 40 configs) for unbiased marginal estimates.

### Block Type

| Block Type          |   Med OWA |   Mean OWA |    Std |   N | Med Params   |
|:--------------------|----------:|-----------:|-------:|----:|:-------------|
| GenericAE           |    1.26   |     1.3164 | 0.3254 |  24 | 1,612,860    |
| BottleneckGenericAE |    1.3654 |     1.6831 | 0.7384 |  96 | 1,464,585    |

**GenericAE** leads with the lowest median OWA; the gap to the worst level (BottleneckGenericAE) is 0.1054.

### Latent Dim

|   Latent Dim |   Med OWA |   Mean OWA |    Std |   N | Med Params   |
|-------------:|----------:|-----------:|-------:|----:|:-------------|
|            2 |    1.3174 |     1.3938 | 0.3515 |  30 | 1,451,940    |
|           16 |    1.3278 |     1.6318 | 0.7424 |  30 | 1,523,760    |
|            4 |    1.3623 |     1.6425 | 0.6324 |  30 | 1,462,200    |
|            8 |    1.3692 |     1.7711 | 0.8987 |  30 | 1,482,720    |

**2.0** leads with the lowest median OWA; the gap to the worst level (8.0) is 0.0518.

### Thetas Dim

|   Thetas Dim |   Med OWA |   Mean OWA |    Std |   N | Med Params   |
|-------------:|----------:|-----------:|-------:|----:|:-------------|
|            5 |    1.2913 |     1.4802 | 0.6025 |  48 | 1,549,815    |
|           10 |    1.3716 |     1.5477 | 0.5252 |  24 | 1,483,440    |
|            8 |    1.3747 |     1.8033 | 0.8666 |  24 | 1,472,460    |
|            3 |    1.4354 |     1.7376 | 0.7787 |  24 | 1,445,010    |

**5.0** leads with the lowest median OWA; the gap to the worst level (3.0) is 0.1441.

### active_g

| active_g   |   Med OWA |   Mean OWA |    Std |   N | Med Params   |
|:-----------|----------:|-----------:|-------:|----:|:-------------|
| forecast   |    1.3009 |     1.3002 | 0.084  |  60 | 1,477,950    |
| False      |    1.4723 |     1.9194 | 0.8723 |  60 | 1,477,950    |

**forecast** leads with the lowest median OWA; the gap to the worst level (False) is 0.1714.


## 3b. Selecting the Optimal Latent Dimension

The **latent dimension** sets the information bottleneck width in the AERootBlock backbone used by GenericAE and BottleneckGenericAE. The encoder compresses each block's input along the path `backcast_length → units/2 → latent_dim`, and the decoder expands it back via `latent_dim → units/2 → units`. The head layers then project to backcast and forecast outputs. A smaller latent_dim enforces stronger compression (regularisation), while a larger value preserves more information at the risk of overfitting.

With backcast_length = 30, the tested latent dimensions are: **2, 4, 8, 16**.

- **latent_dim = 2:** median OWA = 1.3174, std = 0.3515, params ≈ 1,451,940 ← best
- **latent_dim = 4:** median OWA = 1.3623, std = 0.6324, params ≈ 1,462,200
- **latent_dim = 8:** median OWA = 1.3692, std = 0.8987, params ≈ 1,482,720 ← worst
- **latent_dim = 16:** median OWA = 1.3278, std = 0.7424, params ≈ 1,523,760

The optimal setting is **latent_dim = 2** (median OWA 1.3174), outperforming the worst (latent_dim = 8) by Δ = 0.0518. 
The smallest bottleneck wins, suggesting aggressive compression is beneficial. Since GenericAE uses direct linear projections to target lengths (no structured basis), the backbone's inductive bias comes entirely from the AE bottleneck. A narrow latent prevents the network from memorising noise in the lookback window.

**Practical recommendation:** Use `latent_dim = 2` for GenericAE / BottleneckGenericAE stacks on M4-Yearly. For longer horizons or more complex datasets, consider scaling latent_dim proportionally to backcast_length (e.g. latent_dim ≈ backcast_length / 5–10) and re-evaluating via a small grid search.


## 4. Stability Analysis (OWA Spread Across Seeds)

Stability is measured by the OWA spread (max − min) across random seeds for each configuration.

### Round 1

- **Mean spread (max−min):** 0.6933
- **Max spread:** 2.7749 (BottleneckGenericAE_ld8_td8_ag0)
- **Mean std:** 0.3806

**Most stable configs:**

| Config                            |   Median OWA |   Range |    Std |
|:----------------------------------|-------------:|--------:|-------:|
| BottleneckGenericAE_ld16_td5_agF  |       1.2905 |  0.0021 | 0.0011 |
| BottleneckGenericAE_ld16_td10_agF |       1.3031 |  0.0399 | 0.0209 |
| GenericAE_ld4_td5_agF             |       1.3086 |  0.0544 | 0.0288 |
| BottleneckGenericAE_ld8_td8_agF   |       1.2671 |  0.0550 | 0.0317 |
| BottleneckGenericAE_ld4_td3_agF   |       1.2083 |  0.0710 | 0.0355 |

### Round 2

- **Mean spread (max−min):** 0.2291
- **Max spread:** 1.9226 (BottleneckGenericAE_ld2_td8_ag0)
- **Mean std:** 0.1242

**Most stable configs:**

| Config                           |   Median OWA |   Range |    Std |
|:---------------------------------|-------------:|--------:|-------:|
| GenericAE_ld16_td5_agF           |       0.8971 |  0.0220 | 0.0111 |
| BottleneckGenericAE_ld4_td3_agF  |       0.9800 |  0.0263 | 0.0140 |
| GenericAE_ld8_td5_ag0            |       0.9463 |  0.0519 | 0.0266 |
| BottleneckGenericAE_ld2_td5_ag0  |       0.9641 |  0.0528 | 0.0265 |
| BottleneckGenericAE_ld8_td10_agF |       0.9359 |  0.0792 | 0.0404 |

### Round 3

- **Mean spread (max−min):** 0.0174
- **Max spread:** 0.0570 (NBEATS-I+G_10stack)
- **Mean std:** 0.0089

**Most stable configs:**

| Config                           |   Median OWA |   Range |    Std |
|:---------------------------------|-------------:|--------:|-------:|
| GenericAE_ld4_td5_agF            |       0.8111 |  0.0067 | 0.0035 |
| GenericAE_ld16_td5_agF           |       0.7988 |  0.0073 | 0.0039 |
| GenericAE_ld16_td5_ag0           |       0.8074 |  0.0084 | 0.0042 |
| BottleneckGenericAE_ld8_td10_ag0 |       0.8096 |  0.0101 | 0.0051 |
| GenericAE_ld2_td5_agF            |       0.8115 |  0.0104 | 0.0052 |


## 5. Round-over-Round Progression (Final Configs)

Tracking the 11 configurations that survived to Round 3 across all earlier rounds.

| config_name                      |       R1 |       R2 |     R3 |        Δ |    Δ% |
|:---------------------------------|---------:|---------:|-------:|---------:|------:|
| BottleneckGenericAE_ld16_td5_ag0 |   1.5126 |   1.0231 | 0.8143 |  -0.6983 | -46.2 |
| BottleneckGenericAE_ld8_td10_ag0 |   1.4363 |   0.9255 | 0.8096 |  -0.6266 | -43.6 |
| BottleneckGenericAE_ld8_td10_agF |   1.4229 |   0.9359 | 0.81   |  -0.6129 | -43.1 |
| BottleneckGenericAE_ld2_td8_agF  |   1.3637 |   0.9347 | 0.8276 |  -0.5361 | -39.3 |
| GenericAE_ld4_td5_agF            |   1.3086 |   0.9806 | 0.8111 |  -0.4975 | -38   |
| GenericAE_ld2_td5_agF            |   1.3036 |   0.9037 | 0.8115 |  -0.4922 | -37.8 |
| GenericAE_ld8_td5_agF            |   1.2814 |   0.9182 | 0.8096 |  -0.4718 | -36.8 |
| GenericAE_ld16_td5_agF           |   1.2274 |   0.8971 | 0.7988 |  -0.4285 | -34.9 |
| GenericAE_ld2_td5_ag0            |   1.2385 |   0.9081 | 0.8199 |  -0.4187 | -33.8 |
| GenericAE_ld16_td5_ag0           |   1.1456 |   0.9593 | 0.8074 |  -0.3381 | -29.5 |
| NBEATS-I+G_10stack               | nan      | nan      | 0.8023 | nan      | nan   |

10 of 11 configs improved with additional training epochs. Mean Δ = -0.5121.


## 6. Parameter Efficiency (Final Round)

Reference: NBEATS-G 30-stack baseline has 24,700,000 parameters.

| Config                           | Block               | Params     | Reduction   |   Med OWA | Pareto   |
|:---------------------------------|:--------------------|:-----------|:------------|----------:|:---------|
| BottleneckGenericAE_ld2_td8_agF  | BottleneckGenericAE | 1,451,940  | 94.1%       |    0.8276 | ◀ PARETO |
| BottleneckGenericAE_ld8_td10_ag0 | BottleneckGenericAE | 1,493,700  | 94.0%       |    0.8096 | ◀ PARETO |
| BottleneckGenericAE_ld8_td10_agF | BottleneckGenericAE | 1,493,700  | 94.0%       |    0.81   |          |
| BottleneckGenericAE_ld16_td5_ag0 | BottleneckGenericAE | 1,507,290  | 93.9%       |    0.8143 |          |
| GenericAE_ld2_td5_agF            | GenericAE           | 1,592,340  | 93.6%       |    0.8115 |          |
| GenericAE_ld2_td5_ag0            | GenericAE           | 1,592,340  | 93.6%       |    0.8199 |          |
| GenericAE_ld4_td5_agF            | GenericAE           | 1,602,600  | 93.5%       |    0.8111 |          |
| GenericAE_ld8_td5_agF            | GenericAE           | 1,623,120  | 93.4%       |    0.8096 | ◀ PARETO |
| GenericAE_ld16_td5_ag0           | GenericAE           | 1,664,160  | 93.3%       |    0.8074 | ◀ PARETO |
| GenericAE_ld16_td5_agF           | GenericAE           | 1,664,160  | 93.3%       |    0.7988 | ◀ PARETO |
| NBEATS-I+G_10stack               | I+G                 | 19,506,435 | 21.0%       |    0.8023 |          |

Configurations on the Pareto frontier achieve the best OWA for their parameter budget — they cannot be improved on one axis without regressing on the other.


## 7. Statistical Significance vs NBEATS-I+G

Reference: NBEATS-I+G OWA = 0.8057
Using 5 empirical reference samples from block benchmark.

A one-sided Mann-Whitney U test (alternative = 'less') checks whether each configuration's OWA distribution is stochastically lower than NBEATS-I+G.

| Config                           |   Med OWA |   p-value | Sig   | Better?   |
|:---------------------------------|----------:|----------:|:------|:----------|
| GenericAE_ld16_td5_agF           |    0.7988 |    0.2857 | ns    | YES ✓     |
| NBEATS-I+G_10stack               |    0.8023 |    0.3929 | ns    | YES ✓     |
| GenericAE_ld16_td5_ag0           |    0.8074 |    0.6071 | ns    | no        |
| GenericAE_ld8_td5_agF            |    0.8096 |    0.7143 | ns    | no        |
| BottleneckGenericAE_ld8_td10_ag0 |    0.8096 |    0.8036 | ns    | no        |
| BottleneckGenericAE_ld8_td10_agF |    0.81   |    0.9286 | ns    | no        |
| GenericAE_ld4_td5_agF            |    0.8111 |    0.875  | ns    | no        |
| GenericAE_ld2_td5_agF            |    0.8115 |    0.875  | ns    | no        |
| BottleneckGenericAE_ld16_td5_ag0 |    0.8143 |    0.9286 | ns    | no        |
| GenericAE_ld2_td5_ag0            |    0.8199 |    0.9821 | ns    | no        |
| BottleneckGenericAE_ld2_td8_agF  |    0.8276 |    0.9821 | ns    | no        |


## 8. Training Stability (Divergence / Stopping)

Training stability tracks divergence rates and early-stopping behaviour across rounds.

|   Round |   Runs | Diverged   | Early Stopped   | Max Epochs   |
|--------:|-------:|:-----------|:----------------|:-------------|
|       1 |    120 | 0 (0.0%)   | 0 (0.0%)        | 120 (100.0%) |
|       2 |     60 | 0 (0.0%)   | 0 (0.0%)        | 60 (100.0%)  |
|       3 |     33 | 0 (0.0%)   | 1 (3.0%)        | 32 (97.0%)   |

