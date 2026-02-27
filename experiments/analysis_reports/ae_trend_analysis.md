# AE+Trend Architecture Search — Results Analysis

## Abstract

This study applies successive-halving architecture search to AE+Trend hybrid configurations on the M4-Yearly dataset. The AE+Trend stack pairs a compact autoencoder block with a trend block, aiming to match or beat established N-BEATS baselines at a fraction of the parameter cost.

**Key Takeaways:**

- **Best configuration:** `AutoEncoder_ld2_td3_ag0` achieves median OWA = **0.8045** with only 5,161,160 parameters (79% fewer than NBEATS-G).
- **Target OWA < 0.85:** Met ✓
- **Search scope:** 45 initial configs narrowed to 4 across 3 rounds of successive halving.
- **Total compute:** 23.4 minutes across 189 training runs.

- **CSV:** `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/ae_trend_search_results.csv`
- **Rows:** 189 (45 unique configs, 3 rounds)
- **Total training time:** 23.4 min


## 1. Successive Halving Funnel

|   Round |   Configs |   Runs |   Epochs |   Best Med OWA |
|--------:|----------:|-------:|---------:|---------------:|
|       1 |        45 |    135 |        6 |          0.913 |
|       2 |        14 |     42 |       15 |          0.842 |
|       3 |         4 |     12 |       30 |          0.805 |

The successive halving procedure pruned from 45 to 4 configurations across 3 rounds, retaining the top 9% of candidates. Each round increased the training budget while eliminating weaker configurations.


## 2.1 Round 1 Leaderboard

45 configs × 3 runs, 6 epochs each

|   Rank | Config                            |   OWA |     ± |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:----------------------------------|------:|------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld8_td3_ag0     | 0.913 | 0.014 |   15.24 |   3.55 | 5,197,310 | 5.5s   |          |
|      2 | AutoEncoder_ld2_td3_ag0           | 0.914 | 0.076 |   15.23 |   3.56 | 5,161,160 | 5.5s   |          |
|      3 | GenericAEBackcast_ld2_td3_ag0     | 0.934 | 0.037 |   15.63 |   3.6  | 5,143,280 | 5.5s   |          |
|      4 | GenericAE_ld2_td3_agF             | 0.941 | 0.192 |   15.73 |   3.65 | 1,826,585 | 5.0s   |          |
|      5 | GenericAE_ld8_td3_ag0             | 0.946 | 0.079 |   15.71 |   3.7  | 1,841,975 | 4.9s   |          |
|      6 | GenericAE_ld8_td3_agF             | 0.967 | 0.039 |   16.22 |   3.74 | 1,841,975 | 5.0s   |          |
|      7 | GenericAEBackcastAE_ld2_td5_ag0   | 0.987 | 0.12  |   16.6  |   3.8  | 1,862,800 | 5.3s   |          |
|      8 | GenericAEBackcastAE_ld8_td5_ag0   | 0.987 | 0.12  |   16.6  |   3.8  | 1,862,800 | 5.2s   |          |
|      9 | GenericAEBackcastAE_ld16_td5_ag0  | 0.987 | 0.12  |   16.6  |   3.8  | 1,862,800 | 5.2s   |          |
|     10 | GenericAEBackcast_ld2_td3_agF     | 1.003 | 0.099 |   16.26 |   4.02 | 5,143,280 | 5.5s   |          |
|     11 | GenericAE_ld16_td3_ag0            | 1.011 | 0.092 |   16.82 |   3.94 | 1,862,495 | 5.0s   |          |
|     12 | GenericAE_ld16_td3_agF            | 1.029 | 0.044 |   17.01 |   4.04 | 1,862,495 | 5.0s   |          |
|     13 | GenericAEBackcast_ld8_td3_agF     | 1.032 | 0.113 |   16.86 |   4.1  | 5,197,310 | 5.5s   |          |
|     14 | BottleneckGenericAE_ld8_td5_ag0   | 1.036 | 0.182 |   16.95 |   4.11 | 1,766,110 | 5.0s   |          |
|     15 | BottleneckGenericAE_ld8_td10_ag0  | 1.039 | 0.251 |   17.27 |   4.12 | 1,786,260 | 5.1s   |          |
|     16 | GenericAEBackcastAE_ld2_td5_agF   | 1.04  | 0.084 |   17.4  |   4.04 | 1,862,800 | 5.1s   |          |
|     17 | GenericAEBackcastAE_ld8_td5_agF   | 1.04  | 0.084 |   17.4  |   4.04 | 1,862,800 | 5.2s   |          |
|     18 | GenericAEBackcastAE_ld16_td5_agF  | 1.04  | 0.084 |   17.4  |   4.04 | 1,862,800 | 5.1s   |          |
|     19 | AutoEncoderAE_ld2_td5_ag0         | 1.043 | 0.139 |   17.23 |   4.1  | 1,872,880 | 5.2s   |          |
|     20 | AutoEncoder_ld8_td3_ag0           | 1.05  | 0.093 |   17.13 |   4.14 | 5,214,980 | 5.6s   |          |
|     21 | AutoEncoderAE_ld8_td5_ag0         | 1.05  | 0.069 |   17.4  |   4.11 | 1,888,270 | 5.1s   |          |
|     22 | BottleneckGenericAE_ld8_td10_agF  | 1.057 | 0.041 |   17.4  |   4.12 | 1,786,260 | 5.1s   |          |
|     23 | BottleneckGenericAE_ld16_td5_ag0  | 1.058 | 0.104 |   17.57 |   4.11 | 1,786,630 | 5.1s   |          |
|     24 | GenericAEBackcast_ld16_td3_agF    | 1.085 | 0.261 |   17.86 |   4.28 | 5,269,350 | 5.5s   |          |
|     25 | BottleneckGenericAE_ld2_td10_agF  | 1.088 | 0.141 |   17.83 |   4.31 | 1,770,870 | 5.1s   |          |
|     26 | GenericAE_ld2_td3_ag0             | 1.093 | 0.045 |   17.51 |   4.44 | 1,826,585 | 5.0s   |          |
|     27 | BottleneckGenericAE_ld16_td10_agF | 1.1   | 0.393 |   18.11 |   4.34 | 1,806,780 | 5.1s   |          |
|     28 | BottleneckGenericAE_ld2_td5_agF   | 1.1   | 0.121 |   17.87 |   4.38 | 1,750,720 | 5.0s   |          |
|     29 | GenericAEBackcast_ld16_td3_ag0    | 1.103 | 0.211 |   17.38 |   4.54 | 5,269,350 | 5.6s   |          |
|     30 | GenericAEBackcastAE_ld16_td10_ag0 | 1.107 | 0.207 |   18.14 |   4.39 | 1,907,825 | 5.2s   |          |
|     31 | GenericAEBackcastAE_ld2_td10_ag0  | 1.107 | 0.207 |   18.14 |   4.39 | 1,907,825 | 5.3s   |          |
|     32 | GenericAEBackcastAE_ld8_td10_ag0  | 1.107 | 0.207 |   18.14 |   4.39 | 1,907,825 | 5.3s   |          |
|     33 | AutoEncoderAE_ld16_td10_ag0       | 1.107 | 0.311 |   18.27 |   4.36 | 1,953,640 | 5.4s   |          |
|     34 | BottleneckGenericAE_ld16_td5_agF  | 1.117 | 0.062 |   18.3  |   4.43 | 1,786,630 | 5.1s   |          |
|     35 | AutoEncoderAE_ld16_td5_ag0        | 1.122 | 0.206 |   18.18 |   4.49 | 1,908,790 | 5.3s   |          |
|     36 | BottleneckGenericAE_ld2_td10_ag0  | 1.132 | 0.413 |   18.34 |   4.54 | 1,770,870 | 5.1s   |          |
|     37 | GenericAEBackcastAE_ld2_td10_agF  | 1.139 | 0.364 |   18.57 |   4.54 | 1,907,825 | 5.2s   |          |
|     38 | GenericAEBackcastAE_ld8_td10_agF  | 1.139 | 0.364 |   18.57 |   4.54 | 1,907,825 | 5.2s   |          |
|     39 | GenericAEBackcastAE_ld16_td10_agF | 1.139 | 0.364 |   18.57 |   4.54 | 1,907,825 | 5.2s   |          |
|     40 | BottleneckGenericAE_ld8_td5_agF   | 1.141 | 0.132 |   18.31 |   4.61 | 1,766,110 | 5.1s   |          |
|     41 | AutoEncoderAE_ld8_td10_ag0        | 1.143 | 0.177 |   18.68 |   4.54 | 1,933,120 | 5.2s   |          |
|     42 | BottleneckGenericAE_ld16_td10_ag0 | 1.168 | 0.116 |   18.94 |   4.67 | 1,806,780 | 5.1s   |          |
|     43 | AutoEncoderAE_ld2_td10_ag0        | 1.17  | 0.117 |   18.95 |   4.69 | 1,917,730 | 5.2s   |          |
|     44 | BottleneckGenericAE_ld2_td5_ag0   | 1.192 | 0.211 |   18.99 |   4.86 | 1,750,720 | 5.1s   |          |
|     45 | AutoEncoder_ld16_td3_ag0          | 1.243 | 0.163 |   19.43 |   5.16 | 5,286,740 | 5.2s   |          |

The top-ranked configuration achieves a median OWA of 0.913 with 5,197,310 parameters, while the worst scores 1.243. The spread between best and worst is 0.330.


## 2.2 Round 2 Leaderboard

14 configs × 3 runs, 15 epochs each

|   Rank | Config                           |   OWA |     ± |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------|------:|------:|--------:|-------:|:----------|:-------|:---------|
|      1 | AutoEncoder_ld2_td3_ag0          | 0.842 | 0.051 |   14.16 |   3.25 | 5,161,160 | 11.1s  |          |
|      2 | GenericAEBackcast_ld8_td3_ag0    | 0.848 | 0.058 |   14.21 |   3.29 | 5,197,310 | 11.6s  |          |
|      3 | GenericAEBackcast_ld8_td3_agF    | 0.852 | 0.139 |   14.28 |   3.3  | 5,197,310 | 11.7s  |          |
|      4 | GenericAE_ld2_td3_ag0            | 0.855 | 0.102 |   14.33 |   3.31 | 1,826,585 | 10.3s  |          |
|      5 | GenericAEBackcast_ld2_td3_agF    | 0.861 | 0.034 |   14.43 |   3.33 | 5,143,280 | 11.3s  |          |
|      6 | GenericAEBackcastAE_ld8_td5_ag0  | 0.864 | 0.022 |   14.56 |   3.33 | 1,862,800 | 11.1s  |          |
|      7 | GenericAEBackcastAE_ld2_td5_ag0  | 0.864 | 0.022 |   14.56 |   3.33 | 1,862,800 | 10.9s  |          |
|      8 | GenericAEBackcastAE_ld16_td5_ag0 | 0.864 | 0.022 |   14.56 |   3.33 | 1,862,800 | 11.2s  |          |
|      9 | GenericAE_ld2_td3_agF            | 0.866 | 0.088 |   14.38 |   3.39 | 1,826,585 | 10.4s  |          |
|     10 | GenericAE_ld8_td3_agF            | 0.874 | 0.054 |   14.56 |   3.41 | 1,841,975 | 10.4s  |          |
|     11 | GenericAEBackcast_ld2_td3_ag0    | 0.879 | 0.095 |   14.56 |   3.45 | 5,143,280 | 11.4s  |          |
|     12 | GenericAE_ld8_td3_ag0            | 0.893 | 0.111 |   14.62 |   3.54 | 1,841,975 | 10.4s  |          |
|     13 | BottleneckGenericAE_ld8_td5_ag0  | 0.894 | 0.101 |   14.78 |   3.51 | 1,766,110 | 10.5s  |          |
|     14 | GenericAEBackcast_ld16_td3_ag0   | 0.943 | 0.169 |   15.38 |   3.76 | 5,269,350 | 11.8s  |          |

The top-ranked configuration achieves a median OWA of 0.842 with 5,161,160 parameters, while the worst scores 0.943. The spread between best and worst is 0.102.


## 2.3 Round 3 Leaderboard

4 configs × 3 runs, 30 epochs each

|   Rank | Config                  |   OWA |     ± |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:------------------------|------:|------:|--------:|-------:|:----------|:-------|:---------|
|      1 | AutoEncoder_ld2_td3_ag0 | 0.805 | 0.011 |   13.54 |   3.09 | 5,161,160 | 20.6s  |          |
|      2 | GenericAE_ld2_td3_ag0   | 0.81  | 0.029 |   13.68 |   3.11 | 1,826,585 | 19.6s  | ✓        |
|      3 | GenericAE_ld2_td3_agF   | 0.818 | 0.055 |   13.71 |   3.17 | 1,826,585 | 19.9s  | ✓        |
|      4 | GenericAE_ld8_td3_ag0   | 0.834 | 0.058 |   13.88 |   3.26 | 1,841,975 | 20.0s  | ✓        |

The top-ranked configuration achieves a median OWA of 0.805 with 5,161,160 parameters, while the worst scores 0.834. The spread between best and worst is 0.030.
**3 configuration(s) meet both the OWA < 0.85 and params < 5,000,000 targets.**


## 3. Hyperparameter Marginals (Round 1 — Full Grid)


### AE Variant

| Value               |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|:--------------------|----------:|-----------:|----------:|----:|:-------------|
| GenericAE           |  0.999599 |    1.00777 | 0.0644266 |  18 | 1,841,975    |
| GenericAEBackcast   |  1.01728  |    1.02486 | 0.11473   |  18 | 5,197,310    |
| AutoEncoder         |  1.04969  |    1.06462 | 0.151396  |   9 | 5,214,980    |
| GenericAEBackcastAE |  1.07416  |    1.10249 | 0.132762  |  36 | 1,885,312    |
| BottleneckGenericAE |  1.09245  |    1.10586 | 0.102807  |  36 | 1,778,565    |
| AutoEncoderAE       |  1.09922  |    1.11798 | 0.102495  |  18 | 1,913,260    |

**GenericAE** yields the best median OWA while **AutoEncoderAE** is the weakest (Δ = 0.100).


### Latent Dim (search)

|   Value |   Med OWA |   Mean OWA |      Std |   N | Med Params   |
|--------:|----------:|-----------:|---------:|----:|:-------------|
|       8 |   1.04969 |    1.0561  | 0.108882 |  45 | 1,862,800    |
|       2 |   1.05636 |    1.06752 | 0.119271 |  45 | 1,862,800    |
|      16 |   1.09411 |    1.11623 | 0.118973 |  45 | 1,862,800    |

**8** yields the best median OWA while **16** is the weakest (Δ = 0.044).


### Thetas Dim (search)

|   Value |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|--------:|----------:|-----------:|----------:|----:|:-------------|
|       3 |   1.00312 |    1.02597 | 0.106294  |  45 | 5,143,280    |
|       5 |   1.05703 |    1.05802 | 0.0735446 |  45 | 1,862,800    |
|      10 |   1.1071  |    1.15585 | 0.127479  |  45 | 1,907,825    |

**3** yields the best median OWA while **10** is the weakest (Δ = 0.104).


### active_g

| Value    |   Med OWA |   Mean OWA |      Std |   N | Med Params   |
|:---------|----------:|-----------:|---------:|----:|:-------------|
| forecast |   1.05703 |    1.08367 | 0.123464 |  54 | 1,862,648    |
| False    |   1.05835 |    1.07747 | 0.114727 |  81 | 1,888,270    |

**forecast** yields the best median OWA while **False** is the weakest (Δ = 0.001).


## 3b. Selecting the Optimal Latent Dimension

The **latent dimension** controls the information bottleneck width in the AERootBlock backbone. The encoder compresses each block's input from `backcast_length → units/2 → latent_dim`, and the decoder expands it back to `latent_dim → units/2 → units` before the head layers produce backcast and forecast outputs. A smaller latent dim forces stronger compression, which acts as a regulariser but may discard useful signal; a larger latent dim preserves more information but risks overfitting.

Across this experiment (backcast_length = 30, forecast_length = 6), three latent dimensions were tested: **2, 8, 16**.

- **latent_dim = 2:** median OWA = 1.0564, std = 0.1193, params ≈ 1,862,800
- **latent_dim = 8:** median OWA = 1.0497, std = 0.1089, params ≈ 1,862,800 ← best
- **latent_dim = 16:** median OWA = 1.0941, std = 0.1190, params ≈ 1,862,800 ← worst

The optimal setting is **latent_dim = 8** (median OWA 1.0497), outperforming the worst setting (latent_dim = 16) by Δ = 0.0444. 
A mid-range bottleneck achieves the best balance between compression and information preservation, suggesting diminishing returns at higher dimensions while the lowest dimension is too aggressive.

**Practical recommendation:** Use `latent_dim = 8` as the default for AE+Trend configurations on M4-Yearly. When adapting to datasets with longer backcast windows or more complex seasonal patterns, consider scaling latent_dim proportionally (e.g. latent_dim ≈ backcast_length / 5–10).


## 4. Variant Head-to-Head


### Round 1 — Best Config per Variant

| Variant             | Best Config                     |   Med OWA |
|:--------------------|:--------------------------------|----------:|
| GenericAEBackcast   | GenericAEBackcast_ld8_td3_ag0   |  0.913146 |
| AutoEncoder         | AutoEncoder_ld2_td3_ag0         |  0.913729 |
| GenericAE           | GenericAE_ld2_td3_agF           |  0.940908 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld2_td5_ag0 |  0.986567 |
| BottleneckGenericAE | BottleneckGenericAE_ld8_td5_ag0 |  1.03623  |
| AutoEncoderAE       | AutoEncoderAE_ld2_td5_ag0       |  1.04292  |

### Round 2 — Best Config per Variant

| Variant             | Best Config                      |   Med OWA |
|:--------------------|:---------------------------------|----------:|
| AutoEncoder         | AutoEncoder_ld2_td3_ag0          |  0.841577 |
| GenericAEBackcast   | GenericAEBackcast_ld8_td3_ag0    |  0.847533 |
| GenericAE           | GenericAE_ld2_td3_ag0            |  0.854987 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld16_td5_ag0 |  0.863793 |
| BottleneckGenericAE | BottleneckGenericAE_ld8_td5_ag0  |  0.893994 |

### Round 3 — Best Config per Variant

| Variant     | Best Config             |   Med OWA |
|:------------|:------------------------|----------:|
| AutoEncoder | AutoEncoder_ld2_td3_ag0 |  0.804516 |
| GenericAE   | GenericAE_ld2_td3_ag0   |  0.810055 |

This head-to-head comparison reveals which AE backbone architectures are most competitive at each stage of the search. Variants that maintain top positions across rounds demonstrate robust performance independent of training budget.


## 5. Stability Analysis (OWA Spread Across Seeds)


### Round 1

- **Mean spread (max−min):** 0.157
- **Max spread (max−min):** 0.413 (`BottleneckGenericAE_ld2_td10_ag0`)
- **Mean std:** 0.083
- **Most stable configs:** `GenericAEBackcast_ld8_td3_ag0` (±0.014), `GenericAEBackcast_ld2_td3_ag0` (±0.037), `GenericAE_ld8_td3_agF` (±0.039)

### Round 2

- **Mean spread (max−min):** 0.076
- **Max spread (max−min):** 0.169 (`GenericAEBackcast_ld16_td3_ag0`)
- **Mean std:** 0.040
- **Most stable configs:** `GenericAEBackcastAE_ld2_td5_ag0` (±0.022), `GenericAEBackcastAE_ld16_td5_ag0` (±0.022), `GenericAEBackcastAE_ld8_td5_ag0` (±0.022)

### Round 3

- **Mean spread (max−min):** 0.038
- **Max spread (max−min):** 0.058 (`GenericAE_ld8_td3_ag0`)
- **Mean std:** 0.020
- **Most stable configs:** `AutoEncoder_ld2_td3_ag0` (±0.011), `GenericAE_ld2_td3_ag0` (±0.029), `GenericAE_ld2_td3_agF` (±0.055)

Lower spread values indicate more consistent performance across random seeds. Configurations with high spread may be sensitive to initialization.


## 6. Round-over-Round Progression (Final Configs)

| config_name             |    R1 |    R2 |    R3 |      Δ |    Δ% |
|:------------------------|------:|------:|------:|-------:|------:|
| GenericAE_ld2_td3_ag0   | 1.093 | 0.855 | 0.81  | -0.283 | -25.9 |
| GenericAE_ld2_td3_agF   | 0.941 | 0.866 | 0.818 | -0.123 | -13.1 |
| GenericAE_ld8_td3_ag0   | 0.946 | 0.893 | 0.834 | -0.112 | -11.8 |
| AutoEncoder_ld2_td3_ag0 | 0.914 | 0.842 | 0.805 | -0.109 | -12   |

**4 of 4 surviving configurations improved** their OWA from the first to the last round, confirming that additional training epochs benefit the top candidates.


## 7. Parameter Efficiency (Final Round)

| Config                  | Params    | Reduction   |   Med OWA | Target   |
|:------------------------|:----------|:------------|----------:|:---------|
| GenericAE_ld2_td3_ag0   | 1,826,585 | 92.6%       |     0.81  | ✓        |
| GenericAE_ld2_td3_agF   | 1,826,585 | 92.6%       |     0.818 | ✓        |
| GenericAE_ld8_td3_ag0   | 1,841,975 | 92.5%       |     0.834 | ✓        |
| AutoEncoder_ld2_td3_ag0 | 5,161,160 | 79.1%       |     0.805 | ✗        |

All AE+Trend configurations achieve substantial parameter reductions relative to the 24,700,000-parameter Generic baseline. The best-performing config uses 5,161,160 parameters (79% reduction) while achieving OWA = 0.805.


## 8. Final Verdict

**Target:** OWA < 0.85, Params < 5,000,000
**Baseline:** N-BEATS-G 30-stack = 24,700,000 params

✅ **3 configuration(s) MEET the target:**

**GenericAE_ld2_td3_ag0**

- OWA: 0.810 (range 0.801–0.830)
- sMAPE: 13.68, MASE: 3.11
- Params: 1,826,585 (93% reduction)
- Hyperparams: ae=GenericAE, latent_dim=2, thetas_dim=3, active_g=False

**GenericAE_ld2_td3_agF**

- OWA: 0.818 (range 0.805–0.859)
- sMAPE: 13.71, MASE: 3.17
- Params: 1,826,585 (93% reduction)
- Hyperparams: ae=GenericAE, latent_dim=2, thetas_dim=3, active_g=forecast

**GenericAE_ld8_td3_ag0**

- OWA: 0.834 (range 0.808–0.866)
- sMAPE: 13.88, MASE: 3.26
- Params: 1,841,975 (93% reduction)
- Hyperparams: ae=GenericAE, latent_dim=8, thetas_dim=3, active_g=False

⚠️ **1 configuration(s) meet OWA target but exceed parameter budget:**

| Config                  |   OWA | Params    |
|:------------------------|------:|:----------|
| AutoEncoder_ld2_td3_ag0 | 0.805 | 5,161,160 |

