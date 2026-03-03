# AE+Trend Architecture Search - Multi-Dataset Analysis


## Dataset: m4

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/ae_trend_search_results.csv`
- Rows: 189
- Primary metric: `owa`

### Abstract

This analysis covers 45 configurations over 3 rounds (189 runs).
Total training time: 23.4 min.
OWA range: 0.7942 - 1.4551.


## 1. Successive Halving Funnel

|   Round |   Configs |   Runs |   Epochs |   Best Med OWA |
|--------:|----------:|-------:|---------:|---------------:|
|       1 |        45 |    135 |        6 |         0.9131 |
|       2 |        14 |     42 |       15 |         0.8416 |
|       3 |         4 |     12 |       30 |         0.8045 |


## 2.1 Round 1 Leaderboard

45 configs x 3 runs, up to 6 epochs each

|   Rank | Config                            |    OWA |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:----------------------------------|-------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld8_td3_ag0     | 0.9131 |   0.014  |   15.24 |   3.55 | 5,197,310 | 5.5s   |          |
|      2 | AutoEncoder_ld2_td3_ag0           | 0.9137 |   0.0759 |   15.23 |   3.56 | 5,161,160 | 5.5s   |          |
|      3 | GenericAEBackcast_ld2_td3_ag0     | 0.9344 |   0.0369 |   15.63 |   3.6  | 5,143,280 | 5.5s   |          |
|      4 | GenericAE_ld2_td3_agF             | 0.9409 |   0.1922 |   15.73 |   3.65 | 1,826,585 | 5.0s   |          |
|      5 | GenericAE_ld8_td3_ag0             | 0.9465 |   0.0789 |   15.71 |   3.7  | 1,841,975 | 4.9s   |          |
|      6 | GenericAE_ld8_td3_agF             | 0.9667 |   0.039  |   16.22 |   3.74 | 1,841,975 | 5.0s   |          |
|      7 | GenericAEBackcastAE_ld2_td5_ag0   | 0.9866 |   0.1205 |   16.6  |   3.8  | 1,862,800 | 5.3s   |          |
|      8 | GenericAEBackcastAE_ld8_td5_ag0   | 0.9866 |   0.1205 |   16.6  |   3.8  | 1,862,800 | 5.2s   |          |
|      9 | GenericAEBackcastAE_ld16_td5_ag0  | 0.9866 |   0.1205 |   16.6  |   3.8  | 1,862,800 | 5.2s   |          |
|     10 | GenericAEBackcast_ld2_td3_agF     | 1.0028 |   0.0994 |   16.26 |   4.02 | 5,143,280 | 5.5s   |          |
|     11 | GenericAE_ld16_td3_ag0            | 1.0106 |   0.0916 |   16.82 |   3.94 | 1,862,495 | 5.0s   |          |
|     12 | GenericAE_ld16_td3_agF            | 1.0294 |   0.0443 |   17.01 |   4.04 | 1,862,495 | 5.0s   |          |
|     13 | GenericAEBackcast_ld8_td3_agF     | 1.0317 |   0.1132 |   16.86 |   4.1  | 5,197,310 | 5.5s   |          |
|     14 | BottleneckGenericAE_ld8_td5_ag0   | 1.0362 |   0.1816 |   16.95 |   4.11 | 1,766,110 | 5.0s   |          |
|     15 | BottleneckGenericAE_ld8_td10_ag0  | 1.0394 |   0.2507 |   17.27 |   4.12 | 1,786,260 | 5.1s   |          |
|     16 | GenericAEBackcastAE_ld2_td5_agF   | 1.0404 |   0.0837 |   17.4  |   4.04 | 1,862,800 | 5.1s   |          |
|     17 | GenericAEBackcastAE_ld8_td5_agF   | 1.0404 |   0.0837 |   17.4  |   4.04 | 1,862,800 | 5.2s   |          |
|     18 | GenericAEBackcastAE_ld16_td5_agF  | 1.0404 |   0.0837 |   17.4  |   4.04 | 1,862,800 | 5.1s   |          |
|     19 | AutoEncoderAE_ld2_td5_ag0         | 1.0429 |   0.1385 |   17.23 |   4.1  | 1,872,880 | 5.2s   |          |
|     20 | AutoEncoder_ld8_td3_ag0           | 1.0497 |   0.0934 |   17.13 |   4.14 | 5,214,980 | 5.6s   |          |
|     21 | AutoEncoderAE_ld8_td5_ag0         | 1.0498 |   0.069  |   17.4  |   4.11 | 1,888,270 | 5.1s   |          |
|     22 | BottleneckGenericAE_ld8_td10_agF  | 1.057  |   0.0405 |   17.4  |   4.12 | 1,786,260 | 5.1s   |          |
|     23 | BottleneckGenericAE_ld16_td5_ag0  | 1.0583 |   0.1042 |   17.57 |   4.11 | 1,786,630 | 5.1s   |          |
|     24 | GenericAEBackcast_ld16_td3_agF    | 1.0848 |   0.2614 |   17.86 |   4.28 | 5,269,350 | 5.5s   |          |
|     25 | BottleneckGenericAE_ld2_td10_agF  | 1.0884 |   0.1413 |   17.83 |   4.31 | 1,770,870 | 5.1s   |          |
|     26 | GenericAE_ld2_td3_ag0             | 1.0933 |   0.0453 |   17.51 |   4.44 | 1,826,585 | 5.0s   |          |
|     27 | BottleneckGenericAE_ld16_td10_agF | 1.0996 |   0.3934 |   18.11 |   4.34 | 1,806,780 | 5.1s   |          |
|     28 | BottleneckGenericAE_ld2_td5_agF   | 1.0998 |   0.1211 |   17.87 |   4.38 | 1,750,720 | 5.0s   |          |
|     29 | GenericAEBackcast_ld16_td3_ag0    | 1.1027 |   0.2111 |   17.38 |   4.54 | 5,269,350 | 5.6s   |          |
|     30 | GenericAEBackcastAE_ld16_td10_ag0 | 1.1071 |   0.2067 |   18.14 |   4.39 | 1,907,825 | 5.2s   |          |
|     31 | GenericAEBackcastAE_ld2_td10_ag0  | 1.1071 |   0.2067 |   18.14 |   4.39 | 1,907,825 | 5.3s   |          |
|     32 | GenericAEBackcastAE_ld8_td10_ag0  | 1.1071 |   0.2067 |   18.14 |   4.39 | 1,907,825 | 5.3s   |          |
|     33 | AutoEncoderAE_ld16_td10_ag0       | 1.1075 |   0.3113 |   18.27 |   4.36 | 1,953,640 | 5.4s   |          |
|     34 | BottleneckGenericAE_ld16_td5_agF  | 1.1166 |   0.0623 |   18.3  |   4.43 | 1,786,630 | 5.1s   |          |
|     35 | AutoEncoderAE_ld16_td5_ag0        | 1.1217 |   0.2063 |   18.18 |   4.49 | 1,908,790 | 5.3s   |          |
|     36 | BottleneckGenericAE_ld2_td10_ag0  | 1.1318 |   0.4126 |   18.34 |   4.54 | 1,770,870 | 5.1s   |          |
|     37 | GenericAEBackcastAE_ld2_td10_agF  | 1.1388 |   0.3638 |   18.57 |   4.54 | 1,907,825 | 5.2s   |          |
|     38 | GenericAEBackcastAE_ld8_td10_agF  | 1.1388 |   0.3638 |   18.57 |   4.54 | 1,907,825 | 5.2s   |          |
|     39 | GenericAEBackcastAE_ld16_td10_agF | 1.1388 |   0.3638 |   18.57 |   4.54 | 1,907,825 | 5.2s   |          |
|     40 | BottleneckGenericAE_ld8_td5_agF   | 1.1408 |   0.132  |   18.31 |   4.61 | 1,766,110 | 5.1s   |          |
|     41 | AutoEncoderAE_ld8_td10_ag0        | 1.1431 |   0.1768 |   18.68 |   4.54 | 1,933,120 | 5.2s   |          |
|     42 | BottleneckGenericAE_ld16_td10_ag0 | 1.1675 |   0.1164 |   18.94 |   4.67 | 1,806,780 | 5.1s   |          |
|     43 | AutoEncoderAE_ld2_td10_ag0        | 1.1696 |   0.1171 |   18.95 |   4.69 | 1,917,730 | 5.2s   |          |
|     44 | BottleneckGenericAE_ld2_td5_ag0   | 1.192  |   0.2109 |   18.99 |   4.86 | 1,750,720 | 5.1s   |          |
|     45 | AutoEncoder_ld16_td3_ag0          | 1.243  |   0.1631 |   19.43 |   5.16 | 5,286,740 | 5.2s   |          |

Top config median OWA=0.9131; worst=1.2430; delta=0.3298.


## 2.2 Round 2 Leaderboard

14 configs x 3 runs, up to 15 epochs each

|   Rank | Config                           |    OWA |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------|-------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | AutoEncoder_ld2_td3_ag0          | 0.8416 |   0.0513 |   14.16 |   3.25 | 5,161,160 | 11.1s  |          |
|      2 | GenericAEBackcast_ld8_td3_ag0    | 0.8475 |   0.0578 |   14.21 |   3.29 | 5,197,310 | 11.6s  |          |
|      3 | GenericAEBackcast_ld8_td3_agF    | 0.852  |   0.1392 |   14.28 |   3.3  | 5,197,310 | 11.7s  |          |
|      4 | GenericAE_ld2_td3_ag0            | 0.855  |   0.1017 |   14.33 |   3.31 | 1,826,585 | 10.3s  |          |
|      5 | GenericAEBackcast_ld2_td3_agF    | 0.8607 |   0.0338 |   14.43 |   3.33 | 5,143,280 | 11.3s  |          |
|      6 | GenericAEBackcastAE_ld8_td5_ag0  | 0.8638 |   0.0218 |   14.56 |   3.33 | 1,862,800 | 11.1s  |          |
|      7 | GenericAEBackcastAE_ld2_td5_ag0  | 0.8638 |   0.0218 |   14.56 |   3.33 | 1,862,800 | 10.9s  |          |
|      8 | GenericAEBackcastAE_ld16_td5_ag0 | 0.8638 |   0.0218 |   14.56 |   3.33 | 1,862,800 | 11.2s  |          |
|      9 | GenericAE_ld2_td3_agF            | 0.8661 |   0.0881 |   14.38 |   3.39 | 1,826,585 | 10.4s  |          |
|     10 | GenericAE_ld8_td3_agF            | 0.8741 |   0.0544 |   14.56 |   3.41 | 1,841,975 | 10.4s  |          |
|     11 | GenericAEBackcast_ld2_td3_ag0    | 0.8791 |   0.0949 |   14.56 |   3.45 | 5,143,280 | 11.4s  |          |
|     12 | GenericAE_ld8_td3_ag0            | 0.8928 |   0.1108 |   14.62 |   3.54 | 1,841,975 | 10.4s  |          |
|     13 | BottleneckGenericAE_ld8_td5_ag0  | 0.894  |   0.1009 |   14.78 |   3.51 | 1,766,110 | 10.5s  |          |
|     14 | GenericAEBackcast_ld16_td3_ag0   | 0.9432 |   0.1686 |   15.38 |   3.76 | 5,269,350 | 11.8s  |          |

Top config median OWA=0.8416; worst=0.9432; delta=0.1016.


## 2.3 Round 3 Leaderboard

4 configs x 3 runs, up to 30 epochs each

|   Rank | Config                  |    OWA |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:------------------------|-------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | AutoEncoder_ld2_td3_ag0 | 0.8045 |   0.0114 |   13.54 |   3.09 | 5,161,160 | 20.6s  |          |
|      2 | GenericAE_ld2_td3_ag0   | 0.8101 |   0.0291 |   13.68 |   3.11 | 1,826,585 | 19.6s  | YES      |
|      3 | GenericAE_ld2_td3_agF   | 0.8181 |   0.0546 |   13.71 |   3.17 | 1,826,585 | 19.9s  | YES      |
|      4 | GenericAE_ld8_td3_ag0   | 0.8343 |   0.058  |   13.88 |   3.26 | 1,841,975 | 20.0s  | YES      |

Top config median OWA=0.8045; worst=0.8343; delta=0.0298.


## 3. Hyperparameter Marginals (Round 1)


### AE Variant

| Value               |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|:--------------------|----------:|-----------:|----------:|----:|:-------------|
| GenericAE           |  0.999599 |    1.00777 | 0.0644266 |  18 | 1,841,975    |
| GenericAEBackcast   |  1.01728  |    1.02486 | 0.11473   |  18 | 5,197,310    |
| AutoEncoder         |  1.04969  |    1.06462 | 0.151396  |   9 | 5,214,980    |
| GenericAEBackcastAE |  1.07416  |    1.10249 | 0.132762  |  36 | 1,885,312    |
| BottleneckGenericAE |  1.09245  |    1.10586 | 0.102807  |  36 | 1,778,565    |
| AutoEncoderAE       |  1.09922  |    1.11798 | 0.102495  |  18 | 1,913,260    |


### Latent Dim

|   Value |   Med OWA |   Mean OWA |      Std |   N | Med Params   |
|--------:|----------:|-----------:|---------:|----:|:-------------|
|       8 |   1.04969 |    1.0561  | 0.108882 |  45 | 1,862,800    |
|       2 |   1.05636 |    1.06752 | 0.119271 |  45 | 1,862,800    |
|      16 |   1.09411 |    1.11623 | 0.118973 |  45 | 1,862,800    |


### Thetas Dim

|   Value |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|--------:|----------:|-----------:|----------:|----:|:-------------|
|       3 |   1.00312 |    1.02597 | 0.106294  |  45 | 5,143,280    |
|       5 |   1.05703 |    1.05802 | 0.0735446 |  45 | 1,862,800    |
|      10 |   1.1071  |    1.15585 | 0.127479  |  45 | 1,907,825    |


### active_g

| Value    |   Med OWA |   Mean OWA |      Std |   N | Med Params   |
|:---------|----------:|-----------:|---------:|----:|:-------------|
| forecast |   1.05703 |    1.08367 | 0.123464 |  54 | 1,862,648    |
| False    |   1.05835 |    1.07747 | 0.114727 |  81 | 1,888,270    |


## 3b. Latent Dimension Discussion

|   latent_dim_cfg |   med_metric |   std_metric |   n |   med_params |
|-----------------:|-------------:|-------------:|----:|-------------:|
|                8 |      1.04969 |     0.108882 |  45 |   1.8628e+06 |
|                2 |      1.05636 |     0.119271 |  45 |   1.8628e+06 |
|               16 |      1.09411 |     0.118973 |  45 |   1.8628e+06 |

Best latent_dim=8 by median OWA (1.0497).


## 4. Variant Head-to-Head


### Round 1 - Best Config per Variant

| Variant             | Best Config                     |   Med OWA |
|:--------------------|:--------------------------------|----------:|
| GenericAEBackcast   | GenericAEBackcast_ld8_td3_ag0   |  0.913146 |
| AutoEncoder         | AutoEncoder_ld2_td3_ag0         |  0.913729 |
| GenericAE           | GenericAE_ld2_td3_agF           |  0.940908 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld2_td5_ag0 |  0.986567 |
| BottleneckGenericAE | BottleneckGenericAE_ld8_td5_ag0 |  1.03623  |
| AutoEncoderAE       | AutoEncoderAE_ld2_td5_ag0       |  1.04292  |

### Round 2 - Best Config per Variant

| Variant             | Best Config                      |   Med OWA |
|:--------------------|:---------------------------------|----------:|
| AutoEncoder         | AutoEncoder_ld2_td3_ag0          |  0.841577 |
| GenericAEBackcast   | GenericAEBackcast_ld8_td3_ag0    |  0.847533 |
| GenericAE           | GenericAE_ld2_td3_ag0            |  0.854987 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld16_td5_ag0 |  0.863793 |
| BottleneckGenericAE | BottleneckGenericAE_ld8_td5_ag0  |  0.893994 |

### Round 3 - Best Config per Variant

| Variant     | Best Config             |   Med OWA |
|:------------|:------------------------|----------:|
| AutoEncoder | AutoEncoder_ld2_td3_ag0 |  0.804516 |
| GenericAE   | GenericAE_ld2_td3_ag0   |  0.810055 |


## 5. Stability Analysis


### Round 1

- Mean spread: 0.1571
- Max spread:  0.4126 (BottleneckGenericAE_ld2_td10_ag0)
- Mean std:    0.0829

### Round 2

- Mean spread: 0.0762
- Max spread:  0.1686 (GenericAEBackcast_ld16_td3_ag0)
- Mean std:    0.0399

### Round 3

- Mean spread: 0.0383
- Max spread:  0.0580 (GenericAE_ld8_td3_ag0)
- Mean std:    0.0197


## 6. Round-over-Round Progression

| config_name             |     R1 |     R2 |     R3 |   Delta |   DeltaPct |
|:------------------------|-------:|-------:|-------:|--------:|-----------:|
| GenericAE_ld2_td3_ag0   | 1.0933 | 0.855  | 0.8101 | -0.2833 |      -25.9 |
| GenericAE_ld2_td3_agF   | 0.9409 | 0.8661 | 0.8181 | -0.1228 |      -13.1 |
| AutoEncoder_ld2_td3_ag0 | 0.9137 | 0.8416 | 0.8045 | -0.1092 |      -12   |
| GenericAE_ld8_td3_ag0   | 0.9465 | 0.8928 | 0.8343 | -0.1121 |      -11.8 |


## 7. Parameter Efficiency

| Config                  | Params    | Reduction   |   Med OWA |
|:------------------------|:----------|:------------|----------:|
| GenericAE_ld2_td3_ag0   | 1,826,585 | 92.6%       |    0.8101 |
| GenericAE_ld2_td3_agF   | 1,826,585 | 92.6%       |    0.8181 |
| GenericAE_ld8_td3_ag0   | 1,841,975 | 92.5%       |    0.8343 |
| AutoEncoder_ld2_td3_ag0 | 5,161,160 | 79.1%       |    0.8045 |


## 8. Final Verdict

Target: OWA < 0.85, Params < 5,000,000
3 configurations meet both targets.
Best final-round config: AutoEncoder_ld2_td3_ag0 with median OWA=0.8045.


## Dataset: tourism

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/tourism/ae_trend_search_results.csv`
- Rows: 423
- Primary metric: `best_val_loss`

### Abstract

This analysis covers 90 configurations over 3 rounds (423 runs).
Total training time: 4.0 min.
OWA unavailable; using best_val_loss throughout.


## 1. Successive Halving Funnel

|   Round |   Configs |   Runs |   Epochs |   Best Med best_val_loss |
|--------:|----------:|-------:|---------:|-------------------------:|
|       1 |        90 |    270 |        7 |                  29.8453 |
|       2 |        39 |    117 |       15 |                  25.1231 |
|       3 |        12 |     36 |       30 |                  24.7549 |


## 2.1 Round 1 Leaderboard

90 configs x 3 runs, up to 7 epochs each

|   Rank | Config                                 |   best_val_loss |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|----------------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld8_td5_ttd5_agF     |         29.8453 |   3.7474 |   24.33 |   3.68 | 5,052,465 | 0.6s   |          |
|      2 | GenericAEBackcast_ld2_td5_ttd3_agF     |         30.044  |   1.3447 |   25.39 |   3.75 | 5,003,635 | 0.7s   |          |
|      3 | GenericAEBackcast_ld8_td5_ttd3_agF     |         30.3069 |   2.2868 |   25.4  |   3.73 | 5,049,895 | 0.6s   |          |
|      4 | BottleneckGenericAE_ld16_td10_ttd5_agF |         30.7287 |   8.1852 |   27.76 |   4.08 | 1,742,835 | 0.3s   |          |
|      5 | GenericAE_ld16_td5_ttd3_agF            |         30.7795 |   0.9703 |   25.92 |   3.83 | 1,744,735 | 0.3s   |          |
|      6 | BottleneckGenericAE_ld16_td10_ttd3_agF |         30.8546 |   5.4826 |   26.03 |   3.85 | 1,740,265 | 0.3s   |          |
|      7 | AutoEncoderAE_ld2_td5_ttd3_ag0         |         30.892  |   5.194  |   27.6  |   4.11 | 1,752,430 | 0.3s   |          |
|      8 | AutoEncoder_ld2_td5_ttd3_ag0           |         30.8967 |   2.5117 |   27.4  |   4.25 | 5,016,405 | 0.6s   |          |
|      9 | GenericAEBackcast_ld8_td5_ttd5_ag0     |         30.9164 |   1.2908 |   25.61 |   3.76 | 5,052,465 | 0.6s   |          |
|     10 | GenericAE_ld16_td5_ttd3_ag0            |         30.9321 |   1.3172 |   26.6  |   3.91 | 1,744,735 | 0.3s   |          |
|     11 | AutoEncoder_ld8_td5_ttd3_ag0           |         30.9721 |   5.3081 |   26.02 |   3.87 | 5,062,515 | 0.5s   |          |
|     12 | GenericAEBackcast_ld16_td5_ttd5_agF    |         30.9787 |   1.1041 |   26.54 |   3.91 | 5,114,145 | 0.6s   |          |
|     13 | GenericAE_ld16_td5_ttd5_agF            |         30.9788 |   2.3439 |   27.45 |   4.06 | 1,747,305 | 0.3s   |          |
|     14 | GenericAEBackcast_ld16_td5_ttd3_agF    |         31.0889 |   2.588  |   27.32 |   4.05 | 5,111,575 | 0.6s   |          |
|     15 | GenericAEBackcast_ld8_td5_ttd3_ag0     |         31.0963 |   3.437  |   25.66 |   3.78 | 5,049,895 | 0.6s   |          |
|     16 | GenericAEBackcast_ld2_td5_ttd5_agF     |         31.1192 |   5.1402 |   26.03 |   3.92 | 5,006,205 | 0.6s   |          |
|     17 | BottleneckGenericAE_ld16_td10_ttd5_ag0 |         31.1206 |   9.2458 |   27.53 |   4.06 | 1,742,835 | 0.3s   |          |
|     18 | AutoEncoderAE_ld16_td10_ttd5_ag0       |         31.1656 |   7.5833 |   26.54 |   3.93 | 1,829,335 | 0.3s   |          |
|     19 | GenericAEBackcast_ld16_td5_ttd3_ag0    |         31.3069 |   4.2654 |   25.22 |   3.83 | 5,111,575 | 0.6s   |          |
|     20 | GenericAEBackcastAE_ld16_td10_ttd3_ag0 |         31.612  |   1.2552 |   27.57 |   4.14 | 1,785,980 | 0.3s   |          |
|     21 | GenericAEBackcastAE_ld2_td10_ttd3_ag0  |         31.612  |   1.2552 |   27.57 |   4.14 | 1,785,980 | 0.3s   |          |
|     22 | GenericAEBackcastAE_ld8_td10_ttd3_ag0  |         31.612  |   1.2552 |   27.57 |   4.14 | 1,785,980 | 0.4s   |          |
|     23 | AutoEncoder_ld16_td5_ttd3_ag0          |         31.8184 |   0.6545 |   28.78 |   4.23 | 5,123,995 | 0.4s   |          |
|     24 | GenericAE_ld16_td5_ttd5_ag0            |         31.9074 |   2.3284 |   28.26 |   4.25 | 1,747,305 | 0.3s   |          |
|     25 | GenericAEBackcastAE_ld8_td10_ttd3_agF  |         31.9734 |   1.937  |   26.84 |   3.97 | 1,785,980 | 0.3s   |          |
|     26 | GenericAEBackcastAE_ld2_td10_ttd3_agF  |         31.9734 |   1.937  |   26.84 |   3.97 | 1,785,980 | 0.3s   |          |
|     27 | GenericAEBackcastAE_ld16_td10_ttd3_agF |         31.9734 |   1.937  |   26.84 |   3.97 | 1,785,980 | 0.3s   |          |
|     28 | GenericAE_ld8_td5_ttd5_agF             |         32.1183 |   3.2888 |   26.72 |   4.05 | 1,726,785 | 0.3s   |          |
|     29 | BottleneckGenericAE_ld16_td10_ttd3_ag0 |         32.151  |   4.0872 |   26.9  |   4.07 | 1,740,265 | 0.3s   |          |
|     30 | GenericAEBackcast_ld2_td5_ttd3_ag0     |         32.2521 |   4.9132 |   26.68 |   4.01 | 5,003,635 | 0.5s   |          |
|     31 | GenericAE_ld8_td5_ttd3_agF             |         32.5856 |   1.8802 |   27.51 |   4.2  | 1,724,215 | 0.3s   |          |
|     32 | BottleneckGenericAE_ld8_td5_ttd3_agF   |         32.6049 |   3.3295 |   27.64 |   4.12 | 1,706,620 | 0.3s   |          |
|     33 | GenericAE_ld8_td5_ttd3_ag0             |         32.6115 |   3.9018 |   27.93 |   4.33 | 1,724,215 | 0.3s   |          |
|     34 | AutoEncoder_ld8_td5_ttd5_ag0           |         32.6243 |   5.5179 |   28.93 |   4.5  | 5,065,085 | 0.5s   |          |
|     35 | GenericAEBackcast_ld16_td5_ttd5_ag0    |         32.6313 |   1.6323 |   28.12 |   4.17 | 5,114,145 | 0.6s   |          |
|     36 | AutoEncoderAE_ld16_td5_ttd5_ag0        |         32.6539 |   9.7764 |   27.1  |   4.17 | 1,790,910 | 0.3s   |          |
|     37 | GenericAE_ld2_td5_ttd3_ag0             |         32.6897 |   1.6868 |   27.68 |   4.4  | 1,708,825 | 0.3s   |          |
|     38 | GenericAEBackcast_ld2_td5_ttd5_ag0     |         32.7681 |   0.5167 |   27.63 |   4.26 | 5,006,205 | 0.6s   |          |
|     39 | GenericAE_ld2_td5_ttd3_agF             |         32.8525 |   3.094  |   27.99 |   4.49 | 1,708,825 | 0.3s   |          |
|     40 | AutoEncoder_ld2_td5_ttd5_ag0           |         32.9352 |   1.5114 |   28.42 |   4.33 | 5,018,975 | 0.5s   |          |
|     41 | BottleneckGenericAE_ld16_td5_ttd5_agF  |         32.9701 |   4.7483 |   28.3  |   4.32 | 1,729,710 | 0.3s   |          |
|     42 | BottleneckGenericAE_ld16_td5_ttd5_ag0  |         33.0853 |   4.6702 |   27.73 |   4.21 | 1,729,710 | 0.3s   |          |
|     43 | BottleneckGenericAE_ld8_td5_ttd3_ag0   |         33.2265 |   5.544  |   28.18 |   4.29 | 1,706,620 | 0.3s   |          |
|     44 | BottleneckGenericAE_ld2_td10_ttd3_agF  |         33.4195 |   3.7142 |   27.74 |   4.36 | 1,704,355 | 0.3s   |          |
|     45 | BottleneckGenericAE_ld8_td5_ttd5_agF   |         33.4957 |   0.9497 |   28.64 |   4.33 | 1,709,190 | 0.3s   |          |
|     46 | BottleneckGenericAE_ld8_td10_ttd3_agF  |         33.5057 |   1.8431 |   28.78 |   4.44 | 1,719,745 | 0.3s   |          |
|     47 | GenericAE_ld2_td5_ttd5_agF             |         33.802  |   5.3209 |   29.15 |   4.52 | 1,711,395 | 0.3s   |          |
|     48 | GenericAE_ld8_td5_ttd5_ag0             |         33.8608 |   2.7063 |   29.58 |   4.67 | 1,726,785 | 0.3s   |          |
|     49 | AutoEncoder_ld16_td5_ttd5_ag0          |         33.9837 |   1.5574 |   29.38 |   4.54 | 5,126,565 | 0.5s   |          |
|     50 | BottleneckGenericAE_ld16_td5_ttd3_agF  |         34.0231 |   3.8999 |   28.58 |   4.31 | 1,727,140 | 0.3s   |          |
|     51 | AutoEncoderAE_ld16_td10_ttd3_ag0       |         34.1009 |   2.2873 |   28.87 |   4.47 | 1,826,765 | 0.3s   |          |
|     52 | AutoEncoderAE_ld16_td5_ttd3_ag0        |         34.2235 |   3.7042 |   28.25 |   4.31 | 1,788,340 | 0.3s   |          |
|     53 | GenericAEBackcastAE_ld2_td10_ttd5_agF  |         34.3666 |   6.8051 |   28.95 |   4.43 | 1,788,550 | 0.3s   |          |
|     54 | GenericAEBackcastAE_ld8_td10_ttd5_agF  |         34.3666 |   6.8051 |   28.95 |   4.43 | 1,788,550 | 0.3s   |          |
|     55 | GenericAEBackcastAE_ld16_td10_ttd5_agF |         34.3666 |   6.8051 |   28.95 |   4.43 | 1,788,550 | 0.3s   |          |
|     56 | BottleneckGenericAE_ld8_td5_ttd5_ag0   |         34.3868 |   0.4614 |   29    |   4.45 | 1,709,190 | 0.3s   |          |
|     57 | GenericAE_ld2_td5_ttd5_ag0             |         34.476  |   3.6577 |   29.91 |   4.92 | 1,711,395 | 0.3s   |          |
|     58 | BottleneckGenericAE_ld2_td5_ttd3_ag0   |         34.562  |   6.1299 |   29.63 |   4.6  | 1,691,230 | 0.3s   |          |
|     59 | AutoEncoderAE_ld2_td5_ttd5_ag0         |         34.8261 |   5.3885 |   29.62 |   4.59 | 1,755,000 | 0.3s   |          |
|     60 | BottleneckGenericAE_ld2_td10_ttd5_agF  |         34.9887 |   2.117  |   28.88 |   4.43 | 1,706,925 | 0.3s   |          |
|     61 | BottleneckGenericAE_ld8_td10_ttd3_ag0  |         35.0098 |   2.0198 |   29.08 |   4.42 | 1,719,745 | 0.3s   |          |
|     62 | BottleneckGenericAE_ld2_td5_ttd5_agF   |         35.0172 |   5.4919 |   28.74 |   4.29 | 1,693,800 | 0.3s   |          |
|     63 | BottleneckGenericAE_ld16_td5_ttd3_ag0  |         35.084  |   1.083  |   28.12 |   4.35 | 1,727,140 | 0.3s   |          |
|     64 | GenericAEBackcastAE_ld8_td10_ttd5_ag0  |         35.11   |   6.7535 |   29.78 |   4.67 | 1,788,550 | 0.3s   |          |
|     65 | GenericAEBackcastAE_ld16_td10_ttd5_ag0 |         35.11   |   6.7535 |   29.78 |   4.67 | 1,788,550 | 0.3s   |          |
|     66 | GenericAEBackcastAE_ld2_td10_ttd5_ag0  |         35.11   |   6.7535 |   29.78 |   4.67 | 1,788,550 | 0.3s   |          |
|     67 | BottleneckGenericAE_ld2_td10_ttd3_ag0  |         35.3354 |   1.8407 |   29.13 |   4.42 | 1,704,355 | 0.3s   |          |
|     68 | GenericAEBackcastAE_ld8_td5_ttd3_agF   |         35.3495 |   5.1134 |   29.27 |   4.49 | 1,747,430 | 0.3s   |          |
|     69 | GenericAEBackcastAE_ld2_td5_ttd3_agF   |         35.3495 |   5.1134 |   29.27 |   4.49 | 1,747,430 | 0.3s   |          |
|     70 | GenericAEBackcastAE_ld16_td5_ttd3_agF  |         35.3495 |   5.1134 |   29.27 |   4.49 | 1,747,430 | 0.3s   |          |
|     71 | AutoEncoderAE_ld8_td10_ttd5_ag0        |         35.3728 |   3.1516 |   29.1  |   4.54 | 1,808,815 | 0.3s   |          |
|     72 | BottleneckGenericAE_ld2_td5_ttd5_ag0   |         35.4308 |   6.7504 |   28.26 |   4.15 | 1,693,800 | 0.3s   |          |
|     73 | AutoEncoderAE_ld8_td5_ttd3_ag0         |         35.4529 |   2.81   |   28.77 |   4.52 | 1,767,820 | 0.3s   |          |
|     74 | AutoEncoderAE_ld8_td10_ttd3_ag0        |         35.4958 |   2.1173 |   32.46 |   5.29 | 1,806,245 | 0.4s   |          |
|     75 | AutoEncoderAE_ld8_td5_ttd5_ag0         |         35.5389 |   5.4896 |   31.9  |   5.05 | 1,770,390 | 0.4s   |          |
|     76 | BottleneckGenericAE_ld8_td10_ttd5_ag0  |         35.675  |   2.3349 |   29.68 |   4.62 | 1,722,315 | 0.3s   |          |
|     77 | BottleneckGenericAE_ld2_td5_ttd3_agF   |         35.7504 |   6.1444 |   29.55 |   4.57 | 1,691,230 | 0.3s   |          |
|     78 | BottleneckGenericAE_ld8_td10_ttd5_agF  |         35.869  |   5.3874 |   29.01 |   4.36 | 1,722,315 | 0.3s   |          |
|     79 | GenericAEBackcastAE_ld16_td5_ttd3_ag0  |         36.1201 |   4.1577 |   29.82 |   4.71 | 1,747,430 | 0.3s   |          |
|     80 | GenericAEBackcastAE_ld2_td5_ttd3_ag0   |         36.1201 |   4.1577 |   29.82 |   4.71 | 1,747,430 | 0.3s   |          |
|     81 | GenericAEBackcastAE_ld8_td5_ttd3_ag0   |         36.1201 |   4.1577 |   29.82 |   4.71 | 1,747,430 | 0.3s   |          |
|     82 | AutoEncoderAE_ld2_td10_ttd3_ag0        |         36.2756 |   9.7785 |   30.03 |   4.66 | 1,790,855 | 0.3s   |          |
|     83 | BottleneckGenericAE_ld2_td10_ttd5_ag0  |         36.9361 |   3.613  |   29.89 |   4.59 | 1,706,925 | 0.3s   |          |
|     84 | GenericAEBackcastAE_ld8_td5_ttd5_agF   |         36.9784 |   3.8678 |   30.42 |   4.75 | 1,750,000 | 0.3s   |          |
|     85 | GenericAEBackcastAE_ld2_td5_ttd5_agF   |         36.9784 |   3.8678 |   30.42 |   4.75 | 1,750,000 | 0.3s   |          |
|     86 | GenericAEBackcastAE_ld16_td5_ttd5_agF  |         36.9784 |   3.8678 |   30.42 |   4.75 | 1,750,000 | 0.3s   |          |
|     87 | GenericAEBackcastAE_ld16_td5_ttd5_ag0  |         37.6148 |   4.3318 |   31.58 |   4.83 | 1,750,000 | 0.3s   |          |
|     88 | GenericAEBackcastAE_ld2_td5_ttd5_ag0   |         37.6148 |   4.3318 |   31.58 |   4.83 | 1,750,000 | 0.3s   |          |
|     89 | GenericAEBackcastAE_ld8_td5_ttd5_ag0   |         37.6148 |   4.3318 |   31.58 |   4.83 | 1,750,000 | 0.3s   |          |
|     90 | AutoEncoderAE_ld2_td10_ttd5_ag0        |         37.848  |   5.228  |   30.53 |   4.58 | 1,793,425 | 0.3s   |          |

Top config median best_val_loss=29.8453; worst=37.8480; delta=8.0027.


## 2.2 Round 2 Leaderboard

39 configs x 3 runs, up to 15 epochs each

|   Rank | Config                                 |   best_val_loss |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|----------------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld2_td5_ttd5_agF     |         25.1231 |   1.1479 |   21.53 |   3.09 | 5,006,205 | 1.1s   |          |
|      2 | GenericAEBackcast_ld8_td5_ttd3_agF     |         25.1624 |   0.8815 |   21.35 |   3.12 | 5,049,895 | 1.1s   |          |
|      3 | BottleneckGenericAE_ld16_td10_ttd3_agF |         25.1768 |   3.0187 |   21.61 |   3.14 | 1,740,265 | 0.6s   |          |
|      4 | GenericAE_ld8_td5_ttd5_agF             |         25.2311 |   2.3477 |   21.8  |   3.14 | 1,726,785 | 0.6s   |          |
|      5 | GenericAEBackcast_ld16_td5_ttd5_ag0    |         25.2389 |   1.1831 |   21.67 |   3.15 | 5,114,145 | 1.1s   |          |
|      6 | GenericAEBackcast_ld16_td5_ttd3_agF    |         25.2492 |   1.2641 |   21.56 |   3.15 | 5,111,575 | 1.2s   |          |
|      7 | GenericAEBackcast_ld8_td5_ttd5_ag0     |         25.2971 |   0.8271 |   21.29 |   3.08 | 5,052,465 | 1.1s   |          |
|      8 | AutoEncoder_ld8_td5_ttd3_ag0           |         25.3525 |   2.1866 |   21.6  |   3.14 | 5,062,515 | 1.2s   |          |
|      9 | GenericAEBackcast_ld16_td5_ttd3_ag0    |         25.3785 |   0.6615 |   21.62 |   3.18 | 5,111,575 | 1.1s   |          |
|     10 | GenericAE_ld16_td5_ttd3_agF            |         25.4208 |   0.432  |   21.88 |   3.18 | 1,744,735 | 0.6s   |          |
|     11 | GenericAEBackcast_ld2_td5_ttd3_agF     |         25.4298 |   1.1657 |   21.51 |   3.1  | 5,003,635 | 1.3s   |          |
|     12 | GenericAEBackcast_ld16_td5_ttd5_agF    |         25.4322 |   0.7494 |   21.4  |   3.1  | 5,114,145 | 1.2s   |          |
|     13 | GenericAEBackcast_ld8_td5_ttd5_agF     |         25.495  |   0.599  |   21.47 |   3.11 | 5,052,465 | 1.1s   |          |
|     14 | GenericAEBackcast_ld8_td5_ttd3_ag0     |         25.6701 |   2.0834 |   21.81 |   3.2  | 5,049,895 | 0.9s   |          |
|     15 | BottleneckGenericAE_ld16_td10_ttd5_agF |         25.8199 |   2.4503 |   21.39 |   3.15 | 1,742,835 | 0.6s   |          |
|     16 | AutoEncoder_ld2_td5_ttd3_ag0           |         25.8929 |   1.2104 |   21.86 |   3.22 | 5,016,405 | 1.2s   |          |
|     17 | GenericAEBackcastAE_ld2_td10_ttd3_agF  |         25.8968 |   0.1685 |   21.94 |   3.24 | 1,785,980 | 0.7s   |          |
|     18 | GenericAEBackcastAE_ld8_td10_ttd3_agF  |         25.8968 |   0.1685 |   21.94 |   3.24 | 1,785,980 | 0.7s   |          |
|     19 | GenericAEBackcastAE_ld16_td10_ttd3_agF |         25.8968 |   0.1685 |   21.94 |   3.24 | 1,785,980 | 0.7s   |          |
|     20 | BottleneckGenericAE_ld16_td10_ttd5_ag0 |         25.9442 |   1.5871 |   21.78 |   3.16 | 1,742,835 | 0.6s   |          |
|     21 | GenericAE_ld2_td5_ttd5_agF             |         26.0494 |   2.5754 |   23.12 |   3.35 | 1,711,395 | 0.6s   |          |
|     22 | BottleneckGenericAE_ld16_td5_ttd3_agF  |         26.3449 |   0.4756 |   22.36 |   3.26 | 1,727,140 | 0.6s   |          |
|     23 | GenericAE_ld16_td5_ttd5_agF            |         26.3503 |   2.0387 |   22.21 |   3.26 | 1,747,305 | 0.6s   |          |
|     24 | GenericAE_ld16_td5_ttd3_ag0            |         26.5016 |   1.9379 |   22.67 |   3.35 | 1,744,735 | 0.6s   |          |
|     25 | AutoEncoder_ld16_td5_ttd5_ag0          |         26.5152 |   2.3613 |   22.46 |   3.3  | 5,126,565 | 1.2s   |          |
|     26 | AutoEncoderAE_ld16_td10_ttd5_ag0       |         26.5566 |   1.8469 |   23.23 |   3.33 | 1,829,335 | 0.7s   |          |
|     27 | AutoEncoderAE_ld2_td5_ttd3_ag0         |         26.563  |   3.912  |   23.44 |   3.41 | 1,752,430 | 0.8s   |          |
|     28 | GenericAEBackcastAE_ld2_td10_ttd3_ag0  |         26.6484 |   0.1839 |   22.41 |   3.28 | 1,785,980 | 0.7s   |          |
|     29 | GenericAEBackcastAE_ld8_td10_ttd3_ag0  |         26.6484 |   0.1839 |   22.41 |   3.28 | 1,785,980 | 0.7s   |          |
|     30 | GenericAEBackcastAE_ld16_td10_ttd3_ag0 |         26.6484 |   0.1839 |   22.41 |   3.28 | 1,785,980 | 0.7s   |          |
|     31 | GenericAE_ld16_td5_ttd5_ag0            |         26.6791 |   2.1779 |   23.21 |   3.38 | 1,747,305 | 0.5s   |          |
|     32 | AutoEncoder_ld16_td5_ttd3_ag0          |         26.8078 |   1.3381 |   23.06 |   3.41 | 5,123,995 | 0.9s   |          |
|     33 | BottleneckGenericAE_ld16_td10_ttd3_ag0 |         26.8364 |   1.4032 |   22.61 |   3.3  | 1,740,265 | 0.6s   |          |
|     34 | BottleneckGenericAE_ld2_td5_ttd5_agF   |         27.0156 |   0.5434 |   22.71 |   3.31 | 1,693,800 | 0.6s   |          |
|     35 | GenericAE_ld2_td5_ttd3_agF             |         27.0326 |   1.0372 |   23.36 |   3.39 | 1,708,825 | 0.6s   |          |
|     36 | AutoEncoderAE_ld16_td5_ttd5_ag0        |         27.3524 |   1.8163 |   23.37 |   3.42 | 1,790,910 | 0.6s   |          |
|     37 | AutoEncoderAE_ld2_td10_ttd3_ag0        |         27.4079 |   1.6216 |   22.9  |   3.35 | 1,790,855 | 0.6s   |          |
|     38 | GenericAE_ld8_td5_ttd3_ag0             |         27.5101 |   0.5994 |   23    |   3.4  | 1,724,215 | 0.5s   |          |
|     39 | GenericAE_ld2_td5_ttd3_ag0             |         27.8457 |   1.887  |   23.85 |   3.92 | 1,708,825 | 0.6s   |          |

Top config median best_val_loss=25.1231; worst=27.8457; delta=2.7227.


## 2.3 Round 3 Leaderboard

12 configs x 3 runs, up to 30 epochs each

|   Rank | Config                                 |   best_val_loss |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|----------------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld8_td5_ttd5_ag0     |         24.7549 |   0.1873 |   20.93 |   3.03 | 5,052,465 | 1.7s   |          |
|      2 | GenericAEBackcast_ld16_td5_ttd5_ag0    |         24.7927 |   0.5422 |   21    |   3.07 | 5,114,145 | 1.7s   |          |
|      3 | GenericAEBackcast_ld8_td5_ttd3_agF     |         24.7984 |   0.1865 |   21.1  |   3.04 | 5,049,895 | 1.5s   |          |
|      4 | GenericAEBackcast_ld16_td5_ttd3_agF    |         24.8214 |   0.538  |   21.32 |   3.08 | 5,111,575 | 1.6s   |          |
|      5 | GenericAEBackcast_ld16_td5_ttd5_agF    |         24.8266 |   0.1167 |   21.28 |   3.1  | 5,114,145 | 1.6s   |          |
|      6 | GenericAEBackcast_ld16_td5_ttd3_ag0    |         24.8545 |   0.2804 |   21.01 |   3.05 | 5,111,575 | 1.6s   |          |
|      7 | GenericAE_ld16_td5_ttd3_agF            |         24.8619 |   0.1935 |   21.31 |   3.09 | 1,744,735 | 0.9s   |          |
|      8 | GenericAEBackcast_ld2_td5_ttd3_agF     |         24.875  |   0.1399 |   21.34 |   3.08 | 5,003,635 | 1.5s   |          |
|      9 | BottleneckGenericAE_ld16_td10_ttd3_agF |         24.9263 |   0.5125 |   21.03 |   3.05 | 1,740,265 | 1.0s   |          |
|     10 | GenericAE_ld8_td5_ttd5_agF             |         25.0008 |   0.1209 |   21.41 |   3.09 | 1,726,785 | 0.9s   |          |
|     11 | AutoEncoder_ld8_td5_ttd3_ag0           |         25.0399 |   0.3625 |   21.55 |   3.14 | 5,062,515 | 1.6s   |          |
|     12 | GenericAEBackcast_ld2_td5_ttd5_agF     |         25.1231 |   0.4388 |   21.51 |   3.09 | 5,006,205 | 1.7s   |          |

Top config median best_val_loss=24.7549; worst=25.1231; delta=0.3682.


## 3. Hyperparameter Marginals (Round 1)


### AE Variant

| Value               |   Med best_val_loss |   Mean best_val_loss |     Std |   N | Med Params   |
|:--------------------|--------------------:|---------------------:|--------:|----:|:-------------|
| GenericAEBackcast   |             31.0926 |              31.0858 | 1.79452 |  36 | 5,051,180    |
| GenericAE           |             32.5985 |              32.6664 | 1.86855 |  36 | 1,725,500    |
| AutoEncoder         |             32.6636 |              32.624  | 1.86639 |  18 | 5,063,800    |
| GenericAEBackcastAE |             33.9657 |              34.6092 | 2.66279 |  72 | 1,767,990    |
| BottleneckGenericAE |             34.4805 |              34.2883 | 2.3847  |  72 | 1,714,468    |
| AutoEncoderAE       |             35.3516 |              35.0985 | 2.8267  |  36 | 1,790,882    |


### Latent Dim

|   Value |   Med best_val_loss |   Mean best_val_loss |     Std |   N | Med Params   |
|--------:|--------------------:|---------------------:|--------:|----:|:-------------|
|      16 |             32.6272 |              33.2357 | 2.71898 |  90 | 1,750,000    |
|       8 |             33.5144 |              33.77   | 2.52265 |  90 | 1,750,000    |
|       2 |             33.7167 |              34.1774 | 2.75694 |  90 | 1,750,000    |


### Thetas Dim

|   Value |   Med best_val_loss |   Mean best_val_loss |     Std |   N | Med Params   |
|--------:|--------------------:|---------------------:|--------:|----:|:-------------|
|       5 |             33.1177 |              33.4658 | 2.63122 | 180 | 1,748,715    |
|      10 |             33.9843 |              34.2515 | 2.73276 |  90 | 1,785,980    |


### active_g

| Value    |   Med best_val_loss |   Mean best_val_loss |     Std |   N | Med Params   |
|:---------|--------------------:|---------------------:|--------:|----:|:-------------|
| forecast |             32.6583 |              33.1877 | 2.63966 | 108 | 1,747,368    |
| False    |             33.6628 |              34.0877 | 2.66437 | 162 | 1,769,105    |


## 3b. Latent Dimension Discussion

|   latent_dim_cfg |   med_metric |   std_metric |   n |   med_params |
|-----------------:|-------------:|-------------:|----:|-------------:|
|               16 |      32.6272 |      2.71898 |  90 |     1.75e+06 |
|                8 |      33.5144 |      2.52265 |  90 |     1.75e+06 |
|                2 |      33.7167 |      2.75694 |  90 |     1.75e+06 |

Best latent_dim=16 by median best_val_loss (32.6272).


## 4. Variant Head-to-Head


### Round 1 - Best Config per Variant

| Variant             | Best Config                            |   Med best_val_loss |
|:--------------------|:---------------------------------------|--------------------:|
| GenericAEBackcast   | GenericAEBackcast_ld8_td5_ttd5_agF     |             29.8453 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd5_agF |             30.7287 |
| GenericAE           | GenericAE_ld16_td5_ttd3_agF            |             30.7795 |
| AutoEncoderAE       | AutoEncoderAE_ld2_td5_ttd3_ag0         |             30.892  |
| AutoEncoder         | AutoEncoder_ld2_td5_ttd3_ag0           |             30.8967 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld16_td10_ttd3_ag0 |             31.612  |

### Round 2 - Best Config per Variant

| Variant             | Best Config                            |   Med best_val_loss |
|:--------------------|:---------------------------------------|--------------------:|
| GenericAEBackcast   | GenericAEBackcast_ld2_td5_ttd5_agF     |             25.1231 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd3_agF |             25.1768 |
| GenericAE           | GenericAE_ld8_td5_ttd5_agF             |             25.2311 |
| AutoEncoder         | AutoEncoder_ld8_td5_ttd3_ag0           |             25.3525 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld2_td10_ttd3_agF  |             25.8968 |
| AutoEncoderAE       | AutoEncoderAE_ld16_td10_ttd5_ag0       |             26.5566 |

### Round 3 - Best Config per Variant

| Variant             | Best Config                            |   Med best_val_loss |
|:--------------------|:---------------------------------------|--------------------:|
| GenericAEBackcast   | GenericAEBackcast_ld8_td5_ttd5_ag0     |             24.7549 |
| GenericAE           | GenericAE_ld16_td5_ttd3_agF            |             24.8619 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd3_agF |             24.9263 |
| AutoEncoder         | AutoEncoder_ld8_td5_ttd3_ag0           |             25.0399 |


## 5. Stability Analysis


### Round 1

- Mean spread: 3.8452
- Max spread:  9.7785 (AutoEncoderAE_ld2_td10_ttd3_ag0)
- Mean std:    2.0412

### Round 2

- Mean spread: 1.3442
- Max spread:  3.9120 (AutoEncoderAE_ld2_td5_ttd3_ag0)
- Mean std:    0.7101

### Round 3

- Mean spread: 0.3016
- Max spread:  0.5422 (GenericAEBackcast_ld16_td5_ttd5_ag0)
- Mean std:    0.1598


## 6. Round-over-Round Progression

| config_name                            |      R1 |      R2 |      R3 |   Delta |   DeltaPct |
|:---------------------------------------|--------:|--------:|--------:|--------:|-----------:|
| GenericAEBackcast_ld16_td5_ttd5_ag0    | 32.6313 | 25.2389 | 24.7927 | -7.8386 |      -24   |
| GenericAE_ld8_td5_ttd5_agF             | 32.1183 | 25.2311 | 25.0008 | -7.1175 |      -22.2 |
| GenericAEBackcast_ld16_td5_ttd3_ag0    | 31.3069 | 25.3785 | 24.8545 | -6.4524 |      -20.6 |
| GenericAEBackcast_ld16_td5_ttd3_agF    | 31.0889 | 25.2492 | 24.8214 | -6.2675 |      -20.2 |
| GenericAEBackcast_ld8_td5_ttd5_ag0     | 30.9164 | 25.2971 | 24.7549 | -6.1615 |      -19.9 |
| GenericAEBackcast_ld16_td5_ttd5_agF    | 30.9787 | 25.4322 | 24.8266 | -6.1521 |      -19.9 |
| GenericAEBackcast_ld2_td5_ttd5_agF     | 31.1192 | 25.1231 | 25.1231 | -5.9961 |      -19.3 |
| BottleneckGenericAE_ld16_td10_ttd3_agF | 30.8546 | 25.1768 | 24.9263 | -5.9283 |      -19.2 |
| GenericAE_ld16_td5_ttd3_agF            | 30.7795 | 25.4208 | 24.8619 | -5.9176 |      -19.2 |
| AutoEncoder_ld8_td5_ttd3_ag0           | 30.9721 | 25.3525 | 25.0399 | -5.9323 |      -19.2 |
| GenericAEBackcast_ld8_td5_ttd3_agF     | 30.3069 | 25.1624 | 24.7984 | -5.5085 |      -18.2 |
| GenericAEBackcast_ld2_td5_ttd3_agF     | 30.044  | 25.4298 | 24.875  | -5.169  |      -17.2 |


## 7. Parameter Efficiency

| Config                                 | Params    | Reduction   |   Med best_val_loss |
|:---------------------------------------|:----------|:------------|--------------------:|
| GenericAE_ld8_td5_ttd5_agF             | 1,726,785 | 93.0%       |             25.0008 |
| BottleneckGenericAE_ld16_td10_ttd3_agF | 1,740,265 | 93.0%       |             24.9263 |
| GenericAE_ld16_td5_ttd3_agF            | 1,744,735 | 92.9%       |             24.8619 |
| GenericAEBackcast_ld2_td5_ttd3_agF     | 5,003,635 | 79.7%       |             24.875  |
| GenericAEBackcast_ld2_td5_ttd5_agF     | 5,006,205 | 79.7%       |             25.1231 |
| GenericAEBackcast_ld8_td5_ttd3_agF     | 5,049,895 | 79.6%       |             24.7984 |
| GenericAEBackcast_ld8_td5_ttd5_ag0     | 5,052,465 | 79.5%       |             24.7549 |
| AutoEncoder_ld8_td5_ttd3_ag0           | 5,062,515 | 79.5%       |             25.0399 |
| GenericAEBackcast_ld16_td5_ttd3_ag0    | 5,111,575 | 79.3%       |             24.8545 |
| GenericAEBackcast_ld16_td5_ttd3_agF    | 5,111,575 | 79.3%       |             24.8214 |
| GenericAEBackcast_ld16_td5_ttd5_ag0    | 5,114,145 | 79.3%       |             24.7927 |
| GenericAEBackcast_ld16_td5_ttd5_agF    | 5,114,145 | 79.3%       |             24.8266 |


## 8. Final Verdict

Primary metric: median best_val_loss (lower is better).
OWA target checks are skipped because OWA is unavailable for this dataset.
Best final-round config: GenericAEBackcast_ld8_td5_ttd5_ag0 with median best_val_loss=24.7549.


# Summary

- analyzed_count: 2
- skipped_count: 0
- analyzed: ['m4', 'tourism']
