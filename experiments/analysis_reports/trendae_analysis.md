# TrendAE Architecture Search — Results Analysis

## Abstract

This study evaluates **6 AE-backbone variants** paired with Trend blocks across 90 configurations on M4-Yearly, using successive halving over 3 rounds (384 total runs, 55.6 min compute). Each configuration combines an AE block (AutoEncoder, GenericAE, GenericAEBackcast, BottleneckGenericAE, and their AE-backbone counterparts) with a Trend block, varying latent_dim, thetas_dim, trend_thetas_dim, and active_g.

**Key Takeaways:**

1. **Best configuration:** `GenericAE_ld16_td5_ttd3_ag0` — median OWA = **0.8072** with 1,042,095 params (96% fewer than NBEATS-G).
2. **Target OWA < 0.85:** Met ✓
3. **Best AE variant (R1 marginal):** `AutoEncoder`
4. **Search scope:** 90 → 9 configs via successive halving.
5. **Convergence:** 0 diverged runs out of 384 (0.0%).

- **CSV:** `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/trendae_study_results.csv`
- **Rows:** 384 (90 unique configs, 3 rounds)
- **AE variants:** ['AutoEncoder', 'AutoEncoderAE', 'BottleneckGenericAE', 'GenericAE', 'GenericAEBackcast', 'GenericAEBackcastAE']
- **Total training time:** 55.6 min
- **OWA range:** 0.7960 – 10.4784


## 1. Successive Halving Funnel

|   Round |   Configs |   Runs |   Epochs |   Best Med OWA |
|--------:|----------:|-------:|---------:|---------------:|
|       1 |        90 |    270 |       10 |         0.8679 |
|       2 |        29 |     87 |       15 |         0.827  |
|       3 |         9 |     27 |       30 |         0.8072 |

The successive halving procedure pruned from 90 to 9 configurations across 3 rounds, retaining the top 10% of candidates.


## 2.1 Round 1 Leaderboard

90 configs × 3 runs, up to 10 epochs each

|   Rank | Config                                 |    OWA |      ± |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|-------:|-------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld8_td5_ttd3_ag0     | 0.8679 | 0.1514 |   14.49 |   3.37 | 4,355,065 | 7.7s   |          |
|      2 | GenericAEBackcast_ld16_td5_ttd3_ag0    | 0.8714 | 0.067  |   14.66 |   3.36 | 4,416,825 | 7.6s   |          |
|      3 | AutoEncoder_ld16_td5_ttd5_ag0          | 0.8885 | 0.013  |   14.92 |   3.44 | 4,436,785 | 7.3s   |          |
|      4 | AutoEncoder_ld16_td5_ttd3_ag0          | 0.8894 | 0.0731 |   15.07 |   3.4  | 4,434,215 | 7.7s   |          |
|      5 | GenericAEBackcast_ld16_td5_ttd5_agF    | 0.8931 | 0.1304 |   14.83 |   3.49 | 4,419,395 | 7.6s   |          |
|      6 | GenericAEBackcast_ld16_td5_ttd3_agF    | 0.896  | 0.0363 |   14.88 |   3.5  | 4,416,825 | 7.6s   |          |
|      7 | BottleneckGenericAE_ld16_td10_ttd3_ag0 | 0.9002 | 0.0182 |   15.26 |   3.44 | 977,385   | 7.1s   |          |
|      8 | GenericAEBackcast_ld2_td5_ttd3_agF     | 0.9042 | 0.2373 |   15.07 |   3.52 | 4,308,745 | 7.6s   |          |
|      9 | GenericAE_ld16_td5_ttd5_ag0            | 0.9096 | 0.1321 |   15.13 |   3.55 | 1,044,665 | 7.0s   |          |
|     10 | AutoEncoder_ld8_td5_ttd5_ag0           | 0.9104 | 0.0649 |   15.49 |   3.47 | 4,375,305 | 6.7s   |          |
|     11 | GenericAE_ld16_td5_ttd3_agF            | 0.9144 | 0.0122 |   15.23 |   3.56 | 1,042,095 | 7.0s   |          |
|     12 | AutoEncoder_ld8_td5_ttd3_ag0           | 0.917  | 0.0498 |   15.18 |   3.6  | 4,372,735 | 6.8s   |          |
|     13 | GenericAEBackcast_ld2_td5_ttd5_ag0     | 0.9185 | 0.3536 |   15.23 |   3.6  | 4,311,315 | 7.6s   |          |
|     14 | GenericAEBackcastAE_ld8_td10_ttd3_ag0  | 0.9188 | 0.0455 |   15.53 |   3.53 | 1,068,150 | 7.2s   |          |
|     15 | BottleneckGenericAE_ld16_td5_ttd3_ag0  | 0.9208 | 9.5668 |   15.52 |   3.54 | 963,660   | 7.1s   |          |
|     16 | GenericAEBackcast_ld8_td5_ttd5_agF     | 0.9213 | 0.0441 |   15.29 |   3.6  | 4,357,635 | 7.6s   |          |
|     17 | GenericAEBackcast_ld2_td5_ttd5_agF     | 0.9244 | 0.0326 |   15.33 |   3.62 | 4,311,315 | 7.7s   |          |
|     18 | GenericAE_ld8_td5_ttd3_ag0             | 0.9315 | 0.0711 |   15.42 |   3.65 | 1,011,295 | 6.9s   |          |
|     19 | GenericAEBackcastAE_ld16_td10_ttd3_ag0 | 0.9346 | 0.0206 |   15.57 |   3.61 | 1,078,430 | 7.2s   |          |
|     20 | AutoEncoderAE_ld8_td10_ttd3_ag0        | 0.9366 | 0.052  |   15.91 |   3.61 | 1,093,445 | 7.4s   |          |
|     21 | GenericAEBackcast_ld8_td5_ttd5_ag0     | 0.9378 | 0.1157 |   15.37 |   3.72 | 4,357,635 | 7.6s   |          |
|     22 | AutoEncoder_ld2_td5_ttd3_ag0           | 0.9391 | 0.0311 |   15.77 |   3.63 | 4,326,625 | 6.6s   |          |
|     23 | GenericAE_ld16_td5_ttd3_ag0            | 0.9407 | 0.1524 |   15.49 |   3.71 | 1,042,095 | 6.9s   |          |
|     24 | GenericAE_ld2_td5_ttd3_agF             | 0.9415 | 0.0116 |   15.81 |   3.63 | 988,195   | 6.8s   |          |
|     25 | GenericAEBackcast_ld16_td5_ttd5_ag0    | 0.9432 | 0.0768 |   15.47 |   3.72 | 4,419,395 | 7.5s   |          |
|     26 | BottleneckGenericAE_ld16_td5_ttd3_agF  | 0.9436 | 0.1389 |   15.85 |   3.65 | 963,660   | 7.1s   |          |
|     27 | AutoEncoder_ld2_td5_ttd5_ag0           | 0.9483 | 0.0854 |   15.87 |   3.68 | 4,329,195 | 6.8s   |          |
|     28 | GenericAEBackcastAE_ld16_td5_ttd3_ag0  | 0.9512 | 0.1068 |   15.98 |   3.71 | 1,039,830 | 7.2s   |          |
|     29 | GenericAEBackcastAE_ld8_td10_ttd3_agF  | 0.9513 | 0.0551 |   15.94 |   3.71 | 1,068,150 | 7.2s   |          |
|     30 | GenericAE_ld8_td5_ttd3_agF             | 0.9515 | 0.0635 |   15.75 |   3.73 | 1,011,295 | 7.1s   |          |
|     31 | GenericAEBackcastAE_ld16_td10_ttd3_agF | 0.9532 | 0.0545 |   15.96 |   3.7  | 1,078,430 | 7.2s   |          |
|     32 | BottleneckGenericAE_ld16_td10_ttd3_agF | 0.9568 | 0.0412 |   16.07 |   3.66 | 977,385   | 7.1s   |          |
|     33 | GenericAE_ld2_td5_ttd5_ag0             | 0.9669 | 0.1682 |   16.25 |   3.73 | 990,765   | 6.8s   |          |
|     34 | BottleneckGenericAE_ld8_td10_ttd5_agF  | 0.9716 | 0.1979 |   16.24 |   3.77 | 949,155   | 7.1s   |          |
|     35 | GenericAEBackcast_ld2_td5_ttd3_ag0     | 0.9722 | 0.1235 |   16.16 |   3.8  | 4,308,745 | 7.8s   |          |
|     36 | GenericAEBackcastAE_ld16_td10_ttd5_agF | 0.9741 | 0.0806 |   16.36 |   3.76 | 1,081,000 | 7.2s   |          |
|     37 | BottleneckGenericAE_ld2_td10_ttd3_agF  | 0.9747 | 0.103  |   16.45 |   3.75 | 923,485   | 6.9s   |          |
|     38 | AutoEncoderAE_ld16_td5_ttd5_ag0        | 0.9748 | 0.1002 |   16.06 |   3.84 | 1,088,390 | 7.2s   |          |
|     39 | GenericAE_ld8_td5_ttd5_ag0             | 0.9811 | 0.1697 |   16.04 |   3.9  | 1,013,865 | 7.2s   |          |
|     40 | BottleneckGenericAE_ld16_td5_ttd5_agF  | 0.9824 | 0.0777 |   16.36 |   3.87 | 966,230   | 7.1s   |          |
|     41 | GenericAE_ld16_td5_ttd5_agF            | 0.9837 | 0.0621 |   16.18 |   3.85 | 1,044,665 | 7.0s   |          |
|     42 | BottleneckGenericAE_ld16_td10_ttd5_agF | 0.9841 | 0.0965 |   16.34 |   3.85 | 979,955   | 7.1s   |          |
|     43 | BottleneckGenericAE_ld2_td5_ttd3_ag0   | 0.9848 | 0.144  |   16.51 |   3.81 | 909,760   | 7.1s   |          |
|     44 | BottleneckGenericAE_ld2_td5_ttd3_agF   | 0.9849 | 0.0244 |   16.55 |   3.85 | 909,760   | 7.1s   |          |
|     45 | AutoEncoderAE_ld8_td10_ttd5_ag0        | 0.9866 | 0.2356 |   16.59 |   3.81 | 1,096,015 | 7.2s   |          |
|     46 | GenericAE_ld2_td5_ttd3_ag0             | 0.9888 | 0.2933 |   16.57 |   3.83 | 988,195   | 6.8s   |          |
|     47 | GenericAEBackcastAE_ld8_td5_ttd5_ag0   | 0.9908 | 0.0924 |   16.55 |   3.86 | 1,032,120 | 7.1s   |          |
|     48 | GenericAEBackcastAE_ld16_td5_ttd3_agF  | 0.9916 | 0.1707 |   16.53 |   3.86 | 1,039,830 | 7.2s   |          |
|     49 | GenericAE_ld2_td5_ttd5_agF             | 0.9941 | 0.1704 |   16.4  |   3.91 | 990,765   | 6.9s   |          |
|     50 | GenericAEBackcastAE_ld2_td5_ttd5_ag0   | 0.9954 | 0.2073 |   16.64 |   3.87 | 1,024,410 | 7.2s   |          |
|     51 | GenericAEBackcastAE_ld16_td5_ttd5_agF  | 0.9966 | 0.1102 |   16.65 |   3.87 | 1,042,400 | 7.2s   |          |
|     52 | GenericAEBackcastAE_ld16_td5_ttd5_ag0  | 0.9966 | 0.0856 |   16.72 |   3.85 | 1,042,400 | 7.2s   |          |
|     53 | GenericAEBackcastAE_ld8_td10_ttd5_ag0  | 1.0001 | 0.1087 |   16.43 |   3.95 | 1,070,720 | 7.2s   |          |
|     54 | GenericAEBackcastAE_ld2_td5_ttd3_ag0   | 1.0018 | 0.0893 |   16.67 |   3.91 | 1,021,840 | 7.2s   |          |
|     55 | AutoEncoderAE_ld2_td10_ttd3_ag0        | 1.0033 | 0.1238 |   16.85 |   3.88 | 1,070,345 | 7.1s   |          |
|     56 | AutoEncoderAE_ld2_td5_ttd3_ag0         | 1.0037 | 0.0565 |   16.78 |   3.9  | 1,031,920 | 7.2s   |          |
|     57 | AutoEncoderAE_ld16_td10_ttd3_ag0       | 1.0075 | 0.1024 |   16.57 |   3.98 | 1,124,245 | 7.2s   |          |
|     58 | BottleneckGenericAE_ld16_td10_ttd5_ag0 | 1.0089 | 0.0733 |   16.66 |   3.97 | 979,955   | 7.1s   |          |
|     59 | BottleneckGenericAE_ld8_td10_ttd5_ag0  | 1.0117 | 0.2062 |   16.53 |   4.02 | 949,155   | 7.1s   |          |
|     60 | BottleneckGenericAE_ld8_td5_ttd5_agF   | 1.0117 | 0.1698 |   16.96 |   3.92 | 935,430   | 7.1s   |          |
|     61 | BottleneckGenericAE_ld8_td10_ttd3_agF  | 1.0127 | 0.0743 |   16.5  |   4.02 | 946,585   | 7.1s   |          |
|     62 | BottleneckGenericAE_ld8_td5_ttd3_ag0   | 1.0132 | 0.0867 |   16.74 |   3.98 | 932,860   | 7.2s   |          |
|     63 | GenericAE_ld8_td5_ttd5_agF             | 1.0146 | 0.1417 |   16.65 |   4.02 | 1,013,865 | 7.1s   |          |
|     64 | GenericAEBackcast_ld8_td5_ttd3_agF     | 1.0165 | 0.1477 |   16.37 |   4.1  | 4,355,065 | 7.6s   |          |
|     65 | BottleneckGenericAE_ld2_td10_ttd3_ag0  | 1.0186 | 0.0725 |   16.91 |   3.98 | 923,485   | 6.9s   |          |
|     66 | AutoEncoderAE_ld8_td5_ttd5_ag0         | 1.0193 | 0.1097 |   16.84 |   3.99 | 1,057,590 | 7.2s   |          |
|     67 | BottleneckGenericAE_ld8_td5_ttd5_ag0   | 1.0214 | 0.1816 |   16.78 |   4.04 | 935,430   | 7.1s   |          |
|     68 | GenericAEBackcastAE_ld2_td10_ttd5_agF  | 1.0238 | 0.2329 |   17.1  |   3.98 | 1,063,010 | 7.1s   |          |
|     69 | BottleneckGenericAE_ld16_td5_ttd5_ag0  | 1.0253 | 0.2041 |   17.09 |   3.99 | 966,230   | 7.1s   |          |
|     70 | AutoEncoderAE_ld16_td10_ttd5_ag0       | 1.0263 | 0.2698 |   16.92 |   4.04 | 1,126,815 | 7.2s   |          |
|     71 | GenericAEBackcastAE_ld8_td5_ttd3_agF   | 1.0321 | 0.1139 |   17.07 |   4.05 | 1,029,550 | 7.1s   |          |
|     72 | BottleneckGenericAE_ld8_td5_ttd3_agF   | 1.0347 | 0.0657 |   16.96 |   4.1  | 932,860   | 7.1s   |          |
|     73 | GenericAEBackcastAE_ld2_td10_ttd3_ag0  | 1.0416 | 0.0443 |   17.15 |   4.11 | 1,060,440 | 7.1s   |          |
|     74 | AutoEncoderAE_ld8_td5_ttd3_ag0         | 1.0484 | 0.2733 |   17.23 |   4.14 | 1,055,020 | 7.2s   |          |
|     75 | AutoEncoderAE_ld2_td5_ttd5_ag0         | 1.0551 | 0.2276 |   17.38 |   4.16 | 1,034,490 | 7.1s   |          |
|     76 | GenericAEBackcastAE_ld8_td5_ttd3_ag0   | 1.0553 | 0.129  |   17.25 |   4.19 | 1,029,550 | 7.1s   |          |
|     77 | BottleneckGenericAE_ld8_td10_ttd3_ag0  | 1.0558 | 0.2289 |   17.44 |   4.15 | 946,585   | 7.1s   |          |
|     78 | GenericAEBackcastAE_ld2_td5_ttd3_agF   | 1.058  | 0.1982 |   17.39 |   4.18 | 1,021,840 | 7.2s   |          |
|     79 | BottleneckGenericAE_ld2_td5_ttd5_agF   | 1.0601 | 0.3406 |   17.45 |   4.18 | 912,330   | 6.9s   |          |
|     80 | BottleneckGenericAE_ld2_td10_ttd5_agF  | 1.0606 | 0.3599 |   17.53 |   4.17 | 926,055   | 6.9s   |          |
|     81 | BottleneckGenericAE_ld2_td10_ttd5_ag0  | 1.0643 | 0.3314 |   17.67 |   4.16 | 926,055   | 6.9s   |          |
|     82 | GenericAEBackcastAE_ld8_td5_ttd5_agF   | 1.0683 | 0.1398 |   17.51 |   4.23 | 1,032,120 | 7.1s   |          |
|     83 | GenericAEBackcastAE_ld2_td10_ttd5_ag0  | 1.0736 | 0.4235 |   17.68 |   4.23 | 1,063,010 | 7.2s   |          |
|     84 | GenericAEBackcastAE_ld2_td5_ttd5_agF   | 1.0878 | 0.1499 |   17.57 |   4.37 | 1,024,410 | 7.2s   |          |
|     85 | GenericAEBackcastAE_ld16_td10_ttd5_ag0 | 1.1029 | 0.1415 |   17.59 |   4.49 | 1,081,000 | 7.2s   |          |
|     86 | GenericAEBackcastAE_ld8_td10_ttd5_agF  | 1.1039 | 0.1213 |   17.79 |   4.45 | 1,070,720 | 7.2s   |          |
|     87 | BottleneckGenericAE_ld2_td5_ttd5_ag0   | 1.104  | 0.1697 |   18.15 |   4.36 | 912,330   | 7.0s   |          |
|     88 | GenericAEBackcastAE_ld2_td10_ttd3_agF  | 1.1059 | 0.1159 |   17.85 |   4.45 | 1,060,440 | 7.1s   |          |
|     89 | AutoEncoderAE_ld16_td5_ttd3_ag0        | 1.1232 | 0.2348 |   17.81 |   4.6  | 1,085,820 | 7.2s   |          |
|     90 | AutoEncoderAE_ld2_td10_ttd5_ag0        | 1.1241 | 0.3022 |   18.47 |   4.44 | 1,072,915 | 7.2s   |          |

The top-ranked configuration achieves a median OWA of 0.8679 with 4,355,065 parameters, while the worst scores 1.1241. The spread between best and worst is 0.2562.


## 2.2 Round 2 Leaderboard

29 configs × 3 runs, up to 15 epochs each

|   Rank | Config                                 |    OWA |      ± |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|-------:|-------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld16_td5_ttd5_ag0    | 0.827  | 0.075  |   13.85 |   3.21 | 4,419,395 | 10.7s  | ✓        |
|      2 | GenericAEBackcast_ld2_td5_ttd5_ag0     | 0.8433 | 0.0113 |   14.12 |   3.24 | 4,311,315 | 10.8s  | ✓        |
|      3 | BottleneckGenericAE_ld16_td10_ttd3_ag0 | 0.8476 | 0.0512 |   14.28 |   3.26 | 977,385   | 10.0s  | ✓        |
|      4 | GenericAE_ld16_td5_ttd3_agF            | 0.8504 | 0.0051 |   14.26 |   3.3  | 1,042,095 | 10.1s  |          |
|      5 | AutoEncoder_ld2_td5_ttd5_ag0           | 0.8552 | 0.0454 |   14.39 |   3.3  | 4,329,195 | 10.3s  |          |
|      6 | GenericAEBackcast_ld8_td5_ttd5_ag0     | 0.8556 | 0.1418 |   14.36 |   3.31 | 4,357,635 | 10.7s  |          |
|      7 | GenericAEBackcast_ld16_td5_ttd5_agF    | 0.8557 | 0.0441 |   14.3  |   3.32 | 4,419,395 | 10.8s  |          |
|      8 | GenericAEBackcast_ld2_td5_ttd5_agF     | 0.8568 | 0.044  |   14.28 |   3.34 | 4,311,315 | 10.7s  |          |
|      9 | GenericAEBackcast_ld8_td5_ttd3_ag0     | 0.8577 | 0.0149 |   14.34 |   3.33 | 4,355,065 | 10.6s  |          |
|     10 | AutoEncoder_ld8_td5_ttd3_ag0           | 0.8584 | 0.1046 |   14.26 |   3.36 | 4,372,735 | 10.6s  |          |
|     11 | AutoEncoder_ld16_td5_ttd5_ag0          | 0.8617 | 0.113  |   14.33 |   3.37 | 4,436,785 | 10.6s  |          |
|     12 | GenericAE_ld8_td5_ttd3_agF             | 0.863  | 0.0804 |   14.37 |   3.36 | 1,011,295 | 10.0s  |          |
|     13 | GenericAEBackcast_ld16_td5_ttd3_ag0    | 0.8635 | 0.0539 |   14.3  |   3.39 | 4,416,825 | 10.7s  |          |
|     14 | GenericAEBackcast_ld16_td5_ttd3_agF    | 0.8702 | 0.026  |   14.33 |   3.43 | 4,416,825 | 10.8s  |          |
|     15 | AutoEncoder_ld2_td5_ttd3_ag0           | 0.8703 | 0.121  |   14.55 |   3.38 | 4,326,625 | 10.4s  |          |
|     16 | GenericAEBackcastAE_ld16_td10_ttd3_agF | 0.8718 | 0.0507 |   14.51 |   3.4  | 1,078,430 | 10.2s  |          |
|     17 | GenericAE_ld16_td5_ttd3_ag0            | 0.873  | 0.1175 |   14.49 |   3.42 | 1,042,095 | 9.9s   |          |
|     18 | GenericAEBackcast_ld2_td5_ttd3_ag0     | 0.8794 | 0.0707 |   14.7  |   3.42 | 4,308,745 | 10.8s  |          |
|     19 | BottleneckGenericAE_ld16_td5_ttd3_ag0  | 0.8802 | 9.6081 |   14.61 |   3.44 | 963,660   | 10.0s  |          |
|     20 | GenericAEBackcast_ld8_td5_ttd5_agF     | 0.881  | 0.0264 |   14.69 |   3.43 | 4,357,635 | 10.7s  |          |
|     21 | BottleneckGenericAE_ld16_td5_ttd3_agF  | 0.8913 | 0.0937 |   14.84 |   3.47 | 963,660   | 10.0s  |          |
|     22 | GenericAE_ld2_td5_ttd5_agF             | 0.8943 | 0.0265 |   14.99 |   3.43 | 990,765   | 9.9s   |          |
|     23 | GenericAEBackcast_ld2_td5_ttd3_agF     | 0.9009 | 0.0246 |   14.96 |   3.52 | 4,308,745 | 10.7s  |          |
|     24 | AutoEncoder_ld16_td5_ttd3_ag0          | 0.903  | 0.0996 |   14.9  |   3.55 | 4,434,215 | 10.6s  |          |
|     25 | GenericAEBackcastAE_ld8_td10_ttd3_agF  | 0.9042 | 0.0706 |   14.9  |   3.56 | 1,068,150 | 10.2s  |          |
|     26 | GenericAE_ld8_td5_ttd3_ag0             | 0.9065 | 0.0467 |   14.81 |   3.6  | 1,011,295 | 9.9s   |          |
|     27 | GenericAE_ld8_td5_ttd5_ag0             | 0.9194 | 0.0502 |   15.18 |   3.62 | 1,013,865 | 10.0s  |          |
|     28 | GenericAEBackcast_ld8_td5_ttd3_agF     | 0.9458 | 0.092  |   15.41 |   3.77 | 4,355,065 | 10.6s  |          |
|     29 | BottleneckGenericAE_ld16_td10_ttd3_agF | 0.948  | 0.0749 |   15.49 |   3.77 | 977,385   | 10.0s  |          |

The top-ranked configuration achieves a median OWA of 0.8270 with 4,419,395 parameters, while the worst scores 0.9480. The spread between best and worst is 0.1209.
**3 configuration(s) meet both the OWA < 0.85 and params < 5,000,000 targets.**


## 2.3 Round 3 Leaderboard

9 configs × 3 runs, up to 30 epochs each

|   Rank | Config                                 |    OWA |      ± |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|-------:|-------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAE_ld16_td5_ttd3_ag0            | 0.8072 | 0.0136 |   13.62 |   3.11 | 1,042,095 | 17.8s  | ✓        |
|      2 | GenericAEBackcast_ld8_td5_ttd3_ag0     | 0.8075 | 0.035  |   13.59 |   3.11 | 4,355,065 | 19.5s  | ✓        |
|      3 | GenericAEBackcast_ld16_td5_ttd3_ag0    | 0.8081 | 0.029  |   13.57 |   3.12 | 4,416,825 | 19.1s  | ✓        |
|      4 | GenericAE_ld16_td5_ttd3_agF            | 0.8099 | 0.0149 |   13.63 |   3.12 | 1,042,095 | 18.0s  | ✓        |
|      5 | GenericAEBackcast_ld16_td5_ttd5_agF    | 0.8112 | 0.0078 |   13.63 |   3.13 | 4,419,395 | 19.0s  | ✓        |
|      6 | GenericAEBackcast_ld16_td5_ttd3_agF    | 0.8114 | 0.0241 |   13.63 |   3.13 | 4,416,825 | 18.9s  | ✓        |
|      7 | GenericAEBackcast_ld16_td5_ttd5_ag0    | 0.8122 | 0.0927 |   13.63 |   3.14 | 4,419,395 | 19.2s  | ✓        |
|      8 | GenericAEBackcast_ld2_td5_ttd5_ag0     | 0.8134 | 0.0174 |   13.68 |   3.14 | 4,311,315 | 18.8s  | ✓        |
|      9 | GenericAEBackcastAE_ld16_td10_ttd3_agF | 0.8209 | 0.0109 |   13.75 |   3.18 | 1,078,430 | 17.9s  | ✓        |

The top-ranked configuration achieves a median OWA of 0.8072 with 1,042,095 parameters, while the worst scores 0.8209. The spread between best and worst is 0.0136.
**9 configuration(s) meet both the OWA < 0.85 and params < 5,000,000 targets.**


## 3. Hyperparameter Marginals (Round 1 — Full Grid)


### AE Variant

| Value               |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|:--------------------|----------:|-----------:|----------:|----:|:-------------|
| AutoEncoder         |  0.913685 |   0.912922 | 0.0304832 |  18 | 4,374,020    |
| GenericAEBackcast   |  0.919228 |   0.935336 | 0.0808918 |  36 | 4,356,350    |
| GenericAE           |  0.942988 |   0.96995  | 0.0701907 |  36 | 1,012,580    |
| GenericAEBackcastAE |  1.00462  |   1.02708  | 0.0850467 |  72 | 1,051,420    |
| BottleneckGenericAE |  1.00816  |   1.15546  | 1.11832   |  72 | 941,008      |
| AutoEncoderAE       |  1.01139  |   1.03104  | 0.099816  |  36 | 1,079,368    |

**AutoEncoder** yields the best median OWA while **AutoEncoderAE** is the weakest (Δ = 0.0977).


### Latent Dim

|   Value |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|--------:|----------:|-----------:|----------:|----:|:-------------|
|      16 |  0.951584 |   1.07286  | 1.00534   |  90 | 1,044,665    |
|       8 |  0.987947 |   0.996818 | 0.0794262 |  90 | 1,032,120    |
|       2 |  1.0026   |   1.03346  | 0.112582  |  90 | 1,024,410    |

**16** yields the best median OWA while **2** is the weakest (Δ = 0.0510).


### Thetas Dim

|   Value |   Med OWA |   Mean OWA |      Std |   N | Med Params   |
|--------:|----------:|-----------:|---------:|----:|:-------------|
|       5 |  0.977696 |    1.04066 | 0.712939 | 180 | 1,037,160    |
|      10 |  1.00169  |    1.02182 | 0.1005   |  90 | 1,063,010    |

**5** yields the best median OWA while **10** is the weakest (Δ = 0.0240).


### Trend Thetas Dim

|   Value |   Med OWA |   Mean OWA |      Std |   N | Med Params   |
|--------:|----------:|-----------:|---------:|----:|:-------------|
|       3 |  0.968939 |    1.05113 | 0.820959 | 135 | 1,039,830    |
|       5 |  0.9918   |    1.01763 | 0.106308 | 135 | 1,042,400    |

**3** yields the best median OWA while **5** is the weakest (Δ = 0.0229).


### active_g

| Value    |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|:---------|----------:|-----------:|----------:|----:|:-------------|
| False    |  0.982117 |    1.05715 | 0.751337  | 162 | 1,056,305    |
| forecast |  0.983931 |    1.00023 | 0.0868551 | 108 | 1,026,980    |

**False** yields the best median OWA while **forecast** is the weakest (Δ = 0.0018).


## 3b. Selecting the Optimal Latent Dimension

The **latent dimension** controls the information bottleneck width in the AERootBlock backbone. The encoder compresses each block's input from `backcast_length → units/2 → latent_dim`, and the decoder expands it back to `latent_dim → units/2 → units` before the head layers produce backcast and forecast outputs. A smaller latent dim forces stronger compression (regularisation); a larger dim preserves more information but risks overfitting.

With backcast_length = 30 and forecast_length = 6, the tested latent dimensions are: **2, 8, 16**.

- **latent_dim = 2:** median OWA = 1.0026, std = 0.1126, params ≈ 1,024,410 ← worst
- **latent_dim = 8:** median OWA = 0.9879, std = 0.0794, params ≈ 1,032,120
- **latent_dim = 16:** median OWA = 0.9516, std = 1.0053, params ≈ 1,044,665 ← best

The optimal setting is **latent_dim = 16** (median OWA 0.9516), outperforming the worst setting (latent_dim = 2) by Δ = 0.0510. 
The largest bottleneck wins, suggesting that the AE backbone benefits from preserving more information. Despite the Trend block's polynomial constraints, the backbone's richer features improve forecast quality at the cost of mild overfitting risk.

**Practical recommendation:** Use `latent_dim = 16` as the default for TrendAE stacks on M4-Yearly. For longer forecast horizons or higher-frequency data, consider scaling proportionally (e.g. latent_dim ≈ backcast_length / 5–10).


## 4. Variant Head-to-Head


### Round 1 — Best Config per Variant

| Variant             | Best Config                            |   Med OWA |
|:--------------------|:---------------------------------------|----------:|
| GenericAEBackcast   | GenericAEBackcast_ld8_td5_ttd3_ag0     |  0.867879 |
| AutoEncoder         | AutoEncoder_ld16_td5_ttd5_ag0          |  0.888458 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd3_ag0 |  0.900238 |
| GenericAE           | GenericAE_ld16_td5_ttd5_ag0            |  0.909601 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld8_td10_ttd3_ag0  |  0.91881  |
| AutoEncoderAE       | AutoEncoderAE_ld8_td10_ttd3_ag0        |  0.936608 |

### Round 2 — Best Config per Variant

| Variant             | Best Config                            |   Med OWA |
|:--------------------|:---------------------------------------|----------:|
| GenericAEBackcast   | GenericAEBackcast_ld16_td5_ttd5_ag0    |  0.827039 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd3_ag0 |  0.847578 |
| GenericAE           | GenericAE_ld16_td5_ttd3_agF            |  0.850374 |
| AutoEncoder         | AutoEncoder_ld2_td5_ttd5_ag0           |  0.855241 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld16_td10_ttd3_agF |  0.871831 |

### Round 3 — Best Config per Variant

| Variant             | Best Config                            |   Med OWA |
|:--------------------|:---------------------------------------|----------:|
| GenericAE           | GenericAE_ld16_td5_ttd3_ag0            |  0.807247 |
| GenericAEBackcast   | GenericAEBackcast_ld8_td5_ttd3_ag0     |  0.807458 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld16_td10_ttd3_agF |  0.820869 |

Variants that maintain top positions across rounds demonstrate robust performance independent of training budget.


## 5. Stability Analysis (OWA Spread Across Seeds)


### Round 1

- **Mean spread (max−min):** 0.2376
- **Max spread (max−min):** 9.5668 (`BottleneckGenericAE_ld16_td5_ttd3_ag0`)
- **Mean std:** 0.1298
- **Most stable configs:** `GenericAE_ld2_td5_ttd3_agF` (±0.0116), `GenericAE_ld16_td5_ttd3_agF` (±0.0122), `AutoEncoder_ld16_td5_ttd5_ag0` (±0.0130)

### Round 2

- **Mean spread (max−min):** 0.3926
- **Max spread (max−min):** 9.6081 (`BottleneckGenericAE_ld16_td5_ttd3_ag0`)
- **Mean std:** 0.2227
- **Most stable configs:** `GenericAE_ld16_td5_ttd3_agF` (±0.0051), `GenericAEBackcast_ld2_td5_ttd5_ag0` (±0.0113), `GenericAEBackcast_ld8_td5_ttd3_ag0` (±0.0149)

### Round 3

- **Mean spread (max−min):** 0.0273
- **Max spread (max−min):** 0.0927 (`GenericAEBackcast_ld16_td5_ttd5_ag0`)
- **Mean std:** 0.0145
- **Most stable configs:** `GenericAEBackcast_ld16_td5_ttd5_agF` (±0.0078), `GenericAEBackcastAE_ld16_td10_ttd3_agF` (±0.0109), `GenericAE_ld16_td5_ttd3_ag0` (±0.0136)

Lower spread values indicate more consistent performance across random seeds. Configurations with high spread may be sensitive to initialization.


## 6. Round-over-Round Progression (Final Configs)

| config_name                            |     R1 |     R2 |     R3 |       Δ |    Δ% |
|:---------------------------------------|-------:|-------:|-------:|--------:|------:|
| GenericAE_ld16_td5_ttd3_ag0            | 0.9407 | 0.873  | 0.8072 | -0.1335 | -14.2 |
| GenericAEBackcastAE_ld16_td10_ttd3_agF | 0.9532 | 0.8718 | 0.8209 | -0.1323 | -13.9 |
| GenericAEBackcast_ld16_td5_ttd5_ag0    | 0.9432 | 0.827  | 0.8122 | -0.131  | -13.9 |
| GenericAEBackcast_ld2_td5_ttd5_ag0     | 0.9185 | 0.8433 | 0.8134 | -0.1051 | -11.4 |
| GenericAE_ld16_td5_ttd3_agF            | 0.9144 | 0.8504 | 0.8099 | -0.1045 | -11.4 |
| GenericAEBackcast_ld16_td5_ttd3_agF    | 0.896  | 0.8702 | 0.8114 | -0.0846 |  -9.4 |
| GenericAEBackcast_ld16_td5_ttd5_agF    | 0.8931 | 0.8557 | 0.8112 | -0.0819 |  -9.2 |
| GenericAEBackcast_ld16_td5_ttd3_ag0    | 0.8714 | 0.8635 | 0.8081 | -0.0633 |  -7.3 |
| GenericAEBackcast_ld8_td5_ttd3_ag0     | 0.8679 | 0.8577 | 0.8075 | -0.0604 |  -7   |

**9 of 9 surviving configurations improved** their OWA from R1 to R3, confirming that additional training epochs benefit the top candidates.


## 7. Parameter Efficiency (Final Round)

| Config                                 | Params    | Reduction   |   Med OWA | Target   |
|:---------------------------------------|:----------|:------------|----------:|:---------|
| GenericAE_ld16_td5_ttd3_ag0            | 1,042,095 | 95.8%       |    0.8072 | ✓        |
| GenericAE_ld16_td5_ttd3_agF            | 1,042,095 | 95.8%       |    0.8099 | ✓        |
| GenericAEBackcastAE_ld16_td10_ttd3_agF | 1,078,430 | 95.6%       |    0.8209 | ✓        |
| GenericAEBackcast_ld2_td5_ttd5_ag0     | 4,311,315 | 82.5%       |    0.8134 | ✓        |
| GenericAEBackcast_ld8_td5_ttd3_ag0     | 4,355,065 | 82.4%       |    0.8075 | ✓        |
| GenericAEBackcast_ld16_td5_ttd3_ag0    | 4,416,825 | 82.1%       |    0.8081 | ✓        |
| GenericAEBackcast_ld16_td5_ttd3_agF    | 4,416,825 | 82.1%       |    0.8114 | ✓        |
| GenericAEBackcast_ld16_td5_ttd5_ag0    | 4,419,395 | 82.1%       |    0.8122 | ✓        |
| GenericAEBackcast_ld16_td5_ttd5_agF    | 4,419,395 | 82.1%       |    0.8112 | ✓        |

All TrendAE configurations achieve substantial parameter reductions relative to the 24,700,000-parameter Generic baseline. The best-performing config uses 1,042,095 parameters (96% reduction) while achieving OWA = 0.8072.


## 8. Final Verdict

**Target:** OWA < 0.85, Params < 5,000,000
**Baseline:** N-BEATS-G 30-stack = 24,700,000 params

✅ **9 configuration(s) MEET the target:**

**GenericAE_ld16_td5_ttd3_ag0**

- OWA: 0.8072 (range 0.8068–0.8204)
- sMAPE: 13.62, MASE: 3.11
- Params: 1,042,095 (96% reduction)
- Hyperparams: ae=GenericAE, latent_dim=16, thetas_dim=5, trend_thetas_dim=3, active_g=False

**GenericAEBackcast_ld8_td5_ttd3_ag0**

- OWA: 0.8075 (range 0.7960–0.8310)
- sMAPE: 13.59, MASE: 3.11
- Params: 4,355,065 (82% reduction)
- Hyperparams: ae=GenericAEBackcast, latent_dim=8, thetas_dim=5, trend_thetas_dim=3, active_g=False

**GenericAEBackcast_ld16_td5_ttd3_ag0**

- OWA: 0.8081 (range 0.8053–0.8343)
- sMAPE: 13.57, MASE: 3.12
- Params: 4,416,825 (82% reduction)
- Hyperparams: ae=GenericAEBackcast, latent_dim=16, thetas_dim=5, trend_thetas_dim=3, active_g=False

**GenericAE_ld16_td5_ttd3_agF**

- OWA: 0.8099 (range 0.8006–0.8155)
- sMAPE: 13.63, MASE: 3.12
- Params: 1,042,095 (96% reduction)
- Hyperparams: ae=GenericAE, latent_dim=16, thetas_dim=5, trend_thetas_dim=3, active_g=forecast

**GenericAEBackcast_ld16_td5_ttd5_agF**

- OWA: 0.8112 (range 0.8066–0.8144)
- sMAPE: 13.63, MASE: 3.13
- Params: 4,419,395 (82% reduction)
- Hyperparams: ae=GenericAEBackcast, latent_dim=16, thetas_dim=5, trend_thetas_dim=5, active_g=forecast

**GenericAEBackcast_ld16_td5_ttd3_agF**

- OWA: 0.8114 (range 0.7995–0.8236)
- sMAPE: 13.63, MASE: 3.13
- Params: 4,416,825 (82% reduction)
- Hyperparams: ae=GenericAEBackcast, latent_dim=16, thetas_dim=5, trend_thetas_dim=3, active_g=forecast

**GenericAEBackcast_ld16_td5_ttd5_ag0**

- OWA: 0.8122 (range 0.8000–0.8927)
- sMAPE: 13.63, MASE: 3.14
- Params: 4,419,395 (82% reduction)
- Hyperparams: ae=GenericAEBackcast, latent_dim=16, thetas_dim=5, trend_thetas_dim=5, active_g=False

**GenericAEBackcast_ld2_td5_ttd5_ag0**

- OWA: 0.8134 (range 0.8112–0.8287)
- sMAPE: 13.68, MASE: 3.14
- Params: 4,311,315 (83% reduction)
- Hyperparams: ae=GenericAEBackcast, latent_dim=2, thetas_dim=5, trend_thetas_dim=5, active_g=False

**GenericAEBackcastAE_ld16_td10_ttd3_agF**

- OWA: 0.8209 (range 0.8113–0.8223)
- sMAPE: 13.75, MASE: 3.18
- Params: 1,078,430 (96% reduction)
- Hyperparams: ae=GenericAEBackcastAE, latent_dim=16, thetas_dim=10, trend_thetas_dim=3, active_g=forecast

