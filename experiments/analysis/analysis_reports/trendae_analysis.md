# TrendAE Architecture Search - Multi-Dataset Analysis


## Dataset: m4

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/trendae_study_results.csv`
- Rows: 384
- Primary metric: `owa`

### Abstract

This analysis covers 90 configurations over 3 rounds (384 runs).
Total training time: 55.6 min.
OWA range: 0.7960 - 10.4784.


## 1. Successive Halving Funnel

|   Round |   Configs |   Runs |   Epochs |   Best Med OWA |
|--------:|----------:|-------:|---------:|---------------:|
|       1 |        90 |    270 |       10 |         0.8679 |
|       2 |        29 |     87 |       15 |         0.827  |
|       3 |         9 |     27 |       30 |         0.8072 |


## 2.1 Round 1 Leaderboard

90 configs x 3 runs, up to 10 epochs each

|   Rank | Config                                 |    OWA |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|-------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld8_td5_ttd3_ag0     | 0.8679 |   0.1514 |   14.49 |   3.37 | 4,355,065 | 7.7s   |          |
|      2 | GenericAEBackcast_ld16_td5_ttd3_ag0    | 0.8714 |   0.067  |   14.66 |   3.36 | 4,416,825 | 7.6s   |          |
|      3 | AutoEncoder_ld16_td5_ttd5_ag0          | 0.8885 |   0.013  |   14.92 |   3.44 | 4,436,785 | 7.3s   |          |
|      4 | AutoEncoder_ld16_td5_ttd3_ag0          | 0.8894 |   0.0731 |   15.07 |   3.4  | 4,434,215 | 7.7s   |          |
|      5 | GenericAEBackcast_ld16_td5_ttd5_agF    | 0.8931 |   0.1304 |   14.83 |   3.49 | 4,419,395 | 7.6s   |          |
|      6 | GenericAEBackcast_ld16_td5_ttd3_agF    | 0.896  |   0.0363 |   14.88 |   3.5  | 4,416,825 | 7.6s   |          |
|      7 | BottleneckGenericAE_ld16_td10_ttd3_ag0 | 0.9002 |   0.0182 |   15.26 |   3.44 | 977,385   | 7.1s   |          |
|      8 | GenericAEBackcast_ld2_td5_ttd3_agF     | 0.9042 |   0.2373 |   15.07 |   3.52 | 4,308,745 | 7.6s   |          |
|      9 | GenericAE_ld16_td5_ttd5_ag0            | 0.9096 |   0.1321 |   15.13 |   3.55 | 1,044,665 | 7.0s   |          |
|     10 | AutoEncoder_ld8_td5_ttd5_ag0           | 0.9104 |   0.0649 |   15.49 |   3.47 | 4,375,305 | 6.7s   |          |
|     11 | GenericAE_ld16_td5_ttd3_agF            | 0.9144 |   0.0122 |   15.23 |   3.56 | 1,042,095 | 7.0s   |          |
|     12 | AutoEncoder_ld8_td5_ttd3_ag0           | 0.917  |   0.0498 |   15.18 |   3.6  | 4,372,735 | 6.8s   |          |
|     13 | GenericAEBackcast_ld2_td5_ttd5_ag0     | 0.9185 |   0.3536 |   15.23 |   3.6  | 4,311,315 | 7.6s   |          |
|     14 | GenericAEBackcastAE_ld8_td10_ttd3_ag0  | 0.9188 |   0.0455 |   15.53 |   3.53 | 1,068,150 | 7.2s   |          |
|     15 | BottleneckGenericAE_ld16_td5_ttd3_ag0  | 0.9208 |   9.5668 |   15.52 |   3.54 | 963,660   | 7.1s   |          |
|     16 | GenericAEBackcast_ld8_td5_ttd5_agF     | 0.9213 |   0.0441 |   15.29 |   3.6  | 4,357,635 | 7.6s   |          |
|     17 | GenericAEBackcast_ld2_td5_ttd5_agF     | 0.9244 |   0.0326 |   15.33 |   3.62 | 4,311,315 | 7.7s   |          |
|     18 | GenericAE_ld8_td5_ttd3_ag0             | 0.9315 |   0.0711 |   15.42 |   3.65 | 1,011,295 | 6.9s   |          |
|     19 | GenericAEBackcastAE_ld16_td10_ttd3_ag0 | 0.9346 |   0.0206 |   15.57 |   3.61 | 1,078,430 | 7.2s   |          |
|     20 | AutoEncoderAE_ld8_td10_ttd3_ag0        | 0.9366 |   0.052  |   15.91 |   3.61 | 1,093,445 | 7.4s   |          |
|     21 | GenericAEBackcast_ld8_td5_ttd5_ag0     | 0.9378 |   0.1157 |   15.37 |   3.72 | 4,357,635 | 7.6s   |          |
|     22 | AutoEncoder_ld2_td5_ttd3_ag0           | 0.9391 |   0.0311 |   15.77 |   3.63 | 4,326,625 | 6.6s   |          |
|     23 | GenericAE_ld16_td5_ttd3_ag0            | 0.9407 |   0.1524 |   15.49 |   3.71 | 1,042,095 | 6.9s   |          |
|     24 | GenericAE_ld2_td5_ttd3_agF             | 0.9415 |   0.0116 |   15.81 |   3.63 | 988,195   | 6.8s   |          |
|     25 | GenericAEBackcast_ld16_td5_ttd5_ag0    | 0.9432 |   0.0768 |   15.47 |   3.72 | 4,419,395 | 7.5s   |          |
|     26 | BottleneckGenericAE_ld16_td5_ttd3_agF  | 0.9436 |   0.1389 |   15.85 |   3.65 | 963,660   | 7.1s   |          |
|     27 | AutoEncoder_ld2_td5_ttd5_ag0           | 0.9483 |   0.0854 |   15.87 |   3.68 | 4,329,195 | 6.8s   |          |
|     28 | GenericAEBackcastAE_ld16_td5_ttd3_ag0  | 0.9512 |   0.1068 |   15.98 |   3.71 | 1,039,830 | 7.2s   |          |
|     29 | GenericAEBackcastAE_ld8_td10_ttd3_agF  | 0.9513 |   0.0551 |   15.94 |   3.71 | 1,068,150 | 7.2s   |          |
|     30 | GenericAE_ld8_td5_ttd3_agF             | 0.9515 |   0.0635 |   15.75 |   3.73 | 1,011,295 | 7.1s   |          |
|     31 | GenericAEBackcastAE_ld16_td10_ttd3_agF | 0.9532 |   0.0545 |   15.96 |   3.7  | 1,078,430 | 7.2s   |          |
|     32 | BottleneckGenericAE_ld16_td10_ttd3_agF | 0.9568 |   0.0412 |   16.07 |   3.66 | 977,385   | 7.1s   |          |
|     33 | GenericAE_ld2_td5_ttd5_ag0             | 0.9669 |   0.1682 |   16.25 |   3.73 | 990,765   | 6.8s   |          |
|     34 | BottleneckGenericAE_ld8_td10_ttd5_agF  | 0.9716 |   0.1979 |   16.24 |   3.77 | 949,155   | 7.1s   |          |
|     35 | GenericAEBackcast_ld2_td5_ttd3_ag0     | 0.9722 |   0.1235 |   16.16 |   3.8  | 4,308,745 | 7.8s   |          |
|     36 | GenericAEBackcastAE_ld16_td10_ttd5_agF | 0.9741 |   0.0806 |   16.36 |   3.76 | 1,081,000 | 7.2s   |          |
|     37 | BottleneckGenericAE_ld2_td10_ttd3_agF  | 0.9747 |   0.103  |   16.45 |   3.75 | 923,485   | 6.9s   |          |
|     38 | AutoEncoderAE_ld16_td5_ttd5_ag0        | 0.9748 |   0.1002 |   16.06 |   3.84 | 1,088,390 | 7.2s   |          |
|     39 | GenericAE_ld8_td5_ttd5_ag0             | 0.9811 |   0.1697 |   16.04 |   3.9  | 1,013,865 | 7.2s   |          |
|     40 | BottleneckGenericAE_ld16_td5_ttd5_agF  | 0.9824 |   0.0777 |   16.36 |   3.87 | 966,230   | 7.1s   |          |
|     41 | GenericAE_ld16_td5_ttd5_agF            | 0.9837 |   0.0621 |   16.18 |   3.85 | 1,044,665 | 7.0s   |          |
|     42 | BottleneckGenericAE_ld16_td10_ttd5_agF | 0.9841 |   0.0965 |   16.34 |   3.85 | 979,955   | 7.1s   |          |
|     43 | BottleneckGenericAE_ld2_td5_ttd3_ag0   | 0.9848 |   0.144  |   16.51 |   3.81 | 909,760   | 7.1s   |          |
|     44 | BottleneckGenericAE_ld2_td5_ttd3_agF   | 0.9849 |   0.0244 |   16.55 |   3.85 | 909,760   | 7.1s   |          |
|     45 | AutoEncoderAE_ld8_td10_ttd5_ag0        | 0.9866 |   0.2356 |   16.59 |   3.81 | 1,096,015 | 7.2s   |          |
|     46 | GenericAE_ld2_td5_ttd3_ag0             | 0.9888 |   0.2933 |   16.57 |   3.83 | 988,195   | 6.8s   |          |
|     47 | GenericAEBackcastAE_ld8_td5_ttd5_ag0   | 0.9908 |   0.0924 |   16.55 |   3.86 | 1,032,120 | 7.1s   |          |
|     48 | GenericAEBackcastAE_ld16_td5_ttd3_agF  | 0.9916 |   0.1707 |   16.53 |   3.86 | 1,039,830 | 7.2s   |          |
|     49 | GenericAE_ld2_td5_ttd5_agF             | 0.9941 |   0.1704 |   16.4  |   3.91 | 990,765   | 6.9s   |          |
|     50 | GenericAEBackcastAE_ld2_td5_ttd5_ag0   | 0.9954 |   0.2073 |   16.64 |   3.87 | 1,024,410 | 7.2s   |          |
|     51 | GenericAEBackcastAE_ld16_td5_ttd5_agF  | 0.9966 |   0.1102 |   16.65 |   3.87 | 1,042,400 | 7.2s   |          |
|     52 | GenericAEBackcastAE_ld16_td5_ttd5_ag0  | 0.9966 |   0.0856 |   16.72 |   3.85 | 1,042,400 | 7.2s   |          |
|     53 | GenericAEBackcastAE_ld8_td10_ttd5_ag0  | 1.0001 |   0.1087 |   16.43 |   3.95 | 1,070,720 | 7.2s   |          |
|     54 | GenericAEBackcastAE_ld2_td5_ttd3_ag0   | 1.0018 |   0.0893 |   16.67 |   3.91 | 1,021,840 | 7.2s   |          |
|     55 | AutoEncoderAE_ld2_td10_ttd3_ag0        | 1.0033 |   0.1238 |   16.85 |   3.88 | 1,070,345 | 7.1s   |          |
|     56 | AutoEncoderAE_ld2_td5_ttd3_ag0         | 1.0037 |   0.0565 |   16.78 |   3.9  | 1,031,920 | 7.2s   |          |
|     57 | AutoEncoderAE_ld16_td10_ttd3_ag0       | 1.0075 |   0.1024 |   16.57 |   3.98 | 1,124,245 | 7.2s   |          |
|     58 | BottleneckGenericAE_ld16_td10_ttd5_ag0 | 1.0089 |   0.0733 |   16.66 |   3.97 | 979,955   | 7.1s   |          |
|     59 | BottleneckGenericAE_ld8_td10_ttd5_ag0  | 1.0117 |   0.2062 |   16.53 |   4.02 | 949,155   | 7.1s   |          |
|     60 | BottleneckGenericAE_ld8_td5_ttd5_agF   | 1.0117 |   0.1698 |   16.96 |   3.92 | 935,430   | 7.1s   |          |
|     61 | BottleneckGenericAE_ld8_td10_ttd3_agF  | 1.0127 |   0.0743 |   16.5  |   4.02 | 946,585   | 7.1s   |          |
|     62 | BottleneckGenericAE_ld8_td5_ttd3_ag0   | 1.0132 |   0.0867 |   16.74 |   3.98 | 932,860   | 7.2s   |          |
|     63 | GenericAE_ld8_td5_ttd5_agF             | 1.0146 |   0.1417 |   16.65 |   4.02 | 1,013,865 | 7.1s   |          |
|     64 | GenericAEBackcast_ld8_td5_ttd3_agF     | 1.0165 |   0.1477 |   16.37 |   4.1  | 4,355,065 | 7.6s   |          |
|     65 | BottleneckGenericAE_ld2_td10_ttd3_ag0  | 1.0186 |   0.0725 |   16.91 |   3.98 | 923,485   | 6.9s   |          |
|     66 | AutoEncoderAE_ld8_td5_ttd5_ag0         | 1.0193 |   0.1097 |   16.84 |   3.99 | 1,057,590 | 7.2s   |          |
|     67 | BottleneckGenericAE_ld8_td5_ttd5_ag0   | 1.0214 |   0.1816 |   16.78 |   4.04 | 935,430   | 7.1s   |          |
|     68 | GenericAEBackcastAE_ld2_td10_ttd5_agF  | 1.0238 |   0.2329 |   17.1  |   3.98 | 1,063,010 | 7.1s   |          |
|     69 | BottleneckGenericAE_ld16_td5_ttd5_ag0  | 1.0253 |   0.2041 |   17.09 |   3.99 | 966,230   | 7.1s   |          |
|     70 | AutoEncoderAE_ld16_td10_ttd5_ag0       | 1.0263 |   0.2698 |   16.92 |   4.04 | 1,126,815 | 7.2s   |          |
|     71 | GenericAEBackcastAE_ld8_td5_ttd3_agF   | 1.0321 |   0.1139 |   17.07 |   4.05 | 1,029,550 | 7.1s   |          |
|     72 | BottleneckGenericAE_ld8_td5_ttd3_agF   | 1.0347 |   0.0657 |   16.96 |   4.1  | 932,860   | 7.1s   |          |
|     73 | GenericAEBackcastAE_ld2_td10_ttd3_ag0  | 1.0416 |   0.0443 |   17.15 |   4.11 | 1,060,440 | 7.1s   |          |
|     74 | AutoEncoderAE_ld8_td5_ttd3_ag0         | 1.0484 |   0.2733 |   17.23 |   4.14 | 1,055,020 | 7.2s   |          |
|     75 | AutoEncoderAE_ld2_td5_ttd5_ag0         | 1.0551 |   0.2276 |   17.38 |   4.16 | 1,034,490 | 7.1s   |          |
|     76 | GenericAEBackcastAE_ld8_td5_ttd3_ag0   | 1.0553 |   0.129  |   17.25 |   4.19 | 1,029,550 | 7.1s   |          |
|     77 | BottleneckGenericAE_ld8_td10_ttd3_ag0  | 1.0558 |   0.2289 |   17.44 |   4.15 | 946,585   | 7.1s   |          |
|     78 | GenericAEBackcastAE_ld2_td5_ttd3_agF   | 1.058  |   0.1982 |   17.39 |   4.18 | 1,021,840 | 7.2s   |          |
|     79 | BottleneckGenericAE_ld2_td5_ttd5_agF   | 1.0601 |   0.3406 |   17.45 |   4.18 | 912,330   | 6.9s   |          |
|     80 | BottleneckGenericAE_ld2_td10_ttd5_agF  | 1.0606 |   0.3599 |   17.53 |   4.17 | 926,055   | 6.9s   |          |
|     81 | BottleneckGenericAE_ld2_td10_ttd5_ag0  | 1.0643 |   0.3314 |   17.67 |   4.16 | 926,055   | 6.9s   |          |
|     82 | GenericAEBackcastAE_ld8_td5_ttd5_agF   | 1.0683 |   0.1398 |   17.51 |   4.23 | 1,032,120 | 7.1s   |          |
|     83 | GenericAEBackcastAE_ld2_td10_ttd5_ag0  | 1.0736 |   0.4235 |   17.68 |   4.23 | 1,063,010 | 7.2s   |          |
|     84 | GenericAEBackcastAE_ld2_td5_ttd5_agF   | 1.0878 |   0.1499 |   17.57 |   4.37 | 1,024,410 | 7.2s   |          |
|     85 | GenericAEBackcastAE_ld16_td10_ttd5_ag0 | 1.1029 |   0.1415 |   17.59 |   4.49 | 1,081,000 | 7.2s   |          |
|     86 | GenericAEBackcastAE_ld8_td10_ttd5_agF  | 1.1039 |   0.1213 |   17.79 |   4.45 | 1,070,720 | 7.2s   |          |
|     87 | BottleneckGenericAE_ld2_td5_ttd5_ag0   | 1.104  |   0.1697 |   18.15 |   4.36 | 912,330   | 7.0s   |          |
|     88 | GenericAEBackcastAE_ld2_td10_ttd3_agF  | 1.1059 |   0.1159 |   17.85 |   4.45 | 1,060,440 | 7.1s   |          |
|     89 | AutoEncoderAE_ld16_td5_ttd3_ag0        | 1.1232 |   0.2348 |   17.81 |   4.6  | 1,085,820 | 7.2s   |          |
|     90 | AutoEncoderAE_ld2_td10_ttd5_ag0        | 1.1241 |   0.3022 |   18.47 |   4.44 | 1,072,915 | 7.2s   |          |

Top config median OWA=0.8679; worst=1.1241; delta=0.2562.


## 2.2 Round 2 Leaderboard

29 configs x 3 runs, up to 15 epochs each

|   Rank | Config                                 |    OWA |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|-------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld16_td5_ttd5_ag0    | 0.827  |   0.075  |   13.85 |   3.21 | 4,419,395 | 10.7s  | YES      |
|      2 | GenericAEBackcast_ld2_td5_ttd5_ag0     | 0.8433 |   0.0113 |   14.12 |   3.24 | 4,311,315 | 10.8s  | YES      |
|      3 | BottleneckGenericAE_ld16_td10_ttd3_ag0 | 0.8476 |   0.0512 |   14.28 |   3.26 | 977,385   | 10.0s  | YES      |
|      4 | GenericAE_ld16_td5_ttd3_agF            | 0.8504 |   0.0051 |   14.26 |   3.3  | 1,042,095 | 10.1s  |          |
|      5 | AutoEncoder_ld2_td5_ttd5_ag0           | 0.8552 |   0.0454 |   14.39 |   3.3  | 4,329,195 | 10.3s  |          |
|      6 | GenericAEBackcast_ld8_td5_ttd5_ag0     | 0.8556 |   0.1418 |   14.36 |   3.31 | 4,357,635 | 10.7s  |          |
|      7 | GenericAEBackcast_ld16_td5_ttd5_agF    | 0.8557 |   0.0441 |   14.3  |   3.32 | 4,419,395 | 10.8s  |          |
|      8 | GenericAEBackcast_ld2_td5_ttd5_agF     | 0.8568 |   0.044  |   14.28 |   3.34 | 4,311,315 | 10.7s  |          |
|      9 | GenericAEBackcast_ld8_td5_ttd3_ag0     | 0.8577 |   0.0149 |   14.34 |   3.33 | 4,355,065 | 10.6s  |          |
|     10 | AutoEncoder_ld8_td5_ttd3_ag0           | 0.8584 |   0.1046 |   14.26 |   3.36 | 4,372,735 | 10.6s  |          |
|     11 | AutoEncoder_ld16_td5_ttd5_ag0          | 0.8617 |   0.113  |   14.33 |   3.37 | 4,436,785 | 10.6s  |          |
|     12 | GenericAE_ld8_td5_ttd3_agF             | 0.863  |   0.0804 |   14.37 |   3.36 | 1,011,295 | 10.0s  |          |
|     13 | GenericAEBackcast_ld16_td5_ttd3_ag0    | 0.8635 |   0.0539 |   14.3  |   3.39 | 4,416,825 | 10.7s  |          |
|     14 | GenericAEBackcast_ld16_td5_ttd3_agF    | 0.8702 |   0.026  |   14.33 |   3.43 | 4,416,825 | 10.8s  |          |
|     15 | AutoEncoder_ld2_td5_ttd3_ag0           | 0.8703 |   0.121  |   14.55 |   3.38 | 4,326,625 | 10.4s  |          |
|     16 | GenericAEBackcastAE_ld16_td10_ttd3_agF | 0.8718 |   0.0507 |   14.51 |   3.4  | 1,078,430 | 10.2s  |          |
|     17 | GenericAE_ld16_td5_ttd3_ag0            | 0.873  |   0.1175 |   14.49 |   3.42 | 1,042,095 | 9.9s   |          |
|     18 | GenericAEBackcast_ld2_td5_ttd3_ag0     | 0.8794 |   0.0707 |   14.7  |   3.42 | 4,308,745 | 10.8s  |          |
|     19 | BottleneckGenericAE_ld16_td5_ttd3_ag0  | 0.8802 |   9.6081 |   14.61 |   3.44 | 963,660   | 10.0s  |          |
|     20 | GenericAEBackcast_ld8_td5_ttd5_agF     | 0.881  |   0.0264 |   14.69 |   3.43 | 4,357,635 | 10.7s  |          |
|     21 | BottleneckGenericAE_ld16_td5_ttd3_agF  | 0.8913 |   0.0937 |   14.84 |   3.47 | 963,660   | 10.0s  |          |
|     22 | GenericAE_ld2_td5_ttd5_agF             | 0.8943 |   0.0265 |   14.99 |   3.43 | 990,765   | 9.9s   |          |
|     23 | GenericAEBackcast_ld2_td5_ttd3_agF     | 0.9009 |   0.0246 |   14.96 |   3.52 | 4,308,745 | 10.7s  |          |
|     24 | AutoEncoder_ld16_td5_ttd3_ag0          | 0.903  |   0.0996 |   14.9  |   3.55 | 4,434,215 | 10.6s  |          |
|     25 | GenericAEBackcastAE_ld8_td10_ttd3_agF  | 0.9042 |   0.0706 |   14.9  |   3.56 | 1,068,150 | 10.2s  |          |
|     26 | GenericAE_ld8_td5_ttd3_ag0             | 0.9065 |   0.0467 |   14.81 |   3.6  | 1,011,295 | 9.9s   |          |
|     27 | GenericAE_ld8_td5_ttd5_ag0             | 0.9194 |   0.0502 |   15.18 |   3.62 | 1,013,865 | 10.0s  |          |
|     28 | GenericAEBackcast_ld8_td5_ttd3_agF     | 0.9458 |   0.092  |   15.41 |   3.77 | 4,355,065 | 10.6s  |          |
|     29 | BottleneckGenericAE_ld16_td10_ttd3_agF | 0.948  |   0.0749 |   15.49 |   3.77 | 977,385   | 10.0s  |          |

Top config median OWA=0.8270; worst=0.9480; delta=0.1209.


## 2.3 Round 3 Leaderboard

9 configs x 3 runs, up to 30 epochs each

|   Rank | Config                                 |    OWA |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|-------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAE_ld16_td5_ttd3_ag0            | 0.8072 |   0.0136 |   13.62 |   3.11 | 1,042,095 | 17.8s  | YES      |
|      2 | GenericAEBackcast_ld8_td5_ttd3_ag0     | 0.8075 |   0.035  |   13.59 |   3.11 | 4,355,065 | 19.5s  | YES      |
|      3 | GenericAEBackcast_ld16_td5_ttd3_ag0    | 0.8081 |   0.029  |   13.57 |   3.12 | 4,416,825 | 19.1s  | YES      |
|      4 | GenericAE_ld16_td5_ttd3_agF            | 0.8099 |   0.0149 |   13.63 |   3.12 | 1,042,095 | 18.0s  | YES      |
|      5 | GenericAEBackcast_ld16_td5_ttd5_agF    | 0.8112 |   0.0078 |   13.63 |   3.13 | 4,419,395 | 19.0s  | YES      |
|      6 | GenericAEBackcast_ld16_td5_ttd3_agF    | 0.8114 |   0.0241 |   13.63 |   3.13 | 4,416,825 | 18.9s  | YES      |
|      7 | GenericAEBackcast_ld16_td5_ttd5_ag0    | 0.8122 |   0.0927 |   13.63 |   3.14 | 4,419,395 | 19.2s  | YES      |
|      8 | GenericAEBackcast_ld2_td5_ttd5_ag0     | 0.8134 |   0.0174 |   13.68 |   3.14 | 4,311,315 | 18.8s  | YES      |
|      9 | GenericAEBackcastAE_ld16_td10_ttd3_agF | 0.8209 |   0.0109 |   13.75 |   3.18 | 1,078,430 | 17.9s  | YES      |

Top config median OWA=0.8072; worst=0.8209; delta=0.0136.


## 3. Hyperparameter Marginals (Round 1)


### AE Variant

| Value               |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|:--------------------|----------:|-----------:|----------:|----:|:-------------|
| AutoEncoder         |  0.913685 |   0.912922 | 0.0304832 |  18 | 4,374,020    |
| GenericAEBackcast   |  0.919228 |   0.935336 | 0.0808918 |  36 | 4,356,350    |
| GenericAE           |  0.942988 |   0.96995  | 0.0701907 |  36 | 1,012,580    |
| GenericAEBackcastAE |  1.00462  |   1.02708  | 0.0850467 |  72 | 1,051,420    |
| BottleneckGenericAE |  1.00816  |   1.15546  | 1.11832   |  72 | 941,008      |
| AutoEncoderAE       |  1.01139  |   1.03104  | 0.099816  |  36 | 1,079,368    |


### Latent Dim

|   Value |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|--------:|----------:|-----------:|----------:|----:|:-------------|
|      16 |  0.951584 |   1.07286  | 1.00534   |  90 | 1,044,665    |
|       8 |  0.987947 |   0.996818 | 0.0794262 |  90 | 1,032,120    |
|       2 |  1.0026   |   1.03346  | 0.112582  |  90 | 1,024,410    |


### Thetas Dim

|   Value |   Med OWA |   Mean OWA |      Std |   N | Med Params   |
|--------:|----------:|-----------:|---------:|----:|:-------------|
|       5 |  0.977696 |    1.04066 | 0.712939 | 180 | 1,037,160    |
|      10 |  1.00169  |    1.02182 | 0.1005   |  90 | 1,063,010    |


### Trend Thetas Dim

|   Value |   Med OWA |   Mean OWA |      Std |   N | Med Params   |
|--------:|----------:|-----------:|---------:|----:|:-------------|
|       3 |  0.968939 |    1.05113 | 0.820959 | 135 | 1,039,830    |
|       5 |  0.9918   |    1.01763 | 0.106308 | 135 | 1,042,400    |


### active_g

| Value    |   Med OWA |   Mean OWA |       Std |   N | Med Params   |
|:---------|----------:|-----------:|----------:|----:|:-------------|
| False    |  0.982117 |    1.05715 | 0.751337  | 162 | 1,056,305    |
| forecast |  0.983931 |    1.00023 | 0.0868551 | 108 | 1,026,980    |


## 3b. Latent Dimension Discussion

|   latent_dim_cfg |   med_metric |   std_metric |   n |   med_params |
|-----------------:|-------------:|-------------:|----:|-------------:|
|               16 |     0.951584 |    1.00534   |  90 |  1.04466e+06 |
|                8 |     0.987947 |    0.0794262 |  90 |  1.03212e+06 |
|                2 |     1.0026   |    0.112582  |  90 |  1.02441e+06 |

Best latent_dim=16 by median OWA (0.9516).


## 4. Variant Head-to-Head


### Round 1 - Best Config per Variant

| Variant             | Best Config                            |   Med OWA |
|:--------------------|:---------------------------------------|----------:|
| GenericAEBackcast   | GenericAEBackcast_ld8_td5_ttd3_ag0     |  0.867879 |
| AutoEncoder         | AutoEncoder_ld16_td5_ttd5_ag0          |  0.888458 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd3_ag0 |  0.900238 |
| GenericAE           | GenericAE_ld16_td5_ttd5_ag0            |  0.909601 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld8_td10_ttd3_ag0  |  0.91881  |
| AutoEncoderAE       | AutoEncoderAE_ld8_td10_ttd3_ag0        |  0.936608 |

### Round 2 - Best Config per Variant

| Variant             | Best Config                            |   Med OWA |
|:--------------------|:---------------------------------------|----------:|
| GenericAEBackcast   | GenericAEBackcast_ld16_td5_ttd5_ag0    |  0.827039 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd3_ag0 |  0.847578 |
| GenericAE           | GenericAE_ld16_td5_ttd3_agF            |  0.850374 |
| AutoEncoder         | AutoEncoder_ld2_td5_ttd5_ag0           |  0.855241 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld16_td10_ttd3_agF |  0.871831 |

### Round 3 - Best Config per Variant

| Variant             | Best Config                            |   Med OWA |
|:--------------------|:---------------------------------------|----------:|
| GenericAE           | GenericAE_ld16_td5_ttd3_ag0            |  0.807247 |
| GenericAEBackcast   | GenericAEBackcast_ld8_td5_ttd3_ag0     |  0.807458 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld16_td10_ttd3_agF |  0.820869 |


## 5. Stability Analysis


### Round 1

- Mean spread: 0.2376
- Max spread:  9.5668 (BottleneckGenericAE_ld16_td5_ttd3_ag0)
- Mean std:    0.1298

### Round 2

- Mean spread: 0.3926
- Max spread:  9.6081 (BottleneckGenericAE_ld16_td5_ttd3_ag0)
- Mean std:    0.2227

### Round 3

- Mean spread: 0.0273
- Max spread:  0.0927 (GenericAEBackcast_ld16_td5_ttd5_ag0)
- Mean std:    0.0145


## 6. Round-over-Round Progression

| config_name                            |     R1 |     R2 |     R3 |   Delta |   DeltaPct |
|:---------------------------------------|-------:|-------:|-------:|--------:|-----------:|
| GenericAE_ld16_td5_ttd3_ag0            | 0.9407 | 0.873  | 0.8072 | -0.1335 |      -14.2 |
| GenericAEBackcastAE_ld16_td10_ttd3_agF | 0.9532 | 0.8718 | 0.8209 | -0.1323 |      -13.9 |
| GenericAEBackcast_ld16_td5_ttd5_ag0    | 0.9432 | 0.827  | 0.8122 | -0.131  |      -13.9 |
| GenericAEBackcast_ld2_td5_ttd5_ag0     | 0.9185 | 0.8433 | 0.8134 | -0.1051 |      -11.4 |
| GenericAE_ld16_td5_ttd3_agF            | 0.9144 | 0.8504 | 0.8099 | -0.1045 |      -11.4 |
| GenericAEBackcast_ld16_td5_ttd3_agF    | 0.896  | 0.8702 | 0.8114 | -0.0846 |       -9.4 |
| GenericAEBackcast_ld16_td5_ttd5_agF    | 0.8931 | 0.8557 | 0.8112 | -0.0819 |       -9.2 |
| GenericAEBackcast_ld16_td5_ttd3_ag0    | 0.8714 | 0.8635 | 0.8081 | -0.0633 |       -7.3 |
| GenericAEBackcast_ld8_td5_ttd3_ag0     | 0.8679 | 0.8577 | 0.8075 | -0.0604 |       -7   |


## 7. Parameter Efficiency

| Config                                 | Params    | Reduction   |   Med OWA |
|:---------------------------------------|:----------|:------------|----------:|
| GenericAE_ld16_td5_ttd3_ag0            | 1,042,095 | 95.8%       |    0.8072 |
| GenericAE_ld16_td5_ttd3_agF            | 1,042,095 | 95.8%       |    0.8099 |
| GenericAEBackcastAE_ld16_td10_ttd3_agF | 1,078,430 | 95.6%       |    0.8209 |
| GenericAEBackcast_ld2_td5_ttd5_ag0     | 4,311,315 | 82.5%       |    0.8134 |
| GenericAEBackcast_ld8_td5_ttd3_ag0     | 4,355,065 | 82.4%       |    0.8075 |
| GenericAEBackcast_ld16_td5_ttd3_ag0    | 4,416,825 | 82.1%       |    0.8081 |
| GenericAEBackcast_ld16_td5_ttd3_agF    | 4,416,825 | 82.1%       |    0.8114 |
| GenericAEBackcast_ld16_td5_ttd5_ag0    | 4,419,395 | 82.1%       |    0.8122 |
| GenericAEBackcast_ld16_td5_ttd5_agF    | 4,419,395 | 82.1%       |    0.8112 |


## 8. Final Verdict

Target: OWA < 0.85, Params < 5,000,000
9 configurations meet both targets.
Best final-round config: GenericAE_ld16_td5_ttd3_ag0 with median OWA=0.8072.


## Dataset: tourism

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/tourism/trendae_study_results.csv`
- Rows: 459
- Primary metric: `best_val_loss`

### Abstract

This analysis covers 90 configurations over 3 rounds (459 runs).
Total training time: 5.7 min.
OWA unavailable; using best_val_loss throughout.


## 1. Successive Halving Funnel

|   Round |   Configs |   Runs |   Epochs |   Best Med best_val_loss |
|--------:|----------:|-------:|---------:|-------------------------:|
|       1 |        90 |    270 |       10 |                  27.8578 |
|       2 |        48 |    144 |       15 |                  26.154  |
|       3 |        15 |     45 |       30 |                  24.7623 |


## 2.1 Round 1 Leaderboard

90 configs x 3 runs, up to 10 epochs each

|   Rank | Config                                 |   best_val_loss |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|----------------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld8_td5_ttd5_ag0     |         27.8578 |   2.8983 |   23.87 |   3.57 | 4,230,725 | 0.8s   |          |
|      2 | GenericAEBackcast_ld8_td5_ttd3_ag0     |         27.9854 |   0.4332 |   24.92 |   3.65 | 4,228,155 | 0.7s   |          |
|      3 | GenericAEBackcast_ld16_td5_ttd3_agF    |         28.0747 |   1.4349 |   24.43 |   3.63 | 4,289,835 | 0.8s   |          |
|      4 | GenericAEBackcast_ld16_td5_ttd5_ag0    |         28.6123 |   2.2972 |   25.17 |   3.72 | 4,292,405 | 0.9s   |          |
|      5 | GenericAEBackcast_ld16_td5_ttd3_ag0    |         28.6892 |   2.505  |   24.86 |   3.71 | 4,289,835 | 0.8s   |          |
|      6 | GenericAEBackcast_ld2_td5_ttd3_ag0     |         28.7225 |  17.8159 |   23.54 |   3.48 | 4,181,895 | 0.8s   |          |
|      7 | GenericAEBackcast_ld2_td5_ttd3_agF     |         28.8038 |   7.4284 |   24.1  |   3.54 | 4,181,895 | 0.8s   |          |
|      8 | GenericAEBackcast_ld16_td5_ttd5_agF    |         28.8165 |   1.5565 |   24.79 |   3.75 | 4,292,405 | 0.9s   |          |
|      9 | GenericAEBackcast_ld2_td5_ttd5_ag0     |         28.8471 |   1.9875 |   24.98 |   3.73 | 4,184,465 | 0.8s   |          |
|     10 | GenericAEBackcast_ld8_td5_ttd3_agF     |         28.9128 |   1.8597 |   24.1  |   3.49 | 4,228,155 | 0.8s   |          |
|     11 | GenericAEBackcast_ld2_td5_ttd5_agF     |         28.9395 |   1.7942 |   24.57 |   3.63 | 4,184,465 | 0.8s   |          |
|     12 | GenericAEBackcast_ld8_td5_ttd5_agF     |         29.3133 |   1.6966 |   24.87 |   3.7  | 4,230,725 | 0.8s   |          |
|     13 | GenericAEBackcastAE_ld8_td10_ttd3_agF  |         30.3867 |   4.9073 |   26.53 |   3.86 | 969,380   | 0.5s   |          |
|     14 | GenericAE_ld16_td5_ttd3_agF            |         30.4543 |   2.7284 |   26.83 |   4.04 | 938,415   | 0.4s   |          |
|     15 | GenericAE_ld16_td5_ttd3_ag0            |         30.6229 |   1.3797 |   26.16 |   4    | 938,415   | 0.4s   |          |
|     16 | GenericAE_ld16_td5_ttd5_agF            |         30.7074 |   2.9895 |   26.36 |   3.9  | 940,985   | 0.4s   |          |
|     17 | GenericAEBackcastAE_ld2_td10_ttd3_agF  |         30.7381 |   1.9759 |   27.08 |   3.97 | 961,670   | 0.5s   |          |
|     18 | GenericAE_ld2_td5_ttd5_ag0             |         30.9391 |   4.1958 |   29.31 |   4.91 | 887,085   | 0.4s   |          |
|     19 | AutoEncoder_ld2_td5_ttd5_ag0           |         30.9554 |   1.0431 |   25.88 |   3.97 | 4,197,235 | 0.7s   |          |
|     20 | GenericAE_ld8_td5_ttd5_agF             |         31.1659 |   1.3268 |   27.29 |   4.23 | 910,185   | 0.4s   |          |
|     21 | GenericAEBackcastAE_ld8_td10_ttd3_ag0  |         31.2565 |   5.7814 |   27.41 |   4.05 | 969,380   | 0.4s   |          |
|     22 | GenericAEBackcastAE_ld8_td5_ttd3_agF   |         31.2603 |   5.0488 |   28.98 |   4.45 | 930,830   | 0.4s   |          |
|     23 | BottleneckGenericAE_ld8_td10_ttd3_agF  |         31.2806 |   1.9    |   29.63 |   4.64 | 903,145   | 0.4s   |          |
|     24 | GenericAE_ld16_td5_ttd5_ag0            |         31.4275 |   2.469  |   28.87 |   4.34 | 940,985   | 0.4s   |          |
|     25 | GenericAE_ld2_td5_ttd3_agF             |         31.508  |   2.8446 |   28.52 |   4.56 | 884,515   | 0.4s   |          |
|     26 | GenericAE_ld8_td5_ttd3_agF             |         31.5126 |   3.8302 |   29.39 |   4.57 | 907,615   | 0.4s   |          |
|     27 | BottleneckGenericAE_ld16_td10_ttd3_agF |         31.5486 |   4.6596 |   27.85 |   4.18 | 933,945   | 0.4s   |          |
|     28 | AutoEncoder_ld8_td5_ttd5_ag0           |         31.5541 |   2.4581 |   26.91 |   4.21 | 4,243,345 | 0.7s   |          |
|     29 | AutoEncoder_ld2_td5_ttd3_ag0           |         31.6191 |   4.0108 |   26.21 |   4.11 | 4,194,665 | 1.0s   |          |
|     30 | BottleneckGenericAE_ld16_td5_ttd3_ag0  |         31.6463 |   1.2931 |   29.57 |   4.43 | 920,820   | 0.4s   |          |
|     31 | BottleneckGenericAE_ld16_td5_ttd3_agF  |         31.6627 |   1.8953 |   29    |   4.47 | 920,820   | 0.4s   |          |
|     32 | GenericAEBackcastAE_ld16_td10_ttd3_ag0 |         31.6871 |   3.932  |   29.86 |   4.78 | 979,660   | 0.4s   |          |
|     33 | GenericAE_ld8_td5_ttd5_ag0             |         31.8492 |   0.8183 |   28.59 |   4.6  | 910,185   | 0.4s   |          |
|     34 | GenericAEBackcastAE_ld16_td10_ttd5_agF |         31.8985 |   4.6398 |   28.09 |   4.44 | 982,230   | 0.5s   |          |
|     35 | AutoEncoderAE_ld8_td5_ttd5_ag0         |         31.909  |   7.0679 |   32.25 |   5.14 | 953,790   | 0.5s   |          |
|     36 | GenericAEBackcastAE_ld2_td10_ttd3_ag0  |         32.0093 |   4.2139 |   29.46 |   4.57 | 961,670   | 0.4s   |          |
|     37 | BottleneckGenericAE_ld8_td10_ttd3_ag0  |         32.0103 |   1.3089 |   29.06 |   4.7  | 903,145   | 0.4s   |          |
|     38 | GenericAE_ld2_td5_ttd3_ag0             |         32.0291 |   2.3905 |   27.88 |   4.26 | 884,515   | 0.4s   |          |
|     39 | BottleneckGenericAE_ld16_td10_ttd5_ag0 |         32.03   |   2.263  |   31.69 |   4.9  | 936,515   | 0.4s   |          |
|     40 | GenericAEBackcastAE_ld8_td10_ttd5_ag0  |         32.0642 |   1.9928 |   31.44 |   5.03 | 971,950   | 0.4s   |          |
|     41 | GenericAEBackcastAE_ld16_td5_ttd3_ag0  |         32.1744 |   1.5127 |   28.95 |   4.53 | 941,110   | 0.5s   |          |
|     42 | GenericAEBackcastAE_ld8_td5_ttd5_agF   |         32.1779 |   0.6744 |   31.87 |   5.05 | 933,400   | 0.4s   |          |
|     43 | GenericAEBackcastAE_ld8_td10_ttd5_agF  |         32.1792 |   2.3668 |   29.84 |   4.71 | 971,950   | 0.4s   |          |
|     44 | GenericAE_ld8_td5_ttd3_ag0             |         32.1971 |   2.0027 |   27.15 |   4.26 | 907,615   | 0.4s   |          |
|     45 | GenericAEBackcastAE_ld2_td10_ttd5_agF  |         32.2023 |   8.1432 |   29.37 |   4.58 | 964,240   | 0.5s   |          |
|     46 | AutoEncoder_ld8_td5_ttd3_ag0           |         32.2655 |   4.767  |   25.77 |   4.1  | 4,240,775 | 0.6s   |          |
|     47 | BottleneckGenericAE_ld8_td5_ttd3_agF   |         32.2939 |   4.9432 |   28.54 |   4.32 | 890,020   | 0.4s   |          |
|     48 | GenericAEBackcastAE_ld16_td10_ttd5_ag0 |         32.3142 |   3.9681 |   28.96 |   4.6  | 982,230   | 0.5s   |          |
|     49 | GenericAEBackcastAE_ld8_td5_ttd3_ag0   |         32.4225 |   5.2343 |   29    |   4.58 | 930,830   | 0.4s   |          |
|     50 | AutoEncoder_ld16_td5_ttd5_ag0          |         32.4962 |   3.5597 |   25.88 |   4.2  | 4,304,825 | 0.7s   |          |
|     51 | GenericAEBackcastAE_ld16_td10_ttd3_agF |         32.5165 |   2.7902 |   31    |   4.9  | 979,660   | 0.5s   |          |
|     52 | GenericAEBackcastAE_ld8_td5_ttd5_ag0   |         32.6032 |   2.5847 |   28.86 |   4.66 | 933,400   | 0.5s   |          |
|     53 | GenericAEBackcastAE_ld2_td5_ttd5_agF   |         32.6807 |   4.9046 |   29.18 |   4.55 | 925,690   | 0.5s   |          |
|     54 | BottleneckGenericAE_ld16_td10_ttd3_ag0 |         32.6882 |   4.3432 |   31.4  |   4.85 | 933,945   | 0.4s   |          |
|     55 | GenericAEBackcastAE_ld2_td5_ttd5_ag0   |         32.7072 |   3.5424 |   29.95 |   4.98 | 925,690   | 0.5s   |          |
|     56 | GenericAEBackcastAE_ld16_td5_ttd3_agF  |         32.7924 |   1.7261 |   31.59 |   4.95 | 941,110   | 0.5s   |          |
|     57 | AutoEncoderAE_ld16_td5_ttd3_ag0        |         32.8686 |   4.5161 |   29.07 |   4.57 | 982,020   | 0.4s   |          |
|     58 | GenericAEBackcastAE_ld16_td5_ttd5_agF  |         32.8802 |   2.3934 |   31.08 |   4.98 | 943,680   | 0.5s   |          |
|     59 | BottleneckGenericAE_ld2_td5_ttd3_agF   |         32.9369 |   4.7891 |   29.79 |   5.26 | 866,920   | 0.4s   |          |
|     60 | AutoEncoderAE_ld8_td5_ttd3_ag0         |         32.9405 |   2.9794 |   29.5  |   4.53 | 951,220   | 0.4s   |          |
|     61 | BottleneckGenericAE_ld16_td10_ttd5_agF |         32.9642 |   3.5202 |   32.37 |   5.08 | 936,515   | 0.4s   |          |
|     62 | AutoEncoderAE_ld16_td10_ttd3_ag0       |         33.0593 |   3.5879 |   29.71 |   4.54 | 1,020,445 | 0.4s   |          |
|     63 | AutoEncoderAE_ld2_td5_ttd5_ag0         |         33.0748 |   1.6141 |   30.73 |   4.72 | 930,690   | 0.4s   |          |
|     64 | BottleneckGenericAE_ld8_td5_ttd3_ag0   |         33.1456 |   4.805  |   30.24 |   4.78 | 890,020   | 0.4s   |          |
|     65 | AutoEncoderAE_ld16_td10_ttd5_ag0       |         33.1761 |   3.0877 |   32.66 |   5.4  | 1,023,015 | 0.4s   |          |
|     66 | GenericAE_ld2_td5_ttd5_agF             |         33.2258 |   1.6894 |   30.98 |   5.38 | 887,085   | 0.3s   |          |
|     67 | GenericAEBackcastAE_ld2_td5_ttd3_agF   |         33.3816 |   1.4149 |   30.55 |   4.83 | 923,120   | 0.4s   |          |
|     68 | AutoEncoderAE_ld8_td10_ttd3_ag0        |         33.4529 |   1.616  |   32.16 |   5.15 | 989,645   | 0.5s   |          |
|     69 | BottleneckGenericAE_ld8_td10_ttd5_ag0  |         33.4598 |   1.9619 |   30.27 |   4.75 | 905,715   | 0.4s   |          |
|     70 | AutoEncoder_ld16_td5_ttd3_ag0          |         33.5493 |   3.9018 |   34.01 |   5.5  | 4,302,255 | 0.6s   |          |
|     71 | BottleneckGenericAE_ld2_td5_ttd5_agF   |         33.5761 |   5.2743 |   33.95 |   5.7  | 869,490   | 0.4s   |          |
|     72 | BottleneckGenericAE_ld16_td5_ttd5_ag0  |         33.5926 |   3.3762 |   30.73 |   4.57 | 923,390   | 0.4s   |          |
|     73 | BottleneckGenericAE_ld8_td10_ttd5_agF  |         33.6358 |   1.7946 |   30.04 |   4.64 | 905,715   | 0.4s   |          |
|     74 | BottleneckGenericAE_ld2_td5_ttd3_ag0   |         33.6674 |   3.1925 |   30.54 |   5.45 | 866,920   | 0.4s   |          |
|     75 | AutoEncoderAE_ld8_td10_ttd5_ag0        |         33.7341 |   2.4229 |   31.17 |   4.9  | 992,215   | 0.4s   |          |
|     76 | BottleneckGenericAE_ld2_td10_ttd3_agF  |         33.9383 |   4.7535 |   33.29 |   5.79 | 880,045   | 0.4s   |          |
|     77 | GenericAEBackcastAE_ld16_td5_ttd5_ag0  |         34.1264 |   0.9406 |   34.69 |   5.58 | 943,680   | 0.5s   |          |
|     78 | AutoEncoderAE_ld2_td5_ttd3_ag0         |         34.2933 |   0.9331 |   30.05 |   4.86 | 928,120   | 0.4s   |          |
|     79 | GenericAEBackcastAE_ld2_td5_ttd3_ag0   |         34.3594 |   2.7475 |   31.24 |   4.85 | 923,120   | 0.4s   |          |
|     80 | AutoEncoderAE_ld2_td10_ttd3_ag0        |         34.3619 |   3.0369 |   30.43 |   4.49 | 966,545   | 0.5s   |          |
|     81 | GenericAEBackcastAE_ld2_td10_ttd5_ag0  |         34.5252 |   6.0793 |   33.86 |   5.62 | 964,240   | 0.4s   |          |
|     82 | BottleneckGenericAE_ld2_td5_ttd5_ag0   |         34.6052 |   9.4661 |   35.12 |   6.23 | 869,490   | 0.4s   |          |
|     83 | BottleneckGenericAE_ld16_td5_ttd5_agF  |         34.6077 |   2.4968 |   31    |   4.61 | 923,390   | 0.4s   |          |
|     84 | BottleneckGenericAE_ld8_td5_ttd5_ag0   |         34.8039 |   2.3802 |   32.92 |   5.16 | 892,590   | 0.4s   |          |
|     85 | BottleneckGenericAE_ld2_td10_ttd5_ag0  |         34.9202 |   3.9737 |   32    |   5.41 | 882,615   | 0.4s   |          |
|     86 | BottleneckGenericAE_ld8_td5_ttd5_agF   |         35.1562 |  10.5205 |   33.98 |   5.74 | 892,590   | 0.4s   |          |
|     87 | BottleneckGenericAE_ld2_td10_ttd5_agF  |         35.2709 |   0.9244 |   32.99 |   5.84 | 882,615   | 0.4s   |          |
|     88 | AutoEncoderAE_ld2_td10_ttd5_ag0        |         35.3415 |   4.9554 |   32.97 |   5.56 | 969,115   | 0.4s   |          |
|     89 | BottleneckGenericAE_ld2_td10_ttd3_ag0  |         35.4391 |   2.8306 |   32.59 |   5.98 | 880,045   | 0.4s   |          |
|     90 | AutoEncoderAE_ld16_td5_ttd5_ag0        |         36.9233 |   6.5433 |   30.53 |   4.73 | 984,590   | 0.4s   |          |

Top config median best_val_loss=27.8578; worst=36.9233; delta=9.0655.


## 2.2 Round 2 Leaderboard

48 configs x 3 runs, up to 15 epochs each

|   Rank | Config                                 |   best_val_loss |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|----------------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld16_td5_ttd3_ag0    |         26.154  |   0.436  |   21.81 |   3.17 | 4,289,835 | 1.0s   |          |
|      2 | GenericAEBackcast_ld8_td5_ttd5_ag0     |         26.2731 |   1.4536 |   21.95 |   3.24 | 4,230,725 | 1.2s   |          |
|      3 | GenericAEBackcast_ld2_td5_ttd3_agF     |         26.2952 |   2.4488 |   22.03 |   3.27 | 4,181,895 | 1.3s   |          |
|      4 | GenericAEBackcast_ld2_td5_ttd3_ag0     |         26.4096 |   7.9417 |   22.8  |   3.28 | 4,181,895 | 1.2s   |          |
|      5 | GenericAEBackcast_ld16_td5_ttd5_agF    |         26.6357 |   0.5159 |   22.33 |   3.35 | 4,292,405 | 1.3s   |          |
|      6 | GenericAEBackcast_ld8_td5_ttd3_ag0     |         26.686  |   0.4929 |   21.48 |   3.22 | 4,228,155 | 1.1s   |          |
|      7 | GenericAEBackcast_ld16_td5_ttd5_ag0    |         26.7496 |   2.61   |   21.83 |   3.22 | 4,292,405 | 1.3s   |          |
|      8 | GenericAE_ld16_td5_ttd5_agF            |         26.8337 |   3.0655 |   22.91 |   3.34 | 940,985   | 0.6s   |          |
|      9 | GenericAEBackcast_ld16_td5_ttd3_agF    |         26.9837 |   0.678  |   22.28 |   3.3  | 4,289,835 | 1.1s   |          |
|     10 | GenericAEBackcast_ld8_td5_ttd3_agF     |         27.0273 |   0.5824 |   22.39 |   3.3  | 4,228,155 | 1.0s   |          |
|     11 | GenericAEBackcast_ld2_td5_ttd5_ag0     |         27.1155 |   2.4945 |   22.56 |   3.35 | 4,184,465 | 1.1s   |          |
|     12 | GenericAE_ld8_td5_ttd5_agF             |         27.4396 |   3.3572 |   23.24 |   3.43 | 910,185   | 0.6s   |          |
|     13 | AutoEncoder_ld8_td5_ttd5_ag0           |         27.4739 |   2.4942 |   23.17 |   3.43 | 4,243,345 | 1.1s   |          |
|     14 | GenericAEBackcast_ld8_td5_ttd5_agF     |         27.6955 |   1.5395 |   22.36 |   3.37 | 4,230,725 | 1.1s   |          |
|     15 | AutoEncoder_ld2_td5_ttd5_ag0           |         27.7081 |   3.8638 |   22.77 |   3.45 | 4,197,235 | 1.1s   |          |
|     16 | GenericAE_ld16_td5_ttd5_ag0            |         28.0229 |   1.7478 |   23.54 |   3.42 | 940,985   | 0.6s   |          |
|     17 | GenericAE_ld16_td5_ttd3_ag0            |         28.3205 |   1.4691 |   23.71 |   3.56 | 938,415   | 0.6s   |          |
|     18 | GenericAEBackcast_ld2_td5_ttd5_agF     |         28.5325 |   1.8345 |   23.11 |   3.43 | 4,184,465 | 1.0s   |          |
|     19 | BottleneckGenericAE_ld16_td10_ttd5_ag0 |         28.557  |   1.9324 |   24.97 |   3.69 | 936,515   | 0.7s   |          |
|     20 | GenericAE_ld8_td5_ttd5_ag0             |         28.669  |   0.3449 |   24.01 |   3.69 | 910,185   | 1.1s   |          |
|     21 | AutoEncoder_ld2_td5_ttd3_ag0           |         28.7214 |   2.0401 |   23.06 |   3.64 | 4,194,665 | 1.2s   |          |
|     22 | GenericAEBackcastAE_ld16_td10_ttd5_ag0 |         28.8793 |   2.3124 |   24.11 |   3.65 | 982,230   | 0.8s   |          |
|     23 | GenericAE_ld16_td5_ttd3_agF            |         29.0084 |   2.9507 |   24.19 |   3.66 | 938,415   | 0.5s   |          |
|     24 | GenericAEBackcastAE_ld16_td5_ttd3_agF  |         29.0383 |   4.0386 |   24.43 |   3.55 | 941,110   | 0.8s   |          |
|     25 | BottleneckGenericAE_ld8_td10_ttd3_agF  |         29.083  |   3.1132 |   23.93 |   3.59 | 903,145   | 0.6s   |          |
|     26 | BottleneckGenericAE_ld16_td10_ttd3_agF |         29.222  |   1.524  |   24.75 |   3.72 | 933,945   | 0.6s   |          |
|     27 | BottleneckGenericAE_ld16_td10_ttd3_ag0 |         29.4268 |   2.5673 |   24.81 |   3.7  | 933,945   | 0.7s   |          |
|     28 | GenericAEBackcastAE_ld8_td10_ttd3_agF  |         29.5048 |   3.0747 |   24.89 |   3.71 | 969,380   | 0.6s   |          |
|     29 | GenericAE_ld2_td5_ttd3_agF             |         29.5454 |   1.8684 |   24.81 |   4.02 | 884,515   | 0.5s   |          |
|     30 | GenericAEBackcastAE_ld2_td10_ttd3_agF  |         29.8952 |   1.667  |   25.73 |   3.63 | 961,670   | 0.6s   |          |
|     31 | GenericAE_ld2_td5_ttd5_ag0             |         29.9149 |   7.1269 |   25.16 |   4.72 | 887,085   | 0.5s   |          |
|     32 | GenericAEBackcastAE_ld8_td5_ttd3_agF   |         29.9501 |   5.9363 |   24.67 |   3.7  | 930,830   | 0.6s   |          |
|     33 | GenericAEBackcastAE_ld2_td5_ttd3_agF   |         29.9646 |   2.0649 |   26.14 |   3.84 | 923,120   | 0.7s   |          |
|     34 | GenericAEBackcastAE_ld8_td10_ttd3_ag0  |         30.0134 |   3.1488 |   25.62 |   3.83 | 969,380   | 0.7s   |          |
|     35 | BottleneckGenericAE_ld8_td10_ttd3_ag0  |         30.0849 |   2.3286 |   24.54 |   3.72 | 903,145   | 0.6s   |          |
|     36 | BottleneckGenericAE_ld16_td10_ttd5_agF |         30.0854 |   2.7503 |   25.13 |   3.8  | 936,515   | 0.6s   |          |
|     37 | AutoEncoderAE_ld16_td10_ttd3_ag0       |         30.2079 |   1.163  |   25.38 |   3.83 | 1,020,445 | 0.7s   |          |
|     38 | AutoEncoderAE_ld8_td10_ttd5_ag0        |         30.3269 |   2.6241 |   26.03 |   4.13 | 992,215   | 0.7s   |          |
|     39 | AutoEncoderAE_ld8_td5_ttd3_ag0         |         30.3639 |   1.5801 |   24.29 |   3.65 | 951,220   | 1.0s   |          |
|     40 | GenericAE_ld8_td5_ttd3_agF             |         30.6835 |   3.3081 |   24.22 |   3.7  | 907,615   | 0.5s   |          |
|     41 | BottleneckGenericAE_ld2_td5_ttd3_agF   |         30.7537 |   3.366  |   28.32 |   4.45 | 866,920   | 0.7s   |          |
|     42 | GenericAEBackcastAE_ld16_td10_ttd3_ag0 |         30.7597 |   2.8134 |   25.08 |   3.79 | 979,660   | 0.9s   |          |
|     43 | GenericAEBackcastAE_ld8_td10_ttd5_agF  |         31.1401 |   1.2076 |   24.86 |   3.78 | 971,950   | 0.7s   |          |
|     44 | AutoEncoderAE_ld8_td10_ttd3_ag0        |         31.23   |   0.6375 |   25.98 |   3.92 | 989,645   | 0.8s   |          |
|     45 | GenericAEBackcastAE_ld2_td5_ttd3_ag0   |         31.2489 |   2.4632 |   26.34 |   3.87 | 923,120   | 0.8s   |          |
|     46 | AutoEncoderAE_ld2_td10_ttd3_ag0        |         31.3838 |   2.7599 |   27    |   4.49 | 966,545   | 0.7s   |          |
|     47 | GenericAEBackcastAE_ld8_td10_ttd5_ag0  |         31.4755 |   1.7686 |   26.06 |   4.09 | 971,950   | 0.6s   |          |
|     48 | AutoEncoderAE_ld16_td10_ttd5_ag0       |         32.3927 |   2.6466 |   30.61 |   4.74 | 1,023,015 | 0.7s   |          |

Top config median best_val_loss=26.1540; worst=32.3927; delta=6.2387.


## 2.3 Round 3 Leaderboard

15 configs x 3 runs, up to 30 epochs each

|   Rank | Config                              |   best_val_loss |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:------------------------------------|----------------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAEBackcast_ld16_td5_ttd3_ag0 |         24.7623 |   0.6043 |   21.3  |   3.05 | 4,289,835 | 1.7s   |          |
|      2 | GenericAEBackcast_ld8_td5_ttd3_ag0  |         24.9243 |   0.2498 |   21.25 |   3.07 | 4,228,155 | 1.9s   |          |
|      3 | GenericAEBackcast_ld16_td5_ttd3_agF |         24.9566 |   0.5191 |   21.21 |   3.07 | 4,289,835 | 2.2s   |          |
|      4 | GenericAEBackcast_ld8_td5_ttd3_agF  |         24.976  |   0.1377 |   21.37 |   3.08 | 4,228,155 | 2.1s   |          |
|      5 | GenericAE_ld8_td5_ttd5_agF          |         25.026  |   0.8416 |   21.56 |   3.07 | 910,185   | 1.2s   |          |
|      6 | GenericAEBackcast_ld2_td5_ttd3_agF  |         25.0478 |   1.0465 |   21.48 |   3.09 | 4,181,895 | 2.3s   |          |
|      7 | GenericAE_ld16_td5_ttd5_agF         |         25.0989 |   0.1603 |   21.12 |   3.06 | 940,985   | 1.6s   |          |
|      8 | GenericAEBackcast_ld8_td5_ttd5_ag0  |         25.106  |   0.3406 |   21.05 |   3.03 | 4,230,725 | 2.0s   |          |
|      9 | GenericAEBackcast_ld2_td5_ttd5_ag0  |         25.1064 |   0.5877 |   21.22 |   3.07 | 4,184,465 | 2.4s   |          |
|     10 | GenericAEBackcast_ld16_td5_ttd5_agF |         25.2147 |   0.5488 |   21.16 |   3.04 | 4,292,405 | 2.2s   |          |
|     11 | AutoEncoder_ld8_td5_ttd5_ag0        |         25.2183 |   1.0891 |   21.8  |   3.13 | 4,243,345 | 2.1s   |          |
|     12 | GenericAEBackcast_ld8_td5_ttd5_agF  |         25.2745 |   0.3313 |   21.11 |   3.04 | 4,230,725 | 2.4s   |          |
|     13 | GenericAEBackcast_ld16_td5_ttd5_ag0 |         25.311  |   0.2102 |   21.34 |   3.08 | 4,292,405 | 2.0s   |          |
|     14 | GenericAEBackcast_ld2_td5_ttd3_ag0  |         25.3498 |   1.2812 |   21.48 |   3.1  | 4,181,895 | 2.2s   |          |
|     15 | AutoEncoder_ld2_td5_ttd5_ag0        |         25.7969 |   1.0614 |   21.58 |   3.15 | 4,197,235 | 2.5s   |          |

Top config median best_val_loss=24.7623; worst=25.7969; delta=1.0346.


## 3. Hyperparameter Marginals (Round 1)


### AE Variant

| Value               |   Med best_val_loss |   Mean best_val_loss |     Std |   N | Med Params   |
|:--------------------|--------------------:|---------------------:|--------:|----:|:-------------|
| GenericAEBackcast   |             28.789  |              29.3602 | 3.16617 |  36 | 4,229,440    |
| GenericAE           |             31.5465 |              31.9047 | 1.36903 |  36 | 908,900      |
| AutoEncoder         |             31.7441 |              31.8035 | 1.69172 |  18 | 4,242,060    |
| GenericAEBackcastAE |             32.3979 |              32.8559 | 1.8304  |  72 | 952,675      |
| BottleneckGenericAE |             33.38   |              33.5439 | 2.2676  |  72 | 897,868      |
| AutoEncoderAE       |             33.6726 |              34.056  | 1.99328 |  36 | 975,568      |


### Latent Dim

|   Value |   Med best_val_loss |   Mean best_val_loss |     Std |   N | Med Params   |
|--------:|--------------------:|---------------------:|--------:|----:|:-------------|
|      16 |             32.087  |              32.2389 | 2.32513 |  90 | 943,680      |
|       8 |             32.1882 |              32.27   | 2.38979 |  90 | 933,400      |
|       2 |             33.0165 |              33.0999 | 2.85518 |  90 | 925,690      |


### Thetas Dim

|   Value |   Med best_val_loss |   Mean best_val_loss |     Std |   N | Med Params   |
|--------:|--------------------:|---------------------:|--------:|----:|:-------------|
|       5 |             32.0254 |              32.1564 | 2.72352 | 180 | 935,908      |
|      10 |             32.8262 |              33.2961 | 1.98939 |  90 | 964,240      |


### Trend Thetas Dim

|   Value |   Med best_val_loss |   Mean best_val_loss |     Std |   N | Med Params   |
|--------:|--------------------:|---------------------:|--------:|----:|:-------------|
|       3 |             32.0619 |              32.2586 | 2.52456 | 135 | 938,415      |
|       5 |             32.6587 |              32.814  | 2.56708 | 135 | 940,985      |


### active_g

| Value    |   Med best_val_loss |   Mean best_val_loss |     Std |   N | Med Params   |
|:---------|--------------------:|---------------------:|--------:|----:|:-------------|
| forecast |             32.0926 |              32.2392 | 2.50077 | 108 | 933,672      |
| False    |             32.5664 |              32.7343 | 2.58143 | 162 | 952,505      |


## 3b. Latent Dimension Discussion

|   latent_dim_cfg |   med_metric |   std_metric |   n |   med_params |
|-----------------:|-------------:|-------------:|----:|-------------:|
|               16 |      32.087  |      2.32513 |  90 |       943680 |
|                8 |      32.1882 |      2.38979 |  90 |       933400 |
|                2 |      33.0165 |      2.85518 |  90 |       925690 |

Best latent_dim=16 by median best_val_loss (32.0870).


## 4. Variant Head-to-Head


### Round 1 - Best Config per Variant

| Variant             | Best Config                           |   Med best_val_loss |
|:--------------------|:--------------------------------------|--------------------:|
| GenericAEBackcast   | GenericAEBackcast_ld8_td5_ttd5_ag0    |             27.8578 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld8_td10_ttd3_agF |             30.3867 |
| GenericAE           | GenericAE_ld16_td5_ttd3_agF           |             30.4543 |
| AutoEncoder         | AutoEncoder_ld2_td5_ttd5_ag0          |             30.9554 |
| BottleneckGenericAE | BottleneckGenericAE_ld8_td10_ttd3_agF |             31.2806 |
| AutoEncoderAE       | AutoEncoderAE_ld8_td5_ttd5_ag0        |             31.909  |

### Round 2 - Best Config per Variant

| Variant             | Best Config                            |   Med best_val_loss |
|:--------------------|:---------------------------------------|--------------------:|
| GenericAEBackcast   | GenericAEBackcast_ld16_td5_ttd3_ag0    |             26.154  |
| GenericAE           | GenericAE_ld16_td5_ttd5_agF            |             26.8337 |
| AutoEncoder         | AutoEncoder_ld8_td5_ttd5_ag0           |             27.4739 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd5_ag0 |             28.557  |
| GenericAEBackcastAE | GenericAEBackcastAE_ld16_td10_ttd5_ag0 |             28.8793 |
| AutoEncoderAE       | AutoEncoderAE_ld16_td10_ttd3_ag0       |             30.2079 |

### Round 3 - Best Config per Variant

| Variant           | Best Config                         |   Med best_val_loss |
|:------------------|:------------------------------------|--------------------:|
| GenericAEBackcast | GenericAEBackcast_ld16_td5_ttd3_ag0 |             24.7623 |
| GenericAE         | GenericAE_ld8_td5_ttd5_agF          |             25.026  |
| AutoEncoder       | AutoEncoder_ld8_td5_ttd5_ag0        |             25.2183 |


## 5. Stability Analysis


### Round 1

- Mean spread: 3.4073
- Max spread:  17.8159 (GenericAEBackcast_ld2_td5_ttd3_ag0)
- Mean std:    1.7998

### Round 2

- Mean spread: 2.4199
- Max spread:  7.9417 (GenericAEBackcast_ld2_td5_ttd3_ag0)
- Mean std:    1.2808

### Round 3

- Mean spread: 0.6007
- Max spread:  1.2812 (GenericAEBackcast_ld2_td5_ttd3_ag0)
- Mean std:    0.3170


## 6. Round-over-Round Progression

| config_name                         |      R1 |      R2 |      R3 |   Delta |   DeltaPct |
|:------------------------------------|--------:|--------:|--------:|--------:|-----------:|
| AutoEncoder_ld8_td5_ttd5_ag0        | 31.5541 | 27.4739 | 25.2183 | -6.3358 |      -20.1 |
| GenericAE_ld8_td5_ttd5_agF          | 31.1659 | 27.4396 | 25.026  | -6.1399 |      -19.7 |
| GenericAE_ld16_td5_ttd5_agF         | 30.7074 | 26.8337 | 25.0989 | -5.6085 |      -18.3 |
| AutoEncoder_ld2_td5_ttd5_ag0        | 30.9554 | 27.7081 | 25.7969 | -5.1584 |      -16.7 |
| GenericAEBackcast_ld8_td5_ttd5_agF  | 29.3133 | 27.6955 | 25.2745 | -4.0388 |      -13.8 |
| GenericAEBackcast_ld16_td5_ttd3_ag0 | 28.6892 | 26.154  | 24.7623 | -3.9269 |      -13.7 |
| GenericAEBackcast_ld8_td5_ttd3_agF  | 28.9128 | 27.0273 | 24.976  | -3.9368 |      -13.6 |
| GenericAEBackcast_ld2_td5_ttd3_agF  | 28.8038 | 26.2952 | 25.0478 | -3.756  |      -13   |
| GenericAEBackcast_ld2_td5_ttd5_ag0  | 28.8471 | 27.1155 | 25.1064 | -3.7407 |      -13   |
| GenericAEBackcast_ld16_td5_ttd5_agF | 28.8165 | 26.6357 | 25.2147 | -3.6018 |      -12.5 |
| GenericAEBackcast_ld2_td5_ttd3_ag0  | 28.7225 | 26.4096 | 25.3498 | -3.3727 |      -11.7 |
| GenericAEBackcast_ld16_td5_ttd5_ag0 | 28.6123 | 26.7496 | 25.311  | -3.3012 |      -11.5 |
| GenericAEBackcast_ld16_td5_ttd3_agF | 28.0747 | 26.9837 | 24.9566 | -3.1181 |      -11.1 |
| GenericAEBackcast_ld8_td5_ttd3_ag0  | 27.9854 | 26.686  | 24.9243 | -3.0611 |      -10.9 |
| GenericAEBackcast_ld8_td5_ttd5_ag0  | 27.8578 | 26.2731 | 25.106  | -2.7518 |       -9.9 |


## 7. Parameter Efficiency

| Config                              | Params    | Reduction   |   Med best_val_loss |
|:------------------------------------|:----------|:------------|--------------------:|
| GenericAE_ld8_td5_ttd5_agF          | 910,185   | 96.3%       |             25.026  |
| GenericAE_ld16_td5_ttd5_agF         | 940,985   | 96.2%       |             25.0989 |
| GenericAEBackcast_ld2_td5_ttd3_ag0  | 4,181,895 | 83.1%       |             25.3498 |
| GenericAEBackcast_ld2_td5_ttd3_agF  | 4,181,895 | 83.1%       |             25.0478 |
| GenericAEBackcast_ld2_td5_ttd5_ag0  | 4,184,465 | 83.1%       |             25.1064 |
| AutoEncoder_ld2_td5_ttd5_ag0        | 4,197,235 | 83.0%       |             25.7969 |
| GenericAEBackcast_ld8_td5_ttd3_ag0  | 4,228,155 | 82.9%       |             24.9243 |
| GenericAEBackcast_ld8_td5_ttd3_agF  | 4,228,155 | 82.9%       |             24.976  |
| GenericAEBackcast_ld8_td5_ttd5_agF  | 4,230,725 | 82.9%       |             25.2745 |
| GenericAEBackcast_ld8_td5_ttd5_ag0  | 4,230,725 | 82.9%       |             25.106  |
| AutoEncoder_ld8_td5_ttd5_ag0        | 4,243,345 | 82.8%       |             25.2183 |
| GenericAEBackcast_ld16_td5_ttd3_agF | 4,289,835 | 82.6%       |             24.9566 |
| GenericAEBackcast_ld16_td5_ttd3_ag0 | 4,289,835 | 82.6%       |             24.7623 |
| GenericAEBackcast_ld16_td5_ttd5_agF | 4,292,405 | 82.6%       |             25.2147 |
| GenericAEBackcast_ld16_td5_ttd5_ag0 | 4,292,405 | 82.6%       |             25.311  |


## 8. Final Verdict

Primary metric: median best_val_loss (lower is better).
OWA target checks are skipped because OWA is unavailable for this dataset.
Best final-round config: GenericAEBackcast_ld16_td5_ttd3_ag0 with median best_val_loss=24.7623.


## Dataset: weather

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/weather/trendae_study_results.csv`
- Rows: 384
- Primary metric: `best_val_loss`

### Abstract

This analysis covers 90 configurations over 3 rounds (384 runs).
Total training time: 448.9 min.
OWA unavailable; using best_val_loss throughout.


## 1. Successive Halving Funnel

|   Round |   Configs |   Runs |   Epochs |   Best Med best_val_loss |
|--------:|----------:|-------:|---------:|-------------------------:|
|       1 |        90 |    270 |       10 |                  42.9367 |
|       2 |        29 |     87 |       15 |                  42.9367 |
|       3 |         9 |     27 |       30 |                  42.7743 |


## 2.1 Round 1 Leaderboard

90 configs x 3 runs, up to 10 epochs each

|   Rank | Config                                 |   best_val_loss |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|----------------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAE_ld8_td5_ttd3_agF             |         42.9367 |   0.1266 |   65.35 |   0.9  | 1,967,455 | 59.0s  |          |
|      2 | GenericAE_ld16_td5_ttd5_agF            |         43.0544 |   0.8803 |   66.64 |   0.98 | 2,000,825 | 59.3s  |          |
|      3 | GenericAE_ld16_td5_ttd3_agF            |         43.1476 |   0.7414 |   66.1  |   1    | 1,998,255 | 59.1s  |          |
|      4 | AutoEncoderAE_ld8_td5_ttd3_ag0         |         43.3157 |   0.5339 |   66.14 |   1.17 | 2,012,440 | 62.5s  |          |
|      5 | AutoEncoder_ld8_td5_ttd3_ag0           |         43.3928 |   1.0205 |   66.96 |   1.56 | 5,537,515 | 60.8s  |          |
|      6 | GenericAE_ld8_td5_ttd3_ag0             |         43.4482 |   0.4807 |   66.39 |   1.18 | 1,967,455 | 57.4s  |          |
|      7 | AutoEncoderAE_ld2_td10_ttd3_ag0        |         43.4841 |   0.4783 |   66.29 |   1.14 | 2,027,765 | 63.1s  |          |
|      8 | GenericAEBackcast_ld16_td5_ttd3_ag0    |         43.5377 |   0.375  |   65.78 |   1.01 | 5,357,955 | 62.9s  |          |
|      9 | GenericAE_ld16_td5_ttd3_ag0            |         43.5405 |   0.4965 |   66.36 |   0.94 | 1,998,255 | 58.5s  |          |
|     10 | GenericAE_ld2_td5_ttd5_ag0             |         43.5418 |   1.8522 |   66.33 |   0.91 | 1,946,925 | 59.0s  |          |
|     11 | GenericAE_ld8_td5_ttd5_agF             |         43.6499 |   0.9956 |   64.86 |   1.13 | 1,970,025 | 59.3s  |          |
|     12 | AutoEncoderAE_ld8_td10_ttd3_ag0        |         43.6798 |   0.2901 |   66.2  |   1.05 | 2,050,865 | 62.4s  |          |
|     13 | GenericAE_ld16_td5_ttd5_ag0            |         43.7503 |   1.67   |   66.24 |   1.04 | 2,000,825 | 58.7s  |          |
|     14 | AutoEncoder_ld16_td5_ttd5_ag0          |         43.7536 |   0.527  |   66.04 |   1.17 | 5,601,565 | 60.5s  |          |
|     15 | BottleneckGenericAE_ld16_td10_ttd5_agF |         43.8358 |   0.7231 |   66.46 |   0.98 | 1,303,595 | 60.4s  |          |
|     16 | AutoEncoderAE_ld8_td5_ttd5_ag0         |         43.8449 |   0.4508 |   66.68 |   1.29 | 2,015,010 | 61.0s  |          |
|     17 | AutoEncoder_ld8_td5_ttd5_ag0           |         43.8645 |   1.0706 |   66.01 |   1.27 | 5,540,085 | 60.8s  |          |
|     18 | GenericAE_ld2_td5_ttd3_ag0             |         43.8908 |   0.1672 |   65.36 |   0.96 | 1,944,355 | 59.0s  |          |
|     19 | BottleneckGenericAE_ld16_td10_ttd3_agF |         43.9093 |   0.8968 |   65.91 |   0.95 | 1,301,025 | 59.4s  |          |
|     20 | GenericAE_ld8_td5_ttd5_ag0             |         43.9125 |   0.5372 |   66.78 |   1.09 | 1,970,025 | 58.0s  |          |
|     21 | AutoEncoderAE_ld16_td5_ttd5_ag0        |         43.9137 |   0.7736 |   64.96 |   1.27 | 2,045,810 | 61.0s  |          |
|     22 | AutoEncoderAE_ld16_td5_ttd3_ag0        |         43.9295 |   0.4786 |   65.91 |   1.03 | 2,043,240 | 62.5s  |          |
|     23 | AutoEncoderAE_ld8_td10_ttd5_ag0        |         43.9532 |   0.5806 |   66.19 |   1.46 | 2,053,435 | 61.0s  |          |
|     24 | AutoEncoder_ld2_td5_ttd3_ag0           |         43.9722 |   0.6053 |   65.38 |   1.13 | 5,491,405 | 69.0s  |          |
|     25 | AutoEncoderAE_ld2_td10_ttd5_ag0        |         43.984  |   0.2224 |   65.55 |   1.08 | 2,030,335 | 62.7s  |          |
|     26 | GenericAEBackcast_ld16_td5_ttd3_agF    |         43.9912 |   0.2876 |   65.57 |   1    | 5,357,955 | 61.5s  |          |
|     27 | GenericAEBackcast_ld16_td5_ttd5_ag0    |         44.0008 |   1.0263 |   66    |   1.16 | 5,360,525 | 62.8s  |          |
|     28 | AutoEncoder_ld16_td5_ttd3_ag0          |         44.0035 |   1.4384 |   66.08 |   1.14 | 5,598,995 | 60.6s  |          |
|     29 | AutoEncoderAE_ld16_td10_ttd3_ag0       |         44.0297 |   0.3568 |   65.92 |   1.14 | 2,081,665 | 62.1s  |          |
|     30 | BottleneckGenericAE_ld8_td10_ttd5_agF  |         44.0358 |   1.0054 |   65.59 |   0.96 | 1,272,795 | 60.5s  |          |
|     31 | BottleneckGenericAE_ld8_td10_ttd3_agF  |         44.0849 |   0.5145 |   66.56 |   0.97 | 1,270,225 | 60.4s  |          |
|     32 | BottleneckGenericAE_ld2_td10_ttd3_agF  |         44.1174 |   1.5168 |   65.81 |   0.98 | 1,247,125 | 59.8s  |          |
|     33 | GenericAEBackcastAE_ld16_td10_ttd3_ag0 |         44.1175 |   0.8875 |   65.52 |   0.94 | 1,809,500 | 60.2s  |          |
|     34 | GenericAEBackcast_ld16_td5_ttd5_agF    |         44.1281 |   0.1465 |   66.04 |   1.12 | 5,360,525 | 61.6s  |          |
|     35 | GenericAE_ld2_td5_ttd5_agF             |         44.1307 |   0.1097 |   66.85 |   1.08 | 1,946,925 | 58.7s  |          |
|     36 | AutoEncoderAE_ld16_td10_ttd5_ag0       |         44.1483 |   0.1202 |   66.67 |   1.06 | 2,084,235 | 62.2s  |          |
|     37 | GenericAEBackcastAE_ld16_td10_ttd5_ag0 |         44.1615 |   0.7055 |   65.67 |   1.03 | 1,812,070 | 60.8s  |          |
|     38 | GenericAEBackcastAE_ld2_td10_ttd3_agF  |         44.1684 |   0.2477 |   65.78 |   1.05 | 1,791,510 | 62.0s  |          |
|     39 | BottleneckGenericAE_ld16_td5_ttd5_agF  |         44.2508 |   0.5273 |   66.99 |   0.97 | 1,283,570 | 59.8s  |          |
|     40 | BottleneckGenericAE_ld16_td10_ttd5_ag0 |         44.2803 |   0.2936 |   66.27 |   1.03 | 1,303,595 | 60.5s  |          |
|     41 | GenericAEBackcast_ld8_td5_ttd5_agF     |         44.2829 |   0.1313 |   65.46 |   0.99 | 5,295,165 | 62.6s  |          |
|     42 | GenericAEBackcastAE_ld8_td10_ttd3_agF  |         44.2883 |   1.465  |   65.57 |   0.98 | 1,799,220 | 62.0s  |          |
|     43 | AutoEncoderAE_ld2_td5_ttd3_ag0         |         44.3486 |   0.3016 |   65.86 |   1.08 | 1,989,340 | 61.7s  |          |
|     44 | BottleneckGenericAE_ld8_td10_ttd5_ag0  |         44.3878 |   1.0154 |   65.22 |   0.93 | 1,272,795 | 58.6s  |          |
|     45 | GenericAEBackcast_ld8_td5_ttd5_ag0     |         44.3962 |   0.5399 |   64.97 |   0.99 | 5,295,165 | 62.5s  |          |
|     46 | BottleneckGenericAE_ld2_td10_ttd5_ag0  |         44.4224 |   1.6488 |   66.34 |   1.12 | 1,249,695 | 59.4s  |          |
|     47 | AutoEncoderAE_ld2_td5_ttd5_ag0         |         44.4384 |   0.8527 |   66.62 |   1.42 | 1,991,910 | 60.9s  |          |
|     48 | BottleneckGenericAE_ld16_td10_ttd3_ag0 |         44.4629 |   1.2794 |   66.32 |   1.03 | 1,301,025 | 60.4s  |          |
|     49 | BottleneckGenericAE_ld8_td10_ttd3_ag0  |         44.4926 |   0.2374 |   65.82 |   1    | 1,270,225 | 59.9s  |          |
|     50 | GenericAE_ld2_td5_ttd3_agF             |         44.4997 |   0.8622 |   65.82 |   1.02 | 1,944,355 | 58.9s  |          |
|     51 | GenericAEBackcastAE_ld8_td10_ttd5_agF  |         44.5099 |   0.427  |   65.67 |   1.05 | 1,801,790 | 61.2s  |          |
|     52 | GenericAEBackcast_ld8_td5_ttd3_ag0     |         44.5187 |   0.4744 |   65.76 |   0.97 | 5,292,595 | 62.3s  |          |
|     53 | BottleneckGenericAE_ld16_td5_ttd5_ag0  |         44.5224 |   0.7016 |   65.69 |   0.99 | 1,283,570 | 61.3s  |          |
|     54 | GenericAEBackcastAE_ld2_td10_ttd5_ag0  |         44.553  |   1.1622 |   66.06 |   1.19 | 1,794,080 | 61.2s  |          |
|     55 | GenericAEBackcast_ld8_td5_ttd3_agF     |         44.5554 |   0.2622 |   64.79 |   1    | 5,292,595 | 62.0s  |          |
|     56 | GenericAEBackcastAE_ld16_td10_ttd5_agF |         44.5768 |   0.6217 |   65.21 |   0.99 | 1,812,070 | 60.4s  |          |
|     57 | GenericAEBackcastAE_ld16_td10_ttd3_agF |         44.5772 |   0.8705 |   65.79 |   1.05 | 1,809,500 | 62.2s  |          |
|     58 | BottleneckGenericAE_ld8_td5_ttd5_ag0   |         44.5848 |   1.8287 |   65.84 |   1.03 | 1,252,770 | 59.2s  |          |
|     59 | BottleneckGenericAE_ld2_td10_ttd5_agF  |         44.5861 |   0.1585 |   66.44 |   0.94 | 1,249,695 | 60.0s  |          |
|     60 | GenericAEBackcastAE_ld8_td10_ttd5_ag0  |         44.6456 |   0.8849 |   66.39 |   0.96 | 1,801,790 | 60.1s  |          |
|     61 | AutoEncoder_ld2_td5_ttd5_ag0           |         44.6489 |   0.4231 |   66.25 |   1.03 | 5,493,975 | 62.8s  |          |
|     62 | GenericAEBackcastAE_ld8_td5_ttd5_agF   |         44.653  |   0.8893 |   65.05 |   0.82 | 1,760,940 | 62.0s  |          |
|     63 | BottleneckGenericAE_ld8_td5_ttd5_agF   |         44.6596 |   0.3844 |   65.62 |   1.05 | 1,252,770 | 61.6s  |          |
|     64 | GenericAEBackcastAE_ld2_td10_ttd3_ag0  |         44.696  |   0.5862 |   65.63 |   1.07 | 1,791,510 | 61.4s  |          |
|     65 | BottleneckGenericAE_ld16_td5_ttd3_ag0  |         44.759  |   0.3265 |   66.52 |   0.96 | 1,281,000 | 59.3s  |          |
|     66 | GenericAEBackcastAE_ld8_td10_ttd3_ag0  |         44.7863 |   0.5518 |   65.4  |   1.16 | 1,799,220 | 60.3s  |          |
|     67 | BottleneckGenericAE_ld2_td5_ttd5_ag0   |         44.8784 |   1.0414 |   67.24 |   1.21 | 1,229,670 | 58.8s  |          |
|     68 | BottleneckGenericAE_ld16_td5_ttd3_agF  |         44.8902 |   0.9898 |   66.09 |   0.97 | 1,281,000 | 60.0s  |          |
|     69 | GenericAEBackcastAE_ld2_td5_ttd5_ag0   |         44.9375 |   0.8972 |   65.07 |   1.05 | 1,753,230 | 60.5s  |          |
|     70 | GenericAEBackcastAE_ld2_td5_ttd5_agF   |         44.9583 |   0.9034 |   65.1  |   1.14 | 1,753,230 | 62.0s  |          |
|     71 | GenericAEBackcastAE_ld8_td5_ttd5_ag0   |         44.9714 |   0.6129 |   66.16 |   1.15 | 1,760,940 | 60.6s  |          |
|     72 | BottleneckGenericAE_ld8_td5_ttd3_ag0   |         44.98   |   0.7955 |   66.21 |   1    | 1,250,200 | 59.6s  |          |
|     73 | BottleneckGenericAE_ld8_td5_ttd3_agF   |         44.983  |   1.7648 |   66.1  |   0.94 | 1,250,200 | 59.6s  |          |
|     74 | GenericAEBackcastAE_ld8_td5_ttd3_agF   |         45.0741 |   0.9055 |   65.54 |   0.96 | 1,758,370 | 61.6s  |          |
|     75 | GenericAEBackcastAE_ld16_td5_ttd5_agF  |         45.1438 |   0.4547 |   65.94 |   0.97 | 1,771,220 | 61.2s  |          |
|     76 | GenericAEBackcastAE_ld2_td5_ttd3_ag0   |         45.2041 |   0.9981 |   66.12 |   1.22 | 1,750,660 | 61.6s  |          |
|     77 | BottleneckGenericAE_ld2_td5_ttd5_agF   |         45.2094 |   1.1549 |   66.24 |   1.04 | 1,229,670 | 59.5s  |          |
|     78 | GenericAEBackcastAE_ld2_td10_ttd5_agF  |         45.2211 |   0.7124 |   65.78 |   1.13 | 1,794,080 | 61.6s  |          |
|     79 | GenericAEBackcastAE_ld16_td5_ttd5_ag0  |         45.2314 |   0.5791 |   66.28 |   1.03 | 1,771,220 | 60.5s  |          |
|     80 | GenericAEBackcast_ld2_td5_ttd5_agF     |         45.3015 |   0.7011 |   65.31 |   0.97 | 5,246,145 | 61.7s  |          |
|     81 | GenericAEBackcastAE_ld16_td5_ttd3_agF  |         45.426  |   0.9012 |   65.66 |   1.01 | 1,768,650 | 62.1s  |          |
|     82 | BottleneckGenericAE_ld2_td10_ttd3_ag0  |         45.4629 |   1.2993 |   66.28 |   1.17 | 1,247,125 | 59.5s  |          |
|     83 | GenericAEBackcastAE_ld8_td5_ttd3_ag0   |         45.5854 |   0.5447 |   65.76 |   0.95 | 1,758,370 | 61.3s  |          |
|     84 | GenericAEBackcastAE_ld16_td5_ttd3_ag0  |         45.5903 |   0.8181 |   65.51 |   0.86 | 1,768,650 | 60.8s  |          |
|     85 | GenericAEBackcastAE_ld2_td5_ttd3_agF   |         45.6703 |   0.3765 |   66.1  |   1.15 | 1,750,660 | 61.4s  |          |
|     86 | GenericAEBackcast_ld2_td5_ttd3_ag0     |         45.8827 |   1.0003 |   65.84 |   1.07 | 5,243,575 | 63.3s  |          |
|     87 | GenericAEBackcast_ld2_td5_ttd5_ag0     |         45.9793 |   0.9591 |   66.58 |   1.34 | 5,246,145 | 63.1s  |          |
|     88 | BottleneckGenericAE_ld2_td5_ttd3_ag0   |         46.0706 |   1.3567 |   66.7  |   1.01 | 1,227,100 | 58.4s  |          |
|     89 | GenericAEBackcast_ld2_td5_ttd3_agF     |         46.0732 |   0.296  |   66.64 |   1.06 | 5,243,575 | 62.3s  |          |
|     90 | BottleneckGenericAE_ld2_td5_ttd3_agF   |         46.2062 |   1.4845 |   65.67 |   1.13 | 1,227,100 | 59.5s  |          |

Top config median best_val_loss=42.9367; worst=46.2062; delta=3.2695.


## 2.2 Round 2 Leaderboard

29 configs x 3 runs, up to 15 epochs each

|   Rank | Config                                 |   best_val_loss |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:---------------------------------------|----------------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAE_ld8_td5_ttd3_agF             |         42.9367 |   0.1266 |   65.35 |   0.9  | 1,967,455 | 84.5s  |          |
|      2 | GenericAE_ld16_td5_ttd5_agF            |         43.0544 |   0.8803 |   66.64 |   0.98 | 2,000,825 | 85.4s  |          |
|      3 | GenericAE_ld16_td5_ttd3_agF            |         43.1476 |   0.7414 |   66.1  |   1    | 1,998,255 | 80.0s  |          |
|      4 | GenericAEBackcast_ld16_td5_ttd5_ag0    |         43.2274 |   0.6536 |   66.18 |   0.96 | 5,360,525 | 89.5s  |          |
|      5 | AutoEncoderAE_ld16_td5_ttd5_ag0        |         43.2446 |   0.9215 |   64.96 |   1.14 | 2,045,810 | 82.6s  |          |
|      6 | AutoEncoderAE_ld8_td5_ttd3_ag0         |         43.3157 |   0.5339 |   66.14 |   1.17 | 2,012,440 | 88.0s  |          |
|      7 | AutoEncoderAE_ld2_td10_ttd3_ag0        |         43.3776 |   0.3699 |   66.12 |   1.12 | 2,027,765 | 87.5s  |          |
|      8 | AutoEncoder_ld8_td5_ttd3_ag0           |         43.3928 |   1.0205 |   66.96 |   1.56 | 5,537,515 | 81.0s  |          |
|      9 | AutoEncoder_ld16_td5_ttd3_ag0          |         43.4043 |   1.4384 |   65.75 |   1.1  | 5,598,995 | 87.1s  |          |
|     10 | GenericAE_ld8_td5_ttd3_ag0             |         43.4482 |   0.4807 |   66.39 |   1.18 | 1,967,455 | 78.7s  |          |
|     11 | AutoEncoderAE_ld16_td10_ttd3_ag0       |         43.5087 |   0.6012 |   65.19 |   1.18 | 2,081,665 | 88.5s  |          |
|     12 | GenericAEBackcast_ld16_td5_ttd3_ag0    |         43.5377 |   0.375  |   65.78 |   1.01 | 5,357,955 | 88.7s  |          |
|     13 | GenericAE_ld16_td5_ttd3_ag0            |         43.5405 |   0.6057 |   66.36 |   0.92 | 1,998,255 | 84.5s  |          |
|     14 | GenericAE_ld2_td5_ttd5_ag0             |         43.5418 |   1.8522 |   66.33 |   0.91 | 1,946,925 | 84.1s  |          |
|     15 | GenericAE_ld16_td5_ttd5_ag0            |         43.5542 |   1.67   |   65.62 |   0.99 | 2,000,825 | 84.8s  |          |
|     16 | AutoEncoder_ld16_td5_ttd5_ag0          |         43.5985 |   1.0243 |   65.71 |   1.12 | 5,601,565 | 87.1s  |          |
|     17 | AutoEncoderAE_ld8_td5_ttd5_ag0         |         43.6185 |   0.6558 |   66.68 |   1.3  | 2,015,010 | 87.9s  |          |
|     18 | GenericAE_ld8_td5_ttd5_agF             |         43.6499 |   0.9956 |   64.86 |   1.13 | 1,970,025 | 79.1s  |          |
|     19 | AutoEncoderAE_ld8_td10_ttd3_ag0        |         43.6798 |   0.1333 |   65.27 |   0.97 | 2,050,865 | 88.2s  |          |
|     20 | GenericAEBackcast_ld16_td5_ttd3_agF    |         43.6908 |   0.3764 |   65.2  |   0.95 | 5,357,955 | 88.8s  |          |
|     21 | GenericAE_ld2_td5_ttd3_ag0             |         43.7474 |   0.4736 |   65.07 |   0.96 | 1,944,355 | 83.9s  |          |
|     22 | AutoEncoderAE_ld8_td10_ttd5_ag0        |         43.8295 |   0.5806 |   66.19 |   1.15 | 2,053,435 | 88.1s  |          |
|     23 | BottleneckGenericAE_ld16_td10_ttd5_agF |         43.8358 |   0.587  |   66.12 |   1.05 | 1,303,595 | 85.7s  |          |
|     24 | AutoEncoder_ld8_td5_ttd5_ag0           |         43.8645 |   1.0706 |   66.01 |   1.27 | 5,540,085 | 86.4s  |          |
|     25 | AutoEncoderAE_ld16_td5_ttd3_ag0        |         43.8834 |   0.4786 |   65.79 |   1    | 2,043,240 | 88.4s  |          |
|     26 | GenericAE_ld8_td5_ttd5_ag0             |         43.8999 |   0.4957 |   66.78 |   1.09 | 1,970,025 | 84.7s  |          |
|     27 | BottleneckGenericAE_ld16_td10_ttd3_agF |         43.9093 |   0.8968 |   65.91 |   0.95 | 1,301,025 | 85.4s  |          |
|     28 | AutoEncoderAE_ld2_td10_ttd5_ag0        |         43.9636 |   0.9169 |   66.37 |   1.11 | 2,030,335 | 87.7s  |          |
|     29 | AutoEncoder_ld2_td5_ttd3_ag0           |         43.9722 |   0.6981 |   65.28 |   1.13 | 5,491,405 | 87.5s  |          |

Top config median best_val_loss=42.9367; worst=43.9722; delta=1.0355.


## 2.3 Round 3 Leaderboard

9 configs x 3 runs, up to 30 epochs each

|   Rank | Config                              |   best_val_loss |   Spread |   sMAPE |   MASE | Params    | Time   | Target   |
|-------:|:------------------------------------|----------------:|---------:|--------:|-------:|:----------|:-------|:---------|
|      1 | GenericAE_ld16_td5_ttd5_agF         |         42.7743 |   0.6878 |   65.81 |   0.89 | 2,000,825 | 97.9s  |          |
|      2 | GenericAE_ld8_td5_ttd3_agF          |         42.9363 |   0.0232 |   64.83 |   0.9  | 1,967,455 | 103.3s |          |
|      3 | AutoEncoderAE_ld2_td10_ttd3_ag0     |         43.0401 |   0.407  |   66    |   1.11 | 2,027,765 | 173.4s |          |
|      4 | GenericAE_ld16_td5_ttd3_agF         |         43.1476 |   0.7414 |   66.1  |   1    | 1,998,255 | 80.8s  |          |
|      5 | GenericAEBackcast_ld16_td5_ttd5_ag0 |         43.2274 |   0.4528 |   66.01 |   0.96 | 5,360,525 | 140.6s |          |
|      6 | AutoEncoderAE_ld16_td5_ttd5_ag0     |         43.2446 |   1.3736 |   64.96 |   1.11 | 2,045,810 | 82.3s  |          |
|      7 | AutoEncoderAE_ld8_td5_ttd3_ag0      |         43.3157 |   0.5339 |   66.14 |   1.17 | 2,012,440 | 104.0s |          |
|      8 | AutoEncoder_ld8_td5_ttd3_ag0        |         43.3928 |   1.0205 |   66.96 |   1.56 | 5,537,515 | 84.6s  |          |
|      9 | AutoEncoder_ld16_td5_ttd3_ag0       |         43.4043 |   1.4384 |   65.75 |   1.1  | 5,598,995 | 115.2s |          |

Top config median best_val_loss=42.7743; worst=43.4043; delta=0.6300.


## 3. Hyperparameter Marginals (Round 1)


### AE Variant

| Value               |   Med best_val_loss |   Mean best_val_loss |      Std |   N | Med Params   |
|:--------------------|--------------------:|---------------------:|---------:|----:|:-------------|
| GenericAE           |             43.6522 |              43.5666 | 0.540029 |  36 | 1,968,740    |
| AutoEncoderAE       |             43.9231 |              43.8914 | 0.375395 |  36 | 2,036,788    |
| AutoEncoder         |             43.9396 |              43.8733 | 0.593864 |  18 | 5,538,800    |
| GenericAEBackcast   |             44.4182 |              44.7219 | 0.912788 |  36 | 5,293,880    |
| BottleneckGenericAE |             44.5401 |              44.6488 | 0.733465 |  72 | 1,261,498    |
| GenericAEBackcastAE |             44.8977 |              44.8794 | 0.58751  |  72 | 1,781,365    |


### Latent Dim

|   Value |   Med best_val_loss |   Mean best_val_loss |      Std |   N | Med Params   |
|--------:|--------------------:|---------------------:|---------:|----:|:-------------|
|      16 |             44.1512 |              44.1951 | 0.713702 |  90 | 1,812,070    |
|       8 |             44.3081 |              44.2776 | 0.706074 |  90 | 1,801,790    |
|       2 |             44.6614 |              44.7964 | 0.874558 |  90 | 1,794,080    |


### Thetas Dim

|   Value |   Med best_val_loss |   Mean best_val_loss |      Std |   N | Med Params   |
|--------:|--------------------:|---------------------:|---------:|----:|:-------------|
|      10 |             44.2575 |              44.3101 | 0.552587 |  90 | 1,794,080    |
|       5 |             44.4433 |              44.4795 | 0.909307 | 180 | 1,968,740    |


### Trend Thetas Dim

|   Value |   Med best_val_loss |   Mean best_val_loss |      Std |   N | Med Params   |
|--------:|--------------------:|---------------------:|---------:|----:|:-------------|
|       5 |             44.2903 |              44.3511 | 0.688373 | 135 | 1,812,070    |
|       3 |             44.3986 |              44.495  | 0.914267 | 135 | 1,809,500    |


### active_g

| Value    |   Med best_val_loss |   Mean best_val_loss |      Std |   N | Med Params   |
|:---------|--------------------:|---------------------:|---------:|----:|:-------------|
| False    |             44.2377 |              44.3521 | 0.797023 | 162 | 1,968,740    |
| forecast |             44.4746 |              44.5295 | 0.823638 | 108 | 1,781,365    |


## 3b. Latent Dimension Discussion

|   latent_dim_cfg |   med_metric |   std_metric |   n |   med_params |
|-----------------:|-------------:|-------------:|----:|-------------:|
|               16 |      44.1512 |     0.713702 |  90 |  1.81207e+06 |
|                8 |      44.3081 |     0.706074 |  90 |  1.80179e+06 |
|                2 |      44.6614 |     0.874558 |  90 |  1.79408e+06 |

Best latent_dim=16 by median best_val_loss (44.1512).


## 4. Variant Head-to-Head


### Round 1 - Best Config per Variant

| Variant             | Best Config                            |   Med best_val_loss |
|:--------------------|:---------------------------------------|--------------------:|
| GenericAE           | GenericAE_ld8_td5_ttd3_agF             |             42.9367 |
| AutoEncoderAE       | AutoEncoderAE_ld8_td5_ttd3_ag0         |             43.3157 |
| AutoEncoder         | AutoEncoder_ld8_td5_ttd3_ag0           |             43.3928 |
| GenericAEBackcast   | GenericAEBackcast_ld16_td5_ttd3_ag0    |             43.5377 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd5_agF |             43.8358 |
| GenericAEBackcastAE | GenericAEBackcastAE_ld16_td10_ttd3_ag0 |             44.1175 |

### Round 2 - Best Config per Variant

| Variant             | Best Config                            |   Med best_val_loss |
|:--------------------|:---------------------------------------|--------------------:|
| GenericAE           | GenericAE_ld8_td5_ttd3_agF             |             42.9367 |
| GenericAEBackcast   | GenericAEBackcast_ld16_td5_ttd5_ag0    |             43.2274 |
| AutoEncoderAE       | AutoEncoderAE_ld16_td5_ttd5_ag0        |             43.2446 |
| AutoEncoder         | AutoEncoder_ld8_td5_ttd3_ag0           |             43.3928 |
| BottleneckGenericAE | BottleneckGenericAE_ld16_td10_ttd5_agF |             43.8358 |

### Round 3 - Best Config per Variant

| Variant           | Best Config                         |   Med best_val_loss |
|:------------------|:------------------------------------|--------------------:|
| GenericAE         | GenericAE_ld16_td5_ttd5_agF         |             42.7743 |
| AutoEncoderAE     | AutoEncoderAE_ld2_td10_ttd3_ag0     |             43.0401 |
| GenericAEBackcast | GenericAEBackcast_ld16_td5_ttd5_ag0 |             43.2274 |
| AutoEncoder       | AutoEncoder_ld8_td5_ttd3_ag0        |             43.3928 |


## 5. Stability Analysis


### Round 1

- Mean spread: 0.7399
- Max spread:  1.8522 (GenericAE_ld2_td5_ttd5_ag0)
- Mean std:    0.3867

### Round 2

- Mean spread: 0.7467
- Max spread:  1.8522 (GenericAE_ld2_td5_ttd5_ag0)
- Mean std:    0.3890

### Round 3

- Mean spread: 0.7421
- Max spread:  1.4384 (AutoEncoder_ld16_td5_ttd3_ag0)
- Mean std:    0.3801


## 6. Round-over-Round Progression

| config_name                         |      R1 |      R2 |      R3 |   Delta |   DeltaPct |
|:------------------------------------|--------:|--------:|--------:|--------:|-----------:|
| GenericAEBackcast_ld16_td5_ttd5_ag0 | 44.0008 | 43.2274 | 43.2274 | -0.7735 |       -1.8 |
| AutoEncoderAE_ld16_td5_ttd5_ag0     | 43.9137 | 43.2446 | 43.2446 | -0.6692 |       -1.5 |
| AutoEncoder_ld16_td5_ttd3_ag0       | 44.0035 | 43.4043 | 43.4043 | -0.5992 |       -1.4 |
| AutoEncoderAE_ld2_td10_ttd3_ag0     | 43.4841 | 43.3776 | 43.0401 | -0.444  |       -1   |
| GenericAE_ld16_td5_ttd5_agF         | 43.0544 | 43.0544 | 42.7743 | -0.2801 |       -0.7 |
| AutoEncoderAE_ld8_td5_ttd3_ag0      | 43.3157 | 43.3157 | 43.3157 |  0      |        0   |
| AutoEncoder_ld8_td5_ttd3_ag0        | 43.3928 | 43.3928 | 43.3928 |  0      |        0   |
| GenericAE_ld16_td5_ttd3_agF         | 43.1476 | 43.1476 | 43.1476 |  0      |        0   |
| GenericAE_ld8_td5_ttd3_agF          | 42.9367 | 42.9367 | 42.9363 | -0.0004 |       -0   |


## 7. Parameter Efficiency

| Config                              | Params    | Reduction   |   Med best_val_loss |
|:------------------------------------|:----------|:------------|--------------------:|
| GenericAE_ld8_td5_ttd3_agF          | 1,967,455 | 92.0%       |             42.9363 |
| GenericAE_ld16_td5_ttd3_agF         | 1,998,255 | 91.9%       |             43.1476 |
| GenericAE_ld16_td5_ttd5_agF         | 2,000,825 | 91.9%       |             42.7743 |
| AutoEncoderAE_ld8_td5_ttd3_ag0      | 2,012,440 | 91.9%       |             43.3157 |
| AutoEncoderAE_ld2_td10_ttd3_ag0     | 2,027,765 | 91.8%       |             43.0401 |
| AutoEncoderAE_ld16_td5_ttd5_ag0     | 2,045,810 | 91.7%       |             43.2446 |
| GenericAEBackcast_ld16_td5_ttd5_ag0 | 5,360,525 | 78.3%       |             43.2274 |
| AutoEncoder_ld8_td5_ttd3_ag0        | 5,537,515 | 77.6%       |             43.3928 |
| AutoEncoder_ld16_td5_ttd3_ag0       | 5,598,995 | 77.3%       |             43.4043 |


## 8. Final Verdict

Primary metric: median best_val_loss (lower is better).
OWA target checks are skipped because OWA is unavailable for this dataset.
Best final-round config: GenericAE_ld16_td5_ttd5_agF with median best_val_loss=42.7743.


# Summary

- analyzed_count: 3
- skipped_count: 0
- analyzed: ['m4', 'tourism', 'weather']
