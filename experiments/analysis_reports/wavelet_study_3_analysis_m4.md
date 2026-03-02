# Wavelet Study 3 - Multi-Dataset Analysis


## Dataset: m4

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/wavelet_study_3_successive_results.csv`
- Rows: 1194
- Primary metric: `owa`


## 1. Overview & Data Summary

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/wavelet_study_3_successive_results.csv`
- Total rows: 1194
- Unique configs: 112
- Search rounds: [1, 2, 3]
- Primary metric: owa

|   Round |   Configs |   Rows | Epochs   | Passes          |
|--------:|----------:|-------:|:---------|:----------------|
|       1 |       112 |    672 | 7-7      | False, forecast |
|       2 |        58 |    348 | 15-15    | False, forecast |
|       3 |        29 |    174 | 16-30    | False, forecast |


## 2. Successive Halving Funnel

|   Round |   Configs |   Rows |   Best Med OWA | Kept         |
|--------:|----------:|-------:|---------------:|:-------------|
|       1 |       112 |    672 |         0.8837 | -            |
|       2 |        58 |    348 |         0.8222 | 58/112 (52%) |
|       3 |        29 |    174 |         0.8057 | 29/58 (50%)  |


## 3. Round Leaderboards

### Round 1

| Config                     | Pass     |    OWA |    Std |   sMAPE |   MASE |   Params |
|:---------------------------|:---------|-------:|-------:|--------:|-------:|---------:|
| Coif3_bd6_eq_fcast_ttd3    | False    | 0.8853 | 0.0213 | 14.8221 | 3.4319 |  5080335 |
| Symlet20_bd6_eq_fcast_ttd3 | False    | 0.8854 | 0.0153 | 14.7708 | 3.4452 |  5080335 |
| Symlet10_bd6_eq_fcast_ttd3 | False    | 0.8882 | 0.0068 | 14.9363 | 3.4275 |  5080335 |
| Symlet3_bd15_lt_bcast_ttd3 | forecast | 0.8895 | 0.0135 | 14.8725 | 3.4532 |  5103375 |
| DB20_bd15_lt_bcast_ttd5    | False    | 0.8904 | 0.0272 | 14.8841 | 3.4573 |  5105945 |
| Symlet20_bd4_lt_fcast_ttd3 | False    | 0.8923 | 0.0199 | 14.9136 | 3.4655 |  5070095 |
| Symlet20_bd4_lt_fcast_ttd5 | False    | 0.8994 | 0.0124 | 15.0602 | 3.4860 |  5072665 |
| DB3_bd6_eq_fcast_ttd3      | False    | 0.8997 | 0.0463 | 14.9600 | 3.5133 |  5080335 |
| Symlet20_bd6_eq_fcast_ttd5 | False    | 0.9013 | 0.0170 | 15.0470 | 3.5041 |  5082905 |
| Coif3_bd30_eq_bcast_ttd3   | False    | 0.9017 | 0.0207 | 15.0743 | 3.5012 |  5141775 |
| DB3_bd4_lt_fcast_ttd3      | False    | 0.9041 | 0.0044 | 15.1183 | 3.5094 |  5070095 |
| Symlet3_bd15_lt_bcast_ttd3 | False    | 0.9053 | 0.0379 | 15.0808 | 3.5280 |  5103375 |
| DB3_bd6_eq_fcast_ttd5      | forecast | 0.9087 | 0.0171 | 15.1759 | 3.5320 |  5082905 |
| DB3_bd15_lt_bcast_ttd5     | False    | 0.9092 | 0.0034 | 15.2774 | 3.5114 |  5105945 |
| DB4_bd15_lt_bcast_ttd3     | False    | 0.9098 | 0.0132 | 15.1444 | 3.5481 |  5103375 |

### Round 2

| Config                    | Pass     |    OWA |    Std |   sMAPE |   MASE |   Params |
|:--------------------------|:---------|-------:|-------:|--------:|-------:|---------:|
| DB3_bd30_eq_bcast_ttd5    | False    | 0.8189 | 0.0185 | 13.8125 | 3.1499 |  5144345 |
| Coif2_bd6_eq_fcast_ttd3   | False    | 0.8200 | 0.0103 | 13.8477 | 3.1498 |  5080335 |
| DB20_bd30_eq_bcast_ttd3   | forecast | 0.8252 | 0.0140 | 13.8954 | 3.1793 |  5141775 |
| Coif1_bd15_lt_bcast_ttd3  | False    | 0.8252 | 0.0210 | 13.8865 | 3.1821 |  5103375 |
| Coif3_bd30_eq_bcast_ttd3  | forecast | 0.8263 | 0.0058 | 13.8608 | 3.1964 |  5141775 |
| Coif3_bd4_lt_fcast_ttd3   | forecast | 0.8265 | 0.0154 | 13.9086 | 3.1868 |  5070095 |
| Symlet2_bd4_lt_fcast_ttd3 | False    | 0.8285 | 0.0041 | 13.9259 | 3.1982 |  5070095 |
| Symlet3_bd4_lt_fcast_ttd3 | forecast | 0.8292 | 0.0156 | 13.9258 | 3.2040 |  5070095 |
| Coif2_bd15_lt_bcast_ttd3  | forecast | 0.8317 | 0.0144 | 13.9215 | 3.2246 |  5103375 |
| Symlet3_bd4_lt_fcast_ttd3 | False    | 0.8351 | 0.0252 | 13.9763 | 3.2383 |  5070095 |
| Coif3_bd15_lt_bcast_ttd5  | forecast | 0.8357 | 0.0146 | 14.0221 | 3.2325 |  5105945 |
| DB20_bd6_eq_fcast_ttd3    | False    | 0.8360 | 0.0420 | 14.0616 | 3.2248 |  5080335 |
| Haar_bd30_eq_bcast_ttd3   | False    | 0.8370 | 0.0140 | 14.0178 | 3.2433 |  5141775 |
| DB2_bd4_lt_fcast_ttd3     | False    | 0.8370 | 0.0077 | 14.0078 | 3.2462 |  5070095 |
| Haar_bd30_eq_bcast_ttd3   | forecast | 0.8383 | 0.0222 | 14.0281 | 3.2511 |  5141775 |

### Round 3

| Config                     | Pass     |    OWA |    Std |   sMAPE |   MASE |   Params |
|:---------------------------|:---------|-------:|-------:|--------:|-------:|---------:|
| Haar_bd4_lt_fcast_ttd3     | forecast | 0.8061 | 0.0043 | 13.5562 | 3.1102 |  5070095 |
| DB20_bd15_lt_bcast_ttd3    | forecast | 0.8062 | 0.0119 | 13.5729 | 3.1069 |  5103375 |
| Coif2_bd4_lt_fcast_ttd3    | forecast | 0.8079 | 0.0205 | 13.5972 | 3.1144 |  5070095 |
| DB20_bd30_eq_bcast_ttd3    | forecast | 0.8088 | 0.0100 | 13.5808 | 3.1255 |  5141775 |
| DB20_bd15_lt_bcast_ttd3    | False    | 0.8107 | 0.0116 | 13.6108 | 3.1340 |  5103375 |
| Coif2_bd4_lt_fcast_ttd3    | False    | 0.8115 | 0.0089 | 13.6448 | 3.1320 |  5070095 |
| Coif3_bd4_lt_fcast_ttd3    | False    | 0.8119 | 0.0142 | 13.6398 | 3.1358 |  5070095 |
| Coif3_bd30_eq_bcast_ttd3   | forecast | 0.8144 | 0.0225 | 13.6513 | 3.1531 |  5141775 |
| Coif3_bd6_eq_fcast_ttd3    | False    | 0.8147 | 0.0109 | 13.7142 | 3.1404 |  5080335 |
| Symlet20_bd4_lt_fcast_ttd3 | forecast | 0.8147 | 0.0098 | 13.6967 | 3.1448 |  5070095 |
| Symlet10_bd6_eq_fcast_ttd5 | False    | 0.8161 | 0.0075 | 13.7148 | 3.1512 |  5082905 |
| DB10_bd15_lt_bcast_ttd3    | False    | 0.8162 | 0.0117 | 13.6697 | 3.1629 |  5103375 |
| Coif2_bd30_eq_bcast_ttd3   | False    | 0.8164 | 0.0230 | 13.7254 | 3.1510 |  5141775 |
| Symlet2_bd4_lt_fcast_ttd3  | forecast | 0.8171 | 0.0121 | 13.6929 | 3.1643 |  5070095 |
| Symlet3_bd4_lt_fcast_ttd3  | forecast | 0.8173 | 0.0115 | 13.7255 | 3.1581 |  5070095 |


## 4. Hyperparameter Marginals (Round 1)

### wavelet

| Value    |   Mean OWA |    Std |   N |
|:---------|-----------:|-------:|----:|
| DB4      |     0.9599 | 0.0755 |  48 |
| Symlet20 |     0.9628 | 0.0963 |  48 |
| Coif3    |     0.9682 | 0.0801 |  48 |
| Symlet10 |     0.9781 | 0.0910 |  48 |
| DB2      |     0.9816 | 0.0863 |  48 |
| Coif2    |     0.9822 | 0.0890 |  48 |
| Symlet2  |     0.9826 | 0.1018 |  48 |
| Symlet3  |     0.9861 | 0.1117 |  48 |
| Coif1    |     0.9869 | 0.1064 |  48 |
| Coif10   |     0.9875 | 0.0849 |  48 |
| Haar     |     0.9946 | 0.0976 |  48 |
| DB20     |     0.9959 | 0.1243 |  48 |
| DB10     |     1.0018 | 0.1032 |  48 |
| DB3      |     1.0164 | 0.1405 |  48 |

### bd_label

| Value    |   Mean OWA |    Std |   N |
|:---------|-----------:|-------:|----:|
| lt_bcast |     0.9759 | 0.0982 | 168 |
| eq_fcast |     0.9780 | 0.0980 | 168 |
| lt_fcast |     0.9811 | 0.0918 | 168 |
| eq_bcast |     1.0034 | 0.1122 | 168 |

### ttd

|   Value |   Mean OWA |    Std |        N |
|--------:|-----------:|-------:|---------:|
|  3.0000 |     0.9768 | 0.1005 | 336.0000 |
|  5.0000 |     0.9924 | 0.1005 | 336.0000 |

### active_g_cfg

| Value    |   Mean OWA |    Std |   N |
|:---------|-----------:|-------:|----:|
| False    |     0.9812 | 0.0999 | 336 |
| forecast |     0.9880 | 0.1015 | 336 |


## 5. Stability Analysis

### Round 1

- Mean spread: 0.2355
- Max spread: 0.6115 (DB20_bd30_eq_bcast_ttd5)
- Mean std: 0.0903

### Round 2

- Mean spread: 0.1096
- Max spread: 0.2511 (Symlet20_bd15_lt_bcast_ttd3)
- Mean std: 0.0415

### Round 3

- Mean spread: 0.0656
- Max spread: 0.1163 (DB20_bd30_eq_bcast_ttd3)
- Mean std: 0.0252


## 6. Round-over-Round Progression

| config_name                |     R1 |     R2 |     R3 |   Delta |   DeltaPct |
|:---------------------------|-------:|-------:|-------:|--------:|-----------:|
| Haar_bd4_lt_fcast_ttd3     | 0.9706 | 0.878  | 0.8079 | -0.1627 |      -16.8 |
| DB2_bd4_lt_fcast_ttd3      | 0.9913 | 0.844  | 0.8297 | -0.1617 |      -16.3 |
| Symlet3_bd6_eq_fcast_ttd3  | 0.982  | 0.8797 | 0.8258 | -0.1561 |      -15.9 |
| Symlet3_bd4_lt_fcast_ttd3  | 0.9603 | 0.8222 | 0.8111 | -0.1493 |      -15.5 |
| Coif2_bd30_eq_bcast_ttd3   | 0.9734 | 0.8592 | 0.8248 | -0.1486 |      -15.3 |
| DB20_bd15_lt_bcast_ttd3    | 0.9455 | 0.8967 | 0.8057 | -0.1398 |      -14.8 |
| DB20_bd30_eq_bcast_ttd3    | 0.9482 | 0.8481 | 0.8143 | -0.1339 |      -14.1 |
| Symlet10_bd6_eq_fcast_ttd5 | 0.9536 | 0.8469 | 0.8196 | -0.1339 |      -14   |
| DB4_bd6_eq_fcast_ttd3      | 0.9743 | 0.8601 | 0.8399 | -0.1344 |      -13.8 |
| DB2_bd30_eq_bcast_ttd3     | 0.9666 | 0.8821 | 0.8367 | -0.1299 |      -13.4 |
| Haar_bd6_eq_fcast_ttd3     | 0.9405 | 0.8926 | 0.8183 | -0.1221 |      -13   |
| Coif1_bd6_eq_fcast_ttd3    | 0.9583 | 0.8534 | 0.8406 | -0.1177 |      -12.3 |
| Coif3_bd30_eq_bcast_ttd3   | 0.9286 | 0.8411 | 0.8149 | -0.1136 |      -12.2 |
| Coif2_bd4_lt_fcast_ttd3    | 0.9183 | 0.8267 | 0.8072 | -0.1111 |      -12.1 |
| DB10_bd15_lt_bcast_ttd3    | 0.9422 | 0.8625 | 0.8289 | -0.1133 |      -12   |
| Haar_bd30_eq_bcast_ttd3    | 0.9369 | 0.8338 | 0.8276 | -0.1093 |      -11.7 |
| Coif2_bd6_eq_fcast_ttd3    | 0.9457 | 0.8382 | 0.8358 | -0.1099 |      -11.6 |
| Coif2_bd15_lt_bcast_ttd3   | 0.9669 | 0.8391 | 0.8561 | -0.1107 |      -11.5 |
| Symlet2_bd4_lt_fcast_ttd3  | 0.9194 | 0.8306 | 0.8225 | -0.0969 |      -10.5 |
| Coif1_bd15_lt_bcast_ttd3   | 0.9144 | 0.856  | 0.8204 | -0.094  |      -10.3 |
| Coif1_bd4_lt_fcast_ttd3    | 0.9188 | 0.8478 | 0.8242 | -0.0946 |      -10.3 |
| DB3_bd6_eq_fcast_ttd3      | 0.935  | 0.8333 | 0.8456 | -0.0895 |       -9.6 |
| Coif3_bd6_eq_fcast_ttd3    | 0.9025 | 0.8861 | 0.8187 | -0.0838 |       -9.3 |
| Symlet20_bd4_lt_fcast_ttd3 | 0.911  | 0.8616 | 0.8308 | -0.0802 |       -8.8 |
| DB4_bd15_lt_bcast_ttd3     | 0.9302 | 0.8802 | 0.849  | -0.0812 |       -8.7 |
| DB2_bd15_lt_bcast_ttd3     | 0.9189 | 0.8669 | 0.8463 | -0.0726 |       -7.9 |
| DB3_bd4_lt_fcast_ttd3      | 0.902  | 0.8462 | 0.8321 | -0.0699 |       -7.8 |
| Coif3_bd4_lt_fcast_ttd3    | 0.8837 | 0.8407 | 0.8271 | -0.0566 |       -6.4 |
| Symlet20_bd6_eq_fcast_ttd3 | 0.8994 | 0.8742 | 0.8485 | -0.051  |       -5.7 |


## 7. Baseline Comparisons

| Config                     | Pass     |    OWA |   sMAPE |   Params |   vs NBEATS-I+G |
|:---------------------------|:---------|-------:|--------:|---------:|----------------:|
| Haar_bd4_lt_fcast_ttd3     | forecast | 0.8061 | 13.5562 |  5070095 |          0.0004 |
| DB20_bd15_lt_bcast_ttd3    | forecast | 0.8062 | 13.5729 |  5103375 |          0.0005 |
| Coif2_bd4_lt_fcast_ttd3    | forecast | 0.8079 | 13.5972 |  5070095 |          0.0022 |
| DB20_bd30_eq_bcast_ttd3    | forecast | 0.8088 | 13.5808 |  5141775 |          0.0031 |
| DB20_bd15_lt_bcast_ttd3    | False    | 0.8107 | 13.6108 |  5103375 |          0.0050 |
| Coif2_bd4_lt_fcast_ttd3    | False    | 0.8115 | 13.6448 |  5070095 |          0.0058 |
| Coif3_bd4_lt_fcast_ttd3    | False    | 0.8119 | 13.6398 |  5070095 |          0.0062 |
| Coif3_bd30_eq_bcast_ttd3   | forecast | 0.8144 | 13.6513 |  5141775 |          0.0087 |
| Coif3_bd6_eq_fcast_ttd3    | False    | 0.8147 | 13.7142 |  5080335 |          0.0090 |
| Symlet20_bd4_lt_fcast_ttd3 | forecast | 0.8147 | 13.6967 |  5070095 |          0.0090 |

| Baseline    |    OWA |   sMAPE |   Params |
|:------------|-------:|--------:|---------:|
| AE+Trend    | 0.8015 | 13.5300 |  5200000 |
| NBEATS-I+G  | 0.8057 | 13.5300 | 35900000 |
| GenericAE   | 0.8063 | 13.5700 |  4800000 |
| AutoEncoder | 0.8075 | 13.5600 | 24900000 |
| NBEATS-I    | 0.8132 | 13.6700 | 12900000 |
| NBEATS-G    | 0.8198 | 13.7000 | 24700000 |

Loaded M4 block baseline CSV for reference.
Loaded M4 AE+Trend CSV for reference.


## 8. Final Verdict

Best configuration: Coif2_bd4_lt_fcast_ttd3 (pass=forecast) with median OWA=0.7966.
vs NBEATS-I+G (0.8057): beats (delta=-0.0091).

| Config                    | Pass     |   Med OWA |    Std |   Params |   sMAPE |   MASE |
|:--------------------------|:---------|----------:|-------:|---------:|--------:|-------:|
| Coif2_bd4_lt_fcast_ttd3   | forecast |    0.7966 | 0.0205 |  5070095 | 13.4863 | 3.0519 |
| DB20_bd15_lt_bcast_ttd3   | forecast |    0.8041 | 0.0119 |  5103375 | 13.5345 | 3.0993 |
| Coif2_bd15_lt_bcast_ttd3  | forecast |    0.8044 | 0.0630 |  5103375 | 13.5465 | 3.0990 |
| Symlet3_bd4_lt_fcast_ttd3 | False    |    0.8044 | 0.0280 |  5070095 | 13.5877 | 3.0894 |
| Coif3_bd4_lt_fcast_ttd3   | False    |    0.8047 | 0.0142 |  5070095 | 13.5878 | 3.0918 |
| Haar_bd6_eq_fcast_ttd3    | forecast |    0.8048 | 0.0308 |  5080335 | 13.5736 | 3.0957 |
| Haar_bd4_lt_fcast_ttd3    | forecast |    0.8049 | 0.0043 |  5070095 | 13.5606 | 3.0999 |
| Coif1_bd4_lt_fcast_ttd3   | False    |    0.8063 | 0.0482 |  5070095 | 13.6037 | 3.1000 |
| DB4_bd15_lt_bcast_ttd3    | forecast |    0.8068 | 0.0597 |  5103375 | 13.6096 | 3.1028 |
| Coif2_bd30_eq_bcast_ttd3  | False    |    0.8070 | 0.0230 |  5141775 | 13.5777 | 3.1125 |


# Summary

- analyzed_count: 1
- skipped_count: 0
- analyzed: ['m4']
