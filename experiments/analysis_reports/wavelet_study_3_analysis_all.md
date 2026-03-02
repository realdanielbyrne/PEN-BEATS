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

[SKIP] dataset=tourism reason=missing_csv path=/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/tourism/wavelet_study_3_successive_results.csv

## Dataset: weather

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/weather/wavelet_study_3_successive_results.csv`
- Rows: 916
- Primary metric: `best_val_loss`


## 1. Overview & Data Summary

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/weather/wavelet_study_3_successive_results.csv`
- Total rows: 916
- Unique configs: 84
- Search rounds: [1, 2, 3]
- Primary metric: best_val_loss

|   Round |   Configs |   Rows | Epochs   | Passes          |
|--------:|----------:|-------:|:---------|:----------------|
|       1 |        84 |    502 | 7-7      | False, forecast |
|       2 |        43 |    258 | 12-15    | False, forecast |
|       3 |        26 |    156 | 12-45    | False, forecast |


## 2. Successive Halving Funnel

|   Round |   Configs |   Rows |   Best Med best_val_loss | Kept        |
|--------:|----------:|-------:|-------------------------:|:------------|
|       1 |        84 |    502 |                  43.0179 | -           |
|       2 |        43 |    258 |                  42.948  | 43/84 (51%) |
|       3 |        26 |    156 |                  42.7496 | 26/43 (60%) |


## 3. Round Leaderboards

### Round 1

| Config                       | Pass     |   best_val_loss |      Std |   sMAPE |   MASE |   Params |
|:-----------------------------|:---------|----------------:|---------:|--------:|-------:|---------:|
| Coif3_bd96_eq_fcast_ttd3     | False    |         42.7761 | nan      | 65.8384 | 0.8871 |  6163215 |
| DB4_bd96_eq_fcast_ttd3       | forecast |         42.9543 |   0.3149 | 65.1744 | 0.9762 |  6163215 |
| Symlet3_bd94_lt_fcast_ttd3   | forecast |         42.9652 |   0.3851 | 66.1725 | 0.9954 |  6152975 |
| DB2_bd94_lt_fcast_ttd5       | False    |         42.9878 |   0.5113 | 66.3156 | 1.0667 |  6155545 |
| Symlet10_bd96_eq_fcast_ttd3  | False    |         42.9940 |   0.1540 | 66.2599 | 1.0017 |  6163215 |
| DB3_bd96_eq_fcast_ttd3       | False    |         42.9976 |   0.6361 | 65.9169 | 1.0354 |  6163215 |
| Coif3_bd192_eq_bcast_ttd3    | forecast |         43.0069 |   0.3985 | 67.1998 | 1.0129 |  6408975 |
| Symlet2_bd94_lt_fcast_ttd5   | False    |         43.0080 |   0.3893 | 66.3954 | 1.1804 |  6155545 |
| Symlet2_bd96_eq_fcast_ttd5   | forecast |         43.0249 |   0.4272 | 67.4606 | 1.0922 |  6165785 |
| Coif3_bd94_lt_fcast_ttd3     | forecast |         43.0476 |   0.2211 | 66.2721 | 0.9553 |  6152975 |
| DB4_bd192_eq_bcast_ttd3      | forecast |         43.0498 |   0.3836 | 67.1557 | 1.1282 |  6408975 |
| Haar_bd192_eq_bcast_ttd3     | forecast |         43.0731 |   0.5760 | 66.9572 | 1.0294 |  6408975 |
| DB20_bd96_eq_fcast_ttd5      | False    |         43.0789 |   0.0180 | 67.0101 | 1.0162 |  6165785 |
| DB2_bd192_eq_bcast_ttd3      | forecast |         43.0961 |   0.4936 | 66.8599 | 1.0011 |  6408975 |
| Symlet10_bd192_eq_bcast_ttd3 | forecast |         43.1041 |   0.3561 | 66.5492 | 1.0298 |  6408975 |

### Round 2

| Config                      | Pass     |   best_val_loss |    Std |   sMAPE |   MASE |   Params |
|:----------------------------|:---------|----------------:|-------:|--------:|-------:|---------:|
| Coif3_bd94_lt_fcast_ttd3    | forecast |         42.7078 | 0.3632 | 65.5275 | 0.9413 |  6152975 |
| Symlet2_bd94_lt_fcast_ttd5  | False    |         42.7518 | 0.3624 | 66.1931 | 1.0099 |  6155545 |
| Symlet2_bd96_eq_fcast_ttd5  | forecast |         42.7648 | 0.2551 | 66.9592 | 1.0686 |  6165785 |
| Coif3_bd192_eq_bcast_ttd3   | forecast |         42.8563 | 0.4484 | 66.0644 | 0.9593 |  6408975 |
| Symlet3_bd94_lt_fcast_ttd3  | forecast |         42.8596 | 0.2645 | 66.0629 | 0.9758 |  6152975 |
| DB2_bd94_lt_fcast_ttd5      | False    |         42.8676 | 0.3301 | 66.5660 | 1.0232 |  6155545 |
| Coif3_bd192_eq_bcast_ttd5   | False    |         42.8976 | 0.2274 | 66.1378 | 0.9782 |  6411545 |
| DB4_bd96_eq_fcast_ttd5      | forecast |         42.9195 | 0.1300 | 65.6894 | 0.9911 |  6165785 |
| Coif10_bd192_eq_bcast_ttd5  | forecast |         42.9410 | 0.5775 | 65.4946 | 0.9966 |  6411545 |
| DB4_bd96_eq_fcast_ttd3      | forecast |         42.9411 | 0.2975 | 65.0895 | 0.9653 |  6163215 |
| Coif2_bd94_lt_fcast_ttd5    | forecast |         42.9470 | 0.1664 | 66.0194 | 0.9760 |  6155545 |
| Coif3_bd96_eq_fcast_ttd3    | forecast |         42.9496 | 0.1054 | 65.8484 | 0.9228 |  6163215 |
| DB2_bd192_eq_bcast_ttd3     | forecast |         42.9686 | 0.3391 | 67.0038 | 1.0321 |  6408975 |
| Symlet10_bd96_eq_fcast_ttd3 | False    |         42.9940 | 0.1540 | 66.2599 | 1.0017 |  6163215 |
| DB3_bd96_eq_fcast_ttd3      | False    |         42.9976 | 0.6361 | 65.9169 | 1.0354 |  6163215 |

### Round 3

| Config                     | Pass     |   best_val_loss |    Std |   sMAPE |   MASE |   Params |
|:---------------------------|:---------|----------------:|-------:|--------:|-------:|---------:|
| Symlet2_bd96_eq_fcast_ttd5 | forecast |         42.6159 | 0.0924 | 65.8915 | 0.9994 |  6165785 |
| Coif10_bd192_eq_bcast_ttd5 | forecast |         42.6881 | 0.2555 | 65.4560 | 0.9984 |  6411545 |
| Coif3_bd94_lt_fcast_ttd3   | forecast |         42.7078 | 0.3632 | 65.5275 | 0.9413 |  6152975 |
| DB20_bd96_eq_fcast_ttd5    | forecast |         42.7285 | 0.5205 | 65.8451 | 0.9891 |  6165785 |
| Coif3_bd94_lt_fcast_ttd5   | forecast |         42.7321 | 0.2422 | 65.2543 | 0.9512 |  6155545 |
| DB4_bd96_eq_fcast_ttd5     | forecast |         42.7516 | 0.2145 | 65.4820 | 1.0235 |  6165785 |
| Symlet2_bd94_lt_fcast_ttd5 | False    |         42.7518 | 0.3624 | 66.1931 | 1.0099 |  6155545 |
| Coif3_bd192_eq_bcast_ttd3  | forecast |         42.7641 | 0.3062 | 66.0233 | 0.9978 |  6408975 |
| Coif3_bd192_eq_bcast_ttd5  | forecast |         42.8050 | 0.3476 | 65.4177 | 0.9822 |  6411545 |
| Coif2_bd94_lt_fcast_ttd5   | forecast |         42.8103 | 0.2247 | 64.1343 | 0.9956 |  6155545 |
| Coif3_bd96_eq_fcast_ttd3   | forecast |         42.8209 | 0.1080 | 65.6941 | 0.9833 |  6163215 |
| Symlet3_bd94_lt_fcast_ttd3 | forecast |         42.8596 | 0.2645 | 66.0629 | 0.9758 |  6152975 |
| DB2_bd94_lt_fcast_ttd5     | False    |         42.8676 | 0.3301 | 66.5660 | 1.0232 |  6155545 |
| Symlet2_bd94_lt_fcast_ttd5 | forecast |         42.8747 | 0.4999 | 66.2156 | 0.9749 |  6155545 |
| Coif3_bd192_eq_bcast_ttd5  | False    |         42.8976 | 0.2274 | 66.1378 | 0.9782 |  6411545 |


## 4. Hyperparameter Marginals (Round 1)

### wavelet

| Value    |   Mean best_val_loss |    Std |   N |
|:---------|---------------------:|-------:|----:|
| Coif3    |              43.2920 | 0.3298 |  34 |
| Symlet2  |              43.3239 | 0.3909 |  36 |
| DB20     |              43.3747 | 0.3550 |  36 |
| DB4      |              43.4001 | 0.3961 |  36 |
| DB2      |              43.4330 | 0.4135 |  36 |
| DB3      |              43.4371 | 0.3966 |  36 |
| Coif2    |              43.4462 | 0.4041 |  36 |
| DB10     |              43.4475 | 0.3907 |  36 |
| Haar     |              43.4501 | 0.3983 |  36 |
| Symlet3  |              43.4521 | 0.3683 |  36 |
| Coif1    |              43.4738 | 0.3566 |  36 |
| Symlet20 |              43.4961 | 0.3448 |  36 |
| Coif10   |              43.5071 | 0.3550 |  36 |
| Symlet10 |              43.5434 | 0.4301 |  36 |

### bd_label

| Value    |   Mean best_val_loss |    Std |   N |
|:---------|---------------------:|-------:|----:|
| eq_bcast |              43.4208 | 0.3871 | 168 |
| eq_fcast |              43.4242 | 0.4027 | 166 |
| lt_fcast |              43.4587 | 0.3579 | 168 |

### ttd

|   Value |   Mean best_val_loss |    Std |        N |
|--------:|---------------------:|-------:|---------:|
|  3.0000 |              43.3752 | 0.3655 | 250.0000 |
|  5.0000 |              43.4936 | 0.3907 | 252.0000 |

### active_g_cfg

| Value    |   Mean best_val_loss |    Std |   N |
|:---------|---------------------:|-------:|----:|
| forecast |              43.4235 | 0.3960 | 252 |
| False    |              43.4459 | 0.3690 | 250 |


## 5. Stability Analysis

### Round 1

- Mean spread: 0.9755
- Max spread: 1.7812 (Symlet20_bd192_eq_bcast_ttd5)
- Mean std: 0.3643

### Round 2

- Mean spread: 0.8205
- Max spread: 1.5711 (Haar_bd192_eq_bcast_ttd3)
- Mean std: 0.3110

### Round 3

- Mean spread: 0.8666
- Max spread: 1.5711 (Haar_bd192_eq_bcast_ttd3)
- Mean std: 0.3293


## 6. Round-over-Round Progression

| config_name                  |      R1 |      R2 |      R3 |   Delta |   DeltaPct |
|:-----------------------------|--------:|--------:|--------:|--------:|-----------:|
| Symlet2_bd94_lt_fcast_ttd5   | 43.4426 | 42.9611 | 42.8855 | -0.5571 |       -1.3 |
| Symlet2_bd96_eq_fcast_ttd5   | 43.2689 | 43.1061 | 42.7496 | -0.5192 |       -1.2 |
| Coif10_bd192_eq_bcast_ttd5   | 43.3552 | 43.1519 | 42.9164 | -0.4388 |       -1   |
| Symlet3_bd94_lt_fcast_ttd5   | 43.5019 | 43.4315 | 43.0803 | -0.4216 |       -1   |
| Coif3_bd192_eq_bcast_ttd5    | 43.2953 | 43.0043 | 42.9005 | -0.3947 |       -0.9 |
| Coif2_bd94_lt_fcast_ttd5     | 43.3535 | 43.0421 | 42.9716 | -0.3818 |       -0.9 |
| DB4_bd96_eq_fcast_ttd5       | 43.4729 | 43.2152 | 43.1594 | -0.3135 |       -0.7 |
| DB2_bd94_lt_fcast_ttd5       | 43.3273 | 43.1469 | 43.0275 | -0.2998 |       -0.7 |
| DB10_bd96_eq_fcast_ttd5      | 43.3357 | 43.1751 | 43.0727 | -0.2629 |       -0.6 |
| DB3_bd96_eq_fcast_ttd3       | 43.1882 | 42.948  | 42.948  | -0.2402 |       -0.6 |
| Coif3_bd96_eq_fcast_ttd3     | 43.0964 | 43.0005 | 42.8633 | -0.2331 |       -0.5 |
| Haar_bd94_lt_fcast_ttd3      | 43.5407 | 43.3261 | 43.3261 | -0.2146 |       -0.5 |
| Coif3_bd94_lt_fcast_ttd5     | 43.2831 | 43.2531 | 43.0617 | -0.2214 |       -0.5 |
| DB4_bd192_eq_bcast_ttd3      | 43.1889 | 42.9965 | 42.9965 | -0.1924 |       -0.4 |
| Symlet20_bd192_eq_bcast_ttd3 | 43.3397 | 43.1896 | 43.1896 | -0.1501 |       -0.3 |
| Symlet10_bd96_eq_fcast_ttd3  | 43.0658 | 43.0658 | 42.9256 | -0.1401 |       -0.3 |
| Symlet3_bd94_lt_fcast_ttd3   | 43.1633 | 43.0123 | 43.0123 | -0.151  |       -0.3 |
| Haar_bd192_eq_bcast_ttd3     | 43.2728 | 43.1234 | 43.1234 | -0.1493 |       -0.3 |
| DB2_bd96_eq_fcast_ttd3       | 43.2012 | 43.1511 | 43.1511 | -0.0501 |       -0.1 |
| Coif2_bd96_eq_fcast_ttd5     | 43.2519 | 43.2112 | 43.2112 | -0.0407 |       -0.1 |
| Coif3_bd94_lt_fcast_ttd3     | 43.2994 | 43.2417 | 43.2417 | -0.0576 |       -0.1 |
| DB20_bd96_eq_fcast_ttd5      | 43.1093 | 43.1093 | 43.0714 | -0.0378 |       -0.1 |
| Coif3_bd192_eq_bcast_ttd3    | 43.0179 | 43.0179 | 43.0094 | -0.0085 |       -0   |
| DB20_bd192_eq_bcast_ttd3     | 43.3184 | 43.3184 | 43.3184 |  0      |        0   |
| DB3_bd94_lt_fcast_ttd3       | 43.1963 | 43.1963 | 43.1963 |  0      |        0   |
| DB4_bd96_eq_fcast_ttd3       | 43.1238 | 43.1042 | 43.1042 | -0.0197 |       -0   |


## 7. Baseline Comparisons

Section skipped (M4-specific baseline references).


## 8. Final Verdict

Best configuration: Symlet2_bd96_eq_fcast_ttd5 (pass=forecast) with median best_val_loss=42.5936.
Primary metric: best_val_loss (lower is better). OWA-based baseline comparisons are not applicable.

| Config                     | Pass     |   Med best_val_loss |    Std |   Params |   sMAPE |   MASE |
|:---------------------------|:---------|--------------------:|-------:|---------:|--------:|-------:|
| Symlet2_bd96_eq_fcast_ttd5 | forecast |             42.5936 | 0.0924 |  6165785 | 65.4735 | 0.9896 |
| Coif3_bd94_lt_fcast_ttd3   | forecast |             42.5980 | 0.3632 |  6152975 | 65.5643 | 0.9078 |
| Coif2_bd94_lt_fcast_ttd5   | forecast |             42.7240 | 0.2247 |  6155545 | 64.7636 | 0.9507 |
| Coif10_bd192_eq_bcast_ttd5 | forecast |             42.7341 | 0.2555 |  6411545 | 65.5744 | 1.0061 |
| DB3_bd96_eq_fcast_ttd3     | False    |             42.7629 | 0.6361 |  6163215 | 65.9330 | 1.0013 |
| DB4_bd96_eq_fcast_ttd5     | forecast |             42.7760 | 0.2145 |  6165785 | 65.6613 | 0.9730 |
| Coif3_bd96_eq_fcast_ttd3   | forecast |             42.7842 | 0.1080 |  6163215 | 65.6532 | 0.9562 |
| Coif3_bd192_eq_bcast_ttd3  | forecast |             42.7846 | 0.3062 |  6408975 | 66.0135 | 1.0072 |
| Symlet2_bd94_lt_fcast_ttd5 | forecast |             42.8101 | 0.4999 |  6155545 | 66.2385 | 0.9700 |
| Coif3_bd94_lt_fcast_ttd5   | forecast |             42.8155 | 0.2422 |  6155545 | 65.4135 | 0.9888 |


# Summary

- analyzed_count: 2
- skipped_count: 1
- analyzed: ['m4', 'weather']
- skipped:
  - dataset=tourism reason=missing_csv path=/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/tourism/wavelet_study_3_successive_results.csv
