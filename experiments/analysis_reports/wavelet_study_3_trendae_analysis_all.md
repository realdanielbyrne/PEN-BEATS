# Wavelet Study 3 TrendAE - Multi-Dataset Analysis


## Dataset: m4

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/wavelet_study_3_successive_trendae_results.csv`
- Rows: 1184
- Primary metric: `owa`


## 1. Overview & Data Summary

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/wavelet_study_3_successive_trendae_results.csv`
- Total rows: 1184
- Unique configs: 112
- Search rounds: [1, 2, 3]
- Primary metric: owa

|   Round |   Configs |   Rows | Epochs   | Passes          |
|--------:|----------:|-------:|:---------|:----------------|
|       1 |       112 |    662 | 7-7      | False, forecast |
|       2 |        57 |    342 | 15-15    | False, forecast |
|       3 |        30 |    180 | 17-50    | False, forecast |


## 2. Successive Halving Funnel

|   Round |   Configs |   Rows |   Best Med OWA | Kept         |
|--------:|----------:|-------:|---------------:|:-------------|
|       1 |       112 |    662 |         0.906  | -            |
|       2 |        57 |    342 |         0.833  | 57/112 (51%) |
|       3 |        30 |    180 |         0.7996 | 30/57 (53%)  |


## 3. Round Leaderboards

### Round 1

| Config                      | Pass     |    OWA |    Std |   sMAPE |   MASE |   Params |
|:----------------------------|:---------|-------:|-------:|--------:|-------:|---------:|
| DB10_bd6_eq_fcast_ttd5      | False    | 0.9087 | 0.0173 | 15.1362 | 3.5414 |  4247085 |
| Coif10_bd15_lt_bcast_ttd3   | False    | 0.9114 | 0.0039 | 15.1379 | 3.5629 |  4267555 |
| Coif1_bd30_eq_bcast_ttd3    | False    | 0.9116 | 0.0209 | 15.3408 | 3.5151 |  4305955 |
| Haar_bd30_eq_bcast_ttd3     | False    | 0.9131 | 0.0175 | 15.3927 | 3.5140 |  4305955 |
| Symlet20_bd15_lt_bcast_ttd5 | False    | 0.9151 | 0.0184 | 15.2685 | 3.5605 |  4270125 |
| Coif2_bd6_eq_fcast_ttd3     | forecast | 0.9152 | 0.0280 | 15.2445 | 3.5670 |  4244515 |
| Coif1_bd6_eq_fcast_ttd5     | False    | 0.9165 | 0.0309 | 15.3928 | 3.5409 |  4247085 |
| Coif3_bd15_lt_bcast_ttd3    | False    | 0.9192 | 0.0413 | 15.3580 | 3.5709 |  4267555 |
| Symlet2_bd6_eq_fcast_ttd5   | False    | 0.9215 | 0.0477 | 15.3672 | 3.5873 |  4247085 |
| Coif1_bd15_lt_bcast_ttd3    | forecast | 0.9222 | 0.0340 | 15.4627 | 3.5693 |  4267555 |
| DB3_bd6_eq_fcast_ttd3       | forecast | 0.9228 | 0.0591 | 15.4221 | 3.5841 |  4244515 |
| Symlet10_bd6_eq_fcast_ttd3  | False    | 0.9242 | 0.0218 | 15.4421 | 3.5901 |  4244515 |
| Symlet2_bd6_eq_fcast_ttd3   | False    | 0.9278 | 0.0408 | 15.4924 | 3.6067 |  4244515 |
| DB10_bd6_eq_fcast_ttd3      | forecast | 0.9286 | 0.0471 | 15.4952 | 3.6125 |  4244515 |
| DB4_bd6_eq_fcast_ttd3       | False    | 0.9291 | 0.0395 | 15.5687 | 3.5982 |  4244515 |

### Round 2

| Config                     | Pass     |    OWA |    Std |   sMAPE |   MASE |   Params |
|:---------------------------|:---------|-------:|-------:|--------:|-------:|---------:|
| Coif3_bd4_lt_fcast_ttd3    | False    | 0.8270 | 0.0081 | 13.9784 | 3.1737 |  4234275 |
| Coif10_bd6_eq_fcast_ttd3   | False    | 0.8295 | 0.0070 | 13.9079 | 3.2108 |  4244515 |
| Haar_bd6_eq_fcast_ttd5     | False    | 0.8295 | 0.0187 | 13.9254 | 3.2067 |  4247085 |
| Coif1_bd6_eq_fcast_ttd3    | forecast | 0.8305 | 0.0115 | 13.9147 | 3.2169 |  4244515 |
| DB20_bd6_eq_fcast_ttd3     | False    | 0.8335 | 0.0331 | 13.9512 | 3.2324 |  4244515 |
| Symlet3_bd30_eq_bcast_ttd3 | False    | 0.8355 | 0.0138 | 14.0381 | 3.2272 |  4305955 |
| Symlet3_bd6_eq_fcast_ttd3  | forecast | 0.8382 | 0.0137 | 14.0763 | 3.2390 |  4244515 |
| DB4_bd6_eq_fcast_ttd5      | False    | 0.8404 | 0.0463 | 14.0191 | 3.2705 |  4247085 |
| DB10_bd15_lt_bcast_ttd3    | forecast | 0.8405 | 0.0543 | 14.0470 | 3.2641 |  4267555 |
| Symlet20_bd6_eq_fcast_ttd5 | False    | 0.8429 | 0.0249 | 14.1604 | 3.2556 |  4247085 |
| DB10_bd6_eq_fcast_ttd5     | False    | 0.8435 | 0.0104 | 14.1952 | 3.2526 |  4247085 |
| Coif2_bd15_lt_bcast_ttd3   | False    | 0.8442 | 0.0226 | 14.1145 | 3.2773 |  4267555 |
| DB3_bd15_lt_bcast_ttd3     | forecast | 0.8445 | 0.0392 | 14.1399 | 3.2734 |  4267555 |
| Symlet3_bd6_eq_fcast_ttd3  | False    | 0.8464 | 0.0229 | 14.0959 | 3.2991 |  4244515 |
| Coif10_bd15_lt_bcast_ttd3  | False    | 0.8484 | 0.0500 | 14.1567 | 3.3003 |  4267555 |

### Round 3

| Config                     | Pass     |    OWA |    Std |   sMAPE |   MASE |   Params |
|:---------------------------|:---------|-------:|-------:|--------:|-------:|---------:|
| DB4_bd6_eq_fcast_ttd5      | forecast | 0.7979 | 0.0029 | 13.4625 | 3.0676 |  4247085 |
| Symlet3_bd15_lt_bcast_ttd5 | False    | 0.7980 | 0.0044 | 13.4658 | 3.0678 |  4270125 |
| Symlet2_bd6_eq_fcast_ttd3  | forecast | 0.7993 | 0.0065 | 13.4789 | 3.0752 |  4244515 |
| DB2_bd15_lt_bcast_ttd5     | False    | 0.8027 | 0.0098 | 13.5184 | 3.0928 |  4270125 |
| DB20_bd6_eq_fcast_ttd3     | False    | 0.8033 | 0.0084 | 13.5352 | 3.0933 |  4244515 |
| DB3_bd6_eq_fcast_ttd3      | forecast | 0.8038 | 0.0122 | 13.5486 | 3.0942 |  4244515 |
| Coif10_bd15_lt_bcast_ttd3  | False    | 0.8040 | 0.0077 | 13.5372 | 3.0980 |  4267555 |
| Symlet3_bd6_eq_fcast_ttd3  | forecast | 0.8040 | 0.0104 | 13.5327 | 3.0995 |  4244515 |
| Symlet10_bd6_eq_fcast_ttd3 | forecast | 0.8044 | 0.0082 | 13.5286 | 3.1037 |  4244515 |
| Coif1_bd6_eq_fcast_ttd3    | forecast | 0.8045 | 0.0078 | 13.5458 | 3.1003 |  4244515 |
| DB3_bd6_eq_fcast_ttd3      | False    | 0.8063 | 0.0044 | 13.5417 | 3.1152 |  4244515 |
| Symlet20_bd6_eq_fcast_ttd3 | forecast | 0.8067 | 0.0162 | 13.5741 | 3.1108 |  4244515 |
| Symlet3_bd15_lt_bcast_ttd3 | forecast | 0.8073 | 0.0153 | 13.5830 | 3.1131 |  4267555 |
| DB10_bd15_lt_bcast_ttd3    | forecast | 0.8074 | 0.0153 | 13.6004 | 3.1097 |  4267555 |
| DB20_bd6_eq_fcast_ttd3     | forecast | 0.8075 | 0.0080 | 13.5771 | 3.1164 |  4244515 |


## 4. Hyperparameter Marginals (Round 1)

### wavelet

| Value    |   Med OWA |   Mean OWA |    Std |   N |
|:---------|----------:|-----------:|-------:|----:|
| Symlet20 |    0.9583 |     0.9892 | 0.0949 |  38 |
| Coif1    |    0.9606 |     0.9943 | 0.1107 |  48 |
| DB10     |    0.9647 |     1.0011 | 0.0960 |  48 |
| Coif3    |    0.9701 |     0.9790 | 0.0743 |  48 |
| Coif10   |    0.9731 |     1.0072 | 0.1327 |  48 |
| Symlet2  |    0.9750 |     0.9907 | 0.1048 |  48 |
| DB3      |    0.9751 |     0.9941 | 0.0866 |  48 |
| Symlet10 |    0.9771 |     1.0481 | 0.1833 |  48 |
| Symlet3  |    0.9830 |     1.0030 | 0.0873 |  48 |
| DB20     |    0.9832 |     0.9960 | 0.0821 |  48 |
| Coif2    |    0.9849 |     1.0056 | 0.1049 |  48 |
| Haar     |    0.9876 |     1.0396 | 0.1411 |  48 |
| DB4      |    0.9999 |     1.0160 | 0.1173 |  48 |
| DB2      |    1.0055 |     1.0180 | 0.0846 |  48 |

`Symlet20` is best by median OWA with a 0.0472 gap to `DB2`.

### bd_label

| Value    |   Med OWA |   Mean OWA |    Std |   N |
|:---------|----------:|-----------:|-------:|----:|
| eq_fcast |    0.9505 |     0.9702 | 0.0765 | 166 |
| lt_bcast |    0.9598 |     0.9851 | 0.0973 | 168 |
| eq_bcast |    0.9778 |     0.9987 | 0.0917 | 166 |
| lt_fcast |    1.0313 |     1.0723 | 0.1426 | 162 |

`eq_fcast` is best by median OWA with a 0.0808 gap to `lt_fcast`.

### ttd

|   Value |   Med OWA |   Mean OWA |    Std |        N |
|--------:|----------:|-----------:|-------:|---------:|
|  3.0000 |    0.9738 |     0.9994 | 0.1156 | 331.0000 |
|  5.0000 |    0.9803 |     1.0128 | 0.1070 | 331.0000 |

`3` is best by median OWA with a 0.0064 gap to `5`.

### active_g_cfg

| Value    |   Med OWA |   Mean OWA |    Std |   N |
|:---------|----------:|-----------:|-------:|----:|
| False    |    0.9763 |     1.0085 | 0.1209 | 336 |
| forecast |    0.9769 |     1.0036 | 0.1010 | 326 |

`False` is best by median OWA with a 0.0005 gap to `forecast`.


## 5. Stability Analysis

### Round 1

- Mean spread: 0.2277
- Max spread: 0.6678 (Symlet10_bd4_lt_fcast_ttd5)
- Mean std: 0.0880

### Round 2

- Mean spread: 0.1023
- Max spread: 0.2101 (Coif10_bd30_eq_bcast_ttd3)
- Mean std: 0.0390

### Round 3

- Mean spread: 0.0687
- Max spread: 0.1754 (Symlet2_bd6_eq_fcast_ttd3)
- Mean std: 0.0269


## 6. Round-over-Round Progression

| config_name                 |     R1 |     R2 |     R3 |   Delta |   DeltaPct |
|:----------------------------|-------:|-------:|-------:|--------:|-----------:|
| Coif10_bd4_lt_fcast_ttd5    | 1.0325 | 0.8686 | 0.8104 | -0.2222 |      -21.5 |
| Symlet3_bd15_lt_bcast_ttd3  | 0.9923 | 0.875  | 0.8034 | -0.1889 |      -19   |
| Coif3_bd6_eq_fcast_ttd5     | 0.9983 | 0.873  | 0.8102 | -0.1881 |      -18.8 |
| DB20_bd6_eq_fcast_ttd3      | 0.9863 | 0.8432 | 0.8079 | -0.1783 |      -18.1 |
| Symlet2_bd4_lt_fcast_ttd3   | 0.9811 | 0.8461 | 0.8093 | -0.1718 |      -17.5 |
| Symlet3_bd6_eq_fcast_ttd3   | 0.9731 | 0.8407 | 0.803  | -0.1701 |      -17.5 |
| Symlet20_bd6_eq_fcast_ttd3  | 0.9677 | 0.8984 | 0.8057 | -0.1621 |      -16.7 |
| DB3_bd6_eq_fcast_ttd3       | 0.965  | 0.8658 | 0.8053 | -0.1598 |      -16.6 |
| DB2_bd15_lt_bcast_ttd5      | 0.9683 | 0.8641 | 0.8074 | -0.1609 |      -16.6 |
| Coif10_bd6_eq_fcast_ttd3    | 0.9602 | 0.8335 | 0.8045 | -0.1557 |      -16.2 |
| Symlet20_bd4_lt_fcast_ttd5  | 0.957  | 0.8546 | 0.8037 | -0.1533 |      -16   |
| Coif3_bd4_lt_fcast_ttd3     | 0.9763 | 0.833  | 0.8229 | -0.1534 |      -15.7 |
| DB4_bd30_eq_bcast_ttd5      | 0.9735 | 0.8837 | 0.8278 | -0.1457 |      -15   |
| Coif10_bd30_eq_bcast_ttd3   | 0.9435 | 0.8772 | 0.8041 | -0.1395 |      -14.8 |
| Symlet2_bd6_eq_fcast_ttd5   | 0.9595 | 0.8469 | 0.8175 | -0.142  |      -14.8 |
| DB10_bd15_lt_bcast_ttd3     | 0.9573 | 0.8888 | 0.8156 | -0.1417 |      -14.8 |
| Haar_bd15_lt_bcast_ttd3     | 0.9788 | 0.8669 | 0.8394 | -0.1394 |      -14.2 |
| Symlet3_bd15_lt_bcast_ttd5  | 0.9295 | 0.8853 | 0.8015 | -0.128  |      -13.8 |
| DB4_bd6_eq_fcast_ttd5       | 0.927  | 0.8542 | 0.7996 | -0.1274 |      -13.7 |
| Symlet10_bd6_eq_fcast_ttd3  | 0.9367 | 0.8762 | 0.8091 | -0.1276 |      -13.6 |
| Coif1_bd6_eq_fcast_ttd3     | 0.9285 | 0.8371 | 0.807  | -0.1215 |      -13.1 |
| Coif3_bd15_lt_bcast_ttd3    | 0.9405 | 0.8521 | 0.8279 | -0.1126 |      -12   |
| Coif10_bd15_lt_bcast_ttd3   | 0.9135 | 0.8355 | 0.8077 | -0.1058 |      -11.6 |
| Symlet2_bd6_eq_fcast_ttd3   | 0.906  | 0.8734 | 0.8021 | -0.1039 |      -11.5 |
| DB10_bd6_eq_fcast_ttd5      | 0.92   | 0.8596 | 0.8157 | -0.1044 |      -11.3 |
| Symlet20_bd15_lt_bcast_ttd3 | 0.9409 | 0.8471 | 0.8397 | -0.1013 |      -10.8 |
| DB4_bd6_eq_fcast_ttd3       | 0.9117 | 0.8868 | 0.8145 | -0.0972 |      -10.7 |
| Symlet20_bd15_lt_bcast_ttd5 | 0.9347 | 0.8663 | 0.8404 | -0.0943 |      -10.1 |
| Coif2_bd6_eq_fcast_ttd3     | 0.9066 | 0.8588 | 0.8332 | -0.0734 |       -8.1 |
| Symlet2_bd30_eq_bcast_ttd3  | 0.9255 | 0.8896 | 0.8669 | -0.0586 |       -6.3 |

30/30 finalist configs improved as training budget increased.


## 7. Baseline Comparisons

| Config                     | Pass     |    OWA |   sMAPE |   Params |   vs NBEATS-I+G |
|:---------------------------|:---------|-------:|--------:|---------:|----------------:|
| DB4_bd6_eq_fcast_ttd5      | forecast | 0.7979 | 13.4625 |  4247085 |         -0.0078 |
| Symlet3_bd15_lt_bcast_ttd5 | False    | 0.7980 | 13.4658 |  4270125 |         -0.0077 |
| Symlet2_bd6_eq_fcast_ttd3  | forecast | 0.7993 | 13.4789 |  4244515 |         -0.0064 |
| DB2_bd15_lt_bcast_ttd5     | False    | 0.8027 | 13.5184 |  4270125 |         -0.0030 |
| DB20_bd6_eq_fcast_ttd3     | False    | 0.8033 | 13.5352 |  4244515 |         -0.0024 |
| DB3_bd6_eq_fcast_ttd3      | forecast | 0.8038 | 13.5486 |  4244515 |         -0.0019 |
| Coif10_bd15_lt_bcast_ttd3  | False    | 0.8040 | 13.5372 |  4267555 |         -0.0017 |
| Symlet3_bd6_eq_fcast_ttd3  | forecast | 0.8040 | 13.5327 |  4244515 |         -0.0017 |
| Symlet10_bd6_eq_fcast_ttd3 | forecast | 0.8044 | 13.5286 |  4244515 |         -0.0013 |
| Coif1_bd6_eq_fcast_ttd3    | forecast | 0.8045 | 13.5458 |  4244515 |         -0.0012 |

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

Best configuration: Symlet3_bd15_lt_bcast_ttd5 (pass=False) with median OWA=0.7960.
vs NBEATS-I+G (0.8057): beats (delta=-0.0097).

| Config                     | Pass     |   Med OWA |    Std |   Params |   sMAPE |   MASE |
|:---------------------------|:---------|----------:|-------:|---------:|--------:|-------:|
| Symlet3_bd15_lt_bcast_ttd5 | False    |    0.7960 | 0.0044 |  4270125 | 13.4417 | 3.0581 |
| Symlet2_bd6_eq_fcast_ttd3  | forecast |    0.7962 | 0.0065 |  4244515 | 13.4456 | 3.0584 |
| DB4_bd6_eq_fcast_ttd5      | forecast |    0.7979 | 0.0029 |  4247085 | 13.4635 | 3.0677 |
| DB4_bd6_eq_fcast_ttd3      | False    |    0.7979 | 0.0379 |  4244515 | 13.4438 | 3.0727 |
| Symlet20_bd4_lt_fcast_ttd5 | forecast |    0.7989 | 0.0302 |  4236845 | 13.4747 | 3.0726 |
| Symlet3_bd6_eq_fcast_ttd3  | forecast |    0.7994 | 0.0104 |  4244515 | 13.5087 | 3.0684 |
| Symlet2_bd6_eq_fcast_ttd5  | forecast |    0.7994 | 0.0319 |  4247085 | 13.4983 | 3.0713 |
| Symlet3_bd15_lt_bcast_ttd3 | forecast |    0.8000 | 0.0153 |  4267555 | 13.4790 | 3.0803 |
| Symlet20_bd6_eq_fcast_ttd3 | forecast |    0.8003 | 0.0162 |  4244515 | 13.5031 | 3.0772 |
| Symlet2_bd4_lt_fcast_ttd3  | False    |    0.8012 | 0.0328 |  4234275 | 13.5182 | 3.0808 |

[SKIP] dataset=tourism reason=missing_csv path=/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/tourism/wavelet_study_3_successive_trendae_results.csv
[SKIP] dataset=weather reason=missing_csv path=/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/weather/wavelet_study_3_successive_trendae_results.csv

# Summary

- analyzed_count: 1
- skipped_count: 2
- analyzed: ['m4']
- skipped:
  - dataset=tourism reason=missing_csv path=/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/tourism/wavelet_study_3_successive_trendae_results.csv
  - dataset=weather reason=missing_csv path=/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/weather/wavelet_study_3_successive_trendae_results.csv
