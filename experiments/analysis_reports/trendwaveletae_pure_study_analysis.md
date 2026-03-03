# TrendWaveletAE Pure-Stack Study Analysis

# Block Type: TrendWaveletAE

## TrendWaveletAE — Dataset: M4-Yearly

- Rows: 2133
- Unique configs: 336

### Successive-Halving Funnel
|   Round |   Configs |   Rows |   Median best_val_loss | Kept          |
|--------:|----------:|-------:|-----------------------:|:--------------|
|       1 |       336 |   1008 |                16.2828 | -             |
|       2 |       225 |    675 |                14.9406 | 225/336 (67%) |
|       3 |       150 |    450 |                13.7287 | 150/225 (67%) |

### Round 3 Leaderboard (Top 10)
| config_name                           | wavelet_type   | bd_label   |   trend_dim_cfg |   latent_dim_cfg |   median_best_val_loss |   std_best_val_loss |   n |
|:--------------------------------------|:---------------|:-----------|----------------:|-----------------:|-----------------------:|--------------------:|----:|
| TrendWaveletAE_sym10_eq_fcast_td3_ld8 | sym10          | eq_fcast   |               3 |                8 |                13.5869 |              0.1436 |   3 |
| TrendWaveletAE_db2_lt_fcast_td3_ld8   | db2            | lt_fcast   |               3 |                8 |                13.6403 |              0.0668 |   3 |
| TrendWaveletAE_db20_eq_fcast_td3_ld8  | db20           | eq_fcast   |               3 |                8 |                13.6410 |              0.0829 |   3 |
| TrendWaveletAE_haar_eq_fcast_td3_ld8  | haar           | eq_fcast   |               3 |                8 |                13.6454 |              0.0302 |   3 |
| TrendWaveletAE_sym20_eq_fcast_td5_ld8 | sym20          | eq_fcast   |               5 |                8 |                13.6461 |              0.1069 |   3 |
| TrendWaveletAE_db4_eq_fcast_td3_ld8   | db4            | eq_fcast   |               3 |                8 |                13.6464 |              0.0479 |   3 |
| TrendWaveletAE_db3_lt_fcast_td3_ld8   | db3            | lt_fcast   |               3 |                8 |                13.6466 |              0.0892 |   3 |
| TrendWaveletAE_coif1_eq_fcast_td3_ld8 | coif1          | eq_fcast   |               3 |                8 |                13.6510 |              0.0214 |   3 |
| TrendWaveletAE_db2_eq_fcast_td3_ld8   | db2            | eq_fcast   |               3 |                8 |                13.6554 |              0.0345 |   3 |
| TrendWaveletAE_db3_eq_fcast_td3_ld8   | db3            | eq_fcast   |               3 |                8 |                13.6600 |              0.0087 |   3 |

## TrendWaveletAE — Dataset: Tourism-Yearly

- Missing CSV: `experiments/results/tourism/trendwaveletae_pure_study_results.csv`

## TrendWaveletAE — Dataset: Traffic-96

- Missing CSV: `experiments/results/traffic/trendwaveletae_pure_study_results.csv`

## TrendWaveletAE — Dataset: Weather-96

- Missing CSV: `experiments/results/weather/trendwaveletae_pure_study_results.csv`

# Block Type: TrendWaveletAELG

## TrendWaveletAELG — Dataset: M4-Yearly

- Rows: 711
- Unique configs: 112

### Successive-Halving Funnel
|   Round |   Configs |   Rows |   Median best_val_loss | Kept         |
|--------:|----------:|-------:|-----------------------:|:-------------|
|       1 |       112 |    336 |                16.1054 | -            |
|       2 |        75 |    225 |                14.8265 | 75/112 (67%) |
|       3 |        50 |    150 |                13.6977 | 50/75 (67%)  |

### Round 3 Leaderboard (Top 10)
| config_name                             | wavelet_type   | bd_label   |   trend_dim_cfg |   latent_dim_cfg |   median_best_val_loss |   std_best_val_loss |   n |
|:----------------------------------------|:---------------|:-----------|----------------:|-----------------:|-----------------------:|--------------------:|----:|
| TrendWaveletAELG_db2_eq_fcast_td3_ld8   | db2            | eq_fcast   |               3 |                8 |                13.6184 |              0.0416 |   3 |
| TrendWaveletAELG_sym10_eq_fcast_td3_ld8 | sym10          | eq_fcast   |               3 |                8 |                13.6308 |              0.0459 |   3 |
| TrendWaveletAELG_coif3_lt_fcast_td3_ld8 | coif3          | lt_fcast   |               3 |                8 |                13.6334 |              0.0399 |   3 |
| TrendWaveletAELG_sym2_lt_fcast_td3_ld8  | sym2           | lt_fcast   |               3 |                8 |                13.6390 |              0.0331 |   3 |
| TrendWaveletAELG_db2_eq_fcast_td5_ld8   | db2            | eq_fcast   |               5 |                8 |                13.6470 |              0.0632 |   3 |
| TrendWaveletAELG_sym20_lt_fcast_td3_ld8 | sym20          | lt_fcast   |               3 |                8 |                13.6498 |              0.0132 |   3 |
| TrendWaveletAELG_haar_eq_fcast_td5_ld8  | haar           | eq_fcast   |               5 |                8 |                13.6502 |              0.0705 |   3 |
| TrendWaveletAELG_coif2_eq_fcast_td3_ld8 | coif2          | eq_fcast   |               3 |                8 |                13.6541 |              0.0381 |   3 |
| TrendWaveletAELG_sym10_lt_fcast_td3_ld8 | sym10          | lt_fcast   |               3 |                8 |                13.6612 |              0.0601 |   3 |
| TrendWaveletAELG_coif2_eq_fcast_td5_ld8 | coif2          | eq_fcast   |               5 |                8 |                13.6622 |              0.0419 |   3 |

## TrendWaveletAELG — Dataset: Tourism-Yearly

- Missing CSV: `experiments/results/tourism/trendwaveletaelg_pure_study_results.csv`

## TrendWaveletAELG — Dataset: Traffic-96

- Missing CSV: `experiments/results/traffic/trendwaveletaelg_pure_study_results.csv`

## TrendWaveletAELG — Dataset: Weather-96

- Missing CSV: `experiments/results/weather/trendwaveletaelg_pure_study_results.csv`

# AE vs LG Comparison

## M4-Yearly

- AE: latest round = 3
- LG: latest round = 3
| wavelet_type   |      AE |      LG |    diff | winner   |
|:---------------|--------:|--------:|--------:|:---------|
| sym2           | 13.7105 | 13.7158 | -0.0053 | AE       |
| db10           | 13.7418 | 13.7334 |  0.0084 | LG       |
| db3            | 13.7324 | 13.7159 |  0.0165 | LG       |
| haar           | 13.7023 | 13.6808 |  0.0215 | LG       |
| sym10          | 13.7197 | 13.6969 |  0.0228 | LG       |
| db4            | 13.7301 | 13.6820 |  0.0481 | LG       |
| sym20          | 13.7406 | 13.6917 |  0.0488 | LG       |
| coif10         | 13.7348 | 13.6834 |  0.0514 | LG       |
| coif1          | 13.7566 | 13.7033 |  0.0533 | LG       |
| db20           | 13.7788 | 13.7197 |  0.0591 | LG       |
| coif2          | 13.7246 | 13.6622 |  0.0624 | LG       |
| coif3          | 13.7586 | 13.6863 |  0.0723 | LG       |
| db2            | 13.7234 | 13.6470 |  0.0765 | LG       |

Win/Loss/Draw: AE=1, LG=12, tie=0
Wilcoxon signed-rank test: stat=1.0000, p=0.0005

## Tourism-Yearly

Not enough data (need both AE and LG results).

## Traffic-96

Not enough data (need both AE and LG results).

## Weather-96

Not enough data (need both AE and LG results).

# Pure vs Alternating Comparison

## M4-Yearly

Round 3 not available for comparison.

## Tourism-Yearly

Not enough data for comparison.

## Traffic-96

Not enough data for comparison.

## Weather-96

Not enough data for comparison.

# Cross-Dataset Results

## TrendWaveletAE

- Missing or empty cross CSV: `experiments/results/trendwaveletae_pure_cross_dataset_results.csv`

## TrendWaveletAELG

- Missing or empty cross CSV: `experiments/results/trendwaveletaelg_pure_cross_dataset_results.csv`

# Hyperparameter Sensitivity

## TrendWaveletAE

### M4-Yearly

#### wavelet_type
| wavelet_type   |   median_best_val_loss |   mean_best_val_loss |   n |
|:---------------|-----------------------:|---------------------:|----:|
| db4            |                16.0391 |              16.4416 |  72 |
| coif3          |                16.0756 |              16.2860 |  72 |
| sym20          |                16.0882 |              16.3344 |  72 |
| db10           |                16.1987 |              16.3145 |  72 |
| coif10         |                16.2016 |              16.5198 |  72 |
| coif1          |                16.2348 |              16.4734 |  72 |
| db3            |                16.2359 |              16.5891 |  72 |
| haar           |                16.2951 |              16.5768 |  72 |
| coif2          |                16.3010 |              16.4872 |  72 |
| sym3           |                16.3906 |              16.4325 |  72 |
| db2            |                16.3931 |              16.4834 |  72 |
| sym10          |                16.4077 |              16.6014 |  72 |
| db20           |                16.4207 |              16.5687 |  72 |
| sym2           |                16.5316 |              16.7071 |  72 |

#### wavelet_family
| wavelet_family   |   median_best_val_loss |   mean_best_val_loss |   n |
|:-----------------|-----------------------:|---------------------:|----:|
| db4              |                16.0391 |              16.4416 |  72 |
| coif3            |                16.0756 |              16.2860 |  72 |
| sym20            |                16.0882 |              16.3344 |  72 |
| db10             |                16.1987 |              16.3145 |  72 |
| coif10           |                16.2016 |              16.5198 |  72 |
| coif1            |                16.2348 |              16.4734 |  72 |
| db3              |                16.2359 |              16.5891 |  72 |
| haar             |                16.2951 |              16.5768 |  72 |
| coif2            |                16.3010 |              16.4872 |  72 |
| sym3             |                16.3906 |              16.4325 |  72 |
| db2              |                16.3931 |              16.4834 |  72 |
| sym10            |                16.4077 |              16.6014 |  72 |
| db20             |                16.4207 |              16.5687 |  72 |
| sym2             |                16.5316 |              16.7071 |  72 |

#### bd_label
| bd_label   |   median_best_val_loss |   mean_best_val_loss |   n |
|:-----------|-----------------------:|---------------------:|----:|
| lt_fcast   |                15.9001 |              16.1084 | 252 |
| eq_fcast   |                15.9033 |              16.1178 | 252 |
| lt_bcast   |                16.6108 |              16.7469 | 252 |
| eq_bcast   |                16.8166 |              16.9743 | 252 |

#### trend_dim_cfg
|   trend_dim_cfg |   median_best_val_loss |   mean_best_val_loss |        n |
|----------------:|-----------------------:|---------------------:|---------:|
|          3.0000 |                16.1176 |              16.3695 | 504.0000 |
|          5.0000 |                16.4371 |              16.6043 | 504.0000 |

#### latent_dim_cfg
|   latent_dim_cfg |   median_best_val_loss |   mean_best_val_loss |        n |
|-----------------:|-----------------------:|---------------------:|---------:|
|           8.0000 |                15.9693 |              16.1625 | 336.0000 |
|           5.0000 |                16.0127 |              16.2438 | 336.0000 |
|           2.0000 |                16.9139 |              17.0543 | 336.0000 |

### Tourism-Yearly

No data.

### Traffic-96

No data.

### Weather-96

No data.

## TrendWaveletAELG

### M4-Yearly

#### wavelet_type
| wavelet_type   |   median_best_val_loss |   mean_best_val_loss |   n |
|:---------------|-----------------------:|---------------------:|----:|
| sym2           |                15.7800 |              15.9628 |  24 |
| db4            |                15.7901 |              16.0783 |  24 |
| db20           |                15.9155 |              16.2337 |  24 |
| haar           |                15.9658 |              16.2188 |  24 |
| coif3          |                16.0817 |              16.2542 |  24 |
| db3            |                16.0952 |              16.2640 |  24 |
| sym20          |                16.1206 |              16.4311 |  24 |
| sym3           |                16.1608 |              16.5558 |  24 |
| db10           |                16.1802 |              16.5271 |  24 |
| db2            |                16.1982 |              16.3314 |  24 |
| coif2          |                16.2209 |              16.3087 |  24 |
| coif10         |                16.3885 |              16.4360 |  24 |
| sym10          |                16.4421 |              16.9508 |  24 |
| coif1          |                16.5604 |              16.5567 |  24 |

#### wavelet_family
| wavelet_family   |   median_best_val_loss |   mean_best_val_loss |   n |
|:-----------------|-----------------------:|---------------------:|----:|
| sym2             |                15.7800 |              15.9628 |  24 |
| db4              |                15.7901 |              16.0783 |  24 |
| db20             |                15.9155 |              16.2337 |  24 |
| haar             |                15.9658 |              16.2188 |  24 |
| coif3            |                16.0817 |              16.2542 |  24 |
| db3              |                16.0952 |              16.2640 |  24 |
| sym20            |                16.1206 |              16.4311 |  24 |
| sym3             |                16.1608 |              16.5558 |  24 |
| db10             |                16.1802 |              16.5271 |  24 |
| db2              |                16.1982 |              16.3314 |  24 |
| coif2            |                16.2209 |              16.3087 |  24 |
| coif10           |                16.3885 |              16.4360 |  24 |
| sym10            |                16.4421 |              16.9508 |  24 |
| coif1            |                16.5604 |              16.5567 |  24 |

#### bd_label
| bd_label   |   median_best_val_loss |   mean_best_val_loss |   n |
|:-----------|-----------------------:|---------------------:|----:|
| eq_fcast   |                15.6573 |              15.7981 |  84 |
| lt_fcast   |                15.9238 |              16.0451 |  84 |
| lt_bcast   |                16.4785 |              16.6735 |  84 |
| eq_bcast   |                16.7340 |              16.9431 |  84 |

#### trend_dim_cfg
|   trend_dim_cfg |   median_best_val_loss |   mean_best_val_loss |        n |
|----------------:|-----------------------:|---------------------:|---------:|
|          3.0000 |                15.9511 |              16.2231 | 168.0000 |
|          5.0000 |                16.2664 |              16.5068 | 168.0000 |

#### latent_dim_cfg
|   latent_dim_cfg |   median_best_val_loss |   mean_best_val_loss |        n |
|-----------------:|-----------------------:|---------------------:|---------:|
|           8.0000 |                16.1054 |              16.3650 | 336.0000 |

### Tourism-Yearly

No data.

### Traffic-96

No data.

### Weather-96

No data.
