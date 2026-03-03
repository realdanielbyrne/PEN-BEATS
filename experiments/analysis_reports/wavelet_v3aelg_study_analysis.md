# WaveletV3AELG Study Analysis

# Trend Block: TrendAELG

## TrendAELG — Dataset: M4-Yearly

- Rows: 69
- Unique configs: 23

### Successive-Halving Funnel
|   Round |   Configs |   Rows |   Median best_val_loss | Kept   |
|--------:|----------:|-------:|-----------------------:|:-------|
|       1 |        23 |     69 |                16.9083 | -      |

### Round 3 Leaderboard (Top 10)
Round 3 not available.

### Round 1 Hyperparameter Marginals (median best_val_loss)
#### wavelet_family
| wavelet_family    |   median_best_val_loss |   mean_best_val_loss |   n |
|:------------------|-----------------------:|---------------------:|----:|
| DB2WaveletV3AELG  |                16.5923 |              16.9414 |  24 |
| HaarWaveletV3AELG |                17.1080 |              17.0307 |  24 |
| DB3WaveletV3AELG  |                17.5004 |              26.0462 |  21 |

#### bd_label
| bd_label   |   median_best_val_loss |   mean_best_val_loss |   n |
|:-----------|-----------------------:|---------------------:|----:|
| eq_bcast   |                16.2530 |              16.3810 |  18 |
| lt_bcast   |                16.5956 |              28.9423 |  15 |
| eq_fcast   |                17.3111 |              17.2475 |  18 |
| lt_fcast   |                17.8960 |              17.9363 |  18 |

#### trend_thetas_dim_cfg
|   trend_thetas_dim_cfg |   median_best_val_loss |   mean_best_val_loss |       n |
|-----------------------:|-----------------------:|---------------------:|--------:|
|                 3.0000 |                16.4109 |              21.7080 | 36.0000 |
|                 5.0000 |                17.4630 |              17.6004 | 33.0000 |

#### latent_dim_cfg
|   latent_dim_cfg |   median_best_val_loss |   mean_best_val_loss |       n |
|-----------------:|-----------------------:|---------------------:|--------:|
|          16.0000 |                16.9083 |              19.7435 | 69.0000 |

## TrendAELG — Dataset: Tourism-Yearly

- Rows: 762
- Unique configs: 112

### Successive-Halving Funnel
|   Round |   Configs |   Rows |   Median best_val_loss | Kept         |
|--------:|----------:|-------:|-----------------------:|:-------------|
|       1 |       112 |    387 |                32.2916 | -            |
|       2 |        75 |    225 |                30.0219 | 75/112 (67%) |
|       3 |        50 |    150 |                24.9489 | 50/75 (67%)  |

### Round 3 Leaderboard (Top 10)
| config_name                 | wavelet_family        | bd_label   |   trend_thetas_dim_cfg |   latent_dim_cfg |   median_best_val_loss |   std_best_val_loss |   n |
|:----------------------------|:----------------------|:-----------|-----------------------:|-----------------:|-----------------------:|--------------------:|----:|
| Coif1_eq_bcast_ttd5_ld16    | Coif1WaveletV3AELG    | eq_bcast   |                      5 |               16 |                24.8333 |              0.1422 |   3 |
| DB2_eq_fcast_ttd3_ld16      | DB2WaveletV3AELG      | eq_fcast   |                      3 |               16 |                24.8343 |              0.0852 |   3 |
| DB2_lt_bcast_ttd3_ld16      | DB2WaveletV3AELG      | lt_bcast   |                      3 |               16 |                24.8343 |              0.0852 |   3 |
| DB4_lt_bcast_ttd3_ld16      | DB4WaveletV3AELG      | lt_bcast   |                      3 |               16 |                24.8383 |              0.1168 |   3 |
| DB4_eq_fcast_ttd3_ld16      | DB4WaveletV3AELG      | eq_fcast   |                      3 |               16 |                24.8383 |              0.1168 |   3 |
| DB2_eq_bcast_ttd5_ld16      | DB2WaveletV3AELG      | eq_bcast   |                      5 |               16 |                24.8424 |              0.1802 |   3 |
| Symlet20_lt_bcast_ttd3_ld16 | Symlet20WaveletV3AELG | lt_bcast   |                      3 |               16 |                24.8770 |              0.2833 |   3 |
| Symlet20_eq_fcast_ttd3_ld16 | Symlet20WaveletV3AELG | eq_fcast   |                      3 |               16 |                24.8770 |              0.2833 |   3 |
| Coif1_lt_bcast_ttd3_ld16    | Coif1WaveletV3AELG    | lt_bcast   |                      3 |               16 |                24.8802 |              0.1378 |   3 |
| Coif1_eq_fcast_ttd3_ld16    | Coif1WaveletV3AELG    | eq_fcast   |                      3 |               16 |                24.8802 |              0.1378 |   3 |

### Round 1 Hyperparameter Marginals (median best_val_loss)
#### wavelet_family
| wavelet_family        |   median_best_val_loss |   mean_best_val_loss |   n |
|:----------------------|-----------------------:|---------------------:|----:|
| Coif10WaveletV3AELG   |                31.4367 |              32.4114 |  24 |
| HaarWaveletV3AELG     |                31.7400 |              33.3651 |  55 |
| DB3WaveletV3AELG      |                32.0937 |              34.3504 |  24 |
| Coif1WaveletV3AELG    |                32.1047 |              33.9403 |  24 |
| Symlet20WaveletV3AELG |                32.1474 |              32.2915 |  24 |
| Symlet3WaveletV3AELG  |                32.1557 |              32.8763 |  24 |
| Symlet2WaveletV3AELG  |                32.2724 |              32.3873 |  24 |
| DB4WaveletV3AELG      |                32.3908 |              34.0051 |  24 |
| Coif3WaveletV3AELG    |                32.4310 |              33.4566 |  24 |
| DB20WaveletV3AELG     |                32.4861 |              33.9895 |  24 |
| DB10WaveletV3AELG     |                32.5661 |              34.5981 |  24 |
| Symlet10WaveletV3AELG |                32.6264 |              40.3316 |  24 |
| Coif2WaveletV3AELG    |                32.7779 |              34.5219 |  24 |
| DB2WaveletV3AELG      |                32.7907 |              34.0286 |  44 |

#### bd_label
| bd_label   |   median_best_val_loss |   mean_best_val_loss |   n |
|:-----------|-----------------------:|---------------------:|----:|
| lt_bcast   |                31.7336 |              32.0799 |  92 |
| eq_fcast   |                31.7400 |              32.0984 | 102 |
| eq_bcast   |                32.2951 |              32.7833 |  96 |
| lt_fcast   |                37.9263 |              38.9648 |  97 |

#### trend_thetas_dim_cfg
|   trend_thetas_dim_cfg |   median_best_val_loss |   mean_best_val_loss |        n |
|-----------------------:|-----------------------:|---------------------:|---------:|
|                 3.0000 |                31.6371 |              33.3121 | 195.0000 |
|                 5.0000 |                32.9723 |              34.6683 | 192.0000 |

#### latent_dim_cfg
|   latent_dim_cfg |   median_best_val_loss |   mean_best_val_loss |        n |
|-----------------:|-----------------------:|---------------------:|---------:|
|          16.0000 |                32.2916 |              33.9850 | 387.0000 |

## TrendAELG — Dataset: Traffic-96

- Rows: 2
- Unique configs: 1

### Successive-Halving Funnel
|   Round |   Configs |   Rows |   Median best_val_loss | Kept   |
|--------:|----------:|-------:|-----------------------:|:-------|
|       1 |         1 |      2 |               200.0000 | -      |

### Round 3 Leaderboard (Top 10)
Round 3 not available.

### Round 1 Hyperparameter Marginals (median best_val_loss)
#### wavelet_family
| wavelet_family    |   median_best_val_loss |   mean_best_val_loss |   n |
|:------------------|-----------------------:|---------------------:|----:|
| HaarWaveletV3AELG |               200.0000 |             200.0000 |   2 |

#### bd_label
| bd_label   |   median_best_val_loss |   mean_best_val_loss |   n |
|:-----------|-----------------------:|---------------------:|----:|
| eq_fcast   |               200.0000 |             200.0000 |   2 |

#### trend_thetas_dim_cfg
|   trend_thetas_dim_cfg |   median_best_val_loss |   mean_best_val_loss |      n |
|-----------------------:|-----------------------:|---------------------:|-------:|
|                 3.0000 |               200.0000 |             200.0000 | 2.0000 |

#### latent_dim_cfg
|   latent_dim_cfg |   median_best_val_loss |   mean_best_val_loss |      n |
|-----------------:|-----------------------:|---------------------:|-------:|
|          16.0000 |               200.0000 |             200.0000 | 2.0000 |

## TrendAELG — Dataset: Weather-96

- Rows: 24
- Unique configs: 8

### Successive-Halving Funnel
|   Round |   Configs |   Rows |   Median best_val_loss | Kept   |
|--------:|----------:|-------:|-----------------------:|:-------|
|       1 |         8 |     24 |                43.4194 | -      |

### Round 3 Leaderboard (Top 10)
Round 3 not available.

### Round 1 Hyperparameter Marginals (median best_val_loss)
#### wavelet_family
| wavelet_family    |   median_best_val_loss |   mean_best_val_loss |   n |
|:------------------|-----------------------:|---------------------:|----:|
| HaarWaveletV3AELG |                43.4194 |              43.3806 |  24 |

#### bd_label
| bd_label   |   median_best_val_loss |   mean_best_val_loss |   n |
|:-----------|-----------------------:|---------------------:|----:|
| eq_bcast   |                43.2730 |              43.2347 |   6 |
| lt_fcast   |                43.4681 |              43.4019 |   6 |
| eq_fcast   |                43.5525 |              43.4429 |   6 |
| lt_bcast   |                43.5525 |              43.4429 |   6 |

#### trend_thetas_dim_cfg
|   trend_thetas_dim_cfg |   median_best_val_loss |   mean_best_val_loss |       n |
|-----------------------:|-----------------------:|---------------------:|--------:|
|                 5.0000 |                43.4194 |              43.4117 | 12.0000 |
|                 3.0000 |                43.4306 |              43.3495 | 12.0000 |

#### latent_dim_cfg
|   latent_dim_cfg |   median_best_val_loss |   mean_best_val_loss |       n |
|-----------------:|-----------------------:|---------------------:|--------:|
|          16.0000 |                43.4194 |              43.3806 | 24.0000 |

# Per-Dataset Summary

## M4-Yearly

- Round 3 not available.

## Tourism-Yearly

- Overall round-3 median best_val_loss: 24.9489
- Best config: **Coif1_eq_bcast_ttd5_ld16** (median=24.8333, n=3)

## Traffic-96

- Round 3 not available.

## Weather-96

- Round 3 not available.

# Cross-Dataset Robustness

## TrendAELG

- Missing or empty cross CSV: `experiments/results/wavelet_v3aelg_trendaelg_cross_dataset_results.csv`
