# WaveletV3AE Study Analysis

## Dataset: M4-Yearly

- CSV: `experiments/results/m4/wavelet_v3ae_study_results.csv`
- Rows: 1161
- Unique configs: 332

### Successive-Halving Funnel
|   Round |   Configs |   Rows |   Median best_val_loss | Kept   |
|--------:|----------:|-------:|-----------------------:|:-------|
|       1 |       332 |   1161 |                16.6808 | -      |

### Round 3 Leaderboard (Top 10)
Round 3 not available.

### Round 1 Hyperparameter Marginals (median best_val_loss)
#### wavelet_family
| wavelet_family      |   median_best_val_loss |   mean_best_val_loss |   n |
|:--------------------|-----------------------:|---------------------:|----:|
| DB20WaveletV3AE     |                16.4868 |              16.8873 |  73 |
| Coif3WaveletV3AE    |                16.5001 |              16.8307 |  72 |
| Coif2WaveletV3AE    |                16.5588 |              16.9532 |  72 |
| Symlet20WaveletV3AE |                16.5718 |              16.8158 |  59 |
| Coif10WaveletV3AE   |                16.6091 |              17.0956 |  72 |
| DB4WaveletV3AE      |                16.6406 |              17.1116 | 144 |
| DB10WaveletV3AE     |                16.6460 |              16.9741 | 144 |
| DB3WaveletV3AE      |                16.7044 |              17.0148 |  93 |
| HaarWaveletV3AE     |                16.7520 |              17.2416 |  72 |
| Symlet2WaveletV3AE  |                16.7531 |              16.9675 |  72 |
| Coif1WaveletV3AE    |                16.7562 |              17.1589 |  72 |
| Symlet10WaveletV3AE |                16.7716 |              17.4020 |  72 |
| DB2WaveletV3AE      |                16.8633 |              17.1249 |  72 |
| Symlet3WaveletV3AE  |                16.8705 |              17.0842 |  72 |

#### bd_label
| bd_label   |   median_best_val_loss |   mean_best_val_loss |   n |
|:-----------|-----------------------:|---------------------:|----:|
| eq_bcast   |                16.2698 |              16.7333 | 291 |
| lt_bcast   |                16.4919 |              16.6481 | 293 |
| eq_fcast   |                16.5169 |              16.8599 | 289 |
| lt_fcast   |                17.7014 |              17.9641 | 288 |

#### trend_thetas_dim_cfg
|   trend_thetas_dim_cfg |   median_best_val_loss |   mean_best_val_loss |        n |
|-----------------------:|-----------------------:|---------------------:|---------:|
|                 3.0000 |                16.3917 |              16.7941 | 582.0000 |
|                 5.0000 |                16.9382 |              17.3044 | 579.0000 |

#### latent_dim_cfg
|   latent_dim_cfg |   median_best_val_loss |   mean_best_val_loss |        n |
|-----------------:|-----------------------:|---------------------:|---------:|
|           8.0000 |                16.3985 |              16.6535 | 387.0000 |
|           5.0000 |                16.5175 |              16.7861 | 386.0000 |
|           2.0000 |                17.2914 |              17.7038 | 388.0000 |

## Dataset: Tourism-Yearly

- Missing CSV: `experiments/results/tourism/wavelet_v3ae_study_results.csv`

## Dataset: Traffic-96

- Missing CSV: `experiments/results/traffic/wavelet_v3ae_study_results.csv`

## Dataset: Weather-96

- Missing CSV: `experiments/results/weather/wavelet_v3ae_study_results.csv`

## Global Top-10 Selection Overview

No candidates found from round-3 dataset leaderboards.

## Cross-Dataset Robustness Leaderboard

- Missing or empty cross CSV: `experiments/results/wavelet_v3ae_cross_dataset_results.csv`
