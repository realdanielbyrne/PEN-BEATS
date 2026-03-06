# VAE2 Block Study - Multi-Dataset Analysis


## Dataset: m4

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/vae2_study_results.csv`
- Rows: 882
- Primary metric: `owa`


## 1. Overview & Data Summary

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/vae2_study_results.csv`
- Total rows: 882
- Unique configs: 78
- Search rounds: [1, 2, 3]
- Primary metric: `owa`
- Backbone families: ['AELG', 'VAE', 'VAE1', 'VAE2']
- Latent dims: ['16', '32', '8']

|   Round |   Configs |   Rows | Epochs   | Passes                  |
|--------:|----------:|-------:|:---------|:------------------------|
|       1 |        78 |    468 | 15-15    | activeG_fcast, baseline |
|       2 |        52 |    312 | 18-25    | activeG_fcast, baseline |
|       3 |        17 |    102 | 18-95    | activeG_fcast, baseline |


## 2. Successive Halving Funnel

|   Round |   Configs |   Rows |   Best Med OWA | Kept        |
|--------:|----------:|-------:|---------------:|:------------|
|       1 |        78 |    468 |         0.8664 | -           |
|       2 |        52 |    312 |         0.8143 | 52/78 (67%) |
|       3 |        17 |    102 |         0.8038 | 17/52 (33%) |


## 3. Round Leaderboards

### Round 1

| Config                                     | Pass          | Backbone   |   LD |    OWA |    Std |   sMAPE |   MASE |   Params |
|:-------------------------------------------|:--------------|:-----------|-----:|-------:|-------:|--------:|-------:|---------:|
| DB4V3AELG+TrendAELG_ld16                   | activeG_fcast | AELG       |   16 | 0.8652 | 0.0168 | 14.5391 | 3.3409 |   625353 |
| GenericAELG+TrendAELG_ld32                 | baseline      | AELG       |   32 | 0.8717 | 0.0044 | 14.5823 | 3.3823 |   662409 |
| DB4V3AELG+TrendAELG_ld32                   | baseline      | AELG       |   32 | 0.8741 | 0.0258 | 14.6014 | 3.3966 |   662409 |
| DB4V3AELG+TrendAELG_ld32                   | activeG_fcast | AELG       |   32 | 0.8757 | 0.0161 | 14.6278 | 3.4027 |   662409 |
| GenericAELG+TrendAELG_ld16                 | activeG_fcast | AELG       |   16 | 0.8772 | 0.0187 | 14.6961 | 3.3986 |   625353 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld32 | activeG_fcast | AELG       |   32 | 0.8838 | 0.0434 | 14.7173 | 3.4452 |  4976006 |
| DB4V3AELG+TrendAELG_ld16                   | baseline      | AELG       |   16 | 0.8845 | 0.0208 | 14.7497 | 3.4429 |   625353 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld16 | baseline      | AELG       |   16 | 0.8845 | 0.0555 | 14.7636 | 3.4400 |  4885702 |
| TrendAELG+DB4WaveletV3AELG_ld32            | activeG_fcast | AELG       |   32 | 0.8847 | 0.0326 | 14.8546 | 3.4192 |   662409 |
| TrendAELG+GenericAELG_ld16                 | activeG_fcast | AELG       |   16 | 0.8877 | 0.0388 | 14.8280 | 3.4494 |   625353 |
| GenericAELG+TrendAELG_ld16                 | baseline      | AELG       |   16 | 0.8884 | 0.0239 | 14.8247 | 3.4557 |   625353 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld32 | activeG_fcast | AELG       |   32 | 0.8899 | 0.0454 | 14.8170 | 3.4695 |  4976006 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld32 | baseline      | AELG       |   32 | 0.8914 | 0.0253 | 14.8396 | 3.4759 |  4976006 |
| TrendAELG+DB4WaveletV3AELG_ld8             | baseline      | AELG       |    8 | 0.8923 | 0.0233 | 14.9930 | 3.4458 |   606825 |
| GenericAELG+TrendAELG_ld32                 | activeG_fcast | AELG       |   32 | 0.8950 | 0.0274 | 14.8491 | 3.5024 |   662409 |

### Round 2

| Config                                     | Pass          | Backbone   |   LD |    OWA |    Std |   sMAPE |   MASE |   Params |
|:-------------------------------------------|:--------------|:-----------|-----:|-------:|-------:|--------:|-------:|---------:|
| TrendAELG+SeasonalityAELG+GenericAELG_ld16 | baseline      | AELG       |   16 | 0.8126 | 0.0053 | 13.6896 | 3.1292 |  4885702 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld32 | baseline      | AELG       |   32 | 0.8153 | 0.0058 | 13.6905 | 3.1507 |  4976006 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld8  | baseline      | AELG       |    8 | 0.8206 | 0.0160 | 13.8282 | 3.1595 |  4840550 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld32 | activeG_fcast | AELG       |   32 | 0.8213 | 0.0172 | 13.8123 | 3.1692 |  4976006 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld32 | baseline      | AELG       |   32 | 0.8224 | 0.0144 | 13.8545 | 3.1673 |  4976006 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld32 | activeG_fcast | AELG       |   32 | 0.8233 | 0.0113 | 13.8628 | 3.1724 |  4976006 |
| GenericAELG+TrendAELG_ld32                 | activeG_fcast | AELG       |   32 | 0.8238 | 0.0140 | 13.8617 | 3.1770 |   662409 |
| TrendAELG+GenericAELG_ld16                 | activeG_fcast | AELG       |   16 | 0.8239 | 0.0053 | 13.8744 | 3.1744 |   625353 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld8  | activeG_fcast | AELG       |    8 | 0.8245 | 0.0064 | 13.9016 | 3.1728 |  4840550 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld8  | activeG_fcast | AELG       |    8 | 0.8271 | 0.0153 | 13.9942 | 3.1710 |  4840550 |
| GenericAELG+TrendAELG_ld16                 | baseline      | AELG       |   16 | 0.8272 | 0.0053 | 13.9586 | 3.1802 |   625353 |
| TrendAELG+GenericAELG_ld32                 | baseline      | AELG       |   32 | 0.8279 | 0.0136 | 13.9023 | 3.1991 |   662409 |
| TrendAELG+GenericAELG_ld16                 | baseline      | AELG       |   16 | 0.8281 | 0.0139 | 13.9266 | 3.1947 |   625353 |
| DB4V3AELG+TrendAELG_ld16                   | baseline      | AELG       |   16 | 0.8282 | 0.0045 | 13.9614 | 3.1877 |   625353 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld16 | activeG_fcast | AELG       |   16 | 0.8294 | 0.0099 | 13.9081 | 3.2100 |  4885702 |

### Round 3

| Config                                     | Pass          | Backbone   |   LD |    OWA |    Std |   sMAPE |   MASE |   Params |
|:-------------------------------------------|:--------------|:-----------|-----:|-------:|-------:|--------:|-------:|---------:|
| TrendAELG+DB4WaveletV3AELG_ld32            | baseline      | AELG       |   32 | 0.8016 | 0.0016 | 13.5169 | 3.0844 |   662409 |
| GenericAELG+TrendAELG_ld16                 | activeG_fcast | AELG       |   16 | 0.8017 | 0.0070 | 13.5086 | 3.0866 |   625353 |
| DB4V3AELG+TrendAELG_ld8                    | baseline      | AELG       |    8 | 0.8030 | 0.0051 | 13.5307 | 3.0918 |   606825 |
| DB4V3AELG+TrendAELG_ld16                   | baseline      | AELG       |   16 | 0.8044 | 0.0034 | 13.5415 | 3.1000 |   625353 |
| DB4V3AELG+TrendAELG_ld32                   | activeG_fcast | AELG       |   32 | 0.8047 | 0.0038 | 13.5388 | 3.1037 |   662409 |
| DB4V3AELG+TrendAELG_ld16                   | activeG_fcast | AELG       |   16 | 0.8064 | 0.0092 | 13.5621 | 3.1114 |   625353 |
| TrendAELG+DB4WaveletV3AELG_ld8             | baseline      | AELG       |    8 | 0.8067 | 0.0082 | 13.5710 | 3.1114 |   606825 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld32 | activeG_fcast | AELG       |   32 | 0.8071 | 0.0123 | 13.5824 | 3.1117 |  4976006 |
| GenericAELG+TrendAELG_ld32                 | baseline      | AELG       |   32 | 0.8088 | 0.0103 | 13.5831 | 3.1254 |   662409 |
| DB4V3AELG+TrendAELG_ld8                    | activeG_fcast | AELG       |    8 | 0.8119 | 0.0085 | 13.6553 | 3.1323 |   606825 |
| TrendAELG+GenericAELG_ld8                  | activeG_fcast | AELG       |    8 | 0.8126 | 0.0132 | 13.6575 | 3.1375 |   606825 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld32 | activeG_fcast | AELG       |   32 | 0.8144 | 0.0097 | 13.6312 | 3.1583 |  4976006 |
| TrendAELG+GenericAELG_ld8                  | baseline      | AELG       |    8 | 0.8167 | 0.0247 | 13.7450 | 3.1483 |   606825 |
| TrendAELG+DB4WaveletV3AELG_ld16            | baseline      | AELG       |   16 | 0.8167 | 0.0151 | 13.6744 | 3.1660 |   625353 |
| TrendAELG+DB4WaveletV3AELG_ld16            | activeG_fcast | AELG       |   16 | 0.8168 | 0.0168 | 13.6955 | 3.1618 |   625353 |


## 4. Hyperparameter Marginals (Round 1)

### backbone_family

| Value   |   Mean OWA |    Std |   N |
|:--------|-----------:|-------:|----:|
| AELG    |     0.9476 | 0.0942 | 144 |
| VAE1    |     1.6247 | 0.3406 |  36 |
| VAE     |     1.8555 | 0.6946 | 126 |
| VAE2    |     2.9213 | 0.5292 | 162 |

### category_label

| Value                     |   Mean OWA |    Std |   N |
|:--------------------------|-----------:|-------:|----:|
| trend_seasonality_generic |     1.6356 | 0.6535 |  54 |
| trend_generic             |     1.6504 | 0.7237 |  54 |
| generic                   |     1.7965 | 0.8116 |  54 |
| generic_trend             |     1.9335 | 1.0596 |  54 |
| trend_wavelet             |     1.9667 | 0.8634 |  54 |
| generic_seasonality_trend |     2.0504 | 1.2200 |  54 |
| trend_wavelet_generic     |     2.0823 | 0.5563 |  36 |
| generic_wavelet_trend     |     2.1310 | 1.0453 |  54 |
| wavelet_trend             |     2.1511 | 1.1043 |  54 |

### latent_dim

|   Value |   Mean OWA |    Std |   N |
|--------:|-----------:|-------:|----:|
|       8 |     1.8426 | 0.8574 | 156 |
|      16 |     1.9641 | 0.9888 | 156 |
|      32 |     1.9752 | 0.9698 | 156 |

### wavelet_family (wavelet configs only)

| Value     |   Mean OWA |    Std |   N |
|:----------|-----------:|-------:|----:|
| DB4AELG   |     0.9084 | 0.0553 |  36 |
| Coif2AELG |     1.1209 | 0.0710 |  18 |
| Coif2VAE  |     1.6894 | 0.3270 |  36 |
| DB4VAE    |     2.0241 | 0.2356 |  36 |
| Coif2VAE2 |     3.0290 | 0.5497 |  36 |
| DB4VAE2   |     3.2441 | 0.3682 |  36 |


## 5. Stability Analysis

### Round 1

- Mean spread: 0.5997
- Max spread: 3.4805 (GenericVAE+SeasonalityVAE+TrendVAE_ld16)
- Mean std: 0.2371

### Round 2

- Mean spread: 0.2601
- Max spread: 3.4949 (GenericVAE+SeasonalityVAE+TrendVAE_ld32)
- Mean std: 0.1060

### Round 3

- Mean spread: 0.0617
- Max spread: 0.2774 (GenericAELG+SeasonalityAELG+TrendAELG_ld16)
- Mean std: 0.0245


## 6. Round-over-Round Progression

| config_name                                |     R1 |     R2 |     R3 |   Delta |   DeltaPct |
|:-------------------------------------------|-------:|-------:|-------:|--------:|-----------:|
| TrendAELG+GenericAELG_ld32                 | 0.9345 | 0.8357 | 0.814  | -0.1205 |      -12.9 |
| DB4V3AELG+TrendAELG_ld8                    | 0.9098 | 0.8539 | 0.8059 | -0.1039 |      -11.4 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld16 | 0.9246 | 0.8185 | 0.8246 | -0.1    |      -10.8 |
| TrendAELG+DB4WaveletV3AELG_ld32            | 0.9014 | 0.8438 | 0.8088 | -0.0926 |      -10.3 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld16 | 0.9165 | 0.841  | 0.8226 | -0.0939 |      -10.2 |
| TrendAELG+GenericAELG_ld16                 | 0.9072 | 0.8259 | 0.8168 | -0.0904 |      -10   |
| TrendAELG+DB4WaveletV3AELG_ld8             | 0.9057 | 0.8424 | 0.8163 | -0.0893 |       -9.9 |
| TrendAELG+DB4WaveletV3AELG_ld16            | 0.9061 | 0.8383 | 0.8169 | -0.0892 |       -9.8 |
| TrendAELG+GenericAELG_ld8                  | 0.8949 | 0.8384 | 0.8077 | -0.0873 |       -9.8 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld8  | 0.9086 | 0.822  | 0.8256 | -0.083  |       -9.1 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld32 | 0.8967 | 0.8143 | 0.8197 | -0.077  |       -8.6 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld32 | 0.886  | 0.8263 | 0.8154 | -0.0706 |       -8   |
| GenericAELG+TrendAELG_ld16                 | 0.8759 | 0.8296 | 0.8061 | -0.0698 |       -8   |
| DB4V3AELG+TrendAELG_ld16                   | 0.8725 | 0.8298 | 0.8063 | -0.0662 |       -7.6 |
| DB4V3AELG+TrendAELG_ld32                   | 0.8664 | 0.8291 | 0.8038 | -0.0626 |       -7.2 |
| GenericAELG+TrendAELG_ld32                 | 0.874  | 0.8321 | 0.8217 | -0.0523 |       -6   |
| GenericAELG+SeasonalityAELG+TrendAELG_ld8  | 0.8863 | 0.8281 | 0.8475 | -0.0388 |       -4.4 |


## 7. Baseline Comparisons

### Top-10 VAE2-Study Configs (Final Round)

| Config                                     | Pass          | Backbone   |    OWA |   sMAPE |   Params |   vs NBEATS-G |
|:-------------------------------------------|:--------------|:-----------|-------:|--------:|---------:|--------------:|
| TrendAELG+DB4WaveletV3AELG_ld32            | baseline      | AELG       | 0.8016 | 13.5169 |   662409 |       -0.0604 |
| GenericAELG+TrendAELG_ld16                 | activeG_fcast | AELG       | 0.8017 | 13.5086 |   625353 |       -0.0603 |
| DB4V3AELG+TrendAELG_ld8                    | baseline      | AELG       | 0.8030 | 13.5307 |   606825 |       -0.0590 |
| DB4V3AELG+TrendAELG_ld16                   | baseline      | AELG       | 0.8044 | 13.5415 |   625353 |       -0.0576 |
| DB4V3AELG+TrendAELG_ld32                   | activeG_fcast | AELG       | 0.8047 | 13.5388 |   662409 |       -0.0573 |
| DB4V3AELG+TrendAELG_ld16                   | activeG_fcast | AELG       | 0.8064 | 13.5621 |   625353 |       -0.0556 |
| TrendAELG+DB4WaveletV3AELG_ld8             | baseline      | AELG       | 0.8067 | 13.5710 |   606825 |       -0.0553 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld32 | activeG_fcast | AELG       | 0.8071 | 13.5824 |  4976006 |       -0.0549 |
| GenericAELG+TrendAELG_ld32                 | baseline      | AELG       | 0.8088 | 13.5831 |   662409 |       -0.0532 |
| DB4V3AELG+TrendAELG_ld8                    | activeG_fcast | AELG       | 0.8119 | 13.6553 |   606825 |       -0.0501 |

### Reference Baselines

| Baseline   |    OWA | sMAPE   |   Params |
|:-----------|-------:|:--------|---------:|
| NBEATS-G   | 0.8620 | 13.7    | 24700000 |
| Naive2     | 1.0000 | N/A     |        0 |


## 8. Backbone Family Head-to-Head

### VAE2 vs VAE1

No matched pairs found for VAE2 vs VAE1.

### VAE2 vs AELG

No matched pairs found for VAE2 vs AELG.

### VAE1 vs AELG

No matched pairs found for VAE1 vs AELG.


## 9. Final Verdict

Best configuration: **GenericAELG+SeasonalityAELG+TrendAELG_ld8** (backbone=AELG, pass=baseline) with median OWA=0.8010.
vs NBEATS-G (0.8620): beats (delta=-0.0610).

| Config                                     | Pass          | Backbone   |   Med OWA |    Std |   Params |   sMAPE |   MASE |
|:-------------------------------------------|:--------------|:-----------|----------:|-------:|---------:|--------:|-------:|
| GenericAELG+SeasonalityAELG+TrendAELG_ld8  | baseline      | AELG       |    0.8010 | 0.0523 |  4840550 | 13.5097 | 3.0811 |
| TrendAELG+DB4WaveletV3AELG_ld8             | baseline      | AELG       |    0.8023 | 0.0082 |   606825 | 13.5009 | 3.0951 |
| TrendAELG+DB4WaveletV3AELG_ld32            | baseline      | AELG       |    0.8025 | 0.0016 |   662409 | 13.5206 | 3.0902 |
| DB4V3AELG+TrendAELG_ld32                   | baseline      | AELG       |    0.8029 | 0.0390 |   662409 | 13.5203 | 3.0936 |
| DB4V3AELG+TrendAELG_ld32                   | activeG_fcast | AELG       |    0.8047 | 0.0038 |   662409 | 13.5339 | 3.1043 |
| TrendAELG+GenericAELG_ld8                  | baseline      | AELG       |    0.8047 | 0.0247 |   606825 | 13.5562 | 3.0996 |
| DB4V3AELG+TrendAELG_ld8                    | baseline      | AELG       |    0.8053 | 0.0051 |   606825 | 13.5638 | 3.1019 |
| GenericAELG+TrendAELG_ld16                 | activeG_fcast | AELG       |    0.8056 | 0.0070 |   625353 | 13.5488 | 3.1061 |
| GenericAELG+SeasonalityAELG+TrendAELG_ld32 | activeG_fcast | AELG       |    0.8057 | 0.0123 |  4976006 | 13.5390 | 3.1116 |
| DB4V3AELG+TrendAELG_ld16                   | baseline      | AELG       |    0.8061 | 0.0034 |   625353 | 13.5563 | 3.1082 |


## Dataset: weather

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/weather/vae2_study_results.csv`
- Rows: 515
- Primary metric: `norm_mae`


## 1. Overview & Data Summary

- CSV: `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/weather/vae2_study_results.csv`
- Total rows: 515
- Unique configs: 78
- Search rounds: [1, 2]
- Primary metric: `norm_mae`
- Backbone families: ['AELG', 'VAE', 'VAE1', 'VAE2']
- Latent dims: ['16', '32', '8']

|   Round |   Configs |   Rows | Epochs   | Passes                  |
|--------:|----------:|-------:|:---------|:------------------------|
|       1 |        78 |    468 | 13-15    | activeG_fcast, baseline |
|       2 |        16 |     47 | 13-25    | baseline                |


## 2. Successive Halving Funnel

|   Round |   Configs |   Rows |   Best Med norm_mae | Kept        |
|--------:|----------:|-------:|--------------------:|:------------|
|       1 |        78 |    468 |              0.3387 | -           |
|       2 |        16 |     47 |              0.3382 | 16/78 (21%) |


## 3. Round Leaderboards

### Round 1

| Config                                     | Pass          | Backbone   |   LD |   norm_mae |    Std |   sMAPE |   MASE |   Params |
|:-------------------------------------------|:--------------|:-----------|-----:|-----------:|-------:|--------:|-------:|---------:|
| TrendAELG+GenericAELG_ld16                 | activeG_fcast | AELG       |   16 |     0.3368 | 0.0143 | 65.4980 | 0.9297 |  1199049 |
| TrendAELG+GenericAELG_ld16                 | baseline      | AELG       |   16 |     0.3414 | 0.0103 | 65.5957 | 0.9297 |  1199049 |
| TrendAELG+DB4WaveletV3AELG_ld8             | baseline      | AELG       |    8 |     0.3444 | 0.0226 | 65.8869 | 0.9303 |  1082217 |
| GenericAELG+TrendAELG_ld8                  | baseline      | AELG       |    8 |     0.3495 | 0.0137 | 66.1839 | 0.9685 |  1180521 |
| TrendAELG+DB4WaveletV3AELG_ld16            | activeG_fcast | AELG       |   16 |     0.3502 | 0.0239 | 65.5950 | 0.9597 |  1100745 |
| TrendAELG+DB4WaveletV3AELG_ld32            | baseline      | AELG       |   32 |     0.3523 | 0.0369 | 66.1202 | 0.9582 |  1137801 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld16 | activeG_fcast | AELG       |   16 |     0.3525 | 0.0297 | 66.0434 | 0.9839 |  6632134 |
| GenericAELG+TrendAELG_ld32                 | activeG_fcast | AELG       |   32 |     0.3557 | 0.0245 | 66.1198 | 0.9760 |  1236105 |
| TrendAELG+DB4WaveletV3AELG_ld16            | baseline      | AELG       |   16 |     0.3562 | 0.0052 | 66.2063 | 0.9960 |  1100745 |
| TrendAELG+DB4WaveletV3AELG_ld32            | activeG_fcast | AELG       |   32 |     0.3567 | 0.0107 | 65.1620 | 0.9638 |  1137801 |
| TrendAELG+GenericAELG_ld8                  | baseline      | AELG       |    8 |     0.3570 | 0.0138 | 65.7064 | 0.9999 |  1180521 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld32 | baseline      | AELG       |   32 |     0.3581 | 0.0218 | 66.0442 | 0.9723 |  6722438 |
| DB4V3AELG+TrendAELG_ld32                   | activeG_fcast | AELG       |   32 |     0.3591 | 0.0298 | 65.8951 | 0.9629 |  1137801 |
| DB4V3AELG+TrendAELG_ld8                    | activeG_fcast | AELG       |    8 |     0.3609 | 0.0272 | 65.4894 | 0.9509 |  1082217 |
| GenericAELG+TrendAELG_ld16                 | baseline      | AELG       |   16 |     0.3633 | 0.0196 | 65.9290 | 0.9970 |  1199049 |

### Round 2

| Config                                     | Pass     | Backbone   |   LD |   norm_mae |    Std |   sMAPE |   MASE |   Params |
|:-------------------------------------------|:---------|:-----------|-----:|-----------:|-------:|--------:|-------:|---------:|
| TrendAELG+SeasonalityAELG+GenericAELG_ld8  | baseline | AELG       |    8 |     0.3404 | 0.0237 | 65.8925 | 0.9400 |  6586982 |
| GenericVAE2+SeasonalityVAE2+TrendVAE2_ld8  | baseline | VAE2       |    8 |     0.3420 | 0.0290 | 64.9049 | 0.9486 |  6609510 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld32 | baseline | AELG       |   32 |     0.3581 | 0.0218 | 66.0442 | 0.9723 |  6722438 |
| TrendVAE+SeasonalityVAE+GenericVAE_ld16    | baseline | VAE1       |   16 |     0.3761 | 0.0079 | 65.7624 | 1.0480 |  8873414 |
| GenericVAE2+SeasonalityVAE2+TrendVAE2_ld16 | baseline | VAE2       |   16 |     0.3776 | 0.0218 | 65.1425 | 1.0266 |  6677190 |
| GenericAELG_ld8                            | baseline | AELG       |    8 |     0.3800 | 0.0417 | 66.1525 | 1.0111 |  1664080 |
| TrendVAE2+SeasonalityVAE2+GenericVAE2_ld8  | baseline | VAE2       |    8 |     0.3837 | 0.0348 | 65.5093 | 1.1300 |  6609510 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld16 | baseline | AELG       |   16 |     0.3904 | 0.0549 | 66.2057 | 1.1120 |  6632134 |
| TrendVAE2+SeasonalityVAE2+GenericVAE2_ld16 | baseline | VAE2       |   16 |     0.3986 | 0.0234 | 66.0225 | 1.2015 |  6677190 |
| TrendVAE2+SeasonalityVAE2+GenericVAE2_ld32 | baseline | VAE2       |   32 |     0.4032 | 0.0370 | 65.3666 | 1.2531 |  6812550 |
| GenericVAE2_ld8                            | baseline | VAE2       |    8 |     0.4091 | 0.0152 | 66.2492 | 1.1729 |  1674320 |
| GenericVAE_ld16                            | baseline | VAE1       |   16 |     0.4219 | 0.0641 | 66.0908 | 1.1491 |  2003360 |
| GenericAELG_ld16                           | baseline | AELG       |   16 |     0.4435 | 0.0242 | 65.9325 | 1.1422 |  1684640 |
| GenericVAE2+SeasonalityVAE2+TrendVAE2_ld32 | baseline | VAE2       |   32 |     0.4476 | 0.0472 | 65.9564 | 1.3018 |  6812550 |
| GenericAELG_ld32                           | baseline | AELG       |   32 |     0.4590 | 0.0388 | 66.3654 | 1.1976 |  1725760 |


## 4. Hyperparameter Marginals (Round 1)

### backbone_family

| Value   |   Mean norm_mae |    Std |   N |
|:--------|----------------:|-------:|----:|
| AELG    |          0.4085 | 0.0863 | 144 |
| VAE2    |          0.5427 | 0.1796 | 162 |
| VAE     |          0.6814 | 0.2187 | 126 |
| VAE1    |          0.7459 | 0.2476 |  36 |

### category_label

| Value                     |   Mean norm_mae |    Std |   N |
|:--------------------------|----------------:|-------:|----:|
| trend_generic             |          0.4848 | 0.1647 |  54 |
| trend_wavelet             |          0.4920 | 0.1705 |  54 |
| wavelet_trend             |          0.5050 | 0.1604 |  54 |
| generic_trend             |          0.5086 | 0.2142 |  54 |
| trend_seasonality_generic |          0.5422 | 0.2478 |  54 |
| trend_wavelet_generic     |          0.5663 | 0.1402 |  36 |
| generic_seasonality_trend |          0.5993 | 0.2547 |  54 |
| generic_wavelet_trend     |          0.6076 | 0.1704 |  54 |
| generic                   |          0.6874 | 0.2448 |  54 |

### latent_dim

|   Value |   Mean norm_mae |    Std |   N |
|--------:|----------------:|-------:|----:|
|      16 |          0.5243 | 0.1820 | 156 |
|       8 |          0.5285 | 0.1654 | 156 |
|      32 |          0.6103 | 0.2629 | 156 |

### wavelet_family (wavelet configs only)

| Value     |   Mean norm_mae |    Std |   N |
|:----------|----------------:|-------:|----:|
| DB4AELG   |          0.3655 | 0.0311 |  36 |
| DB4VAE2   |          0.4609 | 0.0877 |  36 |
| Coif2VAE2 |          0.5249 | 0.1083 |  36 |
| Coif2AELG |          0.5380 | 0.1202 |  18 |
| DB4VAE    |          0.6692 | 0.1576 |  36 |
| Coif2VAE  |          0.6838 | 0.1764 |  36 |


## 5. Stability Analysis

### Round 1

- Mean spread: 0.3040
- Max spread: 1.2074 (GenericVAE+TrendVAE_ld32)
- Mean std: 0.1139

### Round 2

- Mean spread: 0.0722
- Max spread: 0.2410 (GenericVAE2_ld16)
- Mean std: 0.0387


## 6. Round-over-Round Progression

| config_name                                |     R1 |     R2 |   Delta |   DeltaPct |
|:-------------------------------------------|-------:|-------:|--------:|-----------:|
| GenericVAE2_ld8                            | 0.8479 | 0.4143 | -0.4336 |      -51.1 |
| TrendVAE+SeasonalityVAE+GenericVAE_ld16    | 0.7285 | 0.3771 | -0.3514 |      -48.2 |
| GenericVAE2+SeasonalityVAE2+TrendVAE2_ld32 | 0.7927 | 0.4476 | -0.3451 |      -43.5 |
| GenericVAE2+SeasonalityVAE2+TrendVAE2_ld8  | 0.5667 | 0.3382 | -0.2284 |      -40.3 |
| GenericVAE_ld16                            | 0.6006 | 0.4046 | -0.196  |      -32.6 |
| GenericVAE2_ld16                           | 0.6244 | 0.449  | -0.1754 |      -28.1 |
| TrendVAE2+SeasonalityVAE2+GenericVAE2_ld32 | 0.4917 | 0.3874 | -0.1043 |      -21.2 |
| GenericVAE2+SeasonalityVAE2+TrendVAE2_ld16 | 0.4558 | 0.3889 | -0.067  |      -14.7 |
| TrendVAE2+SeasonalityVAE2+GenericVAE2_ld8  | 0.4315 | 0.3749 | -0.0566 |      -13.1 |
| GenericAELG_ld16                           | 0.5078 | 0.4443 | -0.0635 |      -12.5 |
| TrendVAE2+SeasonalityVAE2+GenericVAE2_ld16 | 0.4515 | 0.3969 | -0.0546 |      -12.1 |
| GenericAELG_ld8                            | 0.4406 | 0.3938 | -0.0468 |      -10.6 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld8  | 0.3706 | 0.3413 | -0.0293 |       -7.9 |
| GenericAELG_ld32                           | 0.4798 | 0.4743 | -0.0055 |       -1.1 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld32 | 0.3607 | 0.3577 | -0.003  |       -0.8 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld16 | 0.3717 | 0.4076 |  0.0359 |        9.6 |


## 7. Baseline Comparisons

Section skipped (M4 OWA baselines not applicable for this dataset).


## 8. Backbone Family Head-to-Head

### VAE2 vs VAE1

| Base Config                                            | Pass     |   VAE2_norm_mae |   VAE1_norm_mae |   Delta (B-A) | Winner   |
|:-------------------------------------------------------|:---------|----------------:|----------------:|--------------:|:---------|
| GenericBACKBONE_ld16                                   | baseline |          0.4490 |          0.4046 |       -0.0444 | VAE1     |
| TrendBACKBONE+SeasonalityBACKBONE+GenericBACKBONE_ld16 | baseline |          0.3969 |          0.3771 |       -0.0199 | VAE1     |

**Score: VAE2 0 — VAE1 2 — Tie 0**

### VAE2 vs AELG

| Base Config                                            | Pass     |   VAE2_norm_mae |   AELG_norm_mae |   Delta (B-A) | Winner   |
|:-------------------------------------------------------|:---------|----------------:|----------------:|--------------:|:---------|
| GenericBACKBONE_ld16                                   | baseline |          0.4490 |          0.4443 |       -0.0047 | AELG     |
| GenericBACKBONE_ld8                                    | baseline |          0.4143 |          0.3938 |       -0.0205 | AELG     |
| TrendBACKBONE+SeasonalityBACKBONE+GenericBACKBONE_ld16 | baseline |          0.3969 |          0.4076 |        0.0106 | VAE2     |
| TrendBACKBONE+SeasonalityBACKBONE+GenericBACKBONE_ld32 | baseline |          0.3874 |          0.3577 |       -0.0296 | AELG     |
| TrendBACKBONE+SeasonalityBACKBONE+GenericBACKBONE_ld8  | baseline |          0.3749 |          0.3413 |       -0.0336 | AELG     |

**Score: VAE2 1 — AELG 4 — Tie 0**

### VAE1 vs AELG

| Base Config                                            | Pass     |   VAE1_norm_mae |   AELG_norm_mae |   Delta (B-A) | Winner   |
|:-------------------------------------------------------|:---------|----------------:|----------------:|--------------:|:---------|
| GenericBACKBONE_ld16                                   | baseline |          0.4046 |          0.4443 |        0.0397 | VAE1     |
| TrendBACKBONE+SeasonalityBACKBONE+GenericBACKBONE_ld16 | baseline |          0.3771 |          0.4076 |        0.0305 | VAE1     |

**Score: VAE1 2 — AELG 0 — Tie 0**


## 9. Final Verdict

Best configuration: **GenericVAE2+SeasonalityVAE2+TrendVAE2_ld8** (backbone=VAE2, pass=baseline) with median norm_mae=0.3382.
Primary metric: norm_mae (lower is better). OWA baselines not available for this dataset.

| Config                                     | Pass     | Backbone   |   Med norm_mae |    Std |   Params |   sMAPE |   MASE |
|:-------------------------------------------|:---------|:-----------|---------------:|-------:|---------:|--------:|-------:|
| GenericVAE2+SeasonalityVAE2+TrendVAE2_ld8  | baseline | VAE2       |         0.3382 | 0.0290 |  6609510 | 64.4914 | 0.9481 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld8  | baseline | AELG       |         0.3413 | 0.0237 |  6586982 | 66.0940 | 0.9499 |
| TrendAELG+SeasonalityAELG+GenericAELG_ld32 | baseline | AELG       |         0.3577 | 0.0218 |  6722438 | 65.7339 | 0.9696 |
| TrendVAE2+SeasonalityVAE2+GenericVAE2_ld8  | baseline | VAE2       |         0.3749 | 0.0348 |  6609510 | 65.5821 | 1.0889 |
| TrendVAE+SeasonalityVAE+GenericVAE_ld16    | baseline | VAE1       |         0.3771 | 0.0079 |  8873414 | 66.1262 | 1.0652 |
| TrendVAE2+SeasonalityVAE2+GenericVAE2_ld32 | baseline | VAE2       |         0.3874 | 0.0370 |  6812550 | 64.8077 | 1.2046 |
| GenericVAE2+SeasonalityVAE2+TrendVAE2_ld16 | baseline | VAE2       |         0.3889 | 0.0218 |  6677190 | 64.8716 | 1.0474 |
| GenericAELG_ld8                            | baseline | AELG       |         0.3938 | 0.0417 |  1664080 | 65.6201 | 1.0562 |
| TrendVAE2+SeasonalityVAE2+GenericVAE2_ld16 | baseline | VAE2       |         0.3969 | 0.0234 |  6677190 | 66.2270 | 1.1785 |
| GenericVAE_ld16                            | baseline | VAE1       |         0.4046 | 0.0641 |  2003360 | 65.9286 | 1.1803 |


# Summary

- analyzed_count: 2
- skipped_count: 0
- analyzed: ['m4', 'weather']
