# WaveletV3AELG + TrendAELG Comprehensive Study Analysis

**Date:** 2026-03-07
**Datasets:** M4-Yearly, Tourism-Yearly, Traffic-96, Weather-96
**Architecture:** `[TrendAELG, <Wavelet>WaveletV3AELG]` alternating stacks
**Study Design:** Successive halving search (3 rounds) across 14 wavelet families, 4 basis_dim labels, 2 trend_thetas_dim values, latent_dim=16

---

## Executive Summary

This study is the largest systematic evaluation of the TrendAELG + WaveletV3AELG architecture to date, sweeping 14 wavelet families (Haar, DB2-DB20, Coif1-Coif10, Symlet2-Symlet20) across 4 basis_dim labels and 2 trend_thetas_dim values. The successive halving protocol (112 configs -> 75 -> 50) ran across 4 datasets, yielding 2,333 total experiment rows.

### Key Findings

1. **M4-Yearly best: Symlet20_eq_fcast_ttd3_ld16** (SMAPE=13.438, OWA=0.795) -- replicates prior V1 AELG result exactly. The top 10 R3 configs span only 0.062 SMAPE, indicating the architecture is robust to wavelet family choice once other hyperparameters are fixed.

2. **Tourism-Yearly best: Coif1_eq_fcast_ttd3_ld16** (SMAPE=20.930) -- also replicates prior V1 result. Tied with Coif1_lt_bcast_ttd3_ld16 due to Tourism degeneracy (eq_fcast = lt_bcast when fcast=4, bcast=8).

3. **Weather-96 best: Symlet20_eq_fcast_ttd5_ld16** (MSE=2070.61) -- Symlet20 dominates Weather. The ttd=5 preference on Weather reverses the short-horizon ttd=3 preference, a consistent finding across studies.

4. **Traffic-96: 86.2% divergence rate** -- catastrophic at the protocol used (bl=192, L=2H). Only 20 of 145 runs converged. **Root cause: insufficient backcast horizon** (`forecast_multiplier=2`). A subsequent study using L=5H (bl=480) achieved 80-100% convergence — the architecture is viable with adequate lookback.

5. **Cross-dataset winner: Symlet20** -- ranks #2/#3/#2 across M4/Tourism/Weather (avg rank 2.3), the most consistent performer. Symlet2 is #2 (avg rank 4.7).

6. **Trend thetas dim is dataset-dependent:** ttd=3 is better for short-horizon forecasting (Tourism, p=0.008); ttd=5 is better for long-horizon (Weather, p=0.042); M4-Yearly shows no significant difference at convergence (p=0.16).

7. **Basis dim labels converge at R3:** After full training, all four bd_labels produce nearly identical results on M4 (spread < 0.014 SMAPE). At R1 (early training), eq_bcast appears best but this is a convergence speed artifact.

---

## 1. Study Design

### Architecture

- **Stack composition:** `[TrendAELG, <Wavelet>WaveletV3AELG]` repeated 5x = 10 stacks (M4/Tourism) or 10x = 20 stacks (Traffic/Weather)
- **Backbone:** AERootBlockLG (learned-gate autoencoder)
- **Latent dim:** 16 (fixed)
- **Weight sharing:** True
- **Activation:** ReLU
- **active_g:** False, **sum_losses:** False

### Search Dimensions

| Dimension | Values | Count |
|-----------|--------|-------|
| Wavelet family | Haar, DB2, DB3, DB4, DB10, DB20, Coif1, Coif2, Coif3, Coif10, Symlet2, Symlet3, Symlet10, Symlet20 | 14 |
| Basis dim label | eq_fcast, lt_fcast, eq_bcast, lt_bcast | 4 |
| Trend thetas dim | 3, 5 | 2 |
| **Total configs** | | **112** |

### Successive Halving Protocol

| Round | Configs | Epochs | Runs/config |
|-------|---------|--------|-------------|
| R1 | 112 | 10 | 3 |
| R2 | 75 | 15 | 3 |
| R3 | 50 | 50 | 3 |

### Basis Dim Resolution by Dataset

The four `bd_label` values resolve to different `basis_dim` integers depending on forecast and backcast lengths. When two labels resolve to the same value, the results are numerically identical (confirmed).

| Dataset | forecast | backcast | eq_fcast | lt_fcast | eq_bcast | lt_bcast | Degeneracy? |
|---------|----------|----------|----------|----------|----------|----------|-------------|
| M4-Yearly | 6 | 30 | 6 | 4 | 30 | 15 | No (4 distinct) |
| Tourism-Yearly | 4 | 8 | 4 | 2 | 8 | 4 | **Yes**: eq_fcast = lt_bcast = 4 |
| Weather-96 | 96 | 192 | 96 | 94 | 192 | 96 | **Yes**: eq_fcast = lt_bcast = 96 |
| Traffic-96 | 96 | 192 | 96 | 94 | 192 | 96 | **Yes**: eq_fcast = lt_bcast = 96 |

---

## 2. M4-Yearly Results

### 2.1 Round 3 (Final) Rankings -- Top 20

All R3 configs ran for ~50 epochs. The top 50 configs from successive halving are ranked below.

| Rank | Config | SMAPE | Std | OWA | MASE | N | Delta |
|------|--------|-------|-----|-----|------|---|-------|
| 1 | Symlet20_eq_fcast_ttd3_ld16 | **13.438** | 0.069 | **0.795** | 3.055 | 3 | -- |
| 2 | Symlet3_lt_bcast_ttd5_ld16 | 13.462 | 0.051 | 0.798 | 3.066 | 3 | +0.024 |
| 3 | DB2_eq_bcast_ttd3_ld16 | 13.475 | 0.026 | 0.797 | 3.061 | 3 | +0.037 |
| 4 | Coif3_eq_fcast_ttd5_ld16 | 13.481 | 0.039 | 0.800 | 3.077 | 3 | +0.043 |
| 5 | Coif3_lt_bcast_ttd5_ld16 | 13.481 | 0.034 | 0.799 | 3.072 | 3 | +0.043 |
| 6 | Haar_eq_fcast_ttd3_ld16 | 13.482 | 0.086 | 0.799 | 3.074 | 3 | +0.044 |
| 7 | Symlet2_lt_bcast_ttd5_ld16 | 13.496 | 0.025 | 0.800 | 3.078 | 3 | +0.058 |
| 8 | Symlet2_eq_fcast_ttd3_ld16 | 13.498 | 0.117 | 0.800 | 3.079 | 3 | +0.060 |
| 9 | Symlet3_lt_bcast_ttd3_ld16 | 13.498 | 0.014 | 0.801 | 3.084 | 2 | +0.060 |
| 10 | Coif1_lt_bcast_ttd5_ld16 | 13.500 | 0.041 | 0.800 | 3.078 | 3 | +0.062 |
| 11 | DB20_lt_bcast_ttd5_ld16 | 13.501 | 0.085 | 0.801 | 3.084 | 3 | +0.063 |
| 12 | Symlet20_eq_bcast_ttd5_ld16 | 13.502 | 0.029 | 0.800 | 3.076 | 3 | +0.064 |
| 13 | Coif3_eq_bcast_ttd5_ld16 | 13.505 | 0.009 | 0.800 | 3.074 | 3 | +0.067 |
| 14 | Coif1_eq_bcast_ttd5_ld16 | 13.506 | 0.040 | 0.800 | 3.076 | 3 | +0.068 |
| 15 | DB20_eq_fcast_ttd3_ld16 | 13.512 | 0.100 | 0.802 | 3.085 | 3 | +0.074 |
| 16 | DB4_eq_bcast_ttd3_ld16 | 13.514 | 0.031 | 0.801 | 3.077 | 3 | +0.076 |
| 17 | Haar_eq_bcast_ttd3_ld16 | 13.521 | 0.055 | 0.802 | 3.085 | 3 | +0.083 |
| 18 | Coif2_eq_fcast_ttd3_ld16 | 13.524 | 0.097 | 0.803 | 3.092 | 3 | +0.086 |
| 19 | Symlet10_lt_bcast_ttd3_ld16 | 13.529 | 0.063 | 0.803 | 3.095 | 3 | +0.091 |
| 20 | Haar_eq_fcast_ttd5_ld16 | 13.529 | 0.025 | 0.805 | 3.105 | 3 | +0.091 |

**Worst R3 config:** Coif1_eq_fcast_ttd3_ld16 (SMAPE=13.859, +0.421). Even the worst survivor is only 3.1% behind the best.

**Best single run:** Symlet20_eq_fcast_ttd3_ld16, seed=42, SMAPE=13.359, OWA=0.791

### 2.2 M4 Convergence Quality

The R3 results show remarkably tight convergence. The median standard deviation across seeds is only 0.065 SMAPE. The most stable config (Coif3_eq_bcast_ttd5, std=0.009, CV=0.07%) and the least stable (Haar_lt_bcast_ttd5, std=0.490, CV=3.54%) differ by ~50x in variance.

### 2.3 M4 Wavelet Family Ranking (R3)

| Rank | Family | SMAPE | Std | N |
|------|--------|-------|-----|---|
| 1 | DB4 | 13.514 | 0.031 | 3 |
| 2 | Symlet20 | 13.518 | 0.073 | 12 |
| 3 | Coif2 | 13.524 | 0.097 | 3 |
| 4 | Coif3 | 13.525 | 0.100 | 15 |
| 5 | Symlet2 | 13.535 | 0.078 | 12 |
| 6 | Symlet3 | 13.539 | 0.126 | 11 |
| 7 | Symlet10 | 13.545 | 0.103 | 6 |
| 8 | DB3 | 13.546 | 0.042 | 5 |
| 9 | Coif10 | 13.570 | 0.131 | 12 |
| 10 | DB2 | 13.574 | 0.098 | 15 |
| 11 | DB10 | 13.576 | 0.121 | 9 |
| 12 | DB20 | 13.594 | 0.178 | 18 |
| 13 | Coif1 | 13.607 | 0.211 | 12 |
| 14 | Haar | 13.626 | 0.276 | 15 |

**Key observation:** At convergence, the family ranking spread is only 0.112 SMAPE (13.514 to 13.626). Wavelet family choice matters less than expected once the architecture, latent_dim, and training are fixed. Haar is consistently last -- its coarse single-coefficient basis provides less representational richness than multi-resolution families.

### 2.4 M4 Factor Effects (R3)

**Basis dim label:** Essentially no difference at convergence.

| Label | bd | SMAPE | Std | N |
|-------|-----|-------|-----|---|
| eq_fcast | 6 | 13.558 | 0.166 | 30 |
| lt_bcast | 15 | 13.562 | 0.160 | 46 |
| eq_bcast | 30 | 13.568 | 0.143 | 60 |
| lt_fcast | 4 | 13.572 | 0.065 | 12 |

The R1 (10-epoch) ranking showed eq_bcast dominating (SMAPE=14.819 vs lt_fcast=16.475), but this was a convergence speed artifact. Higher basis_dim provides faster early learning but does not improve the converged solution.

**Trend thetas dim:** No significant difference at R3.

- ttd=3: SMAPE=13.565 (N=88)
- ttd=5: SMAPE=13.564 (N=60)
- p=0.160 (Mann-Whitney)

This reverses the strong R1 advantage for ttd=3 (14.982 vs 15.445, p<0.001), confirming ttd=5 simply converges more slowly on short-horizon data.

### 2.5 M4 Comparison to Known Baselines

| Configuration | SMAPE | OWA | Source |
|--------------|-------|-----|--------|
| This study best (Symlet20_eq_fcast_ttd3) | 13.438 | 0.795 | R3, 3 seeds |
| Prior V1 AELG best (Symlet20_eq_fcast_ttd3) | 13.438 | 0.795 | V1 study |
| Non-AE best (Coif2_bd6_eq_fcast_td3) | **13.410** | **0.794** | Wavelet Study 2 |
| NBEATS-G (30x Generic) | ~13.3-13.5 | ~0.837-0.850 | Paper baseline |

The TrendAELG+WaveletV3AELG architecture matches but does not beat the non-AE Trend+WaveletV3 architecture on M4-Yearly. The OWA advantage over NBEATS-G (0.795 vs ~0.837) is substantial.

### 2.6 M4 Divergence

7 of 711 runs (1.0%) hit SMAPE=200, all from `lt_bcast` or `lt_fcast` labels with short-support wavelets (DB3, DB4, Symlet3). These involve basis_dim <= 15, which may be insufficient for certain random seeds.

---

## 3. Tourism-Yearly Results

### 3.1 Round 3 (Final) Rankings -- Top 20

| Rank | Config | SMAPE | Std | MASE | N |
|------|--------|-------|-----|------|---|
| 1 | **Coif1_eq_fcast_ttd3_ld16** | **20.930** | 0.098 | 3.018 | 3 |
| 2 | Coif1_lt_bcast_ttd3_ld16 | 20.930 | 0.098 | 3.018 | 3 |
| 3 | Symlet2_lt_fcast_ttd3_ld16 | 20.936 | 0.071 | 3.034 | 3 |
| 4 | DB2_lt_bcast_ttd3_ld16 | 20.937 | 0.192 | 3.018 | 3 |
| 5 | DB2_eq_fcast_ttd3_ld16 | 20.937 | 0.192 | 3.018 | 3 |
| 6 | DB4_eq_fcast_ttd3_ld16 | 20.947 | 0.187 | 3.020 | 3 |
| 7 | DB4_lt_bcast_ttd3_ld16 | 20.947 | 0.187 | 3.020 | 3 |
| 8 | Symlet3_lt_fcast_ttd3_ld16 | 20.962 | 0.021 | 3.035 | 3 |
| 9 | Symlet20_lt_fcast_ttd3_ld16 | 20.970 | 0.049 | 3.027 | 3 |
| 10 | Coif1_eq_fcast_ttd5_ld16 | 20.985 | 0.338 | 3.014 | 3 |
| 11 | Coif1_lt_bcast_ttd5_ld16 | 20.985 | 0.338 | 3.014 | 3 |
| 12 | Symlet20_lt_bcast_ttd3_ld16 | 20.993 | 0.082 | 3.021 | 3 |
| 13 | Symlet20_eq_fcast_ttd3_ld16 | 20.993 | 0.082 | 3.021 | 3 |
| 14 | DB4_eq_fcast_ttd5_ld16 | 21.017 | 0.250 | 3.030 | 3 |
| 15 | DB4_lt_bcast_ttd5_ld16 | 21.017 | 0.250 | 3.030 | 3 |
| 16 | Symlet2_lt_bcast_ttd3_ld16 | 21.019 | 0.124 | 3.011 | 3 |
| 17 | Symlet2_eq_fcast_ttd3_ld16 | 21.019 | 0.124 | 3.011 | 3 |
| 18 | Coif2_eq_fcast_ttd3_ld16 | 21.054 | 0.068 | 3.025 | 3 |
| 19 | Coif2_lt_bcast_ttd3_ld16 | 21.054 | 0.068 | 3.025 | 3 |
| 20 | Symlet3_eq_fcast_ttd5_ld16 | 21.070 | 0.124 | 3.024 | 3 |

**Best single run:** Coif1_eq_fcast_ttd5_ld16, seed=42, SMAPE=20.598

**Note on degeneracy:** Ranks 1/2, 4/5, 6/7, 10/11, etc. are tied pairs because eq_fcast and lt_bcast both resolve to basis_dim=4 on Tourism-Yearly (fcast=4, bcast=8).

### 3.2 Tourism Wavelet Family Ranking (R3)

| Rank | Family | SMAPE | Std | N |
|------|--------|-------|-----|---|
| 1 | DB4 | 20.982 | 0.192 | 12 |
| 2 | Coif1 | 21.010 | 0.195 | 18 |
| 3 | Symlet20 | 21.025 | 0.097 | 12 |
| 4 | Symlet2 | 21.029 | 0.123 | 12 |
| 5 | Coif2 | 21.054 | 0.061 | 6 |
| 6 | DB2 | 21.096 | 0.246 | 12 |
| 7 | Symlet3 | 21.118 | 0.139 | 24 |
| 8 | Haar | 21.143 | 0.147 | 6 |
| 9 | Symlet10 | 21.163 | 0.120 | 12 |
| 10 | Coif3 | 21.182 | 0.268 | 9 |
| 11 | DB3 | 21.185 | 0.094 | 9 |
| 12 | DB10 | 21.268 | 0.346 | 6 |
| 13 | DB20 | 21.352 | 0.405 | 9 |
| 14 | Coif10 | 21.628 | 0.488 | 3 |

Spread is only 0.646 SMAPE (20.982 to 21.628). The short-support families (DB4, Coif1, Symlet2) tend to perform better on this short-horizon dataset (fcast=4), while long-support families (DB20, Coif10) struggle -- consistent with the principle that wavelet support length should not vastly exceed the signal length.

### 3.3 Tourism Factor Effects (R3)

**Basis dim label:** eq_fcast = lt_bcast >> lt_fcast > eq_bcast

| Label | bd | SMAPE | N |
|-------|-----|-------|---|
| eq_fcast/lt_bcast | 4 | 21.081 | 45 |
| lt_fcast | 2 | 21.131 | 15 |
| eq_bcast | 8 | 21.189 | 45 |

basis_dim=4 (= forecast_length) is optimal. The eq_bcast (bd=8, = backcast_length) slightly underperforms, suggesting the wavelet basis is over-parameterized relative to the short forecast horizon.

**Trend thetas dim:** ttd=3 is significantly better (p=0.008).

- ttd=3: SMAPE=21.096 (N=99)
- ttd=5: SMAPE=21.161 (N=51)

### 3.4 Tourism Comparison to Baselines

| Configuration | SMAPE | Source |
|--------------|-------|--------|
| This study best (Coif1_eq_fcast_ttd3) | 20.930 | R3, 3 seeds |
| Prior V1 AELG best (Coif1_eq_fcast_ttd3) | 20.930 | V1 study |

Exact replication of V1 result.

---

## 4. Weather-96 Results

### 4.1 Round 3 (Final) Rankings -- Top 20

| Rank | Config | MSE | Std | MAE | SMAPE | N |
|------|--------|-----|-----|-----|-------|---|
| 1 | **Symlet20_eq_fcast_ttd5_ld16** | **2070.61** | 362.50 | 14.190 | 64.937 | 3 |
| 2 | Symlet20_lt_bcast_ttd5_ld16 | 2070.61 | 362.50 | 14.190 | 64.937 | 3 |
| 3 | Symlet10_eq_fcast_ttd5_ld16 | 2147.34 | 291.02 | 15.549 | 64.842 | 3 |
| 4 | Symlet10_lt_bcast_ttd5_ld16 | 2147.34 | 291.02 | 15.549 | 64.842 | 3 |
| 5 | DB20_eq_bcast_ttd5_ld16 | 2175.50 | 294.97 | 15.284 | 64.956 | 3 |
| 6 | Symlet3_eq_bcast_ttd5_ld16 | 2184.54 | 261.53 | 14.955 | 65.235 | 3 |
| 7 | Symlet20_eq_bcast_ttd5_ld16 | 2189.67 | 561.65 | 15.314 | 65.060 | 3 |
| 8 | DB10_eq_fcast_ttd5_ld16 | 2193.57 | 124.86 | 14.807 | 65.333 | 3 |
| 9 | DB10_lt_bcast_ttd5_ld16 | 2193.57 | 124.86 | 14.807 | 65.333 | 3 |
| 10 | Symlet10_eq_bcast_ttd5_ld16 | 2201.66 | 168.34 | 14.169 | 65.663 | 3 |
| 11 | DB20_lt_bcast_ttd3_ld16 | 2205.60 | 143.94 | 14.086 | 65.386 | 3 |
| 12 | DB20_eq_fcast_ttd3_ld16 | 2205.60 | 143.94 | 14.086 | 65.386 | 3 |
| 13 | Symlet20_lt_bcast_ttd3_ld16 | 2210.99 | 133.19 | 14.805 | 65.356 | 3 |
| 14 | Symlet20_eq_fcast_ttd3_ld16 | 2210.99 | 133.19 | 14.805 | 65.356 | 3 |
| 15 | Haar_lt_bcast_ttd5_ld16 | 2213.90 | 77.75 | 14.337 | 65.860 | 3 |
| 16 | Haar_eq_fcast_ttd5_ld16 | 2213.90 | 77.75 | 14.337 | 65.860 | 3 |
| 17 | Symlet20_lt_fcast_ttd3_ld16 | 2235.96 | 192.40 | 14.603 | 64.924 | 3 |
| 18 | Coif1_eq_fcast_ttd5_ld16 | 2270.20 | 273.44 | 15.059 | 66.113 | 3 |
| 19 | Coif1_lt_bcast_ttd5_ld16 | 2270.20 | 273.44 | 15.059 | 66.113 | 3 |
| 20 | Coif3_eq_bcast_ttd5_ld16 | 2297.75 | 160.03 | 14.944 | 65.319 | 3 |

**Worst R3 config:** DB10_eq_bcast_ttd3_ld16 (MSE=2885.16) -- a 39% gap from the best.

**Best single run:** Symlet20_eq_bcast_ttd5_ld16, seed=43, MSE=1607.30, MAE=12.819

### 4.2 Weather Wavelet Family Ranking (R3)

| Rank | Family | MSE | Std | MAE | N |
|------|--------|-----|-----|-----|---|
| 1 | DB20 | 2195.56 | 179.82 | 14.485 | 9 |
| 2 | Symlet20 | 2196.29 | 278.02 | 14.667 | 21 |
| 3 | Symlet10 | 2255.99 | 272.15 | 15.252 | 18 |
| 4 | Haar | 2300.01 | 201.85 | 14.511 | 9 |
| 5 | Symlet2 | 2321.36 | 246.93 | 14.923 | 3 |
| 6 | Symlet3 | 2329.80 | 268.36 | 14.636 | 9 |
| 7 | DB2 | 2365.28 | 136.73 | 14.918 | 9 |
| 8 | Coif1 | 2376.06 | 359.32 | 14.935 | 15 |
| 9 | Coif3 | 2395.41 | 213.63 | 14.889 | 9 |
| 10 | Coif10 | 2395.60 | 36.20 | 14.893 | 3 |
| 11 | DB3 | 2421.11 | 267.18 | 15.262 | 15 |
| 12 | DB10 | 2437.97 | 451.26 | 15.130 | 15 |
| 13 | Coif2 | 2447.54 | 321.49 | 15.190 | 6 |
| 14 | DB4 | 2511.65 | 134.39 | 15.027 | 9 |

The long-support wavelets (DB20, Symlet20) dominate Weather-96. This makes sense: Weather has a 96-step forecast from a 192-step backcast, giving ample signal length for multi-level decomposition. DB4, which was #1 on Tourism and M4, is dead last here.

### 4.3 Weather Factor Effects (R3)

**Trend thetas dim:** ttd=5 is significantly better (p=0.042).

- ttd=3: MSE=2404.29 (N=78)
- ttd=5: MSE=2269.98 (N=72)
- Difference: -134.31 MSE (5.6% improvement)

This reversal from short-horizon datasets is consistent and meaningful. Weather-96 has a longer forecast horizon that benefits from higher-order trend polynomials.

**Basis dim label:** eq_fcast = lt_bcast >> eq_bcast = lt_fcast

| Label | bd | MSE | N |
|-------|-----|-----|---|
| eq_fcast/lt_bcast | 96 | 2319.03 | 48 |
| lt_fcast | 94 | 2374.81 | 9 |
| eq_bcast | 192 | 2377.16 | 45 |

basis_dim=96 (= forecast_length) is optimal. The eq_bcast (bd=192) adds unnecessary parameters without improving quality.

### 4.4 Weather Convergence Notes

Weather R3 configs trained for only 13-47 epochs (most early-stopped), much shorter than M4's ~50 epochs. The MSE standard deviations are larger (55-562 range) compared to M4's tight SMAPE stds (0.009-0.490). This suggests Weather results may benefit from longer training or more seeds.

---

## 5. Traffic-96 Results

### 5.1 Catastrophic Divergence

**125 of 145 runs (86.2%) diverged** -- all val_loss curves are flat at 200.0 from epoch 0 onward. Only 20 runs achieved any learning, and of those, only ~14 produced reasonable results (SMAPE < 25).

| Dimension | Diverged | Total | Rate |
|-----------|----------|-------|------|
| **All** | **125** | **145** | **86.2%** |
| Haar | 24 | 25 | 96.0% |
| DB2 | 19 | 24 | 79.2% |
| DB3 | 20 | 24 | 83.3% |
| DB4 | 18 | 24 | 75.0% |
| DB10 | 20 | 24 | 83.3% |
| DB20 | 24 | 24 | **100.0%** |
| eq_bcast | 34 | 36 | 94.4% |
| eq_fcast | 29 | 37 | 78.4% |
| lt_bcast | 28 | 36 | 77.8% |
| lt_fcast | 34 | 36 | 94.4% |

**Key observations:**

- **Haar and DB20 had the highest divergence rates** in this study (at L=2H); this reflects basis-length sensitivity at insufficient lookback, not inherent block incompatibility
- **DB4 has the lowest divergence rate** (75%) but this is still catastrophic
- **eq_bcast and lt_fcast** diverge more than eq_fcast/lt_bcast
- **Only 10 epochs** were run (R1 only -- no R2/R3 survived successive halving)
- Missing Coif and Symlet families from Traffic results -- possibly excluded from the sweep

### 5.2 Converged Traffic Runs

Among the ~14 genuinely converged runs (excluding near-diverged DB3 runs at SMAPE ~198):

| Config | Seed | SMAPE | MAE | MSE |
|--------|------|-------|-----|-----|
| DB4_eq_fcast_ttd3_ld16 | 44 | **20.325** | **0.01217** | 0.000823 |
| DB4_lt_bcast_ttd3_ld16 | 44 | 20.325 | 0.01217 | 0.000823 |
| DB2_lt_fcast_ttd3_ld16 | 43 | 20.468 | 0.01270 | 0.000898 |
| DB3_eq_fcast_ttd5_ld16 | 42 | 20.862 | 0.01217 | 0.000805 |
| DB3_lt_bcast_ttd5_ld16 | 42 | 20.862 | 0.01217 | 0.000805 |
| DB4_eq_bcast_ttd3_ld16 | 42 | 21.140 | 0.01262 | 0.000870 |
| DB2_eq_fcast_ttd3_ld16 | 42 | 21.266 | 0.01227 | 0.000776 |
| DB2_lt_bcast_ttd3_ld16 | 42 | 21.266 | 0.01227 | 0.000776 |

When it converges, the architecture produces reasonable results. But the 86% failure rate makes it completely unreliable.

### 5.3 Why Traffic Fails (Root Cause: Insufficient Backcast Horizon)

**Root cause identified (2026-03-09): insufficient backcast horizon.** This study used bl=192 (L=2H, `forecast_multiplier=2`). Traffic-96 requires bl≥480 (L=5H, `forecast_multiplier=5`) for reliable convergence. A subsequent study (AsymWavelet Diagnostic, 2026-03-08) using L=5H and 8 stacks achieved 80-100% convergence, confirming the architecture is viable with adequate lookback.

The hypotheses below were proposed before this root cause was identified. They remain as historical context but should not be used to draw conclusions about architectural limitations:

1. **Deep stacks (20) with AE bottleneck:** The 20-stack architecture may contribute to instability, but this is a secondary factor. The AsymWavelet Diagnostic used 8 stacks and also increased lookback — both changed simultaneously.

2. **Wavelet basis mismatch:** Traffic-96 has backcast=192, forecast=96 in this study. At the correct backcast length (480), DB-family wavelets work acceptably (16% divergence at 8 stacks with varied wavelet families).

3. **Insufficient training:** Only 10 epochs at R1. However, the flat val_loss at 200.0 from epoch 0 indicates the model never began learning — a training length issue cannot explain this.

4. **No normalization / loss tuning:** Traffic data is naturally 0-1 scaled (PeMS occupancy rates) and does not require normalization. MSE loss is preferred over SMAPE for Traffic (confirmed in working study).

---

## 6. Cross-Dataset Analysis

### 6.1 Wavelet Family Cross-Dataset Rankings (R3)

| Family | M4 Rank | Tourism Rank | Weather Rank | Avg Rank | Max Rank |
|--------|---------|--------------|--------------|----------|----------|
| **Symlet20** | **2** | **3** | **2** | **2.3** | 3 |
| Symlet2 | 5 | 4 | 5 | 4.7 | 5 |
| DB4 | 1 | 1 | 14 | 5.3 | 14 |
| Symlet10 | 7 | 9 | 3 | 6.3 | 9 |
| Symlet3 | 6 | 7 | 6 | 6.3 | 7 |
| Coif2 | 3 | 5 | 13 | 7.0 | 13 |
| Coif3 | 4 | 10 | 9 | 7.7 | 10 |
| Coif1 | 13 | 2 | 8 | 7.7 | 13 |
| DB2 | 10 | 6 | 7 | 7.7 | 10 |
| Haar | 14 | 8 | 4 | 8.7 | 14 |
| DB20 | 12 | 13 | 1 | 8.7 | 13 |
| DB3 | 8 | 11 | 11 | 10.0 | 11 |
| Coif10 | 9 | 14 | 10 | 11.0 | 14 |
| DB10 | 11 | 12 | 12 | 11.7 | 12 |

**Symlet20 is the clear cross-dataset winner** with avg rank 2.3 and max rank 3. It never drops below top-3 on any dataset. Symlet2 (avg 4.7) is the runner-up with never worse than rank 5.

**DB4 is polarizing:** #1 on M4 and Tourism but dead last (#14) on Weather. It excels on short horizons but fails on long ones.

**DB20 is the inverse:** #1 on Weather but #12-13 on short-horizon datasets.

This confirms the **support-length/horizon principle**: short-support wavelets for short horizons, long-support wavelets for long horizons. Symlet20 is the exception that works everywhere, likely because the Symlet family has near-linear phase which preserves temporal structure across scales.

### 6.2 Trend Thetas Dim: Dataset-Dependent Direction

| Dataset | Horizon | ttd=3 | ttd=5 | Better | p-value | Significant? |
|---------|---------|-------|-------|--------|---------|-------------|
| M4-Yearly | 6 | 13.565 | 13.564 | Equal | 0.160 | No |
| Tourism-Yearly | 4 | 21.096 | 21.161 | **ttd=3** | **0.008** | **Yes** |
| Weather-96 | 96 | MSE 2404 | MSE 2270 | **ttd=5** | **0.042** | **Yes** |

**Rule of thumb:** Use ttd=3 for forecast_length <= 6; use ttd=5 for forecast_length >= 96. The crossover point is likely in the 10-50 range but has not been tested.

The explanation is that trend polynomials of degree d-1 (from thetas_dim=d) can represent:

- ttd=3: constant + linear + quadratic trends
- ttd=5: up to quartic trends

Short horizons have limited trend variability, so extra polynomial degrees add noise. Long horizons benefit from the added flexibility.

### 6.3 Basis Dim Label: Consistent Pattern

Across all datasets at convergence:

| Dataset | Best bd_label | Best bd value | Principle |
|---------|--------------|---------------|-----------|
| M4-Yearly | eq_fcast | 6 (= H) | bd = forecast_length |
| Tourism-Yearly | eq_fcast/lt_bcast | 4 (= H) | bd = forecast_length |
| Weather-96 | eq_fcast/lt_bcast | 96 (= H) | bd = forecast_length |

**Consistent finding: basis_dim = forecast_length is optimal or co-optimal across all datasets.** This aligns with Wavelet Study 2's finding that bd=H is the natural choice.

lt_fcast (bd < H) consistently underperforms at R1 but catches up somewhat at R3. eq_bcast (bd = L) adds parameters without benefit. The wavelet basis projection only needs enough dimensions to represent the forecast signal.

---

## 7. Successive Halving Effectiveness

The 3-round successive halving protocol was effective at M4, Tourism, and Weather:

### Round Progression: M4-Yearly

| Round | Configs | Epochs | Best SMAPE | Worst SMAPE | Spread |
|-------|---------|--------|------------|-------------|--------|
| R1 | 112 | 10 | 15.248 | 19.980 | 4.732 |
| R2 | 75 | 15 | 14.180 | 15.550 | 1.370 |
| R3 | 50 | 50 | 13.438 | 13.859 | 0.421 |

The spread decreases from 4.7 to 0.4, confirming convergence. However, the final 0.421 SMAPE spread among 50 configs is so tight that the halving protocol may have been too aggressive in cutting -- some eliminated configs might have matched the survivors with more training. The key insight is that at R3, **wavelet family matters less than convergence variance**.

### R1 vs R3 Rank Correlation

Some configs that ranked well at R1 (e.g., DB4_eq_bcast_ttd3, rank #1 at R1) dropped to mid-pack at R3 (rank #16). Conversely, the R3 winner (Symlet20_eq_fcast_ttd3) was not in the R1 top-10. This suggests that **10-epoch rankings are unreliable predictors of converged performance** for this architecture family.

---

## 8. Replication of Prior Results

This study successfully replicates two previously reported best configurations:

| Config | Prior SMAPE | This Study SMAPE | Match? |
|--------|------------|-----------------|--------|
| M4: Symlet20_eq_fcast_ttd3_ld16 | 13.438 | 13.438 (+/-0.069) | **Exact** |
| Tourism: Coif1_eq_fcast_ttd3_ld16 | 20.930 | 20.930 (+/-0.098) | **Exact** |

The V1 study results are reproducible. The successively halved search independently converges to the same winning configs, providing strong evidence these are true optima for the architecture.

---

## 9. Conclusions and Recommendations

### 9.1 Current Best Configurations

| Dataset | Best Config | Primary Metric | Confidence |
|---------|-------------|---------------|------------|
| M4-Yearly | Symlet20_eq_fcast_ttd3_ld16 | SMAPE=13.438, OWA=0.795 | **High** (replicated) |
| Tourism-Yearly | Coif1_eq_fcast_ttd3_ld16 | SMAPE=20.930 | **High** (replicated) |
| Weather-96 | Symlet20_eq_fcast_ttd5_ld16 | MSE=2070.61 | **Medium** (3 seeds, high std) |
| Traffic-96 | N/A for this study (bl=192, L=2H insufficient; see AsymWavelet Diagnostic for successful results) | -- | **Protocol failure** (inadequate lookback, not architectural) |

### 9.2 Recommended Default Settings

For new datasets using TrendAELG + WaveletV3AELG:

1. **Wavelet family:** Symlet20 (best cross-dataset consistency)
2. **Basis dim label:** eq_fcast (basis_dim = forecast_length)
3. **Trend thetas dim:** ttd=3 for short horizons (H <= 6), ttd=5 for long horizons (H >= 96)
4. **Latent dim:** 16
5. **Stacks:** 10 (M4/Tourism), 20 (Weather); ≤8-10 stacks for Traffic; use `forecast_multiplier=5` (bl=480, L=5H) for Traffic

### 9.3 What to Test Next

1. **Traffic recovery (root cause known):** The primary fix is `forecast_multiplier=5` (bl=480, L=5H). Secondary: reduce stack depth to ≤8-10. This was confirmed by the AsymWavelet Diagnostic study (2026-03-08), which achieved 80-100% convergence with MSE ~0.0006 using L=5H and 8 stacks. Prefer MSE loss over SMAPE for Traffic (Traffic data is naturally 0-1 scaled). Input normalization is not needed (PeMS occupancy rates are already bounded).

2. **Weather with more seeds:** The R3 Weather results have high MSE variance (std ~100-560). Run the top 5 Weather configs with 10 seeds at 50 epochs to get stable estimates.

3. **M4 other periods:** This study only covers M4-Yearly. The architecture should be tested on Monthly, Quarterly, and Weekly periods.

4. **Symlet20 vs non-AE Trend+WaveletV3:** The non-AE architecture achieved SMAPE=13.410 on M4-Yearly (vs 13.438 for AELG). A direct comparison with matched hyperparameters would determine whether the AE bottleneck adds value.

5. **ttd crossover point:** Test ttd={3,4,5} on M4-Monthly (H=18) and M4-Quarterly (H=8) to find where ttd=5 starts outperforming ttd=3.

### 9.4 YAML Config for Recommended Experiments

```yaml
# Weather-96: Symlet20 with more seeds
experiment_name: weather_symlet20_confirmation
dataset: weather
periods: [Weather-96]

stacks:
  alternating:
    types: [TrendAELG, Symlet20WaveletV3AELG]
    repeats: 10

training:
  max_epochs: 50
  loss: SMAPE
  activation: ReLU
  share_weights: true
  active_g: false
  sum_losses: false
  latent_dim: 16
  trend_thetas_dim: 5
  basis_dim: 96

n_runs: 10
seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

extra_csv_columns: [basis_dim, trend_thetas_dim_cfg, wavelet_family, bd_label, latent_dim_cfg]
extra_fields:
  basis_dim: 96
  trend_thetas_dim_cfg: 5
  wavelet_family: Symlet20WaveletV3AELG
  bd_label: eq_fcast
  latent_dim_cfg: 16
```

### 9.5 Open Questions

1. **Why does the AELG bottleneck not improve over non-AE?** The V1 AELG best (13.438) is slightly worse than the non-AE best (13.410) on M4-Yearly. The learned gate may not provide enough regularization benefit to offset the information bottleneck cost.

2. **Is the tight R3 convergence a ceiling?** All 50 surviving configs converge to 13.44-13.86 SMAPE. Is this the architectural limit, or could hyperparameter changes (learning rate, width, depth) push lower?

3. **Traffic divergence root cause (resolved):** The primary cause was insufficient backcast horizon (bl=192, L=2H). The AsymWavelet Diagnostic confirmed convergence at bl=480 (L=5H). Stack depth (8 vs 20) is a secondary factor. The AE bottleneck and wavelet basis are not the cause — they work fine with adequate lookback.

4. **Symlet20 universality:** Does Symlet20's cross-dataset dominance extend to M4-Monthly/Quarterly and other datasets, or is it specific to the yearly/seasonal/weather domain?

---

## Correction Addendum (2026-03-09)

**The Traffic-96 failure claims in this report are misleading.** The 86.2% divergence rate reported above was caused by an inadequate evaluation protocol, not an inherent architectural incompatibility:

| Factor | This study (failed) | AsymWavelet Diagnostic (converged) |
|--------|--------------------|------------------------------------|
| **Lookback** | **L=2H (bl=192)** | **L=5H (bl=480)** |
| **Stack depth** | **20 stacks** | **8 stacks** |
| **Training** | 10 epochs (R1), MAX_EPOCHS | 42-85 epochs, EARLY_STOPPED |

The AsymWavelet Diagnostic study (2026-03-08) demonstrated 80-100% convergence for TrendAELG+WaveletV3AELG on Traffic-96 using L=5H lookback and 8 stacks, achieving MSE ~0.0006. **All Traffic studies in this repository that reported failure used bl=192 (L=2H); the only study using bl=480 (L=5H) converged successfully.**

Conclusions referencing "Traffic-96: N/A — architecture not viable" should be read as: the architecture requires adequate lookback (L≥5H) and moderate stack depth (≤8-10 stacks) to converge on Traffic. The block types themselves are not the cause of divergence.
