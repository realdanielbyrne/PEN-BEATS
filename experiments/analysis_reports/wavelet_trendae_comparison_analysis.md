# Wavelet + TrendAE Combination Study — Results Analysis

## Abstract

This study evaluates **Wavelet+TrendAE** hybrid stacks on the M4-Yearly benchmark, exploring 30 configurations across 89 total runs (315.7 min total training time). The best configuration, `Coif2_bd30_eq_bcast_ttd3_ld5`, achieves a median OWA of **0.7954** (Δ = -0.0061 vs AE+Trend baseline at 0.8015) with **4,307,240** parameters (83% fewer than NBEATS-G). All runs converged successfully.

**Key Takeaways:**

1. **Best wavelet family:** `Symlet10`
2. **vs AE+Trend baseline:** beats (Δ = -0.0061)
3. **Parameter efficiency:** 83% reduction vs NBEATS-G


## 1. Overview

- **CSV:** `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/wavelet_trendae_comparison_results.csv`
- **Rows:** 89 (30 unique configs)
- **Total training time:** 315.7 min
- **Wavelet families:** ['Coif1', 'Coif2', 'Coif3', 'DB10', 'DB20', 'Haar', 'Symlet10', 'Symlet3']
- **OWA range:** 0.7927 – 0.9220

| Baseline   |    OWA |   sMAPE |   Params |
|:-----------|-------:|--------:|---------:|
| NBEATS-I+G | 0.8057 |   13.53 | 35900000 |
| AE+Trend   | 0.8015 |   13.53 |  5200000 |
| GenericAE  | 0.8063 |   13.57 |  4800000 |
| NBEATS-G   | 0.8198 |   13.70 | 24700000 |

## 2. Overall Leaderboard

30 configs × 3 runs

|   # | Config                         | Wavelet   |    OWA |      ± |   sMAPE |   MASE |   Params |
|----:|:-------------------------------|:----------|-------:|-------:|--------:|-------:|---------:|
|   1 | Coif2_bd30_eq_bcast_ttd3_ld5   | Coif2     | 0.7954 | 0.0080 |   13.44 | 3.0543 |  4307240 |
|   2 | Symlet3_bd4_lt_fcast_ttd3_ld8  | Symlet3   | 0.7968 | 0.0011 |   13.46 | 3.0607 |  4239415 |
|   3 | DB20_bd15_lt_bcast_ttd3_ld2    | DB20      | 0.7983 | 0.0059 |   13.46 | 3.0715 |  4264985 |
|   4 | Coif1_bd15_lt_bcast_ttd3_ld5   | Coif1     | 0.7991 | 0.0646 |   13.47 | 3.0745 |  4268840 |
|   5 | Haar_bd4_lt_fcast_ttd3_ld8     | Haar      | 0.8004 | 0.0717 |   13.52 | 3.0740 |  4239415 |
|   6 | Symlet10_bd6_eq_fcast_ttd3_ld8 | Symlet10  | 0.8005 | 0.0074 |   13.48 | 3.0819 |  4249655 |
|   7 | Coif2_bd30_eq_bcast_ttd3_ld8   | Coif2     | 0.8024 | 0.1293 |   13.49 | 3.0960 |  4311095 |
|   8 | Coif2_bd4_lt_fcast_ttd3_ld2    | Coif2     | 0.8032 | 0.0022 |   13.52 | 3.0937 |  4231705 |
|   9 | Symlet3_bd4_lt_fcast_ttd3_ld2  | Symlet3   | 0.8035 | 0.0029 |   13.55 | 3.0897 |  4231705 |
|  10 | Haar_bd4_lt_fcast_ttd3_ld5     | Haar      | 0.8037 | 0.0111 |   13.57 | 3.0828 |  4235560 |
|  11 | Coif3_bd4_lt_fcast_ttd3_ld2    | Coif3     | 0.8042 | 0.0401 |   13.60 | 3.0841 |  4231705 |
|  12 | Coif3_bd4_lt_fcast_ttd3_ld8    | Coif3     | 0.8042 | 0.0622 |   13.57 | 3.0923 |  4239415 |
|  13 | Coif3_bd4_lt_fcast_ttd3_ld5    | Coif3     | 0.8045 | 0.0124 |   13.60 | 3.0881 |  4235560 |
|  14 | DB10_bd15_lt_bcast_ttd3_ld2    | DB10      | 0.8049 | 0.0250 |   13.60 | 3.0910 |  4264985 |
|  15 | Coif3_bd6_eq_fcast_ttd3_ld5    | Coif3     | 0.8049 | 0.0200 |   13.57 | 3.0980 |  4245800 |
|  16 | Coif3_bd6_eq_fcast_ttd3_ld2    | Coif3     | 0.8049 | 0.0182 |   13.54 | 3.1045 |  4241945 |
|  17 | Symlet10_bd6_eq_fcast_ttd3_ld5 | Symlet10  | 0.8063 | 0.0349 |   13.54 | 3.1167 |  4245800 |
|  18 | Haar_bd4_lt_fcast_ttd3_ld2     | Haar      | 0.8076 | 0.0124 |   13.61 | 3.1105 |  4231705 |
|  19 | Coif2_bd4_lt_fcast_ttd3_ld8    | Coif2     | 0.8081 | 0.0059 |   13.59 | 3.1170 |  4239415 |
|  20 | Coif2_bd4_lt_fcast_ttd3_ld5    | Coif2     | 0.8081 | 0.0318 |   13.62 | 3.1109 |  4235560 |
|  21 | DB10_bd15_lt_bcast_ttd3_ld5    | DB10      | 0.8104 | 0.0054 |   13.62 | 3.1283 |  4268840 |
|  22 | DB20_bd15_lt_bcast_ttd3_ld5    | DB20      | 0.8115 | 0.0092 |   13.65 | 3.1249 |  4268840 |
|  23 | Coif1_bd15_lt_bcast_ttd3_ld2   | Coif1     | 0.8134 | 0.0649 |   13.73 | 3.1264 |  4264985 |
|  24 | Symlet3_bd4_lt_fcast_ttd3_ld5  | Symlet3   | 0.8163 | 0.0026 |   13.71 | 3.1541 |  4235560 |
|  25 | DB10_bd15_lt_bcast_ttd3_ld8    | DB10      | 0.8212 | 0.0315 |   13.75 | 3.1830 |  4272695 |
|  26 | Coif2_bd30_eq_bcast_ttd3_ld2   | Coif2     | 0.8241 | 0.0588 |   13.73 | 3.2108 |  4303385 |
|  27 | Coif1_bd15_lt_bcast_ttd3_ld8   | Coif1     | 0.8284 | 0.1069 |   13.81 | 3.2261 |  4272695 |
|  28 | Coif3_bd6_eq_fcast_ttd3_ld8    | Coif3     | 0.8351 | 0.0464 |   13.96 | 3.2433 |  4249655 |
|  29 | Symlet10_bd6_eq_fcast_ttd3_ld2 | Symlet10  | 0.8366 | 0.0426 |   13.96 | 3.2435 |  4241945 |
|  30 | DB20_bd15_lt_bcast_ttd3_ld8    | DB20      | 0.8439 | 0.0526 |   14.02 | 3.2965 |  4272695 |

The best configuration achieves a median OWA of **0.7954** while the worst reaches **0.8439**, a spread of 0.0485. The top config uses the **Coif2** wavelet family with **4,307,240** parameters.

## 3. Hyperparameter Marginals


### Wavelet Family

| Value    |   Med OWA |   Mean |    Std |   N |   Med Params |
|:---------|----------:|-------:|-------:|----:|-------------:|
| Symlet10 |    0.8017 | 0.8120 | 0.0192 |   9 |    4,245,800 |
| Symlet3  |    0.8026 | 0.8041 | 0.0082 |   8 |    4,235,560 |
| Coif2    |    0.8034 | 0.8149 | 0.0313 |  18 |    4,271,400 |
| Haar     |    0.8037 | 0.8104 | 0.0228 |   9 |    4,235,560 |
| Coif3    |    0.8049 | 0.8139 | 0.0193 |  18 |    4,240,680 |
| DB20     |    0.8115 | 0.8188 | 0.0278 |   9 |    4,268,840 |
| DB10     |    0.8129 | 0.8147 | 0.0149 |   9 |    4,268,840 |
| Coif1    |    0.8139 | 0.8352 | 0.0417 |   9 |    4,268,840 |

`Symlet10` is the strongest setting (median OWA 0.8017) while `Coif1` is the weakest (0.8139).

### Latent Dim (TrendAE)

|   Value |   Med OWA |   Mean |    Std |   N |   Med Params |
|--------:|----------:|-------:|-------:|----:|-------------:|
|     2.0 |    0.8044 | 0.8128 | 0.0197 |  30 |    4,241,945 |
|     5.0 |    0.8049 | 0.8082 | 0.0139 |  29 |    4,245,800 |
|     8.0 |    0.8108 | 0.8249 | 0.0354 |  30 |    4,249,655 |

`2` is the strongest setting (median OWA 0.8044) while `8` is the weakest (0.8108).

### Basis Dim (WaveletV3)

|   Value |   Med OWA |   Mean |    Std |   N |   Med Params |
|--------:|----------:|-------:|-------:|----:|-------------:|
|    30.0 |    0.8031 | 0.8223 | 0.0433 |   9 |    4,307,240 |
|     4.0 |    0.8042 | 0.8089 | 0.0165 |  35 |    4,235,560 |
|     6.0 |    0.8056 | 0.8134 | 0.0184 |  18 |    4,245,800 |
|    15.0 |    0.8129 | 0.8229 | 0.0304 |  27 |    4,268,840 |

`30` is the strongest setting (median OWA 0.8031) while `15` is the weakest (0.8129).

### Basis Offset (lt_bcast vs eq_fcast)

| Value    |   Med OWA |   Mean |    Std |   N |   Med Params |
|:---------|----------:|-------:|-------:|----:|-------------:|
| eq_bcast |    0.8031 | 0.8223 | 0.0433 |   9 |    4,307,240 |
| lt_fcast |    0.8042 | 0.8089 | 0.0165 |  35 |    4,235,560 |
| eq_fcast |    0.8056 | 0.8134 | 0.0184 |  18 |    4,245,800 |
| lt_bcast |    0.8129 | 0.8229 | 0.0304 |  27 |    4,268,840 |

`eq_bcast` is the strongest setting (median OWA 0.8031) while `lt_bcast` is the weakest (0.8129).

## 3b. Selecting the Optimal Latent Dimension (TrendAE)

In this hybrid stack, the **TrendAE** component uses an AERootBlock backbone whose bottleneck width is controlled by `latent_dim`. The encoder path is `backcast_length → units/2 → latent_dim` and the decoder expands back via `latent_dim → units/2 → units`, after which the trend head applies a Vandermonde polynomial basis expansion. A smaller latent_dim increases regularisation while a larger value preserves more signal for the trend polynomial to fit.

With backcast_length = 30, the tested latent dimensions are: **2, 5, 8**.

- **latent_dim = 2:** median OWA = 0.8044, std = 0.0197, params ≈ 4,241,945 ← best
- **latent_dim = 5:** median OWA = 0.8049, std = 0.0139, params ≈ 4,245,800
- **latent_dim = 8:** median OWA = 0.8108, std = 0.0354, params ≈ 4,249,655 ← worst

The optimal setting is **latent_dim = 2** (median OWA 0.8044), outperforming the worst (latent_dim = 8) by Δ = 0.0063. 
The tightest bottleneck wins. The TrendAE head already imposes strong inductive bias via its polynomial basis, so the backbone needs only a minimal latent representation. Combined with the WaveletV3 stack handling oscillatory components, the TrendAE can afford aggressive compression for the slowly-varying residual.

**Practical recommendation:** Use `latent_dim = 2` for Wavelet+TrendAE stacks on M4-Yearly. Since the TrendAE only needs to model the slowly-varying residual after the wavelet stack, the latent dimension can be kept small. For longer forecast horizons, experiment with slightly larger values.


## 4. Stability Analysis (OWA spread across seeds)

- **Mean spread (max−min):** 0.0329
- **Max spread (max−min):** 0.1293 (`Coif2_bd30_eq_bcast_ttd3_ld8`)
- **Mean std:** 0.0177

### Most Stable Configs (smallest max−min spread)

| Config                        |   Median OWA |   Range |    Std |
|:------------------------------|-------------:|--------:|-------:|
| Symlet3_bd4_lt_fcast_ttd3_ld8 |       0.7968 |  0.0011 | 0.0006 |
| Coif2_bd4_lt_fcast_ttd3_ld2   |       0.8032 |  0.0022 | 0.0012 |
| Symlet3_bd4_lt_fcast_ttd3_ld5 |       0.8163 |  0.0026 | 0.0019 |
| Symlet3_bd4_lt_fcast_ttd3_ld2 |       0.8035 |  0.0029 | 0.0015 |
| DB10_bd15_lt_bcast_ttd3_ld5   |       0.8104 |  0.0054 | 0.0027 |

## 5. Parameter Efficiency

| Config                         | Wavelet   |   Params |   Reduction |   Med OWA | Pareto   |
|:-------------------------------|:----------|---------:|------------:|----------:|:---------|
| Coif2_bd4_lt_fcast_ttd3_ld2    | Coif2     |  4231705 |        82.9 |    0.8032 | ◀        |
| Coif3_bd4_lt_fcast_ttd3_ld2    | Coif3     |  4231705 |        82.9 |    0.8042 |          |
| Symlet3_bd4_lt_fcast_ttd3_ld2  | Symlet3   |  4231705 |        82.9 |    0.8035 |          |
| Haar_bd4_lt_fcast_ttd3_ld2     | Haar      |  4231705 |        82.9 |    0.8076 |          |
| Haar_bd4_lt_fcast_ttd3_ld5     | Haar      |  4235560 |        82.9 |    0.8037 |          |
| Symlet3_bd4_lt_fcast_ttd3_ld5  | Symlet3   |  4235560 |        82.9 |    0.8163 |          |
| Coif3_bd4_lt_fcast_ttd3_ld5    | Coif3     |  4235560 |        82.9 |    0.8045 |          |
| Coif2_bd4_lt_fcast_ttd3_ld5    | Coif2     |  4235560 |        82.9 |    0.8081 |          |
| Coif3_bd4_lt_fcast_ttd3_ld8    | Coif3     |  4239415 |        82.8 |    0.8042 |          |
| Coif2_bd4_lt_fcast_ttd3_ld8    | Coif2     |  4239415 |        82.8 |    0.8081 |          |
| Symlet3_bd4_lt_fcast_ttd3_ld8  | Symlet3   |  4239415 |        82.8 |    0.7968 | ◀        |
| Haar_bd4_lt_fcast_ttd3_ld8     | Haar      |  4239415 |        82.8 |    0.8004 |          |
| Coif3_bd6_eq_fcast_ttd3_ld2    | Coif3     |  4241945 |        82.8 |    0.8049 |          |
| Symlet10_bd6_eq_fcast_ttd3_ld2 | Symlet10  |  4241945 |        82.8 |    0.8366 |          |
| Symlet10_bd6_eq_fcast_ttd3_ld5 | Symlet10  |  4245800 |        82.8 |    0.8063 |          |
| Coif3_bd6_eq_fcast_ttd3_ld5    | Coif3     |  4245800 |        82.8 |    0.8049 |          |
| Symlet10_bd6_eq_fcast_ttd3_ld8 | Symlet10  |  4249655 |        82.8 |    0.8005 |          |
| Coif3_bd6_eq_fcast_ttd3_ld8    | Coif3     |  4249655 |        82.8 |    0.8351 |          |
| Coif1_bd15_lt_bcast_ttd3_ld2   | Coif1     |  4264985 |        82.7 |    0.8134 |          |
| DB20_bd15_lt_bcast_ttd3_ld2    | DB20      |  4264985 |        82.7 |    0.7983 |          |
| DB10_bd15_lt_bcast_ttd3_ld2    | DB10      |  4264985 |        82.7 |    0.8049 |          |
| Coif1_bd15_lt_bcast_ttd3_ld5   | Coif1     |  4268840 |        82.7 |    0.7991 |          |
| DB20_bd15_lt_bcast_ttd3_ld5    | DB20      |  4268840 |        82.7 |    0.8115 |          |
| DB10_bd15_lt_bcast_ttd3_ld5    | DB10      |  4268840 |        82.7 |    0.8104 |          |
| DB20_bd15_lt_bcast_ttd3_ld8    | DB20      |  4272695 |        82.7 |    0.8439 |          |
| DB10_bd15_lt_bcast_ttd3_ld8    | DB10      |  4272695 |        82.7 |    0.8212 |          |
| Coif1_bd15_lt_bcast_ttd3_ld8   | Coif1     |  4272695 |        82.7 |    0.8284 |          |
| Coif2_bd30_eq_bcast_ttd3_ld2   | Coif2     |  4303385 |        82.6 |    0.8241 |          |
| Coif2_bd30_eq_bcast_ttd3_ld5   | Coif2     |  4307240 |        82.6 |    0.7954 | ◀        |
| Coif2_bd30_eq_bcast_ttd3_ld8   | Coif2     |  4311095 |        82.5 |    0.8024 |          |

**3 Pareto-optimal** configurations identified where no other config achieves both lower OWA and fewer parameters.

## 6. Baseline Comparison

| Source     |   Med OWA |   Params |
|:-----------|----------:|---------:|
| AE+Trend   |    0.8015 |  5200000 |
| NBEATS-I+G |    0.8057 | 35900000 |
| GenericAE  |    0.8063 |  4800000 |
| NBEATS-G   |    0.8198 | 24700000 |

### Top-5 Wavelet+TrendAE Configs (this study)

| Config                        |   Med OWA |   Δ vs AE+Trend |
|:------------------------------|----------:|----------------:|
| Coif2_bd30_eq_bcast_ttd3_ld5  |    0.7954 |         -0.0061 |
| Symlet3_bd4_lt_fcast_ttd3_ld8 |    0.7968 |         -0.0047 |
| DB20_bd15_lt_bcast_ttd3_ld2   |    0.7983 |         -0.0032 |
| Coif1_bd15_lt_bcast_ttd3_ld5  |    0.7991 |         -0.0024 |
| Haar_bd4_lt_fcast_ttd3_ld8    |    0.8004 |         -0.0011 |

The best Wavelet+TrendAE config (`Coif2_bd30_eq_bcast_ttd3_ld5`) improves upon the AE+Trend baseline (Δ = -0.0061).

## 7. Training Stability (divergence / stopping)

| Metric        |   Count | %     |
|:--------------|--------:|:------|
| Total runs    |      89 |       |
| Diverged      |       0 | 0.0%  |
| Early stopped |      44 | 49.4% |
| Hit max epoch |      45 | 50.6% |

All 89 runs converged without divergence — the architecture is stable across seeds.
