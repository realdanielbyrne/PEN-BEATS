# WaveletV2 vs WaveletV3 Benchmark Comparison

## Abstract

This report compares **WaveletV2** (8 configs, 86 runs) against **WaveletV3** (10 configs, 51 runs) on the M4 benchmark. On M4-Yearly, the best V2 config is `Trend+HaarWaveletV2` (median OWA **0.8012**) and the best V3 config is `Trend+Coif2WaveletV3` (median OWA **0.7969**). **V3** achieves the lower OWA overall.

**Key Takeaways:**

1. **Overall winner:** V3 (Δ = 0.0043)
2. **vs NBEATS-I+G (0.8057):** Best wavelet beats the baseline
3. **V2 periods tested:** Monthly, Quarterly, Yearly


## 1. Overview

- **V2 CSV:** `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/wavelet_v2_benchmark_results.csv` (86 rows, 8 configs)
- **V3 CSV:** `/home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/wavelet_v3_benchmark_results.csv` (51 rows, 10 configs)

| Baseline   |    OWA |   Params |
|:-----------|-------:|---------:|
| NBEATS-I+G | 0.8057 | 35900000 |
| NBEATS-I   | 0.8132 | 12900000 |
| NBEATS-G   | 0.8198 | 24700000 |

## 2. V2 Leaderboard — Yearly

V2 — Yearly: 8 configs × 5 runs

|   # | Config               |    OWA |      ± |   sMAPE |   Params |
|----:|:---------------------|-------:|-------:|--------:|---------:|
|   1 | Trend+HaarWaveletV2  | 0.8012 | 0.0263 |   13.49 | 16225995 |
|   2 | Trend+DB3WaveletV2   | 0.8047 | 0.0235 |   13.59 | 16225995 |
|   3 | Generic+DB3WaveletV2 | 0.8162 | 1.7574 |   13.70 | 25461120 |
|   4 | Coif2WaveletV2       | 0.8638 | 0.0428 |   14.30 | 26254080 |
|   5 | HaarWaveletV2        | 0.8643 | 0.0159 |   14.36 | 26254080 |
|   6 | DB3WaveletV2         | 0.8765 | 0.0329 |   14.52 | 26254080 |
|   7 | Symlet3WaveletV2     | 0.8765 | 0.0329 |   14.52 | 26254080 |
|   8 | DB3AltWaveletV2      | 0.9573 | 0.0298 |   15.75 | 26115840 |

The best V2 config on Yearly is `Trend+HaarWaveletV2` with median OWA **0.8012**.

## 2b. V2 Leaderboard — Quarterly

V2 — Quarterly: 8 configs × 5 runs

|   # | Config               |    OWA |      ± |   sMAPE |   Params |
|----:|:---------------------|-------:|-------:|--------:|---------:|
|   1 | Trend+DB3WaveletV2   | 0.8984 | 0.0177 |   10.29 | 16364235 |
|   2 | Trend+HaarWaveletV2  | 0.9010 | 0.0284 |   10.26 | 16364235 |
|   3 | Generic+DB3WaveletV2 | 0.9106 | 0.0184 |   10.29 | 25729920 |
|   4 | HaarWaveletV2        | 0.9112 | 0.0296 |   10.34 | 26453760 |
|   5 | Coif2WaveletV2       | 0.9137 | 0.0068 |   10.36 | 26453760 |
|   6 | DB3WaveletV2         | 0.9173 | 0.0139 |   10.35 | 26453760 |
|   7 | Symlet3WaveletV2     | 0.9173 | 0.0139 |   10.35 | 26453760 |
|   8 | DB3AltWaveletV2      | 0.9848 | 0.0132 |   11.06 | 26269440 |

The best V2 config on Quarterly is `Trend+DB3WaveletV2` with median OWA **0.8984**.

## 2c. V2 Leaderboard — Monthly

V2 — Monthly: 2 configs × 1 runs

|   # | Config        |    OWA |      ± |   sMAPE |   Params |
|----:|:--------------|-------:|-------:|--------:|---------:|
|   1 | DB3WaveletV2  | 0.9189 | 0.0000 |   13.32 | 27452160 |
|   2 | HaarWaveletV2 | 0.9305 | 0.0202 |   13.60 | 27452160 |

The best V2 config on Monthly is `DB3WaveletV2` with median OWA **0.9189**.

## 3. V3 Leaderboard — Yearly

V3 — Yearly: 10 configs × 5 runs

|   # | Config               |    OWA |      ± |   sMAPE |   Params |
|----:|:---------------------|-------:|-------:|--------:|---------:|
|   1 | Trend+Coif2WaveletV3 | 0.7969 | 0.0095 |   13.43 | 15433035 |
|   2 | Trend+DB3WaveletV3   | 0.7986 | 0.0220 |   13.44 | 15433035 |
|   3 | Generic+DB3WaveletV3 | 0.8001 | 0.0213 |   13.49 | 24668160 |
|   4 | Trend+HaarWaveletV3  | 0.8030 | 0.0192 |   13.51 | 15433035 |
|   5 | Coif2WaveletV3       | 0.8035 | 0.0099 |   13.52 | 24668160 |
|   6 | DB3WaveletV3         | 0.8045 | 1.8435 |   13.53 | 24668160 |
|   7 | DB4WaveletV3         | 0.8057 | 0.0114 |   13.54 | 24668160 |
|   8 | DB2WaveletV3         | 0.8091 | 0.0302 |   13.57 | 24668160 |
|   9 | HaarWaveletV3        | 0.8127 | 1.8579 |   13.60 | 24668160 |
|  10 | Symlet3WaveletV3     | 0.8132 | 1.7204 |   13.65 | 24668160 |

The best V3 config on Yearly is `Trend+Coif2WaveletV3` with median OWA **0.7969**.

## 4. V2 vs V3 Head-to-Head (Yearly, matching families)

| Family   |   V2 OWA |   V3 OWA |   Δ(V3-V2) | Winner   |
|:---------|---------:|---------:|-----------:|:---------|
| T+DB3    |   0.8047 |   0.7986 |    -0.0061 | V3 ✓     |
| G+DB3    |   0.8162 |   0.8001 |    -0.0161 | V3 ✓     |
| T+Haar   |   0.8012 |   0.8030 |    +0.0018 | V2 ✓     |
| Coif2    |   0.8638 |   0.8035 |    -0.0603 | V3 ✓     |
| DB3      |   0.8765 |   0.8045 |    -0.0720 | V3 ✓     |
| Haar     |   0.8643 |   0.8127 |    -0.0516 | V3 ✓     |
| Symlet3  |   0.8765 |   0.8132 |    -0.0633 | V3 ✓     |

**V3 wins 6/7** head-to-head matchups; V2 wins 1/7.

## 5. Family Rankings

### V2 Yearly (or all available periods)

| Config               |   Med OWA |    Std |   N |
|:---------------------|----------:|-------:|----:|
| Trend+HaarWaveletV2  |    0.8012 | 0.0105 |   5 |
| Trend+DB3WaveletV2   |    0.8047 | 0.0103 |   5 |
| Generic+DB3WaveletV2 |    0.8162 | 0.7843 |   5 |
| Coif2WaveletV2       |    0.8638 | 0.0204 |   5 |
| HaarWaveletV2        |    0.8643 | 0.0067 |   5 |
| DB3WaveletV2         |    0.8765 | 0.0131 |   5 |
| Symlet3WaveletV2     |    0.8765 | 0.0131 |   5 |
| DB3AltWaveletV2      |    0.9573 | 0.0116 |   5 |

### V3 Yearly (or all available periods)

| Config               |   Med OWA |    Std |   N |
|:---------------------|----------:|-------:|----:|
| Trend+Coif2WaveletV3 |    0.7969 | 0.0039 |   5 |
| Trend+DB3WaveletV3   |    0.7986 | 0.0083 |   5 |
| Generic+DB3WaveletV3 |    0.8001 | 0.0087 |   5 |
| Trend+HaarWaveletV3  |    0.8030 | 0.0076 |   5 |
| Coif2WaveletV3       |    0.8035 | 0.0036 |   5 |
| DB3WaveletV3         |    0.8045 | 0.7484 |   6 |
| DB4WaveletV3         |    0.8057 | 0.0047 |   5 |
| DB2WaveletV3         |    0.8091 | 0.0122 |   5 |
| HaarWaveletV3        |    0.8127 | 0.8288 |   5 |
| Symlet3WaveletV3     |    0.8132 | 0.7655 |   5 |

## 6. Stability Analysis (OWA spread across seeds)

### V2

- **Mean spread (max−min):** 0.2914
- **Max spread (max−min):** 1.7574 (`Generic+DB3WaveletV2`)
- **Mean std:** 0.0958

### V3

- **Mean spread (max−min):** 0.5545
- **Max spread (max−min):** 1.8579 (`HaarWaveletV3`)
- **Mean std:** 0.2392


## 7. Parameter Efficiency (Pareto frontiers)


### V2

| Config               |   Params |   Reduction |   Med OWA | Pareto   |
|:---------------------|---------:|------------:|----------:|:---------|
| Trend+DB3WaveletV2   | 16225995 |        34.3 |    0.8047 | ◀        |
| Trend+HaarWaveletV2  | 16225995 |        34.3 |    0.8012 | ◀        |
| Generic+DB3WaveletV2 | 25461120 |        -3.1 |    0.8162 |          |
| DB3AltWaveletV2      | 26115840 |        -5.7 |    0.9573 |          |
| DB3WaveletV2         | 26254080 |        -6.3 |    0.8765 |          |
| Coif2WaveletV2       | 26254080 |        -6.3 |    0.8638 |          |
| Symlet3WaveletV2     | 26254080 |        -6.3 |    0.8765 |          |
| HaarWaveletV2        | 26254080 |        -6.3 |    0.8643 |          |

### V3

| Config               |   Params |   Reduction |   Med OWA | Pareto   |
|:---------------------|---------:|------------:|----------:|:---------|
| Trend+Coif2WaveletV3 | 15433035 |        37.5 |    0.7969 | ◀        |
| Trend+DB3WaveletV3   | 15433035 |        37.5 |    0.7986 |          |
| Trend+HaarWaveletV3  | 15433035 |        37.5 |    0.8030 |          |
| Coif2WaveletV3       | 24668160 |         0.1 |    0.8035 |          |
| DB4WaveletV3         | 24668160 |         0.1 |    0.8057 |          |
| Generic+DB3WaveletV3 | 24668160 |         0.1 |    0.8001 |          |
| DB3WaveletV3         | 24668160 |         0.1 |    0.8045 |          |
| DB2WaveletV3         | 24668160 |         0.1 |    0.8091 |          |
| Symlet3WaveletV3     | 24668160 |         0.1 |    0.8132 |          |
| HaarWaveletV3        | 24668160 |         0.1 |    0.8127 |          |
