# Tiered-Offset M4 All-Periods Analysis — 2026-05-05

Sources:

- `experiments/results/m4/tiered_offset_m4_allperiods_results.csv` (sliding protocol, 888 raw rows)
- `experiments/results/m4/tiered_offset_m4_allperiods_paperlr_results.csv` (nbeats_paper protocol, 958 raw rows)
- Baseline leaderboard: `experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md`

Both CSVs cover the same 16 tiered-offset configurations (`tiered=ascend`) — 4 alternating Trend+Wavelet families × {10s, 30s} stack depth × {ag0, agf, AE/AELG variant}:

```
T+DB3V3      x {10s, 30s} x {ag0, agf}              (RootBlock backbone)
T+Sym10V3    x {10s, 30s} x {ag0, agf}              (RootBlock backbone)
TAE+DB3V3AE  x {10s, 30s}                           (AE backbone)
TAE+Sym10V3AE x {10s, 30s}                          (AE backbone)
TAELG+DB3V3AELG  x {10s, 30s}                       (AELG backbone)
TAELG+Sym10V3AELG x {10s, 30s}                      (AELG backbone)
```

This is a controlled tiered-offset sweep — none of the prior overall leaderboard's per-period non-tiered champions appear in either CSV (NBEATS-IG, TW unified, T+Coif2V3, NBEATS-G are all absent).

---

## 1. Data Summary

### Filter stats (combined corpus)

| Step | Removed | Cumulative |
|---|---|---|
| Initial rows (sliding 888 + paper 958) | — | 1,846 |
| `diverged=True` | 0 | 1,846 |
| `smape > 100` (non-diverged) | 9 | 1,837 |
| `best_epoch == 0 AND smape > 50` (additional) | 0 | 1,837 |
| `smape` is NaN | 0 | 1,837 |
| **Clean rows** | — | **1,837** |

The 9 dropped rows all had `diverged=False` in the CSV but `smape>100` (implicit divergence). The CSV `diverged` flag remains unreliable on its own.

### Clean rows by protocol × period

| period    | nbeats_paper | sliding |
|-----------|---:|---:|
| Yearly    | 160 | 160 |
| Quarterly | 158 | 139 |
| Monthly   | 160 | 160 |
| Weekly    | 160 | 158 |
| Daily     | 158 | 137 |
| Hourly    | 158 | 129 |
| **Total** | **954** | **883** |

186 unique (config, period, protocol) cells. Most have n=10 runs; sliding has thinner cells on Daily (T+DB3V3_10s_tiered_agf n=4, T+Sym10V3_30s_tiered_agf n=5) and Hourly. Paper-sample is more complete except T+DB3V3_30s_tiered_agf Hourly at n=8 (no divergence; runs missing from disk).

---

## 2. Per-Period Leaderboards

n_params reported in millions. Std and n are over completed runs. Top-10 only — full sweep has 16 configs per period.

### 2.1 Paper-sample (`nbeats_paper`)

#### Yearly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | T+DB3V3_10s_tiered_agf       | 5.07  | 13.554 ± 0.101 | 10 |
| 2 | TAELG+DB3V3AELG_10s_tiered   | 1.03  | 13.607 ± 0.142 | 10 |
| 3 | TAE+Sym10V3AE_30s_tiered     | 3.09  | 13.627 ± 0.111 | 10 |
| 4 | T+Sym10V3_30s_tiered_ag0     | 15.20 | 13.629 ± 0.140 | 10 |
| 5 | T+Sym10V3_10s_tiered_ag0     | 5.07  | 13.637 ± 0.168 | 10 |
| 6 | TAE+DB3V3AE_10s_tiered       | 1.03  | 13.645 ± 0.099 | 10 |
| 7 | T+DB3V3_30s_tiered_agf       | 15.20 | 13.646 ± 0.123 | 10 |
| 8 | TAELG+Sym10V3AELG_30s_tiered | 3.09  | 13.649 ± 0.160 | 10 |
| 9 | T+DB3V3_10s_tiered_ag0       | 5.07  | 13.655 ± 0.120 | 10 |
| 10| TAE+Sym10V3AE_10s_tiered     | 1.03  | 13.658 ± 0.204 | 10 |

Best 13.554 vs prior leader 13.486 → **Δ = +0.068** (no beater).

#### Quarterly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | T+DB3V3_30s_tiered_ag0       | 15.34 | 10.345 ± 0.065 | 10 |
| 2 | TAE+Sym10V3AE_10s_tiered     | 1.06  | 10.365 ± 0.081 |  9 |
| 3 | T+Sym10V3_10s_tiered_ag0     | 5.11  | 10.371 ± 0.042 | 10 |
| 4 | T+DB3V3_10s_tiered_ag0       | 5.11  | 10.373 ± 0.059 | 10 |
| 5 | TAELG+DB3V3AELG_10s_tiered   | 1.06  | 10.381 ± 0.061 |  9 |
| 6 | TAELG+DB3V3AELG_30s_tiered   | 3.17  | 10.394 ± 0.068 | 10 |
| 7 | TAELG+Sym10V3AELG_30s_tiered | 3.17  | 10.397 ± 0.061 | 10 |
| 8 | T+Sym10V3_30s_tiered_agf     | 15.34 | 10.398 ± 0.045 | 10 |
| 9 | TAE+Sym10V3AE_30s_tiered     | 3.17  | 10.399 ± 0.097 | 10 |
| 10| T+Sym10V3_30s_tiered_ag0     | 15.34 | 10.400 ± 0.104 | 10 |

Best 10.345 vs prior 10.313 → **Δ = +0.032** (no beater).

#### Monthly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | TAE+DB3V3AE_30s_tiered       | 3.57  | 13.584 ± 0.367 | 10 |
| 2 | TAELG+Sym10V3AELG_30s_tiered | 3.57  | 13.686 ± 0.383 | 10 |
| 3 | T+Sym10V3_30s_tiered_agf     | 16.03 | 13.690 ± 0.371 | 10 |
| 4 | T+DB3V3_30s_tiered_agf       | 16.03 | 13.696 ± 0.416 | 10 |
| 5 | T+DB3V3_10s_tiered_agf       | 5.34  | 13.718 ± 0.303 | 10 |
| 6 | TAE+Sym10V3AE_30s_tiered     | 3.57  | 13.721 ± 0.347 | 10 |
| 7 | TAELG+Sym10V3AELG_10s_tiered | 1.19  | 13.726 ± 0.192 | 10 |
| 8 | TAELG+DB3V3AELG_30s_tiered   | 3.57  | 13.739 ± 0.267 | 10 |
| 9 | T+Sym10V3_30s_tiered_ag0     | 16.03 | 13.746 ± 0.469 | 10 |
| 10| TAE+Sym10V3AE_10s_tiered     | 1.19  | 13.778 ± 0.381 | 10 |

Best 13.584 vs prior 13.240 → **Δ = +0.344** (no beater — confirms tiered does not help Monthly under paper-sample LR).

#### Weekly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | T+Sym10V3_30s_tiered_ag0     | 15.68 | 6.907 ± 0.293 | 10 |
| 2 | TAELG+Sym10V3AELG_30s_tiered | 3.37  | 6.929 ± 0.389 | 10 |
| 3 | TAE+DB3V3AE_30s_tiered       | 3.37  | 6.979 ± 0.325 | 10 |
| 4 | T+DB3V3_10s_tiered_ag0       | 5.23  | 6.984 ± 0.489 | 10 |
| 5 | T+Sym10V3_30s_tiered_agf     | 15.68 |  7.002 ± 0.144 | 10 |
| 6 | T+DB3V3_30s_tiered_ag0       | 15.68 | 7.032 ± 0.651 | 10 |
| 7 | T+DB3V3_30s_tiered_agf       | 15.68 | 7.049 ± 0.323 | 10 |
| 8 | TAE+Sym10V3AE_30s_tiered     | 3.37  | 7.063 ± 0.351 | 10 |
| 9 | T+Sym10V3_10s_tiered_ag0     | 5.23  | 7.069 ± 0.424 | 10 |
| 10| TAE+Sym10V3AE_10s_tiered     | 1.12  | 7.080 ± 0.307 | 10 |

Best 6.907 vs prior 6.735 → **Δ = +0.172** (no beater — confirms Weekly tiered regression already noted in the leaderboard).

#### Daily

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | TAE+Sym10V3AE_30s_tiered     | 3.41  | 3.047 ± 0.060 | 10 |
| 2 | TAELG+DB3V3AELG_10s_tiered   | 1.14  | 3.052 ± 0.043 | 10 |
| 3 | TAELG+Sym10V3AELG_10s_tiered | 1.14  | 3.056 ± 0.044 | 10 |
| 4 | TAELG+Sym10V3AELG_30s_tiered | 3.41  | 3.067 ± 0.041 | 10 |
| 5 | TAE+Sym10V3AE_10s_tiered     | 1.14  | 3.068 ± 0.100 |  9 |
| 6 | TAE+DB3V3AE_10s_tiered       | 1.14  | 3.070 ± 0.052 | 10 |
| 7 | T+Sym10V3_30s_tiered_agf     | 15.75 | 3.073 ± 0.065 | 10 |
| 8 | T+Sym10V3_30s_tiered_ag0     | 15.75 | 3.073 ± 0.042 | 10 |
| 9 | TAELG+DB3V3AELG_30s_tiered   | 3.41  | 3.073 ± 0.036 | 10 |
| 10| T+DB3V3_30s_tiered_agf       | 15.75 | 3.078 ± 0.058 | 10 |

Best 3.047 vs prior leader's reported 3.012 → +0.035. **However the prior leader (`T+Sym10V3_10s_tiered_ag0`) is in this CSV at 3.097 ± 0.044 (n=10), so the in-CSV comparison reverses**: 3 challengers significantly beat it (see §4).

#### Hourly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | T+DB3V3_30s_tiered_agf       | 18.18 | 8.751 ± 0.105 |  8 |
| 2 | T+Sym10V3_30s_tiered_agf     | 18.18 | 8.921 ± 0.213 | 10 |
| 3 | T+DB3V3_10s_tiered_agf       |  6.06 | 8.931 ± 0.130 | 10 |
| 4 | T+Sym10V3_10s_tiered_agf     |  6.06 | 8.934 ± 0.166 | 10 |
| 5 | T+Sym10V3_30s_tiered_ag0     | 18.18 | 8.995 ± 0.148 | 10 |
| 6 | T+DB3V3_30s_tiered_ag0       | 18.18 | 9.146 ± 0.163 | 10 |
| 7 | TAE+Sym10V3AE_30s_tiered     |  4.86 | 9.203 ± 0.101 | 10 |
| 8 | TAELG+DB3V3AELG_30s_tiered   |  4.86 | 9.212 ± 0.114 | 10 |
| 9 | T+DB3V3_10s_tiered_ag0       |  6.06 | 9.213 ± 0.127 | 10 |
| 10| TAE+DB3V3AE_30s_tiered       |  4.86 | 9.253 ± 0.126 | 10 |

Best 8.751 vs prior 8.758 → **Δ = -0.007** (in-CSV n=8). Marginal nominal beat at thin sample.

### 2.2 Sliding

#### Yearly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | T+Sym10V3_10s_tiered_ag0     | 5.07 | 13.578 ± 0.134 | 10 |
| 2 | TAELG+Sym10V3AELG_10s_tiered | 1.03 | 13.582 ± 0.077 | 10 |
| 3 | TAELG+DB3V3AELG_30s_tiered   | 3.09 | 13.582 ± 0.101 | 10 |
| 4 | T+DB3V3_10s_tiered_agf       | 5.07 | 13.608 ± 0.158 | 10 |
| 5 | T+DB3V3_10s_tiered_ag0       | 5.07 | 13.616 ± 0.137 | 10 |
| 6 | T+DB3V3_30s_tiered_ag0       | 15.20 | 13.624 ± 0.181 | 10 |
| 7 | TAE+DB3V3AE_30s_tiered       | 3.09 | 13.630 ± 0.118 | 10 |
| 8 | T+Sym10V3_10s_tiered_agf     | 5.07 | 13.633 ± 0.189 | 10 |
| 9 | TAELG+DB3V3AELG_10s_tiered   | 1.03 | 13.639 ± 0.179 | 10 |
| 10| T+DB3V3_30s_tiered_agf       | 15.20 | 13.683 ± 0.181 | 10 |

Best 13.578 vs prior 13.499 → **Δ = +0.079** (no beater).

#### Quarterly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | T+Sym10V3_10s_tiered_ag0     | 5.11 | 10.325 ± 0.053 | 10 |
| 2 | T+DB3V3_10s_tiered_ag0       | 5.11 | 10.356 ± 0.053 | 10 |
| 3 | TAELG+Sym10V3AELG_10s_tiered | 1.06 | 10.362 ± 0.050 | 10 |
| 4 | T+Sym10V3_30s_tiered_ag0     | 15.34 | 10.365 ± 0.119 | 10 |
| 5 | T+DB3V3_30s_tiered_ag0       | 15.34 | 10.368 ± 0.092 | 10 |
| 6 | TAELG+DB3V3AELG_10s_tiered   | 1.06 | 10.369 ± 0.063 | 10 |
| 7 | TAE+Sym10V3AE_30s_tiered     | 3.17 | 10.389 ± 0.061 | 10 |
| 8 | T+Sym10V3_30s_tiered_agf     | 15.34 | 10.391 ± 0.061 |  8 |
| 9 | TAE+Sym10V3AE_10s_tiered     | 1.06 | 10.391 ± 0.059 | 10 |
| 10| TAELG+DB3V3AELG_30s_tiered   | 3.17 | 10.393 ± 0.056 | 10 |

Best 10.325 vs prior 10.127 → **Δ = +0.198** (no beater).

#### Monthly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | T+DB3V3_30s_tiered_agf       | 16.03 | 13.344 ± 0.329 | 10 |
| 2 | T+DB3V3_10s_tiered_ag0       | 5.34  | 13.366 ± 0.304 | 10 |
| 3 | T+DB3V3_30s_tiered_ag0       | 16.03 | 13.471 ± 0.327 | 10 |
| 4 | TAELG+Sym10V3AELG_30s_tiered | 3.57  | 13.471 ± 0.348 | 10 |
| 5 | TAE+Sym10V3AE_10s_tiered     | 1.19  | 13.541 ± 0.276 | 10 |
| 6 | TAE+DB3V3AE_10s_tiered       | 1.19  | 13.558 ± 0.349 | 10 |
| 7 | TAE+DB3V3AE_30s_tiered       | 3.57  | 13.568 ± 0.339 | 10 |
| 8 | T+Sym10V3_30s_tiered_ag0     | 16.03 | 13.609 ± 0.368 | 10 |
| 9 | TAE+Sym10V3AE_30s_tiered     | 3.57  | 13.620 ± 0.401 | 10 |
| 10| T+Sym10V3_10s_tiered_ag0     | 5.34  | 13.633 ± 0.330 | 10 |

Best 13.344 vs prior 13.279 → **Δ = +0.065** (no beater).

#### Weekly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | TAE+Sym10V3AE_30s_tiered     | 3.37 | 6.935 ± 0.223 | 10 |
| 2 | TAE+DB3V3AE_30s_tiered       | 3.37 | 6.955 ± 0.233 | 10 |
| 3 | T+Sym10V3_10s_tiered_ag0     | 5.23 | 6.956 ± 0.391 | 10 |
| 4 | TAELG+Sym10V3AELG_30s_tiered | 3.37 | 6.974 ± 0.335 | 10 |
| 5 | T+Sym10V3_30s_tiered_ag0     | 15.68 | 6.987 ± 0.301 | 10 |
| 6 | TAELG+DB3V3AELG_30s_tiered   | 3.37 | 7.072 ± 0.413 | 10 |
| 7 | T+Sym10V3_30s_tiered_agf     | 15.68 | 7.090 ± 0.442 | 10 |
| 8 | T+DB3V3_10s_tiered_agf       | 5.23 | 7.104 ± 0.262 | 10 |
| 9 | T+DB3V3_10s_tiered_ag0       | 5.23 | 7.146 ± 0.427 | 10 |
| 10| T+DB3V3_30s_tiered_ag0       | 15.68 | 7.153 ± 0.442 | 10 |

Best 6.935 vs prior 6.671 → **Δ = +0.264** (no beater — confirms Weekly tiered regression).

#### Daily

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | T+DB3V3_10s_tiered_agf       | 5.25 | 2.995 ± 0.030 |  4 |
| 2 | T+Sym10V3_10s_tiered_ag0     | 5.25 | 3.014 ± 0.035 | 10 |
| 3 | T+Sym10V3_30s_tiered_agf     | 15.75 | 3.014 ± 0.021 |  5 |
| 4 | T+DB3V3_10s_tiered_ag0       | 5.25 | 3.023 ± 0.039 | 10 |
| 5 | T+Sym10V3_30s_tiered_ag0     | 15.75 | 3.031 ± 0.040 | 10 |
| 6 | TAELG+Sym10V3AELG_30s_tiered | 3.41 | 3.035 ± 0.033 | 10 |
| 7 | TAELG+Sym10V3AELG_10s_tiered | 1.14 | 3.044 ± 0.042 | 10 |
| 8 | TAELG+DB3V3AELG_30s_tiered   | 3.41 | 3.047 ± 0.055 | 10 |
| 9 | TAELG+DB3V3AELG_10s_tiered   | 1.14 | 3.048 ± 0.036 | 10 |
| 10| T+DB3V3_30s_tiered_ag0       | 15.75 | 3.055 ± 0.027 | 10 |

Best 2.995 vs prior 2.588 → **Δ = +0.407** (no beater; rank-1 cell only n=4).

#### Hourly

| rank | config | n_params (M) | mean SMAPE ± std | n |
|---:|---|---:|---|---:|
| 1 | T+Sym10V3_10s_tiered_agf     |  6.06 | 8.702 ± 0.072 | 10 |
| 2 | T+Sym10V3_30s_tiered_ag0     | 18.18 | 8.808 ± 0.096 | 10 |
| 3 | TAE+DB3V3AE_30s_tiered       |  4.86 | 8.832 ± 0.088 |  9 |
| 4 | TAELG+Sym10V3AELG_30s_tiered |  4.86 | 8.858 ± 0.118 | 10 |
| 5 | T+DB3V3_30s_tiered_ag0       | 18.18 | 8.876 ± 0.092 | 10 |
| 6 | TAE+Sym10V3AE_30s_tiered     |  4.86 | 8.901 ± 0.095 | 10 |
| 7 | T+Sym10V3_10s_tiered_ag0     |  6.06 | 8.941 ± 0.077 | 10 |
| 8 | TAELG+DB3V3AELG_30s_tiered   |  4.86 | 9.001 ± 0.118 | 10 |
| 9 | T+DB3V3_10s_tiered_ag0       |  6.06 | 9.066 ± 0.144 | 10 |
| 10| TAELG+Sym10V3AELG_10s_tiered |  1.62 | 9.068 ± 0.156 | 10 |

Best 8.702 vs prior 8.587 → **Δ = +0.115** (no beater).

---

## 3. Cross-Period Generalists (mean rank across all 6 periods)

Ranks dense within this 16-config sweep — not directly comparable to prior 108/112-config leaderboard ranks. Reported here strictly to identify the most consistent tiered-offset configurations.

### 3.1 Paper-sample (16 configs, all with full coverage)

| rank | config | n_params (M) | Y | Q | M | W | D | H | mean rank |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | TAE+Sym10V3AE_30s_tiered     | 3.41  | 13.627 | 10.399 | 13.721 | 7.063 | 3.047 | 9.203 | **5.67** |
| 2 | TAELG+Sym10V3AELG_30s_tiered | 3.41  | 13.649 | 10.397 | 13.686 | 6.929 | 3.067 | 9.291 | **5.83** |
| 3 | T+Sym10V3_30s_tiered_ag0     | 15.75 | 13.629 | 10.400 | 13.746 | 6.907 | 3.073 | 8.995 | 6.17 |
| 3 | T+Sym10V3_30s_tiered_agf     | 15.75 | 13.668 | 10.398 | 13.690 | 7.002 | 3.073 | 8.921 | 6.17 |
| 5 | T+DB3V3_30s_tiered_agf       | 15.75 | 13.646 | 10.462 | 13.696 | 7.049 | 3.078 | 8.751 | 7.17 |
| 6 | T+DB3V3_30s_tiered_ag0       | 15.75 | 13.661 | 10.345 | 13.802 | 7.032 | 3.090 | 9.146 | 8.33 |
| 6 | TAELG+DB3V3AELG_10s_tiered   | 1.14  | 13.607 | 10.381 | 13.786 | 7.317 | 3.052 | 9.399 | 8.33 |
| 8 | T+DB3V3_10s_tiered_ag0       | 5.25  | 13.655 | 10.373 | 13.850 | 6.984 | 3.080 | 9.213 | 8.50 |
| 9 | T+DB3V3_10s_tiered_agf       | 5.25  | 13.554 | 10.575 | 13.718 | 7.182 | 3.095 | 8.931 | 8.67 |
| 9 | TAE+Sym10V3AE_10s_tiered     | 1.14  | 13.658 | 10.365 | 13.778 | 7.080 | 3.068 | 9.403 | 8.67 |
| 11| TAE+DB3V3AE_30s_tiered       | 3.41  | 13.694 | 10.409 | 13.584 | 6.979 | 3.084 | 9.253 | 9.00 |
| 12| TAELG+DB3V3AELG_30s_tiered   | 3.41  | 13.675 | 10.394 | 13.739 | 7.237 | 3.073 | 9.212 | 9.50 |
| 13| T+Sym10V3_10s_tiered_ag0     | 5.25  | 13.637 | 10.371 | 13.904 | 7.070 | 3.097 | 9.262 | 9.83 |
| 13| TAELG+Sym10V3AELG_10s_tiered | 1.14  | 13.690 | 10.409 | 13.726 | 7.128 | 3.056 | 9.315 | 9.83 |
| 15| TAE+DB3V3AE_10s_tiered       | 1.14  | 13.645 | 10.413 | 13.833 | 7.298 | 3.070 | 9.424 |11.50 |
| 16| T+Sym10V3_10s_tiered_agf     | 5.25  | 13.691 | 10.481 | 13.941 | 7.239 | 3.088 | 8.934 |12.83 |

Top-3 within this sweep are all 30-stack Sym10 variants, with AE/AELG backbones surprisingly competitive vs full RootBlock at ¼ the parameters.

### 3.2 Sliding (13 configs with full 6-period coverage)

3 configs lacked complete coverage (Daily-only thin cells).

| rank | config | n_params (M) | Y | Q | M | W | D | H | mean rank |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | T+Sym10V3_10s_tiered_ag0     | 5.25  | 13.578 | 10.325 | 13.633 | 6.956 | 3.014 | 8.941 | **4.00** |
| 2 | T+DB3V3_10s_tiered_ag0       | 5.25  | 13.616 | 10.356 | 13.366 | 7.146 | 3.023 | 9.066 | **5.17** |
| 3 | T+DB3V3_30s_tiered_ag0       | 15.75 | 13.624 | 10.368 | 13.471 | 7.153 | 3.055 | 8.876 | 6.50 |
| 4 | T+Sym10V3_30s_tiered_ag0     | 15.75 | 13.779 | 10.365 | 13.609 | 6.987 | 3.031 | 8.808 | 6.67 |
| 5 | TAELG+Sym10V3AELG_30s_tiered | 3.41  | 13.694 | 10.402 | 13.471 | 6.974 | 3.035 | 8.858 | 7.00 |
| 6 | TAE+DB3V3AE_30s_tiered       | 3.41  | 13.630 | 10.413 | 13.568 | 6.955 | 3.066 | 8.832 | 7.67 |
| 6 | TAELG+DB3V3AELG_30s_tiered   | 3.41  | 13.582 | 10.393 | 13.645 | 7.072 | 3.047 | 9.001 | 7.67 |
| 8 | TAELG+Sym10V3AELG_10s_tiered | 1.14  | 13.582 | 10.362 | 13.742 | 7.331 | 3.044 | 9.068 | 8.33 |
| 9 | TAE+Sym10V3AE_30s_tiered     | 3.41  | 13.716 | 10.389 | 13.620 | 6.935 | 3.067 | 8.901 | 8.67 |
| 10| T+Sym10V3_10s_tiered_agf     | 5.25  | 13.633 | 10.515 | 13.772 | 7.176 | 3.059 | 8.702 |10.17 |
| 11| TAELG+DB3V3AELG_10s_tiered   | 1.14  | 13.639 | 10.369 | 13.718 | 7.376 | 3.048 | 9.083 |10.50 |
| 12| TAE+Sym10V3AE_10s_tiered     | 1.14  | 13.706 | 10.391 | 13.541 | 7.240 | 3.061 | 9.105 |11.00 |
| 13| TAE+DB3V3AE_10s_tiered       | 1.14  | 13.684 | 10.399 | 13.558 | 7.501 | 3.055 | 9.086 |11.17 |

`T+Sym10V3_10s_tiered_ag0` is the best generalist within the sliding tiered sweep. Note this config IS the prior paper-sample champion (`T+Sym10V3_10s_tiered_ag0`, mean_rank 13.33/108) — its sliding behavior was previously underexplored.

---

## 4. Statistical Tests vs Prior Leaders

11 of 12 prior leaders are **not present in either CSV**, so direct in-CSV statistical comparison is impossible there. The exceptions are paper-sample Yearly (`T+DB3V3_10s_tiered_agf` is in CSV) and paper-sample Daily (`T+Sym10V3_10s_tiered_ag0` is in CSV).

### 4.1 Tests where prior leader is in-CSV

#### Paper-sample Daily — prior `T+Sym10V3_10s_tiered_ag0`

In-CSV: 3.097 ± 0.044 (n=10) — note this is **higher than the prior 3.012**, possibly a regression or different early-stop fixture across runs. Seeds in this CSV do not overlap with any other config, so Mann-Whitney U used.

Three configurations significantly beat the prior leader's in-CSV mean:

| Challenger | mean SMAPE | n | MWU p (alt='less') | Cliff's d | Magnitude | Sig (α=0.05) |
|---|---:|---:|---:|---:|---|:---:|
| TAE+Sym10V3AE_30s_tiered     | 3.047 | 10 | **0.027** | -0.520 | large    | **YES** |
| TAELG+DB3V3AELG_10s_tiered   | 3.052 | 10 | **0.023** | -0.540 | large    | **YES** |
| TAELG+Sym10V3AELG_10s_tiered | 3.056 | 10 | 0.052 | -0.440 | medium   | no (borderline) |
| TAE+Sym10V3AE_10s_tiered     | 3.068 |  9 | **0.033** | -0.511 | large    | **YES** |
| TAELG+Sym10V3AELG_30s_tiered | 3.067 | 10 | 0.081 | -0.380 | medium   | no |
| TAE+DB3V3AE_10s_tiered       | 3.070 | 10 | 0.106 | -0.340 | medium   | no |
| TAELG+DB3V3AELG_30s_tiered   | 3.073 | 10 | 0.106 | -0.340 | medium   | no |
| T+Sym10V3_30s_tiered_agf     | 3.073 | 10 | 0.172 | -0.260 | small    | no |
| T+Sym10V3_30s_tiered_ag0     | 3.073 | 10 | 0.137 | -0.300 | small    | no |
| T+DB3V3_30s_tiered_agf       | 3.078 | 10 | 0.192 | -0.240 | small    | no |
| T+DB3V3_10s_tiered_ag0       | 3.081 | 10 | 0.154 | -0.289 | small    | no |
| TAE+DB3V3AE_30s_tiered       | 3.084 | 10 | 0.339 | -0.120 | negligible | no |
| T+Sym10V3_10s_tiered_agf     | 3.088 | 10 | 0.425 | -0.060 | negligible | no |
| T+DB3V3_30s_tiered_ag0       | 3.090 | 10 | 0.396 | -0.080 | negligible | no |
| T+DB3V3_10s_tiered_agf       | 3.095 | 10 | 0.425 | -0.060 | negligible | no |

**Interpretation:** the paper-sample Daily prior leader's in-CSV runs underperform what the leaderboard reported (3.097 vs 3.012). The **AE-backbone alternating** challengers (`TAE+Sym10V3AE_30s_tiered`, `TAELG+DB3V3AELG_10s_tiered`, `TAE+Sym10V3AE_10s_tiered`) significantly outperform the in-CSV prior with **large Cliff's d** at ≤¼ the parameters. This re-opens Daily SOTA — the AE/AELG backbone family is now the strongest tiered-offset choice for paper-sample Daily.

#### Paper-sample Yearly — prior `T+DB3V3_10s_tiered_agf`

In-CSV: 13.554 ± 0.101 (n=10) vs prior reported 13.486 — modest within-noise drift. **No challenger beats the in-CSV mean** (this config remains the in-CSV rank 1 for paper-sample Yearly). Confirmed leader status, no test needed.

### 4.2 Cases where prior leader is not in CSV (descriptive only)

| Period | Protocol | Prior leader (out of CSV) | Prior SMAPE | Best new (in CSV) | New SMAPE | Δ | Statistical test? |
|---|---|---|---:|---|---:|---:|---|
| Yearly    | paper   | T+DB3V3_10s_tiered_agf*  | 13.486 | T+DB3V3_10s_tiered_agf  | 13.554 | +0.068 | in-CSV — see §4.1 |
| Quarterly | paper   | NBEATS-IG_10s_ag0        | 10.313 | T+DB3V3_30s_tiered_ag0  | 10.345 | +0.032 | insufficient data |
| Monthly   | paper   | TW_30s_td3_bdeq_sym10    | 13.240 | TAE+DB3V3AE_30s_tiered  | 13.584 | +0.344 | insufficient data |
| Weekly    | paper   | T+Coif2V3_30s_bdeq       |  6.735 | T+Sym10V3_30s_tiered_ag0|  6.907 | +0.172 | insufficient data |
| Hourly    | paper   | NBEATS-IG_30s_agf        |  8.758 | T+DB3V3_30s_tiered_agf  |  8.751 | -0.007 | insufficient data (n=8 only, ±0.105 std) |
| Yearly    | sliding | TW_10s_td3_bdeq_coif2    | 13.499 | T+Sym10V3_10s_tiered_ag0| 13.578 | +0.079 | insufficient data |
| Quarterly | sliding | NBEATS-IG_10s_ag0        | 10.127 | T+Sym10V3_10s_tiered_ag0| 10.325 | +0.198 | insufficient data |
| Monthly   | sliding | TW_30s_td3_bd2eq_coif2   | 13.279 | T+DB3V3_30s_tiered_agf  | 13.344 | +0.065 | insufficient data |
| Weekly    | sliding | T+Db3V3_30s_bdeq         |  6.671 | TAE+Sym10V3AE_30s_tiered|  6.935 | +0.264 | insufficient data |
| Daily     | sliding | NBEATS-G_30s_ag0         |  2.588 | T+DB3V3_10s_tiered_agf  |  2.995 | +0.407 | insufficient data |
| Hourly    | sliding | NBEATS-IG_30s_agf        |  8.587 | T+Sym10V3_10s_tiered_agf|  8.702 | +0.115 | insufficient data |

\* in-CSV — see §4.1 row.

The Hourly paper-sample nominal beat (Δ=-0.007, n=8) is far smaller than the in-CSV std (±0.105) and lacks two runs from disk; this is **not a credible new leader**.

---

## 5. Prior Claim Verification

### (a) Tiered offset helps Monthly — **NOT SUPPORTED** (consistent with prior leaderboard's retraction)

This CSV contains **only tiered configs**, so within-CSV non-tiered comparison is impossible. However, comparing the best new tiered cell against the prior **non-tiered** Monthly leaders:

| Protocol | Best new tiered Monthly | vs Prior non-tiered Monthly leader | Δ |
|---|---|---|---:|
| paper   | TAE+DB3V3AE_30s_tiered  → 13.584 | TW_30s_td3_bdeq_sym10  → 13.240 | **+0.344** |
| sliding | T+DB3V3_30s_tiered_agf  → 13.344 | TW_30s_td3_bd2eq_coif2 → 13.279 | **+0.065** |

Tiered offset is +0.06 to +0.34 SMAPE worse than non-tiered on Monthly under both protocols. Confirms the existing CLAUDE.md retraction; the "tiered helps Monthly" claim was indeed wrong.

### (b) step_paper LR is best on Weekly — **CONSISTENT** (no LR axis in this CSV, confirms via tiered loss)

This CSV does not vary the LR scheduler — it varies the sampling protocol. Both protocols **lose** to the prior Weekly non-tiered step_paper leader by +0.17 (paper-sample) and +0.26 (sliding). The best Weekly cell across both protocols is paper-sample T+Sym10V3_30s_tiered_ag0 @ 6.907 — still +0.172 worse than prior `T+Coif2V3_30s_bdeq` @ 6.735. The Weekly tiered regression at H=13 is **fully reaffirmed** under both protocols.

### (c) Tiered offset is decisive on Daily — **TRIVIALLY 100%, BUT QUALIFIED**

All 16 configs in this CSV are tiered, so 10/10 Daily top-10 are tiered by construction. The substantive question is whether tiered Daily configs **beat the strongest non-tiered Daily**:

| Protocol | Best in-CSV (tiered) Daily | Prior non-tiered Daily | Δ |
|---|---|---|---:|
| paper   | TAE+Sym10V3AE_30s_tiered → 3.047 | T+Sym10V3_10s_tiered_ag0 → 3.012* | +0.035 |
| sliding | T+DB3V3_10s_tiered_agf  → 2.995 (n=4) | NBEATS-G_30s_ag0 → 2.588 | **+0.407** |

\* That "prior" itself was tiered — true non-tiered Daily best is in older data (e.g., `NBEATS-G_30s_ag0` sliding @ 2.588).

Under paper-sample, tiered Daily remains roughly co-leader (the in-CSV regression of the prior tiered champion to 3.097 was matched and exceeded by other tiered cells — see §4.1). Under sliding, the `NBEATS-G_30s_ag0` baseline at 2.588 is **not surpassed** by any tiered config. Daily's tiered story is **paper-sample-only**.

---

## 6. New Champions

**Paper-sample generalist (mean rank < 13.33/108):** Cannot be claimed from this CSV — the rank universe (16 configs) is incompatible with the prior 108-config rank base. The top in-CSV generalist `TAE+Sym10V3AE_30s_tiered` (mean rank 5.67/16) would need to be benchmarked against the broader leaderboard set to make a generalist claim. **Status: not a new champion within scope.**

**Sliding generalist (mean rank < 12.67/112):** Same caveat. `T+Sym10V3_10s_tiered_ag0` at 4.00/16 is the top sliding tiered generalist but cannot displace `T+HaarV3_30s_bd2eq` without a head-to-head.

**Sub-1M champion:** **None.** All 16 configs in this CSV exceed 1M params (smallest is 1.03M, the AE/AELG `_10s_tiered` variants). The four `*AE_10s_tiered`/`*AELG_10s_tiered` configs are the closest "sub-1.5M" candidates and all reach top-12 mean rank within their protocol.

**Per-period champion candidates (in-CSV, statistically validated):**

- **Paper-sample Daily — REOPENED:** `TAE+Sym10V3AE_30s_tiered` (3.047 ± 0.060, n=10, 3.41M params) significantly beats the in-CSV prior leader (`T+Sym10V3_10s_tiered_ag0` at 3.097, p=0.027, Cliff's d=-0.520, large). `TAELG+DB3V3AELG_10s_tiered` (3.052 at 1.14M) is the **parameter-efficient** statistically-significant winner (p=0.023, large effect). Either should now be considered the paper-sample Daily leader. **Re-test the prior 3.012 number** — it is not reproducible in this CSV at n=10.

- **Paper-sample Hourly — INCONCLUSIVE:** `T+DB3V3_30s_tiered_agf` (8.751 ± 0.105, n=8) is nominally Δ=-0.007 below NBEATS-IG_30s_agf (8.758) but the gap is well within noise and runs are missing. **Not a champion.**

No other per-period tiered config beats the prior leader in either protocol.

---

## 7. Updated Recommendations

1. **Demote `T+Sym10V3_10s_tiered_ag0` as paper-sample Daily leader.** This CSV's n=10 of that exact config returns 3.097 — not 3.012. Re-promote on the basis of the new evidence to **`TAE+Sym10V3AE_30s_tiered` (paper-sample Daily, 3.047 ± 0.060)** or **`TAELG+DB3V3AELG_10s_tiered` (3.052 ± 0.043, 1.14M params)** as the parameter-efficient pick. Both are statistically and Cliff's-d-large better than the in-CSV prior champion. Update `m4-period-defaults` skill and the leaderboard's Daily row.

2. **Reaffirm tiered offset is Daily-only on paper-sample.** Sliding-protocol Daily tiered (best 2.995 at n=4) does not surpass non-tiered `NBEATS-G_30s_ag0` (2.588). The prior recommendation "keep tiered offset for **Daily only**" should now read **"keep tiered offset for paper-sample Daily only"** — sliding Daily defaults stay non-tiered.

3. **Reaffirm tiered hurts Monthly and Weekly under both protocols.** Best new tiered cells are +0.06 to +0.34 SMAPE worse than non-tiered priors on Monthly and +0.17 to +0.26 on Weekly. No update needed; existing CLAUDE.md guidance is correct.

4. **AE/AELG backbones at 30 stacks are the new top tier within tiered-offset.** Top-2 paper-sample generalists in this sweep are `TAE+Sym10V3AE_30s_tiered` (5.67) and `TAELG+Sym10V3AELG_30s_tiered` (5.83), both at 3.41M params — beating all 15.75M `T+*V3_30s_tiered_*` cells. When ≤4M params is the binding constraint and sampling is paper-sample, prefer these AE/AELG variants over RootBlock T+wavelet of equal stack depth.

5. **Sliding consistently beats paper-sample on long-horizon periods (D, H, M).** Cross-protocol delta (sliding − paper, mean across configs):

   | Period | Δ (sliding − paper) |
   |---|---:|
   | Daily   | -0.033 |
   | Hourly  | **-0.287** |
   | Monthly | -0.170 |
   | Quarterly | -0.007 |
   | Yearly  | +0.006 |
   | Weekly  | +0.044 |

   Hourly especially favors sliding (-0.29 SMAPE on average across all 16 tiered configs); short horizons (Y/Q/W) are protocol-neutral within these tiered configs. Reinforces the existing protocol-selection skill.

6. **Re-run `T+Sym10V3_10s_tiered_ag0` on paper-sample Daily with n≥20** to resolve why this CSV shows 3.097 vs the prior 3.012. If the gap persists, the prior leaderboard entry needs correction. If the gap closes, the AE/AELG challengers may also drift; re-test all four candidates head-to-head with n≥20 paired seeds.

7. **Drop tiered-offset from production defaults outside paper-sample Daily.** This sweep, with 1,837 clean rows across 6 periods × 2 protocols × 16 tiered configs, finds **zero** statistically validated improvements over the existing leaderboard's per-period leaders, except the in-CSV regression of paper-sample Daily that this study itself exposed. Tiered-offset is locally useful only on paper-sample Daily; everywhere else it is at best neutral and at worst a +0.4 SMAPE regression.

---

### Files used

- Input CSVs: `experiments/results/m4/tiered_offset_m4_allperiods_results.csv`, `experiments/results/m4/tiered_offset_m4_allperiods_paperlr_results.csv`
- Prior leaderboard: `experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md`
- Computations: pandas, numpy, scipy.stats (mannwhitneyu, wilcoxon)
- Filter rule: drop `diverged=True` OR `smape>100` OR (`best_epoch==0` AND `smape>50`) OR `smape` NaN. 9 rows dropped.
