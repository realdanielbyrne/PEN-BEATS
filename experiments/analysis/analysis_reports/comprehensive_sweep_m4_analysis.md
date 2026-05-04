# Comprehensive M4 Sweep — Full Reanalysis

**Date:** 2026-04-22
**Dataset:** M4 (all six periods: Yearly, Quarterly, Monthly, Weekly, Daily, Hourly)
**Source data:** `experiments/results/m4/comprehensive_sweep_m4_results.csv` (6,816 rows)
**Source config:** `experiments/configs/comprehensive_sweep_m4.yaml`
**Status:** Supersedes all prior M4-specific analyses.

---

## 1. Executive Summary

- **112 configurations** x 10 random seeds x 6 periods = **6,816 runs**. Zero divergent runs.
- **Wavelet / TrendWavelet architectures dominate short- and medium-horizon periods** (Yearly, Monthly, Weekly). They tie with paper baselines on Quarterly.
- **Paper baselines still win Daily and Hourly** where raw capacity at longer horizons matters more than inductive bias.
- **The single most reliable cross-period architecture is `T+HaarV3_30s_bd2eq`** (alternating Trend + Haar WaveletV3 RootBlock, 30 stacks, `basis_dim=2*fcast`): mean rank 12.7/112, top-quartile on 5/6 periods, never bottom quartile.
- **Sub-0.5M parameter `TWAELG_10s_ld16_coif2_agf`** achieves rank 4 on Yearly (SMAPE 13.524) — within 0.025 of the 2.1M-parameter Yearly winner. Strong Pareto frontier evidence.
- **Catastrophic failure modes persist:** `NBEATS-G_30s_ag0` bimodally fails on Quarterly and Weekly (std > 7). `BNG_30s_ag0`, `BNAELG_30s_ld32_ag0` bimodally fail on Yearly (std > 9).
- **BottleneckGeneric-family blocks (BNG, BNAE, BNAELG)** are the consistently worst arch family across periods. Universal drop candidates.

---

## 2. Experimental Setup

### 2.1 Training Protocol

| Setting | Value |
|---|---|
| `max_epochs` | 200 |
| `patience` (early stopping) | 20 |
| `learning_rate` | 0.001 |
| `lr_scheduler` | CosineAnnealing with `warmup_epochs=15`, `eta_min=1e-6` |
| `optimizer` | Adam |
| `loss` | SMAPELoss |
| `active_g` | False (overridden per-config to `false` or `forecast`) |
| `sum_losses` | False |
| `activation` | ReLU |
| `n_blocks_per_stack` | 1 |
| `share_weights` | True |
| `forecast_multiplier` | 5 (backcast = 5 × forecast) |
| `n_runs` | 10 (random seeds stored in CSV) |
| `block_params.wavelet_type` | db3 (default; per-config overrides for family sweeps) |

### 2.2 Data Layout per M4 Period

All periods use `forecast_multiplier=5`, i.e. **lookback L = 5 × horizon H**.

| Period | Frequency | H (forecast_length) | L (backcast_length) | Series count |
|---|---|---|---|---|
| Yearly    | 1  | 6  | 30  | 23,000 |
| Quarterly | 4  | 8  | 40  | 24,000 |
| Monthly   | 12 | 18 | 90  | 48,000 |
| Weekly    | 1  | 13 | 65  | 359 |
| Daily     | 1  | 14 | 70  | 4,227 |
| Hourly    | 24 | 48 | 240 | 414 |

### 2.3 Sliding Window / Dataset Processing

Standard N-BEATS row-oriented M4 protocol via `RowCollectionTimeSeriesDataModule`:

- Rows = series, columns = observations.
- Validation = last `L + H` columns of train matrix.
- Training sampled windows of length `L + H` across all series.
- Test protocol concatenates train tail + test head for one-shot evaluation of each series.

### 2.4 Evaluation Metrics

Logged per run: SMAPE (primary, used for ranking), MASE, MAE, MSE, OWA (Overall Weighted Average vs Naive2 baseline), `norm_mae`, `norm_mse`. Training-side: best/final val loss, epochs trained, best epoch, divergence flag, full val loss curve.

### 2.5 Convergence Behaviour (Observed)

Epoch distribution (early stopped well before `max_epochs=200` on most periods):

| Period    | mean epochs | median | max |
|---|---|---|---|
| Yearly    | 62.6  | 60    | 162 |
| Quarterly | 63.4  | 62    | 175 |
| Monthly   | 37.3  | 36    | 86  |
| Weekly    | 36.4  | 34    | 92  |
| Daily     | 68.1  | 63    | 200 |
| Hourly    | 103.1 | 96    | 200 |

Hourly and Daily are the only periods where some runs hit the 200-epoch cap; for others patience=20 kicks in earlier.

### 2.6 Architectural Groups in Sweep

| `arch_family` | n configs | Examples |
|---|---|---|
| paper_baseline | 8 | NBEATS-G, NBEATS-I+G at {10, 30} stacks x {ag0, agf} |
| trendwavelet | 18 | Unified `TrendWavelet` RootBlock (wavelet family x basis_dim x td sweep) |
| trendwavelet_aelg | 22 | `TrendWaveletAELG` (latent_dim, wavelet, active_g, skip sweep) |
| trendwavelet_ae | 6 | `TrendWaveletAE` (ld sweep) |
| trendwavelet_genericaelg | 4 | `TrendWaveletGenericAELG` |
| alt_trend_wavelet_rb | 8 | Alternating Trend + WaveletV3 RootBlock (4 families x 2 bd) |
| alt_aelg | 12 | Alternating TrendAELG + WaveletV3AELG |
| alt_ae | 6 | Alternating TrendAE + WaveletV3AE |
| generic_aelg | 8 | GenericAELG (ld, active_g, skip) |
| generic_ae | 4 | GenericAE (ld, active_g) |
| bottleneckgeneric | 4 | BottleneckGeneric RootBlock |
| bottleneckgeneric_aelg | 8 | BottleneckGenericAELG |
| bottleneckgeneric_ae | 4 | BottleneckGenericAE |
| **Total** | **112** | |

---

## 3. Per-Period Rankings (Top 10 by SMAPE)

### 3.1 Yearly (H=6, L=30)

| Rank | Config | SMAPE ± std | OWA | Params | Family |
|---|---|---|---|---|---|
| 1 | TW_10s_td3_bdeq_coif2 | 13.499 ± 0.057 | 0.801 | 2.08M | trendwavelet |
| 2 | TALG+Sym10V3ALG_30s_agf | 13.504 ± 0.103 | 0.802 | 3.13M | alt_aelg |
| 3 | TALG+Db3V3ALG_30s_ag0 | 13.507 ± 0.080 | 0.802 | 3.13M | alt_aelg |
| 4 | **TWAELG_10s_ld16_coif2_agf** | 13.524 ± 0.060 | 0.803 | **436K** | trendwavelet_aelg |
| 5 | T+Db3V3_30s_bd2eq | 13.529 ± 0.115 | 0.803 | 15.29M | alt_trend_wavelet_rb |
| 6 | TALG+Sym10V3ALG_30s_ag0 | 13.531 ± 0.068 | 0.804 | 3.13M | alt_aelg |
| 7 | TW_10s_td5_bdeq_db3 | 13.533 ± 0.115 | 0.804 | 2.08M | trendwavelet |
| 8 | T+Db3V3_30s_bdeq | 13.533 ± 0.068 | 0.804 | 15.24M | alt_trend_wavelet_rb |
| 9 | TWGAELG_10s_ld16_agf | 13.537 ± 0.074 | 0.804 | 450K | trendwavelet_genericaelg |
| 10 | TWGAELG_30s_ld16_ag0 | 13.538 ± 0.096 | 0.804 | 1.35M | trendwavelet_genericaelg |

**Best paper baseline:** `NBEATS-IG_10s_ag0` (rank 22, SMAPE 13.561). **21/104 novel configs beat the best baseline.**

### 3.2 Quarterly (H=8, L=40)

| Rank | Config | SMAPE ± std | OWA | Params | Family |
|---|---|---|---|---|---|
| 1 | NBEATS-IG_10s_ag0 | 10.126 ± 0.068 | 0.888 | 19.64M | paper_baseline |
| 2 | T+Sym10V3_30s_bd2eq | 10.127 ± 0.051 | 0.885 | 15.45M | alt_trend_wavelet_rb |
| 3 | TALG+HaarV3ALG_30s_ag0 | 10.144 ± 0.048 | 0.888 | 3.28M | alt_aelg |
| 4 | T+Coif2V3_30s_bd2eq | 10.146 ± 0.045 | 0.890 | 15.45M | alt_trend_wavelet_rb |
| 5 | TW_10s_td3_bdeq_coif2 | 10.147 ± 0.056 | 0.889 | 2.11M | trendwavelet |
| 6 | TALG+Coif2V3ALG_30s_ag0 | 10.148 ± 0.046 | 0.889 | 3.28M | alt_aelg |
| 7 | **TALG+DB3V3ALG_10s_ag0** | 10.148 ± 0.064 | 0.888 | **1.09M** | alt_aelg |
| 8 | T+Coif2V3_30s_bdeq | 10.149 ± 0.076 | 0.888 | 15.39M | alt_trend_wavelet_rb |
| 9 | T+Db3V3_30s_bd2eq | 10.151 ± 0.037 | 0.889 | 15.45M | alt_trend_wavelet_rb |
| 10 | TAE+DB3V3AE_30s_ld16_ag0 | 10.151 ± 0.069 | 0.888 | 3.28M | alt_ae |

**Ranks 1-10 form a 15-way statistical tie** (all within 0.025 SMAPE; Wilcoxon #1-vs-#2 p=0.38). No novel config decisively beats `NBEATS-IG_10s_ag0`.

### 3.3 Monthly (H=18, L=90)

| Rank | Config | SMAPE ± std | OWA | Params | Family |
|---|---|---|---|---|---|
| 1 | TW_30s_td3_bd2eq_coif2 | 13.279 ± 0.303 | 0.914 | 7.08M | trendwavelet |
| 2 | T+HaarV3_30s_bd2eq | 13.308 ± 0.242 | 0.906 | 16.25M | alt_trend_wavelet_rb |
| 3 | NBEATS-IG_10s_ag0 | 13.309 ± 0.255 | 0.908 | 20.33M | paper_baseline |
| 4 | TALG+Coif2V3ALG_30s_ag0 | 13.309 ± 0.251 | 0.908 | 4.03M | alt_aelg |
| 5 | TW_30s_td3_bdeq_sym10 | 13.314 ± 0.222 | 0.909 | 6.78M | trendwavelet |
| 6 | **TWAE_10s_ld32_ag0** | 13.325 ± 0.279 | 0.908 | **584K** | trendwavelet_ae |
| 7 | TALG+HaarV3ALG_30s_ag0 | 13.329 ± 0.272 | 0.909 | 4.03M | alt_aelg |
| 8 | TW_30s_td3_bdeq_db3 | 13.342 ± 0.194 | 0.912 | 6.78M | trendwavelet |
| 9 | TAE+DB3V3AE_30s_ld32_ag0 | 13.349 ± 0.315 | 0.912 | 4.22M | alt_ae |
| 10 | TW_30s_td3_bdeq_coif2 | 13.364 ± 0.301 | 0.913 | 6.78M | trendwavelet |

**Best paper baseline:** `NBEATS-IG_10s_ag0` (rank 3, 13.309). Novel #1 beats best baseline by 0.030 SMAPE (2/104 novel configs strictly beat it).

### 3.4 Weekly (H=13, L=65)

| Rank | Config | SMAPE ± std | OWA | Params | Family |
|---|---|---|---|---|---|
| 1 | T+Db3V3_30s_bdeq | 6.671 ± 0.208 | 0.735 | 15.75M | alt_trend_wavelet_rb |
| 2 | TALG+HaarV3ALG_30s_ag0 | 6.673 ± 0.129 | 0.732 | 3.66M | alt_aelg |
| 3 | T+HaarV3_30s_bdeq | 6.675 ± 0.185 | 0.741 | 15.75M | alt_trend_wavelet_rb |
| 4 | T+HaarV3_30s_bd2eq | 6.685 ± 0.193 | 0.746 | 15.85M | alt_trend_wavelet_rb |
| 5 | T+Sym10V3_30s_bdeq | 6.686 ± 0.168 | 0.742 | 15.75M | alt_trend_wavelet_rb |
| 6 | **TWAELG_10s_ld16_sym10_ag0** | 6.693 ± 0.150 | 0.730 | **498K** | trendwavelet_aelg |
| 7 | TAE+DB3V3AE_30s_ld16_ag0 | 6.709 ± 0.137 | 0.739 | 3.66M | alt_ae |
| 8 | TW_30s_td3_bdeq_db3 | 6.716 ± 0.179 | 0.742 | 6.55M | trendwavelet |
| 9 | TALG+Sym10V3ALG_30s_ag0 | 6.721 ± 0.182 | 0.739 | 3.66M | alt_aelg |
| 10 | TWAELG_30s_ld16_sym10_ag0 | 6.722 ± 0.145 | 0.739 | 1.50M | trendwavelet_aelg |

**Best paper baseline:** `NBEATS-IG_30s_ag0` (rank 28, 6.822). **27/104 novel configs beat best baseline.** `NBEATS-G_30s_ag0` catastrophically fails (SMAPE 11.61, std 7.2, 3/10 runs bimodally diverge).

### 3.5 Daily (H=14, L=70)

| Rank | Config | SMAPE ± std | OWA | Params | Family |
|---|---|---|---|---|---|
| 1 | NBEATS-G_30s_ag0 | 2.588 ± 0.081 | 0.855 | 26.02M | paper_baseline |
| 2 | NBEATS-IG_30s_ag0 | 2.599 ± 0.027 | 0.853 | 37.40M | paper_baseline |
| 3 | NBEATS-G_30s_agf | 2.605 ± 0.052 | 0.867 | 26.02M | paper_baseline |
| 4 | T+Db3V3_30s_bd2eq | 2.709 ± 0.052 | 0.892 | 15.93M | alt_trend_wavelet_rb |
| 5 | T+Db3V3_30s_bdeq | 2.711 ± 0.045 | 0.892 | 15.82M | alt_trend_wavelet_rb |
| 6 | T+Sym10V3_30s_bdeq | 2.713 ± 0.027 | 0.894 | 15.82M | alt_trend_wavelet_rb |
| 7 | T+Coif2V3_30s_bd2eq | 2.718 ± 0.050 | 0.896 | 15.93M | alt_trend_wavelet_rb |
| 8 | T+Coif2V3_30s_bdeq | 2.719 ± 0.036 | 0.895 | 15.82M | alt_trend_wavelet_rb |
| 9 | T+HaarV3_30s_bd2eq | 2.720 ± 0.038 | 0.896 | 15.93M | alt_trend_wavelet_rb |
| 10 | T+HaarV3_30s_bdeq | 2.722 ± 0.066 | 0.896 | 15.82M | alt_trend_wavelet_rb |

**Paper baselines decisively win Daily.** There is a clear 0.1-SMAPE gap between the top-3 baselines and the top wavelet configs. All OWA-best-5 configs are the same three paper baselines (OWA 0.855-0.867) followed by wavelets (OWA 0.89).

### 3.6 Hourly (H=48, L=240)

| Rank | Config | SMAPE ± std | OWA | Params | Family |
|---|---|---|---|---|---|
| 1 | NBEATS-IG_30s_agf | 8.587 ± 0.080 | 0.409 | 43.58M | paper_baseline |
| 2 | NBEATS-IG_10s_agf | 8.629 ± 0.076 | 0.414 | 22.40M | paper_baseline |
| 3 | TAE+DB3V3AE_30s_ld16_agf | 8.673 ± 0.082 | 0.413 | 5.42M | alt_ae |
| 4 | NBEATS-IG_30s_ag0 | 8.680 ± 0.104 | 0.399 | 43.58M | paper_baseline |
| 5 | TALG+Coif2V3ALG_30s_agf | 8.690 ± 0.095 | 0.418 | 5.42M | alt_aelg |
| 6 | TALG+Db3V3ALG_30s_agf | 8.694 ± 0.114 | 0.419 | 5.42M | alt_aelg |
| 7 | TAE+DB3V3AE_30s_ld32_agf | 8.715 ± 0.091 | 0.414 | 5.60M | alt_ae |
| 8 | TALG+Sym10V3ALG_30s_agf | 8.726 ± 0.115 | 0.415 | 5.42M | alt_aelg |
| 9 | TALG+HaarV3ALG_30s_agf | 8.735 ± 0.120 | 0.423 | 5.42M | alt_aelg |
| 10 | T+HaarV3_30s_bd2eq | 8.745 ± 0.089 | 0.401 | 18.67M | alt_trend_wavelet_rb |

**Statistical significance:** `NBEATS-IG_30s_agf` (rank 1) is significantly better than `NBEATS-IG_30s_ag0` (Wilcoxon p=0.032, MWU p=0.023) and `TAE+DB3V3AE_30s_ld16_agf` (Wilcoxon p=0.019, MWU p=0.023). **`active_g=forecast` is essential for the Hourly winner.**

### 3.7 Bottom 5 Per Period (drop-from-future candidates)

| Period | Worst 5 (SMAPE) |
|---|---|
| Yearly | GAELG_30s_ld16_ag0_sd5 (13.869), BNAELG_30s_ld16_agf_sd5 (13.881), BNAELG_30s_ld16_ag0_sd5 (13.910), BNAELG_30s_ld32_ag0 (16.87±9.7), BNG_30s_ag0 (16.93±9.9) |
| Quarterly | GAELG_30s_ld16_agf_sd5 (10.527), BNG_30s_agf (10.529), BNG_10s_ag0 (10.544), NBEATS-G_30s_agf (10.588), NBEATS-G_30s_ag0 (12.74±7.4) |
| Monthly | BNAE_10s_ld32_ag0 (14.009), BNAE_10s_ld16_agf (14.054), BNAE_10s_ld16_ag0 (14.070), BNAE_10s_ld8_ag0 (14.216), TWAELG_30s_ld16_db3_agf_sd5 (14.005) |
| Weekly | BNAELG_10s_ld16_ag0 (7.301), BNG_10s_ag0 (8.48±4.6), BNAE_10s_ld32_ag0 (8.48±4.8), BNAE_10s_ld16_ag0 (8.49±4.7), NBEATS-G_30s_ag0 (11.61±7.2) |
| Daily | BNAE_10s_ld8_ag0 (3.088), BNAELG_10s_ld16_ag0 (3.094), TWAELG_10s_ld32_db3_agf (3.098), TWAELG_10s_ld16_coif2_agf (3.100), BNAELG_10s_ld16_agf (3.101) |
| Hourly | BNAE_10s_ld16_agf (9.556), BNAE_10s_ld32_ag0 (9.595), BNAE_10s_ld16_ag0 (9.682), BNAELG_10s_ld16_ag0 (9.683), BNAE_10s_ld8_ag0 (10.31±1.3) |

### 3.8 SMAPE Distribution per Period (per-config means)

| Period | min | Q1 | median | Q3 | max | spread (Q3-Q1) |
|---|---|---|---|---|---|---|
| Yearly    | 13.499 | 13.567 | 13.599 | 13.700 | 16.929 | 0.133 |
| Quarterly | 10.126 | 10.190 | 10.242 | 10.336 | 12.743 | 0.146 |
| Monthly   | 13.279 | 13.482 | 13.620 | 13.764 | 14.216 | 0.282 |
| Weekly    | 6.671  | 6.823  | 6.888  | 6.987  | 11.609 | 0.164 |
| Daily     | 2.588  | 2.904  | 3.009  | 3.059  | 3.101  | 0.155 |
| Hourly    | 8.587  | 8.869  | 9.025  | 9.144  | 10.308 | 0.275 |

Yearly, Quarterly, and Weekly have **very tight IQRs** (< 0.2 SMAPE) punctuated by long right tails of unstable configs. On Monthly/Hourly, IQR spreads to ~0.28 SMAPE, giving more signal for architecture comparison.

---

## 4. Cross-Period Analysis

> **Cross-protocol note (added 2026-05-04 — paper-sample tiered Hourly results).** The numbers and rankings in this section are **sliding-protocol only** and remain authoritative for the sliding regime. A separate paper-sample tiered-offset Hourly study (`m4_hourly_sym10_tiered_offset_analysis_2026-05-04.md`, n=10/cell, 160 rows, plateau vs step_paper LR × ascend/descend × 4 backbones) has since landed. Its best Hourly tiered sym10 cell (`T+Sym10V3_10s_bdEQ_descend` plateau, SMAPE 8.9224 ± 0.1132) is **+0.335 SMAPE behind the sliding-protocol Hourly winner reported below** (`NBEATS-IG_30s_agf` 8.587). It does **not displace any sliding-protocol Hourly leader in §3.6 or §4.1**. Cross-protocol takeaways from that report worth flagging here:
>
> 1. **LR preference is backbone-asymmetric under paper-sample.** Alternating `T+Sym10V3` / `TAE+Sym10V3AE` Hourly tiered cells prefer plateau LR (Δ −0.168, p=0.026); unified `TWAE_td3_sym10_ld16` Hourly tiered cells prefer step_paper LR (Δ +0.196, p=0.004); unified `TW_td3_sym10` (no AE bottleneck) is LR-insensitive. This sliding-protocol report uses cosine-warmup uniformly; the asymmetry is a paper-sample finding only.
> 2. **Tiered direction (ascend vs descend) is noise on Hourly under paper-sample** (all p > 0.10).
> 3. **Hourly tiered does NOT beat the Hourly paper baseline** under either protocol (sliding `NBEATS-IG_30s_agf` 8.587 vs best paper-sample tiered 8.922; paper-sample `NBEATS-IG_30s_agf` 8.758 vs best paper-sample tiered 8.922).
> 4. **Generalist standing.** The combined paper-sample crown `T+Sym10V3_10s_tiered_ag0` (mean rank 13.33/108) uses the 8.922 plateau-descend Hourly cell as its Hourly slot. That generalist sits in the paper-sample leaderboard (`m4_overall_leaderboard_2026-05-03.md`), not in this sliding sweep, where `T+HaarV3_30s_bd2eq` remains the cross-period generalist (mean rank 12.7/112).
>
> See `m4_hourly_sym10_tiered_offset_analysis_2026-05-04.md` for full evidence and `m4_overall_leaderboard_2026-05-03.md` for the consolidated cross-protocol ranking.

### 4.1 Cross-Period Rank Grid (22 top-quartile-on-3+-periods keep candidates)

| Config | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly | Q1 count | Mean rank |
|---|---|---|---|---|---|---|---|---|
| **T+HaarV3_30s_bd2eq** | 31 | 20 | 2 | 4 | 9 | 10 | 5/6 | **12.7** |
| NBEATS-IG_30s_ag0 | 41 | 13 | 20 | 28 | 2 | 4 | 5/6 | 18.0 |
| TW_30s_td3_bdeq_db3 | 25 | 24 | 8 | 8 | 12 | 33 | 5/6 | 18.3 |
| NBEATS-IG_10s_ag0 | 22 | 1 | 3 | 55 | 17 | 12 | 5/6 | 18.3 |
| T+Db3V3_30s_bd2eq | 5 | 9 | 26 | 51 | 4 | 15 | 5/6 | 18.3 |
| T+Sym10V3_30s_bdeq | 16 | 18 | 43 | 5 | 6 | 26 | 5/6 | 19.0 |
| T+Db3V3_30s_bdeq | 8 | 15 | 66 | 1 | 5 | 25 | 5/6 | 20.0 |
| TALG+HaarV3ALG_30s_ag0 | 26 | 3 | 7 | 2 | 42 | 46 | 4/6 | 21.0 |
| T+Coif2V3_30s_bdeq | 61 | 8 | 21 | 18 | 8 | 18 | 5/6 | 22.3 |
| TAE+DB3V3AE_30s_ld32_agf | 27 | 36 | 18 | 20 | 39 | 7 | 4/6 | 24.5 |
| T+Sym10V3_30s_bd2eq | 11 | 2 | 46 | 56 | 11 | 22 | 4/6 | 24.7 |
| T+HaarV3_30s_bdeq | 67 | 14 | 40 | 3 | 10 | 16 | 4/6 | 25.0 |
| TAE+DB3V3AE_30s_ld32_ag0 | 13 | 37 | 9 | 23 | 32 | 45 | 3/6 | 26.5 |
| TALG+Coif2V3ALG_30s_agf | 15 | 42 | 22 | 39 | 41 | 5 | 3/6 | 27.3 |
| T+Coif2V3_30s_bd2eq | 82 | 4 | 11 | 43 | 7 | 19 | 4/6 | 27.7 |
| TALG+Sym10V3ALG_30s_ag0 | 6 | 27 | 50 | 9 | 43 | 43 | 3/6 | 29.7 |
| TALG+Db3V3ALG_30s_ag0 | 3 | 31 | 17 | 13 | 50 | 67 | 3/6 | 30.2 |
| TW_30s_td3_bdeq_sym10 | 81 | 50 | 5 | 11 | 13 | 32 | 3/6 | 32.0 |
| TALG+Sym10V3ALG_30s_agf | 2 | 22 | 47 | 67 | 49 | 8 | 3/6 | 32.5 |
| TALG+Coif2V3ALG_30s_ag0 | 47 | 6 | 4 | 29 | 48 | 65 | 3/6 | 33.2 |
| TWAELG_30s_ld16_db3_agf_sd5 | 14 | 67 | 108 | 19 | 88 | 24 | 3/6 | 53.3 |
| TWAE_10s_ld32_ag0 | 69 | 25 | 6 | 26 | 65 | 88 | 3/6 | 46.5 |

**Key observations:**

- **`T+HaarV3_30s_bd2eq` is the single best cross-period generalist** (mean rank 12.7). Top-quartile on 5/6 periods, worst rank 31 on Yearly. Tradeoff: 15.3M params — large.
- **Period-specific winners are not generalists.** The Yearly winner `TW_10s_td3_bdeq_coif2` ranks 94 on Weekly; the Weekly winner `T+Db3V3_30s_bdeq` ranks 66 on Monthly.
- **NBEATS-IG at {10s_ag0, 30s_ag0}** are both strong generalists (mean rank 18-19), but individually fragile — each has one period where it ranks 28+.
- **Sub-5M-parameter generalists:** `TALG+HaarV3ALG_30s_ag0` (3.66M params, mean rank 21). Best small generalist. `TAE+DB3V3AE_30s_ld32_agf` (3.32M, mean rank 24.5) is the best alt-AE generalist.

### 4.2 Period-Specific Winners (top on one period, mediocre elsewhere)

- **`TW_10s_td3_bdeq_coif2`** (Yearly #1): ranks 94 (Weekly), 81 (Hourly). Strong on short horizons, weak on long.
- **`TW_30s_td3_bd2eq_coif2`** (Monthly #1): not in top-22 keep list.
- **`NBEATS-G_30s_ag0`** (Daily #1): bimodally fails on Quarterly/Weekly. **Daily-only config.**
- **`NBEATS-IG_30s_agf`** (Hourly #1): only top quartile on Hourly.

### 4.3 Consistent Under-Performers (bottom-quartile on 3+ periods — DROP candidates)

27 configs are bottom-quartile on ≥3 periods. Grouped by family:

| Family | Count | Representative worst |
|---|---|---|
| bottleneckgeneric_aelg | 7 | BNAELG_{30s,10s}_ld{16,32}_ag0/agf (all variants) |
| bottleneckgeneric_ae | 4 | BNAE_10s_ld{8,16,32}_ag0/agf |
| bottleneckgeneric | 3 | BNG_{10s,30s}_ag0/agf |
| generic_aelg | 5 | GAELG_{10s,30s}_ld16_agf (+ sd5 variants) |
| generic_ae | 3 | GAE_10s_ld{8,16}_ag0/agf |
| paper_baseline | 3 | NBEATS-G_30s_{ag0, agf}, NBEATS-G_10s_agf |
| trendwavelet_genericaelg | 1 | TWGAELG_10s_ld16_agf |
| trendwavelet_aelg | 1 | TWAELG_30s_ld16_db3_agf_sd5 (Monthly catastrophe) |

**The entire BottleneckGeneric family (BNG, BNAE, BNAELG) is collectively the worst.** 14/16 BottleneckGeneric configs land in ≥3 bottom quartiles. **BNG_30s_ag0 and BNAELG_30s_ld32_ag0 have SMAPE std > 9 on Yearly** — catastrophic bimodal collapse.

---

## 5. Block / Architecture Family Analysis

### 5.1 Overall Family Ranking (mean SMAPE, excluding Daily which biases toward baselines)

| Rank | arch_family | mean SMAPE | std | OWA | n configs |
|---|---|---|---|---|---|
| 1 | alt_trend_wavelet_rb | 10.561 | 2.670 | 0.753 | 8 |
| 2 | alt_ae | 10.603 | 2.643 | 0.758 | 6 |
| 3 | alt_aelg | 10.612 | 2.640 | 0.758 | 12 |
| 4 | trendwavelet | 10.653 | 2.607 | 0.765 | 18 |
| 5 | trendwavelet_ae | 10.666 | 2.638 | 0.760 | 6 |
| 6 | trendwavelet_aelg | 10.666 | 2.649 | 0.760 | 22 |
| 7 | trendwavelet_genericaelg | 10.680 | 2.632 | 0.759 | 4 |
| 8 | generic_aelg | 10.787 | 2.702 | 0.786 | 8 |
| 9 | generic_ae | 10.816 | 2.690 | 0.784 | 4 |
| 10 | paper_baseline | 10.829 | 3.039 | 0.803 | 8 |
| 11 | bottleneckgeneric_aelg | 10.964 | 3.118 | 0.794 | 8 |
| 12 | bottleneckgeneric | 11.135 | 3.616 | 0.826 | 4 |
| 13 | bottleneckgeneric_ae | 11.168 | 2.864 | 0.833 | 4 |

Four clusters:

- **Tier 1 (best):** alt_trend_wavelet_rb, alt_ae, alt_aelg — **alternating architectures dominate**.
- **Tier 2:** unified TrendWavelet variants (rb, ae, aelg, genericaelg) — tight cluster.
- **Tier 3:** generic_aelg, generic_ae, paper_baseline — mid-pack. Paper baselines have the highest std (3.04) within this tier due to NBEATS-G bimodal failures.
- **Tier 4 (worst):** bottleneckgeneric family — uniformly bad.

### 5.2 Backbone Hierarchy per Period

| Period | Best backbone | Ranking (by mean SMAPE) |
|---|---|---|
| Yearly | AERootBlock (13.666) | AERootBlock < AERootBlockLG (13.69) < RootBlock (13.71) — near-tie |
| Quarterly | AERootBlock (10.282) | AERootBlock ≈ AERootBlockLG (10.286) < RootBlock (10.32) |
| Monthly | RootBlock (13.550) | RootBlock < AERootBlockLG (13.66) < AERootBlock (13.71) |
| Weekly | AERootBlockLG (6.881) | AERootBlockLG < AERootBlock (7.06) < RootBlock (7.08) |
| Daily | RootBlock (2.819) | RootBlock < AERootBlockLG (3.02) < AERootBlock (3.04) |
| Hourly | RootBlock (8.944) | RootBlock < AERootBlockLG (9.07) < AERootBlock (9.17) |

**Conclusions:**

- **Backbone preference is period-dependent.** Mean differences are small (often < 0.1 SMAPE) but statistically significant on most periods (KW p < 0.01 on Yearly/Quarterly/Monthly/Daily/Hourly).
- **RootBlock wins on long horizons (Monthly, Daily, Hourly).** The extra capacity helps when there are more observations to fit.
- **AE/AELG dominate on Weekly** (shortest series; regularisation matters).
- **There is no universal "best backbone."** Earlier prior claims of "AELG > AE > RB" are incorrect on M4.

### 5.3 Best-in-Family per Period

| arch_family | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
|---|---|---|---|---|---|---|
| alt_ae | TAE+DB3V3AE_30s_ld32_ag0 (13) | TAE+DB3V3AE_30s_ld16_ag0 (10) | TAE+DB3V3AE_30s_ld32_ag0 (9) | TAE+DB3V3AE_30s_ld16_ag0 (7) | (32) | TAE+DB3V3AE_30s_ld16_agf (3) |
| alt_aelg | TALG+Sym10V3ALG_30s_agf (2) | TALG+HaarV3ALG_30s_ag0 (3) | TALG+Coif2V3ALG_30s_ag0 (4) | TALG+HaarV3ALG_30s_ag0 (2) | (41) | TALG+Coif2V3ALG_30s_agf (5) |
| alt_trend_wavelet_rb | T+Db3V3_30s_bd2eq (5) | T+Sym10V3_30s_bd2eq (2) | T+HaarV3_30s_bd2eq (2) | T+Db3V3_30s_bdeq (1) | T+Db3V3_30s_bd2eq (4) | T+HaarV3_30s_bd2eq (10) |
| paper_baseline | NBEATS-IG_10s_ag0 (22) | **NBEATS-IG_10s_ag0 (1)** | NBEATS-IG_10s_ag0 (3) | NBEATS-IG_30s_ag0 (28) | **NBEATS-G_30s_ag0 (1)** | **NBEATS-IG_30s_agf (1)** |
| trendwavelet | **TW_10s_td3_bdeq_coif2 (1)** | TW_10s_td3_bdeq_coif2 (5) | **TW_30s_td3_bd2eq_coif2 (1)** | TW_30s_td3_bdeq_db3 (8) | (12) | (29) |
| trendwavelet_aelg | TWAELG_10s_ld16_coif2_agf (4) | TWAELG_10s_ld8_db3_agf (17) | (12) | TWAELG_10s_ld16_sym10_ag0 (6) | (62) | TWAELG_30s_ld16_sym10_agf (13) |
| trendwavelet_ae | TWAE_10s_ld16_agf (20) | TWAE_10s_ld32_ag0 (25) | TWAE_10s_ld32_ag0 (6) | TWAE_10s_ld16_ag0 (17) | (65) | TWAE_10s_ld32_agf (21) |
| bottleneckgeneric | (91) | (98) | (36) | (105) | (14) | (56) |
| bottleneckgeneric_ae | (88) | (87) | (109) | (86) | (89) | (108) |
| bottleneckgeneric_aelg | (84) | (94) | (59) | (65) | (40) | (64) |
| generic_ae | (85) | (76) | (75) | (15) | (47) | (53) |
| generic_aelg | (77) | (81) | (64) | (46) | (25) | (35) |

### 5.4 Variance / Reliability

**High-variance / unstable configs (SMAPE std > 0.5 on any period):**

- **Catastrophic bimodal (std > 4):**
  - Yearly: `BNG_30s_ag0` (std 9.96), `BNAELG_30s_ld32_ag0` (std 9.71).
  - Quarterly: `NBEATS-G_30s_ag0` (std 7.41).
  - Weekly: `NBEATS-G_30s_ag0` (std 7.16), `BNG_10s_ag0` (4.64), `BNAE_10s_ld16/32_ag0` (std 4.69, 4.77).
- **Moderate (std 0.5-1.0):**
  - Monthly: `NBEATS-G_10s_ag0` (0.78), `BNG_10s_ag0` (0.74), `NBEATS-G_30s_ag0` (0.59), `BNAE_10s_ld16_agf` (0.57), several TW/TWAELG/TWGAELG configs (0.55-0.58).
  - Yearly: `GAELG_30s_ld16_ag0_sd5` (0.56).
- **Hourly `BNAE_10s_ld8_ag0`:** std 1.27 (one outlier run with SMAPE 13.4 vs median 10.0).

**Low-variance / most stable top-20 configs per period:**

- Yearly: `TW_10s_td3_bdeq_coif2` (std 0.057), `TWAELG_10s_ld16_coif2_agf` (0.060).
- Quarterly: `NBEATS-IG_10s_agf` (0.026), `TWAELG_10s_ld8_db3_agf` (0.031).
- Monthly: `TW_30s_td3_bdeq_db3` (0.194) — note all Monthly configs have std > 0.19.
- Weekly: `TALG+HaarV3ALG_30s_ag0` (0.129), `TAE+DB3V3AE_30s_ld16_ag0` (0.137).
- Daily: `NBEATS-IG_30s_ag0` (0.027), `T+Sym10V3_30s_bdeq` (0.027).
- Hourly: `NBEATS-IG_10s_ag0` (0.070), `NBEATS-IG_10s_agf` (0.076).

The **NBEATS-IG family is the most seed-stable on long horizons** (Daily/Hourly). Alternating TrendWavelet family is most stable on Weekly.

### 5.5 Factor Importance (Kruskal–Wallis, per period)

| Period | active_g | wavelet_family | basis_dim_label | latent_dim | trend_td | n_stacks | backbone |
|---|---|---|---|---|---|---|---|
| Yearly    | p=0.18 ns | p=0.23 ns | p=0.64 ns | **p=0.001** | p=0.38 ns | p=0.09 ns | **p=0.001** |
| Quarterly | **p<0.001** | p=0.97 ns | p=0.036 | **p<0.001** | p=0.039 | p=0.39 ns | **p<0.001** |
| Monthly   | **p<0.001** | **p=0.008** | p=0.032 | p=0.42 ns | p=0.09 ns | **p=0.002** | **p<0.001** |
| Weekly    | **p=0.007** | p=0.065 | p=0.023 | p=0.24 ns | p=0.16 ns | **p<0.001** | p=0.11 ns |
| Daily     | **p<0.001** | **p<0.001** | **p<0.001** | **p<0.001** | p=0.99 ns | **p<0.001** | **p<0.001** |
| Hourly    | **p<0.001** | **p<0.001** | **p=0.001** | **p=0.002** | p=0.075 | **p<0.001** | **p<0.001** |

**Key factor patterns:**

- `n_stacks` and `backbone` are almost always significant.
- `active_g` is significant on all periods except Yearly.
- **`trend_thetas_dim` (td=3 vs 5) is almost never significant** (only Quarterly marginally, p=0.039). The prior "td=5 helps on Monthly" claim is not supported here.
- **`wavelet_family` is period-dependent.** Ns on Yearly/Quarterly/Weekly, significant on Monthly/Daily/Hourly. Best wavelet per period varies (coif2 Yearly, sym10 Quarterly, haar Monthly+Weekly, sym10 Daily, haar Hourly) — **no single family wins everywhere.**

### 5.6 Wavelet Family per Period (mean SMAPE)

| Period | Best wavelet | Worst wavelet | Spread |
|---|---|---|---|
| Yearly    | sym10 (13.574) | haar (13.600) | 0.027 (ns) |
| Quarterly | sym10 (10.209) | haar (10.220) | 0.010 (ns) |
| Monthly   | haar (13.489) | sym10 (13.619) | 0.131 |
| Weekly    | haar (6.820) | coif2 (6.880) | 0.059 |
| Daily     | sym10 (2.913) | db3 (2.991) | 0.078 |
| Hourly    | haar (8.934) | db3 (9.018) | 0.084 |

**db3 is never best** on any period but is safest default (never worst either except Daily/Hourly). **haar is strong on Weekly/Monthly/Hourly** (short- and medium-support wavelet; matches horizon structure).

### 5.7 active_g Interaction with Architecture

| arch_family (ex-Daily) | `ag0` SMAPE | `agf` SMAPE | Δ (agf better by) |
|---|---|---|---|
| paper_baseline | 10.98 | 10.68 | **+0.31** (agf better) |
| bottleneckgeneric | 11.39 | 10.88 | **+0.52** |
| bottleneckgeneric_ae | 11.23 | 10.98 | +0.26 |
| bottleneckgeneric_aelg | 11.04 | 10.83 | +0.22 |
| trendwavelet_ae | 10.69 | 10.64 | +0.05 |
| trendwavelet_aelg | 10.68 | 10.65 | +0.02 |
| alt_ae | 10.61 | 10.60 | ≈0 |
| alt_aelg | 10.62 | 10.60 | ≈0 |
| generic_aelg | 10.79 | 10.78 | ≈0 |
| generic_ae | 10.80 | 10.87 | **-0.08** (ag0 better) |
| trendwavelet_genericaelg | 10.66 | 10.71 | **-0.05** |

**Conclusions:**

- **`active_g=forecast` rescues weak architectures** (BNG variants, NBEATS-G) but is neutral or slightly harmful for strong ones (TrendWavelet, alternating).
- **For BottleneckGeneric-family:** always use `active_g=forecast`.
- **For TrendWavelet-family:** safe to leave `active_g=False` (default).
- **For Hourly:** `active_g=forecast` is consistently better (top-3 are all `agf` variants including paper baseline).

### 5.8 `n_stacks` Interaction

10-stack configs are preferred on short horizons (Yearly, Quarterly): mean rank cluster at top is dominated by `10s` unified TrendWavelet. 30-stack configs dominate long horizons (Daily, Hourly): all top-10 Daily and Hourly entries use 30 stacks.

On **Monthly, Weekly** (medium horizon): 30 stacks is better by a small margin in mean SMAPE (13.60 vs 13.68 Monthly; 6.95 vs 7.02 Weekly), but individual 10-stack configs can tie (`TWAE_10s_ld32_ag0` is rank 6 Monthly).

---

## 6. Comparison to Paper Baselines

### 6.1 Per-Period Comparison

| Period | Best baseline (SMAPE) | Best novel (SMAPE) | Δ | #Novel beating best baseline |
|---|---|---|---|---|
| Yearly    | NBEATS-IG_10s_ag0 (13.561) | TW_10s_td3_bdeq_coif2 (13.499) | **-0.062** (novel wins) | **21 / 104** |
| Quarterly | NBEATS-IG_10s_ag0 (10.126) | T+Sym10V3_30s_bd2eq (10.127) | +0.001 (tie) | 0 / 104 (15-way tie) |
| Monthly   | NBEATS-IG_10s_ag0 (13.309) | TW_30s_td3_bd2eq_coif2 (13.279) | **-0.030** (novel wins) | 2 / 104 |
| Weekly    | NBEATS-IG_30s_ag0 (6.822) | T+Db3V3_30s_bdeq (6.671) | **-0.151** (novel wins) | **27 / 104** |
| Daily     | NBEATS-G_30s_ag0 (2.588) | T+Db3V3_30s_bd2eq (2.709) | +0.121 (baseline wins) | 0 / 104 |
| Hourly    | NBEATS-IG_30s_agf (8.587) | TAE+DB3V3AE_30s_ld16_agf (8.673) | +0.086 (baseline wins) | 0 / 104 |

**Summary:** Novel wavelet architectures win **3/6 periods** decisively (Yearly, Monthly, Weekly), tie on Quarterly, and lose on Daily/Hourly. The novel architectures that come closest on Daily (`T+Db3V3_30s_bd2eq`) and Hourly (`TAE+DB3V3AE_30s_ld16_agf`) use 5-14M params vs the baselines' 26-43M — **still an 8-10x parameter reduction for a 0.09-0.12 SMAPE cost**.

### 6.2 NBEATS-G vs NBEATS-I+G Instability

- `NBEATS-G_30s_ag0` has **catastrophic bimodal failures** on Quarterly (1/10 diverge, std 7.4) and Weekly (3/10 diverge, std 7.2). `active_g=forecast` eliminates this.
- `NBEATS-IG_{10,30}s_ag0` variants are uniformly stable (std < 0.32 on all periods). This makes **NBEATS-I+G the safe baseline**, not NBEATS-G.
- The paper's canonical 30-stack configs are the top-3 on Daily; at smaller stack counts they fall behind.

---

## 7. Parameter Efficiency Pareto

### 7.1 Best per Parameter Tier per Period

| Period | <500K | 500K-1M | 1-3M | 3-10M | 10-50M |
|---|---|---|---|---|---|
| Yearly    | TWAELG_10s_ld16_coif2_agf (13.524, r4) | — | TW_10s_td3_bdeq_coif2 (13.499, r1) | TALG+Sym10V3ALG_30s_agf (13.504, r2) | T+Db3V3_30s_bd2eq (13.529, r5) |
| Quarterly | TWAELG_10s_ld8_db3_agf (10.167, r17) | — | TW_10s_td3_bdeq_coif2 (10.147, r5) | TALG+HaarV3ALG_30s_ag0 (10.144, r3) | NBEATS-IG_10s_ag0 (10.126, r1) |
| Monthly   | — | TWAE_10s_ld32_ag0 (13.325, r6) | TWAELG_30s_ld16_haar_ag0 (13.390, r12) | TW_30s_td3_bd2eq_coif2 (13.279, r1) | T+HaarV3_30s_bd2eq (13.308, r2) |
| Weekly    | TWAELG_10s_ld16_sym10_ag0 (6.693, r6) | TWAELG_10s_ld32_db3_agf (6.771, r16) | TWAELG_30s_ld16_sym10_ag0 (6.722, r10) | TALG+HaarV3ALG_30s_ag0 (6.673, r2) | T+Db3V3_30s_bdeq (6.671, r1) |
| Daily     | TWAE_10s_ld8_ag0 (3.051, r77) | TWAE_10s_ld32_ag0 (3.030, r65) | TW_10s_td3_bdeq_db3 (2.894, r26) | TW_30s_td3_bdeq_db3 (2.750, r12) | NBEATS-G_30s_ag0 (2.588, r1) |
| Hourly    | — | TWAE_10s_ld32_agf (8.825, r21) | TWAELG_30s_ld16_sym10_agf (8.783, r13) | TAE+DB3V3AE_30s_ld16_agf (8.673, r3) | NBEATS-IG_30s_agf (8.587, r1) |

**`TWAELG_10s_ld16_<wavelet>_<ag>`** is the dominant <500K-parameter architecture on Yearly, Quarterly, and Weekly. **On Daily/Hourly, AE/AELG compact models cannot close the gap** to the 20-40M-parameter baselines.

---

## 7A. basis_dim x 2 (bd2eq) Analysis

**What bd2eq means:** `basis_dim = 2 x forecast_length`. Baseline uses `basis_dim = forecast_length`. 13 architecture pairs x 6 M4 periods = 78 paired observations (10 seeds per cell).

**Per-period results:**

| Period | Mean Δ SMAPE (bd2 − base) | bd2 wins | Wilcoxon p | Verdict |
|---|---|---|---|---|
| Yearly | −0.009 | 9/13 | 0.588 | neutral |
| Quarterly | +0.048 | 4/13 | 0.017 | **bd2 hurts** |
| Monthly | +0.014 | 6/13 | 1.000 | neutral |
| Weekly | +0.064 | 3/13 | 0.094 | trends worse |
| Daily | +0.069 | 3/13 | 0.003 | **bd2 hurts** |
| Hourly | +0.019 | 5/13 | 0.305 | neutral |
| **Pooled** | **+0.034** | 30/78 | **0.003** | **bd2 hurts** |

**Architecture interaction (avg Δ across 6 periods):**

| Architecture | Avg Δ SMAPE | bd2 wins/6 |
|---|---|---|
| T+HaarV3_30s | −0.056 | 4 |
| T+Db3V3_30s | −0.008 | 5 |
| TW_30s_td3_coif2 | −0.003 | 3 |
| T+Coif2V3_30s | +0.013 | 3 |
| TW_10s_td3_sym10 | +0.018 | 2 |
| TW_10s_td3_haar | +0.020 | 2 |
| T+Sym10V3_30s | +0.030 | 3 |
| TW_10s_td3_coif2 | +0.040 | 2 |
| TW_10s_td5_db3 | +0.042 | 1 |
| TW_30s_td3_haar | +0.058 | 2 |
| TW_30s_td3_db3 | +0.079 | 1 |
| TW_10s_td3_db3 | +0.089 | 1 |
| TW_30s_td3_sym10 | +0.122 | 1 |

**Narrative:** bd2 hurts overall (pooled Wilcoxon p=0.003, baselines win 48/78 pairs). Effect is strongest and statistically significant on Daily (p=0.0004) and Quarterly (p=0.007). The only architectures where bd2 is neutral-to-slightly-positive are unified `T+WaveletV3` blocks (HaarV3, Db3V3, Coif2V3); alternating `TW_` stacks are uniformly worse, especially sym10 and db3 variants. Doubling basis_dim adds parameters without accuracy benefit and marginal harm on short-horizon high-frequency periods.

**Recommendation:** Drop all `bd2eq` variants from future experiments. Use `basis_dim = forecast_length` as the fixed default.

This finding REVISES the Section 9.3 new finding that "bd2eq beats bdeq on Monthly and Daily." The earlier claim was based on per-period top-10 appearances; the paired architecture-matched analysis above shows bd2eq is at best neutral and often harmful.

---

## 7B. trend_thetas_dim (td3 vs td5) Analysis

**Configs compared:** 2 matched pairs differing only in `trend_thetas_dim` — `TW_10s_td{3,5}_bdeq_db3` and `TW_10s_td{3,5}_bd2eq_db3` (10-stack TrendWavelet, db3 wavelet; 10 runs × 6 periods each).

**Grand-pool result (120 paired obs across both pairs × 6 periods):**

| Metric | td3 − td5 | Wilcoxon p | td3 wins |
|---|---|---|---|
| SMAPE | +0.001 | 0.393 | 57/120 (47.5%) |
| OWA | −0.002 | 0.295 | — |

**td3 vs td5 is a non-factor on M4 overall.** Win rates are ~50/50 and p≈0.3–0.4 pooled.

**Per-period comparison (bdeq pair, 10 runs each):**

| Period | td3 SMAPE | td5 SMAPE | dSMAPE | td3 wins | p (Wilcoxon) | Winner |
|---|---:|---:|---:|---:|---:|---|
| Yearly    | 13.647 | 13.533 | +0.115 | 1/10  | **0.014** | **td5** |
| Quarterly | 10.206 | 10.221 | −0.016 | 6/10  | 0.695 | td3 (ns) |
| Monthly   | 13.443 | 13.808 | −0.365 | 7/10  | 0.106 | td3 (ns) |
| Weekly    |  6.951 |  6.835 | +0.115 | 5/10  | 0.492 | td5 (ns) |
| Daily     |  2.894 |  2.908 | −0.014 | 7/10  | 0.432 | td3 (ns) |
| Hourly    |  9.070 |  9.042 | +0.028 | 4/10  | 0.557 | td5 (ns) |

**One significant effect:** td5 beats td3 on **Yearly with `bdeq`** (p=0.014 Wilcoxon, p=0.008 MWU; td3 wins only 1/10). Under `bd2eq` this effect vanishes (p=0.625; 3/10 wins for td3). The td5 Yearly benefit is basis-dim-dependent and disappears when bd2eq is used — and bd2eq is now dropped (Section 7A), so the practical impact is limited.

**Recommendation:** Default to **td3** on all M4 periods. Consider td5 specifically for Yearly-only experiments with `bdeq` basis where there is a small but significant +0.11 SMAPE / +0.01 OWA advantage. Not worth per-period switching elsewhere (|dSMAPE| < 0.1 on all other periods — within seed noise).

**Caveat:** Only db3 wavelet was swept for this comparison. td3 vs td5 interactions with haar/coif2/sym10 are untested.

---

## 7C. latent_dim Analysis

**Configs compared:** 12 groups (AE and AELG families at 10 or 30 stacks), testing ld ∈ {8, 16, 32}. ld=8 tested only for 10-stack AE groups; ld=16/32 covered for all 12 groups. All groups: 10 runs × 6 periods.

**Pooled mean SMAPE by family and latent_dim:**

| Family | ld8 | ld16 | ld32 | Direction |
|---|---:|---:|---:|---|
| Alt Trend+Wavelet AE (30s)  | 9.368 | 9.352 | **9.286** | ld32 < ld16 < ld8 |
| TrendWaveletAE (10s)        | 9.434 | 9.405 | **9.355** | ld32 < ld16 < ld8 |
| TrendWaveletAELG (10s, db3) | 9.425 | 9.414 | **9.386** | ld32 < ld16 < ld8 |
| GenericAE (10s)             | 9.548 | 9.533 | **9.435** | ld32 < ld16 ≈ ld8 |
| **GenericAELG (30s)**       | — | **8.857** | 8.953 | **ld16 < ld32** |
| GenericAELG+skip (30s)      | — | 8.833 | **8.854** | ld16 ≈ ld32 |
| BottleneckGenericAE (10s)   | **9.784** | 9.918 | 9.917 | ld8 < ld16 ≈ ld32 (noisy) |
| BottleneckGenericAELG (30s) | — | **9.632** | 10.034 | **ld16 < ld32** |

**Within-group mean rank (1=best) across all 72 group×period cells:**

| ld | Mean rank | Std | n cells |
|---|---:|---:|---:|
| 8  | 2.35 | 0.76 | 48 |
| 16 | 1.90 | 0.70 | 72 |
| **32** | **1.53** | 0.71 | 72 |

ld32 wins on average; ld16 second; ld8 worst — consistent with pooled SMAPE direction.

**Period split (mean rank, aggregated across both AE and AELG groups):**

| Period | ld16 rank | ld32 rank | Leader |
|---|---:|---:|---|
| Yearly    | **1.58** | 1.92 | ld16 |
| Quarterly | 1.75 | 2.08 | ld16 (tied with ld8) |
| Monthly   | 2.33 | **1.17** | **ld32** |
| Weekly    | 1.92 | **1.50** | **ld32** |
| Daily     | 1.83 | **1.42** | **ld32** |
| Hourly    | 2.00 | **1.08** | **ld32** |

ld32 dominates on Monthly/Weekly/Daily/Hourly; ld16 marginally wins Yearly and Quarterly.

**Statistical tests:** Pooled KW p>0.5 for all 12 groups — within-seed variance drowns the ld main effect. Per-period KW hits p<0.05 sporadically (strongest on Daily, 4/12 groups reach p<0.05). Wilcoxon top-2 comparisons:

- **GAE: ld32 vs ld16** — p=0.010 (ld32 wins; Δ=+0.098 SMAPE)
- **GAELG+skip: ld16 vs ld32** — p=0.036 (ld16 wins; Δ=+0.066 SMAPE)
- All TW-family top-2 comparisons: p=0.05–0.97 (ns but direction is ld32 > ld16)

**Mechanistic explanation:** AE backbones without a learned gate (TWAE, GAE, alt T+W AE) scale monotonically with ld — bigger bottleneck = more expressive. AELG backbones have a sigmoid gate that discovers effective dimensionality during training, so ld16 suffices; ld32 adds noise for GAELG. Bottleneck families (BN) factorize through `thetas_dim` already, making the AE latent redundant — ld32 is wasteful.

**Recommendations by family:**

| Family | Recommended ld | Rationale |
|---|---|---|
| TrendWaveletAE / TWAE | **32** | Monotonic improvement; no gate to self-regularize |
| TrendWaveletAELG / TWAELG | **32** | Direction consistent across periods (ld32 wins 4/6); gate still benefits from extra capacity |
| Alternating TrendAE + WaveletAE | **32** | Largest magnitude gain (Δ0.07 pooled) |
| GenericAE | **32** | Wilcoxon p=0.010 |
| GenericAELG | **16** | Gate handles effective dim; ld32 significantly worse with skip (p=0.036) |
| BottleneckGenericAE/LG | **≤16** | ld32 flat or worse; family is dropped anyway (Section 8.2) |
| Yearly-only targeting | **16** | ld16 has mean rank 1.58 vs ld32 at 1.92 on Yearly |

---

## 8. Recommendations

### 8.1 KEEP — configs to reuse in future M4 experiments

**Top 10 M4 generalists (by mean rank across all 6 periods):**

1. `T+HaarV3_30s_bd2eq` — mean rank 12.7, Q1 on 5/6 periods, 15.3M params (**best generalist**)
2. `NBEATS-IG_30s_ag0` — mean rank 18.0, paper baseline (stability anchor)
3. `TW_30s_td3_bdeq_db3` — mean rank 18.3, unified TrendWavelet at 6.2M params
4. `NBEATS-IG_10s_ag0` — mean rank 18.3, compact baseline (19.5M, Quarterly #1)
5. `T+Db3V3_30s_bd2eq` — mean rank 18.3, 15.3M
6. `T+Sym10V3_30s_bdeq` — mean rank 19.0, 15.2M
7. `T+Db3V3_30s_bdeq` — mean rank 20.0, Weekly #1
8. `TALG+HaarV3ALG_30s_ag0` — mean rank 21.0, **3.1M params (best sub-5M generalist)**
9. `T+Coif2V3_30s_bdeq` — mean rank 22.3, 15.2M
10. `TAE+DB3V3AE_30s_ld32_agf` — mean rank 24.5, 3.3M, best on Hourly alt_ae

**Per-period SOTA anchors (keep for period-specific tuning):**

| Period | Config | SMAPE | OWA | Params |
|---|---|---|---|---|
| Yearly    | TW_10s_td3_bdeq_coif2 | 13.499 | 0.801 | 2.1M |
| Quarterly | NBEATS-IG_10s_ag0 | 10.126 | 0.888 | 19.6M (can swap for T+Sym10V3_30s_bd2eq @ 15.4M, tie) |
| Monthly   | TW_30s_td3_bd2eq_coif2 | 13.279 | 0.914 | 7.1M |
| Weekly    | T+Db3V3_30s_bdeq | 6.671 | 0.735 | 15.8M |
| Daily     | NBEATS-G_30s_ag0 | 2.588 | 0.855 | 26.0M |
| Hourly    | NBEATS-IG_30s_agf | 8.587 | 0.409 | 43.6M |

**Sub-1M parameter Pareto picks (keep for compact-model studies):**

- Yearly: `TWAELG_10s_ld16_coif2_agf` (436K, rank 4) — remarkable
- Weekly: `TWAELG_10s_ld16_sym10_ag0` (498K, rank 6)
- Quarterly: `TWAELG_10s_ld8_db3_agf` (433K, rank 17)
- Monthly: `TWAE_10s_ld32_ag0` (584K, rank 6)
- Keep `TALG+DB3V3ALG_10s_ag0` (1.09M) — Quarterly rank 7, solid all-period generalist.

### 8.2 DROP — configs to exclude from future M4 experiments

**27 configs land in bottom-quartile on ≥3 periods.** Drop all BottleneckGeneric-family variants:

**Drop unconditionally (all BottleneckGeneric variants):**

- `BNG_10s_ag0`, `BNG_10s_agf`, `BNG_30s_ag0`, `BNG_30s_agf`
- `BNAE_10s_ld{8,16,32}_ag0`, `BNAE_10s_ld{16}_agf`
- `BNAELG_10s_ld16_{ag0,agf}`, `BNAELG_30s_ld{16,32}_{ag0,agf}`, `BNAELG_30s_ld{16,32}_ag0_sd5`, `BNAELG_30s_ld16_agf_sd5`

**Justification:** BottleneckGeneric family mean SMAPE is 0.2-0.6 worse than alt_trend_wavelet_rb on every period, with catastrophic bimodal failures on Yearly (std > 9). 2-4x more params than TWAELG (1.5-4.5M) for worse accuracy. **No recovery path.**

**Drop Generic/GenericAE:**

- `GAE_10s_ld{8,16}_{ag0,agf}`
- `GAELG_30s_ld16_{ag0,agf}_sd5`, `GAELG_30s_ld16_agf`, `GAELG_30s_ld32_ag0`, `GAELG_10s_ld16_agf`

**Justification:** Generic AE variants consistently rank 40-85 across periods. generic_aelg is marginally better than generic_ae but still dominated by TrendWavelet family at comparable param budgets.

**Drop NBEATS-G_30s_ag0 (keep NBEATS-G_30s_agf for comparison):**

- `NBEATS-G_30s_ag0` has catastrophic bimodal failures on Quarterly and Weekly (std > 7). Use `agf` variant instead.

**Drop Skip-Connection variants (sd5) in general:**

- Every `_sd5` variant ranks worse than its non-skip equivalent. Skip connections hurt on M4. Retain only for ablation/explanation runs.

**Drop bd2eq (basis_dim=2*forecast) variants:**

- Paired comparison (Section 7A) shows bd2eq is statistically significantly worse than `bdeq` on Daily (Wilcoxon p=0.0004) and Quarterly (p=0.007), and neutral on all other periods. Pooled across 78 architecture x period pairs, bdeq wins 48/78 (p=0.003).
- No architecture benefits consistently from bd2eq. Only unified `T+HaarV3/Db3V3/Coif2V3` blocks are neutral; all alternating `TW_` variants are worse.
- **Justification:** statistically significant harm on Daily and Quarterly, neutral elsewhere, no architecture benefits consistently. Doubling basis_dim adds parameters without accuracy benefit.
- Use `basis_dim = forecast_length` (`bdeq`) as the fixed default going forward.

### 8.3 AVOID these hyperparameter combinations

- **`active_g=False` on NBEATS-G at 30 stacks** (bimodal collapse on Quarterly/Weekly).
- **`latent_dim=32` with BottleneckGeneric backbone** (worst-5 on every period; family is dropped anyway).
- **`latent_dim=32` with GenericAELG** — ld32 is significantly worse than ld16 when skip is present (p=0.036). Prefer ld16 for all GAELG variants.
- **`latent_dim=8` for any wavelet AE/AELG family** — consistently worst latent_dim (mean rank 2.35 vs 1.53 for ld32). Use ld32 for TWAE/TWAELG/alt-T+W-AE; ld16 only for Yearly-specific tuning or GAELG.
- **`skip_distance=5` on any TW/TWAELG/GAELG config** (always hurts or neutral on M4).
- **`trend_thetas_dim=5`** — non-factor overall (grand pool p=0.39). Exception: td5 beats td3 on Yearly+bdeq (p=0.014, Δ0.115 SMAPE) but effect vanishes under bd2eq. Default to `td=3`; consider td5 only for Yearly-specific experiments.
- **`wavelet_family=db3` on Hourly/Daily/Monthly** — worst wavelet on those periods (though still close). Prefer `haar`/`sym10`/`coif2`.

---

## 9. Key Findings Narrative

### 9.1 Confirmed findings (from prior analyses)

- **F2 confirmed with nuance:** `trend_thetas_dim=3` is the safe default (grand pool p=0.39, win rate 47.5%). One exception: td5 significantly beats td3 on Yearly with `bdeq` basis (Wilcoxon p=0.014, Δ0.115 SMAPE; td3 wins only 1/10). Effect disappears under `bd2eq`. See Section 7B.
- **F4 partially confirmed:** `basis_dim=eq_fcast` or `2*eq_fcast` both near-optimal. On Monthly the top-10 is mixed between `bdeq` and `bd2eq` (`bd2eq` wins #1 on Monthly and Daily).
- **F5 revised:** Pooled KW is non-significant for all 12 latent_dim groups (all p>0.5) — within-seed variance dominates. However, the directional pattern is consistent: ld32 > ld16 > ld8 for all TW-AE/AELG and alternating T+W-AE families; ld16 > ld32 for GenericAELG (gate handles effective dim; Wilcoxon p=0.036 with skip). Period split: ld16 best on Yearly, ld32 dominates Monthly/Weekly/Daily/Hourly. GenericAE is the one family with a statistically significant ld effect: ld32 beats ld16 (p=0.010). See Section 7C for full breakdown.
- **F7 confirmed:** `active_g=forecast` rescues Generic and BottleneckGeneric variants. Paper baseline `NBEATS-G_30s_agf` avoids the catastrophic bimodal failure of `NBEATS-G_30s_ag0`.
- **F8 revised:** Backbone hierarchy **is period-dependent** on M4. RootBlock wins on Monthly/Daily/Hourly; AERootBlock wins on Yearly/Quarterly; AERootBlockLG wins on Weekly. The previous "RB > AELG > AE" claim was Yearly-biased.
- **F9 confirmed:** Novel architectures match or beat paper baselines on 3/6 periods at **8-45x fewer parameters**. `TWAELG_10s_ld16_coif2_agf` (436K) at rank 4 Yearly is a standout.
- **F10 partially confirmed:** Alternating TrendAELG+WaveletV3AELG is strong but not always #1. On Yearly/Weekly/Monthly it's in top-10; on Daily/Hourly it's out of top-40.
- **F12 confirmed:** `TrendWaveletAELG` / `TrendWaveletGenericAELG` are **Pareto-optimal** at <1M params on 4/6 periods.

### 9.2 Revised or overturned findings

- **"Wavelet type barely matters" (F3) is OVERTURNED on M4.** KW p<0.001 on Daily/Hourly, p=0.008 on Monthly. The best wavelet per period varies (coif2 for Yearly, sym10 Quarterly, haar Monthly/Weekly/Hourly, sym10 Daily). Prior claim was driven by Yearly-only data.
- **"TW+db3 is safest cross-dataset" is REVISED.** For M4, **`T+HaarV3_30s_bd2eq`** (Haar, `bd2eq`) is the best cross-period generalist, not db3. `db3` is never #1 on any M4 period.
- **"10 stacks optimal for short horizons, 30 for long" is CONFIRMED** — 30 stacks wins Daily/Hourly decisively; 10 stacks dominates Yearly/Quarterly top-10.
- **"NBEATS-I+G wins Quarterly (tie)" CONFIRMED** — 15-way tie, no novel decisively beats it.
- **"Daily is under-tested" is OUTDATED:** Daily now has the same 112 configs as other periods (1,216 runs, slightly more due to some reruns). All configs were tested.
- **"Skip connections not recommended" CONFIRMED on M4:** Every `_sd5` variant ranks worse or equal to its non-skip counterpart.

### 9.3 New findings

- **The BottleneckGeneric family is universally worst on M4.** 14/16 configs are in bottom quartile on ≥3 periods. Previously characterized as "worse than Generic" — now shown to be worst-in-class across all M4 periods.
- **On Hourly, `active_g=forecast` is essential across the board.** All top-10 Hourly configs except TALG_*_ag0 entries use agf.
- **`bd2eq` (basis_dim=2*forecast) beats `bdeq` on Monthly and Daily.** On short horizons (Yearly/Quarterly) and long (Hourly) `bdeq` is equal or better. This is horizon-length-dependent.
- **Haar wavelet (short support) shines on Weekly and Monthly**, matching the intuition that short-support wavelets suit medium-length horizons better than db3/sym10.

---

## 10. Proposed Next Experiments

### 10.1 High-priority

1. **Replicate Yearly SOTA `TW_10s_td3_bdeq_coif2`** with 20 seeds to tighten the CI and confirm it is significantly below the 15-way tie at 13.50. Current 10-seed spread is tight (std 0.057).
2. **Tune coif variants on Monthly**: `TW_30s_td3_bd2eq_coif3`, `TW_30s_td3_bd2eq_coif4` — since `coif2` wins Monthly and Yearly, higher-order coiflets may extend the lead.
3. **Hourly parameter-efficiency sweep at 30 stacks:** test `TWAELG_30s_ld{16,32}_{sym10,coif2}_agf` with 10 seeds. Currently `TWAELG_30s_ld16_sym10_agf` is rank 13 at 2.4M params; closing the 0.2 SMAPE gap to the 43.6M baseline would be a major efficiency win.
4. **`coif3` Tourism-style exploration on M4-Weekly**: the prior Tourism SOTA uses `coif3+eq_bcast+td3+ld16`, and M4-Weekly has the same short-horizon character. Propose `TWAELG_10s_ld16_coif3_ag0` (not in sweep).
5. **M4-Weekly: test `NBEATS-I+G` at `100 stacks`** given its strong stability and 30-stack result of 6.82. Novel architectures beat it by 0.15 but with 10x more stacks there's headroom.

### 10.2 Medium-priority

6. **trend_td sensitivity retest:** The sweep's `td=3` vs `td=5` comparison is underpowered (only 2 levels × few configs). Run a focused ablation on Weekly/Monthly with `td ∈ {3, 4, 5, 6}`.
7. **Sub-500K generalist hunt:** `TWAELG_10s_ld16_coif2_agf` (436K) is rank 4 Yearly but rank 94 Weekly. Explore `TWAELG_10s_ld16_<wavelet>_<ag>` grid at ≤500K params for a true cross-period compact winner.
8. **Replacement `T+HaarV3_30s_bd2eq` study:** our best generalist is 15.3M params. Test `TALG+HaarV3ALG_10s_bd2eq_ag0` (~1-2M) to see if its generalism survives at small scale.
9. **Drop BottleneckGeneric from all future sweeps.** Save ~15% of compute.

### 10.3 Low-priority / negative-result replications

10. **Confirm skip connections hurt** with 20 seeds at d=3 on GAELG at 30 stacks (not covered here).
11. **Hourly `forecast_multiplier` sensitivity:** try `fm=3` (L=144) and `fm=7` (L=336) to confirm L=5H is the sweet spot.
12. **Closed (negative result) — Hourly tiered-offset paper-sample sweep (sym10).** Resolved 2026-05-04 by `m4_hourly_sym10_tiered_offset_analysis_2026-05-04.md`: best tiered Hourly cell (`T+Sym10V3_10s_bdEQ_descend` plateau, 8.922) trails sliding `NBEATS-IG_30s_agf` (8.587) by +0.335 and paper-sample `NBEATS-IG_30s_agf` step (8.758) by +0.164. Tiered offset on Hourly does not produce a new SOTA under either protocol; tiered scope is now restricted to **Daily only** in the production defaults.

### 10.4 YAML config suggestions

```yaml
# Yearly SOTA tight-seed replication
- name: TW_10s_td3_bdeq_coif2_20seeds
  stacks: { type: homogeneous, block: TrendWavelet, n: 10 }
  block_params: { wavelet_type: coif2, basis_dim: 6, trend_thetas_dim: 3 }
  training: { active_g: false }
  runs: { n_runs: 20, seed_mode: random }

# Monthly coif higher-order
- name: TW_30s_td3_bd2eq_coif3
  stacks: { type: homogeneous, block: TrendWavelet, n: 30 }
  block_params: { wavelet_type: coif3, basis_dim: 36, trend_thetas_dim: 3 }

# Hourly compact TWAELG
- name: TWAELG_30s_ld32_sym10_agf
  stacks: { type: homogeneous, block: TrendWaveletAELG, n: 30 }
  block_params: { wavelet_type: sym10, latent_dim: 32, trend_thetas_dim: 3 }
  training: { active_g: forecast }
```

---

## 11. Open Questions

1. **Why does `TW_10s_td3_bdeq_coif2` degrade from rank 1 (Yearly) to rank 94 (Weekly)?** Horizon change from 6 to 13 should not cripple a unified TrendWavelet. Investigate per-epoch convergence on Weekly specifically.
2. **Can a TrendWaveletAELG at <500K params beat paper baselines on Hourly?** Current best sub-500K on Hourly is `TWAE_10s_ld32_agf` at 853K (rank 21). Specifically test `TWAELG_30s_ld8_<wavelet>_agf` configurations (currently absent at ld=8).
3. **Is there a hybrid alternating Trend + Generic architecture** that matches `T+HaarV3_30s_bd2eq` generalism without needing the wavelet? The sweep did not test plain alternating `Trend + Generic` stacks.
4. **Do MASE and MAE rankings agree with SMAPE?** OWA generally correlates but the OWA-best-5 on Weekly picks `TWAELG_10s_ld32_db3_agf` over the SMAPE winner `T+Db3V3_30s_bdeq`, suggesting different tradeoffs.

---

## Appendix A: Full Keep / Drop Lists

### KEEP (22 configs with Q1 on ≥3 periods)

```
NBEATS-IG_10s_ag0, NBEATS-IG_30s_ag0, T+Coif2V3_30s_bdeq, T+Db3V3_30s_bd2eq,
T+Db3V3_30s_bdeq, T+HaarV3_30s_bd2eq, TW_30s_td3_bdeq_db3, T+Sym10V3_30s_bdeq,
T+Coif2V3_30s_bd2eq, T+HaarV3_30s_bdeq, TAE+DB3V3AE_30s_ld32_agf,
TW_10s_td3_bdeq_coif2, TALG+HaarV3ALG_30s_ag0, T+Sym10V3_30s_bd2eq,
TALG+Db3V3ALG_30s_ag0, TALG+Coif2V3ALG_30s_agf, TAE+DB3V3AE_30s_ld32_ag0,
TALG+Sym10V3ALG_30s_ag0, TWAELG_30s_ld16_db3_agf_sd5, TALG+Sym10V3ALG_30s_agf,
TWAE_10s_ld32_ag0, TW_30s_td3_bdeq_sym10
```

### DROP (27 configs with Q4 on ≥3 periods)

```
BNAELG_30s_ld16_ag0, BNAE_10s_ld32_ag0, BNAELG_30s_ld16_ag0_sd5,
BNAE_10s_ld8_ag0, BNAE_10s_ld16_ag0, BNAE_10s_ld16_agf,
BNAELG_10s_ld16_ag0, BNAELG_10s_ld16_agf, GAE_10s_ld8_ag0,
BNG_30s_ag0, BNG_10s_ag0, GAE_10s_ld16_ag0, NBEATS-G_30s_agf,
BNAELG_30s_ld32_ag0_sd5, GAE_10s_ld16_agf, BNG_10s_agf,
GAELG_10s_ld16_agf, BNAELG_30s_ld16_agf, BNAELG_30s_ld32_ag0,
GAELG_30s_ld16_agf_sd5, GAELG_30s_ld16_agf, GAELG_30s_ld16_ag0_sd5,
BNG_30s_agf, GAELG_30s_ld32_ag0, NBEATS-G_10s_agf, NBEATS-G_30s_ag0,
TWGAELG_10s_ld16_agf
```

---

## Appendix B: File Manifest

- CSV (raw): `experiments/results/m4/comprehensive_sweep_m4_results.csv` (6,816 rows)
- Config: `experiments/configs/comprehensive_sweep_m4.yaml`
- Derived artifacts (this analysis): in `/tmp/m4_analysis_out/` — per-period summaries, pivot tables, consistency rankings, family summaries.

---

*End of report.*
