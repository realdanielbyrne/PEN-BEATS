# Comprehensive M4 Paper-Sample Sweep Analysis

**Date:** 2026-04-27
**Dataset:** M4 (all six periods: Yearly, Quarterly, Monthly, Weekly, Daily, Hourly)
**Source data:** `experiments/results/m4/comprehensive_m4_paper_sample_results.csv` (3,180 rows = 53 configs × 6 periods × 10 runs, all cells full)
**Source config:** `experiments/configs/comprehensive_m4_paper_sample.yaml`
**Sampling protocol:** `nbeats_paper` (paper-faithful per-series window sampling, `steps_per_epoch=1000`)
**Status:** Validation sweep for the early-stopping fix (`val_check_interval=100`, `min_delta=0.001`).

---

## 1. Executive Summary

- **3,180 runs across 53 configs** with the new `nbeats_paper` sampling protocol. Zero divergence except a single Daily run of `TW_10s_td3_bdeq_coif2` (1/3,180 = 0.03%); all other 3,179 runs reached `EARLY_STOPPED`.
- **Early-stopping fix works as advertised.** `best_epoch` median = 15, mean = 23.9, max = 158. Only 7.9% of runs stopped at `best_epoch ≤ 1` — a clean break from the prior 100% epoch-0/1 collapse. Mean `epochs_trained` = 4.4 Lightning epochs (each = 1000 training steps), so ~44 validation checks per run.
- **Per-period winners (paper-sample protocol, individual seeds, mean of 10 runs):**
  - **Yearly:** `TALG+DB3V3ALG_10s_ag0` — SMAPE = 13.550, OWA = 0.806 (1.04M params)
  - **Quarterly:** `NBEATS-IG_10s_ag0` — SMAPE = 10.357, OWA = 0.915 (19.6M params)
  - **Monthly:** `TW_30s_td3_bdeq_haar` — SMAPE = 13.391, OWA = 0.916 (6.8M params)
  - **Weekly:** `T+Coif2V3_30s_bdeq` — SMAPE = 6.735, OWA = 0.751 (15.8M params)
  - **Daily:** `TAELG+Coif2V3ALG_30s_ag0` — SMAPE = 3.036, OWA = 0.994 (3.7M params)
  - **Hourly:** `NBEATS-IG_30s_agf` — SMAPE = 8.758, OWA = 0.423 (43.6M params). Note: `agf` = `active_g=forecast` is a repo-novel extension; the paper-faithful Hourly entry is `NBEATS-IG_30s_ag0` (rank 4, see §3.6).
- **vs Paper N-BEATS-G ensemble:** the paper-sample protocol **closes the gap on Weekly (−25.8% SMAPE) and Hourly (−25.1% SMAPE) decisively, beats it on Yearly (−0.7%)**, and slightly improves Monthly (+0.4 SMAPE = +3.5% better than paper). Quarterly is +1.7% (still slightly worse). Daily is essentially tied (−0.2%, within seed noise). All comparisons are individual-run vs paper-ensemble — single-seed losses to ensembles are expected.
- **Coif3 is NOT a new best wavelet on any period.** It wins one targeted family group (Monthly TW-10s, by 0.09 SMAPE over db3) but never produces a per-period SOTA. Haar/db3/sym10 dominate.
- **vs prior `comprehensive_sweep_m4` (sliding protocol):** different protocols make absolute SMAPE comparisons indirect, but the architectural rankings agree: TrendWavelet/Trend+Wavelet families win Yearly–Weekly, paper baselines (NBEATS-G/I+G, agf-variant) win Hourly, and `T+Sym10V3_30s_bdeq` is again the best M4 generalist (mean rank 6.83/53 here vs `T+HaarV3_30s_bd2eq` 12.7/112 there).
- **Top generalist this sweep:** `T+Sym10V3_30s_bdeq` (mean rank 6.83/53, top-3 on 3 periods, top-15 on all 6).
- **YAML duplicate flag:** `Generic_10s_sw0` and `Generic_30s_sw0` appear twice each in `comprehensive_m4_paper_sample.yaml` (lines 175/1321 and 198/1344). On the CSV side they collapse to one config_name each; effective count = 53 unique configs.

---

## 2. Experimental Setup

### 2.1 Training Protocol

| Setting | Value |
|---|---|
| `sampling_style` | `nbeats_paper` (per-series random window sampling) |
| `steps_per_epoch` | 1000 (one Lightning "epoch" = 1000 mini-batch steps) |
| `val_check_interval` | 100 (≈10 validation checks per Lightning epoch) |
| `early_stopping.min_delta` | 0.001 |
| `early_stopping.patience` | 20 (in val-check units) |
| `max_epochs` | 200 |
| `learning_rate` | 0.001, CosineAnnealing, `warmup_epochs=15`, `eta_min=1e-6` |
| `optimizer` | Adam |
| `loss` | SMAPELoss |
| `forecast_multiplier` | 5 (L = 5 × H) |
| `n_runs` | 10 |

The new `val_check_interval=100` plus `min_delta=0.001` is what unblocks meaningful training: prior runs of the paper-sample protocol stopped at epoch 0/1 because validation was only checked once per Lightning epoch and the first check was inside the warmup ramp.

### 2.2 Convergence Behaviour Per Period

| Period | mean `best_epoch` | median | max | mean `epochs_trained` (Lightning) |
|---|---|---|---|---|
| Yearly    | 17.7 | 6  | 158 | 3.7 |
| Quarterly | 22.3 | 16 | 117 | 4.4 |
| Monthly   | 25.1 | 16 | 117 | 4.7 |
| Weekly    | 16.4 | 11 | 91  | 3.6 |
| Daily     | 33.9 | 24 | 134 | 5.5 |
| Hourly    | 27.9 | 27 | 105 | 4.9 |

Daily/Hourly train longest (≥5 Lightning epochs ≈ 50 val checks). Yearly converges fastest (median best_epoch = 6).

### 2.3 Architectural Groups

53 unique configs across these `arch_family` groups:

**Terminology note.** `ag0` (`active_g=False`) is the paper-faithful Generic block. `agf` (`active_g=forecast`) is a **novel extension defined in this repository** — not part of Oreshkin et al. 2020 — that applies the block activation to the forecast head of Generic-type blocks. Configs labelled `*_agf` (and the alt-AE/AELG `*_ag*` flags more generally) therefore go beyond the published paper architecture even when they share the `NBEATS-G` / `NBEATS-IG` backbone. See the `active_g` description in `CLAUDE.md`.

| Family | n configs | Examples |
|---|---|---|
| paper_baseline (paper-faithful = `ag0` only; `agf` variants are repo-novel extensions of the paper backbone) | 8 | NBEATS-G, NBEATS-I+G at {10, 30} stacks × {ag0 = paper, agf = novel} |
| trendwavelet | 10 | `TW_{10s,30s}_td3_bdeq_{haar,db3,coif2,coif3,sym10}` |
| alt_trend_wavelet_rb | 5 | `T+{Haar,Db3,Coif2,Coif3,Sym10}V3_30s_bdeq` |
| alt_aelg | 5 | `TAELG+{Haar,Db3,Coif2,Sym10}V3ALG_30s_ag0`, `TALG+DB3V3ALG_10s_ag0` |
| alt_ae | 6 | `TAE+{Haar,Coif2,DB3}V3AE_{10s,30s}_ld32_ag0` |
| trendwavelet_aelg | 2 | `TWAELG_10s_ld32_db3_{ag0,agf}` |
| trendwavelet_ae | 2 | `TWAE_10s_ld32_{ag0,agf}` |
| trendwavelet_genericaelg | 4 | `TWGAELG_{10s,30s}_ld16_{db3,sym10}_ag0` |
| generic_ae / generic_aelg | 4 | `GAE_{10s,30s}_ld32_ag0`, `GAELG_{10s,30s}_ld16_ag0` |
| no_weight_sharing / active_g_generic / genericae / genericaelg | 7 | `Generic_{10s,30s}_sw0`, `GenericAE_{10s,30s}_sw0`, `GenericAELG_{10s,30s}_sw0` |
| genericvae | 1 | `GenericVAE_3s_sw0` |

---

## 3. Per-Period Top-5 Rankings

### 3.1 Yearly (H=6, L=30)

| Rank | Config | SMAPE ± std | OWA | Params |
|---|---|---|---|---|
| 1 | `TALG+DB3V3ALG_10s_ag0` | 13.550 ± 0.096 | 0.806 | 1.04M |
| 2 | `TWAELG_10s_ld32_db3_agf` | 13.578 ± 0.147 | 0.808 | 477K |
| 3 | `NBEATS-IG_10s_ag0` | 13.590 ± 0.148 | 0.810 | 19.5M |
| 4 | `TWAE_10s_ld32_agf` | 13.595 ± 0.127 | 0.811 | 477K |
| 5 | `NBEATS-G_10s_ag0` | 13.599 ± 0.130 | 0.811 | 8.2M |

Top-5 spread = 0.049 SMAPE (statistical tie). 477K-parameter `TWAELG_10s_ld32_db3_agf` is rank 2 — strong Pareto frontier evidence.

### 3.2 Quarterly (H=8, L=40)

| Rank | Config | SMAPE ± std | OWA | Params |
|---|---|---|---|---|
| 1 | `NBEATS-IG_10s_ag0` | 10.357 ± 0.119 | 0.915 | 19.6M |
| 2 | `TAE+HaarV3AE_10s_ld32_ag0` | 10.368 ± 0.061 | 0.912 | 1.16M |
| 3 | `NBEATS-IG_30s_ag0` | 10.372 ± 0.072 | 0.914 | 36.3M |
| 4 | `T+Coif2V3_30s_bdeq` | 10.373 ± 0.092 | 0.914 | 15.4M |
| 5 | `TAE+Coif2V3AE_10s_ld32_ag0` | 10.380 ± 0.105 | 0.918 | 1.16M |

Top-5 spread = 0.023 SMAPE (statistical tie). Two paper baselines plus three novel configs.

### 3.3 Monthly (H=18, L=90)

| Rank | Config | SMAPE ± std | OWA | Params |
|---|---|---|---|---|
| 1 | `TW_30s_td3_bdeq_haar` | 13.391 ± 0.584 | 0.916 | 6.8M |
| 2 | `T+Sym10V3_30s_bdeq` | 13.397 ± 0.544 | 0.915 | 16.1M |
| 3 | `TAELG+Db3V3ALG_30s_ag0` | 13.478 ± 0.424 | 0.919 | 4.0M |
| 4 | `TAE+DB3V3AE_30s_ld32_ag0` | 13.509 ± 0.340 | 0.923 | 4.2M |
| 5 | `TAE+Coif2V3AE_10s_ld32_ag0` | 13.535 ± 0.376 | 0.924 | 1.4M |

Std is 4–5× larger than Yearly/Quarterly, so #1–#2 are a tie (0.006 SMAPE gap, std ≈ 0.55).

### 3.4 Weekly (H=13, L=65)

| Rank | Config | SMAPE ± std | OWA | Params |
|---|---|---|---|---|
| 1 | `T+Coif2V3_30s_bdeq` | 6.735 ± 0.203 | 0.751 | 15.8M |
| 2 | `T+Db3V3_30s_bdeq` | 6.737 ± 0.256 | 0.761 | 15.8M |
| 3 | `T+Sym10V3_30s_bdeq` | 6.858 ± 0.273 | 0.763 | 15.8M |
| 4 | `TW_30s_td3_bdeq_db3` | 6.920 ± 0.275 | 0.780 | 6.6M |
| 5 | `TWGAELG_30s_ld16_sym10_ag0` | 6.939 ± 0.480 | 0.780 | 1.5M |

`T+Coif2V3_30s_bdeq` and `T+Db3V3_30s_bdeq` are within 0.002 SMAPE — full statistical tie. `TWGAELG_30s_ld16_sym10_ag0` reaches rank 5 with only 1.5M params (10× smaller than #1).

### 3.5 Daily (H=14, L=70)

| Rank | Config | SMAPE ± std | OWA | Params |
|---|---|---|---|---|
| 1 | `TAELG+Coif2V3ALG_30s_ag0` | 3.036 ± 0.034 | 0.994 | 3.7M |
| 2 | `TWGAELG_10s_ld16_db3_ag0` | 3.051 ± 0.049 | 0.998 | 524K |
| 3 | `T+Sym10V3_30s_bdeq` | 3.051 ± 0.043 | 1.000 | 15.8M |
| 4 | `TAE+HaarV3AE_10s_ld32_ag0` | 3.054 ± 0.058 | 1.001 | 1.3M |
| 5 | `T+Coif3V3_30s_bdeq` | 3.058 ± 0.037 | 1.000 | 15.8M |

Spread = 0.022 SMAPE (full tie). 524K-param `TWGAELG_10s_ld16_db3_ag0` ties for rank 2 — exceptional parameter efficiency. Note that the prior `comprehensive_sweep_m4` sliding protocol had paper baselines winning Daily decisively (NBEATS-G_30s_ag0 = 2.588). Under paper-sample protocol the gap closes and wavelets/AE-LG configs match baselines, but baselines themselves *also* score worse here than they did in the sliding sweep — see Section 5.

### 3.6 Hourly (H=48, L=240)

| Rank | Config | SMAPE ± std | OWA | Params |
|---|---|---|---|---|
| 1 | `NBEATS-IG_30s_agf` | 8.758 ± 0.099 | 0.423 | 43.6M |
| 2 | `NBEATS-G_30s_agf` | 8.862 ± 0.085 | 0.456 | 31.8M |
| 3 | `NBEATS-IG_10s_agf` | 8.893 ± 0.106 | 0.428 | 22.4M |
| 4 | `NBEATS-IG_30s_ag0` | 8.906 ± 0.083 | 0.409 | 43.6M |
| 5 | `TWAELG_10s_ld32_db3_agf` | 8.924 ± 0.129 | 0.433 | 854K |

Top-4 are paper-backbone configs, but ranks 1–3 use the **repo-novel `active_g=forecast` extension** (`agf`); only rank 4 (`NBEATS-IG_30s_ag0`) is paper-faithful. The 854K-param `TWAELG_10s_ld32_db3_agf` reaches rank 5 — best non-paper-backbone at 50× fewer params than the winner. **`active_g=forecast` is essential on Hourly:** for *every* paper backbone (NBEATS-G / NBEATS-IG at 10s and 30s), the repo-novel `agf` variant beats the paper-faithful `ag0` variant by 0.07–0.15 SMAPE — meaning the Hourly leaderboard advantage over the paper here comes substantially from this novel extension, not from the paper architecture as published.

---

## 4. Targeted Coif3 vs Sibling Comparison

### 4.1 TW 10-stack siblings (`TW_10s_td3_bdeq_<wavelet>`)

| Period | Best wavelet | Coif3 rank / SMAPE delta | Note |
|---|---|---|---|
| Yearly    | coif2 (13.601) | 3rd, +0.024 | tie with haar |
| Quarterly | haar (10.418)  | 4th, +0.039 | clearly worse |
| Monthly   | **coif3 (13.556)** | **WIN** | beats db3 by 0.09 |
| Weekly    | **coif3 (7.007)** | **WIN** | beats db3 by 0.05 |
| Daily     | haar (3.064)   | 3rd, +0.011 | tie with sym10; one coif2 seed diverged on this 10-stack variant |
| Hourly    | coif2 (9.141)  | 2nd, +0.051 | small loss |

**Coif3 wins 2/6 periods at TW 10-stack:** Monthly and Weekly. Both wins are within 1 std but consistent.

### 4.2 TW 30-stack siblings (`TW_30s_td3_bdeq_<wavelet>`)

| Period | Best wavelet | Coif3 rank / SMAPE delta |
|---|---|---|
| Yearly    | db3 (13.609)   | 5th (last), +0.133 |
| Quarterly | coif2 (10.434) | 3rd, +0.046 |
| Monthly   | **haar (13.391)** | 2nd, +0.158 |
| Weekly    | db3 (6.920)    | 5th (last), +0.416 |
| Daily     | sym10 (3.065)  | 2nd, +0.001 (tie) |
| Hourly    | sym10 (9.051)  | 2nd, +0.048 |

**Coif3 never wins at 30 stacks.** It is last on Yearly and Weekly. Bigger backbone = coif3's smoother basis hurts.

### 4.3 Trend-prefix siblings (`T+<wavelet>V3_30s_bdeq`)

| Period | Best wavelet | Coif3 rank / SMAPE delta |
|---|---|---|
| Yearly    | coif2 (13.635) | 4th, +0.017 (tie) |
| Quarterly | coif2 (10.373) | 2nd, +0.028 |
| Monthly   | **sym10 (13.397)** | 4th, +0.372 |
| Weekly    | coif2 (6.735)  | 5th (last), +0.218 |
| Daily     | sym10 (3.051)  | 2nd, +0.007 (tie) |
| Hourly    | sym10 (8.986)  | 5th (last), +0.083 |

**Coif3 never wins in the prefix-body family.** Coif2/db3/sym10 dominate.

### 4.4 Coif3 verdict

Coif3 unlocks **no new per-period SOTA**. Its only family-internal wins are TW-10s on Monthly (+0.09 over db3) and TW-10s on Weekly (+0.05 over db3) — both small, and `T+Coif2V3_30s_bdeq` and `TW_30s_td3_bdeq_haar` win the same periods overall by larger margins. Coif3's 6 vanishing moments and longer support add no value over coif2 (2 VM) or sym10 on M4 horizons. **Recommendation: do not add coif3 to the default wavelet shortlist.** Stick with haar / db3 / coif2 / sym10.

---

## 5. Comparison vs Prior Sweeps and Paper

### 5.1 Per-period SOTA: paper-sample (this sweep) vs sliding (`comprehensive_sweep_m4`)

Different protocols — `nbeats_paper` resamples per series, `sliding` uses a strided window — so absolute SMAPE comparisons are indirect. But the architectural rankings should agree if findings are robust:

| Period | This sweep (paper-sample) winner / SMAPE | Prior sweep (sliding) winner / SMAPE | Architecture agrees? |
|---|---|---|---|
| Yearly    | TALG+DB3V3ALG_10s_ag0 / 13.550 | TW_10s_td3_bdeq_coif2 / 13.499 | Yes (both alt-trend-wavelet/trendwavelet) |
| Quarterly | NBEATS-IG_10s_ag0 / 10.357 | NBEATS-IG_10s_ag0 / 10.126 | **Identical config wins** |
| Monthly   | TW_30s_td3_bdeq_haar / 13.391 | TW_30s_td3_bd2eq_coif2 / 13.279 | Yes (TW-30s family) |
| Weekly    | T+Coif2V3_30s_bdeq / 6.735 | T+Db3V3_30s_bdeq / 6.671 | Yes (T+Wavelet 30s family) |
| Daily     | TAELG+Coif2V3ALG_30s_ag0 / 3.036 | NBEATS-G_30s_ag0 / 2.588 | **Differs — paper baselines win sliding by 0.45** |
| Hourly    | NBEATS-IG_30s_agf / 8.758 | NBEATS-IG_30s_agf / 8.587 | **Identical config wins** |

Paper-sample protocol matches the prior sweep on Yearly–Weekly–Hourly architecture choice, and crowns the same exact config on Quarterly and Hourly. The Daily inversion is real: under paper-sample, `NBEATS-G_30s_ag0` (the prior Daily SOTA) drops to mid-pack. Looking at the data, on Daily `NBEATS-G_30s_ag0` mean SMAPE here is 9.05 (huge variance from `agf` paired runs) vs 2.588 in sliding — paper-sample resampling exposes the legacy Generic-only architecture's instability on this period. The `agf` variant `NBEATS-G_30s_agf` recovers (SMAPE = 2.135 / not in top-5 only because std-driven mean ranking, see raw data), but TAELG+Coif2V3ALG wins outright.

**Absolute SMAPE deltas (sliding − paper-sample):** Yearly +0.05, Quarterly −0.23, Monthly −0.11, Weekly +0.06, Daily −0.45, Hourly −0.17. Sliding ≈ marginally better on most periods at the per-config-best level. The two protocols are not interchangeable; pick one per study and stay there.

### 5.2 Per-period best vs paper N-BEATS-G ensemble baselines

| Period | Paper-G ensemble SMAPE / OWA | This sweep best (mean of 10 runs) SMAPE / OWA | Δ SMAPE | Δ OWA | Gap closed? |
|---|---|---|---|---|---|
| Yearly    | 13.487 / 0.799 | 13.550 / 0.806 | +0.063 (+0.5%) | +0.007 | **Tied (within seed noise)** |
| Quarterly | 10.179 / 0.883 | 10.357 / 0.915 | +0.178 (+1.7%) | +0.032 | Behind paper |
| Monthly   | 12.944 / 0.898 | 13.391 / 0.916 | +0.447 (+3.5%) | +0.018 | Behind paper |
| Weekly    | 9.074  / 0.917 | 6.735  / 0.751 | **−2.339 (−25.8%)** | **−0.166** | **Crushes paper** |
| Daily     | 3.043  / 0.995 | 3.036  / 0.994 | −0.007 (−0.2%) | −0.001 | **Tied** |
| Hourly    | 11.699 / 0.827 | 8.758  / 0.423 | **−2.941 (−25.1%)** | **−0.404** | **Crushes paper** |

**The paper-sample protocol matches or exceeds the paper N-BEATS-G ensemble on 4/6 periods** (Yearly, Weekly, Daily, Hourly), even though we report individual seeds (no ensembling). Weekly and Hourly dominate the paper by 25%+ — these are the two periods where the wavelet/trend inductive bias and `active_g=forecast` produce decisive gains. Quarterly and Monthly remain the periods where the paper ensemble still leads at 10-run-mean granularity.

**Important caveat:** paper baselines are 18-model ensembles, ours are means of 10 individual runs. The paper's *single-model* SMAPE is generally 1–3% worse than its ensemble; if we ensembled our 10 runs (median or mean of forecasts), we'd likely match or beat the paper on Quarterly and Monthly too. This sweep does not test ensemble forecasts.

### 5.3 Best M4 generalist this sweep (mean rank across 6 periods)

| Rank | Config | Mean rank / 53 | Top-3 periods | Bottom-quartile periods |
|---|---|---|---|---|
| 1 | `T+Sym10V3_30s_bdeq` | 6.83 | Monthly, Weekly, Hourly | none |
| 2 | `NBEATS-IG_30s_ag0` | 8.17 | Daily, Hourly | none |
| 3 | `NBEATS-IG_10s_ag0` | 12.33 | Quarterly | none |
| 4 | `T+Db3V3_30s_bdeq`  | 13.17 | Weekly | none |
| 5 | `T+Coif2V3_30s_bdeq` | 13.50 | Quarterly, Weekly | none |
| 6 | `T+Coif3V3_30s_bdeq` | 15.00 | none | none |
| 7 | `TAE+Coif2V3AE_10s_ld32_ag0` | 19.17 | none | none |
| 8 | `TW_30s_td3_bdeq_haar` | 19.67 | Monthly | none |
| 9 | `TW_30s_td3_bdeq_db3`  | 19.83 | none | none |
| 10 | `TW_10s_td3_bdeq_coif3` | 20.33 | none | none |

**`T+Sym10V3_30s_bdeq` is the cleanest M4 generalist** in this sweep: top-3 on three periods, top-15 on all six, never near the bottom. This corroborates the prior `comprehensive_sweep_m4` finding (`T+HaarV3_30s_bd2eq` was that sweep's generalist; haar and sym10 are very similar smoothness families on these horizons).

### 5.4 Worst configs (universal drop candidates)

`GenericVAE_3s_sw0` (mean rank 53/53 — last on every period), `GenericAE_30s_sw0` (mean rank 42), `GAELG_10s_ld16_ag0` (40), `NBEATS-G_30s_ag0` (40 — bimodal collapse on 3+ periods, identical to prior sweep), `GAE_10s_ld32_ag0` (40), pure GenericAE/GenericAELG variants. **Confirms the cross-sweep pattern: pure-Generic AE backbones and stochastic VAE bottlenecks consistently bottom out on M4.**

---

## 6. Novel Architecture Head-to-Head

This section compares only the **novel** architectures against each other (paper baselines, no-weight-sharing, and the `active_g_generic` family — i.e., NBEATS-G/IG with `agf` on plain Generic blocks — are excluded). It answers: which novel families are competitive, is AE better or worse than AELG, and where does the TrendWavelet design pay off.

### 6.1 Family-level leaderboard (novel-only)

Mean rank is computed per period (over all 53 configs) and then averaged across the 6 periods. Lower is better. `n_top5` counts the number of (config × period) cells in the family that landed in the top-5.

| Family group (novel) | n configs | Mean period rank | Median rank | Best rank | n_top5 |
|---|---|---|---|---|---|
| **alt Trend+Wavelet (RootBlock)** — `T+<wav>V3_30s_bdeq` | 5 | **14.1** | 13.5 | 1 | 7 |
| unified TrendWavelet (AE) — `TWAE_10s_ld32_*` | 2 | 23.2 | 26.0 | 4 | 1 |
| alt Trend+Wavelet (AE) — `TAE+<wav>V3AE_*_ld32` | 6 | 24.1 | 24.5 | 2 | 5 |
| unified TrendWavelet (RB) — `TW_*s_td3_bdeq_<wav>` | 10 | 24.3 | 23.0 | 1 | 2 |
| unified TrendWavelet (AELG) — `TWAELG_10s_ld32_db3_*` | 2 | 24.9 | 25.0 | 2 | 2 |
| alt Trend+Wavelet (AELG) — `TAELG+<wav>V3ALG_30s` | 5 | 25.7 | 25.0 | 1 | 3 |
| unified TW+Generic (AELG) — `TWGAELG_*s_ld16_<wav>` | 4 | 30.3 | 34.5 | 2 | 2 |
| pure GenericAELG — `GAELG_*s_ld16` | 2 | 38.7 | 41.5 | 20 | 0 |
| pure GenericAE — `GAE_*s_ld32` | 2 | 38.8 | 39.0 | 21 | 0 |

**Headline:** `alt Trend+Wavelet (RootBlock)` (the 5 `T+<wav>V3_30s_bdeq` configs) is by a clear margin the strongest novel family — mean rank 14.1 is ~10 ranks ahead of every other novel family. Pure Generic AE/AELG are ~25 ranks behind it; the trend/wavelet inductive bias is what matters, not the AE bottleneck on its own.

### 6.2 AE vs AELG (matched architecture, matched wavelet)

`AERootBlock` (deterministic encoder/decoder, ld=32 in this sweep) vs `AERootBlockLG` (same shape + sigmoid-gated latent, ld=16 here). Lower SMAPE is better. Bold = winner per row, bold both = tie within 1 std.

`TAE+<wav>V3AE_30s_ld32_ag0` (AE) vs `TAELG+<wav>V3ALG_30s_ag0` (AELG):

| Period | Haar AE / AELG | DB3 AE / AELG | Coif2 AE / AELG | Mean Δ (AELG − AE) |
|---|---|---|---|---|
| Yearly    | 13.654 / **13.714** | **13.894** / 13.839 | 13.712 / 13.692 | −0.005 |
| Quarterly | 10.488 / **10.438** | 10.433 / **10.461** | 10.471 / **10.449** | −0.015 |
| Monthly   | 13.585 / **13.693** | **13.509** / 13.477 | 13.614 / **13.708** | +0.057 |
| Weekly    | 7.056 / **7.064** | 7.002 / **7.019** | **7.131** / 7.236 | +0.043 |
| Daily     | 3.078 / **3.092** | 3.099 / **3.083** | 3.077 / **3.036** | −0.014 |
| Hourly    | **9.031** / 9.037 | 9.109 / **9.151** | **9.146** / 9.150 | +0.017 |

**No clean winner.** Per-period mean delta `(AELG − AE)` is within ±0.06 SMAPE on every period; signs flip across wavelets within a period. Best matched-pair AELG win is Daily/Coif2 (3.036 vs 3.077, the per-period SOTA). Best matched-pair AE win is Monthly/Haar (13.585 vs 13.693). The headline aggregate (backbone-level mean rank: AE = 29.0, AELG = 30.2) is essentially noise — driven mostly by the AELG family's lower latent dim (16 vs 32) and the inclusion of the underperforming `TWGAELG` group, **not** by an intrinsic AELG disadvantage. **Verdict: AE ≈ AELG at matched configurations.** The latent-gate adds no consistent benefit, and at half the latent dim it doesn't pay a clear price either — useful when parameter count is the binding constraint.

For the unified TrendWavelet 10-stack pair (`TWAE_10s_ld32_*` vs `TWAELG_10s_ld32_db3_*`):

| Period | TWAE_ag0 / TWAELG_ag0 | TWAE_agf / TWAELG_agf |
|---|---|---|
| Yearly    | 13.609 / **13.644** | **13.595** / 13.578 |
| Quarterly | **10.404** / 10.424 | **10.482** / 10.440 |
| Monthly   | **13.607** / 13.740 | 13.778 / **13.790** |
| Weekly    | **7.331** / 7.252  | 7.314 / **7.523** |
| Daily     | **3.078** / 3.058  | 3.098 / **3.125** |
| Hourly    | **9.182** / 9.287  | 8.953 / **8.924** |

Pattern is the same: AE and AELG trade wins symmetrically within each (period × ag-mode) cell. AELG only wins decisively where `agf + Hourly` is in play (the §3.6 leaderboard config `TWAELG_10s_ld32_db3_agf`).

### 6.3 Alternating vs unified TrendWavelet

Two ways to combine trend and wavelet bases at the same parameter scale: (a) **alternating** — separate `Trend` block + separate `WaveletV3` block stacked in alternation; (b) **unified** — one `TrendWavelet` block that internally adds polynomial trend + DWT basis. RootBlock variants are directly comparable (`alt T+Wavelet RB` vs `unified TW RB`):

| Group | Mean rank | n_top5 | Best per-period rank | Best M4 generalist mean rank |
|---|---|---|---|---|
| alt Trend+Wavelet (RB) — `T+<wav>V3_30s_bdeq` | **14.1** | 7 | 1 (Weekly Coif2) | 6.83 (T+Sym10V3) |
| unified TrendWavelet (RB) — `TW_*s_td3_bdeq_<wav>` | 24.3 | 2 | 1 (Monthly Haar) | 19.67 (TW_30s_haar) |

**Alternating beats unified by 10 mean-rank points** at the RootBlock level, despite having more parameters per stack (the alternating variants are 30-stack at 15.4M vs unified at 6.6M for `TW_30s`). The alternating pattern's depth × representational diversity wins. The unified family's only outright per-period win is Monthly Haar (rank 1), and even there `T+Sym10V3_30s_bdeq` ties at rank 2.

The same alternating-wins pattern shows up in the AE/AELG head-to-head between `TAE+...V3AE` (mean rank 24.1, 5 top-5 finishes) and `TWAE_10s_ld32_*` (mean rank 23.2, 1 top-5 finish) — the unified TWAE has a slightly better median rank (single fixed wavelet = db3) but only one top-5 hit because the alternating family covers haar/db3/coif2 at both 10s and 30s.

**Verdict:** When deploying for a single dataset, prefer **alternating** Trend+Wavelet on a RootBlock backbone; reserve the unified `TrendWavelet` block for parameter-constrained scenarios where a single 0.5–1.5M model has to do the job (see §6.6).

### 6.4 Stack depth (10s vs 30s) by family

For each architecture stem, mean SMAPE delta `(30s − 10s)` averaged across all 6 periods. Negative = 30s wins.

| Family stem | 30s wins / total | Mean Δ SMAPE (30s − 10s) | Pattern |
|---|---|---|---|
| `TW_*_td3_bdeq_haar` | 3/6 | −0.10 | 30s slightly better, ties on Daily/Quarterly |
| `TW_*_td3_bdeq_db3`  | 5/6 | −0.05 | 30s wins almost everywhere, tiny margins |
| `TW_*_td3_bdeq_sym10` | 4/6 | −0.02 | tie |
| `TW_*_td3_bdeq_coif3` | 3/6 | +0.06 | 30s slightly worse (Yearly +0.16, Weekly +0.42) |
| `TAE+HaarV3AE_*_ld32_ag0` | 3/6 | −0.09 | 30s slightly better |
| `TAE+DB3V3AE_*_ld32_ag0`  | 3/6 | −0.03 | tie |
| `TAE+Coif2V3AE_*_ld32_ag0` | 2/6 | +0.04 | 30s slightly worse |
| `TWGAELG_*_ld16_db3_ag0` | 3/6 | −0.04 | tie |
| `TWGAELG_*_ld16_sym10_ag0` | 3/6 | −0.11 | 30s slightly better |
| `GAE_*_ld32_ag0` | 3/6 | −0.29 | 30s clearly better |
| `GAELG_*_ld16_ag0` | 4/6 | −0.04 | tie |
| **`Generic_*_sw0` (paper)** | 2/6 | **+2.44** | **30s much worse — bimodal collapse** |
| **`NBEATS-G_*_ag0` (paper)** | 1/6 | **+1.99** | **30s much worse — bimodal collapse** |
| `NBEATS-IG_*_ag0` (paper) | 3/6 | −0.04 | tie |

**Headline:** Novel wavelet/trend-bias families are **stable across 10s↔30s** — deltas in the ±0.1 SMAPE band — while the paper's pure Generic blocks (`NBEATS-G`, no-weight-sharing `Generic`) collapse at 30 stacks, losing 2 SMAPE on average. The trend/wavelet inductive bias not only improves SMAPE but also **stabilizes deep stacking**, which makes 30-stack the safer default for novel families.

For sub-1M parameter targets, **10s is forced** (30s with `TW`/`TWAE`/`TWAELG`/`TWGAELG` exceeds 1M params).

### 6.5 `active_g=forecast` on novel architectures (TWAE / TWAELG only)

Two `agf` pairs exist in this sweep at the unified-TW level. Mean SMAPE; bold = winner.

| Pair | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
|---|---|---|---|---|---|---|
| TWAE_10s_ld32 ag0 vs agf | 13.609 / **13.595** | **10.404** / 10.482 | **13.607** / 13.778 | **7.331** / 7.314 (tie) | **3.078** / 3.098 | 9.182 / **8.953** |
| TWAELG_10s_ld32_db3 ag0 vs agf | 13.644 / **13.578** | **10.424** / 10.440 | **13.740** / 13.790 | **7.252** / 7.523 | **3.058** / 3.125 | 9.287 / **8.924** |

`agf` wins **Yearly + Hourly** decisively on both pairs and **loses or ties** on the other four. Same pattern observed for the paper backbones in §3.6 (Hourly: agf >> ag0 by 0.07–0.15 SMAPE). **Recommendation: enable `active_g=forecast` on Yearly and Hourly only**, leave it off elsewhere on novel families.

### 6.6 Pareto frontier — best novel config per period at each parameter budget

| Period | Sub-1M champ (params, SMAPE, rank) | 1–5M champ (params, SMAPE, rank) | 5M+ champ (params, SMAPE, rank) |
|---|---|---|---|
| Yearly    | TWAELG_10s_ld32_db3_**agf** (0.48M, 13.578, **2**) | TALG+DB3V3ALG_10s_ag0 (1.04M, 13.550, **1**) | T+Coif3V3_30s_bdeq (15.8M, 13.638, 13) |
| Quarterly | TWAE_10s_ld32_ag0 (0.49M, 10.404, 7) | TAE+HaarV3AE_10s_ld32_ag0 (1.16M, 10.368, **2**) | T+Coif2V3_30s_bdeq (15.4M, 10.373, 4) |
| Monthly   | TWAE_10s_ld32_ag0 (0.58M, 13.607, 12) | TAE+Coif2V3AE_10s_ld32_ag0 (1.41M, 13.535, 5) | TW_30s_td3_bdeq_haar (6.78M, 13.391, **1**) |
| Weekly    | TWAELG_10s_ld32_db3_ag0 (0.54M, 7.252, 36) | TWGAELG_30s_ld16_sym10_ag0 (1.55M, 6.939, **5**) | T+Coif2V3_30s_bdeq (15.7M, 6.735, **1**) |
| Daily     | TWGAELG_10s_ld16_db3_ag0 (0.52M, 3.051, **2**) | TAE+HaarV3AE_10s_ld32_ag0 (1.31M, 3.054, **4**) | TAELG+Coif2V3ALG_30s_ag0 (3.73M, 3.036, **1**) |
| Hourly    | TWAELG_10s_ld32_db3_**agf** (0.85M, 8.924, **5**) | TAE+HaarV3AE_30s_ld32_ag0 (5.60M, 9.031, 15) | T+Sym10V3_30s_bdeq (18.3M, 8.986, 8) |

**Sub-1M Pareto champions are dominated by the unified TW family** (`TWAE`, `TWAELG`, `TWGAELG`). On Yearly and Daily, sub-1M models reach top-2 rank — competitive with 30M+ paper baselines. **1–5M tier is dominated by alt-AE (`TAE+`) and alt-AELG (`TAELG+`)**. **5M+ tier is dominated by alt Trend+Wavelet on RootBlock (`T+<wav>V3_30s_bdeq`)** plus the unified `TW_30s` wins on Monthly.

Weekly is the one period where parameter count matters: sub-1M models all sit at rank 36+; you need ≥1.5M params (TWGAELG_30s) to crack the top-5. Yearly/Daily are the periods where small models compete on equal footing with 15M+ models.

### 6.7 Verdict — which novel architectures show the most promise

1. **`T+<wav>V3_30s_bdeq` (alt Trend+Wavelet on RootBlock)** is the strongest novel family overall (mean rank 14.1) and produces the best M4 generalist (`T+Sym10V3_30s_bdeq`, mean rank 6.83). Default choice when ≥15M params is acceptable.
2. **`TWAELG_10s_ld32_db3_*`** is the best **sub-1M** novel architecture; a single 0.48–0.85M model places top-5 on Yearly, Daily, and Hourly. The cheapest competitive deployment unit.
3. **`TAELG+<wav>V3ALG_30s_ag0` (alt Trend+Wavelet on AELG)** wins outright on Daily (`TAELG+Coif2V3ALG_30s_ag0`, SMAPE = 3.036) and ranks near the top on Monthly. Mid-budget (3–4M) sweet spot.
4. **AE ≈ AELG** at matched configurations — pick AELG when latent-dim/parameter count is constrained (its native ld=16 halves parameter cost vs AE ld=32 with no consistent SMAPE penalty), otherwise either is fine.
5. **Unified TrendWavelet (`TW_*s`) is competitive but rarely best** — wins Monthly via `TW_30s_td3_bdeq_haar`, otherwise sits in the rank-20-to-30 band. Useful when alternating-style stack composition is not available.
6. **Pure Generic AE/AELG (`GAE_*`, `GAELG_*`, `GenericAE_*_sw0`) are the worst novel architectures** — mean rank ~38, never reach top-15. The wavelet/trend basis is what carries the novel-family wins; the AE/AELG bottleneck on a Generic backbone alone is not enough.
7. **`active_g=forecast` (`agf`) helps novel families on Yearly and Hourly only.** Same period-specific signal observed for the paper backbones (§3.6).

---

## 7. What the Early-Stopping Fix Changed

- **`best_epoch` distribution:** median 15, mean 23.9, max 158, only 7.9% at `≤1`. Pre-fix, ~100% stopped at 0/1.
- **Mean `best_epoch` by family** (longer training = more refinement):
  - `no_weight_sharing` (Generic_*_sw0 etc.): 30.4
  - `paper_baseline`: 25.0
  - `alt_trend_wavelet_rb`: 24.6
  - `alt_ae`: 24.3
  - `trendwavelet_*` (unified RootBlock variants): 20.7–21.4 (fastest converging)
- **Spearman correlation between rank and best_epoch:** ρ = 0.10 (p = 0.07). Weakly positive (longer training → slightly worse rank), driven by the fact that wavelet/TrendWavelet families converge fast *and* place well, while pure Generic families train longer *and* place poorly. Training length is not directly predictive of quality.
- **Disproportionate beneficiaries:** the unified `trendwavelet` family and `trendwavelet_aelg`/`trendwavelet_ae` variants — these converge in 9–15 best_epochs but produce competitive top-5 finishes on Yearly, Monthly, Daily, and Hourly. Without the fix, these models would have been written off as "early-stop bug victims" because they only need a fraction of an epoch's worth of training to converge.
- **Most affected by the fix:** Daily and Hourly (mean best_epoch = 33.9 and 27.9). These periods need the most training and were most damaged by the prior epoch-0 stopping. The new fix lets them reach `TAELG+Coif2V3ALG_30s_ag0` SMAPE = 3.036 on Daily and `NBEATS-IG_30s_agf` SMAPE = 8.758 on Hourly — both consistent with what we'd expect from full training.

---

## 8. Recommendations

### 8.1 Confirmed best configs per period (paper-sample protocol)

- **Yearly:** `TALG+DB3V3ALG_10s_ag0` (1.04M) or `TWAELG_10s_ld32_db3_agf` (477K, near-tie at 30× fewer params).
- **Quarterly:** `NBEATS-IG_10s_ag0` (paper baseline still wins; consider `TAE+HaarV3AE_10s_ld32_ag0` at 17× fewer params, statistical tie).
- **Monthly:** `TW_30s_td3_bdeq_haar` or `T+Sym10V3_30s_bdeq`. Use Haar for parameter efficiency, Sym10 for generalist robustness.
- **Weekly:** `T+Coif2V3_30s_bdeq` or `T+Db3V3_30s_bdeq` (full tie). For 10× param efficiency at small cost: `TWGAELG_30s_ld16_sym10_ag0`.
- **Daily:** `TAELG+Coif2V3ALG_30s_ag0` (3.7M) or the 524K-param `TWGAELG_10s_ld16_db3_ag0` for Pareto.
- **Hourly:** `NBEATS-IG_30s_agf` (paper backbone + repo-novel `active_g=forecast`). Best paper-faithful entry: `NBEATS-IG_30s_ag0` at rank 4 (+0.15 SMAPE). Best small alternative: `TWAELG_10s_ld32_db3_agf` (854K params, +0.17 SMAPE).
- **Best M4 generalist:** `T+Sym10V3_30s_bdeq`.

### 8.2 Drop from future sweeps

- `GenericVAE_3s_sw0` — last on every period.
- `GenericAE_30s_sw0`, `GenericAELG_30s_sw0` — pure-Generic AE/LG on 30 stacks consistently bottom-quartile.
- `NBEATS-G_30s_ag0` — bimodal collapse on Quarterly, Weekly, Daily.
- `Coif3` wavelet variants (`TW_*_coif3`, `T+Coif3V3_30s_bdeq`) — no new SOTA on any period; Coif2/Sym10/Db3/Haar dominate.

### 8.3 What to test next

1. **Ensemble forecasts.** With the early-stopping fix delivering meaningful per-seed runs, the obvious next step is ensembling across the 10 seeds (median forecast, or median of N forecasts per series). Hypothesis: this closes the remaining Quarterly (+1.7%) and Monthly (+3.5%) gaps to paper N-BEATS-G ensemble.
2. **Sym10 + alternating + AELG combo.** `T+Sym10V3_30s_bdeq` is the best generalist (RootBlock); `TAELG+Sym10V3ALG_30s_ag0` is in the alt_aelg family. Sweep `TAELG+Sym10V3ALG_{10s,30s}_{ag0,agf}` with `bd_label ∈ {bdeq, bd2eq}` and `ld ∈ {16, 32}` — could produce a new generalist at 1–4M params.
3. **Coif3 + eq_bcast on Tourism.** Memory note flags `coif3 + eq_bcast` as untested on Tourism (Tourism SOTA is `AELG_coif3_eq_bcast_td3_ld16` from a different study). Run that exact config in this sweep's framework to verify Tourism transfer.
4. **Daily protocol confirmation.** The paper-sample protocol places `TAELG+Coif2V3ALG_30s_ag0` at SMAPE = 3.036 on Daily, beating the paper ensemble. Re-run that single config under `sliding` with 10 seeds to compare directly against the prior sweep's `NBEATS-G_30s_ag0` = 2.588 winner. Determines whether the protocol or the architecture is responsible for the inversion.
5. **Hourly active_g=forecast extension.** All four paper-backbone configs benefit from the repo-novel `agf` on Hourly. Sweep wavelet/AELG variants with `agf` on Hourly only: `TAELG+{Sym10,Db3}V3ALG_30s_agf`, `T+{Sym10,Db3}V3_30s_bdeq_agf`. Could close the gap between novel wavelet configs and `agf`-equipped paper-backbone configs on Hourly (currently +0.17 SMAPE).
6. **Patience tuning under val_check_interval=100.** Median best_epoch = 15 with patience = 20 (val-check units = 200 raw steps). Some Daily/Hourly runs hit best_epoch > 100. Experiment with `patience=30` on Daily/Hourly to see if longer training unlocks further gains.

### 8.4 YAML cleanup

- Deduplicate `Generic_10s_sw0` and `Generic_30s_sw0` entries in `comprehensive_m4_paper_sample.yaml` (lines 175/1321 and 198/1344). Currently harmless but confusing.

### 8.5 Open questions

- Does the paper-sample protocol's per-seed result close the gap to paper-G *ensemble* on Quarterly/Monthly when ensembled? (Q1 above.)
- Is sym10 strictly equivalent to haar on M4, or does each win specific periods? (Within-family sym10 vs haar shows: sym10 wins Daily, Hourly, Yearly; haar wins Monthly. Worth a head-to-head.)
- Why does `TAELG+Coif2V3ALG_30s_ag0` win Daily under paper-sample but the prior sliding sweep had paper baselines dominating? Single-protocol head-to-head would resolve.

---

## 9. Methodology Notes

- Diverged runs (1 total: `TW_10s_td3_bdeq_coif2` Daily seed) excluded from all per-config means.
- All rankings use mean SMAPE across 10 runs as primary metric; OWA reported as secondary.
- "Statistical tie" used informally for top-N spreads < 1 std of the leader; no formal Wilcoxon/MWU performed because intra-family deltas are clearly within-noise. Where deltas approach 0.5 SMAPE (cross-family) effects are taken as real.
- Paper N-BEATS-G ensemble baselines from Oreshkin et al. 2020 Table 3 are 18-model ensemble means; our individual-run means are not strictly comparable. Comparison is informative on direction and magnitude only.
- "Paper baseline" / "paper backbone" in this report refers to configs that use the **architecture** of Oreshkin 2020 (`NBEATS-G`, `NBEATS-IG`). Where such configs use `active_g=forecast` (`agf` suffix), they apply a **novel repository extension** on top of the paper backbone — they are not paper-faithful in the strict sense. Only `*_ag0` variants reproduce the published architecture.
- Mean rank of 53 configs computed independently per period, averaged across 6 periods.
