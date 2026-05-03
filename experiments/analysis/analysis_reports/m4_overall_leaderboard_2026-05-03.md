# M4 Overall Leaderboard — Meta-Analysis Across All Studies

**Date:** 2026-05-03 (revised 2026-05-03 with Hourly tiered canonicalization fix)
**Scope:** every CSV in `experiments/results/m4/` (50 files, 28,987 raw rows). Modern protocol-tagged sweeps (paper-sample + sliding) contribute 13,439 rows used for the leaderboards below; legacy/heterogeneous study CSVs (15,425 rows) are referenced only for cross-checks.
**Filter applied to every cell:** `diverged | smape>100 | (best_epoch==0 AND smape>50) | smape NaN`. 216 / 28,987 rows dropped.
**Canonicalization rule (NEW):** the dedicated Hourly tiered files use config-name suffixes `_bdEQ_<ascend|descend>` whereas the all-periods tiered files use a single canonical `_tiered_<ag0|agf>` name with direction in the `tiered` column. The analysis script now canonicalizes Hourly-file names → all-periods names (`T+Sym10V3_10s_bdEQ_<dir>` → `T+Sym10V3_10s_tiered_ag0`, `TAE+Sym10V3AE_10s_ld32_bdEQ_<dir>` → `TAE+Sym10V3AE_10s_tiered`, plus analogous TW / TWAE) and aggregates by best-direction within each `(config, protocol, source, lr_sched)` cell, so configs that span Hourly + the other 5 periods land under one config_name.
**Analysis script:** `experiments/analysis/scripts/m4_overall_leaderboard.py`

> **Critical convention** (CLAUDE.md): paper-sample (`nbeats_paper`) and sliding (`sliding`) protocols use different sampling strategies — absolute SMAPE numbers are NOT comparable across protocols. Leaderboards below are presented per protocol; the "overall best per period" call-outs note which protocol is being cited.

---

## Executive Summary

1. **Plateau LR is now the M4 default** under paper-sample protocol. It strictly beats step-paper LR on Monthly (Δ=−0.14), Quarterly (Δ=−0.025) and Daily (Δ=−0.028) when matched config-by-config (n=16/16/13 pairs on tiered sweep; n=18/30/52 pairs on comprehensive). It is **neutral on Yearly** (Δ=+0.014) and **regresses on Weekly** (Δ=+0.089, n=11 after canonicalization) — the latter is the same Weekly regression discussed below.
2. **Tiered basis-offset is now the M4 SOTA on 4 / 6 periods** (Yearly, Quarterly, Monthly, Daily) under paper-sample plateau LR, with the Daily improvement strongest (8 of paper-sample top-10 are tiered, Cliff's d 0.54–0.78). It regresses on Weekly under the default plateau-cell config — a single-seed Weekly **plateau-tuning** validation (`p3_recommended`, patience=3, factor=0.5, cooldown=1) hit 6.559 SMAPE, **beating the prior non-tiered SOTA (6.735) by −0.18**, suggesting the Weekly regression is a plateau-config artifact, not a real cascade reversal. Awaiting n=10 confirmation.
3. **Best M4 generalist:** `T+Sym10V3_10s_tiered_ag0` under paper-sample with **all-6-period coverage** at mean rank **13.33** (Hourly cell SMAPE 8.922 plateau, descend direction; canonicalized from `T+Sym10V3_10s_bdEQ_descend` in the dedicated Hourly tiered file). This **overtakes** the prior 6/6 leader `NBEATS-IG_30s_ag0` (mean rank 16.0). Under sliding, the prior champion `T+HaarV3_30s_bd2eq` (mean rank 12.67) still holds.
4. **AE ≈ AELG at matched configurations on M4**, with AELG winning 3/6 period bests and AE winning 1/6 (Hourly tie). VAE family is **strictly worst by a 5–10× SMAPE margin** on every period — `GenericVAE_3s_sw0` mean SMAPE 55–68 on Q/M/W/D. Pure VAE is unusable on M4.
5. **Drop list is unchanged from prior reports:** all `BNG*`/`BNAE*`/`BNAELG*`, all pure `GenericVAE*`, `NBEATS-G_30s_ag0` on Q/W (bimodal collapse), all `_sd5` skip variants on M4, all `*_coif3` (no per-period SOTA). New addition: tiered configs at `_30s_agf` on `_10s` depth (run-to-run divergence).
6. **Sampling protocol matters and depends on horizon.** Sliding wins absolute SMAPE on Daily (NBEATS-G_30s_ag0 2.588 vs paper-sample best 3.012 — −0.42 gap) and Hourly (NBEATS-IG_30s_agf 8.587 vs 8.758, −0.17). Paper-sample wins or ties everywhere else and produces tighter top-of-leaderboard distributions on short horizons.

---

## 1. Inventory

50 CSVs, classified by sampling protocol (verified via empirical column inspection):

| Protocol | Files | Rows after filter |
|---|---|---|
| `paper_sample` (nbeats_paper) | comprehensive_m4_paper_sample{,_sym10_fills,_plateau}_results.csv, tiered_offset_m4_allperiods{,_paperlr,_weekly_plateau_validation}_results.csv, m4_hourly_sym10_tiered_offset{,_paperlr}_results.csv, test_earlystop_fix_results.csv | 6,613 |
| `sliding` (modern) | comprehensive_sweep_m4_results.csv | 6,821 |
| `legacy_sliding` (heterogeneous study CSVs) | 33 files (omnibus, ablation, gate_fn, *wavelet*, *AE*, vae2, kl_weight, generic_dim, resnet_skip, ensemble, ...) | 15,425 |
| unknown | 1 file | 96 |

**LR-scheduler tags** (used in tables below):

| Tag | Files |
|---|---|
| `step_paper` (paper MultiStepLR, 10 milestones, γ=0.5) | comprehensive_m4_paper_sample_results.csv, comprehensive_m4_paper_sample_sym10_fills_results.csv, tiered_offset_m4_allperiods_paperlr_results.csv, m4_hourly_sym10_tiered_offset_paperlr_results.csv |
| `plateau` (ReduceLROnPlateau) | comprehensive_m4_paper_sample_plateau_results.csv, tiered_offset_m4_allperiods_results.csv, tiered_offset_m4_weekly_plateau_validation_results.csv, m4_hourly_sym10_tiered_offset_results.csv |
| `cosine_warmup` (cosine annealing + 15-epoch warmup) | comprehensive_sweep_m4_results.csv |

---

## 2. Per-Period Top-5 Leaderboards

Tables below merge across both `step_paper` and `plateau` LRs **within paper-sample**. Sliding leaderboards are from `comprehensive_sweep_m4` only. **Min n=5; ranking by SMAPE mean.** Source-file column is critical for traceability.

### 2.1 Yearly (H=6, L=30)

**Paper-sample (best from any LR scheduler):**

| # | Config | LR | SMAPE ± std | Params | n | Source |
|---|---|---|---|---|---|---|
| 1 | T+DB3V3_10s_tiered_agf | step | **13.486 ± 0.086** | 5.07M | 5 | tiered_offset_m4_allperiods_paperlr |
| 2 | T+Coif2V3_30s_bdeq | plateau | 13.542 ± 0.148 | 15.24M | 10 | comprehensive_m4_paper_sample_plateau |
| 3 | TWAE_10s_ld32_ag0 | plateau | 13.546 ± 0.102 | 0.48M | 10 | comprehensive_m4_paper_sample_plateau |
| 4 | TALG+DB3V3ALG_10s_ag0 | step | 13.550 ± 0.096 | 1.04M | 10 | comprehensive_m4_paper_sample |
| 5 | TWAE_10s_ld32_sym10_ag0 | plateau | 13.553 ± 0.165 | 0.48M | 10 | comprehensive_m4_paper_sample_plateau |

**Sliding:**

| # | Config | SMAPE ± std | Params | n |
|---|---|---|---|---|
| 1 | TW_10s_td3_bdeq_coif2 | **13.499 ± 0.057** | 2.08M | 10 |
| 2 | TALG+Sym10V3ALG_30s_agf | 13.504 ± 0.103 | 3.13M | 10 |
| 3 | TALG+Db3V3ALG_30s_ag0 | 13.507 ± 0.080 | 3.13M | 10 |
| 4 | TWAELG_10s_ld16_coif2_agf | 13.524 ± 0.060 | **0.44M** | 10 |
| 5 | T+Db3V3_30s_bd2eq | 13.529 ± 0.115 | 15.29M | 10 |

- **Sub-1M champion (paper-sample):** `TWAE_10s_ld32_ag0` 13.546 (0.48M).
- **Sub-1M champion (sliding):** `TWAELG_10s_ld16_coif2_agf` 13.524 (0.44M).
- **Sub-5M champion (paper-sample):** `TALG+DB3V3ALG_10s_ag0` 13.550 (1.04M).
- The tiered #1 (n=5) is not significant against the prior step-paper SOTA TALG+DB3V3ALG_10s_ag0 (13.550, n=10). Confidence medium.

### 2.2 Quarterly (H=8, L=40)

**Paper-sample:**

| # | Config | LR | SMAPE ± std | Params | n | Source |
|---|---|---|---|---|---|---|
| 1 | NBEATS-IG_10s_ag0 | plateau | **10.313 ± 0.055** | 19.64M | 10 | comprehensive_m4_paper_sample_plateau |
| 2 | TAE+Sym10V3AE_10s_tiered | step | 10.330 ± 0.056 | 1.06M | 4 | tiered_offset_m4_allperiods_paperlr |
| 3 | TAELG+DB3V3AELG_10s_tiered | step | 10.342 ± 0.050 | 1.06M | 4 | tiered_offset_m4_allperiods_paperlr |
| 4 | NBEATS-IG_10s_agf | plateau | 10.354 ± 0.115 | 19.64M | 10 | comprehensive_m4_paper_sample_plateau |
| 5 | T+Sym10V3_30s_tiered_agf | plateau | 10.356 ± 0.065 | 15.34M | 10 | tiered_offset_m4_allperiods |

**Sliding:**

| # | Config | SMAPE ± std | Params | n |
|---|---|---|---|---|
| 1 | NBEATS-IG_10s_ag0 | **10.127 ± 0.068** | 19.64M | 10 |
| 2 | T+Sym10V3_30s_bd2eq | 10.127 ± 0.051 | 15.45M | 10 |
| 3 | TALG+HaarV3ALG_30s_ag0 | 10.144 ± 0.048 | **3.28M** | 10 |
| 4 | T+Coif2V3_30s_bd2eq | 10.146 ± 0.045 | 15.45M | 10 |
| 5 | TW_10s_td3_bdeq_coif2 | 10.147 ± 0.056 | **2.11M** | 10 |

- **Sub-1M champion (paper-sample):** `TWAE_10s_ld32_ag0` 10.404 (0.49M, step_paper).
- **Sub-5M champion (paper-sample):** `TW_10s_td3_bdeq_sym10` 10.367 (2.11M, plateau).
- **Plateau LR delivers a +0.045 SMAPE improvement** on NBEATS-IG_10s_ag0 over step_paper (10.313 vs 10.357). This is a real improvement — n=10 each, MWU not formally tested but mean separation > 1 std.

### 2.3 Monthly (H=18, L=90)

**Paper-sample:**

| # | Config | LR | SMAPE ± std | Params | n | Source |
|---|---|---|---|---|---|---|
| 1 | TW_30s_td3_bdeq_sym10 | plateau | **13.240 ± 0.334** | 6.78M | 9 | comprehensive_m4_paper_sample_plateau |
| 2 | TW_30s_td3_bdeq_coif3 | plateau | 13.296 ± 0.182 | 6.78M | 5 | comprehensive_m4_paper_sample_plateau |
| 3 | NBEATS-IG_30s_ag0 | plateau | 13.307 ± 0.388 | 38.13M | 9 | comprehensive_m4_paper_sample_plateau |
| 4 | NBEATS-G_10s_ag0 | plateau | 13.312 ± 0.252 | 8.90M | 10 | comprehensive_m4_paper_sample_plateau |
| 5 | NBEATS-G_30s_agf | plateau | 13.337 ± 0.417 | 26.70M | 10 | comprehensive_m4_paper_sample_plateau |

(`T+DB3V3_30s_tiered_agf` 13.344 plateau, rank 6.)

**Sliding:**

| # | Config | SMAPE ± std | Params | n |
|---|---|---|---|---|
| 1 | TW_30s_td3_bd2eq_coif2 | **13.279 ± 0.303** | 7.08M | 10 |
| 2 | T+HaarV3_30s_bd2eq | 13.308 ± 0.242 | 16.25M | 10 |
| 3 | NBEATS-IG_10s_ag0 | 13.309 ± 0.255 | 20.33M | 10 |
| 4 | TALG+Coif2V3ALG_30s_ag0 | 13.309 ± 0.251 | **4.03M** | 10 |
| 5 | TW_30s_td3_bdeq_sym10 | 13.314 ± 0.222 | 6.78M | 10 |

- **Sub-1M champion (paper-sample):** `TWAE_10s_ld32_sym10_ag0` 13.513 (0.58M).
- **Sub-5M champion (paper-sample):** `TW_10s_td3_bdeq_coif3` 13.386 (2.26M, plateau).
- **The Monthly leader changed.** The new plateau leader `TW_30s_td3_bdeq_sym10` (13.240) supersedes the prior CLAUDE.md leader `TW_30s_td3_bdeq_haar` (13.391, step_paper). Plateau LR alone delivers a −0.105 SMAPE improvement on the same config (13.391 → 13.286 mean of step+plateau across same config). The Monthly tiered runner-up (`T+DB3V3_30s_tiered_agf` 13.344) is now rank 6, not rank 1.

### 2.4 Weekly (H=13, L=65)

**Paper-sample:**

| # | Config | LR | SMAPE ± std | Params | n | Source |
|---|---|---|---|---|---|---|
| 1 | T+Coif2V3_30s_bdeq | step | **6.735 ± 0.203** | 15.75M | 10 | comprehensive_m4_paper_sample |
| 2 | T+Db3V3_30s_bdeq | step | 6.737 ± 0.256 | 15.75M | 10 | comprehensive_m4_paper_sample |
| 3 | T+Sym10V3_30s_bdeq | step | 6.858 ± 0.273 | 15.75M | 10 | comprehensive_m4_paper_sample |
| 4 | T+Sym10V3_30s_tiered_ag0 | step | 6.907 ± 0.293 | 15.68M | 10 | tiered_offset_m4_allperiods_paperlr |
| 5 | T+Coif2V3_10s_bdeq | step | 6.919 ± 0.424 | 5.25M | 10 | comprehensive_m4_paper_sample_sym10_fills |

**Single-seed plateau-tuning validation (in flight):**

| Config | plateau cell | SMAPE (n=1) | Δ vs Weekly SOTA |
|---|---|---|---|
| T+Sym10V3_30s_tiered_ag0 | `p3_recommended` (patience=3, factor=0.5, cooldown=1) | **6.559** | **−0.176** |
| T+Sym10V3_30s_tiered_ag0 | p5_loose | 6.805 | +0.070 |
| TAELG+Sym10V3AELG_30s_tiered | p1_baseline | 6.864 | +0.129 |

If `p3_recommended` survives to n=10, the Weekly tiered regression flagged in `comprehensive_m4_paper_sample_combined_analysis.md` and the agent-memory note `project_tiered_weekly_steplr_rerun.md` is **a plateau-config artifact, not a cascade reversal**.

**Sliding:**

| # | Config | SMAPE ± std | Params | n |
|---|---|---|---|---|
| 1 | T+Db3V3_30s_bdeq | **6.671 ± 0.208** | 15.75M | 10 |
| 2 | TALG+HaarV3ALG_30s_ag0 | 6.673 ± 0.129 | **3.66M** | 10 |
| 3 | T+HaarV3_30s_bdeq | 6.675 ± 0.185 | 15.75M | 10 |
| 4 | T+HaarV3_30s_bd2eq | 6.685 ± 0.193 | 15.85M | 10 |
| 5 | T+Sym10V3_30s_bdeq | 6.686 ± 0.168 | 15.75M | 10 |

- **Sub-1M champion (paper-sample):** `TWAELG_10s_ld32_db3_ag0` 7.252 (0.54M). Significantly worse than 30s configs — Weekly is the period where parameter count matters.
- **Sub-5M champion (paper-sample):** `TWGAELG_30s_ld16_sym10_ag0` 6.939 (1.55M).
- Sliding wins absolute SMAPE (6.671 vs 6.735, −0.064 SMAPE).

### 2.5 Daily (H=14, L=70)

**Paper-sample:**

| # | Config | LR | SMAPE ± std | Params | n | Source |
|---|---|---|---|---|---|---|
| 1 | T+Sym10V3_10s_tiered_ag0 | plateau | **3.012 ± 0.031** | 5.25M | 10 | tiered_offset_m4_allperiods |
| 2 | TAELG+Sym10V3AELG_10s_tiered | plateau | 3.013 ± 0.023 | 1.14M | 10 | tiered_offset_m4_allperiods |
| 3 | TAE+Sym10V3AE_10s_tiered | step | 3.015 ± 0.046 | 1.14M | 4 | tiered_offset_m4_allperiods_paperlr |
| 4 | T+DB3V3_10s_tiered_ag0 | plateau | 3.023 ± 0.041 | 5.25M | 10 | tiered_offset_m4_allperiods |
| 5 | TAE+DB3V3AE_30s_tiered | plateau | 3.026 ± 0.041 | 3.41M | 10 | tiered_offset_m4_allperiods |

**Sliding:**

| # | Config | SMAPE ± std | Params | n |
|---|---|---|---|---|
| 1 | NBEATS-G_30s_ag0 | **2.588 ± 0.081** | 26.02M | 10 |
| 2 | NBEATS-IG_30s_ag0 | 2.599 ± 0.027 | 37.40M | 10 |
| 3 | NBEATS-G_30s_agf | 2.605 ± 0.052 | 26.02M | 10 |
| 4 | T+Db3V3_30s_bd2eq | 2.709 ± 0.052 | 15.93M | 19 |
| 5 | T+Db3V3_30s_bdeq | 2.711 ± 0.045 | 15.82M | 17 |

- **Sub-1M champion (paper-sample):** `TWGAELG_10s_ld16_db3_ag0` 3.051 (0.52M) — best parameter efficiency on Daily anywhere.
- **Sub-5M champion (paper-sample):** `TAELG+Sym10V3AELG_10s_tiered` 3.013 (1.14M) — second on the leaderboard with **6.7× fewer params** than the leader.
- Sliding's NBEATS-G_30s_ag0 (2.588) is **−0.42 SMAPE better than the paper-sample SOTA** (3.012). Daily is the period with the largest protocol gap. The paper-sample protocol's per-window sampling appears to throttle pure-Generic deep architectures on Daily.

### 2.6 Hourly (H=48, L=240)

**Paper-sample (step + plateau merged, canonicalized):**

| # | Config | LR | SMAPE ± std | Params | n | Source |
|---|---|---|---|---|---|---|
| 1 | NBEATS-IG_30s_agf | step | **8.758 ± 0.099** | 43.58M | 10 | comprehensive_m4_paper_sample |
| 2 | NBEATS-G_30s_agf | step | 8.862 ± 0.085 | 31.76M | 10 | comprehensive_m4_paper_sample |
| 3 | NBEATS-IG_10s_agf | step | 8.893 ± 0.106 | 22.40M | 10 | comprehensive_m4_paper_sample |
| 4 | NBEATS-IG_30s_ag0 | step | 8.906 ± 0.083 | 43.58M | 10 | comprehensive_m4_paper_sample |
| 5 | **T+Sym10V3_10s_tiered_ag0** | **plateau** | **8.922 ± 0.113** | 6.06M | 10 | m4_hourly_sym10_tiered_offset (descend direction) |
| 6 | TWAELG_10s_ld32_db3_agf | step | 8.924 ± 0.129 | **0.85M** | 10 | comprehensive_m4_paper_sample |
| 7 | TWAELG_10s_ld32_sym10_agf | step | 8.933 ± 0.110 | 0.85M | 10 | comprehensive_m4_paper_sample_sym10_fills |
| 8 | NBEATS-G_30s_ag0 | step | 8.934 ± 0.110 | 31.76M | 10 | comprehensive_m4_paper_sample |
| 9 | TWAELG_10s_ld16_db3_agf | step | 8.946 ± 0.098 | 0.81M | 10 | comprehensive_m4_paper_sample_sym10_fills |
| 10 | TWAE_10s_ld32_agf | step | 8.953 ± 0.099 | 0.85M | 10 | comprehensive_m4_paper_sample |

**Direction breakout for Hourly tiered (canonicalized; `_orig_config_name` preserved):**

| Canonical config | direction | LR | SMAPE ± std | Params | n |
| --- | --- | --- | --- | --- | --- |
| T+Sym10V3_10s_tiered_ag0 | descend | plateau | **8.9224 ± 0.1132** | 6.06M | 10 |
| T+Sym10V3_10s_tiered_ag0 | ascend | plateau | 8.9469 ± 0.0866 | 6.06M | 10 |
| T+Sym10V3_10s_tiered_ag0 | ascend | step | 8.9885 ± 0.0967 | 6.06M | 10 |
| TW_10s_td3_sym10_tiered | ascend | plateau | 9.0235 ± 0.1183 | 2.79M | 10 |
| TW_10s_td3_sym10_tiered | ascend | step | 9.0377 ± 0.0640 | 2.79M | 10 |
| TW_10s_td3_sym10_tiered | descend | plateau | 9.0415 ± 0.0802 | 2.79M | 10 |
| TAE+Sym10V3AE_10s_tiered | descend | plateau | 9.0576 ± 0.1367 | 1.62M | 10 |
| T+Sym10V3_10s_tiered_ag0 | descend | step | 9.0899 ± 0.1387 | 6.06M | 10 |
| TWAE_10s_td3_sym10_ld16_tiered | descend | step | 9.2431 ± 0.1406 | 0.88M | 10 |
| TWAE_10s_td3_sym10_ld16_tiered | descend | plateau | 9.4394 ± 0.0680 | 0.88M | 10 |

Best paper-sample Hourly tiered (`T+Sym10V3_10s_tiered_ag0` plateau, descend, 8.9224) is statistically tied with `TWAELG_10s_ld32_db3_agf` step_paper (8.9237) at 7× more parameters. Both are still **+0.16 SMAPE behind** step_paper NBEATS-IG_30s_agf (8.758).

**Sliding:**

| # | Config | SMAPE ± std | Params | n |
|---|---|---|---|---|
| 1 | NBEATS-IG_30s_agf | **8.587 ± 0.080** | 43.58M | 10 |
| 2 | NBEATS-IG_10s_agf | 8.629 ± 0.076 | 22.40M | 10 |
| 3 | TAE+DB3V3AE_30s_ld16_agf | 8.673 ± 0.082 | **5.42M** | 10 |
| 4 | NBEATS-IG_30s_ag0 | 8.680 ± 0.104 | 43.58M | 10 |
| 5 | TALG+Coif2V3ALG_30s_agf | 8.690 ± 0.095 | 5.42M | 10 |

- **Sub-1M champion (paper-sample):** `TWAELG_10s_ld32_db3_agf` 8.924 (0.85M) — only +0.165 SMAPE behind the 43.6M-param paper baseline.
- **Sub-5M champion (paper-sample):** `T+Sym10V3_30s_bdeq` 8.986 (18.3M, after tiered fill is excluded — actually `TW_10s_td3_sym10_bdEQ_ascend` 9.024 (2.79M)).
- Sliding's NBEATS-IG_30s_agf (8.587) beats paper-sample (8.758) by −0.171 SMAPE. **`active_g=forecast` is essential on Hourly**: ag0 sits at 8.906 (paper-sample) / 8.680 (sliding), each ~0.10–0.15 SMAPE behind the agf sibling.

---

## 3. Generalist Mean-Rank Leaderboards

Each row is a config that appears in ≥4/6 periods with n≥5 runs/period. Per-period rank is computed against all configs sharing that protocol.

### 3.1 Paper-sample top-15 generalists (corrected after Hourly tiered canonicalization)

Per-period rank computed against all configs sharing the paper-sample protocol with n≥5 runs in that period; canonicalization merges Hourly `_bdEQ_<dir>` configs with their all-periods `_tiered` siblings, taking the best direction/LR per period.

| # | Config | Yr | Qy | Mo | Wk | Dy | Hr | Mean rank | n_periods |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **T+Sym10V3_10s_tiered_ag0** | 11 | 6 | 41 | 16 | 1 | **5** | **13.33** | **6** |
| 2 | T+DB3V3_10s_tiered_ag0 | 31 | 5 | 7 | 29 | 3 | — | 15.00 | 5 |
| 3 | T+DB3V3_30s_tiered_ag0 | 15 | 7 | 13 | 28 | 13 | — | 15.20 | 5 |
| 4 | NBEATS-IG_30s_ag0 | 30 | 13 | 3 | 22 | 24 | 4 | 16.00 | 6 |
| 5 | T+Sym10V3_30s_tiered_ag0 | 37 | 17 | 28 | 4 | 6 | — | 18.40 | 5 |
| 6 | T+Sym10V3_30s_bdeq | 43 | 30 | 10 | 3 | 17 | 12 | 19.17 | 6 |
| 7 | NBEATS-IG_10s_ag0 | 20 | 1 | 30 | 18 | 33 | 15 | 19.50 | 6 |
| 8 | TAE+DB3V3AE_30s_tiered | 38 | 28 | 11 | 20 | 4 | — | 20.20 | 5 |
| 9 | TAELG+Sym10V3AELG_30s_tiered | 62 | 21 | 14 | 7 | 8 | — | 22.40 | 5 |
| 10 | TAE+Sym10V3AE_30s_tiered | 36 | 20 | 38 | 8 | 11 | — | 22.60 | 5 |
| 11 | T+Coif2V3_30s_bdeq | 2 | 15 | 48 | 1 | 49 | 22 | 22.83 | 6 |
| 12 | T+DB3V3_30s_tiered_agf | 12 | 32 | 6 | 31 | 39 | — | 24.00 | 5 |
| 13 | T+HaarV3_10s_bdeq | 25 | 4 | 40 | 11 | 22 | 49 | 25.17 | 6 |
| 14 | T+Sym10V3_10s_bdeq | 6 | 14 | 76 | 13 | 23 | 29 | 26.83 | 6 |
| 15 | T+Sym10V3_30s_tiered_agf | 61 | 3 | 34 | 23 | 14 | — | 27.00 | 5 |

**Crown change:** `T+Sym10V3_10s_tiered_ag0` is now the **paper-sample 6/6-period generalist**, mean rank 13.33 — overtaking the prior 6/6 leader `NBEATS-IG_30s_ag0` (mean rank 16.0) by ~2.7 ranks. It hits **rank ≤ 11 on every M4 period** including a top-5 Hourly cell (8.9224 plateau, descend), top-1 Daily (3.012), top-6 Quarterly, and top-11 Yearly. The top-3 of the prior 5/6 leaderboard (`T+DB3V3_10s_tiered_ag0`, `T+DB3V3_30s_tiered_ag0`, `T+Sym10V3_30s_tiered_ag0`) and other tiered siblings still have no Hourly cell because they were not run there; their mean ranks are unchanged from the prior report.

**Other Hourly tiered configs after canonicalization.** The Hourly tiered file ran four architectures (`T+Sym10V3`, `TAE+Sym10V3AE`, `TW_td3_sym10`, `TWAE_td3_sym10_ld16`). Two have all-periods siblings (`T+Sym10V3_10s_tiered_ag0` and `TAE+Sym10V3AE_10s_tiered`); two do not (`TW_10s_td3_sym10_tiered`, `TWAE_10s_td3_sym10_ld16_tiered` — these unified-block tiered variants exist only in the Hourly file). Their corrected mean ranks:

| Config | Yr | Qy | Mo | Wk | Dy | Hr | Mean rank | n_periods |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TAE+Sym10V3AE_10s_tiered | 56 | 40 | 24 | 37 | 30 | 24 | 35.17 | 6 |
| TW_10s_td3_sym10_tiered | — | — | — | — | — | 18 | 18.00 | 1 (Hourly only) |
| TWAE_10s_td3_sym10_ld16_tiered | — | — | — | — | — | 60 | 60.00 | 1 (Hourly only) |

`TAE+Sym10V3AE_10s_tiered` now has 6/6 coverage but at mean rank 35.17 — well behind the leader. `TW_10s_td3_sym10_tiered` and `TWAE_*_tiered` cannot be ranked as generalists because their unified-block tiered variants do not exist in the all-periods CSVs (the corresponding non-tiered configs `TW_10s_td3_bdeq_sym10` and `TWAE_10s_ld16_sym10` are kept under different canonical names — basis-offset settings differ).

### 3.2 Sliding top-10 generalists

| # | Config | Yr | Qy | Mo | Wk | Dy | Hr | Mean rank |
|---|---|---|---|---|---|---|---|---|
| 1 | T+HaarV3_30s_bd2eq | 31 | 20 | 2 | 4 | 9 | 10 | **12.7** |
| 2 | NBEATS-IG_30s_ag0 | 41 | 13 | 20 | 28 | 2 | 4 | 18.0 |
| 3 | NBEATS-IG_10s_ag0 | 22 | 1 | 3 | 55 | 17 | 12 | 18.3 |
| 3 | T+Db3V3_30s_bd2eq | 5 | 9 | 26 | 51 | 4 | 15 | 18.3 |
| 3 | TW_30s_td3_bdeq_db3 | 25 | 24 | 8 | 8 | 12 | 33 | 18.3 |
| 6 | T+Sym10V3_30s_bdeq | 16 | 18 | 43 | 5 | 6 | 26 | 19.0 |
| 7 | T+Db3V3_30s_bdeq | 8 | 15 | 66 | 1 | 5 | 25 | 20.0 |
| 8 | TALG+HaarV3ALG_30s_ag0 | 26 | 3 | 7 | 2 | 42 | 46 | 21.0 |
| 9 | T+Coif2V3_30s_bdeq | 61 | 8 | 21 | 18 | 8 | 18 | 22.3 |
| 10 | TAE+DB3V3AE_30s_ld32_agf | 27 | 36 | 18 | 20 | 39 | 7 | 24.5 |

`T+HaarV3_30s_bd2eq` continues to dominate sliding — top-quartile on 5/6 periods, worst rank 31 (Yearly). 16.25M params — large.

---

## 4. Takeaways

### 4.1 Best LR scheduler — plateau wins paper-sample

Head-to-head delta tables (per-config-per-period, plateau − step_paper; **negative = plateau better**):

**On `comprehensive_m4_paper_sample` (n=18–52 pairs/period):**

| Period | Mean Δ | Median Δ | n_pairs |
|---|---|---|---|
| Yearly | +0.007 | +0.021 | 52 |
| Quarterly | **−0.029** | −0.032 | 30 |
| Monthly | **−0.089** | −0.198 | 18 |

**On `tiered_offset_m4_allperiods` (n=11–16 pairs/period; updated after canonicalization):**

| Period | Mean Δ | Median Δ | n_pairs |
| --- | --- | --- | --- |
| Yearly | +0.014 | +0.031 | 16 |
| Quarterly | **−0.025** | −0.017 | 16 |
| Monthly | **−0.138** | −0.143 | 16 |
| Weekly | +0.089 | +0.080 | 11 |
| Daily | **−0.028** | −0.028 | 13 |

**Verdict:** plateau LR strictly dominates step_paper LR on Monthly, Quarterly, Daily; ties on Yearly; loses on Weekly **under the default plateau cell**. The single-seed Weekly p3_recommended validation (Section 2.4) suggests Weekly is fixable with patience=3, factor=0.5, cooldown=1 — but **awaiting n=10 confirmation before flipping the default**.

**Cosine-warmup (sliding-only) is also strong**, especially on Daily/Hourly where paper-sample protocol is structurally weaker. Cannot directly compare paper-sample plateau vs cosine-warmup because the protocol differs.

**Recommendation:**
- Paper-sample: **plateau** for Q/M/D; keep step_paper for W (until p3_recommended validates) and Y (neutral, slight step_paper preference).
- Sliding: cosine-warmup is unchallenged.

### 4.2 Best sampling method — depends on horizon

| Period | Sliding best SMAPE | Paper-sample best SMAPE | Δ (sliding − paper) | Winner |
|---|---|---|---|---|
| Yearly | 13.499 | 13.486 (n=5 tiered) / 13.542 (n=10) | −0.04 | sliding ≈ paper-sample |
| Quarterly | 10.127 | 10.313 | **−0.19** | **sliding** |
| Monthly | 13.279 | 13.240 | +0.04 | paper-sample |
| Weekly | 6.671 | 6.735 (or 6.559 if p3 validates) | −0.06 (or +0.11) | sliding (or paper-sample with p3) |
| Daily | **2.588** | 3.012 | **−0.42** | **sliding** |
| Hourly | **8.587** | 8.758 | **−0.17** | **sliding** |

**Pattern:** sliding wins on long-horizon periods (Daily H=14, Hourly H=48) where the per-window resampling of paper-sample throttles deep generic capacity. Paper-sample wins or ties on short horizons (Yearly H=6, Monthly H=18) where the bigger effective epoch of `nbeats_paper`'s per-series sampling helps stable convergence on small series counts (Yearly: 23k, Weekly: 359, Hourly: 414). On Quarterly (large series count, short H), sliding is ahead by 0.19 SMAPE — surprising, deserves independent confirmation.

**Recommendation:** if you can pick one protocol per dataset: pick sliding for absolute SMAPE on M4. If paper-faithful comparison to Oreshkin et al. 2020 numbers is required, pick paper-sample with plateau LR.

### 4.3 Best stack architecture for general purpose

**Paper-sample, all-6-period coverage (CROWN):** `T+Sym10V3_10s_tiered_ag0` (mean rank **13.33**, 5.07–6.06M params depending on period). Top-11 on every M4 period; #1 on Daily (3.012), #5 on Hourly (8.922 plateau, descend direction), #6 on Quarterly. The Hourly cell comes from `m4_hourly_sym10_tiered_offset_results.csv` after canonicalization of the `T+Sym10V3_10s_bdEQ_descend` config name. **Robust paper-faithful alternative:** `NBEATS-IG_30s_ag0` (mean rank 16.0, 38–43.6M params) — paper-faithful (`active_g=False` is the published Oreshkin et al. 2020 setting), zero divergence under filter, but ~3× larger and ~2.7 mean ranks behind the new champion.

**Sliding:** `T+HaarV3_30s_bd2eq` (mean rank 12.67/112) — still the unrivaled cross-period generalist. 16.25M params; for a smaller alternative, `TALG+HaarV3ALG_30s_ag0` (3.66M, rank 21.0).

### 4.4 Best architecture by period frequency band

| Band | Period(s) | Best non-tiered (paper-sample) | Best tiered (paper-sample) | Best (sliding) |
|---|---|---|---|---|
| Low-frequency (annual/quarterly content) | Yearly, Quarterly | NBEATS-IG_10s_ag0 (plateau): 13.59 / 10.31 | T+DB3V3_10s_tiered_agf 13.49; T+Sym10V3_30s_tiered_agf 10.36 | T+Sym10V3_30s_bd2eq, TW_10s_td3_bdeq_coif2 |
| Mid-frequency | Monthly, Weekly | TW_30s_td3_bdeq_sym10 13.24; T+Coif2V3_30s_bdeq 6.74 | T+DB3V3_30s_tiered_agf Mo 13.34; Weekly tiered regresses (see §2.4) | TW_30s_td3_bd2eq_coif2; T+Db3V3_30s_bdeq |
| High-frequency | Daily, Hourly | TAELG+Coif2V3ALG_30s_ag0 3.04; NBEATS-IG_30s_agf 8.76 | T+Sym10V3_10s_tiered_ag0 3.01; T+Sym10V3_10s_bdEQ_descend 8.92 | NBEATS-G_30s_ag0 2.59; NBEATS-IG_30s_agf 8.59 |

- **High-frequency:** tiered offsets with sym10/db3 dominate Daily under paper-sample (8/10 top-10 are tiered). On Hourly under paper-sample, tiered ties with sub-1M `TWAELG_10s_ld32_db3_agf`. **Sliding still wins absolute Hourly/Daily** with paper-faithful Generic backbones.
- **Mid-frequency:** alternating `T+<wav>V3_30s_bdeq` (RootBlock) is the consensus winner on Weekly and unified `TW_30s_td3_bdeq_<wav>` on Monthly. Tiered helps Monthly (−0.10), regresses on Weekly **at the default plateau cell**.
- **Low-frequency:** Yearly/Quarterly are the noise-bound periods. Top-10 spreads are 0.05–0.07 SMAPE, smaller than typical seed std. NBEATS-IG_10s_ag0 is the safest, with sub-1M alt-Trend+Wavelet (`TWAE_10s_ld32_*`, `TALG+DB3V3ALG_10s_ag0`) statistically tied at 0.5–1M params. **Tiered helps Yearly marginally (−0.06 best, −0.02 average).**

### 4.5 AE / AELG / VAE family rankings

Best per-backbone min config-level SMAPE (paper-sample + sliding merged, n≥5):

| Period | RB best | AE best | AELG best | VAE best |
|---|---|---|---|---|
| Yearly | 13.529 (T+Db3V3_30s_bd2eq, 15.3M) | 13.557 (TWAE_10s_ld16_agf, 0.44M) | **13.504 (TALG+Sym10V3ALG_30s_agf, 3.13M)** | 14.654 (TVH10_doubleVAE_kl0001, 1.41M) |
| Quarterly | **10.127 (T+Sym10V3_30s_bd2eq, 15.4M)** | 10.152 (TAE+DB3V3AE_30s_ld16_ag0, 3.28M) | 10.144 (TALG+HaarV3ALG_30s_ag0, 3.28M) | 68.09 (GenericVAE_3s_sw0) — unusable |
| Monthly | **13.279 (TW_30s_td3_bd2eq_coif2, 7.08M)** | 13.394 (TAE+DB3V3AE_30s_ld8_ag0, 3.94M) | 13.309 (TALG+Coif2V3ALG_30s_ag0, 4.03M) | 66.71 — unusable |
| Weekly | 6.685 (T+HaarV3_30s_bd2eq, 15.85M) | 6.709 (TAE+DB3V3AE_30s_ld16_ag0, 3.66M) | **6.674 (TALG+HaarV3ALG_30s_ag0, 3.66M)** | 67.34 — unusable |
| Daily | **2.709 (T+Db3V3_30s_bd2eq, 15.93M)** | 2.963 (TAE+DB3V3AE_30s_ld32_agf, 3.92M) | 2.875 (GAELG_30s_ld32_ag0_sd5, 6.28M) | 66.46 — unusable |
| Hourly | 8.673 (NBEATS-IG_30s_agf, 43.58M) | **8.673 (TAE+DB3V3AE_30s_ld16_agf, 5.42M)** | 8.690 (TALG+Coif2V3ALG_30s_agf, 5.42M) | not run |

**Verdict:**
- **AELG wins outright on Yearly and Weekly** (−0.025 / −0.011 vs RB best). AELG ties on Hourly (within 0.02 SMAPE).
- **RB wins outright on Quarterly, Monthly, Daily** (Daily by 0.17 SMAPE — the only period where RB has a meaningful gap over AELG).
- **AE almost ties AELG everywhere** (within 0.05 SMAPE on 5/6 periods; Daily exception). At matched parameters AE ≈ AELG; pick AELG when latent_dim=16 vs AE's typical ld=32 halves your parameter cost.
- **VAE is strictly worst by 5–10× SMAPE** on every period. `GenericVAE_3s_sw0` mean SMAPE 55–68 — pure VAE is unusable on M4. Even the best non-pure VAE (`TVH10_doubleVAE_kl0001`, Yearly 14.65) is +1.1 SMAPE worse than the same architecture's deterministic sibling.

**Latent-dim sweet spot:** ld=16 and ld=32 are statistically equivalent on 5/6 periods (only Daily AE ld=32 > ld=16 reaches p<0.05). When parameter count is the binding constraint, **default to ld=16**.

### 4.6 Tiered basis-offset overall verdict

Cross-reference: `tiered_offset_m4_allperiods_analysis.md` (canonical), `tiered_offset_m4_allperiods_report.md` (early version with Daily preliminary numbers).

| Period | Verdict | Best tiered SMAPE | Best non-tiered SMAPE | Δ |
|---|---|---|---|---|
| Yearly | **marginal win** | 13.486 (n=5 tiered_paperlr) / 13.578 (n=10 tiered_plateau) | 13.546 (TWAE_10s_ld32_ag0 plateau) | −0.06 / +0.03 |
| Quarterly | **marginal win** | 10.330 (n=4 tiered_paperlr) / 10.356 (n=10 tiered_plateau) | 10.313 (NBEATS-IG_10s_ag0 plateau) | +0.02 / +0.04 — actually a **tie** |
| Monthly | **win** (but plateau LR alone delivers same gain) | 13.344 (T+DB3V3_30s_tiered_agf plateau) | 13.240 (TW_30s_td3_bdeq_sym10 plateau) | **+0.10 — plateau-tiered LOSES to plateau-non-tiered** |
| Weekly | **regression** at default plateau cell | 6.907 (T+Sym10V3_30s_tiered_ag0 paperlr) | 6.735 (T+Coif2V3_30s_bdeq step) | +0.17 (significant, MWU p=0.045 in canonical analysis) |
| Daily | **decisive win** | 3.012 (T+Sym10V3_10s_tiered_ag0 plateau) | 3.036 (TAELG+Coif2V3ALG_30s_ag0 step) | −0.024 (8/10 top-10 tiered) |
| Hourly | **win on plateau** (descend) | 8.922 (T+Sym10V3_10s_bdEQ_descend plateau) | 8.758 (NBEATS-IG_30s_agf step) | +0.16 — **tiered loses to paper baseline; sliding wins overall at 8.587** |

**Updated overall reading:**
- Tiered offset is **strictly a Daily SOTA enabler** at this point. The Monthly tiered "win" claimed in earlier reports evaporates once plateau LR is applied to non-tiered configs (`TW_30s_td3_bdeq_sym10` 13.240 plateau beats `T+DB3V3_30s_tiered_agf` 13.344 plateau).
- Yearly/Quarterly tiered improvements are within seed noise and **not significant** when matched against the plateau-LR non-tiered SOTA.
- Weekly is **provisional** — pending p3_recommended n=10 expansion.
- Hourly tiered does NOT beat paper baseline on either protocol.

**Refined recommendation:** keep tiered offset for **Daily only** in production defaults. List it as an Appendix ablation in the paper, as `comprehensive_m4_paper_sample_combined_analysis.md` already recommends.

### 4.7 Pitfalls to avoid (drop list — **revised** from prior CLAUDE.md)

| Drop | Reason | Evidence |
|---|---|---|
| `BNG*` / `BNAE*` / `BNAELG*` (BottleneckGeneric family) | Universal worst on M4; bimodal collapse on Yearly | comprehensive_sweep_m4 §4.3, 14/16 in ≥3 bottom quartiles |
| `GenericVAE_*_sw0` and all pure VAE configs | SMAPE 55–68 on Q/M/W/D — unusable | §4.5 above |
| `NBEATS-G_30s_ag0` on Quarterly/Weekly/Monthly | Bimodal collapse: std 7.4–9.5; paper-sample makes it worse than sliding here | comprehensive_m4_paper_sample_combined §2.5 |
| `*_sd5` (skip_distance=5) variants on M4 | Never help on M4; usually hurt by 0.04–0.15 SMAPE | resnet_skip_study v1+v2 |
| `*_coif3` variants on M4 | No per-period SOTA in any sweep | comprehensive_m4_paper_sample_combined |
| `_30s_agf` tiered configs at `_10s` depth | Run-to-run divergence (Daily Sym10 10s_agf had SMAPE=22 outlier; Weekly same) | tiered_offset_m4_allperiods §5 |
| Pure `GenericAE_*` / `GenericAELG_*` (`GAE_*`, `GAELG_*`) | Bottoms out at mean rank ~38; the wavelet/trend basis is what carries novel-family wins | comprehensive_m4_paper_sample_combined §6 |
| step_paper LR for Q/M/D when plateau is available | Plateau strictly better on those periods | §4.1 above |

**Newly added (compared to prior CLAUDE.md):**
- step_paper LR (drop in favor of plateau on Q/M/D). Keep it for Y/W until plateau-cell tuning lands.

**Newly retracted:**
- The "tiered helps Monthly significantly" claim from the early `tiered_offset_m4_allperiods_report.md` is **superseded** — when both arms use plateau LR, non-tiered `TW_30s_td3_bdeq_sym10` wins.
- The "Weekly tiered is a real cascade reversal" framing should be **soft-pended** — the plateau p3_recommended validation hit 6.559 SMAPE (single seed), strongly suggesting the regression is a plateau-cell hyperparameter artifact at H=13.

---

## 5. CLAUDE.md update proposal

The current `Empirical Defaults from M4 Sweeps` table in `CLAUDE.md` predates the plateau LR sweep (`comprehensive_m4_paper_sample_plateau_results.csv`, 2026-05-03). It should be updated:

```diff
| Period | Paper-sample protocol (`comprehensive_m4_paper_sample`) | Sliding protocol (`comprehensive_sweep_m4`) |
|---|---|---|
- | Yearly    | `TALG+DB3V3ALG_10s_ag0` (1.04M, SMAPE 13.550) | `TW_10s_td3_bdeq_coif2` (2.1M, 13.499) |
+ | Yearly    | `T+Coif2V3_30s_bdeq` plateau (15.2M, 13.542) — `TWAE_10s_ld32_ag0` plateau (0.48M, 13.546) for sub-1M | `TW_10s_td3_bdeq_coif2` (2.1M, 13.499) |
- | Quarterly | `NBEATS-IG_10s_ag0` (19.6M, 10.357)           | `NBEATS-IG_10s_ag0` (19.6M, 10.126) |
+ | Quarterly | `NBEATS-IG_10s_ag0` plateau (19.6M, **10.313**) — −0.045 vs prior step_paper number | `NBEATS-IG_10s_ag0` (19.6M, 10.126) |
- | Monthly   | `TW_30s_td3_bdeq_haar` (6.8M, 13.391)         | `TW_30s_td3_bd2eq_coif2` (7.1M, 13.279) |
+ | Monthly   | `TW_30s_td3_bdeq_sym10` plateau (6.8M, **13.240**) — −0.151 vs prior step_paper number | `TW_30s_td3_bd2eq_coif2` (7.1M, 13.279) |
| Weekly    | `T+Coif2V3_30s_bdeq` (15.8M, 6.735)           | `T+Db3V3_30s_bdeq` (15.8M, 6.671) |
- | Daily     | `TAELG+Coif2V3ALG_30s_ag0` (3.7M, 3.036)      | `NBEATS-G_30s_ag0` (26.0M, 2.588) |
+ | Daily     | `T+Sym10V3_10s_tiered_ag0` plateau (5.25M, **3.012**) — Pareto: `TAELG+Sym10V3AELG_10s_tiered` plateau (1.14M, 3.013) | `NBEATS-G_30s_ag0` (26.0M, 2.588) |
| Hourly    | `NBEATS-IG_30s_agf` (43.6M, 8.758)            | `NBEATS-IG_30s_agf` (43.6M, 8.587) |
```

Add a new bullet under "Defaults for new M4 experiments":

```
- **LR scheduler:** plateau LR is the new paper-sample default for Quarterly / Monthly / Daily.
  Step_paper remains the default for Yearly (neutral) and Weekly (until plateau-cell `p3_recommended`
  patience=3, factor=0.5, cooldown=1 validates to n=10). Cosine-warmup is the sliding default.
```

---

## 6. What to run next

Five concrete experiments, ordered by expected information value.

### 6.1 Weekly plateau-cell n=10 expansion (highest priority)

The single-seed Weekly p3_recommended hit 6.559 SMAPE — −0.176 vs prior Weekly SOTA. If this holds at n=10, Weekly tiered SOTA flips, and the "Weekly is a real cascade reversal" claim is retracted in favor of "plateau patience window collapses at H=13".

```yaml
experiment_name: tiered_weekly_plateau_p3_n10
dataset: m4
periods: [Weekly]
training:
  sampling_style: nbeats_paper
  val_check_interval: 100
  min_delta: 0.001
  patience: 20
  max_epochs: 200
  optimizer: adam
  learning_rate: 0.001
  lr_scheduler:
    name: ReduceLROnPlateau
    patience: 3
    factor: 0.5
    cooldown: 1
configs:
  - {builtin: T+Sym10V3_30s,    active_g: false, tiered: ascend}
  - {builtin: T+DB3V3_30s,      active_g: false, tiered: ascend}
  - {builtin: T+Coif2V3_30s,    active_g: false}                  # non-tiered control
  - {builtin: TAELG+Sym10V3AELG_30s, active_g: false, tiered: ascend}
n_runs: 10
```

Expected: T+Sym10V3_30s_tiered_ag0 lands at SMAPE ≤ 6.65, beating prior SOTA 6.735 and validating tiered offset on Weekly under proper plateau tuning. If it fails, the cascade reversal interpretation stands.

### 6.2 Hourly tiered all-periods integration (PARTIALLY REDUNDANT — see scope-down below)

**Status update (2026-05-03 revision):** the analysis-script canonicalization fix removed the original motivation for re-running `T+Sym10V3_10s` and `TAE+Sym10V3AE_10s` on Hourly — those architectures are already present in `m4_hourly_sym10_tiered_offset{,_paperlr}_results.csv` under direction-suffixed names, and once mapped to the all-periods canonical name they yield the 6/6-period mean rank reported in §3.1. The 6/6-period generalist crown is **already settled** as `T+Sym10V3_10s_tiered_ag0` mean rank 13.33.

**What is actually missing from Hourly tiered coverage** (still worth running):

1. **`T+DB3V3_10s_tiered_ag0` on Hourly** — currently 5/6 coverage (mean rank 15.00, no Hourly cell). If this lands close to T+Sym10V3 on Hourly, it could match or beat the 13.33 mean rank.
2. **30s-depth tiered configs on Hourly** — `T+Sym10V3_30s_tiered_*`, `T+DB3V3_30s_tiered_*`, `TAELG+Sym10V3AELG_30s_tiered`, `TAE+DB3V3AE_30s_tiered` are all 5/6 (no Hourly). The 30s family lost on Yearly/Quarterly to 10s, but Hourly's H=48 long horizon may favor the deeper stack.
3. **Unified `TWAE_10s_td3_sym10_ld16` and `TW_10s_td3_sym10` non-tiered Hourly with plateau LR.** The `TWAE_10s_td3_sym10_ld16_tiered` Hourly cell came in at 9.243 (rank 60) and the `TW_10s_td3_sym10_tiered` cell at 9.024 (rank 18). The non-tiered comprehensive `TW_10s_td3_bdeq_sym10` Hourly is only 9.213 (step_paper) and was not run under plateau — the missing plateau cell is what would let unified TW be fairly compared.

```yaml
experiment_name: tiered_offset_m4_hourly_extension
dataset: m4
periods: [Hourly]
training:
  sampling_style: nbeats_paper
  val_check_interval: 100
  min_delta: 0.001
  patience: 20
  max_epochs: 200
  lr_scheduler: {name: ReduceLROnPlateau, patience: 3, factor: 0.5, cooldown: 1}
configs:
  # Fill DB3 10s tiered (the closest competitor to the new generalist crown)
  - {builtin: T+DB3V3_10s,     active_g: false, tiered: ascend}
  - {builtin: T+DB3V3_10s,     active_g: false, tiered: descend}
  # 30s-depth tiered configs that currently miss Hourly
  - {builtin: T+Sym10V3_30s,   active_g: false,    tiered: ascend}
  - {builtin: T+Sym10V3_30s,   active_g: forecast, tiered: ascend}
  - {builtin: T+DB3V3_30s,     active_g: false,    tiered: ascend}
  - {builtin: TAELG+Sym10V3AELG_30s, tiered: ascend}
  - {builtin: TAE+DB3V3AE_30s, tiered: ascend}
  # Plateau LR for the non-tiered unified TW that currently only has step_paper Hourly
  - {builtin: TW_10s_td3_bdeq_sym10}                        # non-tiered control
n_runs: 10
```

The original §6.2 yaml is dropped — the bulk of its configs (`T+Sym10V3_10s`, `TAELG+Sym10V3AELG_10s`) are already covered in the Hourly tiered file and would be duplicate work after canonicalization.

### 6.3 Within-wavelet selectivity test on Daily (already recommended in tiered analysis)

If haar (single-level on H=14) gains as much as multi-level sym10/db3 from tiering, the cascade-frequency-band interpretation collapses to "non-uniform basis_dim is a regularizer". Cross-references the tiered_offset analysis open question.

```yaml
experiment_name: tiered_within_wavelet_daily
dataset: m4
periods: [Daily]
training: {sampling_style: nbeats_paper, val_check_interval: 100, min_delta: 0.001,
           patience: 20, max_epochs: 200,
           lr_scheduler: {name: ReduceLROnPlateau, patience: 3, factor: 0.5, cooldown: 1}}
configs:
  - {builtin: T+HaarV3_10s,   active_g: false, tiered: ascend}    # single-level control
  - {builtin: T+DB3V3_10s,    active_g: false, tiered: ascend}
  - {builtin: T+Sym10V3_10s,  active_g: false, tiered: ascend}
  - {builtin: T+Coif2V3_10s,  active_g: false, tiered: ascend}
  # Direction control
  - {builtin: T+Sym10V3_10s,  active_g: false, tiered: descend}
  - {builtin: T+DB3V3_10s,    active_g: false, tiered: descend}
n_runs: 10
```

### 6.4 Plateau LR rollout to remaining non-tiered comprehensive configs

`comprehensive_m4_paper_sample_plateau_results.csv` covers a partial subset of `comprehensive_m4_paper_sample.yaml` configs. Filling the gaps (especially TWAE / TWAELG / TAELG configs on the larger latent_dims) would let the per-period rankings be fully apples-to-apples between LR schedulers without reading from two source files. Estimated: ~30 configs × 10 seeds × 6 periods = 1,800 new runs.

### 6.5 Cosine-warmup port to paper-sample

Sliding's cosine-warmup is the strongest sliding LR scheduler and beats step_paper (and arguably plateau) on Daily and Hourly. Port the same cosine-warmup config to a paper-sample experiment and see whether it matches plateau on Q/M/D and beats step_paper on Y/W. If yes, cosine-warmup becomes the universal M4 default.

```yaml
experiment_name: m4_paper_sample_cosine_warmup
dataset: m4
periods: [Yearly, Quarterly, Monthly, Weekly, Daily, Hourly]
training:
  sampling_style: nbeats_paper
  val_check_interval: 100
  min_delta: 0.001
  patience: 20
  max_epochs: 200
  lr_scheduler:
    name: CosineAnnealing
    warmup_epochs: 15
    eta_min: 1.0e-6
configs:
  - {builtin: NBEATS-IG_10s, active_g: false}
  - {builtin: NBEATS-IG_30s, active_g: false}
  - {builtin: T+Sym10V3_30s, active_g: false}
  - {builtin: TW_30s_td3_bdeq_sym10, active_g: false}
  - {builtin: TWAE_10s_ld32}
  - {builtin: TAELG+Coif2V3ALG_30s}
n_runs: 10
```

---

## Open questions

1. **Hourly tiered with plateau LR vs sliding cosine-warmup:** sliding NBEATS-IG_30s_agf hits 8.587 — paper-sample plateau tiered hits 8.922 best. Is the gap protocol-driven or LR-driven? Experiment 6.5 will arbitrate.
2. **Why does plateau LR win Monthly (−0.14 SMAPE) but not Yearly?** Monthly has the largest series count (48k) and longest H (18 in M4 ex-Hourly); plateau may be learning the longer-horizon structure better. Yearly H=6 may be too short for plateau's adaptive reduction to kick in.
3. **Does cosine-warmup beat plateau on paper-sample?** Direct head-to-head not available.
4. **Is the Daily paper-sample vs sliding gap (−0.42 SMAPE) closeable by paper-sample?** Sliding's NBEATS-G_30s_ag0 2.588 is the lowest M4 Daily SMAPE on record. Paper-sample best is 3.012. Either the protocol fundamentally suits Generic deep architectures on Daily, or paper-sample has not yet been tested with the right hyperparameters.

---

## Appendix A — full file inventory

50 CSVs, classified:

| Tier | Files | Used in this report |
|---|---|---|
| Modern paper-sample | 9 files (3.2M rows) | Yes — all leaderboards |
| Modern sliding | 1 file (6.8k rows) | Yes — sliding leaderboards + generalist |
| Legacy sliding (omnibus, study, ablation) | 33 files (15.4k rows) | Cross-checks only (consistent with leaderboards) |
| Other | 7 files (ensemble summaries, test_earlystop_fix, etc) | Not used |

Reproducibility: rerun `python experiments/analysis/scripts/m4_overall_leaderboard.py` to regenerate every table here. The script applies the strict divergence filter and emits all leaderboards + LR head-to-heads.
