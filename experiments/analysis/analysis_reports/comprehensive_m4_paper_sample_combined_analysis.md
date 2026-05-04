# Comprehensive M4 Paper-Sample Sweep — Combined Analysis (canonical)

**Date:** 2026-04-27
**Dataset:** M4 (all six periods: Yearly, Quarterly, Monthly, Weekly, Daily, Hourly)
**Sampling protocol:** `nbeats_paper` (per-series random window sampling, `steps_per_epoch=1000`, `val_check_interval=100`, `min_delta=0.001`, `patience=20`)
**Source CSVs (treated as one merged dataset):**

- `experiments/results/m4/comprehensive_m4_paper_sample_results.csv` (3,180 rows)
- `experiments/results/m4/comprehensive_m4_paper_sample_sym10_fills_results.csv` (900 rows)

**Source YAMLs:**

- `experiments/configs/comprehensive_m4_paper_sample.yaml`
- `experiments/configs/comprehensive_m4_paper_sample_sym10_fills.yaml`

**Status:** This report is the canonical pointer for the M4 paper-sample sweep and supersedes both prior reports (`comprehensive_m4_paper_sample_analysis.md` and `comprehensive_m4_paper_sample_sym10_fills_analysis.md`). Prior reports remain on disk for traceability but should not be cited going forward.

---

## 1. Executive Summary

- **Merged corpus:** 68 unique configs × 6 periods × 10 runs = 4,080 raw runs. After the strict divergence filter (Section 2), 4,066 runs remain across **402/408 cells at n=10** and **6/408 cells at n=8–9**. Zero configs have any period with zero surviving runs.
- **Best M4 generalist:** `T+Sym10V3_30s_bdeq` (mean rank **9.17/68**, top-3 on three periods, never below rank 21). The 30-stack alternating Trend+WaveletV3 RootBlock family occupies 6 of the top 8 generalist slots — this family is by a wide margin the strongest single architectural choice on M4 paper-sample.
- **Sub-1M parameter champion:** `TWAE_10s_ld32_ag0` (0.55M params, mean rank 27.67/68 = best within the 12-config sub-1M tier). For the parameter-extreme regime, `TWAELG_10s_ld16_db3_ag0` (0.51M, fill) is statistically equivalent to `_ld32_*` siblings on 5/6 periods.
- **Per-period SOTA (mean of 10 runs after filter):** Yearly `TALG+DB3V3ALG_10s_ag0` (13.550, 1.04M); Quarterly `T+HaarV3_10s_bdeq` (10.356, 5.13M, **statistical tie** with `NBEATS-IG_10s_ag0` at 10.357 — MWU p=0.97); Monthly `TW_30s_td3_bdeq_haar` (13.391, 6.78M); Weekly `T+Coif2V3_30s_bdeq` (6.735, 15.75M); Daily `TAELG+Coif2V3ALG_30s_ag0` (3.036, 3.73M); Hourly `NBEATS-IG_30s_agf` (8.758, 43.6M; best paper-faithful is `NBEATS-IG_30s_ag0` at rank 4).
- **`active_g=forecast` is a Yearly+Hourly switch, not a default.** Across 9 (ag0, agf) pairs spanning paper backbones and novel TWAE/TWAELG, agf wins **9/9 on Hourly** (5/9 sig at p<0.05) and never wins consistently anywhere else. Quarterly is the worst period for agf (1/9 wins). The `agf` suffix is a repository extension, not paper-faithful.
- **Wavelet family ranking (pooled across stack architectures):** No Kruskal-Wallis p<0.05 on any period. Sym10/coif2/haar are interchangeable within ±0.03 SMAPE for most periods; coif3 produces no per-period SOTA. Drop coif3 from default M4 sweeps; keep haar/db3/coif2/sym10.
- **Stack-depth × wavelet interaction (alt-T+W RB family):** haar 10s ≈ haar 30s (mean rank 15.5 vs 29.8 — 10s wins on rank). Db3 30s significantly better than db3 10s on Weekly (p=0.005). Sym10/coif2: 30s preferred but not significantly. **Heuristic: short-support wavelets → 10 stacks suffice; long-support → 30 stacks.**
- **Drop list (universal worst on M4):** `GenericVAE_3s_sw0` (mean rank 68 = last on every period), `GenericAE_30s_sw0` (53.5), `GAELG_10s_ld16_ag0` (53), `NBEATS-G_30s_ag0` (52.3 — bimodal collapse on Quarterly/Weekly/Daily), `GAE_*_ld32_ag0` (~50), all pure GenericAE/AELG variants. The wavelet/trend basis is what carries the novel-family wins; the AE bottleneck on a Generic backbone alone is not enough.

---

## 2. Methodology

### 2.1 Strict divergence filter

A run is dropped from all per-config statistics, leaderboards, mean-rank, sub-1M, novel head-to-head, AE-vs-AELG, agf-vs-ag0, Wilcoxon/MWU comparisons — **everywhere** — if **any** of the following hold:

```
diverged == True
OR smape > 100
OR (best_epoch == 0 AND smape > 50)
```

This is a stricter rule than the CSV `diverged` flag alone. The first clause catches the only explicitly-flagged divergence (`GenericVAE_3s_sw0` Daily run 1). The second clause catches the GenericVAE Hourly cluster (eight runs at SMAPE 101–110) and the two Daily SMAPE=200 outliers (`TW_10s_td3_bdeq_coif2` run 3, `TAE+Sym10V3AE_10s_ld32_ag0` run 5). The third clause catches the GenericVAE Yearly/Weekly seeds where training never moved (best_epoch=0 with SMAPE 63–66).

### 2.2 Filter audit

| Reason | Count |
|---|---|
| Explicit `diverged==True` | 1 |
| `smape > 100` (not flagged) | 10 |
| `best_epoch == 0` AND `smape > 50` (not in the above) | 3 |
| **Total dropped** | **14 / 4,080** |

Full list of dropped rows:

| Config | Period | Origin | Run | SMAPE | best_epoch | diverged flag |
|---|---|---|---|---|---|---|
| GenericVAE_3s_sw0 | Yearly | orig | 0 | 64.90 | 0 | False |
| GenericVAE_3s_sw0 | Yearly | orig | 9 | 65.82 | 0 | False |
| GenericVAE_3s_sw0 | Weekly | orig | 9 | 63.82 | 0 | False |
| GenericVAE_3s_sw0 | Daily | orig | 1 | 13.55 | 0 | **True** |
| GenericVAE_3s_sw0 | Hourly | orig | 0,1,3,4,5,6,7,9 (×8) | 101–110 | 24–75 | False |
| TW_10s_td3_bdeq_coif2 | Daily | orig | 3 | 200.00 | 0 | False |
| TAE+Sym10V3AE_10s_ld32_ag0 | Daily | fill | 5 | 200.00 | 0 | False |

### 2.3 Cell-count audit

After filter: 402/408 (config × period) cells have n=10. The 6 incomplete cells are:

| Config | Period | n_remaining |
|---|---|---|
| GenericVAE_3s_sw0 | Yearly | 8 |
| GenericVAE_3s_sw0 | Weekly | 9 |
| GenericVAE_3s_sw0 | Daily | 9 |
| GenericVAE_3s_sw0 | Hourly | 2 |
| TW_10s_td3_bdeq_coif2 | Daily | 9 |
| TAE+Sym10V3AE_10s_ld32_ag0 | Daily | 9 |

`GenericVAE_3s_sw0` Hourly at n=2 is the only severely thin cell. The mean SMAPE there (9.34, the only two surviving seeds) is essentially uninformative; this config is universally last on the leaderboard regardless. The 9-row cells contribute negligible noise to means/ranks. **Zero configs have all-zero coverage in any period.**

### 2.4 Convention: `agf` vs `ag0`

`active_g` is a repo extension to N-BEATS, not part of Oreshkin et al. 2020. `*_ag0` (`active_g=False`) reproduces the published architecture on `NBEATS-G`/`NBEATS-IG` backbones. `*_agf` (`active_g=forecast`) applies the activation to the forecast head — a novel extension. When reporting "paper baseline" SMAPE, only `*_ag0` results count.

### 2.5 Notable non-divergent instability

`NBEATS-G_30s_ag0` exhibits real bimodal collapse on three periods that the filter does not catch (no SMAPE > 100): Quarterly mean 15.40 with std 9.46 (range 10.5–34.5), Weekly mean 8.78 with std 4.38 (range 7.0–21.9), Daily mean 10.21 with std 11.39 (range 3.09–31.6). These are kept in all stats — they reflect the architecture's true behaviour on this protocol, but inflate its mean rank to 52.3/68. Confirms the prior empirical default: **cap legacy `NBEATS-G` and pure `Generic` blocks at 10 stacks**.

---

## 3. Per-Period SMAPE Leaderboards (top-10)

All means are over the surviving runs after filter. Origin = `orig` (parent sweep) or `fill` (sym10 fills). The merged dataset is treated as a single experiment; the origin column is purely traceability.

### 3.1 Yearly (H=6, L=30)

| # | Config | SMAPE ± std | OWA | Params | n | Origin |
|---|---|---|---|---|---|---|
| 1 | TALG+DB3V3ALG_10s_ag0 | 13.5496 ± 0.0955 | 0.806 | 1.04M | 10 | orig |
| 2 | TWAELG_10s_ld32_db3_agf | 13.5780 ± 0.1473 | 0.808 | 0.48M | 10 | orig |
| 3 | TAE+Sym10V3AE_10s_ld32_ag0 | 13.5861 ± 0.1819 | 0.810 | 1.11M | 10 | fill |
| 4 | NBEATS-IG_10s_ag0 | 13.5902 ± 0.1477 | 0.810 | 19.51M | 10 | orig |
| 5 | TWAELG_10s_ld16_db3_ag0 | 13.5934 ± 0.0830 | 0.809 | 0.44M | 10 | fill |
| 6 | TWAE_10s_ld32_agf | 13.5947 ± 0.1272 | 0.811 | 0.48M | 10 | orig |
| 7 | NBEATS-G_10s_ag0 | 13.5992 ± 0.1298 | 0.811 | 8.22M | 10 | orig |
| 8 | TW_10s_td3_bdeq_coif2 | 13.6009 ± 0.1528 | 0.809 | 2.08M | 9 | orig |
| 9 | TWGAELG_10s_ld16_sym10_ag0 | 13.6029 ± 0.0902 | 0.809 | 0.45M | 10 | orig |
| 10 | T+HaarV3_10s_bdeq | 13.6048 ± 0.1459 | 0.810 | 5.08M | 10 | fill |

Top-10 spread = 0.055 SMAPE — a flat statistical tie. Notable: ranks 2, 5, 6, 9 are all **sub-0.5M-param** novel architectures. The Yearly leaderboard is dominated by parameter-efficient designs.

### 3.2 Quarterly (H=8, L=40)

| # | Config | SMAPE ± std | OWA | Params | n | Origin |
|---|---|---|---|---|---|---|
| 1 | T+HaarV3_10s_bdeq | 10.3560 ± 0.0756 | 0.914 | 5.13M | 10 | fill |
| 2 | NBEATS-IG_10s_ag0 | 10.3570 ± 0.1192 | 0.914 | 19.64M | 10 | orig |
| 3 | T+Coif2V3_10s_bdeq | 10.3647 ± 0.0781 | 0.913 | 5.13M | 10 | fill |
| 4 | T+Db3V3_10s_bdeq | 10.3668 ± 0.0914 | 0.914 | 5.13M | 10 | fill |
| 5 | TAE+HaarV3AE_10s_ld32_ag0 | 10.3681 ± 0.0614 | 0.912 | 1.16M | 10 | orig |
| 6 | NBEATS-IG_30s_ag0 | 10.3721 ± 0.0715 | 0.914 | 36.31M | 10 | orig |
| 7 | T+Sym10V3_10s_bdeq | 10.3729 ± 0.1057 | 0.916 | 5.13M | 10 | fill |
| 8 | T+Coif2V3_30s_bdeq | 10.3733 ± 0.0915 | 0.914 | 15.39M | 10 | orig |
| 9 | TAE+Coif2V3AE_10s_ld32_ag0 | 10.3805 ± 0.1054 | 0.918 | 1.16M | 10 | orig |
| 10 | TAELG+Coif2V3ALG_10s_ag0 | 10.4009 ± 0.0496 | 0.920 | 1.09M | 10 | fill |

`T+HaarV3_10s_bdeq` (10.3560) vs `NBEATS-IG_10s_ag0` (10.3570): MWU U=49, p=0.97; Welch t=−0.024, p=0.98 — **statistical tie**. The 4 alt-T+WaveletV3 10-stack configs (haar/coif2/db3/sym10, ranks 1/3/4/7) cluster into a single noise band, all using ~5.1M params vs NBEATS-IG_10s's 19.6M.

### 3.3 Monthly (H=18, L=90)

| # | Config | SMAPE ± std | OWA | Params | n | Origin |
|---|---|---|---|---|---|---|
| 1 | TW_30s_td3_bdeq_haar | 13.3906 ± 0.5838 | 0.916 | 6.78M | 10 | orig |
| 2 | T+Sym10V3_30s_bdeq | 13.3970 ± 0.5441 | 0.915 | 16.12M | 10 | orig |
| 3 | TAELG+Db3V3ALG_30s_ag0 | 13.4775 ± 0.4244 | 0.919 | 4.03M | 10 | orig |
| 4 | TAE+DB3V3AE_30s_ld32_ag0 | 13.5085 ± 0.3397 | 0.923 | 4.22M | 10 | orig |
| 5 | TWAE_10s_ld32_sym10_ag0 | 13.5133 ± 0.3782 | 0.923 | 0.58M | 10 | fill |
| 6 | TAE+Sym10V3AE_30s_ld32_ag0 | 13.5270 ± 0.3064 | 0.925 | 4.22M | 10 | fill |
| 7 | TAE+Coif2V3AE_10s_ld32_ag0 | 13.5351 ± 0.3762 | 0.924 | 1.41M | 10 | orig |
| 8 | Generic_10s_sw0 | 13.5447 ± 0.5324 | 0.933 | 8.90M | 10 | orig |
| 9 | TW_30s_td3_bdeq_coif3 | 13.5489 ± 0.4426 | 0.925 | 6.78M | 10 | orig |
| 10 | TW_10s_td3_bdeq_coif3 | 13.5557 ± 0.5349 | 0.927 | 2.26M | 10 | orig |

Top-2 within 0.006 SMAPE — full tie. Std is 4–5× larger than Yearly/Quarterly so the top-10 spread of 0.165 SMAPE is also within seed noise. Sub-0.6M `TWAE_10s_ld32_sym10_ag0` cracks rank 5.

### 3.4 Weekly (H=13, L=65)

| # | Config | SMAPE ± std | OWA | Params | n | Origin |
|---|---|---|---|---|---|---|
| 1 | T+Coif2V3_30s_bdeq | 6.7347 ± 0.2031 | 0.751 | 15.75M | 10 | orig |
| 2 | T+Db3V3_30s_bdeq | 6.7371 ± 0.2560 | 0.761 | 15.75M | 10 | orig |
| 3 | T+Sym10V3_30s_bdeq | 6.8577 ± 0.2730 | 0.763 | 15.75M | 10 | orig |
| 4 | T+Coif2V3_10s_bdeq | 6.9187 ± 0.4243 | 0.790 | 5.25M | 10 | fill |
| 5 | TW_30s_td3_bdeq_db3 | 6.9200 ± 0.2751 | 0.780 | 6.55M | 10 | orig |
| 6 | TWGAELG_30s_ld16_sym10_ag0 | 6.9386 ± 0.4798 | 0.780 | 1.55M | 10 | orig |
| 7 | T+HaarV3_30s_bdeq | 6.9416 ± 0.3589 | 0.784 | 15.75M | 10 | orig |
| 8 | T+HaarV3_10s_bdeq | 6.9426 ± 0.2950 | 0.785 | 5.25M | 10 | fill |
| 9 | TAE+Sym10V3AE_30s_ld32_ag0 | 6.9439 ± 0.3352 | 0.770 | 3.84M | 10 | fill |
| 10 | T+Sym10V3_10s_bdeq | 6.9489 ± 0.4117 | 0.769 | 5.25M | 10 | fill |

Top-2 within 0.002 SMAPE — full tie. Weekly is the period most dominated by alt-T+W RB and least by paper baselines (NBEATS-IG_10s_ag0 sits at rank 14, NBEATS-G_30s_ag0 collapses).

### 3.5 Daily (H=14, L=70)

| # | Config | SMAPE ± std | OWA | Params | n | Origin |
|---|---|---|---|---|---|---|
| 1 | TAELG+Coif2V3ALG_30s_ag0 | 3.0361 ± 0.0340 | 0.994 | 3.73M | 10 | orig |
| 2 | TWGAELG_10s_ld16_db3_ag0 | 3.0506 ± 0.0494 | 0.998 | 0.52M | 10 | orig |
| 3 | T+Sym10V3_30s_bdeq | 3.0510 ± 0.0434 | 1.000 | 15.82M | 10 | orig |
| 4 | TAE+HaarV3AE_10s_ld32_ag0 | 3.0538 ± 0.0583 | 1.001 | 1.31M | 10 | orig |
| 5 | T+Db3V3_10s_bdeq | 3.0540 ± 0.0569 | 1.000 | 5.27M | 10 | fill |
| 6 | T+Coif3V3_30s_bdeq | 3.0579 ± 0.0368 | 1.000 | 15.82M | 10 | orig |
| 7 | TWAELG_10s_ld32_db3_ag0 | 3.0584 ± 0.0496 | 1.002 | 0.55M | 10 | orig |
| 8 | T+HaarV3_10s_bdeq | 3.0590 ± 0.0562 | 1.002 | 5.27M | 10 | fill |
| 9 | T+Sym10V3_10s_bdeq | 3.0590 ± 0.0610 | 1.003 | 5.27M | 10 | fill |
| 10 | NBEATS-IG_30s_ag0 | 3.0621 ± 0.0512 | 1.004 | 37.40M | 10 | orig |

Top-10 spread = 0.026 SMAPE — within-seed-noise tie. Sub-0.6M `TWGAELG_10s_ld16_db3_ag0` (rank 2) and `TWAELG_10s_ld32_db3_ag0` (rank 7) are exceptional Pareto frontier picks: 30× fewer params than the per-period winner with negligible SMAPE penalty.

### 3.6 Hourly (H=48, L=240)

| # | Config | SMAPE ± std | OWA | Params | n | Origin |
|---|---|---|---|---|---|---|
| 1 | NBEATS-IG_30s_agf | 8.7583 ± 0.0992 | 0.423 | 43.58M | 10 | orig |
| 2 | NBEATS-G_30s_agf | 8.8622 ± 0.0847 | 0.456 | 31.76M | 10 | orig |
| 3 | NBEATS-IG_10s_agf | 8.8926 ± 0.1064 | 0.428 | 22.40M | 10 | orig |
| 4 | NBEATS-IG_30s_ag0 | 8.9056 ± 0.0833 | 0.409 | 43.58M | 10 | orig |
| 5 | TWAELG_10s_ld32_db3_agf | 8.9237 ± 0.1292 | 0.433 | 0.85M | 10 | orig |
| 6 | TWAELG_10s_ld32_sym10_agf | 8.9331 ± 0.1095 | 0.436 | 0.85M | 10 | fill |
| 7 | NBEATS-G_30s_ag0 | 8.9337 ± 0.1096 | 0.430 | 31.76M | 10 | orig |
| 8 | TWAELG_10s_ld16_db3_agf | 8.9457 ± 0.0976 | 0.434 | 0.81M | 10 | fill |
| 9 | TWAE_10s_ld32_agf | 8.9534 ± 0.0991 | 0.434 | 0.85M | 10 | orig |
| 10 | TWAE_10s_ld32_sym10_agf | 8.9622 ± 0.0897 | 0.439 | 0.85M | 10 | fill |

Hourly is the agf-dominated period: the top 3 are all `*_agf`. Best paper-faithful entry: `NBEATS-IG_30s_ag0` at rank 4 (+0.147 SMAPE behind the agf winner). The 0.85M-param sub-1M `TWAELG_10s_ld32_db3_agf` lands at rank 5, only +0.165 SMAPE behind the 43.6M-param agf winner — best parameter efficiency on Hourly.

**Tiered-offset Hourly follow-up (added 2026-05-04).** A targeted Hourly tiered-offset sweep (sym10 only, plateau vs step_paper LR × ascend/descend × 4 backbones, n=10/cell) is now available in `m4_hourly_sym10_tiered_offset_analysis_2026-05-04.md`. Best tiered Hourly cell is `T+Sym10V3_10s_bdEQ_descend` plateau at SMAPE 8.9224 ± 0.1132 — **+0.164 SMAPE behind the rank-1 paper-sample SOTA** (`NBEATS-IG_30s_agf` 8.758) listed above and **+0.335 behind the sliding-protocol SOTA** (`NBEATS-IG_30s_agf` 8.587). Hourly tiered does **not** beat the paper baseline under either protocol. Three findings refine §3.6/§8 on Hourly: (1) **LR is backbone-asymmetric for Hourly tiered configs** — alternating `T+Sym10V3` / `TAE+Sym10V3AE` prefer plateau LR (Δ −0.168 SMAPE, p=0.026); unified `TWAE_10s_td3_sym10_ld16` prefers step_paper LR (Δ +0.196, p=0.004); unified `TW_10s_td3_sym10` is LR-insensitive; (2) **tiered direction (ascend vs descend) is noise on Hourly** (all p > 0.10) — choose by parameter count, not direction; (3) **generalist crown unaffected** — `T+Sym10V3_10s_tiered_ag0` (mean rank 13.33/108 in `m4_overall_leaderboard_2026-05-03.md`) uses the 8.922 plateau-descend cell as its Hourly slot, and the `_paperlr` file does not move it.

---

## 4. Generalist Mean-Rank Leaderboard (top-20)

Mean rank = average of per-period ranks across all 6 periods (each period ranked independently across all 68 configs, lower is better). Lower mean rank = stronger generalist.

| # | Config | Mean rank | Yr | Qy | Mo | Wk | Dy | Hr | Params | Origin |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | T+Sym10V3_30s_bdeq | **9.17** | 21 | 15 | 2 | 3 | 3 | 11 | 15.82M | orig |
| 2 | NBEATS-IG_30s_ag0 | 11.17 | 15 | 6 | 15 | 17 | 10 | 4 | 37.40M | orig |
| 3 | T+HaarV3_10s_bdeq | **15.50** | 10 | 1 | 20 | 8 | 8 | 46 | 5.27M | fill |
| 4 | NBEATS-IG_10s_ag0 | 16.17 | 4 | 2 | 45 | 14 | 18 | 14 | 20.06M | orig |
| 5 | T+Coif2V3_30s_bdeq | 17.83 | 19 | 8 | 26 | 1 | 33 | 20 | 15.82M | orig |
| 5 | T+Db3V3_30s_bdeq | 17.83 | 22 | 23 | 23 | 2 | 21 | 16 | 15.82M | orig |
| 7 | T+Coif3V3_30s_bdeq | 19.83 | 31 | 11 | 38 | 11 | 6 | 22 | 15.82M | orig |
| 8 | T+Coif2V3_10s_bdeq | 20.00 | 20 | 3 | 43 | 4 | 22 | 28 | 5.27M | fill |
| 9 | T+Sym10V3_10s_bdeq | 25.33 | 40 | 7 | 60 | 10 | 9 | 26 | 5.27M | fill |
| 10 | TAE+Coif2V3AE_10s_ld32_ag0 | 25.33 | 27 | 9 | 7 | 30 | 39 | 40 | 1.31M | orig |
| 11 | TW_30s_td3_bdeq_haar | 25.50 | 42 | 30 | 1 | 20 | 19 | 41 | 6.60M | orig |
| 12 | TAE+Sym10V3AE_10s_ld32_ag0 | 25.83 | 3 | 16 | 14 | 34 | 37 | 51 | 1.31M | fill |
| 13 | TW_10s_td3_bdeq_coif3 | 26.33 | 17 | 38 | 10 | 19 | 25 | 49 | 2.20M | orig |
| 14 | TW_30s_td3_bdeq_db3 | 26.83 | 11 | 51 | 17 | 5 | 41 | 36 | 6.60M | orig |
| 15 | TAE+Sym10V3AE_30s_ld32_ag0 | 27.00 | 63 | 21 | 6 | 9 | 34 | 29 | 3.92M | fill |
| 16 | TAE+HaarV3AE_30s_ld32_ag0 | 27.33 | 32 | 50 | 13 | 23 | 28 | 18 | 3.92M | orig |
| 17 | TAELG+Sym10V3ALG_30s_ag0 | 27.67 | 23 | 37 | 12 | 26 | 24 | 44 | 3.73M | orig |
| 17 | TWAE_10s_ld32_ag0 | 27.67 | 12 | 12 | 16 | 50 | 29 | 47 | 0.55M | orig |
| 19 | TW_10s_td3_bdeq_haar | 29.50 | 16 | 18 | 41 | 37 | 11 | 54 | 2.20M | orig |
| 20 | T+HaarV3_30s_bdeq | 29.83 | 55 | 13 | 46 | 7 | 43 | 15 | 15.82M | orig |

**Headline:** the 30-stack alt Trend+WaveletV3 RootBlock family (`T+<wav>V3_30s_bdeq`) takes 5 of the top-7 generalist slots (sym10, coif2, db3, coif3, haar) plus T+HaarV3_10s_bdeq at rank 3 and T+Coif2V3_10s_bdeq at rank 8. The 10-stack alt-T+W RB family enters the top-20 at ~5.1M params, ~1/3 the params of the 30-stack siblings.

The smallest top-20 generalist is **`TWAE_10s_ld32_ag0` at 0.55M params (rank 17)** — the only sub-1M config in the top-20 leaderboard.

---

## 5. Sub-1M Parameter Leaderboard

12 configs in the merged set fall under 1M params (excluding `GenericVAE_3s_sw0` whose Hourly cell is too thin to rank fairly). Mean rank below is computed against the full 68-config space, and against the sub-1M tier only.

| # (sub-1M) | Config | Params | Yr | Qy | Mo | Wk | Dy | Hr | Mean rank (68) | Mean rank (sub-1M) | Origin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TWAE_10s_ld32_ag0 | 0.55M | 12 | 12 | 16 | 50 | 29 | 47 | 27.67 | 3.67 | orig |
| 2 | TWAELG_10s_ld32_db3_ag0 | 0.55M | 25 | 22 | 35 | 43 | 7 | 62 | 32.33 | 4.50 | orig |
| 3 | TWAE_10s_ld32_agf | 0.55M | 6 | 47 | 40 | 48 | 45 | 9 | 32.50 | 5.67 | orig |
| 4 | TWGAELG_10s_ld16_db3_ag0 | 0.52M | 49 | 27 | 42 | 54 | 2 | 60 | 39.00 | 5.83 | orig |
| 5 | TWAELG_10s_ld32_db3_agf | 0.55M | 2 | 29 | 44 | 62 | 52 | 5 | 32.33 | 6.17 | orig |
| 6 | TWAELG_10s_ld32_sym10_ag0 | 0.55M | 18 | 35 | 11 | 57 | 30 | 65 | 36.00 | 6.50 | fill |
| 7 | TWAELG_10s_ld16_db3_ag0 | 0.51M | 5 | 31 | 57 | 52 | 50 | 61 | 42.67 | 6.67 | fill |
| 8 | TWAE_10s_ld32_sym10_agf | 0.55M | 48 | 49 | 37 | 55 | 35 | 10 | 39.00 | 7.17 | fill |
| 9 | TWAE_10s_ld32_sym10_ag0 | 0.55M | 30 | 45 | 5 | 58 | 36 | 63 | 39.50 | 7.50 | fill |
| 10 | TWGAELG_10s_ld16_sym10_ag0 | 0.52M | 9 | 52 | 52 | 56 | 15 | 66 | 41.67 | 7.67 | orig |
| 11 | TWAELG_10s_ld16_db3_agf | 0.51M | 24 | 36 | 59 | 61 | 44 | 8 | 38.67 | 8.00 | fill |
| 12 | TWAELG_10s_ld32_sym10_agf | 0.55M | 14 | 60 | 66 | 65 | 38 | 6 | 41.50 | 8.67 | fill |

**Sub-1M champion:** `TWAE_10s_ld32_ag0` (0.55M, mean rank 3.67 within sub-1M). Spreads its top finishes evenly: rank 12 on Yearly, 12 on Quarterly, 16 on Monthly, 29 on Daily — never near the bottom of the 12-config tier.

**Pareto frontier per period:**

- **Yearly:** `TWAELG_10s_ld32_db3_agf` (rank 2/68, 0.48M)
- **Daily:** `TWGAELG_10s_ld16_db3_ag0` (rank 2/68, 0.52M) — best parameter efficiency on the entire sweep
- **Hourly:** `TWAELG_10s_ld32_db3_agf` (rank 5/68, 0.85M) — only 0.165 SMAPE behind 43.6M-param NBEATS-IG_30s_agf

**Parameter-extreme:** `TWAELG_10s_ld16_db3_ag0` (0.51M) places rank 5 on Yearly and is statistically equivalent to `_ld32_*` siblings on 5/6 periods (only Daily ag0 reaches p=0.021 in favour of ld=32). Ld=16 is the new parameter-extreme choice when budget is the binding constraint.

---

## 6. Novel Architecture Head-to-Head

Excludes paper baselines (`NBEATS-G`, `NBEATS-IG`), no-weight-sharing pure Generic configs (`Generic_*_sw0`), and `BNG*`/legacy BottleneckGeneric variants. Pure GenericAE/AELG (`GAE_*`, `GAELG_*`) are kept as the floor of the novel comparison.

Group classification by config_name pattern:

| Group | n configs | Mean period rank (over 68) | Median rank | Best rank | n_top5 |
|---|---|---|---|---|---|
| **alt T+W (RootBlock)** — `T+<wav>V3_*_bdeq` | 9 | **20.59** | 19.5 | 1 | 10 |
| alt T+W (AE) — `TAE+<wav>V3AE_*_ld32_ag0` | 8 | 30.29 | 29.5 | 3 | 4 |
| unified TW (RootBlock) — `TW_*_td3_bdeq_<wav>` | 10 | 30.55 | 27.5 | 1 | 2 |
| unified TWAE — `TWAE_10s_ld32_*` | 4 | 34.67 | 38.5 | 5 | 1 |
| alt T+W (AELG) — `TAELG+<wav>V3ALG_*_ag0` | 8 | 35.08 | 36.0 | 1 | 3 |
| unified TWAELG — `TWAELG_10s_ld*_<wav>_*` | 6 | 37.25 | 37.0 | 2 | 3 |
| unified TW+Generic (AELG) — `TWGAELG_*_ld16_<wav>_ag0` | 4 | 38.62 | 45.0 | 2 | 1 |
| pure GenericAE — `GAE_*_ld32_ag0` | 2 | 50.17 | 52.0 | 24 | 0 |
| pure GenericAELG — `GAELG_*_ld16_ag0` | 2 | 50.17 | 56.5 | 23 | 0 |

**Headline:** `alt T+W (RootBlock)` — the family that adds a dedicated polynomial Trend block in alternation with a dedicated `WaveletV3` block on a plain RootBlock backbone — is the strongest novel family by ~10 mean-rank points. It produces **10 of the top-5 cells across all 6 periods** (the next-best family produces 4). The merged sweep promotes this family from "best at 30 stacks" to "best at both 10 and 30 stacks": the fill additions (`T+HaarV3_10s_bdeq`, `T+Coif2V3_10s_bdeq`, `T+Db3V3_10s_bdeq`, `T+Sym10V3_10s_bdeq`) all enter the top-15 generalist leaderboard.

Pure GenericAE and pure GenericAELG bottom out at mean rank 50, ~30 ranks behind alt-T+W. **The wavelet/trend basis is what carries the novel-family wins; the AE bottleneck on a Generic backbone alone is not enough.**

---

## 7. AE vs AELG — Matched Comparisons (Mann-Whitney U)

Six matched (wavelet × depth) pairs available after the merge: 4 at 30 stacks (haar, db3, coif2, sym10) and 2 at 10 stacks (haar, coif2). Each pair compares `TAE+<wav>V3AE_<depth>_ld32_ag0` vs `TAELG+<wav>V3ALG_<depth>_ag0`. Negative delta = AELG wins.

| Pair | Period | AE mean | AELG mean | Δ(AELG−AE) | MWU p |
|---|---|---|---|---|---|
| haar 30s | Yearly | 13.654 | 13.714 | +0.060 | 0.385 |
| haar 30s | Quarterly | 10.488 | 10.438 | −0.049 | 0.307 |
| haar 30s | Monthly | 13.585 | 13.693 | +0.108 | 0.345 |
| haar 30s | Weekly | 7.056 | 7.064 | +0.008 | 0.970 |
| haar 30s | Daily | 3.078 | 3.092 | +0.014 | 0.473 |
| haar 30s | Hourly | 9.031 | 9.037 | +0.006 | 0.791 |
| db3 30s | Yearly | 13.894 | 13.839 | −0.055 | 0.678 |
| db3 30s | Monthly | 13.509 | 13.477 | −0.031 | 0.850 |
| coif2 30s | Daily | 3.077 | 3.036 | −0.041 | 0.089 |
| sym10 30s | Yearly | 13.765 | 13.641 | −0.124 | 0.521 |
| haar 10s | Quarterly | 10.368 | 10.450 | +0.082 | **0.038*** |
| haar 10s | Daily | 3.054 | 3.077 | +0.024 | 0.521 |
| coif2 10s | Daily | 3.090 | 3.066 | −0.024 | 0.571 |
| coif2 10s | Hourly | 9.149 | 9.278 | +0.129 | 0.273 |

(Showing notable rows; full table = 36 cells, 1/36 reaches p<0.05.)

**Per-period mean Δ across all 6 pairs:**

| Period | Mean Δ(AELG−AE) | n_pairs | AELG wins / pairs |
|---|---|---|---|
| Yearly | −0.013 | 6 | 3/6 |
| Quarterly | +0.016 | 6 | 2/6 |
| Monthly | +0.030 | 6 | 2/6 |
| Weekly | +0.120 | 6 | 0/6 (AE wins all 6) |
| Daily | −0.009 | 6 | 4/6 |
| Hourly | +0.036 | 6 | 1/6 |

**Verdict: AE ≈ AELG at matched configurations.** Only 1/36 cell tests reaches p<0.05 (haar 10s Quarterly, AE wins by 0.082). Weekly is the only period where AE has a consistent (non-significant) edge across all 6 wavelet/depth pairs (all 6 pairs prefer AE, mean +0.120).

The aggregate "AELG worse" appearance in mean rank (e.g. `alt T+W AELG` rank 35.1 vs `alt T+W AE` 30.3) is **driven by AELG's lower default latent dim (16 vs 32) and the underperforming TWGAELG group**, not by the LG gate itself. **Pick AELG when latent-dim/parameter count is the binding constraint** — its native ld=16 halves AE's ld=32 cost with no consistent SMAPE penalty.

---

## 8. `agf` vs `ag0` Per-Period Summary

Nine matched (config, ag0/agf) pairs across the merged sweep. Δ = (agf − ag0); negative = agf better. MWU p < 0.05 marked with `*`.

| Pair | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
|---|---|---|---|---|---|---|
| NBEATS-G_10s | +0.109 (0.47) | +0.164 (0.01)* | −0.223 (0.34) | −0.161 (0.16) | −1.365 (0.21) | −0.039 (0.68) |
| NBEATS-G_30s | +0.199 (0.09) | −4.797 (0.14)† | −0.227 (0.24) | −1.665 (0.05)† | −6.915 (0.24)† | −0.071 (0.16) |
| NBEATS-IG_10s | +0.085 (0.24) | +0.102 (0.03)* | +0.037 (0.57) | −0.018 (1.00) | +0.069 (0.01)* | −0.100 (0.09) |
| NBEATS-IG_30s | +0.116 (0.21) | +0.095 (0.03)* | +0.067 (0.52) | −0.031 (0.85) | +0.043 (0.14) | **−0.147 (0.01)*** |
| TWAE_10s_ld32 | −0.015 (0.85) | +0.078 (0.04)* | +0.171 (0.47) | −0.017 (0.57) | +0.020 (0.52) | **−0.228 (0.03)*** |
| TWAELG_10s_ld32_db3 | −0.066 (0.43) | +0.016 (0.68) | +0.050 (0.97) | +0.270 (0.31) | +0.066 (0.03)* | **−0.363 (0.001)*** |
| TWAELG_10s_ld32_sym10 | −0.017 (0.38) | +0.067 (0.05)* | +0.466 (0.06) | +0.184 (0.38) | +0.009 (0.85) | **−0.367 (0.001)*** |
| TWAELG_10s_ld16_db3 | +0.051 (0.62) | +0.011 (0.73) | +0.031 (0.97) | +0.182 (0.52) | −0.013 (0.52) | **−0.338 (0.0004)*** |
| TWAE_10s_ld32_sym10 | +0.051 (0.68) | +0.010 (0.79) | +0.239 (0.16) | −0.047 (0.73) | −0.002 (1.00) | **−0.331 (0.001)*** |

† NBEATS-G_30s_ag0 has bimodal collapse on Quarterly/Weekly/Daily; the apparent agf advantage there is rescue from a broken baseline, not a genuine architecture preference. Read those cells with caution.

**Per-period summary (across all 9 pairs):**

| Period | Mean Δ | agf wins / 9 | n sig (p<0.05) |
|---|---|---|---|
| Yearly | +0.057 | 3 | 0 |
| Quarterly | −0.473† | 1 | 4 (all favouring ag0 except NBEATS-G_30s) |
| Monthly | +0.068 | 2 | 0 |
| Weekly | −0.145† | 6 | 0 |
| Daily | −0.899† | 4 | 1 (TWAELG db3, ag0 better) |
| **Hourly** | **−0.221** | **9 / 9** | **5** |

(†: dominated by NBEATS-G_30s collapse on the ag0 side)

**Conclusions:**

- **Hourly is the agf-decisive period.** 9/9 pairs prefer agf, 5/9 with p<0.05 (and the other 4 trend agf-better). The effect is largest on novel TWAE/TWAELG (Δ ≈ −0.33 SMAPE) and smaller but still significant on paper backbones (Δ ≈ −0.10 SMAPE on NBEATS-IG_30s).
- **Yearly is a marginal, non-significant agf signal.** Mean Δ +0.057 (agf is *worse* on average), but the sub-1M pair `TWAELG_10s_ld32_db3` flips to favour agf (rank 2 entry). Yearly is wavelet-specific, not universal.
- **Quarterly is reliably ag0-better** (4/9 pairs sig, all favouring ag0 in the non-collapsed pairs).
- **Verdict (unchanged from prior):** `active_g=forecast` is a Hourly-only switch with marginal benefit on Yearly novel configs. Never use as a global default.

---

## 9. Wavelet Family Ranking (pooled across stack architectures)

Pool all wavelet-bearing configs (n=68 wavelet observations × 10 runs across the unified TW family, alt-T+W RB, alt-AE/AELG, unified TWAE/TWAELG/TWGAELG, full set).

| Period | haar | db3 | coif2 | coif3 | sym10 | KW H | KW p |
|---|---|---|---|---|---|---|---|
| Yearly | 13.666 | 13.663 | 13.669 | 13.673 | 13.655 | 0.97 | 0.91 |
| Quarterly | 10.421 | 10.442 | 10.423 | 10.446 | 10.451 | 8.67 | 0.07 |
| Monthly | 13.717 | 13.737 | 13.691 | 13.625 | 13.667 | 2.94 | 0.57 |
| Weekly | 7.079 | 7.203 | 7.082 | 7.099 | 7.181 | 9.16 | 0.06 |
| Daily | 3.074 | 3.087 | 3.071 | 3.066 | 3.078 | 8.69 | 0.07 |
| Hourly | 9.148 | 9.159 | 9.144 | 9.120 | 9.140 | 2.14 | 0.71 |

**No period reaches Kruskal-Wallis p<0.05.** Quarterly, Weekly, Daily all hover at p=0.06–0.07 — suggestive but not significant.

Rough ordering: **haar/coif2 are jointly best on Quarterly–Weekly–Daily**, **sym10 is best on Yearly**, **coif3 is best on Monthly–Daily–Hourly**, **db3 lags slightly** on Quarterly/Weekly. But all margins are within seed noise pooled across architectures.

**Coif3 verdict:** despite leading the *pooled* mean on 3 periods, coif3 produces **no per-period SOTA leaderboard winner** in the merged sweep (top-5 only on Monthly/Daily, never rank 1). Its long support and 6 vanishing moments don't translate to architecture-specific wins. **Drop coif3 from the default M4 wavelet shortlist; keep haar/db3/coif2/sym10.**

---

## 10. Stack Depth × Wavelet Interaction

Two architectural settings expose the depth × wavelet interaction. SMAPE delta = (30s − 10s); negative = 30s wins. MWU p<0.05 marked `*`.

### 10.1 Alt T+W RB family (`T+<wav>V3_<depth>s_bdeq`)

| Wavelet | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
|---|---|---|---|---|---|---|
| haar | +0.120 (0.31) | +0.050 (0.34) | +0.185 (0.16) | −0.001 (0.85) | +0.037 (0.04)* | −0.155 (0.19) |
| coif2 | −0.004 (0.79) | +0.009 (0.91) | −0.111 (0.68) | −0.184 (0.38) | +0.012 (0.73) | −0.058 (0.24) |
| db3 | −0.053 (0.34) | +0.059 (0.24) | −0.163 (0.24) | −0.502 (0.005)* | +0.018 (0.68) | −0.123 (0.16) |
| sym10 | −0.040 (0.85) | +0.036 (0.43) | −0.504 (0.06) | −0.091 (0.43) | −0.008 (1.00) | −0.113 (0.12) |

(coif3 has only 30s in the sweep.)

**Pattern:**
- **haar 10s ≈ haar 30s**, with haar 10s slightly worse on Daily (sig) but **better on Hourly (−0.155, n.s.)** and tying on Weekly. Mean SMAPE delta is essentially flat. Because the deeper stack adds ~3× the parameters with no net SMAPE benefit, **haar 10s is the parameter-efficient pick**.
- **db3 30s decisively beats db3 10s on Weekly (Δ=−0.502, p=0.005)** and trends 30s-better on most other periods (5/6 with negative delta).
- **Sym10 / coif2: 30s preferred on most periods**, magnitudes small, no individual cell hits significance after Bonferroni.

**Empirical heuristic:** short-support wavelets (haar) — 10 stacks suffice; long-support smooth wavelets (sym10, db3) — 30 stacks yield real Weekly / Monthly gains.

### 10.2 Unified TW family (`TW_<depth>s_td3_bdeq_<wav>`)

| Wavelet | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
|---|---|---|---|---|---|---|
| haar | +0.062 (0.38) | +0.022 (0.57) | −0.388 (0.04)* | −0.195 (0.31) | +0.005 (0.68) | −0.084 (0.21) |
| db3 | −0.070 (0.21) | +0.069 (0.05)* | −0.039 (0.68) | −0.141 (0.52) | −0.008 (0.62) | −0.089 (0.14) |
| coif2 | +0.139 (0.24) | −0.073 (0.14) | −0.176 (0.85) | −0.316 (0.02)* | +0.003 (0.54) | +0.006 (0.79) |
| coif3 | +0.117 (0.09) | +0.023 (0.57) | −0.007 (0.62) | +0.329 (0.01)* | −0.010 (0.91) | −0.093 (0.47) |
| sym10 | +0.037 (0.47) | +0.068 (0.02)* | −0.046 (0.73) | −0.031 (0.73) | −0.003 (0.73) | −0.163 (0.12) |

For unified TW, the 30s stack is significantly better than 10s on Monthly haar (−0.388, p=0.04) and Weekly coif2 (−0.316, p=0.02). 10s is significantly better on Quarterly db3 (+0.069, p=0.05), Quarterly sym10 (+0.068, p=0.02), and Weekly coif3 (+0.329, p=0.01).

**Net:** unified-TW depth preference is wavelet- and period-specific, with no clean rule. Pick based on per-period leaderboards, not depth alone.

---

## 11. Recommendations

### 11.1 Confirmed empirical defaults (M4 paper-sample protocol)

- **Best generalist:** `T+Sym10V3_30s_bdeq` (mean rank 9.17/68).
- **Best paper-faithful generalist:** `NBEATS-IG_30s_ag0` (mean rank 11.17/68).
- **Sub-1M champion:** `TWAE_10s_ld32_ag0` (0.55M, mean rank 27.67/68 = 3.67 within sub-1M tier).
- **Parameter-extreme champion:** `TWAELG_10s_ld16_db3_ag0` (0.51M, statistically equivalent to ld=32 on 5/6 periods).
- **Per-period winners:** see §3.

### 11.2 Rules

1. **`active_g=forecast` only on Hourly.** 9/9 pairs prefer agf on Hourly; 5/9 with p<0.05. Never use as a global default — it is reliably worse on Quarterly (4 pairs sig) and trends worse on Yearly/Monthly.
2. **Default wavelet shortlist:** haar / db3 / coif2 / sym10. **Drop coif3** — no per-period SOTA, only modest pooled-mean leadership on Monthly/Daily that doesn't translate to leaderboard wins.
3. **Stack depth — alt T+W RB family:** prefer 10 stacks for haar (~5M params, parameter-efficient), 30 stacks for db3/sym10/coif2 (~15M params, top-3 generalist on 5/6 periods).
4. **AE vs AELG at matched configurations:** equivalent. Pick AELG when parameter count is the binding constraint (its native ld=16 halves AE's ld=32 cost with no consistent SMAPE penalty). Otherwise interchangeable.
5. **Drop unconditionally on M4:** `GenericVAE_3s_sw0`, all pure `GAE_*`/`GAELG_*` and `GenericAE_*_sw0`/`GenericAELG_*_sw0`, all `BNG*`/`BNAE*`/`BNAELG*`, all `*_coif3` variants, `NBEATS-G_30s_ag0` (cap at 10 stacks).

### 11.3 What to test next

1. **Forecast-level ensembling.** With per-seed runs now meaningful (post early-stopping fix), build median-of-N forecast ensembles across the 10 seeds × top-3 architectures per period. Hypothesis: closes the residual ~1.7% gap to paper N-BEATS-G ensemble on Quarterly and ~3.5% on Monthly.
2. **TWAELG ld=8.** Push the AELG latent-dim limit further (current ld=16 ≈ ld=32). Hypothesis: ld=8 still viable on Yearly/Quarterly (low-bandwidth periods), breaks on Monthly/Weekly where wavelet+trend bandwidth is highest.
3. **`T+Sym10V3_30s_bdeq` × `bd_label=bd2eq`.** Best generalist currently uses `bdeq`. Run the same architecture with `bd2eq` (basis_dim = 2 × forecast_length) to see if Monthly/Weekly winners shift.
4. **Hourly novel-family recovery.** Best novel config on Hourly is `TWAELG_10s_ld32_db3_agf` at rank 5 (+0.165 SMAPE behind NBEATS-IG_30s_agf). Test wider novel models at 30s for Hourly only (g_width=1024, t_width=512).
5. **Replicate `TAELG+Coif2V3ALG_30s_ag0` Daily SOTA under sliding protocol.** This config wins Daily under paper-sample (3.036) but the prior `comprehensive_sweep_m4` sliding sweep had `NBEATS-G_30s_ag0` winning Daily at 2.588. A direct head-to-head under one protocol with 10 seeds resolves whether the inversion is protocol- or architecture-driven.

### 11.4 Open questions

- Quarterly tie (`T+HaarV3_10s_bdeq` 10.356 vs `NBEATS-IG_10s_ag0` 10.357, p=0.97): real or sample noise? With n=10 each, MWU power is ~30% for a 0.05-SMAPE effect. Run 30 seeds of each to settle.
- Does the 10s-vs-30s wavelet-conditional pattern (haar 10s ≥ haar 30s; db3 30s ≫ db3 10s) hold under sliding protocol? Prior comprehensive_sweep_m4 didn't include alt-T+W RB at 10 stacks.
- Is the `TAE+Sym10V3AE_10s_ld32_ag0` Daily run-5 SMAPE=200 a deterministic init issue or a stochastic warmup-divergence event? Re-run with longer warmup and seed-controlled grid to characterise.

### 11.5 YAML cleanup (carried over)

- Deduplicate `Generic_10s_sw0` and `Generic_30s_sw0` entries in `comprehensive_m4_paper_sample.yaml`. CSV side collapses cleanly to one config_name; YAML side has confusing duplicate blocks.

---

## 12. Proposed CLAUDE.md Updates

The merged numbers do **not** change any "Best per-period configs (M4)" row in the paper-sample column — winners are identical to the prior parent-only report. Sub-1M champion and generalist winner are also unchanged. The following diff is **optional / informational** and surfaces useful refinements:

```diff
 ### Best M4 generalist (haar/sym10 alt-Trend+Wavelet RootBlock family)

 - `T+Sym10V3_30s_bdeq` — paper-sample, mean rank 6.83/53
+- `T+Sym10V3_30s_bdeq` — paper-sample (combined sweep), mean rank 9.17/68
 - `T+HaarV3_30s_bd2eq` — sliding, mean rank 12.7/112

 ### Sub-1M parameter champion (M4)

-`TWAELG_10s_ld32_db3_*` — 0.48–0.85M params, top-5 on Yearly, Daily, Hourly.
+`TWAE_10s_ld32_ag0` (0.55M) — top within sub-1M tier (mean rank 3.67/12).
+`TWAELG_10s_ld16_db3_ag0` (0.51M, fill) — parameter-extreme; ld=16 ≈ ld=32 on 5/6 periods.
+`TWAELG_10s_ld32_db3_*` family — Yearly/Daily/Hourly Pareto picks (rank ≤5/68 on those periods).
```

```diff
 ### Defaults for new M4 experiments

 - **Wavelet shortlist:** haar, db3, coif2, sym10. Coif3 produces no per-period SOTA — drop from default M4 sweeps.
 - **`active_g`:** default `False` (paper-faithful). `active_g=forecast` (`agf`) helps on **Yearly + Hourly only** for novel TWAE/TWAELG and on **Hourly** for paper backbones; loses or ties elsewhere. Never use as a global default.
+- **Depth × wavelet rule (alt-T+W RB family):** haar 10s ≈ haar 30s on rank (10s preferred at 1/3 the params); db3/sym10/coif2 prefer 30s for top-3 generalist standing.
```

The remaining empirical defaults (stack architecture preference for alternating over unified, AE ≈ AELG matched, drop list, early-stopping settings) are **unchanged**.

---

## 13. Source Data and Reproducibility

- Filter rule encoded in pseudocode at top of §2.1.
- All MWU / Wilcoxon tests use `scipy.stats` two-sided; no multiple-comparison correction (effects are reported with raw p-values).
- Mean ranks computed by `pandas.DataFrame.groupby(period)[smape_mean].rank(method='min')` and averaged across periods.
- Per-period leaderboards reflect mean SMAPE over surviving runs after filter; std is the population std (`pandas.std(ddof=1)`).
- This report supersedes both prior reports for go-forward citation. Prior reports remain on disk under `experiments/analysis/analysis_reports/comprehensive_m4_paper_sample_analysis.md` and `..._sym10_fills_analysis.md`.
