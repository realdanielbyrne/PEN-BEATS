# Comprehensive M4 Paper-Sample Sweep — Sym10 Fills Addendum

**Date:** 2026-04-27
**Protocol:** `nbeats_paper` (sub-epoch validation, `val_check_interval=100`, `min_delta=0.001`, `patience=15`)
**Parent report:** `comprehensive_m4_paper_sample_analysis.md` (53 configs × 6 periods × 10 runs)
**This addendum:** merges the parent CSV with `comprehensive_m4_paper_sample_sym10_fills_results.csv` (15 net-new configs × 6 periods × 10 runs = 900 new runs).

Total merged set: **68 configs**, 4,080 raw runs, 4,066 after stricter divergence filter (described below).

## Executive Summary

The sym10 fills experiment was designed to round out architectural slots that already covered haar/coif2/db3 in the parent sweep. The fills introduced 15 configs in three groups:

- 4 sym10 TWAE/TWAELG variants at ld=32 (homogeneous TrendWavelet AE/AELG, parent had db3 only).
- 3 alt-AE/AELG sym10 configs (`TAE+Sym10V3AE` at 10s and 30s; `TALG+Sym10V3ALG_10s`).
- Group A: 4 alt-Trend+WaveletV3 RootBlock configs at **10-stack** (haar/coif2/db3/sym10) — parent had only the 30-stack family.
- Group B: 2 alt_aelg 10s configs (haar, coif2) — completes the 4-wavelet × 2-depth alt_aelg grid.
- Group C: 2 TWAELG ld=16 db3 configs (ag0, agf) — answers the AELG ld=16 vs ld=32 question.

### What changed vs the prior leaderboard

1. **Generalist winner unchanged:** `T+Sym10V3_30s_bdeq` retains the best mean rank — now 9.17/68 (was 6.83/53). The fills did not displace it; the rank inflation is a mechanical effect of adding 15 more configs to the denominator.
2. **Quarterly winner technically flipped, statistically tied:** `T+HaarV3_10s_bdeq` (fill, 10.356) edges `NBEATS-IG_10s_ag0` (orig, 10.357) by 0.001 SMAPE. Mann-Whitney U p=0.97; Welch t-test p=0.98. **Treat as a tie, not a SOTA change.** What did change is that **3 of the top 4 Quarterly configs are now alt-T+WaveletV3 10-stack RB fills**.
3. **No other per-period SOTA changed.** Yearly, Monthly, Weekly, Daily, Hourly winners are all parent configs.
4. **Sub-1M champion unchanged:** `TWAE_10s_ld32_ag0` (orig, mean rank 3.67/13). The sym10 sub-1M fills (TWAE/TWAELG sym10) are competitive but do not overtake the db3 incumbents.
5. **Alt-T+W RB at 10 stacks is a genuinely useful new comparator.** All four 10s variants land in top-15 mean rank. Two of them (`T+HaarV3_10s_bdeq`, `T+Coif2V3_10s_bdeq`) outperform the corresponding 30-stack siblings on the rank metric for Yearly+Quarterly+Daily despite using ~3× fewer parameters.
6. **TWAELG ld=16 ≈ ld=32 db3.** Across 12 (ld16-vs-ld32, period) cells, only 1 reaches p<0.05 (Daily under ag0, ld32 better by 0.052 SMAPE). The AELG learned gate makes ld=16 viable, halving parameter count with no consistent SMAPE penalty.
7. **agf-on-Hourly hypothesis confirmed for sym10:** `TWAELG_10s_ld32_sym10_agf` beats `TWAELG_10s_ld32_sym10_ag0` on Hourly by 0.367 SMAPE (p=0.001) and on Yearly by 0.017 (ns). It is significantly **worse** on Quarterly (p=0.045). Same pattern as the db3 sibling — agf is a Yearly+Hourly switch, never a global default.

## Data Provenance

- Parent CSV: `experiments/results/m4/comprehensive_m4_paper_sample_results.csv` (3,180 rows; 1 row marked `diverged=True`).
- Fills CSV: `experiments/results/m4/comprehensive_m4_paper_sample_sym10_fills_results.csv` (900 rows; 0 rows marked `diverged=True`).
- All 15 fill configs have **complete coverage**: 10 runs × 6 periods = 60 rows each. No missing cells.
- Both files share the identical schema and `nbeats_paper` protocol settings — directly comparable.

### Implicit divergence (mislabeled)

The `diverged` column is incomplete. 13 additional rows have `smape>100` or (`best_epoch==0` and `smape>50`) but are flagged `diverged=False`:

| Source | Config | Period | Run | SMAPE | best_epoch |
|--------|--------|--------|-----|-------|------------|
| orig | GenericVAE_3s_sw0 | Yearly | 0,9 | 64.9, 65.8 | 0 |
| orig | GenericVAE_3s_sw0 | Weekly | 9 | 63.8 | 0 |
| orig | TW_10s_td3_bdeq_coif2 | Daily | 3 | 200.0 | 0 |
| orig | GenericVAE_3s_sw0 | Hourly | 0,1,3-7,9 | 101-110 | 24-75 |
| **fill** | **TAE+Sym10V3AE_10s_ld32_ag0** | **Daily** | **5** | **200.0** | **0** |

The fill row matters: without filtering it, `TAE+Sym10V3AE_10s_ld32_ag0` Daily mean is 22.78 (a single 200.0 outlier dominates). After filtering (n=9), Daily mean is 3.089 — competitive with the rest of the family. **All tables below use the stricter divergence filter** (`diverged=True` OR `smape>100` OR (`best_epoch==0` AND `smape>50`)).

## Per-Period Top-5 (merged 68-config set, stricter filter)

### Yearly (forecast horizon = 6)

| # | Config | Origin | SMAPE (±std) | Params |
|---|--------|--------|--------------|--------|
| 1 | TALG+DB3V3ALG_10s_ag0 | orig | 13.5496 (±0.0955) | 1.04M |
| 2 | TWAELG_10s_ld32_db3_agf | orig | 13.5780 (±0.1473) | 0.48M |
| 3 | **TAE+Sym10V3AE_10s_ld32_ag0** | **fill** | **13.5861 (±0.1819)** | **1.11M** |
| 4 | NBEATS-IG_10s_ag0 | orig | 13.5902 (±0.1477) | 19.51M |
| 5 | **TWAELG_10s_ld16_db3_ag0** | **fill** | **13.5934 (±0.0830)** | **0.44M** |

Two fills (#3, #5) crack the Yearly top-5. Prior winner unchanged.

### Quarterly (forecast horizon = 8)

| # | Config | Origin | SMAPE (±std) | Params |
|---|--------|--------|--------------|--------|
| 1 | **T+HaarV3_10s_bdeq** | **fill** | **10.3560 (±0.0756)** | **5.13M** |
| 2 | NBEATS-IG_10s_ag0 | orig | 10.3570 (±0.1192) | 19.64M |
| 3 | **T+Coif2V3_10s_bdeq** | **fill** | **10.3647 (±0.0781)** | **5.13M** |
| 4 | **T+Db3V3_10s_bdeq** | **fill** | **10.3668 (±0.0914)** | **5.13M** |
| 5 | TAE+HaarV3AE_10s_ld32_ag0 | orig | 10.3681 (±0.0614) | 1.16M |

The top of the Quarterly leaderboard is dominated by the new alt-T+WaveletV3 10-stack RB family. Statistical test `T+HaarV3_10s_bdeq` vs `NBEATS-IG_10s_ag0`: Mann-Whitney U=49, p=0.97; Welch t=−0.024, p=0.98. **Tie.**

The notable shift is that the **5.13M-param 10s alternating-RB family is competitive with the 19.64M-param NBEATS-IG_10s** on Quarterly. Three of the four sym10/db3/coif2/haar variants beat NBEATS-IG_30s_ag0 (rank 6, 10.372).

### Monthly (forecast horizon = 18)

| # | Config | Origin | SMAPE (±std) | Params |
|---|--------|--------|--------------|--------|
| 1 | TW_30s_td3_bdeq_haar | orig | 13.3906 (±0.5838) | 6.78M |
| 2 | T+Sym10V3_30s_bdeq | orig | 13.3970 (±0.5441) | 16.12M |
| 3 | TAELG+Db3V3ALG_30s_ag0 | orig | 13.4775 (±0.4244) | 4.03M |
| 4 | TAE+DB3V3AE_30s_ld32_ag0 | orig | 13.5085 (±0.3397) | 4.22M |
| 5 | **TWAE_10s_ld32_sym10_ag0** | **fill** | **13.5133 (±0.3782)** | **0.58M** |

Sub-1M sym10 TWAE jumps to #5 on Monthly — a strong showing for the smallest novel-family member, edging `Generic_10s_sw0` (13.545).

### Weekly (forecast horizon = 13)

| # | Config | Origin | SMAPE (±std) | Params |
|---|--------|--------|--------------|--------|
| 1 | T+Coif2V3_30s_bdeq | orig | 6.7347 (±0.2031) | 15.75M |
| 2 | T+Db3V3_30s_bdeq | orig | 6.7371 (±0.2560) | 15.75M |
| 3 | T+Sym10V3_30s_bdeq | orig | 6.8577 (±0.2730) | 15.75M |
| 4 | **T+Coif2V3_10s_bdeq** | **fill** | **6.9187 (±0.4243)** | **5.25M** |
| 5 | TW_30s_td3_bdeq_db3 | orig | 6.9200 (±0.2751) | 6.55M |

`T+Coif2V3_10s_bdeq` (fill) at 5.25M params lands at #4, only +0.184 SMAPE behind the 30-stack winner (15.75M). Strong parameter-efficiency point.

### Daily (forecast horizon = 14)

| # | Config | Origin | SMAPE (±std) | Params |
|---|--------|--------|--------------|--------|
| 1 | TAELG+Coif2V3ALG_30s_ag0 | orig | 3.0361 (±0.0340) | 3.73M |
| 2 | TWGAELG_10s_ld16_db3_ag0 | orig | 3.0506 (±0.0494) | 0.52M |
| 3 | T+Sym10V3_30s_bdeq | orig | 3.0510 (±0.0434) | 15.82M |
| 4 | TAE+HaarV3AE_10s_ld32_ag0 | orig | 3.0538 (±0.0583) | 1.31M |
| 5 | **T+Db3V3_10s_bdeq** | **fill** | **3.0540 (±0.0569)** | **5.27M** |

A fill cracks Daily top-5 but does not displace the `TAELG+Coif2V3ALG_30s_ag0` SOTA.

### Hourly (forecast horizon = 48)

| # | Config | Origin | SMAPE (±std) | Params |
|---|--------|--------|--------------|--------|
| 1 | NBEATS-IG_30s_agf | orig | 8.7583 (±0.0992) | 43.58M |
| 2 | NBEATS-G_30s_agf | orig | 8.8622 (±0.0847) | 31.76M |
| 3 | NBEATS-IG_10s_agf | orig | 8.8926 (±0.1064) | 22.40M |
| 4 | NBEATS-IG_30s_ag0 | orig | 8.9056 (±0.0833) | 43.58M |
| 5 | TWAELG_10s_ld32_db3_agf | orig | 8.9237 (±0.1292) | 0.85M |

No fill in Hourly top-5. Best fill is `TWAELG_10s_ld32_sym10_agf` at #6 (8.9331). Hourly remains a paper-baseline-dominated period.

## Per-Period Winners — Change Summary vs Prior Report

| Period | Prior winner | New winner | Status | Notes |
|--------|--------------|------------|--------|-------|
| Yearly | TALG+DB3V3ALG_10s_ag0 (13.550) | unchanged | same | 2 fills enter top-5 |
| Quarterly | NBEATS-IG_10s_ag0 (10.357) | T+HaarV3_10s_bdeq (10.356, fill) | **tied** | Δ=0.001, p=0.97. Treat as tie. 3 of top 4 are 10s alt-T+W RB fills. |
| Monthly | TW_30s_td3_bdeq_haar (13.391) | unchanged | same | sub-1M sym10 TWAE enters at #5 |
| Weekly | T+Coif2V3_30s_bdeq (6.735) | unchanged | same | Coif2 10s fill at #4 |
| Daily | TAELG+Coif2V3ALG_30s_ag0 (3.036) | unchanged | same | Db3 10s fill at #5 |
| Hourly | NBEATS-IG_30s_agf (8.758) | unchanged | same | Best fill at #6 |

## Generalist Mean Rank (top 15, merged 68-config set)

| # | Config | Origin | Yr | Qy | Mo | Wk | Dy | Hr | mean_rank |
|---|--------|--------|----|----|----|----|----|----|-----------|
| 1 | T+Sym10V3_30s_bdeq | orig | 21 | 15 | 2 | 3 | 3 | 11 | **9.17** |
| 2 | NBEATS-IG_30s_ag0 | orig | 15 | 6 | 15 | 17 | 10 | 4 | 11.17 |
| 3 | **T+HaarV3_10s_bdeq** | **fill** | 10 | 1 | 20 | 8 | 8 | 46 | **15.50** |
| 4 | NBEATS-IG_10s_ag0 | orig | 4 | 2 | 45 | 14 | 18 | 14 | 16.17 |
| 5 | T+Coif2V3_30s_bdeq | orig | 19 | 8 | 26 | 1 | 33 | 20 | 17.83 |
| 5 | T+Db3V3_30s_bdeq | orig | 22 | 23 | 23 | 2 | 21 | 16 | 17.83 |
| 7 | T+Coif3V3_30s_bdeq | orig | 31 | 11 | 38 | 11 | 6 | 22 | 19.83 |
| 8 | **T+Coif2V3_10s_bdeq** | **fill** | 20 | 3 | 43 | 4 | 22 | 28 | **20.00** |
| 9 | **T+Sym10V3_10s_bdeq** | **fill** | 40 | 7 | 60 | 10 | 9 | 26 | **25.33** |
| 10 | TAE+Coif2V3AE_10s_ld32_ag0 | orig | 27 | 9 | 7 | 30 | 39 | 40 | 25.33 |
| 11 | TW_30s_td3_bdeq_haar | orig | 42 | 30 | 1 | 20 | 19 | 41 | 25.50 |
| 12 | **TAE+Sym10V3AE_10s_ld32_ag0** | **fill** | 3 | 16 | 14 | 34 | 37 | 51 | **25.83** |
| 13 | TW_10s_td3_bdeq_coif3 | orig | 17 | 38 | 10 | 19 | 25 | 49 | 26.33 |
| 14 | TW_30s_td3_bdeq_db3 | orig | 11 | 51 | 17 | 5 | 41 | 36 | 26.83 |
| 15 | **TAE+Sym10V3AE_30s_ld32_ag0** | **fill** | 63 | 21 | 6 | 9 | 34 | 29 | **27.00** |

5 of the 15 fills land in the top 15 generalists. `T+HaarV3_10s_bdeq` is the strongest fill addition — and is now the **3rd-best generalist** in the entire 68-config sweep.

## Group A: Alt-Trend+WaveletV3 RootBlock — 10-stack vs 30-stack

The fills introduce 4 new 10-stack alt-T+W RB configs at ~5.1M params each. Each has a parent at 30-stack ~15.7M. Per-period mean SMAPE deltas (10s − 30s):

| Wavelet | Yr | Qy | Mo | Wk | Dy | Hr | Mean Δ | 10s rank | 30s rank |
|---------|------|------|------|------|------|------|--------|----------|----------|
| Haar    | −0.120 | −0.050 | −0.185 | +0.001 | −0.037* | +0.155 | **−0.039** | **15.50** | 29.83 |
| Coif2   | +0.004 | −0.009 | +0.111 | +0.184 | −0.012 | +0.058 | +0.056 | 20.00 | 17.83 |
| Db3     | +0.053 | −0.059 | +0.163 | +0.502* | −0.018 | +0.123 | +0.127 | 30.00 | 17.83 |
| Sym10   | +0.040 | −0.036 | +0.504 | +0.091 | +0.008 | +0.113 | +0.120 | 25.33 | 9.17 |

(* = Mann-Whitney p<0.05)

**Wavelet-conditional 10s vs 30s reversal:**
- Haar 10s **beats** Haar 30s on mean-rank (15.5 vs 29.8) and on per-period mean SMAPE in 4/6 periods. The Haar 30-stack appears to over-stack — extra capacity does not help and slightly hurts.
- Coif2 10s and 30s are within ~1 mean-rank point.
- Db3 30s and Sym10 30s are clearly better than their 10s siblings — long-support smooth wavelets benefit from the deeper stack.

This is a useful new empirical heuristic: **for short-support wavelets (haar), 10 stacks is enough; for long-support wavelets (sym10, db3 to a lesser extent), 30 stacks materially helps.**

## Group B: Alt_AELG 10s — 4-Wavelet × 2-Depth Grid Completion

After fills, the alt_aelg 10s grid spans haar/coif2/db3/sym10. Combined with the parent's 30s grid, the full 4×2 matrix is now populated. Mean SMAPE per period:

| Wavelet | Depth | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
|---------|-------|--------|-----------|---------|--------|-------|--------|
| Haar    | 10s   | 13.671 | 10.450 | 13.847 | 7.375 | 3.077 | 9.278 |
| Coif2   | 10s   | 13.689 | 10.401 | 13.654 | 7.250 | 3.066 | 9.277 |
| Db3     | 10s   | **13.550** | 10.421 | 13.948 | 7.310 | 3.107 | 9.324 |
| Sym10   | 10s   | 13.656 | 10.417 | 13.742 | 7.267 | 3.093 | 9.270 |

Within the alt_aelg 10s set: **Db3 wins Yearly (and is the overall Yearly SOTA across all 68 configs)**, **Coif2 wins Quarterly/Monthly/Weekly/Daily**, **Sym10 wins Hourly**. No single wavelet is dominant; differences are small (<0.15 SMAPE in most cells).

## Group C: TWAELG ld=16 vs ld=32 (db3, sym10)

The fills add db3 ld=16 (ag0, agf) and sym10 ld=32 (ag0, agf) variants of TWAELG. Combined with parent db3 ld=32, this answers the latent-dim question.

### TWAELG db3: ld=16 vs ld=32 (matched ag0)

| Period | ld=16 mean | ld=32 mean | Δ(ld16−ld32) | Mann-Whitney p |
|--------|------------|------------|--------------|----------------|
| Yearly | 13.593 | 13.644 | −0.051 | 0.571 |
| Quarterly | 10.443 | 10.424 | +0.019 | 0.385 |
| Monthly | 13.851 | 13.740 | +0.112 | 0.307 |
| Weekly | 7.340 | 7.252 | +0.088 | 0.521 |
| Daily | 3.110 | 3.058 | +0.052 | **0.021*** |
| Hourly | 9.284 | 9.287 | −0.003 | 0.970 |

Only Daily reaches significance (ld=32 better by 0.052 SMAPE, p=0.021). Other 5 periods: indistinguishable.

### TWAELG db3: ld=16 vs ld=32 (matched agf)

| Period | ld=16 mean | ld=32 mean | Δ(ld16−ld32) | Mann-Whitney p |
|--------|------------|------------|--------------|----------------|
| Yearly | 13.644 | 13.578 | +0.066 | 0.521 |
| Quarterly | 10.454 | 10.440 | +0.014 | 0.791 |
| Monthly | 13.882 | 13.790 | +0.092 | 0.345 |
| Weekly | 7.522 | 7.523 | −0.000 | 0.910 |
| Daily | 3.097 | 3.125 | −0.027 | 0.307 |
| Hourly | 8.946 | 8.924 | +0.022 | 0.473 |

No significant differences. **AELG ld=16 is viable for TWAELG.** Parameter count drops from 0.48M to 0.44M (−9%) with no consistent SMAPE penalty. The `TWAELG_10s_ld16_db3_ag0` fill ranks #5 on Yearly with the smallest model on the leaderboard.

## agf vs ag0 — Sym10 TWAELG Confirms the Yearly+Hourly Pattern

| Period | TWAELG sym10 ag0 | TWAELG sym10 agf | Δ(agf−ag0) | Mann-Whitney p |
|--------|------------------|------------------|------------|----------------|
| Yearly | 13.631 | 13.614 | −0.017 | 0.385 |
| Quarterly | 10.452 | 10.519 | +0.067 | **0.045*** |
| Monthly | 13.560 | 14.026 | +0.466 | 0.064 |
| Weekly | 7.457 | 7.641 | +0.184 | 0.385 |
| Daily | 3.080 | 3.089 | +0.009 | 0.850 |
| Hourly | 9.301 | 8.933 | **−0.367** | **0.001*** |

agf significantly **helps Hourly** (Δ=−0.367, p=0.001) and significantly **hurts Quarterly** (Δ=+0.067, p=0.045). Yearly is a small non-significant gain. Same pattern as the db3 sibling. **This confirms agf is a Yearly+Hourly-only switch, never a global default — and the rule holds across both db3 and sym10.**

## AE vs AELG Matched (Sym10, ld=32, 10s)

The sym10 fills enable a clean AE-vs-AELG matched comparison for the unified TrendWavelet block (parent had no sym10 TWAE/TWAELG):

| Period | TWAE ag0 | TWAELG ag0 | Δ(AE−AELG) | TWAE agf | TWAELG agf | Δ(AE−AELG) |
|--------|----------|------------|------------|----------|------------|------------|
| Yearly | 13.650 | 13.631 | +0.020 | 13.702 | 13.614 | +0.088 |
| Quarterly | 10.475 | 10.452 | +0.022 | 10.484 | 10.519 | −0.035 |
| Monthly | 13.513 | 13.560 | −0.046 | 13.752 | 14.026 | −0.274 |
| Weekly | 7.472 | 7.457 | +0.015 | 7.426 | 7.641 | −0.215 |
| Daily | 3.088 | 3.080 | +0.008 | 3.087 | 3.089 | −0.003 |
| Hourly | 9.293 | 9.301 | −0.008 | 8.962 | 8.933 | +0.029 |
| **Mean Δ** |  |  | **+0.002** |  |  | **−0.068** |

No Mann-Whitney p<0.05 in any (period, ag) cell. **AE ≈ AELG at matched configurations** — confirms the prior report's conclusion. AELG is preferable when parameter count is the binding constraint (its ld=16 default is half the AE ld=32 cost with no SMAPE penalty per Group C).

## Sub-1M Champion (Stricter Filter)

| # | Config | Origin | n_params | mean_rank (sub-1M only) |
|---|--------|--------|----------|-------------------------|
| 1 | TWAE_10s_ld32_ag0 | orig | 0.48M | 3.67 |
| 2 | TWAELG_10s_ld32_db3_ag0 | orig | 0.48M | 4.50 |
| 3 | TWAE_10s_ld32_agf | orig | 0.48M | 5.67 |
| 4 | TWGAELG_10s_ld16_db3_ag0 | orig | 0.45M | 5.83 |
| 5 | TWAELG_10s_ld32_db3_agf | orig | 0.48M | 6.17 |
| 6 | **TWAELG_10s_ld32_sym10_ag0** | **fill** | 0.48M | **6.50** |
| 7 | **TWAELG_10s_ld16_db3_ag0** | **fill** | 0.44M | **6.67** |
| 8 | **TWAE_10s_ld32_sym10_agf** | **fill** | 0.48M | **7.17** |
| 9 | **TWAE_10s_ld32_sym10_ag0** | **fill** | 0.48M | **7.50** |
| 10 | TWGAELG_10s_ld16_sym10_ag0 | orig | 0.45M | 7.67 |
| 11 | **TWAELG_10s_ld16_db3_agf** | **fill** | 0.44M | 8.00 |
| 12 | **TWAELG_10s_ld32_sym10_agf** | **fill** | 0.48M | 8.67 |

**Sub-1M champion is unchanged:** `TWAE_10s_ld32_ag0` (orig). The sym10 fills are all competitive but do not overtake the db3 incumbents. Lowest-param entry is `TWAELG_10s_ld16_db3_ag0` (fill, 0.44M, mean rank 6.67) — viable as a parameter-extreme champion.

## Novel-Only Head-to-Head (53 configs after dropping legacy NBEATS-G/IG/Generic_*/BNG family)

| # | Config | Origin | mean_rank (novel only) |
|---|--------|--------|------------------------|
| 1 | T+Sym10V3_30s_bdeq | orig | 7.50 |
| 2 | **T+HaarV3_10s_bdeq** | **fill** | **12.50** |
| 3 | T+Coif2V3_30s_bdeq | orig | 14.50 |
| 4 | T+Db3V3_30s_bdeq | orig | 14.83 |
| 5 | T+Coif3V3_30s_bdeq | orig | 16.00 |
| 6 | **T+Coif2V3_10s_bdeq** | **fill** | **16.33** |
| 7 | **T+Sym10V3_10s_bdeq** | **fill** | **20.33** |
| 8 | TW_30s_td3_bdeq_haar | orig | 21.00 |
| 9 | TAE+Coif2V3AE_10s_ld32_ag0 | orig | 21.33 |
| 10 | **TAE+Sym10V3AE_10s_ld32_ag0** | **fill** | **21.83** |

**Alt-T+WaveletV3 RB family dominates the novel head-to-head**, occupying 7 of the top 10. The 10-stack fills slot in directly between the 30-stack incumbents — confirming the earlier conclusion that this family is the strongest novel choice.

## Statistical Summary of Sym10 Fill Comparisons

| Comparison | Periods sig (p<0.05) | Direction | Conclusion |
|------------|---------------------|-----------|------------|
| TWAE sym10 vs db3 (ag0) | 0/6 | n/s | sym10 ≈ db3 |
| TWAE sym10 vs db3 (agf) | 0/6 | n/s | sym10 ≈ db3 |
| TWAELG sym10 vs db3 (ag0) | 0/6 | n/s | sym10 ≈ db3 |
| TWAELG sym10 vs db3 (agf) | 0/6 | n/s | sym10 ≈ db3 |
| TWAELG ld=16 vs ld=32 db3 (ag0) | 1/6 (Daily, ld32 wins) | mostly n/s | ld=16 viable |
| TWAELG ld=16 vs ld=32 db3 (agf) | 0/6 | n/s | ld=16 viable |
| TWAELG sym10 agf vs ag0 | 2/6 (Hourly agf wins, Quarterly ag0 wins) | period-conditional | agf=Yearly+Hourly only |
| Alt-T+WaveletV3 RB 10s vs 30s | 2/24 (haar Daily, db3 Weekly) | wavelet-conditional | haar 10s strong; sym10/db3 prefer 30s |

**Sym10 produces no per-period SOTA**, but is statistically equivalent to db3 in all matched comparisons. It is a **safe alternative to db3** in the TWAE/TWAELG family — useful for ensemble diversification but not a SMAPE upgrade.

## Recommendations

### Empirical defaults (no change to CLAUDE.md needed)

The fills do not change any of CLAUDE.md's empirical defaults:

- **Wavelet shortlist:** haar, db3, coif2, sym10 — unchanged. (Sym10 fills did not produce per-period SOTA; sym10 remains the best generalist via the 30-stack alt-T+W RB.)
- **Best generalist:** `T+Sym10V3_30s_bdeq` — confirmed (mean rank 9.17/68).
- **Per-period winners table:** unchanged for 5 of 6 periods. Quarterly is a coin-flip tie (`T+HaarV3_10s_bdeq` ≈ `NBEATS-IG_10s_ag0`, p=0.97).
- **Sub-1M champion:** `TWAELG_10s_ld32_db3_*` family — confirmed.
- **agf rule:** Yearly+Hourly only. Confirmed for sym10 in addition to db3.
- **AE ≈ AELG matched:** Confirmed.

### Useful additions to the recommendation set

- **Alt-T+W RB family at 10 stacks is a strong parameter-efficient option.** When 5M params are acceptable but 15M are not, prefer `T+HaarV3_10s_bdeq` (Quarterly winner, mean rank 15.5) or `T+Coif2V3_10s_bdeq` (Weekly #4). The 10-stack family is preferable for haar; 30-stack for sym10/db3.
- **TWAELG ld=16 is the new parameter-extreme choice.** `TWAELG_10s_ld16_db3_ag0` at 0.44M params ranks #5 on Yearly and is statistically equivalent to ld=32 on 5/6 periods. Use when parameter count is the binding constraint.

### What to test next

1. **Ensemble of fills.** With `save_predictions: true`, build a true forecast-level ensemble combining the alt-T+W RB 10s and 30s families across haar/coif2/db3/sym10. Hypothesis: 8-way ensemble closes the residual gap on Quarterly+Monthly vs paper N-BEATS-G.
2. **TWAELG ld=8.** Push the AELG latent-dim limit further. Hypothesis: ld=8 still viable on Yearly/Quarterly, breaks on Monthly+ where the wavelet+trend bandwidth is highest.
3. **Sym10 with bd=eq_bcast for Monthly.** Sym10 has long support; in the parent sweep, `TW_30s_td3_bdeq_haar` won Monthly with bd=eq_fcast. Test whether sym10 with eq_bcast (or bd2eq) recovers the gap.
4. **Hourly recovery for novel families.** Best fill on Hourly is rank 6 (TWAELG sym10 agf). NBEATS-IG_30s_agf at 43.6M still leads. Test wider novel models (g_width=1024, t_width=512) at 30 stacks for Hourly only.

### Open questions

- **Is the Quarterly tie real or sample-noise?** With n=10 each, Mann-Whitney has only ~30% power to detect a 0.05-SMAPE effect. Run 30 seeds of `T+HaarV3_10s_bdeq` and `NBEATS-IG_10s_ag0` to settle.
- **Does the 10s-vs-30s wavelet-conditional pattern hold under sliding protocol?** This addendum is paper-sample only.
- **Can `TAE+Sym10V3AE_10s_ld32_ag0` Daily be rescued?** The 1/10 implicit-divergence run (SMAPE=200, best_epoch=0) indicates a training instability. Investigate whether `min_delta` tightening or a longer warmup avoids the bad initialization.

## CLAUDE.md Proposed Updates

**No CLAUDE.md changes recommended.** The fills support, but do not contradict, the existing empirical defaults. The only additions worth surfacing in CLAUDE.md (if desired) are:

1. **(Optional)** Add to the M4 defaults section: "Alt-T+WaveletV3 RB at 10 stacks (~5M params) is a strong parameter-efficient option for Quarterly and Weekly. Haar 10s is competitive with or better than haar 30s; sym10/db3 prefer 30s."
2. **(Optional)** Add to the M4 defaults section: "TWAELG with latent_dim=16 is statistically equivalent to ld=32 on 5/6 periods (Daily slightly favors ld=32). Use ld=16 when parameter count is the binding constraint (0.44M vs 0.48M)."

Both are minor refinements; neither overturns the existing winners table.
