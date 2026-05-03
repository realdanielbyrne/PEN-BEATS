# Tiered Basis-Offset on M4 — All-Periods Analysis

**Status (2026-05-03):** standard-LR file complete for 5 periods (Yearly, Quarterly, Monthly, Weekly, Daily) at n=10/config; **Hourly is absent from this run** and is covered separately by `m4_hourly_sym10_tiered_offset_results.csv`. The `*_paperlr_results.csv` companion file is partial (3-10 runs/cell, several 30s configs missing Daily) and is treated here as a sanity-check only.

**Protocol:** paper-sample (`nbeats_paper`), divergence filter `diverged | smape>100 | (best_epoch==0 & smape>50)` removed 5/800 rows. Comparable non-tiered baselines drawn from `comprehensive_m4_paper_sample_results.csv` + `comprehensive_m4_paper_sample_sym10_fills_results.csv` (4080 rows -> 4066 after filter). Seeds do not align across files; significance via Mann-Whitney U with Cliff's delta effect size.

## Executive Summary

- **Cascade hypothesis generalises only partially.** Tiered offset is a clear win on Daily (11/11 configs improve, 4 statistically significant) and a moderate win on Yearly/Quarterly/Monthly (avg delta -0.02 to -0.10 SMAPE). It is **a clear loss on Weekly** (only 2/11 configs improve, avg delta +0.13). The "helps novel wavelet/trend-bias blocks, hurts legacy" Hourly pattern is not visible because no legacy Generic/IG configs were re-run with tiering — selectivity cannot be retested at this perimeter.
- **Three (soft) new per-period bests in the leaderboard, all within noise of prior SOTA.** Daily: `T+Sym10V3_10s_tiered_ag0` 3.012 (vs prior 3.036, p=0.31); Quarterly: `T+Sym10V3_30s_tiered_agf` 10.356 (vs prior 10.357, p=0.73); Monthly: `T+DB3V3_30s_tiered_agf` 13.344 (vs prior 13.391, p=0.62). None reach Wilcoxon/MWU significance against the prior leader, but the same direction holds across many tiered configs in those periods, so the population-level effect is real.
- **Daily is unambiguous — 9 of the top 10 paper-sample Daily entries are now tiered.** The pre-tiered Daily SOTA `TAELG+Coif2V3ALG_30s_ag0` (3.036) drops to rank #9; tiered T+Sym10V3 / T+DB3V3 / TAE+ / TAELG+ variants occupy ranks 1–8. Effect is consistent across stack depth and AE/AELG backbone (Cliff's delta 0.54–0.78).
- **Weekly should drop tiered from defaults.** Best tiered Weekly entry (`T+DB3V3_30s_tiered_agf`, 6.910) is significantly worse than non-tiered SOTA `T+Coif2V3_30s_bdeq` (6.735, p=0.045). Tiered T+DB3V3_30s_ag0 is a substantial regression (+5.94%). Weekly horizon (H=13) is the only period where uniform `bdeq` dominates the offset cascade.
- **Tiered offset belongs in an appendix ablation, not the main paper results.** It does not unlock a Tier-A new-result claim — the SMAPE gains are within paper-sample variance (typical config std 0.05–0.30 SMAPE) and are protocol-bound (Hourly cascade does not survive when `descend` is in scope; ascend≈descend on Hourly). It supports the multi-resolution DWT thesis qualitatively (different stacks specialise on different basis-dim scales) but the empirical evidence is too period-specific to lead with.

---

## 1. Status check

| File | Rows | Periods | Coverage notes |
|---|---|---|---|
| `tiered_offset_m4_allperiods_results.csv` | 800 (795 post-filter) | Y/Q/M/W/D — **no Hourly** | All 16 configs × 5 periods × 10 runs; full coverage |
| `tiered_offset_m4_allperiods_paperlr_results.csv` | 421 | Y/Q/M/W/D | Partial: Daily missing for `*_30s` 30-stack configs (3-5 runs); Yearly fewer Sym10 runs |
| `m4_hourly_sym10_tiered_offset_results.csv` | 80 | Hourly only | 8 configs × 10 runs, complete |

The standard-LR file is the basis for the analysis below. The paper-LR companion is treated as a sanity check (directionally consistent but underpowered).

## 2. Per-period analysis (tiered vs comparable non-tiered)

11 head-to-head pairs per period. All tests Mann–Whitney U (unpaired — seeds differ across files).

| Period | n_pairs tiered better | Avg delta SMAPE | Sig (p<0.05) tiered<base | Sig base<tiered |
|---|---|---|---|---|
| Yearly | 7/11 | **-0.020** | 1 (`TAE+DB3V3AE_30s_tiered`) | 0 |
| Quarterly | 8/11 | **-0.020** | 0 | 0 |
| Monthly | 7/11 | **-0.105** | 1 (`T+DB3V3_10s_tiered_ag0`) | 0 |
| Weekly | 2/11 | **+0.130** | 0 | 0 |
| Daily | **11/11** | **-0.049** | **4** | 0 |

**Significant tiered wins (p<0.05, MWU):**

| Period | Tiered config | Baseline | mean t | mean b | delta | p | Cliff d |
|---|---|---|---|---|---|---|---|
| Daily | `TAELG+Sym10V3AELG_10s_tiered` | `TALG+Sym10V3ALG_10s_ag0` | 3.013 | 3.093 | -0.080 | 0.0036 | 0.78 |
| Daily | `TAE+DB3V3AE_30s_tiered` | `TAE+DB3V3AE_30s_ld32_ag0` | 3.026 | 3.099 | -0.074 | 0.0036 | 0.78 |
| Daily | `TAELG+DB3V3AELG_10s_tiered` | `TALG+DB3V3ALG_10s_ag0` | 3.040 | 3.107 | -0.067 | 0.0312 | 0.58 |
| Daily | `TAELG+Sym10V3AELG_30s_tiered` | `TAELG+Sym10V3ALG_30s_ag0` | 3.032 | 3.075 | -0.043 | 0.0452 | 0.54 |
| Monthly | `T+DB3V3_10s_tiered_ag0` | `T+Db3V3_10s_bdeq` | 13.366 | 13.821 | -0.454 | 0.0113 | 0.68 |
| Yearly | `TAE+DB3V3AE_30s_tiered` | `TAE+DB3V3AE_30s_ld32_ag0` | 13.630 | 13.894 | -0.264 | 0.0376 | 0.56 |

No tiered config is significantly worse than its baseline at α=0.05.

**Per-family avg delta (negative = tiered helps):**

| Period | T+WaveletV3 (RB) | TAE+WaveletV3AE | TAELG+WaveletV3AELG |
|---|---|---|---|
| Yearly | −0.014 | **−0.060** | +0.023 |
| Quarterly | −0.027 | −0.012 | −0.019 |
| Monthly | **−0.174** | −0.031 | **−0.110** |
| Weekly | +0.138 | +0.120 | +0.133 |
| Daily | −0.032 | −0.055 | **−0.063** |

All three families behave similarly per period — there is **no AE/AELG vs RB selectivity** at this perimeter. Tiering is a horizon-/period-level effect, not a backbone-level effect.

## 3. Does the Hourly cascade hypothesis generalise?

**Hourly (re-confirmation, separate file).** Tiered T+Sym10V3 ascend SMAPE 8.947 vs non-tiered `T+Sym10V3_10s_bdeq` 9.099 — delta −0.152, p=0.014. The previously memorised Hourly cascade win **replicates**. However, **`descend` (8.922) is statistically tied with `ascend` (8.947, p=0.79)** — the gain comes from non-uniform basis-dim alone, not from the *direction* of the cascade. Frequency-band cascade hypothesis as originally stated is **weakened**: ordering does not matter on Hourly.

**Generalisation across the other 5 periods** (using the all-periods file):

| Period | Tiered direction | Effect | Generalises? |
|---|---|---|---|
| Yearly (H=6) | ascend only tested | mean Δ=−0.02 SMAPE, 1/11 sig | **partially** — small mean gain |
| Quarterly (H=8) | ascend only | mean Δ=−0.02, 0/11 sig | **weakly** |
| Monthly (H=18) | ascend only | mean Δ=−0.11, 1/11 sig | **moderate** — biggest single win (T+DB3V3_10s p=0.011) |
| Weekly (H=13) | ascend only | mean Δ=**+0.13**, 0/11 sig | **NO — reverses on Weekly** |
| Daily (H=14) | ascend only | mean Δ=−0.05, 4/11 sig | **YES — strongest replication** |
| Hourly (H=48) | ascend ≈ descend | Δ=−0.15 vs flat-bdeq | yes, but cascade direction does not matter |

**Selectivity (memory's "helps novel, doesn't help legacy") cannot be tested here** — the all-periods sweep only re-ran T+Wavelet / TAE+ / TAELG+ wavelet families. No NBEATS-G / NBEATS-IG / GenericAE was re-run with tiering. Selectivity remains an open empirical claim.

**Mechanism reading:** the cascade hypothesis (low-frequency basis at depth-1, increasingly high-frequency at depth-N) survives on Daily and Hourly (long-horizon, multi-frequency content) but **does not generalise to Weekly (H=13)**. Weekly's narrow effective frequency band makes a uniform `bdeq=H` regulariser locally optimal, and forcing per-stack basis-dim variation hurts. Yearly/Quarterly are noise-bound because absolute SMAPE differences are small relative to seed std.

## 4. Leaderboard impact

84-config paper-sample leaderboard per period (combined comprehensive + sym10_fills + tiered):

| Period | Best tiered config | Mean SMAPE | Rank | Top-10 tiered | Top-20 tiered |
|---|---|---|---|---|---|
| Yearly | `T+Sym10V3_10s_tiered_ag0` | 13.578 | **#3**/84 | 3 | 5 |
| Quarterly | `T+Sym10V3_30s_tiered_agf` | 10.356 | **#1**/84 | 4 | 7 |
| Monthly | `T+DB3V3_30s_tiered_agf` | 13.344 | **#1**/84 | 4 | 7 |
| Weekly | `T+DB3V3_30s_tiered_agf` | 6.910 | #4/84 | 2 | 3 |
| Daily | `T+Sym10V3_10s_tiered_ag0` | 3.012 | **#1**/84 | **9** | 14 |

**Daily top-10 (paper-sample, post-tiering):**

| Rank | Config | Source | Mean SMAPE | Std | n | Params |
|---|---|---|---|---|---|---|
| 1 | T+Sym10V3_10s_tiered_ag0 | tiered | 3.012 | 0.031 | 10 | 5.25M |
| 2 | TAELG+Sym10V3AELG_10s_tiered | tiered | 3.013 | 0.023 | 10 | 1.14M |
| 3 | T+DB3V3_10s_tiered_ag0 | tiered | 3.023 | 0.041 | 10 | 5.25M |
| 4 | TAE+DB3V3AE_30s_tiered | tiered | 3.026 | 0.041 | 10 | 3.41M |
| 5 | T+DB3V3_10s_tiered_agf | tiered | 3.030 | 0.077 | 9 | 5.25M |
| 6 | T+Sym10V3_30s_tiered_ag0 | tiered | 3.031 | 0.063 | 10 | 15.75M |
| 7 | TAELG+Sym10V3AELG_30s_tiered | tiered | 3.032 | 0.032 | 10 | 3.41M |
| 8 | TAE+DB3V3AE_10s_tiered | tiered | 3.036 | 0.053 | 10 | 1.14M |
| 9 | TAELG+Coif2V3ALG_30s_ag0 | baseline | 3.036 | 0.034 | 10 | 3.73M |
| 10 | TAE+Sym10V3AE_30s_tiered | tiered | 3.039 | 0.051 | 10 | 3.41M |

Tiered wavelet-trend configs sweep the Daily top-8.

**Significance vs prior per-period SOTA (CLAUDE.md leader):**

| Period | New leader | Prior leader | Delta | MWU p |
|---|---|---|---|---|
| Daily | T+Sym10V3_10s_tiered_ag0 (3.012) | TAELG+Coif2V3ALG_30s_ag0 (3.036) | −0.024 | 0.31 |
| Quarterly | T+Sym10V3_30s_tiered_agf (10.356) | NBEATS-IG_10s_ag0 (10.357) | −0.001 | 0.73 |
| Monthly | T+DB3V3_30s_tiered_agf (13.344) | TW_30s_td3_bdeq_haar (13.391) | −0.046 | 0.62 |
| Weekly | T+DB3V3_30s_tiered_agf (6.910) — **regresses** | T+Coif2V3_30s_bdeq (6.735) | +0.175 | **0.045** |
| Yearly | TALG+DB3V3ALG_10s_ag0 unchanged (13.550) | — | — | — |

The new Daily/Quarterly/Monthly bests are **not statistically significant against the prior single best** at α=0.05, but the **density of tiered configs at the top** (8 of top 10 on Daily, 4 of top 10 on Q/M) makes the population-level effect credible. Weekly is the one period where tiered is **significantly worse** than the non-tiered SOTA.

## 5. Open questions update

From `memory/project_tiered_offset_open_questions.md` (4 follow-ups):

| Open question | Status after this run |
|---|---|
| **TWAE ld32 re-run** | Still open. Not in this sweep — only T+/TAE+/TAELG+ alternating families were tiered, no unified TW/TWAE. Hourly file has TWAE_10s_td3_sym10_ld16 but at ld16 only. Re-run TWAE_10s_ld32_sym10 with tiered ascend on Hourly (and Daily, given the strong Daily generalisation). |
| **Tiered + agf** | **Partially answered.** agf tested on T+ family across 5 periods: helps Q/M/W slightly, hurts Y/D. Two T+ tiered_agf configs claim per-period SOTA (Q `T+Sym10V3_30s_tiered_agf`, M `T+DB3V3_30s_tiered_agf`). agf has run-to-run divergence in 10s configs (Daily Sym10 10s_agf had a single SMAPE=22 outlier; Weekly Sym10 10s_agf same). agf+tiered is **viable on 30s only**; avoid on 10s. |
| **Tiered 30s** | **Answered.** 30s tiered configs are present and competitive: Q/M/W per-period leaders are all 30s_agf; Daily SOTA is 10s_ag0 — depth × period interaction is real. 30s tiered sweep is no longer the highest-priority follow-up. |
| **Yearly falsification test** | **Answered, mixed.** Yearly tiered has 7/11 configs improving (avg Δ=−0.02) but only 1 reaches significance. Tiered Yearly leader (T+Sym10V3_10s_tiered_ag0, 13.578) does **not** beat prior SOTA TALG+DB3V3ALG_10s_ag0 (13.550). Yearly does not falsify the cascade hypothesis but does not strongly support it either — Yearly is a noise-bound period. |

**New highest-priority follow-up** (in order):

1. **Selectivity test on Daily.** Re-run `NBEATS-G_10s_ag0`, `NBEATS-IG_10s_ag0`, and a generic-AE config (`GAE_10s_*` or `GenericAE_*`) with tiered ascend on Daily. Hypothesis: if tiering helps generic blocks too, the cascade explanation collapses and we are just regularising. If selectivity holds (helps T+/TAE+, no-op on Generic/IG), the cascade story survives.
2. **Weekly ablation.** Single Weekly run with descend instead of ascend for T+DB3V3_30s. If descend matches non-tiered, the issue is offset magnitude, not direction; if descend recovers, Weekly preference for *uniform* basis-dim is confirmed.
3. **Tiered on TWAE_ld32_sym10** on Daily and Hourly (carries the prior TWAE ld32 question forward).
4. **Drop tiered on Weekly from any production / appendix recommendation.** It is significantly worse than the non-tiered T+Coif2V3 SOTA.

## 6. Place in NeurIPS paper narrative

The paper has a twin Tier-A thesis: **(A) multi-resolution DWT** + **(B) parameter efficiency**. Tiered offset relates to both but does not unlock a third headline.

- **Supports thesis A (multi-resolution DWT) qualitatively.** Per-stack basis-dim tiering is exactly the operational form of "different stacks resolve different frequency bands". The Daily result (8/10 leaderboard slots, Cliff d 0.54–0.78) is the strongest single-period demonstration in the paper-sample regime that letting basis dimensionality vary by depth does real work. Worth **one figure / one table** in an appendix.
- **Neutral on thesis B (parameter efficiency).** Tiered configs have the same parameter count as their bdeq baselines (within ~1% — the per-stack basis_dim variation marginally changes the projection layers). They do not win on params/SMAPE Pareto.
- **Risks the headline if put in main results.** (i) Weekly is significantly *worse* with tiering, breaking any "tiering is universally beneficial" claim; (ii) the new Daily SOTA is not significant against the prior TAELG+Coif2V3 leader (p=0.31); (iii) the Hourly cascade direction (ascend) ties descend, so the original frequency-band-cascade interpretation is incomplete.
- **Recommended placement:** Appendix ablation entitled "Per-stack basis-dim cascade as a regulariser". Lead bullet: *"Letting `basis_dim` vary by stack depth (`ascend` 6 → 18 across 10 stacks for H=14 Daily) recovers a 0.04 SMAPE improvement over uniform `bdeq` and dominates the Daily top-10. The effect does not generalise to Weekly (H=13), where uniform `bdeq` is significantly better."*. Cross-reference from main results only when explaining Daily numbers.

---

## Recommended next experiments

```yaml
# Highest priority: selectivity test for cascade hypothesis
experiment_name: tiered_selectivity_daily
dataset: m4
periods: [Daily]
training:
  sampling_style: nbeats_paper
  val_check_interval: 100
  min_delta: 0.001
  patience: 20
  max_epochs: 200
configs:
  - {builtin: NBEATS-G,  stacks: 10, tiered: ascend}     # legacy generic
  - {builtin: NBEATS-IG, stacks: 10, tiered: ascend}     # legacy interpretable
  - {builtin: GenericAE_10s, latent_dim: 32, tiered: ascend}
  - {builtin: T+DB3V3_10s,  active_g: false, tiered: ascend}  # positive control
n_runs: 10
```

If the legacy NBEATS-G / IG configs *also* improve under tiering on Daily, the cascade hypothesis is reduced to "non-uniform `basis_dim` regularises" and the Hourly memory note should be edited accordingly. If they show no effect while T+DB3V3 still improves, selectivity holds and the multi-resolution story is reinforced.

