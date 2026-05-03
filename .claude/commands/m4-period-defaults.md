# M4 Per-Period Default Configurations

Quick lookup of the best-known config per (period × protocol × parameter budget) on M4 as of 2026-05-03.

## Headline Configs

### Generalists (use when running across multiple periods)

| Protocol | Best generalist | Mean rank | Periods covered | Params |
|---|---|---|---|---|
| paper-sample (all 6) | `NBEATS-IG_30s_ag0` | 16.0 / 68 | 6/6 | 38.13M |
| paper-sample (5/6, no Hourly) | `T+Sym10V3_10s_tiered_ag0` | 14.8 / 108 | 5/6 | 5.25M |
| sliding | `T+HaarV3_30s_bd2eq` | 12.7 / 112 | 6/6 | 16.25M |
| sliding (sub-5M) | `TALG+HaarV3ALG_30s_ag0` | 21.0 / 112 | 6/6 | 3.66M |

### Per-period bests

| Period | Paper-sample best (LR) | SMAPE | Sliding best | SMAPE |
|---|---|---|---|---|
| Yearly | `T+Coif2V3_30s_bdeq` plateau | 13.542 | `TW_10s_td3_bdeq_coif2` | 13.499 |
| Quarterly | `NBEATS-IG_10s_ag0` plateau | **10.313** | `NBEATS-IG_10s_ag0` | 10.127 |
| Monthly | `TW_30s_td3_bdeq_sym10` plateau | **13.240** | `TW_30s_td3_bd2eq_coif2` | 13.279 |
| Weekly | `T+Coif2V3_30s_bdeq` step_paper | 6.735¹ | `T+Db3V3_30s_bdeq` | 6.671 |
| Daily | `T+Sym10V3_10s_tiered_ag0` plateau | **3.012** | `NBEATS-G_30s_ag0` | 2.588 |
| Hourly | `NBEATS-IG_30s_agf` step_paper | 8.758 | `NBEATS-IG_30s_agf` | 8.587 |

¹ Single-seed plateau tuning hit 6.559 — n=10 confirmation pending.

### Sub-1M parameter champions (paper-sample)

| Period | Sub-1M config | SMAPE | Δ vs leader |
|---|---|---|---|
| Yearly | `TWAE_10s_ld32_ag0` plateau (0.48M) | 13.546 | +0.004 |
| Quarterly | `TWAE_10s_ld32_ag0` step_paper (0.49M) | 10.404 | +0.091 |
| Monthly | `TWAE_10s_ld32_sym10_ag0` (0.58M) | 13.513 | +0.273 |
| Weekly | `TWAELG_10s_ld32_db3_ag0` (0.54M) | 7.252 | +0.517 (Weekly hates small models) |
| Daily | `TWGAELG_10s_ld16_db3_ag0` (0.52M) | 3.051 | +0.039 (best param efficiency on M4) |
| Hourly | `TWAELG_10s_ld32_db3_agf` (0.85M) | 8.924 | +0.166 |

---

## Architecture-by-frequency-band

| Band | Periods | Best non-tiered (paper-sample) | Best tiered (paper-sample) | Best (sliding) |
|---|---|---|---|---|
| Low-frequency | Yearly, Quarterly | `NBEATS-IG_10s_ag0` plateau | T+Sym10V3 / T+DB3V3 tiered (marginal) | T+Sym10V3_30s_bd2eq, TW_10s_td3_bdeq_coif2 |
| Mid-frequency | Monthly, Weekly | `TW_30s_td3_bdeq_sym10` plateau (Mo); `T+Coif2V3_30s_bdeq` (Wk) | regresses Weekly @ default plateau cell | TW_30s_td3_bd2eq_coif2; T+Db3V3_30s_bdeq |
| High-frequency | Daily, Hourly | TAELG+Coif2V3ALG_30s_ag0 (Dy); NBEATS-IG_30s_agf (Hr) | **T+Sym10V3_10s_tiered_ag0** (Dy 3.012 SOTA); H tied | NBEATS-G_30s_ag0; NBEATS-IG_30s_agf |

**High-frequency rule:** tiered-offset sym10/db3 dominates Daily under paper-sample (8/10 top-10 are tiered). Use tiered for Daily.

**Mid-frequency rule:** alternating `T+<wav>V3_30s_bdeq` (RootBlock) wins Weekly. On Monthly, plateau LR has now flipped the leader to unified `TW_30s_td3_bdeq_sym10` (13.240 < alternating 13.279).

**Low-frequency rule:** `NBEATS-IG_10s_ag0` plateau is the safest. Top-10 spreads are 0.05–0.07 SMAPE — within seed std. Tiered helps Yearly marginally (within noise).

---

## Drop list (do not use on M4)

1. All `BNG*` / `BNAE*` / `BNAELG*` (BottleneckGeneric family — universal worst).
2. All pure VAE configs (`GenericVAE_*_sw0`, etc.) — SMAPE 55–68 on Q/M/W/D, unusable.
3. `NBEATS-G_30s_ag0` on Quarterly / Weekly / Monthly — bimodal collapse, std 7.4–9.5.
4. All `*_sd5` (skip_distance=5) variants — never help on M4.
5. All `*_coif3` — no per-period SOTA.
6. `_30s_agf` tiered configs at `_10s` depth — run-to-run divergence outliers.
7. step_paper LR for Q/M/D — use plateau instead.
8. Pure `GenericAE_*` / `GenericAELG_*` — bottoms out at mean rank ~38.

---

## Companion skills

- `lr-scheduler-selection` — plateau vs step_paper vs cosine-warmup
- `sampling-protocol-selection` — paper-sample vs sliding by horizon
- `wavelet-family-selection` — haar/db3/coif2/sym10 within wavelet stacks
- `trendwavelet-block-selection` — unified TW vs alternating T+<wav>V3
- `ae-latent-dim-selection` — ld=8 vs 16 vs 32

---

## Source

- Canonical: `experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md`
- Reproducibility: `experiments/analysis/scripts/m4_overall_leaderboard.py`
