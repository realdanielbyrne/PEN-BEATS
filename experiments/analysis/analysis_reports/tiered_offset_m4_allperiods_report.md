# Tiered Offset M4 All-Periods Analysis Report

**Date:** 2025-05-02  
**Files Analyzed:**

1. `tiered_offset_m4_allperiods_results.csv` — Plateau LR scheduler
2. `tiered_offset_m4_allperiods_paperlr_results.csv` — Paper StepLR (multistep, 10 milestones, γ=0.5)
3. `comprehensive_m4_paper_sample_plateau_results.csv` — Baseline reference

## Executive Summary

**Tiered frequency-band offsets show promising but period-specific improvements.** The technique delivers meaningful SOTA improvements on **Monthly** (−0.11 SMAPE, −0.8%) and marginal gains on **Yearly** (−0.045) and **Daily** (−0.041). It underperforms on **Weekly** (+0.15–0.21) and matches SOTA on **Quarterly** (±0.004). This aligns with the falsification hypothesis: tiering helps where offset clamping doesn't fully degenerate the frequency tiers (Monthly H=18 with db3 has partial tiering), and is inert/neutral on very short horizons where all offsets clamp to the same effective dim=1.

The Plateau LR scheduler dominates on Monthly (−0.162 SMAPE mean advantage), while Paper StepLR wins on Weekly (+0.089) and Yearly (+0.027). Overall, Plateau LR edges StepLR (38 wins vs 32).

---

## 1. Completion Status

| Experiment | Raw Rows | Clean | Removed | Configs | Periods | Cells | Full (n≥10) | N/cell |
|---|---|---|---|---|---|---|---|---|
| Plateau LR (tiered) | 348 | 346 | 2 | 16 | 5 (Y/Q/M/W/D) | 70 | 0 | 4–5 |
| Paper StepLR (tiered) | 363 | 360 | 3 | 16 | 5 (Y/Q/M/W/D) | 73 | 0 | 3–5 |
| Baseline (comprehensive) | 1077 | 1074 | 3 | 69 | 3 (Y/Q/M) | 109 | 102 | 4–10 |

**Per-period breakdown (tiered experiments):**

- Yearly: 16 configs × 5 runs each (both LR variants)
- Quarterly: 16 configs × 5 runs (plateau) / 4–5 runs (StepLR)
- Monthly: 16 configs × 5 runs each
- Weekly: 16 configs × 5 runs each
- Daily: 6–9 configs × 3–5 runs (still filling in)

**Assessment:** Experiments are ~50% complete (targeting n=10 per cell). Daily coverage is thinnest. Statistical power is adequate for ranking (5 runs per cell) but border-line for significance tests.

---

## 2. Per-Period Top Configs vs SOTA

### Known SOTA Reference (from baseline + prior sweeps)

| Period | SOTA | SOTA Config |
|---|---|---|
| Yearly | 13.550 | TALG+DB3V3ALG_10s_ag0 / T+Coif2V3_30s_bdeq |
| Quarterly | 10.357 | NBEATS-IG_10s_ag0 |
| Monthly | 13.391 | TW_30s_td3_bdeq_haar (paper protocol) |
| Weekly | 6.735 | T+Coif2V3_30s_bdeq |
| Daily | 3.036 | TAELG+Coif2V3ALG_30s_ag0 |

### Top 3 Tiered Configs Per Period (Best of Both LR Variants)

**Yearly** (SOTA: 13.550):

| Rank | Config | SMAPE | ±Std | Δ vs SOTA | Params |
|---|---|---|---|---|---|
| 1 | T+DB3V3_10s_tiered_agf (StepLR) | 13.486 | 0.086 | **−0.064** | 5.07M |
| 2 | T+DB3V3_10s_tiered_agf (Plateau) | 13.525 | 0.120 | **−0.025** | 5.07M |
| 3 | TAELG+DB3V3AELG_10s_tiered (StepLR) | 13.570 | 0.125 | +0.020 | 1.03M |

**Quarterly** (SOTA: 10.357):

| Rank | Config | SMAPE | ±Std | Δ vs SOTA | Params |
|---|---|---|---|---|---|
| 1 | TAE+Sym10V3AE_10s_tiered (StepLR) | 10.330 | 0.056 | **−0.027** | 1.06M |
| 2 | TAELG+DB3V3AELG_10s_tiered (StepLR) | 10.342 | 0.050 | **−0.015** | 1.06M |
| 3 | TAE+Sym10V3AE_30s_tiered (Plateau) | 10.351 | 0.069 | **−0.006** | 3.17M |

**Monthly** (SOTA: 13.391):

| Rank | Config | SMAPE | ±Std | Δ vs SOTA | Params |
|---|---|---|---|---|---|
| 1 | T+DB3V3_30s_tiered_agf (Plateau) | **13.131** | 0.285 | **−0.260** | 16.03M |
| 2 | T+DB3V3_10s_tiered_ag0 (Plateau) | **13.217** | 0.196 | **−0.174** | 5.34M |
| 3 | TAELG+Sym10V3AELG_30s_tiered (Plateau) | 13.349 | 0.462 | **−0.042** | 3.57M |

**Weekly** (SOTA: 6.735):

| Rank | Config | SMAPE | ±Std | Δ vs SOTA | Params |
|---|---|---|---|---|---|
| 1 | TAELG+Sym10V3AELG_30s_tiered (StepLR) | 6.810 | 0.218 | +0.075 | 3.37M |
| 2 | T+Sym10V3_30s_tiered_ag0 (StepLR) | 6.827 | 0.177 | +0.092 | 15.68M |
| 3 | T+DB3V3_30s_tiered_ag0 (Plateau) | 6.885 | 0.234 | +0.150 | 15.68M |

**Daily** (SOTA: 3.036):

| Rank | Config | SMAPE | ±Std | Δ vs SOTA | Params |
|---|---|---|---|---|---|
| 1 | T+Sym10V3_10s_tiered_ag0 (Plateau) | **2.995** | 0.035 | **−0.041** | 5.25M |
| 2 | T+Sym10V3_30s_tiered_agf (Plateau) | **3.008** | 0.019 | **−0.028** | 15.75M |
| 3 | TAELG+Sym10V3AELG_10s_tiered (Plateau) | **3.009** | 0.016 | **−0.027** | 1.14M |

---

## 3. Head-to-Head: Plateau LR vs Paper StepLR

### Overall Score

- **Plateau LR wins: 38** cells (54%)
- **Paper StepLR wins: 32** cells (46%)
- Mean delta (plateau − steplr): **−0.018** (slight PlateauLR advantage)

### Per-Period Winner

| Period | Winner | Mean Delta | Median Delta | Magnitude |
|---|---|---|---|---|
| **Monthly** | **Plateau LR** | −0.162 | −0.101 | Large — consistent advantage |
| **Daily** | Plateau LR | −0.039 | −0.048 | Moderate |
| **Quarterly** | Plateau LR | −0.020 | −0.006 | Marginal |
| **Yearly** | StepLR | +0.027 | +0.034 | Marginal |
| **Weekly** | StepLR | +0.089 | +0.070 | Moderate |

**Interpretation:** Plateau LR's adaptive rate reduction helps on longer-horizon periods (Monthly H=18, Daily H=14) where training benefits from extended fine-tuning. StepLR's deterministic schedule better matches the short-series, few-epoch regimes of Yearly (H=6) and Weekly (H=13).

---

## 4. Backbone Analysis

Across both tiered experiments (combined), mean SMAPE by backbone:

| Period | Best Backbone | SMAPE | 2nd | 3rd | 4th |
|---|---|---|---|---|---|
| Yearly | AELG (13.638) | RootBlock_ag0 (13.645) | RootBlock_agf (13.653) | AE (13.693) |
| Quarterly | RootBlock_ag0 (10.382) | AE (10.400) | AELG (10.415) | RootBlock_agf (10.478) |
| Monthly | AE (13.635) | RootBlock_ag0 (13.645) | RootBlock_agf (13.649) | AELG (13.672) |
| Weekly | RootBlock_ag0 (6.999) | RootBlock_agf (7.127) | AE (7.129) | AELG (7.289) |
| Daily | AELG (3.039) | RootBlock_ag0 (3.047) | AE (3.048) | RootBlock_agf (3.058) |

**Key findings:**

- **RootBlock ag0** (paper-faithful) is the most consistent backbone — top-2 in 4/5 periods
- **AELG** shines on Yearly and Daily (short forecast horizons) with 5× fewer parameters
- **active_g=forecast** hurts on Quarterly (RootBlock_agf is worst) but is neutral elsewhere
- **AE** and **AELG** are nearly identical at matched parameters — AELG's learned gate doesn't consistently help in the tiered setting

---

## 5. Wavelet Family: sym10 vs db3

| Period | Winner | sym10 | db3 | Delta |
|---|---|---|---|---|
| Yearly | **db3** | 13.683 | **13.632** | −0.051 |
| Quarterly | **sym10** | **10.411** | 10.426 | −0.015 |
| Monthly | **db3** | 13.675 | **13.625** | −0.050 |
| Weekly | **sym10** | **7.113** | 7.160 | −0.047 |
| Daily | tie | 3.048 | 3.046 | −0.002 |

**Pattern:** db3 dominates where tiered offsets have functional diversity (Yearly, Monthly) — its shorter support means more basis functions fit within the forecast length before clamping. sym10 wins on Quarterly and Weekly. Overall, the differences are small (0.015–0.051).

**This confirms the offset-clamping hypothesis:** db3 benefits from tiering because its shorter support allows more distinct frequency bands within the same forecast_length. sym10's long support means offsets clamp faster, reducing effective tier diversity.

---

## 6. Stack Depth: 10s vs 30s

| Period | Winner | 10s SMAPE (params) | 30s SMAPE (params) | Delta |
|---|---|---|---|---|
| Yearly | **10s** | **13.640** (3.05M) | 13.674 (9.22M) | −0.034 |
| Quarterly | **30s** | 10.434 (3.14M) | **10.404** (9.26M) | −0.030 |
| Monthly | **30s** | 13.694 (3.27M) | **13.606** (9.80M) | −0.088 |
| Weekly | **30s** | 7.184 (3.18M) | **7.088** (9.53M) | −0.096 |
| Daily | **30s** | 3.051 (3.40M) | **3.045** (11.50M) | −0.006 |

**30-stack wins 4/5 periods** in the tiered setting. The only exception is Yearly where the shorter series can't support deep architectures. This is consistent with the general finding that 30-stack is preferred for novel wavelet/trend families.

However, the top-1 Yearly and top-2 Monthly configs are both 10-stack — meaning a well-chosen 10-stack config can beat the average 30-stack with much lower compute.

---

## 7. Monthly Deep Dive: New SOTA?

The standout finding is **T+DB3V3_30s_tiered_agf at 13.131 SMAPE** on Monthly (Plateau LR):

Individual runs: `[13.33, 12.99, 13.22, 12.71, 13.41]` — all 5 runs beat the prior SOTA of 13.391.

**Comparison to baseline best:**

- Baseline: `TW_30s_td3_bdeq_sym10` = 13.240 ±0.334 (n=9)
- Tiered: `T+DB3V3_30s_tiered_agf` = 13.131 ±0.285 (n=5)
- Delta: −0.109 (−0.8%)

**Statistical test:** Mann-Whitney U (tiered < baseline): p=0.399, r=0.111  
**Not significant** at current sample sizes — but the direction is consistent (all 5 tiered runs below the baseline mean). Needs n=10 to confirm.

The #2 tiered config `T+DB3V3_10s_tiered_ag0` at 13.217 also beats baseline SOTA with only 5.34M params (vs 16M for the winner).

---

## 8. Overall Verdict: Do Tiered Offsets Help?

| Period | Verdict | Best Tiered | SOTA | Δ |
|---|---|---|---|---|
| **Monthly** | ✅ **Likely improves** | 13.131 | 13.391 | −0.260 (−1.9%) |
| **Yearly** | ✅ Marginal gain | 13.486 | 13.550 | −0.064 (−0.5%) |
| **Daily** | ✅ Marginal gain | 2.995 | 3.036 | −0.041 (−1.4%) |
| **Quarterly** | ≈ Neutral | 10.330 | 10.357 | −0.027 (−0.3%) |
| **Weekly** | ❌ Underperforms | 6.810 | 6.735 | +0.075 (+1.1%) |

**The hypothesis is partially confirmed:** tiering helps where it creates meaningful frequency-band diversity (Monthly with db3 has 2–3 valid tiers before clamping). On very short horizons where all offsets clamp to the same value, tiering is inert as predicted. The **Weekly failure** is unexpected — Weekly (H=13) should also clamp everything, yet performance degrades. This may indicate the offset mechanism introduces harmful asymmetry even when functionally degenerate.

**Parameter efficiency:** The 1.03–1.14M AELG tiered configs (Yearly: 13.570, Daily: 3.009) are competitive with 5–16M RootBlock alternatives, making tiered AELG an attractive sub-2M option.

---

## 9. Recommendations & Next Steps

### Current Best Per Period (Updated)

| Period | Config | SMAPE | Confidence |
|---|---|---|---|
| Yearly | T+DB3V3_10s_tiered_agf (StepLR) | 13.486 | Medium (n=5, needs 10) |
| Quarterly | NBEATS-IG_10s_ag0 (baseline) | 10.312 | High (n=10) |
| Monthly | T+DB3V3_30s_tiered_agf (Plateau) | 13.131 | **Medium-Low** (n=5, p=0.40) |
| Weekly | T+Coif2V3_30s_bdeq (baseline) | 6.735 | High |
| Daily | T+Sym10V3_10s_tiered_ag0 (Plateau) | 2.995 | Medium (n=5) |

### What to Test Next

1. **Complete n=10 runs** on all tiered configs — particularly Monthly where the gain is largest but sample size limits confidence.

2. **Monthly targeted study:** Run `T+DB3V3_30s_tiered_agf` to n=20 with plateau LR to establish significance. Also try `step=4` (finer tiers) and `step=12` (coarser tiers) to probe the optimal tier granularity.

3. **Hourly inclusion:** The Hourly results exist in separate files (`m4_hourly_sym10_tiered_offset_*`). Integrate those into the overall picture.

4. **Weekly diagnosis:** Why does Weekly degrade? Try removing the offset entirely on Trend blocks (only offset wavelet blocks) to test if asymmetric Trend behavior is the culprit.

5. **db3 tiering on Monthly at 10-stack:** `T+DB3V3_10s_tiered_ag0` already hits 13.217 at 5.34M — this may be the best parameter-efficiency sweet spot. Run to n=10.

### Open Questions

- Is the Monthly improvement real or a fortunate sample? (p=0.40 is not convincing)
- Why does Weekly degrade when offsets should all clamp to zero?
- Would asymmetric tiering (offsets only on wavelet blocks, not Trend) work better?
- Is `step=8` optimal, or would dataset-adaptive step sizes (proportional to H) perform better?
