# AutoEncoder Pure-Stack M4-Yearly Analysis

**Study:** Pure-stack AutoEncoder block family (no trend / wavelet / generic basis downstream) on M4 Yearly
**Config:** `experiments/configs/autoencoder_pure_m4.yaml`
**Results:** `experiments/results/m4/autoencoder_pure_m4_results.csv`
**Protocol:** `sampling_style: nbeats_paper` + ReduceLROnPlateau (matched to `comprehensive_m4_paper_sample_plateau`)
**Comparison set:** All M4-Yearly results from comprehensive paper-sample, paper-sample-plateau, sym10_fills (76 unique configs after dedup)

---

## Executive Summary

**The AutoEncoder pure-stack family is mid-pack on M4-Yearly and dominated at every parameter budget by TrendWavelet AE/AELG variants that already exist in the comprehensive sweep.** Best AE-pure config (AELG_10s_ld16_ag0, 1.92M params) ranks 17/76 with SMAPE 13.6446 — losing to the sub-1M champion `TWAE_10s_ld32_ag0` (0.48M, SMAPE 13.5457) by +0.099 SMAPE while using **4× the parameters**. There is no parameter-efficiency niche where an unmodulated AutoEncoder block is the right tool. **Recommendation: drop AutoEncoder pure-stack from M4 sweeps; do not extend to other periods.**

The result also produces a generalizable architectural lesson, formalized in `llm_inductive_bias_transfer_theory.md`: AE backbones are *parameter-efficient feature extractors with a narrow waist* — they only earn their keep when paired with a downstream structural prior (trend polynomial / wavelet basis) that absorbs the capacity reduction.

---

## 1. AutoEncoder pure configs (Yearly, after divergence filter)

One implicit-divergence run dropped: `AELG_10s_ld16_ag0` seed=46 (SMAPE=45.04, best_epoch=4). All other 59 runs valid. Consistent with the AE-LG bimodal-collapse pattern in MEMORY.

| config | block | active_g | latent_dim | n_params | n | SMAPE mean | SMAPE std | OWA mean |
|---|---|---|---|---:|---:|---:|---:|---:|
| AELG_10s_ld16_ag0 | AutoEncoderAELG | 0 | 16 | 1.92M | 9  | 13.6446 | 0.143 | 0.8162 |
| AE_10s_ld32_ag0   | AutoEncoder     | 0 | 32 | 8.31M | 10 | 13.6547 | 0.207 | 0.8159 |
| AAE_10s_ld32_ag1  | AutoEncoderAE   | 1 | 32 | 2.25M | 10 | 13.6590 | 0.128 | 0.8154 |
| AELG_10s_ld16_ag1 | AutoEncoderAELG | 1 | 16 | 1.92M | 10 | 13.6683 | 0.159 | 0.8174 |
| AAE_10s_ld32_ag0  | AutoEncoderAE   | 0 | 32 | 2.25M | 10 | 13.6699 | 0.180 | 0.8169 |
| AE_10s_ld32_ag1   | AutoEncoder     | 1 | 32 | 8.31M | 10 | 13.7016 | 0.164 | 0.8204 |

All six configs cluster in a tight 0.057-SMAPE band (13.645–13.702). Block choice (AE / AAE / AELG), `active_g` (0 / 1), and `latent_dim` (16 / 32) all wash out — the AE bottleneck itself is the binding constraint, not the variant.

AELG (ld=16) achieves equal-or-better SMAPE than AE (ld=32) at 23% the parameters — within-family parameter efficiency, but not enough to overcome the cross-family gap.

## 2. Combined Yearly leaderboard — protocol-matched (paper-sample family, N=76)

Sources merged: `autoencoder_pure_m4` + `comprehensive_m4_paper_sample` + `comprehensive_m4_paper_sample_plateau` + `comprehensive_m4_paper_sample_sym10_fills`. Deduped by config (preferring plateau > paper-sample > sym10 > AE-pure).

**Top 25 + AE-pure positions:**

| rank | config | source | n | SMAPE | OWA | params |
|---:|---|---|---:|---:|---:|---:|
| 1 | T+Coif2V3_30s_bdeq | plateau | 10 | 13.5417 | 0.8041 | 15.24M |
| 2 | TWAE_10s_ld32_ag0 | plateau | 10 | 13.5457 | 0.8052 | 0.48M |
| 3 | TWAE_10s_ld32_sym10_ag0 | plateau | 10 | 13.5528 | 0.8053 | 0.48M |
| 4 | T+Sym10V3_10s_bdeq | plateau | 9 | 13.5565 | 0.8071 | 5.08M |
| 5 | NBEATS-G_10s_ag0 | plateau | 10 | 13.5635 | 0.8092 | 8.22M |
| 6 | TAELG+Coif2V3ALG_10s_ag0 | plateau | 10 | 13.5636 | 0.8063 | 1.04M |
| 7 | TAELG+HaarV3ALG_30s_ag0 | plateau | 10 | 13.5833 | 0.8095 | 3.13M |
| 8 | TWAE_10s_ld32_sym10_agf | plateau | 10 | 13.5861 | 0.8092 | 0.48M |
| 9 | TWAELG_10s_ld32_db3_ag0 | plateau | 10 | 13.5883 | 0.8075 | 0.48M |
| 10 | TW_10s_td3_bdeq_sym10 | plateau | 10 | 13.6065 | 0.8089 | 2.08M |
| 11 | TAE+HaarV3AE_10s_ld32_ag0 | plateau | 10 | 13.6168 | 0.8109 | 1.11M |
| 12 | GAELG_10s_ld16_ag0 | plateau | 10 | 13.6183 | 0.8127 | 1.66M |
| 13 | TWGAELG_10s_ld16_sym10_ag0 | plateau | 10 | 13.6263 | 0.8121 | 0.45M |
| 14 | GenericAE_10s_sw0 | plateau | 10 | 13.6311 | 0.8150 | 1.75M |
| 15 | TWAELG_10s_ld32_sym10_ag0 | plateau | 10 | 13.6346 | 0.8130 | 0.48M |
| 16 | TW_10s_td3_bdeq_coif2 | plateau | 10 | 13.6371 | 0.8118 | 2.08M |
| **17** | **AELG_10s_ld16_ag0** | **AE_pure** | 9 | **13.6446** | **0.8162** | **1.92M** |
| 18 | GAE_10s_ld32_ag0 | plateau | 10 | 13.6459 | 0.8146 | 1.75M |
| 19 | TW_30s_td3_bdeq_coif3 | plateau | 10 | 13.6509 | 0.8143 | 6.23M |
| 20 | TWAELG_10s_ld32_sym10_agf | plateau | 10 | 13.6516 | 0.8156 | 0.48M |
| **23** | **AE_10s_ld32_ag0** | **AE_pure** | 10 | 13.6547 | 0.8159 | 8.31M |
| **27** | **AAE_10s_ld32_ag1** | **AE_pure** | 10 | 13.6590 | 0.8154 | 2.25M |
| **32** | **AELG_10s_ld16_ag1** | **AE_pure** | 10 | 13.6683 | 0.8174 | 1.92M |
| **33** | **AAE_10s_ld32_ag0** | **AE_pure** | 10 | 13.6699 | 0.8169 | 2.25M |
| **52** | **AE_10s_ld32_ag1** | **AE_pure** | 10 | 13.7016 | 0.8204 | 8.31M |

## 3. AE-pure ranking summary

| AE-pure config | rank/76 | gap vs #1 (T+Coif2V3) | gap vs sub-1M #1 (TWAE_10s_ld32_ag0) | gap vs paper-faithful #1 (same) |
|---|---:|---:|---:|---:|
| AELG_10s_ld16_ag0 | 17 | +0.103 (+0.76%) | +0.099 | +0.099 |
| AE_10s_ld32_ag0   | 23 | +0.113 (+0.83%) | +0.109 | +0.109 |
| AAE_10s_ld32_ag1  | 27 | +0.117 (+0.86%) | +0.113 | +0.113 |
| AELG_10s_ld16_ag1 | 32 | +0.127 (+0.94%) | +0.123 | +0.123 |
| AAE_10s_ld32_ag0  | 33 | +0.128 (+0.94%) | +0.124 | +0.124 |
| AE_10s_ld32_ag1   | 52 | +0.160 (+1.18%) | +0.156 | +0.156 |

PS-family overall #1: `T+Coif2V3_30s_bdeq` (15.2M, plateau) SMAPE=13.5417. Sub-1M and paper-faithful #1 (same config): `TWAE_10s_ld32_ag0` (0.48M, plateau) SMAPE=13.5457.

## 4. Competitiveness assessment

The AE-pure family is **dominated in every parameter regime** of interest on M4-Yearly:

- **Best AE-pure (AELG_10s_ld16_ag0, 1.92M)** lags `TWAE_10s_ld32_ag0` (0.48M) by +0.099 SMAPE while using **4× the parameters**. It also lags `TWAELG_10s_ld32_db3_ag0` (0.48M, rank 9) and `TWGAELG_10s_ld16_sym10_ag0` (0.45M, rank 13) — both ~4× smaller and better.
- **AE_10s_ld32 (8.31M)** sits at rank 23 despite consuming more params than `NBEATS-G_10s_ag0` (rank 5, 8.22M). Strict loss to a paper baseline at matched depth and matched params.
- **No parameter-efficiency win.** The sub-1M leaderboard is owned by TrendWavelet AE/AELG variants (ranks 2, 3, 8, 9, 13, 15, 20). The AE-pure family doesn't even break into sub-1M — its smallest config is AELG at 1.92M.
- **Tightness within family** (0.057 SMAPE spread across 6 configs) confirms the AE bottleneck is the ceiling, not the variant. Adding trend or wavelet inductive bias contributes ~0.1 SMAPE of headroom that pure-AE cannot recover by tuning.
- **One implicit-divergence run** in `AELG_10s_ld16_ag0` (10% rate, seed=46, SMAPE=45.04) — consistent with the GenericAELG / AE-LG bimodal collapse pattern in MEMORY.

## 5. Architectural interpretation (the generalizable finding)

`AERootBlock` and `AERootBlockLG` are **parameter-efficient feature extractors with a narrow waist**, not coefficient generators. The encoder→latent→decoder structure compresses information through `latent_dim` and re-expands it to `units` width. Its only architectural contribution vs `RootBlock` is *capacity reduction* through the bottleneck.

A capacity reduction earns its keep only when paired with a structural prior that absorbs the constraint:

- **TrendAE / TWAE / TWAELG (working):** AE backbone outputs `units`-wide features → `Linear(units → thetas_dim)` → polynomial Vandermonde and/or orthonormal wavelet basis. The *basis* downstream provides the inductive bias. The bottleneck becomes regularization on coefficients of a meaningful family.
- **AutoEncoder / AutoEncoderAE pure (failing):** AE backbone → branch encoder/decoder → `Linear(units → forecast_length)`. No structured basis anywhere. Stacked low-rank linear projections through narrow latents with no prior on the output. The bottleneck costs capacity and gains nothing in exchange.

This is the mechanism behind the empirical result and is the principle the LLM theory report extends.

## 6. Recommendation

**Drop AutoEncoder pure-stack from M4 sweeps.** Specifically:

- Do **not** extend the study to Quarterly / Monthly / Weekly / Daily / Hourly. The Yearly pattern strongly predicts the same dominance, and the structural argument is not period-dependent.
- Do **not** treat AE / AAE / AELG pure as a baseline to beat in future M4 work — the existing TrendWavelet AE/AELG variants already dominate at every budget tested.
- **Retain** AE backbones inside structured-basis blocks (`TrendAE`, `TWAE`, `TWAELG`, `TrendAELG+wavelet` alternating) — this is where they win.

If the AE-pure family is to be revisited, it would only be in a deliberate ablation study isolating the AE backbone's contribution from the structured basis (i.e., to *quantify* the gap rather than search for a Pareto-improving config).

---

**Files referenced:**
- `experiments/results/m4/autoencoder_pure_m4_results.csv`
- `experiments/configs/autoencoder_pure_m4.yaml`
- `experiments/results/m4/comprehensive_m4_paper_sample_plateau_results.csv`
- `experiments/results/m4/comprehensive_m4_paper_sample_results.csv`
- `experiments/results/m4/comprehensive_m4_paper_sample_sym10_fills_results.csv`
- `src/lightningnbeats/blocks/blocks.py:654` (AERootBlock)
- `src/lightningnbeats/blocks/blocks.py:867` (AutoEncoderAE)
- `src/lightningnbeats/blocks/blocks.py:1047` (TrendAE)
- Companion theoretical extension: `experiments/analysis/analysis_reports/llm_inductive_bias_transfer_theory.md`
