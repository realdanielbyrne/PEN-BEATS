# M4 Hourly · sym10 Tiered-Offset · Plateau vs Step-Paper LR

**Date:** 2026-05-04
**Scope:** focused comparison of two parallel sweeps on M4 Hourly (H=48, L=240).
**Source CSVs:**

- `experiments/results/m4/m4_hourly_sym10_tiered_offset_results.csv` — **plateau LR** (ReduceLROnPlateau, factor=0.75, patience=1, min_lr=1e-6)
- `experiments/results/m4/m4_hourly_sym10_tiered_offset_paperlr_results.csv` — **step_paper LR** (MultiStepLR, 10 evenly spaced milestones, γ=0.5)

Both files share: `sampling_style: nbeats_paper`, `max_epochs=150`, `val_check_interval=100`, `min_delta=0.001`, n=10 runs/cell, sym10 wavelet, 10-stack architectures, eq_fcast basis, ascend+descend tiering directions. Different seeds in the two files (160 unique seeds total) → matched-config comparisons are **unpaired**.

> **Naming caveat:** the file *without* `_paperlr` uses **plateau** LR; the `_paperlr` file uses **step_paper** (multistep) LR. The repo convention is to attach `_paperlr` to the multistep paper-style schedule (named after Oreshkin's paper schedule), not to a "paper-sample protocol" tag. Both files run paper-sample protocol — only the LR differs.

---

## 1. Headline

The **best Hourly tiered sym10 cell is `T+Sym10V3_10s_bdEQ_descend` under plateau LR at SMAPE 8.9224 ± 0.1132 (n=10)** — the same number already cited in the 2026-05-03 leaderboard as the generalist crown's Hourly cell (canonicalized to `T+Sym10V3_10s_tiered_ag0`, descend). Plateau LR ties or beats step_paper on 6/8 matched cells and is the right LR default for **trend-RootBlock** (`T+Sym10V3`) and **trend-AE** (`TAE+Sym10V3AE`) tiered configs on Hourly. Step_paper is decisively better only for the unified **TWAE** family (Δ −0.20 SMAPE, p ≈ 0.001–0.004).

**No tiered sym10 configuration beats the paper baseline.** The best tiered cell sits +0.16 SMAPE behind paper-sample `NBEATS-IG_30s_agf` step (8.758) and +0.34 behind sliding `NBEATS-IG_30s_agf` (8.587). The CLAUDE.md statement *"Hourly tiered does NOT beat the paper baseline"* is **confirmed and unchanged**.

---

## 2. CSV inventory and filtering

| File | LR | rows | configs | runs/cfg | divergence | strict-filter drops |
|---|---|---:|---:|---:|---:|---:|
| `m4_hourly_sym10_tiered_offset_results.csv` | plateau | 80 | 8 | 10 | 0 | 0 |
| `m4_hourly_sym10_tiered_offset_paperlr_results.csv` | step_paper | 80 | 8 | 10 | 0 | 0 |

Strict filter (`diverged | smape>100 | (best_epoch==0 AND smape>50)`) drops **0 / 160 rows**. Both sweeps are clean.

**8 configs (identical across both files):**

| Backbone family | Configs |
|---|---|
| Trend-RootBlock + sym10 alternating | `T+Sym10V3_10s_bdEQ_{ascend,descend}` |
| TrendAE + sym10AE alternating | `TAE+Sym10V3AE_10s_ld32_bdEQ_{ascend,descend}` |
| Unified TrendWavelet (RootBlock) | `TW_10s_td3_sym10_bdEQ_{ascend,descend}` |
| Unified TrendWaveletAE | `TWAE_10s_td3_sym10_ld16_bdEQ_{ascend,descend}` |

All configs are **tiered** (every row has a non-trivial `stack_basis_offsets`); there is no non-tiered baseline included in these two files. Non-tiered baselines for the same backbones live in `comprehensive_m4_paper_sample{,_sym10_fills}_results.csv` and are referenced where needed below.

---

## 3. Per-cell results (LR × config), sorted by mean SMAPE

| # | LR | Config | SMAPE ± std | Min | Max | Params |
|---:|---|---|---|---:|---:|---:|
| 1 | plateau | T+Sym10V3_10s_bdEQ_descend | **8.9224 ± 0.1132** | 8.770 | 9.082 | 6.06M |
| 2 | plateau | T+Sym10V3_10s_bdEQ_ascend | 8.9469 ± 0.0866 | 8.811 | 9.115 | 6.06M |
| 3 | step_paper | T+Sym10V3_10s_bdEQ_ascend | 8.9885 ± 0.0967 | 8.853 | 9.108 | 6.06M |
| 4 | plateau | TW_10s_td3_sym10_bdEQ_ascend | 9.0235 ± 0.1183 | 8.851 | 9.189 | 2.79M |
| 5 | step_paper | TW_10s_td3_sym10_bdEQ_ascend | 9.0377 ± 0.0640 | 8.945 | 9.138 | 2.79M |
| 6 | plateau | TW_10s_td3_sym10_bdEQ_descend | 9.0415 ± 0.0802 | 8.919 | 9.174 | 2.79M |
| 7 | step_paper | TW_10s_td3_sym10_bdEQ_descend | 9.0478 ± 0.0927 | 8.926 | 9.231 | 2.79M |
| 8 | plateau | TAE+Sym10V3AE_10s_ld32_bdEQ_descend | 9.0576 ± 0.1367 | 8.746 | 9.242 | 1.62M |
| 9 | plateau | TAE+Sym10V3AE_10s_ld32_bdEQ_ascend | 9.0791 ± 0.1166 | 8.873 | 9.227 | 1.62M |
| 10 | step_paper | T+Sym10V3_10s_bdEQ_descend | 9.0899 ± 0.1387 | 8.898 | 9.298 | 6.06M |
| 11 | step_paper | TAE+Sym10V3AE_10s_ld32_bdEQ_descend | 9.1491 ± 0.1315 | 8.968 | 9.354 | 1.62M |
| 12 | step_paper | TAE+Sym10V3AE_10s_ld32_bdEQ_ascend | 9.1598 ± 0.1457 | 8.937 | 9.384 | 1.62M |
| 13 | step_paper | TWAE_10s_td3_sym10_ld16_bdEQ_descend | 9.2431 ± 0.1406 | 9.044 | 9.442 | 0.88M |
| 14 | step_paper | TWAE_10s_td3_sym10_ld16_bdEQ_ascend | 9.2469 ± 0.0826 | 9.124 | 9.355 | 0.88M |
| 15 | plateau | TWAE_10s_td3_sym10_ld16_bdEQ_descend | 9.4394 ± 0.0680 | 9.347 | 9.556 | 0.88M |
| 16 | plateau | TWAE_10s_td3_sym10_ld16_bdEQ_ascend | 9.4552 ± 0.1158 | 9.267 | 9.614 | 0.88M |

**Best single run:** plateau, `TAE+Sym10V3AE_10s_ld32_bdEQ_descend`, seed 950220921, **SMAPE 8.7460** (the only sub-8.80 run in either file).

---

## 4. Plateau vs Step-Paper LR — matched-config comparison

Mann-Whitney U, two-sided, n=10 vs n=10 (different seeds → unpaired).

| Config | Plateau (n=10) | Step_paper (n=10) | Δ (plat−step) | p | Verdict |
|---|---|---|---:|---:|---|
| T+Sym10V3_10s_bdEQ_ascend | 8.947 ± 0.087 | 8.989 ± 0.097 | −0.042 | 0.521 | tie |
| T+Sym10V3_10s_bdEQ_descend | **8.922 ± 0.113** | 9.090 ± 0.139 | **−0.168** | **0.026** | **plateau wins** |
| TAE+Sym10V3AE_10s_ld32_bdEQ_ascend | 9.079 ± 0.117 | 9.160 ± 0.146 | −0.081 | 0.308 | tie (directional plateau) |
| TAE+Sym10V3AE_10s_ld32_bdEQ_descend | 9.058 ± 0.137 | 9.149 ± 0.131 | −0.091 | 0.241 | tie (directional plateau) |
| TW_10s_td3_sym10_bdEQ_ascend | 9.023 ± 0.118 | 9.038 ± 0.064 | −0.014 | 0.791 | tie |
| TW_10s_td3_sym10_bdEQ_descend | 9.041 ± 0.080 | 9.048 ± 0.093 | −0.006 | 1.000 | tie |
| TWAE_10s_td3_sym10_ld16_bdEQ_ascend | 9.455 ± 0.116 | 9.247 ± 0.083 | **+0.208** | **0.001** | **step_paper wins** |
| TWAE_10s_td3_sym10_ld16_bdEQ_descend | 9.439 ± 0.068 | 9.243 ± 0.141 | **+0.196** | **0.004** | **step_paper wins** |

**Pooled (across all 80 vs 80 runs):** plateau 9.121 ± 0.221 vs step_paper 9.120 ± 0.142. Mann-Whitney p = 0.483. **No global LR winner**; the comparison is backbone-dependent.

**Backbone summary:**

- **`T+Sym10V3` (alternating Trend-RB + sym10WaveletV3):** plateau wins (significant on descend, directional on ascend). Plateau improves the descend direction by 0.17 SMAPE — large enough to flip the leaderboard ordering.
- **`TAE+Sym10V3AE` (alternating TrendAE + sym10WaveletAE):** plateau directionally better but not significant at α=0.05; effect size −0.08 to −0.09 SMAPE on both directions. Treat as **plateau-preferred but tied**.
- **`TW_td3_sym10` (unified TrendWavelet RootBlock):** statistical dead heat (Δ ≤ 0.014 SMAPE on both directions). The unified RB block is **LR-insensitive on Hourly**.
- **`TWAE_td3_sym10_ld16` (unified TrendWaveletAE):** step_paper decisively better, Δ +0.20 SMAPE, p ≈ 0.001–0.004. Plateau LR appears to **harm the unified-AE block on Hourly** — the only family where step_paper should be preferred.

The TWAE asymmetry is the most surprising finding. Hypothesis: the `factor=0.75, patience=1` plateau schedule cuts LR aggressively (10 cuts in 150 epochs ≈ 0.75^10 = 5.6% retained vs step_paper's 0.5^10 = 0.1% retained but uniformly distributed). The TWAE block's deeper bottleneck (latent_dim=16 vs ld=32 for TAE+Sym10V3AE) may need the tail-end small-LR regime that step_paper guarantees.

---

## 5. Tiered direction (ascend vs descend) effect

Mann-Whitney U, two-sided, n=10 vs n=10 within each (LR, backbone) cell.

| LR | Backbone | ascend | descend | Δ (asc−desc) | p |
|---|---|---:|---:|---:|---:|
| plateau | T+Sym10V3 | 8.947 | 8.922 | +0.025 | 0.791 |
| plateau | TAE+Sym10V3AE | 9.079 | 9.058 | +0.022 | 0.850 |
| plateau | TW_td3_sym10 | 9.023 | 9.041 | −0.018 | 0.623 |
| plateau | TWAE_td3_sym10_ld16 | 9.455 | 9.439 | +0.016 | 0.521 |
| step_paper | T+Sym10V3 | 8.989 | 9.090 | **−0.101** | 0.104 |
| step_paper | TAE+Sym10V3AE | 9.160 | 9.149 | +0.011 | 1.000 |
| step_paper | TW_td3_sym10 | 9.038 | 9.048 | −0.010 | 0.910 |
| step_paper | TWAE_td3_sym10_ld16 | 9.247 | 9.243 | +0.004 | 0.910 |

**Pooled across 4 backbones within each LR:** plateau ascend 9.126 vs descend 9.115 (Δ +0.011, p=0.96); step_paper ascend 9.108 vs descend 9.133 (Δ −0.024, p=0.46).

Direction effect is **statistically null (p > 0.10 everywhere)**. The single near-significant cell — `T+Sym10V3` step_paper, ascend 0.10 better than descend (p=0.10) — is the *opposite sign* of the direction win in the matched plateau cell, so direction × LR shows no consistent ranking.

**Operational takeaway:** on Hourly tiered sym10, ascend vs descend is essentially a coin flip. The 0.025 SMAPE plateau-descend lead that the leaderboard's generalist crown picks up over plateau-ascend is real but well within seed noise; either direction would have qualified for the crown.

---

## 6. Best tiered cell vs Hourly SOTA

| Reference | SMAPE | Δ (best tiered − ref) |
|---|---:|---:|
| Best tiered sym10: `T+Sym10V3_10s_bdEQ_descend` plateau | **8.9224 ± 0.1132** | — |
| Paper-sample SOTA: `NBEATS-IG_30s_agf` step | 8.758 ± 0.099 | **+0.164** |
| Paper-sample 2nd: `NBEATS-G_30s_agf` step | 8.862 | +0.060 |
| Paper-sample 4th: `NBEATS-IG_30s_ag0` step (paper-faithful) | 8.906 | +0.016 |
| Paper-sample 6th: `TWAELG_10s_ld32_db3_agf` step (sub-1M) | 8.924 | −0.002 (tied) |
| Sliding SOTA: `NBEATS-IG_30s_agf` cosine | 8.587 ± 0.080 | +0.335 |

**Verdict:** the best paper-sample tiered Hourly cell is statistically **tied with the sub-1M `TWAELG_10s_ld32_db3_agf` step_paper baseline** (Δ = −0.002), and remains **+0.16 SMAPE behind the paper-faithful `NBEATS-IG_30s_agf` step** baseline. The CLAUDE.md guidance — **"Hourly tiered does NOT beat the paper baseline"** — is empirically confirmed at n=10 for both LR schedulers on all four sym10 backbones.

---

## 7. Implications for the M4 generalist crown

The 2026-05-03 leaderboard already uses the plateau-descend cell (8.9224) as the Hourly contribution to `T+Sym10V3_10s_tiered_ag0`'s 6/6-period mean rank of 13.33. The new `_paperlr` file does **not** change that:

- Plateau-descend (8.922) remains the strongest cell for this config and is the one currently feeding the rank-5 Hourly slot.
- Step_paper-ascend (8.989) and step_paper-descend (9.090) are both worse than the plateau-descend cell already in use; switching to step_paper would *raise* the Hourly rank from 5 to ~7-8 and drag the mean rank from 13.33 to roughly 13.7-13.8.
- Therefore: **no crown change**, but the leaderboard's "best LR per period" annotation should record that on Hourly, plateau LR is the chosen scheduler for the `T+Sym10V3` family (descent statistically supported), while step_paper is the chosen scheduler for the `TWAE` family.

**The other Hourly-only canonical configs** (`TW_10s_td3_sym10_tiered`, `TWAE_10s_td3_sym10_ld16_tiered`) inherit similar rank impact:

- `TW_10s_td3_sym10_tiered` best cell stays at **9.024 (plateau, ascend)**; step_paper is +0.014 behind, statistically tied. Hourly rank ≈ 18, unchanged.
- `TWAE_10s_td3_sym10_ld16_tiered` best cell **shifts** from plateau (9.439, prior table) to **step_paper (9.243, ascend)** under the new file. This drops its Hourly rank from 60 to roughly 27-30 (step_paper TWAE-ascend at 9.247 sits between the rank-25 step `TAELG+Coif2V3ALG_30s_ag0` 9.166 and rank-30 `T+HaarV3_10s_bdeq_agf` 9.27 in the all-Hourly leaderboard) — but TWAE_td3_sym10_ld16_tiered is Hourly-only, so this only affects its 1/6 mean rank, not any cross-period crown.

---

## 8. Recommendations

1. **Leaderboard update:** record plateau LR as the chosen Hourly scheduler for trend-RootBlock + sym10 alternating tiered (`T+Sym10V3_10s_tiered_ag0`) and for `TAE+Sym10V3AE_10s_tiered`. Record step_paper as the chosen Hourly LR for unified-AE TWAE tiered (`TWAE_10s_td3_sym10_ld16_tiered`). The Hourly-only `TW_10s_td3_sym10_tiered` is LR-insensitive — pick plateau for consistency with the alternating tiered family.
2. **Update §2.6 of the leaderboard** to reflect the corrected `TWAE_10s_td3_sym10_ld16_tiered` best cell (9.243 step_paper, was 9.439 plateau in the pre-`_paperlr` cut).
3. **CLAUDE.md no change required.** The "Hourly tiered does NOT beat the paper baseline" line stands. The "Weekly is locked on step_paper" / "plateau wins Q/M/D" structure is reinforced — Hourly belongs to the *tied or LR-mixed* category.
4. **No crown change.** `T+Sym10V3_10s_tiered_ag0` mean rank 13.33 stays.
5. **Drop list addition:** none. The TWAE step_paper cell at 9.243 still trails the paper baseline by +0.49 SMAPE; not competitive for a top-N Hourly slot, but useful as the sub-1M tiered champion (0.88M params, +0.49 vs paper baseline at ~50× fewer params than NBEATS-IG_30s_agf).

---

## 9. Open questions

- **Why does plateau hurt TWAE_td3 but help T+Sym10V3?** Both are 10-stack Hourly architectures. The unified-AE block must be sensitive to the late-training small-LR regime that step_paper provides uniformly but plateau provides only conditionally on val-loss stalls. A per-step LR-trace overlay would clarify; not a blocking question.
- **Would cosine-warmup (which sliding uses) match plateau on `T+Sym10V3` Hourly?** The leaderboard's §6.5 already proposes a paper-sample × cosine-warmup experiment to settle this. Pending.
- **Is the descend > ascend lead on `T+Sym10V3` plateau real?** Δ −0.025 SMAPE, p=0.79 is well inside seed noise. Not actionable. Pick descend by current convention; either direction would yield the crown.

---

## Appendix · Reproduction notes

```bash
# Plateau LR (file: m4_hourly_sym10_tiered_offset_results.csv)
python experiments/run_from_yaml.py \
    experiments/configs/m4_hourly_sym10_tiered_offset.yaml

# Step_paper LR (file: m4_hourly_sym10_tiered_offset_paperlr_results.csv)
python experiments/run_from_yaml.py \
    experiments/configs/m4_hourly_sym10_tiered_offset_paperlr.yaml
```

Both YAMLs use `sampling_style: nbeats_paper`, `max_epochs=150`, `val_check_interval=100`, `min_delta=0.001`. Plateau YAML: `factor=0.75, patience=1, min_lr=1e-6`. Step_paper YAML: `type=multistep, n_milestones=10, gamma=0.5`.
