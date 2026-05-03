# Sampling Protocol Selection: paper-sample vs sliding

## Core Rule

| Goal | Recommended `sampling_style` |
|---|---|
| Best absolute SMAPE on M4 | `sliding` (wins Daily by −0.42, Hourly by −0.17, Quarterly by −0.19) |
| Paper-faithful comparison to Oreshkin et al. 2020 | `nbeats_paper` (paper-sample) |
| Long-horizon dataset (Traffic, Weather, M4 Daily/Hourly) | `sliding` |
| Short-horizon, small-series-count (M4 Yearly H=6, Weekly H=13, n_series≤400) | `nbeats_paper` |

**Critical:** absolute SMAPE numbers are NOT comparable across protocols. Pick one protocol per study; never mix `nbeats_paper` and `sliding` rows when ranking absolute SMAPE.

---

## Per-Period Protocol Verdicts (M4)

| Period | H | n_series | Best sliding SMAPE | Best paper-sample SMAPE | Δ (sliding − paper) | Winner |
|---|---|---|---|---|---|---|
| Yearly | 6 | 23,000 | 13.499 | 13.486 (n=5) / 13.542 (n=10) | ≈0 | tie |
| Quarterly | 8 | 24,000 | 10.127 | 10.313 | **−0.19** | **sliding** |
| Monthly | 18 | 48,000 | 13.279 | 13.240 | +0.04 | **paper-sample** |
| Weekly | 13 | 359 | 6.671 | 6.735 (or 6.559 if p3 validates) | −0.06 / +0.11 | sliding (or paper-sample with p3) |
| Daily | 14 | 4,227 | **2.588** | 3.012 | **−0.42** | **sliding** |
| Hourly | 48 | 414 | **8.587** | 8.758 | **−0.17** | **sliding** |

**Pattern:** sliding wins on long-horizon periods (Daily H=14, Hourly H=48) where per-window resampling of paper-sample throttles deep generic capacity. Paper-sample wins or ties on short horizons (Yearly H=6, Monthly H=18) where the bigger effective epoch of `nbeats_paper`'s per-series sampling helps stable convergence on small series counts.

**Surprise:** Quarterly (large series count, short H) — sliding is ahead by 0.19 SMAPE. Independent confirmation worth pursuing.

---

## Detailed Recommendations

### Use `sliding` when:

- Forecast horizon is ≥ 14 (M4 Daily, Hourly; Traffic-96/192/336/720; Weather-96/192/336/720).
- Optimizing for SOTA absolute SMAPE without constraint to paper-faithful methodology.
- Using cosine-warmup LR scheduler (sliding's strongest LR — see `lr-scheduler-selection`).
- Pure-Generic deep architectures (NBEATS-G_30s_ag0 hits 2.588 on Daily under sliding — paper-sample best is 3.012).

```yaml
training:
  sampling_style: sliding
  lr_scheduler:
    name: CosineAnnealing
    warmup_epochs: 15
    eta_min: 1.0e-6
  max_epochs: 200
```

### Use `nbeats_paper` when:

- Reproducing or extending Oreshkin et al. 2020 directly.
- Reporting head-to-head numbers against the published paper baselines (NBEATS-G/I/IG).
- Forecast horizon is short (H ≤ 8) AND series count is small (≤ ~5,000) — paper-sample's per-series sampling helps.
- Validating an architecture that depends on the paper-sampling input distribution (e.g., tiered-offset wavelet experiments — most of the 2026-04 → 2026-05 cascade evidence is paper-sample).

```yaml
training:
  sampling_style: nbeats_paper
  val_check_interval: 100      # required — avoid best_epoch=0 collapse
  min_delta: 0.001
  patience: 20
  max_epochs: 200
  lr_scheduler:
    name: ReduceLROnPlateau    # see lr-scheduler-selection skill
    patience: 3
    factor: 0.5
    cooldown: 1
```

---

## Pitfalls

1. **Never compare absolute SMAPE across protocols.** `13.240` (paper-sample plateau Monthly) and `13.279` (sliding Monthly) are not directly comparable. Re-rank within protocol.
2. **Paper-sample without sub-epoch validation collapses.** `val_check_interval=100`, `min_delta=0.001`, `patience=20` are mandatory. Without them, early-stopping fires on the first warmup-corrupted check.
3. **Sliding does not need sub-epoch validation.** Default Lightning behavior is fine.
4. **Mixing protocols in a leaderboard analysis is the most common mistake.** The 2026-05-03 meta-analysis explicitly partitions all 50 M4 CSVs by protocol before ranking — use the same approach.
5. **For Hourly, neither protocol has a tiered SOTA yet.** Best paper-sample is `NBEATS-IG_30s_agf` step_paper (8.758); best tiered Hourly run is `T+Sym10V3_10s_bdEQ_descend` plateau (8.922) — still +0.16 behind paper baseline. Treat Hourly as an open arena.

---

## Source

- Canonical: `experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md` §4.2
- Per-period leaderboards: same report §2.1–§2.6
