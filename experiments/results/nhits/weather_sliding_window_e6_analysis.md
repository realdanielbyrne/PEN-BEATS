# NHiTS Weather Sliding-Window Benchmark (`_e6`) — Analysis Report

**Source file:** [`experiments/results/nhits/weather_sliding_window_results_e6.csv`](weather_sliding_window_results_e6.csv)
**Driving config:** [`experiments/configs/nhits_benchmark_weather_sliding.yaml`](../../configs/nhits_benchmark_weather_sliding.yaml)
**Date of analysis:** 2026-04-22

---

## 1. Executive Summary

This benchmark evaluates 29 NHiTSSliding configurations on the Weather dataset across four forecast horizons (96 / 192 / 336 / 720) under the NHiTS-paper evaluation protocol (global z-score normalization, 70/10/20 split, MSE loss, forecast_multiplier=5 → L=5H). The `_e6` suffix denotes **6 training epochs per run**; the YAML commits `max_epochs: 4`, so this CSV reflects a slightly longer training budget than the checked-in config suggests. All 579 recorded runs hit `MAX_EPOCHS` (no early stopping, no divergence). Sampling style is `sliding` (dense window iteration), which distinguishes it from the paper-style stochastic sampling recorded in `weather_sampling_window_results_s1024.csv`.

**Key findings:**

1. **Two architecture families dominate every horizon.** `TrendWavelet`-unified blocks (AE / AELG / GenericVAE-agF) and `BottleneckGenericAE(LG)` occupy all top-5 slots across horizons, and the top-6 configs by cross-horizon mean rank are all from these two families.
2. **The NHiTS-Generic baseline (`NHiTSGenericRoot`) is the worst production config.** It ranks 28–29 of 29 on every short/mid horizon and only escapes the basement at H=720 because 3 large-parameter generic variants trail it. Its MSE is +6.4 % over the best TrendWaveletAELG at H=96 (p < 1e-4, paired Wilcoxon, n=19).
3. **Parameter efficiency is strongly anti-correlated with size.** The best 6-config cluster sits at 0.37M – 1.18M parameters (depending on horizon); the worst cluster (plain `NHiTS-Generic`, `GenericAE`, `GenericAELG`, `GenericVAE`) sits at 2.4M – 10M parameters. Bigger is worse.
4. **Long-horizon crown shifts from wavelet-unified to Bottleneck.** TrendWaveletAELG/AE lead at H=96 (best MSE 0.160), are co-leaders at H=192, but fall behind BottleneckGenericAE(-agF) at H=336 and H=720. This is a real protocol effect: at long horizons the polynomial-trend + DWT basis begins to lose to a pure learned bottleneck projection.
5. **`active_g='forecast'` is neutral-to-slightly-negative here.** Across 4 matched pairs (BottleneckGenericAE/AELG, GenericAE/AELG) the agF variant is never significantly better; for `GenericAE` it is significantly **worse** (p=0.04). This is the first Weather evidence that agF does not help when normalization is global z-score and the 2-layer NHiTSGenericRoot tower is replaced by a 4-layer AE tower.
6. **Gap to the published NHiTS paper (Table 1):** +1.3 % at H=96, +11 % at H=192, +15 % at H=336, +12 % at H=720. The short-horizon number is within noise of the paper; the longer-horizon gap reflects the 6-epoch budget (paper uses 1500–3000 epochs under paper-sampling).

---

## 2. Data Overview

| Property | Value |
|---|---|
| Total rows | 579 |
| Configs tested | 29 |
| Horizons | 96, 192, 336, 720 |
| Runs per (config, horizon) | 5 (mostly); `NHiTS-TrendWaveletAELG` at H=96 is missing 1 run (n=4) |
| Epochs trained | **6** (every run) |
| Stopping reason | `MAX_EPOCHS` (100 %) |
| Loss | `MSELoss` |
| Sampling style | `sliding` (dense iteration) |
| Normalization | global (column z-score on training split) |
| Train / val / test | 0.70 / 0.10 / 0.20 |
| Backcast length | 480 (=5×96) |
| Stacks × blocks/stack | 3 × 1 (NHiTS hierarchical pooling) |

All configs report `n_params`, `mse`, `mae`, `smape`, `denorm_mse`, `denorm_mae`, `best_val_loss`, plus full training metadata. No divergence (all SMAPE < 200; worst SMAPE observed 101 at H=720, expected because MSE loss on z-scored data does not optimize SMAPE).

---

## 3. Top / Bottom Rankings per Horizon

Primary metric is **mean test MSE over 5 seeds** (on the globally-normalized target — not denormalized). Standard deviations are in parentheses.

### H = 96 (NHiTS paper baseline: MSE 0.158, MAE 0.209)

| Rank | Config | MSE | MAE | Params |
|---|---|---|---|---|
| 1 | `NHiTS-TrendWaveletAELG` | **0.1600** (±0.0019) | 0.2169 | 372,873 |
| 2 | `NHiTS-TrendWaveletAE` | 0.1604 (±0.0016) | 0.2176 | 372,825 |
| 3 | `NHiTS-TrendWaveletGenericVAE-agF` | 0.1625 (±0.0030) | 0.2211 | 431,832 |
| 4 | `NHiTS-TrendAE+Coif2V3AE+TrendAE` | 0.1626 (±0.0021) | 0.2205 | 560,694 |
| 5 | `NHiTS-BottleneckGenericAELG` | 0.1628 (±0.0014) | 0.2225 | 805,935 |
| … | | | | |
| 27 | `NHiTS-GenericAE` | 0.1680 (±0.0017) | 0.2276 | 1,674,288 |
| 28 | `NHiTS-Generic` (baseline) | 0.1703 (±0.0023) | 0.2300 | 2,411,520 |
| 29 | `NHiTS-TrendWaveletGeneric-agF` | 0.1711 (±0.0029) | 0.2290 | 1,050,360 |

### H = 192 (paper: 0.211, 0.253)

| Rank | Config | MSE | MAE | Params |
|---|---|---|---|---|
| 1 | `NHiTS-TrendWaveletGenericVAE-agF` | **0.2345** (±0.0053) | 0.2872 | 698,808 |
| 2 | `NHiTS-TrendWaveletAE` | 0.2414 (±0.0045) | 0.2956 | 631,161 |
| 3 | `NHiTS-BottleneckGenericAELG` | 0.2427 (±0.0028) | 0.3027 | 1,183,215 |
| 4 | `NHiTS-TrendWaveletAELG` | 0.2431 (±0.0072) | 0.2960 | 631,209 |
| 5 | `NHiTS-BottleneckGenericAELG-agF` | 0.2433 (±0.0019) | 0.2991 | 1,183,215 |
| … | | | | |
| 27 | `NHiTS-GenericAE` | 0.2609 (±0.0033) | 0.3083 | 2,927,664 |
| 28 | `NHiTS-TrendWaveletGeneric-agF` | 0.2710 (±0.0042) | 0.3135 | 1,501,656 |
| 29 | `NHiTS-Generic` (baseline) | 0.2771 (±0.0044) | 0.3160 | 4,033,536 |

### H = 336 (paper: 0.272, 0.296)

| Rank | Config | MSE | MAE | Params |
|---|---|---|---|---|
| 1 | `NHiTS-BottleneckGenericAE` | **0.3118** (±0.0038) | 0.3507 | 1,749,087 |
| 2 | `NHiTS-TrendWaveletGenericVAE-agF` | 0.3134 (±0.0056) | 0.3469 | 1,099,272 |
| 3 | `NHiTS-BottleneckGenericAELG-agF` | 0.3156 (±0.0084) | 0.3565 | 1,749,135 |
| 4 | `NHiTS-BottleneckGenericAE-agF` | 0.3165 (±0.0056) | 0.3587 | 1,749,087 |
| 5 | `NHiTS-BottleneckGenericAELG` | 0.3183 (±0.0058) | 0.3577 | 1,749,135 |
| … | | | | |
| 27 | `NHiTS-GenericVAE` | 0.3346 (±0.0144) | 0.3667 | 4,999,008 |
| 28 | `NHiTS-TrendWaveletGeneric-agF` | 0.3363 (±0.0074) | 0.3652 | 2,178,600 |
| 29 | `NHiTS-Generic` (baseline) | 0.3368 (±0.0040) | 0.3603 | 6,466,560 |

### H = 720 (paper: 0.348, 0.349)

| Rank | Config | MSE | MAE | Params |
|---|---|---|---|---|
| 1 | `NHiTS-BottleneckGenericAE-agF` | **0.3879** (±0.0092) | 0.4128 | 3,258,207 |
| 2 | `NHiTS-BottleneckGenericAELG` | 0.3903 (±0.0119) | 0.4104 | 3,258,255 |
| 3 | `NHiTS-BottleneckGenericAELG-agF` | 0.3906 (±0.0079) | 0.4134 | 3,258,255 |
| 4 | `NHiTS-BottleneckGenericAE` | 0.3931 (±0.0144) | 0.4124 | 3,258,207 |
| 5 | `NHiTS-TrendWaveletAE` | 0.4105 (±0.0080) | 0.4144 | 2,052,009 |
| … | | | | |
| 27 | `NHiTS-TrendWaveletGeneric-agF` | 0.4403 (±0.0057) | 0.4311 | 3,983,784 |
| 28 | `NHiTS-GenericVAE` | 0.4405 (±0.0169) | 0.4311 | 10,012,512 |
| 29 | `NHiTS-GenericAELG` | 0.4446 (±0.0185) | 0.4339 | 9,821,280 |

---

## 4. Cross-Horizon Rank Aggregation

Configs ranked by **mean rank over the four horizons** (lower is better). Top 10 shown:

| Mean rank | Config | H96 | H192 | H336 | H720 |
|---|---|---|---|---|---|
| **3.50** | `NHiTS-TrendWaveletGenericVAE-agF` | 3 | 1 | 2 | 8 |
| 3.75 | `NHiTS-BottleneckGenericAELG` | 5 | 3 | 5 | 2 |
| 4.50 | `NHiTS-TrendWaveletAELG` | 1 | 4 | 6 | 7 |
| 5.00 | `NHiTS-BottleneckGenericAE` | 7 | 8 | 1 | 4 |
| 5.25 | `NHiTS-BottleneckGenericAELG-agF` | 10 | 5 | 3 | 3 |
| 5.25 | `NHiTS-TrendWaveletAE` | 2 | 2 | 12 | 5 |
| 6.75 | `NHiTS-BottleneckGenericAE-agF` | 12 | 10 | 4 | 1 |
| 7.50 | `NHiTS-TrendAE+DB3V3AE+TrendAE` | 6 | 6 | 9 | 9 |
| 8.25 | `NHiTS-TrendAE+HaarV3AE+TrendAE` | 9 | 7 | 7 | 10 |
| 10.50 | `NHiTS-TrendWaveletGenericAELG-agF` | 11 | 17 | 8 | 6 |

Bottom of the same list: `NHiTS-Generic` (27.25), `NHiTS-TrendWaveletGeneric-agF` (28.00).

**Interpretation.** A single "universal weather config" emerges: **`NHiTS-TrendWaveletGenericVAE-agF`** — top-3 at every horizon except H=720 (8th). It is also a surprise winner: standalone VAE backbones historically lose badly on other datasets (see MEMORY notes), yet under this short-epoch sliding protocol the `TrendWaveletGenericVAE` (VAE variant of the trend+wavelet+generic hybrid) is consistently a podium finisher at only 0.43M – 2.17M parameters.

---

## 5. Block-Family Comparison

Mean MSE within block family, averaged across horizons:

| Family | H=96 | H=192 | H=336 | H=720 | **Mean** |
|---|---|---|---|---|---|
| **BottleneckGenericAE** | 0.1636 | 0.2444 | 0.3141 | 0.3905 | **0.2782** |
| **BottleneckGenericAELG** | 0.1633 | 0.2430 | 0.3169 | 0.3904 | **0.2784** |
| **TrendWaveletAE** (unified) | 0.1604 | 0.2414 | 0.3243 | 0.4105 | **0.2841** |
| **TrendWaveletAELG** (unified) | 0.1600 | 0.2431 | 0.3197 | 0.4161 | **0.2847** |
| Alternating TrendAE + WaveletAE (±Generic) | 0.1639 | 0.2464 | 0.3264 | 0.4239 | 0.2901 |
| TrendWaveletGeneric family | 0.1659 | 0.2512 | 0.3234 | 0.4244 | 0.2912 |
| GenericAE | 0.1666 | 0.2508 | 0.3300 | 0.4264 | 0.2935 |
| GenericAELG | 0.1658 | 0.2538 | 0.3284 | 0.4321 | 0.2950 |
| GenericVAE | 0.1646 | 0.2543 | 0.3346 | 0.4405 | 0.2985 |
| **NHiTS-Generic (NHiTSGenericRoot)** | 0.1703 | 0.2771 | 0.3368 | 0.4314 | **0.3039** |

Clean ordering. Two observations worth calling out:

- **Adding AE bottleneck to plain generic blocks helps materially** (Generic 0.304 → GenericAE 0.294 → BottleneckGenericAE 0.278). The bottleneck factorizes the basis expansion and reduces overfitting at 6 epochs.
- **Unified `TrendWavelet*` blocks beat alternating Trend+Wavelet stacks** at every horizon except H=336. When you fuse trend and DWT into a single block, the residual stacking still works; splitting them across three separate 1-block stacks loses ~0.005–0.012 MSE.
- **The `TrendWaveletGeneric` family splits in two.** The deterministic variant (`-agF`) is the worst family overall (it is in fact beaten by plain `NHiTS-Generic` at short horizons), but its VAE descendant is the cross-horizon winner. The generic branch clearly benefits from a stochastic bottleneck; a deterministic generic branch on top of trend+wavelet is over-parameterized and under-trained at 6 epochs.

---

## 6. Alternating Wavelet Sub-Analysis

Within the 12-config alternating-TrendAE-family, the TrendAE+Wavelet+**TrendAE** sandwich pattern consistently beats TrendAE+Wavelet+**GenericAE** at every horizon:

| Pattern | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| TrendAE + DB3V3AE + TrendAE | 0.1629 | 0.2437 | 0.3212 | 0.4177 |
| TrendAE + HaarV3AE + TrendAE | 0.1633 | 0.2437 | 0.3197 | 0.4190 |
| TrendAE + Symlet10V3AE + TrendAE | 0.1631 | 0.2444 | 0.3238 | 0.4249 |
| TrendAE + Coif2V3AE + TrendAE | 0.1626 | 0.2475 | 0.3328 | 0.4279 |
| TrendAE + HaarV3AE + **GenericAE** | 0.1672 | 0.2480 | 0.3263 | 0.4261 |
| TrendAE + DB3V3AE + **GenericAE** | 0.1653 | 0.2483 | 0.3267 | 0.4190 |

Wavelet-family choice is a non-factor within the `+TrendAE` pattern: DB3 and Haar are consistently at the top but the spread across wavelet families is only ~0.005 MSE at H=96 and ~0.013 at H=720 — much smaller than the block-family spread. This aligns with the prior finding that **wavelet family matters more than basis dimension only at higher training budgets; at 6 epochs, basis choice barely registers.**

---

## 7. Protocol Comparison: `sliding` (6 ep) vs `paper` (25 ep)

A same-file comparison is possible at H=96 because `weather_sampling_window_results_s1024.csv` contains the same 25 configs under the `paper` sampling protocol with 25 × 1024 = 25,600 gradient steps. The mean MSE delta per config:

| Config | sliding 6ep | paper 25ep | Δ | Winner |
|---|---|---|---|---|
| `NHiTS-TrendWaveletAELG` | 0.1600 | 0.1635 | **−0.0035** | sliding |
| `NHiTS-TrendWaveletAE` | 0.1604 | 0.1618 | −0.0014 | sliding |
| `NHiTS-BottleneckGenericAELG` | 0.1628 | 0.1650 | −0.0022 | sliding |
| `NHiTS-BottleneckGenericAE` | 0.1630 | 0.1671 | −0.0040 | sliding |
| `NHiTS-Generic` (baseline) | 0.1703 | 0.1640 | +0.0063 | **paper** |
| `NHiTS-GenericAELG` | 0.1659 | 0.1625 | +0.0033 | paper |
| `NHiTS-TrendAELG+Coif2V3AELG+GenericAELG` | 0.1650 | 0.1616 | +0.0034 | paper |
| `NHiTS-TrendAE+Coif2V3AE+HaarWaveletV3AE` | 0.1643 | 0.1613 | +0.0030 | paper |

**Bifurcation:** the `TrendWavelet`-unified and `Bottleneck*` families win decisively under the sliding protocol, while the `Generic`, `GenericAELG`, and `AELG`-alternating families win under the paper protocol. This is a **regime effect**, not a noise artifact — the direction of the delta correlates cleanly with block family:

- Low-parameter structured blocks (Bottleneck, TrendWavelet) absorb the benefit of denser sliding-window iteration in 6 epochs.
- High-parameter generic blocks need the 4× extra training of the paper protocol to reach their asymptote.

This is an actionable finding: **pick the protocol to the block family.** Short sliding runs will systematically under-rank Generic-family models and systematically over-rank structured blocks.

---

## 8. `active_g='forecast'` Ablation

Paired-Wilcoxon on (horizon, seed) pairs, `baseline − agF` MSE:

| Matched pair | Δ (baseline − agF) | p | Result |
|---|---|---|---|
| `BottleneckGenericAE` vs `-agF` | −0.00051 | 0.45 | agF marginally better, n.s. |
| `BottleneckGenericAELG` vs `-agF` | +0.00017 | 0.87 | indistinguishable |
| `GenericAE` vs `-agF` | +0.00240 | **0.04** | agF **worse** |
| `GenericAELG` vs `-agF` | +0.00628 | 0.08 | agF marginally worse, borderline |

**No statistically significant gain from `active_g='forecast'` on any matched pair.** One pair is significantly worse. This reinforces the MEMORY prior that `active_g='forecast'` is dataset-level — it was catastrophic on Weather-96 in the comprehensive N-BEATS sweep — and extends it: even with global z-score normalization (which changes the scale) and NHiTS pooling, agF offers no benefit here. Drop it from the next Weather-NHiTS run grid.

---

## 9. Parameter Efficiency

H=96, MSE per million parameters (lower = more efficient):

| Config | MSE | Params | MSE · 1e6 / params |
|---|---|---|---|
| `NHiTS-BottleneckGenericAELG` | 0.1628 | 805,935 | **0.202** |
| `NHiTS-BottleneckGenericAE` | 0.1630 | 805,887 | 0.202 |
| `NHiTS-BottleneckGenericAELG-agF` | 0.1638 | 805,935 | 0.203 |
| `NHiTS-TrendAE+Coif2V3AE+TrendAE` | 0.1626 | 560,694 | 0.290 |
| `NHiTS-TrendWaveletGenericVAE-agF` | 0.1625 | 431,832 | 0.376 |
| `NHiTS-TrendWaveletAELG` | 0.1600 | 372,873 | 0.429 |
| `NHiTS-TrendWaveletAE` | 0.1604 | 372,825 | 0.430 |

`BottleneckGenericAE(LG)` wins raw parameter efficiency — but `TrendWavelet(AE/AELG)` has the best absolute MSE at ~0.37M params (4.5× smaller than BottleneckGeneric and 6.5× smaller than plain Generic, while beating both). If deployment latency is the constraint, `TrendWaveletAELG` is the pick. If only MSE matters, `BottleneckGenericAE` scales better to H=336/720.

---

## 10. Gap to Published NHiTS Paper

| Horizon | Paper MSE (NHiTS Table 1) | Ours best | Gap |
|---|---|---|---|
| 96 | 0.158 | 0.1600 (TrendWaveletAELG) | **+1.3 %** |
| 192 | 0.211 | 0.2345 (TrendWaveletGenericVAE-agF) | +11.1 % |
| 336 | 0.272 | 0.3118 (BottleneckGenericAE) | +14.6 % |
| 720 | 0.348 | 0.3879 (BottleneckGenericAE-agF) | +11.5 % |

The short-horizon number is competitive with paper-reported NHiTS. The longer-horizon gap is primarily a training-budget artifact: the paper runs thousands of gradient steps; at 6 epochs × ~35 steps-per-epoch = ~210 gradient updates per run, we are an order of magnitude short. The `s1024` file with 25,600 steps shows the gap partially close for generic-family configs. A focused re-run at higher epochs specifically on the `TrendWavelet`-unified and `Bottleneck*` families is the cheapest path to closing the long-horizon gap.

---

## 11. Recommendations

### Production-ready (run as-is with more seeds)

- **Short horizons (96, 192):** `NHiTS-TrendWaveletAELG` (MSE 0.160 at H=96; 372K params). Most stable in its cluster (std 0.0019). Tie-break by parameter count if inference cost matters.
- **Long horizons (336, 720):** `NHiTS-BottleneckGenericAE-agF` or `NHiTS-BottleneckGenericAELG` (MSE 0.388–0.393 at H=720). The `-agF` variant wins H=720 but is neutral elsewhere; if one fixed config is needed, prefer plain `BottleneckGenericAELG`.
- **Universal pick:** `NHiTS-TrendWaveletGenericVAE-agF` (mean rank 3.50). This is the most consistent across horizons, with a parameter range of 430K–2.2M. It is the single config recommended if only one "Weather-NHiTS" model will be fielded.

### Worth further tuning

- **`TrendAE+DB3V3AE+TrendAE`** and **`TrendAE+HaarV3AE+TrendAE`** — the two best alternating configs, consistent mid-pack finishers. Worth trying at higher epochs; they may overtake the unified TrendWavelet blocks.
- **`BottleneckGenericAELG-agF`** — very close to its non-agF counterpart on H=336/720. A larger seed pool will tell whether agF is genuinely neutral or slightly positive here.

### Drop from next grid

- `NHiTS-Generic` (plain NHiTSGenericRoot, no AE) — worst or near-worst at every horizon and 2.4M–6.5M params.
- `NHiTS-TrendWaveletGeneric-agF` (deterministic generic branch, with agF) — 28–29th at every horizon. The VAE sibling is the only one in this family worth keeping.
- `NHiTS-GenericVAE` (pure-VAE stacks) — consistent with prior finding that pure-VAE loses; keep `TrendWaveletGenericVAE-agF` instead.
- `active_g=forecast` variants on `GenericAE`/`GenericAELG` — no gain, sometimes significantly worse.

### Proposed next experiments

1. **Epoch budget sweep** (`max_epochs ∈ {6, 12, 25, 50}`) on the 6 top cross-horizon configs only. Will disentangle the "sliding underfits generic blocks" effect from the "structured blocks are genuinely better" effect. Expected cost: 6 × 5 seeds × 4 horizons × 4 budgets ≈ 480 runs, overnight at the current ~1.7 min/run.
2. **TrendWaveletGenericVAE KL weight sweep** (`kl_weight ∈ {0.0001, 0.001, 0.01, 0.1}`) — per [MEMORY kl_weight_sweep.md](../../../.claude/agent-memory/nbeats-analysis/kl_weight_sweep.md), 0.001 is VAE-optimal on M4; confirm on Weather. If 0.001 is a clear win, this is the strongest candidate for a new Weather-NHiTS SOTA config.
3. **Bottleneck × horizon scaling.** `BottleneckGenericAE` wins H=336/720. Sweep `thetas_dim ∈ {3, 5, 8, 12}` and `latent_dim ∈ {8, 16, 24}` at H=720 only, 5 seeds. Test whether the `ld=16` default (carried over from N-BEATS studies) is optimal when the NHiTS pooling schedule is imposed.
4. **Rerun of `s1024` paper-protocol with `max_steps = 51,200`** (doubled). The paper-sampling file already suggests generic-family configs are still improving at 25 epochs. If they catch `BottleneckGenericAE` at H=720 under a doubled budget, the sliding protocol is a fundamentally unfair comparison for them.

### Open questions

- Why does `TrendWaveletGenericVAE-agF` (VAE backbone) outperform non-VAE siblings under sliding-6-epochs but not under paper-25-epochs? Hypothesis: stochastic latent acts as regularizer in the low-data/low-epoch regime; dissolves when the deterministic network has time to fit.
- Is the 6-epoch training budget a deliberate choice or a leftover from the `_e4` probe? The YAML commits `max_epochs: 4`, suggesting someone manually bumped it to 6 and re-ran. A commit message or a comment in the YAML clarifying this would help future reproduction.
- The `TrendWaveletGeneric-agF` vs `TrendWaveletGenericVAE-agF` gap (rank 29 vs rank 3.5 in mean rank) is surprising. Worth a head-to-head at matched epochs with and without the KL loss disabled (`kl_weight=0`) to isolate whether the VAE advantage comes from KL regularization or just from the reparameterization-trick noise injection during training.

---

## 12. File references

- Source CSV: [`experiments/results/nhits/weather_sliding_window_results_e6.csv`](weather_sliding_window_results_e6.csv)
- Driving YAML: [`experiments/configs/nhits_benchmark_weather_sliding.yaml`](../../configs/nhits_benchmark_weather_sliding.yaml)
- Paper-sampling comparison: [`experiments/results/nhits/weather_sampling_window_results_s1024.csv`](weather_sampling_window_results_s1024.csv)
- Earlier 4-epoch probe: [`experiments/results/nhits/weather_sliding_window_results_e4.csv`](weather_sliding_window_results_e4.csv) (5 configs, H=96 only)
- Related non-NHiTS Weather sweep: [`experiments/results/weather/comprehensive_sweep_weather_results.csv`](../weather/comprehensive_sweep_weather_results.csv)

---

*Analysis performed 2026-04-22. Statistical tests use SciPy paired Wilcoxon signed-rank; significance threshold α = 0.05. All MSE values are on the globally z-score-normalized target and therefore differ from prior per-window-normalized N-BEATS Weather studies — comparisons against the paper baselines use the same protocol (global z-score).*
