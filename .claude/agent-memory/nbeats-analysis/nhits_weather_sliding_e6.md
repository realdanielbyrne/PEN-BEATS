---
name: NHiTS Weather Sliding-Window e6 Findings
description: Weather sliding-window 6-epoch NHiTS benchmark results across 29 configs and 4 horizons (2026-04-22)
type: project
---

## NHiTS Weather Sliding-Window Benchmark (`_e6`, 2026-04-22)

- Source: [`experiments/results/nhits/weather_sliding_window_results_e6.csv`](../../../experiments/results/nhits/weather_sliding_window_results_e6.csv)
- Report: [`experiments/results/nhits/weather_sliding_window_e6_analysis.md`](../../../experiments/results/nhits/weather_sliding_window_e6_analysis.md)
- 579 rows, 29 configs, 4 horizons (96/192/336/720), 5 seeds, 6 epochs, NHiTSSliding w/ sliding sampling + global z-score norm
- Driving YAML: `experiments/configs/nhits_benchmark_weather_sliding.yaml` (commits `max_epochs: 4` but CSV shows 6 — YAML was bumped between runs)

### Cross-horizon winner (mean rank)

- **`NHiTS-TrendWaveletGenericVAE-agF`** — mean rank 3.50/29. Top-3 at H=96/192/336, 8th at H=720. 432K–2.17M params. The single best "universal Weather-NHiTS" config under this protocol.
- `NHiTS-BottleneckGenericAELG` — mean rank 3.75. Best long-horizon stability.
- `NHiTS-TrendWaveletAELG` — mean rank 4.50. Wins H=96 (MSE 0.160).

### Per-horizon winners

- H=96: `TrendWaveletAELG` (MSE 0.160, 373K params) — **+1.3 % over paper 0.158** (essentially matches paper)
- H=192: `TrendWaveletGenericVAE-agF` (0.2345) — +11 % over paper
- H=336: `BottleneckGenericAE` (0.3118) — +15 % over paper
- H=720: `BottleneckGenericAE-agF` (0.3879) — +12 % over paper

### Key generalizations

- **Long-horizon crown shifts from wavelet-unified to Bottleneck.** TrendWaveletAE(LG) lead short; BottleneckGenericAE(LG) lead long (336/720).
- **Unified TrendWavelet blocks > alternating Trend+Wavelet stacks** at 3 of 4 horizons. Fusing trend+DWT in one block beats splitting across three 1-block stacks under this protocol.
- **Within alternating family, `TrendAE+Wavelet+TrendAE` sandwich > `TrendAE+Wavelet+GenericAE`.** Wavelet family choice is a non-factor at 6 epochs (DB3 ≈ Haar ≈ Symlet10 ≈ Coif2, spread < 0.013 MSE).
- **`active_g='forecast'` is neutral-to-worse on Weather-NHiTS-sliding.** Paired Wilcoxon across 4 matched pairs: no sig improvement; GenericAE-agF is sig **worse** (p=0.04). Extends the prior "agF is catastrophic on Weather" finding to the sliding-protocol regime.
- **Parameter size anti-correlates with quality.** Best cluster 370K–1.18M params; worst cluster (plain Generic, GenericAE, GenericAELG, GenericVAE) 2.4M–10M params.

### Protocol effect (sliding vs paper sampling, H=96 overlap)

- **Sliding favors structured low-param blocks** (TrendWavelet, Bottleneck) — they win by 0.001–0.004 MSE vs paper-25-ep.
- **Paper sampling favors high-param generic blocks** (Generic, GenericAELG, alternating-AELG) — they win by 0.003–0.006 MSE vs sliding-6-ep.
- Direction correlates with block family. **Pick protocol to block family, or equalize training budget before ranking.**

### Drop list

- `NHiTS-Generic` (plain NHiTSGenericRoot, 2.4M–6.5M params): rank 27–29 every horizon
- `NHiTS-TrendWaveletGeneric-agF` (deterministic TrendWaveletGeneric): rank 27–29 every horizon. VAE sibling is the opposite end of the ranking.
- `NHiTS-GenericVAE` (pure-VAE stacks): rank 15/23/27/28. Use TrendWaveletGenericVAE-agF instead.
- `active_g='forecast'` on Generic/GenericAELG: no benefit, sometimes worse

### Next experiments (prioritized)

1. Epoch budget sweep {6, 12, 25, 50} on top-6 cross-horizon configs — disentangle "sliding underfits Generic" from "structured blocks genuinely better"
2. KL weight sweep on `TrendWaveletGenericVAE-agF` {0.0001, 0.001, 0.01, 0.1} — 0.001 is VAE-optimal elsewhere, confirm on Weather
3. `BottleneckGenericAE` hyperparameter sweep at H=720 (`thetas_dim`, `latent_dim`) — confirms the NHiTS-pooling-aware optimum
4. Double-budget paper-sampling rerun — test whether generic configs close the H=720 gap with more gradient steps
