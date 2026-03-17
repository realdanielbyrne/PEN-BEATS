# Omnibus Benchmark Analysis — M4-Yearly

**Date:** 2026-03-16
**Dataset:** M4-Yearly (H=6, L=30)
**Training:** 200 max epochs, patience=20, SMAPELoss, Adam lr=0.001, warmup=15 epochs
**Seeds:** 10 runs (seeds 42-51) per config, sequential
**Total runs:** 356 across 36 configs (1 config has only 1 run)

---

## Executive Summary

The omnibus benchmark provides the first head-to-head comparison of paper baselines, novel non-AE, AE, and AELG configurations under identical training conditions (200 epochs, patience 20, 10 seeds). Key findings:

1. **The top 10 configs are separated by only 0.097 SMAPE** (13.482-13.579), and none of the pairwise differences among the top 5 are statistically significant (all MWU p > 0.57). This is an extremely tight field.

2. **NBEATS-I+G-activeG is the best paper baseline** (SMAPE=13.508, OWA=0.803) but requires 36M parameters. Novel architectures match or beat it with 5-80x fewer parameters.

3. **TrendWaveletAELG-10 achieves SMAPE=13.579 with only 436K parameters** — the most parameter-efficient competitive config by a wide margin (31x SMAPE/Mparam ratio vs 0.38 for NBEATS-I+G).

4. **active_g is NOT beneficial for most architectures on M4-Yearly.** Only NBEATS-I+G shows a meaningful (non-significant) improvement. For NBEATS-G, active_g *hurts* significantly (p=0.049).

5. **NBEATS-I-baseline and NBEATS-I-activeG produce identical results** — active_g has zero effect on Trend/Seasonality blocks (which have no generic linear layers to activate).

6. **No config in this benchmark reaches the prior SOTA of 13.410 SMAPE / 0.794 OWA** from wavelet_study_2 (Coif2_bd6_eq_fcast_td3, 50 epochs, 3 seeds). However, that result was from a different training protocol. The closest here is TrendAELG+Coif2WaveletV3AELG-30-activeG at 13.482 (1 run only — needs completion).

7. **Zero divergences across all 356 runs.** All configs are training-stable under these settings.

---

## Completion Status

- **36 of ~47 configs have results** (some YAML configs not yet run)
- **35 configs have full 10-run coverage**
- **1 config incomplete:** TrendAELG+Coif2WaveletV3AELG-30-activeG (1/10 runs)
- **11 YAML configs have no results yet** (wavelet family sweep: Sym3, Haar, mixed AELG+nonAE combos, and some active_g complements)
- **Only Yearly period completed** — Quarterly not yet started
- **Zero diverged runs** out of 356

---

## Full Rankings — M4-Yearly

| Rank | Config | SMAPE | STD | OWA | Params | Runs | Family | Delta |
|---:|:---|---:|---:|---:|---:|---:|:---|:---|
| 1 | TrendAELG+Coif2WavV3AELG-30-activeG* | 13.482 | — | 0.798 | 2.9M | **1** | novel_aelg | BEST |
| 2 | NBEATS-I+G-activeG | 13.508 | 0.125 | 0.803 | 36.0M | 10 | paper_baseline | +0.027 (+0.2%) |
| 3 | TrendAELG+Coif2WavV3AELG-30 | 13.521 | 0.119 | 0.803 | 2.9M | 10 | novel_aelg | +0.039 (+0.3%) |
| 4 | Trend+Coif2WaveletV3-30 | 13.530 | 0.154 | 0.803 | 15.2M | 10 | novel_nonae | +0.048 (+0.4%) |
| 5 | TrendAE+DB3WavV3AE-30-activeG | 13.533 | 0.054 | 0.804 | 2.9M | 10 | novel_ae | +0.051 (+0.4%) |
| 6 | TrendWaveletAELG-20 | 13.542 | 0.097 | 0.806 | 0.87M | 10 | novel_aelg | +0.060 (+0.4%) |
| 7 | TrendWaveletGeneric-10 | 13.545 | 0.183 | 0.805 | 2.1M | 10 | novel_nonae | +0.063 (+0.5%) |
| 8 | TrendAELG+Coif2WavV3AELG-30-skip | 13.560 | 0.097 | 0.807 | 2.9M | 10 | novel_aelg | +0.078 (+0.6%) |
| 9 | TrendAE+DB3WavV3AE-30 | 13.571 | 0.129 | 0.806 | 2.9M | 10 | novel_ae | +0.089 (+0.7%) |
| 10 | TrendWaveletAELG-10 | 13.579 | 0.089 | 0.808 | 0.44M | 10 | novel_aelg | +0.097 (+0.7%) |
| 11 | TrendAELG+DB3WavV3AELG-30-skip | 13.585 | 0.065 | 0.808 | 2.9M | 10 | novel_aelg | +0.103 (+0.8%) |
| 12 | TrendWaveletGeneric-30 | 13.587 | 0.161 | 0.807 | 6.3M | 10 | novel_nonae | +0.105 (+0.8%) |
| 13 | TrendAE+Coif2WavV3AE-30 | 13.591 | 0.138 | 0.808 | 2.9M | 10 | novel_ae | +0.110 (+0.8%) |
| 14 | TrendWaveletGenericAELG-20 | 13.594 | 0.118 | 0.810 | 0.90M | 10 | novel_aelg | +0.112 (+0.8%) |
| 15 | TrendWaveletGeneric-10-activeG | 13.596 | 0.106 | 0.809 | 2.1M | 10 | novel_nonae | +0.114 (+0.8%) |
| 16 | TrendAELG+DB3WavV3AELG-30 | 13.599 | 0.163 | 0.809 | 2.9M | 10 | novel_aelg | +0.117 (+0.9%) |
| 17 | NBEATS-G-baseline | 13.600 | 0.182 | 0.812 | 24.7M | 10 | paper_baseline | +0.118 (+0.9%) |
| 18 | NBEATS-I-activeG | 13.601 | 0.122 | 0.807 | 12.9M | 10 | paper_baseline | +0.119 (+0.9%) |
| 19 | NBEATS-I-baseline | 13.601 | 0.122 | 0.807 | 12.9M | 10 | paper_baseline | +0.119 (+0.9%) |
| 20 | TrendWaveletAELG-20-activeG | 13.605 | 0.064 | 0.811 | 0.87M | 10 | novel_aelg | +0.123 (+0.9%) |
| 21 | TrendWaveletGenericAELG-10 | 13.613 | 0.097 | 0.810 | 0.45M | 10 | novel_aelg | +0.131 (+1.0%) |
| 22 | TrendWaveletAELG-10-activeG | 13.628 | 0.081 | 0.812 | 0.44M | 10 | novel_aelg | +0.146 (+1.1%) |
| 23 | NBEATS-I+G-baseline | 13.641 | 0.196 | 0.815 | 36.0M | 10 | paper_baseline | +0.159 (+1.2%) |
| 24 | TrendAE+Coif2WavV3AE-30-activeG | 13.642 | 0.089 | 0.813 | 2.9M | 10 | novel_ae | +0.161 (+1.2%) |
| 25 | Trend+Coif2WaveletV3-30-activeG | 13.650 | 0.153 | 0.814 | 15.2M | 10 | novel_nonae | +0.168 (+1.2%) |
| 26 | NBEATS-I+GenericAE-noActiveG | 13.687 | 0.255 | 0.817 | 17.5M | 10 | novel_ae | +0.205 (+1.5%) |
| 27 | NBEATS-I+GenericAE | 13.691 | 0.167 | 0.818 | 17.5M | 10 | novel_ae | +0.210 (+1.6%) |
| 28 | GenericAE-10-noActiveG | 13.721 | 0.178 | 0.819 | 1.6M | 10 | novel_ae | +0.240 (+1.8%) |
| 29 | GenericAELG-20-skip | 13.752 | 0.127 | 0.824 | 3.3M | 10 | novel_aelg | +0.271 (+2.0%) |
| 30 | BottleneckGenericAE-10-noActiveG | 13.759 | 0.227 | 0.823 | 1.5M | 10 | novel_ae | +0.277 (+2.1%) |
| 31 | NBEATS-G-activeG | 13.788 | 0.143 | 0.828 | 24.7M | 10 | paper_baseline | +0.306 (+2.3%) |
| 32 | BottleneckGenericAELG-10 | 13.788 | 0.223 | 0.823 | 1.5M | 10 | novel_aelg | +0.306 (+2.3%) |
| 33 | GenericAE-10 | 13.793 | 0.215 | 0.825 | 1.6M | 10 | novel_ae | +0.312 (+2.3%) |
| 34 | BottleneckGenericAE-30 | 13.845 | 0.198 | 0.829 | 4.4M | 10 | novel_ae | +0.363 (+2.7%) |
| 35 | GenericAE-30 | 13.851 | 0.179 | 0.832 | 4.9M | 10 | novel_ae | +0.369 (+2.7%) |
| 36 | BottleneckGenericAE-10 | 13.853 | 0.153 | 0.832 | 1.5M | 10 | novel_ae | +0.372 (+2.8%) |
| 37 | BottleneckGenericAELG-20 | 14.011 | 0.337 | 0.841 | 3.0M | 10 | novel_aelg | +0.530 (+3.9%) |

*Single run only — not statistically reliable.

---

## Architecture Family Comparison

| Family | Configs | Runs | SMAPE | STD | OWA | Param Range |
|:---|---:|---:|---:|---:|---:|:---|
| novel_nonae | 5 | 50 | 13.582 | 0.153 | 0.808 | 2.1M - 15.2M |
| paper_baseline | 6 | 60 | 13.623 | 0.168 | 0.812 | 12.9M - 36.0M |
| novel_aelg | 13 | 127 | 13.645 | 0.193 | 0.813 | 0.4M - 3.3M |
| novel_ae | 12 | 120 | 13.712 | 0.198 | 0.819 | 1.5M - 17.5M |

**Pairwise MWU tests (family-level):**
- novel_ae vs novel_nonae: p=0.0001 (***) — non-AE significantly better
- novel_ae vs paper_baseline: p=0.004 (**) — baselines significantly better than AE
- novel_ae vs novel_aelg: p=0.001 (**) — AELG significantly better than AE
- novel_aelg vs novel_nonae: p=0.040 (*) — non-AE marginally better
- novel_aelg vs paper_baseline: p=0.636 (ns) — no difference
- novel_nonae vs paper_baseline: p=0.192 (ns) — no difference

**Interpretation:** The novel non-AE family (Trend+Wavelet, TrendWaveletGeneric) edges out all others on average, but the differences from paper baselines and AELG are not significant. The plain AE family is consistently the weakest. AELG's learned gate rescues the AE bottleneck to paper-baseline-competitive levels, but does not surpass non-AE approaches on average.

---

## active_g Effect Analysis

| Base Config | Base SMAPE | ActiveG SMAPE | Diff | p-value | Winner |
|:---|---:|---:|---:|---:|:---|
| NBEATS-G-baseline | 13.600 | 13.788 | -0.187 | 0.049* | **Base** |
| NBEATS-I-baseline | 13.601 | 13.601 | 0.000 | 1.000 | Identical |
| NBEATS-I+G-baseline | 13.641 | 13.508 | +0.133 | 0.160 | ActiveG (ns) |
| Trend+Coif2WaveletV3-30 | 13.530 | 13.650 | -0.120 | 0.131 | Base (ns) |
| TrendWaveletGeneric-10 | 13.545 | 13.596 | -0.051 | 0.557 | Base (ns) |
| GenericAE-10-noActiveG | 13.721 | 13.793 | -0.072 | 0.625 | Base (ns) |
| BottleneckGenericAE-10 | 13.759 | 13.853 | -0.094 | 0.131 | Base (ns) |
| NBEATS-I+GenericAE | 13.687 | 13.691 | -0.004 | 0.770 | Tie |
| TrendWaveletAELG-10 | 13.579 | 13.628 | -0.049 | 0.160 | Base (ns) |
| TrendWaveletAELG-20 | 13.542 | 13.605 | -0.063 | 0.156 | Base (ns) |
| TrendAE+DB3WavV3AE-30 | 13.571 | 13.533 | +0.037 | 0.625 | ActiveG (ns) |
| TrendAE+Coif2WavV3AE-30 | 13.591 | 13.642 | -0.051 | 0.625 | Base (ns) |

**Summary:** In 9 of 12 paired comparisons, the base (no active_g) config wins or ties. active_g significantly hurts NBEATS-G (p=0.049). Only NBEATS-I+G shows a potentially beneficial active_g effect (+0.133 improvement, p=0.160). The NBEATS-I comparison reveals that active_g has literally zero effect on Trend/Seasonality blocks — the runs produce bit-identical SMAPE values, confirming active_g only acts on Generic-type blocks' final linear layers.

**Conclusion:** active_g should NOT be a default for M4-Yearly. It is only potentially useful for the specific NBEATS-I+G architecture where it acts on the 28 Generic body stacks.

---

## Backbone Comparison (Matched Alternating Wavelet Configs)

For Coif2 wavelet at 30 stacks (no skip, no active_g):

| Backbone | Config | SMAPE | STD | OWA |
|:---|:---|---:|---:|---:|
| AERootBlockLG | TrendAELG+Coif2WavV3AELG-30 | 13.521 | 0.119 | 0.803 |
| RootBlock | Trend+Coif2WaveletV3-30 | 13.530 | 0.154 | 0.803 |
| AERootBlock | TrendAE+Coif2WavV3AE-30 | 13.591 | 0.138 | 0.808 |

AELG edges out non-AE by 0.009 SMAPE (not significant). Plain AE trails by 0.070. This confirms the backbone hierarchy: **AELG >= RootBlock > AERootBlock**.

---

## DB3 vs Coif2 Wavelet Comparison

| Backbone | Coif2 SMAPE | DB3 SMAPE | Diff | p-value |
|:---|---:|---:|---:|---:|
| AELG (no skip) | 13.521 | 13.599 | -0.078 | 0.734 ns |
| AE (no active_g) | 13.591 | 13.571 | +0.021 | 0.623 ns |
| AELG (skip) | 13.560 | 13.585 | -0.025 | 0.385 ns |

**No significant difference between Coif2 and DB3** on M4-Yearly. Coif2 has a slight edge in the AELG backbone but not in AE. Wavelet family selection remains a non-factor for M4-Yearly, consistent with prior findings.

---

## Depth Comparison

| Block | Shallow | Deep | Shallow SMAPE | Deep SMAPE | p-value | Winner |
|:---|:---|:---|---:|---:|---:|:---|
| TrendWaveletAELG | 10 stacks | 20 stacks | 13.579 | 13.542 | 0.473 | Deep (ns) |
| TrendWaveletGenericAELG | 10 stacks | 20 stacks | 13.613 | 13.594 | 0.571 | Deep (ns) |
| BottleneckGenericAELG | 10 stacks | 20 stacks | 13.788 | 14.011 | 0.064 | Shallow (ns) |
| GenericAE | 10 stacks | 30 stacks | 13.793 | 13.851 | 0.273 | Shallow (ns) |
| BottleneckGenericAE | 10 stacks | 30 stacks | 13.853 | 13.845 | 0.473 | Deep (ns) |
| TrendWaveletGeneric | 10 stacks | 30 stacks | 13.545 | 13.587 | 0.345 | Shallow (ns) |

No depth comparison is statistically significant. The TrendWavelet/AELG families show marginal improvement with more depth, while generic AE families show marginal degradation. BottleneckGenericAELG-20 is the clear outlier — the worst config overall (SMAPE 14.011, std 0.337) — suggesting this specific block type degrades at 20 stacks.

---

## Stability Analysis

**Most stable configs (lowest std):**

| Rank | Config | SMAPE | STD | Range |
|---:|:---|---:|---:|---:|
| 1 | TrendAE+DB3WavV3AE-30-activeG | 13.533 | 0.054 | 0.149 |
| 2 | TrendAELG+DB3WavV3AELG-30-skip | 13.585 | 0.065 | 0.193 |
| 3 | TrendWaveletAELG-20-activeG | 13.605 | 0.064 | 0.206 |
| 4 | TrendWaveletAELG-10-activeG | 13.628 | 0.081 | 0.215 |
| 5 | TrendAE+Coif2WavV3AE-30-activeG | 13.642 | 0.089 | 0.234 |

**Most unstable configs (highest std):**

| Rank | Config | SMAPE | STD | Range |
|---:|:---|---:|---:|---:|
| 1 | BottleneckGenericAELG-20 | 14.011 | 0.337 | 0.939 |
| 2 | NBEATS-I+GenericAE-noActiveG | 13.687 | 0.255 | 0.828 |
| 3 | BottleneckGenericAE-10-noActiveG | 13.759 | 0.227 | 0.778 |
| 4 | BottleneckGenericAELG-10 | 13.788 | 0.223 | 0.704 |
| 5 | GenericAE-10 | 13.793 | 0.215 | 0.743 |

**Pattern:** AE bottleneck blocks without structured basis (Generic/Bottleneck) are systematically more variable. Wavelet-basis and trend-basis configs are the most stable. The AE bottleneck + learned basis combination appears to regularize training.

---

## Parameter Efficiency

**Pareto-optimal configs (best SMAPE at each parameter tier):**

| Config | SMAPE | Params | Efficiency Tier |
|:---|---:|---:|:---|
| TrendWaveletAELG-10 | 13.579 | 436K | Ultra-light |
| TrendWaveletGenericAELG-10 | 13.613 | 450K | Ultra-light |
| TrendWaveletAELG-20 | 13.542 | 871K | Light |
| TrendWaveletGenericAELG-20 | 13.594 | 901K | Light |
| TrendWaveletGeneric-10 | 13.545 | 2.1M | Medium |
| TrendAELG+Coif2WavV3AELG-30 | 13.521 | 2.9M | Medium |
| Trend+Coif2WaveletV3-30 | 13.530 | 15.2M | Heavy |
| NBEATS-I+G-activeG | 13.508 | 36.0M | Very Heavy |

**TrendWaveletAELG-10 at 436K params achieves 99.1% of the best 10-run config's SMAPE** (13.579 vs 13.508). It uses **82x fewer parameters** than NBEATS-I+G-activeG. This is the headline result for parameter efficiency.

---

## Bug Report: NBEATS-I active_g

NBEATS-I-baseline and NBEATS-I-activeG produce **bit-identical** SMAPE values across all 10 seeds. This is expected behavior (Trend/Seasonality blocks have no generic linear layers for active_g to act on), but the experiment should not have been run as a separate config. These are wasted compute cycles.

---

## Comparison to Prior SOTA

Prior M4-Yearly SOTA from wavelet_study_2:
- **Coif2_bd6_eq_fcast_td3** (Trend+WaveletV3, non-AE): SMAPE=13.410, OWA=0.794 (3 seeds, 50 epochs)

This benchmark's best 10-run config:
- **NBEATS-I+G-activeG**: SMAPE=13.508, OWA=0.803 (10 seeds, 200 epochs)

The +0.098 gap could be due to:
1. **Different training protocol:** 200 epochs + patience 20 vs 50 fixed epochs. The warmup period (15 epochs) and early stopping may produce different convergence points.
2. **Sample variance:** The prior result had only 3 seeds — its confidence interval is wide.
3. **basis_dim=forecast vs basis_dim=6:** The YAML specifies `basis_dim: forecast` (=6 for Yearly), which matches `eq_fcast` (bd=H=6). These should be identical.
4. **trend_thetas_dim=3 in both.** Consistent.

The Trend+Coif2WaveletV3-30 config in this benchmark (which is the closest match to the prior SOTA config) achieves SMAPE=13.530, also above 13.410. This suggests the training protocol difference (200 max epochs + early stopping vs 50 fixed epochs) may be the primary factor.

---

## Recommendations

### Current Best Configurations (M4-Yearly)

**For maximum accuracy (high confidence):**
- TrendAELG+Coif2WaveletV3AELG-30 — SMAPE 13.521, 2.9M params, 10 runs

**For parameter efficiency (high confidence):**
- TrendWaveletAELG-20 — SMAPE 13.542, 871K params, 10 runs
- TrendWaveletAELG-10 — SMAPE 13.579, 436K params, 10 runs

**For stability (highest confidence):**
- TrendAE+DB3WaveletV3AE-30-activeG — SMAPE 13.533, std 0.054, 2.9M params

### What to Test Next

1. **Complete the 1-run config:** TrendAELG+Coif2WaveletV3AELG-30-activeG needs 9 more runs.
2. **Run Quarterly period** for all 36 configs to test cross-period generalization.
3. **Run the 11 missing wavelet family sweep configs** (Sym3, Haar, mixed AELG+nonAE).
4. **Training protocol comparison:** Run Trend+Coif2WaveletV3-30 at 50 fixed epochs (no early stopping) to reconcile with prior SOTA of 13.410.
5. **BottleneckGenericAELG-20 investigation:** Why does it collapse (SMAPE 14.011, std 0.337)? May need skip connections.

### Open Questions

1. Why does the 200-epoch early stopping protocol not reach the 50-epoch fixed-training SMAPE of 13.410? Is early stopping too aggressive?
2. Will the wavelet family sweep (Sym3, Haar) reveal any improvement over Coif2/DB3?
3. Does TrendWaveletAELG's parameter efficiency hold on Quarterly (longer horizon, different dynamics)?
4. Can BottleneckGenericAELG-20's instability be rescued by skip connections (as with GenericAELG-20)?
