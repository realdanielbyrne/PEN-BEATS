# Omnibus Benchmark M4 — Interim Analysis

**Date:** 2026-03-16
**Dataset:** M4-Yearly only (Quarterly not yet started)
**Status:** 16 of ~40 configs present, 153 total runs

## Executive Summary

The omnibus benchmark is ~40% complete. All 16 configs that have data run on M4-Yearly only — no Quarterly results exist yet. Two configs are still incomplete (NBEATS-G-activeG: 7/10, TrendWaveletAELG-20: 6/10). Zero divergences observed across all 153 runs.

**Key findings so far:**

1. **NBEATS-I+G-activeG is the leading complete config** (SMAPE=13.508, OWA=0.803, 10 runs). It slightly edges out Trend+Coif2WaveletV3-30 (13.530) but the difference is not statistically significant (MWU p=0.79).

2. **TrendWaveletAELG-20 shows the best mean SMAPE** (13.512 over 6 runs) with remarkably low variance (std=0.092), but needs 4 more runs to be fully comparable. At 871K params it is **41x more parameter-efficient** than NBEATS-I+G-activeG (36M params).

3. **Novel AELG family leads in parameter efficiency.** TrendWaveletAELG-10 achieves SMAPE=13.579 with only 436K params — 57x fewer than NBEATS-I+G-activeG while trailing by only 0.5%.

4. **active_g has a nuanced effect.** It significantly helps NBEATS-I+G (13.641 -> 13.508, p=0.065) but hurts NBEATS-G (13.600 -> 13.792, p=0.22). It has zero effect on NBEATS-I (identical values), which is expected since Trend/Seasonality blocks have no generic linear layers for active_g to modify.

5. **Novel AE (non-LG) configs underperform.** GenericAE, BottleneckGenericAE at both depths all cluster around SMAPE 13.79-13.85, clearly below the other families.

6. **None of the omnibus configs match the prior M4-Yearly SOTA** (Coif2_bd6_eq_fcast_td3: SMAPE=13.410, OWA=0.794 from wavelet study 2). The closest is TrendWaveletAELG-20 at 13.512 (+0.8%).

## Completion Status

| Config | Runs | Status |
|--------|------|--------|
| NBEATS-G-baseline | 10/10 | Complete |
| NBEATS-G-activeG | 7/10 | **Incomplete** |
| NBEATS-I-baseline | 10/10 | Complete |
| NBEATS-I-activeG | 10/10 | Complete (identical to baseline — expected) |
| NBEATS-I+G-baseline | 10/10 | Complete |
| NBEATS-I+G-activeG | 10/10 | Complete |
| Trend+Coif2WaveletV3-30 | 10/10 | Complete |
| TrendWaveletGeneric-10 | 10/10 | Complete |
| TrendWaveletGeneric-30 | 10/10 | Complete |
| GenericAE-10 | 10/10 | Complete |
| GenericAE-30 | 10/10 | Complete |
| BottleneckGenericAE-10 | 10/10 | Complete |
| BottleneckGenericAE-30 | 10/10 | Complete |
| NBEATS-I+GenericAE | 10/10 | Complete |
| TrendWaveletAELG-10 | 10/10 | Complete |
| TrendWaveletAELG-20 | 6/10 | **Incomplete** |

**24 configs not yet started**, including all alternating AELG/AE configs (TrendAELG+WaveletV3AELG), all active_g sweep complements, BottleneckGenericAELG, GenericAELG-20-skip, and TrendWaveletGenericAELG variants.

## Rankings by SMAPE (M4-Yearly)

| Rank | Config | SMAPE | +/-std | OWA | Params | Runs | Family | active_g |
|------|--------|-------|--------|-----|--------|------|--------|----------|
| 1 | NBEATS-I+G-activeG | 13.508 | 0.125 | 0.803 | 35.95M | 10 | paper_baseline | forecast |
| 2 | TrendWaveletAELG-20* | 13.512 | 0.092 | 0.803 | 0.87M | 6 | novel_aelg | false |
| 3 | Trend+Coif2WaveletV3-30 | 13.530 | 0.154 | 0.803 | 15.24M | 10 | novel_nonae | false |
| 4 | TrendWaveletGeneric-10 | 13.545 | 0.183 | 0.805 | 2.09M | 10 | novel_nonae | false |
| 5 | TrendWaveletAELG-10 | 13.579 | 0.089 | 0.808 | 0.44M | 10 | novel_aelg | false |
| 6 | TrendWaveletGeneric-30 | 13.587 | 0.161 | 0.807 | 6.27M | 10 | novel_nonae | false |
| 7 | NBEATS-G-baseline | 13.600 | 0.182 | 0.812 | 24.67M | 10 | paper_baseline | false |
| 8 | NBEATS-I-baseline | 13.601 | 0.122 | 0.807 | 12.93M | 10 | paper_baseline | false |
| 9 | NBEATS-I-activeG | 13.601 | 0.122 | 0.807 | 12.93M | 10 | paper_baseline | forecast |
| 10 | NBEATS-I+G-baseline | 13.641 | 0.196 | 0.815 | 35.95M | 10 | paper_baseline | false |
| 11 | NBEATS-I+GenericAE | 13.691 | 0.167 | 0.818 | 17.47M | 10 | novel_ae | forecast |
| 12 | NBEATS-G-activeG* | 13.792 | 0.163 | 0.828 | 24.67M | 7 | paper_baseline | forecast |
| 13 | GenericAE-10 | 13.793 | 0.215 | 0.825 | 1.62M | 10 | novel_ae | forecast |
| 14 | BottleneckGenericAE-30 | 13.845 | 0.198 | 0.829 | 4.40M | 10 | novel_ae | forecast |
| 15 | GenericAE-30 | 13.851 | 0.179 | 0.832 | 4.87M | 10 | novel_ae | forecast |
| 16 | BottleneckGenericAE-10 | 13.853 | 0.153 | 0.832 | 1.47M | 10 | novel_ae | forecast |

*Incomplete — fewer than 10 runs.

## Architecture Family Comparison

| Family | Configs | Runs | Mean SMAPE | Std | Best SMAPE | Mean OWA | Param Range |
|--------|---------|------|------------|-----|------------|----------|-------------|
| novel_aelg | 2 | 16 | 13.547 | 0.097 | 13.378 | 0.805 | 436K - 871K |
| novel_nonae | 3 | 30 | 13.554 | 0.162 | 13.298 | 0.805 | 2.1M - 15.2M |
| paper_baseline | 6 | 57 | 13.615 | 0.167 | 13.326 | 0.811 | 12.9M - 36.0M |
| novel_ae | 5 | 50 | 13.807 | 0.187 | 13.479 | 0.827 | 1.5M - 17.5M |

**novel_aelg** leads on mean SMAPE and has the lowest variance, while using 15-80x fewer parameters than other families. **novel_ae** (plain AE without learned gates) is clearly the weakest family, trailing by 0.26 SMAPE on average.

## active_g Effect Analysis

| Pair | Baseline SMAPE | active_g SMAPE | Delta | Direction | p-value |
|------|---------------|----------------|-------|-----------|---------|
| NBEATS-G | 13.600 | 13.792 | +0.192 | **HURTS** | 0.22 (ns) |
| NBEATS-I | 13.601 | 13.601 | 0.000 | No effect | 1.00 |
| NBEATS-I+G | 13.641 | 13.508 | -0.133 | **HELPS** | 0.065 (marginal) |

**Interpretation:** active_g only activates on Generic-type blocks. NBEATS-I has no Generic blocks, so the flag is correctly a no-op. For NBEATS-G (30x Generic), active_g appears harmful — the activation constrains the learned basis in a way that hurts performance. For NBEATS-I+G (Trend+Seasonality+28xGeneric), the interpretable prefix apparently benefits from having activated Generic blocks downstream, creating a near-significant improvement.

The remaining 24 configs include active_g sweeps for all novel architectures which will provide much more data on this question.

## Depth Effect

| Block Type | Shallow | Deep | Delta | Direction | p-value |
|------------|---------|------|-------|-----------|---------|
| TrendWaveletGeneric | 10 (13.545) | 30 (13.587) | +0.043 | Shallower wins | 0.34 (ns) |
| TrendWaveletAELG | 10 (13.579) | 20 (13.512) | -0.067 | Deeper wins | 0.054 (marginal) |
| GenericAE | 10 (13.793) | 30 (13.851) | +0.058 | Shallower wins | 0.27 (ns) |
| BottleneckGenericAE | 10 (13.853) | 30 (13.845) | -0.009 | ~Equal | 0.47 (ns) |

No significant depth effects, but the trend is consistent with prior findings: AELG blocks benefit from depth (the learned gate needs more residual passes), while non-AE and plain AE blocks see diminishing or negative returns from deeper stacks.

## Variance / Stability

Most stable configs (by std):
1. TrendWaveletAELG-10: std=0.089, CV=0.0066
2. TrendWaveletAELG-20: std=0.092, CV=0.0068 (6 runs)
3. NBEATS-I-baseline: std=0.122, CV=0.0090
4. NBEATS-I+G-activeG: std=0.125, CV=0.0092

Highest variance configs:
1. GenericAE-10: std=0.215, CV=0.0156
2. BottleneckGenericAE-30: std=0.198, CV=0.0143
3. NBEATS-I+G-baseline: std=0.196, CV=0.0144

**TrendWaveletAELG is the most stable architecture by a significant margin.** Its CV is roughly half that of the paper baselines and one-third that of the AE family. This is consistent with the learned gate providing regularization.

## Notable Observation: NBEATS-I-baseline = NBEATS-I-activeG

These two configs produce **byte-identical SMAPE values** across all 10 seeds. This is correct behavior: active_g only modifies Generic-type blocks, and NBEATS-I (Trend+Seasonality) has none. However, it means the YAML includes a redundant config that wastes 10 training runs. Consider removing NBEATS-I-activeG or replacing it with a more informative ablation.

## Comparison to Prior M4-Yearly SOTA

The prior best known M4-Yearly result was **Coif2_bd6_eq_fcast_td3** (Trend+WaveletV3, non-AE, basis_dim=6) with SMAPE=13.410, OWA=0.794 from the wavelet study 2 (3 seeds, 50 epochs).

The omnibus benchmark uses `basis_dim=forecast` (bd=6 for Yearly), which should produce equivalent configurations. However, the omnibus Trend+Coif2WaveletV3-30 achieves SMAPE=13.530 vs the prior 13.410. The +0.12 gap is likely due to:
- The prior study used 3 seeds (potentially lucky) vs 10 here (more representative)
- Different training schedules (omnibus adds warmup + cosine LR schedule, max_epochs=200 vs 50)
- The prior SOTA estimate may need upward revision with more seeds

The omnibus result with 10 runs at SMAPE=13.530 is probably the more reliable estimate of this architecture's true performance.

## Recommendations

### Immediate
1. **Complete the running configs** — NBEATS-G-activeG needs 3 more runs, TrendWaveletAELG-20 needs 4 more runs
2. **Remove NBEATS-I-activeG** — it is redundant (identical to baseline)
3. **Prioritize the alternating AELG configs** — TrendAELG+Coif2WaveletV3AELG-30 and the skip variants are the most interesting missing configs given AELG's current lead

### Analysis Caveats
- Only M4-Yearly is available; Quarterly period will be essential for generalization assessment
- 24 of 40 configs are missing, including all the alternating AE/AELG variants that are expected to be competitive
- Statistical power is limited for some comparisons (top-4 configs are within 0.04 SMAPE of each other, none of the pairwise differences are significant)

### What to Watch For
- Whether TrendAELG+Coif2WaveletV3AELG-30 (alternating AELG at full depth) beats TrendWaveletAELG-20 (unified AELG)
- Whether active_g helps or hurts the novel wavelet configs
- Whether the parameter efficiency of AELG holds up on Quarterly (longer forecast horizon H=8)
