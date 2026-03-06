# LG vs VAE Block Study — Interpretive Analysis Report

**Study:** Successive halving search comparing AELG (learned-gate autoencoder) and VAE (variational autoencoder) backbone families
**Datasets:** M4, Tourism, Weather-96, Traffic-96
**Search:** 19 configs × 3 rounds, two passes per config (baseline + activeG_fcast)
**Notebook:** `experiments/analysis/notebooks/lg_vae_study_insights.ipynb`

---

## Executive Summary

**AELG wins on 3 of 4 datasets. VAE wins on Traffic. The split is principled, not random.**

Across M4, Tourism, and Weather, AELG (learned-gate autoencoder) is the dominant family. It survives all three halving rounds, achieves the best final-round results, and does so with 4.3M parameters — roughly 8x fewer than NBEATS-I+G while matching or beating its OWA. On Traffic, the roles reverse completely: AELG diverges at a 56% rate and exits the search by Round 3, while VAE provides the only stable high-performing configurations.

The split reflects a meaningful architectural distinction. The AELG gate (`sigmoid(gate) * z`) works when data has learnable structure — it selects which latent dimensions to use. On Traffic's non-stationary sensor data, the gate cannot stabilize, and training collapses. VAE's KL regularization, which is a hindrance on structured data, becomes a stabilizer on Traffic.

---

## Dataset-by-Dataset Results

### M4 (OWA, lower is better)

Best config: **TrendAELG+Coif2** (forecast pass) — OWA ≈ 0.801

| Baseline | OWA | Comparison |
|---|---|---|
| TrendAELG+Coif2 | 0.801 | **New best** |
| AE+Trend | 0.802 | Matched / barely beaten |
| NBEATS-I+G | 0.806 | Beaten by 0.005 OWA |
| NBEATS-G | 0.862 | Beaten by large margin |

No VAE config survives to Round 3. VAE is eliminated by successive halving on every M4 run. AELG achieves competitive state-of-the-art performance with **~4.3M parameters** versus NBEATS-I+G's **~35.9M**.

**Round progression (AELG best config):** R1: 0.909 → R2: 0.884 → R3: 0.801 (-11.9%)

### Tourism (Val Loss, lower is better)

Best config: **TrendAELG+Coif2** (no active_g pass) — Val Loss ≈ 25.15

AELG sweeps all six final-round spots. VAE is eliminated before Round 3 (one VAE config, NBEATS-I-VAE, produces extremely high loss values that cause early elimination). The gap between AELG and VAE in Round 1 is large: mean val loss 36.9 vs 130.3 respectively.

### Weather (Val Loss / norm_mae, lower is better)

Best config: **GenericAELG** (forecast pass) — Val Loss ≈ 42.26

An unexpected result: on Weather, the **pure GenericAELG** wins over the TrendAELG+wavelet configs. This suggests that meteorological data does not benefit from trend inductive biases to the same degree as M4. The wavelet decomposition also provides less advantage when noise dominates signal structure.

AELG wins all final-round spots. VAE is present in Round 2 but eliminated before Round 3.

### Traffic (Val Loss, lower is better)

Best config: **TrendVAE+Symlet3** (forecast pass) — Val Loss ≈ 14.97

The dataset where everything reverses:
- **AELG divergence rate:** 56% of runs (val_loss ≈ 200, smape ≈ 200%)
- **VAE divergence rate:** 20% of runs
- Only VAE configs survive to Round 3

The final round is a VAE-only contest: TrendVAE+Symlet3 and TrendVAE+Coif2 split the top two spots. Even VAE has high standard deviations in Round 3 (Std ≈ 106 for forecast pass, driven by occasional late-round explosions), but the median performance is strong.

---

## Architecture Category Analysis

Across all datasets, a clear hierarchy emerges:

1. **Trend+Wavelet** (TrendAELG+X or TrendVAE+X) — best on M4, Tourism, Traffic (final winner)
2. **Pure Homogeneous** (GenericAELG, GenericVAE alone) — mid-tier on M4/Tourism; **best on Weather**
3. **NBEATS-I Style** (LG/VAE Generic replacing standard Generic) — consistently underperforms; does not replicate NBEATS-I+G advantage

The wavelet block is contributing genuine signal. The TrendAELG+Wavelet combination provides complementary inductive biases: trend extrapolation from TrendAELG plus frequency-domain structure from the wavelet basis. On Weather, this over-constrains the model, and the less-structured GenericAELG is optimal.

---

## Wavelet Family Sensitivity

Four wavelet families were tested: Haar, DB4, Coif2, Symlet3.

| Wavelet | M4 Rank (AELG) | Weather Rank | Traffic Rank (VAE) | Recommended? |
|---|---|---|---|---|
| Coif2 | 1 | 3 | 1 (tied) | Yes — best cross-dataset |
| Symlet3 | 2 | 4 | 1 (tied) | Yes — strong on Traffic |
| DB4 | 3 | 1 | 4 | Good for M4/Tourism |
| Haar | 4 | 2 | 3 | Fast, slightly lower perf |

Wavelet choice is a secondary hyperparameter. The inter-wavelet spread on M4 is ~0.05 OWA — meaningful but smaller than the family choice (AELG vs VAE) or category choice (Trend+Wavelet vs pure). **Coif2** offers the best cross-dataset reliability.

---

## Parameter Efficiency

| Family | Typical n_params (trend_wavelet config) | Notes |
|---|---|---|
| AELG | ~4.3M | Deterministic latent |
| VAE | ~4.4M | Adds fc2_mu + fc2_logvar heads |
| NBEATS-I+G | ~35.9M | Published baseline |

VAE's overhead is ~80K additional parameters (~2%) — modest in absolute terms. But AELG achieves better OWA on every matched pair in M4. The performance-per-parameter advantage for AELG is clear.

At inference time, VAE uses `z = mu` (deterministic) so there is no runtime stochasticity, but the extra head parameters remain.

---

## Cross-Dataset Consistency (Spearman Rank Correlation)

Config rankings correlate across similar datasets but break down for Traffic:

- **M4 ↔ Tourism:** Strong positive correlation — same configs tend to rank well on both
- **M4 ↔ Weather:** Moderate positive correlation
- **Anything ↔ Traffic:** Weak to negative correlation — Traffic is a separate regime

This confirms that no single configuration generalizes across all four datasets. Dataset-type discrimination is necessary for optimal deployment.

---

## Successive Halving as a Diagnostic

The halving procedure correctly identifies the better family on each dataset:
- M4/Tourism/Weather: AELG configs advance through all rounds; VAE exits by Round 2–3
- Traffic: AELG exits by Round 2 (divergence); VAE advances to Round 3

Three rounds of halving is sufficient to produce reliable winners. Running only Round 1 would under-count the AELG advantage on M4 (R1 gap is partially obscured by variance; R3 gap is definitive).

---

## Stability Summary

| Dataset | AELG Mean Std | VAE Mean Std | More Stable |
|---|---|---|---|
| M4 | Lower | Higher | AELG |
| Tourism | Lower | Higher | AELG |
| Weather | Lower | Higher | AELG |
| Traffic | Bimodal (converge or explode) | More consistent | VAE |

AELG is more reproducible across seeds on structured data. N-BEATS is known for low seed variance, and AELG preserves this property better than VAE.

---

## Deployment Decision

### Use AELG (TrendAELG+Coif2 or Symlet3) when:
- Data has structured seasonal/trend patterns (competition series, economic, energy)
- Low training variance is important (production systems needing reproducibility)
- Parameter efficiency matters
- You are uncertain about dataset type (best default)

### Use VAE (TrendVAE+Symlet3 or Coif2) when:
- Data is high-frequency sensor data with non-stationarities (traffic, IoT)
- AELG divergence has been observed in preliminary runs
- KL regularization is acceptable overhead

### Decision Tree

```
Is the data high-frequency sensor / traffic-like?
  YES → TrendVAE + Symlet3 or Coif2
        Expect: ~20% divergence; retrain if needed

  NO  → TrendAELG + Coif2 or Symlet3
        M4/Tourism: OWA ~0.801, beats NBEATS-I+G
        Weather: consider GenericAELG if overfitting occurs
        Expect: low variance, stable training, 4.3M params
```

---

## What This Study Does Not Answer

1. **Longer training:** All final-round results used ≤50 epochs. Both families may improve further.
2. **Latent dimension sensitivity:** `latent_dim` was not varied; the gate mechanism's value depends on this.
3. **Ensemble potential:** AELG and VAE have different error signatures; an ensemble may outperform either.
4. **Large dataset behavior:** VAE's regularization may become advantageous when training data is much larger.

---

*Data sources: `experiments/results/*/lg_vae_study_results.csv`*
*Pre-generated stats: `experiments/analysis/analysis_reports/lg_vae_study_analysis.md`*
*Primary artifact: `experiments/analysis/notebooks/lg_vae_study_insights.ipynb`*
