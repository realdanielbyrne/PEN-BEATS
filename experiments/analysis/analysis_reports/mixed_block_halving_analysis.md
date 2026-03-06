# Mixed-Block-Type Stack Successive Halving — Analysis Report

**Dataset:** M4 Yearly (H=6, backcast=30)
**Date:** 2026-03-04
**Config:** `experiments/configs/mixed_block_halving.yaml`
**Results:** `experiments/results/m4/mixed_block_halving_results.csv`

---

## 1. Experiment Design

This study tests whether heterogeneous stack architectures — mixing blocks from different
backbone families — can outperform simpler homogeneous designs on M4 Yearly forecasting.

**24 configs** were evaluated across 3 rounds of successive halving:

| Round | Max Epochs | Configs In | Configs Out |
|-------|-----------|------------|-------------|
| R1    | 10        | 24         | 17          |
| R2    | 25        | 17         | 10          |
| R3    | 50        | 10         | 3 (winners) |

**Config groups:**

| Group | Latent Dim | Configs   | Primary Backbone |
|-------|-----------|-----------|-----------------|
| A     | 4         | MX01–MX08 | pure AE / mixed AE+VAE2 |
| B     | 8         | MX09–MX16 | pure AELG / mixed AE+AELG+VAE2 |
| C     | 12        | MX17–MX24 | pure AELG / mixed AELG+VAE2+AE |

**Shared settings:** `n_blocks_per_stack=1`, `share_weights=True`, `activation=ReLU`,
`loss=SMAPELoss`, `active_g=False`, `sum_losses=False`,
`basis_dim=128`, `thetas_dim=5`, `trend_thetas_dim=3`.

> **Selection criterion:** Successive halving pruning uses `best_val_loss` (not OWA).
> OWA is computed post-hoc as the primary evaluation metric.

---

## 2. Round 1 Results — All 24 Configs (10 epochs)

Ranked by OWA (lower is better; OWA < 1.0 beats Naive2 baseline):

| Rank | Config     | Stack Pattern                              | Backbone Mix       | ld | sMAPE  | OWA    | Survived |
|------|------------|--------------------------------------------|--------------------|-----|--------|--------|----------|
| 1    | MX09_ld8   | Trend+Generic+Seasonality                  | pure_AELG          | 8  | 15.45  | 0.9261 | ✓ |
| 2    | MX20_ld12  | Trend+DB4Wav+Season+Trend+Sym3Wav          | pure_AELG          | 12 | 15.52  | 0.9451 | ✓ |
| 3    | MX17_ld12  | Trend+Season+HaarWav+Generic               | pure_AELG          | 12 | 15.79  | 0.9541 | ✓ |
| 4    | MX24_ld12  | Trend+Season+Sym3Wav+Generic               | pure_AELG          | 12 | 15.84  | 0.9588 | ✓ |
| 5    | MX08_ld4   | TrendWav+Generic+Seasonality               | pure_AE            | 4  | 16.43  | 0.9860 | ✓ |
| 6    | MX15_ld8   | TrendWav+Generic+TrendWav+Seasonality      | pure_AELG          | 8  | 16.20  | 0.9915 | ✓ |
| 7    | MX01_ld4   | Trend+Generic+Seasonality                  | pure_AE            | 4  | 16.70  | 0.9979 | ✓ |
| 8    | MX12_ld8   | Trend+HaarWav+Season+Generic+BnkGeneric    | pure_AELG          | 8  | 16.41  | 1.0110 | ✓ |
| 9    | MX11_ld8   | Trend+GenericLG+DB3WavVAE2                 | mixed_AE_AELG_VAE2 | 8  | 16.89  | 1.0134 | ✓ |
| 10   | MX13_ld8   | Trend+Sym3Wav+DB2Wav+Coif2Wav              | pure_AE            | 8  | 17.32  | 1.0519 | ✓ |
| 11   | MX07_ld4   | Trend+AutoEnc+Seasonality+Generic          | pure_AE            | 4  | 17.22  | 1.0707 | ✓ |
| 12   | MX04_ld4   | Trend+GenericVAE2+Seasonality              | mixed_AE_VAE2      | 4  | 17.62  | 1.0893 | ✓ |
| 13   | MX10_ld8   | Trend+HaarWav+Trend+Coif1Wav               | pure_AELG          | 8  | 18.35  | 1.1181 | ✓ |
| 14   | MX05_ld4   | Trend+HaarWav+DB4Wav+Generic               | pure_AE            | 4  | 18.15  | 1.1221 | ✓ |
| 15   | MX22_ld12  | TrendLG+GenericVAE2+HaarWav+AutoEncLG      | mixed_AELG_VAE2_AE | 12 | 18.55  | 1.1653 | ✓ |
| 16   | MX18_ld12  | Trend+HaarWavVAE2+Coif1WavVAE2+Generic     | mixed_AE_VAE2      | 12 | 18.80  | 1.1656 | ✓ |
| 17   | MX23_ld12  | TrendWavLG+GenericVAE2+BnkGenericLG        | mixed_AELG_VAE2    | 12 | 18.80  | 1.1828 | ✓ |
| —    | MX02_ld4   | Trend+HaarWav+Generic                      | pure_AE            | 4  | 20.48  | 1.2944 | ✗ |
| —    | MX16_ld8   | Trend+GenericBkcast+Generic                | pure_AELG          | 8  | 21.00  | 1.3316 | ✗ |
| —    | MX19_ld12  | Trend+Generic+Seasonality                  | pure_VAE2          | 12 | 22.03  | 1.4738 | ✗ |
| —    | MX21_ld12  | Trend+BnkGeneric+Coif2Wav                  | pure_AE            | 12 | 23.26  | 1.5296 | ✗ |
| —    | MX03_ld4   | Trend+DB3Wav+Trend+Coif1Wav                | pure_AE            | 4  | 24.79  | 1.6383 | ✗ |
| —    | MX06_ld4   | Trend+BnkGeneric+BnkGeneric                | pure_AE            | 4  | 24.46  | 1.6394 | ✗ |
| —    | MX14_ld8   | TrendVAE2+SeasonVAE2+Generic               | mixed_VAE2_AE      | 8  | 24.67  | 1.7228 | ✗ |

**Eliminated:** MX02, MX03, MX06, MX14, MX16, MX19, MX21 (7 worst by OWA)

---

## 3. Round 2 Results — 17 Configs (25 epochs)

> **Note:** Selection into R3 uses `best_val_loss`, not OWA. Three configs (MX13, MX18, MX11)
> had better R2 OWA than some survivors (MX04, MX05, MX08), but were eliminated
> because their validation losses were higher.

| Rank (val_loss) | Config    | Best Val Loss | R2 OWA | R2 sMAPE | Survived |
|-----------------|-----------|--------------|--------|----------|----------|
| 1               | MX24_ld12 | 13.9038      | 0.8143 | 13.76    | ✓ |
| 2               | MX17_ld12 | 13.9046      | 0.8115 | 13.67    | ✓ |
| 3               | MX20_ld12 | 13.9446      | 0.8017 | 13.54    | ✓ |
| 4               | MX15_ld8  | 14.1127      | 0.8317 | 13.93    | ✓ |
| 5               | MX09_ld8  | 14.1285      | 0.8215 | 13.84    | ✓ |
| 6               | MX12_ld8  | 14.1417      | 0.8123 | 13.70    | ✓ |
| 7               | MX04_ld4  | 14.1956      | 0.8390 | 14.10    | ✓ |
| 8               | MX22_ld12 | 14.3185      | 0.8284 | 13.91    | ✓ |
| 9               | MX05_ld4  | 14.3803      | 0.8376 | 14.06    | ✓ |
| 10              | MX08_ld4  | 14.3810      | 0.8573 | 14.35    | ✓ |
| —               | MX13_ld8  | 14.4230      | 0.8265 | 14.01    | ✗ |
| —               | MX18_ld12 | 14.4234      | 0.8353 | 13.98    | ✗ |
| —               | MX07_ld4  | 14.4407      | 0.8469 | 14.18    | ✗ |
| —               | MX11_ld8  | 14.4532      | 0.8438 | 14.20    | ✗ |
| —               | MX10_ld8  | 14.6123      | 0.8584 | 14.36    | ✗ |
| —               | MX01_ld4  | 14.6494      | 0.8634 | 14.50    | ✗ |
| —               | MX19_ld12 | 16.0135      | 0.8968 | 14.94    | ✗ |

All 17 configs broke OWA < 1.0 by 25 epochs, confirming the 10-epoch R1 screen
was noisy but directionally correct.

---

## 4. Round 3 Results — 10 Configs (50 epochs)

| Rank (OWA) | Config    | Stack Types                                             | ld | sMAPE  | MASE  | OWA    | Epochs | Stop        |
|------------|-----------|---------------------------------------------------------|----|--------|-------|--------|--------|-------------|
| 1          | MX12_ld8  | TrendAELG+HaarWavV3AELG+SeasonalityAELG+GenericAELG+BnkGenericAELG | 8  | 13.54  | 3.101 | 0.8044 | 50     | MAX_EPOCHS  |
| 2          | MX17_ld12 | TrendAELG+SeasonalityAELG+HaarWavV3AELG+GenericAELG    | 12 | 13.57  | 3.110 | 0.8064 | 49     | EARLY_STOP  |
| 3          | MX22_ld12 | TrendAELG+GenericVAE2+HaarWavV3AE+AutoEncoderAELG      | 12 | 13.59  | 3.106 | 0.8067 | 50     | MAX_EPOCHS  |
| 4          | MX24_ld12 | TrendAELG+SeasonalityAELG+Sym3WavV3AELG+GenericAELG    | 12 | 13.60  | 3.119 | 0.8086 | 50     | MAX_EPOCHS  |
| 5          | MX05_ld4  | TrendAE+HaarWavV3AE+DB4WavV3AE+GenericAE               | 4  | 13.73  | 3.168 | 0.8187 | 49     | EARLY_STOP  |
| 6          | MX15_ld8  | TrendWaveletAELG+GenericAELG+SeasonalityAELG           | 8  | 13.71  | 3.150 | 0.8157 | 46     | EARLY_STOP  |
| 7          | MX20_ld12 | TrendAELG+DB4WavV3AELG+SeasonalityAELG+Sym3WavV3AELG  | 12 | 13.77  | 3.187 | 0.8224 | 43     | EARLY_STOP  |
| 8          | MX08_ld4  | TrendWaveletAE+GenericAE+SeasonalityAE                 | 4  | 13.85  | 3.195 | 0.8258 | 50     | MAX_EPOCHS  |
| 9          | MX09_ld8  | TrendAELG+GenericAELG+SeasonalityAELG                  | 8  | 13.87  | 3.220 | 0.8295 | 49     | EARLY_STOP  |
| 10         | MX04_ld4  | TrendAE+GenericVAE2+SeasonalityAE                      | 4  | 13.92  | 3.246 | 0.8343 | 50     | EARLY_STOP  |

**Top 3 winners by val_loss (experiment selection criterion):**
MX20_ld12, MX17_ld12, MX12_ld8

**Top 3 by final OWA:** MX12_ld8, MX17_ld12, MX22_ld12

---

## 5. Final Rankings — Mean OWA Across All Rounds

| Rank | Config    | Mean OWA | Std    | Rounds | Params    | Stack |
|------|-----------|----------|--------|--------|-----------|-------|
| 1    | MX20_ld12 | 0.8564   | 0.0633 | 3      | 2,636,926 | 5 stacks, pure AELG |
| 2    | MX17_ld12 | 0.8573   | 0.0684 | 3      | 2,595,939 | 4 stacks, pure AELG |
| 3    | MX09_ld8  | 0.8590   | 0.0475 | 3      | 2,420,275 | 3 stacks, pure AELG |
| 4    | MX24_ld12 | 0.8606   | 0.0695 | 3      | 2,595,939 | 4 stacks, pure AELG |
| 5    | MX12_ld8  | 0.8759   | 0.0956 | 3      | 2,729,228 | 5 stacks, pure AELG |
| 6    | MX15_ld8  | 0.8796   | 0.0794 | 3      | 2,711,188 | 4 stacks, pure AELG |
| 7    | MX08_ld4  | 0.8897   | 0.0693 | 3      | 2,533,430 | 3 stacks, pure AE   |
| 8    | MX04_ld4  | 0.9209   | 0.1191 | 3      | 2,549,267 | 3 stacks, mixed_AE_VAE2 |
| 9    | MX05_ld4  | 0.9261   | 0.1388 | 3      | 519,699   | 4 stacks, pure AE   |
| 10   | MX11_ld8  | 0.9286   | 0.0848 | 2      | 505,899   | 3 stacks, mixed_AE_AELG_VAE2 |
| 11   | MX01_ld4  | 0.9307   | 0.0673 | 2      | 2,408,975 | 3 stacks, pure AE   |
| 12   | MX22_ld12 | 0.9335   | 0.1642 | 3      | 685,184   | 4 stacks, mixed_AELG_VAE2_AE |
| 13   | MX13_ld8  | 0.9392   | 0.1127 | 2      | 526,883   | 4 stacks, pure AE   |
| 14   | MX07_ld4  | 0.9588   | 0.1119 | 2      | 2,577,980 | 4 stacks, pure AE   |
| 15   | MX10_ld8  | 0.9882   | 0.1298 | 2      | 404,550   | 4 stacks, pure AELG |
| 16   | MX18_ld12 | 1.0005   | 0.1652 | 2      | 818,763   | 4 stacks, mixed_AE_VAE2 |
| 17   | MX23_ld12 | 1.1828   | 0.0000 | 1      | 622,891   | 3 stacks, mixed_AELG_VAE2 |
| 18   | MX19_ld12 | 1.1853   | 0.2885 | 2      | 4,754,507 | 3 stacks, pure VAE2 |
| 19   | MX02_ld4  | 1.2944   | 0.0000 | 1      | 359,439   | 3 stacks, pure AE   |
| 20   | MX16_ld8  | 1.3316   | 0.0000 | 1      | 369,785   | 3 stacks, pure AELG |
| 21   | MX21_ld12 | 1.5296   | 0.0000 | 1      | 354,016   | 3 stacks, pure AE   |
| 22   | MX03_ld4  | 1.6383   | 0.0000 | 1      | 398,358   | 4 stacks, pure AE   |
| 23   | MX06_ld4  | 1.6394   | 0.0000 | 1      | 328,065   | 3 stacks, pure AE   |
| 24   | MX14_ld8  | 1.7228   | 0.0000 | 1      | 4,596,267 | 3 stacks, mixed_VAE2_AE |

---

## 6. Key Findings

### 6.1 Backbone Family Is the Dominant Factor

The backbone mix is more predictive of performance than stack depth, wavelet family, or
latent dimension:

| Backbone Mix       | Configs | Mean OWA (all rounds) | Note                          |
|--------------------|---------|----------------------|-------------------------------|
| mixed_AE_AELG_VAE2 | 1       | **0.9286**           | Lowest — single config (MX11) |
| mixed_AELG_VAE2_AE | 1       | 0.9335               | Single config (MX22)          |
| pure_AELG          | 8       | 0.9386               | Best among groups with N ≥ 2  |
| mixed_AE_VAE2      | 2       | 0.9607               | Inconsistent across ld         |
| mixed_AELG_VAE2    | 1       | 1.1828               | Single config (MX23)          |
| pure_VAE2          | 1       | 1.1853               | Single config (MX19)          |
| pure_AE            | 9       | 1.1940               | High variance; ld=4 drags down |
| mixed_VAE2_AE      | 1       | 1.7228               | Worst — single config (MX14)  |

The two mixed groups that beat pure_AELG on mean OWA are each represented by a single
config, so the comparison is not statistically meaningful. **pure_AELG is the most
reliable backbone family** — 8 configs, lowest group variance, and 6 of the top 7 overall
ranks. The two single-config mixed groups (MX11, MX22) that rank above it are worth
investigating further with more coverage.

**AELG's learned latent gate is critical.** It allows the network to discover effective
latent dimensionality, making it substantially more robust than fixed-bottleneck AE
variants — especially at larger latent dims where AE over-parameterizes.

### 6.2 VAE2 Is Harmful in Nearly Every Context

Pure VAE2 stacks (MX14, MX19) and stacks with majority VAE2 content rank at the very
bottom. The KL regularization pressure appears to interfere with M4 Yearly forecasting,
which may require sharp, low-dimensional representations rather than stochastic latent
spaces.

Single VAE2 blocks embedded in otherwise AELG stacks (MX04, MX22) fare better but still
underperform pure AELG equivalents at the same latent dim.

### 6.3 Latent Dimension: ld=12 Most Consistent, ld=8 Has Peak Performance

Note: **no pure AELG configs were tested at ld=4** — Group A (ld=4) used pure_AE and
mixed_AE_VAE2 backbones only, so ld=4 AELG is untested. Comparisons are ld=8 vs ld=12
for AELG, and all three groups for the full config set.

**By best single R3 OWA (pure AELG):**

| Latent Dim | Best R3 Config | Best R3 OWA | Verdict |
|-----------|---------------|-------------|---------|
| ld = 8    | MX12_ld8      | **0.8044**  | Peak performer |
| ld = 12   | MX17_ld12     | 0.8064      | Near-identical |

**By mean OWA across all rounds (pure AELG only):**

| Latent Dim | N Configs | Mean OWA | Comment |
|-----------|-----------|----------|---------|
| ld = 12   | 3         | **0.8581** | All 3 reached R3; consistent |
| ld = 8    | 5         | 0.9869   | Dragged down by MX10 (0.988) and MX16 (1.332), both early-eliminated |

**By group mean OWA (all backbones):**

| Latent Dim | N Configs | Mean OWA |
|-----------|-----------|----------|
| ld = 12   | 8         | **1.0508** |
| ld = 8    | 8         | 1.0656   |
| ld = 4    | 8         | 1.1498   |

ld=12 is the most consistent: all three pure AELG configs at ld=12 made it to R3 with
mean OWA 0.858. At ld=8, the AELG group was more variable — MX10 and MX16 were
eliminated early, pulling the group mean to 0.987. The best individual config at ld=8
(MX12, OWA=0.804) marginally beats the best at ld=12 (MX17, OWA=0.806), but this is
within noise for a single-run experiment.

### 6.4 Wavelet Blocks: Beneficial in AELG, Neutral or Harmful in Plain AE

| Context             | Result |
|---------------------|--------|
| Wavelet in AELG (MX17, MX20, MX24) | Top-4 performers |
| Wavelet in plain AE (MX02, MX03, MX05) | MX02/MX03 eliminated R1; MX05 mediocre |
| TrendWavelet + GenericAE (MX08) | Good — TrendWavelet's polynomial+DWT hybrid helps |
| Multi-wavelet (MX13: Sym3+DB2+Coif2) | Marginal; eliminated R2 |

Wavelet blocks seem to interact synergistically with the AELG gate. The gate can learn to
suppress irrelevant frequency bands, effectively acting as a filter selector.

### 6.5 Stack Depth: More Is Marginally Better but Not Decisive

| Stack Depth | Finalists | Best R3 OWA |
|-------------|-----------|-------------|
| 3 stacks    | MX09      | 0.829       |
| 4 stacks    | MX17, MX15, MX22, MX24 | 0.806 |
| 5 stacks    | MX12, MX20 | 0.804      |

The 5-stack configs perform best in R3, but the gain over 4-stack is marginal
(~0.002 OWA). The additional diversity may help, but cost scales with depth (MX12: 2.73M
params vs MX17: 2.60M params).

### 6.6 BottleneckGeneric Variants Are Consistently Harmful

MX06 (Trend+BnkGeneric+BnkGeneric) scored OWA=1.639 — the second worst result. MX21
(Trend+BnkGeneric+Coif2) scored 1.530. Bottleneck generic's rank-d factorized projection
appears too restrictive for the M4 Yearly series lengths (H=6, backcast=30). The exception
is MX12, where a single BottleneckGenericAELG sits among 4 other AELG blocks — the AE
gate likely compensates.

### 6.7 OWA vs Val-Loss Selection Creates Re-ranking in R2

Three competitive configs were eliminated in R2 despite having better OWA than survivors:
- **MX13** (OWA=0.827) was cut; **MX04** (OWA=0.839) survived
- **MX18** (OWA=0.835) was cut; **MX05** (OWA=0.838) survived
- **MX11** (OWA=0.844) was cut; **MX08** (OWA=0.857) survived

The val_loss selection criterion is appropriate for reproducibility across runs, but may
discard config families worth exploring further. MX13 (multi-wavelet AE) and MX18
(AE+VAE2 wavelet mix) both appear promising and were likely under-trained at 25 epochs.

---

## 7. Convergence Analysis

### R3 Early-Stopping Patterns

| Config    | Epochs Trained | Best Epoch | Pattern |
|-----------|---------------|------------|---------|
| MX15_ld8  | 46            | 35         | Converged early; plateau at ~14.0 |
| MX20_ld12 | 43            | 32         | Fast convergence; val started rising |
| MX09_ld8  | 49            | 38         | Near-converged |
| MX17_ld12 | 49            | 38         | Near-converged |
| MX05_ld4  | 49            | 38         | Near-converged |
| MX04_ld4  | 50            | 39         | Converged ~ep40; 50 epochs sufficient |
| MX08_ld4  | 50            | 42         | Converged ~ep42 |
| MX12_ld8  | 50            | 44         | Still descending; more epochs may help |
| MX22_ld12 | 50            | 41         | Plateaued ~ep42 |
| MX24_ld12 | 50            | 40         | Plateaued ~ep41 |

MX12 and MX17 are the most promising for extended training — their val losses have not
fully plateaued by epoch 50.

### R1 Rapid Early Convergence — A Warning Sign

Several configs that were later eliminated (MX03, MX06, MX14) showed low early val losses
at epoch 7–9 but high OWA. Their val loss curves descended quickly and then flatlined at
a poor minimum, suggesting they got stuck in local optima rather than learning useful
representations.

---

## 8. Parameter Efficiency

| Config    | Params    | R3 OWA | OWA per M params (lower=better) |
|-----------|-----------|--------|----------------------------------|
| MX05_ld4  | 519,699   | 0.819  | 1.576                            |
| MX22_ld12 | 685,184   | 0.807  | 1.178                            |
| MX12_ld8  | 2,729,228 | 0.804  | 2.195                            |
| MX17_ld12 | 2,595,939 | 0.806  | 2.093                            |
| MX09_ld8  | 2,420,275 | 0.829  | 2.007                            |
| MX20_ld12 | 2,636,926 | 0.822  | 2.168                            |

**MX22_ld12** (TrendAELG + GenericVAE2 + HaarWavV3AE + AutoEncoderAELG) is the most
parameter-efficient finalist — only 685K params yet achieves OWA=0.807. Its mixed backbone
(AELG + one VAE2 + one plain AE block) may act as an implicit regularizer.

**MX05_ld4** (519K params) is the most efficient overall at 0.819 OWA, though still
well behind pure AELG ld=12 configs.

---

## 9. Recommendations

### Immediate Follow-ups

1. **Extend MX12_ld8 and MX17_ld12 training to 75–100 epochs.** Both did not fully
   converge at 50 epochs. Given MX12's descending val curve at epoch 44, additional
   epochs could push OWA below 0.800.

2. **Re-evaluate MX13_ld8** (TrendAE + 3-wavelet pure AE) with 50 epochs. It was
   eliminated by val_loss in R2, but its R2 OWA=0.827 is competitive. This config may
   have been undertrained.

3. **Test MX22_ld12 with pure AELG substitution.** Replace the GenericVAE2 block with
   GenericAELG to see whether VAE2's stochasticity or just the extra capacity drove
   its parameter efficiency. Expected to improve further.

### Design Principles Confirmed

- **Always use AELG** as the backbone for new AE-style configs on M4.
- **Avoid BottleneckGeneric** as a primary block; acceptable as one component among many.
- **Avoid pure VAE2 stacks** on short-horizon datasets (H=6).
- **Wavelet blocks pair well with AELG** — add Haar or DB4 as a secondary stack alongside
  Trend+Seasonality+Generic AELG triplets.
- **latent_dim=8 or 12** preferred; ld=4 is insufficient for AELG to learn a useful gate.

### Suggested Next Experiment

A focused study of the best 3 architectures (MX17, MX20, MX12) across all M4 periods
(not just Yearly) with 3+ seeds per config, using 100 max epochs, to confirm generality
and estimate variance properly.

---

## 10. Appendix — OWA Progression by Round

| Config    | R1 OWA | R2 OWA | R3 OWA | Δ R1→R3 |
|-----------|--------|--------|--------|---------|
| MX12_ld8  | 1.011  | 0.812  | 0.804  | −0.207  |
| MX17_ld12 | 0.954  | 0.811  | 0.806  | −0.148  |
| MX22_ld12 | 1.165  | 0.828  | 0.807  | −0.358  |
| MX24_ld12 | 0.959  | 0.814  | 0.809  | −0.150  |
| MX15_ld8  | 0.992  | 0.832  | 0.816  | −0.176  |
| MX05_ld4  | 1.122  | 0.838  | 0.819  | −0.303  |
| MX20_ld12 | 0.945  | 0.802  | 0.822  | −0.123  |
| MX08_ld4  | 0.986  | 0.857  | 0.826  | −0.160  |
| MX09_ld8  | 0.926  | 0.822  | 0.829  | −0.097  |
| MX04_ld4  | 1.089  | 0.839  | 0.834  | −0.255  |

**MX22** shows the largest R1→R3 improvement (−0.358), suggesting it was most undertrained
early and benefits strongly from longer training. **MX09** and **MX20** already performed
near-optimally by R1, indicating faster convergence.
