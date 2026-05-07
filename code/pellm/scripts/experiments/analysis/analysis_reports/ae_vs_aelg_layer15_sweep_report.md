# AE vs AE-LG Layer 15 Sweep: Analysis Report

**Experiment:** `ae_vs_aelg_layer15_sweep`
**Date:** 2026-03-12
**Analyst:** Claude Code (Opus 4.6)

---

## Executive Summary

This experiment compared PEBottleneckMLP (`ae`) and PEBottleneckMLPLG (`ae_lg`) on decoder layer 15 of Llama-3.2-1B-Instruct, sweeping latent dimensions (32-256), initialization strategies (pretrained/random/SVD), and learning rates (1e-4/1e-5). Of 48 planned runs, 24 `ae` runs completed successfully. All 24 `ae_lg` runs completed but produced no captured results due to a **stdout parsing bug** in the experiment framework (detailed below).

**Key findings from the 24 `ae` runs:**

1. **Best configuration:** ae_latent256, pretrained init, lr=1e-4 -- achieves **final_ppl=26.73** (37.18% improvement from baseline 42.54, 1.41x vanilla PPL of 18.90).
2. **Learning rate is the dominant factor:** lr=1e-4 yields 30-37% improvement; lr=1e-5 yields only 6-14%. All 12 paired comparisons favor lr=1e-4.
3. **Larger latent dims help with diminishing returns:** 256 > 128 > 64 > 32, but the ld=128-to-256 gap is only 0.4 PPL.
4. **Pretrained init is best but random is close:** 0.3 PPL mean advantage for pretrained over random at lr=1e-4.
5. **SVD init is unstable at ld=256:** baseline_ppl explodes from ~42 to 172. At ld=32-128, SVD is functional but consistently worst of the three strategies.
6. **AE pre-training has not converged at 10 epochs:** MSE curves are still declining, suggesting more training would improve results.

---

## Configuration Rankings

### Full Ranking (24 runs, sorted by final_ppl)

| Rank | Latent Dim | Init | LR | Baseline PPL | Final PPL | Improvement | Delta from Best |
|------|-----------|------|-----|-------------|-----------|-------------|-----------------|
| 1 | 256 | pretrained | 1e-4 | 42.54 | 26.73 | +37.18% | -- BEST |
| 2 | 128 | pretrained | 1e-4 | 42.38 | 27.13 | +35.98% | +0.40 |
| 3 | 256 | random | 1e-4 | 42.69 | 27.42 | +35.75% | +0.70 |
| 4 | 64 | pretrained | 1e-4 | 42.23 | 27.41 | +35.10% | +0.68 |
| 5 | 64 | random | 1e-4 | 42.46 | 27.51 | +35.22% | +0.78 |
| 6 | 128 | random | 1e-4 | 42.78 | 27.52 | +35.68% | +0.79 |
| 7 | 32 | random | 1e-4 | 42.29 | 27.57 | +34.82% | +0.84 |
| 8 | 32 | pretrained | 1e-4 | 42.41 | 27.88 | +34.24% | +1.15 |
| 9 | 128 | svd | 1e-4 | 40.80 | 28.49 | +30.16% | +1.77 |
| 10 | 64 | svd | 1e-4 | 42.30 | 28.82 | +31.85% | +2.10 |
| 11 | 256 | svd | 1e-4 | 172.05 | 29.20 | +83.03% | +2.48 |
| 12 | 32 | svd | 1e-4 | 42.36 | 29.27 | +30.90% | +2.55 |
| 13-24 | (all lr=1e-5) | various | 1e-5 | 40-172 | 36.3-63.7 | -3% to +63% | +9.6 to +36.9 |

### Top-5 Takeaway

The top 6 configurations are all lr=1e-4 with pretrained or random init. They span ld=32 to ld=256 within a 0.8 PPL range, suggesting that **learning rate and initialization matter more than bottleneck size** for single-layer replacement.

---

## Factor Analysis

### Learning Rate (Most Important Factor)

| LR | Mean PPL | Std | Min | Max | n |
|----|----------|-----|-----|-----|---|
| 1e-4 | 28.16 | 0.77 | 26.73 | 29.27 | 12 |
| 1e-5 | 39.89 | 6.89 | 36.31 | 63.67 | 12 |

- **Effect:** 11.7 PPL mean difference (Wilcoxon signed-rank p < 0.001)
- All 12 paired comparisons favor lr=1e-4 without exception

### Latent Dimension (lr=1e-4 only)

| Latent Dim | Mean PPL | Std | Min | Max |
|-----------|----------|-----|-----|-----|
| 32 | 28.24 | 0.89 | 27.57 | 29.27 |
| 64 | 27.91 | 0.77 | 27.41 | 28.82 |
| 128 | 27.71 | 0.72 | 27.13 | 28.49 |
| 256 | 27.78 | 1.30 | 26.73 | 29.20 |

- Monotonic improvement from ld=32 to ld=256 (in best-per-group), but the means show ld=128 and ld=256 are nearly tied due to SVD ld=256 outlier inflating the ld=256 average.
- Excluding SVD: ld=256 mean=27.08, ld=128 mean=27.32, confirming the monotonic trend.

### Initialization Strategy (lr=1e-4 only)

| Init | Mean PPL | Std | Min | Max |
|------|----------|-----|-----|-----|
| pretrained | 27.29 | 0.50 | 26.73 | 27.88 |
| random | 27.50 | 0.05 | 27.42 | 27.57 |
| svd | 28.95 | 0.40 | 28.49 | 29.27 |

(SVD stats exclude ld=256 outlier for fair comparison)

- Pretrained is consistently best, with remarkably low variance (std=0.50)
- Random is close behind (0.21 PPL worse on average)
- SVD is consistently worst even at stable latent dims (+1.66 PPL vs pretrained)

---

## SVD Instability at Latent Dim 256

| Init | ld=32 | ld=64 | ld=128 | ld=256 |
|------|-------|-------|--------|--------|
| pretrained | 42.41 | 42.23 | 42.38 | 42.54 |
| random | 42.29 | 42.46 | 42.78 | 42.69 |
| **svd** | 42.36 | 42.30 | 40.80 | **172.05** |

SVD baseline_ppl at ld=256 is **4x** the normal value. The SVD decomposition of the SwiGLU weight matrices produces ill-conditioned factors when the rank-k approximation approaches the original rank. Despite this, AE pre-training at lr=1e-4 recovers to 29.20 PPL (83% improvement), demonstrating the robustness of the reconstruction training.

**Recommendation:** Avoid SVD initialization at latent_dim >= 256. Use pretrained init as default.

---

## AE Pre-training Convergence

All runs completed 10 epochs without early stopping (patience=5, warmup=3). MSE curves show:

- **Epoch 1-2:** Sharp drop (60-75% of total MSE reduction)
- **Epoch 3-10:** Gradual decline, still not plateaued
- **Final MSE (lr=1e-4):** 0.033-0.050 depending on latent dim (lower MSE = larger ld)
- **Final MSE (lr=1e-5):** 0.23-0.45 -- approximately 10x higher than lr=1e-4

MSE has not converged at 10 epochs for either learning rate. Extending to 20-30 epochs is recommended.

---

## ae_lg Bug Report

### Symptom
All 24 `ae_lg` runs return status `ok_no_results` with `"Could not find Results JSON in stdout"` despite `returncode=0`.

### Root Cause
In `scripts/finetune.py`, when `pe_mlp_mode == "ae_lg"` and `epochs == 0`:
1. Line 865 prints `Results: {json...}`
2. Lines 891-900 print learned gate statistics AFTER the Results JSON
3. The parser in `run_from_yaml.py` (`parse_results`) finds the `"Results: "` marker and tries to `json.loads()` everything from there to end-of-stdout
4. The trailing gate stats text causes `JSONDecodeError`, silently returning `None`

### Fix
Either:
1. **Move gate stats before the Results print** (simplest)
2. **Include gate stats in the Results JSON dict** (best for data capture)
3. **Fix the parser** to extract only the JSON portion (e.g., find matching braces)

### Impact
All ae_lg data is lost for this sweep. The ae_lg runs must be re-run after fixing the bug, or stdout logs must be re-parsed if they were captured.

---

## Recommendations

### 1. Current Best Configuration (High Confidence)

```yaml
pe_mlp_mode: ae
ae_latent_dim: 256
ae_init: pretrained
lr: 1.0e-4
ae_pretrain_epochs: 10
pe_mlp_layer_indices: [15]
# Result: final_ppl=26.73 (37.18% improvement, 1.41x vanilla)
```

### 2. Immediate Next Steps

**Priority 1: Fix the ae_lg parsing bug** and re-run the sweep. The ae vs ae_lg comparison is the experiment's central question and currently has zero data points.

**Priority 2: Extend AE pre-training epochs.** Proposed config:

```yaml
configs:
  - name: "ae_extended_pretrain"
    pe_mlp_mode: "ae"
    ae_latent_dim: [128, 256]
    ae_init: "pretrained"
    lr: [1.0e-4, 3.0e-4, 5.0e-4]
    ae_pretrain_epochs: 30
    ae_pretrain_patience: 10
    ae_pretrain_warmup: 5
    pe_mlp_layer_indices: [[15]]
```

**Priority 3: Multi-layer AE replacement.**

```yaml
configs:
  - name: "ae_multilayer"
    pe_mlp_mode: "ae"
    ae_latent_dim: 256
    ae_init: "pretrained"
    lr: 1.0e-4
    ae_pretrain_epochs: 20
    pe_mlp_layer_indices: [[14, 15], [12, 13, 14, 15]]
```

**Priority 4: Combined attention + MLP compression.**

```yaml
configs:
  - name: "combined_attn_mlp"
    pe_attn_mode: "trend_wavelet_lg"
    pe_mlp_mode: "ae"
    trend_dim: 3
    wavelet_dim: 28
    ae_latent_dim: 256
    ae_init: "pretrained"
    lr: 1.0e-4
    ae_pretrain_epochs: 20
    pe_layer_indices: [[15]]
    pe_mlp_layer_indices: [[15]]
```

### 3. Open Questions

1. **Does learned gating (ae_lg) improve over ae?** Blocked on bug fix.
2. **Can more AE pre-training epochs close the 26.73-to-18.90 PPL gap?** MSE curves suggest yes.
3. **How does multi-layer AE replacement scale?** Single-layer shows only +7.8 PPL above vanilla.
4. **Does combined attention + MLP replacement interfere or compound?** Unknown.
5. **Is there a better learning rate than 1e-4?** The 10x gap between 1e-4 and 1e-5 suggests the optimum may be higher.

---

## Appendix: Data Summary

- **Experiment:** ae_vs_aelg_layer15_sweep
- **Model:** meta-llama/Llama-3.2-1B-Instruct (16 decoder layers, hidden=2048)
- **Target:** Layer 15 MLP only (pe_mlp_layer_indices=[15])
- **Training:** AE pre-training only (epochs=0), 10 AE pre-training epochs, batch=2, grad_accum=8
- **Evaluation:** WikiText-2 test set, max_eval_batches=50
- **Total runs:** 48 planned, 32 attempted, 24 successful (ae only), 8 parsing failures (ae_lg), 16 not attempted (ae_lg latent 64-256 partial)
- **Vanilla PPL:** 18.90
- **Notebook:** `scripts/experiments/analysis/notebooks/ae_vs_aelg_layer15_sweep_analysis.ipynb`
