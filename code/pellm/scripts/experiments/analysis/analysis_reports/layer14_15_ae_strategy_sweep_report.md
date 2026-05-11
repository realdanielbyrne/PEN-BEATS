# Layer 14/15 AE Strategy Sweep -- Analysis Report

**Date:** 2026-03-15
**Experiment:** `layer14_15_ae_strategy_sweep`
**Notebook:** `scripts/experiments/analysis/notebooks/layer14_15_ae_strategy_sweep_analysis.ipynb`

---

## Executive Summary

This experiment tested 4 PE MLP modes (ae, ae_lg, vae, vae_lg) with 5 seeded replicates each, replacing **layers 14 and 15 simultaneously** -- the first multi-layer AE replacement experiment. All 20 runs beat vanilla Llama-3.2-1B-Instruct (18.90 PPL), with the best reaching 16.46 PPL (0.871x vanilla).

**Key result:** AE modes significantly outperform VAE modes (p=0.014, d=-0.80). Learned gating (_lg) has no meaningful effect. Two-layer replacement costs ~1.3 PPL vs single-layer (15.44 -> 16.84 mean), a modest and expected degradation.

---

## Experiment Design

| Parameter | Value |
|---|---|
| PE MLP modes | ae, ae_lg, vae, vae_lg |
| Layers replaced | 14, 15 (parallel) |
| ae_init | pretrained |
| ae_inner_init | svd |
| ae_latent_dim | 512 |
| AE pretrain epochs | 5 (early stopping, warmup=2, patience=3) |
| LM FT epochs | 1 |
| lr | 1e-4 |
| batch_size | 2, grad_accum=8 (effective=16) |
| Seeds | 42, 123, 456, 789, 1024 |
| AE dataset | wikitext2 (10k samples) |
| Total runs | 20 (4 modes x 5 seeds) |

---

## Rankings

### By Mode (mean final PPL)

```
1. ae_lg:  final_ppl=16.8292 +/- 0.0984 (89.25% improvement) -- BEST
2. ae:     final_ppl=16.8485 +/- 0.1136 (89.17% improvement) -- +0.019 ppl
3. vae_lg: final_ppl=16.8846 +/- 0.2400 (88.89% improvement) -- +0.055 ppl
4. vae:    final_ppl=17.1488 +/- 0.3108 (88.47% improvement) -- +0.320 ppl
```

### Top 10 Individual Runs

```
 1. vae_lg seed=123:  final_ppl=16.4605 (89.17%) -- BEST OVERALL
 2. ae     seed=1024: final_ppl=16.6804 (89.28%)  +0.220
 3. ae_lg  seed=1024: final_ppl=16.7139 (89.32%)  +0.253
 4. ae_lg  seed=789:  final_ppl=16.7681 (89.28%)  +0.308
 5. ae     seed=789:  final_ppl=16.7914 (89.21%)  +0.331
 6. ae_lg  seed=42:   final_ppl=16.8052 (89.26%)  +0.345
 7. vae    seed=42:   final_ppl=16.8424 (88.68%)  +0.382
 8. ae     seed=42:   final_ppl=16.8857 (89.15%)  +0.425
 9. ae_lg  seed=123:  final_ppl=16.9069 (89.20%)  +0.446
10. ae     seed=123:  final_ppl=16.9200 (89.13%)  +0.460
```

Note: The single best run is vae_lg seed=123 (16.46), but this is an outlier driven by favorable seed -- vae_lg's mean (16.88) is worse than ae/ae_lg. This illustrates why mean+std matters more than single-run winners.

---

## Statistical Analysis

### Pairwise Comparisons (Mann-Whitney U, n=5 per group)

| Comparison | U | p-value | Cohen's d | Delta PPL | Interpretation |
|---|---|---|---|---|---|
| ae vs ae_lg | 14.0 | 0.841 | +0.181 | +0.019 | Not significant |
| ae vs vae | 3.0 | 0.056 | -1.283 | -0.300 | Marginal |
| ae vs vae_lg | 6.0 | 0.222 | -0.192 | -0.036 | Not significant |
| ae_lg vs vae | 2.0 | 0.032 | -1.386 | -0.320 | **Significant** |
| ae_lg vs vae_lg | 6.0 | 0.222 | -0.302 | -0.055 | Not significant |
| vae vs vae_lg | 19.0 | 0.222 | +0.952 | +0.264 | Not significant |

### Grouped Comparisons (n=10 per group)

| Comparison | U | p-value | Cohen's d | Delta PPL |
|---|---|---|---|---|
| AE family vs VAE family | 17.0 | **0.014** | -0.803 | -0.178 |
| Non-gated vs Gated | 62.0 | 0.385 | +0.620 | +0.142 |

**Interpretation:** The AE-vs-VAE distinction is the only statistically significant factor. Learned gating shows a weak trend toward improvement but is not significant, consistent with prior findings.

---

## AE Pre-training Analysis

### MSE Convergence

| Mode | MSE Epoch 1 | MSE Epoch 5 | Reduction | AE Pretrain PPL |
|---|---|---|---|---|
| ae | 0.0662 | 0.0233 | 64.8% | 35.58 |
| ae_lg | 0.0703 | 0.0238 | 66.2% | 36.03 |
| vae | 3.3041 | 3.0633 | 7.3% | 51.62 |
| vae_lg | 3.3021 | 3.0602 | 7.3% | 52.92 |

**Critical finding:** VAE pre-training MSE is ~130x higher than AE and barely decreases (7.3% reduction vs 65%). The VAE loss includes a KL divergence term that appears to dominate the reconstruction objective, resulting in poor activation reconstruction. Despite this, LM fine-tuning recovers most of the gap (from ~16 PPL difference at ae_pretrain stage to ~0.3 PPL at final).

### Baseline PPL Differences

The modes produce different baseline PPL (before any training), indicating architectural differences affect the initial disruption:

| Mode | Baseline PPL | Params Total | Params Trainable |
|---|---|---|---|
| vae | 148.80 | 1,146,685,442 | 716,244,994 |
| vae_lg | 152.03 | 1,146,686,466 | 716,246,018 |
| ae | 155.64 | 1,145,646,080 | 715,205,632 |
| ae_lg | 156.49 | 1,145,647,104 | 715,206,656 |

VAE modes have slightly lower baseline PPL (less initial disruption) but ~1M more parameters due to the VAE encoder/decoder structure.

---

## Parameter Savings vs Vanilla Llama

### Per-Layer MLP Parameter Counts

The vanilla Llama-3.2-1B-Instruct uses SwiGLU MLPs with `hidden_size=2048` and `intermediate_size=8192`. Each AE bottleneck replaces the three large projections (`gate_proj`, `up_proj`, `down_proj`) with a four-layer autoencoder: `hidden(2048) → mid(1024) → latent(512) → mid(1024) → hidden(2048)`.

| Component | Vanilla LlamaMLP | AE Bottleneck | AE+LG | VAE | VAE+LG |
|---|---|---|---|---|---|
| **Architecture** | gate/up/down proj (no bias) | fc1→fc2→fc3→fc4 (with bias) | +latent_gate | fc1→mu/logvar→fc3→fc4 (no bias) | +latent_gate |
| gate_proj / fc1 | 2048×8192 = 16.78M | 2048×1024 = 2.10M | 2.10M | 2048×1024 = 2.10M | 2.10M |
| up_proj / fc2 | 2048×8192 = 16.78M | 1024×512 = 0.52M | 0.52M | mu: 0.52M + logvar: 0.52M | 1.05M |
| down_proj / fc3 | 8192×2048 = 16.78M | 512×1024 = 0.53M | 0.53M | 512×1024 = 0.52M | 0.52M |
| fc4 | — | 1024×2048 = 2.10M | 2.10M | 1024×2048 = 2.10M | 2.10M |
| Gate/beta | — | — | 512 | 1 | 513 |
| **Per-layer total** | **50.33M** | **5.25M** | **5.25M** | **5.77M** | **5.77M** |
| **Reduction** | — | **89.6%** | **89.6%** | **88.5%** | **88.5%** |

### Two-Layer (14+15) Savings

| Metric | Vanilla | ae / ae_lg | vae / vae_lg |
|---|---|---|---|
| MLP params (2 layers) | 100.66M | 10.49M / 10.50M | 11.53M / 11.54M |
| MLP param reduction | — | **90.17M (89.6%)** | **89.13M (88.5%)** |
| Full model params | 1,235.81M | 1,145.65M / 1,145.65M | 1,146.69M / 1,146.69M |
| Full model reduction | — | **90.17M (7.3%)** | **89.13M (7.2%)** |

### Scaling Projection: What If All 16 Layers Were Replaced?

The MLP layers account for 805.31M of the model's 1,235.81M total parameters (65.2%). Replacing all 16 layers with AE bottlenecks would yield dramatic savings:

| Scenario | MLP Params | Total Model Params | Total Reduction | Final PPL (est.) |
|---|---|---|---|---|
| Vanilla (0 layers) | 805.31M | 1,235.81M | — | 18.90 |
| 1 layer (15) | 760.23M | 1,190.73M | 3.6% | 15.44 (measured) |
| 2 layers (14-15) | 715.15M | 1,145.65M | 7.3% | 16.84 (measured) |
| 4 layers (12-15) | 624.98M | 1,055.48M | 14.6% | TBD |
| 8 layers (8-15) | 444.65M | 875.14M | 29.2% | TBD |
| **16 layers (all)** | **83.96M** | **514.47M** | **58.4%** | TBD |

At full replacement, the model would shrink from **1.24B → 0.51B parameters** — a 58% total reduction — with all savings concentrated in the MLP layers (which go from 805M → 84M, a **89.6% MLP reduction**). The non-MLP components (embeddings, attention projections, layer norms, lm_head) remain at 430.5M and are unaffected.

### Efficiency vs Quality Trade-off

The current two-layer configuration achieves a favorable trade-off:

- **7.3% parameter reduction** with **11% perplexity improvement** (18.90 → 16.84)
- The AE bottleneck is not just smaller — it produces a *better* model after fine-tuning
- Parameter savings are modest at 2 layers but scale linearly; the key question is how quality degrades as more layers are replaced
- Current data points: 1 layer = 0.82× vanilla PPL, 2 layers = 0.89× vanilla PPL — both still beat baseline

---

## Two-Layer vs Single-Layer Comparison

| Metric | Layer 15 Only (prior) | Layers 14+15 (this) | Delta |
|---|---|---|---|
| Best final_ppl | 15.44 | 16.46 | +1.02 |
| Mean final_ppl (ae family) | ~15.54 | 16.84 | +1.30 |
| vs Vanilla | 0.817x | 0.871x | |
| All runs beat vanilla? | Yes | Yes | |

The ~1.3 PPL cost of adding layer 14 is moderate. Both layers still beat vanilla. This suggests multi-layer expansion is viable, especially with more LM FT epochs.

---

## Key Findings

1. **AE > VAE (significant):** The AE family beats VAE by 0.18 PPL on average (p=0.014). This is driven by dramatically better AE pre-training reconstruction (MSE 0.023 vs 3.06). The VAE KL term needs tuning.

2. **Gating is a non-factor:** ae_lg vs ae = 0.02 PPL difference. This is consistent across all prior experiments -- learned gating adds negligible value with LM fine-tuning.

3. **Multi-layer replacement is viable:** Adding layer 14 costs ~1.3 PPL vs layer 15 alone. All 20 runs still beat vanilla Llama.

4. **LM fine-tuning dominates:** Despite VAE's terrible pre-training (ae_pretrain_ppl ~52 vs AE's ~35.5), 1 epoch of LM FT brings VAE within 0.3 PPL of AE. The pre-training quality matters less than the LM FT phase.

5. **Low variance within AE family:** ae and ae_lg both have std < 0.12 PPL across seeds, indicating highly reproducible results. VAE has 3x the variance (0.31).

---

## Recommendations

### Next Experiments

1. **More LM FT epochs for two-layer configs:**
   - Rationale: Single-layer with 1 epoch = 15.44; two-layer may close the gap with 2-3 epochs
   - Proposed: Same config as this sweep, but `epochs: [1, 2, 3]` for ae mode only

2. **Three-layer replacement (layers 13-15):**
   - Rationale: Layer 14 cost was ~1.3 PPL; test whether layer 13 adds similar or less
   - Expected baseline PPL: ~200-300 (extrapolating from prior layer sensitivity data)

3. **Sequential (staged) two-layer training:**
   - Rationale: Train layer 15 AE first, freeze it, then add layer 14
   - May preserve layer 15's quality better than parallel training
   - Requires manual two-step process (not supported by run_from_yaml.py)

4. **VAE KL weight tuning (if pursuing VAE):**
   - The 7.3% MSE reduction in 5 epochs suggests the KL penalty is far too high
   - Try `kl_weight: [0.001, 0.01, 0.1]` if the VAE implementation supports it
   - Low priority unless VAE has theoretical advantages worth pursuing

5. **Lower latent dimensions for two-layer:**
   - Prior finding: ld=32 was only 0.095 PPL worse than ld=512 with single layer
   - Test whether this holds for two-layer replacement (higher compression target)

### Proposed YAML for Multi-Epoch Two-Layer Sweep

```yaml
experiment:
  name: "layer14_15_epoch_sweep"

defaults:
  pe_attn_mode: "standard"
  pe_mlp_mode: "ae"
  ae_init: "pretrained"
  ae_inner_init: "svd"
  ae_latent_dim: 512
  dtype: "bfloat16"
  batch_size: 2
  grad_accum_steps: 8
  lr: 1.0e-4
  patience: 2
  freeze_base: true
  ae_dataset: "wikitext2"
  ae_pretrain_epochs: 5
  ae_pretrain_early_stopping: true
  ae_pretrain_warmup: 2
  ae_pretrain_patience: 3
  seed: [42, 123, 456]

configs:
  - name: "epoch_sweep"
    pe_mlp_layer_indices: [[14, 15]]
    epochs: [1, 2, 3]
```

### Current Best Configuration (Two-Layer)

**ae or ae_lg, pretrained init, SVD inner, ld=512, layers 14+15, 5 AE pretrain epochs, 1 LM FT epoch, lr=1e-4**
- Mean final_ppl: 16.83 (ae_lg) / 16.85 (ae)
- 0.89x vanilla Llama
- Confidence: HIGH (5 seeds, tight std < 0.12)

### Overall Project Best

**ae, pretrained init, SVD inner, ld=512, layer 15 only, 5 AE pretrain epochs, 1 LM FT epoch, lr=1e-4**
- Best: 15.44, Mean: ~15.54
- 0.82x vanilla Llama

---

## Open Questions

1. **How many layers can be replaced before quality degrades below vanilla?** Current data: 1 layer = 0.82x, 2 layers = 0.89x. Linear extrapolation suggests ~5-6 layers before crossing 1.0x, but this likely isn't linear.

2. **Does more LM FT close the multi-layer gap?** Two-layer configs may converge to single-layer quality with more FT epochs.

3. **Does sequential training beat parallel?** Theoretical argument: yes (preserves fitted weights). But the parallel approach is simpler and may be close enough.

4. **Is the VAE architecture fundamentally worse, or just poorly tuned?** The KL weight appears too high. If tunable, VAE could potentially match AE while providing a more principled latent space.
