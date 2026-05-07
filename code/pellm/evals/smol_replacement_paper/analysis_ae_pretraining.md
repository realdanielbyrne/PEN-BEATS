# AE Pre-Training: Technique, Effectiveness, and Remaining Questions

**Generated:** 2026-05-07
**Scope:** the AE bottleneck MLP block (`pe_mlp_mode ∈ {ae, ae_lg}`) and its two distinct training regimes:

1. **Fine-tune two-phase pre-training** — the `--ae-pretrain-epochs` workflow in
   [scripts/finetune.py](../../scripts/finetune.py): cache teacher MLP I/O, train AE bottleneck
   to reconstruct, then LM-fine-tune on top of an existing pretrained Llama.
2. **From-scratch joint pre-training** — `pe_mlp_mode: ae` baked into a SmolLM2-class run via
   [scripts/pretrain_smol_replacement.py](../../scripts/pretrain_smol_replacement.py); no separate
   reconstruction phase — the AE block is trained jointly with the LM from random init under
   FineWeb-Edu.

Companion docs: [analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md](analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md)
(downstream-benchmark comparison) and
[smol_replacement_paper_inflight_analysis_2026-05-03.md](../../scripts/experiments/results/smol_replacement_paper_inflight_analysis_2026-05-03.md)
(loss-trajectory comparison).

## TL;DR

- **Fine-tune regime is well-characterized.** Init hierarchy is settled
  (pretrained > fourier > cur > random ≫ SVD); fc2/fc3 must use SVD; pre-training plateaus at
  ~10 epochs on WikiText-2; the learned-gate (`_lg`) variant adds ~0 PPL on safe inits.
- **From-scratch regime shows a real but bounded capacity gap.** `ae_mlp` at SmolLM2-135M scale
  reaches val PPL 26.45 at 3.0B tokens vs baseline 19.82 — a stable +0.288-nat gap that does not
  close in the cosine tail.
- **Downstream impact is concentrated on long-range completion.** Macro-avg accuracy across 7
  zero-shot tasks: baseline 38.51% → ae_mlp 35.43% (-3.08); LAMBADA ppl 142.87 → 431.21 (3.0×)
  is the dominant regression, while reasoning tasks (hellaswag, piqa, arc) move only 3-4 points.
- **The biggest open question is whether the gap is capacity-limited or distillation-limited.**
  `ae_mlp_512` is defined in YAML but not launched; KD has not been ported into the from-scratch
  pretraining loop; the fc2/fc3 SVD rule has not been transferred from fine-tune to from-scratch.

## 1. The technique

### 1.1 Fine-tune two-phase pre-training (`--ae-pretrain-epochs`)

The two-phase loop is gated at [scripts/finetune.py:2634](../../scripts/finetune.py#L2634) on
`args.ae_pretrain_epochs > 0 and args.pe_mlp_mode in _RECON_PE_MLP_MODES`.

**Phase 1 — teacher activation cache.** A frozen teacher (default vanilla Llama; configurable via
`--ae-teacher`) is run over the activation-caching dataset once. For each replaced decoder layer,
the MLP block input/output tensors (flattened to `(batch*seq, hidden)`) are written to per-layer
`.pt` files under `--ae-cache-dir`. The teacher is then unloaded from GPU before Phase 2 begins.
This avoids holding teacher + student + optimizer state simultaneously — the original motivation
for the two-phase split was OOM at 1B-class scale.

**Phase 2 — per-layer reconstruction.** Each AE bottleneck layer is trained in isolation against
its cached `(input → output)` pairs under MSE loss. The teacher is gone; only the AE module + its
optimizer state are resident. Optional early stopping (`--ae-pretrain-early-stopping`,
`--ae-pretrain-warmup`, `--ae-pretrain-patience`) and a per-layer `CachedActivationDataset` driver
(see [scripts/finetune.py:1836](../../scripts/finetune.py#L1836)) finish the loop. The AE state
dict is then loaded into the PE-Llama student before LM fine-tuning starts at
[scripts/finetune.py:2993](../../scripts/finetune.py#L2993).

The activation-caching dataset is **independent** of the LM dataset:

- `--dataset` selects the LM fine-tuning + perplexity dataset.
- `--ae-dataset` selects the activation-caching dataset (default `wikitext2`; `fineweb` and
  `slimpajama` supported with `--ae-cache-num-samples`).

In wave-pipeline runs the prior wave's output directory is auto-wired as `--ae-teacher`, so each
new wave's AE layers see activations *in context* of the layers compressed in earlier waves.

### 1.2 From-scratch joint pre-training

In `pretrain_smol_replacement.py` there is no Phase 1. The AE block is initialized from the random
init path (no teacher exists yet at from-scratch time), then trained jointly with the LM under the
standard cosine-decay recipe documented in [CLAUDE.md](../../CLAUDE.md). The `ae_mlp` variant of
`smol_replacement_paper` is the canonical from-scratch test case; the parallel-launcher recipe and
A/B-integrity rules apply unchanged.

Crucially, **none of the fine-tune-regime findings (init hierarchy, fc2/fc3 SVD rule) are wired
into this path** — they all assume a pretrained Llama exists to project weights from. This is one
of the main open questions in §4.

## 2. What is known (fine-tune regime)

### 2.1 Initialization hierarchy

Source: `ae_init_methods_layer15_sweep` (80 runs, 5 inits × 4 latent dims × 2 modes × 2 seeds,
lr=1e-4, layer 15 only, 2026-03-13).

| Init mode  | Mean PPL | Δ vs best | Status |
|------------|---------:|----------:|---|
| pretrained |   26.84  |        —  | Best — least-squares fit from LlamaMLP rows |
| fourier    |   27.01  |    +0.17  | Strong alternative (FFT denoising of truncated weights) |
| cur        |   27.22  |    +0.37  | Stable but trails |
| random     |   27.47  |    +0.63  | Catches up partially, never overtakes |
| svd        |        — |        —  | Numerically unstable at ld≥256 (baseline blows up 4-27×); even at ld=128 it is the worst |

All pairwise differences are statistically significant (Wilcoxon p<0.001, Cohen's d > 1.3).
Pretrained wins 8/8 paired comparisons against every other init.

### 2.2 fc2/fc3 must use SVD fallback

Source: `ae_init_fc2fc3_strategy`. Extending an algorithm-specific init (pretrained, fourier, cur)
across the inner bottleneck layers (fc2/fc3) degrades downstream PPL by **+0.24 to +0.42** vs the
SVD fallback. SVD control runs are unchanged, isolating the effect to the fc2/fc3 strategy
change. The interpretation is straightforward — SVD gives the Eckart-Young-optimal rank-k
approximation, and the moderate compression ratio at fc2/fc3 (1024→512) is exactly the regime
where that optimality dominates heuristic methods.

**Operational rule:** apply algorithm-specific init to fc1/fc4 only; always use SVD for fc2/fc3.

### 2.3 Pre-training epoch limit on narrow data

Source: epoch sweeps + the early-regime MSE↔PPL correlation (r ≈ 0.985).

WikiText-2's training set is ~4755 chunks of 512 tokens. Teacher activations sampled from this
dataset cover only a narrow slice of the full activation manifold. The MSE-PPL correlation is
high in epochs 1-10, then decays — beyond the threshold further MSE improvement is just memorizing
WikiText-specific quirks, and the AE can land in a basin the downstream LM optimizer struggles to
escape.

**Operational rule:** default `ae_pretrain_epochs: 10` on WikiText-2. Longer pre-training is only
justified with diverse data (FineWeb / SlimPajama), and the diverse-data ceiling has not been
measured (see §4).

### 2.4 Activation-caching dataset choice

`ae_dataset_comparison` (latent dims {32, 256, 512}; ae and ae_lg) showed that
`--ae-dataset fineweb` — even at modest sample counts — produces a more diverse activation cache
than WikiText-2 at fixed sample size. This is the recommended default for any new fine-tune AE
pre-training.

### 2.5 Learned gate (`_lg`) is mostly inert on safe inits

Source: `ae_vs_aelg_layer15_sweep` and the broader `_lg` analysis in
project_lg_gate_findings.md (omitted).
Across 108 paired comparisons, the plain `ae` block wins ~62% of the time at typical effect sizes
of ±0.05–0.5 PPL. The `_lg` gate's only demonstrated value is recovering catastrophic CUR-style
inits (50–220 PPL recovered vs plain's 200–1500). For from-scratch pretraining there is no
pathological init to recover from — `_lg` is a no-op or slight regression.

**Operational rule:** default to plain `ae`. Reserve `ae_lg` for fine-tunes with high-risk init
modes only.

## 3. What is known (from-scratch regime, current run)

The `smol_replacement_paper` experiment is the live data source.
[smol_replacement_paper.yaml](../../scripts/experiments/smol_replacement_paper.yaml) defines a
SmolLM2-135M-class architecture (hidden=576, intermediate=1536, 30 layers) trained from scratch
on FineWeb-Edu under a fixed 3.0B-token budget per variant.

### 3.1 AE-MLP final state at 3.0B tokens

| Variant  | Loss (val) | PPL (val) | Tokens | Compression vs baseline | Status |
|----------|-----------:|----------:|-------:|------------------------:|---|
| baseline |  2.987     |  19.82    | 3.000B | 0% (134.5M params) | complete |
| ae_mlp   |  3.275     |  26.45    | 3.000B | -48.5% (69.3M params; ~90% MLP-block reduction) | complete |

Source: `training_manifest.json` for each variant under
`<pellm_data_root>/trainedmodels/smol_replacement_paper/`. Note both were trained under the legacy
2-GPU DDP recipe (`grad_accum_steps=32`, `gradient_checkpointing=true`); LR trajectory is
mathematically identical to the new 1-GPU recipe but per-step gradient sharding differs (see
`recipe_drift_note.json` and the A/B integrity rule in [CLAUDE.md](../../CLAUDE.md)).

### 3.2 Trajectory: the gap is stable, not closing

From the matched-token comparison in the 2026-05-03 in-flight analysis:

| Tokens | baseline loss | ae_mlp loss | gap (nats) |
|------:|---:|---:|---:|
|   250M | 3.918 | 4.369 | +0.452 |
|   500M | 3.521 | 3.864 | +0.344 |
|   750M | 3.365 | 3.674 | +0.309 |
| **1.00B** | 3.274 | 3.558 | **+0.284** |
|  1.25B | 3.210 | 3.482 | +0.272 |
|  2.75B | 2.989 | 3.278 | +0.289 |

The gap narrows during warmup/early descent, then **flattens at ≈ +0.28 nats from ~1.0B tokens
onward and slightly widens in the cosine tail**. The cosine-tail slopes match between baseline and
ae_mlp (both -0.016 / 250M tokens over the last three eval pairs). The capacity gap is real and
persistent under matched compute — additional training does not erode it.

### 3.3 Downstream impact is concentrated on LAMBADA

Source: [analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md](analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md),
zero-shot via lm-evaluation-harness 0.4.11.

| Task (metric)         | baseline      | ae_mlp        | Δ      |
|-----------------------|---------------|---------------|-------:|
| lambada_openai (acc)  | 24.22         | 18.07         | -6.15  |
| lambada_openai (ppl)  | 142.87        | 431.21        | +288.34 (3.02×) |
| hellaswag (acc_norm)  | 31.32         | 27.63         | -3.69  |
| piqa (acc_norm)       | 61.97         | 59.14         | -2.83  |
| arc_easy (acc_norm)   | 45.50         | 41.29         | -4.21  |
| arc_challenge (acc_norm) | 25.09      | 22.61         | -2.48  |
| winogrande (acc)      | 51.30         | 51.46         | +0.16  |
| openbookqa (acc_norm) | 30.20         | 27.80         | -2.40  |
| **Macro-avg**         | **38.51**     | **35.43**     | **-3.08** |

LAMBADA — long-range next-word completion — is by far the largest regression (3× ppl, vs ~10-15%
relative on the reasoning tasks). This is consistent with the picture that the 90% MLP-block
compression hurts the model's ability to integrate long-range context into a single next-token
prediction, while pattern-matching reasoning tasks tolerate it better.

### 3.4 What the run says about AE pre-training as a technique

- **AE-MLP is genuinely lossy at this scale.** The +0.28-nat plateau is a real capacity floor, not
  a transient artifact of warm-up or the cosine schedule. This is internally consistent with the
  fine-tune regime (where pre-training reduces but cannot eliminate the AE-vs-LlamaMLP gap), now
  confirmed at SmolLM2-135M scale from random init.
- **The fine-tune findings transfer at a coarse level.** Both regimes agree that AE-MLP imposes a
  measurable LM-loss cost the model cannot fully recover from on its own. They do not yet agree on
  whether the cost is bounded by latent-dim, by initialization, or by the absence of a teacher
  signal — that disambiguation is the work of §4 and §5.
- **`ae_mlp` is Pareto-superior to `trendwavelet_db3_32` at matched tokens.** AE-MLP delivers more
  compression (48% vs 17%) at lower loss (3.558 vs 3.818 at 1.0B). This is the strongest
  evidence we have that the AE block is the better building block of the two studied so far for
  from-scratch SmolLM2 training.

### 3.5 In-flight 512-latent variants

Three variants probe the latent-dim and architectural extensions of AE-MLP:

| Variant | Config | Status |
|---|---|---|
| `ae_mlp_512` | pure AE-MLP at ld=512 (rest unchanged from `ae_mlp`) | **defined in YAML, not launched** — no eval/run/trainedmodel directories |
| `ae_basis_latent_db3_64_tiered_512_silu` | AE bottleneck with frozen wavelet basis at ld=512 | run dir exists (May 5), no eval JSONs |
| `tw_noln_ae_mlp_512_db3_128_tiered_fr_silu` | TW attention (no postLN) + AE-MLP at ld=512 | actively training, ~600M tokens, val PPL 85.0 at 600M (vs ae_mlp's 47.67 at 500M — but this bundles TW attention so it is not a clean ld=512 ablation) |

**The clean ld=256→512 capacity test (`ae_mlp_512`) has not yet been run.** This is the single
highest-information-value cheap experiment available; see §5.

## 4. What is not known

The following are open questions ranked by how much they would change the AE-MLP story:

1. **Capacity vs distillation.** Is the +0.28-nat plateau a capacity floor (more latent fixes it)
   or a distillation regime (only a teacher signal closes it)? Two cheap tests disambiguate:
   `ae_mlp_512` for capacity, KD-from-baseline for distillation. Neither has been run.
2. **Whether the fine-tune init hierarchy transfers.** Fine-tune AE-MLP uses `--ae-init pretrained`
   by default — least-squares projection from a pretrained LlamaMLP. From-scratch AE-MLP cannot
   use this (no pretrained weights to project from), and currently uses random init. We have not
   tested whether *any* of the warm-start strategies translate — e.g., extracting activations
   from a partially-trained baseline at, say, 250M tokens and running an AE Phase-2 against them
   to seed the from-scratch AE-MLP.
3. **Whether the fc2/fc3 SVD rule transfers to from-scratch.** Established at Llama-3.2-1B
   fine-tune; not tested at SmolLM2-135M from-scratch. Could plausibly close some of the +0.28-nat
   gap.
4. **Whether the 10-epoch pre-training cap holds on diverse data.** The ceiling was established
   on WikiText-2; FineWeb / SlimPajama support exists in the code but has not been swept beyond
   epoch 10. Untested: whether ld=256 AE-pretrained on FineWeb for 20 or 40 epochs further
   reduces downstream PPL.
5. **Whether the Phase-2 MSE objective is the right loss.** Plain per-layer MSE on cached
   activations is the only loss tested. Plausible alternatives: magnitude-weighted MSE,
   cosine-similarity loss, joint multi-layer MSE that lets layers compensate for each other's
   reconstruction error. None measured.
6. **Wave-pipeline `--ae-teacher` chaining vs vanilla teacher.** The wave pipeline auto-wires the
   prior wave's checkpoint as the AE pre-training teacher (so new layers see activations *in
   context* of already-compressed layers). No controlled A/B vs vanilla-Llama-as-teacher exists,
   so the value of this auto-wiring is asserted but not measured.
7. **Whether `_lg` matters for from-scratch.** `_lg` was tested in the fine-tune regime only.
   In from-scratch, the random-init path has no warm-start to recover from, so the prediction
   from §2.5 is "no effect" — but this prediction has not been verified.

## 5. Recommended next experiments

Ranked by information value vs cost:

1. **Run `ae_mlp_512`.** The cleanest single test of "is AE-MLP capacity-bound at ld=256?" The
   variant is fully defined in the YAML at line 556; it just needs to be launched under the
   parallel-launcher pipeline. If most of the +0.28-nat gap closes, the AE story is "tune the
   latent dim" and the recommendation simplifies. If not, distillation becomes the main lever.
2. **Port `--kd-alpha` into `pretrain_smol_replacement.py` and run `ae_mlp_kd`.**
   `kd-teacher = baseline checkpoint at 3.0B tokens`; alpha=0.5, temperature=2.0 (the fine-tune
   defaults). Tests the distillation interpretation of the plateau directly. The infrastructure
   exists in `finetune.py`; the port is mechanical.
3. **Apply the fc2/fc3 SVD rule to from-scratch AE-MLP.** Single-variant change, no new
   architecture; tests whether a fine-tune rule transfers across regimes. Could be combined with
   #1 (i.e., `ae_mlp_512` *with* the SVD fc2/fc3 rule applied).
4. **AE warm-start for from-scratch.** Extract activations from a partially-trained baseline at
   250M tokens, run the existing fine-tune AE Phase-2 loop against them to seed the AE bottleneck,
   then continue the standard from-scratch run. Tests whether seeding the AE block reduces the
   plateau without requiring a fully-trained teacher.
5. **MSE-objective ablation in fine-tune Phase 2.** Cheap layer-15-only sweep over MSE / cosine /
   magnitude-weighted MSE / joint-multi-layer MSE. Promote the winner to a wave run if the effect
   size justifies it.
6. **Diverse-data epoch-ceiling sweep.** `ae_pretrain_epochs ∈ {10, 20, 40}` on FineWeb at ld=256,
   layer 15 only. Confirms or extends the WikiText-2-derived 10-epoch rule.

## 6. Provenance

**Memory:**
- project_ae_pretraining_limits.md (omitted)
- project_ae_init_hierarchy.md (omitted)
- project_fc2fc3_strategy_findings.md (omitted)
- project_lg_gate_findings.md (omitted)
- project_kd_implementation.md (omitted)

**Code:**
- [pellm/pe_layers.py](../../pellm/pe_layers.py) — `PEBottleneckMLP`, `PEBottleneckMLPLG`
- [scripts/finetune.py:2634](../../scripts/finetune.py#L2634) — Phase 1 / Phase 2 branch (gates on `ae_pretrain_epochs > 0` and `pe_mlp_mode in _RECON_PE_MLP_MODES`)
- [scripts/finetune.py:1836](../../scripts/finetune.py#L1836) — `CachedActivationDataset` (per-layer driver for Phase 2)
- [scripts/finetune.py:58](../../scripts/finetune.py#L58) — `_RECON_PE_MLP_MODES = ("ae", "ae_lg", "vae", "vae_lg")`
- [scripts/pretrain_smol_replacement.py](../../scripts/pretrain_smol_replacement.py) — from-scratch path

**Configs (fine-tune sweeps that established §2):**
- [scripts/experiments/ae_init_methods_layer15_sweep.yaml](../../scripts/experiments/ae_init_methods_layer15_sweep.yaml)
- [scripts/experiments/ae_vs_aelg_layer15_sweep.yaml](../../scripts/experiments/ae_vs_aelg_layer15_sweep.yaml)
- [scripts/experiments/ae_init_fc2fc3_strategy.yaml](../../scripts/experiments/ae_init_fc2fc3_strategy.yaml)
- [scripts/experiments/ae_dataset_comparison.yaml](../../scripts/experiments/ae_dataset_comparison.yaml)
- [scripts/experiments/ae_pretrain_loss_sweep.yaml](../../scripts/experiments/ae_pretrain_loss_sweep.yaml),
  [ae_pretrain_loss_fineweb_resample.yaml](../../scripts/experiments/ae_pretrain_loss_fineweb_resample.yaml),
  [ae_pretrain_loss_mixed_dataset.yaml](../../scripts/experiments/ae_pretrain_loss_mixed_dataset.yaml)

**Configs (from-scratch run, §3):**
- [scripts/experiments/smol_replacement_paper.yaml](../../scripts/experiments/smol_replacement_paper.yaml)
  — `baseline`, `ae_mlp`, `ae_mlp_512` (line 556, not launched), `ae_basis_latent_db3_64_tiered_512_silu`
  (line 250), `tw_noln_ae_mlp_512_db3_128_tiered_fr_silu` (line 704, in flight)

**Result CSVs / JSONs:**
- Fine-tune sweeps: `scripts/experiments/results/{ae_init_methods_layer15_sweep,ae_vs_aelg_layer15_sweep,ae_init_fc2fc3_strategy,ae_dataset_comparison,ae_pretrain_loss_*}/logs/*.{csv,json}`
- From-scratch evals: `<pellm_data_root>/evals/smol_replacement_paper/{baseline,ae_mlp,ae_tw_db3_32,tw_noln_ae_mlp_512_db3_128_tiered_fr_silu,...}/eval_*.json`
- From-scratch manifests: `<pellm_data_root>/trainedmodels/smol_replacement_paper/{baseline,ae_mlp}/training_manifest.json`
- Downstream benchmarks (lm-eval): under `evals/smol_replacement_paper/{baseline,ae_mlp,...}/benchmarks/.../results_*.json`

**Companion analyses:**
- [analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md](analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md) (2026-05-07, downstream benchmarks)
- [smol_replacement_paper_inflight_analysis_2026-05-03.md](../../scripts/experiments/results/smol_replacement_paper_inflight_analysis_2026-05-03.md) (2026-05-03, loss trajectories)
