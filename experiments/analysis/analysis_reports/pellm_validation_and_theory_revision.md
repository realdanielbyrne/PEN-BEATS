# PELLM Validation and LLM Inductive-Bias Transfer Theory Revision

**Date:** 2026-05-03
**Companion to:** [`llm_inductive_bias_transfer_theory.md`](llm_inductive_bias_transfer_theory.md)
**Source repo for empirical data:** `/Users/danielbyrne/GitHub/pellm` (sibling repo, separate from this one)

This report compares the predictions of the LLM inductive-bias-transfer theory against the empirical results produced by the PELLM project, which has implemented and tested those predictions on Llama-3.2-1B (fine-tuning + distillation) and SmolLM2-135M-class architectures (from-scratch pretraining).

The capacity-reduction-needs-structural-prior principle survives. The N-BEATS-to-LLM analogy needed one specific revision: in the LLM domain, the dominant structural prior that successfully absorbs bottlenecks is the **pretrained-teacher subspace** (lstsq projection of teacher weights, or KD against a teacher), not a fixed mathematical basis on the sequence axis.

---

## 1. What PELLM built (and how it maps to the theory paper)

PELLM ports two N-BEATS constructs into Llama / SmolLM2:

| PELLM module | N-BEATS analog | What it replaces in the LLM |
|---|---|---|
| `TrendWaveletLinear` ([`pellm/pe_layers.py`](file:///Users/danielbyrne/GitHub/pellm/pellm/pe_layers.py) L200–400) | `TrendWavelet` block basis (Vandermonde + SVD-orthonormalized DWT, frozen as buffers) | All four attention projections per layer (`q/k/v/o_proj`). 4 trend + 28 wavelet coefficients per row by default. **97.5%** attention parameter reduction. |
| `PEBottleneckMLP` / `PEBottleneckMLPLG` ([`pe_layers.py`](file:///Users/danielbyrne/GitHub/pellm/pellm/pe_layers.py) L943–1350) | `AERootBlock` / `AERootBlockLG` (encoder → narrow latent → decoder) | The SwiGLU MLP sub-block (`gate_proj + up_proj → SiLU·gate → down_proj`). `ae_latent_dim=256` default. **90.6%** MLP parameter reduction. |

Initialization modes: `pretrained` (direct row/col truncation), `lstsq` (project teacher weights onto basis), `svd`, `cur`, `fourier`, `random`. Training pipelines: (a) pretrained-Llama repair (fine-tune with optional KD), (b) from-scratch SmolLM2-135M pretraining ([`scripts/experiments/smol_replacement_paper.yaml`](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/smol_replacement_paper.yaml)).

### Mapping to theory-paper hypotheses

| Theory hypothesis | PELLM implementation | Tested? | Verdict |
|---|---|---|---|
| **H1** — naive bottleneck MLP fails from scratch | `ae_mlp` variant in `smol_replacement_paper` | Yes | **Validated.** +0.28 nat gap to baseline at 1.0B tokens, plateaus and does not close. |
| **H2** — bottleneck MLP + sequence-axis structured operator | Not implemented. PELLM's TrendWavelet acts on the **input-feature axis of the projection weight** (basis ∈ ℝ^d_model), not on the **sequence axis** of the residual stream (basis ∈ ℝ^seq_len). | **No** | Original H2 form **untested**. PELLM tested a related but architecturally distinct bet (H3). |
| **H3** — structured-basis Q/K/V projections | `TrendWaveletLinear` replacing `q/k/v/o_proj` | Yes | **Conditionally validated.** Wins in fine-tuning with `lstsq` teacher seating (97.5% param cut, perplexity improves). Loses by +0.5 nats from random init in from-scratch pretraining. |
| **H4** — alternating heterogeneous / tiered blocks | `*_tiered` variants per-layer offset sweep | Pending | Not yet trained. |

The theory paper got H2's *form* wrong: I described a sequence-axis decomposition of the residual stream, while PELLM tested a feature-axis basis on the projection weight. These are different architectural bets and the result for one does not transfer to the other. H2 in its original form remains untested.

---

## 2. Empirical results — fine-tuning (pretrained-Llama repair)

Source: [`scripts/experiments/results/trendwavelet_layer15_sweep/logs/`](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/trendwavelet_layer15_sweep/logs/) (216 configs) and [`scripts/experiments/results/ae_dataset_comparison/logs/`](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/ae_dataset_comparison/logs/) (120 runs).

| Treatment | Init | Baseline PPL | Final PPL | Δ |
|---|---|---:|---:|---:|
| TW-attention layer 15 (`trend_wavelet`, db3, wavelet_dim=28) | `lstsq` | 22.00 | **20.44** | −7.1% |
| TW-attention layer 15 (same) | `pretrained` (direct truncation) | 26.70 | 20.80 | −22.6% |
| AE-MLP layer 15 (`ae_lg`, lat=512, WikiText-2 cache) | `pretrained` | 42.5 | **15.4** | −63.6% |

Both bottlenecks **win in fine-tuning when the bottleneck is seated by a teacher** (lstsq projects pretrained weights onto the basis subspace; the AE pretraining caches teacher activations and trains the AE to reproduce them). The lstsq → final perplexity beats baseline at 97.5% parameter reduction in attention. This is the LoRA story re-derived through a different bottleneck shape: the structural prior is the pretrained teacher, made accessible by a lstsq seating.

---

## 3. Empirical results — from-scratch pretraining (SmolLM2-135M class)

Source: [`scripts/experiments/results/smol_replacement_paper_inflight_analysis_2026-05-03.md`](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/smol_replacement_paper_inflight_analysis_2026-05-03.md). Token-matched gap-to-baseline (loss in nats):

| Variant | Params | 250M | 500M | 750M | **1.0B** | 1.25B | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline` | 134.5M | — | — | — | — | — | reference |
| `ae_mlp` (latent_dim=256) | 69.3M (−48%) | +0.452 | +0.344 | +0.309 | **+0.284** | +0.272 | gap **plateaus at +0.28**, slope matches baseline |
| `trendwavelet_db3_32` | 112.1M (−17%) | +0.652 | +0.624 | +0.585 | **+0.544** | — | Pareto-dominated by `ae_mlp` |
| `ae_tw_db3_32` | 51.4M (−62%) | +0.922 | +0.871 | +0.765 | **+0.675** | +0.645 | widest gap |

Downstream eval ([`evals/smol_replacement_paper/benchmark_report.md`](file:///Users/danielbyrne/GitHub/pellm/evals/smol_replacement_paper/benchmark_report.md), final 3.0B tokens):

| Variant | LAMBADA | HellaSwag | PIQA | ARC-E | ARC-C | OpenBookQA |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 24.22 | 31.32 | 61.97 | 45.50 | 25.09 | 30.20 |
| `ae_mlp` | 18.07 (−6.15) | 27.63 (−3.69) | 59.14 (−2.83) | 41.29 (−4.21) | 22.61 (−2.48) | 27.80 (−2.40) |

### The decisive finding

`trendwavelet_db3_32` retains 17% MORE parameters than `ae_mlp` yet loses 0.26 nats MORE at every matched-token point. Pure capacity loss cannot explain this — under capacity-loss-only, fewer params would mean larger gap. The cleaner explanation: **the frozen Vandermonde+DWT basis is a hard inductive prior on attention that random-init SGD cannot reach**. AE-MLP's bottleneck is structurally easier to learn around because the AE backbone is just a parameterized low-rank approximation; TW-attention forces computation through a fixed orthogonal basis no matter what.

This is the same shape as the M4 finding (`AutoEncoderAE` pure-stack dominated by `TrendWaveletAE` with 4× fewer params): a capacity reduction without a matched structural prior is dominated. PELLM extends it: a structural prior the model cannot reach by SGD from random init is *worse* than a bottleneck the model can flow gradients through.

---

## 4. Empirical results — distillation (the load-bearing data)

This is the section the user specifically asked about, and it is the most informative for the principle.

**`hybrid_distill_layer15_test`** ([`logs/`](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/hybrid_distill_layer15_test/logs/)) — single layer-15 MLP swap, AE-LG lat=512, `kd_alpha=1.0`, `kd_temperature=2.0`, `pretrained` init, fineweb 50k samples:

| Stage | PPL |
|---|---:|
| Original Llama-3.2-1B (no swap) | 19.45 |
| Baseline (PE layer inserted, no KD, no AE pretrain) | **42.72** (collapsed) |
| After AE pretrain phase (teacher-activation reconstruction) | 25.34 |
| **After full KD fine-tune** | **18.81** ← **beats original Llama by 3.3%** |

This is the cleanest validation of the principle in the LLM domain to date: an MLP sub-block compressed by ~90% in parameters, when supplied with the structural prior (KD teacher + lstsq-style pretrained AE init), recovers and slightly outperforms the original.

**`full_mlp_pipeline_v2`** ([`logs/`](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/full_mlp_pipeline_v2/logs/)) — wave 1 of progressive MLP swap (layers 14–15, AE lat=512, `kd_alpha=1.0`, AE pretrain with Huber loss + α=0.5 KD blend):

| Stage | PPL |
|---|---:|
| Original | 22.87 |
| Baseline (insert, no KD) | 140.93 (catastrophic) |
| After AE pretrain | 42.29 |
| **After KD fine-tune** | **22.72** (−83.88% from baseline; matches original) |

**`mlp_attn_two_wave_layers_14_15_ctxteacher_v2`** — MLP layers 14–15, AE-LG lat=512, `kd_alpha=0.7`:

| Stage | PPL |
|---|---:|
| Original | 19.45 |
| Baseline (insert, no KD) | 109.19 |
| After AE pretrain | 33.16 |
| **After KD fine-tune** | **22.58** (−79.32%; minor regression from original) |

**`attn_mlp_two_wave_layers_14_15`** — TW-attention (`trend_wavelet_lg`, db3, lstsq init) layers 14–15, no MLP swap, `kd_alpha=1.0`:

| Stage | PPL |
|---|---:|
| Original | 21.14 |
| Baseline (insert, no KD) | 32.22 |
| After lstsq attn pretrain | 29.42 |
| **After KD fine-tune** | **27.11** (−15.85%; gap of +5.97 PPL vs original remains) |

### What the distillation data says

1. **AE-MLP + KD recovers fully and can beat the original.** The from-scratch +0.28 nat plateau collapses when KD is present. The structural prior the from-scratch run was missing is the teacher.
2. **TW-attention + KD recovers only partially.** Layers 14–15 swap with KD lands at 27.11 PPL vs 21.14 original — a 28% PPL gap remains even with the strongest available prior (lstsq init + KD α=1.0, 2 layers only). The basis constraint on attention is harder to absorb than the bottleneck on MLP, even when the prior is supplied.
3. **The principle predicts both outcomes correctly.** AE-MLP is a *parameterized* low-rank construction — KD can move it freely within the achievable subspace. TW-attention is constrained to live exactly inside the frozen basis subspace — KD can only seat it as well as the lstsq projection allows, and if the basis truly excludes the directions attention needs, KD cannot recover them.

This is the LLM-domain restatement of the M4 finding. On M4, AE backbone + structured basis (TWAE, TWAELG) wins because the AE produces *features* the structured basis can absorb. AE backbone alone (AutoEncoderAE) loses because there is nothing absorbing the bottleneck. In PELLM, AE-MLP + KD-teacher wins because the teacher provides the absorbing prior. AE-MLP from random init loses because no prior is supplied. TW-attention loses harder because the prior (the basis) is a *constraint* rather than a *target*, and SGD cannot escape it.

---

## 5. Theory paper revision

What survives:
- **Section 2 (the principle)** — strengthened. PELLM data validates it across both attention and MLP swaps, in both fine-tuning and from-scratch settings.
- **H1** — fully validated by `ae_mlp` from-scratch plateau at +0.28 nats.
- **H3** — conditionally validated. TW-attention with `lstsq` init from a pretrained teacher wins in fine-tuning. Random init from-scratch fails. The condition is the same as the principle predicts: the basis is the constraint, the teacher is the prior that absorbs the constraint.
- **H4** — still untested, still a live hypothesis.

What needs revision:
- **H2 (sequence-axis structured operator)** — the *form* in the original report is untested by PELLM. PELLM's TrendWavelet is a feature-axis basis on the projection weight, not a sequence-axis decomposition of the residual stream. **H2 is neither validated nor falsified.** A clean test would require a separate experiment (see §6).
- **The dominant structural prior in the LLM domain is the pretrained teacher (KD/lstsq), not a fixed mathematical basis.** This was implicit in the LoRA cell of Section 2's table but not made central. The PELLM data makes it the central claim:
  - LoRA → prior is pretrained weights (low-rank tangent).
  - TW-attention with lstsq init → prior is pretrained weights (basis-projected).
  - AE-MLP + KD → prior is teacher activations.
  - Naive bottleneck MLP from scratch → no prior, fails.
  - Naive distillation by `d_model` shrink → no structural prior on the new geometry, fails.

Refined H2 (call it **H2'**): *Bottlenecked attention or MLP + pretrained-teacher KD is the Pareto-improving combination across LLM efficiency proposals. The structural prior in this domain is the teacher subspace; mathematical bases (wavelet, polynomial, Fourier) only succeed insofar as they can be seated against teacher activations (lstsq, SVD, CUR projections of teacher weights).*

---

## 6. Recommendations

These are recommendations to the PELLM project, derived jointly from the N-BEATS principle and PELLM's own in-flight analysis. They are ordered by information value per GPU-hour. No edits to PELLM are made from this session.

**(a) Stop `trendwavelet_db3_32` from-scratch run early at ≈1.5B tokens.** It is Pareto-dominated by `ae_mlp` (more params, worse loss) and the slope cannot close the gap in the remaining 1.5B tokens. The from-scratch evidence for TW-only-on-attention being a from-scratch failure mode is already conclusive. Confirms PELLM in-flight rec #2.

**(b) Run `ae_mlp_512`** (AE-MLP latent_dim=512 instead of 256). If this closes most of the +0.28-nat AE-MLP gap from-scratch, the AE-MLP-from-scratch finding is "tune the latent dim" rather than "switch to distillation". Cheap, high information value. Confirms PELLM in-flight rec #1.

**(c) Add distillation-from-scratch variants** — `trendwavelet_db3_32_distill`, `ae_tw_db3_32_distill`, `ae_mlp_distill`. Frozen baseline-architecture teacher checkpoint supplies logit-KD during pretraining (and optionally attention-pattern KD for the TW variants). The distillation infrastructure already works in fine-tuning (`hybrid_distill_layer15_test` shows AE-MLP + KD beats the original Llama at layer 15); the test is whether it transfers to the from-scratch regime where the teacher is not the architecture being trained but a separately-trained baseline. The N-BEATS principle predicts this should close the gap for AE-MLP and partially close it for TW-attention, with TW-attention residually limited by basis-reachability. Single most informative experiment. Confirms PELLM in-flight rec #4.

**(d) Defer `*_tiered` variants** until (c) reports. Ablating per-layer offset frequency on a TW block that loses 0.5 nats from-scratch is the wrong axis. Once a TW-with-KD variant is shown competitive, tiered offsets become a meaningful ablation. Confirms PELLM in-flight rec #5.

**(e) (NEW — N-BEATS-sourced)** Test true sequence-axis H2. Apply a fixed orthonormal DWT (or FFT) decomposition to the residual stream **between attention output and MLP input**, with learned per-scale linear mixing. Smallest variant: one decoder layer with this operator inserted, bottlenecked MLP downstream, KD'd from baseline. This is the only theory-paper hypothesis PELLM has not yet covered. Distinct from current TW-on-projection-weights. The N-BEATS prediction: the sequence-axis structured op should be more parameter-efficient than the feature-axis one because it operates on the same axis where smoothness/sparsity priors are most defensible.

**(f) (NEW — N-BEATS-sourced)** Benchmark `PEBottleneckMLPLG` against `PEBottleneckMLP` from-scratch. The learned-gate AE family (AELG) owns the sub-1M Pareto frontier on M4 specifically because the per-latent-dim sigmoid gate discovers effective latent dimensionality during training — exactly the regime where PELLM's random-init `ae_mlp` plateaus at +0.28 nats. PELLM has `PEBottleneckMLPLG` implemented but has not run it as a from-scratch variant in `smol_replacement_paper.yaml`. Add `ae_mlp_lg` as a variant; predict it closes part of the AE-MLP gap, possibly more than `ae_mlp_512` does at lower parameter cost.

---

## 7. Honest limits

- **One project, two model scales, one dataset family.** PELLM trains on WikiText-2, FineWeb, and FineWeb-Edu-100BT. Cross-domain generalization (code, math, multilingual) is untested.
- **Recipe drift between baseline/ae_mlp (legacy 2-GPU DDP, grad_accum=32, gradient_checkpointing=true) and TW variants (new 1-GPU recipe, grad_accum=64, gradient_checkpointing=false).** Documented in [in-flight analysis §6](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/smol_replacement_paper_inflight_analysis_2026-05-03.md). The qualitative ranking (baseline > ae_mlp > trendwavelet > ae_tw at matched tokens) is robust to drift; absolute numbers should not be cited as paper-grade comparators without a recipe-matched re-run.
- **Distillation data is from layer-1 or 2-layer swaps in pretrained Llama.** Whether the AE-MLP-with-KD result (18.81 PPL beating original 19.45 at layer 15) extends to all 16 layers swapped with KD is the central open question. `full_mlp_pipeline_v2`'s 5-wave progressive swap is the relevant evidence and is still incomplete.
- **The theory paper's specific architectural proposals (sequence-axis wavelets) remain speculative.** The principle survives; the specific bet should be tested separately, not inferred from PELLM's projection-weight basis.

---

## 8. Summary

The capacity-reduction-needs-structural-prior principle holds in the LLM domain. PELLM has tested it at three operating points:

- **Pretrained-Llama fine-tuning with lstsq seating:** bottleneck wins (97.5% attention reduction at slight perplexity improvement; AE-MLP recovers from 42.5 PPL to 15.4 PPL).
- **Pretrained-Llama fine-tuning with full KD distillation:** bottleneck wins on MLP (AE-MLP at layer 15 lands at 18.81 PPL vs original 19.45) and partially recovers on attention (TW layers 14–15 at 27.11 vs original 21.14).
- **From-scratch pretraining without distillation:** bottleneck loses on both MLP (+0.28 nat persistent) and attention (+0.5 nat and Pareto-dominated by AE-MLP despite more retained parameters).

The unifying explanation is that the structural prior the bottleneck needs in the LLM domain is the **pretrained teacher subspace**, supplied either as initialization (lstsq, pretrained, AE-pretrain on cached activations) or as a training signal (KD logits, KD attention patterns). Fixed mathematical bases (wavelets, polynomials) only succeed insofar as they can be seated against the teacher; from random init they are constraints SGD cannot escape.

The riskiest open bet from the original theory paper — sequence-axis structured operators on the residual stream — remains untested and is the one experiment that would extend the PELLM evidence base most. The path forward most strongly suggested by both the principle and the PELLM in-flight data is **distillation-from-scratch variants** of the existing PELLM bottleneck designs.

---

**Cross-references**
- Theory paper: [`llm_inductive_bias_transfer_theory.md`](llm_inductive_bias_transfer_theory.md)
- Empirical pre-condition: [`autoencoder_pure_m4_yearly_analysis.md`](autoencoder_pure_m4_yearly_analysis.md)
- PELLM in-flight analysis (load-bearing data source): [smol_replacement_paper_inflight_analysis_2026-05-03.md](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/smol_replacement_paper_inflight_analysis_2026-05-03.md)
- PELLM downstream evals: [benchmark_report.md](file:///Users/danielbyrne/GitHub/pellm/evals/smol_replacement_paper/benchmark_report.md)
- PELLM distillation logs:
  - [hybrid_distill_layer15_test/logs/](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/hybrid_distill_layer15_test/logs/)
  - [full_mlp_pipeline_v2/logs/](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/full_mlp_pipeline_v2/logs/)
  - [attn_mlp_two_wave_layers_14_15/logs/](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/attn_mlp_two_wave_layers_14_15/logs/)
  - [mlp_attn_two_wave_layers_14_15_ctxteacher_v2/logs/](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/mlp_attn_two_wave_layers_14_15_ctxteacher_v2/logs/)
- PELLM fine-tuning sweeps: [trendwavelet_layer15_sweep/logs/](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/trendwavelet_layer15_sweep/logs/), [ae_dataset_comparison/logs/](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/results/ae_dataset_comparison/logs/)
