# H2 Sequence-Axis Experiment Plan

**Date:** 2026-05-03
**Theory basis:** [`llm_inductive_bias_transfer_theory.md`](llm_inductive_bias_transfer_theory.md)
**Gap identified by:** [`pellm_validation_and_theory_revision.md`](pellm_validation_and_theory_revision.md) §5 and §6(e)
**Implementation target:** `/Users/danielbyrne/GitHub/pellm`

---

## 1. Hypothesis Statement

H2 in its falsifiable sequence-axis form:

> A transformer decoder layer with (a) a bottlenecked MLP (`down(d→r) → act → up(r→d)`, r ≪ d) and (b) a fixed structured operator applied along the **sequence axis** of the post-attention residual stream — such that each output position t receives a causally-masked wavelet projection of past positions 0..t — should, from random initialization and without distillation, close a meaningful fraction of the perplexity gap between the naive AE-MLP bottleneck and the full-MLP baseline at matched token budgets.

This is distinct from PELLM's existing `TrendWaveletLinear`, which applies a basis ∈ ℝ^d_model to the **feature axis** of the projection weight matrix. H2 requires the basis to act on ℝ^T, indexing **sequence positions**. PELLM has tested neither this architecture nor this axis. The hypothesis is neither validated nor falsified by existing PELLM runs.

Falsification: Variant C's gap to baseline is ≥ Variant B's gap at 1.0B tokens. Support: Variant C's gap ≤ 50% of Variant B's gap at 1.0B tokens (≤ +0.14 nats, given B's plateau at +0.28 nats from [`pellm_validation_and_theory_revision.md`](pellm_validation_and_theory_revision.md)).

---

## 2. Architecture Design

### Primary operator: Causal Haar DWT per-channel mixing

**Why Haar.** T = 2048 is a power of two, so the Haar DWT decomposes exactly with no boundary artifacts. Haar has the shortest possible filter (2 taps), which minimizes the range over which causal masking destroys basis orthonormality. The existing `pellm/basis.py:build_wavelet_basis` already constructs SVD-orthonormalized DWT synthesis matrices via `pywt.wavedec` + impulse responses + SVD; the same code path applies to a T×T sequence-axis basis at `target_length=2048`. However, the implementation below is more efficient.

**Operator mechanics.** Apply a cascade of causal 1D convolutions with Haar filters along the sequence axis of `x ∈ ℝ^(B, T, d)`:

```
For each level l in {1, ..., n_levels}:
    h_l = causal_conv1d(x, haar_filter_l, stride=1)    # shape: (B, T, d), causal
    out += h_l * w_l                                    # w_l ∈ ℝ^d, learned per-channel scale
Return out  # added to residual stream
```

Causal 1D convolution is implemented via `F.conv1d(x.transpose(1,2), weight, padding=filter_len-1)[..., :T].transpose(1,2)` — left-pad by `filter_len-1` ensures position t reads only from positions 0..t. Level-l Haar filter has length `2^l`. At n_levels=5, filter lengths are 2, 4, 8, 16, 32. Learned parameters: `n_levels × d = 5 × 576 = 2,880` per layer; across all 30 layers = **86,400 total** (≈0.09M). This is negligible compared to ae_mlp's 14.4M.

**Variable context length.** Causal convolutions are length-agnostic up to `max_position_embeddings`. No zero-padding or reshape needed.

**Causal mask verification.** Unit test: for random `t_cut`, set `x[:, t_cut+1:, :] = 0`. Verify `SequenceAxisWaveletOperator.forward(x)[:, :t_cut+1, :]` matches output of the unmasked `x` on those positions. If not, the implementation leaks future information. This test must pass before any training run.

**Fallback candidates.** (a) Causal FFT mixing: apply rfft along T, mask to lowest `n_freq` bins, irfft — learnable complex scale per bin per channel. (b) S4-style causal state-space convolution. Both are more complex to implement than Haar causal conv; use only if Haar fails pilot kill criteria.

**Where in the block.** Insert between `post_attention_layernorm` and `self.mlp` in `PELlamaDecoderLayer.forward()` (at [`modeling_pe_llama.py:324`](file:///Users/danielbyrne/GitHub/pellm/pellm/modeling_pe_llama.py)). The operator receives layer-normed activations (unit variance), which keeps the Haar projection well-conditioned. Output is added residually: `hidden_states = hidden_states + seq_axis_op(hidden_states)` before the MLP call.

---

## 3. Where to Wire into Pellm

**New file:** `/Users/danielbyrne/GitHub/pellm/pellm/seq_axis_operator.py`

Implements `SequenceAxisWaveletOperator(nn.Module)`. Constructor takes `(hidden_size, n_levels=5, wavelet_type='haar')`. Registers Haar analysis filters as frozen buffers. Holds `w_scale ∈ ℝ^(n_levels, hidden_size)` as the sole learned parameter, initialized to 0 (additive identity at init). Forward: cascade of causal conv1d per level, scaled by `w_scale[l]`, summed.

**`/Users/danielbyrne/GitHub/pellm/pellm/configuration_pe_llama.py`:** Add to `PELlamaConfig.__init__`: `pe_seq_axis_mode: str = "none"` and `seq_axis_n_levels: int = 5`. These default to the no-op case (backward compatible with all existing runs).

**`/Users/danielbyrne/GitHub/pellm/pellm/modeling_pe_llama.py`:** In `PELlamaDecoderLayer.__init__` (around line 278), after MLP construction: instantiate `self.seq_axis_op` when `config.pe_seq_axis_mode == "causal_haar"`. In `forward()` at line 324, after the layernorm call and before the MLP call, insert: `if self.seq_axis_op is not None: hidden_states = hidden_states + self.seq_axis_op(hidden_states)`.

**`/Users/danielbyrne/GitHub/pellm/scripts/pretrain_smol_replacement.py`:** Extend `VariantConfig` dataclass (line 43) with `pe_seq_axis_mode: str = "none"` and `seq_axis_n_levels: int = 5`. Forward both into `build_model()` via `PELlamaConfig(...)` at line 272.

**`/Users/danielbyrne/GitHub/pellm/scripts/experiments/smol_replacement_paper.yaml`:** Add two variants:

```yaml
  seq_axis_haar_ae_mlp:        # Variant C — H2 test
    pe_attn_mode: "standard"
    pe_mlp_mode: "ae"
    ae_latent_dim: 256
    pe_seq_axis_mode: "causal_haar"
    seq_axis_n_levels: 5

  seq_axis_haar_std_mlp:       # Variant D — operator-only ablation
    pe_attn_mode: "standard"
    pe_mlp_mode: "standard"
    pe_seq_axis_mode: "causal_haar"
    seq_axis_n_levels: 5
```

**Test file:** Add `tests/test_seq_axis_operator.py` with the causal masking unit test and a shape check. This is required infrastructure, not optional.

---

## 4. Experimental Matrix

| Label | Description | Params | Status |
|---|---|---|---|
| `baseline` (A) | Full SwiGLU MLP, no operator | 134.5M | Already trained, 3B tokens |
| `ae_mlp` (B) | AE-MLP lat=256, no operator | 69.3M | Already trained, 3B tokens |
| `seq_axis_haar_ae_mlp` (C) | AE-MLP lat=256 + causal Haar op | ~69.4M | NEW — pilot then full |
| `seq_axis_haar_std_mlp` (D) | Standard MLP + causal Haar op | ~134.6M | NEW — pilot then full |

Training recipe: identical to `baseline` and `ae_mlp` (FineWeb-Edu, lr=6e-4, warmup_ratio=0.03, seq_len=2048, micro_batch=2, grad_accum=64, seed=42, gradient_checkpointing=false). This preserves A/B integrity per CLAUDE.md.

Distillation is excluded. H2 claims the sequence-axis prior absorbs the bottleneck from random init, without a teacher. A KD variant of C is a separate follow-on experiment testing whether KD stacks with the sequence-axis prior (neither confirms nor falsifies H2 in its original form).

---

## 5. Success Criteria

All evaluated at matched token budgets. Variant B reference: gap-to-baseline +0.284 nats at 1.0B tokens, +0.28 plateau sustained to 3B.

| Verdict | Criterion at 1.0B tokens |
|---|---|
| H2 supported | Variant C gap ≤ +0.14 nats (closes ≥50% of B's gap) |
| H2 partially supported | +0.14 < Variant C gap < +0.28 (operator helps; not dominant prior) |
| H2 falsified | Variant C gap ≥ +0.28 nats (no improvement over B) |
| Operator destructive | Variant D gap > +0.10 nats vs A |

If Variant D is destructive, Variant C's failure may be operator-induced rather than principle-falsifying. Redesign the causal conv before concluding H2 is false.

---

## 6. Pilot Kill Criteria

At 300M tokens:

- Variant C or D training loss > 6.0, or NaN gradients: kill immediately. Indicates numerical instability in the operator (check filter normalization).
- Variant C gap to A exceeds B's gap by more than +0.20 nats: kill. The operator is actively harmful; the 3B budget is not justified.
- Variant D shows sustained loss spikes (> 0.5 nat per step for > 50 consecutive steps): kill and redesign the operator before running Variant C.

Run C and D in parallel on separate GPUs via `run_smol_replacement_parallel.py`. Check pilot eval at 250M tokens before committing to the full run.

---

## 7. Risks and Mitigations

**(a) Causal mask leakage → false-positive H2 support.** If the causal conv padding is applied on the wrong side, future tokens corrupt past positions. The model learns faster because it can peek, making C appear better than it is. Mitigation: `tests/test_seq_axis_operator.py` causal masking unit test (non-negotiable; must pass before any training). The Haar filter length-2 case is simple enough to verify analytically: `F.conv1d(x, weight, padding=1)[:, :, :T]` with weight `[1, 1]` gives causal output at each position.

**(b) T=2048 is outside the regime where wavelet sparsity is informative.** N-BEATS sequences are 6–48 points; sparsity is empirically confirmed at those lengths. At T=2048, transformer hidden states may not be sparse in any wavelet basis — spectral energy is distributed across all scales because language is not a bandlimited physical signal. The N-BEATS-to-LLM analogy may fail at this sequence length. Mitigation: inspect `w_scale` after pilot training. Uniform scale magnitude across levels indicates no scale-selectivity (the prior is inoperative). Compare against FNet and SSM literature at T=2048; if H2 is weakly falsified, consider whether the locality (not sparsity) interpretation of the prior is more appropriate.

**(c) Throughput regression makes the comparison operationally unfair.** Causal conv1d over 5 levels at T=2048 adds `O(B × T × d × sum_l filter_len_l)` operations = `O(B × T × d × 62)` per layer. At B=2, T=2048, d=576: ~145M ops per layer × 30 = 4.36B ops. The ae_mlp MLP forward is ~180M ops per layer × 30 = 5.4B ops. So the operator roughly doubles the total MLP forward cost for Variant C. Mitigation: measure tokens/sec on one GPU before launching the full run. If throughput drops by > 30%, replace Haar with a FFT-based implementation (O(B × T × d × log T) instead).

---

## 8. Estimated Cost

Throughput: ~14,000 tok/sec per RTX 5090 (1-GPU recipe, gradient_checkpointing=false). Note: Variant C/D may be slightly slower due to conv overhead; estimate conservatively at 13,000 tok/sec.

| Phase | Variants | Tokens | Wall-clock | GPU-hours |
|---|---|---|---|---|
| Pilot | C + D (parallel, 2 GPUs) | 300M each | ~6.4 hrs | ~12.8 |
| Full run | C + D (parallel, 2 GPUs) | 3B each | ~64 hrs | ~128 |
| **Total** | | | | **~141 GPU-hours** |

Variants A and B are already trained. Total new cost: ~141 GPU-hours.

---

## 9. What This Experiment Does Not Test

**(a) H4 (alternating heterogeneous blocks).** The operator is applied uniformly to all 30 layers. A per-depth frequency sweep (low-frequency levels early, high-frequency levels late) is a follow-on ablation after H2 is resolved.

**(b) The refined H2' (bottleneck + pretrained teacher as structural prior).** PELLM data strongly implicates the teacher subspace as the dominant prior in the LLM domain. This experiment tests whether a fixed mathematical basis on the sequence axis changes the from-scratch verdict. If H2 is falsified from scratch, a KD variant of C is a cheap follow-on to test H2' on the sequence axis.

**(c) Whether the sequence-axis operator is useful in fine-tuning.** Fine-tuning with lstsq-initialized operator weights (projecting a pretrained model's inter-position mixing patterns onto the causal Haar basis) may yield a different verdict than from-scratch training, analogous to TW-attention's fine-tuning win vs from-scratch failure. Test separately; it is much cheaper.

**(d) Sequence lengths beyond 2048.** Longer sequences provide more wavelet scales and may make the sparsity prior more defensible. If H2 is weakly supported at T=2048, a T=4096 ablation is informative.

---

**Cross-references:**
- Theory paper: [`llm_inductive_bias_transfer_theory.md`](llm_inductive_bias_transfer_theory.md) §5 H2
- PELLM validation: [`pellm_validation_and_theory_revision.md`](pellm_validation_and_theory_revision.md) §5, §6(e)
- Existing basis construction (reuse): [pellm/basis.py](file:///Users/danielbyrne/GitHub/pellm/pellm/basis.py) `build_wavelet_basis`
- Decoder layer forward (insertion point): [pellm/modeling_pe_llama.py:295–328](file:///Users/danielbyrne/GitHub/pellm/pellm/modeling_pe_llama.py) `PELlamaDecoderLayer.forward`
- Variant config dataclass (extend): [scripts/pretrain_smol_replacement.py:43–53](file:///Users/danielbyrne/GitHub/pellm/scripts/pretrain_smol_replacement.py)
- Model builder (extend): [scripts/pretrain_smol_replacement.py:272–314](file:///Users/danielbyrne/GitHub/pellm/scripts/pretrain_smol_replacement.py) `build_model`
- Experiment config (add variants): [scripts/experiments/smol_replacement_paper.yaml](file:///Users/danielbyrne/GitHub/pellm/scripts/experiments/smol_replacement_paper.yaml)
