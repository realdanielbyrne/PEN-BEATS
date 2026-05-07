# SmolLM2-class Replacement Paper: 3-Way Benchmark Comparison

**Generated:** 2026-05-07
**Comparison:** vanilla `baseline` (3.0B tok) vs `ae_mlp` (3.0B tok) vs `tw_root_fc_db3_64_tiered_silu` (1.7B tok, partial). Evaluated zero-shot on lambada_openai, hellaswag, piqa, arc_easy, arc_challenge, winogrande, openbookqa via `lm-evaluation-harness` 0.4.11.

## Variants

| Variant | Mode | Training tokens | Status | Eval source |
|---|---|---|---|---|
| baseline | vanilla SmolLM2-class Llama (no replacement) | 3.0B | complete | results_2026-05-02T09-58-51 |
| ae_mlp | AE-bottleneck MLP (every layer) | 3.0B | complete | results_2026-05-02T10-00-42 |
| tw_root_fc_db3_64_tiered_silu | TrendWavelet root+fc, db3, wavelet_dim ~64, tiered, SiLU reduction | 1.7B (~57%) | partial-checkpoint, training killed | results_2026-05-07T02-38-41 |

## Per-task Metrics (preferred metric, % unless noted)

| Task (metric) | baseline | ae_mlp | tw_root_fc (1.7B) |
|---|---|---|---|
| lambada_openai (acc) | 24.22 ± 0.60 | 18.07 ± 0.54 | 15.18 ± 0.50 |
| lambada_openai (ppl) | 142.87 ± 6.85 | 431.21 ± 22.75 | 842.24 ± 46.56 |
| hellaswag (acc_norm) | 31.32 ± 0.46 | 27.63 ± 0.45 | 27.52 ± 0.45 |
| piqa (acc_norm) | 61.97 ± 1.13 | 59.14 ± 1.15 | 57.73 ± 1.15 |
| arc_easy (acc_norm) | 45.50 ± 1.02 | 41.29 ± 1.01 | 40.07 ± 1.01 |
| arc_challenge (acc_norm) | 25.09 ± 1.27 | 22.61 ± 1.22 | 23.63 ± 1.24 |
| winogrande (acc) | 51.30 ± 1.40 | 51.46 ± 1.40 | 52.33 ± 1.40 |
| openbookqa (acc_norm) | 30.20 ± 2.06 | 27.80 ± 2.01 | 27.60 ± 2.00 |

## Macro-average Accuracy (mean of 7 task accuracies, ppl excluded)

| Variant | Macro-avg |
|---|---|
| baseline | 38.51% |
| ae_mlp | 35.43% |
| tw_root_fc (1.7B) | 34.87% |
| tw_root_fc (1.8B follow-up) | 34.95% |

## Follow-up Benchmark at 1.8B Tokens

After the initial 1.7B benchmark, training resumed and a fresh `tokens_1800142848` checkpoint was evaluated under the same lm-eval harness/recipe. This is +100M tokens (+5.9%) past the 1.7B point — a small delta but useful as a direction-of-travel check.

| Task (preferred metric) | 1.7B | 1.8B | Δ (1.8 − 1.7) | 2σ band | Sig? |
| --- | --- | --- | --- | --- | --- |
| lambada_openai (acc) | 15.18 | 15.99 | +0.81 pp | 1.43 | no |
| lambada_openai (ppl) | 842.24 | 745.65 | **-96.59** | 124.18 | no (1.5σ) |
| hellaswag (acc_norm) | 27.52 | 27.68 | +0.16 pp | 1.27 | no |
| piqa (acc_norm) | 57.73 | 57.56 | -0.17 pp | 3.25 | no |
| arc_easy (acc_norm) | 40.07 | 39.94 | -0.13 pp | 2.86 | no |
| arc_challenge (acc_norm) | 23.63 | 23.55 | -0.08 pp | 3.51 | no |
| winogrande (acc) | 52.33 | 51.30 | -1.03 pp | 3.96 | no |
| openbookqa (acc_norm) | 27.60 | 28.60 | +1.00 pp | 5.69 | no |
| **Macro-avg** | 34.87% | 34.95% | +0.08 pp | — | — |

No per-task change crosses the 2σ threshold. The largest absolute moves — LAMBADA ppl (−97, ~1.5σ) and LAMBADA acc (+0.8 pp, ~1.1σ) — are directionally consistent with continued LM-loss decay but cannot be distinguished from noise on a single 100M-token delta. The MC tasks (HellaSwag, PIQA, ARC, WinoGrande, OpenBookQA) are essentially flat: the macro-avg moved by 0.08 pp, well inside noise. This is the expected pattern at this scale — MC accuracy on tasks that are already at or near random-baseline saturate slowly with extra tokens.

The validation-ppl trajectory was also updated: 28.06 (1.75B) → **27.84 (1.80B)**, a 0.79% drop over 50M tokens. This sits on the log-linear extrapolation curve from the earlier projection — the 21.5–23.5 ppl band at 3.0B remains the best estimate.

## Significance-flagged Deltas

Threshold: |Δ| > 2·√(σ_a² + σ_b²). SIG = significant under this rule.

| Task | Δ (tw − baseline) | sig? | Δ (tw − ae_mlp) | sig? |
|---|---|---|---|---|
| lambada acc | -9.04 | SIG | -2.89 | SIG |
| lambada ppl | +699.37 | SIG | +411.04 | SIG |
| hellaswag acc_norm | -3.79 | SIG | -0.11 | ns |
| piqa acc_norm | -4.24 | SIG | -1.41 | ns |
| arc_easy acc_norm | -5.42 | SIG | -1.22 | ns |
| arc_challenge acc_norm | -1.45 | ns | +1.03 | ns |
| winogrande acc | +1.03 | ns | +0.87 | ns |
| openbookqa acc_norm | -2.60 | ns | -0.20 | ns |
| Macro-avg | -3.64 | — | -0.56 | — |

## Compression Context

- **baseline**: full 134.5M-param Llama — reference point, no compression.
- **ae_mlp**: MLP replaced by 4-layer AE bottleneck `hidden → hidden/2 → latent → hidden/2 → hidden` with SiLU (`PEBottleneckMLP` in `pe_layers.py`).
- **tw_root_fc_db3_64_tiered_silu**: learned reduction stage + frozen TrendWavelet basis (Vandermonde trend + db3 wavelet, wavelet_dim ~64) on root/fc projections; per-layer tiered offsets. Basis tensors are buffers, not parameters.

### Parameter Counts and Savings vs Baseline

| Variant | Total | MLP block | Attention | Embed + lm_head | Total Δ vs base | MLP Δ vs base |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 134,515,008 | 79,626,240 | 26,542,080 | 28,346,688 | — | — |
| ae_mlp | 69,307,968 | 14,419,200 | 26,542,080 | 28,346,688 | **-65.2M (-48.5%)** | -65.2M (**-81.9%**) |
| tw_root_fc | 67,209,408 | 12,320,640 | 26,542,080 | 28,346,688 | **-67.3M (-50.0%)** | -67.3M (**-84.5%**) |

Both variants leave attention and embedding/lm_head untouched (exact match in absolute counts). Total-model savings are dominated by the MLP block, which accounts for 59% of baseline params.

### VRAM and Inference Footprint

Mixed-precision AdamW formula: 16 B/trainable-param (2 B bf16 weights + 2 B bf16 grads + 12 B fp32 master + Adam m + v). Inference: 2 B/param (bf16 weights). Activations not included — these dominate at training time and are roughly equal across variants at the same micro-batch / sequence length.

| Variant | Training state | Inference (bf16) | State Δ vs base |
| --- | --- | --- | --- |
| baseline | 2.00 GiB | 0.25 GiB | — |
| ae_mlp | 1.03 GiB | 0.13 GiB | -0.97 GiB (-48.5%) |
| tw_root_fc | 1.00 GiB | 0.13 GiB | -1.00 GiB (-50.0%) |

At this scale, parameter-state VRAM is small relative to total training footprint (~25 GB observed per RTX 5090 with `gradient_checkpointing=false` at `micro_batch_size=2`). The savings show up most clearly at inference and in optimizer-state-bound regimes (e.g. larger models, full-precision optimizer, or constrained-VRAM hardware).

### Training Cost Estimate (3.0B-token budget)

Compute-cost proxy uses the standard `6 · N · D` heuristic (N = total params, D = training tokens). This is an upper bound on the compressed variants' compute — it would be tight if every parameter participated in a dense matmul, which is approximately true for AE bottleneck (all matrices dense), and loose for tw_root_fc (frozen basis tensors still produce dense output, so the matmul shape is closer to baseline than the param count implies). Treat this as the "FLOPs-attributable-to-trainable-params" metric, not a direct wall-clock predictor.

| Variant | Training FLOPs | FLOPs Δ vs base |
| --- | --- | --- |
| baseline | 2.42 EFLOPs | — |
| ae_mlp | 1.25 EFLOPs | -48.4% |
| tw_root_fc | 1.21 EFLOPs | -50.0% |

**Wall-clock and energy** (single RTX 5090 per variant, ~14k tok/s observed throughput at `micro_batch_size=2`, `grad_accum_steps=64`, `gradient_checkpointing=false`; 575 W TDP):

| Run | Tokens | Wall-clock | Energy @ 575 W |
| --- | --- | --- | --- |
| Full 3.0B-token training (any variant) | 3.0B | ~59.5 h (~2.5 d) | ~34 kWh |
| `tw_root_fc` already trained | 1.7B | ~33.7 h | ~19 kWh |
| `tw_root_fc` remaining to 3.0B | 1.3B | **~25.8 h** | **~15 kWh** |

Wall-clock parity across variants reflects that throughput is dominated by attention + dense embedding/lm_head + dataloader, all of which are unchanged. The MLP-FLOP savings show up as headroom (more VRAM, larger feasible batch) rather than as faster steps at the current recipe.

Cost/benefit summary: both compressed variants halve the parameter footprint and (per the FLOPs proxy) the trainable-compute, while paying a ~3-point macro-avg accuracy hit and ~2–3× LAMBADA-ppl regression at full budget. Wall-clock training time is approximately the same across all three.

## Validation Perplexity Trajectory and Projection

Pretraining validation perplexity is logged at every eval checkpoint under `<pellm_data_root>/evals/smol_replacement_paper/{variant}/eval_<tokens>.json`. The two trajectories are converging — `tw_root_fc` is closing the gap with `baseline`, not parallel-tracking it.

**Validation ppl at shared checkpoints (in-domain val set, same evaluation harness):**

| Tokens | baseline ppl | tw_root_fc ppl | tw / base | Δ ppl | Δ loss (nats) |
| --- | --- | --- | --- | --- | --- |
| 250M | 50.28 | 68.81 | 1.369 | +18.53 | +0.314 |
| 500M | 33.81 | 44.25 | 1.309 | +10.44 | +0.269 |
| 1.00B | 26.42 | 33.50 | 1.268 | +7.08 | +0.238 |
| 1.25B | 24.77 | 31.28 | 1.263 | +6.51 | +0.234 |
| 1.50B | 23.41 | 29.49 | 1.260 | +6.08 | +0.231 |
| 1.75B | 22.41 | 28.06 | 1.252 | +5.65 | +0.225 |
| 1.80B | — | 27.84 | — | — | — |

The ratio drops monotonically (1.369 → 1.252) and the loss gap (0.314 → 0.225 nats) is shrinking. This is consistent with `tw_root_fc`'s frozen-basis MLP being a weaker but not asymptotically-stuck approximator — it is still catching up.

**Log-linear fit `loss = a + b · log(tokens)`** (using all eval points after the initial 250M-token transient):

| Variant | a | b | R² | n |
| --- | --- | --- | --- | --- |
| baseline (≥750M) | 9.4887 | -0.2999 | 0.9953 | 9 |
| tw_root_fc (≥750M) | 10.8040 | -0.3512 | 0.9754 | 20 |

`tw_root_fc` has a steeper (more negative) slope than baseline, which is what drives the gap-closure. Sanity: baseline's actual at 2.75B is 19.87 ppl; the fit predicts 19.13 there — within ~4%.

**Projected validation ppl at 3.0B tokens (late-half fit):**

| Variant | Projected loss | Projected ppl | Δ vs baseline |
| --- | --- | --- | --- |
| baseline | 2.945 | **~19.0** | — |
| tw_root_fc | 3.140 | **~23.1** | +4.1 ppl (ratio 1.215) |

Using the all-points fit (slightly more aggressive on the closing slope) instead gives baseline ~18.3 / tw_root_fc ~21.7 / ratio 1.185 — that is the optimistic edge of the projection band. **Reasonable projection band for `tw_root_fc` validation ppl at 3.0B tokens: 21.5–23.5**, depending on how much of the late-stage slope holds through the cosine-decay tail.

Caveats: (1) cosine LR decays toward zero by 3.0B tokens, so the slope flattens in the final ~10–15% of training — actual 3.0B values may end ~0.05–0.10 nats higher than the log-linear extrapolation. (2) The fit assumes no representational ceiling. If tw's frozen basis bottoms out before baseline does, the projection over-shoots the closure. The 1.50B → 1.75B segment (loss gap 0.231 → 0.225, only 0.006 nats closure over 250M tokens) is already showing a slowdown in convergence, so the upper end of the band (~23.5) is more likely than the lower.

## Calibrated Interpretation under Asymmetric Token Budget (tw_root_fc)

`tw_root_fc` has trained on 1.7B / 3.0B tokens (~57%). In the log-linear regime characteristic of small LMs at this scale, the remaining 1.3B tokens (1.76× token multiplier) typically yield non-trivial gains. Combining the validation-ppl projection above with downstream-task scaling intuition:

- **Validation ppl at 3.0B**: 28.06 → **~21.5–23.5** (projection band above), vs baseline ~19.0. Gap closes from 1.25× to ~1.18–1.22×.
- **LAMBADA ppl at 3.0B**: 842 → ~450–550 (rough projection — LAMBADA scales similarly but is more sensitive to tail-of-distribution tokens, still likely worse than `ae_mlp`'s 431).
- **Macro-avg accuracy at 3.0B**: 34.87 → ~36–38 (likely matching or modestly exceeding `ae_mlp`'s 35.43, still below baseline 38.51).

The validation trajectory makes the "tw is fundamentally weaker due to the frozen basis" hypothesis less likely than it appeared from the lm-eval table alone — the gap is closing, not stuck. **No claim of "tw beats X" can be made from the 1.7B checkpoint**, but the projection supports finishing the run.

The deltas vs `ae_mlp` (Δmacro = -0.56, Δhellaswag = -0.11, Δopenbookqa = -0.20) are already small or non-significant on most reasoning tasks at 57% of training. LAMBADA is the only task where tw is meaningfully behind ae_mlp at parity-adjusted projection, and LAMBADA is also the metric most directly tied to remaining LM-loss decay.

## Recommendation

**Resume training `tw_root_fc_db3_64_tiered_silu` to the 3.0B-token budget.** The current evaluation cannot support any A/B claim against `ae_mlp` or `baseline` due to the 1.3B-token deficit; finishing the run is the only way to make the comparison valid.

Conditions that would change this recommendation:

- If a higher-priority variant is queued and GPU time is constrained, deprioritize.
- If the tw_root_fc loss curve has visibly plateaued (not just decelerated) over the last several hundred million tokens in W&B, finishing buys little — log-linear improvement is the assumption underpinning the projection above.
- If a recipe-drift issue is discovered (LR trajectory, grad_accum mismatch) between tw_root_fc and the other variants, restart rather than resume (per the `A/B integrity rule` in CLAUDE.md).

## Eval JSON Sources

- baseline: `<project_root>/evals/smol_replacement_paper/baseline/benchmarks/__mnt__data__pellm__trainedmodels__smol_replacement_paper__baseline/results_2026-05-02T09-58-51.929300.json`
- ae_mlp: `<project_root>/evals/smol_replacement_paper/ae_mlp/benchmarks/__mnt__data__pellm__trainedmodels__smol_replacement_paper__ae_mlp/results_2026-05-02T10-00-42.149646.json`
- tw_root_fc_db3_64_tiered_silu (1.7B): `<project_root>/evals/smol_replacement_paper/tw_root_fc_db3_64_tiered_silu/benchmarks_checkpoint_1700003840/__mnt__data__pellm__checkpoints__smol_replacement_paper__tw_root_fc_db3_64_tiered_silu__tokens_1700003840__hf_model/results_2026-05-07T02-38-41.370459.json`
- tw_root_fc_db3_64_tiered_silu (1.8B follow-up): `<project_root>/evals/smol_replacement_paper/tw_root_fc_db3_64_tiered_silu/benchmarks_checkpoint_1800142848/__mnt__data__pellm__checkpoints__smol_replacement_paper__tw_root_fc_db3_64_tiered_silu__tokens_1800142848__hf_model/results_2026-05-07T03-46-43.384872.json`
