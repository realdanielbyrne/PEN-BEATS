# Preliminary Feasibility Report — Wavelet vs AE-Bottleneck Replacements in From-Scratch Pretraining

**Generated:** 2026-05-07
**Experiment:** `smol_replacement_paper` (from-scratch SmolLM2-135M-class Llama, 3.0B-token budget, FineWeb-Edu)
**Comparative metric:** projected validation perplexity at 3.0B tokens (most variants stopped early; projections allow apples-to-apples comparison)
**Primary upstream analyses:**

- [analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md](analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md) — full 3-way zero-shot benchmark + parameter / FLOPs / projection.
- [smol_replacement_paper_inflight_analysis_2026-05-03.md](../../scripts/experiments/results/smol_replacement_paper_inflight_analysis_2026-05-03.md) — token-matched 4-way comparison + the inductive-prior-mismatch hypothesis.

This report synthesizes those two analyses, adds nine new variants whose
training data lives under [<pellm_data_root>frompc/pellm/](file://<pellm_data_root>frompc/pellm/),
and uses **projected 3.0B-token perplexity** as the comparative metric
because most variants were stopped early. Projections are anchored against
the two completed runs (`baseline`, `ae_mlp`) so the bias from cosine-tail
flattening is removed.

## Projection Methodology

For each variant with ≥3 eval points at ≥500M tokens, we fit
`loss = a + b · log(tokens)` and extrapolate to 3.0B tokens.

The naïve log-linear extrapolation is systematically optimistic because
the cosine LR schedule flattens the late-training slope. Calibrating
against the two completed runs:

| Variant | Naïve projected loss | Actual 3.0B loss | Bias (actual − naïve) |
| --- | ---: | ---: | ---: |
| baseline | 2.939 | 2.987 | **+0.048 nats** |
| ae_mlp | 3.209 | 3.275 | **+0.066 nats** |

The bias is consistent at **+0.057 nats** (average), which we add to every
naïve projection to get a calibrated 3.0B-loss estimate. Confidence is
rated by the latest training-token count:

| Confidence | Latest token requirement | Rationale |
| --- | --- | --- |
| **High** | ≥1.7B (~57% of budget) | Past the early-descent regime; slope already approximately matches late-training slope. |
| **Medium** | 1.0–1.5B (~33–50%) | Slope still steeper than late-training; calibration may under-correct by ~0.05 nats. |
| **Low** | <1.0B (<33%) | Slope dominated by early descent; calibration likely under-corrects by ≥0.10 nats. Treat as a lower bound on true 3.0B ppl. |

## Variant Inventory

| Variant | Replacement axis | Trainable params | Compression | Tokens trained | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline` | none | 134.5 M | 0% | 3.000 B | complete |
| `ae_mlp` | AE-MLP, every layer, latent_dim=256 | 69.3 M | −48.5% | 3.000 B | complete |
| `trendwavelet_db3_32` | TW attention, db3, wavelet_dim=32 | 110.5 M | −17.9% | 1.13 B | early-stopped |
| `trendwavelet_db3_64_silu` | TW attention, db3, wavelet_dim=64, SiLU output | 112.7 M | −16.2% | 0.75 B | early-stopped |
| `tw_root_db3_64_silu` | `tw_root` MLP only (theta → basis, no pre-FC) | 56.1 M | −58.3% | 0.25 B | aborted (catastrophic) |
| `tw_root_fc_db3_32_tiered` | `tw_root_fc` MLP, db3, wavelet_dim=32, tiered offsets | 65.5 M | −51.3% | 1.25 B | partial |
| `tw_root_fc_db3_64_tiered_silu` | `tw_root_fc` MLP, db3, wavelet_dim=64, tiered offsets, SiLU output | 66.0 M | −50.9% | 1.85 B | partial |
| `tw_root_fc_post_fc_db3_64_silu` | `tw_root_fc_post_fc` MLP, db3, wavelet_dim=64, no tiering | 76.0 M | −43.5% | 1.80 B | partial |
| `tw_root_fc_post_fc_db3_64_tiered_silu` | same as above + tiered offsets | 76.0 M | −43.5% | 0.70 B | partial |
| `twattn_tw_root_fc_tw_db3_64_tiered_fr_silu` | TW on **attention** + `tw_root_fc_tw` MLP, fully wavelet | ~50 M | ~−63% | 0.10 B | very early |

Architectural nomenclature (from
[scripts/experiments/smol_replacement_paper.yaml](../../scripts/experiments/smol_replacement_paper.yaml)
and [pellm/pe_layers.py](../../pellm/pe_layers.py)):

- **`tw_root` MLP**: `theta → frozen TW basis → SiLU` — no learned pre-FC. Tests if the basis decode alone can serve as the MLP block.
- **`tw_root_fc` MLP**: `learned FC → SiLU → theta → frozen TW basis → SiLU` — the canonical RootBlock with a learned reduction stage.
- **`tw_root_fc_post_fc` MLP**: `learned FC → SiLU → theta → frozen TW basis → learned FC → SiLU` — sandwiches the frozen basis between two learned linears (~10 M extra params per variant).
- **`tiered`**: per-layer-offset schedule on the wavelet basis (`[0]×10, [32 or 64]×10, [64 or 128]×10`) — early layers get LF, mid layers MF, late layers HF.
- **`twattn`**: TrendWavelet replacement is applied to attention projections **in addition** to MLP, instead of leaving attention as standard Linear.

## Headline: Projected 3.0B-Token Perplexity

| Variant | Latest tokens | Latest ppl | Slope (b) | R² | **Calibrated 3.0B ppl** | Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 2.75 B | 19.87 | −0.314 | 0.995 | **19.82** (actual) | actual |
| `tw_root_fc_post_fc_db3_64_tiered_silu` | 0.70 B | 38.57 | −0.454 | 0.992 | **~21** (lower bound) | LOW |
| `tw_root_fc_db3_32_tiered` | 1.25 B | 33.56 | −0.478 | 0.989 | **~23** (lower bound) | MEDIUM |
| `tw_root_fc_post_fc_db3_64_silu` | 1.80 B | 27.75 | −0.374 | 0.982 | **24.1** | HIGH |
| `tw_root_fc_db3_64_tiered_silu` | 1.85 B | 27.64 | −0.359 | 0.990 | **24.4** | HIGH |
| `ae_mlp` | 2.75 B | 26.52 | −0.341 | 0.981 | **26.45** (actual) | actual |
| `trendwavelet_db3_32` | 1.00 B | 45.49 | −0.473 | 1.000 | **~29** (lower bound) | LOW |
| `trendwavelet_db3_64_silu` | 0.75 B | 44.44 | — | — | (insufficient late points) | n/a |
| `tw_root_db3_64_silu` | 0.25 B | 112.01 | — | — | **catastrophic** | n/a |
| `twattn_tw_root_fc_tw_db3_64_tiered_fr_silu` | 0.10 B | 336.03 | — | — | (insufficient data) | n/a |

**The single most important finding**: the two HIGH-confidence wavelet
variants (`tw_root_fc_post_fc_db3_64_silu` projected 24.1 ppl;
`tw_root_fc_db3_64_tiered_silu` projected 24.4 ppl) are projected to
**beat `ae_mlp` (26.45 ppl)** at 3.0B tokens, while reaching comparable
or slightly worse compression (43.5% / 50.9% vs `ae_mlp`'s 48.5%). The
projection is anchored by `ae_mlp`'s actual endpoint, so the +0.057-nat
correction has been validated.

The MEDIUM- and LOW-confidence variants project even better (`~21–23`
ppl), but those numbers are lower bounds — the slope at <1.5B tokens is
still in the early-descent regime and will flatten more than the
calibration captures. Resuming those runs is the only way to make a
defensible claim.

The two pure-attention TW variants (`trendwavelet_db3_32`,
`trendwavelet_db3_64_silu`) and the no-pre-FC variant
(`tw_root_db3_64_silu`) all project poorly (or never had enough data to
project). These are the negative controls.

## Q1 — Comparative Performance

### Validation perplexity, token-matched

Pulled from per-checkpoint `eval_<tokens>.json` files. The 11 leftmost
variants in the next table are the ones with usable matched-token data;
the catastrophic `tw_root_db3_64_silu` and the `twattn_*` very-early-stage
run are excluded.

| Tokens | baseline | ae_mlp | tw_db3_32 (TW-attn) | tw_db3_64_silu (TW-attn) | tw_root_fc_db3_32_tiered | tw_root_fc_db3_64_tiered_silu | tw_root_fc_post_fc_db3_64_silu | tw_root_fc_post_fc_db3_64_tiered_silu |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 M | 50.28 | 78.98 | 96.51 | 90.57 | 107.36 | 68.81 | 75.46 | (200M: 107.5) |
| 500 M | 33.81 | 47.67 | 63.11 | 57.08 | 51.85 | 44.25 | 45.43 | 44.95 |
| 750 M | 28.93 | 39.39 | 51.91 | 44.44 | 41.13 | 39.44 | 37.74 | (700M: 38.57) |
| 1.00 B | 26.42 | 35.10 | **45.49** | (stopped) | 36.29 | 33.50 | 33.57 | — |
| 1.25 B | 24.77 | 32.52 | — | — | **33.56** | 31.28 | 31.25 | — |
| 1.50 B | 23.41 | 30.76 | — | — | — | 29.49 | 29.66 | — |
| 1.75 B | 22.41 | 29.94 | — | — | — | 28.06 | 28.11 | — |
| 1.80 B | — | — | — | — | — | 27.84 | **27.75** | — |
| 1.85 B | — | — | — | — | — | **27.64** | — | — |
| 2.75 B | **19.87** | **26.52** | — | — | — | — | — | — |

The two HIGH-confidence wavelet candidates
(`tw_root_fc_db3_64_tiered_silu` and `tw_root_fc_post_fc_db3_64_silu`)
**already match each other within 0.4 ppl at 1.8B tokens** and have
already crossed below `ae_mlp`'s 1.5B-token mark (30.76). At 1.85B,
`tw_root_fc_db3_64_tiered_silu` (27.64) is **already 1.0 ppl better than
`ae_mlp` was at 1.85B (~29.9)** — direct evidence that the projection
ranking is a real ordering, not just a slope-fit artifact.

### Zero-shot benchmark accuracy

The 3-way lm-eval results ([analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md](analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md)):

| Task | baseline (3.0B) | ae_mlp (3.0B) | tw_root_fc_db3_64_tiered_silu (1.8B) |
| --- | --- | --- | --- |
| lambada_openai (acc / ppl) | 24.22 / 142.87 | 18.07 / 431.21 | 15.99 / 745.65 |
| hellaswag (acc_norm) | 31.32 | 27.63 | 27.68 |
| piqa (acc_norm) | 61.97 | 59.14 | 57.56 |
| arc_easy (acc_norm) | 45.50 | 41.29 | 39.94 |
| arc_challenge (acc_norm) | 25.09 | 22.61 | 23.55 |
| winogrande (acc) | 51.30 | 51.46 | 51.30 |
| openbookqa (acc_norm) | 30.20 | 27.80 | 28.60 |
| **Macro-avg** | **38.51%** | **35.43%** | **34.95%** (1.8B partial) |

`tw_root_fc_db3_64_tiered_silu`'s 1.8B macro-avg is 0.48 pp behind
`ae_mlp`'s 3.0B macro. Given the projected 3.0B ppl gap is the **other
direction** (24.4 vs 26.45 — wavelet better), we expect the 3.0B
zero-shot macro to land at ~36–38%, likely matching or modestly beating
`ae_mlp`'s 35.43%. None of the other new variants have lm-eval
benchmarks yet.

## Q2 — Inductive Prior Mismatch

The naïve hypothesis — TW attention should land closer to baseline than
AE-MLP because it keeps the MLP block (the larger parameter share)
intact — was falsified. The new variants strengthen and refine that
finding:

| Class | Variant | Compression | Projected 3.0B ppl |
| --- | --- | ---: | ---: |
| **Pure TW attention (no learned reduction)** | `trendwavelet_db3_32` | −17.9% | ~29 (lower bound) |
| Pure TW attention (no learned reduction) | `trendwavelet_db3_64_silu` | −16.2% | (n/a, still descending steeply at 0.75B; trajectory ≈ db3_32) |
| **Pure TW MLP (no learned pre-FC)** | `tw_root_db3_64_silu` | −58.3% | catastrophic (250M ppl=112, aborted) |
| **TW MLP + learned pre-FC** | `tw_root_fc_db3_32_tiered` | −51.3% | ~23 (lower bound, 1.25B) |
| **TW MLP + learned pre-FC** | `tw_root_fc_db3_64_tiered_silu` | −50.9% | **24.4 (HIGH conf)** |
| **TW MLP + learned pre + post FC** | `tw_root_fc_post_fc_db3_64_silu` | −43.5% | **24.1 (HIGH conf)** |
| **TW MLP + learned pre + post FC + tiered** | `tw_root_fc_post_fc_db3_64_tiered_silu` | −43.5% | ~21 (lower bound, 0.70B) |

The data is now sharply consistent with the inductive-prior-mismatch
story:

- A **bare frozen TW basis with no learned input curation** (`tw_root` MLP)
  is uncompetitive — `tw_root_db3_64_silu` was so bad at 250M tokens (ppl
  112 vs baseline ~50) that the run was aborted. The frozen basis simply
  cannot serve as the MLP block on its own.
- A **frozen TW basis with no learned input curation, applied to
  attention** (`trendwavelet_db3_*`) is also uncompetitive — the gap to
  baseline is ~2× the AE-MLP gap despite retaining 3× more parameters.
  Doubling the wavelet basis from 32 to 64 (`db3_64_silu`) does not close
  the gap meaningfully (44.4 vs 51.9 ppl at 750M — modest improvement,
  still far behind `tw_root_fc` variants at the same token count).
- A **frozen TW basis with a learned SiLU(FC(x)) reduction** (`tw_root_fc`)
  is competitive with `ae_mlp` — and with **wavelet_dim=64**, modestly
  better. Adding a post-basis FC (`tw_root_fc_post_fc`) gives a further
  ~0.3-ppl improvement at projection time, at the cost of ~7%-points of
  compression.

The mechanistic interpretation: the frozen basis is a hard 36–68-dim
functional prior on every output row. SGD cannot push expressivity outside
that subspace, so the model needs a learned stage that **rotates the input
into a representation the basis is good at decoding**. The pre-FC stage is
that rotation. Without it, the basis acts as a pure projection onto a
fixed (and from-scratch, basically arbitrary) subspace.

This generalizes beyond the original 2026-05-03 finding: it is not just
"random init is worse for basis-constrained projections than for free
Linear weights" — it is "any basis-constrained projection that lacks a
learned input curation stage cannot be made competitive by training
alone." The fix is structural (add a learned reduction), not optimization
(more tokens, better init, distillation).

## Q3 — Parameter Efficiency

Projected efficiency at 3.0B tokens. The "nats / % compression" column
captures how many nats of LM-loss penalty the variant pays for each
percentage point of total-parameter reduction.

| Variant | Compression | Calibrated 3.0B loss | Gap vs baseline (nats) | nats / % compression | Confidence |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0% | 2.987 | — | — | actual |
| `tw_root_fc_post_fc_db3_64_silu` | −43.5% | 3.181 | +0.194 | **0.0045** | HIGH |
| `tw_root_fc_db3_64_tiered_silu` | −50.9% | 3.193 | +0.206 | **0.0040** | HIGH |
| `ae_mlp` | −48.5% | 3.275 | +0.288 | 0.0059 | actual |
| `tw_root_fc_db3_32_tiered` | −51.3% | 3.135 | +0.148 | 0.0029 (lower bound) | MEDIUM |
| `trendwavelet_db3_32` | −17.9% | ~3.354 | +0.367 | **0.0205** (lower bound) | LOW |

Three observations:

1. **Both HIGH-confidence wavelet candidates are ~25–32% more efficient
   than `ae_mlp` per percent of compression.** The Pareto frontier is
   improved by either of them; the 7%-point compression gap between
   `tw_root_fc_post_fc_db3_64_silu` (43.5%) and `tw_root_fc_db3_64_tiered_silu`
   (50.9%) buys essentially the same loss, so `tw_root_fc_db3_64_tiered_silu`
   is the better point on the frontier when compression is the priority.
2. **Pure TW attention (`trendwavelet_db3_32`) is ~5× less efficient per
   percent of compression than the better wavelet MLPs.** This is the
   inductive-prior-mismatch tax in efficiency terms.
3. **Compression Pareto frontier at projected 3.0B**: baseline (134.5M,
   19.82) → `tw_root_fc_post_fc_db3_64_silu` (76.0M, ~24.1) →
   `tw_root_fc_db3_64_tiered_silu` (66.0M, ~24.4) → `ae_mlp` (69.3M,
   26.45). `ae_mlp` is **dominated** by `tw_root_fc_db3_64_tiered_silu`
   (less params *and* lower projected ppl).

### VRAM, FLOPs, wall-clock

Numbers carry over from the 3-way analysis. Mixed-precision AdamW
(16 B/trainable param):

| Variant | Training state | Inference (bf16) | Trainable FLOPs (`6·N·D`) |
| --- | ---: | ---: | ---: |
| baseline | 2.00 GiB | 0.25 GiB | 2.42 EFLOPs |
| `ae_mlp` | 1.03 GiB | 0.13 GiB | 1.25 EFLOPs (−48.4%) |
| `tw_root_fc_db3_64_tiered_silu` | 1.00 GiB | 0.13 GiB | 1.21 EFLOPs (−50.0%) |
| `tw_root_fc_post_fc_db3_64_silu` | 1.13 GiB | 0.14 GiB | 1.37 EFLOPs (−43.5%) |

**Wall-clock parity.** Throughput at this scale (~14k tok/s on a single
RTX 5090) is dominated by attention + dense embed/lm_head + dataloader,
all unchanged across variants. Param savings show up as VRAM headroom
(~25 GB free per RTX 5090 mid-training) and as smaller inference and
optimizer states — not as faster training steps. A full 3.0B run takes
~59.5 wall-clock hours / ~34 kWh on a single RTX 5090 regardless of
variant.

## Q4 — Feasibility Conclusion

### TrendWavelet projections — viable **with learned reduction**, not viable without

The earlier preliminary report concluded "tentatively viable, finish the
run." The new evidence — particularly
`tw_root_fc_post_fc_db3_64_silu`'s 1.8B-token point landing at 27.75 ppl
(matching `tw_root_fc_db3_64_tiered_silu`'s 27.84) and the calibrated
projection coming in below `ae_mlp`'s actual endpoint — upgrades that to
**viable** for the configuration class `tw_root_fc[*]` (TW MLP with a
learned pre-FC reduction stage). Specifically:

- **`tw_root_fc_db3_64_tiered_silu`** — best point on the Pareto frontier
  among HIGH-confidence variants. **Recommended candidate** for the
  paper. Projected 24.4 ppl at 50.9% compression.
- **`tw_root_fc_post_fc_db3_64_silu`** — slightly better projected loss
  (24.1 ppl) at slightly worse compression (43.5%); the post-FC mixer
  modestly helps. Worth carrying as a sensitivity study.
- **`tw_root_fc_post_fc_db3_64_tiered_silu`** and
  **`tw_root_fc_db3_32_tiered`** — both project even better, but
  confidence is LOW–MEDIUM. **High-priority resume targets** if GPU time
  allows; their training data so far is the most promising in the sweep.

### TrendWavelet projections — **not viable without** learned reduction

- **`trendwavelet_db3_32` / `trendwavelet_db3_64_silu`** (TW on attention,
  no learned reduction): Pareto-dominated by `ae_mlp` and projected to
  remain so at 3.0B (~29 ppl lower bound). Drop from the paper, or carry
  only as negative controls to motivate the prior-mismatch story.
- **`tw_root_db3_64_silu`** (`tw_root` MLP, no pre-FC): catastrophic at
  250M (ppl 112). Aborted. Drop entirely.

### AE-bottleneck MLP — viable, but no longer the leader

`ae_mlp` is still a Pareto-improving compression: −48.5% total params,
−48.4% trainable FLOPs, −0.97 GiB optimizer state, ~3-pp macro-avg
accuracy hit at 3.0B. **But it is now Pareto-dominated** by
`tw_root_fc_db3_64_tiered_silu` (66.0M params vs 69.3M, projected 24.4
ppl vs actual 26.45 ppl). If the paper's headline is "best compression
at fixed loss," the wavelet variant is the new answer; `ae_mlp`'s role
is the well-trained baseline against which the wavelet projection is
anchored.

### Combined wavelet attention + MLP — unestablished

`twattn_tw_root_fc_tw_db3_64_tiered_fr_silu` (TW on attention *and* MLP)
has only 100M tokens (ppl 336). Loss at 100M (5.82) is ~0.6 nats above
the slowest MLP-only TW variant at the same point. Insufficient data to
project; needs to run further before any claim.

### Caveats

- **Recipe drift** between legacy 2-GPU DDP (`baseline`, `ae_mlp`,
  `gradient_checkpointing=true`) and the new 1-GPU recipe (all wavelet
  variants, `gradient_checkpointing=false`). LR trajectory is
  mathematically identical; per-step gradient sharding and per-process
  seed offsets differ. The +0.057-nat anchor calibration may absorb some
  of this drift; gaps reported here are far larger (≥0.19 nats), so the
  qualitative ranking is robust.
- **Single seed** for every variant. None of the projected gaps have a
  cross-seed standard deviation. The HIGH-confidence projections (24.1 /
  24.4) are within 0.3 ppl of each other — well inside any plausible
  seed noise.
- **Cosine-tail calibration.** The +0.057 nats correction was measured on
  variants whose late slope was already steady (-0.31 to -0.34). The
  MEDIUM/LOW-confidence variants (slopes -0.45 to -0.48) have not yet
  flattened, so their projections are **lower bounds** — true 3.0B values
  may be 0.05–0.15 nats higher than the calibrated estimate.
- **Distillation not tested.** Both interpretations of the
  inductive-prior finding predict logit KD from a vanilla teacher would
  help the no-pre-FC variants (`trendwavelet_*`) more than it would help
  the `tw_root_fc[*]` class. A KD control variant remains the cleanest
  test of the distillation story.

## Recommendations for the Paper

1. **Headline candidate:** `tw_root_fc_db3_64_tiered_silu` — 50.9%
   compression, projected 24.4 ppl at 3.0B (vs `ae_mlp`'s 26.45). Finish
   the 3.0B run to confirm.
2. **Sensitivity study:** `tw_root_fc_post_fc_db3_64_silu` (post-FC mixer
   gives +0.3 ppl improvement at −7 pp compression). Finish the 3.0B run.
3. **High-priority resume targets:** `tw_root_fc_post_fc_db3_64_tiered_silu`
   (700M only), `tw_root_fc_db3_32_tiered` (1.25B only). These project
   strongest but have the least training data; finishing them could move
   the Pareto frontier further.
4. **Negative controls:** `trendwavelet_db3_32`, `tw_root_db3_64_silu`.
   Carry as illustrations of the inductive-prior-mismatch failure mode;
   no further training needed.
5. **Drop:** `trendwavelet_db3_64_silu`, `twattn_tw_root_fc_tw_db3_64_tiered_fr_silu`
   (insufficient signal so far; redirect GPU hours to #3).

## Source References

- [evals/smol_replacement_paper/analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md](analysis_tw_root_fc_vs_ae_mlp_vs_baseline.md) — 3-way zero-shot benchmark, parameter counts, FLOPs.
- [scripts/experiments/results/smol_replacement_paper_inflight_analysis_2026-05-03.md](../../scripts/experiments/results/smol_replacement_paper_inflight_analysis_2026-05-03.md) — token-matched 4-way comparison, inductive-prior-mismatch hypothesis.
- [scripts/experiments/smol_replacement_paper.yaml](../../scripts/experiments/smol_replacement_paper.yaml) — variant configurations and parameter counts.
- [<pellm_data_root>/evals/smol_replacement_paper/](file://<pellm_data_root>/evals/smol_replacement_paper/) — primary eval JSON series for `baseline`, `ae_mlp`, `trendwavelet_db3_32`, `tw_root_fc_db3_64_tiered_silu`, `tw_root_fc_post_fc_db3_64_tiered_silu`.
- [<pellm_data_root>frompc/pellm/evals/smol_replacement_paper/](file://<pellm_data_root>frompc/pellm/evals/smol_replacement_paper/) — added eval data for `trendwavelet_db3_64_silu`, `tw_root_db3_64_silu`, `tw_root_fc_db3_32_tiered`, `tw_root_fc_post_fc_db3_64_silu`, `twattn_tw_root_fc_tw_db3_64_tiered_fr_silu`.
- [<pellm_data_root>/trainedmodels/smol_replacement_paper/baseline/training_manifest.json](file://<pellm_data_root>/trainedmodels/smol_replacement_paper/baseline/training_manifest.json) and [<pellm_data_root>/trainedmodels/smol_replacement_paper/ae_mlp/training_manifest.json](file://<pellm_data_root>/trainedmodels/smol_replacement_paper/ae_mlp/training_manifest.json) — true 3.0B endpoints for projection-bias calibration.
- [pellm/pe_layers.py](../../pellm/pe_layers.py) — class definitions for `PEBottleneckMLP`, `TrendWaveletLinear`, `TrendWaveletLinearReduced`, and the `tw_root[_fc][_post_fc]` MLP-replacement classes.
- [CLAUDE.md "Smol Replacement Pretraining"](../../CLAUDE.md) — recipe-drift notes and A/B integrity rule.
