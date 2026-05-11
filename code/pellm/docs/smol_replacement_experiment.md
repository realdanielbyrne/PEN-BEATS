# Smol Replacement Paper Experiment

This is the from-scratch language-modeling experiment for the architecture
pivot. It is separate from the older pretrained Llama replacement/repair
pipelines.

## Experiment File

`scripts/experiments/smol_replacement_paper.yaml`

The runner is:

`scripts/pretrain_smol_replacement.py`

## Variants

- `baseline`: dense Smol-style Llama baseline
- `ae_mlp`: AE MLP replacement only
- `trendwavelet`: TrendWavelet attention projections only
- `ae_tw`: both AE MLP and TrendWavelet attention projections

## Start Commands

Smoke test without loading data:

```bash
.venv/bin/python scripts/pretrain_smol_replacement.py \
  scripts/experiments/smol_replacement_paper.yaml \
  --variant ae_tw \
  --dry-run
```

Single-GPU/local launch:

```bash
.venv/bin/python scripts/pretrain_smol_replacement.py \
  scripts/experiments/smol_replacement_paper.yaml \
  --variant baseline
```

Two-GPU launch:

```bash
.venv/bin/accelerate launch --num_processes 2 scripts/pretrain_smol_replacement.py \
  scripts/experiments/smol_replacement_paper.yaml \
  --variant baseline
```

Run the four main variants independently:

```bash
for variant in baseline ae_mlp trendwavelet ae_tw; do
  .venv/bin/accelerate launch --num_processes 2 scripts/pretrain_smol_replacement.py \
    scripts/experiments/smol_replacement_paper.yaml \
    --variant "$variant"
done
```

Pilot budget override:

```bash
.venv/bin/accelerate launch --num_processes 2 scripts/pretrain_smol_replacement.py \
  scripts/experiments/smol_replacement_paper.yaml \
  --variant ae_tw \
  --token-budget 300000000
```

## Data Download Behavior

The datasource is `HuggingFaceFW/fineweb_edu_100BT-shuffled`, streamed through
the Hugging Face `datasets` library. It does not download the whole dataset
upfront; it streams shards on demand and uses the configured HF cache.

The runner sets these defaults unless already defined:

```bash
HF_HOME=<pellm_data_root>/hf_home
HF_DATASETS_CACHE=<pellm_data_root>/datasets/hf_cache
TRANSFORMERS_CACHE=<pellm_data_root>/datasets/transformers_cache
TMPDIR=<pellm_data_root>/tmp
```

Model outputs go to `<pellm_data_root>/trainedmodels/smol_replacement_paper/`.
Intermediate checkpoints go to `<pellm_data_root>/checkpoints/smol_replacement_paper/`.
Raw eval summaries go to `<pellm_data_root>/evals/smol_replacement_paper/`.

## Training Recipe

All variants share an identical optimizer/schedule recipe — this is required for
A/B validity. The current values in `scripts/experiments/smol_replacement_paper.yaml`:

| Field | Value | Notes |
|---|---|---|
| `optimizer` | AdamW | `betas=[0.9, 0.95]`, `weight_decay=0.1`, `grad_clip=1.0` |
| `lr` | `6.0e-4` | Peak LR; matches GPT-2 small / OPT-125M / Pythia-160M convention for ~135M-class from-scratch training |
| `warmup_ratio` | `0.03` | Linear warmup from 0 → peak LR over the first 3% of optimizer steps |
| `schedule` | cosine to zero | Standard `get_cosine_schedule_with_warmup` from `transformers` |
| `sequence_length` | 2048 | Matches SmolLM2-135M pretraining |
| `micro_batch_size` | 2 | Per-GPU |
| `grad_accum_steps` | 32 | |
| `token_budget` | 3,000,000,000 | ~Chinchilla-optimal for 135M params |

With 2 GPUs, `tokens_per_update = 2 × 2048 × 32 × 2 = 262,144`, giving
`total_steps = ⌈3e9 / 262,144⌉ = 11,445` optimizer updates and
`warmup_steps = int(11,445 × 0.03) = 343`.

### Scheduler scaling under `accelerate`

`scripts/pretrain_smol_replacement.py` constructs the cosine scheduler with
`warmup_steps * accelerator.num_processes` and
`total_steps * accelerator.num_processes`. This compensates for the
`AcceleratedScheduler` wrapper, which by default advances the underlying
scheduler once per process per step. Without this scaling, the LR reaches zero
at ~50% of training on a 2-GPU run. See the "Smol Replacement Pretraining"
section of `CLAUDE.md` for the full rationale.

The training loop also gates `scheduler.step()` and the `step` counter on
`accelerator.sync_gradients` so that scheduler advances align with optimizer
updates rather than micro-batches.

## W&B Logging

W&B is enabled in `scripts/experiments/smol_replacement_paper.yaml`:

```yaml
experiment:
  wandb:
    enabled: true
    project: "pellm-smol-replacement"
    group: "smol_replacement_paper"
    mode: "online"
```

The runner logs training loss, learning rate (`train/lr`), tokens/sec, eval loss,
perplexity, checkpoint paths, and final model/eval locations. Console log lines
include the same `lr=` value (scientific notation) so warmup and decay can be
monitored directly from the terminal. Local W&B files are written under
`<pellm_data_root>/runs/smol_replacement_paper/<variant>/wandb`.

Use `mode: "offline"` in the YAML if you want to train without syncing during
the run.
