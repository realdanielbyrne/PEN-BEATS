# PELLM: Parameter-Efficient LLM Layers

N-BEATS-inspired parameter-efficient replacements for standard Llama attention projections and MLP blocks. Wraps `meta-llama/Llama-3.2-1B-Instruct` with two independent extension points that can be tested separately or combined.

## Inspiration

This project ports proven architectures from the N-BEATS Lightning codebase (included in this archive under `../nbeats/`), a time-series forecasting framework, into the LLM domain:

- **TrendWaveletLinear** (attention projections) — from the `TrendWaveletAE` block. Replaces dense `nn.Linear` layers with a factorized representation: input is projected to a small coefficient vector, then expanded through two frozen basis matrices (Vandermonde polynomial + SVD-orthonormalized DWT). Only the coefficient projection is trainable, yielding ~97% parameter reduction per attention layer.

- **PEBottleneckMLP** (MLP replacement) — from the `AERootBlock` backbone. Replaces the 3-projection SwiGLU MLP (gate + up + down, totaling 50.3M params/layer) with a 4-layer autoencoder bottleneck (`hidden -> hidden/2 -> latent -> hidden/2 -> hidden`), yielding ~90% parameter reduction.

- **PEBottleneckMLPLG** (MLP with learned gate) — from the `AERootBlockLG` backbone. Same as above but adds a learnable `sigmoid(gate) * z` at the bottleneck, letting the network discover effective latent dimensionality during training.

## Install

```bash
pip install -e ".[experiments]"   # installs all dependencies for paper experiments
```

Requires Python >= 3.10, PyTorch >= 2.1 (CUDA build), Transformers >= 4.40.

## Hardware Requirements

PELLM experiments require a **CUDA-capable NVIDIA GPU** with at least **16 GB VRAM**.
The smol-replacement pretraining uses `accelerate launch` and was developed on 2× RTX 5090
(32 GB each); a single 24 GB GPU works with the default settings.

Run the pre-flight check **before committing to a long run**:

```bash
python scripts/check_hardware.py
```

The script will print one of:

- `[ OK ]` — CUDA found, VRAM sufficient, `accelerate` importable; ready to train.
- `[WARN]` — CUDA found but VRAM is below 24 GB; training may OOM at default batch
  settings — see the warning message for mitigation steps.
- `[FAIL]` — **No CUDA GPU detected.** This means the machine cannot run the PELLM
  training experiments. Pre-computed lm-eval-harness results are available in
  `evals/smol_replacement_paper/` for reviewer inspection without retraining.

## Target Model

`meta-llama/Llama-3.2-1B-Instruct`: 16 decoder layers, hidden_size=2048, 32 Q-heads / 8 KV-heads, head_dim=64, intermediate_size=8192.

## Architecture

Two independent axes of modification, controlled by config flags:

| Component | Mode | Layer class | Description |
| --- | --- | --- | --- |
| Attention | `standard` | `PELinear` | Identical to `nn.Linear` (baseline) |
| Attention | `trend_wavelet` | `TrendWaveletLinear` | Frozen Vandermonde + DWT basis expansion |
| MLP | `standard` | `LlamaMLP` | Standard SwiGLU (baseline) |
| MLP | `ae` | `PEBottleneckMLP` | AE bottleneck |
| MLP | `ae_lg` | `PEBottleneckMLPLG` | AE bottleneck with learned gate |

### Parameter Savings

**Attention projections (trend_wavelet, basis_dim=32 = trend_dim 4 + wavelet_dim 28):**

| Projection | Standard | TrendWaveletLinear | Savings |
|---|---|---|---|
| q_proj (2048 -> 2048) | 4.19M | 65.5K | 98.4% |
| k_proj (2048 -> 512) | 1.05M | 65.5K | 93.8% |
| v_proj (2048 -> 512) | 1.05M | 65.5K | 93.8% |
| o_proj (2048 -> 2048) | 4.19M | 65.5K | 98.4% |
| **All 16 layers** | **168M** | **4.2M** | **97.5%** |

**MLP (ae/ae_lg, latent_dim=256):**

| Component | Standard SwiGLU | AE Bottleneck | Savings |
|---|---|---|---|
| Per layer | 50.3M | 4.7M | 90.7% |
| **All 16 layers** | **805M** | **75.5M** | **90.6%** |

## Paper Experiments

The two experiments referenced in the paper are:

1. **AE pretraining** — `scripts/experiments/ae_pretrain_loss_sweep.yaml`
2. **Smol-replacement paper** — `scripts/experiments/smol_replacement_paper.yaml`

### Step 1 — Hardware check

```bash
python scripts/check_hardware.py
```

Exit 0 = ready to train. Exit 1 = CUDA GPU unavailable; inspect
`evals/smol_replacement_paper/` for pre-computed results instead.

### Step 2 — Model access

Accept the Llama 3.2 Community License on Hugging Face, then authenticate:

```bash
huggingface-cli login
```

The smol-replacement experiment uses `HuggingFaceTB/SmolLM2-135M` (public,
no license gate). The AE-pretraining experiment targets
`meta-llama/Llama-3.2-1B-Instruct` and **requires** the HF login above.

### Step 3 — Artifact directories

```bash
python scripts/setup_artifact_dirs.py
```

### Experiment A: AE pretraining loss sweep

Sweeps 7 pre-training loss functions (MSE, cosine, Huber, soft-KL at T=2/4,
combined α=0.5/0.8) for the AE-bottleneck MLP replacement of Llama layer 15.
Runs `finetune.py` with `ae_pretrain_epochs=5` and `epochs=0` (eval only after
pre-training). GPU required; ~5-10 min per config on a 24 GB GPU.

```bash
python scripts/run_from_yaml.py scripts/experiments/ae_pretrain_loss_sweep.yaml
```

Run a single config only:

```bash
python scripts/run_from_yaml.py scripts/experiments/ae_pretrain_loss_sweep.yaml \
    --config loss_mse
```

Dry-run (no training):

```bash
python scripts/run_from_yaml.py scripts/experiments/ae_pretrain_loss_sweep.yaml \
    --dry-run
```

### Experiment B: Smol-replacement paper

From-scratch Smol-style architecture comparison: trains 30+ architecture
variants (baseline, TrendWavelet attention, AE MLP, combined) from random
initialization on FineWeb-edu. Each variant trains to a 3 B-token budget.
Designed for parallel execution: **one variant per GPU, two in parallel**.

**Single-variant launch (one GPU):**

```bash
# First check a dry-run
python scripts/pretrain_smol_replacement.py \
    scripts/experiments/smol_replacement_paper.yaml \
    --variant ae_mlp --dry-run

# Full run
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    scripts/pretrain_smol_replacement.py \
    scripts/experiments/smol_replacement_paper.yaml \
    --variant tw_root_fc_db3_64_tiered_silu
```

**Parallel launcher (two variants concurrently on two GPUs):**

```bash
python scripts/run_smol_replacement_parallel.py \
    scripts/experiments/smol_replacement_paper.yaml
```

Pre-computed lm-eval-harness results for `baseline`, `ae_mlp`, and
`tw_root_fc_db3_64_tiered_silu` are under `evals/smol_replacement_paper/`.

## Experiments (exploratory / ablation)

All experiments below use `scripts/finetune.py`. Each run loads pretrained Llama weights, measures baseline perplexity on the selected dataset, optionally fine-tunes, then measures final perplexity.

### Experiment 0: Baseline (no modifications)

Establish the reference perplexity with the unmodified model.

```bash
python scripts/finetune.py \
    --pe-attn-mode standard \
    --pe-mlp-mode standard \
    --epochs 0
```

### Experiment 1: TrendWavelet Attention Only

Test the TrendWavelet attention projection replacement in isolation. MLP is unchanged (standard SwiGLU). Pretrained attention weights are projected onto the combined basis via pseudo-inverse.

```bash
python scripts/finetune.py \
    --pe-attn-mode trend_wavelet \
    --pe-mlp-mode standard \
    --trend-dim 4 \
    --wavelet-dim 28 \
    --wavelet-type db3 \
    --epochs 3 --lr 1e-4 --batch-size 4
```

**Variations to try:**

```bash
# Different wavelet families
python scripts/finetune.py \
    --pe-attn-mode trend_wavelet --pe-mlp-mode standard \
    --wavelet-type haar --epochs 3 --lr 1e-4

python scripts/finetune.py \
    --pe-attn-mode trend_wavelet --pe-mlp-mode standard \
    --wavelet-type sym5 --epochs 3 --lr 1e-4

# Larger basis (more coefficients, less compression)
python scripts/finetune.py \
    --pe-attn-mode trend_wavelet --pe-mlp-mode standard \
    --trend-dim 8 --wavelet-dim 56 --epochs 3 --lr 1e-4

# Higher-frequency band (skip low-frequency rows)
python scripts/finetune.py \
    --pe-attn-mode trend_wavelet --pe-mlp-mode standard \
    --wavelet-basis-offset 16 --epochs 3 --lr 1e-4

# Per-layer frequency sweep (low -> high across 16 layers)
python scripts/finetune.py \
    --pe-attn-mode trend_wavelet --pe-mlp-mode standard \
    --0 \
    --epochs 3 --lr 1e-4
```

### Experiment 2: AE Bottleneck MLP Only

Test the AE-bottleneck MLP replacement in isolation. Attention is unchanged (standard PELinear). MLP layers are randomly initialized (architecture is fundamentally different from SwiGLU).

```bash
python scripts/finetune.py \
    --pe-attn-mode standard \
    --pe-mlp-mode ae \
    --ae-latent-dim 256 \
    --epochs 3 --lr 1e-4 --batch-size 4
```

**Variations to try:**

```bash
# Smaller bottleneck (more compression, more parameter savings)
python scripts/finetune.py \
    --pe-attn-mode standard --pe-mlp-mode ae \
    --ae-latent-dim 128 --epochs 3 --lr 1e-4

# Larger bottleneck (less compression, closer to original capacity)
python scripts/finetune.py \
    --pe-attn-mode standard --pe-mlp-mode ae \
    --ae-latent-dim 512 --epochs 3 --lr 1e-4

# AE pre-training with diverse FineWeb activations
python scripts/finetune.py \
    --pe-attn-mode standard --pe-mlp-mode ae \
    --ae-latent-dim 256 --ae-pretrain-epochs 10 \
    --ae-dataset fineweb --ae-cache-num-samples 10000 \
    --epochs 3 --lr 1e-4
```

### Experiment 3: AE Bottleneck MLP with Learned Gate Only

Same as experiment 2 but with the learned-gate variant. After training, the script prints per-layer gate statistics showing which latent dimensions the network kept active.

```bash
python scripts/finetune.py \
    --pe-attn-mode standard \
    --pe-mlp-mode ae_lg \
    --ae-latent-dim 256 \
    --epochs 3 --lr 1e-4 --batch-size 4
```

**Variations to try:**

```bash
# Over-provision latent dim and let gate prune
python scripts/finetune.py \
    --pe-attn-mode standard --pe-mlp-mode ae_lg \
    --ae-latent-dim 512 --epochs 5 --lr 1e-4
```

### Experiment 4: TrendWavelet Attention + AE MLP (Combined)

Test both replacements together. This is the maximum compression configuration.

```bash
python scripts/finetune.py \
    --pe-attn-mode trend_wavelet \
    --attn-init pretrained \
    --pe-mlp-mode ae_lg \
    --trend-dim 4 --wavelet-dim 28 --wavelet-type db3 \
    --ae-latent-dim 256 \
    --epochs 3 --lr 1e-4 --batch-size 4
```

### Experiment 5: TrendWavelet Attention + AE MLP + Per-Layer Frequency Sweep

Combined mode with mixed frequency bands across layers.

```bash
python scripts/finetune.py \
    --pe-attn-mode trend_wavelet \
    --pe-mlp-mode ae_lg \
    --ae-latent-dim 256 \
    --per-layer-offsets 0 0 4 4 8 8 12 12 16 16 20 20 24 24 28 28 \
    --epochs 3 --lr 1e-4 --batch-size 4
```

## Disk Cleanup

Large YAML sweeps can leave shared activation caches behind if a run is interrupted or fails mid-extraction. Before launching another big sweep, or whenever disk usage spikes unexpectedly, preview reclaimable generated artifacts with:

```bash
.venv/bin/python scripts/prune_experiment_artifacts.py
```

Apply the safe cache cleanup:

```bash
.venv/bin/python scripts/prune_experiment_artifacts.py --apply
```

Target one experiment only:

```bash
.venv/bin/python scripts/prune_experiment_artifacts.py --experiment trendwavelet_layer15_sweep --apply
```

Optional, higher-impact cleanup:

- Add `--include-checkpoints` to remove `trainedmodels/<experiment>/` directories.
- Add `--include-wandb` to remove local `wandb/` run folders after runs have already synced to W&B.

## Artifact Storage

Heavy training artifacts should live on the large data drive under
`<pellm_data_root>` rather than inside the git checkout. The YAML orchestrators
default model checkpoints, run logs, caches, and temporary files to that root
unless the environment or YAML explicitly overrides them.

Create the canonical directories with:

```bash
python scripts/setup_artifact_dirs.py
```

See [docs/artifact_storage_and_publishing.md](docs/artifact_storage_and_publishing.md)
for the full directory layout and Hugging Face publishing workflow.

## Smol Replacement Paper Experiment

The from-scratch architecture experiment lives in
`scripts/experiments/smol_replacement_paper.yaml` and is run with
`scripts/pretrain_smol_replacement.py`. This is the recommended path for the
paper pivot; it is separate from the older pretrained Llama replacement/repair
pipelines.

See [docs/smol_replacement_experiment.md](docs/smol_replacement_experiment.md)
for start commands and data-source behavior.

## CLI Reference

```text
scripts/finetune.py

Model:
  --model-name          Base Llama model (default: meta-llama/Llama-3.2-1B-Instruct)

Attention:
  --pe-attn-mode        standard | trend_wavelet (default: standard)
  --attn-init           pretrained | lstsq | svd | cur | fourier | random
  --trend-dim           Vandermonde polynomial degree (default: 4)
  --wavelet-dim         DWT basis rows (default: 28)
  --wavelet-type        PyWavelets family string (default: db3)
  --wavelet-basis-offset  Frequency band offset, 0=low (default: 0)
  --per-layer-offsets   Space-separated per-layer basis_offset overrides

MLP:
  --pe-mlp-mode         standard | ae | ae_lg (default: standard)
  --ae-latent-dim       AE bottleneck width (default: 256)

Dataset:
  --dataset             wikitext2 | wikitext103 | fineweb (default: wikitext2)
                        Main LM fine-tuning and perplexity benchmark dataset.
                        wikitext2: ~2M tokens; wikitext103: ~103M tokens;
                        fineweb: streaming subset of HuggingFaceFW/fineweb.
  --dataset-num-samples Raw documents to stream when --dataset fineweb (default: 50000)
                        Train uses N docs; val and test each use N/10 docs.

Attention Pre-training:
  --attn-pretrain-epochs  Reconstruction pre-training epochs for TrendWavelet attention (default: 0)
  --attn-dataset          wikitext2 | wikitext103 | fineweb (default: wikitext2)
  --attn-cache-dir        Directory for cached teacher attention activations
  --attn-cache-num-samples  FineWeb sample count for attention caching (default: 10000)

AE Pre-training:
  --ae-pretrain-epochs  Reconstruction pre-training epochs (default: 0)
  --ae-dataset          wikitext2 | fineweb — AE activation caching only,
                        independent of --dataset (default: wikitext2)
  --ae-cache-num-samples  Raw text samples for fineweb caching (default: 10000)
  --ae-cache-dir        Persistent activation cache directory (default: temp)

Training:
  --epochs              Number of fine-tuning epochs, 0=eval only (default: 3)
  --lr                  Learning rate (default: 1e-4)
  --batch-size          Batch size (default: 4)
  --max-length          Max sequence length (default: 512)
  --grad-accum-steps    Gradient accumulation steps (default: 4)
  --max-eval-batches    Max batches for perplexity eval, 0=all (default: 50)

Output:
  --output-dir          Directory to save fine-tuned model
  --dtype               float32 | float16 | bfloat16 (default: float32)
```

## Weight Initialization

- **TrendWavelet attention**: Pretrained weights are projected onto the combined basis via pseudo-inverse (`W @ pinv(B)`). This gives a least-squares optimal approximation with some reconstruction loss due to the dimensionality reduction.
- **AE MLP**: Randomly initialized. The AE bottleneck has fundamentally different architecture than SwiGLU, so no clean projection exists. Fine-tuning is required.
- **Everything else** (embeddings, layer norms, lm_head): Copied directly from the pretrained checkpoint.

## Llama 3.2-1B Architecture Reference

```mermaid
flowchart TB
    classDef replaceSlot fill:#fff1cf,stroke:#b7791f,stroke-width:2px,color:#4a2f00;
    classDef standard fill:#f7f7f7,stroke:#777,stroke-width:1px,color:#222;

    TOK["Input token IDs<br/>[batch x seq_len]"]
    ET["embed_tokens<br/>nn.Embedding(vocab_size=128256, hidden_size=2048)<br/>output: [batch x seq_len x 2048]"]

    subgraph Model["LlamaForCausalLM / LlamaModel"]
        direction TB
        DL["LlamaDecoderLayer x16"]:::standard
        SLOT_A["ATTN replacement slot<br/>q/k/v/o proj: Linear -> TrendWaveletLinear<br/>enabled when pe_attn_mode=trend_wavelet"]:::replaceSlot
        SLOT_M["MLP replacement slot<br/>LlamaMLP -> PEBottleneckMLP or PEBottleneckMLPLG<br/>enabled when pe_mlp_mode=ae or ae_lg"]:::replaceSlot
        DL -. contains .-> SLOT_A
        DL -. contains .-> SLOT_M
    end

    NORM["final norm<br/>LlamaRMSNorm<br/>2048"]
    LH["lm_head<br/>Linear 2048 -> 128256<br/>weight tied to embed_tokens"]
    OUT["Logits<br/>[batch x seq_len x vocab]"]

    TOK --> ET --> DL --> NORM --> LH --> OUT
```

### Decoder Layer with Explicit Replacement Slots

```mermaid
flowchart LR
    classDef replaceSlot fill:#fff1cf,stroke:#b7791f,stroke-width:2px,color:#4a2f00;
    classDef standard fill:#f7f7f7,stroke:#777,stroke-width:1px,color:#222;

    XIN["x<br/>hidden state<br/>2048"]
    IL["input_layernorm<br/>LlamaRMSNorm"]
    ATTN["self_attn<br/>LlamaSdpaAttention"]
    RES1["+ residual add"]
    PAL["post_attention_layernorm<br/>LlamaRMSNorm"]
    MLP["mlp<br/>LlamaMLP (SwiGLU)"]
    RES2["+ residual add"]
    XOUT["x'<br/>output<br/>2048"]

    QA["q_proj<br/>Linear 2048->2048"]:::replaceSlot
    KA["k_proj<br/>Linear 2048->512"]:::replaceSlot
    VA["v_proj<br/>Linear 2048->512"]:::replaceSlot
    OA["o_proj<br/>Linear 2048->2048"]:::replaceSlot

    XIN  --> IL --> ATTN --> RES1 --> PAL --> MLP --> RES2 --> XOUT
    XIN  -.->|"residual"| RES1
    RES1 -.->|"residual"| RES2
    ATTN --- QA
    ATTN --- KA
    ATTN --- VA
    ATTN --- OA
    TWNOTE["These 4 projection slots are replaced together<br/>by TrendWaveletLinear in trend_wavelet mode"]:::standard
    MLPNOTE["This MLP slot is replaced by<br/>PEBottleneckMLP or PEBottleneckMLPLG"]:::standard
    QA -.-> TWNOTE
    OA -.-> TWNOTE
    MLP -.-> MLPNOTE
```

### Unmodified Decoder Layer (Reference)

```mermaid
flowchart LR
    XIN0["x<br/>2048"] --> IL0["input_layernorm"] --> A0["self_attn<br/>LlamaSdpaAttention"] --> R10["+"] --> PL0["post_attention_layernorm"] --> M0["mlp<br/>LlamaMLP (SwiGLU)"] --> R20["+"] --> XOUT0["x'"]
    XIN0 -. residual .-> R10
    R10 -. residual .-> R20
    A0 --- Q00["q_proj Linear"]
    A0 --- K00["k_proj Linear"]
    A0 --- V00["v_proj Linear"]
    A0 --- O00["o_proj Linear"]
```

### Experimental Blocks and Replacement Points

```mermaid
flowchart TB
    classDef replaced fill:#fff1cf,stroke:#b7791f,stroke-width:2px,color:#4a2f00;
    classDef exp fill:#e9f7ff,stroke:#1d6fa5,stroke-width:1.5px,color:#123b56;
    classDef base fill:#f7f7f7,stroke:#777,stroke-width:1px,color:#222;

    subgraph STD["Standard Llama Decoder Layer (reference)"]
        direction LR
        SA["self_attn<br/>LlamaSdpaAttention"]:::base
        M0["mlp<br/>LlamaMLP (SwiGLU)"]:::base
        Q0["q_proj<br/>Linear slot"]:::replaced
        K0["k_proj<br/>Linear slot"]:::replaced
        V0["v_proj<br/>Linear slot"]:::replaced
        O0["o_proj<br/>Linear slot"]:::replaced
        SA --- Q0
        SA --- K0
        SA --- V0
        SA --- O0
    end

    subgraph EXP1["Experimental Attention Block"]
        direction TB
        TW["TrendWaveletLinear<br/>theta: trainable<br/>trend_basis + wavelet_basis: frozen"]:::exp
        TWQ["Used for q_proj, k_proj, v_proj, o_proj<br/>when pe_attn_mode=trend_wavelet"]:::exp
        TW --> TWQ
    end

    subgraph EXP2["Experimental MLP Blocks"]
        direction TB
        AE["PEBottleneckMLP<br/>hidden -> hidden/2 -> latent -> hidden/2 -> hidden"]:::exp
        AELG["PEBottleneckMLPLG<br/>AE bottleneck + learned sigmoid gate at latent"]:::exp
    end

    Q0 -.replaced by.-> TW
    K0 -.replaced by.-> TW
    V0 -.replaced by.-> TW
    O0 -.replaced by.-> TW
    M0 -.replaced by.-> AE
    M0 -.replaced by.-> AELG
```

### TrendWaveletLinear Block Internals

```mermaid
flowchart LR
    classDef input fill:#f7f7f7,stroke:#777,stroke-width:1px,color:#222;
    classDef train fill:#e9f7ff,stroke:#1d6fa5,stroke-width:1.5px,color:#123b56;
    classDef frozen fill:#eef8ee,stroke:#2f855a,stroke-width:1.5px,color:#1f5138;
    classDef mix fill:#fff1cf,stroke:#b7791f,stroke-width:2px,color:#4a2f00;

    X["x<br/>[..., in_features]"]:::input
    TH["theta<br/>Linear(in_features -> trend_dim + wavelet_dim)<br/>trainable"]:::train
    SPLIT["split coefficients"]:::mix
    TC["trend_c<br/>[..., trend_dim]"]:::train
    WC["wavelet_c<br/>[..., wavelet_dim_eff]"]:::train

    TB["trend_basis<br/>[trend_dim x out_features]<br/>frozen buffer"]:::frozen
    WB["wavelet_basis<br/>[wavelet_dim_eff x out_features]<br/>frozen buffer"]:::frozen

    TM["trend_c @ trend_basis<br/>[..., out_features]"]:::mix
    WM["wavelet_c @ wavelet_basis<br/>[..., out_features]"]:::mix
    OUT["output = trend_term + wavelet_term<br/>[..., out_features]"]:::input

    X --> TH --> SPLIT
    SPLIT --> TC
    SPLIT --> WC
    TC --> TM
    TB --> TM
    WC --> WM
    WB --> WM
    TM --> OUT
    WM --> OUT
```

### Standard Linear Projection Block (Reference)

In `standard` mode this is exactly a plain `nn.Linear` projection.

```mermaid
flowchart LR
    classDef input fill:#f7f7f7,stroke:#777,stroke-width:1px,color:#222;
    classDef train fill:#e9f7ff,stroke:#1d6fa5,stroke-width:1.5px,color:#123b56;
    classDef param fill:#fff1cf,stroke:#b7791f,stroke-width:2px,color:#4a2f00;

    X["x<br/>[..., in_features]"]:::input
    LIN["nn.Linear(in_features -> out_features)<br/>y = xW^T + b"]:::train
    Y["y<br/>[..., out_features]"]:::input
    W["weight W<br/>[out_features x in_features]<br/>trainable"]:::param
    B["bias b (optional)<br/>[out_features]<br/>trainable"]:::param

    X --> LIN --> Y
    W --> LIN
    B --> LIN
```

### MLP Block Internals (LlamaMLP vs PEBottleneckMLP)

```mermaid
flowchart TB
    classDef std fill:#f7f7f7,stroke:#777,stroke-width:1px,color:#222;
    classDef pe fill:#e9f7ff,stroke:#1d6fa5,stroke-width:1.5px,color:#123b56;

    subgraph MLP_COMPARE["Side-by-side MLP internals"]
        direction TB
        L0["LlamaMLP (SwiGLU)<br/>x [..., 2048]"]:::std
        R0["PEBottleneckMLP (AE)<br/>x [..., 2048]"]:::pe
        L0 --- R0

        L1["gate_proj: 2048->8192<br/>up_proj: 2048->8192"]:::std
        R1["fc1: 2048->1024<br/>SiLU"]:::pe
        L0 --> L1
        R0 --> R1
        L1 --- R1

        L2["SiLU(gate_proj(x)) * up_proj(x)"]:::std
        R2["fc2: 1024->latent<br/>SiLU"]:::pe
        L1 --> L2
        R1 --> R2
        L2 --- R2

        L3["down_proj: 8192->2048"]:::std
        R3["fc3: latent->1024<br/>SiLU"]:::pe
        L2 --> L3
        R2 --> R3
        L3 --- R3

        L4["y [..., 2048]"]:::std
        R4["fc4: 1024->2048<br/>y [..., 2048]"]:::pe
        L3 --> L4
        R3 --> R4
        L4 --- R4
    end
```

### PEBottleneckMLPLG Internals (Learned-Gate Variant)

```mermaid
flowchart TB
    classDef pe fill:#e9f7ff,stroke:#1d6fa5,stroke-width:1.5px,color:#123b56;
    classDef gate fill:#fff1cf,stroke:#b7791f,stroke-width:2px,color:#4a2f00;

    LX["x<br/>[... x hidden_size=2048]"]:::pe
    L1["fc1<br/>Linear 2048 -> 1024"]:::pe
    L1A["SiLU"]:::pe
    L2["fc2<br/>Linear 1024 -> latent_dim"]:::pe
    L2A["SiLU"]:::pe
    LG["sigmoid(latent_gate)<br/>learned parameter [latent_dim]"]:::gate
    LM["elementwise multiply<br/>z * sigmoid(latent_gate)"]:::gate
    L3["fc3<br/>Linear latent_dim -> 1024"]:::pe
    L3A["SiLU"]:::pe
    L4["fc4<br/>Linear 1024 -> 2048"]:::pe
    LY["output<br/>[... x 2048]"]:::pe
    LX --> L1 --> L1A --> L2 --> L2A --> LM --> L3 --> L3A --> L4 --> LY
    LG --> LM
```
