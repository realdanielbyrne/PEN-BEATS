#!/usr/bin/env python3
"""
Minimal fine-tuning + perplexity evaluation for PE-Llama models.

Usage::

    # Standard model (baseline perplexity)
    python scripts/finetune.py --pe-attn-mode standard --pe-mlp-mode standard --epochs 0

    # TrendWavelet attention only
    python scripts/finetune.py --pe-attn-mode trend_wavelet --epochs 3 --lr 1e-4

    # PE bottleneck MLP only
    python scripts/finetune.py --pe-mlp-mode vae_lg --ae-latent-dim 256 --epochs 3

    # Both TrendWavelet + PE MLP
    python scripts/finetune.py \
        --pe-attn-mode trend_wavelet --trend-dim 4 --wavelet-dim 28 \
        --pe-mlp-mode vae_lg --ae-latent-dim 256 \
        --epochs 3 --lr 1e-4 --batch-size 4

Requires: pip install pellm[train]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

_PE_MLP_CHOICES = (
    "standard",
    "ae",
    "ae_lg",
    "vae",
    "vae_lg",
    "ae_basis_latent",
    "ae_basis_reexpand",
    "tw_root",
    "tw_root_fc",
)
# Modes that support reconstruction-based AE pre-training and pretrained-weight
# initialization. ``ae_basis_*`` modes are excluded — they are from-scratch only.
_RECON_PE_MLP_MODES = ("ae", "ae_lg", "vae", "vae_lg")
_GATE_REPORT_PE_MLP_MODES = ("ae_lg", "vae", "vae_lg")


def print_learned_gate_values(model) -> None:
    """Print latent gate statistics for AE-LG/VAE-LG layers, and β for VAE layers."""
    from pellm.pe_layers import PEBottleneckMLPVAE, PEBottleneckMLPVAELG

    printed_gate_header = False

    for layer_idx, decoder_layer in enumerate(model.model.layers):
        mlp = decoder_layer.mlp

        # Latent gate (ae_lg / vae_lg)
        latent_gate = getattr(mlp, "latent_gate", None)
        if latent_gate is not None:
            if not printed_gate_header:
                print("\nLearned gate values (sigmoid) per layer:")
                printed_gate_header = True
            gate_values = torch.sigmoid(latent_gate)
            active_gate_count = (gate_values > 0.5).sum().item()
            print(
                f"  Layer {layer_idx:2d}: mean={gate_values.mean():.3f}  "
                f"min={gate_values.min():.3f}  max={gate_values.max():.3f}  "
                f"active(>0.5)={active_gate_count}/{len(gate_values)}"
            )


def _get_ae_target_layers(model) -> list[int]:
    """Return decoder layers whose MLP is one of the PE bottleneck families."""
    from pellm.pe_layers import (
        PEBottleneckMLP,
        PEBottleneckMLPLG,
        PEBottleneckMLPVAE,
        PEBottleneckMLPVAELG,
    )

    ae_types = (
        PEBottleneckMLP,
        PEBottleneckMLPLG,
        PEBottleneckMLPVAE,
        PEBottleneckMLPVAELG,
    )
    return [
        i
        for i, layer in enumerate(model.model.layers)
        if isinstance(layer.mlp, ae_types)
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PE-Llama fine-tuning & evaluation")

    # Model
    p.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base Llama model to load",
    )

    # PE attention config
    p.add_argument(
        "--pe-attn-mode",
        default="standard",
        choices=[
            "standard",
            "trend_wavelet",
            "trend_wavelet_generic",
            "trend_wavelet_lg",
            "trend_wavelet_generic_lg",
            "trend_wavelet_reduced",
            "trend_wavelet_lg_reduced",
            "trend_wavelet_generic_reduced",
            "trend_wavelet_generic_lg_reduced",
            "trend_wavelet_no_postln",
            "svd",
            "svd_lg",
        ],
    )
    p.add_argument("--trend-dim", type=int, default=3)
    p.add_argument("--wavelet-dim", type=int, default=28)
    p.add_argument("--wavelet-type", default="db3")
    p.add_argument("--wavelet-basis-offset", type=int, default=0)
    p.add_argument(
        "--per-layer-offsets",
        type=int,
        nargs="+",
        default=None,
        help="Per-layer basis offset overrides",
    )
    p.add_argument(
        "--svd-rank",
        type=int,
        default=32,
        help="Rank for SVD attention modes (svd, svd_lg). "
        "Equivalent rank budget to trend_dim + wavelet_dim.",
    )
    p.add_argument(
        "--generic-dim",
        type=int,
        default=5,
        help="Generic branch dimensions for trend_wavelet_generic modes",
    )
    p.add_argument(
        "--reduction-dim",
        type=int,
        default=None,
        help="Bottleneck dimension for reduced TrendWavelet modes. "
        "If None, defaults to trend_dim + wavelet_dim inside the layer.",
    )
    p.add_argument(
        "--attn-init",
        default="pretrained",
        choices=["pretrained", "lstsq", "svd", "cur", "fourier", "random"],
        help="TrendWavelet attention weight initialization. "
        "'pretrained' (default) directly truncates the dense "
        "weight into coefficient space; 'lstsq' preserves the "
        "existing least-squares basis projection; 'svd', 'cur', "
        "and 'fourier' preprocess the dense weight before the "
        "TrendWavelet projection; 'random' leaves default init.",
    )
    p.add_argument(
        "--active-g-pretrain",
        action="store_true",
        help="Apply SiLU activation to TrendWavelet output during attention pre-training",
    )
    p.add_argument(
        "--active-g-finetune",
        action="store_true",
        help="Apply SiLU activation to TrendWavelet output during LM fine-tuning",
    )
    p.add_argument(
        "--pe-projections",
        nargs="+",
        default=None,
        metavar="PROJ",
        help="Projection types to replace: q k v o (short) or full names. "
        "Default: all 4 projections.",
    )
    p.add_argument(
        "--pe-layer-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help="Decoder layer indices (0-based) to replace with PE attention. "
        "Default: all layers.",
    )

    # PE MLP config
    p.add_argument(
        "--pe-mlp-mode",
        default="standard",
        choices=list(_PE_MLP_CHOICES),
        help="PE MLP family: standard, ae, ae_lg, vae, or vae_lg.",
    )
    p.add_argument("--ae-latent-dim", type=int, default=256)
    p.add_argument(
        "--pe-mlp-layer-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help="Decoder layer indices (0-based) to replace with PE MLP. "
        "Default: all layers.",
    )
    p.add_argument(
        "--ae-init",
        default="pretrained",
        choices=["pretrained", "random", "svd", "cur", "fourier"],
        help="AE MLP weight initialization: 'pretrained' (default) uses "
        "truncation + partial SVD from LlamaMLP weights; 'svd' uses "
        "full SVD on all 4 matrices for optimal rank-k init; "
        "'cur' uses CUR decomposition (leverage-score row/column "
        "selection); 'fourier' uses FFT filtering to denoise "
        "truncated weights; 'random' uses random init.",
    )
    p.add_argument(
        "--ae-inner-init",
        default="svd",
        choices=["svd", "match"],
        help="fc2/fc3 init strategy: 'svd' (default) always uses SVD "
        "for the inner bottleneck layers regardless of --ae-init; "
        "'match' uses the same algorithm as fc1/fc4.",
    )

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience: stop training after this many "
        "epochs without validation perplexity improvement. "
        "Set to 0 to disable early stopping.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for tokenization",
    )
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument(
        "--max-eval-batches",
        type=int,
        default=50,
        help="Max batches for perplexity evaluation (0=all)",
    )

    # Staged training
    p.add_argument(
        "--resume-from",
        default=None,
        help="Path to a previously saved PE model for staged training. "
        "Loads the saved model and applies the current PE mode "
        "flags, preserving already-fitted weights.",
    )

    # Freezing
    p.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze embeddings, lm_head, and standard MLP layers. "
        "Only PE attention layers and layer norms remain trainable "
        "(Option C). PE MLP layers are always left trainable.",
    )
    p.add_argument(
        "--freeze-except-layers",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help="Freeze ALL parameters except those in the specified "
        "decoder layer indices (0-based). Freezes embeddings, "
        "lm_head, and every other decoder layer.",
    )

    # AE pre-training
    p.add_argument(
        "--ae-pretrain-epochs",
        type=int,
        default=0,
        help="Number of AE reconstruction pre-training epochs before LM fine-tuning. "
        "Only meaningful when --pe-mlp-mode is ae, ae_lg, vae, or vae_lg.",
    )
    p.add_argument(
        "--ae-pretrain-layer-indices",
        type=int,
        nargs="+",
        default=None,
        help="Subset of decoder layer indices to AE-pre-train (must be a subset of "
        "--pe-mlp-layer-indices). Useful when resuming a wave from a previous "
        "wave whose AE layers are already trained — restricting pre-training to "
        "only the newly-added layers avoids wasted activation extraction and "
        "training on already-converged layers. Defaults to all detected AE layers.",
    )
    p.add_argument(
        "--ae-pretrain-early-stopping",
        action="store_true",
        help="Enable early stopping for AE pre-training based on MSE loss.",
    )
    p.add_argument(
        "--ae-pretrain-warmup",
        type=int,
        default=0,
        help="Number of warmup epochs before early stopping can trigger "
        "during AE pre-training.",
    )
    p.add_argument(
        "--ae-pretrain-patience",
        type=int,
        default=3,
        help="Early stopping patience for AE pre-training: stop after this "
        "many epochs without MSE loss improvement.",
    )
    p.add_argument(
        "--ae-pretrain-lr",
        type=float,
        default=None,
        help="Separate learning rate for AE pre-training. Falls back to --lr if not set.",
    )
    p.add_argument(
        "--ae-pretrain-scheduler",
        default="none",
        choices=["none", "exponential"],
        help="LR scheduler for AE pre-training. 'exponential' applies ExponentialLR "
        "with --ae-pretrain-gamma decay factor.",
    )
    p.add_argument(
        "--ae-pretrain-lr-warmup",
        type=int,
        default=0,
        help="Number of linear LR warmup epochs for AE pre-training "
        "(ramp from ~0 to --ae-pretrain-lr). Distinct from --ae-pretrain-warmup "
        "which controls early-stopping warmup.",
    )
    p.add_argument(
        "--ae-pretrain-gamma",
        type=float,
        default=0.85,
        help="Exponential decay factor per epoch for AE pre-training LR scheduler.",
    )
    p.add_argument(
        "--ae-cache-dir",
        default=None,
        help="Directory for caching teacher activations during AE pre-training. "
        "If not set, a temporary directory is used and cleaned up automatically. "
        "When set, cached activations persist for reuse across runs.",
    )
    p.add_argument(
        "--ae-dataset",
        default="wikitext2",
        choices=["wikitext2", "fineweb"],
        help="Dataset for AE activation caching only (independent of --dataset).",
    )
    p.add_argument(
        "--ae-cache-num-samples",
        type=int,
        default=10000,
        help="Max raw text samples for AE caching with fineweb. "
        "Ignored for wikitext2.",
    )
    p.add_argument(
        "--ae-cache-skip",
        type=int,
        default=0,
        help="Number of FineWeb docs to skip before collecting ae-cache-num-samples. "
        "Use to give each wave a different slice of the dataset.",
    )
    p.add_argument(
        "--ae-pretrain-loss",
        default="huber",
        choices=["mse", "cosine", "huber", "soft_kl", "combined", "combined_huber"],
        help="Loss function for AE pre-training (default: huber). "
        "'soft_kl' applies Hinton-style temperature-softened KL divergence over "
        "the hidden-dim axis. 'combined' is alpha*MSE + (1-alpha)*cosine. "
        "'combined_huber' is alpha*Huber + (1-alpha)*cosine.",
    )
    p.add_argument(
        "--ae-pretrain-loss-temperature",
        type=float,
        default=2.0,
        help="Temperature T for --ae-pretrain-loss=soft_kl (default: 2.0). "
        "Ignored for other losses.",
    )
    p.add_argument(
        "--ae-pretrain-loss-alpha",
        type=float,
        default=0.5,
        help="MSE weight for --ae-pretrain-loss=combined: "
        "alpha*MSE + (1-alpha)*cosine (default: 0.5). Ignored for other losses.",
    )
    p.add_argument(
        "--ae-pretrain-resample",
        action="store_true",
        default=False,
        help="Re-stream a fresh FineWeb slice and re-extract teacher activations "
        "before each AE pre-training epoch. Requires --ae-dataset fineweb. "
        "Each epoch skips (epoch-1)*ae-cache-num-samples docs to avoid overlap. "
        "Ignored when ae-dataset is not fineweb.",
    )
    p.add_argument(
        "--ae-pretrain-fineweb-epochs",
        type=int,
        default=0,
        help="Number of leading AE pre-training epochs that use FineWeb per-epoch "
        "resampling. Remaining epochs (up to --ae-pretrain-epochs) switch to "
        "WikiText-2. 0 means all epochs use --ae-dataset (existing behavior).",
    )

    # Attention pre-training
    p.add_argument(
        "--attn-pretrain-epochs",
        type=int,
        default=0,
        help="Number of TrendWavelet attention reconstruction pre-training "
        "epochs before AE pre-training / LM fine-tuning.",
    )
    p.add_argument(
        "--attn-pretrain-early-stopping",
        action="store_true",
        help="Enable early stopping for attention pre-training based on MSE loss.",
    )
    p.add_argument(
        "--attn-pretrain-warmup",
        type=int,
        default=0,
        help="Number of warmup epochs before early stopping can trigger "
        "during attention pre-training.",
    )
    p.add_argument(
        "--attn-pretrain-patience",
        type=int,
        default=3,
        help="Early stopping patience for attention pre-training: stop after this "
        "many epochs without MSE loss improvement.",
    )
    p.add_argument(
        "--attn-cache-dir",
        default=None,
        help="Directory for caching teacher attention activations during "
        "TrendWavelet pre-training. If not set, a temporary directory "
        "is used and cleaned up automatically.",
    )
    p.add_argument(
        "--attn-dataset",
        default="wikitext2",
        choices=["wikitext2", "wikitext103", "fineweb"],
        help="Dataset for attention activation caching only, independent of "
        "--dataset.",
    )
    p.add_argument(
        "--attn-cache-num-samples",
        type=int,
        default=10000,
        help="Max raw text samples for attention caching with fineweb. "
        "Ignored for wikitext2 and wikitext103.",
    )

    # Main training / evaluation dataset
    p.add_argument(
        "--dataset",
        default="wikitext2",
        choices=["wikitext2", "wikitext103", "fineweb"],
        help="Main dataset for LM fine-tuning and perplexity evaluation. "
        "wikitext2 (default, ~2M tokens), wikitext103 (~103M tokens), "
        "or fineweb (streaming subset). "
        "Distinct from --ae-dataset which controls AE activation caching only.",
    )
    p.add_argument(
        "--dataset-num-samples",
        type=int,
        default=50000,
        help="Number of raw text samples to stream when --dataset fineweb. "
        "Train uses this count; validation and test each use 1/10 of it. "
        "Ignored for wikitext2 and wikitext103.",
    )

    # Output
    p.add_argument(
        "--output-dir", default=None, help="Directory to save the fine-tuned model"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, load model from local cache only, print "
        "parameter summary, and exit without training or evaluation.",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model/tokenizer from local HF cache; "
        "never contact the Hugging Face Hub. Useful when the "
        "model has already been downloaded.",
    )
    p.add_argument(
        "--shared-tokenizer-dir",
        default=None,
        help="Directory for shared tokenizer files. When set, "
        "tokenizer/template files are saved once here and "
        "symlinked into --output-dir to avoid duplication "
        "across checkpoints.",
    )
    p.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Sets torch, "
        "torch.cuda, and Python random seeds.",
    )

    # Logging
    p.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    p.add_argument(
        "--wandb-project", default="pellm", help="W&B project name (default: pellm)"
    )
    p.add_argument(
        "--log-csv",
        default=None,
        help="Path to append a CSV row with all metrics after training",
    )

    # AE teacher (for wave pipeline: use prior wave's model as activation source)
    p.add_argument(
        "--ae-teacher",
        default=None,
        help="Model path for AE pre-training activation extraction. "
        "When set, this model (instead of --model-name) is loaded "
        "as the teacher for AE Phase 1 activation caching. "
        "Useful for wave pipelines where earlier layers are already "
        "compressed. Defaults to --model-name (vanilla Llama).",
    )

    # Attention teacher (for wave pipeline: use prior wave's model as activation source)
    p.add_argument(
        "--attn-teacher",
        default=None,
        help="Model path for attention pre-training activation extraction. "
        "When set, this model (instead of --model-name) is loaded "
        "as the teacher for attention Phase 1 activation caching. "
        "Useful for wave pipelines where MLP layers are already "
        "compressed. Defaults to --model-name (vanilla Llama).",
    )

    # Knowledge distillation
    p.add_argument(
        "--kd-alpha",
        type=float,
        default=1.0,
        help="CE weight (0-1). KD weight = 1-alpha. " "Default 1.0 (KD disabled).",
    )
    p.add_argument(
        "--kd-temperature",
        type=float,
        default=2.0,
        help="Softmax temperature for KD loss (default: 2.0)",
    )
    p.add_argument(
        "--kd-teacher",
        default=None,
        help="Teacher model path/name (default: same as --model-name)",
    )
    p.add_argument(
        "--kd-attn-weight",
        type=float,
        default=0.0,
        help="Weight for attention pattern KD loss (default: 0.0 = disabled). "
        "Aligns softmax(QK^T/√d) between student and teacher on targeted layers.",
    )
    p.add_argument(
        "--kd-attn-layers",
        type=int,
        nargs="+",
        default=None,
        help="Decoder layers for attention pattern KD (default: pe_layer_indices).",
    )

    return p.parse_args()


class TokenizedDataset(Dataset):
    """Pre-tokenized dataset from HuggingFace datasets."""

    def __init__(self, encodings: dict):
        self.input_ids = encodings["input_ids"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def load_wikitext2(
    tokenizer, max_length: int, split: str = "train", local_files_only: bool = False
):
    """Load and tokenize WikiText-2."""
    from datasets import load_dataset
    from datasets import DownloadConfig

    dl_cfg = DownloadConfig(local_files_only=local_files_only)
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        download_config=dl_cfg,
    )

    # Filter empty lines and concatenate
    texts = [t for t in dataset["text"] if t.strip()]
    full_text = "\n".join(texts)

    # Tokenize into chunks
    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )

    # Split into max_length chunks
    input_ids = encodings["input_ids"].squeeze(0)
    n_chunks = len(input_ids) // max_length
    if n_chunks == 0:
        n_chunks = 1

    chunks = input_ids[: n_chunks * max_length].view(n_chunks, max_length)
    return TokenizedDataset({"input_ids": chunks})


def load_wikitext103(
    tokenizer, max_length: int, split: str = "train", local_files_only: bool = False
):
    """Load and tokenize WikiText-103 (~103M training tokens).

    Identical chunking logic to :func:`load_wikitext2` but uses the
    ``wikitext-103-raw-v1`` config which has ~50× more training text.
    Supports the same ``split`` values: ``"train"``, ``"validation"``, ``"test"``.
    """
    from datasets import load_dataset
    from datasets import DownloadConfig

    dl_cfg = DownloadConfig(local_files_only=local_files_only)
    dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split=split,
        download_config=dl_cfg,
    )

    texts = [t for t in dataset["text"] if t.strip()]
    full_text = "\n".join(texts)

    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )

    input_ids = encodings["input_ids"].squeeze(0)
    n_chunks = len(input_ids) // max_length
    if n_chunks == 0:
        n_chunks = 1

    chunks = input_ids[: n_chunks * max_length].view(n_chunks, max_length)
    logger.info(
        "WikiText-103 (%s): %d chunks of %d tokens.", split, n_chunks, max_length
    )
    return TokenizedDataset({"input_ids": chunks})


def load_fineweb_split(
    tokenizer,
    max_length: int,
    split: str = "train",
    num_train_samples: int = 50000,
    local_files_only: bool = False,
):
    """Load FineWeb as train / validation / test splits for LM fine-tuning.

    Streams ``num_train_samples`` documents for the training split, and
    ``num_train_samples // 10`` each for validation and test — using
    non-overlapping skip offsets so the three splits never share documents.

    Primary source: ``HuggingFaceFW/fineweb`` (sample-10BT config).
    Falls back to ``DKYoon/SlimPajama-6B`` then to WikiText-2 on failure.

    Tokenization and chunking are identical to :func:`load_wikitext2`.
    """
    from datasets import load_dataset  # noqa: PLC0415

    _FINEWEB_ID = "HuggingFaceFW/fineweb"
    _FINEWEB_CFG = "sample-10BT"
    _FALLBACK_ID = "DKYoon/SlimPajama-6B"

    split_sizes = {
        "train": num_train_samples,
        "validation": max(1, num_train_samples // 10),
        "test": max(1, num_train_samples // 10),
    }
    split_offsets = {
        "train": 0,
        "validation": num_train_samples,
        "test": num_train_samples + max(1, num_train_samples // 10),
    }
    if split not in split_sizes:
        raise ValueError(f"Unknown split '{split}'; expected train/validation/test.")

    n_docs = split_sizes[split]
    skip_n = split_offsets[split]

    logger.info(
        "Streaming %d FineWeb docs for split=%s (skip=%d) from %s/%s ...",
        n_docs,
        split,
        skip_n,
        _FINEWEB_ID,
        _FINEWEB_CFG,
    )
    try:
        ds = load_dataset(
            _FINEWEB_ID,
            name=_FINEWEB_CFG,
            split="train",
            streaming=True,
        )
        source_label = f"{_FINEWEB_ID}/{_FINEWEB_CFG}"
    except Exception as primary_exc:
        logger.warning(
            "Failed to load %s (%s). Falling back to %s.",
            _FINEWEB_ID,
            primary_exc,
            _FALLBACK_ID,
        )
        try:
            ds = load_dataset(_FALLBACK_ID, split="train", streaming=True)
            source_label = _FALLBACK_ID
        except Exception as fallback_exc:
            logger.error(
                "Failed to load fallback %s (%s). Falling back to WikiText-2.",
                _FALLBACK_ID,
                fallback_exc,
            )
            return load_wikitext2(
                tokenizer, max_length, split=split, local_files_only=local_files_only
            )

    ds = ds.skip(skip_n)
    texts = []
    for i, example in enumerate(ds):
        if i >= n_docs:
            break
        text = example.get("text", "")
        if text.strip():
            texts.append(text)

    logger.info(
        "Collected %d non-empty docs from %s (split=%s).",
        len(texts),
        source_label,
        split,
    )
    full_text = "\n".join(texts)

    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )

    input_ids = encodings["input_ids"].squeeze(0)
    n_chunks = len(input_ids) // max_length
    if n_chunks == 0:
        n_chunks = 1

    chunks = input_ids[: n_chunks * max_length].view(n_chunks, max_length)
    logger.info("FineWeb (%s): %d chunks of %d tokens.", split, n_chunks, max_length)
    return TokenizedDataset({"input_ids": chunks})


def load_fineweb(
    tokenizer,
    max_length: int,
    num_samples: int = 10000,
    skip: int = 0,
    local_files_only: bool = False,
):
    """Load and tokenize a subset of FineWeb (streaming).

    Primary source: ``HuggingFaceFW/fineweb`` (sample-10BT config).
    Fallback:       ``DKYoon/SlimPajama-6B`` if FineWeb is unreachable.

    Streams *num_samples* raw text documents (after skipping *skip* docs),
    concatenates, tokenizes, and chunks — same pattern as ``load_wikitext2``.
    """
    from datasets import load_dataset

    _FINEWEB_ID = "HuggingFaceFW/fineweb"
    _FINEWEB_CFG = "sample-10BT"
    _FALLBACK_ID = "DKYoon/SlimPajama-6B"

    logger.info(
        "Streaming %d samples (skip=%d) from %s (config=%s) ...",
        num_samples,
        skip,
        _FINEWEB_ID,
        _FINEWEB_CFG,
    )
    try:
        ds = load_dataset(
            _FINEWEB_ID,
            name=_FINEWEB_CFG,
            split="train",
            streaming=True,
        )
        source_label = f"{_FINEWEB_ID}/{_FINEWEB_CFG}"
    except Exception as primary_exc:
        logger.warning(
            "Failed to load %s (%s). " "Falling back to %s for AE activation caching.",
            _FINEWEB_ID,
            primary_exc,
            _FALLBACK_ID,
        )
        try:
            ds = load_dataset(
                _FALLBACK_ID,
                split="train",
                streaming=True,
            )
            source_label = _FALLBACK_ID
        except Exception as fallback_exc:
            logger.error(
                "Failed to load fallback dataset %s (%s). "
                "Falling back to WikiText-2.",
                _FALLBACK_ID,
                fallback_exc,
            )
            return load_wikitext2(tokenizer, max_length)

    if skip > 0:
        ds = ds.skip(skip)

    texts = []
    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        text = example.get("text", "")
        if text.strip():
            texts.append(text)

    logger.info("Collected %d non-empty texts from %s.", len(texts), source_label)
    full_text = "\n".join(texts)

    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )

    input_ids = encodings["input_ids"].squeeze(0)
    n_chunks = len(input_ids) // max_length
    if n_chunks == 0:
        n_chunks = 1

    chunks = input_ids[: n_chunks * max_length].view(n_chunks, max_length)
    logger.info("FineWeb: %d chunks of %d tokens.", n_chunks, max_length)
    return TokenizedDataset({"input_ids": chunks})


def freeze_base_parameters(model, pe_mlp_mode: str) -> None:
    """Freeze embeddings, lm_head, standard MLP, and non-PE attention projections.

    Keeps trainable:
    - PE attention parameters (theta, coeff_gate, generic_basis)
    - Layer norms
    - MLP parameters when pe_mlp_mode is a PE bottleneck family

    Freezes:
    - embed_tokens, lm_head
    - MLP when pe_mlp_mode is 'standard'
    - nn.Linear attention projections (non-targeted layers/projections hold
      pretrained weights and should not be trained)
    """
    from pellm.pe_layers import (
        TrendWaveletLinear,
        TrendWaveletGenericLinear,
        TrendWaveletLinearLG,
        TrendWaveletGenericLinearLG,
        TrendWaveletLinearReduced,
        TrendWaveletGenericLinearReduced,
        SVDLinear,
        SVDLinearLG,
    )

    _TW_CLASSES = (
        TrendWaveletLinear,
        TrendWaveletGenericLinear,
        TrendWaveletLinearLG,
        TrendWaveletGenericLinearLG,
        TrendWaveletLinearReduced,
        TrendWaveletGenericLinearReduced,
        SVDLinear,
        SVDLinearLG,
    )
    mlp_is_pe = pe_mlp_mode != "standard"

    # Build set of parameter full-names that belong to PE attention modules
    pe_attn_param_names: set[str] = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, _TW_CLASSES):
            for pname, _ in mod.named_parameters():
                pe_attn_param_names.add(f"{mod_name}.{pname}")

    for name, param in model.named_parameters():
        is_embed = "embed_tokens" in name
        is_lmhead = "lm_head" in name
        is_mlp = ".mlp." in name and not mlp_is_pe
        if is_embed or is_lmhead or is_mlp:
            param.requires_grad_(False)
        elif name in pe_attn_param_names:
            param.requires_grad_(True)
        elif ".self_attn." in name:
            # nn.Linear (pretrained weights, non-targeted) — freeze
            param.requires_grad_(False)
        else:
            # Layer norms, rotary embeddings, etc. — keep trainable
            param.requires_grad_(True)

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("freeze-base: %d trainable, %d frozen", trainable, frozen)


def freeze_except_layers(model, keep_layers: list[int]) -> None:
    """Freeze everything except the specified decoder layer indices.

    Freezes embeddings, lm_head, and all decoder layers not in *keep_layers*.
    Parameters inside *keep_layers* remain trainable.
    """
    keep_prefixes = [f"model.layers.{idx}." for idx in keep_layers]

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in keep_prefixes):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "freeze-except-layers %s: %d trainable, %d frozen",
        keep_layers,
        trainable,
        frozen,
    )


# Files produced by tokenizer.save_pretrained()
_TOKENIZER_FILES = (
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
)

# Files produced by model.save_pretrained() that are identical across
# checkpoints sharing the same base model.
_MODEL_SHARED_FILES = ("generation_config.json",)

_ALL_SHARED_FILES = _TOKENIZER_FILES + _MODEL_SHARED_FILES


def _symlink_files(
    file_names: tuple[str, ...],
    output_path: Path,
    shared_path: Path,
) -> None:
    """Replace *file_names* in *output_path* with relative symlinks to *shared_path*."""
    for fname in file_names:
        src = shared_path / fname
        dst = output_path / fname
        if src.exists() and dst.exists():
            dst.unlink()
            rel = os.path.relpath(src, output_path)
            dst.symlink_to(rel)


def save_checkpoint_with_symlinks(
    model,
    tokenizer,
    output_dir: str,
    shared_tokenizer_dir: str | None,
) -> None:
    """Save model + tokenizer, optionally symlinking shared files.

    When *shared_tokenizer_dir* is provided, tokenizer and generation
    config files are saved once to the shared directory and replaced
    with relative symlinks inside *output_dir*.  This avoids duplicating
    ~9 MB of identical data across every checkpoint directory.

    When *shared_tokenizer_dir* is ``None``, everything is saved
    directly into *output_dir* (standard HuggingFace behaviour).
    """
    output_path = Path(output_dir)

    # Always save model weights + config (these differ per checkpoint)
    model.save_pretrained(output_dir)

    if shared_tokenizer_dir is None:
        tokenizer.save_pretrained(output_dir)
        return

    shared_path = Path(shared_tokenizer_dir)
    shared_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer to the shared directory (idempotent)
    if not (shared_path / "tokenizer.json").exists():
        tokenizer.save_pretrained(str(shared_path))
        logger.info("Saved shared tokenizer to %s", shared_path)

    # Copy generation_config.json to shared dir if not already there
    gen_cfg = output_path / "generation_config.json"
    shared_gen_cfg = shared_path / "generation_config.json"
    if gen_cfg.exists() and not shared_gen_cfg.exists():
        import shutil

        shutil.copy2(gen_cfg, shared_gen_cfg)

    # Save tokenizer into output_dir (creates the files we'll replace)
    tokenizer.save_pretrained(output_dir)

    # Replace all shared files with symlinks
    _symlink_files(_ALL_SHARED_FILES, output_path, shared_path)
    logger.info("Symlinked shared files in %s → %s", output_path, shared_path)


def count_parameters(model) -> dict:
    """Count parameters by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # Per-component breakdown
    components = {}
    for name, p in model.named_parameters():
        # Extract component type from name
        if "self_attn" in name:
            key = "attention"
        elif "mlp" in name:
            key = "mlp"
        elif "embed" in name:
            key = "embeddings"
        elif "lm_head" in name:
            key = "lm_head"
        elif "norm" in name:
            key = "norms"
        else:
            key = "other"

        if key not in components:
            components[key] = {"total": 0, "trainable": 0}
        components[key]["total"] += p.numel()
        if p.requires_grad:
            components[key]["trainable"] += p.numel()

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "components": components,
    }


def print_param_summary(param_info: dict):
    """Print a formatted parameter summary."""
    print("\n" + "=" * 60)
    print("Parameter Summary")
    print("=" * 60)
    print(f"  Total:     {param_info['total']:>12,}")
    print(f"  Trainable: {param_info['trainable']:>12,}")
    print(f"  Frozen:    {param_info['frozen']:>12,}")
    print("-" * 60)
    for comp, info in sorted(param_info["components"].items()):
        pct = 100 * info["total"] / param_info["total"]
        print(
            f"  {comp:<15s} {info['total']:>12,}  ({pct:5.1f}%)  "
            f"trainable: {info['trainable']:,}"
        )
    print("=" * 60 + "\n")


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, max_batches: int = 0) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        # Loss is per-token cross-entropy (averaged over non-padding tokens)
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
        n_batches += 1

        if max_batches > 0 and n_batches >= max_batches:
            break

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg_loss)


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    grad_accum_steps: int,
    teacher_model=None,
    kd_alpha: float = 1.0,
    kd_temperature: float = 2.0,
    teacher_device=None,
    kd_attn_weight: float = 0.0,
    kd_attn_layers: list[int] | None = None,
) -> float:
    """Train for one epoch, return average CE loss.

    When *teacher_model* is provided and *kd_alpha* < 1.0, the loss becomes:
        ``kd_alpha * CE + (1 - kd_alpha) * KD + kd_attn_weight * attn_KD + kl_total``
    where KD is the KL-divergence between temperature-softened student and
    teacher logits (Hinton et al., 2015), and attn_KD is the KL-divergence
    between student and teacher attention patterns on targeted layers.
    """
    import torch.nn.functional as _F
    from pellm.pe_layers import PEBottleneckMLPVAE, PEBottleneckMLPVAELG

    _VAE_TYPES = (PEBottleneckMLPVAE, PEBottleneckMLPVAELG)

    use_kd = teacher_model is not None and kd_alpha < 1.0
    use_attn_kd = (
        teacher_model is not None
        and kd_attn_weight > 0.0
        and kd_attn_layers is not None
        and len(kd_attn_layers) > 0
    )
    # Need teacher forward when either logit KD or attention KD is active
    need_teacher = use_kd or use_attn_kd
    # Request attention outputs only when attention pattern KD is active
    output_attentions = use_attn_kd

    if teacher_device is None:
        teacher_device = device

    model.train()
    total_loss = 0.0
    total_kd_loss = 0.0
    total_attn_kd_loss = 0.0
    n_steps = 0
    n_batches = len(dataloader)
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            output_attentions=output_attentions,
        )
        ce_loss = outputs.loss

        # Collect VAE KL losses from all active VAE MLP layers (no-op for non-VAE modes).
        kl_total = sum(
            layer.mlp.kl_loss
            for layer in model.model.layers
            if isinstance(layer.mlp, _VAE_TYPES)
        )

        # Teacher forward pass (shared by logit KD and attention KD)
        kd_loss = torch.tensor(0.0, device=device)
        attn_kd_loss = torch.tensor(0.0, device=device)
        if need_teacher:
            with torch.no_grad():
                teacher_out = teacher_model(
                    input_ids=input_ids.to(teacher_device),
                    output_attentions=output_attentions,
                )

            # Logit KD
            if use_kd:
                teacher_logits = teacher_out.logits.to(device)
                T = kd_temperature
                student_log_probs = _F.log_softmax(outputs.logits / T, dim=-1)
                teacher_probs = _F.softmax(teacher_logits / T, dim=-1)
                kd_loss = _F.kl_div(
                    student_log_probs, teacher_probs, reduction="batchmean"
                ) * (T * T)
                total_kd_loss += kd_loss.item()

            # Attention pattern KD: KL(student_attn || teacher_attn) on target layers
            if use_attn_kd:
                student_attns = outputs.attentions  # tuple of (batch, heads, seq, seq)
                teacher_attns = teacher_out.attentions
                attn_kd_losses = []
                for layer_idx in kd_attn_layers:
                    s_attn = student_attns[layer_idx]  # (batch, heads, seq, seq)
                    t_attn = teacher_attns[layer_idx].to(device)
                    # Clamp to avoid log(0); teacher attns are already probabilities
                    s_attn_clamped = s_attn.clamp(min=1e-8)
                    t_attn_clamped = t_attn.clamp(min=1e-8)
                    # KL divergence per attention head, averaged over heads and layers
                    layer_kd = _F.kl_div(
                        s_attn_clamped.log(),
                        t_attn_clamped,
                        reduction="batchmean",
                    )
                    attn_kd_losses.append(layer_kd)
                attn_kd_loss = torch.stack(attn_kd_losses).mean()
                total_attn_kd_loss += attn_kd_loss.item()

        # Compose final loss
        if use_kd or use_attn_kd:
            loss = kd_alpha * ce_loss + kl_total
            if use_kd:
                loss = loss + (1.0 - kd_alpha) * kd_loss
            if use_attn_kd:
                loss = loss + kd_attn_weight * attn_kd_loss
            loss = loss / grad_accum_steps
        else:
            loss = (ce_loss + kl_total) / grad_accum_steps

        loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            n_steps += 1

        total_loss += ce_loss.item()

        # Log progress every 500 batches or at the end
        if (i + 1) % 500 == 0 or (i + 1) == n_batches:
            avg_loss_so_far = total_loss / (i + 1)
            if use_kd or use_attn_kd:
                log_parts = [f"Batch {i+1}/{n_batches}  ce={ce_loss.item():.4f}"]
                if use_kd:
                    log_parts.append(f"kd={kd_loss.item():.4f}")
                if use_attn_kd:
                    log_parts.append(f"attn_kd={attn_kd_loss.item():.4f}")
                log_parts.append(f"avg_ce={avg_loss_so_far:.4f}  steps={n_steps}")
                logger.info("  %s", "  ".join(log_parts))
            else:
                logger.info(
                    "  Batch %d/%d  loss=%.4f  avg_loss=%.4f  steps=%d",
                    i + 1,
                    n_batches,
                    ce_loss.item(),
                    avg_loss_so_far,
                    n_steps,
                )

    # Handle remaining gradients
    if (i + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        n_steps += 1

    return total_loss / (i + 1)


def _compute_ae_loss(pred, target, loss_type="mse", temperature=2.0, alpha=0.5):
    """Compute AE reconstruction loss with configurable loss function.

    Parameters
    ----------
    pred : Tensor
        AE output, shape ``(batch*seq, hidden_dim)``.
    target : Tensor
        Teacher activation. Cast to ``pred.dtype`` internally.
    loss_type : str
        One of ``"mse"``, ``"cosine"``, ``"huber"``, ``"soft_kl"``, ``"combined"``.
    temperature : float
        Temperature ``T`` for ``soft_kl``; logits are divided by ``T`` before
        softmax and the loss is scaled by ``T**2`` (Hinton et al., 2015).
    alpha : float
        Weight on MSE for ``combined``: ``alpha*MSE + (1-alpha)*cosine``.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    target = target.to(pred.dtype)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    elif loss_type == "cosine":
        return 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    elif loss_type == "huber":
        return F.smooth_l1_loss(pred, target)
    elif loss_type == "soft_kl":
        T = temperature
        log_p = F.log_softmax(pred / T, dim=-1)
        q = F.softmax(target / T, dim=-1)
        return F.kl_div(log_p, q, reduction="batchmean") * (T * T)
    elif loss_type == "combined":
        mse_part = F.mse_loss(pred, target)
        cos_part = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
        return alpha * mse_part + (1 - alpha) * cos_part
    elif loss_type == "combined_huber":
        huber_part = F.smooth_l1_loss(pred, target)
        cos_part = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
        return alpha * huber_part + (1 - alpha) * cos_part
    else:
        raise ValueError(f"Unknown ae_pretrain_loss: {loss_type}")


def train_ae_epoch(
    pe_model,
    teacher_model,
    dataloader,
    optimizer,
    device,
    grad_accum_steps,
    teacher_device=None,
    loss_type="mse",
    loss_temperature=2.0,
    loss_alpha=0.5,
):
    """AE reconstruction pre-training: teach each PEBottleneckMLP to reconstruct
    the full SwiGLU output of the corresponding original LlamaMLP.

    Teacher activations (input x and target y) are captured via forward hooks.
    Only PEBottleneckMLP layers are updated; attention is implicitly frozen.

    *teacher_device* is the device where the teacher model lives.  When the
    teacher is on a separate GPU (e.g. cuda:1) the input batch is sent there
    before the teacher forward pass; captured activations are then moved back
    to *device* (the PE model's device) for the AE loss computation.
    If *teacher_device* is None it defaults to *device*.
    """
    if teacher_device is None:
        teacher_device = device
    from pellm.pe_layers import PEBottleneckMLP, PEBottleneckMLPLG

    pe_model.train()
    teacher_model.eval()

    n_layers = len(teacher_model.model.layers)
    captured_x = {}
    captured_y = {}

    hooks = []
    for i in range(n_layers):

        def _make_hooks(idx):
            def _pre(module, inp):
                captured_x[idx] = inp[0].detach()

            def _post(module, inp, out):
                captured_y[idx] = out.detach()

            return _pre, _post

        pre_h, post_h = _make_hooks(i)
        hooks.append(teacher_model.model.layers[i].mlp.register_forward_pre_hook(pre_h))
        hooks.append(teacher_model.model.layers[i].mlp.register_forward_hook(post_h))

    total_loss = 0.0
    step = 0
    optimizer.zero_grad()

    try:
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                teacher_model(input_ids=input_ids.to(teacher_device))

            ae_loss = torch.tensor(0.0, device=device)
            for i, layer in enumerate(pe_model.model.layers):
                if isinstance(layer.mlp, (PEBottleneckMLP, PEBottleneckMLPLG)):
                    x = captured_x[i].to(device)
                    y = captured_y[i].to(device)
                    pred = layer.mlp(x.to(layer.mlp.fc1.weight.dtype))
                    ae_loss = ae_loss + _compute_ae_loss(
                        pred,
                        y,
                        loss_type=loss_type,
                        temperature=loss_temperature,
                        alpha=loss_alpha,
                    )

            (ae_loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += ae_loss.item()
    finally:
        for h in hooks:
            h.remove()

    if (step + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / (step + 1)


def extract_teacher_activations(
    teacher_model,
    dataloader,
    target_layer_indices,
    cache_dir,
    device,
    teacher_device=None,
):
    """Phase 1 of two-phase AE pre-training: extract and cache teacher MLP activations.

    Runs the frozen teacher model once over the entire training set, capturing
    the input and output activations of each targeted MLP layer via forward hooks.
    Activations are flattened to ``(batch*seq_len, hidden_size)`` and saved as
    ``.pt`` files to *cache_dir*.

    After this function returns, the teacher model can be deleted to free memory
    before Phase 2 (AE training from cache).

    Args:
        teacher_model: Frozen teacher LlamaForCausalLM.
        dataloader: Training DataLoader.
        target_layer_indices: List of decoder layer indices whose MLP activations
            to capture.  If ``None``, captures all layers.
        cache_dir: Directory to save activation files.
        device: Device for input tensors (when teacher_device is None).
        teacher_device: Device where the teacher model lives (defaults to *device*).

    Returns:
        Number of batches extracted.
    """
    if teacher_device is None:
        teacher_device = device

    os.makedirs(cache_dir, exist_ok=True)
    teacher_model.eval()

    n_layers = len(teacher_model.model.layers)
    if target_layer_indices is None:
        target_layer_indices = list(range(n_layers))

    captured_x: dict[int, torch.Tensor] = {}
    captured_y: dict[int, torch.Tensor] = {}

    hooks = []
    for i in target_layer_indices:

        def _make_hooks(idx):
            def _pre(module, inp):
                captured_x[idx] = inp[0].detach()

            def _post(module, inp, out):
                captured_y[idx] = out.detach()

            return _pre, _post

        pre_h, post_h = _make_hooks(i)
        hooks.append(teacher_model.model.layers[i].mlp.register_forward_pre_hook(pre_h))
        hooks.append(teacher_model.model.layers[i].mlp.register_forward_hook(post_h))

    n_batches = 0
    try:
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(teacher_device)

            with torch.no_grad():
                teacher_model(input_ids=input_ids)

            # Save activations per layer, flattened to (N, hidden)
            for idx in target_layer_indices:
                x = captured_x[idx].cpu()  # (batch, seq, hidden)
                y = captured_y[idx].cpu()
                # Flatten batch and sequence dimensions; clone to own storage
                # so torch.save's zip writer doesn't hit shared-storage races.
                x = x.view(-1, x.shape[-1]).clone()
                y = y.view(-1, y.shape[-1]).clone()
                fpath = os.path.join(cache_dir, f"layer{idx}_batch{step}.pt")
                torch.save({"x": x, "y": y}, fpath)

            captured_x.clear()
            captured_y.clear()
            n_batches += 1
    finally:
        for h in hooks:
            h.remove()

    logger.info(
        "Extracted teacher activations: %d batches × %d layers → %s",
        n_batches,
        len(target_layer_indices),
        cache_dir,
    )
    return n_batches


def _cache_is_valid(cache_dir: str, target_layer_indices: list[int]) -> bool:
    """Check whether *cache_dir* already contains activation files for all target layers.

    Returns ``True`` if at least one ``layerN_batch*.pt`` file exists for
    every layer index in *target_layer_indices*, meaning a prior extraction
    populated this directory and it can be reused without re-running Phase 1.
    """
    if not os.path.isdir(cache_dir):
        return False
    for idx in target_layer_indices:
        matches = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith(f"layer{idx}_batch") and f.endswith(".pt")
        ]
        if not matches:
            return False
    return True


_ATTN_CACHE_VERSION = 1
_ALL_ATTN_PROJS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def _attn_cache_metadata_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "metadata.json")


def _read_attn_cache_metadata(cache_dir: str) -> dict | None:
    path = _attn_cache_metadata_path(cache_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_attn_cache_metadata(cache_dir: str, metadata: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(_attn_cache_metadata_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _attn_cache_matches_metadata(cache_dir: str, expected_metadata: dict) -> bool:
    metadata = _read_attn_cache_metadata(cache_dir)
    if metadata is None:
        return False
    compare_keys = (
        "version",
        "model_name",
        "dataset",
        "max_length",
        "batch_size",
        "cache_num_samples",
    )
    return all(metadata.get(k) == expected_metadata.get(k) for k in compare_keys)


def _attn_cached_layers(cache_dir: str) -> set[int]:
    metadata = _read_attn_cache_metadata(cache_dir) or {}
    return {int(idx) for idx in metadata.get("cached_layers", [])}


def _attn_cache_is_valid(
    cache_dir: str, expected_metadata: dict, target_layer_indices: list[int]
) -> bool:
    """Check whether attention cache metadata matches and covers all layers."""
    if not os.path.isdir(cache_dir):
        return False
    if not _attn_cache_matches_metadata(cache_dir, expected_metadata):
        return False
    cached_layers = _attn_cached_layers(cache_dir)
    if not set(target_layer_indices).issubset(cached_layers):
        return False
    for idx in target_layer_indices:
        matches = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith(f"layer{idx}_batch") and f.endswith(".pt")
        ]
        if not matches:
            return False
    return True


def extract_attn_teacher_activations(
    teacher_model,
    dataloader,
    target_layer_indices,
    cache_dir,
    device,
    teacher_device=None,
):
    """Phase 1 attention pre-training: cache teacher projection activations."""
    if teacher_device is None:
        teacher_device = device

    os.makedirs(cache_dir, exist_ok=True)
    teacher_model.eval()

    captured: dict[int, dict] = {}
    hooks = []
    for layer_idx in target_layer_indices:
        attn = teacher_model.model.layers[layer_idx].self_attn

        def _make_qkv_pre(idx):
            def _pre(module, inp):
                captured.setdefault(idx, {"outputs": {}})["qkv_x"] = inp[0].detach()

            return _pre

        def _make_o_pre(idx):
            def _pre(module, inp):
                captured.setdefault(idx, {"outputs": {}})["o_x"] = inp[0].detach()

            return _pre

        def _make_out(idx, proj_name):
            def _post(module, inp, out):
                captured.setdefault(idx, {"outputs": {}})["outputs"][
                    proj_name
                ] = out.detach()

            return _post

        hooks.append(attn.q_proj.register_forward_pre_hook(_make_qkv_pre(layer_idx)))
        hooks.append(attn.o_proj.register_forward_pre_hook(_make_o_pre(layer_idx)))
        for proj_name in _ALL_ATTN_PROJS:
            hooks.append(
                getattr(attn, proj_name).register_forward_hook(
                    _make_out(layer_idx, proj_name)
                )
            )

    n_batches = 0
    try:
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(teacher_device)

            with torch.no_grad():
                teacher_model(input_ids=input_ids)

            for layer_idx in target_layer_indices:
                layer_capture = captured[layer_idx]
                # clone() after view() to give each tensor its own storage,
                # preventing torch.save zip-writer corruption from shared storage.
                qkv_x = (
                    layer_capture["qkv_x"]
                    .cpu()
                    .view(-1, layer_capture["qkv_x"].shape[-1])
                    .clone()
                )
                o_x = (
                    layer_capture["o_x"]
                    .cpu()
                    .view(-1, layer_capture["o_x"].shape[-1])
                    .clone()
                )
                outputs = {
                    proj_name: out.cpu().view(-1, out.shape[-1]).clone()
                    for proj_name, out in layer_capture["outputs"].items()
                }
                payload = {
                    "qkv_x": qkv_x,
                    "o_x": o_x,
                    "outputs": outputs,
                    "available_projs": sorted(outputs.keys()),
                    "shape_meta": {
                        "tokens": qkv_x.shape[0],
                        "qkv_in_features": qkv_x.shape[-1],
                        "o_in_features": o_x.shape[-1],
                        "output_dims": {
                            proj_name: tensor.shape[-1]
                            for proj_name, tensor in outputs.items()
                        },
                    },
                }
                fpath = os.path.join(cache_dir, f"layer{layer_idx}_batch{step}.pt")
                torch.save(payload, fpath)

            captured.clear()
            n_batches += 1
    finally:
        for h in hooks:
            h.remove()

    logger.info(
        "Extracted attention activations: %d batches × %d layers → %s",
        n_batches,
        len(target_layer_indices),
        cache_dir,
    )
    return n_batches


class AttnActivationCacheDataset(Dataset):
    """Dataset that loads cached attention activation bundles from disk."""

    def __init__(self, cache_dir: str, layer_idx: int):
        self.files = sorted(
            [
                os.path.join(cache_dir, f)
                for f in os.listdir(cache_dir)
                if f.startswith(f"layer{layer_idx}_batch") and f.endswith(".pt")
            ],
        )
        if not self.files:
            raise FileNotFoundError(
                f"No cached attention activations found for layer {layer_idx} in {cache_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], map_location="cpu", weights_only=True)


def train_attn_pretrain_epoch_from_cache(
    pe_model,
    cache_dir,
    target_layer_indices,
    target_proj_names,
    optimizer,
    device,
    grad_accum_steps,
):
    """Phase 2 attention pre-training: train TrendWavelet projections from cache."""
    from pellm.pe_layers import (
        TrendWaveletLinear,
        TrendWaveletGenericLinear,
        TrendWaveletLinearLG,
        TrendWaveletGenericLinearLG,
    )

    _TW_TYPES = (
        TrendWaveletLinear,
        TrendWaveletGenericLinear,
        TrendWaveletLinearLG,
        TrendWaveletGenericLinearLG,
    )
    pe_model.train()

    total_loss = 0.0
    total_steps = 0
    optimizer.zero_grad()

    for layer_idx in target_layer_indices:
        attn = pe_model.model.layers[layer_idx].self_attn
        cache_dataset = AttnActivationCacheDataset(cache_dir, layer_idx)
        cache_loader = DataLoader(cache_dataset, batch_size=None, shuffle=True)
        n_batches = len(cache_loader)

        logger.info(
            "  Attn layer %d: training from %d cached batches", layer_idx, n_batches
        )

        for step, cache_item in enumerate(cache_loader):
            proj_losses = []
            for proj_name in target_proj_names:
                proj_module = getattr(attn, proj_name)
                if not isinstance(proj_module, _TW_TYPES):
                    continue
                x_key = "o_x" if proj_name == "o_proj" else "qkv_x"
                x = cache_item[x_key].to(device)
                y = cache_item["outputs"][proj_name].to(device)
                pred = proj_module(x.to(proj_module.theta.weight.dtype))
                proj_losses.append(F.mse_loss(pred, y.to(pred.dtype)))

            if not proj_losses:
                continue

            loss = torch.stack(proj_losses).mean()
            (loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_steps += 1

            if (step + 1) % 500 == 0 or (step + 1) == n_batches:
                logger.info(
                    "    Batch %d/%d  loss=%.6f",
                    step + 1,
                    n_batches,
                    loss.item(),
                )

        if n_batches and n_batches % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(total_steps, 1)


class ActivationCacheDataset(Dataset):
    """Dataset that loads cached (input, output) activation pairs from disk.

    Each element is a dict ``{"x": Tensor, "y": Tensor}`` where both tensors
    have shape ``(n_tokens, hidden_size)`` (already flattened across batch and
    sequence dimensions by :func:`extract_teacher_activations`).
    """

    def __init__(self, cache_dir: str, layer_idx: int):
        self.files = sorted(
            [
                os.path.join(cache_dir, f)
                for f in os.listdir(cache_dir)
                if f.startswith(f"layer{layer_idx}_batch") and f.endswith(".pt")
            ],
        )
        if not self.files:
            raise FileNotFoundError(
                f"No cached activations found for layer {layer_idx} in {cache_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], map_location="cpu", weights_only=True)


def train_ae_epoch_from_cache(
    pe_model,
    cache_dir,
    target_layer_indices,
    optimizer,
    device,
    grad_accum_steps,
    loss_type="mse",
    loss_temperature=2.0,
    loss_alpha=0.5,
):
    """Phase 2 of two-phase AE pre-training: train AE layers from cached activations.

    Iterates over each targeted MLP layer independently, loading cached
    ``(input, output)`` activation pairs from disk and training the
    ``PEBottleneckMLP`` to reconstruct the teacher's output.

    No teacher model is needed — only the PE model's AE layers are in memory.

    Args:
        pe_model: The PE model containing PEBottleneckMLP layers.
        cache_dir: Directory containing cached activation files.
        target_layer_indices: List of decoder layer indices with AE MLPs.
            If ``None``, discovers layers from the PE model.
        optimizer: Optimizer for AE parameters.
        device: Device for AE computation.
        grad_accum_steps: Gradient accumulation steps.
        loss_type: Reconstruction loss function — one of ``"mse"``, ``"cosine"``,
            ``"huber"``, ``"soft_kl"``, ``"combined"`` (see :func:`_compute_ae_loss`).
        loss_temperature: Temperature for ``soft_kl``.
        loss_alpha: MSE weight for ``combined``.

    Returns:
        Average reconstruction loss across all layers and batches.
    """
    from pellm.pe_layers import (
        PEBottleneckMLP,
        PEBottleneckMLPLG,
        PEBottleneckMLPVAE,
        PEBottleneckMLPVAELG,
    )

    _AE_TYPES = (
        PEBottleneckMLP,
        PEBottleneckMLPLG,
        PEBottleneckMLPVAE,
        PEBottleneckMLPVAELG,
    )
    _VAE_TYPES = (PEBottleneckMLPVAE, PEBottleneckMLPVAELG)

    pe_model.train()

    # Discover target layers from the model if not specified
    if target_layer_indices is None:
        target_layer_indices = [
            i
            for i, layer in enumerate(pe_model.model.layers)
            if isinstance(layer.mlp, _AE_TYPES)
        ]

    total_loss = 0.0
    total_steps = 0
    optimizer.zero_grad()

    for layer_idx in target_layer_indices:
        layer = pe_model.model.layers[layer_idx]
        if not isinstance(layer.mlp, _AE_TYPES):
            continue

        cache_dataset = ActivationCacheDataset(cache_dir, layer_idx)
        # batch_size=1 because each file already contains a full batch of tokens
        cache_loader = DataLoader(cache_dataset, batch_size=1, shuffle=True)
        n_batches = len(cache_loader)

        logger.info(
            "  AE Layer %d: training from %d cached batches", layer_idx, n_batches
        )

        for step, batch_data in enumerate(cache_loader):
            # batch_data has shape (1, n_tokens, hidden) from DataLoader batching
            x = batch_data["x"].squeeze(0).to(device)
            y = batch_data["y"].squeeze(0).to(device)

            pred = layer.mlp(x.to(layer.mlp.fc1.weight.dtype))
            recon_loss = _compute_ae_loss(
                pred,
                y,
                loss_type=loss_type,
                temperature=loss_temperature,
                alpha=loss_alpha,
            )
            if isinstance(layer.mlp, _VAE_TYPES):
                loss = recon_loss + layer.mlp.kl_loss
            else:
                loss = recon_loss

            (loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_steps += 1

            # Log progress every 500 batches or at the end
            if (step + 1) % 500 == 0 or (step + 1) == n_batches:
                logger.info(
                    "    Batch %d/%d  loss=%.6f", step + 1, n_batches, loss.item()
                )

        # Flush remaining gradients for this layer
        if (step + 1) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(total_steps, 1)


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    if args.seed is not None:
        import random

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Wandb init (lazy import — only required when --wandb is set)
    wandb_run = None
    if args.wandb:
        import wandb as _wandb

        wandb_run = _wandb.init(
            project=args.wandb_project,
            config={
                k: getattr(args, k)
                for k in [
                    "pe_attn_mode",
                    "pe_mlp_mode",
                    "trend_dim",
                    "wavelet_dim",
                    "wavelet_type",
                    "svd_rank",
                    "generic_dim",
                    "attn_init",
                    "ae_latent_dim",
                    "pe_layer_indices",
                    "pe_mlp_layer_indices",
                    "epochs",
                    "lr",
                    "batch_size",
                    "grad_accum_steps",
                    "freeze_base",
                    "dtype",
                    "resume_from",
                    "attn_pretrain_epochs",
                    "ae_pretrain_epochs",
                    "ae_init",
                    "ae_inner_init",
                    "seed",
                    "active_g_pretrain",
                    "active_g_finetune",
                ]
            },
            name=Path(args.output_dir).name if args.output_dir else None,
        )

    # Resolve local-files-only: explicit flag OR dry-run implies it
    _local_only: bool = args.local_files_only or args.dry_run

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Determine device and device_map
    if torch.cuda.is_available():
        device_map = "auto"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_map = {"": "mps"}
    else:
        device_map = {"": "cpu"}
    logger.info("Using device_map: %s", device_map)

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN — loading from local cache only.")
        print("No datasets will be loaded; no training or evaluation.")
        print("=" * 60 + "\n")

    # Import pellm (triggers HF registration)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import pellm  # noqa: F401
    from pellm import PELlamaConfig, PELlamaForCausalLM
    from pellm.modeling_pe_llama import _resolve_proj_names

    # Resolve short projection names (q/k/v/o → q_proj/k_proj/v_proj/o_proj)
    pe_proj_names = (
        _resolve_proj_names(args.pe_projections) if args.pe_projections else None
    )

    # Build PE config
    pe_config = PELlamaConfig.from_pretrained(
        args.model_name,
        pe_attn_mode=args.pe_attn_mode,
        trend_dim=args.trend_dim,
        wavelet_dim=args.wavelet_dim,
        wavelet_type=args.wavelet_type,
        wavelet_basis_offset=args.wavelet_basis_offset,
        per_layer_offsets=args.per_layer_offsets,
        generic_dim=args.generic_dim,
        svd_rank=args.svd_rank,
        attn_init_mode=args.attn_init,
        pe_proj_names=pe_proj_names,
        pe_layer_indices=args.pe_layer_indices,
        pe_mlp_mode=args.pe_mlp_mode,
        ae_latent_dim=args.ae_latent_dim,
        pe_mlp_layer_indices=args.pe_mlp_layer_indices,
        ae_init_mode=args.ae_init,
        ae_inner_init=args.ae_inner_init,
        active_g=args.active_g_pretrain,
        local_files_only=_local_only,
    )

    logger.info(
        "PE Config: attn=%s, mlp=%s, projs=%s, attn_layers=%s, mlp_layers=%s",
        pe_config.pe_attn_mode,
        pe_config.pe_mlp_mode,
        pe_config.pe_proj_names or "all",
        pe_config.pe_layer_indices or "all",
        pe_config.pe_mlp_layer_indices or "all",
    )

    # Load tokenizer and datasets early so we can run pre-swap evaluation
    from transformers import AutoTokenizer, LlamaForCausalLM as _VanillaLlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=_local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Skip dataset loading in dry-run (no training or evaluation performed)
    train_loader = val_loader = test_loader = None
    if not args.dry_run:
        selected_dataset = args.dataset
        if selected_dataset == "wikitext103":
            logger.info("Loading WikiText-103 for training/validation/test...")
            train_dataset = load_wikitext103(
                tokenizer, args.max_length, split="train", local_files_only=_local_only
            )
            val_dataset = load_wikitext103(
                tokenizer,
                args.max_length,
                split="validation",
                local_files_only=_local_only,
            )
            test_dataset = load_wikitext103(
                tokenizer, args.max_length, split="test", local_files_only=_local_only
            )
        elif selected_dataset == "fineweb":
            logger.info(
                "Loading FineWeb for training/validation/test "
                "(train=%d docs, val/test=%d docs each)...",
                args.dataset_num_samples,
                max(1, args.dataset_num_samples // 10),
            )
            train_dataset = load_fineweb_split(
                tokenizer,
                args.max_length,
                split="train",
                num_train_samples=args.dataset_num_samples,
                local_files_only=_local_only,
            )
            val_dataset = load_fineweb_split(
                tokenizer,
                args.max_length,
                split="validation",
                num_train_samples=args.dataset_num_samples,
                local_files_only=_local_only,
            )
            test_dataset = load_fineweb_split(
                tokenizer,
                args.max_length,
                split="test",
                num_train_samples=args.dataset_num_samples,
                local_files_only=_local_only,
            )
        else:
            # Default: wikitext2
            logger.info("Loading WikiText-2 for training/validation/test...")
            train_dataset = load_wikitext2(
                tokenizer, args.max_length, split="train", local_files_only=_local_only
            )
            val_dataset = load_wikitext2(
                tokenizer,
                args.max_length,
                split="validation",
                local_files_only=_local_only,
            )
            test_dataset = load_wikitext2(
                tokenizer, args.max_length, split="test", local_files_only=_local_only
            )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        logger.info(
            "Train: %d chunks, Val: %d chunks, Test: %d chunks",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )

    # Pre-swap perplexity: evaluate the original unmodified Llama before any
    # layer replacement so we have a true untouched baseline to compare against.
    original_ppl = None
    is_pe_mode = not args.resume_from and not (
        pe_config.pe_attn_mode == "standard" and pe_config.pe_mlp_mode == "standard"
    )
    if is_pe_mode and not args.dry_run:
        logger.info("Evaluating original Llama perplexity (pre-swap)...")
        orig_model = _VanillaLlamaForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch_dtype,
            device_map=device_map,
            local_files_only=_local_only,
        )
        orig_device = next(orig_model.parameters()).device
        original_ppl = evaluate_perplexity(
            orig_model,
            test_loader,
            orig_device,
            args.max_eval_batches,
        )
        print(
            f"\nOriginal Llama test perplexity (pre-swap): {original_ppl:.2f}",
            flush=True,
        )
        del orig_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Force eager attention if attention pattern KD is active (SDPA does not
    # support output_attentions=True which is required to extract attention maps).
    if args.kd_attn_weight > 0.0:
        pe_config._attn_implementation = "eager"
        logger.info(
            "Attention pattern KD enabled (weight=%.3f) — using eager attention.",
            args.kd_attn_weight,
        )

    # Load model
    # In dry-run or local-files-only mode, restrict to the local HF cache
    # so no network download is triggered.  resume_from is always a local path.
    _cache_kwargs: dict = {"local_files_only": True} if _local_only else {}
    try:
        if args.resume_from:
            # Staged training — load a previously saved PE model and
            # optionally change PE modes (e.g., add MLP replacement on top
            # of already-fitted attention layers)
            logger.info(
                "Staged training: loading saved PE model from %s", args.resume_from
            )
            model = PELlamaForCausalLM.from_pretrained_pe_llama(
                args.resume_from,
                pe_config=pe_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        elif (
            pe_config.pe_attn_mode == "standard" and pe_config.pe_mlp_mode == "standard"
        ):
            # Pure standard mode — just load normally
            model = PELlamaForCausalLM.from_pretrained(
                args.model_name,
                config=pe_config,
                dtype=torch_dtype,
                device_map=device_map,
                **_cache_kwargs,
            )
        else:
            # PE mode — use the projection loader
            model = PELlamaForCausalLM.from_pretrained_llama(
                args.model_name,
                pe_config=pe_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **_cache_kwargs,
            )
    except OSError as exc:
        if args.dry_run:
            print(f"\nModel '{args.model_name}' not found in local cache.")
            print("Run without --dry-run to download it, then retry.")
            print(f"(OSError: {exc})")
            return
        raise

    # With device_map, the model is already placed; derive the input device
    # from the first parameter (embedding layer, always on the first device).
    device = next(model.parameters()).device
    logger.info("Input tensors will be sent to: %s", device)

    if args.freeze_base:
        freeze_base_parameters(model, args.pe_mlp_mode)
    if args.freeze_except_layers is not None:
        freeze_except_layers(model, args.freeze_except_layers)

    # Parameter summary
    param_info = count_parameters(model)
    print_param_summary(param_info)

    if args.dry_run:
        n_gpus = torch.cuda.device_count()
        print("=" * 60)
        print("DRY RUN summary")
        print("=" * 60)
        print(f"  model:        {args.model_name}")
        print(f"  attn_mode:    {pe_config.pe_attn_mode}")
        print(f"  attn_init:    {args.attn_init}")
        print(f"  mlp_mode:     {pe_config.pe_mlp_mode}")
        print(f"  attn_layers:  {pe_config.pe_layer_indices or 'all'}")
        print(f"  mlp_layers:   {pe_config.pe_mlp_layer_indices or 'all'}")
        print(f"  dtype:        {args.dtype}")
        print(f"  epochs:       {args.epochs}  lr={args.lr}  batch={args.batch_size}")
        print(f"  freeze_base:  {args.freeze_base}")
        print(f"  attn_pretrain:{args.attn_pretrain_epochs} epoch(s)")
        print(f"  attn_dataset: {args.attn_dataset}")
        if args.attn_dataset == "fineweb":
            print(f"  attn_samples: {args.attn_cache_num_samples}")
        print(f"  ae_pretrain:  {args.ae_pretrain_epochs} epoch(s)")
        if args.ae_pretrain_epochs > 0:
            _ae_lr_display = (
                args.ae_pretrain_lr if args.ae_pretrain_lr is not None else args.lr
            )
            print(f"  ae_lr:        {_ae_lr_display}")
            print(f"  ae_loss_fn:   {args.ae_pretrain_loss}")
            if args.ae_pretrain_loss == "soft_kl":
                print(f"  ae_loss_temp: {args.ae_pretrain_loss_temperature}")
            elif args.ae_pretrain_loss == "combined":
                print(f"  ae_loss_alpha: {args.ae_pretrain_loss_alpha}")
            if args.ae_pretrain_scheduler != "none":
                print(
                    f"  ae_scheduler: {args.ae_pretrain_scheduler} "
                    f"(gamma={args.ae_pretrain_gamma}, lr_warmup={args.ae_pretrain_lr_warmup})"
                )
        print(f"  ae_dataset:   {args.ae_dataset}")
        if args.ae_dataset == "fineweb":
            print(f"  ae_samples:   {args.ae_cache_num_samples}")
        if args.attn_pretrain_early_stopping:
            print(
                f"  attn_early_stop: warmup={args.attn_pretrain_warmup}, patience={args.attn_pretrain_patience}"
            )
        if args.ae_pretrain_early_stopping:
            print(
                f"  ae_early_stop: warmup={args.ae_pretrain_warmup}, patience={args.ae_pretrain_patience}"
            )
        if args.kd_alpha < 1.0:
            print(f"  kd_alpha:     {args.kd_alpha}")
            print(f"  kd_temp:      {args.kd_temperature}")
            print(f"  kd_teacher:   {args.kd_teacher or args.model_name}")
        if args.kd_attn_weight > 0.0:
            print(f"  kd_attn_wt:   {args.kd_attn_weight}")
            print(
                f"  kd_attn_lyrs: {args.kd_attn_layers or pe_config.pe_layer_indices or 'all'}"
            )
        if n_gpus > 1:
            print(f"  GPUs:         {n_gpus} detected — teacher will use cuda:1")
        print("=" * 60)
        print("Config looks good. Remove --dry-run to start training.")
        return

    # Post-swap baseline perplexity (PE layers in place, no training yet)
    logger.info("Evaluating post-swap baseline perplexity...")
    baseline_ppl = evaluate_perplexity(
        model, test_loader, device, args.max_eval_batches
    )
    if original_ppl is not None:
        print(f"\nOriginal Llama test perplexity (pre-swap): {original_ppl:.2f}")
        logger.info(
            f"Post-swap baseline test perplexity: {baseline_ppl:.2f}  (delta: {baseline_ppl - original_ppl:+.2f})"
        )
        print(
            f"Post-swap baseline test perplexity:        {baseline_ppl:.2f}  "
            f"(delta: {baseline_ppl - original_ppl:+.2f})\n",
            flush=True,
        )
    else:
        print(f"\nBaseline test perplexity: {baseline_ppl:.2f}\n", flush=True)

    # ---- Attention Pre-training Phase (Two-Phase: Extract → Train from Cache) ----
    attn_pretrain_ppl = None
    attn_pretrain_stats = []  # list of {attn_epoch, mse_loss}
    if args.attn_pretrain_epochs > 0 and pe_config.pe_attn_mode in (
        "trend_wavelet",
        "trend_wavelet_generic",
        "trend_wavelet_lg",
        "trend_wavelet_generic_lg",
        "trend_wavelet_reduced",
        "trend_wavelet_lg_reduced",
        "trend_wavelet_generic_reduced",
        "trend_wavelet_generic_lg_reduced",
    ):
        from pellm.pe_layers import (
            TrendWaveletLinear,
            TrendWaveletGenericLinear,
            TrendWaveletLinearLG,
            TrendWaveletGenericLinearLG,
        )

        _TW_TYPES = (
            TrendWaveletLinear,
            TrendWaveletGenericLinear,
            TrendWaveletLinearLG,
            TrendWaveletGenericLinearLG,
        )
        _attn_cache_user_dir = getattr(args, "attn_cache_dir", None)
        _attn_cache_tmpdir = None
        if _attn_cache_user_dir:
            attn_cache_dir = _attn_cache_user_dir
        else:
            _attn_cache_tmpdir = tempfile.mkdtemp(prefix="pellm_attn_cache_")
            attn_cache_dir = _attn_cache_tmpdir

        attn_target_layers = [
            i
            for i, layer in enumerate(model.model.layers)
            if any(
                isinstance(getattr(layer.self_attn, proj_name), _TW_TYPES)
                for proj_name in (pe_config.pe_proj_names or list(_ALL_ATTN_PROJS))
            )
        ]
        attn_target_projs = pe_config.pe_proj_names or list(_ALL_ATTN_PROJS)
        if not attn_target_layers:
            logger.warning(
                "No TrendWavelet attention layers found for attention pre-training."
            )
            args.attn_pretrain_epochs = 0
        attn_cache_metadata = {
            "version": _ATTN_CACHE_VERSION,
            "model_name": args.model_name,
            "dataset": args.attn_dataset,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "cache_num_samples": (
                args.attn_cache_num_samples if args.attn_dataset == "fineweb" else None
            ),
            "available_projs": list(_ALL_ATTN_PROJS),
        }

        if (
            args.attn_pretrain_epochs > 0
            and os.path.isdir(attn_cache_dir)
            and not _attn_cache_matches_metadata(
                attn_cache_dir,
                attn_cache_metadata,
            )
        ):
            logger.info(
                "Attention cache metadata mismatch at %s; clearing stale cache files.",
                attn_cache_dir,
            )
            os.makedirs(attn_cache_dir, exist_ok=True)
            for fname in os.listdir(attn_cache_dir):
                if fname == "metadata.json" or (
                    fname.startswith("layer") and fname.endswith(".pt")
                ):
                    os.remove(os.path.join(attn_cache_dir, fname))

        if args.attn_pretrain_epochs > 0 and args.attn_dataset == "fineweb":
            logger.info("Loading FineWeb for attention activation caching...")
            attn_train_dataset = load_fineweb(
                tokenizer,
                args.max_length,
                num_samples=args.attn_cache_num_samples,
                local_files_only=_local_only,
            )
        elif args.attn_pretrain_epochs > 0 and args.attn_dataset == "wikitext103":
            logger.info("Loading WikiText-103 for attention activation caching...")
            attn_train_dataset = load_wikitext103(
                tokenizer,
                args.max_length,
                split="train",
                local_files_only=_local_only,
            )
        elif args.attn_pretrain_epochs > 0:
            logger.info("Loading WikiText-2 for attention activation caching...")
            attn_train_dataset = load_wikitext2(
                tokenizer,
                args.max_length,
                split="train",
                local_files_only=_local_only,
            )
        if args.attn_pretrain_epochs > 0:
            attn_train_loader = DataLoader(
                attn_train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
            )

        if args.attn_pretrain_epochs > 0 and _attn_cache_is_valid(
            attn_cache_dir, attn_cache_metadata, attn_target_layers
        ):
            logger.info(
                "Attention Phase 1: Reusing existing activation cache at %s "
                "(%d layers already extracted — skipping teacher load)",
                attn_cache_dir,
                len(attn_target_layers),
            )
        elif args.attn_pretrain_epochs > 0:
            cached_layers = _attn_cached_layers(attn_cache_dir)
            missing_layers = [
                idx for idx in attn_target_layers if idx not in cached_layers
            ]
            logger.info(
                "Attention Phase 1: Extracting teacher activations to %s ...",
                attn_cache_dir,
            )
            from transformers import AutoModelForCausalLM as _AutoLM

            if torch.cuda.device_count() > 1:
                teacher_device_map = {"": "cuda:1"}
                teacher_device = torch.device("cuda:1")
                logger.info("Loading teacher model on cuda:1 (separate GPU)")
            else:
                teacher_device_map = device_map
                teacher_device = device
            attn_teacher_name = getattr(args, "attn_teacher", None) or args.model_name
            logger.info("Attention teacher model: %s", attn_teacher_name)
            teacher = _AutoLM.from_pretrained(
                attn_teacher_name,
                dtype=torch_dtype,
                device_map=teacher_device_map,
                local_files_only=_local_only,
            )
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            extract_attn_teacher_activations(
                teacher,
                attn_train_loader,
                missing_layers,
                attn_cache_dir,
                device,
                teacher_device=teacher_device,
            )
            attn_cache_metadata["cached_layers"] = sorted(
                cached_layers.union(missing_layers)
            )
            _write_attn_cache_metadata(attn_cache_dir, attn_cache_metadata)

            del teacher
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "Teacher model unloaded. Starting attention training from cache."
            )

        attn_params = []
        for layer_idx in attn_target_layers:
            attn = model.model.layers[layer_idx].self_attn
            for proj_name in attn_target_projs:
                attn_params.extend(list(getattr(attn, proj_name).parameters()))
        if not attn_params:
            logger.warning(
                "No TrendWavelet attention parameters selected for attention pre-training."
            )
            args.attn_pretrain_epochs = 0
        attn_optimizer = (
            torch.optim.AdamW(attn_params, lr=args.lr) if attn_params else None
        )

        from pellm.pe_layers import set_tw_active_g

        set_tw_active_g(model, args.active_g_pretrain)
        logger.info(
            "Attention pre-training: %d epoch(s), %d params, %d target layers, %s (active_g=%s)",
            args.attn_pretrain_epochs,
            sum(p.numel() for p in attn_params),
            len(attn_target_layers),
            attn_target_projs,
            args.active_g_pretrain,
        )
        attn_best_loss = float("inf")
        attn_epochs_no_improve = 0
        attn_use_early_stop = getattr(args, "attn_pretrain_early_stopping", False)
        attn_warmup = getattr(args, "attn_pretrain_warmup", 0)
        attn_patience = getattr(args, "attn_pretrain_patience", 3)
        if attn_use_early_stop:
            logger.info(
                "Attention early stopping enabled: warmup=%d, patience=%d",
                attn_warmup,
                attn_patience,
            )
        for attn_ep in range(1, args.attn_pretrain_epochs + 1):
            attn_loss = train_attn_pretrain_epoch_from_cache(
                model,
                attn_cache_dir,
                attn_target_layers,
                attn_target_projs,
                attn_optimizer,
                device,
                args.grad_accum_steps,
            )
            logger.info(
                "Attention pre-train epoch %d/%d — avg MSE loss %.6f",
                attn_ep,
                args.attn_pretrain_epochs,
                attn_loss,
            )
            attn_pretrain_stats.append({"attn_epoch": attn_ep, "mse_loss": attn_loss})
            if wandb_run is not None:
                wandb_run.log(
                    {"attn_pretrain_epoch": attn_ep, "attn_mse_loss": attn_loss}
                )

            if attn_use_early_stop and attn_ep > attn_warmup:
                if attn_loss < attn_best_loss:
                    attn_best_loss = attn_loss
                    attn_epochs_no_improve = 0
                else:
                    attn_epochs_no_improve += 1
                    if attn_epochs_no_improve >= attn_patience:
                        logger.info(
                            "Attention early stopping: no improvement for %d epoch(s) "
                            "(best MSE loss=%.6f)",
                            attn_patience,
                            attn_best_loss,
                        )
                        break
            elif attn_loss < attn_best_loss:
                attn_best_loss = attn_loss

        if _attn_cache_tmpdir and os.path.isdir(_attn_cache_tmpdir):
            shutil.rmtree(_attn_cache_tmpdir)
            logger.info("Cleaned up temporary attention activation cache.")
        logger.info(
            "Attention pre-training complete. Proceeding to AE pre-training / LM fine-tuning."
        )

        attn_ppl = evaluate_perplexity(
            model, test_loader, device, args.max_eval_batches
        )
        attn_pretrain_ppl = attn_ppl
        print(
            f"Post-attn-pretrain test perplexity: {attn_ppl:.2f}  "
            f"(delta from baseline: {attn_ppl - baseline_ppl:+.2f})"
        )
        if wandb_run is not None:
            wandb_run.log({"attn_pretrain_ppl": attn_pretrain_ppl})
    elif args.attn_pretrain_epochs > 0:
        logger.warning(
            "--attn-pretrain-epochs ignored: pe_attn_mode is '%s' "
            "(requires a TrendWavelet attention mode)",
            pe_config.pe_attn_mode,
        )

    # ---- AE Pre-training Phase (Two-Phase: Extract → Train from Cache) ----
    ae_pretrain_ppl = None
    ae_pretrain_stats = []  # list of {epoch, ae_loss}
    if args.ae_pretrain_epochs > 0 and args.pe_mlp_mode in _RECON_PE_MLP_MODES:
        # Determine activation cache directory
        _ae_cache_user_dir = getattr(args, "ae_cache_dir", None)
        _ae_cache_tmpdir = None
        if _ae_cache_user_dir:
            ae_cache_dir = _ae_cache_user_dir
        else:
            _ae_cache_tmpdir = tempfile.mkdtemp(prefix="pellm_ae_cache_")
            ae_cache_dir = _ae_cache_tmpdir

        # Determine which layers are AE/VAE targets for cache extraction.
        ae_target_layers = _get_ae_target_layers(model)
        _user_ae_pretrain_layers = getattr(args, "ae_pretrain_layer_indices", None)
        if _user_ae_pretrain_layers:
            _detected = set(ae_target_layers)
            _requested = set(_user_ae_pretrain_layers)
            _missing = _requested - _detected
            if _missing:
                raise ValueError(
                    f"--ae-pretrain-layer-indices includes layers {sorted(_missing)} "
                    f"which are not AE/VAE layers in the model "
                    f"(detected AE layers: {sorted(_detected)})"
                )
            ae_target_layers = sorted(_requested)
            logger.info(
                "AE pre-training restricted to user-specified layers: %s "
                "(skipping already-trained AE layers: %s)",
                ae_target_layers,
                sorted(_detected - _requested),
            )

        # Build AE-specific data loader (may differ from training data)
        logger.debug(f"AE dataset choice: args.ae_dataset={args.ae_dataset!r}")
        if args.ae_dataset == "fineweb":
            logger.info("Loading FineWeb for AE activation caching...")
            ae_train_dataset = load_fineweb(
                tokenizer,
                args.max_length,
                num_samples=args.ae_cache_num_samples,
                skip=getattr(args, "ae_cache_skip", 0) or 0,
                local_files_only=_local_only,
            )
            ae_train_loader = DataLoader(
                ae_train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
            )
        else:
            logger.info(
                f"Using training dataset (WikiText-2) for AE activation caching (ae_dataset={args.ae_dataset!r})"
            )
            ae_train_loader = train_loader

        # --- Phase 1: Extract teacher activations to disk (or reuse cache) ---
        # Skip when per-epoch resampling is enabled: all epochs are served by
        # either per-epoch FineWeb caches or the WikiText-2 tail cache, so the
        # shared ae_cache_dir is never read and writing it wastes disk space.
        _ae_resample = getattr(args, "ae_pretrain_resample", False) and args.ae_dataset == "fineweb"
        _ae_fineweb_epochs = getattr(args, "ae_pretrain_fineweb_epochs", 0) or args.ae_pretrain_epochs
        if _ae_resample:
            logger.info(
                "AE per-epoch resampling enabled: skipping shared Phase 1 extraction. "
                "Fresh FineWeb stream (%d docs) for epochs 1–%d.",
                args.ae_cache_num_samples,
                _ae_fineweb_epochs,
            )
        elif _cache_is_valid(ae_cache_dir, ae_target_layers):
            logger.info(
                "AE Phase 1: Reusing existing activation cache at %s "
                "(%d layers already extracted — skipping teacher load)",
                ae_cache_dir,
                len(ae_target_layers),
            )
        else:
            logger.info(
                "AE Phase 1: Extracting teacher activations to %s ...", ae_cache_dir
            )
            from transformers import AutoModelForCausalLM as _AutoLM

            if torch.cuda.device_count() > 1:
                teacher_device_map = {"": "cuda:1"}
                teacher_device = torch.device("cuda:1")
                logger.info("Loading teacher model on cuda:1 (separate GPU)")
            else:
                teacher_device_map = device_map
                teacher_device = device
            ae_teacher_name = getattr(args, "ae_teacher", None) or args.model_name
            logger.info("AE teacher model: %s", ae_teacher_name)
            teacher = _AutoLM.from_pretrained(
                ae_teacher_name,
                dtype=torch_dtype,
                device_map=teacher_device_map,
                local_files_only=_local_only,
            )
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            extract_teacher_activations(
                teacher,
                ae_train_loader,
                ae_target_layers,
                ae_cache_dir,
                device,
                teacher_device=teacher_device,
            )

            # Free teacher memory before Phase 2
            del teacher
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Teacher model unloaded. Starting AE training from cache.")

        # --- Phase 2: Train AE layers from cached activations ---
        ae_params = [
            p
            for n, p in model.named_parameters()
            if ".mlp.fc" in n or ".mlp.latent_gate" in n
        ]
        ae_lr = args.ae_pretrain_lr if args.ae_pretrain_lr is not None else args.lr
        ae_optimizer = torch.optim.AdamW(ae_params, lr=ae_lr)

        logger.info(
            "AE pre-training: %d epoch(s), %d MLP params, %d target layers, lr=%.6f",
            args.ae_pretrain_epochs,
            sum(p.numel() for p in ae_params),
            len(ae_target_layers),
            ae_lr,
        )

        # LR scheduler
        ae_scheduler = None
        if args.ae_pretrain_scheduler == "exponential":
            from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR

            lr_warmup_epochs = args.ae_pretrain_lr_warmup
            if lr_warmup_epochs > 0:
                warmup_sched = LinearLR(
                    ae_optimizer,
                    start_factor=0.001,
                    end_factor=1.0,
                    total_iters=lr_warmup_epochs,
                )
                decay_sched = ExponentialLR(ae_optimizer, gamma=args.ae_pretrain_gamma)
                ae_scheduler = SequentialLR(
                    ae_optimizer,
                    schedulers=[warmup_sched, decay_sched],
                    milestones=[lr_warmup_epochs],
                )
            else:
                ae_scheduler = ExponentialLR(ae_optimizer, gamma=args.ae_pretrain_gamma)
            logger.info(
                "AE LR scheduler: %s (lr=%.6f, warmup=%d, gamma=%.4f)",
                args.ae_pretrain_scheduler,
                ae_lr,
                lr_warmup_epochs,
                args.ae_pretrain_gamma,
            )

        ae_best_loss = float("inf")
        ae_epochs_no_improve = 0
        ae_use_early_stop = getattr(args, "ae_pretrain_early_stopping", False)
        ae_warmup = getattr(args, "ae_pretrain_warmup", 0)
        ae_patience = getattr(args, "ae_pretrain_patience", 3)
        if ae_use_early_stop:
            logger.info(
                "AE early stopping enabled: warmup=%d, patience=%d",
                ae_warmup,
                ae_patience,
            )
        # WikiText-2 tail cache is extracted lazily on first use to avoid
        # coexisting on disk with the FineWeb per-epoch cache.
        _wikitext2_tail_cache_dir = None
        _wikitext2_tail_tmpdir = None
        _needs_wikitext2_tail = _ae_resample and _ae_fineweb_epochs < args.ae_pretrain_epochs

        for ae_ep in range(1, args.ae_pretrain_epochs + 1):
            epoch_cache_dir = None
            if _ae_resample and ae_ep <= _ae_fineweb_epochs:
                # Stream a fresh, non-overlapping FineWeb slice for this epoch.
                # ae_cache_skip provides a per-wave base offset (set by run_wave_pipeline.py
                # rotate_ae_samples), so each wave starts at a different doc position and
                # epochs within a wave don't overlap with any other wave's epochs.
                _wave_skip = getattr(args, "ae_cache_skip", 0) or 0
                skip = _wave_skip + (ae_ep - 1) * args.ae_cache_num_samples
                logger.info(
                    "AE resample epoch %d/%d: loading FineWeb (skip=%d, n=%d) ...",
                    ae_ep, args.ae_pretrain_epochs, skip, args.ae_cache_num_samples,
                )
                ae_resample_dataset = load_fineweb(
                    tokenizer,
                    args.max_length,
                    num_samples=args.ae_cache_num_samples,
                    skip=skip,
                    local_files_only=_local_only,
                )
                ae_resample_loader = DataLoader(
                    ae_resample_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                epoch_cache_dir = tempfile.mkdtemp(prefix=f"pellm_ae_resample_ep{ae_ep}_")
                from transformers import AutoModelForCausalLM as _AutoLM
                if torch.cuda.device_count() > 1:
                    _resample_teacher_device_map = {"": "cuda:1"}
                    _resample_teacher_device = torch.device("cuda:1")
                else:
                    _resample_teacher_device_map = device_map
                    _resample_teacher_device = device
                ae_teacher_name = getattr(args, "ae_teacher", None) or args.model_name
                logger.info("AE resample: loading teacher %s ...", ae_teacher_name)
                _resample_teacher = _AutoLM.from_pretrained(
                    ae_teacher_name,
                    dtype=torch_dtype,
                    device_map=_resample_teacher_device_map,
                    local_files_only=_local_only,
                )
                _resample_teacher.eval()
                for p in _resample_teacher.parameters():
                    p.requires_grad_(False)
                extract_teacher_activations(
                    _resample_teacher,
                    ae_resample_loader,
                    ae_target_layers,
                    epoch_cache_dir,
                    device,
                    teacher_device=_resample_teacher_device,
                )
                del _resample_teacher
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("AE resample: teacher unloaded. Training epoch %d ...", ae_ep)
                _epoch_ae_cache = epoch_cache_dir
            elif _needs_wikitext2_tail:
                # Tail epochs use WikiText-2. Extract lazily on first use so its
                # cache never coexists on disk with a FineWeb per-epoch cache.
                if _wikitext2_tail_cache_dir is None:
                    logger.info(
                        "AE mixed-dataset: extracting WikiText-2 cache for epochs %d–%d ...",
                        _ae_fineweb_epochs + 1,
                        args.ae_pretrain_epochs,
                    )
                    from transformers import AutoModelForCausalLM as _AutoLM
                    if torch.cuda.device_count() > 1:
                        _wt2_teacher_device_map = {"": "cuda:1"}
                        _wt2_teacher_device = torch.device("cuda:1")
                    else:
                        _wt2_teacher_device_map = device_map
                        _wt2_teacher_device = device
                    ae_teacher_name = getattr(args, "ae_teacher", None) or args.model_name
                    logger.info("AE WikiText-2 tail: loading teacher %s ...", ae_teacher_name)
                    _wt2_teacher = _AutoLM.from_pretrained(
                        ae_teacher_name,
                        dtype=torch_dtype,
                        device_map=_wt2_teacher_device_map,
                        local_files_only=_local_only,
                    )
                    _wt2_teacher.eval()
                    for p in _wt2_teacher.parameters():
                        p.requires_grad_(False)
                    _wikitext2_tail_tmpdir = tempfile.mkdtemp(prefix="pellm_ae_wt2_tail_")
                    _wikitext2_tail_cache_dir = _wikitext2_tail_tmpdir
                    extract_teacher_activations(
                        _wt2_teacher,
                        train_loader,
                        ae_target_layers,
                        _wikitext2_tail_cache_dir,
                        device,
                        teacher_device=_wt2_teacher_device,
                    )
                    del _wt2_teacher
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("AE WikiText-2 tail: teacher unloaded. Cache ready at %s.", _wikitext2_tail_cache_dir)
                logger.info(
                    "AE mixed-dataset epoch %d/%d: using WikiText-2 cache.",
                    ae_ep, args.ae_pretrain_epochs,
                )
                _epoch_ae_cache = _wikitext2_tail_cache_dir
            else:
                _epoch_ae_cache = ae_cache_dir

            try:
                ae_loss = train_ae_epoch_from_cache(
                    model,
                    _epoch_ae_cache,
                    ae_target_layers,
                    ae_optimizer,
                    device,
                    args.grad_accum_steps,
                    loss_type=args.ae_pretrain_loss,
                    loss_temperature=args.ae_pretrain_loss_temperature,
                    loss_alpha=args.ae_pretrain_loss_alpha,
                )
            finally:
                if epoch_cache_dir is not None and os.path.isdir(epoch_cache_dir):
                    shutil.rmtree(epoch_cache_dir)
                    logger.info("AE resample: cleaned up epoch %d cache.", ae_ep)
            current_ae_lr = ae_optimizer.param_groups[0]["lr"]
            logger.info(
                "AE pre-train epoch %d/%d — avg %s loss %.6f  lr=%.6f",
                ae_ep,
                args.ae_pretrain_epochs,
                args.ae_pretrain_loss,
                ae_loss,
                current_ae_lr,
            )
            ae_pretrain_stats.append(
                {"ae_epoch": ae_ep, "ae_loss": ae_loss, "ae_lr": current_ae_lr}
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "ae_pretrain_epoch": ae_ep,
                        "ae_loss": ae_loss,
                        "ae_lr": current_ae_lr,
                    }
                )

            # Early stopping check (only after warmup epochs)
            if ae_use_early_stop and ae_ep > ae_warmup:
                if ae_loss < ae_best_loss:
                    ae_best_loss = ae_loss
                    ae_epochs_no_improve = 0
                else:
                    ae_epochs_no_improve += 1
                    if ae_epochs_no_improve >= ae_patience:
                        logger.info(
                            "AE early stopping: no improvement for %d epoch(s) "
                            "(best MSE loss=%.6f)",
                            ae_patience,
                            ae_best_loss,
                        )
                        break
            elif ae_loss < ae_best_loss:
                ae_best_loss = ae_loss

            # Step LR scheduler
            if ae_scheduler is not None:
                ae_scheduler.step()

        # Clean up temporary cache directory (preserve user-specified dirs)
        if _ae_cache_tmpdir and os.path.isdir(_ae_cache_tmpdir):
            shutil.rmtree(_ae_cache_tmpdir)
            logger.info("Cleaned up temporary activation cache.")
        if _wikitext2_tail_tmpdir and os.path.isdir(_wikitext2_tail_tmpdir):
            shutil.rmtree(_wikitext2_tail_tmpdir)
            logger.info("Cleaned up WikiText-2 tail activation cache.")
        logger.info("AE pre-training complete. Proceeding to LM fine-tuning.")

        # Re-evaluate perplexity after AE pre-training
        ae_ppl = evaluate_perplexity(model, test_loader, device, args.max_eval_batches)
        ae_pretrain_ppl = ae_ppl
        print(
            f"Post-AE-pretrain test perplexity: {ae_ppl:.2f}  "
            f"(delta from baseline: {ae_ppl - baseline_ppl:+.2f})"
        )
        if wandb_run is not None:
            wandb_run.log({"ae_pretrain_ppl": ae_pretrain_ppl})
    elif args.ae_pretrain_epochs > 0:
        logger.warning(
            "--ae-pretrain-epochs ignored: pe_mlp_mode is '%s' (requires ae, ae_lg, vae, or vae_lg)",
            args.pe_mlp_mode,
        )

    if args.epochs == 0:
        logger.info("No training requested (--epochs 0).")
        # final_ppl = last evaluated perplexity (ae_pretrain, attn_pretrain, or baseline)
        final_ppl = (
            ae_pretrain_ppl
            if ae_pretrain_ppl is not None
            else attn_pretrain_ppl if attn_pretrain_ppl is not None else baseline_ppl
        )
        print(f"\nFinal test perplexity: {final_ppl:.2f}")
        if baseline_ppl != final_ppl:
            print(
                f"Improvement: {baseline_ppl:.2f} -> {final_ppl:.2f} "
                f"({100 * (baseline_ppl - final_ppl) / baseline_ppl:+.1f}%)"
            )

        results = {
            "pe_attn_mode": args.pe_attn_mode,
            "pe_mlp_mode": args.pe_mlp_mode,
            "ae_latent_dim": args.ae_latent_dim,
            "trend_dim": args.trend_dim,
            "wavelet_dim": args.wavelet_dim,
            "wavelet_type": args.wavelet_type,
            "generic_dim": args.generic_dim,
            "params_total": param_info["total"],
            "params_trainable": param_info["trainable"],
            "original_ppl": original_ppl,
            "baseline_ppl": baseline_ppl,
            "attn_pretrain_ppl": attn_pretrain_ppl,
            "ae_pretrain_ppl": ae_pretrain_ppl,
            "final_ppl": final_ppl,
            "epochs": args.epochs,
            "lr": args.lr,
            "attn_pretrain_epochs": args.attn_pretrain_epochs,
            "ae_pretrain_epochs": args.ae_pretrain_epochs,
            "dataset": args.dataset,
            "dataset_num_samples": args.dataset_num_samples,
            "attn_dataset": args.attn_dataset,
            "attn_cache_num_samples": args.attn_cache_num_samples,
            "ae_dataset": args.ae_dataset,
            "ae_cache_num_samples": args.ae_cache_num_samples,
            "attn_pretrain_stats": attn_pretrain_stats,
            "ae_pretrain_stats": ae_pretrain_stats,
            "per_epoch_stats": [],
            "kd_alpha": args.kd_alpha,
            "kd_temperature": args.kd_temperature,
            "kd_teacher": args.kd_teacher,
            "kd_attn_weight": args.kd_attn_weight,
            "kd_attn_layers": args.kd_attn_layers,
            "active_g_pretrain": args.active_g_pretrain,
            "active_g_finetune": args.active_g_finetune,
            "ae_pretrain_lr": (
                args.ae_pretrain_lr if args.ae_pretrain_lr is not None else args.lr
            ),
            "ae_pretrain_scheduler": args.ae_pretrain_scheduler,
            "ae_pretrain_lr_warmup": args.ae_pretrain_lr_warmup,
            "ae_pretrain_gamma": args.ae_pretrain_gamma,
            "ae_pretrain_loss": args.ae_pretrain_loss,
            "ae_pretrain_loss_temperature": args.ae_pretrain_loss_temperature,
            "ae_pretrain_loss_alpha": args.ae_pretrain_loss_alpha,
            "ae_pretrain_resample": args.ae_pretrain_resample,
            "ae_pretrain_fineweb_epochs": args.ae_pretrain_fineweb_epochs,
        }

        # Inspect learned gate values if using ae_lg (before Results JSON
        # so that the JSON remains the last printed line for parse_results)
        if args.pe_mlp_mode in _GATE_REPORT_PE_MLP_MODES:
            print_learned_gate_values(model)

        print(f"\nResults: {json.dumps(results, indent=2)}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "original_ppl": original_ppl,
                    "baseline_ppl": baseline_ppl,
                    "attn_pretrain_ppl": attn_pretrain_ppl,
                    "ae_pretrain_ppl": ae_pretrain_ppl,
                    "final_ppl": final_ppl,
                    "params_total": param_info["total"],
                    "params_trainable": param_info["trainable"],
                }
            )
            wandb_run.finish()

        if args.log_csv:
            import csv

            csv_row = {
                k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                for k, v in results.items()
            }
            csv_path = Path(args.log_csv)
            write_header = not csv_path.exists()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(csv_row)

        return

    # Fine-tune — load KD teacher if requested (logit KD or attention pattern KD)
    kd_teacher = None
    kd_teacher_device = None
    _need_kd_teacher = (
        args.kd_alpha < 1.0 or args.kd_attn_weight > 0.0
    ) and args.epochs > 0
    if _need_kd_teacher:
        from transformers import AutoModelForCausalLM as _KD_AutoLM

        kd_teacher_name = args.kd_teacher or args.model_name
        if torch.cuda.device_count() > 1:
            kd_teacher_device_map = {"": "cuda:1"}
            kd_teacher_device = torch.device("cuda:1")
            logger.info("Loading KD teacher '%s' on cuda:1", kd_teacher_name)
        else:
            kd_teacher_device_map = device_map
            kd_teacher_device = device
            logger.info(
                "Loading KD teacher '%s' on %s (single GPU)", kd_teacher_name, device
            )
        _kd_teacher_kwargs = {
            "dtype": torch_dtype,
            "device_map": kd_teacher_device_map,
            "local_files_only": _local_only,
        }
        if args.kd_attn_weight > 0.0:
            _kd_teacher_kwargs["attn_implementation"] = "eager"
        kd_teacher = _KD_AutoLM.from_pretrained(
            kd_teacher_name,
            **_kd_teacher_kwargs,
        )
        kd_teacher.eval()
        for p in kd_teacher.parameters():
            p.requires_grad_(False)
        logger.info(
            "KD teacher loaded: alpha=%.2f, temperature=%.1f",
            args.kd_alpha,
            args.kd_temperature,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    best_val_ppl = float("inf")
    epochs_without_improvement = 0
    use_early_stopping = args.patience > 0
    per_epoch_stats = []

    # Toggle active_g for LM fine-tuning phase
    from pellm.pe_layers import set_tw_active_g as _set_tw_active_g

    _set_tw_active_g(model, args.active_g_finetune)
    pe_config.active_g = args.active_g_finetune

    logger.info(
        "Starting LM fine-tuning: %d epoch(s), lr=%.6f, batch_size=%d (active_g=%s)",
        args.epochs,
        args.lr,
        args.batch_size,
        args.active_g_finetune,
    )

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d: starting training...", epoch, args.epochs)
        # Resolve attention pattern KD layers (default to pe_layer_indices)
        _kd_attn_layers = args.kd_attn_layers
        if _kd_attn_layers is None and args.kd_attn_weight > 0.0:
            _kd_attn_layers = pe_config.pe_layer_indices
            if _kd_attn_layers is None:
                # pe_layer_indices=None means all layers
                _kd_attn_layers = list(range(pe_config.num_hidden_layers))

        avg_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.grad_accum_steps,
            teacher_model=kd_teacher,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
            teacher_device=kd_teacher_device,
            kd_attn_weight=args.kd_attn_weight,
            kd_attn_layers=_kd_attn_layers,
        )
        logger.info(
            "Epoch %d/%d: training complete, evaluating validation perplexity...",
            epoch,
            args.epochs,
        )
        val_ppl = evaluate_perplexity(model, val_loader, device, args.max_eval_batches)

        print(
            f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  val_ppl={val_ppl:.2f}"
        )
        per_epoch_stats.append(
            {"epoch": epoch, "train_loss": avg_loss, "val_ppl": val_ppl}
        )
        if wandb_run is not None:
            log_dict = {"epoch": epoch, "train_loss": avg_loss, "val_ppl": val_ppl}
            if args.kd_alpha < 1.0:
                log_dict["kd_alpha"] = args.kd_alpha
                log_dict["kd_temperature"] = args.kd_temperature
            wandb_run.log(log_dict)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            epochs_without_improvement = 0
            if args.output_dir:
                logger.info("Saving best model to %s", args.output_dir)
                save_checkpoint_with_symlinks(
                    model,
                    tokenizer,
                    args.output_dir,
                    args.shared_tokenizer_dir,
                )
        else:
            epochs_without_improvement += 1
            if use_early_stopping and epochs_without_improvement >= args.patience:
                logger.info(
                    "Early stopping: no improvement for %d epoch(s) "
                    "(best val_ppl=%.2f)",
                    args.patience,
                    best_val_ppl,
                )
                break

    # Unload KD teacher to free memory before final evaluation
    if kd_teacher is not None:
        del kd_teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("KD teacher model unloaded.")

    # Reload best checkpoint for final evaluation if one was saved
    if args.output_dir and Path(args.output_dir).exists():
        logger.info(
            "Reloading best checkpoint from %s for final evaluation", args.output_dir
        )
        model = PELlamaForCausalLM.from_pretrained(
            args.output_dir,
            dtype=torch_dtype,
            device_map=device_map,
        )
        device = next(model.parameters()).device

    # Final evaluation
    final_ppl = evaluate_perplexity(model, test_loader, device, args.max_eval_batches)
    print(f"\nFinal test perplexity: {final_ppl:.2f}")
    print(
        f"Improvement: {baseline_ppl:.2f} -> {final_ppl:.2f} "
        f"({100 * (baseline_ppl - final_ppl) / baseline_ppl:+.1f}%)"
    )

    # Save results summary
    results = {
        "pe_attn_mode": args.pe_attn_mode,
        "pe_mlp_mode": args.pe_mlp_mode,
        "ae_latent_dim": args.ae_latent_dim,
        "trend_dim": args.trend_dim,
        "wavelet_dim": args.wavelet_dim,
        "wavelet_type": args.wavelet_type,
        "generic_dim": args.generic_dim,
        "params_total": param_info["total"],
        "params_trainable": param_info["trainable"],
        "original_ppl": original_ppl,
        "baseline_ppl": baseline_ppl,
        "attn_pretrain_ppl": attn_pretrain_ppl,
        "ae_pretrain_ppl": ae_pretrain_ppl,
        "final_ppl": final_ppl,
        "epochs": args.epochs,
        "lr": args.lr,
        "attn_pretrain_epochs": args.attn_pretrain_epochs,
        "ae_pretrain_epochs": args.ae_pretrain_epochs,
        "dataset": args.dataset,
        "dataset_num_samples": args.dataset_num_samples,
        "attn_dataset": args.attn_dataset,
        "attn_cache_num_samples": args.attn_cache_num_samples,
        "ae_dataset": args.ae_dataset,
        "ae_cache_num_samples": args.ae_cache_num_samples,
        "attn_pretrain_stats": attn_pretrain_stats,
        "ae_pretrain_stats": ae_pretrain_stats,
        "per_epoch_stats": per_epoch_stats,
        "kd_alpha": args.kd_alpha,
        "kd_temperature": args.kd_temperature,
        "kd_teacher": args.kd_teacher,
        "kd_attn_weight": args.kd_attn_weight,
        "kd_attn_layers": args.kd_attn_layers,
        "ae_teacher": getattr(args, "ae_teacher", None),
        "attn_teacher": getattr(args, "attn_teacher", None),
        "active_g_pretrain": args.active_g_pretrain,
        "active_g_finetune": args.active_g_finetune,
        "ae_pretrain_lr": (
            args.ae_pretrain_lr if args.ae_pretrain_lr is not None else args.lr
        ),
        "ae_pretrain_scheduler": args.ae_pretrain_scheduler,
        "ae_pretrain_lr_warmup": args.ae_pretrain_lr_warmup,
        "ae_pretrain_gamma": args.ae_pretrain_gamma,
        "ae_pretrain_loss": args.ae_pretrain_loss,
        "ae_pretrain_loss_temperature": args.ae_pretrain_loss_temperature,
        "ae_pretrain_loss_alpha": args.ae_pretrain_loss_alpha,
        "ae_pretrain_resample": args.ae_pretrain_resample,
    }

    # Inspect learned gate values if using ae_lg (before Results JSON
    # so that the JSON remains the last printed line for parse_results)
    if args.pe_mlp_mode in _GATE_REPORT_PE_MLP_MODES:
        print_learned_gate_values(model)

    print(f"\nResults: {json.dumps(results, indent=2)}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "original_ppl": original_ppl,
                "baseline_ppl": baseline_ppl,
                "attn_pretrain_ppl": attn_pretrain_ppl,
                "ae_pretrain_ppl": ae_pretrain_ppl,
                "final_ppl": final_ppl,
                "params_total": param_info["total"],
                "params_trainable": param_info["trainable"],
            }
        )
        wandb_run.finish()

    if args.log_csv:
        import csv

        csv_row = {
            k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
            for k, v in results.items()
        }
        csv_path = Path(args.log_csv)
        write_header = not csv_path.exists()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row)


if __name__ == "__main__":
    main()
