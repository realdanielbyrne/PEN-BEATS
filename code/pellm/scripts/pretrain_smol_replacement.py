#!/usr/bin/env python3
"""From-scratch Smol-style pretraining for the PELLM paper experiment.

This is intentionally separate from ``finetune.py``.  The older runner is for
pretrained Llama replacement/repair; this script trains matched Smol-style
architectures from random initialization to test AE MLP and TrendWavelet layers
as native model components.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_paths import (  # noqa: E402
    CHECKPOINTS_DIR,
    EVALS_DIR,
    RUNS_DIR,
    TRAINED_MODELS_DIR,
    apply_data_drive_env_defaults,
    ensure_artifact_dirs,
    repo_id_for_variant,
)


@dataclass(frozen=True)
class VariantConfig:
    pe_attn_mode: str
    pe_mlp_mode: str
    ae_latent_dim: int
    trend_dim: int
    wavelet_dim: int
    wavelet_type: str
    wavelet_basis_offset: int = 0
    per_layer_offsets: tuple[int, ...] | None = None
    active_g: bool = False
    mlp_active_g: bool = False
    mlp_wavelet_type: str | None = None


# Mutable container used by the signal handler and the main loop.
_sig_state: dict[str, Any] = {}
# Set to True by the signal handler; the main loop performs the save.
_shutdown_requested = False


def _graceful_exit_handler(signum: int, _frame: Any) -> None:
    """Catch SIGINT/SIGTERM and request a graceful shutdown from the main loop.

    First signal sets the shutdown flag so the main loop can checkpoint cleanly.
    A second signal exits immediately to avoid corrupting a half-written checkpoint.
    """
    global _shutdown_requested
    del _frame
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        # Second signal — abort immediately rather than interrupt the save.
        sys.exit(1)
    if (
        _sig_state.get("accelerator") is None
        or _sig_state["accelerator"].is_main_process
    ):
        print(
            f"\n[{sig_name}] caught — will checkpoint and exit after current step...",
            flush=True,
        )
    _shutdown_requested = True


class PackedTextDataset(IterableDataset):
    """Stream text rows, tokenize, append EOS, and pack fixed-length chunks."""

    def __init__(
        self,
        *,
        dataset_id: str,
        dataset_config: str | None,
        split: str,
        text_column: str,
        tokenizer,
        sequence_length: int,
        seed: int,
        shuffle_buffer: int,
        streaming: bool,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_config = dataset_config
        self.split = split
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.streaming = streaming

    def __iter__(self):
        from datasets import load_dataset
        from torch.utils.data import get_worker_info

        kwargs: dict[str, Any] = {
            "path": self.dataset_id,
            "split": self.split,
            "streaming": self.streaming,
        }
        if self.dataset_config:
            kwargs["name"] = self.dataset_config

        ds = load_dataset(**kwargs)
        if self.streaming:
            worker_info = get_worker_info()
            if worker_info is not None and hasattr(ds, "shard"):
                ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            if self.shuffle_buffer > 0 and hasattr(ds, "shuffle"):
                ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        eos_id = self.tokenizer.eos_token_id
        buffer: list[int] = []
        for row in ds:
            text = row.get(self.text_column)
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                ids.append(eos_id)
            buffer.extend(ids)

            while len(buffer) >= self.sequence_length:
                chunk = buffer[: self.sequence_length]
                del buffer[: self.sequence_length]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the from-scratch Smol replacement paper experiment.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="scripts/experiments/smol_replacement_paper.yaml",
        help="YAML experiment config.",
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Architecture variant name (must exist in YAML variants block).",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Override config training.token_budget.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-detect the latest checkpoint in checkpoint_dir and resume training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit before downloading/loading data.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a saved final model and run validation without training.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory for --eval-only. Defaults to the variant output dir.",
    )
    parser.add_argument(
        "--repair-metadata",
        action="store_true",
        help="With --eval-only, rewrite summary.json and training_manifest.json.",
    )
    parser.add_argument(
        "--trained-tokens",
        type=int,
        default=None,
        help="Token count to record for --eval-only metadata repair.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Install the experiments extra: pip install -e '.[experiments]'"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "torch": torch.__version__}
    for name in ("transformers", "datasets", "accelerate"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    versions["pellm"] = "0.1.0"
    return versions


def wandb_config(cfg: dict, resolved: dict) -> dict[str, Any]:
    """Build a compact, serializable W&B config payload."""

    return {
        "experiment": resolved["experiment"],
        "variant": resolved["variant"],
        "variant_config": resolved["variant_config"],
        "token_budget": resolved["token_budget"],
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "training": cfg["training"],
        "artifact_paths": {
            "output_dir": resolved["output_dir"],
            "checkpoint_dir": resolved["checkpoint_dir"],
            "run_dir": resolved["run_dir"],
            "eval_dir": resolved["eval_dir"],
        },
        "git_commit": git_commit(),
    }


def variant_from_config(cfg: dict, name: str) -> VariantConfig:
    variants = cfg.get("variants", {})
    if name not in variants:
        valid = ", ".join(sorted(variants))
        raise ValueError(f"Unknown variant {name!r}; expected one of: {valid}")
    raw = variants[name]

    raw_offsets = raw.get("per_layer_offsets", cfg["model"].get("per_layer_offsets"))
    per_layer: tuple[int, ...] | None = None
    if raw_offsets is not None:
        per_layer = tuple(int(x) for x in raw_offsets)
        n_layers = int(cfg["model"]["num_hidden_layers"])
        if len(per_layer) != n_layers:
            raise ValueError(
                f"per_layer_offsets has {len(per_layer)} entries, "
                f"expected {n_layers} (num_hidden_layers)"
            )

    return VariantConfig(
        pe_attn_mode=raw.get("pe_attn_mode", "standard"),
        pe_mlp_mode=raw.get("pe_mlp_mode", "standard"),
        ae_latent_dim=int(
            raw.get("ae_latent_dim", cfg["model"].get("ae_latent_dim", 144))
        ),
        trend_dim=int(raw.get("trend_dim", cfg["model"].get("trend_dim", 4))),
        wavelet_dim=int(raw.get("wavelet_dim", cfg["model"].get("wavelet_dim", 40))),
        wavelet_type=raw.get("wavelet_type", cfg["model"].get("wavelet_type", "db3")),
        wavelet_basis_offset=int(
            raw.get(
                "wavelet_basis_offset",
                cfg["model"].get("wavelet_basis_offset", 0),
            )
        ),
        per_layer_offsets=per_layer,
        active_g=bool(raw.get("active_g", cfg["model"].get("active_g", False))),
        mlp_active_g=bool(
            raw.get("mlp_active_g", cfg["model"].get("mlp_active_g", False))
        ),
        mlp_wavelet_type=raw.get("mlp_wavelet_type", None),
    )


def build_model(cfg: dict, variant: VariantConfig):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import pellm  # noqa: F401
    from pellm import PELlamaConfig, PELlamaForCausalLM

    model_cfg = cfg["model"]
    pe_config = PELlamaConfig(
        vocab_size=int(model_cfg["vocab_size"]),
        hidden_size=int(model_cfg["hidden_size"]),
        intermediate_size=int(model_cfg["intermediate_size"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        num_attention_heads=int(model_cfg["num_attention_heads"]),
        num_key_value_heads=int(model_cfg["num_key_value_heads"]),
        head_dim=int(model_cfg["head_dim"]),
        max_position_embeddings=int(model_cfg["max_position_embeddings"]),
        rope_theta=float(model_cfg["rope_theta"]),
        rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        hidden_act=model_cfg["hidden_act"],
        tie_word_embeddings=bool(model_cfg["tie_word_embeddings"]),
        attention_bias=bool(model_cfg.get("attention_bias", False)),
        pe_attn_mode=variant.pe_attn_mode,
        pe_mlp_mode=variant.pe_mlp_mode,
        ae_latent_dim=variant.ae_latent_dim,
        ae_init_mode="random",
        attn_init_mode="random",
        trend_dim=variant.trend_dim,
        wavelet_dim=variant.wavelet_dim,
        wavelet_type=variant.wavelet_type,
        wavelet_basis_offset=variant.wavelet_basis_offset,
        per_layer_offsets=(
            list(variant.per_layer_offsets)
            if variant.per_layer_offsets is not None
            else None
        ),
        active_g=variant.active_g,
        mlp_active_g=variant.mlp_active_g,
        mlp_wavelet_type=variant.mlp_wavelet_type,
    )
    model = PELlamaForCausalLM(pe_config)
    # PELlamaForCausalLM swaps in PELlamaModel after the upstream constructor;
    # run the standard HF initializer once more so from-scratch PE layers use
    # the same init path as a normal Llama model.
    model.post_init()
    model.tie_weights()
    return model


@torch.no_grad()
def evaluate(model, dataloader, accelerator, max_batches: int) -> dict[str, float]:
    model.eval()
    losses = []
    dataloader_iter = None
    try:
        dataloader_iter = iter(dataloader)
        for step, batch in enumerate(dataloader_iter):
            if step >= max_batches:
                break
            outputs = model(**batch)
            losses.append(accelerator.gather(outputs.loss.detach().float()).mean())
        if not losses:
            return {"loss": float("nan"), "perplexity": float("nan")}
        loss = torch.stack(losses).mean().item()
        return {
            "loss": loss,
            "perplexity": math.exp(loss) if loss < 50 else float("inf"),
        }
    finally:
        best_effort_shutdown_dataloader_workers(dataloader_iter, dataloader)
        model.train()


def save_manifest(
    *,
    cfg: dict,
    variant_name: str,
    variant: VariantConfig,
    output_dir: Path,
    checkpoint_dir: Path,
    run_dir: Path,
    eval_dir: Path,
    training_cmd: str,
    token_budget: int,
    final_metrics: dict,
) -> None:
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    manifest = {
        "model_id": repo_id_for_variant(variant_name),
        "experiment_name": cfg["experiment"]["name"],
        "architecture": variant_name,
        "variant_config": asdict(variant),
        "dataset_id": dataset_cfg["id"],
        "dataset_config": dataset_cfg.get("config"),
        "text_column": dataset_cfg.get("text_column", "text"),
        "tokenizer_id": cfg["model"]["tokenizer_id"],
        "sequence_length": train_cfg["sequence_length"],
        "token_budget": token_budget,
        "seed": train_cfg["seed"],
        "hardware": cfg["experiment"].get(
            "hardware", "2x NVIDIA RTX 5090, 32GB VRAM each"
        ),
        "git_commit": git_commit(),
        "training_command": training_cmd,
        "package_versions": package_versions(),
        "artifact_paths": {
            "model_dir": str(output_dir),
            "checkpoint_root": str(checkpoint_dir),
            "run_root": str(run_dir),
            "eval_root": str(eval_dir),
        },
        "final_metrics": final_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "training_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Checkpoint resumability helpers
# ---------------------------------------------------------------------------

_TRAINING_STATE_FILENAME = "training_state.json"


def save_training_state(
    accelerator,
    model,
    checkpoint_dir: Path,
    trained_tokens: int,
    step: int,
    next_eval: int,
    tokenizer,
) -> Path:
    """Save full training state (model, optimizer, scheduler, RNG) plus metadata.

    Uses ``accelerator.save_state()`` for optimizer/scheduler/RNG, then writes
    a small JSON with loop counters so the training loop can resume exactly
    where it left off.  All artifacts land on the data drive under
    *checkpoint_dir*.
    """
    ckpt_name = f"tokens_{trained_tokens}"
    ckpt_path = checkpoint_dir / ckpt_name
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Accelerate saves optimizer, scheduler, RNG, and model weights
    accelerator.save_state(str(ckpt_path))

    # Also save a standalone HF-format model snapshot so the checkpoint is
    # usable for inference without accelerate
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(ckpt_path / "hf_model", safe_serialization=True)
    tokenizer.save_pretrained(ckpt_path / "hf_model")

    # Save loop-level metadata that accelerate doesn't track
    state = {
        "trained_tokens": trained_tokens,
        "step": step,
        "next_eval": next_eval,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (ckpt_path / _TRAINING_STATE_FILENAME).open("w") as f:
        json.dump(state, f, indent=2)

    # Keep only the latest checkpoint to save disk space.
    if accelerator.is_main_process:
        for subdir in checkpoint_dir.iterdir():
            if (
                subdir.is_dir()
                and subdir.name.startswith("tokens_")
                and subdir.name != ckpt_name
            ):
                shutil.rmtree(subdir, ignore_errors=True)

    return ckpt_path


def save_training_state_signal_safe(
    accelerator,
    model,
    checkpoint_dir: Path,
    trained_tokens: int,
    step: int,
    next_eval: int,
    tokenizer,
) -> Path:
    """Save a checkpoint while deferring SIGINT/SIGTERM on Unix.

    This prevents a second Ctrl+C from interrupting Python while a checkpoint is
    half-written.  Deferred signals are delivered after the save completes.
    """
    blocked_signals = False
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        blocked_signals = True
    except AttributeError:
        pass

    try:
        return save_training_state(
            accelerator=accelerator,
            model=model,
            checkpoint_dir=checkpoint_dir,
            trained_tokens=trained_tokens,
            step=step,
            next_eval=next_eval,
            tokenizer=tokenizer,
        )
    finally:
        if blocked_signals:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent resumable checkpoint in *checkpoint_dir*.

    Looks for subdirectories matching ``tokens_<N>`` that contain both
    accelerator state and the training metadata JSON.  Returns the path
    with the highest token count, or ``None`` if no valid checkpoint exists.
    """
    if not checkpoint_dir.is_dir():
        return None

    best_path: Path | None = None
    best_tokens: int = -1

    for subdir in checkpoint_dir.iterdir():
        if not subdir.is_dir() or not subdir.name.startswith("tokens_"):
            continue
        state_file = subdir / _TRAINING_STATE_FILENAME
        if not state_file.exists():
            continue
        try:
            token_count = int(subdir.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if token_count > best_tokens:
            best_tokens = token_count
            best_path = subdir

    return best_path


def load_training_state(checkpoint_path: Path) -> dict:
    """Load loop-level metadata from a checkpoint directory."""
    with (checkpoint_path / _TRAINING_STATE_FILENAME).open() as f:
        return json.load(f)


def should_run_final_eval(final_metrics: dict[str, Any], trained_tokens: int) -> bool:
    """Return True when recorded metrics do not describe the final weights."""

    if not final_metrics:
        return True
    try:
        return int(final_metrics.get("tokens", -1)) < int(trained_tokens)
    except (TypeError, ValueError):
        return True


def next_eval_for_cadence(trained_tokens: int, eval_interval_tokens: int) -> int:
    """Return the next eval boundary for the current cadence."""

    if eval_interval_tokens <= 0:
        raise ValueError("eval_interval_tokens must be positive")
    return max(
        eval_interval_tokens,
        ((trained_tokens // eval_interval_tokens) + 1) * eval_interval_tokens,
    )


def best_effort_shutdown_dataloader_workers(*objects: object) -> int:
    """Explicitly stop DataLoader workers hidden behind Accelerate wrappers."""

    seen: set[int] = set()
    stack = [obj for obj in objects if obj is not None]
    candidates: list[object] = []

    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        candidates.append(obj)

        for attr in (
            "_iterator",
            "base_dataloader",
            "_dataset_fetcher",
            "dataset_iter",
            "_dataset_iter",
            "dataset",
            "_dataset",
        ):
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if child is not None:
                stack.append(child)

        frame = getattr(obj, "gi_frame", None)
        if frame is not None:
            for name in ("dataloader_iter", "self"):
                child = frame.f_locals.get(name)
                if child is not None:
                    stack.append(child)

    shutdown_count = 0
    for obj in candidates:
        shutdown = getattr(obj, "_shutdown_workers", None)
        if not callable(shutdown):
            continue
        try:
            shutdown()
            shutdown_count += 1
        except Exception:
            pass

    for obj in candidates:
        close = getattr(obj, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            pass

    gc.collect()
    return shutdown_count


def repair_eval_metadata(
    *,
    output_dir: Path,
    eval_dir: Path,
    final_metrics: dict[str, Any],
    reason: str = "eval_only_repair",
) -> None:
    """Rewrite local final metric metadata after a post-hoc eval-only run."""

    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, sort_keys=True)

    manifest_path = output_dir / "training_manifest.json"
    if not manifest_path.is_file():
        return

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["final_metrics"] = final_metrics
    repairs = manifest.setdefault("metadata_repairs", [])
    repairs.append(
        {
            "reason": reason,
            "repaired_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_saved_model_for_eval(model_dir: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import pellm  # noqa: F401
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")


def hard_exit_on_success_if_enabled() -> None:
    """Bypass fragile extension finalizers after all explicit cleanup is done."""

    if os.environ.get("PELLM_DISABLE_HARD_EXIT") == "1":
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    global _shutdown_requested

    args = parse_args()
    os.environ.update(apply_data_drive_env_defaults(os.environ))
    ensure_artifact_dirs()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    exp_name = cfg["experiment"]["name"]
    variant = variant_from_config(cfg, args.variant)
    token_budget = int(args.token_budget or cfg["training"]["token_budget"])

    output_dir = TRAINED_MODELS_DIR / exp_name / args.variant
    checkpoint_dir = CHECKPOINTS_DIR / exp_name / args.variant
    run_dir = RUNS_DIR / exp_name / args.variant
    eval_dir = EVALS_DIR / exp_name / args.variant
    stop_file = run_dir / "STOP_REQUESTED"

    resolved = {
        "experiment": exp_name,
        "variant": args.variant,
        "variant_config": asdict(variant),
        "token_budget": token_budget,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "run_dir": str(run_dir),
        "stop_file": str(stop_file),
        "eval_dir": str(eval_dir),
        "dataset": cfg["dataset"],
        "wandb": cfg["experiment"].get("wandb", {}),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    from accelerate import Accelerator
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg["training"]["grad_accum_steps"])
    )
    if accelerator.is_main_process:
        for path in (output_dir, checkpoint_dir, run_dir, eval_dir):
            path.mkdir(parents=True, exist_ok=True)
        with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
            json.dump(resolved, f, indent=2, sort_keys=True)
    accelerator.wait_for_everyone()

    wandb_run = None
    wandb_cfg = cfg["experiment"].get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", False)) and not args.eval_only
    if accelerator.is_main_process and wandb_enabled:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but wandb is not installed. "
                "Install with: pip install -e '.[experiments]'"
            ) from exc

        wandb_dir = run_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(wandb_dir))
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "pellm-smol-replacement"),
            entity=wandb_cfg.get("entity"),
            group=wandb_cfg.get("group", exp_name),
            name=wandb_cfg.get("name", f"{exp_name}-{args.variant}"),
            mode=wandb_cfg.get("mode", "online"),
            dir=str(wandb_dir),
            config=wandb_config(cfg, resolved),
            tags=wandb_cfg.get("tags", ["smol", "from-scratch", "pellm"]),
        )

    seed = int(cfg["training"]["seed"])
    torch.manual_seed(seed + accelerator.process_index)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    eval_dataset = PackedTextDataset(
        dataset_id=dataset_cfg["id"],
        dataset_config=dataset_cfg.get("config"),
        split=dataset_cfg.get(
            "validation_split", dataset_cfg.get("train_split", "train")
        ),
        text_column=dataset_cfg.get("text_column", "text"),
        tokenizer=tokenizer,
        sequence_length=int(train_cfg["sequence_length"]),
        seed=seed + 10_000,
        shuffle_buffer=int(dataset_cfg.get("eval_shuffle_buffer", 1000)),
        streaming=bool(dataset_cfg.get("streaming", True)),
    )
    # Eval is small (max_eval_batches every eval_interval_tokens) and gains
    # nothing from worker parallelism, while spawn-worker shutdown at the end
    # of each eval reliably trips a PyGILState_Release race in
    # huggingface_hub/hf_xet/aiohttp background threads.
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(train_cfg["micro_batch_size"]),
        num_workers=0,
    )

    if args.eval_only:
        model_dir = args.model_dir or output_dir
        model = load_saved_model_for_eval(model_dir)
        model, eval_loader = accelerator.prepare(model, eval_loader)
        metrics = evaluate(
            model,
            eval_loader,
            accelerator,
            int(train_cfg["max_eval_batches"]),
        )
        final_metrics = {
            "tokens": int(args.trained_tokens or token_budget),
            "step": None,
            **metrics,
        }
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"eval-only model={model_dir} {final_metrics}", flush=True)
            if args.repair_metadata:
                repair_eval_metadata(
                    output_dir=model_dir,
                    eval_dir=eval_dir,
                    final_metrics=final_metrics,
                )
                print(
                    f"Repaired eval summary and manifest metadata for {model_dir}",
                    flush=True,
                )
        accelerator.wait_for_everyone()
        accelerator.end_training()
        hard_exit_on_success_if_enabled()
        return

    model = build_model(cfg, variant)
    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    elif accelerator.is_main_process:
        print(
            "[pretrain] gradient_checkpointing=false "
            "(activation recompute disabled; faster but more VRAM)",
            flush=True,
        )

    train_dataset = PackedTextDataset(
        dataset_id=dataset_cfg["id"],
        dataset_config=dataset_cfg.get("config"),
        split=dataset_cfg.get("train_split", "train"),
        text_column=dataset_cfg.get("text_column", "text"),
        tokenizer=tokenizer,
        sequence_length=int(train_cfg["sequence_length"]),
        seed=seed,
        shuffle_buffer=int(dataset_cfg.get("shuffle_buffer", 10000)),
        streaming=bool(dataset_cfg.get("streaming", True)),
    )
    num_workers = int(train_cfg.get("num_workers", 0))
    # Streaming datasets rely on aiohttp/fsspec which is not fork-safe.
    # Use spawn when num_workers > 0 to avoid deadlocks after resolving files.
    mp_context = "spawn" if num_workers > 0 else None
    train_loader_kwargs: dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": int(train_cfg["micro_batch_size"]),
        "num_workers": num_workers,
    }
    if num_workers > 0:
        train_loader_kwargs.update(
            {
                "multiprocessing_context": mp_context,
                "persistent_workers": bool(train_cfg.get("persistent_workers", True)),
                "prefetch_factor": int(train_cfg.get("prefetch_factor", 2)),
            }
        )
    train_loader = DataLoader(**train_loader_kwargs)

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    seq_len = int(train_cfg["sequence_length"])
    tokens_per_update = (
        int(train_cfg["micro_batch_size"])
        * seq_len
        * int(train_cfg["grad_accum_steps"])
        * accelerator.num_processes
    )
    total_steps = math.ceil(token_budget / tokens_per_update)
    warmup_steps = max(1, int(total_steps * float(train_cfg["warmup_ratio"])))
    # AcceleratedScheduler advances the underlying scheduler num_processes times
    # per optimizer step (split_batches=False default), so scale the schedule
    # length to match. Mirrors HF's run_clm_no_trainer.py convention.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps * accelerator.num_processes,
        total_steps * accelerator.num_processes,
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )
    model.train()

    eval_interval_tokens = int(train_cfg["eval_interval_tokens"])
    max_eval_batches = int(train_cfg["max_eval_batches"])
    grad_clip = float(train_cfg["grad_clip"])
    next_eval = eval_interval_tokens
    trained_tokens = 0
    step = 0
    skip_batches = 0
    final_metrics: dict[str, Any] = {}
    log_interval_steps = int(train_cfg.get("log_interval_steps", 10))

    # ---- Resume from checkpoint if requested ----
    if args.resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            if accelerator.is_main_process:
                print(f"Resuming from checkpoint: {latest_ckpt}", flush=True)
            accelerator.load_state(str(latest_ckpt))
            state = load_training_state(latest_ckpt)
            trained_tokens = state["trained_tokens"]
            step = state["step"]
            stored_next_eval = state["next_eval"]
            # Re-align resumed eval/checkpoint cadence to the current config.
            # This lets cadence changes in YAML take effect after restart.
            next_eval = next_eval_for_cadence(trained_tokens, eval_interval_tokens)
            # Older checkpoints stored a separate "next_save" counter; ignore it
            # silently — checkpointing is now coupled to the eval cadence.

            # Calculate how many batches to skip in the streaming dataset.
            # Each batch contributes batch_size * seq_len * num_processes tokens.
            micro_batch_size = int(train_cfg["micro_batch_size"])
            tokens_per_batch = micro_batch_size * seq_len * accelerator.num_processes
            skip_batches = trained_tokens // tokens_per_batch
            if accelerator.is_main_process:
                print(
                    f"  Restored state: tokens={trained_tokens}, step={step}, "
                    f"next_eval={next_eval}",
                    flush=True,
                )
                if stored_next_eval != next_eval:
                    print(
                        f"  Re-aligned next_eval from checkpoint value "
                        f"{stored_next_eval} to current cadence value {next_eval}",
                        flush=True,
                    )
                print(
                    f"  Will skip {skip_batches} batches to fast-forward dataset",
                    flush=True,
                )
        else:
            if accelerator.is_main_process:
                print(
                    "No checkpoint found in checkpoint_dir — starting from scratch",
                    flush=True,
                )

    # Populate shared state so the signal handler can save on interrupt.
    _sig_state.clear()
    _sig_state.update(
        {
            "accelerator": accelerator,
            "model": model,
            "checkpoint_dir": checkpoint_dir,
            "trained_tokens": trained_tokens,
            "step": step,
            "next_eval": next_eval,
            "tokenizer": tokenizer,
        }
    )
    signal.signal(signal.SIGINT, _graceful_exit_handler)
    signal.signal(signal.SIGTERM, _graceful_exit_handler)

    start = time.time()
    emergency_shutdown = False

    def save_emergency_checkpoint(reason: str) -> Path:
        ckpt = save_training_state_signal_safe(
            accelerator=accelerator,
            model=model,
            checkpoint_dir=checkpoint_dir,
            trained_tokens=trained_tokens,
            step=step,
            next_eval=next_eval,
            tokenizer=tokenizer,
        )
        if accelerator.is_main_process:
            print(f"Saved emergency checkpoint ({reason}): {ckpt}", flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "checkpoint/tokens": trained_tokens,
                        "checkpoint/step": step,
                        "checkpoint/path": str(ckpt),
                        "checkpoint/emergency": True,
                        "checkpoint/reason": reason,
                    },
                    step=step,
                )
        return ckpt

    batches_skipped = 0
    train_iter = None
    try:
        train_iter = iter(train_loader)
        for batch in train_iter:
            if stop_file.exists() and not _shutdown_requested:
                if accelerator.is_main_process:
                    print(
                        f"Stop file detected at {stop_file}; "
                        "will checkpoint and exit after current step.",
                        flush=True,
                    )
                    try:
                        stop_file.unlink()
                    except FileNotFoundError:
                        pass
                _shutdown_requested = True

            # Fast-forward past already-consumed batches when resuming
            if batches_skipped < skip_batches:
                batches_skipped += 1
                if batches_skipped % 5000 == 0 and accelerator.is_main_process:
                    print(
                        f"  Skipping batch {batches_skipped}/{skip_batches}...",
                        flush=True,
                    )
                continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                accelerator.backward(outputs.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            batch_tokens = batch["input_ids"].numel() * accelerator.num_processes
            trained_tokens += batch_tokens
            if accelerator.sync_gradients:
                step += 1
                # Keep signal handler state fresh.
                _sig_state["trained_tokens"] = trained_tokens
                _sig_state["step"] = step
                _sig_state["next_eval"] = next_eval

            if (
                accelerator.is_main_process
                and step % log_interval_steps == 0
                and accelerator.sync_gradients
            ):
                elapsed = max(time.time() - start, 1e-6)
                loss_value = outputs.loss.detach().float().item()
                tokens_per_second = trained_tokens / elapsed
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"step={step} tokens={trained_tokens} "
                    f"loss={loss_value:.4f} "
                    f"lr={current_lr:.3e} "
                    f"tok_s={tokens_per_second:.1f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss_value,
                            "train/tokens": trained_tokens,
                            "train/tokens_per_second": tokens_per_second,
                            "train/lr": current_lr,
                            "train/step": step,
                        },
                        step=step,
                    )

            if trained_tokens >= next_eval and accelerator.sync_gradients:
                metrics = evaluate(model, eval_loader, accelerator, max_eval_batches)
                final_metrics = {"tokens": trained_tokens, "step": step, **metrics}
                if accelerator.is_main_process:
                    with (eval_dir / f"eval_{trained_tokens}.json").open(
                        "w", encoding="utf-8"
                    ) as f:
                        json.dump(final_metrics, f, indent=2, sort_keys=True)
                    print(f"eval tokens={trained_tokens} {metrics}", flush=True)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "eval/loss": metrics["loss"],
                                "eval/perplexity": metrics["perplexity"],
                                "eval/tokens": trained_tokens,
                                "eval/step": step,
                            },
                            step=step,
                        )
                next_eval += eval_interval_tokens
                _sig_state["next_eval"] = next_eval

                # Save a resumable checkpoint immediately after every eval so any
                # interruption between evals leaves a usable resume point.
                accelerator.wait_for_everyone()
                ckpt = save_training_state(
                    accelerator=accelerator,
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    trained_tokens=trained_tokens,
                    step=step,
                    next_eval=next_eval,
                    tokenizer=tokenizer,
                )
                if accelerator.is_main_process:
                    print(f"Saved resumable checkpoint: {ckpt}", flush=True)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "checkpoint/tokens": trained_tokens,
                                "checkpoint/step": step,
                                "checkpoint/path": str(ckpt),
                            },
                            step=step,
                        )

            if _shutdown_requested and accelerator.sync_gradients:
                accelerator.wait_for_everyone()
                save_emergency_checkpoint("requested_shutdown")
                emergency_shutdown = True
                break

            if trained_tokens >= token_budget or step >= total_steps:
                break
    except BaseException as exc:
        # Ctrl+C under `accelerate launch` can tear down the parent/DataLoader
        # before the loop reaches the normal shutdown branch.  Preserve the last
        # completed optimizer step whenever possible.
        should_treat_as_shutdown = _shutdown_requested or isinstance(
            exc, (KeyboardInterrupt, SystemExit)
        )
        if step > 0:
            reason = (
                "requested_shutdown"
                if should_treat_as_shutdown
                else f"exception_{type(exc).__name__}"
            )
            try:
                save_emergency_checkpoint(reason)
            except BaseException as save_exc:
                if accelerator.is_main_process:
                    print(
                        f"Failed to save emergency checkpoint after "
                        f"{type(exc).__name__}: {save_exc}",
                        flush=True,
                    )
                raise exc from save_exc
        if not should_treat_as_shutdown:
            raise
        emergency_shutdown = True
    finally:
        best_effort_shutdown_dataloader_workers(train_iter, train_loader)
        train_iter = None

    if emergency_shutdown:
        if accelerator.is_main_process and wandb_run is not None:
            wandb_run.finish()
        accelerator.end_training()
        hard_exit_on_success_if_enabled()
        return

    if should_run_final_eval(final_metrics, trained_tokens):
        metrics = evaluate(model, eval_loader, accelerator, max_eval_batches)
        final_metrics = {"tokens": trained_tokens, "step": step, **metrics}
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2, sort_keys=True)
        save_manifest(
            cfg=cfg,
            variant_name=args.variant,
            variant=variant,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir,
            eval_dir=eval_dir,
            training_cmd=" ".join(sys.argv),
            token_budget=token_budget,
            final_metrics=final_metrics,
        )
        print(f"Saved final model to {output_dir}")
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "final/loss": final_metrics.get("loss"),
                    "final/perplexity": final_metrics.get("perplexity"),
                    "final/tokens": final_metrics.get("tokens", trained_tokens),
                    "final/model_dir": str(output_dir),
                    "final/eval_summary": str(eval_dir / "summary.json"),
                }
            )
            wandb_run.finish()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    hard_exit_on_success_if_enabled()


if __name__ == "__main__":
    main()
