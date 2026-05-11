#!/usr/bin/env python3
"""Publish a validated PELLM checkpoint directory to Hugging Face Hub.

The script uploads model weights directly from the final checkpoint directory
and stages lightweight metadata files under ``<pellm_data_root>/hf_staging``.
It defaults to private repos so draft paper artifacts do not become public
before the results are ready.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_paths import HF_STAGING_DIR, repo_id_for_variant

REQUIRED_MODEL_FILES = ("config.json",)
WEIGHT_PATTERNS = ("*.safetensors",)
TOKENIZER_PATTERNS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a PELLM model to HF Hub.")
    parser.add_argument(
        "--variant",
        choices=["baseline", "ae_mlp", "trendwavelet", "ae_tw"],
        required=True,
        help="Paper model variant; selects the default Hugging Face repo id.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Final HuggingFace-format checkpoint directory to publish.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Override the default Hugging Face repo id for the selected variant.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to training_manifest.json. Defaults to <model-dir>/training_manifest.json.",
    )
    parser.add_argument(
        "--eval-summary",
        type=Path,
        action="append",
        default=[],
        help="Evaluation summary file to upload. May be supplied multiple times.",
    )
    parser.add_argument(
        "--allow-missing-evals",
        action="store_true",
        help="Allow publishing without eval summary files. Intended only for draft uploads.",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=None,
        help="Custom README.md/model card. If omitted, a standard card is generated.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create/update the Hub repo as public. Default is private.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Hub commit message.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print upload actions without contacting Hugging Face.",
    )
    return parser.parse_args()


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


def has_any(path: Path, patterns: tuple[str, ...]) -> bool:
    return any(next(path.glob(pattern), None) is not None for pattern in patterns)


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    for filename in REQUIRED_MODEL_FILES:
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(f"Missing required model file: {model_dir / filename}")
    if not has_any(model_dir, WEIGHT_PATTERNS):
        raise FileNotFoundError(
            f"No safetensors model weights found in {model_dir}."
        )
    if not has_any(model_dir, TOKENIZER_PATTERNS):
        raise FileNotFoundError(
            f"No tokenizer files found in {model_dir}; expected a saved tokenizer."
        )


def load_manifest(manifest_path: Path | None, model_dir: Path) -> tuple[dict, Path]:
    candidate = manifest_path or model_dir / "training_manifest.json"
    if candidate.is_file():
        with candidate.open("r", encoding="utf-8") as f:
            return json.load(f), candidate
    raise FileNotFoundError(
        f"Missing training manifest: {candidate}. "
        "Publishable paper checkpoints must include training_manifest.json."
    )


def copy_metadata_file(source: Path, target_dir: Path, target_name: str | None = None) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {source}")
    target = target_dir / (target_name or source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def generate_model_card(
    *,
    variant: str,
    repo_id: str,
    model_dir: Path,
    manifest: dict,
    eval_files: list[Path],
    target: Path,
) -> None:
    config = {}
    config_path = model_dir / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

    dataset_id = manifest.get("dataset_id", manifest.get("dataset", "not recorded"))
    token_budget = manifest.get("token_budget", manifest.get("train_tokens", "not recorded"))
    hardware = manifest.get("hardware", "2x NVIDIA RTX 5090, 32GB VRAM each")
    architecture = manifest.get("architecture", variant)
    eval_list = "\n".join(f"- `{p.name}`" for p in eval_files) or "- Not included yet"

    text = f"""---
license: apache-2.0
tags:
- text-generation
- causal-lm
- llama
- pellm
- parameter-efficient
library_name: transformers
base_model: none
---

# {repo_id}

This is a PELLM paper checkpoint for the `{variant}` variant.

## Model Details

- Architecture variant: `{architecture}`
- Dataset: `{dataset_id}`
- Training token budget: `{token_budget}`
- Hardware: `{hardware}`
- Git commit: `{manifest.get("git_commit", git_commit())}`
- Hidden size: `{config.get("hidden_size", "not recorded")}`
- Layers: `{config.get("num_hidden_layers", "not recorded")}`
- Attention heads: `{config.get("num_attention_heads", "not recorded")}`
- KV heads: `{config.get("num_key_value_heads", "not recorded")}`

## Loading

PE checkpoints require importing `pellm` before loading so the custom
`pe_llama` config and model classes are registered with Transformers.

```python
import pellm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
```

## Included Evaluation Summaries

{eval_list}

## Reproducibility

The repository should include `training_manifest.json` with the exact dataset
IDs, token budget, seeds, git commit, and relevant package versions used for
the run. Large raw eval dumps are intentionally kept off-repo and off-model-card
unless promoted to summarized artifacts.

Generated on {datetime.now(timezone.utc).isoformat()}.
"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    repo_id = args.repo_id or repo_id_for_variant(args.variant)
    commit_message = args.commit_message or f"Publish {args.variant} PELLM checkpoint"

    validate_model_dir(model_dir)
    if not args.eval_summary and not args.allow_missing_evals:
        raise ValueError(
            "At least one --eval-summary is required for publishable paper checkpoints. "
            "Use --allow-missing-evals only for draft uploads."
        )
    manifest, manifest_source = load_manifest(args.manifest, model_dir)

    staging_dir = HF_STAGING_DIR / repo_id.replace("/", "__")
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_files: list[Path] = []
    staged_files.append(copy_metadata_file(manifest_source, staging_dir, "training_manifest.json"))

    eval_files = []
    for eval_summary in args.eval_summary:
        staged = copy_metadata_file(eval_summary.expanduser().resolve(), staging_dir / "evals")
        staged_files.append(staged)
        eval_files.append(staged)

    if args.model_card:
        staged_files.append(copy_metadata_file(args.model_card.expanduser().resolve(), staging_dir, "README.md"))
    else:
        readme_path = staging_dir / "README.md"
        generate_model_card(
            variant=args.variant,
            repo_id=repo_id,
            model_dir=model_dir,
            manifest=manifest,
            eval_files=eval_files,
            target=readme_path,
        )
        staged_files.append(readme_path)

    print(f"Model dir: {model_dir}")
    print(f"Repo id:   {repo_id}")
    print(f"Private:   {not args.public}")
    print(f"Staging:   {staging_dir}")
    print("Metadata files:")
    for path in staged_files:
        print(f"  {path}")

    if args.dry_run:
        print("\nDRY RUN: upload skipped.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=commit_message,
    )
    for path in staged_files:
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(path),
            path_in_repo=str(path.relative_to(staging_dir)),
            commit_message=commit_message,
        )
    print(f"Uploaded {repo_id}")


if __name__ == "__main__":
    main()
