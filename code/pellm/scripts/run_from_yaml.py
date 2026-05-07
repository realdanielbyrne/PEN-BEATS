#!/usr/bin/env python3
"""
YAML-driven experiment orchestrator for PE-Llama.

Loads an experiment config from a YAML file, expands any list-valued
parameters into a Cartesian product of runs, dispatches each run as a
subprocess call to scripts/finetune.py, and collects the results into
a JSON summary and CSV log.

Usage::

    python scripts/run_from_yaml.py scripts/experiments/my_sweep.yaml
    python scripts/run_from_yaml.py scripts/experiments/my_sweep.yaml --gpu-id 1
    python scripts/run_from_yaml.py scripts/experiments/my_sweep.yaml --dry-run
    python scripts/run_from_yaml.py scripts/experiments/my_sweep.yaml --config ae_lg_layer15

YAML schema::

    experiment:
      name: "my_sweep"
      python: ".venv/bin/python"      # default: "python3"
      env:                            # merged into subprocess environment
        PYTORCH_ALLOC_CONF: "expandable_segments:True"
      wandb: false
      wandb_project: "pellm-experiments"
      run_root: "<pellm_data_root>/runs"
      models_root: "<pellm_data_root>/trainedmodels"
      cache_root: "<pellm_data_root>/runs"
    # Paths are derived from experiment.name:
    #   logs:   {run_root}/{name}/logs/
    #   models: {models_root}/{name}/
    #   caches: {cache_root}/{name}/activation_caches/

    defaults:                         # applied to every run unless overridden
      dtype: "bfloat16"
      batch_size: 2
      epochs: 3
      lr: 1e-4
      freeze_base: true
      ...

    configs:
      - name: "attn_sweep"
        pe_attn_mode: ["trend_wavelet_lg", "svd_lg"]  # list → grid search
        trend_dim: [3, 4]                              # list × 2 → 4 total runs
        pe_layer_indices: [[12,13,14,15]]              # list-of-lists → single nargs arg

Grid search: any YAML param whose value is a Python list is swept.
Exception: nargs fields (pe_layer_indices, pe_mlp_layer_indices, etc.)
use list-of-lists — [[12,13,14,15]] is ONE argument value, not a sweep;
[[12,13,14,15],[8,9,10,11]] sweeps over two distinct layer sets.

Requires: pip install -e ".[experiments]"
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import cast

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_paths import (  # noqa: E402
    RUNS_DIR,
    TRAINED_MODELS_DIR,
    apply_data_drive_env_defaults,
)

# ---------------------------------------------------------------------------
# Parameter → CLI flag mapping
# ---------------------------------------------------------------------------

PARAM_TO_FLAG: dict[str, str] = {
    "model_name": "--model-name",
    "pe_attn_mode": "--pe-attn-mode",
    "trend_dim": "--trend-dim",
    "wavelet_dim": "--wavelet-dim",
    "wavelet_type": "--wavelet-type",
    "wavelet_basis_offset": "--wavelet-basis-offset",
    "per_layer_offsets": "--per-layer-offsets",
    "svd_rank": "--svd-rank",
    "generic_dim": "--generic-dim",
    "reduction_dim": "--reduction-dim",
    "attn_init": "--attn-init",
    "pe_projections": "--pe-projections",
    "pe_layer_indices": "--pe-layer-indices",
    "pe_mlp_mode": "--pe-mlp-mode",
    "ae_latent_dim": "--ae-latent-dim",
    "pe_mlp_layer_indices": "--pe-mlp-layer-indices",
    "ae_init": "--ae-init",
    "ae_inner_init": "--ae-inner-init",
    "epochs": "--epochs",
    "lr": "--lr",
    "patience": "--patience",
    "batch_size": "--batch-size",
    "max_length": "--max-length",
    "grad_accum_steps": "--grad-accum-steps",
    "max_eval_batches": "--max-eval-batches",
    "resume_from": "--resume-from",
    "freeze_base": "--freeze-base",
    "freeze_except_layers": "--freeze-except-layers",
    "attn_pretrain_epochs": "--attn-pretrain-epochs",
    "attn_pretrain_early_stopping": "--attn-pretrain-early-stopping",
    "attn_pretrain_warmup": "--attn-pretrain-warmup",
    "attn_pretrain_patience": "--attn-pretrain-patience",
    "ae_pretrain_epochs": "--ae-pretrain-epochs",
    "ae_pretrain_layer_indices": "--ae-pretrain-layer-indices",
    "ae_pretrain_early_stopping": "--ae-pretrain-early-stopping",
    "ae_pretrain_warmup": "--ae-pretrain-warmup",
    "ae_pretrain_patience": "--ae-pretrain-patience",
    "ae_pretrain_lr": "--ae-pretrain-lr",
    "ae_pretrain_scheduler": "--ae-pretrain-scheduler",
    "ae_pretrain_lr_warmup": "--ae-pretrain-lr-warmup",
    "ae_pretrain_gamma": "--ae-pretrain-gamma",
    "attn_cache_dir": "--attn-cache-dir",
    "attn_dataset": "--attn-dataset",
    "attn_cache_num_samples": "--attn-cache-num-samples",
    "dataset": "--dataset",
    "dataset_num_samples": "--dataset-num-samples",
    "ae_cache_dir": "--ae-cache-dir",
    "ae_dataset": "--ae-dataset",
    "ae_cache_num_samples": "--ae-cache-num-samples",
    "ae_cache_skip": "--ae-cache-skip",
    "ae_pretrain_loss": "--ae-pretrain-loss",
    "ae_pretrain_loss_temperature": "--ae-pretrain-loss-temperature",
    "ae_pretrain_loss_alpha": "--ae-pretrain-loss-alpha",
    "ae_pretrain_resample": "--ae-pretrain-resample",
    "ae_pretrain_fineweb_epochs": "--ae-pretrain-fineweb-epochs",
    "output_dir": "--output-dir",
    "shared_tokenizer_dir": "--shared-tokenizer-dir",
    "dtype": "--dtype",
    "seed": "--seed",
    "wandb": "--wandb",
    "wandb_project": "--wandb-project",
    "log_csv": "--log-csv",
    "local_files_only": "--local-files-only",
    "kd_alpha": "--kd-alpha",
    "kd_temperature": "--kd-temperature",
    "kd_teacher": "--kd-teacher",
    "kd_attn_weight": "--kd-attn-weight",
    "kd_attn_layers": "--kd-attn-layers",
    "ae_teacher": "--ae-teacher",
    "attn_teacher": "--attn-teacher",
    "active_g_pretrain": "--active-g-pretrain",
    "active_g_finetune": "--active-g-finetune",
}

# Fields that map to nargs="+" in finetune.py.
# In YAML, specify as list-of-lists: [[12,13,14,15]] for one value,
# [[0,1,2,3],[12,13,14,15]] to sweep two distinct layer sets.
NARGS_FIELDS: frozenset[str] = frozenset(
    {
        "per_layer_offsets",
        "pe_projections",
        "pe_layer_indices",
        "pe_mlp_layer_indices",
        "ae_pretrain_layer_indices",
        "freeze_except_layers",
        "kd_attn_layers",
    }
)

# Fields that map to store_true flags in finetune.py.
BOOL_FLAGS: frozenset[str] = frozenset(
    {
        "freeze_base",
        "wandb",
        "local_files_only",
        "attn_pretrain_early_stopping",
        "ae_pretrain_early_stopping",
        "ae_pretrain_resample",
        "active_g_pretrain",
        "active_g_finetune",
    }
)

# Keys in the YAML that are orchestrator-level (not forwarded to finetune.py).
_ORCHESTRATOR_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "output_dir",  # output_dir is injected by orchestrator
        "log_csv",  # CSV is written by the orchestrator, not finetune.py
        "run_root",
        "models_root",
        "cache_root",
    }
)

# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print(
            "ERROR: PyYAML not found. Install with: pip install -e '.[experiments]'",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping at top level: {path}")
    return cfg


# ---------------------------------------------------------------------------
# Config merging and grid expansion
# ---------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Shallow key-level merge: override wins."""
    return {**base, **override}


def expand_grid(cfg: dict) -> list[dict]:
    """Expand list-valued parameters into all Cartesian product combinations.

    For NARGS_FIELDS: a list-of-lists with >1 element triggers a sweep;
    a single-element list-of-lists collapses to its inner list (one value).
    For all other fields: a plain list triggers a sweep.
    """
    grid_keys: list[str] = []
    grid_values: list[list] = []
    fixed: dict = {}

    for key, value in cfg.items():
        if key in NARGS_FIELDS:
            if (
                isinstance(value, list)
                and len(value) > 1
                and isinstance(value[0], list)
            ):
                # list-of-lists with multiple elements → sweep
                grid_keys.append(key)
                grid_values.append(value)
            elif (
                isinstance(value, list)
                and len(value) == 1
                and isinstance(value[0], list)
            ):
                # single-element list-of-lists → unwrap to the inner list
                fixed[key] = value[0]
            elif isinstance(value, list) and not isinstance(value[0], list):
                # plain list for a nargs field → treat as a single multi-value arg
                fixed[key] = value
            else:
                fixed[key] = value
        elif key in BOOL_FLAGS:
            if isinstance(value, list):
                grid_keys.append(key)
                grid_values.append(value)
            else:
                fixed[key] = value
        elif isinstance(value, list):
            grid_keys.append(key)
            grid_values.append(value)
        else:
            fixed[key] = value

    if not grid_keys:
        return [fixed]

    combos = []
    for combo_values in itertools.product(*grid_values):
        combo = {**fixed, **dict(zip(grid_keys, combo_values))}
        combos.append(combo)
    return combos


# ---------------------------------------------------------------------------
# Run ID and command building
# ---------------------------------------------------------------------------


def make_run_id(experiment_name: str, config_name: str, combo_index: int) -> str:
    return f"{experiment_name}_{config_name}_{combo_index:03d}"


def build_cmd(
    python: str,
    script: str,
    params: dict,
    output_dir: str,
    wandb_cfg: dict,
    finetune_dry_run: bool = False,
    save_model: bool = True,
) -> list[str]:
    """Build the subprocess argv list for finetune.py.

    *finetune_dry_run=True* appends ``--dry-run`` so finetune.py loads the
    model from local cache and validates config without training.

    Note: ``--log-csv`` is NOT passed to finetune.py because the orchestrator
    writes its own CSV with richer metadata (run_id, config_name, derived
    metrics).  finetune.py's ``--log-csv`` flag still works for standalone
    runs outside the orchestrator.
    """
    cmd = [python, script]

    for key, value in params.items():
        if key in _ORCHESTRATOR_KEYS:
            continue
        flag = PARAM_TO_FLAG.get(key)
        if flag is None:
            continue
        if key in BOOL_FLAGS:
            if value:
                cmd.append(flag)
            # False → omit flag entirely
        elif key in NARGS_FIELDS:
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        elif value is None:
            continue  # skip None values
        else:
            cmd.extend([flag, str(value)])

    # Inject output_dir so finetune.py saves model checkpoints.
    # When save_model is False the orchestrator still uses output_dir
    # internally for sentinel / claim tracking but finetune.py won't
    # persist model weights to disk.
    if save_model:
        cmd.extend(["--output-dir", output_dir])

    # Wandb from experiment-level config
    if wandb_cfg.get("enabled"):
        cmd.append("--wandb")
        if wandb_cfg.get("project"):
            cmd.extend(["--wandb-project", wandb_cfg["project"]])

    if finetune_dry_run:
        cmd.append("--dry-run")

    return cmd


# ---------------------------------------------------------------------------
# Subprocess execution and result parsing
# ---------------------------------------------------------------------------


def run_single(cmd: list[str], env: dict) -> tuple[int, str, str]:
    """Run a subprocess, streaming output to the terminal in real time.

    stdout and stderr are interleaved on the terminal (both piped through
    stdout) while still being captured separately for result parsing.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,  # line-buffered
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    import selectors

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    sel.register(proc.stderr, selectors.EVENT_READ)

    open_streams = 2
    while open_streams > 0:
        for key, _ in sel.select():
            stream = cast(TextIOWrapper, key.fileobj)
            line = stream.readline()
            if not line:
                sel.unregister(stream)
                open_streams -= 1
                continue
            if stream is proc.stdout:
                stdout_lines.append(line)
                sys.stdout.write(f"    {line}")
            else:
                stderr_lines.append(line)
                sys.stderr.write(f"    {line}")

    sel.close()
    proc.wait()
    return proc.returncode, "".join(stdout_lines), "".join(stderr_lines)


def parse_results(stdout: str) -> dict | None:
    """Extract the Results JSON printed by finetune.py."""
    marker = "Results: "
    idx = stdout.rfind(marker)
    if idx == -1:
        return None
    try:
        return json.loads(stdout[idx + len(marker) :])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def is_run_complete(output_dir: str) -> bool:
    """Return True if this run has already been completed."""
    p = Path(output_dir)
    return (p / "run_complete.json").exists() or (p / "config.json").exists()


def write_sentinel(output_dir: str, results: dict) -> None:
    """Write run_complete.json to mark this run as done."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "run_complete.json", "w") as f:
        json.dump(results, f, indent=2)


_CLAIM_FILENAME = "run_in_progress.lock"
_WORKER_ID = os.environ.get("PELLM_WORKER_ID")


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with the given PID exists.

    ``os.kill(pid, 0)`` raises ``ProcessLookupError`` when the PID does not
    exist, and ``PermissionError`` when it exists but we lack permission to
    signal it (e.g. PID 1).  Both are ``OSError`` subclasses, so we must
    catch ``PermissionError`` first and treat it as "alive".
    """
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # process exists, we just can't signal it
    except ProcessLookupError:
        return False


def claim_run(output_dir: str) -> bool:
    """Attempt to claim a run by writing a lock file atomically.

    Returns ``True`` if the claim was acquired (no other live process holds
    it), ``False`` if another process already owns the run.  Stale locks
    (from a process that is no longer alive) are automatically cleaned up
    and re-claimed.
    """
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    lock_path = p / _CLAIM_FILENAME

    if lock_path.exists():
        try:
            with open(lock_path) as f:
                lock_data = json.load(f)
            owner_pid = lock_data.get("pid")
            if owner_pid is not None:
                owner_pid = int(owner_pid)
                if owner_pid == os.getpid():
                    return True  # we already own this lock
                if _pid_is_alive(owner_pid):
                    return False  # another live process owns this run
            # Stale lock — previous owner is gone; reclaim below
        except (json.JSONDecodeError, ValueError, OSError):
            pass  # corrupt lock file — reclaim below

    lock_data = {
        "pid": os.getpid(),
        "worker_id": _WORKER_ID,
        "timestamp": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
    }
    # Write atomically via temp file + rename to avoid partial reads
    tmp_path = lock_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(lock_data, f)
    tmp_path.rename(lock_path)
    return True


def is_run_claimed(output_dir: str) -> bool:
    """Return True if another live process has claimed this run."""
    lock_path = Path(output_dir) / _CLAIM_FILENAME
    if not lock_path.exists():
        return False
    try:
        with open(lock_path) as f:
            lock_data = json.load(f)
        owner_pid = lock_data.get("pid")
        if owner_pid is None:
            return False
        if int(owner_pid) == os.getpid():
            return False  # we own it ourselves
        return _pid_is_alive(int(owner_pid))
    except (json.JSONDecodeError, ValueError, OSError):
        return False


def release_claim(output_dir: str) -> None:
    """Remove the lock file for a run (called on completion or failure)."""
    lock_path = Path(output_dir) / _CLAIM_FILENAME
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Summary persistence (JSON = orchestration metadata + nested data)
# ---------------------------------------------------------------------------


def save_summary(results_list: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_list, f, indent=2)


# ---------------------------------------------------------------------------
# CSV persistence (flat scalars for statistical analysis)
# ---------------------------------------------------------------------------

# Result keys that contain nested data — kept in JSON only, excluded from CSV.
_NESTED_RESULT_KEYS: frozenset[str] = frozenset(
    {
        "attn_pretrain_stats",
        "ae_pretrain_stats",
        "per_epoch_stats",
    }
)

# Stable column ordering: orchestrator fields → result scalars → params.
# DictWriter uses this as the canonical header; missing keys become empty cells.
_CSV_META_FIELDS: list[str] = [
    "run_id",
    "config_name",
    "combo_index",
    "status",
]

_CSV_RESULT_FIELDS: list[str] = [
    "original_ppl",
    "baseline_ppl",
    "attn_pretrain_ppl",
    "ae_pretrain_ppl",
    "final_ppl",
    "ppl_improvement_pct",
    "params_total",
    "params_trainable",
    "epochs_completed",
    "best_val_ppl",
    "lr",
    "attn_pretrain_mse",
    "ae_pretrain_loss",
]

# Param fields derived from PARAM_TO_FLAG ordering (stable across runs).
# Computed once at import time so the header never shifts.
_CSV_PARAM_FIELDS: list[str] = [
    k
    for k in PARAM_TO_FLAG
    if k
    not in {
        "output_dir",
        "shared_tokenizer_dir",
        "log_csv",
        "wandb",
        "wandb_project",
        "attn_cache_dir",
        "ae_cache_dir",
        "lr",
    }  # lr already in result fields
]

CSV_FIELDNAMES: list[str] = _CSV_META_FIELDS + _CSV_RESULT_FIELDS + _CSV_PARAM_FIELDS


def _flatten_for_csv(value):
    """Serialize lists/dicts as JSON strings; pass scalars through."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def append_csv_row(
    csv_path: Path,
    run_id: str,
    config_name: str,
    combo_index: int,
    params: dict,
    results: dict | None,
    status: str,
) -> None:
    """Append one flat CSV row combining orchestrator metadata, params, and result scalars.

    Nested result fields (per-epoch stats, AE pretrain stats) are excluded —
    they live in the JSON summary only.
    """
    row: dict = {
        "run_id": run_id,
        "config_name": config_name,
        "combo_index": combo_index,
        "status": status,
    }

    # Flatten params (JSON-encode any list-valued params like layer indices)
    for key, value in params.items():
        if key in CSV_FIELDNAMES:
            row[key] = _flatten_for_csv(value)

    # Merge scalar results (skip nested keys)
    if results is not None:
        for key, value in results.items():
            if key in _NESTED_RESULT_KEYS:
                continue
            row[key] = _flatten_for_csv(value)

        # Compute derived metrics
        baseline = results.get("baseline_ppl")
        final = results.get("final_ppl")
        if baseline is not None and final is not None:
            try:
                row["ppl_improvement_pct"] = round(
                    100.0 * (baseline - final) / baseline, 2
                )
            except (TypeError, ZeroDivisionError):
                pass

        # Extract epochs_completed and best_val_ppl from per_epoch_stats
        per_epoch = results.get("per_epoch_stats")
        if per_epoch and isinstance(per_epoch, list):
            row["epochs_completed"] = len(per_epoch)
            val_ppls = [
                e.get("val_ppl") for e in per_epoch if e.get("val_ppl") is not None
            ]
            if val_ppls:
                row["best_val_ppl"] = min(val_ppls)

        # Flatten attn_pretrain_stats into a single JSON list of MSE floats
        attn_stats = results.get("attn_pretrain_stats")
        if attn_stats and isinstance(attn_stats, list):
            mse_values = [
                e.get("mse_loss") for e in attn_stats if e.get("mse_loss") is not None
            ]
            if mse_values:
                row["attn_pretrain_mse"] = json.dumps(mse_values)

        # Flatten ae_pretrain_stats into a single JSON list of loss floats
        # (stat key is "ae_loss"; accept legacy "mse_loss" for older runs)
        # Note: CSV column name is "ae_pretrain_loss" to reflect that it can hold
        # losses other than MSE (cosine, huber, soft_kl, combined).
        ae_stats = results.get("ae_pretrain_stats")
        if ae_stats and isinstance(ae_stats, list):
            loss_values = [
                e.get("ae_loss", e.get("mse_loss"))
                for e in ae_stats
                if e.get("ae_loss", e.get("mse_loss")) is not None
            ]
            if loss_values:
                row["ae_pretrain_loss"] = json.dumps(loss_values)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YAML-driven experiment orchestrator for PE-Llama"
    )
    p.add_argument("yaml", help="Path to experiment YAML config file")
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them"
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run finetune.py --dry-run for each config: loads the model "
        "from local cache and validates config/params without training",
    )
    p.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Re-run experiments even if output directory already exists",
    )
    p.add_argument(
        "--config",
        nargs="+",
        metavar="NAME",
        help="Only run the named config blocks (by 'name' field)",
    )
    p.add_argument(
        "--worker-id",
        help="Optional label for this worker process, recorded in lock files and logs",
    )
    p.add_argument(
        "--gpu-id",
        help="Optional GPU id to expose via CUDA_VISIBLE_DEVICES for all subprocesses",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global _WORKER_ID
    if args.worker_id:
        _WORKER_ID = args.worker_id
        os.environ["PELLM_WORKER_ID"] = args.worker_id

    cfg = load_yaml(args.yaml)

    experiment = cfg.get("experiment", {})
    exp_name = experiment.get("name", "experiment")
    python = experiment.get("python", "python3")
    extra_env = experiment.get("env", {})
    save_model = experiment.get("save_model", True)
    wandb_cfg = {
        "enabled": experiment.get("wandb", False),
        "project": experiment.get("wandb_project", "pellm"),
    }

    _scripts_dir = Path(__file__).resolve().parent
    script = str(_scripts_dir / "finetune.py")

    # Heavy run artifacts default to the large data drive.
    run_root = Path(experiment.get("run_root", RUNS_DIR)).expanduser()
    models_root = Path(experiment.get("models_root", TRAINED_MODELS_DIR)).expanduser()
    cache_root_cfg = experiment.get("cache_root")
    cache_root = (
        Path(cache_root_cfg).expanduser() / exp_name
        if cache_root_cfg
        else run_root / exp_name / "activation_caches"
    )

    logs_dir = run_root / exp_name / "logs"
    summary_path = logs_dir / f"{exp_name}_results.json"
    csv_path = logs_dir / f"{exp_name}_log.csv"

    models_dir = models_root / exp_name

    defaults = cfg.get("defaults", {})
    config_blocks = cfg.get("configs", [])

    if not config_blocks:
        print("ERROR: No 'configs' list found in YAML.", file=sys.stderr)
        sys.exit(1)

    # Filter config blocks if --config was specified
    if args.config:
        config_blocks = [b for b in config_blocks if b.get("name") in args.config]
        if not config_blocks:
            print(f"ERROR: No config blocks matched: {args.config}", file=sys.stderr)
            sys.exit(1)

    subprocess_env = apply_data_drive_env_defaults(
        {**os.environ, **{str(k): str(v) for k, v in extra_env.items()}}
    )
    if args.gpu_id is not None:
        subprocess_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Shared activation caches: when pretraining is enabled in defaults and no
    # explicit cache dir is provided, create a shared cache dir so teacher
    # activations are extracted once by the first run and reused by all others.
    _shared_attn_cache_dir: str | None = None
    _attn_pretrain_epochs = defaults.get("attn_pretrain_epochs", 0)
    _has_user_attn_cache = defaults.get("attn_cache_dir") is not None or any(
        b.get("attn_cache_dir") is not None for b in config_blocks
    )
    if _attn_pretrain_epochs and _attn_pretrain_epochs > 0 and not _has_user_attn_cache:
        _shared_attn_cache_dir = str(cache_root / "attn_activation_cache")
        os.makedirs(_shared_attn_cache_dir, exist_ok=True)
        print(f"Shared attention cache: {_shared_attn_cache_dir}")

    _shared_ae_cache_dir: str | None = None
    _ae_pretrain_epochs = defaults.get("ae_pretrain_epochs", 0)
    _ae_pretrain_resample = defaults.get("ae_pretrain_resample", False)
    _has_user_ae_cache = defaults.get("ae_cache_dir") is not None or any(
        b.get("ae_cache_dir") is not None for b in config_blocks
    )
    # Don't create a shared cache when per-epoch resampling is enabled — finetune.py
    # manages its own per-epoch temp dirs and cleans them up automatically. Injecting
    # a shared cache_dir forces a full upfront extraction that can exhaust disk space.
    if _ae_pretrain_epochs and _ae_pretrain_epochs > 0 and not _has_user_ae_cache and not _ae_pretrain_resample:
        _shared_ae_cache_dir = str(cache_root / "ae_activation_cache")
        os.makedirs(_shared_ae_cache_dir, exist_ok=True)
        print(f"Shared AE cache: {_shared_ae_cache_dir}")

    all_results: list[dict] = []
    total_runs = 0
    skipped = 0

    # Count total runs for progress display
    for i, block in enumerate(config_blocks):
        merged = deep_merge(defaults, {k: v for k, v in block.items() if k != "name"})
        combos = expand_grid(merged)
        total_runs += len(combos)

    print(f"\nExperiment: {exp_name}")
    if _WORKER_ID:
        print(f"Worker ID:  {_WORKER_ID}")
    if args.gpu_id is not None:
        print(f"GPU ID:     {args.gpu_id}")
    print(f"Config blocks: {len(config_blocks)}")
    print(f"Total runs: {total_runs}")
    print(f"Logs dir:   {logs_dir}")
    print(f"Models dir: {models_dir}")
    print(f"Cache root: {cache_root}")
    if not save_model:
        print("Save model: DISABLED (no --output-dir passed to finetune.py)")
    if args.dry_run:
        print("DRY RUN — commands will be printed but not executed\n")
    elif args.validate:
        print(
            "VALIDATE — running finetune.py --dry-run for each config "
            "(model loading tested, no training)\n"
        )

    run_counter = 0
    for block in config_blocks:
        config_name = block.get("name", f"config_{config_blocks.index(block)}")
        block_params = {k: v for k, v in block.items() if k != "name"}
        merged = deep_merge(defaults, block_params)
        combos = expand_grid(merged)

        print(f"\n[Config: {config_name}]  {len(combos)} run(s)")

        for i, combo in enumerate(combos):
            run_counter += 1
            run_id = make_run_id(exp_name, config_name, i)
            output_dir = str(models_dir / run_id)

            print(f"\n  [{run_counter}/{total_runs}] {run_id}")

            # Resumability check (skip in validate mode too — always re-validate)
            if not args.no_skip_completed and not args.dry_run and not args.validate:
                if is_run_complete(output_dir):
                    print(f"  SKIP — already complete ({output_dir})")
                    skipped += 1
                    continue
                if is_run_claimed(output_dir):
                    print(f"  SKIP — claimed by another process ({output_dir})")
                    skipped += 1
                    continue

            # Inject shared cache dirs if applicable
            if _shared_attn_cache_dir and "attn_cache_dir" not in combo:
                combo = {**combo, "attn_cache_dir": _shared_attn_cache_dir}
            if _shared_ae_cache_dir and "ae_cache_dir" not in combo:
                combo = {**combo, "ae_cache_dir": _shared_ae_cache_dir}

            cmd = build_cmd(
                python,
                script,
                combo,
                output_dir,
                wandb_cfg,
                finetune_dry_run=args.validate,
                save_model=save_model,
            )

            if args.dry_run:
                import shlex

                print(f"  CMD: {shlex.join(cmd)}")
                continue

            # Claim this run before starting (skip if another process beat us)
            if not args.validate and not claim_run(output_dir):
                print(f"  SKIP — claimed by another process ({output_dir})")
                skipped += 1
                continue

            print(f"  CMD: {' '.join(cmd)}")

            try:
                returncode, stdout, stderr = run_single(cmd, subprocess_env)

                run_record: dict = {
                    "run_id": run_id,
                    "config_name": config_name,
                    "combo_index": i,
                    "params": combo,
                    "output_dir": output_dir,
                    "returncode": returncode,
                }

                if returncode == 0:
                    if args.validate:
                        # Dry-run subprocesses don't emit a Results JSON — just report pass.
                        # Do NOT write a sentinel: validate runs should not block real training.
                        print(f"  PASS — config loaded successfully")
                        run_record["validate"] = "pass"
                        append_csv_row(
                            csv_path,
                            run_id,
                            config_name,
                            i,
                            combo,
                            results=None,
                            status="validate_pass",
                        )
                    else:
                        results = parse_results(stdout)
                        run_record["results"] = results
                        if results is None:
                            run_record["parse_warning"] = (
                                "Could not find Results JSON in stdout"
                            )
                            print(
                                f"  WARNING: Could not parse Results JSON from stdout"
                            )
                            append_csv_row(
                                csv_path,
                                run_id,
                                config_name,
                                i,
                                combo,
                                results=None,
                                status="ok_no_results",
                            )
                        else:
                            final_ppl = results.get("final_ppl", "N/A")
                            baseline_ppl = results.get("baseline_ppl", "N/A")
                            original_ppl = results.get("original_ppl", "N/A")
                            attn_pretrain_ppl = results.get("attn_pretrain_ppl", "N/A")
                            ae_pretrain_ppl = results.get("ae_pretrain_ppl", "N/A")
                            print(
                                f"  original_ppl={original_ppl}  "
                                f"baseline_ppl={baseline_ppl}  "
                                f"attn_pretrain_ppl={attn_pretrain_ppl}  "
                                f"ae_pretrain_ppl={ae_pretrain_ppl}  "
                                f"final_ppl={final_ppl}"
                            )
                            append_csv_row(
                                csv_path,
                                run_id,
                                config_name,
                                i,
                                combo,
                                results=results,
                                status="ok",
                            )
                        write_sentinel(output_dir, run_record)
                else:
                    run_record["stderr_tail"] = stderr[-2000:]
                    print(f"  FAILED (returncode={returncode})")
                    print(f"  stderr tail:\n{stderr[-800:]}")
                    append_csv_row(
                        csv_path,
                        run_id,
                        config_name,
                        i,
                        combo,
                        results=None,
                        status="failed",
                    )

                all_results.append(run_record)
                save_summary(all_results, summary_path)
            finally:
                # Always release the claim — on success, failure, or interrupt
                if not args.validate:
                    release_claim(output_dir)

    # Clean up shared activation caches after all runs
    if _shared_attn_cache_dir and os.path.isdir(_shared_attn_cache_dir):
        import shutil

        shutil.rmtree(_shared_attn_cache_dir, ignore_errors=True)
        print(
            f"\nCleaned up shared attention activation cache: {_shared_attn_cache_dir}"
        )

    if _shared_ae_cache_dir and os.path.isdir(_shared_ae_cache_dir):
        import shutil

        shutil.rmtree(_shared_ae_cache_dir, ignore_errors=True)
        print(f"\nCleaned up shared AE activation cache: {_shared_ae_cache_dir}")

    if not args.dry_run:
        print(f"\n{'=' * 60}")
        if args.validate:
            n_pass = sum(1 for r in all_results if r.get("validate") == "pass")
            n_fail = sum(1 for r in all_results if r.get("returncode", 1) != 0)
            print(f"Validation: {n_pass} passed, {n_fail} failed ({run_counter} total)")
        else:
            print(f"Completed {run_counter - skipped} run(s), skipped {skipped}")
        print(f"Logs dir: {logs_dir}")
        print(f"Summary:  {summary_path}")
        print(f"CSV log:  {csv_path}")

        # Print a quick results table (only for full training runs)
        successful = (
            []
            if args.validate
            else [
                r
                for r in all_results
                if r.get("returncode") == 0 and r.get("results") is not None
            ]
        )
        if successful:
            print(
                f"\n{'Run ID':<50} {'original_ppl':>12} {'baseline_ppl':>12} "
                f"{'attn_pre':>10} {'ae_pre':>10} {'final_ppl':>10}"
            )
            print("-" * 112)
            for r in successful:
                res = r["results"]
                print(
                    f"  {r['run_id']:<48} "
                    f"{_fmt(res.get('original_ppl')):>12} "
                    f"{_fmt(res.get('baseline_ppl')):>12} "
                    f"{_fmt(res.get('attn_pretrain_ppl')):>10} "
                    f"{_fmt(res.get('ae_pretrain_ppl')):>10} "
                    f"{_fmt(res.get('final_ppl')):>10}"
                )


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    main()
