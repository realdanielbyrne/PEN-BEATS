"""
run_from_yaml.py — YAML-driven Unified Experiment Launcher for N-BEATS Lightning

Standardizes the experimental workflow across all study types by accepting
all experiment parameters through a YAML configuration file.

Supports:
  - Single or multi-config benchmark runs
  - Homogeneous / mixed / alternating / custom stack architectures (no code changes)
  - Two-pass design (baseline + activeG_fcast) via 'passes' key
  - Successive halving hyperparameter search
  - Resumable runs (skips already-completed result rows)
  - Extended CSV schemas via 'extra_csv_columns' and per-config 'extra_fields'
  - All datasets: M4, Tourism, Traffic, Weather, Milk
  - W&B, TensorBoard, and CSV logging
  - Optional post-experiment ranking analysis

Stack specification formats (in config 'stacks' or top-level 'stacks'):
  - Explicit list:    [Generic, Generic, ..., Generic]
  - Homogeneous:      {type: homogeneous, block: Generic, n: 30}
  - Prefix + body:    {type: prefix_body, prefix: [Trend, Seasonality],
                        body: GenericAE, total: 30}
  - Alternating:      {type: alternating, blocks: [TrendAE, GenericAE], repeats: 5}
  - Concat of parts:  {type: concat, parts: [...]}
  - Builtin ref:      {builtin: NBEATS-I+G}
  - String shorthand: "Generic:30" or "Generic*30"

Usage:
    python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml
    python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml \\
        --dataset m4 --periods Yearly --n-runs 3
    python experiments/run_from_yaml.py config.yaml --max-epochs 50 --wandb
    python experiments/run_from_yaml.py config.yaml --dry-run
    python experiments/run_from_yaml.py config.yaml --analyze-only
"""

import argparse
import copy
import csv
import gc
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from project root or experiments/
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXPERIMENTS_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", "src"))

# Import shared infrastructure from the unified benchmark
from run_unified_benchmark import (
    run_single_experiment,
    result_exists,
    get_batch_size,
    init_csv,
    load_dataset,
    CSV_COLUMNS,
    N_RUNS_DEFAULT,
    MAX_EPOCHS,
    THETAS_DIM,
    LOSS,
    LEARNING_RATE,
    LATENT_DIM,
    BASE_SEED,
    BASIS_DIM,
    FORECAST_MULTIPLIERS,
    DATASET_PERIODS,
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    UNIFIED_CONFIGS,
    _shutdown_requested,
)

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Default Parameter Dictionaries
# ---------------------------------------------------------------------------

DEFAULT_TRAINING = {
    "active_g": False,
    "sum_losses": False,
    "activation": "ReLU",
    "loss": LOSS,
    "optimizer": "Adam",          # Adam | SGD | RMSprop | Adagrad | Adadelta | AdamW
    "learning_rate": LEARNING_RATE,
    "max_epochs": MAX_EPOCHS,
    "patience": EARLY_STOPPING_PATIENCE,
    "n_blocks_per_stack": 1,
    "share_weights": True,
    "batch_size": None,          # None = auto-resolve per dataset/period
    "forecast_multiplier": None, # None = auto-resolve per dataset
}

DEFAULT_BLOCK_PARAMS = {
    "thetas_dim": THETAS_DIM,
    "latent_dim": LATENT_DIM,
    "basis_dim": BASIS_DIM,
    "forecast_basis_dim": None,
    "trend_thetas_dim": None,
    "wavelet_type": None,
}

DEFAULT_RUNS = {
    "n_runs": N_RUNS_DEFAULT,
    "base_seed": BASE_SEED,
    "seed_mode": "sequential",  # sequential | random | fixed
    "seed": None,               # fixed seed used when seed_mode="fixed"
    "seeds": None,              # explicit per-run list; overrides seed_mode + base_seed
}

DEFAULT_OUTPUT = {
    "results_dir": os.path.join(_EXPERIMENTS_DIR, "results"),
    "csv_filename": None,            # None → "{experiment_name}_results.csv"
    "save_predictions": True,
    "predictions_subdir": "predictions",
}

DEFAULT_LOGGING = {
    "wandb": {
        "enabled": False,
        "project": "nbeats-lightning",
        "group": None,
    },
    "tensorboard": False,
    "csv_log": True,
}

DEFAULT_HARDWARE = {
    "accelerator": "auto",
    "num_workers": 0,
    "gpu_id": None,
}

# ---------------------------------------------------------------------------
# Utility: Deep Merge
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Override values take precedence.  Sub-dicts are merged recursively;
    all other types (lists, scalars) are replaced wholesale.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if (key in result
                and isinstance(result[key], dict)
                and isinstance(val, dict)):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# YAML Loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    """Load and return a YAML configuration file as a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping at top level: {path}")
    return cfg


# ---------------------------------------------------------------------------
# Stack Type Parsing
# ---------------------------------------------------------------------------

def parse_stack_spec(spec) -> list:
    """Parse a stack specification into a flat list of block-type strings.

    Supported forms
    ---------------
    list
        ``["Generic", "Generic", ...]`` — used verbatim.
    str
        ``"Generic:30"`` or ``"Generic*30"`` — homogeneous shorthand.
        A bare string with no separator is treated as a single block.
    dict  (type key selects the pattern)
        ``{type: homogeneous, block: Generic, n: 30}``
        ``{type: prefix_body, prefix: [Trend, Seasonality],
            body: GenericAE, total: 30}``
        ``{type: alternating, blocks: [TrendAE, GenericAE], repeats: 5}``
        ``{type: concat, parts: [...]}``
        ``{type: explicit, blocks: [...]}``
        ``{builtin: NBEATS-I+G}`` — reference a UNIFIED_CONFIGS entry.
    """
    if spec is None:
        raise ValueError("Stack specification is required (got None).")

    # ----- Direct list -----
    if isinstance(spec, list):
        return [str(s) for s in spec]

    # ----- String shorthand -----
    if isinstance(spec, str):
        for sep in (":", "*", "x"):
            if sep in spec:
                parts = spec.split(sep, 1)
                block = parts[0].strip()
                n = int(parts[1].strip())
                return [block] * n
        # Single block name
        return [spec.strip()]

    if not isinstance(spec, dict):
        raise ValueError(
            f"Stack spec must be a list, string, or dict; got {type(spec).__name__}."
        )

    # ----- Builtin reference -----
    if "builtin" in spec:
        name = spec["builtin"]
        if name not in UNIFIED_CONFIGS:
            raise ValueError(
                f"Unknown builtin config: {name!r}. "
                f"Valid keys: {list(UNIFIED_CONFIGS.keys())}"
            )
        return list(UNIFIED_CONFIGS[name]["stack_types"])

    spec_type = spec.get("type", "homogeneous")

    if spec_type == "homogeneous":
        block = spec.get("block", spec.get("blocks"))
        if isinstance(block, list):
            block = block[0]
        n = int(spec.get("n", spec.get("n_stacks", 30)))
        if block is None:
            raise ValueError("homogeneous stack spec requires 'block' key.")
        return [block] * n

    elif spec_type == "prefix_body":
        prefix = list(spec.get("prefix", []))
        body_block = spec.get("body", spec.get("remainder"))
        total = int(spec.get("total", 30))
        if body_block is None:
            raise ValueError("prefix_body requires 'body' (or 'remainder') key.")
        n_body = total - len(prefix)
        if n_body < 0:
            raise ValueError(
                f"prefix_body: prefix length ({len(prefix)}) exceeds total ({total})."
            )
        return prefix + [body_block] * n_body

    elif spec_type == "alternating":
        blocks = list(spec["blocks"])
        repeats = int(spec.get("repeats", 1))
        return blocks * repeats

    elif spec_type == "concat":
        result = []
        for part in spec.get("parts", []):
            result.extend(parse_stack_spec(part))
        return result

    elif spec_type == "explicit":
        return [str(b) for b in spec.get("blocks", spec.get("stack_types", []))]

    else:
        raise ValueError(f"Unknown stack spec type: {spec_type!r}.")


# ---------------------------------------------------------------------------
# Config Building
# ---------------------------------------------------------------------------

def _extract_stack_spec(config_spec: dict, top_level_cfg: dict):
    """Extract the stack specification from config_spec or top-level config."""
    # Config-level 'stacks' key takes priority, then 'stack_types'
    spec = config_spec.get("stacks", config_spec.get("stack_types"))
    if spec is None:
        spec = top_level_cfg.get("stacks", top_level_cfg.get("stack_types"))
    return spec


def resolve_config(config_spec: dict, top_level_cfg: dict) -> dict:
    """Resolve a single config dict from config_spec + top-level defaults.

    Parameters
    ----------
    config_spec : dict
        One entry from the ``configs`` list in the YAML file.
    top_level_cfg : dict
        The top-level YAML mapping (provides global defaults).

    Returns
    -------
    dict
        Resolved config with keys: ``name``, ``category``, ``stack_types``,
        ``n_blocks_per_stack``, ``share_weights``, ``training``,
        ``block_params``, ``lr_scheduler``, ``extra_fields``.
    """
    # ── Builtin shortcut ──────────────────────────────────────────────────
    if "builtin" in config_spec:
        builtin_name = config_spec["builtin"]
        if builtin_name not in UNIFIED_CONFIGS:
            raise ValueError(
                f"Unknown builtin config: {builtin_name!r}. "
                f"Valid: {list(UNIFIED_CONFIGS.keys())}"
            )
        built = UNIFIED_CONFIGS[builtin_name]
        base = {
            "name": config_spec.get("name", builtin_name),
            "category": config_spec.get(
                "category", built.get("category", "builtin")
            ),
            "stack_types": list(built["stack_types"]),
            "n_blocks_per_stack": config_spec.get(
                "n_blocks_per_stack", built.get("n_blocks_per_stack", 1)
            ),
            "share_weights": config_spec.get(
                "share_weights", built.get("share_weights", True)
            ),
        }
    else:
        # ── Custom config ─────────────────────────────────────────────────
        stack_spec = _extract_stack_spec(config_spec, top_level_cfg)
        if stack_spec is None:
            raise ValueError(
                f"Config {config_spec.get('name', '?')!r} has no stack "
                "specification (add 'stacks' or 'stack_types')."
            )
        stack_types = parse_stack_spec(stack_spec)

        top_tr = deep_merge(DEFAULT_TRAINING, top_level_cfg.get("training", {}))
        base = {
            "name": config_spec.get("name", "unnamed_config"),
            "category": config_spec.get(
                "category", top_level_cfg.get("category", "custom")
            ),
            "stack_types": stack_types,
            "n_blocks_per_stack": config_spec.get(
                "n_blocks_per_stack",
                top_tr.get("n_blocks_per_stack", 1),
            ),
            "share_weights": config_spec.get(
                "share_weights",
                top_tr.get("share_weights", True),
            ),
        }

    # ── Merge training params: global defaults → top-level → config-level ─
    top_training = deep_merge(DEFAULT_TRAINING, top_level_cfg.get("training", {}))
    cfg_training = deep_merge(top_training, config_spec.get("training", {}))
    # Propagate n_blocks_per_stack and share_weights into training for convenience
    cfg_training["n_blocks_per_stack"] = base["n_blocks_per_stack"]
    cfg_training["share_weights"] = base["share_weights"]
    base["training"] = cfg_training

    # ── Merge block params ────────────────────────────────────────────────
    top_block = deep_merge(
        DEFAULT_BLOCK_PARAMS, top_level_cfg.get("block_params", {})
    )
    cfg_block = deep_merge(top_block, config_spec.get("block_params", {}))
    base["block_params"] = cfg_block

    # ── LR scheduler ─────────────────────────────────────────────────────
    top_sched = top_level_cfg.get("lr_scheduler")
    cfg_sched = config_spec.get("lr_scheduler", top_sched)
    base["lr_scheduler"] = cfg_sched

    # ── Extra fields for CSV ──────────────────────────────────────────────
    top_extra = top_level_cfg.get("extra_fields", {}) or {}
    cfg_extra = config_spec.get("extra_fields", {}) or {}
    base["extra_fields"] = {**top_extra, **cfg_extra}

    return base


def build_configs(top_level_cfg: dict) -> list:
    """Build the list of resolved config dicts from a top-level YAML config.

    When a ``configs`` list is present, each entry is resolved via
    :func:`resolve_config`.  Otherwise, a single config is built from
    the top-level ``stacks`` / ``stack_types`` spec.
    """
    configs_spec = top_level_cfg.get("configs")
    if configs_spec is not None:
        if not isinstance(configs_spec, list):
            raise ValueError("'configs' must be a YAML list.")
        return [resolve_config(cs, top_level_cfg) for cs in configs_spec]

    # Single-config mode
    stacks_spec = top_level_cfg.get("stacks", top_level_cfg.get("stack_types"))
    if stacks_spec is None:
        raise ValueError(
            "YAML must have either a 'configs' list or a top-level "
            "'stacks' / 'stack_types' specification."
        )
    single_spec: dict = {
        "name": top_level_cfg.get(
            "config_name", top_level_cfg.get("experiment_name", "config")
        ),
        "category": top_level_cfg.get("category", "custom"),
        "stacks": stacks_spec,
    }
    # Propagate any config-level overrides from top-level
    for key in (
        "n_blocks_per_stack", "share_weights",
        "training", "block_params", "lr_scheduler", "extra_fields",
    ):
        if key in top_level_cfg:
            single_spec[key] = top_level_cfg[key]
    return [resolve_config(single_spec, top_level_cfg)]


# ---------------------------------------------------------------------------
# Pass Building
# ---------------------------------------------------------------------------

def build_passes(top_level_cfg: dict) -> list:
    """Build the list of pass dicts from a YAML config.

    Each pass has keys ``name`` (str) and ``training`` (dict of overrides).

    When a ``passes`` list is present in the YAML, it is used directly::

        passes:
          - name: baseline
            training: {active_g: false}
          - name: activeG_fcast
            training: {active_g: forecast}

    Otherwise a single pass is returned whose ``name`` is the
    ``experiment_name`` value and whose ``training`` dict is empty
    (inheriting all values from the config-level training dict).
    """
    passes_spec = top_level_cfg.get("passes")
    if passes_spec:
        return [
            {
                "name": str(p.get("name", f"pass{i}")),
                "training": p.get("training", {}) or {},
            }
            for i, p in enumerate(passes_spec)
        ]
    return [
        {
            "name": top_level_cfg.get("experiment_name", "run"),
            "training": {},
        }
    ]


# ---------------------------------------------------------------------------
# CSV Column Building
# ---------------------------------------------------------------------------

def build_csv_columns(top_level_cfg: dict) -> list:
    """Return the CSV column list: base columns + any extras from YAML."""
    extras = top_level_cfg.get("extra_csv_columns") or []
    return CSV_COLUMNS + [c for c in extras if c not in CSV_COLUMNS]


# ---------------------------------------------------------------------------
# LR Scheduler Resolution
# ---------------------------------------------------------------------------

def resolve_lr_scheduler(lr_scheduler_cfg, max_epochs: int):
    """Convert YAML lr_scheduler mapping to a run_single_experiment-compatible dict.

    Returns ``None`` if the scheduler is disabled or max_epochs is too
    small for the warmup period.

    Expected YAML structure::

        lr_scheduler:
          warmup_epochs: 15
          T_max: null      # null → max_epochs - warmup_epochs
          eta_min: 0.000001
    """
    if not lr_scheduler_cfg:
        return None
    warmup = int(lr_scheduler_cfg.get("warmup_epochs", 15))
    if max_epochs <= warmup:
        return None
    t_max_raw = lr_scheduler_cfg.get("T_max")
    t_max = int(t_max_raw) if t_max_raw is not None else max(max_epochs - warmup, 1)
    eta_min = float(lr_scheduler_cfg.get("eta_min", 1e-6))
    return {"warmup_epochs": warmup, "T_max": t_max, "eta_min": eta_min}


# ---------------------------------------------------------------------------
# Dataset Key Helper
# ---------------------------------------------------------------------------

def _dataset_key_from_dataset(dataset) -> str:
    """Derive the FORECAST_MULTIPLIERS / BATCH_SIZES key from a dataset object."""
    name = dataset.name.lower()
    for key in ("m4", "tourism", "milk", "traffic", "weather"):
        if key in name:
            return key
    return "m4"   # safe fallback


def _resolve_batch_size(training: dict, dataset, period: str) -> int:
    """Resolve batch size: tuned-table value > training override > default."""
    override = training.get("batch_size")
    dataset_key = _dataset_key_from_dataset(dataset)
    return get_batch_size(dataset_key, period, override)


def _resolve_forecast_multiplier(training: dict, dataset) -> int:
    """Resolve forecast multiplier from training dict or per-dataset default."""
    fm = training.get("forecast_multiplier")
    if fm is not None:
        return int(fm)
    return FORECAST_MULTIPLIERS.get(_dataset_key_from_dataset(dataset), 5)


def _compute_seed(runs_cfg: dict, run_idx: int) -> int:
    """Compute the seed for a given run index.

    Behaviour is controlled by ``runs.seed_mode`` in the YAML::

        seed_mode: sequential  # seed = base_seed + run_idx  (default)
        seed_mode: random      # new random seed per run
        seed_mode: fixed       # every run uses the same seed (runs.seed)

    An explicit ``runs.seeds`` list always takes precedence; if the list is
    shorter than the total run count it wraps (cycles).
    """
    seeds_list = runs_cfg.get("seeds")
    if seeds_list is not None:
        return int(seeds_list[run_idx % len(seeds_list)])

    seed_mode = runs_cfg.get("seed_mode", "sequential")
    if seed_mode == "random":
        return int(np.random.randint(0, 2 ** 31))
    elif seed_mode == "fixed":
        fixed = runs_cfg.get("seed")
        return int(fixed) if fixed is not None else BASE_SEED
    else:  # sequential (default)
        base = int(runs_cfg.get("base_seed", BASE_SEED))
        return base + run_idx


# ---------------------------------------------------------------------------
# Single Run Wrapper
# ---------------------------------------------------------------------------

def run_single_config(
    config: dict,
    pass_name: str,
    pass_training_override: dict,
    period: str,
    run_idx: int,
    dataset,
    train_series_list,
    csv_path: str,
    output_cfg: dict,
    hardware_cfg: dict,
    logging_cfg: dict,
    csv_columns: list,
    round_num: int = None,
    seed: int = None,
    dry_run: bool = False,
):
    """Execute one experiment (config × pass × period × run_idx).

    Thin wrapper that translates the resolved YAML config into the flat
    keyword arguments expected by :func:`run_single_experiment`.

    Parameters
    ----------
    config : dict
        Resolved config dict (from :func:`resolve_config`).
    pass_name : str
        Pass label used as the ``experiment_name`` in CSV rows.
    pass_training_override : dict
        Training param overrides specific to this pass (e.g. active_g).
    period : str
        Dataset period string (e.g. "Yearly", "Traffic-96").
    run_idx : int
        Zero-based run index.
    seed : int or None
        Explicit RNG seed.  ``None`` defers to the ``BASE_SEED + run_idx``
        formula inside :func:`run_single_experiment`.
    dataset
        Loaded benchmark dataset object.
    train_series_list : list
        List of training series (from dataset.get_training_series()).
    csv_path : str
        Path to the output CSV file.
    output_cfg, hardware_cfg, logging_cfg : dict
        Resolved output / hardware / logging settings.
    csv_columns : list
        Column list for the output CSV.
    round_num : int or None
        If in a successive halving search, the current round number.
        Appended to the experiment tag and to the ``search_round`` extra field.
    dry_run : bool
        If True, print what would run without actually running it.
    """
    config_name = config["name"]

    # Merge training: config-level ← pass-level override
    training = deep_merge(config["training"], pass_training_override)
    block_params = config["block_params"]

    # Build the experiment tag (= "experiment" column in CSV)
    exp_tag = pass_name
    if round_num is not None:
        exp_tag = f"{pass_name}_r{round_num}"

    if dry_run:
        print(
            f"  [DRY-RUN] {exp_tag} / {config_name} / {period} / run {run_idx}  "
            f"stacks={len(config['stack_types'])}×{config['stack_types'][0]}"
            f"{'...' if len(set(config['stack_types'])) > 1 else ''}"
        )
        return

    max_epochs = int(training.get("max_epochs", MAX_EPOCHS))
    lr_scheduler_cfg = resolve_lr_scheduler(config.get("lr_scheduler"), max_epochs)
    forecast_multiplier = _resolve_forecast_multiplier(training, dataset)

    # Extra CSV fields
    extra_row = dict(config.get("extra_fields") or {})
    if round_num is not None:
        extra_row.setdefault("search_round", round_num)

    # Predictions directory
    predictions_dir = None
    if output_cfg.get("save_predictions", True):
        predictions_dir = os.path.join(
            output_cfg["results_dir"],
            output_cfg.get("predictions_subdir", "predictions"),
        )

    wandb_cfg = logging_cfg.get("wandb") or {}

    run_single_experiment(
        experiment_name=exp_tag,
        config_name=config_name,
        category=config["category"],
        stack_types=config["stack_types"],
        period=period,
        run_idx=run_idx,
        dataset=dataset,
        train_series_list=train_series_list,
        csv_path=csv_path,
        n_blocks_per_stack=config["n_blocks_per_stack"],
        share_weights=config["share_weights"],
        active_g=training.get("active_g", False),
        sum_losses=training.get("sum_losses", False),
        activation=training.get("activation", "ReLU"),
        max_epochs=max_epochs,
        patience=int(training.get("patience", EARLY_STOPPING_PATIENCE)),
        batch_size=_resolve_batch_size(training, dataset, period),
        accelerator_override=hardware_cfg.get("accelerator", "auto"),
        forecast_multiplier=forecast_multiplier,
        num_workers=int(hardware_cfg.get("num_workers", 0)),
        wandb_enabled=bool(wandb_cfg.get("enabled", False)),
        wandb_project=str(wandb_cfg.get("project", "nbeats-lightning")),
        save_predictions=bool(output_cfg.get("save_predictions", True)),
        predictions_dir=predictions_dir,
        gpu_id=hardware_cfg.get("gpu_id"),
        basis_dim=int(block_params.get("basis_dim", BASIS_DIM)),
        forecast_basis_dim=block_params.get("forecast_basis_dim"),
        extra_row=extra_row if extra_row else None,
        csv_columns=csv_columns,
        tb_enabled=bool(logging_cfg.get("tensorboard", False)),
        thetas_dim_override=block_params.get("thetas_dim"),
        latent_dim_override=block_params.get("latent_dim"),
        trend_thetas_dim=block_params.get("trend_thetas_dim"),
        wavelet_type=block_params.get("wavelet_type", "db3"),
        lr_scheduler_config=lr_scheduler_cfg,
        optimizer_name=training.get("optimizer", "Adam"),
        learning_rate=training.get("learning_rate"),
        loss_override=training.get("loss"),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Standard Run Loop
# ---------------------------------------------------------------------------

def _run_standard(
    configs: list,
    passes: list,
    n_runs: int,
    period: str,
    dataset,
    train_series_list,
    csv_path: str,
    output_cfg: dict,
    hardware_cfg: dict,
    logging_cfg: dict,
    csv_columns: list,
    runs_cfg: dict = None,
    dry_run: bool = False,
):
    """Execute all (config × pass × run_idx) combinations for one period."""
    runs_cfg = runs_cfg or {}
    total = len(configs) * len(passes) * n_runs
    done = 0

    for pass_info in passes:
        pass_name = pass_info["name"]
        pass_training = pass_info.get("training") or {}

        print(f"\n    Pass: {pass_name}")

        for config in configs:
            config_name = config["name"]
            for run_idx in range(n_runs):
                if _shutdown_requested:
                    print("[SHUTDOWN] Exiting run loop.")
                    return

                done += 1
                exp_tag = pass_name
                if (not dry_run
                        and result_exists(csv_path, exp_tag, config_name,
                                          period, run_idx)):
                    print(
                        f"    [{done}/{total}] [SKIP] "
                        f"{config_name} / {period} / run {run_idx}"
                    )
                    continue

                seed = _compute_seed(runs_cfg, run_idx)
                print(f"\n    [{done}/{total}] "
                      f"{config_name} / {period} / run {run_idx}"
                      f"  (seed={seed})")

                run_single_config(
                    config=config,
                    pass_name=pass_name,
                    pass_training_override=pass_training,
                    period=period,
                    run_idx=run_idx,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    output_cfg=output_cfg,
                    hardware_cfg=hardware_cfg,
                    logging_cfg=logging_cfg,
                    csv_columns=csv_columns,
                    seed=seed,
                    dry_run=dry_run,
                )


# ---------------------------------------------------------------------------
# Successive Halving
# ---------------------------------------------------------------------------

def _load_round_results(csv_path: str, round_num: int) -> list:
    """Return all CSV rows that belong to search round *round_num*."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Match on the search_round column (if present) OR on "_r{n}" in
            # the experiment tag — whichever was written by run_single_config.
            search_round_col = row.get("search_round", "")
            exp_col = row.get("experiment", "")
            if search_round_col == str(round_num) or f"_r{round_num}" in exp_col:
                rows.append(row)
    return rows


def rank_configs_for_promotion(
    csv_path: str,
    round_num: int,
    keep_fraction: float = 0.5,
    top_k: int = None,
) -> list:
    """Rank configs from a search round and return promoted config-name strings.

    Ranking metric: median best_val_loss across runs.
    Configs with >50 % divergence rate are sent to the bottom.

    Returns
    -------
    list[str]
        Config names to carry forward to the next round.
    """
    rows = _load_round_results(csv_path, round_num)
    if not rows:
        print(f"  [WARN] No results for round {round_num} in {csv_path}.")
        return []

    config_results: dict = defaultdict(list)
    for row in rows:
        config_results[row["config_name"]].append(row)

    rankings = []
    for name, result_rows in config_results.items():
        val_losses = []
        n_diverged = 0
        for r in result_rows:
            try:
                bvl = float(r.get("best_val_loss", "inf"))
                if math.isfinite(bvl):
                    val_losses.append(bvl)
            except (ValueError, TypeError):
                pass
            if r.get("diverged", "").lower() in ("true", "1"):
                n_diverged += 1

        median_bvl = float(np.median(val_losses)) if val_losses else float("inf")
        div_rate = n_diverged / len(result_rows) if result_rows else 0.0
        rankings.append(
            {
                "config_name": name,
                "median_best_val_loss": median_bvl,
                "n_runs": len(val_losses),
                "divergence_rate": div_rate,
            }
        )

    rankings.sort(
        key=lambda r: (r["divergence_rate"] > 0.5, r["median_best_val_loss"])
    )
    total = len(rankings)
    if top_k is not None:
        keep_n = min(int(top_k), total)
    else:
        keep_n = max(1, int(total * keep_fraction))

    promoted = {r["config_name"] for r in rankings[:keep_n]}

    # Print ranking table
    print(f"\n  {'='*65}")
    print(
        f"  Round {round_num} Ranking — {total} configs, "
        f"promoting top {keep_n}"
    )
    print(f"  {'='*65}")
    print(
        f"  {'Rank':<5} {'Config':<42} {'ValLoss':>9} "
        f"{'Div%':>5} {'Runs':>4}"
    )
    print(f"  {'-'*65}")
    for i, r in enumerate(rankings[:min(40, total)]):
        marker = " *" if r["config_name"] in promoted else "  "
        div_str = f"{r['divergence_rate'] * 100:.0f}%"
        print(
            f"  {i + 1:<5}{marker} {r['config_name']:<40} "
            f"{r['median_best_val_loss']:>9.4f} {div_str:>5} "
            f"{r['n_runs']:>4}"
        )

    return list(promoted)


def _run_successive_halving(
    configs: list,
    passes: list,
    search_rounds: list,
    period: str,
    dataset,
    train_series_list,
    csv_path: str,
    output_cfg: dict,
    hardware_cfg: dict,
    logging_cfg: dict,
    csv_columns: list,
    runs_cfg: dict = None,
    dry_run: bool = False,
):
    """Run successive halving over the config list.

    Each round runs all (active_config × pass × n_runs_round) combinations
    then eliminates the bottom configs.  The final round skips elimination.

    Parameters
    ----------
    search_rounds : list[dict]
        Each dict may contain:
        ``max_epochs`` (int), ``n_runs`` (int),
        ``keep_fraction`` (float, default 0.5),
        ``top_k`` (int, optional — overrides keep_fraction for the last round).
    """
    runs_cfg = runs_cfg or {}
    # Build lookup: config_name → config dict
    active_configs = {c["name"]: c for c in configs}

    for round_idx, round_spec in enumerate(search_rounds):
        round_num = round_idx + 1

        if _shutdown_requested:
            print("[SHUTDOWN] Exiting successive halving.")
            return

        max_epochs_override = round_spec.get("max_epochs")
        n_runs_round = int(round_spec.get("n_runs", 3))
        keep_fraction = float(round_spec.get("keep_fraction", 0.5))
        top_k = round_spec.get("top_k")
        is_last_round = (round_idx == len(search_rounds) - 1)

        print(
            f"\n  -- Search Round {round_num}/{len(search_rounds)} --  "
            f"configs={len(active_configs)}  "
            f"n_runs={n_runs_round}  "
            f"max_epochs={max_epochs_override}"
        )

        for pass_info in passes:
            pass_name = pass_info["name"]
            pass_training = dict(pass_info.get("training") or {})
            if max_epochs_override is not None:
                pass_training["max_epochs"] = int(max_epochs_override)

            for config_name, config in active_configs.items():
                for run_idx in range(n_runs_round):
                    if _shutdown_requested:
                        print("[SHUTDOWN] Exiting search round.")
                        return

                    exp_tag = f"{pass_name}_r{round_num}"
                    if (not dry_run
                            and result_exists(
                                csv_path, exp_tag, config_name,
                                period, run_idx,
                            )):
                        print(
                            f"    [SKIP] r{round_num} / {config_name} "
                            f"/ {period} / run {run_idx}"
                        )
                        continue

                    run_single_config(
                        config=config,
                        pass_name=pass_name,
                        pass_training_override=pass_training,
                        period=period,
                        run_idx=run_idx,
                        dataset=dataset,
                        train_series_list=train_series_list,
                        csv_path=csv_path,
                        output_cfg=output_cfg,
                        hardware_cfg=hardware_cfg,
                        logging_cfg=logging_cfg,
                        csv_columns=csv_columns,
                        round_num=round_num,
                        seed=_compute_seed(runs_cfg, run_idx),
                        dry_run=dry_run,
                    )

        # Promote for next round (skip on last round)
        if not is_last_round and not dry_run:
            promoted = rank_configs_for_promotion(
                csv_path=csv_path,
                round_num=round_num,
                keep_fraction=keep_fraction,
                top_k=top_k,
            )
            if not promoted:
                print(
                    f"  [WARN] No configs promoted from round {round_num}; "
                    "keeping all."
                )
            else:
                active_configs = {
                    k: v for k, v in active_configs.items() if k in promoted
                }
                print(
                    f"\n  -> Carrying {len(active_configs)} configs "
                    f"into round {round_num + 1}."
                )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Post-Experiment Analysis
# ---------------------------------------------------------------------------

def _run_analysis(csv_path: str, analysis_cfg: dict, dataset_name: str):
    """Print an optional post-experiment summary after all runs complete."""
    if not os.path.exists(csv_path):
        print(f"  [ANALYSIS] CSV not found: {csv_path}")
        return

    print(f"\n  {'='*60}")
    print(f"  Post-experiment Analysis — {dataset_name.upper()}")
    print(f"  {'='*60}")

    rows = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        print("  No results found.")
        return

    if analysis_cfg.get("ranking", True):
        _print_ranking_table(rows)


def _print_ranking_table(rows: list):
    """Print a mean-OWA ranking table from result rows."""
    config_metrics: dict = defaultdict(list)
    for row in rows:
        try:
            owa = float(row.get("owa", "nan"))
            if math.isfinite(owa):
                config_metrics[row["config_name"]].append(owa)
        except (ValueError, TypeError):
            pass

    if not config_metrics:
        print("  No valid OWA values found in results.")
        return

    ranked = sorted(
        [
            (name, float(np.mean(owas)), float(np.std(owas)), len(owas))
            for name, owas in config_metrics.items()
        ],
        key=lambda x: x[1],
    )

    print(
        f"\n  {'Rank':<5} {'Config':<42} "
        f"{'Mean OWA':>9} {'Std':>8} {'N':>4}"
    )
    print(f"  {'-'*67}")
    for i, (name, mean_owa, std_owa, n) in enumerate(ranked):
        print(
            f"  {i + 1:<5} {name:<42} "
            f"{mean_owa:>9.4f} {std_owa:>8.4f} {n:>4}"
        )


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_experiment(
    top_level_cfg: dict,
    cli_overrides: dict,
    dry_run: bool = False,
    analyze_only: bool = False,
):
    """Top-level orchestrator: parse config, iterate datasets/periods, run.

    Parameters
    ----------
    top_level_cfg : dict
        Parsed YAML config.
    cli_overrides : dict
        Overrides from CLI arguments (merged over top_level_cfg).
    dry_run : bool
        Print execution plan without actually training.
    analyze_only : bool
        Skip all training; print analysis summary for existing results.
    """
    # CLI overrides take precedence
    cfg = deep_merge(top_level_cfg, cli_overrides)

    experiment_name = cfg.get("experiment_name", "yaml_experiment")

    # ── Dataset selection ────────────────────────────────────────────────
    datasets_raw = cfg.get("dataset", "m4")
    if isinstance(datasets_raw, str):
        dataset_names = [datasets_raw]
    else:
        dataset_names = list(datasets_raw)

    # ── Build configs, passes, CSV columns ───────────────────────────────
    configs = build_configs(cfg)
    passes = build_passes(cfg)
    csv_columns = build_csv_columns(cfg)

    # ── Resolve global settings ──────────────────────────────────────────
    output_cfg = deep_merge(DEFAULT_OUTPUT, cfg.get("output") or {})
    logging_cfg = deep_merge(DEFAULT_LOGGING, cfg.get("logging") or {})
    hardware_cfg = deep_merge(DEFAULT_HARDWARE, cfg.get("hardware") or {})
    runs_cfg = deep_merge(DEFAULT_RUNS, cfg.get("runs") or {})
    n_runs = int(runs_cfg.get("n_runs", N_RUNS_DEFAULT))

    # ── Search (successive halving) config ───────────────────────────────
    search_cfg = cfg.get("search") or {}
    use_search = bool(search_cfg.get("enabled", False))
    search_rounds = search_cfg.get("rounds") or []

    # ── Analysis config ──────────────────────────────────────────────────
    analysis_cfg = cfg.get("analysis") or {}
    do_analysis = bool(analysis_cfg.get("enabled", False))

    # ── Header ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"YAML Experiment Launcher")
    print(f"  Experiment:  {experiment_name}")
    print(f"  Configs:     {len(configs)}  "
          f"({', '.join(c['name'] for c in configs[:5])}"
          f"{'...' if len(configs) > 5 else ''})")
    print(f"  Passes:      {[p['name'] for p in passes]}")
    print(f"  Datasets:    {dataset_names}")
    print(f"  Runs/config: {n_runs}")
    if use_search:
        print(f"  Search:      successive halving, {len(search_rounds)} rounds")
    if dry_run:
        print(f"  *** DRY RUN — no training will occur ***")
    print(f"{'=' * 70}\n")

    # ── Per-dataset loop ─────────────────────────────────────────────────
    for dataset_name in dataset_names:
        if _shutdown_requested:
            print("[SHUTDOWN] Exiting before next dataset.")
            return

        # Resolve periods
        all_periods = DATASET_PERIODS.get(dataset_name) or {}
        if not all_periods:
            print(f"[WARN] Unknown dataset '{dataset_name}'; skipping.")
            continue

        requested_periods = cfg.get("periods")
        if requested_periods:
            periods = [p for p in requested_periods if p in all_periods]
            if not periods:
                print(
                    f"[WARN] None of the requested periods "
                    f"{requested_periods} are valid for '{dataset_name}'. "
                    f"Available: {list(all_periods.keys())}"
                )
                continue
        else:
            periods = list(all_periods.keys())

        # Output paths
        results_dir = os.path.join(output_cfg["results_dir"], dataset_name)
        os.makedirs(results_dir, exist_ok=True)

        csv_filename = (
            output_cfg.get("csv_filename")
            or f"{experiment_name}_results.csv"
        )
        csv_path = os.path.join(results_dir, csv_filename)

        if not dry_run and not analyze_only:
            init_csv(csv_path, columns=csv_columns)

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"  Periods : {periods}")
        print(f"  CSV     : {csv_path}")
        print(f"{'=' * 60}")

        # ── Per-period loop ─────────────────────────────────────────────
        for period in periods:
            if _shutdown_requested:
                print("[SHUTDOWN] Exiting before next period.")
                return

            print(f"\n  Period: {period}")

            if analyze_only:
                _run_analysis(csv_path, {**analysis_cfg, "enabled": True},
                              dataset_name)
                continue

            if not dry_run:
                dataset = load_dataset(dataset_name, period)
                train_series_list = dataset.get_training_series()
            else:
                dataset = None
                train_series_list = None

            if use_search:
                _run_successive_halving(
                    configs=configs,
                    passes=passes,
                    search_rounds=search_rounds,
                    period=period,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    output_cfg=output_cfg,
                    hardware_cfg=hardware_cfg,
                    logging_cfg=logging_cfg,
                    csv_columns=csv_columns,
                    runs_cfg=runs_cfg,
                    dry_run=dry_run,
                )
            else:
                _run_standard(
                    configs=configs,
                    passes=passes,
                    n_runs=n_runs,
                    period=period,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    output_cfg=output_cfg,
                    hardware_cfg=hardware_cfg,
                    logging_cfg=logging_cfg,
                    csv_columns=csv_columns,
                    runs_cfg=runs_cfg,
                    dry_run=dry_run,
                )

            if not dry_run:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ── Per-dataset post-analysis ────────────────────────────────────
        if do_analysis and not dry_run and not analyze_only:
            _run_analysis(csv_path, analysis_cfg, dataset_name)

    print(f"\n{'=' * 70}")
    print(f"Experiment complete: {experiment_name}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YAML-driven Unified Experiment Launcher for N-BEATS Lightning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml
  python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml \\
      --dataset m4 --periods Yearly --n-runs 3
  python experiments/run_from_yaml.py experiments/configs/genericae_pure.yaml \\
      --max-epochs 50 --wandb
  python experiments/run_from_yaml.py config.yaml --dry-run
  python experiments/run_from_yaml.py config.yaml --analyze-only
        """,
    )

    parser.add_argument("config", help="Path to YAML configuration file.")
    parser.add_argument(
        "--dataset",
        choices=["m4", "tourism", "milk", "traffic", "weather"],
        default=None,
        help="Override dataset (default: from YAML).",
    )
    parser.add_argument(
        "--periods", nargs="+", default=None,
        help="Override period(s) (default: from YAML).",
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help="Override number of runs per config (default: from YAML).",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override maximum training epochs (default: from YAML).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size (default: auto per dataset/period).",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="Override accelerator (default: from YAML or 'auto').",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override DataLoader workers (default: from YAML).",
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project", default=None,
        help="W&B project name (default: from YAML).",
    )
    parser.add_argument(
        "--no-save-predictions", action="store_true",
        help="Disable NPZ prediction saving.",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override results directory (default: experiments/results).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing any training.",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Skip all training; run analysis on existing results only.",
    )

    args = parser.parse_args()

    # ── Load YAML ────────────────────────────────────────────────────────
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    top_level_cfg = load_yaml_config(config_path)

    # ── Build CLI overrides ──────────────────────────────────────────────
    cli_overrides: dict = {}

    if args.dataset is not None:
        cli_overrides["dataset"] = args.dataset
    if args.periods is not None:
        cli_overrides["periods"] = args.periods
    if args.n_runs is not None:
        cli_overrides.setdefault("runs", {})["n_runs"] = args.n_runs
    if args.max_epochs is not None:
        cli_overrides.setdefault("training", {})["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cli_overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.accelerator is not None:
        cli_overrides.setdefault("hardware", {})["accelerator"] = args.accelerator
    if args.num_workers is not None:
        cli_overrides.setdefault("hardware", {})["num_workers"] = args.num_workers
    if args.wandb:
        cli_overrides.setdefault("logging", {}).setdefault(
            "wandb", {}
        )["enabled"] = True
    if args.wandb_project is not None:
        cli_overrides.setdefault("logging", {}).setdefault(
            "wandb", {}
        )["project"] = args.wandb_project
    if args.no_save_predictions:
        cli_overrides.setdefault("output", {})["save_predictions"] = False
    if args.results_dir is not None:
        cli_overrides.setdefault("output", {})["results_dir"] = args.results_dir

    # ── Run ──────────────────────────────────────────────────────────────
    run_experiment(
        top_level_cfg,
        cli_overrides,
        dry_run=args.dry_run,
        analyze_only=args.analyze_only,
    )


if __name__ == "__main__":
    main()
