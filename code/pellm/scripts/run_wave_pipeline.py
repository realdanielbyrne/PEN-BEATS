#!/usr/bin/env python3
"""Multi-wave incremental MLP replacement pipeline for PE-Llama.

Orchestrates sequential waves of AE MLP layer replacement, where each wave
resumes from the previous wave's trained model.  Unlike ``run_from_yaml.py``
(parallel grid sweeps), waves are strictly sequential and dependent — if wave
N fails, wave N+1 cannot proceed.

Usage::

    python scripts/run_wave_pipeline.py scripts/experiments/full_mlp_pipeline.yaml
    python scripts/run_wave_pipeline.py scripts/experiments/full_mlp_pipeline.yaml --dry-run
    python scripts/run_wave_pipeline.py scripts/experiments/full_mlp_pipeline.yaml --restart-from-wave wave3_layers_8_15
    python scripts/run_wave_pipeline.py scripts/experiments/full_mlp_pipeline.yaml --force-wave wave2_layers_12_15
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import utilities from run_from_yaml.py (same directory)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_from_yaml import (  # noqa: E402
    PARAM_TO_FLAG,
    NARGS_FIELDS,
    BOOL_FLAGS,
    _ORCHESTRATOR_KEYS,
    load_yaml,
    deep_merge,
    build_cmd,
    run_single,
    parse_results,
    append_csv_row,
    save_summary,
    _fmt,
)
from artifact_paths import (  # noqa: E402
    RUNS_DIR,
    TRAINED_MODELS_DIR,
    apply_data_drive_env_defaults,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WAVE_SENTINEL = "wave_complete.json"

# Keys that are pipeline-level only (not forwarded to finetune.py).
_PIPELINE_KEYS = frozenset(
    {
        "name",
        "resume_from",
        "cleanup_intermediate",
        "models_root",
        "run_root",
        "cache_root",
    }
)


def _cleanup_model_files(output_dir: str) -> None:
    """Remove large model weight files but preserve the wave sentinel and config.

    This reclaims disk space from intermediate waves while keeping the sentinel
    so the wave is still considered complete (won't be re-run on resume).
    """
    _preserve = {_WAVE_SENTINEL, "config.json", "generation_config.json"}
    d = Path(output_dir)
    if not d.is_dir():
        return
    for f in d.iterdir():
        if f.name in _preserve:
            continue
        if f.is_file() and (
            f.suffix in (".safetensors", ".bin", ".pt", ".pth")
            or f.name.startswith("model")
        ):
            f.unlink()
        elif f.is_symlink() or f.name.startswith("tokenizer") or f.name.startswith("special_tokens"):
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-wave incremental MLP replacement pipeline for PE-Llama",
    )
    p.add_argument("yaml", help="Path to wave pipeline YAML config file")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running them",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="Run finetune.py --dry-run for each wave (config validation)",
    )
    p.add_argument(
        "--restart-from-wave", metavar="NAME",
        help="Mark this wave and all subsequent as incomplete, then run",
    )
    p.add_argument(
        "--force-wave", metavar="NAME",
        help="Rerun just this one specific wave (even if complete)",
    )
    p.add_argument(
        "--gpu-id", metavar="ID",
        help="Set CUDA_VISIBLE_DEVICES for all subprocesses",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# YAML loading and validation
# ---------------------------------------------------------------------------

def load_pipeline_config(path: str) -> tuple[dict, dict, list[dict]]:
    """Load and validate a wave pipeline YAML config.

    Returns ``(pipeline_meta, defaults, waves)`` where *waves* is an ordered
    list of wave dicts (each with at least a ``name`` key).
    """
    cfg = load_yaml(path)
    pipeline = cfg.get("pipeline", {})
    defaults = cfg.get("defaults", {})
    waves = cfg.get("waves", [])

    if not isinstance(waves, list) or len(waves) == 0:
        raise ValueError("YAML must contain a non-empty 'waves' list")

    names_seen: set[str] = set()
    for i, wave in enumerate(waves):
        name = wave.get("name")
        if not name:
            raise ValueError(f"Wave {i} is missing a 'name' field")
        if name in names_seen:
            raise ValueError(f"Duplicate wave name: '{name}'")
        names_seen.add(name)

        ref = wave.get("resume_from")
        if ref and not os.path.isabs(ref) and ref not in names_seen:
            raise ValueError(
                f"Wave '{name}' references resume_from='{ref}' which is not "
                f"a preceding wave name or absolute path"
            )

    return pipeline, defaults, waves


# ---------------------------------------------------------------------------
# Wave parameter merging
# ---------------------------------------------------------------------------

def merge_wave_params(defaults: dict, wave: dict) -> dict:
    """Merge *defaults* into *wave*, stripping pipeline-only keys."""
    overrides = {k: v for k, v in wave.items() if k not in _PIPELINE_KEYS}
    return deep_merge(defaults, overrides)


# ---------------------------------------------------------------------------
# resume_from resolution
# ---------------------------------------------------------------------------

def resolve_resume_from(
    resume_from: str | None,
    wave_output_dirs: dict[str, str],
) -> str | None:
    """Map a wave name reference to its output directory path.

    Absolute paths are passed through unchanged.
    """
    if resume_from is None:
        return None
    if os.path.isabs(resume_from):
        return resume_from
    if resume_from in wave_output_dirs:
        return wave_output_dirs[resume_from]
    raise ValueError(
        f"Cannot resolve resume_from='{resume_from}': "
        f"not found in wave output dirs or as absolute path"
    )


# ---------------------------------------------------------------------------
# Wave sentinel (completion tracking)
# ---------------------------------------------------------------------------

def is_wave_complete(output_dir: str) -> bool:
    """Check whether a wave has a completion sentinel."""
    return (Path(output_dir) / _WAVE_SENTINEL).exists()


def read_wave_sentinel(output_dir: str) -> dict | None:
    """Read and return the sentinel data, or None if not present."""
    sentinel = Path(output_dir) / _WAVE_SENTINEL
    if not sentinel.exists():
        return None
    with open(sentinel) as f:
        return json.load(f)


def write_wave_sentinel(output_dir: str, data: dict) -> None:
    """Write a completion sentinel with results data."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / _WAVE_SENTINEL, "w") as f:
        json.dump(data, f, indent=2)


def remove_wave_sentinel(output_dir: str) -> None:
    """Remove a wave's completion sentinel if it exists."""
    (Path(output_dir) / _WAVE_SENTINEL).unlink(missing_ok=True)


def invalidate_from_wave(
    waves: list[dict],
    start_name: str,
    wave_output_dirs: dict[str, str],
) -> None:
    """Remove sentinels from *start_name* and all subsequent waves."""
    found = False
    for wave in waves:
        if wave["name"] == start_name:
            found = True
        if found:
            output_dir = wave_output_dirs[wave["name"]]
            remove_wave_sentinel(output_dir)
    if not found:
        raise ValueError(f"Wave '{start_name}' not found in pipeline")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    pipeline, defaults, waves = load_pipeline_config(args.yaml)

    # Pipeline metadata
    name = pipeline.get("name", "wave_pipeline")
    python = pipeline.get("python", "python3")
    extra_env = pipeline.get("env", {})
    wandb_cfg = {
        "enabled": pipeline.get("wandb", False),
        "project": pipeline.get("wandb_project", "pellm"),
    }

    # Directory layout
    scripts_dir = Path(__file__).resolve().parent
    script = str(scripts_dir / "finetune.py")

    run_root = Path(pipeline.get("run_root", RUNS_DIR)).expanduser()
    logs_dir = run_root / name / "logs"
    models_dir = Path(pipeline.get("models_root", TRAINED_MODELS_DIR)).expanduser() / name
    summary_path = logs_dir / f"{name}_results.json"
    csv_path = logs_dir / f"{name}_log.csv"

    # Optional cache_root: redirect AE/attention activation caches to a different
    # filesystem (e.g. a large data drive) instead of the default logs dir.
    _cache_root_cfg = pipeline.get("cache_root")
    cache_root = (
        Path(_cache_root_cfg).expanduser() / name
        if _cache_root_cfg
        else run_root / name / "activation_caches"
    )

    # Subprocess environment
    subprocess_env = apply_data_drive_env_defaults(
        {**os.environ, **{str(k): str(v) for k, v in extra_env.items()}}
    )
    if args.gpu_id is not None:
        subprocess_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Build wave name → output dir mapping
    wave_output_dirs: dict[str, str] = {}
    for wave in waves:
        wave_output_dirs[wave["name"]] = str(models_dir / wave["name"])

    # Handle --restart-from-wave
    if args.restart_from_wave:
        invalidate_from_wave(waves, args.restart_from_wave, wave_output_dirs)

    # Handle --force-wave
    if args.force_wave:
        if args.force_wave not in wave_output_dirs:
            print(f"ERROR: --force-wave '{args.force_wave}' not found in pipeline",
                  file=sys.stderr)
            sys.exit(1)
        remove_wave_sentinel(wave_output_dirs[args.force_wave])

    # Print header
    print(f"\nPipeline: {name}")
    if pipeline.get("description"):
        print(f"  {pipeline['description']}")
    print(f"  Waves: {len(waves)}")
    print(f"  Logs:  {logs_dir}")
    print(f"  Models: {models_dir}")
    print(f"  Cache: {cache_root}")

    all_results: list[dict] = []

    for i, wave in enumerate(waves):
        wave_name = wave["name"]
        output_dir = wave_output_dirs[wave_name]

        print(f"\n{'=' * 60}")
        print(f"[Wave {i + 1}/{len(waves)}] {wave_name}")
        print(f"  Output: {output_dir}")

        # --- Skip check ---
        if not args.dry_run and not args.validate:
            force_this = args.force_wave and args.force_wave == wave_name
            if not force_this and is_wave_complete(output_dir):
                print("  SKIP -- already complete")
                sentinel_data = read_wave_sentinel(output_dir)
                if sentinel_data:
                    all_results.append(sentinel_data)
                continue

        # --- Merge defaults + wave overrides ---
        params = merge_wave_params(defaults, wave)

        # --- Resolve resume_from ---
        raw_resume = wave.get("resume_from")
        resolved_resume = resolve_resume_from(raw_resume, wave_output_dirs)
        if resolved_resume:
            params["resume_from"] = resolved_resume
            print(f"  Resume from: {resolved_resume}")

            # Validate resume source is complete
            if not args.dry_run and not args.validate:
                if not is_wave_complete(resolved_resume):
                    print(f"  ERROR: resume source '{raw_resume}' is not complete",
                          file=sys.stderr)
                    print("  Pipeline halted. Fix the issue or use --restart-from-wave",
                          file=sys.stderr)
                    sys.exit(1)

        # --- Auto-wire ae_teacher for waves 2+ ---
        ae_pretrain_epochs = params.get("ae_pretrain_epochs", 0)
        if resolved_resume and ae_pretrain_epochs and ae_pretrain_epochs > 0:
            if "ae_teacher" not in params or params.get("ae_teacher") is None:
                params["ae_teacher"] = resolved_resume
                print(f"  AE teacher: {resolved_resume} (auto-wired from prior wave)")

        # --- Auto-wire ae_pretrain_layer_indices to only newly-added layers ---
        # When resuming from a prior wave whose AE layers are already trained,
        # restrict AE pre-training to layers added in this wave. Skipped when
        # the user has explicitly set ae_pretrain_layer_indices.
        if (
            resolved_resume
            and ae_pretrain_epochs and ae_pretrain_epochs > 0
            and raw_resume
            and not os.path.isabs(str(raw_resume))
            and "ae_pretrain_layer_indices" not in params
        ):
            prior_wave_cfg = next(
                (w for w in waves if w["name"] == raw_resume), None
            )
            if prior_wave_cfg is not None:
                prior_params = merge_wave_params(defaults, prior_wave_cfg)
                prior_layers = set(prior_params.get("pe_mlp_layer_indices") or [])
                current_layers = set(params.get("pe_mlp_layer_indices") or [])
                new_layers = sorted(current_layers - prior_layers)
                if new_layers and prior_layers:
                    params["ae_pretrain_layer_indices"] = new_layers
                    print(
                        f"  AE pre-train layers: {new_layers} "
                        f"(skipping already-trained: {sorted(prior_layers & current_layers)})"
                    )

        # --- Auto-wire attn_teacher for waves 2+ ---
        attn_pretrain_epochs = params.get("attn_pretrain_epochs", 0)
        if resolved_resume and attn_pretrain_epochs and attn_pretrain_epochs > 0:
            if "attn_teacher" not in params or params.get("attn_teacher") is None:
                params["attn_teacher"] = resolved_resume
                print(f"  Attention teacher: {resolved_resume} (auto-wired from prior wave)")

        # --- Per-wave AE cache dir ---
        if ae_pretrain_epochs and ae_pretrain_epochs > 0:
            wave_cache_dir = str(cache_root / "ae_activation_cache" / wave_name)
            params["ae_cache_dir"] = wave_cache_dir

        # --- Per-wave AE sample rotation ---
        rotate_ae = pipeline.get("rotate_ae_samples", False)
        if rotate_ae and ae_pretrain_epochs and ae_pretrain_epochs > 0:
            if "ae_cache_skip" not in wave or wave.get("ae_cache_skip") is None:
                num_samples = params.get("ae_cache_num_samples", 10000)
                params["ae_cache_skip"] = i * num_samples
                print(f"  AE sample offset: skip {params['ae_cache_skip']} docs (auto-rotated)")

        # --- Per-wave attention cache dir ---
        attn_pretrain_epochs = params.get("attn_pretrain_epochs", 0)
        if attn_pretrain_epochs and attn_pretrain_epochs > 0:
            wave_attn_cache_dir = str(cache_root / "attn_activation_cache" / wave_name)
            params["attn_cache_dir"] = wave_attn_cache_dir

        # --- Build command ---
        cmd = build_cmd(python, script, params, output_dir, wandb_cfg,
                        finetune_dry_run=args.validate)

        if args.dry_run:
            print(f"  CMD: {shlex.join(cmd)}")
            all_results.append({
                "wave_name": wave_name, "wave_index": i, "status": "dry_run",
            })
            continue

        print(f"  Running: {shlex.join(cmd)}")
        print()

        # --- Execute ---
        returncode, stdout, stderr = run_single(cmd, subprocess_env)

        wave_record: dict = {
            "wave_name": wave_name,
            "wave_index": i,
            "params": params,
            "output_dir": output_dir,
            "returncode": returncode,
        }

        if returncode == 0:
            if args.validate:
                print(f"  PASS -- config validated")
                wave_record["status"] = "validated"
            else:
                results = parse_results(stdout)
                wave_record["results"] = results
                wave_record["status"] = "ok"

                if results:
                    final_ppl = results.get("final_ppl")
                    baseline_ppl = results.get("baseline_ppl")
                    ae_ppl = results.get("ae_pretrain_ppl")
                    print(f"\n  baseline_ppl={_fmt(baseline_ppl)}  "
                          f"ae_pretrain_ppl={_fmt(ae_ppl)}  "
                          f"final_ppl={_fmt(final_ppl)}")
                    append_csv_row(csv_path, wave_name, wave_name, i,
                                   params, results, "ok")
                else:
                    print("  WARNING: Could not parse Results JSON from output")
                    wave_record["status"] = "ok_no_results"
                    append_csv_row(csv_path, wave_name, wave_name, i,
                                   params, None, "ok_no_results")

                write_wave_sentinel(output_dir, wave_record)

                # Clean up per-wave AE cache after successful completion
                wave_cache = params.get("ae_cache_dir")
                if wave_cache and os.path.isdir(wave_cache):
                    shutil.rmtree(wave_cache, ignore_errors=True)
                    print(f"  Cleaned up AE cache: {wave_cache}")

                # Clean up per-wave attention cache after successful completion
                wave_attn_cache = params.get("attn_cache_dir")
                if wave_attn_cache and os.path.isdir(wave_attn_cache):
                    shutil.rmtree(wave_attn_cache, ignore_errors=True)
                    print(f"  Cleaned up attention cache: {wave_attn_cache}")

                # Clean up intermediate wave model files if the resumed wave
                # was marked with cleanup_intermediate
                raw_resume = wave.get("resume_from")
                if raw_resume:
                    resume_wave_name = raw_resume if not os.path.isabs(str(raw_resume)) else None
                    if resume_wave_name:
                        resume_wave_cfg = next(
                            (w for w in waves if w["name"] == resume_wave_name), None
                        )
                        if resume_wave_cfg and (
                            resume_wave_cfg.get("cleanup_intermediate")
                            or defaults.get("cleanup_intermediate")
                        ):
                            resume_dir = wave_output_dirs[resume_wave_name]
                            _cleanup_model_files(resume_dir)
                            print(f"  Cleaned up intermediate model: {resume_dir}")
        else:
            wave_record["status"] = "failed"
            wave_record["stderr_tail"] = stderr[-2000:] if stderr else ""
            print(f"\n  FAILED (returncode={returncode})")
            if stderr:
                print(f"  stderr tail:\n{stderr[-800:]}")
            append_csv_row(csv_path, wave_name, wave_name, i,
                           params, None, "failed")

            all_results.append(wave_record)
            save_summary(all_results, summary_path)

            print(f"\n  Pipeline halted at wave '{wave_name}'.")
            print("  Fix the issue and rerun (completed waves will be skipped).")
            sys.exit(1)

        all_results.append(wave_record)
        save_summary(all_results, summary_path)

    # --- Final summary ---
    if not args.dry_run and not args.validate:
        print(f"\n{'=' * 60}")
        print(f"Pipeline complete: {len(all_results)} wave(s)")
        print(f"  Summary: {summary_path}")
        print(f"  CSV log: {csv_path}")
        print()

        # Results table
        print(f"  {'Wave':<30} {'Baseline':>10} {'AE-PT':>10} {'Final':>10}")
        print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")
        for r in all_results:
            res = r.get("results")
            if res:
                print(f"  {r['wave_name']:<30} "
                      f"{_fmt(res.get('baseline_ppl')):>10} "
                      f"{_fmt(res.get('ae_pretrain_ppl')):>10} "
                      f"{_fmt(res.get('final_ppl')):>10}")
    elif args.dry_run:
        print(f"\n{'=' * 60}")
        print(f"Dry run complete: {len(waves)} wave(s) shown")


if __name__ == "__main__":
    main()
