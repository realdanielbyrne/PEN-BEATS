#!/usr/bin/env python3
"""Run benchmark evaluations for completed smol paper variants.

Completion is detected by the presence of
``trainedmodels/<exp>/<variant>/training_manifest.json``. Benchmark completion
is tracked separately from the training validation summaries under
``<output_dir>/<exp>/<variant>/benchmarks/`` so this can run periodically while
the parallel training launcher is still finishing variants.

By default benchmark results, manifests, and the markdown report are written to
``<repo_root>/evals/`` so they can be version-controlled.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_paths import (  # noqa: E402
    EVALS_DIR,
    TRAINED_MODELS_DIR,
    apply_data_drive_env_defaults,
)

REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_CONFIG = REPO_ROOT / "scripts" / "experiments" / "smol_replacement_paper.yaml"
DEFAULT_TASKS = (
    "lambada_openai",
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "openbookqa",
)
LM_EVAL_BOOTSTRAP = (
    "import pellm, runpy; "
    "runpy.run_module('lm_eval', run_name='__main__', alter_sys=True)"
)
RESULTS_FILENAME = "results.json"
MANIFEST_FILENAME = "benchmark_manifest.json"
REPORT_FILENAME = "benchmark_report.md"
TASK_DISPLAY_NAMES = {
    "lambada_openai": "LAMBADA",
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "winogrande": "WinoGrande",
    "openbookqa": "OpenBookQA",
}
PREFERRED_TASK_METRICS = {
    "lambada_openai": ("acc,none", "acc"),
    "hellaswag": ("acc_norm,none", "acc_norm"),
    "piqa": ("acc_norm,none", "acc_norm"),
    "arc_easy": ("acc_norm,none", "acc_norm"),
    "arc_challenge": ("acc_norm,none", "acc_norm"),
    "winogrande": ("acc,none", "acc"),
    "openbookqa": ("acc_norm,none", "acc_norm"),
}
FALLBACK_METRICS = (
    "acc_norm,none",
    "acc_norm",
    "acc,none",
    "acc",
    "exact_match,none",
    "exact_match",
    "f1,none",
    "f1",
    "word_perplexity,none",
    "word_perplexity",
    "perplexity,none",
    "perplexity",
)
# Map metric base name to whether a larger value is better.
METRIC_DIRECTION: dict[str, bool] = {
    "acc": True,
    "acc_norm": True,
    "exact_match": True,
    "f1": True,
    "perplexity": False,
    "word_perplexity": False,
}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Install the experiments extra: pip install -e '.[experiments]'"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_complete(exp_name: str, variant: str) -> bool:
    return (
        TRAINED_MODELS_DIR / exp_name / variant / "training_manifest.json"
    ).is_file()


def parse_tasks(tasks: str) -> list[str]:
    parsed = [task.strip() for task in tasks.split(",") if task.strip()]
    if not parsed:
        raise ValueError("--tasks must include at least one lm-eval task")
    return parsed


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_results_file(benchmark_dir: Path) -> Path | None:
    """Locate the most recent lm-eval results JSON anywhere under *benchmark_dir*."""
    exact = benchmark_dir / RESULTS_FILENAME
    if exact.is_file():
        return exact
    # lm-eval may nest results in a subdirectory derived from the model path
    candidates = sorted(
        benchmark_dir.rglob("results*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def is_valid_results(path: Path) -> bool:
    try:
        results = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(results, dict) and isinstance(results.get("results"), dict)


def is_valid_benchmark_manifest(
    path: Path,
    *,
    expected_tasks: list[str],
    expected_num_fewshot: int,
    expected_limit: str | None,
    expected_model_args_extra: str | None,
) -> bool:
    try:
        manifest = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(manifest, dict):
        return False
    if manifest.get("status") != "success":
        return False
    if manifest.get("tasks") != expected_tasks:
        return False
    if manifest.get("num_fewshot") != expected_num_fewshot:
        return False
    if manifest.get("limit") != expected_limit:
        return False
    if manifest.get("model_args_extra") != expected_model_args_extra:
        return False

    results_path = manifest.get("results_path")
    if not isinstance(results_path, str):
        return False
    return is_valid_results(Path(results_path))


def build_command(
    *,
    model_dir: Path,
    tasks: list[str],
    benchmark_dir: Path,
    device: str,
    batch_size: str,
    num_fewshot: int,
    limit: str | None,
    model_args_extra: str | None,
) -> list[str]:
    model_args = [
        f"pretrained={model_dir}",
        f"tokenizer={model_dir}",
    ]
    if model_args_extra:
        model_args.extend(
            arg.strip() for arg in model_args_extra.split(",") if arg.strip()
        )

    cmd = [
        sys.executable,
        "-c",
        LM_EVAL_BOOTSTRAP,
        "--model",
        "hf",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(tasks),
        "--device",
        device,
        "--batch_size",
        batch_size,
        "--num_fewshot",
        str(num_fewshot),
        "--output_path",
        str(benchmark_dir),
    ]
    if limit:
        cmd += ["--limit", limit]
    return cmd


def write_manifest(
    *,
    path: Path,
    exp_name: str,
    variant: str,
    model_dir: Path,
    results_path: Path,
    tasks: list[str],
    device: str,
    batch_size: str,
    num_fewshot: int,
    limit: str | None,
    model_args_extra: str | None,
    command: list[str],
) -> None:
    manifest = {
        "status": "success",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "experiment": exp_name,
        "variant": variant,
        "model_dir": str(model_dir),
        "results_path": str(results_path),
        "tasks": tasks,
        "device": device,
        "batch_size": batch_size,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "model_args_extra": model_args_extra,
        "command": command,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def shell_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def lm_eval_available() -> bool:
    return importlib.util.find_spec("lm_eval") is not None


def task_display_name(task: str) -> str:
    return TASK_DISPLAY_NAMES.get(task, task)


def normalize_metric_name(metric: str) -> str:
    return metric.split(",", 1)[0]


def metric_candidates(task: str) -> list[str]:
    candidates: list[str] = []
    preferred = PREFERRED_TASK_METRICS.get(task, ())
    for metric in (*preferred, *FALLBACK_METRICS):
        if metric not in candidates:
            candidates.append(metric)
    return candidates


def is_percent_metric(metric: str) -> bool:
    name = normalize_metric_name(metric)
    return name in {"acc", "acc_norm", "exact_match", "f1"}


def higher_is_better(metric: str) -> bool | None:
    """Return True if a larger value is better, False if smaller is better, None if unknown."""
    return METRIC_DIRECTION.get(normalize_metric_name(metric))


def direction_arrow(metric: str) -> str:
    """Return ↑ if larger is better, ↓ if smaller is better, or empty string if unknown."""
    hib = higher_is_better(metric)
    if hib is True:
        return "↑"
    if hib is False:
        return "↓"
    return ""


def format_metric_value(value: Any, metric: str) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    if is_percent_metric(metric):
        return f"{100.0 * float(value):.2f}"
    if "perplexity" in normalize_metric_name(metric):
        return f"{float(value):.2f}"
    return f"{float(value):.4g}"


def metric_header(task: str) -> str:
    preferred = PREFERRED_TASK_METRICS.get(task)
    if preferred:
        metric = normalize_metric_name(preferred[0])
        suffix = " (%)" if is_percent_metric(preferred[0]) else ""
        arrow = direction_arrow(preferred[0])
        return f"{task_display_name(task)} {metric}{suffix} {arrow}"
    return task_display_name(task)


def select_task_metric(task_results: Any, task: str) -> tuple[str, Any] | None:
    if not isinstance(task_results, dict):
        return None
    for metric in metric_candidates(task):
        if metric in task_results:
            return metric, task_results[metric]
    return None


def format_task_cell(results: dict[str, Any], task: str) -> str:
    task_results = results.get(task)
    selected = select_task_metric(task_results, task)
    if selected is None:
        return "-"
    metric, value = selected
    formatted = format_metric_value(value, metric)
    arrow = direction_arrow(metric)
    preferred = PREFERRED_TASK_METRICS.get(task)
    if preferred and metric == preferred[0]:
        return f"{formatted} {arrow}"
    return f"{formatted} ({normalize_metric_name(metric)}) {arrow}"


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def benchmark_paths(
    exp_name: str, variant: str, output_dir: Path = EVALS_DIR
) -> tuple[Path, Path, Path]:
    benchmark_dir = output_dir / exp_name / variant / "benchmarks"
    return (
        benchmark_dir,
        benchmark_dir / RESULTS_FILENAME,
        benchmark_dir / MANIFEST_FILENAME,
    )


def read_matching_benchmark(
    *,
    manifest_path: Path,
    expected_tasks: list[str],
    expected_num_fewshot: int,
    expected_limit: str | None,
    expected_model_args_extra: str | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not is_valid_benchmark_manifest(
        manifest_path,
        expected_tasks=expected_tasks,
        expected_num_fewshot=expected_num_fewshot,
        expected_limit=expected_limit,
        expected_model_args_extra=expected_model_args_extra,
    ):
        return None

    manifest = load_json(manifest_path)
    results = load_json(Path(manifest["results_path"]))
    if not isinstance(manifest, dict) or not isinstance(results, dict):
        return None
    return manifest, results


def write_markdown_report(
    *,
    path: Path,
    exp_name: str,
    config: Path,
    variants: list[str],
    tasks: list[str],
    num_fewshot: int,
    limit: str | None,
    model_args_extra: str | None,
    failures: list[tuple[str, int]],
    output_dir: Path = EVALS_DIR,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Smol Replacement Benchmark Report",
        "",
        f"- **Generated:** `{generated_at}`",
        f"- **Experiment:** `{exp_name}`",
        f"- **Config:** `{config}`",
        f"- **Tasks:** `{','.join(tasks)}`",
        f"- **Few-shot:** `{num_fewshot}`",
        f"- **Limit:** `{limit if limit is not None else 'none'}`",
        "",
        "## Metric Legend",
        "",
        "| Symbol | Meaning |",
        "|--------|---------|",
        "| **↑** | Higher value is better |",
        "| **↓** | Lower value is better |",
        "| **(%)** | Percentage (0–100 scale, derived from 0–1 accuracy scores) |",
        "",
        "### Task Metrics",
        "",
        "- **LAMBADA** — `acc` (%): Next-token prediction accuracy on the LAMBADA dataset (↑). A perplexity score is also reported for LAMBADA (↓).",
        "- **HellaSwag** — `acc_norm` (%): Normalized accuracy on commonsense sentence completion (↑).",
        "- **PIQA** — `acc_norm` (%): Normalized accuracy on physical commonsense reasoning (↑).",
        "- **ARC-Easy / ARC-Challenge** — `acc_norm` (%): Normalized accuracy on grade-school science questions (↑).",
        "- **WinoGrande** — `acc` (%): Accuracy on pronoun resolution / commonsense reasoning (↑).",
        "- **OpenBookQA** — `acc_norm` (%): Normalized accuracy on open-book science QA (↑).",
        "",
        "*All accuracy-like metrics (`acc`, `acc_norm`, `exact_match`, `f1`) are reported as percentages (0–100). Higher is better. Perplexity is reported as a raw value; lower is better.*",
        "",
    ]

    headers = [
        "Variant",
        "Status",
        *[metric_header(task) for task in tasks],
        "Completed",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for variant in variants:
        _, _, manifest_path = benchmark_paths(exp_name, variant, output_dir=output_dir)
        row_status = "training incomplete"
        completed_at = "-"
        task_cells = ["-"] * len(tasks)

        if is_complete(exp_name, variant):
            benchmark = read_matching_benchmark(
                manifest_path=manifest_path,
                expected_tasks=tasks,
                expected_num_fewshot=num_fewshot,
                expected_limit=limit,
                expected_model_args_extra=model_args_extra,
            )
            if benchmark is None:
                row_status = "pending benchmark"
            else:
                manifest, raw_results = benchmark
                results = raw_results.get("results", {})
                if isinstance(results, dict):
                    task_cells = [format_task_cell(results, task) for task in tasks]
                row_status = "complete"
                completed_at = str(manifest.get("completed_at", "-"))

        row = [variant, row_status, *task_cells, completed_at]
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in row) + " |")

    if failures:
        lines.extend(["", "## Latest Run Failures", ""])
        lines.append("| Variant | Return code |")
        lines.append("| --- | ---: |")
        for variant, returncode in failures:
            lines.append(f"| {markdown_escape(variant)} | {returncode} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run lm-evaluation-harness benchmarks for completed smol "
            "replacement paper variants."
        ),
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        type=Path,
        help=(
            "YAML experiment config "
            "(default: scripts/experiments/smol_replacement_paper.yaml)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "evals",
        help=(
            "Directory for benchmark results, manifests, and the markdown report. "
            "Defaults to <repo>/evals so artifacts are version-controlled."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run benchmarks even when benchmark_manifest.json is already valid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and skip reasons without launching benchmarks.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help=(
            "Comma-separated lm-eval tasks for pretrained/base models "
            f"(default: {','.join(DEFAULT_TASKS)})."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device passed to lm-eval (default: cuda:0).",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size passed to lm-eval (default: auto).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Few-shot examples passed to lm-eval (default: 0 for base models).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="Optional lm-eval --limit value for smoke tests.",
    )
    parser.add_argument(
        "--model-args-extra",
        default=None,
        help=(
            "Optional extra comma-separated hf model_args, for example "
            "dtype=bfloat16."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help=(
            "Markdown report path. Defaults to "
            "<output-dir>/<experiment>/benchmark_report.md."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = args.config.expanduser()
    if not config.is_absolute():
        config = config.resolve()

    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")

    cfg = load_yaml(config)
    exp_name = cfg["experiment"]["name"]
    variants = list((cfg.get("variants") or {}).keys())
    if not variants:
        raise RuntimeError(f"No variants defined in {config}")
    tasks = parse_tasks(args.tasks)
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    report_path = (
        args.report_path.expanduser()
        if args.report_path
        else output_dir / exp_name / REPORT_FILENAME
    )
    if not report_path.is_absolute():
        report_path = report_path.resolve()

    print(f"Experiment: {exp_name}")
    print(f"Config: {config}")
    print(f"Output dir: {output_dir}")
    print(f"Tasks: {','.join(tasks)}")
    print(f"Few-shot: {args.num_fewshot}")
    print(f"Report: {report_path}")
    print(f"Mode: {'dry-run' if args.dry_run else 'run'}")
    if args.force:
        print("Force: enabled")
    print()

    env = apply_data_drive_env_defaults(os.environ)
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not env.get("PYTHONPATH")
        else f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
    )
    runnable: list[tuple[str, Path, Path, Path, list[str]]] = []
    skipped_benchmarked = 0
    skipped_incomplete = 0
    launched = 0
    failures: list[tuple[str, int]] = []

    for variant in variants:
        model_dir = TRAINED_MODELS_DIR / exp_name / variant
        benchmark_dir, results_path, manifest_path = benchmark_paths(
            exp_name, variant, output_dir=output_dir
        )

        if not is_complete(exp_name, variant):
            print(f"skip  {variant}: training incomplete")
            skipped_incomplete += 1
            continue

        if (
            is_valid_benchmark_manifest(
                manifest_path,
                expected_tasks=tasks,
                expected_num_fewshot=args.num_fewshot,
                expected_limit=args.limit,
                expected_model_args_extra=args.model_args_extra,
            )
            and not args.force
        ):
            print(f"skip  {variant}: benchmarks already complete at {manifest_path}")
            skipped_benchmarked += 1
            continue

        cmd = build_command(
            model_dir=model_dir,
            tasks=tasks,
            benchmark_dir=benchmark_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            model_args_extra=args.model_args_extra,
        )
        runnable.append((variant, model_dir, benchmark_dir, manifest_path, cmd))
        if args.dry_run:
            print(f"would benchmark {variant}: {shell_command(cmd)}")

    if not args.dry_run and runnable:
        if not lm_eval_available():
            print(
                "lm_eval is not installed. Install the benchmark dependency "
                "with: pip install -e '.[experiments]'",
                file=sys.stderr,
            )
            failures.extend((variant, 1) for variant, *_ in runnable)
        else:
            for (
                variant,
                model_dir,
                benchmark_dir,
                manifest_path,
                cmd,
            ) in runnable:
                try:
                    benchmark_dir.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    print(
                        f"fail  {variant}: could not create {benchmark_dir}: {exc}",
                        flush=True,
                    )
                    failures.append((variant, 1))
                    continue
                print(f"bench {variant}: {shell_command(cmd)}", flush=True)
                ret = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
                launched += 1
                if ret.returncode != 0:
                    print(
                        f"fail  {variant}: returncode={ret.returncode}",
                        flush=True,
                    )
                    failures.append((variant, ret.returncode))
                    continue
                results_path = find_results_file(benchmark_dir)
                if results_path is None or not is_valid_results(results_path):
                    print(
                        f"fail  {variant}: missing or invalid results in {benchmark_dir}",
                        flush=True,
                    )
                    failures.append((variant, 1))
                    continue
                write_manifest(
                    path=manifest_path,
                    exp_name=exp_name,
                    variant=variant,
                    model_dir=model_dir,
                    results_path=results_path,
                    tasks=tasks,
                    device=args.device,
                    batch_size=args.batch_size,
                    num_fewshot=args.num_fewshot,
                    limit=args.limit,
                    model_args_extra=args.model_args_extra,
                    command=cmd,
                )

    if args.dry_run:
        print(f"would write report: {report_path}")
    else:
        write_markdown_report(
            path=report_path,
            exp_name=exp_name,
            config=config,
            variants=variants,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            model_args_extra=args.model_args_extra,
            failures=failures,
            output_dir=output_dir,
        )
        print(f"wrote report: {report_path}")

    print()
    print("Summary:")
    print(f"  skipped completed benchmarks: {skipped_benchmarked}")
    print(f"  skipped incomplete training: {skipped_incomplete}")
    benchmark_count = len(runnable) if args.dry_run else launched
    print(
        f"  benchmarks {'planned' if args.dry_run else 'launched'}: {benchmark_count}"
    )
    print(f"  failures: {len(failures)}")
    for variant, returncode in failures:
        print(f"    {variant}: returncode={returncode}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
