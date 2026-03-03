"""
NHiTS-Protocol Benchmark for Novel Block Types

Aligns our pipeline with the NHiTS evaluation protocol, then benchmarks
our best novel block types through both NBeatsNet and NHiTSNet.

Protocol differences addressed:
  1. Z-score normalized data (train on normalized, evaluate on normalized)
  2. 70/10/20 train/val/test split (vs our 80/20)
  3. All columns including OT target (21 Weather cols, 862 Traffic cols)
  4. MSE loss (vs our SMAPE)
  5. L = 5*H lookback (matching NHiTS)
  6. 8 runs per config with seeds 1-8

Usage:
    python experiments/run_nhits_benchmark.py --dry-run
    python experiments/run_nhits_benchmark.py --dataset weather --horizons 96
    python experiments/run_nhits_benchmark.py --dataset traffic --horizons 96 192 336 720
    python experiments/run_nhits_benchmark.py --analyze
"""

import argparse
import csv
import gc
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
from contextlib import contextmanager

# File locking — cross-platform
if sys.platform == "win32":
    import msvcrt

    @contextmanager
    def _exclusive_lock(file_obj):
        file_obj.seek(0)
        while True:
            try:
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.05)
        try:
            yield
        finally:
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    @contextmanager
    def _exclusive_lock(file_obj):
        fcntl.flock(file_obj, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file_obj, fcntl.LOCK_UN)

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Allow running from project root or experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lightningnbeats.models import NBeatsNet, NHiTSNet
from lightningnbeats.loaders import (
    ColumnarCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
)
from lightningnbeats.data import TrafficDataset, WeatherDataset

torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------------------
# Signal Handling
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\n[SIGNAL] Force exit.")
        os._exit(1)
    _shutdown_requested = True
    print("\n[SIGNAL] Shutdown requested. Finishing current run... "
          "(signal again to force)")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# NHiTS Protocol Constants
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
LOSS = "MSELoss"
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
FORECAST_MULTIPLIER = 5   # L = 5*H
THETAS_DIM = 5
LATENT_DIM = 4
BASIS_DIM = 128
MAX_EPOCHS = 100
PATIENCE = 10
N_RUNS = 8
BASE_SEED = 1              # Seeds 1-8 matching NHiTS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_FILENAME = "nhits_benchmark_results.csv"

# ---------------------------------------------------------------------------
# CSV Schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "model_type", "config_name", "dataset", "horizon",
    "backcast_length", "n_stacks", "n_blocks_per_stack",
    "run", "seed",
    "mse", "mae", "norm_mse", "norm_mae", "smape",
    "n_params", "training_time_seconds", "epochs_trained",
    "stopping_reason", "best_val_loss",
    "loss_function", "normalize", "train_ratio", "val_ratio",
    "include_target", "stack_types",
]


# ---------------------------------------------------------------------------
# Benchmark Configs
# ---------------------------------------------------------------------------

def get_benchmark_configs():
    """Return the 13 benchmark configurations."""
    configs = []

    # --- NBeatsNet configs (6) ---
    configs.append({
        "model_type": "NBeatsNet",
        "config_name": "GenericAELG-10",
        "stack_types": ["GenericAELG"] * 10,
        "n_blocks_per_stack": 1,
    })
    configs.append({
        "model_type": "NBeatsNet",
        "config_name": "TrendAELG+HaarV3AELG",
        "stack_types": ["TrendAELG", "HaarWaveletV3AELG"] * 5,
        "n_blocks_per_stack": 1,
    })
    configs.append({
        "model_type": "NBeatsNet",
        "config_name": "TrendAELG+DB2V3AELG",
        "stack_types": ["TrendAELG", "DB2WaveletV3AELG"] * 5,
        "n_blocks_per_stack": 1,
    })
    configs.append({
        "model_type": "NBeatsNet",
        "config_name": "TrendAELG+Coif1V3AELG",
        "stack_types": ["TrendAELG", "Coif1WaveletV3AELG"] * 5,
        "n_blocks_per_stack": 1,
    })
    configs.append({
        "model_type": "NBeatsNet",
        "config_name": "TrendAELG+Symlet2V3AELG",
        "stack_types": ["TrendAELG", "Symlet2WaveletV3AELG"] * 5,
        "n_blocks_per_stack": 1,
    })
    configs.append({
        "model_type": "NBeatsNet",
        "config_name": "TrendWaveletAELG-10",
        "stack_types": ["TrendWaveletAELG"] * 10,
        "n_blocks_per_stack": 1,
    })

    # --- NHiTSNet configs (7) ---
    configs.append({
        "model_type": "NHiTSNet",
        "config_name": "NHiTS-GenericAELG",
        "stack_types": ["GenericAELG"] * 3,
        "n_blocks_per_stack": 1,
        "n_pools_kernel_size": [8, 4, 1],
        "n_freq_downsample": [24, 12, 1],
    })
    configs.append({
        "model_type": "NHiTSNet",
        "config_name": "NHiTS-TrendAELG+HaarV3AELG",
        "stack_types": ["TrendAELG", "HaarWaveletV3AELG", "GenericAELG"],
        "n_blocks_per_stack": 1,
        "n_pools_kernel_size": [8, 4, 1],
        "n_freq_downsample": [24, 12, 1],
    })
    configs.append({
        "model_type": "NHiTSNet",
        "config_name": "NHiTS-TrendAELG+DB2V3AELG",
        "stack_types": ["TrendAELG", "DB2WaveletV3AELG", "GenericAELG"],
        "n_blocks_per_stack": 1,
        "n_pools_kernel_size": [8, 4, 1],
        "n_freq_downsample": [24, 12, 1],
    })
    configs.append({
        "model_type": "NHiTSNet",
        "config_name": "NHiTS-TrendAELG+Coif1V3AELG",
        "stack_types": ["TrendAELG", "Coif1WaveletV3AELG", "GenericAELG"],
        "n_blocks_per_stack": 1,
        "n_pools_kernel_size": [8, 4, 1],
        "n_freq_downsample": [24, 12, 1],
    })
    configs.append({
        "model_type": "NHiTSNet",
        "config_name": "NHiTS-TrendAELG+Symlet2V3AELG",
        "stack_types": ["TrendAELG", "Symlet2WaveletV3AELG", "GenericAELG"],
        "n_blocks_per_stack": 1,
        "n_pools_kernel_size": [8, 4, 1],
        "n_freq_downsample": [24, 12, 1],
    })
    configs.append({
        "model_type": "NHiTSNet",
        "config_name": "NHiTS-TrendWaveletAELG",
        "stack_types": ["TrendWaveletAELG"] * 3,
        "n_blocks_per_stack": 1,
        "n_pools_kernel_size": [8, 4, 1],
        "n_freq_downsample": [24, 12, 1],
    })

    return configs


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def set_seed(seed):
    pl.seed_everything(seed, workers=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_smape(y_pred, y_true):
    eps = 1e-8
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return np.mean(numerator / denominator) * 100.0


def compute_mae(y_pred, y_true):
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(y_pred, y_true):
    return float(np.mean((y_true - y_pred) ** 2))


def compute_normalized_mae_mse(preds, targets, train_data_df):
    """Compute Z-score normalized MAE and MSE using training data statistics."""
    eps = 1e-8
    cols = train_data_df.columns
    n_series = preds.shape[0]

    norm_abs_errors = []
    norm_sq_errors = []

    for i in range(n_series):
        col = cols[i] if i < len(cols) else cols[-1]
        series_vals = train_data_df[col].dropna().values
        if len(series_vals) == 0:
            continue

        mu = float(np.mean(series_vals))
        sigma = float(np.std(series_vals))
        if sigma < eps:
            sigma = 1.0

        norm_pred = (preds[i] - mu) / sigma
        norm_true = (targets[i] - mu) / sigma
        norm_abs_errors.append(np.mean(np.abs(norm_pred - norm_true)))
        norm_sq_errors.append(np.mean((norm_pred - norm_true) ** 2))

    norm_mae = float(np.mean(norm_abs_errors)) if norm_abs_errors else float("nan")
    norm_mse = float(np.mean(norm_sq_errors)) if norm_sq_errors else float("nan")
    return norm_mae, norm_mse


def resolve_accelerator():
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    elif torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


# ---------------------------------------------------------------------------
# Callbacks (from run_unified_benchmark.py)
# ---------------------------------------------------------------------------

class DivergenceDetector(pl.Callback):
    def __init__(self, relative_threshold=3.0, consecutive_epochs=3):
        super().__init__()
        self.relative_threshold = relative_threshold
        self.consecutive_epochs = consecutive_epochs
        self.best_val_loss = float("inf")
        self.bad_epoch_count = 0
        self.diverged = False

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get("val_loss")
        if v is None:
            return
        val_loss = float(v)
        if not math.isfinite(val_loss):
            self.diverged = True
            trainer.should_stop = True
            return
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.bad_epoch_count = 0
            return
        if self.best_val_loss > 0 and val_loss > self.best_val_loss * self.relative_threshold:
            self.bad_epoch_count += 1
        else:
            self.bad_epoch_count = 0
        if self.bad_epoch_count >= self.consecutive_epochs:
            self.diverged = True
            trainer.should_stop = True


class ConvergenceTracker(pl.Callback):
    def __init__(self):
        super().__init__()
        self.val_losses = []
        self.train_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get("val_loss")
        if v is not None:
            self.val_losses.append(float(v))

    def on_train_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("train_loss")
        if v is not None:
            self.train_losses.append(float(v))


# ---------------------------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------------------------

def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        return

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            existing_header = next(reader)
        except StopIteration:
            existing_header = []
    if existing_header == CSV_COLUMNS:
        return

    # Migrate header
    print(f"  [MIGRATE] {os.path.basename(path)}: "
          f"header {len(existing_header)} cols -> {len(CSV_COLUMNS)} cols")
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        old_header = next(reader)
        raw_rows = list(reader)

    migrated = []
    for raw in raw_rows:
        row_dict = {}
        for i, col_name in enumerate(old_header):
            if col_name in CSV_COLUMNS and i < len(raw):
                row_dict[col_name] = raw[i]
        for col in CSV_COLUMNS:
            if col not in row_dict:
                row_dict[col] = ""
        migrated.append(row_dict)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated)


def append_result(path, row_dict):
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock_f:
        with _exclusive_lock(lock_f):
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(row_dict)


def result_exists(path, config_name, dataset, horizon, run):
    if not os.path.exists(path):
        return False
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("config_name") == config_name
                    and row.get("dataset") == dataset
                    and row.get("horizon") == str(horizon)
                    and row.get("run") == str(run)):
                return True
    return False


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, test_dm, device):
    model.eval()
    model.to(device)
    all_preds = []
    all_targets = []
    test_dm.setup("test")
    with torch.no_grad():
        for batch in test_dm.test_dataloader():
            x, y = batch
            x = x.to(device)
            _, forecast = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


# ---------------------------------------------------------------------------
# Single Experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    config,
    dataset_name,
    horizon,
    run_idx,
    csv_path,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
):
    """Run a single NHiTS-protocol experiment."""
    global _shutdown_requested
    if _shutdown_requested:
        return

    config_name = config["config_name"]
    model_type = config["model_type"]

    # Check resumability
    if result_exists(csv_path, config_name, dataset_name, horizon, run_idx):
        print(f"  [SKIP] {config_name} / {dataset_name}-{horizon} / run {run_idx}")
        return

    seed = BASE_SEED + run_idx
    set_seed(seed)

    # Load dataset with NHiTS protocol params
    if dataset_name == "weather":
        dataset = WeatherDataset(
            horizon, train_ratio=TRAIN_RATIO, include_target=True)
    elif dataset_name == "traffic":
        dataset = TrafficDataset(
            horizon, train_ratio=TRAIN_RATIO, include_target=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    forecast_length = dataset.forecast_length
    backcast_length = forecast_length * FORECAST_MULTIPLIER

    stack_types = config["stack_types"]
    n_stacks = len(stack_types)
    n_blocks_per_stack = config.get("n_blocks_per_stack", 1)

    accelerator, device = resolve_accelerator()

    # Precision
    if accelerator == "cuda" and torch.cuda.is_bf16_supported():
        precision = "bf16-mixed"
    elif accelerator == "cuda":
        precision = "32-true"
    else:
        precision = "32-true"

    # Create model
    model_kwargs = dict(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        stack_types=stack_types,
        n_blocks_per_stack=n_blocks_per_stack,
        share_weights=True,
        thetas_dim=THETAS_DIM,
        loss=LOSS,
        active_g=False,
        sum_losses=False,
        activation="ReLU",
        latent_dim=LATENT_DIM,
        basis_dim=BASIS_DIM,
        learning_rate=LEARNING_RATE,
        optimizer_name="Adam",
        no_val=False,
    )

    if model_type == "NHiTSNet":
        model_kwargs["n_pools_kernel_size"] = config["n_pools_kernel_size"]
        model_kwargs["n_freq_downsample"] = config["n_freq_downsample"]
        ModelClass = NHiTSNet
    else:
        ModelClass = NBeatsNet

    model = ModelClass(**model_kwargs)
    n_params = count_parameters(model)

    # Data modules — NHiTS protocol: normalize=True, val_ratio=0.1
    train_data = dataset.train_data
    test_data = dataset.test_data

    dm = ColumnarCollectionTimeSeriesDataModule(
        train_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=batch_size,
        no_val=False,
        normalize=True,
        val_ratio=VAL_RATIO,
    )

    # We need to call setup() on dm first to get normalization stats
    dm.setup()

    test_dm = ColumnarCollectionTimeSeriesTestDataModule(
        train_data,
        test_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=batch_size,
        col_means=dm.col_means,
        col_stds=dm.col_stds,
    )

    # Trainer
    ckpt_tmp_dir = tempfile.mkdtemp(prefix="nhits_ckpt_")
    chk_callback = ModelCheckpoint(
        dirpath=ckpt_tmp_dir,
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=False,
    )
    divergence_detector = DivergenceDetector(relative_threshold=3.0, consecutive_epochs=3)
    convergence_tracker = ConvergenceTracker()

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        precision=precision,
        gradient_clip_val=1.0,
        callbacks=[chk_callback, early_stop_callback, divergence_detector, convergence_tracker],
        logger=False,
        enable_progress_bar=True,
        deterministic=False,
        log_every_n_steps=1,
    )

    # Train
    stack_summary = (f"{n_stacks}x{stack_types[0]}" if len(set(stack_types)) == 1
                     else f"{n_stacks} mixed")
    print(f"  [RUN]  {config_name} / {dataset_name}-{horizon} / run {run_idx} "
          f"(seed={seed}, {stack_summary}, params={n_params:,})")
    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    training_time = time.time() - t0

    # Load best checkpoint
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        model = ModelClass.load_from_checkpoint(best_path, weights_only=False)
    shutil.rmtree(ckpt_tmp_dir, ignore_errors=True)
    epochs_trained = trainer.current_epoch

    # Stopping reason
    if divergence_detector.diverged:
        stopping_reason = "DIVERGED"
    elif hasattr(early_stop_callback, "stopped_epoch") and early_stop_callback.stopped_epoch > 0:
        stopping_reason = "EARLY_STOPPED"
    else:
        stopping_reason = "MAX_EPOCHS"

    best_val_loss = (float(trainer.checkpoint_callback.best_model_score)
                     if trainer.checkpoint_callback.best_model_score is not None
                     else float("nan"))

    # Inference — predictions are in normalized space since test_dm uses the
    # same col_means/col_stds from training
    preds, targets = run_inference(model, test_dm, device)

    # Metrics on normalized data (primary — matches NHiTS evaluation)
    mse_norm = compute_mse(preds, targets)
    mae_norm = compute_mae(preds, targets)

    # Also compute sMAPE and unnormalized metrics for our reference
    # Unnormalized metrics via compute_normalized_mae_mse on raw data
    norm_mae, norm_mse = compute_normalized_mae_mse(preds, targets, train_data)
    smape = compute_smape(preds, targets)

    print(f"         MSE={mse_norm:.4f}  MAE={mae_norm:.4f}  "
          f"nMSE={norm_mse:.4f}  nMAE={norm_mae:.4f}  sMAPE={smape:.2f}  "
          f"time={training_time:.1f}s  epochs={epochs_trained}  [{stopping_reason}]")

    # Save result
    row = {
        "model_type": model_type,
        "config_name": config_name,
        "dataset": dataset_name,
        "horizon": horizon,
        "backcast_length": backcast_length,
        "n_stacks": n_stacks,
        "n_blocks_per_stack": n_blocks_per_stack,
        "run": run_idx,
        "seed": seed,
        "mse": f"{mse_norm:.6f}",
        "mae": f"{mae_norm:.6f}",
        "norm_mse": f"{norm_mse:.6f}" if math.isfinite(norm_mse) else "nan",
        "norm_mae": f"{norm_mae:.6f}" if math.isfinite(norm_mae) else "nan",
        "smape": f"{smape:.6f}" if math.isfinite(smape) else "nan",
        "n_params": n_params,
        "training_time_seconds": f"{training_time:.2f}",
        "epochs_trained": epochs_trained,
        "stopping_reason": stopping_reason,
        "best_val_loss": f"{best_val_loss:.8f}" if math.isfinite(best_val_loss) else "nan",
        "loss_function": LOSS,
        "normalize": True,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "include_target": True,
        "stack_types": str(list(dict.fromkeys(stack_types))),
    }
    append_result(csv_path, row)

    # Cleanup
    del model, trainer, dm, test_dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(csv_path):
    """Print a comparison table from existing results."""
    if not os.path.exists(csv_path):
        print(f"No results file found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Results file is empty.")
        return

    print("\n" + "=" * 100)
    print("NHiTS-Protocol Benchmark Results")
    print("=" * 100)

    # Published NHiTS baselines for comparison
    nhits_baselines = {
        ("weather", 96): {"mse": 0.158, "mae": 0.209},
        ("weather", 192): {"mse": 0.211, "mae": 0.253},
        ("weather", 336): {"mse": 0.272, "mae": 0.296},
        ("weather", 720): {"mse": 0.348, "mae": 0.349},
        ("traffic", 96): {"mse": 0.401, "mae": 0.267},
        ("traffic", 192): {"mse": 0.411, "mae": 0.270},
        ("traffic", 336): {"mse": 0.423, "mae": 0.278},
        ("traffic", 720): {"mse": 0.461, "mae": 0.299},
    }

    for dataset_name in df["dataset"].unique():
        print(f"\n{'─' * 100}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'─' * 100}")

        ds_df = df[df["dataset"] == dataset_name]

        for horizon in sorted(ds_df["horizon"].unique()):
            h_df = ds_df[ds_df["horizon"] == horizon]

            baseline = nhits_baselines.get((dataset_name, int(horizon)), {})
            baseline_mse = baseline.get("mse", float("nan"))
            baseline_mae = baseline.get("mae", float("nan"))

            print(f"\n  Horizon: {int(horizon)}  "
                  f"(NHiTS published: MSE={baseline_mse:.3f}, MAE={baseline_mae:.3f})")
            print(f"  {'Config':<40s} {'Model':<12s} {'MSE':>8s} {'MAE':>8s} "
                  f"{'vs NHiTS':>9s} {'Runs':>5s}")
            print(f"  {'─' * 85}")

            summary = (h_df.groupby(["config_name", "model_type"])
                       .agg(
                           mse_mean=("mse", lambda x: pd.to_numeric(x, errors="coerce").mean()),
                           mae_mean=("mae", lambda x: pd.to_numeric(x, errors="coerce").mean()),
                           mse_std=("mse", lambda x: pd.to_numeric(x, errors="coerce").std()),
                           n_runs=("run", "count"),
                       )
                       .reset_index()
                       .sort_values("mse_mean"))

            for _, row in summary.iterrows():
                mse_m = row["mse_mean"]
                mae_m = row["mae_mean"]
                mse_s = row["mse_std"]
                n = int(row["n_runs"])
                gap = ((mse_m - baseline_mse) / baseline_mse * 100
                       if math.isfinite(baseline_mse) and baseline_mse > 0
                       else float("nan"))
                gap_str = f"{gap:+.1f}%" if math.isfinite(gap) else "N/A"

                print(f"  {row['config_name']:<40s} {row['model_type']:<12s} "
                      f"{mse_m:8.4f} {mae_m:8.4f} {gap_str:>9s} {n:5d}")

    print(f"\n{'=' * 100}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NHiTS-Protocol Benchmark for Novel Block Types")
    parser.add_argument("--dataset", type=str, nargs="+",
                        default=["weather", "traffic"],
                        choices=["weather", "traffic"],
                        help="Dataset(s) to benchmark")
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[96, 192, 336, 720],
                        help="Forecast horizons to test")
    parser.add_argument("--n-runs", type=int, default=N_RUNS,
                        help="Number of runs per config (default: 8)")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                        help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--configs", type=str, nargs="*", default=None,
                        help="Filter to specific config names")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment plan without running")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results and print comparison table")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Override CSV output path")

    args = parser.parse_args()

    csv_path = args.csv_path or os.path.join(RESULTS_DIR, CSV_FILENAME)

    if args.analyze:
        analyze_results(csv_path)
        return

    all_configs = get_benchmark_configs()
    if args.configs:
        all_configs = [c for c in all_configs if c["config_name"] in args.configs]
        if not all_configs:
            print(f"No configs matched: {args.configs}")
            print(f"Available: {[c['config_name'] for c in get_benchmark_configs()]}")
            return

    # Build job list
    jobs = []
    for dataset_name in args.dataset:
        for horizon in args.horizons:
            for config in all_configs:
                for run_idx in range(args.n_runs):
                    jobs.append({
                        "config": config,
                        "dataset_name": dataset_name,
                        "horizon": horizon,
                        "run_idx": run_idx,
                    })

    total_jobs = len(jobs)
    print(f"\nNHiTS-Protocol Benchmark")
    print(f"  Datasets:   {args.dataset}")
    print(f"  Horizons:   {args.horizons}")
    print(f"  Configs:    {len(all_configs)}")
    print(f"  Runs/cfg:   {args.n_runs}")
    print(f"  Total jobs: {total_jobs}")
    print(f"  CSV path:   {csv_path}")
    print(f"  Protocol:   normalize=True, train_ratio={TRAIN_RATIO}, "
          f"val_ratio={VAL_RATIO}, loss={LOSS}, L=5H")
    print()

    if args.dry_run:
        print("Configs to run:")
        for config in all_configs:
            n = len(config["stack_types"])
            unique = list(dict.fromkeys(config["stack_types"]))
            print(f"  {config['config_name']:<40s} {config['model_type']:<12s} "
                  f"{n} stacks: {unique}")
        print(f"\n{total_jobs} total experiments (dry run — not executing)")
        return

    init_csv(csv_path)

    completed = 0
    for job in jobs:
        if _shutdown_requested:
            print("\n[SHUTDOWN] Stopping after current run.")
            break

        run_single_experiment(
            config=job["config"],
            dataset_name=job["dataset_name"],
            horizon=job["horizon"],
            run_idx=job["run_idx"],
            csv_path=csv_path,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
        completed += 1

    print(f"\nCompleted {completed}/{total_jobs} experiments.")
    print(f"Results saved to: {csv_path}")

    # Auto-analyze if we completed all jobs
    if completed == total_jobs:
        analyze_results(csv_path)


if __name__ == "__main__":
    main()
