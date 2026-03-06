"""Regression tests for YAML launcher protocol plumbing."""

import os
import sys
from types import SimpleNamespace

import pytest


_EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

import run_from_yaml as ryf  # noqa: E402
import run_unified_benchmark as rub  # noqa: E402


def test_resolve_config_uses_protocol_fallbacks_for_training_fields():
    top_level_cfg = {
        "experiment_name": "protocol_fallbacks",
        "category": "regression",
        "protocol": {
            "normalize": True,
            "val_ratio": 0.1,
            "loss": "MSELoss",
            "forecast_multiplier": 5,
            "batch_size": 256,
        },
        "training": {"max_epochs": 3},
    }

    resolved = ryf.resolve_config(
        {"name": "cfg", "stacks": "Generic:1"},
        top_level_cfg,
    )

    assert resolved["protocol"]["normalize"] is True
    assert resolved["protocol"]["val_ratio"] == pytest.approx(0.1)
    assert resolved["training"]["loss"] == "MSELoss"
    assert resolved["training"]["forecast_multiplier"] == 5
    assert resolved["training"]["batch_size"] == 256


def test_run_experiment_forwards_protocol_settings(monkeypatch, tmp_path):
    fake_dataset = SimpleNamespace(
        name="Traffic-96",
        get_training_series=lambda: [[1.0, 2.0, 3.0]],
    )
    captured = {}

    def fake_load_dataset(dataset_name, period, train_ratio=None, include_target=None):
        captured["load_dataset"] = {
            "dataset_name": dataset_name,
            "period": period,
            "train_ratio": train_ratio,
            "include_target": include_target,
        }
        return fake_dataset

    def fake_run_single_experiment(**kwargs):
        captured["run_single_experiment"] = kwargs

    monkeypatch.setattr(ryf, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(ryf, "run_single_experiment", fake_run_single_experiment)
    monkeypatch.setattr(ryf, "result_exists", lambda *args, **kwargs: False)

    top_level_cfg = {
        "experiment_name": "protocol_launcher",
        "dataset": "traffic",
        "periods": ["Traffic-96"],
        "stacks": "Generic:1",
        "protocol": {
            "normalize": True,
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "include_target": True,
            "loss": "MSELoss",
            "forecast_multiplier": 5,
            "batch_size": 256,
        },
        "training": {"max_epochs": 1, "patience": 1},
        "runs": {"n_runs": 1, "base_seed": 7},
        "output": {"results_dir": str(tmp_path), "save_predictions": False},
    }

    ryf.run_experiment(top_level_cfg, cli_overrides={}, dry_run=False, analyze_only=False)

    assert captured["load_dataset"] == {
        "dataset_name": "traffic",
        "period": "Traffic-96",
        "train_ratio": pytest.approx(0.7),
        "include_target": True,
    }
    assert captured["run_single_experiment"]["normalize"] is True
    assert captured["run_single_experiment"]["val_ratio"] == pytest.approx(0.1)
    assert captured["run_single_experiment"]["loss_override"] == "MSELoss"
    assert captured["run_single_experiment"]["forecast_multiplier"] == 5
    assert captured["run_single_experiment"]["batch_size"] == rub.BATCH_SIZES[("traffic", "Traffic-96")]


@pytest.mark.parametrize(
    "dataset_name,period,module_attr,expected_horizon",
    [
        ("traffic", "Traffic-96", "TrafficDataset", 96),
        ("weather", "Weather-96", "WeatherDataset", 96),
    ],
)
def test_load_dataset_forwards_dataset_protocol_kwargs(
    monkeypatch,
    dataset_name,
    period,
    module_attr,
    expected_horizon,
):
    captured = {}

    def fake_dataset_ctor(horizon, train_ratio=0.8, include_target=False):
        captured.update(
            horizon=horizon,
            train_ratio=train_ratio,
            include_target=include_target,
        )
        return SimpleNamespace(name=f"{dataset_name}-{horizon}")

    monkeypatch.setattr(rub, module_attr, fake_dataset_ctor)

    dataset = rub.load_dataset(
        dataset_name,
        period,
        train_ratio=0.7,
        include_target=True,
    )

    assert dataset.name == f"{dataset_name}-{expected_horizon}"
    assert captured == {
        "horizon": expected_horizon,
        "train_ratio": pytest.approx(0.7),
        "include_target": True,
    }