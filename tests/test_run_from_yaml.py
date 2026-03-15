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
        "block_params": {"latent_gate_fn": "wavy_sigmoid"},
        "runs": {"n_runs": 1, "base_seed": 7},
        "output": {"results_dir": str(tmp_path), "save_predictions": False},
        "hardware": {"worker_id": "yaml-worker-1"},
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
    assert captured["run_single_experiment"]["latent_gate_fn"] == "wavy_sigmoid"
    assert captured["run_single_experiment"]["dataset_name"] == "traffic"
    assert captured["run_single_experiment"]["worker_id"] == "yaml-worker-1"


def test_claim_job_prevents_duplicate_work_per_pass(monkeypatch, tmp_path):
    claims_dir = tmp_path / ".claims"
    monkeypatch.setattr(rub, "CLAIMS_DIR", str(claims_dir))

    baseline_claim_name = rub.build_claim_config_name("baseline", "TWG_cfg")
    active_g_claim_name = rub.build_claim_config_name("activeG_fcast", "TWG_cfg")

    assert rub.claim_job(baseline_claim_name, "m4", "Yearly", 0, worker_id="worker-a")
    assert not rub.claim_job(baseline_claim_name, "m4", "Yearly", 0, worker_id="worker-b")
    assert rub.claim_job(active_g_claim_name, "m4", "Yearly", 0, worker_id="worker-c")

    assert (claims_dir / "baseline__TWG_cfg__m4__Yearly__run0.claim").exists()
    assert (claims_dir / "activeG_fcast__TWG_cfg__m4__Yearly__run0.claim").exists()


def test_run_single_experiment_claims_with_pass_aware_identity(monkeypatch):
    captured = {}

    monkeypatch.setattr(rub, "result_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        rub,
        "claim_job",
        lambda config_name, dataset_name, horizon_label, run_idx, worker_id="": (
            captured.update(
                config_name=config_name,
                dataset_name=dataset_name,
                horizon_label=horizon_label,
                run_idx=run_idx,
                worker_id=worker_id,
            )
            or False
        ),
    )
    monkeypatch.setattr(
        rub,
        "set_seed",
        lambda seed: pytest.fail("set_seed should not run when claim acquisition fails"),
    )

    rub.run_single_experiment(
        experiment_name="baseline",
        config_name="TWG_cfg",
        category="regression",
        stack_types=["Generic"],
        period="Yearly",
        run_idx=2,
        dataset=SimpleNamespace(name="M4-Yearly"),
        train_series_list=[],
        csv_path="unused.csv",
        dataset_name="m4",
        worker_id="yaml-worker-2",
    )

    assert captured == {
        "config_name": "baseline__TWG_cfg",
        "dataset_name": "m4",
        "horizon_label": "Yearly",
        "run_idx": 2,
        "worker_id": "yaml-worker-2",
    }


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
