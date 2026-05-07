"""Tests for experiments/run_trendwaveletae_study.py core planning logic."""

import os
import sys

import pytest


_EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

import run_trendwaveletae_study as study  # noqa: E402


def _build_cfg(dataset: str, period: str, repeats: int,
               block_type: str = "TrendWaveletAE") -> study.StudyConfig:
    return study.StudyConfig(
        path=f"/tmp/{dataset}.yaml",
        dataset=dataset,
        period=period,
        architecture={"block_type": block_type, "repeats": repeats},
        training={
            "active_g": False,
            "sum_losses": False,
            "activation": "ReLU",
            "n_blocks_per_stack": 1,
            "share_weights": True,
            "loss": "SMAPELoss",
            "optimizer": "Adam",
            "learning_rate": 0.001,
        },
        lr_scheduler={"warmup_epochs": 15, "eta_min": 1e-6},
        search_space={
            "wavelet_types": list(study.WAVELET_TYPES),
            "basis_labels": ["eq_fcast", "lt_fcast", "eq_bcast", "lt_bcast"],
            "trend_dims": [3, 5],
            "latent_dims": [2, 5, 8],
        },
        search_rounds=[
            {"max_epochs": 10, "n_runs": 3, "keep_fraction": 0.67},
            {"max_epochs": 15, "n_runs": 3, "keep_fraction": 0.67},
            {"max_epochs": 50, "n_runs": 3, "top_k": 10},
        ],
        output={
            "results_dir": "experiments/results",
            "search_csv_filename": "trendwaveletae_pure_study_results.csv",
            "cross_csv_path": "experiments/results/trendwaveletae_pure_cross_dataset_results.csv",
        },
        hardware={"accelerator": "auto", "num_workers": 0},
        runs={"base_seed": 42},
        meta_forecaster={"enabled": False},
    )


@pytest.mark.parametrize(
    "dataset,period,repeats,expected_stack_len",
    [
        ("m4", "Yearly", 10, 10),
        ("tourism", "Tourism-Yearly", 10, 10),
        ("traffic", "Traffic-96", 20, 20),
        ("weather", "Weather-96", 20, 20),
    ],
)
def test_generate_search_configs_count_and_stack_len(dataset, period, repeats, expected_stack_len):
    cfg = _build_cfg(dataset, period, repeats)
    configs = study.generate_search_configs(cfg)

    # 14 wavelet_types * 4 basis_labels * 2 trend_dims * 3 latent_dims = 336
    assert len(configs) == 336

    sample = next(iter(configs.values()))
    assert len(sample["stack_types"]) == expected_stack_len
    # All stacks should be the same block type (pure stack)
    assert all(s == "TrendWaveletAE" for s in sample["stack_types"])


def test_pure_stack_configs_have_wavelet_type_field():
    """Each config must have a wavelet_type string (not encoded in block name)."""
    cfg = _build_cfg("m4", "Yearly", 10)
    configs = study.generate_search_configs(cfg)

    for config_name, config in configs.items():
        assert "wavelet_type" in config
        assert config["wavelet_type"] in study.WAVELET_TYPES
        # Block type should be constant, not wavelet-specific
        assert config["block_type"] == "TrendWaveletAE"


def test_canonical_config_id_format():
    """canonical_config_id should include block_type, wavelet_type, bd_label, td, ld."""
    cid = study.canonical_config_id("TrendWaveletAE", "haar", "eq_fcast", 3, 5)
    assert cid == "TrendWaveletAE|haar|eq_fcast|td3|ld5"


def test_config_name_from_canonical():
    cid = "TrendWaveletAE|haar|eq_fcast|td3|ld5"
    name = study.config_name_from_canonical(cid)
    assert name == "TrendWaveletAE_haar_eq_fcast_td3_ld5"


def test_parse_canonical():
    cid = "TrendWaveletAELG|db3|lt_bcast|td5|ld8"
    parsed = study._parse_canonical(cid)
    assert parsed["block_type"] == "TrendWaveletAELG"
    assert parsed["wavelet_type"] == "db3"
    assert parsed["bd_label"] == "lt_bcast"
    assert parsed["trend_dim"] == 5
    assert parsed["latent_dim"] == 8


def test_basis_label_collisions_not_deduplicated_for_traffic96():
    cfg = _build_cfg("traffic", "Traffic-96", repeats=20)
    configs = study.generate_search_configs(cfg)

    labels = {c["bd_label"] for c in configs.values()}
    assert labels == {"eq_fcast", "lt_fcast", "eq_bcast", "lt_bcast"}

    by_label = {}
    for label in labels:
        by_label[label] = sum(1 for c in configs.values() if c["bd_label"] == label)
    assert by_label == {
        "eq_fcast": 84,
        "lt_fcast": 84,
        "eq_bcast": 84,
        "lt_bcast": 84,
    }


def test_compute_keep_n_uses_floor_rule():
    assert study.compute_keep_n(336, 0.67) == 225
    assert study.compute_keep_n(224, 0.67) == 150
    assert study.compute_keep_n(150, 0.67) == 100


def test_trendwaveletaelg_config_generation():
    """TrendWaveletAELG pure stack configs should work identically."""
    cfg = _build_cfg("m4", "Yearly", 10, block_type="TrendWaveletAELG")
    configs = study.generate_search_configs(cfg)

    assert len(configs) == 336

    sample = next(iter(configs.values()))
    assert all(s == "TrendWaveletAELG" for s in sample["stack_types"])
    assert sample["block_type"] == "TrendWaveletAELG"


def test_generate_search_configs_list_block_type():
    """block_type as a list should produce configs for each type."""
    cfg = study.StudyConfig(
        path="/tmp/v2.yaml",
        dataset="m4",
        period="Yearly",
        architecture={"block_type": ["TrendWaveletAE", "TrendWaveletAELG"], "repeats": 10},
        training={
            "active_g": False,
            "sum_losses": False,
            "activation": "ReLU",
            "n_blocks_per_stack": 1,
            "share_weights": True,
            "loss": "SMAPELoss",
            "optimizer": "Adam",
            "learning_rate": 0.001,
        },
        lr_scheduler={"warmup_epochs": 15, "eta_min": 1e-6},
        search_space={
            "wavelet_types": ["haar", "db3", "db20", "coif2", "sym10"],
            "basis_labels": ["eq_fcast", "lt_fcast"],
            "trend_dims": [3],
            "latent_dims": [8, 12],
        },
        search_rounds=[
            {"max_epochs": 15, "n_runs": 3, "keep_fraction": 0.50},
            {"max_epochs": 50, "n_runs": 5, "top_k": 10},
        ],
        output={
            "results_dir": "experiments/results",
            "search_csv_filename": "trendwaveletae_v2_study_results.csv",
            "cross_csv_path": "experiments/results/trendwaveletae_v2_cross_dataset_results.csv",
        },
        hardware={"accelerator": "auto", "num_workers": 0},
        runs={"base_seed": 42},
        meta_forecaster={"enabled": False},
    )
    configs = study.generate_search_configs(cfg)

    # 2 block_types * 5 wavelets * 2 basis_labels * 1 trend_dim * 2 latent_dims = 40
    assert len(configs) == 40

    ae_configs = [c for c in configs.values() if c["block_type"] == "TrendWaveletAE"]
    lg_configs = [c for c in configs.values() if c["block_type"] == "TrendWaveletAELG"]
    assert len(ae_configs) == 20
    assert len(lg_configs) == 20

    # Verify stack types match block type
    for config in ae_configs:
        assert all(s == "TrendWaveletAE" for s in config["stack_types"])
    for config in lg_configs:
        assert all(s == "TrendWaveletAELG" for s in config["stack_types"])


def test_generate_search_configs_scalar_block_type_backward_compat():
    """Scalar block_type should still work (backward compatibility)."""
    cfg = _build_cfg("m4", "Yearly", 10, block_type="TrendWaveletAE")
    configs = study.generate_search_configs(cfg)

    # 14 wavelet_types * 4 basis_labels * 2 trend_dims * 3 latent_dims = 336
    assert len(configs) == 336

    for config in configs.values():
        assert config["block_type"] == "TrendWaveletAE"


def test_compute_keep_n_v2_50pct():
    assert study.compute_keep_n(40, 0.50) == 20


def test_select_global_top10_penalizes_missing():
    ds_rankings = {
        "m4": (
            [("A", 0.1), ("B", 0.2), ("C", 0.3)],
            {"A": 0.0, "B": 0.5, "C": 1.0},
            {"A", "B"},
        ),
        "tourism": (
            [("B", 0.1), ("C", 0.2), ("D", 0.3)],
            {"B": 0.0, "C": 0.5, "D": 1.0},
            {"B", "C"},
        ),
        "traffic": (
            [("A", 0.1), ("D", 0.2), ("E", 0.3)],
            {"A": 0.0, "D": 0.5, "E": 1.0},
            {"A", "D"},
        ),
        "weather": (
            [("A", 0.1), ("B", 0.2), ("E", 0.3)],
            {"A": 0.0, "B": 0.5, "E": 1.0},
            {"A", "B"},
        ),
    }

    top = study.select_global_top10(ds_rankings)
    assert top

    # A appears in 3 dataset top10 sets and has strong percentiles.
    assert top[0]["canonical_config_id"] == "A"
    assert top[0]["source_datasets"] == ["m4", "traffic", "weather"]
