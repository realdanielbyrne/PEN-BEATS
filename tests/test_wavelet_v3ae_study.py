"""Tests for experiments/run_wavelet_v3ae_study.py core planning logic."""

import os
import sys

import pytest


_EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

import run_wavelet_v3ae_study as study  # noqa: E402


def _build_cfg(dataset: str, period: str, repeats: int) -> study.StudyConfig:
    return study.StudyConfig(
        path=f"/tmp/{dataset}.yaml",
        dataset=dataset,
        period=period,
        architecture={"trend_block": "TrendAE", "repeats": repeats},
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
            "wavelets": list(study.WAVELET_V3AE_FAMILIES),
            "basis_labels": ["eq_fcast", "lt_fcast", "eq_bcast", "lt_bcast"],
            "trend_thetas_dims": [3, 5],
            "latent_dims": [2, 5, 8],
        },
        search_rounds=[
            {"max_epochs": 10, "n_runs": 3, "keep_fraction": 0.67},
            {"max_epochs": 15, "n_runs": 3, "keep_fraction": 0.67},
            {"max_epochs": 50, "n_runs": 3, "top_k": 10},
        ],
        output={
            "results_dir": "experiments/results",
            "search_csv_filename": "wavelet_v3ae_study_results.csv",
            "cross_csv_path": "experiments/results/wavelet_v3ae_cross_dataset_results.csv",
        },
        hardware={"accelerator": "auto", "num_workers": 0},
        runs={"base_seed": 42},
        meta_forecaster={"enabled": False},
    )


@pytest.mark.parametrize(
    "dataset,period,repeats,expected_stack_len",
    [
        ("m4", "Yearly", 5, 10),
        ("tourism", "Tourism-Yearly", 5, 10),
        ("traffic", "Traffic-96", 10, 20),
        ("weather", "Weather-96", 10, 20),
    ],
)
def test_generate_search_configs_count_and_stack_len(dataset, period, repeats, expected_stack_len):
    cfg = _build_cfg(dataset, period, repeats)
    configs = study.generate_search_configs(cfg)

    assert len(configs) == 336

    sample = next(iter(configs.values()))
    assert len(sample["stack_types"]) == expected_stack_len


def test_basis_label_collisions_not_deduplicated_for_traffic96():
    cfg = _build_cfg("traffic", "Traffic-96", repeats=10)
    configs = study.generate_search_configs(cfg)

    # Traffic-96 has forecast=96, backcast=192, so eq_fcast and lt_bcast both map to 96.
    # We still keep all 4 labels in the grid (no dedupe), preserving 336 configs.
    labels = {c["bd_label"] for c in configs.values()}
    assert labels == {"eq_fcast", "lt_fcast", "eq_bcast", "lt_bcast"}

    # Every wavelet x ttd x latent should appear for each label.
    # 14 wavelets * 2 trend dims * 3 latent dims = 84 entries per label.
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


def test_select_global_top10_penalizes_missing_with_one_point_zero_percentile():
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

    # Candidate C is missing from m4/traffic/weather percentile maps and should be penalized.
    c_row = next(r for r in top if r["canonical_config_id"] == "C")
    assert c_row["mean_percentile"] >= 0.625


# --- V3AELG-specific tests ---

WAVELET_V3AELG_FAMILIES = [
    "HaarWaveletV3AELG", "DB2WaveletV3AELG", "DB3WaveletV3AELG", "DB4WaveletV3AELG",
    "DB10WaveletV3AELG", "DB20WaveletV3AELG",
    "Coif1WaveletV3AELG", "Coif2WaveletV3AELG", "Coif3WaveletV3AELG", "Coif10WaveletV3AELG",
    "Symlet2WaveletV3AELG", "Symlet3WaveletV3AELG", "Symlet10WaveletV3AELG", "Symlet20WaveletV3AELG",
]


@pytest.mark.parametrize("wavelet_family,expected", [
    ("HaarWaveletV3AELG", "Haar"),
    ("DB2WaveletV3AELG", "DB2"),
    ("Symlet20WaveletV3AELG", "Symlet20"),
    ("Coif10WaveletV3AELG", "Coif10"),
    # V3AE should still work
    ("HaarWaveletV3AE", "Haar"),
    ("DB3WaveletV3AE", "DB3"),
])
def test_wavelet_short_name_v3aelg(wavelet_family, expected):
    assert study.wavelet_short_name(wavelet_family) == expected


def _build_cfg_v3aelg(dataset: str, period: str, repeats: int,
                       trend_block: str = "TrendAELG") -> study.StudyConfig:
    return study.StudyConfig(
        path=f"/tmp/{dataset}_v3aelg.yaml",
        dataset=dataset,
        period=period,
        architecture={"trend_block": trend_block, "repeats": repeats},
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
            "wavelets": WAVELET_V3AELG_FAMILIES,
            "basis_labels": ["eq_fcast", "lt_fcast", "eq_bcast", "lt_bcast"],
            "trend_thetas_dims": [3, 5],
            "latent_dims": [2, 5, 8],
        },
        search_rounds=[
            {"max_epochs": 10, "n_runs": 3, "keep_fraction": 0.67},
            {"max_epochs": 15, "n_runs": 3, "keep_fraction": 0.67},
            {"max_epochs": 50, "n_runs": 3, "top_k": 10},
        ],
        output={
            "results_dir": "experiments/results",
            "search_csv_filename": "wavelet_v3aelg_study_results.csv",
            "cross_csv_path": "experiments/results/wavelet_v3aelg_cross_dataset_results.csv",
        },
        hardware={"accelerator": "auto", "num_workers": 0},
        runs={"base_seed": 42},
        meta_forecaster={"enabled": False},
    )


@pytest.mark.parametrize(
    "dataset,period,repeats",
    [
        ("m4", "Yearly", 5),
        ("traffic", "Traffic-96", 10),
    ],
)
def test_v3aelg_generate_search_configs_count(dataset, period, repeats):
    cfg = _build_cfg_v3aelg(dataset, period, repeats)
    configs = study.generate_search_configs(cfg)
    # 14 wavelets * 4 basis_labels * 2 trend_thetas_dims * 3 latent_dims = 336
    assert len(configs) == 336


def test_v3aelg_stack_types_contain_trendaelg():
    cfg = _build_cfg_v3aelg("m4", "Yearly", repeats=5, trend_block="TrendAELG")
    configs = study.generate_search_configs(cfg)
    sample = next(iter(configs.values()))
    assert "TrendAELG" in sample["stack_types"]


def test_v3aelg_stack_types_contain_trendae():
    cfg = _build_cfg_v3aelg("m4", "Yearly", repeats=5, trend_block="TrendAE")
    configs = study.generate_search_configs(cfg)
    sample = next(iter(configs.values()))
    assert "TrendAE" in sample["stack_types"]
