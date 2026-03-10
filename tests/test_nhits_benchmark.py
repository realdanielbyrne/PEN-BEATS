"""Tests for run_nhits_benchmark.py YAML config loading and parse_stack_types."""

import os
import sys
import tempfile

import pytest
import yaml

_EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

import run_nhits_benchmark as rnb  # noqa: E402


# ---------------------------------------------------------------------------
# parse_stack_types
# ---------------------------------------------------------------------------

class TestParseStackTypes:
    """Tests for the parse_stack_types helper."""

    def test_list_passthrough(self):
        """A plain list should be returned as-is."""
        result = rnb.parse_stack_types(["Generic", "Generic", "Generic"])
        assert result == ["Generic", "Generic", "Generic"]

    def test_string_expression_single(self):
        """'[\"Generic\"] * 10' should produce 10 Generic entries."""
        result = rnb.parse_stack_types("['Generic'] * 10")
        assert result == ["Generic"] * 10

    def test_string_expression_multi(self):
        """'[\"TrendAELG\", \"Sym20V3AELG\"] * 5' should alternate 5 times."""
        result = rnb.parse_stack_types("['TrendAELG', 'Sym20V3AELG'] * 5")
        assert len(result) == 10
        assert result == ["TrendAELG", "Sym20V3AELG"] * 5

    def test_plain_list_string(self):
        """'[\"A\", \"B\", \"C\"]' should parse as a list."""
        result = rnb.parse_stack_types("['A', 'B', 'C']")
        assert result == ["A", "B", "C"]

    def test_invalid_expression_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            rnb.parse_stack_types("not a valid expression")

    def test_non_string_non_list_raises(self):
        with pytest.raises(ValueError, match="must be a list or string"):
            rnb.parse_stack_types(42)


# ---------------------------------------------------------------------------
# load_yaml_configs
# ---------------------------------------------------------------------------

class TestLoadYamlConfigs:
    """Tests for load_yaml_configs function."""

    @pytest.fixture
    def weather_yaml_path(self):
        return os.path.join(
            os.path.dirname(__file__), "..", "experiments", "configs",
            "nhits_benchmark_weather.yaml"
        )

    @pytest.fixture
    def traffic_yaml_path(self):
        return os.path.join(
            os.path.dirname(__file__), "..", "experiments", "configs",
            "nhits_benchmark_traffic.yaml"
        )

    def test_loads_weather_yaml(self, weather_yaml_path):
        """Should load all configs and protocol from weather YAML."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        assert "configs" in result
        assert "protocol" in result
        assert "training" in result
        assert "block_params" in result
        assert "runs" in result
        assert result["dataset"] == "weather"
        assert result["horizons"] == [96, 192, 336, 720]

    def test_weather_protocol_values(self, weather_yaml_path):
        """Protocol values should match the YAML file."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        p = result["protocol"]
        assert p["train_ratio"] == 0.7
        assert p["val_ratio"] == 0.1
        assert p["loss"] == "MSELoss"
        assert p["forecast_multiplier"] == 5
        assert p["batch_size"] == 256
        assert p["normalize"] is True

    def test_weather_configs_count(self, weather_yaml_path):
        """Weather YAML should have 18 configs."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        assert len(result["configs"]) == 18

    def test_traffic_normalize_false(self, traffic_yaml_path):
        """Traffic YAML should have normalize=false."""
        result = rnb.load_yaml_configs(traffic_yaml_path)
        assert result["protocol"]["normalize"] is False

    def test_stack_types_parsed_correctly(self, weather_yaml_path):
        """Stack types expressions should be expanded to lists."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        configs = result["configs"]

        # Generic-10 should have 10 Generic entries
        generic10 = next(c for c in configs if c["config_name"] == "Generic-10")
        assert generic10["stack_types"] == ["Generic"] * 10

        # TrendAELG+Sym20V3AELG should alternate 5 times = 10 entries
        trend_sym = next(c for c in configs if c["config_name"] == "TrendAELG+Sym20V3AELG")
        assert len(trend_sym["stack_types"]) == 10
        assert trend_sym["stack_types"][0] == "TrendAELG"
        assert trend_sym["stack_types"][1] == "Symlet20WaveletV3AELG"

    def test_nhits_configs_have_pooling_params(self, weather_yaml_path):
        """NHiTSNet configs should have n_pools_kernel_size and n_freq_downsample."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        configs = result["configs"]

        nhits_generic = next(c for c in configs if c["config_name"] == "NHiTS-Generic")
        assert nhits_generic["model_type"] == "NHiTSNet"
        assert nhits_generic["n_pools_kernel_size"] == [8, 4, 1]
        assert nhits_generic["n_freq_downsample"] == [24, 12, 1]

    def test_training_params(self, weather_yaml_path):
        result = rnb.load_yaml_configs(weather_yaml_path)
        t = result["training"]
        assert t["max_epochs"] == 100
        assert t["patience"] == 10
        assert t["activation"] == "ReLU"
        assert t["active_g"] is False
        assert t["sum_losses"] is False

    def test_block_params(self, weather_yaml_path):
        result = rnb.load_yaml_configs(weather_yaml_path)
        bp = result["block_params"]
        assert bp["thetas_dim"] == 5
        assert bp["latent_dim"] == 16
        assert bp["basis_dim"] == "eq_fcast"

    def test_runs_params(self, weather_yaml_path):
        result = rnb.load_yaml_configs(weather_yaml_path)
        r = result["runs"]
        assert r["n_runs"] == 8
        assert r["base_seed"] == 1

