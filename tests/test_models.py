"""Tests for NBeatsNet and NHiTSNet models — width selection, optimizer dispatch, NHiTS semantics."""
import pytest
import torch
from torch import optim

from lightningnbeats.models import NBeatsNet, NHiTSNet


def _make_model(stack_types, **kwargs):
    """Helper to create a minimal NBeatsNet model."""
    defaults = dict(
        backcast_length=20,
        forecast_length=5,
        stack_types=stack_types,
        n_blocks_per_stack=1,
        share_weights=False,
        thetas_dim=4,
        active_g=False,
        latent_dim=4,
        basis_dim=16,
    )
    defaults.update(kwargs)
    return NBeatsNet(**defaults)


# --- Width selection tests ---

class TestWidthSelection:
    """Verify each block type uses the correct width parameter."""

    def test_generic_uses_g_width(self):
        model = _make_model(["Generic"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_generic_ae_uses_g_width(self):
        model = _make_model(["GenericAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_generic_ae_backcast_uses_g_width(self):
        model = _make_model(["GenericAEBackcast"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_generic_ae_backcast_ae_uses_g_width(self):
        model = _make_model(["GenericAEBackcastAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_seasonality_uses_s_width(self):
        model = _make_model(["Seasonality"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 128

    def test_seasonality_ae_uses_s_width(self):
        model = _make_model(["SeasonalityAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 128

    def test_trend_uses_t_width(self):
        model = _make_model(["Trend"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 32

    def test_trend_ae_uses_t_width(self):
        model = _make_model(["TrendAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 32

    def test_autoencoder_uses_ae_width(self):
        model = _make_model(["AutoEncoder"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 96

    def test_autoencoder_ae_uses_ae_width(self):
        model = _make_model(["AutoEncoderAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 96

    def test_wavelet_uses_g_width_fallback(self):
        model = _make_model(["HaarWaveletV3"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64


# --- Optimizer dispatch tests ---

class TestOptimizerDispatch:
    """Verify optimizer_name parameter is respected."""

    def test_adam_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="Adam")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.Adam)

    def test_sgd_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="SGD")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.SGD)

    def test_adamw_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="AdamW")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.AdamW)

    def test_rmsprop_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="RMSprop")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.RMSprop)

    def test_invalid_optimizer_raises(self):
        model = _make_model(["Generic"])
        model.optimizer_name = "InvalidOptimizer"
        with pytest.raises(ValueError, match="Unknown optimizer name"):
            model.configure_optimizers()


# --- Forward pass shape tests ---

class TestForwardPass:
    """Verify model forward pass produces correct output shapes."""

    def test_generic_forward_shape(self):
        model = _make_model(["Generic"], g_width=32)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_trend_seasonality_forward_shape(self):
        model = _make_model(["Trend", "Seasonality"], t_width=32, s_width=64)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_invalid_stack_type_raises(self):
        with pytest.raises(ValueError, match="Stack architecture must be specified"):
            NBeatsNet(backcast_length=20, forecast_length=5, stack_types=None)


# --- Model defaults tests ---

class TestModelDefaults:
    """Verify NBeatsNet default parameter values."""

    def test_active_g_defaults_to_false(self):
        model = NBeatsNet(
            backcast_length=20,
            forecast_length=5,
            stack_types=["Generic"],
            n_blocks_per_stack=1
        )
        assert model.active_g is False

    def test_learning_rate_defaults_to_1e_minus_3(self):
        model = NBeatsNet(
            backcast_length=20,
            forecast_length=5,
            stack_types=["Generic"],
            n_blocks_per_stack=1
        )
        assert model.learning_rate == 1e-3


# --- sum_losses semantic fix tests ---

class TestSumLossesBehavior:
    """Verify sum_losses uses zero target for backcast loss."""

    def test_sum_losses_backcast_uses_zero_target_in_training(self):
        model = _make_model(["Generic"], sum_losses=True, g_width=32)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.training_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_sum_losses_backcast_uses_zero_target_in_validation(self):
        model = _make_model(["Generic"], sum_losses=True, g_width=32, no_val=False)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.validation_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_sum_losses_backcast_uses_zero_target_in_test(self):
        model = _make_model(["Generic"], sum_losses=True, g_width=32)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.test_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)


# --- NormalizedDeviationLoss configuration tests ---

class TestNormalizedDeviationLossConfiguration:
    """Verify NormalizedDeviationLoss is handled in configure_loss."""

    def test_model_with_normalized_deviation_loss(self):
        from lightningnbeats.losses import NormalizedDeviationLoss
        model = _make_model(["Generic"], loss="NormalizedDeviationLoss", g_width=32)
        assert isinstance(model.loss_fn, NormalizedDeviationLoss)

    def test_normalized_deviation_loss_forward_pass(self):
        model = _make_model(["Generic"], loss="NormalizedDeviationLoss", g_width=32)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.training_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)


# --- BottleneckGeneric width selection tests ---

class TestBottleneckGenericWidthSelection:
    """Verify BottleneckGeneric and BottleneckGenericAE use g_width."""

    def test_bottleneck_generic_uses_g_width(self):
        model = _make_model(["BottleneckGeneric"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_bottleneck_generic_ae_uses_g_width(self):
        model = _make_model(["BottleneckGenericAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64


# --- active_g split mode model-level tests ---

class TestActiveGSplitModesModel:
    """Verify string active_g modes are accepted and passed to blocks."""

    def test_active_g_backcast_accepted(self):
        model = _make_model(["Generic"], active_g='backcast', g_width=32)
        assert model.active_g == 'backcast'
        block = model.stacks[0][0]
        assert block.active_g == 'backcast'

    def test_active_g_forecast_accepted(self):
        model = _make_model(["Generic"], active_g='forecast', g_width=32)
        assert model.active_g == 'forecast'
        block = model.stacks[0][0]
        assert block.active_g == 'forecast'

    def test_active_g_true_still_works(self):
        model = _make_model(["Generic"], active_g=True, g_width=32)
        assert model.active_g is True
        block = model.stacks[0][0]
        assert block.active_g is True

    def test_active_g_false_still_works(self):
        model = _make_model(["Generic"], active_g=False, g_width=32)
        assert model.active_g is False

    def test_active_g_invalid_string_raises(self):
        with pytest.raises(ValueError, match="active_g must be"):
            _make_model(["Generic"], active_g='invalid')

    def test_active_g_backcast_forward_pass(self):
        model = _make_model(["Generic"], active_g='backcast', g_width=32)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_active_g_forecast_forward_pass(self):
        model = _make_model(["Generic"], active_g='forecast', g_width=32)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)


# --- forecast_basis_dim passthrough tests ---

class TestForecastBasisDim:
    """Verify forecast_basis_dim is passed through to V3 wavelet blocks."""

    def test_forecast_basis_dim_passed_to_v3_blocks(self):
        model = _make_model(["HaarWaveletV3"] * 2, basis_dim=16, forecast_basis_dim=4,
                            backcast_length=30, forecast_length=6)
        for stack in model.stacks:
            for block in stack:
                assert block.forecast_linear.out_features == min(4, 6)
                assert block.backcast_linear.out_features == min(16, 30)
        x = torch.randn(4, 30)
        _, forecast = model(x)
        assert forecast.shape == (4, 6)

    def test_forecast_basis_dim_none_uses_basis_dim(self):
        model = _make_model(["HaarWaveletV3"], basis_dim=16, forecast_basis_dim=None,
                            backcast_length=30, forecast_length=6)
        block = model.stacks[0][0]
        assert block.forecast_linear.out_features == min(16, 6)
        assert block.backcast_linear.out_features == min(16, 30)



# --- trend_thetas_dim tests ---

class TestTrendThetasDim:
    """Verify trend_thetas_dim routes only to Trend/TrendAE blocks."""

    def test_trend_thetas_dim_overrides_trend_block(self):
        """Trend block should use trend_thetas_dim instead of global thetas_dim."""
        model = _make_model(["Trend"], t_width=32, thetas_dim=8, trend_thetas_dim=3)
        block = model.stacks[0][0]
        # Trend projects to thetas_dim coefficients
        assert block.backcast_linear.out_features == 3
        assert block.forecast_linear.out_features == 3

    def test_trend_thetas_dim_overrides_trend_ae_block(self):
        """TrendAE block should use trend_thetas_dim instead of global thetas_dim."""
        model = _make_model(["TrendAE"], t_width=32, thetas_dim=8, trend_thetas_dim=2, latent_dim=4)
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 2
        assert block.forecast_linear.out_features == 2

    def test_trend_thetas_dim_does_not_affect_bottleneck_generic(self):
        """BottleneckGeneric block should still use global thetas_dim."""
        model = _make_model(["BottleneckGeneric"], g_width=32, thetas_dim=8, trend_thetas_dim=3)
        block = model.stacks[0][0]
        # BottleneckGeneric uses thetas_dim as its bottleneck
        assert block.backcast_linear.out_features == 8
        assert block.forecast_linear.out_features == 8

    def test_trend_thetas_dim_defaults_to_3(self):
        """When trend_thetas_dim is not specified, Trend uses default of 3."""
        model = _make_model(["Trend"], t_width=32, thetas_dim=8)
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 3

    def test_trend_thetas_dim_none_falls_back_to_global_thetas_dim(self):
        """trend_thetas_dim=None should use global thetas_dim for Trend blocks."""
        model = _make_model(["Trend"], t_width=32, thetas_dim=8, trend_thetas_dim=None)
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 8
        assert block.forecast_linear.out_features == 8

    def test_trend_thetas_dim_invalid_raises(self):
        """trend_thetas_dim must be a positive integer."""
        with pytest.raises(ValueError, match="trend_thetas_dim must be a positive integer"):
            _make_model(["Trend"], t_width=32, trend_thetas_dim=0)
        with pytest.raises(ValueError, match="trend_thetas_dim must be a positive integer"):
            _make_model(["Trend"], t_width=32, trend_thetas_dim=-1)

    def test_trend_thetas_dim_accepts_any_positive_int(self):
        """trend_thetas_dim=5 should work as a valid positive integer."""
        model = _make_model(["Trend"], t_width=32, trend_thetas_dim=5)
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 5

    def test_mixed_stack_routing(self):
        """In a Trend+BottleneckGeneric stack, each gets the correct thetas_dim."""
        model = _make_model(["Trend", "BottleneckGeneric"], t_width=32, g_width=32,
                            thetas_dim=7, trend_thetas_dim=3)
        trend_block = model.stacks[0][0]
        bg_block = model.stacks[1][0]
        assert trend_block.backcast_linear.out_features == 3
        assert bg_block.backcast_linear.out_features == 7

    def test_trend_thetas_dim_forward_shape(self):
        """Forward pass works correctly with trend_thetas_dim set."""
        model = _make_model(["Trend", "Generic"], t_width=32, g_width=32,
                            thetas_dim=7, trend_thetas_dim=3)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)


# --- wavelet_type forwarding tests ---

class TestWaveletTypeForwarding:
    """Verify wavelet_type is forwarded to TrendWaveletAE/TrendWaveletAELG blocks."""

    def test_wavelet_type_forwarded_to_trendwaveletae(self):
        """TrendWaveletAE blocks should receive the wavelet_type parameter."""
        model = _make_model(["TrendWaveletAE"], g_width=64, wavelet_type='haar')
        block = model.stacks[0][0]
        assert block.wavelet_type == 'haar'

    def test_wavelet_type_forwarded_to_trendwaveletaelg(self):
        """TrendWaveletAELG blocks should receive the wavelet_type parameter."""
        model = _make_model(["TrendWaveletAELG"], g_width=64, wavelet_type='haar')
        block = model.stacks[0][0]
        assert block.wavelet_type == 'haar'

    def test_wavelet_type_default_is_db3(self):
        """Default wavelet_type should be 'db3'."""
        model = _make_model(["TrendWaveletAE"], g_width=64)
        block = model.stacks[0][0]
        assert block.wavelet_type == 'db3'

    def test_wavelet_type_forward_pass(self):
        """Forward pass works with non-default wavelet_type."""
        model = _make_model(["TrendWaveletAE"] * 2, g_width=64, wavelet_type='haar')
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)



# ---------------------------------------------------------------------------
# NHiTSNet tests
# ---------------------------------------------------------------------------

def _make_nhits(stack_types, **kwargs):
    """Helper to create a minimal NHiTSNet model."""
    defaults = dict(
        backcast_length=24,
        forecast_length=6,
        stack_types=stack_types,
        n_blocks_per_stack=1,
        share_weights=False,
        thetas_dim=4,
        active_g=False,
        latent_dim=4,
        basis_dim=16,
        g_width=32,
    )
    defaults.update(kwargs)
    return NHiTSNet(**defaults)


class TestNHiTSNetConstruction:
    """Verify NHiTSNet instantiates correctly and stores parameters."""

    def test_basic_construction(self):
        model = _make_nhits(["Generic", "Generic", "Generic"])
        assert model.n_stacks == 3
        assert len(model.stacks) == 3

    def test_stack_types_none_raises(self):
        with pytest.raises(ValueError, match="Stack architecture must be specified"):
            NHiTSNet(backcast_length=24, forecast_length=6, stack_types=None)

    def test_invalid_active_g_raises(self):
        with pytest.raises(ValueError, match="active_g must be"):
            _make_nhits(["Generic"], active_g='invalid')

    def test_invalid_trend_thetas_dim_raises(self):
        with pytest.raises(ValueError, match="trend_thetas_dim must be a positive integer"):
            _make_nhits(["Trend"], trend_thetas_dim=0)

    def test_n_pools_kernel_size_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_pools_kernel_size length"):
            _make_nhits(["Generic", "Generic"], n_pools_kernel_size=[2])

    def test_n_freq_downsample_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_freq_downsample length"):
            _make_nhits(["Generic", "Generic"], n_freq_downsample=[2])

    def test_default_pools_and_downsample_filled_in(self):
        model = _make_nhits(["Generic", "Generic"])
        assert model.n_pools_kernel_size == [2, 2]
        assert model.n_freq_downsample == [1, 1]

    def test_custom_pools_stored(self):
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            n_pools_kernel_size=[1, 2, 4],
            n_freq_downsample=[1, 2, 6],
        )
        assert model.n_pools_kernel_size == [1, 2, 4]
        assert model.n_freq_downsample == [1, 2, 6]

    def test_pooled_backcast_lengths_computed(self):
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            backcast_length=24,
            n_pools_kernel_size=[1, 2, 4],
        )
        # 24//1=24, 24//2=12, 24//4=6
        assert model.pooled_backcast_lengths == [24, 12, 6]

    def test_reduced_forecast_lengths_computed(self):
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            forecast_length=6,
            n_freq_downsample=[1, 2, 6],
        )
        # 6//1=6, 6//2=3, 6//6=1
        assert model.reduced_forecast_lengths == [6, 3, 1]

    def test_blocks_use_effective_forecast_length(self):
        """Each block's forecast output should match its reduced_forecast_length."""
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            backcast_length=24, forecast_length=6,
            n_pools_kernel_size=[1, 2, 4],
            n_freq_downsample=[1, 2, 6],
        )
        expected_fl = [6, 3, 1]
        for stack_id, stack in enumerate(model.stacks):
            block = stack[0]
            assert block.forecast_length == expected_fl[stack_id], (
                f"stack {stack_id}: expected forecast_length={expected_fl[stack_id]}, "
                f"got {block.forecast_length}"
            )

    def test_blocks_use_effective_backcast_length(self):
        """Each block's backcast length should match its pooled_backcast_length."""
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            backcast_length=24, forecast_length=6,
            n_pools_kernel_size=[1, 2, 4],
            n_freq_downsample=[1, 2, 6],
        )
        expected_bl = [24, 12, 6]
        for stack_id, stack in enumerate(model.stacks):
            block = stack[0]
            assert block.backcast_length == expected_bl[stack_id], (
                f"stack {stack_id}: expected backcast_length={expected_bl[stack_id]}, "
                f"got {block.backcast_length}"
            )


class TestNHiTSNetForwardShape:
    """Verify NHiTSNet forward pass produces correct output shapes."""

    def test_no_pooling_forward_shape(self):
        """With pool=1 and freq_down=1 the output is identical to N-BEATS."""
        model = _make_nhits(
            ["Generic", "Generic"],
            backcast_length=20, forecast_length=5,
            n_pools_kernel_size=[1, 1],
            n_freq_downsample=[1, 1],
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_multirate_forward_shape(self):
        """Multi-rate pooling + interpolation must still produce correct shapes."""
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            backcast_length=24, forecast_length=6,
            n_pools_kernel_size=[1, 2, 4],
            n_freq_downsample=[1, 2, 6],
        )
        x = torch.randn(4, 24)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 24)
        assert forecast.shape == (4, 6)

    def test_single_stack_forward_shape(self):
        model = _make_nhits(
            ["Generic"],
            backcast_length=16, forecast_length=4,
            n_pools_kernel_size=[2],
            n_freq_downsample=[2],
        )
        x = torch.randn(2, 16)
        backcast, forecast = model(x)
        assert backcast.shape == (2, 16)
        assert forecast.shape == (2, 4)

    def test_trend_seasonality_generic_forward_shape(self):
        model = _make_nhits(
            ["Trend", "Seasonality", "Generic"],
            backcast_length=24, forecast_length=6,
            t_width=32, s_width=64,
            n_pools_kernel_size=[1, 1, 2],
            n_freq_downsample=[1, 1, 2],
        )
        x = torch.randn(4, 24)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 24)
        assert forecast.shape == (4, 6)

    def test_ae_block_forward_shape(self):
        model = _make_nhits(
            ["GenericAE", "GenericAE"],
            backcast_length=20, forecast_length=5,
            n_pools_kernel_size=[1, 2],
            n_freq_downsample=[1, 5],
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)


class TestNHiTSNetParameters:
    """Verify NHiTS-specific parameters are stored and passed correctly."""

    def test_active_g_forecast_passthrough(self):
        model = _make_nhits(["Generic"], active_g='forecast')
        assert model.active_g == 'forecast'
        assert model.stacks[0][0].active_g == 'forecast'

    def test_active_g_backcast_passthrough(self):
        model = _make_nhits(["Generic"], active_g='backcast')
        assert model.active_g == 'backcast'
        assert model.stacks[0][0].active_g == 'backcast'

    def test_interpolation_mode_stored(self):
        model = _make_nhits(["Generic"], interpolation_mode='nearest')
        assert model.interpolation_mode == 'nearest'

    def test_nearest_interpolation_forward_shape(self):
        model = _make_nhits(
            ["Generic", "Generic"],
            backcast_length=20, forecast_length=4,
            n_pools_kernel_size=[1, 2],
            n_freq_downsample=[1, 2],
            interpolation_mode='nearest',
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 4)

    def test_sum_losses_training_step(self):
        model = _make_nhits(["Generic", "Generic"], sum_losses=True)
        x = torch.randn(4, 24)
        y = torch.randn(4, 6)
        loss = model.training_step((x, y), 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_optimizer_dispatch(self):
        model = _make_nhits(["Generic"], optimizer_name='AdamW')
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.AdamW)

    def test_trend_thetas_dim_routing(self):
        """Trend blocks in NHiTSNet should respect trend_thetas_dim."""
        model = _make_nhits(
            ["Trend"], t_width=32, thetas_dim=8, trend_thetas_dim=3,
            n_pools_kernel_size=[1], n_freq_downsample=[1],
        )
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 3
        assert block.forecast_linear.out_features == 3
