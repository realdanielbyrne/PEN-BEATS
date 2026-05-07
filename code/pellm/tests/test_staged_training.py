"""Tests for staged training workflow via from_pretrained_pe_llama.

Verifies that a PE model can be saved after stage 1 training, then
reloaded with a different PE config for stage 2 while preserving
already-fitted weights.
"""

import os
import tempfile
import unittest

import torch

from pellm.configuration_pe_llama import PELlamaConfig
from pellm.modeling_pe_llama import PELlamaForCausalLM


def _make_tiny_config(**overrides) -> PELlamaConfig:
    """Create a minimal PELlamaConfig for fast unit tests."""
    defaults = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=64,
        pe_attn_mode="standard",
        pe_mlp_mode="standard",
        ae_latent_dim=16,
        trend_dim=4,
        wavelet_dim=12,
        wavelet_type="db3",
    )
    defaults.update(overrides)
    return PELlamaConfig(**defaults)


class TestStagedTrainingAttnThenMLP(unittest.TestCase):
    """Stage 1: trend_wavelet attn + standard MLP → Stage 2: add AE MLP."""

    def test_attn_weights_preserved_after_mlp_swap(self):
        # Stage 1: build model with TrendWavelet attention only
        cfg1 = _make_tiny_config(pe_attn_mode="trend_wavelet", pe_mlp_mode="standard")
        model1 = PELlamaForCausalLM(cfg1)

        # Simulate training by mutating attention theta weights
        with torch.no_grad():
            for layer in model1.model.layers:
                layer.self_attn.q_proj.theta.weight.fill_(0.42)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            # Stage 2: reload with AE MLP mode
            cfg2 = _make_tiny_config(pe_attn_mode="trend_wavelet", pe_mlp_mode="ae_lg")
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        # Attention weights should be preserved exactly
        for layer in model2.model.layers:
            theta_w = layer.self_attn.q_proj.theta.weight.data
            self.assertTrue(torch.allclose(theta_w, torch.full_like(theta_w, 0.42)))

        # MLP should now be PEBottleneckMLPLG (has latent_gate)
        self.assertTrue(hasattr(model2.model.layers[0].mlp, "latent_gate"))

    def test_embedding_weights_preserved(self):
        cfg1 = _make_tiny_config(pe_attn_mode="trend_wavelet", pe_mlp_mode="standard")
        model1 = PELlamaForCausalLM(cfg1)

        # Snapshot embedding weight
        embed_snapshot = model1.model.embed_tokens.weight.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            cfg2 = _make_tiny_config(pe_attn_mode="trend_wavelet", pe_mlp_mode="ae")
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        torch.testing.assert_close(
            model2.model.embed_tokens.weight.data,
            embed_snapshot,
        )


class TestStagedTrainingMLPThenAttn(unittest.TestCase):
    """Stage 1: standard attn + AE MLP → Stage 2: add TrendWavelet attn."""

    def test_mlp_weights_preserved_after_attn_swap(self):
        cfg1 = _make_tiny_config(pe_attn_mode="standard", pe_mlp_mode="ae")
        model1 = PELlamaForCausalLM(cfg1)

        # Simulate training by mutating MLP fc1 weights
        with torch.no_grad():
            for layer in model1.model.layers:
                layer.mlp.fc1.weight.fill_(0.37)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            cfg2 = _make_tiny_config(pe_attn_mode="trend_wavelet", pe_mlp_mode="ae")
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        # MLP fc1 weights should be preserved exactly
        for layer in model2.model.layers:
            fc1_w = layer.mlp.fc1.weight.data
            self.assertTrue(torch.allclose(fc1_w, torch.full_like(fc1_w, 0.37)))

        # Attention should now be TrendWaveletLinear (has theta)
        self.assertTrue(hasattr(model2.model.layers[0].self_attn.q_proj, "theta"))

    def test_attn_projection_from_pelinear(self):
        """When swapping standard→trend_wavelet, PELinear weights should be
        projected via least-squares (non-random init)."""
        cfg1 = _make_tiny_config(pe_attn_mode="standard", pe_mlp_mode="standard")
        model1 = PELlamaForCausalLM(cfg1)

        # Set known attention weights
        with torch.no_grad():
            for layer in model1.model.layers:
                layer.self_attn.q_proj.weight.fill_(0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            cfg2 = _make_tiny_config(pe_attn_mode="trend_wavelet", pe_mlp_mode="standard")
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        # theta weights should be non-zero (projected, not random default)
        theta_w = model2.model.layers[0].self_attn.q_proj.theta.weight.data
        self.assertFalse(torch.all(theta_w == 0))

    def test_attn_init_mode_pretrained_used_for_new_trendwavelet_slots(self):
        cfg1 = _make_tiny_config(pe_attn_mode="standard", pe_mlp_mode="standard")
        model1 = PELlamaForCausalLM(cfg1)

        with torch.no_grad():
            for layer in model1.model.layers:
                layer.self_attn.q_proj.weight.fill_(0.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            cfg2 = _make_tiny_config(
                pe_attn_mode="trend_wavelet",
                pe_mlp_mode="standard",
                attn_init_mode="pretrained",
            )
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        theta_w = model2.model.layers[0].self_attn.q_proj.theta.weight.data
        expected = torch.full_like(theta_w, 0.25)
        torch.testing.assert_close(theta_w, expected)


class TestStagedTrainingNoChange(unittest.TestCase):
    """Reload with same config — should return saved model unchanged."""

    def test_same_config_returns_identical(self):
        cfg = _make_tiny_config(pe_attn_mode="trend_wavelet", pe_mlp_mode="ae_lg")
        model1 = PELlamaForCausalLM(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg, torch_dtype=torch.float32,
            )

        # All parameters should match
        sd1 = model1.state_dict()
        sd2 = model2.state_dict()
        self.assertEqual(set(sd1.keys()), set(sd2.keys()))
        for key in sd1:
            torch.testing.assert_close(sd1[key], sd2[key], msg=f"Mismatch in {key}")

    def test_no_config_returns_saved_model(self):
        cfg = _make_tiny_config(pe_attn_mode="trend_wavelet", pe_mlp_mode="ae")
        model1 = PELlamaForCausalLM(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            # No pe_config → should return as-is
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, torch_dtype=torch.float32,
            )

        self.assertEqual(model2.config.pe_attn_mode, "trend_wavelet")
        self.assertEqual(model2.config.pe_mlp_mode, "ae")


class TestTrendWaveletInitDefaults(unittest.TestCase):
    def test_config_default_attn_init_mode_is_pretrained(self):
        cfg = _make_tiny_config()
        self.assertEqual(cfg.attn_init_mode, "pretrained")


class _FakeLlamaMLP(torch.nn.Module):
    """Minimal SwiGLU-shaped module for testing from_pretrained_mlp."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)


class TestFromPretrainedMLP(unittest.TestCase):
    """Unit tests for PEBottleneckMLP.from_pretrained_mlp()."""

    def _make_mlp(self, cfg):
        return _FakeLlamaMLP(cfg.hidden_size, cfg.intermediate_size)

    def test_fc1_weight_copied(self):
        """fc1.weight equals gate_proj.weight[:mid_dim, :]."""
        from pellm.pe_layers import PEBottleneckMLP
        cfg = _make_tiny_config(pe_mlp_mode="ae", ae_latent_dim=16)
        mlp = self._make_mlp(cfg)
        mid_dim = cfg.hidden_size // 2

        layer = PEBottleneckMLP.from_pretrained_mlp(mlp, cfg)

        expected = mlp.gate_proj.weight[:mid_dim, :]
        torch.testing.assert_close(layer.fc1.weight, expected)

    def test_fc4_weight_copied(self):
        """fc4.weight equals down_proj.weight[:, :mid_dim]."""
        from pellm.pe_layers import PEBottleneckMLP
        cfg = _make_tiny_config(pe_mlp_mode="ae", ae_latent_dim=16)
        mlp = self._make_mlp(cfg)
        mid_dim = cfg.hidden_size // 2

        layer = PEBottleneckMLP.from_pretrained_mlp(mlp, cfg)

        expected = mlp.down_proj.weight[:, :mid_dim]
        torch.testing.assert_close(layer.fc4.weight, expected)

    def test_fc2_fc3_shapes_and_finite(self):
        """fc2 and fc3 have correct shapes and contain no NaN/Inf."""
        from pellm.pe_layers import PEBottleneckMLP
        cfg = _make_tiny_config(pe_mlp_mode="ae", ae_latent_dim=16)
        mlp = self._make_mlp(cfg)
        mid_dim = cfg.hidden_size // 2
        latent_dim = cfg.ae_latent_dim

        layer = PEBottleneckMLP.from_pretrained_mlp(mlp, cfg)

        self.assertEqual(layer.fc2.weight.shape, (latent_dim, mid_dim))
        self.assertEqual(layer.fc3.weight.shape, (mid_dim, latent_dim))
        self.assertTrue(layer.fc2.weight.isfinite().all())
        self.assertTrue(layer.fc3.weight.isfinite().all())

    def test_fc2_fc3_product_approximates_gate_subblock(self):
        """fc2 is row truncation, fc3 is col truncation of gate_proj subblock."""
        from pellm.pe_layers import PEBottleneckMLP
        cfg = _make_tiny_config(pe_mlp_mode="ae", ae_latent_dim=16, ae_inner_init="match")
        mlp = self._make_mlp(cfg)
        mid_dim = cfg.hidden_size // 2
        latent_dim = cfg.ae_latent_dim

        layer = PEBottleneckMLP.from_pretrained_mlp(mlp, cfg)

        W_sub = mlp.gate_proj.weight[:mid_dim, :mid_dim].float()
        # fc2 should be the first latent_dim rows of W_sub
        torch.testing.assert_close(
            layer.fc2.weight.float(), W_sub[:latent_dim, :], atol=1e-5, rtol=1e-5,
        )
        # fc3 should be the first latent_dim columns of W_sub
        torch.testing.assert_close(
            layer.fc3.weight.float(), W_sub[:, :latent_dim], atol=1e-5, rtol=1e-5,
        )

    def test_biases_zeroed(self):
        """All fc biases are zero after initialization."""
        from pellm.pe_layers import PEBottleneckMLP
        cfg = _make_tiny_config(pe_mlp_mode="ae", ae_latent_dim=16)
        mlp = self._make_mlp(cfg)

        layer = PEBottleneckMLP.from_pretrained_mlp(mlp, cfg)

        for name in ("fc1", "fc2", "fc3", "fc4"):
            fc = getattr(layer, name)
            if fc.bias is not None:
                self.assertTrue(fc.bias.data.eq(0).all(), f"{name}.bias not zeroed")

    def test_dtype_bfloat16_preserved(self):
        """All fc weights stay bfloat16 when the source MLP is bfloat16."""
        from pellm.pe_layers import PEBottleneckMLP
        cfg = _make_tiny_config(pe_mlp_mode="ae", ae_latent_dim=16)
        mlp = self._make_mlp(cfg).to(torch.bfloat16)

        layer = PEBottleneckMLP.from_pretrained_mlp(mlp, cfg)

        for name in ("fc1", "fc2", "fc3", "fc4"):
            self.assertEqual(
                getattr(layer, name).weight.dtype, torch.bfloat16,
                f"{name}.weight dtype mismatch",
            )

    def test_lg_variant_latent_gate_is_ones(self):
        """PEBottleneckMLPLG.from_pretrained_mlp leaves latent_gate at ones."""
        from pellm.pe_layers import PEBottleneckMLPLG
        cfg = _make_tiny_config(pe_mlp_mode="ae_lg", ae_latent_dim=16)
        mlp = self._make_mlp(cfg)

        layer = PEBottleneckMLPLG.from_pretrained_mlp(mlp, cfg)

        self.assertTrue(hasattr(layer, "latent_gate"))
        torch.testing.assert_close(
            layer.latent_gate.data, torch.ones(cfg.ae_latent_dim)
        )

    def test_lg_variant_fc_weights_match_base(self):
        """PEBottleneckMLPLG fc weights are identical to PEBottleneckMLP for same source."""
        from pellm.pe_layers import PEBottleneckMLP, PEBottleneckMLPLG
        cfg_ae = _make_tiny_config(pe_mlp_mode="ae",    ae_latent_dim=16)
        cfg_lg = _make_tiny_config(pe_mlp_mode="ae_lg", ae_latent_dim=16)
        mlp = self._make_mlp(cfg_ae)

        layer_ae = PEBottleneckMLP.from_pretrained_mlp(mlp, cfg_ae)
        layer_lg = PEBottleneckMLPLG.from_pretrained_mlp(mlp, cfg_lg)

        for name in ("fc1", "fc2", "fc3", "fc4"):
            torch.testing.assert_close(
                getattr(layer_ae, name).weight,
                getattr(layer_lg, name).weight,
                msg=f"{name}.weight differs between ae and ae_lg",
            )


class TestStagedTrainingMLPLayerExpansion(unittest.TestCase):
    """Same pe_mlp_mode but pe_mlp_layer_indices expands between stages.

    Stage 1: AE MLP on layer 1 only.
    Stage 2: --resume-from stage1, AE MLP on layers 0 and 1.

    Expected behaviour
    ------------------
    - Layer 1 (overlapping): trained weights must be preserved exactly.
    - Layer 0 (newly added): must become a PE bottleneck MLP, not standard LlamaMLP.
    """

    def test_same_mode_ae_expansion_preserves_trained_layer(self):
        """Expanding pe_mlp_layer_indices [1]->[0,1] preserves layer-1 weights."""
        cfg1 = _make_tiny_config(pe_mlp_mode="ae", pe_mlp_layer_indices=[1])
        model1 = PELlamaForCausalLM(cfg1)

        # Simulate training by setting a recognisable value in layer 1's AE MLP
        with torch.no_grad():
            model1.model.layers[1].mlp.fc1.weight.fill_(0.55)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            cfg2 = _make_tiny_config(pe_mlp_mode="ae", pe_mlp_layer_indices=[0, 1])
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        fc1_w = model2.model.layers[1].mlp.fc1.weight.data
        self.assertTrue(
            torch.allclose(fc1_w, torch.full_like(fc1_w, 0.55)),
            "Layer 1 fc1.weight was not preserved after MLP layer-set expansion",
        )

    def test_same_mode_ae_expansion_adds_pe_mlp_at_new_layer(self):
        """Newly added layer 0 must be a PEBottleneckMLP, not a standard LlamaMLP."""
        from pellm.pe_layers import PEBottleneckMLP

        cfg1 = _make_tiny_config(pe_mlp_mode="ae", pe_mlp_layer_indices=[1])
        model1 = PELlamaForCausalLM(cfg1)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            cfg2 = _make_tiny_config(pe_mlp_mode="ae", pe_mlp_layer_indices=[0, 1])
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        # Layer 0 must be a bottleneck MLP, not a plain LlamaMLP
        self.assertIsInstance(
            model2.model.layers[0].mlp, PEBottleneckMLP,
            "Layer 0 mlp is not a PEBottleneckMLP after expansion",
        )
        # Standard LlamaMLP has gate_proj; PEBottleneckMLP has fc1
        self.assertTrue(
            hasattr(model2.model.layers[0].mlp, "fc1"),
            "Layer 0 mlp is missing fc1 (expected PEBottleneckMLP)",
        )
        self.assertFalse(
            hasattr(model2.model.layers[0].mlp, "gate_proj"),
            "Layer 0 mlp still has gate_proj (should have been replaced)",
        )

    def test_same_mode_ae_lg_expansion_preserves_trained_layer_and_adds_new(self):
        """ae_lg variant: layer 1 weights preserved and layer 0 becomes PEBottleneckMLPLG."""
        from pellm.pe_layers import PEBottleneckMLPLG

        cfg1 = _make_tiny_config(pe_mlp_mode="ae_lg", pe_mlp_layer_indices=[1])
        model1 = PELlamaForCausalLM(cfg1)

        with torch.no_grad():
            model1.model.layers[1].mlp.fc1.weight.fill_(0.77)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            cfg2 = _make_tiny_config(pe_mlp_mode="ae_lg", pe_mlp_layer_indices=[0, 1])
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        # Layer 1 trained weights preserved
        fc1_w = model2.model.layers[1].mlp.fc1.weight.data
        self.assertTrue(
            torch.allclose(fc1_w, torch.full_like(fc1_w, 0.77)),
            "Layer 1 fc1.weight was not preserved for ae_lg after expansion",
        )

        # Layer 0 must be PEBottleneckMLPLG (has latent_gate)
        self.assertIsInstance(
            model2.model.layers[0].mlp, PEBottleneckMLPLG,
            "Layer 0 mlp is not a PEBottleneckMLPLG after ae_lg expansion",
        )
        self.assertTrue(
            hasattr(model2.model.layers[0].mlp, "latent_gate"),
            "Layer 0 mlp is missing latent_gate (expected PEBottleneckMLPLG)",
        )

    def test_same_mode_vae_expansion_preserves_trained_layer_and_adds_new(self):
        """vae variant: layer 1 weights preserved and layer 0 becomes PEBottleneckMLPVAE."""
        from pellm.pe_layers import PEBottleneckMLPVAE

        cfg1 = _make_tiny_config(pe_mlp_mode="vae", pe_mlp_layer_indices=[1])
        model1 = PELlamaForCausalLM(cfg1)

        with torch.no_grad():
            model1.model.layers[1].mlp.fc1.weight.fill_(0.33)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            cfg2 = _make_tiny_config(pe_mlp_mode="vae", pe_mlp_layer_indices=[0, 1])
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        fc1_w = model2.model.layers[1].mlp.fc1.weight.data
        self.assertTrue(
            torch.allclose(fc1_w, torch.full_like(fc1_w, 0.33)),
            "Layer 1 fc1.weight was not preserved for vae after expansion",
        )
        self.assertIsInstance(
            model2.model.layers[0].mlp, PEBottleneckMLPVAE,
            "Layer 0 mlp is not a PEBottleneckMLPVAE after vae expansion",
        )
        self.assertTrue(
            hasattr(model2.model.layers[0].mlp, "fc2_mu"),
            "Layer 0 mlp is missing fc2_mu (expected PEBottleneckMLPVAE)",
        )

    def test_same_mode_vae_lg_expansion_preserves_trained_layer_and_adds_new(self):
        """vae_lg variant: layer 1 weights preserved and layer 0 becomes PEBottleneckMLPVAELG."""
        from pellm.pe_layers import PEBottleneckMLPVAELG

        cfg1 = _make_tiny_config(pe_mlp_mode="vae_lg", pe_mlp_layer_indices=[1])
        model1 = PELlamaForCausalLM(cfg1)

        with torch.no_grad():
            model1.model.layers[1].mlp.fc1.weight.fill_(0.91)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)

            cfg2 = _make_tiny_config(pe_mlp_mode="vae_lg", pe_mlp_layer_indices=[0, 1])
            model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                tmpdir, pe_config=cfg2, torch_dtype=torch.float32,
            )

        fc1_w = model2.model.layers[1].mlp.fc1.weight.data
        self.assertTrue(
            torch.allclose(fc1_w, torch.full_like(fc1_w, 0.91)),
            "Layer 1 fc1.weight was not preserved for vae_lg after expansion",
        )
        self.assertIsInstance(
            model2.model.layers[0].mlp, PEBottleneckMLPVAELG,
            "Layer 0 mlp is not a PEBottleneckMLPVAELG after vae_lg expansion",
        )
        self.assertTrue(
            hasattr(model2.model.layers[0].mlp, "latent_gate"),
            "Layer 0 mlp is missing latent_gate (expected PEBottleneckMLPVAELG)",
        )
        self.assertTrue(
            hasattr(model2.model.layers[0].mlp, "fc2_mu"),
            "Layer 0 mlp is missing fc2_mu (expected PEBottleneckMLPVAELG)",
        )


if __name__ == "__main__":
    unittest.main()
