import unittest

import torch

from pellm.pe_layers import (
    TrendWaveletGenericLinear,
    TrendWaveletGenericLinearLG,
    TrendWaveletGenericLinearLGReduced,
    TrendWaveletGenericLinearReduced,
    TrendWaveletLinear,
    TrendWaveletLinearLG,
    TrendWaveletLinearLGReduced,
    TrendWaveletLinearReduced,
)


def _expected_reduction_weight(in_features: int, reduction_dim: int) -> torch.Tensor:
    expected = torch.zeros(reduction_dim, in_features)
    diagonal_dim = min(in_features, reduction_dim)
    expected[:diagonal_dim, :diagonal_dim] = 2.0 * torch.eye(diagonal_dim)
    return expected


class TrendWaveletLinearProjectionTests(unittest.TestCase):
    def test_lstsq_matches_legacy_pinv_impulse_response(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(16, 32, bias=False)

        layer = TrendWaveletLinear.from_pretrained_linear(
            linear,
            trend_dim=4,
            wavelet_dim=12,
        )
        basis = torch.cat([layer.trend_basis, layer.wavelet_basis], dim=0)
        legacy_theta_weight = torch.linalg.pinv(basis).T @ linear.weight.detach()

        impulses = torch.eye(linear.in_features)
        current_response = layer(impulses)
        legacy_response = (impulses @ legacy_theta_weight.T) @ basis

        torch.testing.assert_close(
            current_response,
            legacy_response,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_lstsq_matches_legacy_pinv_reconstruction_residual(self):
        torch.manual_seed(1)
        linear = torch.nn.Linear(24, 40, bias=False)

        layer = TrendWaveletLinear.from_pretrained_linear(
            linear,
            trend_dim=5,
            wavelet_dim=16,
            wavelet_type="db2",
        )
        basis = torch.cat([layer.trend_basis, layer.wavelet_basis], dim=0)
        legacy_theta_weight = torch.linalg.pinv(basis).T @ linear.weight.detach()

        current_residual = torch.linalg.norm(
            basis.T @ layer.theta.weight.detach() - linear.weight.detach()
        )
        legacy_residual = torch.linalg.norm(
            basis.T @ legacy_theta_weight - linear.weight.detach()
        )

        self.assertLessEqual(
            abs(current_residual.item() - legacy_residual.item()), 1e-5
        )

    def test_pretrained_truncates_dense_weight_rows(self):
        torch.manual_seed(2)
        linear = torch.nn.Linear(16, 32, bias=False)

        layer = TrendWaveletLinear.from_pretrained_linear_pretrained(
            linear,
            trend_dim=4,
            wavelet_dim=12,
        )

        expected = linear.weight[: layer.theta.out_features]
        torch.testing.assert_close(layer.theta.weight, expected)

    def test_generic_pretrained_zeros_generic_branch(self):
        torch.manual_seed(3)
        linear = torch.nn.Linear(16, 32, bias=False)

        layer = TrendWaveletGenericLinear.from_pretrained_linear_pretrained(
            linear,
            trend_dim=4,
            wavelet_dim=12,
            generic_dim=5,
        )

        frozen_dim = layer.trend_basis.shape[0] + layer.wavelet_basis.shape[0]
        torch.testing.assert_close(
            layer.theta.weight[:frozen_dim],
            linear.weight[:frozen_dim],
        )
        self.assertTrue(torch.all(layer.theta.weight[frozen_dim:] == 0))
        self.assertTrue(torch.all(layer.generic_basis.weight == 0))

    def test_reduced_pretrained_uses_structured_input_projection(self):
        torch.manual_seed(31)
        linear = torch.nn.Linear(16, 32, bias=False)

        layer = TrendWaveletLinearReduced.from_pretrained_linear_pretrained(
            linear,
            trend_dim=4,
            wavelet_dim=12,
            reduction_dim=6,
        )

        expected_theta = linear.weight[
            : layer.theta.out_features, : layer.reduction_dim
        ]
        torch.testing.assert_close(layer.theta.weight, expected_theta)
        torch.testing.assert_close(
            layer.reduction.weight,
            _expected_reduction_weight(linear.in_features, layer.reduction_dim),
        )

    def test_generic_reduced_pretrained_zeros_generic_branch(self):
        torch.manual_seed(32)
        linear = torch.nn.Linear(16, 32, bias=False)

        layer = TrendWaveletGenericLinearReduced.from_pretrained_linear_pretrained(
            linear,
            trend_dim=4,
            wavelet_dim=12,
            generic_dim=5,
            reduction_dim=6,
        )

        frozen_dim = layer.trend_basis.shape[0] + layer.wavelet_basis.shape[0]
        torch.testing.assert_close(
            layer.theta.weight[:frozen_dim],
            linear.weight[:frozen_dim, : layer.reduction_dim],
        )
        self.assertTrue(torch.all(layer.theta.weight[frozen_dim:] == 0))
        self.assertTrue(torch.all(layer.generic_basis.weight == 0))
        torch.testing.assert_close(
            layer.reduction.weight,
            _expected_reduction_weight(linear.in_features, layer.reduction_dim),
        )

    def test_lg_variants_keep_gate_at_ones(self):
        torch.manual_seed(4)
        linear = torch.nn.Linear(16, 32, bias=False)

        layer = TrendWaveletLinearLG.from_pretrained_linear_fourier(
            linear,
            trend_dim=4,
            wavelet_dim=12,
        )
        generic_layer = TrendWaveletGenericLinearLG.from_pretrained_linear_svd(
            linear,
            trend_dim=4,
            wavelet_dim=12,
            generic_dim=5,
        )

        self.assertTrue(torch.all(layer.coeff_gate == 1))
        self.assertTrue(torch.all(generic_layer.coeff_gate == 1))

    def test_reduced_lg_variants_keep_gate_at_ones(self):
        torch.manual_seed(41)
        linear = torch.nn.Linear(16, 32, bias=False)

        layer = TrendWaveletLinearLGReduced.from_pretrained_linear_fourier(
            linear,
            trend_dim=4,
            wavelet_dim=12,
            reduction_dim=6,
        )
        generic_layer = TrendWaveletGenericLinearLGReduced.from_pretrained_linear_svd(
            linear,
            trend_dim=4,
            wavelet_dim=12,
            generic_dim=5,
            reduction_dim=6,
        )

        self.assertTrue(torch.all(layer.coeff_gate == 1))
        self.assertTrue(torch.all(generic_layer.coeff_gate == 1))

    def test_svd_cur_and_fourier_return_finite_weights(self):
        torch.manual_seed(5)
        linear = torch.nn.Linear(24, 40, bias=False)
        variants = [
            TrendWaveletLinear.from_pretrained_linear_svd(
                linear,
                trend_dim=5,
                wavelet_dim=16,
                wavelet_type="db2",
            ),
            TrendWaveletLinear.from_pretrained_linear_cur(
                linear,
                trend_dim=5,
                wavelet_dim=16,
                wavelet_type="db2",
            ),
            TrendWaveletLinear.from_pretrained_linear_fourier(
                linear,
                trend_dim=5,
                wavelet_dim=16,
                wavelet_type="db2",
            ),
        ]

        for layer in variants:
            self.assertEqual(layer.theta.weight.shape[1], linear.in_features)
            self.assertTrue(layer.theta.weight.isfinite().all())

    def test_reduced_svd_cur_and_fourier_return_finite_weights(self):
        torch.manual_seed(51)
        linear = torch.nn.Linear(24, 40, bias=False)
        reduction_dim = 10
        variants = [
            TrendWaveletLinearReduced.from_pretrained_linear_svd(
                linear,
                trend_dim=5,
                wavelet_dim=16,
                wavelet_type="db2",
                reduction_dim=reduction_dim,
            ),
            TrendWaveletLinearReduced.from_pretrained_linear_cur(
                linear,
                trend_dim=5,
                wavelet_dim=16,
                wavelet_type="db2",
                reduction_dim=reduction_dim,
            ),
            TrendWaveletLinearReduced.from_pretrained_linear_fourier(
                linear,
                trend_dim=5,
                wavelet_dim=16,
                wavelet_type="db2",
                reduction_dim=reduction_dim,
            ),
        ]

        expected_reduction = _expected_reduction_weight(
            linear.in_features,
            reduction_dim,
        )
        for layer in variants:
            self.assertEqual(layer.theta.weight.shape[1], reduction_dim)
            self.assertTrue(layer.theta.weight.isfinite().all())
            torch.testing.assert_close(layer.reduction.weight, expected_reduction)


class ReducedTrendWaveletModelWiringTests(unittest.TestCase):
    """Verify all 4 reduced attention modes can be instantiated at the model level."""

    _REDUCED_MODES = [
        "trend_wavelet_reduced",
        "trend_wavelet_lg_reduced",
        "trend_wavelet_generic_reduced",
        "trend_wavelet_generic_lg_reduced",
    ]

    _REDUCED_CLS_MAP = {
        "trend_wavelet_reduced": TrendWaveletLinearReduced,
        "trend_wavelet_lg_reduced": TrendWaveletLinearLGReduced,
        "trend_wavelet_generic_reduced": TrendWaveletGenericLinearReduced,
        "trend_wavelet_generic_lg_reduced": TrendWaveletGenericLinearLGReduced,
    }

    def _make_tiny_config(self, **overrides):
        from pellm.configuration_pe_llama import PELlamaConfig

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
            generic_dim=5,
            reduction_dim=8,
        )
        defaults.update(overrides)
        return PELlamaConfig(**defaults)

    def test_reduced_modes_instantiate_and_forward(self):
        from pellm.modeling_pe_llama import PELlamaForCausalLM

        for mode in self._REDUCED_MODES:
            with self.subTest(mode=mode):
                cfg = self._make_tiny_config(pe_attn_mode=mode)
                model = PELlamaForCausalLM(cfg)

                # Check attention projections are correct class
                layer0 = model.model.layers[0]
                expected_cls = self._REDUCED_CLS_MAP[mode]
                self.assertIsInstance(layer0.self_attn.q_proj, expected_cls)
                self.assertIsInstance(layer0.self_attn.k_proj, expected_cls)

                # Forward pass should not raise
                input_ids = torch.randint(0, 256, (1, 8))
                with torch.no_grad():
                    output = model(input_ids)
                self.assertEqual(output.logits.shape, (1, 8, 256))

    def test_reduced_modes_save_and_reload(self):
        import tempfile
        from pellm.modeling_pe_llama import PELlamaForCausalLM

        for mode in self._REDUCED_MODES:
            with self.subTest(mode=mode):
                cfg = self._make_tiny_config(pe_attn_mode=mode)
                model1 = PELlamaForCausalLM(cfg)

                with tempfile.TemporaryDirectory() as tmpdir:
                    model1.save_pretrained(tmpdir)
                    model2 = PELlamaForCausalLM.from_pretrained_pe_llama(
                        tmpdir,
                        pe_config=cfg,
                        torch_dtype=torch.float32,
                    )

                # All parameters should match
                sd1 = model1.state_dict()
                sd2 = model2.state_dict()
                self.assertEqual(set(sd1.keys()), set(sd2.keys()))
                for key in sd1:
                    torch.testing.assert_close(
                        sd1[key], sd2[key], msg=f"Mismatch in {key}"
                    )

    def test_reduced_lg_gates_initialized_to_ones(self):
        from pellm.modeling_pe_llama import PELlamaForCausalLM

        for mode in ("trend_wavelet_lg_reduced", "trend_wavelet_generic_lg_reduced"):
            with self.subTest(mode=mode):
                cfg = self._make_tiny_config(pe_attn_mode=mode)
                model = PELlamaForCausalLM(cfg)
                for layer in model.model.layers:
                    gate = layer.self_attn.q_proj.coeff_gate
                    self.assertTrue(torch.all(gate == 1.0))

    def test_active_g_toggle_works_on_reduced(self):
        from pellm.modeling_pe_llama import PELlamaForCausalLM
        from pellm.pe_layers import set_tw_active_g

        for mode in self._REDUCED_MODES:
            with self.subTest(mode=mode):
                cfg = self._make_tiny_config(pe_attn_mode=mode)
                model = PELlamaForCausalLM(cfg)

                set_tw_active_g(model, True)
                self.assertTrue(model.model.layers[0].self_attn.q_proj.active_g)

                set_tw_active_g(model, False)
                self.assertFalse(model.model.layers[0].self_attn.q_proj.active_g)


if __name__ == "__main__":
    unittest.main()
