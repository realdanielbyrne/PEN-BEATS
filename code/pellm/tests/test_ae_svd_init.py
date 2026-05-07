import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from pellm.pe_layers import PEBottleneckMLP, PEBottleneckMLPLG


def _make_mock_llama_mlp(hidden_size=64, intermediate_size=256):
    """Create a mock LlamaMLP with gate_proj, up_proj, down_proj."""
    mlp = SimpleNamespace(
        gate_proj=nn.Linear(hidden_size, intermediate_size, bias=False),
        up_proj=nn.Linear(hidden_size, intermediate_size, bias=False),
        down_proj=nn.Linear(intermediate_size, hidden_size, bias=False),
    )
    return mlp


def _make_config(hidden_size=64, intermediate_size=256, ae_latent_dim=16,
                 ae_inner_init="svd"):
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ae_latent_dim=ae_latent_dim,
        ae_inner_init=ae_inner_init,
    )


class TestAEBottleneckSVDInit(unittest.TestCase):
    """Tests for PEBottleneckMLP.from_pretrained_mlp_svd."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden = 64
        self.intermediate = 256
        self.latent = 16
        self.config = _make_config(self.hidden, self.intermediate, self.latent)
        self.mlp = _make_mock_llama_mlp(self.hidden, self.intermediate)

    def test_output_shapes(self):
        """SVD-initialized layer has correct weight shapes."""
        layer = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, self.config)
        mid = self.hidden // 2  # 32
        self.assertEqual(layer.fc1.weight.shape, (mid, self.hidden))
        self.assertEqual(layer.fc2.weight.shape, (self.latent, mid))
        self.assertEqual(layer.fc3.weight.shape, (mid, self.latent))
        self.assertEqual(layer.fc4.weight.shape, (self.hidden, mid))

    def test_forward_runs(self):
        """SVD-initialized layer produces output with correct shape."""
        layer = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, self.config)
        x = torch.randn(2, 8, self.hidden)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, self.hidden))

    def test_svd_differs_from_random(self):
        """SVD init produces different weights than random init."""
        svd_layer = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, self.config)
        rand_layer = PEBottleneckMLP(self.config)
        # fc1 weights should differ
        self.assertFalse(
            torch.allclose(svd_layer.fc1.weight, rand_layer.fc1.weight, atol=1e-6),
            "SVD fc1 weights should differ from random init",
        )

    def test_svd_differs_from_truncation(self):
        """SVD init produces different fc1 weights than truncation init."""
        svd_layer = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, self.config)
        trunc_layer = PEBottleneckMLP.from_pretrained_mlp(self.mlp, self.config)
        # Both are pretrained but use different strategies
        self.assertFalse(
            torch.allclose(svd_layer.fc1.weight, trunc_layer.fc1.weight, atol=1e-6),
            "SVD fc1 weights should differ from truncation init",
        )

    def test_fc1_captures_top_singular_directions(self):
        """fc1 weights from SVD should capture top singular values of gate_proj."""
        layer = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, self.config)
        mid = self.hidden // 2
        W_gate = self.mlp.gate_proj.weight.float()
        _U, S, _Vh = torch.linalg.svd(W_gate, full_matrices=False)
        # fc1.weight = diag(S[:mid]) @ Vh[:mid, :], so its Frobenius norm
        # should equal sqrt(sum(S[:mid]^2))
        expected_norm = torch.sqrt((S[:mid] ** 2).sum())
        actual_norm = torch.linalg.norm(layer.fc1.weight.float())
        torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-4, atol=1e-5)

    def test_fc4_captures_top_singular_directions(self):
        """fc4 weights from SVD should capture top singular values of down_proj."""
        layer = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, self.config)
        mid = self.hidden // 2
        W_down = self.mlp.down_proj.weight.float()
        _U, S, _Vh = torch.linalg.svd(W_down, full_matrices=False)
        expected_norm = torch.sqrt((S[:mid] ** 2).sum())
        actual_norm = torch.linalg.norm(layer.fc4.weight.float())
        torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-4, atol=1e-5)

    def test_deterministic(self):
        """Same seed + same weights → identical SVD init."""
        layer1 = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, self.config)
        layer2 = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, self.config)
        for name in ("fc1", "fc2", "fc3", "fc4"):
            w1 = getattr(layer1, name).weight
            w2 = getattr(layer2, name).weight
            torch.testing.assert_close(w1, w2, rtol=0, atol=0, msg=f"{name} not deterministic")


class TestAEBottleneckLGSVDInit(unittest.TestCase):
    """Tests for PEBottleneckMLPLG.from_pretrained_mlp_svd."""

    def setUp(self):
        torch.manual_seed(42)
        self.config = _make_config(64, 256, 16)
        self.mlp = _make_mock_llama_mlp(64, 256)

    def test_lg_output_shapes(self):
        """LG variant SVD init has correct shapes and latent_gate."""
        layer = PEBottleneckMLPLG.from_pretrained_mlp_svd(self.mlp, self.config)
        self.assertIsInstance(layer, PEBottleneckMLPLG)
        self.assertEqual(layer.latent_gate.shape, (16,))
        # Gate should still be ones (default init, not overwritten by SVD)
        torch.testing.assert_close(
            layer.latent_gate.data, torch.ones(16), rtol=0, atol=0,
        )

    def test_lg_forward_runs(self):
        """LG variant forward pass works after SVD init."""
        layer = PEBottleneckMLPLG.from_pretrained_mlp_svd(self.mlp, self.config)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 64))


class TestAEBottleneckCURInit(unittest.TestCase):
    """Tests for PEBottleneckMLP.from_pretrained_mlp_cur."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden = 64
        self.intermediate = 256
        self.latent = 16
        # Use ae_inner_init="match" so CUR-specific fc2/fc3 tests work
        self.config = _make_config(self.hidden, self.intermediate, self.latent,
                                   ae_inner_init="match")
        self.mlp = _make_mock_llama_mlp(self.hidden, self.intermediate)

    def test_output_shapes(self):
        """CUR-initialized layer has correct weight shapes."""
        layer = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        mid = self.hidden // 2
        self.assertEqual(layer.fc1.weight.shape, (mid, self.hidden))
        self.assertEqual(layer.fc2.weight.shape, (self.latent, mid))
        self.assertEqual(layer.fc3.weight.shape, (mid, self.latent))
        self.assertEqual(layer.fc4.weight.shape, (self.hidden, mid))

    def test_forward_runs(self):
        """CUR-initialized layer produces output with correct shape."""
        layer = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        x = torch.randn(2, 8, self.hidden)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, self.hidden))

    def test_cur_differs_from_truncation(self):
        """CUR init produces different fc1 weights than truncation init."""
        cur_layer = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        trunc_layer = PEBottleneckMLP.from_pretrained_mlp(self.mlp, self.config)
        self.assertFalse(
            torch.allclose(cur_layer.fc1.weight, trunc_layer.fc1.weight, atol=1e-6),
            "CUR fc1 weights should differ from truncation init",
        )

    def test_cur_differs_from_random(self):
        """CUR init produces different weights than random init."""
        cur_layer = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        rand_layer = PEBottleneckMLP(self.config)
        self.assertFalse(
            torch.allclose(cur_layer.fc1.weight, rand_layer.fc1.weight, atol=1e-6),
            "CUR fc1 weights should differ from random init",
        )

    def test_fc1_rows_are_actual_gate_rows(self):
        """CUR fc1 rows should be actual rows from gate_proj."""
        layer = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        W_gate = self.mlp.gate_proj.weight.float()
        for i in range(layer.fc1.weight.shape[0]):
            row = layer.fc1.weight[i].float()
            diffs = (W_gate - row.unsqueeze(0)).abs().sum(dim=1)
            self.assertTrue(
                (diffs < 1e-5).any(),
                f"fc1 row {i} should be an actual row from gate_proj",
            )

    def test_fc4_cols_are_actual_down_cols(self):
        """CUR fc4 columns should be actual columns from down_proj."""
        layer = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        W_down = self.mlp.down_proj.weight.float()
        for j in range(layer.fc4.weight.shape[1]):
            col = layer.fc4.weight[:, j].float()
            diffs = (W_down - col.unsqueeze(1)).abs().sum(dim=0)
            self.assertTrue(
                (diffs < 1e-5).any(),
                f"fc4 column {j} should be an actual column from down_proj",
            )

    def test_fc2_rows_are_actual_submatrix_rows(self):
        """CUR fc2 rows should be actual rows from the W_sub submatrix."""
        layer = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        mid = self.hidden // 2
        # Reconstruct W_sub the same way the init does (CUR-selected fc1 rows, first mid cols)
        W_sub = layer.fc1.weight[:, :mid].float()
        for i in range(layer.fc2.weight.shape[0]):
            row = layer.fc2.weight[i].float()
            diffs = (W_sub - row.unsqueeze(0)).abs().sum(dim=1)
            self.assertTrue(
                (diffs < 1e-5).any(),
                f"fc2 row {i} should be an actual row from W_sub",
            )

    def test_fc3_cols_are_actual_submatrix_cols(self):
        """CUR fc3 columns should be actual columns from the W_sub submatrix."""
        layer = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        mid = self.hidden // 2
        W_sub = layer.fc1.weight[:, :mid].float()
        for j in range(layer.fc3.weight.shape[1]):
            col = layer.fc3.weight[:, j].float()
            diffs = (W_sub - col.unsqueeze(1)).abs().sum(dim=0)
            self.assertTrue(
                (diffs < 1e-5).any(),
                f"fc3 column {j} should be an actual column from W_sub",
            )

    def test_deterministic(self):
        """Same weights -> identical CUR init."""
        layer1 = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        layer2 = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, self.config)
        for name in ("fc1", "fc2", "fc3", "fc4"):
            w1 = getattr(layer1, name).weight
            w2 = getattr(layer2, name).weight
            torch.testing.assert_close(w1, w2, rtol=0, atol=0, msg=f"{name} not deterministic")


class TestAEBottleneckLGCURInit(unittest.TestCase):
    """Tests for PEBottleneckMLPLG.from_pretrained_mlp_cur."""

    def setUp(self):
        torch.manual_seed(42)
        self.config = _make_config(64, 256, 16)
        self.mlp = _make_mock_llama_mlp(64, 256)

    def test_lg_output_shapes(self):
        """LG variant CUR init has correct shapes and latent_gate."""
        layer = PEBottleneckMLPLG.from_pretrained_mlp_cur(self.mlp, self.config)
        self.assertIsInstance(layer, PEBottleneckMLPLG)
        self.assertEqual(layer.latent_gate.shape, (16,))
        torch.testing.assert_close(
            layer.latent_gate.data, torch.ones(16), rtol=0, atol=0,
        )

    def test_lg_forward_runs(self):
        """LG variant forward pass works after CUR init."""
        layer = PEBottleneckMLPLG.from_pretrained_mlp_cur(self.mlp, self.config)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 64))


class TestAEBottleneckFourierInit(unittest.TestCase):
    """Tests for PEBottleneckMLP.from_pretrained_mlp_fourier."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden = 64
        self.intermediate = 256
        self.latent = 16
        self.config = _make_config(self.hidden, self.intermediate, self.latent)
        self.mlp = _make_mock_llama_mlp(self.hidden, self.intermediate)

    def test_output_shapes(self):
        """Fourier-initialized layer has correct weight shapes."""
        layer = PEBottleneckMLP.from_pretrained_mlp_fourier(self.mlp, self.config)
        mid = self.hidden // 2
        self.assertEqual(layer.fc1.weight.shape, (mid, self.hidden))
        self.assertEqual(layer.fc2.weight.shape, (self.latent, mid))
        self.assertEqual(layer.fc3.weight.shape, (mid, self.latent))
        self.assertEqual(layer.fc4.weight.shape, (self.hidden, mid))

    def test_forward_runs(self):
        """Fourier-initialized layer produces output with correct shape."""
        layer = PEBottleneckMLP.from_pretrained_mlp_fourier(self.mlp, self.config)
        x = torch.randn(2, 8, self.hidden)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, self.hidden))

    def test_fourier_differs_from_truncation(self):
        """Fourier init produces different fc1 weights than truncation init."""
        fourier_layer = PEBottleneckMLP.from_pretrained_mlp_fourier(self.mlp, self.config)
        trunc_layer = PEBottleneckMLP.from_pretrained_mlp(self.mlp, self.config)
        self.assertFalse(
            torch.allclose(fourier_layer.fc1.weight, trunc_layer.fc1.weight, atol=1e-6),
            "Fourier fc1 weights should differ from truncation init",
        )

    def test_fourier_reduces_high_frequency_energy(self):
        """Fourier-filtered fc1 should have more near-zero frequency bins."""
        fourier_layer = PEBottleneckMLP.from_pretrained_mlp_fourier(self.mlp, self.config)
        trunc_layer = PEBottleneckMLP.from_pretrained_mlp(self.mlp, self.config)
        spec_fourier = torch.fft.rfft(fourier_layer.fc1.weight.float(), dim=1).abs()
        spec_trunc = torch.fft.rfft(trunc_layer.fc1.weight.float(), dim=1).abs()
        fourier_near_zero = (spec_fourier < 1e-6).sum()
        trunc_near_zero = (spec_trunc < 1e-6).sum()
        self.assertGreater(
            fourier_near_zero.item(), trunc_near_zero.item(),
            "Fourier filtering should zero out more frequency bins than truncation",
        )

    def test_deterministic(self):
        """Same weights -> identical Fourier init."""
        layer1 = PEBottleneckMLP.from_pretrained_mlp_fourier(self.mlp, self.config)
        layer2 = PEBottleneckMLP.from_pretrained_mlp_fourier(self.mlp, self.config)
        for name in ("fc1", "fc2", "fc3", "fc4"):
            w1 = getattr(layer1, name).weight
            w2 = getattr(layer2, name).weight
            torch.testing.assert_close(w1, w2, rtol=0, atol=0, msg=f"{name} not deterministic")


class TestAEBottleneckLGFourierInit(unittest.TestCase):
    """Tests for PEBottleneckMLPLG.from_pretrained_mlp_fourier."""

    def setUp(self):
        torch.manual_seed(42)
        self.config = _make_config(64, 256, 16)
        self.mlp = _make_mock_llama_mlp(64, 256)

    def test_lg_output_shapes(self):
        """LG variant Fourier init has correct shapes and latent_gate."""
        layer = PEBottleneckMLPLG.from_pretrained_mlp_fourier(self.mlp, self.config)
        self.assertIsInstance(layer, PEBottleneckMLPLG)
        self.assertEqual(layer.latent_gate.shape, (16,))
        torch.testing.assert_close(
            layer.latent_gate.data, torch.ones(16), rtol=0, atol=0,
        )

    def test_lg_forward_runs(self):
        """LG variant forward pass works after Fourier init."""
        layer = PEBottleneckMLPLG.from_pretrained_mlp_fourier(self.mlp, self.config)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 64))


class TestAEInnerInit(unittest.TestCase):
    """Tests for the ae_inner_init config parameter."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden = 64
        self.intermediate = 256
        self.latent = 16
        self.mlp = _make_mock_llama_mlp(self.hidden, self.intermediate)

    def test_pretrained_svd_inner_differs_from_match(self):
        """Pretrained mode: ae_inner_init='svd' produces different fc2/fc3 than 'match'."""
        cfg_svd = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="svd")
        cfg_match = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="match")
        layer_svd = PEBottleneckMLP.from_pretrained_mlp(self.mlp, cfg_svd)
        layer_match = PEBottleneckMLP.from_pretrained_mlp(self.mlp, cfg_match)
        # fc1 and fc4 should be identical (outer layers unaffected)
        torch.testing.assert_close(layer_svd.fc1.weight, layer_match.fc1.weight)
        torch.testing.assert_close(layer_svd.fc4.weight, layer_match.fc4.weight)
        # fc2 and fc3 should differ
        self.assertFalse(
            torch.allclose(layer_svd.fc2.weight, layer_match.fc2.weight, atol=1e-6),
            "fc2 weights should differ between svd and match inner init",
        )
        self.assertFalse(
            torch.allclose(layer_svd.fc3.weight, layer_match.fc3.weight, atol=1e-6),
            "fc3 weights should differ between svd and match inner init",
        )

    def test_cur_svd_inner_differs_from_match(self):
        """CUR mode: ae_inner_init='svd' produces different fc2/fc3 than 'match'."""
        cfg_svd = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="svd")
        cfg_match = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="match")
        layer_svd = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, cfg_svd)
        layer_match = PEBottleneckMLP.from_pretrained_mlp_cur(self.mlp, cfg_match)
        # fc1 and fc4 should be identical
        torch.testing.assert_close(layer_svd.fc1.weight, layer_match.fc1.weight)
        torch.testing.assert_close(layer_svd.fc4.weight, layer_match.fc4.weight)
        # fc2 and fc3 should differ
        self.assertFalse(
            torch.allclose(layer_svd.fc2.weight, layer_match.fc2.weight, atol=1e-6),
            "CUR fc2 weights should differ between svd and match inner init",
        )

    def test_fourier_svd_inner_differs_from_match(self):
        """Fourier mode: ae_inner_init='svd' produces different fc2/fc3 than 'match'."""
        cfg_svd = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="svd")
        cfg_match = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="match")
        layer_svd = PEBottleneckMLP.from_pretrained_mlp_fourier(self.mlp, cfg_svd)
        layer_match = PEBottleneckMLP.from_pretrained_mlp_fourier(self.mlp, cfg_match)
        # fc1 and fc4 should be identical
        torch.testing.assert_close(layer_svd.fc1.weight, layer_match.fc1.weight)
        torch.testing.assert_close(layer_svd.fc4.weight, layer_match.fc4.weight)
        # fc2 and fc3 should differ
        self.assertFalse(
            torch.allclose(layer_svd.fc2.weight, layer_match.fc2.weight, atol=1e-6),
            "Fourier fc2 weights should differ between svd and match inner init",
        )

    def test_svd_mode_unaffected_by_inner_init(self):
        """SVD init mode already uses SVD for fc2/fc3 — both settings should match."""
        cfg_svd = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="svd")
        cfg_match = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="match")
        layer_svd = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, cfg_svd)
        layer_match = PEBottleneckMLP.from_pretrained_mlp_svd(self.mlp, cfg_match)
        # SVD mode doesn't read ae_inner_init — both should be identical
        for name in ("fc1", "fc2", "fc3", "fc4"):
            torch.testing.assert_close(
                getattr(layer_svd, name).weight,
                getattr(layer_match, name).weight,
                msg=f"SVD mode {name} should be unaffected by ae_inner_init",
            )

    def test_pretrained_match_preserves_old_behavior(self):
        """ae_inner_init='match' with pretrained mode produces truncation-based fc2/fc3."""
        cfg = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="match")
        layer = PEBottleneckMLP.from_pretrained_mlp(self.mlp, cfg)
        mid = self.hidden // 2
        W_gate = self.mlp.gate_proj.weight.float()
        W_sub = W_gate[:mid, :mid]
        # fc2 should be row truncation of W_sub
        torch.testing.assert_close(
            layer.fc2.weight.float(), W_sub[:self.latent, :],
            msg="match mode fc2 should be row truncation of W_sub",
        )
        # fc3 should be column truncation of W_sub
        torch.testing.assert_close(
            layer.fc3.weight.float(), W_sub[:, :self.latent],
            msg="match mode fc3 should be column truncation of W_sub",
        )

    def test_lg_variant_svd_inner(self):
        """LG variant respects ae_inner_init='svd'."""
        cfg = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="svd")
        layer = PEBottleneckMLPLG.from_pretrained_mlp(self.mlp, cfg)
        self.assertIsInstance(layer, PEBottleneckMLPLG)
        # Verify fc2/fc3 are SVD-based (same as non-LG variant)
        cfg2 = _make_config(self.hidden, self.intermediate, self.latent, ae_inner_init="svd")
        layer_base = PEBottleneckMLP.from_pretrained_mlp(self.mlp, cfg2)
        torch.testing.assert_close(layer.fc2.weight, layer_base.fc2.weight)
        torch.testing.assert_close(layer.fc3.weight, layer_base.fc3.weight)

    def test_forward_runs_both_modes(self):
        """Forward pass works with both ae_inner_init values."""
        x = torch.randn(2, 8, self.hidden)
        for inner_init in ("svd", "match"):
            cfg = _make_config(self.hidden, self.intermediate, self.latent,
                               ae_inner_init=inner_init)
            for init_fn in ("from_pretrained_mlp", "from_pretrained_mlp_cur",
                            "from_pretrained_mlp_fourier"):
                layer = getattr(PEBottleneckMLP, init_fn)(self.mlp, cfg)
                out = layer(x)
                self.assertEqual(out.shape, (2, 8, self.hidden),
                                 f"{init_fn} with {inner_init} failed")


if __name__ == "__main__":
    unittest.main()

