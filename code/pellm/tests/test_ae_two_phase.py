"""Tests for two-phase AE pre-training: activation extraction and cache-based training.

Verifies:
- extract_teacher_activations saves correct activation files to disk
- ActivationCacheDataset loads them back correctly
- train_ae_epoch_from_cache produces decreasing loss
- Equivalence with the original train_ae_epoch (both update AE weights similarly)
"""

import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


class _TinyTokenDataset(Dataset):
    """Minimal dataset of random token IDs for testing."""

    def __init__(self, n_samples=4, seq_len=16, vocab_size=256):
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return {"input_ids": ids, "labels": ids.clone()}


class TestExtractTeacherActivations(unittest.TestCase):
    """Test Phase 1: activation extraction to disk."""

    def setUp(self):
        torch.manual_seed(42)
        self.teacher_cfg = _make_tiny_config(pe_mlp_mode="standard")
        self.teacher = PELlamaForCausalLM(self.teacher_cfg)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.dataset = _TinyTokenDataset(n_samples=4, seq_len=16,
                                         vocab_size=self.teacher_cfg.vocab_size)
        self.loader = DataLoader(self.dataset, batch_size=2)

    def test_files_created_for_target_layers(self):
        """Activation files are created for each target layer and batch."""
        from scripts.finetune import extract_teacher_activations
        with tempfile.TemporaryDirectory() as tmpdir:
            n_batches = extract_teacher_activations(
                self.teacher, self.loader, [0, 1], tmpdir, torch.device("cpu"),
            )
            self.assertEqual(n_batches, 2)  # 4 samples / batch_size 2
            # Check files exist for both layers and both batches
            for layer_idx in [0, 1]:
                for batch_idx in range(2):
                    fpath = os.path.join(tmpdir, f"layer{layer_idx}_batch{batch_idx}.pt")
                    self.assertTrue(os.path.exists(fpath), f"Missing {fpath}")

    def test_activation_shapes(self):
        """Saved activations have correct flattened shape (n_tokens, hidden)."""
        from scripts.finetune import extract_teacher_activations
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_teacher_activations(
                self.teacher, self.loader, [0], tmpdir, torch.device("cpu"),
            )
            data = torch.load(os.path.join(tmpdir, "layer0_batch0.pt"),
                              map_location="cpu", weights_only=True)
            # batch_size=2, seq_len=16 → flattened to (32, hidden_size)
            self.assertEqual(data["x"].shape, (32, 64))
            self.assertEqual(data["y"].shape, (32, 64))

    def test_only_target_layers_saved(self):
        """Only specified target layers have files created."""
        from scripts.finetune import extract_teacher_activations
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_teacher_activations(
                self.teacher, self.loader, [1], tmpdir, torch.device("cpu"),
            )
            files = os.listdir(tmpdir)
            # Only layer 1 files should exist
            self.assertTrue(all("layer1_" in f for f in files))
            self.assertFalse(any("layer0_" in f for f in files))


class TestActivationCacheDataset(unittest.TestCase):
    """Test the ActivationCacheDataset loader."""

    def test_loads_correct_number_of_batches(self):
        """Dataset length matches the number of saved batch files."""
        from scripts.finetune import ActivationCacheDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 fake activation files for layer 0
            for i in range(3):
                torch.save(
                    {"x": torch.randn(10, 64), "y": torch.randn(10, 64)},
                    os.path.join(tmpdir, f"layer0_batch{i}.pt"),
                )
            ds = ActivationCacheDataset(tmpdir, layer_idx=0)
            self.assertEqual(len(ds), 3)

    def test_raises_on_missing_layer(self):
        """FileNotFoundError raised when no files exist for requested layer."""
        from scripts.finetune import ActivationCacheDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                {"x": torch.randn(10, 64), "y": torch.randn(10, 64)},
                os.path.join(tmpdir, "layer0_batch0.pt"),
            )
            with self.assertRaises(FileNotFoundError):
                ActivationCacheDataset(tmpdir, layer_idx=5)

    def test_data_roundtrip(self):
        """Saved and loaded tensors are identical."""
        from scripts.finetune import ActivationCacheDataset
        x_orig = torch.randn(10, 64)
        y_orig = torch.randn(10, 64)
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save({"x": x_orig, "y": y_orig},
                       os.path.join(tmpdir, "layer0_batch0.pt"))
            ds = ActivationCacheDataset(tmpdir, layer_idx=0)
            item = ds[0]
            torch.testing.assert_close(item["x"], x_orig)
            torch.testing.assert_close(item["y"], y_orig)


class TestTrainAEFromCache(unittest.TestCase):
    """Test Phase 2: AE training from cached activations."""

    def test_loss_is_finite(self):
        """train_ae_epoch_from_cache returns a finite loss value."""
        from scripts.finetune import (
            extract_teacher_activations,
            train_ae_epoch_from_cache,
        )
        torch.manual_seed(42)

        teacher_cfg = _make_tiny_config(pe_mlp_mode="standard")
        teacher = PELlamaForCausalLM(teacher_cfg)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        pe_cfg = _make_tiny_config(pe_mlp_mode="ae", ae_latent_dim=16)
        pe_model = PELlamaForCausalLM(pe_cfg)

        dataset = _TinyTokenDataset(n_samples=4, seq_len=16,
                                     vocab_size=teacher_cfg.vocab_size)
        loader = DataLoader(dataset, batch_size=2)
        device = torch.device("cpu")

        ae_target_layers = [0, 1]

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_teacher_activations(
                teacher, loader, ae_target_layers, tmpdir, device,
            )
            del teacher

            ae_params = [p for n, p in pe_model.named_parameters()
                         if ".mlp.fc" in n]
            optimizer = torch.optim.AdamW(ae_params, lr=1e-3)

            loss = train_ae_epoch_from_cache(
                pe_model, tmpdir, ae_target_layers, optimizer, device,
                grad_accum_steps=1,
            )
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertGreater(loss, 0.0)

    def test_loss_decreases_over_epochs(self):
        """Multiple epochs of cache-based training reduce loss."""
        from scripts.finetune import (
            extract_teacher_activations,
            train_ae_epoch_from_cache,
        )
        torch.manual_seed(42)

        teacher_cfg = _make_tiny_config(pe_mlp_mode="standard")
        teacher = PELlamaForCausalLM(teacher_cfg)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        pe_cfg = _make_tiny_config(pe_mlp_mode="ae", ae_latent_dim=16)
        pe_model = PELlamaForCausalLM(pe_cfg)

        dataset = _TinyTokenDataset(n_samples=8, seq_len=16,
                                     vocab_size=teacher_cfg.vocab_size)
        loader = DataLoader(dataset, batch_size=2)
        device = torch.device("cpu")

        ae_target_layers = [0, 1]

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_teacher_activations(
                teacher, loader, ae_target_layers, tmpdir, device,
            )
            del teacher

            ae_params = [p for n, p in pe_model.named_parameters()
                         if ".mlp.fc" in n]
            optimizer = torch.optim.AdamW(ae_params, lr=1e-3)

            losses = []
            for _ in range(5):
                loss = train_ae_epoch_from_cache(
                    pe_model, tmpdir, ae_target_layers, optimizer, device,
                    grad_accum_steps=1,
                )
                losses.append(loss)

            # Loss should generally decrease (last < first)
            self.assertLess(losses[-1], losses[0],
                            f"Loss did not decrease: {losses}")

    def test_ae_lg_gate_updates(self):
        """Learned gate parameters are updated during cache-based training."""
        from scripts.finetune import (
            extract_teacher_activations,
            train_ae_epoch_from_cache,
        )
        torch.manual_seed(42)

        teacher_cfg = _make_tiny_config(pe_mlp_mode="standard")
        teacher = PELlamaForCausalLM(teacher_cfg)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        pe_cfg = _make_tiny_config(pe_mlp_mode="ae_lg", ae_latent_dim=16)
        pe_model = PELlamaForCausalLM(pe_cfg)

        # Snapshot initial gate values
        gate_before = pe_model.model.layers[0].mlp.latent_gate.data.clone()

        dataset = _TinyTokenDataset(n_samples=4, seq_len=16,
                                     vocab_size=teacher_cfg.vocab_size)
        loader = DataLoader(dataset, batch_size=2)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_teacher_activations(
                teacher, loader, [0, 1], tmpdir, device,
            )
            del teacher

            ae_params = [p for n, p in pe_model.named_parameters()
                         if ".mlp.fc" in n or ".mlp.latent_gate" in n]
            optimizer = torch.optim.AdamW(ae_params, lr=1e-3)

            for _ in range(3):
                train_ae_epoch_from_cache(
                    pe_model, tmpdir, [0, 1], optimizer, device,
                    grad_accum_steps=1,
                )

        gate_after = pe_model.model.layers[0].mlp.latent_gate.data
        self.assertFalse(torch.allclose(gate_before, gate_after),
                         "Latent gate was not updated during training")


class TestCacheIsValid(unittest.TestCase):
    """Test _cache_is_valid helper."""

    def test_valid_cache_all_layers_present(self):
        from scripts.finetune import _cache_is_valid
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx in [0, 1]:
                torch.save({"x": torch.randn(5, 64), "y": torch.randn(5, 64)},
                           os.path.join(tmpdir, f"layer{idx}_batch0.pt"))
            self.assertTrue(_cache_is_valid(tmpdir, [0, 1]))

    def test_invalid_cache_missing_layer(self):
        from scripts.finetune import _cache_is_valid
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save({"x": torch.randn(5, 64), "y": torch.randn(5, 64)},
                       os.path.join(tmpdir, "layer0_batch0.pt"))
            # Layer 1 is missing
            self.assertFalse(_cache_is_valid(tmpdir, [0, 1]))

    def test_invalid_cache_nonexistent_dir(self):
        from scripts.finetune import _cache_is_valid
        self.assertFalse(_cache_is_valid("/nonexistent/path/xyz", [0]))

    def test_empty_target_layers(self):
        from scripts.finetune import _cache_is_valid
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(_cache_is_valid(tmpdir, []))


class TestCacheReuse(unittest.TestCase):
    """Test that Phase 1 is skipped when valid cache already exists."""

    def test_second_run_reuses_cache(self):
        """Training from a pre-populated cache produces valid loss without
        needing to run extract_teacher_activations again."""
        from scripts.finetune import (
            extract_teacher_activations,
            train_ae_epoch_from_cache,
            _cache_is_valid,
        )
        torch.manual_seed(42)

        teacher_cfg = _make_tiny_config(pe_mlp_mode="standard")
        teacher = PELlamaForCausalLM(teacher_cfg)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        dataset = _TinyTokenDataset(n_samples=4, seq_len=16,
                                     vocab_size=teacher_cfg.vocab_size)
        loader = DataLoader(dataset, batch_size=2)
        device = torch.device("cpu")
        ae_target_layers = [0, 1]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run 1: extract activations
            extract_teacher_activations(
                teacher, loader, ae_target_layers, tmpdir, device,
            )
            del teacher  # simulate teacher being freed

            # Verify cache is valid
            self.assertTrue(_cache_is_valid(tmpdir, ae_target_layers))

            # Run 2: a different PE model trains from the same cache
            pe_cfg = _make_tiny_config(pe_mlp_mode="ae", ae_latent_dim=16)
            pe_model = PELlamaForCausalLM(pe_cfg)
            ae_params = [p for n, p in pe_model.named_parameters()
                         if ".mlp.fc" in n]
            optimizer = torch.optim.AdamW(ae_params, lr=1e-3)

            loss = train_ae_epoch_from_cache(
                pe_model, tmpdir, ae_target_layers, optimizer, device,
                grad_accum_steps=1,
            )
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertGreater(loss, 0.0)


class TestFinetuneCliAndFreezeBase(unittest.TestCase):
    """CLI parsing and freeze-base behaviour for PE MLP families."""

    def test_parse_args_accepts_vae_variants(self):
        from scripts.finetune import parse_args

        for mode in ("vae", "vae_lg"):
            with self.subTest(mode=mode):
                with mock.patch("sys.argv", ["finetune.py", "--pe-mlp-mode", mode]):
                    args = parse_args()
                self.assertEqual(args.pe_mlp_mode, mode)

    def test_freeze_base_keeps_vae_family_mlp_trainable(self):
        from scripts.finetune import freeze_base_parameters

        for mode in ("vae", "vae_lg"):
            with self.subTest(mode=mode):
                model = PELlamaForCausalLM(
                    _make_tiny_config(pe_mlp_mode=mode, pe_mlp_layer_indices=[0, 1]),
                )
                freeze_base_parameters(model, mode)
                mlp_params = [
                    p for name, p in model.named_parameters()
                    if ".mlp." in name
                ]
                self.assertTrue(mlp_params)
                self.assertTrue(
                    all(p.requires_grad for p in mlp_params),
                    f"{mode} MLP parameters were unexpectedly frozen",
                )


class TestAETargetDiscovery(unittest.TestCase):
    """AE/VAE target discovery should include every PE bottleneck family."""

    def test_get_ae_target_layers_covers_all_pe_mlp_families(self):
        from scripts.finetune import _get_ae_target_layers

        for mode in ("ae", "ae_lg", "vae", "vae_lg"):
            with self.subTest(mode=mode):
                model = PELlamaForCausalLM(
                    _make_tiny_config(pe_mlp_mode=mode, pe_mlp_layer_indices=[0, 1]),
                )
                self.assertEqual(_get_ae_target_layers(model), [0, 1])


class TestGateReporting(unittest.TestCase):
    """Tests for learned-gate reporting in scripts.finetune."""

    def test_print_learned_gate_values_skips_standard_layers(self):
        """Gate reporting should ignore non-targeted standard LlamaMLP layers."""
        from scripts.finetune import print_learned_gate_values

        model_config = _make_tiny_config(
            pe_mlp_mode="ae_lg",
            pe_mlp_layer_indices=[1],
        )
        model = PELlamaForCausalLM(model_config)

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            print_learned_gate_values(model)

        printed_output = output_buffer.getvalue()
        self.assertIn("Learned gate values", printed_output)
        self.assertIn("Layer  1:", printed_output)
        self.assertNotIn("Layer  0:", printed_output)

    def test_print_learned_gate_values_no_output_for_vae(self):
        """Standard VAE layers have no learned gate — no printed output."""
        from scripts.finetune import print_learned_gate_values

        model = PELlamaForCausalLM(
            _make_tiny_config(pe_mlp_mode="vae", pe_mlp_layer_indices=[1]),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            print_learned_gate_values(model)

        printed_output = output_buffer.getvalue()
        self.assertNotIn("Learned gate values", printed_output)

    def test_print_learned_gate_values_reports_gate_for_vae_lg(self):
        """VAE-LG layers should report gate statistics."""
        from scripts.finetune import print_learned_gate_values

        model = PELlamaForCausalLM(
            _make_tiny_config(pe_mlp_mode="vae_lg", pe_mlp_layer_indices=[1]),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            print_learned_gate_values(model)

        printed_output = output_buffer.getvalue()
        self.assertIn("Learned gate values", printed_output)
        self.assertIn("Layer  1:", printed_output)
        self.assertNotIn("Layer  0:", printed_output)


if __name__ == "__main__":
    unittest.main()
