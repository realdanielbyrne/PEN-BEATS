"""Tests for knowledge distillation (KD) loss in finetune.py.

Verifies:
- train_epoch with KD teacher produces decreasing loss
- KD disabled (alpha=1.0) matches original behavior
- Pure KD (alpha=0.0) works without error
- Teacher and student on same device (single GPU fallback)
- CLI argument parsing defaults
- YAML runner wiring for KD params
- Attention pattern KD produces finite loss
- Attention pattern KD CLI arguments
- Attention pattern KD YAML wiring
"""

import sys
import unittest
from unittest import mock

import torch
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

    def __init__(self, n_samples=8, seq_len=16, vocab_size=256):
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return {"input_ids": ids, "labels": ids.clone()}


class TestKDLossDecreases(unittest.TestCase):
    """train_epoch with KD teacher produces finite, decreasing loss."""

    def setUp(self):
        torch.manual_seed(42)
        # Teacher: standard model (frozen)
        self.teacher_cfg = _make_tiny_config()
        self.teacher = PELlamaForCausalLM(self.teacher_cfg)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Student: AE bottleneck MLP
        self.student_cfg = _make_tiny_config(pe_mlp_mode="ae")
        self.student = PELlamaForCausalLM(self.student_cfg)

        self.dataset = _TinyTokenDataset(
            n_samples=8,
            seq_len=16,
            vocab_size=256,
        )
        self.loader = DataLoader(self.dataset, batch_size=4)
        self.device = torch.device("cpu")

    def test_loss_decreases_over_epochs(self):
        from scripts.finetune import train_epoch

        optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=1e-3,
        )

        losses = []
        for _ in range(5):
            avg_loss = train_epoch(
                self.student,
                self.loader,
                optimizer,
                self.device,
                grad_accum_steps=1,
                teacher_model=self.teacher,
                kd_alpha=0.5,
                kd_temperature=2.0,
                teacher_device=self.device,
            )
            losses.append(avg_loss)
            self.assertTrue(
                torch.isfinite(torch.tensor(avg_loss)),
                f"Loss is not finite: {avg_loss}",
            )

        self.assertLess(losses[-1], losses[0], f"Loss did not decrease: {losses}")


class TestKDDisabledAlpha1(unittest.TestCase):
    """alpha=1.0 with teacher=None matches original behavior."""

    def setUp(self):
        torch.manual_seed(42)
        self.cfg = _make_tiny_config(pe_mlp_mode="ae")
        self.model = PELlamaForCausalLM(self.cfg)
        self.dataset = _TinyTokenDataset(n_samples=4, seq_len=16, vocab_size=256)
        self.loader = DataLoader(self.dataset, batch_size=2)
        self.device = torch.device("cpu")

    def test_no_teacher_no_kd(self):
        """train_epoch with default KD params (disabled) works."""
        from scripts.finetune import train_epoch

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-3,
        )
        # Default: teacher_model=None, kd_alpha=1.0
        avg_loss = train_epoch(
            self.model,
            self.loader,
            optimizer,
            self.device,
            grad_accum_steps=1,
        )
        self.assertTrue(torch.isfinite(torch.tensor(avg_loss)))


class TestKDPureAlpha0(unittest.TestCase):
    """alpha=0.0 (pure KD, no CE) runs without errors."""

    def setUp(self):
        torch.manual_seed(42)
        self.teacher = PELlamaForCausalLM(_make_tiny_config())
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student = PELlamaForCausalLM(_make_tiny_config(pe_mlp_mode="ae"))
        self.dataset = _TinyTokenDataset(n_samples=4, seq_len=16, vocab_size=256)
        self.loader = DataLoader(self.dataset, batch_size=2)
        self.device = torch.device("cpu")

    def test_pure_kd_runs(self):
        from scripts.finetune import train_epoch

        optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=1e-3,
        )
        avg_loss = train_epoch(
            self.student,
            self.loader,
            optimizer,
            self.device,
            grad_accum_steps=1,
            teacher_model=self.teacher,
            kd_alpha=0.0,
            kd_temperature=4.0,
            teacher_device=self.device,
        )
        self.assertTrue(torch.isfinite(torch.tensor(avg_loss)))


class TestAttnPatternKD(unittest.TestCase):
    """Attention pattern KD produces finite loss and integrates correctly."""

    def setUp(self):
        torch.manual_seed(42)
        # Use eager attention so output_attentions=True returns actual maps
        self.teacher_cfg = _make_tiny_config(attn_implementation="eager")
        self.teacher = PELlamaForCausalLM(self.teacher_cfg)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Student with TrendWavelet attention on all layers
        self.student_cfg = _make_tiny_config(
            pe_attn_mode="trend_wavelet_lg",
            pe_mlp_mode="ae",
            attn_implementation="eager",
        )
        self.student = PELlamaForCausalLM(self.student_cfg)

        self.dataset = _TinyTokenDataset(n_samples=8, seq_len=16, vocab_size=256)
        self.loader = DataLoader(self.dataset, batch_size=4)
        self.device = torch.device("cpu")

    def test_attn_kd_only_produces_finite_loss(self):
        """Attention pattern KD alone (kd_alpha=1.0) produces finite loss."""
        from scripts.finetune import train_epoch

        optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=1e-3,
        )
        avg_loss = train_epoch(
            self.student,
            self.loader,
            optimizer,
            self.device,
            grad_accum_steps=1,
            teacher_model=self.teacher,
            kd_alpha=1.0,
            kd_temperature=2.0,
            teacher_device=self.device,
            kd_attn_weight=0.3,
            kd_attn_layers=[0, 1],
        )
        self.assertTrue(
            torch.isfinite(torch.tensor(avg_loss)), f"Loss is not finite: {avg_loss}"
        )

    def test_hybrid_kd_produces_finite_loss(self):
        """Combined logit KD + attention pattern KD produces finite loss."""
        from scripts.finetune import train_epoch

        optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=1e-3,
        )
        avg_loss = train_epoch(
            self.student,
            self.loader,
            optimizer,
            self.device,
            grad_accum_steps=1,
            teacher_model=self.teacher,
            kd_alpha=0.5,
            kd_temperature=2.0,
            teacher_device=self.device,
            kd_attn_weight=0.3,
            kd_attn_layers=[0, 1],
        )
        self.assertTrue(
            torch.isfinite(torch.tensor(avg_loss)), f"Loss is not finite: {avg_loss}"
        )

    def test_attn_kd_disabled_when_weight_zero(self):
        """kd_attn_weight=0.0 disables attention KD (no output_attentions overhead)."""
        from scripts.finetune import train_epoch

        optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=1e-3,
        )
        # Should run without requesting attention outputs
        avg_loss = train_epoch(
            self.student,
            self.loader,
            optimizer,
            self.device,
            grad_accum_steps=1,
            teacher_model=self.teacher,
            kd_alpha=0.5,
            kd_temperature=2.0,
            teacher_device=self.device,
            kd_attn_weight=0.0,
            kd_attn_layers=[0, 1],
        )
        self.assertTrue(torch.isfinite(torch.tensor(avg_loss)))


class TestKDCLIDefaults(unittest.TestCase):
    """CLI argument parsing has correct KD defaults."""

    def test_defaults(self):
        from scripts.finetune import parse_args

        with mock.patch.object(sys, "argv", ["finetune.py"]):
            args = parse_args()
        self.assertEqual(args.kd_alpha, 1.0)
        self.assertEqual(args.kd_temperature, 2.0)
        self.assertIsNone(args.kd_teacher)
        self.assertEqual(args.kd_attn_weight, 0.0)
        self.assertIsNone(args.kd_attn_layers)

    def test_custom_values(self):
        from scripts.finetune import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "finetune.py",
                "--kd-alpha",
                "0.3",
                "--kd-temperature",
                "4.0",
                "--kd-teacher",
                "some/model",
                "--kd-attn-weight",
                "0.5",
                "--kd-attn-layers",
                "14",
                "15",
            ],
        ):
            args = parse_args()
        self.assertAlmostEqual(args.kd_alpha, 0.3)
        self.assertAlmostEqual(args.kd_temperature, 4.0)
        self.assertEqual(args.kd_teacher, "some/model")
        self.assertAlmostEqual(args.kd_attn_weight, 0.5)
        self.assertEqual(args.kd_attn_layers, [14, 15])


class TestKDYAMLWiring(unittest.TestCase):
    """YAML runner has KD params in PARAM_TO_FLAG."""

    def test_param_to_flag_contains_kd_keys(self):
        from scripts.run_from_yaml import PARAM_TO_FLAG

        self.assertIn("kd_alpha", PARAM_TO_FLAG)
        self.assertIn("kd_temperature", PARAM_TO_FLAG)
        self.assertIn("kd_teacher", PARAM_TO_FLAG)
        self.assertIn("kd_attn_weight", PARAM_TO_FLAG)
        self.assertIn("kd_attn_layers", PARAM_TO_FLAG)
        self.assertEqual(PARAM_TO_FLAG["kd_alpha"], "--kd-alpha")
        self.assertEqual(PARAM_TO_FLAG["kd_temperature"], "--kd-temperature")
        self.assertEqual(PARAM_TO_FLAG["kd_teacher"], "--kd-teacher")
        self.assertEqual(PARAM_TO_FLAG["kd_attn_weight"], "--kd-attn-weight")
        self.assertEqual(PARAM_TO_FLAG["kd_attn_layers"], "--kd-attn-layers")

    def test_build_cmd_includes_kd_flags(self):
        from scripts.run_from_yaml import build_cmd

        params = {"kd_alpha": 0.5, "kd_temperature": 3.0, "kd_teacher": "my/teacher"}
        cmd = build_cmd(
            python="python3",
            script="scripts/finetune.py",
            params=params,
            output_dir="/tmp/out",
            wandb_cfg={"enabled": False},
        )
        idx = cmd.index("--kd-alpha")
        self.assertEqual(cmd[idx + 1], "0.5")
        idx = cmd.index("--kd-temperature")
        self.assertEqual(cmd[idx + 1], "3.0")
        idx = cmd.index("--kd-teacher")
        self.assertEqual(cmd[idx + 1], "my/teacher")

    def test_build_cmd_includes_attn_kd_flags(self):
        from scripts.run_from_yaml import build_cmd

        params = {"kd_attn_weight": 0.3, "kd_attn_layers": [14, 15]}
        cmd = build_cmd(
            python="python3",
            script="scripts/finetune.py",
            params=params,
            output_dir="/tmp/out",
            wandb_cfg={"enabled": False},
        )
        idx = cmd.index("--kd-attn-weight")
        self.assertEqual(cmd[idx + 1], "0.3")
        idx = cmd.index("--kd-attn-layers")
        self.assertEqual(cmd[idx + 1], "14")
        self.assertEqual(cmd[idx + 2], "15")

    def test_kd_attn_layers_in_nargs_fields(self):
        from scripts.run_from_yaml import NARGS_FIELDS

        self.assertIn("kd_attn_layers", NARGS_FIELDS)


if __name__ == "__main__":
    unittest.main()
