"""Tests for the wave pipeline orchestrator in run_wave_pipeline.py.

Covers:
- load_pipeline_config: YAML parsing, validation (missing waves, duplicate names,
  forward references, missing name)
- resolve_resume_from: wave name resolution, absolute path passthrough, unknown name
- merge_wave_params: defaults applied, wave overrides, name stripped
- Wave sentinel: is_wave_complete, write/read/remove sentinel
- invalidate_from_wave: target + subsequent invalidation
- Dry-run integration: command building with resume_from and ae_teacher auto-wiring
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# run_wave_pipeline lives under scripts/, not an installed package.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_wave_pipeline import (  # noqa: E402
    _WAVE_SENTINEL,
    invalidate_from_wave,
    is_wave_complete,
    load_pipeline_config,
    merge_wave_params,
    read_wave_sentinel,
    remove_wave_sentinel,
    resolve_resume_from,
    write_wave_sentinel,
)


# ---------------------------------------------------------------------------
# Helper: write a YAML string to a temp file and return its path
# ---------------------------------------------------------------------------

def _write_yaml(tmpdir: str, content: str) -> str:
    path = os.path.join(tmpdir, "test_pipeline.yaml")
    with open(path, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# TestLoadPipelineConfig
# ---------------------------------------------------------------------------

class TestLoadPipelineConfig(unittest.TestCase):

    def test_valid_yaml_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, """
pipeline:
  name: "test_pipeline"
defaults:
  lr: 0.001
waves:
  - name: "wave1"
    pe_mlp_layer_indices: [14, 15]
  - name: "wave2"
    resume_from: "wave1"
    pe_mlp_layer_indices: [12, 13, 14, 15]
""")
            pipeline, defaults, waves = load_pipeline_config(path)
            self.assertEqual(pipeline["name"], "test_pipeline")
            self.assertEqual(defaults["lr"], 0.001)
            self.assertEqual(len(waves), 2)
            self.assertEqual(waves[0]["name"], "wave1")
            self.assertEqual(waves[1]["resume_from"], "wave1")

    def test_missing_waves_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, """
pipeline:
  name: "test"
defaults: {}
""")
            with self.assertRaises(ValueError, msg="non-empty 'waves'"):
                load_pipeline_config(path)

    def test_empty_waves_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, """
pipeline:
  name: "test"
waves: []
""")
            with self.assertRaises(ValueError):
                load_pipeline_config(path)

    def test_duplicate_wave_names_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, """
waves:
  - name: "wave1"
  - name: "wave1"
""")
            with self.assertRaises(ValueError, msg="Duplicate"):
                load_pipeline_config(path)

    def test_missing_wave_name_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, """
waves:
  - pe_mlp_layer_indices: [14, 15]
""")
            with self.assertRaises(ValueError, msg="missing"):
                load_pipeline_config(path)

    def test_forward_reference_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, """
waves:
  - name: "wave1"
    resume_from: "wave2"
  - name: "wave2"
""")
            with self.assertRaises(ValueError, msg="preceding"):
                load_pipeline_config(path)

    def test_absolute_path_resume_from_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_yaml(tmpdir, """
waves:
  - name: "wave1"
    resume_from: "/absolute/path/to/model"
""")
            _, _, waves = load_pipeline_config(path)
            self.assertEqual(waves[0]["resume_from"], "/absolute/path/to/model")


# ---------------------------------------------------------------------------
# TestResolveResumeFrom
# ---------------------------------------------------------------------------

class TestResolveResumeFrom(unittest.TestCase):

    def test_none_returns_none(self):
        self.assertIsNone(resolve_resume_from(None, {}))

    def test_wave_name_resolved(self):
        dirs = {"wave1": "/models/wave1", "wave2": "/models/wave2"}
        self.assertEqual(resolve_resume_from("wave1", dirs), "/models/wave1")

    def test_absolute_path_passthrough(self):
        dirs = {"wave1": "/models/wave1"}
        self.assertEqual(
            resolve_resume_from("/ext/checkpoint", dirs),
            "/ext/checkpoint",
        )

    def test_unknown_name_raises(self):
        dirs = {"wave1": "/models/wave1"}
        with self.assertRaises(ValueError, msg="not found"):
            resolve_resume_from("wave_unknown", dirs)


# ---------------------------------------------------------------------------
# TestMergeWaveParams
# ---------------------------------------------------------------------------

class TestMergeWaveParams(unittest.TestCase):

    def test_defaults_applied(self):
        defaults = {"lr": 0.001, "epochs": 5, "seed": 42}
        wave = {"name": "wave1", "pe_mlp_layer_indices": [14, 15]}
        merged = merge_wave_params(defaults, wave)
        self.assertEqual(merged["lr"], 0.001)
        self.assertEqual(merged["epochs"], 5)
        self.assertEqual(merged["pe_mlp_layer_indices"], [14, 15])

    def test_wave_override_wins(self):
        defaults = {"lr": 0.001, "epochs": 5}
        wave = {"name": "wave2", "lr": 0.0001}
        merged = merge_wave_params(defaults, wave)
        self.assertEqual(merged["lr"], 0.0001)
        self.assertEqual(merged["epochs"], 5)

    def test_name_stripped(self):
        defaults = {"lr": 0.001}
        wave = {"name": "wave1", "resume_from": "wave0"}
        merged = merge_wave_params(defaults, wave)
        self.assertNotIn("name", merged)
        self.assertNotIn("resume_from", merged)


# ---------------------------------------------------------------------------
# TestWaveSentinel
# ---------------------------------------------------------------------------

class TestWaveSentinel(unittest.TestCase):

    def test_not_complete_initially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(is_wave_complete(tmpdir))

    def test_write_and_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"wave_name": "wave1", "status": "ok"}
            write_wave_sentinel(tmpdir, data)
            self.assertTrue(is_wave_complete(tmpdir))

            loaded = read_wave_sentinel(tmpdir)
            self.assertEqual(loaded["wave_name"], "wave1")

    def test_remove_sentinel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_wave_sentinel(tmpdir, {"status": "ok"})
            self.assertTrue(is_wave_complete(tmpdir))

            remove_wave_sentinel(tmpdir)
            self.assertFalse(is_wave_complete(tmpdir))

    def test_remove_nonexistent_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            remove_wave_sentinel(tmpdir)

    def test_write_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_dir = os.path.join(tmpdir, "a", "b", "c")
            write_wave_sentinel(deep_dir, {"status": "ok"})
            self.assertTrue(is_wave_complete(deep_dir))


# ---------------------------------------------------------------------------
# TestInvalidateFromWave
# ---------------------------------------------------------------------------

class TestInvalidateFromWave(unittest.TestCase):

    def test_invalidates_target_and_subsequent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            waves = [
                {"name": "w1"}, {"name": "w2"},
                {"name": "w3"}, {"name": "w4"},
            ]
            dirs = {}
            for w in waves:
                d = os.path.join(tmpdir, w["name"])
                dirs[w["name"]] = d
                write_wave_sentinel(d, {"status": "ok"})

            # All 4 should be complete
            for d in dirs.values():
                self.assertTrue(is_wave_complete(d))

            # Invalidate from w2 onward
            invalidate_from_wave(waves, "w2", dirs)

            self.assertTrue(is_wave_complete(dirs["w1"]))
            self.assertFalse(is_wave_complete(dirs["w2"]))
            self.assertFalse(is_wave_complete(dirs["w3"]))
            self.assertFalse(is_wave_complete(dirs["w4"]))

    def test_unknown_wave_raises(self):
        waves = [{"name": "w1"}]
        dirs = {"w1": "/tmp/w1"}
        with self.assertRaises(ValueError, msg="not found"):
            invalidate_from_wave(waves, "w_unknown", dirs)


# ---------------------------------------------------------------------------
# TestBuildCmdIntegration
# ---------------------------------------------------------------------------

class TestBuildCmdIntegration(unittest.TestCase):
    """Test that merged wave params produce correct finetune.py commands."""

    def test_resume_from_in_cmd(self):
        from run_from_yaml import build_cmd

        params = {
            "pe_mlp_mode": "ae_lg",
            "ae_latent_dim": 512,
            "resume_from": "/models/wave1",
            "pe_mlp_layer_indices": [12, 13, 14, 15],
            "epochs": 5,
            "lr": 0.0001,
            "freeze_base": True,
        }
        cmd = build_cmd("python", "finetune.py", params, "/out/wave2",
                        {"enabled": False})

        self.assertIn("--resume-from", cmd)
        idx = cmd.index("--resume-from")
        self.assertEqual(cmd[idx + 1], "/models/wave1")

        self.assertIn("--pe-mlp-layer-indices", cmd)
        idx2 = cmd.index("--pe-mlp-layer-indices")
        self.assertEqual(cmd[idx2 + 1:idx2 + 5], ["12", "13", "14", "15"])

        self.assertIn("--freeze-base", cmd)

    def test_ae_teacher_in_cmd(self):
        from run_from_yaml import build_cmd

        params = {
            "pe_mlp_mode": "ae_lg",
            "ae_teacher": "/models/wave1",
            "ae_pretrain_epochs": 10,
        }
        cmd = build_cmd("python", "finetune.py", params, "/out/wave2",
                        {"enabled": False})

        self.assertIn("--ae-teacher", cmd)
        idx = cmd.index("--ae-teacher")
        self.assertEqual(cmd[idx + 1], "/models/wave1")


# ---------------------------------------------------------------------------
# TestDryRun
# ---------------------------------------------------------------------------

class TestDryRun(unittest.TestCase):
    """Integration test: dry-run a 3-wave pipeline and verify output."""

    def test_dry_run_prints_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = """
pipeline:
  name: "test_dry"
  python: "python3"
defaults:
  pe_mlp_mode: "ae_lg"
  ae_latent_dim: 256
  ae_pretrain_epochs: 5
  epochs: 3
  lr: 0.0001
  dtype: "bfloat16"
  freeze_base: true
waves:
  - name: "w1"
    pe_mlp_layer_indices: [14, 15]
  - name: "w2"
    resume_from: "w1"
    pe_mlp_layer_indices: [12, 13, 14, 15]
  - name: "w3"
    resume_from: "w2"
    pe_mlp_layer_indices: [8, 9, 10, 11, 12, 13, 14, 15]
"""
            yaml_path = _write_yaml(tmpdir, yaml_content)

            from run_wave_pipeline import main
            from io import StringIO
            from contextlib import redirect_stdout

            with mock.patch("sys.argv", ["prog", yaml_path, "--dry-run"]):
                buf = StringIO()
                with redirect_stdout(buf):
                    main()

            output = buf.getvalue()

            # All 3 waves should have CMD lines
            self.assertEqual(output.count("CMD:"), 3)

            # Wave 2 should have --resume-from pointing to w1's output dir
            self.assertIn("--resume-from", output)

            # Wave 2+ should have --ae-teacher auto-wired
            self.assertIn("--ae-teacher", output)

    def test_dry_run_no_ae_teacher_on_wave1(self):
        """Wave 1 should NOT get --ae-teacher (no prior wave)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = """
pipeline:
  name: "test_no_teacher"
  python: "python3"
defaults:
  pe_mlp_mode: "ae_lg"
  ae_pretrain_epochs: 5
waves:
  - name: "w1"
    pe_mlp_layer_indices: [14, 15]
  - name: "w2"
    resume_from: "w1"
    pe_mlp_layer_indices: [12, 13, 14, 15]
"""
            yaml_path = _write_yaml(tmpdir, yaml_content)

            from run_wave_pipeline import main
            from io import StringIO
            from contextlib import redirect_stdout

            with mock.patch("sys.argv", ["prog", yaml_path, "--dry-run"]):
                buf = StringIO()
                with redirect_stdout(buf):
                    main()

            output = buf.getvalue()
            lines = output.split("\n")

            # Find CMD lines
            cmd_lines = [l for l in lines if "CMD:" in l]
            self.assertEqual(len(cmd_lines), 2)

            # Wave 1 CMD should NOT have --ae-teacher
            self.assertNotIn("--ae-teacher", cmd_lines[0])

            # Wave 2 CMD should have --ae-teacher
            self.assertIn("--ae-teacher", cmd_lines[1])


# ---------------------------------------------------------------------------
# TestAeTeacherAutoWiring
# ---------------------------------------------------------------------------

class TestAeTeacherAutoWiring(unittest.TestCase):
    """Verify ae_teacher is set correctly based on resume_from and ae_pretrain_epochs."""

    def test_no_ae_pretrain_no_teacher(self):
        """When ae_pretrain_epochs=0, ae_teacher should NOT be auto-wired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = """
pipeline:
  name: "test_no_pt"
  python: "python3"
defaults:
  pe_mlp_mode: "ae_lg"
  ae_pretrain_epochs: 0
  epochs: 3
waves:
  - name: "w1"
    pe_mlp_layer_indices: [14, 15]
  - name: "w2"
    resume_from: "w1"
    pe_mlp_layer_indices: [12, 13, 14, 15]
"""
            yaml_path = _write_yaml(tmpdir, yaml_content)

            from run_wave_pipeline import main
            from io import StringIO
            from contextlib import redirect_stdout

            with mock.patch("sys.argv", ["prog", yaml_path, "--dry-run"]):
                buf = StringIO()
                with redirect_stdout(buf):
                    main()

            output = buf.getvalue()
            # Neither wave should have --ae-teacher when ae_pretrain_epochs=0
            self.assertNotIn("--ae-teacher", output)


class TestBuildCmdAePretrainLrFlags(unittest.TestCase):
    """Verify new AE pretrain LR scheduler flags are forwarded correctly."""

    def test_ae_pretrain_lr_flags_in_cmd(self):
        from run_from_yaml import build_cmd

        params = {
            "pe_mlp_mode": "ae_lg",
            "ae_pretrain_epochs": 10,
            "ae_pretrain_lr": 0.001,
            "ae_pretrain_scheduler": "exponential",
            "ae_pretrain_lr_warmup": 2,
            "ae_pretrain_gamma": 0.85,
        }
        cmd = build_cmd("python", "finetune.py", params, "/out/test",
                        {"enabled": False})

        self.assertIn("--ae-pretrain-lr", cmd)
        idx = cmd.index("--ae-pretrain-lr")
        self.assertEqual(cmd[idx + 1], "0.001")

        self.assertIn("--ae-pretrain-scheduler", cmd)
        idx = cmd.index("--ae-pretrain-scheduler")
        self.assertEqual(cmd[idx + 1], "exponential")

        self.assertIn("--ae-pretrain-lr-warmup", cmd)
        idx = cmd.index("--ae-pretrain-lr-warmup")
        self.assertEqual(cmd[idx + 1], "2")

        self.assertIn("--ae-pretrain-gamma", cmd)
        idx = cmd.index("--ae-pretrain-gamma")
        self.assertEqual(cmd[idx + 1], "0.85")


class TestCleanupModelFiles(unittest.TestCase):
    """Verify _cleanup_model_files removes weights but preserves sentinel."""

    def test_removes_weights_preserves_sentinel(self):
        from run_wave_pipeline import _cleanup_model_files, write_wave_sentinel

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake model files
            (Path(tmpdir) / "model.safetensors").write_text("fake_weights")
            (Path(tmpdir) / "model-00001-of-00002.safetensors").write_text("shard")
            (Path(tmpdir) / "config.json").write_text("{}")
            (Path(tmpdir) / "tokenizer.json").write_text("{}")
            (Path(tmpdir) / "special_tokens_map.json").write_text("{}")
            write_wave_sentinel(tmpdir, {"status": "ok"})

            _cleanup_model_files(tmpdir)

            # Sentinel and config preserved
            self.assertTrue((Path(tmpdir) / "wave_complete.json").exists())
            self.assertTrue((Path(tmpdir) / "config.json").exists())
            # Model weights removed
            self.assertFalse((Path(tmpdir) / "model.safetensors").exists())
            self.assertFalse(
                (Path(tmpdir) / "model-00001-of-00002.safetensors").exists()
            )
            # Tokenizer files removed
            self.assertFalse((Path(tmpdir) / "tokenizer.json").exists())
            self.assertFalse((Path(tmpdir) / "special_tokens_map.json").exists())

    def test_nonexistent_dir_is_noop(self):
        from run_wave_pipeline import _cleanup_model_files

        # Should not raise
        _cleanup_model_files("/nonexistent/path/12345")


if __name__ == "__main__":
    unittest.main()
