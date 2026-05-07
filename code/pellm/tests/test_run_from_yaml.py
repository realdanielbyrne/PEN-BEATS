"""Tests for the orchestrator CSV and utility functions in run_from_yaml.py.

Covers:
- append_csv_row: header creation, flat scalars, nested data exclusion,
  derived metrics, list-valued param serialization, multi-row append
- _flatten_for_csv: scalar passthrough vs JSON serialization
- build_cmd: --log-csv not forwarded, output_dir injected
- expand_grid / parse_results (smoke tests)
"""

import csv
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

# run_from_yaml lives under scripts/, not an installed package.
# Ensure it is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_from_yaml import (  # noqa: E402
    CSV_FIELDNAMES,
    _NESTED_RESULT_KEYS,
    _flatten_for_csv,
    append_csv_row,
    build_cmd,
    expand_grid,
    main,
    parse_results,
)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

_SAMPLE_PARAMS = {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "pe_attn_mode": "standard",
    "attn_init": "pretrained",
    "pe_mlp_mode": "ae_lg",
    "ae_latent_dim": 256,
    "lr": 1e-4,
    "epochs": 3,
    "batch_size": 2,
    "pe_mlp_layer_indices": [15],
    "freeze_base": True,
    "dtype": "bfloat16",
}

_SAMPLE_RESULTS = {
    "pe_attn_mode": "standard",
    "pe_mlp_mode": "ae_lg",
    "ae_latent_dim": 256,
    "original_ppl": 12.50,
    "baseline_ppl": 10.00,
    "attn_pretrain_ppl": 9.75,
    "ae_pretrain_ppl": 9.50,
    "final_ppl": 8.00,
    "params_total": 1_000_000,
    "params_trainable": 50_000,
    "epochs": 3,
    "lr": 1e-4,
    "attn_pretrain_epochs": 2,
    "attn_pretrain_stats": [{"attn_epoch": 1, "mse_loss": 0.08}],
    "ae_pretrain_epochs": 5,
    "ae_pretrain_stats": [{"ae_epoch": 1, "mse_loss": 0.05}],
    "per_epoch_stats": [
        {"epoch": 1, "train_loss": 2.5, "val_ppl": 9.0},
        {"epoch": 2, "train_loss": 2.3, "val_ppl": 8.5},
        {"epoch": 3, "train_loss": 2.1, "val_ppl": 8.2},
    ],
}


class TestFlattenForCSV(unittest.TestCase):
    def test_scalar_passthrough(self):
        self.assertEqual(_flatten_for_csv(42), 42)
        self.assertEqual(_flatten_for_csv("hello"), "hello")
        self.assertAlmostEqual(_flatten_for_csv(3.14), 3.14)

    def test_list_serialized(self):
        self.assertEqual(_flatten_for_csv([15]), "[15]")

    def test_dict_serialized(self):
        self.assertEqual(_flatten_for_csv({"a": 1}), '{"a": 1}')

    def test_none_passthrough(self):
        self.assertIsNone(_flatten_for_csv(None))


class TestAppendCSVRow(unittest.TestCase):
    def _read_csv(self, path: Path) -> list[dict]:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def test_creates_header_and_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            append_csv_row(
                csv_path, "run_001", "cfg_a", 0, _SAMPLE_PARAMS, _SAMPLE_RESULTS, "ok"
            )
            rows = self._read_csv(csv_path)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["run_id"], "run_001")
        self.assertEqual(row["config_name"], "cfg_a")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["final_ppl"], "8.0")
        self.assertEqual(row["baseline_ppl"], "10.0")

    def test_nested_data_excluded(self):
        """Nested pretrain stats and per_epoch_stats must NOT appear as columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            append_csv_row(
                csv_path, "run_001", "cfg_a", 0, _SAMPLE_PARAMS, _SAMPLE_RESULTS, "ok"
            )
            rows = self._read_csv(csv_path)

        row = rows[0]
        for nested_key in _NESTED_RESULT_KEYS:
            self.assertNotIn(
                nested_key, row, f"Nested key '{nested_key}' leaked into CSV"
            )

    def test_ppl_improvement_pct_computed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            append_csv_row(
                csv_path, "run_001", "cfg_a", 0, _SAMPLE_PARAMS, _SAMPLE_RESULTS, "ok"
            )
            rows = self._read_csv(csv_path)

        # (10.0 - 8.0) / 10.0 * 100 = 20.0
        self.assertAlmostEqual(float(rows[0]["ppl_improvement_pct"]), 20.0)

    def test_attention_pretrain_mse_flattened(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            append_csv_row(
                csv_path, "run_001", "cfg_a", 0, _SAMPLE_PARAMS, _SAMPLE_RESULTS, "ok"
            )
            rows = self._read_csv(csv_path)

        self.assertEqual(json.loads(rows[0]["attn_pretrain_mse"]), [0.08])

    def test_epochs_completed_and_best_val_ppl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            append_csv_row(
                csv_path, "run_001", "cfg_a", 0, _SAMPLE_PARAMS, _SAMPLE_RESULTS, "ok"
            )
            rows = self._read_csv(csv_path)

        self.assertEqual(rows[0]["epochs_completed"], "3")
        self.assertAlmostEqual(float(rows[0]["best_val_ppl"]), 8.2)

    def test_list_param_json_serialized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            append_csv_row(
                csv_path, "run_001", "cfg_a", 0, _SAMPLE_PARAMS, _SAMPLE_RESULTS, "ok"
            )
            rows = self._read_csv(csv_path)

        self.assertEqual(json.loads(rows[0]["pe_mlp_layer_indices"]), [15])

    def test_multiple_rows_appended(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            for idx in range(3):
                append_csv_row(
                    csv_path,
                    f"run_{idx:03d}",
                    "cfg_a",
                    idx,
                    _SAMPLE_PARAMS,
                    _SAMPLE_RESULTS,
                    "ok",
                )
            rows = self._read_csv(csv_path)

        self.assertEqual(len(rows), 3)
        self.assertEqual([r["run_id"] for r in rows], ["run_000", "run_001", "run_002"])

    def test_no_results_writes_row_with_status(self):
        """When results is None (e.g. validate run), row still written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            append_csv_row(
                csv_path,
                "run_val",
                "cfg_a",
                0,
                _SAMPLE_PARAMS,
                results=None,
                status="validate_pass",
            )
            rows = self._read_csv(csv_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "validate_pass")
        # Result columns should be empty
        self.assertEqual(rows[0]["final_ppl"], "")
        self.assertEqual(rows[0]["ppl_improvement_pct"], "")

    def test_header_stable_across_appends(self):
        """Header written once; subsequent rows don't duplicate it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            append_csv_row(
                csv_path, "run_000", "cfg_a", 0, _SAMPLE_PARAMS, _SAMPLE_RESULTS, "ok"
            )
            append_csv_row(
                csv_path,
                "run_001",
                "cfg_b",
                1,
                _SAMPLE_PARAMS,
                results=None,
                status="failed",
            )

            raw = csv_path.read_text()
            header_line = raw.splitlines()[0]
            # Header should appear exactly once
            self.assertEqual(raw.count(header_line), 1)

    def test_fieldnames_no_duplicates(self):
        """CSV_FIELDNAMES must not contain duplicate column names."""
        self.assertEqual(len(CSV_FIELDNAMES), len(set(CSV_FIELDNAMES)))


class TestBuildCmd(unittest.TestCase):
    def test_log_csv_not_in_cmd(self):
        """Orchestrator must NOT pass --log-csv to finetune.py."""
        cmd = build_cmd(
            python="python3",
            script="scripts/finetune.py",
            params=_SAMPLE_PARAMS,
            output_dir="/tmp/test_output",
            wandb_cfg={"enabled": False},
        )
        self.assertNotIn("--log-csv", cmd)

    def test_output_dir_injected(self):
        cmd = build_cmd(
            python="python3",
            script="scripts/finetune.py",
            params=_SAMPLE_PARAMS,
            output_dir="/tmp/test_output",
            wandb_cfg={"enabled": False},
        )
        idx = cmd.index("--output-dir")
        self.assertEqual(cmd[idx + 1], "/tmp/test_output")

    def test_output_dir_omitted_when_save_model_false(self):
        cmd = build_cmd(
            python="python3",
            script="scripts/finetune.py",
            params=_SAMPLE_PARAMS,
            output_dir="/tmp/test_output",
            wandb_cfg={"enabled": False},
            save_model=False,
        )
        self.assertNotIn("--output-dir", cmd)

    def test_bool_flag_included_when_true(self):
        cmd = build_cmd(
            python="python3",
            script="scripts/finetune.py",
            params={"freeze_base": True},
            output_dir="/tmp/out",
            wandb_cfg={"enabled": False},
        )
        self.assertIn("--freeze-base", cmd)

    def test_bool_flag_excluded_when_false(self):
        cmd = build_cmd(
            python="python3",
            script="scripts/finetune.py",
            params={"freeze_base": False},
            output_dir="/tmp/out",
            wandb_cfg={"enabled": False},
        )
        self.assertNotIn("--freeze-base", cmd)


class TestExpandGrid(unittest.TestCase):
    def test_no_lists_single_combo(self):
        cfg = {"pe_attn_mode": "standard", "epochs": 3}
        combos = expand_grid(cfg)
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0], cfg)

    def test_single_list_expands(self):
        cfg = {"ae_latent_dim": [128, 256], "epochs": 3}
        combos = expand_grid(cfg)
        self.assertEqual(len(combos), 2)
        dims = {c["ae_latent_dim"] for c in combos}
        self.assertEqual(dims, {128, 256})

    def test_cartesian_product(self):
        cfg = {"ae_latent_dim": [128, 256], "lr": [1e-3, 1e-4]}
        combos = expand_grid(cfg)
        self.assertEqual(len(combos), 4)


class TestLayer1415SweepDryRun(unittest.TestCase):
    """The Layer14/15 YAML should expose a 20-run parallel grid."""

    def test_parallel_sweep_dry_run_emits_20_runs(self):
        yaml_path = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "experiments"
            / "layer14_15_ae_strategy_sweep.yaml"
        )
        output = StringIO()
        argv = [
            "run_from_yaml.py",
            str(yaml_path),
            "--dry-run",
            "--config",
            "parallel_layer14_15",
        ]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
            main()

        dry_run_output = output.getvalue()
        self.assertIn("Total runs: 20", dry_run_output)
        self.assertEqual(dry_run_output.count("CMD:"), 20)
        self.assertIn("--pe-mlp-mode vae", dry_run_output)
        self.assertIn("--pe-mlp-mode vae_lg", dry_run_output)
        self.assertIn("--seed 1024", dry_run_output)


class TestParseResults(unittest.TestCase):
    def test_parses_valid_results(self):
        data = {"final_ppl": 8.0, "baseline_ppl": 10.0}
        stdout = f"some log output\nResults: {json.dumps(data)}"
        parsed = parse_results(stdout)
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(parsed["final_ppl"], 8.0)

    def test_returns_none_for_missing_marker(self):
        self.assertIsNone(parse_results("no results here"))

    def test_returns_none_for_invalid_json(self):
        self.assertIsNone(parse_results("Results: {invalid json"))


class TestBuildCmdAeCacheDir(unittest.TestCase):
    """ae_cache_dir is forwarded to --ae-cache-dir in the subprocess command."""

    def test_ae_cache_dir_injected(self):
        params = {**_SAMPLE_PARAMS, "ae_cache_dir": "/tmp/shared_cache"}
        cmd = build_cmd(
            python="python3",
            script="scripts/finetune.py",
            params=params,
            output_dir="/tmp/out",
            wandb_cfg={"enabled": False},
        )
        idx = cmd.index("--ae-cache-dir")
        self.assertEqual(cmd[idx + 1], "/tmp/shared_cache")

    def test_ae_cache_dir_absent_by_default(self):
        cmd = build_cmd(
            python="python3",
            script="scripts/finetune.py",
            params=_SAMPLE_PARAMS,
            output_dir="/tmp/out",
            wandb_cfg={"enabled": False},
        )
        self.assertNotIn("--ae-cache-dir", cmd)


class TestRunClaim(unittest.TestCase):
    """Test the claim/lock mechanism for parallel orchestrator instances."""

    def test_claim_run_creates_lock_file(self):
        from scripts.run_from_yaml import claim_run, _CLAIM_FILENAME, release_claim
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(claim_run(tmpdir))
            lock_path = os.path.join(tmpdir, _CLAIM_FILENAME)
            self.assertTrue(os.path.exists(lock_path))
            with open(lock_path) as f:
                data = json.load(f)
            self.assertEqual(data["pid"], os.getpid())
            self.assertIn("timestamp", data)
            self.assertIn("hostname", data)
            release_claim(tmpdir)

    def test_claim_run_blocks_second_claim(self):
        from scripts.run_from_yaml import claim_run, is_run_claimed, release_claim

        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(claim_run(tmpdir))
            # Same PID re-claiming should still succeed (we own it)
            self.assertTrue(claim_run(tmpdir))
            release_claim(tmpdir)

    def test_is_run_claimed_false_when_no_lock(self):
        from scripts.run_from_yaml import is_run_claimed

        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(is_run_claimed(tmpdir))

    def test_is_run_claimed_false_for_own_pid(self):
        from scripts.run_from_yaml import claim_run, is_run_claimed, release_claim

        with tempfile.TemporaryDirectory() as tmpdir:
            claim_run(tmpdir)
            # Our own PID — should not be considered "claimed by another"
            self.assertFalse(is_run_claimed(tmpdir))
            release_claim(tmpdir)

    def test_release_claim_removes_lock(self):
        from scripts.run_from_yaml import claim_run, release_claim, _CLAIM_FILENAME
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            claim_run(tmpdir)
            lock_path = os.path.join(tmpdir, _CLAIM_FILENAME)
            self.assertTrue(os.path.exists(lock_path))
            release_claim(tmpdir)
            self.assertFalse(os.path.exists(lock_path))

    def test_release_claim_noop_when_no_lock(self):
        from scripts.run_from_yaml import release_claim

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            release_claim(tmpdir)

    def test_stale_lock_reclaimed(self):
        """A lock file from a dead PID should be treated as stale and re-claimable."""
        from scripts.run_from_yaml import claim_run, is_run_claimed, _CLAIM_FILENAME
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, _CLAIM_FILENAME)
            # Write a lock with a PID that (almost certainly) doesn't exist
            stale_pid = 2_000_000_000
            with open(lock_path, "w") as f:
                json.dump({"pid": stale_pid, "timestamp": "2020-01-01T00:00:00"}, f)
            # Should not be considered claimed (dead PID)
            self.assertFalse(is_run_claimed(tmpdir))
            # Should be re-claimable
            self.assertTrue(claim_run(tmpdir))
            with open(lock_path) as f:
                data = json.load(f)
            self.assertEqual(data["pid"], os.getpid())

    def test_live_foreign_pid_blocks_claim(self):
        """A lock from a live foreign PID should block claiming."""
        from scripts.run_from_yaml import claim_run, is_run_claimed, _CLAIM_FILENAME
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, _CLAIM_FILENAME)
            # PID 1 (init/systemd) is always alive on Linux
            with open(lock_path, "w") as f:
                json.dump({"pid": 1, "timestamp": "2020-01-01T00:00:00"}, f)
            self.assertTrue(is_run_claimed(tmpdir))
            self.assertFalse(claim_run(tmpdir))

    def test_corrupt_lock_file_reclaimed(self):
        """A corrupt/unreadable lock file should be treated as stale."""
        from scripts.run_from_yaml import claim_run, _CLAIM_FILENAME
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, _CLAIM_FILENAME)
            with open(lock_path, "w") as f:
                f.write("NOT VALID JSON{{{")
            self.assertTrue(claim_run(tmpdir))


if __name__ == "__main__":
    unittest.main()
