"""Tests for pretrain_smol_replacement.py checkpoint resumability helpers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Ensure the scripts directory is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Import the module under test (not the __main__ block)
import importlib

_mod = importlib.import_module("pretrain_smol_replacement")
find_latest_checkpoint = _mod.find_latest_checkpoint
load_training_state = _mod.load_training_state
should_run_final_eval = _mod.should_run_final_eval
next_eval_for_cadence = _mod.next_eval_for_cadence
best_effort_shutdown_dataloader_workers = _mod.best_effort_shutdown_dataloader_workers
repair_eval_metadata = _mod.repair_eval_metadata
_TRAINING_STATE_FILENAME = _mod._TRAINING_STATE_FILENAME


class TestFindLatestCheckpoint(unittest.TestCase):
    """Test find_latest_checkpoint picks the highest-token checkpoint."""

    def test_returns_none_for_empty_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_checkpoint(Path(tmpdir))
            self.assertIsNone(result)

    def test_returns_none_for_nonexistent_dir(self):
        result = find_latest_checkpoint(Path("/nonexistent/path/abc123"))
        self.assertIsNone(result)

    def test_returns_none_when_no_state_file(self):
        """Dirs matching tokens_N but missing training_state.json are skipped."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "tokens_1000").mkdir()
            result = find_latest_checkpoint(Path(tmpdir))
            self.assertIsNone(result)

    def test_finds_single_checkpoint(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "tokens_500000"
            ckpt.mkdir()
            state = {"trained_tokens": 500000, "step": 10, "next_eval": 1000000}
            with (ckpt / _TRAINING_STATE_FILENAME).open("w") as f:
                json.dump(state, f)

            result = find_latest_checkpoint(Path(tmpdir))
            self.assertEqual(result, ckpt)

    def test_finds_latest_among_multiple(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for tokens in [250000, 500000, 750000]:
                ckpt = Path(tmpdir) / f"tokens_{tokens}"
                ckpt.mkdir()
                state = {
                    "trained_tokens": tokens,
                    "step": tokens // 1000,
                    "next_eval": tokens + 250000,
                }
                with (ckpt / _TRAINING_STATE_FILENAME).open("w") as f:
                    json.dump(state, f)

            result = find_latest_checkpoint(Path(tmpdir))
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "tokens_750000")

    def test_ignores_non_token_dirs(self):
        """Dirs not matching tokens_N pattern are ignored."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid checkpoint
            ckpt = Path(tmpdir) / "tokens_100"
            ckpt.mkdir()
            state = {"trained_tokens": 100, "step": 1, "next_eval": 200}
            with (ckpt / _TRAINING_STATE_FILENAME).open("w") as f:
                json.dump(state, f)

            # Non-matching dirs
            (Path(tmpdir) / "some_other_dir").mkdir()
            (Path(tmpdir) / "hf_model").mkdir()

            result = find_latest_checkpoint(Path(tmpdir))
            self.assertEqual(result, ckpt)


class TestLoadTrainingState(unittest.TestCase):
    """Test load_training_state reads the JSON metadata correctly."""

    def test_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "tokens_42"
            ckpt.mkdir()
            expected = {
                "trained_tokens": 42000,
                "step": 100,
                "next_eval": 50000,
                "timestamp": "2026-04-28T12:00:00+00:00",
            }
            with (ckpt / _TRAINING_STATE_FILENAME).open("w") as f:
                json.dump(expected, f)

            loaded = load_training_state(ckpt)
            self.assertEqual(loaded["trained_tokens"], 42000)
            self.assertEqual(loaded["step"], 100)
            self.assertEqual(loaded["next_eval"], 50000)

    def test_legacy_next_save_field_is_ignored(self):
        """Older checkpoints carry a ``next_save`` field; loading must succeed."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "tokens_99"
            ckpt.mkdir()
            legacy = {
                "trained_tokens": 99000,
                "step": 7,
                "next_eval": 100000,
                "next_save": 250000,
                "timestamp": "2025-12-01T00:00:00+00:00",
            }
            with (ckpt / _TRAINING_STATE_FILENAME).open("w") as f:
                json.dump(legacy, f)

            loaded = load_training_state(ckpt)
            self.assertEqual(loaded["trained_tokens"], 99000)
            self.assertEqual(loaded["next_eval"], 100000)
            # Field is preserved in the dict (json.load is faithful) but the
            # training loop no longer consumes it.
            self.assertEqual(loaded.get("next_save"), 250000)


class TestSkipBatchCalculation(unittest.TestCase):
    """Test the skip-batch arithmetic used in main()."""

    def test_skip_batch_count(self):
        """Verify the skip calculation matches: trained_tokens // tokens_per_batch."""
        trained_tokens = 250_000_000
        micro_batch_size = 2
        seq_len = 2048
        num_processes = 2
        tokens_per_batch = micro_batch_size * seq_len * num_processes
        expected_skip = trained_tokens // tokens_per_batch
        self.assertEqual(expected_skip, 250_000_000 // (2 * 2048 * 2))
        self.assertEqual(expected_skip, 30517)


class TestSaveTrainingStateSignature(unittest.TestCase):
    """``save_training_state`` no longer takes a ``next_save`` parameter."""

    def test_no_next_save_parameter(self):
        import inspect

        sig = inspect.signature(_mod.save_training_state)
        self.assertNotIn("next_save", sig.parameters)
        # Sanity: the params we still rely on are present.
        for required in ("trained_tokens", "step", "next_eval"):
            self.assertIn(required, sig.parameters)


class TestFinalEvalHelpers(unittest.TestCase):
    """Test final metric freshness and metadata repair helpers."""

    def test_next_eval_for_cadence_realigns_from_trained_tokens(self):
        self.assertEqual(next_eval_for_cadence(10_485_760, 100_000_000), 100_000_000)
        self.assertEqual(next_eval_for_cadence(250_000_000, 100_000_000), 300_000_000)
        self.assertEqual(next_eval_for_cadence(300_000_000, 100_000_000), 400_000_000)

    def test_next_eval_for_cadence_rejects_non_positive_interval(self):
        with self.assertRaises(ValueError):
            next_eval_for_cadence(100, 0)

    def test_final_eval_needed_when_metrics_are_missing_or_stale(self):
        self.assertTrue(should_run_final_eval({}, 100))
        self.assertTrue(should_run_final_eval({"tokens": 90}, 100))
        self.assertTrue(should_run_final_eval({"tokens": "unknown"}, 100))
        self.assertFalse(should_run_final_eval({"tokens": 100}, 100))
        self.assertFalse(should_run_final_eval({"tokens": 120}, 100))

    def test_repair_eval_metadata_updates_summary_and_manifest(self):
        import tempfile

        final_metrics = {
            "tokens": 123,
            "step": 7,
            "loss": 2.5,
            "perplexity": 12.18,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "model"
            eval_dir = root / "evals"
            output_dir.mkdir()
            manifest_path = output_dir / "training_manifest.json"
            with manifest_path.open("w") as f:
                json.dump({"final_metrics": {"tokens": 10}}, f)

            repair_eval_metadata(
                output_dir=output_dir,
                eval_dir=eval_dir,
                final_metrics=final_metrics,
                reason="unit_test",
            )

            with (eval_dir / "summary.json").open() as f:
                summary = json.load(f)
            with manifest_path.open() as f:
                manifest = json.load(f)

            self.assertEqual(summary, final_metrics)
            self.assertEqual(manifest["final_metrics"], final_metrics)
            self.assertEqual(manifest["metadata_repairs"][0]["reason"], "unit_test")


class TestDataLoaderCleanup(unittest.TestCase):
    """Test best-effort worker cleanup across wrapper shapes."""

    def test_shutdowns_loader_iterator_attribute(self):
        class FakeIterator:
            def __init__(self):
                self.shutdown_count = 0

            def _shutdown_workers(self):
                self.shutdown_count += 1

        class FakeLoader:
            def __init__(self):
                self._iterator = FakeIterator()

        loader = FakeLoader()
        count = best_effort_shutdown_dataloader_workers(loader)
        self.assertEqual(count, 1)
        self.assertEqual(loader._iterator.shutdown_count, 1)

    def test_shutdowns_iterator_hidden_in_generator_frame(self):
        class FakeIterator:
            def __init__(self):
                self.shutdown_count = 0

            def _shutdown_workers(self):
                self.shutdown_count += 1

        def wrapper(inner):
            dataloader_iter = inner
            yield dataloader_iter

        inner = FakeIterator()
        gen = wrapper(inner)
        next(gen)

        count = best_effort_shutdown_dataloader_workers(gen)
        self.assertEqual(count, 1)
        self.assertEqual(inner.shutdown_count, 1)

    def test_cleanup_tolerates_shutdown_errors(self):
        class BrokenIterator:
            def _shutdown_workers(self):
                raise RuntimeError("already stopped")

        count = best_effort_shutdown_dataloader_workers(BrokenIterator())
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
