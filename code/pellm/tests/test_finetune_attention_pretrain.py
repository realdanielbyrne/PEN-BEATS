import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from finetune import (  # noqa: E402
    _ATTN_CACHE_VERSION,
    _attn_cache_is_valid,
    _write_attn_cache_metadata,
    parse_args,
)


class TestFinetuneAttentionCLI(unittest.TestCase):
    def test_attn_init_defaults_to_pretrained(self):
        with mock.patch.object(sys, "argv", ["finetune.py"]):
            args = parse_args()

        self.assertEqual(args.attn_init, "pretrained")
        self.assertEqual(args.attn_pretrain_epochs, 0)
        self.assertEqual(args.attn_dataset, "wikitext2")


class TestAttentionCacheMetadata(unittest.TestCase):
    def test_attn_cache_is_valid_requires_matching_metadata_and_layers(self):
        expected = {
            "version": _ATTN_CACHE_VERSION,
            "model_name": "demo",
            "dataset": "wikitext2",
            "max_length": 32,
            "batch_size": 2,
            "cache_num_samples": None,
            "available_projs": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "cached_layers": [0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_attn_cache_metadata(tmpdir, expected)
            torch.save(
                {
                    "qkv_x": torch.zeros(4, 8),
                    "o_x": torch.zeros(4, 8),
                    "outputs": {name: torch.zeros(4, 8) for name in expected["available_projs"]},
                    "available_projs": expected["available_projs"],
                    "shape_meta": {"tokens": 4},
                },
                Path(tmpdir) / "layer0_batch0.pt",
            )

            self.assertTrue(_attn_cache_is_valid(tmpdir, expected, [0]))
            self.assertFalse(_attn_cache_is_valid(tmpdir, expected, [1]))

            mismatched = {**expected, "dataset": "fineweb"}
            self.assertFalse(_attn_cache_is_valid(tmpdir, mismatched, [0]))
