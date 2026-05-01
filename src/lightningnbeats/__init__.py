import os
import torch
from .models import *
from .loaders import *
from .losses import *
from .blocks.blocks import *
from .constants import DEPRECATED_BLOCKS, get_deprecated_block_message


def get_best_accelerator() -> str:
    """Detect the best available accelerator for PyTorch.

    Returns 'cuda' for NVIDIA GPUs, 'xpu' for Intel Arc (PyTorch 2.4+),
    'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def __getattr__(name):
    if name in DEPRECATED_BLOCKS:
        raise AttributeError(get_deprecated_block_message(name))
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

