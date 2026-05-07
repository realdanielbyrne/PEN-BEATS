#!/usr/bin/env python3
"""Hardware pre-flight check for PELLM experiments.

Run this before any experiment to confirm a CUDA GPU is available and that
the GPU has enough VRAM.  The script exits with code 0 on success and
code 1 on any failure so it can be used in CI or launch scripts.

Usage::

    python scripts/check_hardware.py

Minimum requirements
--------------------
- CUDA-capable GPU (NVIDIA)
- >= 16 GB GPU VRAM  (smol-replacement pretraining with micro_batch=2)
- >= 24 GB GPU VRAM  (recommended for grad_accum_steps=64 without OOM)
- ``accelerate`` installed (for distributed-launch variants)
"""

from __future__ import annotations

import sys

MIN_VRAM_GB = 16.0
RECOMMENDED_VRAM_GB = 24.0

_FAIL = "\033[91m[FAIL]\033[0m"
_WARN = "\033[93m[WARN]\033[0m"
_OK   = "\033[92m[ OK ]\033[0m"


def _gb(n_bytes: int) -> float:
    return n_bytes / 1024 ** 3


def check_torch() -> bool:
    try:
        import torch  # noqa: F401
        print(f"{_OK} torch {torch.__version__} importable")
        return True
    except ImportError:
        print(f"{_FAIL} torch is not installed — run: pip install -e '.[experiments]'")
        return False


def check_cuda() -> tuple[bool, list[dict]]:
    """Return (cuda_ok, list-of-gpu-info-dicts)."""
    import torch

    if not torch.cuda.is_available():
        msg = (
            "CUDA is not available on this machine.\n"
            "\n"
            "  The PELLM experiments (ae_pretrain_loss_sweep and smol_replacement_paper)\n"
            "  require at least one NVIDIA GPU with >= 16 GB VRAM.\n"
            "\n"
            "  Common causes:\n"
            "    - Running on a CPU-only machine or Apple Silicon (MPS not supported here).\n"
            "    - PyTorch was installed without CUDA support (e.g. the CPU wheel).\n"
            "      Fix: pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "    - NVIDIA drivers are missing or outdated.\n"
            "      Fix: install drivers >= 520 and reboot.\n"
            "\n"
            "  Pre-computed results are available in evals/smol_replacement_paper/\n"
            "  if you only need to inspect the evaluation artefacts without retraining."
        )
        print(f"{_FAIL} {msg}")
        return False, []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = _gb(props.total_memory)
        gpus.append({"index": i, "name": props.name, "total_gb": total_gb})

    return True, gpus


def check_vram(gpus: list[dict]) -> bool:
    all_ok = True
    for g in gpus:
        total = g["total_gb"]
        tag = _OK if total >= RECOMMENDED_VRAM_GB else (_WARN if total >= MIN_VRAM_GB else _FAIL)
        print(f"{tag} GPU {g['index']}: {g['name']}  {total:.1f} GB VRAM")
        if total < MIN_VRAM_GB:
            print(
                f"      {_FAIL} GPU {g['index']} has {total:.1f} GB < {MIN_VRAM_GB} GB minimum.\n"
                "             Reduce micro_batch_size to 1 and grad_accum_steps accordingly,\n"
                "             or use a larger GPU."
            )
            all_ok = False
        elif total < RECOMMENDED_VRAM_GB:
            print(
                f"      {_WARN} GPU {g['index']} has {total:.1f} GB — may OOM at grad_accum_steps=64.\n"
                "             Consider setting micro_batch_size: 1 in the YAML."
            )
    return all_ok


def check_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401
        print(f"{_OK} accelerate {accelerate.__version__} importable")
        return True
    except ImportError:
        print(
            f"{_WARN} accelerate not installed — needed for smol_replacement_paper\n"
            "         distributed launch.  Fix: pip install -e '.[experiments]'"
        )
        return False  # warn, not fatal (finetune.py works without it)


def main() -> int:
    print("=" * 60)
    print("PELLM hardware pre-flight check")
    print("=" * 60)

    if not check_torch():
        return 1

    cuda_ok, gpus = check_cuda()
    if not cuda_ok:
        return 1

    vram_ok = check_vram(gpus)
    accel_ok = check_accelerate()

    print("=" * 60)
    if vram_ok and accel_ok:
        print(f"{_OK} All checks passed — ready to run PELLM experiments.")
        return 0
    elif vram_ok:
        print(f"{_WARN} Checks passed with warnings (see above).")
        return 0
    else:
        print(
            f"{_FAIL} Hardware requirements not met.\n"
            "       See evals/smol_replacement_paper/ for pre-computed results."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
