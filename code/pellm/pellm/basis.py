"""
Frozen basis tensor factories ported from N-BEATS Lightning.

- ``build_trend_basis``: Vandermonde polynomial basis (from ``_TrendGenerator``)
- ``build_wavelet_basis``: SVD-orthonormalized DWT basis (from ``_WaveletGeneratorV3._build_basis``)

Both return plain ``torch.Tensor`` — no ``nn.Module`` wrapper.
"""

import logging
from typing import Any, cast

import numpy as np
import pywt
import torch


def build_trend_basis(thetas_dim: int, target_length: int) -> torch.Tensor:
    """Build a Vandermonde polynomial basis matrix.

    Returns a ``(thetas_dim, target_length)`` tensor where row *i* is
    ``(t / target_length) ** i`` for ``t`` in ``[0, target_length)``.

    Ported from ``_TrendGenerator`` in N-BEATS ``blocks/blocks.py``.
    """
    basis = torch.stack(
        [(torch.arange(target_length, dtype=torch.float32) / target_length) ** i
         for i in range(thetas_dim)],
        dim=0,
    )  # (thetas_dim, target_length)
    return basis


def build_wavelet_basis(
    target_length: int,
    wavelet_type: str = "db3",
    basis_dim: int | None = None,
    basis_offset: int = 0,
    max_decomp_level: int = 5,
) -> torch.Tensor:
    """Build an SVD-orthonormalized DWT basis matrix.

    Returns a ``(effective_dim, target_length)`` tensor whose rows are
    orthonormal wavelet basis vectors ordered low-to-high frequency by
    singular value.  ``basis_dim`` and ``basis_offset`` select a frequency
    band window.

    Ported from ``_WaveletGeneratorV3._build_basis`` in N-BEATS ``blocks/blocks.py``.
    """
    # PyWavelets' constructor typing is incomplete, so we narrow it for Pylance.
    wavelet = cast(Any, pywt.Wavelet)(wavelet_type)
    max_level = pywt.dwt_max_level(target_length, wavelet.dec_len)
    level = min(max_level, max_decomp_level)

    # Get DWT coefficient structure
    dummy = np.zeros(target_length)
    coeffs = pywt.wavedec(dummy, wavelet_type, level=level)
    coeff_lengths = [len(c) for c in coeffs]

    # Build raw synthesis matrix via impulse responses
    basis_rows = []
    for band_idx, band_len in enumerate(coeff_lengths):
        for j in range(band_len):
            impulse = [np.zeros(length) for length in coeff_lengths]
            impulse[band_idx][j] = 1.0
            reconstructed = pywt.waverec(impulse, wavelet_type)
            basis_rows.append(reconstructed[:target_length])

    raw_basis = np.array(basis_rows, dtype=np.float64)

    # SVD orthogonalization — rows of Vt ordered low->high frequency
    _, S, Vt = np.linalg.svd(raw_basis, full_matrices=False)
    tol = S[0] * max(raw_basis.shape) * np.finfo(np.float64).eps
    full_rank = int(np.sum(S > tol))

    if full_rank < target_length:
        logging.warning(
            f"WaveletV3 rank-deficient: {full_rank}/{target_length} "
            f"for '{wavelet_type}'"
        )

    # Select frequency band via offset + window
    offset = min(basis_offset, full_rank - 1)
    available = full_rank - offset
    effective_dim = available if basis_dim is None else min(available, basis_dim)
    ortho_basis = Vt[offset : offset + effective_dim, :]

    return torch.tensor(ortho_basis, dtype=torch.float32)
