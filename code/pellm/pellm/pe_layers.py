"""
Parameter-efficient layer replacements inspired by N-BEATS block architectures.

- ``TrendWaveletLinear``: Replaces ``nn.Linear`` in attention projections using
  parallel Vandermonde + DWT frozen bases (from TrendWaveletAE pattern).
- ``TrendWaveletGenericLinear``: Extends ``TrendWaveletLinear`` with a third
  learned generic basis branch (from TrendWaveletGeneric pattern).
- ``TrendWaveletLinearLG``: Learned-gate variant of ``TrendWaveletLinear``.
- ``TrendWaveletGenericLinearLG``: Learned-gate variant of ``TrendWaveletGenericLinear``.
- ``PEBottleneckMLP``: Replaces SwiGLU MLP with AE-bottleneck (from AERootBlock).
- ``PEBottleneckMLPLG``: Learned-gate variant (from AERootBlockLG).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis import build_trend_basis, build_wavelet_basis


def _fourier_filter_rows(W: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-*k* frequency components per row of *W* via FFT.

    For each row, computes the real FFT, identifies the *k* bins with largest
    magnitude, zeros the rest, and inverse-FFTs back to the spatial domain.

    Args:
        W: 2-D tensor of shape ``(rows, cols)``.
        k: Number of frequency bins to retain per row.

    Returns:
        Filtered tensor, same shape as *W*.
    """
    n = W.shape[1]
    W_freq = torch.fft.rfft(W, dim=1)  # (rows, n//2+1)
    magnitudes = W_freq.abs()
    k_clamped = min(k, W_freq.shape[1])
    _, top_indices = torch.topk(magnitudes, k_clamped, dim=1)  # (rows, k)
    mask = torch.zeros_like(W_freq, dtype=torch.bool)
    mask.scatter_(1, top_indices, True)
    return torch.fft.irfft(W_freq * mask, n=n, dim=1)


def _svd_fc2fc3(fc1_w: torch.Tensor, latent_dim: int):
    """Compute fc2/fc3 weights via rank-latent_dim SVD of fc1's column subspace.

    Args:
        fc1_w: (mid_dim, hidden_size) — the already-computed fc1 weight matrix
        latent_dim: bottleneck dimension

    Returns:
        (fc2_w, fc3_w) tensors
    """
    mid_dim = fc1_w.shape[0]
    W_sub = fc1_w[:, :mid_dim]
    U_sub, S_sub, Vh_sub = torch.linalg.svd(W_sub, full_matrices=False)
    fc2_w = Vh_sub[:latent_dim, :]  # (latent_dim, mid_dim)
    fc3_w = U_sub[:, :latent_dim] * S_sub[:latent_dim]  # (mid_dim, latent_dim)
    return fc2_w, fc3_w


def _tw_pretrained_weight(W: torch.Tensor, coeff_dim: int) -> torch.Tensor:
    """Directly truncate *W* to the TrendWavelet coefficient width."""
    coeff_weight = torch.zeros(coeff_dim, W.shape[1], dtype=W.dtype, device=W.device)
    copy_rows = min(coeff_dim, W.shape[0])
    coeff_weight[:copy_rows].copy_(W[:copy_rows])
    return coeff_weight


def _tw_lstsq_weight(W: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project *W* onto the TrendWavelet basis via least-squares."""
    return torch.linalg.lstsq(basis.T, W).solution


def _tw_svd_weight(W: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Approximate *W* with a rank-k SVD before TrendWavelet projection."""
    rank = min(basis.shape[0], W.shape[0], W.shape[1])
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W_k = (U[:, :rank] * S[:rank]) @ Vh[:rank]
    return _tw_lstsq_weight(W_k, basis)


def _tw_cur_weight(W: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Approximate *W* with leverage-score CUR before TrendWavelet projection."""
    rank = min(basis.shape[0], W.shape[0], W.shape[1])
    U, _, Vh = torch.linalg.svd(W, full_matrices=False)
    row_scores = (U[:, :rank] ** 2).sum(dim=1) / rank
    col_scores = (Vh[:rank, :] ** 2).sum(dim=0) / rank
    row_idx = torch.argsort(row_scores, descending=True)[:rank]
    col_idx = torch.argsort(col_scores, descending=True)[:rank]
    row_idx, _ = torch.sort(row_idx)
    col_idx, _ = torch.sort(col_idx)

    C = W[:, col_idx]
    R = W[row_idx, :]
    W_intersection = W[row_idx][:, col_idx]
    W_cur = C @ torch.linalg.pinv(W_intersection) @ R
    return _tw_lstsq_weight(W_cur, basis)


def _tw_fourier_weight(W: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Denoise *W* with FFT filtering before TrendWavelet projection."""
    filtered = _fourier_filter_rows(W, basis.shape[0])
    return _tw_lstsq_weight(filtered, basis)


def _tw_weight_by_mode(W: torch.Tensor, basis: torch.Tensor, mode: str) -> torch.Tensor:
    """Return the frozen-basis coefficient matrix for a TrendWavelet init mode."""
    if mode == "pretrained":
        return _tw_pretrained_weight(W, basis.shape[0])
    if mode == "lstsq":
        return _tw_lstsq_weight(W, basis)
    if mode == "svd":
        return _tw_svd_weight(W, basis)
    if mode == "cur":
        return _tw_cur_weight(W, basis)
    if mode == "fourier":
        return _tw_fourier_weight(W, basis)
    raise ValueError(f"Unsupported TrendWavelet init mode: {mode}")


def _build_tw_layer(
    cls,
    linear: nn.Linear,
    *,
    trend_dim: int = 3,
    wavelet_dim: int = 28,
    wavelet_type: str = "db3",
    basis_offset: int = 0,
    active_g: bool = False,
    **extra_kwargs,
):
    layer = cls(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        trend_dim=trend_dim,
        wavelet_dim=wavelet_dim,
        wavelet_type=wavelet_type,
        basis_offset=basis_offset,
        active_g=active_g,
        **extra_kwargs,
    )
    return layer.to(linear.weight.dtype)


def _build_tw_generic_layer(
    cls,
    linear: nn.Linear,
    *,
    trend_dim: int = 3,
    wavelet_dim: int = 28,
    wavelet_type: str = "db3",
    basis_offset: int = 0,
    generic_dim: int = 5,
    active_g: bool = False,
    **extra_kwargs,
):
    layer = cls(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        trend_dim=trend_dim,
        wavelet_dim=wavelet_dim,
        wavelet_type=wavelet_type,
        basis_offset=basis_offset,
        generic_dim=generic_dim,
        active_g=active_g,
        **extra_kwargs,
    )
    return layer.to(linear.weight.dtype)


def _init_tw_layer(layer, linear: nn.Linear, mode: str) -> None:
    basis = torch.cat([layer.trend_basis, layer.wavelet_basis], dim=0).float()
    weight = _tw_weight_by_mode(linear.weight.float(), basis, mode)
    with torch.no_grad():
        layer.theta.weight.copy_(weight.to(layer.theta.weight.dtype))
        if linear.bias is not None and layer.theta.bias is not None:
            layer.theta.bias.zero_()


def _init_tw_generic_layer(layer, linear: nn.Linear, mode: str) -> None:
    basis = torch.cat([layer.trend_basis, layer.wavelet_basis], dim=0).float()
    frozen_dim = basis.shape[0]
    weight = _tw_weight_by_mode(linear.weight.float(), basis, mode)
    with torch.no_grad():
        layer.theta.weight[:frozen_dim].copy_(weight.to(layer.theta.weight.dtype))
        layer.theta.weight[frozen_dim:].zero_()
        layer.generic_basis.weight.zero_()
        if linear.bias is not None and layer.theta.bias is not None:
            layer.theta.bias.zero_()


_TW_REDUCTION_SILU_LINEAR_SCALE = 2.0


def _build_tw_reduced_input_projection(
    in_features: int,
    reduction_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the structured input projection used by reduced TrendWavelet init.

    Reduced TrendWavelet layers insert ``silu(reduction(x))`` before ``theta``.
    To keep initialization analogous to unreduced layers, we start from a
    truncated identity projection so the first ``reduction_dim`` input channels
    are preserved in order. Callers store ``2 * projection`` in
    ``reduction.weight`` so ``silu(reduction(x))`` is locally close to the
    identity map around zero.
    """
    input_projection = torch.zeros(
        reduction_dim,
        in_features,
        dtype=dtype,
        device=device,
    )
    diagonal_dim = min(in_features, reduction_dim)
    input_projection[:diagonal_dim, :diagonal_dim] = torch.eye(
        diagonal_dim,
        dtype=dtype,
        device=device,
    )
    return input_projection


def _build_tw_reduced_init_weights(
    linear: nn.Linear,
    reduction_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(input_projection, reduced_weight)`` for reduced TW init.

    ``input_projection`` is the structured truncated input map before the SiLU
    linearization scale is applied. ``reduced_weight`` is the dense weight matrix
    expressed in the reduced coordinates used to initialize ``theta``.
    """
    input_projection = _build_tw_reduced_input_projection(
        linear.in_features,
        reduction_dim,
        dtype=linear.weight.dtype,
        device=linear.weight.device,
    )
    reduced_weight = linear.weight.float() @ input_projection.float().T
    return input_projection, reduced_weight


def _init_tw_reduced_layer(layer, linear: nn.Linear, mode: str) -> None:
    """Init a reduced TrendWavelet layer with structured reduction + TW theta."""
    basis = torch.cat([layer.trend_basis, layer.wavelet_basis], dim=0).float()
    input_projection, reduced_weight = _build_tw_reduced_init_weights(
        linear,
        layer.reduction_dim,
    )
    theta_weight = _tw_weight_by_mode(reduced_weight, basis, mode)
    with torch.no_grad():
        layer.theta.weight.copy_(theta_weight.to(layer.theta.weight.dtype))
        layer.reduction.weight.copy_(
            (_TW_REDUCTION_SILU_LINEAR_SCALE * input_projection).to(
                layer.reduction.weight.dtype
            )
        )
        if linear.bias is not None and layer.theta.bias is not None:
            layer.theta.bias.zero_()


def _init_tw_generic_reduced_layer(layer, linear: nn.Linear, mode: str) -> None:
    """Init a reduced generic TW layer with structured reduction + frozen TW slice."""
    basis = torch.cat([layer.trend_basis, layer.wavelet_basis], dim=0).float()
    frozen_dim = basis.shape[0]
    input_projection, reduced_weight = _build_tw_reduced_init_weights(
        linear,
        layer.reduction_dim,
    )
    frozen_theta_weight = _tw_weight_by_mode(reduced_weight, basis, mode)
    with torch.no_grad():
        layer.theta.weight[:frozen_dim].copy_(
            frozen_theta_weight.to(layer.theta.weight.dtype)
        )
        layer.theta.weight[frozen_dim:].zero_()
        layer.reduction.weight.copy_(
            (_TW_REDUCTION_SILU_LINEAR_SCALE * input_projection).to(
                layer.reduction.weight.dtype
            )
        )
        layer.generic_basis.weight.zero_()
        if linear.bias is not None and layer.theta.bias is not None:
            layer.theta.bias.zero_()


def set_tw_active_g(model: nn.Module, active_g: bool) -> None:
    """Toggle ``active_g`` on all TrendWavelet layers in *model*."""
    for module in model.modules():
        if isinstance(
            module,
            (
                TrendWaveletLinear,
                TrendWaveletGenericLinear,
                TrendWaveletLinearLG,
                TrendWaveletGenericLinearLG,
                TrendWaveletLinearReduced,
                TrendWaveletLinearLGReduced,
                TrendWaveletGenericLinearReduced,
                TrendWaveletGenericLinearLGReduced,
            ),
        ):
            module.active_g = active_g


class TrendWaveletLinear(nn.Module):
    """TrendWavelet-based replacement for ``nn.Linear`` in attention projections.

    Mirrors the proven TrendWaveletAE pattern: routes through a learned projection
    to trend + wavelet coefficient vectors, expands through two frozen bases, and
    sums.

    Factorization::

        x -> theta(x) -> [trend_coeffs | wavelet_coeffs]
        output = trend_coeffs @ Vandermonde + wavelet_coeffs @ Wavelet_basis

    Trainable params: ``in_features * (trend_dim + effective_wavelet_dim)``
    Frozen params: Vandermonde basis + DWT basis (registered as buffers)

    The potential wavelet types are those supported by PyWavelets:
    - "haar"
    - "db1"
    - "db2",
    - "db3",
    - "db4",
    - "db10",
    - "db20",
    - "coif1"
    - "coif2",
    - "coif3",
    - "coif4",
    - "coif10",
    - "sym2",
    - "sym3",
    - "sym4",
    - "sym10",
    - "sym20"
    - etc
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        trend_dim: int = 3,
        wavelet_dim: int = 28,
        wavelet_type: str = "db3",
        basis_offset: int = 0,
        active_g: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.active_g = active_g

        # Build frozen bases
        trend_basis = build_trend_basis(trend_dim, out_features)
        wavelet_basis = build_wavelet_basis(
            out_features, wavelet_type, wavelet_dim, basis_offset
        )

        self.register_buffer("trend_basis", trend_basis)  # (trend_dim, out)
        self.register_buffer("wavelet_basis", wavelet_basis)  # (eff_wave_dim, out)

        self.trend_dim = trend_dim
        effective_wavelet_dim = wavelet_basis.shape[0]
        total_coeff_dim = trend_dim + effective_wavelet_dim

        self.theta = nn.Linear(in_features, total_coeff_dim, bias=bias)

    # -- Class-level hooks for build/init (overridden by subclasses) ----------
    _tw_build_fn = staticmethod(_build_tw_layer)
    _tw_init_fn = staticmethod(_init_tw_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = self.theta(x)  # (..., total_dim)
        trend_c = coeffs[..., : self.trend_dim]  # (..., trend_dim)
        wavelet_c = coeffs[..., self.trend_dim :]  # (..., eff_wave_dim)
        dtype = coeffs.dtype
        out = trend_c @ self.trend_basis.to(dtype) + wavelet_c @ self.wavelet_basis.to(
            dtype
        )  # (..., out)
        return F.silu(out) if self.active_g else out

    @classmethod
    def _from_pretrained_with_mode(cls, linear: nn.Linear, mode: str, **kwargs):
        """Shared factory: build layer via ``_tw_build_fn``, init via ``_tw_init_fn``."""
        layer = cls._tw_build_fn(cls, linear, **kwargs)
        cls._tw_init_fn(layer, linear, mode=mode)
        return layer

    @classmethod
    def from_pretrained_linear(cls, linear, **kwargs):
        """Initialize from a pretrained ``nn.Linear`` (least-squares)."""
        return cls._from_pretrained_with_mode(linear, "lstsq", **kwargs)

    @classmethod
    def from_pretrained_linear_pretrained(cls, linear, **kwargs):
        return cls._from_pretrained_with_mode(linear, "pretrained", **kwargs)

    @classmethod
    def from_pretrained_linear_svd(cls, linear, **kwargs):
        return cls._from_pretrained_with_mode(linear, "svd", **kwargs)

    @classmethod
    def from_pretrained_linear_cur(cls, linear, **kwargs):
        return cls._from_pretrained_with_mode(linear, "cur", **kwargs)

    @classmethod
    def from_pretrained_linear_fourier(cls, linear, **kwargs):
        return cls._from_pretrained_with_mode(linear, "fourier", **kwargs)


class TrendWaveletGenericLinear(nn.Module):
    """Three-way additive decomposition: trend + wavelet + learned generic.

    Extends ``TrendWaveletLinear`` with a third generic branch using a trainable
    ``nn.Linear`` projection (no bias), allowing the network to learn residual
    patterns not captured by the frozen trend + wavelet bases.

    Factorization::

        x -> theta(x) -> [trend_coeffs | wavelet_coeffs | generic_coeffs]
        output = trend_coeffs @ Vandermonde
               + wavelet_coeffs @ Wavelet_basis
               + generic_basis(generic_coeffs)

    Trainable params: ``theta`` + ``generic_basis`` weights
    Frozen params: Vandermonde basis + DWT basis (registered as buffers)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        trend_dim: int = 3,
        wavelet_dim: int = 28,
        wavelet_type: str = "db3",
        basis_offset: int = 0,
        generic_dim: int = 5,
        active_g: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trend_dim = trend_dim
        self.generic_dim = generic_dim
        self.active_g = active_g

        trend_basis = build_trend_basis(trend_dim, out_features)
        wavelet_basis = build_wavelet_basis(
            out_features, wavelet_type, wavelet_dim, basis_offset
        )
        self.register_buffer("trend_basis", trend_basis)  # (trend_dim, out)
        self.register_buffer("wavelet_basis", wavelet_basis)  # (eff_wave_dim, out)

        self.effective_wavelet_dim = wavelet_basis.shape[0]
        total_coeff_dim = trend_dim + self.effective_wavelet_dim + generic_dim

        self.theta = nn.Linear(in_features, total_coeff_dim, bias=bias)
        self.generic_basis = nn.Linear(generic_dim, out_features, bias=False)

    # -- Class-level hooks for build/init (overridden by subclasses) ----------
    _tw_build_fn = staticmethod(_build_tw_generic_layer)
    _tw_init_fn = staticmethod(_init_tw_generic_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = self.theta(x)
        wave_end = self.trend_dim + self.effective_wavelet_dim
        dtype = coeffs.dtype
        out = (
            coeffs[..., : self.trend_dim] @ self.trend_basis.to(dtype)
            + coeffs[..., self.trend_dim : wave_end] @ self.wavelet_basis.to(dtype)
            + self.generic_basis(coeffs[..., wave_end:])
        )
        return F.silu(out) if self.active_g else out

    @classmethod
    def _from_pretrained_with_mode(cls, linear: nn.Linear, mode: str, **kwargs):
        """Shared factory: build layer via ``_tw_build_fn``, init via ``_tw_init_fn``."""
        layer = cls._tw_build_fn(cls, linear, **kwargs)
        cls._tw_init_fn(layer, linear, mode=mode)
        return layer

    @classmethod
    def from_pretrained_linear(cls, linear, **kwargs):
        """Initialize from a pretrained ``nn.Linear`` (least-squares)."""
        return cls._from_pretrained_with_mode(linear, "lstsq", **kwargs)

    @classmethod
    def from_pretrained_linear_pretrained(cls, linear, **kwargs):
        return cls._from_pretrained_with_mode(linear, "pretrained", **kwargs)

    @classmethod
    def from_pretrained_linear_svd(cls, linear, **kwargs):
        return cls._from_pretrained_with_mode(linear, "svd", **kwargs)

    @classmethod
    def from_pretrained_linear_cur(cls, linear, **kwargs):
        return cls._from_pretrained_with_mode(linear, "cur", **kwargs)

    @classmethod
    def from_pretrained_linear_fourier(cls, linear, **kwargs):
        return cls._from_pretrained_with_mode(linear, "fourier", **kwargs)


class TrendWaveletLinearLG(TrendWaveletLinear):
    """Learned-gate variant of ``TrendWaveletLinear``.

    Adds a per-dimension sigmoid gate on the full coefficient vector, allowing
    the network to discover which coefficient dimensions are most useful during
    training.

    Gate init: ``ones`` → ``sigmoid(1) ≈ 0.73`` (consistent with
    ``PEBottleneckMLPLG.latent_gate`` convention).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        trend_dim: int = 3,
        wavelet_dim: int = 28,
        wavelet_type: str = "db3",
        basis_offset: int = 0,
        active_g: bool = False,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            trend_dim,
            wavelet_dim,
            wavelet_type,
            basis_offset,
            active_g=active_g,
        )
        self.coeff_gate = nn.Parameter(torch.ones(self.theta.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = torch.sigmoid(self.coeff_gate) * self.theta(x)
        dtype = coeffs.dtype
        out = coeffs[..., : self.trend_dim] @ self.trend_basis.to(dtype) + coeffs[
            ..., self.trend_dim :
        ] @ self.wavelet_basis.to(dtype)
        return F.silu(out) if self.active_g else out

    # Inherits _tw_build_fn, _tw_init_fn, and all from_pretrained_linear_*
    # classmethods from TrendWaveletLinear — cls ensures correct type.


class TrendWaveletGenericLinearLG(TrendWaveletGenericLinear):
    """Learned-gate variant of ``TrendWaveletGenericLinear``.

    Adds a per-dimension sigmoid gate on the full coefficient vector
    (trend + wavelet + generic).

    Gate init: ``ones`` → ``sigmoid(1) ≈ 0.73``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        trend_dim: int = 3,
        wavelet_dim: int = 28,
        wavelet_type: str = "db3",
        basis_offset: int = 0,
        generic_dim: int = 5,
        active_g: bool = False,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            trend_dim,
            wavelet_dim,
            wavelet_type,
            basis_offset,
            generic_dim,
            active_g=active_g,
        )
        self.coeff_gate = nn.Parameter(torch.ones(self.theta.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = torch.sigmoid(self.coeff_gate) * self.theta(x)
        wave_end = self.trend_dim + self.effective_wavelet_dim
        dtype = coeffs.dtype
        out = (
            coeffs[..., : self.trend_dim] @ self.trend_basis.to(dtype)
            + coeffs[..., self.trend_dim : wave_end] @ self.wavelet_basis.to(dtype)
            + self.generic_basis(coeffs[..., wave_end:])
        )
        return F.silu(out) if self.active_g else out

    # Inherits _tw_build_fn, _tw_init_fn, and all from_pretrained_linear_*
    # classmethods from TrendWaveletGenericLinear — cls ensures correct type.


# ---------------------------------------------------------------------------
# Reduced TrendWavelet layers (dimensionality-reduction stage before theta)
# ---------------------------------------------------------------------------


class TrendWaveletLinearReduced(TrendWaveletLinear):
    """TrendWavelet with a learned dimensionality-reduction stage before ``theta``.

    Forward::

        x -> reduction(x) -> silu -> theta -> coefficients -> basis expansion

    The ``reduction`` layer projects from ``in_features`` to ``reduction_dim``,
    reducing the effective input dimension before the coefficient projection.
    This adds a small amount of trainable parameters while potentially
    improving the quality of the coefficient representation.

    Args:
        reduction_dim: Intermediate dimension after reduction.  Defaults to
            ``in_features // 2`` (1024 for Llama 3.2-1B).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        trend_dim: int = 3,
        wavelet_dim: int = 28,
        wavelet_type: str = "db3",
        basis_offset: int = 0,
        active_g: bool = False,
        reduction_dim: int | None = None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            trend_dim,
            wavelet_dim,
            wavelet_type,
            basis_offset,
            active_g=active_g,
        )
        if reduction_dim is None:
            reduction_dim = in_features // 2
        self.reduction_dim = reduction_dim
        self.reduction = nn.Linear(in_features, reduction_dim, bias=False)
        # Re-create theta to project from reduction_dim instead of in_features
        total_coeff_dim = self.theta.out_features
        self.theta = nn.Linear(reduction_dim, total_coeff_dim, bias=bias)

    # Override init hook — reduced layers need SVD-factored reduction init
    _tw_init_fn = staticmethod(_init_tw_reduced_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.reduction(x))  # (..., reduction_dim)
        coeffs = self.theta(h)  # (..., total_dim)
        trend_c = coeffs[..., : self.trend_dim]
        wavelet_c = coeffs[..., self.trend_dim :]
        dtype = coeffs.dtype
        out = trend_c @ self.trend_basis.to(dtype) + wavelet_c @ self.wavelet_basis.to(
            dtype
        )
        return F.silu(out) if self.active_g else out

    # Inherits from_pretrained_linear_* from TrendWaveletLinear via cls +
    # _tw_init_fn override.


class TrendWaveletLinearLGReduced(TrendWaveletLinearReduced):
    """Learned-gate + reduction variant of ``TrendWaveletLinear``.

    Combines the dimensionality-reduction stage with a per-dimension sigmoid
    gate on the coefficient vector.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        trend_dim=3,
        wavelet_dim=28,
        wavelet_type="db3",
        basis_offset=0,
        active_g=False,
        reduction_dim=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            trend_dim,
            wavelet_dim,
            wavelet_type,
            basis_offset,
            active_g=active_g,
            reduction_dim=reduction_dim,
        )
        self.coeff_gate = nn.Parameter(torch.ones(self.theta.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.reduction(x))
        coeffs = torch.sigmoid(self.coeff_gate) * self.theta(h)
        dtype = coeffs.dtype
        out = coeffs[..., : self.trend_dim] @ self.trend_basis.to(dtype) + coeffs[
            ..., self.trend_dim :
        ] @ self.wavelet_basis.to(dtype)
        return F.silu(out) if self.active_g else out


class TrendWaveletGenericLinearReduced(TrendWaveletGenericLinear):
    """TrendWaveletGeneric with a learned dimensionality-reduction stage before ``theta``.

    Three-way decomposition (trend + wavelet + generic) with a reduction layer
    that projects the input to a lower dimension before the coefficient projection.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        trend_dim=3,
        wavelet_dim=28,
        wavelet_type="db3",
        basis_offset=0,
        generic_dim=5,
        active_g=False,
        reduction_dim=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            trend_dim,
            wavelet_dim,
            wavelet_type,
            basis_offset,
            generic_dim,
            active_g=active_g,
        )
        if reduction_dim is None:
            reduction_dim = in_features // 2
        self.reduction_dim = reduction_dim
        self.reduction = nn.Linear(in_features, reduction_dim, bias=False)
        total_coeff_dim = self.theta.out_features
        self.theta = nn.Linear(reduction_dim, total_coeff_dim, bias=bias)

    # Override init hook — reduced generic layers need SVD-factored reduction init
    _tw_init_fn = staticmethod(_init_tw_generic_reduced_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.reduction(x))
        coeffs = self.theta(h)
        wave_end = self.trend_dim + self.effective_wavelet_dim
        dtype = coeffs.dtype
        out = (
            coeffs[..., : self.trend_dim] @ self.trend_basis.to(dtype)
            + coeffs[..., self.trend_dim : wave_end] @ self.wavelet_basis.to(dtype)
            + self.generic_basis(coeffs[..., wave_end:])
        )
        return F.silu(out) if self.active_g else out

    # Inherits from_pretrained_linear_* from TrendWaveletGenericLinear via cls +
    # _tw_init_fn override.


class TrendWaveletGenericLinearLGReduced(TrendWaveletGenericLinearReduced):
    """Learned-gate + reduction variant of ``TrendWaveletGenericLinear``.

    Combines the dimensionality-reduction stage with a per-dimension sigmoid
    gate on the three-way coefficient vector (trend + wavelet + generic).
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        trend_dim=3,
        wavelet_dim=28,
        wavelet_type="db3",
        basis_offset=0,
        generic_dim=5,
        active_g=False,
        reduction_dim=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            trend_dim,
            wavelet_dim,
            wavelet_type,
            basis_offset,
            generic_dim,
            active_g=active_g,
            reduction_dim=reduction_dim,
        )
        self.coeff_gate = nn.Parameter(torch.ones(self.theta.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.reduction(x))
        coeffs = torch.sigmoid(self.coeff_gate) * self.theta(h)
        wave_end = self.trend_dim + self.effective_wavelet_dim
        dtype = coeffs.dtype
        out = (
            coeffs[..., : self.trend_dim] @ self.trend_basis.to(dtype)
            + coeffs[..., self.trend_dim : wave_end] @ self.wavelet_basis.to(dtype)
            + self.generic_basis(coeffs[..., wave_end:])
        )
        return F.silu(out) if self.active_g else out


# ---------------------------------------------------------------------------
# SVD-basis linear layers
# ---------------------------------------------------------------------------


class SVDLinear(nn.Module):
    """Rank-k linear layer using top-k SVD components of the pretrained weight matrix.

    By the Eckart-Young theorem, this gives the **optimal** rank-k approximation
    of the original weight W:

    - Frozen basis: ``svd_basis = U_k.T``  shape ``(rank, out_features)``
    - Trainable theta initialized as ``diag(S_k) @ Vh_k``  shape ``(rank, in_features)``
    - Forward: ``y = theta(x) @ svd_basis``

    At initialization the layer is exactly equal to the best rank-k approximation
    of W (no lstsq error on top of rank truncation). During fine-tuning, theta
    adapts freely within the frozen output subspace.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.theta = nn.Linear(in_features, rank, bias=bias)
        self.register_buffer("svd_basis", torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.theta(x) @ self.svd_basis.to(x.dtype)

    @classmethod
    def from_pretrained_linear(
        cls,
        linear: nn.Linear,
        rank: int,
    ) -> "SVDLinear":
        """Initialize from a pretrained ``nn.Linear`` via truncated SVD.

        Computes W = U @ S @ Vh and stores U_k.T as the frozen basis, initializing
        theta.weight = diag(S_k) @ Vh_k so the layer exactly reproduces the
        optimal rank-k approximation of W at t=0.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
        )
        W = linear.weight.float()  # (out_features, in_features)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        layer.svd_basis.copy_(U[:, :rank].T)  # (rank, out_features)
        with torch.no_grad():
            layer.theta.weight.copy_(
                (S[:rank, None] * Vh[:rank, :]).to(linear.weight.dtype)
            )
            if linear.bias is not None and layer.theta.bias is not None:
                layer.theta.bias.zero_()
        return layer


class SVDLinearLG(SVDLinear):
    """Learned-gate variant of ``SVDLinear``.

    Adds a per-dimension sigmoid gate on the coefficient vector, allowing the
    network to discover which rank dimensions are most useful during training.

    Gate init: ``ones`` → ``sigmoid(1) ≈ 0.73`` (matching LG convention).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
    ):
        super().__init__(in_features, out_features, rank, bias)
        self.coeff_gate = nn.Parameter(torch.ones(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = torch.sigmoid(self.coeff_gate) * self.theta(x)
        return coeffs @ self.svd_basis.to(x.dtype)

    @classmethod
    def from_pretrained_linear(
        cls,
        linear: nn.Linear,
        rank: int,
    ) -> "SVDLinearLG":
        """Initialize from a pretrained ``nn.Linear`` via truncated SVD."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
        )
        W = linear.weight.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        layer.svd_basis.copy_(U[:, :rank].T)
        with torch.no_grad():
            layer.theta.weight.copy_(
                (S[:rank, None] * Vh[:rank, :]).to(linear.weight.dtype)
            )
            if linear.bias is not None and layer.theta.bias is not None:
                layer.theta.bias.zero_()
        return layer


class PEBottleneckMLP(nn.Module):
    """AE-bottleneck replacement for LlamaMLP (SwiGLU).

    Mirrors ``AERootBlock``: ``hidden -> hidden/2 -> latent_dim -> hidden/2 -> hidden``

    The bottleneck IS the parameter savings — no pruning needed.
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        self.fc1 = nn.Linear(hidden_size, mid_dim)
        self.fc2 = nn.Linear(mid_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, mid_dim)
        self.fc4 = nn.Linear(mid_dim, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc4(x)

    @classmethod
    def from_pretrained_mlp(cls, mlp, config) -> "PEBottleneckMLP":
        """Initialize from a pretrained LlamaMLP via direct truncation.

        Supported source: any module with ``gate_proj``, ``up_proj``, and
        ``down_proj`` ``nn.Linear`` attributes (i.e. ``LlamaMLP`` / SwiGLU-style
        MLP).

        Initialization mapping
        ----------------------
        ``fc1`` ← ``gate_proj.weight[:mid_dim, :]``
            Direct row truncation; preserves the first ``mid_dim`` output
            directions of the pretrained gate projection.
        ``fc2`` ← ``W_sub[:latent_dim, :]`` (row truncation of ``W_sub``)
            First ``latent_dim`` rows of the ``(mid_dim, mid_dim)`` submatrix.
        ``fc3`` ← ``W_sub[:, :latent_dim]`` (column truncation of ``W_sub``)
            First ``latent_dim`` columns of the ``(mid_dim, mid_dim)`` submatrix.
        ``fc4`` ← ``down_proj.weight[:, :mid_dim]``
            Direct column truncation; preserves the first ``mid_dim`` input
            directions of the pretrained down projection.

        where ``W_sub = gate_proj.weight[:mid_dim, :mid_dim]`` and
        ``mid_dim = hidden_size // 2``.

        All biases are zeroed.

        .. note::
            This is a **best-effort structural approximation only**.
            ``PEBottleneckMLP`` (4-layer AE bottleneck with SiLU activations)
            and ``LlamaMLP`` (SwiGLU gating) have incompatible forward passes.
            ``up_proj`` is intentionally unused — it has no structural analog in
            the bottleneck. Fine-tuning is required after initialization.

        Args:
            mlp: A pretrained ``LlamaMLP`` instance (or any module exposing
                ``gate_proj``, ``up_proj``, ``down_proj`` as ``nn.Linear``).
            config: A ``PELlamaConfig`` providing ``hidden_size`` and
                ``ae_latent_dim``.

        Returns:
            A new ``PEBottleneckMLP`` with weights initialized from ``mlp``.
        """
        orig_dtype = mlp.gate_proj.weight.dtype
        layer = cls(config)
        layer = layer.to(orig_dtype)
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        W_gate = mlp.gate_proj.weight.float()  # (intermediate_size, hidden_size)
        W_down = mlp.down_proj.weight.float()  # (hidden_size, intermediate_size)

        # fc1: first mid_dim rows of gate_proj  →  (mid_dim, hidden_size)
        fc1_w = W_gate[:mid_dim, :]

        # fc4: first mid_dim columns of down_proj  →  (hidden_size, mid_dim)
        fc4_w = W_down[:, :mid_dim]

        # fc2 / fc3: SVD or algorithm-specific (truncation)
        ae_inner_init = getattr(config, "ae_inner_init", "svd")
        if ae_inner_init == "svd":
            fc2_w, fc3_w = _svd_fc2fc3(fc1_w, latent_dim)
        else:
            W_sub = W_gate[:mid_dim, :mid_dim]  # (mid_dim, mid_dim)
            fc2_w = W_sub[:latent_dim, :]  # (latent_dim, mid_dim) — encoder
            fc3_w = W_sub[:, :latent_dim]  # (mid_dim, latent_dim) — decoder

        with torch.no_grad():
            layer.fc1.weight.copy_(fc1_w.to(orig_dtype))
            layer.fc2.weight.copy_(fc2_w.to(orig_dtype))
            layer.fc3.weight.copy_(fc3_w.to(orig_dtype))
            layer.fc4.weight.copy_(fc4_w.to(orig_dtype))
            if layer.fc1.bias is not None:
                layer.fc1.bias.zero_()
                layer.fc2.bias.zero_()
                layer.fc3.bias.zero_()
                layer.fc4.bias.zero_()
        return layer

    @classmethod
    def from_pretrained_mlp_svd(cls, mlp, config) -> "PEBottleneckMLP":
        """Initialize from a pretrained LlamaMLP using full SVD on all matrices.

        Unlike :meth:`from_pretrained_mlp` which truncates rows/columns of
        ``gate_proj`` and ``down_proj``, this method uses truncated SVD on the
        **full** weight matrices so that the optimal rank-``mid_dim`` directions
        are preserved rather than an arbitrary slice.

        Initialization mapping
        ----------------------
        ``fc1`` ← rank-``mid_dim`` SVD of ``gate_proj.weight`` (8192 × 2048)
            ``fc1.weight = diag(S[:mid_dim]) @ Vh[:mid_dim, :]``
            shape ``(mid_dim, hidden_size)`` — captures the ``mid_dim`` most
            important input→intermediate directions.
        ``fc4`` ← rank-``mid_dim`` SVD of ``down_proj.weight`` (2048 × 8192)
            ``fc4.weight = U[:, :mid_dim] * S[:mid_dim]``
            shape ``(hidden_size, mid_dim)`` — captures the ``mid_dim`` most
            important intermediate→output directions.
        ``fc2`` / ``fc3`` ← rank-``latent_dim`` SVD of
            ``gate_proj.weight[:, :mid_dim]`` (8192 × mid_dim):
            ``fc2.weight = Vh[:latent_dim, :]`` (encoder),
            ``fc3.weight = U[:, :latent_dim] * S[:latent_dim]`` (decoder).

        All biases are zeroed.  SVD is computed in ``float32``; weights are cast
        back to the source dtype before copying.

        Args:
            mlp: A pretrained ``LlamaMLP`` instance (or any module exposing
                ``gate_proj``, ``up_proj``, ``down_proj`` as ``nn.Linear``).
            config: A ``PELlamaConfig`` providing ``hidden_size`` and
                ``ae_latent_dim``.

        Returns:
            A new ``PEBottleneckMLP`` with weights initialized from ``mlp``.
        """
        orig_dtype = mlp.gate_proj.weight.dtype
        layer = cls(config)
        layer = layer.to(orig_dtype)
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        W_gate = mlp.gate_proj.weight.float()  # (intermediate_size, hidden_size)
        W_down = mlp.down_proj.weight.float()  # (hidden_size, intermediate_size)

        # fc1: rank-mid_dim SVD of gate_proj  →  (mid_dim, hidden_size)
        # W_gate ≈ U_g @ diag(S_g) @ Vh_g
        # fc1.weight = diag(S_g[:mid_dim]) @ Vh_g[:mid_dim, :]
        U_g, S_g, Vh_g = torch.linalg.svd(W_gate, full_matrices=False)
        fc1_w = S_g[:mid_dim, None] * Vh_g[:mid_dim, :]  # (mid_dim, hidden_size)

        # fc4: rank-mid_dim SVD of down_proj  →  (hidden_size, mid_dim)
        # W_down ≈ U_d @ diag(S_d) @ Vh_d
        # fc4.weight = U_d[:, :mid_dim] * S_d[:mid_dim]
        U_d, S_d, _Vh_d = torch.linalg.svd(W_down, full_matrices=False)
        fc4_w = U_d[:, :mid_dim] * S_d[:mid_dim]  # (hidden_size, mid_dim)

        # fc2 / fc3: rank-latent_dim SVD of the full gate_proj weight
        # projected into the mid_dim output subspace defined by fc1's SVD.
        # The mid_dim left singular vectors U_g[:, :mid_dim] span fc1's output
        # space; project gate_proj into that subspace for the inner bottleneck.
        # The inner bottleneck (fc2/fc3) operates in the mid_dim-dimensional
        # space defined by fc1's output.  Use an identity-like SVD of that
        # subspace to seed fc2/fc3 with a smooth rank-latent_dim factorization.
        W_mid = U_g[:, :mid_dim].T @ U_g[:, :mid_dim]  # (mid_dim, mid_dim)
        U_m, S_m, Vh_m = torch.linalg.svd(W_mid, full_matrices=False)
        fc2_w = Vh_m[:latent_dim, :]  # (latent_dim, mid_dim)
        fc3_w = U_m[:, :latent_dim] * S_m[:latent_dim]  # (mid_dim, latent_dim)

        with torch.no_grad():
            layer.fc1.weight.copy_(fc1_w.to(orig_dtype))
            layer.fc2.weight.copy_(fc2_w.to(orig_dtype))
            layer.fc3.weight.copy_(fc3_w.to(orig_dtype))
            layer.fc4.weight.copy_(fc4_w.to(orig_dtype))
            if layer.fc1.bias is not None:
                layer.fc1.bias.zero_()
                layer.fc2.bias.zero_()
                layer.fc3.bias.zero_()
                layer.fc4.bias.zero_()
        return layer

    @classmethod
    def from_pretrained_mlp_cur(cls, mlp, config) -> "PEBottleneckMLP":
        """Initialize from a pretrained LlamaMLP using CUR decomposition.

        Unlike :meth:`from_pretrained_mlp` which truncates the first ``mid_dim``
        rows/columns, this method uses **leverage scores** from SVD to select the
        highest-leverage rows of ``gate_proj`` and columns of ``down_proj``.
        Selected rows/columns are actual entries from the original weight matrices,
        preserving scale, distribution, and sparsity patterns.

        Initialization mapping
        ----------------------
        ``fc1`` ← top-``mid_dim`` rows of ``gate_proj.weight`` by row leverage score
            Row leverage: ``l_i = ||U[i, :k]||² / k`` for rank ``k = mid_dim``.
        ``fc4`` ← top-``mid_dim`` columns of ``down_proj.weight`` by column leverage
            Column leverage: ``l_j = ||Vh[:k, j]||² / k``.
        ``fc2`` / ``fc3`` ← CUR selection of ``fc1_w[:, :mid_dim]``
            Row leverage for fc2, column leverage for fc3.

        All biases are zeroed.  SVD is computed in ``float32``; weights are cast
        back to the source dtype before copying.

        Args:
            mlp: A pretrained ``LlamaMLP`` instance.
            config: A ``PELlamaConfig`` providing ``hidden_size`` and
                ``ae_latent_dim``.

        Returns:
            A new ``PEBottleneckMLP`` with weights initialized from ``mlp``.
        """
        orig_dtype = mlp.gate_proj.weight.dtype
        layer = cls(config)
        layer = layer.to(orig_dtype)
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        W_gate = mlp.gate_proj.weight.float()  # (intermediate_size, hidden_size)
        W_down = mlp.down_proj.weight.float()  # (hidden_size, intermediate_size)

        # fc1: CUR row selection from gate_proj via leverage scores
        U_g, _S_g, _Vh_g = torch.linalg.svd(W_gate, full_matrices=False)
        row_leverage = (U_g[:, :mid_dim] ** 2).sum(dim=1) / mid_dim
        top_row_indices = torch.argsort(row_leverage, descending=True)[:mid_dim]
        top_row_indices, _ = torch.sort(top_row_indices)  # preserve ordering
        fc1_w = W_gate[top_row_indices, :]  # (mid_dim, hidden_size)

        # fc4: CUR column selection from down_proj via leverage scores
        _U_d, _S_d, Vh_d = torch.linalg.svd(W_down, full_matrices=False)
        col_leverage = (Vh_d[:mid_dim, :] ** 2).sum(dim=0) / mid_dim
        top_col_indices = torch.argsort(col_leverage, descending=True)[:mid_dim]
        top_col_indices, _ = torch.sort(top_col_indices)  # preserve ordering
        fc4_w = W_down[:, top_col_indices]  # (hidden_size, mid_dim)

        # fc2 / fc3: SVD or algorithm-specific (CUR selection)
        ae_inner_init = getattr(config, "ae_inner_init", "svd")
        if ae_inner_init == "svd":
            fc2_w, fc3_w = _svd_fc2fc3(fc1_w, latent_dim)
        else:
            W_sub = fc1_w[:, :mid_dim]  # (mid_dim, mid_dim)
            U_sub, _S_sub, Vh_sub = torch.linalg.svd(W_sub, full_matrices=False)
            # fc2: select top-latent_dim rows by row leverage
            row_lev = (U_sub[:, :latent_dim] ** 2).sum(dim=1) / latent_dim
            top_rows = torch.argsort(row_lev, descending=True)[:latent_dim]
            top_rows, _ = torch.sort(top_rows)
            fc2_w = W_sub[top_rows, :]  # (latent_dim, mid_dim)
            # fc3: select top-latent_dim columns by column leverage
            col_lev = (Vh_sub[:latent_dim, :] ** 2).sum(dim=0) / latent_dim
            top_cols = torch.argsort(col_lev, descending=True)[:latent_dim]
            top_cols, _ = torch.sort(top_cols)
            fc3_w = W_sub[:, top_cols]  # (mid_dim, latent_dim)

        with torch.no_grad():
            layer.fc1.weight.copy_(fc1_w.to(orig_dtype))
            layer.fc2.weight.copy_(fc2_w.to(orig_dtype))
            layer.fc3.weight.copy_(fc3_w.to(orig_dtype))
            layer.fc4.weight.copy_(fc4_w.to(orig_dtype))
            if layer.fc1.bias is not None:
                layer.fc1.bias.zero_()
                layer.fc2.bias.zero_()
                layer.fc3.bias.zero_()
                layer.fc4.bias.zero_()
        return layer

    @classmethod
    def from_pretrained_mlp_fourier(cls, mlp, config) -> "PEBottleneckMLP":
        """Initialize from a pretrained LlamaMLP with Fourier domain filtering.

        Uses the same row/column truncation as :meth:`from_pretrained_mlp`
        (pretrained mode) but denoises the truncated weight matrices by
        retaining only the top-*k* frequency components per row (via FFT).

        The number of retained frequency components is ``ae_latent_dim``:
        lower latent dims get more aggressive filtering, creating a natural
        synergy between bottleneck capacity and initialization quality.

        Initialization mapping
        ----------------------
        ``fc1`` ← ``gate_proj.weight[:mid_dim, :]``, Fourier-filtered along dim=1
        ``fc4`` ← ``down_proj.weight[:, :mid_dim]``, Fourier-filtered along dim=0
        ``fc2`` / ``fc3`` ← Fourier-filtered truncation of the filtered submatrix

        All biases are zeroed.  Filtering is computed in ``float32``; weights
        are cast back to the source dtype before copying.

        Args:
            mlp: A pretrained ``LlamaMLP`` instance.
            config: A ``PELlamaConfig`` providing ``hidden_size`` and
                ``ae_latent_dim``.

        Returns:
            A new ``PEBottleneckMLP`` with weights initialized from ``mlp``.
        """
        orig_dtype = mlp.gate_proj.weight.dtype
        layer = cls(config)
        layer = layer.to(orig_dtype)
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        W_gate = mlp.gate_proj.weight.float()  # (intermediate_size, hidden_size)
        W_down = mlp.down_proj.weight.float()  # (hidden_size, intermediate_size)

        # fc1: truncate rows (same as pretrained), then Fourier-filter
        fc1_w = _fourier_filter_rows(W_gate[:mid_dim, :], latent_dim)

        # fc4: truncate columns, then Fourier-filter along dim=0
        fc4_w = _fourier_filter_rows(W_down[:, :mid_dim].T, latent_dim).T

        # fc2 / fc3: SVD or algorithm-specific (Fourier filtering)
        ae_inner_init = getattr(config, "ae_inner_init", "svd")
        if ae_inner_init == "svd":
            fc2_w, fc3_w = _svd_fc2fc3(fc1_w, latent_dim)
        else:
            W_sub = fc1_w[:, :mid_dim]  # (mid_dim, mid_dim)
            W_sub_filtered = _fourier_filter_rows(W_sub, latent_dim)
            fc2_w = W_sub_filtered[
                :latent_dim, :
            ]  # (latent_dim, mid_dim) — row truncation
            # Filter along columns (transpose, filter rows, transpose back), then truncate
            W_sub_filtered_cols = _fourier_filter_rows(W_sub.T, latent_dim).T
            fc3_w = W_sub_filtered_cols[
                :, :latent_dim
            ]  # (mid_dim, latent_dim) — col truncation

        with torch.no_grad():
            layer.fc1.weight.copy_(fc1_w.to(orig_dtype))
            layer.fc2.weight.copy_(fc2_w.to(orig_dtype))
            layer.fc3.weight.copy_(fc3_w.to(orig_dtype))
            layer.fc4.weight.copy_(fc4_w.to(orig_dtype))
            if layer.fc1.bias is not None:
                layer.fc1.bias.zero_()
                layer.fc2.bias.zero_()
                layer.fc3.bias.zero_()
                layer.fc4.bias.zero_()
        return layer


class PEBottleneckMLPLG(PEBottleneckMLP):
    """Learned-gate variant of ``PEBottleneckMLP``.

    Mirrors ``AERootBlockLG``: adds ``sigmoid(gate) * z`` at the bottleneck,
    allowing the network to discover effective latent dimensionality during
    training.
    """

    def __init__(self, config):
        super().__init__(config)
        self.latent_gate = nn.Parameter(torch.ones(config.ae_latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = x * torch.sigmoid(self.latent_gate)
        x = self.act(self.fc3(x))
        return self.fc4(x)

    @classmethod
    def from_pretrained_mlp(cls, mlp, config) -> "PEBottleneckMLPLG":
        """Initialize from a pretrained LlamaMLP.

        Delegates to :meth:`PEBottleneckMLP.from_pretrained_mlp`.  The
        ``latent_gate`` parameter is left at its ``__init__`` default of
        ``ones(latent_dim)`` (i.e. ``sigmoid(gate) ≈ 0.73`` at t=0, close to
        pass-through).
        """
        return super().from_pretrained_mlp(mlp, config)  # type: ignore[return-value]

    @classmethod
    def from_pretrained_mlp_svd(cls, mlp, config) -> "PEBottleneckMLPLG":
        """Initialize from a pretrained LlamaMLP using full SVD.

        Delegates to :meth:`PEBottleneckMLP.from_pretrained_mlp_svd`.  The
        ``latent_gate`` parameter is left at its ``__init__`` default of
        ``ones(latent_dim)`` (i.e. ``sigmoid(gate) ≈ 0.73`` at t=0, close to
        pass-through).
        """
        return super().from_pretrained_mlp_svd(mlp, config)  # type: ignore[return-value]

    @classmethod
    def from_pretrained_mlp_cur(cls, mlp, config) -> "PEBottleneckMLPLG":
        """Initialize from a pretrained LlamaMLP using CUR decomposition.

        Delegates to :meth:`PEBottleneckMLP.from_pretrained_mlp_cur`.  The
        ``latent_gate`` parameter is left at its ``__init__`` default of
        ``ones(latent_dim)`` (i.e. ``sigmoid(gate) ≈ 0.73`` at t=0, close to
        pass-through).
        """
        return super().from_pretrained_mlp_cur(mlp, config)  # type: ignore[return-value]

    @classmethod
    def from_pretrained_mlp_fourier(cls, mlp, config) -> "PEBottleneckMLPLG":
        """Initialize from a pretrained LlamaMLP with Fourier domain filtering.

        Delegates to :meth:`PEBottleneckMLP.from_pretrained_mlp_fourier`.  The
        ``latent_gate`` parameter is left at its ``__init__`` default of
        ``ones(latent_dim)`` (i.e. ``sigmoid(gate) ≈ 0.73`` at t=0, close to
        pass-through).
        """
        return super().from_pretrained_mlp_fourier(mlp, config)  # type: ignore[return-value]


class PEBottleneckMLPVAE(nn.Module):
    """Standard VAE bottleneck replacement for LlamaMLP.

    Mirrors ``AERootBlockVAE``/``VAE2RootBlock`` from N-BEATS-Lightning:
    ``hidden → mid_dim → (mu, logvar) → z → mid_dim → hidden``.

    The deterministic ``fc2`` bottleneck of ``PEBottleneckMLP`` is replaced by
    two parallel linear heads (``fc2_mu``, ``fc2_logvar``) plus the
    reparameterization trick.  The KL divergence term is weighted with a
    fixed coefficient of 1.0 (standard ELBO), unlike β-VAE variants that
    use a learnable or tunable β.

    During **training** the forward pass samples ``z ~ N(mu, exp(logvar))``;
    during **eval** it returns ``mu`` deterministically, so perplexity
    evaluation is unaffected by stochasticity.

    After each forward pass ``self.kl_loss`` holds the current KL divergence
    term (scalar tensor).  Training loops collect these and add
    ``kl_loss`` to the CE loss.
    """

    LOGVAR_MIN: float = -10.0
    LOGVAR_MAX: float = 4.0
    LOGVAR_INIT: float = -3.0  # → std ≈ 0.22 at t=0; near-deterministic start

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        self.fc1 = nn.Linear(hidden_size, mid_dim, bias=False)
        self.fc2_mu = nn.Linear(mid_dim, latent_dim, bias=False)
        self.fc2_logvar = nn.Linear(mid_dim, latent_dim, bias=False)
        self.fc3 = nn.Linear(latent_dim, mid_dim, bias=False)
        self.fc4 = nn.Linear(mid_dim, hidden_size, bias=False)
        self.act = nn.SiLU()

        # Side-channel updated each forward pass; consumed by the training loop.
        self.kl_loss: torch.Tensor = torch.tensor(0.0)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + std * eps during training; return mu during eval."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,I)), mean-reduced for batch-size independence."""
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))  # (*, mid_dim)
        mu = self.fc2_mu(h)  # (*, latent_dim)
        logvar = torch.clamp(self.fc2_logvar(h), self.LOGVAR_MIN, self.LOGVAR_MAX)
        z = self._reparameterize(mu, logvar)  # (*, latent_dim)
        self.kl_loss = self._kl_divergence(mu, logvar)
        return self.fc4(self.act(self.fc3(z)))  # (*, hidden_size)

    # ------------------------------------------------------------------
    # Shared initialization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _init_fc2_fc3(fc1_w: torch.Tensor, latent_dim: int, ae_inner_init: str):
        """Return (fc2_mu_w, fc3_w) for the given inner init strategy."""
        mid_dim = fc1_w.shape[0]
        if ae_inner_init == "svd":
            fc2_w, fc3_w = _svd_fc2fc3(fc1_w, latent_dim)
        else:
            W_sub = fc1_w[:, :mid_dim]  # (mid_dim, mid_dim)
            fc2_w = W_sub[:latent_dim, :]  # (latent_dim, mid_dim)
            fc3_w = W_sub[:, :latent_dim]  # (mid_dim, latent_dim)
        return fc2_w, fc3_w

    def _copy_weights(self, fc1_w, fc2_w, fc3_w, fc4_w, orig_dtype):
        """Copy computed weight tensors into this layer's parameters."""
        with torch.no_grad():
            self.fc1.weight.copy_(fc1_w.to(orig_dtype))
            self.fc2_mu.weight.copy_(fc2_w.to(orig_dtype))
            self.fc2_logvar.weight.fill_(self.LOGVAR_INIT)
            self.fc3.weight.copy_(fc3_w.to(orig_dtype))
            self.fc4.weight.copy_(fc4_w.to(orig_dtype))

    # ------------------------------------------------------------------
    # from_pretrained_* classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_mlp(cls, mlp, config) -> "PEBottleneckMLPVAE":
        """Initialize from a pretrained ``LlamaMLP`` via direct truncation.

        ``fc2_mu`` receives the same initialization as ``PEBottleneckMLP.fc2``
        (row/col truncation or SVD depending on ``config.ae_inner_init``).
        ``fc2_logvar`` is filled with ``LOGVAR_INIT = -3.0`` so the initial
        latent variance is small (near-deterministic at t=0).
        """
        orig_dtype = mlp.gate_proj.weight.dtype
        layer = cls(config)
        layer = layer.to(orig_dtype)
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        W_gate = mlp.gate_proj.weight.float()
        W_down = mlp.down_proj.weight.float()
        fc1_w = W_gate[:mid_dim, :]
        fc4_w = W_down[:, :mid_dim]
        ae_inner_init = getattr(config, "ae_inner_init", "svd")
        fc2_w, fc3_w = cls._init_fc2_fc3(fc1_w, latent_dim, ae_inner_init)
        layer._copy_weights(fc1_w, fc2_w, fc3_w, fc4_w, orig_dtype)
        return layer

    @classmethod
    def from_pretrained_mlp_svd(cls, mlp, config) -> "PEBottleneckMLPVAE":
        """Initialize from a pretrained ``LlamaMLP`` using full SVD on all matrices.

        Delegates fc1/fc4 computation to rank-``mid_dim`` SVD
        (same as ``PEBottleneckMLP.from_pretrained_mlp_svd``).
        ``fc2_mu`` uses the inner SVD strategy; ``fc2_logvar`` filled with
        ``LOGVAR_INIT``.
        """
        orig_dtype = mlp.gate_proj.weight.dtype
        layer = cls(config)
        layer = layer.to(orig_dtype)
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        W_gate = mlp.gate_proj.weight.float()
        W_down = mlp.down_proj.weight.float()

        U_g, S_g, Vh_g = torch.linalg.svd(W_gate, full_matrices=False)
        fc1_w = S_g[:mid_dim, None] * Vh_g[:mid_dim, :]
        U_d, S_d, _ = torch.linalg.svd(W_down, full_matrices=False)
        fc4_w = U_d[:, :mid_dim] * S_d[:mid_dim]

        W_mid = U_g[:, :mid_dim].T @ U_g[:, :mid_dim]
        U_m, S_m, Vh_m = torch.linalg.svd(W_mid, full_matrices=False)
        fc2_w = Vh_m[:latent_dim, :]
        fc3_w = U_m[:, :latent_dim] * S_m[:latent_dim]

        layer._copy_weights(fc1_w, fc2_w, fc3_w, fc4_w, orig_dtype)
        return layer

    @classmethod
    def from_pretrained_mlp_cur(cls, mlp, config) -> "PEBottleneckMLPVAE":
        """Initialize from a pretrained ``LlamaMLP`` using CUR decomposition.

        Delegates to the same CUR leverage-score selection as
        ``PEBottleneckMLP.from_pretrained_mlp_cur``.
        ``fc2_logvar`` filled with ``LOGVAR_INIT``.
        """
        orig_dtype = mlp.gate_proj.weight.dtype
        layer = cls(config)
        layer = layer.to(orig_dtype)
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        W_gate = mlp.gate_proj.weight.float()
        W_down = mlp.down_proj.weight.float()

        U_g, _, _ = torch.linalg.svd(W_gate, full_matrices=False)
        row_leverage = (U_g[:, :mid_dim] ** 2).sum(dim=1) / mid_dim
        top_rows = torch.sort(torch.argsort(row_leverage, descending=True)[:mid_dim])[0]
        fc1_w = W_gate[top_rows, :]

        _, _, Vh_d = torch.linalg.svd(W_down, full_matrices=False)
        col_leverage = (Vh_d[:mid_dim, :] ** 2).sum(dim=0) / mid_dim
        top_cols = torch.sort(torch.argsort(col_leverage, descending=True)[:mid_dim])[0]
        fc4_w = W_down[:, top_cols]

        ae_inner_init = getattr(config, "ae_inner_init", "svd")
        fc2_w, fc3_w = cls._init_fc2_fc3(fc1_w, latent_dim, ae_inner_init)
        layer._copy_weights(fc1_w, fc2_w, fc3_w, fc4_w, orig_dtype)
        return layer

    @classmethod
    def from_pretrained_mlp_fourier(cls, mlp, config) -> "PEBottleneckMLPVAE":
        """Initialize from a pretrained ``LlamaMLP`` with Fourier domain filtering.

        Applies FFT denoising to truncated weight matrices
        (same as ``PEBottleneckMLP.from_pretrained_mlp_fourier``).
        ``fc2_logvar`` filled with ``LOGVAR_INIT``.
        """
        orig_dtype = mlp.gate_proj.weight.dtype
        layer = cls(config)
        layer = layer.to(orig_dtype)
        hidden_size = config.hidden_size
        latent_dim = config.ae_latent_dim
        mid_dim = hidden_size // 2

        W_gate = mlp.gate_proj.weight.float()
        W_down = mlp.down_proj.weight.float()

        fc1_w = _fourier_filter_rows(W_gate[:mid_dim, :], latent_dim)
        fc4_w = _fourier_filter_rows(W_down[:, :mid_dim].T, latent_dim).T

        ae_inner_init = getattr(config, "ae_inner_init", "svd")
        if ae_inner_init == "svd":
            fc2_w, fc3_w = _svd_fc2fc3(fc1_w, latent_dim)
        else:
            W_sub = fc1_w[:, :mid_dim]
            W_sub_f = _fourier_filter_rows(W_sub, latent_dim)
            fc2_w = W_sub_f[:latent_dim, :]
            fc3_w = _fourier_filter_rows(W_sub.T, latent_dim).T[:, :latent_dim]

        layer._copy_weights(fc1_w, fc2_w, fc3_w, fc4_w, orig_dtype)
        return layer


class PEBottleneckMLPVAELG(PEBottleneckMLPVAE):
    """Learned-gate variant of ``PEBottleneckMLPVAE``.

    Adds a per-dimension ``sigmoid(latent_gate) * z`` scaling at the
    bottleneck, allowing the network to discover effective latent
    dimensionality during training (matching the ``PEBottleneckMLPLG``
    convention).

    Gate init: ``ones`` → ``sigmoid(1) ≈ 0.73`` at t=0.
    """

    def __init__(self, config):
        super().__init__(config)
        self.latent_gate = nn.Parameter(torch.ones(config.ae_latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = torch.clamp(self.fc2_logvar(h), self.LOGVAR_MIN, self.LOGVAR_MAX)
        z = self._reparameterize(mu, logvar)
        self.kl_loss = self._kl_divergence(mu, logvar)
        z = z * torch.sigmoid(self.latent_gate)
        return self.fc4(self.act(self.fc3(z)))

    @classmethod
    def from_pretrained_mlp(cls, mlp, config) -> "PEBottleneckMLPVAELG":
        """Delegates to :meth:`PEBottleneckMLPVAE.from_pretrained_mlp`.

        The ``latent_gate`` parameter is left at its ``__init__`` default of
        ``ones(latent_dim)`` (i.e. ``sigmoid(gate) ≈ 0.73`` at t=0).
        """
        return super().from_pretrained_mlp(mlp, config)  # type: ignore[return-value]

    @classmethod
    def from_pretrained_mlp_svd(cls, mlp, config) -> "PEBottleneckMLPVAELG":
        """Delegates to :meth:`PEBottleneckMLPVAE.from_pretrained_mlp_svd`.

        The ``latent_gate`` parameter is left at its ``__init__`` default.
        """
        return super().from_pretrained_mlp_svd(mlp, config)  # type: ignore[return-value]

    @classmethod
    def from_pretrained_mlp_cur(cls, mlp, config) -> "PEBottleneckMLPVAELG":
        """Delegates to :meth:`PEBottleneckMLPVAE.from_pretrained_mlp_cur`.

        The ``latent_gate`` parameter is left at its ``__init__`` default.
        """
        return super().from_pretrained_mlp_cur(mlp, config)  # type: ignore[return-value]

    @classmethod
    def from_pretrained_mlp_fourier(cls, mlp, config) -> "PEBottleneckMLPVAELG":
        """Delegates to :meth:`PEBottleneckMLPVAE.from_pretrained_mlp_fourier`.

        The ``latent_gate`` parameter is left at its ``__init__`` default.
        """
        return super().from_pretrained_mlp_fourier(mlp, config)  # type: ignore[return-value]


class AETrendWaveletLatentMLP(nn.Module):
    """N-BEATS-Lightning AERootBlock-style MLP replacement.

    The AE narrow waist *is* the trend+wavelet coefficient vector; a frozen
    structural-prior basis decodes ``coeff_dim -> hidden`` directly. There is no
    re-expansion stage and no learnable decoder — the basis serves as the AE's
    decoder. Drop-in replacement for ``LlamaMLP`` with matched I/O dims.

    Encoder::

        x: (..., hidden) -> fc1 -> SiLU -> fc2 -> SiLU -> z: (..., coeff_dim)

    Synthesis (frozen)::

        out = z[..., :T] @ trend_basis + z[..., T:] @ wavelet_basis

    where ``coeff_dim = trend_dim + effective_wavelet_dim``.
    """

    def __init__(
        self,
        config,
        wavelet_basis_offset: int | None = None,
        active_g: bool = False,
    ):
        super().__init__()
        hidden = config.hidden_size
        mid = hidden // 2
        offset = (
            wavelet_basis_offset
            if wavelet_basis_offset is not None
            else config.wavelet_basis_offset
        )

        trend_basis = build_trend_basis(config.trend_dim, hidden)
        wavelet_type = getattr(config, "mlp_wavelet_type", config.wavelet_type)
        wavelet_basis = build_wavelet_basis(
            hidden,
            wavelet_type,
            config.wavelet_dim,
            offset,
        )
        coeff_dim = config.trend_dim + wavelet_basis.shape[0]

        self.fc1 = nn.Linear(hidden, mid)
        self.fc2 = nn.Linear(mid, coeff_dim)
        self.act = nn.SiLU()
        self.register_buffer("trend_basis", trend_basis)
        self.register_buffer("wavelet_basis", wavelet_basis)
        self.trend_dim = config.trend_dim
        self.active_g = active_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.act(self.fc1(x))
        z = self.act(self.fc2(e))
        dtype = z.dtype
        out = z[..., : self.trend_dim] @ self.trend_basis.to(dtype) + z[
            ..., self.trend_dim :
        ] @ self.wavelet_basis.to(dtype)
        return F.silu(out) if self.active_g else out


class AETrendWaveletFilterMLP(nn.Module):
    """Hedged AE+basis MLP replacement.

    Full AE reconstructs at hidden_size (same shape as ``PEBottleneckMLP``);
    a frozen concatenated trend+wavelet basis ``B`` filters the reconstruction
    via ``x_recon @ B^T @ B``. The wavelet rows are SVD-orthonormalized in
    :func:`pellm.basis.build_wavelet_basis`, so the wavelet portion of this
    operation is a true projection onto its span; the trend rows (Vandermonde)
    are not orthonormal, so the overall operation is a Gram-weighted smoothing
    rather than a strict projection. In the limit of identity-like filter
    behavior the module degrades to plain ``PEBottleneckMLP``, which is the
    intended architectural hedge against a poor basis fit.
    """

    def __init__(
        self,
        config,
        wavelet_basis_offset: int | None = None,
        active_g: bool = False,
    ):
        super().__init__()
        hidden = config.hidden_size
        mid = hidden // 2
        latent = config.ae_latent_dim
        offset = (
            wavelet_basis_offset
            if wavelet_basis_offset is not None
            else config.wavelet_basis_offset
        )

        trend_basis = build_trend_basis(config.trend_dim, hidden)
        wavelet_type = getattr(config, "mlp_wavelet_type", config.wavelet_type)
        wavelet_basis = build_wavelet_basis(
            hidden,
            wavelet_type,
            config.wavelet_dim,
            offset,
        )
        basis = torch.cat([trend_basis, wavelet_basis], dim=0)

        self.fc1 = nn.Linear(hidden, mid)
        self.fc2 = nn.Linear(mid, latent)
        self.fc3 = nn.Linear(latent, mid)
        self.fc4 = nn.Linear(mid, hidden)
        self.act = nn.SiLU()
        self.register_buffer("basis", basis)
        self.active_g = active_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.act(self.fc1(x))
        z = self.act(self.fc2(e))
        d = self.act(self.fc3(z))
        x_recon = self.fc4(d)
        dtype = x_recon.dtype
        B = self.basis.to(dtype)
        coeffs = x_recon @ B.transpose(-1, -2)
        out = coeffs @ B
        return F.silu(out) if self.active_g else out


class TrendWaveletRootMLP(nn.Module):
    """N-BEATS RootBlock-style MLP — minimal variant.

    The MLP is a single ``TrendWaveletLinear(hidden, hidden)``: ``theta``
    projects to the trend+wavelet coefficient vector and a frozen basis
    decodes back to ``hidden``. There is no narrow-waist autoencoder and no
    nonlinearity inside the MLP — only the residual layernorm and surrounding
    attention contribute nonlinearity to the layer. Floor variant in the
    AE-vs-RootBlock comparison: tests how much of the gain attributed to the
    AE backbone in :class:`AETrendWaveletLatentMLP` is really coming from the
    encoder vs the basis.
    """

    def __init__(
        self,
        config,
        wavelet_basis_offset: int | None = None,
        active_g: bool = False,
    ):
        super().__init__()
        hidden = config.hidden_size
        offset = (
            wavelet_basis_offset
            if wavelet_basis_offset is not None
            else config.wavelet_basis_offset
        )
        wavelet_type = getattr(config, "mlp_wavelet_type", config.wavelet_type)
        self.tw = TrendWaveletLinear(
            in_features=hidden,
            out_features=hidden,
            bias=False,
            trend_dim=config.trend_dim,
            wavelet_dim=config.wavelet_dim,
            wavelet_type=wavelet_type,
            basis_offset=offset,
            active_g=active_g,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tw(x)


class TrendWaveletRootFCMLP(nn.Module):
    """N-BEATS RootBlock-style MLP — canonical variant.

    Full-width pre-theta FC + SiLU, then ``TrendWaveletLinear(hidden, hidden)``
    handles the theta-projection and frozen basis synthesis. This is the
    closest analog to N-BEATS-Lightning's RootBlock: the pre-theta backbone
    preserves full hidden width (no compression), runs through one
    nonlinearity, then projects to the coefficient vector. Direct A/B
    against :class:`AETrendWaveletLatentMLP` isolates the "narrow-waist
    autoencoder vs flat-width backbone" question for LLM MLPs.
    """

    def __init__(
        self,
        config,
        wavelet_basis_offset: int | None = None,
        active_g: bool = False,
    ):
        super().__init__()
        hidden = config.hidden_size
        offset = (
            wavelet_basis_offset
            if wavelet_basis_offset is not None
            else config.wavelet_basis_offset
        )
        self.fc = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()
        wavelet_type = getattr(config, "mlp_wavelet_type", config.wavelet_type)
        self.tw = TrendWaveletLinear(
            in_features=hidden,
            out_features=hidden,
            bias=False,
            trend_dim=config.trend_dim,
            wavelet_dim=config.wavelet_dim,
            wavelet_type=wavelet_type,
            basis_offset=offset,
            active_g=active_g,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tw(self.act(self.fc(x)))


class TrendWaveletRootPostFCMLP(nn.Module):
    """RootBlock with post-basis FC mixer.

    ``x -> TrendWaveletLinear -> SiLU -> nn.Linear(hidden, hidden) -> (SiLU if active_g)``.
    Adds a learned full-width linear after the frozen basis decode so the
    basis output can be re-mixed before the residual add.
    """

    def __init__(
        self,
        config,
        wavelet_basis_offset: int | None = None,
        active_g: bool = False,
    ):
        super().__init__()
        hidden = config.hidden_size
        offset = (
            wavelet_basis_offset
            if wavelet_basis_offset is not None
            else config.wavelet_basis_offset
        )
        wavelet_type = getattr(config, "mlp_wavelet_type", config.wavelet_type)
        self.tw = TrendWaveletLinear(
            in_features=hidden,
            out_features=hidden,
            bias=False,
            trend_dim=config.trend_dim,
            wavelet_dim=config.wavelet_dim,
            wavelet_type=wavelet_type,
            basis_offset=offset,
            active_g=True,
        )
        self.post_fc = nn.Linear(hidden, hidden)
        self.active_g = active_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.post_fc(self.tw(x))
        return F.silu(out) if self.active_g else out


class TrendWaveletRootFCPostFCMLP(nn.Module):
    """RootBlock with pre- and post-basis FC mixers.

    ``x -> nn.Linear -> SiLU -> TrendWaveletLinear -> SiLU -> nn.Linear -> (SiLU if active_g)``.
    Combines the pre-theta backbone of ``TrendWaveletRootFCMLP`` with a
    post-basis mixer, sandwiching the frozen basis between two learned
    full-width linears.
    """

    def __init__(
        self,
        config,
        wavelet_basis_offset: int | None = None,
        active_g: bool = False,
    ):
        super().__init__()
        hidden = config.hidden_size
        offset = (
            wavelet_basis_offset
            if wavelet_basis_offset is not None
            else config.wavelet_basis_offset
        )
        wavelet_type = getattr(config, "mlp_wavelet_type", config.wavelet_type)
        self.pre_fc = nn.Linear(hidden, hidden)
        self.pre_act = nn.SiLU()
        self.tw = TrendWaveletLinear(
            in_features=hidden,
            out_features=hidden,
            bias=False,
            trend_dim=config.trend_dim,
            wavelet_dim=config.wavelet_dim,
            wavelet_type=wavelet_type,
            basis_offset=offset,
            active_g=True,
        )
        self.post_fc = nn.Linear(hidden, hidden)
        self.active_g = active_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_act(self.pre_fc(x))
        out = self.post_fc(self.tw(h))
        return F.silu(out) if self.active_g else out
