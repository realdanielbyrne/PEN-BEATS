"""
PE-Llama model: HuggingFace-compatible extension of Llama 3.2 that replaces
attention projections and/or MLP blocks with parameter-efficient alternatives
inspired by N-BEATS block architectures.

Layer options
-------------
Attention projections:
  - ``nn.Linear`` (standard): unmodified attention projections
  - ``TrendWaveletLinear``: frozen Vandermonde + DWT basis expansion

MLP blocks:
  - Standard ``LlamaMLP`` (SwiGLU)
  - ``PEBottleneckMLP``: AE-bottleneck (hidden -> hidden/2 -> latent -> hidden/2 -> hidden)
  - ``PEBottleneckMLPLG``: AE-bottleneck with learned per-dimension gate
  - ``PEBottleneckMLPVAE``: β-VAE bottleneck with learnable β (stochastic latent space)
  - ``PEBottleneckMLPVAELG``: β-VAE bottleneck + learned gate

Weight loading
--------------
``from_pretrained_llama()`` loads a standard Llama checkpoint, creates the PE
model, projects attention weights onto TrendWavelet bases (if enabled), and
copies all non-replaced weights directly. AE MLP layers use random init.
"""

from __future__ import annotations

import logging

from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM as _LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)

from .configuration_pe_llama import PELlamaConfig
from .pe_layers import (
    AETrendWaveletFilterMLP,
    AETrendWaveletLatentMLP,
    PEBottleneckMLP,
    PEBottleneckMLPLG,
    PEBottleneckMLPVAE,
    PEBottleneckMLPVAELG,
    TrendWaveletLinear,
    TrendWaveletGenericLinear,
    TrendWaveletLinearLG,
    TrendWaveletGenericLinearLG,
    TrendWaveletLinearReduced,
    TrendWaveletLinearLGReduced,
    TrendWaveletGenericLinearReduced,
    TrendWaveletGenericLinearLGReduced,
    TrendWaveletRootFCMLP,
    TrendWaveletRootFCPostFCMLP,
    TrendWaveletRootMLP,
    TrendWaveletRootPostFCMLP,
    SVDLinear,
    SVDLinearLG,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Targeting helpers
# ---------------------------------------------------------------------------

_ALL_PROJ_NAMES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
_PE_ATTN_MODES: frozenset[str] = frozenset(
    (
        "trend_wavelet",
        "trend_wavelet_generic",
        "trend_wavelet_lg",
        "trend_wavelet_generic_lg",
        "trend_wavelet_reduced",
        "trend_wavelet_lg_reduced",
        "trend_wavelet_generic_reduced",
        "trend_wavelet_generic_lg_reduced",
        "svd",
        "svd_lg",
    )
)
_PROJ_SHORT_MAP: dict[str, str] = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
}
_TW_ATTN_CLS_MAP: dict[str, type] = {
    "trend_wavelet": TrendWaveletLinear,
    "trend_wavelet_generic": TrendWaveletGenericLinear,
    "trend_wavelet_lg": TrendWaveletLinearLG,
    "trend_wavelet_generic_lg": TrendWaveletGenericLinearLG,
    "trend_wavelet_reduced": TrendWaveletLinearReduced,
    "trend_wavelet_lg_reduced": TrendWaveletLinearLGReduced,
    "trend_wavelet_generic_reduced": TrendWaveletGenericLinearReduced,
    "trend_wavelet_generic_lg_reduced": TrendWaveletGenericLinearLGReduced,
}


def _resolve_proj_names(raw: list[str]) -> list[str]:
    """Expand short names (q/k/v/o) to full names and validate."""
    result = []
    for name in raw:
        full = _PROJ_SHORT_MAP.get(name, name)
        if full not in _ALL_PROJ_NAMES:
            raise ValueError(
                f"Unknown projection name: {name!r}. "
                f"Use q/k/v/o or full names {_ALL_PROJ_NAMES}."
            )
        result.append(full)
    return result


def _active_proj_names(config: "PELlamaConfig") -> list[str]:
    """Return the projection names that are active (None means all 4)."""
    if config.pe_proj_names is None:
        return list(_ALL_PROJ_NAMES)
    return list(config.pe_proj_names)


def _active_layer_indices(config: "PELlamaConfig") -> list[int]:
    """Return the decoder layer indices that are active (None means all)."""
    if config.pe_layer_indices is None:
        return list(range(config.num_hidden_layers))
    return list(config.pe_layer_indices)


def _copy_tw_projection_state(dst_proj, src_proj) -> None:
    """Copy shared TrendWavelet projection parameters/buffers."""
    with torch.no_grad():
        dst_proj.theta.weight.copy_(src_proj.theta.weight)
        if dst_proj.theta.bias is not None and src_proj.theta.bias is not None:
            dst_proj.theta.bias.copy_(src_proj.theta.bias)
        if hasattr(dst_proj, "generic_basis") and hasattr(src_proj, "generic_basis"):
            dst_proj.generic_basis.weight.copy_(src_proj.generic_basis.weight)
        if hasattr(dst_proj, "coeff_gate") and hasattr(src_proj, "coeff_gate"):
            dst_proj.coeff_gate.copy_(src_proj.coeff_gate)
        if hasattr(dst_proj, "reduction") and hasattr(src_proj, "reduction"):
            dst_proj.reduction.weight.copy_(src_proj.reduction.weight)
            if (
                dst_proj.reduction.bias is not None
                and src_proj.reduction.bias is not None
            ):
                dst_proj.reduction.bias.copy_(src_proj.reduction.bias)


def _init_trendwavelet_projection_from_linear(pe_cls, linear, init_mode: str, **kwargs):
    """Build a TrendWavelet projection initialized from a dense ``nn.Linear``."""
    if init_mode == "random":
        return pe_cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            **kwargs,
        ).to(linear.weight.dtype)

    method_name_map = {
        "pretrained": "from_pretrained_linear_pretrained",
        "lstsq": "from_pretrained_linear",
        "svd": "from_pretrained_linear_svd",
        "cur": "from_pretrained_linear_cur",
        "fourier": "from_pretrained_linear_fourier",
    }
    if init_mode not in method_name_map:
        raise ValueError(f"Unsupported attn_init_mode: {init_mode}")
    method = getattr(pe_cls, method_name_map[init_mode])
    return method(linear, **kwargs)


# ---------------------------------------------------------------------------
# PELlamaAttention
# ---------------------------------------------------------------------------


class PELlamaAttention(LlamaAttention):
    """
    Drop-in replacement for :class:`LlamaAttention` that conditionally uses
    ``TrendWaveletLinear`` or ``nn.Linear`` for attention projections based on
    ``config.pe_attn_mode``.
    """

    def __init__(self, config: PELlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        hidden_size: int = config.hidden_size  # type: ignore[assignment]
        num_attention_heads: int = config.num_attention_heads  # type: ignore[assignment]
        num_key_value_heads: int = config.num_key_value_heads  # type: ignore[assignment]
        attention_bias: bool = config.attention_bias  # type: ignore[assignment]

        # Projection output dimensions and vanilla classes for each slot.
        _proj_specs: dict[str, tuple[int, int, type[nn.Linear]]] = {
            "q_proj": (hidden_size, num_attention_heads * self.head_dim, nn.Linear),
            "k_proj": (hidden_size, num_key_value_heads * self.head_dim, nn.Linear),
            "v_proj": (hidden_size, num_key_value_heads * self.head_dim, nn.Linear),
            "o_proj": (num_attention_heads * self.head_dim, hidden_size, nn.Linear),
        }

        # Determine which slots this layer should replace
        active_layers = _active_layer_indices(config)
        active_projs = set(_active_proj_names(config))
        is_pe_layer = layer_idx in active_layers

        # Build per-layer TrendWavelet kwargs (shared across projections)
        offset = config.wavelet_basis_offset
        if config.per_layer_offsets and layer_idx < len(config.per_layer_offsets):
            offset = config.per_layer_offsets[layer_idx]
        tw_kwargs: dict = dict(
            trend_dim=config.trend_dim,
            wavelet_dim=config.wavelet_dim,
            wavelet_type=config.wavelet_type,
            basis_offset=offset,
            active_g=getattr(config, "active_g", False),
        )
        if config.pe_attn_mode in (
            "trend_wavelet_generic",
            "trend_wavelet_generic_lg",
            "trend_wavelet_generic_reduced",
            "trend_wavelet_generic_lg_reduced",
        ):
            tw_kwargs["generic_dim"] = config.generic_dim
        if config.pe_attn_mode in (
            "trend_wavelet_reduced",
            "trend_wavelet_lg_reduced",
            "trend_wavelet_generic_reduced",
            "trend_wavelet_generic_lg_reduced",
        ):
            if config.reduction_dim is not None:
                tw_kwargs["reduction_dim"] = config.reduction_dim

        # Map mode → PE class (None for standard)
        _mode_to_cls: dict[str, type] = {
            "trend_wavelet": TrendWaveletLinear,
            "trend_wavelet_generic": TrendWaveletGenericLinear,
            "trend_wavelet_lg": TrendWaveletLinearLG,
            "trend_wavelet_generic_lg": TrendWaveletGenericLinearLG,
            "trend_wavelet_reduced": TrendWaveletLinearReduced,
            "trend_wavelet_lg_reduced": TrendWaveletLinearLGReduced,
            "trend_wavelet_generic_reduced": TrendWaveletGenericLinearReduced,
            "trend_wavelet_generic_lg_reduced": TrendWaveletGenericLinearLGReduced,
            "trend_wavelet_no_postln": TrendWaveletLinear,
            "svd": SVDLinear,
            "svd_lg": SVDLinearLG,
        }
        pe_cls = _mode_to_cls.get(config.pe_attn_mode)
        _svd_modes = frozenset(("svd", "svd_lg"))

        for proj_name, (in_dim, out_dim, vanilla_cls) in _proj_specs.items():
            if is_pe_layer and pe_cls is not None and proj_name in active_projs:
                if config.pe_attn_mode in _svd_modes:
                    layer = pe_cls(
                        in_dim, out_dim, config.svd_rank, bias=attention_bias
                    )
                else:
                    layer = pe_cls(in_dim, out_dim, bias=attention_bias, **tw_kwargs)
            else:
                layer = vanilla_cls(in_dim, out_dim, bias=attention_bias)
            setattr(self, proj_name, layer)


# ---------------------------------------------------------------------------
# PELlamaDecoderLayer
# ---------------------------------------------------------------------------


class PELlamaDecoderLayer(LlamaDecoderLayer):
    """
    Drop-in replacement for :class:`LlamaDecoderLayer` that uses
    ``PELlamaAttention`` and conditionally replaces the MLP with an
    AE-bottleneck variant.

    The ``forward()`` method is copied verbatim from
    ``LlamaDecoderLayer.forward()`` so that the full decoder-layer logic is
    visible in this repository without needing to inspect the upstream
    HuggingFace source.
    """

    def __init__(self, config: PELlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = PELlamaAttention(config=config, layer_idx=layer_idx)

        # Skip post_attention_layernorm on PE attention layers when the
        # `trend_wavelet_no_postln` mode is active. The RMSNorm module is left
        # in place so the state-dict shape is unchanged; it is simply not
        # invoked in forward(). Non-PE layers retain stock Llama behavior.
        is_pe_attn_layer = layer_idx in _active_layer_indices(config)
        self._skip_post_attn_ln = (
            is_pe_attn_layer and config.pe_attn_mode == "trend_wavelet_no_postln"
        )

        # Per-layer MLP targeting: replace only specified layers (None = all)
        mlp_layers = config.pe_mlp_layer_indices
        is_mlp_layer = mlp_layers is None or layer_idx in mlp_layers

        # Resolve per-layer wavelet basis offset for basis-using MLP modes
        mlp_offset = config.wavelet_basis_offset
        if config.per_layer_offsets and layer_idx < len(config.per_layer_offsets):
            mlp_offset = config.per_layer_offsets[layer_idx]

        # Whether to apply SiLU on the final basis-projection output of the
        # trendwavelet-MLP modes. Independent of attention's ``active_g``.
        mlp_active_g = getattr(config, "mlp_active_g", False)

        if is_mlp_layer and config.pe_mlp_mode == "ae":
            self.mlp = PEBottleneckMLP(config)
        elif is_mlp_layer and config.pe_mlp_mode == "ae_lg":
            self.mlp = PEBottleneckMLPLG(config)
        elif is_mlp_layer and config.pe_mlp_mode == "vae":
            self.mlp = PEBottleneckMLPVAE(config)
        elif is_mlp_layer and config.pe_mlp_mode == "vae_lg":
            self.mlp = PEBottleneckMLPVAELG(config)
        elif is_mlp_layer and config.pe_mlp_mode == "ae_basis_latent":
            self.mlp = AETrendWaveletLatentMLP(
                config, wavelet_basis_offset=mlp_offset, active_g=mlp_active_g
            )
        elif is_mlp_layer and config.pe_mlp_mode == "ae_basis_reexpand":
            self.mlp = AETrendWaveletFilterMLP(
                config, wavelet_basis_offset=mlp_offset, active_g=mlp_active_g
            )
        elif is_mlp_layer and config.pe_mlp_mode == "tw_root":
            self.mlp = TrendWaveletRootMLP(
                config, wavelet_basis_offset=mlp_offset, active_g=mlp_active_g
            )
        elif is_mlp_layer and config.pe_mlp_mode == "tw_root_fc":
            self.mlp = TrendWaveletRootFCMLP(
                config, wavelet_basis_offset=mlp_offset, active_g=mlp_active_g
            )
        elif is_mlp_layer and config.pe_mlp_mode == "tw_root_post_fc":
            self.mlp = TrendWaveletRootPostFCMLP(
                config, wavelet_basis_offset=mlp_offset, active_g=mlp_active_g
            )
        elif is_mlp_layer and config.pe_mlp_mode == "tw_root_fc_post_fc":
            self.mlp = TrendWaveletRootFCPostFCMLP(
                config, wavelet_basis_offset=mlp_offset, active_g=mlp_active_g
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # ---------- Self-Attention ----------
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # ---------- Feed-Forward / MLP ----------
        residual = hidden_states
        if not self._skip_post_attn_ln:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# PELlamaPreTrainedModel
# ---------------------------------------------------------------------------


class PELlamaPreTrainedModel(LlamaPreTrainedModel):
    """Shared base class for PE-Llama models."""

    config_class = PELlamaConfig
    _no_split_modules = ["LlamaDecoderLayer", "PELlamaDecoderLayer"]


# ---------------------------------------------------------------------------
# PELlamaModel (backbone)
# ---------------------------------------------------------------------------


class PELlamaModel(LlamaModel):
    """Backbone model with every decoder layer replaced by ``PELlamaDecoderLayer``."""

    config_class = PELlamaConfig

    def __init__(self, config: PELlamaConfig):
        super().__init__(config)
        num_hidden_layers: int = config.num_hidden_layers  # type: ignore[assignment]
        self.layers = nn.ModuleList(
            [
                PELlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(num_hidden_layers)
            ]
        )


# ---------------------------------------------------------------------------
# Device-map helper
# ---------------------------------------------------------------------------


def _apply_device_map(model, device_map, no_split_module_classes=None):
    """Dispatch a CPU-resident model according to *device_map*.

    Mirrors the behavior of ``from_pretrained(..., device_map=...)`` for models
    that are built manually (weight manipulation before dispatch).
    """
    if device_map is None or device_map == {"": "cpu"} or device_map == "cpu":
        return model
    if device_map == "auto":
        from accelerate import dispatch_model, infer_auto_device_map

        model.tie_weights()  # must precede infer_auto_device_map
        computed = infer_auto_device_map(
            model,
            no_split_module_classes=no_split_module_classes or [],
        )
        return dispatch_model(model, device_map=computed)
    if isinstance(device_map, dict):
        if len(device_map) == 1 and "" in device_map:
            return model.to(device_map[""])
        from accelerate import dispatch_model

        return dispatch_model(model, device_map=device_map)
    # Plain string: "cuda", "cuda:0", "mps", etc.
    return model.to(device_map)


# ---------------------------------------------------------------------------
# PELlamaForCausalLM
# ---------------------------------------------------------------------------


class PELlamaForCausalLM(_LlamaForCausalLM):
    """
    Causal language model with parameter-efficient attention projections
    and/or MLP replacements.

    Use ``from_pretrained_llama()`` to load a standard Llama checkpoint into
    this PE model with automatic weight projection.
    """

    config_class = PELlamaConfig
    _no_split_modules = ["LlamaDecoderLayer", "PELlamaDecoderLayer"]

    def __init__(self, config: PELlamaConfig):
        super().__init__(config)
        self.model = PELlamaModel(config)

    @classmethod
    def from_pretrained_llama(
        cls,
        pretrained_model_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct",
        pe_config: PELlamaConfig | None = None,
        device_map: str = "auto",
        torch_dtype=None,
        **kwargs,
    ) -> "PELlamaForCausalLM":
        """Load a standard Llama checkpoint and convert to PE model.

        1. Loads vanilla ``LlamaForCausalLM`` from the checkpoint.
        2. Creates ``PELlamaForCausalLM`` with the PE config.
        3. If ``pe_attn_mode="trend_wavelet"``: projects attention weights via
           ``TrendWaveletLinear.from_pretrained_linear()``.
        4. If ``pe_mlp_mode`` is ``"ae"``, ``"ae_lg"``, ``"vae"``, or ``"vae_lg"``:
           MLP layers get structured or random init (logs a warning).
        5. Copies all non-replaced weights (embeddings, norms, lm_head).
        6. Returns model ready for fine-tuning.
        """
        import torch as _torch

        if torch_dtype is None:
            torch_dtype = _torch.float32

        # Load the standard Llama model
        logger.info(
            "Loading standard Llama model from %s", pretrained_model_name_or_path
        )
        base_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            device_map="cpu",  # load to CPU first for weight manipulation
            dtype=torch_dtype,
            **kwargs,
        )

        # Build PE config from the base config if not provided
        if pe_config is None:
            pe_config = PELlamaConfig(**base_model.config.to_dict())
        else:
            # Ensure architectural params match the base model
            for key in (
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "num_hidden_layers",
                "intermediate_size",
                "vocab_size",
            ):
                base_val = getattr(base_model.config, key)
                pe_val = getattr(pe_config, key)
                if base_val != pe_val:
                    logger.warning(
                        "Config mismatch: %s base=%s pe=%s, using base value",
                        key,
                        base_val,
                        pe_val,
                    )
                    setattr(pe_config, key, base_val)

        # Create PE model (random init)
        pe_model = cls(pe_config)
        pe_model = pe_model.to(dtype=torch_dtype)

        # Copy non-attention, non-MLP weights directly
        base_sd = base_model.state_dict()
        pe_sd = pe_model.state_dict()

        # Identify which keys belong to PE-replaced attention projections and MLPs.
        # Only PE-replaced (layer, proj) pairs are excluded from direct copy;
        # nn.Linear slots (non-targeted layers/projections) are copied directly.
        is_pe_attn_mode = pe_config.pe_attn_mode in _PE_ATTN_MODES
        active_layers_set = set(_active_layer_indices(pe_config))
        active_projs_list = _active_proj_names(pe_config)

        attn_proj_prefixes = []
        mlp_prefixes = []  # only MLP prefixes for PE-replaced layers
        active_mlp_layers = (
            set(pe_config.pe_mlp_layer_indices)
            if pe_config.pe_mlp_layer_indices is not None
            else set(range(pe_config.num_hidden_layers))
        )
        for layer_idx in range(pe_config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.self_attn."
            if is_pe_attn_mode and layer_idx in active_layers_set:
                for proj in active_projs_list:
                    attn_proj_prefixes.append(f"{prefix}{proj}.")
            if layer_idx in active_mlp_layers:
                mlp_prefixes.append(f"model.layers.{layer_idx}.mlp.")

        copied, skipped = 0, 0
        for key, param in base_sd.items():
            is_attn = any(key.startswith(p) for p in attn_proj_prefixes)
            is_mlp = any(key.startswith(p) for p in mlp_prefixes)

            if is_attn:  # PE-replaced slot — handled by lstsq projection below
                skipped += 1
                continue
            if is_mlp and pe_config.pe_mlp_mode in ("ae", "ae_lg", "vae", "vae_lg"):
                skipped += 1
                continue

            if key in pe_sd and pe_sd[key].shape == param.shape:
                pe_sd[key].copy_(param)
                copied += 1
            else:
                skipped += 1

        pe_model.load_state_dict(pe_sd, strict=False)
        logger.info("Copied %d parameters, skipped %d", copied, skipped)

        # Project attention weights onto TrendWavelet bases (only targeted slots)
        if is_pe_attn_mode:
            attn_init_mode = getattr(pe_config, "attn_init_mode", "pretrained")
            logger.info(
                "Projecting attention weights (mode=%s, init=%s, layers=%s, projs=%s)",
                pe_config.pe_attn_mode,
                attn_init_mode if pe_config.pe_attn_mode in _TW_ATTN_CLS_MAP else "n/a",
                pe_config.pe_layer_indices or "all",
                pe_config.pe_proj_names or "all",
            )
            for layer_idx in range(pe_config.num_hidden_layers):
                if layer_idx not in active_layers_set:
                    continue
                base_layer = base_model.model.layers[layer_idx].self_attn
                pe_layer = pe_model.model.layers[layer_idx].self_attn

                offset = pe_config.wavelet_basis_offset
                if pe_config.per_layer_offsets and layer_idx < len(
                    pe_config.per_layer_offsets
                ):
                    offset = pe_config.per_layer_offsets[layer_idx]

                tw_kwargs = dict(
                    trend_dim=pe_config.trend_dim,
                    wavelet_dim=pe_config.wavelet_dim,
                    wavelet_type=pe_config.wavelet_type,
                    basis_offset=offset,
                    active_g=getattr(pe_config, "active_g", False),
                )

                for proj_name in active_projs_list:
                    base_linear = getattr(base_layer, proj_name)
                    pe_proj = getattr(pe_layer, proj_name)

                    if pe_config.pe_attn_mode in _TW_ATTN_CLS_MAP:
                        tw_kwargs_for_proj = dict(tw_kwargs)
                        if pe_config.pe_attn_mode in (
                            "trend_wavelet_generic",
                            "trend_wavelet_generic_lg",
                            "trend_wavelet_generic_reduced",
                            "trend_wavelet_generic_lg_reduced",
                        ):
                            tw_kwargs_for_proj["generic_dim"] = pe_config.generic_dim
                        if (
                            pe_config.pe_attn_mode
                            in (
                                "trend_wavelet_reduced",
                                "trend_wavelet_lg_reduced",
                                "trend_wavelet_generic_reduced",
                                "trend_wavelet_generic_lg_reduced",
                            )
                            and pe_config.reduction_dim is not None
                        ):
                            tw_kwargs_for_proj["reduction_dim"] = (
                                pe_config.reduction_dim
                            )
                        projected = _init_trendwavelet_projection_from_linear(
                            _TW_ATTN_CLS_MAP[pe_config.pe_attn_mode],
                            base_linear,
                            attn_init_mode,
                            **tw_kwargs_for_proj,
                        )
                        _copy_tw_projection_state(pe_proj, projected)
                    elif pe_config.pe_attn_mode == "svd":
                        projected = SVDLinear.from_pretrained_linear(
                            base_linear,
                            pe_config.svd_rank,
                        )
                        pe_proj.svd_basis.copy_(projected.svd_basis)
                        pe_proj.theta.weight.data.copy_(projected.theta.weight.data)
                    elif pe_config.pe_attn_mode == "svd_lg":
                        projected = SVDLinearLG.from_pretrained_linear(
                            base_linear,
                            pe_config.svd_rank,
                        )
                        pe_proj.svd_basis.copy_(projected.svd_basis)
                        pe_proj.theta.weight.data.copy_(projected.theta.weight.data)
                        # coeff_gate stays at ones (already set in __init__)

                    if (
                        hasattr(projected, "theta")
                        and projected.theta.bias is not None
                        and pe_proj.theta.bias is not None
                        and pe_config.pe_attn_mode not in _TW_ATTN_CLS_MAP
                    ):
                        pe_proj.theta.bias.data.copy_(projected.theta.bias.data)

        _MLP_CLS_MAP = {
            "ae": PEBottleneckMLP,
            "ae_lg": PEBottleneckMLPLG,
            "vae": PEBottleneckMLPVAE,
            "vae_lg": PEBottleneckMLPVAELG,
        }
        if pe_config.pe_mlp_mode in _MLP_CLS_MAP and pe_config.ae_init_mode in (
            "pretrained",
            "svd",
            "cur",
            "fourier",
        ):
            init_method = pe_config.ae_init_mode
            logger.info(
                "Initializing AE/VAE MLP layers from pretrained LlamaMLP "
                "(ae_init_mode='%s' — fine-tuning required).",
                init_method,
            )
            mlp_cls = _MLP_CLS_MAP[pe_config.pe_mlp_mode]
            for layer_idx in active_mlp_layers:
                src_mlp = base_model.model.layers[layer_idx].mlp
                if init_method == "svd":
                    initialized = mlp_cls.from_pretrained_mlp_svd(src_mlp, pe_config)
                elif init_method == "cur":
                    initialized = mlp_cls.from_pretrained_mlp_cur(src_mlp, pe_config)
                elif init_method == "fourier":
                    initialized = mlp_cls.from_pretrained_mlp_fourier(
                        src_mlp, pe_config
                    )
                else:
                    initialized = mlp_cls.from_pretrained_mlp(src_mlp, pe_config)
                pe_model.model.layers[layer_idx].mlp.load_state_dict(
                    initialized.state_dict(), strict=False
                )
        elif pe_config.pe_mlp_mode in _MLP_CLS_MAP:
            logger.info(
                "AE/VAE MLP layers randomly initialized (ae_init_mode='random')."
            )

        return _apply_device_map(pe_model, device_map, cls._no_split_modules)

    @classmethod
    def from_pretrained_pe_llama(
        cls,
        pretrained_pe_path: str,
        pe_config: PELlamaConfig | None = None,
        torch_dtype=None,
        device_map=None,
        **kwargs,
    ) -> "PELlamaForCausalLM":
        """Load a previously saved PE model and optionally change PE modes.

        Supports staged training workflows:

        **Stage 1** — train TrendWavelet attention with standard MLP, save.
        **Stage 2** — reload with ``pe_mlp_mode="ae_lg"`` to train only the
        MLP replacements while keeping fitted attention weights.

        The reverse order (MLP first, attention second) also works: saved
        ``nn.Linear`` attention weights will be projected onto TrendWavelet
        bases via least-squares.

        Parameters
        ----------
        pretrained_pe_path : str
            Path to a directory saved via ``save_pretrained()``.
        pe_config : PELlamaConfig, optional
            New config with updated PE modes.  If ``None``, loads with the
            saved config (equivalent to plain ``from_pretrained``).
        torch_dtype : torch.dtype, optional
            Override weight dtype (default ``float32``).
        """
        import torch as _torch

        if torch_dtype is None:
            torch_dtype = _torch.float32

        # Load the saved PE model
        logger.info("Loading saved PE model from %s", pretrained_pe_path)
        saved_model = cls.from_pretrained(
            pretrained_pe_path,
            device_map="cpu",
            dtype=torch_dtype,
            **kwargs,
        )
        saved_config: PELlamaConfig = saved_model.config  # type: ignore[assignment]

        # If no new config, nothing to change — return as-is
        if pe_config is None:
            return _apply_device_map(saved_model, device_map, cls._no_split_modules)

        # Ensure architectural params match
        for key in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
        ):
            saved_val = getattr(saved_config, key)
            pe_val = getattr(pe_config, key)
            if saved_val != pe_val:
                logger.warning(
                    "Config mismatch: %s saved=%s new=%s, using saved value",
                    key,
                    saved_val,
                    pe_val,
                )
                setattr(pe_config, key, saved_val)

        attn_changed = pe_config.pe_attn_mode != saved_config.pe_attn_mode
        mlp_changed = pe_config.pe_mlp_mode != saved_config.pe_mlp_mode

        # Also detect same-mode MLP layer-set expansion (e.g. [15] → [14, 15])
        _saved_mlp_layers = set(
            saved_config.pe_mlp_layer_indices
            if saved_config.pe_mlp_layer_indices is not None
            else range(saved_config.num_hidden_layers)
        )
        _new_mlp_layers = set(
            pe_config.pe_mlp_layer_indices
            if pe_config.pe_mlp_layer_indices is not None
            else range(pe_config.num_hidden_layers)
        )
        mlp_layers_changed = _saved_mlp_layers != _new_mlp_layers

        # Warn if targeting changed between stages
        _saved_projs = set(saved_config.pe_proj_names or _ALL_PROJ_NAMES)
        _new_projs = set(pe_config.pe_proj_names or _ALL_PROJ_NAMES)
        _saved_layers = set(
            saved_config.pe_layer_indices or range(saved_config.num_hidden_layers)
        )
        _new_layers = set(
            pe_config.pe_layer_indices or range(pe_config.num_hidden_layers)
        )
        targeting_changed = _saved_projs != _new_projs or _saved_layers != _new_layers
        if targeting_changed:
            logger.warning(
                "pe_proj_names or pe_layer_indices differ between saved and new config. "
                "Non-matching slots will be randomly initialized."
            )
            attn_changed = True  # force dispatch blocks to run

        if not attn_changed and not mlp_changed and not mlp_layers_changed:
            logger.info("No PE mode changes — returning saved model as-is")
            return _apply_device_map(saved_model, device_map, cls._no_split_modules)

        logger.info(
            "Staged loading: attn %s→%s, mlp %s→%s",
            saved_config.pe_attn_mode,
            pe_config.pe_attn_mode,
            saved_config.pe_mlp_mode,
            pe_config.pe_mlp_mode,
        )

        # Create new model with updated config
        new_model = cls(pe_config)
        new_model = new_model.to(dtype=torch_dtype)

        saved_sd = saved_model.state_dict()
        new_sd = new_model.state_dict()

        # Identify keys belonging to attention projections and MLPs.
        # Skip any slot that was PE in saved OR is PE in new model; non-PE
        # slots (nn.Linear in both) are copied directly by the copy loop.
        _saved_pe_layers = (
            _saved_layers if saved_config.pe_attn_mode in _PE_ATTN_MODES else set()
        )
        _saved_pe_projs = (
            _saved_projs if saved_config.pe_attn_mode in _PE_ATTN_MODES else set()
        )
        _new_pe_layers = (
            _new_layers if pe_config.pe_attn_mode in _PE_ATTN_MODES else set()
        )
        _new_pe_projs = (
            _new_projs if pe_config.pe_attn_mode in _PE_ATTN_MODES else set()
        )

        attn_proj_prefixes = []
        mlp_prefixes = []
        for layer_idx in range(pe_config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.self_attn."
            for proj in _ALL_PROJ_NAMES:
                saved_is_pe = layer_idx in _saved_pe_layers and proj in _saved_pe_projs
                new_is_pe = layer_idx in _new_pe_layers and proj in _new_pe_projs
                if saved_is_pe or new_is_pe:
                    attn_proj_prefixes.append(f"{prefix}{proj}.")
            mlp_prefixes.append(f"model.layers.{layer_idx}.mlp.")

        # Copy compatible weights
        copied, skipped = 0, 0
        for key, param in saved_sd.items():
            is_attn = any(key.startswith(p) for p in attn_proj_prefixes)
            is_mlp = any(key.startswith(p) for p in mlp_prefixes)

            # Skip weights from components whose mode changed
            if is_attn and attn_changed:
                skipped += 1
                continue
            if is_mlp and mlp_changed:
                skipped += 1
                continue

            if key in new_sd and new_sd[key].shape == param.shape:
                new_sd[key].copy_(param)
                copied += 1
            else:
                skipped += 1

        new_model.load_state_dict(new_sd, strict=False)
        logger.info("Copied %d parameters, skipped %d", copied, skipped)

        # Helper to build per-layer tw_kwargs
        def _tw_kwargs(
            layer_idx: int,
            include_generic: bool = False,
            include_reduction: bool = False,
        ) -> dict:
            offset = pe_config.wavelet_basis_offset
            if pe_config.per_layer_offsets and layer_idx < len(
                pe_config.per_layer_offsets
            ):
                offset = pe_config.per_layer_offsets[layer_idx]
            kw = dict(
                trend_dim=pe_config.trend_dim,
                wavelet_dim=pe_config.wavelet_dim,
                wavelet_type=pe_config.wavelet_type,
                basis_offset=offset,
                active_g=getattr(pe_config, "active_g", False),
            )
            if include_generic:
                kw["generic_dim"] = pe_config.generic_dim
            if include_reduction and pe_config.reduction_dim is not None:
                kw["reduction_dim"] = pe_config.reduction_dim
            return kw

        _tw_modes = frozenset(
            (
                "trend_wavelet",
                "trend_wavelet_generic",
                "trend_wavelet_lg",
                "trend_wavelet_generic_lg",
                "trend_wavelet_reduced",
                "trend_wavelet_lg_reduced",
                "trend_wavelet_generic_reduced",
                "trend_wavelet_generic_lg_reduced",
            )
        )
        attn_init_mode = getattr(pe_config, "attn_init_mode", "pretrained")

        # Compute active targeting for dispatch loops (new config's active set)
        _act_layers = _active_layer_indices(pe_config)
        _act_projs = _active_proj_names(pe_config)

        # Handle attention mode change: standard → trend_wavelet
        if attn_changed and pe_config.pe_attn_mode == "trend_wavelet":
            if saved_config.pe_attn_mode == "standard":
                logger.info(
                    "Projecting saved nn.Linear attention weights onto "
                    "TrendWavelet bases (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx)
                    for proj_name in _act_projs:
                        saved_linear = getattr(saved_attn, proj_name)
                        projected = _init_trendwavelet_projection_from_linear(
                            TrendWaveletLinear,
                            saved_linear,
                            attn_init_mode,
                            **kw,
                        )
                        new_proj = getattr(new_attn, proj_name)
                        _copy_tw_projection_state(new_proj, projected)
            elif saved_config.pe_attn_mode == "trend_wavelet":
                logger.info(
                    "Same mode (trend_wavelet), layer targeting changed: "
                    "copying trained layers, init-projecting new layers (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx)
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        if isinstance(saved_proj, TrendWaveletLinear):
                            with torch.no_grad():
                                new_proj.theta.weight.data.copy_(
                                    saved_proj.theta.weight.data
                                )
                                if (
                                    saved_proj.theta.bias is not None
                                    and new_proj.theta.bias is not None
                                ):
                                    new_proj.theta.bias.data.copy_(
                                        saved_proj.theta.bias.data
                                    )
                        else:
                            projected = _init_trendwavelet_projection_from_linear(
                                TrendWaveletLinear,
                                saved_proj,
                                attn_init_mode,
                                **kw,
                            )
                            _copy_tw_projection_state(new_proj, projected)
            else:
                logger.warning(
                    "Attention mode changed from %s to trend_wavelet but "
                    "source is not 'standard' — attention layers randomly "
                    "initialized",
                    saved_config.pe_attn_mode,
                )

        elif attn_changed and pe_config.pe_attn_mode == "trend_wavelet_generic":
            if saved_config.pe_attn_mode == "standard":
                logger.info(
                    "Projecting saved nn.Linear attention weights onto "
                    "TrendWaveletGeneric bases (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx, include_generic=True)
                    for proj_name in _act_projs:
                        saved_linear = getattr(saved_attn, proj_name)
                        projected = _init_trendwavelet_projection_from_linear(
                            TrendWaveletGenericLinear,
                            saved_linear,
                            attn_init_mode,
                            **kw,
                        )
                        new_proj = getattr(new_attn, proj_name)
                        _copy_tw_projection_state(new_proj, projected)
            elif saved_config.pe_attn_mode == "trend_wavelet":
                logger.info(
                    "Upgrading trend_wavelet → trend_wavelet_generic: "
                    "copying frozen theta slice, zeroing generic branch"
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        frozen_dim = saved_proj.theta.out_features
                        with torch.no_grad():
                            new_proj.theta.weight[:frozen_dim].copy_(
                                saved_proj.theta.weight
                            )
                            new_proj.theta.weight[frozen_dim:].zero_()
                            new_proj.generic_basis.weight.data.zero_()
                            if (
                                saved_proj.theta.bias is not None
                                and new_proj.theta.bias is not None
                            ):
                                new_proj.theta.bias.data.copy_(
                                    saved_proj.theta.bias.data
                                )
            elif saved_config.pe_attn_mode == "trend_wavelet_generic":
                logger.info(
                    "Same mode (trend_wavelet_generic), layer targeting changed: "
                    "copying trained layers, init-projecting new layers (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx, include_generic=True)
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        if isinstance(saved_proj, TrendWaveletGenericLinear):
                            with torch.no_grad():
                                new_proj.theta.weight.data.copy_(
                                    saved_proj.theta.weight.data
                                )
                                new_proj.generic_basis.weight.data.copy_(
                                    saved_proj.generic_basis.weight.data
                                )
                                if (
                                    saved_proj.theta.bias is not None
                                    and new_proj.theta.bias is not None
                                ):
                                    new_proj.theta.bias.data.copy_(
                                        saved_proj.theta.bias.data
                                    )
                        else:
                            projected = _init_trendwavelet_projection_from_linear(
                                TrendWaveletGenericLinear,
                                saved_proj,
                                attn_init_mode,
                                **kw,
                            )
                            _copy_tw_projection_state(new_proj, projected)
            else:
                logger.warning(
                    "Attention mode changed from %s to trend_wavelet_generic — "
                    "source is not 'standard' or 'trend_wavelet': attention layers "
                    "randomly initialized",
                    saved_config.pe_attn_mode,
                )

        elif attn_changed and pe_config.pe_attn_mode == "trend_wavelet_lg":
            if saved_config.pe_attn_mode == "standard":
                logger.info(
                    "Projecting saved nn.Linear attention weights onto "
                    "TrendWaveletLG bases (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx)
                    for proj_name in _act_projs:
                        saved_linear = getattr(saved_attn, proj_name)
                        projected = _init_trendwavelet_projection_from_linear(
                            TrendWaveletLinearLG,
                            saved_linear,
                            attn_init_mode,
                            **kw,
                        )
                        new_proj = getattr(new_attn, proj_name)
                        _copy_tw_projection_state(new_proj, projected)
            elif saved_config.pe_attn_mode == "trend_wavelet":
                logger.info(
                    "Upgrading trend_wavelet → trend_wavelet_lg: "
                    "copying theta.weight directly, gate stays ones"
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        with torch.no_grad():
                            new_proj.theta.weight.data.copy_(
                                saved_proj.theta.weight.data
                            )
                            if (
                                saved_proj.theta.bias is not None
                                and new_proj.theta.bias is not None
                            ):
                                new_proj.theta.bias.data.copy_(
                                    saved_proj.theta.bias.data
                                )
            elif saved_config.pe_attn_mode == "trend_wavelet_generic_lg":
                logger.info(
                    "Downgrading trend_wavelet_generic_lg → trend_wavelet_lg: "
                    "copying theta.weight (gate dropped)"
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        frozen_dim = new_proj.theta.out_features
                        with torch.no_grad():
                            new_proj.theta.weight.data.copy_(
                                saved_proj.theta.weight[:frozen_dim]
                            )
                            if (
                                saved_proj.theta.bias is not None
                                and new_proj.theta.bias is not None
                            ):
                                new_proj.theta.bias.data.copy_(
                                    saved_proj.theta.bias.data
                                )
            elif saved_config.pe_attn_mode == "trend_wavelet_lg":
                logger.info(
                    "Same mode (trend_wavelet_lg), layer targeting changed: "
                    "copying trained layers, init-projecting new layers (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx)
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        if isinstance(saved_proj, TrendWaveletLinearLG):
                            with torch.no_grad():
                                new_proj.theta.weight.data.copy_(
                                    saved_proj.theta.weight.data
                                )
                                new_proj.coeff_gate.data.copy_(
                                    saved_proj.coeff_gate.data
                                )
                                if (
                                    saved_proj.theta.bias is not None
                                    and new_proj.theta.bias is not None
                                ):
                                    new_proj.theta.bias.data.copy_(
                                        saved_proj.theta.bias.data
                                    )
                        else:
                            projected = _init_trendwavelet_projection_from_linear(
                                TrendWaveletLinearLG,
                                saved_proj,
                                attn_init_mode,
                                **kw,
                            )
                            _copy_tw_projection_state(new_proj, projected)
            else:
                logger.warning(
                    "Attention mode changed from %s to trend_wavelet_lg — "
                    "unrecognized source: attention layers randomly initialized",
                    saved_config.pe_attn_mode,
                )

        elif attn_changed and pe_config.pe_attn_mode == "trend_wavelet_generic_lg":
            if saved_config.pe_attn_mode == "standard":
                logger.info(
                    "Projecting saved nn.Linear attention weights onto "
                    "TrendWaveletGenericLG bases (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx, include_generic=True)
                    for proj_name in _act_projs:
                        saved_linear = getattr(saved_attn, proj_name)
                        projected = _init_trendwavelet_projection_from_linear(
                            TrendWaveletGenericLinearLG,
                            saved_linear,
                            attn_init_mode,
                            **kw,
                        )
                        new_proj = getattr(new_attn, proj_name)
                        _copy_tw_projection_state(new_proj, projected)
            elif saved_config.pe_attn_mode == "trend_wavelet":
                logger.info(
                    "Upgrading trend_wavelet → trend_wavelet_generic_lg: "
                    "copying frozen theta slice, zeroing generic branch, gate stays ones"
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        frozen_dim = saved_proj.theta.out_features
                        with torch.no_grad():
                            new_proj.theta.weight[:frozen_dim].copy_(
                                saved_proj.theta.weight
                            )
                            new_proj.theta.weight[frozen_dim:].zero_()
                            new_proj.generic_basis.weight.data.zero_()
                            if (
                                saved_proj.theta.bias is not None
                                and new_proj.theta.bias is not None
                            ):
                                new_proj.theta.bias.data.copy_(
                                    saved_proj.theta.bias.data
                                )
            elif saved_config.pe_attn_mode == "trend_wavelet_generic_lg":
                logger.info(
                    "Same mode (trend_wavelet_generic_lg), layer targeting changed: "
                    "copying trained layers, init-projecting new layers (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx, include_generic=True)
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        if isinstance(saved_proj, TrendWaveletGenericLinearLG):
                            with torch.no_grad():
                                new_proj.theta.weight.data.copy_(
                                    saved_proj.theta.weight.data
                                )
                                new_proj.coeff_gate.data.copy_(
                                    saved_proj.coeff_gate.data
                                )
                                new_proj.generic_basis.weight.data.copy_(
                                    saved_proj.generic_basis.weight.data
                                )
                                if (
                                    saved_proj.theta.bias is not None
                                    and new_proj.theta.bias is not None
                                ):
                                    new_proj.theta.bias.data.copy_(
                                        saved_proj.theta.bias.data
                                    )
                        else:
                            projected = _init_trendwavelet_projection_from_linear(
                                TrendWaveletGenericLinearLG,
                                saved_proj,
                                attn_init_mode,
                                **kw,
                            )
                            _copy_tw_projection_state(new_proj, projected)
            elif saved_config.pe_attn_mode in _tw_modes:
                logger.info(
                    "Upgrading %s → trend_wavelet_generic_lg: copying theta where "
                    "dims match, zeroing remainder",
                    saved_config.pe_attn_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        with torch.no_grad():
                            copy_dim = min(
                                saved_proj.theta.weight.shape[0],
                                new_proj.theta.weight.shape[0],
                            )
                            new_proj.theta.weight[:copy_dim].copy_(
                                saved_proj.theta.weight[:copy_dim]
                            )
                            if copy_dim < new_proj.theta.weight.shape[0]:
                                new_proj.theta.weight[copy_dim:].zero_()
                            if hasattr(saved_proj, "generic_basis"):
                                new_proj.generic_basis.weight.data.copy_(
                                    saved_proj.generic_basis.weight.data
                                )
                            else:
                                new_proj.generic_basis.weight.data.zero_()
                            if (
                                saved_proj.theta.bias is not None
                                and new_proj.theta.bias is not None
                            ):
                                new_proj.theta.bias.data.copy_(
                                    saved_proj.theta.bias.data
                                )
            else:
                logger.warning(
                    "Attention mode changed from %s to trend_wavelet_generic_lg — "
                    "unrecognized source: attention layers randomly initialized",
                    saved_config.pe_attn_mode,
                )

        elif attn_changed and pe_config.pe_attn_mode in (
            "trend_wavelet_reduced",
            "trend_wavelet_lg_reduced",
        ):
            _reduced_cls = _TW_ATTN_CLS_MAP[pe_config.pe_attn_mode]
            _reduced_base_cls = (
                TrendWaveletLinearLGReduced
                if pe_config.pe_attn_mode == "trend_wavelet_lg_reduced"
                else TrendWaveletLinearReduced
            )
            if saved_config.pe_attn_mode == "standard":
                logger.info(
                    "Projecting saved nn.Linear attention weights onto "
                    "TrendWaveletReduced bases (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx, include_reduction=True)
                    for proj_name in _act_projs:
                        saved_linear = getattr(saved_attn, proj_name)
                        projected = _init_trendwavelet_projection_from_linear(
                            _reduced_cls,
                            saved_linear,
                            attn_init_mode,
                            **kw,
                        )
                        new_proj = getattr(new_attn, proj_name)
                        _copy_tw_projection_state(new_proj, projected)
            elif saved_config.pe_attn_mode == pe_config.pe_attn_mode:
                logger.info(
                    "Same mode (%s), layer targeting changed: "
                    "copying trained layers, init-projecting new layers (init=%s)",
                    pe_config.pe_attn_mode,
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(layer_idx, include_reduction=True)
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        if isinstance(saved_proj, _reduced_base_cls):
                            _copy_tw_projection_state(new_proj, saved_proj)
                        else:
                            projected = _init_trendwavelet_projection_from_linear(
                                _reduced_cls,
                                saved_proj,
                                attn_init_mode,
                                **kw,
                            )
                            _copy_tw_projection_state(new_proj, projected)
            else:
                logger.warning(
                    "Attention mode changed from %s to %s — "
                    "unrecognized source: attention layers randomly initialized",
                    saved_config.pe_attn_mode,
                    pe_config.pe_attn_mode,
                )

        elif attn_changed and pe_config.pe_attn_mode in (
            "trend_wavelet_generic_reduced",
            "trend_wavelet_generic_lg_reduced",
        ):
            _reduced_cls = _TW_ATTN_CLS_MAP[pe_config.pe_attn_mode]
            _reduced_base_cls = (
                TrendWaveletGenericLinearLGReduced
                if pe_config.pe_attn_mode == "trend_wavelet_generic_lg_reduced"
                else TrendWaveletGenericLinearReduced
            )
            if saved_config.pe_attn_mode == "standard":
                logger.info(
                    "Projecting saved nn.Linear attention weights onto "
                    "TrendWaveletGenericReduced bases (init=%s)",
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(
                        layer_idx, include_generic=True, include_reduction=True
                    )
                    for proj_name in _act_projs:
                        saved_linear = getattr(saved_attn, proj_name)
                        projected = _init_trendwavelet_projection_from_linear(
                            _reduced_cls,
                            saved_linear,
                            attn_init_mode,
                            **kw,
                        )
                        new_proj = getattr(new_attn, proj_name)
                        _copy_tw_projection_state(new_proj, projected)
            elif saved_config.pe_attn_mode == pe_config.pe_attn_mode:
                logger.info(
                    "Same mode (%s), layer targeting changed: "
                    "copying trained layers, init-projecting new layers (init=%s)",
                    pe_config.pe_attn_mode,
                    attn_init_mode,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    kw = _tw_kwargs(
                        layer_idx, include_generic=True, include_reduction=True
                    )
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        if isinstance(saved_proj, _reduced_base_cls):
                            _copy_tw_projection_state(new_proj, saved_proj)
                        else:
                            projected = _init_trendwavelet_projection_from_linear(
                                _reduced_cls,
                                saved_proj,
                                attn_init_mode,
                                **kw,
                            )
                            _copy_tw_projection_state(new_proj, projected)
            else:
                logger.warning(
                    "Attention mode changed from %s to %s — "
                    "unrecognized source: attention layers randomly initialized",
                    saved_config.pe_attn_mode,
                    pe_config.pe_attn_mode,
                )

        elif attn_changed and pe_config.pe_attn_mode in ("svd", "svd_lg"):
            svd_cls = SVDLinear if pe_config.pe_attn_mode == "svd" else SVDLinearLG
            if saved_config.pe_attn_mode == "standard":
                logger.info(
                    "Projecting saved nn.Linear attention weights onto SVD bases "
                    "(rank=%d, gate stays ones for svd_lg)",
                    pe_config.svd_rank,
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    for proj_name in _act_projs:
                        saved_linear = getattr(saved_attn, proj_name)
                        projected = svd_cls.from_pretrained_linear(
                            saved_linear,
                            pe_config.svd_rank,
                        )
                        new_proj = getattr(new_attn, proj_name)
                        new_proj.svd_basis.copy_(projected.svd_basis)
                        new_proj.theta.weight.data.copy_(projected.theta.weight.data)
                        if (
                            projected.theta.bias is not None
                            and new_proj.theta.bias is not None
                        ):
                            new_proj.theta.bias.data.copy_(projected.theta.bias.data)
            elif (
                saved_config.pe_attn_mode == "svd"
                and pe_config.pe_attn_mode == "svd_lg"
            ):
                logger.info(
                    "Upgrading svd → svd_lg: copying svd_basis + theta.weight, "
                    "gate stays ones"
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        with torch.no_grad():
                            new_proj.svd_basis.copy_(saved_proj.svd_basis)
                            new_proj.theta.weight.data.copy_(
                                saved_proj.theta.weight.data
                            )
                            if (
                                saved_proj.theta.bias is not None
                                and new_proj.theta.bias is not None
                            ):
                                new_proj.theta.bias.data.copy_(
                                    saved_proj.theta.bias.data
                                )
            elif (
                saved_config.pe_attn_mode == "svd_lg"
                and pe_config.pe_attn_mode == "svd"
            ):
                logger.info(
                    "Downgrading svd_lg → svd: copying svd_basis + theta.weight, "
                    "dropping gate"
                )
                for layer_idx in _act_layers:
                    saved_attn = saved_model.model.layers[layer_idx].self_attn
                    new_attn = new_model.model.layers[layer_idx].self_attn
                    for proj_name in _act_projs:
                        saved_proj = getattr(saved_attn, proj_name)
                        new_proj = getattr(new_attn, proj_name)
                        with torch.no_grad():
                            new_proj.svd_basis.copy_(saved_proj.svd_basis)
                            new_proj.theta.weight.data.copy_(
                                saved_proj.theta.weight.data
                            )
                            if (
                                saved_proj.theta.bias is not None
                                and new_proj.theta.bias is not None
                            ):
                                new_proj.theta.bias.data.copy_(
                                    saved_proj.theta.bias.data
                                )
            else:
                logger.warning(
                    "Attention mode changed from %s to %s — unrecognized source: "
                    "attention layers randomly initialized",
                    saved_config.pe_attn_mode,
                    pe_config.pe_attn_mode,
                )

        if attn_changed and pe_config.pe_attn_mode == "standard":
            logger.warning(
                "Attention mode changed from %s to standard — attention "
                "layers randomly initialized (no reverse projection)",
                saved_config.pe_attn_mode,
            )

        _MLP_CLS_MAP_STAGED = {
            "ae": PEBottleneckMLP,
            "ae_lg": PEBottleneckMLPLG,
            "vae": PEBottleneckMLPVAE,
            "vae_lg": PEBottleneckMLPVAELG,
        }
        if mlp_changed and pe_config.pe_mlp_mode in _MLP_CLS_MAP_STAGED:
            if saved_config.pe_mlp_mode == "standard" and pe_config.ae_init_mode in (
                "pretrained",
                "svd",
                "cur",
                "fourier",
            ):
                # Source has gate_proj/down_proj — structured initialization is possible
                init_method = pe_config.ae_init_mode
                logger.info(
                    "Initializing AE/VAE MLP layers from saved LlamaMLP weights "
                    "(ae_init_mode='%s' — fine-tuning required).",
                    init_method,
                )
                mlp_cls = _MLP_CLS_MAP_STAGED[pe_config.pe_mlp_mode]
                _active_mlp = (
                    set(pe_config.pe_mlp_layer_indices)
                    if pe_config.pe_mlp_layer_indices is not None
                    else set(range(pe_config.num_hidden_layers))
                )
                for layer_idx in _active_mlp:
                    src_mlp = saved_model.model.layers[layer_idx].mlp
                    if init_method == "svd":
                        initialized = mlp_cls.from_pretrained_mlp_svd(
                            src_mlp, pe_config
                        )
                    elif init_method == "cur":
                        initialized = mlp_cls.from_pretrained_mlp_cur(
                            src_mlp, pe_config
                        )
                    elif init_method == "fourier":
                        initialized = mlp_cls.from_pretrained_mlp_fourier(
                            src_mlp, pe_config
                        )
                    else:
                        initialized = mlp_cls.from_pretrained_mlp(src_mlp, pe_config)
                    new_model.model.layers[layer_idx].mlp.load_state_dict(
                        initialized.state_dict(), strict=False
                    )
            elif saved_config.pe_mlp_mode == "standard":
                logger.info(
                    "AE/VAE MLP layers randomly initialized (ae_init_mode='random')."
                )
            else:
                # Source is ae/ae_lg/vae/vae_lg — no gate_proj/down_proj to factorize
                logger.warning(
                    "MLP layers initialized randomly (source mode %s → %s, "
                    "no structural mapping). Fine-tuning required.",
                    saved_config.pe_mlp_mode,
                    pe_config.pe_mlp_mode,
                )

        if mlp_changed and pe_config.pe_mlp_mode == "standard":
            logger.warning(
                "MLP mode changed to standard but source was %s — MLP "
                "layers randomly initialized (no reverse mapping)",
                saved_config.pe_mlp_mode,
            )

        # Same pe_mlp_mode but layer set expanded: preserve trained overlapping
        # layers (handled by the copy loop above) and initialize newly-added
        # layers from the saved model's standard LlamaMLP weights.
        if (
            not mlp_changed
            and mlp_layers_changed
            and pe_config.pe_mlp_mode in _MLP_CLS_MAP_STAGED
        ):
            newly_added = _new_mlp_layers - _saved_mlp_layers
            if newly_added:
                init_method = pe_config.ae_init_mode
                mlp_cls = _MLP_CLS_MAP_STAGED[pe_config.pe_mlp_mode]
                logger.info(
                    "MLP layer set expanded (same mode=%s): initializing %d new "
                    "layer(s) %s from saved weights (ae_init_mode=%s). "
                    "Overlapping layers %s preserve trained weights.",
                    pe_config.pe_mlp_mode,
                    len(newly_added),
                    sorted(newly_added),
                    init_method,
                    sorted(_new_mlp_layers & _saved_mlp_layers),
                )
                for layer_idx in sorted(newly_added):
                    src_mlp = saved_model.model.layers[layer_idx].mlp
                    if hasattr(src_mlp, "gate_proj") and init_method in (
                        "pretrained",
                        "svd",
                        "cur",
                        "fourier",
                    ):
                        if init_method == "svd":
                            initialized = mlp_cls.from_pretrained_mlp_svd(
                                src_mlp, pe_config
                            )
                        elif init_method == "cur":
                            initialized = mlp_cls.from_pretrained_mlp_cur(
                                src_mlp, pe_config
                            )
                        elif init_method == "fourier":
                            initialized = mlp_cls.from_pretrained_mlp_fourier(
                                src_mlp, pe_config
                            )
                        else:
                            initialized = mlp_cls.from_pretrained_mlp(
                                src_mlp, pe_config
                            )
                        new_model.model.layers[layer_idx].mlp.load_state_dict(
                            initialized.state_dict(), strict=False
                        )
                    else:
                        logger.info(
                            "New AE MLP at layer %d randomly initialized "
                            "(ae_init_mode='%s' or source is not standard LlamaMLP).",
                            layer_idx,
                            init_method,
                        )

        return _apply_device_map(new_model, device_map, cls._no_split_modules)
