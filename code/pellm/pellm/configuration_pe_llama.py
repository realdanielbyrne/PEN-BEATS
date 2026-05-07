from __future__ import annotations

from transformers import LlamaConfig


class PELlamaConfig(LlamaConfig):
    """
    Configuration class for PELlama models.

    Extends :class:`~transformers.LlamaConfig` with parameter-efficient layer
    options: TrendWavelet attention projections and AE-bottleneck MLPs.

    Attention projection modes (``pe_attn_mode``):
        - ``"standard"``: Use ``nn.Linear``.
        - ``"trend_wavelet"``: Use ``TrendWaveletLinear`` with frozen
          Vandermonde + DWT bases.
        - ``"trend_wavelet_generic"``: Use ``TrendWaveletGenericLinear`` —
          adds a third learned generic branch on top of frozen bases.
        - ``"trend_wavelet_lg"``: Use ``TrendWaveletLinearLG`` — learned
          per-dimension sigmoid gate on the two-way coefficient vector.
        - ``"trend_wavelet_generic_lg"``: Use ``TrendWaveletGenericLinearLG``
          — learned gate on the three-way coefficient vector.
        - ``"svd"``: Use ``SVDLinear`` — frozen basis from top-k left singular
          vectors of each pretrained weight matrix (Eckart-Young optimal rank-k).
        - ``"svd_lg"``: Use ``SVDLinearLG`` — SVD basis with learned gate.
        - ``"trend_wavelet_reduced"``: Use ``TrendWaveletLinearReduced`` —
          learned reduction stage before TrendWavelet basis expansion.
        - ``"trend_wavelet_lg_reduced"``: Use ``TrendWaveletLinearLGReduced``
          — reduction + learned gate.
        - ``"trend_wavelet_generic_reduced"``: Use
          ``TrendWaveletGenericLinearReduced`` — reduction + generic branch.
        - ``"trend_wavelet_generic_lg_reduced"``: Use
          ``TrendWaveletGenericLinearLGReduced`` — reduction + generic + gate.
        - ``"trend_wavelet_no_postln"``: Use ``TrendWaveletLinear`` for q/k/v/o
          (same as ``"trend_wavelet"``) and additionally **bypass**
          ``post_attention_layernorm`` on every PE attention layer — the
          post-attention residual stream feeds directly into the MLP. The
          RMSNorm module is left constructed (state-dict shape unchanged) but
          is not invoked in ``forward``. Scoped to ``pe_layer_indices``;
          non-PE layers retain stock Llama behavior.

    Generic branch (``generic_dim``):
        Number of learned generic basis dimensions added by
        ``"trend_wavelet_generic"`` and ``"trend_wavelet_generic_lg"`` modes.
        Ignored for other modes.

    MLP modes (``pe_mlp_mode``):
        - ``"standard"``: Use the default ``LlamaMLP`` (SwiGLU).
        - ``"ae"``: Use ``PEBottleneckMLP`` (AE-bottleneck).
        - ``"ae_lg"``: Use ``PEBottleneckMLPLG`` (AE-bottleneck with learned gate).
        - ``"vae"``: Use ``PEBottleneckMLPVAE`` (β-VAE bottleneck with learnable β).
        - ``"vae_lg"``: Use ``PEBottleneckMLPVAELG`` (β-VAE bottleneck + learned gate).
        - ``"ae_basis_latent"``: Use ``AETrendWaveletLatentMLP`` — AE encoder
          narrow waist *is* the trend+wavelet coefficient vector; frozen basis
          decodes ``coeff_dim -> hidden`` directly. N-BEATS-Lightning AERootBlock
          pattern (from-scratch only — no pretrained-init helper).
        - ``"ae_basis_reexpand"``: Use ``AETrendWaveletFilterMLP`` — full AE
          reconstructs at hidden, frozen basis filters the reconstruction
          (``x_recon @ Bᵀ @ B``). Hedged variant; from-scratch only.
        - ``"tw_root"``: Use ``TrendWaveletRootMLP`` — N-BEATS RootBlock-style
          MLP. Single ``TrendWaveletLinear(hidden, hidden)``; theta projects
          to coefficients and a frozen basis decodes to hidden. No narrow
          waist, no nonlinearity inside the MLP. From-scratch only.
        - ``"tw_root_fc"``: Use ``TrendWaveletRootFCMLP`` — canonical RootBlock.
          Full-width pre-theta FC + SiLU, then ``TrendWaveletLinear(hidden,
          hidden)``. Direct A/B against ``ae_basis_latent`` to isolate the
          narrow-waist AE vs flat-width backbone question. From-scratch only.
        - ``"tw_root_post_fc"``: Use ``TrendWaveletRootPostFCMLP`` —
          ``TrendWaveletLinear`` followed by SiLU and a learned full-width
          ``nn.Linear(hidden, hidden)``. Adds a post-basis mixer to the bare
          RootBlock. From-scratch only.
        - ``"tw_root_fc_post_fc"``: Use ``TrendWaveletRootFCPostFCMLP`` —
          pre-theta FC + SiLU, ``TrendWaveletLinear`` + SiLU, then a learned
          ``nn.Linear(hidden, hidden)``. Sandwiches the frozen basis between
          two learned full-width linears. From-scratch only.

    Example::

        from pellm import PELlamaConfig

        config = PELlamaConfig.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            pe_attn_mode="trend_wavelet",
            pe_mlp_mode="ae_lg",
            ae_latent_dim=256,
        )
    """

    model_type = "pe_llama"

    def __init__(
        self,
        # Attention projection mode
        pe_attn_mode: str = "standard",
        trend_dim: int = 3,
        wavelet_dim: int = 28,
        wavelet_type: str = "db3",
        mlp_wavelet_type: str | None = None,
        wavelet_basis_offset: int = 0,
        per_layer_offsets: list[int] | None = None,
        # Generic branch
        generic_dim: int = 5,
        # Targeting (None = all)
        pe_proj_names: list[str] | None = None,
        pe_layer_indices: list[int] | None = None,
        # SVD attention mode
        svd_rank: int = 32,
        attn_init_mode: str = "pretrained",
        # Reduced TrendWavelet
        reduction_dim: int | None = None,
        # TrendWavelet output activation
        active_g: bool = False,
        # MLP mode
        pe_mlp_mode: str = "standard",
        ae_latent_dim: int = 256,
        pe_mlp_layer_indices: list[int] | None = None,
        ae_init_mode: str = "pretrained",
        ae_inner_init: str = "svd",
        mlp_active_g: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Attention
        self.pe_attn_mode = pe_attn_mode
        self.trend_dim = trend_dim
        self.wavelet_dim = wavelet_dim
        self.wavelet_type = wavelet_type
        self.mlp_wavelet_type = (
            mlp_wavelet_type if mlp_wavelet_type is not None else wavelet_type
        )
        self.wavelet_basis_offset = wavelet_basis_offset
        self.per_layer_offsets = per_layer_offsets
        self.generic_dim = generic_dim
        # SVD
        self.svd_rank = svd_rank
        self.attn_init_mode = attn_init_mode  # "pretrained", "lstsq", "svd", "cur", "fourier", or "random"
        # Reduced TrendWavelet
        self.reduction_dim = reduction_dim
        self.active_g = active_g
        # Targeting
        self.pe_proj_names = pe_proj_names  # None = all 4 projections
        self.pe_layer_indices = pe_layer_indices  # None = all decoder layers
        # MLP
        self.pe_mlp_mode = pe_mlp_mode
        self.ae_latent_dim = ae_latent_dim
        self.pe_mlp_layer_indices = pe_mlp_layer_indices  # None = all decoder layers
        self.ae_init_mode = (
            ae_init_mode  # "pretrained", "svd", "cur", "fourier", or "random"
        )
        self.ae_inner_init = ae_inner_init  # "svd" or "match"
        # Independent of attention's ``active_g``: when True, the trendwavelet
        # MLP modes (ae_basis_latent, ae_basis_reexpand, tw_root, tw_root_fc,
        # tw_root_post_fc, tw_root_fc_post_fc) apply ``F.silu`` on the MLP
        # block's final output.
        self.mlp_active_g = mlp_active_g
