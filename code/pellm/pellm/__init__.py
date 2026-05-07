"""
pellm — Parameter Efficient LLM

Importing this package registers ``"pe_llama"`` with HuggingFace's
``AutoConfig`` and ``AutoModelForCausalLM``, so that after a ``save_pretrained``
/ ``push_to_hub``, the model can be reloaded via Auto classes without passing
the model class explicitly.

    import pellm  # triggers registration
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("./pe-llama-local")
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_pe_llama import PELlamaConfig
from .modeling_pe_llama import (
    PELlamaAttention,
    PELlamaDecoderLayer,
    PELlamaForCausalLM,
    PELlamaModel,
    PELlamaPreTrainedModel,
)
from .pe_layers import (PEBottleneckMLP, PEBottleneckMLPLG, TrendWaveletLinear,
                        TrendWaveletGenericLinear, TrendWaveletLinearLG,
                        TrendWaveletGenericLinearLG, SVDLinear, SVDLinearLG)

# Register with HuggingFace Auto classes.
AutoConfig.register("pe_llama", PELlamaConfig)
AutoModelForCausalLM.register(PELlamaConfig, PELlamaForCausalLM)

__all__ = [
    "PELlamaConfig",
    "PELlamaAttention",
    "PELlamaDecoderLayer",
    "PELlamaPreTrainedModel",
    "PELlamaModel",
    "PELlamaForCausalLM",
    "TrendWaveletLinear",
    "TrendWaveletGenericLinear",
    "TrendWaveletLinearLG",
    "TrendWaveletGenericLinearLG",
    "SVDLinear",
    "SVDLinearLG",
    "PEBottleneckMLP",
    "PEBottleneckMLPLG",
]
