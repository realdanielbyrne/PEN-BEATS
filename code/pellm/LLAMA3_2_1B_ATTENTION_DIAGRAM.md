# Llama 3.2-1B Attention Projection Diagram

This diagram shows where the projection layers sit inside a single Llama 3.2-1B decoder self-attention block.

Note: the request listed `o_proj, k_proj, v_proj, and o_proj`. I assumed the first `o_proj` was intended to be `q_proj`, since the attention layer uses `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

## Model dimensions used

- Hidden size: `2048`
- Query heads: `32`
- Key/value heads: `8`
- Head dimension: `64`
- Grouped-query attention ratio: `32 / 8 = 4` query heads per KV head

## Mermaid diagram

```mermaid
flowchart TD
    X["Residual stream input<br/>[batch, seq, 2048]"]
    N["RMSNorm before self-attention<br/>[batch, seq, 2048]"]

    Q["q_proj<br/>Linear: 2048 -> 2048<br/>Q before reshape: [batch, seq, 2048]<br/>Q after reshape: [batch, seq, 32, 64]"]
    K["k_proj<br/>Linear: 2048 -> 512<br/>K before reshape: [batch, seq, 512]<br/>K after reshape: [batch, seq, 8, 64]"]
    V["v_proj<br/>Linear: 2048 -> 512<br/>V before reshape: [batch, seq, 512]<br/>V after reshape: [batch, seq, 8, 64]"]

    QR["Apply RoPE to Q<br/>[batch, seq, 32, 64]"]
    KR["Apply RoPE to K<br/>[batch, seq, 8, 64]"]

    A["Grouped-Query Self-Attention<br/>32 Q heads attend using 8 KV heads<br/>4 Q heads share each KV head group<br/>Attention output: [batch, seq, 32, 64]"]

    M["Merge heads<br/>[batch, seq, 2048]"]
    O["o_proj<br/>Linear: 2048 -> 2048<br/>Output: [batch, seq, 2048]"]
    R["Residual add back to decoder stream<br/>[batch, seq, 2048]"]

    X --> N
    N --> Q
    N --> K
    N --> V
    Q --> QR
    K --> KR
    QR --> A
    KR --> A
    V --> A
    A --> M
    M --> O
    O --> R
    X --> R
```

## Mermaid diagram with `TrendWaveletLinear` inserted

This is the same decoder self-attention block, but with each attention projection
(`q_proj`, `k_proj`, `v_proj`, `o_proj`) replaced by `TrendWaveletLinear`.

Using the defaults in this repo:

- `trend_dim = 3`
- `wavelet_dim = 28`
- Total coefficient width inside each `TrendWaveletLinear` block: `31`

```mermaid
flowchart TD
    X2["Residual stream input<br/>[batch, seq, 2048]"]
    N2["RMSNorm before self-attention<br/>[batch, seq, 2048]"]

    subgraph QTWL["q_proj as TrendWaveletLinear"]
        QIN["Input to q block<br/>[batch, seq, 2048]"]
        QTH["theta<br/>Linear: 2048 -> 31<br/>Coeff tensor: [batch, seq, 31]"]
        QT["Trend coeffs<br/>[batch, seq, 3]"]
        QW["Wavelet coeffs<br/>[batch, seq, 28]"]
        QB["Frozen bases<br/>Trend basis: [3, 2048]<br/>Wavelet basis: [28, 2048]"]
        QOUT["q block output<br/>[batch, seq, 2048]<br/>reshape -> [batch, seq, 32, 64]"]
        QIN --> QTH
        QTH --> QT
        QTH --> QW
        QT --> QB
        QW --> QB
        QB --> QOUT
    end

    subgraph KTWL["k_proj as TrendWaveletLinear"]
        KIN["Input to k block<br/>[batch, seq, 2048]"]
        KTH["theta<br/>Linear: 2048 -> 31<br/>Coeff tensor: [batch, seq, 31]"]
        KT["Trend coeffs<br/>[batch, seq, 3]"]
        KW["Wavelet coeffs<br/>[batch, seq, 28]"]
        KB["Frozen bases<br/>Trend basis: [3, 512]<br/>Wavelet basis: [28, 512]"]
        KOUT["k block output<br/>[batch, seq, 512]<br/>reshape -> [batch, seq, 8, 64]"]
        KIN --> KTH
        KTH --> KT
        KTH --> KW
        KT --> KB
        KW --> KB
        KB --> KOUT
    end

    subgraph VTWL["v_proj as TrendWaveletLinear"]
        VIN["Input to v block<br/>[batch, seq, 2048]"]
        VTH["theta<br/>Linear: 2048 -> 31<br/>Coeff tensor: [batch, seq, 31]"]
        VT["Trend coeffs<br/>[batch, seq, 3]"]
        VW["Wavelet coeffs<br/>[batch, seq, 28]"]
        VB["Frozen bases<br/>Trend basis: [3, 512]<br/>Wavelet basis: [28, 512]"]
        VOUT["v block output<br/>[batch, seq, 512]<br/>reshape -> [batch, seq, 8, 64]"]
        VIN --> VTH
        VTH --> VT
        VTH --> VW
        VT --> VB
        VW --> VB
        VB --> VOUT
    end

    QR2["Apply RoPE to Q<br/>[batch, seq, 32, 64]"]
    KR2["Apply RoPE to K<br/>[batch, seq, 8, 64]"]

    A2["Grouped-Query Self-Attention<br/>32 Q heads attend using 8 KV heads<br/>4 Q heads share each KV head group<br/>Attention output: [batch, seq, 32, 64]"]

    M2["Merge heads<br/>[batch, seq, 2048]"]

    subgraph OTWL["o_proj as TrendWaveletLinear"]
        OIN["Input to o block<br/>[batch, seq, 2048]"]
        OTH["theta<br/>Linear: 2048 -> 31<br/>Coeff tensor: [batch, seq, 31]"]
        OT["Trend coeffs<br/>[batch, seq, 3]"]
        OW["Wavelet coeffs<br/>[batch, seq, 28]"]
        OB["Frozen bases<br/>Trend basis: [3, 2048]<br/>Wavelet basis: [28, 2048]"]
        OOUT["o block output<br/>[batch, seq, 2048]"]
        OIN --> OTH
        OTH --> OT
        OTH --> OW
        OT --> OB
        OW --> OB
        OB --> OOUT
    end

    R2["Residual add back to decoder stream<br/>[batch, seq, 2048]"]

    X2 --> N2
    N2 --> QIN
    N2 --> KIN
    N2 --> VIN
    QOUT --> QR2
    KOUT --> KR2
    QR2 --> A2
    KR2 --> A2
    VOUT --> A2
    A2 --> M2
    M2 --> OIN
    OOUT --> R2
    X2 --> R2
```

## Projection summary

| Layer | Weight shape | Input width | Output width | Interpreted tensor shape |
|---|---:|---:|---:|---|
| `q_proj` | `2048 x 2048` | `2048` | `2048` | `32` heads x `64` |
| `k_proj` | `512 x 2048` | `2048` | `512` | `8` heads x `64` |
| `v_proj` | `512 x 2048` | `2048` | `512` | `8` heads x `64` |
| `o_proj` | `2048 x 2048` | `2048` | `2048` | merged `32 x 64` back to model width |

## `TrendWaveletLinear` summary

With the repo defaults (`trend_dim = 3`, `wavelet_dim = 28`, `wavelet_type = db3`),
each replaced attention projection uses:

| Layer | `theta` shape | Coeff width | Frozen basis shapes | Block output width |
|---|---:|---:|---|---:|
| `q_proj` | `31 x 2048` | `31 = 3 + 28` | trend `[3, 2048]`, wavelet `[28, 2048]` | `2048` |
| `k_proj` | `31 x 2048` | `31 = 3 + 28` | trend `[3, 512]`, wavelet `[28, 512]` | `512` |
| `v_proj` | `31 x 2048` | `31 = 3 + 28` | trend `[3, 512]`, wavelet `[28, 512]` | `512` |
| `o_proj` | `31 x 2048` | `31 = 3 + 28` | trend `[3, 2048]`, wavelet `[28, 2048]` | `2048` |

## Sources

- Hugging Face Llama config docs: https://huggingface.co/docs/transformers/v4.51.3/model_doc/llama
- Llama 3.2-1B config mirror showing `hidden_size=2048`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=64`: https://huggingface.co/onnx-community/Llama-3.2-1B/blob/main/config.json
