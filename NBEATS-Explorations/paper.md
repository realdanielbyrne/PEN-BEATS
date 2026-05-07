# Wavelets and Autoencoders are All You Need

## Abstract

N-BEATS, Oreshkin et al. (2019), broke new ground in univariate time series forecasting by using fixed basis functions neural basis expansion in which the network learns coefficients for a frozen basis function then expanded those coefficients into the desired forecast and backcast lengths through a linear transformation. Their innovation came at an extreme cost though as a typical N-BEATS model can range from 25M-44M parameters. In this paper I extend the idea of solving for coefficients of a frozen basis functions but instead of relying on a global Fourier function, wavelets provide localization in that they can identify and locate components of the global signal which is a key requirement in timeseries analysis algorithms. Wavelets have been used extensively in signal analysis for feature extraction, denoising, and multi-resolution analysis.  Likewise, Autoencoders are useful constructs for identifying key characteristics of a signal while filtering out nonessential transient signals, and are useful in domains such as denoising signals and stock trading.  In this paper I introduce a variety of novel block types that use using a combination of discrete wavelet transforms, DWT, and AutoEncoders. My novel architecture improvements reduce parameter counts by 60–95% over N-BEATS while surpassing or equalling prior published results without ensembiling. In addition I introduce a family of combined TrendWavelet block types which combine the low pass Trend and a high pass Wavelet basis that can be used to produce <1M parameter models that score within 0.5% SMAPE of 19M-43M parameter baseline N-BEATS models representing a 12-80x compression with little to no loss of accuracy. Furthermore, I show that these new block constructs are transferrable to other architectures like NHiTs and Transformers with equally impressive reductions in parameter counts and compettitive accuracies.

## Introduction and Prior Work

### Introduction

Time series forecasting and signal analysis are two names for the same thing even though practitioners in each discipline use their own terminologies. Moving Average vs Finite Impulse Response (FIR) filter, Autoregressive Filter vs Infinite Impulse Response (IIR) filter. Seasonal decomposition -> band pass filter of seasonal frequencies. A time series dataset (sales figures, temperature, stock prices) is just a discrete set of sampled data.  Likewise all Deep Learning problems can be reconsidered as a signal analysis problem.  For example image recognition in deep learning is often performed by sampling an image and passing the samples through a convolutional filter, similar in principle to what you might do in photoshop when applying a sharpen filter.  Deep learning then can be seen as a type of probablistic signal analysis.  As such why then don't we use more techniques develop in signal analysis in machine learning.

Signal analysis tools like the Discrete Fourier transform and the Discrete Wavelet Transform were developed as techniques used in signal processing, migrated into forecasting, and they are trickling into machine learning. N-BEATS was a pioneer of sort in bringing some traditional signal analysis tools to AI with its use basis expansion blocks. N-BEATS Seasonality and Trend blocks are literally signal decomposition operators applied inside of a neural network.  The novelty was using the neural network to learn which decomposition coefficients best reconstruct the observed signal.

### Wavelets

Fourier analysis decomposes a signal into a sum of collection of sine and cosine waves of fixed frequency.  It is a useful tool for analyzing stationary signals whose frequency content does not change over time. Real world signals routinely break that mold however. For example a stock price may trend smoothly following the normal sales cycle and then spike abruptly because of a news article, a comment made by a prominet person in the news, or when a large hedge fund liquidates their considerable position.  Fourier bases represent these transient features poorly, and thus they tend to smear localized events across the other frequency components of the signal.

Wavelets address this problem by providing basis functions that are localized in both time and frequency. Where a sine wave oscillates forever, a wavelet is a brief oscillatory burst with exponentially decaying tails and a zero mean. They osciallate in a particular pattern defined by their mother wavelet and then decay to zero.

The mother wavelet $\psi(t)$ is scaled and translated to produce a family of basis functions:

$$\psi_{j,k}(t) = 2^{j/2}\,\psi(2^j t - k)$$

where $j$ controls scale (frequency resolution) and $k$ controls translation (time position). Short, high-frequency wavelets have narrow time resolution; long, low-frequency wavelets have narrow frequency resolution. This makes wavelets substantially more effective for real-world time series analysis, where variance and frequency content change over time.

A discrete wavelet transform (DWT) brings the continuous wavelet into the digital domain. DWTs computes the inner product of a signal with a basis family producing a set of orthonormal coefficients that can then be used to reconstruct the original signal.

Wavelets have been used sparsely in machine learning. WaveNet (2022) arXiv:2505.11781 learns a filter-bank specialized to fit non-stationary signals by learning the basis function by decompsing its nth order derivitives and then leaarning an affine transformation that lets the network supress and cross-mix frequencies in training.

MultiWave (arXiv:2306.10164) is a model-agnostic preprocessor around any neural network architecture.  It takes the concept of machine learning/ signal processing equivalency to heart by preprocessing input data with a DWT, grouping the outputs by frequecy, and then feeding the groups into any neural network (Transformer, FFN, CNN, etc), feeding different groups into different networks. Then they take the networks' outputs, combine them, and then do an inverse DWT to get the prediction.

W-Transformers, 2209.03945v1 ises a similar approach by preprocessing timeseries data with a Discrete Overlapping Wavelet Transform and then feeding the results into a collection of transformer models to analyze the decomposed signals independantly.

### Autoencoders

An autoencoder is a neural network trained to reproduce its input. It has an encoder that maps input x into a smaller dimensional, latent, code z, and a decoder that maps z back into a reconstructed version of the original input, x_hat.

`
x -> encoder -> z -> decoder -> x_hat
loss = reconstruction_error(x, x_hat) + optional_regularization
`

The premise is that the model is constrained by the reduced dimensionality of the latent space. Therefore, since the decoder network doesn't have access to the original signal a perfect copy is difficult.  Thus the network is pushed to learn how to reconstruct the original signal from the information it can extract from the latest representation. The classic paper G. E. Hinton, Salakhutdinov "Reducing the Dimensionality of Data with Neural Networks.Science" (2006) showed  that autoencoders generalize PCA too non-linear domains.  Autoencoders are also used widely for self-supervised pretraining, generative modeling, and image denoising and generation.  

## Model Description

N-BEATS \citep{oreshkin2020nbeats} analyzes a time series through a branching stack of parallel blocks, each contributing a backcast $\hat{\mathbf{x}}_{\ell}$ and forecast $\hat{\mathbf{y}}_{\ell}$ via a learned linear mapping from a shared 4 layer fully connected backbone:

$$\mathbf{h}_{\ell} = f_{\text{bb}}(\mathbf{x}_{\ell}^{\text{in}}), \quad \hat{\mathbf{x}}_{\ell} = g_b(\boldsymbol{\theta}^b), \quad \hat{\mathbf{y}}_{\ell} = g_f(\boldsymbol{\theta}^f)$$

where $\mathbf{x}_{\ell}^{\text{in}} = \mathbf{x} - \sum_{k < \ell}\hat{\mathbf{x}}_k$ is the residual backcast input to block $\ell$, and $g_b, g_f$ are basis expansion generators (defined explicitly in Section~\ref{sec:wavelet}).

All N-BEATS blocks accept a residual backcast(or input if first block) $\mathbf{x} \in \mathbb{R}^L$ and return $(\hat{\mathbf{x}}, \hat{\mathbf{y}}) \in \mathbb{R}^L \times \mathbb{R}^H$.  

### Backbone Blocks

#### RootBlock

The legacy RootBlock is a 4-layer, uniform width, FFN with RELU activation $\sigma$:

$$\mathbf{h} = \sigma\!\left(\mathbf{W}_4\,\sigma\!\left(\mathbf{W}_3\,\sigma\!\left(\mathbf{W}_2\,\sigma(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2\right)+\mathbf{b}_3\right)+\mathbf{b}_4\right), \quad \mathbf{W}_i \in \mathbb{R}^{U\times U}$$

No bottleneck is imposed; all layers share trunk width $U$.

#### AERootBlock

The AERootBlock is a drop-in replacement for the legacy RootBlock backbone.  It introduces an hourglass-shaped autoencoder that routes all information through a low-dimensional latent bottleneck $z \in \mathbb{R}^{d}$:

$$\mathbf{h} = f_{\mathrm{dec}}(f_{\mathrm{enc}}(\mathbf{x})), \quad f_{\mathrm{enc}}: \mathbb{R}^L \!\to\! \mathbb{R}^{U/2} \!\to\! \mathbb{R}^{d}, \quad f_{\mathrm{dec}}: \mathbb{R}^{d} \!\to\! \mathbb{R}^{U/2} \!\to\! \mathbb{R}^U$$

The bottleneck acts as a signal denoiser, regularizer, and feature extractor.  The decoder must reconstruct both output paths from $z$, forcing the backbone to discard irrelevant information and retain the most informative features.

The parameter count of the backbone reduces to $O(UL/2 + Ud/2 + Ud/2 + U^2/2)$. At $U = 512$, $L = 480$, $d = 16$ this is approximately 147K — a 5.5$\times$ reduction relative to RootBlock at equivalent $U$. The interface to any downstream projection head remains identical: both backbones produce $h \in \mathbb{R}^U$.

#### AERootBlockLG

The LG variant RootBlock introduces a learnable sigmoid vector and scales the interior latent representation element-wise:

$$z_{\text{gated}} = \sigma(\gamma) \odot z$$

Because $\sigma(\gamma_i)$ at its limits the sigmoid function approach zero, the network can shut off unnecessary latent dimensions, effectively performing a soft bandpass filter. This is particularly useful when the optimal bottleneck size is unknown. You can get away with selecting a lower latent dimension because the netowork only selects the dimensions it needs.  In testing this manifests as networks using AERootBlocks needing a latent dimension, $ld = 32$, for the M4 dataset, but AELG based netwworks only needing $ld = 16$. For example in a 10 stack TrendWavelet block with AELG root and a latent dimention of 32 using a Daubechies 3 Wavelet, TWAELG_10s_ld32_db3, uses only 0.48M total parameters while remaining competitive with 15M+ RootBlock models.

Table 1 - Backbone Architecture Comparison

| Backbone | Layer Topology | Latent Operation | Extra Params |
| --- | --- | --- | --- |
| RootBlock | input → units → units → units → units | None | Baseline |
| AERootBlock | input → units//2 → latent_dim → units//2 → units | Deterministic pass-through | Negligible |
| AERootBlockLG | Same as AE | z * sigmoid(latent_gate) | latent_dim (gate vector) |

#### Parameter efficiency

The standard width parameters (`g_width=512`, `s_width=2048`, `t_width=256`, `ae_width=512`) govern the main trunk width $U$ for each block family. The `latent_dim` $d$ is an additional hyperparameter that controls only the AE bottleneck and does not change the output dimension $\mathbf{h} \in \mathbb{R}^U$ seen by downstream projection heads.

The backbone parameter count formula for each family is:

- **RootBlock**: $UL + 3U^2$ — four uniform-width layers, no bottleneck.
- **AERootBlock**: $UL/2 + Ud + U^2/2$ — parameters scale roughly as $O(U \cdot d)$ at the bottleneck, which is minor compared to the main width cost.
- **AERootBlockLG**: same as AERootBlock plus $d$ gate parameters — negligible overhead; practical benefit is that $d$ can be halved relative to AERootBlock without accuracy loss.

Because `latent_dim` is typically 5–32, AE variants achieve comparable forecast accuracy with dramatically fewer total backbone parameters than their RootBlock counterparts when `latent_dim` is small and width is constrained. Table 2 gives concrete counts at default widths and a representative backcast length of $L = 480$.

**Table 2 — Backbone parameter counts at default widths ($L = 480$)**

| Backbone | Width $U$ | Latent $d$ | Backbone params | vs. RootBlock |
| --- | --- | --- | --- | --- |
| RootBlock (Generic) | 512 | — | ~810K | 1.0× |
| AERootBlock (Generic) | 512 | 32 | ~163K | 5.0× fewer |
| AERootBlock (Generic) | 512 | 16 | ~147K | 5.5× fewer |
| AERootBlockLG (Generic) | 512 | 16 | ~147K + 16 | ≈5.5× fewer |
| RootBlock (Trend) | 256 | — | ~319K | 1.0× (Trend) |
| AERootBlock (Trend) | 256 | 16 | ~46K | 6.9× fewer |
| AERootBlockLG (Trend) | 256 | 16 | ~46K + 16 | ≈6.9× fewer |

The compression ratio grows with width: wider trunks benefit more from the hourglass bottleneck because the three large $U \times U$ layers in RootBlock scale quadratically while the AE bottleneck scales linearly in $d$. At Generic/Wavelet width ($U = 512$, $d = 16$) the ratio reaches 5.5×; at Trend width ($U = 256$, $d = 16$) it reaches 6.9×. This makes AE backbones attractive for TrendWavelet and TrendWaveletGeneric configurations, where the Trend trunk width is narrow by design and every saved parameter can be reinvested in stack depth or wavelet family selection which improves expressivity of the model.

### Branch-specific encoder-decoder designs

The backbone variants described above produce a single shared hidden representation $\mathbf{h}$ that is then split by separate linear heads into backcast and forecast coefficient vectors. A complementary design gives each output branch its own dedicated encoder-decoder sub-network. The `AutoEncoderAE` and `GenericAEBackcastAE` blocks follow this pattern: after the shared AE trunk compresses the input into a trunk latent code $z$, two independent hourglass sub-branches — one for the backcast path and one for the forecast path — each re-encode $z$ through their own bottleneck before projecting to the output length.

$$z \xrightarrow{\text{enc}_b} z_b \xrightarrow{\text{dec}_b} \hat{\mathbf{x}}, \qquad z \xrightarrow{\text{enc}_f} z_f \xrightarrow{\text{dec}_f} \hat{\mathbf{y}}$$

This gives the network two independent compression points per block. The trunk bottleneck enforces a shared, task-agnostic representation of the input window; the branch bottlenecks then cleans up that signal independently. The two objectives are structurally separated rather than implicitly traded off through a shared linear head.

The `latent_dim` hyperparameter controls the branch-local bottleneck size in these blocks, while `thetas_dim` remains reserved for explicit coefficient projections such as the `GenericAEBackcast*` forecast head. `AutoEncoderAE` omits `thetas_dim` entirely and lets both branch decoders project directly to the target lengths.

**Table 3 — Single-trunk vs branch-specific AE designs**

| Block | Shared trunk | Backcast branch | Forecast branch | `thetas_dim` role |
| --- | --- | --- | --- | --- |
| `GenericAE` | AERootBlock | shared $\mathbf{h}$ → linear | shared $\mathbf{h}$ → linear | unused |
| `AutoEncoderAE` | AERootBlock | $z$ → enc-dec → output | $z$ → enc-dec → output | unused |
| `GenericAEBackcastAE` | AERootBlock | $z$ → enc-dec → output | $z$ → enc-dec → $\boldsymbol{\theta}^f$ → basis | coefficient projection |

The branch-specific design adds approximately $4 \cdot U_b \cdot d_b$ parameters per branch, where $U_b$ is the branch width and $d_b$ the branch latent size. For typical settings ($U_b = U/2$, $d_b = d$) this doubles the AE parameter budget relative to a shared-head block, but keeps the total well below the RootBlock baseline since both the trunk and the branches remain in the hourglass regime.

### Wavelet Family Blocks

The legacy N-BEATS basis-expansion blocks predict coefficient vectors and expand them through a frozen basis into backcast and forecast outputs matching the target lengths. The original interpretable version uses polynomial trend and Fourier seasonality bases. The wavelet blocks replace the Fourier seasonality basis with a fixed, orthonormal discrete wavelet basis. Each block first maps the input window through a RootBlock variant, producing a hidden representation $\mathbf{h}$. Separate trainable linear heads then produce backcast and forecast coefficient vectors,

$$\boldsymbol{\theta}^b = \mathbf{W}_b\mathbf{h},\qquad \boldsymbol{\theta}^f = \mathbf{W}_f\mathbf{h}$$

For each target length $T \in \{L, H\}$, a wavelet synthesis matrix is constructed at model initialization by applying inverse DWT reconstruction to unit impulses in each wavelet coefficient band. This raw synthesis matrix is then orthogonalized via SVD, and the rows of $V^\top$ form the fixed orthonormal bases $\boldsymbol{\Psi}_b \in \mathbb{R}^{k \times L}$ and $\boldsymbol{\Psi}_f \in \mathbb{R}^{k \times H}$ (one per path). The basis expansion generators therefore evaluate to

$$g_b(\boldsymbol{\theta}^b) = \boldsymbol{\theta}^b\,\boldsymbol{\Psi}_b = \hat{\mathbf{x}}, \qquad g_f(\boldsymbol{\theta}^f) = \boldsymbol{\theta}^f\,\boldsymbol{\Psi}_f = \hat{\mathbf{y}}$$

Because $\boldsymbol{\Psi}_b$ and $\boldsymbol{\Psi}_f$ are frozen at initialization and stored as non-trainable buffers, inference requires only a single matrix multiply per path, which GPUs excel at. A full DWT is never computed at training or inference time. Implementation details are given in Appendix~\ref{app:hyperparams}.

#### Composition blocks: TrendWavelet and TrendWaveletGeneric

The strongest single-block family in our benchmarks combines a frozen polynomial trend basis with a frozen orthonormal wavelet basis in a single additive decomposition. The `TrendWavelet` block produces the coefficient vector $\boldsymbol{\theta} \in \mathbb{R}^{p + k}$ from the shared hidden representation $\mathbf{h}$, then partitions and expands it:

$$\hat{\mathbf{y}} = \underbrace{\boldsymbol{\theta}^f_{1:p}\,\mathbf{T}_f}_{\text{trend}} + \underbrace{\boldsymbol{\theta}^f_{p+1:p+k}\,\boldsymbol{\Psi}_f}_{\text{wavelet}}$$

where $\mathbf{T}_f \in \mathbb{R}^{p \times H}$, $\mathbf{T}_b \in \mathbb{R}^{p \times L}$ is the Vandermonde polynomial basis of order $p$ (`trend_thetas_dim`), $\boldsymbol{\Psi}_f \in \mathbb{R}^{k \times H}$, $\boldsymbol{\Psi}_b \in \mathbb{R}^{k \times L}$ is the orthonormal wavelet basis of rank $k$ (`basis_dim`), and the same additive structure holds for the backcast path. The two bases are complementary by construction: the polynomial captures smooth low-frequency trend while the wavelet captures localized oscillatory residuals.

`TrendWaveletGeneric` extends this to a three-way decomposition by appending a third learned (data-driven) branch of rank $r$ (`generic_dim`):

$$\hat{\mathbf{y}} = \boldsymbol{\theta}^f_{1:p}\,\mathbf{T}_f + \boldsymbol{\theta}^f_{p+1:p+k}\,\boldsymbol{\Psi}_f + \boldsymbol{\theta}^f_{p+k+1:p+k+r}\,\mathbf{G}_f$$

where $\mathbf{G}_f \in \mathbb{R}^{r \times H}$ is a trainable rank-$r$ basis matrix. The two structured bases remain frozen; only coefficients and the generic basis are learned. This gives the block a fallback data-driven capacity for signal components that neither polynomial nor wavelet bases can represent compactly.

Both block families are fully composable with both AE backbone variants (AERootBlock, AERootBlockLG), yielding the `TrendWaveletAE`, `TrendWaveletAELG`, `TrendWaveletGenericAE`, and `TrendWaveletGenericAELG` configurations benchmarked in Section~\ref{sec:results}.

#### Asymmetric and tiered wavelet bases

Two additional hyperparameters control how the wavelet basis is oriented relative to the backcast and forecast windows.

**Asymmetric wavelet families.** When backcast and forecast lengths differ substantially — as in Traffic-96 ($L=480$, $H=96$, ratio $5\times$) — the optimal wavelet support length differs between paths. The `backcast_wavelet_type` and `forecast_wavelet_type` parameters allow independent wavelet family selection per path. A long-support wavelet (e.g.\ Symlet-20) captures broad low-frequency structure in the long backcast; a short-support wavelet (e.g.\ Daubechies-3, Coiflet-2) provides compact localized bases for the short forecast. When these overrides are omitted, both paths share the same `wavelet_type`.

**Tiered basis offset.** The `basis_offset` parameter shifts the starting row of the basis matrix selected from $V^\top$, cycling through the SVD spectrum in successive stacks. In a depth-$S$ model, stack $\ell$ uses rows $[\ell \cdot \Delta, \ell \cdot \Delta + k)$ of the orthonormal basis, where $\Delta$ is a stride derived from `basis_offset`. This spreads each stack's basis coverage across different frequency bands, encouraging the doubly residual stack to decompose the signal rather than repeatedly fitting the same spectral region. Tiered offsets are most effective on the Daily M4 period, where they appear in 8 of the top-10 configurations (Section~\ref{sec:ablations}).
