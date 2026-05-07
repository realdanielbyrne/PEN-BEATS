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

This gives the network two independent compression points per block. The trunk bottleneck enforces a shared representation of the input window; the branch bottlenecks then cleans up each signal independently. The two objectives are separated downstream again to try and impart more diversity into the learned parameters.

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

## Experimental Setup

This section describes the datasets, training protocols, architecture sweep, and ablation studies used to evaluate the proposed blocks. Methodology is presented here; results follow in Section~\ref{sec:results}.

### Datasets

The M4 Competition \citep{makridakis2020m4} dataset is the primary evaluation dataset: six frequencies (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly) over 100,000 series with horizons $H \in \{6, 8, 24, 13, 14, 48\}$. This benchmark requires a single architecture to generalize across regimes that differ in sample size, sampling rate, and seasonality without time-seris specific feature engineering or input scaling. We complement M4 with four out-of-distribution datasets to verify that conclusions transfer beyond M4. These are Tourism (1,311 series, Yearly/Quarterly/Monthly), PeMS Traffic (862 hourly sensors), Weather (21 meteorological channels at 10-min resolution), and the univariate Milk production series.

**Table 4 — Datasets and evaluation parameters**

| Dataset | Frequency | Series | Horizon $H$ | Lookback $L$ | Primary metric |
| --- | --- | --- | --- | --- | --- |
| M4 (Y/Q/M/W/D/H) | mixed | 100,000 | $\{6, 8, 24, 13, 14, 48\}$ | $L = 5H$; $L_h = m H$ with $m \in \{1.5, 10\}$ | sMAPE, OWA |
| Tourism (Y/Q/M) | mixed | 1,311 | $\{4, 8, 24\}$ | $L = 5H$ | sMAPE |
| Traffic-96 | hourly | 862 | 96 | $5H$ | MSE, MAE |
| Weather-96 | 10-min | 21 channels | 96 | $5H$ | MSE, MAE |
| Milk | monthly | 1 | 6 | $5H$ | sMAPE |

The paper-sample protocol additionally enforces the N-BEATS in-sample constraint $L_h$ \citep[Appendix~D]{oreshkin2020nbeats}, restricting valid training windows to the trailing $L_h \cdot H$ timesteps of each series; we use $L_h = 1.5$ for Yearly/Quarterly and $L_h = 10$ for the remaining M4 frequencies.

### Training Protocols

Absolute sMAPE differences in basis-expansion benchmarks are often dominated by sampling and learning-rate artifacts rather than by architectural quality. To control for this we evaluate every architecture under **three complementary protocols**. The protocols share the same backbone, loss, and seed budget, but they differ in how training data is presented to the optimizer and in how the learning rate is decayed. A finding is treated as robust when it survives all three.

**Sliding-Window Protocol** (`comprehensive_sweep_m4.yaml`, the `lightningnbeats` default). Epoch-based budget with batches enumerated by sliding the lookback window over each series. Adam at $10^{-3}$, sMAPE loss, max 75 epochs (min 10), early-stopping patience 20 with $\Delta_{\min} = 10^{-3}$, cosine annealing with 15-epoch linear warmup and $\eta_{\min} = 10^{-6}$.

**N-BEATS Paper-Sampling Protocol, faithful replication** (`comprehensive_m4_paper_sample.yaml`). A direct copy of the original N-BEATS training procedure \citep{oreshkin2020nbeats}: iteration-based budget of $10^5$ gradient steps at batch size 1024, with each step drawing (series, anchor) pairs uniformly at random with replacement from the windows that satisfy the $L_h$ constraint. Adam at $10^{-3}$, sMAPE loss, MultiStepLR with 10 evenly-spaced milestones and $\gamma = 0.5$ (LR halved every 10% of training). This protocol exists to put our proposed blocks on the same training footing as the published baseline numbers.

**N-BEATS Paper-Sampling Protocol, plateau LR variant** (`comprehensive_m4_paper_sample_plateau.yaml`). The sampling, batch size, step budget, and early-stopping configuration are identical to the faithful replication. The MultiStepLR is replaced with a more aggressive `ReduceLROnPlateau` scheduler ($\text{factor}=0.5$, patience 3 val-check units, $\text{cooldown}=1$, $\text{min\_lr}=10^{-6}$, monitored on `val_loss` every 100 steps). The plateau variant is included because validation-driven LR decay can resolve to a finer effective schedule than the fixed-milestone schedule. In our experiments it changes which architectures look competitive on the Quarterly, Monthly, and Daily M4 cells, while the rankings on Yearly and Weekly are essentially unchanged. By reporting both variants we can separate wins that come from structured priors from wins that come from the LR schedule happening to suit a particular architecture.

Both paper-sample variants share the same sub-epoch validation configuration: `val_check_interval=100`, patience 15 in val-check units (so a minimum of $1{,}500$ steps must elapse before stopping can fire), and $\Delta_{\min} = 10^{-3}$. Without sub-epoch validation, early stopping under iteration-based training fires on warmup-corrupted checks and collapses to `best_epoch` $\in \{0, 1\}$. This is a small methodological correction that we found is necessary for reproducible paper-sample evaluation.

**Seeds and statistics.** Every (configuration, period, protocol) cell is evaluated over 10 random seeds. Pairwise comparisons use the Wilcoxon signed-rank test on matched seeds with Bonferroni correction; the full multiple-comparison procedure and the verbatim YAML for each protocol are given in Appendix~\ref{app:protocols}.

### Architecture Sweep

The central hypothesis under evaluation is:

> *Structured priors (orthonormal wavelet bases and autoencoder bottlenecks) enable 60–95% parameter compression relative to paper-faithful Generic stacks while matching or exceeding their forecasting accuracy.*

To test this, we define a single 112-configuration **Comprehensive Sweep** evaluated under the sliding-window protocol. A curated 53-configuration subset is re-run under both paper-sample variants (faithful MultiStepLR and plateau LR). The families dropped from this subset (BottleneckGeneric, pure VAE) are excluded *a priori* on reproducibility grounds. They exhibit run-to-run divergence under iteration-based sampling that adds variance without informing the comparison axis. The full configuration list is given in Appendix~\ref{app:configs}; the five groups are:

1. **Paper baselines (8 configs).** `NBEATS-G` (homogeneous Generic) and `NBEATS-I+G` (Trend + Seasonality + Generic), each at depths $\{10, 30\}$ and `active_g` $\in \{\text{False}, \text{forecast}\}$. Reproduces \citet{oreshkin2020nbeats} and supplies the upper-bound parameter count.
2. **Homogeneous TrendWavelet stacks (16 configs).** `TrendWavelet` (RootBlock backbone) at depths $\{10, 30\}$, sweeping `wavelet_type` $\in \{\text{haar}, \text{db3}, \text{coif2}, \text{sym10}\}$, `trend_thetas_dim` $\in \{3, 5\}$, and `basis_dim` $\in \{\text{eq\_fcast}, 2 \times \text{eq\_fcast}\}$. Tests whether one composite basis (polynomial trend + orthonormal DWT) suffices without an explicit Seasonality stack.
3. **Alternating dual-stack architectures (~30 configs).** A *top-quality* pattern alternating `TrendAELG` and `WaveletV3AELG`, and a *stability* pattern alternating `TrendAE` and `WaveletV3AE`. Each is swept across the four wavelets at depths $\{10, 30\}$. The two halves act as separate doubly-residual specialists rather than a single fused block.
4. **Efficiency bottlenecks (~12 configs).** `TrendWaveletAE` and `TrendWaveletAELG` at depth 10, with `latent_dim` $\in \{8, 16, 32\}$. The direct test of the compression hypothesis: does an AE bottleneck of dimension $d \ll \text{units}/2$ preserve accuracy at a fraction of the dense-MLP parameter count?
5. **Generic-AE controls and weight-sharing variants (~46 configs).** `GenericAE`, `GenericAELG`, and `TrendWaveletGenericAELG` at varied depths and latent dimensions, plus eight `share_weights=False` variants. Isolates AE compression *without* wavelet structure (groups 2–4 confound the two) and quantifies the cost of removing weight sharing.

### Ablation: `active_g`

The `active_g` parameter is a `lightningnbeats` extension that gates whether the final linear projection of Generic-family blocks passes through the block's nonlinearity (`active_g='forecast'`) or returns a raw linear map (`active_g=False`, the paper-faithful setting). We evaluate `active_g` $\in \{\text{False}, \text{forecast}\}$ across the GenericAELG group at depths $\{10, 30\}$, holding all other hyperparameters fixed and sweeping under both protocols. The aim is to isolate the impact of architectural regularization on Generic stack stability, independent of the wavelet and AE structure introduced elsewhere. The companion `skip_distance` ablation (cross-stack residual injection cadence) and the `learned_gate` magnitude analysis on AELG blocks are deferred to Appendix~\ref{app:ablations}.

### NHiTS Transferability

To confirm that the proposed blocks are not specific to the N-BEATS doubly-residual stack, we re-run a curated subset of the novel families on the NHiTS backbone \citep{challu2023nhits}, which adds multi-rate input pooling and hierarchical forecast interpolation around the same block interface. The benchmark (`run_nhits_benchmark.py`) covers Weather-96 and Traffic at horizons $\{96, 192, 336, 720\}$ with MSE loss, batch size 256, max 100 epochs, patience 10, $L = 5H$, and 8 seeds. The block sweep includes `Generic`, `GenericAELG`, `BottleneckGenericAELG`, alternating `TrendAELG + Sym20V3AELG`, and `TrendWaveletAELG`, paired with their NHiTS-pooled counterparts using horizon-adaptive pooling/interpolation schedules. When block-level effects survive the backbone change, this is evidence that the structured priors are driving them rather than interactions with N-BEATS-specific residual scheduling.

## Appendix A: Paper-Sample Protocol Details \label{app:protocols}

This appendix documents the two paper-sample training protocols in full so that the M4 numbers in Section~\ref{sec:results} can be reproduced from a single YAML file. Both variants share the sampling strategy and step budget of the original N-BEATS training procedure; they differ only in the learning-rate scheduler.

### A.1 Faithful replication (MultiStepLR)

The faithful variant (`comprehensive_m4_paper_sample.yaml`) is intended as a direct copy of the training procedure described in \citet[Section~5.2 and Appendix~D]{oreshkin2020nbeats}, with no deviations except the small validation-cadence correction described below.

**Sampling.** Each gradient step draws a batch of 1024 (series, anchor) pairs uniformly at random *with replacement* from the training pool. A series is eligible at step $t$ if it has at least $L + H$ observations; for a chosen series of length $T$, the anchor index is sampled uniformly from $[T - L_h \cdot H,\ T - H]$, restricting valid windows to the trailing $L_h \cdot H$ timesteps. Per-period $L_h$ is fixed to the values in Table~\ref{tab:datasets}: $L_h = 1.5$ for Yearly and Quarterly, $L_h = 10$ for Monthly, Weekly, Daily, and Hourly. The lookback length is $L = 5H$.

**Optimization.** Adam with $\eta_0 = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, no weight decay. sMAPE loss is computed per-step on the 1024-window batch. The gradient budget is $10^5$ total steps, with one Lightning epoch defined as 1,000 steps (≈100 paper-epochs).

**Learning-rate schedule.** MultiStepLR with 10 evenly spaced milestones at $\{10\text{k}, 20\text{k}, \ldots, 100\text{k}\}$ steps, $\gamma = 0.5$. The LR is therefore halved at each 10% boundary of the training budget. This is exactly the schedule used by Oreshkin et al.

**Validation and early stopping.** Validation is run every 100 training steps (`val_check_interval=100`) on the held-out window of each series. Early stopping monitors validation sMAPE with patience 15 (counted in val-check units, so a minimum of $1{,}500$ steps must elapse before stopping can fire) and $\Delta_{\min} = 10^{-3}$. The sub-epoch validation cadence is the only departure from a strict line-by-line copy of \citet{oreshkin2020nbeats}; without it, early stopping fires on the first warmup-corrupted check and collapses to `best_epoch` $\in \{0, 1\}$.

**Other settings.** `n_blocks_per_stack = 1`, `share_weights = true`, ReLU activations throughout, `forecast_multiplier = 5`, default block widths (`g_width = 512`, `s_width = 2048`, `t_width = 256`).

### A.2 Plateau LR variant (ReduceLROnPlateau)

The plateau variant (`comprehensive_m4_paper_sample_plateau.yaml`) is identical to A.1 in every respect except the learning-rate schedule. The motivation is that MultiStepLR commits to a fixed decay schedule independent of how training is actually progressing. A plateau scheduler can react to validation stalls and find a more aggressive effective decay on cells where the loss surface is well-behaved.

**Learning-rate schedule.** `ReduceLROnPlateau` with `factor = 0.5`, `patience = 3` (val-check units, so the LR is reduced after 3 consecutive 100-step val checks without improvement of at least $\Delta_{\min} = 10^{-3}$), `cooldown = 1` (one val-check skipped after each reduction before monitoring resumes), `min_lr = 10^{-6}`, `mode = min`, monitored on `val_loss`. The scheduler is stepped on the same cadence as validation (every 100 training steps) so that its trigger logic operates in the same time units as early stopping.

**Effect.** Empirically the plateau scheduler delivers more LR reductions than MultiStepLR on M4 Quarterly, Monthly, and Daily and fewer on Yearly and Weekly, which matches the observed differences in the per-period leaderboards (Section~\ref{sec:results}).

All other settings (sampling, $L_h$, batch size, step budget, sub-epoch validation, early stopping, optimizer, loss, block widths, weight sharing, seed budget) are inherited unchanged from A.1.

### A.3 Why two paper-sample protocols

The N-BEATS paper reports a single training procedure, so any one-protocol comparison conflates the architectural change with whatever artifacts that specific schedule introduces on the new architecture. By running each architecture under both A.1 (faithful) and A.2 (plateau) we can attribute observed gains to one of three categories. (i) Gains under both protocols are robust evidence for the architectural claim. (ii) Gains under the faithful protocol only constitute a head-to-head replacement for the published baseline. (iii) Gains under the plateau protocol only are a finding contingent on a more responsive LR schedule, which we report as such rather than as a clean architectural win.
