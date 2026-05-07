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

Most directly related, \citet{pramanick2024fusion} apply DWT preprocessing to N-BEATS itself, training separate Generic stacks per sub-band; our work differs by embedding the wavelet basis inside each block rather than as an external preprocessor.

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

## Results \label{sec:results}

All tables apply the strict health filter used in the analysis reports: a run is excluded if `diverged=True`, if sMAPE exceeds 100, or if `best_epoch=0` with sMAPE above 50. Means and standard deviations are computed over surviving seeds. For M4, the main comparison uses the paper-sample protocol to stay close to the N-BEATS training setup of \citet{oreshkin2020nbeats}; sliding-window M4 results, plateau-LR variants, and full per-period tables are deferred to Appendix~\ref{app:m4_tables} because absolute sMAPE is not comparable across sampling protocols.

### Per-Period Leaderboards \label{sec:m4_leaderboards}

Table 5 gives the compact leaderboard used for the main claims. Wavelet and TrendWavelet variants win or tie most short- and medium-horizon tasks, but the result is not universal: N-BEATS-I+G \citep{oreshkin2020nbeats} remains the M4-Quarterly and M4-Hourly reference. M4-Hourly also illustrates an important convention: the winning `agf` configuration uses `active_g=forecast`, a `lightningnbeats` extension rather than the paper-faithful N-BEATS block.

**Table 5 — Protocol-local winners used for the main comparison.** M4 rows use the paper-sample protocol; Tourism, Weather, and Milk use their comprehensive-sweep protocols. Scores are mean $\pm$ std over the surviving seeds. Asterisks mark `active_g=forecast`, which is not part of the original N-BEATS paper~\citep{oreshkin2020nbeats}.

| Task | Metric | Best config | Score | Params | $n$ | N-BEATS ref. |
| --- | --- | --- | --- | --- | --- | --- |
| M4-Yearly    | sMAPE | `T+Coif2V3 30s bdeq`     | $13.542 \pm 0.148$ | 15.2M | 10 | G-10: 13.564 |
| M4-Quarterly | sMAPE | `NBEATS-IG 10s ag0`      | $10.312 \pm 0.055$ | 19.6M | 10 | same |
| M4-Monthly   | sMAPE | `TW 30s td3 sym10`       | $13.240 \pm 0.334$ | 6.8M  | 9  | IG-30: 13.307 |
| M4-Weekly    | sMAPE | `T+Coif2V3 30s bdeq`     | $6.735 \pm 0.203$  | 15.8M | 10 | IG-10: 6.974 |
| M4-Daily     | sMAPE | `T+Sym10V3 10s tiered`   | $3.014 \pm 0.035$  | 5.3M  | 10 | IG-30: 3.062 |
| M4-Hourly    | sMAPE | `NBEATS-IG 30s agf`*     | $8.758 \pm 0.099$  | 43.6M | 10 | IG-30: 8.906 |
| Tourism-Y    | sMAPE | `TW 10s td3 db3`         | $21.773 \pm 0.384$ | 2.0M  | 10 | IG-10*: 22.180 |
| Weather-96   | MSE   | `TAE+DB3V3AE 30s ld8`    | $0.138 \pm 0.027$  | 7.1M  | 10 | IG-10: 0.183 |
| Milk         | sMAPE | `TALG+DB3V3ALG 10s`      | $1.512 \pm 0.572$  | 1.0M  | 10 | IG-10: 1.785 |

The strongest cross-dataset statement is therefore conditional rather than absolute: structured wavelet bases are a better use of parameters on most tasks in the sweep, but the winning architecture changes with horizon, sampling protocol, and dataset size. Tourism also has a targeted result outside the 112-config sweep (`TrendWaveletAELG_coif3_eq_bcast_td3_ld16`, sMAPE 20.864), so Table 5 should not be read as a final Tourism SOTA claim.

### Parameter-Efficiency Frontier \label{sec:param_efficiency}

Parameter efficiency is the most stable positive result and is the direct test of the central hypothesis stated in Section~\ref{sec:experimental_setup}. Table 6 isolates the best sub-1M M4 model for each paper-sample period. These models are not always winners — especially on Weekly — but they repeatedly sit close to the top of the leaderboard at one-tenth to one-fiftieth the parameter count. The clearest example is M4-Hourly: `TWAELG_10s_ld32_db3_agf` uses 0.85M parameters and trails the 43.6M `NBEATS-IG_30s_agf` reference by only 0.166 sMAPE.

**Table 6 — Best sub-1M M4 configurations under the paper-sample protocol.** "Gap" is relative to the period winner in Table 5; "Param ratio" compares the period winner's parameter count with the compact model.

| Period | Compact config | Params | sMAPE | Gap | Param ratio |
| --- | --- | --- | --- | --- | --- |
| Yearly    | `TWAE 10s ld32 ag0`        | 0.48M | $13.546 \pm 0.102$ | +0.03% | 32× |
| Quarterly | `TWAE 10s ld32 ag0`        | 0.49M | $10.404 \pm 0.028$ | +0.89% | 40× |
| Monthly   | `TWAE 10s ld32 sym10`      | 0.58M | $13.513 \pm 0.378$ | +2.06% | 12× |
| Weekly    | `TWAELG 10s ld32 db3`      | 0.54M | $7.252 \pm 0.263$  | +7.68% | 29× |
| Daily     | `TWGAELG 10s ld16 db3`     | 0.52M | $3.051 \pm 0.049$  | +1.21% | 10× |
| Hourly    | `TWAELG 10s ld32 db3 agf`* | 0.85M | $8.924 \pm 0.129$  | +1.89% | 51× |

### Stability and Ablations \label{sec:depth_stability}

The negative results are as important as the winners. First, overparameterized Generic stacks are unstable: `NBEATS-G_30s_ag0` (paper baseline group of Section~\ref{sec:experimental_setup}) has bimodal paper-sample behavior on M4-Quarterly ($15.40 \pm 9.97$), M4-Weekly ($8.78 \pm 4.61$), and M4-Daily ($10.21 \pm 12.01$), even though its best seeds can look competitive. TrendWavelet blocks do not show the same stuck-mode behavior. Second, the Milk sweep shows the small-data version of the same effect: RootBlock/Generic families have high divergence, while AE-family TrendWavelet models reduce divergence to the low single digits and give the reliable Milk winner in Table 5.

The ablations support five practical rules; the underlying matched-seed Wilcoxon tables are listed in Appendix~\ref{app:ablations}.

1. `active_g=forecast` is not a global default. It is decisive on M4-Hourly (9/9 matched pairs improve, 5 significant at $p<0.05$) and rescues some Generic convergence failures, but is worse or neutral elsewhere and harmful for Weather unified stacks.
2. AE and AELG are statistically similar on M4 at matched configurations; AELG is preferred only when its smaller latent dimension is needed for parameter budget.
3. Tiered basis offsets are strongest on paper-sample M4-Daily but regress on Weekly, so they belong to a Daily-specific recipe rather than the main default.
4. Wavelet family matters across datasets, but the M4 shortlist reduces to `haar`, `db3`, `coif2`, and `sym10`; `coif3` has no per-period M4 SOTA and is dropped from the main defaults.
5. Backbone preference reverses by data regime: RootBlock variants are strongest on M4 and Tourism, while AE variants lead on Weather and Milk.

### NHiTS Transferability and Weather-Only Transfer \label{sec:transfer_summary}

The block interface transfers more cleanly than the hyperparameters. Re-running the curated NHiTS sweep of Section~\ref{sec:experimental_setup} on the NHiTS backbone~\citep{challu2023nhits}, we report Weather-96 / Weather long-horizon transfer as the primary transfer evidence; the matching Traffic horizons run under the same harness are deferred to Appendix~\ref{app:ablations} because their per-horizon variance is still settling and they do not yet alter the qualitative picture.

The Weather NHiTS sweep yields four refined findings:

1. *NHiTS-Generic is not the best use of the NHiTS stack.* TrendWavelet and BottleneckGenericAE/AELG families dominate the short-epoch Weather sweep when substituted for the vanilla NHiTS Generic blocks, while preserving the NHiTS pooling and hierarchical interpolation schedules.
2. *Horizon shifts the ranking.* Short-horizon NHiTS (H=96) favors unified `TrendWavelet` blocks, whereas long horizons (H=336, 720) favor `BottleneckGenericAE`. The same horizon-dependent reordering is visible on the partial Traffic sweep but is reported as supporting evidence only.
3. *NHiTS reorders the hyperparameter axes seen on N-BEATS.* `active_g=forecast`, which is decisive on M4-Hourly under the doubly-residual N-BEATS stack, is neutral-to-negative across the NHiTS Weather benchmark.
4. *The strong VAE result is architecture-specific.* `TrendWaveletGenericVAE` is unexpectedly competitive in the NHiTS Weather sweep. We treat this as an NHiTS-specific observation rather than as evidence that VAE backbones generalize broadly — pure VAE configurations remain excluded *a priori* elsewhere on reproducibility grounds (Section~\ref{sec:limitations}).

The cross-architecture takeaway is therefore narrow but genuine: structured wavelet bases and AE/AELG bottlenecks are useful reusable components even when removed from the N-BEATS doubly-residual schedule, but their optimal stack pattern, activation setting, latent dimension, and scheduler remain dataset- and architecture-dependent.

## Limitations \label{sec:limitations}

The architectural sweep is centered on the M4 benchmark; while we include Tourism, Traffic, Weather, and Milk as out-of-distribution checks, the breadth of structured-prior conclusions outside M4 is bounded by the size of those auxiliary sweeps. The compute budget supported 10 seeds per cell on a single-author setup; some 1–2 sigma effects therefore retain residual seed sensitivity. Pure VAE-family blocks were observed to diverge under iteration-based paper sampling and are excluded *a priori*; we do not claim VAE backbones are unworkable in general, only that they were not stabilized within this study. Finally, all reported gains assume a single architecture is trained per dataset/period; ensembling, common in published M4 results, is intentionally outside scope to keep the parameter-count comparison clean.

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

## Appendix B: Hyperparameters and Empirical Defaults \label{app:hyperparams}

This appendix is the single reference for every hyperparameter exposed by the `lightningnbeats` model classes used in this paper. For each hyperparameter we report (i) the constructor default in the released package, (ii) the value used in the M4 sweeps reported in Section~\ref{sec:results}, (iii) the swept range when applicable, and (iv) the empirical recommendation that came out of those sweeps. Recommendations are M4-only; out-of-distribution dataset deltas (Tourism, Traffic, Weather, Milk) are summarized in Section~\ref{sec:experimental_setup} and not repeated here. Reproducibility pointers to canonical CSVs and YAML configs are collected in Section~\ref{app:hyperparams:repro}.

### B.1 Code defaults vs paper-sweep values \label{app:hyperparams:defaults}

Table 7 enumerates every keyword argument of `NBeatsNet.__init__` (`src/lightningnbeats/models.py`, lines 88–313). The constructor defaults define the released package surface; the paper-sweep column lists the value or grid actually used in the experiments of Section~\ref{sec:experimental_setup}. Where the paper-sweep value disagrees with the constructor default — most notably `share_weights` — the override is documented at Appendix~\ref{app:protocols} (see "Other settings").

**Table 7 — N-BEATS / lightningnbeats hyperparameters: constructor defaults and the values used in this paper's M4 sweeps.**

| Parameter | Constructor default | Paper sweep value (M4) |
| --- | --- | --- |
| *Optimization (`_NBeatsBase`)* | | |
| `loss` | `SMAPELoss` | `SMAPELoss` |
| `optimizer_name` | `Adam` | `Adam` ($\beta_1{=}0.9$, $\beta_2{=}0.999$) |
| `learning_rate` | $10^{-3}$ | $10^{-3}$ |
| `sum_losses` | `False` | `False` |
| `kl_weight` | $10^{-3}$ | $0.1$ for VAE families only |
| *Stack composition (`NBeatsNet`)* | | |
| `stack_types` | *required* | per-config (Section~\ref{sec:experimental_setup}) |
| `n_blocks_per_stack` | $1$ | $1$ |
| `share_weights` | `False` | `True` (paper-sample, Appendix~\ref{app:protocols}) |
| `active_g` | `False` | swept over $\{\text{False},\,\text{'forecast'}\}$ |
| *Block widths* | | |
| `g_width` | $512$ | $512$ |
| `s_width` | $2048$ | $2048$ |
| `t_width` | $256$ | $256$ |
| `ae_width` | $512$ | $512$ |
| *Block-level priors* | | |
| `latent_dim` | $5$ | swept $\{8, 16, 32\}$ |
| `basis_dim` | $32$ | set per-period via `eq_fcast` or `2`$\,\cdot\,$`eq_fcast` |
| `forecast_basis_dim` | `None` | not used on M4 |
| `trend_thetas_dim` | $3$ | swept $\{3, 5\}$ |
| `wavelet_type` | `db3` | swept $\{\text{haar},\,\text{db3},\,\text{coif2},\,\text{sym10}\}$ |
| `backcast_wavelet_type` | `None` | not used on M4 (Traffic-96 only) |
| `forecast_wavelet_type` | `None` | not used on M4 (Traffic-96 only) |
| *Cross-stack residuals* | | |
| `skip_distance` | $0$ | $0$ |
| `skip_alpha` | $0.1$ | not exercised on M4 |

**Lookback and forecast multiplier.** For all M4 cells we use $L = 5H$ (`forecast_multiplier=5`). The paper-sample protocol additionally clamps the valid sampling window to the trailing $L_h \cdot H$ time steps with $L_h = 1.5$ on Yearly/Quarterly and $L_h = 10$ elsewhere; see Appendix~\ref{app:protocols}.

### B.2 Architectural sweep grids \label{app:hyperparams:grids}

Table 8 lays out the grid actually executed for each block family in the 112-config Comprehensive Sweep (sliding) and the curated 53-config subset (paper-sample MultiStepLR + plateau). Block-family abbreviations match Section~\ref{sec:experimental_setup}: `T+W` = alternating `Trend` + `WaveletV3` (RootBlock), `T+W AELG` = alternating `TrendAELG` + `WaveletV3AELG`, `TW` = unified `TrendWavelet` block, `TWAE`/`TWAELG` = unified `TrendWaveletAE`/`TrendWaveletAELG`, `TWGAELG` = `TrendWaveletGenericAELG`.

**Table 8 — Hyperparameter grids per block family.** "—" means the parameter does not apply to the family.

| Family | $n_\text{stacks}$ | wavelet | `trend_thetas_dim` | `basis_dim` | `latent_dim` | `active_g` | `share_weights` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `NBEATS-G`    | $\{10, 30\}$ | —    | —    | —    | —    | $\{\text{F},\,\text{f}\}$ | `True` |
| `NBEATS-IG`   | $\{10, 30\}$ | —    | $3$   | —    | —    | $\{\text{F},\,\text{f}\}$ | `True` |
| `TW`          | $\{10, 30\}$ | shortlist | $\{3, 5\}$ | $\{\text{eq},\,2{\cdot}\text{eq}\}$ | —   | `F` | `True` |
| `T+W` (RB)    | $\{10, 30\}$ | shortlist | $3$ | $\{\text{eq},\,2{\cdot}\text{eq}\}$ | — | $\{\text{F},\,\text{f}\}$ | `True` |
| `T+W AELG`    | $\{10, 30\}$ | shortlist | $3$ | `eq` | $\{16, 32\}$ | $\{\text{F},\,\text{f}\}$ | `True` |
| `TWAE`        | $10$ | shortlist | $3$ | `eq` | $\{8, 16, 32\}$ | $\{\text{F},\,\text{f}\}$ | `True` |
| `TWAELG`      | $10$ | shortlist | $3$ | `eq` | $\{16, 32\}$ | $\{\text{F},\,\text{f}\}$ | `True` |
| `TWGAELG`     | $10$ | shortlist | $3$ | `eq` | $\{16, 32\}$ | $\{\text{F},\,\text{f}\}$ | `True` |

*Wavelet shortlist:* $\{\text{haar},\,\text{db3},\,\text{coif2},\,\text{sym10}\}$. *$\text{eq} \equiv \text{forecast\_length}$.* $\text{F} \equiv$ `False`, $\text{f} \equiv$ `'forecast'`.

**Wavelet basis implementation.** For each target length $T \in \{L, H\}$, `_WaveletGeneratorV3` (`src/lightningnbeats/blocks/blocks.py:2621`) builds a raw synthesis matrix of size $T \times T$ by reconstructing each unit impulse in each DWT coefficient band through `pywt.waverec`. The decomposition level is $\min(L_{\max}, 5)$, where $L_{\max}=\texttt{pywt.dwt\_max\_level}(T,\,\texttt{dec\_len})$; when $L_{\max}=0$ (filter longer than target), we keep level $0$ rather than forcing a boundary-corrupted level-$1$ decomposition. The raw matrix is then SVD-orthogonalized: rows of $V^\top$ form an orthonormal basis ordered low-to-high frequency by singular value. `basis_offset` chooses the starting row and `basis_dim` the row count, so a stack at offset $\ell\,\Delta$ with width $k$ uses $V^\top[\ell\,\Delta:\ell\,\Delta+k,\;:]$. The result is registered as a non-trainable buffer; inference is a single `matmul`.

### B.3 Training-protocol hyperparameters (sliding) \label{app:hyperparams:sliding}

The faithful and plateau paper-sample variants are documented in Appendix~\ref{app:protocols}. The sliding-window protocol used by the Comprehensive Sweep (`experiments/configs/comprehensive_sweep_m4.yaml`) is: epoch-based training with sMAPE loss, Adam at $10^{-3}$, batches drawn by sliding the lookback over each series; `max_epochs`$=75$, `min_epochs`$=10$, early-stop `patience`$=20$ with $\Delta_{\min}=10^{-3}$; cosine annealing with a $15$-epoch linear warmup down to $\eta_{\min}=10^{-6}$. Sub-epoch validation is not required under sliding because the per-epoch validation pass is on the natural held-out tail of each series.

### B.4 Per-period recommendations \label{app:hyperparams:perperiod}

Sections~\ref{app:hyperparams:lr}–\ref{app:hyperparams:wavelet} are the empirical findings that came out of the sweeps in Section~\ref{sec:results}. Numbers are taken verbatim from the meta-analysis report `experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md`; absolute sMAPE is not comparable across protocols.

#### B.4.1 LR scheduler by period and protocol \label{app:hyperparams:lr}

**Table 9 — Recommended LR scheduler per (period, protocol).** Plateau $\equiv$ `ReduceLROnPlateau` with $\text{patience}=3$ val-check units, $\text{factor}=0.5$, $\text{cooldown}=1$, $\text{min\_lr}=10^{-6}$. Step-paper $\equiv$ MultiStepLR with $10$ evenly spaced milestones, $\gamma=0.5$. Cosine-warmup $\equiv$ $15$-epoch linear warmup then cosine annealing to $\eta_{\min}=10^{-6}$.

| Period | Paper-sample (`nbeats_paper`) | Sliding |
| --- | --- | --- |
| Yearly    | step-paper (neutral; either works) | cosine-warmup |
| Quarterly | **plateau** ($\Delta_{\text{step}\to\text{plateau}}=-0.029$, $n=30$) | cosine-warmup |
| Monthly   | **plateau** ($\Delta=-0.089$, median $-0.198$, $n=18$) | cosine-warmup |
| Weekly    | step-paper (plateau regresses by $+0.20$ at $n=10$, locked out) | cosine-warmup |
| Daily     | **plateau** ($\Delta=-0.028$, $n=13$) | cosine-warmup |
| Hourly    | step-paper (head-to-head against plateau not yet run on novel families) | cosine-warmup |

**Mandatory.** Plateau LR on `nbeats_paper` requires sub-epoch validation: `val_check_interval=100`, `patience=20` val-check units, $\Delta_{\min}=10^{-3}$. Without these, early stopping fires on the first warmup-corrupted check and collapses to `best_epoch` $\in \{0, 1\}$ (Appendix~\ref{app:protocols}).

#### B.4.2 Sampling protocol verdict per period \label{app:hyperparams:sampling}

**Table 10 — Sampling-protocol verdict per M4 period.** $\Delta = \text{sliding} - \text{paper-sample}$ on the best config for each protocol; negative $\Delta$ favors sliding. Absolute sMAPE within a row is not comparable across protocols (Appendix~\ref{app:protocols}).

| Period | $H$ | Best sliding sMAPE | Best paper-sample sMAPE | $\Delta$ | Winner |
| --- | --- | --- | --- | --- | --- |
| Yearly    | $6$  | $13.499$ | $13.542$ | $\approx 0$ | tie |
| Quarterly | $8$  | $10.127$ | $10.313$ | $-0.19$ | **sliding** |
| Monthly   | $18$ | $13.279$ | $13.240$ | $+0.04$ | paper-sample |
| Weekly    | $13$ | $6.671$  | $6.735$  | $-0.06$ | sliding |
| Daily     | $14$ | $2.588$  | $3.012$  | $\mathbf{-0.42}$ | **sliding** |
| Hourly    | $48$ | $8.587$  | $8.758$  | $-0.17$ | **sliding** |

For paper-faithful comparison to \citet{oreshkin2020nbeats} we report paper-sample numbers in the main body. For absolute-sMAPE leaderboards on long-horizon periods (Daily, Hourly, Quarterly), sliding is the stronger protocol. We never mix the two within a leaderboard.

#### B.4.3 Wavelet family per period \label{app:hyperparams:wavelet}

The shortlist $\{\text{haar},\,\text{db3},\,\text{coif2},\,\text{sym10}\}$ is the M4 default. `coif3` is dropped because it produced no per-period SOTA in either protocol despite being tested. Per-period leaders are summarized in Table 11. The cross-period generalist under paper-sample is `sym10` (configuration `T+Sym10V3_10s_tiered_ag0`, mean rank $13.33/108$); under sliding it is `haar` (`T+HaarV3_30s_bd2eq`, mean rank $12.7/112$).

**Table 11 — Best wavelet family per M4 period within the shortlist.** Paper-sample uses the LR scheduler from Table 9; sliding uses cosine-warmup throughout.

| Period | Paper-sample best wavelet | Sliding best wavelet |
| --- | --- | --- |
| Yearly    | db3 / coif2 (within seed noise)         | coif2 |
| Quarterly | sym10 (paper baseline NBEATS-IG wins overall) | sym10 / haar |
| Monthly   | sym10 (unified TW)                      | coif2 (unified TW) |
| Weekly    | coif2 (alternating T+W)                 | db3 (alternating T+W) |
| Daily     | sym10 (tiered)                          | paper baseline (NBEATS-G) |
| Hourly    | sym10 (tiered) or paper baseline        | paper baseline (NBEATS-IG) |

### B.5 Block-level recommendations \label{app:hyperparams:block}

The following compact rules summarize the empirical findings for each block-level hyperparameter on M4. Each rule is sourced from the meta-analysis report and the sweep CSVs in `experiments/results/m4/`.

**`basis_dim`.** Use the label `eq_fcast`, i.e.\ $k = H$, as the default. At convergence (R3, $\geq 50$ epochs equivalent) the spread between `eq_fcast`, `lt_bcast`, and `eq_bcast` is $<0.014$ sMAPE on M4-Yearly across $14$ wavelet families and three datasets; the apparent advantage of larger labels at $10$ epochs is a convergence-speed artifact. Avoid `lt_fcast` for $H \le 8$ ($0\%$ R3 survival on M4-Yearly in the AELG sweep). For unified `TrendWavelet` blocks the basis-dim sensitivity is higher (spread $0.37$ sMAPE), so `eq_fcast` is even more critical there.

**`latent_dim`.** For `AERootBlockLG`-backed blocks (`TWAELG`, `TAELG+WV3AELG`, etc.), `latent_dim`$=16$ is the default ($n\geq 600$ runs, $p<0.02$ vs $32$ on both M4-Yearly and Tourism-Yearly). For plain `AERootBlock`, `latent_dim`$\in\{5, 8\}$ are statistically equivalent (Mann–Whitney $p=0.71$, $n=330$ each); `latent_dim`$=2$ is catastrophically underparameterized ($p<3\times 10^{-20}$ vs $\{5, 8\}$). The unified `TrendWaveletAELG` block prefers `latent_dim`$=16$, with one known catastrophic combination: `db4` + `eq_fcast` + `latent_dim`$=16$ (sMAPE $\sim 76$); `db4` at `latent_dim`$=8$ or with other basis-dim labels is fine.

**`trend_thetas_dim`.** Use $3$ for $H \le 10$ and $5$ for $H \ge 50$. Significant on Tourism-Yearly ($H=4$, $p=0.008$, ttd$=3$) and on Weather-96 ($H=96$, $p=0.042$, ttd$=5$) at convergence. Mid-horizon crossover (Monthly $H=18$, Weekly $H=13$, Daily $H=14$) defaults to $3$ on insufficient evidence. The earlier "always ttd$=3$" finding was a $10$-epoch convergence-speed artifact on Weather, reversed at $50$ epochs.

**Trend backbone hierarchy.** On M4 the ordering is RootBlock $>$ AELG $>$ AE for alternating Trend+Wavelet stacks. The plain AE bottleneck homogenizes wavelet-family signal in alternating configurations ($\eta^2 = 0.003$, ns) and is not viable; the learned gate in AELG restores competitiveness. On Weather-96 and Milk this hierarchy reverses (AE $>$ AELG, Mann–Whitney $p=0.036$ on Weather), but those datasets are out of scope for this appendix.

**Block ordering.** Always place `Trend*` stacks before `Wavelet*`, `Generic*`, or `Seasonality*` stacks. In the VAE2 study (M4-Yearly + Weather-96) every significantly-different ordering pair — including one Bonferroni-significant pair on M4 (Cohen's $r=0.83$) — favored Trend-first; no pair favored the reverse. For VAE-family backbones, reversing the order can $2{-}3\times$ the OWA. Treat this as a fixed convention, not a hyperparameter to search.

**TrendWavelet vs alternating T+W.** Under paper-sample protocol on M4, alternating $T{+}\langle\text{wav}\rangle$V3 (RootBlock) beats unified `TrendWavelet` by $\approx 10$ mean-rank points ($14.1$ vs $24.3$ across $53$ configs $\times 6$ periods). Default to alternating when $\geq 15$M parameters is acceptable. Reserve unified `TrendWavelet` (or its AE/AELG variants) for parameter-constrained scenarios: `TWAELG_10s_ld32_db3` achieves $0.48$–$0.85$M parameters and top-$5$ on Yearly, Daily, and Hourly. Plateau LR has flipped the Monthly winner to unified `TW_30s_td3_bdeq_sym10` ($13.240$); on every other period alternating is at least co-optimal.

**`active_g`.** The constructor default `False` (paper-faithful, abbreviated `ag0`) is the robust choice on M4. `active_g`$=$`'forecast'` (`agf`) helps on Hourly only — every paper backbone gains $0.07$–$0.15$ sMAPE there — and ties or loses on every other period. `active_g` is a `lightningnbeats` extension and is not part of the original N-BEATS architecture: configurations marked `ag0` are paper-faithful, configurations marked `agf` apply the extension and are not strictly comparable to \citet{oreshkin2020nbeats}.

**Other ablations (not exercised on M4).** `skip_distance` (cross-stack residual injection) never helps on M4 in the configurations tested and slightly hurts deep stacks; `sum_losses` adds a backcast-reconstruction term ($0.25 \cdot \mathrm{sMAPE}(\hat{x}, 0)$) and is reserved for the convergence study reported in Appendix~\ref{app:ablations}. Both default to off in this paper.

### B.6 Drop list \label{app:hyperparams:drop}

Configurations excluded *a priori* from the curated paper-sample subset and from any paper-headline leaderboard. Each is supported by at least one CSV in `experiments/results/m4/`.

- **BottleneckGeneric family** (`BNG*`, `BNAE*`, `BNAELG*`): universally worst on M4.
- **Pure VAE configs** (`GenericVAE_3s_sw0`, etc.): mean sMAPE $55$–$68$ on Q/M/W/D, unusable.
- **`NBEATS-G_30s_ag0` on Q/W/M**: bimodal collapse, sMAPE std $7.4$–$9.5$.
- **All `*_sd5` variants** (`skip_distance`$=5$): never help on M4.
- **All `*_coif3` variants on M4**: no per-period SOTA across either protocol.
- **`_30s_agf` tiered configs at `_10s` depth**: run-to-run divergence outliers.
- **Step-paper LR on Q/M/D**: dominated by plateau LR (Table 9).
- **Pure `GenericAE_*`, `GenericAELG_*`**: bottoms out at mean rank $\approx 38$.

### B.7 Quick-pick decision table \label{app:hyperparams:quickpick}

Table 12 is the per-period recipe to copy into a YAML config for a new M4 experiment. Each row reproduces the per-period top cell of the appropriate leaderboard (paper-sample if the column says `nbeats_paper`, sliding otherwise) from `m4_overall_leaderboard_2026-05-03.md`.

**Table 12 — Best M4 configuration per (period, protocol).** All cells are $n=10$ unless noted. "ttd" = `trend_thetas_dim`; "ld" = `latent_dim`; "bd" = `basis_dim` label; "`ag`" = `active_g` (`0` = `False`, `f` = `'forecast'`).

| Period | Protocol | LR | Block family / config | Depth | Wavelet | ttd | bd | ld | ag | sMAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Yearly    | paper-sample | plateau    | `T+Coif2V3` (RB)             | $30$ | coif2 | $3$ | `eq`    | —   | 0 | $13.542$ |
| Quarterly | paper-sample | plateau    | `NBEATS-IG`                  | $10$ | —    | $3$ | —             | —   | 0 | $10.313$ |
| Monthly   | paper-sample | plateau    | `TW` (unified)               | $30$ | sym10 | $3$ | `eq`    | —   | 0 | $13.240$ |
| Weekly    | paper-sample | step-paper | `T+Coif2V3` (RB)             | $30$ | coif2 | $3$ | `eq`    | —   | 0 | $6.735$ |
| Daily     | paper-sample | plateau    | `T+Sym10V3` (RB, tiered)     | $10$ | sym10 | $3$ | `eq`    | —   | 0 | $3.012$ |
| Hourly    | paper-sample | step-paper | `NBEATS-IG`                  | $30$ | —    | $3$ | —             | —   | f | $8.758$ |
| Yearly    | sliding      | cos-warmup | `TW` (unified)               | $10$ | coif2 | $3$ | `eq`    | —   | 0 | $13.499$ |
| Quarterly | sliding      | cos-warmup | `NBEATS-IG`                  | $10$ | —    | $3$ | —             | —   | 0 | $10.127$ |
| Monthly   | sliding      | cos-warmup | `TW` (unified)               | $30$ | coif2 | $3$ | `2eq`   | —   | 0 | $13.279$ |
| Weekly    | sliding      | cos-warmup | `T+Db3V3` (RB)               | $30$ | db3   | $3$ | `eq`    | —   | 0 | $6.671$ |
| Daily     | sliding      | cos-warmup | `NBEATS-G`                   | $30$ | —    | —  | —             | —   | 0 | $2.588$ |
| Hourly    | sliding      | cos-warmup | `NBEATS-IG`                  | $30$ | —    | $3$ | —             | —   | f | $8.587$ |
| *Sub-1M parameter champions (paper-sample):* | | | | | | | | | | |
| Yearly    | paper-sample | plateau | `TWAE_10s_ld32`             | $10$ | db3   | $3$ | `eq` | $32$ | 0 | $13.546$ |
| Daily     | paper-sample | plateau | `TWGAELG_10s_ld16_db3`      | $10$ | db3   | $3$ | `eq` | $16$ | 0 | $3.051$ |

### B.8 Cross-period generalists \label{app:hyperparams:generalist}

When a single configuration must run across all six M4 periods (cross-period reporting, ensemble bases, ablation harnesses), three configurations dominate by mean per-period rank:

- **Paper-sample, all-6-period coverage:** `T+Sym10V3_10s_tiered_ag0`, mean rank $13.33/108$, $5.07$–$6.06$M parameters; top-$11$ on every M4 period.
- **Paper-sample, paper-faithful alternative:** `NBEATS-IG_30s_ag0`, mean rank $16.0/68$, $38$–$44$M parameters. Use when reproducing \citet{oreshkin2020nbeats} or when zero divergence is required.
- **Sliding, all-6-period coverage:** `T+HaarV3_30s_bd2eq`, mean rank $12.7/112$, $16.25$M parameters; smaller alternative `TALG+HaarV3ALG_30s_ag0` ($3.66$M, mean rank $21.0$).

### B.9 Reproducibility \label{app:hyperparams:repro}

**Canonical CSVs (`experiments/results/m4/`).** Paper-sample (step-paper LR): `comprehensive_m4_paper_sample_results.csv`, `comprehensive_m4_paper_sample_sym10_fills_results.csv`, `tiered_offset_m4_allperiods_paperlr_results.csv`, `m4_hourly_sym10_tiered_offset_paperlr_results.csv`. Paper-sample (plateau LR): `comprehensive_m4_paper_sample_plateau_results.csv`, `tiered_offset_m4_allperiods_results.csv`, `m4_hourly_sym10_tiered_offset_results.csv`. Sliding: `comprehensive_sweep_m4_results.csv`.

**Canonical analysis.** Meta-leaderboard at `experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md`, regenerable from `experiments/analysis/scripts/m4_overall_leaderboard.py`. The strict health filter `diverged $\lor$ smape$>$100 $\lor$ (best_epoch$=$0 $\land$ smape$>$50) $\lor$ smape NaN` is applied to every cell.

**Canonical YAML configs (`experiments/configs/`).** Paper-sample faithful: `comprehensive_m4_paper_sample.yaml`. Paper-sample plateau: `comprehensive_m4_paper_sample_plateau.yaml`. Sliding: `comprehensive_sweep_m4.yaml`. Tiered: `tiered_offset_m4_allperiods.yaml` (and `m4_hourly_sym10_tiered_offset.yaml` for the dedicated Hourly file).

## Appendix C: Configuration list \label{app:configs}

## Appendix D: Additional ablations \label{app:ablations}

## Appendix E: M4 full per-period tables \label{app:m4_tables}
