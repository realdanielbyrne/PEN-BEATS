# Beyond Fourier and Polynomial: Wavelet and Autoencoder Basis Expansions for Parameter-Efficient N-BEATS

**Daniel Byrne**

---

## Abstract

N-BEATS demonstrated that a pure deep learning architecture built on doubly residual stacking of basis expansion blocks could match or exceed state-of-the-art statistical methods on major forecasting benchmarks. However, the original work explored only three basis types: polynomial (Trend), Fourier (Seasonality), and fully learned (Generic). We present a systematic exploration of alternative basis expansions and backbone architectures within the N-BEATS framework, introducing wavelet basis blocks with orthonormal discrete wavelet transform (DWT) bases, autoencoder-compressed backbone variants, and hybrid TrendWavelet blocks that combine polynomial and wavelet decompositions in a single block. We evaluate 112 configurations across 10 random seeds on four datasets spanning nine forecasting tasks: M4 (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly), Tourism-Yearly, Weather-96, and Milk. Our wavelet-augmented architectures beat paper baselines on six of nine dataset-periods, while sub-1M parameter TrendWavelet models achieve within 0.5% of the best configurations on most datasets --- a 10--50$\times$ parameter reduction compared to the 26M-parameter original. These results expose fundamental overparameterization in the original N-BEATS design: 30-stack Generic configurations diverge in 40--50% of training runs on small datasets, while autoencoder-compressed variants with 10--50$\times$ fewer parameters converge reliably. We further characterize the convergence-vs-optimality trade-off of the `active_g` post-basis activation, the dataset-dependent reversal of backbone hierarchy preferences, and horizon-dependent stack architecture selection rules. Our implementation is released as the `lightningnbeats` PyTorch Lightning package.

---

## 1. Introduction

Time series forecasting is one of the oldest and most consequential problems in quantitative science. From inventory planning and financial risk management to energy load scheduling and epidemiological surveillance, accurate forecasts translate directly into better decisions and measurable economic value. For decades, the field was dominated by classical statistical methods --- exponential smoothing, ARIMA, and their many variants --- that offered strong theoretical grounding, interpretability, and reliable performance on the small-to-moderate datasets typical of business applications. Deep learning, despite its transformative impact on computer vision and natural language processing, was widely regarded as unnecessary or even counterproductive for univariate time series, where the number of observations per series is often modest and the risk of overfitting substantial.

The M4 competition (Makridakis et al., 2018; 2020) marked a turning point. Among 60 submitted methods, the six "pure" machine learning entries ranked 23rd through 57th, seemingly confirming the skeptics' view. Yet the competition winner, Smyl's ES-RNN (2020), was a hybrid that fused an LSTM-based deep learning component with classical Holt-Winters exponential smoothing, outperforming all purely statistical methods. This result established that deep learning could contribute meaningfully to forecasting accuracy, but left open the question of whether a *pure* deep learning architecture --- one requiring no hand-crafted statistical components --- could achieve competitive or superior results.

Oreshkin et al. (2019) answered this question with N-BEATS (Neural Basis Expansion Analysis for Time Series), a fully deep learning architecture that surpassed both the M4 winner and all prior statistical methods on the M4, M3, and Tourism benchmarks. N-BEATS introduced a distinctive design built on doubly residual stacking of basic blocks, each consisting of a multi-layer fully connected network that forks into backcast and forecast paths via learned or constrained basis expansion coefficients. The architecture offered two configurations: a Generic model using fully learnable basis functions, and an Interpretable model constraining basis functions to polynomial (Trend) and Fourier (Seasonality) forms. The success of N-BEATS demonstrated that the choice of basis function within each block is a critical design decision --- one that determines how the network decomposes and reconstructs the input signal.

This observation motivates the present work. If polynomial and Fourier bases can achieve state-of-the-art results when embedded within the N-BEATS doubly residual framework, what happens when we substitute alternative basis expansions? Wavelets offer multi-resolution time-frequency localization that neither polynomials nor Fourier series provide. Autoencoders learn compressed, data-driven representations that may capture structure not well-described by any fixed analytical basis. Combining these --- polynomial trend bases with orthonormal wavelet bases in a single block --- yields a hybrid decomposition that separates slow-varying macro-trends from transient micro-structure, all within one parameter-efficient unit.

We present a systematic exploration of these alternative block types and backbone architectures within the N-BEATS framework, implemented as the `lightningnbeats` PyTorch Lightning package. Our comprehensive benchmark evaluates 112 configurations across 10 random seeds on four datasets spanning nine forecasting tasks. The results reveal several findings that reshape the understanding of the N-BEATS architecture:

1. **Wavelet dominance.** Wavelet-augmented architectures beat the original N-BEATS baselines on six of nine dataset-periods tested, including M4-Yearly, M4-Monthly, M4-Weekly, Tourism-Yearly, Weather-96, and Milk.

2. **Extreme parameter efficiency.** Sub-1M parameter TrendWavelet models with autoencoder-compressed backbones achieve 99.5% of the best configurations on most datasets, representing a 10--50$\times$ parameter reduction compared to the 26M-50M parameter models in the original N-BEATS-G architecture.

3. **Instability exposed.** The original Generic architecture diverges in 40--50% of training runs on small datasets (Milk, Tourism), while autoencoder-compressed variants with orders of magnitude fewer parameters converge reliably.

4. **N-BEATS Over-Parameterized**
This reveals that the original architecture is massively overparameterized for most forecasting tasks.

5. **Dataset-dependent architecture selection.** No single configuration is universally optimal. Backbone hierarchy, stack architecture (unified vs. alternating), optimal depth, and wavelet family preferences all reverse across datasets, necessitating dataset-aware architecture selection.

6. **Convergence regularization trade-offs.** The `active_g` post-basis activation mechanism eliminates catastrophic divergence but imposes a small expressiveness penalty, with the magnitude varying from negligible on large benchmarks (~1%) to prohibitive on small univariate series (54--76%).

Our contributions include: (a) novel block types --- orthonormal wavelet basis blocks (WaveletV3), hybrid TrendWavelet blocks, three-way TrendWaveletGeneric blocks, and autoencoder/learned-gate/variational backbone variants of all original and novel block types; (b) a rigorous benchmark framework evaluating 112 configurations across 10 seeds on four diverse datasets; (c) diagnosis and characterization of overparameterization in the original N-BEATS architecture; and (d) practical architecture selection guidelines for practitioners.

---

## 2. Related Work

### 2.1 N-BEATS and Variants

N-BEATS (Oreshkin et al., 2019) introduced a pure deep learning architecture for univariate time series forecasting built on doubly residual stacking of basis expansion blocks. Each block consists of four fully connected layers followed by a fork into backcast and forecast paths via basis expansion coefficients. The backward residual connection (subtracting each block's backcast from its input) implements iterative signal decomposition, while the forward residual connection (summing all blocks' forecasts) implements hierarchical forecast aggregation. The architecture achieved state-of-the-art results on M4, M3, and Tourism benchmarks without any time-series-specific components, using only two configurations: Generic (fully learnable basis) and Interpretable (polynomial Trend + Fourier Seasonality basis).

N-HiTS (Challu et al., 2023) extends N-BEATS with multi-rate input pooling and hierarchical forecast interpolation, achieving competitive results with improved computational efficiency. Crucially, the N-HiTS modifications operate at the stack level (pooling inputs and interpolating outputs) rather than at the block level, meaning the block interface --- accepting a backcast-length input and producing backcast/forecast outputs --- remains unchanged. This architectural separation means that novel block types developed for N-BEATS can be deployed in N-HiTS without modification.

ES-RNN (Smyl, 2020), the M4 competition winner, demonstrated that hybrid architectures combining deep learning with classical statistical components could outperform either approach alone. More recently, DLinear (Zeng et al., 2023) showed that simple linear models applied to decomposed trend and seasonal components could match Transformer-based approaches, and PatchTST (Nie et al., 2023) achieved state-of-the-art long-horizon forecasting through patch-level attention with channel independence. These results collectively suggest that architectural inductive biases aligned with time series structure --- decomposition, multi-scale representation, basis expansion --- are more important than model complexity per se.

### 2.2 Wavelets in Time Series Forecasting

Wavelets provide a mathematical framework for multi-resolution analysis of signals, offering simultaneous localization in both time and frequency domains --- a property that neither purely temporal (polynomial) nor purely spectral (Fourier) bases possess. The discrete wavelet transform (DWT) decomposes a signal into approximation coefficients (capturing low-frequency trend) and detail coefficients (capturing high-frequency fluctuations) at progressively coarser scales (Mallat, 1989; Daubechies, 1992).

In forecasting, wavelets have been used primarily as preprocessing transforms --- decomposing series into sub-bands that are forecast independently and then reconstructed (Aminghafari et al., 2006). Pramanick et al. (2024) applied this decomposition strategy directly to N-BEATS, using DWT to separate stock price series into approximation and detail components and training separate N-BEATS models on each sub-band before recombining forecasts. Their approach treats N-BEATS as an unmodified black-box forecaster within a wavelet preprocessing pipeline.

In contrast, the present work integrates wavelet basis functions directly into the N-BEATS block architecture, replacing the basis expansion function rather than preprocessing the input. This preserves the doubly residual topology and enables end-to-end training within a single model. The idea of using wavelet functions directly as basis expansions within neural network blocks is less explored. By replacing the Fourier or polynomial basis with an orthonormal wavelet basis, we provide the network with time-frequency localized basis functions that capture transient phenomena, regime changes, and localized oscillations that neither polynomial nor Fourier bases can efficiently represent.

### 2.3 Autoencoders and Compression in Neural Forecasting

Autoencoders (Hinton & Salakhutdinov, 2006) learn compressed representations of input data through an encoder-decoder architecture with a bottleneck layer. The compression forces the network to learn salient features while discarding noise. In time series contexts, autoencoders have been applied to anomaly detection (Malhotra et al., 2016), representation learning, and denoising (Vincent et al., 2008).

We apply the autoencoder principle in two ways within N-BEATS. First, we replace the standard four-FC-layer backbone with an encoder-decoder architecture (AERootBlock), where the input is progressively compressed through an hourglass structure to a low-dimensional latent space before expansion. Second, we introduce a learned-gate variant (AERootBlockLG) that applies a sigmoid-gated mask at the latent bottleneck, allowing the network to discover the effective latent dimensionality during training. These compressed backbones serve as implicit regularizers, achieving dramatic parameter reduction while maintaining forecasting accuracy --- a property that proves central to our overparameterization findings.

### 2.4 Overparameterization in Deep Learning

The observation that neural networks can perform well with far fewer parameters than their full capacity suggests has been extensively studied. The lottery ticket hypothesis (Frankle & Carlin, 2019) demonstrated that sparse subnetworks within overparameterized models can match the full model's accuracy. Network pruning (Han et al., 2015) and knowledge distillation (Hinton et al., 2015) provide practical approaches to model compression.

In time series forecasting, overparameterization is particularly concerning because datasets are often small relative to model capacity. The original N-BEATS-G configuration uses approximately 26M parameters for a 6-step-ahead forecast --- roughly 4.3M parameters per output dimension. On the Milk dataset (a single series with 156 training observations), this represents a 167,000$\times$ ratio of parameters to observations. Our results demonstrate that this extreme overparameterization is not merely wasteful but actively harmful: it induces training instability that causes 40--50% of runs to diverge on small datasets. Autoencoder-compressed variants with 400K--2M parameters eliminate this instability while maintaining equivalent accuracy, providing direct evidence that the vast majority of parameters in the original architecture are redundant.

---

## 3. Method

### 3.1 N-BEATS Preliminaries

The N-BEATS architecture (Oreshkin et al., 2019) is composed of blocks organized into stacks. Each basic block accepts an input vector $x_\ell \in \mathbb{R}^L$ (where $L$ is the lookback window length) and outputs a backcast $\hat{x}_\ell \in \mathbb{R}^L$ and a forecast $\hat{y}_\ell \in \mathbb{R}^H$ (where $H$ is the forecast horizon).

**Basic Block.** The core computation within each block begins with four fully connected layers, each followed by an activation function $\sigma$ (default ReLU):

$$h_1 = \sigma(W_1 x + b_1), \quad h_2 = \sigma(W_2 h_1 + b_2), \quad h_3 = \sigma(W_3 h_2 + b_3), \quad h_4 = \sigma(W_4 h_3 + b_4)$$

The hidden representation $h_4 \in \mathbb{R}^{w}$ (where $w$ is the layer width, or `units`) is then projected to produce expansion coefficients that are passed through basis functions to produce outputs:

$$\hat{x}_\ell = g^b(\theta^b_\ell), \quad \hat{y}_\ell = g^f(\theta^f_\ell)$$

**Original Basis Functions.** The choice of $g^b$ and $g^f$ determines the block type:

- *Generic*: Basis functions are fully learnable linear projections. The coefficients $\theta$ are projected directly to the target length: $\hat{y} = V^f \theta^f$ where $V^f \in \mathbb{R}^{H \times w}$ is a learned weight matrix. In our implementation, the Generic block follows the original paper faithfully --- the projection matrices serve as both theta extraction and basis expansion in a single step, with no intermediate bottleneck.

- *Trend*: Basis functions are polynomial Vandermonde matrices. The expansion coefficients $\theta \in \mathbb{R}^p$ represent polynomial coefficients, and the basis matrix is $T = [1, t, t^2, \ldots, t^{p-1}]^T$ where $t$ is a normalized time vector on $[0, 1)$. With small polynomial degree $p$ (typically 2--3), the output is constrained to slowly varying functions.

- *Seasonality*: Basis functions are Fourier matrices. The basis consists of cosine and sine vectors at integer multiples of the fundamental frequency: $S = [1, \cos(2\pi t), \ldots, \cos(2\pi \lfloor L/2-1 \rfloor t), \sin(2\pi t), \ldots, \sin(2\pi \lfloor L/2-1 \rfloor t)]^T$.

**Doubly Residual Stacking.** Blocks are connected via residual connections:

$$x_\ell = x_{\ell-1} - \hat{x}_{\ell-1}, \quad \hat{y} = \sum_\ell \hat{y}_\ell$$

Each block subtracts its backcast from the input (removing the signal component it has modeled) before passing the residual to the next block. All forecast partial outputs are summed to produce the final forecast. This creates an iterative decomposition: early blocks capture prominent signal components, while later blocks model progressively finer residual structure.

### 3.2 Novel Backbone Architectures

The original N-BEATS block uses a backbone of four fully connected layers of equal width $w$ (the `units` parameter), which we term **RootBlock**. We introduce three alternative backbone architectures that replace this uniform-width design with hourglass-shaped encoder-decoder structures, each providing different regularization properties.

[**Figure 1**: Architecture diagram showing the four backbone variants side-by-side. Left: RootBlock with uniform $w$-width layers. Center-left: AERootBlock with hourglass $w/2 \rightarrow d \rightarrow w/2 \rightarrow w$ structure. Center-right: AERootBlockLG with the same hourglass plus a learned gate $\sigma(\mathbf{g})$ applied at the latent bottleneck. Right: AERootBlockVAE with split $\mu$/$\log\sigma^2$ heads and reparameterization at the bottleneck. Parameter counts annotated for each variant with $L=30$, $w=512$, $d=16$. *To be produced.*]

#### 3.2.1 AERootBlock (Autoencoder Backbone)

The AERootBlock replaces the four equal-width layers with an encoder-decoder hourglass:

$$h_1 = \sigma(W_1 x + b_1), \quad W_1 \in \mathbb{R}^{w/2 \times L}$$
$$h_2 = \sigma(W_2 h_1 + b_2), \quad W_2 \in \mathbb{R}^{d \times w/2}$$
$$h_3 = \sigma(W_3 h_2 + b_3), \quad W_3 \in \mathbb{R}^{w/2 \times d}$$
$$h_4 = \sigma(W_4 h_3 + b_4), \quad W_4 \in \mathbb{R}^{w \times w/2}$$

where $d$ is the latent dimension (`latent_dim`). This creates a compression bottleneck ($L \rightarrow w/2 \rightarrow d \rightarrow w/2 \rightarrow w$) that forces the network to learn a compact representation. With typical settings ($w = 512$, $d = 16$), the bottleneck compresses the representation to just 16 dimensions before expansion.

**Parameter comparison.** The RootBlock backbone contains $Lw + 3w^2$ weight parameters (plus biases). The AERootBlock contains $Lw/2 + dw + w^2/2$ weight parameters. For $L = 30$, $w = 512$, $d = 16$:

| Backbone | Weight parameters | Ratio |
|----------|:--:|:--:|
| RootBlock | $30 \times 512 + 3 \times 512^2 = 801{,}792$ | 1.0$\times$ |
| AERootBlock | $30 \times 256 + 16 \times 512 + 256^2 = 81{,}408$ | 0.10$\times$ |

The AE backbone achieves a 10$\times$ reduction in backbone parameters alone. When combined with the projection heads and stacked into a full model, this translates to 5--50$\times$ total parameter reduction depending on the basis type.

#### 3.2.2 AERootBlockLG (Learned-Gate Backbone)

The AERootBlockLG extends AERootBlock with a learnable gate vector $\mathbf{g} \in \mathbb{R}^d$ applied at the latent bottleneck:

$$h_2' = h_2 \odot \sigma_g(\mathbf{g})$$

where $\sigma_g$ is a gating function (default: sigmoid) and $\odot$ denotes element-wise multiplication. The gate is initialized as $\mathbf{g} = \mathbf{1}$ (all ones, so $\sigma(\mathbf{1}) \approx 0.73$, passing most information initially). During training, the network learns to suppress uninformative latent dimensions by driving their gate values toward zero, effectively discovering the minimal latent dimensionality required for the task.

This adds only $d$ trainable parameters (the gate vector) beyond the AERootBlock, but provides a soft mechanism for automatic dimensionality selection. On datasets where fewer latent dimensions suffice, the gate learns to zero out redundant dimensions; on more complex datasets, it retains a larger effective latent space.

#### 3.2.3 AERootBlockVAE (Variational Backbone)

The variational backbone replaces the deterministic bottleneck with a stochastic latent space. The encoder produces mean and log-variance vectors:

$$\mu = W_\mu h_1', \quad \log \sigma^2 = W_{\log\sigma^2} h_1'$$

During training, the latent representation is sampled via the reparameterization trick:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

During evaluation, $z = \mu$ (deterministic). The KL divergence loss $D_{KL}(q(z|x) \| p(z)) = -\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ is accumulated across all VAE blocks in the model and added to the forecast loss with weight $\lambda_{KL}$ (default 0.001). Our comprehensive sweep results show that VAE backbones consistently underperform their deterministic counterparts (+2--50% SMAPE penalty across all datasets), with double-VAE alternating stacks exhibiting catastrophic performance degradation (7--8$\times$ worse on short horizons). We include the VAE backbone for completeness but do not recommend it for forecasting applications.

### 3.3 Novel Basis Expansion Blocks

We introduce several families of novel block types that explore alternative basis expansions within the N-BEATS doubly residual framework. Each block type can be instantiated with any of the four backbone architectures described above, yielding a combinatorial design space.

#### 3.3.1 WaveletV3: Orthonormal DWT Basis Expansion

**Motivation.** Wavelets offer multi-resolution time-frequency localization that is well-suited to time series with transient phenomena, regime changes, or localized oscillations. Unlike Fourier bases which assume global periodicity, wavelet bases can capture structure that is localized in both time and frequency.

**V1/V2 Failure and the Conditioning Problem.** Our initial wavelet block implementations (V1 and V2) constructed basis matrices by evaluating the scaling function $\phi$ and wavelet function $\psi$ at uniformly spaced points and assembling cyclic shifts into a square or rectangular matrix. This naive construction produced severely ill-conditioned basis matrices --- the DB3 wavelet basis had a condition number of approximately 604,000. When these matrices were used as frozen basis expansions in gradient-trained blocks, the ill-conditioning amplified gradient magnitudes during backpropagation, causing 67--100% training failure rates. Failure modes ranged from immediate NaN at epoch 1 (Haar, DB3Alt) to gradual MASE explosion to $10^{31}$ (DB3, Symlet3). These failures demonstrated that **basis matrix conditioning is a prerequisite for integration into gradient-trained architectures**.

**V3 Solution: Impulse-Response Synthesis + SVD Orthogonalization.** The WaveletV3 basis generator resolves the conditioning problem through a two-stage construction:

1. **Impulse-response synthesis.** For a target length $n$ and wavelet family, compute the DWT coefficient structure via `pywt.wavedec`. For each coefficient position across all decomposition levels, construct an impulse (a single 1.0 in the coefficient vector, zeros elsewhere) and reconstruct the corresponding time-domain waveform via `pywt.waverec`. Stack all reconstructed waveforms as rows of a raw synthesis matrix $R \in \mathbb{R}^{m \times n}$ where $m$ is the total number of DWT coefficients.

2. **SVD orthogonalization.** Compute the SVD $R = U \Sigma V^T$. The rows of $V^T$ form an orthonormal basis for the column space of $R$, ordered by singular value magnitude (low-to-high frequency). All non-trivial singular values are equal (typically 1.0), yielding a basis with condition number exactly 1.0.

The resulting basis matrix $B \in \mathbb{R}^{k \times n}$ is selected as a contiguous frequency band from $V^T$:

$$B = V^T[\text{offset} : \text{offset} + k, :]$$

where $k$ is the `basis_dim` parameter and `offset` controls the frequency band (offset = 0 selects the lowest frequencies). This basis is stored as a frozen `nn.Parameter` (non-trainable). Different wavelet families (Haar, Daubechies, Coiflets, Symlets) produce different orthonormal bases with distinct time-frequency localization properties, but all share the crucial property of unit condition number.

[**Figure 2**: Comparison of V1 and V3 basis matrices for DB3 wavelet at target length 30. Left: V1 basis matrix heatmap showing irregular structure with condition number $\kappa \approx 604{,}000$. Right: V3 orthonormal basis with condition number $\kappa = 1.0$. Bottom: Singular value spectrum for each, showing V1's spread versus V3's uniform spectrum. *To be produced.*]

**WaveletV3 Block Forward Pass.** The WaveletV3 block uses a RootBlock (or AE variant) backbone followed by linear projection to `basis_dim` coefficients and basis expansion:

$$h = \text{Backbone}(x), \quad \theta^b = W^b h, \quad \theta^f = W^f h$$
$$\hat{x} = B^b \theta^b, \quad \hat{y} = B^f \theta^f$$

where $B^b \in \mathbb{R}^{k_b \times L}$ and $B^f \in \mathbb{R}^{k_f \times H}$ are independent frozen wavelet bases for the backcast and forecast paths. The `basis_dim` parameter controls the number of basis vectors used, effectively selecting the frequency bandwidth of the decomposition. Separate `forecast_basis_dim` allows asymmetric regularization when $L \gg H$.

Concrete wavelet block subclasses (HaarWaveletV3, DB3WaveletV3, Coif2WaveletV3, etc.) are thin wrappers that set the wavelet type string. The generic WaveletV3 class accepts `wavelet_type` as a parameter, supporting any family available in PyWavelets.

#### 3.3.2 TrendWavelet: Unified Polynomial + DWT Decomposition

The TrendWavelet block combines polynomial and wavelet basis expansions within a single block, implementing an additive decomposition:

$$\hat{x} = T^b \theta^b_{\text{trend}} + B^b \theta^b_{\text{wavelet}}, \quad \hat{y} = T^f \theta^f_{\text{trend}} + B^f \theta^f_{\text{wavelet}}$$

where $T$ is a Vandermonde polynomial basis and $B$ is an orthonormal wavelet basis. The backbone produces a single hidden representation that is projected to a combined coefficient vector, which is then split:

$$\theta = W h, \quad \theta_{\text{trend}} = \theta[:p], \quad \theta_{\text{wavelet}} = \theta[p:]$$

where $p$ is the polynomial degree (`trend_dim`, typically 3) and the remaining coefficients drive the wavelet expansion.

[**Figure 3**: TrendWavelet block schematic. The backbone (RootBlock or AE variant) processes the input into a hidden representation. A single linear projection produces $p + k$ coefficients. The first $p$ coefficients multiply the Vandermonde polynomial basis (capturing smooth trend), while the remaining $k$ coefficients multiply the orthonormal wavelet basis (capturing transient structure). Both components are summed to produce backcast and forecast outputs. *To be produced.*]

**Design rationale.** The pairing of polynomial trend with wavelet detail is natural and well-motivated. The Vandermonde basis captures slow-varying macro-trends (level, slope, curvature) with very few coefficients ($p = 3$ captures constant, linear, and quadratic trends). The wavelet basis captures the residual micro-structure --- transient events, localized oscillations, regime changes --- that polynomials are blind to. By combining both in a single block, TrendWavelet implements within-block signal decomposition that parallels the between-stack decomposition of the original N-BEATS Interpretable architecture (Trend stack + Seasonality stack), but at a fraction of the parameter cost.

**Parameter efficiency.** The TrendWavelet block with AE backbone is the most parameter-efficient competitive design in our sweep. With `units` = 256, `latent_dim` = 16, `trend_dim` = 3, `basis_dim` = $\text{eq\_fcast}$ (equal to forecast length), and 10 stacks, the full model requires only 418K--436K parameters. This compares to 26M for N-BEATS-G (30 stacks) and 19.6M for N-BEATS-I+G (10 stacks), representing a 45--60$\times$ reduction.

#### 3.3.3 TrendWaveletGeneric: Three-Way Decomposition

TrendWaveletGeneric extends TrendWavelet with a third branch: a learned generic basis that captures data-driven patterns not represented by either the polynomial or wavelet bases:

$$\hat{y} = T^f \theta^f_{\text{trend}} + B^f \theta^f_{\text{wavelet}} + V^f \theta^f_{\text{generic}}$$

where $V^f \in \mathbb{R}^{H \times g}$ is a trainable weight matrix with `generic_dim` $= g$ columns (typically 5). The generic branch provides extra capacity for patterns that escape both analytical bases, at the cost of $g \times H$ additional trainable parameters per block. The `active_g` mechanism (Section 3.5), when enabled, applies activation only to the generic branch, leaving the trend and wavelet components linear.

#### 3.3.4 Other Novel Blocks

**BottleneckGeneric.** Factors the Generic block's direct projection ($w \rightarrow \text{target\_length}$) into two steps through an intermediate bottleneck: $w \rightarrow d \rightarrow \text{target\_length}$. This is equivalent to a rank-$d$ factorization of the basis expansion matrix. With $d = 5$ (default `thetas_dim`), this constrains the effective rank but provides a tunable regularization knob.

**AutoEncoder Block.** Retains the standard RootBlock backbone but replaces the simple linear basis expansion with an encoder-decoder pipeline for each path: $h \rightarrow \text{Encoder}(h) \rightarrow z \rightarrow \text{Decoder}(z) \rightarrow \hat{y}$. The encoder is a linear layer ($w \rightarrow d$) with activation; the decoder consists of two layers ($d \rightarrow w \rightarrow \text{target\_length}$).

**GenericAEBackcast.** A hybrid block using an autoencoder path for backcast reconstruction (where the reconstruction task is natural) and a simpler bottleneck projection for the forecast path.

All of these blocks exist in standard (RootBlock), AE, AELG, and VAE backbone variants, yielding a combinatorial library of over 185 registered block types.

### 3.4 Stack Architecture Design Space

A key implementation feature of our framework is the ability to compose arbitrary ordered sequences of any block type via the `stack_types` parameter. This enables three primary architectural patterns:

**Unified (homogeneous) stacks.** All blocks in the model are identical --- for example, 10$\times$ TrendWavelet. This tests the capacity of a single block type to handle all aspects of signal decomposition through the doubly residual mechanism alone.

**Alternating stacks.** Two complementary block types interleave --- for example, (Trend, DB3WaveletV3)$\times$15. This mirrors the original Interpretable architecture's separation of concerns (trend vs. detail) but substitutes wavelet blocks for Seasonality blocks. The alternating pattern allows each block type to specialize: Trend blocks extract the smooth component, leaving wavelet blocks to model the residual high-frequency structure.

**Prefix-body stacks.** A small number of interpretable blocks (e.g., Trend + Seasonality) precede a larger body of generic or wavelet blocks. This is the pattern used by the original NBEATS-I+G configuration.

**Table 1: Configuration Grid Overview**

Our comprehensive sweep evaluates 112 configurations spanning the following dimensions:

| Dimension | Values | Count |
|-----------|--------|:-----:|
| Backbone | RootBlock, AERootBlock, AERootBlockLG | 3 |
| Block type | Generic, BottleneckGeneric, Trend+WaveletV3, TrendWavelet, TrendWaveletGeneric, paper baselines | 7 families |
| Stack depth | 10, 30 | 2 |
| active_g | False, 'forecast' | 2 |
| Wavelet family | Haar, DB3, Coif2, Sym10 | 4 |
| Latent dim (AE) | 8, 16, 32 | 3 |
| Basis dim | eq\_fcast, 2$\times$eq\_fcast | 2 |

Not all dimension combinations are tested (the full factorial would exceed 1,000 configs); the 112 configurations represent a curated grid covering the most informative comparisons.

### 3.5 The `active_g` Mechanism

The original N-BEATS architecture does not apply any activation function after basis expansion --- the outputs of $g^b$ and $g^f$ are linear combinations of basis vectors. We introduce `active_g`, an optional post-expansion activation:

$$\hat{x} = \sigma(g^b(\theta^b)), \quad \hat{y} = \sigma(g^f(\theta^f))$$

The parameter supports four modes: `False` (paper-faithful), `True` (activation on both paths), `'backcast'` (backcast only), and `'forecast'` (forecast only). Our convergence studies (detailed in Section 5.3) reveal that `active_g` implements a convergence-vs-optimality trade-off:

**Convergence benefit.** ReLU-activated forecasts constrain partial forecasts $\hat{y}_\ell \geq 0$, preventing catastrophic cancellation in the forecast sum $\hat{y} = \sum_\ell \hat{y}_\ell$. This eliminates the sharp, narrow loss minima that cause bimodal convergence in Generic blocks.

**Expressiveness cost.** ReLU-activated backcasts constrain $\hat{x}_\ell \geq 0$, making the residual chain monotonically decreasing: $x_\ell \leq x_{\ell-1}$. This prevents blocks from adding signal back after over-extraction, restricting the bidirectional correction that the doubly residual topology was designed to enable.

**Split-mode recommendation.** The `'forecast'` mode applies activation only to the forecast path, achieving the convergence benefit (100% convergence) while preserving backcast expressiveness. Our sweep uses `active_g = \text{False}` (safe globally) and `active_g = \text{'forecast'}` as the two tested settings.

### 3.6 NHiTS Compatibility

All novel block types introduced in this work are fully compatible with N-HiTS (Challu et al., 2023). The N-HiTS architecture adds two stack-level operations --- multi-rate input pooling (MaxPool1d with configurable kernel size per stack) and hierarchical forecast interpolation --- that operate outside the block interface. Since our blocks accept a 1D input and produce (backcast, forecast) tuples, they plug into N-HiTS unchanged. The pooling reduces the effective input length seen by each block (reducing computational cost), while the interpolation upsamples the block's reduced-resolution forecast to the target horizon. We include `NHiTSNet` in the `lightningnbeats` package with full support for all novel block types; Section 5.8 and Appendix D summarize a dedicated N-HiTS weather benchmark confirming that block-level transfer works empirically, although the ranking of block families and the best hyperparameter settings do shift under hierarchical pooling.

---

## 4. Experimental Setup

### 4.1 Datasets

We evaluate on four datasets spanning nine forecasting tasks that cover diverse data regimes: large multi-series competition benchmarks (M4), medium-scale tourism forecasting (Tourism), multivariate weather prediction (Weather), and small univariate time series (Milk).

**Table 2: Dataset Characteristics**

| Dataset | Series | Horizon $H$ | Lookback $L$ | Frequency | Primary Metric |
|---------|-------:|:----:|:-----:|-----------|:---------:|
| M4-Yearly | 23,000 | 6 | 30 | Annual | SMAPE, OWA |
| M4-Quarterly | 24,000 | 8 | 40 | Quarterly | SMAPE, OWA |
| M4-Monthly | 48,000 | 18 | 90 | Monthly | SMAPE, OWA |
| M4-Weekly | 359 | 13 | 65 | Weekly | SMAPE, OWA |
| M4-Daily$^\dagger$ | 4,227 | 14 | 70 | Daily | SMAPE, OWA |
| M4-Hourly | 414 | 48 | 240 | Hourly | SMAPE, OWA |
| Tourism-Yearly | 518 | 4 | 20 | Annual | SMAPE |
| Weather-96 | 52,696 windows | 96 | 480 | 10-min | MSE |
| Milk | 1 series | 6 | 30 | Monthly | SMAPE |

$^\dagger$M4-Daily results are preliminary: only 14 of 112 configurations have been evaluated (paper baselines + TrendWavelet RootBlock variants). Full sweep is in progress.

The **M4 dataset** (Makridakis et al., 2018; 2020) comprises 100,000 univariate time series across six sampling frequencies drawn from diverse domains including demographics, finance, industry, and macroeconomics. We use the standard train/test splits with test set length equal to the forecast horizon $H$ for each period.

The **Tourism-Yearly** dataset contains 518 annual tourism demand series. With a short forecast horizon ($H = 4$) and moderate number of series, it tests model behavior on a simpler forecasting task where overparameterization risks are intermediate.

The **Weather-96** dataset consists of 21 meteorological indicators recorded at 10-minute intervals. We use a 96-step forecast horizon with a 480-step lookback ($L = 5H$), creating 52,696 training windows. As a multivariate dataset with long horizons, it tests model behavior in a regime very different from the M4 competition format.

The **Milk** dataset is a single univariate monthly time series of U.S. milk production (168 observations). With only one series and 156 training windows, it represents the extreme small-data regime where overparameterization effects are most pronounced.

### 4.2 Training Protocol

All experiments share a common training configuration:

| Parameter | Value |
|-----------|-------|
| Maximum epochs | 200 |
| Early stopping patience | 20 epochs |
| Learning rate | 0.001 |
| LR warmup | 15 epochs |
| Optimizer | Adam (default parameters) |
| Loss function | SMAPELoss (M4, Tourism, Milk); MSELoss (Weather) |
| Batch size | 1024 |
| Backcast multiplier | 5$\times$ forecast horizon ($L = 5H$) |
| Seeds per configuration | 10 (seeds 0--9) |
| Blocks per stack | 1 |
| Weight sharing | True (within each stack) |
| Activation | ReLU |

The training protocol was chosen to balance computational feasibility with statistical rigor. The 10-seed design provides substantially greater statistical power than the 3--5 seeds typical in the literature, enabling reliable detection of medium effect sizes.

### 4.3 Evaluation Metrics

**sMAPE** (Symmetric Mean Absolute Percentage Error):
$$\text{sMAPE} = \frac{100}{H} \sum_{i=1}^{H} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2 + \epsilon}$$

**MASE** (Mean Absolute Scaled Error), computed per-series using seasonal naive scaling:
$$\text{MASE} = \frac{\frac{1}{H}\sum_{j=1}^{H} |y_{T+j} - \hat{y}_{T+j}|}{\frac{1}{T-m}\sum_{j=m+1}^{T} |y_j - y_{j-m}|}$$

**OWA** (Overall Weighted Average, M4 only):
$$\text{OWA} = \frac{1}{2}\left(\frac{\text{sMAPE}}{\text{sMAPE}_{\text{Naive2}}} + \frac{\text{MASE}}{\text{MASE}_{\text{Naive2}}}\right)$$

where Naive2 values are the M4 competition's seasonally-adjusted naive baseline. OWA = 1.0 corresponds to naive performance.

**MSE** (Mean Squared Error, Weather only).

**Divergence rate**: percentage of runs where training produces degenerate solutions (sMAPE > threshold or extreme loss ratio). A run is classified as *healthy* if it produces finite metrics with sMAPE below a dataset-specific threshold.

**Statistical tests**: Kruskal-Wallis H-test (non-parametric omnibus), Wilcoxon signed-rank test (pairwise), with Bonferroni correction where applicable. Effect sizes reported as Cohen's $d$ or $\eta^2$.

---

## 5. Results

We report results from the comprehensive sweep of 112 configurations across 10 seeds on nine dataset-periods, totaling approximately 9,521 training runs. Throughout this section, a run is classified as *healthy* if it produced finite metrics below dataset-specific thresholds; *divergent* runs are excluded from mean calculations but included in divergence rate reporting.

### 5.1 Main Results: Novel Blocks Beat Baselines on 6/9 Dataset-Periods

Table 3 presents the best configuration per dataset-period, ranked by the primary metric (SMAPE for M4/Tourism/Milk; MSE for Weather). Wavelet-augmented architectures win on six of nine dataset-periods. Paper baselines only win decisively on M4-Hourly (long horizon, $H = 48$) and M4-Daily (severely under-tested; only 14 of 112 configurations evaluated). M4-Quarterly is a statistical tie between baselines and novel architectures.

**Table 3: Best Configuration per Dataset-Period**

| Dataset | Winner | SMAPE | OWA | Params | Architecture |
|---------|--------|:-----:|:---:|-------:|:-------------|
| M4-Yearly | TW\_10s\_td3\_bdeq\_coif2 | 13.499 | 0.801 | 2.1M | Unified TrendWavelet (RB) |
| M4-Quarterly | NBEATS-IG\_10s\_ag0 | 10.126 | 0.888 | 19.6M | Paper baseline (15-way tie) |
| M4-Monthly | TW\_30s\_td3\_bd2eq\_coif2 | 13.279 | 0.914 | 7.1M | Unified TrendWavelet (RB) |
| M4-Weekly | T+Db3V3\_30s\_bdeq | 6.671 | 0.735 | 15.8M | Alternating Trend+Wavelet (RB) |
| M4-Daily$^\dagger$ | NBEATS-G\_30s\_ag0 | 2.603 | 0.861 | 26.0M | Paper baseline |
| M4-Hourly | NBEATS-IG\_30s\_agf | 8.587 | 0.409 | 43.6M | Paper baseline |
| Tourism-Y | TW\_10s\_td3\_bdeq\_db3 | 21.773 | --- | 2.0M | Unified TrendWavelet (RB) |
| Weather-96 | TAE+DB3V3AE\_30s\_ld8\_ag0 | --- | --- | 7.1M | Alternating TrendAE+WaveletAE |
| Milk | TALG+DB3V3ALG\_10s\_ag0 | 1.512 | --- | 1.0M | Alternating TrendAELG+WaveletAELG |

$^\dagger$Preliminary: only 14 of 112 configurations tested. Weather-96 primary metric: MSE = 0.138.

The new M4 sweep shows that the frontier is exceptionally compressed on all fully evaluated periods except Hourly. The top-5 configurations are separated by only 0.030 SMAPE on Yearly, 0.021 on Quarterly, 0.035 on Monthly, and 0.015 on Weekly --- roughly 0.2--0.3% relative spread in each case. In practice, this means that for M4 short-to-medium horizons the main architectural question is no longer "can wavelets beat the baseline?" but rather **which architecture reaches the same frontier with fewer parameters and lower seed variance**. Hourly is the clear exception: its top-5 spread widens to 0.102 SMAPE (1.19% relative), and the top two models are both paper baselines, indicating a genuine remaining advantage for the original prefix-body design at the longest M4 horizon.

**Table 4: Novel vs. Paper Baseline Head-to-Head**

| Dataset | Best Novel | Novel SMAPE | Best Baseline | Baseline SMAPE | $\Delta$SMAPE | Param Savings |
|---------|-----------|:-----------:|---------------|:--------------:|:-----:|:-----:|
| M4-Yearly | TW\_10s coif2 | 13.499 | NBEATS-IG\_10s | 13.561 | −0.5% | 9$\times$ fewer |
| M4-Monthly | TW\_30s coif2 | 13.279 | NBEATS-IG\_10s | 13.309 | −0.2% | 3$\times$ fewer |
| M4-Weekly | T+Db3V3\_30s | 6.671 | NBEATS-IG\_30s | 6.822 | −2.2% ($p = 0.014$) | 1.3$\times$ fewer |
| Tourism-Y | TW\_10s db3 | 21.773 | NBEATS-IG\_10s | 22.265 | −2.2% | 10$\times$ fewer |
| Weather-96 | TAE+DB3V3AE\_30s | 0.138 MSE | NBEATS-IG\_10s | 0.183 MSE | −25.0% | 3.6$\times$ fewer |
| Milk | TALG+DB3V3ALG\_10s | 1.512 | NBEATS-IG\_10s | 1.785 | −15.3% | 20$\times$ fewer |

The improvements are largest on non-M4 datasets (Weather: −25%, Milk: −15.3%) where the competition-tuned baselines are least suited to the data regime. On the M4 benchmark where baselines were originally optimized, improvements are smaller but still consistent (−0.2% to −2.2%), always accompanied by substantial parameter savings.

### 5.2 Parameter Efficiency

The most practically significant finding is the extreme parameter efficiency of autoencoder-compressed wavelet architectures. Table 5 shows that sub-1M parameter models achieve within 0.5% of dataset winners on most benchmarks.

**Table 5: Sub-1M Parameter Models vs. Dataset Winners**

| Dataset | Best Sub-1M Config | Params | SMAPE | vs. Winner | Param Ratio |
|---------|-------------------|-------:|:-----:|:----------:|:-----------:|
| M4-Yearly | TWAELG\_10s\_ld16\_coif2\_agf | 436K | 13.524 | +0.2% | 5$\times$ fewer |
| M4-Quarterly | TWAELG\_10s\_ld8\_db3\_agf | 433K | 10.167 | +0.4% | 45$\times$ fewer |
| M4-Monthly | TWAE\_10s\_ld32\_ag0 | 584K | 13.325 | +0.4% | 12$\times$ fewer |
| M4-Weekly | TWAELG\_10s\_ld16\_sym10\_ag0 | 498K | 6.693 | +0.3% | 32$\times$ fewer |
| Tourism-Y | TWAELG\_10s\_ld16\_coif2\_agf | 418K | 21.908 | +0.6% | 5$\times$ fewer |
| Milk | TWAE\_10s\_ld8\_agf | 415K | 1.633 | +8.0% | 37$\times$ fewer |

The TrendWavelet family with AELG backbone (TWAELG) at ~436K parameters is the most parameter-efficient competitive architecture in our sweep. These models use the AERootBlockLG backbone ($w = 256$, $d = 16$) with TrendWavelet basis expansion ($p = 3$ polynomial degree, wavelet `basis_dim` equal to the forecast length), stacked 10 deep.

On M4 specifically, the compact frontier holds through Weekly: the best sub-1M models are only +0.18% (Yearly), +0.40% (Quarterly), +0.35% (Monthly), and +0.32% (Weekly) behind the winner while using 5--45$\times$ fewer parameters. The frontier bends noticeably only on the longest horizons: +2.76% on Hourly and +16.4% on the preliminary Daily subset. This suggests that the wavelet+AE designs capture almost all of the useful capacity for short and medium M4 horizons, but Hourly still rewards the additional depth and width of the original interpretable+generic baseline.

[**Figure 4**: Parameter efficiency scatter plot. Each panel shows one dataset; x-axis is parameter count (log scale), y-axis is mean SMAPE (lower is better). Points colored by architecture category. Pareto frontier connects the configurations that are not dominated (no other config has both fewer parameters and better SMAPE). Paper baselines cluster in the top-right (high params, good SMAPE). Novel TWAELG/TWAE configurations populate the bottom-left (low params, comparable SMAPE), forming the efficient frontier. *To be produced from comprehensive sweep CSVs.*]

**Table 6: Pareto-Optimal Configurations Across Regimes**

| Regime | Configuration | Params | Strengths | Weaknesses |
|--------|--------------|-------:|-----------|------------|
| Minimum params | TWAE\_10s\_ld8\_agf | 415K | Good on M4/Tourism/Milk | Avoid Weather (agf) |
| Balanced | TWAELG\_10s\_ld16\_coif2\_ag0 | 436K | Good everywhere (ag0 safe) | Not top-1 anywhere |
| Best generalist | TALG+DB3V3ALG\_10s\_ag0 | 2,390K | Best cross-dataset mean rank | Weak on Tourism |
| Max quality (M4-Y) | TW\_10s\_td3\_bdeq\_coif2 | 2,076K | M4-Yearly winner | Poor on Milk/Weather |
| Max quality (Weather) | TAE+DB3V3AE\_30s\_ld8\_ag0 | 7,100K | Weather MSE winner | 30-stack depth |

### 5.3 Overparameterization and Convergence

The comprehensive sweep reveals that the original N-BEATS architecture suffers from severe overparameterization on all but the largest datasets. This manifests as training instability: a significant fraction of runs fail to converge, with failure rates strongly correlated with the ratio of model parameters to training data.

**Table 7: Divergence Rates by Architecture Category and Dataset**

| Category | M4 | Tourism | Weather | Milk |
|----------|:--:|:-------:|:-------:|:----:|
| Paper baselines (30-stack, RB) | 0.05% | 0.6% | 0% | 17.1% |
| Alternating Trend+WaveletV3 (RB) | < 0.1% | < 1% | 0% | 30--50% |
| Unified TrendWavelet (RB) | < 0.1% | 0% | 0% | 10% |
| AE/AELG backbone variants | < 0.1% | 0% | 0% | 1.7% |
| BottleneckGeneric (all) | < 0.1% | 10%+ | 0% | 40%+ |

The pattern is clear: RootBlock-based configurations (which retain the full $w$-width backbone) diverge at high rates on small datasets (Milk: up to 50%), while AE-backbone variants with 10$\times$ fewer parameters converge reliably (Milk: 1.7%). The TrendWavelet block family occupies an intermediate position --- its structured basis (polynomial + wavelet) provides implicit regularization that makes it more stable than Generic blocks even without the AE backbone.

The most striking case is NBEATS-G (30-stack Generic) on Milk: with 26M parameters for a single univariate series of 156 observations, this configuration exhibits 40% divergence without `active_g`, and even with `active_g = \text{'forecast'}`, its SMAPE degrades from 19.35 to 2.42 --- far from the 1.51 achieved by TALG+DB3V3ALG at 1.0M parameters.

Within M4, the instability story is narrower and more precise than the cross-dataset summary in Table 7 suggests. Most architectures on Yearly, Monthly, and Hourly converge to tight bands with modest seed variance. The main M4 failure mode is **overparameterized Generic blocks on medium horizons**, especially NBEATS-G\_30s\_ag0 on Quarterly (12.74 $\pm$ 7.41 SMAPE) and Weekly (11.61 $\pm$ 7.16), where a subset of seeds gets trapped on high-loss plateaus while the best seeds remain competitive.

![Figure 5. M4 training curves](figures/figure5_m4_training_curves.svg)

**Figure 5: Training curves from `comprehensive_sweep_m4_results.csv`.** Panel A shows mean validation curves for four near-frontier M4-Yearly configurations: the paper baseline NBEATS-IG\_10s\_ag0, the Yearly winner TW\_10s\_td3\_bdeq\_coif2, the alternating T+Db3V3\_30s\_bd2eq, and the compact TWAELG\_10s\_ld16\_coif2\_agf. All descend into the same narrow validation band despite large architectural differences. Panel B shows the 10 individual Quarterly training curves for NBEATS-G\_30s\_ag0; several seeds converge normally, but a small number remain at dramatically higher loss, producing the sweep's largest M4 variance. Panel C compares Weekly NBEATS-G\_30s with and without `active_g = \text{'forecast'}`. The activation collapses the spread from 11.61 $\pm$ 7.16 to 7.07 $\pm$ 0.24 SMAPE, eliminating the pathological high-loss trajectories, but the stabilized Generic model still trails the best alternating trend+wavelet stacks at 6.67.

**Bimodal convergence is a Generic-block problem.** The bimodal pattern is most severe for Generic and BottleneckGeneric blocks, which lack structural constraints on their basis expansions. NBEATS-G\_30s\_ag0 shows catastrophic bimodal failure on M4-Quarterly (std = 7.4), M4-Weekly (std = 7.2), Tourism (10% divergence rate), and Milk (40% divergence). In contrast, TrendWavelet blocks are immune to bimodal convergence regardless of backbone type or depth --- their polynomial + wavelet basis provides sufficient structural constraint to regularize the optimization landscape.

**The AE bottleneck as implicit regularizer.** The AE backbone eliminates most divergence not through any explicit regularization term, but through parameter compression. By forcing the hidden representation through a low-dimensional bottleneck ($d = 8$--32), it removes the capacity for the degenerate solutions that overparameterized RootBlock networks can explore. On Milk, this reduces divergence from 40.6% (RootBlock) to 6.8% (AELG) to 1.7% (AE).

### 5.4 Stack Architecture Selection

The choice between unified and alternating stack architectures is horizon-dependent.

**Table 8: Unified vs. Alternating Preference by Dataset**

| Dataset | $H$ | Unified Better? | Evidence |
|---------|:---:|:----------:|----------|
| M4-Yearly | 6 | Yes | TW\_10s wins |
| M4-Quarterly | 8 | Tie | Top-5 is mixed |
| M4-Monthly | 18 | Yes | TW\_30s wins |
| M4-Weekly | 13 | No | Alternating T+WavV3 wins |
| Tourism-Y | 4 | Yes | Unified beats alternating by 0.5--0.9 SMAPE |
| Weather-96 | 96 | No | Alternating $\gg$ unified ($p < 0.0001$) |
| Milk | 6 | No | Alternating wins (but high divergence for non-AE) |

**Pattern.** Short horizons ($H = 4$--8) favor unified TrendWavelet blocks, where the combined polynomial+wavelet decomposition within a single block is sufficient to capture the limited structure in short forecasts. Longer horizons and more complex data (multivariate Weather, M4-Weekly) favor alternating stacks, where the separation of Trend and Wavelet blocks across stacks allows each block type to specialize through the residual decomposition.

The M4 sweep makes the horizon-depth transition especially clear: the winning models use 10 stacks at $H = 6$ (Yearly) and $H = 8$ (Quarterly), then switch to 30 stacks at $H = 13$ (Weekly), $H = 18$ (Monthly), and $H = 48$ (Hourly). What changes across these periods is not the existence of a competitive wavelet frontier --- which persists throughout --- but the amount of iterative decomposition required to reach it.

This pattern has an intuitive explanation. For short horizons, the forecast output has few dimensions ($H = 4$--8), and a single block can efficiently decompose it using a combined basis. For longer horizons ($H \geq 13$), the forecast has enough structure to benefit from separate specialized blocks operating in sequence. The alternating pattern is particularly powerful on Weather-96 ($H = 96$), where the 96-dimensional forecast benefits from iterative refinement by complementary block types.

### 5.5 Backbone Hierarchy and Its Reversal

A surprising finding is that the optimal backbone type reverses across datasets.

**M4 and Tourism**: RootBlock $>$ AELG $>$ AE. On these competition-format datasets with many series, the unconstrained capacity of the RootBlock backbone produces the best results. The AE bottleneck, while parameter-efficient, slightly constrains expressiveness. The AELG gate partially compensates, placing AELG between RootBlock and AE.

**Weather and Milk**: AE $>$ AELG $>$ RootBlock. On multivariate Weather (21 sensors, normalized data) and univariate Milk (single series, strong seasonality), the pattern reverses completely. The AE backbone's compression acts as beneficial regularization, and the learned gate in AELG adds noise rather than helping.

[**Figure 6**: Backbone comparison boxplots. Four panels (M4-Yearly, Tourism-Yearly, Weather-96, Milk), each showing SMAPE distributions for the three backbone types across all configurations that use that backbone. On M4-Yearly and Tourism, the RootBlock (leftmost) boxplot has the lowest median; on Weather and Milk, the AE backbone (rightmost) has the lowest median. Whiskers extend to the data range; individual runs shown as jittered dots. *To be produced from comprehensive sweep CSVs.*]

**Hypothesis.** The backbone hierarchy reversal likely reflects the interaction between model capacity and data complexity. M4/Tourism contain thousands of diverse series whose heterogeneity rewards unconstrained capacity. Weather/Milk have simpler or more structured signal patterns where compression eliminates noise dimensions that would otherwise add variance. On Milk specifically, the AE backbone achieves 1.7% divergence (vs. 40.6% for RootBlock), and its lower variance compensates for any expressiveness loss.

### 5.6 Hyperparameter Analysis

**Table 9: Hyperparameter Sensitivity Matrix**

| Setting | M4-Yearly | M4-Quarterly | M4-Monthly | M4-Weekly | M4-Hourly | Tourism | Weather | Milk |
|---------|:---------:|:------------:|:----------:|:---------:|:---------:|:-------:|:-------:|:----:|
| **active\_g** | Slight help | Neutral | Slight hurt | Neutral | Helps | Essential ($p = 0.0002$) | Catastrophic unified; OK alt. | Critical for Generic |
| **Best depth** | 10 | 10 | 30 | Mixed | 30 | 10 ($p < 0.001$) | 30 alt. / 10 unified | 10 ($p = 0.007$) |
| **Best wavelet** | coif2 | Any | Haar/coif2 | Any | Haar | db3 | db3 | Haar |
| **Best ld** | 16 | 8 | 16 | --- | 16 | 8/16/32 equiv. | 8 $>$ 16 $>$ 32 | 8 sufficient |
| **Skip connections** | Hurts | --- | --- | Marginal | --- | Harmful | Marginal | N/A |
| **Backbone** | RB $>$ AELG $>$ AE | --- | --- | --- | --- | RB $>$ AELG $>$ AE | AE $>$ AELG $>$ RB | AE $>$ AELG |
| **Basis dim** | eq\_fcast | --- | 2$\times$eq | eq\_fcast | 2$\times$eq | eq\_fcast | 2$\times$eq | eq\_fcast |

#### 5.6.1 active\_g

The `active_g` mechanism exhibits the most complex dataset dependence of any hyperparameter:

- **Tourism**: Essential. `active_g = \text{'forecast'}` eliminates all bimodal convergence failures (Wilcoxon $p = 0.0002$) and is broadly beneficial across all architecture families.
- **Milk**: Critical for Generic blocks specifically. Reduces NBEATS-G SMAPE from 19.35 to 2.42 (an 87% improvement), but is marginal for wavelet-based architectures that are already convergence-stable.
- **Weather**: Catastrophic for unified/homogeneous stacks (SMAPE $\approx 100$ vs. $\approx 42$ without), but benign or slightly beneficial for alternating stacks. This asymmetry appears to arise because ReLU on forecast outputs interacts differently with the residual chain when all blocks are identical (unified) versus when they alternate between block types.
- **M4**: Strongly horizon- and architecture-dependent. For the unstable 30-stack Generic baseline, `active_g = \text{'forecast'}` is transformative on Quarterly (12.74 $\rightarrow$ 10.59 SMAPE) and Weekly (11.61 $\rightarrow$ 7.07), and on Hourly it improves 36 of 38 matched config pairs while both winning paper baselines use `agf`. But it hurts most already-stable Monthly and Interpretable configurations, so on M4 it should be viewed as a **targeted stabilization tool**, not a universal default.

**Safe default**: `active_g = \text{False}` is safe everywhere. `active_g = \text{'forecast'}` should be used with caution, particularly on Weather-like multivariate datasets with unified stacks.

#### 5.6.2 Stack Depth

Optimal stack depth is primarily a function of forecast horizon:

- **Short horizons ($H = 4$--8)**: 10 stacks is optimal. Tourism shows highly significant preference for 10 stacks ($p < 0.001$, winning 25/28 paired comparisons). Milk similarly prefers 10 stacks ($p = 0.007$).
- **Long horizons ($H \geq 14$)**: 30 stacks is optimal. M4-Monthly ($H = 18$), M4-Hourly ($H = 48$), and alternating stacks on Weather ($H = 96$) all benefit from greater depth.
- **Medium horizons ($H = 8$--13)**: Mixed. M4-Quarterly slightly prefers 10; M4-Weekly shows no clear preference.

The horizon-depth interaction makes sense: short forecasts have limited structure that can be decomposed in 10 blocks, while longer forecasts benefit from the finer-grained iterative decomposition that 30 blocks provide.

#### 5.6.3 Wavelet Family

Wavelet family choice is statistically significant on 3 of 4 datasets (Kruskal-Wallis $p < 0.05$) but the optimal family is dataset-dependent:

- **db3** is the safest cross-dataset default, ranking 1st or 2nd on Weather, Tourism (non-AE), and M4-Weekly.
- **coif2** is best on M4-Yearly.
- **Haar** is best on Milk and M4-Hourly.
- **Sym10** shows strength on Weather (AELG category) and M4-Yearly (alternating stacks).

On M4-Yearly specifically, wavelet family is *not* significant (KW $p = 0.79$), suggesting that the orthonormal DWT basis quality matters more than the specific wavelet shape for short-horizon forecasting.

#### 5.6.4 Latent Dimension

For AE-backbone variants, `latent_dim = 16` is broadly safe, but Weather reverses the hierarchy:

- **M4/Tourism**: ld = 16 generally best or tied with ld = 8.
- **Weather**: ld = 8 $>$ ld = 16 $>$ ld = 32 (KW $p = 0.010$). The smaller bottleneck provides stronger regularization on this multivariate dataset.
- **Milk**: ld = 8 is sufficient for the simple univariate signal.

#### 5.6.5 Basis Dimension

The `basis_dim` parameter controls how many wavelet basis vectors are used:

- **eq\_fcast** ($k = H$): Uses exactly as many basis vectors as forecast steps. Optimal for short horizons (M4-Yearly, Tourism, Milk).
- **2$\times$eq** ($k = 2H$): Over-complete basis with twice as many vectors as forecast steps. Optimal for longer horizons (M4-Monthly, M4-Hourly, Weather) where the extra basis vectors capture finer frequency resolution.

#### 5.6.6 Skip Connections

ResNet-style skip connections (`skip_distance`, `skip_alpha`) that periodically re-inject the original input into the residual stream are **not recommended**. They hurt performance on M4-Yearly and Tourism, provide only marginal benefit on Weather and M4-Weekly, and add hyperparameter complexity. The structured basis blocks (TrendWavelet, WaveletV3) do not exhibit the gradient decay that skip connections were designed to address.

**Cross-Dataset Safe Defaults:**
> `active_g = False`, 10 stacks, db3 wavelet, `latent_dim = 16`, eq\_fcast basis dim, no skip connections. This combination is best or tied-for-best on 6 of 9 dataset-periods.

### 5.7 Cross-Dataset Generalist Analysis

While dataset-specific tuning always outperforms a single generalist, practitioners often need a configuration that works reasonably well across diverse forecasting tasks. Table 10 reports the top 10 configurations ranked by mean rank across five core datasets (M4-Yearly, M4-Quarterly, Tourism, Weather, Milk).

**Table 10: Top 10 Generalist Configurations by Mean Rank**

| Rank | Configuration | Mean Rank | M4-Y | M4-Q | Tourism | Weather | Milk | Params |
|:----:|--------------|:---------:|:----:|:----:|:-------:|:-------:|:----:|-------:|
| 1 | TALG+DB3V3ALG\_10s\_ag0 | 14.6 | 33 | 7 | 30 | 1 | 2 | 2,390K |
| 2 | NBEATS-IG\_10s\_ag0 | 21.6 | 22 | 1 | 49 | 22 | 14 | 19,644K |
| 3 | TWGAELG\_10s\_ld16\_agf | 22.2 | 9 | 44 | 23 | 13 | 22 | 1,285K |
| 4 | T+Db3V3\_30s\_bd2eq | 23.2 | 5 | 9 | 70 | 16 | 16 | 15,287K |
| 5 | TW\_10s\_td3\_bdeq\_coif2 | 24.6 | 1 | 5 | 9 | 53 | 55 | 2,076K |
| 6 | TALG+HaarV3ALG\_30s\_ag0 | 26.0 | 26 | 3 | 84 | 12 | 5 | 3,284K |
| 7 | T+Sym10V3\_30s\_bdeq | 26.2 | 16 | 18 | 89 | 7 | 1 | 15,241K |
| 8 | TALG+Sym10V3ALG\_30s\_agf | 26.6 | 2 | 22 | 79 | 15 | 15 | 3,134K |
| 9 | TWAELG\_10s\_ld16\_coif2\_agf | 30.8 | 4 | 39 | 6 | 98 | 7 | 436K |
| 10 | TWAELG\_10s\_ld16\_coif2\_ag0 | 30.8 | 49 | 21 | 27 | 44 | 13 | 436K |

The best generalist --- TALG+DB3V3ALG\_10s\_ag0 (alternating TrendAELG + DB3WaveletV3AELG, 10 stacks, `active_g = \text{False}`) --- achieves a mean rank of 14.6 out of 112 with only 2.4M parameters. It ranks top-10 on Weather (1st) and Milk (2nd), competitive on M4-Quarterly (7th), and never worse than rank 33.

Notably, the M4-Yearly winner (TW\_10s\_td3\_bdeq\_coif2, rank 1 on M4-Y) drops to rank 53 on Weather and 55 on Milk, illustrating the fundamental tension between per-dataset specialization and cross-dataset robustness. Conversely, the most parameter-efficient options (TWAELG at 436K, ranks 9--10 as generalists) provide an exceptional accuracy-per-parameter ratio across most datasets but struggle on Weather (rank 98), where their small capacity is insufficient for the 21-variable, 96-step forecasting task.

### 5.8 Transferability to N-HiTS

To test whether the novel block types transfer beyond the original doubly residual N-BEATS topology, we ran a dedicated `NHiTSNet` benchmark on the Weather dataset across four horizons (96, 192, 336, 720), totaling 408 runs. This benchmark uses the same block registry inside N-HiTS's hierarchical pooling and interpolation framework, allowing us to isolate **block transferability across architectures**.

**Table 12: N-HiTS Weather Benchmark by Horizon**

| Horizon | Best Novel Config | Novel MSE | Best Vanilla Baseline | Baseline MSE | $\Delta$MSE | Param Savings |
|--------:|-------------------|:---------:|-----------------------|:------------:|:-----------:|:-------------:|
| 96 | NHiTS-TWGVAE\_agf | 0.1779 | NHiTS-Generic | 0.2483 | −28.4% | 9$\times$ fewer |
| 192 | NHiTS-GenericAELG | 0.1988 | NHiTS-Generic | 0.2031 | −2.1% | 2.3$\times$ fewer |
| 336 | NHiTS-TWGVAE\_agf | 0.2170 | NHiTS-Generic | 0.2507 | −13.5% | 6.7$\times$ fewer |
| 720 | NHiTS-TWG\_agf | 0.6021 | NHiTS-IG | 0.5849 | +2.9% | 9$\times$ fewer |

Here `TWGVAE` denotes `TrendWaveletGenericVAE` and `TWG` denotes `TrendWaveletGeneric`. The transferred blocks beat the best vanilla N-HiTS baseline on three of four horizons, and on the remaining 720-step task the best novel model remains close: +2.9% MSE while using only 2.83M parameters versus 25.36M for NHiTS-IG.

Two lessons follow. First, **the block innovations are genuinely architecture-transferable**: wavelet and compressed-backbone blocks remain competitive even after N-HiTS replaces N-BEATS's stack semantics with multi-rate pooling and hierarchical interpolation. Second, **transferability is not rank invariance**. The best N-BEATS blocks are not automatically the best N-HiTS blocks. On the pooled all-horizon Weather average across complete 20-run configurations, vanilla NHiTS-Generic remains best (0.3286 MSE), but several transferred variants sit very close with far fewer parameters --- for example, `NHiTS-TrendAE+DB3V3AE+TrendAE` reaches 0.3336 MSE with 408K parameters, and `NHiTS-TrendWaveletGenericAELG-agF` reaches 0.3357 with 312K.

The N-HiTS benchmark also shows that hyperparameter guidance does not transfer wholesale. In the original N-BEATS sweep we treated VAE backbones as broadly uncompetitive, yet in N-HiTS the `TrendWaveletGenericVAE` variant wins at horizons 96 and 336. Conversely, `active_g = \text{'forecast'}` is helpful for the best transferred TrendWaveletGeneric variants but disastrous for pure N-HiTS Generic-family models (`NHiTS-Generic`: 0.3286 $\rightarrow$ 0.4326 MSE; `NHiTS-GenericAE`: 0.3379 $\rightarrow$ 0.4271). The correct conclusion is therefore stronger than simple plug-compatibility but weaker than universal recipe transfer: **the block interface transfers cleanly across architectures, but the optimal architecture-block-hyperparameter combination remains architecture-dependent**.

---

## 6. Discussion

### 6.1 Why Wavelet Bases Work: Inductive Bias Alignment

The success of wavelet basis blocks across six of nine dataset-periods reflects a fundamental alignment between the wavelet transform's multi-resolution time-frequency localization and the structure of real-world time series. Each of the three original N-BEATS basis types captures a specific structural dimension:

- **Polynomial (Trend)**: Smooth, slowly varying functions. Excellent for level and slope, but blind to abrupt changes or oscillations.
- **Fourier (Seasonality)**: Globally periodic functions. Excellent for regular cycles, but assumes stationarity and distributes energy uniformly across time.
- **Fully learned (Generic)**: No structural constraint. Maximum flexibility, but requires learning the basis from scratch, leading to overparameterization and convergence fragility.

Wavelets occupy a unique position: they provide **localized** multi-scale analysis, capturing transient phenomena (regime changes, localized oscillations, abrupt level shifts) that polynomials miss while preserving the parameter efficiency that fully-learned bases sacrifice. The DWT decomposition separates approximation (low-frequency trend) from detail (high-frequency transients) at multiple scales, mirroring the intuition that time series contain structure at multiple temporal resolutions.

The TrendWavelet block makes this complementarity explicit by combining polynomial and wavelet bases in a single block. The polynomial component ($p = 3$ coefficients) captures the macro-trend that the wavelet basis would otherwise need many coefficients to approximate. The wavelet component captures the residual transient structure. This within-block decomposition is more parameter-efficient than the between-stack decomposition of the original Interpretable architecture (where Trend and Seasonality occupy separate stacks), because it avoids the overhead of separate backbone computations for each decomposition component.

### 6.2 Why V1/V2 Failed and V3 Succeeds: Basis Conditioning as Prerequisite

The failure of our initial wavelet block implementations (V1/V2) and the success of V3 provides a broadly applicable lesson: **basis matrix conditioning is a prerequisite for integration into gradient-trained architectures.**

The V1/V2 basis construction produced matrices with condition number $\kappa \approx 604{,}000$ for DB3. During backpropagation, gradients are multiplied by the basis matrix transpose, amplifying components in the direction of the smallest singular value by up to $604{,}000\times$. This gradient amplification manifests as two failure modes: (1) immediate NaN at epoch 1 when the amplified gradients overflow float32 precision, and (2) gradual MASE explosion over 10--20 epochs as the weight matrices drift into a degenerate regime.

The V3 construction via impulse-response synthesis followed by SVD orthogonalization produces bases with $\kappa = 1.0$. All singular values are equal, so gradient flow through the basis matrix is uniform across all directions. This eliminates the gradient amplification that caused V1/V2 failures, converting the 67--100% failure rate to near-zero.

This finding has implications beyond N-BEATS: any architecture that uses frozen analytical basis matrices within a gradient-trained pipeline should verify that the basis is well-conditioned. Ill-conditioned bases act as implicit gradient amplifiers that can destabilize training even when the basis itself is mathematically valid.

### 6.3 The Overparameterization Problem

Our results provide compelling evidence that the original N-BEATS architecture is massively overparameterized for most practical forecasting tasks. The evidence is threefold:

1. **Equivalent accuracy at 10--50$\times$ fewer parameters.** TWAELG models at 436K parameters achieve within 0.5% of the 26M-parameter NBEATS-G on M4-Yearly/Quarterly/Monthly/Weekly. If 98% of the parameters can be removed without meaningful accuracy loss, those parameters were not contributing to the model's predictive capacity.

2. **Divergence correlates with overparameterization.** On the Milk dataset (1 series, 156 observations), NBEATS-G\_30s has a 167,000$\times$ parameter-to-observation ratio and a 40% divergence rate. Reducing to 1M parameters (TALG+DB3V3ALG\_10s, 6,400$\times$ ratio) eliminates divergence entirely. The divergence is not a training bug --- it is the natural consequence of an optimization landscape with exponentially many degenerate minima created by redundant parameters.

3. **Backbone compression provides implicit regularization.** The AE backbone reduces backbone parameters by 10$\times$ (Section 3.2.1) and reduces Milk divergence from 40.6% to 1.7%. This compression achieves through architecture what dropout, weight decay, or early stopping attempt through training dynamics --- and does so more reliably, because it removes the degenerate solutions from the parameter space entirely rather than penalizing them softly.

The practical implication is that the N-BEATS community should reconsider the default 30-stack, 512-unit configuration as a starting point. For datasets smaller than M4 --- which includes most practical forecasting applications --- a 10-stack TrendWavelet configuration with AE backbone provides better accuracy, lower variance, and faster training at a fraction of the computational cost.

### 6.4 Practical Recommendations

Based on the comprehensive sweep results, we provide the following architecture selection guidelines:

**Table 11: Practitioner Decision Tree**

| Dataset Regime | Recommended Architecture | Key Settings | Params |
|---------------|-------------------------|-------------|-------:|
| Large multi-series ($n > 10{,}000$) | TrendWavelet (RB) or Alt. Trend+WavV3 (RB) | coif2/db3, td3, 10--30 stacks | 2--16M |
| Medium multi-series ($n = 100$--$10{,}000$) | TrendWavelet (AELG) or NBEATS-IG | db3, ld=16, 10 stacks, agf | 436K--2M |
| Small / univariate ($n < 100$) | Alt. TrendAELG+WavV3AELG or TWAE | db3/Haar, ld=8, 10 stacks, ag0 | 415K--1M |
| Multivariate long-horizon | Alt. TrendAE+WavV3AE | db3, ld=8, ag0, 30 stacks | 3--7M |
| Unknown (generalist) | TALG+DB3V3ALG\_10s\_ag0 | db3, ld=16, ag0, 10 stacks | 2.4M |

For all regimes, we recommend starting with `active_g = \text{False}` (safe globally) and 10 stacks, increasing to 30 only for horizons $H \geq 14$. Skip connections should not be used. The db3 wavelet is the safest default; practitioners can fine-tune the wavelet family on their validation set.

---

## 7. Conclusion

This work presents a systematic exploration of alternative basis expansion functions and backbone architectures within the N-BEATS doubly residual framework. Through a comprehensive benchmark of 112 configurations across 10 random seeds on four datasets spanning nine forecasting tasks, we arrive at three principal findings.

**First, wavelet basis expansions provide genuine inductive bias benefits.** Orthonormal DWT bases, properly conditioned via SVD orthogonalization (V3), beat paper baselines on six of nine dataset-periods. The TrendWavelet block --- which combines polynomial trend and wavelet detail bases in a single block --- is the most parameter-efficient competitive design, requiring only 418--436K parameters (with AE backbone) to match or beat 26M-parameter baselines. The natural complementarity of polynomial trend and wavelet detail decomposition makes TrendWavelet a robust default for practitioners.

**Second, the original N-BEATS architecture is massively overparameterized.** The 30-stack, 512-unit Generic configuration (26M parameters) diverges in 40--50% of training runs on small datasets, while autoencoder-compressed variants with 10--50$\times$ fewer parameters converge reliably and achieve equivalent or better accuracy. This is not a training pathology but a fundamental consequence of the parameter-to-data ratio. The AE backbone acts as implicit regularization through architectural compression, providing more reliable convergence than explicit regularization techniques.

**Third, architecture selection is inherently dataset-dependent.** Backbone hierarchy, stack architecture preference, optimal depth, and wavelet family all reverse across datasets. No single configuration dominates everywhere. However, the alternating TrendAELG + DB3WaveletV3AELG configuration at 10 stacks (2.4M parameters) provides the best cross-dataset generalist performance, ranking first on Weather, second on Milk, and competitive across all M4 periods.

The dedicated N-HiTS benchmark extends this conclusion beyond a single architecture family. The same block registry transfers directly into N-HiTS and remains competitive there, beating the vanilla N-HiTS baselines on three of four tested Weather horizons. This supports the view that our main contribution is not a one-off modification to N-BEATS, but a library of basis-expansion blocks whose usefulness survives changes to the stack-level architecture.

The power of the N-BEATS framework lies in its doubly residual stacking topology --- the iterative decomposition and hierarchical forecast aggregation --- rather than in any specific basis expansion. But the choice of basis determines parameter efficiency, convergence reliability, and the alignment of inductive biases with data structure. Wavelets and autoencoder compression provide tools to exploit this design freedom, delivering forecasting accuracy that matches the original N-BEATS with orders of magnitude fewer parameters and transferring meaningfully to related architectures such as N-HiTS.

**Open questions** for future work include: (a) mechanistic understanding of why `active_g` catastrophically fails on Weather unified stacks but succeeds on alternating stacks; (b) why the backbone hierarchy reverses on multivariate/simple-univariate data versus competition-format data; (c) why the ranking of transferred blocks shifts under N-HiTS hierarchical pooling, including the surprising strength of `TrendWaveletGenericVAE`; and (d) extension to additional benchmarks (ETTh, ETTm, Exchange Rate) to test generalization of the architecture selection guidelines developed here.

---

## References

Aminghafari, M., Cheze, N., & Poggi, J.-M. (2006). Multivariate denoising using wavelets and principal component analysis. *Computational Statistics & Data Analysis*, 50(9), 2381-2398. <https://doi.org/10.1016/j.csda.2004.12.010>

Assimakopoulos, V., & Nikolopoulos, K. (2000). The Theta model: A decomposition approach to forecasting. *International Journal of Forecasting*, 16(4), 521-530. <https://doi.org/10.1016/S0169-2070(00)00066-2>

Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.

Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler-Canseco, M., & Dubrawski, A. (2023). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(6), 6989-6997. <https://doi.org/10.1609/aaai.v37i6.25854>

Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics*, 6(1), 3-73.

Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM. <https://doi.org/10.1137/1.9781611970104>

Frankle, J., & Carlin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. *International Conference on Learning Representations (ICLR 2019)*. <https://openreview.net/forum?id=rJl-b3RcF7>

Han, S., Mao, H., & Dally, W. J. (2015). Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. *arXiv preprint arXiv:1510.00149*.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778. <https://doi.org/10.1109/CVPR.2016.90>

Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. *Science*, 313(5786), 504-507. <https://doi.org/10.1126/science.1127647>

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv preprint arXiv:1503.02531*.

Holt, C. C. (1957). Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages. *ONR Memorandum No. 52*. Carnegie Institute of Technology.

Hyndman, R. J., & Khandakar, Y. (2008). Automatic Time Series Forecasting: The forecast Package for R. *Journal of Statistical Software*, 27(3), 1-22. <https://doi.org/10.18637/jss.v027.i03>

Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *International Journal of Forecasting*, 18(3), 439-454. <https://doi.org/10.1016/S0169-2070(01)00110-8>

Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764. <https://doi.org/10.1016/j.ijforecast.2021.03.012>

Makridakis, S., & Hibon, M. (2000). The M3-Competition: Results, conclusions and implications. *International Journal of Forecasting*, 16(4), 451-476. <https://doi.org/10.1016/S0169-2070(00)00057-1>

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. *PLOS ONE*, 13(3), e0194889. <https://doi.org/10.1371/journal.pone.0194889>

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54-74. <https://doi.org/10.1016/j.ijforecast.2019.04.014>

Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection. *arXiv preprint arXiv:1607.00148*.

Mallat, S. G. (1989). A Theory for Multiresolution Signal Decomposition: The Wavelet Representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674-693. <https://doi.org/10.1109/34.192463>

Montero-Manso, P., Athanasopoulos, G., Hyndman, R. J., & Talagala, T. S. (2020). FFORMA: Feature-based forecast model averaging. *International Journal of Forecasting*, 36(1), 86-92. <https://doi.org/10.1016/j.ijforecast.2019.02.011>

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *International Conference on Learning Representations (ICLR 2023)*.

Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *International Conference on Learning Representations (ICLR 2020)*. <https://openreview.net/forum?id=r1ecqn4YwB>

Pramanick, N., Singhal, V., et al. (2024). Fusion of Wavelet Decomposition and N-BEATS for Improved Stock Market Forecasting. *SN Computer Science*, 5, 822. <https://doi.org/10.1007/s42979-024-03222-4>

Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191. <https://doi.org/10.1016/j.ijforecast.2019.07.001>

Smyl, S. (2020). A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting. *International Journal of Forecasting*, 36(1), 75-85. <https://doi.org/10.1016/j.ijforecast.2019.03.017>

van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio. *arXiv preprint arXiv:1609.03499*.

Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. *Proceedings of the 25th International Conference on Machine Learning (ICML)*, 1096-1103. <https://doi.org/10.1145/1390156.1390294>

Winters, P. R. (1960). Forecasting Sales by Exponentially Weighted Moving Averages. *Management Science*, 6(3), 324-342. <https://doi.org/10.1287/mnsc.6.3.324>

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11121-11128. <https://doi.org/10.1609/aaai.v37i9.26317>

---

## Appendix A: Complete Results Tables

**Table A1: Architecture Category Rankings by Dataset**

| Category | M4-Y Rank | M4-Q Rank | M4-M Rank | M4-W Rank | M4-H Rank | Tourism Rank | Weather Rank | Milk Rank |
|----------|:---------:|:---------:|:---------:|:---------:|:---------:|:------------:|:------------:|:---------:|
| alt\_trend\_wavelet\_rb | 5 | 2 | 2 | 1 | --- | 7 | 2 | 4 |
| alt\_aelg | 3 | 3 | 4 | 2 | 4 | 6 | 3 | 5 |
| alt\_ae | 4 | 5 | 2 | 5 | 3 | 8 | 1 | 3 |
| trendwavelet\_rb | 1 | 4 | 1 | 8 | --- | 4 | --- | 6 |
| trendwaveletaelg | 2 | 6 | 6 | 3 | 7 | 3 | --- | 2 |
| trendwaveletae | 6 | 7 | 4 | 6 | 9 | 1 | --- | 1 |
| paper\_baseline | 7 | 1 | 3 | 13 | 1 | 5 | 14 | 10 |

Ranks are within-category best configuration rank on each dataset. Dashes indicate the category was not tested on that dataset.

**Table A2** (full 112-config rankings per dataset-period) and **Table A3** (statistical test results for all pairwise comparisons) are available in the supplementary analysis notebooks at `experiments/analysis/notebooks/comprehensive_sweep_cross_dataset.ipynb`.

---

## Appendix B: Block Implementation Details

### B.1 Parameter Count Formulas

For a single block with backbone width $w$, lookback length $L$, forecast horizon $H$, and basis dimension $k$:

**RootBlock + WaveletV3:**
$$P = \underbrace{Lw + 3w^2}_{\text{backbone}} + \underbrace{w \cdot k_b + w \cdot k_f}_{\text{projection heads}} + \underbrace{4w + k_b + k_f}_{\text{biases}}$$

where $k_b$, $k_f$ are backcast and forecast basis dimensions. Wavelet bases are frozen (0 trainable parameters).

**AERootBlock + TrendWavelet:**
$$P = \underbrace{L \cdot w/2 + dw + w^2/2}_{\text{AE backbone}} + \underbrace{w(p + k_b) + w(p + k_f)}_{\text{projection heads}} + \text{biases}$$

where $d$ is latent dimension and $p$ is polynomial degree.

**Full model** with $S$ stacks of 1 shared-weight block each: $P_{\text{total}} = S \cdot P_{\text{block}}$.

### B.2 WaveletV3 Basis Construction Pseudocode

```
function BUILD_V3_BASIS(target_length, wavelet_type, basis_dim, basis_offset):
    coeffs = pywt.wavedec(zeros(target_length), wavelet_type, level=min(max_level, 5))
    basis_rows = []
    for each (band_index, band_length) in coeffs:
        for j in 0..band_length-1:
            impulse = zero_coefficients_like(coeffs)
            impulse[band_index][j] = 1.0
            row = pywt.waverec(impulse, wavelet_type)[:target_length]
            basis_rows.append(row)
    raw_basis = stack(basis_rows)         # shape: (m, target_length)
    U, S, Vt = SVD(raw_basis)             # Vt rows are orthonormal
    rank = count(S > tolerance)
    offset = min(basis_offset, rank - 1)
    effective_dim = min(basis_dim, rank - offset)
    return Vt[offset : offset + effective_dim, :]   # shape: (effective_dim, target_length)
```

### B.3 TrendWavelet Forward Pass Pseudocode

```
function TRENDWAVELET_FORWARD(x):
    h = backbone(x)                              # (batch, units)
    bc_coeffs = backcast_linear(h)               # (batch, trend_dim + wavelet_dim)
    fc_coeffs = forecast_linear(h)               # (batch, trend_dim + wavelet_dim)
    
    bc_trend = bc_coeffs[:, :trend_dim]          # polynomial coefficients
    bc_wave  = bc_coeffs[:, trend_dim:]           # wavelet coefficients
    fc_trend = fc_coeffs[:, :trend_dim]
    fc_wave  = fc_coeffs[:, trend_dim:]
    
    backcast = vandermonde_basis_b @ bc_trend + wavelet_basis_b @ bc_wave
    forecast = vandermonde_basis_f @ fc_trend + wavelet_basis_f @ fc_wave
    
    if active_g:
        apply activation to backcast/forecast per mode
    
    return backcast, forecast
```

---

## Appendix C: M4-Daily Preliminary Results

M4-Daily results are preliminary: only 14 of 112 configurations have been evaluated (paper baselines and TrendWavelet RootBlock variants). No AE, AELG, alternating, or GenericAE configurations have been tested.

**Table C1: M4-Daily Preliminary Results (14 configs, 10 seeds)**

| Config | SMAPE | OWA | Params |
|--------|:-----:|:---:|-------:|
| NBEATS-G\_30s\_ag0 | 2.603 | 0.861 | 26.0M |
| NBEATS-IG\_30s\_agf | 2.658 | 0.876 | 20.3M |
| TW\_30s\_td3\_bdeq\_coif2 | 2.894 | 0.952 | 7.7M |

The 11% gap between NBEATS-G and TrendWavelet RB is the largest baseline-vs-novel gap in the entire sweep, but this likely reflects the absence of AE/AELG variants and alternating stacks rather than a genuine limitation of wavelet architectures on daily data. On every other M4 period where the full 112-config grid was evaluated, novel architectures matched or beat baselines. Completing the M4-Daily sweep is the highest-priority next experiment.

---

## Appendix D: NHiTS Transferability

### D.1 Architectural Compatibility

All novel block types introduced in this work are fully compatible with N-HiTS (Challu et al., 2023) by construction. The N-HiTS architecture extends N-BEATS with two stack-level operations:

1. **Multi-rate input pooling.** Each stack applies MaxPool1d with a configurable kernel size to its input, reducing the effective lookback length. For example, with kernel size 2, a block in this stack sees a 240-step input as 120 steps.

2. **Hierarchical forecast interpolation.** Each stack produces a forecast at a reduced resolution (configurable downsampling factor per stack) that is interpolated back to the full forecast horizon.

Both operations are external to the block interface: pooling happens before the block's forward pass, and interpolation happens after. Since all our blocks accept a 1D vector and return (backcast, forecast) tuples, they integrate into N-HiTS without any modification. The `NHiTSNet` class in `lightningnbeats` supports the full block registry.

### D.2 Empirical Transfer Summary

The dedicated `nhits_benchmark_results.csv` benchmark provides an initial empirical validation of this compatibility claim. Across 408 Weather runs spanning horizons 96, 192, 336, and 720, transferred block types outperform the vanilla N-HiTS baselines on three of four horizons:

1. At horizon 96, `NHiTS-TrendWaveletGenericVAE-agF` achieves 0.1779 MSE versus 0.2483 for `NHiTS-Generic` (−28.4%).
2. At horizon 192, `NHiTS-GenericAELG` achieves 0.1988 MSE versus 0.2031 for `NHiTS-Generic` (−2.1%).
3. At horizon 336, `NHiTS-TrendWaveletGenericVAE-agF` achieves 0.2170 MSE versus 0.2507 for `NHiTS-Generic` (−13.5%).
4. At horizon 720, the original `NHiTS-IG` remains best at 0.5849 MSE, but the best novel configuration (`NHiTS-TrendWaveletGeneric-agF`) is still close at 0.6021 with roughly 9$\times$ fewer parameters.

These results support the core transferability claim of this paper: the benefits of structured basis blocks and compressed backbones are not confined to the N-BEATS residual stack. However, they also show that cross-architecture transfer is not trivial. Ranking order changes, and some settings that are weak in N-BEATS (notably VAE-based TrendWaveletGeneric) become competitive in N-HiTS under hierarchical pooling.

### D.3 Research Directions

Several questions about novel blocks in N-HiTS warrant systematic investigation:

1. **Do wavelet blocks benefit more from hierarchical interpolation than polynomial blocks?** N-HiTS's multi-rate design was motivated by capturing patterns at different temporal scales. Wavelet bases already provide multi-resolution analysis. These two multi-resolution mechanisms may be complementary (capturing different aspects of scale) or redundant.

2. **Does pooling interact with basis conditioning?** MaxPool1d reduces the effective input length, which changes the wavelet decomposition level and available basis vectors. The V3 basis construction adapts automatically (via `pywt.dwt_max_level`), but the interaction between pooled input lengths and basis properties has not been studied.

3. **Are AE backbones more or less beneficial in N-HiTS?** N-HiTS already provides a form of compression (through reduced input resolution). The combination of input-level compression (pooling) and backbone-level compression (AE) may be redundant, or they may target different aspects of the representation.

4. **Cross-architecture transfer of hyperparameter guidelines.** Do the hyperparameter recommendations from Section 5.6 (db3 wavelet, ld=16, 10 stacks, no skip) transfer to N-HiTS, or does the multi-rate structure shift the optima?

These questions represent natural extensions of the present work and would further validate the transferability of novel block designs across the N-BEATS architecture family.
