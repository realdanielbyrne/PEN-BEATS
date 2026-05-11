# Abstract

    It is widely believed that the capacity of a neural network to absorb information is limited by its number of parameters. While this general statement seems to be true, the scale of this parameterization is overblown an unnecessary.

## Introduction

    1. Discussion of overparameterizaton in neural networks of every type.
    2. Motivation for parameter reduction.
        1. Redundancies
        2. Noise
    2. Inspiration from real neural networks structures which limits downstream connections. 
    3. Discussion of common sucessful pre, post and online training parameter reduction techniques which imply over parameterization and interference between adjacent knowledge centers.
        1. Optimal Brain Damage
        2. Dropout
        3. Quantization
    4. Discussion of localization and geometric isolation of facts inside a NN.
        1. ICA, PCA path identification
        2. AEs and CNNs latent visulaizations 
    5. Background and inspiration from autoencoders.
        1. How autoencoders work
        2. how they are shaped
        3. what they are good for.
    6. Background and inspiration from Wavelets
        1. what they are
        2. how they work
        3. what they are used for
            - Traditional Signal Analysis
            - Multiresolution analysis: decompose a signal into coarse approximation plus detail bands. This lets analysts separate long-run trend, medium oscillations, and short transients.
            - Denoising: transform the signal, shrink or threshold small high-frequency coefficients, then invert. 
            - Feature extraction: use wavelet coefficient energies, entropies, maxima, cross-scale patterns, or selected bands as features for classification, anomaly detection, compression, biomedical signal analysis, geophysics, finance, and audio.
    7. N-Beats architecture
        1. Split stack simulating sparse connections 
        2. Basis functions
        
    8. Opportunities for improvement of the base N-BEATS architecture
        1. Saturation evident from quick training
        2. localized 

## Related Work

Time-series forecasting baselines, wavelets in deep learning, autoencoder compression, doubly residual architectures. Keep it tight; NeurIPS reviewers know N-BEATS.
doubly residual stack, trend/seasonality/generic blocks. Define notation for backcast/forecast/theta.

## Method

- Wavelet Basis Expansion (WaveletV3, TrendWavelet, TrendWaveletGeneric) — orthonormal DWT bases, tiered cascade, asymmetric wavelet families.
- Compressed Backbones (AERootBlock, AERootBlockLG) — hourglass MLPs, learned gates.
- Integration into N-BEATS, N-Hits, Transformers — stack composition rules, width selection, active_g scoping.

## Experiments

- M4
  - Sampling Methods
  - Architectures Tested
  - Uniform stacks versus alternating
  - Autoencoder's reliance on Trend

- Milk
  - Divergence
  - Active_g

- Transformers
  - Attention Replacement
  - MLP Replacement

## Results

- M4 Per Period Leaderboard
- Effects of Tiering
- Sub 1M Frontier
- Depth and Stability Analysis
- Ablation

## Discussion

Why structured bases might work.  Localization, ultiresolution, fewer degrees of freedom

## Conclusions and Future Work

- NHiTS replacement
- Transformers linear layer replacements and mlp replacements.
-

Notes:

N-BEATS (Neural Basis Expansion Analysis for Time Series) is a deep learning architecture designed for univariate time series forecasting. Unlike RNNs or Transformers, it relies on a hierarchical "doubly residual" structure of simple fully connected layers.

Core Architecture Components
The Basic Block:

Input: A window of past observations (the "lookback" or backcast).
Processing: A series of fully connected (Linear) layers with ReLU activations.
Outputs: Two separate "heads."
Backcast Head: Predicts the input window itself (reconstruction).
Forecast Head: Predicts the future values (prediction).
Doubly Residual Stacking:

Backward Residual: The block's backcast prediction is subtracted from the input. The next block only sees the error (residual) that the previous block couldn't explain.
Forward Residual: The forecast predictions from every block are summed together to create the final global forecast.
Stacks and Interpretability:

Blocks are grouped into Stacks. By constraining what the "heads" can output, N-BEATS provides interpretability:
Trend Stack: Blocks are forced to output polynomials, capturing smooth long-term growth or decay.
Seasonality Stack: Blocks are forced to output periodic waves (Fourier series), capturing repeating patterns like weekly or daily cycles.
Generic Stack: Uses standard linear layers for complex patterns that don't fit trend or seasonality.
Why it Works
By using residual connections in both directions, the model decomposes a complex signal into simple, additive parts. It doesn't require complex architectures like Attention or Recurrence, making it very fast to train while remaining highly accurate.

That's a significant framing shift — it elevates the work from "N-BEATS improvements" to "general-purpose architectural components." Even with preliminary evidence, positioning the compressed backbones and wavelet bases as portable building blocks is a much stronger NeurIPS story.

Here are revised titles that reflect this broader scope:

Leading with generalizability
Compressed Backbones and Wavelet Basis Projections: Portable Efficiency Gains from N-BEATS to Language Models
Autoencoder-Compressed MLPs and Orthonormal Wavelet Projections as Drop-In Replacements for Dense Layers
From Basis Expansion to General-Purpose Compression: Wavelet and Autoencoder Blocks Beyond N-BEATS
Grounded in N-BEATS but signaling broader applicability
Wavelet Bases and Compressed Backbones for Neural Basis Expansion: Efficiency Gains That Transfer
Rethinking Dense Layers Through Basis Expansion: Wavelets and Autoencoder Compression from Time Series to Language Models
Parameter-Efficient Basis Expansion Blocks: Evidence from Time Series Forecasting and Beyond
Emphasizing the architectural primitive angle
Hourglass MLPs and Orthonormal Projections: Toward General-Purpose Parameter-Efficient Building Blocks
Learnable Compression Meets Fixed Bases: Autoencoder Backbones and Wavelet Projections as Efficient Neural Primitives
Shorter / punchier
Basis Expansion Blocks: Wavelet Projections and Compressed Backbones That Generalize
Less Dense, More Structured: Compressed Backbones and Wavelet Bases as Portable Architectural Components
My updated top picks:

# 7 — positions these as general-purpose building blocks without overclaiming; "toward" hedges appropriately given the early LLM evidence

# 4 — keeps N-BEATS as the grounding while the subtitle signals transferability

# 2 — strongest claim, works if you can include even preliminary NHiTS + LLM results in the paper as supporting evidence

One strategic question: do you plan to include any of the NHiTS/LLaMA results in the submission (even as a brief "preliminary transferability" section), or keep the paper focused on N-BEATS and mention generalizability only in the discussion/future work? That changes how boldly you should frame the title.

I plan on including nhits_benchmark results in an appendix at the very least.  My intention on the pellm respository was a different paper, but I'm wavering on that point considering the overlap.
That's a reasonable dilemma. A few thoughts:

Case for keeping them separate:

The N-BEATS/NHiTS paper has a clean narrative: basis expansion and backbone compression in doubly-residual time series architectures, with rigorous benchmarks
The LLM application is a fundamentally different domain with different evaluation criteria, audiences, and related work
A focused paper is easier to review and harder to reject on scope
Case for folding LLaMA results in:

"These components transfer to a completely different architecture family" is a dramatically stronger contribution than "we improved N-BEATS"
NeurIPS reviewers value generality — a paper that shows the same architectural primitives work in both time series and language modeling is rare
Even a short section (not just appendix) showing preliminary LLM results could be the difference between "nice incremental work" and "novel architectural insight"
My suggestion: Include the NHiTS results in the main body (they're the same architecture family, same benchmarks — it's natural). For the LLaMA results, if you have even one solid comparison (e.g., hourglass MLP vs standard MLP in a transformer block, perplexity comparison), put it as a short Section 6 or 7 ("Preliminary Evidence of Transferability") rather than buried in an appendix. It doesn't need to be exhaustive — just enough to substantiate the claim. Save the full LLM study for the second paper.

With that framing, here are my final top title candidates:

Rethinking N-BEATS: Wavelet Bases and Compressed Backbones Expose Massive Overparameterization
— safe, grounded, lets the results speak

Wavelet Projections and Hourglass Backbones: Parameter-Efficient Building Blocks for Basis Expansion Architectures
— positions them as reusable components without overclaiming

Beyond Polynomial and Fourier: Orthonormal Wavelet Bases and Autoencoder Backbones for Doubly Residual Forecasting
— most descriptive, signals novelty on both fronts

N-BEATS Needs Fewer Parameters: Wavelet Bases and Compressed Backbones with Evidence of Transferability
— provocative lead, hedged transferability claim in subtitle

I'd lean toward #2 if you include even brief LLM results, or #1 if you keep the paper purely time series. Both fit the NeurIPS "in-depth analysis of existing methods" call well.
