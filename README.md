# N-BEATS Architecture Explorations and the Impact of Block Type on Performance

A [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) implementation of the [N-BEATS](https://arxiv.org/pdf/1905.10437.pdf) (Neural Basis Expansion Analysis for Time Series) forecasting architecture, extended with novel block types and published as the [`lightningnbeats`](https://pypi.org/project/lightningnbeats/) PyPI package.

## Introduction

Time series forecasting is one of the oldest and most consequential problems in quantitative science. For decades, the field was dominated by classical statistical methods -- exponential smoothing, ARIMA, and their many variants -- that offered strong theoretical grounding, interpretability, and reliable performance. The M4 competition (Makridakis et al., 2018; 2020) marked a turning point: while pure machine learning entries ranked poorly, the hybrid ES-RNN winner (Smyl, 2020) demonstrated that deep learning could contribute meaningfully to forecasting accuracy, leaving open the question of whether a *pure* deep learning architecture could achieve competitive results.

Oreshkin et al. (2019) answered this question with N-BEATS, a fully deep learning architecture that surpassed both the M4 winner and all prior statistical methods. N-BEATS introduced a distinctive design built on **doubly residual stacking** of basic blocks, each consisting of a multi-layer fully connected network that forks into backcast and forecast paths via learned or constrained basis expansion coefficients. The architecture offered two configurations: a **Generic** model using fully learnable basis functions, and an **Interpretable** model constraining basis functions to polynomial (Trend) and Fourier (Seasonality) forms.

This observation motivates the present work: if polynomial and Fourier bases can achieve state-of-the-art results when embedded within the N-BEATS doubly residual framework, what happens when we substitute alternative basis expansions? This repository provides a systematic exploration of alternative block types including:

- **Wavelet basis expansion blocks** (V3 variants: Haar, Daubechies, Coiflets, Symlets) offering multi-resolution time-frequency localization
- **Autoencoder blocks** with separated encoder-decoder paths for data-driven compressed representations
- **Bottleneck generic blocks** with rank-*d* factorized projections for parameter-efficient basis expansion
- **AE-backbone variants** (`AERootBlock`, `AERootBlockLG`, `AERootBlockVAE`) replacing the uniform FC-layer backbone with a hourglass encoder-decoder structure that achieves 5–10× parameter reduction
- **Learned-gate autoencoder blocks** (AELG variants) with an adaptive sigmoid-gated latent bottleneck that learns to selectively suppress or amplify compressed representations

A **key finding** from systematic benchmarks is that the original N-BEATS architecture (Oreshkin et al., 2019) was **significantly over-parameterized**. Our novel AE-backbone block types achieve 60–95% parameter reductions compared to the paper's NBEATS-G baseline (24.7M parameters) while matching or improving forecasting accuracy (see [Section 4.1 Table 2 of the research paper](NBEATS-Explorations/paper.md)):

| Configuration | Parameters | Reduction vs NBEATS-G | M4-Yearly OWA |
|---|---|---|---|
| NBEATS-G (paper baseline) | 24.7M | — | 0.820 |
| GenericAE | 4.8M | **81%** | **0.808** |
| BottleneckGenericAE | 4.3M | **83%** | **0.806** |
| NBEATS-I-AE (TrendAE + SeasonalityAE) | 2.2M | **91%** | **0.805** |

These AE-backbone variants match or outperform the 24.7M-parameter NBEATS-G baseline while using a fraction of the parameters, with direct implications for deployment in memory- or latency-constrained environments.

Among healthy, converging configurations, **block type does not produce statistically significant differences in OWA forecasting accuracy** (Kruskal-Wallis p > 0.09 across all periods tested). Configuration rankings are inconsistent across periods (Spearman rho near zero), confirming that the doubly residual stacking framework itself — rather than the specific basis expansion — is the primary driver of accuracy. Block type *does* significantly affect parameter count (5–10× variation), training stability (0–100% convergence rate for wavelets), and convergence speed, making deployment constraints the practical basis for block selection.

A **convergence study** (Part 6) across M4-Yearly and Weather-96 with 50 random seeds per configuration reveals that `active_g` eliminates catastrophic initialization sensitivity on M4-Yearly (reducing sMAPE coefficient of variation from 31.4% to 0.9%). The recommended default is `active_g='forecast'` (forecast-path activation only), which achieves 100% convergence reliability while recovering significant expressiveness compared to full `active_g=True`.

See [NBEATS-Explorations/paper.md](NBEATS-Explorations/paper.md) for the full research paper.

## N-BEATS Algorithm

N-BEATS (Oreshkin et al., 2019) is composed of blocks organized into stacks. Each basic block accepts an input lookback window and outputs a **backcast** (reconstruction of the input) and a **forecast** (prediction of future values). The core computation begins with four fully connected layers with ReLU activations, producing a hidden representation that is then projected to expansion coefficients via linear layers. These coefficients are passed through basis functions to produce the outputs.

The key innovation is the **doubly residual topology** inspired by deep residual learning (He et al., 2016): each block's backcast is subtracted from its input before passing to the next block (backward residual connection, enabling iterative signal decomposition), while each block's forecast is summed into the global forecast (forward residual connection, enabling hierarchical forecast aggregation). The final forecast is the sum of all blocks' partial forecasts.

- **Block Architecture**: Each block consists of a multi-layer FC backbone followed by a fork into backcast and forecast paths via basis expansion. The choice of basis function determines the block type.

- **Generic and Interpretable Configurations**: The Generic configuration uses fully learnable linear projections as basis functions. The Interpretable configuration constrains bases to polynomial (Trend) and Fourier (Seasonality) forms, producing decomposable forecasts.

- **Weight Sharing**: When enabled within a stack, all blocks share parameters, improving validation performance particularly for the Interpretable architecture.

- **Fast Learning**: N-BEATS converges quickly, typically within 10-30 epochs, making it practical to train ensembles of diverse models. The original paper achieved its best results with a 180-model ensemble combining multiple loss functions, backcast lengths, and random seeds.

## Requirements

- **Python** >= 3.12
- **PyTorch** >= 2.1.0
- **Lightning** >= 2.1.0

## Getting Started

### Installation

Download the source from the [github repository](https://github.com/realdanielbyrne/N-BEATS-Lightning) or install it as a pip package to use in your project using the following command:

```bash
pip install lightningnbeats
```

To install from source for development:

```bash
git clone https://github.com/realdanielbyrne/N-BEATS-Lightning.git
cd N-BEATS-Lightning
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### GPU / CUDA Installation

`pip install` and `pyproject.toml` always resolve `torch` from PyPI, which only hosts the CPU-only build. PyTorch's CUDA-enabled wheels are hosted on a separate index and must be installed explicitly **after** the base install.

**Step 1 — install the package as normal:**

```bash
pip install -e .          # from source
# or
pip install lightningnbeats  # from PyPI
```

**Step 2 — upgrade torch to the CUDA build:**

```bash
# CUDA 12.8 (recommended — compatible with drivers 525+ including CUDA 13.x drivers)
pip install -r requirements-cuda.txt

# Or install torch directly for a specific CUDA version:
# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

Check available builds and match your driver at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

**Verify CUDA is detected:**

```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # e.g. NVIDIA GeForce RTX 4070
```

### Simple Example

The following is a simple example of how to use this model.

#### Load Data

First load the required libraries and your data.

```python
# Import necessary libraries
from lightningnbeats import NBeatsNet
from lightningnbeats.loaders import *
import pandas as pd

# Load the milk.csv dataset
milk = pd.read_csv('src/data/milk.csv', index_col=0)
milkvals = milk.values.flatten()
```

#### Define the model and its hyperparameters

Define the model by defining the architecture in the `stack_types` parameter.  The `stack_types` parameter is simply a list of strings that specify the type of block to use in each stack.  The following block types are available:

- Generic
- BottleneckGeneric
- GenericAE
- BottleneckGenericAE
- GenericAEBackcast
- GenericAEBackcastAE
- Trend
- TrendAE
- Seasonality
- SeasonalityAE
- AutoEncoder
- AutoEncoderAE
- HaarWaveletV3
- DB2WaveletV3
- DB3WaveletV3
- DB4WaveletV3
- DB10WaveletV3
- DB20WaveletV3
- Coif1WaveletV3
- Coif2WaveletV3
- Coif3WaveletV3
- Coif10WaveletV3
- Symlet2WaveletV3
- Symlet3WaveletV3
- Symlet10WaveletV3
- Symlet20WaveletV3

Note: `WaveletV2`/`AltWaveletV2` block families were removed due to instability. Historical experiment results are retained under `experiments/results/` for reference.

This implementation extends the design original paper with several additional block types and by allowing any combination blocks in any order simply by specifying the block types in the stack_types parameter.

```python
forecast_length = 6
backcast_length = 4 * forecast_length
batch_size = 64
n_stacks = 6

interpretable_milkmodel = NBeatsNet(
  stack_types=['Trend', 'Seasonality'],
  backcast_length = backcast_length,
  forecast_length = forecast_length,
  n_blocks_per_stack = 3,
  thetas_dim = 5,
  t_width=256,
  s_width=2048,
  share_weights = True
)
```

This model will forecast 6 steps into the future. The common practice is to use a multiple of the forecast horizon for the backcast length.  In this case, we will use 4 times the forecast horizon.

Larger batch sizes will result in faster training, but may require more memory.  The number of blocks per stack is a hyperparameter that can be tuned.  The share_weights parameter is set to True to share weights across the blocks. Gerneally deeper stacks do not result in more accurate predictions as the model saturates pretty quickly.  To improve accuracy it is best to build moultiple models from differnt architectures and combine them in an ensemble by taking the forecast from each and using something like the mean or median of the combined results.

#### Define a Pytorch Lightning DataModule

Instantiate one of the predefined PyTorch Lightning Time Series Data Modules in this repository to help organize and load your data.

- *TimeSeriesDataModule* - A PyTorch Lightning DataModule that takes a univariate time series as input and returns batches of samples of the time series. This is the most basic DataModule and is useful for single univariate time series data.
- *RowCollectionTimeSeriesDataModule* - A PyTorch Lightning DataModule accepts a dataset that is a collection of time series organized into rows where each row represents a time series, and each column represents subsequent observations. For instance this is how the M4 dataset is organized.
- *ColumnarCollectionTimeSeriesDataModule* - A PyTorch Datamodule that takes a collection of time series as input and returns batches of samples. The input dataset is a collection of time series organized such that columns represent individual time series and rows represent subsequent observations. This is how the Tourism dataset is organized.

```python
dm = TimeSeriesDataModule(
  train_data = milkvals[:-forecast_length],
  val_data = milkvals[-(forecast_length + backcast_length):],
  batch_size = batch_size,
  backcast_length = backcast_length,
  forecast_length = forecast_length,
  shuffle = True
)

```

#### Define a Pytorch Lightning ModelCheckpoint (optional)

Define a Pytorch Ligntning ModelCheckpoint callback to save the best model during training.  The model checkpoints will be saved to the default  `lightning/logs` directory unless otherwise specified.

```python
i_chk_callback = ModelCheckpoint(
  save_top_k = 2, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard logger
i_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=i_name)

```

#### Define a Pytorch Lightning Trainer and train the model

Use a standard Pytorch Lightning Trainer to train the model.  Be careful not to overtrain, as performance will begin to degrade after a certain point.  The N-BEATS architecture tends to settly fairly quickly into a relative minimum.   If the model is not performing well, try a different architecture or hyperparameters.  It is best to build an ensemble of models and combine the results to improve accuracy.  This can be done by taking the mean or median of the forecast from each model in the ensemble.

```python
interpretable_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=100
  ,callbacks=[i_chk_callback]
  ,logger=[i_tb_logger]
)

interpretable_trainer.fit(interpretable_milkmodel, datamodule=dm)
interpretable_trainer.validate(interpretable_milkmodel, datamodule=dm)
```

#### Hardware Acceleration (GPU Support)

This package supports multiple hardware accelerators. The PyTorch Lightning Trainer with `accelerator='auto'` will automatically select the best available device in this priority order: **CUDA GPU** > **Apple MPS** > **CPU**.

| Accelerator | Platform | Notes |
| --- | --- | --- |
| **CUDA** | NVIDIA GPUs (Linux/Windows) | Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and the matching [PyTorch CUDA build](https://pytorch.org/get-started/locally/) |
| **MPS** | Apple Silicon (macOS) | Metal Performance Shaders — available out of the box on M1/M2/M3/M4 Macs with PyTorch >= 2.1.0 |
| **CPU** | Any | Fallback when no GPU is available |

You can check which accelerator is available on your system:

```python
from lightningnbeats import get_best_accelerator
print(get_best_accelerator())  # Returns 'cuda', 'mps', or 'cpu'
```

Or check individually:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available:  {torch.backends.mps.is_available()}")
```

The `get_best_accelerator()` utility function returns a string compatible with Lightning's `accelerator` parameter:

```python
from lightningnbeats import NBeatsNet, get_best_accelerator
import lightning.pytorch as pl

accelerator = get_best_accelerator()
trainer = pl.Trainer(accelerator=accelerator, max_epochs=100)
```

### Running Examples

Example scripts are in the `examples/` directory and can be run as standalone scripts or cell-by-cell in Jupyter (using `#%%` markers):

```bash
cd examples
python M4AllBlks.py       # M4 dataset benchmark across all block types
python TourismAllBlks.py  # Tourism dataset benchmark
```

### Running Tests

Tests use pytest:

```bash
pytest tests/              # run all tests
pytest tests/ -v           # verbose output
pytest tests/test_blocks.py  # run a specific test file
```

## N-BEATS Extensions and Variations

This repository provides an implementation of N-BEATS that has been extended to include extended features which can be used to augment the basic design with more advanced features.

### Active G (`active_g`)

When enabled, `active_g` applies the block's activation function (default: ReLU) to the basis expansion outputs after the final linear layer of each block — i.e., to `g^b(θ^b)` and `g^f(θ^f)`. This parameter is not in the original N-BEATS paper.

The parameter supports four modes:

- `active_g=False` — paper-faithful, no post-expansion activation (default)
- `active_g=True` — activation on both backcast and forecast paths
- `active_g='forecast'` — activation on forecast path only (**recommended for production**)
- `active_g='backcast'` — activation on backcast path only (not recommended)

**Convergence study findings** (50–200 random seeds per configuration across M4-Yearly, Tourism-Yearly, Weather-96, and Milk datasets):

- **Baseline** (`active_g=False`): best per-seed accuracy, but 7–30% of seeds fail catastrophically on M4-Yearly (sMAPE coefficient of variation: 31.4%). When seeds converge they produce the most accurate models.
- **Balanced** (`active_g=True`): eliminates convergence failures (100% success rate); sMAPE CV drops to 0.9% on M4-Yearly. Accuracy cost: ~1% on large multi-series benchmarks, up to 76% on small univariate datasets.
- **Forecast-only** (`active_g='forecast'`): matches the 100% convergence rate of balanced activation while recovering ~13% of the expressiveness gap. **This is the recommended default** for single-run production deployments. It also trains fastest (7.5s vs 9.5s on the Milk benchmark).
- **Backcast-only** (`active_g='backcast'`): worst mode — reduces convergence rate *below* the unconstrained baseline (63% vs 70%) while constraining expressiveness. Avoid this mode.

The core intuition: ReLU on forecast outputs prevents catastrophic cancellation in the forecast accumulation sum (∑ ŷ_ℓ), smoothing the loss landscape. ReLU on backcast outputs forces monotonically decreasing residuals — removing the ability to produce negative corrective backcasts — which destabilizes the doubly residual decomposition chain. Forecast-only activation captures the stability benefit while preserving residual expressiveness.

On large multi-series benchmarks, the ~1% accuracy gap from `active_g='forecast'` is smaller than the ~3% gain from median ensembling across seeds, making it a net positive in production pipelines. On small univariate datasets, prefer `active_g=False` with multiple seeds and best-run selection.

### BottleneckGeneric Block

The `BottleneckGeneric` block is a variant of the paper's Generic block that uses a two-stage projection through an intermediate `thetas_dim` bottleneck instead of a single linear projection. This is equivalent to a rank-d factorization of the basis expansion matrix, where `d = thetas_dim`. The bottleneck regularizes the learned basis by limiting its rank, providing a tunable knob to control basis complexity — analogous to how Trend and Seasonality blocks use low-dimensional parameterizations (polynomial degree, number of Fourier harmonics) to constrain the function space.

A corresponding `BottleneckGenericAE` variant uses the AERootBlock backbone instead of the standard RootBlock.

```python
n_stacks = 5
stack_types = ['BottleneckGeneric'] * n_stacks
stack_types = ['Trend', 'BottleneckGeneric'] * n_stacks
```

### Wavelet Basis Expansion Blocks

This repository contains a number of experimental Wavelet Basis Expansion Blocks. Wavelet basis expansion is a mathematical technique used to represent signals or functions in terms of simpler, fixed building blocks called wavelets. Unlike Fourier transforms, which use sine and cosine functions as basis elements, wavelets can be localized in both time and frequency. This means they can represent both the frequency content of a signal and when these frequencies occur.

This method is particularly useful for analyzing functions or signals that contain features at multiple scales.  The multi-resolution analysis capability of wavelets is particularly suited to capturing the essence of time series data then, which can have complex, hierarchical structures due to the presence of trends, seasonal effects, cycles, and irregular fluctuations.

Wavelet blocks can be used in isolation or in combination with other blocks freely. V1 and V2 wavelet blocks were removed due to instability (NaN failures and MASE explosion). Use V3 variants. Historical V2 benchmark results are kept in `experiments/results/` for reference. For instance:

```python
n_stacks = 5
stack_types = ['DB3WaveletV3'] * n_stacks # 5 stacks of DB3WaveletV3 blocks
stack_types = ['Trend','DB3WaveletV3'] * n_stacks # 5 stacks of 1 Trend and 1 DB3WaveletV3
stack_types = ['DB3WaveletV3','Generic'] # 5 stacks of 1 DB3WaveletV3 followed by 1 Generic
```

The Wavelet blocks available in this repository are as follows:

- HaarWaveletV3
- DB2WaveletV3
- DB3WaveletV3
- DB4WaveletV3
- DB10WaveletV3
- DB20WaveletV3
- Coif1WaveletV3
- Coif2WaveletV3
- Coif3WaveletV3
- Coif10WaveletV3
- Symlet2WaveletV3
- Symlet3WaveletV3
- Symlet10WaveletV3
- Symlet20WaveletV3

For short targets, prefer shorter-support families such as `HaarWaveletV3`, `DB2WaveletV3`, or `DB3WaveletV3`. Longer filters (`DB20`, `Symlet20`, `Coif10`) need substantially longer backcast/forecast windows to expose real multilevel structure. When PyWavelets reports `dwt_max_level(...) == 0`, WaveletV3 now respects `level=0` instead of forcing an invalid level-1 decomposition, which avoids boundary-effect warnings while keeping the basis orthonormal.

### AutoEncoder Block

The AutoEncoder Block utilizes an AutoEncoider structure in both the forecast and backcast branches in the N-BEATS architecture.  The AutoEncoder block is useful for noisey time series data like Electric generation or in highly varied datasets like the M4.   It struggles with simpler more predictable datasets like the Milk Production Dataset taht rely on mostly trend.  However, combining this block with a Trend block often eliminates this problem.

Like any other blocks in this implementation, the AutoEncoder block can be used in isolation or in combination with other blocks freely. For instance

```python
n_stacks = 5
stack_types = ['AutoEncoder'] * n_stacks # 5 stacks of AutoEncoder blocks
stack_types = ['Trend','AutoEncoder'] * n_stacks # 5 stacks of 1 Trend block followed by 1 AutoEncoder block
```

### GenericAEBackcast

The GenericAEBackcast block is a Generic block that uses an AutoEncoder structure in only the backcast branch of the N-BEATS architecture.  This block is useful for noisey time series data like Electric generation or in highly varied datasets like the M4.   It doesn't struggle like the AutoEncoder block does with simpler more predictable datasets like the Milk Production Dataset.  It is genreally more accurate than the AutoEncoder block, and it settles on a solution faster.

```python
n_stacks = 5
stack_types = ['GenericAEBackcast'] * n_stacks # 5 stacks of GenericAEBackcast blocks
stack_types = ['Trend','GenericAEBackcast'] * n_stacks # 5 stacks of 1 Trend and 1 GenericAEBackcast block
```

### AERootBlock and its Variations

The standard N-BEATS backbone uses four equal-width fully connected layers before the signal splits into backcast and forecast branches. `AERootBlock` replaces this with a hourglass encoder-decoder structure: input → `units/2` → `latent_dim` → `units/2` → `units`. With `latent_dim=4` and `units=512`, the bottleneck compresses the representation to just 4 dimensions, delivering 5–10× parameter reduction compared to the standard backbone.

Three AE backbone variants are available:

**`AERootBlock`** — Standard hourglass encoder-decoder:

- `GenericAE`, `BottleneckGenericAE`, `GenericAEBackcastAE`
- `TrendAE`, `SeasonalityAE`, `AutoEncoderAE`

**`AERootBlockLG`** — Learned-gate backbone: adds a learnable `nn.Parameter` gate vector of size `latent_dim`. Applies `sigmoid(gate) × z` after the latent layer, enabling adaptive suppression or amplification of individual latent dimensions:

- `GenericAELG`, `BottleneckGenericAELG`, `GenericAEBackcastAELG`
- `TrendAELG`, `SeasonalityAELG`, `AutoEncoderAELG`

**`AERootBlockVAE`** — Variational AE backbone: stochastic latent space with `fc2_mu`/`fc2_logvar` heads and reparameterization trick. KL divergence loss (weight 0.001) is accumulated and added during `training_step()`:

- `GenericVAE`, `BottleneckGenericVAE`, `GenericAEBackcastVAE`
- `TrendVAE`, `SeasonalityVAE`, `AutoEncoderVAE`

For AE-style descendants, `latent_dim` controls the local autoencoder bottlenecks while `thetas_dim` remains reserved for explicit theta/basis projections where they still exist. In practice this means `AutoEncoderAE`/`AutoEncoderAELG`/`AutoEncoderVAE` and `GenericAEBackcastAE`/`GenericAEBackcastAELG`/`GenericAEBackcastVAE` compress branch-local paths with `latent_dim`, while `GenericAEBackcast*` forecast heads still project through `thetas_dim`.

All three backbone variants can be used in isolation or freely mixed:

```python
n_stacks = 5
stack_types = ['GenericAE'] * n_stacks                       # 5 stacks of GenericAE blocks
stack_types = ['TrendAE', 'GenericAE'] * n_stacks            # alternating TrendAE and GenericAE
stack_types = ['TrendAELG', 'Coif2WaveletV3'] * n_stacks     # learned-gate + wavelet (V3)
```

### Parameter Efficiency

A central finding from our systematic benchmarks is that the original N-BEATS architecture (Oreshkin et al., 2019) was significantly **over-parameterized**. AE-backbone block types deliver 60–95% parameter reductions while matching or improving M4-Yearly OWA (full results in [Section 4.1 Table 2 of the research paper](NBEATS-Explorations/paper.md)):

| Configuration | Parameters | Reduction | M4-Yearly OWA | Notes |
|---|---|---|---|---|
| NBEATS-G (paper baseline) | 24.7M | — | 0.820 | 30-stack Generic |
| GenericAE | 4.8M | 81% | 0.808 | 30-stack AE-backbone Generic |
| BottleneckGenericAE | 4.3M | 83% | 0.806 | Best cross-period stability |
| NBEATS-I-AE | 2.2M | 91% | 0.805 | TrendAE + SeasonalityAE (2×3 stacks) |
| TrendWaveletAE / TrendWaveletAELG | ~8–9M | ~65% | Competitive | Mixed interpretable-wavelet |

`BottleneckGenericAE` offers the best balance of efficiency and cross-period generalization, never ranking below 5th across Yearly, Quarterly, and Monthly M4 periods. `NBEATS-I-AE` achieves the highest parameter efficiency (91% reduction) but its extreme 4-dimensional latent bottleneck becomes limiting for higher-frequency series with more complex seasonal patterns.

```python
# Parameter-efficient 30-stack model: 83% fewer parameters than NBEATS-G, matching OWA
model = NBeatsNet(
    stack_types=['BottleneckGenericAE'] * 30,
    backcast_length=30,
    forecast_length=6,
    n_blocks_per_stack=1,
    share_weights=True,
    latent_dim=4,
)
```

### Custom Loss Functions

In addition to standard PyTorch loss functions (MSELoss, L1Loss, SmoothL1Loss, etc.), this implementation provides several custom loss functions commonly used in time series forecasting:

- **`SMAPELoss`** — Symmetric Mean Absolute Percentage Error. Scale-independent metric that treats over- and under-predictions symmetrically.
- **`MAPELoss`** — Mean Absolute Percentage Error. Measures forecast accuracy as a percentage of the true values.
- **`MASELoss`** — Mean Absolute Scaled Error. Compares forecast errors against a naive seasonal baseline. Accepts a `seasonal_period` parameter.
- **`NormalizedDeviationLoss`** — Normalized Deviation (ND). Ratio of total absolute error to total absolute true values.

Specify the loss function by name when creating the model:

```python
model = NBeatsNet(
  stack_types=['Trend', 'Seasonality'],
  backcast_length=24,
  forecast_length=6,
  loss='SMAPELoss'  # or 'MAPELoss', 'MASELoss', 'NormalizedDeviationLoss', 'MSELoss', etc.
)
```

## Upcoming Work

NHiTS-architecture benchmarks using the same parameter-efficient AE-backbone block types are in progress. These experiments evaluate both `NBeatsNet` (10-stack) and `NHiTSNet` (3-stack with hierarchical pooling and multi-rate signal sampling) configurations with AELG block variants against NHiTS published baselines on the Weather and Traffic datasets across four forecast horizons (96, 192, 336, 720 steps), using the NHiTS evaluation protocol (Z-score normalization, 70/10/20 train/val/test split, MSE loss). Configuration files:

- [`experiments/configs/nhits_benchmark_weather.yaml`](experiments/configs/nhits_benchmark_weather.yaml)
- [`experiments/configs/nhits_benchmark_traffic.yaml`](experiments/configs/nhits_benchmark_traffic.yaml)

## Architecture Naming

The parameter-efficient AE-backbone block family represents a substantive departure from the original N-BEATS formulation — 60–95% parameter reduction while matching accuracy across diverse benchmarks — and warrants a distinctive name for publication and community reference. The following candidates apply to both N-BEATS and NHiTS base architectures:

| Name | Rationale |
|---|---|
| **N-BEATS-AE** / **NHiTS-AE** | Directly maps to the architectural contribution (`AERootBlock`). Short, technically precise, and cleanly namespaces within the existing N-BEATS/NHiTS family while signaling the encoder-decoder innovation. Works well as a paper identifier: *N-BEATS-AE: Parameter-Efficient Neural Basis Expansion via Autoencoder Backbones*. |
| **LeanBeats** | A standalone portmanteau emphasizing parameter leanness and the N-BEATS foundation. Works across variants (`LeanBeats-G`, `LeanBeats-I`, `LeanNHiTS`) without relying on the original trademark-adjacent capitalization. Memorable and distinctive in literature search. |
| **N-BEATS-Compact** / **NHiTS-Compact** | Signals the compression insight unambiguously. Natural as a paper title suffix: *N-BEATS-Compact: 90% Parameter Reduction with Matched Forecasting Accuracy*. Immediately communicates practical value to deployment-focused practitioners. |
| **N-BEATS v2** / **NHiTS v2** | Simplest versioning approach, making lineage explicit while signaling a systematic improvement. Most appropriate if the work is framed as a direct successor to Oreshkin et al. (2019) / Challu et al. (2023) rather than a novel architectural family. |
| **AE-BEATS** | Foregrounds the autoencoder innovation first, signaling a broader architectural rethinking rather than a minor variant. Most distinctive in literature search. Best suited if the backbone redesign — rather than forecasting accuracy per se — is the primary contribution being claimed. |
