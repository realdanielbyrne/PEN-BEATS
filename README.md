# N-BEATS Lightning — Wavelet and Autoencoder Block Extensions

A [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) implementation of [N-BEATS](https://arxiv.org/pdf/1905.10437.pdf) (Oreshkin et al., 2019) and an `NHiTSNet` variant, extended with wavelet and autoencoder block families. Published as [`lightningnbeats`](https://pypi.org/project/lightningnbeats/) on PyPI.

The companion research paper, *Wavelets and Autoencoders are All You Need*, lives at [NBEATS-Explorations/paper.md](NBEATS-Explorations/paper.md) and is the authoritative source for empirical findings, leaderboards, and ablation evidence. This README documents the library: what blocks exist, how to compose them, and where to find the protocols and reproducibility artifacts.

## What this package adds beyond N-BEATS

The original N-BEATS architecture pairs a 4-layer fully connected backbone with a basis-expansion head, organized into doubly residual stacks. This package keeps the doubly residual scaffold intact and extends three orthogonal axes:

1. **New basis families** — orthonormal Discrete Wavelet Transform (DWT) bases (`WaveletV3` and family-named subclasses), and composition blocks that fuse polynomial trend with wavelet bases (`TrendWavelet`) or with an additional learned generic branch (`TrendWaveletGeneric`).
2. **New backbones** — `AERootBlock` (hourglass encoder–decoder), `AERootBlockLG` (adds a learned sigmoid gate on the latent), and `AERootBlockVAE` (variational latent with KL regularization). These are drop-in replacements for the standard 4-layer trunk and are paired with every block family above.
3. **Composability** — `stack_types` is a free list of strings; any block can follow any other. Optional features include weight sharing, `active_g` (post-expansion activation), `skip_distance`/`skip_alpha` (ResNet-style cross-stack reinjection), `sum_losses`, asymmetric backcast/forecast wavelet families, and tiered SVD basis offsets.

The headline empirical claims from the paper are summarized once below; the paper has the matched-seed Wilcoxon tables, per-period leaderboards, and the strict health filter behind every number.

- AE-backbone variants reach **60–95% parameter reduction** versus paper-faithful Generic stacks at matched or better M4 sMAPE.
- `TrendWavelet*AELG` configurations at <1M parameters land within ~0.5% sMAPE of 19–43M paper baselines (12–80× compression) on most M4 periods.
- Among healthy configurations the doubly residual scaffold dominates accuracy variance — the basis family choice mostly trades compute, parameters, and stability rather than headline accuracy.
- See [paper.md §4–§5](NBEATS-Explorations/paper.md) for per-period winners, the parameter-efficiency frontier, and the `active_g` / wavelet-family / latent-dim / `basis_dim` ablations.

## Requirements

- **Python** ≥ 3.12, < 3.15
- **PyTorch** ≥ 2.1.0
- **Lightning** ≥ 2.1.0

Supports CUDA, Apple MPS, and CPU. `lightningnbeats.get_best_accelerator()` returns the highest-priority device available.

## Installation

```bash
pip install lightningnbeats         # from PyPI
# or, from source for development:
git clone https://github.com/realdanielbyrne/N-BEATS-Lightning.git
cd N-BEATS-Lightning
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### CUDA

PyPI hosts a CPU-only `torch` wheel. To use a CUDA build, install the package first, then upgrade `torch` from PyTorch's CUDA index:

```bash
pip install -r requirements-cuda.txt                       # CUDA 12.8 (recommended)
# or pick a specific build:
pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade
pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade
pip install torch --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

Match your driver at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

## Quickstart

```python
from lightningnbeats import NBeatsNet, get_best_accelerator
from lightningnbeats.loaders import TimeSeriesDataModule
import lightning.pytorch as pl
import pandas as pd

milk = pd.read_csv('src/data/milk.csv', index_col=0).values.flatten()
forecast_length, backcast_length = 6, 24

model = NBeatsNet(
    stack_types=['Trend', 'Seasonality'],
    backcast_length=backcast_length,
    forecast_length=forecast_length,
    n_blocks_per_stack=3,
    thetas_dim=5,
    share_weights=True,
)

dm = TimeSeriesDataModule(
    train_data=milk[:-forecast_length],
    val_data=milk[-(forecast_length + backcast_length):],
    backcast_length=backcast_length,
    forecast_length=forecast_length,
    batch_size=64,
    shuffle=True,
)

trainer = pl.Trainer(accelerator=get_best_accelerator(), max_epochs=100)
trainer.fit(model, datamodule=dm)
```

`stack_types` is required — there is no default architecture. Any combination of registered block names is legal; blocks compose freely in any order.

## Block reference

Every entry in [`constants.BLOCKS`](src/lightningnbeats/constants.py) is a valid `stack_types` string and maps to a concrete class in [`src/lightningnbeats/blocks/blocks.py`](src/lightningnbeats/blocks/blocks.py).

### Backbones

| Backbone | Topology | Notes |
|---|---|---|
| `RootBlock` | input → U → U → U → U | Paper-faithful 4-layer FC trunk. |
| `AERootBlock` | input → U/2 → d → U/2 → U | Hourglass; `latent_dim` controls bottleneck `d`. |
| `AERootBlockLG` | same as AE | Adds learned `sigmoid(γ) ⊙ z` gate; allows smaller `d` without accuracy loss. |
| `AERootBlockVAE` | μ/logσ heads, reparameterization | KL loss accumulated and added during `training_step`, scaled by `kl_weight` (default `0.1`). |

### Block families

Each backbone is paired with the same set of basis-expansion heads:

| Family | RootBlock | AERootBlock | AERootBlockLG | AERootBlockVAE |
| --- | --- | --- | --- | --- |
| Generic (paper) | `Generic` | `GenericAE` | `GenericAELG` | `GenericVAE` |
| Bottleneck Generic | `BottleneckGeneric` | `BottleneckGenericAE` | `BottleneckGenericAELG` | `BottleneckGenericVAE` |
| Backcast-only AE | `GenericAEBackcast` | `GenericAEBackcastAE` | `GenericAEBackcastAELG` | `GenericAEBackcastVAE` |
| Branch-specific AE | `AutoEncoder` | `AutoEncoderAE` | `AutoEncoderAELG` | `AutoEncoderVAE` |
| Trend (polynomial) | `Trend` | `TrendAE` | `TrendAELG` | `TrendVAE` |
| Seasonality (Fourier) | `Seasonality` | `SeasonalityAE` | `SeasonalityAELG` | `SeasonalityVAE` |
| Wavelet (DWT) | `WaveletV3` | `WaveletV3AE` | `WaveletV3AELG` | `WaveletV3VAE` |
| Trend + Wavelet | `TrendWavelet` | `TrendWaveletAE` | `TrendWaveletAELG` | — |
| Trend + Wavelet + Generic | `TrendWaveletGeneric` | `TrendWaveletGenericAE` | `TrendWaveletGenericAELG` | `TrendWaveletGenericVAE` |

`WaveletV3` is the recommended public API: pass `wavelet_type` (e.g. `'haar'`, `'db3'`, `'coif2'`, `'sym10'`). Family-named subclasses (`HaarWaveletV3`, `DB3WaveletV3`, `Coif2WaveletV3`, `Symlet10WaveletV3`, …) remain as compatibility wrappers and are listed in `BLOCKS`. V1/V2 wavelet families were removed for instability; V2 results are preserved under `experiments/results/` for reference.

For short forecast targets, prefer short-support wavelets (`haar`, `db2`, `db3`). When `pywt.dwt_max_level(...) == 0`, `_WaveletGeneratorV3` keeps `level=0` (approximation-only orthonormal basis) rather than forcing an invalid level-1 decomposition.

### Composing stacks

```python
# Pure generic (paper-equivalent at depth 30)
NBeatsNet(stack_types=['Generic'] * 30, ...)

# Interpretable (Trend + Seasonality)
NBeatsNet(stack_types=['Trend', 'Seasonality'], n_blocks_per_stack=3, share_weights=True, ...)

# Alternating Trend + Wavelet (RootBlock)
NBeatsNet(stack_types=['Trend', 'DB3WaveletV3'] * 5, ...)

# Unified TrendWavelet (single composite block)
NBeatsNet(stack_types=['TrendWavelet'] * 30, wavelet_type='sym10', trend_thetas_dim=3, basis_dim=H, ...)

# Sub-1M parameter compact model
NBeatsNet(
    stack_types=['TrendWaveletAELG'] * 10,
    wavelet_type='db3', trend_thetas_dim=3, basis_dim=H, latent_dim=32,
    share_weights=True, ...,
)

# Asymmetric wavelets for L >> H
NBeatsNet(
    stack_types=['WaveletV3AELG'] * 10,
    wavelet_type='coif2',
    backcast_wavelet_type='sym20', forecast_wavelet_type='coif2',
    ...,
)
```

> **Convention.** Place `Trend*` stacks before `Wavelet*` / `Generic*` / `Seasonality*` stacks. The block-ordering ablation (paper Appendix B.5) found Trend-first either neutral or strongly preferred on every dataset; reversing it can 2–3× the OWA on VAE backbones.

## Width and dimension parameters

| Parameter | Default | Applies to |
|---|---|---|
| `g_width` | 512 | Generic / BottleneckGeneric / GenericAEBackcast / Wavelet families |
| `s_width` | 2048 | Seasonality / SeasonalityAE |
| `t_width` | 256 | Trend / TrendAE / TrendWavelet / TrendWaveletGeneric families |
| `ae_width` | 512 | AutoEncoder / AutoEncoderAE |
| `latent_dim` | 5 | AE bottleneck width `d` (and branch latents in `AutoEncoderAE`, `GenericAEBackcastAE`) |
| `thetas_dim` | 3 | Reserved for explicit theta projections (`BottleneckGeneric`, `GenericAEBackcast*` forecast head) |
| `trend_thetas_dim` | 3 | Polynomial order in `Trend*` and `TrendWavelet*` families |
| `basis_dim` | 32 | Wavelet basis rank `k` per path |
| `forecast_basis_dim` | `None` | Optional asymmetric basis rank on the forecast path |
| `generic_dim` | — | Rank of the learned generic branch in `TrendWaveletGeneric*` |

For AE descendants, `latent_dim` controls the branch-local bottlenecks; `thetas_dim` is reserved for explicit basis-coefficient projections such as the `GenericAEBackcast*` forecast head.

## Selected extensions (cross-references to the paper)

These knobs are repository extensions and not part of Oreshkin et al. (2019). The paper's appendices give the empirical evidence; this section just describes the API.

- **`active_g`** — `False` (paper-faithful, recommended default), `True`, `'forecast'`, or `'backcast'`. Applies the block's nonlinearity to the basis-expansion outputs. The `'forecast'` mode helps decisively on M4-Hourly and rescues some Generic convergence failures, but is not a global default; see paper Appendix B.5.
- **`skip_distance` / `skip_alpha`** — ResNet-style reinjection of the original input every `skip_distance` stacks, scaled by `skip_alpha` (float or `'learnable'`). Useful for `GenericAELG` at depth ≥ 20; off by default. Detailed recommendation table in the paper's ResNet-skip ablation.
- **`sum_losses`** — Adds `0.25 · L(backcast, 0)` to the loss to push backcasts toward full reconstruction. Off by default; reserved for the convergence study (paper Appendix D).
- **Asymmetric wavelet families** — `backcast_wavelet_type` and `forecast_wavelet_type` override `wavelet_type` per path. Useful when `L ≫ H` (e.g., Traffic-96).
- **Tiered basis offsets** (`basis_offset`) — Cycle each stack through a different SVD spectrum slice of the orthonormal wavelet basis. Strongest on M4-Daily under paper-sample; regresses on Weekly.

## Custom losses

In addition to the standard `torch.nn` losses, the package registers `SMAPELoss`, `MAPELoss`, `MASELoss` (accepts `seasonal_period`), and `NormalizedDeviationLoss`. Select by name via the `loss=` argument.

## Datasets

`lightningnbeats.data` ships dataset adapters that subclass `BenchmarkDataset` and expose a uniform `train_data` / `test_data` / `forecast_length` / `frequency` / `name` interface, plus `compute_owa()` where Naive2 baselines are available:

- **M4** — bundled CSVs for all six periods (Yearly / Quarterly / Monthly / Weekly / Daily / Hourly). OWA supported.
- **Tourism** — Yearly / Quarterly / Monthly. Columnar layout.
- **Traffic-96** and **Weather-96** — downloaded from Hugging Face on first use, cached at `~/.cache/lightningnbeats/`. Parameterized by horizon.
- **Milk** — single-series univariate.

Use the matching `DataModule` for the layout: `RowCollectionTimeSeriesDataModule` (M4-style row-oriented), `ColumnarCollectionTimeSeriesDataModule` (Tourism/Traffic/Weather-style), or `TimeSeriesDataModule` (single series).

## Experiments

Reproducing or extending the paper's sweeps is driven by YAML through [`experiments/run_from_yaml.py`](experiments/run_from_yaml.py):

```bash
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml --dry-run
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml
python experiments/run_from_yaml.py experiments/configs/genericae_pure.yaml          # successive halving
python experiments/run_from_yaml.py experiments/configs/trendae_ae_alternating.yaml
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml --analyze-only # no retraining
```

The launcher supports two-pass studies, atomic claim files for parallel workers, search/successive-halving rounds, custom CSV columns, and full CLI overrides. Schema reference: [`experiments/configs/schema.md`](experiments/configs/schema.md). Canonical training-protocol YAMLs (sliding, paper-sample MultiStepLR, paper-sample plateau) and their reproducibility CSVs are listed in [paper.md Appendix B.9](NBEATS-Explorations/paper.md).

Examples in [`examples/`](examples/) (`#%%` cell markers, runnable as scripts or in Jupyter):

```bash
python examples/M4AllBlks.py
python examples/TourismAllBlks.py
```

## Testing

```bash
pytest tests/                         # all
pytest tests/test_blocks.py -v        # block shapes / registries
```

Conventions and troubleshooting: [TESTING.md](TESTING.md). CI does not run tests; no linter or formatter is configured.

## Adding a new block

1. Create the class in [`src/lightningnbeats/blocks/blocks.py`](src/lightningnbeats/blocks/blocks.py), inheriting from `RootBlock`, `AERootBlock`, `AERootBlockLG`, or `AERootBlockVAE`. `forward()` must return `(backcast, forecast)`.
2. Add the class name to `BLOCKS` in [`constants.py`](src/lightningnbeats/constants.py).
3. If a new width parameter is needed, extend the dispatch in `NBeatsNet.create_stack()`.
4. Add a shape test; the parametrized `TestAllBlocksOutputShapes` picks up new entries automatically.

## Citation

If the wavelet / AE block families or the parameter-efficiency results are useful in your work, please cite *Wavelets and Autoencoders are All You Need* — see [NBEATS-Explorations/paper.md](NBEATS-Explorations/paper.md) for the canonical text and reproducibility pointers.
