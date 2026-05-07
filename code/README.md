# Anonymous Code Supplement — Wavelets and Autoencoders are All You Need

This is the anonymized code archive accompanying the NeurIPS 2026 submission
*Wavelets and Autoencoders are All You Need*. It contains the
`lightningnbeats` package (model + block implementations), the YAML-driven
experiment launcher, the YAML configurations for every experiment cited in
the paper, the convergence-tracking analysis script, and the pytest test
suite.

## Repository layout

```
code/
├── pyproject.toml                           Package metadata (anonymized)
├── requirements.txt                         Pip-installable dependencies
├── src/lightningnbeats/                     Library: blocks, models, loaders, losses
│   ├── blocks/blocks.py                     All block implementations
│   ├── models.py                            NBeatsNet + NHiTSNet
│   ├── loaders.py                           Lightning DataModules
│   ├── losses.py                            sMAPE / MAPE / MASE / ND
│   ├── constants.py                         Block / activation / optimizer registries
│   └── data/                                Dataset loaders + bundled small datasets
├── experiments/
│   ├── run_from_yaml.py                     Unified YAML-driven launcher (recommended)
│   ├── run_unified_benchmark.py             Underlying benchmark harness
│   ├── tools/                               MetaForecaster + LLM commentary helpers
│   ├── configs/                             All YAML configs cited in the paper
│   └── analysis/scripts/m4_overall_leaderboard.py   Cross-period leaderboard builder
└── tests/                                   Pytest suite
```

## Installation

Python 3.12 ≤ x < 3.15. PyTorch ≥ 2.1 (CUDA, Apple MPS, or CPU).

```bash
cd code
python -m venv .venv && source .venv/bin/activate  # or your preferred env
pip install -r requirements.txt
pip install -e .
```

## Datasets

Small datasets used in the paper (Tourism, Milk, the bundled univariate
demonstration series) are included under `src/lightningnbeats/data/`.
Traffic and Weather are downloaded automatically from Hugging Face on first
use and cached in `~/.cache/lightningnbeats/`.

The M4 raw series (≈ 250 MB) are **not** bundled, to keep this archive under
the 100 MB NeurIPS supplementary limit. To run any M4 experiment, download
the official M4 competition data from the M4 organisers (Makridakis et al.,
2020) and place the files at:

```
src/lightningnbeats/data/M4/
├── M4-info.csv
├── Train/{Yearly,Quarterly,Monthly,Weekly,Daily,Hourly}-train.csv
└── Test/{Yearly,Quarterly,Monthly,Weekly,Daily,Hourly}-test.csv
```

(File names and the column layout are exactly those of the original
competition release.)

## Running experiments

All paper experiments are driven by `run_from_yaml.py` and a YAML config.
Run from inside `code/` so the relative `experiments/configs/...` and
`experiments/results/...` paths resolve correctly.

```bash
# Smoke test (Milk dataset, 1 run, 2 epochs — does not need M4 data)
python experiments/run_from_yaml.py experiments/configs/milk_convergence.yaml \
    --max-epochs 2 --n-runs 1

# Quick check of an M4 config (needs M4 data installed; 1 run, few epochs)
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml \
    --periods Yearly --max-epochs 5 --n-runs 1

# Full reproduction of paper-sample results (long; multi-day on a single GPU)
python experiments/run_from_yaml.py experiments/configs/comprehensive_m4_paper_sample.yaml
python experiments/run_from_yaml.py experiments/configs/comprehensive_m4_paper_sample_plateau.yaml
python experiments/run_from_yaml.py experiments/configs/tiered_offset_m4_allperiods.yaml
python experiments/run_from_yaml.py experiments/configs/m4_hourly_sym10_tiered_offset.yaml

# Sliding-window protocol
python experiments/run_from_yaml.py experiments/configs/comprehensive_sweep_m4.yaml
```

Useful flags for `run_from_yaml.py`:

| Flag | Effect |
| --- | --- |
| `--dry-run` | Print the resolved config plan without training |
| `--analyze-only` | Re-run post-hoc analysis on existing results CSV |
| `--n-runs N` | Override seed budget |
| `--max-epochs N` | Override epoch budget |
| `--periods Yearly Monthly` | Restrict to selected M4 periods |
| `--dataset {m4,tourism,traffic,weather,milk}` | Override dataset |
| `--wandb` | Enable W&B logging (requires `pip install wandb`) |

Resumability is built in: re-running the same config after an interrupt skips
already-completed `(config, dataset, period, seed)` rows.

## Config index for the experiments cited in the paper

| Cited section / table | Config file |
| --- | --- |
| Section 4.1 (M4 paper-sample) | `comprehensive_m4_paper_sample.yaml` |
| Section 4.1 (M4 paper-sample, plateau LR) | `comprehensive_m4_paper_sample_plateau.yaml` |
| Section 4.1 (M4 sliding sweep) | `comprehensive_sweep_m4.yaml` |
| Section 4.1 / Appendix B (tiered offset) | `tiered_offset_m4_allperiods.yaml`, `tiered_offset_m4_allperiods_paperlr.yaml` |
| Section 4.1 (M4-Hourly tiered) | `m4_hourly_sym10_tiered_offset.yaml`, `m4_hourly_sym10_tiered_offset_paperlr.yaml` |
| Table 5 (Tourism) | `omnibus_benchmark_tourism.yaml`, `comprehensive_sweep_tourism.yaml` |
| Table 5 (Traffic) | `comprehensive_sweep_traffic.yaml` |
| Table 5 (Weather) | `comprehensive_sweep_weather.yaml` |
| Table 5 (Milk) | `unified_benchmark_milk.yaml`, `milk_convergence.yaml` |
| Section 4.4 (NHiTS transferability) | `nhits_benchmark_weather.yaml`, `nhits_novel_ae_weather.yaml` |
| Appendix B.9 (cross-period leaderboard) | `experiments/analysis/scripts/m4_overall_leaderboard.py` |

A complete schema for the YAML format is in
`experiments/configs/schema.md`.

## Reproducing leaderboard tables

After running the relevant configs, regenerate the per-period leaderboard
with:

```bash
python experiments/analysis/scripts/m4_overall_leaderboard.py
```

This consumes the result CSVs under `experiments/results/m4/` and writes a
leaderboard markdown file mirroring the structure of Appendix B.

## Tests

```bash
pytest tests/
pytest tests/test_blocks.py -k "TestAllBlocksOutputShapes"
```

Tests cover block output shapes, model construction, datamodule splits, and
the YAML launcher's stack-spec parsing.

## Notes for reviewers

- The package version in `pyproject.toml` is set to `0.0.0` for anonymization;
  the pre-anonymization release version is intentionally not disclosed.
- All author, institutional, repository-URL, and email metadata have been
  scrubbed. If any oversight remains, please flag it via the OpenReview
  comment system.
- Hardware: every experiment in the paper was run on a single consumer GPU
  (16–24 GB VRAM) or Apple Silicon (MPS). Compact configurations
  (sub-1M parameter blocks) train comfortably on CPU.
