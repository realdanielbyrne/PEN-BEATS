# nbeats_anon — Time-Series Forecasting Package

This sub-directory contains the `nbeats_anon` package: N-BEATS / NHiTS
forecasting library used for all time-series experiments
(Sections 3–4, Appendix B) of *Wavelets and Autoencoders are All You Need*.

The companion transformer experiments (Llama attention/MLP replacement) live
in the sibling directory `../pellm/` and are installed separately. See the
top-level `code/README.md` for the overall archive layout.

## Layout

```text
nbeats/
├── pyproject.toml                       Package metadata (anonymized)
├── requirements.txt                     Pip-installable dependencies
├── conftest.py                          Adds src/ to sys.path so pytest works without install
├── src/nbeats_anon/                     Library: blocks, models, loaders, losses
│   ├── blocks/blocks.py                 All block implementations
│   ├── models.py                        NBeatsNet + NHiTSNet
│   ├── loaders.py                       Lightning DataModules
│   ├── losses.py                        sMAPE / MAPE / MASE / ND
│   ├── constants.py                     Block / activation / optimizer registries
│   └── data/                            Dataset loaders + bundled small datasets
├── experiments/
│   ├── run_from_yaml.py                 Unified YAML-driven launcher (recommended)
│   ├── run_unified_benchmark.py         Underlying benchmark harness
│   ├── tools/                           MetaForecaster ensemble helper
│   ├── configs/                         All YAML configs cited in the paper
│   └── analysis/scripts/m4_overall_leaderboard.py   Cross-period leaderboard builder
└── tests/                               Pytest suite
```

## Installation

Requirements: Python 3.12–3.14, PyTorch ≥ 2.1 (CUDA, Apple MPS, or CPU).

```bash
cd code                                        # all commands run from here
python -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                               # installs nbeats_anon in editable mode
```

**Quick verification** (no M4 data needed):

```bash
python experiments/run_from_yaml.py experiments/configs/milk_convergence.yaml \
    --n-runs 1 --max-epochs 2 --dry-run   # prints plan only
python experiments/run_from_yaml.py experiments/configs/milk_convergence.yaml \
    --n-runs 1 --max-epochs 2             # trains 2 configs × 1 run (~5 s on CPU)
```

## Datasets

Small datasets used in the paper (Tourism, Milk, the bundled univariate
demonstration series) are included under `src/nbeats_anon/data/`.
Traffic and Weather are downloaded automatically from Hugging Face on first
use and cached in `~/.cache/nbeats_anon/`.

The M4 raw series (≈ 250 MB) are **not** bundled, to keep this archive under
the 100 MB NeurIPS supplementary limit.

### Adding the M4 dataset (required for M4 experiments only)

**Step 1 — Download** from the official M4 competition repository:
`https://github.com/Mcompetitions/M4-methods/tree/master/Dataset`

Download `M4-info.csv`, all six `*-train.csv` files (under `Dataset/Train/`),
and all six `*-test.csv` files (under `Dataset/Test/`).

**Step 2 — Place files** into the expected layout inside `code/`:

```text
src/nbeats_anon/data/M4/
├── M4-info.csv
├── Train/
│   ├── Yearly-train.csv
│   ├── Quarterly-train.csv
│   ├── Monthly-train.csv
│   ├── Weekly-train.csv
│   ├── Daily-train.csv
│   └── Hourly-train.csv
└── Test/
    ├── Yearly-test.csv
    ├── Quarterly-test.csv
    ├── Monthly-test.csv
    ├── Weekly-test.csv
    ├── Daily-test.csv
    └── Hourly-test.csv
```

File names and column layout must match the original competition release
exactly (row-per-series, NaN-padded). No preprocessing is required.

**Step 3 — Verify** the loader resolves the data:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from nbeats_anon.data.M4 import M4Dataset
ds = M4Dataset("Yearly")
print(f"Yearly: {ds.train_data.shape[1]} series, H={ds.forecast_length}")
EOF
```

## Running experiments

All paper experiments are driven by `run_from_yaml.py` and a YAML config.
Run from inside `code/` so the relative `experiments/configs/...` and
`experiments/results/...` paths resolve correctly.

```bash
# Smoke test — Milk dataset, no M4 data required
python experiments/run_from_yaml.py experiments/configs/milk_convergence.yaml \
    --max-epochs 2 --n-runs 1

# M4 paper-sample baseline (requires M4 data; 1 run, 5 epochs — for a quick check)
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml \
    --periods Yearly --max-epochs 5 --n-runs 1

# Full reproduction — paper-sample protocol (long; multi-day on a single GPU)
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

The four core test modules cover: block output shapes and registry completeness
(`test_blocks.py`), DataModule splits (`test_loaders.py`), model construction
and optimizer dispatch (`test_models.py`), and YAML launcher protocol plumbing
(`test_run_from_yaml.py`).

```bash
# Run all tests (after pip install -e .)
pytest tests/ -v

# Block shapes only — fast, no GPU needed
pytest tests/test_blocks.py -k "TestAllBlocksOutputShapes" -v

# Tests also work without pip install (conftest.py adds src/ to sys.path)
PYTHONPATH=src pytest tests/ -v
```

## Notes for reviewers

- The package version in `pyproject.toml` is set to `0.0.0` for anonymization;
  the pre-anonymization release version is intentionally not disclosed.
- All author, institutional, repository-URL, and email metadata have been
  scrubbed. If any oversight remains, please flag it via the OpenReview
  comment system.
- Hardware: every experiment in the paper was run on a single consumer GPU
  (16–24 GB VRAM) or Apple Silicon (MPS). Compact configurations
  (sub-1M parameter blocks) train comfortably on CPU.
