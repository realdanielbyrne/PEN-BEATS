# Anonymous Code Supplement — Wavelets and Autoencoders are All You Need

This is the anonymized code archive accompanying the NeurIPS 2026 submission
*Wavelets and Autoencoders are All You Need*. It contains two independent
Python packages, each with its own dependency set, experiment launcher,
configurations, and pytest suite. The split keeps the time-series
environment free of HuggingFace `transformers` and vice versa.

## Layout

```text
code/
├── README.md                  This file
├── nbeats/                    Time-series forecasting package
│   ├── pyproject.toml         nbeats_anon metadata (anonymized)
│   ├── requirements.txt       Pip-installable dependencies (forecasting only)
│   ├── conftest.py            Adds src/ to sys.path for pytest
│   ├── README.md              Detailed install / data / experiment guide
│   ├── src/nbeats_anon/       Library: blocks, models, loaders, losses
│   ├── experiments/           run_from_yaml.py + all paper YAML configs
│   └── tests/                 Pytest suite for the forecasting package
└── pellm/                     Transformer experiments package
    ├── pyproject.toml         pellm metadata (anonymized)
    ├── README.md              Detailed install / data / experiment guide
    ├── pellm/                 Library: PE-Llama, custom attention/MLP layers
    ├── scripts/               finetune.py, run_from_yaml.py, eval scripts
    │   └── experiments/       YAML configs for transformer experiments
    ├── tests/                 Pytest suite for the transformer package
    ├── docs/                  PELLM design notes
    └── evals/                 Pre-computed lm-eval-harness results
```

## Which package covers which paper section

| Paper section / table | Package |
| --- | --- |
| Sections 3–4, Appendix B (M4, Tourism, Traffic, Weather, Milk, NHiTS) | `nbeats/` |
| Section on parameter-efficient Llama (PELLM) | `pellm/` |

## Installation

The two packages are installed independently into separate virtual
environments. Each subdirectory contains its own README with the full set
of commands, data-access notes, smoke tests, and reproduction recipes.

### Forecasting (`nbeats/`)

Python 3.12–3.14, PyTorch ≥ 2.1.

```bash
cd nbeats
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -v        # validate install
```

Full instructions, the M4 data download recipe, and the YAML-driven
experiment commands are in [`nbeats/README.md`](nbeats/README.md).

### Transformer (`pellm/`)

Python ≥ 3.10, PyTorch ≥ 2.1, Transformers ≥ 4.40.

```bash
cd pellm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[experiments]"
pytest tests/ -v        # validate install
```

Full instructions, the Hugging Face / Llama 3.2 access steps, and the
PELLM experiment commands are in [`pellm/README.md`](pellm/README.md).

## Notes for reviewers

- All author, institutional, repository-URL, and email metadata have been
  scrubbed across both packages. If any oversight remains, please flag it
  via the OpenReview comment system.
- Hardware: every experiment in the paper was run on a single consumer GPU
  (16–24 GB VRAM) or Apple Silicon (MPS). Sub-1M-parameter forecasting
  configurations also train comfortably on CPU.
- Archive size: well below the 100 MB NeurIPS supplementary limit. Heavy
  artefacts (raw experiment logs, training notebooks, model checkpoints,
  M4 raw CSVs) are not bundled; the per-package READMEs document how to
  download or regenerate them.
