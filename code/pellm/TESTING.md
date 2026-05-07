# TESTING

This repository currently uses Python's built-in `unittest` test runner.
At the moment, the repository has a single test module in `tests/test_trend_wavelet_linear.py`.

## Prerequisites

From the repository root:

1. Create and activate a virtual environment if you do not already have one.
2. Install the package in editable mode so the tests import the local `pellm` code.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> If you also want to run the training script experiments, install the optional training extras:
>
> ```bash
> pip install -e ".[train]"
> ```

## Run the current test suite

Because the repository currently has one test module, this command runs the full test suite:

```bash
source .venv/bin/activate
python -m unittest tests.test_trend_wavelet_linear -v
```

## Run a single test case

```bash
source .venv/bin/activate
python -m unittest \
  tests.test_trend_wavelet_linear.TrendWaveletLinearProjectionTests \
  -v
```

## What is currently covered

The existing tests in `tests/test_trend_wavelet_linear.py` verify that
`TrendWaveletLinear.from_pretrained_linear(...)` matches the legacy pseudo-inverse
behavior for:

- impulse-response reconstruction
- least-squares reconstruction residuals
- pretrained truncation behavior for standard and reduced TrendWavelet layers
- generic-branch zeroing for standard and reduced generic TrendWavelet layers
- learned-gate initialization and finite structured-init weights for reduced variants

## Notes

- Run commands from the repository root.
- These tests rely on the core package dependencies such as `torch` and `PyWavelets`.
- There is currently no separate `pytest` configuration in this repository; `unittest` is the canonical runner.
- `python -m unittest discover ...` is not currently the most reliable option here because the `tests/`
  directory is not set up as an importable package.
