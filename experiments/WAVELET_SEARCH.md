# Wavelet Hyperparameter Search

Automated hyperparameter search for wavelet block configurations using successive
halving and an N-BEATS meta-forecaster. Finds optimal `basis_dim`, `basis_offset`,
wavelet family, stack size, and architecture pattern through short training runs.

## Quick Start

```bash
# Full search pipeline (all 4 rounds, M4-Yearly)
python experiments/run_wavelet_study.py --mode search --dataset m4

# Full search on all datasets
python experiments/run_wavelet_study.py --mode search --dataset all
```

## How It Works

### Successive Halving

The search runs 4 rounds of progressively longer training, pruning the bottom
67% of configs after each round:

| Round | Epochs | Seeds | Configs (approx) | Purpose |
|-------|--------|-------|-------------------|---------|
| 1 | 6 | 1 | ~2,200 (full grid) | Coarse screen |
| 2 | 15 | 1 | ~730 (top 33%) | Medium screen |
| 3 | 30 | 3 | ~240 (top 33%) | Fine screen with variance |
| 4 | 100 | 5 | ~80 (top 33%) | Full validation + active_g |

Config counts vary by dataset. M4-Yearly has ~189 configs (most offsets are
invalid for forecast_length=6). Traffic-96 and Weather-96 have ~2,205.

### N-BEATS Meta-Forecaster

Before round 1, a tiny N-BEATS model (Generic x3, width=64) is trained on
val_loss curves from prior experiments. It learns typical training dynamics
(steep initial descent, gradual plateau). After round 1, it takes each config's
6-epoch val_loss curve and predicts future training performance, producing a
`predicted_best` score used for ranking.

Training data comes from existing CSVs in `experiments/results/` (unified
benchmark, block benchmark, convergence studies). The trained model is cached
at `experiments/results/.meta_cache/meta_forecaster.ckpt`.

### Parameter Space

| Parameter | Values | Notes |
|-----------|--------|-------|
| Wavelet family | Haar, DB2, DB3, DB4, DB10, Coif2, Coif3, Sym3, Sym10 | 9 families |
| basis_dim | 4, 8, 16, 32, 48, 64, 96 | Clamped to available SVD rows |
| basis_offset | 0, 8, 16, 32, 48 | Filtered by forecast_length |
| Stack size | 10, 20, 30 | Number of stacks |
| Architecture | homogeneous, trend_mixed, trend_season_mixed | Mixing patterns |

Invalid combinations are automatically filtered (e.g., basis_offset=16 with
M4-Yearly's forecast_length=6 would leave <2 effective dimensions).

## Running Round by Round

Run each round individually to inspect results between rounds:

```bash
# Round 1: coarse scan
python experiments/run_wavelet_study.py --mode search --dataset weather --round 1

# Analyze round 1 results (no training)
python experiments/run_wavelet_study.py --mode search --dataset weather --round 1 --analyze

# Round 2: auto-reads round 1 results, promotes top 33%
python experiments/run_wavelet_study.py --mode search --dataset weather --round 2

# Round 3
python experiments/run_wavelet_study.py --mode search --dataset weather --round 3

# Round 4: full validation with active_g="forecast" pass
python experiments/run_wavelet_study.py --mode search --dataset weather --round 4
```

## Overrides

```bash
# Keep more configs between rounds
--top-k 100

# Override epoch count for a round
--search-max-epochs 10

# Override seeds per config for a round
--search-n-runs 2

# Smoke test (2 epochs, 1 seed — validates the pipeline)
python experiments/run_wavelet_study.py --mode search --dataset m4 --round 1 \
    --search-max-epochs 2 --search-n-runs 1
```

## Output

Results are saved to `experiments/results/<dataset>/wavelet_search_results.csv`
with these extra columns beyond the standard benchmark schema:

- `search_round` — which round produced this result (1-4, or 4_ag)
- `arch_pattern` — homogeneous, trend_mixed, or trend_season_mixed
- `wavelet_family` — full block type name
- `meta_predicted_best` — meta-forecaster's predicted minimum val_loss
- `meta_convergence_score` — predicted_best / initial_loss (lower = better)

After each round, a ranking summary and marginal analysis are printed showing
the effect of each parameter dimension on performance.

## Estimated Runtime

Per dataset, single GPU:

| Dataset | Round 1 | Round 2 | Round 3 | Round 4 | Total |
|---------|---------|---------|---------|---------|-------|
| M4-Yearly | ~3.5h | ~3h | ~6h | ~13h | ~25h |
| Traffic-96 | ~20h | ~15h | ~35h | ~75h | ~145h |
| Weather-96 | ~20h | ~15h | ~35h | ~75h | ~145h |

Scale linearly with GPU count. On 4 GPUs, M4-Yearly completes in ~6h total.
Traffic/Weather benefit most from multi-GPU (`--n-gpus 4`).

## Existing Study Mode

The original 16-config wavelet study is unchanged and remains the default:

```bash
python experiments/run_wavelet_study.py --dataset m4 --periods Yearly
```
