---
name: NHiTS Protocol Benchmark Findings
description: Results from NHiTS-protocol benchmark comparing novel block types against NHiTS baselines on Weather/Traffic (2026-03-11)
type: project
---

## NHiTS-Protocol Benchmark (Weather, Traffic) (2026-03-11)

- See `experiments/analysis/analysis_reports/nhits_benchmark_analysis.md`
- See notebook: `experiments/analysis/notebooks/nhits_benchmark_analysis.ipynb`
- **590 rows, 18 configs, 2 datasets, 4 horizons (96/192/336/720), 8 seeds, NHiTS evaluation protocol**
- **Protocol:** Z-score norm (Weather), 70/10/20 split, MSE loss, L=5H, patience=10, batch_size=256

### Key Results

- **We beat NHiTS at Weather H=192:** NHiTS-GenericAE (MSE=0.198, 0.938x) and BnGenericAE-10 (MSE=0.210, 0.993x)
- **We beat NHiTS at Weather H=336 by 22.7%:** BottleneckGenericAELG-10 (MSE=0.210 vs NHiTS 0.272)
- **We narrowly miss at H=96:** NHiTS-GenericAELG (MSE=0.165, 1.047x)
- **We badly miss at H=720:** All configs > 1.45x NHiTS. All SMAPE > 100. Undertrained (patience fires at 11 epochs).

### Architecture Crossover

- **NHiTSNet wins at H=96/192** (hierarchical pooling is efficient regularization at short horizons)
- **NBeatsNet wins at H=336/720** (flat residual preserves temporal resolution at long horizons)
- GenericAELG matched comparison: NHiTS wins significantly at H=96 (p=0.001), NBeats wins at H=720 (p=0.065)

### Cross-Horizon Best

- **BottleneckGenericAELG-10** (NBeatsNet): avg rank 3.5/14 -- most robust across all horizons
- **NHiTS-GenericAELG** (NHiTSNet): avg rank 4.0 -- best at short horizons, collapses at H=720 (rank 11)

### AE vs AELG on Weather

- Non-factor. AELG wins 17/28 comparisons but only 1 is significant. No consistent advantage.
- Consistent with prior M4 findings. Use AELG as default (no downside) but gate is not differentiating.

### Traffic (Preliminary)

- Only H=96 (24 runs) and H=192 (2 runs) collected. Experiment incomplete.
- MSE values (~0.001) are ~400x lower than NHiTS paper (~0.4). Metric reporting difference (per-variate vs joint).
- Internal ranking: TrendAELG+Sym20V3AELG leads but all configs within 6%.

### H=720 Failure Analysis

- 72% of runs have SMAPE > 100. Not divergence -- early stopping fires too soon.
- Best: TrendWavelet-10 (MSE=0.506, only 3 runs, 11 epochs trained).
- Needs: patience=25, learning rate warm-up, complete runs.

### Next Experiments

1. H=720 recovery: increase patience, disable early stopping for training curves
2. NHiTSNet + BottleneckGenericAELG blocks (untested combination)
3. NHiTSNet pooling ablation at H=720 (gentler pooling: kernel=[4,2,1])
4. Complete Traffic experiment
5. Run active_g configs (YAMLs exist, no results yet)
