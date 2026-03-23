---
name: training_dynamics_unified_vs_comprehensive
description: Findings from comparing unified benchmark (max100/pat10) vs comprehensive sweep (max200/pat20/LR warmup) training dynamics. Explains divergence differences and optimal epoch settings.
type: project
---

## Training Dynamics: Unified vs Comprehensive Sweep (2026-03-22)

See `experiments/analysis/analysis_reports/unified_vs_comprehensive_training_dynamics.md`
See notebook: `experiments/analysis/notebooks/unified_vs_comprehensive_training_dynamics.ipynb`

### Three Compounding Factors Explain Performance Gap

1. **LR warmup (most impactful)**: Prevents bimodal convergence. M4-Yearly mild divergence drops from 1.7% (UB) to 0.2% (CS). Affected configs: BottleneckGeneric, standalone Wavelet blocks at 30 stacks.
2. **patience=20 vs 10**: 47% of M4-Yearly runs find a better minimum with patience=20. Mean improvement 2.25% val_loss. Alternating Trend+Wavelet configs benefit in 70%+ of runs.
3. **max_epochs=200 vs 100**: Only 3.2% of runs have best_epoch > 100. Mainly multi-component architectures (NBEATS-I+G).

**Why:** Understanding this prevents misattributing performance differences to architecture when they are actually training dynamics artifacts.

**How to apply:** Always use max_epochs=200, patience=20, warmup=15 for new experiments. If comparing to old unified benchmark results, account for the ~3.8% average SMAPE penalty from the inferior training settings.

### Architecture Sensitivity Rankings (M4-Yearly)

- **Most sensitive to extended training:** Alternating Trend+WaveletV3 RootBlock (70% improvement rate)
- **Moderately sensitive:** RootBlock backbones generally (52.6%), GenericAE (60%)
- **Least sensitive:** TrendWaveletAELG (40.5%, <1% improvement). AELG bottleneck constrains search space.
- **NBEATS-G (Generic):** Converges fast, insensitive to training dynamics (~0% delta)

### Dataset-Specific Optimal Settings

- **Tourism:** patience=10 sufficient (5.8% benefit rate). best_epoch ~13. max_epochs=100 is fine.
- **M4:** patience=20 essential (47% benefit rate). best_epoch ~39.
- **Weather/Traffic:** Standard settings (max200/pat20).
- **Milk:** Divergence-dominated. active_g=forecast helps stability more than training dynamics.

### Unified Benchmark CSV Data Quality Issue

466/1479 rows in `unified_benchmark_results.csv` have column alignment issues: `epochs_trained` contains 'False' (shifted from `diverged`), `stopping_reason` contains numeric val_loss values. These rows still have valid SMAPE/OWA but missing epoch metadata. Filter with `pd.to_numeric(epochs_trained, errors='coerce')`.
