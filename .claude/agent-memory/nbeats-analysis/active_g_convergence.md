---
name: active_g convergence study findings
description: active_g eliminates bimodal convergence failures in Generic blocks at small quality cost; activeG=True and activeG="forecast" are equivalent; stuck rate scales with dataset size not depth
type: project
---

## Active_G Convergence Study (2026-03-17)

- See `experiments/analysis/analysis_reports/active_g_convergence_study_analysis.md`
- See notebook: `experiments/analysis/notebooks/active_g_convergence_study.ipynb`
- **1,599 runs across Milk (6/10 stacks), M4-Yearly (30 stacks), Tourism-Yearly (30 stacks)**

**Why:** Generic blocks have a bimodal convergence failure where ~63% of runs on simple datasets (Milk) get stuck at flat-prediction saddle points. active_g eliminates this.

**How to apply:**
- Use `active_g="forecast"` as the default for all Generic block configurations
- Expect ~0.6% quality cost on multi-series datasets (M4, Tourism) vs converged baseline
- Expect ~48% quality cost on single univariate series (Milk) -- the regularization is too strong for trivially simple signals
- `activeG=True` and `activeG="forecast"` are statistically equivalent (p>0.08 on both M4 and Tourism with n=200)
- Stuck rate scales with dataset size (63% for 1 series, 4.5% for 23K series, 1.5% for 518 series), NOT with stack depth
- SMAPE variance reduction: 93-98% across all datasets
- For Milk specifically, only `active_g=True` has been tested; `active_g="forecast"` may have lower quality cost (untested)
