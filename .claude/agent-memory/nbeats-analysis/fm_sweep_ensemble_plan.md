---
name: fm-sweep-ensemble-plan
description: Post-omnibus experiment plan — FM sweep (3-7) on top 3-5 omnibus winners for lookback-diverse ensembling, replicating paper methodology.
type: project
---

## FM Sweep + Ensemble Experiment (planned, pending omnibus completion)

**Why:** The N-BEATS paper used 180-model ensembles with diversity from different lookback windows. Our omnibus uses fixed FM per dataset. A lookback-diverse ensemble (FM=3,4,5,6,7) would provide less-correlated predictions than seed-only ensembles, closer to the paper's methodology.

**How to apply:** After all 4 omnibus benchmarks complete:
1. Identify top 3-5 configs per dataset by mean SMAPE/MSE
2. Build a FM-sweep YAML: FM={3,4,5,6,7} × 10 seeds × top configs
3. Ensemble via element-wise median across all 50 members per config (5 FM × 10 seeds)
4. Compare single-FM-best vs seed-only ensemble vs FM-diverse ensemble vs paper ensemble
5. The omnibus already saves predictions (`save_predictions: true`), so same-FM seed ensembles can be computed from existing omnibus data without retraining

**Gate fn study evidence:** FM5 vs FM7 produce meaningfully different predictions (not just noise) — real diversity exists across lookback lengths.

**Scale estimate:** 3-5 configs × 5 FM × 10 seeds × 4 datasets = 600-1000 runs.
