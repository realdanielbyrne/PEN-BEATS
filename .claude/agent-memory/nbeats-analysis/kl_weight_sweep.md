---
name: KL Weight Sweep Findings
description: Optimal kl_weight for VAE blocks is 0.001, confirmed on M4-Yearly and Weather-96. Non-monotonic on M4 (V-shape). Double-VAE still fails at low kl.
type: project
---

## KL Weight Sweep (2026-03-19)

**kl_weight=0.001 is the confirmed optimal default for AERootBlockVAE.**

- See `experiments/analysis/analysis_reports/kl_weight_sweep_analysis.md`
- See notebook: `experiments/analysis/notebooks/kl_weight_sweep_insights.ipynb`

### Key findings:
- **M4-Yearly:** kl=0.001 SMAPE=13.571 beats kl=0.1 (old default) SMAPE=14.294 (MWU p=0.004, d=2.45). Beats ALL other tested values including kl=0.0001.
- **Weather-96:** kl=0.001 MSE=0.148 vs kl=1.0 (old hardcoded) MSE=0.198 (+34%). Monotonic within tested range (0.001-1.0). Not tested below 0.001.
- **Non-monotonic on M4:** kl=0.0001 is worse than kl=0.001 (p=0.048). There IS a sweet spot, not just "lower is better."
- **Higher kl_weight increases variance:** Spearman rho=0.83 (p=0.04) between log(kl) and SMAPE std on M4. kl=0.001 gives lowest variance.
- **Double-VAE NOT rehabilitated:** TrendVAE+HaarWavV3VAE at kl=0.001 SMAPE=14.654 vs single-VAE 13.571 (p=0.004). Equivalent to single-VAE at kl=0.1.
- **VAE+AELG mix not helpful:** TrendVAE+HaarWavV3AELG SMAPE=13.766 vs pure single-VAE 13.571 (p=0.048). Deterministic wavelet is better paired with VAE trend.

**Why:** kl_weight=0.001 balances KL regularization (prevents degenerate posteriors) against reconstruction fidelity (the primary N-BEATS objective). Higher values force posterior collapse; lower values lose regularization benefit.

**How to apply:** Always set `block_params: {kl_weight: 0.001}` for any config using VAE blocks. When pairing TrendVAE with a wavelet block, use deterministic wavelet (HaarWaveletV3, not VAE or AELG variant). Prior VAE results at kl>=0.1 should be discounted -- they tested at a suboptimal hyperparameter.
