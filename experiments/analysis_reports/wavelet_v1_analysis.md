==========================================================================================
  WAVELET V1 BLOCKS — DEPRECATION ANALYSIS & LESSONS LEARNED
==========================================================================================

  Source:       NBEATS-Explorations/paper.md, Section 5.1.3
  Status:       DEPRECATED — V1 wavelet blocks removed from codebase
  Replacement:  WaveletV2 (stabilized) and WaveletV3 (orthonormal DWT)
  CSV data:     wavelet_v2_benchmark_results.csv, wavelet_v3_benchmark_results.csv
                (V1 results never saved to CSV; all runs produced divergent metrics)

==========================================================================================
  1. OVERVIEW
==========================================================================================

  Wavelet V1 blocks implemented a naive learned-downsampling wavelet basis without
  numerical stabilization. The basis matrix values could exceed representable ranges
  after multiplication with randomly initialized weights, leading to immediate or
  gradual numerical failure across all M4 periods (Yearly, Quarterly, Monthly).

  Result: V1 was removed and replaced by V2 (numerical stabilization) and V3
  (orthonormal DWT bases with no learned downsampling).

==========================================================================================
  2. V1 FAILURE MODES (M4-Yearly, Quarterly, Monthly — 3 seeds each)
==========================================================================================

  Config           Yearly    Quarterly  Monthly    Failure Mode
  ---------------  --------  ---------  ---------  ---------------------------------
  HaarWavelet      NaN 3/3   NaN 3/3    NaN 3/3    Immediate (epoch 1) NaN
  DB3AltWavelet    NaN 3/3   NaN 3/3    NaN 3/3    Immediate (epoch 1) NaN
  DB3Wavelet       Div 2/3   Div 3/3    Div 2/3    MASE explosion (10^10 – 10^31)
  Symlet3Wavelet   Div 3/3   Div 3/3    Div 2/3    MASE explosion (identical to DB3)
  Coif2Wavelet     Div 3/3   Div 3/3    Div 2/3    MASE explosion
  Generic+DB3Wav   1/3 hlthy 0/3 hlthy  0/3 hlthy  Progressive instability

  NOTE: DB3Wavelet and Symlet3Wavelet produced *identical* divergent MASE values on
  matching seeds, indicating they share the same numerical instability pathway.

  Healthy run definition: OWA < 2.0 AND MASE < 10^6 AND non-NaN output.

==========================================================================================
  3. ROOT CAUSE ANALYSIS
==========================================================================================

  Two distinct failure modes were identified:

  A. IMMEDIATE NaN FAILURE (HaarWavelet, DB3AltWavelet):
     - NaN loss at epoch 1, across all seeds and periods.
     - Cause: The V1 basis matrix construction produces values that cause overflow
       in the forward pass before any training signal can correct the weights.
     - The basis matrix (especially for discrete wavelets with sharp transitions
       like Haar) creates large-magnitude coefficients after matrix multiplication
       with zero-mean random init weights.

  B. GRADUAL MASE EXPLOSION (DB3Wavelet, Symlet3Wavelet, Coif2Wavelet):
     - Training proceeds for 11+ epochs but forecasts grow to astronomical values.
     - sMAPE saturates at 200 (theoretical maximum).
     - MASE values reach up to 10^31 — unrepresentable in float32.
     - Consistent across seeds, indicating a deterministic instability pathway
       triggered by the wavelet basis structure rather than random initialization.

==========================================================================================
  4. EXCEPTION: TREND+DB3WAVELET
==========================================================================================

  The one partial success from V1 was the hybrid Trend+DB3Wavelet configuration:
  - Achieved FULL CONVERGENCE on both Yearly (OWA ≈ 0.809) and Quarterly (OWA ≈ 0.896).
  - Monthly data was partial (wavelet component still unstable at higher frequencies).
  - Interpretation: The stable Trend stack absorbs early gradient instability,
    providing a well-conditioned training signal that prevents the wavelet component
    from diverging. This validated the concept of wavelet basis expansion while
    confirming that standalone wavelet blocks require additional stabilization.

  This result directly motivated the V2 and V3 wavelet designs.

==========================================================================================
  5. DESIGN CHANGES IN V2 AND V3
==========================================================================================

  WaveletV2 changes (stabilized):
  - Explicit numerical stabilization: basis values clamped/normalized before use
  - Learned downsampling retained but guarded against overflow
  - Square basis (WaveletV2) and rectangular basis (AltWaveletV2) variants
  - Result: Eliminated immediate NaN failures; mixed configs fully converge

  WaveletV3 changes (orthonormal DWT):
  - Replaced learned downsampling with proper orthonormal DWT basis matrices
  - Bases computed analytically from wavelet filter coefficients (non-trainable)
  - Registered as buffers (no gradient through basis matrix)
  - Optional forecast_basis_dim parameter for asymmetric backcast/forecast dims
  - Result: Best OWA of all wavelet variants; achieved OWA < 0.800 on M4-Yearly

  V1 was removed from the codebase entirely (no migration path).
  Users should use WaveletV2 or WaveletV3 variants exclusively.

==========================================================================================
  6. V2 AND V3 BENCHMARK SUMMARY (M4-Yearly)
==========================================================================================

  See wavelet_v2_v3_analysis.md for the full comparison.

  V2 best OWA (Yearly):     ~0.797  (HaarWaveletV2)
  V2 worst OWA (Yearly):    ~2.569  (diverged configs)
  V3 best OWA (Yearly):     ~0.791  (DB20WaveletV3 + TrendAE stack)
  V3 worst OWA (Yearly):    ~2.664  (diverged configs)

  Published baselines (M4-Yearly, 30-stack):
    NBEATS-I+G      OWA = 0.8057
    NBEATS-G        OWA = 0.8198

  V3 top configs beat the NBEATS-I+G baseline when paired with a Trend or TrendAE stack.

==========================================================================================
  7. RECOMMENDATION
==========================================================================================

  - DO NOT use any V1 wavelet block. The class no longer exists in the codebase.
  - USE WaveletV3 variants for highest accuracy (orthonormal DWT, stable, interpretable).
  - USE WaveletV2 variants as lightweight alternatives where V3 basis dim is mismatched.
  - ALWAYS pair wavelet blocks with at least one stable Trend or Generic stack in
    a multi-stack architecture to guard against residual initialization instability.
  - AVOID pure-wavelet-only architectures at all frequencies until further stabilization
    research is completed.

==========================================================================================
  8. CODE REFERENCES
==========================================================================================

  - Block implementations:  src/lightningnbeats/blocks/blocks.py
  - V2 classes:  HaarWaveletV2, DB2WaveletV2, DB3WaveletV2, Coif2WaveletV2,
                 Symlet3WaveletV2, DB3AltWaveletV2 (all inherit RootBlock)
  - V3 classes:  HaarWaveletV3, DB2WaveletV3, DB3WaveletV3, DB4WaveletV3,
                 Coif2WaveletV3, Coif3WaveletV3, Symlet3WaveletV3,
                 Symlet10WaveletV3, DB10WaveletV3, DB20WaveletV3 (inherit RootBlock)
  - Paper notes: NBEATS-Explorations/paper.md, Section 5.1.3

  Basis generators:
    _WaveletGeneratorV2:     square basis, learned downsampling + stabilization
    _AltWaveletGeneratorV2:  rectangular basis, direct output + stabilization
    _WaveletGeneratorV3:     orthonormal DWT basis (buffer, non-trainable)

==========================================================================================
  9. CONCLUSIONS
==========================================================================================

  Wavelet V1 is not a recoverable production option in this repository. The observed
  failure modes (immediate NaNs and deterministic MASE explosions) are structural to
  the old basis construction approach and were reproduced across periods and seeds.

  The deprecation decision is therefore supported by both training stability evidence
  and replacement maturity: WaveletV2 addressed numerical issues, and WaveletV3
  provides the strongest accuracy and interpretability among wavelet variants.

==========================================================================================
  10. RECOMMENDED PARAMETER CONFIGURATIONS
==========================================================================================

  Since V1 is removed, recommendations target supported replacements:

  - Primary default: use a WaveletV3 + Trend/TrendAE hybrid (e.g., Coif2/DB-family)
    with trend_thetas_dim=3 as the first search setting.
  - Secondary default: use WaveletV2 + Trend when migrating older pipelines that
    rely on V2-compatible basis behavior.
  - Avoid standalone pure-wavelet stacks as a first choice; include at least one
    stable Trend or Generic stack in mixed-stack compositions.

