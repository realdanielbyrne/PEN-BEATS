# LR Scheduler Selection for M4 Experiments

## Core Rule

| Protocol | Periods | Default LR scheduler |
|---|---|---|
| `nbeats_paper` (paper-sample) | Quarterly, Monthly, Daily | **`ReduceLROnPlateau`** |
| `nbeats_paper` | Yearly | `step_paper` (10-milestone MultiStepLR, γ=0.5) — neutral |
| `nbeats_paper` | Weekly | `step_paper` for now — flip to plateau (`p3_recommended`) once n=10 lands |
| `nbeats_paper` | Hourly | `step_paper` for now — head-to-head against plateau not yet run on novel families |
| `sliding` | all periods | **`CosineAnnealing` + 15-epoch warmup** |

**General principle:** plateau's adaptive LR reduction matches the post-warmup convergence regime of paper-sample sweeps better than step_paper's deterministic 10-milestone schedule, especially on longer-H periods. step_paper still wins on Y (H=6, too short for plateau to differentiate).

---

## Detailed Recommendations

### Plateau LR (`ReduceLROnPlateau`)

**When to use:** any new `sampling_style: nbeats_paper` experiment on M4 Quarterly, Monthly, or Daily. Use this configuration:

```yaml
training:
  sampling_style: nbeats_paper
  optimizer: adam
  learning_rate: 0.001
  val_check_interval: 100      # required — plateau needs sub-epoch validation
  min_delta: 0.001
  patience: 20                  # for early stopping (in val-check units)
  max_epochs: 200
  lr_scheduler:
    name: ReduceLROnPlateau
    patience: 3                 # `p3_recommended` cell — winning Weekly validation cell
    factor: 0.5
    cooldown: 1
```

**Why patience=3, factor=0.5, cooldown=1:**
- `p1_baseline` (patience=1) reduces too aggressively — locks in to a local min before convergence.
- `p5_loose` (patience=5) under-reduces — Daily/Weekly seed std stays elevated.
- `p3_recommended` is the only cell tested on Weekly that beat the prior step_paper SOTA (single-seed 6.559 vs 6.735, Δ−0.176).

**Evidence:** `experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md` §4.1. Plateau strictly beats step_paper by:
- Quarterly: Δ−0.029 mean (n=30 pairs, comprehensive) / −0.025 (n=16, tiered)
- Monthly: Δ−0.089 mean (n=18 pairs, median Δ−0.198)
- Daily: Δ−0.028 mean (n=13 pairs)

### Step-Paper LR (paper MultiStepLR)

**When to use:** Yearly (neutral vs plateau, slight historical preference), Weekly (until plateau p3 validates), and any reproduction targeting Oreshkin et al. 2020 numbers exactly.

10-milestone schedule, γ=0.5 — built into the unified runner under `lr_scheduler: {name: paper_step}` or equivalent. See existing CSVs: `comprehensive_m4_paper_sample_results.csv`, `tiered_offset_m4_allperiods_paperlr_results.csv`.

### Cosine Annealing + Warmup (sliding only)

**When to use:** any `sampling_style: sliding` experiment on M4. This is the LR used in `comprehensive_sweep_m4_results.csv` (the canonical sliding leaderboard).

15-epoch linear warmup → cosine decay to `eta_min=1e-6`. Has not been ported to paper-sample — open experiment in §6.5 of the leaderboard report.

---

## Decision Tree

```
sampling_style == 'sliding'?
├── YES → use cosine_warmup
└── NO (paper-sample)
    ├── period in {Quarterly, Monthly, Daily} → ReduceLROnPlateau (p3_recommended)
    ├── period == Yearly → step_paper (neutral; either works)
    ├── period == Weekly → step_paper (default) OR plateau p3_recommended (if validating tiered)
    └── period == Hourly → step_paper (head-to-head not yet run)
```

---

## Pitfalls

1. **Never use plateau LR without `val_check_interval=100`.** Plateau requires sub-epoch validation to avoid `best_epoch=0/1` collapse on paper-sample. CLAUDE.md "Early-stopping settings" section makes this mandatory.
2. **Do not compare plateau vs step_paper across protocols.** Paper-sample plateau and sliding cosine-warmup live on different SMAPE scales — Δ is uninterpretable. Match protocol within an LR-scheduler comparison.
3. **Tiered Weekly regression at default plateau cell is not the cascade reversal.** It's a plateau-config artifact at H=13. Use `p3_recommended` cell, not `p1_baseline`.

---

## Source

- Canonical: `experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md` §4.1 + §2.4
- Plateau-cell single-seed validation: `experiments/results/m4/tiered_offset_m4_weekly_plateau_validation_results.csv`
- Memory: `project_m4_plateau_lr_findings.md`
