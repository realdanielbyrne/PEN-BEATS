# Smol Replacement Benchmark Report

- **Generated:** `2026-05-02T15:04:02.832560+00:00`
- **Experiment:** `smol_replacement_paper`
- **Config:** `<project_root>/scripts/experiments/smol_replacement_paper.yaml`
- **Tasks:** `lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa`
- **Few-shot:** `0`
- **Limit:** `none`

## Metric Legend

| Symbol | Meaning |
|--------|---------|
| **↑** | Higher value is better |
| **↓** | Lower value is better |
| **(%)** | Percentage (0–100 scale, derived from 0–1 accuracy scores) |

### Task Metrics

- **LAMBADA** — `acc` (%): Next-token prediction accuracy on the LAMBADA dataset (↑). A perplexity score is also reported for LAMBADA (↓).
- **HellaSwag** — `acc_norm` (%): Normalized accuracy on commonsense sentence completion (↑).
- **PIQA** — `acc_norm` (%): Normalized accuracy on physical commonsense reasoning (↑).
- **ARC-Easy / ARC-Challenge** — `acc_norm` (%): Normalized accuracy on grade-school science questions (↑).
- **WinoGrande** — `acc` (%): Accuracy on pronoun resolution / commonsense reasoning (↑).
- **OpenBookQA** — `acc_norm` (%): Normalized accuracy on open-book science QA (↑).

*All accuracy-like metrics (`acc`, `acc_norm`, `exact_match`, `f1`) are reported as percentages (0–100). Higher is better. Perplexity is reported as a raw value; lower is better.*

| Variant | Status | LAMBADA acc (%) ↑ | HellaSwag acc_norm (%) ↑ | PIQA acc_norm (%) ↑ | ARC-Easy acc_norm (%) ↑ | ARC-Challenge acc_norm (%) ↑ | WinoGrande acc (%) ↑ | OpenBookQA acc_norm (%) ↑ | Completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | complete | 24.22 ↑ | 31.32 ↑ | 61.97 ↑ | 45.50 ↑ | 25.09 ↑ | 51.30 ↑ | 30.20 ↑ | 2026-05-02T14:58:52.877286+00:00 |
| ae_mlp | complete | 18.07 ↑ | 27.63 ↑ | 59.14 ↑ | 41.29 ↑ | 22.61 ↑ | 51.46 ↑ | 27.80 ↑ | 2026-05-02T15:00:43.094645+00:00 |
| trendwavelet_db3_32 | training incomplete | - | - | - | - | - | - | - | - |
| ae_tw_db3_32 | training incomplete | - | - | - | - | - | - | - | - |
| trendwavelet_db3_32_tiered | training incomplete | - | - | - | - | - | - | - | - |
| trendwavelet_sym10_64_tiered | training incomplete | - | - | - | - | - | - | - | - |
| ae_tw_sym10_64_tiered | training incomplete | - | - | - | - | - | - | - | - |
