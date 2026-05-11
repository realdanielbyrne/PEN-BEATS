# Dual-GPU Launch Commands for ae_dataset_comparison Experiment

## Overview
The `ae_dataset_comparison` experiment contains 120 total runs:
- **WikiText-2 config**: 60 runs (3 latent dims × 2 mlp modes × 2 inner inits × 5 seeds)
- **FineWeb config**: 60 runs (3 latent dims × 2 mlp modes × 2 inner inits × 5 seeds)

These commands run each config on a separate GPU in parallel.

---

## Option 1: Individual Commands (Run in Separate Terminals)

### Terminal 1 - GPU 0 (WikiText-2 config):
```bash
cd <project_root>
PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/run_from_yaml.py \
    scripts/experiments/ae_dataset_comparison.yaml \
    --config wikitext2 \
    --gpu-id 0
```

### Terminal 2 - GPU 1 (FineWeb config):
```bash
cd <project_root>
PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/run_from_yaml.py \
    scripts/experiments/ae_dataset_comparison.yaml \
    --config fineweb \
    --gpu-id 1
```

---

## Option 2: Automated Script (Run Both in Parallel)

```bash
cd <project_root>
chmod +x run_experiment_dual_gpu.sh
./run_experiment_dual_gpu.sh
```

This script:
- Launches both configs in background processes
- Logs output to separate files
- Waits for both to complete
- Reports final status

Monitor progress:
```bash
tail -f scripts/experiments/results/ae_dataset_comparison/logs/wikitext2_gpu0.log
tail -f scripts/experiments/results/ae_dataset_comparison/logs/fineweb_gpu1.log
```

---

## Expected Runtime
- **Per run**: ~5-10 minutes (depends on GPU, batch size, epochs)
- **Total per config**: ~5-10 hours
- **Both in parallel**: ~5-10 hours (vs ~10-20 hours sequential)

## Output Locations
- Results: `scripts/experiments/results/ae_dataset_comparison/`
- Models: `trainedmodels/ae_dataset_comparison/`
- Logs: `scripts/experiments/results/ae_dataset_comparison/logs/`

## Cleanup After Interrupted Runs

Large sweeps can leave activation caches behind if they are stopped mid-run. Preview reclaimable generated artifacts with:

```bash
.venv/bin/python scripts/prune_experiment_artifacts.py
```

Remove only cache artifacts for this experiment:

```bash
.venv/bin/python scripts/prune_experiment_artifacts.py \
  --experiment ae_dataset_comparison \
  --apply
```

Add `--include-checkpoints` only when you also want to delete `trainedmodels/ae_dataset_comparison/`.
