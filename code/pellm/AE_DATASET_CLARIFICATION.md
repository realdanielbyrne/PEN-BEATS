# AE Dataset Clarification

## The Issue

When running experiments with `--ae-dataset fineweb`, you may see both of these log messages:

```
INFO: Loading WikiText-2...
INFO: Loading FineWeb for AE activation caching...
```

This is **correct behavior** and not a bug. Here's why:

## Two Separate Datasets

The `--ae-dataset` flag controls **only** the dataset used for AE pre-training activation caching. It does **not** affect the training/validation/testing benchmark.

### WikiText-2 (Always Loaded)
- **Purpose**: Training, validation, and testing benchmark
- **When**: Always loaded when `--ae-pretrain-epochs > 0` and AE MLP layers are being trained
- **Message**: `INFO: Loading WikiText-2...`
- **Controlled by**: Hard-coded in the training loop (line 970 of `finetune.py`)

### FineWeb (Optional, for AE Pre-training Only)
- **Purpose**: Diverse activation distribution for AE pre-training Phase 1
- **When**: Only loaded when `--ae-dataset fineweb` is specified
- **Message**: `INFO: Loading FineWeb for AE activation caching...`
- **Controlled by**: `--ae-dataset` flag

## Why Two Datasets?

1. **WikiText-2** is the evaluation benchmark — it's what we measure perplexity on
2. **FineWeb** provides more diverse activations for better AE initialization during pre-training

The two-phase AE pre-training strategy:
- **Phase 1**: Extract teacher activations from FineWeb (diverse) to disk
- **Phase 2**: Train AE layers from cached activations (no teacher model in memory)
- **LM Fine-tuning**: Train on WikiText-2 (the benchmark)

## Example Output

When running with `--ae-dataset fineweb --ae-pretrain-epochs 5`:

```
INFO: Loading WikiText-2...                          ← Training benchmark
INFO: Train: 4755 chunks, Val: 493 chunks, Test: 564 chunks
...
INFO: Loading FineWeb for AE activation caching...   ← AE pre-training dataset
INFO: Streaming 10000 samples from HuggingFaceFW/fineweb...
INFO: FineWeb: 19531 chunks of 512 tokens.
INFO: AE Phase 1: Extracting teacher activations...
```

Both messages are expected and indicate the system is working correctly.

## Debugging

If you want to see which dataset is being used for AE caching, look for:
- `Loading FineWeb...` → Using FineWeb (diverse)
- `Using training dataset (WikiText-2)...` → Using WikiText-2 (default)

The debug log also includes: `AE dataset choice: args.ae_dataset='fineweb'`

