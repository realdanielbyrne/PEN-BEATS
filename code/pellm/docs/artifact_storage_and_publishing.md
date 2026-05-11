# Artifact Storage And Publishing

Heavy training state should live under `<pellm_data_root>`, not inside the git
checkout. Set `PELLM_DATA_ROOT` to override this root on another machine.

## Layout

- `<pellm_data_root>/datasets`: HF caches and tokenized shards
- `<pellm_data_root>/checkpoints`: intermediate training checkpoints
- `<pellm_data_root>/trainedmodels`: final HuggingFace-format model directories
- `<pellm_data_root>/runs`: trainer logs, local W&B/TensorBoard, profiler traces
- `<pellm_data_root>/evals`: raw evaluation outputs
- `<pellm_data_root>/hf_staging`: model-card and upload metadata staging
- `<pellm_data_root>/tmp`: temporary files and distributed-training scratch

Create or verify the directories with:

```bash
python scripts/setup_artifact_dirs.py
```

The orchestrators apply these defaults to subprocesses unless the environment
already defines them:

```bash
export HF_HOME=<pellm_data_root>/hf_home
export HF_DATASETS_CACHE=<pellm_data_root>/datasets/hf_cache
export TRANSFORMERS_CACHE=<pellm_data_root>/datasets/transformers_cache
export TMPDIR=<pellm_data_root>/tmp
```

## Hugging Face Publishing

Default model repos:

- `anon/pellm-smol-135m-baseline`
- `anon/pellm-smol-135m-ae-mlp`
- `anon/pellm-smol-135m-trendwavelet`
- `anon/pellm-smol-135m-ae-tw`

Publish privately by default:

```bash
python scripts/publish_model_to_hf.py \
  --variant ae_tw \
  --model-dir <pellm_data_root>/trainedmodels/pellm-smol-135m-ae-tw \
  --manifest <pellm_data_root>/trainedmodels/pellm-smol-135m-ae-tw/training_manifest.json \
  --eval-summary <pellm_data_root>/evals/pellm-smol-135m-ae-tw/summary.json
```

Add `--public` only when the paper artifacts are ready.

Use `scripts/templates/training_manifest.template.json` as the starting shape
for the required `training_manifest.json` that travels with every published
model.
