---
license: apache-2.0
tags:
- text-generation
- causal-lm
- llama
- pellm
- parameter-efficient
library_name: transformers
base_model: none
---

# {{ repo_id }}

This model is one of the PELLM paper checkpoints for evaluating
parameter-efficient architectural replacements in small decoder-only language
models.

## Loading

PE checkpoints require importing `pellm` before loading so the custom
`pe_llama` config and model classes are registered with Transformers.

```python
import pellm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{{ repo_id }}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
```

## Expected Artifacts

- `config.json`
- tokenizer files
- model weights in `safetensors`
- `training_manifest.json`
- summarized eval tables

Large raw eval dumps are intentionally kept outside git and outside the model
repo unless they are promoted to summarized paper artifacts.
