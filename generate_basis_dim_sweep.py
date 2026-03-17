#!/usr/bin/env python3
"""Generate basis_dim sweep entries for omnibus_benchmark_m4.yaml"""

import yaml

# Remaining configs to add (after Coif2WaveletV3 and TrendWaveletGeneric variants)
configs_to_sweep = [
    # TrendAE+DB3WaveletV3AE variants
    {
        "base_name": "TrendAE+DB3WaveletV3AE-30",
        "stacks": {"type": "alternating", "blocks": ["TrendAE", "DB3WaveletV3AE"], "repeats": 15},
        "training": {"active_g": False},
        "arch_family": "novel_ae",
        "block_type_primary": "TrendAE+DB3WaveletV3AE",
        "backbone": "AERootBlock",
        "stack_pattern": "alternating",
        "n_stacks": 30,
    },
    {
        "base_name": "TrendAE+DB3WaveletV3AE-30-activeG",
        "stacks": {"type": "alternating", "blocks": ["TrendAE", "DB3WaveletV3AE"], "repeats": 15},
        "training": {"active_g": "forecast"},
        "arch_family": "novel_ae",
        "block_type_primary": "TrendAE+DB3WaveletV3AE",
        "backbone": "AERootBlock",
        "stack_pattern": "alternating",
        "n_stacks": 30,
    },
    {
        "base_name": "TrendAE+Coif2WaveletV3AE-30-activeG",
        "stacks": {"type": "alternating", "blocks": ["TrendAE", "Coif2WaveletV3AE"], "repeats": 15},
        "training": {"active_g": "forecast"},
        "arch_family": "novel_ae",
        "block_type_primary": "TrendAE+Coif2WaveletV3AE",
        "backbone": "AERootBlockLG",
        "stack_pattern": "alternating",
        "n_stacks": 30,
    },
    # TrendWaveletAELG variants
    {
        "base_name": "TrendWaveletAELG-10",
        "stacks": {"type": "homogeneous", "block": "TrendWaveletAELG", "n": 10},
        "training": {"active_g": False},
        "arch_family": "novel_aelg",
        "block_type_primary": "TrendWaveletAELG",
        "backbone": "AERootBlockLG",
        "stack_pattern": "homogeneous",
        "n_stacks": 10,
    },
    {
        "base_name": "TrendWaveletAELG-20",
        "stacks": {"type": "homogeneous", "block": "TrendWaveletAELG", "n": 20},
        "training": {"active_g": False},
        "arch_family": "novel_aelg",
        "block_type_primary": "TrendWaveletAELG",
        "backbone": "AERootBlockLG",
        "stack_pattern": "homogeneous",
        "n_stacks": 20,
    },
    {
        "base_name": "TrendWaveletAELG-10-activeG",
        "stacks": {"type": "homogeneous", "block": "TrendWaveletAELG", "n": 10},
        "training": {"active_g": "forecast"},
        "arch_family": "novel_aelg",
        "block_type_primary": "TrendWaveletAELG",
        "backbone": "AERootBlockLG",
        "stack_pattern": "homogeneous",
        "n_stacks": 10,
    },
    {
        "base_name": "TrendWaveletAELG-20-activeG",
        "stacks": {"type": "homogeneous", "block": "TrendWaveletAELG", "n": 20},
        "training": {"active_g": "forecast"},
        "arch_family": "novel_aelg",
        "block_type_primary": "TrendWaveletAELG",
        "backbone": "AERootBlockLG",
        "stack_pattern": "homogeneous",
        "n_stacks": 20,
    },
]

basis_dims = [32, 64, 128]

def generate_config(base_config, basis_dim):
    """Generate a single config variant with given basis_dim."""
    config = {}
    config["name"] = f"{base_config['base_name']}_bd{basis_dim}"
    config["category"] = "basis_dim_sweep"
    config["stacks"] = base_config["stacks"]
    config["training"] = base_config["training"]
    config["block_params"] = {"basis_dim": basis_dim}
    
    extra = {
        "arch_family": base_config["arch_family"],
        "block_type_primary": base_config["block_type_primary"],
        "backbone": base_config["backbone"],
        "stack_pattern": base_config["stack_pattern"],
        "n_stacks": base_config["n_stacks"],
        "innovation_claim": f"basis_dim={basis_dim} sweep for {base_config['block_type_primary'].lower()}",
    }
    config["extra_fields"] = extra
    return config

# Generate YAML
output = []
for base_cfg in configs_to_sweep:
    for bd in basis_dims:
        cfg = generate_config(base_cfg, bd)
        output.append(cfg)

# Print as YAML
for cfg in output:
    print(f"  - name: {cfg['name']}")
    print(f"    category: {cfg['category']}")
    print(f"    stacks: {yaml.dump(cfg['stacks'], default_flow_style=False).rstrip()}")
    print(f"    training: {yaml.dump(cfg['training'], default_flow_style=False).rstrip()}")
    print(f"    block_params: {yaml.dump(cfg['block_params'], default_flow_style=False).rstrip()}")
    print(f"    extra_fields:")
    for k, v in cfg['extra_fields'].items():
        print(f"      {k}: {v}")
    print()

