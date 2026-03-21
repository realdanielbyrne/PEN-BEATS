#!/usr/bin/env python3
"""Generate paper-ready comprehensive sweep YAML configs.

Produces one YAML file per dataset with ~96 configs designed to prove
every major finding in a single clean experiment:

  F1:  active_g stabilizes Generic blocks, can hurt TrendWavelet accuracy
  F2:  trend_thetas_dim=3 is best
  F3:  Wavelet type barely matters (4 families spanning types and orders)
  F4:  basis_dim=eq_fcast is best
  F5:  Higher latent_dim hurts AE, helps AELG
  F6:  Skip connections rescue deep AELG and Generic stacks
  F7:  active_g can rescue Generic without skip (linear proj instability)
  F8:  Backbone hierarchy: AELG >= RootBlock > AE
  F9:  Novel architectures match/beat baselines at 10-80x fewer params
  F10: Alternating TrendAELG+WaveletV3AELG = top quality
  F11: TrendAE+WaveletV3AE-activeG = most stable (lowest std)
  F12: TrendWaveletAELG/TrendWaveletGenericAELG = Pareto optimal

Uses random seeds (saved in CSV for reproducibility).

Usage:
    python experiments/generate_comprehensive_sweep.py
"""

import os
import textwrap

# ── Sweep dimensions ────────────────────────────────────────────────────────

STACKS = [10, 30]
WAVELET_FAMILIES = ["haar", "coif2", "db3", "sym10"]
# Named subclass block suffixes for alternating patterns
WAVELET_AELG_BLOCKS = {
    "haar": "HaarWaveletV3AELG",
    "coif2": "Coif2WaveletV3AELG",
    "db3": "DB3WaveletV3AELG",
    "sym10": "Symlet10WaveletV3AELG",
}
WAVELET_AE_BLOCKS = {
    "haar": "HaarWaveletV3AE",
    "coif2": "Coif2WaveletV3AE",
    "db3": "DB3WaveletV3AE",
    "sym10": "Symlet10WaveletV3AE",
}
WAVELET_RB_BLOCKS = {
    "haar": "HaarWaveletV3",
    "coif2": "Coif2WaveletV3",
    "db3": "DB3WaveletV3",
    "sym10": "Symlet10WaveletV3",
}
BASIS_DIMS = ["eq_fcast", "2*eq_fcast"]
LATENT_DIMS = [8, 16, 32]
ACTIVE_G_VALUES = [False, "forecast"]
SKIP_ALPHA = 0.1

# ── Dataset definitions ────────────────────────────────────────────────────

DATASETS = {
    "m4": {
        "dataset": "m4",
        "periods": ["Yearly", "Quarterly"],
        "loss": "SMAPELoss",
        "protocol": None,
    },
    "tourism": {
        "dataset": "tourism",
        "periods": ["Tourism-Yearly"],
        "loss": "SMAPELoss",
        "protocol": None,
    },
    "weather": {
        "dataset": "weather",
        "periods": ["Weather-96"],
        "loss": "SMAPELoss",
        "protocol": {
            "normalize": True,
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "include_target": True,
        },
    },
    "traffic": {
        "dataset": "traffic",
        "periods": ["Traffic-96"],
        "loss": "MSELoss",
        "protocol": {
            "normalize": False,
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "include_target": True,
        },
    },
    "milk": {
        "dataset": "milk",
        "periods": ["Milk"],
        "loss": "SMAPELoss",
        "protocol": None,
    },
}

# ── Extra CSV columns ──────────────────────────────────────────────────────

EXTRA_CSV_COLUMNS = [
    "arch_family",
    "block_type_primary",
    "backbone",
    "stack_pattern",
    "n_stacks",
    "wavelet_family",
    "basis_dim_label",
    "latent_dim_cfg",
    "trend_thetas_dim_cfg",
    "skip_distance_cfg",
    "active_g_cfg",
    "kl_weight_cfg",
]

# ── Helpers ─────────────────────────────────────────────────────────────────


def _ag_tag(ag):
    return "ag0" if ag is False else "agf"


def _ag_yaml(ag):
    return "false" if ag is False else "forecast"


def _bd_tag(bd):
    if bd == "eq_fcast":
        return "bdeq"
    elif bd == "2*eq_fcast":
        return "bd2eq"
    return f"bd{bd}"


def _extra(
    *,
    arch_family,
    block_type_primary,
    backbone,
    stack_pattern,
    n_stacks,
    wavelet_family=None,
    basis_dim_label=None,
    latent_dim_cfg=None,
    trend_thetas_dim_cfg=None,
    skip_distance_cfg=0,
    active_g_cfg="false",
):
    """Build extra_fields dict."""
    return {
        "arch_family": arch_family,
        "block_type_primary": block_type_primary,
        "backbone": backbone,
        "stack_pattern": stack_pattern,
        "n_stacks": n_stacks,
        "wavelet_family": wavelet_family,
        "basis_dim_label": basis_dim_label,
        "latent_dim_cfg": latent_dim_cfg,
        "trend_thetas_dim_cfg": trend_thetas_dim_cfg,
        "skip_distance_cfg": skip_distance_cfg,
        "active_g_cfg": active_g_cfg,
    }


# ── Config generators ──────────────────────────────────────────────────────


def gen_A_paper_baselines():
    """A. Paper baselines: NBEATS-G, NBEATS-I+G. 8 configs.

    Proves: F1 (active_g on Generic), F7 (active_g rescues NBEATS-G),
            F9 (parameter count reference).
    """
    configs = []

    # NBEATS-G: homogeneous Generic
    for n in STACKS:
        for ag in ACTIVE_G_VALUES:
            name = f"NBEATS-G_{n}s_{_ag_tag(ag)}"
            configs.append(
                {
                    "name": name,
                    "category": "paper_baseline",
                    "stacks": {"type": "homogeneous", "block": "Generic", "n": n},
                    "training": {"active_g": ag},
                    "extra_fields": _extra(
                        arch_family="paper_baseline",
                        block_type_primary="Generic",
                        backbone="RootBlock",
                        stack_pattern="homogeneous",
                        n_stacks=n,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )

    # NBEATS-I+G: prefix_body [Trend, Seasonality] + Generic
    for n in STACKS:
        for ag in ACTIVE_G_VALUES:
            name = f"NBEATS-IG_{n}s_{_ag_tag(ag)}"
            configs.append(
                {
                    "name": name,
                    "category": "paper_baseline",
                    "stacks": {
                        "type": "prefix_body",
                        "prefix": ["Trend", "Seasonality"],
                        "body": "Generic",
                        "total": n,
                    },
                    "training": {"active_g": ag},
                    "block_params": {"trend_thetas_dim": 3},
                    "extra_fields": _extra(
                        arch_family="paper_baseline",
                        block_type_primary="Trend+Seasonality+Generic",
                        backbone="RootBlock",
                        stack_pattern="prefix_body",
                        n_stacks=n,
                        trend_thetas_dim_cfg=3,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )

    return configs


def gen_B_trendwavelet_rootblock():
    """B. TrendWavelet RootBlock — wavelet + basis_dim sweep. 16 configs.

    Proves: F3 (wavelet type doesn't matter), F4 (basis_dim eq_fcast best),
            F8 (backbone comparison anchor).
    """
    configs = []
    for n in STACKS:
        for wt in WAVELET_FAMILIES:
            for bd in BASIS_DIMS:
                name = f"TW_{n}s_td3_{_bd_tag(bd)}_{wt}"
                configs.append(
                    {
                        "name": name,
                        "category": "trendwavelet_rb",
                        "stacks": {
                            "type": "homogeneous",
                            "block": "TrendWavelet",
                            "n": n,
                        },
                        "training": {"active_g": False},
                        "block_params": {
                            "trend_thetas_dim": 3,
                            "basis_dim": bd,
                            "wavelet_type": wt,
                        },
                        "extra_fields": _extra(
                            arch_family="trendwavelet",
                            block_type_primary="TrendWavelet",
                            backbone="RootBlock",
                            stack_pattern="homogeneous",
                            n_stacks=n,
                            wavelet_family=wt,
                            basis_dim_label=bd,
                            trend_thetas_dim_cfg=3,
                        ),
                    }
                )
    return configs


def gen_C_trendwavelet_td_sweep():
    """C. TrendWavelet trend_thetas_dim sweep. 2 configs.

    Proves: F2 (td=3 is best).
    Uses db3 at 10 stacks x td5 x {eq_fcast,2*eq_fcast}.
    td=3 comparison comes from group B (same wavelet/stacks/basis_dim).
    """
    configs = []
    for bd in BASIS_DIMS:
        name = f"TW_10s_td5_{_bd_tag(bd)}_db3"
        configs.append(
            {
                "name": name,
                "category": "trendwavelet_td",
                "stacks": {"type": "homogeneous", "block": "TrendWavelet", "n": 10},
                "training": {"active_g": False},
                "block_params": {
                    "trend_thetas_dim": 5,
                    "basis_dim": bd,
                    "wavelet_type": "db3",
                },
                "extra_fields": _extra(
                    arch_family="trendwavelet",
                    block_type_primary="TrendWavelet",
                    backbone="RootBlock",
                    stack_pattern="homogeneous",
                    n_stacks=10,
                    wavelet_family="db3",
                    basis_dim_label=bd,
                    trend_thetas_dim_cfg=5,
                ),
            }
        )
    return configs


def gen_D_trendwaveletae_ld():
    """D. TrendWaveletAE latent_dim sweep. 6 configs.

    Proves: F5 (higher ld hurts AE), F8 (backbone comparison).
    """
    configs = []
    for ld in LATENT_DIMS:
        for ag in ACTIVE_G_VALUES:
            name = f"TWAE_10s_ld{ld}_{_ag_tag(ag)}"
            configs.append(
                {
                    "name": name,
                    "category": "trendwaveletae",
                    "stacks": {
                        "type": "homogeneous",
                        "block": "TrendWaveletAE",
                        "n": 10,
                    },
                    "training": {"active_g": ag},
                    "block_params": {
                        "trend_thetas_dim": 3,
                        "basis_dim": "eq_fcast",
                        "wavelet_type": "db3",
                        "latent_dim": ld,
                    },
                    "extra_fields": _extra(
                        arch_family="trendwavelet_ae",
                        block_type_primary="TrendWaveletAE",
                        backbone="AERootBlock",
                        stack_pattern="homogeneous",
                        n_stacks=10,
                        wavelet_family="db3",
                        basis_dim_label="eq_fcast",
                        latent_dim_cfg=ld,
                        trend_thetas_dim_cfg=3,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )
    return configs


def gen_E_trendwaveletaelg():
    """E. TrendWaveletAELG — wavelet, latent_dim, skip sweeps. 22 configs.

    Proves: F1 (active_g), F3 (wavelet), F5 (ld helps AELG),
            F6 (skip rescue), F8 (backbone), F12 (Pareto).

    E1: Wavelet family sweep — {10,30} x 4 wavelets x {ag0,agf} x ld16 = 16
    E2: Latent dim sweep — 10s x {ld8,ld32} x {ag0,agf} = 4  (ld16 in E1)
    E3: Skip at depth — 30s x {ag0,agf} x skip5 x ld16 = 2  (no-skip in E1)
    """
    configs = []

    # E1: Wavelet family sweep at ld=16
    for n in STACKS:
        for wt in WAVELET_FAMILIES:
            for ag in ACTIVE_G_VALUES:
                name = f"TWAELG_{n}s_ld16_{wt}_{_ag_tag(ag)}"
                configs.append(
                    {
                        "name": name,
                        "category": "trendwaveletaelg",
                        "stacks": {
                            "type": "homogeneous",
                            "block": "TrendWaveletAELG",
                            "n": n,
                        },
                        "training": {"active_g": ag},
                        "block_params": {
                            "trend_thetas_dim": 3,
                            "basis_dim": "eq_fcast",
                            "wavelet_type": wt,
                            "latent_dim": 16,
                        },
                        "extra_fields": _extra(
                            arch_family="trendwavelet_aelg",
                            block_type_primary="TrendWaveletAELG",
                            backbone="AERootBlockLG",
                            stack_pattern="homogeneous",
                            n_stacks=n,
                            wavelet_family=wt,
                            basis_dim_label="eq_fcast",
                            latent_dim_cfg=16,
                            trend_thetas_dim_cfg=3,
                            active_g_cfg=_ag_yaml(ag),
                        ),
                    }
                )

    # E2: Latent dim extremes at 10 stacks (ld=16 already in E1)
    for ld in [8, 32]:
        for ag in ACTIVE_G_VALUES:
            name = f"TWAELG_10s_ld{ld}_db3_{_ag_tag(ag)}"
            configs.append(
                {
                    "name": name,
                    "category": "trendwaveletaelg",
                    "stacks": {
                        "type": "homogeneous",
                        "block": "TrendWaveletAELG",
                        "n": 10,
                    },
                    "training": {"active_g": ag},
                    "block_params": {
                        "trend_thetas_dim": 3,
                        "basis_dim": "eq_fcast",
                        "wavelet_type": "db3",
                        "latent_dim": ld,
                    },
                    "extra_fields": _extra(
                        arch_family="trendwavelet_aelg",
                        block_type_primary="TrendWaveletAELG",
                        backbone="AERootBlockLG",
                        stack_pattern="homogeneous",
                        n_stacks=10,
                        wavelet_family="db3",
                        basis_dim_label="eq_fcast",
                        latent_dim_cfg=ld,
                        trend_thetas_dim_cfg=3,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )

    # E3: Skip rescue at 30 stacks
    for ag in ACTIVE_G_VALUES:
        name = f"TWAELG_30s_ld16_db3_{_ag_tag(ag)}_sd5"
        configs.append(
            {
                "name": name,
                "category": "trendwaveletaelg",
                "stacks": {"type": "homogeneous", "block": "TrendWaveletAELG", "n": 30},
                "training": {
                    "active_g": ag,
                    "skip_distance": 5,
                    "skip_alpha": SKIP_ALPHA,
                },
                "block_params": {
                    "trend_thetas_dim": 3,
                    "basis_dim": "eq_fcast",
                    "wavelet_type": "db3",
                    "latent_dim": 16,
                },
                "extra_fields": _extra(
                    arch_family="trendwavelet_aelg",
                    block_type_primary="TrendWaveletAELG",
                    backbone="AERootBlockLG",
                    stack_pattern="homogeneous",
                    n_stacks=30,
                    wavelet_family="db3",
                    basis_dim_label="eq_fcast",
                    latent_dim_cfg=16,
                    trend_thetas_dim_cfg=3,
                    skip_distance_cfg=5,
                    active_g_cfg=_ag_yaml(ag),
                ),
            }
        )

    return configs


def gen_F_alt_aelg_waveletv3():
    """F. Alternating TrendAELG + WaveletV3AELG. 12 configs.

    Top-quality architecture from omnibus rankings.
    Proves: F1 (active_g), F3 (wavelet), F10 (best quality), F11 (stability).

    F1: Wavelet sweep at 30 stacks — 4 wavelets x {ag0,agf} = 8
    F2: Depth comparison — 10s x DB3 x {ag0,agf} = 2
    F3: Skip at depth — 30s x {Coif2,DB3} x ag0 x skip5 = 2
    """
    configs = []

    # F1: 30-stack wavelet sweep (15 repeats of [TrendAELG, XWaveletV3AELG])
    for wt in WAVELET_FAMILIES:
        wav_block = WAVELET_AELG_BLOCKS[wt]
        for ag in ACTIVE_G_VALUES:
            short = wt.capitalize()
            name = f"TALG+{short}V3ALG_30s_{_ag_tag(ag)}"
            configs.append(
                {
                    "name": name,
                    "category": "alt_aelg",
                    "stacks": {
                        "type": "alternating",
                        "blocks": ["TrendAELG", wav_block],
                        "repeats": 15,
                    },
                    "training": {"active_g": ag},
                    "block_params": {"latent_dim": 16},
                    "extra_fields": _extra(
                        arch_family="alt_aelg",
                        block_type_primary=f"TrendAELG+{wav_block}",
                        backbone="AERootBlockLG",
                        stack_pattern="alternating",
                        n_stacks=30,
                        wavelet_family=wt,
                        latent_dim_cfg=16,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )

    # F2: 10-stack depth comparison with DB3
    wav_block = WAVELET_AELG_BLOCKS["db3"]
    for ag in ACTIVE_G_VALUES:
        name = f"TALG+DB3V3ALG_10s_{_ag_tag(ag)}"
        configs.append(
            {
                "name": name,
                "category": "alt_aelg",
                "stacks": {
                    "type": "alternating",
                    "blocks": ["TrendAELG", wav_block],
                    "repeats": 5,
                },
                "training": {"active_g": ag},
                "block_params": {"latent_dim": 16},
                "extra_fields": _extra(
                    arch_family="alt_aelg",
                    block_type_primary=f"TrendAELG+{wav_block}",
                    backbone="AERootBlockLG",
                    stack_pattern="alternating",
                    n_stacks=10,
                    wavelet_family="db3",
                    latent_dim_cfg=16,
                    active_g_cfg=_ag_yaml(ag),
                ),
            }
        )

    # F3: Skip at 30 stacks for Coif2 and DB3
    for wt in ["coif2", "db3"]:
        wav_block = WAVELET_AELG_BLOCKS[wt]
        short = wt.capitalize()
        name = f"TALG+{short}V3ALG_30s_ag0_sd5"
        configs.append(
            {
                "name": name,
                "category": "alt_aelg",
                "stacks": {
                    "type": "alternating",
                    "blocks": ["TrendAELG", wav_block],
                    "repeats": 15,
                },
                "training": {
                    "active_g": False,
                    "skip_distance": 5,
                    "skip_alpha": SKIP_ALPHA,
                },
                "block_params": {"latent_dim": 16},
                "extra_fields": _extra(
                    arch_family="alt_aelg",
                    block_type_primary=f"TrendAELG+{wav_block}",
                    backbone="AERootBlockLG",
                    stack_pattern="alternating",
                    n_stacks=30,
                    wavelet_family=wt,
                    latent_dim_cfg=16,
                    skip_distance_cfg=5,
                ),
            }
        )

    return configs


def gen_G_alt_ae_waveletv3():
    """G. Alternating TrendAE + WaveletV3AE. 6 configs.

    Stability champion from omnibus (TrendAE+DB3WaveletV3AE-30-activeG).
    Proves: F5 (ld effect on AE), F8 (backbone), F11 (most stable).

    30 stacks x DB3 x {ld8,ld16,ld32} x {ag0,agf} = 6
    """
    configs = []
    wav_block = WAVELET_AE_BLOCKS["db3"]
    for ld in LATENT_DIMS:
        for ag in ACTIVE_G_VALUES:
            name = f"TAE+DB3V3AE_30s_ld{ld}_{_ag_tag(ag)}"
            configs.append(
                {
                    "name": name,
                    "category": "alt_ae",
                    "stacks": {
                        "type": "alternating",
                        "blocks": ["TrendAE", wav_block],
                        "repeats": 15,
                    },
                    "training": {"active_g": ag},
                    "block_params": {"latent_dim": ld},
                    "extra_fields": _extra(
                        arch_family="alt_ae",
                        block_type_primary=f"TrendAE+{wav_block}",
                        backbone="AERootBlock",
                        stack_pattern="alternating",
                        n_stacks=30,
                        wavelet_family="db3",
                        latent_dim_cfg=ld,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )
    return configs


def gen_H_alt_trend_waveletv3_rb():
    """H. Alternating Trend + WaveletV3 RootBlock. 8 configs.

    Cross-period champion (#1 on Quarterly).
    Proves: F3 (wavelet), F4 (basis_dim), F8 (backbone), F9 (params).

    30 stacks x 4 wavelets x {eq_fcast,2*eq_fcast} = 8
    """
    configs = []
    for wt in WAVELET_FAMILIES:
        wav_block = WAVELET_RB_BLOCKS[wt]
        for bd in BASIS_DIMS:
            short = wt.capitalize()
            name = f"T+{short}V3_30s_{_bd_tag(bd)}"
            configs.append(
                {
                    "name": name,
                    "category": "alt_trend_wavelet_rb",
                    "stacks": {
                        "type": "alternating",
                        "blocks": ["Trend", wav_block],
                        "repeats": 15,
                    },
                    "training": {"active_g": False},
                    "block_params": {
                        "trend_thetas_dim": 3,
                        "basis_dim": bd,
                    },
                    "extra_fields": _extra(
                        arch_family="alt_trend_wavelet_rb",
                        block_type_primary=f"Trend+{wav_block}",
                        backbone="RootBlock",
                        stack_pattern="alternating",
                        n_stacks=30,
                        wavelet_family=wt,
                        basis_dim_label=bd,
                        trend_thetas_dim_cfg=3,
                    ),
                }
            )
    return configs


def gen_I_genericaelg_skip_ag():
    """I. GenericAELG skip + active_g interaction. 8 configs.

    Proves: F6 (skip rescue at depth), F7 (active_g rescues without skip).

    I1: 30s x ld16 x {ag0,agf} x {skip0,skip5} = 4
    I2: 10s x ld16 x {ag0,agf} = 2  (baseline depth)
    I3: 30s x ld32 x ag0 x {skip0,skip5} = 2  (ld effect on deep Generic)
    """
    configs = []

    # I1: 2x2 interaction at 30 stacks
    for ag in ACTIVE_G_VALUES:
        for sd in [0, 5]:
            sd_tag = f"_sd{sd}" if sd > 0 else ""
            name = f"GAELG_30s_ld16_{_ag_tag(ag)}{sd_tag}"
            training = {"active_g": ag}
            if sd > 0:
                training["skip_distance"] = sd
                training["skip_alpha"] = SKIP_ALPHA
            configs.append(
                {
                    "name": name,
                    "category": "genericaelg_skip",
                    "stacks": {"type": "homogeneous", "block": "GenericAELG", "n": 30},
                    "training": training,
                    "block_params": {"latent_dim": 16},
                    "extra_fields": _extra(
                        arch_family="generic_aelg",
                        block_type_primary="GenericAELG",
                        backbone="AERootBlockLG",
                        stack_pattern="homogeneous",
                        n_stacks=30,
                        latent_dim_cfg=16,
                        skip_distance_cfg=sd,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )

    # I2: 10-stack baseline
    for ag in ACTIVE_G_VALUES:
        name = f"GAELG_10s_ld16_{_ag_tag(ag)}"
        configs.append(
            {
                "name": name,
                "category": "genericaelg_skip",
                "stacks": {"type": "homogeneous", "block": "GenericAELG", "n": 10},
                "training": {"active_g": ag},
                "block_params": {"latent_dim": 16},
                "extra_fields": _extra(
                    arch_family="generic_aelg",
                    block_type_primary="GenericAELG",
                    backbone="AERootBlockLG",
                    stack_pattern="homogeneous",
                    n_stacks=10,
                    latent_dim_cfg=16,
                    active_g_cfg=_ag_yaml(ag),
                ),
            }
        )

    # I3: ld=32 at 30 stacks (ld sensitivity)
    for sd in [0, 5]:
        sd_tag = f"_sd{sd}" if sd > 0 else ""
        name = f"GAELG_30s_ld32_ag0{sd_tag}"
        training = {"active_g": False}
        if sd > 0:
            training["skip_distance"] = sd
            training["skip_alpha"] = SKIP_ALPHA
        configs.append(
            {
                "name": name,
                "category": "genericaelg_skip",
                "stacks": {"type": "homogeneous", "block": "GenericAELG", "n": 30},
                "training": training,
                "block_params": {"latent_dim": 32},
                "extra_fields": _extra(
                    arch_family="generic_aelg",
                    block_type_primary="GenericAELG",
                    backbone="AERootBlockLG",
                    stack_pattern="homogeneous",
                    n_stacks=30,
                    latent_dim_cfg=32,
                    skip_distance_cfg=sd,
                ),
            }
        )

    return configs


def gen_J_trendwaveletgenericaelg():
    """J. TrendWaveletGenericAELG — Pareto optimal. 4 configs.

    Proves: F12 (Pareto optimal at ~0.45M params).

    {10,30} x db3 x td3 x eq_fcast x ld16 x {ag0,agf} = 4
    """
    configs = []
    for n in STACKS:
        for ag in ACTIVE_G_VALUES:
            name = f"TWGAELG_{n}s_ld16_{_ag_tag(ag)}"
            configs.append(
                {
                    "name": name,
                    "category": "trendwaveletgenericaelg",
                    "stacks": {
                        "type": "homogeneous",
                        "block": "TrendWaveletGenericAELG",
                        "n": n,
                    },
                    "training": {"active_g": ag},
                    "block_params": {
                        "trend_thetas_dim": 3,
                        "basis_dim": "eq_fcast",
                        "wavelet_type": "db3",
                        "latent_dim": 16,
                    },
                    "extra_fields": _extra(
                        arch_family="trendwavelet_genericaelg",
                        block_type_primary="TrendWaveletGenericAELG",
                        backbone="AERootBlockLG",
                        stack_pattern="homogeneous",
                        n_stacks=n,
                        wavelet_family="db3",
                        basis_dim_label="eq_fcast",
                        latent_dim_cfg=16,
                        trend_thetas_dim_cfg=3,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )
    return configs


def gen_K_genericae_ld():
    """K. GenericAE latent_dim contrast. 4 configs.

    Proves: F5 (higher ld hurts AE), F8 (AE < AELG).

    10s x {ld8,ld16,ld32} x ag0 = 3
    10s x ld16 x agf = 1
    """
    configs = []
    for ld in LATENT_DIMS:
        name = f"GAE_10s_ld{ld}_ag0"
        configs.append(
            {
                "name": name,
                "category": "genericae",
                "stacks": {"type": "homogeneous", "block": "GenericAE", "n": 10},
                "training": {"active_g": False},
                "block_params": {"latent_dim": ld},
                "extra_fields": _extra(
                    arch_family="generic_ae",
                    block_type_primary="GenericAE",
                    backbone="AERootBlock",
                    stack_pattern="homogeneous",
                    n_stacks=10,
                    latent_dim_cfg=ld,
                ),
            }
        )

    # active_g check at ld=16
    configs.append(
        {
            "name": "GAE_10s_ld16_agf",
            "category": "genericae",
            "stacks": {"type": "homogeneous", "block": "GenericAE", "n": 10},
            "training": {"active_g": "forecast"},
            "block_params": {"latent_dim": 16},
            "extra_fields": _extra(
                arch_family="generic_ae",
                block_type_primary="GenericAE",
                backbone="AERootBlock",
                stack_pattern="homogeneous",
                n_stacks=10,
                latent_dim_cfg=16,
                active_g_cfg="forecast",
            ),
        }
    )

    return configs


def gen_L_bottleneckgeneric():
    """L. BottleneckGeneric (RootBlock) — active_g x depth sweep. 4 configs.

    Proves: F1 (active_g on Bottleneck), F7 (active_g rescues without skip),
            F9 (parameter contrast vs Generic baseline).

    {10,30} x {ag0,agf} = 4
    """
    configs = []
    for n in STACKS:
        for ag in ACTIVE_G_VALUES:
            name = f"BNG_{n}s_{_ag_tag(ag)}"
            configs.append(
                {
                    "name": name,
                    "category": "bottleneckgeneric",
                    "stacks": {
                        "type": "homogeneous",
                        "block": "BottleneckGeneric",
                        "n": n,
                    },
                    "training": {"active_g": ag},
                    "extra_fields": _extra(
                        arch_family="bottleneckgeneric",
                        block_type_primary="BottleneckGeneric",
                        backbone="RootBlock",
                        stack_pattern="homogeneous",
                        n_stacks=n,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )
    return configs


def gen_M_bottleneckgenericae():
    """M. BottleneckGenericAE latent_dim contrast. 4 configs.

    Proves: F5 (higher ld hurts AE), F8 (AE < AELG backbone).
    Mirrors group K but with the Bottleneck projection.

    10s x {ld8,ld16,ld32} x ag0 = 3
    10s x ld16 x agf = 1
    """
    configs = []
    for ld in LATENT_DIMS:
        name = f"BNAE_10s_ld{ld}_ag0"
        configs.append(
            {
                "name": name,
                "category": "bottleneckgenericae",
                "stacks": {
                    "type": "homogeneous",
                    "block": "BottleneckGenericAE",
                    "n": 10,
                },
                "training": {"active_g": False},
                "block_params": {"latent_dim": ld},
                "extra_fields": _extra(
                    arch_family="bottleneckgeneric_ae",
                    block_type_primary="BottleneckGenericAE",
                    backbone="AERootBlock",
                    stack_pattern="homogeneous",
                    n_stacks=10,
                    latent_dim_cfg=ld,
                ),
            }
        )

    # active_g check at ld=16
    configs.append(
        {
            "name": "BNAE_10s_ld16_agf",
            "category": "bottleneckgenericae",
            "stacks": {
                "type": "homogeneous",
                "block": "BottleneckGenericAE",
                "n": 10,
            },
            "training": {"active_g": "forecast"},
            "block_params": {"latent_dim": 16},
            "extra_fields": _extra(
                arch_family="bottleneckgeneric_ae",
                block_type_primary="BottleneckGenericAE",
                backbone="AERootBlock",
                stack_pattern="homogeneous",
                n_stacks=10,
                latent_dim_cfg=16,
                active_g_cfg="forecast",
            ),
        }
    )

    return configs


def gen_N_bottleneckgenericaelg():
    """N. BottleneckGenericAELG — skip + active_g interaction. 8 configs.

    Proves: F6 (skip rescue at depth), F7 (active_g rescues without skip).
    Mirrors group I but with the Bottleneck+LG backbone.

    N1: 30s x ld16 x {ag0,agf} x {skip0,skip5} = 4
    N2: 10s x ld16 x {ag0,agf} = 2  (baseline depth)
    N3: 30s x ld32 x ag0 x {skip0,skip5} = 2  (ld sensitivity at depth)
    """
    configs = []

    # N1: 2x2 interaction at 30 stacks
    for ag in ACTIVE_G_VALUES:
        for sd in [0, 5]:
            sd_tag = f"_sd{sd}" if sd > 0 else ""
            name = f"BNAELG_30s_ld16_{_ag_tag(ag)}{sd_tag}"
            training = {"active_g": ag}
            if sd > 0:
                training["skip_distance"] = sd
                training["skip_alpha"] = SKIP_ALPHA
            configs.append(
                {
                    "name": name,
                    "category": "bottleneckgenericaelg",
                    "stacks": {
                        "type": "homogeneous",
                        "block": "BottleneckGenericAELG",
                        "n": 30,
                    },
                    "training": training,
                    "block_params": {"latent_dim": 16},
                    "extra_fields": _extra(
                        arch_family="bottleneckgeneric_aelg",
                        block_type_primary="BottleneckGenericAELG",
                        backbone="AERootBlockLG",
                        stack_pattern="homogeneous",
                        n_stacks=30,
                        latent_dim_cfg=16,
                        skip_distance_cfg=sd,
                        active_g_cfg=_ag_yaml(ag),
                    ),
                }
            )

    # N2: 10-stack baseline
    for ag in ACTIVE_G_VALUES:
        name = f"BNAELG_10s_ld16_{_ag_tag(ag)}"
        configs.append(
            {
                "name": name,
                "category": "bottleneckgenericaelg",
                "stacks": {
                    "type": "homogeneous",
                    "block": "BottleneckGenericAELG",
                    "n": 10,
                },
                "training": {"active_g": ag},
                "block_params": {"latent_dim": 16},
                "extra_fields": _extra(
                    arch_family="bottleneckgeneric_aelg",
                    block_type_primary="BottleneckGenericAELG",
                    backbone="AERootBlockLG",
                    stack_pattern="homogeneous",
                    n_stacks=10,
                    latent_dim_cfg=16,
                    active_g_cfg=_ag_yaml(ag),
                ),
            }
        )

    # N3: ld=32 at 30 stacks (ld sensitivity)
    for sd in [0, 5]:
        sd_tag = f"_sd{sd}" if sd > 0 else ""
        name = f"BNAELG_30s_ld32_ag0{sd_tag}"
        training = {"active_g": False}
        if sd > 0:
            training["skip_distance"] = sd
            training["skip_alpha"] = SKIP_ALPHA
        configs.append(
            {
                "name": name,
                "category": "bottleneckgenericaelg",
                "stacks": {
                    "type": "homogeneous",
                    "block": "BottleneckGenericAELG",
                    "n": 30,
                },
                "training": training,
                "block_params": {"latent_dim": 32},
                "extra_fields": _extra(
                    arch_family="bottleneckgeneric_aelg",
                    block_type_primary="BottleneckGenericAELG",
                    backbone="AERootBlockLG",
                    stack_pattern="homogeneous",
                    n_stacks=30,
                    latent_dim_cfg=32,
                    skip_distance_cfg=sd,
                ),
            }
        )

    return configs


# ── YAML serialization ─────────────────────────────────────────────────────


def _yaml_val(v):
    """Convert a Python value to YAML-safe string."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, str):
        return v
    return str(v)


def _emit_stacks(stacks):
    """Emit the stacks block as YAML lines."""
    st = stacks["type"]
    if st == "homogeneous":
        return (
            f"    stacks:\n"
            f"      type: homogeneous\n"
            f"      block: {stacks['block']}\n"
            f"      n: {stacks['n']}"
        )
    elif st == "prefix_body":
        prefix_str = "[" + ", ".join(stacks["prefix"]) + "]"
        return (
            f"    stacks:\n"
            f"      type: prefix_body\n"
            f"      prefix: {prefix_str}\n"
            f"      body: {stacks['body']}\n"
            f"      total: {stacks['total']}"
        )
    elif st == "alternating":
        blocks_str = "[" + ", ".join(stacks["blocks"]) + "]"
        return (
            f"    stacks:\n"
            f"      type: alternating\n"
            f"      blocks: {blocks_str}\n"
            f"      repeats: {stacks['repeats']}"
        )
    raise ValueError(f"Unknown stacks type: {st}")


def _emit_config(cfg):
    """Emit a single config entry as YAML lines."""
    lines = [f"  - name: {cfg['name']}"]
    lines.append(f"    category: {cfg['category']}")

    # Stacks
    lines.append(_emit_stacks(cfg["stacks"]))

    # Training overrides
    training = cfg.get("training", {})
    if training:
        lines.append("    training:")
        for k, v in training.items():
            lines.append(f"      {k}: {_yaml_val(v)}")

    # Block params overrides
    block_params = cfg.get("block_params", {})
    if block_params:
        lines.append("    block_params:")
        for k, v in block_params.items():
            lines.append(f"      {k}: {_yaml_val(v)}")

    # Extra fields
    ef = cfg.get("extra_fields", {})
    if ef:
        lines.append("    extra_fields:")
        for k, v in ef.items():
            lines.append(f"      {k}: {_yaml_val(v)}")

    return "\n".join(lines)


def _build_header(ds_key, ds_info, n_configs):
    """Build YAML file header."""
    exp_name = f"comprehensive_sweep_{ds_key}"

    header = textwrap.dedent(
        f"""\
    # ============================================================
    # Paper-Ready Comprehensive Sweep — {ds_key.upper()}
    #
    # Designed to prove all major findings in one clean experiment.
    # Random seeds (saved in CSV for reproducibility).
    #
    # Total configs: {n_configs}
    # Architecture groups:
    #   A. Paper baselines (NBEATS-G, NBEATS-I+G)
    #   B. TrendWavelet RootBlock (wavelet + basis_dim sweep)
    #   C. TrendWavelet td sweep (trend_thetas_dim = 3 vs 5)
    #   D. TrendWaveletAE (latent_dim sweep)
    #   E. TrendWaveletAELG (wavelet, latent_dim, skip sweeps)
    #   F. Alternating TrendAELG+WaveletV3AELG (top quality)
    #   G. Alternating TrendAE+WaveletV3AE (stability champion)
    #   H. Alternating Trend+WaveletV3 RootBlock (cross-period)
    #   I. GenericAELG (skip + active_g interaction)
    #   J. TrendWaveletGenericAELG (Pareto optimal)
    #   K. GenericAE (latent_dim contrast)
    #
    # Findings proven:
    #   F1:  active_g stabilizes Generic, can hurt TrendWavelet
    #   F2:  trend_thetas_dim=3 is best
    #   F3:  Wavelet type barely matters
    #   F4:  basis_dim=eq_fcast is best
    #   F5:  Higher latent_dim hurts AE, helps AELG
    #   F6:  Skip connections rescue deep AELG and Generic
    #   F7:  active_g rescues Generic without skip
    #   F8:  Backbone hierarchy: AELG >= RootBlock > AE
    #   F9:  Novel arches match baselines at 10-80x fewer params
    #   F10: Alt TrendAELG+WaveletV3AELG = top quality
    #   F11: TrendAE+WaveletV3AE-activeG = most stable
    #   F12: TrendWaveletAELG/GenericAELG = Pareto optimal
    #
    # Usage:
    #   python experiments/run_from_yaml.py \\
    #       experiments/configs/{exp_name}.yaml --dry-run
    #   python experiments/run_from_yaml.py \\
    #       experiments/configs/{exp_name}.yaml
    #   python experiments/run_from_yaml.py \\
    #       experiments/configs/{exp_name}.yaml --worker-id gpu0
    # ============================================================

    experiment_name: {exp_name}
    category: comprehensive_sweep
    description: >
      Paper-ready comprehensive sweep on {ds_key.upper()}.
      Covers paper baselines, TrendWavelet (RootBlock/AE/AELG),
      alternating TrendAELG+WaveletV3AELG, alternating Trend+WaveletV3,
      GenericAELG skip/active_g interaction, and Pareto-optimal configs.
      Random seeds for unbiased estimation.

    dataset: {ds_info['dataset']}
    periods:
    """
    )

    for p in ds_info["periods"]:
        header += f"  - {p}\n"

    return header


def _build_protocol(ds_info):
    """Build protocol section if needed."""
    proto = ds_info.get("protocol")
    if not proto:
        return ""
    lines = ["\nprotocol:"]
    for k, v in proto.items():
        lines.append(f"  {k}: {_yaml_val(v)}")
    return "\n".join(lines) + "\n"


def _build_training(ds_info):
    """Build training section."""
    return textwrap.dedent(
        f"""\

    training:
      active_g: false
      sum_losses: false
      activation: ReLU
      max_epochs: 200
      patience: 20
      n_blocks_per_stack: 1
      share_weights: true
      loss: {ds_info['loss']}
      optimizer: Adam
      learning_rate: 0.001
      forecast_multiplier: 5
    """
    )


def _build_static_sections():
    """Build lr_scheduler and default block_params."""
    return textwrap.dedent(
        """\
    lr_scheduler:
      warmup_epochs: 15
      eta_min: 0.000001

    block_params:
      wavelet_type: db3

    """
    )


def _build_extra_csv_columns():
    """Build extra_csv_columns section."""
    lines = ["extra_csv_columns:"]
    for col in EXTRA_CSV_COLUMNS:
        lines.append(f"  - {col}")
    return "\n".join(lines) + "\n"


def _build_runs_and_output(ds_key):
    """Build runs, output, logging, hardware, analysis sections."""
    csv_name = f"comprehensive_sweep_{ds_key}_results.csv"
    return textwrap.dedent(
        f"""\

    runs:
      n_runs: 10
      seed_mode: random

    output:
      results_dir: experiments/results
      csv_filename: {csv_name}
      save_predictions: false

    logging:
      wandb:
        enabled: false
        project: nbeats-lightning
      tensorboard: false

    hardware:
      accelerator: auto
      num_workers: 0
      worker_id: ""

    analysis:
      enabled: true
      ranking: true
    """
    )


# ── Category metadata ──────────────────────────────────────────────────────

CATEGORY_LABELS = [
    ("paper_baseline", "Paper Baselines (NBEATS-G, NBEATS-I+G)"),
    ("trendwavelet_rb", "TrendWavelet RootBlock (wavelet + basis_dim)"),
    ("trendwavelet_td", "TrendWavelet td Sweep (trend_thetas_dim 3 vs 5)"),
    ("trendwaveletae", "TrendWaveletAE (latent_dim sweep)"),
    ("trendwaveletaelg", "TrendWaveletAELG (wavelet, latent_dim, skip)"),
    ("alt_aelg", "Alternating TrendAELG+WaveletV3AELG (top quality)"),
    ("alt_ae", "Alternating TrendAE+WaveletV3AE (stability champion)"),
    ("alt_trend_wavelet_rb", "Alternating Trend+WaveletV3 RootBlock (cross-period)"),
    ("genericaelg_skip", "GenericAELG Skip + active_g Interaction"),
    ("trendwaveletgenericaelg", "TrendWaveletGenericAELG (Pareto optimal)"),
    ("genericae", "GenericAE (latent_dim contrast)"),
    ("bottleneckgeneric", "BottleneckGeneric RootBlock (active_g x depth)"),
    ("bottleneckgenericae", "BottleneckGenericAE (latent_dim contrast)"),
    ("bottleneckgenericaelg", "BottleneckGenericAELG (skip + active_g interaction)"),
]


def write_yaml(ds_key, ds_info, all_configs):
    """Write a complete YAML config file for one dataset."""
    n_configs = len(all_configs)
    out_dir = os.path.join(os.path.dirname(__file__), "configs")
    out_path = os.path.join(out_dir, f"comprehensive_sweep_{ds_key}.yaml")

    parts = []
    parts.append(_build_header(ds_key, ds_info, n_configs))
    parts.append(_build_protocol(ds_info))
    parts.append(_build_training(ds_info))
    parts.append(_build_static_sections())
    parts.append(_build_extra_csv_columns())

    # Configs section with category headers
    parts.append(
        "\n# ── Configs ──────────────────────────────────────────────────────────────\n"
    )
    parts.append("configs:")

    for cat_key, cat_label in CATEGORY_LABELS:
        cat_configs = [c for c in all_configs if c["category"] == cat_key]
        if cat_configs:
            parts.append(f"\n  # ── {cat_label} ({len(cat_configs)} configs) ──")
            for cfg in cat_configs:
                parts.append(_emit_config(cfg))

    parts.append(_build_runs_and_output(ds_key))

    content = "\n".join(parts)

    with open(out_path, "w", newline="\n", encoding="utf-8") as f:
        f.write(content)

    print(f"  Written: {out_path} ({n_configs} configs)")
    return out_path


def main():
    all_configs = (
        gen_A_paper_baselines()
        + gen_B_trendwavelet_rootblock()
        + gen_C_trendwavelet_td_sweep()
        + gen_D_trendwaveletae_ld()
        + gen_E_trendwaveletaelg()
        + gen_F_alt_aelg_waveletv3()
        + gen_G_alt_ae_waveletv3()
        + gen_H_alt_trend_waveletv3_rb()
        + gen_I_genericaelg_skip_ag()
        + gen_J_trendwaveletgenericaelg()
        + gen_K_genericae_ld()
        + gen_L_bottleneckgeneric()
        + gen_M_bottleneckgenericae()
        + gen_N_bottleneckgenericaelg()
    )

    print(f"Total configs per dataset: {len(all_configs)}")
    print()

    # Breakdown by category
    from collections import Counter

    cat_counts = Counter(c["category"] for c in all_configs)
    for cat_key, cat_label in CATEGORY_LABELS:
        count = cat_counts.get(cat_key, 0)
        if count:
            print(f"  {cat_label}: {count}")
    print()

    # Check for duplicate names
    names = [c["name"] for c in all_configs]
    dups = [n for n in names if names.count(n) > 1]
    if dups:
        print(f"WARNING: Duplicate config names: {set(dups)}")
    else:
        print(f"  No duplicate names. {len(names)} unique configs.")
    print()

    # Findings coverage matrix
    findings = {
        "F1 active_g": lambda c: "active_g" in str(c.get("training", {})),
        "F2 td sweep": lambda c: c["category"] == "trendwavelet_td",
        "F3 wavelet": lambda c: c.get("extra_fields", {}).get("wavelet_family")
        is not None,
        "F4 basis_dim": lambda c: c.get("extra_fields", {}).get("basis_dim_label")
        is not None,
        "F5 latent_dim": lambda c: c.get("extra_fields", {}).get("latent_dim_cfg")
        is not None,
        "F6 skip": lambda c: c.get("extra_fields", {}).get("skip_distance_cfg", 0) != 0,
        "F8 backbone": lambda c: True,
    }
    print("Findings coverage:")
    for finding, pred in findings.items():
        count = sum(1 for c in all_configs if pred(c))
        print(f"  {finding}: {count} configs")
    print()

    for ds_key, ds_info in DATASETS.items():
        write_yaml(ds_key, ds_info, all_configs)

    print(
        f"\nDone. Generated {len(DATASETS)} YAML files with {len(all_configs)} configs each."
    )


if __name__ == "__main__":
    main()
