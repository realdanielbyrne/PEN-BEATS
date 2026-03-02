"""Comprehensive analysis of all M4 benchmark results for wavelet study planning."""
import pandas as pd
import numpy as np

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
pd.set_option('display.width', 140)
pd.set_option('display.max_colwidth', 50)

NUM_COLS = ['smape','mase','mae','mse','owa','n_params','training_time_seconds',
            'epochs_trained','best_val_loss','final_val_loss','final_train_loss',
            'best_epoch','loss_ratio','norm_mae','norm_mse','basis_dim','basis_offset']

def load_csv(path):
    df = pd.read_csv(path)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ── 1. Unified Benchmark ──────────────────────────────────────────
print("=" * 80)
print("1. UNIFIED BENCHMARK (experiments/results/m4/unified_benchmark_results.csv)")
print("=" * 80)
ub = load_csv('experiments/results/m4/unified_benchmark_results.csv')
print(f"Total rows: {len(ub)}")
print(f"Periods: {sorted(ub['period'].unique())}")
print(f"Experiments: {sorted(ub['experiment'].unique())}")
print(f"Config names: {sorted(ub['config_name'].unique())}")
print()

# Per-config aggregate across all periods
agg = ub.groupby(['config_name', 'period']).agg(
    smape_mean=('smape','mean'), smape_std=('smape','std'),
    owa_mean=('owa','mean'), runs=('smape','count'),
    epochs_mean=('epochs_trained','mean'),
).reset_index()

# Show Yearly results (comparable to wavelet search)
yearly = agg[agg['period'] == 'Yearly'].sort_values('smape_mean')
print("── Yearly (sorted by sMAPE) ──")
print(yearly[['config_name','smape_mean','smape_std','owa_mean','runs','epochs_mean']].to_string(index=False))
print()

# All-period summary (mean across periods)
allp = agg.groupby('config_name').agg(
    smape_mean=('smape_mean','mean'),
    owa_mean=('owa_mean','mean'),
    periods=('period','count'),
).sort_values('smape_mean')
print("── All-period mean sMAPE ──")
print(allp.head(20).to_string())
print()

# ── 2. Block Benchmark ────────────────────────────────────────────
print("=" * 80)
print("2. BLOCK BENCHMARK (experiments/results/m4/block_benchmark_results.csv)")
print("=" * 80)
bb = load_csv('experiments/results/m4/block_benchmark_results.csv')
print(f"Total rows: {len(bb)}")

bb_agg = bb.groupby(['config_name','period']).agg(
    smape_mean=('smape','mean'), smape_std=('smape','std'),
    owa_mean=('owa','mean'), runs=('smape','count')
).reset_index()

bb_yearly = bb_agg[bb_agg['period'] == 'Yearly'].sort_values('smape_mean')
print("── Yearly ──")
print(bb_yearly[['config_name','smape_mean','smape_std','owa_mean','runs']].to_string(index=False))
print()

# ── 3. Wavelet V3 Benchmark ──────────────────────────────────────
print("=" * 80)
print("3. WAVELET V3 BENCHMARK (experiments/results/m4/wavelet_v3_benchmark_results.csv)")
print("=" * 80)
wv3 = load_csv('experiments/results/m4/wavelet_v3_benchmark_results.csv')
print(f"Total rows: {len(wv3)}")

wv3_agg = wv3.groupby(['config_name','period']).agg(
    smape_mean=('smape','mean'), smape_std=('smape','std'),
    owa_mean=('owa','mean'), runs=('smape','count')
).reset_index()
wv3_yearly = wv3_agg[wv3_agg['period'] == 'Yearly'].sort_values('smape_mean')
print("── Yearly ──")
print(wv3_yearly[['config_name','smape_mean','smape_std','owa_mean','runs']].to_string(index=False))
print()

# ── 4. Wavelet Search (final rounds) ─────────────────────────────
print("=" * 80)
print("4. WAVELET SEARCH — ROUND 4 FINAL (experiments/results/m4/wavelet_search_results.csv)")
print("=" * 80)
ws = load_csv('experiments/results/m4/wavelet_search_results.csv')
r4 = ws[ws['search_round'].astype(str) == '4']
r4_agg = r4.groupby('config_name').agg(
    smape_mean=('smape','mean'), smape_std=('smape','std'),
    owa_mean=('owa','mean'), runs=('smape','count')
).sort_values('smape_mean')
print("── Round 4 baseline ──")
print(r4_agg.to_string())
print()

r4ag = ws[ws['search_round'].astype(str) == '4_ag']
if len(r4ag) > 0:
    r4ag_agg = r4ag.groupby('config_name').agg(
        smape_mean=('smape','mean'), smape_std=('smape','std'),
        owa_mean=('owa','mean'), runs=('smape','count')
    ).sort_values('smape_mean')
    print("── Round 4 active_g=forecast ──")
    print(r4ag_agg.to_string())
    print()

# ── 5. Cross-source comparison ────────────────────────────────────
print("=" * 80)
print("5. CROSS-SOURCE COMPARISON — YEARLY BEST CONFIGS")
print("=" * 80)
rows = []
for src, df_agg in [('unified_bench', agg[agg['period']=='Yearly']),
                      ('block_bench', bb_yearly),
                      ('wavelet_v3', wv3_yearly),
                      ('wavelet_search_r4', r4_agg.reset_index())]:
    if len(df_agg) == 0:
        continue
    best_idx = df_agg['smape_mean'].idxmin()
    best = df_agg.loc[best_idx]
    rows.append({
        'source': src,
        'best_config': best['config_name'],
        'smape': round(best['smape_mean'], 4),
        'owa': round(best['owa_mean'], 4),
        'runs': int(best['runs']),
    })
comp = pd.DataFrame(rows)
print(comp.to_string(index=False))
print()

# ── 6. Hyperparameter patterns in search data ────────────────────
print("=" * 80)
print("6. SEARCH DATA — HYPERPARAMETER MARGINALS (all rounds)")
print("=" * 80)
for col in ['arch_pattern','n_stacks_requested','basis_dim','basis_offset','wavelet_family']:
    if col in ws.columns:
        grp = ws.groupby(col)['smape'].agg(['mean','std','count']).sort_values('mean')
        print(f"\n── {col} ──")
        print(grp.to_string())

# ── 7. Unified benchmark: baseline vs active_g ───────────────────
print()
print("=" * 80)
print("7. UNIFIED BENCHMARK — BASELINE vs ACTIVE_G (Yearly)")
print("=" * 80)
ub_y = ub[ub['period'] == 'Yearly'].copy()
for exp in ['baseline', 'activeG_fcast']:
    sub = ub_y[ub_y['experiment'] == exp]
    if len(sub) == 0:
        continue
    agg_e = sub.groupby('config_name').agg(
        smape_mean=('smape','mean'), smape_std=('smape','std'),
        owa_mean=('owa','mean'), runs=('smape','count'),
        diverged_pct=('diverged', lambda x: (x.astype(str).str.lower() == 'true').mean()*100),
    ).sort_values('smape_mean')
    print(f"\n── {exp} ──")
    print(agg_e.head(10).to_string())

# ── 8. Unified benchmark convergence patterns ────────────────────
print()
print("=" * 80)
print("8. CONVERGENCE PATTERNS — STOPPING REASON (Yearly, baseline)")
print("=" * 80)
bl_y = ub_y[ub_y['experiment'] == 'baseline']
if 'stopping_reason' in bl_y.columns:
    stop = bl_y.groupby(['config_name','stopping_reason']).size().unstack(fill_value=0)
    print(stop.to_string())
    print()
    print("Mean epochs by config:")
    ep = bl_y.groupby('config_name')['epochs_trained'].agg(['mean','std']).sort_values('mean')
    print(ep.to_string())

# ── 9. Top configs: multi-period consistency ──────────────────────
print()
print("=" * 80)
print("9. MULTI-PERIOD CONSISTENCY — Top unified bench configs")
print("=" * 80)
bl = ub[ub['experiment'] == 'baseline']
pivot = bl.groupby(['config_name','period'])['smape'].mean().unstack()
if len(pivot) > 0:
    pivot['overall_mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('overall_mean')
    print(pivot.head(12).round(3).to_string())

