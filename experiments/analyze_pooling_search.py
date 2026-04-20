"""Analyze the NHiTS pooling + freq_downsample grid search results."""
import pandas as pd
import numpy as np
from pathlib import Path

CSV = Path('experiments/results/nhits_pooling_search.csv')
df = pd.read_csv(CSV)

# Parse pool / freq strings from config name
def parse_pool_freq(name):
    # e.g. Pool-2x2x2_Freq-168x24x1
    pool_str, freq_str = name.replace('Pool-', '').split('_Freq-')
    pool = tuple(int(x) for x in pool_str.split('x'))
    freq = tuple(int(x) for x in freq_str.split('x'))
    return pool, freq

df[['pool', 'freq']] = df['config_name'].apply(
    lambda n: pd.Series(parse_pool_freq(n))
)
df['pool_str'] = df['pool'].apply(lambda t: str(list(t)))
df['freq_str'] = df['freq'].apply(lambda t: str(list(t)))

# Aggregate across runs
agg = (
    df.groupby(['config_name', 'pool_str', 'freq_str', 'horizon'])
    .agg(mse_mean=('mse', 'mean'),
         mse_std=('mse', 'std'),
         mae_mean=('mae', 'mean'),
         n=('mse', 'size'))
    .reset_index()
    .sort_values(['horizon', 'mse_mean'])
)

# Paper baselines (NHiTS Table 3, Weather multivariate)
PAPER = {96: 0.158, 192: 0.211, 336: 0.274, 720: 0.351}

print('\n=== TOP 5 CONFIGS PER HORIZON (by MSE) ===')
for H in sorted(df['horizon'].unique()):
    sub = agg[agg['horizon'] == H].head(5)
    print(f'\nH={H}  (paper MSE = {PAPER[H]:.3f})')
    print(f"{'pool':<14}{'freq':<14}{'mse':>8}{'±std':>8}{'mae':>8}{'gap%':>8}")
    for _, r in sub.iterrows():
        gap = 100 * (r.mse_mean - PAPER[H]) / PAPER[H]
        print(f"{r.pool_str:<14}{r.freq_str:<14}{r.mse_mean:>8.4f}"
              f"{r.mse_std:>8.4f}{r.mae_mean:>8.4f}{gap:>+7.1f}%")

print('\n=== BEST PER HORIZON ===')
best_per_h = agg.loc[agg.groupby('horizon')['mse_mean'].idxmin()]
print(best_per_h[['horizon', 'pool_str', 'freq_str', 'mse_mean', 'mae_mean']])

print('\n=== MARGINAL: MSE aggregated across POOL kernels ===')
pool_marg = (
    agg.groupby(['horizon', 'pool_str'])['mse_mean']
    .mean().unstack('horizon').round(4)
)
print(pool_marg)

print('\n=== MARGINAL: MSE aggregated across FREQ schedules ===')
freq_marg = (
    agg.groupby(['horizon', 'freq_str'])['mse_mean']
    .mean().unstack('horizon').round(4)
)
print(freq_marg)

print('\n=== FULL GRID (mean MSE) per horizon ===')
for H in sorted(df['horizon'].unique()):
    sub = agg[agg['horizon'] == H]
    pivot = sub.pivot(index='pool_str', columns='freq_str',
                      values='mse_mean').round(4)
    print(f'\nH={H}')
    print(pivot)
