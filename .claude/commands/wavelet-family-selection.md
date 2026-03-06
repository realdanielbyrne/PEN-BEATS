# Wavelet Family Selection by Forecast / Target Length

## Core Rule

Choose wavelet families by the **actual target length used to build the basis** (`backcast_length` for backcast basis, `forecast_length` for forecast basis), not just by model name.

- Short targets need **short-support filters**
- Long-support filters only make sense when the target is long enough to permit real multilevel decomposition
- In WaveletV3, if `pywt.dwt_max_level(target_length, dec_len) == 0`, the code now keeps `level=0` instead of forcing an invalid level-1 decomposition

That means very short targets with long filters still get an orthonormal basis, but it is effectively an **approximation-only** basis rather than a true multiscale wavelet decomposition.

---

## Practical Thresholds

Rule of thumb: you usually want `target_length >= 2 * (dec_len - 1)` to allow at least one non-boundary DWT level.

| Family | Example blocks | Filter length (`dec_len`) | Rough minimum target length for level ≥ 1 |
|---|---|---:|---:|
| Haar | `HaarWaveletV3*` | 2 | 2 |
| DB2 / Sym2 | `DB2WaveletV3*`, `Symlet2WaveletV3*` | 4 | 6 |
| DB3 / Sym3 / Coif1 | `DB3WaveletV3*`, `Symlet3WaveletV3*`, `Coif1WaveletV3*` | 6 | 10 |
| DB4 | `DB4WaveletV3*` | 8 | 14 |
| Coif2 | `Coif2WaveletV3*` | 12 | 22 |
| Coif3 | `Coif3WaveletV3*` | 18 | 34 |
| DB10 / Sym10 | `DB10WaveletV3*`, `Symlet10WaveletV3*` | 20 | 38 |
| DB20 / Sym20 | `DB20WaveletV3*`, `Symlet20WaveletV3*` | 40 | 78 |
| Coif10 | `Coif10WaveletV3*` | 60 | 118 |

---

## Selection Guide

### Very short horizons / targets (`<= 8`)
- Prefer: `Haar`, `DB2`, `Sym2`
- Usually avoid: `DB3+`, `Coif*`, `DB10+`, `Sym10+`

### Short targets (`9-16`)
- Good defaults: `DB2`, `DB3`, `Sym2`, `Sym3`, `DB4`
- Use caution with: `Coif2+`, `DB10+`, `Sym10+`

### Medium targets (`17-37`)
- Safe: `DB3`, `DB4`, `Sym3`, `Coif1`, `Coif2`, `Coif3`
- `DB10` / `Sym10` become reasonable near the top of this range

### Long targets (`38-77`)
- `DB10` and `Sym10` are appropriate
- `DB20` / `Sym20` are still usually too long below ~78

### Very long targets (`>= 78`)
- `DB20` and `Sym20` become viable
- `Coif10` still needs a much longer target (~118+)

---

## Important Practical Note

N-BEATS often uses `backcast_length = 4 * forecast_length`.

So a configuration can have:

- a **reasonable backcast wavelet basis**, but
- a **forecast basis that is too short** for the same family to behave multiscale

Example:

- `forecast_length=6`, `backcast_length=24`
- `DB20WaveletV3` will still be far too long for the forecast basis

When in doubt, pick the family using the **shorter of the two target lengths**.

---

## Recommendation

If you want a generally safe default across common short-horizon forecasting setups, start with:

1. `HaarWaveletV3`
2. `DB2WaveletV3`
3. `DB3WaveletV3`

Only move to longer-support families when the target lengths are long enough that you actually benefit from deeper multiscale structure.