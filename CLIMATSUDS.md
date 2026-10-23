# The `climatsuds` branch of SBCK

The branch `climatsuds` of this fork is upstream SBCK (branch `version-2.X.Y`, 2.0.0a76, commit `6d2b84a`) with the fixes below. Each fix is on its own branch and merged into `climatsuds`. Each build is tagged `climatsuds-YYYY-MM-DD[.n]`, and `SBCK.__version__` then reads `2.0.0a76+climatsuds.YYYYMMDD[.n]`.

## Changes

| Branch | What changes |
|---|---|
| `feat/clim-flags` | `SBCK.clim` collects the diagnostic flags of the methods (attributes `flag_names` / `flags`); with `return_flags=True` the four `apply_bcm*` / `zapply_bcm*` functions return a table of the flagged cells, groups and windows. |
| `fix/ltr-unchanged-on-pr4` | `LimitTailsRatio` leaves a tail unchanged, and flags it (`LimitTailsRatio.bound_inside_right` / `_left`), when its bound is inside the tail; before, the order of the tail was reversed, or NaN with ties. |
| `fix/qm-outside-calibration` | `QM` no longer raises `ValueError` when all the values to correct are outside the calibration range: they are mapped as when only some are outside, and the column is flagged `QM.outside_calibration`. `dOTC1d` raises a clear `ValueError` when X0 is constant. |
| `fix/refit-window-parallel-rank` | A second `fit` starts from the parameters given to `__init__`; `seas_cycle="window"` no longer changes the projection window; rank shuffles work with SciPy >= 1.18; `apply_bcm*` accept dask-backed inputs; `zapply_bcm*` start one dask cluster per call and close it, also on error. |
| `fix/nan-robustness` | NaN values stay NaN in `SSR`, `rv_empirical_ratio`, `QM`, `LFLogistic` and `PreserveOrder`, instead of a plausible value; `PreserveOrder` no longer warns at each call. |
| `fix/dotc1d-reinfer` | `dOTC1d.predict` with `reinfer_X1` / `reinfer_X0` no longer raises `AttributeError`. |
| `fix/gpd-degenerate-tail` | `rv_empirical_gpd`: a tail that cannot be fitted (fewer than 2 values, or ties such as the zeros of precipitation) is degenerate at its location instead of NaN; `scalel` / `shapel` / `scaler` / `shaper` work. |
| `fix/cdft-unbounded-laws` | With laws of unbounded support (scipy.stats laws, `rv_empirical_gpd`) and `oob` other than `"None"`, the correction of the projection period was NaN without any message; `CDFt` now raises a clear `ValueError` there. The fit and the correction of the calibration period are unchanged. |
| `test/climatsuds` | `tests/test_climatsuds.py`: regression tests of these changes (they fail on upstream SBCK by design). |

Wherever upstream SBCK gives a result, `climatsuds` gives bitwise the same result, except where that result was the bug: a `LimitTailsRatio` tail with its bound inside, NaN values to correct, a second `fit` of the same object, and `seas_cycle="window"`. Cases that raised or returned only NaN now give a result or a clear error.

## Known issues, not changed

- `import SBCK.clim` imports zxarray, which calls `logging.captureWarnings(True)`: Python warnings are dropped unless logging is configured (`logging.basicConfig()` shows them).
- Stationary methods such as `QM` cannot be used in `SBCK.clim`, which always calls `fit(Y0, X0, X1)`.
- `LimitTailsRatio` with NaN in the correction: the column is NaN, or flagged `bound_inside_*`.

## Tests

```
cd tests
python -m unittest test_climatsuds
bash test_all.sh
```
