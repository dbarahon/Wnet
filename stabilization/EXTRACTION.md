# Extract daily raw inputs

From the repository root:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python stabilization/extract_xall.py \
  --years 2020 2021 --n-days 150 --seed 1234 --outdir /path/to/Xall_daily \
  --workers 16 --threads-per-worker 1
```

Dependencies: numpy, pandas, xarray, netCDF4, dask[array,distributed].
Output is always physical/raw; `--raw` is accepted for compatibility. The
retraining script applies Wnet normalization. Use CPU nodes for extraction.
Days run sequentially; workers process chunks within each day.

For a short benchmark use `--dates 20200101 20200102 20200103` instead of years.
Compare 8, 16 and 32 workers with separate output directories. `--workers 0`
uses the default local Dask scheduler without a distributed cluster.
`--memory-limit` sets a per-worker limit, and `--local-directory` selects local
scratch for spills. Memory and filesystem throughput determine useful scaling.

The extractor reads only the exact ASM and TRB files for each date. It uses
`AIRD = PL / (287.0 * T)` with actual temperature, matching Wnet_prior.py,
and `W = -OMEGA / (AIRD * 9.80665)`.
KM and RI are selected using the ASM model-level labels under the established
Ne/Nv convention; no vertical interpolation is applied. Other shared
coordinates must match exactly, so a mismatched time/grid fails explicitly.

The four near-surface features come from lev=71 (override with
`--surface-level`). They retain only time/lat/lon dimensions, saving disk space.
The existing retraining reader broadcasts them over levels. All 14 feature
names are provided; output is float32, uncompressed NetCDF4. Source files stay
open until the write finishes and are closed on errors. Completed temporary
files are atomically renamed; existing outputs are skipped unless --overwrite
is set. Failed dates are reported with a nonzero exit status after processing.
Do not run overlapping date jobs against the same output directory.

For training, hold out whole daily files using separate train/validation globs.
No observations or Wstd labels are needed for the two-teacher stabilization.

Tests (including a synthetic NetCDF write and retraining-reader round trip):

```bash
python -m unittest discover -s stabilization -p 'test_*.py'
```

## Random date selection

`--n-days N` selects N total days uniformly without replacement from the union
of all selected years (not N per year). It is required unless `--dates` is used,
preventing accidental full-year extraction. Dates are processed chronologically
and printed at startup. The same years, count and seed reproduce the selection;
year order and repeated years do not change it. Leap days are included.
`--dates` supplies exact dates and cannot be combined with `--n-days`.
Existing output files still count toward the selected N and are skipped as usual.
Missing input days are reported as failures, not silently replaced by other dates.
