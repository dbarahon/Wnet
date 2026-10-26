# Density-based Wnet stabilization

Fine-tune the trained 14-input Wnet without retraining the GAN. Two frozen
models supply cell-by-cell targets:

    target = (1 - alpha) * original_Wnet(X) + alpha * prior(X)
    u = clip((rho_start - AIRD) / (rho_start - rho_full), 0, 1)
    alpha = u*u*(3 - 2*u)

AIRD is physical air density in kg m-3. Above rho_start the target preserves
Wnet; below rho_full it equals the prior. Between them it blends smoothly.
Choose thresholds from your profiles near the desired upper-atmosphere
transition. Density is not an exact height coordinate. The numbers below are
illustrative, not calibrated thresholds for 14 km.

## Run from the repository root

Use the TensorFlow/Keras environment that can load your original HDF5 models.
Dependencies: numpy, tensorflow, h5py; also xarray and a NetCDF backend for NetCDF.
No observations or nature-run target labels are needed.

```bash
python stabilization/retrain_wnet.py \
  --wnet Wnet.h5 \
  --prior data/Wnet_prior.h5 \
  --train '/path/to/train/*.npz' \
  --validation '/path/to/held_out/*.npz' \
  --rho-start 0.23 --rho-full 0.12 \
  --output-dir stabilization/run01
```

The default trains the last two Dense layers at learning rate 1e-5. Use
`--trainable-dense 1` for output-only adjustment or `--trainable-dense 0` to
unfreeze all layers. The architecture and input normalization remain the same.
The student starts with the supplied Wnet weights; both teachers stay frozen.

## Inputs

NPZ files contain `X` with shape `(samples, 14)` in this exact column order:

```text
T AIRD U V W KM RI QV QI QL AIRD_sfc KM_sfc RI_sfc QV_sfc
```

Use K, kg m-3, m s-1, m2 s-1, dimensionless RI, and kg kg-1 mixing ratios,
consistent with Wnet.F90. By default X contains physical values. Pass
`--standardized` only for NPZ arrays already standardized with the original
Wnet constants. Density is then reconstructed before computing the gate.
Both teachers must expect the same 14 standardized inputs and physical sigmaW
in m s-1; this script is not for the alternative-input Wnetb model.

```python
import numpy as np
np.savez('samples.npz', X=X_physical)
```

NetCDF files may instead provide the 14 named variables on a common grid.
Physical units are required. If the four `_sfc` fields are missing, supply
`--surface-level 71` for the original GEOS model-level convention, or the
appropriate **coordinate value** for your data. The script never guesses the
surface from level ordering. Surface variables are broadcast across levels.
NetCDF is read one time slice at a time; each slice (or complete NPZ file) must
fit in memory. Use moderate-sized shards for global datasets. Inputs must
already be collocated; this script does not interpolate separate met files.
Invalid/nonfinite cells fail explicitly; remove missing or below-ground cells
before training.

Use representative full-column data, including rare conditions and all three
density regions. Training and validation must each cover all three regions.
Split by date or independent atmospheric cases, not randomly neighboring cells;
file paths are checked for overlap, but duplicate contents cannot be detected.
Training shuffles files and samples within each block each epoch.

## Loss and outputs

The loss is mean squared error divided cell by cell by
`max(target, scale_floor)**2`, with `--scale-floor 0.1` m s-1 by default.
This balances relative deviations without giving arbitrarily large weight to
near-zero targets. Negative blended targets fail rather than being silently
clipped. This blends toward the prior even where it is higher than Wnet.

A new output directory receives:

- `Wnet_stabilized.h5`: best validation checkpoint, including epoch zero if
  fine-tuning never improves validation loss; load with `compile=False`.
- `config.json`: arguments, resolved input paths, feature order and normalization.
- `history.jsonl`: baseline and epoch metrics, including target MAE, departure
  from Wnet, departure from prior, and negative prediction counts in each region.

Early stopping uses total validation loss. Review the regional metrics before
accepting a checkpoint: shared weights mean preservation and relaxation are
approximate, and a better aggregate score can hide lower-atmosphere drift.
This is distillation, not a hard output bound or a guarantee outside sampled
conditions. Check upper tails and independent observational performance before
using the result operationally. The existing models and Fortran weights are
not overwritten; export new Fortran weights separately after validation.

## Checks

```bash
python -m unittest discover -s stabilization -p 'test_*.py'
```

These checks exercise density gating, physical/standardized equivalence,
NPZ batching, and invalid-input handling without TensorFlow. Actual model
loading and training require your TensorFlow environment and training data.
