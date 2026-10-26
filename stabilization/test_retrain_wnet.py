"""Numerical/data checks independent of TensorFlow."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from retrain_wnet import MEAN, STD, batches, density_gate, expand_files, split_daily_files, input_blocks


class StabilizationTests(unittest.TestCase):
    def test_day_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            for day in range(1, 11):
                (Path(tmp) / f'Xall_202001{day:02d}.nc4').touch()
            train, val = split_daily_files(tmp, .2, 42)
            self.assertEqual((len(train), len(val)), (8, 2))
            self.assertFalse(set(train) & set(val))
            self.assertEqual((train, val), split_daily_files(tmp, .2, 42))

    def test_gridded_sampling(self):
        import xarray as xr
        from retrain_wnet import FEATURES
        with tempfile.TemporaryDirectory() as tmp:
            variables = {}
            for i, name in enumerate(FEATURES):
                dims = ('time', 'lat', 'lon') if name.endswith('_sfc') else ('time', 'lev', 'lat', 'lon')
                shape = (2, 3, 4) if name.endswith('_sfc') else (2, 2, 3, 4)
                variables[name] = (dims, np.full(shape, MEAN[i]))
            ds = xr.Dataset(variables)
            ds['U'].values[:] = np.arange(48).reshape(2, 2, 3, 4)
            path = str(Path(tmp) / 'Xall_20200101.nc4')
            ds.to_netcdf(path)
            first = list(input_blocks(path, samples_per_level=5, seed=1))
            second = list(input_blocks(path, samples_per_level=5, seed=1))
            self.assertEqual(len(first), 4)
            for a, b in zip(first, second):
                self.assertEqual(a.shape, (5, 14))
                np.testing.assert_array_equal(a, b)
                self.assertEqual(len(np.unique(a[:, 2])), 5)
                np.testing.assert_allclose(a[:, 10], MEAN[10])

    def test_gate(self):
        rho = np.array([.4, .23, .175, .12, .05])
        np.testing.assert_allclose(density_gate(rho, .23, .12), [0, 0, .5, 1, 1])
        self.assertTrue(np.all(np.diff(density_gate(rho, .23, .12)) >= 0))
        for start, full in [(.1, .2), (.2, 0), (np.nan, .1)]:
            with self.assertRaises(ValueError):
                density_gate(rho, start, full)

    def test_physical_and_standardized_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = np.tile(MEAN, (5, 1))
            raw[:, 1] = [.4, .23, .175, .12, .05]
            args = SimpleNamespace(standardized=False, surface_level=None,
                                   rho_start=.23, rho_full=.12, batch_size=2)
            path = str(Path(tmp) / 'data.npz')
            np.savez(path, X=raw)
            physical = list(batches([path], args))
            self.assertEqual([len(x) for x, _ in physical], [2, 2, 1])
            np.savez(path, X=(raw - MEAN) / STD)
            args.standardized = True
            scaled = list(batches([path], args))
            for (x, a), (xx, aa) in zip(physical, scaled):
                np.testing.assert_allclose(x, xx)
                np.testing.assert_allclose(a, aa, atol=1e-6)
            self.assertEqual(expand_files([path, path]), [path])
            np.savez(path, X=np.full((2, 14), np.nan))
            with self.assertRaises(ValueError):
                list(batches([path], args))
            np.savez(path, X=np.zeros((2, 13)))
            with self.assertRaises(ValueError):
                list(batches([path], args))


if __name__ == '__main__':
    unittest.main()
