"""Numerical/data checks independent of TensorFlow."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from retrain_wnet import MEAN, STD, batches, density_gate, expand_files


class StabilizationTests(unittest.TestCase):
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
