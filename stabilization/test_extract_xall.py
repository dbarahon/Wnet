import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
from extract_xall import make_inputs, write_day, GRAV


class ExtractionTests(unittest.TestCase):
    def sources(self):
        dims = ('time', 'lev', 'lat', 'lon')
        coords = {'time': [0], 'lev': [1, 71, 72], 'lat': [0], 'lon': [0]}
        values = dict(T=250., PL=50000., U=1., V=2., OMEGA=-.1,
                      QV=.02, QI=0., QL=0.)
        asm = xr.Dataset({k: (dims, np.full((1, 3, 1, 1), v))
                          for k, v in values.items()}, coords=coords)
        trb = xr.Dataset({'KM': (dims, np.array([1., 7., 9.]).reshape(1, 3, 1, 1)),
                          'RI': (dims, np.array([2., 8., 10.]).reshape(1, 3, 1, 1))},
                         coords=coords)
        return asm, trb

    def test_density_and_surface(self):
        asm, trb = self.sources()
        result = make_inputs(asm, trb)
        rho = 50000 / (287 * 250)
        np.testing.assert_allclose(result.AIRD, rho)
        np.testing.assert_allclose(result.W, .1 / (rho * GRAV))
        self.assertNotIn('lev', result.KM_sfc.dims)
        self.assertEqual(float(result.KM_sfc.values.item()), 7.)
        self.assertEqual(float(result.RI_sfc.values.item()), 8.)
        self.assertEqual(len(result.data_vars), 14)
        with self.assertRaises(ValueError):
            make_inputs(asm, trb.assign_coords(time=[1]))

    def test_write_and_training_reader(self):
        from retrain_wnet import input_blocks
        asm, trb = self.sources()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'Y2020' / 'M01'
            directory.mkdir(parents=True)
            asm.to_netcdf(directory / 'MERRA2.tavg3_3d_asm_Nv.20200101.nc4')
            trb.to_netcdf(directory / 'MERRA2.tavg3_3d_trb_Ne.20200101.nc4')
            args = SimpleNamespace(base_dir=tmp, outdir=tmp, overwrite=False, surface_level=71)
            write_day(pd.Timestamp('2020-01-01'), args)
            output = str(Path(tmp) / 'Xall_20200101.nc4')
            x = list(input_blocks(output))[0]
            self.assertEqual(x.shape, (3, 14))
            np.testing.assert_allclose(x[:, 11], 7.)
            with xr.open_dataset(output) as result:
                self.assertEqual(result.T.dtype, np.float32)
                self.assertNotIn('lev', result.AIRD_sfc.dims)
            self.assertFalse(Path(output + '.tmp').exists())


if __name__ == '__main__':
    unittest.main()
