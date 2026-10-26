#!/usr/bin/env python3
"""Extract raw daily Wnet inputs from MERRA-2, using original prior conventions."""
import argparse
import random
from contextlib import ExitStack
from pathlib import Path
import traceback

import pandas as pd
import xarray as xr

RD = 287.0  # Match Wnet_prior.py: PL / (287.0 * T), not virtual temperature.
GRAV = 9.80665
VARS_ASM = ['T', 'PL', 'U', 'V', 'OMEGA', 'QV', 'QI', 'QL']
VARS_IN = ['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL']
SURF_VARS = ['AIRD', 'KM', 'RI', 'QV']
CHUNKS = {'lat': -1, 'lon': -1, 'lev': 1, 'time': 8}
ENCODING = {'dtype': 'float32', '_FillValue': -9999.0, 'zlib': False}


def make_inputs(asm, trb, surface_level=71):
    """Lazy transformation; caller keeps both source datasets open until write."""
    met = asm[VARS_ASM].copy()
    met['AIRD'] = met.PL / (RD * met.T)
    met['AIRD'].attrs = {'long_name': 'air density using actual temperature',
                         'units': 'kg m-3', 'formula': 'PL / (287.0 * T)'}
    met['W'] = -met.OMEGA / (met.AIRD * GRAV)
    met['W'].attrs = {'long_name': 'vertical velocity', 'units': 'm s-1',
                      'positive': 'up'}
    met = met.drop_vars(['PL', 'OMEGA'])
    # Retain the established Ne/Nv level-label convention; no interpolation.
    turbulence = trb[['KM', 'RI']].sel(lev=met.lev)
    met, turbulence = xr.align(met, turbulence, join='exact', copy=False)
    result = xr.merge([met, turbulence], join='exact')[VARS_IN]
    for name in SURF_VARS:
        result[name + '_sfc'] = result[name].sel(lev=surface_level, drop=True)
        result[name + '_sfc'].attrs = dict(result[name].attrs)
        result[name + '_sfc'].attrs['description'] = (
            f'{name} at near-surface lev={surface_level}; broadcast at training time')
    result.attrs.update(source='MERRA2 daily Wnet input extraction',
                        standardized='false', surface_level=surface_level,
                        density_formula='PL / (287.0 * T)',
                        turbulence_levels='Selected by matching model-level labels; no interpolation')
    return result.unify_chunks()


def write_day(date, args):
    tag = date.strftime('%Y%m%d')
    directory = Path(args.base_dir) / f'Y{date.year}' / date.strftime('M%m')
    asm = directory / f'MERRA2.tavg3_3d_asm_Nv.{tag}.nc4'
    trb = directory / f'MERRA2.tavg3_3d_trb_Ne.{tag}.nc4'
    output = Path(args.outdir) / f'Xall_{tag}.nc4'
    if output.exists() and not args.overwrite:
        print(f'Skipping existing file: {output}', flush=True)
        return
    for source in (asm, trb):
        if not source.is_file():
            raise FileNotFoundError(f'Missing input: {source}')
    temporary = output.with_suffix('.nc4.tmp')
    try:
        with ExitStack() as stack:
            met = stack.enter_context(xr.open_dataset(asm, chunks=CHUNKS))
            turbulence = stack.enter_context(xr.open_dataset(trb, chunks=CHUNKS))
            data = make_inputs(met, turbulence, args.surface_level)
            data.attrs.update(date=date.strftime('%Y-%m-%d'),
                              source_files=f'{asm} | {trb}')
            print(f'Writing {output}', flush=True)
            data.to_netcdf(temporary, mode='w', format='NETCDF4', engine='netcdf4',
                           encoding={v: ENCODING.copy() for v in data.data_vars})
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f'Finished {output}', flush=True)


def select_dates(years, explicit_dates=None, n_days=None, seed=1234):
    """Sample uniformly without replacement from the union of selected years."""
    if explicit_dates:
        if n_days is not None:
            raise ValueError('--dates and --n-days cannot be combined')
        return sorted(set(pd.to_datetime(explicit_dates, format='%Y%m%d')))
    if n_days is None or n_days < 1:
        raise ValueError('Supply --n-days with a positive total number of days')
    candidates = [date for year in sorted(set(years))
                  for date in pd.date_range(f'{year}-01-01', f'{year}-12-31')]
    if n_days > len(candidates):
        raise ValueError(f'--n-days exceeds the {len(candidates)} available dates')
    return sorted(random.Random(seed).sample(candidates, n_days))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years', nargs='+', type=int, default=[2020, 2021])
    parser.add_argument('--dates', nargs='+', help='Optional explicit YYYYMMDD dates; overrides years')
    parser.add_argument('--n-days', type=int,
                        help='Total random days across all selected years, without replacement')
    parser.add_argument('--seed', type=int, default=1234, help='Date sampling seed')
    parser.add_argument('--base-dir', default='/css/merra2/MERRA2_all')
    parser.add_argument('--outdir', default='./Xall_daily')
    parser.add_argument('--surface-level', type=int, default=71,
                        help='Near-surface model-level label (default 71)')
    parser.add_argument('--raw', action='store_true', help='Compatibility flag; output is always raw')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--threads-per-worker', type=int, default=1)
    parser.add_argument('--memory-limit', default='auto', help='Dask memory limit per worker')
    parser.add_argument('--local-directory', help='Local scratch directory for Dask spill files')
    args = parser.parse_args()
    if args.workers < 0 or args.threads_per_worker < 1:
        parser.error('workers must be nonnegative and threads-per-worker positive')
    try:
        dates = select_dates(args.years, args.dates, args.n_days, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Selected {len(dates)} dates (seed={args.seed}): ' +
          ', '.join(date.strftime('%Y%m%d') for date in dates), flush=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    failed = []
    with ExitStack() as stack:
        if args.workers:
            from dask.distributed import Client, LocalCluster
            cluster = stack.enter_context(LocalCluster(
                n_workers=args.workers, threads_per_worker=args.threads_per_worker,
                processes=True, memory_limit=args.memory_limit,
                local_directory=args.local_directory))
            client = stack.enter_context(Client(cluster))
            print(client, flush=True)
        for date in dates:
            try:
                write_day(date, args)
            except Exception:
                failed.append(date.strftime('%Y%m%d'))
                traceback.print_exc()
    if failed:
        raise SystemExit('Failed dates: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
