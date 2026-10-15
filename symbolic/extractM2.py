#!/usr/bin/env python3
"""
extract_pysr_merra2_wstd_hps.py

Extract random MERRA-2 input samples plus matching Wstd output into one NetCDF
file for PySR symbolic-regression training.

This version adds these variables from the same MERRA-2 inst3_3d_asm_Np file
that contains T:

  H   : height / edge_heights [m]
  PS  : surface pressure [Pa]
  P   : pressure coordinate broadcast from lev [hPa]

Default paths are the current production locations, but can be overridden by
MERRAPATH and SWPATH environment variables or by command-line arguments.

csh example:

  setenv MERRAPATH /css/merra2/MERRA2_all
  setenv SWPATH    /css/gmao/Wstd-MERRA2/pub/Wnet/MERRA2

  python extract_pysr_merra2_wstd_hps.py \
      --ndays 150 \
      --year0 2010 --year1 2020 \
      --samples-per-day-level 4000 \
      --seed 1234 \
      --out pysr_merra2_wstd_hps_150d_2010_2020.nc \
      --overwrite

Default output format is a compact sample-table NetCDF:

  dimension: sample

  variables:
    T, P, H, AIRD, U, V, W, QV, QI, QL,
    PS, PBLH, TQV, AIRD_surf, Wstd

  coordinates per sample:
    time, lev, lat, lon, yyyymmdd

where:
  P         = lev, broadcast to each sample [hPa]
  AIRD      = P[Pa] / (Rd*T)
  W         = -OMEGA / (g*AIRD)
  AIRD_surf = PS_from_inst3_3d_asm_Np / (Rd*T2M)
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

RD = 287.0
GRAV = 9.81

DEFAULT_MERRAPATH = "/css/merra2/MERRA2_all"
DEFAULT_SWPATH = "/css/gmao/Wstd-MERRA2/pub/Wnet/MERRA2"

# Variables read from inst3_3d_asm_Np.  PS is 2-D in that file, but is later
# broadcast to time,lev,lat,lon so every PySR row has PS(sample).
VARS_FROM_3D_FILE = ["T", "H", "U", "V", "OMEGA", "QV", "QI", "QL", "PS"]

FEATURES_3D_OUT = ["T", "P", "H", "AIRD", "U", "V", "W", "QV", "QI", "QL", "PS"]
SURF_VARS_OUT = ["PBLH", "TQV", "AIRD_surf"]
ALL_FEATURES = FEATURES_3D_OUT + SURF_VARS_OUT
TARGET = "Wstd"

COLL_3D = "inst3_3d_asm_Np"
COLL_2D_FLX = "tavg1_2d_flx_Nx"
COLL_2D_ASM = "inst1_2d_asm_Nx"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract random MERRA-2 input/output samples for PySR. "
            "Defaults use MERRAPATH/SWPATH if set, otherwise production paths."
        )
    )
    p.add_argument(
        "--merra-path",
        default=os.environ.get("MERRAPATH", DEFAULT_MERRAPATH),
        help=f"MERRA-2 root path. Default: $MERRAPATH or {DEFAULT_MERRAPATH}",
    )
    p.add_argument(
        "--sw-path",
        default=os.environ.get("SWPATH", DEFAULT_SWPATH),
        help=f"Wstd output root. Default: $SWPATH or {DEFAULT_SWPATH}",
    )
    p.add_argument(
        "--out",
        default="pysr_merra2_wstd_hps_150d_2010_2020.nc",
        help="Output NetCDF file",
    )
    p.add_argument("--ndays", type=int, default=150, help="Number of random valid days to extract")
    p.add_argument("--year0", type=int, default=2010, help="First year, inclusive")
    p.add_argument("--year1", type=int, default=2020, help="Last year, inclusive")
    p.add_argument("--seed", type=int, default=1234, help="Random seed for day and gridpoint sampling")
    p.add_argument("--pmin", type=float, default=10.0, help="Keep pressure levels lev > pmin hPa")
    p.add_argument("--resample", default="3H", help="Time resampling frequency")
    p.add_argument(
        "--min-wstd",
        type=float,
        default=0.001,
        help="In sample mode, keep only Wstd > min_wstd. Use 0 for no threshold.",
    )
    p.add_argument(
        "--mode",
        choices=["samples", "gridded"],
        default="samples",
        help="samples = compact PySR table; gridded = full time,lev,lat,lon fields",
    )
    p.add_argument(
        "--samples-per-day-level",
        type=int,
        default=4000,
        help=(
            "In sample mode, number of random lat/lon samples for each time and level. "
            "Use <=0 to keep all valid gridpoints."
        ),
    )
    p.add_argument(
        "--chunks",
        default=None,
        help="Optional xarray chunking preset. Use 'time' for time=1 chunks.",
    )
    p.add_argument("--tmpdir", default=None, help="Temporary daily NetCDF directory")
    p.add_argument("--compress", action="store_true", help="Use zlib compression in final NetCDF")
    p.add_argument("--keep-tmp", action="store_true", help="Keep temporary daily files")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output/temp if present")
    p.add_argument(
        "--no-load-day",
        action="store_true",
        help="Do not call .load() on each daily merged dataset before sampling",
    )
    return p.parse_args()


def require_path(path: Optional[str], label: str) -> Path:
    if not path:
        raise SystemExit(f"ERROR: {label} is not set")
    return Path(path).expanduser()


def all_dates(year0: int, year1: int) -> List[date]:
    start = date(year0, 1, 1)
    end = date(year1, 12, 31)
    out: List[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def unique_existing(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        sp = str(p)
        if sp not in seen and p.exists():
            out.append(p)
            seen.add(sp)
    return out


def candidate_merra_dirs(root: Path, d: date) -> List[Path]:
    """Support /MERRA2_all/Y2020/M10 and several common variants."""
    y = f"{d.year:04d}"
    m = f"M{d.month:02d}"
    m2 = f"{d.month:02d}"
    dd = f"D{d.day:02d}"

    bases = [root, root / f"Y{y}", root / y, Path(str(root) + y)]
    dirs: List[Path] = []
    for b in bases:
        dirs.extend([b / m, b / m / dd, b / m2, b / m2 / dd])

    seen = set()
    out: List[Path] = []
    for dr in dirs:
        s = str(dr)
        if s not in seen:
            out.append(dr)
            seen.add(s)
    return out


def candidate_sw_dirs(root: Path, d: date) -> List[Path]:
    """Support /Wnet/MERRA2/2019/01 and common Y/M variants."""
    y = f"{d.year:04d}"
    m2 = f"{d.month:02d}"
    mM = f"M{d.month:02d}"
    dd = f"D{d.day:02d}"
    dirs = [
        root / y / m2,
        root / y / mM,
        root / y / m2 / dd,
        root / f"Y{y}" / m2,
        root / f"Y{y}" / mM,
        root / f"Y{y}" / mM / dd,
        Path(str(root) + y) / m2,
        Path(str(root) + y) / mM,
    ]
    seen = set()
    out: List[Path] = []
    for dr in dirs:
        s = str(dr)
        if s not in seen:
            out.append(dr)
            seen.add(s)
    return out


def find_merra_file(root: Path, collection: str, d: date) -> Optional[Path]:
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    patterns = [
        f"MERRA2.{collection}.{ymd}.nc4",
        f"MERRA2_*.{collection}.{ymd}.nc4",
        f"*{collection}*{ymd}*.nc4",
    ]
    candidates: List[Path] = []
    for dr in candidate_merra_dirs(root, d):
        for pat in patterns:
            candidates.extend(sorted(dr.glob(pat)))
    found = unique_existing(candidates)
    return found[0] if found else None


def find_wstd_file(root: Path, d: date) -> Optional[Path]:
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    patterns = [
        f"MERRA2.inst_3d_Wstd_Np.{ymd}.nc4",
        f"*Wstd*{ymd}*.nc4",
        f"*SigmaW*{ymd}*.nc4",
    ]
    candidates: List[Path] = []
    for dr in candidate_sw_dirs(root, d):
        for pat in patterns:
            candidates.extend(sorted(dr.glob(pat)))
    found = unique_existing(candidates)
    return found[0] if found else None


def resolve_day_files(merra_root: Path, sw_root: Path, d: date) -> Optional[Tuple[Path, Path, Path, Path]]:
    f3d = find_merra_file(merra_root, COLL_3D, d)
    fflx = find_merra_file(merra_root, COLL_2D_FLX, d)
    fasm = find_merra_file(merra_root, COLL_2D_ASM, d)
    fout = find_wstd_file(sw_root, d)
    if None in (f3d, fflx, fasm, fout):
        return None
    return f3d, fflx, fasm, fout


def open_keep(path: Path, variables: Sequence[str], chunks=None) -> xr.Dataset:
    ds = xr.open_dataset(path, chunks=chunks)
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        ds.close()
        raise KeyError(f"{path} is missing variables: {missing}")
    return ds[list(variables)]


def resample_if_needed(ds: xr.Dataset, freq: str) -> xr.Dataset:
    if "time" in ds.dims and freq:
        return ds.resample(time=freq).mean(keep_attrs=True)
    return ds


def open_day_dataset(
    files: Tuple[Path, Path, Path, Path],
    pmin: float,
    resample_freq: str,
    chunks=None,
) -> xr.Dataset:
    f3d, fflx, fasm, fout = files

    # Main 3-D pressure-level MERRA file.  H and PS are intentionally read here.
    ds3 = open_keep(f3d, VARS_FROM_3D_FILE, chunks=chunks)
    ds3 = resample_if_needed(ds3, resample_freq)
    if pmin > 0.0 and "lev" in ds3.coords:
        ds3 = ds3.sel(lev=ds3["lev"] > pmin)

    if ds3.sizes.get("time", 0) == 0:
        raise ValueError("No time coordinate in 3-D input file")
    if ds3.sizes.get("lev", 0) == 0:
        raise ValueError("No pressure levels left after pmin filtering")

    template4 = xr.zeros_like(ds3["T"], dtype="float32")

    # Pressure feature from coordinate, in hPa. AIRD uses Pa internally.
    ds3["P"] = (template4 + ds3["lev"]).astype("float32")
    ds3["AIRD"] = ((ds3["P"] * 100.0) / (RD * ds3["T"])).astype("float32")
    ds3["W"] = (-ds3["OMEGA"] / (GRAV * ds3["AIRD"])).astype("float32")

    # Broadcast PS from time,lat,lon to time,lev,lat,lon for sample-table output.
    ps_from_3d_file = ds3["PS"]
    ds3["PS"] = (template4 + ps_from_3d_file).astype("float32")

    ds3 = ds3[FEATURES_3D_OUT]

    # 2-D variables still used by the original NN template.
    dsflx = open_keep(fflx, ["PBLH"], chunks=chunks)
    dsflx = resample_if_needed(dsflx, resample_freq)

    # TQV and T2M come from inst1_2d_asm_Nx.  PS does NOT come from this file anymore.
    dsasm = open_keep(fasm, ["TQV", "T2M"], chunks=chunks)
    dsasm = resample_if_needed(dsasm, resample_freq)
    ds2 = xr.merge([dsflx, dsasm], compat="override")

    # Align before computing AIRD_surf so PS and T2M are on common time/lat/lon.
    # Use the original 2-D PS from inst3_3d_asm_Np, not the broadcasted one.
    ps2, ds2 = xr.align(ps_from_3d_file, ds2, join="inner")
    ds2["AIRD_surf"] = (ps2 / (RD * ds2["T2M"])).astype("float32")
    ds2 = ds2[SURF_VARS_OUT]

    dsout = open_keep(fout, [TARGET], chunks=chunks)
    dsout = resample_if_needed(dsout, resample_freq)
    if "lev" in dsout.coords:
        dsout = dsout.interp(lev=ds3["lev"])

    # Align common coordinates across inputs and target.
    ds3, ds2, dsout = xr.align(ds3, ds2, dsout, join="inner")

    if ds3.sizes.get("time", 0) == 0:
        raise ValueError("No common time coordinate after alignment")
    if ds3.sizes.get("lev", 0) == 0:
        raise ValueError("No pressure levels left after Wstd interpolation/alignment")

    template4 = xr.zeros_like(ds3["T"], dtype="float32")
    for v in SURF_VARS_OUT:
        ds3[v] = (template4 + ds2[v]).astype("float32")

    merged = xr.merge([ds3[ALL_FEATURES], dsout[[TARGET]]], compat="override")
    merged = merged.transpose("time", "lev", "lat", "lon")

    for v in ALL_FEATURES + [TARGET]:
        merged[v] = merged[v].astype("float32")

    merged.attrs["source_3d"] = str(f3d)
    merged.attrs["source_2d_flx"] = str(fflx)
    merged.attrs["source_2d_asm"] = str(fasm)
    merged.attrs["source_wstd"] = str(fout)
    merged.attrs["PS_source"] = "inst3_3d_asm_Np"
    merged.attrs["H_source"] = "inst3_3d_asm_Np"
    merged.attrs["P_source"] = "lev coordinate from inst3_3d_asm_Np, units hPa"
    return merged


def yyyymmdd_from_npdatetime(t: np.datetime64) -> int:
    return int(np.datetime_as_string(t, unit="D").replace("-", ""))


def sample_day(
    ds: xr.Dataset,
    samples_per_day_level: int,
    rng: np.random.Generator,
    min_wstd: float,
) -> Optional[xr.Dataset]:
    """Return compact Dataset with dimension sample."""
    data_parts = {v: [] for v in ALL_FEATURES + [TARGET]}
    time_parts = []
    lev_parts = []
    lat_parts = []
    lon_parts = []
    date_parts = []

    lats = ds["lat"].values.astype("float32")
    lons = ds["lon"].values.astype("float32")
    times = ds["time"].values
    levs = ds["lev"].values.astype("float32")

    for it, tval in enumerate(times):
        day_int = yyyymmdd_from_npdatetime(np.datetime64(tval))
        for iz, lev in enumerate(levs):
            w2 = ds[TARGET].isel(time=it, lev=iz).transpose("lat", "lon").values
            valid = np.isfinite(w2)
            if min_wstd > 0.0:
                valid &= w2 > min_wstd

            valid_flat = np.flatnonzero(valid.ravel())
            if valid_flat.size == 0:
                continue

            if samples_per_day_level <= 0 or samples_per_day_level >= valid_flat.size:
                chosen = valid_flat
            else:
                chosen = rng.choice(valid_flat, size=samples_per_day_level, replace=False)

            iy, ix = np.unravel_index(chosen, w2.shape)
            n = chosen.size

            for v in ALL_FEATURES + [TARGET]:
                a2 = ds[v].isel(time=it, lev=iz).transpose("lat", "lon").values
                data_parts[v].append(np.asarray(a2[iy, ix], dtype="float32"))

            time_parts.append(np.full(n, np.datetime64(tval, "ns"), dtype="datetime64[ns]"))
            lev_parts.append(np.full(n, lev, dtype="float32"))
            lat_parts.append(lats[iy])
            lon_parts.append(lons[ix])
            date_parts.append(np.full(n, day_int, dtype="int32"))

    if not time_parts:
        return None

    data_vars = {
        v: ("sample", np.concatenate(data_parts[v]).astype("float32"))
        for v in ALL_FEATURES + [TARGET]
    }
    coords = {
        "time": ("sample", np.concatenate(time_parts)),
        "lev": ("sample", np.concatenate(lev_parts).astype("float32")),
        "lat": ("sample", np.concatenate(lat_parts).astype("float32")),
        "lon": ("sample", np.concatenate(lon_parts).astype("float32")),
        "yyyymmdd": ("sample", np.concatenate(date_parts).astype("int32")),
    }
    out = xr.Dataset(data_vars=data_vars, coords=coords)
    out.attrs["feature_variables"] = ",".join(ALL_FEATURES)
    out.attrs["target_variable"] = TARGET
    out.attrs["format"] = "sample_table"
    out.attrs["min_wstd_filter"] = float(min_wstd)
    out.attrs["samples_per_day_level"] = int(samples_per_day_level)
    out.attrs["PS_source"] = "inst3_3d_asm_Np"
    out.attrs["H_source"] = "inst3_3d_asm_Np"
    out.attrs["P_source"] = "lev coordinate from inst3_3d_asm_Np, units hPa"
    return out


def encoding_for(ds: xr.Dataset, compress: bool) -> dict:
    enc = {}
    for v in ds.data_vars:
        enc[v] = {"dtype": "float32", "_FillValue": -9999.0}
        if compress:
            enc[v].update({"zlib": True, "complevel": 4, "shuffle": True})
    for c in ds.coords:
        if c == "time":
            continue
        if c == "yyyymmdd":
            enc[c] = {"dtype": "int32"}
        elif ds[c].dtype.kind == "f":
            enc[c] = {"dtype": "float32"}
    return enc


def prepare_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    out = Path(args.out).expanduser()
    tmp = Path(args.tmpdir).expanduser() if args.tmpdir else Path(str(out) + ".tmp_extract")

    if out.exists():
        if args.overwrite:
            out.unlink()
        else:
            raise SystemExit(f"ERROR: output exists: {out}. Use --overwrite.")

    if tmp.exists():
        if args.overwrite:
            shutil.rmtree(tmp)
        else:
            raise SystemExit(f"ERROR: temporary directory exists: {tmp}. Use --overwrite or --tmpdir.")
    tmp.mkdir(parents=True, exist_ok=True)
    return out, tmp


def main() -> int:
    args = parse_args()
    merra_root = require_path(args.merra_path, "MERRAPATH")
    sw_root = require_path(args.sw_path, "SWPATH")
    out_path, tmpdir = prepare_output_paths(args)

    chunks = {"time": 1} if args.chunks == "time" else None

    rng = np.random.default_rng(args.seed)
    dates = all_dates(args.year0, args.year1)
    rng.shuffle(dates)

    tmp_files: List[Path] = []
    selected_days: List[str] = []
    skipped: List[str] = []

    print("============================================================")
    print("Extracting MERRA-2/Wstd data for PySR")
    print("============================================================")
    print(f"MERRAPATH: {merra_root}")
    print(f"SWPATH   : {sw_root}")
    print(f"mode     : {args.mode}")
    print(f"target   : {args.ndays} valid random days from {args.year0}-{args.year1}")
    print(f"features : {', '.join(ALL_FEATURES)}")
    print(f"target   : {TARGET}")
    print(f"out      : {out_path}")
    print(f"tmpdir   : {tmpdir}")
    print("============================================================")

    for d in dates:
        if len(selected_days) >= args.ndays:
            break

        ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
        files = resolve_day_files(merra_root, sw_root, d)
        if files is None:
            skipped.append(f"{ymd}: missing one or more files")
            continue

        try:
            print(f"[{len(selected_days) + 1:03d}/{args.ndays:03d}] reading {ymd}")
            ds = open_day_dataset(files, pmin=args.pmin, resample_freq=args.resample, chunks=chunks)

            if args.mode == "samples":
                if not args.no_load_day:
                    ds = ds.load()
                dso = sample_day(
                    ds,
                    samples_per_day_level=args.samples_per_day_level,
                    rng=rng,
                    min_wstd=args.min_wstd,
                )
                if dso is None or dso.sizes.get("sample", 0) == 0:
                    skipped.append(f"{ymd}: no valid samples")
                    ds.close()
                    continue
                print(f"       samples: {dso.sizes['sample']}")
            else:
                dso = ds
                dso.attrs["format"] = "gridded_time_lev_lat_lon"
                dso.attrs["min_wstd_filter"] = "not applied in gridded mode"

            dso.attrs["source_day"] = ymd
            dso.attrs["source_files"] = " | ".join(str(x) for x in files)
            tmp_file = tmpdir / f"extract_{ymd}.nc"
            dso.to_netcdf(tmp_file, encoding=encoding_for(dso, compress=args.compress))
            tmp_files.append(tmp_file)
            selected_days.append(ymd)
            ds.close()
            dso.close()
        except Exception as exc:
            skipped.append(f"{ymd}: {type(exc).__name__}: {exc}")
            print(f"       SKIP {ymd}: {type(exc).__name__}: {exc}")
            continue

    if len(selected_days) < args.ndays:
        raise SystemExit(
            f"ERROR: only found {len(selected_days)} valid days, requested {args.ndays}. "
            f"First skipped entries: {skipped[:10]}"
        )

    print("============================================================")
    print("Combining temporary files")
    print("============================================================")

    if args.mode == "samples":
        combined = xr.open_mfdataset(
            tmp_files,
            combine="nested",
            concat_dim="sample",
            chunks={"sample": 250_000},
            parallel=False,
        )
    else:
        combined = xr.open_mfdataset(
            tmp_files,
            combine="nested",
            concat_dim="time",
            chunks={"time": 8},
            parallel=False,
        )

    combined.attrs["created_by"] = "extract_pysr_merra2_wstd_hps.py"
    combined.attrs["selected_days"] = ",".join(selected_days)
    combined.attrs["MERRAPATH"] = str(merra_root)
    combined.attrs["SWPATH"] = str(sw_root)
    combined.attrs["features"] = ",".join(ALL_FEATURES)
    combined.attrs["target"] = TARGET
    combined.attrs["year_range"] = f"{args.year0}-{args.year1} inclusive"
    combined.attrs["random_seed"] = int(args.seed)
    combined.attrs["notes"] = "H, PS, and P are from inst3_3d_asm_Np / its lev coordinate."

    combined.to_netcdf(out_path, encoding=encoding_for(combined, compress=args.compress))
    combined.close()

    days_file = out_path.with_suffix(out_path.suffix + ".days.txt")
    days_file.write_text("\n".join(selected_days) + "\n")

    skipped_file = out_path.with_suffix(out_path.suffix + ".skipped.txt")
    skipped_file.write_text("\n".join(skipped) + ("\n" if skipped else ""))

    if not args.keep_tmp:
        shutil.rmtree(tmpdir)

    print("============================================================")
    print("Done")
    print("============================================================")
    print(f"Output file : {out_path}")
    print(f"Days file   : {days_file}")
    print(f"Skipped log : {skipped_file}")
    print(f"Valid days  : {len(selected_days)}")
    if args.mode == "samples":
        print("NetCDF dimension: sample")
    else:
        print("NetCDF dimensions: time, lev, lat, lon")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

