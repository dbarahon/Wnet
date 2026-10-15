#!/usr/bin/env python3
"""
evaluate_merra2_wstd_expression_v10_zero.py

Evaluate a trained PySR symbolic Wstd expression directly on gridded MERRA-2
inputs and the Wstd database.  This v10-zero drop-in changes the map
difference panels from percent difference to mean bias in m s-1, adds
standardized one-at-a-time sensitivity plots, can emit a Fortran 90
module for the selected expression, and can ablate selected predictors by
forcing them to zero in standardized expression space.

This drop-in version is compatible with the v9 stratified training script.  In
addition to the older derived predictors, it supports these new continuous
regime predictors:

    LOGP       = log(P_hPa)
    SIGMA      = P_hPa / PS_hPa
    ZPBL       = H / (PBLH + 1)
    ABOVE_PBL  = H - PBLH
    LOG_TQV    = log1p(TQV)
    LOG_QV     = log(QV + 1e-12)
    LOG_QI     = log(QI + 1e-12)
    LOG_QL     = log(QL + 1e-12)

The script keeps one global expression and evaluates it at all available MERRA2
levels for metrics/zonal means, while plotting selected map levels.

Typical use:

  python evaluate_merra2_wstd_expression_v9.py \
      --year 2017 \
      --ndays 5 \
      --run-dir symbolic_Wstd_MERRA2_long_pysr_merra2_wstd_150d_2010_2015 \
      --outdir eval_MERRA2_2017_v9 \
      --overwrite

With v9 training output, --equation-index best first uses the
selected_equation_validation.json reranked equation when available.  To force a
classic hall-of-fame selection, pass --equation-index min_loss, score, last, or a
row number together with the desired --equations-csv if needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = 287.0
GRAV = 9.81

DEFAULT_MERRAPATH = "/css/merra2/MERRA2_all"
DEFAULT_SWPATH = "/css/gmao/Wstd-MERRA2/pub/Wnet/MERRA2"

COLL_3D = "inst3_3d_asm_Np"
COLL_2D_FLX = "tavg1_2d_flx_Nx"
COLL_2D_ASM = "inst1_2d_asm_Nx"
TARGET = "Wstd"

BASE_FEATURES = [
    "T", "P", "H", "AIRD", "U", "V", "W", "QV", "QI", "QL", "PS",
    "PBLH", "TQV", "AIRD_surf",
]
DERIVED_FEATURES = [
    "WIND", "WABS", "RH", "QVTQV", "RHO_DIFF", "PBLH_MINUS_H",
    "LOGP", "SIGMA", "ZPBL", "ABOVE_PBL",
    "LOG_TQV", "LOG_QV", "LOG_QI", "LOG_QL",
]
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES

SYMBOLIC_NAME = {name: name for name in ALL_FEATURES}
SYMBOLIC_NAME["PBLH_MINUS_H"] = "PBLH_minus_H"
SYMBOLIC_TO_INTERNAL = {v: k for k, v in SYMBOLIC_NAME.items()}

# Common aliases from older ERA5 expressions and possible hand-edited names.
SYMBOLIC_TO_INTERNAL.update({
    "Q": "QV",
    "BLH": "PBLH",
    "TCWV": "TQV",
    "QTCWV": "QVTQV",
    "CIWC": "QI",
    "CLWC": "QL",
    "PBLH_H": "PBLH_MINUS_H",
    "PBLHmH": "PBLH_MINUS_H",
    "PBLHminusH": "PBLH_MINUS_H",
    "PBLH_MINUS_HEIGHT": "PBLH_MINUS_H",
    "LOG_TCWV": "LOG_TQV",
    "LOGTCWV": "LOG_TQV",
    "LOG_Q": "LOG_QV",
})

FEATURE_DATA_ALIASES = dict(SYMBOLIC_TO_INTERNAL)


def data_feature_name(name: str) -> str:
    return FEATURE_DATA_ALIASES.get(str(name), str(name))


def manifest_get(manifest: Dict[str, object], key: str, default=None):
    if isinstance(manifest, dict) and key in manifest:
        return manifest.get(key, default)
    res = manifest.get("result", {}) if isinstance(manifest, dict) else {}
    if isinstance(res, dict) and key in res:
        return res.get(key, default)
    return default


VARS_FROM_3D_FILE = ["T", "H", "U", "V", "OMEGA", "QV", "QI", "QL", "PS"]
FEATURES_3D_OUT = ["T", "P", "H", "AIRD", "U", "V", "W", "QV", "QI", "QL", "PS"]
SURF_VARS_OUT = ["PBLH", "TQV", "AIRD_surf"]


@dataclass
class DayFiles:
    yyyymmdd: str
    f3d: Path
    fflx: Path
    fasm: Path
    fwstd: Path


class OnlineMetrics:
    def __init__(self) -> None:
        self.n = 0
        self.sum_y = 0.0
        self.sum_p = 0.0
        self.sum_y2 = 0.0
        self.sum_p2 = 0.0
        self.sum_yp = 0.0
        self.sum_err = 0.0
        self.sum_abs_err = 0.0
        self.sum_sq_err = 0.0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        yt = np.asarray(y_true, dtype=np.float64).ravel()
        yp = np.asarray(y_pred, dtype=np.float64).ravel()
        m = np.isfinite(yt) & np.isfinite(yp)
        if not np.any(m):
            return
        yt = yt[m]
        yp = yp[m]
        e = yp - yt
        self.n += int(yt.size)
        self.sum_y += float(np.sum(yt))
        self.sum_p += float(np.sum(yp))
        self.sum_y2 += float(np.sum(yt * yt))
        self.sum_p2 += float(np.sum(yp * yp))
        self.sum_yp += float(np.sum(yt * yp))
        self.sum_err += float(np.sum(e))
        self.sum_abs_err += float(np.sum(np.abs(e)))
        self.sum_sq_err += float(np.sum(e * e))

    def finalize(self) -> Dict[str, float]:
        if self.n == 0:
            return {"n": 0, "rmse": math.nan, "mae": math.nan, "bias": math.nan, "corr": math.nan, "r2": math.nan}
        n = float(self.n)
        rmse = math.sqrt(self.sum_sq_err / n)
        mae = self.sum_abs_err / n
        bias = self.sum_err / n
        var_y = self.sum_y2 - self.sum_y * self.sum_y / n
        var_p = self.sum_p2 - self.sum_p * self.sum_p / n
        cov = self.sum_yp - self.sum_y * self.sum_p / n
        corr = cov / math.sqrt(var_y * var_p) if var_y > 0.0 and var_p > 0.0 and self.n > 2 else math.nan
        r2 = 1.0 - self.sum_sq_err / var_y if var_y > 0.0 else math.nan
        return {"n": int(self.n), "rmse": rmse, "mae": mae, "bias": bias, "corr": corr, "r2": r2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained symbolic Wstd expression on gridded MERRA-2/Wstd data.")
    p.add_argument("--year", type=int, default=None, help="Evaluation year; required unless --fortran-only is used")
    p.add_argument("--ndays", type=int, default=5, help="Number of random valid days to evaluate")
    p.add_argument("--seed", type=int, default=5678, help="Random seed for day selection")
    p.add_argument("--levels-hpa", default="250,500,900", help="Comma-separated levels for maps/metrics")

    p.add_argument("--merra-path", default=os.environ.get("MERRAPATH", DEFAULT_MERRAPATH))
    p.add_argument("--sw-path", default=os.environ.get("SWPATH", DEFAULT_SWPATH))
    p.add_argument("--pmin", type=float, default=10.0)
    p.add_argument("--resample", default="3H", help="Time resampling frequency used for MERRA2 and Wstd files")
    p.add_argument("--chunks", default=None, help="Use 'time' for time=1 chunks, otherwise no explicit chunks")

    p.add_argument("--run-dir", default=None, help="Training run directory containing manifest/equations files")
    p.add_argument("--manifest", default=None, help="Training manifest JSON; defaults to --run-dir manifest files")
    p.add_argument("--equations-csv", default=None, help="PySR equations CSV; defaults to --run-dir/equations_global*.csv")
    p.add_argument("--expression", default=None, help="Expression string. May be RHS only, or include an LHS such as 'sqrt(Wstd) = ...'.")
    p.add_argument("--expression-file", default=None, help="Text file containing a single expression; overrides --equations-csv")
    p.add_argument("--expression-from-manifest", action="store_true", help="Use manifest/result.expression explicitly")
    p.add_argument(
        "--equation-index",
        default="best",
        help=(
            "Equation row/source: best, validation, min_loss/loss, score, last, or integer row. "
            "For v9 run dirs, best/validation uses selected_equation_validation.json when available."
        ),
    )
    p.add_argument("--target-transform", choices=["auto", "none", "log", "log1p", "sqrt", "power", "asinh", "log1p_scaled"], default="auto")
    p.add_argument("--target-scale", type=float, default=0.05, help="Scale y0 [m/s] for asinh/log1p_scaled; manifest value is used by default when available")
    p.add_argument("--target-power", type=float, default=0.50, help="Power alpha for power transform; manifest value is used by default when available")
    p.add_argument("--target-bins", default=None, help="Backward-compatible alias for --wstd-bracket-edges")
    p.add_argument("--wstd-bracket-edges", default="auto", help="Ascending true-Wstd bin edges for metrics; auto reads manifest or uses 0.1,1,10")
    p.add_argument("--pressure-bracket-edges", default="auto", help="Descending pressure-bin edges in hPa; auto reads manifest or uses 1000,800,500,200,10")
    p.add_argument("--ps-tolerance-hpa", type=float, default=0.25)
    p.add_argument("--allow-p-greater-than-ps", action="store_true")
    p.add_argument("--feature-space", choices=["auto", "standardized", "physical"], default="auto")
    p.add_argument("--clip-pred-min", type=float, default=0.0, help="Clip physical predicted Wstd below this value; use nan to disable")
    p.add_argument("--pct-denom-min", type=float, default=1.0e-4)
    p.add_argument("--pct-clip", type=float, default=200.0, help="Deprecated for plots; retained for old command compatibility")
    p.add_argument("--bias-clip", type=float, default=None, help="Symmetric absolute color limit for bias panels [m/s]. Default uses robust data limits.")
    p.add_argument("--sensitivity-png", default=None, help="Output PNG for one-at-a-time normalized-space sensitivity curves. Default is inside --outdir.")
    p.add_argument("--sensitivity-range", default="-3,3", help="Standardized feature range for sensitivity curves, e.g. '-3,3'.")
    p.add_argument("--sensitivity-points", type=int, default=241, help="Number of points per sensitivity curve")
    p.add_argument("--no-sensitivity-plot", action="store_true", help="Skip the standardized one-at-a-time sensitivity plot")
    p.add_argument(
        "--zero-standardized",
        default="",
        help=(
            "Comma/space-separated expression variables to force to 0 after standardization. "
            "Names may be symbolic names from the expression or internal/data names, e.g. "
            "'QI,QL,LOG_QI' or 'PBLH_minus_H'."
        ),
    )
    p.add_argument(
        "--zero-standardized-ignore-missing",
        action="store_true",
        help="Warn instead of stopping if --zero-standardized names are not used by the selected expression.",
    )
    p.add_argument("--fortran-out", default=None, help="Write a Fortran 90 module implementing the selected expression")
    p.add_argument("--fortran-module", default="sigmaw_symbolic_mod", help="Fortran module name used by --fortran-out")
    p.add_argument("--fortran-subroutine", default="compute_sigmaw_symbolic", help="Fortran subroutine name used by --fortran-out")
    p.add_argument("--fortran-real-kind", default="real64", choices=["real64", "real32"], help="Fortran real kind for generated module")
    p.add_argument("--fortran-only", action="store_true", help="Only resolve the expression and write --fortran-out / sensitivity plot; do not read MERRA2 data")

    p.add_argument("--outdir", default="eval_wstd_expression_merra2")
    p.add_argument("--png", default=None, help="Output PNG path. Default is inside --outdir")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def require_path(path: Optional[str], label: str) -> Path:
    if not path:
        raise SystemExit(f"ERROR: {label} is not set")
    return Path(path).expanduser()


def all_dates(year: int) -> List[date]:
    d = date(year, 1, 1)
    end = date(year, 12, 31)
    out: List[date] = []
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
    y = f"{d.year:04d}"
    m2 = f"{d.month:02d}"
    mM = f"M{d.month:02d}"
    dd = f"D{d.day:02d}"
    dirs = [
        root / y / m2, root / y / mM, root / y / m2 / dd,
        root / f"Y{y}" / m2, root / f"Y{y}" / mM, root / f"Y{y}" / mM / dd,
        Path(str(root) + y) / m2, Path(str(root) + y) / mM,
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
    patterns = [f"MERRA2.{collection}.{ymd}.nc4", f"MERRA2_*.{collection}.{ymd}.nc4", f"*{collection}*{ymd}*.nc4"]
    candidates: List[Path] = []
    for dr in candidate_merra_dirs(root, d):
        for pat in patterns:
            candidates.extend(sorted(dr.glob(pat)))
    found = unique_existing(candidates)
    return found[0] if found else None


def find_wstd_file(root: Path, d: date) -> Optional[Path]:
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    patterns = [f"MERRA2.inst_3d_Wstd_Np.{ymd}.nc4", f"*Wstd*{ymd}*.nc4", f"*SigmaW*{ymd}*.nc4"]
    candidates: List[Path] = []
    for dr in candidate_sw_dirs(root, d):
        for pat in patterns:
            candidates.extend(sorted(dr.glob(pat)))
    found = unique_existing(candidates)
    return found[0] if found else None


def resolve_day_files(merra_root: Path, sw_root: Path, d: date) -> Optional[DayFiles]:
    f3d = find_merra_file(merra_root, COLL_3D, d)
    fflx = find_merra_file(merra_root, COLL_2D_FLX, d)
    fasm = find_merra_file(merra_root, COLL_2D_ASM, d)
    fout = find_wstd_file(sw_root, d)
    if None in (f3d, fflx, fasm, fout):
        return None
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    return DayFiles(ymd, f3d, fflx, fasm, fout)  # type: ignore[arg-type]


def choose_random_valid_days(merra_root: Path, sw_root: Path, year: int, ndays: int, rng: np.random.Generator) -> List[DayFiles]:
    dates = all_dates(year)
    rng.shuffle(dates)
    out: List[DayFiles] = []
    skipped = []
    for d in dates:
        files = resolve_day_files(merra_root, sw_root, d)
        if files is None:
            skipped.append(f"{d:%Y%m%d}: missing files")
            continue
        out.append(files)
        if len(out) >= ndays:
            break
    if len(out) < ndays:
        raise SystemExit(f"ERROR: found only {len(out)} valid days for {year}; requested {ndays}. First skipped: {skipped[:10]}")
    return sorted(out, key=lambda x: x.yyyymmdd)


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


def open_day_dataset(files: DayFiles, pmin: float, resample_freq: str, chunks=None) -> xr.Dataset:
    ds3 = open_keep(files.f3d, VARS_FROM_3D_FILE, chunks=chunks)
    ds3 = resample_if_needed(ds3, resample_freq)
    if pmin > 0.0 and "lev" in ds3.coords:
        ds3 = ds3.sel(lev=ds3["lev"] > pmin)
    if ds3.sizes.get("time", 0) == 0:
        raise ValueError("No time coordinate in 3-D input file")
    if ds3.sizes.get("lev", 0) == 0:
        raise ValueError("No pressure levels left after pmin filtering")

    template4 = xr.zeros_like(ds3["T"], dtype="float32")
    ds3["P"] = (template4 + ds3["lev"]).astype("float32")
    ds3["AIRD"] = ((ds3["P"] * 100.0) / (RD * ds3["T"])).astype("float32")
    ds3["W"] = (-ds3["OMEGA"] / (GRAV * ds3["AIRD"])).astype("float32")
    ps_from_3d_file = ds3["PS"]
    ds3["PS"] = (template4 + ps_from_3d_file).astype("float32")
    ds3 = ds3[FEATURES_3D_OUT]

    dsflx = open_keep(files.fflx, ["PBLH"], chunks=chunks)
    dsflx = resample_if_needed(dsflx, resample_freq)
    dsasm = open_keep(files.fasm, ["TQV", "T2M"], chunks=chunks)
    dsasm = resample_if_needed(dsasm, resample_freq)
    ds2 = xr.merge([dsflx, dsasm], compat="override")

    ps2, ds2 = xr.align(ps_from_3d_file, ds2, join="inner")
    ds2["AIRD_surf"] = (ps2 / (RD * ds2["T2M"])).astype("float32")
    ds2 = ds2[SURF_VARS_OUT]

    dsout = open_keep(files.fwstd, [TARGET], chunks=chunks)
    dsout = resample_if_needed(dsout, resample_freq)
    if "lev" in dsout.coords:
        dsout = dsout.interp(lev=ds3["lev"])

    ds3, ds2, dsout = xr.align(ds3, ds2, dsout, join="inner")
    if ds3.sizes.get("time", 0) == 0:
        raise ValueError("No common time coordinate after alignment")
    if ds3.sizes.get("lev", 0) == 0:
        raise ValueError("No pressure levels left after Wstd interpolation/alignment")

    template4 = xr.zeros_like(ds3["T"], dtype="float32")
    for v in SURF_VARS_OUT:
        ds3[v] = (template4 + ds2[v]).astype("float32")

    merged = xr.merge([ds3[BASE_FEATURES], dsout[[TARGET]]], compat="override")
    merged = merged.transpose("time", "lev", "lat", "lon")
    for v in BASE_FEATURES + [TARGET]:
        merged[v] = merged[v].astype("float32")
    merged.attrs["source_3d"] = str(files.f3d)
    merged.attrs["source_2d_flx"] = str(files.fflx)
    merged.attrs["source_2d_asm"] = str(files.fasm)
    merged.attrs["source_wstd"] = str(files.fwstd)
    return merged


def qsat_liquid_specific_humidity_np(t_k: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    eps = 0.622
    p_pa = p_hpa.astype(np.float64) * 100.0
    t = t_k.astype(np.float64)
    arg = 17.67 * (t - 273.15) / (t - 29.65)
    arg = np.clip(arg, -80.0, 80.0)
    es = 611.2 * np.exp(arg)
    es = np.minimum(es, 0.99 * p_pa)
    qsat = eps * es / (p_pa - (1.0 - eps) * es)
    return qsat.astype(np.float32)


def pressure_to_hpa_np(arr: np.ndarray) -> np.ndarray:
    aa = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(aa) & (np.abs(aa) < 1.0e7) & (aa > -9000.0)
    if not np.any(finite):
        return aa
    scale = 0.01 if float(np.nanmedian(np.abs(aa[finite]))) > 2000.0 else 1.0
    return aa * scale


def p_le_ps_mask(p: np.ndarray, ps: np.ndarray, tolerance_hpa: float = 0.25) -> np.ndarray:
    p_hpa = pressure_to_hpa_np(p)
    ps_hpa = pressure_to_hpa_np(ps)
    return (
        np.isfinite(p_hpa) & np.isfinite(ps_hpa)
        & (p_hpa > 0.0) & (ps_hpa > 0.0)
        & (p_hpa <= ps_hpa + float(tolerance_hpa))
    )


def add_derived_features(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(raw)
    out["WIND"] = np.sqrt(out["U"] * out["U"] + out["V"] * out["V"])
    out["WABS"] = np.abs(out["W"])
    qsat = qsat_liquid_specific_humidity_np(out["T"], pressure_to_hpa_np(out["P"]))
    out["RH"] = np.clip(out["QV"] / np.where(qsat > 1.0e-12, qsat, np.nan), 0.0, 2.0)
    out["QVTQV"] = out["QV"] * out["TQV"]
    out["RHO_DIFF"] = out["AIRD_surf"] - out["AIRD"]
    out["PBLH_MINUS_H"] = out["PBLH"] - out["H"]

    p_hpa = pressure_to_hpa_np(out["P"])
    ps_hpa = pressure_to_hpa_np(out["PS"])
    out["LOGP"] = np.log(np.maximum(p_hpa, 1.0e-6))
    out["SIGMA"] = p_hpa / np.where(ps_hpa > 1.0e-6, ps_hpa, np.nan)
    out["ZPBL"] = out["H"] / np.maximum(out["PBLH"] + 1.0, 1.0)
    out["ABOVE_PBL"] = out["H"] - out["PBLH"]
    out["LOG_TQV"] = np.log1p(np.maximum(out["TQV"], 0.0))
    out["LOG_QV"] = np.log(np.maximum(out["QV"], 0.0) + 1.0e-12)
    out["LOG_QI"] = np.log(np.maximum(out["QI"], 0.0) + 1.0e-12)
    out["LOG_QL"] = np.log(np.maximum(out["QL"], 0.0) + 1.0e-12)
    return out


def read_json(path: Path) -> Dict[str, object]:
    with open(path) as f:
        return json.load(f)


def find_manifest(args: argparse.Namespace) -> Optional[Path]:
    if args.manifest:
        return Path(args.manifest).expanduser()
    if args.run_dir:
        rd = Path(args.run_dir).expanduser()
        for name in ["manifest_and_results.json", "manifest_partial.json"]:
            p = rd / name
            if p.exists():
                return p
    return None


def find_equations_csv(args: argparse.Namespace) -> Optional[Path]:
    if args.equations_csv:
        return Path(args.equations_csv).expanduser()
    if not args.run_dir:
        return None
    rd = Path(args.run_dir).expanduser()
    spec = str(args.equation_index).strip().lower()
    if spec in {"validation", "val", "rerank", "reranked", "validation_rerank"}:
        p = rd / "equation_validation_ranking.csv"
        if p.exists():
            return p
    for name in ["equations_global_latest.csv", "equations_global.csv", "hall_of_fame.csv"]:
        p = rd / name
        if p.exists():
            return p
    pysr = rd / "pysr"
    if pysr.exists():
        hits = sorted(pysr.rglob("hall_of_fame.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if hits:
            return hits[0]
    return None


def strip_lhs(expr: str) -> Tuple[Optional[str], str]:
    line = str(expr).strip().splitlines()[0]
    if "=" in line:
        lhs, rhs = line.split("=", 1)
        return lhs.strip(), rhs.strip()
    return None, line


def select_equation_from_csv(csv_path: Path, equation_index: str) -> Tuple[str, Dict[str, object]]:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if "equation" not in df.columns or len(df) == 0:
        raise RuntimeError(f"No equation column/rows found in {csv_path}; available columns={list(df.columns)}")

    spec = str(equation_index).strip().lower()
    if spec in {"validation", "val", "rerank", "reranked", "validation_rerank"}:
        if "validation_score" in df.columns:
            score = pd.to_numeric(df["validation_score"], errors="coerce").to_numpy(float)
            finite = np.isfinite(score)
            idx = int(np.flatnonzero(finite)[np.argmax(score[finite])]) if np.any(finite) else 0
            selection = "maximum_validation_score"
        else:
            idx = 0
            selection = "first_row_no_validation_score"
    elif spec in {"best", "loss", "min_loss", "minimum_loss"}:
        if "loss" not in df.columns:
            idx = len(df) - 1
            selection = "last_row_no_loss_column"
        else:
            loss = pd.to_numeric(df["loss"], errors="coerce").to_numpy(float)
            finite = np.isfinite(loss)
            idx = int(np.flatnonzero(finite)[np.argmin(loss[finite])]) if np.any(finite) else len(df) - 1
            selection = "minimum_loss"
    elif spec == "score":
        if "score" not in df.columns:
            idx = len(df) - 1
            selection = "last_row_no_score_column"
        else:
            score = pd.to_numeric(df["score"], errors="coerce").to_numpy(float)
            finite = np.isfinite(score)
            idx = int(np.flatnonzero(finite)[np.argmax(score[finite])]) if np.any(finite) else len(df) - 1
            selection = "maximum_score"
    elif spec == "last":
        idx = len(df) - 1
        selection = "last_row"
    else:
        idx = int(equation_index)
        if idx < 0 or idx >= len(df):
            raise IndexError(f"--equation-index out of range: {idx}; table has {len(df)} rows")
        selection = "explicit_row_index"

    info: Dict[str, object] = {
        "csv_path": str(csv_path), "row_index": int(idx), "selection": selection,
        "n_rows": int(len(df)), "columns": list(df.columns),
    }
    for col in ["complexity", "loss", "score", "score_pysr", "validation_score", "corr_global", "corr_high", "corr_extreme"]:
        if col in df.columns:
            try:
                info[col] = float(df.iloc[idx][col])
            except Exception:
                info[col] = str(df.iloc[idx][col])
    return str(df.iloc[idx]["equation"]), info


def expression_from_manifest(manifest: Dict[str, object]) -> Optional[str]:
    val = manifest_get(manifest, "expression", None)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def selected_validation_expression(args: argparse.Namespace) -> Tuple[Optional[str], Dict[str, object]]:
    if not args.run_dir or args.equations_csv:
        return None, {}
    spec = str(args.equation_index).strip().lower()
    if spec not in {"best", "validation", "val", "rerank", "reranked", "validation_rerank"}:
        return None, {}
    p = Path(args.run_dir).expanduser() / "selected_equation_validation.json"
    if not p.exists():
        return None, {}
    try:
        data = read_json(p)
        row = data.get("ranking_row", {}) if isinstance(data, dict) else {}
        rhs = None
        if isinstance(data, dict) and isinstance(data.get("rhs"), str):
            rhs = str(data["rhs"])
        if rhs is None and isinstance(row, dict) and isinstance(row.get("equation"), str):
            rhs = str(row["equation"])
        if rhs:
            info = {
                "selection": "selected_equation_validation_json",
                "selected_equation_validation_json": str(p),
            }
            if isinstance(row, dict):
                info.update({f"validation_{k}": v for k, v in row.items() if k != "equation"})
            return rhs, info
    except Exception as e:
        warnings.warn(f"Could not read {p}: {e}")
    return None, {}


def load_expression(args: argparse.Namespace, manifest: Optional[Dict[str, object]] = None) -> Tuple[Optional[str], str, Dict[str, object]]:
    if getattr(args, "expression", None):
        lhs, rhs = strip_lhs(str(args.expression).strip())
        return lhs, rhs, {"selection": "cli_expression"}

    if args.expression_file:
        p = Path(args.expression_file).expanduser()
        text = p.read_text().strip().splitlines()[0]
        lhs, rhs = strip_lhs(text)
        return lhs, rhs, {"expression_file": str(p), "selection": "expression_file"}

    mexpr = expression_from_manifest(manifest or {})
    if mexpr and getattr(args, "expression_from_manifest", False):
        lhs, rhs = strip_lhs(mexpr)
        return lhs, rhs, {"selection": "manifest_expression"}

    val_expr, val_info = selected_validation_expression(args)
    if val_expr:
        lhs, rhs = strip_lhs(val_expr)
        return lhs, rhs, val_info

    if mexpr and (not args.equations_csv and not args.run_dir):
        lhs, rhs = strip_lhs(mexpr)
        return lhs, rhs, {"selection": "manifest_expression"}

    csv_path = find_equations_csv(args)
    if csv_path is None:
        if mexpr:
            lhs, rhs = strip_lhs(mexpr)
            return lhs, rhs, {"selection": "manifest_expression_fallback_no_csv"}
        raise RuntimeError("No expression source found. Provide --expression, --expression-file, --equations-csv, --run-dir, or a manifest containing result.expression.")
    expr, info = select_equation_from_csv(csv_path, args.equation_index)
    lhs, rhs = strip_lhs(expr)
    return lhs, rhs, info


TARGET_TRANSFORMS = {"none", "log", "log1p", "sqrt", "power", "asinh", "log1p_scaled"}


def cli_option_present(*names: str) -> bool:
    for tok in sys.argv[1:]:
        for name in names:
            if tok == name or tok.startswith(name + "="):
                return True
    return False


def resolve_target_transform(args: argparse.Namespace, manifest: Dict[str, object], lhs: Optional[str] = None) -> str:
    if args.target_transform != "auto":
        return str(args.target_transform).lower()
    for source in (manifest, manifest.get("result", {}) if isinstance(manifest.get("result", {}), dict) else {}):
        if isinstance(source, dict) and isinstance(source.get("target_transform"), str):
            val = str(source["target_transform"]).lower()
            if val in TARGET_TRANSFORMS:
                return val
    if lhs:
        ll = lhs.lower().replace(" ", "")
        if "asinh" in ll:
            return "asinh"
        if "log1p" in ll and "/" in ll:
            return "log1p_scaled"
        if "log1p" in ll:
            return "log1p"
        if "sqrt" in ll or ll.startswith("wstd**0.5") or ll.startswith("wstd^0.5"):
            return "sqrt"
        if "wstd^" in ll or "wstd**" in ll:
            return "power"
        if "log" in ll:
            return "log"
    warnings.warn("Could not infer target transform; assuming physical-space Wstd. Pass --target-transform sqrt if needed.")
    return "none"


def resolve_target_params(args: argparse.Namespace, manifest: Dict[str, object]) -> Tuple[float, float]:
    scale = float(args.target_scale)
    power = float(args.target_power)
    if manifest and not cli_option_present("--target-scale"):
        for source in (manifest, manifest.get("result", {}) if isinstance(manifest.get("result", {}), dict) else {}):
            if isinstance(source, dict) and "target_scale" in source:
                try:
                    scale = float(source["target_scale"]); break
                except Exception:
                    pass
    if manifest and not cli_option_present("--target-power"):
        for source in (manifest, manifest.get("result", {}) if isinstance(manifest.get("result", {}), dict) else {}):
            if isinstance(source, dict) and "target_power" in source:
                try:
                    power = float(source["target_power"]); break
                except Exception:
                    pass
    return scale, power


def inverse_target(y: np.ndarray, mode: str, target_scale: float = 0.05, target_power: float = 0.50) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64)
    if mode == "none":
        return yy
    if mode == "log":
        return np.exp(yy)
    if mode == "log1p":
        return np.expm1(yy)
    if mode == "sqrt":
        return np.square(np.maximum(yy, 0.0))
    if mode == "power":
        alpha = float(target_power)
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("target_power must be > 0")
        return np.power(np.maximum(yy, 0.0), 1.0 / alpha)
    if mode == "asinh":
        scale = float(target_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("target_scale must be > 0")
        return scale * np.sinh(yy)
    if mode == "log1p_scaled":
        scale = float(target_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("target_scale must be > 0")
        return scale * np.expm1(yy)
    raise ValueError(mode)


def inverse_target_description(mode: str, target_scale: float = 0.05, target_power: float = 0.50) -> str:
    if mode == "none": return "Wstd = expression"
    if mode == "log": return "Wstd = exp(expression)"
    if mode == "log1p": return "Wstd = expm1(expression)"
    if mode == "sqrt": return "Wstd = max(expression, 0)^2"
    if mode == "power": return f"Wstd = max(expression, 0)^(1/{float(target_power):.6g})"
    if mode == "asinh": return f"Wstd = {float(target_scale):.6g} * sinh(expression)"
    if mode == "log1p_scaled": return f"Wstd = {float(target_scale):.6g} * expm1(expression)"
    return f"unknown target inverse for {mode}"


def parse_optional_edges(spec: object, *, default: Optional[Sequence[float]], descending: bool, name: str) -> Optional[np.ndarray]:
    disabled = {"", "none", "off", "disable", "disabled", "no", "false"}
    if spec is None or (isinstance(spec, str) and spec.strip().lower() == "auto"):
        vals = None if default is None else [float(x) for x in default]
    elif isinstance(spec, (list, tuple, np.ndarray)):
        vals = [float(x) for x in spec]
    else:
        text = str(spec).strip()
        if text.lower() in disabled:
            return None
        text = text.strip("[]()")
        vals = [float(x) for x in re.split(r"[,\s]+", text) if str(x).strip()]
    if vals is None:
        return None
    if len(vals) < 2:
        raise ValueError(f"{name} needs at least two edges, or use 'none'")
    if descending and any(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError(f"{name} must be strictly descending")
    if (not descending) and any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError(f"{name} must be strictly ascending")
    return np.asarray(vals, dtype=np.float64)


def target_bin_indices(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64)
    b = np.searchsorted(edges, yy, side="right") - 1
    b[(yy < edges[0]) | (yy >= edges[-1])] = -1
    b[yy == edges[-1]] = len(edges) - 2
    return b.astype(np.int32, copy=False)


def resolve_target_bins(args: argparse.Namespace, manifest: Dict[str, object]) -> Optional[np.ndarray]:
    if args.target_bins is not None:
        return parse_optional_edges(args.target_bins, default=None, descending=False, name="target/Wstd bins")
    if str(args.wstd_bracket_edges).strip().lower() != "auto":
        return parse_optional_edges(args.wstd_bracket_edges, default=None, descending=False, name="Wstd bracket edges")
    if manifest:
        for source in (manifest, manifest.get("result", {}) if isinstance(manifest.get("result", {}), dict) else {}, manifest.get("sample_info", {}) if isinstance(manifest.get("sample_info", {}), dict) else {}):
            for key in ("wstd_bracket_edges", "target_bins"):
                if isinstance(source, dict) and key in source:
                    return parse_optional_edges(source.get(key), default=None, descending=False, name=key)
    return parse_optional_edges("auto", default=(0.1, 1.0, 10.0), descending=False, name="Wstd bracket edges")


def resolve_pressure_bins(args: argparse.Namespace, manifest: Dict[str, object]) -> Optional[np.ndarray]:
    if str(args.pressure_bracket_edges).strip().lower() != "auto":
        return parse_optional_edges(args.pressure_bracket_edges, default=None, descending=True, name="pressure bracket edges")
    if manifest:
        for source in (manifest, manifest.get("result", {}) if isinstance(manifest.get("result", {}), dict) else {}):
            for key in ("pressure_bracket_edges_hpa", "pressure_bins_hpa"):
                if isinstance(source, dict) and key in source:
                    return parse_optional_edges(source.get(key), default=None, descending=True, name=key)
    return parse_optional_edges("auto", default=(1000.0, 800.0, 500.0, 200.0, 10.0), descending=True, name="pressure bracket edges")


def pressure_bracket_label(lower: float, upper: float) -> str:
    return f"{lower:.6g}-{upper:.6g}_hPa"


def load_feature_config(manifest: Dict[str, object], rhs_expr: str, args: argparse.Namespace) -> Tuple[List[str], List[str], Dict[str, Dict[str, float]]]:
    internal = manifest_get(manifest, "feature_names_internal", None)
    symbolic = manifest_get(manifest, "feature_names_symbolic", None)
    std = manifest_get(manifest, "feature_standardization", None)

    tokens = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", rhs_expr)))
    exclude = {
        "sin", "cos", "tanh", "atan", "exp", "log", "sqrt", "abs", "Abs", "Max", "Min",
        "square", "cube", "sqrt_abs", "log_abs", "exp_clip", "asinh", "sinh", "E", "pi", "nan", "inf",
    }
    expr_symbolic_names = [t for t in tokens if t not in exclude]

    if isinstance(internal, list) and isinstance(symbolic, list) and len(internal) == len(symbolic):
        manifest_internal_names = [str(x) for x in internal]
        symbolic_names = [str(x) for x in symbolic]
        # Old/new manifests usually carry the complete training feature list.
        # Still add any expression variable not present in that list so a
        # hand-edited expression or validation-ranking CSV cannot leave a free
        # SymPy symbol outside the lambdified argument list.
        for t in expr_symbolic_names:
            if t not in symbolic_names:
                symbolic_names.append(t)
                manifest_internal_names.append(SYMBOLIC_TO_INTERNAL.get(t, t))
    else:
        symbolic_names = expr_symbolic_names
        manifest_internal_names = [SYMBOLIC_TO_INTERNAL.get(t, t) for t in symbolic_names]

    internal_names = [data_feature_name(name) for name in manifest_internal_names]

    std_raw: Dict[str, Dict[str, float]] = {}
    if isinstance(std, dict):
        for k, v in std.items():
            if isinstance(v, dict) and "mean" in v and "std" in v:
                try:
                    std_raw[str(k)] = {"mean": float(v["mean"]), "std": float(v["std"])}
                except Exception:
                    pass

    std_out: Dict[str, Dict[str, float]] = {}
    for mname, sname, dname in zip(manifest_internal_names, symbolic_names, internal_names):
        entry = None
        for key in (mname, sname, dname, data_feature_name(mname), data_feature_name(sname)):
            if key in std_raw:
                entry = std_raw[key]
                break
        if entry is not None:
            std_out[dname] = dict(entry)

    if args.feature_space == "physical":
        std_out = {name: {"mean": 0.0, "std": 1.0} for name in internal_names}
    elif args.feature_space == "standardized" or (args.feature_space == "auto" and std_raw):
        missing = [dname for dname in internal_names if dname not in std_out]
        if missing:
            details = []
            for mname, sname, dname in zip(manifest_internal_names, symbolic_names, internal_names):
                if dname in missing:
                    details.append(f"{sname}: manifest_internal={mname}, data_name={dname}")
            raise RuntimeError(
                "Expression appears to use standardized features, but standardization metadata is missing for: "
                + "; ".join(details)
                + ". Provide the training manifest, or use --feature-space physical only for unstandardized expressions."
            )
    else:
        std_out = {name: {"mean": 0.0, "std": 1.0} for name in internal_names}

    return internal_names, symbolic_names, std_out



def parse_sympy_expression(rhs_expr: str, symbolic_names: Sequence[str]):
    import sympy as sp

    rhs = str(rhs_expr).strip().replace("^", "**")
    syms = {name: sp.Symbol(name) for name in symbolic_names}
    locals_map = dict(syms)
    locals_map.update({
        "abs": sp.Abs, "Abs": sp.Abs, "Max": sp.Max, "Min": sp.Min,
        "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp,
        "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh, "atan": sp.atan,
        "asinh": sp.asinh, "sinh": sp.sinh,
        "sqrt_abs": sp.Function("sqrt_abs"), "log_abs": sp.Function("log_abs"),
        "exp_clip": sp.Function("exp_clip"), "cube": sp.Function("cube"), "square": sp.Function("square"),
    })
    expr = sp.sympify(rhs, locals=locals_map)
    return expr, syms


def build_lambdified_expression(rhs_expr: str, symbolic_names: Sequence[str]):
    import sympy as sp

    expr, syms = parse_sympy_expression(rhs_expr, symbolic_names)

    def sqrt_abs(x): return np.sqrt(np.abs(x))
    def log_abs(x): return np.log(np.abs(x) + 1.0e-6)
    def exp_clip(x): return np.exp(np.clip(x, -20.0, 20.0))
    def cube(x): return x * x * x
    def square(x): return x * x

    modules = [{"sqrt_abs": sqrt_abs, "log_abs": log_abs, "exp_clip": exp_clip, "cube": cube, "square": square}, "numpy"]
    func = sp.lambdify([syms[name] for name in symbolic_names], expr, modules=modules)
    return func, str(expr)

def standardize_feature(arr: np.ndarray, name: str, std_info: Dict[str, Dict[str, float]]) -> np.ndarray:
    info = std_info.get(name, {"mean": 0.0, "std": 1.0})
    std = float(info.get("std", 1.0))
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    return (arr - float(info.get("mean", 0.0))) / std


def parse_levels(value: str) -> List[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def nearest_level_values(ds: xr.Dataset, requested: Sequence[float]) -> List[float]:
    levs = np.asarray(ds["lev"].values, dtype=float)
    out = []
    for r in requested:
        out.append(float(levs[int(np.argmin(np.abs(levs - r)))]))
    return out


def pct_diff(pred: np.ndarray, true: np.ndarray, denom_min: float, clip: float) -> np.ndarray:
    den = np.asarray(true, dtype=float)
    out = np.full_like(den, np.nan, dtype=float)
    m = np.isfinite(pred) & np.isfinite(true) & (np.abs(den) >= denom_min)
    out[m] = 100.0 * (pred[m] - true[m]) / den[m]
    if clip and clip > 0:
        out = np.clip(out, -clip, clip)
    return out


def robust_limits(a: np.ndarray, b: Optional[np.ndarray] = None, qlo: float = 2.0, qhi: float = 98.0) -> Tuple[float, float]:
    vals = [np.asarray(a).ravel()]
    if b is not None:
        vals.append(np.asarray(b).ravel())
    x = np.concatenate(vals)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(x, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(x)); hi = float(np.nanmax(x))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def weighted_nanmean_2d_latlon(field: np.ndarray, lat: np.ndarray) -> float:
    a = np.asarray(field, dtype=np.float64)
    if a.ndim != 2:
        return float("nan")
    wlat = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    wlat = np.where(np.isfinite(wlat) & (wlat > 0.0), wlat, np.nan)
    w2 = wlat[:, None] * np.ones((1, a.shape[1]), dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(w2)
    if not np.any(m):
        return float("nan")
    return float(np.nansum(a[m] * w2[m]) / np.nansum(w2[m]))


def weighted_nanmean_latlev(field: np.ndarray, lat: np.ndarray) -> float:
    a = np.asarray(field, dtype=np.float64)
    if a.ndim != 2:
        return float("nan")
    wlat = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    wlat = np.where(np.isfinite(wlat) & (wlat > 0.0), wlat, np.nan)
    w2 = np.ones((a.shape[0], 1), dtype=np.float64) * wlat[None, :]
    m = np.isfinite(a) & np.isfinite(w2)
    if not np.any(m):
        return float("nan")
    return float(np.nansum(a[m] * w2[m]) / np.nansum(w2[m]))


def title_with_mean(label: str, mean_value: float, unit: str) -> str:
    if not np.isfinite(mean_value):
        return f"{label}\nmean = NaN"
    if unit == "%":
        return f"{label}\nmean = {mean_value:.1f}%"
    if unit:
        return f"{label}\nmean = {mean_value:.4g} {unit}"
    return f"{label}\nmean = {mean_value:.4g}"



def bias_diff(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out = np.full_like(pred, np.nan, dtype=float)
    m = np.isfinite(pred) & np.isfinite(true)
    out[m] = pred[m] - true[m]
    return out


def symmetric_abs_limit(field: np.ndarray, clip: Optional[float] = None, q: float = 98.0) -> float:
    if clip is not None and np.isfinite(clip) and clip > 0.0:
        return float(clip)
    x = np.asarray(field, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    val = float(np.nanpercentile(np.abs(x), q))
    if not np.isfinite(val) or val <= 0.0:
        val = float(np.nanmax(np.abs(x))) if x.size else 1.0
    if not np.isfinite(val) or val <= 0.0:
        val = 1.0
    return val


def parse_range_pair(spec: str, *, name: str = "range") -> Tuple[float, float]:
    parts = [p for p in re.split(r"[,\s]+", str(spec).strip()) if p]
    if len(parts) != 2:
        raise ValueError(f"{name} must have exactly two values, e.g. '-3,3'")
    lo, hi = float(parts[0]), float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"{name} must be finite and increasing")
    return lo, hi


def parse_name_list(spec: object) -> List[str]:
    """Parse a comma/space separated name list while preserving case."""
    if spec is None:
        return []
    text = str(spec).strip()
    if not text:
        return []
    return [tok for tok in re.split(r"[,\s]+", text) if tok]


def name_key(name: object) -> str:
    """Case-insensitive key that also ignores small punctuation differences."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def zero_output_suffix(spec: object) -> str:
    names = parse_name_list(spec)
    if not names:
        return ""
    joined = "_".join(safe_filename_component(x, max_len=24) for x in names)
    digest = hashlib.sha1("|".join(names).encode("utf-8")).hexdigest()[:6]
    return f"_zero_{safe_filename_component(joined, max_len=90)}_{digest}"


def resolve_zero_standardized_features(
    spec: object,
    internal_features: Sequence[str],
    symbolic_features: Sequence[str],
    *,
    ignore_missing: bool = False,
) -> Tuple[set, List[Dict[str, object]]]:
    """Resolve --zero-standardized names to expression argument indices.

    Matching accepts symbolic names, internal/data names, and common aliases
    used by older ERA5/MERRA2 expressions.  The returned set contains indices
    in the lambdified expression argument list.
    """
    requested = parse_name_list(spec)
    if not requested:
        return set(), []

    lookup: Dict[str, int] = {}
    for i, (iname, sname) in enumerate(zip(internal_features, symbolic_features)):
        candidates = {iname, sname, data_feature_name(iname), data_feature_name(sname)}
        if sname in SYMBOLIC_TO_INTERNAL:
            candidates.add(SYMBOLIC_TO_INTERNAL[sname])
        if iname in SYMBOLIC_NAME:
            candidates.add(SYMBOLIC_NAME[iname])
        for cand in candidates:
            lookup[name_key(cand)] = i

    zero_indices = set()
    records: List[Dict[str, object]] = []
    missing: List[str] = []
    for req in requested:
        candidates = [req, data_feature_name(req), SYMBOLIC_TO_INTERNAL.get(req, req)]
        idx = None
        for cand in candidates:
            idx = lookup.get(name_key(cand))
            if idx is not None:
                break
        if idx is None:
            missing.append(req)
            continue
        zero_indices.add(int(idx))
        records.append({
            "requested": req,
            "arg_index": int(idx),
            "symbolic": str(symbolic_features[idx]),
            "internal": str(internal_features[idx]),
        })

    if missing:
        msg = (
            "--zero-standardized names not used by selected expression/features: "
            + ", ".join(missing)
            + ". Available symbolic/internal pairs: "
            + ", ".join(f"{s}->{i}" if s != i else str(s) for i, s in zip(internal_features, symbolic_features))
        )
        if ignore_missing:
            warnings.warn(msg)
        else:
            raise SystemExit("ERROR: " + msg)
    return zero_indices, records


def apply_zero_standardized_args(args_for_expr: List[np.ndarray], zero_indices: set) -> List[np.ndarray]:
    if not zero_indices:
        return args_for_expr
    out = list(args_for_expr)
    for idx in zero_indices:
        out[int(idx)] = np.zeros_like(out[int(idx)], dtype=np.float64)
    return out


def sensitivity_curves(
    eval_func,
    internal_features: Sequence[str],
    symbolic_features: Sequence[str],
    target_transform: str,
    target_scale: float,
    target_power: float,
    clip_pred_min: float,
    xmin: float,
    xmax: float,
    npoints: int,
    zero_indices: Optional[set] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    """Evaluate one-at-a-time sensitivities in normalized feature space.

    Each panel varies one symbolic feature over [xmin, xmax] while all other
    symbolic features are held at 0, i.e. at their training means for
    standardized expressions.
    """
    npoints = max(int(npoints), 3)
    x = np.linspace(float(xmin), float(xmax), npoints, dtype=np.float64)
    zero_indices = set() if zero_indices is None else set(int(i) for i in zero_indices)
    base_args = [np.zeros_like(x) for _ in symbolic_features]
    curves: Dict[str, np.ndarray] = {}

    base_fit = np.asarray(eval_func(*base_args), dtype=np.float64)
    if base_fit.shape == ():
        base_fit = np.zeros_like(x) + float(base_fit)
    base_sigma = inverse_target(base_fit, target_transform, target_scale=target_scale, target_power=target_power)
    if np.isfinite(clip_pred_min):
        base_sigma = np.maximum(base_sigma, float(clip_pred_min))
    baseline = float(np.nanmean(base_sigma))

    for i, (iname, sname) in enumerate(zip(internal_features, symbolic_features)):
        args = [a.copy() for a in base_args]
        if i in zero_indices:
            args[i] = np.zeros_like(x)
        else:
            args[i] = x
        yfit = np.asarray(eval_func(*args), dtype=np.float64)
        if yfit.shape == ():
            yfit = np.zeros_like(x) + float(yfit)
        sigma = inverse_target(yfit, target_transform, target_scale=target_scale, target_power=target_power)
        if np.isfinite(clip_pred_min):
            sigma = np.maximum(sigma, float(clip_pred_min))
        label = sname if sname == iname else f"{sname} ({iname})"
        curves[label] = sigma
    return x, curves, baseline


def plot_sensitivity_curves(
    eval_func,
    internal_features: Sequence[str],
    symbolic_features: Sequence[str],
    target_transform: str,
    target_scale: float,
    target_power: float,
    clip_pred_min: float,
    out_png: Path,
    dpi: int,
    x_range: Tuple[float, float],
    npoints: int,
    zero_indices: Optional[set] = None,
) -> None:
    x, curves, baseline = sensitivity_curves(
        eval_func, internal_features, symbolic_features, target_transform,
        target_scale, target_power, clip_pred_min, x_range[0], x_range[1], npoints, zero_indices
    )
    n = len(curves)
    if n == 0:
        warnings.warn("No expression features found; skipping sensitivity plot")
        return
    ncols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (name, y) in zip(axes_arr, curves.items()):
        ax.plot(x, y, linewidth=1.8)
        ax.axvline(0.0, linewidth=0.8, alpha=0.5)
        if np.isfinite(baseline):
            ax.axhline(baseline, linewidth=0.8, alpha=0.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("standardized anomaly")
        ax.set_ylabel(r"$\sigma_W$ [m s$^{-1}$]")
        ax.grid(True, alpha=0.25)
    for ax in axes_arr[n:]:
        ax.axis("off")
    fig.suptitle("One-at-a-time symbolic $\\sigma_W$ sensitivities; all other variables held at normalized mean (0)", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_results(lat: np.ndarray, lon: np.ndarray, levels: Sequence[float], pred_maps: Dict[float, np.ndarray], true_maps: Dict[float, np.ndarray], bias_maps: Dict[float, np.ndarray], zonal_levels: np.ndarray, pred_zonal: np.ndarray, true_zonal: np.ndarray, bias_zonal: np.ndarray, out_png: Path, dpi: int, bias_clip: Optional[float]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), constrained_layout=True)

    for i, lev in enumerate(levels):
        pred = pred_maps[lev]
        true = true_maps[lev]
        diff = bias_maps[lev]
        vmin, vmax = robust_limits(pred, true)
        diff_abs = symmetric_abs_limit(diff, bias_clip)
        fields = [pred, true, diff]
        labels = ["Predicted Wstd", "True Wstd", "Bias (pred - true)"]
        units = ["m s$^{-1}$", "m s$^{-1}$", "m s$^{-1}$"]
        means = [weighted_nanmean_2d_latlon(pred, lat), weighted_nanmean_2d_latlon(true, lat), weighted_nanmean_2d_latlon(diff, lat)]
        for j, data in enumerate(fields):
            ax = axes[i, j]
            if j < 2:
                im = ax.pcolormesh(lon, lat, data, shading="auto", vmin=vmin, vmax=vmax)
                cb_label = r"$\sigma_W$ [m s$^{-1}$]"
            else:
                im = ax.pcolormesh(lon, lat, data, shading="auto", vmin=-diff_abs, vmax=diff_abs, cmap="coolwarm")
                cb_label = r"bias [m s$^{-1}$]"
            ax.set_title(title_with_mean(f"{labels[j]} at {lev:g} hPa", means[j], units[j]), fontsize=10)
            if j == 0:
                ax.set_ylabel("Latitude")
            ax.set_xlabel("Longitude")
            ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
            ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
            fig.colorbar(im, ax=ax, shrink=0.80, label=cb_label)

    z_vmin, z_vmax = robust_limits(pred_zonal, true_zonal)
    z_diff_abs = symmetric_abs_limit(bias_zonal, bias_clip)
    z_fields = [pred_zonal, true_zonal, bias_zonal]
    z_labels = ["Predicted zonal mean", "True zonal mean", "Bias zonal mean"]
    z_units = ["m s$^{-1}$", "m s$^{-1}$", "m s$^{-1}$"]
    z_means = [weighted_nanmean_latlev(pred_zonal, lat), weighted_nanmean_latlev(true_zonal, lat), weighted_nanmean_latlev(bias_zonal, lat)]
    for j, data in enumerate(z_fields):
        ax = axes[3, j]
        if j < 2:
            im = ax.pcolormesh(lat, zonal_levels, data, shading="auto", vmin=z_vmin, vmax=z_vmax)
            cb_label = r"$\sigma_W$ [m s$^{-1}$]"
        else:
            im = ax.pcolormesh(lat, zonal_levels, data, shading="auto", vmin=-z_diff_abs, vmax=z_diff_abs, cmap="coolwarm")
            cb_label = r"bias [m s$^{-1}$]"
        ax.set_title(title_with_mean(z_labels[j], z_means[j], z_units[j]), fontsize=10)
        ax.set_xlabel("Latitude")
        if j == 0:
            ax.set_ylabel("Pressure (hPa)")
        ax.set_ylim(float(np.nanmax(zonal_levels)), float(np.nanmin(zonal_levels)))
        ax.grid(True, alpha=0.25)
        fig.colorbar(im, ax=ax, shrink=0.80, label=cb_label)

    fig.suptitle(r"MERRA2 symbolic $\sigma_W$ evaluation: maps, bias, and latitude-pressure zonal means", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def fortran_name(name: str) -> str:
    out = re.sub(r"\W+", "_", str(name)).strip("_")
    if not out:
        out = "x"
    if out[0].isdigit():
        out = "x_" + out
    return out[:60]


def fortran_float(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError(f"Cannot emit non-finite Fortran constant: {value}")
    return f"{float(value):.17e}_r8"


def wrap_fortran_assignment(lhs: str, rhs: str, indent: str = "        ", width: int = 118) -> str:
    line = f"{indent}{lhs} = {rhs}"
    if len(line) <= width:
        return line
    out = []
    current = line
    while len(current) > width:
        cuts = [current.rfind(op, 0, width) for op in [" + ", " - ", " * ", " / ", ", "]]
        cut = max(cuts)
        if cut <= len(indent) + 8:
            cut = width
        out.append(current[:cut].rstrip() + " &")
        current = indent + "  & " + current[cut:].lstrip()
    out.append(current)
    return "\n".join(out)


def normalize_fortran_literals(code: str) -> str:
    code = re.sub(r"([0-9]+\.[0-9]*)([dD])([+-]?[0-9]+)", r"\1e\3_r8", code)
    code = re.sub(r"([0-9]+)([dD])([+-]?[0-9]+)", r"\1e\3_r8", code)
    return code


def sympy_fortran_rhs(rhs_expr: str, symbolic_features: Sequence[str], fvar_by_symbolic: Dict[str, str]) -> str:
    import sympy as sp
    expr, syms = parse_sympy_expression(rhs_expr, symbolic_features)
    repl = {syms[s]: sp.Symbol(fvar_by_symbolic[s]) for s in symbolic_features}
    expr2 = expr.xreplace(repl)
    code = sp.fcode(expr2, source_format="free", standard=95, assign_to=None, strict=False)
    code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("!"))
    code = normalize_fortran_literals(code)
    return " ".join(code.split())



DERIVED_FEATURE_DEPENDENCIES = {
    "WIND": ["U", "V"],
    "WABS": ["W"],
    "RH": ["T", "P", "QV"],
    "QVTQV": ["QV", "TQV"],
    "RHO_DIFF": ["AIRD_surf", "AIRD"],
    "PBLH_MINUS_H": ["PBLH", "H"],
    "LOGP": ["P"],
    "SIGMA": ["P", "PS"],
    "ZPBL": ["H", "PBLH"],
    "ABOVE_PBL": ["H", "PBLH"],
    "LOG_TQV": ["TQV"],
    "LOG_QV": ["QV"],
    "LOG_QI": ["QI"],
    "LOG_QL": ["QL"],
}

BASE_FEATURE_TO_FORTRAN_ARG = {
    "T": "T",
    "P": "P",
    "H": "Z",
    "AIRD": "AIRD",
    "U": "U",
    "V": "V",
    "W": "W",
    "QV": "QV",
    "QI": "QI",
    "QL": "QL",
    "PS": "PS",
    "PBLH": "PBLH",
    "TQV": "TQV",
    "AIRD_surf": "AIRD_surf",
}

FORTRAN_3D_ARGS = ["T", "P", "Z", "AIRD", "U", "V", "W", "QV", "QI", "QL"]
FORTRAN_2D_ARGS = ["PS", "PBLH", "TQV", "AIRD_surf"]


def expression_used_symbolic_features(rhs_expr: str, symbolic_features: Sequence[str]) -> List[str]:
    """Return symbolic feature names that actually appear in the selected RHS."""
    expr, syms = parse_sympy_expression(rhs_expr, symbolic_features)
    free = set(expr.free_symbols)
    return [s for s in symbolic_features if syms[s] in free]


def required_internal_feature_closure(used_internal_features: Sequence[str]) -> List[str]:
    """Return used features plus base-feature dependencies for derived predictors."""
    required: List[str] = []
    seen = set()

    def add(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        required.append(name)
        for dep in DERIVED_FEATURE_DEPENDENCIES.get(name, []):
            add(dep)

    for name in used_internal_features:
        add(str(name))
    return required


def fortran_required_arguments(required_internal_features: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """Map required internal/base features to the minimal Fortran raw argument list."""
    args3d: List[str] = []
    args2d: List[str] = []
    unknown: List[str] = []
    for feat in required_internal_features:
        if feat in DERIVED_FEATURE_DEPENDENCIES:
            continue
        arg = BASE_FEATURE_TO_FORTRAN_ARG.get(str(feat))
        if arg is None:
            unknown.append(str(feat))
            continue
        if arg in FORTRAN_3D_ARGS and arg not in args3d:
            args3d.append(arg)
        elif arg in FORTRAN_2D_ARGS and arg not in args2d:
            args2d.append(arg)
    args3d = [arg for arg in FORTRAN_3D_ARGS if arg in args3d]
    args2d = [arg for arg in FORTRAN_2D_ARGS if arg in args2d]
    return args3d, args2d, unknown


def write_fortran_module(
    out_path: Path,
    rhs_expr: str,
    internal_features: Sequence[str],
    symbolic_features: Sequence[str],
    std_info: Dict[str, Dict[str, float]],
    target_transform: str,
    target_scale: float,
    target_power: float,
    clip_pred_min: float,
    module_name: str,
    subroutine_name: str,
    real_kind: str,
    lhs: Optional[str],
    equation_info: Dict[str, object],
    zero_indices: Optional[set] = None,
    zero_records: Optional[List[Dict[str, object]]] = None,
) -> None:
    """Emit a self-contained Fortran 90 module for the selected expression.

    The generated subroutine is intentionally minimal: it only includes raw
    arguments required by symbols that actually appear in the selected RHS, plus
    base inputs needed to compute any derived predictors in that RHS.  For
    example, an expression using LOGP receives P, while one using RHO_DIFF
    receives AIRD and AIRD_surf.
    """
    rkind = "selected_real_kind(12, 300)" if real_kind == "real64" else "selected_real_kind(6, 37)"
    selected = str(equation_info.get("selection", "unknown"))
    zero_indices = set() if zero_indices is None else set(int(i) for i in zero_indices)
    zero_records = [] if zero_records is None else list(zero_records)

    used_symbolic_features = expression_used_symbolic_features(rhs_expr, symbolic_features)
    used_pairs = [
        (iname, sname, i)
        for i, (iname, sname) in enumerate(zip(internal_features, symbolic_features))
        if sname in used_symbolic_features
    ]
    used_internal_features = [iname for iname, _, _ in used_pairs]
    required_internal_features = required_internal_feature_closure(used_internal_features)
    local_feature_names = required_internal_features
    args3d, args2d, unknown_base = fortran_required_arguments(required_internal_features)
    if unknown_base:
        raise ValueError(
            "Cannot generate minimal Fortran for unknown required feature(s): "
            + ", ".join(unknown_base)
            + ". Known base features are: "
            + ", ".join(BASE_FEATURE_TO_FORTRAN_ARG)
        )

    fvar_by_symbolic = {s: "x_" + fortran_name(s) for s in symbolic_features}
    expr_rhs = sympy_fortran_rhs(rhs_expr, symbolic_features, fvar_by_symbolic)
    feature_var = {name: "v_" + fortran_name(name) for name in local_feature_names}

    x_assignments = []
    used_arg_indices = {i for _, _, i in used_pairs}
    for iname, sname, original_index in used_pairs:
        info = std_info.get(iname, {"mean": 0.0, "std": 1.0})
        mean = float(info.get("mean", 0.0))
        std = float(info.get("std", 1.0))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        x_assignments.append((fvar_by_symbolic[sname], feature_var[iname], mean, std, sname, iname, original_index))

    # Warn-but-continue if a zeroed variable is not used by the expression; the
    # evaluator can still zero it in Python, but it has no Fortran effect.
    inactive_zero_records = [r for r in zero_records if int(r.get("arg_index", -9999)) not in used_arg_indices]
    if inactive_zero_records:
        warnings.warn(
            "Fortran generation: requested zero-standardized feature(s) not present in RHS and omitted from minimal subroutine: "
            + ", ".join(f"{r.get('symbolic')}->{r.get('internal')}" for r in inactive_zero_records)
        )

    local_decls = []
    needs_p_hpa = any(name in local_feature_names for name in ["LOGP", "SIGMA", "RH"])
    needs_ps_hpa = "SIGMA" in local_feature_names
    needs_qsat = "RH" in local_feature_names
    for v in ["p_hpa", "ps_hpa", "qsat", "fit", "tmp_sigma", "arg", "es"]:
        if v == "p_hpa" and not needs_p_hpa:
            continue
        if v == "ps_hpa" and not needs_ps_hpa:
            continue
        if v in {"qsat", "arg", "es"} and not needs_qsat:
            continue
        local_decls.append(f"    real(r8) :: {v}")
    for name in local_feature_names:
        local_decls.append(f"    real(r8) :: {feature_var[name]}")
    for _, sname, _ in used_pairs:
        local_decls.append(f"    real(r8) :: {fvar_by_symbolic[sname]}")

    feature_assign: List[str] = []

    def has(name: str) -> bool:
        return name in local_feature_names

    base_assignments = {
        "T": f"        {feature_var.get('T','v_T')} = T(i,j,k)",
        "P": f"        {feature_var.get('P','v_P')} = P(i,j,k)",
        "H": f"        {feature_var.get('H','v_H')} = Z(i,j,k)",
        "AIRD": f"        {feature_var.get('AIRD','v_AIRD')} = AIRD(i,j,k)",
        "U": f"        {feature_var.get('U','v_U')} = U(i,j,k)",
        "V": f"        {feature_var.get('V','v_V')} = V(i,j,k)",
        "W": f"        {feature_var.get('W','v_W')} = W(i,j,k)",
        "QV": f"        {feature_var.get('QV','v_QV')} = QV(i,j,k)",
        "QI": f"        {feature_var.get('QI','v_QI')} = QI(i,j,k)",
        "QL": f"        {feature_var.get('QL','v_QL')} = QL(i,j,k)",
        "PS": f"        {feature_var.get('PS','v_PS')} = PS(i,j)",
        "PBLH": f"        {feature_var.get('PBLH','v_PBLH')} = PBLH(i,j)",
        "TQV": f"        {feature_var.get('TQV','v_TQV')} = TQV(i,j)",
        "AIRD_surf": f"        {feature_var.get('AIRD_surf','v_AIRD_surf')} = AIRD_surf(i,j)",
    }
    for name in BASE_FEATURES:
        if has(name):
            feature_assign.append(base_assignments[name])

    if needs_p_hpa:
        feature_assign.append(f"        p_hpa = pressure_to_hpa({feature_var['P']})")
    if needs_ps_hpa:
        feature_assign.append(f"        ps_hpa = pressure_to_hpa({feature_var['PS']})")

    if has("WIND"):
        feature_assign.append(f"        {feature_var['WIND']} = sqrt({feature_var['U']}*{feature_var['U']} + {feature_var['V']}*{feature_var['V']})")
    if has("WABS"):
        feature_assign.append(f"        {feature_var['WABS']} = abs({feature_var['W']})")
    if has("RH"):
        feature_assign.extend([
            f"        arg = 17.67_r8 * ({feature_var['T']} - 273.15_r8) / max({feature_var['T']} - 29.65_r8, 1.0e-12_r8)",
            "        arg = min(max(arg, -80.0_r8), 80.0_r8)",
            "        es = min(611.2_r8 * exp(arg), 0.99_r8 * p_hpa * 100.0_r8)",
            "        qsat = 0.622_r8 * es / max(p_hpa * 100.0_r8 - (1.0_r8 - 0.622_r8) * es, 1.0e-12_r8)",
            f"        {feature_var['RH']} = min(max({feature_var['QV']} / max(qsat, 1.0e-12_r8), 0.0_r8), 2.0_r8)",
        ])
    if has("QVTQV"):
        feature_assign.append(f"        {feature_var['QVTQV']} = {feature_var['QV']} * {feature_var['TQV']}")
    if has("RHO_DIFF"):
        feature_assign.append(f"        {feature_var['RHO_DIFF']} = {feature_var['AIRD_surf']} - {feature_var['AIRD']}")
    if has("PBLH_MINUS_H"):
        feature_assign.append(f"        {feature_var['PBLH_MINUS_H']} = {feature_var['PBLH']} - {feature_var['H']}")
    if has("LOGP"):
        feature_assign.append(f"        {feature_var['LOGP']} = log(max(p_hpa, 1.0e-6_r8))")
    if has("SIGMA"):
        feature_assign.append(f"        {feature_var['SIGMA']} = p_hpa / max(ps_hpa, 1.0e-6_r8)")
    if has("ZPBL"):
        feature_assign.append(f"        {feature_var['ZPBL']} = {feature_var['H']} / max({feature_var['PBLH']} + 1.0_r8, 1.0_r8)")
    if has("ABOVE_PBL"):
        feature_assign.append(f"        {feature_var['ABOVE_PBL']} = {feature_var['H']} - {feature_var['PBLH']}")
    if has("LOG_TQV"):
        feature_assign.append(f"        {feature_var['LOG_TQV']} = log(1.0_r8 + max({feature_var['TQV']}, 0.0_r8))")
    if has("LOG_QV"):
        feature_assign.append(f"        {feature_var['LOG_QV']} = log(max({feature_var['QV']}, 0.0_r8) + 1.0e-12_r8)")
    if has("LOG_QI"):
        feature_assign.append(f"        {feature_var['LOG_QI']} = log(max({feature_var['QI']}, 0.0_r8) + 1.0e-12_r8)")
    if has("LOG_QL"):
        feature_assign.append(f"        {feature_var['LOG_QL']} = log(max({feature_var['QL']}, 0.0_r8) + 1.0e-12_r8)")

    x_lines = []
    for xvar, fvar, mean, std, sname, iname, original_index in x_assignments:
        x_lines.append(f"        ! {sname} <= {iname}, mean={mean:.17g}, std={std:.17g}")
        if original_index in zero_indices:
            x_lines.append(f"        ! Ablation: forcing standardized {sname} ({iname}) to zero")
            x_lines.append(f"        {xvar} = 0.0_r8")
        else:
            x_lines.append(wrap_fortran_assignment(xvar, f"({fvar} - {fortran_float(mean)}) / {fortran_float(std)}"))

    target_lines = []
    if target_transform == "none":
        target_lines.append("        tmp_sigma = fit")
    elif target_transform == "log":
        target_lines.append("        tmp_sigma = exp(fit)")
    elif target_transform == "log1p":
        target_lines.append("        tmp_sigma = exp(fit) - 1.0_r8")
    elif target_transform == "sqrt":
        target_lines.append("        tmp_sigma = max(fit, 0.0_r8)**2")
    elif target_transform == "power":
        target_lines.append(f"        tmp_sigma = max(fit, 0.0_r8)**({fortran_float(1.0 / float(target_power))})")
    elif target_transform == "asinh":
        target_lines.append(f"        tmp_sigma = {fortran_float(float(target_scale))} * sinh(fit)")
    elif target_transform == "log1p_scaled":
        target_lines.append(f"        tmp_sigma = {fortran_float(float(target_scale))} * (exp(fit) - 1.0_r8)")
    else:
        raise ValueError(f"Unsupported target transform for Fortran: {target_transform}")
    if np.isfinite(clip_pred_min):
        target_lines.append(f"        sigmaw(i,j,k) = max(tmp_sigma, {fortran_float(float(clip_pred_min))})")
    else:
        target_lines.append("        sigmaw(i,j,k) = tmp_sigma")

    fit_assignment = wrap_fortran_assignment("fit", expr_rhs)

    arg_list = args3d + args2d + ["sigmaw"]
    subroutine_args = ", ".join(arg_list)

    input_decl_lines: List[str] = []
    for arg in args3d:
        input_decl_lines.append(f"    real(r8), intent(in)  :: {arg}(:,:,:)")
    for arg in args2d:
        input_decl_lines.append(f"    real(r8), intent(in)  :: {arg}(:,:)")
    input_decls = "\n".join(input_decl_lines)
    if input_decls:
        input_decls += "\n"

    shape_checks: List[str] = []
    for arg in args3d:
        shape_checks.append(f"    if (size({arg},1) /= im .or. size({arg},2) /= jm .or. size({arg},3) /= lm) stop \"{arg} has incompatible shape\"")
    for arg in args2d:
        shape_checks.append(f"    if (size({arg},1) /= im .or. size({arg},2) /= jm) stop \"{arg} has incompatible shape\"")
    shape_checks_text = "\n".join(shape_checks)
    if shape_checks_text:
        shape_checks_text += "\n"

    used_feature_comment = ", ".join(f"{s}->{i}" if s != i else s for i, s, _ in used_pairs) or "none/constant expression"
    raw_arg_comment = ", ".join(args3d + args2d) or "none"

    module_text = f"""module {module_name}
  implicit none
  private
  public :: {subroutine_name}
  integer, parameter :: r8 = {rkind}

contains

  pure real(r8) function sqrt_abs(x)
    real(r8), intent(in) :: x
    sqrt_abs = sqrt(abs(x))
  end function sqrt_abs

  pure real(r8) function log_abs(x)
    real(r8), intent(in) :: x
    log_abs = log(abs(x) + 1.0e-6_r8)
  end function log_abs

  pure real(r8) function exp_clip(x)
    real(r8), intent(in) :: x
    exp_clip = exp(min(max(x, -20.0_r8), 20.0_r8))
  end function exp_clip

  pure real(r8) function square(x)
    real(r8), intent(in) :: x
    square = x*x
  end function square

  pure real(r8) function cube(x)
    real(r8), intent(in) :: x
    cube = x*x*x
  end function cube

  pure real(r8) function pressure_to_hpa(p_in)
    real(r8), intent(in) :: p_in
    if (abs(p_in) > 2000.0_r8) then
      pressure_to_hpa = 0.01_r8 * p_in
    else
      pressure_to_hpa = p_in
    end if
  end function pressure_to_hpa

  subroutine {subroutine_name}({subroutine_args})
    ! Auto-generated from evaluate_merra2_wstd_expression_v10_zero_minfortran.py
    ! Equation selection: {selected}
    ! LHS: {str(lhs) if lhs else 'none'}
    ! RHS: {rhs_expr}
    ! Target inverse: {inverse_target_description(target_transform, target_scale, target_power)}
    ! RHS feature symbols used: {used_feature_comment}
    ! Minimal raw inputs used: {raw_arg_comment}
    ! Units expected by this generated evaluator:
    !   T [K], P [hPa or Pa], Z/H [m], AIRD [kg m-3], U/V/W [m s-1],
    !   QV/QI/QL [kg kg-1], PS in the same raw units used by training,
    !   PBLH [m], TQV [kg m-2], AIRD_surf [kg m-3].
{input_decls}    real(r8), intent(out) :: sigmaw(:,:,:)
    integer :: i, j, k
    integer :: im, jm, lm
{chr(10).join(local_decls)}

    im = size(sigmaw, 1)
    jm = size(sigmaw, 2)
    lm = size(sigmaw, 3)

{shape_checks_text}    do k = 1, lm
      do j = 1, jm
        do i = 1, im
{chr(10).join(feature_assign)}
{chr(10).join(x_lines)}
{fit_assignment}
{chr(10).join(target_lines)}
        end do
      end do
    end do
  end subroutine {subroutine_name}

end module {module_name}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(module_text)

def json_safe(obj):
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [json_safe(v) for v in obj]
    return obj


def safe_filename_component(value: object, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (text[:max_len] if text else "expr")


def expression_output_tag(args: argparse.Namespace, manifest: Optional[Dict[str, object]] = None) -> str:
    if getattr(args, "expression", None):
        digest = hashlib.sha1(str(args.expression).encode("utf-8")).hexdigest()[:8]
        return f"expr_{digest}"
    if getattr(args, "expression_file", None):
        return safe_filename_component(Path(str(args.expression_file)).stem)
    if getattr(args, "run_dir", None) and str(args.equation_index).strip().lower() in {"best", "validation", "val", "rerank", "reranked", "validation_rerank"}:
        p = Path(args.run_dir).expanduser() / "selected_equation_validation.json"
        if p.exists() and not args.equations_csv:
            return "validation_rerank"
    mexpr = expression_from_manifest(manifest or {})
    if mexpr and getattr(args, "expression_from_manifest", False):
        digest = hashlib.sha1(mexpr.encode("utf-8")).hexdigest()[:8]
        return f"manifest_expr_{digest}"
    return safe_filename_component(args.equation_index)


def main() -> int:
    args = parse_args()
    merra_root = require_path(args.merra_path, "MERRAPATH")
    sw_root = require_path(args.sw_path, "SWPATH")
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.year is None and not args.fortran_only:
        raise SystemExit("ERROR: --year is required unless --fortran-only is used")

    manifest_path = find_manifest(args)
    manifest: Dict[str, object] = read_json(manifest_path) if manifest_path is not None and manifest_path.exists() else {}

    output_tag = expression_output_tag(args, manifest) + zero_output_suffix(args.zero_standardized)
    out_png = None if args.fortran_only else (Path(args.png).expanduser() if args.png else outdir / f"wstd_eval_{args.year}_{args.ndays}days_{output_tag}.png")
    if out_png is not None and out_png.exists() and not args.overwrite:
        raise SystemExit(f"ERROR: output PNG exists: {out_png}. Use --overwrite.")

    lhs, rhs_expr, equation_info = load_expression(args, manifest)
    target_transform = resolve_target_transform(args, manifest, lhs)
    target_scale, target_power = resolve_target_params(args, manifest)
    target_bins = resolve_target_bins(args, manifest)
    pressure_bins = resolve_pressure_bins(args, manifest)
    internal_features, symbolic_features, std_info = load_feature_config(manifest, rhs_expr, args)
    eval_func, sympy_expr = build_lambdified_expression(rhs_expr, symbolic_features)
    zero_indices, zero_records = resolve_zero_standardized_features(
        args.zero_standardized, internal_features, symbolic_features,
        ignore_missing=args.zero_standardized_ignore_missing,
    )

    sensitivity_png: Optional[Path] = None
    if not args.no_sensitivity_plot:
        sensitivity_png = Path(args.sensitivity_png).expanduser() if args.sensitivity_png else outdir / f"wstd_sensitivity_{output_tag}.png"
        if sensitivity_png.exists() and not args.overwrite:
            raise SystemExit(f"ERROR: sensitivity PNG exists: {sensitivity_png}. Use --overwrite.")
        sens_range = parse_range_pair(args.sensitivity_range, name="--sensitivity-range")
        plot_sensitivity_curves(
            eval_func, internal_features, symbolic_features, target_transform,
            target_scale, target_power, args.clip_pred_min, sensitivity_png,
            args.dpi, sens_range, args.sensitivity_points, zero_indices,
        )
        print(f"Wrote sensitivity plot: {sensitivity_png}")

    fortran_path: Optional[Path] = None
    if args.fortran_out:
        fortran_path = Path(args.fortran_out).expanduser()
        if fortran_path.exists() and not args.overwrite:
            raise SystemExit(f"ERROR: Fortran output exists: {fortran_path}. Use --overwrite.")
        write_fortran_module(
            fortran_path, rhs_expr, internal_features, symbolic_features, std_info,
            target_transform, target_scale, target_power, args.clip_pred_min,
            args.fortran_module, args.fortran_subroutine, args.fortran_real_kind,
            lhs, equation_info, zero_indices, zero_records,
        )
        print(f"Wrote Fortran module: {fortran_path}")

    if args.fortran_only:
        print("Expression RHS:", rhs_expr)
        print("Equation selection:", equation_info.get("selection"))
        print("Target transform:", target_transform)
        print("Target inverse:", inverse_target_description(target_transform, target_scale, target_power))
        print("Zeroed standardized features:", "none" if not zero_records else ", ".join(f"{r['symbolic']}->{r['internal']}" for r in zero_records))
        return 0

    assert args.year is not None
    assert out_png is not None

    rng = np.random.default_rng(args.seed)
    day_files = choose_random_valid_days(merra_root, sw_root, args.year, args.ndays, rng)
    requested_levels = parse_levels(args.levels_hpa)
    chunks = {"time": 1} if args.chunks == "time" else None

    print("Selected days:", ", ".join(d.yyyymmdd for d in day_files))
    print("Expression RHS:", rhs_expr)
    print("Equation selection:", equation_info.get("selection"))
    print("Target transform:", target_transform)
    print("Target inverse:", inverse_target_description(target_transform, target_scale, target_power))
    print("Wstd bins:", "disabled" if target_bins is None else ", ".join(f"{x:g}" for x in target_bins))
    print("Pressure bins hPa:", "disabled" if pressure_bins is None else ", ".join(f"{x:g}" for x in pressure_bins))
    print("P<=PS filter:", "disabled" if args.allow_p_greater_than_ps else f"enabled, tolerance={args.ps_tolerance_hpa:g} hPa")
    print("Data/internal features:", ", ".join(internal_features))
    print("Symbolic features:", ", ".join(symbolic_features))
    print("Zeroed standardized features:", "none" if not zero_records else ", ".join(f"{r['symbolic']}->{r['internal']}" for r in zero_records))
    alias_pairs = [f"{s}->{d}" for s, d in zip(symbolic_features, internal_features) if s != d]
    if alias_pairs:
        print("Feature aliases applied:", ", ".join(alias_pairs))

    unavailable = [f for f in internal_features if f not in ALL_FEATURES]
    if unavailable:
        warnings.warn("Expression has feature names not in known MERRA2 feature catalog: " + ", ".join(unavailable))

    pred_sum: Dict[float, Optional[np.ndarray]] = {lev: None for lev in requested_levels}
    true_sum: Dict[float, Optional[np.ndarray]] = {lev: None for lev in requested_levels}
    count_sum: Dict[float, Optional[np.ndarray]] = {lev: None for lev in requested_levels}
    actual_levels: Dict[float, float] = {}
    metrics_global = OnlineMetrics()
    metrics_level: Dict[float, OnlineMetrics] = {lev: OnlineMetrics() for lev in requested_levels}
    metrics_pressure_bin: Dict[str, OnlineMetrics] = {}
    if pressure_bins is not None:
        metrics_pressure_bin = {pressure_bracket_label(pressure_bins[i + 1], pressure_bins[i]): OnlineMetrics() for i in range(len(pressure_bins) - 1)}
    metrics_target_bin: Dict[str, OnlineMetrics] = {}
    if target_bins is not None:
        metrics_target_bin = {f"{target_bins[i]:.6g}-{target_bins[i+1]:.6g}": OnlineMetrics() for i in range(len(target_bins) - 1)}
        metrics_target_bin["outside_bins"] = OnlineMetrics()

    lat_vals: Optional[np.ndarray] = None
    lon_vals: Optional[np.ndarray] = None
    zonal_levels: Optional[np.ndarray] = None
    pred_zonal_sum: Optional[np.ndarray] = None
    true_zonal_sum: Optional[np.ndarray] = None
    zonal_count_sum: Optional[np.ndarray] = None
    evaluated_days: List[str] = []

    for dfiles in day_files:
        print(f"Reading/evaluating {dfiles.yyyymmdd}")
        try:
            ds = open_day_dataset(dfiles, pmin=args.pmin, resample_freq=args.resample, chunks=chunks)
            if lat_vals is None:
                lat_vals = np.asarray(ds["lat"].values, dtype=float)
                lon_vals = np.asarray(ds["lon"].values, dtype=float)
            actual_for_day = nearest_level_values(ds, requested_levels)
            requested_by_actual: Dict[float, List[float]] = {}
            for req_lev, actual_lev in zip(requested_levels, actual_for_day):
                actual_levels[req_lev] = actual_lev
                requested_by_actual.setdefault(actual_lev, []).append(req_lev)

            ds_levs = np.asarray(ds["lev"].values, dtype=float)
            if zonal_levels is None:
                zonal_levels = ds_levs.astype(np.float64)
                nlev = len(zonal_levels)
                nlat = len(lat_vals) if lat_vals is not None else int(ds.sizes["lat"])
                pred_zonal_sum = np.zeros((nlev, nlat), dtype=np.float64)
                true_zonal_sum = np.zeros((nlev, nlat), dtype=np.float64)
                zonal_count_sum = np.zeros((nlev, nlat), dtype=np.float64)

            assert zonal_levels is not None
            assert pred_zonal_sum is not None and true_zonal_sum is not None and zonal_count_sum is not None
            evaluated_days.append(dfiles.yyyymmdd)

            for iz, actual_lev in enumerate(ds_levs):
                dsl = ds.sel(lev=float(actual_lev))
                raw: Dict[str, np.ndarray] = {v: np.asarray(dsl[v].load().values, dtype=np.float32) for v in BASE_FEATURES}
                raw = add_derived_features(raw)

                args_for_expr = []
                for iname, sname in zip(internal_features, symbolic_features):
                    if iname not in raw:
                        raise KeyError(f"Feature {iname!r} required by expression variable {sname!r} is not available")
                    args_for_expr.append(standardize_feature(raw[iname], iname, std_info))
                args_for_expr = apply_zero_standardized_args(args_for_expr, zero_indices)

                pred_fit = np.asarray(eval_func(*args_for_expr), dtype=np.float64)
                if pred_fit.shape == ():
                    pred_fit = np.zeros_like(raw["T"], dtype=np.float64) + float(pred_fit)
                pred = inverse_target(pred_fit, target_transform, target_scale=target_scale, target_power=target_power)
                if np.isfinite(args.clip_pred_min):
                    pred = np.maximum(pred, float(args.clip_pred_min))
                true = np.asarray(dsl[TARGET].load().values, dtype=np.float64)
                valid = np.isfinite(pred) & np.isfinite(true)
                if not args.allow_p_greater_than_ps:
                    valid &= p_le_ps_mask(raw["P"], raw["PS"], args.ps_tolerance_hpa)

                metrics_global.update(true[valid], pred[valid])
                if pressure_bins is not None:
                    p_hpa = pressure_to_hpa_np(raw["P"])
                    for ib in range(len(pressure_bins) - 1):
                        upper = pressure_bins[ib]; lower = pressure_bins[ib + 1]
                        if ib == len(pressure_bins) - 2:
                            bm = valid & np.isfinite(p_hpa) & (p_hpa <= upper) & (p_hpa >= lower)
                        else:
                            bm = valid & np.isfinite(p_hpa) & (p_hpa <= upper) & (p_hpa > lower)
                        if np.any(bm):
                            metrics_pressure_bin[pressure_bracket_label(lower, upper)].update(true[bm], pred[bm])
                if target_bins is not None:
                    yt_valid = true[valid]
                    yp_valid = pred[valid]
                    b_valid = target_bin_indices(yt_valid, target_bins)
                    for ib in range(len(target_bins) - 1):
                        bm = b_valid == ib
                        if np.any(bm):
                            metrics_target_bin[f"{target_bins[ib]:.6g}-{target_bins[ib+1]:.6g}"].update(yt_valid[bm], yp_valid[bm])
                    bm = b_valid < 0
                    if np.any(bm):
                        metrics_target_bin["outside_bins"].update(yt_valid[bm], yp_valid[bm])

                pred_valid = np.where(valid, pred, np.nan)
                true_valid = np.where(valid, true, np.nan)
                pred_zonal_sum[iz, :] += np.nansum(pred_valid, axis=(0, 2))
                true_zonal_sum[iz, :] += np.nansum(true_valid, axis=(0, 2))
                zonal_count_sum[iz, :] += np.sum(valid, axis=(0, 2))

                for req_lev in requested_by_actual.get(float(actual_lev), []):
                    metrics_level[req_lev].update(true[valid], pred[valid])
                    if pred_sum[req_lev] is None:
                        pred_sum[req_lev] = np.zeros(true.shape[-2:], dtype=np.float64)
                        true_sum[req_lev] = np.zeros(true.shape[-2:], dtype=np.float64)
                        count_sum[req_lev] = np.zeros(true.shape[-2:], dtype=np.float64)
                    pred_sum[req_lev] += np.nansum(pred_valid, axis=0)  # type: ignore[operator]
                    true_sum[req_lev] += np.nansum(true_valid, axis=0)  # type: ignore[operator]
                    count_sum[req_lev] += np.sum(valid, axis=0)         # type: ignore[operator]

            ds.close()
        except Exception as exc:
            warnings.warn(f"Skipping/evaluation failed for {dfiles.yyyymmdd}: {type(exc).__name__}: {exc}")
            continue

    if lat_vals is None or lon_vals is None:
        raise SystemExit("ERROR: no days were successfully evaluated")

    pred_maps: Dict[float, np.ndarray] = {}
    true_maps: Dict[float, np.ndarray] = {}
    bias_maps: Dict[float, np.ndarray] = {}
    for lev in requested_levels:
        if pred_sum[lev] is None or true_sum[lev] is None or count_sum[lev] is None:
            raise SystemExit(f"ERROR: no successful evaluations for level {lev:g} hPa")
        cnt = count_sum[lev]
        pred_avg = np.divide(pred_sum[lev], cnt, out=np.full_like(pred_sum[lev], np.nan), where=cnt > 0)
        true_avg = np.divide(true_sum[lev], cnt, out=np.full_like(true_sum[lev], np.nan), where=cnt > 0)
        pred_maps[lev] = pred_avg
        true_maps[lev] = true_avg
        bias_maps[lev] = bias_diff(pred_avg, true_avg)

    if zonal_levels is None or pred_zonal_sum is None or true_zonal_sum is None or zonal_count_sum is None:
        raise SystemExit("ERROR: no zonal-mean fields were accumulated")
    pred_zonal = np.divide(pred_zonal_sum, zonal_count_sum, out=np.full_like(pred_zonal_sum, np.nan), where=zonal_count_sum > 0)
    true_zonal = np.divide(true_zonal_sum, zonal_count_sum, out=np.full_like(true_zonal_sum, np.nan), where=zonal_count_sum > 0)
    bias_zonal = bias_diff(pred_zonal, true_zonal)

    plot_results(lat_vals, lon_vals, requested_levels, pred_maps, true_maps, bias_maps, zonal_levels, pred_zonal, true_zonal, bias_zonal, out_png, args.dpi, args.bias_clip)

    metrics = {
        "year": args.year,
        "ndays_requested": args.ndays,
        "selected_days": [d.yyyymmdd for d in day_files],
        "evaluated_days": evaluated_days,
        "requested_levels_hpa": requested_levels,
        "zonal_levels_hpa": [] if zonal_levels is None else [float(x) for x in zonal_levels],
        "actual_levels_hpa_by_requested": {str(k): v for k, v in actual_levels.items()},
        "metrics_global": metrics_global.finalize(),
        "metrics_by_level_hpa": {f"{lev:g}": metrics_level[lev].finalize() for lev in requested_levels},
        "equation_info": equation_info,
        "expression_lhs": lhs,
        "expression_rhs": rhs_expr,
        "expression_source": equation_info.get("selection"),
        "output_tag": output_tag,
        "zero_standardized_requested": parse_name_list(args.zero_standardized),
        "zero_standardized_features": zero_records,
        "expression_sympy": sympy_expr,
        "target_transform": target_transform,
        "target_inverse": inverse_target_description(target_transform, target_scale, target_power),
        "target_scale": float(target_scale),
        "target_power": float(target_power),
        "pressure_bracket_edges_hpa": None if pressure_bins is None else [float(x) for x in pressure_bins],
        "wstd_bracket_edges": None if target_bins is None else [float(x) for x in target_bins],
        "target_bins": None if target_bins is None else [float(x) for x in target_bins],
        "p_gt_ps_filter_enabled": bool(not args.allow_p_greater_than_ps),
        "ps_tolerance_hpa": float(args.ps_tolerance_hpa),
        "metrics_by_pressure_bracket_hpa": {k: v.finalize() for k, v in metrics_pressure_bin.items()},
        "metrics_by_target_bin": {k: v.finalize() for k, v in metrics_target_bin.items()},
        "manifest_path": str(manifest_path) if manifest_path else None,
        "feature_names_internal": internal_features,
        "feature_names_symbolic": symbolic_features,
        "feature_data_mapping_symbolic_to_internal": {s: d for s, d in zip(symbolic_features, internal_features)},
        "feature_standardization_used": std_info,
        "available_features": ALL_FEATURES,
        "output_png": str(out_png),
        "sensitivity_png": str(sensitivity_png) if sensitivity_png else None,
        "fortran_out": str(fortran_path) if fortran_path else None,
        "plot_difference_quantity": "bias_pred_minus_true_m_s-1",
        "merra_path": str(merra_root),
        "sw_path": str(sw_root),
    }
    metrics_json = outdir / f"wstd_eval_metrics_{args.year}_{args.ndays}days_{output_tag}.json"
    metrics_txt = outdir / f"wstd_eval_metrics_{args.year}_{args.ndays}days_{output_tag}.txt"
    with open(metrics_json, "w") as f:
        json.dump(json_safe(metrics), f, indent=2)
        f.write("\n")
    with open(metrics_txt, "w") as f:
        f.write("MERRA2 symbolic Wstd evaluation\n")
        f.write("================================\n")
        f.write(f"Year: {args.year}\n")
        f.write(f"Selected days: {', '.join(d.yyyymmdd for d in day_files)}\n")
        f.write(f"Evaluated days: {', '.join(evaluated_days)}\n")
        f.write(f"PNG: {out_png}\n")
        if sensitivity_png:
            f.write(f"Sensitivity PNG: {sensitivity_png}\n")
        if fortran_path:
            f.write(f"Fortran module: {fortran_path}\n")
        if lhs:
            f.write(f"Expression LHS: {lhs}\n")
        f.write(f"Expression RHS: {rhs_expr}\n")
        f.write(f"Expression source: {equation_info.get('selection')}\n")
        f.write("Zeroed standardized features: " + ("none" if not zero_records else ", ".join(f"{r['symbolic']}->{r['internal']}" for r in zero_records)) + "\n")
        f.write(f"Target transform: {target_transform}\n")
        f.write(f"Target inverse: {inverse_target_description(target_transform, target_scale, target_power)}\n")
        f.write("Wstd bins: " + ("disabled" if target_bins is None else ", ".join(f"{x:g}" for x in target_bins)) + "\n")
        f.write("Pressure bins hPa: " + ("disabled" if pressure_bins is None else ", ".join(f"{x:g}" for x in pressure_bins)) + "\n")
        f.write("P<=PS filter: " + ("disabled" if args.allow_p_greater_than_ps else f"enabled, tolerance={args.ps_tolerance_hpa:g} hPa") + "\n")
        f.write("Features: " + ", ".join(f"{s}->{d}" if s != d else s for s, d in zip(symbolic_features, internal_features)) + "\n\n")
        f.write("Global metrics, all evaluated levels after P<=PS mask:\n")
        f.write(json.dumps(metrics["metrics_global"], indent=2) + "\n\n")
        f.write("Metrics by requested map level hPa:\n")
        f.write(json.dumps(metrics["metrics_by_level_hpa"], indent=2) + "\n\n")
        f.write("Metrics by pressure bracket hPa:\n")
        f.write(json.dumps(metrics["metrics_by_pressure_bracket_hpa"], indent=2) + "\n\n")
        f.write("Metrics by true-Wstd target bin:\n")
        f.write(json.dumps(metrics["metrics_by_target_bin"], indent=2) + "\n")

    print("Wrote:")
    print(f"  {out_png}")
    if sensitivity_png:
        print(f"  {sensitivity_png}")
    if fortran_path:
        print(f"  {fortran_path}")
    print(f"  {metrics_json}")
    print(f"  {metrics_txt}")
    print("Global metrics:")
    print(json.dumps(metrics["metrics_global"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
