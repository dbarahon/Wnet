#!/usr/bin/env python3
"""
Fit ONE compact symbolic expression for MERRA2 training-ready Wstd samples.

This version reads a pre-extracted 1-D sample-table NetCDF file such as

    pysr_merra2_wstd_150d_2010_2015.nc

with variables on dimension ``sample``:

    T, P, H, AIRD, U, V, W, QV, QI, QL, PS, PBLH, TQV, AIRD_surf, Wstd,
    time, lev, lat, lon, yyyymmdd

The expensive MERRA2/Wstd file matching and day-level sampling are therefore
skipped.  The script can randomly draw rows or build a stratified training pool from
the sample table, filters invalid values, computes derived predictors in
physical units, standardizes the final training pool, optionally applies
tail-aware sample weights, and fits a single global expression:

    Wstd = f(features)

New MERRA2-ready derived feature:

    PBLH_MINUS_H = PBLH - H

Other useful derived features retained from the ERA5 workflow:

    WIND     = sqrt(U**2 + V**2)
    WABS     = abs(W)
    RH       = QV / Qsat_liq(T, P), clipped to [0, 2]
    QVTQV    = QV * TQV
    RHO_DIFF = AIRD_surf - AIRD

Additional smooth vertical/regime predictors for global expressions:

    LOGP       = log(P)
    SIGMA      = P / PS
    ZPBL       = H / (PBLH + 1)
    ABOVE_PBL  = H - PBLH
    LOG_TQV    = log1p(TQV)
    LOG_QV     = log(QV + 1e-12)
    LOG_QI     = log(QI + 1e-12)
    LOG_QL     = log(QL + 1e-12)

Preset workflow:

    # 30-minute diagnostic. Tests whether the setup is learnable.
    python Wstd_sym_MERRA2_v8_two_presets.py pysr_merra2_wstd_150d_2010_2015.nc --run-preset quick

    # Long Slurm run. If inside Slurm, runs until scheduler walltime minus buffer.
    python Wstd_sym_MERRA2_v8_two_presets.py pysr_merra2_wstd_150d_2010_2015.nc --run-preset long

Important preset behavior:
    * quick is now a 2-hour exploratory screen with fewer samples, more populations,
      extended operators, and no maxsize warmup.
    * long keeps large niterations and sets warmup_maxsize_by=0.0; the
      practical stopping criterion is the resolved timeout/walltime.
    * You can override any preset field explicitly on the command line, e.g.
      --run-preset quick --max-total-samples 200000 --timeout-seconds 3600.

Requires symbolic mode:
    pip install pysr
    # plus a working Julia installation, which PySR will use.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import threading
import traceback
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import xarray as xr

try:
    from sklearn.linear_model import RidgeCV
except Exception:
    RidgeCV = None

try:
    import netCDF4 as nc4
except Exception:
    nc4 = None


RD = 287.0
FILL_ABS_LIMIT = 9.0e3  # the training-ready file uses _FillValue=-9999.f

BASE_FEATURES = [
    "T",
    "P",           # hPa in the provided table
    "H",           # height/geopotential-height-like coordinate from inst3_3d_asm_Np
    "AIRD",
    "U",
    "V",
    "W",
    "QV",
    "QI",
    "QL",
    "PS",
    "PBLH",
    "TQV",         # MERRA2 total precipitable water / TCWV analogue
    "AIRD_surf",
]

DERIVED_FEATURES = [
    "WIND",
    "WABS",
    "RH",
    "QVTQV",
    "RHO_DIFF",
    "PBLH_MINUS_H",
    "LOGP",
    "SIGMA",
    "ZPBL",
    "ABOVE_PBL",
    "LOG_TQV",
    "LOG_QV",
    "LOG_QI",
    "LOG_QL",
]

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES

# Good first-pass feature set: avoids U and V separately because WIND is already
# supplied, but keeps the thermodynamic, vertical-location, and condensate terms.
COMPACT_FEATURES = [
    "T",
    "P",
    "LOGP",
    "WIND",
    "W",
    "QV",
    "PS",
    "PBLH",
    "TQV",
    "AIRD_surf",
    "RHO_DIFF",
    "ZPBL",
    "ABOVE_PBL",
]

DERIVED_FEATURE_DESCRIPTIONS = {
    "WIND": "sqrt(U**2 + V**2), horizontal wind speed in m/s before standardization",
    "WABS": "abs(W), vertical velocity magnitude in m/s before standardization",
    "RH": "QV / Qsat_liq(T, P), clipped to [0, 2] before standardization; P is hPa",
    "QVTQV": "QV * TQV, local-column moisture interaction before standardization",
    "RHO_DIFF": "AIRD_surf - AIRD, density-depth/level proxy before standardization",
    "PBLH_MINUS_H": "PBLH - H, boundary-layer-depth relative to sample height before standardization",
    "LOGP": "log(P_hPa), smooth pressure coordinate before standardization",
    "SIGMA": "P_hPa / PS_hPa, terrain-following pressure coordinate before standardization",
    "ZPBL": "H / (PBLH + 1), sample height relative to boundary-layer depth before standardization",
    "ABOVE_PBL": "H - PBLH, height above boundary-layer top before standardization",
    "LOG_TQV": "log1p(TQV), compressed column moisture before standardization",
    "LOG_QV": "log(QV + 1e-12), compressed local vapor before standardization",
    "LOG_QI": "log(QI + 1e-12), compressed cloud-ice mixing ratio before standardization",
    "LOG_QL": "log(QL + 1e-12), compressed cloud-liquid mixing ratio before standardization",
}

# PySR/SymPy-safe names.  Keep symbolic names close to physical names.
SYMBOLIC_NAME = {name: name for name in ALL_FEATURES}
SYMBOLIC_NAME.update({
    "PBLH_MINUS_H": "PBLH_minus_H",
})

# Optional old ERA5/NN constants.  The default for this MERRA2 sample-table
# workflow is sample-pool standardization, because the new table includes P, H,
# PS, PBLH, TQV, QI, and QL directly and these should be scaled consistently.
ORIGINAL_MEANS = {
    "T": 243.9,
    "AIRD": 0.6,
    "U": 6.3,
    "V": 0.013,
    "W": 0.0002,
    "QV": 0.002,
    "QI": 9.75e-7,
    "QL": 7.87e-6,
    "PBLH": 889.0,
    "TQV": 23.4,
    "AIRD_surf": 1.0,
}
ORIGINAL_STDS = {
    "T": 30.3,
    "AIRD": 0.42,
    "U": 16.1,
    "V": 7.9,
    "W": 0.05,
    "QV": 0.0036,
    "QI": 7.09e-6,
    "QL": 2.7e-5,
    "PBLH": 387.0,
    "TQV": 15.6,
    "AIRD_surf": 0.1,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit one global symbolic expression for Wstd from a MERRA2 "
            "training-ready 1-D sample-table NetCDF file."
        )
    )
    p.add_argument(
        "input_file",
        nargs="?",
        help="Training-ready sample-table NetCDF file, e.g. pysr_merra2_wstd_150d_2010_2015.nc",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: symbolic_Wstd_MERRA2_<input-stem>, or symbolic_Wstd_MERRA2_<preset>_<input-stem> when --run-preset is used",
    )
    p.add_argument(
        "--run-preset",
        "--preset",
        dest="run_preset",
        choices=["none", "quick", "long"],
        default="none",
        help=(
            "Apply a pre-tuned run configuration. 'quick' is a short learnability test, "
            "'long' is a final Slurm-style run. "
            "Explicit command-line options override preset values."
        ),
    )
    p.add_argument(
        "--print-config-only",
        action="store_true",
        help="Apply any preset, print the resolved run configuration, and exit before sampling/fitting.",
    )

    p.add_argument(
        "--features",
        default="compact",
        help=(
            "Comma-separated feature subset, 'compact', or 'all'. Default: compact. "
            "Allowed names: " + ",".join(ALL_FEATURES) + ". "
            "Aliases accepted: Q->QV, BLH->PBLH, TCWV->TQV, QTCWV->QVTQV, "
            "PBLH-H/PBLH_H/PBLHmH->PBLH_MINUS_H."
        ),
    )
    p.add_argument("--list-features", action="store_true", help="Print allowed feature names and exit")

    p.add_argument(
        "--max-total-samples",
        type=int,
        default=1_000_000,
        help=(
            "Number of valid rows to draw from the training-ready table. "
            "Use a larger value for final runs; 1e6 is a good first pass."
        ),
    )
    p.add_argument(
        "--candidate-batch-size",
        type=int,
        default=750_000,
        help="Random candidate rows read per sampling attempt before filtering.",
    )
    p.add_argument("--oversample", type=float, default=2.5, help="Candidate multiplier relative to remaining needed rows")
    p.add_argument("--max-sampling-attempts", type=int, default=80)
    p.add_argument(
        "--sampling-mode",
        choices=["random", "wstd_stratified", "wstd_pressure_stratified"],
        default="random",
        help=(
            "Training-pool sampling strategy. 'random' preserves the old behavior. "
            "'wstd_stratified' balances true-Wstd bins. 'wstd_pressure_stratified' "
            "balances true-Wstd x pressure bins while still fitting one global expression."
        ),
    )
    p.add_argument(
        "--wstd-sampling-edges",
        default="0.001,0.01,0.1, 1,5,10",
        help=(
            "Ascending true-Wstd edges for stratified sampling and inverse-bin weighting. "
            "Default: 0.001,0.01,,0.1,,1, 10."
        ),
    )
    p.add_argument(
        "--pressure-sampling-edges",
        default="1000,900,700,500,300,100,10",
        help=(
            "Descending pressure edges in hPa for Wstd x pressure stratified sampling. "
            "Default: 1000,900,700,500,300,100."
        ),
    )
    p.add_argument(
        "--stratified-relax-frac",
        type=float,
        default=0.70,
        help=(
            "After this fraction of max sampling attempts, allow extra valid rows from "
            "already-full strata so sparse/empty bins do not prevent filling the pool."
        ),
    )
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--levels-hpa", default="all", help="Comma-separated levels to retain, or 'all'")
    p.add_argument(
        "--level-tolerance-hpa",
        type=float,
        default=0.75,
        help="Tolerance when matching --levels-hpa to the table lev coordinate.",
    )
    p.add_argument(
        "--ps-tolerance-hpa",
        type=float,
        default=0.25,
        help=(
            "Tolerance for rejecting below-surface samples. By default a sample is rejected "
            "when pressure P is greater than surface pressure PS by more than this many hPa. "
            "The code auto-detects whether P/PS are in Pa or hPa before comparing."
        ),
    )
    p.add_argument(
        "--allow-p-greater-than-ps",
        action="store_true",
        help="Disable the default hard filter that removes samples with P > PS.",
    )
    p.add_argument(
        "--pressure-bracket-edges",
        default="1000,800,500,200,10",
        help=(
            "Comma-separated descending pressure-bin edges in hPa for diagnostic metrics. "
            "Use 'none' or 'off' to disable pressure-bracket diagnostics. "
            "Default: 1000,800,500,200,10."
        ),
    )
    p.add_argument(
        "--wstd-bracket-edges",
        default="0.1,1,10",
        help=(
            "Comma-separated ascending true-Wstd bin edges for diagnostic metrics. "
            "Use 'none' or 'off' to disable Wstd-bracket diagnostics. "
            "Default: 0.1,1,10."
        ),
    )
    p.add_argument(
        "--days",
        default=None,
        help="Comma-separated YYYYMMDD dates to retain. Overrides --max-days/--day-selection.",
    )
    p.add_argument(
        "--max-days",
        type=int,
        default=0,
        help=(
            "Optional number of dates to retain from the file's selected_days global attribute. "
            "0 means use all dates in the table."
        ),
    )
    p.add_argument(
        "--day-selection",
        choices=["first", "random"],
        default="random",
        help="How to choose --max-days from the selected_days global attribute.",
    )

    p.add_argument("--engine", choices=["pysr", "linear"], default="pysr", help="Fit engine")
    p.add_argument(
        "--standardization",
        choices=["sample", "original", "none"],
        default="sample",
        help=(
            "Feature scaling before fitting. 'sample' standardizes every selected feature "
            "using the final sampled pool; 'original' uses old NN constants where available "
            "and sample stats otherwise; 'none' uses physical units."
        ),
    )
    p.add_argument(
        "--no-standardize",
        action="store_true",
        help="Alias for --standardization none, kept for compatibility with the ERA5 script.",
    )

    # Compact/high-accuracy PySR defaults for the first MERRA2 sample-table run.
    p.add_argument("--niterations", type=int, default=800, help="PySR iterations")
    p.add_argument("--populations", type=int, default=64, help="PySR populations")
    p.add_argument("--population-size", type=int, default=80, help="Candidate expressions per population")
    p.add_argument("--ncycles-per-iteration", type=int, default=1000, help="Cycles per PySR iteration")
    p.add_argument("--maxsize", type=int, default=35, help="PySR maximum expression size")
    p.add_argument("--maxdepth", type=int, default=8, help="PySR maximum expression-tree depth; 0 means unconstrained")
    p.add_argument(
        "--model-selection",
        choices=["accuracy", "best", "score"],
        default="best",
        help=(
            "Equation selected for prediction. 'best' is the compact/accurate compromise; "
            "use 'accuracy' when final error matters more than compactness."
        ),
    )
    p.add_argument(
        "--complexity-of-variables",
        type=float,
        default=0.20,
        help="Variable complexity; values < 1 make meteorological variables cheaper.",
    )
    p.add_argument("--complexity-of-constants", type=float, default=1.0)
    p.add_argument(
        "--parsimony",
        type=float,
        default=0.0,
        help="Extra complexity penalty. Keep 0 for high accuracy; maxsize/model-selection control compactness.",
    )
    p.add_argument("--warmup-maxsize-by", type=float, default=0.15)
    p.add_argument("--no-use-frequency", action="store_true", help="Disable PySR frequency-based exploration")
    p.add_argument("--batching", dest="batching", action="store_true", default=True, help="Enable PySR mini-batching; default on")
    p.add_argument("--no-batching", dest="batching", action="store_false", help="Disable PySR mini-batching")
    p.add_argument("--batch-size", type=int, default=16384, help="PySR mini-batch size")
    p.add_argument("--timeout-seconds", type=float, default=0.0)
    p.add_argument("--timeout-hours", type=float, default=0.0)
    p.add_argument("--timeout-slurm", action="store_true")
    p.add_argument("--timeout-buffer-minutes", type=float, default=10.0)
    p.add_argument(
        "--harvest-every-minutes",
        type=float,
        default=5.0,
        help=(
            "While PySR is running, copy the newest hall_of_fame.csv/checkpoint "
            "to stable top-level files every N minutes and write "
            "expression_global_latest.txt. Use 0 to disable. This protects long "
            "Slurm runs that are killed before PySR returns normally."
        ),
    )
    p.add_argument(
        "--harvest-selection",
        choices=["loss", "score", "last"],
        default="loss",
        help=(
            "How expression_global_latest.txt is chosen from hall_of_fame.csv during "
            "best-effort harvesting. 'loss' is most accurate; 'score' is often more compact; "
            "'last' follows the last Pareto row."
        ),
    )
    p.add_argument("--procs", type=int, default=0, help="Optional Julia worker processes; 0 leaves PySR default")
    p.add_argument(
        "--operator-set",
        choices=["basic", "protected", "extended"],
        default="protected",
        help=(
            "basic: + - * / square abs; protected: adds cube/sqrt_abs/log_abs; "
            "extended: adds guarded exp, trig, tanh, atan."
        ),
    )
    p.add_argument("--resume-from", default=None)
    p.add_argument("--resume-allow-mismatch", action="store_true")

    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--target-transform", choices=["none", "log", "log1p", "sqrt"], default="sqrt")
    p.add_argument("--log-eps", type=float, default=1.0e-8)
    p.add_argument("--y-min", type=float, default=0.0, help="Minimum valid Wstd")
    p.add_argument("--y-max", type=float, default=10.0, help="Maximum valid Wstd")
    p.add_argument(
        "--sample-weight-mode",
        choices=["none", "sqrt_y", "tail", "inverse_wstd_bin", "inverse_wstd_pressure_bin"],
        default="none",
        help=(
            "Optional training weights. 'tail' upweights high-Wstd cases; "
            "'inverse_*' approximately equalizes Wstd or Wstd x pressure bins."
        ),
    )
    p.add_argument(
        "--tail-weight-thresholds",
        default="0.3,1,3",
        help="Ascending Wstd thresholds for --sample-weight-mode tail.",
    )
    p.add_argument(
        "--tail-weight-additions",
        default="2,5,10",
        help="Additive weights for each threshold in --tail-weight-thresholds.",
    )
    p.add_argument(
        "--sqrt-weight-factor",
        type=float,
        default=2.0,
        help="Weight factor for --sample-weight-mode sqrt_y: w = 1 + factor*sqrt(Wstd).",
    )
    p.add_argument(
        "--max-sample-weight",
        type=float,
        default=25.0,
        help="Clip sample weights to this value before normalizing them to mean 1. Use <=0 to disable clipping.",
    )
    p.add_argument(
        "--no-rerank-equations",
        dest="rerank_equations",
        action="store_false",
        help="Disable validation re-ranking of the PySR hall-of-fame equations.",
    )
    p.set_defaults(rerank_equations=True)
    p.add_argument(
        "--rerank-high-threshold",
        type=float,
        default=1.0,
        help="High-Wstd threshold used in validation re-ranking score.",
    )
    p.add_argument(
        "--rerank-extreme-threshold",
        type=float,
        default=3.0,
        help="Extreme-Wstd threshold used in validation re-ranking score.",
    )
    p.add_argument(
        "--rerank-complexity-penalty",
        type=float,
        default=0.005,
        help="Complexity penalty coefficient in the validation re-ranking score.",
    )
    p.add_argument("--save-samples", action="store_true", help="Save the final sampled training pool to NPZ")

    return p.parse_args()


def _parse_slurm_time_left_to_seconds(value: str) -> Optional[float]:
    s = value.strip()
    if not s or s.upper() in {"UNLIMITED", "NOT_SET", "N/A", "INVALID", "TIME_LIMIT"}:
        return None
    days = 0
    if "-" in s:
        day_part, s = s.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = [int(float(x)) for x in parts]
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = [int(float(x)) for x in parts]
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = int(float(parts[0]))
        else:
            return None
    except ValueError:
        return None
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def detect_slurm_remaining_seconds() -> Optional[float]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", str(job_id), "-o", "%L"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip().splitlines()
    except Exception as e:
        warnings.warn(f"Could not query SLURM remaining walltime with squeue: {e}")
        return None
    if not out:
        return None
    return _parse_slurm_time_left_to_seconds(out[0])



def _argv_has_option(option_names: Sequence[str]) -> bool:
    """Return True if any option name was supplied as --flag or --flag=value."""
    argv = sys.argv[1:]
    for tok in argv:
        for opt in option_names:
            if tok == opt or tok.startswith(opt + "="):
                return True
    return False


def _set_if_not_overridden(args: argparse.Namespace, attr: str, value: object, option_names: Sequence[str]) -> None:
    """Set a preset value only when the user did not explicitly supply that option."""
    if not _argv_has_option(option_names):
        setattr(args, attr, value)


def _timeout_was_overridden() -> bool:
    return _argv_has_option(["--timeout-seconds", "--timeout-hours", "--timeout-slurm"])


def apply_run_preset(args: argparse.Namespace) -> None:
    """Apply quick/long presets while preserving explicit CLI overrides.

    The quick preset is now a real exploratory screen rather than a smoke test:
    fewer samples, many more populations/cycles, extended operators, and a
    2-hour timeout by default.  The long preset is intended to run until the
    Slurm walltime limit minus a safety buffer, emphasizing exploration and
    accuracy over compactness.
    """
    preset = getattr(args, "run_preset", "none")
    if preset in (None, "none"):
        return

    if preset == "quick":
        values = {
            # Fewer samples than the old quick preset so PySR spends the short
            # walltime mutating/evolving expressions rather than only evaluating
            # a huge batch.  This is meant to produce a meaningful Pareto front.
            "features": "compact",
            "target_transform": "sqrt",
            "sampling_mode": "wstd_pressure_stratified",
            "sample_weight_mode": "sqrt_y",
            "rerank_equations": True,
            "max_total_samples": 30_000,
            "candidate_batch_size": 100_000,

            # Timeout-driven search: niterations is deliberately unreachable.
            "niterations": 1_000_000,
            "populations": 48,
            "population_size": 100,
            "ncycles_per_iteration": 300,
            "maxsize": 45,
            "maxdepth": 9,
            "model_selection": "accuracy",
            "operator_set": "extended",
            "complexity_of_variables": 0.05,
            "warmup_maxsize_by": 0.0,
            "batching": True,
            "batch_size": 4096,
            "harvest_every_minutes": 0.5,
            "harvest_selection": "loss",
        }
        default_timeout_seconds = 2 * 3600
        default_timeout_buffer_minutes = 5.0

    elif preset == "long":
        values = {
            # Deep final search: keep a large training sample, but use batching
            # so expression evolution remains feasible.  This run should be
            # launched as a separate job from quick and reported separately.
            "features": "compact",
            "target_transform": "sqrt",
            "sampling_mode": "wstd_pressure_stratified",
            "sample_weight_mode": "sqrt_y",
            "rerank_equations": True,
            "max_total_samples": 1_000_000,
            "candidate_batch_size": 750_000,
            "niterations": 1_000_000,
            "populations": 128,
            "population_size": 120,
            "ncycles_per_iteration": 1_000,
            "maxsize": 80,
            "maxdepth": 12,
            "model_selection": "accuracy",
            "operator_set": "extended",
            "complexity_of_variables": 0.05,
            "warmup_maxsize_by": 0.0,
            "batching": True,
            "batch_size": 16384,
            "harvest_every_minutes": 2.0,
            "harvest_selection": "loss",
        }
        default_timeout_seconds = None
        default_timeout_buffer_minutes = 60.0

    else:
        raise ValueError(f"Unknown run preset: {preset}. Allowed presets are: none, quick, long")

    option_map = {
        "features": ["--features"],
        "target_transform": ["--target-transform"],
        "sampling_mode": ["--sampling-mode"],
        "sample_weight_mode": ["--sample-weight-mode"],
        "rerank_equations": ["--no-rerank-equations"],
        "max_total_samples": ["--max-total-samples"],
        "candidate_batch_size": ["--candidate-batch-size"],
        "niterations": ["--niterations"],
        "populations": ["--populations"],
        "population_size": ["--population-size"],
        "ncycles_per_iteration": ["--ncycles-per-iteration"],
        "maxsize": ["--maxsize"],
        "maxdepth": ["--maxdepth"],
        "model_selection": ["--model-selection"],
        "operator_set": ["--operator-set"],
        "complexity_of_variables": ["--complexity-of-variables"],
        "warmup_maxsize_by": ["--warmup-maxsize-by"],
        "batching": ["--batching", "--no-batching"],
        "batch_size": ["--batch-size"],
        "harvest_every_minutes": ["--harvest-every-minutes"],
        "harvest_selection": ["--harvest-selection"],
    }
    for attr, value in values.items():
        _set_if_not_overridden(args, attr, value, option_map[attr])

    if not _argv_has_option(["--timeout-buffer-minutes"]):
        args.timeout_buffer_minutes = default_timeout_buffer_minutes

    if not _timeout_was_overridden():
        if preset == "quick":
            args.timeout_seconds = float(default_timeout_seconds)
        elif preset == "long":
            # In a Slurm allocation, make the Python/PySR timeout follow the
            # scheduler walltime. Outside Slurm, use 24 hours so a non-Slurm
            # test behaves like a deep run instead of a short screen.
            if os.environ.get("SLURM_JOB_ID"):
                args.timeout_slurm = True
            else:
                args.timeout_hours = 24.0
                warnings.warn(
                    "--run-preset long was used outside a detected Slurm job; "
                    "defaulting to --timeout-hours 24. Use --timeout-slurm inside Slurm "
                    "or pass an explicit --timeout-hours/--timeout-seconds."
                )

def print_resolved_run_config(args: argparse.Namespace) -> None:
    rows = [
        ("run_preset", args.run_preset),
        ("features", args.features),
        ("max_total_samples", args.max_total_samples),
        ("candidate_batch_size", args.candidate_batch_size),
        ("sampling_mode", args.sampling_mode),
        ("wstd_sampling_edges", getattr(args, "wstd_sampling_edges_resolved", args.wstd_sampling_edges)),
        ("pressure_sampling_edges", getattr(args, "pressure_sampling_edges_resolved", args.pressure_sampling_edges)),
        ("target_transform", args.target_transform),
        ("sample_weight_mode", args.sample_weight_mode),
        ("rerank_equations", getattr(args, "rerank_equations", True)),
        ("standardization", args.standardization),
        ("niterations", args.niterations),
        ("populations", args.populations),
        ("population_size", args.population_size),
        ("ncycles_per_iteration", args.ncycles_per_iteration),
        ("maxsize", args.maxsize),
        ("maxdepth", args.maxdepth),
        ("model_selection", args.model_selection),
        ("operator_set", args.operator_set),
        ("complexity_of_variables", args.complexity_of_variables),
        ("warmup_maxsize_by", args.warmup_maxsize_by),
        ("batching", args.batching),
        ("batch_size", args.batch_size),
        ("timeout_seconds", args.timeout_seconds),
        ("timeout_hours", args.timeout_hours),
        ("timeout_slurm", args.timeout_slurm),
        ("timeout_buffer_minutes", args.timeout_buffer_minutes),
        ("resolved_timeout_in_seconds", getattr(args, "timeout_in_seconds", None)),
        ("harvest_every_minutes", args.harvest_every_minutes),
        ("harvest_selection", args.harvest_selection),
        ("pressure_bracket_edges", getattr(args, "pressure_bracket_edges_resolved", args.pressure_bracket_edges)),
        ("wstd_bracket_edges", getattr(args, "wstd_bracket_edges_resolved", args.wstd_bracket_edges)),
        ("outdir", args.outdir),
    ]
    width = max(len(k) for k, _ in rows)
    print("Resolved run configuration:")
    for k, v in rows:
        print(f"  {k:<{width}} : {v}")


def resolve_timeout_in_seconds(args: argparse.Namespace) -> float:
    timeout_seconds = float(args.timeout_seconds or 0.0)
    timeout_hours = float(args.timeout_hours or 0.0)
    if timeout_seconds < 0.0 or timeout_hours < 0.0:
        raise ValueError("Timeout values must be >= 0")
    if float(args.timeout_buffer_minutes) < 0.0:
        raise ValueError("--timeout-buffer-minutes must be >= 0")
    explicit_count = int(timeout_seconds > 0.0) + int(timeout_hours > 0.0) + int(bool(args.timeout_slurm))
    if explicit_count > 1:
        raise ValueError("Use only one of --timeout-seconds, --timeout-hours, or --timeout-slurm")
    if timeout_seconds > 0.0:
        return timeout_seconds
    if timeout_hours > 0.0:
        return timeout_hours * 3600.0
    if args.timeout_slurm:
        remaining = detect_slurm_remaining_seconds()
        if remaining is None:
            raise RuntimeError("--timeout-slurm requested, but SLURM remaining walltime could not be detected")
        usable = remaining - float(args.timeout_buffer_minutes) * 60.0
        if usable <= 0.0:
            raise RuntimeError("Detected SLURM remaining time is not larger than the requested buffer")
        return usable
    return 0.0


def print_feature_catalog() -> None:
    print("Base table features:")
    for f in BASE_FEATURES:
        print(f"  {f}")
    print("\nDerived features:")
    for f in DERIVED_FEATURES:
        print(f"  {f}: {DERIVED_FEATURE_DESCRIPTIONS[f]}")
    print("\nDefault compact set:")
    print("  " + ",".join(COMPACT_FEATURES))


def feature_names_from_args(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    alias: Dict[str, str] = {name: name for name in ALL_FEATURES}
    alias.update({
        "Q": "QV",
        "BLH": "PBLH",
        "TCWV": "TQV",
        "QTCWV": "QVTQV",
        "PBLH-H": "PBLH_MINUS_H",
        "PBLH_H": "PBLH_MINUS_H",
        "PBLHmH": "PBLH_MINUS_H",
        "PBLHminusH": "PBLH_MINUS_H",
        "PBLH_MINUS_HEIGHT": "PBLH_MINUS_H",
    })

    spec = args.features.strip()
    if spec.lower() == "all":
        selected = list(ALL_FEATURES)
    elif spec.lower() == "compact":
        selected = list(COMPACT_FEATURES)
    else:
        selected = []
        bad = []
        for raw in [s.strip() for s in spec.split(",") if s.strip()]:
            key = raw
            val = alias.get(key)
            if val is None:
                # try case-insensitive lookup while preserving canonical names
                upper_map = {k.upper(): v for k, v in alias.items()}
                val = upper_map.get(key.upper())
            if val is None:
                bad.append(raw)
            elif val not in selected:
                selected.append(val)
        if bad:
            raise ValueError(
                "Unknown feature(s): " + ", ".join(bad) +
                "\nAllowed: " + ", ".join(ALL_FEATURES) +
                "\nAliases: Q->QV, BLH->PBLH, TCWV->TQV, QTCWV->QVTQV, PBLH-H->PBLH_MINUS_H"
            )
        if not selected:
            raise ValueError("No valid features selected")

    symbolic = [SYMBOLIC_NAME[f] for f in selected]
    return selected, symbolic


def required_table_vars_for_features(features: Sequence[str]) -> List[str]:
    req: Set[str] = set()
    for f in features:
        if f in BASE_FEATURES:
            req.add(f)
        elif f == "WIND":
            req.update(["U", "V"])
        elif f == "WABS":
            req.add("W")
        elif f == "RH":
            req.update(["QV", "T", "P"])
        elif f == "QVTQV":
            req.update(["QV", "TQV"])
        elif f == "RHO_DIFF":
            req.update(["AIRD_surf", "AIRD"])
        elif f == "PBLH_MINUS_H":
            req.update(["PBLH", "H"])
        elif f == "LOGP":
            req.add("P")
        elif f == "SIGMA":
            req.update(["P", "PS"])
        elif f == "ZPBL":
            req.update(["H", "PBLH"])
        elif f == "ABOVE_PBL":
            req.update(["H", "PBLH"])
        elif f == "LOG_TQV":
            req.add("TQV")
        elif f == "LOG_QV":
            req.add("QV")
        elif f == "LOG_QI":
            req.add("QI")
        elif f == "LOG_QL":
            req.add("QL")
        else:
            raise ValueError(f"Unhandled feature: {f}")
    return sorted(req)


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


class SampleTableReader:
    """Small reader abstraction for efficient random reads from 1-D sample tables."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._mode = "netcdf4" if nc4 is not None else "xarray"
        self._nc = None
        self._xr = None
        if self._mode == "netcdf4":
            self._nc = nc4.Dataset(str(self.path), "r")
            self.var_names = set(self._nc.variables.keys())
            self.attrs = {name: self._nc.getncattr(name) for name in self._nc.ncattrs()}
            self.n_samples = int(self._nc.dimensions["sample"].size)
        else:
            self._xr = xr.open_dataset(self.path, decode_times=False, mask_and_scale=True)
            self.var_names = set(self._xr.data_vars.keys()) | set(self._xr.coords.keys())
            self.attrs = dict(self._xr.attrs)
            self.n_samples = int(self._xr.sizes["sample"])

    def close(self) -> None:
        if self._nc is not None:
            self._nc.close()
        if self._xr is not None:
            self._xr.close()

    def read(self, name: str, idx: np.ndarray) -> np.ndarray:
        if name not in self.var_names:
            raise KeyError(f"Variable {name!r} not found in {self.path}; available: {sorted(self.var_names)}")
        if self._mode == "netcdf4":
            assert self._nc is not None
            arr = self._nc.variables[name][idx]
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)
            arr = np.asarray(arr)
        else:
            assert self._xr is not None
            arr = self._xr[name].isel(sample=xr.DataArray(idx, dims="point")).values
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)
            arr = np.asarray(arr)
        return arr


def parse_days_list(value: Optional[str]) -> Optional[Set[int]]:
    if value is None or not str(value).strip():
        return None
    days = set()
    for token in re.split(r"[,\s]+", str(value).strip()):
        if not token:
            continue
        if not re.match(r"^\d{8}$", token):
            raise ValueError(f"Invalid YYYYMMDD date in --days: {token}")
        days.add(int(token))
    return days


def parse_selected_days_attr(attrs: Dict[str, object]) -> List[int]:
    raw = attrs.get("selected_days", "")
    if raw is None:
        return []
    days = []
    for token in str(raw).split(","):
        token = token.strip()
        if re.match(r"^\d{8}$", token):
            days.append(int(token))
    return days


def choose_day_filter(args: argparse.Namespace, attrs: Dict[str, object], rng: np.random.Generator) -> Optional[Set[int]]:
    explicit = parse_days_list(args.days)
    if explicit is not None:
        return explicit
    if args.max_days <= 0:
        return None
    days = parse_selected_days_attr(attrs)
    if not days:
        raise ValueError(
            "--max-days was requested but the file has no parseable selected_days global attribute. "
            "Use --days YYYYMMDD,... instead."
        )
    unique_days = sorted(set(days))
    if args.max_days >= len(unique_days):
        return set(unique_days)
    if args.day_selection == "first":
        chosen = unique_days[: args.max_days]
    else:
        idx = np.sort(rng.choice(len(unique_days), size=args.max_days, replace=False))
        chosen = [unique_days[int(i)] for i in idx]
    return set(chosen)


def parse_levels_hpa(value: str) -> Optional[np.ndarray]:
    if value.strip().lower() == "all":
        return None
    vals = [float(s) for s in value.split(",") if s.strip()]
    if not vals:
        raise ValueError("--levels-hpa was provided but no valid levels were parsed")
    return np.asarray(vals, dtype=np.float32)


def parse_optional_metric_edges(value: object, *, descending: bool, name: str) -> Optional[Tuple[float, ...]]:
    """Parse metric-bin edges, or return None when disabled.

    Accepts plain CLI strings, lists/tuples, and stringified lists.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [float(v) for v in value]
    else:
        text = str(value).strip()
        if text.lower() in {"", "none", "off", "disable", "disabled", "no", "false"}:
            return None
        text = text.strip("[]()")
        vals = [float(s) for s in re.split(r"[,\s]+", text) if s.strip()]
    if len(vals) < 2:
        raise ValueError(f"--{name} needs at least two edges, or use 'none' to disable it")
    if descending:
        bad = any(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        if bad:
            raise ValueError(f"--{name} must be strictly descending, e.g. 1000,800,500,200,10")
    else:
        bad = any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        if bad:
            raise ValueError(f"--{name} must be strictly ascending, e.g. 0.1,1,10")
    return tuple(float(v) for v in vals)


def mask_requested_levels(lev: np.ndarray, requested: Optional[np.ndarray], tol: float) -> np.ndarray:
    if requested is None:
        return np.ones(len(lev), dtype=bool)
    d = np.abs(lev.astype(np.float32)[:, None] - requested[None, :])
    return np.min(d, axis=1) <= float(tol)


def valid_numeric_mask(arr: np.ndarray, name: str) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        return np.ones(arr.shape, dtype=bool)
    aa = arr.astype(np.float64, copy=False)
    m = np.isfinite(aa)
    # Training table fill value is -9999.f; avoid killing valid negative winds by
    # only treating values near that sentinel as missing.
    m &= aa > -FILL_ABS_LIMIT
    return m


def pressure_to_hpa_np(arr: np.ndarray) -> np.ndarray:
    """Return pressure in hPa, auto-detecting Pa vs hPa.

    MERRA2 PS is commonly in Pa, while this training table documents P in hPa.
    This helper makes the below-surface filter robust if either field was stored
    in Pa. Values with a typical magnitude above 2000 are interpreted as Pa.
    """
    aa = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(aa) & (np.abs(aa) < 1.0e7) & (aa > -FILL_ABS_LIMIT)
    if not np.any(finite):
        return aa.astype(np.float32)
    scale = 0.01 if float(np.nanmedian(np.abs(aa[finite]))) > 2000.0 else 1.0
    return (aa * scale).astype(np.float32)


def parse_float_sequence(value: object, *, name: str, min_len: int = 1) -> Tuple[float, ...]:
    if value is None:
        raise ValueError(f"--{name} is required")
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [float(v) for v in value]
    else:
        text = str(value).strip().strip("[]()")
        vals = [float(s) for s in re.split(r"[,\s]+", text) if s.strip()]
    if len(vals) < min_len:
        raise ValueError(f"--{name} needs at least {min_len} value(s)")
    return tuple(float(v) for v in vals)


def assign_wstd_bins(y: np.ndarray, edges: Sequence[float]) -> Tuple[np.ndarray, List[str]]:
    e = np.asarray(edges, dtype=float)
    if len(e) < 2 or np.any(np.diff(e) <= 0):
        raise ValueError("Wstd edges must be strictly ascending")
    yy = np.asarray(y, dtype=float)
    ids = np.full(yy.shape, -1, dtype=np.int32)
    labels: List[str] = []
    for i in range(len(e) - 1):
        lo, hi = e[i], e[i + 1]
        if i == len(e) - 2:
            m = np.isfinite(yy) & (yy >= lo) & (yy <= hi)
        else:
            m = np.isfinite(yy) & (yy >= lo) & (yy < hi)
        ids[m] = i
        labels.append(f"Wstd_{lo:g}_to_{hi:g}")
    return ids, labels


def assign_pressure_bins(p_hpa: np.ndarray, edges: Sequence[float]) -> Tuple[np.ndarray, List[str]]:
    e = np.asarray(edges, dtype=float)
    if len(e) < 2 or np.any(np.diff(e) >= 0):
        raise ValueError("Pressure edges must be strictly descending")
    pp = np.asarray(p_hpa, dtype=float)
    ids = np.full(pp.shape, -1, dtype=np.int32)
    labels: List[str] = []
    for i in range(len(e) - 1):
        upper, lower = e[i], e[i + 1]
        if i == len(e) - 2:
            m = np.isfinite(pp) & (pp <= upper) & (pp >= lower)
        else:
            m = np.isfinite(pp) & (pp <= upper) & (pp > lower)
        ids[m] = i
        labels.append(f"P_{lower:g}_to_{upper:g}_hPa")
    return ids, labels


def assign_sampling_strata(
    y: np.ndarray,
    lev_hpa: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[int, str]]:
    mode = getattr(args, "sampling_mode", "random")
    if mode == "random":
        ids = np.zeros(len(y), dtype=np.int32)
        return ids, {0: "all_random"}

    w_edges = getattr(args, "wstd_sampling_edges_resolved", None)
    if w_edges is None:
        w_edges = parse_optional_metric_edges(args.wstd_sampling_edges, descending=False, name="wstd-sampling-edges")
    if w_edges is None:
        raise ValueError("Stratified sampling requires --wstd-sampling-edges")
    w_id, w_labels = assign_wstd_bins(y, w_edges)

    if mode == "wstd_stratified":
        labels = {i: lab for i, lab in enumerate(w_labels)}
        return w_id, labels

    if mode == "wstd_pressure_stratified":
        p_edges = getattr(args, "pressure_sampling_edges_resolved", None)
        if p_edges is None:
            p_edges = parse_optional_metric_edges(args.pressure_sampling_edges, descending=True, name="pressure-sampling-edges")
        if p_edges is None:
            raise ValueError("Wstd x pressure stratified sampling requires --pressure-sampling-edges")
        p_id, p_labels = assign_pressure_bins(lev_hpa, p_edges)
        n_p = len(p_labels)
        ids = np.full(len(y), -1, dtype=np.int32)
        valid = (w_id >= 0) & (p_id >= 0)
        ids[valid] = w_id[valid] * n_p + p_id[valid]
        labels = {wi * n_p + pi: f"{w_labels[wi]}__{p_labels[pi]}" for wi in range(len(w_labels)) for pi in range(n_p)}
        return ids, labels

    raise ValueError(f"Unhandled sampling mode: {mode}")


def choose_stratified_indices(
    yv: np.ndarray,
    levv: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    strata_counts: Dict[int, int],
    target_per_stratum: int,
    attempts: int,
    max_attempts: int,
    remaining: int,
) -> Tuple[np.ndarray, Dict[int, str]]:
    strata_id, labels = assign_sampling_strata(yv, levv, args)
    valid = np.flatnonzero(strata_id >= 0)
    if len(valid) == 0:
        return np.asarray([], dtype=np.int64), labels

    selected_chunks: List[np.ndarray] = []
    unique_ids = np.unique(strata_id[valid])
    unique_ids = rng.permutation(unique_ids)
    for sid in unique_ids:
        need = max(0, int(target_per_stratum) - int(strata_counts.get(int(sid), 0)))
        if need <= 0:
            continue
        cand = np.flatnonzero(strata_id == sid)
        if len(cand) == 0:
            continue
        take_n = min(need, len(cand), max(0, remaining - sum(len(x) for x in selected_chunks)))
        if take_n <= 0:
            break
        selected_chunks.append(rng.choice(cand, size=take_n, replace=False))

    selected = np.concatenate(selected_chunks) if selected_chunks else np.asarray([], dtype=np.int64)

    # Late in sampling, relax the strict per-stratum cap so sparse/empty strata
    # cannot leave the run badly under-filled.  This keeps the new behavior robust
    # on partially populated pressure/Wstd combinations.
    relax_frac = float(getattr(args, "stratified_relax_frac", 0.70))
    if len(selected) < remaining and max_attempts > 0 and attempts >= max(1, int(relax_frac * max_attempts)):
        already = set(int(i) for i in selected.tolist())
        extra_pool = np.asarray([int(i) for i in valid.tolist() if int(i) not in already], dtype=np.int64)
        extra_need = min(remaining - len(selected), len(extra_pool))
        if extra_need > 0:
            extra = rng.choice(extra_pool, size=extra_need, replace=False)
            selected = np.concatenate([selected, extra]) if len(selected) else extra

    if len(selected) > remaining:
        selected = rng.choice(selected, size=remaining, replace=False)
    return selected.astype(np.int64, copy=False), labels


def compute_feature_matrix(raw: Dict[str, np.ndarray], selected_features: Sequence[str]) -> np.ndarray:
    cols: List[np.ndarray] = []
    for f in selected_features:
        if f in BASE_FEATURES:
            arr = raw[f]
        elif f == "WIND":
            arr = np.sqrt(raw["U"] * raw["U"] + raw["V"] * raw["V"])
        elif f == "WABS":
            arr = np.abs(raw["W"])
        elif f == "RH":
            qsat = qsat_liquid_specific_humidity_np(raw["T"], raw["P"])
            arr = raw["QV"] / np.where(qsat > 1.0e-12, qsat, np.nan)
            arr = np.clip(arr, 0.0, 2.0)
        elif f == "QVTQV":
            arr = raw["QV"] * raw["TQV"]
        elif f == "RHO_DIFF":
            arr = raw["AIRD_surf"] - raw["AIRD"]
        elif f == "PBLH_MINUS_H":
            arr = raw["PBLH"] - raw["H"]
        elif f == "LOGP":
            p_hpa = pressure_to_hpa_np(raw["P"])
            arr = np.log(np.maximum(p_hpa, 1.0e-6))
        elif f == "SIGMA":
            p_hpa = pressure_to_hpa_np(raw["P"])
            ps_hpa = pressure_to_hpa_np(raw["PS"])
            arr = p_hpa / np.where(ps_hpa > 1.0e-6, ps_hpa, np.nan)
        elif f == "ZPBL":
            arr = raw["H"] / np.maximum(raw["PBLH"] + 1.0, 1.0)
        elif f == "ABOVE_PBL":
            arr = raw["H"] - raw["PBLH"]
        elif f == "LOG_TQV":
            arr = np.log1p(np.maximum(raw["TQV"], 0.0))
        elif f == "LOG_QV":
            arr = np.log(np.maximum(raw["QV"], 0.0) + 1.0e-12)
        elif f == "LOG_QI":
            arr = np.log(np.maximum(raw["QI"], 0.0) + 1.0e-12)
        elif f == "LOG_QL":
            arr = np.log(np.maximum(raw["QL"], 0.0) + 1.0e-12)
        else:
            raise ValueError(f"Unhandled feature: {f}")
        cols.append(np.asarray(arr, dtype=np.float32))
    if not cols:
        raise ValueError("No feature columns computed")
    return np.stack(cols, axis=1).astype(np.float32, copy=False)


def sample_training_table(
    input_file: Path,
    args: argparse.Namespace,
    selected_features: Sequence[str],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    reader = SampleTableReader(input_file)
    try:
        required = required_table_vars_for_features(selected_features)
        # Always require P and PS unless explicitly disabled, so we can reject
        # samples below the surface even when P/PS are not selected as predictors.
        pressure_filter_vars = [] if args.allow_p_greater_than_ps else ["P", "PS"]
        for req in sorted(set(required + ["Wstd", "lev"] + pressure_filter_vars)):
            if req not in reader.var_names:
                raise KeyError(f"Required variable {req!r} not found in {input_file}")

        requested_levels = parse_levels_hpa(args.levels_hpa)
        day_filter = choose_day_filter(args, reader.attrs, rng)
        need_y_min = args.y_min
        if args.target_transform == "log":
            need_y_min = max(need_y_min, args.log_eps)

        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        lev_parts: List[np.ndarray] = []
        strata_counts: Dict[int, int] = {}
        strata_labels: Dict[int, str] = {}
        sampling_mode = str(getattr(args, "sampling_mode", "random"))
        if sampling_mode == "random":
            target_per_stratum = int(args.max_total_samples)
            requested_n_strata = 1
        else:
            w_edges_tmp = getattr(args, "wstd_sampling_edges_resolved", None)
            if w_edges_tmp is None:
                w_edges_tmp = parse_optional_metric_edges(args.wstd_sampling_edges, descending=False, name="wstd-sampling-edges")
            requested_n_strata = max(1, len(w_edges_tmp) - 1)
            if sampling_mode == "wstd_pressure_stratified":
                p_edges_tmp = getattr(args, "pressure_sampling_edges_resolved", None)
                if p_edges_tmp is None:
                    p_edges_tmp = parse_optional_metric_edges(args.pressure_sampling_edges, descending=True, name="pressure-sampling-edges")
                requested_n_strata *= max(1, len(p_edges_tmp) - 1)
            target_per_stratum = int(math.ceil(int(args.max_total_samples) / float(requested_n_strata)))
        n_pressure_checked = 0
        n_rejected_p_gt_ps = 0
        have = 0
        attempts = 0
        n_total = int(reader.n_samples)
        want = int(args.max_total_samples)
        if want <= 0:
            raise ValueError("--max-total-samples must be > 0 for the MERRA2 sample-table workflow")

        print(f"Input sample rows: {n_total}")
        print(f"Requested valid rows: {want}")
        print(f"Sampling mode: {sampling_mode}; requested_strata={requested_n_strata}; target_per_stratum={target_per_stratum}")
        if requested_levels is not None:
            print("Filtering levels hPa: " + ", ".join(f"{x:g}" for x in requested_levels))
        if day_filter is not None:
            print(f"Filtering dates: {len(day_filter)} selected YYYYMMDD values")

        while have < want and attempts < int(args.max_sampling_attempts):
            attempts += 1
            remaining = want - have
            n_try = int(max(args.candidate_batch_size, math.ceil(remaining * float(args.oversample))))
            n_try = min(n_try, n_total)
            idx = np.unique(rng.integers(0, n_total, size=n_try, dtype=np.int64))
            idx.sort()

            y = np.asarray(reader.read("Wstd", idx), dtype=np.float32)
            lev = np.asarray(reader.read("lev", idx), dtype=np.float32)

            mask = valid_numeric_mask(y, "Wstd")
            mask &= y > need_y_min
            mask &= y < float(args.y_max)
            mask &= valid_numeric_mask(lev, "lev")
            mask &= mask_requested_levels(lev, requested_levels, args.level_tolerance_hpa)

            if day_filter is not None:
                if "yyyymmdd" not in reader.var_names:
                    raise KeyError("Date filtering requested, but yyyymmdd is not present in the table")
                days = np.asarray(reader.read("yyyymmdd", idx)).astype(np.int64)
                mask &= np.isin(days, np.fromiter(day_filter, dtype=np.int64))

            if not np.any(mask):
                print(f"  attempt {attempts:02d}: no valid candidates from {len(idx)} rows")
                continue

            raw: Dict[str, np.ndarray] = {}
            vars_to_read = sorted(set(required + pressure_filter_vars))
            for v in vars_to_read:
                raw[v] = np.asarray(reader.read(v, idx), dtype=np.float32)
                mask &= valid_numeric_mask(raw[v], v)

            if not args.allow_p_greater_than_ps:
                p_hpa = pressure_to_hpa_np(raw["P"])
                ps_hpa = pressure_to_hpa_np(raw["PS"])
                pressure_ok = (
                    valid_numeric_mask(p_hpa, "P_hpa")
                    & valid_numeric_mask(ps_hpa, "PS_hpa")
                    & np.isfinite(p_hpa)
                    & np.isfinite(ps_hpa)
                    & (p_hpa > 0.0)
                    & (ps_hpa > 0.0)
                )
                currently_candidate = mask.copy()
                below_surface = pressure_ok & (p_hpa > ps_hpa + float(args.ps_tolerance_hpa))
                n_pressure_checked += int(np.count_nonzero(currently_candidate & pressure_ok))
                n_rejected_p_gt_ps += int(np.count_nonzero(currently_candidate & below_surface))
                mask &= pressure_ok & (~below_surface)

            if not np.any(mask):
                print(f"  attempt {attempts:02d}: candidates failed feature-validity/P<=PS filtering")
                continue

            X = compute_feature_matrix(raw, selected_features)
            mask &= np.isfinite(X).all(axis=1)

            Xv = X[mask]
            yv = y[mask]
            levv = lev[mask]
            if len(yv) == 0:
                print(f"  attempt {attempts:02d}: no rows after derived-feature filtering")
                continue

            if sampling_mode == "random":
                if len(yv) > remaining:
                    take = rng.choice(len(yv), size=remaining, replace=False)
                    Xv = Xv[take]
                    yv = yv[take]
                    levv = levv[take]
            else:
                take, labels = choose_stratified_indices(
                    yv,
                    levv,
                    args,
                    rng,
                    strata_counts,
                    target_per_stratum,
                    attempts,
                    int(args.max_sampling_attempts),
                    remaining,
                )
                strata_labels.update(labels)
                if len(take) == 0:
                    print(f"  attempt {attempts:02d}: no needed stratified rows from {len(idx)} candidates")
                    continue
                Xv = Xv[take]
                yv = yv[take]
                levv = levv[take]
                sid_take, labels_take = assign_sampling_strata(yv, levv, args)
                strata_labels.update(labels_take)
                for sid in sid_take:
                    if sid >= 0:
                        strata_counts[int(sid)] = strata_counts.get(int(sid), 0) + 1

            X_parts.append(Xv.astype(np.float32, copy=False))
            y_parts.append(yv.astype(np.float32, copy=False))
            lev_parts.append(levv.astype(np.float32, copy=False))
            have += len(yv)
            if sampling_mode == "random":
                extra = ""
            else:
                filled = sum(1 for v in strata_counts.values() if v > 0)
                extra = f"; active_strata={filled}/{requested_n_strata}"
            print(
                f"  attempt {attempts:02d}: accepted {len(yv)} rows "
                f"({have}/{want}) from {len(idx)} candidates" + extra
            )

        if not X_parts:
            raise RuntimeError("No usable samples were collected from the table")

        X_all = np.concatenate(X_parts, axis=0)[:want]
        y_all = np.concatenate(y_parts, axis=0)[:want]
        lev_all = np.concatenate(lev_parts, axis=0)[:want]

        order = rng.permutation(len(y_all))
        X_all = X_all[order]
        y_all = y_all[order]
        lev_all = lev_all[order]

        sample_info = {
            "input_file": str(input_file),
            "input_n_samples": int(n_total),
            "requested_valid_samples": int(want),
            "collected_valid_samples": int(len(y_all)),
            "sampling_attempts": int(attempts),
            "sampling_mode": sampling_mode,
            "sampling_requested_strata": int(requested_n_strata),
            "sampling_target_per_stratum": int(target_per_stratum),
            "sampling_strata_counts": {strata_labels.get(int(k), str(k)): int(v) for k, v in sorted(strata_counts.items())},
            "levels_hpa_sampled": [float(z) for z in sorted(np.unique(lev_all.astype(float)))],
            "levels_hpa_filter": "all" if requested_levels is None else [float(x) for x in requested_levels],
            "level_tolerance_hpa": float(args.level_tolerance_hpa),
            "p_gt_ps_filter_enabled": bool(not args.allow_p_greater_than_ps),
            "ps_tolerance_hpa": float(args.ps_tolerance_hpa),
            "pressure_checked_candidate_rows": int(n_pressure_checked),
            "rejected_p_gt_ps_candidate_rows": int(n_rejected_p_gt_ps),
            "days_filter_count": None if day_filter is None else int(len(day_filter)),
            "days_filter": None if day_filter is None else sorted(int(x) for x in day_filter),
            "global_attributes": {str(k): str(v) for k, v in reader.attrs.items() if k != "selected_days"},
        }
        if "selected_days" in reader.attrs:
            sample_info["selected_days_attribute_count"] = len(parse_selected_days_attr(reader.attrs))
        return X_all, y_all, lev_all, sample_info
    finally:
        reader.close()


def standardize_sampled_features(
    X: np.ndarray,
    selected_features: List[str],
    mode: str,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float | str]]]:
    info: Dict[str, Dict[str, float | str]] = {}
    if mode == "none":
        for name in selected_features:
            info[name] = {"mean": 0.0, "std": 1.0, "method": "none_physical_units"}
        return X.astype(np.float32, copy=False), info

    Xs = X.astype(np.float32, copy=True)
    for j, name in enumerate(selected_features):
        use_original = mode == "original" and name in ORIGINAL_MEANS and name in ORIGINAL_STDS
        if use_original:
            mean = float(ORIGINAL_MEANS[name])
            std = float(ORIGINAL_STDS[name])
            method = "fixed_original_NN_preprocessing"
        else:
            col = np.asarray(Xs[:, j], dtype=np.float64)
            finite = np.isfinite(col)
            if not np.any(finite):
                raise RuntimeError(f"Cannot standardize feature {name}: no finite sampled values")
            mean = float(np.mean(col[finite]))
            std = float(np.std(col[finite]))
            if not np.isfinite(std) or std <= 0.0:
                warnings.warn(f"Feature {name} has near-zero sampled std; using std=1")
                std = 1.0
            method = "sample_pool_dynamic"
        Xs[:, j] = (Xs[:, j] - mean) / std
        info[name] = {"mean": mean, "std": std, "method": method}
    return Xs, info


def transform_target(y: np.ndarray, mode: str, log_eps: float = 1.0e-8) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "log":
        return np.log(np.maximum(y, log_eps))
    if mode == "log1p":
        return np.log1p(np.maximum(y, 0.0))
    if mode == "sqrt":
        return np.sqrt(np.maximum(y, 0.0))
            
    raise ValueError(mode)


def inverse_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "log":
        return np.exp(y)
    if mode == "log1p":
        return np.expm1(y)
    if mode == "sqrt":
        return np.square(np.maximum(y, 0.0))
    raise ValueError(mode)


def lhs_name(target_transform: Optional[str]) -> str:
    if target_transform == "log":
        return "log(Wstd)"
    if target_transform == "log1p":
        return "log1p(Wstd)"
    if target_transform == "sqrt":
        return "sqrt(Wstd)"
    return "Wstd"


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return {"n": 0, "rmse": math.nan, "mae": math.nan, "bias": math.nan, "corr": math.nan, "r2": math.nan}
    yt = y_true[m]
    yp = y_pred[m]
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    bias = float(np.mean(yp - yt))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 and np.std(yt) > 0 and np.std(yp) > 0 else math.nan
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    return {"n": int(m.sum()), "rmse": rmse, "mae": mae, "bias": bias, "corr": corr, "r2": r2}


def metrics_by_level(y_true: np.ndarray, y_pred: np.ndarray, lev: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for z in sorted(np.unique(lev.astype(float))):
        m = lev == z
        out[f"{z:g}"] = calc_metrics(y_true[m], y_pred[m])
    return out


def metrics_by_pressure_bracket(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_hpa: np.ndarray,
    edges: Optional[Sequence[float]] = (1000.0, 800.0, 500.0, 200.0, 10.0),
) -> Dict[str, Dict[str, float]]:
    """Metrics in descending pressure-layer brackets.

    The default bins are 800<p<=1000, 500<p<=800, 200<p<=500,
    and 10<=p<=200 hPa.
    """
    if edges is None:
        return {}
    p = np.asarray(p_hpa, dtype=float)
    e = [float(x) for x in edges]
    if len(e) < 2 or any(e[i] <= e[i + 1] for i in range(len(e) - 1)):
        raise ValueError("Pressure bracket edges must be strictly descending, e.g. 1000,800,500,200,10")
    out: Dict[str, Dict[str, float]] = {}
    for i in range(len(e) - 1):
        upper = e[i]
        lower = e[i + 1]
        if i == len(e) - 2:
            m = np.isfinite(p) & (p <= upper) & (p >= lower)
        else:
            m = np.isfinite(p) & (p <= upper) & (p > lower)
        out[f"{lower:g}_to_{upper:g}_hPa"] = calc_metrics(y_true[m], y_pred[m])
    return out


def metrics_by_wstd_bracket(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edges: Optional[Sequence[float]] = (0.1, 1.0, 10.0),
    include_outside: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Metrics stratified by true physical Wstd brackets."""
    if edges is None:
        return {}
    y = np.asarray(y_true, dtype=float)
    e = [float(x) for x in edges]
    if len(e) < 2 or any(e[i] >= e[i + 1] for i in range(len(e) - 1)):
        raise ValueError("Wstd bracket edges must be strictly ascending, e.g. 0.1,1,10")
    out: Dict[str, Dict[str, float]] = {}
    if include_outside:
        m = np.isfinite(y) & (y < e[0])
        out[f"lt_{e[0]:g}"] = calc_metrics(y_true[m], y_pred[m])
    for i in range(len(e) - 1):
        lower = e[i]
        upper = e[i + 1]
        if i == len(e) - 2:
            m = np.isfinite(y) & (y >= lower) & (y <= upper)
        else:
            m = np.isfinite(y) & (y >= lower) & (y < upper)
        out[f"{lower:g}_to_{upper:g}"] = calc_metrics(y_true[m], y_pred[m])
    if include_outside:
        m = np.isfinite(y) & (y > e[-1])
        out[f"gt_{e[-1]:g}"] = calc_metrics(y_true[m], y_pred[m])
    return out


def _normalize_and_clip_weights(w: np.ndarray, max_weight: float) -> np.ndarray:
    ww = np.asarray(w, dtype=np.float64)
    ww[~np.isfinite(ww)] = 1.0
    ww = np.maximum(ww, 0.0)
    if float(max_weight) > 0.0:
        ww = np.minimum(ww, float(max_weight))
    mean = float(np.mean(ww)) if len(ww) else 1.0
    if not np.isfinite(mean) or mean <= 0.0:
        mean = 1.0
    return (ww / mean).astype(np.float32)


def compute_sample_weights(
    y_phys: np.ndarray,
    lev_hpa: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    mode = str(getattr(args, "sample_weight_mode", "none"))
    y = np.asarray(y_phys, dtype=np.float64)
    lev = np.asarray(lev_hpa, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64)
    info: Dict[str, object] = {"mode": mode}

    if mode == "none":
        pass
    elif mode == "sqrt_y":
        factor = float(getattr(args, "sqrt_weight_factor", 2.0))
        w = 1.0 + factor * np.sqrt(np.maximum(y, 0.0))
        info["sqrt_weight_factor"] = factor
    elif mode == "tail":
        thresholds = parse_float_sequence(getattr(args, "tail_weight_thresholds", "0.3,1,3"), name="tail-weight-thresholds", min_len=1)
        additions = parse_float_sequence(getattr(args, "tail_weight_additions", "2,5,10"), name="tail-weight-additions", min_len=1)
        if len(thresholds) != len(additions):
            raise ValueError("--tail-weight-thresholds and --tail-weight-additions must have the same length")
        for thr, add in zip(thresholds, additions):
            w += float(add) * (y >= float(thr))
        info["tail_weight_thresholds"] = [float(x) for x in thresholds]
        info["tail_weight_additions"] = [float(x) for x in additions]
    elif mode in {"inverse_wstd_bin", "inverse_wstd_pressure_bin"}:
        w_edges = getattr(args, "wstd_sampling_edges_resolved", None)
        if w_edges is None:
            w_edges = parse_optional_metric_edges(args.wstd_sampling_edges, descending=False, name="wstd-sampling-edges")
        if w_edges is None:
            raise ValueError(f"{mode} requires Wstd bin edges")
        w_id, w_labels = assign_wstd_bins(y, w_edges)
        if mode == "inverse_wstd_bin":
            sid = w_id
            label_map = {i: lab for i, lab in enumerate(w_labels)}
        else:
            p_edges = getattr(args, "pressure_sampling_edges_resolved", None)
            if p_edges is None:
                p_edges = parse_optional_metric_edges(args.pressure_sampling_edges, descending=True, name="pressure-sampling-edges")
            if p_edges is None:
                raise ValueError(f"{mode} requires pressure bin edges")
            p_id, p_labels = assign_pressure_bins(lev, p_edges)
            sid = np.full(len(y), -1, dtype=np.int32)
            valid = (w_id >= 0) & (p_id >= 0)
            sid[valid] = w_id[valid] * len(p_labels) + p_id[valid]
            label_map = {wi * len(p_labels) + pi: f"{w_labels[wi]}__{p_labels[pi]}" for wi in range(len(w_labels)) for pi in range(len(p_labels))}
        valid = sid >= 0
        if np.any(valid):
            ids, counts = np.unique(sid[valid], return_counts=True)
            median_count = float(np.median(counts.astype(float)))
            count_map = {int(i): int(c) for i, c in zip(ids, counts)}
            for i, c in count_map.items():
                w[sid == i] = median_count / max(float(c), 1.0)
            # Outside the requested bins keep weight 1.
            info["bin_counts"] = {label_map.get(int(i), str(i)): int(c) for i, c in sorted(count_map.items())}
        else:
            warnings.warn(f"No samples fell inside the requested bins for {mode}; using unit weights")
    else:
        raise ValueError(f"Unhandled sample weight mode: {mode}")

    max_weight = float(getattr(args, "max_sample_weight", 25.0))
    w = _normalize_and_clip_weights(w, max_weight)
    info.update({
        "max_sample_weight": max_weight,
        "min": float(np.min(w)) if len(w) else math.nan,
        "mean": float(np.mean(w)) if len(w) else math.nan,
        "max": float(np.max(w)) if len(w) else math.nan,
    })
    return w, info


def split_train_test(
    X: np.ndarray,
    y_phys: np.ndarray,
    lev_tag: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y_phys)
    order = rng.permutation(n)
    n_test = max(1, int(args.test_fraction * n))
    test_idx = order[:n_test]
    train_idx = order[n_test:]
    y_fit = transform_target(y_phys, args.target_transform, args.log_eps).astype(np.float32)
    ww = np.asarray(weights, dtype=np.float32)
    return (
        X[train_idx],
        X[test_idx],
        y_fit[train_idx],
        y_phys[test_idx],
        lev_tag[test_idx],
        ww[train_idx],
        ww[test_idx],
        test_idx,
    )


def fit_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test_phys: np.ndarray,
    lev_test: np.ndarray,
    weights_train: Optional[np.ndarray],
    target_transform: str,
    symbolic_feature_names: List[str],
    pressure_bracket_edges: Optional[Sequence[float]] = (1000.0, 800.0, 500.0, 200.0, 10.0),
    wstd_bracket_edges: Optional[Sequence[float]] = (0.1, 1.0, 10.0),
) -> Tuple[str, Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], object]:
    if RidgeCV is None:
        raise RuntimeError("scikit-learn is required for --engine linear")
    model = RidgeCV(alphas=np.logspace(-8, 4, 25))
    if weights_train is not None:
        try:
            model.fit(X_train, y_train, sample_weight=np.asarray(weights_train, dtype=float))
        except TypeError:
            warnings.warn("This scikit-learn RidgeCV does not accept sample_weight; fitting linear baseline unweighted")
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    pred = inverse_target(model.predict(X_test), target_transform)
    metrics = calc_metrics(y_test_phys, pred)
    per_level = metrics_by_level(y_test_phys, pred, lev_test)
    per_pressure_bracket = metrics_by_pressure_bracket(y_test_phys, pred, lev_test, pressure_bracket_edges)
    per_wstd_bracket = metrics_by_wstd_bracket(y_test_phys, pred, wstd_bracket_edges)
    terms = [f"{model.intercept_:.8g}"]
    for name, coef in zip(symbolic_feature_names, model.coef_):
        if abs(float(coef)) > 0.0:
            terms.append(f"({coef:.8g})*{name}")
    expr = lhs_name(target_transform) + " = " + " + ".join(terms)
    return expr, metrics, per_level, per_pressure_bracket, per_wstd_bracket, model


def pysr_operator_kwargs(operator_set: str) -> Dict[str, object]:
    import sympy as sp
    if operator_set == "basic":
        return {
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["square", "abs"],
            "constraints": {"/": (-1, 10), "square": 12, "abs": 12},
            "nested_constraints": {"square": {"square": 0}, "abs": {"abs": 0}},
        }
    if operator_set == "protected":
        return {
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": [
                "square",
                "cube",
                "abs",
                "sqrt_abs(x)=sqrt(abs(x))",
                "log_abs(x)=log(abs(x) + 1.0f-6)",
            ],
            "extra_sympy_mappings": {
                "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x)),
                "log_abs": lambda x: sp.log(sp.Abs(x) + sp.Float("1.0e-6")),
                "cube": lambda x: x**3,
            },
            "constraints": {"/": (-1, 10), "square": 12, "cube": 12, "sqrt_abs": 12, "log_abs": 12, "abs": 12},
            "nested_constraints": {
                "square": {"square": 0, "cube": 0, "sqrt_abs": 0, "log_abs": 0},
                "cube": {"square": 0, "cube": 0, "sqrt_abs": 0, "log_abs": 0},
                "sqrt_abs": {"sqrt_abs": 0, "log_abs": 0, "square": 0, "cube": 0},
                "log_abs": {"log_abs": 0, "sqrt_abs": 0, "square": 0, "cube": 0},
                "abs": {"abs": 0},
            },
        }
    if operator_set == "extended":
        return {
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": [
                "square",
                "cube",
                "abs",
                "sqrt_abs(x)=sqrt(abs(x))",
                "log_abs(x)=log(abs(x) + 1.0f-6)",
                "exp_clip(x)=exp(max(-20.0f0, min(20.0f0, x)))",
                "sin",
                "cos",
                "tanh",
                "atan",
            ],
            "extra_sympy_mappings": {
                "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x)),
                "log_abs": lambda x: sp.log(sp.Abs(x) + sp.Float("1.0e-6")),
                "exp_clip": lambda x: sp.exp(sp.Max(sp.Float("-20.0"), sp.Min(sp.Float("20.0"), x))),
                "cube": lambda x: x**3,
            },
            "constraints": {
                "/": (-1, 12),
                "square": 14,
                "cube": 14,
                "sqrt_abs": 14,
                "log_abs": 14,
                "abs": 14,
                "exp_clip": 10,
                "sin": 10,
                "cos": 10,
                "tanh": 10,
                "atan": 10,
            },
            "nested_constraints": {
                "square": {"square": 0, "cube": 0, "sqrt_abs": 0, "log_abs": 1, "exp_clip": 1, "sin": 1, "cos": 1, "tanh": 1, "atan": 1},
                "cube": {"square": 0, "cube": 0, "sqrt_abs": 0, "log_abs": 1, "exp_clip": 1, "sin": 1, "cos": 1, "tanh": 1, "atan": 1},
                "abs": {"abs": 0},
                "sqrt_abs": {"sqrt_abs": 0, "log_abs": 0, "square": 0, "cube": 0, "exp_clip": 0},
                "log_abs": {"log_abs": 0, "sqrt_abs": 0, "square": 0, "cube": 0, "exp_clip": 0},
                "exp_clip": {"exp_clip": 0, "log_abs": 0, "sqrt_abs": 0, "square": 0, "cube": 0},
                "sin": {"sin": 0, "cos": 0, "tanh": 0, "atan": 0, "exp_clip": 0},
                "cos": {"sin": 0, "cos": 0, "tanh": 0, "atan": 0, "exp_clip": 0},
                "tanh": {"sin": 0, "cos": 0, "tanh": 0, "atan": 0, "exp_clip": 0},
                "atan": {"sin": 0, "cos": 0, "tanh": 0, "atan": 0, "exp_clip": 0},
            },
        }
    raise ValueError(f"Unknown operator set: {operator_set}")


def resolve_resume_path(resume_from: Optional[str]) -> Optional[Path]:
    if not resume_from:
        return None
    p = Path(resume_from).expanduser()
    if p.is_dir():
        checkpoint = p / "checkpoint.pkl"
        if checkpoint.exists():
            return checkpoint
        pkls = sorted(p.glob("*.pkl"), key=lambda q: q.stat().st_mtime, reverse=True)
        if pkls:
            return pkls[0]
        raise FileNotFoundError(f"No checkpoint.pkl or *.pkl was found in {p}")
    if not p.exists():
        raise FileNotFoundError(f"--resume-from file does not exist: {p}")
    return p


def _candidate_manifest_paths_from_resume(resume_path: Path) -> List[Path]:
    candidates: List[Path] = []
    parents = [resume_path.parent] + list(resume_path.parents)
    for parent in parents[:5]:
        candidates.append(parent / "manifest_and_results.json")
        if parent.name == "pysr":
            candidates.append(parent.parent / "manifest_and_results.json")
    out: List[Path] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def read_previous_manifest_for_resume(resume_path: Optional[Path]) -> Dict[str, object]:
    if resume_path is None:
        return {}
    for path in _candidate_manifest_paths_from_resume(resume_path):
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                print(f"Read previous manifest for resume: {path}")
                return data
            except Exception as e:
                warnings.warn(f"Could not read previous manifest {path}: {e}")
    return {}


def check_resume_compatibility(args: argparse.Namespace, previous_manifest: Dict[str, object]) -> None:
    if not previous_manifest or args.resume_allow_mismatch:
        return
    mismatches: List[str] = []
    prev_features = previous_manifest.get("feature_names_symbolic")
    if isinstance(prev_features, list) and list(prev_features) != list(args.selected_symbolic_features):
        mismatches.append("feature_names_symbolic differ")
    prev_transform = previous_manifest.get("target_transform")
    if isinstance(prev_transform, str) and prev_transform != args.target_transform:
        mismatches.append(f"target_transform differs: previous={prev_transform} current={args.target_transform}")
    prev_standardization = previous_manifest.get("standardization_mode")
    if isinstance(prev_standardization, str) and prev_standardization != args.standardization:
        mismatches.append(f"standardization differs: previous={prev_standardization} current={args.standardization}")
    prev_options = previous_manifest.get("pysr_options", {})
    if isinstance(prev_options, dict):
        prev_operator_set = prev_options.get("operator_set")
        if isinstance(prev_operator_set, str) and prev_operator_set != args.operator_set:
            mismatches.append(f"operator_set differs: previous={prev_operator_set} current={args.operator_set}")
    if mismatches:
        raise RuntimeError(
            "Refusing to resume because the previous run appears incompatible:\n  - "
            + "\n  - ".join(mismatches)
            + "\nUse --resume-allow-mismatch to override."
        )


def set_model_params_one_by_one(model, params: Dict[str, object]) -> None:
    for key, value in params.items():
        try:
            model.set_params(**{key: value})
        except Exception:
            try:
                setattr(model, key, value)
            except Exception as e2:
                warnings.warn(f"Could not set resumed PySR parameter {key}={value!r}: {e2}")


def build_resume_param_updates(args: argparse.Namespace, pysr_dir: Path) -> Dict[str, object]:
    params: Dict[str, object] = {
        "warm_start": True,
        "niterations": args.niterations,
        "populations": args.populations,
        "population_size": args.population_size,
        "ncycles_per_iteration": args.ncycles_per_iteration,
        "maxsize": args.maxsize,
        "model_selection": args.model_selection,
        "verbosity": 1,
        "complexity_of_variables": args.complexity_of_variables,
        "complexity_of_constants": args.complexity_of_constants,
        "parsimony": args.parsimony,
        "use_frequency": not args.no_use_frequency,
        "output_directory": str(pysr_dir),
    }
    if args.maxdepth and args.maxdepth > 0:
        params["maxdepth"] = int(args.maxdepth)
    if getattr(args, "timeout_in_seconds", 0.0) and float(args.timeout_in_seconds) > 0.0:
        params["timeout_in_seconds"] = float(args.timeout_in_seconds)
    if args.batching:
        params["batching"] = True
        params["batch_size"] = int(args.batch_size)
    if args.procs and args.procs > 0:
        params["procs"] = int(args.procs)
    return params


def finite_or(value: float, default: float = -1.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def validation_score_for_candidate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    complexity: float,
    args: argparse.Namespace,
) -> Dict[str, float]:
    global_m = calc_metrics(y_true, y_pred)
    high_thr = float(getattr(args, "rerank_high_threshold", 1.0))
    extreme_thr = float(getattr(args, "rerank_extreme_threshold", 3.0))
    high_mask = np.asarray(y_true) >= high_thr
    extreme_mask = np.asarray(y_true) >= extreme_thr
    high_m = calc_metrics(np.asarray(y_true)[high_mask], np.asarray(y_pred)[high_mask]) if np.any(high_mask) else calc_metrics([], [])
    extreme_m = calc_metrics(np.asarray(y_true)[extreme_mask], np.asarray(y_pred)[extreme_mask]) if np.any(extreme_mask) else calc_metrics([], [])

    y_mean = float(np.nanmean(y_true)) if len(y_true) else 1.0
    high_mean = float(np.nanmean(np.asarray(y_true)[high_mask])) if np.any(high_mask) else y_mean
    extreme_mean = float(np.nanmean(np.asarray(y_true)[extreme_mask])) if np.any(extreme_mask) else high_mean
    eps = 1.0e-12

    corr_global = finite_or(global_m.get("corr", math.nan), -1.0)
    corr_high = finite_or(high_m.get("corr", math.nan), 0.0)
    corr_extreme = finite_or(extreme_m.get("corr", math.nan), 0.0)
    bias_global_norm = abs(finite_or(global_m.get("bias", math.nan), 0.0)) / max(abs(y_mean), eps)
    bias_high_norm = abs(finite_or(high_m.get("bias", math.nan), 0.0)) / max(abs(high_mean), eps)
    bias_extreme_norm = abs(finite_or(extreme_m.get("bias", math.nan), 0.0)) / max(abs(extreme_mean), eps)
    cpen = float(getattr(args, "rerank_complexity_penalty", 0.005)) * max(float(complexity), 0.0)

    score = (
        corr_global
        + 0.50 * corr_high
        + 0.25 * corr_extreme
        - 0.10 * bias_global_norm
        - 0.25 * bias_high_norm
        - 0.15 * bias_extreme_norm
        - cpen
    )
    return {
        "validation_score": float(score),
        "corr_global": float(corr_global),
        "corr_high": float(corr_high),
        "corr_extreme": float(corr_extreme),
        "bias_global_norm": float(bias_global_norm),
        "bias_high_norm": float(bias_high_norm),
        "bias_extreme_norm": float(bias_extreme_norm),
        "complexity_penalty": float(cpen),
        "n_high": int(np.count_nonzero(high_mask)),
        "n_extreme": int(np.count_nonzero(extreme_mask)),
        "rmse_global": finite_or(global_m.get("rmse", math.nan), math.nan),
        "mae_global": finite_or(global_m.get("mae", math.nan), math.nan),
        "bias_global": finite_or(global_m.get("bias", math.nan), math.nan),
        "r2_global": finite_or(global_m.get("r2", math.nan), math.nan),
    }


def eval_pysr_rhs_on_matrix(rhs: str, X: np.ndarray, symbolic_feature_names: Sequence[str]) -> np.ndarray:
    import sympy as sp

    symbols = [sp.Symbol(str(name)) for name in symbolic_feature_names]
    local_dict = {str(name): sym for name, sym in zip(symbolic_feature_names, symbols)}
    local_dict.update({
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x)),
        "log_abs": lambda x: sp.log(sp.Abs(x) + sp.Float("1.0e-6")),
        "exp_clip": lambda x: sp.exp(sp.Max(sp.Float("-20.0"), sp.Min(sp.Float("20.0"), x))),
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "sin": sp.sin,
        "cos": sp.cos,
        "tanh": sp.tanh,
        "atan": sp.atan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
    })
    expr = sp.sympify(str(rhs), locals=local_dict)
    modules = [{
        "Abs": np.abs,
        "Max": np.maximum,
        "Min": np.minimum,
        "sin": np.sin,
        "cos": np.cos,
        "tanh": np.tanh,
        "atan": np.arctan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
    }, "numpy"]
    fn = sp.lambdify(symbols, expr, modules=modules)
    vals = fn(*[X[:, j] for j in range(X.shape[1])])
    vals = np.asarray(vals, dtype=np.float64)
    if vals.shape == ():
        vals = np.full(X.shape[0], float(vals), dtype=np.float64)
    return vals


def rerank_pysr_equations_on_validation(
    model: object,
    X_test: np.ndarray,
    y_test_phys: np.ndarray,
    lev_test: np.ndarray,
    target_transform: str,
    args: argparse.Namespace,
    outdir: Path,
) -> Optional[Dict[str, object]]:
    if not bool(getattr(args, "rerank_equations", True)):
        return None

    df = None
    if hasattr(model, "equations_"):
        try:
            df = model.equations_.copy()
        except Exception:
            df = None
    if df is None:
        csv_path = outdir / "equations_global.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = None
    if df is None or len(df) == 0 or "equation" not in df.columns:
        return None

    rows: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    y_true = np.asarray(y_test_phys, dtype=np.float64)

    for idx, row in df.reset_index(drop=True).iterrows():
        rhs = str(row.get("equation", ""))
        if not rhs or rhs.lower() == "nan":
            continue
        complexity = finite_or(row.get("complexity", math.nan), float(idx + 1))
        try:
            pred_fit = eval_pysr_rhs_on_matrix(rhs, X_test, args.selected_symbolic_features)
            pred_phys = inverse_target(pred_fit, target_transform)
            pred_phys = np.asarray(pred_phys, dtype=np.float64)
            pred_phys = np.where(np.isfinite(pred_phys), pred_phys, np.nan)
            pred_phys = np.clip(pred_phys, 0.0, float(getattr(args, "y_max", 10.0)) * 2.0)
            score_info = validation_score_for_candidate(y_true, pred_phys, complexity, args)
            out_row: Dict[str, object] = {
                "row_index": int(idx),
                "equation": rhs,
                "complexity": float(complexity),
                "loss": finite_or(row.get("loss", math.nan), math.nan),
                "score_pysr": finite_or(row.get("score", math.nan), math.nan),
                "status": "ok",
            }
            out_row.update(score_info)
            rows.append(out_row)
            if best is None or float(out_row["validation_score"]) > float(best["validation_score"]):
                best = dict(out_row)
                best["prediction_physical"] = pred_phys
        except Exception as e:
            rows.append({
                "row_index": int(idx),
                "equation": rhs,
                "complexity": float(complexity),
                "status": "failed",
                "error": repr(e),
            })

    if rows:
        rank_df = pd.DataFrame(rows)
        if "validation_score" in rank_df.columns:
            rank_df = rank_df.sort_values("validation_score", ascending=False, na_position="last")
        try:
            rank_df.to_csv(outdir / "equation_validation_ranking.csv", index=False)
        except Exception as e:
            warnings.warn(f"Could not write equation_validation_ranking.csv: {e}")

    if best is None:
        return None

    pred_best = np.asarray(best.pop("prediction_physical"), dtype=np.float64)
    metrics = calc_metrics(y_true, pred_best)
    per_level = metrics_by_level(y_true, pred_best, lev_test)
    per_pressure_bracket = metrics_by_pressure_bracket(
        y_true, pred_best, lev_test, getattr(args, "pressure_bracket_edges_resolved", (1000.0, 800.0, 500.0, 200.0, 10.0))
    )
    per_wstd_bracket = metrics_by_wstd_bracket(
        y_true, pred_best, getattr(args, "wstd_bracket_edges_resolved", (0.1, 1.0, 10.0))
    )
    return {
        "rhs": str(best["equation"]),
        "ranking_row": best,
        "metrics": metrics,
        "per_level": per_level,
        "per_pressure_bracket": per_pressure_bracket,
        "per_wstd_bracket": per_wstd_bracket,
        "ranking_csv": str(outdir / "equation_validation_ranking.csv"),
    }


def fit_pysr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test_phys: np.ndarray,
    lev_test: np.ndarray,
    weights_train: Optional[np.ndarray],
    target_transform: str,
    args: argparse.Namespace,
    outdir: Path,
) -> Tuple[str, Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], object]:
    try:
        from pysr import PySRRegressor
    except Exception as e:
        raise RuntimeError(
            "PySR is not importable. Use '--engine linear' to test the pipeline, "
            "or install PySR and Julia for symbolic regression. Original error: " + repr(e)
        )

    pysr_dir = outdir / "pysr"
    pysr_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        niterations=args.niterations,
        populations=args.populations,
        population_size=args.population_size,
        ncycles_per_iteration=args.ncycles_per_iteration,
        maxsize=args.maxsize,
        model_selection=args.model_selection,
        verbosity=1,
        random_state=args.seed,
        output_directory=str(pysr_dir),
        complexity_of_variables=args.complexity_of_variables,
        complexity_of_constants=args.complexity_of_constants,
        parsimony=args.parsimony,
        warmup_maxsize_by=args.warmup_maxsize_by,
        select_k_features=None,
        use_frequency=not args.no_use_frequency,
        use_frequency_in_tournament=not args.no_use_frequency,
    )
    if args.maxdepth and args.maxdepth > 0:
        kwargs["maxdepth"] = int(args.maxdepth)
    if getattr(args, "timeout_in_seconds", 0.0) and float(args.timeout_in_seconds) > 0.0:
        kwargs["timeout_in_seconds"] = float(args.timeout_in_seconds)
    kwargs.update(pysr_operator_kwargs(args.operator_set))
    if args.batching:
        kwargs["batching"] = True
        kwargs["batch_size"] = int(args.batch_size)
    if args.procs and args.procs > 0:
        kwargs["procs"] = int(args.procs)

    resume_path = resolve_resume_path(args.resume_from)
    try:
        if resume_path is not None:
            print(f"Resuming PySR search from: {resume_path}")
            model = PySRRegressor.from_file(str(resume_path))
            set_model_params_one_by_one(model, build_resume_param_updates(args, pysr_dir))
        else:
            model = PySRRegressor(**kwargs)
    except TypeError as e:
        if resume_path is not None:
            raise
        retry_kwargs = dict(kwargs)
        retry = False
        for key in ["output_directory", "random_state", "warmup_maxsize_by", "use_frequency_in_tournament"]:
            if key in retry_kwargs:
                retry_kwargs.pop(key, None)
                retry = True
        if retry:
            try:
                model = PySRRegressor(**retry_kwargs)
            except TypeError as e2:
                raise TypeError("PySRRegressor rejected supplied options: " + repr(e) + "; retry: " + repr(e2))
        else:
            raise

    print(
        "PySR options: "
        f"niterations={args.niterations}, populations={args.populations}, "
        f"population_size={args.population_size}, ncycles_per_iteration={args.ncycles_per_iteration}, "
        f"maxsize={args.maxsize}, maxdepth={args.maxdepth if args.maxdepth and args.maxdepth > 0 else 'none'}, "
        f"model_selection={args.model_selection}, complexity_of_variables={args.complexity_of_variables}, "
        f"operator_set={args.operator_set}, batching={bool(args.batching)}, batch_size={int(args.batch_size)}, "
        f"timeout_in_seconds={float(getattr(args, 'timeout_in_seconds', 0.0)) if float(getattr(args, 'timeout_in_seconds', 0.0)) > 0.0 else 'none'}, "
        f"resume_from={args.resume_from if args.resume_from else 'none'}"
    )

    if weights_train is not None and str(getattr(args, "sample_weight_mode", "none")) != "none":
        try:
            model.fit(
                X_train,
                y_train,
                variable_names=args.selected_symbolic_features,
                weights=np.asarray(weights_train, dtype=np.float32),
            )
        except TypeError:
            warnings.warn(
                "This PySR version rejected fit(..., weights=...). "
                "Fitting unweighted, but keeping stratified sampling and validation re-ranking."
            )
            model.fit(X_train, y_train, variable_names=args.selected_symbolic_features)
    else:
        model.fit(X_train, y_train, variable_names=args.selected_symbolic_features)

    if hasattr(model, "equations_"):
        try:
            model.equations_.to_csv(outdir / "equations_global.csv", index=False)
        except Exception as e:
            warnings.warn(f"Could not save PySR equation table: {e}")

    reranked = rerank_pysr_equations_on_validation(model, X_test, y_test_phys, lev_test, target_transform, args, outdir)
    if reranked is not None:
        rhs = str(reranked["rhs"])
        metrics = reranked["metrics"]
        per_level = reranked["per_level"]
        per_pressure_bracket = reranked["per_pressure_bracket"]
        per_wstd_bracket = reranked["per_wstd_bracket"]
        try:
            atomic_write_json(outdir / "selected_equation_validation.json", {
                "selection_method": "validation_rerank",
                "target_transform": target_transform,
                "ranking_row": reranked.get("ranking_row", {}),
                "ranking_csv": reranked.get("ranking_csv"),
                "metrics": metrics,
            })
        except Exception as e:
            warnings.warn(f"Could not write selected_equation_validation.json: {e}")
    else:
        pred = inverse_target(np.asarray(model.predict(X_test)), target_transform)
        metrics = calc_metrics(y_test_phys, pred)
        per_level = metrics_by_level(y_test_phys, pred, lev_test)
        per_pressure_bracket = metrics_by_pressure_bracket(
            y_test_phys, pred, lev_test, getattr(args, "pressure_bracket_edges_resolved", (1000.0, 800.0, 500.0, 200.0, 10.0))
        )
        per_wstd_bracket = metrics_by_wstd_bracket(
            y_test_phys, pred, getattr(args, "wstd_bracket_edges_resolved", (0.1, 1.0, 10.0))
        )
        try:
            rhs = str(model.sympy())
        except Exception:
            try:
                rhs = str(model.get_best()["equation"])
            except Exception:
                rhs = str(model)
    expr = lhs_name(target_transform) + " = " + rhs
    return expr, metrics, per_level, per_pressure_bracket, per_wstd_bracket, model


def train_global(
    X: np.ndarray,
    y_phys: np.ndarray,
    lev_tag: np.ndarray,
    args: argparse.Namespace,
    outdir: Path,
    rng: np.random.Generator,
) -> Dict[str, object]:
    if len(y_phys) < 50:
        raise RuntimeError(f"Too few pooled samples: {len(y_phys)}")
    sample_weights, sample_weight_info = compute_sample_weights(y_phys, lev_tag, args)
    args.sample_weight_info = sample_weight_info
    print("Sample weighting:", json.dumps(sample_weight_info, indent=2)[:2000])
    (
        X_train,
        X_test,
        y_train,
        y_test_phys,
        lev_test,
        weights_train,
        weights_test,
        _,
    ) = split_train_test(X, y_phys, lev_tag, sample_weights, args, rng)
    if args.engine == "linear":
        expr, metrics, per_level, per_pressure_bracket, per_wstd_bracket, _ = fit_linear(
            X_train,
            y_train,
            X_test,
            y_test_phys,
            lev_test,
            weights_train,
            args.target_transform,
            args.selected_symbolic_features,
            getattr(args, "pressure_bracket_edges_resolved", (1000.0, 800.0, 500.0, 200.0, 10.0)),
            getattr(args, "wstd_bracket_edges_resolved", (0.1, 1.0, 10.0)),
        )
    else:
        expr, metrics, per_level, per_pressure_bracket, per_wstd_bracket, _ = fit_pysr(
            X_train, y_train, X_test, y_test_phys, lev_test, weights_train, args.target_transform, args, outdir
        )
    return {
        "fit_type": "one_global_expression_merra2_sample_table",
        "run_preset": getattr(args, "run_preset", "none"),
        "n_samples": int(len(y_phys)),
        "levels_hpa_sampled": [float(x) for x in sorted(np.unique(lev_tag.astype(float)))],
        "target_transform": args.target_transform,
        "log_eps": float(args.log_eps),
        "sampling_mode": getattr(args, "sampling_mode", "random"),
        "wstd_sampling_edges": (
            None if getattr(args, "wstd_sampling_edges_resolved", None) is None
            else [float(x) for x in getattr(args, "wstd_sampling_edges_resolved")]
        ),
        "pressure_sampling_edges_hpa": (
            None if getattr(args, "pressure_sampling_edges_resolved", None) is None
            else [float(x) for x in getattr(args, "pressure_sampling_edges_resolved")]
        ),
        "sample_weight_mode": getattr(args, "sample_weight_mode", "none"),
        "sample_weight_info": getattr(args, "sample_weight_info", {}),
        "validation_reranking_enabled": bool(getattr(args, "rerank_equations", True)),
        "pressure_bracket_edges_hpa": (
            None if getattr(args, "pressure_bracket_edges_resolved", None) is None
            else [float(x) for x in getattr(args, "pressure_bracket_edges_resolved")]
        ),
        "wstd_bracket_edges": (
            None if getattr(args, "wstd_bracket_edges_resolved", None) is None
            else [float(x) for x in getattr(args, "wstd_bracket_edges_resolved")]
        ),
        "operator_set": args.operator_set,
        "feature_space": "physical" if args.standardization == "none" else f"standardized_{args.standardization}",
        "standardization_mode": args.standardization,
        "feature_names_internal": args.selected_features,
        "feature_names_symbolic": args.selected_symbolic_features,
        "feature_standardization": getattr(args, "feature_standardization", {}),
        "derived_feature_descriptions": {
            k: DERIVED_FEATURE_DESCRIPTIONS[k]
            for k in args.selected_features
            if k in DERIVED_FEATURE_DESCRIPTIONS
        },
        "level_feature_included": "P" in args.selected_features,
        "expression": expr,
        "test_metrics_physical_Wstd_global": metrics,
        "test_metrics_physical_Wstd_by_level_hpa": per_level,
        "test_metrics_physical_Wstd_by_pressure_bracket_hpa": per_pressure_bracket,
        "test_metrics_physical_Wstd_by_Wstd_bracket": per_wstd_bracket,
    }


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def atomic_write_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(_json_safe(obj), f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def find_latest_pysr_file(outdir: Path, names: List[str]) -> Optional[Path]:
    pysr_root = outdir / "pysr"
    candidates: List[Path] = []
    if not pysr_root.exists():
        return None
    for name in names:
        candidates.extend(pysr_root.rglob(name))
    candidates = [p for p in candidates if p.exists() and p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def best_equation_from_csv(csv_path: Path, selection: str = "loss") -> Tuple[Optional[int], Optional[str], Dict[str, object]]:
    info: Dict[str, object] = {"csv_path": str(csv_path)}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        info["error"] = f"could not read CSV: {e}"
        return None, None, info
    info["n_rows"] = int(len(df))
    info["columns"] = list(df.columns)
    if len(df) == 0 or "equation" not in df.columns:
        return None, None, info
    selection = str(selection or "loss").lower()
    idx = len(df) - 1

    if selection == "score" and "score" in df.columns:
        score = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(score)
        if np.any(finite):
            finite_idx = np.flatnonzero(finite)
            idx = int(finite_idx[np.argmax(score[finite])])
            info["selection"] = "maximum_score"
            info["score"] = float(score[idx])
        else:
            info["selection"] = "last_row_no_finite_score"
    elif selection == "last":
        info["selection"] = "last_row"
    elif "loss" in df.columns:
        loss = pd.to_numeric(df["loss"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(loss)
        if np.any(finite):
            finite_idx = np.flatnonzero(finite)
            idx = int(finite_idx[np.argmin(loss[finite])])
            info["selection"] = "minimum_loss"
            info["loss"] = float(loss[idx])
        else:
            info["selection"] = "last_row_no_finite_loss"
    else:
        info["selection"] = "last_row_no_loss_column"
    eq = str(df.iloc[idx]["equation"])
    info["row_index"] = int(idx)
    for col in ["complexity", "score"]:
        if col in df.columns:
            try:
                info[col] = float(df.iloc[idx][col])
            except Exception:
                info[col] = str(df.iloc[idx][col])
    return idx, eq, info


def harvest_pysr_outputs(outdir: Path, manifest: Optional[Dict[str, object]] = None, reason: str = "harvest") -> Dict[str, object]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {
        "reason": reason,
        "time_unix": time.time(),
        "time_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "outdir": str(outdir),
    }
    latest_hof = find_latest_pysr_file(outdir, ["hall_of_fame.csv", "hall_of_fame.csv.bak"])
    if latest_hof is not None:
        report["latest_hall_of_fame_csv"] = str(latest_hof)
        for target_name in ["equations_global_latest.csv", "equations_global.csv"]:
            try:
                shutil.copy2(latest_hof, outdir / target_name)
                report[target_name] = str(outdir / target_name)
            except Exception as e:
                report[f"{target_name}_error"] = str(e)
        _, rhs, eq_info = best_equation_from_csv(latest_hof, selection=(getattr(manifest, "get", lambda *_: None)("harvest_selection") if manifest is not None else "loss") or "loss")
        report["best_equation_info"] = eq_info
        if rhs:
            target_transform = None
            if manifest:
                target_transform = manifest.get("target_transform")
                if not isinstance(target_transform, str):
                    result = manifest.get("result", {})
                    if isinstance(result, dict):
                        target_transform = result.get("target_transform")
            expression_text = f"{lhs_name(target_transform)} = {rhs}\n"
            try:
                atomic_write_text(outdir / "expression_global_latest.txt", expression_text)
                if not (outdir / "expression_global.txt").exists():
                    atomic_write_text(outdir / "expression_global.txt", expression_text)
                report["expression_global_latest.txt"] = str(outdir / "expression_global_latest.txt")
            except Exception as e:
                report["expression_write_error"] = str(e)
    else:
        report["latest_hall_of_fame_csv"] = None

    latest_checkpoint = find_latest_pysr_file(outdir, ["checkpoint.pkl"])
    if latest_checkpoint is not None:
        report["latest_checkpoint"] = str(latest_checkpoint)
        try:
            shutil.copy2(latest_checkpoint, outdir / "checkpoint_latest.pkl")
            report["checkpoint_latest.pkl"] = str(outdir / "checkpoint_latest.pkl")
        except Exception as e:
            report["checkpoint_copy_error"] = str(e)
    else:
        report["latest_checkpoint"] = None

    if manifest is not None:
        partial = dict(manifest)
        partial["run_status"] = reason
        partial["latest_harvest"] = report
        try:
            atomic_write_json(outdir / "manifest_partial.json", partial)
            if not (outdir / "manifest_and_results.json").exists():
                atomic_write_json(outdir / "manifest_and_results.json", partial)
            report["manifest_partial.json"] = str(outdir / "manifest_partial.json")
        except Exception as e:
            report["manifest_write_error"] = str(e)
    try:
        atomic_write_json(outdir / "harvest_status.json", report)
    except Exception:
        pass
    return report


def install_harvest_signal_handlers(outdir: Path, manifest_ref: Dict[str, object]) -> None:
    def _handler(signum, frame):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(f"\nReceived {signame}; harvesting latest PySR outputs before exit...", file=sys.stderr, flush=True)
        try:
            harvest_pysr_outputs(outdir, manifest_ref, reason=f"signal_{signame}")
        except Exception as e:
            print(f"Harvest failed during {signame}: {e}", file=sys.stderr, flush=True)
        raise SystemExit(128 + int(signum))
    for sig in [signal.SIGTERM, signal.SIGINT]:
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass
    if hasattr(signal, "SIGUSR1"):
        try:
            signal.signal(signal.SIGUSR1, _handler)
        except Exception:
            pass
    try:
        atexit.register(lambda: harvest_pysr_outputs(outdir, manifest_ref, reason="atexit"))
    except Exception:
        pass



class PeriodicHarvester:
    """Background best-effort PySR harvester for long blocking model.fit calls."""

    def __init__(self, outdir: Path, manifest_ref: Dict[str, object], interval_seconds: float):
        self.outdir = Path(outdir)
        self.manifest_ref = manifest_ref
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.interval_seconds <= 0.0:
            return
        self._thread = threading.Thread(target=self._run, name="pysr-output-harvester", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run(self) -> None:
        # Wait one interval before the first periodic harvest.  The main code
        # already performs a harvest immediately before and after the fit.
        while not self._stop.wait(self.interval_seconds):
            try:
                report = harvest_pysr_outputs(self.outdir, self.manifest_ref, reason="periodic_during_fit")
                latest = report.get("expression_global_latest.txt")
                hof = report.get("latest_hall_of_fame_csv")
                if latest or hof:
                    print(
                        f"Periodic harvest: latest_hall_of_fame={hof}, expression={latest}",
                        flush=True,
                    )
            except Exception as e:
                print(f"Periodic harvest failed: {e}", file=sys.stderr, flush=True)


def main() -> int:
    args = parse_args()
    if args.list_features:
        print_feature_catalog()
        return 0
    if args.no_standardize:
        args.standardization = "none"

    try:
        apply_run_preset(args)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 9

    try:
        args.timeout_in_seconds = resolve_timeout_in_seconds(args)
        if args.timeout_in_seconds > 0.0:
            print(f"Resolved PySR timeout_in_seconds={args.timeout_in_seconds:.0f}")
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 7

    try:
        args.pressure_bracket_edges_resolved = parse_optional_metric_edges(
            args.pressure_bracket_edges, descending=True, name="pressure-bracket-edges"
        )
        args.wstd_bracket_edges_resolved = parse_optional_metric_edges(
            args.wstd_bracket_edges, descending=False, name="wstd-bracket-edges"
        )
        args.wstd_sampling_edges_resolved = parse_optional_metric_edges(
            args.wstd_sampling_edges, descending=False, name="wstd-sampling-edges"
        )
        args.pressure_sampling_edges_resolved = parse_optional_metric_edges(
            args.pressure_sampling_edges, descending=True, name="pressure-sampling-edges"
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 5

    if args.print_config_only:
        print_resolved_run_config(args)
        return 0

    try:
        selected_features, selected_symbolic_features = feature_names_from_args(args)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 5
    args.selected_features = selected_features
    args.selected_symbolic_features = selected_symbolic_features

    try:
        args.resume_path_resolved = resolve_resume_path(args.resume_from)
        previous_manifest = read_previous_manifest_for_resume(args.resume_path_resolved)
        check_resume_compatibility(args, previous_manifest)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 6

    if not args.input_file:
        print("Input file is required unless --list-features is used.", file=sys.stderr)
        return 2
    input_file = Path(args.input_file).expanduser()
    if not input_file.exists():
        print(f"Input file not found: {input_file}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", input_file.stem)
    if args.outdir:
        outdir = Path(args.outdir)
    elif args.run_preset and args.run_preset != "none":
        outdir = Path(f"symbolic_Wstd_MERRA2_{args.run_preset}_{safe_stem}")
    else:
        outdir = Path(f"symbolic_Wstd_MERRA2_{safe_stem}")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.run_preset and args.run_preset != "none":
        print(f"Using run preset: {args.run_preset}")
    print("Selected features for symbolic regression:")
    print("  internal: " + ", ".join(args.selected_features))
    print("  symbolic: " + ", ".join(args.selected_symbolic_features))
    selected_derived = [v for v in args.selected_features if v in DERIVED_FEATURE_DESCRIPTIONS]
    if selected_derived:
        print("  derived: " + "; ".join(f"{v}={DERIVED_FEATURE_DESCRIPTIONS[v]}" for v in selected_derived))

    manifest: Dict[str, object] = {
        "input_file": str(input_file),
        "run_preset": args.run_preset,
        "feature_names_internal": args.selected_features,
        "feature_names_symbolic": args.selected_symbolic_features,
        "available_features_internal": ALL_FEATURES,
        "available_features_symbolic": [SYMBOLIC_NAME[f] for f in ALL_FEATURES],
        "compact_default_features": COMPACT_FEATURES,
        "derived_feature_descriptions": {
            k: DERIVED_FEATURE_DESCRIPTIONS[k]
            for k in args.selected_features
            if k in DERIVED_FEATURE_DESCRIPTIONS
        },
        "standardization_mode": args.standardization,
        "target_transform": args.target_transform,
        "log_eps": float(args.log_eps),
        "sampling_mode": getattr(args, "sampling_mode", "random"),
        "wstd_sampling_edges": (
            None if getattr(args, "wstd_sampling_edges_resolved", None) is None
            else [float(x) for x in getattr(args, "wstd_sampling_edges_resolved")]
        ),
        "pressure_sampling_edges_hpa": (
            None if getattr(args, "pressure_sampling_edges_resolved", None) is None
            else [float(x) for x in getattr(args, "pressure_sampling_edges_resolved")]
        ),
        "sample_weight_mode": getattr(args, "sample_weight_mode", "none"),
        "sample_weight_info": getattr(args, "sample_weight_info", {}),
        "validation_reranking_enabled": bool(getattr(args, "rerank_equations", True)),
        "pressure_bracket_edges_hpa": (
            None if getattr(args, "pressure_bracket_edges_resolved", None) is None
            else [float(x) for x in getattr(args, "pressure_bracket_edges_resolved")]
        ),
        "wstd_bracket_edges": (
            None if getattr(args, "wstd_bracket_edges_resolved", None) is None
            else [float(x) for x in getattr(args, "wstd_bracket_edges_resolved")]
        ),
        "fit_type": "one_global_expression_merra2_sample_table",
        "level_feature_included": "P" in args.selected_features,
        "pysr_options": {
            "niterations": int(args.niterations),
            "populations": int(args.populations),
            "population_size": int(args.population_size),
            "ncycles_per_iteration": int(args.ncycles_per_iteration),
            "maxsize": int(args.maxsize),
            "maxdepth": int(args.maxdepth),
            "model_selection": args.model_selection,
            "operator_set": args.operator_set,
            "complexity_of_variables": float(args.complexity_of_variables),
            "complexity_of_constants": float(args.complexity_of_constants),
            "parsimony": float(args.parsimony),
            "warmup_maxsize_by": float(args.warmup_maxsize_by),
            "use_frequency": bool(not args.no_use_frequency),
            "batching": bool(args.batching),
            "batch_size": int(args.batch_size),
            "sample_weight_mode": getattr(args, "sample_weight_mode", "none"),
            "validation_reranking_enabled": bool(getattr(args, "rerank_equations", True)),
            "timeout_in_seconds": float(getattr(args, "timeout_in_seconds", 0.0)),
            "timeout_source": (
                "slurm_remaining_walltime" if bool(getattr(args, "timeout_slurm", False))
                else "timeout_hours" if float(getattr(args, "timeout_hours", 0.0) or 0.0) > 0.0
                else "timeout_seconds" if float(getattr(args, "timeout_seconds", 0.0) or 0.0) > 0.0
                else "none"
            ),
            "timeout_buffer_minutes": float(getattr(args, "timeout_buffer_minutes", 0.0)),
            "harvest_every_minutes": float(getattr(args, "harvest_every_minutes", 0.0)),
            "harvest_selection": args.harvest_selection,
            "procs": int(args.procs) if args.procs else None,
            "resume_from": args.resume_from,
        },
        "harvest_selection": args.harvest_selection,
        "run_status": "starting",
    }
    atomic_write_json(outdir / "manifest_partial.json", manifest)

    try:
        X_all, y_all, lev_all, sample_info = sample_training_table(input_file, args, args.selected_features, rng)
    except Exception as e:
        manifest["run_status"] = "sampling_failed"
        manifest["error"] = repr(e)
        manifest["traceback"] = traceback.format_exc()
        atomic_write_json(outdir / "manifest_partial.json", manifest)
        print("Sampling failed:", repr(e), file=sys.stderr)
        return 4

    manifest["sample_info"] = sample_info
    X_all, feature_standardization = standardize_sampled_features(X_all, args.selected_features, args.standardization)
    args.feature_standardization = feature_standardization
    manifest["feature_standardization"] = feature_standardization
    manifest["n_samples"] = int(len(y_all))
    manifest["levels_hpa_sampled"] = [float(z) for z in sorted(np.unique(lev_all.astype(float)))]

    print("Feature standardization:")
    for v in args.selected_features:
        st = feature_standardization[v]
        print(f"  {v}: mean={st['mean']:.6g}, std={st['std']:.6g}, method={st['method']}")

    if args.save_samples:
        np.savez_compressed(
            outdir / "samples_global.npz",
            X=X_all,
            y=y_all,
            lev_hpa=lev_all,
            features=np.array(args.selected_features),
            symbolic_features=np.array(args.selected_symbolic_features),
            feature_standardization_json=np.array(json.dumps(args.feature_standardization)),
        )

    manifest["run_status"] = "sampling_complete_before_fit"
    atomic_write_json(outdir / "manifest_partial.json", manifest)
    if not (outdir / "manifest_and_results.json").exists():
        atomic_write_json(outdir / "manifest_and_results.json", manifest)

    try:
        args.timeout_in_seconds = resolve_timeout_in_seconds(args)
        manifest["pysr_options"]["timeout_in_seconds"] = float(getattr(args, "timeout_in_seconds", 0.0))
        if args.timeout_in_seconds > 0.0:
            print(f"Resolved PySR timeout_in_seconds before fit={args.timeout_in_seconds:.0f}")
    except Exception as e:
        manifest["run_status"] = "failed_before_fit_timeout_resolution"
        manifest["error"] = str(e)
        atomic_write_json(outdir / "manifest_partial.json", manifest)
        print(str(e), file=sys.stderr)
        return 7

    install_harvest_signal_handlers(outdir, manifest)
    harvest_pysr_outputs(outdir, manifest, reason="before_fit")
    periodic_harvester = PeriodicHarvester(
        outdir,
        manifest,
        interval_seconds=max(0.0, float(args.harvest_every_minutes)) * 60.0,
    )

    print(f"\nFitting ONE global expression with {len(y_all)} pooled samples using engine={args.engine}")
    print(f"Using {X_all.shape[1]} feature(s): " + ", ".join(args.selected_symbolic_features))
    print(f"Target transform: {args.target_transform}" + (f" with log_eps={args.log_eps:g}" if args.target_transform == "log" else ""))
    print("Pressure-bracket diagnostics:", "disabled" if args.pressure_bracket_edges_resolved is None else ",".join(f"{x:g}" for x in args.pressure_bracket_edges_resolved))
    print("Wstd-bracket diagnostics:", "disabled" if args.wstd_bracket_edges_resolved is None else ",".join(f"{x:g}" for x in args.wstd_bracket_edges_resolved))
    print("Sampled levels:", ", ".join(f"{z:g} hPa" for z in sorted(np.unique(lev_all.astype(float)))))
    if args.engine == "pysr":
        print("PySR batching:", f"enabled={bool(args.batching)},", f"batch_size={int(args.batch_size)},", f"procs={int(args.procs) if args.procs else 'default'}")

    periodic_harvester.start()
    try:
        manifest["run_status"] = "fit_running"
        atomic_write_json(outdir / "manifest_partial.json", manifest)
        result = train_global(X_all, y_all, lev_all, args, outdir, rng)
    except SystemExit:
        harvest_pysr_outputs(outdir, manifest, reason="system_exit_during_fit")
        raise
    except BaseException as e:
        manifest["run_status"] = "fit_failed_or_interrupted"
        manifest["error"] = repr(e)
        manifest["traceback"] = traceback.format_exc()
        harvest_pysr_outputs(outdir, manifest, reason="exception_during_fit")
        print("\nFit failed or was interrupted. Latest PySR outputs, if any, were harvested.", file=sys.stderr)
        print(repr(e), file=sys.stderr)
        return 8
    finally:
        periodic_harvester.stop()
        harvest_pysr_outputs(outdir, manifest, reason="finally_after_fit_attempt")

    manifest["result"] = result
    manifest["run_status"] = "completed"
    atomic_write_json(outdir / "manifest_and_results.json", manifest)

    atomic_write_text(
        outdir / "expression_global.txt",
        result["expression"] + "\n\n"
        + "# Global test metrics, physical Wstd\n"
        + json.dumps(result["test_metrics_physical_Wstd_global"], indent=2) + "\n\n"
        + "# Test metrics by sampled pressure level, physical Wstd\n"
        + json.dumps(result["test_metrics_physical_Wstd_by_level_hpa"], indent=2) + "\n\n"
        + "# Test metrics by pressure bracket, physical Wstd\n"
        + json.dumps(result["test_metrics_physical_Wstd_by_pressure_bracket_hpa"], indent=2) + "\n\n"
        + "# Test metrics by true Wstd bracket, physical Wstd\n"
        + json.dumps(result["test_metrics_physical_Wstd_by_Wstd_bracket"], indent=2) + "\n",
    )
    harvest_pysr_outputs(outdir, manifest, reason="completed")

    print("\nExpression:")
    print(result["expression"])
    print("\nGlobal metrics:")
    print(json.dumps(result["test_metrics_physical_Wstd_global"], indent=2))
    print("\nMetrics by level:")
    print(json.dumps(result["test_metrics_physical_Wstd_by_level_hpa"], indent=2))
    print("\nMetrics by pressure bracket:")
    print(json.dumps(result["test_metrics_physical_Wstd_by_pressure_bracket_hpa"], indent=2))
    print("\nMetrics by true Wstd bracket:")
    print(json.dumps(result["test_metrics_physical_Wstd_by_Wstd_bracket"], indent=2))
    print(f"\nDone. Results written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
