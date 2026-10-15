#!/usr/bin/env python3
"""
Find the best equation index for each metric in a PySR/equation metrics CSV.

Robust to files that do NOT contain a row_index column:
  - uses row_index/equation_index/index if present
  - otherwise uses the CSV row number / pandas index as the equation index

Metric direction:
  - maximize correlations, scores, r2
  - minimize loss, rmse, mae, penalties, normalized absolute biases
  - minimize abs(value) for signed bias columns
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


MetricMode = Literal["max", "min", "absmin"]


NON_METRIC_COLUMNS = {
    "equation",
    "sympy_format",
    "lambda_format",
    "status",
    "pick",
    "notes",
}

COUNT_COLUMNS = {
    "n",
    "n_high",
    "n_extreme",
    "count",
    "samples",
    "n_samples",
}

INDEX_CANDIDATES = [
    "row_index",
    "equation_index",
    "equation_idx",
    "idx",
    "index",
]


def infer_index_column(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Return equation index series and a label describing where it came from.
    Falls back to the CSV row number if no explicit index column exists.
    """
    for col in INDEX_CANDIDATES:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype("Int64"), col

    # Common PySR hall_of_fame/equations_global files often do not store an
    # explicit row_index. In that case the equation index is the dataframe row.
    return pd.Series(df.index, index=df.index, dtype="Int64"), "csv_row_number"


def metric_mode(metric: str) -> MetricMode:
    m = metric.lower()

    # Signed bias: best means closest to zero.
    # Normalized bias columns below are usually magnitudes, handled as min.
    if m.startswith("bias_") and not m.endswith("_norm"):
        return "absmin"

    # Error/loss/penalty metrics: lower is better.
    minimize_terms = [
        "loss",
        "error",
        "rmse",
        "mae",
        "mse",
        "nrmse",
        "bias_norm",
        "_norm",
        "penalty",
        "complexity_penalty",
    ]
    if any(term in m for term in minimize_terms):
        return "min"

    # Skill metrics: higher is better.
    maximize_terms = [
        "score",
        "validation_score",
        "corr",
        "correlation",
        "r2",
        "r_squared",
        "skill",
        "explained_variance",
    ]
    if any(term in m for term in maximize_terms):
        return "max"

    # Complexity itself is usually not a validation metric; if included,
    # "best" generally means simpler.
    if m == "complexity":
        return "min"

    # Conservative default: higher is better.
    return "max"


def numeric_metric_columns(df: pd.DataFrame, include_counts: bool, include_complexity: bool) -> list[str]:
    metric_cols: list[str] = []

    for col in df.columns:
        if col in NON_METRIC_COLUMNS:
            continue
        if col in INDEX_CANDIDATES:
            continue
        if (not include_counts) and col in COUNT_COLUMNS:
            continue
        if (not include_complexity) and col == "complexity":
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            metric_cols.append(col)

    return metric_cols


def best_rows_for_metric(df: pd.DataFrame, metric: str, top: int) -> pd.DataFrame:
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = df.loc[values.notna()].copy()
    valid["_metric_value"] = values.loc[values.notna()]

    mode = metric_mode(metric)

    if mode == "max":
        valid = valid.sort_values("_metric_value", ascending=False)
        objective = valid["_metric_value"]
    elif mode == "min":
        valid = valid.sort_values("_metric_value", ascending=True)
        objective = valid["_metric_value"]
    elif mode == "absmin":
        valid["_abs_metric_value"] = valid["_metric_value"].abs()
        valid = valid.sort_values("_abs_metric_value", ascending=True)
        objective = valid["_abs_metric_value"]
    else:
        raise RuntimeError(f"Unhandled mode: {mode}")

    out = valid.head(top).copy()
    out["_rank_for_metric"] = np.arange(1, len(out) + 1)
    out["_selection_mode"] = mode
    out["_objective_value"] = objective.loc[out.index].to_numpy()

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find best equation indexes for each metric in an equations CSV."
    )
    parser.add_argument("csv_file", help="Input CSV, e.g. symbolic_Wstd_MERRA2/equations_global.csv")
    parser.add_argument("--out", default="best_equations_by_metric.csv", help="Output CSV summary")
    parser.add_argument("--top", type=int, default=1, help="Number of best equations to keep per metric")
    parser.add_argument(
        "--include-counts",
        action="store_true",
        help="Also consider count columns like n_high/n_extreme as metrics",
    )
    parser.add_argument(
        "--include-complexity",
        action="store_true",
        help="Also consider complexity itself as a metric; lower is treated as better",
    )
    parser.add_argument(
        "--index-col",
        default=None,
        help="Force a specific column to use as the equation index",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if args.index_col is not None:
        if args.index_col not in df.columns:
            raise ValueError(f"--index-col {args.index_col!r} not found in columns: {list(df.columns)}")
        eq_index = pd.to_numeric(df[args.index_col], errors="coerce").astype("Int64")
        index_source = args.index_col
    else:
        eq_index, index_source = infer_index_column(df)

    df = df.copy()
    df["_equation_index"] = eq_index

    if "equation" not in df.columns:
        # Some PySR files use a different case/name; make the output still useful.
        possible = [c for c in df.columns if c.lower() == "equation"]
        if possible:
            df["equation"] = df[possible[0]]
        else:
            df["equation"] = ""

    metric_cols = numeric_metric_columns(
        df,
        include_counts=args.include_counts,
        include_complexity=args.include_complexity,
    )

    if not metric_cols:
        raise ValueError("No numeric metric columns found.")

    results = []

    for metric in metric_cols:
        winners = best_rows_for_metric(df, metric, args.top)

        for _, row in winners.iterrows():
            results.append(
                {
                    "metric": metric,
                    "rank_for_metric": int(row["_rank_for_metric"]),
                    "selection_mode": row["_selection_mode"],
                    "equation_index": int(row["_equation_index"]) if pd.notna(row["_equation_index"]) else None,
                    "metric_value": row["_metric_value"],
                    "objective_value": row["_objective_value"],
                    "index_source": index_source,
                    "complexity": row.get("complexity", np.nan),
                    "loss": row.get("loss", np.nan),
                    "score": row.get("score", row.get("score_pysr", np.nan)),
                    "score_pysr": row.get("score_pysr", np.nan),
                    "validation_score": row.get("validation_score", np.nan),
                    "corr_global": row.get("corr_global", np.nan),
                    "corr_high": row.get("corr_high", np.nan),
                    "corr_extreme": row.get("corr_extreme", np.nan),
                    "rmse_global": row.get("rmse_global", np.nan),
                    "mae_global": row.get("mae_global", np.nan),
                    "bias_global": row.get("bias_global", np.nan),
                    "r2_global": row.get("r2_global", np.nan),
                    "equation": row.get("equation", ""),
                }
            )

    summary = pd.DataFrame(results)
    summary.to_csv(args.out, index=False)

    print(f"\nRead: {csv_path}")
    print(f"Equation index source: {index_source}")
    print(f"Wrote: {args.out}\n")

    width = max(len(m) for m in summary["metric"].astype(str))
    for _, row in summary.iterrows():
        if int(row["rank_for_metric"]) != 1:
            continue
        print(
            f"{row['metric']:{width}s}  "
            f"mode={row['selection_mode']:6s}  "
            f"idx={int(row['equation_index']):5d}  "
            f"value={row['metric_value']}"
        )

    print("\nKey winners:")
    key_metrics = [
        "validation_score",
        "corr_global",
        "corr_high",
        "corr_extreme",
        "r2_global",
        "loss",
        "rmse_global",
        "mae_global",
        "bias_global",
        "score",
        "score_pysr",
    ]
    for metric in key_metrics:
        hit = summary[(summary["metric"] == metric) & (summary["rank_for_metric"] == 1)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        print(
            f"  {metric:18s} -> idx {int(row['equation_index'])}, "
            f"value {row['metric_value']}, mode {row['selection_mode']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
