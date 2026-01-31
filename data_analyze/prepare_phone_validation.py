"""
Convert WLTP parquet data into validation CSV: time_s,current_a,voltage_v,temp_c.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _pick_columns(columns: Iterable[str], keywords: Iterable[str]) -> list[str]:
    lowered = {c: c.lower() for c in columns}
    matches = []
    for col, low in lowered.items():
        if any(k in low for k in keywords):
            matches.append(col)
    return matches


def _choose_best(columns: list[str], prefer: Iterable[str]) -> str | None:
    for key in prefer:
        for col in columns:
            if key in col.lower():
                return col
    return columns[0] if columns else None


def _to_time_seconds(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.number):
        return series.to_numpy(dtype=float)
    ts = pd.to_datetime(series, errors="coerce")
    if ts.isna().all():
        raise ValueError("Unable to parse time column to timestamps.")
    t0 = ts.iloc[0]
    return (ts - t0).dt.total_seconds().to_numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare phone validation CSV from parquet.")
    parser.add_argument("--parquet", required=True, help="Path to parquet input.")
    parser.add_argument("--out-csv", required=True, help="Path to output CSV.")
    parser.add_argument("--time-col", default=None, help="Optional time column name.")
    parser.add_argument("--current-col", default=None, help="Optional current column name.")
    parser.add_argument("--voltage-col", default=None, help="Optional voltage column name.")
    parser.add_argument("--temp-col", default=None, help="Optional temperature column name.")
    parser.add_argument("--series", type=int, default=12, help="Series cells (default 12).")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel branches (default 3).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.parquet)
    cols = list(df.columns)

    time_candidates = _pick_columns(cols, ["time", "timestamp", "datetime"])
    time_col = args.time_col or _choose_best(time_candidates, ["timestamp", "time"])
    if not time_col:
        raise ValueError("No time column detected. Provide --time-col.")

    current_candidates = _pick_columns(cols, ["current", "i"])
    current_col = args.current_col or _choose_best(
        current_candidates,
        ["current_actual_p1", "current_actual_p2", "current_actual_p3", "branch", "pack", "total", "current"],
    )
    if not current_col:
        raise ValueError("No current column detected. Provide --current-col.")

    voltage_candidates = _pick_columns(cols, ["voltage", "v"])
    voltage_cell_cols = [c for c in voltage_candidates if "cell" in c.lower()]
    voltage_avg_cell = _choose_best(voltage_candidates, ["voltage_avg_cell", "avg_cell"])
    if args.voltage_col:
        voltage_col = args.voltage_col
        voltage_mode = "direct"
    elif voltage_avg_cell:
        voltage_col = voltage_avg_cell
        voltage_mode = "cell_avg_column"
    elif voltage_cell_cols:
        voltage_col = None
        voltage_mode = "cell_average"
    else:
        voltage_col = _choose_best(voltage_candidates, ["pack", "total", "branch", "voltage"])
        voltage_mode = "pack_scaled"
    if not voltage_candidates:
        raise ValueError("No voltage column detected. Provide --voltage-col.")

    temp_candidates = _pick_columns(cols, ["temp"])
    temp_col = args.temp_col or _choose_best(temp_candidates, ["avg", "mean", "ambient", "temp"])

    soc_candidates = _pick_columns(cols, ["soc"])
    soc_col = _choose_best(soc_candidates, ["soc_actual", "soc"])

    time_s = _to_time_seconds(df[time_col])
    current_a = df[current_col].to_numpy(dtype=float)
    voltage_v: np.ndarray

    if voltage_mode == "cell_average":
        voltage_v = df[voltage_cell_cols].mean(axis=1).to_numpy(dtype=float)
        current_a = current_a / max(args.parallel, 1)
    elif voltage_mode == "cell_avg_column":
        voltage_v = df[voltage_col].to_numpy(dtype=float)
        current_a = current_a / max(args.parallel, 1)
    elif voltage_mode == "pack_scaled":
        voltage_v = df[voltage_col].to_numpy(dtype=float) / max(args.series, 1)
        current_a = current_a / max(args.parallel, 1)
    else:
        voltage_v = df[voltage_col].to_numpy(dtype=float)

    if temp_col:
        temp_c = df[temp_col].to_numpy(dtype=float)
    else:
        temp_c = np.full_like(current_a, np.nan, dtype=float)

    payload = {
        "time_s": time_s,
        "current_a": current_a,
        "voltage_v": voltage_v,
        "temp_c": temp_c,
    }
    if soc_col:
        payload["soc"] = df[soc_col].to_numpy(dtype=float)

    out = pd.DataFrame(payload)
    out.to_csv(args.out_csv, index=False)

    meta = {
        "source_parquet": str(Path(args.parquet).resolve()),
        "time_col": time_col,
        "current_col": current_col,
        "voltage_mode": voltage_mode,
        "voltage_col": voltage_col,
        "voltage_cell_cols": voltage_cell_cols,
        "temp_col": temp_col,
        "soc_col": soc_col,
        "series": args.series,
        "parallel": args.parallel,
        "rows": int(len(out)),
    }
    meta_path = Path(args.out_csv).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out_csv}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
