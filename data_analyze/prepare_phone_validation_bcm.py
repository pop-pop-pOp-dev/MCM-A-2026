"""
Convert BCMes smartphone battery JSON data into validation CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEVICE_CAPACITY_MAH = {
    "samsung galaxy s9": 3000,
    "huawei p8": 3400,
    "meizu m5c": 3000,
    "xiaomi redmi 4a": 3120,
    "xiaomi redmi 5a": 3000,
    "lenovo yoga tablet 2 8.0": 6400,
}


def _find_json_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.json") if p.is_file()]


def _pick_file(files: List[Path], device_hint: str | None) -> Path:
    if not files:
        raise ValueError("No JSON files found in dataset.")
    if device_hint:
        key = device_hint.lower()
        ref_matches = [p for p in files if "reference" in p.as_posix().lower() and key in p.as_posix().lower()]
        if ref_matches:
            return ref_matches[0]
        for p in files:
            if key in p.as_posix().lower():
                return p
    for p in files:
        if "s9" in p.as_posix().lower():
            return p
    return files[0]


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_capacity(device_hint: str | None, default_mah: int) -> int:
    if device_hint:
        key = device_hint.lower()
        for name, mah in DEVICE_CAPACITY_MAH.items():
            if name in key:
                return mah
    return default_mah


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BCMes validation CSV.")
    parser.add_argument("--input-dir", required=True, help="Extracted BCMes directory.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--device", default="Samsung Galaxy S9", help="Device hint for file selection.")
    parser.add_argument("--capacity-mah", type=int, default=3000, help="Default battery capacity (mAh).")
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine multiple traces (Reference/PoA/PoW) for the same device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    files = _find_json_files(root)

    if args.combine:
        key = args.device.lower().replace(" ", "")
        candidates = [
            p
            for p in files
            if key in p.as_posix().lower().replace(" ", "")
            and p.suffix.lower() == ".json"
        ]
        if not candidates:
            candidates = files
    else:
        candidates = [_pick_file(files, args.device)]

    cap_ah = _get_capacity(args.device, args.capacity_mah) / 1000.0
    all_rows: List[pd.DataFrame] = []
    offset = 0.0
    for idx, target in enumerate(sorted(candidates)):
        rows = _load_json(target)
        minutes = np.array(
            [
                float(
                    r.get("Minutes from start", r.get("Minutes from the start", 0.0))
                )
                for r in rows
            ],
            dtype=float,
        )
        level = np.array([float(r.get("Level", np.nan)) for r in rows], dtype=float)
        voltage_mv = np.array([float(r.get("Voltage", np.nan)) for r in rows], dtype=float)
        temp_c = np.array([float(r.get("Temperature", np.nan)) for r in rows], dtype=float)

        time_s = minutes * 60.0 + offset
        soc = np.clip(level / 100.0, 0.0, 1.0)
        voltage_v = voltage_mv / 1000.0

        current_a = np.zeros_like(time_s)
        for i in range(1, len(time_s)):
            dt = max(time_s[i] - time_s[i - 1], 1.0)
            dsoc = soc[i - 1] - soc[i]
            current_a[i] = max(dsoc * cap_ah * 3600.0 / dt, 0.0)
        if len(current_a) > 1:
            current_a[0] = current_a[1]

        df = pd.DataFrame(
            {
                "time_s": time_s,
                "current_a": current_a,
                "voltage_v": voltage_v,
                "temp_c": temp_c,
                "soc": soc,
                "segment": idx,
                "source_file": target.name,
            }
        )
        all_rows.append(df)
        offset = time_s[-1] + 300.0

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.out_csv, index=False)

    meta = {
        "source_json": [str(p.resolve()) for p in candidates],
        "device_hint": args.device,
        "capacity_mah": _get_capacity(args.device, args.capacity_mah),
        "rows": int(len(out)),
    }
    Path(args.out_csv).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out_csv}")
    print(f"Meta: {Path(args.out_csv).with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
