"""
Derive thermal/electrical parameters from Panasonic 18650PF dataset.

Outputs:
- Ea_R (Arrhenius for internal resistance)
- Ea_cap (Arrhenius for capacity temperature factor)

Default input paths point to Cycle_1 files at -10/0/10/25C.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


R_GAS = 8.314
T_REF = 298.15


def load_mat_cycle(path: Path) -> Dict[str, np.ndarray]:
    data = loadmat(path)
    meas = data["meas"][0, 0]
    time_s = np.asarray(meas["Time"]).squeeze().astype(float)
    voltage = np.asarray(meas["Voltage"]).squeeze().astype(float)
    current = np.asarray(meas["Current"]).squeeze().astype(float)
    temp_c = np.asarray(meas["Battery_Temp_degC"]).squeeze().astype(float)
    return {"time_s": time_s, "voltage_v": voltage, "current_a": current, "temp_c": temp_c}


def estimate_internal_resistance(data: Dict[str, np.ndarray], di_thresh: float = 0.5) -> float:
    v = data["voltage_v"]
    i = data["current_a"]
    dv = np.diff(v)
    di = np.diff(i)
    mask = np.abs(di) >= di_thresh
    if not np.any(mask):
        return float("nan")
    r_inst = np.abs(dv[mask] / di[mask])
    r_inst = r_inst[np.isfinite(r_inst)]
    return float(np.median(r_inst)) if r_inst.size else float("nan")


def estimate_capacity_ah(data: Dict[str, np.ndarray], discharge_threshold: float = -0.1) -> float:
    t = data["time_s"]
    i = data["current_a"]
    dt = np.diff(t, prepend=t[0])
    mask = i < discharge_threshold
    ah = np.sum((-i[mask]) * dt[mask] / 3600.0)
    return float(ah)


def fit_arrhenius(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # Fit y = a * x + b
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive Ea_R and Ea_cap from Panasonic dataset.")
    parser.add_argument(
        "--file",
        action="append",
        nargs=2,
        metavar=("TEMP_C", "PATH"),
        help="Temperature and .mat path. Can be provided multiple times.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="outputs",
        help="Directory to save diagnostic plots.",
    )
    args = parser.parse_args()

    if not args.file:
        base = Path("datasets") / "panasonic_temp" / "Panasonic-18650PF-Data-master" / "Panasonic-18650PF-Data-master"
        args.file = [
            ("-10", str(base / "-10degC" / "Drive Cycles" / "06-10-17_11.25 n10degC_Cycle_1_Pan18650PF.mat")),
            ("0", str(base / "0degC" / "Drive cycles" / "05-30-17_12.56 0degC_Cycle_1_Pan18650PF.mat")),
            ("10", str(base / "10degC" / "Drive Cycles" / "03-28-17_12.51 10degC_Cycle_1_Pan18650PF.mat")),
            ("25", str(base / "Panasonic 18650PF Data" / "25degC" / "Drive cycles" / "03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat")),
        ]

    temps_c: List[float] = []
    resistances: List[float] = []
    capacities: List[float] = []

    for temp_c_str, path_str in args.file:
        temp_c = float(temp_c_str)
        path = Path(path_str)
        data = load_mat_cycle(path)
        r_val = estimate_internal_resistance(data)
        c_val = estimate_capacity_ah(data)
        temps_c.append(temp_c)
        resistances.append(r_val)
        capacities.append(c_val)
        print(f"{temp_c:+}C  R~{r_val:.4f} ohm  Q~{c_val:.3f} Ah  ({path.name})")

    temps_k = np.array(temps_c) + 273.15
    resistances = np.array(resistances)
    capacities = np.array(capacities)

    # Fit Ea_R from ln(R/R_ref) vs (1/T - 1/T_ref)
    ref_idx = int(np.argmin(np.abs(temps_k - T_REF)))
    r_ref = resistances[ref_idx]
    x_r = (1.0 / temps_k) - (1.0 / T_REF)
    y_r = np.log(resistances / r_ref)
    slope_r, _ = fit_arrhenius(x_r, y_r)
    ea_r = slope_r * R_GAS

    # Fit Ea_cap from ln(C/C_ref) vs (1/T - 1/T_ref)
    c_ref = capacities[ref_idx]
    x_c = (1.0 / temps_k) - (1.0 / T_REF)
    y_c = np.log(capacities / c_ref)
    slope_c, _ = fit_arrhenius(x_c, y_c)
    ea_cap = -slope_c * R_GAS

    print(f"Estimated Ea_R: {ea_r:.1f} J/mol")
    print(f"Estimated Ea_cap: {ea_cap:.1f} J/mol")
    print(f"Reference R_ref at {T_REF:.2f}K: {r_ref:.4f} ohm")

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.4, 4.0))
    plt.scatter(x_r, y_r, label="Data")
    plt.plot(x_r, slope_r * x_r + 0.0, linestyle="--", label="Fit")
    plt.xlabel("1/T - 1/T_ref (1/K)")
    plt.ylabel("ln(R/R_ref)")
    plt.title("Arrhenius Fit for Resistance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "fit_ea_r.png", dpi=300)

    plt.figure(figsize=(6.4, 4.0))
    plt.scatter(x_c, y_c, label="Data")
    plt.plot(x_c, slope_c * x_c + 0.0, linestyle="--", label="Fit")
    plt.xlabel("1/T - 1/T_ref (1/K)")
    plt.ylabel("ln(C/C_ref)")
    plt.title("Arrhenius Fit for Capacity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "fit_ea_cap.png", dpi=300)


if __name__ == "__main__":
    main()
