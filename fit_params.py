"""
Parameter estimation utility for OCV fitting.

Replace the data_soc/data_ocv arrays with points digitized from
published OCV curves or datasheets, then run this script to obtain
fit coefficients for config.yaml.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ==========================================
# 2. Combined OCV model
# ==========================================
def combined_ocv_model(soc: np.ndarray, k0: float, k1: float, k2: float, k3: float, k4: float) -> np.ndarray:
    """
    Combined Model:
    V = k0 - k1/soc - k2*soc + k3*ln(soc) + k4*ln(1-soc)
    """
    s = np.clip(soc, 0.001, 0.999)
    term1 = k0
    term2 = -k1 / s
    term3 = -k2 * s
    term4 = k3 * np.log(s)
    term5 = k4 * np.log(1.0 - s)
    return term1 + term2 + term3 + term4 + term5


def load_ocv_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        soc_vals = []
        ocv_vals = []
        for row in reader:
            if not row:
                continue
            soc = row.get("soc") or row.get("soc_frac") or row.get("SOC")
            ocv = row.get("ocv") or row.get("ocv_v") or row.get("voltage_v") or row.get("V")
            if soc is None or ocv is None:
                continue
            soc_vals.append(float(soc))
            ocv_vals.append(float(ocv))
    if not soc_vals:
        raise ValueError("No SOC/OCV columns found in CSV.")
    return np.array(soc_vals, dtype=float), np.array(ocv_vals, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Combined OCV model to SOC/OCV data.")
    parser.add_argument(
        "--ocv_csv",
        type=str,
        default=str(Path("datasets") / "calce_ocv" / "cs2_8_ocv_curve.csv"),
        help="CSV with columns soc and ocv/ocv_v/voltage_v.",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default="parameter_validation_ocv.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--p0",
        type=float,
        nargs=5,
        default=[4.0, 0.0, 0.0, 0.0, 0.0],
        help="Initial guess for [k0 k1 k2 k3 k4].",
    )
    args = parser.parse_args()

    data_soc, data_ocv = load_ocv_csv(Path(args.ocv_csv))
    p0 = list(args.p0)

    try:
        popt, _ = curve_fit(combined_ocv_model, data_soc, data_ocv, p0=p0, maxfev=10000)
        print("Fit success.")
        print("ocv_coeffs:", popt.tolist())
    except RuntimeError:
        print("Fit failed. Adjust p0 or data.")
        popt = np.array(p0, dtype=float)

    soc_smooth = np.linspace(0.001, 0.999, 200)
    v_fit = combined_ocv_model(soc_smooth, *popt)

    plt.figure(figsize=(8, 5))
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.scatter(data_soc, data_ocv, color="red", label="Experimental Data (Extracted)")
    plt.plot(soc_smooth, v_fit, color="blue", linewidth=2, label="Combined Model Fit")
    plt.xlabel("State of Charge (SOC)")
    plt.ylabel("Open Circuit Voltage (V)")
    plt.title("Parameter Estimation: OCV Curve Fitting")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.plot_path, dpi=300)
    plt.show()

    rmse = float(np.sqrt(np.mean((combined_ocv_model(data_soc, *popt) - data_ocv) ** 2)))
    print(f"Fitting RMSE: {rmse:.4f} V")


if __name__ == "__main__":
    main()
