"""
Parameter estimation utility for OCV fitting.

Replace the data_soc/data_ocv arrays with points digitized from
published OCV curves or datasheets, then run this script to obtain
fit coefficients for config.yaml.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ==========================================
# 1. Data (replace with extracted points)
# ==========================================
data_soc = np.array(
    [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
)
data_ocv = np.array(
    [4.18, 4.12, 4.08, 4.02, 3.95, 3.88, 3.82, 3.75, 3.68, 3.58, 3.45, 3.35, 3.20, 3.00]
)


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


# ==========================================
# 3. Fit
# ==========================================
p0 = [4.0, 0.0, 0.0, 0.0, 0.0]

try:
    popt, pcov = curve_fit(combined_ocv_model, data_soc, data_ocv, p0=p0, maxfev=10000)
    print("Fit success.")
    print("ocv_coeffs:", popt.tolist())
except RuntimeError:
    print("Fit failed. Adjust p0 or data.")
    popt = np.array(p0, dtype=float)


# ==========================================
# 4. Validation plot
# ==========================================
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
plt.savefig("parameter_validation_ocv.png", dpi=300)
plt.show()

rmse = float(np.sqrt(np.mean((combined_ocv_model(data_soc, *popt) - data_ocv) ** 2)))
print(f"Fitting RMSE: {rmse:.4f} V")
