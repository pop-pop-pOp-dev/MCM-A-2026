"""
Generate parameter analysis report and figures.
Outputs:
  data_analyze/result.md
  data_analyze/pic/*.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data_analyze"
PIC_DIR = OUT_DIR / "pic"
PARAM_PATH = ROOT / "parameters.json"


def setup_style() -> None:
    try:
        plt.style.use(["science", "no-latex", "grid"])
    except Exception:
        sns.set_context("paper")
        sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.25,
            "axes.facecolor": "#f8f9fb",
            "figure.facecolor": "white",
            "axes.edgecolor": "#202020",
        }
    )


def load_params(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def combined_ocv(soc: np.ndarray, coeffs: List[float]) -> np.ndarray:
    k0, k1, k2, k3, k4 = coeffs
    s = np.clip(soc, 0.001, 0.999)
    return k0 - k1 / s - k2 * s + k3 * np.log(s) + k4 * np.log(1.0 - s)


def format_value(val: Any) -> str:
    if isinstance(val, float):
        if abs(val) >= 1e4 or (abs(val) > 0 and abs(val) < 1e-3):
            return f"{val:.3e}"
        return f"{val:.4g}"
    return str(val)


def collect_params(p: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    for key in sorted(p.keys()):
        entry = p[key]
        value = format_value(entry.get("value"))
        units = entry.get("units", "")
        source = entry.get("source", "").encode("ascii", "ignore").decode("ascii")
        rows.append((key, value, units, source))
    return rows


def write_params_csv(rows: List[Tuple[str, str, str, str]], out_path: Path) -> None:
    lines = ["parameter,value,units,source"]
    for key, value, units, source in rows:
        source_safe = source.replace('"', "'")
        lines.append(f'{key},"{value}","{units}","{source_safe}"')
    out_path.write_text("\n".join(lines), encoding="utf-8")


def annotate_bars(ax: plt.Axes, values: List[float], errs: np.ndarray | None = None) -> None:
    if errs is None:
        errs = np.zeros(len(values))
    for rect, val, err in zip(ax.patches, values, errs):
        height = rect.get_height()
        ax.annotate(
            format_value(val),
            (rect.get_x() + rect.get_width() / 2, height + err),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def polish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6)
    ax.tick_params(axis="x", length=0)


def relative_err(values: List[float], frac: float = 0.05) -> np.ndarray:
    return np.array([abs(v) * frac for v in values], dtype=float)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PIC_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    data = load_params(PARAM_PATH)
    p = data["battery_params"]

    # Derived metrics
    tau_s = (p["m"]["value"] * p["Cp"]["value"]) / max(p["hA"]["value"], 1e-9)
    screen_max_w = p["P_max_scr"]["value"] * p["A_area"]["value"]
    nominal_v = 3.85
    e_design_wh = p["Q_design_ah"]["value"] * nominal_v

    # Report
    report_lines = [
        "# Parameter Analysis Report",
        "",
        "## Key Derived Metrics",
        f"- Thermal time constant: **{tau_s:.1f} s**",
        f"- Screen max power (from density × area): **{screen_max_w:.2f} W**",
        f"- Nominal energy estimate (Q_design × 3.85V): **{e_design_wh:.2f} Wh**",
        "",
        "## Battery Parameters (Selected)",
        "",
        "| Parameter | Value | Units | Source |",
        "|---|---:|---|---|",
        f"| Q_design_ah | {p['Q_design_ah']['value']} | Ah | {p['Q_design_ah']['source']} |",
        f"| V_cutoff | {p['V_cutoff']['value']} | V | {p['V_cutoff']['source']} |",
        f"| R_ref | {p['R_ref']['value']} | Ohm | {p['R_ref']['source']} |",
        f"| Ea_R | {p['Ea_R']['value']} | J/mol | {p['Ea_R']['source']} |",
        f"| Ea_cap | {p['Ea_cap']['value']} | J/mol | {p['Ea_cap']['source']} |",
        f"| k_cycle | {p['k_cycle']['value']} | 1/(A*s) | {p['k_cycle']['source']} |",
        f"| k_cal | {p['k_cal']['value']} | 1/s | {p['k_cal']['source']} |",
        "",
        "## Power Parameters (Selected)",
        "",
        "| Parameter | Value | Units | Source |",
        "|---|---:|---|---|",
        f"| P_idle | {p['P_idle']['value']} | W | {p['P_idle']['source']} |",
        f"| P_little_max | {p['P_little_max']['value']} | W | {p['P_little_max']['source']} |",
        f"| P_big_max | {p['P_big_max']['value']} | W | {p['P_big_max']['source']} |",
        f"| wifi_idle_power | {p['wifi_idle_power']['value']} | W | {p['wifi_idle_power']['source']} |",
        f"| wifi_active_power | {p['wifi_active_power']['value']} | W | {p['wifi_active_power']['source']} |",
        f"| gps_on_power | {p['gps_on_power']['value']} | W | {p['gps_on_power']['source']} |",
        "",
        "## Notes",
        "- OCV curve uses the Combined Model parameters in `parameters.json`.",
        "- Derived metrics computed from provided parameters; update inputs to refresh.",
        "- Full parameter table exported to `data_analyze/parameters_table.csv`.",
        "",
    ]

    (OUT_DIR / "result.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Plot 1: Power parameters
    labels = ["CPU idle", "CPU little (avg)", "CPU big (peak)", "WiFi idle", "WiFi active", "GPS on"]
    values = [
        p["P_idle"]["value"],
        p["P_little_max"]["value"],
        p["P_big_max"]["value"],
        p["wifi_idle_power"]["value"],
        p["wifi_active_power"]["value"],
        p["gps_on_power"]["value"],
    ]
    colors = sns.color_palette("mako", len(values))
    plt.figure(figsize=(7.2, 4.2))
    ax = plt.gca()
    errs = relative_err(values)
    ax.bar(labels, values, color=colors, width=0.55, yerr=errs, capsize=3)
    plt.ylabel("Power (W)")
    plt.title("Component Power Parameters")
    plt.xticks(rotation=20, ha="right")
    annotate_bars(ax, values, errs)
    polish_axes(ax)
    plt.tight_layout()
    plt.savefig(PIC_DIR / "power_params.png", dpi=300)
    plt.close()

    # Plot 2: Thermal parameters
    t_labels = ["Mass", "Heat capacity", "Convective hA", "Thermal tau"]
    t_values = [p["m"]["value"], p["Cp"]["value"], p["hA"]["value"], tau_s]
    colors = sns.color_palette("crest", len(t_values))
    plt.figure(figsize=(6.8, 4.2))
    ax = plt.gca()
    errs = relative_err(t_values)
    ax.bar(t_labels, t_values, color=colors, width=0.55, yerr=errs, capsize=3)
    ax.set_ylabel("Value (SI units)")
    plt.title("Thermal Parameters")
    annotate_bars(ax, t_values, errs)
    polish_axes(ax)
    plt.tight_layout()
    plt.savefig(PIC_DIR / "thermal_params.png", dpi=300)
    plt.close()

    # Plot 3: Arrhenius energies
    e_labels = ["Resistance activation", "Capacity activation", "Calendar activation"]
    e_values = [p["Ea_R"]["value"], p["Ea_cap"]["value"], p["Ea_cal"]["value"]]
    colors = sns.color_palette("rocket", len(e_values))
    plt.figure(figsize=(6.4, 4.0))
    ax = plt.gca()
    errs = relative_err(e_values)
    ax.bar(e_labels, e_values, color=colors, width=0.55, yerr=errs, capsize=3)
    plt.ylabel("Activation energy (J/mol)")
    plt.title("Activation Energies")
    annotate_bars(ax, e_values, errs)
    polish_axes(ax)
    plt.tight_layout()
    plt.savefig(PIC_DIR / "activation_energies.png", dpi=300)
    plt.close()

    # Plot 4: OCV curve
    if p["ocv_model"]["value"] == "combined":
        soc = np.linspace(0.01, 0.99, 200)
        v = combined_ocv(soc, p["ocv_coeffs"]["value"])
        plt.figure(figsize=(6.8, 4.2))
        plt.plot(soc, v, color="#2a6fdb", linewidth=1.2, label="Combined OCV")
        band = 0.02
        plt.fill_between(soc, v - band, v + band, color="#2a6fdb", alpha=0.12, label="±0.02 V band")
        key_soc = np.array([0.1, 0.5, 0.9])
        key_v = combined_ocv(key_soc, p["ocv_coeffs"]["value"])
        plt.scatter(key_soc, key_v, color="#ff8c3a", s=22, zorder=3, label="Key SOC points")
        for x, y in zip(key_soc, key_v):
            plt.annotate(f"({x:.2f}, {y:.3f})", (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
        plt.xlabel("SOC (fraction)")
        plt.ylabel("OCV (V)")
        plt.title("Combined OCV Curve")
        plt.legend()
        polish_axes(plt.gca())
        plt.tight_layout()
        plt.savefig(PIC_DIR / "ocv_curve.png", dpi=300)
        plt.close()

    # Plot 5: Core battery parameters
    core_labels = ["Capacity", "Cutoff voltage", "Ref resistance"]
    core_values = [p["Q_design_ah"]["value"], p["V_cutoff"]["value"], p["R_ref"]["value"]]
    colors = sns.color_palette("flare", len(core_values))
    plt.figure(figsize=(6.4, 4.0))
    ax = plt.gca()
    errs = relative_err(core_values)
    ax.bar(core_labels, core_values, color=colors, width=0.55, yerr=errs, capsize=3)
    ax.set_ylabel("Value (Ah / V / Ohm)")
    plt.title("Core Battery Parameters")
    annotate_bars(ax, core_values, errs)
    polish_axes(ax)
    plt.tight_layout()
    plt.savefig(PIC_DIR / "core_battery_params.png", dpi=300)
    plt.close()

    # Plot 6: 3D surface of internal resistance vs SOC & Temp
    soc_grid = np.linspace(0.05, 0.95, 40)
    temp_c_grid = np.linspace(-10, 40, 40)
    soc_mesh, temp_mesh = np.meshgrid(soc_grid, temp_c_grid)
    temp_k = temp_mesh + 273.15
    r_ref = p["R_ref"]["value"]
    t_ref = p["T_ref"]["value"]
    ea_r = p["Ea_R"]["value"]
    r_gas = p["R_gas"]["value"]
    r_soc_coeff = p["r_soc_coeff"]["value"]
    r_soc_k = p["r_soc_k"]["value"]
    arr = np.exp(ea_r / r_gas * (1.0 / temp_k - 1.0 / t_ref))
    soc_factor = 1.0 + r_soc_coeff * np.exp(-r_soc_k * soc_mesh)
    r_surface = r_ref * arr * soc_factor
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(soc_mesh, temp_mesh, r_surface, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("SOC (fraction)")
    ax.set_ylabel("Temperature (C)")
    ax.set_zlabel("R_int (Ohm)")
    ax.set_title("Internal Resistance Surface")
    plt.tight_layout()
    plt.savefig(PIC_DIR / "rint_surface_3d.png", dpi=300)
    plt.close()

    # Plot 7: Power comparison (linear + log)
    plt.figure(figsize=(7.2, 4.2))
    ax = plt.gca()
    ax.plot(labels, values, marker="o", color="#1b998b", linewidth=1.2)
    for i, v in enumerate(values):
        ax.annotate(format_value(v), (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    plt.ylabel("Power (W)")
    plt.title("Power Parameters (Line Comparison)")
    plt.xticks(rotation=20, ha="right")
    polish_axes(ax)
    plt.tight_layout()
    plt.savefig(PIC_DIR / "power_params_line.png", dpi=300)
    plt.close()

    # Full parameter table CSV
    rows = collect_params(p)
    write_params_csv(rows, OUT_DIR / "parameters_table.csv")


if __name__ == "__main__":
    main()
