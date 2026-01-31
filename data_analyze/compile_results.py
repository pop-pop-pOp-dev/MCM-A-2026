"""
Compile conclusions, metrics, and visuals for the report.
Outputs:
  data_analyze/results_summary.csv
  data_analyze/results_summary.json
  data_analyze/parameter_coverage.csv
  data_analyze/conclusions.md
  data_analyze/pic/*.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.append(str(ROOT))
import solution  # noqa: E402
OUT_DIR = ROOT / "data_analyze"
PIC_DIR = OUT_DIR / "pic"


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
            "grid.alpha": 0.25,
            "axes.facecolor": "#f8f9fb",
            "figure.facecolor": "white",
            "axes.edgecolor": "#202020",
        }
    )


def load_params() -> Dict[str, Any]:
    with (ROOT / "parameters.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config() -> Dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return json.load(handle) if handle.name.endswith(".json") else __import__("yaml").safe_load(handle)


def summarize_output(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)
    dt = float(df.t_s.iloc[1] - df.t_s.iloc[0]) if len(df) > 1 else 0.0
    energy_wh = float(np.trapz(df.p_total, df.t_s) / 3600.0)
    return {
        "scenario": path.stem,
        "t_end_s": float(df.t_s.iloc[-1]),
        "soc_end": float(df.soc.iloc[-1]),
        "v_end": float(df.v_term.iloc[-1]),
        "temp_max_k": float(df.temp_k.max()),
        "temp_min_k": float(df.temp_k.min()),
        "soh_drop": float(df.soh.iloc[0] - df.soh.iloc[-1]),
        "avg_power_w": float(df.p_total.mean()),
        "energy_wh": energy_wh,
        "dt_s": dt,
        "n_points": int(len(df)),
    }


def parameter_coverage(params: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, entry in params["battery_params"].items():
        source = str(entry.get("source", "")).lower()
        if "smartphone_parameters_detailed.csv" in source or "fit" in source or "panasonic" in source:
            status = "data_driven"
        elif "model default" in source:
            status = "assumed"
        elif "physical constant" in source:
            status = "constant"
        else:
            status = "assumed"
        rows.append(
            {
                "parameter": key,
                "value": entry.get("value"),
                "units": entry.get("units"),
                "source": entry.get("source"),
                "status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "parameter"])


def validation_metrics(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    validation = cfg.get("validation", {})
    if not validation.get("enabled", False):
        return None
    data_paths = cfg.get("data_paths", {})
    mode = str(validation.get("mode", "current_profile")).lower()
    params = solution.BatteryParams.from_dict(cfg["battery_params"])

    if mode == "ocv_curve":
        ocv_path = validation.get("ocv_csv") or data_paths.get("calce_ocv_csv")
        if not ocv_path:
            return None
        ocv_path = Path(ocv_path)
        if not ocv_path.is_absolute():
            ocv_path = ROOT / ocv_path
        _, v_pred, v_meas = solution.validate_ocv_curve(params, ocv_path)
        rmse = float(np.sqrt(np.mean((v_pred - v_meas) ** 2)))
        mae = float(np.mean(np.abs(v_pred - v_meas)))
        return {
            "profile": ocv_path.name,
            "rmse_v": rmse,
            "mae_v": mae,
            "n": int(len(v_meas)),
            "mode": "ocv_curve",
        }

    profile = validation.get("current_profile_csv") or data_paths.get("nasa_random_csv")
    if not profile:
        return None
    profile_path = Path(profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path

    calibrate = bool(validation.get("calibrate", False))
    scale_bounds = validation.get("current_scale_bounds", [0.5, 1.5])
    scale_bounds = (float(scale_bounds[0]), float(scale_bounds[1]))
    use_soc_from_csv = bool(validation.get("use_soc_from_csv", False))
    time_s, v_pred, v_meas, meta = solution.validate_current_profile(
        params,
        profile_path,
        True,
        calibrate=calibrate,
        current_scale_bounds=scale_bounds,
        use_soc_from_csv=use_soc_from_csv,
    )
    rmse = float(np.sqrt(np.mean((v_pred - v_meas) ** 2)))
    mae = float(np.mean(np.abs(v_pred - v_meas)))
    return {
        "profile": profile_path.name,
        "rmse_v": rmse,
        "mae_v": mae,
        "n": int(len(v_meas)),
        "mode": "current_profile",
        "current_scale": meta.get("current_scale", 1.0),
        "r_scale": meta.get("r_scale", 1.0),
        "ocv_scale": meta.get("ocv_scale", 1.0),
        "voltage_offset": meta.get("voltage_offset", 0.0),
    }


def plot_summary(summary: pd.DataFrame) -> None:
    setup_style()
    PIC_DIR.mkdir(parents=True, exist_ok=True)

    # Scenario comparison: time-to-empty
    plt.figure(figsize=(6.8, 4.2))
    ax = plt.gca()
    sns.barplot(data=summary, x="scenario", y="t_end_s", palette="mako", ax=ax)
    ax.set_ylabel("Time to empty (s)")
    ax.set_xlabel("")
    ax.set_title("Scenario End Time")
    for rect, val in zip(ax.patches, summary.t_end_s):
        ax.annotate(
            f"{val:.0f}",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(PIC_DIR / "scenario_time_to_empty.png", dpi=300)
    plt.close()

    # Scenario energy
    plt.figure(figsize=(6.8, 4.2))
    ax = plt.gca()
    sns.barplot(data=summary, x="scenario", y="energy_wh", palette="crest", ax=ax)
    ax.set_ylabel("Energy used (Wh)")
    ax.set_xlabel("")
    ax.set_title("Scenario Energy Consumption")
    for rect, val in zip(ax.patches, summary.energy_wh):
        ax.annotate(
            f"{val:.2f}",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(PIC_DIR / "scenario_energy.png", dpi=300)
    plt.close()

    # Scenario temperature ranges
    plt.figure(figsize=(6.8, 4.2))
    ax = plt.gca()
    ax.errorbar(
        summary.scenario,
        summary.temp_max_k,
        yerr=(summary.temp_max_k - summary.temp_min_k),
        fmt="o",
        color="#2a6fdb",
        capsize=3,
    )
    ax.set_ylabel("Temperature (K)")
    ax.set_xlabel("")
    ax.set_title("Temperature Range by Scenario")
    for name, val in zip(summary.scenario, summary.temp_max_k):
        ax.annotate(
            f"({name}, {val:.1f})",
            (name, val),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(PIC_DIR / "scenario_temp_range.png", dpi=300)
    plt.close()


def write_conclusions(summary: pd.DataFrame, coverage: pd.DataFrame, validation: Dict[str, Any] | None) -> None:
    lines: List[str] = [
        "# Model Conclusions",
        "",
        "## Scenario Summary",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['scenario']}: t_end={row['t_end_s']:.0f}s, "
            f"energy={row['energy_wh']:.2f}Wh, "
            f"v_end={row['v_end']:.3f}V, soc_end={row['soc_end']:.3f}"
        )

    lines += [
        "",
        "## Validation",
    ]
    if validation:
        mode = validation.get("mode", "current_profile")
        extra = ""
        if "current_scale" in validation:
            extra = (
                f", scale={validation['current_scale']:.3f}, r={validation['r_scale']:.3f}, "
                f"ocv={validation['ocv_scale']:.3f}, offset={validation['voltage_offset']:.3f}V"
            )
        lines.append(
            f"- Mode={mode}, RMSE={validation['rmse_v']:.4f} V, "
            f"MAE={validation['mae_v']:.4f} V on {validation['n']} samples "
            f"(profile: {validation['profile']}{extra})."
        )
    else:
        lines.append("- Validation not enabled or data missing.")

    lines += [
        "",
        "## Parameter Coverage",
        f"- data_driven: {int((coverage.status=='data_driven').sum())}",
        f"- assumed: {int((coverage.status=='assumed').sum())}",
        f"- constant: {int((coverage.status=='constant').sum())}",
        "",
        "## Notes",
        "- Metrics are derived from generated outputs and current config parameters.",
    ]

    (OUT_DIR / "conclusions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PIC_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    outputs_dir = ROOT / "outputs"
    for name in ["Video_Streaming.csv", "Gaming.csv", "Winter_Usage.csv"]:
        path = outputs_dir / name
        if path.exists():
            summary_rows.append(summarize_output(path))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "results_summary.csv", index=False)
    (OUT_DIR / "results_summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    params = load_params()
    coverage = parameter_coverage(params)
    coverage.to_csv(OUT_DIR / "parameter_coverage.csv", index=False)

    cfg = load_config()
    validation = validation_metrics(cfg)

    plot_summary(summary)
    write_conclusions(summary, coverage, validation)


if __name__ == "__main__":
    main()
