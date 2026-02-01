"""
Batch runner for baseline + ablation experiments.
Outputs are stored under outputs/experiments/<experiment_id>/.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.append(str(ROOT))
import solution  # noqa: E402


@dataclass
class Experiment:
    exp_id: str
    label: str
    group: str
    overrides: Dict[str, Any]


def load_config() -> Dict[str, Any]:
    cfg_path = ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        return __import__("yaml").safe_load(handle)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def summarize_output(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)
    if "t_s" not in df.columns:
        raise ValueError(f"Missing t_s column in {path.name}")
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
    _, v_pred, v_meas, meta = solution.validate_current_profile(
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


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def absolutize_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_paths = cfg.get("data_paths", {})
    abs_paths = {}
    for key, value in data_paths.items():
        if not value:
            abs_paths[key] = value
            continue
        p = Path(str(value))
        abs_paths[key] = str(p if p.is_absolute() else ROOT / p)
    if abs_paths:
        cfg["data_paths"] = abs_paths

    validation = cfg.get("validation", {})
    if "current_profile_csv" in validation and validation["current_profile_csv"]:
        p = Path(str(validation["current_profile_csv"]))
        validation["current_profile_csv"] = str(p if p.is_absolute() else ROOT / p)
    if "ocv_csv" in validation and validation["ocv_csv"]:
        p = Path(str(validation["ocv_csv"]))
        validation["ocv_csv"] = str(p if p.is_absolute() else ROOT / p)
    cfg["validation"] = validation
    return cfg


def build_experiments() -> List[Experiment]:
    return [
        Experiment(
            exp_id="baseline_ocv_r",
            label="Baseline (OCV-R only, no temp corr, no RC, no calib)",
            group="baseline",
            overrides={
                "battery_params": {"rc_enabled": False, "ocv_temp_coeff": 0.0, "ocv_temp_quad": 0.0},
                "validation": {"calibrate": False},
            },
        ),
        Experiment(
            exp_id="temp_only",
            label="Temp correction only (no RC, no calib)",
            group="baseline",
            overrides={
                "battery_params": {"rc_enabled": False},
                "validation": {"calibrate": False},
            },
        ),
        Experiment(
            exp_id="rc_only",
            label="RC only (no temp corr, no calib)",
            group="baseline",
            overrides={
                "battery_params": {"rc_enabled": True, "ocv_temp_coeff": 0.0, "ocv_temp_quad": 0.0},
                "validation": {"calibrate": False},
            },
        ),
        Experiment(
            exp_id="full_model",
            label="Full model (RC + temp corr + calib)",
            group="baseline",
            overrides={},
        ),
        Experiment(
            exp_id="ablation_no_mc",
            label="Ablation: no Monte Carlo",
            group="ablation",
            overrides={"uncertainty": {"enabled": False}},
        ),
        Experiment(
            exp_id="ablation_no_thermal",
            label="Ablation: freeze thermal coupling",
            group="ablation",
            overrides={
                "battery_params": {"m": 1.0e9, "Cp": 1.0e9, "hA": 1.0e9},
            },
        ),
        Experiment(
            exp_id="ablation_no_polarization",
            label="Ablation: no polarization (RC off)",
            group="ablation",
            overrides={"battery_params": {"rc_enabled": False}},
        ),
        Experiment(
            exp_id="ablation_no_calibration",
            label="Ablation: no calibration",
            group="ablation",
            overrides={"validation": {"calibrate": False}},
        ),
    ]


def clear_previous_runs(root_out: Path) -> None:
    for run_dir in root_out.glob("run_*"):
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)


def plot_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    ensure_dir(out_dir)
    plt.rcParams.update({"axes.labelpad": 6})

    def short_label(label: str) -> str:
        mapping = {
            "Baseline (OCV-R only, no temp corr, no RC, no calib)": "Baseline",
            "Temp correction only (no RC, no calib)": "TempOnly",
            "RC only (no temp corr, no calib)": "RCOnly",
            "Full model (RC + temp corr + calib)": "FullModel",
            "Ablation: no Monte Carlo": "NoMC",
            "Ablation: freeze thermal coupling": "NoThermal",
            "Ablation: no polarization (RC off)": "NoPolar",
            "Ablation: no calibration": "NoCalib",
        }
        return mapping.get(label, label)

    # RMSE / MAE bar plots
    metrics = summary.drop_duplicates(subset=["experiment_id"])[
        ["experiment_id", "label", "rmse_v", "mae_v"]
    ]
    metrics = metrics.assign(label_short=metrics["label"].map(short_label))
    for metric, color in [("rmse_v", "#ff6f61"), ("mae_v", "#2a9d8f")]:
        plt.figure(figsize=(7.6, 4.6))
        ax = plt.gca()
        sns.barplot(data=metrics, x="label_short", y=metric, palette="mako", ax=ax)
        ax.set_ylabel(metric.upper().replace("_", " "))
        ax.set_xlabel("")
        ax.set_title(f"Validation {metric.upper()} by Experiment")
        ax.tick_params(axis="x", rotation=0)
        ax.margins(y=0.15)
        for rect, val in zip(ax.patches, metrics[metric]):
            ax.annotate(
                f"{val:.4f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        plt.tight_layout(pad=1.2)
        plt.savefig(out_dir / f"comparison_{metric}.png", dpi=300)
        plt.close()

    # Scenario energy and time per experiment
    for metric, title, ylabel in [
        ("energy_wh", "Energy Consumption by Scenario", "Energy (Wh)"),
        ("t_end_s", "Time to Empty by Scenario", "Time (s)"),
        ("temp_max_k", "Peak Temperature by Scenario", "Temperature (K)"),
    ]:
        plt.figure(figsize=(9.2, 4.8))
        ax = plt.gca()
        summary = summary.assign(label_short=summary["label"].map(short_label))
        sns.barplot(data=summary, x="scenario", y=metric, hue="label_short", ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.legend(fontsize=8, title="", ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        ax.margins(y=0.2)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=6, padding=1, rotation=90)
        plt.tight_layout(pad=1.2)
        plt.savefig(out_dir / f"comparison_{metric}.png", dpi=300)
        plt.close()


def main() -> None:
    base_cfg = load_config()
    experiments = build_experiments()

    root_out = ROOT / "outputs" / "experiments"
    ensure_dir(root_out)
    clear_previous_runs(root_out)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_out / f"run_{run_tag}"
    ensure_dir(run_dir)

    all_rows: List[Dict[str, Any]] = []
    registry: List[Dict[str, Any]] = []

    for exp in experiments:
        exp_dir = run_dir / exp.exp_id
        ensure_dir(exp_dir)
        cfg = copy.deepcopy(base_cfg)
        cfg = deep_update(cfg, exp.overrides)

        # Force output to experiment folder and keep visuals non-interactive
        cfg = deep_update(
            cfg,
            {
                "output": {"enabled": True, "format": "csv", "directory": str(exp_dir)},
                "visualization": {"show": False, "realtime": False, "save_frames": False},
            },
        )
        cfg = absolutize_paths(cfg)

        write_yaml(exp_dir / "config_effective.yaml", cfg)

        solution.run_scenarios(str(exp_dir / "config_effective.yaml"))

        # Summaries
        scenario_rows: List[Dict[str, Any]] = []
        for scenario_csv in exp_dir.glob("*.csv"):
            if scenario_csv.name in {"summary.csv", "validation_metrics.csv"}:
                continue
            try:
                row = summarize_output(scenario_csv)
            except ValueError:
                continue
            scenario_rows.append(row)

        validation = validation_metrics(cfg) or {}

        for row in scenario_rows:
            row.update(
                {
                    "experiment_id": exp.exp_id,
                    "label": exp.label,
                    "group": exp.group,
                    "rmse_v": validation.get("rmse_v"),
                    "mae_v": validation.get("mae_v"),
                    "current_scale": validation.get("current_scale"),
                    "r_scale": validation.get("r_scale"),
                    "ocv_scale": validation.get("ocv_scale"),
                    "voltage_offset": validation.get("voltage_offset"),
                }
            )
            all_rows.append(row)

        exp_summary = pd.DataFrame(scenario_rows)
        exp_summary.to_csv(exp_dir / "summary.csv", index=False)
        with (exp_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(validation, handle, indent=2, ensure_ascii=False)

        registry.append(
            {
                "experiment_id": exp.exp_id,
                "label": exp.label,
                "group": exp.group,
                "directory": str(exp_dir),
            }
        )

    summary = pd.DataFrame(all_rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    with (run_dir / "registry.json").open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, ensure_ascii=False)

    plot_comparison(summary, run_dir / "comparison_figures")


if __name__ == "__main__":
    main()
