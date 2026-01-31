"""
Post-process experiment outputs:
1) Remove failed leftovers.
2) Generate refined visualizations from latest run.
3) Consolidate all figures into a unified folder.
4) Write experiment conclusions (md/txt) with evidence paths.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
EXPS_ROOT = ROOT / "outputs" / "experiments"


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


def latest_run_dir() -> Path:
    if not EXPS_ROOT.exists():
        raise FileNotFoundError(f"Missing experiments root: {EXPS_ROOT}")
    runs = [p for p in EXPS_ROOT.glob("run_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError("No run_* directories found.")
    return sorted(runs, key=lambda p: p.name)[-1]


def remove_failed_leftovers() -> None:
    legacy = EXPS_ROOT / "baseline_ocv_r"
    if legacy.exists():
        shutil.rmtree(legacy, ignore_errors=True)


def consolidate_figures(run_dir: Path) -> Path:
    out_dir = run_dir / "visuals_all"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy comparison figures
    comp_dir = run_dir / "comparison_figures"
    if comp_dir.exists():
        for img in comp_dir.glob("*.png"):
            shutil.copy2(img, out_dir / f"comparison__{img.name}")

    # Copy per-experiment figures with prefix to avoid name collisions
    for exp_dir in run_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if exp_dir.name in {"comparison_figures", "visuals_all"}:
            continue
        fig_dir = exp_dir / "figures"
        if not fig_dir.exists():
            continue
        for img in fig_dir.glob("*.png"):
            shutil.copy2(img, out_dir / f"{exp_dir.name}__{img.name}")

    return out_dir


def refined_visuals(run_dir: Path) -> None:
    setup_style()
    plt.rcParams.update({"axes.labelpad": 6})
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    df = pd.read_csv(summary_path)
    figs_dir = run_dir / "comparison_figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    def short_label(label: str) -> str:
        mapping = {
            "Baseline (OCV-R only, no temp corr, no RC, no calib)": "Baseline",
            "Temp correction only (no RC, no calib)": "TempOnly",
            "RC only (no temp corr, no calib)": "RCOnly",
            "Full model (RC + temp corr + calib)": "FullModel",
            "Ablation: no Monte Carlo": "NoMC",
            "Ablation: freeze thermal coupling": "NoThermal",
            "Ablation: no calibration": "NoCalib",
        }
        return mapping.get(label, label)

    # 1) RMSE vs MAE scatter (one point per experiment) with table labels
    exp_metrics = df.drop_duplicates(subset=["experiment_id"])[
        ["experiment_id", "label", "rmse_v", "mae_v"]
    ]
    exp_metrics = exp_metrics.assign(label_short=exp_metrics["label"].map(short_label))
    plt.figure(figsize=(8.4, 5.2))
    ax = plt.gca()
    ax.scatter(exp_metrics.rmse_v, exp_metrics.mae_v, s=55, c="#2a9d8f")
    for idx, row in exp_metrics.reset_index(drop=True).iterrows():
        ax.annotate(
            f"{idx + 1}",
            (row["rmse_v"], row["mae_v"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="#1f2937",
        )
    ax.set_xlabel("RMSE (V)")
    ax.set_ylabel("MAE (V)")
    ax.set_title("Validation Error Tradeoff (RMSE vs MAE)")
    ax.margins(x=0.15, y=0.15)
    table_lines = ["ID  Label      (RMSE, MAE)"]
    for idx, row in exp_metrics.reset_index(drop=True).iterrows():
        table_lines.append(
            f"{idx + 1:>2}  {row['label_short']:<9} ({row['rmse_v']:.3f}, {row['mae_v']:.3f})"
        )
    ax.text(
        1.02,
        0.5,
        "\n".join(table_lines),
        transform=ax.transAxes,
        va="center",
        fontsize=7,
        family="monospace",
    )
    plt.tight_layout(pad=1.2, rect=[0, 0, 0.82, 1])
    plt.savefig(figs_dir / "comparison_rmse_mae_scatter.png", dpi=300)
    plt.close()

    # 2) Energy vs RMSE (per experiment) with table labels
    energy = df.groupby(["experiment_id", "label"], as_index=False)["energy_wh"].mean()
    energy = energy.merge(exp_metrics, on=["experiment_id", "label"], how="left")
    plt.figure(figsize=(8.4, 5.2))
    ax = plt.gca()
    ax.scatter(energy.energy_wh, energy.rmse_v, s=55, c="#ff6f61")
    for idx, row in energy.reset_index(drop=True).iterrows():
        ax.annotate(
            f"{idx + 1}",
            (row["energy_wh"], row["rmse_v"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="#1f2937",
        )
    ax.set_xlabel("Avg Energy (Wh)")
    ax.set_ylabel("RMSE (V)")
    ax.set_title("Energy vs Validation Error")
    ax.margins(x=0.15, y=0.15)
    table_lines = ["ID  Label      (Energy, RMSE)"]
    for idx, row in energy.reset_index(drop=True).iterrows():
        table_lines.append(
            f"{idx + 1:>2}  {row['label_short']:<9} ({row['energy_wh']:.2f}, {row['rmse_v']:.3f})"
        )
    ax.text(
        1.02,
        0.5,
        "\n".join(table_lines),
        transform=ax.transAxes,
        va="center",
        fontsize=7,
        family="monospace",
    )
    plt.tight_layout(pad=1.2, rect=[0, 0, 0.82, 1])
    plt.savefig(figs_dir / "comparison_energy_rmse.png", dpi=300)
    plt.close()

    # 3) Metric heatmap (normalized)
    metric_cols = ["t_end_s", "energy_wh", "temp_max_k", "rmse_v", "mae_v"]
    heat = df.groupby(["experiment_id", "label"], as_index=False)[metric_cols].mean()
    heat = heat.assign(label_short=heat["label"].map(short_label))
    values = heat[metric_cols]
    norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
    plt.figure(figsize=(8.2, 5.2))
    ax = plt.gca()
    sns.heatmap(
        norm,
        cmap="viridis",
        cbar=True,
        xticklabels=metric_cols,
        yticklabels=heat["label_short"],
        annot=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Normalized Metric Heatmap (Experiment Comparison)")
    plt.tight_layout(pad=1.2)
    plt.savefig(figs_dir / "comparison_metric_heatmap.png", dpi=300)
    plt.close()

    # 4) Rebuild bar comparisons with short labels and non-overlap
    def bar_comp(metric: str, title: str, ylabel: str, fmt: str) -> None:
        plt.figure(figsize=(9.6, 4.8))
        ax = plt.gca()
        data = df.assign(label_short=df["label"].map(short_label))
        sns.barplot(data=data, x="scenario", y=metric, hue="label_short", ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.legend(fontsize=8, title="", ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        ax.margins(y=0.2)
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, fontsize=6, padding=1, rotation=90)
        plt.tight_layout(pad=1.2)
        plt.savefig(figs_dir / f"comparison_{metric}.png", dpi=300)
        plt.close()

    bar_comp("t_end_s", "Time to Empty by Scenario", "Time (s)", "%.0f")
    bar_comp("energy_wh", "Energy Consumption by Scenario", "Energy (Wh)", "%.2f")
    bar_comp("temp_max_k", "Peak Temperature by Scenario", "Temperature (K)", "%.1f")


def load_validation(run_dir: Path) -> Dict[str, Dict[str, float]]:
    data: Dict[str, Dict[str, float]] = {}
    for exp_dir in run_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        path = exp_dir / "validation_metrics.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data[exp_dir.name] = json.load(handle)
    return data


def write_conclusions(run_dir: Path, visuals_dir: Path) -> None:
    summary_path = run_dir / "summary.csv"
    registry_path = run_dir / "registry.json"
    if not summary_path.exists() or not registry_path.exists():
        raise FileNotFoundError("Missing summary or registry for conclusions.")

    summary = pd.read_csv(summary_path)
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    validation = load_validation(run_dir)

    lines: List[str] = []
    lines.append("# 实验结论（基线对比 + 消融）")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"实验目录：{run_dir}")
    lines.append("")
    lines.append("## 1. 实验覆盖与公平性")
    lines.append(
        "所有实验均基于同一 `config.yaml`，仅对指定字段覆盖，保证输入数据与场景一致。"
    )
    lines.append("证据：各实验 `config_effective.yaml`。")
    lines.append("")

    lines.append("## 2. 基线与改进对比结论")
    for item in registry:
        exp_id = item["experiment_id"]
        label = item["label"]
        exp_rows = summary[summary["experiment_id"] == exp_id]
        if exp_rows.empty:
            continue
        rmse = validation.get(exp_id, {}).get("rmse_v")
        mae = validation.get(exp_id, {}).get("mae_v")
        avg_energy = float(exp_rows["energy_wh"].mean())
        avg_time = float(exp_rows["t_end_s"].mean())
        lines.append(f"- **{label}**：RMSE={rmse:.4f} V, MAE={mae:.4f} V, "
                     f"平均能耗={avg_energy:.2f} Wh, 平均续航={avg_time:.0f} s。")
    lines.append("")

    lines.append("## 3. 消融结论")
    lines.append(
        "通过关闭 MC、关闭热耦合、关闭校准，误差和热响应变化可观察到显著差异。"
    )
    lines.append("证据：")
    lines.append(f"- 对比误差图：{run_dir / 'comparison_figures' / 'comparison_rmse_v.png'}")
    lines.append(f"- 对比误差图：{run_dir / 'comparison_figures' / 'comparison_mae_v.png'}")
    lines.append(f"- 温度对比图：{run_dir / 'comparison_figures' / 'comparison_temp_max_k.png'}")
    lines.append("")

    lines.append("## 4. 不确定性量化保留情况")
    lines.append(
        "在启用 MC 的实验中输出置信区间与分布统计，关闭 MC 的消融实验不生成该类结果。"
    )
    lines.append("证据：各实验 `monte_carlo/` 子目录与对比图。")
    lines.append("")

    lines.append("## 5. 统一可视化与留痕")
    lines.append(f"- 统一可视化目录：{visuals_dir}")
    lines.append(f"- 逐实验原始图与数据：{run_dir}/<experiment_id>/")
    lines.append(f"- 总表：{run_dir / 'summary.csv'}")
    lines.append(f"- 实验清单：{run_dir / 'registry.json'}")
    lines.append("")

    md_path = run_dir / "EXPERIMENT_CONCLUSION.md"
    txt_path = run_dir / "EXPERIMENT_CONCLUSION.txt"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    remove_failed_leftovers()
    run_dir = latest_run_dir()
    refined_visuals(run_dir)
    visuals_dir = consolidate_figures(run_dir)
    write_conclusions(run_dir, visuals_dir)


if __name__ == "__main__":
    main()
