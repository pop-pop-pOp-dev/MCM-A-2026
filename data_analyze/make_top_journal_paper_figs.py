"""
Generate top-journal (science/nature-like) paper figures and write them to D:\\MCM-Article.

Figures generated:
1) validation_phone_validation_bcm_s9_combined.png (replaces existing): multi-panel input/output alignment
2) uncertainty_hist_Video_Streaming.png (replaces existing): ECDF + quantiles + raincloud-style summary
3) comparison_rmse_v.png (replaces existing): mechanism ablation trade-offs (Gaming) with TTE + RMSE panels

Design goals:
- publication-style typography
- thin lines, high-contrast but tasteful colors
- dense but readable numeric annotations
- no interactive windows (savefig only)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _try_set_science_style() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Prefer SciencePlots if available, otherwise fall back to a clean custom theme.
    try:
        import scienceplots  # noqa: F401

        # Prefer a Nature/Science-like look, without LaTeX dependency.
        # (If "nature" is unavailable, SciencePlots will ignore it.)
        plt.style.use(["science", "nature", "no-latex"])
    except Exception:
        # Clean, "journal-ish" defaults without relying on external style packages.
        mpl.rcParams.update(
            {
                "figure.dpi": 160,
                "savefig.dpi": 600,
                "font.family": "DejaVu Sans",
                "font.size": 10.5,
                "axes.titlesize": 11.5,
                "axes.labelsize": 10.5,
                "xtick.labelsize": 9.5,
                "ytick.labelsize": 9.5,
                "legend.fontsize": 9.0,
                "axes.grid": True,
                "grid.alpha": 0.20,
                "grid.linewidth": 0.7,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "lines.linewidth": 1.0,
                "lines.solid_capstyle": "round",
                "lines.solid_joinstyle": "round",
                "axes.facecolor": "white",
                "figure.facecolor": "white",
            }
        )
    # Enforce a consistent “thin, precise” look even if external styles are loaded.
    mpl.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "axes.linewidth": 0.9,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "lines.linewidth": 1.05,
        }
    )


def _palette() -> dict[str, str]:
    # High-contrast, science-friendly, vivid but tasteful.
    # (Chosen to avoid “flat default” red/green and reduce visual fatigue.)
    return {
        "ink": "#111827",  # near-black
        "muted": "#6B7280",  # gray
        "grid": "#94A3B8",  # slate
        "accent_a": "#1D4ED8",  # royal blue (model)
        "accent_b": "#0891B2",  # deep cyan (current)
        "accent_c": "#F59E0B",  # amber (temperature / quantile)
        "accent_d": "#C026D3",  # vivid magenta (emphasis)
        "accent_e": "#16A34A",  # green (median/anchor)
        "accent_f": "#DC2626",  # red (warnings/cutoff)
        "bg": "#FFFFFF",
    }


def _ensure_out_dir() -> Path:
    out_dir = Path(r"D:\MCM-Article")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_config_params():
    # Import locally to avoid importing heavy modules before style setup.
    from solution import BatteryParams, load_yaml_config

    cfg = load_yaml_config(Path(r"D:\MCM\config.yaml"))
    params = BatteryParams.from_dict(cfg["battery_params"])
    validation = cfg.get("validation", {}) or {}
    profile_rel = validation.get("current_profile_csv")
    if not profile_rel:
        raise ValueError("config.yaml missing validation.current_profile_csv")
    profile_path = Path(r"D:\MCM") / str(profile_rel)
    return cfg, params, profile_path


def _read_outputs_csv(stem: str) -> pd.DataFrame:
    p = Path(r"D:\MCM\outputs") / f"{stem}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing outputs CSV: {p}")
    return pd.read_csv(p)


def _format_time_h(seconds: float) -> str:
    return f"{seconds/3600.0:.2f} h"


def make_validation_alignment_figure() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from solution import validate_current_profile

    colors = _palette()
    out_dir = _ensure_out_dir()

    cfg, params, profile_path = _load_config_params()
    validation = cfg.get("validation", {}) or {}

    time_s, v_pred, v_meas, meta = validate_current_profile(
        params=params,
        profile_path=profile_path,
        use_temp_from_csv=bool(validation.get("use_temp_from_csv", True)),
        calibrate=bool(validation.get("calibrate", True)),
        current_scale_bounds=tuple(validation.get("current_scale_bounds", [0.8, 1.2])),
        use_soc_from_csv=bool(validation.get("use_soc_from_csv", True)),
    )

    df = pd.read_csv(profile_path)
    # Normalize column names (dataset uses time_s/current_a/voltage_v/temp_c/soc).
    t = df["time_s"].to_numpy(dtype=float)
    i_raw = df["current_a"].to_numpy(dtype=float)
    temp_c = df["temp_c"].to_numpy(dtype=float) if "temp_c" in df.columns else np.full_like(t, np.nan)
    soc = df["soc"].to_numpy(dtype=float) if "soc" in df.columns else np.full_like(t, np.nan)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    soc = np.clip(soc, 0.0, 1.0)

    i_eff = i_raw * float(meta.get("current_scale", 1.0))
    resid = v_pred - v_meas
    rmse = float(np.sqrt(np.mean((resid) ** 2)))
    mae = float(np.mean(np.abs(resid)))

    t_h = t / 3600.0

    fig = plt.figure(figsize=(7.6, 6.6), constrained_layout=False)
    # Top two panels span both columns; bottom row is split to avoid occlusion.
    gs = GridSpec(
        3,
        2,
        height_ratios=[1.05, 1.55, 1.05],
        width_ratios=[1.0, 0.62],
        hspace=0.22,
        wspace=0.22,
        figure=fig,
    )

    # Panel A: Current + Temperature
    ax0 = fig.add_subplot(gs[0, :])
    ax0.fill_between(t_h, 0.0, i_eff, color=colors["accent_b"], alpha=0.18, linewidth=0.0, label="Current (scaled)")
    ax0.plot(t_h, i_eff, color=colors["accent_b"], linewidth=0.75, alpha=0.95)
    ax0.set_ylabel("Current (A)")
    ax0.set_xlim(float(np.nanmin(t_h)), float(np.nanmax(t_h)))

    ax0_t = ax0.twinx()
    if np.isfinite(temp_c).any():
        ax0_t.plot(t_h, temp_c, color=colors["accent_c"], linewidth=0.85, alpha=0.95, label="Temperature")
        ax0_t.set_ylabel("Temp (°C)")
    ax0.set_title("Phone-level validation: input excitation and voltage response", loc="left", color=colors["ink"])
    # Legend (combine both axes)
    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax0_t.get_legend_handles_labels()
    ax0.legend(h0 + h1, l0 + l1, loc="upper right", frameon=True, framealpha=0.92)

    # Panel B: Voltage (measured vs predicted)
    ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
    ax1.plot(t_h, v_meas, color=colors["muted"], linewidth=0.95, alpha=0.92, label="Measured $V$")
    ax1.plot(t_h, v_pred, color=colors["accent_a"], linewidth=1.05, alpha=0.95, label="Model $\\hat{V}$")
    ax1.fill_between(
        t_h,
        v_meas,
        v_pred,
        color=colors["accent_a"],
        alpha=0.08,
        linewidth=0.0,
        label="Error envelope",
    )
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True, alpha=0.18)

    # Annotate metrics and calibration factors (compact box)
    text = (
        f"RMSE = {rmse*1000:.1f} mV\n"
        f"MAE  = {mae*1000:.1f} mV\n"
        f"$s_I$={meta.get('current_scale', 1.0):.3f}, "
        f"$s_R$={meta.get('r_scale', 1.0):.3f}, "
        f"$s_V$={meta.get('ocv_scale', 1.0):.3f}\n"
        f"$b$={meta.get('voltage_offset', 0.0):+.3f} V"
    )
    ax1.text(
        0.012,
        0.98,
        text,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        color=colors["ink"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=colors["grid"], alpha=0.88),
    )

    # Panel C-left: Residual vs SOC (colored scatter)
    ax2 = fig.add_subplot(gs[2, 0])
    c = t_h
    sc = ax2.scatter(
        soc,
        resid * 1000.0,
        c=c,
        cmap="viridis",
        s=12,
        alpha=0.70,
        linewidths=0.0,
    )
    ax2.axhline(0.0, color=colors["muted"], linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("SOC (fraction)")
    ax2.set_ylabel("Residual (mV)")
    ax2.set_xlim(0.0, 1.0)
    # Horizontal colorbar to reduce right-side clutter.
    cb = fig.colorbar(sc, ax=ax2, pad=0.18, fraction=0.12, orientation="horizontal")
    cb.set_label("Time (h)")

    # Panel C-right: Residual vs time (separate subplot to eliminate occlusion)
    ax3 = fig.add_subplot(gs[2, 1], sharey=ax2)
    ax3.plot(t_h, resid * 1000.0, color=colors["accent_a"], linewidth=0.85, alpha=0.9)
    ax3.axhline(0.0, color=colors["muted"], linewidth=0.7, alpha=0.8)
    ax3.set_title("Residual vs time", fontsize=9.3, pad=2.5)
    ax3.set_xlabel("Time (h)")
    ax3.tick_params(labelleft=False)
    ax3.grid(True, alpha=0.18)

    # Shared formatting
    for ax in (ax0, ax1):
        ax.tick_params(labelbottom=False)
    # Keep legend away from dense segments / metric box.
    ax1.legend(loc="upper right", ncol=1, frameon=True, framealpha=0.86, edgecolor=colors["grid"])

    fig.savefig(out_dir / "validation_phone_validation_bcm_s9_combined.png", bbox_inches="tight")
    plt.close(fig)


def make_uncertainty_ecdf_raincloud() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = _palette()
    out_dir = _ensure_out_dir()

    tte_path = Path(r"D:\MCM\outputs\monte_carlo\uncertainty_tte_Video_Streaming.csv")
    if not tte_path.exists():
        # Fallback to experiment folder if needed.
        alt = Path(r"D:\MCM\outputs\experiments\run_20260201_151319\full_model\monte_carlo\uncertainty_tte_Video_Streaming.csv")
        tte_path = alt

    df = pd.read_csv(tte_path)
    col = "time_to_empty_s" if "time_to_empty_s" in df.columns else ("tte_s" if "tte_s" in df.columns else df.columns[0])
    tte_s = df[col].to_numpy(dtype=float)
    tte_h = tte_s / 3600.0
    tte_h = tte_h[np.isfinite(tte_h)]
    tte_h.sort()

    n = len(tte_h)
    y = np.arange(1, n + 1) / n

    q05, q50, q95 = np.quantile(tte_h, [0.05, 0.50, 0.95])
    mu = float(np.mean(tte_h))
    sigma = float(np.std(tte_h, ddof=1)) if n > 1 else 0.0
    cv = (sigma / mu * 100.0) if mu > 0 else 0.0

    # Avoid overriding our global style; only set minor seaborn helpers.
    sns.set_context("paper")
    fig = plt.figure(figsize=(7.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    # ECDF with a bold but clean look.
    ax.plot(tte_h, y, color=colors["accent_a"], linewidth=1.35)
    ax.fill_between(tte_h, 0, y, color=colors["accent_a"], alpha=0.06, linewidth=0)

    # Quantile markers with callouts (avoid bottom overlaps).
    quantiles = [
        (q05, "5th", colors["accent_b"], 0.18),
        (q50, "median", colors["accent_e"], 0.55),
        (q95, "95th", colors["accent_c"], 0.88),
    ]
    for q, label, colr, yloc in quantiles:
        ax.axvline(q, color=colr, linewidth=1.0, alpha=0.95)
        ax.annotate(
            f"{label}: {q:.3f} h",
            xy=(q, yloc),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.2,
            color=colors["ink"],
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=colors["grid"], alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=colr, lw=0.9, alpha=0.8),
        )

    ax.set_xlabel("Time-to-Empty (h)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Uncertainty in TTE (Monte Carlo): ECDF + quantile anchors", loc="left", color=colors["ink"])

    # Raincloud-style inset (violin + box + points)
    inset = ax.inset_axes([0.08, 0.58, 0.46, 0.34])
    sns.violinplot(x=tte_h, orient="h", ax=inset, inner=None, color=colors["accent_a"], linewidth=0.0, cut=0)
    sns.boxplot(x=tte_h, orient="h", ax=inset, width=0.25, showcaps=False, showfliers=False, whis=(5, 95),
                boxprops=dict(facecolor="white", edgecolor=colors["ink"], linewidth=0.9),
                medianprops=dict(color=colors["accent_e"], linewidth=1.2),
                whiskerprops=dict(color=colors["ink"], linewidth=0.9))
    sns.stripplot(x=tte_h, orient="h", ax=inset, size=2.4, color=colors["ink"], alpha=0.35, jitter=0.08)
    inset.set_yticks([])
    inset.set_xlabel("")
    inset.set_title(f"$\\mu$={mu:.3f} h, $\\sigma$={sigma*60:.2f} min, CV={cv:.2f}%", fontsize=9.0, pad=2.0)
    inset.grid(True, axis="x", alpha=0.18)
    inset.set_facecolor((1, 1, 1, 0.85))

    fig.savefig(out_dir / "uncertainty_hist_Video_Streaming.png", bbox_inches="tight")
    plt.close(fig)


def make_ablation_tradeoff_figure() -> None:
    import matplotlib.pyplot as plt

    colors = _palette()
    out_dir = _ensure_out_dir()

    variants = [
        ("Full model", Path(r"D:\MCM\outputs\experiments\run_20260201_151319\full_model\summary.csv"), colors["accent_e"]),
        ("No thermal", Path(r"D:\MCM\outputs\experiments\run_20260201_151319\ablation_no_thermal\summary.csv"), colors["accent_d"]),
        ("RC off", Path(r"D:\MCM\outputs\experiments\run_20260201_151319\ablation_no_polarization\summary.csv"), colors["accent_b"]),
        ("No calib", Path(r"D:\MCM\outputs\experiments\run_20260201_151319\ablation_no_calibration\summary.csv"), colors["accent_a"]),
    ]

    rows = []
    for name, p, colr in variants:
        df = pd.read_csv(p)
        g = df[df["scenario"] == "Gaming"].iloc[0].to_dict()
        rows.append(
            {
                "variant": name,
                "t_end_s": float(g["t_end_s"]),
                "tte_min": float(g["t_end_s"]) / 60.0,
                "rmse_v": float(g.get("rmse_v", np.nan)),
                "color": colr,
            }
        )

    d = pd.DataFrame(rows)
    # Reference (full)
    ref_t = float(d.loc[d["variant"] == "Full model", "tte_min"].iloc[0])
    ref_rmse = float(d.loc[d["variant"] == "Full model", "rmse_v"].iloc[0])

    # Modern two-panel "trade-off card": TTE vs RMSE side-by-side.
    order = ["Full model", "No thermal", "RC off", "No calib"]
    d["variant"] = pd.Categorical(d["variant"], categories=order, ordered=True)
    d = d.sort_values("variant")
    y = np.arange(len(d))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True, gridspec_kw={"wspace": 0.18})

    # Left: TTE (min)
    ax0.axvline(ref_t, color=colors["muted"], linewidth=1.0, alpha=0.65)
    for yi, row in zip(y, d.to_dict("records")):
        ax0.plot([ref_t, row["tte_min"]], [yi, yi], color=row["color"], alpha=0.30, linewidth=2.6)
        ax0.scatter(row["tte_min"], yi, s=82, color=row["color"], edgecolor="white", linewidth=0.9, zorder=3)
        delta = (row["tte_min"] / ref_t - 1.0) * 100.0
        ax0.text(
            row["tte_min"],
            yi - 0.22,
            f"{row['tte_min']:.1f} min",
            ha="center",
            va="top",
            fontsize=9.2,
            color=colors["ink"],
        )
        # Avoid redundant +0.0% labels (reduces clutter).
        if abs(delta) >= 0.05:
            ax0.text(
                row["tte_min"],
                yi + 0.26,
                f"{delta:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.0,
                color=colors["muted"],
            )
    ax0.set_title("Gaming: TTE shift", loc="left", color=colors["ink"])
    ax0.set_xlabel("Time-to-Empty (min)")
    ax0.set_yticks(y)
    ax0.set_yticklabels(d["variant"])

    # Right: RMSE (mV)
    rmse_mv = d["rmse_v"].to_numpy(dtype=float) * 1000.0
    ref_rmse_mv = ref_rmse * 1000.0
    ax1.axvline(ref_rmse_mv, color=colors["muted"], linewidth=1.0, alpha=0.65)
    for yi, row, x in zip(y, d.to_dict("records"), rmse_mv):
        ax1.plot([ref_rmse_mv, x], [yi, yi], color=row["color"], alpha=0.30, linewidth=2.6)
        ax1.scatter(x, yi, s=82, color=row["color"], edgecolor="white", linewidth=0.9, zorder=3)
        ax1.text(
            x,
            yi - 0.22,
            f"{x:.1f} mV",
            ha="center",
            va="top",
            fontsize=9.2,
            color=colors["ink"],
        )
    ax1.set_title("Gaming: voltage RMSE", loc="left", color=colors["ink"])
    ax1.set_xlabel("Voltage RMSE (mV)")
    ax1.tick_params(labelleft=False)

    # Tight ranges with padding
    ax0.set_xlim(min(d["tte_min"].min(), ref_t) - 1.6, max(d["tte_min"].max(), ref_t) + 1.6)
    ax1.set_xlim(min(rmse_mv.min(), ref_rmse_mv) - 1.2, max(rmse_mv.max(), ref_rmse_mv) + 1.2)

    fig.suptitle("Ablation trade-offs (Gaming): runtime vs fit accuracy", x=0.52, y=0.995, fontsize=12.0, color=colors["ink"])
    fig.savefig(out_dir / "comparison_rmse_v.png", bbox_inches="tight")
    plt.close(fig)


def make_uncertainty_band() -> None:
    import matplotlib.pyplot as plt

    colors = _palette()
    out_dir = _ensure_out_dir()

    p = Path(r"D:\MCM\outputs\monte_carlo\uncertainty_percentiles_Video_Streaming.csv")
    if not p.exists():
        alt = Path(r"D:\MCM\outputs\experiments\run_20260201_151319\full_model\monte_carlo\uncertainty_percentiles_Video_Streaming.csv")
        p = alt
    df = pd.read_csv(p)
    t_h = df["t_s"].to_numpy(dtype=float) / 3600.0
    p5 = df["soc_p5"].to_numpy(dtype=float) * 100.0
    p50 = df["soc_p50"].to_numpy(dtype=float) * 100.0
    p95 = df["soc_p95"].to_numpy(dtype=float) * 100.0

    # Focus on the informative window (avoid large blank tail after cutoff).
    cutoff_idx = int(np.argmax(p50 <= 0.1)) if np.any(p50 <= 0.1) else len(t_h) - 1
    t_end = float(t_h[min(cutoff_idx + 60, len(t_h) - 1)])  # small buffer
    t_end = max(t_end, float(np.nanmax(t_h[t_h <= 2.0])) if np.any(t_h <= 2.0) else float(np.nanmax(t_h)))

    fig, ax = plt.subplots(figsize=(7.4, 3.95))
    ax.fill_between(t_h, p5, p95, color=colors["accent_a"], alpha=0.14, linewidth=0.0, label="5–95% band")
    ax.plot(t_h, p50, color=colors["accent_a"], linewidth=1.10, label="Median SOC")
    ax.plot(t_h, p5, color=colors["accent_a"], linewidth=0.75, alpha=0.38)
    ax.plot(t_h, p95, color=colors["accent_a"], linewidth=0.75, alpha=0.38)
    ax.set_title("Uncertainty band (Video Streaming): SOC with quantile anchors", loc="left", color=colors["ink"])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("SOC (%)")
    ax.set_xlim(0.0, t_end)
    ax.set_ylim(-2.5, 102.5)
    ax.grid(True, alpha=0.18)

    # Anchor annotations: start, median cutoff time (when median reaches ~0)
    ax.annotate("start: 100%", xy=(0.03 * t_end, 97.0), xytext=(0, 0), textcoords="offset points",
                fontsize=9.0, color=colors["ink"],
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=colors["grid"], alpha=0.85))
    if cutoff_idx > 0:
        t_cut = float(t_h[cutoff_idx])
        ax.axvline(t_cut, color=colors["accent_c"], linewidth=1.0, alpha=0.9)
        ax.annotate(
            f"median cutoff\n{t_cut:.2f} h",
            xy=(t_cut, 8.0),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.2,
            color=colors["ink"],
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=colors["grid"], alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=colors["accent_c"], lw=0.9, alpha=0.8),
        )

    # Add “reading anchors” at SOC = 80/50/20% on the median curve.
    for target in (80.0, 50.0, 20.0):
        if np.any(p50 <= target):
            idx = int(np.argmax(p50 <= target))
            t0 = float(t_h[idx])
            y0 = float(p50[idx])
            ax.scatter([t0], [y0], s=18, color=colors["accent_e"], edgecolor="white", linewidth=0.7, zorder=4)
            ax.annotate(
                f"{target:.0f}% @ {t0:.2f} h",
                xy=(t0, y0),
                xytext=(8, -14),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8.8,
                color=colors["ink"],
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=colors["grid"], alpha=0.84),
                arrowprops=dict(arrowstyle="-", color=colors["accent_e"], lw=0.85, alpha=0.7),
            )

    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.savefig(out_dir / "uncertainty_band_Video_Streaming.png", bbox_inches="tight")
    plt.close(fig)


def make_phantom_battery_schematic() -> None:
    """
    Visual explanation of inaccessible capacity (phantom battery) under cold conditions.
    We plot OCV and implied V_term under a representative constant-power load at warm vs cold,
    and highlight the extra inaccessible SOC interval in cold conditions.
    """
    import matplotlib.pyplot as plt

    from solution import BatteryPhysics

    colors = _palette()
    out_dir = _ensure_out_dir()

    cfg, params, _ = _load_config_params()
    physics = BatteryPhysics(params)

    # Choose a representative constant-power demand by matching the observed Winter cutoff SOC.
    # This keeps the schematic aligned with the reported “~13.1% capacity remaining” phenomenon.
    summary_path = Path(r"D:\MCM\outputs\experiments\run_20260201_151319\full_model\summary.csv")
    target_soc_cold = 0.13
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        try:
            target_soc_cold = float(s[s["scenario"] == "Winter_Usage"]["soc_end"].iloc[0])
        except Exception:
            target_soc_cold = 0.13

    v_cut = float(params.V_cutoff)
    soc = np.linspace(0.02, 0.99, 600)

    # Two ambient operating temperatures (warm vs cold).
    T_warm = 298.15   # 25C
    T_cold = 268.15   # -5C (winter scenario)

    def vterm_under_cpl(Tk: float, P: float) -> np.ndarray:
        voc = np.array([physics.get_ocv(float(z), params.ocv_temp_ref) for z in soc], dtype=float)
        r = np.array([physics.get_r_int(float(z), Tk, params.SOH_init) for z in soc], dtype=float)
        disc = voc**2 - 4.0 * r * P
        stable = disc > 0
        out = np.full_like(voc, np.nan, dtype=float)
        disc_clip = np.maximum(disc, 0.0)
        i = (voc - np.sqrt(disc_clip)) / (2.0 * np.maximum(r, 1e-9))
        out[stable] = (voc - i * r)[stable]
        return out

    voc = np.array([physics.get_ocv(float(z), params.ocv_temp_ref) for z in soc], dtype=float)
    def soc_star_from_power(Tk: float, P: float) -> float:
        vt = vterm_under_cpl(Tk, P)
        # s*: smallest SOC where V_term >= V_cutoff (shutdown when SOC drops below s*).
        idx = np.where(np.isfinite(vt) & (vt >= v_cut))[0]
        if len(idx) == 0:
            return float(soc[-1])
        return float(soc[idx[0]])

    # Binary search P so that s*_cold approximately matches the observed Winter soc_end.
    # Larger P generally increases s* (earlier cutoff).
    lo, hi = 0.5, 15.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s_mid = soc_star_from_power(T_cold, mid)
        if s_mid >= target_soc_cold:
            hi = mid
        else:
            lo = mid
    p_rep = 0.5 * (lo + hi)

    vt_w = vterm_under_cpl(T_warm, p_rep)
    vt_c = vterm_under_cpl(T_cold, p_rep)

    def soc_star(vt: np.ndarray) -> float:
        idx = np.where(np.isfinite(vt) & (vt >= v_cut))[0]
        if len(idx) == 0:
            return float(soc[-1])
        return float(soc[idx[0]])

    s_w = soc_star(vt_w)
    s_c = soc_star(vt_c)

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.plot(soc, voc, color=colors["muted"], linewidth=1.0, alpha=0.85, label="$V_{OCV}(SOC)$")
    ax.plot(soc, vt_w, color=colors["accent_e"], linewidth=1.45, label=f"$V_{{term}}$ (warm, {T_warm-273.15:.0f}°C)")
    ax.plot(soc, vt_c, color=colors["accent_d"], linewidth=1.45, label=f"$V_{{term}}$ (cold, {T_cold-273.15:.0f}°C)")
    ax.axhline(v_cut, color=colors["accent_c"], linewidth=1.1, alpha=0.95, label="$V_{cutoff}$")

    # Shade additional inaccessible capacity due to cold.
    if s_c > s_w:
        ax.axvspan(s_w, s_c, color=colors["accent_d"], alpha=0.10, lw=0)
        ax.annotate(
            "extra inaccessible\ncapacity in cold",
            xy=((s_w + s_c) / 2, v_cut + 0.05),
            ha="center",
            va="bottom",
            fontsize=9.2,
            color=colors["ink"],
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=colors["grid"], alpha=0.85),
        )

    # SOC* markers
    for s_star, vt, colr, tag, dy in [
        (s_w, vt_w, colors["accent_e"], "warm", -16),
        (s_c, vt_c, colors["accent_d"], "cold", +10),
    ]:
        ax.scatter([s_star], [v_cut], s=48, color=colr, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(
            f"$s^*_{{{tag}}}$={s_star*100:.1f}%",
            xy=(s_star, v_cut),
            xytext=(10, dy),
            textcoords="offset points",
            fontsize=9.2,
            color=colors["ink"],
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=colors["grid"], alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=colr, lw=0.9, alpha=0.85),
        )

    ax.set_title("Phantom battery: cold-induced cutoff and inaccessible capacity", loc="left", color=colors["ink"])
    ax.set_xlabel("SOC (fraction)")
    ax.set_ylabel("Voltage (V)")
    ax.set_xlim(0.0, 1.0)
    ymin = min(v_cut - 0.15, float(np.nanmin(np.nan_to_num(vt_c, nan=v_cut))) - 0.05)
    ax.set_ylim(ymin, float(np.nanmax(voc)) + 0.05)
    ax.grid(True, alpha=0.18)
    ax.text(
        0.99,
        0.03,
        f"Representative demand: $P_{{sys}}$ ≈ {p_rep:.1f} W (matched to winter cutoff)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        color=colors["muted"],
    )
    ax.legend(loc="upper right", frameon=True, framealpha=0.90)
    fig.savefig(out_dir / "phantom_battery_cutoff.png", bbox_inches="tight")
    plt.close(fig)


def make_soa_with_collapse_region() -> None:
    """
    Safe Operating Area (SOA) map with an explicit CPL collapse region (Δ < 0).
    We visualize regions in the (SOC, T) plane under a representative high-load power demand.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from solution import BatteryPhysics

    colors = _palette()
    out_dir = _ensure_out_dir()

    cfg, params, _ = _load_config_params()
    physics = BatteryPhysics(params)

    # Use a representative high-load power level: 95th percentile of Gaming P_total from outputs.
    gaming_csv = Path(r"D:\MCM\outputs\Gaming.csv")
    if not gaming_csv.exists():
        gaming_csv = Path(r"D:\MCM\outputs\experiments\run_20260201_151319\full_model\Gaming.csv")
    df_g = pd.read_csv(gaming_csv)
    if "p_total" in df_g.columns:
        p_rep = float(np.quantile(df_g["p_total"].to_numpy(dtype=float), 0.95))
    else:
        p_rep = 20.0

    v_cut = float(params.V_cutoff)
    T_th = 318.15  # 45C

    soc_grid = np.linspace(0.02, 0.98, 160)
    temp_grid = np.linspace(268.15, 323.15, 140)  # -5C to 50C
    S, T = np.meshgrid(soc_grid, temp_grid)

    Voc = np.vectorize(lambda z: physics.get_ocv(float(z), params.ocv_temp_ref))(S)
    R = np.vectorize(lambda z, tk: physics.get_r_int(float(z), float(tk), params.SOH_init))(S, T)

    Delta = Voc**2 - 4.0 * R * p_rep
    collapse = Delta < 0
    disc = np.maximum(Delta, 0.0)
    I = (Voc - np.sqrt(disc)) / (2.0 * np.maximum(R, 1e-9))
    Vt = Voc - I * R
    cutoff = (Vt <= v_cut) & (~collapse)
    thermal = (T >= T_th) & (~collapse)  # thermal-limited zone (independent overlay)

    # Region encoding: 0 safe, 1 cutoff, 2 collapse, 3 thermal-limited
    region = np.zeros_like(S, dtype=int)
    region[cutoff] = 1
    region[collapse] = 2
    region[thermal] = 3

    cmap = ListedColormap(
        [
            "#E8F5E9",  # safe
            "#FFF7ED",  # cutoff (voltage sag)
            "#FCE7F3",  # collapse (Δ<0)
            "#EEF2FF",  # thermal-limited
        ]
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = ax.imshow(
        region,
        origin="lower",
        aspect="auto",
        extent=[soc_grid.min(), soc_grid.max(), temp_grid.min() - 273.15, temp_grid.max() - 273.15],
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_title("Safe Operating Area with CPL collapse region ($\\Delta<0$)", loc="left", color=colors["ink"])
    ax.set_xlabel("SOC (fraction)")
    ax.set_ylabel("Temperature (°C)")

    # Compact legend as labeled boxes
    legend_items = [
        ("Safe", cmap.colors[0]),
        ("Cutoff (voltage sag)", cmap.colors[1]),
        ("CPL collapse ($\\Delta<0$)", cmap.colors[2]),
        ("Thermal-limited ($T>45^{\\circ}$C)", cmap.colors[3]),
    ]
    x0, y0 = 0.04, 0.96
    for i, (lab, colr) in enumerate(legend_items):
        yy = y0 - i * 0.08
        ax.add_patch(plt.Rectangle((x0, yy - 0.03), 0.03, 0.045, transform=ax.transAxes, color=colr, ec=colors["grid"], lw=0.6))
        ax.text(x0 + 0.04, yy - 0.01, lab, transform=ax.transAxes, va="center", ha="left", fontsize=9.2, color=colors["ink"])

    ax.text(
        0.98,
        0.03,
        f"Representative demand: $P_{{sys}}$ (Gaming, 95th pct) ≈ {p_rep:.1f} W",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        color=colors["muted"],
    )

    fig.savefig(out_dir / "soa_heatmap_Video_Streaming.png", bbox_inches="tight")
    plt.close(fig)


def make_power_params_compact() -> None:
    """
    Replace the older “line over categories” plot with a compact, readable split-panel
    (CPU cluster vs radios), with numeric + uncertainty annotations.
    Output: power_params_line.png (paper folder).
    """
    import matplotlib.pyplot as plt

    colors = _palette()
    out_dir = _ensure_out_dir()
    _, params, _ = _load_config_params()

    cpu_labels = ["CPU idle", "CPU little (max)", "CPU big (max)"]
    cpu_vals = [params.P_idle, params.P_little_max, params.P_big_max]
    net_labels = ["WiFi idle", "WiFi active", "GPS on"]
    net_vals = [params.wifi_idle_power, params.wifi_active_power, params.gps_on_power]

    # Reference uncertainty (visual aid): ±5% of value.
    cpu_err = np.array(cpu_vals, dtype=float) * 0.05
    net_err = np.array(net_vals, dtype=float) * 0.05

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.6),
        gridspec_kw={"wspace": 0.22, "width_ratios": [1.15, 1.0]},
    )

    def lollipop(ax, labels, vals, errs, color):
        x = np.arange(len(labels))
        ax.vlines(x, 0, vals, color=color, alpha=0.22, linewidth=3.0, zorder=1)
        ax.errorbar(
            x,
            vals,
            yerr=errs,
            fmt="o",
            color=color,
            ecolor=colors["muted"],
            elinewidth=0.9,
            capsize=2.5,
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(True, axis="y", alpha=0.16)
        for xi, v, e in zip(x, vals, errs):
            ax.annotate(
                f"{v:.2f}±{e:.2f} W",
                xy=(xi, v),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.0,
                color=colors["ink"],
            )
        ax.set_xlim(-0.6, len(labels) - 0.4)

    lollipop(ax0, cpu_labels, cpu_vals, cpu_err, colors["accent_a"])
    ax0.set_title("Device power parameters (CPU)", loc="left", color=colors["ink"])
    ax0.set_ylabel("Power (W)")
    ax0.set_ylim(0, max(cpu_vals) * 1.16)

    lollipop(ax1, net_labels, net_vals, net_err, colors["accent_b"])
    ax1.set_title("Device power parameters (Radios)", loc="left", color=colors["ink"])
    ax1.set_ylim(0, max(net_vals) * 1.36)
    ax1.set_ylabel("Power (W)")

    fig.suptitle(
        "Component power calibration snapshot (current config)",
        x=0.52,
        y=0.99,
        fontsize=12.0,
        color=colors["ink"],
    )
    fig.savefig(out_dir / "power_params_line.png", bbox_inches="tight")
    plt.close(fig)


def make_activation_energies_compact() -> None:
    """
    Replace the older thick bar chart with a thin, labeled, non-overlapping plot.
    Output: activation_energies.png (paper folder).
    """
    import matplotlib.pyplot as plt

    colors = _palette()
    out_dir = _ensure_out_dir()
    _, params, _ = _load_config_params()

    labels = ["Resistance (Arrhenius)", "Capacity factor", "Calendar aging"]
    vals = np.array([params.Ea_R, params.Ea_cap, params.Ea_cal], dtype=float) / 1000.0  # kJ/mol
    errs = vals * 0.05

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(labels))[::-1]
    cols = [colors["accent_a"], colors["accent_b"], colors["accent_c"]]

    ax.barh(y, vals[::-1], color=cols[::-1], alpha=0.88, height=0.42)
    ax.errorbar(vals[::-1], y, xerr=errs[::-1], fmt="none", ecolor=colors["ink"], elinewidth=0.9, capsize=2.8, alpha=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Activation energy (kJ/mol)")
    ax.set_title("Activation energies (±5% reference uncertainty)", loc="left", color=colors["ink"])
    ax.grid(True, axis="x", alpha=0.16)

    for yi, v, e in zip(y, vals[::-1], errs[::-1]):
        ax.text(
            v + e + 0.8,
            yi,
            f"{v:.1f} ± {e:.1f}",
            va="center",
            ha="left",
            fontsize=9.2,
            color=colors["ink"],
        )

    ax.set_xlim(0, float((vals + errs).max()) * 1.25)
    fig.savefig(out_dir / "activation_energies.png", bbox_inches="tight")
    plt.close(fig)


def make_ocv_curve_with_data() -> None:
    """
    OCV figure for the paper:
    - show CALCE OCV data points (light)
    - show model curve from config (bold)
    - show ±RMSE band from fitting residuals
    - annotate key SOC points with coordinates
    Output: ocv_curve.png (paper folder).
    """
    import matplotlib.pyplot as plt

    from solution import BatteryPhysics, load_ocv_curve

    colors = _palette()
    out_dir = _ensure_out_dir()
    cfg, params, _ = _load_config_params()
    physics = BatteryPhysics(params)

    ocv_rel = ((cfg.get("data_paths", {}) or {}).get("calce_ocv_csv")) or "datasets/calce_ocv/cs2_8_ocv_curve.csv"
    ocv_path = Path(r"D:\MCM") / str(ocv_rel)
    soc_raw, v_raw = load_ocv_curve(ocv_path)
    soc_raw = np.asarray(soc_raw, dtype=float)
    v_raw = np.asarray(v_raw, dtype=float)
    mask = np.isfinite(soc_raw) & np.isfinite(v_raw) & (soc_raw > 0) & (soc_raw < 1.0)
    soc_raw = soc_raw[mask]
    v_raw = v_raw[mask]

    soc_grid = np.linspace(0.005, 0.995, 700)
    v_model = np.array([physics.get_ocv(float(s), params.ocv_temp_ref) for s in soc_grid], dtype=float)
    v_pred_raw = np.array([physics.get_ocv(float(s), params.ocv_temp_ref) for s in np.clip(soc_raw, 0.001, 0.999)], dtype=float)
    rmse = float(np.sqrt(np.mean((v_pred_raw - v_raw) ** 2))) if soc_raw.size > 0 else 0.02

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.scatter(soc_raw, v_raw, s=10, color=colors["muted"], alpha=0.22, linewidths=0.0, label="CALCE OCV points")
    ax.plot(soc_grid, v_model, color=colors["accent_a"], linewidth=1.25, label="Combined model (fit)")
    ax.fill_between(
        soc_grid,
        v_model - rmse,
        v_model + rmse,
        color=colors["accent_a"],
        alpha=0.10,
        linewidth=0.0,
        label=f"±RMSE band ({rmse*1000:.0f} mV)",
    )

    key_soc = np.array([0.10, 0.50, 0.90], dtype=float)
    key_v = np.array([physics.get_ocv(float(s), params.ocv_temp_ref) for s in key_soc], dtype=float)
    ax.scatter(
        key_soc,
        key_v,
        s=28,
        color=colors["accent_c"],
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label="Key SOC anchors",
    )
    for s, v in zip(key_soc, key_v):
        ax.annotate(
            f"({s:.2f}, {v:.3f})",
            xy=(s, v),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9.0,
            color=colors["ink"],
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor=colors["grid"], alpha=0.84),
            arrowprops=dict(arrowstyle="-", color=colors["accent_c"], lw=0.9, alpha=0.8),
        )

    ax.set_title("Open-circuit voltage curve: data-driven fit with stable endpoints", loc="left", color=colors["ink"])
    ax.set_xlabel("SOC (fraction)")
    ax.set_ylabel("OCV (V)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.16)
    ax.legend(loc="lower right")

    # Zoom inset for low-SOC end behavior (where polynomial fits often fail).
    ins = ax.inset_axes([0.08, 0.14, 0.35, 0.45])
    ins.scatter(soc_raw, v_raw, s=8, color=colors["muted"], alpha=0.18, linewidths=0.0)
    ins.plot(soc_grid, v_model, color=colors["accent_a"], linewidth=1.0)
    ins.fill_between(soc_grid, v_model - rmse, v_model + rmse, color=colors["accent_a"], alpha=0.10, linewidth=0.0)
    ins.set_xlim(0.0, 0.12)
    v_low = v_model[soc_grid <= 0.12]
    ins.set_ylim(float(np.nanmin(v_low)) - 0.05, float(np.nanmax(v_low)) + 0.05)
    ins.set_title("Low-SOC zoom", fontsize=9.0, pad=2.0, loc="left")
    ins.grid(True, alpha=0.12)
    ins.tick_params(labelsize=8)

    fig.savefig(out_dir / "ocv_curve.png", bbox_inches="tight")
    plt.close(fig)


def make_rint_surface_and_contour() -> None:
    """
    Rebuild internal resistance figures with consistent style.
    Outputs:
      - rint_surface_3d.png
      - rint_contour.png
    """
    import matplotlib.pyplot as plt

    from solution import BatteryPhysics

    colors = _palette()
    out_dir = _ensure_out_dir()
    _, params, _ = _load_config_params()
    physics = BatteryPhysics(params)

    soc = np.linspace(0.03, 0.98, 120)
    temp_c = np.linspace(-10.0, 40.0, 120)
    S, Tc = np.meshgrid(soc, temp_c)
    Tk = Tc + 273.15

    # Compute in mΩ for readability.
    R = np.vectorize(lambda z, tk: physics.get_r_int(float(z), float(tk), params.SOH_init))(S, Tk) * 1000.0

    # 2D contour (paper-friendly)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    r_min = float(np.nanmin(R))
    r_hi = float(np.nanquantile(R, 0.985))
    levels = np.linspace(r_min, r_hi, 18)
    cf = ax.contourf(S, Tc, np.clip(R, r_min, r_hi), levels=levels, cmap="cividis")
    cs = ax.contour(S, Tc, np.clip(R, r_min, r_hi), levels=levels[::3], colors="white", linewidths=0.6, alpha=0.55)
    ax.clabel(cs, inline=True, fontsize=8.0, fmt="%.0f")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r"$R_{int}$ (m$\Omega$)")
    ax.set_xlabel("SOC (fraction)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(r"Internal resistance landscape $R_{int}(\mathrm{SOC},T)$ (SOH=1)", loc="left", color=colors["ink"])
    ax.grid(False)
    fig.savefig(out_dir / "rint_contour.png", bbox_inches="tight")
    plt.close(fig)

    # 3D surface (show “wall” geometry)
    fig = plt.figure(figsize=(7.2, 4.8))
    ax3 = fig.add_subplot(111, projection="3d")
    surf = ax3.plot_surface(
        S,
        Tc,
        np.clip(R, r_min, r_hi),
        cmap="cividis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.92,
        rcount=100,
        ccount=100,
    )
    # Contour projection (ground)
    z0 = r_min - 0.05 * (r_hi - r_min)
    ax3.contour(S, Tc, np.clip(R, r_min, r_hi), zdir="z", offset=z0, levels=12, cmap="cividis", linewidths=0.7, alpha=0.7)
    ax3.set_zlim(z0, r_hi * 1.03)
    ax3.set_xlabel("SOC (fraction)")
    ax3.set_ylabel("Temperature (°C)")
    ax3.set_zlabel(r"$R_{int}$ (m$\Omega$)")
    ax3.set_title("Resistance geometry (3D surface + contour projection)", loc="left", color=colors["ink"])
    ax3.view_init(elev=28, azim=-55)
    # Clean panes
    ax3.xaxis.pane.set_alpha(0.0)
    ax3.yaxis.pane.set_alpha(0.0)
    ax3.zaxis.pane.set_alpha(0.0)
    ax3.grid(True, alpha=0.10)
    fig.colorbar(surf, ax=ax3, pad=0.08, fraction=0.04, label=r"$R_{int}$ (m$\Omega$)")
    fig.savefig(out_dir / "rint_surface_3d.png", bbox_inches="tight")
    plt.close(fig)


def make_cross_scenario_metrics() -> None:
    """
    Rebuild cross-scenario summary plots to match paper captions (FULL MODEL only).
    Outputs:
      - comparison_t_end_s.png
      - comparison_temp_max_k.png
    """
    import matplotlib.pyplot as plt

    colors = _palette()
    out_dir = _ensure_out_dir()

    scenario_stems = ["Day_in_the_Life", "Gaming", "Video_Streaming", "Winter_Usage"]
    pretty = {
        "Day_in_the_Life": "Day in the Life",
        "Gaming": "Gaming",
        "Video_Streaming": "Video Streaming",
        "Winter_Usage": "Winter Usage",
    }
    t_end_s = []
    temp_max_c = []
    for s in scenario_stems:
        df = _read_outputs_csv(s)
        t_end_s.append(float(df["t_s"].iloc[-1]))
        temp_max_c.append(float(df["temp_k"].max() - 273.15))

    # TTE comparison
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.arange(len(scenario_stems))
    bar = ax.bar(x, t_end_s, width=0.55, color=colors["accent_a"], alpha=0.88, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[s] for s in scenario_stems], rotation=12, ha="right")
    ax.set_ylabel("Time-to-empty (s)")
    ax.set_title("Cross-scenario time-to-empty (full model)", loc="left", color=colors["ink"])
    ax.grid(True, axis="y", alpha=0.16)
    for rect, sec in zip(bar, t_end_s):
        ax.annotate(
            f"{sec:.0f}s\n({_format_time_h(sec)})",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=colors["ink"],
        )
    fig.savefig(out_dir / "comparison_t_end_s.png", bbox_inches="tight")
    plt.close(fig)

    # Peak temperature comparison
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bar = ax.bar(x, temp_max_c, width=0.55, color=colors["accent_c"], alpha=0.88, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[s] for s in scenario_stems], rotation=12, ha="right")
    ax.set_ylabel("Peak temperature (°C)")
    ax.set_title("Cross-scenario peak temperature (full model)", loc="left", color=colors["ink"])
    ax.grid(True, axis="y", alpha=0.16)
    for rect, v in zip(bar, temp_max_c):
        ax.annotate(
            f"{v:.1f}°C",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color=colors["ink"],
        )
    fig.savefig(out_dir / "comparison_temp_max_k.png", bbox_inches="tight")
    plt.close(fig)


def make_phase_portrait() -> None:
    """
    Rebuild SOC–Temperature phase portrait with thin lines and endpoint callouts.
    Output: phase_portrait.png (paper folder).
    """
    import matplotlib.pyplot as plt

    colors = _palette()
    out_dir = _ensure_out_dir()

    series = [
        ("Video_Streaming", "Video Streaming", colors["accent_b"]),
        ("Gaming", "Gaming", colors["accent_d"]),
        ("Winter_Usage", "Winter Usage", colors["accent_a"]),
        ("Day_in_the_Life", "Day in the Life", colors["accent_c"]),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for stem, label, col in series:
        df = _read_outputs_csv(stem)
        soc = df["soc"].to_numpy(dtype=float) * 100.0
        temp = df["temp_k"].to_numpy(dtype=float) - 273.15
        ax.plot(soc, temp, color=col, linewidth=1.05, alpha=0.95, label=label)
        # Endpoint markers
        ax.scatter([soc[0], soc[-1]], [temp[0], temp[-1]], s=18, color=col, edgecolor="white", linewidth=0.6, zorder=4)
        ax.annotate(
            f"end {soc[-1]:.1f}%",
            xy=(soc[-1], temp[-1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8.6,
            color=colors["ink"],
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor=colors["grid"], alpha=0.80),
            arrowprops=dict(arrowstyle="-", color=col, lw=0.85, alpha=0.75),
        )

    ax.set_title("Phase portrait: SOC vs temperature (coupled dynamics)", loc="left", color=colors["ink"])
    ax.set_xlabel("State of charge (%)")
    ax.set_ylabel("Temperature (°C)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.16)
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(out_dir / "phase_portrait.png", bbox_inches="tight")
    plt.close(fig)


def make_day_in_life_soc_temp() -> None:
    """
    Rebuild Day-in-the-Life figure with phase bands and non-overlapping labels.
    Output: day_in_life_soc_temp.png (paper folder).
    """
    import matplotlib.pyplot as plt

    colors = _palette()
    out_dir = _ensure_out_dir()

    df = _read_outputs_csv("Day_in_the_Life")
    t_h = df["t_s"].to_numpy(dtype=float) / 3600.0
    soc_pct = df["soc"].to_numpy(dtype=float) * 100.0
    temp_c = df["temp_k"].to_numpy(dtype=float) - 273.15
    p_total = df["p_total"].to_numpy(dtype=float) if "p_total" in df.columns else None

    fig, ax = plt.subplots(figsize=(7.8, 4.1))

    # Phase bands (match config segments; even if cutoff occurs early, they serve as context).
    phases = [
        ("Standby", 0.0, 4.0, "#E0F2FE"),
        ("Gaming", 4.0, 5.0, "#FEE2E2"),
        ("Social", 5.0, 10.0, "#ECFDF5"),
        ("Video", 10.0, 12.0, "#FEF3C7"),
    ]
    for name, a, b, col in phases:
        ax.axvspan(a, b, color=col, alpha=0.60, lw=0)
        ax.text((a + b) / 2, 103.5, name, ha="center", va="top", fontsize=9.2, color=colors["muted"])

    ax.plot(t_h, soc_pct, color=colors["accent_a"], linewidth=1.10, label="SOC")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("SOC (%)")
    ax.set_ylim(0, 105)

    ax2 = ax.twinx()
    ax2.plot(t_h, temp_c, color=colors["accent_c"], linewidth=1.05, label="Temperature")
    ax2.set_ylabel("Temperature (°C)")

    # Cutoff marker at end time
    t_end = float(t_h[-1])
    ax.axvline(t_end, color=colors["accent_f"], linewidth=0.9, alpha=0.85)
    ax.annotate(
        f"cutoff @ {t_end:.2f} h\nSOC={soc_pct[-1]:.1f}%",
        xy=(t_end, soc_pct[-1]),
        xytext=(-90, 18),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9.0,
        color=colors["ink"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=colors["grid"], alpha=0.86),
        arrowprops=dict(arrowstyle="-", color=colors["accent_f"], lw=0.85, alpha=0.8),
    )

    if p_total is not None:
        # Annotate average power for each phase where data exists.
        for name, a, b, _ in phases:
            m = (t_h >= a) & (t_h < min(b, t_end))
            if np.any(m):
                p_avg = float(np.mean(p_total[m]))
                ax.text((a + min(b, t_end)) / 2, 3.0, f"{p_avg:.1f} W", ha="center", va="bottom", fontsize=8.8, color=colors["ink"])

    ax.set_title("Day-in-the-Life: mixed-use trajectory with phase context", loc="left", color=colors["ink"])
    # Combined legend
    h0, l0 = ax.get_legend_handles_labels()
    h1, l1 = ax2.get_legend_handles_labels()
    ax.legend(h0 + h1, l0 + l1, loc="upper right", framealpha=0.92)
    fig.savefig(out_dir / "day_in_life_soc_temp.png", bbox_inches="tight")
    plt.close(fig)


def make_climate_stress_gaming() -> None:
    """
    Re-run the climate stress test (Gaming at 0/25/40°C) and render a clean overlay plot.
    Output: climate_stress_Gaming.png (paper folder).
    """
    import matplotlib.pyplot as plt

    from solution import BatteryParams, PowerSystem, build_scenario, load_yaml_config

    colors = _palette()
    out_dir = _ensure_out_dir()

    cfg = load_yaml_config(Path(r"D:\MCM\config.yaml"))
    params = BatteryParams.from_dict(cfg["battery_params"])
    system = PowerSystem(params)

    climate_cfg = cfg.get("climate_stress", {}) or {}
    temps_c = list(map(float, climate_cfg.get("temps_c", [0, 25, 40])))
    base_name = str(climate_cfg.get("base_scenario", "Gaming"))

    base_cfg = None
    for sc in cfg.get("scenarios", []) or []:
        if str(sc.get("name", "")).strip().lower() == base_name.strip().lower():
            base_cfg = sc
            break
    if base_cfg is None:
        raise ValueError(f"Base scenario not found for climate stress: {base_name}")

    results = []
    for tc in temps_c:
        sc_cfg = dict(base_cfg)
        sc_cfg["env_temp_k"] = float(tc + 273.15)
        sc_cfg["name"] = f"{base_cfg['name']} ({tc:.0f}°C)"
        scenario = build_scenario(sc_cfg)
        results.append(system.solve(scenario, cfg.get("solver", {}) or {}))

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.0), sharex=True, gridspec_kw={"hspace": 0.10})
    line_cols = [colors["accent_a"], colors["accent_b"], colors["accent_d"]]
    for idx, (tc, res) in enumerate(zip(temps_c, results)):
        col = line_cols[idx % len(line_cols)]
        t_hr = res.t / 3600.0
        label = f"{tc:.0f}°C  (TTE={_format_time_h(res.time_to_empty_s)})"
        axes[0].plot(t_hr, res.soc * 100.0, color=col, linewidth=1.05, label=label)
        axes[1].plot(t_hr, res.voltages["V_term"], color=col, linewidth=1.05)
        axes[2].plot(t_hr, res.y[2, :] - 273.15, color=col, linewidth=1.05)
        # End markers
        axes[0].scatter([t_hr[-1]], [res.soc[-1] * 100.0], s=18, color=col, edgecolor="white", linewidth=0.6, zorder=4)
        axes[1].scatter([t_hr[-1]], [res.voltages["V_term"][-1]], s=18, color=col, edgecolor="white", linewidth=0.6, zorder=4)
        axes[2].scatter([t_hr[-1]], [res.y[2, -1] - 273.15], s=18, color=col, edgecolor="white", linewidth=0.6, zorder=4)

    axes[0].set_ylabel("SOC (%)")
    axes[1].set_ylabel("Voltage (V)")
    axes[2].set_ylabel("Temp (°C)")
    axes[2].set_xlabel("Time (h)")
    axes[0].set_title("Climate stress test (Gaming): regime shift across ambient temperature", loc="left", color=colors["ink"])
    axes[0].legend(loc="upper right", ncol=1, fontsize=9.0, framealpha=0.92)
    for ax in axes:
        ax.grid(True, alpha=0.16)
    fig.savefig(out_dir / "climate_stress_Gaming.png", bbox_inches="tight")
    plt.close(fig)


def make_3d_trajectory() -> None:
    """
    Rebuild 3D coupling trajectory with clean styling.
    Output: 3d_trajectory.png (paper folder).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    from solution import BatteryPhysics

    colors = _palette()
    out_dir = _ensure_out_dir()
    _, params, _ = _load_config_params()
    physics = BatteryPhysics(params)

    series = [
        ("Video_Streaming", "Video Streaming"),
        ("Gaming", "Gaming"),
        ("Winter_Usage", "Winter Usage"),
        ("Day_in_the_Life", "Day in the Life"),
    ]

    fig = plt.figure(figsize=(7.6, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    all_r = []
    trajs = []
    for stem, label in series:
        df = _read_outputs_csv(stem)
        t_h = df["t_s"].to_numpy(dtype=float) / 3600.0
        soc = df["soc"].to_numpy(dtype=float) * 100.0
        temp_c = df["temp_k"].to_numpy(dtype=float) - 273.15
        soh = df["soh"].to_numpy(dtype=float) if "soh" in df.columns else np.full_like(t_h, params.SOH_init)
        r_mohm = np.array(
            [physics.get_r_int(float(s / 100.0), float(tc + 273.15), float(sh)) for s, tc, sh in zip(soc, temp_c, soh)],
            dtype=float,
        ) * 1000.0
        all_r.append(r_mohm)
        trajs.append((soc, temp_c, t_h, r_mohm, label))

    r_all = np.concatenate(all_r) if all_r else np.array([0.0, 1.0])
    rmin = float(np.nanmin(r_all))
    rmax = float(np.nanquantile(r_all, 0.995))

    cmap = plt.get_cmap("cividis")
    norm = plt.Normalize(vmin=rmin, vmax=rmax)

    for soc, temp_c, t_h, r_mohm, label in trajs:
        pts = np.column_stack([soc, temp_c, t_h])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, cmap=cmap, norm=norm)
        lc.set_array(np.clip(r_mohm[:-1], rmin, rmax))
        lc.set_linewidth(2.2)
        lc.set_alpha(0.95)
        ax.add_collection(lc)
        # Add a small label near end point
        ax.text(soc[-1], temp_c[-1], t_h[-1], f" {label}", fontsize=9.0, color=colors["ink"])

    ax.set_title("3D coupling trajectory: (SOC, Temperature, Time) colored by $R_{int}$", loc="left", color=colors["ink"])
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_zlabel("Time (h)")
    ax.invert_xaxis()
    ax.view_init(elev=24, azim=-55)
    ax.grid(True, alpha=0.10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.10, fraction=0.045)
    cbar.set_label(r"$R_{int}$ (m$\Omega$)")

    fig.savefig(out_dir / "3d_trajectory.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _try_set_science_style()
    make_validation_alignment_figure()
    make_uncertainty_ecdf_raincloud()
    make_ablation_tradeoff_figure()
    make_uncertainty_band()
    make_power_params_compact()
    make_activation_energies_compact()
    make_ocv_curve_with_data()
    make_rint_surface_and_contour()
    make_cross_scenario_metrics()
    make_day_in_life_soc_temp()
    make_phase_portrait()
    make_climate_stress_gaming()
    make_3d_trajectory()
    make_phantom_battery_schematic()
    make_soa_with_collapse_region()
    print("Wrote figures to D:\\MCM-Article")


if __name__ == "__main__":
    main()

