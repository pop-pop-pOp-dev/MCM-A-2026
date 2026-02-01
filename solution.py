"""
Advanced Thermo-Electric Coupled Dynamical System (ACM)
Problem A: Modeling Smartphone Battery Drain

This module provides a modular, research-grade simulation framework with:
- Coupled ODEs for SOC, temperature, and SOH
- Nonlinear electrical model (OCV, internal resistance)
- Thermal model (Joule heat, logic heat, convection)
- Degradation model (cycle + calendar aging)
- Publication-ready visualizations
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


@dataclass
class BatteryParams:
    # Electrical parameters
    Q_design_ah: float  # Design capacity [Ah]
    V_cutoff: float  # Cutoff voltage [V]

    # Thermal parameters
    m: float  # Effective mass of device [kg]
    Cp: float  # Effective heat capacity [J/(kg*K)]
    hA: float  # Lumped convection coefficient [W/K]
    T_env: float  # Ambient temperature [K]
    # OCV model configuration
    ocv_model: str  # "polynomial" or "combined"
    ocv_coeffs: Tuple[float, ...]  # polynomial: high->const, combined: [k0..k4]
    ocv_temp_ref: float  # Reference temperature for OCV correction [K]
    ocv_temp_coeff: float  # Linear OCV temp coefficient [V/K]
    ocv_temp_quad: float  # Quadratic OCV temp coefficient [V/K^2]

    # Internal resistance model parameters
    R_ref: float  # Reference internal resistance at T_ref and SOC ~ 0.5 [Ohm]
    T_ref: float  # Reference temperature [K]
    Ea_R: float  # Arrhenius activation energy for resistance [J/mol]
    R_gas: float  # Universal gas constant [J/(mol*K)]
    r_soc_coeff: float  # Strength of low-SOC resistance rise [-]
    r_soc_k: float  # Exponential sharpness for low-SOC rise [-]
    r_soh_coeff: float  # Resistance increase per SOH loss [-]

    # Coulombic efficiency model parameters
    eta_coulomb_ref: float  # Coulombic efficiency at T_ref [-]
    eta_coulomb_temp_coeff: float  # Linear drop per K away from T_ref [-/K]

    # Degradation parameters
    k_cycle: float  # Cycle aging coefficient [1/(A*s)]
    k_cal: float  # Calendar aging coefficient [1/s]
    Ea_cal: float  # Activation energy for calendar aging [J/mol]
    alpha_soc: float  # SOC acceleration factor for calendar aging [-]
    Ea_cap: float  # Activation energy for capacity temperature dependence [J/mol]

    # KiBaM parameters
    kibam_enabled: bool  # Enable KiBaM recovery dynamics
    kibam_c: float  # Available charge fraction [-]
    kibam_k: float  # Diffusion rate constant [1/s]

    # Polarization RC branch parameters
    rc_enabled: bool  # Enable RC polarization dynamics
    r_polar: float  # Polarization resistance [Ohm]
    c_polar: float  # Polarization capacitance [F]

    # Screen parameters
    P_max_scr: float  # Max screen power density [W/m^2]
    A_area: float  # Screen active area [m^2]
    gamma_scr: float  # Brightness nonlinearity exponent [-]

    # CPU parameters
    P_idle: float  # CPU idle power [W]
    P_little_max: float  # Little cores max power [W]
    P_big_max: float  # Big cores max power [W]
    u_thresh: float  # Threshold for big cores [-]
    T_throttle: float  # Throttle midpoint temperature [K] ~ 45C
    throttle_k: float  # Throttle sigmoid steepness [-]

    # Network parameters
    # Macro communication states (coarse-grained)
    wifi_idle_power: float  # WiFi connected idle power [W]
    wifi_active_power: float  # WiFi active transfer power [W]
    gps_on_power: float  # GPS navigation power [W]

    # OS feedback (low power mode)
    lpm_soc_thresh: float  # SOC threshold for LPM [-]
    lpm_k: float  # Sigmoid steepness [-]
    lpm_min_factor: float  # Minimum scaling factor [-]

    # Charging parameters
    charger_efficiency: float  # Charger efficiency [-]
    v_stress_thresh: float  # Voltage stress threshold [V]
    k_stress: float  # Voltage stress coefficient [1/s]
    beta_stress: float  # Voltage stress exponential slope [1/V]

    # Initial state
    SOC_init: float  # Initial state of charge (available) [-]
    T_init: float  # Initial internal temperature [K]
    SOH_init: float  # Initial state of health [-]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatteryParams":
        payload = dict(data)
        payload["ocv_coeffs"] = tuple(payload["ocv_coeffs"])
        return cls(**payload)


@dataclass
class Scenario:
    name: str
    duration_s: float
    dt: float
    env_temp_k: float
    brightness: Callable[[float], float]
    apl: Callable[[float], float]
    cpu_util: Callable[[float], float]
    wifi_state: Callable[[float], float]
    gps_state: Callable[[float], float]
    charger_power: Callable[[float], float]
    eta_radiation: float


@dataclass
class SimulationResult:
    t: np.ndarray
    y: np.ndarray  # shape: (4, n) => y1, y2, T, SOH
    soc: np.ndarray
    voltages: Dict[str, np.ndarray]
    powers: Dict[str, np.ndarray]
    time_to_empty_s: float
    scenario: Scenario


class ComponentModel:
    def __init__(self, params: BatteryParams):
        self.params = params

    def screen_power(self, brightness: float, apl: float) -> float:
        # OLED power depends on brightness and average picture level
        beta = np.clip(brightness, 0.0, 1.0)
        apl = np.clip(apl, 0.0, 1.0)
        return self.params.P_max_scr * self.params.A_area * (beta ** self.params.gamma_scr) * apl

    def cpu_power(self, util: float, temp_k: float) -> float:
        # big.LITTLE nonlinear model with thermal throttling
        util = np.clip(util, 0.0, 1.0)
        if util < self.params.u_thresh:
            p_cpu = self.params.P_little_max * (util / max(self.params.u_thresh, 1e-6))
        else:
            p_cpu = self.params.P_little_max + self.params.P_big_max * (
                (util - self.params.u_thresh) / max(1.0 - self.params.u_thresh, 1e-6)
            )
        phi = 1.0 / (1.0 + np.exp(self.params.throttle_k * (temp_k - self.params.T_throttle)))
        return phi * (self.params.P_idle + p_cpu)

    def wifi_power(self, state: float) -> float:
        # Macro states: 0=off, 1=idle, 2=active
        state_idx = int(np.clip(round(state), 0, 2))
        if state_idx == 1:
            return self.params.wifi_idle_power
        if state_idx == 2:
            return self.params.wifi_active_power
        return 0.0

    def gps_power(self, state: float) -> float:
        # Macro states: 0=off, 1=on
        state_idx = int(np.clip(round(state), 0, 1))
        return self.params.gps_on_power if state_idx == 1 else 0.0


class BatteryPhysics:
    def __init__(self, params: BatteryParams):
        self.params = params

    def get_ocv(self, soc: float, temp_k: float | None = None) -> float:
        model = getattr(self.params, "ocv_model", "polynomial").lower()
        if model == "combined":
            k0, k1, k2, k3, k4 = self.params.ocv_coeffs
            s = np.clip(soc, 0.001, 0.999)
            base = k0 - k1 / s - k2 * s + k3 * np.log(s) + k4 * np.log(1.0 - s)
        else:
            soc = np.clip(soc, 0.0, 1.0)
            base = float(np.polyval(self.params.ocv_coeffs, soc))
        if temp_k is None:
            temp_k = self.params.ocv_temp_ref
        d_t = float(temp_k - self.params.ocv_temp_ref)
        temp_corr = self.params.ocv_temp_coeff * d_t + self.params.ocv_temp_quad * (d_t ** 2)
        return float(base + temp_corr)

    def get_r_int(self, soc: float, temp_k: float, soh: float) -> float:
        # Arrhenius temperature dependence for resistance
        temp_k = max(temp_k, 250.0)
        arrhenius = np.exp(self.params.Ea_R / self.params.R_gas * (1.0 / temp_k - 1.0 / self.params.T_ref))

        # Low-SOC exponential rise in internal resistance
        soc = np.clip(soc, 0.0, 1.0)
        soc_factor = 1.0 + self.params.r_soc_coeff * np.exp(-self.params.r_soc_k * soc)

        # SOH degradation factor (resistance increases as SOH drops)
        soh = np.clip(soh, 0.05, 1.0)
        soh_factor = 1.0 + self.params.r_soh_coeff * (1.0 - soh)

        return self.params.R_ref * arrhenius * soc_factor * soh_factor

    def coulomb_efficiency(self, temp_k: float) -> float:
        # Simple linear degradation away from reference temperature
        eta = self.params.eta_coulomb_ref - self.params.eta_coulomb_temp_coeff * abs(temp_k - self.params.T_ref)
        return float(np.clip(eta, 0.90, 1.0))

    def capacity_temp_factor(self, temp_k: float) -> float:
        # Arrhenius-like temperature dependence of usable capacity
        temp_k = max(temp_k, 250.0)
        return float(np.exp(-self.params.Ea_cap / self.params.R_gas * (1.0 / temp_k - 1.0 / self.params.T_ref)))

    def lpm_factor(self, soc: float) -> float:
        # Low power mode feedback: reduces demand when SOC is low
        soc = np.clip(soc, 0.0, 1.0)
        sigmoid = 1.0 / (1.0 + np.exp(-self.params.lpm_k * (soc - self.params.lpm_soc_thresh)))
        return float(self.params.lpm_min_factor + (1.0 - self.params.lpm_min_factor) * sigmoid)


class PowerSystem:
    def __init__(self, params: BatteryParams):
        self.params = params
        self.components = ComponentModel(params)
        self.physics = BatteryPhysics(params)

    def _solve_current(self, ocv: float, r_int: float, p_total: float) -> float:
        # Solve R*I^2 - OCV*I + P = 0 for discharge current I >= 0
        if r_int <= 1e-9:
            return p_total / max(ocv, 1e-6)
        disc = ocv ** 2 - 4.0 * r_int * p_total
        disc = max(disc, 0.0)
        return (ocv - np.sqrt(disc)) / (2.0 * r_int)

    def _component_powers(self, t: float, temp_k: float, scenario: Scenario) -> Dict[str, float]:
        brightness = scenario.brightness(t)
        apl = scenario.apl(t)
        util = scenario.cpu_util(t)
        wifi_state = scenario.wifi_state(t)
        gps_state = scenario.gps_state(t)
        charger_power = scenario.charger_power(t)

        p_scr = self.components.screen_power(brightness, apl)
        p_cpu = self.components.cpu_power(util, temp_k)
        p_wifi = self.components.wifi_power(wifi_state)
        p_gps = self.components.gps_power(gps_state)
        p_load = p_scr + p_cpu + p_wifi + p_gps
        p_charge = max(charger_power, 0.0) * self.params.charger_efficiency
        p_net = p_wifi
        p_total = p_load - p_charge

        return {
            "P_screen": p_scr,
            "P_cpu": p_cpu,
            "P_net": p_net,
            "P_wifi": p_wifi,
            "P_gps": p_gps,
            "P_load": p_load,
            "P_charge": p_charge,
            "P_total": p_total,
        }

    def _capacity_total(self, temp_k: float, soh: float) -> float:
        # Total capacity in Ah adjusted by temperature and SOH
        cap_factor = self.physics.capacity_temp_factor(temp_k)
        return self.params.Q_design_ah * max(soh, 0.05) * cap_factor

    def _soc_from_state(self, y1: float, temp_k: float, soh: float) -> float:
        c_total = self._capacity_total(temp_k, soh)
        if not self.params.kibam_enabled:
            return float(np.clip(y1 / max(c_total, 1e-6), 0.0, 1.0))
        c_available = max(self.params.kibam_c * c_total, 1e-6)
        return float(np.clip(y1 / c_available, 0.0, 1.0))

    def _voltage(self, t: float, y: np.ndarray, scenario: Scenario) -> Tuple[float, float, float]:
        y1, y2, temp_k, soh, v_rc = y
        soc = self._soc_from_state(y1, temp_k, soh)
        ocv = self.physics.get_ocv(soc, temp_k)
        r_int = self.physics.get_r_int(soc, temp_k, soh)
        powers = self._component_powers(t, temp_k, scenario)
        current = self._solve_current(ocv, r_int, powers["P_total"])
        v_polar = v_rc if self.params.rc_enabled else 0.0
        v_term = ocv - current * r_int - v_polar
        return v_term, ocv, current

    def derivative(self, t: float, y: np.ndarray, scenario: Scenario) -> np.ndarray:
        y1, y2, temp_k, soh, v_rc = y
        soh_c = np.clip(soh, 0.05, 1.0)
        soc_c = self._soc_from_state(y1, temp_k, soh_c)

        powers = self._component_powers(t, temp_k, scenario)
        phi_soc = self.physics.lpm_factor(soc_c)
        powers["P_total"] *= phi_soc
        powers["P_screen"] *= phi_soc
        powers["P_cpu"] *= phi_soc
        powers["P_net"] *= phi_soc

        ocv = self.physics.get_ocv(soc_c, temp_k)
        r_int = self.physics.get_r_int(soc_c, temp_k, soh_c)
        current = self._solve_current(ocv, r_int, powers["P_total"])
        v_polar = v_rc if self.params.rc_enabled else 0.0
        v_term = ocv - current * r_int - v_polar

        # KiBaM dynamics (optional)
        if self.params.kibam_enabled:
            h1 = y1 / max(self.params.kibam_c, 1e-6)
            h2 = y2 / max(1.0 - self.params.kibam_c, 1e-6)
            dy1 = -(current / 3600.0) + self.params.kibam_k * (h2 - h1)
            dy2 = -self.params.kibam_k * (h2 - h1)
        else:
            dy1 = -(current / 3600.0)
            dy2 = 0.0

        # Polarization RC dynamics
        if self.params.rc_enabled:
            tau = max(self.params.r_polar * self.params.c_polar, 1e-6)
            dv_rc = -(v_rc / tau) + (current / max(self.params.c_polar, 1e-6))
        else:
            dv_rc = 0.0

        # Thermal dynamics (Joule + load heat - convection)
        p_logic_heat = powers["P_load"] * (1.0 - scenario.eta_radiation)
        p_charge_heat = powers["P_charge"] * (1.0 - self.params.charger_efficiency)
        d_temp = (
            current ** 2 * r_int
            + p_logic_heat
            + p_charge_heat
            - self.params.hA * (temp_k - scenario.env_temp_k)
        ) / (self.params.m * self.params.Cp)

        # Degradation dynamics (cycle + calendar aging)
        arr_cal = np.exp(-self.params.Ea_cal / (self.params.R_gas * max(temp_k, 250.0)))
        v_stress = max(v_term - self.params.v_stress_thresh, 0.0)
        stress_term = self.params.k_stress * np.exp(self.params.beta_stress * v_stress)
        d_soh = (
            -self.params.k_cycle * abs(current) * (1.0 + self.params.alpha_soc * soc_c)
            - self.params.k_cal * arr_cal
            - stress_term
        )

        # If voltage is below cutoff, freeze dynamics to help event detection
        if v_term <= self.params.V_cutoff:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array([dy1, dy2, d_temp, d_soh, dv_rc])

    def solve(self, scenario: Scenario, solver_cfg: Dict[str, Any] | None = None) -> SimulationResult:
        c_total = self._capacity_total(self.params.T_init, self.params.SOH_init)
        if self.params.kibam_enabled:
            y1_0 = self.params.SOC_init * (self.params.kibam_c * c_total)
            y2_0 = self.params.SOC_init * ((1.0 - self.params.kibam_c) * c_total)
        else:
            y1_0 = self.params.SOC_init * c_total
            y2_0 = 0.0
        v_rc0 = 0.0
        y0 = np.array([y1_0, y2_0, self.params.T_init, self.params.SOH_init, v_rc0], dtype=float)
        t_eval = np.arange(0.0, scenario.duration_s + scenario.dt, scenario.dt)

        def cutoff_event(t: float, y: np.ndarray) -> float:
            v_term, _, _ = self._voltage(t, y, scenario)
            return v_term - self.params.V_cutoff

        cutoff_event.terminal = True
        cutoff_event.direction = -1.0

        solver_cfg = solver_cfg or {}
        method = str(solver_cfg.get("method", "DOP853"))
        rtol = float(solver_cfg.get("rtol", 1e-6))
        atol = float(solver_cfg.get("atol", 1e-8))
        max_step = solver_cfg.get("max_step")
        if max_step is not None:
            max_step = float(max_step)

        sol = solve_ivp(
            fun=lambda t, y: self.derivative(t, y, scenario),
            t_span=(0.0, scenario.duration_s),
            y0=y0,
            t_eval=t_eval,
            events=cutoff_event,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        t = sol.t
        y = sol.y
        soc = np.array([self._soc_from_state(y[0, i], y[2, i], y[3, i]) for i in range(t.size)])

        # Determine time-to-empty from event
        if sol.t_events and len(sol.t_events[0]) > 0:
            t_empty = float(sol.t_events[0][0])
        else:
            t_empty = float(t[-1])

        # Recompute power and voltage traces for plotting
        n = t.size
        v_term = np.zeros(n)
        v_ocv = np.zeros(n)
        current = np.zeros(n)
        p_screen = np.zeros(n)
        p_cpu = np.zeros(n)
        p_net = np.zeros(n)
        p_wifi = np.zeros(n)
        p_gps = np.zeros(n)
        p_load = np.zeros(n)
        p_charge = np.zeros(n)
        p_total = np.zeros(n)
        p_heat = np.zeros(n)

        for i in range(n):
            y1_i, y2_i, temp_i, soh_i, v_rc_i = y[:, i]
            soc_i = self._soc_from_state(y1_i, temp_i, soh_i)
            powers = self._component_powers(t[i], temp_i, scenario)
            phi_soc = self.physics.lpm_factor(soc_i)
            powers["P_total"] *= phi_soc
            powers["P_screen"] *= phi_soc
            powers["P_cpu"] *= phi_soc
            powers["P_net"] *= phi_soc
            ocv = self.physics.get_ocv(soc_i, temp_i)
            r_int = self.physics.get_r_int(soc_i, temp_i, soh_i)
            i_cur = self._solve_current(ocv, r_int, powers["P_total"])
            v_t = ocv - i_cur * r_int - (v_rc_i if self.params.rc_enabled else 0.0)
            v_term[i] = v_t
            v_ocv[i] = ocv
            current[i] = i_cur
            p_screen[i] = powers["P_screen"]
            p_cpu[i] = powers["P_cpu"]
            p_net[i] = powers["P_net"]
            p_wifi[i] = powers["P_wifi"]
            p_gps[i] = powers["P_gps"]
            p_load[i] = powers["P_load"]
            p_charge[i] = powers["P_charge"]
            p_total[i] = powers["P_total"]
            p_charge_heat = powers["P_charge"] * (1.0 - self.params.charger_efficiency)
            p_heat[i] = i_cur ** 2 * r_int + p_load[i] * (1.0 - scenario.eta_radiation) + p_charge_heat

        voltages = {"V_term": v_term, "V_ocv": v_ocv, "I": current, "V_rc": y[4, :] if self.params.rc_enabled else np.zeros(n)}
        powers = {
            "P_screen": p_screen,
            "P_cpu": p_cpu,
            "P_net": p_net,
            "P_wifi": p_wifi,
            "P_gps": p_gps,
            "P_load": p_load,
            "P_charge": p_charge,
            "P_total": p_total,
            "P_heat": p_heat,
        }

        return SimulationResult(
            t=t,
            y=y,
            soc=soc,
            voltages=voltages,
            powers=powers,
            time_to_empty_s=t_empty,
            scenario=scenario,
        )


class Visualizer:
    def __init__(self):
        try:
            plt.style.use(["science", "no-latex", "grid"])
        except OSError:
            sns.set_context("paper")
            sns.set_style("whitegrid")

    def realtime_dashboard(
        self,
        result: SimulationResult,
        interval_ms: int = 80,
        max_points: int | None = None,
        save_dir: Path | None = None,
        frame_stride: int = 10,
        show: bool = False,
    ) -> None:
        # Real-time dashboard: English labels, journal-style aesthetics
        t = result.t
        soc = result.soc * 100.0
        temp_c = result.y[2, :] - 273.15
        soh = result.y[3, :] * 100.0
        v_term = result.voltages["V_term"]
        v_ocv = result.voltages["V_ocv"]
        current = result.voltages["I"]
        p_total = result.powers["P_total"]
        p_screen = result.powers["P_screen"]
        p_cpu = result.powers["P_cpu"]
        p_wifi = result.powers["P_wifi"]
        p_gps = result.powers["P_gps"]
        p_heat = result.powers["P_heat"]

        n = len(t) if max_points is None else min(len(t), max_points)
        step = max(1, int(len(t) / n))

        fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8), sharex=True)
        fig.suptitle(f"Real-Time Battery Telemetry: {result.scenario.name}", fontsize=12)

        ax_soc, ax_temp, ax_soh, ax_volt, ax_power, ax_curr = axes.flatten()

        ax_soc.set_title("State of Charge (SOC)")
        ax_soc.set_ylabel("SOC (%)")
        ax_soc.set_xlabel("Time (h)")
        ax_soc.set_ylim(0, 105)
        soc_line, = ax_soc.plot([], [], linewidth=2.0, label="SOC")
        ax_soc.legend()

        ax_temp.set_title("Internal Temperature")
        ax_temp.set_ylabel("Temperature (C)")
        ax_temp.set_xlabel("Time (h)")
        temp_line, = ax_temp.plot([], [], linewidth=2.0, label="Temp")
        ax_temp.legend()

        ax_soh.set_title("State of Health (SOH)")
        ax_soh.set_ylabel("SOH (%)")
        ax_soh.set_xlabel("Time (h)")
        ax_soh.set_ylim(0, 105)
        soh_line, = ax_soh.plot([], [], linewidth=2.0, label="SOH")
        ax_soh.legend()

        ax_volt.set_title("Voltage Dynamics")
        ax_volt.set_ylabel("Voltage (V)")
        ax_volt.set_xlabel("Time (h)")
        vterm_line, = ax_volt.plot([], [], linewidth=2.0, label="Terminal")
        vocv_line, = ax_volt.plot([], [], linestyle="--", linewidth=1.8, label="OCV")
        ax_volt.legend()

        ax_power.set_title("Power Decomposition")
        ax_power.set_ylabel("Power (W)")
        ax_power.set_xlabel("Time (h)")
        ptotal_line, = ax_power.plot([], [], linewidth=2.0, label="Total")
        pscreen_line, = ax_power.plot([], [], linewidth=1.6, label="Screen")
        pcpu_line, = ax_power.plot([], [], linewidth=1.6, label="CPU")
        pwifi_line, = ax_power.plot([], [], linewidth=1.6, label="WiFi")
        pgps_line, = ax_power.plot([], [], linewidth=1.6, label="GPS")
        pheat_line, = ax_power.plot([], [], linewidth=1.6, label="Heat")
        ax_power.legend(ncol=2, fontsize=8)

        ax_curr.set_title("Current Draw")
        ax_curr.set_ylabel("Current (A)")
        ax_curr.set_xlabel("Time (h)")
        curr_line, = ax_curr.plot([], [], linewidth=2.0, label="Current")
        ax_curr.legend()

        plt.tight_layout()

        frame_idx = 0
        for idx in range(1, len(t), step):
            t_hr = t[:idx] / 3600.0
            soc_line.set_data(t_hr, soc[:idx])
            temp_line.set_data(t_hr, temp_c[:idx])
            soh_line.set_data(t_hr, soh[:idx])
            vterm_line.set_data(t_hr, v_term[:idx])
            vocv_line.set_data(t_hr, v_ocv[:idx])
            ptotal_line.set_data(t_hr, p_total[:idx])
            pscreen_line.set_data(t_hr, p_screen[:idx])
            pcpu_line.set_data(t_hr, p_cpu[:idx])
            pwifi_line.set_data(t_hr, p_wifi[:idx])
            pgps_line.set_data(t_hr, p_gps[:idx])
            pheat_line.set_data(t_hr, p_heat[:idx])
            curr_line.set_data(t_hr, current[:idx])

            for ax in axes.flatten():
                ax.relim()
                ax.autoscale_view(scalex=True, scaley=True)

            if save_dir is not None and (frame_idx % max(frame_stride, 1) == 0):
                save_dir.mkdir(parents=True, exist_ok=True)
                frame_path = save_dir / f"{result.scenario.name.replace(' ', '_')}_frame_{frame_idx:04d}.png"
                fig.savefig(frame_path, dpi=200)
            frame_idx += 1

            if show:
                plt.pause(max(interval_ms, 1) / 1000.0)

        if save_dir is not None:
            final_path = save_dir / f"{result.scenario.name.replace(' ', '_')}_final.png"
            fig.savefig(final_path, dpi=300)
        if not show:
            plt.close(fig)

    def plot_phase_portrait(self, results: List[SimulationResult]) -> None:
        plt.figure(figsize=(7.0, 4.2))
        for res in results:
            soc_pct = res.soc * 100.0
            temp_c = res.y[2, :] - 273.15
            plt.plot(soc_pct, temp_c, label=res.scenario.name, linewidth=2.0)
        plt.gca().invert_xaxis()
        plt.xlabel("State of Charge (%)")
        plt.ylabel("Temperature (C)")
        plt.title("Death Spiral: SOC vs Temperature")
        plt.legend()
        plt.tight_layout()

    def plot_voltage_sag(self, result: SimulationResult, cutoff_v: float) -> None:
        plt.figure(figsize=(7.0, 4.2))
        t_hr = result.t / 3600.0
        plt.plot(t_hr, result.voltages["V_term"], label="Terminal Voltage", linewidth=2.0)
        plt.plot(t_hr, result.voltages["V_ocv"], label="OCV", linestyle="--", linewidth=1.8)
        plt.axhline(cutoff_v, color="red", linestyle=":", label="Cutoff Voltage")
        plt.xlabel("Time (h)")
        plt.ylabel("Voltage (V)")
        plt.title(f"Voltage Sag and Cutoff: {result.scenario.name}")
        # Mark cutoff time
        cutoff_t_hr = result.time_to_empty_s / 3600.0
        plt.axvline(cutoff_t_hr, color="red", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

    def plot_voltage_validation(
        self,
        time_s: np.ndarray,
        v_pred: np.ndarray,
        v_meas: np.ndarray,
        title: str,
    ) -> None:
        plt.figure(figsize=(7.2, 4.2))
        t_hr = time_s / 3600.0
        plt.plot(t_hr, v_meas, label="Measured", linewidth=1.6, alpha=0.85)
        plt.plot(t_hr, v_pred, label="Model", linewidth=1.4, linestyle="--")
        plt.xlabel("Time (h)")
        plt.ylabel("Voltage (V)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

    def plot_ocv_validation(
        self,
        soc: np.ndarray,
        v_pred: np.ndarray,
        v_meas: np.ndarray,
        title: str,
    ) -> None:
        plt.figure(figsize=(7.2, 4.2))
        plt.scatter(soc, v_meas, label="Measured OCV", s=16, alpha=0.8)
        order = np.argsort(soc)
        plt.plot(soc[order], v_pred[order], label="Model OCV", linewidth=1.8)
        if soc.size > 0:
            idx = np.unique(np.linspace(0, soc.size - 1, 6, dtype=int))
            for i in idx:
                plt.annotate(
                    f"({soc[i]:.2f}, {v_meas[i]:.3f})",
                    (soc[i], v_meas[i]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                )
        plt.xlabel("SOC (-)")
        plt.ylabel("Voltage (V)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

    def plot_power_decomposition(self, result: SimulationResult) -> None:
        plt.figure(figsize=(7.5, 4.5))
        t_hr = result.t / 3600.0
        p_screen = result.powers["P_screen"]
        p_cpu = result.powers["P_cpu"]
        p_net = result.powers["P_net"]
        p_gps = result.powers["P_gps"]
        p_heat = result.powers["P_heat"]

        plt.stackplot(
            t_hr,
            p_screen,
            p_cpu,
            p_net,
            p_gps,
            p_heat,
            labels=["Screen", "CPU", "Network", "GPS", "Heat"],
            alpha=0.85,
        )
        plt.xlabel("Time (h)")
        plt.ylabel("Power (W)")
        plt.title(f"Power Decomposition: {result.scenario.name}")
        plt.legend(loc="upper right")
        plt.tight_layout()

    def plot_day_in_life(self, result: SimulationResult) -> None:
        plt.figure(figsize=(7.4, 4.4))
        t_hr = result.t / 3600.0
        soc_pct = result.soc * 100.0
        temp_c = result.y[2, :] - 273.15
        ax = plt.gca()
        ax.plot(t_hr, soc_pct, color="#1f77b4", linewidth=1.6, label="SOC")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("SOC (%)")
        ax.set_ylim(0, 105)
        ax2 = ax.twinx()
        ax2.plot(t_hr, temp_c, color="#e45756", linewidth=1.4, label="Temp")
        ax2.set_ylabel("Temperature (C)")
        ax.set_title(f"Day in the Life: SOC & Temperature ({result.scenario.name})")
        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper right")
        plt.tight_layout()

    def plot_climate_stress(
        self,
        results: List[SimulationResult],
        temps_c: List[float],
        base_name: str,
    ) -> None:
        plt.figure(figsize=(7.6, 7.2))
        colors = sns.color_palette("mako", n_colors=len(results))
        axes = [plt.subplot(3, 1, i + 1) for i in range(3)]
        for idx, res in enumerate(results):
            t_hr = res.t / 3600.0
            label = f"{temps_c[idx]:.0f}C"
            axes[0].plot(t_hr, res.soc * 100.0, color=colors[idx], linewidth=1.4, label=label)
            axes[1].plot(t_hr, res.voltages["V_term"], color=colors[idx], linewidth=1.4, label=label)
            axes[2].plot(t_hr, res.y[2, :] - 273.15, color=colors[idx], linewidth=1.4, label=label)
        axes[0].set_ylabel("SOC (%)")
        axes[1].set_ylabel("Voltage (V)")
        axes[2].set_ylabel("Temp (C)")
        axes[2].set_xlabel("Time (h)")
        axes[0].set_title(f"Climate Stress Test: {base_name}")
        for ax in axes:
            ax.legend(ncol=3, fontsize=8, loc="upper right")
        plt.tight_layout()

    def plot_sensitivity_heatmap(
        self,
        times: np.ndarray,
        brightness_grid: np.ndarray,
        temp_grid_c: np.ndarray,
    ) -> None:
        plt.figure(figsize=(7.0, 4.8))
        sns.heatmap(
            times,
            xticklabels=[f"{b:.1f}" for b in brightness_grid],
            yticklabels=[f"{t:.0f}" for t in temp_grid_c],
            cmap="viridis",
            cbar_kws={"label": "Time-to-Empty (h)"},
        )
        plt.xlabel("Screen Brightness (0-1)")
        plt.ylabel("Ambient Temperature (C)")
        plt.title("Sensitivity Heatmap: Brightness vs Ambient Temperature")
        plt.tight_layout()

    def plot_uncertainty_band(
        self,
        t_s: np.ndarray,
        soc_p5: np.ndarray,
        soc_p50: np.ndarray,
        soc_p95: np.ndarray,
        scenario_name: str,
    ) -> None:
        plt.figure(figsize=(7.2, 4.2))
        t_hr = t_s / 3600.0
        band_color = "#5a7d9a"
        median_color = "#0f4c5c"
        plt.fill_between(t_hr, soc_p5 * 100.0, soc_p95 * 100.0, color=band_color, alpha=0.18, label="5-95% band")
        plt.plot(t_hr, soc_p50 * 100.0, color=median_color, linewidth=1.1, label="Median SOC")
        plt.plot(t_hr, soc_p5 * 100.0, color=band_color, linewidth=0.8, alpha=0.7)
        plt.plot(t_hr, soc_p95 * 100.0, color=band_color, linewidth=0.8, alpha=0.7)
        if t_hr.size > 0:
            idx = np.unique(np.linspace(0, t_hr.size - 1, 5, dtype=int))
            for i in idx:
                plt.annotate(
                    f"{t_hr[i]:.1f}h, {soc_p50[i]*100.0:.1f}%",
                    (t_hr[i], soc_p50[i] * 100.0),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                )
        plt.xlabel("Time (h)")
        plt.ylabel("SOC (%)")
        plt.title(f"Uncertainty Band (5-95%): {scenario_name}")
        plt.legend()
        plt.tight_layout()

    def plot_uncertainty_histogram(self, tte_s: np.ndarray, scenario_name: str) -> None:
        plt.figure(figsize=(6.5, 4.0))
        tte_hr = tte_s / 3600.0
        plt.hist(tte_hr, bins=24, color="tab:green", alpha=0.8)
        plt.xlabel("Time-to-Empty (h)")
        plt.ylabel("Count")
        plt.title(f"Uncertainty: Time-to-Empty Distribution ({scenario_name})")
        plt.tight_layout()


def constant_profile(value: float) -> Callable[[float], float]:
    return lambda t: float(value)


def piecewise_profile(segments: List[Tuple[float, float | None, float]]) -> Callable[[float], float]:
    sorted_segments = sorted(segments, key=lambda s: s[0])

    def profile(t: float) -> float:
        if t < sorted_segments[0][0]:
            return float(sorted_segments[0][2])
        for start, end, value in sorted_segments:
            if end is None and t >= start:
                return float(value)
            if end is not None and start <= t < end:
                return float(value)
        return float(sorted_segments[-1][2])

    return profile


def build_profile(cfg: Dict[str, Any]) -> Callable[[float], float]:
    profile_type = cfg.get("type", "constant")
    if profile_type == "constant":
        return constant_profile(float(cfg["value"]))
    if profile_type == "piecewise":
        segments = [(float(s["start"]), s.get("end"), float(s["value"])) for s in cfg["segments"]]
        normalized = []
        for start, end, value in segments:
            end_val = float(end) if end is not None else None
            normalized.append((start, end_val, value))
        return piecewise_profile(normalized)
    raise ValueError(f"Unsupported profile type: {profile_type}")


def build_grid(cfg: Any) -> np.ndarray:
    if isinstance(cfg, list):
        return np.array(cfg, dtype=float)
    if isinstance(cfg, dict) and cfg.get("type") == "linspace":
        return np.linspace(float(cfg["start"]), float(cfg["end"]), int(cfg["num"]))
    raise ValueError("Grid config must be a list or a linspace dict.")


def load_current_profile(path: Path) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)

    def pick(names: List[str]) -> np.ndarray:
        for name in names:
            if name in data.dtype.names:
                return data[name].astype(float)
        return np.array([])

    time_s = pick(["time_s", "t_s", "time", "Time"])
    current_a = pick(["current_a", "current", "I", "Current"])
    voltage_v = pick(["voltage_v", "v", "V", "Voltage"])
    temp_c = pick(["temp_c", "temperature_c", "temp", "Temperature"])
    soc = pick(["soc", "soc_frac", "soc_fraction", "soc_pct", "soc_percent", "SoC"])

    if time_s.size == 0 or current_a.size == 0:
        raise ValueError("CSV must include time_s and current_a columns.")

    return {
        "time_s": time_s,
        "current_a": current_a,
        "voltage_v": voltage_v,
        "temp_c": temp_c,
        "soc": soc,
    }


def load_ocv_curve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)

    def pick(names: List[str]) -> np.ndarray:
        for name in names:
            if name in data.dtype.names:
                return data[name].astype(float)
        return np.array([])

    soc = pick(["soc", "SOC", "z", "soc_frac", "soc_fraction"])
    voltage_v = pick(["voltage_v", "ocv", "voltage", "V"])

    if soc.size == 0 or voltage_v.size == 0:
        raise ValueError("OCV CSV must include soc and voltage_v columns.")

    return soc, voltage_v


def sample_params(
    base: BatteryParams,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> BatteryParams:
    perturb = cfg.get("perturb", {})
    payload = dict(base.__dict__)
    for name, spec in perturb.items():
        if name not in payload:
            continue
        value = payload[name]
        if not isinstance(value, (int, float)):
            continue
        std_frac = float(spec.get("std_frac", 0.0))
        if std_frac <= 0.0:
            continue
        sample = rng.normal(loc=float(value), scale=abs(float(value)) * std_frac)
        if "min" in spec:
            sample = max(sample, float(spec["min"]))
        if "max" in spec:
            sample = min(sample, float(spec["max"]))
        payload[name] = sample
    return BatteryParams(**payload)


def validate_current_profile(
    params: BatteryParams,
    profile_path: Path,
    use_temp_from_csv: bool = True,
    calibrate: bool = False,
    current_scale_bounds: Tuple[float, float] = (0.5, 1.5),
    use_soc_from_csv: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    data = load_current_profile(profile_path)
    time_s = data["time_s"]
    current_raw = data["current_a"]
    voltage_meas = data["voltage_v"]
    temp_c = data["temp_c"]
    soc_meas = data.get("soc", np.array([]))

    if voltage_meas.size == 0:
        raise ValueError("CSV must include voltage_v for validation.")

    current_model = current_raw.copy()
    if np.nanmedian(current_model) < 0:
        current_model = -current_model

    physics = BatteryPhysics(params)

    def predict_voltage(current_series: np.ndarray, r_scale: float, ocv_scale: float) -> np.ndarray:
        soc = np.zeros_like(time_s, dtype=float)
        v_pred = np.zeros_like(time_s, dtype=float)
        soc[0] = params.SOC_init
        use_soc = bool(use_soc_from_csv and soc_meas.size == time_s.size)
        if use_soc:
            soc_series = soc_meas.astype(float).copy()
            if np.nanmax(soc_series) > 1.5:
                soc_series = soc_series / 100.0
            soc_series = np.clip(soc_series, 0.0, 1.0)
            soc[0] = soc_series[0]

        for i in range(1, len(time_s)):
            dt = max(time_s[i] - time_s[i - 1], 0.0)
            if use_temp_from_csv and temp_c.size == time_s.size:
                temp_k = float(temp_c[i]) + 273.15
            else:
                temp_k = params.T_init
            cap_ah = params.Q_design_ah * max(params.SOH_init, 0.05) * physics.capacity_temp_factor(temp_k)
            if use_soc:
                soc[i] = soc_series[i]
            else:
                soc[i] = soc[i - 1] - current_series[i] * dt / 3600.0 / max(cap_ah, 1e-9)
                soc[i] = float(np.clip(soc[i], 0.0, 1.0))
            r_int = physics.get_r_int(soc[i], temp_k, params.SOH_init) * r_scale
            ocv = physics.get_ocv(soc[i]) * ocv_scale
            v_pred[i] = ocv - current_series[i] * r_int

        if len(time_s) > 0:
            temp_k0 = float(temp_c[0]) + 273.15 if (use_temp_from_csv and temp_c.size == time_s.size) else params.T_init
            r_int0 = physics.get_r_int(soc[0], temp_k0, params.SOH_init) * r_scale
            ocv0 = physics.get_ocv(soc[0]) * ocv_scale
            v_pred[0] = ocv0 - current_series[0] * r_int0
        return v_pred

    meta: Dict[str, float] = {
        "current_scale": 1.0,
        "voltage_offset": 0.0,
        "r_scale": 1.0,
        "ocv_scale": 1.0,
    }

    if calibrate:
        def rmse(params_vec: np.ndarray) -> float:
            scale, r_scale, ocv_scale, offset = params_vec
            v_pred = predict_voltage(current_model * scale, r_scale, ocv_scale)
            v_adj = v_pred + offset
            return float(np.sqrt(np.mean((v_adj - voltage_meas) ** 2)))

        bounds = [
            (current_scale_bounds[0], current_scale_bounds[1]),
            (0.7, 1.5),
            (0.95, 1.05),
            (-0.2, 0.2),
        ]
        x0 = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
        res = minimize(rmse, x0=x0, bounds=bounds, method="L-BFGS-B")
        best_scale, best_r, best_ocv, best_offset = res.x
        v_pred = predict_voltage(current_model * best_scale, best_r, best_ocv) + best_offset
        meta["current_scale"] = float(best_scale)
        meta["r_scale"] = float(best_r)
        meta["ocv_scale"] = float(best_ocv)
        meta["voltage_offset"] = float(best_offset)
    else:
        v_pred = predict_voltage(current_model, 1.0, 1.0)

    return time_s, v_pred, voltage_meas, meta


def validate_ocv_curve(
    params: BatteryParams,
    ocv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    soc, v_meas = load_ocv_curve(ocv_path)
    physics = BatteryPhysics(params)
    v_pred = np.array([physics.get_ocv(float(z), params.ocv_temp_ref) for z in soc], dtype=float)
    return soc, v_pred, v_meas


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping at the top level.")
    return data


def build_scenario(cfg: Dict[str, Any]) -> Scenario:
    return Scenario(
        name=str(cfg["name"]),
        duration_s=float(cfg["duration_s"]),
        dt=float(cfg["dt"]),
        env_temp_k=float(cfg["env_temp_k"]),
        brightness=build_profile(cfg["brightness"]),
        apl=build_profile(cfg["apl"]),
        cpu_util=build_profile(cfg["cpu_util"]),
        wifi_state=build_profile(cfg["wifi_state"]),
        gps_state=build_profile(cfg["gps_state"]),
        charger_power=build_profile(cfg["charger_power"]),
        eta_radiation=float(cfg["eta_radiation"]),
    )


def run_scenarios(config_path: str = "config.yaml") -> None:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parent / config_file

    config = load_yaml_config(config_file)
    params = BatteryParams.from_dict(config["battery_params"])
    system = PowerSystem(params)
    viz = Visualizer()

    scenarios_cfg = config["scenarios"]
    scenarios = [build_scenario(cfg) for cfg in scenarios_cfg]
    solver_cfg = config.get("solver", {})
    results = [system.solve(sc, solver_cfg) for sc in scenarios]

    viz_cfg = config.get("visualization", {})
    show_plots = bool(viz_cfg.get("show", False))
    plt.ioff()

    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("directory", "outputs"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    viz.plot_phase_portrait(results)
    plt.savefig(figures_dir / "phase_portrait.png", dpi=300)
    if not show_plots:
        plt.close()

    for res in results:
        viz.plot_voltage_sag(res, params.V_cutoff)
        plt.savefig(figures_dir / f"voltage_sag_{res.scenario.name.replace(' ', '_')}.png", dpi=300)
        if not show_plots:
            plt.close()
        viz.plot_power_decomposition(res)
        plt.savefig(figures_dir / f"power_decomposition_{res.scenario.name.replace(' ', '_')}.png", dpi=300)
        if not show_plots:
            plt.close()
        if res.scenario.name.lower() == "day in the life":
            viz.plot_day_in_life(res)
            plt.savefig(figures_dir / "day_in_life_soc_temp.png", dpi=300)
            if not show_plots:
                plt.close()

    if viz_cfg.get("realtime", False):
        interval_ms = int(viz_cfg.get("interval_ms", 80))
        max_points = viz_cfg.get("max_points")
        max_points = int(max_points) if max_points is not None else None
        frame_stride = int(viz_cfg.get("frame_stride", 10))
        save_frames = bool(viz_cfg.get("save_frames", True))
        frames_dir = output_dir / "frames"
        for res in results:
            viz.realtime_dashboard(
                res,
                interval_ms=interval_ms,
                max_points=max_points,
                save_dir=frames_dir if save_frames else None,
                frame_stride=frame_stride,
                show=show_plots,
            )

    climate_cfg = config.get("climate_stress", {})
    if climate_cfg.get("enabled", False):
        base_name = climate_cfg.get("base_scenario", "Gaming")
        base = scenarios[0]
        for sc in scenarios:
            if sc.name == base_name:
                base = sc
                break
        temps_c = [float(t) for t in climate_cfg.get("temps_c", [0, 25, 40])]
        climate_results: List[SimulationResult] = []
        for temp_c in temps_c:
            scenario = Scenario(
                name=f"{base.name} {temp_c:.0f}C",
                duration_s=base.duration_s,
                dt=base.dt,
                env_temp_k=temp_c + 273.15,
                brightness=base.brightness,
                apl=base.apl,
                cpu_util=base.cpu_util,
                wifi_state=base.wifi_state,
                gps_state=base.gps_state,
                charger_power=base.charger_power,
                eta_radiation=base.eta_radiation,
            )
            climate_results.append(system.solve(scenario, solver_cfg))
        viz.plot_climate_stress(climate_results, temps_c, base.name)
        plt.savefig(figures_dir / f"climate_stress_{base.name.replace(' ', '_')}.png", dpi=300)
        if not show_plots:
            plt.close()
        if output_cfg.get("enabled", False):
            summary_rows = []
            for res, temp_c in zip(climate_results, temps_c):
                summary_rows.append(
                    {
                        "scenario": res.scenario.name,
                        "temp_c": temp_c,
                        "t_end_s": float(res.t[-1]),
                        "tte_h": float(res.time_to_empty_s / 3600.0),
                        "soc_end": float(res.soc[-1]),
                        "v_end": float(res.voltages["V_term"][-1]),
                        "temp_max_c": float(np.max(res.y[2, :] - 273.15)),
                    }
                )
            summary_path = output_dir / f"climate_stress_summary_{base.name.replace(' ', '_')}.csv"
            np.savetxt(
                summary_path,
                np.array(
                    [
                        [
                            row["temp_c"],
                            row["t_end_s"],
                            row["tte_h"],
                            row["soc_end"],
                            row["v_end"],
                            row["temp_max_c"],
                        ]
                        for row in summary_rows
                    ]
                ),
                delimiter=",",
                header="temp_c,t_end_s,tte_h,soc_end,v_end,temp_max_c",
                comments="",
            )

    sensitivity = config.get("sensitivity", {})
    if sensitivity:
        brightness_grid = build_grid(sensitivity["brightness_grid"])
        temp_grid_c = build_grid(sensitivity["temp_grid_c"])
        time_grid = np.zeros((len(temp_grid_c), len(brightness_grid)))

        base_name = sensitivity.get("base_scenario")
        base = scenarios[0]
        if base_name:
            for sc in scenarios:
                if sc.name == base_name:
                    base = sc
                    break

        duration_s = float(sensitivity.get("duration_s", 8 * 3600.0))
        dt = float(sensitivity.get("dt", 20.0))

        for i, temp_c in enumerate(temp_grid_c):
            for j, b in enumerate(brightness_grid):
                scenario = Scenario(
                    name=f"Sens_{temp_c:.0f}C_{b:.2f}",
                    duration_s=duration_s,
                    dt=dt,
                    env_temp_k=temp_c + 273.15,
                    brightness=constant_profile(float(b)),
                    apl=base.apl,
                    cpu_util=base.cpu_util,
                    wifi_state=base.wifi_state,
                    gps_state=base.gps_state,
                    charger_power=base.charger_power,
                    eta_radiation=base.eta_radiation,
                )
                res = system.solve(scenario, solver_cfg)
                time_grid[i, j] = res.time_to_empty_s / 3600.0

        viz.plot_sensitivity_heatmap(time_grid, brightness_grid, temp_grid_c)

    uncertainty = config.get("uncertainty", {})
    if uncertainty.get("enabled", False):
        base_name = uncertainty.get("base_scenario")
        base = scenarios[0]
        if base_name:
            for sc in scenarios:
                if sc.name == base_name:
                    base = sc
                    break

        n_samples = int(uncertainty.get("n_samples", 200))
        seed = int(uncertainty.get("seed", 42))
        rng = np.random.default_rng(seed)

        time_grid = np.arange(0.0, base.duration_s + base.dt, base.dt)
        soc_samples = np.zeros((n_samples, len(time_grid)))
        tte_samples = np.zeros(n_samples)

        for i in range(n_samples):
            params_i = sample_params(params, uncertainty, rng)
            system_i = PowerSystem(params_i)
            res = system_i.solve(base, solver_cfg)
            soc_samples[i, :] = np.interp(
                time_grid,
                res.t,
                res.soc,
                left=res.soc[0],
                right=0.0,
            )
            tte_samples[i] = res.time_to_empty_s

        soc_p5 = np.percentile(soc_samples, 5.0, axis=0)
        soc_p50 = np.percentile(soc_samples, 50.0, axis=0)
        soc_p95 = np.percentile(soc_samples, 95.0, axis=0)
        viz.plot_uncertainty_band(time_grid, soc_p5, soc_p50, soc_p95, base.name)
        plt.savefig(figures_dir / f"uncertainty_band_{base.name.replace(' ', '_')}.png", dpi=300)
        if not show_plots:
            plt.close()
        viz.plot_uncertainty_histogram(tte_samples, base.name)
        plt.savefig(figures_dir / f"uncertainty_hist_{base.name.replace(' ', '_')}.png", dpi=300)
        if not show_plots:
            plt.close()

        if output_cfg.get("enabled", False):
            mc_dir = output_dir / "monte_carlo"
            mc_dir.mkdir(parents=True, exist_ok=True)
            pct_csv = mc_dir / f"uncertainty_percentiles_{base.name.replace(' ', '_')}.csv"
            pct_data = np.column_stack(
                [
                    time_grid,
                    soc_p5,
                    soc_p50,
                    soc_p95,
                ]
            )
            np.savetxt(
                pct_csv,
                pct_data,
                delimiter=",",
                header="t_s,soc_p5,soc_p50,soc_p95",
                comments="",
            )
            tte_csv = mc_dir / f"uncertainty_tte_{base.name.replace(' ', '_')}.csv"
            np.savetxt(
                tte_csv,
                tte_samples,
                delimiter=",",
                header="time_to_empty_s",
                comments="",
            )

    validation = config.get("validation", {})
    if validation.get("enabled", False):
        data_paths = config.get("data_paths", {})
        mode = str(validation.get("mode", "current_profile")).lower()
        if mode == "ocv_curve":
            ocv_path = validation.get("ocv_csv") or data_paths.get("calce_ocv_csv")
            if ocv_path:
                ocv_path = Path(ocv_path)
                if not ocv_path.is_absolute():
                    ocv_path = config_file.parent / ocv_path
                soc, v_pred, v_meas = validate_ocv_curve(params, ocv_path)
                title = f"OCV Validation: {ocv_path.name}"
                viz.plot_ocv_validation(soc, v_pred, v_meas, title)
                plt.savefig(figures_dir / f"validation_ocv_{ocv_path.stem}.png", dpi=300)
                if not show_plots:
                    plt.close()
        else:
            profile_path = validation.get("current_profile_csv") or data_paths.get("nasa_random_csv")
            if profile_path:
                profile_path = Path(profile_path)
                if not profile_path.is_absolute():
                    profile_path = config_file.parent / profile_path
                calibrate = bool(validation.get("calibrate", False))
                scale_bounds = validation.get("current_scale_bounds", [0.5, 1.5])
                scale_bounds = (float(scale_bounds[0]), float(scale_bounds[1]))
                use_soc_from_csv = bool(validation.get("use_soc_from_csv", False))
                time_s, v_pred, v_meas, meta = validate_current_profile(
                    params,
                    profile_path,
                    use_temp_from_csv=bool(validation.get("use_temp_from_csv", True)),
                    calibrate=calibrate,
                    current_scale_bounds=scale_bounds,
                    use_soc_from_csv=use_soc_from_csv,
                )
                title = f"Validation: {profile_path.name}"
                if calibrate:
                    title += (
                        f" (scale={meta['current_scale']:.3f}, r={meta['r_scale']:.3f}, "
                        f"ocv={meta['ocv_scale']:.3f}, offset={meta['voltage_offset']:.3f}V)"
                    )
                viz.plot_voltage_validation(time_s, v_pred, v_meas, title)
                plt.savefig(figures_dir / f"validation_{profile_path.stem}.png", dpi=300)
                if not show_plots:
                    plt.close()

    if output_cfg.get("enabled", False):
        output_format = str(output_cfg.get("format", "csv")).lower()

        for res in results:
            t = res.t
            data = {
                "t_s": t,
                "soc": res.soc,
                "temp_k": res.y[2, :],
                "soh": res.y[3, :],
                "v_term": res.voltages["V_term"],
                "v_ocv": res.voltages["V_ocv"],
                "current_a": res.voltages["I"],
                "p_screen": res.powers["P_screen"],
                "p_cpu": res.powers["P_cpu"],
                "p_wifi": res.powers["P_wifi"],
                "p_gps": res.powers["P_gps"],
                "p_net": res.powers["P_net"],
                "p_load": res.powers["P_load"],
                "p_charge": res.powers["P_charge"],
                "p_total": res.powers["P_total"],
                "p_heat": res.powers["P_heat"],
            }

            if output_format == "csv":
                csv_path = output_dir / f"{res.scenario.name.replace(' ', '_')}.csv"
                header = ",".join(data.keys())
                rows = np.column_stack([data[key] for key in data])
                np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")
            elif output_format == "json":
                json_path = output_dir / f"{res.scenario.name.replace(' ', '_')}.json"
                payload = {key: data[key].tolist() for key in data}
                meta = {"scenario": res.scenario.name, "time_to_empty_s": res.time_to_empty_s}
                with json_path.open("w", encoding="utf-8") as handle:
                    json.dump({"meta": meta, "data": payload}, handle, indent=2)
            else:
                raise ValueError("output.format must be 'csv' or 'json'.")

    if show_plots:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACM battery simulation runner.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_scenarios(args.config)
