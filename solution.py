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

    def get_ocv(self, soc: float) -> float:
        model = getattr(self.params, "ocv_model", "polynomial").lower()
        if model == "combined":
            k0, k1, k2, k3, k4 = self.params.ocv_coeffs
            s = np.clip(soc, 0.001, 0.999)
            return float(k0 - k1 / s - k2 * s + k3 * np.log(s) + k4 * np.log(1.0 - s))
        soc = np.clip(soc, 0.0, 1.0)
        return float(np.polyval(self.params.ocv_coeffs, soc))

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
        y1, y2, temp_k, soh = y
        soc = self._soc_from_state(y1, temp_k, soh)
        ocv = self.physics.get_ocv(soc)
        r_int = self.physics.get_r_int(soc, temp_k, soh)
        powers = self._component_powers(t, temp_k, scenario)
        current = self._solve_current(ocv, r_int, powers["P_total"])
        v_term = ocv - current * r_int
        return v_term, ocv, current

    def derivative(self, t: float, y: np.ndarray, scenario: Scenario) -> np.ndarray:
        y1, y2, temp_k, soh = y
        soh_c = np.clip(soh, 0.05, 1.0)
        soc_c = self._soc_from_state(y1, temp_k, soh_c)

        powers = self._component_powers(t, temp_k, scenario)
        phi_soc = self.physics.lpm_factor(soc_c)
        powers["P_total"] *= phi_soc
        powers["P_screen"] *= phi_soc
        powers["P_cpu"] *= phi_soc
        powers["P_net"] *= phi_soc

        ocv = self.physics.get_ocv(soc_c)
        r_int = self.physics.get_r_int(soc_c, temp_k, soh_c)
        current = self._solve_current(ocv, r_int, powers["P_total"])
        v_term = ocv - current * r_int

        # KiBaM dynamics (optional)
        if self.params.kibam_enabled:
            h1 = y1 / max(self.params.kibam_c, 1e-6)
            h2 = y2 / max(1.0 - self.params.kibam_c, 1e-6)
            dy1 = -current + self.params.kibam_k * (h2 - h1)
            dy2 = -self.params.kibam_k * (h2 - h1)
        else:
            dy1 = -current
            dy2 = 0.0

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
            return np.array([0.0, 0.0, 0.0, 0.0])

        return np.array([dy1, dy2, d_temp, d_soh])

    def solve(self, scenario: Scenario) -> SimulationResult:
        c_total = self._capacity_total(self.params.T_init, self.params.SOH_init)
        if self.params.kibam_enabled:
            y1_0 = self.params.SOC_init * (self.params.kibam_c * c_total)
            y2_0 = self.params.SOC_init * ((1.0 - self.params.kibam_c) * c_total)
        else:
            y1_0 = self.params.SOC_init * c_total
            y2_0 = 0.0
        y0 = np.array([y1_0, y2_0, self.params.T_init, self.params.SOH_init], dtype=float)
        t_eval = np.arange(0.0, scenario.duration_s + scenario.dt, scenario.dt)

        def cutoff_event(t: float, y: np.ndarray) -> float:
            v_term, _, _ = self._voltage(t, y, scenario)
            return v_term - self.params.V_cutoff

        cutoff_event.terminal = True
        cutoff_event.direction = -1.0

        sol = solve_ivp(
            fun=lambda t, y: self.derivative(t, y, scenario),
            t_span=(0.0, scenario.duration_s),
            y0=y0,
            t_eval=t_eval,
            events=cutoff_event,
            rtol=1e-6,
            atol=1e-8,
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
            y1_i, y2_i, temp_i, soh_i = y[:, i]
            soc_i = self._soc_from_state(y1_i, temp_i, soh_i)
            powers = self._component_powers(t[i], temp_i, scenario)
            phi_soc = self.physics.lpm_factor(soc_i)
            powers["P_total"] *= phi_soc
            powers["P_screen"] *= phi_soc
            powers["P_cpu"] *= phi_soc
            powers["P_net"] *= phi_soc
            ocv = self.physics.get_ocv(soc_i)
            r_int = self.physics.get_r_int(soc_i, temp_i, soh_i)
            i_cur = self._solve_current(ocv, r_int, powers["P_total"])
            v_t = ocv - i_cur * r_int
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

        voltages = {"V_term": v_term, "V_ocv": v_ocv, "I": current}
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

            plt.pause(max(interval_ms, 1) / 1000.0)

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
        plt.fill_between(t_hr, soc_p5 * 100.0, soc_p95 * 100.0, color="tab:blue", alpha=0.25)
        plt.plot(t_hr, soc_p50 * 100.0, color="tab:blue", linewidth=2.0, label="Median SOC")
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
    results = [system.solve(sc) for sc in scenarios]

    viz.plot_phase_portrait(results)
    for res in results:
        viz.plot_voltage_sag(res, params.V_cutoff)
        viz.plot_power_decomposition(res)

    viz_cfg = config.get("visualization", {})
    if viz_cfg.get("realtime", False):
        interval_ms = int(viz_cfg.get("interval_ms", 80))
        max_points = viz_cfg.get("max_points")
        max_points = int(max_points) if max_points is not None else None
        for res in results:
            viz.realtime_dashboard(res, interval_ms=interval_ms, max_points=max_points)

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
                    eta_logic=base.eta_logic,
                )
                res = system.solve(scenario)
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
            res = system_i.solve(base)
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
        viz.plot_uncertainty_histogram(tte_samples, base.name)

    output_cfg = config.get("output", {})
    if output_cfg.get("enabled", False):
        output_dir = Path(output_cfg.get("directory", "outputs"))
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
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
