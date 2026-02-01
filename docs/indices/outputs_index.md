# Outputs Index (Figures & Tables)

This index lists **all current figures and tables**, with meaning and experimental context.

## 1) Main Outputs (`outputs/`)

### 1.1 Scenario Tables (CSV) — Main Run
- `outputs/Video_Streaming.csv` — Scenario time series (SOC, temp, voltage, current, power) — *Scenario Analysis / Video Streaming*.
- `outputs/Gaming.csv` — Scenario time series — *Scenario Analysis / Gaming*.
- `outputs/Winter_Usage.csv` — Scenario time series — *Scenario Analysis / Winter Usage*.
- `outputs/Day_in_the_Life.csv` — 12h mixed-load time series — *New Scenario / Day-in-the-Life*.
- `outputs/climate_stress_summary_Gaming.csv` — Climate matrix summary (0/25/40C) — *Climate Stress Test*.

### 1.2 Validation + Parameter Tables/Plots
- `outputs/parameter_validation_ocv_phone.png` — OCV curve fit validation — *Parameter Estimation*.
- `outputs/figures/validation_phone_validation_bcm_s9_combined.png` — Model vs measured voltage — *Phone-level Validation*.

### 1.3 Scenario Figures (PNG) — Main Run
- `outputs/figures/voltage_sag_Video_Streaming.png` — Terminal/OCV voltage & cutoff — *Scenario Analysis / Video*.
- `outputs/figures/voltage_sag_Gaming.png` — Voltage sag & cutoff — *Scenario Analysis / Gaming*.
- `outputs/figures/voltage_sag_Winter_Usage.png` — Voltage sag & cutoff — *Scenario Analysis / Winter*.
- `outputs/figures/voltage_sag_Day_in_the_Life.png` — Voltage sag under mixed load — *Day-in-the-Life*.
- `outputs/figures/power_decomposition_Video_Streaming.png` — Component power breakdown — *Scenario Analysis / Video*.
- `outputs/figures/power_decomposition_Gaming.png` — Component power breakdown — *Scenario Analysis / Gaming*.
- `outputs/figures/power_decomposition_Winter_Usage.png` — Component power breakdown — *Scenario Analysis / Winter*.
- `outputs/figures/power_decomposition_Day_in_the_Life.png` — Component power breakdown — *Day-in-the-Life*.
- `outputs/figures/phase_portrait.png` — SOC vs temperature phase portrait — *Coupling Dynamics*.
- `outputs/figures/day_in_life_soc_temp.png` — SOC & temperature over 12h — *Day-in-the-Life*.
- `outputs/figures/climate_stress_Gaming.png` — SOC/Voltage/Temp overlays for 0/25/40C — *Climate Stress Test*.

### 1.4 Uncertainty Outputs (Monte Carlo)
- `outputs/figures/uncertainty_band_Video_Streaming.png` — SOC confidence band — *Uncertainty Quantification*.
- `outputs/figures/uncertainty_hist_Video_Streaming.png` — TTE distribution — *Uncertainty Quantification*.
- `outputs/monte_carlo/uncertainty_percentiles_Video_Streaming.csv` — SOC percentiles over time — *Uncertainty Quantification*.
- `outputs/monte_carlo/uncertainty_tte_Video_Streaming.csv` — TTE samples — *Uncertainty Quantification*.

## 2) SOH History Comparison (`outputs/soh_comparison/20260131_195801/`)

### 2.1 Summary + Conclusions
- `outputs/soh_comparison/20260131_195801/soh_comparison.csv` — Summary across SOH_init values — *Aging/History Study*.
- `outputs/soh_comparison/20260131_195801/SOH_COMPARISON_CONCLUSION.md` — Text conclusions — *Aging/History Study*.
- `outputs/soh_comparison/20260131_195801/SOH_COMPARISON_CONCLUSION.txt` — Text conclusions — *Aging/History Study*.

### 2.2 Comparison Figures
- `outputs/soh_comparison/20260131_195801/soh_voltage_time.png` — Voltage vs time by SOH — *Aging/History Study*.
- `outputs/soh_comparison/20260131_195801/soh_temp_time.png` — Temperature vs time by SOH — *Aging/History Study*.
- `outputs/soh_comparison/20260131_195801/soh_energy_bar.png` — Energy usage by SOH — *Aging/History Study*.

### 2.3 Per-SOH Scenarios (Gaming)
- `outputs/soh_comparison/20260131_195801/SOH_1.0/Gaming.csv` — Time series — *SOH=1.0*.
- `outputs/soh_comparison/20260131_195801/SOH_0.8/Gaming.csv` — Time series — *SOH=0.8*.
- `outputs/soh_comparison/20260131_195801/SOH_0.5/Gaming.csv` — Time series — *SOH=0.5*.
- `outputs/soh_comparison/20260131_195801/SOH_0.3/Gaming.csv` — Time series — *SOH=0.3*.
- `outputs/soh_comparison/20260131_195801/SOH_1.0/figures/phase_portrait.png` — Phase portrait — *SOH=1.0*.
- `outputs/soh_comparison/20260131_195801/SOH_1.0/figures/power_decomposition_Gaming.png` — Power breakdown — *SOH=1.0*.
- `outputs/soh_comparison/20260131_195801/SOH_1.0/figures/voltage_sag_Gaming.png` — Voltage sag — *SOH=1.0*.
- `outputs/soh_comparison/20260131_195801/SOH_0.8/figures/phase_portrait.png` — Phase portrait — *SOH=0.8*.
- `outputs/soh_comparison/20260131_195801/SOH_0.8/figures/power_decomposition_Gaming.png` — Power breakdown — *SOH=0.8*.
- `outputs/soh_comparison/20260131_195801/SOH_0.8/figures/voltage_sag_Gaming.png` — Voltage sag — *SOH=0.8*.
- `outputs/soh_comparison/20260131_195801/SOH_0.5/figures/phase_portrait.png` — Phase portrait — *SOH=0.5*.
- `outputs/soh_comparison/20260131_195801/SOH_0.5/figures/power_decomposition_Gaming.png` — Power breakdown — *SOH=0.5*.
- `outputs/soh_comparison/20260131_195801/SOH_0.5/figures/voltage_sag_Gaming.png` — Voltage sag — *SOH=0.5*.
- `outputs/soh_comparison/20260131_195801/SOH_0.3/figures/phase_portrait.png` — Phase portrait — *SOH=0.3*.
- `outputs/soh_comparison/20260131_195801/SOH_0.3/figures/power_decomposition_Gaming.png` — Power breakdown — *SOH=0.3*.
- `outputs/soh_comparison/20260131_195801/SOH_0.3/figures/voltage_sag_Gaming.png` — Voltage sag — *SOH=0.3*.

## 3) Ablation/Experiments (`outputs/experiments/run_20260201_151319/`)

### 3.1 Run-Level Summary
- `outputs/experiments/run_20260201_151319/summary.csv` — All experiments summary — *Ablation Suite*.
- `outputs/experiments/run_20260201_151319/registry.json` — Experiment registry — *Ablation Suite*.

### 3.2 Comparison Figures
- `outputs/experiments/run_20260201_151319/comparison_figures/comparison_rmse_v.png` — RMSE by experiment — *Ablation Comparison*.
- `outputs/experiments/run_20260201_151319/comparison_figures/comparison_mae_v.png` — MAE by experiment — *Ablation Comparison*.
- `outputs/experiments/run_20260201_151319/comparison_figures/comparison_energy_wh.png` — Energy usage — *Ablation Comparison*.
- `outputs/experiments/run_20260201_151319/comparison_figures/comparison_t_end_s.png` — Time-to-empty — *Ablation Comparison*.
- `outputs/experiments/run_20260201_151319/comparison_figures/comparison_temp_max_k.png` — Peak temperature — *Ablation Comparison*.

### 3.3 Variants (each has the same set of scenario outputs + figures)
**Common meaning across all variants below:**
- `Day_in_the_Life.csv`, `Gaming.csv`, `Video_Streaming.csv`, `Winter_Usage.csv` — scenario time series.
- `climate_stress_summary_Gaming.csv` — climate matrix summary for that variant.
- `figures/*.png` — scenario plots (voltage sag, power decomposition, phase portrait, climate stress, day-in-life, uncertainty, validation).
  - `voltage_sag_*.png` — voltage sag & cutoff.
  - `power_decomposition_*.png` — component power.
  - `phase_portrait.png` — SOC vs temp phase portrait.
  - `day_in_life_soc_temp.png` — SOC & temp over 12h.
  - `climate_stress_Gaming.png` — 0/25/40C overlays.
  - `validation_phone_validation_bcm_s9_combined.png` — phone-level validation.
  - `uncertainty_band_*` + `uncertainty_hist_*` — Monte Carlo.
  - `monte_carlo/*.csv` — uncertainty percentiles & TTE samples (if enabled).
  - `summary.csv` — per-variant summary.
  - `validation_metrics.json` — per-variant validation metrics.
  - `config_effective.yaml` — exact config used for that variant.

**Variants and their files:**

#### baseline_ocv_r
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/Day_in_the_Life.csv`
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/Gaming.csv`
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/Video_Streaming.csv`
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/Winter_Usage.csv`
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/climate_stress_summary_Gaming.csv`
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/summary.csv`
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/validation_metrics.json`
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/config_effective.yaml`
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/figures/*.png`
- `outputs/experiments/run_20260201_151319/baseline_ocv_r/monte_carlo/*.csv`

#### temp_only
- `outputs/experiments/run_20260201_151319/temp_only/Day_in_the_Life.csv`
- `outputs/experiments/run_20260201_151319/temp_only/Gaming.csv`
- `outputs/experiments/run_20260201_151319/temp_only/Video_Streaming.csv`
- `outputs/experiments/run_20260201_151319/temp_only/Winter_Usage.csv`
- `outputs/experiments/run_20260201_151319/temp_only/climate_stress_summary_Gaming.csv`
- `outputs/experiments/run_20260201_151319/temp_only/summary.csv`
- `outputs/experiments/run_20260201_151319/temp_only/validation_metrics.json`
- `outputs/experiments/run_20260201_151319/temp_only/config_effective.yaml`
- `outputs/experiments/run_20260201_151319/temp_only/figures/*.png`
- `outputs/experiments/run_20260201_151319/temp_only/monte_carlo/*.csv`

#### rc_only
- `outputs/experiments/run_20260201_151319/rc_only/Day_in_the_Life.csv`
- `outputs/experiments/run_20260201_151319/rc_only/Gaming.csv`
- `outputs/experiments/run_20260201_151319/rc_only/Video_Streaming.csv`
- `outputs/experiments/run_20260201_151319/rc_only/Winter_Usage.csv`
- `outputs/experiments/run_20260201_151319/rc_only/climate_stress_summary_Gaming.csv`
- `outputs/experiments/run_20260201_151319/rc_only/summary.csv`
- `outputs/experiments/run_20260201_151319/rc_only/validation_metrics.json`
- `outputs/experiments/run_20260201_151319/rc_only/config_effective.yaml`
- `outputs/experiments/run_20260201_151319/rc_only/figures/*.png`
- `outputs/experiments/run_20260201_151319/rc_only/monte_carlo/*.csv`

#### full_model
- `outputs/experiments/run_20260201_151319/full_model/Day_in_the_Life.csv`
- `outputs/experiments/run_20260201_151319/full_model/Gaming.csv`
- `outputs/experiments/run_20260201_151319/full_model/Video_Streaming.csv`
- `outputs/experiments/run_20260201_151319/full_model/Winter_Usage.csv`
- `outputs/experiments/run_20260201_151319/full_model/climate_stress_summary_Gaming.csv`
- `outputs/experiments/run_20260201_151319/full_model/summary.csv`
- `outputs/experiments/run_20260201_151319/full_model/validation_metrics.json`
- `outputs/experiments/run_20260201_151319/full_model/config_effective.yaml`
- `outputs/experiments/run_20260201_151319/full_model/figures/*.png`
- `outputs/experiments/run_20260201_151319/full_model/monte_carlo/*.csv`

#### ablation_no_mc
- `outputs/experiments/run_20260201_151319/ablation_no_mc/Day_in_the_Life.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_mc/Gaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_mc/Video_Streaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_mc/Winter_Usage.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_mc/climate_stress_summary_Gaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_mc/summary.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_mc/validation_metrics.json`
- `outputs/experiments/run_20260201_151319/ablation_no_mc/config_effective.yaml`
- `outputs/experiments/run_20260201_151319/ablation_no_mc/figures/*.png`

#### ablation_no_thermal
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/Day_in_the_Life.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/Gaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/Video_Streaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/Winter_Usage.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/climate_stress_summary_Gaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/summary.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/validation_metrics.json`
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/config_effective.yaml`
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/figures/*.png`
- `outputs/experiments/run_20260201_151319/ablation_no_thermal/monte_carlo/*.csv`

#### ablation_no_polarization
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/Day_in_the_Life.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/Gaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/Video_Streaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/Winter_Usage.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/climate_stress_summary_Gaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/summary.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/validation_metrics.json`
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/config_effective.yaml`
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/figures/*.png`
- `outputs/experiments/run_20260201_151319/ablation_no_polarization/monte_carlo/*.csv`

#### ablation_no_calibration
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/Day_in_the_Life.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/Gaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/Video_Streaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/Winter_Usage.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/climate_stress_summary_Gaming.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/summary.csv`
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/validation_metrics.json`
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/config_effective.yaml`
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/figures/*.png`
- `outputs/experiments/run_20260201_151319/ablation_no_calibration/monte_carlo/*.csv`
