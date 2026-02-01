# Experiments & Scripts Index

This document lists **all experiments performed in this project** and the **scripts that generate the corresponding outputs**.

## 1) Main Scenario Simulation (baseline run)
- **Script:** `solution.py`
- **Config:** `config.yaml` (scenarios, solver, validation, uncertainty, climate_stress)
- **Outputs:**
  - Scenario CSVs: `outputs/Video_Streaming.csv`, `outputs/Gaming.csv`, `outputs/Winter_Usage.csv`
  - Figures: `outputs/figures/voltage_sag_*.png`, `outputs/figures/power_decomposition_*.png`, `outputs/figures/phase_portrait.png`

## 2) Day-in-the-Life (mixed 12h load)
- **Script:** `solution.py` (scenario defined in `config.yaml`)
- **Scenario Name:** `Day in the Life`
- **Outputs:**
  - Data: `outputs/Day_in_the_Life.csv`
  - Figure: `outputs/figures/day_in_life_soc_temp.png`

## 3) Climate Stress Matrix (0C / 25C / 40C)
- **Script:** `solution.py` (climate_stress block in `config.yaml`)
- **Base Scenario:** `Gaming`
- **Outputs:**
  - Summary: `outputs/climate_stress_summary_Gaming.csv`
  - Figure: `outputs/figures/climate_stress_Gaming.png`

## 4) Uncertainty Quantification (Monte Carlo)
- **Script:** `solution.py` (uncertainty block in `config.yaml`)
- **Base Scenario:** `Video Streaming`
- **Outputs:**
  - Figures: `outputs/figures/uncertainty_band_Video_Streaming.png`, `outputs/figures/uncertainty_hist_Video_Streaming.png`
  - Tables: `outputs/monte_carlo/uncertainty_percentiles_Video_Streaming.csv`, `outputs/monte_carlo/uncertainty_tte_Video_Streaming.csv`

## 5) Validation (phone current profile)
- **Script:** `solution.py`
- **Validation Data:** `datasets/phone_validation_real/phone_validation_bcm_s9_combined.csv`
- **Outputs:**
  - Figure: `outputs/figures/validation_phone_validation_bcm_s9_combined.png`
  - Metrics: included in `data_analyze/conclusions.md`

## 6) SOH History Comparison (aging effect)
- **Script:** `data_analyze/run_soh_comparison.py`
- **Outputs:**
  - Summary: `outputs/soh_comparison/20260131_195801/soh_comparison.csv`
  - Figures: `outputs/soh_comparison/20260131_195801/soh_voltage_time.png`, `soh_temp_time.png`, `soh_energy_bar.png`
  - Per-SOH runs: `outputs/soh_comparison/20260131_195801/SOH_*/Gaming.csv`

## 7) Ablation Suite (model complexity vs performance)
- **Script:** `data_analyze/run_experiments.py`
- **Run Directory:** `outputs/experiments/run_20260201_151319/`
- **Outputs:**
  - Summary: `outputs/experiments/run_20260201_151319/summary.csv`
  - Comparison figures: `outputs/experiments/run_20260201_151319/comparison_figures/`
  - Variant outputs: `baseline_ocv_r/`, `temp_only/`, `rc_only/`, `full_model/`,
    `ablation_no_mc/`, `ablation_no_thermal/`, `ablation_no_polarization/`, `ablation_no_calibration/`

## 8) Parameter Estimation (data-driven)
- **OCV fitting:** `fit_params.py` → example output: `outputs/parameter_validation_ocv_phone.png`
- **Thermal parameter fitting:** `derive_params.py` → outputs in `outputs/` (fit plots)
- **Power profile extraction:** `extract_power_profile.py` → console + config values

## 9) Validation Data Preparation (phone profile)
- **Scripts:** `data_analyze/prepare_phone_validation.py`, `data_analyze/prepare_phone_validation_bcm.py`
- **Outputs:** `datasets/phone_validation/*.csv`, `datasets/phone_validation_real/*.csv`

