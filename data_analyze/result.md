# Parameter Analysis Report

## Key Derived Metrics
- Thermal time constant: **638.2 s**
- Screen max power (from density × area): **4.50 W**
- Nominal energy estimate (Q_design × 3.85V): **19.25 Wh**

## Battery Parameters (Selected)

| Parameter | Value | Units | Source |
|---|---:|---|---|
| Q_design_ah | 5.0 | Ah | smartphone_parameters_detailed.csv (2026通用推荐值, Q_design 5000 mAh) |
| V_cutoff | 3.0 | V | smartphone_parameters_detailed.csv (V_cutoff) |
| R_ref | 0.03 | Ohm | smartphone_parameters_detailed.csv (R_ref 30 mΩ) |
| Ea_R | 27538.0 | J/mol | derive_params.py (Panasonic 18650PF multi-temp fit) |
| Ea_cap | 10100.0 | J/mol | derive_params.py (Panasonic 18650PF multi-temp fit) |
| k_cycle | 1.111e-08 | 1/(A*s) | smartphone_parameters_detailed.csv (k_cycle 4e-5 1/Ah => /3600) |
| k_cal | 1e-07 | 1/s | smartphone_parameters_detailed.csv (k_cal 1e-7 1/s) |

## Power Parameters (Selected)

| Parameter | Value | Units | Source |
|---|---:|---|---|
| P_idle | 0.1 | W | smartphone_parameters_detailed.csv (P_idle 0.1 W) |
| P_little_max | 6.0 | W | smartphone_parameters_detailed.csv (P_game_avg mapped to little cluster) |
| P_big_max | 15.0 | W | smartphone_parameters_detailed.csv (P_peak_cpu mapped to big cluster) |
| wifi_idle_power | 0.1 | W | recommended smartphone measured idle (~0.1W) |
| wifi_active_power | 1.5 | W | recommended smartphone measured active (~1.2-1.8W) |
| gps_on_power | 0.5 | W | recommended smartphone measured GPS (~0.4-0.6W) |

## Notes
- OCV curve uses the Combined Model parameters in `parameters.json`.
- Derived metrics computed from provided parameters; update inputs to refresh.
- Full parameter table exported to `data_analyze/parameters_table.csv`.
