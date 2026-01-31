# Model Conclusions

## Scenario Summary
- Video_Streaming: t_end=4855s, energy=12.76Wh, v_end=3.174V, soc_end=0.031
- Gaming: t_end=1955s, energy=10.19Wh, v_end=3.182V, soc_end=0.089
- Winter_Usage: t_end=6555s, energy=12.27Wh, v_end=3.170V, soc_end=0.143

## Validation
- Mode=current_profile, RMSE=0.0232 V, MAE=0.0187 V on 344 samples (profile: phone_validation_bcm_s9_combined.csv, scale=0.800, r=0.700, ocv=0.984, offset=0.075V).

## Parameter Coverage
- data_driven: 18
- assumed: 28
- constant: 1

## Notes
- Metrics are derived from generated outputs and current config parameters.