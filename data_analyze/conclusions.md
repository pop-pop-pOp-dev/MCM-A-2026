# Model Conclusions

## Scenario Summary
- Video_Streaming: t_end=4875s, energy=12.78Wh, v_end=3.168V, soc_end=0.029
- Gaming: t_end=1980s, energy=10.27Wh, v_end=3.183V, soc_end=0.082
- Winter_Usage: t_end=6615s, energy=12.33Wh, v_end=3.178V, soc_end=0.131

## Validation
- Mode=current_profile, RMSE=0.0232 V, MAE=0.0187 V on 344 samples (profile: phone_validation_bcm_s9_combined.csv, scale=0.800, r=0.700, ocv=0.984, offset=0.075V).

## Parameter Coverage
- data_driven: 18
- assumed: 28
- constant: 1

## Notes
- Metrics are derived from generated outputs and current config parameters.