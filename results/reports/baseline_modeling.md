# Baseline Modeling

These are small-data screening baselines over 20 family-temperature rows.

## Validation Splits

- Leave-one-temperature-out.
- Leave-one-family-out.
- Li-ion to sodium-ion transfer.
- Sodium-ion to Li-ion transfer.

## Best Ridge Scores

- target_usable_energy_retention_vs_25c: best split summary is leave_one_temperature_out with mean RMSE 0.0109 and mean skill 0.745.
- target_pulse_r_1s_growth_vs_25c: best split summary is leave_one_temperature_out with mean RMSE 0.2128 and mean skill 0.699.
- target_pulse_power_proxy_vs_25c: best split summary is leave_one_temperature_out with mean RMSE 0.2816 and mean skill -0.183.
- target_low_temperature_penalty: best split summary is leave_one_family_out with mean RMSE 0.0153 and mean skill 0.478.

## Largest Ridge Coefficients

- target_usable_energy_retention_vs_25c: discharge_capacity_retention_vs_25c (0.030), pulse_1c_discharge_median_polarization_1s_mohm (-0.006), entropy_discharge_mean_abs_dudt_mv_per_k (-0.004)
- target_pulse_r_1s_growth_vs_25c: temperature_deg_c (-0.204), bol_thermal_discharge_mean_temperature_rise_max_k (-0.181), pulse_1c_discharge_median_polarization_1s_mohm (0.179)
- target_pulse_power_proxy_vs_25c: temperature_deg_c (0.227), entropy_discharge_mean_abs_dudt_mv_per_k (-0.203), bol_thermal_discharge_mean_temperature_rise_max_k (0.183)
- target_low_temperature_penalty: discharge_capacity_retention_vs_25c (-0.028), pulse_1c_discharge_median_polarization_1s_mohm (0.009), pulse_1c_charge_median_polarization_1s_mohm (0.006)

## Cautions

- The sample size is too small for final model claims.
- Coefficients are standardized and should be interpreted as screening indicators only.
- Transfer splits are intentionally hard and should mainly reveal extrapolation risk.
