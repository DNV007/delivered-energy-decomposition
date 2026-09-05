# Initial Descriptor Extraction Report

- cell_stream_availability: 37 rows, 5 columns
- hppc_resistance_long: 1512 rows, 18 columns
- capacity_temperature_summary: 20 rows, 16 columns
- ocv_temperature_points: 4200 rows, 15 columns
- ocv_current_rate_summary: 20 rows, 15 columns
- eis_spectral_descriptors: 40 rows, 21 columns
- drt_descriptors: 40 rows, 13 columns
- entropy_dudt_long: 88 rows, 12 columns
- thermal_response_descriptors: 14 rows, 16 columns

## Notes

- Raw files were read directly from `data_raw/data_EvalSIB.zip`; the archive was not extracted or modified.
- Temperature checkup capacity uses StepIDs 96/97 and 106/107, following the upstream `OCV_T_plotting.py` script.
- OCV current-rate summaries use StepIDs 6, 12, 18, 24, and 30, following `OCV_I_plotting.py`.
- Thermal summaries use dS025C StepIDs 6-7 and 9-10, following `temperature_plotting.py`.
- EIS descriptors are operational spectral descriptors, not equivalent-circuit claims.
