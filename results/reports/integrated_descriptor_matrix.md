# Integrated Descriptor Matrix

Long descriptor rows: 14463
Family-temperature rows: 20
Target rows: 20

## Long Matrix Coverage

- raw_pulse_soc_map: 9077 descriptor rows
- ocv_slope: 2880 descriptor rows
- raw_pulse: 1558 descriptor rows
- eis: 320 descriptor rows
- entropy: 264 descriptor rows
- ocv_usable_energy: 200 descriptor rows
- thermal_response: 84 descriptor rows
- capacity: 80 descriptor rows

## Modeling Targets

- `target_usable_energy_retention_vs_25c`: discharge OCV-aware usable-energy retention.
- `target_capacity_retention_vs_25c`: discharge capacity retention.
- `target_pulse_r_1s_growth_vs_25c`: 1C discharge raw-pulse R_1s normalized to 25 deg C.
- `target_pulse_power_proxy_w`: OCV-based V^2/(4R_1s) power proxy.
- `target_energy_retention_loss_vs_25c`: 1 minus usable-energy retention.
- `target_low_temperature_penalty`: energy-retention loss below 25 deg C; zero at and above 25 deg C.

## Cautions

- The long matrix mixes granularities; use `aggregation_level` before modeling.
- BOL entropy and thermal descriptors are family-level context and are repeated in the wide matrix by family where used.
- Raw pulse SOC has been validated as an ordered protocol block index, not as exact coulomb-counted SOC.
- EIS descriptors are robust spectral descriptors; ECM fitted parameters are intentionally excluded from the core matrix.
