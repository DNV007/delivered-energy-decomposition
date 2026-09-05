# EIS-HPPC Consistency

Room-temperature EIS descriptors were compared with 25 deg C raw 1C discharge pulse descriptors.

## Correlations

- eis_series_resistance_mohm vs pulse_1c_discharge_r_fast_mohm: Pearson r = 0.900, Spearman r = 0.800 (n = 4 families).
- eis_min_abs_zim_resistance_mohm vs pulse_1c_discharge_r_fast_mohm: Pearson r = 0.902, Spearman r = 0.800 (n = 4 families).
- eis_low_freq_zre_mohm vs pulse_1c_discharge_r_1s_mohm: Pearson r = 0.989, Spearman r = 1.000 (n = 4 families).
- eis_arc_width_zre_mohm vs pulse_1c_discharge_polarization_1s_mohm: Pearson r = 0.860, Spearman r = 0.800 (n = 4 families).

## Cautions

- This is a family-level check with only four chemistries, so it is a qualitative consistency screen.
- EIS spectra span voltage points near room temperature, while pulse descriptors are summarized over protocol SOC at 25 deg C.
- Strong disagreement would signal descriptor mismatch; agreement does not prove a circuit model.
