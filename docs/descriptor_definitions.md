# Descriptor Definitions

This file tracks the project descriptors that must be computed reproducibly and reported in the manuscript.

| Descriptor | Source | Definition | Output Folder |
|---|---|---|---|
| Capacity retention | Capacity check | Capacity at temperature normalized to 25 deg C | `data_processed/capacity/` |
| Usable Energy Retention Surface, UERS | OCV + capacity | Capacity-scaled OCV integral over a defined SOC window, normalized to 25 deg C | `data_processed/ocv/` |
| Instantaneous Resistance Surface, IRS | HPPC | Immediate voltage step divided by current step | `data_processed/hppc/` |
| Pulse-duration resistance | HPPC | Voltage change at fixed pulse duration divided by current step | `data_processed/hppc/` |
| Polarization Relaxation Index, PRI | HPPC | Relaxation contribution after pulse or fitted RC relaxation contribution | `data_processed/hppc/` |
| Raw temperature pulse resistance | Raw checkup pulse parquet | Operational resistance from segmented raw pulses at fast, 100 ms, 1 s, 10 s, and pulse end; use completeness flags for fixed-duration summaries | `data_processed/hppc/` |
| Raw pulse polarization | Raw checkup pulse parquet | Difference between pulse-duration resistance and fast resistance for the same pulse | `data_processed/hppc/` |
| Raw pulse relaxation | Raw checkup pulse parquet | Voltage recovery after pulse end divided by pulse current step | `data_processed/hppc/` |
| Temperature Fragility Index, TFI | HPPC or EIS | Slope of log resistance versus inverse absolute temperature | `data_processed/descriptors/` |
| OCV Slope Penalty, OSP | OCV | Integral of absolute dV/dcapacity-fraction over the selected capacity window | `data_processed/ocv/` |
| Series resistance proxy | EIS | High-frequency intercept or closest robust high-frequency real impedance estimate | `data_processed/eis/` |
| Charge-transfer/arc proxy | EIS | Mid-frequency arc contribution or robust semicircle proxy | `data_processed/eis/` |
| Low-frequency transport proxy | EIS | Low-frequency impedance growth, slope, or Warburg-like descriptor | `data_processed/eis/` |
| Entropic Heat Sensitivity, EHS | Entropy coefficient | Absolute T times dE/dT over SOC | `data_processed/entropy/` |
| Thermal-response descriptor | Thermal response | Temperature-rise or cooling metric normalized by current/capacity where possible | `data_processed/thermal/` |
| Sodium-Ion Performance Fragility Number, SIPFN | Integrated descriptors | Optional composite; use only if weights are justified | `data_processed/descriptors/` |

## Rules

1. Every descriptor must include units, cell identity, temperature, SOC or SOC window, and source file.
2. Every smoothing, interpolation, or filtering decision must be logged.
3. Composite metrics are exploratory until weights are justified by regression, PCA, or a clearly stated equal-weight baseline.
4. Do not overinterpret EIS circuit parameters unless the fitting quality and parameter stability support it.

## Equivalent-Circuit Fitting Decision

The initial reliability check rejects equivalent-circuit parameters for the core analysis. Randles-CPE and Randles-CPE-Warburg candidates were fit to all room-temperature EIS spectra, but neither model produced spectra passing the combined residual, parameter-boundary, and multi-start stability criteria. Use the robust EIS descriptors instead.
