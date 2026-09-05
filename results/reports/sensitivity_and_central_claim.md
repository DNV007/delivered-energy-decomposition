# Sensitivity Checks and Central Claim

Central claim status: locked for the initial descriptor pass.

Claim: The commercial sodium-ion cell shows a mixed low-temperature bottleneck: usable-energy loss is visible in the OCV/capacity descriptors, but the differentiating limitation versus Li-ion references is the stronger resistance/polarization/transport penalty seen in raw pulse and EIS descriptors. Equivalent-circuit fitted parameters are not used because they fail reliability checks.

Evidence items supporting the claim: 6/6.

## Evidence

- energy_penalty_5c: supports. SIB has the largest 5 deg C OCV-aware energy-retention penalty. Metric=0.1746692932683803; reference=0.0518013548586872.
- pulse_timepoint_robustness: supports. SIB 5 deg C resistance excess is robust across pulse time points. Metric=5; reference=5.
- ocv_window_robustness: supports. SIB 5 deg C usable-energy retention remains lowest across OCV capacity windows. Metric=4; reference=4.
- eis_transport_proxy: supports. SIB has the largest room-temperature low-frequency EIS Zre proxy. Metric=88.88945; reference=40.020783333333334.
- ecm_rejection: supports. Equivalent-circuit fitted parameters are rejected for core interpretation. Metric=0; reference=160.
- baseline_importance_split: supports. Baseline screening separates energy retention from resistance-growth bottlenecks. Metric=energy=discharge_capacity_retention_vs_25c; growth=temperature_deg_c; reference=ridge_core top-ranked standardized coefficients.

## Remaining Limits

- This claim is locked for the current descriptor pass, not for final manuscript submission.
- Protocol SOC has been checked against integrated raw block capacity and should be treated as a protocol block index, not a precise coulomb-counted SOC.
- Capacity and OCV extraction has been checked against source-publication trend values; independent figure digitization was not performed.
- Flexible models are not recommended yet because there are only 20 family-temperature modeling rows.
