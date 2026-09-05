# Raw Pulse Findings

These descriptors come from segmented raw checkup pulse responses, not the processed HPPC CSV files.

## Immediate Signals

- SIB 1C discharge median R_1s at 25 deg C is 91.1 mohm; the Li-ion reference mean is 50.3 mohm (SIB/Li ratio 1.81).
- SIB 1C discharge median R_1s at 5 deg C is 286.6 mohm, 3.15x its 25 deg C value.

## Cautions

- Pulse SOC is a protocol index assigned from pulse-block order, not an independently integrated SOC estimate.
- Clipped pulses are retained in the raw table but excluded from 1 s or 10 s summaries using completeness flags.
- These are operational resistance descriptors; equivalent-circuit fitting has been checked separately and rejected for core interpretation.
