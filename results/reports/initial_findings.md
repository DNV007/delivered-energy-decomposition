# Initial Findings

These are preliminary descriptor outputs generated from the downloaded DepositOnce archive.

## Immediate Signals

- SIB discharge capacity retention at 5 deg C is 0.816; OCV-aware usable-energy retention is 0.825.
- SIB 25 deg C baseline discharge capacity is 1.573 Ah and OCV-aware discharge energy is 4.908 Wh.
- The largest median low-frequency EIS Zre proxy near 25 deg C is SIB at 88.9 mohm.

## Cautions

- HPPC processed CSVs do not encode temperature in the filename; treat them as processed BOL/nominal-temperature descriptors until confirmed from the source paper.
- EIS descriptors here are robust spectral proxies, not fitted equivalent-circuit parameters.
- Capacity and OCV extraction follows upstream plotting-script StepIDs and now has publication-trend validation output.
