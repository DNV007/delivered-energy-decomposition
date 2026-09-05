# Raw Pulse Descriptor Report

Rows: 2520

## Coverage

| cell_family   |   temperature_deg_c |   pulse_count |
|:--------------|--------------------:|--------------:|
| LFP           |                   5 |           126 |
| LFP           |                  15 |           126 |
| LFP           |                  25 |           126 |
| LFP           |                  35 |           126 |
| LFP           |                  45 |           126 |
| LTO           |                   5 |           126 |
| LTO           |                  15 |           126 |
| LTO           |                  25 |           126 |
| LTO           |                  35 |           126 |
| LTO           |                  45 |           126 |
| NMC           |                   5 |           126 |
| NMC           |                  15 |           126 |
| NMC           |                  25 |           126 |
| NMC           |                  35 |           126 |
| NMC           |                  45 |           126 |
| SIB           |                   5 |           126 |
| SIB           |                  15 |           126 |
| SIB           |                  25 |           126 |
| SIB           |                  35 |           126 |
| SIB           |                  45 |           126 |

## Extraction Rules

- Pulse StepIDs are mapped from the raw checkup protocol: 28/32/36/40/44/48 and 62/66/70/74/78/82.
- Resistance uses absolute voltage change divided by current step from the previous rest segment.
- `r_fast_ohm`, `r_100ms_ohm`, `r_1s_ohm`, and `r_10s_ohm` are operational pulse descriptors.
- `complete_10s` flags pulses that lasted at least 9.9 s; shorter boundary-clipped pulses should not be used for 10 s resistance.
- `nominal_soc_percent` is assigned from pulse-block order and should be treated as a protocol SOC index.
