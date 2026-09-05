# OCV Slope Analysis

OCV curves were smoothed with centered rolling windows of 1, 5, and 11 capacity segments. The primary OCV slope penalty (OSP) uses the 5-segment smoothed curve and integrates abs(dV/dcapacity-fraction) over the central 20-80% capacity window.

5 deg C central discharge OSP ranking is led by SIB.
SIB central discharge OSP at 5 deg C is 0.911 V, versus a Li-ion reference mean of 0.240 V (SIB/Li=3.80).

Interpretation: OSP is a thermodynamic/voltage-shape descriptor. It should be used as context beside capacity, raw-pulse, and EIS descriptors rather than as a standalone kinetic metric.
