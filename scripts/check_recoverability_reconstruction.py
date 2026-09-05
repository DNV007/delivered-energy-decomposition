#!/usr/bin/env python3
"""Internal stress test: can the recoverable charge be bounded without a low-rate run?

The manuscript names a matched low-rate cold discharge as the decisive next
experiment and declines to estimate its outcome. A referee may reasonably ask why,
since the archive contains a rested-voltage curve and a SOC-resolved pulse
resistance. The obvious reconstruction is

    V_load(q, I) = V_rest(q) - I * R(q, T),

solved for the charge at which V_load meets the cutoff, and evaluated at currents
below the protocol's 1.5 A.

This script tests that reconstruction before trusting it. The test is calibration
against a known answer: at I = 1.5 A the reconstruction must reproduce the
measured loaded curve and the measured endpoint Q(5 degC) = 1.284 Ah. It is run
across every defensible choice of pulse timescale, charge-axis normalisation and
interpolation, so that the spread between choices can be compared with the effect
being estimated.

Result: the reconstruction fails calibration and is not used in the manuscript.
See results/internal/ for the written assessment.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "results/tables/table_delivered_energy_segments.csv"
RMAP = ROOT / "results/tables/table_raw_pulse_soc_temperature_map.csv"
OUT = ROOT / "results/tables/table_recoverability_stress_test.csv"
CUTOFF, FAMILY, TEMP = 1.49, "SIB", 5.0
RCOLS = {"fast": "median_r_fast_mohm", "1s": "median_r_1s_mohm", "10s": "median_r_10s_mohm"}


def main() -> int:
    s = pd.read_csv(SEG)
    s = s[(s.direction == "discharge") & (s.cell_family == FAMILY)
          & (s.temperature_deg_c == TEMP)].sort_values("segment_index")
    q = np.cumsum(s.segment_capacity_ah.values)
    v_rest = s.ocv_voltage_v.values
    v_meas = s.mean_terminal_voltage_v.values
    Q5 = float(q[-1])

    rm = pd.read_csv(RMAP)
    r = rm[(rm.cell_family == FAMILY) & (rm.temperature_deg_c == TEMP)
           & (rm.pulse_direction == "discharge") & (rm.abs_c_rate == 1.0)] \
        .sort_values("nominal_soc_percent")

    rows = []
    for tag, col in RCOLS.items():
        R = r[col].values / 1000.0
        for Qn, nlab in ((Q5, "Q(5C)"), (1.5735, "Q(25C)")):
            qR = (1.0 - r.nominal_soc_percent.values / 100.0) * Qn
            o = np.argsort(qR)
            for kind in ("linear", "pchip"):
                f = (PchipInterpolator(qR[o], R[o], extrapolate=True) if kind == "pchip"
                     else interp1d(qR[o], R[o], fill_value="extrapolate"))
                pred = v_rest - 1.5 * f(q)
                err = 1000.0 * (pred - v_meas)
                below = np.where(pred <= CUTOFF)[0]
                rows.append(dict(r_timescale=tag, charge_norm=nlab, interp=kind,
                                 rms_err_mv=float(np.sqrt(np.mean(err ** 2))),
                                 endpoint_err_mv=float(err[-1]),
                                 predicted_v_end=float(pred[-1]),
                                 q_at_cutoff_1p5A=(float(q[below[0]]) if len(below) else np.nan)))
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Calibration target: measured Q(5 degC) = {Q5:.4f} Ah at 1.5 A, cutoff {CUTOFF} V")
    print(f"Measured last-segment mean terminal voltage: {v_meas[-1]:.3f} V")
    print(f"Measured minimum segment-mean voltage:       {v_meas.min():.3f} V\n")
    print(out.round(1).to_string(index=False))
    n_ok = out.q_at_cutoff_1p5A.notna().sum()
    print(f"\nvariants reaching the cutoff at 1.5 A: {n_ok} of {len(out)}")
    print(f"spread of predicted end voltage across variants: "
          f"{1000*(out.predicted_v_end.max()-out.predicted_v_end.min()):.0f} mV")
    print(f"margin from the last measured segment mean to the cutoff: "
          f"{1000*(v_meas[-1]-CUTOFF):.0f} mV")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
