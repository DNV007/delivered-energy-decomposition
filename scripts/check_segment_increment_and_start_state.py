#!/usr/bin/env python3
"""Three controls on the incremental discharge that the decomposition rests on.

(1) Increment size. The checkup discharge uses a constant-current segment whose
    duration is set per temperature, not globally. Reporting a single nominal
    increment (or duty cycle) for the protocol is therefore inaccurate, and the
    5 degC and 25 degC discharges being compared do not share a segment length.

(2) Is Q(T) set by the cell or by the 105-segment schedule? If the discharge
    exhausted its schedule while still above cutoff, Q(T) would be an artefact of
    schedule length. Comparing the delivered charge against the schedule target
    (105 x median increment) settles this directly, without extrapolation.

(3) State preparation. The shorter-window term can be challenged as an initial-
    state artefact if the cold discharge began from a lower state of charge. The
    first post-rest OCV of each discharge is the direct test.

Reads the published segment table; writes its own output and touches nothing else.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "results/tables/table_delivered_energy_segments.csv"
OUTPUT   = ROOT / "results/tables/table_segment_increment_start_state.csv"
FAMILIES = ["SIB", "LFP", "NMC", "LTO"]
REST_S   = 360.0
SCHEDULE = 105


def main() -> int:
    d = pd.read_csv(SEGMENTS)
    d = d[d.direction == "discharge"]
    rows = []
    for fam in FAMILIES:
        for T in sorted(d.temperature_deg_c.unique()):
            s = d[(d.cell_family == fam) & (d.temperature_deg_c == T)].sort_values("segment_index")
            if s.empty:
                continue
            q = s.segment_capacity_ah.values
            inc = float(np.median(q))
            dur = float(np.median(s.duration_s.values))
            rows.append(dict(
                family=fam, temperature_deg_c=T,
                increment_mah=1000 * inc,
                segment_duration_s=dur,
                duty_cycle_pct=100 * dur / (dur + REST_S),
                n_segments=len(q),
                n_cut_short_vs_own_median=int((q < 0.5 * inc).sum()),
                schedule_target_ah=SCHEDULE * inc,
                delivered_ah=float(q.sum()),
                short_of_schedule_ah=SCHEDULE * inc - float(q.sum()),
                cutoff_limited=bool(SCHEDULE * inc - float(q.sum()) > 0),
                first_rested_ocv_v=float(s.ocv_voltage_v.iloc[0]),
                last_rested_ocv_v=float(s.ocv_voltage_v.iloc[-1]),
                min_mean_terminal_v=float(s.mean_terminal_voltage_v.min()),
            ))
    out = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print(f"wrote {OUTPUT.relative_to(ROOT)}  ({len(out)} rows)\n")
    print("(2) schedule target vs delivered -- positive 'short of schedule' means cutoff-limited")
    print(out[out.temperature_deg_c.isin([5.0, 25.0])]
          [["family", "temperature_deg_c", "increment_mah", "segment_duration_s", "duty_cycle_pct",
            "schedule_target_ah", "delivered_ah", "short_of_schedule_ah"]].to_string(index=False))
    print("\n(3) start state: first rested OCV, 5 degC minus 25 degC")
    for fam in FAMILIES:
        a = out[(out.family == fam) & (out.temperature_deg_c == 5.0)].first_rested_ocv_v.iloc[0]
        b = out[(out.family == fam) & (out.temperature_deg_c == 25.0)].first_rested_ocv_v.iloc[0]
        print(f"   {fam:4s} {1000*(a-b):+7.1f} mV")
    return 0


if __name__ == "__main__":
    sys.exit(main())
