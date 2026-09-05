#!/usr/bin/env python3
"""Per-segment instantaneous terminal-voltage minima of the incremental discharge.

The decomposition is built from per-segment current-weighted mean voltages, which
approach the cutoff without reaching it. The cutoff acts on the instantaneous
voltage, so the evidence that the discharge is cutoff-limited lives in the
sample-level minima. This extracts them so the trace figure can show the cutoff
being met rather than asserting it in prose.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.zip_readers import read_parquet_member

ARCHIVE = ROOT / "data_raw/data_EvalSIB.zip"
OUT = ROOT / "results/tables/table_segment_voltage_minima.csv"
CELL, STEP = "TC23SIB09", 96


def main() -> int:
    rows = []
    for T, tag in ((5, "05deg"), (25, "25deg")):
        d = read_parquet_member(ARCHIVE, f"raw_CU_diffT/{CELL}/{CELL}_{tag}.parquet")
        d.columns = [c.split("[")[0] for c in d.columns]
        d = d.sort_values("Testtime")
        s = d[d.StepID == STEP].copy()
        s["seg"] = (s.Testtime.diff() > 30).cumsum()
        cum = 0.0
        for k, g in s.groupby("seg"):
            t = g.Testtime.values
            dq = float(np.sum(np.abs(g.Current.values) * np.diff(t, prepend=t[0])) / 3600)
            cum += dq
            rows.append(dict(temperature_deg_c=T, segment_index=int(k),
                             cumulative_charge_ah=cum,
                             v_min_instantaneous=float(g.Voltage.min()),
                             v_mean_loaded=float(np.average(g.Voltage.values,
                                                            weights=np.abs(g.Current.values) + 1e-12))))
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    for T in (5, 25):
        sub = out[out.temperature_deg_c == T]
        print(f"SIB {T:>2d} degC: lowest instantaneous sample {sub.v_min_instantaneous.min():.4f} V; "
              f"{int((sub.v_min_instantaneous <= 1.495).sum())} segments reach 1.495 V or below")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
