#!/usr/bin/env python3
"""Independent test of the rate dependence predicted by the three-term decomposition.

The DepositOnce decomposition attributes 58 percent of the sodium-ion cell's
25->5 degC delivered-energy shortfall to a shorter delivered-charge window: charge
the cold cell never reaches because the loaded voltage arrives at the cutoff
first. If that reading is right, the effect must be rate dependent. Lowering the
current lowers the polarisation that carries the voltage to the cutoff, so the
cold and warm charge windows must converge, and the cold shortfall must shrink,
as the discharge rate falls. At a low enough rate it should nearly vanish.

The DepositOnce archive cannot test this: it discharges at one current only.

The Stanford/SLAC Molicel INR-21700-P42A dataset can (Khan et al., Sci Data 12,
1506, 2025; OSF 10.17605/OSF.IO/9CEAV). Its constant-discharge test (CDT) runs
C/3, C/10 and C/20 at 5, 25 and 40 degC on twelve cells, with every discharge
terminated on the same 2.5 V cutoff. This script computes, per cell and per rate,
the delivered charge and delivered energy at 5 and 25 degC, and their retentions.

It is a test of the predicted *direction and rate dependence* on an independent
chemistry (NMC / Si-graphite), not a reproduction of the 58/27/15 split: the CDT
is a continuous discharge with no per-increment rest, so the rested-voltage
counterfactual the three-term identity needs does not exist on this trajectory.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_external/stanford_p42a"
OUT  = ROOT / "results/tables/table_stanford_rate_dependence.csv"
RATES = {"cby3": "C/3", "cby10": "C/10", "cby20": "C/20"}


def discharge_metrics(struct):
    t, I, V = struct.time_dis, struct.current_dis, struct.voltage_dis
    cap = struct.capacity_dis
    dt = np.diff(t, prepend=t[0])
    E = float(np.sum(V * np.abs(I) * dt) / 3600.0)        # Wh
    Q = float(np.nanmax(cap))                              # Ah
    return Q, E, float(np.median(np.abs(I))), float(V.min())


def main() -> int:
    rows = []
    for f in sorted(DATA.glob("*_CDT_*.mat")):
        m = re.match(r"(A\d+)_CDT_(\d+)degC\.mat", f.name)
        cell, T = m.group(1), int(m.group(2))
        d = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        for key, label in RATES.items():
            if key not in d:
                continue
            Q, E, I, vmin = discharge_metrics(d[key])
            rows.append(dict(cell=cell, temperature_deg_c=T, rate=label,
                             current_a=I, Q_ah=Q, E_wh=E, v_min=vmin))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    piv = df.pivot_table(index=["cell", "rate"], columns="temperature_deg_c",
                         values=["Q_ah", "E_wh"])
    paired = []
    for (cell, rate), r in piv.iterrows():
        try:
            q5, q25 = r[("Q_ah", 5)], r[("Q_ah", 25)]
            e5, e25 = r[("E_wh", 5)], r[("E_wh", 25)]
        except KeyError:
            continue
        if not np.isfinite([q5, q25, e5, e25]).all():
            continue
        paired.append(dict(cell=cell, rate=rate, Q_ret=q5 / q25, E_ret=e5 / e25))
    p = pd.DataFrame(paired)

    print(f"cells x temperature x rate rows: {len(df)}   paired cells: {p.cell.nunique()}")
    print(f"every discharge ends at V_min = {df.v_min.min():.3f}-{df.v_min.max():.3f} V\n")
    print("Delivered charge and energy, mean over cells (Ah / Wh):")
    print(df.groupby(["temperature_deg_c", "rate"])[["Q_ah", "E_wh"]]
            .agg(["mean", "std"]).round(4).to_string())
    print("\n5/25 degC RETENTION, paired per cell:")
    print(f"{'rate':6s} {'n':>3s} {'Q_ret mean':>11s} {'sd':>7s} {'E_ret mean':>11s} {'sd':>7s} "
          f"{'charge loss %':>14s} {'energy loss %':>14s}")
    for rate in ("C/3", "C/10", "C/20"):
        s = p[p.rate == rate]
        if s.empty:
            continue
        print(f"{rate:6s} {len(s):3d} {s.Q_ret.mean():11.4f} {s.Q_ret.std():7.4f} "
              f"{s.E_ret.mean():11.4f} {s.E_ret.std():7.4f} "
              f"{100*(1-s.Q_ret.mean()):14.2f} {100*(1-s.E_ret.mean()):14.2f}")

    print("\nPer-cell charge retention Q(5)/Q(25), by rate:")
    w = p.pivot(index="cell", columns="rate", values="Q_ret")[["C/3", "C/10", "C/20"]]
    print(w.round(4).to_string())
    mono = (w["C/20"] > w["C/10"]) & (w["C/10"] > w["C/3"])
    print(f"\nstrictly monotone improvement C/3 < C/10 < C/20: {int(mono.sum())} of {len(w)} cells")
    print(f"C/20 > C/3 in {int((w['C/20'] > w['C/3']).sum())} of {len(w)} cells")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
