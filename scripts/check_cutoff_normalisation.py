#!/usr/bin/env python3
"""Are the cross-family shares an artefact of four different cutoff voltages?

The four products stop at four vendor limits (1.49 V SIB, 1.98 V LFP, 2.49 V NMC,
1.48 V LTO), so the shorter-window shares compared across families in Table 2 are
measured against four differently drawn lines. A fair objection is that the SIB's
large shorter-window share reflects where its line happens to fall on a steep
hard-carbon profile rather than anything about the cell.

This tests that directly. For each family a common stopping line is set at both
temperatures -- the higher of the two per-segment mean terminal voltage minima --
and then raised by a margin delta. Raising the line simulates a more conservative
cutoff. Both temperatures are truncated by the same line, so the comparison stays
internally matched, and the three-term decomposition is recomputed at each delta.

If the shares are stable as the line moves, the cross-family comparison is not an
artefact of where the lines are drawn. If they move strongly, the comparison has
to be dropped or restated.

The sweep is run for both cold points against the same 25 degC reference. The
manuscripts compare each family's 5 and 15 degC shares under every tested shift,
so that comparison has to come out of this script rather than being asserted: it
was quoted for a year from a 5 degC-only table that could not support it.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "results/tables/table_delivered_energy_segments.csv"
OUT = ROOT / "results/tables/table_cutoff_normalisation.csv"
FAMS = ["SIB", "LFP", "NMC", "LTO"]
DELTAS = [0.0, 0.05, 0.10, 0.15, 0.20]
COLD = [5.0, 15.0]
WARM = 25.0


def series(d, fam, T):
    s = d[(d.cell_family == fam) & (d.temperature_deg_c == T)].sort_values("segment_index")
    return (s.segment_capacity_ah.values, s.mean_terminal_voltage_v.values,
            s.ocv_voltage_v.values, s.delivered_energy_wh.values, s.ocv_energy_wh.values)


def truncate(dq, vbar, vocv, edel, eocv, line):
    """Keep segments up to the last one whose mean terminal voltage is >= line."""
    keep = vbar >= line
    if not keep.any():
        return None
    last = np.max(np.where(keep)[0])
    sl = slice(0, last + 1)
    return dq[sl], vocv[sl], edel[sl].sum(), eocv[sl].sum()


def integral_to(dq, v, Qc):
    tot, prev = 0.0, 0.0
    for qi, vi, ci in zip(dq, v, np.cumsum(dq)):
        if ci <= Qc:
            tot += qi * vi
        else:
            tot += qi * vi * max(0.0, Qc - prev) / qi
            break
        prev = ci
    return tot


def main() -> int:
    d = pd.read_csv(SEG)
    d = d[d.direction == "discharge"]
    rows = []
    # 5 degC first: check_manuscript_numbers.py and the SI tables quote that block,
    # and a reader scanning the CSV should meet the published rows first.
    for cold_t in COLD:
        for fam in FAMS:
            cc = series(d, fam, cold_t)
            c25 = series(d, fam, WARM)
            # Each cold point gets its own common line, because the line is the
            # higher of the two minima and the cold minimum moves with temperature.
            line0 = max(cc[1].min(), c25[1].min())
            for delta in DELTAS:
                L = line0 + delta
                tc = truncate(*cc, L)
                t25 = truncate(*c25, L)
                if tc is None or t25 is None:
                    continue
                dqc, vc, Ec, Oc = tc
                dq25, v25, E25, O25 = t25
                Qcold, Q25 = dqc.sum(), dq25.sum()
                short = E25 - Ec
                if short <= 1e-6:
                    continue
                Qc = min(Qcold, Q25)
                trunc = (O25 - integral_to(dq25, v25, Qc)) - (Oc - integral_to(dqc, vc, Qc))
                shift = integral_to(dq25, v25, Qc) - integral_to(dqc, vc, Qc)
                gap = (Oc - Ec) - (O25 - E25)
                rows.append(dict(cold_deg_c=cold_t, family=fam,
                                 delta_mv=int(1000 * delta), line_v=L,
                                 Q_cold=Qcold, Q25=Q25, shortfall_wh=short,
                                 pct_shortfall=100 * short / E25,
                                 share_window=100 * trunc / short,
                                 share_gap=100 * gap / short,
                                 share_shift=100 * shift / short,
                                 closure=100 * (trunc + gap + shift) / short))
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    shares = {}
    for cold_t in COLD:
        block = out[out.cold_deg_c == cold_t]
        w = block.pivot(index="family", columns="delta_mv", values="share_window").reindex(FAMS)
        shares[cold_t] = w
        print(f"Shorter-window share (%), {cold_t:g} vs {WARM:g} degC, "
              f"as the stopping line is raised at both temperatures:\n")
        print("  delta above the family's own line (mV)")
        print(w.round(1).to_string())
        print("\nLoad-to-rest gap share (%):")
        print(block.pivot(index="family", columns="delta_mv", values="share_gap").reindex(FAMS).round(1).to_string())
        print("\nDelivered-energy shortfall (% of the 25 degC value):")
        print(block.pivot(index="family", columns="delta_mv", values="pct_shortfall").reindex(FAMS).round(2).to_string())
        rng = (w.max(axis=1) - w.min(axis=1))
        print("\nrange of the shorter-window share over a 0-200 mV line shift (pp):")
        for f in FAMS:
            print(f"   {f:4s} {rng[f]:5.1f}")
        print(f"\nSIB remains the largest of the four at every delta: "
              f"{bool((w.loc['SIB'] >= w.drop(index='SIB').max()).all())}\n")

    print(f"identity closure across all cases: "
          f"{out.closure.min():.4f} to {out.closure.max():.4f} %")

    # The quantity the manuscripts quote: how far each family's shorter-window
    # share moves between its two cold points, at every tested delta.
    print("\nchange in the shorter-window share from 5 to 15 degC (pp, signed):")
    print("  delta (mV) " + " ".join(f"{int(1000 * x):>6d}" for x in DELTAS) + "     max |change|")
    for f in FAMS:
        signed = [shares[15.0].loc[f, int(1000 * x)] - shares[5.0].loc[f, int(1000 * x)]
                  for x in DELTAS]
        print(f"   {f:4s}      " + " ".join(f"{v:+6.1f}" for v in signed)
              + f"     {max(abs(v) for v in signed):5.1f}")
    print(f"SIB above both graphite-anode references at every delta: "
          f"{bool((w.loc['SIB'] > w.loc[['LFP','NMC']].max()).all())}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
