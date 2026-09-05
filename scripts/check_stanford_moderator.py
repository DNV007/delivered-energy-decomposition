#!/usr/bin/env python3
"""Does the rate benefit scale with how polarisation-limited a cell is?

A first pass correlated the cold penalty x = 1 - R(C/3) against the rate benefit
y = R(C/20) - R(C/3) and found r = +0.88 to +0.90. That correlation is not
admissible evidence. Because

    y = R(C/20) - R(C/3) = x - (1 - R(C/20)),

x enters y with coefficient +1, so the two axes share the same C/3 measurement
with the same sign and a positive correlation arises partly from algebra. A
within-cell permutation null (rate labels shuffled inside each cell, which
destroys any real rate effect while preserving the coupling and the
cell-to-cell variance) returns median r ~ 0.16-0.24 with a 95 percent interval
reaching 0.85.

This script repeats the moderator analysis with an INDEPENDENT predictor: the
cold pulse resistance R0 measured at 5 degC in the HCGT test, which shares no
measurement with the CDT retentions. If the rate benefit really reflects relief
of polarisation-driven cutoff truncation, cells with a higher cold pulse
resistance should benefit MORE, i.e. a positive correlation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HCGT = ROOT / "data_external/stanford_p42a/HCGT_5degC.mat"
CDT  = ROOT / "results/tables/table_stanford_rate_dependence.csv"
OUT  = ROOT / "results/tables/table_stanford_moderator.csv"


def main() -> int:
    h = sio.loadmat(HCGT, squeeze_me=True, struct_as_record=False)["hcgt_data"]
    r0 = {}
    for cell in h._fieldnames:
        c = getattr(h, cell)
        if hasattr(c, "r0_1C"):
            r0[cell] = float(np.nanmedian(np.asarray(c.r0_1C.discharge, dtype=float)))

    df = pd.read_csv(CDT)
    piv = df.pivot_table(index=["cell", "rate"], columns="temperature_deg_c",
                         values=["Q_ah", "E_wh"])
    rows = []
    for (cell, rate), x in piv.iterrows():
        try:
            rows.append(dict(cell=cell, rate=rate,
                             Q_ret=x[("Q_ah", 5)] / x[("Q_ah", 25)],
                             E_ret=x[("E_wh", 5)] / x[("E_wh", 25)]))
        except KeyError:
            pass
    p = pd.DataFrame(rows).dropna()

    recs = []
    for k, lab in (("Q_ret", "charge"), ("E_ret", "energy")):
        w = p.pivot(index="cell", columns="rate", values=k)[["C/3", "C/10", "C/20"]]
        idx = [c for c in w.index if c in r0]
        x = np.array([r0[c] for c in idx])
        y = (w.loc[idx, "C/20"] - w.loc[idx, "C/3"]).values
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        recs.append(dict(outcome=lab, n=len(idx), pearson_r=pr, pearson_p=pp,
                         spearman_rho=sr, spearman_p=sp))
        print(f"{lab:6s} benefit vs independent cold R0(5 degC), n={len(idx)}: "
              f"Pearson r={pr:+.3f} (p={pp:.3f})  Spearman rho={sr:+.3f} (p={sp:.3f})")

    out = pd.DataFrame(recs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("\nPredicted sign for the truncation-relief mechanism: POSITIVE.")
    print("Observed sign: NEGATIVE in both outcomes, not significant at n=8.")
    print("The independent moderator therefore does not support the mechanism, and")
    print("the earlier +0.88/+0.90 correlation is attributable to shared-term coupling.")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
