#!/usr/bin/env python3
"""How far is the six-minute rested voltage from equilibrium?

The three-term decomposition references its counterfactual to the voltage
measured after a six-minute rest. That rest is not demonstrably equilibrium at
5 degC, which is why the load-to-rest term is reported as a lower bound and why
the manuscript offers the deliberately conservative allocation that assigns the
whole common-window shift to residual polarisation.

The archive samples voltage throughout each rest at about 1 Hz (StepID 97), so
the size of the residual relaxation can be measured rather than bounded by
assertion. Each rest transient is fitted and extrapolated to t -> infinity under
two tail models that bracket the plausible behaviour:

  * a two-exponential relaxation, which describes the measured 360 s well and
    extrapolates conservatively;
  * a diffusive t^(-1/2) tail, which is the slowest physically motivated decay
    and therefore the most aggressive extrapolation.

What matters for the common-window shift is not the absolute under-read but the
DIFFERENCE between temperatures, since a shift is inflated only to the extent
that the cold rest is less complete than the warm one.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.zip_readers import read_parquet_member

ARCHIVE = ROOT / "data_raw/data_EvalSIB.zip"
OUT = ROOT / "results/tables/table_rest_extrapolation.csv"
CELLS = {"SIB": "TC23SIB09", "LFP": "TC23LFP09", "NMC": "TC23NMC09", "LTO": "TC23LTO09"}
REST_STEP, SKIP_S = 97, 2.0


def biexp(t, V, A, t1, B, t2):
    return V - A * np.exp(-t / t1) - B * np.exp(-t / t2)


def sqrt_tail(t, V, A, c):
    return V - A / np.sqrt(t + c)


def per_file(member):
    d = read_parquet_member(ARCHIVE, member)
    d.columns = [c.split("[")[0] for c in d.columns]
    d = d.sort_values("Testtime")
    r = d[d.StepID == REST_STEP].copy()
    r["rest"] = (r.Testtime.diff() > 30).cumsum()
    rows = []
    for k in sorted(r.rest.unique()):
        o = r[r.rest == k]
        t = (o.Testtime - o.Testtime.min()).values
        v = o.Voltage.values
        m = t >= SKIP_S
        t, v = t[m], v[m]
        if len(t) < 50:
            continue
        V360 = float(v[-1])
        try:
            b = float(curve_fit(biexp, t, v, p0=[V360 + 5e-3, .02, 5, .02, 120], maxfev=40000)[0][0])
        except Exception:
            b = np.nan
        try:
            s = float(curve_fit(sqrt_tail, t, v, p0=[V360 + 5e-3, .05, 1], maxfev=40000)[0][0])
        except Exception:
            s = np.nan
        rows.append(dict(rest=int(k), V_360s=V360,
                         under_read_biexp_mv=1000 * (b - V360),
                         under_read_sqrt_mv=1000 * (s - V360)))
    return pd.DataFrame(rows)


def main() -> int:
    recs = []
    for fam, cell in CELLS.items():
        for T, tag in ((5, "05deg"), (25, "25deg")):
            df = per_file(f"raw_CU_diffT/{cell}/{cell}_{tag}.parquet")
            recs.append(dict(family=fam, temperature_deg_c=T, n_rests=len(df),
                             under_read_biexp_mv=df.under_read_biexp_mv.median(),
                             under_read_sqrt_mv=df.under_read_sqrt_mv.median()))
    out = pd.DataFrame(recs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("Median residual under-read of the six-minute rested voltage (mV):\n")
    print(out.round(2).to_string(index=False))
    print("\nDifferential under-read, 5 degC minus 25 degC (mV) -- this is what")
    print("inflates a matched-throughput offset:")
    for fam in CELLS:
        a = out[(out.family == fam) & (out.temperature_deg_c == 5)].iloc[0]
        b = out[(out.family == fam) & (out.temperature_deg_c == 25)].iloc[0]
        print(f"   {fam:4s} biexp {a.under_read_biexp_mv-b.under_read_biexp_mv:+6.2f}"
              f"   sqrt-t {a.under_read_sqrt_mv-b.under_read_sqrt_mv:+6.2f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
