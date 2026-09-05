#!/usr/bin/env python3
"""Score the benchmark relaxation dataset against the Eq. 2 measurement geometry.

Section S1 of the supplement claims that this archive resolves the rested voltage
across throughput at one temperature (Protocol 1) and spans fifteen temperatures
at one depth of discharge (Protocol 2), and so satisfies neither the paired-condition
nor the common-window requirement. Those are claims about someone else's data, so
they are re-derived here from the released files rather than from the readme.

Requires scipy built against the installed numpy (the project venv).
Run scripts/fetch_relaxation_benchmark.py first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_external/relaxation_benchmark"
OUT = ROOT / "results/tables/table_relaxation_geometry.csv"

# Released column order, per the deposit readme; there is no current column and
# no coulomb count, which is itself part of the finding.
COLUMNS = ["voltage_v", "time_s", "previous_c_rate", "dod_percent", "temperature_deg_c"]


def summarise(path: Path) -> list[dict]:
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    rows = []
    for direction in ("cha", "dis"):
        segments = [np.asarray(s) for s in np.asarray(data[direction]).ravel()]
        widths = {s.shape[1] for s in segments if s.ndim == 2}
        rows.append(dict(
            file=path.name,
            direction=direction,
            segments=len(segments),
            n_columns=sorted(widths)[0] if len(widths) == 1 else -1,
            n_temperatures=len({round(float(s[0, 4]), 1) for s in segments}),
            n_dod=len({round(float(s[0, 3]), 1) for s in segments}),
            temperatures=";".join(f"{t:g}" for t in
                                  sorted({round(float(s[0, 4]), 1) for s in segments})),
            dod=";".join(f"{d:g}" for d in
                         sorted({round(float(s[0, 3]), 1) for s in segments})),
        ))
    return rows


def main() -> int:
    files = sorted(DATA.rglob("*.mat"))
    if not files:
        print(f"no .mat files under {DATA}; run scripts/fetch_relaxation_benchmark.py",
              file=sys.stderr)
        return 1

    rows = [r for f in files for r in summarise(f)]
    import pandas as pd
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(frame[["file", "direction", "segments", "n_columns",
                 "n_temperatures", "n_dod"]].to_string(index=False))

    p1 = frame[frame.file.str.contains("protocol_1")]
    p2 = frame[frame.file.str.contains("protocol_2")]
    failures = 0

    def claim(label: str, ok: bool) -> None:
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
        failures += 0 if ok else 1

    print("\nclaims made in Section S1:")
    claim("released columns carry no current and no coulomb count",
          set(frame.n_columns) == {len(COLUMNS)})
    claim("Protocol 1 is a single temperature", set(p1.n_temperatures) == {1})
    claim("Protocol 1 resolves several depths of discharge", set(p1.n_dod) == {10})
    claim("Protocol 2 spans fifteen temperatures", set(p2.n_temperatures) == {15})
    claim("Protocol 2 sits at a single depth of discharge", set(p2.n_dod) == {1})

    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if failures:
        print(f"{failures} claim(s) no longer hold against the released files.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
