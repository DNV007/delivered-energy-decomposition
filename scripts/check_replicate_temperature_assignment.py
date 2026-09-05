#!/usr/bin/env python3
"""Can the replicate HPPC files be assigned to a checkup temperature?

The archive's processed HPPC resistance files (`data_Ri_HPPC`) carry three
cells per family but no temperature in their name or their contents, so the
manuscript had been calling their spread a room-temperature dispersion without
evidence. This script supplies the missing test.

R_1s is strongly temperature dependent, so for a family whose resistance
changes sharply between checkup setpoints the replicate range itself locates
the condition: if the range brackets one setpoint of the temperature series
and no other, the assignment is a measurement rather than an assumption. Where
cell-to-cell spread exceeds the resistance change per 10 K step the test has no
power, and this script says so instead of reporting a match.

Written for critique item A9. The comparison is between different physical
cells (the replicate files and the temperature series are separate streams), so
a family that passes shows only that the replicate condition is consistent with
that setpoint, not that either cell is typical of the other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPLICATE_SOURCE = "results/tables/table_cell_replicate_spread.csv"
SERIES_SOURCE = "results/tables/table_raw_pulse_temperature_summary.csv"
OUTPUT = "results/tables/table_replicate_temperature_assignment.csv"

FAMILY_ORDER = ["SIB", "LFP", "NMC", "LTO"]
NOMINAL_C_RATE = 1.0
DIRECTION = "discharge"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicates = pd.read_csv(project_root / REPLICATE_SOURCE)
    replicates = replicates[replicates["pulse_direction"] == DIRECTION]
    replicates = replicates.set_index("cell_family")

    series = pd.read_csv(project_root / SERIES_SOURCE)
    series = series[
        (series["pulse_direction"] == DIRECTION)
        & (series["abs_c_rate"] == NOMINAL_C_RATE)
    ]
    series = series.pivot_table(
        index="cell_family", columns="temperature_deg_c", values="median_r_1s_mohm"
    )
    return replicates, series


def assign(replicates: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        if family not in replicates.index or family not in series.index:
            continue
        replicate = replicates.loc[family]
        low = float(replicate["min_r_1s_mohm"])
        high = float(replicate["max_r_1s_mohm"])
        median = float(replicate["median_r_1s_mohm"])
        setpoints = series.loc[family]

        bracketed = [float(t) for t in setpoints.index if low <= setpoints[t] <= high]
        nearest = min(setpoints.index, key=lambda t: abs(setpoints[t] - median))

        # Distance from the replicate range to each setpoint, as a percentage of
        # that setpoint's value; zero when the setpoint falls inside the range.
        def gap_percent(temperature: float) -> float:
            value = float(setpoints[temperature])
            if low <= value <= high:
                return 0.0
            gap = low - value if value < low else value - high
            return 100.0 * gap / value

        gaps = {float(t): gap_percent(t) for t in setpoints.index}
        alternatives = sorted(t for t in gaps if t != 25.0)
        closest_alternative = min(alternatives, key=lambda t: gaps[t])

        # The assignment is decided only where exactly one setpoint is
        # compatible with the replicate range, allowing a small tolerance so a
        # family that misses by less than its own spread is not failed on a
        # rounding-scale gap.
        tolerance_percent = 100.0 * (high - low) / high
        compatible = [t for t, g in gaps.items() if g <= tolerance_percent]
        decided = len(compatible) == 1

        rows.append(
            {
                "cell_family": family,
                "n_cells": int(replicate["n_cells"]),
                "replicate_min_r_1s_mohm": low,
                "replicate_median_r_1s_mohm": median,
                "replicate_max_r_1s_mohm": high,
                "replicate_spread_percent_of_mean": float(
                    replicate["spread_percent_of_mean"]
                ),
                "series_r_1s_at_25c_mohm": float(setpoints[25.0]),
                "gap_to_25c_percent": gaps[25.0],
                "closest_alternative_setpoint_deg_c": closest_alternative,
                "gap_to_closest_alternative_percent": gaps[closest_alternative],
                "setpoints_bracketed": "|".join(f"{t:.0f}" for t in bracketed)
                or "none",
                "nearest_setpoint_to_replicate_median_deg_c": float(nearest),
                "assignment": "25 C" if (decided and compatible == [25.0]) else "undecided",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    replicates, series = load(args.project_root)
    table = assign(replicates, series)

    output = args.project_root / OUTPUT
    ensure_parent(output)
    table.to_csv(output, index=False)

    if not args.quiet:
        print("Replicate condition against the 1C-discharge temperature series:")
        for _, row in table.iterrows():
            print(
                f"  {row['cell_family']:<4} replicates "
                f"{row['replicate_min_r_1s_mohm']:6.1f}-"
                f"{row['replicate_max_r_1s_mohm']:6.1f} mOhm  |  "
                f"25 C series {row['series_r_1s_at_25c_mohm']:6.1f} "
                f"(gap {row['gap_to_25c_percent']:4.1f} %)  |  "
                f"closest alternative {row['closest_alternative_setpoint_deg_c']:.0f} C "
                f"(gap {row['gap_to_closest_alternative_percent']:5.1f} %)  ->  "
                f"{row['assignment']}"
            )
        decided = table[table["assignment"] == "25 C"]["cell_family"].tolist()
        undecided = table[table["assignment"] == "undecided"]["cell_family"].tolist()
        print(f"\nAssigned to 25 C: {', '.join(decided) or 'none'}")
        print(f"Undecided (spread exceeds the temperature step): "
              f"{', '.join(undecided) or 'none'}")
        print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
