#!/usr/bin/env python3
"""Cell-to-cell spread and pulse-current dependence from the archive HPPC files.

Two questions that the temperature-resolved raw traces cannot answer, because
they cover one cell per family, are answered by the archive's processed HPPC
resistance files (`data_Ri_HPPC`), which carry three cells per family at a
single checkup index:

1. **Cell-to-cell spread.** How much does R_1s vary between nominally identical
   cells of the same family? This is the only replicate-level dispersion the
   archive supports, and it bounds how much of a family-to-family difference
   can be attributed to chemistry rather than to which cell was drawn.

2. **Pulse-current dependence.** R_t = |dV_t| / |I| is a secant resistance, so
   it falls as the operating current rises. The HPPC blocks measure 1C, 2C and
   3C at every SOC, which makes the size of that effect a measurement rather
   than an assumption.

The files carry no temperature in their name or their contents, so both
results are stated at the archive's checkup-00 ambient condition and are not
used for any temperature-resolved claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE = "data_processed/hppc/hppc_resistance_long.csv"
SPREAD_OUTPUT = "results/tables/table_cell_replicate_spread.csv"
RATE_OUTPUT = "results/tables/table_pulse_rate_dependence.csv"

FAMILY_ORDER = ["SIB", "LFP", "LTO", "NMC"]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load() -> pd.DataFrame:
    frame = pd.read_csv(PROJECT_ROOT / SOURCE)
    frame["r_1s_mohm"] = frame["r_1s_ohm"] * 1000.0
    frame["c_rate_magnitude"] = frame["c_rate"].abs()
    return frame


def cell_replicate_spread(frame: pd.DataFrame) -> pd.DataFrame:
    """Median 1C R_1s per cell, and the spread across cells of each family."""
    at_1c = frame[np.isclose(frame["c_rate_magnitude"], 1.0)]

    per_cell = (
        at_1c.groupby(["cell_family", "cell_id", "pulse_direction"])["r_1s_mohm"]
        .median()
        .reset_index()
        .rename(columns={"r_1s_mohm": "median_r_1s_mohm"})
    )

    rows = []
    for (family, direction), group in per_cell.groupby(["cell_family", "pulse_direction"]):
        values = group["median_r_1s_mohm"].to_numpy()
        mean = float(values.mean())
        rows.append(
            {
                "cell_family": family,
                "pulse_direction": direction,
                "n_cells": int(len(values)),
                "min_r_1s_mohm": float(values.min()),
                "median_r_1s_mohm": float(np.median(values)),
                "max_r_1s_mohm": float(values.max()),
                "mean_r_1s_mohm": mean,
                # Range as a percentage of the family mean: a three-cell sample
                # does not support a standard deviation worth quoting.
                "spread_percent_of_mean": float(100.0 * (values.max() - values.min()) / mean),
                # Sample CV, reported only because the robustness scenario of
                # Section S3 needs a dispersion parameter. From three cells it
                # is an order-of-magnitude figure, not an estimate.
                "cv_percent": float(100.0 * values.std(ddof=1) / mean),
            }
        )

    summary = pd.DataFrame(rows)
    summary["family_rank"] = summary["cell_family"].map(
        {family: index for index, family in enumerate(FAMILY_ORDER)}
    )
    summary = summary.sort_values(["pulse_direction", "family_rank"]).drop(columns="family_rank")

    per_cell = per_cell.merge(
        summary[["cell_family", "pulse_direction", "mean_r_1s_mohm"]],
        on=["cell_family", "pulse_direction"],
        how="left",
    )
    per_cell["deviation_percent_of_family_mean"] = 100.0 * (
        per_cell["median_r_1s_mohm"] / per_cell["mean_r_1s_mohm"] - 1.0
    )
    return summary, per_cell


def rate_dependence(frame: pd.DataFrame) -> pd.DataFrame:
    """Median R_1s at 1C, 2C and 3C, per family and pulse direction."""
    per_cell_rate = (
        frame.groupby(["cell_family", "cell_id", "pulse_direction", "c_rate_magnitude"])[
            "r_1s_mohm"
        ]
        .median()
        .reset_index()
    )

    wide = per_cell_rate.pivot_table(
        index=["cell_family", "cell_id", "pulse_direction"],
        columns="c_rate_magnitude",
        values="r_1s_mohm",
    ).rename(columns={1.0: "r_1s_1c_mohm", 2.0: "r_1s_2c_mohm", 3.0: "r_1s_3c_mohm"})
    wide["ratio_3c_over_1c"] = wide["r_1s_3c_mohm"] / wide["r_1s_1c_mohm"]
    wide = wide.reset_index()

    family = (
        wide.groupby(["cell_family", "pulse_direction"])[
            ["r_1s_1c_mohm", "r_1s_2c_mohm", "r_1s_3c_mohm", "ratio_3c_over_1c"]
        ]
        .median()
        .reset_index()
    )
    family["n_cells"] = (
        wide.groupby(["cell_family", "pulse_direction"])["cell_id"].nunique().to_numpy()
    )
    # Percentage fall in the secant resistance per unit increase in C-rate,
    # linearised over the measured 1C-3C span.
    family["secant_fall_percent_per_c"] = 100.0 * (1.0 - family["ratio_3c_over_1c"]) / 2.0
    family["family_rank"] = family["cell_family"].map(
        {name: index for index, name in enumerate(FAMILY_ORDER)}
    )
    family = family.sort_values(["pulse_direction", "family_rank"]).drop(columns="family_rank")
    return family, wide


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet", action="store_true", help="write the tables without printing a summary"
    )
    args = parser.parse_args()

    frame = load()
    spread, per_cell = cell_replicate_spread(frame)
    rate, per_cell_rate = rate_dependence(frame)

    spread_path = PROJECT_ROOT / SPREAD_OUTPUT
    rate_path = PROJECT_ROOT / RATE_OUTPUT
    ensure_parent(spread_path)
    ensure_parent(rate_path)

    spread.to_csv(spread_path, index=False)
    per_cell.to_csv(spread_path.with_name("table_cell_replicate_per_cell.csv"), index=False)
    rate.to_csv(rate_path, index=False)
    per_cell_rate.to_csv(rate_path.with_name("table_pulse_rate_per_cell.csv"), index=False)

    if not args.quiet:
        print("Cell-to-cell spread of median 1C R_1s (discharge):")
        discharge = spread[spread["pulse_direction"] == "discharge"]
        for _, row in discharge.iterrows():
            print(
                f"  {row['cell_family']:<4} n={row['n_cells']}  "
                f"{row['min_r_1s_mohm']:6.1f} - {row['max_r_1s_mohm']:6.1f} mOhm   "
                f"spread {row['spread_percent_of_mean']:5.1f} % of mean"
            )
        print("\nPulse-current dependence of median R_1s (discharge):")
        discharge_rate = rate[rate["pulse_direction"] == "discharge"]
        for _, row in discharge_rate.iterrows():
            print(
                f"  {row['cell_family']:<4} 1C={row['r_1s_1c_mohm']:6.1f}  "
                f"2C={row['r_1s_2c_mohm']:6.1f}  3C={row['r_1s_3c_mohm']:6.1f} mOhm   "
                f"3C/1C={row['ratio_3c_over_1c']:.3f}"
            )
        print(f"\nWrote {SPREAD_OUTPUT}")
        print(f"Wrote {RATE_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
