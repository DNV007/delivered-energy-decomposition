#!/usr/bin/env python3
"""Is the accessible window Q(T) set by the cutoff, or by the 105-segment budget?

The truncation term of the three-term split is the largest single component of
the sodium-ion cell's cold shortfall, and it is defined entirely by where the
discharge stopped. Two properties of that stopping condition therefore have to
be established rather than assumed.

First, the cutoff itself. The archive does not state a discharge cutoff
voltage, so it is recovered here as the lowest terminal voltage the cycler
reached on the discharge step (StepID 96). If the cutoff drifted with
temperature, the truncation term would be partly an artefact of the threshold
rather than of the cell.

Second, the segment budget. The checkup discharge is a GITT-style sequence of
105 current segments, each followed by a rest. Near the end of discharge the
cell reaches the cutoff inside a segment, the cycler ends that segment early,
and the following rest lets the relaxed voltage recover above the cutoff so
that the next segment can run. The late segments therefore shorten
progressively, and the discharge ends by exhausting the 105-segment schedule
rather than at a single termination event. That raises a fair objection: if
the cold cell is more deeply into this recovery-limited regime than the
reference at 25 degC, part of the capacity gap the truncation term prices
would be an artefact of the schedule length rather than a property of the
cell.

This script tests that objection. The per-segment charge in the recovery-
limited tail decays geometrically, so the charge an unbounded schedule would
have added is the sum of that geometric series. It is evaluated three ways --
at the geometric-mean ratio over the last ten segments, at the final observed
ratio, and at the largest observed ratio, the last two being conservative
because the ratio drifts towards unity as the tail develops. What matters is
not the absolute recovery but the difference between the cold cell and its own
25 degC reference, since the truncation term prices a gap and not a level.

Written for the round-45 recheck. It reads the raw archive for the cutoff and
the published segment table for the tail, and writes its own output, so the
released decomposition is untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metadata import normal_metadata
from src.zip_readers import iter_members, read_parquet_member

ARCHIVE = "data_raw/data_EvalSIB.zip"
SEGMENTS = "results/tables/table_delivered_energy_segments.csv"
OUTPUT = "results/tables/table_cutoff_step_budget.csv"

COLUMNS = ["Testtime[s]", "StepID", "Voltage[V]", "Current[A]"]
DISCHARGE_CURRENT_STEP = 96

REFERENCE_TEMPERATURE_C = 25.0
COLD_TEMPERATURE_C = 5.0
FAMILY_ORDER = ["SIB", "LFP", "NMC", "LTO"]

# A segment delivering less than half the nominal increment has been cut short
# by the cutoff rather than by its own charge target.
SHORT_SEGMENT_FRACTION = 0.5
# The nominal increment is taken from the first 40 segments, well clear of the
# tail at either temperature.
NOMINAL_SEGMENT_WINDOW = 40
# The decay is fitted over the final segments of the cut-short tail. The
# successive ratio is not constant across the whole tail -- it climbs from
# about 0.43 to about 0.96 as the recovery-limited regime develops -- so what
# governs the sum of the unobserved remainder is the terminal ratio, not the
# average over the tail. Fitting the last few segments is therefore both the
# correct window and the conservative one: it returns the slowest decay and so
# the largest recovery. TAIL_WINDOW_SWEEP records how little that choice
# matters to the conclusion.
TAIL_SEGMENTS = 4
TAIL_WINDOW_SWEEP = (4, 6, 8, 10, 12)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def effective_cutoff(archive: Path) -> pd.DataFrame:
    """Lowest terminal voltage reached on the discharge step, per file."""
    members = [
        member
        for member in iter_members(archive, {".parquet"})
        if member.startswith("raw_CU_diffT/")
    ]
    rows: list[dict[str, object]] = []
    for member in sorted(members):
        meta = normal_metadata(member)
        frame = read_parquet_member(archive, member, columns=COLUMNS)
        voltage = frame.loc[
            frame["StepID"] == DISCHARGE_CURRENT_STEP, "Voltage[V]"
        ].dropna()
        if voltage.empty:
            continue
        rows.append(
            {
                "cell_family": meta["cell_family"],
                "temperature_deg_c": float(meta["temperature_deg_c"]),
                "cutoff_voltage_v": float(voltage.min()),
            }
        )
    return pd.DataFrame.from_records(rows)


def geometric_fit(tail: np.ndarray) -> dict[str, float]:
    """Straight-line fit of log charge against segment index, and its quality.

    A geometric decay is a straight line in these coordinates, so the fit
    quality is the evidence for the assumption rather than a restatement of it.
    """
    index = np.arange(tail.size, dtype=float)
    log_charge = np.log(tail)
    slope, intercept = np.polyfit(index, log_charge, 1)
    residual = log_charge - (slope * index + intercept)
    return {
        "ratio": float(np.exp(slope)),
        "r_squared": float(1.0 - residual.var() / log_charge.var())
        if log_charge.var() > 0
        else np.nan,
        "rms_percent": float(100.0 * np.sqrt(np.mean(np.expm1(residual) ** 2))),
    }


def tail_recovery(capacity_ah: np.ndarray) -> dict[str, float]:
    """Charge an unbounded schedule would have added, from the tail decay."""
    nominal = float(np.median(capacity_ah[:NOMINAL_SEGMENT_WINDOW]))
    short = capacity_ah < SHORT_SEGMENT_FRACTION * nominal
    last = float(capacity_ah[-1])

    tail = capacity_ah[-TAIL_SEGMENTS:]
    ratios = tail[1:] / tail[:-1]

    recovery: dict[str, float] = {"tail_segments_fitted": int(tail.size)}
    for name, ratio in (
        ("geometric", float(np.exp(np.mean(np.log(ratios))))),
        ("final", float(ratios[-1])),
        ("maximum", float(ratios.max())),
    ):
        recovery[f"ratio_{name}"] = ratio
        # sum_{k>=1} last * ratio^k, divergent only if the tail stopped decaying
        recovery[f"recovery_{name}_ah"] = (
            last * ratio / (1.0 - ratio) if ratio < 1.0 else np.inf
        )

    fit = geometric_fit(tail)
    recovery["tail_ratio_from_log_fit"] = fit["ratio"]
    recovery["tail_log_fit_r_squared"] = fit["r_squared"]
    recovery["tail_log_fit_rms_percent"] = fit["rms_percent"]
    recovery["tail_ratio_spread"] = float(ratios.max() - ratios.min())
    recovery["tail_ratio_first_observed"] = float(
        capacity_ah[-(int(short.sum())) + 1] / capacity_ah[-int(short.sum())]
    ) if short.sum() >= 2 else np.nan

    # Sensitivity of the extrapolation to how much of the tail is fitted.
    for window in TAIL_WINDOW_SWEEP:
        if window > capacity_ah.size:
            continue
        swept = geometric_fit(capacity_ah[-window:])
        recovery[f"recovery_window{window}_ah"] = (
            last * swept["ratio"] / (1.0 - swept["ratio"])
            if swept["ratio"] < 1.0
            else np.inf
        )
        recovery[f"ratio_window{window}"] = swept["ratio"]
        recovery[f"r_squared_window{window}"] = swept["r_squared"]
    return {
        "accessible_capacity_ah": float(capacity_ah.sum()),
        "nominal_segment_ah": nominal,
        "cutoff_limited_segments": int(short.sum()),
        "cutoff_limited_charge_ah": float(capacity_ah[short].sum()),
        "final_segment_ah": last,
        **recovery,
    }


def build(archive: Path, segments_path: Path) -> pd.DataFrame:
    cutoffs = effective_cutoff(archive)
    segments = pd.read_csv(segments_path)
    segments = segments[segments["direction"] == "discharge"]

    rows: list[dict[str, object]] = []
    for (family, temperature), group in segments.groupby(
        ["cell_family", "temperature_deg_c"], sort=False
    ):
        group = group.sort_values("segment_index")
        capacity = group["segment_capacity_ah"].to_numpy(dtype=float)
        rows.append(
            {
                "cell_family": family,
                "temperature_deg_c": float(temperature),
                "final_rested_voltage_v": float(group["ocv_voltage_v"].iloc[-1]),
                **tail_recovery(capacity),
            }
        )

    table = pd.DataFrame.from_records(rows).merge(
        cutoffs, on=["cell_family", "temperature_deg_c"], how="left"
    )
    order = {family: index for index, family in enumerate(FAMILY_ORDER)}
    return table.sort_values(
        ["cell_family", "temperature_deg_c"],
        key=lambda column: column.map(order) if column.name == "cell_family" else column,
    ).reset_index(drop=True)


def gap_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Does an unbounded schedule close the 5-to-25 degC capacity gap?"""
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        block = table[table["cell_family"] == family].set_index("temperature_deg_c")
        if COLD_TEMPERATURE_C not in block.index or REFERENCE_TEMPERATURE_C not in block.index:
            continue
        cold = block.loc[COLD_TEMPERATURE_C]
        warm = block.loc[REFERENCE_TEMPERATURE_C]
        measured = float(warm["accessible_capacity_ah"] - cold["accessible_capacity_ah"])
        row: dict[str, object] = {"cell_family": family, "measured_gap_ah": measured}
        for name in ("geometric", "final", "maximum") + tuple(
            f"window{window}" for window in TAIL_WINDOW_SWEEP
        ):
            if f"recovery_{name}_ah" not in cold.index:
                continue
            extrapolated = measured - (
                float(cold[f"recovery_{name}_ah"]) - float(warm[f"recovery_{name}_ah"])
            )
            row[f"extrapolated_gap_{name}_ah"] = extrapolated
            row[f"gap_change_{name}_percent"] = 100.0 * (extrapolated - measured) / measured
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=ARCHIVE)
    parser.add_argument("--segments", default=SEGMENTS)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    table = build(PROJECT_ROOT / args.archive, PROJECT_ROOT / args.segments)
    output = PROJECT_ROOT / args.output
    ensure_parent(output)
    table.to_csv(output, index=False)

    gaps = gap_comparison(table)

    if not args.quiet:
        print("Effective discharge cutoff, recovered as the lowest terminal voltage")
        print("reached on StepID 96 (the archive does not state one):\n")
        for family in FAMILY_ORDER:
            block = table[table["cell_family"] == family]
            if block.empty:
                continue
            cutoff = block["cutoff_voltage_v"]
            print(
                f"  {family:4s}  {cutoff.min():.3f}-{cutoff.max():.3f} V across the five "
                f"temperatures  (spread {1000 * (cutoff.max() - cutoff.min()):.0f} mV)"
            )

        print("\nCutoff-limited tail of the 105-segment schedule:\n")
        for family in FAMILY_ORDER:
            for temperature in (COLD_TEMPERATURE_C, REFERENCE_TEMPERATURE_C):
                block = table[
                    (table["cell_family"] == family)
                    & (table["temperature_deg_c"] == temperature)
                ]
                if block.empty:
                    continue
                row = block.iloc[0]
                print(
                    f"  {family:4s} {temperature:4.0f} degC: "
                    f"{int(row['cutoff_limited_segments']):2d} cut-short segments carrying "
                    f"{1000 * row['cutoff_limited_charge_ah']:5.1f} mAh of "
                    f"{1000 * row['accessible_capacity_ah']:6.1f} mAh; "
                    f"tail ratio {row['ratio_geometric']:.3f}, unbounded-schedule "
                    f"recovery {1000 * row['recovery_geometric_ah']:5.1f} mAh"
                )

        print("\nDoes the tail actually decay geometrically?\n")
        for family in FAMILY_ORDER:
            for temperature in (COLD_TEMPERATURE_C, REFERENCE_TEMPERATURE_C):
                block = table[
                    (table["cell_family"] == family)
                    & (table["temperature_deg_c"] == temperature)
                ]
                if block.empty:
                    continue
                row = block.iloc[0]
                print(
                    f"  {family:4s} {temperature:4.0f} degC: "
                    f"{int(row['tail_segments_fitted']):2d} segments fitted, "
                    f"ratio {row['tail_ratio_from_log_fit']:.3f} "
                    f"(ratio climbs {row['tail_ratio_first_observed']:.2f} -> "
                    f"{row['ratio_final']:.2f} across the tail), "
                    f"R2 {row['tail_log_fit_r_squared']:.4f}, "
                    f"residual {row['tail_log_fit_rms_percent']:.1f} % per segment"
                )

        print("\nWould an unbounded schedule close the 5-to-25 degC capacity gap?\n")
        for _, row in gaps.iterrows():
            print(
                f"  {row['cell_family']:4s}: measured {1000 * row['measured_gap_ah']:6.1f} mAh"
                f"  ->  {1000 * row['extrapolated_gap_geometric_ah']:6.1f} mAh"
                f"  ({row['gap_change_geometric_percent']:+.1f} %"
                f", {row['gap_change_maximum_percent']:+.1f} % at the largest observed ratio)"
            )
        print("\nSensitivity of that change to the number of tail segments fitted:\n")
        for _, row in gaps.iterrows():
            spans = ", ".join(
                f"{window}: {row[f'gap_change_window{window}_percent']:+.1f} %"
                for window in TAIL_WINDOW_SWEEP
                if f"gap_change_window{window}_percent" in row.index
            )
            print(f"  {row['cell_family']:4s}: {spans}")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
