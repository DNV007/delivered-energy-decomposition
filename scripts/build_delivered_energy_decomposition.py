#!/usr/bin/env python3
"""Decompose the low-temperature energy shortfall into window and polarisation terms.

Capacity retention and pulse resistance are dimensionally unlike quantities, so
calling one of them the larger penalty requires a common objective. The
temperature checkup supplies one directly. Its discharge (StepID 96) is a
GITT-style sequence of 105 constant-current segments at 1.5 A -- the same
absolute current for every family -- each followed by a rest whose final
voltage is the six-minute rested voltage at that depth of discharge
(StepID 97). Delivered energy and its rested-voltage counterfactual therefore
live on the same charge axis, measured on the same cell in the same file at
the same temperature, and their difference is the load-to-rest energy gap:

    E_del(T)  = sum_i dq_i * Vbar_i        delivered at 1.5 A
    E_ocv(T)  = sum_i dq_i * V_ocv,i       same window, valued at rest
    L_pol(T)  = E_ocv(T) - E_del(T)        load-to-rest energy gap

where dq_i is the segment charge throughput, V_ocv,i the rested voltage after
segment i, and Vbar_i the current-weighted mean terminal voltage during it.

L_pol is a finite-rest proxy for voltage-polarisation energy, NOT a measurement
of dissipated heat. The six-minute rest is not demonstrably equilibrium and no
calorimetry is available, so where relaxation is incomplete the gap under-reads
its equilibrium-referenced counterpart. The manuscript states this explicitly;
do not describe this term as energy dissipated inside the cell.
The shortfall against 25 degC then splits additively:

    E_del(25) - E_del(T) = [E_ocv(25) - E_ocv(T)]  +  [L_pol(T) - L_pol(25)]
                            \_ window term _/         \_ polarisation term _/

The polarisation term is the growth of that load-to-rest gap over the charge
actually delivered. The window term, however, is NOT simply energy not
delivered because the discharge ended earlier: each temperature is integrated
over its own accessible window, so the term mixes two effects with opposite
signs on the mean voltage -- the window shortens on cooling, and the rested
voltage at matched charge throughput is itself depressed. Those are different
physical quantities, so the window term is split again on the capacity window
common to both temperatures, Q_c = min(Q(T), Q(25)):

    E_ocv(25) - E_ocv(T)
        = int_{Q_c}^{Q(25)} V_ocv,25 dq - int_{Q_c}^{Q(T)} V_ocv,T dq   <- truncation
        + int_{0}^{Q_c} (V_ocv,25 - V_ocv,T) dq                         <- relaxed shift

giving the three-term split

    E_del(25) - E_del(T) = truncation + relaxed shift + polarisation.

Only one of the two truncation integrals is non-zero at any temperature: below
25 degC the reference reaches further, above it the tail changes sign. Both
integrals use the same sum_i dq_i V_ocv,i discretisation as E_ocv itself, with
the segments straddling Q_c pro-rated by the fraction of their charge inside
the window, so the identity closes to floating-point precision rather than
approximately.

The truncation term is energy the cell never delivered because the discharge
reached its cutoff earlier, valued at the reference relaxed voltage. The
relaxed-shift term is energy lost because the relaxed voltage curve itself sits
lower in the cold over the shared window; Section 3.2 of the manuscript argues
that this term is largely not thermodynamic, which is why it is reported
separately rather than folded into either neighbour. The split does not claim
the truncation term is thermodynamic in origin -- a larger polarisation reaches
the cutoff sooner, so the terms are coupled -- it quantifies where the missing
energy went, not why.

Vbar_i is computed as the ratio of two trapezoidal integrals over the same
samples, int(V*|I|dt) / int(|I|dt), so the 1 Hz sampling bias cancels; dq_i is
taken from the cycler's own coulomb count, which keeps the capacity axis
identical to the published capacity descriptors.
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
SEGMENT_OUTPUT = "results/tables/table_delivered_energy_segments.csv"
SUMMARY_OUTPUT = "results/tables/table_delivered_energy_decomposition.csv"
SHORTFALL_OUTPUT = "results/tables/table_energy_shortfall_split.csv"
COMMON_WINDOW_OUTPUT = "results/tables/table_common_window_ocv_comparison.csv"

COLUMNS = [
    "Testtime[s]",
    "StepID",
    "Voltage[V]",
    "Current[A]",
    "Temperature[°C]",
    "Capacity_Step[Ah]",
]

DISCHARGE_CURRENT_STEP = 96
DISCHARGE_REST_STEP = 97
CHARGE_CURRENT_STEP = 106
CHARGE_REST_STEP = 107

# The relaxed-voltage offset between temperatures is computed in both
# directions. If it were a shift of the equilibrium curve it would keep its
# sign; residual polarisation after a fixed rest reverses with the current.
DIRECTION_STEPS = {
    "discharge": (DISCHARGE_CURRENT_STEP, DISCHARGE_REST_STEP),
    "charge": (CHARGE_CURRENT_STEP, CHARGE_REST_STEP),
}

FAMILY_ORDER = ["SIB", "LFP", "LTO", "NMC"]
REFERENCE_TEMPERATURE_C = 25.0


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def consecutive_groups(frame: pd.DataFrame):
    if frame.empty:
        return []
    groups = frame.index.to_series().diff().ne(1).cumsum()
    return (group for _, group in frame.groupby(groups))


def segment_records(frame: pd.DataFrame, direction: str = "discharge") -> list[dict[str, float]]:
    """Per-segment charge throughput, current-weighted mean voltage, and OCV."""
    current_step, rest_step = DIRECTION_STEPS[direction]
    current_segments = list(consecutive_groups(frame[frame["StepID"] == current_step]))
    rest_segments = list(consecutive_groups(frame[frame["StepID"] == rest_step]))

    records: list[dict[str, float]] = []
    for index, segment in enumerate(current_segments):
        time_s = segment["Testtime[s]"].to_numpy(dtype=float)
        voltage = segment["Voltage[V]"].to_numpy(dtype=float)
        current = np.abs(segment["Current[A]"].to_numpy(dtype=float))
        capacity = segment["Capacity_Step[Ah]"].dropna()

        finite = np.isfinite(time_s) & np.isfinite(voltage) & np.isfinite(current)
        if finite.sum() < 2 or capacity.empty:
            continue
        time_s, voltage, current = time_s[finite], voltage[finite], current[finite]

        charge_integral = np.trapezoid(current, time_s)
        if charge_integral <= 0:
            continue
        # Ratio of integrals over identical samples: the sampling bias that
        # affects both cancels, leaving the current-weighted mean voltage.
        mean_voltage = float(np.trapezoid(voltage * current, time_s) / charge_integral)

        rest = rest_segments[index] if index < len(rest_segments) else None
        ocv_voltage = np.nan
        if rest is not None:
            rest_voltage = rest["Voltage[V]"].dropna()
            if not rest_voltage.empty:
                ocv_voltage = float(rest_voltage.iloc[-1])

        records.append(
            {
                "direction": direction,
                "segment_index": index,
                "segment_capacity_ah": abs(float(capacity.iloc[-1])),
                "mean_terminal_voltage_v": mean_voltage,
                "ocv_voltage_v": ocv_voltage,
                "mean_current_a": float(charge_integral / (time_s[-1] - time_s[0]))
                if time_s[-1] > time_s[0]
                else np.nan,
                "duration_s": float(time_s[-1] - time_s[0]),
            }
        )
    return records


def build_segments(archive: Path) -> pd.DataFrame:
    members = [
        member
        for member in iter_members(archive, {".parquet"})
        if member.startswith("raw_CU_diffT/")
    ]
    rows: list[dict[str, object]] = []
    for member in sorted(members):
        meta = normal_metadata(member)
        frame = read_parquet_member(archive, member, columns=COLUMNS)
        for direction in DIRECTION_STEPS:
            for record in segment_records(frame, direction):
                rows.append({**meta, **record})

    segments = pd.DataFrame.from_records(rows)
    sign = np.where(segments["direction"] == "charge", -1.0, 1.0)
    segments["overpotential_v"] = sign * (
        segments["ocv_voltage_v"] - segments["mean_terminal_voltage_v"]
    )
    segments["delivered_energy_wh"] = (
        segments["segment_capacity_ah"] * segments["mean_terminal_voltage_v"]
    )
    segments["ocv_energy_wh"] = segments["segment_capacity_ah"] * segments["ocv_voltage_v"]
    segments["polarisation_loss_wh"] = segments["ocv_energy_wh"] - segments["delivered_energy_wh"]
    return segments


def summarise(segments: pd.DataFrame) -> pd.DataFrame:
    usable = segments[segments["direction"] == "discharge"].dropna(
        subset=["ocv_voltage_v", "mean_terminal_voltage_v"]
    )
    summary = (
        usable.groupby(["cell_family", "cell_id", "temperature_deg_c"], dropna=False)
        .agg(
            n_segments=("segment_index", "count"),
            capacity_ah=("segment_capacity_ah", "sum"),
            delivered_energy_wh=("delivered_energy_wh", "sum"),
            ocv_energy_wh=("ocv_energy_wh", "sum"),
            polarisation_loss_wh=("polarisation_loss_wh", "sum"),
            mean_current_a=("mean_current_a", "median"),
        )
        .reset_index()
    )
    summary["mean_overpotential_v"] = summary["polarisation_loss_wh"] / summary["capacity_ah"]
    summary["polarisation_loss_percent_of_ocv"] = (
        100.0 * summary["polarisation_loss_wh"] / summary["ocv_energy_wh"]
    )
    summary["energy_efficiency"] = summary["delivered_energy_wh"] / summary["ocv_energy_wh"]
    # Effective whole-cell resistance implied by the mean overpotential at the
    # protocol current. A 46 s segment average, not a pulse resistance.
    summary["implied_resistance_mohm"] = (
        1000.0 * summary["mean_overpotential_v"] / summary["mean_current_a"]
    )

    for column in ["delivered_energy_wh", "ocv_energy_wh", "capacity_ah"]:
        summary[f"{column}_retention_vs_25c"] = np.nan
    for _, group in summary.groupby("cell_family", dropna=False):
        reference = group[group["temperature_deg_c"] == REFERENCE_TEMPERATURE_C]
        if reference.empty:
            continue
        for column in ["delivered_energy_wh", "ocv_energy_wh", "capacity_ah"]:
            denominator = float(reference[column].iloc[0])
            if denominator:
                summary.loc[group.index, f"{column}_retention_vs_25c"] = (
                    summary.loc[group.index, column] / denominator
                )

    summary["family_rank"] = summary["cell_family"].map(
        {family: index for index, family in enumerate(FAMILY_ORDER)}
    )
    return summary.sort_values(["family_rank", "temperature_deg_c"]).drop(columns="family_rank")


def ocv_energy_over_window(
    segment_frame: pd.DataFrame, q_low: float, q_high: float
) -> tuple[float, float]:
    """sum_i dq_i * V_ocv,i restricted to cumulative capacity [q_low, q_high].

    The segments straddling either bound are pro-rated by the fraction of their
    charge throughput that falls inside the window, so this reduces exactly to
    the full E_ocv when the window spans the whole discharge. Returns
    (energy_wh, charge_ah).
    """
    ordered = segment_frame.sort_values("segment_index")
    delta_q = ordered["segment_capacity_ah"].to_numpy(dtype=float)
    voltage = ordered["ocv_voltage_v"].to_numpy(dtype=float)
    upper = np.cumsum(delta_q)
    lower = upper - delta_q
    overlap = np.clip(np.minimum(upper, q_high) - np.maximum(lower, q_low), 0.0, None)
    return float(np.sum(overlap * voltage)), float(np.sum(overlap))


def split_shortfall(summary: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Additive three-term split of the delivered-energy shortfall vs 25 degC."""
    discharge = segments[segments["direction"] == "discharge"].dropna(
        subset=["ocv_voltage_v", "mean_terminal_voltage_v"]
    )
    rows: list[dict[str, object]] = []
    for family, group in summary.groupby("cell_family", dropna=False):
        reference = group[group["temperature_deg_c"] == REFERENCE_TEMPERATURE_C]
        if reference.empty:
            continue
        reference = reference.iloc[0]
        family_segments = discharge[discharge["cell_family"] == family]
        reference_segments = family_segments[
            family_segments["temperature_deg_c"] == REFERENCE_TEMPERATURE_C
        ]
        capacity_reference = float(reference["capacity_ah"])
        for _, row in group.iterrows():
            if row["temperature_deg_c"] == REFERENCE_TEMPERATURE_C:
                continue
            total = float(reference["delivered_energy_wh"] - row["delivered_energy_wh"])
            window = float(reference["ocv_energy_wh"] - row["ocv_energy_wh"])
            polarisation = float(row["polarisation_loss_wh"] - reference["polarisation_loss_wh"])

            # Split the window term on the capacity window common to both
            # temperatures. Only one of the two tails is non-zero.
            cold_segments = family_segments[
                family_segments["temperature_deg_c"] == row["temperature_deg_c"]
            ]
            capacity_cold = float(row["capacity_ah"])
            common = min(capacity_cold, capacity_reference)
            energy_reference_common, charge_common = ocv_energy_over_window(
                reference_segments, 0.0, common
            )
            energy_reference_tail, charge_reference_tail = ocv_energy_over_window(
                reference_segments, common, capacity_reference
            )
            energy_cold_common, _ = ocv_energy_over_window(cold_segments, 0.0, common)
            energy_cold_tail, charge_cold_tail = ocv_energy_over_window(
                cold_segments, common, capacity_cold
            )
            truncation = energy_reference_tail - energy_cold_tail
            relaxed_shift = energy_reference_common - energy_cold_common
            charge_truncated = charge_reference_tail - charge_cold_tail

            rows.append(
                {
                    "cell_family": family,
                    "temperature_deg_c": row["temperature_deg_c"],
                    "delivered_energy_wh": row["delivered_energy_wh"],
                    "shortfall_total_wh": total,
                    "shortfall_window_wh": window,
                    "shortfall_polarisation_wh": polarisation,
                    "shortfall_truncation_wh": truncation,
                    "shortfall_relaxed_shift_wh": relaxed_shift,
                    "shortfall_percent_of_25c": 100.0
                    * total
                    / float(reference["delivered_energy_wh"]),
                    "polarisation_share_percent": 100.0 * polarisation / total
                    if total
                    else np.nan,
                    "window_share_percent": 100.0 * window / total if total else np.nan,
                    "truncation_share_percent": 100.0 * truncation / total
                    if total
                    else np.nan,
                    "relaxed_shift_share_percent": 100.0 * relaxed_shift / total
                    if total
                    else np.nan,
                    # Upper bound on the dissipation-attributable share: the
                    # whole relaxed-voltage shift credited to polarisation.
                    "dissipation_upper_share_percent": 100.0
                    * (polarisation + relaxed_shift)
                    / total
                    if total
                    else np.nan,
                    "truncated_capacity_ah": charge_truncated,
                    "truncation_mean_voltage_v": truncation / charge_truncated
                    if charge_truncated
                    else np.nan,
                    "relaxed_shift_mean_mv": 1000.0 * relaxed_shift / charge_common
                    if charge_common
                    else np.nan,
                    "residual_wh": total - window - polarisation,
                    "window_residual_wh": window - truncation - relaxed_shift,
                }
            )
    shortfall = pd.DataFrame.from_records(rows)
    shortfall["family_rank"] = shortfall["cell_family"].map(
        {family: index for index, family in enumerate(FAMILY_ORDER)}
    )
    return shortfall.sort_values(["family_rank", "temperature_deg_c"]).drop(columns="family_rank")


def common_window_comparison(segments: pd.DataFrame) -> pd.DataFrame:
    """Relaxed-voltage comparison on a capacity window common to all temperatures.

    Integrating each temperature over its own accessible window entangles two
    effects: the window shortens on cooling, and the relaxed voltage at matched
    charge throughput changes. Restricting every temperature of a family to the
    same window -- the shortest one, which is 5 degC -- separates them.
    """
    rows: list[dict[str, object]] = []
    usable = segments.dropna(subset=["ocv_voltage_v"])
    for (family, direction), group in usable.groupby(["cell_family", "direction"], dropna=False):
        totals = group.groupby("temperature_deg_c")["segment_capacity_ah"].sum()
        common_window_ah = float(totals.min())
        curves = {}
        for temperature, sub in group.groupby("temperature_deg_c"):
            sub = sub.sort_values("segment_index")
            curves[float(temperature)] = (
                sub["segment_capacity_ah"].cumsum().to_numpy(),
                sub["ocv_voltage_v"].to_numpy(),
            )
        grid = np.linspace(0.0, common_window_ah, 200)
        reference = None
        if REFERENCE_TEMPERATURE_C in curves:
            q_ref, v_ref = curves[REFERENCE_TEMPERATURE_C]
            reference = np.interp(grid, q_ref, v_ref)
        for temperature, (q, v) in sorted(curves.items()):
            resampled = np.interp(grid, q, v)
            energy = float(np.trapezoid(resampled, grid))
            offset = resampled - reference if reference is not None else np.full_like(grid, np.nan)
            rows.append(
                {
                    "cell_family": family,
                    "direction": direction,
                    "temperature_deg_c": temperature,
                    "common_window_ah": common_window_ah,
                    "common_window_ocv_energy_wh": energy,
                    "common_window_mean_ocv_v": energy / common_window_ah,
                    "median_offset_vs_25c_mv": 1000.0 * float(np.median(offset)),
                    "max_offset_vs_25c_mv": 1000.0 * float(offset[np.argmax(np.abs(offset))]),
                }
            )
    comparison = pd.DataFrame.from_records(rows)
    comparison["common_window_energy_retention_vs_25c"] = np.nan
    for _, group in comparison.groupby(["cell_family", "direction"], dropna=False):
        reference = group[group["temperature_deg_c"] == REFERENCE_TEMPERATURE_C]
        if reference.empty:
            continue
        denominator = float(reference["common_window_ocv_energy_wh"].iloc[0])
        if denominator:
            comparison.loc[group.index, "common_window_energy_retention_vs_25c"] = (
                comparison.loc[group.index, "common_window_ocv_energy_wh"] / denominator
            )
    comparison["family_rank"] = comparison["cell_family"].map(
        {family: index for index, family in enumerate(FAMILY_ORDER)}
    )
    return comparison.sort_values(["direction", "family_rank", "temperature_deg_c"]).drop(
        columns="family_rank"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=ARCHIVE, help="path to the raw DepositOnce zip")
    parser.add_argument("--quiet", action="store_true", help="write tables without a summary")
    args = parser.parse_args()

    segments = build_segments(PROJECT_ROOT / args.archive)
    summary = summarise(segments)
    shortfall = split_shortfall(summary, segments)
    common_window = common_window_comparison(segments)

    for frame, output in [
        (segments, SEGMENT_OUTPUT),
        (summary, SUMMARY_OUTPUT),
        (shortfall, SHORTFALL_OUTPUT),
        (common_window, COMMON_WINDOW_OUTPUT),
    ]:
        path = PROJECT_ROOT / output
        ensure_parent(path)
        frame.to_csv(path, index=False)

    if not args.quiet:
        print("Delivered energy at 1.5 A, and its open-circuit counterfactual:\n")
        print(
            f"  {'family':<5}{'T':>5}{'Q (Ah)':>9}{'E_del':>9}{'E_ocv':>9}"
            f"{'L_pol':>8}{'eta (mV)':>10}{'eff':>7}"
        )
        for _, row in summary.iterrows():
            print(
                f"  {row['cell_family']:<5}{row['temperature_deg_c']:>5.0f}"
                f"{row['capacity_ah']:>9.3f}{row['delivered_energy_wh']:>9.3f}"
                f"{row['ocv_energy_wh']:>9.3f}{row['polarisation_loss_wh']:>8.3f}"
                f"{1000 * row['mean_overpotential_v']:>10.0f}"
                f"{row['energy_efficiency']:>7.3f}"
            )
        print("\n25 degC -> 5 degC shortfall in delivered energy, three-term split:\n")
        cold = shortfall[shortfall["temperature_deg_c"] == 5.0]
        print(
            f"  {'family':<5}{'total (Wh)':>12}{'trunc.':>9}{'shift':>9}{'polar.':>9}"
            f"{'trunc. %':>10}{'shift %':>9}{'polar. %':>10}{'polar+shift %':>15}"
        )
        for _, row in cold.iterrows():
            print(
                f"  {row['cell_family']:<5}{row['shortfall_total_wh']:>12.3f}"
                f"{row['shortfall_truncation_wh']:>9.3f}"
                f"{row['shortfall_relaxed_shift_wh']:>9.3f}"
                f"{row['shortfall_polarisation_wh']:>9.3f}"
                f"{row['truncation_share_percent']:>10.1f}"
                f"{row['relaxed_shift_share_percent']:>9.1f}"
                f"{row['polarisation_share_percent']:>10.1f}"
                f"{row['dissipation_upper_share_percent']:>15.1f}"
            )
        worst = float(shortfall["window_residual_wh"].abs().max())
        print(f"\n  window-term split closes to {worst:.2e} Wh")
        print("\nRelaxed voltage on a window common to all temperatures:\n")
        print(f"  {'family':<5}{'window (Ah)':>13}{'mean OCV 5 degC':>17}{'25 degC':>10}"
              f"{'median offset':>15}")
        for family in FAMILY_ORDER:
            sub = common_window[
                (common_window["cell_family"] == family)
                & (common_window["direction"] == "discharge")
            ]
            cold = sub[sub["temperature_deg_c"] == 5.0]
            warm = sub[sub["temperature_deg_c"] == REFERENCE_TEMPERATURE_C]
            if cold.empty or warm.empty:
                continue
            print(
                f"  {family:<5}{float(cold['common_window_ah'].iloc[0]):>13.3f}"
                f"{float(cold['common_window_mean_ocv_v'].iloc[0]):>17.3f}"
                f"{float(warm['common_window_mean_ocv_v'].iloc[0]):>10.3f}"
                f"{float(cold['median_offset_vs_25c_mv'].iloc[0]):>12.0f} mV"
            )
        print(f"\nWrote {SUMMARY_OUTPUT}")
        print(f"Wrote {SHORTFALL_OUTPUT}")
        print(f"Wrote {SEGMENT_OUTPUT}")
        print(f"Wrote {COMMON_WINDOW_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
