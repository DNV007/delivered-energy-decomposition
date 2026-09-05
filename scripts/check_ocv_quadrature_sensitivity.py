#!/usr/bin/env python3
"""Does the rested-voltage quadrature bias the three-term energy split?

`build_delivered_energy_decomposition.py` evaluates the open-circuit
counterfactual as E_ocv = sum_i dq_i * V_ocv,i, where V_ocv,i is the relaxed
voltage measured after segment i. That is a right-endpoint rule on a
monotonically falling discharge curve, so it under-reads E_ocv, while E_del
uses the current-weighted mean voltage within the segment and is unbiased.
The two integrals of Eq. 3 therefore do not share a quadrature, and L_pol
inherits the difference.

This script rebuilds the whole decomposition under three conventions -- the
published right-endpoint rule, the midpoint (trapezoidal) rule, and the
left-endpoint rule as the opposite extreme -- and reports how far the reported
shares move. The leading rested voltage, which has no measured predecessor, is
linearly extrapolated from the first two rested points; it carries one segment
of about 105, so the choice does not affect the result.

Written for critique item A3. It reads the published segment table and writes
its own output, so the released decomposition is untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE = "results/tables/table_delivered_energy_segments.csv"
OUTPUT = "results/tables/table_ocv_quadrature_sensitivity.csv"

REFERENCE_TEMPERATURE_C = 25.0
COLD_TEMPERATURE_C = 5.0
FAMILY_ORDER = ["SIB", "LFP", "NMC", "LTO"]
SCHEMES = ["right", "mid", "left"]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def apply_scheme(segments: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Re-evaluate every segment's counterfactual voltage under one rule."""
    reworked: list[pd.DataFrame] = []
    for _, group in segments.groupby(["cell_family", "temperature_deg_c"], sort=False):
        group = group.sort_values("segment_index").copy()
        voltage = group["ocv_voltage_v"].to_numpy(dtype=float)
        if scheme == "right":
            quadrature = voltage
        else:
            leading = voltage[0] + (voltage[0] - voltage[1]) if voltage.size > 1 else voltage[0]
            previous = np.concatenate(([leading], voltage[:-1]))
            quadrature = 0.5 * (previous + voltage) if scheme == "mid" else previous
        group["ocv_voltage_v"] = quadrature
        reworked.append(group)

    out = pd.concat(reworked, ignore_index=True)
    out["ocv_energy_wh"] = out["segment_capacity_ah"] * out["ocv_voltage_v"]
    out["polarisation_loss_wh"] = out["ocv_energy_wh"] - out["delivered_energy_wh"]
    return out


def summarise(segments: pd.DataFrame) -> pd.DataFrame:
    return segments.groupby(["cell_family", "temperature_deg_c"], as_index=False).agg(
        capacity_ah=("segment_capacity_ah", "sum"),
        delivered_energy_wh=("delivered_energy_wh", "sum"),
        ocv_energy_wh=("ocv_energy_wh", "sum"),
        polarisation_loss_wh=("polarisation_loss_wh", "sum"),
    )


def ocv_energy_over_window(
    segment_frame: pd.DataFrame, q_low: float, q_high: float
) -> float:
    """sum_i dq_i * V_ocv,i restricted to cumulative capacity [q_low, q_high]."""
    ordered = segment_frame.sort_values("segment_index")
    delta_q = ordered["segment_capacity_ah"].to_numpy(dtype=float)
    voltage = ordered["ocv_voltage_v"].to_numpy(dtype=float)
    upper = np.cumsum(delta_q)
    lower = upper - delta_q
    overlap = np.clip(np.minimum(upper, q_high) - np.maximum(lower, q_low), 0.0, None)
    return float(np.sum(overlap * voltage))


def split(segments: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Three-term split of the 5 degC shortfall, mirroring Eqs. 5 and 6."""
    summary = summarise(segments)
    rows: list[dict[str, object]] = []
    for family, group in summary.groupby("cell_family"):
        reference = group[group["temperature_deg_c"] == REFERENCE_TEMPERATURE_C]
        cold = group[group["temperature_deg_c"] == COLD_TEMPERATURE_C]
        if reference.empty or cold.empty:
            continue
        reference, cold = reference.iloc[0], cold.iloc[0]

        family_segments = segments[segments["cell_family"] == family]
        reference_segments = family_segments[
            family_segments["temperature_deg_c"] == REFERENCE_TEMPERATURE_C
        ]
        cold_segments = family_segments[
            family_segments["temperature_deg_c"] == COLD_TEMPERATURE_C
        ]

        capacity_reference = float(reference["capacity_ah"])
        capacity_cold = float(cold["capacity_ah"])
        common = min(capacity_cold, capacity_reference)

        total = float(reference["delivered_energy_wh"] - cold["delivered_energy_wh"])
        polarisation = float(
            cold["polarisation_loss_wh"] - reference["polarisation_loss_wh"]
        )
        truncation = ocv_energy_over_window(
            reference_segments, common, capacity_reference
        ) - ocv_energy_over_window(cold_segments, common, capacity_cold)
        relaxed_shift = ocv_energy_over_window(
            reference_segments, 0.0, common
        ) - ocv_energy_over_window(cold_segments, 0.0, common)

        rows.append(
            {
                "quadrature": scheme,
                "cell_family": family,
                "shortfall_total_wh": total,
                "truncation_share_percent": 100.0 * truncation / total,
                "relaxed_shift_share_percent": 100.0 * relaxed_shift / total,
                "polarisation_share_percent": 100.0 * polarisation / total,
                "dissipation_upper_share_percent": 100.0
                * (polarisation + relaxed_shift)
                / total,
                "polarisation_loss_5c_wh": float(cold["polarisation_loss_wh"]),
                "polarisation_loss_25c_wh": float(reference["polarisation_loss_wh"]),
                "mean_overpotential_5c_mv": 1000.0
                * float(cold["polarisation_loss_wh"])
                / float(cold["capacity_ah"]),
                "energy_efficiency_5c": float(cold["delivered_energy_wh"])
                / float(cold["ocv_energy_wh"]),
                "identity_closure_wh": total - (truncation + relaxed_shift + polarisation),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    segments = pd.read_csv(args.project_root / SOURCE)
    segments = segments[segments["direction"] == "discharge"].dropna(
        subset=["ocv_voltage_v", "mean_terminal_voltage_v"]
    )

    table = pd.concat(
        [split(apply_scheme(segments, scheme), scheme) for scheme in SCHEMES],
        ignore_index=True,
    )
    table["family_rank"] = table["cell_family"].map(
        {family: index for index, family in enumerate(FAMILY_ORDER)}
    )
    table = table.sort_values(["family_rank", "quadrature"]).drop(columns="family_rank")

    output = args.project_root / OUTPUT
    ensure_parent(output)
    table.to_csv(output, index=False)

    if not args.quiet:
        print("Three-term split of the 5 degC shortfall under three quadrature rules")
        print("(right = published; identity closes to floating-point precision in all):")
        for family in FAMILY_ORDER:
            block = table[table["cell_family"] == family]
            if block.empty:
                continue
            print(f"\n  {family}  (shortfall {block.iloc[0]['shortfall_total_wh']:.4f} Wh)")
            for _, row in block.iterrows():
                print(
                    f"    {row['quadrature']:>5}: truncation "
                    f"{row['truncation_share_percent']:5.1f} %  shift "
                    f"{row['relaxed_shift_share_percent']:5.1f} %  polarisation "
                    f"{row['polarisation_share_percent']:5.1f} %  |  dissipation "
                    f"upper bound {row['dissipation_upper_share_percent']:5.1f} %  |  "
                    f"closure {row['identity_closure_wh']:.1e} Wh"
                )
        span = (
            table.groupby("cell_family")["truncation_share_percent"].max()
            - table.groupby("cell_family")["truncation_share_percent"].min()
        )
        print(
            f"\nWidest movement of the truncation share across all three rules: "
            f"{span.max():.1f} percentage points ({span.idxmax()})"
        )
        print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
