#!/usr/bin/env python3
"""Build integrated descriptor matrices and modeling targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LONG_COLUMNS = [
    "descriptor_id",
    "source_table",
    "measurement_type",
    "aggregation_level",
    "descriptor_name",
    "descriptor_value",
    "descriptor_unit",
    "cell_id",
    "cell_family",
    "chemistry_type",
    "temperature_deg_c",
    "actual_temperature_deg_c",
    "direction",
    "pulse_direction",
    "abs_c_rate",
    "nominal_soc_percent",
    "voltage_v",
    "voltage_window_v",
    "source_member_path",
    "source_notes",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def value(row: pd.Series, column: str, default: object = np.nan) -> object:
    return row[column] if column in row and pd.notna(row[column]) else default


def add_descriptor(
    records: list[dict[str, object]],
    row: pd.Series,
    source_table: str,
    measurement_type: str,
    aggregation_level: str,
    descriptor_name: str,
    descriptor_value: object,
    descriptor_unit: str,
    *,
    actual_temperature_column: str | None = None,
    direction_column: str | None = None,
    pulse_direction_column: str | None = None,
    abs_c_rate_column: str | None = None,
    nominal_soc_column: str | None = None,
    voltage_column: str | None = "voltage_v",
    voltage_window_v: object = np.nan,
    source_notes: str = "",
) -> None:
    if pd.isna(descriptor_value):
        return
    records.append(
        {
            "descriptor_id": "",
            "source_table": source_table,
            "measurement_type": measurement_type,
            "aggregation_level": aggregation_level,
            "descriptor_name": descriptor_name,
            "descriptor_value": float(descriptor_value),
            "descriptor_unit": descriptor_unit,
            "cell_id": value(row, "cell_id", ""),
            "cell_family": value(row, "cell_family", ""),
            "chemistry_type": value(row, "chemistry_type", ""),
            "temperature_deg_c": value(row, "temperature_deg_c"),
            "actual_temperature_deg_c": value(row, actual_temperature_column) if actual_temperature_column else np.nan,
            "direction": value(row, direction_column, "") if direction_column else "",
            "pulse_direction": value(row, pulse_direction_column, "") if pulse_direction_column else "",
            "abs_c_rate": value(row, abs_c_rate_column) if abs_c_rate_column else np.nan,
            "nominal_soc_percent": value(row, nominal_soc_column) if nominal_soc_column else np.nan,
            "voltage_v": value(row, voltage_column) if voltage_column else np.nan,
            "voltage_window_v": voltage_window_v,
            "source_member_path": value(row, "member_path", ""),
            "source_notes": source_notes,
        }
    )


def append_capacity(records: list[dict[str, object]], frame: pd.DataFrame) -> None:
    descriptors = [
        ("discharge_capacity_ah", "discharge_capacity", "Ah"),
        ("charge_capacity_ah", "charge_capacity", "Ah"),
        ("discharge_capacity_retention_vs_25c", "discharge_capacity_retention_vs_25c", "fraction"),
        ("charge_capacity_retention_vs_25c", "charge_capacity_retention_vs_25c", "fraction"),
    ]
    for _, row in frame.iterrows():
        for column, name, unit in descriptors:
            add_descriptor(
                records,
                row,
                "data_processed/capacity/capacity_temperature_summary.csv",
                "capacity",
                "cell_temperature",
                name,
                value(row, column),
                unit,
                actual_temperature_column="actual_temperature_mean_deg_c",
                source_notes="Capacity and retention from raw_CU_diffT checkup StepIDs.",
            )


def append_usable_energy(records: list[dict[str, object]], frame: pd.DataFrame) -> None:
    descriptors = [
        ("usable_energy_wh", "usable_energy", "Wh"),
        ("capacity_from_segments_ah", "capacity_from_ocv_segments", "Ah"),
        ("mean_ocv_v", "mean_ocv", "V"),
        ("energy_retention_vs_25c", "usable_energy_retention_vs_25c", "fraction"),
        ("capacity_retention_from_segments_vs_25c", "capacity_retention_from_ocv_segments_vs_25c", "fraction"),
    ]
    for _, row in frame.iterrows():
        for column, name, unit in descriptors:
            add_descriptor(
                records,
                row,
                "data_processed/descriptors/usable_energy_retention.csv",
                "ocv_usable_energy",
                "cell_temperature_direction",
                name,
                value(row, column),
                unit,
                direction_column="direction",
                source_notes="OCV-aware energy integration from checkup OCV segments.",
            )


def append_ocv_slope(records: list[dict[str, object]], frame: pd.DataFrame) -> None:
    descriptors = [
        ("ocv_slope_penalty_v", "ocv_slope_penalty", "V"),
        ("ocv_slope_penalty_vs_25c", "ocv_slope_penalty_vs_25c", "fraction"),
        ("mean_abs_dv_dcapacity_fraction_v", "mean_abs_dv_dcapacity_fraction", "V/fraction"),
        ("median_abs_dv_dcapacity_fraction_v", "median_abs_dv_dcapacity_fraction", "V/fraction"),
        ("max_abs_dv_dcapacity_fraction_v", "max_abs_dv_dcapacity_fraction", "V/fraction"),
        ("voltage_span_v", "ocv_voltage_span", "V"),
    ]
    for _, row in frame.iterrows():
        voltage_window = f"{value(row, 'window_lower_fraction'):.2f}-{value(row, 'window_upper_fraction'):.2f}"
        for column, name, unit in descriptors:
            add_descriptor(
                records,
                row,
                "data_processed/ocv/ocv_slope_descriptors.csv",
                "ocv_slope",
                "cell_temperature_direction_capacity_window_smoothing",
                name,
                value(row, column),
                unit,
                direction_column="direction",
                voltage_column=None,
                voltage_window_v=voltage_window,
                source_notes="Smoothed OCV derivative descriptor; SOC axis is capacity fraction traversed.",
            )


def append_raw_pulse_summary(records: list[dict[str, object]], frame: pd.DataFrame) -> None:
    descriptors = [
        ("pulse_count", "pulse_count", "count"),
        ("complete_1s_count", "complete_1s_count", "count"),
        ("complete_10s_count", "complete_10s_count", "count"),
        ("median_actual_temperature_deg_c", "median_actual_pulse_temperature", "deg C"),
        ("median_r_fast_mohm", "median_r_fast", "mohm"),
        ("median_r_100ms_mohm", "median_r_100ms", "mohm"),
        ("median_r_1s_mohm", "median_r_1s", "mohm"),
        ("median_r_10s_mohm", "median_r_10s", "mohm"),
        ("median_r_end_mohm", "median_r_end", "mohm"),
        ("median_polarization_1s_mohm", "median_polarization_1s", "mohm"),
        ("median_polarization_10s_mohm", "median_polarization_10s", "mohm"),
        ("median_polarization_end_mohm", "median_polarization_end", "mohm"),
        ("median_relaxation_end_mohm", "median_relaxation_end", "mohm"),
    ]
    for _, row in frame.iterrows():
        for column, name, unit in descriptors:
            add_descriptor(
                records,
                row,
                "results/tables/table_raw_pulse_temperature_summary.csv",
                "raw_pulse",
                "family_temperature_direction_c_rate",
                name,
                value(row, column),
                unit,
                actual_temperature_column="median_actual_temperature_deg_c",
                pulse_direction_column="pulse_direction",
                abs_c_rate_column="abs_c_rate",
                voltage_column=None,
                source_notes="Temperature-resolved raw pulse summary over protocol SOC.",
            )


def append_raw_pulse_soc_map(records: list[dict[str, object]], frame: pd.DataFrame) -> None:
    descriptors = [
        ("pulse_count", "soc_pulse_count", "count"),
        ("complete_1s_count", "soc_complete_1s_count", "count"),
        ("complete_10s_count", "soc_complete_10s_count", "count"),
        ("median_r_fast_mohm", "soc_median_r_fast", "mohm"),
        ("median_r_1s_mohm", "soc_median_r_1s", "mohm"),
        ("median_r_10s_mohm", "soc_median_r_10s", "mohm"),
        ("median_polarization_1s_mohm", "soc_median_polarization_1s", "mohm"),
    ]
    for _, row in frame.iterrows():
        for column, name, unit in descriptors:
            add_descriptor(
                records,
                row,
                "results/tables/table_raw_pulse_soc_temperature_map.csv",
                "raw_pulse_soc_map",
                "family_temperature_soc_direction_c_rate",
                name,
                value(row, column),
                unit,
                pulse_direction_column="pulse_direction",
                abs_c_rate_column="abs_c_rate",
                nominal_soc_column="nominal_soc_percent",
                voltage_column=None,
                source_notes="Protocol-SOC pulse map; SOC is inferred from pulse-block order.",
            )


def append_eis(records: list[dict[str, object]], frame: pd.DataFrame) -> None:
    descriptors = [
        ("series_resistance_high_freq_ohm", "eis_series_resistance_high_freq", "ohm"),
        ("series_resistance_min_abs_zim_ohm", "eis_series_resistance_min_abs_zim", "ohm"),
        ("low_freq_zabs_ohm", "eis_low_freq_zabs", "ohm"),
        ("low_freq_zre_ohm", "eis_low_freq_zre", "ohm"),
        ("low_freq_zim_ohm", "eis_low_freq_zim", "ohm"),
        ("arc_width_zre_ohm", "eis_arc_width_zre", "ohm"),
        ("min_zim_ohm", "eis_min_zim", "ohm"),
        ("characteristic_freq_min_zim_hz", "eis_characteristic_freq_min_zim", "Hz"),
    ]
    for _, row in frame.iterrows():
        voltage_window = f"{value(row, 'voltage_v'):.3f} V" if pd.notna(value(row, "voltage_v")) else ""
        for column, name, unit in descriptors:
            add_descriptor(
                records,
                row,
                "data_processed/eis/eis_spectral_descriptors.csv",
                "eis",
                "cell_voltage_temperature_spectrum",
                name,
                value(row, column),
                unit,
                actual_temperature_column="temperature_mean_deg_c",
                voltage_window_v=voltage_window,
                source_notes="Robust EIS spectral descriptor; ECM fitted parameters are excluded from core analysis.",
            )


def append_entropy(records: list[dict[str, object]], frame: pd.DataFrame) -> None:
    descriptors = [
        ("dudt_v_per_k", "dudt_v_per_k", "V/K"),
        ("dudt_mv_per_k", "dudt_mv_per_k", "mV/K"),
        ("entropic_heat_sensitivity_v_at_25c", "entropic_heat_sensitivity_at_25c", "V"),
    ]
    for _, row in frame.iterrows():
        for column, name, unit in descriptors:
            add_descriptor(
                records,
                row,
                "data_processed/entropy/entropy_dudt_long.csv",
                "entropy",
                "family_soc_direction",
                name,
                value(row, column),
                unit,
                direction_column="direction",
                nominal_soc_column="soc_percent",
                voltage_column=None,
                source_notes="Entropy coefficient curves from upstream NPY arrays.",
            )


def append_thermal(records: list[dict[str, object]], frame: pd.DataFrame) -> None:
    descriptors = [
        ("duration_s", "thermal_response_duration", "s"),
        ("temperature_start_deg_c", "thermal_temperature_start", "deg C"),
        ("temperature_max_deg_c", "thermal_temperature_max", "deg C"),
        ("temperature_rise_max_k", "thermal_temperature_rise_max", "K"),
        ("mean_abs_current_a", "thermal_mean_abs_current", "A"),
        ("mean_voltage_v", "thermal_mean_voltage", "V"),
    ]
    for _, row in frame.iterrows():
        for column, name, unit in descriptors:
            add_descriptor(
                records,
                row,
                "data_processed/thermal/thermal_response_descriptors.csv",
                "thermal_response",
                "cell_direction_bol",
                name,
                value(row, column),
                unit,
                actual_temperature_column="temperature_start_deg_c",
                direction_column="direction",
                voltage_column=None,
                source_notes="BOL thermal response from dS025C temperature traces.",
            )


def build_long_descriptor_matrix(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    append_capacity(records, inputs["capacity"])
    append_usable_energy(records, inputs["usable_energy"])
    append_ocv_slope(records, inputs["ocv_slope"])
    append_raw_pulse_summary(records, inputs["raw_pulse_summary"])
    append_raw_pulse_soc_map(records, inputs["raw_pulse_soc_map"])
    append_eis(records, inputs["eis"])
    append_entropy(records, inputs["entropy"])
    append_thermal(records, inputs["thermal"])
    matrix = pd.DataFrame.from_records(records, columns=LONG_COLUMNS)
    matrix["descriptor_id"] = [f"D{index:06d}" for index in range(1, len(matrix) + 1)]
    return matrix


def flatten_pulse_summary(pulse_summary: pd.DataFrame) -> pd.DataFrame:
    pulse = pulse_summary[pulse_summary["abs_c_rate"] == 1.0].copy()
    keep = [
        "median_r_fast_mohm",
        "median_r_100ms_mohm",
        "median_r_1s_mohm",
        "median_r_10s_mohm",
        "median_polarization_1s_mohm",
        "median_polarization_10s_mohm",
        "median_relaxation_end_mohm",
    ]
    pieces = []
    for direction, group in pulse.groupby("pulse_direction"):
        renamed = group[["cell_family", "temperature_deg_c"] + keep].rename(
            columns={column: f"pulse_1c_{direction}_{column}" for column in keep}
        )
        pieces.append(renamed)
    if not pieces:
        return pd.DataFrame()
    output = pieces[0]
    for piece in pieces[1:]:
        output = output.merge(piece, on=["cell_family", "temperature_deg_c"], how="outer")
    return output


def flatten_directional_summary(
    frame: pd.DataFrame,
    value_columns: list[str],
    prefix: str,
    join_columns: list[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    join_columns = join_columns or ["cell_family"]
    pieces = []
    for direction, group in frame.groupby("direction"):
        keep = join_columns + value_columns
        renamed = group[keep].rename(columns={column: f"{prefix}_{direction}_{column}" for column in value_columns})
        pieces.append(renamed)
    output = pieces[0]
    for piece in pieces[1:]:
        output = output.merge(piece, on=join_columns, how="outer")
    return output


def build_family_temperature_matrix(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    capacity = inputs["capacity"][
        [
            "cell_id",
            "cell_family",
            "chemistry_type",
            "temperature_deg_c",
            "actual_temperature_mean_deg_c",
            "discharge_capacity_ah",
            "charge_capacity_ah",
            "discharge_capacity_retention_vs_25c",
            "charge_capacity_retention_vs_25c",
        ]
    ].copy()

    energy = inputs["usable_energy"]
    discharge_energy = energy[energy["direction"] == "discharge"][
        [
            "cell_id",
            "cell_family",
            "temperature_deg_c",
            "usable_energy_wh",
            "energy_retention_vs_25c",
            "mean_ocv_v",
        ]
    ].rename(
        columns={
            "usable_energy_wh": "discharge_usable_energy_wh",
            "energy_retention_vs_25c": "discharge_usable_energy_retention_vs_25c",
            "mean_ocv_v": "discharge_mean_ocv_v",
        }
    )
    wide = capacity.merge(discharge_energy, on=["cell_id", "cell_family", "temperature_deg_c"], how="left")

    ocv_slope = inputs["ocv_slope"]
    if not ocv_slope.empty:
        primary_slope = ocv_slope[
            (ocv_slope["smoothing_window_segments"] == 5)
            & (ocv_slope["capacity_window"] == "central_20_80")
        ]
        slope_flat = flatten_directional_summary(
            primary_slope,
            [
                "ocv_slope_penalty_v",
                "ocv_slope_penalty_vs_25c",
                "mean_abs_dv_dcapacity_fraction_v",
            ],
            "ocv_slope",
            join_columns=["cell_family", "temperature_deg_c"],
        )
        if not slope_flat.empty:
            wide = wide.merge(slope_flat, on=["cell_family", "temperature_deg_c"], how="left")

    pulse = flatten_pulse_summary(inputs["raw_pulse_summary"])
    wide = wide.merge(pulse, on=["cell_family", "temperature_deg_c"], how="left")

    eis_consistency = inputs["eis_hppc_consistency"].copy()
    if not eis_consistency.empty:
        eis_consistency["temperature_deg_c"] = 25.0
        keep = [
            "cell_family",
            "temperature_deg_c",
            "eis_series_resistance_mohm",
            "eis_low_freq_zre_mohm",
            "eis_arc_width_zre_mohm",
            "eis_low_freq_to_pulse_r_1s_ratio",
        ]
        wide = wide.merge(eis_consistency[keep], on=["cell_family", "temperature_deg_c"], how="left")

    entropy_summary = inputs["entropy_summary"]
    entropy_flat = flatten_directional_summary(
        entropy_summary,
        ["mean_abs_dudt_mv_per_k", "max_abs_dudt_mv_per_k", "mean_ehs_v_at_25c"],
        "entropy",
    )
    if not entropy_flat.empty:
        wide = wide.merge(entropy_flat, on="cell_family", how="left")

    thermal_summary = inputs["thermal_summary"]
    thermal_flat = flatten_directional_summary(
        thermal_summary,
        ["mean_temperature_rise_max_k", "median_temperature_rise_max_k", "mean_abs_current_a"],
        "bol_thermal",
    )
    if not thermal_flat.empty:
        wide = wide.merge(thermal_flat, on="cell_family", how="left")

    return wide.sort_values(["cell_family", "temperature_deg_c"]).reset_index(drop=True)


def add_target_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    targets = matrix[
        [
            "cell_family",
            "chemistry_type",
            "temperature_deg_c",
            "discharge_capacity_retention_vs_25c",
            "discharge_usable_energy_retention_vs_25c",
            "discharge_mean_ocv_v",
            "pulse_1c_discharge_median_r_1s_mohm",
        ]
    ].copy()
    targets = targets.rename(
        columns={
            "discharge_capacity_retention_vs_25c": "target_capacity_retention_vs_25c",
            "discharge_usable_energy_retention_vs_25c": "target_usable_energy_retention_vs_25c",
            "pulse_1c_discharge_median_r_1s_mohm": "target_pulse_r_1s_mohm",
        }
    )
    resistance_ohm = targets["target_pulse_r_1s_mohm"] / 1000.0
    targets["target_pulse_power_proxy_w"] = (targets["discharge_mean_ocv_v"] ** 2) / (4.0 * resistance_ohm)
    targets.loc[~np.isfinite(targets["target_pulse_power_proxy_w"]), "target_pulse_power_proxy_w"] = np.nan
    targets["target_pulse_r_1s_growth_vs_25c"] = np.nan
    targets["target_pulse_power_proxy_vs_25c"] = np.nan

    for _, group in targets.groupby("cell_family", dropna=False):
        ref = group[group["temperature_deg_c"] == 25]
        if ref.empty:
            continue
        ref_r = float(ref["target_pulse_r_1s_mohm"].iloc[0])
        ref_power = float(ref["target_pulse_power_proxy_w"].iloc[0])
        index = group.index
        if ref_r:
            targets.loc[index, "target_pulse_r_1s_growth_vs_25c"] = targets.loc[index, "target_pulse_r_1s_mohm"] / ref_r
        if ref_power:
            targets.loc[index, "target_pulse_power_proxy_vs_25c"] = (
                targets.loc[index, "target_pulse_power_proxy_w"] / ref_power
            )
    targets["target_energy_retention_loss_vs_25c"] = 1.0 - targets["target_usable_energy_retention_vs_25c"]
    targets["target_low_temperature_penalty"] = targets["target_energy_retention_loss_vs_25c"].where(
        targets["temperature_deg_c"] < 25.0, 0.0
    )
    return targets.sort_values(["cell_family", "temperature_deg_c"]).reset_index(drop=True)


def write_report(path: Path, long_matrix: pd.DataFrame, wide_matrix: pd.DataFrame, targets: pd.DataFrame) -> None:
    ensure_parent(path)
    measurement_counts = long_matrix.groupby("measurement_type").size().sort_values(ascending=False)
    lines = [
        "# Integrated Descriptor Matrix",
        "",
        f"Long descriptor rows: {len(long_matrix)}",
        f"Family-temperature rows: {len(wide_matrix)}",
        f"Target rows: {len(targets)}",
        "",
        "## Long Matrix Coverage",
        "",
    ]
    for measurement, count in measurement_counts.items():
        lines.append(f"- {measurement}: {count} descriptor rows")
    lines.extend(
        [
            "",
            "## Modeling Targets",
            "",
            "- `target_usable_energy_retention_vs_25c`: discharge OCV-aware usable-energy retention.",
            "- `target_capacity_retention_vs_25c`: discharge capacity retention.",
            "- `target_pulse_r_1s_growth_vs_25c`: 1C discharge raw-pulse R_1s normalized to 25 deg C.",
            "- `target_pulse_power_proxy_w`: OCV-based V^2/(4R_1s) power proxy.",
            "- `target_energy_retention_loss_vs_25c`: 1 minus usable-energy retention.",
            "- `target_low_temperature_penalty`: energy-retention loss below 25 deg C; zero at and above 25 deg C.",
            "",
            "## Cautions",
            "",
            "- The long matrix mixes granularities; use `aggregation_level` before modeling.",
            "- BOL entropy and thermal descriptors are family-level context and are repeated in the wide matrix by family where used.",
            "- Raw pulse SOC has been validated as an ordered protocol block index, not as exact coulomb-counted SOC.",
            "- EIS descriptors are robust spectral descriptors; ECM fitted parameters are intentionally excluded from the core matrix.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    inputs = {
        "capacity": read_csv("data_processed/capacity/capacity_temperature_summary.csv"),
        "usable_energy": read_csv("data_processed/descriptors/usable_energy_retention.csv"),
        "ocv_slope": read_csv("data_processed/ocv/ocv_slope_descriptors.csv"),
        "raw_pulse_summary": read_csv("results/tables/table_raw_pulse_temperature_summary.csv"),
        "raw_pulse_soc_map": read_csv("results/tables/table_raw_pulse_soc_temperature_map.csv"),
        "eis": read_csv("data_processed/eis/eis_spectral_descriptors.csv"),
        "entropy": read_csv("data_processed/entropy/entropy_dudt_long.csv"),
        "thermal": read_csv("data_processed/thermal/thermal_response_descriptors.csv"),
        "eis_hppc_consistency": read_csv("results/tables/table_eis_hppc_consistency.csv"),
        "entropy_summary": read_csv("results/tables/table_entropy_summary.csv"),
        "thermal_summary": read_csv("results/tables/table_thermal_response_summary.csv"),
    }

    long_matrix = build_long_descriptor_matrix(inputs)
    wide_matrix = build_family_temperature_matrix(inputs)
    targets = add_target_columns(wide_matrix)

    outputs = {
        "data_processed/descriptors/integrated_descriptor_long.csv": long_matrix,
        "data_processed/descriptors/family_temperature_descriptor_matrix.csv": wide_matrix,
        "data_processed/descriptors/family_temperature_targets.csv": targets,
    }
    for file_name, frame in outputs.items():
        output_path = Path(file_name)
        ensure_parent(output_path)
        frame.to_csv(output_path, index=False)
        print(f"{file_name}: {len(frame)} rows")

    write_report(Path("results/reports/integrated_descriptor_matrix.md"), long_matrix, wide_matrix, targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
