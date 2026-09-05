#!/usr/bin/env python3
"""Build manuscript-facing Tables 1-3 from generated descriptor artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, divider] + rows)


def write_table(path: Path, title: str, frame: pd.DataFrame, note: str = "") -> None:
    ensure_parent(path)
    lines = [f"# {title}", "", markdown_table(frame)]
    if note:
        lines.extend(["", f"Note: {note}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_table1_dataset_summary() -> pd.DataFrame:
    summary = json.loads(Path("data_processed/qc/raw_file_inventory_summary.json").read_text(encoding="utf-8"))
    stream_counts = summary["measurement_stream_counts"]
    rows = []
    for stream, count in sorted(stream_counts.items()):
        rows.append({"Category": "measurement_stream", "Item": stream, "Count": count})
    for family, count in sorted(summary["cell_family_counts"].items()):
        rows.append({"Category": "cell_family", "Item": family, "Count": count})
    rows.append({"Category": "record_type", "Item": "total_records", "Count": summary["record_count"]})
    rows.append({"Category": "temperature", "Item": "detected_deg_c", "Count": ", ".join(summary["temperatures_detected_deg_c"])})
    return pd.DataFrame(rows)


def build_table2_descriptor_definitions() -> pd.DataFrame:
    rows = []
    in_table = False
    for line in Path("docs/descriptor_definitions.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| Descriptor |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cells = [cell.strip(" `") for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 4:
                rows.append(
                    {
                        "Descriptor": cells[0].strip(),
                        "Source": cells[1].strip(),
                        "Definition": cells[2].strip(),
                        "Output": cells[3].strip(),
                    }
                )
        elif in_table and line.strip() == "":
            break
    return pd.DataFrame(rows)


def li_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame.loc[frame["cell_family"].isin(["LFP", "LTO", "NMC"]), column].mean())


def sib_value(frame: pd.DataFrame, column: str) -> float:
    return float(frame.loc[frame["cell_family"] == "SIB", column].iloc[0])


def build_table3_sib_li_metrics() -> pd.DataFrame:
    targets = pd.read_csv("data_processed/descriptors/family_temperature_targets.csv")
    pulse = pd.read_csv("results/tables/table_raw_pulse_sib_vs_li_reference.csv")
    eis = pd.read_csv("results/tables/table_eis_family_summary.csv")
    ocv_slope = pd.read_csv("results/tables/table_ocv_slope_summary.csv")
    entropy = pd.read_csv("results/tables/table_entropy_summary.csv")
    thermal = pd.read_csv("results/tables/table_thermal_response_summary.csv")

    target_5 = targets[targets["temperature_deg_c"] == 5]
    target_25 = targets[targets["temperature_deg_c"] == 25]
    pulse_5 = pulse[
        (pulse["temperature_deg_c"] == 5)
        & (pulse["pulse_direction"] == "discharge")
        & (pulse["abs_c_rate"] == 1.0)
    ].iloc[0]
    pulse_25 = pulse[
        (pulse["temperature_deg_c"] == 25)
        & (pulse["pulse_direction"] == "discharge")
        & (pulse["abs_c_rate"] == 1.0)
    ].iloc[0]
    entropy_dis = entropy[entropy["direction"] == "discharge"]
    ocv_slope_5 = ocv_slope[(ocv_slope["direction"] == "discharge") & (ocv_slope["temperature_deg_c"] == 5)]
    thermal_dis = thermal[thermal["direction"] == "discharge"]

    rows = [
        {
            "Metric": "5 deg C usable-energy retention",
            "SIB": f"{sib_value(target_5, 'target_usable_energy_retention_vs_25c'):.3f}",
            "Li-ion reference mean": f"{li_mean(target_5, 'target_usable_energy_retention_vs_25c'):.3f}",
            "Comparison": "lower is worse",
            "Source": "family_temperature_targets.csv",
        },
        {
            "Metric": "5 deg C capacity retention",
            "SIB": f"{sib_value(target_5, 'target_capacity_retention_vs_25c'):.3f}",
            "Li-ion reference mean": f"{li_mean(target_5, 'target_capacity_retention_vs_25c'):.3f}",
            "Comparison": "lower is worse",
            "Source": "family_temperature_targets.csv",
        },
        {
            "Metric": "5 deg C 1C discharge R1s",
            "SIB": f"{pulse_5['sib_median_r_1s_mohm']:.1f} mohm",
            "Li-ion reference mean": f"{pulse_5['li_reference_median_r_1s_mohm']:.1f} mohm",
            "Comparison": f"SIB/Li = {pulse_5['sib_to_li_r_1s_ratio']:.2f}",
            "Source": "table_raw_pulse_sib_vs_li_reference.csv",
        },
        {
            "Metric": "5 deg C R1s growth vs 25 deg C",
            "SIB": f"{pulse_5['sib_r_1s_temperature_factor_vs_25c']:.2f}x",
            "Li-ion reference mean": f"{pulse_5['li_reference_r_1s_temperature_factor_vs_25c']:.2f}x",
            "Comparison": "higher is worse",
            "Source": "table_raw_pulse_sib_vs_li_reference.csv",
        },
        {
            "Metric": "25 deg C 1C discharge R1s",
            "SIB": f"{pulse_25['sib_median_r_1s_mohm']:.1f} mohm",
            "Li-ion reference mean": f"{pulse_25['li_reference_median_r_1s_mohm']:.1f} mohm",
            "Comparison": f"SIB/Li = {pulse_25['sib_to_li_r_1s_ratio']:.2f}",
            "Source": "table_raw_pulse_sib_vs_li_reference.csv",
        },
        {
            "Metric": "Room-temperature low-frequency EIS Zre",
            "SIB": f"{sib_value(eis, 'median_low_freq_zre_mohm'):.1f} mohm",
            "Li-ion reference mean": f"{li_mean(eis, 'median_low_freq_zre_mohm'):.1f} mohm",
            "Comparison": "higher is worse",
            "Source": "table_eis_family_summary.csv",
        },
        {
            "Metric": "5 deg C central discharge OSP",
            "SIB": f"{sib_value(ocv_slope_5, 'ocv_slope_penalty_v'):.3f} V",
            "Li-ion reference mean": f"{li_mean(ocv_slope_5, 'ocv_slope_penalty_v'):.3f} V",
            "Comparison": "voltage-shape context",
            "Source": "table_ocv_slope_summary.csv",
        },
        {
            "Metric": "Discharge mean abs dU/dT",
            "SIB": f"{sib_value(entropy_dis, 'mean_abs_dudt_mv_per_k'):.3f} mV/K",
            "Li-ion reference mean": f"{li_mean(entropy_dis, 'mean_abs_dudt_mv_per_k'):.3f} mV/K",
            "Comparison": "context descriptor",
            "Source": "table_entropy_summary.csv",
        },
        {
            "Metric": "BOL discharge temperature rise",
            "SIB": f"{sib_value(thermal_dis, 'mean_temperature_rise_max_k'):.2f} K",
            "Li-ion reference mean": f"{li_mean(thermal_dis, 'mean_temperature_rise_max_k'):.2f} K",
            "Comparison": "higher is worse",
            "Source": "table_thermal_response_summary.csv",
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    table1 = build_table1_dataset_summary()
    table2 = build_table2_descriptor_definitions()
    table3 = build_table3_sib_li_metrics()

    write_table(
        Path("manuscript/tables/table1_dataset_summary.md"),
        "Table 1. Dataset Summary",
        table1,
        "Counts come from the raw file inventory generated from the downloaded DepositOnce archive.",
    )
    write_table(
        Path("manuscript/tables/table2_descriptor_definitions.md"),
        "Table 2. Descriptor Definitions",
        table2,
        "Equivalent-circuit fitted parameters are excluded from the core descriptor list after reliability screening.",
    )
    write_table(
        Path("manuscript/tables/table3_sib_li_normalized_metrics.md"),
        "Table 3. Sodium-Ion Versus Lithium-Ion Metrics",
        table3,
        "Li-ion reference means average LFP, LTO, and NMC descriptors.",
    )
    print("manuscript/tables/table1_dataset_summary.md")
    print("manuscript/tables/table2_descriptor_definitions.md")
    print("manuscript/tables/table3_sib_li_normalized_metrics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
