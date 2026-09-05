#!/usr/bin/env python3
"""Validate protocol-SOC pulse labels against capacity integrated from raw checkup traces."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-energystorage")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/energystorage-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metadata import normal_metadata
from src.zip_readers import iter_members, read_parquet_member


RAW_COLUMNS = [
    "Testtime[s]",
    "StepID",
    "Voltage[V]",
    "Current[A]",
    "Capacity_Step[Ah]",
    "Temperature[°C]",
]

PULSE_STEPS = {28, 32, 36, 40, 44, 48, 62, 66, 70, 74, 78, 82}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def consecutive_segments(frame: pd.DataFrame) -> pd.DataFrame:
    breaks = (frame["StepID"] != frame["StepID"].shift(1)) | (frame.index.to_series().diff() != 1)
    segment_ids = breaks.cumsum()
    records = []
    for segment_index, segment in frame.groupby(segment_ids):
        current = pd.to_numeric(segment["Current[A]"], errors="coerce")
        capacity = pd.to_numeric(segment["Capacity_Step[Ah]"], errors="coerce")
        voltage = pd.to_numeric(segment["Voltage[V]"], errors="coerce")
        temperature = pd.to_numeric(segment["Temperature[°C]"], errors="coerce")
        testtime = pd.to_numeric(segment["Testtime[s]"], errors="coerce")
        cap_delta = np.nan
        if capacity.notna().sum() > 1:
            cap_delta = float(capacity.dropna().iloc[-1] - capacity.dropna().iloc[0])
        duration = np.nan
        if testtime.notna().sum() > 1:
            duration = float(testtime.dropna().iloc[-1] - testtime.dropna().iloc[0])
        records.append(
            {
                "segment_index": int(segment_index),
                "step_id": int(segment["StepID"].iloc[0]),
                "n_points": len(segment),
                "current_median_a": float(current.median()),
                "capacity_delta_ah": cap_delta,
                "capacity_abs_delta_ah": abs(cap_delta) if pd.notna(cap_delta) else np.nan,
                "voltage_start_v": float(voltage.dropna().iloc[0]) if voltage.notna().any() else np.nan,
                "voltage_end_v": float(voltage.dropna().iloc[-1]) if voltage.notna().any() else np.nan,
                "temperature_mean_deg_c": float(temperature.mean()) if temperature.notna().any() else np.nan,
                "duration_s": duration,
            }
        )
    return pd.DataFrame.from_records(records)


def transition_capacity_between(segments: pd.DataFrame, start_segment: int, end_segment: int) -> float:
    between = segments[
        (segments["segment_index"] > start_segment)
        & (segments["segment_index"] < end_segment)
        & (~segments["step_id"].isin(PULSE_STEPS))
        & (segments["capacity_abs_delta_ah"] >= 0.02)
    ]
    return float(between["capacity_abs_delta_ah"].sum())


def validate_sweep(segments: pd.DataFrame, sweep_name: str, block_start_step: int) -> list[dict[str, object]]:
    starts = segments[
        (segments["step_id"] == block_start_step)
        & (segments["current_median_a"].abs() > 0.5)
    ].sort_values("segment_index")
    if len(starts) < 2:
        return []

    start_indices = list(starts["segment_index"])
    cumulative = [0.0]
    running = 0.0
    for previous, current in zip(start_indices[:-1], start_indices[1:]):
        running += transition_capacity_between(segments, previous, current)
        cumulative.append(running)

    first_nominal = 100.0 if sweep_name == "discharge_sweep" else 0.0
    last_nominal = (
        max(0.0, 100.0 - 10.0 * (len(starts) - 1))
        if sweep_name == "discharge_sweep"
        else min(100.0, 10.0 * (len(starts) - 1))
    )
    observed_fraction = abs(first_nominal - last_nominal) / 100.0
    denominator = cumulative[-1] / observed_fraction if cumulative[-1] and observed_fraction else np.nan
    records = []
    for block_index, (segment_index, cumulative_ah) in enumerate(zip(start_indices, cumulative)):
        nominal_soc = 100.0 - 10.0 * block_index if sweep_name == "discharge_sweep" else 10.0 * block_index
        nominal_soc = max(0.0, min(100.0, nominal_soc))
        if pd.isna(denominator) or denominator == 0:
            measured_soc = np.nan
        elif sweep_name == "discharge_sweep":
            measured_soc = 100.0 * (1.0 - cumulative_ah / denominator)
        else:
            measured_soc = 100.0 * cumulative_ah / denominator
        segment_row = segments.loc[segments["segment_index"] == segment_index].iloc[0]
        records.append(
            {
                "sweep_name": sweep_name,
                "pulse_block_index": block_index,
                "block_count": len(starts),
                "segment_index": int(segment_index),
                "block_start_step": block_start_step,
                "nominal_soc_percent": nominal_soc,
                "capacity_integrated_soc_percent": measured_soc,
                "soc_abs_error_percent": abs(measured_soc - nominal_soc) if pd.notna(measured_soc) else np.nan,
                "cumulative_transition_capacity_ah": cumulative_ah,
                "observed_protocol_soc_span_percent": abs(first_nominal - last_nominal),
                "total_transition_capacity_ah": denominator,
                "block_voltage_start_v": float(segment_row["voltage_start_v"]),
                "temperature_mean_deg_c": float(segment_row["temperature_mean_deg_c"]),
            }
        )
    return records


def validate_member(zip_path: Path, member: str) -> list[dict[str, object]]:
    meta = normal_metadata(member)
    frame = read_parquet_member(zip_path, member, columns=RAW_COLUMNS)
    segments = consecutive_segments(frame)
    records = []
    for sweep_name, block_start_step in [("discharge_sweep", 28), ("charge_sweep", 62)]:
        for record in validate_sweep(segments, sweep_name, block_start_step):
            records.append({**meta, **record})
    return records


def build_summary(validation: pd.DataFrame) -> pd.DataFrame:
    return (
        validation.groupby(["cell_family", "temperature_deg_c", "sweep_name"], dropna=False)
        .agg(
            block_count=("pulse_block_index", "count"),
            max_soc_abs_error_percent=("soc_abs_error_percent", "max"),
            mean_soc_abs_error_percent=("soc_abs_error_percent", "mean"),
            total_transition_capacity_ah=("total_transition_capacity_ah", "median"),
        )
        .reset_index()
        .sort_values(["cell_family", "temperature_deg_c", "sweep_name"])
    )


def plot_validation(validation: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    colors = {"LFP": "#eb6834", "LTO": "#4a3aa7", "NMC": "#1baf7a", "SIB": "#2a78d6"}
    for axis, sweep_name in zip(axes, ["discharge_sweep", "charge_sweep"]):
        view = validation[validation["sweep_name"] == sweep_name]
        for family, group in view.groupby("cell_family"):
            axis.scatter(
                group["nominal_soc_percent"],
                group["capacity_integrated_soc_percent"],
                s=18,
                alpha=0.65,
                color=colors.get(family, "black"),
                label=family,
            )
        axis.plot([0, 100], [0, 100], color="0.35", linestyle="--", linewidth=1)
        axis.set_title(sweep_name.replace("_", " "))
        axis.set_xlabel("Protocol SOC (%)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Capacity-integrated SOC (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(path: Path, validation: pd.DataFrame, summary: pd.DataFrame) -> None:
    ensure_parent(path)
    max_error = float(summary["max_soc_abs_error_percent"].max())
    mean_error = float(summary["mean_soc_abs_error_percent"].mean())
    decision = (
        "protocol-SOC indexing is consistent with integrated block capacity within a 5 percentage-point tolerance."
        if max_error <= 5.0
        else "protocol-SOC indexing is acceptable as a protocol block index, but not as a precise coulomb-counted SOC."
    )
    lines = [
        "# Protocol-SOC Validation",
        "",
        "The protocol-SOC labels used in raw pulse maps were checked against capacity integrated from large inter-block transition steps in each raw checkup trace.",
        "",
        f"Rows checked: {len(validation)} pulse-block positions.",
        f"Maximum family-temperature sweep error: {max_error:.3f} percentage points.",
        f"Mean family-temperature sweep error: {mean_error:.3f} percentage points.",
        "",
        f"Decision: {decision}",
        "",
        "Caveat: the validation checks block positions within each sweep. It does not convert every pulse sample to a fully coulomb-counted SOC trajectory.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="data_raw/data_EvalSIB.zip")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    records: list[dict[str, object]] = []
    members = [member for member in iter_members(zip_path, {".parquet"}) if member.startswith("raw_CU_diffT/")]
    for member in members:
        member_records = validate_member(zip_path, member)
        records.extend(member_records)
        print(f"{member}: {len(member_records)} SOC block checks")

    validation = pd.DataFrame.from_records(records)
    summary = build_summary(validation)
    outputs = {
        "data_processed/qc/protocol_soc_validation.csv": validation,
        "results/tables/table_protocol_soc_validation_summary.csv": summary,
    }
    for file_name, frame in outputs.items():
        output_path = Path(file_name)
        ensure_parent(output_path)
        frame.to_csv(output_path, index=False)
        print(f"{file_name}: {len(frame)} rows")
    plot_validation(validation, Path("results/figures/figure_protocol_soc_validation.png"))
    plot_validation(validation, Path("manuscript/figures/figure_protocol_soc_validation.png"))
    write_report(Path("results/reports/protocol_soc_validation.md"), validation, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
