#!/usr/bin/env python3
"""Extract temperature-resolved pulse descriptors from raw checkup parquet files."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metadata import normal_metadata
from src.zip_readers import iter_members


RAW_COLUMNS = [
    "Testtime[s]",
    "StepID",
    "Voltage[V]",
    "Current[A]",
    "Temperature[°C]",
    "Capacity_Step[Ah]",
]

PULSE_STEP_TO_C_RATE = {
    28: -3.0,
    32: 3.0,
    36: -2.0,
    40: 2.0,
    44: -1.0,
    48: 1.0,
    62: 3.0,
    66: -3.0,
    70: 2.0,
    74: -2.0,
    78: 1.0,
    82: -1.0,
}

PULSE_STEPS = set(PULSE_STEP_TO_C_RATE)
SET_A_STEPS = {28, 32, 36, 40, 44, 48}
SET_B_STEPS = {62, 66, 70, 74, 78, 82}


def read_parquet_member(zip_path: Path, member_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_path) as handle:
            payload = handle.read()
    return pd.read_parquet(io.BytesIO(payload), columns=RAW_COLUMNS)


def consecutive_step_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    breaks = (frame["StepID"] != frame["StepID"].shift(1)) | (frame.index.to_series().diff() != 1)
    segment_ids = breaks.cumsum()
    return [segment for _, segment in frame.groupby(segment_ids)]


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def value_at_elapsed(segment: pd.DataFrame, column: str, elapsed_s: float) -> float:
    times = numeric_series(segment, "Testtime[s]")
    values = pd.to_numeric(segment.loc[times.index, column], errors="coerce")
    valid = values.notna()
    times = times[valid]
    values = values[valid]
    if times.empty:
        return np.nan
    elapsed = times.to_numpy(dtype=float) - float(times.iloc[0])
    idx = int(np.searchsorted(elapsed, elapsed_s, side="left"))
    if idx >= len(values):
        if elapsed[-1] >= elapsed_s - 0.1:
            return float(values.iloc[-1])
        return np.nan
    return float(values.iloc[idx])


def segment_duration_s(segment: pd.DataFrame) -> float:
    times = numeric_series(segment, "Testtime[s]")
    if len(times) < 2:
        return 0.0
    return float(times.iloc[-1] - times.iloc[0])


def last_value(segment: pd.DataFrame, column: str) -> float:
    values = numeric_series(segment, column)
    return float(values.iloc[-1]) if not values.empty else np.nan


def first_value(segment: pd.DataFrame, column: str) -> float:
    values = numeric_series(segment, column)
    return float(values.iloc[0]) if not values.empty else np.nan


def mean_value(segment: pd.DataFrame, column: str) -> float:
    values = numeric_series(segment, column)
    return float(values.mean()) if not values.empty else np.nan


def median_value(segment: pd.DataFrame, column: str) -> float:
    values = numeric_series(segment, column)
    return float(values.median()) if not values.empty else np.nan


def cap_delta(segment: pd.DataFrame) -> float:
    values = numeric_series(segment, "Capacity_Step[Ah]")
    if len(values) < 2:
        return 0.0
    return float(values.iloc[-1] - values.iloc[0])


def is_rest_segment(segment: pd.DataFrame) -> bool:
    current = numeric_series(segment, "Current[A]")
    return not current.empty and float(current.abs().median()) < 0.05


def nearest_previous_rest(segments: list[pd.DataFrame], index: int) -> pd.DataFrame | None:
    for previous in range(index - 1, -1, -1):
        if is_rest_segment(segments[previous]):
            return segments[previous]
    return None


def nearest_next_rest(segments: list[pd.DataFrame], index: int) -> pd.DataFrame | None:
    for next_index in range(index + 1, len(segments)):
        if is_rest_segment(segments[next_index]):
            return segments[next_index]
    return None


def resistance(pre_voltage: float, pulse_voltage: float, delta_current: float) -> float:
    if np.isnan(pre_voltage) or np.isnan(pulse_voltage) or not delta_current:
        return np.nan
    return abs((pulse_voltage - pre_voltage) / delta_current)


def extract_member(zip_path: Path, member_path: str) -> list[dict[str, object]]:
    frame = read_parquet_member(zip_path, member_path)
    segments = consecutive_step_segments(frame)
    meta = normal_metadata(member_path)
    records: list[dict[str, object]] = []
    block_count_by_set: dict[str, int] = defaultdict(lambda: -1)

    for segment_index, segment in enumerate(segments):
        step_id = int(segment["StepID"].iloc[0])
        if step_id not in PULSE_STEPS:
            continue

        if step_id in SET_A_STEPS:
            pulse_set = "discharge_sweep"
            block_start_step = 28
            soc_direction = -1
        else:
            pulse_set = "charge_sweep"
            block_start_step = 62
            soc_direction = 1

        if step_id == block_start_step:
            block_count_by_set[pulse_set] += 1
        block_index = block_count_by_set[pulse_set]
        if block_index < 0:
            block_count_by_set[pulse_set] = 0
            block_index = 0

        nominal_soc = 100 - 10 * block_index if soc_direction < 0 else 10 * block_index
        nominal_soc = max(0, min(100, nominal_soc))

        duration = segment_duration_s(segment)
        current_median = median_value(segment, "Current[A]")
        if abs(current_median) < 0.5:
            continue

        previous_rest = nearest_previous_rest(segments, segment_index)
        next_rest = nearest_next_rest(segments, segment_index)
        pre_voltage = last_value(previous_rest, "Voltage[V]") if previous_rest is not None else first_value(segment, "Voltage[V]")
        pre_current = median_value(previous_rest, "Current[A]") if previous_rest is not None else 0.0
        delta_current = current_median - pre_current

        v_fast = value_at_elapsed(segment, "Voltage[V]", 0.02)
        v_100ms = value_at_elapsed(segment, "Voltage[V]", 0.10)
        v_1s = value_at_elapsed(segment, "Voltage[V]", 1.0)
        v_10s = value_at_elapsed(segment, "Voltage[V]", 10.0)
        v_end = last_value(segment, "Voltage[V]")
        v_relax_60s = value_at_elapsed(next_rest, "Voltage[V]", 60.0) if next_rest is not None else np.nan
        v_relax_end = last_value(next_rest, "Voltage[V]") if next_rest is not None else np.nan

        r_fast = resistance(pre_voltage, v_fast, delta_current)
        r_100ms = resistance(pre_voltage, v_100ms, delta_current)
        r_1s = resistance(pre_voltage, v_1s, delta_current)
        r_10s = resistance(pre_voltage, v_10s, delta_current)
        r_end = resistance(pre_voltage, v_end, delta_current)
        relax_60s = resistance(v_end, v_relax_60s, delta_current)
        relax_end = resistance(v_end, v_relax_end, delta_current)

        records.append(
            {
                **meta,
                "pulse_set": pulse_set,
                "pulse_block_index": block_index,
                "nominal_soc_percent": nominal_soc,
                "segment_index": segment_index,
                "step_id": step_id,
                "nominal_c_rate": PULSE_STEP_TO_C_RATE[step_id],
                "pulse_direction": "charge" if current_median > 0 else "discharge",
                "duration_s": duration,
                "complete_1s": duration >= 1.0,
                "complete_10s": duration >= 9.9,
                "n_points": len(segment),
                "current_median_a": current_median,
                "current_mean_a": mean_value(segment, "Current[A]"),
                "pre_voltage_v": pre_voltage,
                "pulse_voltage_fast_v": v_fast,
                "pulse_voltage_100ms_v": v_100ms,
                "pulse_voltage_1s_v": v_1s,
                "pulse_voltage_10s_v": v_10s,
                "pulse_voltage_end_v": v_end,
                "relax_voltage_60s_v": v_relax_60s,
                "relax_voltage_end_v": v_relax_end,
                "temperature_mean_deg_c": mean_value(segment, "Temperature[°C]"),
                "capacity_delta_ah": cap_delta(segment),
                "r_fast_ohm": r_fast,
                "r_100ms_ohm": r_100ms,
                "r_1s_ohm": r_1s,
                "r_10s_ohm": r_10s,
                "r_end_ohm": r_end,
                "polarization_1s_minus_fast_ohm": r_1s - r_fast if not np.isnan(r_1s) and not np.isnan(r_fast) else np.nan,
                "polarization_end_minus_fast_ohm": r_end - r_fast if not np.isnan(r_end) and not np.isnan(r_fast) else np.nan,
                "relaxation_60s_ohm": relax_60s,
                "relaxation_end_ohm": relax_end,
            }
        )
    return records


def write_report(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Raw Pulse Descriptor Report",
        "",
        f"Rows: {len(frame)}",
        "",
        "## Coverage",
        "",
    ]
    if not frame.empty:
        coverage = (
            frame.groupby(["cell_family", "temperature_deg_c"])
            .size()
            .reset_index(name="pulse_count")
            .sort_values(["cell_family", "temperature_deg_c"])
        )
        lines.append(coverage.to_markdown(index=False))
        lines.extend(
            [
                "",
                "## Extraction Rules",
                "",
                "- Pulse StepIDs are mapped from the raw checkup protocol: 28/32/36/40/44/48 and 62/66/70/74/78/82.",
                "- Resistance uses absolute voltage change divided by current step from the previous rest segment.",
                "- `r_fast_ohm`, `r_100ms_ohm`, `r_1s_ohm`, and `r_10s_ohm` are operational pulse descriptors.",
                "- `complete_10s` flags pulses that lasted at least 9.9 s; shorter boundary-clipped pulses should not be used for 10 s resistance.",
                "- `nominal_soc_percent` is assigned from pulse-block order and should be treated as a protocol SOC index.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="data_raw/data_EvalSIB.zip")
    parser.add_argument("--output", default="data_processed/hppc/raw_temperature_pulse_descriptors.csv")
    parser.add_argument("--report", default="logs/raw_pulse_descriptor_report.md")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    records: list[dict[str, object]] = []
    members = [member for member in iter_members(zip_path, {".parquet"}) if member.startswith("raw_CU_diffT/")]
    for member in members:
        member_records = extract_member(zip_path, member)
        records.extend(member_records)
        print(f"{member}: {len(member_records)} pulses")

    frame = pd.DataFrame.from_records(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    write_report(Path(args.report), frame)
    print(f"output: {output}")
    print(f"rows: {len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
