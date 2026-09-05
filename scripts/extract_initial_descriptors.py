#!/usr/bin/env python3
"""Extract initial descriptor tables from the raw DepositOnce archive."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

# numpy 2.0 renamed trapz -> trapezoid; keep both working so the pipeline
# runs on the pinned stack and on newer numpy alike.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metadata import (
    entropy_direction,
    extract_temperature_deg_c,
    hppc_file_direction,
    normal_metadata,
)
from src.zip_readers import iter_members, read_csv_member, read_npy_member, read_parquet_member


CHECKUP_COLUMNS = [
    "Testtime[s]",
    "StepID",
    "Voltage[V]",
    "Current[A]",
    "Temperature[°C]",
    "Capacity_Step[Ah]",
]

THERMAL_COLUMNS = [
    "Testtime[s]",
    "StepID",
    "Voltage[V]",
    "Current[A]",
    "Temperature[°C]",
]


def clean_float(value: object, missing_minus_one: bool = True) -> float | None:
    if value is None:
        return None
    try:
        result = float(str(value).strip().replace("C", ""))
    except ValueError:
        return None
    if missing_minus_one and math.isclose(result, -1.0, abs_tol=1e-9):
        return None
    return result


def consecutive_groups(frame: pd.DataFrame) -> Iterable[pd.DataFrame]:
    if frame.empty:
        return []
    groups = frame.index.to_series().diff().ne(1).cumsum()
    return (group for _, group in frame.groupby(groups))


def segment_last_values(frame: pd.DataFrame, step_id: int) -> list[dict[str, float]]:
    subset = frame[frame["StepID"] == step_id]
    rows: list[dict[str, float]] = []
    for group in consecutive_groups(subset):
        capacity = group["Capacity_Step[Ah]"].dropna()
        voltage = group["Voltage[V]"].dropna()
        temperature = group["Temperature[°C]"].dropna()
        testtime = group["Testtime[s]"].dropna()
        rows.append(
            {
                "capacity_ah": abs(float(capacity.iloc[-1])) if not capacity.empty else np.nan,
                "voltage_v": float(voltage.iloc[-1]) if not voltage.empty else np.nan,
                "temperature_mean_deg_c": float(temperature.mean()) if not temperature.empty else np.nan,
                "duration_s": float(testtime.iloc[-1] - testtime.iloc[0]) if len(testtime) > 1 else np.nan,
                "n_rows": float(len(group)),
            }
        )
    return rows


def extract_hppc(zip_path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    members = [member for member in iter_members(zip_path, {".csv"}) if member.startswith("data_Ri_HPPC/")]
    for member in members:
        frame = read_csv_member(zip_path, member, sep=";")
        frame.columns = [column.strip() for column in frame.columns]
        meta = normal_metadata(member)
        direction_from_file = hppc_file_direction(member)
        columns = list(frame.columns)
        if not columns:
            continue
        soc_column = columns[0]
        group_width = 7
        for _, row in frame.iterrows():
            soc = clean_float(row.get(soc_column))
            for start in range(1, len(columns), group_width):
                group = columns[start : start + group_width]
                if len(group) < group_width:
                    continue
                c_rate = clean_float(row.get(group[0]), missing_minus_one=False)
                if c_rate is None:
                    continue
                records.append(
                    {
                        **meta,
                        "source_file_direction": direction_from_file,
                        "pulse_direction": "charge" if c_rate > 0 else "discharge",
                        "c_rate": c_rate,
                        "soc_fraction": soc,
                        "soc_percent": soc * 100 if soc is not None else None,
                        "r_fast_ohm": clean_float(row.get(group[1])),
                        "dt_fast_s": clean_float(row.get(group[2])),
                        "r_100ms_ohm": clean_float(row.get(group[3])),
                        "dt_100ms_s": clean_float(row.get(group[4])),
                        "r_1s_ohm": clean_float(row.get(group[5])),
                        "dt_1s_s": clean_float(row.get(group[6])),
                    }
                )
    return pd.DataFrame.from_records(records)


def add_capacity_retention(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output["discharge_capacity_retention_vs_25c"] = np.nan
    output["charge_capacity_retention_vs_25c"] = np.nan
    for _, group in output.groupby(["cell_id", "cell_family"], dropna=False):
        ref = group[group["temperature_deg_c"] == 25]
        if ref.empty:
            continue
        ref_dis = float(ref["discharge_capacity_ah"].iloc[0])
        ref_ch = float(ref["charge_capacity_ah"].iloc[0])
        idx = group.index
        if ref_dis:
            output.loc[idx, "discharge_capacity_retention_vs_25c"] = output.loc[idx, "discharge_capacity_ah"] / ref_dis
        if ref_ch:
            output.loc[idx, "charge_capacity_retention_vs_25c"] = output.loc[idx, "charge_capacity_ah"] / ref_ch
    return output


def extract_capacity_and_temperature_ocv(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    capacity_records: list[dict[str, object]] = []
    ocv_records: list[dict[str, object]] = []
    members = [member for member in iter_members(zip_path, {".parquet"}) if member.startswith("raw_CU_diffT/")]
    for member in members:
        meta = normal_metadata(member)
        frame = read_parquet_member(zip_path, member, columns=CHECKUP_COLUMNS)
        dis_cap = segment_last_values(frame, 96)
        dis_ocv = segment_last_values(frame, 97)
        ch_cap = segment_last_values(frame, 106)
        ch_ocv = segment_last_values(frame, 107)

        dis_total = float(np.nansum([row["capacity_ah"] for row in dis_cap]))
        ch_total = float(np.nansum([row["capacity_ah"] for row in ch_cap]))
        capacity_records.append(
            {
                **meta,
                "ambient_temperature_from_file_deg_c": meta["temperature_deg_c"],
                "actual_temperature_mean_deg_c": float(frame["Temperature[°C]"].dropna().mean()),
                "discharge_capacity_ah": dis_total,
                "charge_capacity_ah": ch_total,
                "n_discharge_capacity_segments": len(dis_cap),
                "n_charge_capacity_segments": len(ch_cap),
                "source_step_ids": "discharge_capacity=96;discharge_ocv=97;charge_capacity=106;charge_ocv=107",
            }
        )

        for direction, cap_rows, ocv_rows, total in [
            ("discharge", dis_cap, dis_ocv, dis_total),
            ("charge", ch_cap, ch_ocv, ch_total),
        ]:
            cumulative = 0.0
            for segment_index, cap_row in enumerate(cap_rows):
                cumulative += cap_row["capacity_ah"]
                ocv_row = ocv_rows[segment_index] if segment_index < len(ocv_rows) else {}
                ocv_records.append(
                    {
                        **meta,
                        "ambient_temperature_from_file_deg_c": meta["temperature_deg_c"],
                        "direction": direction,
                        "segment_index": segment_index,
                        "segment_capacity_ah": cap_row["capacity_ah"],
                        "cumulative_capacity_ah": cumulative,
                        "soc_fraction_by_capacity": cumulative / total if total else np.nan,
                        "ocv_voltage_v": ocv_row.get("voltage_v", np.nan),
                        "ocv_temperature_mean_deg_c": ocv_row.get("temperature_mean_deg_c", np.nan),
                    }
                )

    return add_capacity_retention(pd.DataFrame.from_records(capacity_records)), pd.DataFrame.from_records(ocv_records)


def extract_ocv_current_rate_curves(zip_path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    step_to_rate = {6: 0.25, 12: 0.50, 18: 1.00, 24: 2.00, 30: 3.00}
    members = [member for member in iter_members(zip_path, {".parquet"}) if member.startswith("raw_OCV_I_curves/")]
    for member in members:
        meta = normal_metadata(member)
        frame = read_parquet_member(zip_path, member, columns=CHECKUP_COLUMNS)
        for step_id, c_rate in step_to_rate.items():
            subset = frame[frame["StepID"] == step_id].copy()
            if subset.empty:
                continue
            capacity = subset["Capacity_Step[Ah]"].astype(float)
            voltage = subset["Voltage[V]"].astype(float)
            temperature = subset["Temperature[°C]"].astype(float)
            records.append(
                {
                    **meta,
                    "step_id": step_id,
                    "c_rate": c_rate,
                    "n_points": len(subset),
                    "capacity_span_ah": float(abs(capacity.iloc[-1] - capacity.iloc[0])),
                    "voltage_start_v": float(voltage.iloc[0]),
                    "voltage_end_v": float(voltage.iloc[-1]),
                    "temperature_mean_deg_c": float(temperature.mean()),
                    "temperature_span_deg_c": float(temperature.max() - temperature.min()),
                }
            )
    return pd.DataFrame.from_records(records)


def extract_eis(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    eis_records: list[dict[str, object]] = []
    drt_records: list[dict[str, object]] = []
    members = [member for member in iter_members(zip_path, {".csv"}) if member.startswith("raw_EIS_plotting/")]
    for member in members:
        meta = normal_metadata(member)
        if member.endswith("_EIS.csv"):
            frame = read_csv_member(zip_path, member)
            valid = frame.dropna(subset=["EISFreq[Hz]", "Zre[Ohm]", "Zim[Ohm]"]).copy()
            if valid.empty:
                continue
            valid = valid.sort_values("EISFreq[Hz]")
            low = valid.iloc[0]
            high = valid.iloc[-1]
            min_abs_zim = valid.loc[valid["Zim[Ohm]"].abs().idxmin()]
            most_negative_zim = valid.loc[valid["Zim[Ohm]"].idxmin()]
            eis_records.append(
                {
                    **meta,
                    "n_points": len(valid),
                    "freq_min_hz": float(valid["EISFreq[Hz]"].min()),
                    "freq_max_hz": float(valid["EISFreq[Hz]"].max()),
                    "voltage_mean_v": float(valid["Voltage[V]"].dropna().mean()),
                    "temperature_mean_deg_c": float(valid["Temperature[°C]"].dropna().mean()),
                    "series_resistance_high_freq_ohm": float(high["Zre[Ohm]"]),
                    "series_resistance_min_abs_zim_ohm": float(min_abs_zim["Zre[Ohm]"]),
                    "freq_at_min_abs_zim_hz": float(min_abs_zim["EISFreq[Hz]"]),
                    "low_freq_zabs_ohm": float(low["Zabs[Ohm]"]),
                    "low_freq_zre_ohm": float(low["Zre[Ohm]"]),
                    "low_freq_zim_ohm": float(low["Zim[Ohm]"]),
                    "arc_width_zre_ohm": float(valid["Zre[Ohm]"].max() - valid["Zre[Ohm]"].min()),
                    "min_zim_ohm": float(valid["Zim[Ohm]"].min()),
                    "characteristic_freq_min_zim_hz": float(most_negative_zim["EISFreq[Hz]"]),
                }
            )
        elif member.endswith("_EISDRT.csv"):
            frame = read_csv_member(zip_path, member)
            valid = frame.dropna(subset=["Tau", "Gamma"]).copy()
            if valid.empty:
                continue
            max_gamma = valid.loc[valid["Gamma"].idxmax()]
            drt_records.append(
                {
                    **meta,
                    "n_points": len(valid),
                    "tau_min_s": float(valid["Tau"].min()),
                    "tau_max_s": float(valid["Tau"].max()),
                    "gamma_max": float(max_gamma["Gamma"]),
                    "tau_at_gamma_max_s": float(max_gamma["Tau"]),
                    "gamma_integral_logtau": float(_trapezoid(valid["Gamma"], np.log(valid["Tau"]))),
                }
            )
    return pd.DataFrame.from_records(eis_records), pd.DataFrame.from_records(drt_records)


def extract_entropy(zip_path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    members = [member for member in iter_members(zip_path, {".npy"}) if member.startswith("data_dUdT/")]
    for member in members:
        meta = normal_metadata(member)
        direction = entropy_direction(member)
        array = read_npy_member(zip_path, member).astype(float)
        soc_values = np.linspace(100, 0, len(array), dtype=int)
        for soc_percent, value in zip(soc_values, array):
            records.append(
                {
                    **meta,
                    "direction": direction,
                    "soc_percent": int(soc_percent),
                    "dudt_v_per_k": float(value),
                    "dudt_mv_per_k": float(value * 1000.0),
                    "entropic_heat_sensitivity_v_at_25c": float(abs(298.15 * value)),
                }
            )
    return pd.DataFrame.from_records(records)


def extract_thermal_response(zip_path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    members = [
        member
        for member in iter_members(zip_path, {".parquet"})
        if member.startswith("raw_temperature_BOL/") and member.endswith("_dS025C.parquet")
    ]
    for member in members:
        meta = normal_metadata(member)
        frame = read_parquet_member(zip_path, member, columns=THERMAL_COLUMNS)
        for direction, step_ids in [("charge", [6, 7]), ("discharge", [9, 10])]:
            subset = frame[frame["StepID"].isin(step_ids)].copy()
            if subset.empty:
                continue
            temp = subset["Temperature[°C]"].dropna().astype(float)
            current = subset["Current[A]"].dropna().astype(float)
            time = subset["Testtime[s]"].dropna().astype(float)
            voltage = subset["Voltage[V]"].dropna().astype(float)
            records.append(
                {
                    **meta,
                    "direction": direction,
                    "step_ids": ",".join(str(step_id) for step_id in step_ids),
                    "n_points": len(subset),
                    "duration_s": float(time.iloc[-1] - time.iloc[0]) if len(time) > 1 else np.nan,
                    "temperature_start_deg_c": float(temp.iloc[0]) if not temp.empty else np.nan,
                    "temperature_max_deg_c": float(temp.max()) if not temp.empty else np.nan,
                    "temperature_rise_max_k": float(temp.max() - temp.iloc[0]) if not temp.empty else np.nan,
                    "mean_abs_current_a": float(current.abs().mean()) if not current.empty else np.nan,
                    "mean_voltage_v": float(voltage.mean()) if not voltage.empty else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def build_cell_stream_availability(inventory_csv: Path) -> pd.DataFrame:
    with inventory_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        cell_id = row.get("cell_id") or ""
        family = row.get("cell_family") or ""
        chemistry = row.get("chemistry_type") or ""
        stream = row.get("measurement_stream") or "unknown"
        if not family and not cell_id:
            continue
        counts[(cell_id, family, chemistry, stream)]["files"] += 1

    records = [
        {
            "cell_id": cell_id,
            "cell_family": family,
            "chemistry_type": chemistry,
            "measurement_stream": stream,
            "file_count": counter["files"],
        }
        for (cell_id, family, chemistry, stream), counter in sorted(counts.items())
    ]
    return pd.DataFrame.from_records(records)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_report(path: Path, outputs: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Initial Descriptor Extraction Report", ""]
    for name, frame in outputs.items():
        lines.append(f"- {name}: {len(frame)} rows, {len(frame.columns)} columns")
    lines.extend(["", "## Notes", ""])
    lines.append("- Raw files were read directly from `data_raw/data_EvalSIB.zip`; the archive was not extracted or modified.")
    lines.append("- Temperature checkup capacity uses StepIDs 96/97 and 106/107, following the upstream `OCV_T_plotting.py` script.")
    lines.append("- OCV current-rate summaries use StepIDs 6, 12, 18, 24, and 30, following `OCV_I_plotting.py`.")
    lines.append("- Thermal summaries use dS025C StepIDs 6-7 and 9-10, following `temperature_plotting.py`.")
    lines.append("- EIS descriptors are operational spectral descriptors, not equivalent-circuit claims.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="data_raw/data_EvalSIB.zip")
    parser.add_argument("--inventory-csv", default="docs/data_inventory.csv")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    outputs: dict[str, pd.DataFrame] = {}

    outputs["cell_stream_availability"] = build_cell_stream_availability(Path(args.inventory_csv))
    outputs["hppc_resistance_long"] = extract_hppc(zip_path)
    capacity, ocv_temperature = extract_capacity_and_temperature_ocv(zip_path)
    outputs["capacity_temperature_summary"] = capacity
    outputs["ocv_temperature_points"] = ocv_temperature
    outputs["ocv_current_rate_summary"] = extract_ocv_current_rate_curves(zip_path)
    eis, drt = extract_eis(zip_path)
    outputs["eis_spectral_descriptors"] = eis
    outputs["drt_descriptors"] = drt
    outputs["entropy_dudt_long"] = extract_entropy(zip_path)
    outputs["thermal_response_descriptors"] = extract_thermal_response(zip_path)

    write_frame(outputs["cell_stream_availability"], Path("data_processed/qc/cell_stream_availability.csv"))
    write_frame(outputs["hppc_resistance_long"], Path("data_processed/hppc/hppc_resistance_long.csv"))
    write_frame(outputs["capacity_temperature_summary"], Path("data_processed/capacity/capacity_temperature_summary.csv"))
    write_frame(outputs["ocv_temperature_points"], Path("data_processed/ocv/ocv_temperature_points.csv"))
    write_frame(outputs["ocv_current_rate_summary"], Path("data_processed/ocv/ocv_current_rate_summary.csv"))
    write_frame(outputs["eis_spectral_descriptors"], Path("data_processed/eis/eis_spectral_descriptors.csv"))
    write_frame(outputs["drt_descriptors"], Path("data_processed/eis/drt_descriptors.csv"))
    write_frame(outputs["entropy_dudt_long"], Path("data_processed/entropy/entropy_dudt_long.csv"))
    write_frame(outputs["thermal_response_descriptors"], Path("data_processed/thermal/thermal_response_descriptors.csv"))

    write_report(Path("logs/initial_descriptor_extraction_report.md"), outputs)
    summary = {name: {"rows": len(frame), "columns": list(frame.columns)} for name, frame in outputs.items()}
    Path("data_processed/qc/initial_descriptor_outputs.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
