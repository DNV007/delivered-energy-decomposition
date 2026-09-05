#!/usr/bin/env python3
"""Build smoothed OCV-slope descriptors and OCV slope penalty tables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-energystorage")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/energystorage-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# numpy 2.0 renamed trapz -> trapezoid; keep both working so the pipeline
# runs on the pinned stack and on newer numpy alike.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
import pandas as pd


SMOOTHING_WINDOWS = [1, 5, 11]
CAPACITY_WINDOWS = {
    "full": (0.0, 1.0),
    "early_0_20": (0.0, 0.2),
    "central_20_80": (0.2, 0.8),
    "late_80_100": (0.8, 1.0),
}
# Palette, markers and rcParams are shared with the manuscript figures so
# that a diagnostic plot and its published counterpart cannot drift apart.
from figure_style import COLORS, MARKERS, use_style

use_style()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def smoothed_derivative(group: pd.DataFrame, smoothing_window: int) -> pd.DataFrame:
    ordered = (
        group.dropna(subset=["soc_fraction_by_capacity", "ocv_voltage_v"])
        .sort_values("soc_fraction_by_capacity")
        .copy()
    )
    if ordered.empty:
        return ordered
    ordered["smoothing_window_segments"] = smoothing_window
    ordered["smoothed_ocv_voltage_v"] = (
        ordered["ocv_voltage_v"].rolling(smoothing_window, center=True, min_periods=1).mean()
    )
    x = ordered["soc_fraction_by_capacity"].to_numpy(dtype=float)
    y = ordered["smoothed_ocv_voltage_v"].to_numpy(dtype=float)
    if len(ordered) >= 3 and np.nanmax(x) > np.nanmin(x):
        slope = np.gradient(y, x)
    else:
        slope = np.full(len(ordered), np.nan)
    ordered["dv_dcapacity_fraction_v"] = slope
    ordered["abs_dv_dcapacity_fraction_v"] = np.abs(slope)
    return ordered


def build_slope_points(ocv_points: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_columns = ["cell_id", "cell_family", "chemistry_type", "temperature_deg_c", "direction"]
    for _, group in ocv_points.groupby(group_columns, dropna=False):
        for smoothing_window in SMOOTHING_WINDOWS:
            records.append(smoothed_derivative(group, smoothing_window))
    return pd.concat(records, ignore_index=True)


def window_view(points: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    if lower == 0.0 and upper == 1.0:
        return points[(points["soc_fraction_by_capacity"] >= lower) & (points["soc_fraction_by_capacity"] <= upper)]
    return points[(points["soc_fraction_by_capacity"] > lower) & (points["soc_fraction_by_capacity"] <= upper)]


def integrate_abs_slope(window: pd.DataFrame) -> float:
    clean = window.dropna(subset=["soc_fraction_by_capacity", "abs_dv_dcapacity_fraction_v"]).sort_values(
        "soc_fraction_by_capacity"
    )
    if len(clean) < 2:
        return np.nan
    return float(_trapezoid(clean["abs_dv_dcapacity_fraction_v"], clean["soc_fraction_by_capacity"]))


def voltage_span(window: pd.DataFrame) -> float:
    clean = window.dropna(subset=["smoothed_ocv_voltage_v"]).sort_values("soc_fraction_by_capacity")
    if len(clean) < 2:
        return np.nan
    return float(clean["smoothed_ocv_voltage_v"].iloc[-1] - clean["smoothed_ocv_voltage_v"].iloc[0])


def build_slope_descriptors(slope_points: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_columns = [
        "cell_id",
        "cell_family",
        "chemistry_type",
        "temperature_deg_c",
        "direction",
        "smoothing_window_segments",
    ]
    for keys, group in slope_points.groupby(group_columns, dropna=False):
        key_record = dict(zip(group_columns, keys))
        for capacity_window, (lower, upper) in CAPACITY_WINDOWS.items():
            view = window_view(group, lower, upper)
            if view.empty:
                continue
            width = upper - lower
            integrated_abs = integrate_abs_slope(view)
            records.append(
                {
                    **key_record,
                    "capacity_window": capacity_window,
                    "window_lower_fraction": lower,
                    "window_upper_fraction": upper,
                    "window_width_fraction": width,
                    "n_points": int(len(view)),
                    "ocv_slope_penalty_v": integrated_abs,
                    "mean_abs_dv_dcapacity_fraction_v": float(view["abs_dv_dcapacity_fraction_v"].mean()),
                    "median_abs_dv_dcapacity_fraction_v": float(view["abs_dv_dcapacity_fraction_v"].median()),
                    "max_abs_dv_dcapacity_fraction_v": float(view["abs_dv_dcapacity_fraction_v"].max()),
                    "mean_dv_dcapacity_fraction_v": float(view["dv_dcapacity_fraction_v"].mean()),
                    "voltage_span_v": voltage_span(view),
                    "mean_smoothed_ocv_v": float(view["smoothed_ocv_voltage_v"].mean()),
                }
            )
    output = pd.DataFrame.from_records(records)
    output["ocv_slope_penalty_vs_25c"] = np.nan
    for _, group in output.groupby(
        ["cell_family", "direction", "smoothing_window_segments", "capacity_window"], dropna=False
    ):
        ref = group[group["temperature_deg_c"] == 25.0]
        if ref.empty:
            continue
        ref_value = float(ref["ocv_slope_penalty_v"].iloc[0])
        if ref_value:
            output.loc[group.index, "ocv_slope_penalty_vs_25c"] = group["ocv_slope_penalty_v"] / ref_value
    return output.sort_values(
        ["cell_family", "direction", "smoothing_window_segments", "capacity_window", "temperature_deg_c"]
    ).reset_index(drop=True)


def build_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    return descriptors[
        (descriptors["smoothing_window_segments"] == 5)
        & (descriptors["capacity_window"] == "central_20_80")
    ].sort_values(["cell_family", "direction", "temperature_deg_c"])


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, direction in zip(axes, ["discharge", "charge"]):
        view = summary[summary["direction"] == direction]
        for family, group in view.groupby("cell_family"):
            ordered = group.sort_values("temperature_deg_c")
            axis.plot(
                ordered["temperature_deg_c"],
                ordered["ocv_slope_penalty_v"],
                marker="o",
                linewidth=2,
                label=family,
                color=COLORS.get(family, "black"),
            )
        axis.set_title(direction)
        axis.set_xlabel("Temperature (deg C)")
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Central OCV slope penalty (V)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame) -> None:
    ensure_parent(path)
    discharge_5 = summary[(summary["direction"] == "discharge") & (summary["temperature_deg_c"] == 5.0)]
    discharge_5 = discharge_5.sort_values("ocv_slope_penalty_v", ascending=False)
    sib = discharge_5[discharge_5["cell_family"] == "SIB"]
    li = discharge_5[discharge_5["cell_family"].isin(["LFP", "LTO", "NMC"])]
    if not sib.empty and not li.empty:
        sib_penalty = float(sib["ocv_slope_penalty_v"].iloc[0])
        li_mean = float(li["ocv_slope_penalty_v"].mean())
        ratio = sib_penalty / li_mean if li_mean else np.nan
        comparison = (
            f"SIB central discharge OSP at 5 deg C is {sib_penalty:.3f} V, "
            f"versus a Li-ion reference mean of {li_mean:.3f} V (SIB/Li={ratio:.2f})."
        )
    else:
        comparison = "SIB-vs-Li central discharge OSP comparison is unavailable."

    top_family = discharge_5["cell_family"].iloc[0] if not discharge_5.empty else "n/a"
    lines = [
        "# OCV Slope Analysis",
        "",
        "OCV curves were smoothed with centered rolling windows of 1, 5, and 11 capacity segments. The primary OCV slope penalty (OSP) uses the 5-segment smoothed curve and integrates abs(dV/dcapacity-fraction) over the central 20-80% capacity window.",
        "",
        f"5 deg C central discharge OSP ranking is led by {top_family}.",
        comparison,
        "",
        "Interpretation: OSP is a thermodynamic/voltage-shape descriptor. It should be used as context beside capacity, raw-pulse, and EIS descriptors rather than as a standalone kinetic metric.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    ocv_points = pd.read_csv("data_processed/ocv/ocv_temperature_points.csv")
    slope_points = build_slope_points(ocv_points)
    descriptors = build_slope_descriptors(slope_points)
    summary = build_summary(descriptors)

    outputs = {
        "data_processed/ocv/ocv_slope_points.csv": slope_points,
        "data_processed/ocv/ocv_slope_descriptors.csv": descriptors,
        "results/tables/table_ocv_slope_summary.csv": summary,
    }
    for file_name, frame in outputs.items():
        output_path = Path(file_name)
        ensure_parent(output_path)
        frame.to_csv(output_path, index=False)
        print(f"{file_name}: {len(frame)} rows")

    plot_summary(summary, Path("results/figures/fig12_ocv_slope_penalty.png"))
    write_report(Path("results/reports/ocv_slope_analysis.md"), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
