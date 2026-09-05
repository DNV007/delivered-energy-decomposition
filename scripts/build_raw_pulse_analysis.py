#!/usr/bin/env python3
"""Build tables and figures from temperature-resolved raw pulse descriptors."""

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
import pandas as pd


# Palette, markers and rcParams are shared with the manuscript figures so
# that a diagnostic plot and its published counterpart cannot drift apart.
from figure_style import COLORS, MARKERS, use_style

use_style()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def nanmedian_mohm(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(np.nanmedian(numeric) * 1000.0)


def build_temperature_summary(pulses: pd.DataFrame) -> pd.DataFrame:
    frame = pulses.copy()
    frame["abs_c_rate"] = frame["nominal_c_rate"].abs()
    frame["r_1s_complete_ohm"] = frame["r_1s_ohm"].where(frame["complete_1s"])
    frame["r_10s_complete_ohm"] = frame["r_10s_ohm"].where(frame["complete_10s"])
    frame["polarization_1s_complete_ohm"] = frame["polarization_1s_minus_fast_ohm"].where(frame["complete_1s"])
    frame["polarization_10s_complete_ohm"] = (frame["r_10s_ohm"] - frame["r_fast_ohm"]).where(
        frame["complete_10s"]
    )

    summary = (
        frame.groupby(
            ["cell_family", "chemistry_type", "temperature_deg_c", "pulse_direction", "abs_c_rate"],
            dropna=False,
        )
        .agg(
            pulse_count=("member_path", "count"),
            complete_1s_count=("complete_1s", "sum"),
            complete_10s_count=("complete_10s", "sum"),
            median_actual_temperature_deg_c=("temperature_mean_deg_c", "median"),
            median_r_fast_mohm=("r_fast_ohm", nanmedian_mohm),
            median_r_100ms_mohm=("r_100ms_ohm", nanmedian_mohm),
            median_r_1s_mohm=("r_1s_complete_ohm", nanmedian_mohm),
            median_r_10s_mohm=("r_10s_complete_ohm", nanmedian_mohm),
            median_r_end_mohm=("r_end_ohm", nanmedian_mohm),
            median_polarization_1s_mohm=("polarization_1s_complete_ohm", nanmedian_mohm),
            median_polarization_10s_mohm=("polarization_10s_complete_ohm", nanmedian_mohm),
            median_polarization_end_mohm=("polarization_end_minus_fast_ohm", nanmedian_mohm),
            median_relaxation_end_mohm=("relaxation_end_ohm", nanmedian_mohm),
        )
        .reset_index()
        .sort_values(["cell_family", "temperature_deg_c", "pulse_direction", "abs_c_rate"])
    )
    return summary


def build_soc_temperature_map(pulses: pd.DataFrame) -> pd.DataFrame:
    frame = pulses.copy()
    frame["abs_c_rate"] = frame["nominal_c_rate"].abs()
    frame["r_1s_complete_ohm"] = frame["r_1s_ohm"].where(frame["complete_1s"])
    frame["r_10s_complete_ohm"] = frame["r_10s_ohm"].where(frame["complete_10s"])
    frame["polarization_1s_complete_ohm"] = frame["polarization_1s_minus_fast_ohm"].where(frame["complete_1s"])

    soc_map = (
        frame.groupby(
            [
                "cell_family",
                "chemistry_type",
                "temperature_deg_c",
                "pulse_direction",
                "abs_c_rate",
                "nominal_soc_percent",
            ],
            dropna=False,
        )
        .agg(
            pulse_count=("member_path", "count"),
            complete_1s_count=("complete_1s", "sum"),
            complete_10s_count=("complete_10s", "sum"),
            median_r_fast_mohm=("r_fast_ohm", nanmedian_mohm),
            median_r_1s_mohm=("r_1s_complete_ohm", nanmedian_mohm),
            median_r_10s_mohm=("r_10s_complete_ohm", nanmedian_mohm),
            median_polarization_1s_mohm=("polarization_1s_complete_ohm", nanmedian_mohm),
        )
        .reset_index()
        .sort_values(
            [
                "cell_family",
                "temperature_deg_c",
                "pulse_direction",
                "abs_c_rate",
                "nominal_soc_percent",
            ]
        )
    )
    return soc_map


def build_sib_vs_li_reference(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["temperature_deg_c", "pulse_direction", "abs_c_rate"]
    sib = summary[summary["cell_family"] == "SIB"][
        keys + ["median_r_1s_mohm", "median_r_fast_mohm", "median_polarization_1s_mohm"]
    ].rename(
        columns={
            "median_r_1s_mohm": "sib_median_r_1s_mohm",
            "median_r_fast_mohm": "sib_median_r_fast_mohm",
            "median_polarization_1s_mohm": "sib_median_polarization_1s_mohm",
        }
    )
    li = (
        summary[summary["cell_family"].isin(["LFP", "LTO", "NMC"])]
        .groupby(keys, dropna=False)
        .agg(
            li_reference_family_count=("cell_family", "nunique"),
            li_reference_median_r_1s_mohm=("median_r_1s_mohm", "mean"),
            li_reference_median_r_fast_mohm=("median_r_fast_mohm", "mean"),
            li_reference_median_polarization_1s_mohm=("median_polarization_1s_mohm", "mean"),
        )
        .reset_index()
    )
    comparison = sib.merge(li, on=keys, how="left").sort_values(keys)
    comparison["sib_to_li_r_1s_ratio"] = (
        comparison["sib_median_r_1s_mohm"] / comparison["li_reference_median_r_1s_mohm"]
    )
    comparison["sib_to_li_r_fast_ratio"] = (
        comparison["sib_median_r_fast_mohm"] / comparison["li_reference_median_r_fast_mohm"]
    )

    comparison["sib_r_1s_temperature_factor_vs_25c"] = np.nan
    comparison["li_reference_r_1s_temperature_factor_vs_25c"] = np.nan
    for _, group in comparison.groupby(["pulse_direction", "abs_c_rate"], dropna=False):
        ref = group[group["temperature_deg_c"] == 25]
        if ref.empty:
            continue
        sib_ref = float(ref["sib_median_r_1s_mohm"].iloc[0])
        li_ref = float(ref["li_reference_median_r_1s_mohm"].iloc[0])
        idx = group.index
        if sib_ref:
            comparison.loc[idx, "sib_r_1s_temperature_factor_vs_25c"] = (
                comparison.loc[idx, "sib_median_r_1s_mohm"] / sib_ref
            )
        if li_ref:
            comparison.loc[idx, "li_reference_r_1s_temperature_factor_vs_25c"] = (
                comparison.loc[idx, "li_reference_median_r_1s_mohm"] / li_ref
            )
    return comparison


def plot_r1s_temperature(summary: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    subset = summary[(summary["abs_c_rate"] == 1.0) & summary["median_r_1s_mohm"].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, direction in zip(axes, ["discharge", "charge"]):
        view = subset[subset["pulse_direction"] == direction]
        for family, group in view.groupby("cell_family"):
            group = group.sort_values("temperature_deg_c")
            axis.plot(
                group["temperature_deg_c"],
                group["median_r_1s_mohm"],
                marker="o",
                label=family,
                color=COLORS.get(family, "black"),
            )
        axis.set_title(f"1C {direction}")
        axis.set_xlabel("Temperature (deg C)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Median R_1s (mohm)")
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_soc_temperature_maps(soc_map: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    subset = soc_map[
        (soc_map["pulse_direction"] == "discharge")
        & (soc_map["abs_c_rate"] == 1.0)
        & soc_map["median_r_1s_mohm"].notna()
    ].copy()
    families = ["LFP", "LTO", "NMC", "SIB"]
    temperatures = sorted(subset["temperature_deg_c"].dropna().unique())
    soc_values = sorted(subset["nominal_soc_percent"].dropna().unique(), reverse=True)
    vmax = float(np.nanpercentile(subset["median_r_1s_mohm"], 98))
    vmin = float(np.nanpercentile(subset["median_r_1s_mohm"], 2))

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    last_image = None
    for axis, family in zip(axes.ravel(), families):
        view = subset[subset["cell_family"] == family]
        pivot = (
            view.pivot_table(
                index="nominal_soc_percent",
                columns="temperature_deg_c",
                values="median_r_1s_mohm",
                aggfunc="median",
            )
            .reindex(index=soc_values, columns=temperatures)
            .to_numpy(dtype=float)
        )
        last_image = axis.imshow(pivot, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(family)
        axis.set_xticks(np.arange(len(temperatures)))
        axis.set_xticklabels([f"{int(temp)}" for temp in temperatures])
        axis.set_yticks(np.arange(len(soc_values)))
        axis.set_yticklabels([f"{int(soc)}" for soc in soc_values])
        axis.set_xlabel("Temperature (deg C)")
        axis.set_ylabel("Protocol SOC (%)")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), label="Median R_1s (mohm)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(path: Path, comparison: pd.DataFrame) -> None:
    ensure_parent(path)
    lines = [
        "# Raw Pulse Findings",
        "",
        "These descriptors come from segmented raw checkup pulse responses, not the processed HPPC CSV files.",
        "",
        "## Immediate Signals",
        "",
    ]
    subset = comparison[(comparison["pulse_direction"] == "discharge") & (comparison["abs_c_rate"] == 1.0)]
    sib_25 = subset[subset["temperature_deg_c"] == 25]
    sib_5 = subset[subset["temperature_deg_c"] == 5]
    if not sib_25.empty:
        row = sib_25.iloc[0]
        lines.append(
            f"- SIB 1C discharge median R_1s at 25 deg C is {row['sib_median_r_1s_mohm']:.1f} mohm; "
            f"the Li-ion reference mean is {row['li_reference_median_r_1s_mohm']:.1f} mohm "
            f"(SIB/Li ratio {row['sib_to_li_r_1s_ratio']:.2f})."
        )
    if not sib_5.empty:
        row = sib_5.iloc[0]
        lines.append(
            f"- SIB 1C discharge median R_1s at 5 deg C is {row['sib_median_r_1s_mohm']:.1f} mohm, "
            f"{row['sib_r_1s_temperature_factor_vs_25c']:.2f}x its 25 deg C value."
        )
    lines.extend(
        [
            "",
            "## Cautions",
            "",
            "- Pulse SOC is a protocol index assigned from pulse-block order, not an independently integrated SOC estimate.",
            "- Clipped pulses are retained in the raw table but excluded from 1 s or 10 s summaries using completeness flags.",
            "- These are operational resistance descriptors; equivalent-circuit fitting has been checked separately and rejected for core interpretation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_processed/hppc/raw_temperature_pulse_descriptors.csv")
    args = parser.parse_args()

    pulses = pd.read_csv(args.input)
    summary = build_temperature_summary(pulses)
    soc_map = build_soc_temperature_map(pulses)
    comparison = build_sib_vs_li_reference(summary)

    outputs = {
        "results/tables/table_raw_pulse_temperature_summary.csv": summary,
        "results/tables/table_raw_pulse_soc_temperature_map.csv": soc_map,
        "results/tables/table_raw_pulse_sib_vs_li_reference.csv": comparison,
    }
    for file_name, frame in outputs.items():
        output_path = Path(file_name)
        ensure_parent(output_path)
        frame.to_csv(output_path, index=False)
        print(f"{file_name}: {len(frame)} rows")

    plot_r1s_temperature(summary, Path("results/figures/fig05_raw_pulse_r1s_temperature.png"))
    plot_soc_temperature_maps(soc_map, Path("results/figures/fig06_raw_pulse_soc_temperature_map.png"))
    write_report(Path("results/reports/raw_pulse_findings.md"), comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
