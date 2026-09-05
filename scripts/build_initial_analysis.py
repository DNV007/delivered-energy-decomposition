#!/usr/bin/env python3
"""Build first analysis tables and QC figures from extracted descriptors."""

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


def build_usable_energy(ocv_points: pd.DataFrame) -> pd.DataFrame:
    clean = ocv_points.dropna(subset=["segment_capacity_ah", "ocv_voltage_v"]).copy()
    clean["segment_energy_wh"] = clean["segment_capacity_ah"] * clean["ocv_voltage_v"]
    grouped = (
        clean.groupby(
            [
                "cell_id",
                "cell_family",
                "chemistry_type",
                "temperature_deg_c",
                "ambient_temperature_from_file_deg_c",
                "direction",
            ],
            dropna=False,
        )
        .agg(
            usable_energy_wh=("segment_energy_wh", "sum"),
            capacity_from_segments_ah=("segment_capacity_ah", "sum"),
            mean_ocv_v=("ocv_voltage_v", "mean"),
            n_segments=("segment_index", "count"),
        )
        .reset_index()
    )
    grouped["energy_retention_vs_25c"] = np.nan
    grouped["capacity_retention_from_segments_vs_25c"] = np.nan
    for _, group in grouped.groupby(["cell_id", "cell_family", "direction"], dropna=False):
        ref = group[group["temperature_deg_c"] == 25]
        if ref.empty:
            continue
        ref_energy = float(ref["usable_energy_wh"].iloc[0])
        ref_capacity = float(ref["capacity_from_segments_ah"].iloc[0])
        idx = group.index
        if ref_energy:
            grouped.loc[idx, "energy_retention_vs_25c"] = grouped.loc[idx, "usable_energy_wh"] / ref_energy
        if ref_capacity:
            grouped.loc[idx, "capacity_retention_from_segments_vs_25c"] = (
                grouped.loc[idx, "capacity_from_segments_ah"] / ref_capacity
            )
    return grouped


def build_capacity_energy_table(capacity: pd.DataFrame, usable_energy: pd.DataFrame) -> pd.DataFrame:
    discharge_energy = usable_energy[usable_energy["direction"] == "discharge"].copy()
    keep_energy = discharge_energy[
        [
            "cell_id",
            "cell_family",
            "temperature_deg_c",
            "usable_energy_wh",
            "energy_retention_vs_25c",
            "mean_ocv_v",
        ]
    ]
    merged = capacity.merge(
        keep_energy,
        on=["cell_id", "cell_family", "temperature_deg_c"],
        how="left",
    )
    return merged[
        [
            "cell_family",
            "chemistry_type",
            "cell_id",
            "temperature_deg_c",
            "actual_temperature_mean_deg_c",
            "discharge_capacity_ah",
            "discharge_capacity_retention_vs_25c",
            "usable_energy_wh",
            "energy_retention_vs_25c",
            "mean_ocv_v",
        ]
    ].sort_values(["cell_family", "temperature_deg_c"])


# Nominal room-temperature window admitted to the EIS family summary. The
# archive's room-temperature EIS block drifts over 24-27 degC; all of it is the
# same nominal condition, so the window spans the full range. An earlier
# (24, 26) window silently dropped the 27 degC spectrum of every family, which
# put the family summary on 9 spectra per family while the Kramers-Kronig
# validation ran on all 10. Sensitivity to the window is reported by
# build_eis_subset_sensitivity below.
EIS_ROOM_T_WINDOW = (24.0, 27.0)

# characteristic_freq_min_zim_hz saturates at the lowest measured frequency
# (0.05 Hz) whenever -Z'' rises monotonically into the low-frequency tail, i.e.
# whenever no mid-frequency arc apex is resolved in the measured window. The
# descriptor is therefore bimodal and its median is unstable; the fraction of
# spectra with a resolved apex is the stable statistic.
EIS_MIN_MEASURED_FREQ_HZ = 0.05


def build_eis_family_summary(eis: pd.DataFrame) -> pd.DataFrame:
    lo, hi = EIS_ROOM_T_WINDOW
    room_t = eis[eis["temperature_deg_c"].between(lo, hi, inclusive="both")].copy()
    room_t["arc_apex_resolved"] = (
        room_t["characteristic_freq_min_zim_hz"] > EIS_MIN_MEASURED_FREQ_HZ
    )
    summary = (
        room_t.groupby(["cell_family", "chemistry_type"], dropna=False)
        .agg(
            spectra_count=("member_path", "count"),
            median_series_resistance_mohm=("series_resistance_high_freq_ohm", lambda x: float(np.nanmedian(x) * 1000)),
            median_low_freq_zre_mohm=("low_freq_zre_ohm", lambda x: float(np.nanmedian(x) * 1000)),
            median_arc_width_mohm=("arc_width_zre_ohm", lambda x: float(np.nanmedian(x) * 1000)),
            median_characteristic_freq_hz=("characteristic_freq_min_zim_hz", "median"),
            n_arc_apex_resolved=("arc_apex_resolved", "sum"),
        )
        .reset_index()
        .sort_values("cell_family")
    )
    summary["arc_apex_resolved_fraction"] = (
        summary["n_arc_apex_resolved"] / summary["spectra_count"]
    )
    return summary


def build_eis_subset_sensitivity(
    eis: pd.DataFrame, kk: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Show that the SIB-to-reference impedance ratio does not depend on which
    defensible subset of the room-temperature spectra is used."""
    frame = eis.copy()
    if kk is not None:
        frame = frame.merge(
            kk[["member_path", "kk_consistent"]], on="member_path", how="left"
        )
    lo, hi = EIS_ROOM_T_WINDOW
    in_window = frame["temperature_deg_c"].between(lo, hi, inclusive="both")

    subsets: dict[str, pd.Series] = {
        "all room-temperature spectra": in_window,
        "narrow 24-26 degC window": frame["temperature_deg_c"].between(24, 26, inclusive="both"),
    }
    if kk is not None:
        subsets["Kramers-Kronig consistent only"] = in_window & (frame["kk_consistent"] == True)  # noqa: E712
        subsets["KK-consistent and 24-26 degC"] = (
            frame["temperature_deg_c"].between(24, 26, inclusive="both")
            & (frame["kk_consistent"] == True)  # noqa: E712
        )

    rows = []
    for label, mask in subsets.items():
        sub = frame[mask]
        med = sub.groupby("cell_family")["low_freq_zre_ohm"].median() * 1000
        if not {"SIB", "LFP", "LTO", "NMC"}.issubset(med.index):
            continue
        # The paper's reference average is the graphite-anode mean (LFP, NMC);
        # LTO is reported per family but excluded (Section 2.1 of the manuscript).
        reference_mean = float(med[["LFP", "NMC"]].mean())
        rows.append(
            {
                "subset": label,
                "n_spectra": int(len(sub)),
                "sib_low_freq_zre_mohm": round(float(med["SIB"]), 2),
                "lfp_low_freq_zre_mohm": round(float(med["LFP"]), 2),
                "lto_low_freq_zre_mohm": round(float(med["LTO"]), 2),
                "nmc_low_freq_zre_mohm": round(float(med["NMC"]), 2),
                "graphite_mean_mohm": round(reference_mean, 2),
                "sib_to_reference_ratio": round(float(med["SIB"]) / reference_mean, 3),
            }
        )
    return pd.DataFrame(rows)


def build_hppc_summary(hppc: pd.DataFrame) -> pd.DataFrame:
    frame = hppc.copy()
    frame["abs_c_rate"] = frame["c_rate"].abs()
    frame["soc_percent_round"] = (frame["soc_percent"] / 10).round() * 10
    summary = (
        frame.groupby(
            ["cell_family", "chemistry_type", "pulse_direction", "abs_c_rate", "soc_percent_round"],
            dropna=False,
        )
        .agg(
            n=("member_path", "count"),
            median_r_fast_mohm=("r_fast_ohm", lambda x: float(np.nanmedian(x) * 1000)),
            median_r_100ms_mohm=("r_100ms_ohm", lambda x: float(np.nanmedian(x) * 1000)),
            median_r_1s_mohm=("r_1s_ohm", lambda x: float(np.nanmedian(x) * 1000)),
        )
        .reset_index()
    )
    return summary.sort_values(["cell_family", "pulse_direction", "abs_c_rate", "soc_percent_round"])


def build_entropy_summary(entropy: pd.DataFrame) -> pd.DataFrame:
    return (
        entropy.groupby(["cell_family", "chemistry_type", "direction"], dropna=False)
        .agg(
            n=("member_path", "count"),
            mean_abs_dudt_mv_per_k=("dudt_mv_per_k", lambda x: float(np.nanmean(np.abs(x)))),
            max_abs_dudt_mv_per_k=("dudt_mv_per_k", lambda x: float(np.nanmax(np.abs(x)))),
            mean_ehs_v_at_25c=("entropic_heat_sensitivity_v_at_25c", "mean"),
        )
        .reset_index()
        .sort_values(["cell_family", "direction"])
    )


def build_thermal_summary(thermal: pd.DataFrame) -> pd.DataFrame:
    return (
        thermal.groupby(["cell_family", "chemistry_type", "direction"], dropna=False)
        .agg(
            n=("member_path", "count"),
            mean_temperature_rise_max_k=("temperature_rise_max_k", "mean"),
            median_temperature_rise_max_k=("temperature_rise_max_k", "median"),
            mean_abs_current_a=("mean_abs_current_a", "mean"),
            mean_duration_s=("duration_s", "mean"),
        )
        .reset_index()
        .sort_values(["cell_family", "direction"])
    )


def plot_capacity_energy(table: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for family, group in table.groupby("cell_family"):
        group = group.sort_values("temperature_deg_c")
        color = COLORS.get(family, "black")
        axes[0].plot(
            group["temperature_deg_c"],
            group["discharge_capacity_retention_vs_25c"],
            marker="o",
            label=family,
            color=color,
        )
        axes[1].plot(
            group["temperature_deg_c"],
            group["energy_retention_vs_25c"],
            marker="o",
            label=family,
            color=color,
        )
    axes[0].set_title("Capacity retention")
    axes[1].set_title("OCV-aware energy retention")
    for axis in axes:
        axis.axhline(1.0, color="0.6", linewidth=1, linestyle="--")
        axis.set_xlabel("Temperature (deg C)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Retention vs 25 deg C")
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_hppc_summary(summary: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    subset = summary[(summary["pulse_direction"] == "discharge") & (summary["abs_c_rate"] == 1.0)].copy()
    fig, axis = plt.subplots(figsize=(6, 4))
    for family, group in subset.groupby("cell_family"):
        group = group.sort_values("soc_percent_round")
        axis.plot(
            group["soc_percent_round"],
            group["median_r_fast_mohm"],
            marker="o",
            label=family,
            color=COLORS.get(family, "black"),
        )
    axis.set_xlabel("SOC (%)")
    axis.set_ylabel("Median R_fast (mohm), 1C discharge")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_eis_summary(summary: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    families = list(summary["cell_family"])
    x = np.arange(len(families))
    width = 0.35
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        x - width / 2,
        summary["median_series_resistance_mohm"],
        width=width,
        label="Series proxy",
        color="#4c6fff",
    )
    axis.bar(
        x + width / 2,
        summary["median_low_freq_zre_mohm"],
        width=width,
        label="Low-frequency Zre",
        color="#9b5de5",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(families)
    axis.set_ylabel("Resistance proxy (mohm)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_findings(path: Path, capacity_energy: pd.DataFrame, eis_summary: pd.DataFrame) -> None:
    ensure_parent(path)
    sib_5 = capacity_energy[(capacity_energy["cell_family"] == "SIB") & (capacity_energy["temperature_deg_c"] == 5)]
    sib_25 = capacity_energy[(capacity_energy["cell_family"] == "SIB") & (capacity_energy["temperature_deg_c"] == 25)]
    lines = [
        "# Initial Findings",
        "",
        "These are preliminary descriptor outputs generated from the downloaded DepositOnce archive.",
        "",
        "## Immediate Signals",
        "",
    ]
    if not sib_5.empty:
        row = sib_5.iloc[0]
        lines.append(
            f"- SIB discharge capacity retention at 5 deg C is {row['discharge_capacity_retention_vs_25c']:.3f}; "
            f"OCV-aware usable-energy retention is {row['energy_retention_vs_25c']:.3f}."
        )
    if not sib_25.empty:
        row = sib_25.iloc[0]
        lines.append(
            f"- SIB 25 deg C baseline discharge capacity is {row['discharge_capacity_ah']:.3f} Ah "
            f"and OCV-aware discharge energy is {row['usable_energy_wh']:.3f} Wh."
        )
    if not eis_summary.empty:
        best = eis_summary.sort_values("median_low_freq_zre_mohm", ascending=False).iloc[0]
        lines.append(
            f"- The largest median low-frequency EIS Zre proxy near 25 deg C is {best['cell_family']} "
            f"at {best['median_low_freq_zre_mohm']:.1f} mohm."
        )
    lines.extend(
        [
            "",
            "## Cautions",
            "",
            "- HPPC processed CSVs do not encode temperature in the filename; treat them as processed BOL/nominal-temperature descriptors until confirmed from the source paper.",
            "- EIS descriptors here are robust spectral proxies, not fitted equivalent-circuit parameters.",
            "- Capacity and OCV extraction follows upstream plotting-script StepIDs and now has publication-trend validation output.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    capacity = pd.read_csv("data_processed/capacity/capacity_temperature_summary.csv")
    ocv = pd.read_csv("data_processed/ocv/ocv_temperature_points.csv")
    hppc = pd.read_csv("data_processed/hppc/hppc_resistance_long.csv")
    eis = pd.read_csv("data_processed/eis/eis_spectral_descriptors.csv")
    entropy = pd.read_csv("data_processed/entropy/entropy_dudt_long.csv")
    thermal = pd.read_csv("data_processed/thermal/thermal_response_descriptors.csv")

    kk_path = Path("results/tables/table_eis_kramers_kronig.csv")
    kk = pd.read_csv(kk_path) if kk_path.exists() else None

    usable_energy = build_usable_energy(ocv)
    capacity_energy = build_capacity_energy_table(capacity, usable_energy)
    hppc_summary = build_hppc_summary(hppc)
    eis_summary = build_eis_family_summary(eis)
    eis_subset_sensitivity = build_eis_subset_sensitivity(eis, kk)
    entropy_summary = build_entropy_summary(entropy)
    thermal_summary = build_thermal_summary(thermal)

    outputs = {
        "data_processed/descriptors/usable_energy_retention.csv": usable_energy,
        "results/tables/table_capacity_energy_retention.csv": capacity_energy,
        "results/tables/table_hppc_resistance_summary.csv": hppc_summary,
        "results/tables/table_eis_family_summary.csv": eis_summary,
        "results/tables/table_eis_subset_sensitivity.csv": eis_subset_sensitivity,
        "results/tables/table_entropy_summary.csv": entropy_summary,
        "results/tables/table_thermal_response_summary.csv": thermal_summary,
    }
    for file_name, frame in outputs.items():
        output_path = Path(file_name)
        ensure_parent(output_path)
        frame.to_csv(output_path, index=False)
        print(f"{file_name}: {len(frame)} rows")

    plot_capacity_energy(capacity_energy, Path("results/figures/fig02_capacity_energy_retention.png"))
    plot_hppc_summary(hppc_summary, Path("results/figures/fig03_hppc_rfast_summary.png"))
    plot_eis_summary(eis_summary, Path("results/figures/fig04_eis_family_summary.png"))
    write_findings(Path("results/reports/initial_findings.md"), capacity_energy, eis_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
