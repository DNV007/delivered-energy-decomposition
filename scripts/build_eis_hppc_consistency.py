#!/usr/bin/env python3
"""Compare room-temperature EIS descriptors with raw pulse resistance descriptors."""

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


def median_mohm(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(np.nanmedian(numeric) * 1000.0)


def build_consistency_table(eis: pd.DataFrame, raw_pulse_summary: pd.DataFrame) -> pd.DataFrame:
    near_room = eis[eis["temperature_deg_c"].between(24, 26, inclusive="both")].copy()
    eis_summary = (
        near_room.groupby(["cell_family", "chemistry_type"], dropna=False)
        .agg(
            eis_spectra_count=("member_path", "count"),
            eis_voltage_min_v=("voltage_v", "min"),
            eis_voltage_max_v=("voltage_v", "max"),
            eis_temperature_median_deg_c=("temperature_mean_deg_c", "median"),
            eis_series_resistance_mohm=("series_resistance_high_freq_ohm", median_mohm),
            eis_min_abs_zim_resistance_mohm=("series_resistance_min_abs_zim_ohm", median_mohm),
            eis_low_freq_zre_mohm=("low_freq_zre_ohm", median_mohm),
            eis_arc_width_zre_mohm=("arc_width_zre_ohm", median_mohm),
        )
        .reset_index()
    )
    pulse = raw_pulse_summary[
        (raw_pulse_summary["temperature_deg_c"] == 25)
        & (raw_pulse_summary["pulse_direction"] == "discharge")
        & (raw_pulse_summary["abs_c_rate"] == 1.0)
    ][
        [
            "cell_family",
            "median_r_fast_mohm",
            "median_r_100ms_mohm",
            "median_r_1s_mohm",
            "median_r_10s_mohm",
            "median_polarization_1s_mohm",
            "median_polarization_10s_mohm",
        ]
    ].rename(
        columns={
            "median_r_fast_mohm": "pulse_1c_discharge_r_fast_mohm",
            "median_r_100ms_mohm": "pulse_1c_discharge_r_100ms_mohm",
            "median_r_1s_mohm": "pulse_1c_discharge_r_1s_mohm",
            "median_r_10s_mohm": "pulse_1c_discharge_r_10s_mohm",
            "median_polarization_1s_mohm": "pulse_1c_discharge_polarization_1s_mohm",
            "median_polarization_10s_mohm": "pulse_1c_discharge_polarization_10s_mohm",
        }
    )
    merged = eis_summary.merge(pulse, on="cell_family", how="left").sort_values("cell_family")
    merged["eis_series_to_pulse_r_fast_ratio"] = (
        merged["eis_series_resistance_mohm"] / merged["pulse_1c_discharge_r_fast_mohm"]
    )
    merged["eis_low_freq_to_pulse_r_1s_ratio"] = (
        merged["eis_low_freq_zre_mohm"] / merged["pulse_1c_discharge_r_1s_mohm"]
    )
    return merged


def build_correlation_table(consistency: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("eis_series_resistance_mohm", "pulse_1c_discharge_r_fast_mohm"),
        ("eis_min_abs_zim_resistance_mohm", "pulse_1c_discharge_r_fast_mohm"),
        ("eis_low_freq_zre_mohm", "pulse_1c_discharge_r_1s_mohm"),
        ("eis_arc_width_zre_mohm", "pulse_1c_discharge_polarization_1s_mohm"),
    ]
    records: list[dict[str, object]] = []
    for eis_col, pulse_col in pairs:
        clean = consistency[[eis_col, pulse_col]].dropna()
        records.append(
            {
                "eis_descriptor": eis_col,
                "pulse_descriptor": pulse_col,
                "n_families": len(clean),
                "pearson_r": clean[eis_col].corr(clean[pulse_col], method="pearson") if len(clean) >= 3 else np.nan,
                "spearman_r": clean[eis_col].corr(clean[pulse_col], method="spearman") if len(clean) >= 3 else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def plot_consistency(consistency: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = [
        (
            "eis_series_resistance_mohm",
            "pulse_1c_discharge_r_fast_mohm",
            "EIS series proxy (mohm)",
            "Raw pulse R_fast (mohm)",
        ),
        (
            "eis_low_freq_zre_mohm",
            "pulse_1c_discharge_r_1s_mohm",
            "EIS low-frequency Zre (mohm)",
            "Raw pulse R_1s (mohm)",
        ),
    ]
    for axis, (x_col, y_col, x_label, y_label) in zip(axes, panels):
        for _, row in consistency.iterrows():
            family = row["cell_family"]
            axis.scatter(row[x_col], row[y_col], s=55, color=COLORS.get(family, "black"))
            axis.annotate(family, (row[x_col], row[y_col]), xytext=(5, 4), textcoords="offset points")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(path: Path, correlations: pd.DataFrame) -> None:
    ensure_parent(path)
    lines = [
        "# EIS-HPPC Consistency",
        "",
        "Room-temperature EIS descriptors were compared with 25 deg C raw 1C discharge pulse descriptors.",
        "",
        "## Correlations",
        "",
    ]
    for _, row in correlations.iterrows():
        lines.append(
            f"- {row['eis_descriptor']} vs {row['pulse_descriptor']}: "
            f"Pearson r = {row['pearson_r']:.3f}, Spearman r = {row['spearman_r']:.3f} "
            f"(n = {int(row['n_families'])} families)."
        )
    lines.extend(
        [
            "",
            "## Cautions",
            "",
            "- This is a family-level check with only four chemistries, so it is a qualitative consistency screen.",
            "- EIS spectra span voltage points near room temperature, while pulse descriptors are summarized over protocol SOC at 25 deg C.",
            "- Strong disagreement would signal descriptor mismatch; agreement does not prove a circuit model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eis", default="data_processed/eis/eis_spectral_descriptors.csv")
    parser.add_argument("--pulse-summary", default="results/tables/table_raw_pulse_temperature_summary.csv")
    args = parser.parse_args()

    eis = pd.read_csv(args.eis)
    pulse_summary = pd.read_csv(args.pulse_summary)
    consistency = build_consistency_table(eis, pulse_summary)
    correlations = build_correlation_table(consistency)

    outputs = {
        "results/tables/table_eis_hppc_consistency.csv": consistency,
        "results/tables/table_eis_hppc_consistency_correlations.csv": correlations,
    }
    for file_name, frame in outputs.items():
        output_path = Path(file_name)
        ensure_parent(output_path)
        frame.to_csv(output_path, index=False)
        print(f"{file_name}: {len(frame)} rows")

    plot_consistency(consistency, Path("results/figures/fig07_eis_hppc_consistency.png"))
    write_report(Path("results/reports/eis_hppc_consistency.md"), correlations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
