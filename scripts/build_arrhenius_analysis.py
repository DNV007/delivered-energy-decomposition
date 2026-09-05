#!/usr/bin/env python3
"""Fit Arrhenius activation energies to the raw-pulse resistance descriptors.

The five-temperature checkup grid supports a slope-based effective activation
energy for each cell family, which is a stronger statement than a two-point
25/5 degC ratio: it uses every temperature, and the fit quality is itself a
check on whether a single thermally activated process dominates the response.

    R(T) = A * exp(Ea / (k_B T))     =>     ln R = ln A + (Ea / k_B) * (1/T)

Ea is an *effective* activation energy for the whole-cell pulse response. It
aggregates every series contribution (electrolyte transport, interphase and
charge-transfer kinetics, solid-state diffusion, contact terms) and is not the
activation energy of a single identified process.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-energystorage")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/energystorage-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOLTZMANN_EV_PER_K = 8.617333262e-5

TIMEPOINT_COLUMNS = {
    "fast": "median_r_fast_mohm",
    "100ms": "median_r_100ms_mohm",
    "1s": "median_r_1s_mohm",
    "10s": "median_r_10s_mohm",
    "end": "median_r_end_mohm",
}

from figure_style import (COL_FULL, COLORS, GRID, INK_MUTED, MARKERS,
                          panel_label, style_axis, use_style)

use_style()

MIN_POINTS = 4


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fit_arrhenius(temperature_c: np.ndarray, resistance_mohm: np.ndarray) -> dict[str, float]:
    """Ordinary least squares of ln(R) on 1/(k_B T). Slope is Ea in eV."""
    mask = np.isfinite(temperature_c) & np.isfinite(resistance_mohm) & (resistance_mohm > 0)
    temperature_c = temperature_c[mask]
    resistance_mohm = resistance_mohm[mask]
    n = len(temperature_c)
    if n < MIN_POINTS:
        return {"n_points": n}

    x = 1.0 / (BOLTZMANN_EV_PER_K * (temperature_c + 273.15))
    y = np.log(resistance_mohm)

    design = np.vstack([x, np.ones_like(x)]).T
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    activation_energy_ev = float(coefficients[0])

    predicted = design @ coefficients
    ss_residual = float(((y - predicted) ** 2).sum())
    ss_total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else np.nan

    # Standard error of the slope under the usual OLS assumptions. With n = 5
    # this is an optimistic interval; it describes the fit, not cell-to-cell
    # spread, of which there is none available (one cell per family).
    x_variance = float(((x - x.mean()) ** 2).sum())
    if n > 2 and x_variance > 0:
        standard_error_ev = float(np.sqrt(ss_residual / (n - 2) / x_variance))
    else:
        standard_error_ev = np.nan

    return {
        "activation_energy_ev": activation_energy_ev,
        "activation_energy_mev": activation_energy_ev * 1000.0,
        "standard_error_ev": standard_error_ev,
        "standard_error_mev": standard_error_ev * 1000.0,
        "r_squared": r_squared,
        "prefactor_mohm": float(np.exp(coefficients[1])),
        "n_points": n,
        "temperature_min_deg_c": float(temperature_c.min()),
        "temperature_max_deg_c": float(temperature_c.max()),
    }


def fit_vtf(temperature_c: np.ndarray, resistance_mohm: np.ndarray) -> dict[str, float]:
    """Fit the Vogel-Tammann-Fulcher form R(T) = A exp(B / (T - T0)).

    This is the functional form that bulk carbonate-electrolyte transport
    actually follows, and it is the obvious alternative to Arrhenius. Fitting
    both lets us state whether a 5-45 degC window can discriminate them at all,
    rather than asserting Arrhenius behaviour because an Arrhenius plot looks
    straight.

    T0 is profiled on a grid (the fit is linear in ln A and B once T0 is
    fixed), so no nonlinear optimiser and no starting guess are involved.
    """
    mask = np.isfinite(temperature_c) & np.isfinite(resistance_mohm) & (resistance_mohm > 0)
    temperature_k = temperature_c[mask] + 273.15
    y = np.log(resistance_mohm[mask])
    n = len(temperature_k)
    if n < MIN_POINTS:
        return {"n_points": n}

    # T0 must stay below the lowest measured temperature for the form to be
    # defined; the upper cap keeps a margin so B does not diverge.
    t0_grid = np.linspace(0.0, temperature_k.min() - 20.0, 600)
    best = None
    for t0 in t0_grid:
        design = np.vstack([1.0 / (temperature_k - t0), np.ones_like(temperature_k)]).T
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = float(((y - design @ coefficients) ** 2).sum())
        if best is None or residual < best[0]:
            best = (residual, t0, coefficients)

    ss_residual, t0, coefficients = best
    ss_total = float(((y - y.mean()) ** 2).sum())
    return {
        "vtf_b_k": float(coefficients[0]),
        "vtf_t0_k": float(t0),
        "vtf_prefactor_mohm": float(np.exp(coefficients[1])),
        "vtf_r_squared": 1.0 - ss_residual / ss_total if ss_total > 0 else np.nan,
        "vtf_ss_residual": ss_residual,
        "n_points": n,
    }


def compare_arrhenius_vtf(summary: pd.DataFrame) -> pd.DataFrame:
    """Ask whether the measured window can distinguish Arrhenius from VTF.

    Both forms are fitted to the same points and compared by Akaike information
    criterion. VTF carries one extra free parameter (T0), so AIC penalises it;
    a |dAIC| below about 2 means the data do not support preferring either form.

    Note the small-sample caveat, which is itself part of the answer: with
    n = 5 checkup temperatures the corrected criterion AICc is undefined for
    the three-parameter VTF form (its n - k - 1 denominator vanishes). A five
    point series simply does not carry enough information to arbitrate between
    a two- and a three-parameter transport law, which is the point being made.
    """
    records = []
    keys = ["cell_family", "chemistry_type", "pulse_direction", "abs_c_rate"]
    for key_values, group in summary.groupby(keys, dropna=False):
        ordered = group.sort_values("temperature_deg_c")
        temperature = ordered["median_actual_temperature_deg_c"].to_numpy(dtype=float)
        for timepoint, column in TIMEPOINT_COLUMNS.items():
            if column not in ordered.columns:
                continue
            resistance = ordered[column].to_numpy(dtype=float)
            arrhenius = fit_arrhenius(temperature, resistance)
            vtf = fit_vtf(temperature, resistance)
            if "activation_energy_ev" not in arrhenius or "vtf_b_k" not in vtf:
                continue

            n = int(arrhenius["n_points"])
            mask = np.isfinite(temperature) & np.isfinite(resistance) & (resistance > 0)
            y = np.log(resistance[mask])
            ss_total = float(((y - y.mean()) ** 2).sum())
            ss_arrhenius = (1.0 - arrhenius["r_squared"]) * ss_total

            def aic(ss: float, k: int) -> float:
                # k counts the fitted parameters plus the residual variance.
                if ss <= 0:
                    return np.nan
                return n * np.log(ss / n) + 2 * k

            def aicc(ss: float, k: int) -> float:
                # Undefined once k + 1 reaches n, which is the case for VTF at
                # the five checkup temperatures available here.
                if ss <= 0 or n - k - 1 <= 0:
                    return np.nan
                return aic(ss, k) + (2 * k * (k + 1)) / (n - k - 1)

            aic_arrhenius = aic(ss_arrhenius, 3)
            aic_vtf = aic(vtf["vtf_ss_residual"], 4)
            delta_aic = aic_vtf - aic_arrhenius
            records.append(
                dict(zip(keys, key_values))
                | {
                    "pulse_timepoint": timepoint,
                    "arrhenius_r_squared": arrhenius["r_squared"],
                    "arrhenius_activation_energy_mev": arrhenius["activation_energy_mev"],
                    "vtf_r_squared": vtf["vtf_r_squared"],
                    "vtf_b_k": vtf["vtf_b_k"],
                    "vtf_t0_k": vtf["vtf_t0_k"],
                    "aic_arrhenius": aic_arrhenius,
                    "aic_vtf": aic_vtf,
                    "delta_aic_vtf_minus_arrhenius": delta_aic,
                    "aicc_arrhenius": aicc(ss_arrhenius, 3),
                    "aicc_vtf": aicc(vtf["vtf_ss_residual"], 4),
                    "vtf_fits_at_least_as_well": bool(
                        vtf["vtf_r_squared"] >= arrhenius["r_squared"]
                    ),
                    "forms_indistinguishable": bool(
                        np.isfinite(delta_aic) and abs(delta_aic) < 2.0
                    ),
                }
            )
    table = pd.DataFrame.from_records(records)
    return table.sort_values(
        ["pulse_direction", "abs_c_rate", "pulse_timepoint", "cell_family"]
    ).reset_index(drop=True)


def build_activation_table(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_keys = ["cell_family", "chemistry_type", "pulse_direction", "abs_c_rate"]
    for keys, group in summary.groupby(group_keys, dropna=False):
        ordered = group.sort_values("temperature_deg_c")
        temperature = ordered["median_actual_temperature_deg_c"].to_numpy(dtype=float)
        for timepoint, column in TIMEPOINT_COLUMNS.items():
            if column not in ordered.columns:
                continue
            fit = fit_arrhenius(temperature, ordered[column].to_numpy(dtype=float))
            if "activation_energy_ev" not in fit:
                continue
            records.append(dict(zip(group_keys, keys)) | {"pulse_timepoint": timepoint} | fit)

    table = pd.DataFrame.from_records(records)
    return table.sort_values(
        ["pulse_direction", "abs_c_rate", "pulse_timepoint", "cell_family"]
    ).reset_index(drop=True)


def add_reference_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Attach the Li-ion reference mean as a mean of per-family activation energies.

    This is deliberately a mean of per-family values, matching the convention
    used for every other reference-mean descriptor in the manuscript. It is a
    descriptive benchmark across three distinct chemistries, not a class estimate.
    """
    lithium = table[table["chemistry_type"] == "lithium-ion"]
    reference = (
        lithium.groupby(["pulse_direction", "abs_c_rate", "pulse_timepoint"])
        .agg(
            li_reference_family_count=("cell_family", "nunique"),
            li_reference_mean_activation_energy_ev=("activation_energy_ev", "mean"),
            li_reference_min_activation_energy_ev=("activation_energy_ev", "min"),
            li_reference_max_activation_energy_ev=("activation_energy_ev", "max"),
        )
        .reset_index()
    )
    merged = table.merge(reference, on=["pulse_direction", "abs_c_rate", "pulse_timepoint"], how="left")
    merged["activation_energy_excess_vs_li_mean_ev"] = (
        merged["activation_energy_ev"] - merged["li_reference_mean_activation_energy_ev"]
    )
    merged["exceeds_every_li_reference"] = (
        merged["activation_energy_ev"] > merged["li_reference_max_activation_energy_ev"]
    )
    return merged


def plot_arrhenius(summary: pd.DataFrame, table: pd.DataFrame, output_path: Path) -> None:
    selection = summary[(summary["pulse_direction"] == "discharge") & (summary["abs_c_rate"] == 1.0)]
    fits = table[
        (table["pulse_direction"] == "discharge")
        & (table["abs_c_rate"] == 1.0)
        & (table["pulse_timepoint"] == "1s")
    ].set_index("cell_family")

    figure, axis = plt.subplots(figsize=(COL_FULL * 0.72, 3.9))
    for family in ["SIB", "LFP", "NMC", "LTO"]:
        group = selection[selection["cell_family"] == family].sort_values("temperature_deg_c")
        if group.empty or family not in fits.index:
            continue
        temperature_k = group["median_actual_temperature_deg_c"].to_numpy(dtype=float) + 273.15
        resistance = group["median_r_1s_mohm"].to_numpy(dtype=float)
        inverse_temperature = 1000.0 / temperature_k

        axis.plot(
            inverse_temperature,
            resistance,
            linestyle="none",
            marker=MARKERS[family],
            color=COLORS[family],
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=(
                f"{family}: $E_\\mathrm{{a}}$ = "
                f"{fits.loc[family, 'activation_energy_mev']:.0f} meV "
                f"($R^2$ = {fits.loc[family, 'r_squared']:.3f})"
            ),
        )
        grid = np.linspace(inverse_temperature.min(), inverse_temperature.max(), 100)
        activation_energy = fits.loc[family, "activation_energy_ev"]
        prefactor = fits.loc[family, "prefactor_mohm"]
        axis.plot(
            grid,
            prefactor * np.exp(activation_energy / (BOLTZMANN_EV_PER_K * (1000.0 / grid))),
            "-",
            color=COLORS[family],
            linewidth=1.6,
            alpha=0.9,
        )

    axis.set_yscale("log")
    axis.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.set_yticks([20, 30, 50, 100, 200, 300])
    axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_xlabel(r"$1000/T$ (K$^{-1}$)")
    axis.set_ylabel(r"1C-discharge $R_{1\,\mathrm{s}}$ (m$\Omega$)")
    axis.set_title("Arrhenius fit of the 1 s raw-pulse resistance", pad=28)
    axis.legend(fontsize=8.5, loc="upper left")
    style_axis(axis, grid="both")
    axis.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)

    secondary = axis.secondary_xaxis(
        "top", functions=(lambda v: 1000.0 / v - 273.15, lambda t: 1000.0 / (t + 273.15))
    )
    secondary.set_xlabel(r"Temperature ($^\circ$C)")

    figure.tight_layout()
    ensure_parent(output_path)
    figure.savefig(output_path)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pulse-summary",
        default="results/tables/table_raw_pulse_temperature_summary.csv",
    )
    parser.add_argument(
        "--output-table",
        default="results/tables/table_arrhenius_activation_energy.csv",
    )
    parser.add_argument(
        "--output-figure",
        default="manuscript/figures/figure_s1_arrhenius_activation.png",
    )
    parser.add_argument(
        "--output-form-comparison",
        default="results/tables/table_arrhenius_vtf_comparison.csv",
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.pulse_summary)
    table = add_reference_comparison(build_activation_table(summary))

    output_table = Path(args.output_table)
    ensure_parent(output_table)
    table.to_csv(output_table, index=False)

    form_comparison = compare_arrhenius_vtf(summary)
    form_path = Path(args.output_form_comparison)
    ensure_parent(form_path)
    form_comparison.to_csv(form_path, index=False)

    plot_arrhenius(summary, table, Path(args.output_figure))

    headline = table[
        (table["pulse_direction"] == "discharge")
        & (table["abs_c_rate"] == 1.0)
        & (table["pulse_timepoint"] == "1s")
    ]
    print(f"Wrote {output_table} ({len(table)} fits)")
    print(f"Wrote {args.output_figure}")
    print("\n1C discharge, R_1s effective activation energies:")
    for _, row in headline.sort_values("activation_energy_ev", ascending=False).iterrows():
        print(
            f"  {row['cell_family']:>4}: {row['activation_energy_mev']:6.1f} "
            f"+/- {row['standard_error_mev']:4.1f} meV   "
            f"R2 = {row['r_squared']:.4f}   n = {int(row['n_points'])}"
        )
    lithium = headline[headline["chemistry_type"] == "lithium-ion"]
    print(f"\n  Li-ion mean of per-family Ea: {lithium['activation_energy_ev'].mean()*1000:.1f} meV")
    print(f"  Li-ion range:                 {lithium['activation_energy_ev'].min()*1000:.1f}"
          f" to {lithium['activation_energy_ev'].max()*1000:.1f} meV")

    print(f"\nWrote {form_path} ({len(form_comparison)} form comparisons)")
    indistinguishable = int(form_comparison["forms_indistinguishable"].sum())
    vtf_better = int(form_comparison["vtf_fits_at_least_as_well"].sum())
    print(
        f"  Arrhenius vs VTF indistinguishable (|dAIC| < 2): "
        f"{indistinguishable}/{len(form_comparison)} fits"
    )
    print(f"  VTF fits at least as well as Arrhenius:          "
          f"{vtf_better}/{len(form_comparison)} fits")
    print("  AICc is undefined for VTF at n = 5 (k + 1 = n); AIC reported instead.")
    headline_forms = form_comparison[
        (form_comparison["pulse_direction"] == "discharge")
        & (form_comparison["abs_c_rate"] == 1.0)
        & (form_comparison["pulse_timepoint"] == "1s")
    ]
    print("\n1C discharge, R_1s Arrhenius vs VTF:")
    for _, row in headline_forms.iterrows():
        print(
            f"  {row['cell_family']:>4}: R2 Arrhenius = {row['arrhenius_r_squared']:.4f}  "
            f"R2 VTF = {row['vtf_r_squared']:.4f}  "
            f"dAIC = {row['delta_aic_vtf_minus_arrhenius']:+.2f}  "
            f"VTF T0 = {row['vtf_t0_k']:5.1f} K"
        )


if __name__ == "__main__":
    main()
