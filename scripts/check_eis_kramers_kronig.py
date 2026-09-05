#!/usr/bin/env python3
"""Linear Kramers-Kronig (lin-KK) validity test for the raw EIS spectra.

The lin-KK test fits a series Voigt chain with *fixed* logarithmically spaced
time constants to the measured spectrum. Because the Voigt chain is
Kramers-Kronig transformable by construction, a good fit implies the data are
KK-consistent; systematic residuals imply non-stationarity, non-linearity, or
instrumental artefacts. The model is linear in its parameters, so the fit is a
single weighted least-squares solve with no starting-point dependence.

    Z(w) = Rs + jwL + 1/(jwC) + sum_k Rk / (1 + jw tau_k)

Model order M is selected by residual plateau rather than by the mu-criterion
of Schoenleber, Klotz and Ivers-Tiffee (2014). On these spectra mu is
non-monotonic in M and drops below its conventional 0.85 threshold at orders
that are still grossly underfitted (residuals above 20 %), which would report a
fitting failure as a causality violation. See select_model_order.

This test validates the *measured spectra*. It is independent of, and logically
prior to, the equivalent-circuit reliability screen in
check_ecm_fit_reliability.py: a spectrum can be KK-consistent (trustworthy data)
and still be poorly described by a specific two-element candidate circuit.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metadata import normal_metadata
from src.zip_readers import iter_members, read_csv_member


MIN_ELEMENTS = 4
MAX_ELEMENTS = 25
PLATEAU_TOLERANCE = 0.05

# Primary acceptance criterion. RMS is used rather than the maximum because a
# single-point excursion at a frequency extreme is an instrument artefact, not
# evidence that the spectrum violates causality; the maximum and the frequency
# at which it occurs are reported alongside so the reader can apply a stricter
# rule.
RMS_RESIDUAL_THRESHOLD = 0.01
MAX_RESIDUAL_THRESHOLD = 0.03

from figure_style import (COL_FULL, COLORS, INK_MUTED, MARKERS,
                          panel_label, style_axis, use_style)

use_style()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_eis_spectra(zip_path: Path) -> list[tuple[dict[str, object], np.ndarray, np.ndarray]]:
    spectra = []
    members = [m for m in iter_members(zip_path, {".csv"}) if m.startswith("raw_EIS_plotting/")]
    for member in members:
        if not member.endswith("_EIS.csv"):
            continue
        frame = read_csv_member(zip_path, member)
        valid = frame.dropna(subset=["EISFreq[Hz]", "Zre[Ohm]", "Zim[Ohm]"]).sort_values("EISFreq[Hz]")
        if valid.empty:
            continue
        freq = valid["EISFreq[Hz]"].to_numpy(dtype=float)
        z = valid["Zre[Ohm]"].to_numpy(dtype=float) + 1j * valid["Zim[Ohm]"].to_numpy(dtype=float)
        spectra.append((normal_metadata(member), freq, z))
    return spectra


def fit_voigt_chain(
    freq: np.ndarray, z: np.ndarray, n_elements: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted linear least-squares fit of a fixed-tau Voigt chain.

    Returns (fitted impedance, Voigt resistances). Weighting is proportional
    (1/|Z|), which is the standard choice for lin-KK and gives real and
    imaginary parts comparable influence across decades of impedance.
    """
    omega = 2.0 * np.pi * freq
    tau = np.logspace(np.log10(1.0 / omega.max()), np.log10(1.0 / omega.min()), n_elements)

    # Columns: Rs, L, 1/C, then one per Voigt element.
    denominator = 1.0 + np.outer(omega, tau) ** 2
    real_block = np.column_stack(
        [
            np.ones_like(omega),
            np.zeros_like(omega),
            np.zeros_like(omega),
            1.0 / denominator,
        ]
    )
    imag_block = np.column_stack(
        [
            np.zeros_like(omega),
            omega,
            -1.0 / omega,
            -np.outer(omega, tau) / denominator,
        ]
    )

    weight = 1.0 / np.abs(z)
    design = np.vstack([real_block * weight[:, None], imag_block * weight[:, None]])
    target = np.concatenate([z.real * weight, z.imag * weight])

    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = (real_block @ solution) + 1j * (imag_block @ solution)
    return fitted, solution[3:]


def mu_value(resistances: np.ndarray) -> float:
    """Overfitting indicator: fraction of negative to positive fitted resistance."""
    positive = resistances[resistances >= 0].sum()
    negative = -resistances[resistances < 0].sum()
    if positive <= 0:
        return 0.0
    return float(1.0 - negative / positive)


def select_model_order(freq: np.ndarray, z: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Select the smallest Voigt chain whose residual has plateaued.

    The mu-criterion alone is not usable here: mu is non-monotonic in M for
    these spectra and dips below its threshold at model orders that are still
    grossly underfitted (residuals above 20 %), which would report a fitting
    failure as a KK violation.

    A Voigt chain is Kramers-Kronig transformable at *every* model order, so
    increasing M can never make KK-inconsistent data fit; the residual that
    survives once the fit has converged is the KK-inconsistent part of the
    spectrum. We therefore scan M and take the smallest order whose RMS
    residual is within PLATEAU_TOLERANCE of the best achievable, capped at half
    the number of measured points so the fit stays well determined.
    """
    max_elements = max(MIN_ELEMENTS, min(len(freq) // 2, MAX_ELEMENTS))

    candidates = []
    for n_elements in range(MIN_ELEMENTS, max_elements + 1):
        fitted, resistances = fit_voigt_chain(freq, z, n_elements)
        residual = np.concatenate(
            [(z.real - fitted.real) / np.abs(z), (z.imag - fitted.imag) / np.abs(z)]
        )
        candidates.append((n_elements, mu_value(resistances), fitted, float(np.sqrt(np.mean(residual**2)))))

    best_rms = min(candidate[3] for candidate in candidates)
    for n_elements, mu, fitted, rms in candidates:
        if rms <= best_rms * (1.0 + PLATEAU_TOLERANCE):
            return n_elements, mu, fitted
    return candidates[-1][:3]


def evaluate_spectrum(metadata: dict, freq: np.ndarray, z: np.ndarray) -> dict:
    n_elements, mu, fitted = select_model_order(freq, z)

    magnitude = np.abs(z)
    residual_real = (z.real - fitted.real) / magnitude
    residual_imag = (z.imag - fitted.imag) / magnitude

    combined = np.maximum(np.abs(residual_real), np.abs(residual_imag))
    max_abs_residual = float(combined.max())
    rms_residual = float(np.sqrt(np.mean(residual_real**2 + residual_imag**2)))

    worst_index = int(np.argmax(combined))
    order = np.argsort(freq)
    rank_of_worst = int(np.where(order == worst_index)[0][0])
    at_frequency_extreme = rank_of_worst < 2 or rank_of_worst >= len(freq) - 2

    return dict(metadata) | {
        "n_points": len(freq),
        "freq_min_hz": float(freq.min()),
        "freq_max_hz": float(freq.max()),
        "n_voigt_elements": n_elements,
        "mu": mu,
        "max_abs_residual_real": float(np.abs(residual_real).max()),
        "max_abs_residual_imag": float(np.abs(residual_imag).max()),
        "max_abs_residual": max_abs_residual,
        "rms_residual": rms_residual,
        "worst_residual_freq_hz": float(freq[worst_index]),
        "worst_residual_at_frequency_extreme": at_frequency_extreme,
        "kk_consistent": bool(
            rms_residual <= RMS_RESIDUAL_THRESHOLD and max_abs_residual <= MAX_RESIDUAL_THRESHOLD
        ),
    }


def plot_residuals(results: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(COL_FULL, 3.1))

    families = ["SIB", "LFP", "NMC", "LTO"]
    for index, column, label in (
        (0, "max_abs_residual", "Max |relative residual|"),
        (1, "rms_residual", "RMS relative residual"),
    ):
        axis = axes[index]
        for position, family in enumerate(families):
            values = results.loc[results["cell_family"] == family, column].to_numpy() * 100.0
            axis.scatter(
                np.full_like(values, position) + np.linspace(-0.12, 0.12, len(values)),
                values,
                color=COLORS[family],
                marker=MARKERS[family],
                s=34,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        axis.set_xticks(range(len(families)))
        axis.set_xticklabels(families)
        axis.set_ylabel(f"{label} (%)")
        axis.xaxis.set_tick_params(which="minor", bottom=False)
        style_axis(axis)
        panel_label(axis, "ab"[index])
        threshold = (
            MAX_RESIDUAL_THRESHOLD if column == "max_abs_residual" else RMS_RESIDUAL_THRESHOLD
        )
        axis.axhline(
            threshold * 100.0,
            color=INK_MUTED,
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            label=f"{threshold*100:.0f}% acceptance threshold",
        )
        axis.legend(fontsize=8.5, loc="upper right")

    figure.tight_layout(w_pad=2.2)
    ensure_parent(output_path)
    figure.savefig(output_path)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", default="data_raw/data_EvalSIB.zip")
    parser.add_argument("--output-table", default="results/tables/table_eis_kramers_kronig.csv")
    parser.add_argument(
        "--output-summary", default="results/tables/table_eis_kramers_kronig_summary.csv"
    )
    parser.add_argument(
        "--output-figure", default="manuscript/figures/figure_s2_kramers_kronig.png"
    )
    args = parser.parse_args()

    spectra = read_eis_spectra(Path(args.zip_path))
    if not spectra:
        raise SystemExit("No EIS spectra found in archive.")

    results = pd.DataFrame.from_records(
        [evaluate_spectrum(metadata, freq, z) for metadata, freq, z in spectra]
    )

    output_table = Path(args.output_table)
    ensure_parent(output_table)
    results.to_csv(output_table, index=False)

    summary = (
        results.groupby("cell_family")
        .agg(
            spectra_count=("kk_consistent", "size"),
            kk_consistent_count=("kk_consistent", "sum"),
            median_max_abs_residual=("max_abs_residual", "median"),
            worst_max_abs_residual=("max_abs_residual", "max"),
            median_rms_residual=("rms_residual", "median"),
            worst_rms_residual=("rms_residual", "max"),
            median_n_voigt_elements=("n_voigt_elements", "median"),
            worst_residual_at_extreme_count=("worst_residual_at_frequency_extreme", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_summary, index=False)

    plot_residuals(results, Path(args.output_figure))

    print(f"Wrote {output_table} ({len(results)} spectra)")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_figure}\n")
    print(summary.to_string(index=False))
    total_pass = int(results["kk_consistent"].sum())
    print(
        f"\nKK-consistent (max |residual| <= {MAX_RESIDUAL_THRESHOLD*100:.0f}%): "
        f"{total_pass} of {len(results)} spectra"
    )


if __name__ == "__main__":
    main()
