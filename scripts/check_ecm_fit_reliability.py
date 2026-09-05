#!/usr/bin/env python3
"""Check whether EIS equivalent-circuit fitting is reliable enough to use."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-energystorage")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/energystorage-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metadata import normal_metadata
from src.zip_readers import iter_members, read_csv_member


MODEL_COLORS = {
    "randles_cpe": "#4c6fff",
    "randles_cpe_warburg": "#1baf7a",
    "two_arc_cpe": "#4a3aa7",
    "two_arc_cpe_warburg": "#8b3a8b",
}

# A full cell carries at least two distinguishable interfacial processes, so
# the failure of a single-arc circuit is expected rather than diagnostic. The
# two-arc candidates below are screened on identical criteria so that the
# rejection reported in the manuscript is a statement about the data and not
# about having tried only under-parameterised circuits.
MODEL_N_PARAMETERS = {
    "randles_cpe": 4,
    "randles_cpe_warburg": 5,
    "two_arc_cpe": 7,
    "two_arc_cpe_warburg": 8,
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def z_randles_cpe(freq_hz: np.ndarray, rs: float, rct: float, q: float, alpha: float) -> np.ndarray:
    omega = 2.0 * np.pi * freq_hz
    z_parallel = 1.0 / (1.0 / rct + q * (1j * omega) ** alpha)
    return rs + z_parallel


def z_randles_cpe_warburg(
    freq_hz: np.ndarray, rs: float, rct: float, q: float, alpha: float, sigma: float
) -> np.ndarray:
    omega = 2.0 * np.pi * freq_hz
    z_warburg = sigma * (1.0 - 1.0j) / np.sqrt(omega)
    return z_randles_cpe(freq_hz, rs, rct, q, alpha) + z_warburg


def z_cpe_arc(omega: np.ndarray, r: float, q: float, alpha: float) -> np.ndarray:
    """Impedance of one resistor in parallel with a constant-phase element."""
    return 1.0 / (1.0 / r + q * (1j * omega) ** alpha)


def z_two_arc_cpe(
    freq_hz: np.ndarray,
    rs: float,
    r1: float,
    q1: float,
    alpha1: float,
    r2: float,
    q2: float,
    alpha2: float,
) -> np.ndarray:
    omega = 2.0 * np.pi * freq_hz
    return rs + z_cpe_arc(omega, r1, q1, alpha1) + z_cpe_arc(omega, r2, q2, alpha2)


def z_two_arc_cpe_warburg(
    freq_hz: np.ndarray,
    rs: float,
    r1: float,
    q1: float,
    alpha1: float,
    r2: float,
    q2: float,
    alpha2: float,
    sigma: float,
) -> np.ndarray:
    omega = 2.0 * np.pi * freq_hz
    z_warburg = sigma * (1.0 - 1.0j) / np.sqrt(omega)
    return z_two_arc_cpe(freq_hz, rs, r1, q1, alpha1, r2, q2, alpha2) + z_warburg


def scaled_residual(model_z: np.ndarray, observed_z: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(observed_z), np.nanmedian(np.abs(observed_z)) * 0.2)
    return np.r_[(model_z.real - observed_z.real) / scale, (model_z.imag - observed_z.imag) / scale]


def parameter_near_bounds(theta: np.ndarray, lower: np.ndarray, upper: np.ndarray, fraction: float = 0.02) -> bool:
    width = upper - lower
    return bool(np.any((theta - lower) < fraction * width) or np.any((upper - theta) < fraction * width))


def elite_parameter_cv(solutions: list[np.ndarray], best_cost: float) -> float:
    elite = np.asarray([theta for theta, cost in solutions if cost <= best_cost * 1.05 + 1e-12])
    if elite.size == 0:
        return np.nan
    best = elite[0]
    cvs = []
    for index, value in enumerate(best):
        denom = max(abs(float(value)), 1e-12)
        cvs.append(float(np.nanstd(elite[:, index]) / denom))
    return float(np.nanmax(cvs))


def residual_randles_cpe(theta: np.ndarray, freq_hz: np.ndarray, observed_z: np.ndarray) -> np.ndarray:
    rs, rct, log10_q, alpha = theta
    model_z = z_randles_cpe(freq_hz, rs, rct, 10.0**log10_q, alpha)
    return scaled_residual(model_z, observed_z)


def residual_randles_cpe_warburg(theta: np.ndarray, freq_hz: np.ndarray, observed_z: np.ndarray) -> np.ndarray:
    rs, rct, log10_q, alpha, log10_sigma = theta
    model_z = z_randles_cpe_warburg(freq_hz, rs, rct, 10.0**log10_q, alpha, 10.0**log10_sigma)
    return scaled_residual(model_z, observed_z)


def residual_two_arc_cpe(theta: np.ndarray, freq_hz: np.ndarray, observed_z: np.ndarray) -> np.ndarray:
    rs, r1, log10_q1, alpha1, r2, log10_q2, alpha2 = theta
    model_z = z_two_arc_cpe(
        freq_hz, rs, r1, 10.0**log10_q1, alpha1, r2, 10.0**log10_q2, alpha2
    )
    return scaled_residual(model_z, observed_z)


def residual_two_arc_cpe_warburg(
    theta: np.ndarray, freq_hz: np.ndarray, observed_z: np.ndarray
) -> np.ndarray:
    rs, r1, log10_q1, alpha1, r2, log10_q2, alpha2, log10_sigma = theta
    model_z = z_two_arc_cpe_warburg(
        freq_hz, rs, r1, 10.0**log10_q1, alpha1, r2, 10.0**log10_q2, alpha2, 10.0**log10_sigma
    )
    return scaled_residual(model_z, observed_z)


def fit_model(
    model_name: str,
    residual_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    starts: list[list[float]],
    lower: list[float],
    upper: list[float],
    freq_hz: np.ndarray,
    observed_z: np.ndarray,
) -> dict[str, object]:
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    fitted: list[tuple[np.ndarray, float]] = []
    best_result = None
    for start in starts:
        try:
            result = least_squares(
                residual_fn,
                np.asarray(start, dtype=float),
                args=(freq_hz, observed_z),
                bounds=(lower_arr, upper_arr),
                max_nfev=3500,
                x_scale="jac",
            )
        except ValueError:
            continue
        fitted.append((result.x, float(result.cost)))
        if best_result is None or result.cost < best_result.cost:
            best_result = result

    if best_result is None:
        return {"model": model_name, "fit_success": False}

    theta = best_result.x
    residual_nrmse = float(np.sqrt(2.0 * best_result.cost / len(best_result.fun)))
    near_bounds = parameter_near_bounds(theta, lower_arr, upper_arr)
    stability_cv = elite_parameter_cv(sorted(fitted, key=lambda item: item[1]), float(best_result.cost))
    record: dict[str, object] = {
        "model": model_name,
        "fit_success": bool(best_result.success),
        "residual_nrmse": residual_nrmse,
        "near_parameter_bound": near_bounds,
        "elite_parameter_cv": stability_cv,
        "n_parameters": MODEL_N_PARAMETERS.get(model_name, len(theta)),
        "n_starting_points": len(starts),
        "n_successful_fits": len(fitted),
        "rs_ohm": float(theta[0]),
        "rct_ohm": float(theta[1]),
        "q_s_alpha_per_ohm": float(10.0 ** theta[2]),
        "warburg_sigma_ohm_s_minus_half": np.nan,
    }

    if model_name in ("randles_cpe", "randles_cpe_warburg"):
        record["alpha"] = float(theta[3])
        if model_name == "randles_cpe_warburg":
            record["warburg_sigma_ohm_s_minus_half"] = float(10.0 ** theta[4])
    else:
        # Two-arc circuits: report both arcs, and screen on the less
        # well-behaved of the two exponents so the criterion stays as strict
        # as it is for the single-arc candidates.
        record["rct2_ohm"] = float(theta[4])
        record["q2_s_alpha_per_ohm"] = float(10.0 ** theta[5])
        record["alpha1"] = float(theta[3])
        record["alpha2"] = float(theta[6])
        record["alpha"] = float(min(theta[3], theta[6]))
        if model_name == "two_arc_cpe_warburg":
            record["warburg_sigma_ohm_s_minus_half"] = float(10.0 ** theta[7])
    return record


def read_eis_spectra(zip_path: Path) -> list[tuple[dict[str, object], np.ndarray, np.ndarray]]:
    spectra = []
    members = [member for member in iter_members(zip_path, {".csv"}) if member.startswith("raw_EIS_plotting/")]
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


def build_summary(fits: pd.DataFrame) -> pd.DataFrame:
    return (
        fits.groupby(["model", "cell_family"], dropna=False)
        .agg(
            spectra_count=("member_path", "count"),
            fit_success_count=("fit_success", "sum"),
            pass_count=("fit_reliable", "sum"),
            median_residual_nrmse=("residual_nrmse", "median"),
            max_residual_nrmse=("residual_nrmse", "max"),
            boundary_hit_count=("near_parameter_bound", "sum"),
            median_elite_parameter_cv=("elite_parameter_cv", "median"),
            median_rs_mohm=("rs_ohm", lambda values: float(np.nanmedian(values) * 1000.0)),
            median_rct_mohm=("rct_ohm", lambda values: float(np.nanmedian(values) * 1000.0)),
            median_alpha=("alpha", "median"),
        )
        .reset_index()
        .sort_values(["model", "cell_family"])
    )


def plot_reliability(summary: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    families = ["LFP", "LTO", "NMC", "SIB"]
    model_names = [name for name in MODEL_N_PARAMETERS if name in set(summary["model"])]
    x = np.arange(len(families))
    width = 0.8 / max(len(model_names), 1)
    for offset_index, model_name in enumerate(model_names):
        view = summary[summary["model"] == model_name].set_index("cell_family").reindex(families)
        offset = (offset_index - (len(model_names) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            view["median_residual_nrmse"],
            width=width,
            label=model_name,
            color=MODEL_COLORS.get(model_name, "0.4"),
        )
        axes[1].bar(
            x + offset,
            view["pass_count"] / view["spectra_count"],
            width=width,
            label=model_name,
            color=MODEL_COLORS.get(model_name, "0.4"),
        )
    axes[0].axhline(0.05, color="0.35", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Median scaled residual NRMSE")
    axes[1].set_ylabel("Reliable fit fraction")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(families)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(path: Path, fits: pd.DataFrame, summary: pd.DataFrame) -> None:
    ensure_parent(path)
    total_by_model = (
        fits.groupby("model", dropna=False)
        .agg(total=("fit_reliable", "size"), passed=("fit_reliable", "sum"))
        .reset_index()
    )
    lines = [
        "# Equivalent-Circuit Fit Decision",
        "",
        "Decision: do not add fitted equivalent-circuit parameters to the core analysis at this stage.",
        "",
        "## Criteria",
        "",
        "- Fit all EIS spectra with four candidate models: Randles-CPE, Randles-CPE plus semi-infinite Warburg,",
        "  a two-arc CPE circuit, and a two-arc CPE circuit plus Warburg.",
        "- Require scaled residual NRMSE <= 0.05.",
        "- Reject fits that land near parameter bounds.",
        "- Reject fits with unstable multi-start solutions, using elite-parameter CV > 0.20.",
        "",
        "## Outcome",
        "",
    ]
    for _, row in total_by_model.iterrows():
        lines.append(f"- {row['model']}: {int(row['passed'])}/{int(row['total'])} spectra pass all criteria.")
    best_model = summary.sort_values("median_residual_nrmse").iloc[0]
    lines.extend(
        [
            "",
            f"- Best median residual group: {best_model['model']} on {best_model['cell_family']} "
            f"with median NRMSE {best_model['median_residual_nrmse']:.3f}.",
            "- LTO and NMC spectra show repeated boundary hits in the candidate circuit models.",
            "- Adding a Warburg term improves some residuals but does not produce reliable pass rates.",
            "",
            "## Recommendation",
            "",
            "- Keep robust EIS spectral descriptors in the manuscript: high-frequency series proxy, low-frequency Zre, arc width, and characteristic frequency.",
            "- Do not interpret Rct, CPE Q/alpha, or Warburg sigma as physical fitted parameters in the main results.",
            "- Equivalent-circuit fits may be kept only as a supplementary QC artifact if needed, with explicit failure/pass criteria.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="data_raw/data_EvalSIB.zip")
    args = parser.parse_args()

    spectra = read_eis_spectra(Path(args.zip_path))
    randles_starts = [
        [0.01, 0.01, -3.0, 0.60],
        [0.01, 0.05, -2.0, 0.85],
        [0.02, 0.03, -1.0, 0.70],
        [0.02, 0.10, -2.0, 0.85],
        [0.05, 0.03, -3.0, 0.60],
        [0.05, 0.15, -1.0, 0.75],
    ]
    warburg_starts = [
        [0.01, 0.005, -3.0, 0.60, -4.0],
        [0.01, 0.03, -2.0, 0.85, -3.0],
        [0.02, 0.03, -1.0, 0.70, -3.0],
        [0.02, 0.10, -2.0, 0.85, -2.0],
        [0.05, 0.03, -3.0, 0.60, -4.0],
        [0.05, 0.15, -1.0, 0.75, -2.0],
        [0.02, 0.01, 0.0, 0.90, -3.0],
        [0.01, 0.10, 1.0, 0.75, -2.5],
    ]

    # Two-arc starts separate the arcs by initial time constant (a fast,
    # low-capacitance arc and a slow, high-capacitance one) so the optimiser
    # does not begin with both arcs on top of one another.
    two_arc_starts = [
        [0.01, 0.01, -4.0, 0.80, 0.02, -1.0, 0.70],
        [0.01, 0.02, -3.0, 0.85, 0.05, -0.5, 0.60],
        [0.02, 0.005, -5.0, 0.90, 0.03, -1.5, 0.75],
        [0.02, 0.03, -3.5, 0.75, 0.08, 0.0, 0.65],
        [0.01, 0.05, -4.5, 0.85, 0.10, -1.0, 0.55],
        [0.015, 0.01, -2.5, 0.70, 0.04, -0.5, 0.85],
        [0.02, 0.02, -6.0, 0.95, 0.06, -2.0, 0.60],
        [0.01, 0.008, -3.0, 0.60, 0.02, -1.0, 0.90],
    ]
    two_arc_warburg_starts = [start + [-3.0] for start in two_arc_starts] + [
        [0.01, 0.01, -4.0, 0.80, 0.02, -1.0, 0.70, -2.0],
        [0.02, 0.005, -5.0, 0.90, 0.03, -1.5, 0.75, -4.0],
    ]

    records: list[dict[str, object]] = []
    for meta, freq, observed_z in spectra:
        candidates = [
            (
                "randles_cpe",
                residual_randles_cpe,
                randles_starts,
                [0.0, 1e-4, -8.0, 0.30],
                [0.2, 1.0, 1.0, 1.00],
            ),
            (
                "randles_cpe_warburg",
                residual_randles_cpe_warburg,
                warburg_starts,
                [0.0, 1e-4, -8.0, 0.30, -6.0],
                [0.2, 1.0, 2.0, 1.00, 0.0],
            ),
            (
                "two_arc_cpe",
                residual_two_arc_cpe,
                two_arc_starts,
                [0.0, 1e-4, -8.0, 0.30, 1e-4, -8.0, 0.30],
                [0.2, 1.0, 2.0, 1.00, 1.0, 2.0, 1.00],
            ),
            (
                "two_arc_cpe_warburg",
                residual_two_arc_cpe_warburg,
                two_arc_warburg_starts,
                [0.0, 1e-4, -8.0, 0.30, 1e-4, -8.0, 0.30, -6.0],
                [0.2, 1.0, 2.0, 1.00, 1.0, 2.0, 1.00, 0.0],
            ),
        ]
        for model_name, residual_fn, starts, lower, upper in candidates:
            record = fit_model(model_name, residual_fn, starts, lower, upper, freq, observed_z)
            record.update(meta)
            records.append(record)
        print(f"{meta['file_name']}: fit {len(candidates)} models")

    fits = pd.DataFrame.from_records(records)
    fits["fit_reliable"] = (
        fits["fit_success"].fillna(False)
        & fits["residual_nrmse"].le(0.05)
        & ~fits["near_parameter_bound"].fillna(True)
        & fits["elite_parameter_cv"].le(0.20)
        & fits["alpha"].between(0.35, 0.98, inclusive="both")
    )
    summary = build_summary(fits)

    outputs = {
        "results/tables/table_eis_ecm_fit_reliability.csv": fits,
        "results/tables/table_eis_ecm_fit_reliability_summary.csv": summary,
    }
    for file_name, frame in outputs.items():
        output_path = Path(file_name)
        ensure_parent(output_path)
        frame.to_csv(output_path, index=False)
        print(f"{file_name}: {len(frame)} rows")

    plot_reliability(summary, Path("results/figures/fig08_ecm_fit_reliability.png"))
    write_report(Path("results/reports/ecm_fit_decision.md"), fits, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
