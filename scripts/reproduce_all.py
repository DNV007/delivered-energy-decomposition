#!/usr/bin/env python3
"""Regenerate every released table, figure and check from the raw archive.

This is the single end-to-end reproduction command named in the manuscript's
Data and Code Availability statement. It runs the pipeline in dependency
order, then the four verification scripts that pin the manuscript to it.

    python scripts/reproduce_all.py

The raw archive is expected at data_raw/data_EvalSIB.zip. Pass --fetch to
download it first from TU Berlin DepositOnce (about 213 MB), or fetch it by
hand with scripts/fetch_depositonce_dataset.py.

Ordering is not alphabetical and not arbitrary. Each stage below consumes
outputs written by an earlier one; the comments name the dependency that fixes
the position, because those are the edges that break silently when a script is
added. Two are easy to get wrong: the Kramers-Kronig screen must precede
build_initial_analysis.py, which reads its output table, and
run_baseline_models.py must precede run_sensitivity_checks.py even though the
exploratory models it fits were cut from the manuscript -- the sensitivity
tables that survived are written by the same script that consumes them.

Verification is deliberately last and deliberately separate. The pipeline
writing a table is not evidence the manuscript agrees with it; that is what
check_manuscript_numbers.py is for.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT_ROOT / "data_raw/data_EvalSIB.zip"

# (stage title, [(script, why it sits here), ...])
STAGES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Extract descriptors from the raw archive",
        [
            ("inventory_raw_data.py", "archive -> qc/raw_file_inventory.csv"),
            ("profile_raw_tables.py", "archive -> qc/raw_table_schemas.json"),
            ("extract_initial_descriptors.py", "archive -> capacity, ocv, eis, entropy, hppc, thermal"),
            ("extract_raw_pulse_descriptors.py", "archive -> hppc/raw_temperature_pulse_descriptors.csv"),
            ("validate_protocol_soc_indexing.py", "archive -> qc/protocol_soc_validation.csv"),
        ],
    ),
    (
        "Screen the impedance spectra",
        [
            # Must precede build_initial_analysis.py, which reads the KK table.
            ("check_eis_kramers_kronig.py", "archive -> table_eis_kramers_kronig.csv"),
            ("check_ecm_fit_reliability.py", "archive -> table_eis_ecm_fit_reliability*.csv"),
        ],
    ),
    (
        "First-level descriptor analysis",
        [
            ("build_initial_analysis.py", "needs the KK table and every data_processed/ stream"),
            ("build_raw_pulse_analysis.py", "needs raw_temperature_pulse_descriptors.csv"),
            ("build_ocv_slope_analysis.py", "needs ocv/ocv_temperature_points.csv"),
            ("build_delivered_energy_decomposition.py", "archive -> delivered-energy segments and split"),
            ("validate_capacity_ocv_against_publication.py", "needs capacity and ocv summaries"),
        ],
    ),
    (
        "Second-level analysis and sensitivity",
        [
            ("build_arrhenius_analysis.py", "needs table_raw_pulse_temperature_summary.csv"),
            ("build_eis_hppc_consistency.py", "needs the pulse summary and eis descriptors"),
            ("build_cell_replicate_analysis.py", "needs hppc/hppc_resistance_long.csv"),
            ("check_replicate_temperature_assignment.py", "needs table_cell_replicate_spread.csv"),
            ("check_ocv_quadrature_sensitivity.py", "needs table_delivered_energy_segments.csv"),
            ("check_cutoff_and_step_budget.py", "needs the segment table and the archive"),
            ("check_charge_input_and_alignment.py", "archive + segment table -> preparation controls"),
            ("check_discharge_thermal_excursion.py", "archive -> in-discharge cell temperature"),
            ("check_segment_increment_and_start_state.py", "needs table_delivered_energy_segments.csv"),
            ("check_cutoff_normalisation.py", "segment table -> stopping-line sweep, 40 cases"),
            ("check_rest_extrapolation.py", "archive + segment table -> finite-rest bounds"),
            ("check_recoverability_reconstruction.py", "segment table + pulse SOC map -> stress test"),
            ("build_segment_voltage_minima.py", "archive -> per-segment instantaneous minima"),
            ("build_integrated_descriptor_matrix.py", "needs the eis/hppc consistency and pulse maps"),
        ],
    ),
    (
        "Manuscript package",
        [
            # Cut from the paper as evidence, but still writes the two
            # sensitivity tables the figures and claim package consume.
            ("run_baseline_models.py", "needs the descriptor matrix; feeds run_sensitivity_checks.py"),
            ("run_sensitivity_checks.py", "writes the pulse-timepoint and OCV-window sensitivity tables"),
            ("build_ordering_robustness.py", "needs the replicate spread and the decomposition"),
            ("build_manuscript_tables.py", "needs the family summaries"),
            ("build_manuscript_figures.py", "needs every table above"),
            ("build_decomposition_trace_figure.py", "needs the segment table and the minima table"),
            ("build_decomposition_trace_figure.py --acs", "the ACS variant of the same figure"),
            ("build_graphical_abstract.py", "single-pane Elsevier graphical abstract"),
            ("build_graphical_abstract.py --acs", "the 3 x 2 in ACS TOC graphic and its variant"),
        ],
    ),
    (
        "Verification",
        [
            ("run_final_reproducibility_check.py", "end-to-end reproducibility audit"),
            ("check_manuscript_numbers.py", "re-derives every number quoted in the manuscript"),
            ("check_results_table_layout.py", "results-table cells against the family they claim"),
            ("check_manuscript_prose_defects.py", "scans the LaTeX sources for known defects"),
        ],
    ),
]

# Needs main.aux and supplementary.aux, so it runs only after a LaTeX build.
CROSSREF_SCRIPT = "check_manuscript_crossrefs.py"

# None of these is a pipeline stage: the first downloads the archive, the
# second belongs to the DFT work that was cut from the paper, and the third
# packages a release once the pipeline has already run.
EXCLUDED = {
    # Network fetchers: they bring in data rather than derive anything.
    "fetch_depositonce_dataset.py",
    "fetch_relaxation_benchmark.py",
    # Need third-party archives that are not redistributed here
    # (Stanford/SLAC pulse data, the relaxation benchmark; see data_external/).
    "check_stanford_rate_dependence.py",
    "check_stanford_moderator.py",
    "build_stanford_rate_figure.py",
    "check_relaxation_geometry.py",
    # Release packaging, not analysis.
    "make_release_archive.py",
    "finalise_release.py",
    "write_zenodo_upload_fields.py",
    # Abandoned DFT study, cut from the manuscript.
    "check_vasp_relax_results.py",
}

# These read the LaTeX sources. The manuscript is not part of every
# distribution of this repository, so they are skipped -- loudly, and reported
# as skipped -- when manuscript/latex is absent.
MANUSCRIPT_DEPENDENT = {
    "run_final_reproducibility_check.py",
    "check_results_table_layout.py",
    "check_manuscript_prose_defects.py",
}
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript" / "latex"


def run(script: str, extra: list[str] | None = None) -> tuple[bool, float]:
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *(extra or [])],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        sys.stdout.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])
    return result.returncode == 0, elapsed


def check_script_coverage() -> tuple[list[str], list[str]]:
    """Stages and the scripts directory must agree, in both directions.

    A script on disk that no stage runs is the obvious gap. The opposite gap is
    worse and went unnoticed: a stage naming a script that has since been
    deleted fails only when the pipeline reaches it, a stage or two after the
    run looked healthy.
    """
    listed = {script.split()[0] for _, scripts in STAGES for script, _ in scripts}
    known = listed | EXCLUDED | {CROSSREF_SCRIPT}
    on_disk = {
        path.name
        for path in (PROJECT_ROOT / "scripts").glob("*.py")
        # figure_style.py is an imported module; this file is the runner.
        if path.name not in {"figure_style.py", Path(__file__).name}
    }
    absent = {name for name in listed if not (PROJECT_ROOT / "scripts" / name).is_file()}
    return sorted(on_disk - known), sorted(absent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true",
                        help="download the raw archive before running")
    parser.add_argument("--with-crossrefs", action="store_true",
                        help="also run the cross-reference checker (needs both .aux files)")
    args = parser.parse_args()

    unlisted, absent = check_script_coverage()
    if unlisted:
        print("Scripts present but not in any stage -- add them or list them in "
              f"EXCLUDED: {', '.join(unlisted)}")
    if absent:
        print("Stages name scripts that do not exist: " + ", ".join(absent))
    if unlisted or absent:
        return 1

    if args.fetch:
        print("Fetching the raw archive from DepositOnce ...")
        ok, elapsed = run("fetch_depositonce_dataset.py")
        if not ok:
            print(f"FAILED after {elapsed:.1f} s: fetch_depositonce_dataset.py")
            return 1
    if not ARCHIVE.exists():
        print(f"Raw archive not found at {ARCHIVE.relative_to(PROJECT_ROOT)}.")
        print("Run with --fetch, or download it from DOI 10.14279/depositonce-25036.")
        return 1

    total = sum(len(scripts) for _, scripts in STAGES)
    done = 0
    started = time.monotonic()

    skipped: list[str] = []
    have_manuscript = MANUSCRIPT_DIR.is_dir()

    for title, scripts in STAGES:
        print(f"\n=== {title} ===")
        for entry, why in scripts:
            script, *extra = entry.split()
            done += 1
            print(f"  [{done:2d}/{total}] {entry:<45s} {why}", flush=True)
            if script in MANUSCRIPT_DEPENDENT and not have_manuscript:
                skipped.append(script)
                print(f"           {'':45s} SKIPPED: reads manuscript/latex, which "
                      "this distribution does not carry", flush=True)
                continue
            ok, elapsed = run(script, extra)
            if not ok:
                print(f"\nFAILED after {elapsed:.1f} s: {entry}")
                return 1
            print(f"           {'':45s} ok, {elapsed:.1f} s", flush=True)

    if args.with_crossrefs:
        print(f"\n=== Cross-references ===\n  {CROSSREF_SCRIPT}")
        ok, _ = run(CROSSREF_SCRIPT)
        if not ok:
            print(f"\nFAILED: {CROSSREF_SCRIPT}")
            return 1

    ran = total - len(skipped)
    print(f"\n{ran} of {total} stages completed in {(time.monotonic() - started) / 60:.1f} min.")
    if skipped:
        print("Skipped because the manuscript sources are not in this distribution: "
              + ", ".join(skipped))
    print("Regenerated: data_processed/, results/tables/, results/figures/, "
          "manuscript/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
