#!/usr/bin/env python3
"""Check that the current manuscript package has the expected reproducible outputs.

Only files that are actually distributed are checked. Entries for the abandoned
DFT study, the superseded draft/claim package and the working project_steps
scaffold were removed: none of them ships in the repository or the Zenodo
deposit, so anyone verifying the audit from the deposit saw them fail.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Present only after scripts/fetch_depositonce_dataset.py has run: the raw
# archive belongs to TU Berlin and is not redistributed, so a freshly unpacked
# deposit legitimately lacks them. Reported as "fetch" rather than "fail".
FETCH_DEPENDENT_FILES = [
    "data_raw/Readme.md",          # TU Berlin's own readme, shipped inside the archive
    "data_raw/data_EvalSIB.zip",
]

REQUIRED_FILES = [
    "data_raw/download_manifest.json",
    "docs/data_inventory.csv",
    "data_processed/descriptors/integrated_descriptor_long.csv",
    "data_processed/descriptors/family_temperature_descriptor_matrix.csv",
    "data_processed/descriptors/family_temperature_targets.csv",
    "data_processed/ocv/ocv_slope_descriptors.csv",
    "data_processed/qc/protocol_soc_validation.csv",
    "results/tables/table_capacity_ocv_publication_validation.csv",
    "results/tables/table_ocv_slope_summary.csv",
    "results/tables/table_protocol_soc_validation_summary.csv",
    "results/tables/table_eis_ecm_fit_reliability_summary.csv",
    "results/tables/table_central_claim_evidence.csv",
    "results/tables/table_arrhenius_activation_energy.csv",
    "results/tables/table_arrhenius_vtf_comparison.csv",
    "results/tables/table_eis_kramers_kronig.csv",
    "results/tables/table_eis_kramers_kronig_summary.csv",
    "results/tables/table_eis_subset_sensitivity.csv",
    "results/tables/table_cell_replicate_spread.csv",
    "results/tables/table_delivered_energy_decomposition.csv",
    "results/tables/table_energy_shortfall_split.csv",
    "results/tables/table_common_window_ocv_comparison.csv",
    "results/tables/table_pulse_rate_dependence.csv",
    "results/tables/table_ordering_robustness.csv",
    "results/tables/table_cutoff_normalisation.csv",
    "results/tables/table_relaxation_geometry.csv",
    "results/reports/ocv_slope_analysis.md",
    "manuscript/submission/cover_letter_skeleton.md",
    # The manuscripts themselves. Their absence used to pass this audit
    # silently, which is how a checker came to point at a manuscript path that
    # no longer existed.
    "manuscript/latex/Manuscript_ACS_Energy_Letters.tex",
    "manuscript/latex/SM_ACS_Energy_Letters.tex",
    "manuscript/latex/JES/Manuscript.tex",
    "manuscript/latex/JES/SM.tex",]

REQUIRED_FIGURES = [
    "manuscript/figures/figure1_data_architecture.png",
    "manuscript/figures/figure2_energy_decomposition.png",
    # ACS Energy Letters uses its own variants of the two main figures.
    "manuscript/figures/figure_decomposition_trace.png",
    "manuscript/figures/figure_decomposition_trace_acs.png",
    "manuscript/figures/figure2_energy_decomposition_acs.png",
    "manuscript/figures/figure4_raw_pulse_bottleneck.png",
    "manuscript/figures/figure5_eis_bottleneck_decomposition.png",
    "manuscript/figures/figure6_ocv_entropy_thermal_coupling.png",
    "manuscript/figures/graphical_abstract.png",
    "results/figures/fig12_ocv_slope_penalty.png",
]

CSV_ROW_CHECKS = {
    "data_processed/descriptors/integrated_descriptor_long.csv": 14_000,
    "data_processed/descriptors/family_temperature_descriptor_matrix.csv": 20,
    "data_processed/descriptors/family_temperature_targets.csv": 20,
    "data_processed/ocv/ocv_slope_descriptors.csv": 480,
    "data_processed/qc/protocol_soc_validation.csv": 420,
    "results/tables/table_capacity_ocv_publication_validation.csv": 8,
    "results/tables/table_ocv_slope_summary.csv": 40,
    "results/tables/table_central_claim_evidence.csv": 6,
    "results/tables/table_arrhenius_activation_energy.csv": 120,
    "results/tables/table_arrhenius_vtf_comparison.csv": 120,
    "results/tables/table_eis_kramers_kronig.csv": 40,
    "results/tables/table_eis_subset_sensitivity.csv": 4,
    "results/tables/table_eis_ecm_fit_reliability.csv": 160,
    "results/tables/table_eis_ecm_fit_reliability_summary.csv": 16,
    # 4 families x 5 margins x 2 cold points.
    "results/tables/table_cutoff_normalisation.csv": 40,
}


def check_file(path: str, min_size: int = 1) -> dict[str, object]:
    file_path = Path(path)
    exists = file_path.exists()
    size = file_path.stat().st_size if exists else 0
    return {
        "check": "file_exists",
        "path": path,
        "status": "pass" if exists and size >= min_size else "fail",
        "detail": f"size={size}",
    }


def check_csv_rows(path: str, min_rows: int) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.exists():
        return {"check": "csv_rows", "path": path, "status": "fail", "detail": "missing"}
    with file_path.open(newline="", encoding="utf-8") as handle:
        rows = max(0, sum(1 for _ in csv.DictReader(handle)))
    return {
        "check": "csv_rows",
        "path": path,
        "status": "pass" if rows >= min_rows else "fail",
        "detail": f"rows={rows}; expected_min={min_rows}",
    }


def check_validation_status(path: str, column: str, expected: str) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.exists():
        return {"check": "status_column", "path": path, "status": "fail", "detail": "missing"}
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            return {"check": "status_column", "path": path, "status": "fail", "detail": f"missing {column}"}
        values = [row.get(column, "") for row in reader]
    passed = bool(values) and all(value == expected for value in values)
    return {
        "check": "status_column",
        "path": path,
        "status": "pass" if passed else "fail",
        "detail": f"{column} all {expected}={passed}",
    }


def check_ecm_rejection(path: str) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.exists():
        return {"check": "ecm_rejection", "path": path, "status": "fail", "detail": "missing"}
    pass_count = 0
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "pass_count" not in reader.fieldnames:
            return {"check": "ecm_rejection", "path": path, "status": "fail", "detail": "missing pass_count"}
        for row in reader:
            pass_count += int(float(row.get("pass_count") or 0))
    return {
        "check": "ecm_rejection",
        "path": path,
        "status": "pass" if pass_count == 0 else "fail",
        "detail": f"pass_count_sum={pass_count}",
    }


def check_protocol_soc(path: str) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.exists():
        return {"check": "protocol_soc_mean_error", "path": path, "status": "fail", "detail": "missing"}
    values = []
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "mean_soc_abs_error_percent" not in reader.fieldnames:
            return {
                "check": "protocol_soc_mean_error",
                "path": path,
                "status": "fail",
                "detail": "missing mean_soc_abs_error_percent",
            }
        for row in reader:
            value = row.get("mean_soc_abs_error_percent")
            if value not in (None, ""):
                values.append(float(value))
    mean_error = sum(values) / len(values) if values else float("inf")
    return {
        "check": "protocol_soc_mean_error",
        "path": path,
        "status": "pass" if mean_error < 2.0 else "fail",
        "detail": f"mean_error_percent={mean_error:.3f}",
    }


def check_fetch_dependent(path: str) -> dict[str, object]:
    """A file that only exists once the raw archive has been downloaded."""
    target = Path(path)
    if target.is_file() and target.stat().st_size > 0:
        return {"check": "file_exists", "path": path, "status": "pass",
                "detail": f"size={target.stat().st_size}"}
    return {"check": "file_exists", "path": path, "status": "fetch",
            "detail": "run scripts/fetch_depositonce_dataset.py"}


def build_checks() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    records.extend(check_fetch_dependent(path) for path in FETCH_DEPENDENT_FILES)
    records.extend(check_file(path) for path in REQUIRED_FILES)
    records.extend(check_file(path, min_size=10_000) for path in REQUIRED_FIGURES)
    records.extend(check_csv_rows(path, min_rows) for path, min_rows in CSV_ROW_CHECKS.items())
    records.append(
        check_validation_status(
            "results/tables/table_capacity_ocv_publication_validation.csv",
            "validation_status",
            "pass",
        )
    )
    records.append(
        check_validation_status(
            "results/tables/table_central_claim_evidence.csv",
            "support_status",
            "supports",
        )
    )
    records.append(check_ecm_rejection("results/tables/table_eis_ecm_fit_reliability_summary.csv"))
    records.append(check_protocol_soc("results/tables/table_protocol_soc_validation_summary.csv"))
    return records


def write_checks_csv(path: Path, checks: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "path", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)


def write_report(path: Path, checks: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pass_count = sum(1 for check in checks if check["status"] == "pass")
    total_count = len(checks)
    fetch_count = sum(1 for row in checks if row["status"] == "fetch")
    decision = "pass" if pass_count + fetch_count == total_count else "fail"
    lines = [
        "# Final Reproducibility Check",
        "",
        f"Decision: {decision}.",
        f"Checks passed: {pass_count}/{total_count}."
        + (f" {fetch_count} awaiting the raw archive"
           " (run scripts/fetch_depositonce_dataset.py)." if fetch_count else ""),
        "",
        "This check verifies the presence and basic integrity of the current processed data, manuscript figures, validation tables, draft text, and submission skeleton. It does not redownload the raw dataset.",
        "",
        "## Check Details",
        "",
    ]
    for row in checks:
        lines.append(f"- {row['status']}: {row['check']} | {row['path']} | {row['detail']}")
    lines.extend(
        [
            "",
            "## Separate Test Command",
            "",
            "The Python regression suite should also pass with `python3 -m pytest -q`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    checks = build_checks()
    write_checks_csv(Path("results/tables/table_final_reproducibility_check.csv"), checks)
    write_report(Path("results/reports/final_reproducibility_check.md"), checks)

    # all([]) is True, so an audit that assembled no checks used to report a
    # clean pass. An audit that inspected nothing has not audited anything.
    if not checks:
        print("FAIL: the audit assembled no checks at all.")
        return 1

    failed = [c for c in checks if c["status"] not in ("pass", "fetch")]
    fetch = [c for c in checks if c["status"] == "fetch"]
    # This script wrote its whole result to files and nothing to the terminal,
    # so a human running it saw an empty, successful-looking run either way.
    print(f"{len(checks) - len(failed)}/{len(checks)} reproducibility checks passed"
          + (f", {len(fetch)} awaiting the raw archive" if fetch else ""))
    for check in failed:
        print(f"  FAIL  {check['check']} | {check['path']} | {check['detail']}")
    print("report: results/reports/final_reproducibility_check.md")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
