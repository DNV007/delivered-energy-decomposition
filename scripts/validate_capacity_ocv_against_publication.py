#!/usr/bin/env python3
"""Validate capacity and OCV extraction against publication-level reference values."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PUBLICATION_URL = "https://doi.org/10.3390/batteries11110420"

TEMPERATURE_EXPECTATIONS = [
    {
        "cell_family": "LTO",
        "expected_relation": "approx",
        "expected_percent": 2.2,
        "tolerance_percent": 0.25,
        "publication_statement": "LTO capacity reduction of only 2.2% between 5 deg C and 45 deg C.",
    },
    {
        "cell_family": "LFP",
        "expected_relation": "approx",
        "expected_percent": 4.5,
        "tolerance_percent": 0.60,
        "publication_statement": "Graphite-anode LIBs decrease about 4.5% between 5 deg C and 45 deg C.",
    },
    {
        "cell_family": "NMC",
        "expected_relation": "approx",
        "expected_percent": 4.5,
        "tolerance_percent": 0.60,
        "publication_statement": "Graphite-anode LIBs decrease about 4.5% between 5 deg C and 45 deg C.",
    },
    {
        "cell_family": "SIB",
        "expected_relation": "minimum",
        "expected_percent": 20.0,
        "tolerance_percent": 0.0,
        "publication_statement": "SIB usable discharge capacity is reduced by over 20%.",
    },
]

OCV_RATE_EXPECTATIONS = [
    {
        "cell_family": "LFP",
        "expected_relation": "max_abs",
        "expected_percent": 0.2,
        "tolerance_percent": 0.10,
        "publication_statement": "NMC and LFP CC-discharge capacity deviation is less than 0.2%.",
    },
    {
        "cell_family": "NMC",
        "expected_relation": "max_abs",
        "expected_percent": 0.2,
        "tolerance_percent": 0.10,
        "publication_statement": "NMC and LFP CC-discharge capacity deviation is less than 0.2%.",
    },
    {
        "cell_family": "LTO",
        "expected_relation": "approx",
        "expected_percent": 1.45,
        "tolerance_percent": 0.15,
        "publication_statement": "LTO capacity reduction from 0.25C to 3C is around 1.45%.",
    },
    {
        "cell_family": "SIB",
        "expected_relation": "approx",
        "expected_percent": 2.35,
        "tolerance_percent": 0.15,
        "publication_statement": "SIB capacity changes by 2.35% from 0.25C to 3C.",
    },
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def percent_reduction(high_reference: float, low_value: float) -> float:
    if high_reference == 0 or pd.isna(high_reference) or pd.isna(low_value):
        return np.nan
    return 100.0 * (high_reference - low_value) / high_reference


def check_value(actual: float, expected: float, tolerance: float, relation: str) -> tuple[str, float]:
    if pd.isna(actual):
        return "fail", np.nan
    if relation == "approx":
        deviation = abs(actual - expected)
        passed = deviation <= tolerance
    elif relation == "minimum":
        deviation = max(0.0, expected - actual)
        passed = actual >= expected - tolerance
    elif relation == "max_abs":
        deviation = max(0.0, abs(actual) - expected)
        passed = abs(actual) <= expected + tolerance
    else:
        raise ValueError(f"Unsupported validation relation: {relation}")
    return "pass" if passed else "fail", float(deviation)


def build_temperature_validation(capacity: pd.DataFrame) -> list[dict[str, object]]:
    pivot = capacity.pivot(index="cell_family", columns="temperature_deg_c", values="discharge_capacity_ah")
    records: list[dict[str, object]] = []
    for expectation in TEMPERATURE_EXPECTATIONS:
        family = expectation["cell_family"]
        actual = percent_reduction(float(pivot.loc[family, 45.0]), float(pivot.loc[family, 5.0]))
        status, deviation = check_value(
            actual,
            float(expectation["expected_percent"]),
            float(expectation["tolerance_percent"]),
            str(expectation["expected_relation"]),
        )
        records.append(
            {
                "validation_group": "temperature_capacity",
                "metric": "discharge_capacity_reduction_45c_to_5c_percent",
                "cell_family": family,
                "actual_percent": actual,
                "expected_relation": expectation["expected_relation"],
                "expected_percent": expectation["expected_percent"],
                "tolerance_percent": expectation["tolerance_percent"],
                "deviation_percent": deviation,
                "validation_status": status,
                "publication_statement": expectation["publication_statement"],
                "source_url": PUBLICATION_URL,
            }
        )
    return records


def build_ocv_rate_validation(ocv: pd.DataFrame) -> list[dict[str, object]]:
    pivot = ocv.pivot(index="cell_family", columns="c_rate", values="capacity_span_ah")
    records: list[dict[str, object]] = []
    for expectation in OCV_RATE_EXPECTATIONS:
        family = expectation["cell_family"]
        actual = percent_reduction(float(pivot.loc[family, 0.25]), float(pivot.loc[family, 3.0]))
        status, deviation = check_value(
            actual,
            float(expectation["expected_percent"]),
            float(expectation["tolerance_percent"]),
            str(expectation["expected_relation"]),
        )
        records.append(
            {
                "validation_group": "ocv_current_rate",
                "metric": "cc_discharge_capacity_reduction_0p25c_to_3c_percent",
                "cell_family": family,
                "actual_percent": actual,
                "expected_relation": expectation["expected_relation"],
                "expected_percent": expectation["expected_percent"],
                "tolerance_percent": expectation["tolerance_percent"],
                "deviation_percent": deviation,
                "validation_status": status,
                "publication_statement": expectation["publication_statement"],
                "source_url": PUBLICATION_URL,
            }
        )
    return records


def build_validation(capacity: pd.DataFrame, ocv: pd.DataFrame) -> pd.DataFrame:
    records = build_temperature_validation(capacity) + build_ocv_rate_validation(ocv)
    output = pd.DataFrame.from_records(records)
    return output.sort_values(["validation_group", "cell_family"]).reset_index(drop=True)


def write_report(path: Path, validation: pd.DataFrame) -> None:
    ensure_parent(path)
    pass_count = int(validation["validation_status"].eq("pass").sum())
    total_count = len(validation)
    decision = (
        "pass: capacity and OCV extraction is reliable enough for manuscript-level descriptor use."
        if pass_count == total_count
        else "fail: inspect the extraction before using capacity or OCV descriptors in manuscript claims."
    )

    temperature = validation[validation["validation_group"] == "temperature_capacity"]
    ocv_rate = validation[validation["validation_group"] == "ocv_current_rate"]
    temperature_values = ", ".join(
        f"{row.cell_family} {row.actual_percent:.2f}%" for row in temperature.itertuples(index=False)
    )
    ocv_values = ", ".join(
        f"{row.cell_family} {row.actual_percent:.2f}%" for row in ocv_rate.itertuples(index=False)
    )

    lines = [
        "# Capacity and OCV Publication Validation",
        "",
        f"Source: Droese et al., Batteries 2025, 11(11), 420. DOI: {PUBLICATION_URL}.",
        "",
        "Scope: this check compares processed descriptor-table metrics against numerical trends stated in the publication text. It is not independent pixel digitization of the published figures.",
        "",
        f"Decision: {decision}",
        f"Checks passed: {pass_count}/{total_count}.",
        "",
        "## Temperature Capacity Check",
        "",
        "Computed discharge-capacity reduction from 45 deg C to 5 deg C:",
        "",
        f"- {temperature_values}.",
        "",
        "## OCV Current-Rate Check",
        "",
        "Computed CC-discharge capacity reduction from 0.25C to 3C:",
        "",
        f"- {ocv_values}.",
        "",
        "## Caveat",
        "",
        "The validation supports the StepID-based capacity and OCV extraction for trend-level manuscript analysis. Charge-side capacity asymmetry remains a separate caution and should not be used as a central claim without further review.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    capacity = pd.read_csv("data_processed/capacity/capacity_temperature_summary.csv")
    ocv = pd.read_csv("data_processed/ocv/ocv_current_rate_summary.csv")

    validation = build_validation(capacity, ocv)

    table_path = Path("results/tables/table_capacity_ocv_publication_validation.csv")
    report_path = Path("results/reports/capacity_ocv_publication_validation.md")
    ensure_parent(table_path)
    validation.to_csv(table_path, index=False)
    write_report(report_path, validation)

    if not validation["validation_status"].eq("pass").all():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
