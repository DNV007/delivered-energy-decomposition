#!/usr/bin/env python3
"""Verify that every cell of the primary results table sits under the right family column.

`check_manuscript_numbers.py` re-derives the *values* the primary results table quotes, but it
reads the released CSVs and never looks at the LaTeX. It therefore cannot see a
column transposition: swap two family columns and every quoted value is still
present and still correct, just attached to the wrong cell. That is exactly the
failure mode risked when the columns were reordered to SIB, LFP, NMC, LTO, so
this script parses both results tables out of Manuscript.tex and checks each
cell against the family it claims to describe.

Run after any change to either results table's column order or row set.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The JES pair moved into manuscript/latex/JES/ when the ACS Energy Letters
# version was branched; the ACS files carry their own _ACS_Energy_Letters names.
MAIN = ROOT / "manuscript" / "latex" / "JES" / "Manuscript.tex"
SUPPLEMENT = ROOT / "manuscript" / "latex" / "JES" / "SM.tex"
GRAPHITE = ["LFP", "NMC"]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / name)


def parse_results_table() -> tuple[list[str], dict[str, list[str]]]:
    """Return the family column order and each data row's cells, keyed by label."""
    text = MAIN.read_text()
    # The energy and resistance tables share a column layout, so both are
    # parsed and their rows pooled: a transposition in either is a failure.
    blocks = []
    for match in re.finditer(r"Descriptor &\nSIB & ", text):
        end = text.index("\\bottomrule", match.start())
        blocks.append(text[match.start():end])
    if not blocks:
        raise SystemExit("no results table found in Manuscript.tex")

    header = re.search(r"Descriptor &\s*\n(.*?) \\\\", blocks[0], re.S).group(1)
    order = [c.strip() for c in header.split("&")]
    families = order[:-1]  # drop the trailing aggregate column

    rows: dict[str, list[str]] = {}
    for block in blocks:
        rows.update(_parse_rows(block))
    return families, rows


def _parse_rows(block: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    parts: list[str] = []
    in_heading = False
    for line in block.split("\n"):
        stripped = line.strip()
        if in_heading:                      # \multicolumn headings may wrap
            if stripped.endswith("\\\\"):
                in_heading = False
            continue
        if not stripped or stripped.startswith(("\\midrule", "\\addlinespace")):
            parts = []
            continue
        if stripped.startswith("\\multicolumn"):
            parts = []
            in_heading = not stripped.endswith("\\\\")
            continue
        if stripped.endswith("\\\\") and stripped.count("&") == 4 and parts:
            rows[" ".join(parts)] = [c.strip() for c in stripped[:-2].split("&")]
            parts = []
            continue
        # A row label may wrap across several lines and only the last of them
        # carries the "&" that opens the data cells, so accumulate every line
        # rather than only the terminating one.
        if "&" not in stripped[:-1]:
            parts.append(stripped[:-1].strip() if stripped.endswith("&") else stripped)
    return rows


def number(cell: str) -> float:
    """Strip the LaTeX a cell may carry and return the leading number."""
    cell = re.sub(r"\$|\\times|\\,|\\%", "", cell)
    cell = re.sub(r"\(\d+\)", "", cell)        # 373(17) -> 373
    cell = re.sub(r"\^\{[a-z]\}", "", cell)    # footnote markers
    match = re.search(r"-?\d+\.?\d*", cell)
    return float(match.group()) if match else np.nan


def main() -> int:
    families, rows = parse_results_table()
    print(f"family column order in Manuscript.tex: {families}\n")

    # label in the primary results table -> series indexed by cell_family, and a tolerance
    expected: dict[str, tuple[pd.Series, float]] = {}

    cap = read("table_capacity_energy_retention.csv")
    cap5 = cap[cap.temperature_deg_c == 5.0].set_index("cell_family")
    expected["Discharge capacity retention"] = (cap5.discharge_capacity_retention_vs_25c, 0.001)
    expected["Rested-voltage energy proxy retention"] = (cap5.energy_retention_vs_25c, 0.001)

    dec = read("table_delivered_energy_decomposition.csv")
    d25 = dec[dec.temperature_deg_c == 25.0].set_index("cell_family")
    d5 = dec[dec.temperature_deg_c == 5.0].set_index("cell_family")
    expected[r"$E_\mathrm{del}$ at \SI{25}{\celsius} (\si{\Wh})"] = (d25.delivered_energy_wh, 0.001)
    expected[r"$E_\mathrm{del}$ retention at \SI{5}{\celsius}"] = (d5.delivered_energy_wh_retention_vs_25c, 0.001)
    expected[r"Effective mean load-to-rest voltage gap at \SI{5}{\celsius} (\si{\milli\volt})"] = (
        d5.mean_overpotential_v * 1000, 1.0)
    expected[r"Discharge energy ratio at \SI{5}{\celsius}$^{c}$"] = (d5.energy_efficiency, 0.001)

    split = read("table_energy_shortfall_split.csv")
    s5 = split[split.temperature_deg_c == 5.0].set_index("cell_family")
    expected[r"Shorter-window share of the shortfall (\%)$^{a}$"] = (
        s5.truncation_share_percent, 0.6)
    expected[r"Rested-voltage-shift share of the shortfall (\%)$^{a,b}$"] = (
        s5.relaxed_shift_share_percent, 0.6)
    expected[r"Load-to-rest-gap growth share of the shortfall (\%)$^{a}$"] = (
        s5.polarisation_share_percent, 0.6)

    pulse = read("table_raw_pulse_temperature_summary.csv")
    pulse = pulse[(pulse.pulse_direction == "discharge") & (pulse.abs_c_rate == 1.0)]
    r25 = pulse[pulse.temperature_deg_c == 25.0].set_index("cell_family")["median_r_1s_mohm"]
    r5 = pulse[pulse.temperature_deg_c == 5.0].set_index("cell_family")["median_r_1s_mohm"]
    expected[r"At \SI{25}{\celsius} (\si{\milli\ohm})"] = (r25, 0.1)
    expected[r"At \SI{5}{\celsius} (\si{\milli\ohm})"] = (r5, 0.1)
    expected[r"SIB\,/\,family ratio at \SI{5}{\celsius}"] = (r5["SIB"] / r5, 0.01)
    expected[r"$R_{1\,\text{s}}$ growth: 25\,°C $\to$ 5\,°C"] = (r5 / r25, 0.01)

    cw = read("table_common_window_ocv_comparison.csv")
    cw5 = cw[(cw.direction == "discharge") & (cw.temperature_deg_c == 5.0)].set_index("cell_family")
    expected[r"Common-window rested-voltage shift (\si{\milli\volt})"] = (
        cw5.median_offset_vs_25c_mv, 0.6)

    cpulse = read("table_raw_pulse_temperature_summary.csv")
    cpulse = cpulse[(cpulse.pulse_direction == "charge") & (cpulse.abs_c_rate == 1.0)]
    c25 = cpulse[cpulse.temperature_deg_c == 25.0].set_index("cell_family")["median_r_1s_mohm"]
    c5 = cpulse[cpulse.temperature_deg_c == 5.0].set_index("cell_family")["median_r_1s_mohm"]
    expected[r"$R_{1\,\text{s}}$ growth, 1C charge"] = (c5 / c25, 0.01)

    r45 = pulse[pulse.temperature_deg_c == 45.0].set_index("cell_family")["median_r_1s_mohm"]
    expected[r"$R(\SI{5}{\celsius})/R(\SI{45}{\celsius})$, model-free"] = (r5 / r45, 0.01)

    spread = read("table_cell_replicate_spread.csv")
    spread = spread[spread.pulse_direction == "discharge"].set_index("cell_family")
    expected[r"Cell-to-cell range, 3 cells (\%)$^{c}$"] = (spread.spread_percent_of_mean, 0.1)

    ea = read("table_arrhenius_activation_energy.csv")
    ea = ea[(ea.pulse_direction == "discharge") & (ea.abs_c_rate == 1.0)
            & (ea.pulse_timepoint == "1s")].set_index("cell_family")
    expected[r"$E_\mathrm{a}$ (\si{\milli\eV})$^{a}$"] = (ea.activation_energy_mev, 0.6)

    eis = read("table_eis_family_summary.csv").set_index("cell_family")
    expected[r"Median low-frequency $Z'$ (\si{\milli\ohm})"] = (eis.median_low_freq_zre_mohm, 0.06)

    span = read("table_ocv_slope_summary.csv")
    span = span[span.direction == "discharge"]
    sp25 = span[span.temperature_deg_c == 25.0].set_index("cell_family")["ocv_slope_penalty_v"]
    sp5 = span[span.temperature_deg_c == 5.0].set_index("cell_family")["ocv_slope_penalty_v"]
    expected[r"Central discharge OCV span at \SI{25}{\celsius} (\si{\volt})"] = (sp25, 0.001)
    expected[r"OCV-span temperature ratio (5\,°C\,/\,25\,°C)"] = (sp5 / sp25, 0.001)

    # The entropy coefficient and the beginning-of-life temperature rise left
    # the primary results table in round 23: Supplementary Section S4 quotes both in its own
    # prose, so carrying them here as well was duplication, and the table ran
    # to the foot of its page. check_manuscript_numbers.py still pins them.

    ecm = read("table_eis_ecm_fit_reliability_summary.csv")
    single = ecm[ecm.model == "randles_cpe"].set_index("cell_family")["median_residual_nrmse"]
    twoarc = ecm[ecm.model == "two_arc_cpe"].set_index("cell_family")["median_residual_nrmse"]
    expected[r"Median residual NRMSE$^{d}$"] = (single, 0.001)
    expected["Median NRMSE, two-arc circuit"] = (twoarc, 0.001)

    failures = checked = 0
    unmatched = []
    for label, (series, tol) in expected.items():
        key = next((k for k in rows if k.strip() == label.strip()), None)
        if key is None:
            unmatched.append(label)
            continue
        cells = rows[key]
        for family, cell in zip(families, cells):
            if family not in series.index or cell.strip() in {"---", ""}:
                continue
            quoted, derived = number(cell), float(series[family])
            checked += 1
            if not np.isfinite(quoted) or abs(quoted - derived) > tol:
                failures += 1
                print(f"  FAIL  {label[:46]:46s} {family:4s} "
                      f"table={cell:>10s} source={derived:.4g}")
    print(f"{checked} family cells checked against source, {failures} mismatched")
    if unmatched:
        print(f"rows not covered by this check: {unmatched}")

    failures += check_supplement_family_order()
    return 1 if failures else 0


CANONICAL_ORDER = ["SIB", "LFP", "NMC", "LTO"]


def check_supplement_family_order(latex_dir: Path | None = None) -> int:
    """Every family-ordered table in the SI must use the main text's order.

    Round 21 reordered the primary results table to SIB, LFP, NMC, LTO so the two
    graphite-anode families sit together, and never propagated it: five SI
    tables kept SIB, LFP, LTO, NMC until round 25. A reader comparing the two
    documents, or two tables inside the SI, had to re-orient. Nothing else
    catches this -- check_manuscript_numbers.py reads the released CSVs and
    never opens the LaTeX.
    """
    latex_dir = latex_dir or (MAIN.parent)
    path = latex_dir / SUPPLEMENT.name
    if not path.exists():
        # Not a skip: an absent supplement means the check verified nothing,
        # and a checker that verifies nothing must not report success.
        print(f"ERROR: supplement not found at {path}; SI family-order check "
              f"could not run")
        return 1
    text = re.sub(r"(?m)^\s*%.*$", "", path.read_text())

    failures = 0
    tables = 0
    for match in re.finditer(
        r"\\label\{(tab:[^}]*)\}(.*?)\\end\{(?:tabular|xltabular)\}", text, re.S
    ):
        label, body = match.group(1), match.group(2)
        seen: list[str] = []
        for family in re.findall(r"\b(SIB|LFP|NMC|LTO)\b", body):
            if family not in seen:
                seen.append(family)
        if len(seen) < 3:
            continue
        tables += 1
        expected_order = [f for f in CANONICAL_ORDER if f in seen]
        if seen != expected_order:
            failures += 1
            print(f"  FAIL  {label}: families appear as {seen}, expected {expected_order}")
    print(f"{tables} family-ordered supplementary table(s) checked, {failures} out of order")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
