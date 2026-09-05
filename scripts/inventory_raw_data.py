#!/usr/bin/env python3
"""Inventory raw dataset files and archive members.

Outputs:
- docs/data_inventory.csv
- data_processed/qc/raw_file_inventory.csv
- data_processed/qc/raw_file_inventory_summary.json
- logs/data_inventory_report.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


STREAM_PATTERNS = [
    ("metadata", re.compile(r"(readme|license|metadata|manifest|plotting\.py|\.ipynb|beschreibung)", re.I)),
    ("entropy", re.compile(r"(data_dUdT|dUdT|dS025C|dSrest|_ent\b|entropy|entrop|dEdT|dE_dT|entropie)", re.I)),
    ("thermal", re.compile(r"(raw_temperature_BOL|thermal|therm|heat|temperature[_ -]?response)", re.I)),
    ("checkup_raw", re.compile(r"(raw_CU_diffT|checkup)", re.I)),
    ("ocv", re.compile(r"(raw_OCV_I_curves|ocv|open[_ -]?circuit|incremental[_ -]?open)", re.I)),
    ("hppc", re.compile(r"(hppc|hybrid[_ -]?pulse[_ -]?power)", re.I)),
    ("eis", re.compile(r"(raw_EIS_plotting|eis|impedance|nyquist|frequency|zreal|zimag|zre|zim|drt)", re.I)),
    ("capacity", re.compile(r"(capacity|kapazitaet|kapazität|capcheck|checkup)", re.I)),
    ("pulse_current_response", re.compile(r"(pulse[_ -]?current|current[_ -]?response)", re.I)),
]

TEMPERATURE_PATTERNS = [
    re.compile(r"(?<!\d)(\d{1,3})\s*(?:degc|deg_c|deg|gradc|grad_c|celsius|°c)(?![a-z0-9])", re.I),
    re.compile(r"(?:temp|temperature|t)[_ -]?(\d{1,3})(?!\d)", re.I),
]

SOC_PATTERNS = [
    re.compile(r"(?:soc|SoC)[_ -]?(\d{1,3})(?:pct|percent|%)?", re.I),
    re.compile(r"(\d{1,3})(?:pct|percent|%)?[_ -]?(?:soc|SoC)", re.I),
]

CELL_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])(TC\d{2}(?:SIB|LFP|LTO|NMC)\d{2})(?![A-Za-z0-9])", re.I),
    re.compile(r"(?:cell|zelle)[_ -]?([A-Za-z0-9.-]+)", re.I),
    re.compile(r"\b((?:na|li|sib|lib|nmc|lfp)[A-Za-z0-9._-]{0,16})\b", re.I),
]

TEXT_SUFFIXES = {".csv", ".txt", ".tsv", ".md", ".json", ".xlsx", ".xls", ".mat"}
ARCHIVE_SUFFIXES = {".zip"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_stream(text: str) -> str:
    for stream, pattern in STREAM_PATTERNS:
        if pattern.search(text):
            return stream
    return "unknown"


def extract_temperature(text: str) -> str:
    for pattern in TEMPERATURE_PATTERNS:
        match = pattern.search(text)
        if match:
            return str(int(match.group(1)))
    return ""


def extract_soc(text: str) -> str:
    for pattern in SOC_PATTERNS:
        match = pattern.search(text)
        if match:
            value = int(match.group(1))
            if 0 <= value <= 100:
                return str(value)
    return ""


def extract_cell_id(text: str) -> str:
    for pattern in CELL_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip("._-").upper()
    return ""


def infer_cell_family(text: str) -> str:
    match = re.search(r"(?<![A-Za-z0-9])TC\d{2}(SIB|LFP|LTO|NMC)\d{2}(?![A-Za-z0-9])", text, re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?:^|[_/\W])(SIB|LFP|LTO|NMC)(?:$|[_/\W])", text, re.I)
    if match:
        return match.group(1).upper()
    return ""


def infer_chemistry(text: str) -> str:
    family = infer_cell_family(text)
    if family == "SIB":
        return "sodium-ion"
    if family in {"LFP", "LTO", "NMC"}:
        return "lithium-ion"
    lowered = text.lower()
    if any(token in lowered for token in ["sodium", "na-ion", "na_ion", "natrium"]):
        return "sodium-ion"
    if any(token in lowered for token in ["lithium", "li-ion", "li_ion", "lib"]):
        return "lithium-ion"
    return ""


def row_for_path(path: Path, raw_root: Path) -> dict[str, str]:
    relative = path.relative_to(raw_root).as_posix()
    text = relative.replace("/", " ")
    return {
        "record_type": "file",
        "source_container": "",
        "raw_path": relative,
        "member_path": "",
        "file_name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
        "cell_id": extract_cell_id(text),
        "cell_family": infer_cell_family(text),
        "chemistry_type": infer_chemistry(text),
        "temperature_deg_c": extract_temperature(text),
        "soc_percent": extract_soc(text),
        "measurement_stream": classify_stream(text),
        "parsed": "no",
        "notes": "",
    }


def rows_for_zip(path: Path, raw_root: Path) -> Iterable[dict[str, str]]:
    container = path.relative_to(raw_root).as_posix()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = member.filename
            text = f"{container} {member_path}".replace("/", " ")
            yield {
                "record_type": "archive_member",
                "source_container": container,
                "raw_path": container,
                "member_path": member_path,
                "file_name": Path(member_path).name,
                "suffix": Path(member_path).suffix.lower(),
                "size_bytes": str(member.file_size),
                "sha256": "",
                "cell_id": extract_cell_id(text),
                "cell_family": infer_cell_family(text),
                "chemistry_type": infer_chemistry(text),
                "temperature_deg_c": extract_temperature(text),
                "soc_percent": extract_soc(text),
                "measurement_stream": classify_stream(text),
                "parsed": "no",
                "notes": "",
            }


def inventory(raw_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        rows.append(row_for_path(path, raw_root))
        if path.suffix.lower() in ARCHIVE_SUFFIXES:
            rows.extend(rows_for_zip(path, raw_root))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_type",
        "source_container",
        "raw_path",
        "member_path",
        "file_name",
        "suffix",
        "size_bytes",
        "sha256",
        "cell_id",
        "cell_family",
        "chemistry_type",
        "temperature_deg_c",
        "soc_percent",
        "measurement_stream",
        "parsed",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    stream_counts = Counter(row["measurement_stream"] for row in rows)
    suffix_counts = Counter(row["suffix"] for row in rows)
    record_counts = Counter(row["record_type"] for row in rows)
    chemistry_counts = Counter(row["chemistry_type"] or "unknown" for row in rows)
    family_counts = Counter(row["cell_family"] or "unknown" for row in rows)
    temperatures = sorted({row["temperature_deg_c"] for row in rows if row["temperature_deg_c"]}, key=int)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(rows),
        "record_counts": dict(record_counts),
        "measurement_stream_counts": dict(stream_counts),
        "suffix_counts": dict(suffix_counts),
        "chemistry_counts": dict(chemistry_counts),
        "cell_family_counts": dict(family_counts),
        "temperatures_detected_deg_c": temperatures,
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    streams = summary["measurement_stream_counts"]
    assert isinstance(streams, dict)
    lines = [
        "# Data Inventory Report",
        "",
        f"Created: {summary['created_at']}",
        f"Total records: {summary['record_count']}",
        "",
        "## Measurement Streams",
        "",
    ]
    for stream, count in sorted(streams.items()):
        lines.append(f"- {stream}: {count}")
    lines.extend(
        [
            "",
            "## Temperatures Detected",
            "",
            ", ".join(summary["temperatures_detected_deg_c"]) or "None detected",
            "",
            "## Next Actions",
            "",
            "1. Review `unknown` records and refine classification rules.",
            "2. Confirm cell IDs and chemistry labels against dataset metadata.",
            "3. Implement parsers for each available measurement stream.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data_raw")
    parser.add_argument("--inventory-csv", default="docs/data_inventory.csv")
    parser.add_argument("--qc-csv", default="data_processed/qc/raw_file_inventory.csv")
    parser.add_argument("--summary-json", default="data_processed/qc/raw_file_inventory_summary.json")
    parser.add_argument("--report", default="logs/data_inventory_report.md")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"Raw data folder does not exist: {raw_root}")

    rows = inventory(raw_root)
    write_csv(Path(args.inventory_csv), rows)
    write_csv(Path(args.qc_csv), rows)
    summary = build_summary(rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(Path(args.report), summary)

    print(f"records: {len(rows)}")
    print(f"inventory: {args.inventory_csv}")
    print(f"qc: {args.qc_csv}")
    print(f"summary: {args.summary_json}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
