#!/usr/bin/env python3
"""Profile table-like files inside the raw DepositOnce archive."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sniff_csv_dialect(sample: bytes) -> str:
    text = sample.decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        return ","


def profile_csv(archive: zipfile.ZipFile, member_path: str) -> dict[str, Any]:
    with archive.open(member_path) as handle:
        sample = handle.read(65536)
    delimiter = sniff_csv_dialect(sample)
    frame = pd.read_csv(io.BytesIO(sample), sep=delimiter, nrows=5)
    return {
        "format": "csv",
        "delimiter": delimiter,
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "preview": frame.head(3).astype(str).to_dict(orient="records"),
    }


def profile_parquet(archive: zipfile.ZipFile, member_path: str) -> dict[str, Any]:
    with archive.open(member_path) as handle:
        payload = handle.read()
    parquet = pq.ParquetFile(io.BytesIO(payload))
    preview = parquet.read_row_group(0).slice(0, 3).to_pandas()
    return {
        "format": "parquet",
        "num_rows": parquet.metadata.num_rows,
        "num_row_groups": parquet.metadata.num_row_groups,
        "columns": parquet.schema.names,
        "schema": str(parquet.schema),
        "preview": preview.astype(str).to_dict(orient="records"),
    }


def profile_npy(archive: zipfile.ZipFile, member_path: str) -> dict[str, Any]:
    with archive.open(member_path) as handle:
        array = np.load(io.BytesIO(handle.read()), allow_pickle=False)
    preview = array.ravel()[:8].astype(float).tolist() if np.issubdtype(array.dtype, np.number) else []
    return {
        "format": "npy",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "preview": preview,
    }


def profile_archive(zip_path: Path) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in {".csv", ".parquet", ".npy"}:
                continue
            record: dict[str, Any] = {
                "member_path": info.filename,
                "file_name": Path(info.filename).name,
                "suffix": suffix,
                "compressed_size": info.compress_size,
                "file_size": info.file_size,
            }
            try:
                if suffix == ".csv":
                    record.update(profile_csv(archive, info.filename))
                elif suffix == ".parquet":
                    record.update(profile_parquet(archive, info.filename))
                elif suffix == ".npy":
                    record.update(profile_npy(archive, info.filename))
            except Exception as exc:  # noqa: BLE001 - report profiling failures, keep going.
                record["format"] = suffix.lstrip(".")
                record["error"] = repr(exc)
            profiles.append(record)
    return profiles


def write_report(path: Path, profiles: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix_counts = Counter(profile["suffix"] for profile in profiles)
    top_dirs = Counter(profile["member_path"].split("/", 1)[0] for profile in profiles)
    lines = [
        "# Raw Table Schema Report",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Profiled files: {len(profiles)}",
        "",
        "## File Types",
        "",
    ]
    for suffix, count in sorted(suffix_counts.items()):
        lines.append(f"- {suffix}: {count}")
    lines.extend(["", "## Top-Level Folders", ""])
    for top_dir, count in sorted(top_dirs.items()):
        lines.append(f"- {top_dir}: {count}")
    lines.extend(["", "## Representative Schemas", ""])

    seen: set[str] = set()
    for profile in profiles:
        top_dir = profile["member_path"].split("/", 1)[0]
        key = f"{top_dir}:{profile['suffix']}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"### {profile['member_path']}")
        if "error" in profile:
            lines.append(f"- Error: `{profile['error']}`")
        else:
            if profile["format"] == "parquet":
                lines.append(f"- Rows: {profile['num_rows']}")
            if profile["format"] == "npy":
                lines.append(f"- Shape: {profile['shape']}; dtype: `{profile['dtype']}`")
            columns = profile.get("columns")
            if columns:
                lines.append("- Columns: " + ", ".join(f"`{column}`" for column in columns))
            preview = profile.get("preview")
            if preview:
                lines.append("- Preview:")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(preview[:2], indent=2)[:2000])
                lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="data_raw/data_EvalSIB.zip")
    parser.add_argument("--output-json", default="data_processed/qc/raw_table_schemas.json")
    parser.add_argument("--report", default="logs/raw_table_schema_report.md")
    args = parser.parse_args()

    profiles = profile_archive(Path(args.zip_path))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    write_report(Path(args.report), profiles)
    print(f"profiled: {len(profiles)}")
    print(f"schemas: {output_json}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
