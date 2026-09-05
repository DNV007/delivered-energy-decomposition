#!/usr/bin/env python3
"""Fetch the benchmark relaxation dataset used in the SI applicability check.

Fernando, Kuipers, Angenendt, Kairies and Dubarry, "Relaxation data for
commercial NMC and LFP cells", Mendeley Data 10.17632/y8nstxmdrg.1 (CC BY 4.0),
supporting Cell Reports Physical Science 5, 101754 (2024).

The deposit is small (1.2 MB) and is not redistributed here, for the same reason
the DepositOnce archive is not: it belongs to its authors. Section S1 of the
supplement reports what it does and does not supply, and check_relaxation_geometry.py
re-derives that from these files.

The archived release is a single version, so the payload is pinned by SHA-256
rather than by version string.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data_external/relaxation_benchmark"
URL = ("https://data.mendeley.com/public-files/datasets/y8nstxmdrg/files/"
       "434dc098-a969-49a4-9289-a5a4e40d1b3e/file_downloaded")
SHA256 = "e8254086952f6b8de281d7b1c0faccd67156219ca9eaed50c181f65fc7c0966a"
SIZE = 1_247_758


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "formatted_relaxation_data.zip"

    if not archive.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)

    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != SIZE or digest != SHA256:
        print(f"FAIL: expected {SIZE} bytes / {SHA256}", file=sys.stderr)
        print(f"      got      {len(payload)} bytes / {digest}", file=sys.stderr)
        return 1
    print(f"verified {len(payload)} bytes, sha256 {digest}")

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DEST)
    for path in sorted(DEST.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(DEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
