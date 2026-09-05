#!/usr/bin/env python3
"""Download DepositOnce item bitstreams into data_raw.

The script uses the public DSpace REST API exposed by DepositOnce. It does not
modify downloaded files and writes a manifest describing exactly what was
fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_ROOT = "https://api-depositonce.tu-berlin.de/server/api"


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "energy-storage-bottleneck-workflow/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def embedded_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    embedded = payload.get("_embedded", {})
    value = embedded.get(key, [])
    return value if isinstance(value, list) else []


def paged_embedded(url: str, key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    next_url: str | None = url
    while next_url:
        payload = fetch_json(next_url)
        records.extend(embedded_list(payload, key))
        next_link = payload.get("_links", {}).get("next", {}).get("href")
        next_url = next_link if next_link else None
    return records


def bitstream_content_url(bitstream: dict[str, Any]) -> str | None:
    content = bitstream.get("_links", {}).get("content", {})
    href = content.get("href")
    return href if isinstance(href, str) else None


def bitstream_name(bitstream: dict[str, Any]) -> str:
    name = bitstream.get("name") or bitstream.get("uuid") or "bitstream"
    return Path(str(name)).name


def download_file(url: str, destination: Path, overwrite: bool) -> dict[str, Any]:
    if destination.exists() and not overwrite:
        return {
            "path": str(destination),
            "status": "skipped_existing",
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "energy-storage-bottleneck-workflow/0.1"},
    )
    digest = hashlib.sha256()
    bytes_written = 0
    started = time.time()

    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)

    return {
        "path": str(destination),
        "status": "downloaded",
        "size_bytes": bytes_written,
        "sha256": digest.hexdigest(),
        "elapsed_seconds": round(time.time() - started, 2),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_item_bitstreams(item_uuid: str) -> list[dict[str, Any]]:
    bundles_url = f"{API_ROOT}/core/items/{urllib.parse.quote(item_uuid)}/bundles"
    bundles = paged_embedded(bundles_url, "bundles")
    bitstreams: list[dict[str, Any]] = []

    for bundle in bundles:
        bundle_name = bundle.get("name")
        bitstreams_url = bundle.get("_links", {}).get("bitstreams", {}).get("href")
        if not isinstance(bitstreams_url, str):
            continue
        for bitstream in paged_embedded(bitstreams_url, "bitstreams"):
            bitstream["_bundle_name"] = bundle_name
            bitstreams.append(bitstream)

    return bitstreams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item-uuid",
        default="5e9d5736-191b-4a8e-ba11-cb04b962a381",
        help="DepositOnce item UUID from the item URL.",
    )
    parser.add_argument("--output-dir", default="data_raw")
    parser.add_argument("--manifest", default="data_raw/download_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional exact bitstream file names to download.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    bitstreams = resolve_item_bitstreams(args.item_uuid)

    selected = []
    for bitstream in bitstreams:
        name = bitstream_name(bitstream)
        if args.names and name not in args.names:
            continue
        content_url = bitstream_content_url(bitstream)
        if not content_url:
            continue
        selected.append((name, content_url, bitstream))

    if not selected:
        print("No downloadable bitstreams found.", file=sys.stderr)
        return 1

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "item_uuid": args.item_uuid,
        "api_root": API_ROOT,
        "files": [],
    }

    for name, content_url, bitstream in selected:
        destination = output_dir / name
        try:
            result = download_file(content_url, destination, args.overwrite)
        except urllib.error.URLError as exc:
            result = {
                "path": str(destination),
                "status": "failed",
                "error": str(exc),
            }
        result.update(
            {
                "name": name,
                "bundle": bitstream.get("_bundle_name"),
                "uuid": bitstream.get("uuid"),
                "content_url": content_url,
            }
        )
        manifest["files"].append(result)
        print(f"{result['status']}: {name} -> {destination}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
