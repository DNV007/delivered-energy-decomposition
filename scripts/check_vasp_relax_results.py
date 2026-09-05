#!/usr/bin/env python3
"""Inspect VASP relaxation outputs copied back from JUSTUS2."""

from __future__ import annotations

import argparse
import csv
import re
import tarfile
import tempfile
from pathlib import Path


ENERGY_RE = re.compile(r"free\s+energy\s+TOTEN\s+=\s+([-+0-9.Ee]+)")
FORCE_RE = re.compile(r"reached required accuracy|aborting loop because EDIFF is reached")


def find_file(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[-1] if matches else None


def extract_archive(archive: Path, destination: Path) -> Path:
    target = destination / archive.stem.replace(".tar", "")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as handle:
        handle.extractall(target)
    return target


def parse_outcar(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {
            "outcar_present": False,
            "converged_text": False,
            "final_toten_ev": "",
            "n_toten_steps": 0,
        }
    text = path.read_text(errors="ignore")
    energies = [float(match.group(1)) for match in ENERGY_RE.finditer(text)]
    return {
        "outcar_present": True,
        "converged_text": bool(FORCE_RE.search(text)),
        "final_toten_ev": energies[-1] if energies else "",
        "n_toten_steps": len(energies),
    }


def parse_oszicar(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {"oszicar_present": False, "ionic_steps": 0}
    ionic_steps = 0
    for line in path.read_text(errors="ignore").splitlines():
        if "F=" in line and "E0=" in line:
            ionic_steps += 1
    return {"oszicar_present": True, "ionic_steps": ionic_steps}


def inspect_root(root: Path, label: str) -> dict[str, object]:
    outcar = find_file(root, "OUTCAR")
    oszicar = find_file(root, "OSZICAR")
    contcar = find_file(root, "CONTCAR")
    stdout = sorted(root.rglob("*.out")) + sorted(root.rglob("*.err"))
    record = {
        "label": label,
        "root": str(root),
        "contcar_present": bool(contcar and contcar.exists() and contcar.stat().st_size > 0),
        "contcar_path": str(contcar) if contcar else "",
        "stdout_files": ";".join(str(path) for path in stdout),
    }
    record.update(parse_outcar(outcar))
    record.update(parse_oszicar(oszicar))
    record["status"] = "pass" if record["contcar_present"] and record["outcar_present"] else "review"
    return record


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "label",
        "status",
        "root",
        "contcar_present",
        "contcar_path",
        "outcar_present",
        "converged_text",
        "final_toten_ev",
        "n_toten_steps",
        "oszicar_present",
        "ionic_steps",
        "stdout_files",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def write_report(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# VASP Relaxation Result Check", ""]
    for record in records:
        lines.extend(
            [
                f"## {record['label']}",
                "",
                f"- Status: {record['status']}",
                f"- CONTCAR present: {record['contcar_present']}",
                f"- OUTCAR present: {record['outcar_present']}",
                f"- Convergence text found: {record['converged_text']}",
                f"- Final TOTEN: {record['final_toten_ev']} eV",
                f"- Ionic steps: {record['ionic_steps']}",
                f"- CONTCAR: `{record['contcar_path']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="VASP job directories or JUSTUS2 .tgz archives.")
    parser.add_argument("--out", default="dft_calculations/results/relax_result_check.csv")
    parser.add_argument("--report", default="dft_calculations/results/relax_result_check.md")
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="vasp-relax-check-") as tmp:
        tmp_path = Path(tmp)
        for item in args.paths:
            path = Path(item)
            if path.suffix in {".tgz", ".gz", ".tar"}:
                root = extract_archive(path, tmp_path)
                label = path.name
            else:
                root = path
                label = path.name
            records.append(inspect_root(root, label))
    write_csv(Path(args.out), records)
    write_report(Path(args.report), records)
    for record in records:
        print(f"{record['label']}: {record['status']} final_TOTEN={record['final_toten_ev']} contcar={record['contcar_present']}")
    return 0 if all(record["status"] == "pass" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
