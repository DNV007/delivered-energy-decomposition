"""Metadata helpers for the EnergyStorage workflow."""

from __future__ import annotations

import re
from pathlib import Path


CELL_ID_RE = re.compile(r"(?<![A-Za-z0-9])(TC\d{2}(?:SIB|LFP|LTO|NMC)\d{2})(?![A-Za-z0-9])", re.I)
CELL_FAMILY_RE = re.compile(r"(?<![A-Za-z0-9])TC\d{2}(SIB|LFP|LTO|NMC)\d{2}(?![A-Za-z0-9])", re.I)
STANDALONE_FAMILY_RE = re.compile(r"(?:^|[_/\W])(SIB|LFP|LTO|NMC)(?:$|[_/\W])", re.I)
TEMP_RE = re.compile(r"(?<!\d)(\d{1,3})\s*(?:degc|deg_c|deg|gradc|grad_c|celsius|°c)(?![a-z0-9])", re.I)
VOLTAGE_MV_RE = re.compile(r"(?<!\d)(\d{3,5})\s*mV(?![a-z0-9])", re.I)


def extract_cell_id(text: str) -> str:
    match = CELL_ID_RE.search(text)
    return match.group(1).upper() if match else ""


def infer_cell_family(text: str) -> str:
    match = CELL_FAMILY_RE.search(text)
    if match:
        return match.group(1).upper()
    match = STANDALONE_FAMILY_RE.search(text)
    return match.group(1).upper() if match else ""


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


def extract_temperature_deg_c(text: str) -> float | None:
    match = TEMP_RE.search(text)
    return float(int(match.group(1))) if match else None


def extract_voltage_v(text: str) -> float | None:
    match = VOLTAGE_MV_RE.search(text)
    return float(match.group(1)) / 1000.0 if match else None


def hppc_file_direction(path: str) -> str:
    name = Path(path).name.lower()
    if "_ch_" in name:
        return "charge_file"
    if "_dis_" in name:
        return "discharge_file"
    return ""


def entropy_direction(path: str) -> str:
    name = Path(path).name.lower()
    if "discharge" in name:
        return "discharge"
    if "charge" in name:
        return "charge"
    return ""


def normal_metadata(member_path: str) -> dict[str, object]:
    return {
        "member_path": member_path,
        "file_name": Path(member_path).name,
        "cell_id": extract_cell_id(member_path),
        "cell_family": infer_cell_family(member_path),
        "chemistry_type": infer_chemistry(member_path),
        "temperature_deg_c": extract_temperature_deg_c(member_path),
        "voltage_v": extract_voltage_v(member_path),
    }
