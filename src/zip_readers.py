"""Read raw data members directly from the DepositOnce zip archive."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def iter_members(zip_path: Path, suffixes: set[str] | None = None) -> Iterable[str]:
    with zipfile.ZipFile(zip_path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffixes is None or suffix in suffixes:
                yield info.filename


def read_csv_member(zip_path: Path, member_path: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_path) as handle:
            return pd.read_csv(handle, **kwargs)


def read_parquet_member(zip_path: Path, member_path: str, columns: list[str] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_path) as handle:
            payload = handle.read()
    return pd.read_parquet(io.BytesIO(payload), columns=columns)


def read_npy_member(zip_path: Path, member_path: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_path) as handle:
            return np.load(io.BytesIO(handle.read()), allow_pickle=False)
