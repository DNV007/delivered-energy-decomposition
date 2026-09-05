#!/usr/bin/env python3
"""How far does the cell temperature move inside the incremental discharge?

The Arrhenius fits use the median measured cell temperature, but the
decomposition and the pulse resistances are read off traces whose comparability
between temperatures assumes the cell stayed near its setpoint throughout. With
Ea = 373 meV a 3.5 K rise changes sodium-ion resistance by about a fifth, so a
self-heating excursion of that size inside the 5 degC discharge would affect
the shape of the polarisation curve, the eighteen cut-short segments, and the
comparability of the tail. The load duty cycle is only about 10 %, which makes
a large rise unlikely -- but unlikely is not measured.

This script reports, for every family and temperature, the distribution of the
measured cell temperature over the discharge: during the current segments
(StepID 96), during the rests (97), and over both together. The excursion is
quoted against the median of the rests, which is the closest available proxy
for the chamber setpoint, and converted into the resistance error it implies
through the family's own fitted activation energy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zip_readers import read_parquet_member

ARCHIVE = PROJECT_ROOT / "data_raw/data_EvalSIB.zip"
ARRHENIUS = PROJECT_ROOT / "results/tables/table_arrhenius_activation_energy.csv"
OUTPUT = PROJECT_ROOT / "results/tables/table_discharge_thermal_excursion.csv"

COLUMNS = ["Testtime[s]", "StepID", "Temperature[°C]", "Current[A]"]
CELLS = {"SIB": "TC23SIB09", "LFP": "TC23LFP09", "NMC": "TC23NMC09", "LTO": "TC23LTO09"}
TEMPERATURES = {5: "05deg", 15: "15deg", 25: "25deg", 35: "35deg", 45: "45deg"}
FAMILY_ORDER = ["SIB", "LFP", "NMC", "LTO"]

DISCHARGE_STEP = 96
REST_STEP = 97
BOLTZMANN_EV_PER_K = 8.617333262e-5


def load_discharge(family: str, temperature_c: int) -> pd.DataFrame:
    cell = CELLS[family]
    member = f"raw_CU_diffT/{cell}/{cell}_{TEMPERATURES[temperature_c]}.parquet"
    frame = read_parquet_member(ARCHIVE, member, columns=COLUMNS)
    frame.columns = [name.split("[")[0] for name in frame.columns]
    frame = frame.sort_values("Testtime").reset_index(drop=True)

    window = frame[frame["StepID"].isin({DISCHARGE_STEP, REST_STEP})]
    # Keep only the first contiguous run of the discharge/rest pair: StepIDs 96
    # and 97 do not recur elsewhere in the checkup, but a gap check is cheap.
    blocks = window["Testtime"].diff().gt(600).cumsum()
    return window[blocks == 0]


def activation_energies() -> dict[str, float]:
    if not ARRHENIUS.exists():
        return {}
    table = pd.read_csv(ARRHENIUS)
    column = next(
        (name for name in table.columns if "activation_energy" in name and "ev" in name.lower()),
        None,
    )
    if column is None:
        return {}
    # The largest fitted value per family, so the excursion is converted into
    # the resistance error at the family's most temperature-sensitive
    # condition rather than at a typical one.
    return (
        table.dropna(subset=[column])
        .groupby("cell_family")[column]
        .max()
        .to_dict()
    )


def main() -> int:
    pd.set_option("display.width", 200)
    energies = activation_energies()

    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        for temperature_c in TEMPERATURES:
            discharge = load_discharge(family, temperature_c)
            under_load = discharge[discharge["StepID"] == DISCHARGE_STEP]["Temperature"].dropna()
            at_rest = discharge[discharge["StepID"] == REST_STEP]["Temperature"].dropna()
            everything = discharge["Temperature"].dropna()
            if everything.empty:
                continue

            baseline = float(at_rest.median()) if not at_rest.empty else float(everything.median())
            peak_rise = float(everything.max() - baseline)

            duty = float(
                discharge["StepID"].eq(DISCHARGE_STEP).sum() / discharge["StepID"].size
            )

            activation_ev = energies.get(family, np.nan)
            reference_k = 273.15 + baseline
            resistance_error = (
                100.0
                * (
                    np.exp(
                        activation_ev
                        / BOLTZMANN_EV_PER_K
                        * (1.0 / (reference_k + peak_rise) - 1.0 / reference_k)
                    )
                    - 1.0
                )
                if np.isfinite(activation_ev)
                else np.nan
            )

            rows.append(
                {
                    "cell_family": family,
                    "temperature_deg_c": float(temperature_c),
                    "rest_median_c": baseline,
                    "load_median_c": float(under_load.median()) if not under_load.empty else np.nan,
                    "median_c": float(everything.median()),
                    "iqr_k": float(everything.quantile(0.75) - everything.quantile(0.25)),
                    "min_c": float(everything.min()),
                    "max_c": float(everything.max()),
                    "peak_rise_over_rest_k": peak_rise,
                    "p99_rise_over_rest_k": float(everything.quantile(0.99) - baseline),
                    "load_duty_fraction": duty,
                    "activation_energy_ev": activation_ev,
                    "implied_resistance_error_percent": resistance_error,
                }
            )

    table = pd.DataFrame.from_records(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)

    print("Cell temperature over the incremental discharge (StepID 96 + 97)")
    print(
        table[
            [
                "cell_family",
                "temperature_deg_c",
                "median_c",
                "iqr_k",
                "max_c",
                "peak_rise_over_rest_k",
                "load_duty_fraction",
                "activation_energy_ev",
                "implied_resistance_error_percent",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    worst = table.loc[table["peak_rise_over_rest_k"].idxmax()]
    print(
        f"\nLargest excursion: {worst['cell_family']} at {worst['temperature_deg_c']:.0f} degC, "
        f"{worst['peak_rise_over_rest_k']:.2f} K above the rest median, implying "
        f"{worst['implied_resistance_error_percent']:.1f} % on resistance."
    )
    print(f"wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
