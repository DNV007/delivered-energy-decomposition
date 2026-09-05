#!/usr/bin/env python3
"""How much cell selection would have to change to reverse each reported ordering.

Every headline comparison in this work is a ratio between two individual cells.
Two questions follow, and neither needs a distributional assumption to state
the first:

1. **Reversal threshold.** If the larger value is shifted down by a fraction x
   and the smaller up by the same fraction, the ordering reverses when
   r(1-x) = (1+x), that is x = (r-1)/(r+1). This is a descriptive robustness
   threshold, not a confidence interval.

2. **Scenario probability.** Given a dispersion for each family, and treating
   two draws as independent lognormals, the ordering survives with probability
   Phi(ln r / sqrt(sigma_a^2 + sigma_b^2)), sigma = sqrt(ln(1 + CV^2)).

The dispersion used here is measured, not assumed: the archive's processed
HPPC files carry three cells per family, giving a CV of the median 1C
discharge R_1s (`build_cell_replicate_analysis.py`). Three important limits
follow the script into the manuscript:

- those files carry no temperature label, so a room-temperature CV is being
  transferred to 5 degC as a *scenario*; published population work reports
  that dispersion itself changes with temperature, so this is an assumption
  and not a measurement;
- a CV from three cells is an order-of-magnitude figure;
- the growth factor and activation energy are within-cell ratios, so no
  cell-to-cell dispersion measured at one temperature can be applied to them.
  Their reversal thresholds are reported; no probability is.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PULSE = "results/tables/table_raw_pulse_temperature_summary.csv"
CAPACITY = "results/tables/table_capacity_energy_retention.csv"
ENERGY = "results/tables/table_delivered_energy_decomposition.csv"
SPREAD = "results/tables/table_cell_replicate_spread.csv"
OUTPUT = "results/tables/table_ordering_robustness.csv"

# Two independent populations of the same vendor's sodium-ion 18650 product
# bracket its resistance dispersion. Neither uses our descriptor, so both are
# scenarios, never measurements of our cells; three archive replicates cannot
# sample a tail, which is why an external bound is worth having at all.
#
#   low  - Kemeny et al. (2026), 100 cells: GITT-derived internal resistance
#          varies by less than 10 % across all SOC regions (their fitted ohmic
#          term varies by 2.96 %, their charge-transfer term by 21.6 %).
#   high - Wittman et al. (2026), 20 cells: normalised resistance 159 mOhm Ah
#          with a spread of 38, i.e. 24 %, including one cell beyond 2 sigma.
EXTERNAL_SIB_CV_LOW = 0.10
EXTERNAL_SIB_CV_HIGH = 38.0 / 159.0


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / path)


def reversal_threshold(ratio: float) -> float:
    """Equal and opposite multiplicative shift that erases the ordering."""
    return 100.0 * (ratio - 1.0) / (ratio + 1.0)


def survival_probability(ratio: float, cv_a: float, cv_b: float) -> float:
    """P(a > b) for independent lognormals with the given coefficients of variation."""
    sigma_a = math.sqrt(math.log(1.0 + cv_a**2))
    sigma_b = math.sqrt(math.log(1.0 + cv_b**2))
    spread = math.hypot(sigma_a, sigma_b)
    if spread == 0.0:
        return 100.0
    z = math.log(ratio) / spread
    return 100.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    pulse = read(PULSE)
    discharge = pulse[(pulse["pulse_direction"] == "discharge") & (pulse["abs_c_rate"] == 1.0)]
    grid = discharge.pivot_table(
        index="cell_family", columns="temperature_deg_c", values="median_r_1s_mohm"
    )
    capacity = read(CAPACITY)
    cold_capacity = capacity[capacity["temperature_deg_c"] == 5.0].set_index("cell_family")
    energy = read(ENERGY)
    cold_energy = energy[energy["temperature_deg_c"] == 5.0].set_index("cell_family")

    spread = read(SPREAD)
    spread = spread[spread["pulse_direction"] == "discharge"].set_index("cell_family")
    cv = {family: float(spread.loc[family, "cv_percent"]) / 100.0 for family in spread.index}

    growth = grid[5.0] / grid[25.0]

    comparisons = [
        # label, ratio, whether the measured CVs may be applied
        (
            "R_1s at 5 degC, SIB / LFP",
            float(grid.loc["SIB", 5.0] / grid.loc["LFP", 5.0]),
            ("SIB", "LFP"),
        ),
        (
            "R_1s at 5 degC, SIB / NMC",
            float(grid.loc["SIB", 5.0] / grid.loc["NMC", 5.0]),
            ("SIB", "NMC"),
        ),
        (
            "R_1s at 25 degC, SIB / LFP",
            float(grid.loc["SIB", 25.0] / grid.loc["LFP", 25.0]),
            ("SIB", "LFP"),
        ),
        (
            "R_1s growth 25->5 degC, SIB / LFP",
            float(growth["SIB"] / growth["LFP"]),
            None,
        ),
        (
            "R_1s growth 25->5 degC, SIB / NMC",
            float(growth["SIB"] / growth["NMC"]),
            None,
        ),
        (
            "capacity retention at 5 degC, LFP / SIB",
            float(
                cold_capacity.loc["LFP", "discharge_capacity_retention_vs_25c"]
                / cold_capacity.loc["SIB", "discharge_capacity_retention_vs_25c"]
            ),
            None,
        ),
        (
            "delivered-energy retention at 5 degC, LFP / SIB",
            float(
                cold_energy.loc["LFP", "delivered_energy_wh_retention_vs_25c"]
                / cold_energy.loc["SIB", "delivered_energy_wh_retention_vs_25c"]
            ),
            None,
        ),
    ]

    rows = []
    for label, ratio, families in comparisons:
        row = {
            "comparison": label,
            "ratio": ratio,
            "reversal_threshold_percent": reversal_threshold(ratio),
            "cv_a_percent": None,
            "cv_b_percent": None,
            "survival_probability_percent": None,
            "cv_a_external_low_percent": None,
            "survival_probability_external_low_percent": None,
            "cv_a_external_high_percent": None,
            "survival_probability_external_high_percent": None,
        }
        if families is not None:
            a, b = families
            row["cv_a_percent"] = 100.0 * cv[a]
            row["cv_b_percent"] = 100.0 * cv[b]
            row["survival_probability_percent"] = survival_probability(ratio, cv[a], cv[b])
            # Further scenarios: the SIB dispersion replaced by each external
            # population bound, the references left as measured.
            for suffix, external in [
                ("external_low", EXTERNAL_SIB_CV_LOW),
                ("external_high", EXTERNAL_SIB_CV_HIGH),
            ]:
                external_a = external if a == "SIB" else cv[a]
                external_b = external if b == "SIB" else cv[b]
                row[f"cv_a_{suffix}_percent"] = 100.0 * external_a
                row[f"survival_probability_{suffix}_percent"] = survival_probability(
                    ratio, external_a, external_b
                )
        rows.append(row)

    table = pd.DataFrame(rows)
    path = PROJECT_ROOT / OUTPUT
    ensure_parent(path)
    table.to_csv(path, index=False)

    if not args.quiet:
        print("Reversal thresholds and, where within-cell ratios do not forbid it,")
        print("survival probabilities under the archive's measured dispersion:\n")
        for _, row in table.iterrows():
            probability = row["survival_probability_percent"]
            tail = (
                f"   P = {probability:5.1f} % [archive]"
                f"   P = {row['survival_probability_external_high_percent']:5.1f}"
                f"-{row['survival_probability_external_low_percent']:.1f} %"
                f" [external SIB CV {row['cv_a_external_high_percent']:.0f}"
                f"-{row['cv_a_external_low_percent']:.0f} %]"
                if pd.notna(probability)
                else "   (within-cell ratio: no dispersion estimate applies)"
            )
            print(
                f"  {row['comparison']:<48} r={row['ratio']:6.3f}"
                f"  reversal at +/-{row['reversal_threshold_percent']:5.1f} %{tail}"
            )
        print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
