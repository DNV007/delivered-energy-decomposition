#!/usr/bin/env python3
"""Verify every headline number quoted in the manuscript against its source table.

The manuscript quotes the same descriptors in the abstract, the results text,
Table 3, the figure captions, and the Supplementary Information. Keeping those
in agreement by hand does not survive revision: the two substantive errors
found in review were both cases of a number that had stopped matching its
table. This script re-derives each quoted value from the CSV that is cited as
its source and fails loudly on any drift.

Run after regenerating any results table and before every submission build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAMILIES = ["SIB", "LFP", "LTO", "NMC"]
REFERENCES = ["LFP", "LTO", "NMC"]
# The paper's reference average excludes LTO (Section 2.1); REFERENCES is
# retained only for the two LTO-inclusive conventions quoted in Table 3's
# footnote c.
GRAPHITE = ["LFP", "NMC"]


ROWS_MATCHED = 0
TABLES_READ: set[str] = set()


def read(path: str) -> pd.DataFrame:
    full = PROJECT_ROOT / path
    if not full.exists():
        raise SystemExit(f"source table missing: {path}")
    frame = pd.read_csv(full)
    if frame.empty:
        raise SystemExit(f"source table is empty: {path}")
    TABLES_READ.add(path)
    return frame


def one_row(frame: pd.DataFrame, what: str, **keys):
    """Select the single row matching every key, or fail.

    Every lookup here used .iloc[0] on a filtered frame, which returns the first
    match and says nothing about how many there were. That is safe only while a
    table has exactly the key columns the caller assumes. When the stopping-line
    sweep gained a second cold point, the old filter silently began choosing
    between two rows on row order alone. Uniqueness is now asserted, so a table
    that grows a dimension breaks loudly instead of answering with a coin flip.
    """
    global ROWS_MATCHED
    mask = np.ones(len(frame), dtype=bool)
    for column, value in keys.items():
        if column not in frame.columns:
            raise SystemExit(f"{what}: no column {column!r} in table "
                             f"(has {list(frame.columns)})")
        mask &= (frame[column] == value).to_numpy()
    hits = frame[mask]
    if len(hits) != 1:
        raise SystemExit(f"{what}: expected exactly 1 row for {keys}, found {len(hits)}")
    ROWS_MATCHED += 1
    return hits.iloc[0]


class Checker:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, float, float, bool]] = []

    def check(self, label: str, quoted: float, derived: float, tolerance: float) -> None:
        ok = bool(np.isfinite(derived)) and abs(quoted - derived) <= tolerance
        self.records.append((label, f"{tolerance:g}", quoted, derived, ok))

    def report(self) -> int:
        width = max(len(record[0]) for record in self.records)
        failures = 0
        for label, tolerance, quoted, derived, ok in self.records:
            status = "ok  " if ok else "FAIL"
            if not ok:
                failures += 1
            print(
                f"  {status}  {label:<{width}}  quoted={quoted:<10.4g} "
                f"derived={derived:<10.4g} tol={tolerance}"
            )
        print(f"\n{len(self.records) - failures}/{len(self.records)} manuscript numbers verified "
              f"from {len(TABLES_READ)} source table(s), {ROWS_MATCHED} keyed row(s) matched")
        if not self.records or not TABLES_READ:
            print("nothing was checked; treating that as a failure")
            failures += 1
        return failures


def main() -> int:
    checker = Checker()

    # --- capacity and OCV-window energy retention at 5 degC -------------
    capacity = read("results/tables/table_capacity_energy_retention.csv")
    cold = capacity[capacity["temperature_deg_c"] == 5.0].set_index("cell_family")
    for family, quoted in {"SIB": 0.816, "LFP": 0.956, "LTO": 0.984, "NMC": 0.966}.items():
        checker.check(
            f"capacity retention 5C {family}",
            quoted,
            float(cold.loc[family, "discharge_capacity_retention_vs_25c"]),
            0.001,
        )
    checker.check(
        "capacity retention 5C Li-ion mean",
        0.961,
        float(cold.loc[GRAPHITE, "discharge_capacity_retention_vs_25c"].mean()),
        0.001,
    )
    for family, quoted in {"SIB": 0.825, "LFP": 0.948, "LTO": 0.986, "NMC": 0.965}.items():
        checker.check(
            f"OCV-window energy retention 5C {family}",
            quoted,
            float(cold.loc[family, "energy_retention_vs_25c"]),
            0.001,
        )

    # --- raw-pulse resistance -------------------------------------------
    pulse = read("results/tables/table_raw_pulse_temperature_summary.csv")
    discharge = pulse[(pulse["pulse_direction"] == "discharge") & (pulse["abs_c_rate"] == 1.0)]
    grid = discharge.pivot_table(
        index="cell_family", columns="temperature_deg_c", values="median_r_1s_mohm"
    )
    for family, quoted in {"SIB": 91.1, "LFP": 83.2, "LTO": 24.8, "NMC": 42.9}.items():
        checker.check(f"R_1s 25C {family}", quoted, float(grid.loc[family, 25.0]), 0.1)
    for family, quoted in {"SIB": 286.6, "LFP": 189.0, "LTO": 32.7, "NMC": 99.0}.items():
        checker.check(f"R_1s 5C {family}", quoted, float(grid.loc[family, 5.0]), 0.1)
    checker.check("R_1s 25C Li-ion mean", 63.0, float(grid.loc[GRAPHITE, 25.0].mean()), 0.1)
    checker.check("R_1s 5C Li-ion mean", 144.0, float(grid.loc[GRAPHITE, 5.0].mean()), 0.1)

    growth = grid[5.0] / grid[25.0]
    for family, quoted in {"SIB": 3.15, "LFP": 2.27, "LTO": 1.32, "NMC": 2.31}.items():
        checker.check(f"R_1s growth 25->5C {family}", quoted, float(growth[family]), 0.01)
    # Convention: mean of per-family growth factors, NOT growth of the mean.
    checker.check("growth factor, mean of ratios", 2.29, float(growth[GRAPHITE].mean()), 0.01)
    checker.check(
        "growth factor, ratio of means (stated as the alternative)",
        2.29,
        float(grid.loc[GRAPHITE, 5.0].mean() / grid.loc[GRAPHITE, 25.0].mean()),
        0.01,
    )

    sib_over_family = grid.loc["SIB", 5.0] / grid[5.0]
    for family, quoted in {"LFP": 1.52, "LTO": 8.77, "NMC": 2.89}.items():
        checker.check(f"SIB/{family} at 5C", quoted, float(sib_over_family[family]), 0.01)
    # Table 3's aggregate entry is the graphite-anode comparison: LTO's 8.77
    # dominates any average containing it, and the paper argues LTO is not a
    # competing cell. The two LTO-inclusive conventions stay pinned because
    # footnote c still quotes them.
    graphite = ["LFP", "NMC"]
    checker.check(
        "SIB/graphite-anode mean at 5C (Table 3 aggregate)",
        1.99,
        float(grid.loc["SIB", 5.0] / grid.loc[graphite, 5.0].mean()),
        0.01,
    )
    checker.check(
        "SIB/graphite-anode mean at 25C",
        1.45,
        float(grid.loc["SIB", 25.0] / grid.loc[graphite, 25.0].mean()),
        0.01,
    )
    checker.check(
        "SIB/family at 5C, mean of the three ratios (footnote c)",
        4.39,
        float(sib_over_family[REFERENCES].mean()),
        0.01,
    )
    checker.check(
        "SIB/reference at 5C, ratio of means (stated as the alternative)",
        2.68,
        float(grid.loc["SIB", 5.0] / grid.loc[REFERENCES, 5.0].mean()),
        0.01,
    )
    checker.check("SIB/LFP at 25C", 1.10, float(grid.loc["SIB", 25.0] / grid.loc["LFP", 25.0]), 0.01)
    checker.check("SIB/NMC at 25C", 2.13, float(grid.loc["SIB", 25.0] / grid.loc["NMC", 25.0]), 0.01)

    # Model-free temperature sensitivity, quoted in place of a fitted form.
    model_free = grid[5.0] / grid[45.0]
    for family, quoted in {"SIB": 6.73, "LFP": 4.21, "LTO": 1.47, "NMC": 3.43}.items():
        checker.check(f"R(5C)/R(45C) {family}", quoted, float(model_free[family]), 0.01)

    charge = pulse[(pulse["pulse_direction"] == "charge") & (pulse["abs_c_rate"] == 1.0)]
    charge_grid = charge.pivot_table(
        index="cell_family", columns="temperature_deg_c", values="median_r_1s_mohm"
    )
    checker.check("charge R_1s 5C SIB", 232.8, float(charge_grid.loc["SIB", 5.0]), 0.1)
    checker.check(
        "charge R_1s 5C Li-ion mean", 154.8, float(charge_grid.loc[GRAPHITE, 5.0].mean()), 0.1
    )
    checker.check(
        "charge R_1s growth SIB",
        2.58,
        float(charge_grid.loc["SIB", 5.0] / charge_grid.loc["SIB", 25.0]),
        0.01,
    )

    # --- pulse currents are common across families ----------------------
    raw = read("data_processed/hppc/raw_temperature_pulse_descriptors.csv")
    one_c = raw[raw["nominal_c_rate"].abs() == 1.0]
    checker.check(
        "nominal-1C pulse current (A), common to all families",
        1.5,
        float(one_c["current_median_a"].abs().median()),
        0.001,
    )
    checker.check(
        "max spread of 1C pulse current across families (A)",
        0.0,
        float(one_c.groupby("cell_family")["current_median_a"].apply(lambda s: s.abs().median()).std()),
        0.001,
    )
    capacity_25 = read("data_processed/capacity/capacity_temperature_summary.csv")
    capacity_25 = capacity_25[capacity_25["temperature_deg_c"] == 25.0]
    effective_c = 1.5 / capacity_25.groupby("cell_family")["discharge_capacity_ah"].median()
    for family, quoted in {"SIB": 0.95, "LFP": 0.84, "LTO": 0.98, "NMC": 0.92}.items():
        checker.check(f"effective C-rate of the 1.5 A pulse, {family}", quoted, float(effective_c[family]), 0.01)

    # --- effective activation energies ----------------------------------
    arrhenius = read("results/tables/table_arrhenius_activation_energy.csv")
    headline = arrhenius[
        (arrhenius["pulse_direction"] == "discharge")
        & (arrhenius["abs_c_rate"] == 1.0)
        & (arrhenius["pulse_timepoint"] == "1s")
    ].set_index("cell_family")
    for family, quoted in {"SIB": 373, "LFP": 284, "LTO": 74, "NMC": 238}.items():
        checker.check(f"Ea 1s discharge {family} (meV)", quoted, float(headline.loc[family, "activation_energy_mev"]), 0.6)
    for family, quoted in {"SIB": 17, "LFP": 10, "LTO": 10, "NMC": 25}.items():
        checker.check(f"Ea standard error {family} (meV)", quoted, float(headline.loc[family, "standard_error_mev"]), 0.6)
    for family, quoted in {"SIB": 0.994, "LFP": 0.996, "LTO": 0.950, "NMC": 0.969}.items():
        checker.check(f"Ea fit R2 {family}", quoted, float(headline.loc[family, "r_squared"]), 0.001)
    checker.check(
        "Ea graphite mean (meV)", 261, float(headline.loc[GRAPHITE, "activation_energy_mev"].mean()), 0.6
    )

    # The Discussion states that the SIB activation-energy excess over the
    # graphite-anode references is directionally compatible with an
    # interphase contribution, given the higher computed Na-ion migration
    # barriers in NaF than in LiF (Yildirim et al. 2015, ~200 meV averaged
    # over their defect types). Round 31 cut the quantitative comparison to
    # one sentence, so 200 is no longer quoted in the text -- but the
    # compatibility claim still requires the measured whole-cell excess to be
    # SMALLER than that pure-phase increment, which is what series dilution
    # predicts. If it ever exceeds it, the sentence is wrong, so the ceiling
    # stays asserted here even though the number is no longer printed.
    graphite = headline.loc[["LFP", "NMC"], "activation_energy_mev"]
    sib_excess = float(headline.loc["SIB", "activation_energy_mev"] - graphite.mean())
    checker.check("SIB Ea excess over graphite-anode mean (meV)", 112, sib_excess, 1.0)
    checker.check(
        "SIB Ea excess over LFP (meV)",
        89,
        float(headline.loc["SIB", "activation_energy_mev"] - graphite["LFP"]),
        1.0,
    )
    checker.check(
        "SIB Ea excess over NMC (meV)",
        135,
        float(headline.loc["SIB", "activation_energy_mev"] - graphite["NMC"]),
        1.0,
    )
    assert sib_excess < 200, (
        "Measured whole-cell Ea excess now exceeds the pure-phase NaF/LiF "
        "increment; the series-dilution argument in the Discussion no longer holds."
    )

    # The manuscript states explicitly that the model-free ratio is NOT
    # independent of the fitted slope, and quotes this correlation as the
    # reason. If it ever drifts, the honesty caveat needs rewording too.
    checker.check(
        "corr(ln R5/R45, fitted Ea) across families, text says 0.9998",
        0.9998,
        float(np.corrcoef(np.log(model_free[headline.index]), headline["activation_energy_mev"])[0, 1]),
        0.0005,
    )

    # --- EIS spectral proxies -------------------------------------------
    eis = read("results/tables/table_eis_family_summary.csv").set_index("cell_family")
    for family, quoted in {"SIB": 88.9, "LFP": 70.6, "LTO": 18.4, "NMC": 31.0}.items():
        checker.check(f"low-freq Zre {family} (mOhm)", quoted, float(eis.loc[family, "median_low_freq_zre_mohm"]), 0.06)
    checker.check(
        "low-freq Zre graphite mean (mOhm)", 50.8, float(eis.loc[GRAPHITE, "median_low_freq_zre_mohm"].mean()), 0.06
    )
    for family, quoted in {"SIB": 71.8, "LFP": 50.2, "LTO": 6.1, "NMC": 15.7}.items():
        checker.check(f"arc width {family} (mOhm)", quoted, float(eis.loc[family, "median_arc_width_mohm"]), 0.06)
    for family, quoted in {"SIB": 10, "LFP": 5, "LTO": 0, "NMC": 1}.items():
        checker.check(f"spectra with resolved arc apex {family}", quoted, float(eis.loc[family, "n_arc_apex_resolved"]), 0)
    checker.check("SIB characteristic frequency (Hz)", 15.4, float(eis.loc["SIB", "median_characteristic_freq_hz"]), 0.05)

    subsets = read("results/tables/table_eis_subset_sensitivity.csv").set_index("subset")
    for label, quoted in {
        "all room-temperature spectra": 1.749,
        "Kramers-Kronig consistent only": 1.739,
        "narrow 24-26 degC window": 1.744,
        "KK-consistent and 24-26 degC": 1.749,
    }.items():
        checker.check(f"SIB/reference ratio, {label}", quoted, float(subsets.loc[label, "sib_to_reference_ratio"]), 0.001)

    # --- Kramers-Kronig --------------------------------------------------
    kk = read("results/tables/table_eis_kramers_kronig.csv")
    checker.check("KK spectra tested", 40, float(len(kk)), 0)
    checker.check("KK spectra passing", 37, float(kk["kk_consistent"].sum()), 0)
    checker.check("KK median RMS residual (%)", 0.33, float(kk["rms_residual"].median() * 100), 0.005)
    for family, quoted in {"SIB": 9, "LFP": 8, "LTO": 10, "NMC": 10}.items():
        checker.check(f"KK pass count {family}", quoted, float(kk[kk["cell_family"] == family]["kk_consistent"].sum()), 0)

    # --- equivalent-circuit screen ---------------------------------------
    ecm = read("results/tables/table_eis_ecm_fit_reliability.csv")
    checker.check("ECM fits total", 160, float(len(ecm)), 0)
    checker.check("ECM fits passing", 0, float(ecm["fit_reliable"].sum()), 0)
    by_model = ecm.groupby("model")["residual_nrmse"].median()
    checker.check("median NRMSE randles_cpe", 0.098, float(by_model["randles_cpe"]), 0.001)
    checker.check("median NRMSE two_arc_cpe", 0.091, float(by_model["two_arc_cpe"]), 0.001)
    checker.check("median NRMSE randles_cpe_warburg", 0.084, float(by_model["randles_cpe_warburg"]), 0.001)
    checker.check("median NRMSE two_arc_cpe_warburg", 0.084, float(by_model["two_arc_cpe_warburg"]), 0.001)

    # Threshold-sensitivity claims in Section 2.7. The screen's verdict rests on
    # the residual criterion alone, so the exact 0.05 must be shown not to matter
    # up to the best NRMSE any of the 160 cases attains.
    checker.check(
        "best NRMSE over all 160 spectrum-circuit cases",
        0.068, float(ecm["residual_nrmse"].min()), 0.0005,
    )
    checker.check(
        "cases passing residual at threshold 0.07",
        13, float(ecm["residual_nrmse"].le(0.07).sum()), 0,
    )
    checker.check(
        "cases passing residual at threshold 0.10",
        117, float(ecm["residual_nrmse"].le(0.10).sum()), 0,
    )
    checker.check(
        "cases passing residual at the stated 0.05 threshold",
        0, float(ecm["residual_nrmse"].le(0.05).sum()), 0,
    )
    # Section 2.7 states the multi-start counts per circuit.
    starts = ecm.groupby("model")["n_starting_points"].max()
    for model_name, quoted in (
        ("randles_cpe", 6), ("randles_cpe_warburg", 8),
        ("two_arc_cpe", 8), ("two_arc_cpe_warburg", 10),
    ):
        checker.check(f"starting points {model_name}", quoted,
                      float(starts[model_name]), 0)

    # Claims in Results 3.6 and the Discussion about what the second arc buys.
    # Round 43 cut the abstract to the journal's 250-word limit and these two
    # claims are no longer stated there; the labels below said "abstract" and
    # were retargeted. The pins still guard live text, so keep them.
    checker.check(
        "two-arc residual change vs single arc (%), text says <= 7",
        6.7,
        float((by_model["randles_cpe"] - by_model["two_arc_cpe"]) / by_model["randles_cpe"] * 100),
        0.2,
    )
    checker.check(
        "two-arc residual change with Warburg (%), text says none",
        0.0,
        float(
            (by_model["randles_cpe_warburg"] - by_model["two_arc_cpe_warburg"])
            / by_model["randles_cpe_warburg"]
            * 100
        ),
        0.05,
    )
    cv_by_model = ecm.groupby("model")["elite_parameter_cv"].median()
    checker.check(
        "orders of magnitude the second arc raises parameter dispersion",
        5.0,
        float(np.log10(cv_by_model["two_arc_cpe"] / cv_by_model["randles_cpe"])),
        0.5,
    )

    ecm_summary = read("results/tables/table_eis_ecm_fit_reliability_summary.csv")
    single_arc_cv = ecm_summary[
        ecm_summary["model"].isin(["randles_cpe", "randles_cpe_warburg"])
    ]["median_elite_parameter_cv"]
    # "three to five orders of magnitude inside the 0.20 stability criterion"
    checker.check(
        "min orders inside stability criterion (text says three)",
        3.0,
        float(np.log10(0.20 / single_arc_cv.max())),
        0.5,
    )
    checker.check(
        "max orders inside stability criterion (text says five)",
        5.0,
        float(np.log10(0.20 / single_arc_cv.min())),
        0.5,
    )
    single = ecm_summary[ecm_summary["model"] == "randles_cpe"].set_index("cell_family")
    for family, quoted in {"SIB": 0.071, "LFP": 0.081, "LTO": 0.115, "NMC": 0.098}.items():
        checker.check(f"randles_cpe median NRMSE {family}", quoted, float(single.loc[family, "median_residual_nrmse"]), 0.001)
    two_arc = ecm_summary[ecm_summary["model"] == "two_arc_cpe"].set_index("cell_family")
    for family, quoted in {"SIB": 0.071, "LFP": 0.080, "LTO": 0.111, "NMC": 0.098}.items():
        checker.check(f"two_arc_cpe median NRMSE {family}", quoted, float(two_arc.loc[family, "median_residual_nrmse"]), 0.001)

    # --- OCV slope, entropy, thermal -------------------------------------
    osp = read("results/tables/table_ocv_slope_summary.csv")
    osp_25 = osp[(osp["temperature_deg_c"] == 25.0) & (osp["direction"] == "discharge")].set_index("cell_family")
    column = "ocv_slope_penalty_v" if "ocv_slope_penalty_v" in osp_25.columns else osp_25.columns[-1]
    for family, quoted in {"SIB": 0.943, "LFP": 0.094, "LTO": 0.179, "NMC": 0.453}.items():
        checker.check(f"OSP 25C {family} (V)", quoted, float(osp_25.loc[family, column]), 0.001)

    # --- delivered energy and its decomposition ---------------------------
    energy = read("results/tables/table_delivered_energy_decomposition.csv")
    cold_energy = energy[energy["temperature_deg_c"] == 5.0].set_index("cell_family")
    warm_energy = energy[energy["temperature_deg_c"] == 25.0].set_index("cell_family")
    checker.check("E_del SIB at 25 degC (Wh)", 4.655, float(warm_energy.loc["SIB", "delivered_energy_wh"]), 0.001)
    checker.check("E_del SIB at 5 degC (Wh)", 3.486, float(cold_energy.loc["SIB", "delivered_energy_wh"]), 0.001)
    for family, quoted in {"SIB": 0.749, "LFP": 0.878, "LTO": 0.976, "NMC": 0.931}.items():
        checker.check(
            f"delivered-energy retention at 5 degC {family}",
            quoted,
            float(cold_energy.loc[family, "delivered_energy_wh_retention_vs_25c"]),
            0.001,
        )
    for family, quoted in {"SIB": 25.1, "LFP": 12.2, "LTO": 2.4, "NMC": 6.9}.items():
        checker.check(
            f"delivered-energy loss at 5 degC {family} (%)",
            quoted,
            100.0 * (1.0 - float(cold_energy.loc[family, "delivered_energy_wh_retention_vs_25c"])),
            0.05,
        )
    for family, quoted in {"SIB": 440, "LFP": 393, "LTO": 77, "NMC": 209}.items():
        checker.check(
            f"mean overpotential at 5 degC {family} (mV)",
            quoted,
            1000.0 * float(cold_energy.loc[family, "mean_overpotential_v"]),
            0.5,
        )
    for family, quoted in {"SIB": 293, "LFP": 262, "LTO": 51, "NMC": 140}.items():
        checker.check(
            f"implied segment resistance at 5 degC {family} (mOhm)",
            quoted,
            float(cold_energy.loc[family, "implied_resistance_mohm"]),
            0.5,
        )
    for family, quoted in {"SIB": 107, "LFP": 112, "LTO": 35, "NMC": 54}.items():
        checker.check(
            f"implied segment resistance at 25 degC {family} (mOhm)",
            quoted,
            float(warm_energy.loc[family, "implied_resistance_mohm"]),
            0.5,
        )
    for family, quoted in {"SIB": 0.861, "LFP": 0.878, "LTO": 0.969, "NMC": 0.945}.items():
        checker.check(
            f"energy efficiency at 5 degC {family}",
            quoted,
            float(cold_energy.loc[family, "energy_efficiency"]),
            0.001,
        )
    for family, quoted in {"SIB": 0.948, "LFP": 0.948, "LTO": 0.979, "NMC": 0.979}.items():
        checker.check(
            f"energy efficiency at 25 degC {family}",
            quoted,
            float(warm_energy.loc[family, "energy_efficiency"]),
            0.001,
        )

    split = read("results/tables/table_energy_shortfall_split.csv")
    cold_split = split[split["temperature_deg_c"] == 5.0].set_index("cell_family")
    checker.check("SIB total shortfall (Wh)", 1.169, float(cold_split.loc["SIB", "shortfall_total_wh"]), 0.001)
    checker.check("SIB dissipated term (Wh)", 0.312, float(cold_split.loc["SIB", "shortfall_polarisation_wh"]), 0.001)
    checker.check("SIB window term (Wh)", 0.857, float(cold_split.loc["SIB", "shortfall_window_wh"]), 0.001)
    for family, quoted in {"SIB": 27.0, "LFP": 55.0, "LTO": 41.0, "NMC": 47.0}.items():
        checker.check(
            f"dissipated share at 5 degC {family} (%)",
            quoted,
            float(cold_split.loc[family, "polarisation_share_percent"]),
            0.5,
        )
    # The additive split must close exactly.
    checker.check("shortfall split residual (Wh)", 0.0, float(cold_split["residual_wh"].abs().max()), 1e-9)

    # Three-term split: the window term resolved into truncation at the cutoff
    # and the depression of the relaxed-voltage curve over the common window.
    checker.check(
        "SIB truncation term (Wh)", 0.678, float(cold_split.loc["SIB", "shortfall_truncation_wh"]), 0.001
    )
    checker.check(
        "SIB relaxed-voltage shift term (Wh)",
        0.179,
        float(cold_split.loc["SIB", "shortfall_relaxed_shift_wh"]),
        0.001,
    )
    checker.check(
        "window-term split residual (Wh)", 0.0, float(cold_split["window_residual_wh"].abs().max()), 1e-9
    )
    for family, quoted in {"SIB": 58.0, "LFP": 32.8, "LTO": 60.9, "NMC": 40.5}.items():
        checker.check(
            f"truncation share at 5 degC {family} (%)",
            quoted,
            float(cold_split.loc[family, "truncation_share_percent"]),
            0.5,
        )
    for family, quoted in {"SIB": 15.3, "LFP": 12.0, "LTO": -1.8, "NMC": 12.0}.items():
        checker.check(
            f"relaxed-shift share at 5 degC {family} (%)",
            quoted,
            float(cold_split.loc[family, "relaxed_shift_share_percent"]),
            0.5,
        )
    # Upper bound on the dissipation-attributable share: the entire
    # relaxed-voltage shift credited to polarisation.
    for family, quoted in {"SIB": 42.0, "LFP": 67.2, "NMC": 59.5}.items():
        checker.check(
            f"dissipation upper-bound share {family} (%)",
            quoted,
            float(cold_split.loc[family, "dissipation_upper_share_percent"]),
            0.5,
        )
    checker.check(
        "SIB truncated capacity (Ah)", 0.289, float(cold_split.loc["SIB", "truncated_capacity_ah"]), 0.001
    )
    checker.check(
        "SIB truncation mean relaxed voltage (V)",
        2.344,
        float(cold_split.loc["SIB", "truncation_mean_voltage_v"]),
        0.001,
    )
    for family, quoted in {"SIB": 139.3, "LFP": 47.1, "NMC": 31.8}.items():
        checker.check(
            f"energy-weighted mean relaxed-voltage offset {family} (mV)",
            quoted,
            float(cold_split.loc[family, "relaxed_shift_mean_mv"]),
            0.5,
        )

    common = read("results/tables/table_common_window_ocv_comparison.csv")
    cold_discharge = common[
        (common["temperature_deg_c"] == 5.0) & (common["direction"] == "discharge")
    ].set_index("cell_family")
    for family, quoted in {"SIB": -131.0, "LFP": -27.0, "LTO": 1.0, "NMC": -24.0}.items():
        checker.check(
            f"common-window relaxed-voltage offset {family} (mV)",
            quoted,
            float(cold_discharge.loc[family, "median_offset_vs_25c_mv"]),
            0.6,
        )
    for family, quoted in {"SIB": 0.958, "LFP": 0.986, "LTO": 1.000, "NMC": 0.992}.items():
        checker.check(
            f"common-window energy retention {family}",
            quoted,
            float(cold_discharge.loc[family, "common_window_energy_retention_vs_25c"]),
            0.001,
        )
    cold_charge = common[
        (common["temperature_deg_c"] == 5.0) & (common["direction"] == "charge")
    ].set_index("cell_family")
    for family, quoted in {"LFP": 17.0, "LTO": 11.0, "NMC": 17.0}.items():
        checker.check(
            f"charge-direction offset {family} (mV)",
            quoted,
            float(cold_charge.loc[family, "median_offset_vs_25c_mv"]),
            0.6,
        )
    # The median offset times the common window is a rectangle approximation to
    # the relaxed-shift term. It is retired in favour of the exact integral, but
    # the direction of its error is pinned: because the offset is not uniform
    # across the window, the rectangle understates the term for every family.
    for family in FAMILIES:
        rectangle_wh = (
            -float(cold_discharge.loc[family, "median_offset_vs_25c_mv"])
            / 1000.0
            * float(cold_discharge.loc[family, "common_window_ah"])
        )
        exact_wh = float(cold_split.loc[family, "shortfall_relaxed_shift_wh"])
        checker.check(
            f"median-offset rectangle understates the relaxed-shift term {family}",
            1.0,
            1.0 if rectangle_wh <= exact_wh + 1e-9 else 0.0,
            0.0,
        )

    # graphite mean entries of Table 3 follow the arithmetic-mean-of-families
    # convention stated in its caption.
    checker.check(
        "Table 3 graphite mean E_del at 25 degC (Wh)",
        5.800,
        float(warm_energy.loc[GRAPHITE, "delivered_energy_wh"].mean()),
        0.001,
    )
    checker.check(
        "Table 3 graphite mean delivered-energy retention",
        0.905,
        float(cold_energy.loc[GRAPHITE, "delivered_energy_wh_retention_vs_25c"].mean()),
        0.001,
    )
    checker.check(
        "Table 3 graphite mean dissipated share (%)",
        51.0,
        float(cold_split.loc[GRAPHITE, "polarisation_share_percent"].mean()),
        0.5,
    )
    checker.check(
        "Table 3 graphite mean overpotential at 5 degC (mV)",
        301.0,
        1000.0 * float(cold_energy.loc[GRAPHITE, "mean_overpotential_v"].mean()),
        0.5,
    )
    checker.check(
        "Table 3 graphite mean energy efficiency at 5 degC",
        0.911,
        float(cold_energy.loc[GRAPHITE, "energy_efficiency"].mean()),
        0.001,
    )

    # --- cell-to-cell spread and pulse-current dependence ----------------
    spread = read("results/tables/table_cell_replicate_spread.csv")
    spread = spread[spread["pulse_direction"] == "discharge"].set_index("cell_family")
    for family, (low, high, percent) in {
        "SIB": (89.4, 93.0, 3.9),
        "LFP": (52.0, 84.3, 48.0),
        "LTO": (26.8, 31.4, 15.4),
        "NMC": (38.8, 41.8, 7.5),
    }.items():
        checker.check(f"replicate min R1s {family} (mOhm)", low, float(spread.loc[family, "min_r_1s_mohm"]), 0.05)
        checker.check(f"replicate max R1s {family} (mOhm)", high, float(spread.loc[family, "max_r_1s_mohm"]), 0.05)
        checker.check(
            f"replicate spread {family} (% of mean)",
            percent,
            float(spread.loc[family, "spread_percent_of_mean"]),
            0.05,
        )
    # The most conservative SIB/LFP room-temperature ratio available: the
    # analysed SIB against the lowest-resistance LFP replicate.
    checker.check(
        "SIB/LFP ratio against the lowest LFP replicate",
        1.75,
        91.1 / float(spread.loc["LFP", "min_r_1s_mohm"]),
        0.01,
    )

    rate = read("results/tables/table_pulse_rate_dependence.csv")
    rate = rate[rate["pulse_direction"] == "discharge"].set_index("cell_family")
    for family, quoted in {"SIB": 0.900, "LFP": 0.944, "LTO": 1.004, "NMC": 0.988}.items():
        checker.check(f"3C/1C R1s ratio {family}", quoted, float(rate.loc[family, "ratio_3c_over_1c"]), 0.001)
    for family, quoted in {"SIB": 10.0, "LFP": 5.6, "NMC": 1.2}.items():
        checker.check(
            f"secant compression 1C->3C {family} (%)",
            quoted,
            100.0 * (1.0 - float(rate.loc[family, "ratio_3c_over_1c"])),
            0.05,
        )
    # LFP is pulsed at 0.84C against the SIB's 0.95C; closing that gap at
    # LFP's measured rate of fall moves its descriptor by ~0.3 per cent.
    checker.check(
        "LFP fractional-rate correction (%)",
        0.3,
        (0.95 - 0.84) * float(rate.loc["LFP", "secant_fall_percent_per_c"]),
        0.05,
    )

    # --- robustness of the ordering to cell selection ---------------------
    robustness = read("results/tables/table_ordering_robustness.csv").set_index("comparison")
    for label, ratio, threshold in [
        ("R_1s at 5 degC, SIB / LFP", 1.517, 20.5),
        ("R_1s at 5 degC, SIB / NMC", 2.894, 48.6),
        ("R_1s at 25 degC, SIB / LFP", 1.095, 4.5),
        ("R_1s growth 25->5 degC, SIB / LFP", 1.385, 16.1),
        ("R_1s growth 25->5 degC, SIB / NMC", 1.361, 15.3),
        ("capacity retention at 5 degC, LFP / SIB", 1.172, 7.9),
        ("delivered-energy retention at 5 degC, LFP / SIB", 1.173, 7.9),
    ]:
        checker.check(f"ratio, {label}", ratio, float(robustness.loc[label, "ratio"]), 0.001)
        checker.check(
            f"reversal threshold, {label} (%)",
            threshold,
            float(robustness.loc[label, "reversal_threshold_percent"]),
            0.05,
        )
    for label, quoted, low, high in [
        ("R_1s at 5 degC, SIB / LFP", 96.0, 89.3, 94.7),
        ("R_1s at 25 degC, SIB / LFP", 64.9, 60.7, 63.8),
    ]:
        checker.check(
            f"ordering survival probability, {label} (%)",
            quoted,
            float(robustness.loc[label, "survival_probability_percent"]),
            0.05,
        )
        checker.check(
            f"survival under the 24 % external dispersion, {label} (%)",
            low,
            float(robustness.loc[label, "survival_probability_external_high_percent"]),
            0.05,
        )
        checker.check(
            f"survival under the 10 % external dispersion, {label} (%)",
            high,
            float(robustness.loc[label, "survival_probability_external_low_percent"]),
            0.05,
        )
    # The external bracket quoted in the text: 10 per cent from the 100-cell
    # population, 24 per cent (38/159) from the 20-cell one.
    checker.check(
        "external SIB dispersion, lower bound (%)",
        10.0,
        float(robustness.loc["R_1s at 5 degC, SIB / LFP", "cv_a_external_low_percent"]),
        0.05,
    )
    checker.check(
        "external SIB dispersion, upper bound (%), text says 24",
        24.0,
        float(robustness.loc["R_1s at 5 degC, SIB / LFP", "cv_a_external_high_percent"]),
        0.15,
    )
    checker.check(
        "ordering survival probability, SIB/NMC at 5 degC (%), text says >99.9",
        100.0,
        float(robustness.loc["R_1s at 5 degC, SIB / NMC", "survival_probability_percent"]),
        0.1,
    )
    # Dispersion feeding the scenario, quoted in the SI.
    for family, quoted in {"SIB": 2.2, "LFP": 24.1, "LTO": 8.8, "NMC": 4.2}.items():
        checker.check(
            f"replicate CV {family} (%)",
            quoted,
            float(spread.loc[family, "cv_percent"]),
            0.05,
        )

    # --- absolute polarisation loss, added when the manuscript was corrected --
    # Section 3.3 previously called the 0.312 Wh polarisation *increment* the
    # energy dissipated at 5 degC. It is the rise against 25 degC; these pins
    # hold the two absolute values the corrected text now quotes.
    decomposition = read("results/tables/table_delivered_energy_decomposition.csv")
    sib_energy = decomposition[decomposition["cell_family"] == "SIB"].set_index(
        "temperature_deg_c"
    )
    checker.check(
        "SIB polarisation loss at 5 degC (Wh)",
        0.565,
        float(sib_energy.loc[5.0, "polarisation_loss_wh"]),
        0.001,
    )
    checker.check(
        "SIB polarisation loss at 25 degC (Wh)",
        0.253,
        float(sib_energy.loc[25.0, "polarisation_loss_wh"]),
        0.001,
    )
    checker.check(
        "SIB polarisation loss at 5 degC (% of E_OCV)",
        14.0,
        float(sib_energy.loc[5.0, "polarisation_loss_percent_of_ocv"]),
        0.1,
    )

    # --- EIS state-trim subset, added with Table S3's central-state row -------
    # The ten spectra of a family are a voltage sweep on one cell, so the
    # comparison is also reported with the extreme state points dropped.
    spectra = read("data_processed/eis/eis_spectral_descriptors.csv")
    trimmed = {}
    for family in FAMILIES:
        values = (
            spectra[spectra["cell_family"] == family]
            .sort_values("voltage_v")["low_freq_zre_ohm"]
            .to_numpy()
            * 1000.0
        )
        trimmed[family] = float(np.median(values[2:-2]))
    for family, quoted in {"SIB": 87.65, "LFP": 71.45, "LTO": 18.23, "NMC": 30.54}.items():
        checker.check(
            f"central-state EIS median {family} (mOhm)", quoted, trimmed[family], 0.01
        )
    checker.check(
        "central-state SIB / graphite-anode ratio",
        1.719,
        trimmed["SIB"] / float(np.mean([trimmed[f] for f in GRAPHITE])),
        0.001,
    )

    # --- graphite-anode means introduced when the reference column changed ----
    cap5 = capacity[capacity["temperature_deg_c"] == 5.0].set_index("cell_family")
    checker.check(
        "OCV-window energy retention 5C graphite mean",
        0.956,
        float(cap5.loc[GRAPHITE, "energy_retention_vs_25c"].mean()),
        0.001,
    )
    span = read("results/tables/table_ocv_slope_summary.csv")
    span = span[span["direction"] == "discharge"]
    span25 = span[span["temperature_deg_c"] == 25.0].set_index("cell_family")["ocv_slope_penalty_v"]
    span5 = span[span["temperature_deg_c"] == 5.0].set_index("cell_family")["ocv_slope_penalty_v"]
    checker.check("central OCV span 25C graphite mean (V)", 0.274,
                  float(span25.reindex(GRAPHITE).mean()), 0.001)
    checker.check("OCV-span ratio 5/25 graphite mean", 1.047,
                  float((span5 / span25).reindex(GRAPHITE).mean()), 0.001)
    entropy_all = read("results/tables/table_entropy_summary.csv")
    entropy_all = entropy_all[entropy_all["direction"] == "discharge"].set_index("cell_family")
    checker.check("mean |dU/dT| graphite mean (mV/K)", 0.165,
                  float(entropy_all.loc[GRAPHITE, "mean_abs_dudt_mv_per_k"].mean()), 0.001)
    # The entropic bound on the relaxed-voltage offset is taken at the largest
    # SOC-resolved coefficient, not the mean: a mean cannot bound a local shift.
    checker.check("max |dU/dT| SIB discharge (mV/K)", 0.452,
                  float(entropy_all.loc["SIB", "max_abs_dudt_mv_per_k"]), 0.001)
    checker.check("entropic bound over 20 K SIB (mV)", 9.0,
                  20.0 * float(entropy_all.loc["SIB", "max_abs_dudt_mv_per_k"]), 0.1)
    thermal_all = read("results/tables/table_thermal_response_summary.csv")
    thermal_all = thermal_all[thermal_all["direction"] == "discharge"].set_index("cell_family")
    checker.check("BOL discharge dT graphite mean (K)", 1.84,
                  float(thermal_all.loc[GRAPHITE, "mean_temperature_rise_max_k"].mean()), 0.01)

    # --- condition of the replicate HPPC files (Section 3.5) -----------
    # The files carry no temperature label; Section 3.5 recovers it from the
    # resistance for the two families whose R_1s is sharp enough to do so.
    assignment = read(
        "results/tables/table_replicate_temperature_assignment.csv"
    ).set_index("cell_family")
    checker.check("SIB replicate gap to the 25 C series value (%)", 0.0,
                  float(assignment.loc["SIB", "gap_to_25c_percent"]), 0.05)
    checker.check("SIB replicate gap to the nearest other setpoint (%)", 43.6,
                  float(assignment.loc["SIB", "gap_to_closest_alternative_percent"]), 0.05)
    checker.check("NMC replicate gap to the 25 C series value (%)", 2.4,
                  float(assignment.loc["NMC", "gap_to_25c_percent"]), 0.05)
    checker.check("NMC replicate gap to the nearest other setpoint (%)", 17.8,
                  float(assignment.loc["NMC", "gap_to_closest_alternative_percent"]), 0.05)

    # --- quadrature sensitivity of the three-term split (SI S3.1) ------
    quadrature = read(
        "results/tables/table_ocv_quadrature_sensitivity.csv"
    ).set_index(["cell_family", "quadrature"])
    for family, rule, truncation, shift, polarisation, upper in [
        ("SIB", "right", 58.0, 15.3, 26.7, 42.0),
        ("SIB", "mid", 58.5, 15.2, 26.3, 41.5),
        ("SIB", "left", 59.0, 15.1, 26.0, 41.0),
        ("LFP", "right", 32.8, 12.0, 55.2, 67.2),
        ("LFP", "mid", 33.4, 11.7, 54.9, 66.6),
        ("LFP", "left", 33.9, 11.4, 54.7, 66.1),
        ("NMC", "right", 40.5, 12.0, 47.4, 59.5),
        ("NMC", "mid", 41.3, 11.9, 46.8, 58.7),
        ("NMC", "left", 42.1, 11.8, 46.1, 57.9),
        ("LTO", "right", 60.9, -1.8, 40.9, 39.1),
        ("LTO", "mid", 61.5, -1.7, 40.2, 38.5),
        ("LTO", "left", 62.2, -1.7, 39.5, 37.8),
    ]:
        row = quadrature.loc[(family, rule)]
        checker.check(f"Table S7 truncation share, {family} {rule} (%)", truncation,
                      float(row["truncation_share_percent"]), 0.05)
        checker.check(f"Table S7 shift share, {family} {rule} (%)", shift,
                      float(row["relaxed_shift_share_percent"]), 0.05)
        checker.check(f"Table S7 polarisation share, {family} {rule} (%)", polarisation,
                      float(row["polarisation_share_percent"]), 0.05)
        checker.check(f"Table S7 dissipation upper bound, {family} {rule} (%)", upper,
                      float(row["dissipation_upper_share_percent"]), 0.05)
    # The two absolute quantities the rule does move at reported precision.
    checker.check("SIB mean overpotential 5C under the midpoint rule (mV)", 450.0,
                  float(quadrature.loc[("SIB", "mid"), "mean_overpotential_5c_mv"]), 0.5)
    checker.check("SIB energy efficiency 5C under the midpoint rule", 0.858,
                  float(quadrature.loc[("SIB", "mid"), "energy_efficiency_5c"]), 0.001)
    # "no share moves by more than 1.6 percentage points" (Section 2.4, SI S3.1)
    widest = 0.0
    for column in [
        "truncation_share_percent",
        "relaxed_shift_share_percent",
        "polarisation_share_percent",
    ]:
        by_family = quadrature.groupby("cell_family")[column]
        widest = max(widest, float((by_family.max() - by_family.min()).max()))
    checker.check("widest share movement across quadrature rules (pp)", 1.6, widest, 0.05)

    # --- discharge cutoff and the segment budget (Section 2.4, SI S3.2) ---
    budget = read("results/tables/table_cutoff_step_budget.csv").set_index(
        ["cell_family", "temperature_deg_c"]
    )
    # Cutoffs quoted to 2 dp as the median over the five temperatures.
    for family, cutoff in [("SIB", 1.49), ("LFP", 1.98), ("NMC", 2.49), ("LTO", 1.48)]:
        median = float(
            budget.xs(family, level="cell_family")["cutoff_voltage_v"].median()
        )
        checker.check(f"discharge cutoff, {family} (V)", cutoff, median, 0.005)
    # "stable to within 13 mV across the five temperatures in every family"
    spread_mv = 1000.0 * float(
        budget.groupby("cell_family")["cutoff_voltage_v"].agg(lambda s: s.max() - s.min()).max()
    )
    checker.check("widest cutoff spread across temperature (mV)", 13.0, spread_mv, 0.5)
    for family, cold_n, warm_n, cold_rec, warm_rec in [
        ("SIB", 18, 18, 13.6, 21.1),
        ("LFP", 8, 6, 7.4, 8.0),
        ("NMC", 6, 5, 10.0, 5.5),
        ("LTO", 6, 6, 10.6, 6.5),
    ]:
        cold = budget.loc[(family, 5.0)]
        warm = budget.loc[(family, 25.0)]
        checker.check(f"Table S8 cut-short segments, {family} 5C", cold_n,
                      float(cold["cutoff_limited_segments"]), 0.5)
        checker.check(f"Table S8 cut-short segments, {family} 25C", warm_n,
                      float(warm["cutoff_limited_segments"]), 0.5)
        checker.check(f"Table S8 unbounded-schedule recovery, {family} 5C (mAh)", cold_rec,
                      1000.0 * float(cold["recovery_geometric_ah"]), 0.05)
        checker.check(f"Table S8 unbounded-schedule recovery, {family} 25C (mAh)", warm_rec,
                      1000.0 * float(warm["recovery_geometric_ah"]), 0.05)
    for family, measured, extrapolated in [
        ("SIB", 289.4, 297.0),
        ("LFP", 78.4, 79.1),
        ("NMC", 56.0, 51.6),
        ("LTO", 24.3, 20.2),
    ]:
        cold = budget.loc[(family, 5.0)]
        warm = budget.loc[(family, 25.0)]
        gap = 1000.0 * float(
            warm["accessible_capacity_ah"] - cold["accessible_capacity_ah"]
        )
        widened = gap - 1000.0 * float(
            cold["recovery_geometric_ah"] - warm["recovery_geometric_ah"]
        )
        checker.check(f"Table S8 measured 5-25C capacity gap, {family} (mAh)", measured,
                      gap, 0.05)
        checker.check(f"Table S8 extrapolated 5-25C capacity gap, {family} (mAh)",
                      extrapolated, widened, 0.05)
    # "widens it by 2.6 %", and "3.1 %" at the largest observed ratio (Section 2.4)
    sib_cold, sib_warm = budget.loc[("SIB", 5.0)], budget.loc[("SIB", 25.0)]
    sib_gap = 1000.0 * float(
        sib_warm["accessible_capacity_ah"] - sib_cold["accessible_capacity_ah"]
    )
    for label, quoted, column in [
        ("geometric", 2.6, "recovery_geometric_ah"),
        ("largest observed ratio", 3.1, "recovery_maximum_ah"),
    ]:
        widened = sib_gap - 1000.0 * float(sib_cold[column] - sib_warm[column])
        checker.check(f"SIB gap widening, {label} (%)", quoted,
                      100.0 * (widened - sib_gap) / sib_gap, 0.05)
    # End-of-discharge rested voltages quoted in Section 3.3.
    checker.check("SIB final rested voltage at 5C (V)", 2.238,
                  float(sib_cold["final_rested_voltage_v"]), 0.001)
    checker.check("SIB final rested voltage at 25C (V)", 1.996,
                  float(sib_warm["final_rested_voltage_v"]), 0.001)

    # --- quantities added in the final revision -----------------------------
    # rest-transient extrapolation (SM S3.3)
    rex = read("results/tables/table_rest_extrapolation.csv")

    def rex_get(fam, T, col):
        row = one_row(rex, f"rest extrapolation {fam} {T} C", family=fam, temperature_deg_c=T)
        return float(row[col])
    for fam, b5, s5, b25, s25 in (("SIB", 1.36, 13.95, 1.31, 6.43),
                                  ("LFP", 2.04, 21.62, 1.91, 12.49),
                                  ("NMC", 0.76, 13.87, 0.33, 4.44),
                                  ("LTO", -0.09, 4.98, -0.02, 2.79)):
        checker.check(f"rest under-read biexp 5C {fam} (mV)", b5,
                      rex_get(fam, 5, "under_read_biexp_mv"), 0.02)
        checker.check(f"rest under-read sqrt 5C {fam} (mV)", s5,
                      rex_get(fam, 5, "under_read_sqrt_mv"), 0.02)
        checker.check(f"rest under-read biexp 25C {fam} (mV)", b25,
                      rex_get(fam, 25, "under_read_biexp_mv"), 0.02)
        checker.check(f"rest under-read sqrt 25C {fam} (mV)", s25,
                      rex_get(fam, 25, "under_read_sqrt_mv"), 0.02)
    checker.check("SIB differential under-read, sqrt tail (mV)", 7.5,
                  rex_get("SIB", 5, "under_read_sqrt_mv") - rex_get("SIB", 25, "under_read_sqrt_mv"), 0.05)
    checker.check("SIB differential under-read, biexp (mV)", 0.04,
                  rex_get("SIB", 5, "under_read_biexp_mv") - rex_get("SIB", 25, "under_read_biexp_mv"), 0.02)
    checker.check("unexplained common-window offset (mV)", 115.0,
                  131.0 - (rex_get("SIB", 5, "under_read_sqrt_mv")
                           - rex_get("SIB", 25, "under_read_sqrt_mv")) - 9.0, 0.6)

    # stopping-line perturbation sweep (SM S3.4)
    cn = read("results/tables/table_cutoff_normalisation.csv")

    def sweep(fam: str, delta: int, cold: float) -> float:
        row = one_row(cn, f"stopping-line sweep {fam} +{delta} mV {cold} C",
                      family=fam, delta_mv=delta, cold_deg_c=cold)
        return float(row.share_window)

    for fam, d, q in (("SIB", 0, 58.0), ("SIB", 100, 52.9), ("SIB", 200, 61.6),
                      ("LFP", 0, 34.2), ("NMC", 0, 37.0), ("LTO", 200, 85.3)):
        checker.check(f"stopping-line sweep share {fam} +{d} mV, 5 C (%)", q,
                      sweep(fam, d, 5.0), 0.15)

    # The 5-to-15 degC comparison the manuscripts quote. This was asserted for a
    # year against a 5 degC-only table, and the reference range it stated (10-24
    # points) understated NMC, which moves 26.1. Both ends are now checked.
    deltas = sorted(cn.delta_mv.unique())
    moves = {fam: [sweep(fam, d, 15.0) - sweep(fam, d, 5.0) for d in deltas]
             for fam in FAMILIES}
    checker.check("SIB max |share change| 5->15 C (pp)", 6.0,
                  max(abs(v) for v in moves["SIB"]), 0.1)
    graphite_moves = [abs(v) for fam in GRAPHITE for v in moves[fam]]
    checker.check("LFP/NMC smallest share fall 5->15 C (pp)", 10.8,
                  min(graphite_moves), 0.1)
    checker.check("LFP/NMC largest share fall 5->15 C (pp)", 26.1,
                  max(graphite_moves), 0.1)
    # Regression pin on the row that broke the old prose claim: NMC at the
    # largest tested margin is where the graphite range runs past "10-24".
    checker.check("NMC share change 5->15 C at +200 mV (pp)", -26.1,
                  sweep("NMC", 200, 15.0) - sweep("NMC", 200, 5.0), 0.05)
    # And the SIB bound the SI now quotes, which is 6.047 and so needs a 6.1 cap.
    checker.check("SIB largest |share change| 5->15 C (pp)", 6.047,
                  max(abs(v) for v in moves["SIB"]), 0.01)
    # the sign matters to the claim: the graphite references fall, they do not drift
    checker.check("LFP/NMC share changes that are falls", float(len(graphite_moves)),
                  float(sum(1 for fam in GRAPHITE for v in moves[fam] if v < 0)), 0.0)

    # --- preparation controls added in the sparring-review revision --------
    # Section 3.2 and SM S3.3: the charge-input ledger, the rested start state,
    # the CV-taper bound and the charge-axis alignment fit.
    ledger = read("results/tables/table_charge_input_ledger.csv")
    taper = read("results/tables/table_cv_taper_extrapolation.csv")
    align = read("results/tables/table_charge_axis_alignment.csv")

    def pick(frame, fam, T):
        return one_row(frame, f"preparation row {fam} {T} C", cell_family=fam,
                       temperature_deg_c=T)

    sib5, sib25, sib45 = pick(ledger, "SIB", 5.0), pick(ledger, "SIB", 25.0), pick(ledger, "SIB", 45.0)
    checker.check("SIB total charge input 5C (Ah)", 2.480, float(sib5.charge_input_total_ah), 0.002)
    checker.check("SIB total charge input 25C (Ah)", 2.489, float(sib25.charge_input_total_ah), 0.002)
    checker.check("SIB charge-input deficit 5C (Ah)", 0.009,
                  float(sib5.charge_input_deficit_vs_25c_ah), 0.002)
    checker.check("SIB constant-current top-ups 5C (Ah)", 1.589, float(sib5.charge_input_topup_ah), 0.002)
    checker.check("SIB constant-current top-ups 25C (Ah)", 1.519, float(sib25.charge_input_topup_ah), 0.002)
    checker.check("SIB rested start voltage 5C (V)", 4.048, float(sib5.rested_start_voltage_v), 0.002)
    checker.check("SIB rested start voltage 25C (V)", 3.997, float(sib25.rested_start_voltage_v), 0.002)
    checker.check("SIB rested start offset 5C (mV)", 51.0,
                  float(sib5.rested_start_offset_vs_25c_mv), 0.6)

    tsib5, tsib25, tsib45 = pick(taper, "SIB", 5.0), pick(taper, "SIB", 25.0), pick(taper, "SIB", 45.0)
    checker.check("SIB terminating-step charge 5C (Ah)", 0.881, float(tsib5.charge_ah), 0.002)
    checker.check("SIB terminating-step charge 25C (Ah)", 1.029, float(tsib25.charge_ah), 0.002)
    checker.check("SIB terminating-step final current 5C (mA)", 162.0, float(tsib5.final_current_ma), 0.6)
    checker.check("SIB terminating-step final current 25C (mA)", 73.0, float(tsib25.final_current_ma), 0.6)
    checker.check("SIB CV-taper extrapolated extra charge 5C (Ah)", 0.081,
                  float(tsib5.extrapolated_extra_charge_ah), 0.002)
    checker.check("SIB taper-corrected charge 5C (Ah)", 0.963,
                  float(tsib5.charge_ah_taper_corrected), 0.002)
    checker.check("SIB 45C terminating-step deficit (Ah)", 0.854,
                  float(tsib25.charge_ah - tsib45.charge_ah), 0.002)
    checker.check("SIB 45C-to-5C deficit ratio", 5.8,
                  float((tsib25.charge_ah - tsib45.charge_ah) / (tsib25.charge_ah - tsib5.charge_ah)),
                  0.05)

    asib5, asib45 = pick(align, "SIB", 5.0), pick(align, "SIB", 45.0)
    checker.check("SIB fitted charge-axis shift 5C (Ah)", 0.095, float(asib5.fitted_shift_ah), 0.002)
    checker.check("SIB fitted charge-axis shift 45C (Ah)", 0.000, float(asib45.fitted_shift_ah), 0.002)
    checker.check("SIB alignment RMS, unshifted 5C (mV)", 115.6, float(asib5.rms_unshifted_mv), 0.3)
    checker.check("SIB alignment RMS, at fitted shift 5C (mV)", 42.9,
                  float(asib5.rms_at_fitted_shift_mv), 0.3)
    checker.check("SIB alignment RMS reduction 5C (%)", 63.0, float(asib5.rms_reduction_percent), 0.6)
    # The alignment test's calibration: LFP needs a translation its own charge
    # ledger excludes and still removes 72 % of the mismatch (SM S2.1).
    alfp5 = pick(align, "LFP", 5.0)
    lfp5 = pick(ledger, "LFP", 5.0)
    checker.check("LFP fitted charge-axis shift 5C (Ah)", 0.300,
                  float(alfp5.fitted_shift_ah), 0.002)
    checker.check("LFP charge-input deficit 5C (Ah)", 0.073,
                  float(lfp5.charge_input_deficit_vs_25c_ah), 0.002)

    # The two documents quote the size of their own verification suite. Those
    # counts drifted (52 -> 62) once already, because nothing could contradict
    # them; derive both from the artefacts themselves.
    audit = read("results/tables/table_final_reproducibility_check.csv")
    checker.check("reproducibility audit size quoted in the SI", 62, float(len(audit)), 0.5)
    test_files = sorted((PROJECT_ROOT / "tests").glob("test_*.py"))
    assert test_files, "no test modules found under tests/"
    n_tests = sum(f.read_text(encoding="utf-8").count("\ndef test_") for f in test_files)
    checker.check("regression suite size quoted in the JES manuscript", 29, float(n_tests), 0.5)
    checker.check("SIB central OCV slope (V/Ah)", 1.04, float(asib5.central_ocv_slope_v_per_ah), 0.01)
    for third, quoted in (("shift_first_third_ah", 0.060), ("shift_middle_third_ah", 0.091),
                          ("shift_last_third_ah", 0.136)):
        checker.check(f"SIB alignment {third} 5C (Ah)", quoted, float(asib5[third]), 0.002)
    for fam, quoted in (("LFP", 72.0), ("NMC", 72.0)):
        checker.check(f"{fam} alignment RMS reduction 5C (%)", quoted,
                      float(pick(align, fam, 5.0).rms_reduction_percent), 0.6)

    # Section 3.2: self-heating inside the discharge.
    thermal = read("results/tables/table_discharge_thermal_excursion.csv")
    tsib = pick(thermal, "SIB", 5.0)
    checker.check("SIB discharge median cell temperature 5C (degC)", 5.75, float(tsib.median_c), 0.01)
    checker.check("SIB discharge cell-temperature IQR 5C (K)", 0.22, float(tsib.iqr_k), 0.01)
    checker.check("largest in-discharge excursion (K)", 0.77,
                  float(thermal.peak_rise_over_rest_k.max()), 0.01)
    checker.check("largest implied resistance bias (%)", 4.2,
                  float(abs(thermal.implied_resistance_error_percent).max()), 0.1)

    # Section 3.2 and SM S3.2: the tail decay is fitted, not assumed.
    budget = read("results/tables/table_cutoff_step_budget.csv")
    bsib5, bsib25 = pick(budget, "SIB", 5.0), pick(budget, "SIB", 25.0)
    checker.check("SIB tail ratio 5C", 0.962, float(bsib5.tail_ratio_from_log_fit), 0.002)
    checker.check("SIB tail ratio 25C", 0.957, float(bsib25.tail_ratio_from_log_fit), 0.002)
    checker.check("SIB tail fit R2 5C", 0.9997, float(bsib5.tail_log_fit_r_squared), 0.0005)
    checker.check("SIB tail fit R2 25C", 0.9974, float(bsib25.tail_log_fit_r_squared), 0.0005)

    # Section 3.6: charge-direction growth factors, Table 5.
    charge = pulse[(pulse["pulse_direction"] == "charge") & (pulse["abs_c_rate"] == 1.0)]
    cgrid = charge.pivot_table(index="cell_family", columns="temperature_deg_c",
                               values="median_r_1s_mohm")
    cgrowth = cgrid[5.0] / cgrid[25.0]
    for family, quoted in {"SIB": 2.58, "LFP": 2.38, "NMC": 2.35, "LTO": 1.33}.items():
        checker.check(f"R_1s charge growth 25->5C {family}", quoted, float(cgrowth[family]), 0.01)
    checker.check("R_1s charge growth, graphite mean", 2.36, float(cgrowth[GRAPHITE].mean()), 0.01)

    # The anchor is a voltage condition and not a common state of charge, which
    # is why the ledger bounds charge delivered and not charge stored: the
    # anchor sequence removes 0.289 Ah less in the cold, the same size as the
    # window gap it is being used to control for.
    checker.check("SIB anchor discharge 5C (Ah)", 1.224,
                  abs(float(sib5.anchor_discharge_ah)), 0.002)
    checker.check("SIB anchor discharge 25C (Ah)", 1.513,
                  abs(float(sib25.anchor_discharge_ah)), 0.002)
    checker.check("SIB anchor-state inequality (Ah)", 0.289,
                  abs(float(sib25.anchor_discharge_ah)) - abs(float(sib5.anchor_discharge_ah)),
                  0.002)

    # Section 3.1: the sensitivity envelope on the shorter-window share. These
    # are deterministic excursions summed at their worst, not a statistical band.
    checker.check("shorter-window worst-case excursion (pp)", 3.1, 1.6 + 1.5, 0.01)

    failures = checker.report()
    if failures:
        print(f"\n{failures} manuscript number(s) no longer match their source table.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
