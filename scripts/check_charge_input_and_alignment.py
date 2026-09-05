#!/usr/bin/env python3
"""Was the cold cell simply charged less, and would that explain the shortfall?

The three-term split reads its largest term as charge the cold cell never
reached before cutoff. That reading survives only if the two discharges start
from comparably filled cells, and the terminating charge step of the SIB does
not look comparable: at 5 degC it is cut by a 3600 s duration cap and stops at
162 mA, where the 25 degC run holds for 23,086 s and tapers to 73 mA, passing
1.029 Ah against 0.882 Ah. Read naively, that 0.147 Ah gap is half the 0.289 Ah
discharge-window gap the decomposition attributes to cutoff, and on the SIB's
1.04 V/Ah central slope a 0.13 Ah misalignment would reproduce the whole
common-window rested-voltage offset. The objection has to be met with
measurement.

WHAT THIS SCRIPT CAN AND CANNOT SETTLE. It measures charge put into the cell,
and the displacement of the rested-voltage curve. Neither is the same as
charge stored at the start of the discharge, and this archive does not measure
that. Two gaps stand between them, both recorded here:

  * The ledger anchor is where the cell last sat at its lower voltage cutoff,
    which is not a common state of charge. A cold cell reaches the same
    terminal voltage holding more charge, and `anchor_discharge_ah` shows how
    much less the anchor sequence removed in the cold: 0.289 Ah less for the
    SIB. That inequality runs toward the cold cell starting fuller, so it does
    not rescue an undercharge reading, but it does mean equal input does not
    imply equal stored charge.
  * Coulombic efficiency is neither unity nor temperature-independent here.
    The 25 degC terminating step is still passing 73 mA after 6.4 h at the
    voltage limit, twice the criterion the three references reach, so part of
    its input is a leakage current that was never stored. That too runs toward
    the cold cell storing more of an equal input.

So the four measurements below disfavour preparation as the dominant
explanation of the window gap. They do not establish that the two discharges
began from equal stored charge.

1. CHARGE-INPUT LEDGER. The checkup drives every cell to its lower voltage
   cutoff (the last StepID 52 block) before walking it back up through eleven
   HPPC blocks, each closed by a constant-current top-up (86), and finally the
   terminating constant-current/constant-voltage charge (89). A signed
   coulomb count from that cutoff to the first sample of the discharge (96) --
   every charge and discharge pulse in between carried with its own sign --
   gives the total charge put in, which is the descriptor the single-step
   comparison is missing.

2. RESTED START STATE. The discharge is preceded by a 3600 s rest (93). Its
   final voltage is the rested voltage the discharge starts from. This paper
   also shows that relation to be temperature- and path-dependent, so the
   value is reported as a descriptor and is not read as a state of charge.

3. CV-TAPER EXTRAPOLATION. Where step 89 is cut by the cap the current is still
   decaying. Fitting the decay over the constant-voltage phase and integrating
   down to the current the reference reached bounds the charge the cap cost.

4. CHARGE-AXIS ALIGNMENT. A cold run beginning Delta_q short would show a
   rested curve translated along the charge axis. Fitting that translation
   says how much rigid misalignment the observed offset would need. It cannot
   bound the true Delta_q, because a preparation offset and a genuine
   temperature-dependent voltage shift can coexist:
   V_5(q) = V_25(q + Delta_q) + dV_T(q), with dV_T unknown and demonstrably
   non-zero. What the fit does show is whether a rigid translation alone
   describes the offset, which is why it is also run over thirds of the
   window: one Delta_q fits everywhere under preparation, and does not here.

All four run at 15, 35 and 45 degC as well. Those are the controls: the same
3600 s cap binds there, so if the cap were what displaces the rested curve, the
displacement should not depend on which side of the reference the temperature
falls.
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
SEGMENTS = PROJECT_ROOT / "results/tables/table_delivered_energy_segments.csv"

LEDGER_OUTPUT = PROJECT_ROOT / "results/tables/table_charge_input_ledger.csv"
TAPER_OUTPUT = PROJECT_ROOT / "results/tables/table_cv_taper_extrapolation.csv"
ALIGNMENT_OUTPUT = PROJECT_ROOT / "results/tables/table_charge_axis_alignment.csv"

COLUMNS = ["Testtime[s]", "StepID", "Voltage[V]", "Current[A]"]

CELLS = {"SIB": "TC23SIB09", "LFP": "TC23LFP09", "NMC": "TC23NMC09", "LTO": "TC23LTO09"}
TEMPERATURES = {5: "05deg", 15: "15deg", 25: "25deg", 35: "35deg", 45: "45deg"}
FAMILY_ORDER = ["SIB", "LFP", "NMC", "LTO"]
REFERENCE_TEMPERATURE_C = 25.0

DEEP_DISCHARGE_STEP = 52
HPPC_STEPS = range(61, 85)
CC_TOPUP_STEP = 86
FINAL_CHARGE_STEP = 89
FINAL_REST_STEP = 93
DISCHARGE_STEP = 96

# Charge-axis translations scanned by the alignment fit, in Ah. The evaluation
# window is shortened by the largest of these at every candidate shift, so the
# residual is always compared over an identical window and the fit cannot buy a
# lower sum of squares by evaluating over less charge.
SHIFT_GRID = np.arange(-0.10, 0.4001, 0.001)
MAX_SHIFT_AH = float(SHIFT_GRID.max())


def load_run(family: str, temperature_c: int) -> pd.DataFrame:
    cell = CELLS[family]
    member = f"raw_CU_diffT/{cell}/{cell}_{TEMPERATURES[temperature_c]}.parquet"
    frame = read_parquet_member(ARCHIVE, member, columns=COLUMNS)
    frame.columns = [name.split("[")[0] for name in frame.columns]
    return frame.sort_values("Testtime").reset_index(drop=True)


def block_index(frame: pd.DataFrame) -> pd.Series:
    """Contiguous runs of one StepID. StepIDs recur -- the HPPC sequence eleven
    times -- so grouping on the ID alone would merge separate visits."""
    return frame["StepID"].ne(frame["StepID"].shift()).cumsum()


def coulomb_count_ah(frame: pd.DataFrame) -> float:
    """Signed charge by trapezoidal integration. Valid only within one block:
    across blocks the sampling gaps would be integrated as if current flowed."""
    time_s = frame["Testtime"].to_numpy(dtype=float)
    current = frame["Current"].to_numpy(dtype=float)
    finite = np.isfinite(time_s) & np.isfinite(current)
    if finite.sum() < 2:
        return 0.0
    return float(np.trapezoid(current[finite], time_s[finite]) / 3600.0)


def run_blocks(family: str, temperature_c: int):
    """The run, its block labels, and the block table up to the discharge."""
    frame = load_run(family, temperature_c)
    blocks = block_index(frame)
    table = frame.groupby(blocks).agg(step=("StepID", "first"))
    discharge_block = table.index[table["step"] == DISCHARGE_STEP][0]
    return frame, blocks, table, discharge_block


def charge_input_ledger() -> pd.DataFrame:
    """Net charge from the last lower-cutoff visit to the start of the discharge."""
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        for temperature_c in TEMPERATURES:
            frame, blocks, table, discharge_block = run_blocks(family, temperature_c)
            before = table.loc[: discharge_block - 1]
            anchor = before.index[before["step"] == DEEP_DISCHARGE_STEP][-1]

            # Block by block, so no integration crosses a sampling gap.
            totals = {"hppc": 0.0, "topup": 0.0, "final": 0.0, "total": 0.0}
            topup_blocks = 0
            for block_id in range(anchor + 1, discharge_block):
                block = frame[blocks == block_id]
                charge = coulomb_count_ah(block)
                totals["total"] += charge
                step = int(table.loc[block_id, "step"])
                if step in HPPC_STEPS:
                    totals["hppc"] += charge
                elif step == CC_TOPUP_STEP:
                    totals["topup"] += charge
                    topup_blocks += 1
                elif step == FINAL_CHARGE_STEP:
                    totals["final"] += charge

            # The anchor is where the cell last sat at its lower voltage
            # cutoff, and that is not a common state of charge: a cold cell
            # reaches the same terminal voltage holding more charge. Recording
            # what the anchor sequence removed makes the size of that
            # inequality visible, and it is why the ledger below bounds charge
            # put in and not charge stored.
            anchor_step_ids = [
                block_id for block_id in before.index
                if int(before.loc[block_id, "step"]) == DEEP_DISCHARGE_STEP
            ]
            anchor_removed = sum(
                coulomb_count_ah(frame[blocks == block_id]) for block_id in anchor_step_ids
            )

            anchor_block = frame[blocks == anchor]
            rest_block = frame[blocks == before.index[before["step"] == FINAL_REST_STEP][-1]]
            charge_block = frame[blocks == before.index[before["step"] == FINAL_CHARGE_STEP][-1]]

            rows.append(
                {
                    "cell_family": family,
                    "temperature_deg_c": float(temperature_c),
                    "anchor_final_voltage_v": float(anchor_block["Voltage"].iloc[-1]),
                    "anchor_discharge_ah": anchor_removed,
                    "n_topup_blocks": topup_blocks,
                    "charge_input_hppc_ah": totals["hppc"],
                    "charge_input_topup_ah": totals["topup"],
                    "charge_input_final_step_ah": totals["final"],
                    "charge_input_total_ah": totals["total"],
                    "charge_end_voltage_v": float(charge_block["Voltage"].iloc[-1]),
                    "charge_end_current_ma": 1000.0 * float(charge_block["Current"].iloc[-1]),
                    "rested_start_voltage_v": float(rest_block["Voltage"].iloc[-1]),
                    "relaxation_drop_mv": 1000.0
                    * float(charge_block["Voltage"].iloc[-1] - rest_block["Voltage"].iloc[-1]),
                }
            )

    ledger = pd.DataFrame.from_records(rows)
    for family, group in ledger.groupby("cell_family"):
        reference = group[group["temperature_deg_c"] == REFERENCE_TEMPERATURE_C].iloc[0]
        ledger.loc[group.index, "charge_input_deficit_vs_25c_ah"] = (
            float(reference["charge_input_total_ah"]) - group["charge_input_total_ah"]
        )
        ledger.loc[group.index, "final_step_deficit_vs_25c_ah"] = (
            float(reference["charge_input_final_step_ah"]) - group["charge_input_final_step_ah"]
        )
        ledger.loc[group.index, "anchor_discharge_deficit_vs_25c_ah"] = float(
            reference["anchor_discharge_ah"]
        ) - group["anchor_discharge_ah"]
        ledger.loc[group.index, "rested_start_offset_vs_25c_mv"] = 1000.0 * (
            group["rested_start_voltage_v"] - float(reference["rested_start_voltage_v"])
        )
    return ledger


def cv_taper_extrapolation() -> pd.DataFrame:
    """Charge a truncated constant-voltage hold cost, bounded by extrapolation.

    Inside the constant-voltage phase the current decays smoothly. Fitting
    log I against log t over the second half of that phase -- the region whose
    decay rate governs the tail -- and integrating the fit from the observed
    endpoint down to the current the 25 degC run reached gives the charge that
    would have entered had the step run on to the same endpoint.
    """
    records: dict[tuple[str, int], dict] = {}
    reference_current: dict[str, float] = {}
    for family in FAMILY_ORDER:
        for temperature_c in TEMPERATURES:
            frame, blocks, table, discharge_block = run_blocks(family, temperature_c)
            before = table.loc[: discharge_block - 1]
            block = frame[blocks == before.index[before["step"] == FINAL_CHARGE_STEP][-1]]

            time_s = block["Testtime"].to_numpy(dtype=float)
            time_s = time_s - time_s[0]
            current = block["Current"].to_numpy(dtype=float)
            voltage = block["Voltage"].to_numpy(dtype=float)
            limit_v = float(np.max(voltage))
            at_limit = voltage >= limit_v - 5e-3

            record = {
                "duration_s": float(time_s[-1]),
                "charge_ah": coulomb_count_ah(block),
                "final_current_ma": 1000.0 * float(current[-1]),
                "voltage_limit_v": limit_v,
                "fraction_at_limit": float(np.mean(at_limit)),
                "_time": time_s,
                "_current": current,
                "_at_limit": at_limit,
            }
            records[(family, temperature_c)] = record
            if temperature_c == int(REFERENCE_TEMPERATURE_C):
                reference_current[family] = float(current[-1])

    rows: list[dict[str, object]] = []
    for (family, temperature_c), record in records.items():
        target = reference_current[family]
        time_s = record.pop("_time")
        current = record.pop("_current")
        at_limit = record.pop("_at_limit")

        exponent, extra_s, extra_ah = np.nan, np.nan, 0.0
        if at_limit.any() and 1e-3 * record["final_current_ma"] > target > 0:
            cv_time, cv_current = time_s[at_limit], current[at_limit]
            half = cv_time >= 0.5 * (cv_time[0] + cv_time[-1])
            t_fit, i_fit = cv_time[half], cv_current[half]
            usable = (t_fit > 0) & (i_fit > 0)
            if usable.sum() >= 8:
                # I(t) = a t^-b, integrated analytically from the observed
                # endpoint to the time at which I reaches the reference current.
                slope, intercept = np.polyfit(np.log(t_fit[usable]), np.log(i_fit[usable]), 1)
                exponent, amplitude = float(-slope), float(np.exp(intercept))
                if exponent > 0:
                    t_end = float(t_fit[usable][-1])
                    t_target = float((amplitude / target) ** (1.0 / exponent))
                    if t_target > t_end:
                        extra_s = t_target - t_end
                        integral = (
                            amplitude * np.log(t_target / t_end)
                            if abs(exponent - 1.0) < 1e-6
                            else amplitude
                            * (t_target ** (1.0 - exponent) - t_end ** (1.0 - exponent))
                            / (1.0 - exponent)
                        )
                        extra_ah = float(integral / 3600.0)

        rows.append(
            {
                "cell_family": family,
                "temperature_deg_c": float(temperature_c),
                **record,
                "reference_final_current_ma": 1000.0 * target,
                "taper_exponent": exponent,
                "extrapolated_extra_time_s": extra_s,
                "extrapolated_extra_charge_ah": extra_ah,
                "charge_ah_taper_corrected": record["charge_ah"] + extra_ah,
            }
        )

    taper = pd.DataFrame.from_records(rows)
    taper["rank"] = taper["cell_family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    return taper.sort_values(["rank", "temperature_deg_c"]).drop(columns="rank")


def rested_curve(segments: pd.DataFrame, family: str, temperature_c: float):
    """Cumulative charge removed and rested voltage at the end of each segment."""
    run = (
        segments[
            (segments["cell_family"] == family)
            & (segments["temperature_deg_c"] == temperature_c)
            & (segments["direction"] == "discharge")
        ]
        .dropna(subset=["ocv_voltage_v"])
        .sort_values("segment_index")
    )
    return (
        run["segment_capacity_ah"].to_numpy(dtype=float).cumsum(),
        run["ocv_voltage_v"].to_numpy(dtype=float),
    )


def best_shift(grid: np.ndarray, cold_on_grid: np.ndarray, warm_q, warm_v):
    residuals = np.array(
        [np.interp(grid + shift, warm_q, warm_v) - cold_on_grid for shift in SHIFT_GRID]
    )
    rms = np.sqrt(np.mean(residuals**2, axis=1))
    best = int(np.argmin(rms))
    zero = int(np.argmin(np.abs(SHIFT_GRID)))
    return SHIFT_GRID[best], rms[best], rms[zero], residuals[zero].mean(), best


def charge_axis_alignment(segments: pd.DataFrame) -> pd.DataFrame:
    """Translation along the charge axis that best superposes the rested curves."""
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        warm_q, warm_v = rested_curve(segments, family, REFERENCE_TEMPERATURE_C)
        span = warm_q[-1]
        central = (warm_q >= 0.2 * span) & (warm_q <= 0.8 * span)
        slope = abs(float(np.polyfit(warm_q[central], warm_v[central], 1)[0]))

        for temperature_c in sorted(TEMPERATURES):
            if float(temperature_c) == REFERENCE_TEMPERATURE_C:
                continue
            cold_q, cold_v = rested_curve(segments, family, float(temperature_c))
            upper = min(cold_q[-1], warm_q[-1]) - MAX_SHIFT_AH
            if upper <= cold_q[0]:
                continue
            grid = np.linspace(cold_q[0], upper, 600)
            cold_on_grid = np.interp(grid, cold_q, cold_v)

            shift, rms_fit, rms_zero, mean_offset, index = best_shift(
                grid, cold_on_grid, warm_q, warm_v
            )

            # A preparation offset is rigid: the same shift fits every part of
            # the window. Fitting thirds separately tests that.
            thirds = []
            for piece in np.array_split(np.arange(grid.size), 3):
                thirds.append(
                    float(
                        best_shift(grid[piece], cold_on_grid[piece], warm_q, warm_v)[0]
                    )
                )

            rows.append(
                {
                    "cell_family": family,
                    "temperature_deg_c": float(temperature_c),
                    "window_low_ah": float(grid[0]),
                    "window_high_ah": float(grid[-1]),
                    "central_ocv_slope_v_per_ah": slope,
                    "mean_offset_unshifted_mv": 1000.0 * float(mean_offset),
                    "rms_unshifted_mv": 1000.0 * float(rms_zero),
                    "fitted_shift_ah": float(shift),
                    "rms_at_fitted_shift_mv": 1000.0 * float(rms_fit),
                    "rms_reduction_percent": 100.0 * (1.0 - rms_fit / rms_zero)
                    if rms_zero > 0
                    else np.nan,
                    "shift_first_third_ah": thirds[0],
                    "shift_middle_third_ah": thirds[1],
                    "shift_last_third_ah": thirds[2],
                    "shift_spread_ah": float(max(thirds) - min(thirds)),
                    "shift_at_grid_edge": bool(index in (0, SHIFT_GRID.size - 1)),
                }
            )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    pd.set_option("display.width", 220)

    ledger = charge_input_ledger()
    taper = cv_taper_extrapolation()
    alignment = charge_axis_alignment(pd.read_csv(SEGMENTS))

    for path, table in (
        (LEDGER_OUTPUT, ledger),
        (TAPER_OUTPUT, taper),
        (ALIGNMENT_OUTPUT, alignment),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    print("=" * 100)
    print("1-2. CHARGE-INPUT LEDGER: Ah from the lower-cutoff anchor to StepID 96,")
    print("     and the rested voltage the discharge starts from")
    print("=" * 100)
    print(
        ledger[
            [
                "cell_family",
                "temperature_deg_c",
                "charge_input_hppc_ah",
                "charge_input_topup_ah",
                "charge_input_final_step_ah",
                "charge_input_total_ah",
                "charge_input_deficit_vs_25c_ah",
                "anchor_discharge_ah",
                "rested_start_voltage_v",
                "rested_start_offset_vs_25c_mv",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print("=" * 100)
    print("3. TERMINATING CHARGE STEP (89) AND ITS CV TAPER")
    print("=" * 100)
    print(
        taper[
            [
                "cell_family",
                "temperature_deg_c",
                "duration_s",
                "fraction_at_limit",
                "final_current_ma",
                "charge_ah",
                "taper_exponent",
                "extrapolated_extra_charge_ah",
                "charge_ah_taper_corrected",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print("=" * 100)
    print("4. CHARGE-AXIS ALIGNMENT of the rested-voltage curves against 25 degC")
    print("=" * 100)
    print(
        alignment[
            [
                "cell_family",
                "temperature_deg_c",
                "central_ocv_slope_v_per_ah",
                "mean_offset_unshifted_mv",
                "rms_unshifted_mv",
                "fitted_shift_ah",
                "rms_at_fitted_shift_mv",
                "rms_reduction_percent",
                "shift_first_third_ah",
                "shift_middle_third_ah",
                "shift_last_third_ah",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print("=" * 100)
    print("VERDICT: misalignment the offset would need, against the charge deficit measured")
    print("=" * 100)
    verdict = alignment.merge(
        ledger[
            [
                "cell_family",
                "temperature_deg_c",
                "charge_input_deficit_vs_25c_ah",
                "final_step_deficit_vs_25c_ah",
                "rested_start_offset_vs_25c_mv",
            ]
        ],
        on=["cell_family", "temperature_deg_c"],
    )
    verdict["shift_explained_by_ledger_percent"] = (
        100.0 * verdict["charge_input_deficit_vs_25c_ah"] / verdict["fitted_shift_ah"]
    )
    print(
        verdict[
            [
                "cell_family",
                "temperature_deg_c",
                "fitted_shift_ah",
                "charge_input_deficit_vs_25c_ah",
                "final_step_deficit_vs_25c_ah",
                "shift_explained_by_ledger_percent",
                "rested_start_offset_vs_25c_mv",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    for path in (LEDGER_OUTPUT, TAPER_OUTPUT, ALIGNMENT_OUTPUT):
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
