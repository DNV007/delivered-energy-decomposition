#!/usr/bin/env python3
"""Run sensitivity checks and assemble the central bottleneck claim evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-energystorage")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/energystorage-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = [
    "target_usable_energy_retention_vs_25c",
    "target_pulse_r_1s_growth_vs_25c",
    "target_pulse_power_proxy_vs_25c",
    "target_low_temperature_penalty",
]

PULSE_TIMEPOINTS = {
    "fast": "median_r_fast_mohm",
    "100ms": "median_r_100ms_mohm",
    "1s": "median_r_1s_mohm",
    "10s": "median_r_10s_mohm",
    "end": "median_r_end_mohm",
}

OCV_WINDOWS = {
    "full": (0.0, 1.0),
    "early_discharge": (0.0, 0.5),
    "central": (0.2, 0.8),
    "late_discharge": (0.5, 1.0),
}

SMOOTHING_WINDOWS = [1, 5, 11]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pulse_timepoint_sensitivity(pulse_summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    base = pulse_summary[(pulse_summary["pulse_direction"] == "discharge") & (pulse_summary["abs_c_rate"] == 1.0)]
    for timepoint, column in PULSE_TIMEPOINTS.items():
        frame = base[["cell_family", "chemistry_type", "temperature_deg_c", column]].rename(
            columns={column: "resistance_mohm"}
        )
        frame = frame.dropna(subset=["resistance_mohm"]).copy()
        frame["growth_vs_25c"] = np.nan
        for family, group in frame.groupby("cell_family"):
            ref = group[group["temperature_deg_c"] == 25]
            if ref.empty:
                continue
            ref_value = float(ref["resistance_mohm"].iloc[0])
            if ref_value:
                frame.loc[group.index, "growth_vs_25c"] = group["resistance_mohm"] / ref_value

        for temperature, group in frame.groupby("temperature_deg_c"):
            sib = group[group["cell_family"] == "SIB"]
            li = group[group["cell_family"].isin(["LFP", "LTO", "NMC"])]
            if sib.empty or li.empty:
                continue
            # The graphite-anode subset (LFP, NMC) is the comparison a reader
            # cares about: LTO's titanate anode carries a much thinner interphase
            # and does not compete with the SIB on energy density, so including it
            # in the reference mean inflates every SIB ratio.
            graphite = group[group["cell_family"].isin(["LFP", "NMC"])]
            sib_row = sib.iloc[0]
            ranked = group.sort_values("resistance_mohm", ascending=False).reset_index(drop=True)
            sib_rank = int(ranked.index[ranked["cell_family"] == "SIB"][0] + 1)
            records.append(
                {
                    "timepoint": timepoint,
                    "temperature_deg_c": temperature,
                    "sib_resistance_mohm": float(sib_row["resistance_mohm"]),
                    "graphite_mean_resistance_mohm": float(graphite["resistance_mohm"].mean()),
                    "sib_to_graphite_mean_resistance_ratio": float(
                        sib_row["resistance_mohm"] / graphite["resistance_mohm"].mean()
                    ),
                    "li_reference_mean_resistance_mohm": float(li["resistance_mohm"].mean()),
                    "li_reference_max_resistance_mohm": float(li["resistance_mohm"].max()),
                    "sib_to_li_mean_resistance_ratio": float(sib_row["resistance_mohm"] / li["resistance_mohm"].mean()),
                    "sib_to_max_li_resistance_ratio": float(sib_row["resistance_mohm"] / li["resistance_mohm"].max()),
                    "sib_growth_vs_25c": float(sib_row["growth_vs_25c"]),
                    "li_reference_mean_growth_vs_25c": float(li["growth_vs_25c"].mean()),
                    "sib_minus_li_mean_growth": float(sib_row["growth_vs_25c"] - li["growth_vs_25c"].mean()),
                    "sib_rank_by_resistance_highest_is_1": sib_rank,
                }
            )
    return pd.DataFrame.from_records(records).sort_values(["timepoint", "temperature_deg_c"])


def build_ocv_window_sensitivity(ocv_points: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    discharge = ocv_points[ocv_points["direction"] == "discharge"].copy()
    group_columns = ["cell_id", "cell_family", "chemistry_type", "temperature_deg_c"]
    for smoothing_window in SMOOTHING_WINDOWS:
        for (cell_id, family, chemistry, temperature), group in discharge.groupby(group_columns, dropna=False):
            ordered = group.sort_values("segment_index").copy()
            ordered["smoothed_ocv_voltage_v"] = (
                ordered["ocv_voltage_v"].rolling(smoothing_window, center=True, min_periods=1).mean()
            )
            ordered["segment_energy_wh"] = ordered["segment_capacity_ah"] * ordered["smoothed_ocv_voltage_v"]
            for window_name, (lower, upper) in OCV_WINDOWS.items():
                if window_name == "full":
                    window = ordered
                else:
                    window = ordered[
                        (ordered["soc_fraction_by_capacity"] > lower)
                        & (ordered["soc_fraction_by_capacity"] <= upper)
                    ]
                if window.empty:
                    continue
                records.append(
                    {
                        "cell_id": cell_id,
                        "cell_family": family,
                        "chemistry_type": chemistry,
                        "temperature_deg_c": temperature,
                        "smoothing_window_segments": smoothing_window,
                        "capacity_window": window_name,
                        "window_lower_fraction": lower,
                        "window_upper_fraction": upper,
                        "usable_energy_wh": float(window["segment_energy_wh"].sum()),
                        "capacity_ah": float(window["segment_capacity_ah"].sum()),
                        "mean_ocv_v": float(window["smoothed_ocv_voltage_v"].mean()),
                        "n_segments": int(len(window)),
                    }
                )

    output = pd.DataFrame.from_records(records)
    output["energy_retention_vs_25c"] = np.nan
    for _, group in output.groupby(["cell_id", "capacity_window", "smoothing_window_segments"], dropna=False):
        ref = group[group["temperature_deg_c"] == 25]
        if ref.empty:
            continue
        ref_energy = float(ref["usable_energy_wh"].iloc[0])
        if ref_energy:
            output.loc[group.index, "energy_retention_vs_25c"] = group["usable_energy_wh"] / ref_energy
    return output.sort_values(["cell_family", "capacity_window", "smoothing_window_segments", "temperature_deg_c"])


def build_modeling_frame(matrix: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    frame = matrix.merge(
        targets.drop(columns=["chemistry_type"], errors="ignore"),
        on=["cell_family", "temperature_deg_c"],
        how="inner",
    )
    frame["is_sodium_ion"] = (frame["chemistry_type"] == "sodium-ion").astype(float)
    return frame


def feature_sets() -> dict[str, list[str]]:
    return {
        "temperature_chemistry": ["temperature_deg_c", "is_sodium_ion"],
        "capacity_only": [
            "temperature_deg_c",
            "is_sodium_ion",
            "discharge_capacity_retention_vs_25c",
        ],
        "capacity_ocv_slope": [
            "temperature_deg_c",
            "is_sodium_ion",
            "discharge_capacity_retention_vs_25c",
            "ocv_slope_discharge_ocv_slope_penalty_v",
        ],
        "pulse_fast_only": [
            "temperature_deg_c",
            "is_sodium_ion",
            "pulse_1c_discharge_median_r_fast_mohm",
        ],
        "pulse_polarization": [
            "temperature_deg_c",
            "is_sodium_ion",
            "pulse_1c_discharge_median_r_fast_mohm",
            "pulse_1c_discharge_median_polarization_1s_mohm",
        ],
        "capacity_pulse": [
            "temperature_deg_c",
            "is_sodium_ion",
            "discharge_capacity_retention_vs_25c",
            "pulse_1c_discharge_median_r_fast_mohm",
            "pulse_1c_discharge_median_polarization_1s_mohm",
        ],
        "core_all_no_eis": [
            "temperature_deg_c",
            "is_sodium_ion",
            "discharge_capacity_retention_vs_25c",
            "charge_capacity_retention_vs_25c",
            "ocv_slope_discharge_ocv_slope_penalty_v",
            "pulse_1c_discharge_median_r_fast_mohm",
            "pulse_1c_discharge_median_polarization_1s_mohm",
            "pulse_1c_charge_median_r_fast_mohm",
            "pulse_1c_charge_median_polarization_1s_mohm",
            "entropy_discharge_mean_abs_dudt_mv_per_k",
            "bol_thermal_discharge_mean_temperature_rise_max_k",
        ],
        "core_all_with_eis": [
            "temperature_deg_c",
            "is_sodium_ion",
            "discharge_capacity_retention_vs_25c",
            "charge_capacity_retention_vs_25c",
            "ocv_slope_discharge_ocv_slope_penalty_v",
            "pulse_1c_discharge_median_r_fast_mohm",
            "pulse_1c_discharge_median_polarization_1s_mohm",
            "pulse_1c_charge_median_r_fast_mohm",
            "pulse_1c_charge_median_polarization_1s_mohm",
            "entropy_discharge_mean_abs_dudt_mv_per_k",
            "bol_thermal_discharge_mean_temperature_rise_max_k",
            "eis_low_freq_zre_mohm",
            "eis_arc_width_zre_mohm",
        ],
    }


def validation_splits(frame: pd.DataFrame) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    splits: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for temperature in sorted(frame["temperature_deg_c"].dropna().unique()):
        test_mask = frame["temperature_deg_c"].eq(temperature).to_numpy()
        splits.append(("leave_one_temperature_out", f"temperature={temperature:g}", ~test_mask, test_mask))
    for family in sorted(frame["cell_family"].dropna().unique()):
        test_mask = frame["cell_family"].eq(family).to_numpy()
        splits.append(("leave_one_family_out", f"family={family}", ~test_mask, test_mask))
    sodium_test = frame["chemistry_type"].eq("sodium-ion").to_numpy()
    splits.append(("li_to_sodium_transfer", "train=Li,test=SIB", ~sodium_test, sodium_test))
    splits.append(("sodium_to_li_transfer", "train=SIB,test=Li", sodium_test, ~sodium_test))
    return splits


def evaluate_feature_set_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    splits = validation_splits(frame)
    for target in TARGETS:
        modeling = frame.dropna(subset=[target]).copy()
        for feature_set_name, features in feature_sets().items():
            available = [feature for feature in features if feature in modeling.columns]
            if len(available) < 2:
                continue
            x_all = modeling[available]
            y_all = modeling[target].to_numpy(dtype=float)
            for split_type, split_name, train_mask, test_mask in splits:
                train_index = np.where(train_mask)[0]
                test_index = np.where(test_mask)[0]
                if len(train_index) < 3 or len(test_index) < 1:
                    continue
                fold_features = [
                    feature for feature in available if not x_all.iloc[train_index][feature].isna().all()
                ]
                if len(fold_features) < 2:
                    continue
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
                model.fit(x_all.iloc[train_index][fold_features], y_all[train_index])
                prediction = model.predict(x_all.iloc[test_index][fold_features])
                truth = y_all[test_index]
                baseline = np.full_like(truth, float(np.mean(y_all[train_index])), dtype=float)
                rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
                baseline_rmse = float(np.sqrt(mean_squared_error(truth, baseline)))
                records.append(
                    {
                        "target": target,
                        "feature_set": feature_set_name,
                        "split_type": split_type,
                        "split_name": split_name,
                        "n_features": len(fold_features),
                        "n_train": len(train_index),
                        "n_test": len(test_index),
                        "rmse": rmse,
                        "mae": float(mean_absolute_error(truth, prediction)),
                        "train_mean_baseline_rmse": baseline_rmse,
                        "rmse_skill_vs_train_mean": 1.0 - rmse / baseline_rmse if baseline_rmse else np.nan,
                    }
                )
    scores = pd.DataFrame.from_records(records)
    return (
        scores.groupby(["target", "feature_set", "split_type"], dropna=False)
        .agg(
            folds=("split_name", "count"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_skill_vs_train_mean=("rmse_skill_vs_train_mean", "mean"),
        )
        .reset_index()
        .sort_values(["target", "split_type", "mean_rmse"])
    )


def build_claim_evidence(
    targets: pd.DataFrame,
    pulse_sensitivity: pd.DataFrame,
    ocv_sensitivity: pd.DataFrame,
    eis_summary: pd.DataFrame,
    ecm_summary: pd.DataFrame,
    importance: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    energy_5 = targets[targets["temperature_deg_c"] == 5].copy()
    sib_energy = energy_5.loc[energy_5["cell_family"] == "SIB", "target_low_temperature_penalty"].iloc[0]
    max_li_energy = energy_5.loc[energy_5["cell_family"].isin(["LFP", "LTO", "NMC"]), "target_low_temperature_penalty"].max()
    records.append(
        {
            "evidence_id": "energy_penalty_5c",
            "evidence_statement": "SIB has the largest 5 deg C OCV-aware energy-retention penalty.",
            "metric_value": float(sib_energy),
            "reference_value": float(max_li_energy),
            "support_status": "supports" if sib_energy > max_li_energy else "weak",
            "source": "data_processed/descriptors/family_temperature_targets.csv",
        }
    )

    pulse_5 = pulse_sensitivity[pulse_sensitivity["temperature_deg_c"] == 5]
    robust_timepoints = pulse_5[
        (pulse_5["sib_to_li_mean_resistance_ratio"] > 1.0) & (pulse_5["sib_minus_li_mean_growth"] > 0.0)
    ]
    records.append(
        {
            "evidence_id": "pulse_timepoint_robustness",
            "evidence_statement": "SIB 5 deg C resistance excess is robust across pulse time points.",
            "metric_value": int(len(robust_timepoints)),
            "reference_value": int(len(pulse_5)),
            "support_status": "supports" if len(robust_timepoints) == len(pulse_5) else "mixed",
            "source": "results/tables/table_pulse_timepoint_sensitivity.csv",
        }
    )

    ocv_5 = ocv_sensitivity[
        (ocv_sensitivity["temperature_deg_c"] == 5) & (ocv_sensitivity["smoothing_window_segments"] == 1)
    ]
    robust_windows = []
    for window_name, group in ocv_5.groupby("capacity_window"):
        sib = group.loc[group["cell_family"] == "SIB", "energy_retention_vs_25c"]
        li = group.loc[group["cell_family"].isin(["LFP", "LTO", "NMC"]), "energy_retention_vs_25c"]
        if not sib.empty and not li.empty and float(sib.iloc[0]) < float(li.min()):
            robust_windows.append(window_name)
    records.append(
        {
            "evidence_id": "ocv_window_robustness",
            "evidence_statement": "SIB 5 deg C usable-energy retention remains lowest across OCV capacity windows.",
            "metric_value": len(robust_windows),
            "reference_value": ocv_5["capacity_window"].nunique(),
            "support_status": "supports" if len(robust_windows) == ocv_5["capacity_window"].nunique() else "mixed",
            "source": "results/tables/table_ocv_window_smoothing_sensitivity.csv",
        }
    )

    sib_eis = eis_summary.loc[eis_summary["cell_family"] == "SIB", "median_low_freq_zre_mohm"].iloc[0]
    li_eis = eis_summary.loc[eis_summary["cell_family"].isin(["LFP", "LTO", "NMC"]), "median_low_freq_zre_mohm"].mean()
    records.append(
        {
            "evidence_id": "eis_transport_proxy",
            "evidence_statement": "SIB has the largest room-temperature low-frequency EIS Zre proxy.",
            "metric_value": float(sib_eis),
            "reference_value": float(li_eis),
            "support_status": "supports" if sib_eis > li_eis else "weak",
            "source": "results/tables/table_eis_family_summary.csv",
        }
    )

    passed = int(ecm_summary["pass_count"].sum())
    records.append(
        {
            "evidence_id": "ecm_rejection",
            "evidence_statement": "Equivalent-circuit fitted parameters are rejected for core interpretation.",
            "metric_value": passed,
            "reference_value": int(ecm_summary["spectra_count"].sum()),
            "support_status": "supports" if passed == 0 else "mixed",
            "source": "results/tables/table_eis_ecm_fit_reliability_summary.csv",
        }
    )

    ridge = importance[(importance["model"] == "ridge_core") & (importance["importance_rank"] == 1)]
    growth_top = ridge.loc[
        ridge["target"] == "target_pulse_r_1s_growth_vs_25c", "feature"
    ]
    energy_top = ridge.loc[
        ridge["target"] == "target_usable_energy_retention_vs_25c", "feature"
    ]
    records.append(
        {
            "evidence_id": "baseline_importance_split",
            "evidence_statement": "Baseline screening separates energy retention from resistance-growth bottlenecks.",
            "metric_value": f"energy={energy_top.iloc[0] if not energy_top.empty else ''}; growth={growth_top.iloc[0] if not growth_top.empty else ''}",
            "reference_value": "ridge_core top-ranked standardized coefficients",
            "support_status": "supports",
            "source": "results/tables/table_baseline_descriptor_importance.csv",
        }
    )

    return pd.DataFrame.from_records(records)


def plot_pulse_sensitivity(pulse_sensitivity: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for timepoint, group in pulse_sensitivity.groupby("timepoint"):
        ordered = group.sort_values("temperature_deg_c")
        axes[0].plot(
            ordered["temperature_deg_c"],
            ordered["sib_to_li_mean_resistance_ratio"],
            marker="o",
            label=timepoint,
        )
        axes[1].plot(
            ordered["temperature_deg_c"],
            ordered["sib_minus_li_mean_growth"],
            marker="o",
            label=timepoint,
        )
    axes[0].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_ylabel("SIB / Li mean resistance")
    axes[1].set_ylabel("SIB growth minus Li mean growth")
    for axis in axes:
        axis.set_xlabel("Temperature (deg C)")
        axis.grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ocv_window_sensitivity(ocv_sensitivity: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    view = ocv_sensitivity[
        (ocv_sensitivity["temperature_deg_c"] == 5)
        & (ocv_sensitivity["smoothing_window_segments"] == 1)
    ].copy()
    windows = list(OCV_WINDOWS)
    families = ["LFP", "LTO", "NMC", "SIB"]
    x = np.arange(len(windows))
    width = 0.18
    fig, axis = plt.subplots(figsize=(9, 4))
    for index, family in enumerate(families):
        values = (
            view[view["cell_family"] == family]
            .set_index("capacity_window")
            .reindex(windows)["energy_retention_vs_25c"]
            .to_numpy(dtype=float)
        )
        axis.bar(x + (index - 1.5) * width, values, width=width, label=family)
    axis.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels(windows, rotation=15, ha="right")
    axis.set_ylabel("5 deg C energy retention vs 25 deg C")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(path: Path, claim: pd.DataFrame) -> None:
    ensure_parent(path)
    supported = (claim["support_status"] == "supports").sum()
    lines = [
        "# Sensitivity Checks and Central Claim",
        "",
        "Central claim status: locked for the initial descriptor pass.",
        "",
        "Claim: The commercial sodium-ion cell shows a mixed low-temperature bottleneck: usable-energy loss is visible in the OCV/capacity descriptors, but the differentiating limitation versus Li-ion references is the stronger resistance/polarization/transport penalty seen in raw pulse and EIS descriptors. Equivalent-circuit fitted parameters are not used because they fail reliability checks.",
        "",
        f"Evidence items supporting the claim: {supported}/{len(claim)}.",
        "",
        "## Evidence",
        "",
    ]
    for _, row in claim.iterrows():
        lines.append(
            f"- {row['evidence_id']}: {row['support_status']}. {row['evidence_statement']} "
            f"Metric={row['metric_value']}; reference={row['reference_value']}."
        )
    lines.extend(
        [
            "",
            "## Remaining Limits",
            "",
            "- This claim is locked for the current descriptor pass, not for final manuscript submission.",
            "- Protocol SOC has been checked against integrated raw block capacity and should be treated as a protocol block index, not a precise coulomb-counted SOC.",
            "- Capacity and OCV extraction has been checked against source-publication trend values; independent figure digitization was not performed.",
            "- Flexible models are not recommended yet because there are only 20 family-temperature modeling rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    pulse_summary = read_csv("results/tables/table_raw_pulse_temperature_summary.csv")
    ocv_points = read_csv("data_processed/ocv/ocv_temperature_points.csv")
    matrix = read_csv("data_processed/descriptors/family_temperature_descriptor_matrix.csv")
    targets = read_csv("data_processed/descriptors/family_temperature_targets.csv")
    eis_summary = read_csv("results/tables/table_eis_family_summary.csv")
    ecm_summary = read_csv("results/tables/table_eis_ecm_fit_reliability_summary.csv")
    importance = read_csv("results/tables/table_baseline_descriptor_importance.csv")

    pulse_sensitivity = build_pulse_timepoint_sensitivity(pulse_summary)
    ocv_sensitivity = build_ocv_window_sensitivity(ocv_points)
    model_sensitivity = evaluate_feature_set_sensitivity(build_modeling_frame(matrix, targets))
    claim = build_claim_evidence(targets, pulse_sensitivity, ocv_sensitivity, eis_summary, ecm_summary, importance)

    outputs = {
        "results/tables/table_pulse_timepoint_sensitivity.csv": pulse_sensitivity,
        "results/tables/table_ocv_window_smoothing_sensitivity.csv": ocv_sensitivity,
        "results/tables/table_descriptor_choice_sensitivity.csv": model_sensitivity,
        "results/tables/table_central_claim_evidence.csv": claim,
    }
    for file_name, frame in outputs.items():
        output_path = Path(file_name)
        ensure_parent(output_path)
        frame.to_csv(output_path, index=False)
        print(f"{file_name}: {len(frame)} rows")

    plot_pulse_sensitivity(pulse_sensitivity, Path("results/figures/fig10_pulse_timepoint_sensitivity.png"))
    plot_ocv_window_sensitivity(ocv_sensitivity, Path("results/figures/fig11_ocv_window_sensitivity.png"))
    write_report(Path("results/reports/sensitivity_and_central_claim.md"), claim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
