#!/usr/bin/env python3
"""Run interpretable baseline models on the integrated descriptor matrix."""

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
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = [
    "target_usable_energy_retention_vs_25c",
    "target_pulse_r_1s_growth_vs_25c",
    "target_pulse_power_proxy_vs_25c",
    "target_low_temperature_penalty",
]

BASE_FEATURES = [
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
]

MODELS = {
    "linear_core": LinearRegression(),
    "ridge_core": Ridge(alpha=1.0),
    "lasso_core": Lasso(alpha=0.005, max_iter=100000, tol=1e-5),
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_modeling_frame(matrix_path: str, target_path: str) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path)
    targets = pd.read_csv(target_path)
    frame = matrix.merge(
        targets.drop(columns=["chemistry_type"], errors="ignore"),
        on=["cell_family", "temperature_deg_c"],
        how="inner",
    )
    frame["is_sodium_ion"] = (frame["chemistry_type"] == "sodium-ion").astype(float)
    return frame


def features_for_target(target: str) -> list[str]:
    features = list(BASE_FEATURES)
    if target == "target_capacity_retention_vs_25c":
        features = [feature for feature in features if feature != "discharge_capacity_retention_vs_25c"]
    return features


def build_pipeline(model_name: str):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MODELS[model_name],
    )


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


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    baseline = np.full_like(y_true, float(np.nanmean(y_train)), dtype=float)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline)))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "train_mean_baseline_rmse": baseline_rmse,
        "rmse_skill_vs_train_mean": 1.0 - rmse / baseline_rmse if baseline_rmse else np.nan,
    }


def run_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    coefficient_records: list[dict[str, object]] = []
    splits = validation_splits(frame)

    for target in TARGETS:
        features = features_for_target(target)
        modeling = frame.dropna(subset=[target]).copy()
        x_all = modeling[features]
        y_all = modeling[target].to_numpy(dtype=float)

        for model_name in MODELS:
            full_pipeline = build_pipeline(model_name)
            full_pipeline.fit(x_all, y_all)
            regressor = full_pipeline.steps[-1][1]
            if hasattr(regressor, "coef_"):
                for feature, coefficient in zip(features, regressor.coef_):
                    coefficient_records.append(
                        {
                            "target": target,
                            "model": model_name,
                            "feature": feature,
                            "standardized_coefficient": float(coefficient),
                        }
                    )

            for split_type, split_name, train_mask, test_mask in splits:
                train_index = np.where(train_mask)[0]
                test_index = np.where(test_mask)[0]
                if len(train_index) < 3 or len(test_index) < 1:
                    continue
                x_train = x_all.iloc[train_index]
                y_train = y_all[train_index]
                x_test = x_all.iloc[test_index]
                y_test = y_all[test_index]

                pipeline = build_pipeline(model_name)
                pipeline.fit(x_train, y_train)
                y_pred = pipeline.predict(x_test)
                metrics = evaluate_predictions(y_test, y_pred, y_train)
                score_records.append(
                    {
                        "target": target,
                        "model": model_name,
                        "split_type": split_type,
                        "split_name": split_name,
                        "n_train": len(train_index),
                        "n_test": len(test_index),
                        **metrics,
                    }
                )
                for row_index, true_value, prediction in zip(test_index, y_test, y_pred):
                    row = modeling.iloc[row_index]
                    prediction_records.append(
                        {
                            "target": target,
                            "model": model_name,
                            "split_type": split_type,
                            "split_name": split_name,
                            "cell_family": row["cell_family"],
                            "chemistry_type": row["chemistry_type"],
                            "temperature_deg_c": row["temperature_deg_c"],
                            "y_true": float(true_value),
                            "y_pred": float(prediction),
                            "residual": float(true_value - prediction),
                        }
                    )

    return (
        pd.DataFrame.from_records(score_records),
        pd.DataFrame.from_records(prediction_records),
        pd.DataFrame.from_records(coefficient_records),
    )


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["target", "model", "split_type"], dropna=False)
        .agg(
            folds=("split_name", "count"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            median_r2=("r2", "median"),
            mean_skill_vs_train_mean=("rmse_skill_vs_train_mean", "mean"),
        )
        .reset_index()
        .sort_values(["target", "split_type", "mean_rmse"])
    )


def summarize_importance(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return coefficients
    importance = coefficients.copy()
    importance["absolute_standardized_coefficient"] = importance["standardized_coefficient"].abs()
    importance["importance_rank"] = (
        importance.groupby(["target", "model"])["absolute_standardized_coefficient"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return importance.sort_values(["target", "model", "importance_rank"]).reset_index(drop=True)


def plot_score_summary(summary: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    view = summary[
        (summary["model"] == "ridge_core")
        & summary["split_type"].isin(["leave_one_temperature_out", "leave_one_family_out"])
    ].copy()
    targets = list(dict.fromkeys(view["target"]))
    split_types = ["leave_one_temperature_out", "leave_one_family_out"]
    x = np.arange(len(targets))
    width = 0.35
    fig, axis = plt.subplots(figsize=(10, 4))
    for offset_index, split_type in enumerate(split_types):
        values = (
            view[view["split_type"] == split_type]
            .set_index("target")
            .reindex(targets)["mean_rmse"]
            .to_numpy(dtype=float)
        )
        axis.bar(x + (offset_index - 0.5) * width, values, width=width, label=split_type)
    axis.set_xticks(x)
    axis.set_xticklabels([target.replace("target_", "").replace("_", "\n") for target in targets], fontsize=8)
    axis.set_ylabel("Mean RMSE")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    ensure_parent(path)
    lines = [
        "# Baseline Modeling",
        "",
        "These are small-data screening baselines over 20 family-temperature rows.",
        "",
        "## Validation Splits",
        "",
        "- Leave-one-temperature-out.",
        "- Leave-one-family-out.",
        "- Li-ion to sodium-ion transfer.",
        "- Sodium-ion to Li-ion transfer.",
        "",
        "## Best Ridge Scores",
        "",
    ]
    ridge = summary[summary["model"] == "ridge_core"]
    for target in TARGETS:
        target_view = ridge[ridge["target"] == target].sort_values("mean_rmse")
        if target_view.empty:
            continue
        best = target_view.iloc[0]
        lines.append(
            f"- {target}: best split summary is {best['split_type']} with mean RMSE {best['mean_rmse']:.4f} "
            f"and mean skill {best['mean_skill_vs_train_mean']:.3f}."
        )

    lines.extend(["", "## Largest Ridge Coefficients", ""])
    ridge_coefficients = coefficients[coefficients["model"] == "ridge_core"].copy()
    if not ridge_coefficients.empty:
        ridge_coefficients["abs_coefficient"] = ridge_coefficients["standardized_coefficient"].abs()
        for target in TARGETS:
            top = ridge_coefficients[ridge_coefficients["target"] == target].nlargest(3, "abs_coefficient")
            if top.empty:
                continue
            items = ", ".join(
                f"{row['feature']} ({row['standardized_coefficient']:.3f})" for _, row in top.iterrows()
            )
            lines.append(f"- {target}: {items}")

    lines.extend(
        [
            "",
            "## Cautions",
            "",
            "- The sample size is too small for final model claims.",
            "- Coefficients are standardized and should be interpreted as screening indicators only.",
            "- Transfer splits are intentionally hard and should mainly reveal extrapolation risk.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="data_processed/descriptors/family_temperature_descriptor_matrix.csv")
    parser.add_argument("--targets", default="data_processed/descriptors/family_temperature_targets.csv")
    args = parser.parse_args()

    frame = load_modeling_frame(args.matrix, args.targets)
    scores, predictions, coefficients = run_models(frame)
    summary = summarize_scores(scores)
    importance = summarize_importance(coefficients)

    outputs = {
        "results/models/baseline_model_scores.csv": scores,
        "results/models/baseline_model_score_summary.csv": summary,
        "results/models/baseline_model_predictions.csv": predictions,
        "results/models/baseline_model_coefficients.csv": coefficients,
        "results/tables/table_baseline_descriptor_importance.csv": importance,
    }
    for file_name, output in outputs.items():
        output_path = Path(file_name)
        ensure_parent(output_path)
        output.to_csv(output_path, index=False)
        print(f"{file_name}: {len(output)} rows")

    plot_score_summary(summary, Path("results/figures/fig09_baseline_model_scores.png"))
    write_report(Path("results/reports/baseline_modeling.md"), summary, coefficients)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
