import re
from pathlib import Path

import pandas as pd
import pytest


REQUIRED_FAMILIES = {"LFP", "LTO", "NMC", "SIB"}


def read_required_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        pytest.skip(f"Descriptor output has not been generated: {path}")
    return pd.read_csv(csv_path)


def test_capacity_temperature_table_has_expected_families_and_temperatures():
    frame = read_required_csv("data_processed/capacity/capacity_temperature_summary.csv")
    assert set(frame["cell_family"]) == REQUIRED_FAMILIES
    assert set(frame["temperature_deg_c"]) == {5.0, 15.0, 25.0, 35.0, 45.0}
    assert len(frame) == 20


def test_hppc_table_keeps_negative_one_c_rate():
    frame = read_required_csv("data_processed/hppc/hppc_resistance_long.csv")
    assert set(frame["cell_family"]) == REQUIRED_FAMILIES
    assert -1.0 in set(frame["c_rate"])
    assert frame["r_fast_ohm"].notna().any()


def test_usable_energy_table_has_discharge_retention():
    frame = read_required_csv("data_processed/descriptors/usable_energy_retention.csv")
    discharge = frame[frame["direction"] == "discharge"]
    assert set(discharge["cell_family"]) == REQUIRED_FAMILIES
    assert discharge["energy_retention_vs_25c"].notna().all()
    assert len(discharge) == 20


def test_eis_family_summary_has_all_families():
    frame = read_required_csv("results/tables/table_eis_family_summary.csv")
    assert set(frame["cell_family"]) == REQUIRED_FAMILIES
    assert frame["median_low_freq_zre_mohm"].notna().all()

    # The family summary must cover the same ten room-temperature spectra per
    # family that the Kramers-Kronig validation runs on. An earlier narrower
    # temperature window silently dropped one spectrum per family, which put
    # the main text and the Supplementary Information on different subsets.
    assert (frame["spectra_count"] == 10).all()


def test_eis_subset_choice_does_not_change_the_family_comparison():
    frame = read_required_csv("results/tables/table_eis_subset_sensitivity.csv")
    assert len(frame) >= 3
    ratios = frame["sib_to_reference_ratio"]
    # The reference is the graphite-anode mean (LFP, NMC); LTO is excluded, so
    # this ratio is smaller than the 2.2 the three-family mean used to give.
    # What matters is that the SIB stays clearly above every reference.
    assert ratios.min() > 1.5
    # Every admissible subset must agree to within 1 percent; the manuscript
    # states this explicitly rather than resting on one subset.
    assert (ratios.max() - ratios.min()) / ratios.mean() < 0.01


def test_arc_apex_resolution_separates_the_sib():
    frame = read_required_csv("results/tables/table_eis_family_summary.csv").set_index("cell_family")
    # The characteristic-frequency descriptor is bimodal, so the manuscript
    # reports the fraction of spectra with a resolved apex rather than a
    # median. That fraction must remain monotone with the SIB at the top.
    assert frame.loc["SIB", "arc_apex_resolved_fraction"] == 1.0
    assert (
        frame.loc["SIB", "arc_apex_resolved_fraction"]
        > frame.loc["LFP", "arc_apex_resolved_fraction"]
        > frame.loc["NMC", "arc_apex_resolved_fraction"]
        >= frame.loc["LTO", "arc_apex_resolved_fraction"]
    )


def test_arrhenius_activation_energies_match_reported_values():
    frame = read_required_csv("results/tables/table_arrhenius_activation_energy.csv")
    headline = frame[
        (frame["pulse_direction"] == "discharge")
        & (frame["abs_c_rate"] == 1.0)
        & (frame["pulse_timepoint"] == "1s")
    ].set_index("cell_family")
    assert set(headline.index) == REQUIRED_FAMILIES

    # Values quoted in the abstract, Table 3, and the Conclusions.
    expected_mev = {"SIB": 373, "LFP": 284, "NMC": 238, "LTO": 74}
    for family, value in expected_mev.items():
        assert abs(headline.loc[family, "activation_energy_mev"] - value) < 1.0
        assert headline.loc[family, "r_squared"] > 0.94
        assert headline.loc[family, "n_points"] == 5

    # The SIB must exceed every reference at every discharge time point; this
    # is the claim the manuscript makes, not merely the 1 s comparison.
    discharge = frame[(frame["pulse_direction"] == "discharge") & (frame["abs_c_rate"] == 1.0)]
    for _, block in discharge.groupby("pulse_timepoint"):
        indexed = block.set_index("cell_family")["activation_energy_mev"]
        assert indexed["SIB"] > indexed[["LFP", "LTO", "NMC"]].max()


def test_activation_energy_ordering_survives_without_a_fitted_form():
    """The family ordering must not depend on assuming Arrhenius behaviour."""
    summary = read_required_csv("results/tables/table_raw_pulse_temperature_summary.csv")
    discharge = summary[
        (summary["pulse_direction"] == "discharge") & (summary["abs_c_rate"] == 1.0)
    ]
    pivot = discharge.pivot_table(
        index="cell_family", columns="temperature_deg_c", values="median_r_1s_mohm"
    )
    model_free = (pivot[5.0] / pivot[45.0]).sort_values(ascending=False)
    assert list(model_free.index) == ["SIB", "LFP", "NMC", "LTO"]

    fits = read_required_csv("results/tables/table_arrhenius_activation_energy.csv")
    headline = fits[
        (fits["pulse_direction"] == "discharge")
        & (fits["abs_c_rate"] == 1.0)
        & (fits["pulse_timepoint"] == "1s")
    ].sort_values("activation_energy_mev", ascending=False)
    assert list(headline["cell_family"]) == list(model_free.index)


def test_arrhenius_and_vtf_are_not_discriminated_by_the_measured_window():
    frame = read_required_csv("results/tables/table_arrhenius_vtf_comparison.csv")
    assert len(frame) == 120
    # The manuscript states that VTF describes the data at least as well in the
    # large majority of series, which is why Ea is presented as an empirical
    # sensitivity measure rather than as evidence of a single process.
    assert frame["vtf_fits_at_least_as_well"].mean() > 0.9
    assert frame["aicc_vtf"].isna().all()  # undefined at n = 5 for 3 parameters


def test_kramers_kronig_validation_matches_reported_pass_rate():
    frame = read_required_csv("results/tables/table_eis_kramers_kronig.csv")
    assert len(frame) == 40
    assert set(frame["cell_family"]) == REQUIRED_FAMILIES
    assert int(frame["kk_consistent"].sum()) == 37
    per_family = frame.groupby("cell_family")["kk_consistent"].sum().to_dict()
    assert per_family == {"SIB": 9, "LFP": 8, "LTO": 10, "NMC": 10}
    assert abs(frame["rms_residual"].median() * 100 - 0.33) < 0.05
    # All three failures sit at the low-frequency end of the sweep, which is
    # the relaxation signature the manuscript attributes them to.
    failures = frame[~frame["kk_consistent"]]
    assert (failures["worst_residual_freq_hz"] <= 0.15).all()


def test_raw_temperature_pulse_descriptors_have_expected_protocol_coverage():
    frame = read_required_csv("data_processed/hppc/raw_temperature_pulse_descriptors.csv")
    assert set(frame["cell_family"]) == REQUIRED_FAMILIES
    assert set(frame["temperature_deg_c"]) == {5.0, 15.0, 25.0, 35.0, 45.0}
    assert set(frame["pulse_direction"]) == {"charge", "discharge"}
    assert set(frame["nominal_c_rate"]) == {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0}
    assert frame.groupby(["cell_family", "temperature_deg_c"]).size().eq(126).all()
    assert frame["complete_10s"].any()
    assert frame["r_1s_ohm"].notna().any()


def test_raw_pulse_temperature_summary_has_sib_li_reference():
    summary = read_required_csv("results/tables/table_raw_pulse_temperature_summary.csv")
    soc_map = read_required_csv("results/tables/table_raw_pulse_soc_temperature_map.csv")
    comparison = read_required_csv("results/tables/table_raw_pulse_sib_vs_li_reference.csv")
    assert set(summary["cell_family"]) == REQUIRED_FAMILIES
    assert len(summary) == 120
    assert set(soc_map["cell_family"]) == REQUIRED_FAMILIES
    assert set(soc_map["nominal_soc_percent"]) == {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
    assert comparison["li_reference_family_count"].eq(3).all()
    row = comparison[
        (comparison["temperature_deg_c"] == 25.0)
        & (comparison["pulse_direction"] == "discharge")
        & (comparison["abs_c_rate"] == 1.0)
    ]
    assert not row.empty
    assert row["sib_to_li_r_1s_ratio"].notna().all()


def test_eis_hppc_consistency_table_has_room_temperature_family_comparison():
    consistency = read_required_csv("results/tables/table_eis_hppc_consistency.csv")
    correlations = read_required_csv("results/tables/table_eis_hppc_consistency_correlations.csv")
    assert set(consistency["cell_family"]) == REQUIRED_FAMILIES
    assert consistency["pulse_1c_discharge_r_1s_mohm"].notna().all()
    assert consistency["eis_low_freq_zre_mohm"].notna().all()
    assert correlations["n_families"].eq(4).all()


def test_ecm_fit_reliability_decision_outputs_are_present():
    fits = read_required_csv("results/tables/table_eis_ecm_fit_reliability.csv")
    summary = read_required_csv("results/tables/table_eis_ecm_fit_reliability_summary.csv")
    assert set(fits["model"]) == {
        "randles_cpe",
        "randles_cpe_warburg",
        "two_arc_cpe",
        "two_arc_cpe_warburg",
    }
    assert set(fits["cell_family"]) == REQUIRED_FAMILIES
    assert len(fits) == 160
    assert len(summary) == 16
    assert summary["pass_count"].sum() == 0

    # The two-arc candidates exist to test whether the single-arc failure is
    # one of model order. The manuscript argues it is not, on two grounds that
    # must both hold: the residual does not improve materially, and the extra
    # arc is not identified.
    by_model = fits.groupby("model")["residual_nrmse"].median()
    assert by_model["two_arc_cpe"] > 0.05
    assert by_model["randles_cpe"] - by_model["two_arc_cpe"] < 0.01
    two_arc_cv = fits[fits["model"] == "two_arc_cpe"]["elite_parameter_cv"].median()
    single_arc_cv = fits[fits["model"] == "randles_cpe"]["elite_parameter_cv"].median()
    assert single_arc_cv < 0.20 < two_arc_cv


def test_integrated_descriptor_matrix_and_targets_are_complete():
    long_matrix = read_required_csv("data_processed/descriptors/integrated_descriptor_long.csv")
    wide_matrix = read_required_csv("data_processed/descriptors/family_temperature_descriptor_matrix.csv")
    targets = read_required_csv("data_processed/descriptors/family_temperature_targets.csv")
    assert set(long_matrix["cell_family"].dropna()) == REQUIRED_FAMILIES
    assert {"capacity", "ocv_usable_energy", "ocv_slope", "raw_pulse", "eis", "entropy", "thermal_response"}.issubset(
        set(long_matrix["measurement_type"])
    )
    assert len(wide_matrix) == 20
    assert len(targets) == 20
    assert targets["target_usable_energy_retention_vs_25c"].notna().all()
    assert targets["target_pulse_r_1s_growth_vs_25c"].notna().all()
    assert (targets.loc[targets["temperature_deg_c"] >= 25.0, "target_low_temperature_penalty"] == 0.0).all()


def test_baseline_model_outputs_cover_core_splits_and_targets():
    scores = read_required_csv("results/models/baseline_model_scores.csv")
    summary = read_required_csv("results/models/baseline_model_score_summary.csv")
    predictions = read_required_csv("results/models/baseline_model_predictions.csv")
    importance = read_required_csv("results/tables/table_baseline_descriptor_importance.csv")
    expected_targets = {
        "target_usable_energy_retention_vs_25c",
        "target_pulse_r_1s_growth_vs_25c",
        "target_pulse_power_proxy_vs_25c",
        "target_low_temperature_penalty",
    }
    expected_splits = {
        "leave_one_temperature_out",
        "leave_one_family_out",
        "li_to_sodium_transfer",
        "sodium_to_li_transfer",
    }
    assert expected_targets.issubset(set(scores["target"]))
    assert expected_splits.issubset(set(scores["split_type"]))
    assert {"linear_core", "ridge_core", "lasso_core"}.issubset(set(scores["model"]))
    assert summary["mean_rmse"].notna().all()
    assert predictions["y_pred"].notna().all()
    assert importance["importance_rank"].min() == 1
    assert importance["absolute_standardized_coefficient"].notna().all()


def test_sensitivity_outputs_lock_initial_claim():
    pulse = read_required_csv("results/tables/table_pulse_timepoint_sensitivity.csv")
    ocv = read_required_csv("results/tables/table_ocv_window_smoothing_sensitivity.csv")
    descriptor_choices = read_required_csv("results/tables/table_descriptor_choice_sensitivity.csv")
    claim = read_required_csv("results/tables/table_central_claim_evidence.csv")
    assert set(pulse["timepoint"]) == {"fast", "100ms", "1s", "10s", "end"}
    pulse_5c = pulse[pulse["temperature_deg_c"] == 5.0]
    assert pulse_5c["sib_to_li_mean_resistance_ratio"].gt(1.0).all()
    assert pulse_5c["sib_minus_li_mean_growth"].gt(0.0).all()
    assert set(ocv["capacity_window"]) == {"full", "early_discharge", "central", "late_discharge"}
    assert set(ocv["smoothing_window_segments"]) == {1, 5, 11}
    assert descriptor_choices["mean_rmse"].notna().all()
    assert len(claim) == 6
    assert claim["support_status"].eq("supports").all()



def test_final_manuscript_figures_and_tables_exist():
    figures = [
        "figure1_data_architecture.png",
        "figure2_energy_decomposition.png",
        "figure4_raw_pulse_bottleneck.png",
        "figure_s2_pulse_block_maps.png",
        "figure5_eis_bottleneck_decomposition.png",
        "figure6_ocv_entropy_thermal_coupling.png",
    ]
    for name in figures:
        manuscript_path = Path("manuscript/figures") / name
        results_path = Path("results/figures") / name
        assert manuscript_path.exists()
        assert results_path.exists()
        assert manuscript_path.stat().st_size > 10_000
        assert results_path.stat().st_size > 10_000



def test_protocol_soc_validation_outputs_exist():
    validation = read_required_csv("data_processed/qc/protocol_soc_validation.csv")
    summary = read_required_csv("results/tables/table_protocol_soc_validation_summary.csv")
    assert len(validation) == 420
    assert len(summary) == 40
    assert set(validation["sweep_name"]) == {"charge_sweep", "discharge_sweep"}
    assert validation["capacity_integrated_soc_percent"].notna().all()
    assert summary["mean_soc_abs_error_percent"].mean() < 2.0


def test_capacity_ocv_publication_validation_outputs_exist():
    validation = read_required_csv("results/tables/table_capacity_ocv_publication_validation.csv")
    report = Path("results/reports/capacity_ocv_publication_validation.md")
    assert len(validation) == 8
    assert set(validation["validation_group"]) == {"temperature_capacity", "ocv_current_rate"}
    assert validation["validation_status"].eq("pass").all()
    assert validation["actual_percent"].notna().all()
    assert report.exists()
    assert report.stat().st_size > 100


def test_ocv_slope_outputs_exist():
    descriptors = read_required_csv("data_processed/ocv/ocv_slope_descriptors.csv")
    summary = read_required_csv("results/tables/table_ocv_slope_summary.csv")
    report = Path("results/reports/ocv_slope_analysis.md")
    figure = Path("results/figures/fig12_ocv_slope_penalty.png")
    assert len(descriptors) == 480
    assert len(summary) == 40
    assert set(descriptors["smoothing_window_segments"]) == {1, 5, 11}
    assert set(descriptors["capacity_window"]) == {"full", "early_0_20", "central_20_80", "late_80_100"}
    assert summary["ocv_slope_penalty_v"].notna().all()
    assert report.exists()
    assert figure.exists()
    assert figure.stat().st_size > 10_000



def _require_manuscript() -> None:
    """Skip when the manuscript sources are not part of this distribution.

    The public repository ships the analysis, not the papers, so the three
    tests that read the LaTeX report as skipped rather than as failures. Every
    other test runs against the released tables and must still pass.
    """
    if not Path("manuscript/latex").is_dir():
        pytest.skip("manuscript sources are not distributed with this repository")


def test_final_manuscript_draft_and_reproducibility_outputs_exist():
    _require_manuscript()
    required = [
        "manuscript/submission/cover_letter_skeleton.md",
        "manuscript/submission/highlights.md",
        "results/reports/final_reproducibility_check.md",
    ]
    for path in required:
        file_path = Path(path)
        assert file_path.exists()
        assert file_path.stat().st_size > 100
    checks = read_required_csv("results/tables/table_final_reproducibility_check.csv")
    # Floor, not an exact count: the audit covers the distributed file set, and a
    # hardcoded total is what let it drift while guarding files that ship nowhere.
    assert len(checks) >= 50
    assert checks["status"].eq("pass").all()


def test_every_quoted_manuscript_number_matches_its_source_table():
    """Guard against the failure mode that produced both errors found in review:
    a number that drifted away from the table the manuscript cites for it."""
    import subprocess
    import sys

    script = Path("scripts/check_manuscript_numbers.py")
    if not script.exists():
        pytest.skip("Number-verification script is not present")
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_results_table_cells_sit_under_the_right_family_column():
    _require_manuscript()
    """check_manuscript_numbers.py re-derives the results table's values from the
    CSVs and never reads the LaTeX, so it cannot see a column transposition:
    swapping two family columns leaves every value present and correct but
    attached to the wrong cell. This parses the table itself and checks each
    cell against the family it claims to describe. It also checks that every
    family-ordered supplementary table uses the same SIB, LFP, NMC, LTO order
    as the main text, which five of them did not until round 25."""
    import subprocess
    import sys

    script = Path("scripts/check_results_table_layout.py")
    if not script.exists():
        pytest.skip("Results-table layout script is not present")
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 mismatched" in result.stdout, result.stdout
    assert "rows not covered" not in result.stdout, result.stdout
    assert "0 out of order" in result.stdout, result.stdout


def test_no_cross_reference_between_main_and_supplement_is_broken():
    _require_manuscript()
    """The two documents compile separately, so a pointer from one into the
    other is a hardcoded string that LaTeX cannot check. Round 9 inserted
    three equations and a subsection and left four SI pointers stale for two
    rounds while both documents compiled cleanly.

    Every pointer now carries an `% xref: <label>` binding, so a stale number
    fails here instead of resolving quietly to whichever float happens to
    print as it. Round 22 renumbered two equations and two SI captions went
    stale under the old existence-only check, which passed them because 6 and
    8 were still real equation numbers.
    """
    import shutil
    import subprocess
    import sys

    script = Path("scripts/check_manuscript_crossrefs.py")
    if not script.exists():
        pytest.skip("Cross-reference script is not present")

    # This test used to guard on manuscript/latex/main.aux and
    # supplementary.aux, which the checker never reads -- it reads
    # Manuscript.aux, SM.aux and Manuscript.bbl from the JES directory. Those
    # two names never appeared there, so the guard was always true and the
    # check never ran. It also asserted "0 resolved pointer(s) to eyeball",
    # which the checker never prints: resolved pointers are the ones that are
    # fine. Both mistakes are only visible when the test actually executes, so
    # it now builds its own inputs and skips only when LaTeX is absent.
    if not (shutil.which("pdflatex") and shutil.which("bibtex")):
        pytest.skip("pdflatex/bibtex are needed to produce the .aux and .bbl inputs")

    latex = Path("manuscript/latex/JES")
    jobs = ("Manuscript", "SM")
    produced = [latex / f"{job}{ext}" for job in jobs
                for ext in (".aux", ".bbl", ".blg", ".log", ".out", ".toc", ".spl")]
    pre_existing = {path for path in produced if path.exists()}

    def latex_pass(job: str) -> None:
        # -draftmode skips graphics entirely: the labels and citations land in
        # the .aux just the same, and the run does not depend on the figures.
        subprocess.run(["pdflatex", "-draftmode", "-interaction=nonstopmode", job],
                       cwd=latex, capture_output=True, text=True)

    try:
        for job in jobs:
            latex_pass(job)
        subprocess.run(["bibtex", "Manuscript"], cwd=latex, capture_output=True, text=True)
        for job in jobs:
            latex_pass(job)

        for name in ("Manuscript.aux", "SM.aux", "Manuscript.bbl"):
            assert (latex / name).exists(), f"{name} was not produced in {latex}"

        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASS: 0 broken pointer(s)" in result.stdout, result.stdout
        # A scan that matches nothing would otherwise exit 0 with no findings.
        examined = re.search(r"(\d+) pointer\(s\) examined", result.stdout)
        assert examined, result.stdout
        assert int(examined.group(1)) > 0, result.stdout
    finally:
        for path in produced:
            if path.exists() and path not in pre_existing:
                path.unlink()


def test_replicate_files_are_assigned_to_a_condition_rather_than_assumed():
    """Section 3.5 calls the replicate spread a room-temperature bound. The
    files carry no temperature label, so the assignment is recovered from the
    resistance itself and holds only for the two families whose R_1s changes
    sharply between setpoints.
    """
    import subprocess
    import sys

    script = Path("scripts/check_replicate_temperature_assignment.py")
    if not script.exists():
        pytest.skip("Replicate temperature-assignment script is not present")
    result = subprocess.run(
        [sys.executable, str(script), "--quiet"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr

    table = pd.read_csv("results/tables/table_replicate_temperature_assignment.csv")
    assigned = set(table.loc[table["assignment"] == "25 C", "cell_family"])
    assert assigned == {"SIB", "NMC"}, assigned
    # The claim in the text is that the alternative setpoints are far outside
    # the replicate range for those two, not merely further away.
    for family in assigned:
        row = table[table["cell_family"] == family].iloc[0]
        assert row["gap_to_closest_alternative_percent"] > 15.0, row.to_dict()


def test_quadrature_choice_does_not_move_the_three_term_split():
    """L_pol is built from two sums that do not share a quadrature, so the
    published right-endpoint rule understates it. The reported shares must
    survive the full bracket, and must not move towards dissipation.
    """
    import subprocess
    import sys

    script = Path("scripts/check_ocv_quadrature_sensitivity.py")
    if not script.exists():
        pytest.skip("Quadrature sensitivity script is not present")
    result = subprocess.run(
        [sys.executable, str(script), "--quiet"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr

    table = pd.read_csv("results/tables/table_ocv_quadrature_sensitivity.csv")
    assert abs(table["identity_closure_wh"]).max() < 1e-9

    for family, block in table.groupby("cell_family"):
        span = (
            block["truncation_share_percent"].max()
            - block["truncation_share_percent"].min()
        )
        assert span <= 2.0, (family, span)

    sib = table[table["cell_family"] == "SIB"].set_index("quadrature")
    # The shorter-window term exceeds the polarisation-associated share even
    # polarisation, under every rule.
    assert (sib["dissipation_upper_share_percent"] < 50.0).all()
    # And the published rule is the least favourable of the three.
    assert (
        sib.loc["right", "dissipation_upper_share_percent"]
        == sib["dissipation_upper_share_percent"].max()
    )
