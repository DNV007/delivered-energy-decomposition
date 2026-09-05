# Final Reproducibility Check

Decision: pass.
Checks passed: 62/62.

This check verifies the presence and basic integrity of the current processed data, manuscript figures, validation tables, draft text, and submission skeleton. It does not redownload the raw dataset.

## Check Details

- pass: file_exists | data_raw/Readme.md | size=1295
- pass: file_exists | data_raw/data_EvalSIB.zip | size=222415780
- pass: file_exists | data_raw/download_manifest.json | size=1132
- pass: file_exists | docs/data_inventory.csv | size=31884
- pass: file_exists | data_processed/descriptors/integrated_descriptor_long.csv | size=3966389
- pass: file_exists | data_processed/descriptors/family_temperature_descriptor_matrix.csv | size=16807
- pass: file_exists | data_processed/descriptors/family_temperature_targets.csv | size=3630
- pass: file_exists | data_processed/ocv/ocv_slope_descriptors.csv | size=108080
- pass: file_exists | data_processed/qc/protocol_soc_validation.csv | size=103335
- pass: file_exists | results/tables/table_capacity_ocv_publication_validation.csv | size=2030
- pass: file_exists | results/tables/table_ocv_slope_summary.csv | size=9841
- pass: file_exists | results/tables/table_protocol_soc_validation_summary.csv | size=3576
- pass: file_exists | results/tables/table_eis_ecm_fit_reliability_summary.csv | size=2548
- pass: file_exists | results/tables/table_central_claim_evidence.csv | size=1219
- pass: file_exists | results/tables/table_arrhenius_activation_energy.csv | size=33278
- pass: file_exists | results/tables/table_arrhenius_vtf_comparison.csv | size=25631
- pass: file_exists | results/tables/table_eis_kramers_kronig.csv | size=12100
- pass: file_exists | results/tables/table_eis_kramers_kronig_summary.csv | size=592
- pass: file_exists | results/tables/table_eis_subset_sensitivity.csv | size=417
- pass: file_exists | results/tables/table_cell_replicate_spread.csv | size=868
- pass: file_exists | results/tables/table_delivered_energy_decomposition.csv | size=5113
- pass: file_exists | results/tables/table_energy_shortfall_split.csv | size=5951
- pass: file_exists | results/tables/table_common_window_ocv_comparison.csv | size=5017
- pass: file_exists | results/tables/table_pulse_rate_dependence.csv | size=710
- pass: file_exists | results/tables/table_ordering_robustness.csv | size=1092
- pass: file_exists | results/tables/table_cutoff_normalisation.csv | size=6846
- pass: file_exists | results/tables/table_relaxation_geometry.csv | size=742
- pass: file_exists | results/reports/ocv_slope_analysis.md | size=612
- pass: file_exists | manuscript/submission/cover_letter_skeleton.md | size=2839
- pass: file_exists | manuscript/latex/Manuscript_ACS_Energy_Letters.tex | size=21105
- pass: file_exists | manuscript/latex/SM_ACS_Energy_Letters.tex | size=68854
- pass: file_exists | manuscript/latex/JES/Manuscript.tex | size=99157
- pass: file_exists | manuscript/latex/JES/SM.tex | size=64877
- pass: file_exists | manuscript/figures/figure1_data_architecture.png | size=232320
- pass: file_exists | manuscript/figures/figure2_energy_decomposition.png | size=509470
- pass: file_exists | manuscript/figures/figure_decomposition_trace.png | size=572405
- pass: file_exists | manuscript/figures/figure_decomposition_trace_acs.png | size=564085
- pass: file_exists | manuscript/figures/figure2_energy_decomposition_acs.png | size=513899
- pass: file_exists | manuscript/figures/figure4_raw_pulse_bottleneck.png | size=294410
- pass: file_exists | manuscript/figures/figure5_eis_bottleneck_decomposition.png | size=228642
- pass: file_exists | manuscript/figures/figure6_ocv_entropy_thermal_coupling.png | size=256302
- pass: file_exists | manuscript/figures/graphical_abstract.png | size=221893
- pass: file_exists | results/figures/fig12_ocv_slope_penalty.png | size=106283
- pass: csv_rows | data_processed/descriptors/integrated_descriptor_long.csv | rows=14463; expected_min=14000
- pass: csv_rows | data_processed/descriptors/family_temperature_descriptor_matrix.csv | rows=20; expected_min=20
- pass: csv_rows | data_processed/descriptors/family_temperature_targets.csv | rows=20; expected_min=20
- pass: csv_rows | data_processed/ocv/ocv_slope_descriptors.csv | rows=480; expected_min=480
- pass: csv_rows | data_processed/qc/protocol_soc_validation.csv | rows=420; expected_min=420
- pass: csv_rows | results/tables/table_capacity_ocv_publication_validation.csv | rows=8; expected_min=8
- pass: csv_rows | results/tables/table_ocv_slope_summary.csv | rows=40; expected_min=40
- pass: csv_rows | results/tables/table_central_claim_evidence.csv | rows=6; expected_min=6
- pass: csv_rows | results/tables/table_arrhenius_activation_energy.csv | rows=120; expected_min=120
- pass: csv_rows | results/tables/table_arrhenius_vtf_comparison.csv | rows=120; expected_min=120
- pass: csv_rows | results/tables/table_eis_kramers_kronig.csv | rows=40; expected_min=40
- pass: csv_rows | results/tables/table_eis_subset_sensitivity.csv | rows=4; expected_min=4
- pass: csv_rows | results/tables/table_eis_ecm_fit_reliability.csv | rows=160; expected_min=160
- pass: csv_rows | results/tables/table_eis_ecm_fit_reliability_summary.csv | rows=16; expected_min=16
- pass: csv_rows | results/tables/table_cutoff_normalisation.csv | rows=40; expected_min=40
- pass: status_column | results/tables/table_capacity_ocv_publication_validation.csv | validation_status all pass=True
- pass: status_column | results/tables/table_central_claim_evidence.csv | support_status all supports=True
- pass: ecm_rejection | results/tables/table_eis_ecm_fit_reliability_summary.csv | pass_count_sum=0
- pass: protocol_soc_mean_error | results/tables/table_protocol_soc_validation_summary.csv | mean_error_percent=1.633

## Separate Test Command

The Python regression suite should also pass with `python3 -m pytest -q`.
