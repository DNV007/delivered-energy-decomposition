# Step 1 Status

## Current State

- Project folder structure is initialized.
- Dataset source metadata is recorded in `configs/dataset_sources.json`.
- Download script is available at `scripts/fetch_depositonce_dataset.py`.
- Raw data inventory script is available at `scripts/inventory_raw_data.py`.
- Raw schema profiling script is available at `scripts/profile_raw_tables.py`.

## Next Command Sequence

1. Download `Readme.md` and `data_EvalSIB.zip` into `data_raw/`.
2. Run `scripts/inventory_raw_data.py`.
3. Run `scripts/profile_raw_tables.py`.
4. Review `logs/data_inventory_report.md` and `logs/raw_table_schema_report.md`.
5. Refine classification rules if too many files remain `unknown`.
