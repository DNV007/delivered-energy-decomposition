# Data Inventory Template

Use this template during Step 1. Convert it to CSV later if needed.

| Raw path | File type | Cell ID | Chemistry/type | Temperature deg C | SOC | Measurement stream | Parsed? | Notes |
|---|---|---|---|---:|---:|---|---|---|
|  |  |  |  |  |  | capacity / HPPC / OCV / EIS / entropy / thermal / metadata / unknown | no |  |

## Classification Notes

- Keep raw files untouched in `data_raw/`.
- Put cleaned tables in the matching `data_processed/<measurement_type>/` folder.
- If metadata are unclear, keep the file classified as `unknown` until resolved.
- Record assumptions explicitly in the `Notes` column.
