# Raw Table Schema Report

Created: 2026-07-30T09:17:15.671247+00:00
Profiled files: 152

## File Types

- .csv: 104
- .npy: 8
- .parquet: 40

## Top-Level Folders

- data_Ri_HPPC: 24
- data_dUdT: 8
- raw_CU_diffT: 20
- raw_EIS_plotting: 80
- raw_OCV_I_curves: 4
- raw_temperature_BOL: 16

## Representative Schemas

### data_Ri_HPPC/TC23LFP01_HPPC_ch_00.csv
- Columns: `    SOC`, ` C-Rate`, `  R_fast`, ` dt_fast`, ` R_100ms`, ` dt_100ms`, `    R_1s`, `   dt_1s`, ` C-Rate.1`, `  R_fast.1`, ` dt_fast.1`, ` R_100ms.1`, ` dt_100ms.1`, `    R_1s.1`, `   dt_1s.1`, ` C-Rate.2`, `  R_fast.2`, ` dt_fast.2`, ` R_100ms.2`, ` dt_100ms.2`, `    R_1s.2`, `   dt_1s.2`, ` C-Rate.3`, `  R_fast.3`, ` dt_fast.3`, ` R_100ms.3`, ` dt_100ms.3`, `    R_1s.3`, `   dt_1s.3`, ` C-Rate.4`, `  R_fast.4`, ` dt_fast.4`, ` R_100ms.4`, ` dt_100ms.4`, `    R_1s.4`, `   dt_1s.4`, ` C-Rate.5`, `  R_fast.5`, ` dt_fast.5`, ` R_100ms.5`, ` dt_100ms.5`, `    R_1s.5`, `   dt_1s.5`
- Preview:

```json
[
  {
    "    SOC": "0.0",
    " C-Rate": "     1C",
    "  R_fast": "0.0496",
    " dt_fast": "0.0162",
    " R_100ms": "0.05604",
    " dt_100ms": "0.10594",
    "    R_1s": "0.06893",
    "   dt_1s": "1.00605",
    " C-Rate.1": "     2C",
    "  R_fast.1": "0.04941",
    " dt_fast.1": "0.01582",
    " R_100ms.1": "0.05568",
    " dt_100ms.1": "0.10608",
    "    R_1s.1": "0.06709",
    "   dt_1s.1": "1.00609",
    " C-Rate.2": "     3C",
    "  R_fast.2": "0.04916",
    " dt_fast.2": "0.01585",
    " R_100ms.2": "0.05515",
    " dt_100ms.2": "0.10609",
    "    R_1s.2": "0.06508",
    "   dt_1s.2": "1.00638",
    " C-Rate.3": "    -1C",
    "  R_fast.3": "0.0491",
    " dt_fast.3": "0.0161",
    " R_100ms.3": "0.05499",
    " dt_100ms.3": "0.10581",
    "    R_1s.3": "0.06868",
    "   dt_1s.3": "1.00605",
    " C-Rate.4": "    -2C",
    "  R_fast.4": "0.04833",
    " dt_fast.4": "0.01571",
    " R_100ms.4": "0.05397",
    " dt_100ms.4": "0.10613",
    "    R_1s.4": "0.06703",
    "   dt_1s.4": "1.00635",
    " C-Rate.5": "    -3C",
    "  R_fast.5": "0.0479",
    " dt_fast.5": "0.01604",
    " R_100ms.5": "0.053",
    " dt_100ms.5": "0.10549",
    "    R_1s.5": "0.0652",
    "   dt_1s.5": "1.00556"
  },
  {
    "    SOC": "0.10045",
    " C-Rate": "     1C",
    "  R_fast": "0.04477",
    " dt_fast": "0.02211",
    " R_100ms": "0.04732",
    " dt_100ms": "0.10799",
    "    R_1s": "0.05418",
    "   dt_1s": "1.01103",
    " C-Rate.1": "     2C",
    "  R_fast.1": "0.04451",
    " dt_fast.1": "0.02227",
    " R_100ms.1": "0.04706",
    " dt_100ms.1": "0.11074",
    "    R_1s.1": "0.05316",
    "   dt_1s.1": "1.00465",
    " C-Rate.2": "     3C",
    "  R_fast.2": "0.04434",
    " dt_fast.2": "0.02228",
    " R_100ms.2": "0.0468",
    " dt_100ms.2": "0.11072",
    "    R_1s.2": "0.05257",
    "   dt_1s.2": "1.0068",
    " C-Rate.3": "    -1C",
    "  R_fast.3": "0.04503",
    " dt_fast.3": "0.02213",
    " R_100ms.3": "0.04744",
    " dt_100ms.3": "0.10821",
    
```

### data_dUdT/a_dUdT_discharge_LFP.npy
- Shape: [11]; dtype: `float64`
- Preview:

```json
[
  0.00012449973693875704,
  3.2204960946500566e-05
]
```

### raw_CU_diffT/TC23LFP09/TC23LFP09_05deg.parquet
- Rows: 1254310
- Columns: `Absolute Time[yyyy-mm-dd hh:mm:ss]`, `Testtime[s]`, `StepID`, `Steptime[s]`, `Voltage[V]`, `Current[A]`, `Temperature[°C]`, `Aux_U[V]`, `Aux_I[A]`, `Aux_T[°C]`, `EISFreq[Hz]`, `Zabs[Ohm]`, `Phase[°]`, `Zre[Ohm]`, `Zim[Ohm]`, `DC_Current[A]`, `Capacity_Step[Ah]`
- Preview:

```json
[
  {
    "Absolute Time[yyyy-mm-dd hh:mm:ss]": NaN,
    "Testtime[s]": "0.0",
    "StepID": "4",
    "Steptime[s]": NaN,
    "Voltage[V]": "3.22466797001274",
    "Current[A]": "0.0",
    "Temperature[\u00b0C]": "5.52631",
    "Aux_U[V]": NaN,
    "Aux_I[A]": NaN,
    "Aux_T[\u00b0C]": NaN,
    "EISFreq[Hz]": NaN,
    "Zabs[Ohm]": NaN,
    "Phase[\u00b0]": NaN,
    "Zre[Ohm]": NaN,
    "Zim[Ohm]": NaN,
    "DC_Current[A]": NaN,
    "Capacity_Step[Ah]": "0.0"
  },
  {
    "Absolute Time[yyyy-mm-dd hh:mm:ss]": NaN,
    "Testtime[s]": "0.005771999999999988",
    "StepID": "4",
    "Steptime[s]": NaN,
    "Voltage[V]": "3.22447724025761",
    "Current[A]": "0.0",
    "Temperature[\u00b0C]": "5.52631",
    "Aux_U[V]": NaN,
    "Aux_I[A]": NaN,
    "Aux_T[\u00b0C]": NaN,
    "EISFreq[Hz]": NaN,
    "Zabs[Ohm]": NaN,
    "Phase[\u00b0]": NaN,
    "Zre[Ohm]": NaN,
    "Zim[Ohm]": NaN,
    "DC_Current[A]": NaN,
    "Capacity_Step[Ah]": "0.0"
  }
]
```

### raw_EIS_plotting/TC23LFP04/auxiliary/20231113_TC23LFP04_3187mV_25gradC_EIS.csv
- Columns: `Absolute Time[yyyy-mm-dd hh:mm:ss]`, `Testtime [s]`, `StepID`, `Steptime[s]`, `Voltage[V]`, `Current[A]`, `Temperature[°C]`, `Aux_U[V]`, `Aux_I[A]`, `Aux_T[°C]`, `EISFreq[Hz]`, `Zabs[Ohm]`, `Phase[°]`, `Zre[Ohm]`, `Zim[Ohm]`, `DC_Current[A]`
- Preview:

```json
[
  {
    "Absolute Time[yyyy-mm-dd hh:mm:ss]": "2023-11-13 12:46:36",
    "Testtime [s]": NaN,
    "StepID": NaN,
    "Steptime[s]": NaN,
    "Voltage[V]": "3.18778",
    "Current[A]": NaN,
    "Temperature[\u00b0C]": "25.2672",
    "Aux_U[V]": NaN,
    "Aux_I[A]": NaN,
    "Aux_T[\u00b0C]": NaN,
    "EISFreq[Hz]": "0.05",
    "Zabs[Ohm]": "0.0688854070562118",
    "Phase[\u00b0]": "1.8176688597015864",
    "Zre[Ohm]": "0.0667969",
    "Zim[Ohm]": "-0.0168337",
    "DC_Current[A]": "-0.00212509"
  },
  {
    "Absolute Time[yyyy-mm-dd hh:mm:ss]": "2023-11-13 12:46:36",
    "Testtime [s]": NaN,
    "StepID": NaN,
    "Steptime[s]": NaN,
    "Voltage[V]": "3.18778",
    "Current[A]": NaN,
    "Temperature[\u00b0C]": "25.2672",
    "Aux_U[V]": NaN,
    "Aux_I[A]": NaN,
    "Aux_T[\u00b0C]": NaN,
    "EISFreq[Hz]": "0.1",
    "Zabs[Ohm]": "0.0698128336841737",
    "Phase[\u00b0]": "1.6955698705121425",
    "Zre[Ohm]": "0.0692701",
    "Zim[Ohm]": "-0.00868821",
    "DC_Current[A]": "-0.00212509"
  }
]
```

### raw_OCV_I_curves/TC23LFP12/TC23LFP12_OCVvgl.parquet
- Rows: 63134
- Columns: `Absolute Time[yyyy-mm-dd hh:mm:ss]`, `Testtime[s]`, `StepID`, `Steptime[s]`, `Voltage[V]`, `Current[A]`, `Temperature[°C]`, `Aux_U[V]`, `Aux_I[A]`, `Aux_T[°C]`, `EISFreq[Hz]`, `Zabs[Ohm]`, `Phase[°]`, `Zre[Ohm]`, `Zim[Ohm]`, `DC_Current[A]`, `Capacity_Step[Ah]`
- Preview:

```json
[
  {
    "Absolute Time[yyyy-mm-dd hh:mm:ss]": NaN,
    "Testtime[s]": "0.0",
    "StepID": "2",
    "Steptime[s]": NaN,
    "Voltage[V]": "3.30057841255519",
    "Current[A]": "0.0",
    "Temperature[\u00b0C]": "25.72388",
    "Aux_U[V]": NaN,
    "Aux_I[A]": NaN,
    "Aux_T[\u00b0C]": NaN,
    "EISFreq[Hz]": NaN,
    "Zabs[Ohm]": NaN,
    "Phase[\u00b0]": NaN,
    "Zre[Ohm]": NaN,
    "Zim[Ohm]": NaN,
    "DC_Current[A]": NaN,
    "Capacity_Step[Ah]": "0.0"
  },
  {
    "Absolute Time[yyyy-mm-dd hh:mm:ss]": NaN,
    "Testtime[s]": "0.007688000000000017",
    "StepID": "2",
    "Steptime[s]": NaN,
    "Voltage[V]": "3.30057841255519",
    "Current[A]": "0.0",
    "Temperature[\u00b0C]": "25.72388",
    "Aux_U[V]": NaN,
    "Aux_I[A]": NaN,
    "Aux_T[\u00b0C]": NaN,
    "EISFreq[Hz]": NaN,
    "Zabs[Ohm]": NaN,
    "Phase[\u00b0]": NaN,
    "Zre[Ohm]": NaN,
    "Zim[Ohm]": NaN,
    "DC_Current[A]": NaN,
    "Capacity_Step[Ah]": "0.0"
  }
]
```

### raw_temperature_BOL/TC23LFP01/TC23LFP01_dS025C.parquet
- Rows: 82353
- Columns: `Absolute Time[yyyy-mm-dd hh:mm:ss]`, `Testtime[s]`, `StepID`, `Steptime[s]`, `Voltage[V]`, `Current[A]`, `Temperature[°C]`, `Aux_U[V]`, `Aux_I[A]`, `Aux_T[°C]`, `EISFreq[Hz]`, `Zabs[Ohm]`, `Phase[°]`, `Zre[Ohm]`, `Zim[Ohm]`, `DC_Current[A]`
- Preview:

```json
[
  {
    "Absolute Time[yyyy-mm-dd hh:mm:ss]": NaN,
    "Testtime[s]": "0.0",
    "StepID": "2",
    "Steptime[s]": NaN,
    "Voltage[V]": "3.31195285821821",
    "Current[A]": "0.0",
    "Temperature[\u00b0C]": "26.63713",
    "Aux_U[V]": NaN,
    "Aux_I[A]": NaN,
    "Aux_T[\u00b0C]": NaN,
    "EISFreq[Hz]": NaN,
    "Zabs[Ohm]": NaN,
    "Phase[\u00b0]": NaN,
    "Zre[Ohm]": NaN,
    "Zim[Ohm]": NaN,
    "DC_Current[A]": NaN
  },
  {
    "Absolute Time[yyyy-mm-dd hh:mm:ss]": NaN,
    "Testtime[s]": "0.002005000000000002",
    "StepID": "2",
    "Steptime[s]": NaN,
    "Voltage[V]": "3.31195285821821",
    "Current[A]": "0.0",
    "Temperature[\u00b0C]": "26.63713",
    "Aux_U[V]": NaN,
    "Aux_I[A]": NaN,
    "Aux_T[\u00b0C]": NaN,
    "EISFreq[Hz]": NaN,
    "Zabs[Ohm]": NaN,
    "Phase[\u00b0]": NaN,
    "Zre[Ohm]": NaN,
    "Zim[Ohm]": NaN,
    "DC_Current[A]": NaN
  }
]
```
