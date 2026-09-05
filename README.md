# Three-term energy accounting for a cold commercial sodium-ion 18650 cell

This repository holds the analysis code and the derived data behind a reanalysis
of a public commercial-cell archive. No new measurements were made. The
manuscript reporting the work is not distributed here; what is here is
everything needed to regenerate its numbers from the raw archive and to check
them.

**Source data.** Droese, D., Schlösser, A., Otto, M. *Dataset: Evaluation of
Commercial Sodium-Ion Batteries by State-of-the-Art Lithium-Ion Battery
Configurations.* Technische Universität Berlin, DepositOnce, 2025.
DOI [10.14279/depositonce-25036](https://doi.org/10.14279/depositonce-25036),
CC BY 4.0. Associated paper:
[10.3390/batteries11110420](https://doi.org/10.3390/batteries11110420).

## What the analysis finds

The archive's checkup discharge is a sequence of up to 105 constant-current
segments at 1.5 A, each followed by a six-minute rest. Loaded and rested
voltages therefore sit on one coulomb-counted charge axis, for the same cell, in
the same file. That geometry is what makes the decomposition possible.

Cooling the sodium-ion cell from 25 to 5 °C costs 25.1 % of its delivered
energy, against 12.2 % for LFP, 6.9 % for NMC and 2.4 % for LTO. For the
sodium-ion cell that shortfall splits three ways:

| Term | Share |
|---|---|
| Charge the cold discharge never reaches, valued on the 25 °C rested curve | 58 % |
| Growth of the integrated load-to-rest gap | 27 % |
| Lower rested voltage over the charge window common to both temperatures | 15 % |

Capacity retention (0.816) and rested-voltage energy retention (0.825) differ by
0.009 at 5 °C, while the rested curve runs a median 131 mV lower at matched
discharged throughput. Two retention numbers that agree can sit above voltage
curves that do not.

Three limits are worth stating up front. The load-to-rest gap is an electrical
proxy for polarisation energy, not a calorimetric measurement. The charge ledger
counts charge passed, not charge stored, so the difference in starting state
between the two discharges is not bounded by anything in this archive. And with
one cell per family, the shares describe these cells under this protocol rather
than the chemistries as classes.

## What is here, and what is not

Here: the pipeline (`scripts/`, `src/`), the descriptor tables it extracts
(`data_processed/`), the result tables and figures it derives (`results/`), the
regression suite (`tests/`), and the deposit metadata.

Not here, each for its own reason:

- **The raw archive** (213 MB). It belongs to TU Berlin and is already published
  under its own DOI. `scripts/fetch_depositonce_dataset.py` downloads it and
  records a SHA-256 per file in `data_raw/download_manifest.json`.
- **Two third-party archives** used for external cross-checks: a multi-rate
  pulse dataset and a relaxation benchmark. Not ours to redistribute. The
  scripts that read them are here; the data are not.
- **The manuscript and its Supporting Information.** They are the journal's to
  publish.
- **The figures.** The artwork belongs with the paper, so it is held back until
  publication. `reproduce_all.py` regenerates it from the released tables, into
  `results/figures/` and `manuscript/figures/`. The one exception is a figure
  drawn from the third-party pulse dataset above, which cannot be rebuilt
  without that archive.

## Reproducing it

### 1. Environment

```bash
uv venv .venv --python 3.11
VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt
```

Use this environment rather than a system Anaconda installation, whose
`pyarrow` and `numexpr` are built against NumPy 1.x and fail to import under the
NumPy 2.x this pipeline needs. `uv venv` ships no `pip`, so install with
`uv pip install` as above.

### 2. Raw data

```bash
.venv/bin/python scripts/fetch_depositonce_dataset.py
```

Nothing in the pipeline writes to `data_raw/`.

### 3. Run the pipeline

```bash
.venv/bin/python scripts/reproduce_all.py
```

One command, 39 stages, a few minutes. It rebuilds every descriptor table,
result table and figure from the raw archive in dependency order, then runs the
verification stages. `scripts/reproduce_all.py` records why each script sits
where it does, and refuses to start if the stage list and `scripts/` disagree in
either direction: a script that no stage runs, or a stage naming a script that
is not there.

Three verification stages read the LaTeX sources. Those are not distributed
here, so the run reports them as skipped and continues. The figure stages write
their output twice, to `results/figures/` and to `manuscript/figures/`; the
second is where the paper's build picks them up, and the run creates it.

Re-running against the released outputs reproduces every table byte for byte.
Three kinds of file legitimately differ between runs: those carrying a run
timestamp, figure PDFs (matplotlib embeds a creation date), and two intermediate
descriptor files that drift in the last floating-point digit because summation
order is not fixed across BLAS builds. No reported value is affected, which is
what `check_manuscript_numbers.py` establishes.

### 4. Verify

```bash
.venv/bin/python -m pytest tests/                        # 29 tests, 3 skipped here
.venv/bin/python scripts/check_manuscript_numbers.py     # 427 reported values
```

Three of the 29 tests read the LaTeX sources and report as skipped in this
distribution. The other 26 run against the released tables.

`check_manuscript_numbers.py` is the one to run after regenerating anything. It
re-derives every value the manuscript reports from the released tables, keyed by
cell and temperature, and fails on a missing table, an ambiguous lookup, or any
drift. It reads only CSVs, so it runs here without the manuscript.

## Layout

```text
data_raw/           Source archive (fetched, not redistributed)
data_processed/     Descriptor tables extracted from the archive
scripts/            The pipeline; reproduce_all.py runs it in order
src/                Shared parsing and metadata helpers
results/tables/     Derived result tables; the reported numbers come from here
results/figures/    Figures, written here by the pipeline (not shipped yet)
results/reports/    Audit output
results/internal/   Notes on two analyses that were run and left out of the paper
tests/              Regression suite
docs/               Descriptor definitions and the data inventory
logs/               Extraction QC reports
configs/            Cell and protocol metadata
notebooks/          Exploratory only; nothing depends on it
```

## Provenance

Every reported quantity traces to the script that computes it and the file it is
written to. `results/reports/final_reproducibility_check.md` is the audit
output; `check_manuscript_numbers.py` ties reported values to released tables.
Both are regenerated by the pipeline.

## Licence and citation

- Code (`scripts/`, `src/`, `tests/`, `notebooks/`) — MIT, see [`LICENSE`](LICENSE)
- Derived data, figures and tables — CC BY 4.0, see [`LICENSE-DATA`](LICENSE-DATA)
- Raw data — CC BY 4.0 from TU Berlin, under its own DOI above

Citation metadata is in [`CITATION.cff`](CITATION.cff), which carries the
archived version and its DOI once a release is deposited. Please cite both this
deposit and the source dataset it reanalyses.
