# Internal record — multi-rate low-temperature data assessed, not used in the manuscript

**Date:** 2 September 2026
**Status:** examined, negative result, deliberately excluded from the manuscript.
**Purpose of this note:** if a referee asks whether available multi-rate low-temperature
discharge data were examined in support of the shorter-window interpretation, the answer
is yes, and this is what was found.

---

## 1. Why it was examined

The three-term decomposition attributes 58 % of the SIB's 25→5 °C delivered-energy
shortfall to a shorter delivered-charge window — charge the cold cell never reaches
because the loaded voltage arrives at the cutoff first. That reading implies a testable
rate dependence: lowering the current lowers the polarisation that carries the voltage to
the cutoff, so the cold and warm windows should converge and the cold shortfall should
shrink as rate falls.

The DepositOnce archive cannot test this — it discharges at one current only. Two public
datasets were assessed as external tests.

## 2. Datasets screened

| Dataset | Verdict | Reason |
|---|---|---|
| **Shandong SIB** (Zenodo 13836819; Wang et al., *J. Energy Storage* 2024) | **Rejected at screening** | Pulse protocol is 5 s pulse / **15 s rest**, SOC in **10 % steps**. No incremental discharge with per-increment relaxation, so no rested-voltage counterfactual on the discharge trajectory. Useful only for coarse retention/resistance context across two SIB products (−5 to 45 °C), not for the decomposition. |
| **Stanford/SLAC Molicel P42A** (OSF 10.17605/OSF.IO/9CEAV; Khan et al., *Sci. Data* **12**, 1506, 2025) | **Analysed in full, then excluded** | Structurally suitable for a rate-dependence test — CDT at C/3, C/10, C/20 at 5/25/40 °C, 12 cells, all discharges terminated on a common 2.5 V cutoff. Results below. |

Note the Stanford CDT cannot reproduce the three-term identity either: it is a continuous
discharge with no per-increment rest, so the rested-voltage counterfactual does not exist
on that trajectory. It was used only as a test of the predicted rate dependence.

## 3. What was found (8 cells paired across 5 and 25 °C)

**Primary test — retention against rate:**

| | C/3 | C/10 | C/20 | paired C/20−C/3 |
|---|---|---|---|---|
| charge retention *Q*(5)/*Q*(25) | 0.9713 | 0.9754 | 0.9753 | +0.0040, p = 0.13 (n.s.) |
| energy retention | 0.9667 | 0.9744 | 0.9755 | +0.0088, p = 0.015 |

**Rate × temperature interaction** (within-cell slope of log y vs C-rate, per temperature):

| | 5 °C slope | 25 °C slope | interaction |
|---|---|---|---|
| charge | −0.0320 (p = 0.0001) | −0.0161 (p = 0.053) | −0.0159, **p = 0.09 (n.s.)**, 6/8 cells |
| energy | −0.0753 (p < 0.0001) | −0.0424 (p = 0.0004) | −0.0329, **p = 0.012**, 7/8 cells |

Holm-corrected pairwise on retention: energy C/3→C/10 p = 0.018, C/3→C/20 p = 0.031,
C/10→C/20 n.s.; charge — nothing significant.

**A correlation that was found and then discarded.** An initial pass correlated the cold
penalty x = 1 − R(C/3) against the rate benefit y = R(C/20) − R(C/3) and obtained
r = +0.878 (charge) and +0.901 (energy). This is **not admissible**: since
y = x − (1 − R(C/20)), x enters y with coefficient +1, so the two axes share the same
C/3 measurement with the same sign. A within-cell permutation null (rate labels shuffled
inside each cell — destroys any real rate effect, preserves coupling and cell-to-cell
variance) returns median r ≈ +0.16 / +0.24 with a 95 % interval reaching +0.83 / +0.85.
The observed value clears that only marginally at n = 8.

**Independent moderator — the decisive test.** Repeating the moderator analysis with cold
pulse resistance *R*₀(5 °C) from the HCGT test, which shares no measurement with the CDT
retentions:

| predictor → outcome | Pearson r | p | Spearman ρ | p |
|---|---|---|---|---|
| cold *R*₀ → charge benefit | −0.598 | 0.118 | −0.619 | 0.102 |
| cold *R*₀ → energy benefit | −0.659 | 0.076 | −0.690 | 0.058 |

The mechanism predicts a **positive** sign (more polarisation to relieve → more benefit
from lowering rate). Observed is **negative** in both, non-significant at n = 8.

## 4. Why it is excluded from the manuscript

1. The charge-window interaction — the direct analogue of the 58 % truncation term — does
   **not** reach significance (p = 0.09). Only the energy interaction does (p = 0.012).
   Reporting the energy result alone would be selective.
2. The independent moderator trends against the mechanism.
3. The effect is small because the cell is wrong for the question: the P42A is a
   high-power cell losing only ~2.9 % charge / 3.3 % energy at C/3 and 5 °C, against the
   SIB's 18.4 % / 25.1 % at 1C. There is almost no truncation to relieve.
4. Including the full analysis would add substantial complexity without materially
   strengthening the central claim.

**No inference about a preferred test rate for the SIB should be drawn from the P42A's
apparent saturation by C/10.** Different cell, chemistry and polarisation regime. The
defensible design recommendation is only to test several lower rates.

## 5. Effect on the manuscript

One change was retained, and it is a narrowing rather than a strengthening. The Discussion
and Conclusions previously implied that matched low-rate cold discharge data do not exist;
they now say the decisive follow-up is needed **for this sodium-ion cell**, and note that
such data exist for other commercial chemistries (Khan et al. 2025) but not for the cell
analysed here. That is more defensible precisely because the external test was run and did
not deliver confirmation.

## 6. Where the material lives

| Item | Path |
|---|---|
| Raw data (CDT 5/25 °C, 20 cells; HCGT 5 °C) | `data_external/stanford_p42a/` (74 MB + 16 MB) |
| Primary rate-dependence analysis | `scripts/check_stanford_rate_dependence.py` |
| Independent moderator analysis | `scripts/check_stanford_moderator.py` |
| Per-cell results | `results/tables/table_stanford_rate_dependence.csv` |
| Moderator results | `results/tables/table_stanford_moderator.csv` |
| Figure (not used in the paper) | `manuscript/figures/figure_stanford_rate_dependence.pdf` / `.png` |

Both scripts require the project venv (`.venv/bin/python`); the system Python's scipy is
built against numpy 1.x and cannot load these `.mat` files.
