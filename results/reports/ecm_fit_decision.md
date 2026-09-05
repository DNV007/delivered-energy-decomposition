# Equivalent-Circuit Fit Decision

Decision: do not add fitted equivalent-circuit parameters to the core analysis at this stage.

## Criteria

- Fit all EIS spectra with four candidate models: Randles-CPE, Randles-CPE plus semi-infinite Warburg,
  a two-arc CPE circuit, and a two-arc CPE circuit plus Warburg.
- Require scaled residual NRMSE <= 0.05.
- Reject fits that land near parameter bounds.
- Reject fits with unstable multi-start solutions, using elite-parameter CV > 0.20.

## Outcome

- randles_cpe: 0/40 spectra pass all criteria.
- randles_cpe_warburg: 0/40 spectra pass all criteria.
- two_arc_cpe: 0/40 spectra pass all criteria.
- two_arc_cpe_warburg: 0/40 spectra pass all criteria.

- Best median residual group: randles_cpe_warburg on SIB with median NRMSE 0.070.
- LTO and NMC spectra show repeated boundary hits in the candidate circuit models.
- Adding a Warburg term improves some residuals but does not produce reliable pass rates.

## Recommendation

- Keep robust EIS spectral descriptors in the manuscript: high-frequency series proxy, low-frequency Zre, arc width, and characteristic frequency.
- Do not interpret Rct, CPE Q/alpha, or Warburg sigma as physical fitted parameters in the main results.
- Equivalent-circuit fits may be kept only as a supplementary QC artifact if needed, with explicit failure/pass criteria.
