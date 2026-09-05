# Internal record — recoverability reconstruction tested and rejected

**Date:** 2 September 2026
**Status:** run as an internal stress test, failed calibration, not used in the manuscript.
**Purpose:** if a referee asks why the paper names a low-rate cold discharge as the
decisive experiment and declines to estimate its outcome, this is the answer.

## The proposal

Reviewers suggested bounding the recoverable charge without new measurement, using

    V_load(q, I) = V_rest(q) − I · R(q, T)

solved for the charge at which the loaded voltage meets the 1.49 V cutoff, evaluated
at currents below the protocol's 1.5 A. Both inputs exist in the archive: the
rested-voltage curve from the incremental discharge, and a SOC-resolved pulse
resistance at 5 °C.

## The test

Calibration against a known answer. At *I* = 1.5 A the reconstruction must reproduce
the measured loaded curve and the measured endpoint *Q*(5 °C) = 1.2840 Ah. Twelve
variants were run: three pulse timescales (first post-step sample, 1 s, 10 s) × two
charge-axis normalisations (*Q*(5 °C), *Q*(25 °C)) × two interpolations (linear,
monotone cubic). Script: `scripts/check_recoverability_reconstruction.py`;
output: `results/tables/table_recoverability_stress_test.csv`.

## Result — it fails, three ways

| Check | Outcome |
|---|---|
| Variants reproducing the measured 1.5 A endpoint | **0 of 12** |
| Endpoint error | **−172 to +188 mV** |
| Spread of predicted end voltage across variants | **360 mV** |
| Margin it would need to resolve (last segment mean → cutoff) | **267 mV** |
| Best RMS error over the whole curve (1 s, *Q*(5 °C)) | 36 mV |

1. **No variant reaches the cutoff at the current where the answer is known.** The
   reconstruction predicts the cell should have kept discharging past where it
   actually stopped, so it cannot locate a crossing at any lower current either.
2. **The choice of pulse timescale moves the endpoint by more than the effect.**
   360 mV of disagreement between defensible choices, against a 267 mV margin. No
   timescale is privileged: a 1 s secant is not the resistance governing a sustained
   36 s segment, still less a sustained low-rate discharge.
3. **The predicted quantity is not the one the cutoff acts on.** The cutoff is an
   instantaneous within-segment criterion; *V*_rest − *IR* yields a quasi-steady value
   comparable to a segment mean. The measured minimum segment *mean* is 1.574 V and
   never reaches 1.49 V.

A fourth error compounds these: protocol SOC is not coulomb-counted SOC (maximum
deviation 19.48 pp, main text §2.6), so placing *R* on the charge axis carries its own
error before any of the above.

## Decision

Not promoted. The pre-agreed criterion was that several defensible choices of *R*,
timescale and interpolation must give essentially the same bound; they do not agree
even in sign of the endpoint error. An unvalidatable bound is worse than none, and
there is no low-rate ground truth in this archive against which to calibrate one.

One sentence was added to the manuscript Discussion giving this reason, so that
declining to estimate reads as a judgement rather than an omission.

## What would make it work

A single low-rate cold discharge on the same cell — the experiment the paper already
names. With that as an anchor the construction could be calibrated rather than
assumed, and the timescale ambiguity resolved empirically.
