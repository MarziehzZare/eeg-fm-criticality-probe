# Paper 02 v2 supersession record

Date: 2026-07-31

## Public record being corrected

- arXiv identifier: `2607.24834`
- v1 title: *Foundation Models for EEG Are Blind to Long-Range Temporal
  Correlations: A Spectral-Temporal Dissociation Behind Their Cross-Population
  Fragility*
- v2 title: *Cross-Cohort Spectral–Temporal Dissociation in Frozen EEG
  Foundation-Model Representations*
- public v1 code tip: `a8d6622`
- preserved tag: `paper02-v1-code`

## Claims withdrawn

Version 2 withdraws all of the following:

1. The universal statement that the five evaluated frozen EEG foundation
   models are blind to LRTC.
2. The claim that loss of LRTC is a demonstrated mechanism of
   cross-population fragility.
3. The interpretation of dataset membership as a causal site, geography,
   device, or population axis.
4. The claim that mean pooling discards approximately 60% of recoverable
   signal. That comparison crossed readout architectures.
5. All tail values from the unmatched two-configuration shuffle
   implementation.
6. Any claim that the post-hoc chronological/shuffled analysis supplies a
   calibrated LRTC-specific family of p-values.

## Replacement evidence

The revised target uses:

- 240 seconds of harmonized input;
- a linear-phase 8–13 Hz FIR filter spanning two cycles at the lower passband;
- Hilbert amplitude envelopes;
- DFA scales from 2 seconds to no more than 10% of analyzable duration;
- 50% overlapping windows;
- artifact masking, retained-sample reporting, per-channel fit quality, and
  explicit recording eligibility.

The fixed primary readout is trained entirely within each outer fold:
standardization, PCA, and ridge regression. The descriptive `R²=0.10`
reference was adopted during the revision and is not presented as a
prospectively prespecified confirmatory threshold.

The robust positive result is replicated aperiodic-exponent decoding by CBraMod
and BIOT in both cohorts. DFA recoverability is model- and cohort-dependent.

## Shuffle-reference correction

The invalid implementation used a configuration-dependent shuffle seed:

```text
1000 + 97*k + c
```

The corrected implementation uses one realization for each Monte Carlo draw
under every configuration:

```text
1000 + k
```

The corrected artifact retains the full permutation-by-configuration matrix and
optimization diagnostics. The reported `tau_MC` is a descriptive tail
proportion. Chronological EEG is not exchangeable, and preserving chronology
also preserves several non-LRTC processes. No Holm rejection, Boolean crossing,
threshold bolding, or “complete multiplicity accounting” is used.

## Model-specific order-control interpretation

- CBraMod shows a narrow, post-hoc positive order increment, conditional on
  probe training and configuration.
- BIOT is setting-dependent: its primary batch-128 analysis detects no
  advantage, while batch 32 is positive but optimization-sensitive.
- LaBraM is uninformative where both ordered and shuffled arms perform below a
  mean predictor.
- REVE and BENDR do not provide a positive order claim in this analysis.
- No order result is cross-population replicated or LRTC-specific.

## Artifact policy

No result files are distributed in this code-only repository. Generated
artifacts must carry their own metadata and hashes. Any locally retained v1
two-configuration JSON produced with the invalid seed formula must be moved
under a `superseded/` directory, named `*_INVALID.json`, and accompanied by a
manifest stating that no tail value may be quoted.
