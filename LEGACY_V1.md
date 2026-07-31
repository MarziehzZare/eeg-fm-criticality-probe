# Legacy v1 and exploratory code

The files marked `LEGACY V1 / EXPLORATORY SCRIPT` are retained on `main`
for auditability. They reproduce historical targets, probe searches, transfer
analyses, or proposed-loss experiments and do not produce the revised Paper-02
claims.

The complete repository state before the v2 code update is preserved by:

- git tag `paper02-v1-code`
- commit `a8d6622`

Do not combine outputs from these scripts with the v2 artifacts unless a
documented compatibility audit establishes identical subjects, target
estimands, preprocessing, folds, and readouts. In particular:

- legacy 0.5–30 s DFA targets are not interchangeable with the calibrated v2
  2 s to ≤10%-duration target;
- best-of-probe searches are not interchangeable with the fixed nested v2
  readout;
- legacy transfer/site analyses do not identify a causal population or site
  effect;
- the synthetic auxiliary-loss harness does not validate the proposed loss;
- historical chronological/shuffled comparisons are superseded by the
  matched-realization implementation.

Use the canonical script map in [README.md](README.md) for v2 reproduction.
