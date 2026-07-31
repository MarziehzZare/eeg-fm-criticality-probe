# Cross-Cohort Spectral–Temporal Dissociation in Frozen EEG Foundation-Model Representations

Reproducibility code for the revised Paper 02 manuscript:

> **Cross-Cohort Spectral–Temporal Dissociation in Frozen EEG Foundation-Model Representations**

The existing [arXiv record](https://arxiv.org/abs/2607.24834) may continue to
show v1 until the replacement is processed. This repository already reflects
the v2 analysis and interpretation.

## Revision status

Version 2 is a substantive correction, not a cosmetic update.

The v1 universal claim that EEG foundation models are “blind” to long-range
temporal correlations is withdrawn. The proposed causal explanation of
cross-population fragility is also withdrawn. The revised evidence supports a
narrower result:

- CBraMod and BIOT decode the fixed-mode aperiodic exponent in both evaluated
  cohorts.
- Recovery of the revised alpha-envelope DFA target is model- and
  cohort-dependent.
- A post-hoc ordered-versus-shuffled sequence probe is configuration-sensitive.
  Its tail proportions are descriptive because chronological EEG epochs are
  not exchangeable.
- Order sensitivity does not isolate LRTC; non-stationarity, arousal drift,
  local continuity, artifact clustering, and recording-position effects remain
  alternatives.
- Cross-cohort transfer and causal mechanisms remain exploratory.

See [V2_SUPERSESSION.md](V2_SUPERSESSION.md) for the complete correction record
and [LEGACY_V1.md](LEGACY_V1.md) before using any historical script.
The v1 code is preserved in git history and tagged `paper02-v1-code`.

## Critical statistical correction

An earlier exploratory implementation generated a different shuffled dataset
for each configuration:

```text
shuffle_seed = 1000 + 97*k + c
```

That changed the cross-configuration covariance of the reference statistic.
Its tail values are invalid for the intended matched statistic and must not be
quoted. The corrected implementation uses one subject-wise shuffled
realization for every configuration in a Monte Carlo draw:

```text
shuffle_seed = 1000 + k
```

The canonical output reports `tau_MC` as a descriptive shuffle-reference tail
proportion. It is not treated as an LRTC-specific p-value, and no Holm or other
family-wise rejection procedure is applied to it.

## What is distributed

This is a code-only release. It contains:

- the calibrated alpha-envelope DFA estimator and QC utilities;
- revised target extraction, encoding, aperiodic-orthogonality, demographic,
  site-decoding, and transfer scripts;
- corrected matched-realization order-control code;
- the canonical evidence consolidator and figure generator; and
- unit and regression tests that require no restricted EEG data.

It does **not** contain EEG recordings, derived embeddings, result JSON/NPZ
files, figures, or model weights. Generated `data/` and `results/` trees are
ignored by git.

## Data access

| Cohort | Role in v2 | Access |
| --- | --- | --- |
| CAUEEG | DFA target, encoding, aperiodic comparison, order control | Restricted; obtain from the dataset custodians under their data-use agreement |
| BrainLat | DFA target, encoding, aperiodic comparison, transfer | Synapse registration and approval |
| ds004504 | Context and legacy analyses; not part of the two-cohort v2 DFA encoding result | OpenNeuro accession `ds004504` |
| TDBRAIN | Legacy/context analyses only | Synapse registration and approval |
| ASZED-153 | Excluded from confirmatory cross-population inference | Obtain from its original custodian |

Nothing in this repository grants access to, or redistributes, third-party
data.

## Installation

The producing environment used Python 3.12. A clean environment can be created
with:

```bash
conda create -n neurogenis python=3.12
conda activate neurogenis
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The wrappers download public pretrained weights from their original hosts at
runtime. Review each upstream model card and license before use.

Run the data-independent checks first:

```bash
pytest -q
python -m compileall -q src scripts
```

## Canonical v2 script map

Only the files in this table produce or package the revised claims. Other
analysis files remain in history for auditability and must not be used to
recreate v2 conclusions.

| Stage | Script | Purpose |
| --- | --- | --- |
| Estimator validation | `src/lrtc.py`, `paper02_filter_calibration.py` | FIR/Hilbert/DFA implementation and simulation calibration |
| Target extraction | `paper02_extract_dfa_v2.py` | Fixed 240-s target, artifact mask, per-channel QC, explicit eligibility |
| Primary encoding | `paper02_primary_encoding_v2.py` | Fixed nested out-of-fold scaler → PCA → ridge readout |
| Aperiodic control | `paper02_aperiodic_orthogonality_v2.py` | Fixed-mode aperiodic comparison and residualized sensitivity |
| Confound audit | `paper02_demographic_audit.py` | Missingness and descriptive adjusted associations |
| Site/transfer context | `paper02_dfa_site_v2.py`, `paper02_dfa_transfer_v2.py` | Exploratory dataset/site and two-cohort transfer analyses |
| Sequence model | `paper02_sequence_probe.py` | Shared seeded 1D-CNN and ridge readouts |
| Order reference | `paper02_prepool_order_permutation.py` | Corrected matched-realization chronological/shuffled comparison |
| Packaging | `paper02_consolidate_prepool_results.py` | Five-model artifact, hashes, invariants, descriptive ledger |
| Figures | `plot_paper02_v2.py` | Generate figures from revised artifacts |

The cache-builders `reve_prepool_token_probe.py` and
`multifm_prepool_probe.py` must be invoked with `--extract-only` for v2.
Their historical exploratory readout paths are retained solely to preserve the
original cache construction and are not the canonical analysis.

## Reproduction order

All commands run from the repository root.

### 1. Validate the estimator

```bash
pytest -q tests/test_lrtc.py
python scripts/analysis/paper02_filter_calibration.py --n-simulations 1000
```

### 2. Extract the fixed v2 targets

```bash
python scripts/analysis/paper02_extract_dfa_v2.py --cohort caueeg
python scripts/analysis/paper02_extract_dfa_v2.py --cohort brainlat
```

Automatic QC is reproducible but does not replace blinded visual review of
rejected recordings and a sample of retained recordings.

### 3. Run the fixed primary and diagnostic analyses

```bash
python scripts/analysis/paper02_primary_encoding_v2.py --cohort caueeg
python scripts/analysis/paper02_primary_encoding_v2.py --cohort brainlat
python scripts/analysis/paper02_aperiodic_orthogonality_v2.py
python scripts/analysis/paper02_demographic_audit.py --cohort caueeg
python scripts/analysis/paper02_demographic_audit.py --cohort brainlat
python scripts/analysis/paper02_dfa_site_v2.py
python scripts/analysis/paper02_dfa_transfer_v2.py
```

The bootstrap intervals condition on fixed out-of-fold predictions; they are
not full-pipeline generalization intervals.

### 4. Build pre-pool sequence caches

```bash
python scripts/analysis/reve_prepool_token_probe.py --n-per-class 150 --extract-only
python scripts/analysis/multifm_prepool_probe.py --extract-only
```

### 5. Run the corrected matched-realization reference

The released evidence used four configurations and 150 shuffle realizations for
the primary cells:

```bash
python scripts/analysis/paper02_prepool_order_permutation.py \
  --n-perm 150 --config-seeds 0 1 2 3 \
  --models CBraMod BIOT --sensitivity-models CBraMod BIOT \
  --out results/paper02/prepool_order_permutation_matched.json

python scripts/analysis/paper02_prepool_order_permutation.py \
  --n-perm 150 --config-seeds 0 1 2 3 \
  --models REVE LaBraM BENDR --sensitivity-models \
  --out results/paper02/prepool_order_permutation_matched_rest.json
```

The configuration-stability extension used eight additional configurations
and the first 30 matched shuffles:

```bash
python scripts/analysis/paper02_prepool_order_permutation.py \
  --n-perm 30 --config-seeds 4 5 6 7 8 9 10 11 \
  --models CBraMod BIOT --sensitivity-models CBraMod BIOT \
  --out results/paper02/prepool_order_permutation_extension_cfg4_11_K30.json
```

Consolidate only after all source artifacts are present:

```bash
python scripts/analysis/paper02_consolidate_prepool_results.py \
  --primary results/paper02/prepool_order_permutation_matched.json \
  --rest results/paper02/prepool_order_permutation_matched_rest.json \
  --config-extension results/paper02/prepool_order_permutation_extension_cfg4_11_K30.json \
  --out results/paper02/prepool_order_permutation_5model.json
```

The consolidator validates model membership, run dimensions, matched-shuffle
metadata, configuration identities, and matrix shapes before combining inputs.
It cannot reconstruct subject identity from historical artifacts because those
artifacts did not store subject IDs; this limitation is recorded in the output.

### 6. Render figures

```bash
python scripts/analysis/plot_paper02_v2.py
```

## Interpretation guardrails

- Do not call a large tail fraction evidence of no order information.
- Do not call a small tail fraction an LRTC-specific detection.
- Do not choose a batch size because it yields the preferred result.
- Do not describe BIOT as simply positive or null; its order increment is
  setting-dependent.
- Treat LaBraM's below-mean-predictor arms as uninformative, not negative.
- Do not compare the pooled ridge and sequence CNN as a percentage of
  “recoverable signal”; they are different readout architectures.
- Do not describe the post-hoc tested-cell ledger as complete multiplicity
  accounting.

## Legacy code

The repository history remains available so v1 can be audited. Scripts not
listed in the canonical v2 map implement earlier exploratory targets, probe
selection, transfer, or synthetic-loss analyses. They are not deleted, but
their outputs do not support the revised paper unless explicitly regenerated
through the v2 pipeline above.

## Citation

Until arXiv v2 is processed, cite the version explicitly:

```bibtex
@article{zare_spectral_temporal_2026,
  title   = {Cross-Cohort Spectral--Temporal Dissociation in Frozen EEG
             Foundation-Model Representations},
  author  = {Zare, Marzieh},
  journal = {arXiv preprint arXiv:2607.24834},
  year    = {2026},
  note    = {Version 2}
}
```

## License

Original repository code is licensed under Apache-2.0. Third-party model code
and weights retain their upstream licenses; see [NOTICE](NOTICE).
