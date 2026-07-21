# Foundation Models for EEG Are Blind to Long-Range Temporal Correlations

Research code accompanying the manuscript:

> **Foundation Models for EEG Are Blind to Long-Range Temporal Correlations: A Candidate Mechanism for Their Cross-Population Fragility**
> (submitted to the *Journal of Neural Engineering*)

## Summary

Pretrained EEG foundation models (FMs) generalise poorly across recording sites and clinical
populations. This repository contains the analysis code behind a candidate mechanistic explanation:
frozen FM embeddings do not encode the **scale-free temporal dynamics** of the signal — specifically
the alpha-envelope long-range temporal correlation (LRTC) exponent measured by detrended fluctuation
analysis (DFA) — even though a simple classical feature set recovers it. The code (i) probes whether
five frozen encoders (REVE, LaBraM, CBraMod, BIOT, BENDR) linearly or non-linearly encode the
per-subject DFA/LRTC exponent versus the static 1/f aperiodic slope; (ii) tests whether the
DFA exponent transfers disease-vs-control classification across cohorts where the FMs collapse;
(iii) shows that recording-site identity dominates the frozen embedding and is not removed by
harmonisation (ComBat), so cross-population transfer is not rescued; and (iv) asks, as a
mechanistic control, whether REVE *computes* LRTC in its pre-pool token sequence but *discards* it
at mean-pooling. Baselines, permutation nulls, matched Gaussian nulls, split-half reliability
ceilings, and leakage-safe harmonisation controls are included so the negative ("not encoded")
results can be scrutinised.

**This is a code-only release.** No EEG recordings, no derived embeddings, no cached
intermediate results, no figures, and no model weights are distributed. Every input must be
obtained from its original custodian and every intermediate regenerated locally (see below).

## Data availability

**No EEG data, no derived embeddings, and no result files are included in this repository.**
All datasets are third-party and must be obtained directly from their custodians under each
custodian's own access terms. Pretrained model weights are downloaded at runtime from their
original hosts (Hugging Face) by the wrappers in `src/` and are likewise not redistributed here.

| Cohort | Population / domain | Used for | How to obtain |
| --- | --- | --- | --- |
| **ds004504** | Greek — Alzheimer's / FTD / healthy | Encoding probe, cross-population transfer, site decoding | Open access on **OpenNeuro** (accession `ds004504`). |
| **CAUEEG** | Korean — dementia / MCI / normal | Encoding probe, DFA reliability, transfer, site decoding | Restricted; requires a **signed data-use agreement** with the dataset custodians. |
| **TDBRAIN** | Dutch — psychiatric cohort | Multi-FM site decoding / embedding extraction | **Synapse** registration and access approval. Note: TDBRAIN is in REVE's own pretraining corpus and is therefore excluded from all REVE-embedding claims in the paper. |
| **BrainLat** | Latin-American — Alzheimer's + bvFTD / healthy | 3-cohort transfer, encoding probe | **Synapse** registration and access approval. |
| **ASZED-153** | Nigerian — schizophrenia | Auxiliary (short recordings; excluded from the main site/transfer arms) | **OpenNeuro / Zenodo**. |

Follow each provider's licence and citation requirements. Nothing in this repository grants access
to, or redistributes, any of these datasets.

## Installation

Tested with Python 3.12 on macOS (Apple Silicon, PyTorch `mps`; CPU fallback works everywhere).

Using conda:

```bash
conda create -n neurogenis python=3.12
conda activate neurogenis
pip install -r requirements.txt
```

Using pip / venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `braindecode`, `transformers`, and `huggingface_hub` download pretrained weights
  (REVE, LaBraM, CBraMod, BIOT, BENDR) on first use; a network connection and sufficient disk
  space are required the first time each model is loaded.
- `neuroCombat` provides the harmonisation (ComBat) arm; `specparam` provides the 1/f aperiodic
  fit. If either is missing, the affected arm degrades gracefully or is skipped with a note in the
  output JSON.

## Repository layout

```
poc/
├── src/                                  # Reusable library code
│   ├── embeddings.py                     # LaBraM loading/adaptation + shared capture-hook utility
│   ├── reve_wrapper.py                   # REVE-Base loader + frozen embedding extraction
│   ├── cbramod_wrapper.py                # CBraMod model definition + loader + extraction
│   ├── biot_wrapper.py                   # BIOT loader + extraction (braindecode)
│   ├── bendr_wrapper.py                  # BENDR loader + extraction (braindecode)
│   ├── baselines.py                      # Classical features (band power, Hjorth, spectral entropy)
│   ├── io_safety.py                      # Path-whitelisted, SHA-256-audited pickle/npz loaders
│   └── runs.py                           # Optional run-manifest mirroring (no-op unless POC_RUN_DIR set)
├── scripts/analysis/                     # The analysis scripts for this paper (see "Script map")
├── data/                                 # NOT SHIPPED — you provide raw EEG + generate embeddings
│   ├── raw/                              #   raw cohort files (ds004504, caueeg-dataset, tdbrain, ...)
│   ├── processed/                        #   preprocessed epoch caches (e.g. alzheimer_processed.pkl)
│   └── embeddings/                       #   cached FM/classical subject-mean embeddings (*.npz)
└── results/cross_population/             # NOT SHIPPED — regenerated by the analysis scripts
```

Only `src/` and `scripts/analysis/` contain code. Per-script configuration (device, model IDs,
scale ranges, thresholds) is defined inline at the top of each script — there is no separate config
file to edit. `data/` and `results/` are git-ignored and must be populated locally.

## How to run

All scripts are written to be executed **from the repository root** (imports use
`from src.module import ...` and `import scripts.analysis... `):

```bash
cd poc
python scripts/analysis/<script>.py
```

Before any analysis script will run you must stage two things that are **not** included:

1. **Raw EEG under `data/`** — e.g. `data/raw/ds004504/...` (`.set`), `data/raw/caueeg-dataset/...`
   (`.edf` + `dementia.json`), `data/raw/tdbrain/...` (`.bdf`), plus preprocessed caches such as
   `data/processed/alzheimer_processed.pkl`. Paths and montage handling are defined at the top of
   each script.
2. **Cached embeddings under `data/embeddings/`** — the transfer/site/encoding scripts read
   frozen subject-mean embeddings as `*.npz` files (e.g. `caueeg_3way_reve_19ch_subjects.npz`,
   `ds004504_3way_cbramod_19ch_subjects.npz`). Generate these first with
   `scripts/analysis/extract_ds004504_multifm_embeddings.py` and
   `scripts/analysis/extract_tdbrain_multifm_embeddings.py` (and the upstream REVE/classical
   extraction used by the project's evaluation pipeline). The extractors download pretrained
   weights and run inference on your local copy of the raw data.

Typical order of operations: obtain raw data → extract embeddings into `data/embeddings/` →
run the encoding / transfer / site-decoding scripts (which cache target and feature matrices into
`results/cross_population/`) → render figures with `plot_lrtc_paper_figures.py` and
`fig_predicted_vs_true_dfa.py`. Most scripts print their output path and accept `--out` /
`--n` / `--perms` overrides; run with `-h` for details.

## Synthetic validation of the proposed loss (Section 5)

`scripts/analysis/lrtc_synthetic_validation.py` is a **self-contained** harness for the
LRTC-aware auxiliary loss proposed in Section 5 — it needs no EEG data. It generates fractional
Gaussian noise (fGn) of known Hurst exponent, trains a small masked-reconstruction transformer
with and without the auxiliary loss, and probes whether the frozen embedding recovers the
exponent. Requires `numpy`, `torch`, `scikit-learn`.

```bash
python scripts/analysis/lrtc_synthetic_validation.py --compare-all
```

Four modes: `baseline`, `lrtc_correct`, `lrtc_linearity_only`, and `lrtc_fixed_target` (an
explicit **negative control** implementing the fixed-target loss the paper argues against).

**Honest status — this run does _not_ validate the loss.** On pure fGn the Hurst exponent is the
entire signal, so masked reconstruction already recovers it near-ceiling (baseline R² ≈ 0.93);
with no deficit to rescue, every auxiliary variant lands slightly _below_ baseline and the
negative control is not worse than the correct variant:

| mode | frozen-embedding probe R² |
|------|---------------------------|
| baseline            | 0.935 |
| lrtc_correct        | 0.892 |
| lrtc_linearity_only | 0.826 |
| lrtc_fixed_target   | 0.909 |

This is a property of the _test_, not of the loss: pure fGn removes the very confound (LRTC
competing with dominant spectral/oscillatory structure) that makes real EEG foundation models
blind to LRTC. A faithful test needs a synthetic input in which the dominant power is spectral
and only a weak component carries the target scaling exponent, so that the baseline is genuinely
blind and the auxiliary loss has something to rescue. That experiment, and a single-model
head-retrained pilot on a pretrained encoder, are left to future work. The fGn generator and DFA
estimator themselves are validated (`validate_generator()`: r(H, α_DFA) ≈ 0.95, unbiased) — run
them first before trusting any downstream number.

## Citation

If you use this code, please cite the paper (full reference to be finalised on acceptance):

```bibtex
@article{zare_lrtc_eeg_fm_2026,
  title   = {Foundation Models for EEG Are Blind to Long-Range Temporal Correlations:
             A Candidate Mechanism for Their Cross-Population Fragility},
  author  = {Zare, Marzieh},
  journal = {Journal of Neural Engineering},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the foundation-model papers and libraries listed in `NOTICE` when you use the
corresponding wrappers.

## License and attribution

This repository's original analysis and wrapper code is licensed under the **Apache License 2.0**
(see [`LICENSE`](LICENSE)).

Third-party components carry their own licenses, reproduced in [`NOTICE`](NOTICE). In particular,
the CBraMod model architecture in `src/cbramod_wrapper.py` is adapted from the upstream
[CBraMod](https://github.com/wjq-learning/CBraMod) implementation (MIT License, © 2025 Jiquan Wang).
Pretrained model weights are **not** redistributed here; they are downloaded at runtime from their
original hosts under their own terms. No EEG dataset is covered by this repository's license — each
cohort remains governed by its own custodian's access terms.
