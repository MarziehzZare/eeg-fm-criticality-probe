# Latest analysis code

These scripts implement the November 2026 paired comparison and family-wise
correction. They operate on user-supplied aggregate prediction arrays. No EEG,
embeddings, participant identifiers, model weights, or private manifests are
included.

## Input format

Create an `.npz` file with one-dimensional arrays named `y`, `model`, and
`baseline`, all with the same length. For a paired comparison, `model` is the
model prediction and `baseline` is the band-power prediction. For the family
wise correction, use `model_reve`, `model_labram`, `model_cbramod`,
`baseline_adjusted`, and optionally the corresponding unadjusted arrays.

The arrays must be fixed out-of-fold predictions for the same records. The
scripts do not refit models, infer patient identity, or calculate uncertainty
from training splits.

## Commands

```bash
python scripts/latest_results/paired_model_comparison.py \
  --input results/latest/ds004504_predictions.npz \
  --output results/latest/ds004504_paired.json \
  --bootstrap 10000 --seed 20261103

python scripts/latest_results/familywise_correction.py \
  --input results/latest/ds004504_family.npz \
  --output results/latest/ds004504_familywise.json \
  --bootstrap 10000 --seed 20261104
```

The first command reports the conditional paired BCa interval used in the
November 3 exploratory comparison. The second command reports max-statistic
simultaneous intervals and family-wise tail proportions used in the November 4
correction. The latter is the applicable result when several model contrasts
are interpreted together.

The scripts accept only local paths. Data access and model extraction remain
the responsibility of the user and must follow the custodians' terms.
