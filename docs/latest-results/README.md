# Latest results

Evidence cutoff: 4 November 2026. Repository update: 6 November 2026.

Study: **Window order in decoding alpha-amplitude dynamics from frozen EEG foundation models**.
This summary follows the current local journal manuscript, including the latest
multiplicity correction. It is distinct from the historical Paper 02 v2 analysis
still documented at the repository root. The manuscript is prepared for author
review; this repository update is not a journal submission or acceptance.

## Controlled synthetic experiment

Each of 96 independent families reused the same sixty four-second windows in
five orders associated with generating H values. Each frozen encoder processed
windows independently; downstream readouts used lagged relationships.

| Predictor | Held-out R² for generating H |
| --- | ---: |
| REVE | 0.561 |
| LaBraM | 0.396 |
| CBraMod | 0.613 |
| Observable power | 0.650 |

Paired model-minus-power differences were -0.089 [-0.115, -0.019] for REVE,
-0.254 [-0.294, -0.164] for LaBraM, and -0.037 [-0.058, 0.011] for CBraMod
(two-sided 95% basic bootstrap intervals). Power outperformed REVE and LaBraM;
the CBraMod difference remained unresolved. The power comparator was specified
in the protocol but executed after the model results. Features and projections
were not matched between models and power.

These results concern controlled order-dependent accessibility under the
specified generator and downstream readout. They do not establish encoder-internal
long-context computation, physiological LRTC preservation, or a pretraining benefit.

## Exploratory real EEG

The locked bag-summary ridge cell discards window order. Scores below are rounded
R² values, not a claim that the best pipeline was selected independently of outcomes.

| Predictor | ds004504, 45 participants | CAUEEG, 11 evaluation records | CAUEEG strict mask, 6 records |
| --- | ---: | ---: | ---: |
| REVE | 0.630 | 0.738 | 0.279 |
| LaBraM | 0.356 | 0.475 | 0.564 |
| CBraMod | -0.102 | 0.681 | 0.289 |
| Band power | 0.508 | 0.684 | 0.475 |
| Mask control | -0.048 | -0.009 | -0.061 |
| Training-mean baseline | -0.035 | -0.068 | -0.089 |

Order-free summaries predicted the saved DFA target. Adding order did not give a
consistent advantage across models and populations. ds004504 scores pool fixed
out-of-fold predictions. CAUEEG uses a provider train/evaluation split; the strict
mask changes both training and evaluation samples, and its six evaluation records
are nested in the eleven. Unique individuals within CAUEEG partitions were not
verified. These are within-cohort exploratory analyses, not cross-cohort transfer.

## Latest correction: paired model-versus-power comparisons

On 3 November, the selected direct REVE contrast with an aperiodic exponent in
both predictors gave ds004504 delta R² = +0.125467, with an individual conditional
95% BCa interval [0.013681, 0.330450]. That interval alone is not the final inference.

On 4 November, shared paired bootstrap resamples and a maximum absolute-deviation
statistic produced simultaneous intervals across the three adjusted contrasts:

| Adjusted model minus band power | Delta R² | Simultaneous 95% interval | Family-wise tail proportion |
| --- | ---: | --- | ---: |
| REVE | +0.125467 | [-0.233539, 0.484474] | 0.6009 |
| LaBraM | -0.251525 | [-0.610531, 0.107482] | 0.1643 |
| CBraMod | -0.548038 | [-0.907044, -0.189031] | 0.0083 |

REVE and LaBraM comparisons remain unresolved. Only the evaluated CBraMod ridge
pipeline has an adjusted interval entirely below zero. The six-contrast family,
including raw comparisons, also leaves REVE unresolved: adjusted interval
[-0.264036, 0.514970], tail proportion 0.6960.

CAUEEG adjusted REVE-minus-power differences are +0.076667 for eleven evaluation
records and -0.013105 for the six-record sensitivity. These remain descriptive.

All of these paired intervals condition on fixed predictions, omit refitting and
fold uncertainty, and involve overlapping CV training sets. The direct contrast
was selected after component scores were inspected; correction followed the
individual interval. The local hash-bound protocols are not public preregistration.
A nonsignificant difference does not establish equivalence. This comparison also
does not test adding REVE to a predictor already containing exponent and power.

## Evidence and release scope

The public implementation is in
[`scripts/latest_results`](../../scripts/latest_results). It reproduces the
paired statistic, BCa interval, and max-statistic family-wise correction from
fixed prediction arrays. See its README for the `.npz` input schema and
commands. This is the portable analysis layer. The private extraction and
readout stages remain local because they require restricted recordings,
embeddings, or model assets.

- [familywise.json](familywise.json): unchanged aggregate November 4 output,
  including both contrast families and the exact protocol hash.
- [paired_aggregate.json](paired_aggregate.json): selected aggregate scores and
  individual intervals from November 3. Interpret with the November 4 correction.
- [provenance.json](provenance.json): SHA-256 hashes and workspace-relative source
  locations for the saved evidence used here. Source locations describe local
  provenance and are not portable reproduction commands.

No analysis was rerun for this update. No raw EEG, embeddings, participant-level
predictions, model weights, or private manifests are distributed. The historical
scripts do not reproduce this newer study. Reproducing it requires a separately
packaged newer pipeline and authorized access to its inputs; this release makes
aggregate findings inspectable rather than claiming end-to-end reproducibility.
