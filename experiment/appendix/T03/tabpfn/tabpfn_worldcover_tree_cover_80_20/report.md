# WorldCover tree-cover augmentation and evaluation

Prepared and evaluated on 30 October 2026 (Asia/Singapore). This is a separate
extension of the frozen 77-feature TabPFN local-economy/landscape baseline.
The combined 80-feature model completes the full chronological holdout but
slightly worsens RMSE and MAE. Retain the 77-feature baseline as the preferred
model on this development holdout. The 80-column datasets remain available
for further experiments.

## Full combined-feature result

Both models use all 121,005 January 2021–May 2024 context rows and the same
28,995 June 2024–March 2025 query rows. The combined model adds the two tree
fractions and their availability flag, using native TabPFN preprocessing.

| Model | Input features | RMSE (SGD) | MAE (SGD) | Bias (SGD) | Delta RMSE (SGD) |
| --- | --- | --- | --- | --- | --- |
| Frozen baseline | 77 | 455.813570 | 329.748476 | 0.767535 | 0.000000 |
| Both tree radii + unavailable flag | 80 | 456.282720 | 330.578040 | -0.908893 | +0.469150 |

The combined addition does not improve this holdout: RMSE increases by
S$0.469150 and MAE by S$0.829564. No uncertainty interval or significance claim
is attached to this small difference. This remains a previously examined
development holdout, not a new untouched test.

The full run fits in 14.99 seconds and predicts in 663.47 seconds, with
7.191 GiB peak PyTorch-allocated VRAM, no context subsampling and unchanged
model parameter versions. Complete predictions have identical row IDs and
actual rents to the frozen baseline. Separate 100 m / 300 m ablations and
tree-feature screening folds have not been run.

Evidence: [scores.csv](scores.csv),
[full-run timing](timing_outer_tree_100m_300m.json), and
[full-run input manifest](job_outer_tree_100m_300m.json).

## Data and definitions

Two public Singapore tiles, `N00E102`, were downloaded anonymously from ESA's
public AWS bucket: 2020 v100 (42,791,583 bytes) and 2021 v200 (36,505,930 bytes).
No Earth Engine account, Google Cloud project, API key or paid service is needed.
Original rasters and extracted CSVs are in `.cache/tabpfn_worldcover_tree_cover_80_20/`.
Source URLs, release dates, retrieval timestamps, attribution and SHA-256 hashes
are recorded in [source_manifest.json](source_manifest.json).

The raster window around the canonical buildings is reprojected to EPSG:3414
at 10 m with nearest-neighbour resampling. Features use pixel centres inside
100 m and 300 m metric circles; this is a raster approximation to circular area.

- `TREE_CLASS_FRACTION_100M`: class-10 pixels / valid pixels in the 100 m buffer.
- `TREE_CLASS_FRACTION_300M`: the equivalent fraction in the 300 m buffer.
- `TREE_COVER_UNAVAILABLE`: 1 when the tree fractions cannot be supplied.

Class 10 is WorldCover's tree-cover class. Mangroves (class 95) are deliberately
excluded from this primary definition. Fractions estimate area assigned to the
tree class, not precise canopy closure or individual street-tree coverage.
Water and other valid classes remain in the denominator. Nodata is excluded;
less than 95% valid buffer support makes the feature missing, never zero.
Coverage fractions are retained in the extraction audit but are not additional
model inputs.

All 9,464 canonical buildings have 100% valid mapped support at both radii in
both products. Geographic coverage does not establish classification accuracy.
The two products use different algorithm versions; differences between years
must not be interpreted as measured tree loss or growth.

## Historical availability

Each training row retains the baseline's original pseudo-origin; each query
fold retains its fixed forecast origin. Choose the latest published product,
using conservative next-month cutoffs:

| Product | Public release | First eligible forecast origin |
| --- | --- | --- |
| 2020 v100 | 20 October 2021 | November 2021 |
| 2021 v200 | 28 October 2022 | November 2022 |

Before November 2021 the fractions remain missing and the unavailable flag is
1. The strict outer context has 43,218 rows using 2020, 40,483 using 2021, and
37,304 unavailable rows. All 28,995 June 2024–March 2025 holdout rows use 2021.
The earlier September 2022 screening origin uses 2020; July 2023 uses 2021.
No rental targets enter extraction. No unlabelled test predictions are generated.

## Experiment interface

### Training-data integration and preprocessing

Ready-to-fit bundles now explicitly contain the original 77 columns followed by
`TREE_CLASS_FRACTION_100M`, `TREE_CLASS_FRACTION_300M`, and
`TREE_COVER_UNAVAILABLE`. Build or verify them with:

```powershell
.venv/Scripts/python.exe analysis/prepare_tabpfn_worldcover_training.py
```

The saved bundle paths are:

- `.cache/tabpfn_worldcover_tree_cover_80_20/training_outer_80.joblib`:
  121,005 training rows and 28,995 query rows, each with 80 features.
- `.cache/tabpfn_worldcover_tree_cover_80_20/training_inner_2022-09_80.joblib`:
  60,079 training rows and 600 query rows, each with 80 features.
- `.cache/tabpfn_worldcover_tree_cover_80_20/training_inner_2023-07_80.joblib`:
  87,937 training rows and 600 query rows, each with 80 features.

The tree fractions are normalized by valid buffer area at extraction, so their
units are already dimensionless fractions in [0, 1]. The unavailable flag is
binary 0/1. No external z-score, min-max, log or quantile transform is applied.
This follows [TabPFN's guidance to avoid external feature scaling](https://github.com/PriorLabs/TabPFN#usage-tips).
The installed TabPFN-3.5 implementation captures missingness, imputes using the
training context, fits its internal standard scaler on the training context,
and reuses cached training statistics for queries. The two tree columns pass
through this same native preprocessing as the original numeric inputs.

There is no preprocessing fit on held-out rows and no zero-filling of unavailable
historical tree cover. Validation rejects infinities, fractions outside [0, 1],
inconsistent missingness flags, duplicate IDs and row-order mismatches. The nine
original categorical dtypes, all 77 original columns, targets, residual baselines
and temporal metadata survive serialization unchanged. Saved matrices are
checked cell-for-cell against the matrices passed to the frozen experiment
runner.

To load the outer bundle:

```python
import joblib
b = joblib.load('.cache/tabpfn_worldcover_tree_cover_80_20/training_outer_80.joblib')
X_train, X_query = b['X_train'], b['X_query']
y_residual = b['y_train'] - b['train_baseline']
# The existing frozen runner fits y_residual, then restores query_baseline.
```

Audit files: [training_dataset_manifest.json](training_dataset_manifest.json),
[training_dataset_settings.json](training_dataset_settings.json), and
[training_feature_statistics.csv](training_feature_statistics.csv). Training
means and standard deviations in that CSV are descriptive audits; they are not
an externally fitted scaler. The settings record the actual checkpoint's
default inference configuration; the experiment runner retains its explicit
overrides (one estimator, full context, seed 42).

All original 77 columns, residual targets, context rows, checkpoint, categorical
encodings and frozen inference settings are retained. The predeclared ablations
are independent additions to that baseline:

| Variant | Added features | Total |
| --- | --- | --- |
| `tree_100m` | 100 m fraction + unavailable flag | 79 |
| `tree_300m` | 300 m fraction + unavailable flag | 79 |
| `tree_100m_300m` | Both fractions + unavailable flag | 80 |

Two earlier screening folds retain the existing full contexts and identical
600-query samples. Outer experiments retain all 121,005 context rows and all
28,995 holdout queries. There is no parameter optimization or context sampling.
This remains the previously examined development holdout.

From the repository root, in PowerShell:

```powershell
# Dependencies and reproducible data preparation (already completed here).
.venv/Scripts/python.exe -m pip install -r requirements-worldcover.txt
.venv/Scripts/python.exe analysis/prepare_worldcover_tree_cover.py --stage all
.venv/Scripts/python.exe analysis/compare_worldcover_tree_cover.py --stage prepare

# Baseline replay and combined-feature inference, each on 200 queries.
.venv/Scripts/python.exe analysis/compare_worldcover_tree_cover.py --stage smoke

# Run both earlier screening folds, then all three full-holdout ablations.
.venv/Scripts/python.exe analysis/compare_worldcover_tree_cover.py --stage all
.venv/Scripts/python.exe analysis/summarize_worldcover_tree_cover.py
```

Use `--stage screen` or `--stage outer` to run those stages separately.
For one experiment, use `--stage one --fold outer --variant tree_100m`.
Inference saves resumable 200-query batches; separate child processes isolate
full experimental runs. Input/code hashes prevent reusing incompatible results.
Smoke predictions are excluded from the score summary and are not performance
estimates. The summary compares every completed variant with matching frozen
baseline predictions; it does not select a winner from outer labels.

## Setup verification

Completed successfully on the RTX 4080:

- 13 unit tests across the new tree-cover and existing local spatial suites.
  These check metric buffers, nodata, mangrove exclusion, publication cutoffs,
  missing locations and the inherited feature/history policies.
- Cached source and extraction reuse passes checksum validation without network
  access. Feature preparation is also resumable and hash-checked.
- A 200-query baseline replay with all 121,005 context rows matches the frozen
  prediction artifact to a maximum absolute difference of S$0.00000000495.
- A 200-query combined tree-feature smoke run succeeds with 80 features,
  unchanged model parameters, no context subsampling and 7.190 GiB peak
  PyTorch-allocated VRAM. Fit took 14.79 seconds; prediction took 4.80 seconds.
- Six targeted tree-cover/integration tests pass after adding explicit
  training bundles, including category preservation, row-order validation,
  numeric range checks and missingness consistency.
- The combined full-holdout run is complete, using all 121,005 context rows
  and all 28,995 queries. The score summarizer runs and excludes smoke
  predictions. Only the combined full experiment has been evaluated so far.

Machine-readable evidence: [baseline replay](baseline_replay_outer.json),
[combined smoke timing](timing_outer_tree_100m_300m_smoke.json), and
[smoke job inputs](job_outer_tree_100m_300m_smoke.json).

## Sources

- [ESA WorldCover downloads, product versions and CC BY 4.0 attribution](https://esa-worldcover.org/en/data-access)
- [ESA's 2020 and 2021 publication dates](https://eo4society.esa.int/2022/11/11/2021-worldcover-product/)
- [WorldCover classification codes](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200)

See also [extraction_manifest.json](extraction_manifest.json),
[feature_manifest.json](feature_manifest.json), [feature_coverage.csv](feature_coverage.csv),
and [source_year_coverage.csv](source_year_coverage.csv).
